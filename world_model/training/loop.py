"""Differentiable RGB measurement pretraining and causal closed-loop unrolls.

The functions in this module consume canonical batch-major simulator episodes,
but simulator state is used only to construct losses.  The runtime receives RGB
pixels, timestamps, and calibrated camera tensors through
``ObservationPacket``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from world_model.belief import BeliefTrajectory, MotionMode, WorldBelief
from world_model.observations import (
    MeasurementSet,
    ObservationContext,
    ObservationPacket,
    PredictedMeasurements,
    SensorContext,
)
from world_model.runtime import (
    DifferentiableIngestTrace,
    OnlineWorldModel,
    PreparedPropagation,
)
from world_model.training.event_windows import observation_window_query_plan
from world_model.training.losses import (
    balanced_binary_cross_entropy,
    correction_error,
    gaussian_nll,
    masked_huber,
    masked_mean,
    posterior_improvement_hinge,
    weighted_total,
)
from world_model.training.matching import match_measurements_to_targets
from world_model.training.perturbations import perturb_belief
from world_model.utils.config import OrpheusConfig

_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M = 0.5
_MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M = 0.5
_MIN_DETERMINISTIC_RGB_VISIBLE_FRACTION = 0.5


@dataclass
class TrainingBatchResult:
    """Losses and detached diagnostics for one optimiser step."""

    total_loss: Tensor
    loss_terms: dict[str, Tensor]
    metrics: dict[str, float]
    phase: str
    # Non-double-counted intermediate objectives used only to prove that an
    # optimizer-relevant branch contributed to ``total_loss``. They are never
    # added to the optimized total a second time.
    support_terms: dict[str, Tensor] = field(default_factory=dict)
    # Optional differentiable sufficient statistics for the frozen-reference
    # non-regression objective.  Ordinary training and validation leave this
    # empty, preserving the historical result contract.
    protected_objective_cells: dict[str, ProtectedObjectiveCell] = field(default_factory=dict)


@dataclass(frozen=True)
class ProtectedObjectiveCell:
    """Per-scenario additive error and support for one semantic cell."""

    error_sum: Tensor
    coordinate_count: Tensor


@dataclass
class _RolloutLossResult:
    """One posterior rollout shared by losses, diagnostics, and correction."""

    losses: dict[str, Tensor]
    frame_offsets: list[int]
    query_seconds: list[float]
    positions: Tensor | None
    velocities: Tensor | None
    position_log_variance: Tensor | None
    active_mask: Tensor | None
    physical_metrics: dict[str, float]
    protected_objective_cells: dict[str, ProtectedObjectiveCell]


@dataclass(frozen=True)
class _DeferredValidationRollout:
    """Immutable post-ingest anchor used by validation-only rollout batching."""

    frame_index: int
    belief: WorldBelief
    indices: Tensor
    matched: Tensor


@dataclass(frozen=True)
class _ValidationRolloutExecution:
    """Per-chunk results plus truthful posterior-rollout execution counts."""

    results: list[_RolloutLossResult]
    batched_anchor_count: int
    serial_fallback_anchor_count: int
    rollout_call_count: int


def _accumulate_float_metrics(
    destination: dict[str, float],
    source: Mapping[str, float],
) -> None:
    for name, value in source.items():
        destination[name] = destination.get(name, 0.0) + float(value)


def _masked_squared_error(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[float, int]:
    """Return detached coordinate-wise SSE/count under an object mask."""

    if prediction.shape != target.shape:
        raise ValueError("physical diagnostic prediction and target shapes must match")
    expanded = mask
    while expanded.ndim < prediction.ndim:
        expanded = expanded.unsqueeze(-1)
    values = (prediction - target).masked_select(expanded.expand_as(prediction))
    if values.numel() == 0:
        return 0.0, 0
    detached = values.detach().float()
    return float(detached.square().sum().cpu()), int(detached.numel())


def _add_squared_error_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> None:
    squared_error, coordinate_count = _masked_squared_error(
        prediction,
        target,
        mask,
    )
    metrics[f"{prefix}_sse"] = squared_error
    metrics[f"{prefix}_coordinate_count"] = float(coordinate_count)


def _add_gaussian_coverage_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
    z_quantile: float = 1.64485363,
) -> None:
    """Add detached coordinate counts inside a marginal Gaussian interval."""

    if mean.shape != target.shape or mean.shape != log_variance.shape:
        raise ValueError("coverage mean, target, and log variance shapes must match")
    expanded = mask
    while expanded.ndim < mean.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(mean)
    error = (mean - target).detach().abs()
    standard_deviation = (0.5 * log_variance.detach().clamp(-12.0, 8.0)).exp()
    covered = (error <= z_quantile * standard_deviation) & expanded
    metrics[f"{prefix}_hit_count"] = float(covered.sum().cpu())
    metrics[f"{prefix}_coordinate_count"] = float(expanded.sum().cpu())


def _add_gaussian_calibration_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
) -> None:
    """Add detached Gaussian NLL/sharpness sufficient statistics.

    These are validation diagnostics only. Persisting sums and coordinate
    counts lets the fixed-manifest selector pool episodes and scenario slices
    exactly instead of averaging already-derived NLL or sharpness values.
    """

    if mean.shape != target.shape or mean.shape != log_variance.shape:
        raise ValueError("calibration mean, target, and log variance shapes must match")
    expanded = mask
    while expanded.ndim < mean.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(mean)
    error = (mean - target).detach().masked_select(expanded).float()
    selected_log_variance = log_variance.detach().clamp(-12.0, 8.0).masked_select(expanded).float()
    if error.numel() == 0:
        metrics[f"{prefix}_gaussian_nll_sum"] = 0.0
        metrics[f"{prefix}_sharpness_std_sum"] = 0.0
        metrics[f"{prefix}_calibration_coordinate_count"] = 0.0
        return
    variance = selected_log_variance.exp()
    gaussian_nll = 0.5 * (
        error.square() / variance + selected_log_variance + math.log(2.0 * math.pi)
    )
    metrics[f"{prefix}_gaussian_nll_sum"] = float(gaussian_nll.sum().cpu())
    metrics[f"{prefix}_sharpness_std_sum"] = float((0.5 * selected_log_variance).exp().sum().cpu())
    metrics[f"{prefix}_calibration_coordinate_count"] = float(error.numel())


def _write_pooled_gaussian_calibration_from_axes(
    metrics: dict[str, float],
    *,
    prefix: str,
) -> None:
    """Write one canonical pooled calibration sum from emitted axes.

    The pooled and axis reductions describe the same detached coordinates.
    Reducing the pooled float32 tensor independently can nevertheless differ
    from the sum of three axis reductions, especially when signed Gaussian NLL
    values cancel.  Emit the axis sufficient statistics once and derive the
    pooled values with ``math.fsum`` so the persisted additive partition is
    exact without widening the fail-closed validator tolerance.
    """

    if "@" in prefix:
        base, horizon = prefix.split("@", 1)
        axis_prefixes = tuple(f"{base}_{axis_name}@{horizon}" for axis_name in ("x", "y", "z"))
    else:
        axis_prefixes = tuple(f"{prefix}_{axis_name}" for axis_name in ("x", "y", "z"))
    for metric_suffix in (
        "gaussian_nll_sum",
        "sharpness_std_sum",
        "calibration_coordinate_count",
    ):
        metrics[f"{prefix}_{metric_suffix}"] = math.fsum(
            metrics[f"{axis_prefix}_{metric_suffix}"] for axis_prefix in axis_prefixes
        )


def _write_axis_partitioned_gaussian_calibration_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
) -> None:
    """Emit x/y/z calibration statistics and their canonical pooled sum."""

    if mean.shape != target.shape or mean.shape != log_variance.shape:
        raise ValueError("calibration mean, target, and log variance shapes must match")
    if mean.ndim < 1 or mean.shape[-1] != 3:
        raise ValueError("axis-partitioned calibration tensors must end in x/y/z")
    if "@" in prefix:
        base, horizon = prefix.split("@", 1)
        axis_prefixes = tuple(f"{base}_{axis_name}@{horizon}" for axis_name in ("x", "y", "z"))
    else:
        axis_prefixes = tuple(f"{prefix}_{axis_name}" for axis_name in ("x", "y", "z"))
    for axis_index, axis_prefix in enumerate(axis_prefixes):
        _add_gaussian_calibration_metrics(
            metrics,
            prefix=axis_prefix,
            mean=mean[..., axis_index],
            target=target[..., axis_index],
            log_variance=log_variance[..., axis_index],
            mask=mask,
        )
    _write_pooled_gaussian_calibration_from_axes(metrics, prefix=prefix)


def _match_positions_to_targets(
    prediction_position: Tensor,
    prediction_active: Tensor,
    target_position: Tensor,
    target_active: Tensor,
) -> tuple[Tensor, Tensor]:
    """Hungarian-match arbitrary predicted positions to simulator targets.

    This evaluation-only helper is also used for forecast identity evidence;
    it never changes the runtime belief or rollout.
    """

    if prediction_position.ndim != 3 or prediction_position.shape[-1] != 3:
        raise ValueError("prediction_position must have shape [B,N,3]")
    if prediction_active.shape != prediction_position.shape[:2]:
        raise ValueError("prediction_active must have shape [B,N]")
    if target_position.ndim != 3 or target_position.shape[0] != prediction_position.shape[0]:
        raise ValueError("target_position must have shape [B,M,3]")
    if target_active.shape != target_position.shape[:2]:
        raise ValueError("target_active must have shape [B,M]")
    if prediction_active.dtype is not torch.bool or target_active.dtype is not torch.bool:
        raise TypeError("prediction and target active masks must be torch.bool")
    batch, prediction_count = prediction_active.shape
    indices = torch.full(
        (batch, prediction_count),
        -1,
        device=prediction_position.device,
        dtype=torch.int64,
    )
    matched = torch.zeros(
        (batch, prediction_count),
        device=prediction_position.device,
        dtype=torch.bool,
    )
    for batch_index in range(batch):
        prediction_slots = torch.nonzero(prediction_active[batch_index], as_tuple=False).flatten()
        target_slots = torch.nonzero(target_active[batch_index], as_tuple=False).flatten()
        if prediction_slots.numel() == 0 or target_slots.numel() == 0:
            continue
        cost = torch.cdist(
            prediction_position[batch_index, prediction_slots].detach().cpu(),
            target_position[batch_index, target_slots].detach().cpu(),
        )
        rows, columns = linear_sum_assignment(np.asarray(cost))
        selected_predictions = prediction_slots[
            torch.as_tensor(rows, device=prediction_slots.device, dtype=torch.int64)
        ]
        selected_targets = target_slots[
            torch.as_tensor(columns, device=target_slots.device, dtype=torch.int64)
        ]
        indices[batch_index, selected_predictions] = selected_targets
        matched[batch_index, selected_predictions] = True
    return indices, matched


def _f1_from_confusion(
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> tuple[float, float]:
    denominator = 2.0 * true_positive + false_positive + false_negative
    return (
        (2.0 * true_positive / denominator) if denominator > 0 else 0.0,
        denominator,
    )


def _distance_gate_physical_matches(
    prediction: Tensor,
    aligned_target: Tensor,
    assignment_mask: Tensor,
    *,
    threshold_m: float = _PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M,
) -> Tensor:
    """Apply the evaluator's metric-distance gate to assigned object pairs."""

    if threshold_m <= 0:
        raise ValueError("physical selection distance threshold must be positive")
    if prediction.shape != aligned_target.shape or prediction.shape[-1] != 3:
        raise ValueError("physical selection positions must share shape [B,N,3]")
    if assignment_mask.shape != prediction.shape[:-1]:
        raise ValueError("physical selection assignment mask must have shape [B,N]")
    distance = torch.linalg.vector_norm(prediction - aligned_target, dim=-1)
    return assignment_mask & torch.isfinite(distance) & (distance <= threshold_m)


class PhysicalMetricSupportError(RuntimeError):
    """A valid physical metric schema has no samples for a required score."""


def physical_validation_metrics(
    additive_metrics: Mapping[str, float],
    config: OrpheusConfig,
) -> dict[str, float]:
    """Convert additive loop diagnostics into physical selection metrics.

    Callers must first sum each ``physical_*_sse``/count over the complete
    validation manifest. Applying a common scaling to every additive field
    (for example, averaging all batch-one episodes) is also safe because each
    reported metric is a ratio of matching sums.
    """

    def required(name: str, *, signed: bool = False) -> float:
        if name not in additive_metrics:
            raise RuntimeError(f"missing additive physical validation metric {name!r}")
        value = float(additive_metrics[name])
        if not math.isfinite(value) or (value < 0 and not signed):
            raise ValueError(f"additive physical validation metric {name!r} is invalid")
        return value

    def rmse(sse_name: str, count_name: str) -> float:
        squared_error = required(sse_name)
        count = required(count_name)
        if count <= 0:
            raise PhysicalMetricSupportError(
                f"physical validation metric {count_name!r} has no support"
            )
        return math.sqrt(squared_error / count)

    def bounded_ratio(numerator_name: str, denominator_name: str) -> float:
        numerator = required(numerator_name)
        denominator = required(denominator_name)
        if denominator <= 0:
            return 0.0
        return min(1.0, max(0.0, numerator / denominator))

    def supported_mean(
        sum_name: str,
        count_name: str,
        *,
        signed: bool = False,
    ) -> float:
        total = required(sum_name, signed=signed)
        count = required(count_name)
        if count <= 0:
            raise PhysicalMetricSupportError(
                f"physical validation metric {count_name!r} has no support"
            )
        return total / count

    required_names = {
        "physical_state_position_sse",
        "physical_state_position_coordinate_count",
        "physical_state_velocity_sse",
        "physical_state_velocity_coordinate_count",
        "physical_distance_gated_matched_object_frames",
        "physical_distance_gated_target_object_frames",
        "physical_distance_gated_predicted_object_frames",
        "physical_distance_gated_identity_switches",
        "physical_distance_gated_object_frame_associations",
        "physical_position_coverage90_hit_count",
        "physical_position_coverage90_coordinate_count",
        "physical_state_position_coverage90_hit_count",
        "physical_state_position_coverage90_coordinate_count",
        "physical_state_position_gaussian_nll_sum",
        "physical_state_position_sharpness_std_sum",
        "physical_state_position_calibration_coordinate_count",
        "physical_collision_true_positive_count",
        "physical_collision_false_positive_count",
        "physical_collision_false_negative_count",
        "physical_collision_true_negative_count",
    }
    for axis_name in ("x", "y", "z"):
        required_names.update(
            {
                f"physical_state_position_{axis_name}_sse",
                f"physical_state_position_{axis_name}_coordinate_count",
                f"physical_state_velocity_{axis_name}_sse",
                f"physical_state_velocity_{axis_name}_coordinate_count",
                f"physical_state_position_{axis_name}_gaussian_nll_sum",
                f"physical_state_position_{axis_name}_sharpness_std_sum",
                f"physical_state_position_{axis_name}_calibration_coordinate_count",
            }
        )
    seen_offsets: set[int] = set()
    physical_suffixes: list[str] = []
    for horizon in config.evaluation.horizons_seconds:
        frame_offset = max(1, int(round(float(horizon) * config.simulator.frame_rate)))
        if frame_offset in seen_offsets:
            continue
        seen_offsets.add(frame_offset)
        physical_suffix = f"@{frame_offset / config.simulator.frame_rate:.3f}s"
        physical_suffixes.append(physical_suffix)
        required_names.update(
            {
                f"physical_rollout_position{physical_suffix}_sse",
                f"physical_rollout_position{physical_suffix}_coordinate_count",
                f"physical_rollout_velocity{physical_suffix}_sse",
                f"physical_rollout_velocity{physical_suffix}_coordinate_count",
                f"physical_rollout_position_coverage90{physical_suffix}_hit_count",
                f"physical_rollout_position_coverage90{physical_suffix}_coordinate_count",
                f"physical_rollout_position{physical_suffix}_gaussian_nll_sum",
                f"physical_rollout_position{physical_suffix}_sharpness_std_sum",
                f"physical_rollout_position{physical_suffix}_calibration_coordinate_count",
                f"physical_forecast_active_count{physical_suffix}",
                f"physical_forecast_tracked_count{physical_suffix}",
                f"physical_forecast_target_count{physical_suffix}",
                f"physical_forecast_predictable_target_count{physical_suffix}",
                f"physical_rollout_predictable_target_count{physical_suffix}",
                f"physical_rollout_censored_external_actuation_count{physical_suffix}",
                f"physical_forecast_identity_eligible_count{physical_suffix}",
                f"physical_forecast_identity_mismatch_count{physical_suffix}",
                f"physical_forecast_identity_association_count{physical_suffix}",
                f"physical_collision_true_positive_count{physical_suffix}",
                f"physical_collision_false_positive_count{physical_suffix}",
                f"physical_collision_false_negative_count{physical_suffix}",
                f"physical_collision_true_negative_count{physical_suffix}",
            }
        )
        for axis_name in ("x", "y", "z"):
            required_names.update(
                {
                    f"physical_rollout_position_{axis_name}{physical_suffix}_sse",
                    (f"physical_rollout_position_{axis_name}{physical_suffix}_coordinate_count"),
                    f"physical_rollout_velocity_{axis_name}{physical_suffix}_sse",
                    (f"physical_rollout_velocity_{axis_name}{physical_suffix}_coordinate_count"),
                    (f"physical_rollout_position_{axis_name}{physical_suffix}_gaussian_nll_sum"),
                    (f"physical_rollout_position_{axis_name}{physical_suffix}_sharpness_std_sum"),
                    (
                        f"physical_rollout_position_{axis_name}{physical_suffix}_"
                        "calibration_coordinate_count"
                    ),
                }
            )
    # Validate the complete schema before deciding that any zero denominator is
    # ordinary insufficient support. Otherwise an early zero can hide missing
    # or corrupt metrics later in the mapping.
    validated = {
        name: required(
            name,
            signed=name.endswith("_gaussian_nll_sum"),
        )
        for name in sorted(required_names)
    }

    def require_axis_partition(
        aggregate_name: str,
        axis_names: tuple[str, ...],
        *,
        quantity: str,
    ) -> None:
        """Reject internally contradictory additive evidence.

        Aggregate and axis statistics are emitted from the same masked tensor.
        Persisting both makes every derived axis/horizon score independently
        reproducible, but only if a checkpoint cannot substitute unrelated
        axis sums while retaining a plausible pooled value.  A small relative
        tolerance allows the different floating-point reduction orders used by
        the aggregate and three scalar reductions; support counts remain exact.
        """

        aggregate = validated[aggregate_name]
        axis_total = sum(validated[name] for name in axis_names)
        relative_tolerance = 0.0 if quantity == "count" else 1.0e-6
        absolute_tolerance = 1.0e-9 if quantity == "count" else 1.0e-7
        if not math.isclose(
            aggregate,
            axis_total,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"additive physical validation metric {aggregate_name!r} does not "
                f"equal its x/y/z {quantity} partition"
            )

    for stem in ("physical_state_position", "physical_state_velocity"):
        require_axis_partition(
            f"{stem}_sse",
            tuple(f"{stem}_{axis_name}_sse" for axis_name in ("x", "y", "z")),
            quantity="SSE",
        )
        require_axis_partition(
            f"{stem}_coordinate_count",
            tuple(f"{stem}_{axis_name}_coordinate_count" for axis_name in ("x", "y", "z")),
            quantity="count",
        )
    if (
        validated["physical_state_position_coordinate_count"]
        != validated["physical_state_velocity_coordinate_count"]
    ):
        raise ValueError("current position and velocity evidence must use identical support")
    for axis_name in ("x", "y", "z"):
        if (
            validated[f"physical_state_position_{axis_name}_coordinate_count"]
            != validated[f"physical_state_velocity_{axis_name}_coordinate_count"]
        ):
            raise ValueError(
                f"current position and velocity {axis_name}-axis evidence must use identical support"
            )
    current_position_count = validated["physical_state_position_coordinate_count"]
    for count_name in (
        "physical_state_position_coverage90_coordinate_count",
        "physical_state_position_calibration_coordinate_count",
    ):
        if validated[count_name] != current_position_count:
            raise ValueError(
                f"current uncertainty evidence {count_name!r} must use current position support"
            )
    for physical_suffix in physical_suffixes:
        for stem in (
            f"physical_rollout_position{physical_suffix}",
            f"physical_rollout_velocity{physical_suffix}",
        ):
            base, suffix = stem.split("@", 1)
            suffix = f"@{suffix}"
            require_axis_partition(
                f"{stem}_sse",
                tuple(f"{base}_{axis_name}{suffix}_sse" for axis_name in ("x", "y", "z")),
                quantity="SSE",
            )
            require_axis_partition(
                f"{stem}_coordinate_count",
                tuple(
                    f"{base}_{axis_name}{suffix}_coordinate_count" for axis_name in ("x", "y", "z")
                ),
                quantity="count",
            )
        position_count = validated[f"physical_rollout_position{physical_suffix}_coordinate_count"]
        velocity_count = validated[f"physical_rollout_velocity{physical_suffix}_coordinate_count"]
        if position_count != velocity_count:
            raise ValueError(
                f"rollout position and velocity evidence at {physical_suffix} must use identical support"
            )
        for axis_name in ("x", "y", "z"):
            if (
                validated[
                    f"physical_rollout_position_{axis_name}{physical_suffix}_coordinate_count"
                ]
                != validated[
                    f"physical_rollout_velocity_{axis_name}{physical_suffix}_coordinate_count"
                ]
            ):
                raise ValueError(
                    f"rollout position and velocity {axis_name}-axis evidence at "
                    f"{physical_suffix} must use identical support"
                )
        calibration_count = validated[
            f"physical_rollout_position{physical_suffix}_calibration_coordinate_count"
        ]
        coverage_count = validated[
            f"physical_rollout_position_coverage90{physical_suffix}_coordinate_count"
        ]
        if coverage_count != calibration_count:
            raise ValueError(
                f"rollout coverage and likelihood at {physical_suffix} must use identical support"
            )

    calibration_stems = ["physical_state_position"]
    calibration_stems.extend(
        f"physical_rollout_position{physical_suffix}" for physical_suffix in physical_suffixes
    )
    for stem in calibration_stems:
        if "@" in stem:
            base, suffix = stem.split("@", 1)
            suffix = f"@{suffix}"
        else:
            base, suffix = stem, ""
        for metric_suffix, quantity in (
            ("gaussian_nll_sum", "Gaussian NLL sum"),
            ("sharpness_std_sum", "sharpness sum"),
            ("calibration_coordinate_count", "count"),
        ):
            require_axis_partition(
                f"{stem}_{metric_suffix}",
                tuple(
                    f"{base}_{axis_name}{suffix}_{metric_suffix}" for axis_name in ("x", "y", "z")
                ),
                quantity=quantity,
            )

    squared_error_pairs = [
        (
            "physical_state_position_sse",
            "physical_state_position_coordinate_count",
        ),
        (
            "physical_state_velocity_sse",
            "physical_state_velocity_coordinate_count",
        ),
    ]
    squared_error_pairs.extend(
        (
            f"physical_state_position_{axis_name}_sse",
            f"physical_state_position_{axis_name}_coordinate_count",
        )
        for axis_name in ("x", "y", "z")
    )
    squared_error_pairs.extend(
        (
            f"physical_state_velocity_{axis_name}_sse",
            f"physical_state_velocity_{axis_name}_coordinate_count",
        )
        for axis_name in ("x", "y", "z")
    )
    squared_error_pairs.extend(
        (
            f"physical_rollout_position{physical_suffix}_sse",
            f"physical_rollout_position{physical_suffix}_coordinate_count",
        )
        for physical_suffix in physical_suffixes
    )
    squared_error_pairs.extend(
        (
            f"physical_rollout_position_{axis_name}{physical_suffix}_sse",
            f"physical_rollout_position_{axis_name}{physical_suffix}_coordinate_count",
        )
        for physical_suffix in physical_suffixes
        for axis_name in ("x", "y", "z")
    )
    squared_error_pairs.extend(
        (
            f"physical_rollout_velocity{physical_suffix}_sse",
            f"physical_rollout_velocity{physical_suffix}_coordinate_count",
        )
        for physical_suffix in physical_suffixes
    )
    squared_error_pairs.extend(
        (
            f"physical_rollout_velocity_{axis_name}{physical_suffix}_sse",
            f"physical_rollout_velocity_{axis_name}{physical_suffix}_coordinate_count",
        )
        for physical_suffix in physical_suffixes
        for axis_name in ("x", "y", "z")
    )
    for sse_name, count_name in squared_error_pairs:
        if validated[count_name] == 0.0 and validated[sse_name] != 0.0:
            raise ValueError(
                f"additive physical validation metric {sse_name!r} must be "
                f"zero when {count_name!r} has no support"
            )

    calibration_triplets = [
        (
            "physical_state_position_gaussian_nll_sum",
            "physical_state_position_sharpness_std_sum",
            "physical_state_position_calibration_coordinate_count",
        ),
        *[
            (
                f"physical_state_position_{axis_name}_gaussian_nll_sum",
                f"physical_state_position_{axis_name}_sharpness_std_sum",
                f"physical_state_position_{axis_name}_calibration_coordinate_count",
            )
            for axis_name in ("x", "y", "z")
        ],
        *[
            (
                f"physical_rollout_position{physical_suffix}_gaussian_nll_sum",
                f"physical_rollout_position{physical_suffix}_sharpness_std_sum",
                f"physical_rollout_position{physical_suffix}_calibration_coordinate_count",
            )
            for physical_suffix in physical_suffixes
        ],
        *[
            (
                f"physical_rollout_position_{axis_name}{physical_suffix}_gaussian_nll_sum",
                f"physical_rollout_position_{axis_name}{physical_suffix}_sharpness_std_sum",
                (
                    f"physical_rollout_position_{axis_name}{physical_suffix}_"
                    "calibration_coordinate_count"
                ),
            )
            for physical_suffix in physical_suffixes
            for axis_name in ("x", "y", "z")
        ],
    ]
    for nll_name, sharpness_name, count_name in calibration_triplets:
        if validated[count_name] == 0.0 and (
            validated[nll_name] != 0.0 or validated[sharpness_name] != 0.0
        ):
            raise ValueError(
                f"calibration sums {nll_name!r}/{sharpness_name!r} must be zero "
                f"when {count_name!r} has no support"
            )

    bounded_count_pairs = [
        (
            "physical_distance_gated_matched_object_frames",
            "physical_distance_gated_target_object_frames",
        ),
        (
            "physical_distance_gated_matched_object_frames",
            "physical_distance_gated_predicted_object_frames",
        ),
        (
            "physical_distance_gated_identity_switches",
            "physical_distance_gated_object_frame_associations",
        ),
        (
            "physical_position_coverage90_hit_count",
            "physical_position_coverage90_coordinate_count",
        ),
        (
            "physical_state_position_coverage90_hit_count",
            "physical_state_position_coverage90_coordinate_count",
        ),
    ]
    for physical_suffix in physical_suffixes:
        bounded_count_pairs.extend(
            [
                (
                    f"physical_forecast_active_count{physical_suffix}",
                    f"physical_forecast_tracked_count{physical_suffix}",
                ),
                (
                    f"physical_forecast_tracked_count{physical_suffix}",
                    f"physical_forecast_target_count{physical_suffix}",
                ),
                (
                    f"physical_forecast_predictable_target_count{physical_suffix}",
                    f"physical_forecast_target_count{physical_suffix}",
                ),
                (
                    f"physical_rollout_predictable_target_count{physical_suffix}",
                    f"physical_forecast_tracked_count{physical_suffix}",
                ),
                (
                    f"physical_rollout_censored_external_actuation_count{physical_suffix}",
                    f"physical_forecast_tracked_count{physical_suffix}",
                ),
                (
                    f"physical_rollout_position_coverage90{physical_suffix}_hit_count",
                    (f"physical_rollout_position_coverage90{physical_suffix}_coordinate_count"),
                ),
                (
                    f"physical_forecast_identity_eligible_count{physical_suffix}",
                    f"physical_forecast_active_count{physical_suffix}",
                ),
                (
                    f"physical_forecast_identity_association_count{physical_suffix}",
                    f"physical_forecast_identity_eligible_count{physical_suffix}",
                ),
                (
                    f"physical_forecast_identity_mismatch_count{physical_suffix}",
                    f"physical_forecast_identity_association_count{physical_suffix}",
                ),
            ]
        )
    for numerator_name, denominator_name in bounded_count_pairs:
        if validated[numerator_name] > validated[denominator_name]:
            raise ValueError(
                f"additive physical validation metric {numerator_name!r} "
                f"exceeds denominator {denominator_name!r}"
            )

    pooled_coverage_hits = sum(
        validated[f"physical_rollout_position_coverage90{suffix}_hit_count"]
        for suffix in physical_suffixes
    )
    pooled_coverage_count = sum(
        validated[f"physical_rollout_position_coverage90{suffix}_coordinate_count"]
        for suffix in physical_suffixes
    )
    if (
        validated["physical_position_coverage90_hit_count"] != pooled_coverage_hits
        or validated["physical_position_coverage90_coordinate_count"] != pooled_coverage_count
    ):
        raise ValueError("pooled rollout coverage evidence must equal the horizon partition")
    for confusion_name in (
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ):
        pooled_name = f"physical_collision_{confusion_name}_count"
        horizon_total = sum(
            validated[f"physical_collision_{confusion_name}_count{suffix}"]
            for suffix in physical_suffixes
        )
        if validated[pooled_name] != horizon_total:
            raise ValueError(
                f"pooled collision {confusion_name} evidence must equal the horizon partition"
            )

    output = {
        "validation_position_rmse_m": rmse(
            "physical_state_position_sse",
            "physical_state_position_coordinate_count",
        ),
        "validation_velocity_rmse_mps": rmse(
            "physical_state_velocity_sse",
            "physical_state_velocity_coordinate_count",
        ),
        "validation_target_coverage": bounded_ratio(
            "physical_distance_gated_matched_object_frames",
            "physical_distance_gated_target_object_frames",
        ),
        "validation_prediction_precision": bounded_ratio(
            "physical_distance_gated_matched_object_frames",
            "physical_distance_gated_predicted_object_frames",
        ),
        "validation_id_switch_rate": bounded_ratio(
            "physical_distance_gated_identity_switches",
            "physical_distance_gated_object_frame_associations",
        ),
        "validation_position_coverage90": bounded_ratio(
            "physical_position_coverage90_hit_count",
            "physical_position_coverage90_coordinate_count",
        ),
        "validation_current_position_coverage90": bounded_ratio(
            "physical_state_position_coverage90_hit_count",
            "physical_state_position_coverage90_coordinate_count",
        ),
        "validation_current_position_gaussian_nll": supported_mean(
            "physical_state_position_gaussian_nll_sum",
            "physical_state_position_calibration_coordinate_count",
            signed=True,
        ),
        "validation_current_position_sharpness_std": supported_mean(
            "physical_state_position_sharpness_std_sum",
            "physical_state_position_calibration_coordinate_count",
        ),
    }
    for axis_name in ("x", "y", "z"):
        axis_sse = f"physical_state_position_{axis_name}_sse"
        axis_count = f"physical_state_position_{axis_name}_coordinate_count"
        output[f"validation_position_rmse_{axis_name}_m"] = rmse(
            axis_sse,
            axis_count,
        )
        output[f"validation_velocity_rmse_{axis_name}_mps"] = rmse(
            f"physical_state_velocity_{axis_name}_sse",
            f"physical_state_velocity_{axis_name}_coordinate_count",
        )
        output[f"validation_current_position_gaussian_nll_{axis_name}"] = supported_mean(
            f"physical_state_position_{axis_name}_gaussian_nll_sum",
            f"physical_state_position_{axis_name}_calibration_coordinate_count",
            signed=True,
        )
        output[f"validation_current_position_sharpness_std_{axis_name}"] = supported_mean(
            f"physical_state_position_{axis_name}_sharpness_std_sum",
            f"physical_state_position_{axis_name}_calibration_coordinate_count",
        )
    true_positive = required("physical_collision_true_positive_count")
    false_positive = required("physical_collision_false_positive_count")
    false_negative = required("physical_collision_false_negative_count")
    output["validation_collision_f1"] = _f1_from_confusion(
        true_positive,
        false_positive,
        false_negative,
    )[0]
    for physical_suffix in physical_suffixes:
        output[f"validation_position_rmse{physical_suffix}"] = rmse(
            f"physical_rollout_position{physical_suffix}_sse",
            f"physical_rollout_position{physical_suffix}_coordinate_count",
        )
        for axis_name in ("x", "y", "z"):
            axis_sse = f"physical_rollout_position_{axis_name}{physical_suffix}_sse"
            axis_count = f"physical_rollout_position_{axis_name}{physical_suffix}_coordinate_count"
            output[f"validation_position_rmse_{axis_name}{physical_suffix}"] = rmse(
                axis_sse, axis_count
            )
            output[f"validation_velocity_rmse_{axis_name}{physical_suffix}"] = rmse(
                f"physical_rollout_velocity_{axis_name}{physical_suffix}_sse",
                f"physical_rollout_velocity_{axis_name}{physical_suffix}_coordinate_count",
            )
            output[f"validation_position_gaussian_nll_{axis_name}{physical_suffix}"] = (
                supported_mean(
                    f"physical_rollout_position_{axis_name}{physical_suffix}_gaussian_nll_sum",
                    (
                        f"physical_rollout_position_{axis_name}{physical_suffix}_"
                        "calibration_coordinate_count"
                    ),
                    signed=True,
                )
            )
            output[f"validation_position_sharpness_std_{axis_name}{physical_suffix}"] = (
                supported_mean(
                    f"physical_rollout_position_{axis_name}{physical_suffix}_sharpness_std_sum",
                    (
                        f"physical_rollout_position_{axis_name}{physical_suffix}_"
                        "calibration_coordinate_count"
                    ),
                )
            )
        output[f"validation_velocity_rmse{physical_suffix}"] = rmse(
            f"physical_rollout_velocity{physical_suffix}_sse",
            f"physical_rollout_velocity{physical_suffix}_coordinate_count",
        )
        output[f"validation_forecast_target_coverage{physical_suffix}"] = bounded_ratio(
            f"physical_forecast_active_count{physical_suffix}",
            f"physical_forecast_target_count{physical_suffix}",
        )
        output[f"validation_forecast_identity_mismatch_rate{physical_suffix}"] = bounded_ratio(
            f"physical_forecast_identity_mismatch_count{physical_suffix}",
            f"physical_forecast_identity_association_count{physical_suffix}",
        )
        output[f"validation_forecast_identity_association_coverage{physical_suffix}"] = (
            bounded_ratio(
                f"physical_forecast_identity_association_count{physical_suffix}",
                f"physical_forecast_identity_eligible_count{physical_suffix}",
            )
        )
        output[f"validation_position_coverage90{physical_suffix}"] = bounded_ratio(
            f"physical_rollout_position_coverage90{physical_suffix}_hit_count",
            f"physical_rollout_position_coverage90{physical_suffix}_coordinate_count",
        )
        output[f"validation_position_gaussian_nll{physical_suffix}"] = supported_mean(
            f"physical_rollout_position{physical_suffix}_gaussian_nll_sum",
            f"physical_rollout_position{physical_suffix}_calibration_coordinate_count",
            signed=True,
        )
        output[f"validation_position_sharpness_std{physical_suffix}"] = supported_mean(
            f"physical_rollout_position{physical_suffix}_sharpness_std_sum",
            f"physical_rollout_position{physical_suffix}_calibration_coordinate_count",
        )
        horizon_true_positive = required(f"physical_collision_true_positive_count{physical_suffix}")
        horizon_false_positive = required(
            f"physical_collision_false_positive_count{physical_suffix}"
        )
        horizon_false_negative = required(
            f"physical_collision_false_negative_count{physical_suffix}"
        )
        horizon_true_negative = required(f"physical_collision_true_negative_count{physical_suffix}")
        if (
            horizon_true_positive
            + horizon_false_positive
            + horizon_false_negative
            + horizon_true_negative
            <= 0
        ):
            raise PhysicalMetricSupportError(
                f"collision validation at {physical_suffix} has no evaluated support"
            )
        if horizon_true_positive + horizon_false_negative <= 0:
            raise PhysicalMetricSupportError(
                f"collision validation at {physical_suffix} has no positive-label support"
            )
        if horizon_false_positive + horizon_true_negative <= 0:
            raise PhysicalMetricSupportError(
                f"collision validation at {physical_suffix} has no negative-label support"
            )
        output[f"validation_collision_f1{physical_suffix}"] = _f1_from_confusion(
            horizon_true_positive,
            horizon_false_positive,
            horizon_false_negative,
        )[0]
        if required(f"physical_forecast_identity_eligible_count{physical_suffix}") <= 0:
            raise PhysicalMetricSupportError(
                f"forecast identity validation at {physical_suffix} has no eligible support"
            )
        if required(f"physical_forecast_identity_association_count{physical_suffix}") <= 0:
            raise PhysicalMetricSupportError(
                f"forecast identity validation at {physical_suffix} has no support"
            )
    return output


@dataclass
class PersistentTargetMatcher:
    """Keep training-only simulator targets aligned to persistent belief IDs.

    A distance-gated positional Hungarian match bootstraps new runtime tracks.
    Once an internal object ID is mapped, that target slot is retained while
    both remain active. This prevents close contacts from silently swapping the
    velocity/event supervision of two nearby objects while ensuring a far
    false-positive track cannot acquire positive state/existence supervision
    merely because a simulator target remains available.
    """

    mappings: list[dict[int, int]] = field(default_factory=list)

    def match(
        self,
        belief: WorldBelief,
        target_position: Tensor,
        target_active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        objects = belief.objects
        batch, belief_count = objects.active.shape
        if target_position.ndim != 3 or target_position.shape[0] != batch:
            raise ValueError("target_position must have shape [B,N,3]")
        if target_active.shape != target_position.shape[:2]:
            raise ValueError("target_active must have shape [B,N]")
        if len(self.mappings) != batch:
            self.mappings = [{} for _ in range(batch)]

        indices = torch.full(
            (batch, belief_count),
            -1,
            device=objects.position.device,
            dtype=torch.int64,
        )
        matched = torch.zeros(
            (batch, belief_count),
            device=objects.position.device,
            dtype=torch.bool,
        )
        updated_mappings: list[dict[int, int]] = []
        for batch_index in range(batch):
            previous = self.mappings[batch_index]
            active_beliefs = torch.nonzero(
                objects.active[batch_index],
                as_tuple=False,
            ).flatten()
            active_targets = torch.nonzero(
                target_active[batch_index],
                as_tuple=False,
            ).flatten()
            current: dict[int, int] = {}
            used_targets: set[int] = set()
            unmatched_beliefs: list[int] = []
            for belief_slot_tensor in active_beliefs:
                belief_slot = int(belief_slot_tensor)
                object_id = int(objects.object_id[batch_index, belief_slot])
                target_slot = previous.get(object_id) if object_id >= 0 else None
                if (
                    target_slot is not None
                    and bool(target_active[batch_index, target_slot])
                    and target_slot not in used_targets
                ):
                    indices[batch_index, belief_slot] = target_slot
                    matched[batch_index, belief_slot] = True
                    current[object_id] = target_slot
                    used_targets.add(target_slot)
                else:
                    unmatched_beliefs.append(belief_slot)

            available_targets = [
                int(target_slot)
                for target_slot in active_targets
                if int(target_slot) not in used_targets
            ]
            if unmatched_beliefs and available_targets:
                belief_slots = torch.as_tensor(
                    unmatched_beliefs,
                    device=objects.position.device,
                    dtype=torch.int64,
                )
                target_slots = torch.as_tensor(
                    available_targets,
                    device=target_position.device,
                    dtype=torch.int64,
                )
                cost = torch.cdist(
                    objects.position[batch_index, belief_slots].detach().cpu(),
                    target_position[batch_index, target_slots].detach().cpu(),
                )
                admissible = torch.isfinite(cost) & (
                    cost <= _PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M
                )
                # Gate impossible pairs before Hungarian assignment.  A
                # uniform penalty larger than the sum of every possible valid
                # edge makes the solver maximize admissible cardinality first
                # and minimize metric distance second. Invalid solver pairs
                # are discarded below, leaving both the belief and target
                # explicitly unmatched.
                maximum_assignment_count = min(cost.shape)
                invalid_cost = (
                    maximum_assignment_count + 1
                ) * _PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M + 1.0
                assignment_cost = torch.where(
                    admissible,
                    cost,
                    torch.full_like(cost, invalid_cost),
                )
                rows, columns = linear_sum_assignment(np.asarray(assignment_cost))
                for row, column in zip(rows, columns, strict=True):
                    if not bool(admissible[int(row), int(column)]):
                        continue
                    belief_slot = unmatched_beliefs[int(row)]
                    target_slot = available_targets[int(column)]
                    indices[batch_index, belief_slot] = target_slot
                    matched[batch_index, belief_slot] = True
                    object_id = int(objects.object_id[batch_index, belief_slot])
                    if object_id >= 0:
                        current[object_id] = target_slot
            updated_mappings.append(current)
        self.mappings = updated_mappings
        return indices, matched


def move_batch_to_device(
    value: Any,
    device: torch.device | str,
) -> Any:
    """Move tensors in a nested episode batch while preserving integer types."""

    if isinstance(value, Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device) for item in value)
    return value


def _shared_timestamp(timestamps: Tensor, frame_index: int) -> float:
    values = timestamps[:, frame_index]
    first = values[0]
    if not torch.allclose(values, first.expand_as(values), atol=1.0e-7, rtol=0):
        raise ValueError("batched ObservationPacket creation requires a shared frame timestamp")
    return float(first.detach().cpu())


def make_rgb_packet(
    batch: Mapping[str, Any],
    frame_index: int,
) -> ObservationPacket:
    """Create a batched RGB-only packet from canonical ``[B,T,...]`` data."""

    rgb = batch["rgb"]
    if not isinstance(rgb, Tensor) or rgb.ndim != 5:
        raise ValueError("batch.rgb must have shape [B,T,3,H,W]")
    if not 0 <= frame_index < rgb.shape[1]:
        raise IndexError(frame_index)
    camera = batch["camera"]
    timestamp = _shared_timestamp(batch["timestamps"], frame_index)
    return ObservationPacket(
        modality="rgb",
        sensor_id="camera0",
        timestamp=timestamp,
        payload=rgb[:, frame_index],
        calibration={
            "world_from_camera": camera["world_from_camera"][:, frame_index],
            "intrinsics": camera["intrinsics"][:, frame_index],
        },
        frame_id="camera:camera0",
        confidence=1.0,
        metadata={
            "image_size": (int(rgb.shape[-2]), int(rgb.shape[-1])),
            "training_frame_index": int(frame_index),
        },
    )


def _target_measurement_values(
    batch: Mapping[str, Any],
    frame_index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    labels = batch["labels"]
    values = torch.cat(
        (
            labels["projected_center"][:, frame_index],
            labels["log_apparent_radius_normalized"][:, frame_index].unsqueeze(-1),
            labels["inverse_depth"][:, frame_index].unsqueeze(-1),
            labels["albedo"][:, frame_index],
        ),
        dim=-1,
    )
    mask = (
        labels["existence"][:, frame_index].bool()
        & labels["projected_valid"][:, frame_index].bool()
        & labels["visible"][:, frame_index].bool()
    )
    visibility = labels["visible_fraction"][:, frame_index]
    return values, mask, visibility


def _reliably_observable_rgb_geometry(
    target_values: Tensor,
    visibility: Tensor,
) -> Tensor:
    """Return where exact RGB geometry is supported by visible pixels.

    Any visible rim is enough to retain an existence target and an uncertainty
    calibration signal.  It is not enough to demand an exact projected centre,
    radius, depth, or metric position when most of the sphere is occluded or
    its centre lies outside the image.  The latter is a conservative,
    resolution-independent definition of severe clipping.
    """

    if target_values.ndim != 3 or target_values.shape[-1] < 4:
        raise ValueError("target_values must have shape [B,N,D>=4]")
    if visibility.shape != target_values.shape[:2]:
        raise ValueError("visibility must match target [B,N] axes")
    centre = target_values[..., :2]
    return (
        torch.isfinite(target_values[..., :4]).all(dim=-1)
        & (visibility >= _MIN_DETERMINISTIC_RGB_VISIBLE_FRACTION)
        & (centre.abs() <= 1.0).all(dim=-1)
    )


def _target_disc_overlaps_rois(
    target_values: Tensor,
    rois: Tensor,
) -> Tensor:
    """Return whether each target sphere has pixel support inside its ROI.

    Fast measurements are residual queries over a projected crop.  A
    persistent assignment can remain valid after the target has moved far
    outside that crop; supervising its exact geometry in that case teaches the
    ROI head from pixels that cannot contain the requested object.
    """

    if target_values.ndim != 3 or target_values.shape[-1] < 3:
        raise ValueError("target_values must have shape [B,N,D>=3]")
    if rois.shape != (*target_values.shape[:2], 4):
        raise ValueError("rois must match target [B,N] axes and end in four bounds")
    centre = target_values[..., :2]
    radius = target_values[..., 2].exp()
    target_minimum = centre - radius.unsqueeze(-1)
    target_maximum = centre + radius.unsqueeze(-1)
    roi_minimum = rois[..., :2]
    roi_maximum = rois[..., 2:]
    finite = torch.isfinite(target_values[..., :3]).all(dim=-1) & torch.isfinite(rois).all(dim=-1)
    nondegenerate = (roi_maximum > roi_minimum).all(dim=-1)
    overlaps = (target_maximum >= roi_minimum).all(dim=-1) & (target_minimum <= roi_maximum).all(
        dim=-1
    )
    return finite & nondegenerate & overlaps


def gather_target_slots(target: Tensor, indices: Tensor) -> Tensor:
    """Gather target object slots into belief/proposal order.

    ``target`` is ``[B,N,...]`` and ``indices`` is ``[B,M]`` with ``-1`` for
    unmatched entries.  Unmatched outputs are zero and must be masked by the
    caller.
    """

    if target.ndim < 2 or indices.ndim != 2 or target.shape[0] != indices.shape[0]:
        raise ValueError("target/indices must begin with compatible [B,N] axes")
    safe = indices.clamp_min(0)
    tail = tuple(target.shape[2:])
    gather_index = safe.reshape(*safe.shape, *((1,) * len(tail))).expand(
        *safe.shape,
        *tail,
    )
    gathered = torch.gather(target, 1, gather_index)
    unmatched = indices < 0
    while unmatched.ndim < gathered.ndim:
        unmatched = unmatched.unsqueeze(-1)
    return torch.where(unmatched, torch.zeros_like(gathered), gathered)


def _soft_association_surrogate_losses(
    model: OnlineWorldModel,
    trace: DifferentiableIngestTrace,
    *,
    aligned_target_position: Tensor,
    aligned_target_velocity: Tensor,
    matched_belief_slots: Tensor,
    temperature: float,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Train the continuous evidence behind hard Hungarian association.

    Runtime identity and lifecycle remain discrete.  This relaxation instead
    differentiates the expected RGB world state under the same gated cost
    matrix used by Hungarian matching.  It therefore teaches proposal values,
    association features, projected priors, and recurrent state without
    pretending that integer assignment indices have a derivative.
    """

    predicted = trace.predicted_measurements
    measured = trace.measurements
    world_position = measured.auxiliary.get("world_position")
    if world_position is None:
        return {}, {"soft_association_supported_coordinate_count": 0.0}
    if world_position.shape != (*measured.measurement_mask.shape, 3):
        raise ValueError("soft association world_position must have shape [B,M,3]")
    cost = trace.association_cost_matrix
    admissible = (
        predicted.valid_mask.unsqueeze(-1)
        & measured.measurement_mask.unsqueeze(1)
        & torch.isfinite(cost)
        & (cost <= model.associator.maximum_cost)
    )
    logits = (-cost / temperature).masked_fill(~admissible, -1.0e9)
    weights = torch.softmax(logits, dim=-1) * admissible.to(dtype=cost.dtype)
    row_mass = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(
        row_mass > 0,
        weights / row_mass.clamp_min(torch.finfo(weights.dtype).eps),
        torch.zeros_like(weights),
    )
    expected_position = torch.einsum("bnm,bmd->bnd", weights, world_position)

    belief_indices = predicted.belief_indices
    target_position = gather_target_slots(aligned_target_position, belief_indices)
    target_velocity = gather_target_slots(aligned_target_velocity, belief_indices)
    target_matched = (
        gather_target_slots(
            matched_belief_slots.to(dtype=cost.dtype).unsqueeze(-1),
            belief_indices,
        )
        .squeeze(-1)
        .bool()
    )
    supported_rows = predicted.valid_mask & target_matched & (row_mass.squeeze(-1) > 0)
    independent_axis = measured.auxiliary.get("world_position_independent_axis_mask")
    if independent_axis is None:
        position_axis_support = supported_rows.unsqueeze(-1).expand_as(expected_position)
    else:
        if independent_axis.shape != world_position.shape or independent_axis.dtype != torch.bool:
            raise ValueError(
                "soft association world_position_independent_axis_mask must be boolean [B,M,3]"
            )
        expected_axis_support = torch.einsum(
            "bnm,bmd->bnd",
            weights,
            independent_axis.to(dtype=weights.dtype),
        )
        position_axis_support = supported_rows.unsqueeze(-1) & (expected_axis_support > 0.5)

    losses: dict[str, Tensor] = {
        "soft_association_state": masked_huber(
            expected_position,
            target_position,
            position_axis_support,
            batch_macro=True,
        )
    }
    velocity = measured.auxiliary.get("world_velocity")
    velocity_axis_valid = measured.auxiliary.get("world_velocity_axis_valid_mask")
    velocity_support_count = 0
    if velocity is not None and velocity_axis_valid is not None:
        if velocity.shape != world_position.shape or velocity_axis_valid.shape != velocity.shape:
            raise ValueError("soft association velocity tensors must have shape [B,M,3]")
        if velocity_axis_valid.dtype != torch.bool:
            raise TypeError("soft association velocity axis validity must be boolean")
        expected_velocity = torch.einsum("bnm,bmd->bnd", weights, velocity)
        expected_velocity_support = torch.einsum(
            "bnm,bmd->bnd",
            weights,
            velocity_axis_valid.to(dtype=weights.dtype),
        )
        velocity_mask = supported_rows.unsqueeze(-1) & (expected_velocity_support > 0.5)
        velocity_support_count = int(velocity_mask.sum().detach().cpu())
        if velocity_support_count:
            losses["soft_association_velocity"] = masked_huber(
                expected_velocity,
                target_velocity,
                velocity_mask,
                batch_macro=True,
            )

    # Sinkhorn is unnecessary at this tiny object count: a row-normalized
    # expectation plus a column-cap penalty provides the useful one-to-one
    # inductive bias without iterative normalization or extra rollout work.
    column_mass = weights.sum(dim=1)
    valid_columns = measured.measurement_mask
    overflow = torch.relu(column_mass - 1.0)
    losses["soft_association_exclusivity"] = masked_mean(
        overflow.square(),
        valid_columns,
        batch_macro=True,
    )
    metrics = {
        "soft_association_supported_coordinate_count": float(
            position_axis_support.sum().detach().cpu()
        ),
        "soft_association_velocity_supported_coordinate_count": float(velocity_support_count),
        "soft_association_supported_row_count": float(supported_rows.sum().detach().cpu()),
    }
    return losses, metrics


def gather_target_pairs(target: Tensor, indices: Tensor) -> Tensor:
    """Gather both object axes of a target relation matrix into belief order.

    ``target`` begins with ``[B,N,N]`` and ``indices`` is ``[B,M]`` with ``-1``
    for unmatched belief slots. Any relation incident to an unmatched slot is
    returned as zero and must remain masked by the caller.
    """

    if (
        target.ndim < 3
        or target.shape[1] != target.shape[2]
        or indices.ndim != 2
        or target.shape[0] != indices.shape[0]
    ):
        raise ValueError("target/indices must begin with compatible [B,N,N] and [B,M] axes")
    batch, belief_count = indices.shape
    safe = indices.clamp_min(0)
    batch_index = torch.arange(batch, device=target.device)[:, None, None]
    first_index = safe[:, :, None].expand(batch, belief_count, belief_count)
    second_index = safe[:, None, :].expand(batch, belief_count, belief_count)
    gathered = target[batch_index, first_index, second_index]
    unmatched = (indices < 0)[:, :, None] | (indices < 0)[:, None, :]
    while unmatched.ndim < gathered.ndim:
        unmatched = unmatched.unsqueeze(-1)
    return torch.where(unmatched, torch.zeros_like(gathered), gathered)


def _add_world_position_supervision(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    measurements: MeasurementSet,
    batch: Mapping[str, Any],
    frame_index: int,
    target_indices: Tensor,
) -> None:
    """Attach calibrated metric-space supervision when the RGB module emits it."""

    world_position = measurements.auxiliary.get("world_position")
    if world_position is None:
        return
    if not isinstance(world_position, Tensor):
        raise TypeError("measurements.auxiliary.world_position must be a Tensor")
    target_objects = batch.get("objects")
    if not isinstance(target_objects, Mapping):
        raise ValueError("metric RGB supervision requires batch.objects")
    target_position = target_objects.get("position")
    if not isinstance(target_position, Tensor):
        raise ValueError("metric RGB supervision requires batch.objects.position")
    outputs["world_position"] = world_position
    targets["world_position"] = gather_target_slots(
        target_position[:, frame_index],
        target_indices,
    )
    world_log_variance = measurements.auxiliary.get("world_position_log_variance")
    if world_log_variance is not None:
        if not isinstance(world_log_variance, Tensor):
            raise TypeError("measurements.auxiliary.world_position_log_variance must be a Tensor")
        outputs["world_position_log_variance"] = world_log_variance


def _add_appearance_supervision(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    measurements: MeasurementSet,
    batch: Mapping[str, Any],
    frame_index: int,
    target_indices: Tensor,
) -> None:
    """Train the runtime association embedding from RGB colour identity.

    Simulator albedo is a training-only label.  It is embedded into the
    modality's existing appearance dimension, so runtime association still
    consumes only RGB-derived embeddings and no privileged state.
    """

    if measurements.appearance is None:
        return
    target_objects = batch.get("objects")
    if not isinstance(target_objects, Mapping):
        raise ValueError("appearance supervision requires batch.objects")
    target_albedo = target_objects.get("albedo")
    if not isinstance(target_albedo, Tensor):
        raise ValueError("appearance supervision requires batch.objects.albedo")
    aligned_albedo = gather_target_slots(
        target_albedo[:, frame_index],
        target_indices,
    )
    appearance_target = torch.zeros_like(measurements.appearance)
    dimensions = min(aligned_albedo.shape[-1], appearance_target.shape[-1])
    appearance_target[..., :dimensions] = aligned_albedo[..., :dimensions]
    outputs["appearance"] = measurements.appearance
    targets["appearance"] = F.normalize(appearance_target, dim=-1)


def supervised_measurement_losses(
    module: torch.nn.Module,
    measurements: MeasurementSet,
    batch: Mapping[str, Any],
    frame_index: int,
) -> dict[str, Tensor]:
    """Hungarian-align RGB proposals and compute structured label losses."""

    target_values, target_mask, visibility = _target_measurement_values(batch, frame_index)
    aligned, matched, target_indices = match_measurements_to_targets(
        measurements.values,
        target_values,
        target_mask,
        existence_logits=measurements.existence_logits,
    )
    aligned_visibility = gather_target_slots(visibility.unsqueeze(-1), target_indices).squeeze(-1)
    aligned_geometry_support = (
        gather_target_slots(
            _reliably_observable_rgb_geometry(
                target_values,
                visibility,
            ).unsqueeze(-1),
            target_indices,
        )
        .squeeze(-1)
        .bool()
    )
    outputs = {
        "values": measurements.values,
        "log_variance": measurements.log_variance,
        "existence_logits": measurements.existence_logits,
    }
    raw_existence_logits = measurements.auxiliary.get("raw_existence_logits")
    if raw_existence_logits is not None:
        if not isinstance(raw_existence_logits, Tensor):
            raise TypeError("measurements.auxiliary.raw_existence_logits must be a Tensor")
        if raw_existence_logits.shape != measurements.existence_logits.shape:
            raise ValueError(
                "measurements.auxiliary.raw_existence_logits must match existence_logits"
            )
        outputs["existence_logits"] = raw_existence_logits
    raw_centre = measurements.auxiliary.get("raw_centre")
    if raw_centre is not None:
        if not isinstance(raw_centre, Tensor):
            raise TypeError("measurements.auxiliary.raw_centre must be a Tensor")
        outputs["raw_centre"] = raw_centre
    visibility_logits = measurements.auxiliary.get("visibility_logits")
    if visibility_logits is None:
        visibility_logits = measurements.auxiliary.get("visibility_logit")
    if visibility_logits is not None:
        outputs["visibility_logits"] = visibility_logits
    targets = {
        "values": aligned,
        "visibility": aligned_visibility,
    }
    _add_world_position_supervision(
        outputs,
        targets,
        measurements,
        batch,
        frame_index,
        target_indices,
    )
    _add_appearance_supervision(
        outputs,
        targets,
        measurements,
        batch,
        frame_index,
        target_indices,
    )
    masks = {
        "matched": matched,
        "existence": matched,
        # Every emitted discovery query is a real positive/negative proposal
        # decision, unlike inactive factory belief slots.
        "existence_valid": measurements.measurement_mask,
        "visibility_valid": matched,
        "geometry": matched & aligned_geometry_support,
    }
    losses = module.training_losses(outputs, targets, masks)
    dense_center_logits = measurements.auxiliary.get("dense_center_logits")
    if dense_center_logits is not None:
        if not isinstance(dense_center_logits, Tensor):
            raise TypeError("measurements.auxiliary.dense_center_logits must be a Tensor")
        if dense_center_logits.ndim != 4 or dense_center_logits.shape[:2] != (
            target_values.shape[0],
            1,
        ):
            raise ValueError("dense_center_logits must have shape [B,1,H,W]")
        # The dense focal map is the complete objectness objective. Avoid a
        # second, top-k-only BCE gradient that would change the qualified
        # center optimization problem.
        if "rgb_existence" in losses:
            losses["rgb_existence"] = losses["rgb_existence"].detach()
        labels = batch.get("labels")
        if not isinstance(labels, Mapping):
            raise ValueError("dense center supervision requires batch.labels")
        projected_center = labels.get("projected_center")
        projected_valid = labels.get("projected_valid")
        label_existence = labels.get("existence")
        label_visibility = labels.get("visible")
        if not all(
            isinstance(value, Tensor)
            for value in (
                projected_center,
                projected_valid,
                label_existence,
                label_visibility,
            )
        ):
            raise ValueError("dense center supervision requires projected RGB labels")
        assert isinstance(projected_center, Tensor)
        assert isinstance(projected_valid, Tensor)
        assert isinstance(label_existence, Tensor)
        assert isinstance(label_visibility, Tensor)
        centres = projected_center[:, frame_index]
        valid = (
            projected_valid[:, frame_index].bool()
            & label_existence[:, frame_index].bool()
            & label_visibility[:, frame_index].bool()
        )
        height, width = dense_center_logits.shape[-2:]
        x = ((centres[..., 0] + 1.0) * 0.5 * (width - 1)).round()
        y = ((centres[..., 1] + 1.0) * 0.5 * (height - 1)).round()
        yy = torch.arange(
            height,
            device=centres.device,
            dtype=centres.dtype,
        ).view(1, 1, height, 1)
        xx = torch.arange(
            width,
            device=centres.device,
            dtype=centres.dtype,
        ).view(1, 1, 1, width)
        distance_squared = (xx - x[..., None, None]).square() + (yy - y[..., None, None]).square()
        target_heatmap = torch.exp(-distance_squared / (2.0 * 2.0**2))
        target_heatmap = torch.where(
            valid[..., None, None],
            target_heatmap,
            torch.zeros_like(target_heatmap),
        ).amax(dim=1, keepdim=True)
        probability = dense_center_logits.sigmoid().clamp(1.0e-6, 1.0 - 1.0e-6)
        positive = target_heatmap.eq(1.0)
        negative = ~positive
        negative_weight = (1.0 - target_heatmap).pow(4.0)
        positive_loss = torch.log(probability) * (1.0 - probability).pow(2.0) * positive
        negative_loss = (
            torch.log(1.0 - probability) * probability.pow(2.0) * negative_weight * negative
        )
        losses["rgb_dense_center_heatmap"] = -(
            positive_loss.sum() + negative_loss.sum()
        ) / positive.sum().clamp_min(1)
    if not losses:
        raise RuntimeError("RGB observation module returned no training losses")
    return losses


def supervised_slot_measurement_losses(
    module: torch.nn.Module,
    measurements: MeasurementSet,
    batch: Mapping[str, Any],
    frame_index: int,
    *,
    target_indices: Tensor,
    matched_slots: Tensor,
    roi_bounds: Tensor | None = None,
) -> dict[str, Tensor]:
    """Supervise prior-conditioned RGB measurements in persistent slot order.

    Fast ROI measurements are conditioned on one particular belief object per
    output slot.  Their targets must therefore follow the belief-to-target
    assignment rather than being freely rematched according to the current
    measurement values.
    """

    slot_shape = measurements.values.shape[:2]
    if target_indices.shape != slot_shape or matched_slots.shape != slot_shape:
        raise ValueError("target_indices and matched_slots must match measurement [B,M] axes")
    if matched_slots.dtype != torch.bool:
        raise TypeError("matched_slots must be torch.bool")
    if roi_bounds is not None and roi_bounds.shape != (*slot_shape, 4):
        raise ValueError("roi_bounds must match measurement [B,M] axes and end in four bounds")

    target_values, target_mask, visibility = _target_measurement_values(batch, frame_index)
    aligned = gather_target_slots(target_values, target_indices)
    aligned_target_mask = (
        gather_target_slots(target_mask.unsqueeze(-1), target_indices).squeeze(-1).bool()
    )
    aligned_visibility = gather_target_slots(
        visibility.unsqueeze(-1),
        target_indices,
    ).squeeze(-1)
    mapped_valid = matched_slots & (target_indices >= 0)
    # Query validity and target identity are separate. Every projected,
    # runtime-valid persistent query is real evidence for the ROI
    # existence/visibility heads, even when training-only matching finds no
    # corresponding target. Such an unmapped query is a false-track negative;
    # only mapped queries with target pixels in their crop may supervise
    # attributes or exact state.
    roi_valid = measurements.measurement_mask
    crop_overlap = torch.ones_like(roi_valid)
    if roi_bounds is not None:
        crop_overlap = _target_disc_overlaps_rois(
            aligned,
            roi_bounds,
        )
    crop_evidence = mapped_valid & roi_valid & aligned_target_mask & crop_overlap
    exact_geometry = crop_evidence & _reliably_observable_rgb_geometry(
        aligned,
        aligned_visibility,
    )
    visibility_target = torch.where(
        crop_evidence,
        aligned_visibility,
        torch.zeros_like(aligned_visibility),
    )

    outputs = {
        "values": measurements.values,
        "log_variance": measurements.log_variance,
        "existence_logits": measurements.existence_logits,
    }
    raw_centre = measurements.auxiliary.get("raw_centre")
    if raw_centre is not None:
        if not isinstance(raw_centre, Tensor):
            raise TypeError("measurements.auxiliary.raw_centre must be a Tensor")
        outputs["raw_centre"] = raw_centre
    visibility_logits = measurements.auxiliary.get("visibility_logits")
    if visibility_logits is None:
        visibility_logits = measurements.auxiliary.get("visibility_logit")
    if visibility_logits is not None:
        outputs["visibility_logits"] = visibility_logits
    targets = {
        "values": aligned,
        "visibility": visibility_target,
    }
    _add_world_position_supervision(
        outputs,
        targets,
        measurements,
        batch,
        frame_index,
        target_indices,
    )
    _add_appearance_supervision(
        outputs,
        targets,
        measurements,
        batch,
        frame_index,
        target_indices,
    )
    masks = {
        # Identity validity, crop evidence, and exact-geometry reliability are
        # deliberately distinct. A valid query whose assigned sphere missed
        # the crop is a real negative for the ROI existence/visibility heads,
        # but the crop cannot supervise colour, appearance, NLL, metric
        # position, or projected geometry.
        "slot_identity": mapped_valid,
        "roi_valid": roi_valid,
        "crop_evidence": crop_evidence,
        "matched": crop_evidence,
        "existence": crop_evidence,
        "existence_valid": roi_valid,
        "visibility_valid": roi_valid,
        "geometry": exact_geometry,
        "world": exact_geometry,
        "world_nll": crop_evidence,
        "colour": crop_evidence,
        "nll": crop_evidence,
        "appearance": crop_evidence,
    }
    losses = module.training_losses(outputs, targets, masks)
    if not losses:
        raise RuntimeError("RGB observation module returned no training losses")
    return losses


@torch.no_grad()
def measurement_localization_metrics(
    measurements: MeasurementSet,
    batch: Mapping[str, Any],
    frame_index: int,
    *,
    birth_confidence_threshold: float,
    distance_threshold_m: float = _MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M,
) -> dict[str, float]:
    """Report calibrated proposal accuracy at the runtime birth threshold.

    The training objective contains differently scaled terms and a Gaussian NLL
    that may legitimately be negative.  Those values are useful optimisation
    diagnostics, but they are not a truthful checkpoint-selection proxy for
    localization.  Recall, precision, and runtime-qualified position error use
    the lifecycle confidence gate that determines whether an unmatched proposal
    can actually become persistent belief state.
    """

    if not 0.0 <= birth_confidence_threshold <= 1.0:
        raise ValueError("birth_confidence_threshold must lie in [0,1]")
    if distance_threshold_m <= 0:
        raise ValueError("distance_threshold_m must be positive")
    target_values, target_mask, _ = _target_measurement_values(batch, frame_index)
    aligned, matched, target_indices = match_measurements_to_targets(
        measurements.values,
        target_values,
        target_mask,
        existence_logits=measurements.existence_logits,
    )
    predicted_world = measurements.auxiliary.get("world_position")
    if not isinstance(predicted_world, Tensor):
        raise ValueError("RGB measurements require auxiliary.world_position")
    target_world = gather_target_slots(
        batch["objects"]["position"][:, frame_index],
        target_indices,
    )
    world_error = torch.linalg.vector_norm(predicted_world - target_world, dim=-1)
    sensor_error = (measurements.values - aligned).abs()
    confident = measurements.measurement_mask & (
        measurements.existence_logits.sigmoid() >= birth_confidence_threshold
    )
    _, runtime_matched, runtime_target_indices = match_measurements_to_targets(
        measurements.values,
        target_values,
        target_mask,
        existence_logits=measurements.existence_logits,
        proposal_mask=confident,
    )
    runtime_target_world = gather_target_slots(
        batch["objects"]["position"][:, frame_index],
        runtime_target_indices,
    )
    runtime_world_error = torch.linalg.vector_norm(
        predicted_world - runtime_target_world,
        dim=-1,
    )
    close = runtime_matched & (runtime_world_error <= distance_threshold_m)
    true_positive = close.sum()
    raw_target_count = target_mask.sum()
    raw_proposal_count = confident.sum()
    target_count = raw_target_count.clamp_min(1)
    proposal_count = raw_proposal_count.clamp_min(1)
    recall = true_positive / target_count
    precision = true_positive / proposal_count
    f1 = (2.0 * recall * precision) / (recall + precision).clamp_min(1.0e-8)
    runtime_world_position_mae = (
        float(runtime_world_error[runtime_matched].mean().cpu())
        if runtime_matched.any()
        else math.inf
    )
    return {
        "rgb_centre_mae_normalized": (
            float(sensor_error[..., :2][matched].mean().cpu()) if matched.any() else math.inf
        ),
        "rgb_inverse_depth_mae": (
            float(sensor_error[..., 3][matched].mean().cpu()) if matched.any() else math.inf
        ),
        "rgb_world_position_mae_m": (
            float(world_error[matched].mean().cpu()) if matched.any() else math.inf
        ),
        # Retain the established aliases, now evaluated with deployment/runtime
        # birth semantics rather than the detector's lower training threshold.
        "rgb_detection_recall_at_0_5m": float(recall.cpu()),
        "rgb_detection_precision_at_0_5m": float(precision.cpu()),
        "rgb_runtime_birth_world_position_mae_m": runtime_world_position_mae,
        "rgb_runtime_birth_recall_at_0_5m": float(recall.cpu()),
        "rgb_runtime_birth_precision_at_0_5m": float(precision.cpu()),
        "rgb_runtime_birth_f1_at_0_5m": float(f1.cpu()),
        "rgb_runtime_birth_true_positive_count_at_0_5m": float(true_positive.cpu()),
        "rgb_runtime_birth_target_count": float(raw_target_count.cpu()),
        "rgb_runtime_birth_proposal_count": float(raw_proposal_count.cpu()),
        "rgb_runtime_birth_world_position_absolute_error_sum_m": float(
            runtime_world_error[runtime_matched].sum().cpu()
        ),
        "rgb_runtime_birth_matched_proposal_count": float(runtime_matched.sum().cpu()),
        "rgb_world_position_absolute_error_sum_m": float(world_error[matched].sum().cpu()),
        "rgb_world_position_matched_proposal_count": float(matched.sum().cpu()),
        "rgb_runtime_birth_confidence_threshold": birth_confidence_threshold,
    }


def _observation_context(
    packet: ObservationPacket,
    config: OrpheusConfig,
    *,
    training: bool,
) -> ObservationContext:
    image = packet.payload
    if not isinstance(image, Tensor):
        raise TypeError("RGB packet payload must be a Tensor")
    return ObservationContext(
        timestamp=packet.timestamp,
        calibration=packet.calibration,
        frame_id=packet.frame_id,
        max_objects=config.model.max_objects,
        device=image.device,
        dtype=image.dtype,
        training=training,
        metadata=packet.metadata,
    )


def _bootstrap_rgb_belief(
    model: OnlineWorldModel,
    packet: ObservationPacket,
    measurements: MeasurementSet,
) -> WorldBelief:
    """Create a training-only provisional belief for paired fast-ROI loss.

    The global proposal remains connected to its direct measurement loss, but
    the detached copy is allocated for one paired pretraining sequence without
    mutating the online runtime or claiming a confirmed object. This deliberate
    bootstrap keeps fast-ROI pretraining supported when runtime lifecycle needs
    multiple global discoveries. No simulator label participates in this state
    construction, and causal training/evaluation use real tentative
    confirmation instead.
    """

    initial_belief = model._initial_belief(packet)
    detached_measurements = measurements.detach()
    belief = model.lifecycle.birth_from_measurements(
        initial_belief,
        detached_measurements,
        detached_measurements.measurement_mask,
        confidence_threshold=model.birth_confidence,
        initial_velocity_variance=model.initial_velocity_variance,
    )
    return belief.with_timestamp(packet.timestamp)


def _distance_gated_anchor_targets(
    belief: WorldBelief,
    batch: Mapping[str, Any],
    frame_index: int,
) -> tuple[Tensor, Tensor]:
    """Align RGB-born tracks to training labels without feeding labels to state."""

    _, target_mask, _ = _target_measurement_values(batch, frame_index)
    target_position = batch["objects"]["position"][:, frame_index]
    target_indices, matched = match_belief_to_targets(
        belief,
        target_position,
        target_mask,
    )
    aligned_position = gather_target_slots(target_position, target_indices)
    matched &= _distance_gate_physical_matches(
        belief.objects.position,
        aligned_position,
        matched,
        threshold_m=_MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M,
    )
    target_indices = torch.where(
        matched,
        target_indices,
        torch.full_like(target_indices, -1),
    )
    return target_indices, matched


def _map_anchor_targets_to_predictions(
    target_indices: Tensor,
    matched: Tensor,
    predicted: PredictedMeasurements,
) -> tuple[Tensor, Tensor]:
    """Map belief-slot supervision into prior-conditioned prediction rows."""

    valid_belief_indices = predicted.belief_indices >= 0
    mapped_indices = (
        gather_target_slots(
            target_indices.unsqueeze(-1),
            predicted.belief_indices,
        )
        .squeeze(-1)
        .to(torch.int64)
    )
    mapped_indices = torch.where(
        valid_belief_indices,
        mapped_indices,
        torch.full_like(mapped_indices, -1),
    )
    mapped_matched = (
        gather_target_slots(
            matched.unsqueeze(-1),
            predicted.belief_indices,
        )
        .squeeze(-1)
        .bool()
        & valid_belief_indices
    )
    return mapped_indices, mapped_matched


def _fast_pair_support(
    batch: Mapping[str, Any],
    frame_index: int,
    predicted: PredictedMeasurements,
    target_indices: Tensor,
    anchor_matched: Tensor,
) -> Tensor:
    """Return identity-linked crops that contain observable target pixels."""

    if predicted.rois is None:
        raise ValueError("paired RGB pretraining requires projected ROIs")
    target_values, target_mask, _ = _target_measurement_values(batch, frame_index)
    aligned_values = gather_target_slots(target_values, target_indices)
    aligned_target_mask = (
        gather_target_slots(
            target_mask.unsqueeze(-1),
            target_indices,
        )
        .squeeze(-1)
        .bool()
    )
    return (
        anchor_matched
        & predicted.valid_mask
        & (target_indices >= 0)
        & aligned_target_mask
        & _target_disc_overlaps_rois(aligned_values, predicted.rois)
    )


def _fast_pair_eligible_slots(
    predicted: PredictedMeasurements,
    target_indices: Tensor,
    anchor_matched: Tensor,
) -> Tensor:
    """Return every runtime-valid prior-conditioned ROI query.

    Target identity is deliberately not part of query eligibility. A valid
    persistent query without a training-only target mapping is a false-positive
    example for ROI confidence, not padding to omit from supervision/selection.
    """

    if target_indices.shape != predicted.valid_mask.shape:
        raise ValueError("fast target indices must match predicted measurement slots")
    if anchor_matched.shape != predicted.valid_mask.shape:
        raise ValueError("fast anchor matches must match predicted measurement slots")
    return predicted.valid_mask


_FAST_PAIR_ADDITIVE_METRICS = (
    "rgb_fast_bootstrap_matched_target_count",
    "rgb_fast_bootstrap_target_count",
    "rgb_fast_roi_supported_target_count",
    "rgb_fast_roi_target_count",
    "rgb_fast_roi_eligible_proposal_count",
    "rgb_fast_roi_world_position_absolute_error_sum_m",
    "rgb_fast_roi_world_position_matched_count",
    "rgb_fast_prior_world_position_absolute_error_sum_m",
    "rgb_fast_prior_world_position_matched_count",
    "rgb_fast_roi_confident_proposal_count",
    "rgb_fast_roi_true_positive_count_at_0_5m",
)


def _derive_fast_pair_metrics(additive: Mapping[str, float]) -> dict[str, float]:
    """Derive handoff ratios only after additive counts have been pooled."""

    metrics = {name: float(additive.get(name, 0.0)) for name in _FAST_PAIR_ADDITIVE_METRICS}
    bootstrap_targets = metrics["rgb_fast_bootstrap_target_count"]
    targets = metrics["rgb_fast_roi_target_count"]
    support = metrics["rgb_fast_roi_world_position_matched_count"]
    prior_support = metrics["rgb_fast_prior_world_position_matched_count"]
    confident = metrics["rgb_fast_roi_confident_proposal_count"]
    true_positive = metrics["rgb_fast_roi_true_positive_count_at_0_5m"]
    measured_mae = (
        metrics["rgb_fast_roi_world_position_absolute_error_sum_m"] / support
        if support > 0
        else math.inf
    )
    prior_mae = (
        metrics["rgb_fast_prior_world_position_absolute_error_sum_m"] / prior_support
        if prior_support > 0
        else math.inf
    )
    recall = true_positive / targets if targets > 0 else 0.0
    precision = true_positive / confident if confident > 0 else 0.0
    metrics.update(
        {
            "rgb_fast_bootstrap_target_coverage": (
                metrics["rgb_fast_bootstrap_matched_target_count"] / bootstrap_targets
                if bootstrap_targets > 0
                else 0.0
            ),
            "rgb_fast_roi_target_coverage": (
                metrics["rgb_fast_roi_supported_target_count"] / targets if targets > 0 else 0.0
            ),
            "rgb_fast_roi_world_position_mae_m": measured_mae,
            "rgb_fast_prior_world_position_mae_m": prior_mae,
            "rgb_fast_roi_improvement_m": (
                prior_mae - measured_mae
                if math.isfinite(prior_mae) and math.isfinite(measured_mae)
                else math.nan
            ),
            "rgb_fast_roi_recall_at_0_5m": recall,
            "rgb_fast_roi_precision_at_0_5m": precision,
            "rgb_fast_roi_f1_at_0_5m": (2.0 * recall * precision / max(recall + precision, 1.0e-8)),
        }
    )
    return metrics


@torch.no_grad()
def _fast_pair_metrics(
    model: OnlineWorldModel,
    measurements: MeasurementSet | None,
    predicted: PredictedMeasurements,
    batch: Mapping[str, Any],
    anchor_frame_index: int,
    frame_index: int,
    target_indices: Tensor,
    bootstrap_matched: Tensor,
    eligible: Tensor,
    support: Tensor,
) -> dict[str, float]:
    """Additive and derived diagnostics for the global-to-fast RGB handoff."""

    _, anchor_target_mask, _ = _target_measurement_values(
        batch,
        anchor_frame_index,
    )
    _, current_target_mask, _ = _target_measurement_values(batch, frame_index)
    bootstrap_target_count = int(anchor_target_mask.sum().detach().cpu())
    bootstrap_matched_count = int(bootstrap_matched.sum().detach().cpu())
    target_count = int(current_target_mask.sum().detach().cpu())
    eligible_count = int(eligible.sum().detach().cpu())
    support_count = int(support.sum().detach().cpu())
    additive: dict[str, float] = {
        "rgb_fast_bootstrap_matched_target_count": float(bootstrap_matched_count),
        "rgb_fast_bootstrap_target_count": float(bootstrap_target_count),
        "rgb_fast_roi_supported_target_count": float(support_count),
        "rgb_fast_roi_target_count": float(target_count),
        "rgb_fast_roi_eligible_proposal_count": float(eligible_count),
        "rgb_fast_roi_world_position_absolute_error_sum_m": 0.0,
        "rgb_fast_roi_world_position_matched_count": float(support_count),
        "rgb_fast_prior_world_position_absolute_error_sum_m": 0.0,
        "rgb_fast_prior_world_position_matched_count": float(support_count),
        "rgb_fast_roi_confident_proposal_count": 0.0,
        "rgb_fast_roi_true_positive_count_at_0_5m": 0.0,
    }
    if measurements is None:
        return _derive_fast_pair_metrics(additive)

    measured_world = measurements.auxiliary.get("world_position")
    prior_world = predicted.auxiliary.get("world_position")
    if not isinstance(measured_world, Tensor) or not isinstance(prior_world, Tensor):
        raise ValueError("paired RGB metrics require measured and predicted world positions")
    target_world = gather_target_slots(
        batch["objects"]["position"][:, frame_index],
        target_indices,
    )
    measured_error = torch.linalg.vector_norm(measured_world - target_world, dim=-1)
    prior_error = torch.linalg.vector_norm(prior_world - target_world, dim=-1)
    confident = (
        eligible
        & measurements.measurement_mask
        & (
            measurements.existence_logits.sigmoid()
            >= model.associator.minimum_measurement_confidence
        )
    )
    close = confident & support & (measured_error <= _MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M)
    confident_count = int(confident.sum().detach().cpu())
    true_positive = int(close.sum().detach().cpu())
    measured_error_sum = (
        float(measured_error[support].sum().detach().cpu()) if support_count else 0.0
    )
    prior_error_sum = float(prior_error[support].sum().detach().cpu()) if support_count else 0.0
    additive.update(
        {
            "rgb_fast_roi_confident_proposal_count": float(confident_count),
            "rgb_fast_roi_true_positive_count_at_0_5m": float(true_positive),
            "rgb_fast_roi_world_position_absolute_error_sum_m": measured_error_sum,
            "rgb_fast_prior_world_position_absolute_error_sum_m": prior_error_sum,
        }
    )
    return _derive_fast_pair_metrics(additive)


def pretrain_rgb_measurements(
    model: OnlineWorldModel,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    frame_index: int,
) -> TrainingBatchResult:
    """Jointly train global discovery and a cached short fast-RGB sequence.

    ``frame_index`` is the discovery anchor and must have a successor when the
    episode contains multiple frames. The anchor RGB frame creates a detached
    runtime belief through normal lifecycle birth. Up to two following frames
    then exercise the real residual ROI encoder, carrying its modality cache
    across the second fast update. A one-frame batch retains the global-only
    path. Simulator state is used only after RGB forwards to align and score
    supervision.
    """

    if "rgb" not in model.observation_modules:
        raise ValueError("RGB measurement pretraining requires the RGB module")
    rgb = batch.get("rgb")
    if not isinstance(rgb, Tensor) or rgb.ndim != 5:
        raise ValueError("RGB measurement pretraining requires batch.rgb [B,T,3,H,W]")
    if not 0 <= frame_index < rgb.shape[1]:
        raise IndexError(frame_index)
    paired_runtime = rgb.shape[1] > 1
    if paired_runtime and frame_index + 1 >= rgb.shape[1]:
        raise IndexError(
            "paired RGB pretraining anchor must have an adjacent successor "
            f"(anchor={frame_index}, total_frames={rgb.shape[1]})"
        )
    anchor_index = frame_index
    current_index = frame_index + 1 if paired_runtime else frame_index
    fast_frame_indices = (
        list(range(frame_index + 1, min(int(rgb.shape[1]), frame_index + 3)))
        if paired_runtime
        else []
    )
    packet = make_rgb_packet(batch, anchor_index)
    module = model.observation_modules["rgb"]
    measurements = module.initialise_measurements(
        [packet],
        _observation_context(packet, config, training=True),
    )
    global_details = supervised_measurement_losses(
        module,
        measurements,
        batch,
        anchor_index,
    )
    global_measurement = _weighted_measurement_total(
        global_details,
        config.training.measurement_loss_weights,
    )
    fast_weight = config.training.fast_roi_pretrain_weight
    measurement = _combine_measurement_objectives(
        global_measurement=global_measurement,
        fast_measurement=None,
        fast_weight=fast_weight,
    )
    metrics = {name: float(value.detach().cpu()) for name, value in global_details.items()}
    metrics.update(
        measurement_localization_metrics(
            measurements,
            batch,
            anchor_index,
            birth_confidence_threshold=config.model.lifecycle.birth_confidence,
        )
    )
    metrics["proposals_above_birth_threshold"] = float(
        (
            measurements.measurement_mask
            & (
                measurements.existence_logits.sigmoid().detach()
                >= config.model.lifecycle.birth_confidence
            )
        )
        .sum()
        .cpu()
    )
    metrics.update(
        {
            "rgb_pretrain_pair_anchor_frame": float(anchor_index),
            "rgb_pretrain_pair_current_frame": float(current_index),
            "rgb_pretrain_fast_frame_count": 0.0,
            "rgb_pretrain_fast_last_frame": float(current_index),
            "fast_path_supervised": 0.0,
            "fast_supervised_slots": 0.0,
            "fast_supervised_frames": 0.0,
        }
    )

    if paired_runtime:
        with torch.no_grad():
            bootstrap = _bootstrap_rgb_belief(model, packet, measurements)
            anchor_target_indices, anchor_matched = _distance_gated_anchor_targets(
                bootstrap,
                batch,
                anchor_index,
            )

        fast_measurement_losses: list[Tensor] = []
        fast_detail_lists: dict[str, list[Tensor]] = {}
        pooled_fast_metrics = {name: 0.0 for name in _FAST_PAIR_ADDITIVE_METRICS}
        fast_cache = None
        supervised_fast_slots = 0
        supervised_fast_frames = 0
        for fast_frame_index in fast_frame_indices:
            current_packet = make_rgb_packet(batch, fast_frame_index)
            with torch.no_grad():
                requested = bootstrap.timestamp.new_full(
                    bootstrap.timestamp.shape,
                    current_packet.timestamp,
                )
                prior = model.dynamics.predict(
                    bootstrap,
                    requested - bootstrap.timestamp,
                )
                predicted = module.project(
                    prior,
                    SensorContext(
                        sensor_id=current_packet.sensor_id,
                        timestamp=current_packet.timestamp,
                        calibration=current_packet.calibration,
                        frame_id=current_packet.frame_id,
                        image_size=current_packet.metadata["image_size"],
                        metadata=current_packet.metadata,
                    ),
                )
                target_indices, mapped_anchor_matched = _map_anchor_targets_to_predictions(
                    anchor_target_indices,
                    anchor_matched,
                    predicted,
                )
                fast_eligible = _fast_pair_eligible_slots(
                    predicted,
                    target_indices,
                    mapped_anchor_matched,
                )
                fast_support = _fast_pair_support(
                    batch,
                    fast_frame_index,
                    predicted,
                    target_indices,
                    mapped_anchor_matched,
                )

            fast_measurements: MeasurementSet | None = None
            if fast_eligible.any():
                fast_measurements, fast_cache = module.encode_measurements(
                    [current_packet],
                    prior,
                    predicted,
                    fast_cache,
                )
                valid_supervision = fast_eligible & fast_measurements.measurement_mask
                if valid_supervision.any():
                    fast_details = supervised_slot_measurement_losses(
                        module,
                        fast_measurements,
                        batch,
                        fast_frame_index,
                        target_indices=target_indices,
                        matched_slots=mapped_anchor_matched,
                        roi_bounds=predicted.rois,
                    )
                    fast_measurement_losses.append(
                        _weighted_measurement_total(
                            fast_details,
                            config.training.measurement_loss_weights,
                        )
                    )
                    for name, value in fast_details.items():
                        fast_detail_lists.setdefault(name, []).append(value)
                    supervised_fast_slots += int(valid_supervision.sum().detach().cpu())
                    supervised_fast_frames += 1
            frame_metrics = _fast_pair_metrics(
                model,
                fast_measurements,
                predicted,
                batch,
                anchor_index,
                fast_frame_index,
                target_indices,
                mapped_anchor_matched,
                fast_eligible,
                fast_support,
            )
            for name in _FAST_PAIR_ADDITIVE_METRICS:
                pooled_fast_metrics[name] += frame_metrics[name]

        if fast_measurement_losses:
            fast_measurement = torch.stack(fast_measurement_losses).mean()
            measurement = _combine_measurement_objectives(
                global_measurement=global_measurement,
                fast_measurement=fast_measurement,
                fast_weight=fast_weight,
            )
            for name, values in fast_detail_lists.items():
                metrics[f"fast_{name}"] = float(torch.stack(values).mean().detach().cpu())
            metrics["fast_path_supervised"] = 1.0
        metrics["rgb_pretrain_fast_frame_count"] = float(len(fast_frame_indices))
        metrics["rgb_pretrain_fast_last_frame"] = float(fast_frame_indices[-1])
        metrics["fast_supervised_slots"] = float(supervised_fast_slots)
        metrics["fast_supervised_frames"] = float(supervised_fast_frames)
        metrics.update(_derive_fast_pair_metrics(pooled_fast_metrics))

    terms = {"measurement": measurement}
    total = weighted_total(terms, config.training.loss_weights)
    return TrainingBatchResult(
        total_loss=total,
        loss_terms=terms,
        metrics=metrics,
        phase="rgb_pretrain",
    )


def match_belief_to_targets(
    belief: WorldBelief,
    target_position: Tensor,
    target_active: Tensor,
) -> tuple[Tensor, Tensor]:
    """Hungarian-match active belief objects to current simulator slots."""

    objects = belief.objects
    if target_position.ndim != 3 or target_position.shape[0] != belief.batch_size:
        raise ValueError("target_position must have shape [B,N,3]")
    if target_active.shape != target_position.shape[:2]:
        raise ValueError("target_active must have shape [B,N]")
    indices = torch.full(
        objects.active.shape,
        -1,
        device=belief.device,
        dtype=torch.int64,
    )
    matched = torch.zeros_like(objects.active)
    for batch_index in range(belief.batch_size):
        belief_slots = torch.nonzero(objects.active[batch_index], as_tuple=False).flatten()
        target_slots = torch.nonzero(target_active[batch_index], as_tuple=False).flatten()
        if belief_slots.numel() == 0 or target_slots.numel() == 0:
            continue
        cost = torch.cdist(
            objects.position[batch_index, belief_slots].detach().cpu(),
            target_position[batch_index, target_slots].detach().cpu(),
        )
        rows, columns = linear_sum_assignment(np.asarray(cost))
        selected_beliefs = belief_slots[
            torch.as_tensor(rows, device=belief.device, dtype=torch.int64)
        ]
        selected_targets = target_slots[
            torch.as_tensor(columns, device=belief.device, dtype=torch.int64)
        ]
        indices[batch_index, selected_beliefs] = selected_targets
        matched[batch_index, selected_beliefs] = True
    return indices, matched


def _update_geometric_identity_metrics(
    belief: WorldBelief,
    target_ids: Tensor,
    target_indices: Tensor,
    matched: Tensor,
    previous_predicted_id_for_target: list[dict[int, int]],
) -> tuple[int, int]:
    """Count track-ID changes under an independent framewise association.

    Training targets deliberately use :class:`PersistentTargetMatcher` so an
    identity swap is punished as state error instead of silently relabelling
    supervision.  Reusing that locked mapping for the ID-switch metric,
    however, made a geometric swap look like zero switches (usually only
    coverage fell).  This helper consumes a fresh framewise Hungarian match
    and compares the persistent runtime IDs assigned to each simulator target.
    """

    objects = belief.objects
    if target_ids.ndim != 2 or target_ids.shape[0] != belief.batch_size:
        raise ValueError("target_ids must have shape [B,N]")
    if target_indices.shape != objects.active.shape or matched.shape != objects.active.shape:
        raise ValueError("target_indices and matched must have belief-slot shape [B,N]")
    if matched.dtype is not torch.bool:
        raise TypeError("matched must be torch.bool")
    if len(previous_predicted_id_for_target) != belief.batch_size:
        raise ValueError("identity history must contain one mapping per batch item")

    switches = 0
    associations = 0
    for batch_index in range(belief.batch_size):
        for belief_slot_tensor in torch.nonzero(
            matched[batch_index],
            as_tuple=False,
        ).flatten():
            belief_slot = int(belief_slot_tensor)
            target_slot = int(target_indices[batch_index, belief_slot])
            target_id = int(target_ids[batch_index, target_slot])
            predicted_id = int(objects.object_id[batch_index, belief_slot].detach().cpu())
            if target_id < 0 or predicted_id < 0:
                continue
            associations += 1
            previous = previous_predicted_id_for_target[batch_index].get(target_id)
            if previous is not None and previous != predicted_id:
                switches += 1
            previous_predicted_id_for_target[batch_index][target_id] = predicted_id
    return switches, associations


def _mean_losses(losses: list[Tensor], reference: Tensor) -> Tensor:
    if not losses:
        return reference.sum() * 0
    return torch.stack(losses).mean()


def _weighted_measurement_total(
    losses: Mapping[str, Tensor],
    weights: Mapping[str, float],
) -> Tensor:
    """Combine RGB supervision without letting heteroscedastic NLL dominate.

    Unknown future diagnostic terms retain unit weight.  Explicit weights make
    the metric localization objective materially stronger while preserving
    uncertainty NLL as a bounded calibration signal.
    """

    if not losses:
        raise ValueError("measurement loss mapping cannot be empty")
    return torch.stack(
        [value * float(weights.get(name, 1.0)) for name, value in losses.items()]
    ).sum()


def _combine_measurement_objectives(
    *,
    global_measurement: Tensor | None,
    fast_measurement: Tensor | None,
    fast_weight: float,
) -> Tensor:
    """Combine available RGB branches without support-dependent reweighting.

    ``fast_weight`` declares fixed relative coefficients ``1:fast_weight``.
    A temporarily absent or frozen branch contributes zero; it does not cause
    the remaining branch to inherit the missing branch's coefficient.
    """

    if global_measurement is None and fast_measurement is None:
        raise ValueError("at least one RGB measurement objective is required")
    if not math.isfinite(fast_weight) or fast_weight <= 0.0:
        raise ValueError("fast measurement weight must be finite and positive")
    denominator = 1.0 + fast_weight
    if global_measurement is None:
        assert fast_measurement is not None
        return fast_weight * fast_measurement / denominator
    if fast_measurement is None:
        return global_measurement / denominator
    return (global_measurement + fast_weight * fast_measurement) / denominator


def _global_measurement_has_trainable_path(module: torch.nn.Module) -> bool:
    """Return whether global-discovery supervision can update any parameter.

    The RGB backbone's ``fast_projection`` belongs exclusively to the ROI
    path.  Looking at every parameter under ``backbone`` therefore produces a
    false positive for the frozen-shared-backbone training scope: the global
    loss is constant with respect to every trainable tensor even though the
    ROI projection remains trainable.
    """

    backbone = getattr(module, "backbone", None)
    global_detector = getattr(module, "global_detector", None)
    if backbone is None or global_detector is None:
        raise TypeError("RGB module is missing backbone or global_detector")
    dense_global_detector = getattr(module, "dense_global_detector", None)
    global_modules = (
        global_detector,
        *((dense_global_detector,) if dense_global_detector is not None else ()),
        *backbone.stages,
        *backbone.projections,
    )
    return any(
        parameter.requires_grad
        for component in global_modules
        for parameter in component.parameters()
    )


def _fast_measurement_has_trainable_perception_path(module: torch.nn.Module) -> bool:
    """Return whether fast-ROI supervision can update RGB perception.

    A fast residual is conditioned on the propagated prior.  When RGB
    perception is frozen, the resulting loss can still require gradients via
    that prior and therefore accidentally train dynamics or the belief
    corrector.  Measurement supervision is a perception-local auxiliary: the
    physical stack has its own current-state, rollout, event, uncertainty, and
    correction objectives.  Treat only the shared fast encoder, ROI-only
    projection, and ROI updater as valid trainable paths for this branch.
    """

    backbone = getattr(module, "backbone", None)
    roi_updater = getattr(module, "roi_updater", None)
    if backbone is None or roi_updater is None:
        raise TypeError("RGB module is missing backbone or roi_updater")
    fast_modules = (
        *backbone.stages[:2],
        backbone.fast_projection,
        roi_updater,
    )
    return any(
        parameter.requires_grad
        for component in fast_modules
        for parameter in component.parameters()
    )


def _valid_rollout_offsets(
    config: OrpheusConfig,
    frame_index: int,
    total_frames: int,
) -> tuple[list[int], list[float], list[float]]:
    frame_rate = config.simulator.frame_rate
    offsets: list[int] = []
    seconds: list[float] = []
    weights: list[float] = []
    for horizon, weight in zip(
        config.evaluation.horizons_seconds,
        config.training.horizon_weights,
        strict=True,
    ):
        frame_offset = max(1, int(round(float(horizon) * frame_rate)))
        if frame_index + frame_offset >= total_frames:
            continue
        if frame_offset in offsets:
            continue
        offsets.append(frame_offset)
        seconds.append(frame_offset / frame_rate)
        weights.append(float(weight))
    ordering = sorted(range(len(offsets)), key=offsets.__getitem__)
    return (
        [offsets[index] for index in ordering],
        [seconds[index] for index in ordering],
        [weights[index] for index in ordering],
    )


def _select_rollout_anchor_frames(
    config: OrpheusConfig,
    *,
    window_start: int,
    window_stop: int,
    total_frames: int,
    rollout_anchors_per_window: int | None,
) -> tuple[int, ...]:
    """Select deterministic, horizon-capable rollout anchors.

    Every frame is still ingested and receives current-state supervision.
    Bounding anchors only avoids repeated expensive recursive rollouts. The
    earliest eligible frame is retained because it supports the widest horizon
    set; additional anchors are spread across the remaining causal window.
    """

    if rollout_anchors_per_window is not None and rollout_anchors_per_window <= 0:
        raise ValueError("rollout_anchors_per_window must be positive or null")
    candidates = [
        frame_index
        for frame_index in range(window_start, window_stop)
        if frame_index >= config.training.minimum_rollout_age_steps
        if _valid_rollout_offsets(config, frame_index, total_frames)[0]
    ]
    if rollout_anchors_per_window is None or rollout_anchors_per_window >= len(candidates):
        return tuple(candidates)
    if rollout_anchors_per_window == 1:
        return (candidates[0],)
    last = len(candidates) - 1
    selected_indices = [
        round(anchor_index * last / (rollout_anchors_per_window - 1))
        for anchor_index in range(rollout_anchors_per_window)
    ]
    return tuple(candidates[index] for index in selected_indices)


def rollout_horizon_loss_key(name: str, seconds: float) -> str:
    """Return the stable metric key for one physical rollout horizon."""

    return f"{name}@{seconds:.3f}s"


def future_scene_predictable_mask(
    batch: Mapping[str, Any],
    *,
    anchor_index: int,
    target_index: int,
) -> Tensor:
    """Return one causal deterministic-support marker per batch scene."""

    events = batch.get("events")
    if not isinstance(events, Mapping):
        raise ValueError("closed-loop training requires batch.events")
    externally_actuated = events.get("externally_actuated")
    if not isinstance(externally_actuated, Tensor):
        objects = batch.get("objects")
        active = objects.get("active") if isinstance(objects, Mapping) else None
        if not isinstance(active, Tensor) or active.ndim != 3:
            raise ValueError(
                "a batch without events.externally_actuated requires objects.active [B,T,N]"
            )
        return torch.ones(
            active.shape[0],
            dtype=torch.bool,
            device=active.device,
        )
    if externally_actuated.ndim != 3:
        raise ValueError("events.externally_actuated must have shape [B,T,N]")
    if not 0 <= anchor_index < target_index < externally_actuated.shape[1]:
        raise ValueError("forecast anchor/target indices are outside the event sequence")
    # Dynamics are coupled: an unobserved impulse on one object can change any
    # other object's target through a subsequent interaction. Censor the
    # complete scene until the next observation has exposed the actuation.
    scene_intervened = (
        externally_actuated[
            :,
            anchor_index + 1 : target_index + 1,
        ]
        .bool()
        .flatten(start_dim=1)
        .any(dim=1)
    )
    return ~scene_intervened


def future_predictable_mask(
    batch: Mapping[str, Any],
    *,
    anchor_index: int,
    target_index: int,
    target_indices: Tensor,
) -> Tensor:
    """Return target-aligned support before any unseen external actuation.

    Random future impulses are intentionally not present in RGB at the anchor.
    Point losses after such an intervention have no deterministic target and
    teach only an average of mutually incompatible futures.
    """

    scene_predictable = future_scene_predictable_mask(
        batch,
        anchor_index=anchor_index,
        target_index=target_index,
    )
    return scene_predictable[:, None].expand_as(target_indices)


# Preserve the internal name used by focused regression tests and downstream
# research code while exposing one shared public evaluator/trainer contract.
_future_predictable_mask = future_predictable_mask


def _globally_weight_horizon_details(
    details: dict[str, Tensor],
    config: OrpheusConfig,
    reference: Tensor,
) -> dict[str, Tensor]:
    """Aggregate each horizon after averaging only its eligible anchors.

    A short horizon is available at more episode anchors than a long one.
    Normalising horizon weights independently at every anchor therefore lets
    numerous late short-only anchors overwhelm the configured long-horizon
    weight. Per-horizon means make ``training.horizon_weights`` describe the
    intended global objective.
    """

    output = dict(details)
    unique_horizons: list[tuple[float, float]] = []
    seen_offsets: set[int] = set()
    for horizon, weight in zip(
        config.evaluation.horizons_seconds,
        config.training.horizon_weights,
        strict=True,
    ):
        frame_offset = max(1, int(round(float(horizon) * config.simulator.frame_rate)))
        if frame_offset in seen_offsets:
            continue
        seen_offsets.add(frame_offset)
        unique_horizons.append((frame_offset / config.simulator.frame_rate, float(weight)))
    configured_weight_total = sum(weight for _, weight in unique_horizons)

    globally_normalized_names = [
        "rollout_position",
        "rollout_velocity",
        "rollout_position_nll",
        "event_collision",
        "event_collision_node",
        "event_collision_pair",
        "correction_future",
        "correction_future_velocity",
    ]
    if config.training.normalize_rollout_axes_over_configured_horizons:
        globally_normalized_names.extend(
            (
                "rollout_position_x",
                "rollout_position_y",
                "rollout_position_z",
            )
        )
    for name in globally_normalized_names:
        values: list[Tensor] = []
        weights: list[float] = []
        for seconds, weight in unique_horizons:
            key = rollout_horizon_loss_key(name, seconds)
            if key in details:
                values.append(details[key])
                weights.append(weight)
        if values:
            horizon_weights = reference.new_tensor(weights)
            output[name] = (torch.stack(values) * horizon_weights).sum() / max(
                configured_weight_total, 1.0e-8
            )
    return output


@dataclass(frozen=True)
class _ParameterSupervisionMasks:
    """Causal, observation-supported slow-parameter supervision.

    These masks are expressed in persistent belief-slot order.  Simulator
    events decide which labelled parameter could explain an interval, but an
    event is eligible only when the RGB runtime actually observed the required
    pre/post states.  The masks are never passed into the runtime.
    """

    drag: Tensor
    restitution: Tensor
    runtime_observed: Tensor
    temporal_baseline: Tensor
    pair_restitution: Tensor
    boundary_restitution: Tensor
    drag_speed_only_rejected: Tensor
    pair_higher_restitution_rejected: Tensor

    def detached_metrics(self) -> dict[str, float]:
        names_and_masks = (
            ("parameter_runtime_observed_object_count", self.runtime_observed),
            ("parameter_temporal_baseline_object_count", self.temporal_baseline),
            ("parameter_drag_observable_object_count", self.drag),
            ("parameter_restitution_observable_object_count", self.restitution),
            (
                "parameter_restitution_pair_observable_object_count",
                self.pair_restitution,
            ),
            (
                "parameter_restitution_boundary_observable_object_count",
                self.boundary_restitution,
            ),
            (
                "parameter_drag_speed_only_rejected_object_count",
                self.drag_speed_only_rejected,
            ),
            (
                "parameter_restitution_pair_higher_rejected_object_count",
                self.pair_higher_restitution_rejected,
            ),
        )
        counts = (
            torch.stack([mask.sum() for _, mask in names_and_masks])
            .detach()
            .cpu()
            .to(dtype=torch.float64)
            .tolist()
        )
        return {
            name: float(count) for (name, _), count in zip(names_and_masks, counts, strict=True)
        }


def _runtime_observed_belief_slots(
    model: OnlineWorldModel,
    belief: WorldBelief,
) -> Tensor:
    """Return slots supported by the RGB packet that was just ingested.

    Slow-parameter updates consume the identifier's accepted-association
    features. A lifecycle birth is genuine RGB evidence for fast state, but
    it has no association/innovation path through the identifier on that same
    frame and therefore cannot open a slow-parameter supervision gate.
    """

    diagnostics = model.updater.last_diagnostics
    associated = (
        diagnostics.observed_mask
        if diagnostics is not None
        else torch.zeros_like(belief.objects.active)
    )
    if associated.shape != belief.objects.active.shape or associated.dtype is not torch.bool:
        raise ValueError("runtime observed mask must be boolean belief-slot [B,N]")
    return associated & belief.objects.active


def _target_observation_mask(
    indices: Tensor,
    matched: Tensor,
    runtime_observed: Tensor,
    *,
    target_count: int,
) -> Tensor:
    """Map actual runtime observations into stable simulator target slots."""

    if indices.shape != matched.shape or indices.shape != runtime_observed.shape:
        raise ValueError("parameter observation masks must share belief-slot shape [B,N]")
    if matched.dtype is not torch.bool or runtime_observed.dtype is not torch.bool:
        raise TypeError("parameter matched/observed masks must be torch.bool")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    safe_indices = indices.clamp(0, target_count - 1)
    one_hot_targets = F.one_hot(safe_indices, num_classes=target_count).bool()
    return (one_hot_targets & (matched & runtime_observed).unsqueeze(-1)).any(dim=1)


def _target_observed_runtime_ids(
    indices: Tensor,
    matched: Tensor,
    runtime_observed: Tensor,
    runtime_object_ids: Tensor,
    *,
    target_count: int,
) -> Tensor:
    """Map accepted observations to the persistent runtime ID for each target."""

    if (
        indices.shape != matched.shape
        or indices.shape != runtime_observed.shape
        or indices.shape != runtime_object_ids.shape
    ):
        raise ValueError("parameter identity tensors must share belief-slot shape [B,N]")
    if matched.dtype is not torch.bool or runtime_observed.dtype is not torch.bool:
        raise TypeError("parameter matched/observed masks must be torch.bool")
    if runtime_object_ids.dtype is not torch.int64:
        raise TypeError("runtime object IDs must be int64")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    safe_indices = indices.clamp(0, target_count - 1)
    assignment = F.one_hot(safe_indices, num_classes=target_count).bool()
    valid = matched & runtime_observed & (runtime_object_ids >= 0)
    candidate_ids = torch.where(
        assignment & valid.unsqueeze(-1),
        runtime_object_ids.unsqueeze(-1),
        torch.full_like(runtime_object_ids.unsqueeze(-1), -1),
    )
    if candidate_ids.device.type == "mps":
        # MPS cannot reliably compile the NaN-propagating backward companion
        # of this integer ``amax`` while the complete differentiable training
        # loop is present.  Target counts are deliberately small; a sequential
        # elementwise maximum is exactly the same reduction (including the
        # duplicate-candidate case) without invoking that reduction kernel.
        pooled = torch.full_like(candidate_ids[:, 0], -1)
        for belief_slot in range(candidate_ids.shape[1]):
            pooled = torch.maximum(pooled, candidate_ids[:, belief_slot])
        return pooled
    return candidate_ids.amax(dim=1)


def _reset_parameter_history_for_identity_change(
    last_observed_target_frame: Tensor,
    last_observed_runtime_id: Tensor,
    observed_runtime_id: Tensor,
) -> tuple[Tensor, Tensor]:
    """Invalidate a slow-parameter baseline when a target receives a new ID."""

    if (
        last_observed_target_frame.shape != last_observed_runtime_id.shape
        or last_observed_target_frame.shape != observed_runtime_id.shape
    ):
        raise ValueError("parameter observation histories must share target shape [B,N]")
    if (
        last_observed_target_frame.dtype is not torch.int64
        or last_observed_runtime_id.dtype is not torch.int64
        or observed_runtime_id.dtype is not torch.int64
    ):
        raise TypeError("parameter observation histories must use int64")
    identity_changed = (
        (observed_runtime_id >= 0)
        & (last_observed_runtime_id >= 0)
        & (observed_runtime_id != last_observed_runtime_id)
    )
    return (
        torch.where(
            identity_changed,
            torch.full_like(last_observed_target_frame, -1),
            last_observed_target_frame,
        ),
        identity_changed,
    )


def _parameter_supervision_masks(
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    frame_index: int,
    *,
    indices: Tensor,
    matched: Tensor,
    runtime_observed: Tensor,
    last_observed_target_frame: Tensor,
) -> tuple[_ParameterSupervisionMasks, Tensor]:
    """Build causal drag/restitution masks and advance observation history.

    Drag is identifiable only from a genuine temporal baseline with no
    contact, collision, external actuation, or lifecycle gap in between.
    Restitution needs observed states on both sides of an impact.  A boundary
    impact identifies the object's own restitution.  A sphere pair uses
    ``min(e_i, e_j)`` in both simulator and analytic dynamics, so only a
    minimum-restitution partner is labelled by that pair.
    """

    objects = batch["objects"]
    events = batch["events"]
    target_active = objects["active"]
    target_velocity = objects["velocity"]
    target_restitution = objects["restitution"]
    if target_active.ndim != 3:
        raise ValueError("objects.active must have shape [B,T,N]")
    batch_size, total_frames, target_count = target_active.shape
    expected_target_shape = (batch_size, target_count)
    if last_observed_target_frame.shape != expected_target_shape:
        raise ValueError("last observed target frame must have shape [B,N]")
    if last_observed_target_frame.dtype is not torch.int64:
        raise TypeError("last observed target frame must be torch.int64")
    if not 0 <= frame_index < total_frames:
        raise IndexError(frame_index)
    if target_velocity.shape != (batch_size, total_frames, target_count, 3):
        raise ValueError("objects.velocity must have shape [B,T,N,3]")
    if target_restitution.shape != (batch_size, total_frames, target_count, 1):
        raise ValueError("objects.restitution must have shape [B,T,N,1]")

    def event_tensor(name: str, suffix: tuple[int, ...]) -> Tensor:
        value = events.get(name)
        expected = (batch_size, total_frames, target_count, *suffix)
        if (
            not isinstance(value, Tensor)
            or value.shape != expected
            or value.dtype is not torch.bool
        ):
            raise ValueError(f"events.{name} must be boolean {list(expected)}")
        return value

    collision = event_tensor("collision", ())
    contact = event_tensor("contact", ())
    externally_actuated = event_tensor("externally_actuated", ())
    pair_collision = event_tensor("pair_collision", (target_count,))
    boundary_collision = events.get("boundary_collision")
    if (
        not isinstance(boundary_collision, Tensor)
        or boundary_collision.ndim != 4
        or boundary_collision.shape[:3] != (batch_size, total_frames, target_count)
        or boundary_collision.dtype is not torch.bool
    ):
        raise ValueError("events.boundary_collision must be boolean [B,T,N,P]")

    target_observed = _target_observation_mask(
        indices,
        matched,
        runtime_observed,
        target_count=target_count,
    )
    previous_frame = last_observed_target_frame
    temporal_target = target_observed & (previous_frame >= 0) & (previous_frame < frame_index)
    time = torch.arange(
        total_frames,
        device=last_observed_target_frame.device,
        dtype=last_observed_target_frame.dtype,
    )[None, :, None]
    causal_interval = (time > previous_frame[:, None, :]) & (time <= frame_index)
    collision_since_observation = (collision & causal_interval).any(dim=1)
    contact_since_observation = (contact & causal_interval).any(dim=1)
    actuation_since_observation = (externally_actuated & causal_interval).any(dim=1)
    inactive_since_observation = ((~target_active.bool()) & causal_interval).any(dim=1)

    speed = torch.linalg.vector_norm(target_velocity[:, frame_index], dim=-1)
    speed_supported = speed >= config.model.identification.drag_speed_threshold
    drag_target = (
        temporal_target
        & speed_supported
        & ~collision_since_observation
        & ~contact_since_observation
        & ~actuation_since_observation
        & ~inactive_since_observation
    )

    boundary_since_observation = (boundary_collision.any(dim=-1) & causal_interval).any(dim=1)
    boundary_target = (
        temporal_target
        & boundary_since_observation
        & ~actuation_since_observation
        & ~inactive_since_observation
    )

    pair_time = time.unsqueeze(-1)
    pair_pre_observed = (
        (pair_time > previous_frame[:, None, :, None])
        & (pair_time > previous_frame[:, None, None, :])
        & (pair_time <= frame_index)
    )
    current_pair_observed = temporal_target[:, None, :, None] & temporal_target[:, None, None, :]
    pair_clean = (
        ~actuation_since_observation[:, None, :, None]
        & ~actuation_since_observation[:, None, None, :]
        & ~inactive_since_observation[:, None, :, None]
        & ~inactive_since_observation[:, None, None, :]
    )
    causally_supported_pair = (
        pair_collision & pair_pre_observed & current_pair_observed & pair_clean
    )
    restitution = target_restitution[:, frame_index, :, 0]
    pair_minimum = restitution[:, :, None] <= restitution[:, None, :]
    pair_target = (causally_supported_pair & pair_minimum[:, None, :, :]).any(dim=(1, 3))
    pair_higher_rejected_target = (causally_supported_pair & ~pair_minimum[:, None, :, :]).any(
        dim=(1, 3)
    )
    restitution_target = boundary_target | pair_target

    def align_target_mask(mask: Tensor) -> Tensor:
        return gather_target_slots(mask.unsqueeze(-1), indices).squeeze(-1).bool() & matched

    masks = _ParameterSupervisionMasks(
        drag=align_target_mask(drag_target),
        restitution=align_target_mask(restitution_target),
        runtime_observed=runtime_observed & matched,
        temporal_baseline=align_target_mask(temporal_target),
        pair_restitution=align_target_mask(pair_target),
        boundary_restitution=align_target_mask(boundary_target),
        drag_speed_only_rejected=align_target_mask(
            target_observed & speed_supported & ~drag_target
        ),
        pair_higher_restitution_rejected=align_target_mask(pair_higher_rejected_target),
    )
    updated_last_observed = torch.where(
        target_observed,
        torch.full_like(last_observed_target_frame, frame_index),
        last_observed_target_frame,
    )
    return masks, updated_last_observed


def _belief_state_losses(
    belief: WorldBelief,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    frame_index: int,
    *,
    indices: Tensor | None = None,
    matched: Tensor | None = None,
    parameter_supervision: _ParameterSupervisionMasks | None = None,
    velocity_axis_support: Tensor | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    objects = belief.objects
    target_objects = batch["objects"]
    target_position = target_objects["position"][:, frame_index]
    target_active = target_objects["active"][:, frame_index].bool()
    if (indices is None) != (matched is None):
        raise ValueError("indices and matched must be supplied together")
    if indices is None or matched is None:
        indices, matched = match_belief_to_targets(
            belief,
            target_position,
            target_active,
        )
    aligned_position = gather_target_slots(target_position, indices)
    aligned_velocity = gather_target_slots(target_objects["velocity"][:, frame_index], indices)
    batch_macro = config.training.closed_loop_batch_macro_physical_losses_enabled
    scenario_tail_fraction = config.training.closed_loop_scenario_tail_fraction
    state_position = _axis_separable_masked_huber(
        objects.position,
        aligned_position,
        matched,
        batch_macro=batch_macro,
        batch_tail_fraction=scenario_tail_fraction,
    )
    if velocity_axis_support is None:
        velocity_axis_support = _state_velocity_objective_axis_support(
            belief,
            matched,
        )
    elif (
        velocity_axis_support.shape != objects.velocity.shape
        or velocity_axis_support.dtype is not torch.bool
    ):
        raise ValueError("velocity axis support must be boolean belief-slot coordinates [B,N,3]")
    elif (velocity_axis_support & ~_state_velocity_objective_axis_support(belief, matched)).any():
        raise ValueError("velocity axis support cannot include a newborn or unmatched slot")
    uncertainty = _axis_separable_gaussian_nll(
        objects.position,
        aligned_position,
        objects.fast_log_variance[..., :3],
        matched,
        detach_mean_error=True,
        batch_macro=batch_macro,
        batch_tail_fraction=scenario_tail_fraction,
        standardized_error_gradient_cap=(
            config.training.closed_loop_uncertainty_standardized_error_gradient_cap
        ),
    )
    # Inactive factory padding is not a causal prediction.  Its zero logits
    # previously contributed BCE(0, 0) on every empty frame, creating a
    # constant 0.693 objective even when no trainable path had any support.
    # Active false-positive tracks remain useful negative existence examples.
    active_prediction = objects.active.bool()
    existence_target = matched.to(objects.existence_logit.dtype)

    if parameter_supervision is not None:
        for name, mask in (
            ("drag", parameter_supervision.drag),
            ("restitution", parameter_supervision.restitution),
        ):
            if mask.shape != matched.shape or mask.dtype is not torch.bool:
                raise ValueError(f"parameter {name} supervision must be boolean belief-slot [B,N]")
    drag_observable = (
        parameter_supervision.drag
        if parameter_supervision is not None
        else torch.zeros_like(matched)
    )
    restitution_observable = (
        parameter_supervision.restitution
        if parameter_supervision is not None
        else torch.zeros_like(matched)
    )
    aligned_drag = gather_target_slots(target_objects["drag"][:, frame_index], indices)
    aligned_restitution = gather_target_slots(
        target_objects["restitution"][:, frame_index], indices
    )
    losses: dict[str, Tensor] = {}
    if active_prediction.any():
        losses["existence_belief"] = F.binary_cross_entropy_with_logits(
            objects.existence_logit[active_prediction],
            existence_target[active_prediction],
        )
    # Omit unsupported objectives instead of appending differentiable zeros.
    # Window-level averaging then normalizes over informative frames rather
    # than diluting rare state/parameter evidence by the TBPTT length.
    if matched.any():
        losses.update(
            {
                "state_position": state_position,
                "uncertainty_position_nll": uncertainty,
            }
        )
    if velocity_axis_support.any():
        losses["state_velocity"] = _axis_separable_masked_huber(
            objects.velocity,
            aligned_velocity,
            velocity_axis_support,
            batch_macro=batch_macro,
            batch_tail_fraction=scenario_tail_fraction,
        )
    if drag_observable.any():
        losses["parameter_drag"] = masked_huber(
            objects.drag,
            aligned_drag,
            drag_observable,
        )
    if restitution_observable.any():
        losses["parameter_restitution"] = masked_huber(
            objects.restitution,
            aligned_restitution,
            restitution_observable,
        )
    return (
        losses,
        indices,
        matched,
    )


def _coordinate_mask(mask: Tensor, value: Tensor, coordinate: int) -> Tensor:
    """Return a mask for one final-axis coordinate without changing support."""

    if mask.shape == value.shape:
        return mask[..., coordinate]
    if mask.shape == value.shape[:-1]:
        return mask
    raise ValueError("physical mask must match values or omit only their coordinate axis")


def _protected_objective_cell(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> ProtectedObjectiveCell:
    """Return batch-row additive Smooth-L1 evidence without reducing rows."""

    if prediction.shape != target.shape or prediction.ndim < 1:
        raise ValueError("protected objective predictions and targets must match")
    expanded = mask
    while expanded.ndim < prediction.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(prediction)
    flat_mask = expanded.reshape(prediction.shape[0], -1)
    flat_error = F.smooth_l1_loss(prediction, target, reduction="none").reshape(
        prediction.shape[0], -1
    )
    return ProtectedObjectiveCell(
        error_sum=torch.where(
            flat_mask,
            flat_error,
            torch.zeros_like(flat_error),
        ).sum(dim=-1),
        coordinate_count=flat_mask.sum(dim=-1),
    )


def _protected_event_objective_cell(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
) -> ProtectedObjectiveCell:
    """Return unweighted per-row event BCE for reference non-regression."""

    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError("protected event logits, targets, and mask must match")
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target.to(logits.dtype),
        reduction="none",
    )
    flat_mask = mask.reshape(mask.shape[0], -1)
    flat_error = elementwise.reshape(elementwise.shape[0], -1)
    return ProtectedObjectiveCell(
        error_sum=torch.where(
            flat_mask,
            flat_error,
            torch.zeros_like(flat_error),
        ).sum(dim=-1),
        coordinate_count=flat_mask.sum(dim=-1),
    )


def _merge_protected_objective_cells(
    cells: list[ProtectedObjectiveCell],
) -> ProtectedObjectiveCell:
    if not cells:
        raise ValueError("protected objective cell aggregation requires evidence")
    shape = cells[0].error_sum.shape
    if any(cell.error_sum.shape != shape or cell.coordinate_count.shape != shape for cell in cells):
        raise ValueError("protected objective cell rows must have identical shapes")
    return ProtectedObjectiveCell(
        error_sum=torch.stack([cell.error_sum for cell in cells]).sum(dim=0),
        coordinate_count=torch.stack([cell.coordinate_count for cell in cells]).sum(dim=0),
    )


def protected_reference_nonregression_loss(
    candidate: Mapping[str, ProtectedObjectiveCell],
    reference: Mapping[str, ProtectedObjectiveCell],
) -> tuple[Tensor, dict[str, float]]:
    """Penalise any scenario/axis/horizon error above a frozen reference.

    Support is part of the contract.  A candidate may not satisfy the hinge by
    dropping a track or horizon that the protected model scored.
    """

    if not candidate or candidate.keys() != reference.keys():
        raise ValueError(
            "protected candidate/reference objective schemas must match and be nonempty"
        )
    excesses: list[Tensor] = []
    supported_cells = 0
    for name in sorted(candidate):
        candidate_cell = candidate[name]
        reference_cell = reference[name]
        if (
            candidate_cell.error_sum.ndim != 1
            or candidate_cell.error_sum.shape != reference_cell.error_sum.shape
            or candidate_cell.coordinate_count.shape != candidate_cell.error_sum.shape
            or reference_cell.coordinate_count.shape != candidate_cell.error_sum.shape
        ):
            raise ValueError(f"protected objective cell {name!r} must be batch-row vectors")
        if not torch.equal(
            candidate_cell.coordinate_count.detach(),
            reference_cell.coordinate_count.detach(),
        ):
            raise ValueError(
                f"protected objective support changed for {name!r}; refusing a fail-open update"
            )
        support = candidate_cell.coordinate_count > 0
        if support.any():
            candidate_mean = candidate_cell.error_sum / candidate_cell.coordinate_count.clamp_min(
                1
            ).to(candidate_cell.error_sum.dtype)
            reference_mean = reference_cell.error_sum / reference_cell.coordinate_count.clamp_min(
                1
            ).to(reference_cell.error_sum.dtype)
            excesses.append(F.relu(candidate_mean - reference_mean.detach()).masked_select(support))
            supported_cells += int(support.sum().detach().cpu())
    if not excesses:
        raise ValueError("protected objective has no supported scenario cells")
    concatenated = torch.cat(excesses)
    return concatenated.mean(), {
        "protected_reference_supported_scenario_cell_count": float(supported_cells),
        "protected_reference_regressed_scenario_cell_count": float(
            (concatenated.detach() > 0).sum().cpu()
        ),
        "protected_reference_maximum_error_excess": float(concatenated.detach().max().cpu()),
    }


def _axis_separable_masked_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_macro: bool,
    batch_tail_fraction: float | None,
) -> Tensor:
    """Apply scenario-tail reduction independently to each physical axis."""

    if batch_tail_fraction is None:
        return masked_huber(
            prediction,
            target,
            mask,
            batch_macro=batch_macro,
        )
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("axis-separable physical values must share a coordinate axis")
    axis_losses = [
        masked_huber(
            prediction[..., coordinate],
            target[..., coordinate],
            _coordinate_mask(mask, prediction, coordinate),
            batch_macro=True,
            batch_tail_fraction=batch_tail_fraction,
        )
        for coordinate in range(prediction.shape[-1])
    ]
    return torch.stack(axis_losses).mean()


def _axis_separable_gaussian_nll(
    mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
    *,
    detach_mean_error: bool,
    batch_macro: bool,
    batch_tail_fraction: float | None,
    standardized_error_gradient_cap: float | None = None,
) -> Tensor:
    """Apply scenario-tail NLL independently to each physical axis."""

    if batch_tail_fraction is None:
        return gaussian_nll(
            mean,
            target,
            log_variance,
            mask,
            detach_mean_error=detach_mean_error,
            batch_macro=batch_macro,
            standardized_error_gradient_cap=standardized_error_gradient_cap,
        )
    if mean.shape != target.shape or mean.shape != log_variance.shape or mean.ndim < 2:
        raise ValueError("axis-separable Gaussian values must have matching shapes")
    axis_losses = [
        gaussian_nll(
            mean[..., coordinate],
            target[..., coordinate],
            log_variance[..., coordinate],
            _coordinate_mask(mask, mean, coordinate),
            detach_mean_error=detach_mean_error,
            batch_macro=True,
            batch_tail_fraction=batch_tail_fraction,
            standardized_error_gradient_cap=standardized_error_gradient_cap,
        )
        for coordinate in range(mean.shape[-1])
    ]
    return torch.stack(axis_losses).mean()


def _state_velocity_objective_axis_support(
    belief: WorldBelief,
    matched: Tensor,
) -> Tensor:
    """Return causal coordinate support for current velocity supervision.

    RGB discovery births initialise velocity to a hard zero and have no prior
    state, temporal baseline, or position innovation from which a trainable
    velocity correction could be computed on that same frame.  Once a track
    has survived one predict-observe cycle, analytic/learned dynamics and the
    position-to-velocity correction path make every velocity coordinate a
    causal training target.  The coordinate-shaped mask keeps that contract
    explicit without incorrectly requiring direct temporal velocity evidence.
    """

    objects = belief.objects
    if matched.shape != objects.active.shape or matched.dtype is not torch.bool:
        raise ValueError("matched velocity support must be boolean belief-slot [B,N]")
    supported_slots = matched & objects.active.bool() & (objects.age_steps > 0)
    return supported_slots.unsqueeze(-1).expand_as(objects.velocity)


def _current_correction_objective_support(
    belief: WorldBelief,
    prior_belief: WorldBelief,
    matched: Tensor,
) -> Tensor:
    """Return slots with a real incoming prior for correction objectives."""

    if matched.shape != belief.objects.active.shape or matched.dtype is not torch.bool:
        raise ValueError("matched correction support must be boolean belief-slot [B,N]")
    if prior_belief.objects.active.shape != matched.shape:
        raise ValueError("prior correction support must use the same belief-slot shape")
    return (
        matched
        & belief.objects.active.bool()
        & prior_belief.objects.active.bool()
        & (belief.objects.age_steps > 0)
    )


def _correction_non_regression_loss(
    posterior: Tensor,
    prior: Tensor,
    target: Tensor,
    object_support: Tensor,
    config: OrpheusConfig,
    *,
    coordinate_support: Tensor | None = None,
) -> Tensor:
    """Apply the configured vector or coordinate-wise correction hinge."""

    if posterior.shape != prior.shape or posterior.shape != target.shape:
        raise ValueError("correction states and target must have matching shapes")
    if object_support.shape != posterior.shape[:-1] or object_support.dtype is not torch.bool:
        raise ValueError("correction object support must be boolean [B,N]")
    if config.training.closed_loop_axiswise_correction_hinge_enabled:
        posterior_error = correction_error(posterior, target, axiswise=True)
        prior_error = correction_error(prior, target, axiswise=True)
        if coordinate_support is None:
            support = object_support.unsqueeze(-1).expand_as(posterior_error)
        else:
            if (
                coordinate_support.shape != posterior.shape
                or coordinate_support.dtype is not torch.bool
            ):
                raise ValueError("correction coordinate support must be boolean [B,N,C]")
            support = coordinate_support & object_support.unsqueeze(-1)
    else:
        # Preserve the exact historical vector-norm objective and object mask.
        posterior_error = correction_error(posterior, target)
        prior_error = correction_error(prior, target)
        support = object_support
    scenario_tail_fraction = config.training.closed_loop_scenario_tail_fraction
    if scenario_tail_fraction is not None and posterior_error.ndim == posterior.ndim:
        return torch.stack(
            [
                posterior_improvement_hinge(
                    posterior_error[..., coordinate],
                    prior_error[..., coordinate],
                    support[..., coordinate],
                    batch_macro=True,
                    batch_tail_fraction=scenario_tail_fraction,
                )
                for coordinate in range(posterior_error.shape[-1])
            ]
        ).mean()
    return posterior_improvement_hinge(
        posterior_error,
        prior_error,
        support,
        batch_macro=config.training.closed_loop_batch_macro_physical_losses_enabled,
        batch_tail_fraction=scenario_tail_fraction,
    )


def _rollout_metadata_equal(left: Any, right: Any) -> bool:
    """Compare heterogeneous metadata without ambiguous tensor truth values."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _rollout_metadata_equal(left[name], right[name]) for name in left
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            _rollout_metadata_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    return bool(equal) if isinstance(equal, bool) else False


def _rollout_beliefs_are_batch_compatible(
    left: WorldBelief,
    right: WorldBelief,
) -> bool:
    return left.active_modalities == right.active_modalities and _rollout_metadata_equal(
        left.metadata,
        right.metadata,
    )


def _concatenate_rollout_beliefs(beliefs: list[WorldBelief]) -> WorldBelief:
    """Concatenate anchor-major belief rows without weakening belief contracts."""

    if not beliefs:
        raise ValueError("rollout belief concatenation requires at least one belief")
    first = beliefs[0]
    for belief in beliefs[1:]:
        if not _rollout_beliefs_are_batch_compatible(first, belief):
            raise ValueError("batched rollout anchors must have identical modalities and metadata")

    objects = first.objects.replace(
        **{
            item.name: torch.cat(
                [getattr(belief.objects, item.name) for belief in beliefs],
                dim=0,
            )
            for item in fields(first.objects)
        }
    )
    camera = first.camera.replace(
        **{
            item.name: torch.cat(
                [getattr(belief.camera, item.name) for belief in beliefs],
                dim=0,
            )
            for item in fields(first.camera)
        }
    )
    return first.replace(
        timestamp=torch.cat([belief.timestamp for belief in beliefs], dim=0),
        objects=objects,
        camera=camera,
        gravity=torch.cat([belief.gravity for belief in beliefs], dim=0),
        global_code=torch.cat([belief.global_code for belief in beliefs], dim=0),
        global_log_variance=torch.cat(
            [belief.global_log_variance for belief in beliefs],
            dim=0,
        ),
        next_object_id=torch.cat([belief.next_object_id for belief in beliefs], dim=0),
    ).validate()


def _slice_rollout_trajectory(
    trajectory: BeliefTrajectory,
    *,
    batch_start: int,
    batch_stop: int,
    query_count: int,
) -> BeliefTrajectory:
    """Recover one anchor's exact unpadded trajectory from a batched rollout."""

    if not 0 <= batch_start < batch_stop <= trajectory.timestamps.shape[0]:
        raise ValueError("rollout trajectory batch slice lies outside the trajectory")
    if not 0 < query_count <= trajectory.timestamps.shape[1]:
        raise ValueError("rollout trajectory query prefix must be nonempty and in bounds")
    for name, value in trajectory.auxiliary.items():
        if value.ndim < 2 or value.shape[:2] != trajectory.timestamps.shape:
            raise ValueError(f"rollout auxiliary {name} must begin with the trajectory [B,Q] axes")
    return BeliefTrajectory(
        timestamps=trajectory.timestamps[batch_start:batch_stop, :query_count],
        positions=trajectory.positions[batch_start:batch_stop, :query_count],
        velocities=trajectory.velocities[batch_start:batch_stop, :query_count],
        orientations=trajectory.orientations[batch_start:batch_stop, :query_count],
        motion_mode_logits=(trajectory.motion_mode_logits[batch_start:batch_stop, :query_count]),
        fast_log_variance=(trajectory.fast_log_variance[batch_start:batch_stop, :query_count]),
        active_mask=trajectory.active_mask[batch_start:batch_stop, :query_count],
        event_logits=(
            None
            if trajectory.event_logits is None
            else trajectory.event_logits[batch_start:batch_stop, :query_count]
        ),
        auxiliary={
            name: value[batch_start:batch_stop, :query_count]
            for name, value in trajectory.auxiliary.items()
        },
    ).validate()


def _rollout_loss_result(
    model: OnlineWorldModel,
    belief: WorldBelief,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    frame_index: int,
    indices: Tensor,
    matched: Tensor,
    *,
    trajectory: BeliefTrajectory | None = None,
    compute_event_loss: bool = True,
    collect_promotion_metrics: bool = False,
    collect_protected_objective_cells: bool = False,
) -> _RolloutLossResult:
    if not isinstance(collect_promotion_metrics, bool):
        raise TypeError("collect_promotion_metrics must be bool")
    if not isinstance(collect_protected_objective_cells, bool):
        raise TypeError("collect_protected_objective_cells must be bool")
    total_frames = int(batch["rgb"].shape[1])
    frame_offsets, query_seconds, horizon_weights = _valid_rollout_offsets(
        config,
        frame_index,
        total_frames,
    )
    reference = belief.objects.position
    if not frame_offsets:
        return _RolloutLossResult(
            losses={},
            frame_offsets=[],
            query_seconds=[],
            positions=None,
            velocities=None,
            position_log_variance=None,
            active_mask=None,
            physical_metrics={},
            protected_objective_cells={},
        )
    event_query_plan = observation_window_query_plan(
        frame_offsets,
        frame_rate=config.simulator.frame_rate,
    )
    smooth_event_hazard_enabled = config.model.dynamics.smooth_event_hazard_enabled
    pair_collision_targets: Tensor | None = None
    if smooth_event_hazard_enabled and compute_event_loss:
        event_targets = batch.get("events")
        pair_collision_targets = (
            event_targets.get("pair_collision") if isinstance(event_targets, Mapping) else None
        )
        if (
            not isinstance(pair_collision_targets, Tensor)
            or pair_collision_targets.ndim != 4
            or pair_collision_targets.shape[1] != total_frames
            or pair_collision_targets.shape[2] != pair_collision_targets.shape[3]
        ):
            raise ValueError(
                "smooth event-hazard training requires events.pair_collision [B,T,N,N]"
            )
    if trajectory is None:
        request_pair_event_logits = smooth_event_hazard_enabled and compute_event_loss
        trajectory = model.dynamics.rollout(
            belief,
            event_query_plan.query_seconds,
            return_events=True,
            return_auxiliary=request_pair_event_logits,
            **({"auxiliary_names": ("pair_event_logits",)} if request_pair_event_logits else {}),
        )
    else:
        trajectory.validate()
        if trajectory.timestamps.shape != (
            belief.batch_size,
            len(event_query_plan.query_seconds),
        ):
            raise ValueError(
                "precomputed rollout trajectory must match the anchor batch and query plan"
            )
    target_positions = event_query_plan.select_target_endpoints(trajectory.positions)
    target_velocities = event_query_plan.select_target_endpoints(trajectory.velocities)
    target_position_log_variance = event_query_plan.select_target_endpoints(
        trajectory.fast_log_variance[..., :3]
    )
    target_active_mask = event_query_plan.select_target_endpoints(trajectory.active_mask)
    target_event_logits = (
        None
        if trajectory.event_logits is None
        else event_query_plan.select_target_endpoints(trajectory.event_logits)
    )
    target_pair_event_logits = (
        event_query_plan.select_target_endpoints(trajectory.auxiliary["pair_event_logits"])
        if smooth_event_hazard_enabled and compute_event_loss
        else None
    )
    position_losses: list[Tensor] = []
    position_axis_losses: dict[str, list[Tensor]] = {
        "x": [],
        "y": [],
        "z": [],
    }
    velocity_losses: list[Tensor] = []
    position_nll_losses: list[Tensor] = []
    point_horizon_weights: list[float] = []
    position_nll_weights: list[float] = []
    horizon_losses: dict[str, Tensor] = {}
    event_losses: list[Tensor] = []
    event_weights: list[float] = []
    node_event_losses: list[Tensor] = []
    node_event_weights: list[float] = []
    pair_event_losses: list[Tensor] = []
    pair_event_weights: list[float] = []
    physical_metrics: dict[str, float] = {}
    protected_objective_cells: dict[str, ProtectedObjectiveCell] = {}
    batch_macro = config.training.closed_loop_batch_macro_physical_losses_enabled
    scenario_tail_fraction = config.training.closed_loop_scenario_tail_fraction
    for query_index, frame_offset in enumerate(frame_offsets):
        target_index = frame_index + frame_offset
        future_active = (
            gather_target_slots(
                batch["objects"]["active"][:, target_index].unsqueeze(-1),
                indices,
            )
            .squeeze(-1)
            .bool()
        )
        # A dropped/deactivated forecast remains an error. Mask only by the
        # common target support so lifecycle collapse cannot lower the loss.
        valid = matched & future_active
        predictable = future_predictable_mask(
            batch,
            anchor_index=frame_index,
            target_index=target_index,
            target_indices=indices,
        )
        scene_predictable = future_scene_predictable_mask(
            batch,
            anchor_index=frame_index,
            target_index=target_index,
        )
        point_valid = valid & predictable
        mature = belief.objects.age_steps >= config.training.minimum_rollout_age_steps
        loss_valid = point_valid & mature
        target_position = gather_target_slots(
            batch["objects"]["position"][:, target_index], indices
        )
        target_velocity = gather_target_slots(
            batch["objects"]["velocity"][:, target_index], indices
        )
        position_loss = _axis_separable_masked_huber(
            target_positions[:, query_index],
            target_position,
            loss_valid,
            batch_macro=batch_macro,
            batch_tail_fraction=scenario_tail_fraction,
        )
        velocity_loss = _axis_separable_masked_huber(
            target_velocities[:, query_index],
            target_velocity,
            loss_valid,
            batch_macro=batch_macro,
            batch_tail_fraction=scenario_tail_fraction,
        )
        position_nll = _axis_separable_gaussian_nll(
            target_positions[:, query_index],
            target_position,
            target_position_log_variance[:, query_index],
            # An unseen intervention has no deterministic point target, but
            # its outcome still teaches the predictive distribution to widen.
            valid,
            detach_mean_error=True,
            batch_macro=batch_macro,
            batch_tail_fraction=scenario_tail_fraction,
            standardized_error_gradient_cap=(
                config.training.closed_loop_uncertainty_standardized_error_gradient_cap
            ),
        )
        seconds = query_seconds[query_index]
        # Do not represent an unsupported horizon as a zero-valued training
        # example. Downstream aggregation averages each horizon across anchors;
        # inserting cold/censored/unmatched zeros there silently shrinks real
        # gradients as the fraction of unsupported anchors changes.
        if loss_valid.any():
            point_horizon_weights.append(horizon_weights[query_index])
            position_losses.append(position_loss)
            velocity_losses.append(velocity_loss)
            horizon_losses[rollout_horizon_loss_key("rollout_position", seconds)] = position_loss
            horizon_losses[rollout_horizon_loss_key("rollout_velocity", seconds)] = velocity_loss
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                axis_loss = masked_huber(
                    target_positions[:, query_index, :, axis_index],
                    target_position[:, :, axis_index],
                    loss_valid,
                    batch_macro=batch_macro,
                    batch_tail_fraction=scenario_tail_fraction,
                )
                position_axis_losses[axis_name].append(axis_loss)
                horizon_losses[
                    rollout_horizon_loss_key(
                        f"rollout_position_{axis_name}",
                        seconds,
                    )
                ] = axis_loss
                if collect_protected_objective_cells:
                    protected_objective_cells[
                        rollout_horizon_loss_key(
                            f"rollout_position_{axis_name}",
                            seconds,
                        )
                    ] = _protected_objective_cell(
                        target_positions[:, query_index, :, axis_index],
                        target_position[:, :, axis_index],
                        loss_valid,
                    )
                    protected_objective_cells[
                        rollout_horizon_loss_key(
                            f"rollout_velocity_{axis_name}",
                            seconds,
                        )
                    ] = _protected_objective_cell(
                        target_velocities[:, query_index, :, axis_index],
                        target_velocity[:, :, axis_index],
                        loss_valid,
                    )
        if valid.any():
            position_nll_weights.append(horizon_weights[query_index])
            position_nll_losses.append(position_nll)
            horizon_losses[rollout_horizon_loss_key("rollout_position_nll", seconds)] = position_nll
        horizon_suffix = f"@{seconds:.3f}s"
        _add_squared_error_metrics(
            physical_metrics,
            prefix=f"physical_rollout_position{horizon_suffix}",
            prediction=target_positions[:, query_index],
            target=target_position,
            mask=point_valid,
        )
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            _add_squared_error_metrics(
                physical_metrics,
                prefix=f"physical_rollout_position_{axis_name}{horizon_suffix}",
                prediction=target_positions[:, query_index, :, axis_index],
                target=target_position[:, :, axis_index],
                mask=point_valid,
            )
        _add_gaussian_coverage_metrics(
            physical_metrics,
            prefix=f"physical_rollout_position_coverage90{horizon_suffix}",
            mean=target_positions[:, query_index],
            target=target_position,
            log_variance=target_position_log_variance[:, query_index],
            # Coverage is a stochastic calibration diagnostic, like the
            # proper forecast NLL above. Hidden interventions are excluded
            # from deterministic point RMSE but remain valid realised samples
            # for whether the predictive distribution widened enough.
            mask=valid,
        )
        _add_squared_error_metrics(
            physical_metrics,
            prefix=f"physical_rollout_velocity{horizon_suffix}",
            prediction=target_velocities[:, query_index],
            target=target_velocity,
            mask=point_valid,
        )
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            _add_squared_error_metrics(
                physical_metrics,
                prefix=f"physical_rollout_velocity_{axis_name}{horizon_suffix}",
                prediction=target_velocities[:, query_index, :, axis_index],
                target=target_velocity[:, :, axis_index],
                mask=point_valid,
            )
        _write_axis_partitioned_gaussian_calibration_metrics(
            physical_metrics,
            prefix=f"physical_rollout_position{horizon_suffix}",
            mean=target_positions[:, query_index],
            target=target_position,
            log_variance=target_position_log_variance[:, query_index],
            mask=valid,
        )
        _add_squared_error_metrics(
            physical_metrics,
            prefix=f"physical_rollout_mature_position{horizon_suffix}",
            prediction=target_positions[:, query_index],
            target=target_position,
            mask=point_valid & mature,
        )
        _add_squared_error_metrics(
            physical_metrics,
            prefix=f"physical_rollout_cold_start_position{horizon_suffix}",
            prediction=target_positions[:, query_index],
            target=target_position,
            mask=point_valid & ~mature,
        )
        physical_metrics[f"physical_forecast_target_count{horizon_suffix}"] = float(
            batch["objects"]["active"][:, target_index].sum().detach().cpu()
        )
        physical_metrics[f"physical_forecast_tracked_count{horizon_suffix}"] = float(
            valid.sum().detach().cpu()
        )
        physical_metrics[f"physical_forecast_active_count{horizon_suffix}"] = float(
            (valid & target_active_mask[:, query_index]).sum().detach().cpu()
        )
        physical_metrics[f"physical_rollout_predictable_target_count{horizon_suffix}"] = float(
            point_valid.sum().detach().cpu()
        )
        physical_metrics[f"physical_forecast_predictable_target_count{horizon_suffix}"] = float(
            (batch["objects"]["active"][:, target_index].bool() & scene_predictable[:, None])
            .sum()
            .detach()
            .cpu()
        )
        physical_metrics[f"physical_rollout_censored_external_actuation_count{horizon_suffix}"] = (
            float((valid & ~predictable).sum().detach().cpu())
        )
        target_ids = batch["objects"].get("id")
        if collect_promotion_metrics:
            physical_metrics[f"physical_forecast_identity_eligible_count{horizon_suffix}"] = 0.0
            physical_metrics[f"physical_forecast_identity_association_count{horizon_suffix}"] = 0.0
            physical_metrics[f"physical_forecast_identity_mismatch_count{horizon_suffix}"] = 0.0
        if collect_promotion_metrics and isinstance(target_ids, Tensor):
            anchor_aligned_position = gather_target_slots(
                batch["objects"]["position"][:, frame_index],
                indices,
            )
            anchor_distance_gated = _distance_gate_physical_matches(
                belief.objects.position,
                anchor_aligned_position,
                matched,
            )
            forecast_target_indices, forecast_matched = _match_positions_to_targets(
                target_positions[:, query_index],
                target_active_mask[:, query_index].bool(),
                batch["objects"]["position"][:, target_index],
                batch["objects"]["active"][:, target_index].bool(),
            )
            forecast_aligned_position = gather_target_slots(
                batch["objects"]["position"][:, target_index],
                forecast_target_indices,
            )
            forecast_distance_gated = _distance_gate_physical_matches(
                target_positions[:, query_index],
                forecast_aligned_position,
                forecast_matched,
            )
            anchor_target_ids = gather_target_slots(
                target_ids[:, frame_index],
                indices,
            )
            forecast_target_ids = gather_target_slots(
                target_ids[:, target_index],
                forecast_target_indices,
            )
            identity_eligible = (
                anchor_distance_gated
                & future_active
                & target_active_mask[:, query_index].bool()
                & (anchor_target_ids >= 0)
            )
            identity_associated = (
                identity_eligible & forecast_distance_gated & (forecast_target_ids >= 0)
            )
            physical_metrics[f"physical_forecast_identity_eligible_count{horizon_suffix}"] = float(
                identity_eligible.sum().detach().cpu()
            )
            physical_metrics[f"physical_forecast_identity_association_count{horizon_suffix}"] = (
                float(identity_associated.sum().detach().cpu())
            )
            physical_metrics[f"physical_forecast_identity_mismatch_count{horizon_suffix}"] = float(
                ((anchor_target_ids != forecast_target_ids) & identity_associated)
                .sum()
                .detach()
                .cpu()
            )
        for confusion_name in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ):
            physical_metrics[f"physical_collision_{confusion_name}_count{horizon_suffix}"] = 0.0
        if target_event_logits is not None:
            event_target = (
                gather_target_slots(
                    batch["events"]["collision"][:, target_index].unsqueeze(-1),
                    indices,
                )
                .squeeze(-1)
                .to(reference.dtype)
            )
            # Event labels at ``target_index`` cover exactly the simulator
            # interval from the preceding observation through this frame.
            # The expanded rollout query plan makes this endpoint logit cover
            # the same interval, independent of the other forecast horizons.
            event_scores = target_event_logits[:, query_index, :, MotionMode.COLLISION]
            if loss_valid.any():
                if collect_protected_objective_cells:
                    protected_objective_cells[
                        rollout_horizon_loss_key("event_collision_node", seconds)
                    ] = _protected_event_objective_cell(
                        event_scores,
                        event_target,
                        loss_valid,
                    )
                if compute_event_loss:
                    node_event_loss = balanced_binary_cross_entropy(
                        event_scores,
                        event_target,
                        loss_valid,
                        maximum_positive_weight=(config.training.collision_positive_weight_max),
                        batch_tail_fraction=scenario_tail_fraction,
                    )
                    event_loss = node_event_loss
                    node_event_losses.append(node_event_loss)
                    node_event_weights.append(horizon_weights[query_index])
                    horizon_losses[rollout_horizon_loss_key("event_collision_node", seconds)] = (
                        node_event_loss
                    )
                    if target_pair_event_logits is not None:
                        assert pair_collision_targets is not None
                        pair_target = gather_target_pairs(
                            pair_collision_targets[:, target_index],
                            indices,
                        ).to(reference.dtype)
                        belief_count = loss_valid.shape[1]
                        unique_pair = torch.triu(
                            torch.ones(
                                belief_count,
                                belief_count,
                                device=loss_valid.device,
                                dtype=torch.bool,
                            ),
                            diagonal=1,
                        ).unsqueeze(0)
                        pair_loss_valid = (
                            loss_valid[:, :, None] & loss_valid[:, None, :] & unique_pair
                        )
                        if pair_loss_valid.any():
                            pair_event_scores = target_pair_event_logits[:, query_index, :, :, 1]
                            pair_event_loss = balanced_binary_cross_entropy(
                                pair_event_scores,
                                pair_target,
                                pair_loss_valid,
                                maximum_positive_weight=(
                                    config.training.collision_positive_weight_max
                                ),
                                batch_tail_fraction=scenario_tail_fraction,
                            )
                            pair_event_losses.append(pair_event_loss)
                            pair_event_weights.append(horizon_weights[query_index])
                            horizon_losses[
                                rollout_horizon_loss_key("event_collision_pair", seconds)
                            ] = pair_event_loss
                            # Keep the historical event objective's aggregate
                            # scale while adding direct relation ownership.
                            event_loss = 0.5 * (event_loss + pair_event_loss)
                    event_losses.append(event_loss)
                    event_weights.append(horizon_weights[query_index])
                    horizon_losses[rollout_horizon_loss_key("event_collision", seconds)] = (
                        event_loss
                    )
                event_prediction = event_scores.detach().sigmoid() >= 0.5
                event_truth = event_target.detach().bool()
                confusion = {
                    "true_positive": event_prediction & event_truth,
                    "false_positive": event_prediction & ~event_truth,
                    "false_negative": ~event_prediction & event_truth,
                    "true_negative": ~event_prediction & ~event_truth,
                }
                for confusion_name, confusion_mask in confusion.items():
                    physical_metrics[
                        f"physical_collision_{confusion_name}_count{horizon_suffix}"
                    ] = float((confusion_mask & loss_valid).sum().cpu())

    def weighted_mean(losses: list[Tensor], weights: list[float]) -> Tensor:
        if not losses:
            raise ValueError("weighted rollout mean requires supported losses")
        weight = reference.new_tensor(weights)
        return (torch.stack(losses) * weight).sum() / weight.sum().clamp_min(1.0e-8)

    aggregate_losses: dict[str, Tensor] = {}
    if position_losses:
        aggregate_losses["rollout_position"] = weighted_mean(
            position_losses,
            point_horizon_weights,
        )
        for axis_name, axis_losses in position_axis_losses.items():
            aggregate_losses[f"rollout_position_{axis_name}"] = weighted_mean(
                axis_losses,
                point_horizon_weights,
            )
    if velocity_losses:
        aggregate_losses["rollout_velocity"] = weighted_mean(
            velocity_losses,
            point_horizon_weights,
        )
    if position_nll_losses:
        aggregate_losses["rollout_position_nll"] = weighted_mean(
            position_nll_losses,
            position_nll_weights,
        )
    if event_losses:
        aggregate_losses["event_collision"] = weighted_mean(event_losses, event_weights)
    if node_event_losses:
        aggregate_losses["event_collision_node"] = weighted_mean(
            node_event_losses,
            node_event_weights,
        )
    if pair_event_losses:
        aggregate_losses["event_collision_pair"] = weighted_mean(
            pair_event_losses,
            pair_event_weights,
        )
    aggregate_losses.update(horizon_losses)
    return _RolloutLossResult(
        losses=aggregate_losses,
        frame_offsets=frame_offsets,
        query_seconds=query_seconds,
        positions=target_positions,
        velocities=target_velocities,
        position_log_variance=target_position_log_variance,
        active_mask=target_active_mask,
        physical_metrics=physical_metrics,
        protected_objective_cells=protected_objective_cells,
    )


def _batched_validation_rollout_loss_results(
    model: OnlineWorldModel,
    anchors: list[_DeferredValidationRollout],
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    collect_promotion_metrics: bool,
) -> _ValidationRolloutExecution:
    """Evaluate one anchor-major validation chunk with exact serial scoring.

    Later anchors can support fewer horizons. Their observation-window query
    plans are exact prefixes of the earliest plan, so their rows are padded by
    repeating only their terminal query time. The padded suffix is discarded
    before the unchanged per-anchor scorer sees the trajectory.
    """

    if not anchors:
        return _ValidationRolloutExecution([], 0, 0, 0)
    if len(anchors) == 1:
        anchor = anchors[0]
        return _ValidationRolloutExecution(
            results=[
                _rollout_loss_result(
                    model,
                    anchor.belief,
                    batch,
                    config,
                    anchor.frame_index,
                    anchor.indices,
                    anchor.matched,
                    collect_promotion_metrics=collect_promotion_metrics,
                )
            ],
            batched_anchor_count=0,
            serial_fallback_anchor_count=1,
            rollout_call_count=1,
        )
    if config.model.dynamics.learned_effect_interval_seconds is not None:
        raise ValueError(
            "validation rollout anchor batching does not support multi-rate learned "
            "effects; set training.validation_rollout_anchor_batch_size=1"
        )

    compatible_groups: list[list[_DeferredValidationRollout]] = []
    for anchor in anchors:
        if not compatible_groups or not _rollout_beliefs_are_batch_compatible(
            compatible_groups[-1][0].belief,
            anchor.belief,
        ):
            compatible_groups.append([anchor])
        else:
            compatible_groups[-1].append(anchor)
    if len(compatible_groups) > 1:
        grouped_results: list[_RolloutLossResult] = []
        batched_anchor_count = 0
        serial_fallback_anchor_count = 0
        rollout_call_count = 0
        for group in compatible_groups:
            execution = _batched_validation_rollout_loss_results(
                model,
                group,
                batch,
                config,
                collect_promotion_metrics=collect_promotion_metrics,
            )
            grouped_results.extend(execution.results)
            batched_anchor_count += execution.batched_anchor_count
            serial_fallback_anchor_count += execution.serial_fallback_anchor_count
            rollout_call_count += execution.rollout_call_count
        return _ValidationRolloutExecution(
            results=grouped_results,
            batched_anchor_count=batched_anchor_count,
            serial_fallback_anchor_count=serial_fallback_anchor_count,
            rollout_call_count=rollout_call_count,
        )

    total_frames = int(batch["rgb"].shape[1])
    plans = []
    for anchor in anchors:
        frame_offsets, _, _ = _valid_rollout_offsets(
            config,
            anchor.frame_index,
            total_frames,
        )
        if not frame_offsets:
            raise ValueError("deferred validation rollout anchor has no valid horizon")
        plans.append(
            observation_window_query_plan(
                frame_offsets,
                frame_rate=config.simulator.frame_rate,
            )
        )
    longest_plan = max(plans, key=lambda plan: len(plan.query_frame_offsets))
    longest_offsets = longest_plan.query_frame_offsets
    for plan in plans:
        query_count = len(plan.query_frame_offsets)
        if plan.query_frame_offsets != longest_offsets[:query_count]:
            raise ValueError(
                "validation rollout anchor query plans must be prefixes before batching"
            )

    belief = _concatenate_rollout_beliefs([anchor.belief for anchor in anchors])
    query_count = len(longest_offsets)
    query_rows: list[Tensor] = []
    for anchor, plan in zip(anchors, plans, strict=True):
        padded_seconds = (
            *plan.query_seconds,
            *(plan.query_seconds[-1:] * (query_count - len(plan.query_seconds))),
        )
        query_rows.append(
            anchor.belief.timestamp.new_tensor(padded_seconds)
            .unsqueeze(0)
            .expand(anchor.belief.batch_size, -1)
            .clone()
        )
    query_times = torch.cat(query_rows, dim=0)
    smooth_event_hazard_enabled = config.model.dynamics.smooth_event_hazard_enabled
    trajectory = model.dynamics.rollout(
        belief,
        query_times,
        return_events=True,
        return_auxiliary=smooth_event_hazard_enabled,
        **({"auxiliary_names": ("pair_event_logits",)} if smooth_event_hazard_enabled else {}),
    )

    results: list[_RolloutLossResult] = []
    batch_cursor = 0
    for anchor, plan in zip(anchors, plans, strict=True):
        batch_stop = batch_cursor + anchor.belief.batch_size
        anchor_trajectory = _slice_rollout_trajectory(
            trajectory,
            batch_start=batch_cursor,
            batch_stop=batch_stop,
            query_count=len(plan.query_frame_offsets),
        )
        results.append(
            _rollout_loss_result(
                model,
                anchor.belief,
                batch,
                config,
                anchor.frame_index,
                anchor.indices,
                anchor.matched,
                trajectory=anchor_trajectory,
                collect_promotion_metrics=collect_promotion_metrics,
            )
        )
        batch_cursor = batch_stop
    return _ValidationRolloutExecution(
        results=results,
        batched_anchor_count=len(anchors),
        serial_fallback_anchor_count=0,
        rollout_call_count=1,
    )


def _rollout_losses(
    model: OnlineWorldModel,
    belief: WorldBelief,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    frame_index: int,
    indices: Tensor,
    matched: Tensor,
) -> dict[str, Tensor]:
    """Backward-compatible loss-only wrapper used by focused unit tests."""

    return _rollout_loss_result(
        model,
        belief,
        batch,
        config,
        frame_index,
        indices,
        matched,
    ).losses


def _group_closed_loop_terms(
    details: dict[str, Tensor],
    reference: Tensor,
) -> dict[str, Tensor]:
    def optional_total(*names: str) -> Tensor | None:
        selected = [details[name] for name in names if name in details]
        return _mean_losses(selected, reference) if selected else None

    terms: dict[str, Tensor] = {}
    for name in (
        "measurement",
        "state_position",
        "state_velocity",
        "rollout_position",
        "rollout_position_x",
        "rollout_position_y",
        "rollout_position_z",
        "rollout_velocity",
        "attention_node_complexity",
        "attention_node_activity",
        "attention_node_drift",
        "soft_association_state",
        "soft_association_velocity",
        "soft_association_exclusivity",
    ):
        if name in details:
            terms[name] = details[name]
    state = optional_total("state_position", "state_velocity")
    if state is not None:
        terms["state"] = state
    rollout = optional_total("rollout_position", "rollout_velocity")
    if rollout is not None:
        terms["rollout"] = rollout
    if "rollout_position_nll" in details:
        terms["rollout_nll"] = details["rollout_position_nll"]
    if "event_collision" in details:
        terms["event"] = details["event_collision"]
    parameter = optional_total("parameter_drag", "parameter_restitution")
    if parameter is not None:
        terms["parameter"] = parameter
    for detail_name, term_name in (
        ("existence_belief", "existence"),
        ("uncertainty_position_nll", "uncertainty"),
    ):
        if detail_name in details:
            terms[term_name] = details[detail_name]

    correction_position = optional_total(
        "correction_current",
        "correction_future",
    )
    correction_velocity = optional_total(
        "correction_current_velocity",
        "correction_future_velocity",
    )
    correction_regularization = details.get("correction_magnitude")
    if correction_position is not None:
        terms["correction_position"] = correction_position
    if correction_velocity is not None:
        terms["correction_velocity"] = correction_velocity
    if correction_regularization is not None:
        terms["correction_regularization"] = correction_regularization
    correction = [
        value
        for value in (
            correction_position,
            correction_velocity,
            correction_regularization,
        )
        if value is not None
    ]
    if correction:
        terms["correction"] = _mean_losses(
            correction,
            reference,
        )
    if not terms:
        # Preserve an explicit no-gradient result for the trainer's retry
        # policy without fabricating support for any physical objective.
        terms["unsupported"] = reference.sum() * 0
    return terms


def _event_horizon_support_terms(details: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Retain already-built node-event horizon tensors for gradient audits."""

    return {
        name: value for name, value in details.items() if name.startswith("event_collision_node@")
    }


def _weighted_closed_loop_total(
    terms: dict[str, Tensor],
    weights: dict[str, float],
) -> Tensor:
    """Weight physical loss components without double-counting aliases.

    ``state`` and ``rollout`` remain backward-compatible aggregate terms for
    existing configs, logs, and checkpoints.  When a config supplies an exact
    component weight such as ``state_position`` or ``rollout_velocity``, the
    two physical components replace their aggregate for optimisation.  This
    makes position/velocity trade-offs explicit while keeping old profiles
    numerically unchanged. Terms with an effective zero weight are structurally
    omitted from the autograd graph. Multiplying a disabled branch by zero is
    insufficient because a nonfinite local derivative can still contaminate
    otherwise healthy gradients.
    """

    aggregate_families = {
        "state": ("state_position", "state_velocity"),
        "rollout": ("rollout_position", "rollout_velocity"),
        "correction": (
            "correction_position",
            "correction_velocity",
            "correction_regularization",
        ),
    }
    rollout_position_axes = (
        "rollout_position_x",
        "rollout_position_y",
        "rollout_position_z",
    )
    use_axis_position = any(name in weights for name in rollout_position_axes)
    # Optional objectives added after older explicit YAML profiles were
    # written must be opt-in by exact key.  Letting ``weighted_total`` fall
    # back from ``rollout_nll`` to an absent ``rollout`` key assigns weight
    # 1.0, fifty times the declared default 0.02.
    optional_exact_weight_terms = {
        "rollout_nll",
        "attention_node_complexity",
        "attention_node_activity",
        "attention_node_drift",
        "soft_association_state",
        "soft_association_velocity",
        "soft_association_exclusivity",
    }
    selected: dict[str, Tensor] = {
        name: value
        for name, value in terms.items()
        if name
        not in {component for components in aggregate_families.values() for component in components}
        and name not in aggregate_families
        and name not in rollout_position_axes
        and not (name in optional_exact_weight_terms and name not in weights)
    }
    for aggregate, components in aggregate_families.items():
        if any(component in weights for component in components):
            for component in components:
                if component == "rollout_position" and use_axis_position:
                    for axis_name in rollout_position_axes:
                        if axis_name in terms:
                            selected[axis_name] = terms[axis_name]
                elif component in terms:
                    selected[component] = terms[component]
        elif aggregate in terms:
            selected[aggregate] = terms[aggregate]
    selected = {
        name: value
        for name, value in selected.items()
        if float(weights.get(name, weights.get(name.split("_", 1)[0], 1.0))) != 0.0
    }
    if not selected:
        # Every supported term may be optional and disabled by the resolved
        # weights. Return a detached no-op so disabled branches cannot receive
        # hooks or propagate a zero-times-nonfinite derivative.
        return next(iter(terms.values())).detach().new_zeros(())
    return weighted_total(selected, weights)


def _closed_loop_loss_weights_for_scope(
    config: OrpheusConfig,
    *,
    active_trainable_scope: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve a causal scope's objective weights and numeric provenance.

    A missing scope override is exactly the historical behavior: the event
    term uses ``loss_weights.event`` (or the weighted-total unit fallback when
    that legacy key is absent). The caller passes the already-resolved causal
    scope so a staged transition cannot depend on window or data details.
    """

    weights = dict(config.training.loss_weights)
    legacy_event_weight = float(weights.get("event", 1.0))
    scope_weights = config.training.closed_loop_event_loss_weights
    override_active = active_trainable_scope is not None and active_trainable_scope in scope_weights
    effective_event_weight = (
        float(scope_weights[active_trainable_scope])
        if override_active and active_trainable_scope is not None
        else legacy_event_weight
    )
    weights["event"] = effective_event_weight
    scopes_without_direct_event_owner = {
        "attention_node_x",
        "attention_node_y",
        "attention_node_z",
        "fast_roi",
        "state_roi",
        "updater",
        "updater_state_heads",
        "updater_state_heads_xy",
        "updater_mean",
        "updater_mean_y",
    }
    suppressed_without_owner = (
        override_active
        and active_trainable_scope in scopes_without_direct_event_owner
        and effective_event_weight == 0.0
    )
    diagnostics = {
        "effective_event_loss_weight": effective_event_weight,
        "event_loss_scope_override_active": float(override_active),
        "event_loss_legacy_weight_active": float(not override_active),
        "event_loss_suppressed_no_trainable_owner": float(suppressed_without_owner),
    }
    return weights, diagnostics


def _attention_node_complexity_details(model: OnlineWorldModel) -> dict[str, Tensor]:
    """Return an axis-neutral parsimony prior for attention node residuals.

    RMS-normalized entity tokens make decoder-row energy a stable proxy for
    the capacity assigned to persistent learned acceleration.  Averaging the
    squared L2 energy of the three typed rows penalizes a disproportionately
    large axis more strongly without forbidding evidence-supported residuals
    or changing inference.  The objective remains opt-in through the exact
    ``attention_node_complexity`` loss weight.
    """

    attention = model.dynamics.attention_interactions
    if attention is None:
        return {}
    weight = attention.node_decoder.weight
    bias = attention.node_decoder.bias
    row_energy = weight.square().sum(dim=-1)
    if bias is not None:
        row_energy = row_energy + bias.square()
    details = {
        "attention_node_complexity": row_energy.mean(),
    }
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        details[f"attention_node_complexity_{axis_name}"] = row_energy[axis_index]
    return details


def _attention_node_activity_details(model: OnlineWorldModel) -> dict[str, Tensor]:
    """Return the optional causal-rollout node acceleration activity prior."""

    attention = model.dynamics.attention_interactions
    if attention is None:
        return {}
    return attention.node_activity_details()


def select_closed_loop_window(
    batch: Mapping[str, Any],
    window_steps: int,
    *,
    event_condition_probability: float = 0.5,
    maximum_rollout_frame_offset: int | None = None,
    minimum_rollout_frame_offset: int | None = None,
    long_horizon_probability: float = 0.0,
    joint_collision_long_horizon_sampling: bool = False,
) -> int:
    """Sample a valid TBPTT window, preferentially covering collision frames.

    The returned start is stochastic under the trainer's seeded Python RNG.
    Under legacy sampling, collision conditioning has first priority so late
    events remain trainable.  Joint sampling draws both intents independently:
    a window satisfying both is used when possible, while an incompatible late
    collision cannot erase a sampled maximum-horizon example. Labels select
    only the loss window; they are never passed to the RGB runtime.
    """

    rgb = batch.get("rgb")
    if not isinstance(rgb, Tensor) or rgb.ndim != 5:
        raise ValueError("closed-loop batch must contain rgb [B,T,3,H,W]")
    total_frames = int(rgb.shape[1])
    if not 0 < window_steps <= total_frames:
        raise ValueError("window_steps must lie in [1, episode_frames]")
    if not 0.0 <= event_condition_probability <= 1.0:
        raise ValueError("event_condition_probability must lie in [0, 1]")
    if not 0.0 <= long_horizon_probability <= 1.0:
        raise ValueError("long_horizon_probability must lie in [0, 1]")
    if not isinstance(joint_collision_long_horizon_sampling, bool):
        raise ValueError("joint_collision_long_horizon_sampling must be boolean")
    if maximum_rollout_frame_offset is not None and maximum_rollout_frame_offset <= 0:
        raise ValueError("maximum_rollout_frame_offset must be positive")
    if minimum_rollout_frame_offset is not None and minimum_rollout_frame_offset <= 0:
        raise ValueError("minimum_rollout_frame_offset must be positive")
    if (
        minimum_rollout_frame_offset is not None
        and maximum_rollout_frame_offset is not None
        and minimum_rollout_frame_offset > maximum_rollout_frame_offset
    ):
        raise ValueError("minimum rollout frame offset cannot exceed maximum")
    maximum_start = total_frames - window_steps
    if minimum_rollout_frame_offset is not None:
        last_any_horizon_anchor = total_frames - minimum_rollout_frame_offset - 1
        if last_any_horizon_anchor < 0:
            raise ValueError("minimum rollout frame offset exceeds the episode")
        maximum_start = min(maximum_start, last_any_horizon_anchor)
    if maximum_start == 0:
        return 0

    events = batch.get("events")
    collision = events.get("collision") if isinstance(events, Mapping) else None
    pair_collision = events.get("pair_collision") if isinstance(events, Mapping) else None
    collision_frames: list[int] = []
    collision_source = pair_collision if isinstance(pair_collision, Tensor) else collision
    if isinstance(collision_source, Tensor):
        if collision_source.ndim < 2 or tuple(collision_source.shape[:2]) != tuple(rgb.shape[:2]):
            source_name = (
                "events.pair_collision"
                if collision_source is pair_collision
                else "events.collision"
            )
            raise ValueError(f"{source_name} must begin with batch/time axes [B,T]")
        collision_by_frame = (
            collision_source.bool()
            .reshape(
                collision_source.shape[0],
                collision_source.shape[1],
                -1,
            )
            .any(dim=(0, 2))
        )
        collision_frames = (
            torch.nonzero(collision_by_frame, as_tuple=False).flatten().detach().cpu().tolist()
        )
        # Pair interactions are the scarce, causally informative examples.
        # If an episode contains none, retain the historical boundary/any-event
        # fallback so collision-conditioned sampling still has useful support.
        if (
            not collision_frames
            and collision_source is pair_collision
            and isinstance(collision, Tensor)
        ):
            if collision.ndim < 2 or tuple(collision.shape[:2]) != tuple(rgb.shape[:2]):
                raise ValueError("events.collision must begin with batch/time axes [B,T]")
            collision_by_frame = (
                collision.bool().reshape(collision.shape[0], collision.shape[1], -1).any(dim=(0, 2))
            )
            collision_frames = (
                torch.nonzero(collision_by_frame, as_tuple=False).flatten().detach().cpu().tolist()
            )
    condition_on_long_horizon = False
    if joint_collision_long_horizon_sampling:
        condition_on_long_horizon = long_horizon_probability >= 1.0 or (
            long_horizon_probability > 0.0 and random.random() < long_horizon_probability
        )
    if collision_frames and random.random() < event_condition_probability:
        compatible_collision_frames = collision_frames
        last_eligible_anchor: int | None = None
        if (
            joint_collision_long_horizon_sampling
            and condition_on_long_horizon
            and maximum_rollout_frame_offset is not None
        ):
            last_eligible_anchor = total_frames - maximum_rollout_frame_offset - 1
            if last_eligible_anchor < 0:
                raise ValueError("maximum rollout frame offset exceeds the episode")
            compatible_collision_frames = [
                event_frame
                for event_frame in collision_frames
                if max(0, event_frame - window_steps + 1)
                <= min(
                    maximum_start,
                    event_frame - 1 if event_frame > 0 else event_frame,
                    last_eligible_anchor,
                )
            ]
        if compatible_collision_frames:
            event_frame = int(random.choice(compatible_collision_frames))
            minimum_start = max(0, event_frame - window_steps + 1)
            maximum_event_start = min(maximum_start, event_frame)
            if event_frame > 0:
                maximum_event_start = min(maximum_event_start, event_frame - 1)
            if last_eligible_anchor is not None:
                maximum_event_start = min(maximum_event_start, last_eligible_anchor)
            if minimum_start <= maximum_event_start:
                if minimum_rollout_frame_offset is not None:
                    aligned_start = event_frame - minimum_rollout_frame_offset
                    if minimum_start <= aligned_start <= maximum_event_start:
                        # The trainer scores rollout events at configured
                        # horizon endpoints. Align the scarce collision to the
                        # shortest endpoint so a conditioned window actually
                        # contributes a positive event BCE target.
                        return aligned_start
                return random.randint(minimum_start, maximum_event_start)
    if not joint_collision_long_horizon_sampling:
        condition_on_long_horizon = long_horizon_probability >= 1.0 or (
            long_horizon_probability > 0.0 and random.random() < long_horizon_probability
        )
    if condition_on_long_horizon and maximum_rollout_frame_offset is not None:
        last_eligible_anchor = total_frames - maximum_rollout_frame_offset - 1
        if last_eligible_anchor < 0:
            raise ValueError("maximum rollout frame offset exceeds the episode")
        # Among non-event-conditioned windows, make the first trainable frame
        # itself a valid maximum-horizon anchor.
        maximum_start = min(maximum_start, last_eligible_anchor)
    return random.randint(0, maximum_start)


def _burn_in_causal_prefix(
    model: OnlineWorldModel,
    batch: Mapping[str, Any],
    window_start: int,
    target_matcher: PersistentTargetMatcher,
    last_observed_target_frame: Tensor,
    last_observed_runtime_id: Tensor,
) -> tuple[Tensor, Tensor]:
    """Advance the real RGB filter to a mid-episode loss window.

    Prefix frames update the persistent belief, lifecycle, modality caches,
    association state, and scheduler exactly as online inference would.  They
    are deliberately outside the autograd graph; the selected TBPTT window
    starts from their detached numerical posterior instead of a cold reset.
    """

    if window_start < 0:
        raise ValueError("window_start must be nonnegative")
    if last_observed_runtime_id.shape != last_observed_target_frame.shape:
        raise ValueError("parameter observation frame/identity histories must match")
    with torch.no_grad():
        for frame_index in range(window_start):
            belief = model.ingest(make_rgb_packet(batch, frame_index))
            indices, matched = target_matcher.match(
                belief,
                batch["objects"]["position"][:, frame_index],
                batch["objects"]["active"][:, frame_index].bool(),
            )
            runtime_observed = _runtime_observed_belief_slots(
                model,
                belief,
            )
            aligned_position = gather_target_slots(
                batch["objects"]["position"][:, frame_index],
                indices,
            )
            distance_gated_matched = _distance_gate_physical_matches(
                belief.objects.position,
                aligned_position,
                matched,
            )
            observed_runtime_id = _target_observed_runtime_ids(
                indices,
                distance_gated_matched,
                runtime_observed,
                belief.objects.object_id,
                target_count=last_observed_target_frame.shape[1],
            )
            target_observed = observed_runtime_id >= 0
            last_observed_target_frame = torch.where(
                target_observed,
                torch.full_like(last_observed_target_frame, frame_index),
                last_observed_target_frame,
            )
            last_observed_runtime_id = torch.where(
                target_observed,
                observed_runtime_id,
                last_observed_runtime_id,
            )
    if window_start:
        model.detach_state()
    return last_observed_target_frame, last_observed_runtime_id


def _validate_scenario_tail_training_batch(
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    training_with_gradients: bool,
) -> None:
    """Bind tail-risk rows to the declared scenario-balanced protocol."""

    if config.training.closed_loop_scenario_tail_fraction is None or not training_with_gradients:
        return
    metadata = batch.get("metadata")
    scenarios = metadata.get("scenario") if isinstance(metadata, Mapping) else None
    if not isinstance(scenarios, list) or tuple(str(item) for item in scenarios) != tuple(
        config.simulator.scenario_mixture
    ):
        raise ValueError(
            "scenario-tail training requires one batch row from every declared "
            "scenario in canonical order"
        )


def run_closed_loop_batch(
    model: OnlineWorldModel,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    window_start: int = 0,
    window_steps: int | None = None,
    apply_perturbations: bool = True,
    include_measurement_supervision: bool = True,
    rollout_anchors_per_window: int | None = None,
    validation_rollout_anchor_batch_size: int = 1,
    compute_future_correction: bool | None = None,
    active_trainable_scope: str | None = None,
    collect_promotion_metrics: bool = False,
    collect_protected_objective_cells: bool = False,
) -> TrainingBatchResult:
    """Run one causal RGB-only sequence window through the real runtime.

    The belief is never reset to simulator state.  Labels are read only after
    each RGB ingest to compute supervised losses. Validation may disable the
    extra prior future rollout used only by the correction-improvement guard.
    ``None`` follows the protocol-bound training configuration. Current
    correction and every posterior physical forecast remain measured.
    """

    if config.runtime.modality != "rgb":
        raise ValueError("closed-loop milestone training requires runtime.modality=rgb")
    rgb = batch["rgb"]
    if not isinstance(rgb, Tensor) or rgb.ndim != 5:
        raise ValueError("closed-loop batch must contain rgb [B,T,3,H,W]")
    batch_size, total_frames = rgb.shape[:2]
    _validate_scenario_tail_training_batch(
        batch,
        config,
        training_with_gradients=model.training and torch.is_grad_enabled(),
    )
    if compute_future_correction is None:
        compute_future_correction = config.training.closed_loop_prior_future_correction_enabled
    elif not isinstance(compute_future_correction, bool):
        raise ValueError("compute_future_correction must be boolean or None")
    if not isinstance(collect_promotion_metrics, bool):
        raise ValueError("collect_promotion_metrics must be boolean")
    if not isinstance(collect_protected_objective_cells, bool):
        raise ValueError("collect_protected_objective_cells must be boolean")
    loss_weights, event_weight_metrics = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope=active_trainable_scope,
    )
    compute_event_loss = float(loss_weights["event"]) != 0.0
    soft_association_temperature = config.training.closed_loop_soft_association_temperature
    soft_association_enabled = (
        model.training
        and torch.is_grad_enabled()
        and soft_association_temperature is not None
        and any(
            float(loss_weights.get(name, 0.0)) > 0.0
            for name in (
                "soft_association_state",
                "soft_association_velocity",
                "soft_association_exclusivity",
            )
        )
    )
    if (
        isinstance(validation_rollout_anchor_batch_size, bool)
        or not isinstance(validation_rollout_anchor_batch_size, int)
        or validation_rollout_anchor_batch_size <= 0
    ):
        raise ValueError("validation_rollout_anchor_batch_size must be a positive integer")
    batch_validation_rollouts = validation_rollout_anchor_batch_size > 1
    if batch_validation_rollouts:
        if model.training or torch.is_grad_enabled():
            raise ValueError(
                "validation rollout anchor batching requires model.eval() under torch.no_grad()"
            )
        if batch_size != 1:
            raise ValueError(
                "validation rollout anchor batching requires episode loader batch size one"
            )
        if apply_perturbations or compute_future_correction:
            raise ValueError(
                "validation rollout anchor batching requires unperturbed posterior-only validation"
            )
        if config.model.dynamics.learned_effect_interval_seconds is not None:
            raise ValueError(
                "validation rollout anchor batching does not support multi-rate learned "
                "effects; set training.validation_rollout_anchor_batch_size=1"
            )
    if window_steps is None:
        window_steps = min(total_frames - window_start, config.training.tbptt_steps)
    if window_steps <= 0 or window_start < 0:
        raise ValueError("closed-loop window must contain at least one frame")
    window_stop = min(total_frames, window_start + window_steps)
    if window_start >= window_stop:
        raise ValueError("closed-loop window lies outside the episode")
    rollout_anchor_frames = _select_rollout_anchor_frames(
        config,
        window_start=window_start,
        window_stop=window_stop,
        total_frames=total_frames,
        rollout_anchors_per_window=rollout_anchors_per_window,
    )
    rollout_anchor_set = set(rollout_anchor_frames)

    model.reset(batch_size=batch_size)
    target_matcher = PersistentTargetMatcher()
    target_count = int(batch["objects"]["active"].shape[-1])
    last_observed_target_frame = torch.full(
        (batch_size, target_count),
        -1,
        device=rgb.device,
        dtype=torch.int64,
    )
    last_observed_runtime_id = torch.full_like(last_observed_target_frame, -1)
    last_observed_target_frame, last_observed_runtime_id = _burn_in_causal_prefix(
        model,
        batch,
        window_start,
        target_matcher,
        last_observed_target_frame,
        last_observed_runtime_id,
    )
    detail_lists: dict[str, list[Tensor]] = {}
    current_correction_improvements: list[float] = []
    future_correction_improvements: list[float] = []
    current_velocity_correction_improvements: list[float] = []
    future_velocity_correction_improvements: list[float] = []
    perturbed_updates = 0
    matched_count = 0
    existence_negative_support_count = 0
    state_velocity_objective_supported_coordinate_count = 0
    state_velocity_objective_excluded_coordinate_count = 0
    current_velocity_correction_supported_object_count = 0
    future_velocity_correction_supported_object_horizon_count = 0
    target_object_frames = 0
    predicted_object_frames = 0
    fast_supervised_frames = 0
    # Count prior-conditioned slots that carry at least one real fast-path
    # objective. This includes positive crop evidence and mapped, valid empty
    # crops supervised as existence/visibility negatives; invalid or unmapped
    # padding never counts.
    fast_supervised_slots = 0
    physical_metrics: dict[str, float] = {}
    protected_cell_lists: dict[str, list[ProtectedObjectiveCell]] = {}
    deferred_validation_rollouts: list[_DeferredValidationRollout] = []
    rollout_execution_batched_anchor_count = 0
    rollout_execution_serial_fallback_anchor_count = 0
    rollout_execution_posterior_call_count = 0
    parameter_supervision_metrics: dict[str, float] = {}
    identity_switches = 0
    object_frame_associations = 0
    distance_gated_identity_switches = 0
    distance_gated_object_frame_associations = 0
    distance_gated_matched_count = 0
    distance_gated_target_object_frames = 0
    distance_gated_predicted_object_frames = 0
    last_predicted_id_for_target: list[dict[int, int]] = [{} for _ in range(batch_size)]
    last_distance_gated_predicted_id_for_target: list[dict[int, int]] = [
        {} for _ in range(batch_size)
    ]

    def add(name: str, value: Tensor) -> None:
        detail_lists.setdefault(name, []).append(value)

    def add_protected(name: str, value: ProtectedObjectiveCell) -> None:
        protected_cell_lists.setdefault(name, []).append(value)

    for frame_index in range(window_start, window_stop):
        packet = make_rgb_packet(batch, frame_index)
        score_rollout = frame_index in rollout_anchor_set
        prior_belief: WorldBelief | None = None
        prior_rollout = None
        prior_rollout_positions: Tensor | None = None
        prior_rollout_velocities: Tensor | None = None
        frame_offsets: list[int] = []
        query_seconds: list[float] = []
        prepared_propagation: PreparedPropagation | None = None
        if model.belief is not None:
            source_belief = model.belief
            if apply_perturbations and random.random() < config.training.perturbation_probability:
                source_belief = perturb_belief(
                    source_belief,
                    position_std=config.training.perturbation_position_std,
                    velocity_std=config.training.perturbation_velocity_std,
                    covariance_log_bias=0.25,
                )
                # Preserve the previous timestamp.  ``ingest`` remains the
                # authoritative predictor, sees the real dt, and can therefore
                # infer velocity from consecutive RGB position measurements.
                model.state.belief = source_belief
                perturbed_updates += 1
            prepared_propagation = model.prepare_propagation(packet.timestamp)
            prior = prepared_propagation.prior
            prior_belief = prior
            if score_rollout:
                frame_offsets, query_seconds, _ = _valid_rollout_offsets(
                    config,
                    frame_index,
                    total_frames,
                )
                if frame_offsets and compute_future_correction:
                    event_query_plan = observation_window_query_plan(
                        frame_offsets,
                        frame_rate=config.simulator.frame_rate,
                    )
                    # This trajectory is only the detached reference for the
                    # posterior-improvement hinge and scalar diagnostics.
                    # Building its recursive autograd graph roughly doubles
                    # rollout memory without contributing any gradient.
                    with torch.no_grad():
                        prior_rollout = model.dynamics.rollout(
                            prior,
                            event_query_plan.query_seconds,
                            return_events=False,
                            return_auxiliary=False,
                        )
                        prior_rollout_positions = event_query_plan.select_target_endpoints(
                            prior_rollout.positions
                        )
                        prior_rollout_velocities = event_query_plan.select_target_endpoints(
                            prior_rollout.velocities
                        )
            if include_measurement_supervision:
                module = model.observation_modules["rgb"]
                # Fast measurement supervision owns only the RGB perception
                # branch.  Its projected ROI is conditioned on the live
                # propagated prior, but that conditioning must remain an
                # input rather than an auxiliary gradient route back through
                # the updater, identifier, or dynamics.  Clone as well as
                # detach so the diagnostic forward cannot alias persistent
                # belief storage.  A cached ROI feature can carry the same
                # prior ancestry from an earlier runtime ingest, so present a
                # detached cache to this auxiliary-only encode too.  The
                # ordinary ingest below still consumes ``prepared_propagation``
                # and the live runtime cache unchanged; state and rollout
                # objectives therefore retain their complete causal graph.
                auxiliary_prior = prior.detach().clone()
                runtime_cache = model.state.caches.get(packet.sensor_id)
                auxiliary_cache = None if runtime_cache is None else runtime_cache.detach()
                predicted = module.project(
                    auxiliary_prior,
                    SensorContext(
                        sensor_id=packet.sensor_id,
                        timestamp=packet.timestamp,
                        calibration=packet.calibration,
                        frame_id=packet.frame_id,
                        image_size=packet.metadata["image_size"],
                        metadata=packet.metadata,
                    ),
                )
                belief_target_indices, belief_matched_slots = target_matcher.match(
                    auxiliary_prior,
                    batch["objects"]["position"][:, frame_index],
                    batch["objects"]["active"][:, frame_index].bool(),
                )
                target_indices, matched_slots = _map_anchor_targets_to_predictions(
                    belief_target_indices,
                    belief_matched_slots,
                    predicted,
                )
                eligible_slots = _fast_pair_eligible_slots(
                    predicted,
                    target_indices,
                    matched_slots,
                )
                if bool(eligible_slots.any()):
                    fast_measurements, _ = module.encode_measurements(
                        [packet],
                        auxiliary_prior,
                        predicted,
                        auxiliary_cache,
                    )
                    valid_supervision = eligible_slots & fast_measurements.measurement_mask
                    if bool(valid_supervision.any()):
                        fast_supervised = supervised_slot_measurement_losses(
                            module,
                            fast_measurements,
                            batch,
                            frame_index,
                            target_indices=target_indices,
                            matched_slots=matched_slots,
                            roi_bounds=predicted.rois,
                        )
                        fast_trainable = (
                            torch.is_grad_enabled()
                            and _fast_measurement_has_trainable_perception_path(module)
                        )
                        fast_total = _weighted_measurement_total(
                            fast_supervised,
                            config.training.measurement_loss_weights,
                        )
                        add(
                            "fast_measurement" if fast_trainable else "frozen_fast_measurement",
                            fast_total if fast_trainable else fast_total.detach(),
                        )
                        for name, value in fast_supervised.items():
                            add(f"fast_{name}", value if fast_trainable else value.detach())
                        fast_supervised_frames += 1
                        fast_supervised_slots += int(valid_supervision.sum().detach().cpu())

        ingest_trace: DifferentiableIngestTrace | None = None
        if soft_association_enabled:
            belief, ingest_trace = model.ingest_with_trace(
                packet,
                prepared=prepared_propagation,
            )
        else:
            belief = model.ingest(
                packet,
                prepared=prepared_propagation,
            )
        indices, matched = target_matcher.match(
            belief,
            batch["objects"]["position"][:, frame_index],
            batch["objects"]["active"][:, frame_index].bool(),
        )
        aligned_position = gather_target_slots(
            batch["objects"]["position"][:, frame_index],
            indices,
        )
        aligned_velocity = gather_target_slots(
            batch["objects"]["velocity"][:, frame_index],
            indices,
        )
        if ingest_trace is not None:
            assert soft_association_temperature is not None
            soft_losses, soft_metrics = _soft_association_surrogate_losses(
                model,
                ingest_trace,
                aligned_target_position=aligned_position,
                aligned_target_velocity=aligned_velocity,
                matched_belief_slots=matched,
                temperature=soft_association_temperature,
            )
            for name, value in soft_losses.items():
                add(name, value)
            _accumulate_float_metrics(parameter_supervision_metrics, soft_metrics)
        distance_gated_matched = _distance_gate_physical_matches(
            belief.objects.position,
            aligned_position,
            matched,
        )
        runtime_observed = _runtime_observed_belief_slots(
            model,
            belief,
        )
        observed_runtime_id = _target_observed_runtime_ids(
            indices,
            distance_gated_matched,
            runtime_observed,
            belief.objects.object_id,
            target_count=target_count,
        )
        (
            last_observed_target_frame,
            identity_history_reset,
        ) = _reset_parameter_history_for_identity_change(
            last_observed_target_frame,
            last_observed_runtime_id,
            observed_runtime_id,
        )
        parameter_supervision_metrics["parameter_identity_history_reset_count"] = (
            parameter_supervision_metrics.get(
                "parameter_identity_history_reset_count",
                0.0,
            )
            + float(identity_history_reset.sum().detach().cpu())
        )
        parameter_supervision, last_observed_target_frame = _parameter_supervision_masks(
            batch,
            config,
            frame_index,
            indices=indices,
            # Slow physical parameters may only receive privileged simulator
            # labels after the real RGB runtime has both observed the slot and
            # localized it within the declared physical assignment gate.
            # A persistent-ID/Hungarian match alone can be arbitrarily far
            # away and is not causal evidence about drag or restitution.
            matched=distance_gated_matched,
            runtime_observed=runtime_observed,
            last_observed_target_frame=last_observed_target_frame,
        )
        last_observed_runtime_id = torch.where(
            observed_runtime_id >= 0,
            observed_runtime_id,
            last_observed_runtime_id,
        )
        _accumulate_float_metrics(
            parameter_supervision_metrics,
            parameter_supervision.detached_metrics(),
        )
        velocity_axis_support = _state_velocity_objective_axis_support(
            belief,
            matched,
        )
        matched_velocity_axes = matched.unsqueeze(-1).expand_as(belief.objects.velocity)
        state_velocity_objective_supported_coordinate_count += int(
            velocity_axis_support.sum().detach().cpu()
        )
        state_velocity_objective_excluded_coordinate_count += int(
            (matched_velocity_axes & ~velocity_axis_support).sum().detach().cpu()
        )
        current, indices, matched = _belief_state_losses(
            belief,
            batch,
            config,
            frame_index,
            indices=indices,
            matched=matched,
            parameter_supervision=parameter_supervision,
            velocity_axis_support=velocity_axis_support,
        )
        matched_count += int(matched.sum().detach().cpu())
        if "existence_belief" in current:
            existence_negative_support_count += int(
                (belief.objects.active.bool() & ~matched).sum().detach().cpu()
            )
        frame_target_count = int(batch["objects"]["active"][:, frame_index].sum().detach().cpu())
        frame_predicted_count = int(belief.objects.active.sum().detach().cpu())
        target_object_frames += frame_target_count
        predicted_object_frames += frame_predicted_count
        distance_gated_target_object_frames += frame_target_count
        distance_gated_predicted_object_frames += frame_predicted_count
        distance_gated_matched_count += int(distance_gated_matched.sum().detach().cpu())
        if collect_protected_objective_cells:
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                add_protected(
                    f"state_position_{axis_name}@current",
                    _protected_objective_cell(
                        belief.objects.position[..., axis_index],
                        aligned_position[..., axis_index],
                        matched,
                    ),
                )
                add_protected(
                    f"state_velocity_{axis_name}@current",
                    _protected_objective_cell(
                        belief.objects.velocity[..., axis_index],
                        aligned_velocity[..., axis_index],
                        velocity_axis_support[..., axis_index],
                    ),
                )
        state_physical: dict[str, float] = {}
        _add_squared_error_metrics(
            state_physical,
            prefix="physical_state_position",
            prediction=belief.objects.position,
            target=aligned_position,
            mask=matched,
        )
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            _add_squared_error_metrics(
                state_physical,
                prefix=f"physical_state_position_{axis_name}",
                prediction=belief.objects.position[..., axis_index],
                target=aligned_position[..., axis_index],
                mask=matched,
            )
        _add_squared_error_metrics(
            state_physical,
            prefix="physical_state_velocity",
            prediction=belief.objects.velocity,
            target=aligned_velocity,
            mask=matched,
        )
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            _add_squared_error_metrics(
                state_physical,
                prefix=f"physical_state_velocity_{axis_name}",
                prediction=belief.objects.velocity[..., axis_index],
                target=aligned_velocity[..., axis_index],
                mask=matched,
            )
        _add_gaussian_coverage_metrics(
            state_physical,
            prefix="physical_state_position_coverage90",
            mean=belief.objects.position,
            target=aligned_position,
            log_variance=belief.objects.fast_log_variance[..., :3],
            mask=matched,
        )
        _write_axis_partitioned_gaussian_calibration_metrics(
            state_physical,
            prefix="physical_state_position",
            mean=belief.objects.position,
            target=aligned_position,
            log_variance=belief.objects.fast_log_variance[..., :3],
            mask=matched,
        )
        _accumulate_float_metrics(physical_metrics, state_physical)
        target_ids = batch["objects"].get("id")
        if isinstance(target_ids, Tensor):
            # Identity metrics require an independent framewise association.
            # The locked training matcher above intentionally refuses to
            # relabel a swapped track and therefore cannot itself reveal an
            # ID switch.
            identity_indices, identity_matched = match_belief_to_targets(
                belief,
                batch["objects"]["position"][:, frame_index],
                batch["objects"]["active"][:, frame_index].bool(),
            )
            identity_distance_gated = _distance_gate_physical_matches(
                belief.objects.position,
                gather_target_slots(
                    batch["objects"]["position"][:, frame_index],
                    identity_indices,
                ),
                identity_matched,
            )
            switches, associations = _update_geometric_identity_metrics(
                belief,
                target_ids[:, frame_index],
                identity_indices,
                identity_matched,
                last_predicted_id_for_target,
            )
            identity_switches += switches
            object_frame_associations += associations
            switches, associations = _update_geometric_identity_metrics(
                belief,
                target_ids[:, frame_index],
                identity_indices,
                identity_distance_gated,
                last_distance_gated_predicted_id_for_target,
            )
            distance_gated_identity_switches += switches
            distance_gated_object_frame_associations += associations
        for name, value in current.items():
            add(name, value)
        if prior_belief is not None:
            correction_valid = _current_correction_objective_support(
                belief,
                prior_belief,
                matched,
            )
            prior_current_error = correction_error(
                prior_belief.objects.position,
                aligned_position,
            )
            posterior_current_error = correction_error(
                belief.objects.position,
                aligned_position,
            )
            prior_current_velocity_error = correction_error(
                prior_belief.objects.velocity,
                aligned_velocity,
            )
            posterior_current_velocity_error = correction_error(
                belief.objects.velocity,
                aligned_velocity,
            )
            if correction_valid.any():
                current_velocity_correction_supported_object_count += int(
                    correction_valid.sum().detach().cpu()
                )
                add(
                    "correction_current",
                    _correction_non_regression_loss(
                        belief.objects.position,
                        prior_belief.objects.position,
                        aligned_position,
                        correction_valid,
                        config,
                    ),
                )
                add(
                    "correction_current_velocity",
                    _correction_non_regression_loss(
                        belief.objects.velocity,
                        prior_belief.objects.velocity,
                        aligned_velocity,
                        correction_valid,
                        config,
                        coordinate_support=velocity_axis_support,
                    ),
                )
                current_correction_improvements.append(
                    float(
                        (prior_current_error - posterior_current_error)
                        .masked_select(correction_valid)
                        .mean()
                        .detach()
                        .cpu()
                    )
                )
                current_velocity_correction_improvements.append(
                    float(
                        (prior_current_velocity_error - posterior_current_velocity_error)
                        .masked_select(correction_valid)
                        .mean()
                        .detach()
                        .cpu()
                    )
                )
        rollout_result: _RolloutLossResult | None = None
        if score_rollout:
            if batch_validation_rollouts:
                deferred_validation_rollouts.append(
                    _DeferredValidationRollout(
                        frame_index=frame_index,
                        belief=belief.clone(),
                        indices=indices.clone(),
                        matched=matched.clone(),
                    )
                )
            else:
                rollout_result = _rollout_loss_result(
                    model,
                    belief,
                    batch,
                    config,
                    frame_index,
                    indices,
                    matched,
                    compute_event_loss=compute_event_loss,
                    collect_promotion_metrics=collect_promotion_metrics,
                    collect_protected_objective_cells=(collect_protected_objective_cells),
                )
                rollout_execution_posterior_call_count += 1
                for name, value in rollout_result.losses.items():
                    add(name, value)
                for name, value in rollout_result.protected_objective_cells.items():
                    add_protected(name, value)
                _accumulate_float_metrics(
                    physical_metrics,
                    rollout_result.physical_metrics,
                )

        if (
            prior_rollout is not None
            and prior_rollout_positions is not None
            and prior_rollout_velocities is not None
            and rollout_result is not None
            and rollout_result.positions is not None
            and frame_offsets
        ):
            deltas: list[Tensor] = []
            correction_losses: list[Tensor] = []
            velocity_deltas: list[Tensor] = []
            velocity_correction_losses: list[Tensor] = []
            for query_index, frame_offset in enumerate(frame_offsets):
                target_index = frame_index + frame_offset
                target_position = gather_target_slots(
                    batch["objects"]["position"][:, target_index], indices
                )
                target_velocity = gather_target_slots(
                    batch["objects"]["velocity"][:, target_index],
                    indices,
                )
                future_active = (
                    gather_target_slots(
                        batch["objects"]["active"][:, target_index].unsqueeze(-1),
                        indices,
                    )
                    .squeeze(-1)
                    .bool()
                )
                valid = matched & future_active & prior_belief.objects.active
                valid &= future_predictable_mask(
                    batch,
                    anchor_index=frame_index,
                    target_index=target_index,
                    target_indices=indices,
                )
                valid &= belief.objects.age_steps >= config.training.minimum_rollout_age_steps
                if valid.any():
                    future_velocity_correction_supported_object_horizon_count += int(
                        valid.sum().detach().cpu()
                    )
                    prior_error = correction_error(
                        prior_rollout_positions[:, query_index],
                        target_position,
                    )
                    posterior_error = correction_error(
                        rollout_result.positions[:, query_index],
                        target_position,
                    )
                    deltas.append((prior_error - posterior_error).masked_select(valid).mean())
                    correction_losses.append(
                        _correction_non_regression_loss(
                            rollout_result.positions[:, query_index],
                            prior_rollout_positions[:, query_index],
                            target_position,
                            valid,
                            config,
                        )
                    )
                    add(
                        rollout_horizon_loss_key(
                            "correction_future",
                            query_seconds[query_index],
                        ),
                        correction_losses[-1],
                    )
                    prior_velocity_error = correction_error(
                        prior_rollout_velocities[:, query_index],
                        target_velocity,
                    )
                    posterior_velocity_error = correction_error(
                        rollout_result.velocities[:, query_index],
                        target_velocity,
                    )
                    velocity_deltas.append(
                        (prior_velocity_error - posterior_velocity_error)
                        .masked_select(valid)
                        .mean()
                    )
                    velocity_correction_losses.append(
                        _correction_non_regression_loss(
                            rollout_result.velocities[:, query_index],
                            prior_rollout_velocities[:, query_index],
                            target_velocity,
                            valid,
                            config,
                        )
                    )
                    add(
                        rollout_horizon_loss_key(
                            "correction_future_velocity",
                            query_seconds[query_index],
                        ),
                        velocity_correction_losses[-1],
                    )
            if deltas:
                future_correction_improvements.append(
                    float(torch.stack(deltas).mean().detach().cpu())
                )
                add("correction_future", torch.stack(correction_losses).mean())
            if velocity_deltas:
                future_velocity_correction_improvements.append(
                    float(torch.stack(velocity_deltas).mean().detach().cpu())
                )
                add(
                    "correction_future_velocity",
                    torch.stack(velocity_correction_losses).mean(),
                )

        diagnostics = model.updater.last_diagnostics
        if diagnostics is not None and diagnostics.correction_norm.numel() > 0:
            add("correction_magnitude", diagnostics.correction_norm.mean())

        # Continue direct supervision during the closed-loop stage so global
        # discovery cannot drift while downstream rollout objectives train.
        if include_measurement_supervision and frame_index == window_start:
            module = model.observation_modules["rgb"]
            measurements = module.initialise_measurements(
                [packet],
                _observation_context(packet, config, training=True),
            )
            supervised = supervised_measurement_losses(module, measurements, batch, frame_index)
            global_measurement = _weighted_measurement_total(
                supervised,
                config.training.measurement_loss_weights,
            )
            global_trainable = torch.is_grad_enabled() and _global_measurement_has_trainable_path(
                module
            )
            add(
                "global_measurement" if global_trainable else "frozen_global_measurement",
                global_measurement,
            )
            for name, value in supervised.items():
                add(name, value)

        if (
            frame_index - window_start + 1
        ) % config.training.tbptt_steps == 0 and frame_index + 1 < window_stop:
            model.detach_state()

    for chunk_start in range(
        0,
        len(deferred_validation_rollouts),
        validation_rollout_anchor_batch_size,
    ):
        chunk = deferred_validation_rollouts[
            chunk_start : chunk_start + validation_rollout_anchor_batch_size
        ]
        execution = _batched_validation_rollout_loss_results(
            model,
            chunk,
            batch,
            config,
            collect_promotion_metrics=collect_promotion_metrics,
        )
        rollout_execution_batched_anchor_count += execution.batched_anchor_count
        rollout_execution_serial_fallback_anchor_count += execution.serial_fallback_anchor_count
        rollout_execution_posterior_call_count += execution.rollout_call_count
        for rollout_result in execution.results:
            for name, value in rollout_result.losses.items():
                add(name, value)
            _accumulate_float_metrics(
                physical_metrics,
                rollout_result.physical_metrics,
            )

    reference = rgb
    details = {name: _mean_losses(values, reference) for name, values in detail_lists.items()}
    protected_objective_cells = {
        name: _merge_protected_objective_cells(values)
        for name, values in protected_cell_lists.items()
    }
    global_measurement = details.pop("global_measurement", None)
    fast_measurement = details.pop("fast_measurement", None)
    if global_measurement is not None or fast_measurement is not None:
        details["measurement"] = _combine_measurement_objectives(
            global_measurement=global_measurement,
            fast_measurement=fast_measurement,
            fast_weight=config.training.fast_roi_pretrain_weight,
        )
    if global_measurement is not None:
        details["measurement_global"] = global_measurement
    if fast_measurement is not None:
        details["measurement_fast"] = fast_measurement
    details = _globally_weight_horizon_details(details, config, reference)
    details.update(_attention_node_complexity_details(model))
    details.update(_attention_node_activity_details(model))
    terms = _group_closed_loop_terms(details, reference)
    total = _weighted_closed_loop_total(terms, loss_weights)
    metrics = {name: float(value.detach().cpu()) for name, value in details.items()}
    metrics.update(event_weight_metrics)
    metrics["closed_loop_scenario_tail_objective_active"] = float(
        config.training.closed_loop_scenario_tail_fraction is not None
    )
    metrics["closed_loop_scenario_tail_fraction"] = float(
        config.training.closed_loop_scenario_tail_fraction or 0.0
    )
    metrics["closed_loop_uncertainty_standardized_error_gradient_cap"] = float(
        config.training.closed_loop_uncertainty_standardized_error_gradient_cap or 0.0
    )
    metrics.update(physical_metrics)
    metrics.update(parameter_supervision_metrics)
    metrics.update(
        {
            "window_start_frame": float(window_start),
            "window_stop_frame": float(window_stop),
            "rollout_anchor_min_frame": (
                float(min(rollout_anchor_frames)) if rollout_anchor_frames else math.nan
            ),
            "rollout_anchor_max_frame": (
                float(max(rollout_anchor_frames)) if rollout_anchor_frames else math.nan
            ),
            "rollout_anchor_count": float(len(rollout_anchor_frames)),
            "rollout_execution_batch_requested_anchor_count": float(
                len(deferred_validation_rollouts)
            ),
            "rollout_execution_batched_anchor_count": float(rollout_execution_batched_anchor_count),
            "rollout_execution_serial_fallback_anchor_count": float(
                rollout_execution_serial_fallback_anchor_count
            ),
            "rollout_execution_posterior_call_count": float(rollout_execution_posterior_call_count),
            "prior_future_correction_rollout_enabled": float(compute_future_correction),
            "event_loss_objective_graph_enabled": float(compute_event_loss),
            "promotion_metric_collection_enabled": float(collect_promotion_metrics),
            "rollout_anchor_candidate_count": float(
                sum(
                    frame_index >= config.training.minimum_rollout_age_steps
                    and bool(
                        _valid_rollout_offsets(
                            config,
                            frame_index,
                            total_frames,
                        )[0]
                    )
                    for frame_index in range(window_start, window_stop)
                )
            ),
            "physical_target_object_frames": float(target_object_frames),
            "physical_predicted_object_frames": float(predicted_object_frames),
            "physical_matched_object_frames": float(matched_count),
            "physical_identity_switches": float(identity_switches),
            "physical_object_frame_associations": float(object_frame_associations),
            "physical_distance_gated_target_object_frames": float(
                distance_gated_target_object_frames
            ),
            "physical_distance_gated_predicted_object_frames": float(
                distance_gated_predicted_object_frames
            ),
            "physical_distance_gated_matched_object_frames": float(distance_gated_matched_count),
            "physical_distance_gated_identity_switches": float(distance_gated_identity_switches),
            "physical_distance_gated_object_frame_associations": float(
                distance_gated_object_frame_associations
            ),
            "window_pair_collision_interval_count": float(
                batch["events"]["pair_collision"][:, window_start:window_stop]
                .bool()
                .flatten(start_dim=2)
                .any(dim=-1)
                .sum()
                .detach()
                .cpu()
            ),
            "window_ground_collision_object_count": float(
                batch["events"]["ground_collision"][:, window_start:window_stop]
                .bool()
                .sum()
                .detach()
                .cpu()
            ),
            "window_wall_collision_count": float(
                batch["events"]["wall_collision"][:, window_start:window_stop]
                .bool()
                .sum()
                .detach()
                .cpu()
            ),
            "window_external_actuation_object_count": float(
                batch["events"]["externally_actuated"][:, window_start:window_stop]
                .bool()
                .sum()
                .detach()
                .cpu()
            ),
        }
    )
    metrics["physical_current_target_coverage"] = (
        matched_count / target_object_frames if target_object_frames else 0.0
    )
    metrics["physical_current_prediction_coverage"] = (
        matched_count / predicted_object_frames if predicted_object_frames else 0.0
    )
    metrics["physical_identity_switch_rate"] = (
        identity_switches / object_frame_associations if object_frame_associations else 0.0
    )
    metrics["physical_current_distance_gated_target_coverage"] = (
        distance_gated_matched_count / distance_gated_target_object_frames
        if distance_gated_target_object_frames
        else 0.0
    )
    metrics["physical_current_distance_gated_prediction_precision"] = (
        distance_gated_matched_count / distance_gated_predicted_object_frames
        if distance_gated_predicted_object_frames
        else 0.0
    )
    metrics["physical_distance_gated_identity_switch_rate"] = (
        distance_gated_identity_switches / distance_gated_object_frame_associations
        if distance_gated_object_frame_associations
        else 0.0
    )
    state_position_count = metrics.get("physical_state_position_coordinate_count", 0.0)
    state_velocity_count = metrics.get("physical_state_velocity_coordinate_count", 0.0)
    metrics["physical_state_position_rmse_m"] = (
        math.sqrt(metrics["physical_state_position_sse"] / state_position_count)
        if state_position_count
        else math.nan
    )
    metrics["physical_state_velocity_rmse_mps"] = (
        math.sqrt(metrics["physical_state_velocity_sse"] / state_velocity_count)
        if state_velocity_count
        else math.nan
    )
    for axis_name in ("x", "y", "z"):
        axis_count = metrics.get(
            f"physical_state_velocity_{axis_name}_coordinate_count",
            0.0,
        )
        metrics[f"physical_state_velocity_{axis_name}_rmse_mps"] = (
            math.sqrt(metrics[f"physical_state_velocity_{axis_name}_sse"] / axis_count)
            if axis_count
            else math.nan
        )
    collision_true_positive = sum(
        value
        for name, value in physical_metrics.items()
        if name.startswith("physical_collision_true_positive_count@")
    )
    collision_false_positive = sum(
        value
        for name, value in physical_metrics.items()
        if name.startswith("physical_collision_false_positive_count@")
    )
    collision_false_negative = sum(
        value
        for name, value in physical_metrics.items()
        if name.startswith("physical_collision_false_negative_count@")
    )
    collision_true_negative = sum(
        value
        for name, value in physical_metrics.items()
        if name.startswith("physical_collision_true_negative_count@")
    )
    collision_f1, collision_f1_denominator = _f1_from_confusion(
        collision_true_positive,
        collision_false_positive,
        collision_false_negative,
    )
    metrics.update(
        {
            "physical_collision_true_positive_count": collision_true_positive,
            "physical_collision_false_positive_count": collision_false_positive,
            "physical_collision_false_negative_count": collision_false_negative,
            "physical_collision_true_negative_count": collision_true_negative,
            "physical_collision_f1_denominator": collision_f1_denominator,
            "physical_collision_f1_proxy": collision_f1,
        }
    )
    for seconds in {
        frame_offset / config.simulator.frame_rate
        for frame_offset in {
            max(1, int(round(float(horizon) * config.simulator.frame_rate)))
            for horizon in config.evaluation.horizons_seconds
        }
    }:
        horizon_suffix = f"@{seconds:.3f}s"
        position_count = metrics.get(
            f"physical_rollout_position{horizon_suffix}_coordinate_count",
            0.0,
        )
        velocity_count = metrics.get(
            f"physical_rollout_velocity{horizon_suffix}_coordinate_count",
            0.0,
        )
        if position_count:
            metrics[f"physical_rollout_position_rmse_m{horizon_suffix}"] = math.sqrt(
                metrics[f"physical_rollout_position{horizon_suffix}_sse"] / position_count
            )
        else:
            metrics[f"physical_rollout_position_rmse_m{horizon_suffix}"] = math.nan
        if velocity_count:
            metrics[f"physical_rollout_velocity_rmse_mps{horizon_suffix}"] = math.sqrt(
                metrics[f"physical_rollout_velocity{horizon_suffix}_sse"] / velocity_count
            )
        else:
            metrics[f"physical_rollout_velocity_rmse_mps{horizon_suffix}"] = math.nan
        for axis_name in ("x", "y", "z"):
            axis_count = metrics.get(
                f"physical_rollout_velocity_{axis_name}{horizon_suffix}_coordinate_count",
                0.0,
            )
            metrics[f"physical_rollout_velocity_{axis_name}_rmse_mps{horizon_suffix}"] = (
                math.sqrt(
                    metrics[f"physical_rollout_velocity_{axis_name}{horizon_suffix}_sse"]
                    / axis_count
                )
                if axis_count
                else math.nan
            )
        target_count = metrics.get(
            f"physical_forecast_target_count{horizon_suffix}",
            0.0,
        )
        active_count = metrics.get(
            f"physical_forecast_active_count{horizon_suffix}",
            0.0,
        )
        if target_count:
            metrics[f"physical_forecast_target_coverage{horizon_suffix}"] = (
                active_count / target_count
            )
        else:
            metrics[f"physical_forecast_target_coverage{horizon_suffix}"] = 0.0
        horizon_true_positive = metrics.get(
            f"physical_collision_true_positive_count{horizon_suffix}",
            0.0,
        )
        horizon_false_positive = metrics.get(
            f"physical_collision_false_positive_count{horizon_suffix}",
            0.0,
        )
        horizon_false_negative = metrics.get(
            f"physical_collision_false_negative_count{horizon_suffix}",
            0.0,
        )
        horizon_f1, horizon_f1_denominator = _f1_from_confusion(
            horizon_true_positive,
            horizon_false_positive,
            horizon_false_negative,
        )
        metrics[f"physical_collision_f1_proxy{horizon_suffix}"] = horizon_f1
        metrics[f"physical_collision_f1_denominator{horizon_suffix}"] = horizon_f1_denominator
        if collect_promotion_metrics:
            identity_association_count = metrics.get(
                f"physical_forecast_identity_association_count{horizon_suffix}",
                0.0,
            )
            identity_mismatch_count = metrics.get(
                f"physical_forecast_identity_mismatch_count{horizon_suffix}",
                0.0,
            )
            metrics[f"physical_forecast_identity_mismatch_rate{horizon_suffix}"] = (
                identity_mismatch_count / identity_association_count
                if identity_association_count
                else math.nan
            )
    rollout_coverage_hits = sum(
        value
        for name, value in physical_metrics.items()
        if name.startswith("physical_rollout_position_coverage90@") and name.endswith("_hit_count")
    )
    rollout_coverage_count = sum(
        value
        for name, value in physical_metrics.items()
        if name.startswith("physical_rollout_position_coverage90@")
        and name.endswith("_coordinate_count")
    )
    metrics["physical_position_coverage90_hit_count"] = rollout_coverage_hits
    metrics["physical_position_coverage90_coordinate_count"] = rollout_coverage_count
    metrics["physical_position_coverage90"] = (
        rollout_coverage_hits / rollout_coverage_count if rollout_coverage_count else 0.0
    )
    metrics.update(
        {
            "matched_object_frames": float(matched_count),
            "existence_negative_supervision_object_frames": float(existence_negative_support_count),
            "state_velocity_objective_supported_coordinate_count": float(
                state_velocity_objective_supported_coordinate_count
            ),
            "state_velocity_objective_excluded_coordinate_count": float(
                state_velocity_objective_excluded_coordinate_count
            ),
            "correction_current_velocity_objective_supported_object_count": float(
                current_velocity_correction_supported_object_count
            ),
            "correction_future_velocity_objective_supported_object_horizon_count": float(
                future_velocity_correction_supported_object_horizon_count
            ),
            "perturbed_updates": float(perturbed_updates),
            "fast_path_supervised": float(fast_supervised_frames > 0),
            "fast_supervised_frames": float(fast_supervised_frames),
            "fast_supervised_slots": float(fast_supervised_slots),
            "current_correction_improvement_m": (
                float(sum(current_correction_improvements) / len(current_correction_improvements))
                if current_correction_improvements
                else math.nan
            ),
            "current_velocity_correction_improvement_mps": (
                float(
                    sum(current_velocity_correction_improvements)
                    / len(current_velocity_correction_improvements)
                )
                if current_velocity_correction_improvements
                else math.nan
            ),
            "future_correction_improvement_m": (
                float(sum(future_correction_improvements) / len(future_correction_improvements))
                if future_correction_improvements
                else math.nan
            ),
            "future_velocity_correction_improvement_mps": (
                float(
                    sum(future_velocity_correction_improvements)
                    / len(future_velocity_correction_improvements)
                )
                if future_velocity_correction_improvements
                else math.nan
            ),
            "correction_improvement_m": (
                float(sum(future_correction_improvements) / len(future_correction_improvements))
                if future_correction_improvements
                else math.nan
            ),
        }
    )
    return TrainingBatchResult(
        total_loss=total,
        loss_terms=terms,
        metrics=metrics,
        phase="closed_loop_rgb",
        support_terms={
            **({"fast_measurement": fast_measurement} if fast_measurement is not None else {}),
            **(
                {"event_collision_node": details["event_collision_node"]}
                if "event_collision_node" in details
                else {}
            ),
            **(
                {"event_collision_pair": details["event_collision_pair"]}
                if "event_collision_pair" in details
                else {}
            ),
            **_event_horizon_support_terms(details),
        },
        protected_objective_cells=protected_objective_cells,
    )


__all__ = [
    "PhysicalMetricSupportError",
    "TrainingBatchResult",
    "future_predictable_mask",
    "future_scene_predictable_mask",
    "gather_target_slots",
    "make_rgb_packet",
    "match_belief_to_targets",
    "measurement_localization_metrics",
    "move_batch_to_device",
    "physical_validation_metrics",
    "pretrain_rgb_measurements",
    "run_closed_loop_batch",
    "select_closed_loop_window",
    "supervised_measurement_losses",
    "supervised_slot_measurement_losses",
]
