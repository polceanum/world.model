"""Held-out RGB-only evaluation with transparent physical baselines."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from world_model.belief import BeliefTrajectory, MotionMode, WorldBelief
from world_model.datasets import SyntheticSphereDataset, collate_episodes
from world_model.evaluation.baselines import baseline_bundle
from world_model.evaluation.collision_conditioned import (
    CollisionConditionedForecastAccumulator,
    collision_class_masks_for_forecast_window,
)
from world_model.evaluation.latency import synchronize
from world_model.evaluation.occlusion_metrics import (
    OcclusionTransitionAccumulator,
)
from world_model.evaluation.parameter_metrics import OnlineParameterUpdateAccumulator
from world_model.evaluation.reports import write_evaluation_report
from world_model.evaluation.seed_protocol import (
    STANDARD_SEED_PROTOCOL,
    make_evaluation_seed_protocol,
)
from world_model.evaluation.velocity_metrics import (
    MaskedVelocityErrorAccumulator,
    OrdinaryVelocityCorrectionAccumulator,
    TemporalVelocityMeasurementAccumulator,
)
from world_model.identification import ParameterUpdateDiagnostics
from world_model.runtime import OnlineWorldModel
from world_model.simulator.sphere_world import SphereWorldConfig
from world_model.training.checkpointing import (
    CapturedCheckpoint as _CapturedCheckpoint,
)
from world_model.training.checkpointing import (
    capture_checkpoint_snapshot as _capture_checkpoint_snapshot,
)
from world_model.training.checkpointing import (
    capture_git_metadata,
    load_checkpoint,
)
from world_model.training.event_windows import (
    ObservationWindowQueryPlan,
    observation_window_query_plan,
)
from world_model.training.loop import (
    _match_positions_to_targets,
    future_predictable_mask,
    future_scene_predictable_mask,
    gather_target_slots,
    make_rgb_packet,
    match_belief_to_targets,
    move_batch_to_device,
)
from world_model.training.perturbations import perturb_belief
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import OrpheusConfig
from world_model.utils.device import DeviceInfo, select_device
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION

_IDENTIFIER_PARAMETERS = ("mass", "restitution", "drag", "friction", "radius")
_CURRENT_DETECTION_DISTANCE_THRESHOLD_M = 0.5
_EVALUATION_PROTOCOL_SCHEMA_VERSION = "held_out_rgb_online_v2"
_EVALUATION_METRIC_SCHEMA_VERSION = "held_out_rgb_metrics_v3"
_PER_SCENARIO_METRIC_SCHEMA = "clean_primary_additive_support_diagnostic_v3"
_RECOVERY_ONLY_METRIC_NAMES = frozenset(
    {
        "perturbation_prior_position_error_m",
        "perturbation_posterior_position_error_m",
        "perturbation_correction_improvement_m",
        "perturbation_correction_improvement_fraction",
        "perturbation_positive_correction_rate",
        "perturbation_evaluated_object_horizons",
        "injected_perturbation_batch_updates",
        "recovery_probe_evaluated_episodes",
        "recovery_probe_nonfinite_output_count",
        "recovery_probe_post_observation_std_contraction_mean_m",
        "post_observation_std_contraction_mean_m",
    }
)
_PRIMARY_PHYSICAL_LATENCY_NAME_SUBSTRING = "latency"
_RUNTIME_HYPOTHESIS_CANDIDATES = (
    "learned",
    "constant_velocity",
    "damped_constant_velocity",
    "ballistic_contact",
)
_RUNTIME_HYPOTHESIS_REGIMES = (
    "free",
    "ground_contact",
    "pair_contact",
    "collision",
    "occluded",
    "externally_actuated",
)


def _runtime_hypothesis_candidate_names(config: OrpheusConfig) -> tuple[str, ...]:
    """Return the exact evaluator candidate schema for the resolved policy."""

    return _RUNTIME_HYPOTHESIS_CANDIDATES + (
        ("online_local_acceleration",)
        if config.runtime.hypothesis_online_acceleration_enabled
        else ()
    )


def _primary_physical_metrics_hash_exclusion_declaration() -> dict[str, object]:
    """Describe every final-report metric excluded from the primary digest."""

    return {
        "latency_metric_name_substring": _PRIMARY_PHYSICAL_LATENCY_NAME_SUBSTRING,
        "recovery_only_metric_names": sorted(_RECOVERY_ONLY_METRIC_NAMES),
    }


def _primary_physical_metrics(
    metrics: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Return the exact clean-primary metric scope covered by its digest.

    Recovery-probe metrics are appended to the public report only after the
    clean primary digest is frozen.  Keeping this derivation explicit lets a
    verifier reconstruct that same scope from the final report without either
    admitting timing data or accidentally requiring the later probe fields.
    """

    return {
        name: value
        for name, value in metrics.items()
        if _PRIMARY_PHYSICAL_LATENCY_NAME_SUBSTRING not in name
        and name not in _RECOVERY_ONLY_METRIC_NAMES
    }


def enable_runtime_hypothesis_pool(
    model: OnlineWorldModel,
    config: OrpheusConfig,
) -> None:
    """Attach the explicitly requested evaluation-only runtime policy.

    Checkpoints are loaded against their original runtime configuration first.
    This post-load attachment is therefore an auditable evaluation intervention,
    not a weakened checkpoint-semantic comparison or a hidden model transfer.
    It owns no learnable state and remains outside ``WorldBelief``.
    """

    from world_model.dynamics import (
        BallisticContactDynamics,
        ConstantVelocityDynamics,
        HypothesisDynamicsPool,
        OnlineLocalAccelerationDynamics,
        RuntimeHypothesisController,
    )

    runtime = config.runtime
    candidates: list[object] = [
        model.dynamics,
        ConstantVelocityDynamics(),
        ConstantVelocityDynamics(damping=0.05),
        BallisticContactDynamics(),
    ]
    if runtime.hypothesis_online_acceleration_enabled:
        candidates.append(
            OnlineLocalAccelerationDynamics(
                minimum_support_count=(
                    runtime.hypothesis_online_acceleration_minimum_support_count
                ),
                maximum_acceleration=runtime.hypothesis_online_acceleration_maximum_mps2,
                minimum_delta_time=config.model.rgb.temporal_velocity_min_dt,
            )
        )
    model.hypothesis_controller = RuntimeHypothesisController(
        HypothesisDynamicsPool(
            tuple(candidates),
            evidence_decay=runtime.hypothesis_evidence_decay,
        ),
        evidence_horizons_seconds=runtime.hypothesis_evidence_horizons_seconds,
        axis_independent_axes=runtime.hypothesis_axis_independent_axes,
        axis_prior_strength=runtime.hypothesis_axis_prior_strength,
        timestamp_tolerance_seconds=runtime.hypothesis_timestamp_tolerance_seconds,
        local_applicability_enabled=runtime.hypothesis_local_applicability_enabled,
        minimum_support_count=runtime.hypothesis_minimum_support_count,
        maximum_evidence_age_seconds=runtime.hypothesis_maximum_evidence_age_seconds,
        minimum_observability=runtime.hypothesis_minimum_observability,
        minimum_confidence_margin=runtime.hypothesis_minimum_confidence_margin,
        velocity_evidence_weight=runtime.hypothesis_velocity_evidence_weight,
        velocity_nonregression_gate_enabled=(
            runtime.hypothesis_velocity_nonregression_gate_enabled
        ),
        residual_correction_gain_by_axis=(runtime.hypothesis_residual_correction_gain_by_axis),
        robust_influence_delta=runtime.hypothesis_robust_influence_delta,
        composition_step_seconds=runtime.hypothesis_composition_step_seconds,
    )


@dataclass
class _ErrorAccumulator:
    squared_sum: float = 0.0
    absolute_sum: float = 0.0
    count: int = 0
    axis_squared_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_absolute_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_count: list[int] = field(default_factory=lambda: [0, 0, 0])

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        if prediction.shape != target.shape or prediction.shape[-1] != 3:
            raise ValueError("position error inputs must have matching [...,3] shapes")
        expanded = mask
        while expanded.ndim < prediction.ndim:
            expanded = expanded.unsqueeze(-1)
        residual = prediction - target
        values = residual.masked_select(expanded.expand_as(prediction))
        if values.numel() == 0:
            return
        detached = values.detach().float().cpu()
        self.squared_sum += float(detached.square().sum())
        self.absolute_sum += float(detached.abs().sum())
        self.count += int(detached.numel())
        for axis in range(3):
            axis_values = residual[..., axis].masked_select(mask).detach().float().cpu()
            self.axis_squared_sum[axis] += float(axis_values.square().sum())
            self.axis_absolute_sum[axis] += float(axis_values.abs().sum())
            self.axis_count[axis] += int(axis_values.numel())

    def metrics(self, prefix: str) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {}
        for axis, label in enumerate(("x", "y", "z")):
            count = self.axis_count[axis]
            metrics.update(
                {
                    f"{prefix}_position_{label}_rmse_m": (
                        math.sqrt(self.axis_squared_sum[axis] / count) if count else None
                    ),
                    f"{prefix}_position_{label}_mae_m": (
                        self.axis_absolute_sum[axis] / count if count else None
                    ),
                    f"{prefix}_position_{label}_sse": self.axis_squared_sum[axis],
                    f"{prefix}_position_{label}_absolute_error_sum_m": (
                        self.axis_absolute_sum[axis]
                    ),
                    f"{prefix}_position_{label}_count": float(count),
                }
            )
        if self.count == 0:
            metrics.update(
                {
                    f"{prefix}_position_rmse_m": None,
                    f"{prefix}_position_mae_m": None,
                    f"{prefix}_position_sse": self.squared_sum,
                    f"{prefix}_position_absolute_error_sum_m": self.absolute_sum,
                    f"{prefix}_position_coordinate_count": 0.0,
                }
            )
            return metrics
        metrics.update(
            {
                f"{prefix}_position_rmse_m": math.sqrt(self.squared_sum / self.count),
                f"{prefix}_position_mae_m": self.absolute_sum / self.count,
                f"{prefix}_position_sse": self.squared_sum,
                f"{prefix}_position_absolute_error_sum_m": self.absolute_sum,
                f"{prefix}_position_coordinate_count": float(self.count),
            }
        )
        return metrics


@dataclass
class _CorrectionAccumulator:
    prior_error_sum: float = 0.0
    posterior_error_sum: float = 0.0
    positive: int = 0
    count: int = 0

    def update(
        self,
        prior: Tensor,
        posterior: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        prior_error = torch.linalg.vector_norm(prior - target, dim=-1)
        posterior_error = torch.linalg.vector_norm(posterior - target, dim=-1)
        prior_values = prior_error.masked_select(mask).detach().float().cpu()
        posterior_values = posterior_error.masked_select(mask).detach().float().cpu()
        if prior_values.numel() == 0:
            return
        delta = prior_values - posterior_values
        self.prior_error_sum += float(prior_values.sum())
        self.posterior_error_sum += float(posterior_values.sum())
        self.positive += int((delta > 0).sum())
        self.count += int(delta.numel())

    def metrics(self) -> dict[str, float | None]:
        if self.count == 0:
            return {
                "perturbation_prior_position_error_m": None,
                "perturbation_posterior_position_error_m": None,
                "perturbation_correction_improvement_m": None,
                "perturbation_correction_improvement_fraction": None,
                "perturbation_positive_correction_rate": None,
                "perturbation_evaluated_object_horizons": 0.0,
            }
        prior = self.prior_error_sum / self.count
        posterior = self.posterior_error_sum / self.count
        improvement = prior - posterior
        return {
            "perturbation_prior_position_error_m": prior,
            "perturbation_posterior_position_error_m": posterior,
            "perturbation_correction_improvement_m": improvement,
            "perturbation_correction_improvement_fraction": (improvement / max(prior, 1.0e-8)),
            "perturbation_positive_correction_rate": self.positive / self.count,
            "perturbation_evaluated_object_horizons": float(self.count),
        }


@dataclass
class _RecoveryProbeResult:
    """Metrics produced only by an isolated recovery-prefix replay."""

    correction: _CorrectionAccumulator = field(default_factory=_CorrectionAccumulator)
    perturbation_updates: int = 0
    evaluated_episodes: int = 0
    nonfinite_outputs: int = 0
    uncertainty_contraction: list[float] = field(default_factory=list)

    def metrics(self) -> dict[str, float | None]:
        contraction = _mean_or_none(self.uncertainty_contraction)
        return {
            **self.correction.metrics(),
            "injected_perturbation_batch_updates": float(self.perturbation_updates),
            "recovery_probe_evaluated_episodes": float(self.evaluated_episodes),
            "recovery_probe_nonfinite_output_count": float(self.nonfinite_outputs),
            "recovery_probe_post_observation_std_contraction_mean_m": contraction,
            # Backwards-compatible name; this has always been populated only
            # by the injected recovery update, never ordinary observations.
            "post_observation_std_contraction_mean_m": contraction,
        }


@dataclass
class _PosteriorTraceHasher:
    """Hash the clean primary posterior sequence without retaining device tensors."""

    _digest: Any = field(default_factory=hashlib.sha256)
    frame_count: int = 0

    @staticmethod
    def _named_tensors(belief: WorldBelief) -> tuple[tuple[str, Tensor], ...]:
        tensors: list[tuple[str, Tensor]] = [
            ("timestamp", belief.timestamp),
            ("gravity", belief.gravity),
            ("global_code", belief.global_code),
            ("global_log_variance", belief.global_log_variance),
            ("next_object_id", belief.next_object_id),
        ]
        tensors.extend(
            (f"objects.{name}", value)
            for name, value in vars(belief.objects).items()
            if isinstance(value, Tensor)
        )
        tensors.extend(
            (f"camera.{name}", value)
            for name, value in vars(belief.camera).items()
            if isinstance(value, Tensor)
        )
        return tuple(tensors)

    def update(self, *, batch_index: int, frame_index: int, belief: WorldBelief) -> None:
        self._digest.update(f"batch={batch_index};frame={frame_index};".encode("ascii"))
        for name, value in self._named_tensors(belief):
            detached = value.detach().cpu()
            self._digest.update(name.encode("utf-8"))
            self._digest.update(str(detached.dtype).encode("ascii"))
            self._digest.update(str(tuple(detached.shape)).encode("ascii"))
            self._digest.update(detached.numpy().tobytes(order="C"))
        self.frame_count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _require_finite_belief(belief: WorldBelief, *, context: str) -> None:
    """Fail before any metric or trace consumes a nonfinite belief tensor."""

    for name, value in _PosteriorTraceHasher._named_tensors(belief):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise FloatingPointError(f"{context} belief tensor {name} contains NaN or Inf")


def _require_finite_trajectory(
    trajectory: BeliefTrajectory,
    *,
    context: str,
) -> None:
    """Fail before accumulators consume any nonfinite rollout tensor."""

    named: list[tuple[str, Tensor]] = [
        (name, value) for name, value in vars(trajectory).items() if isinstance(value, Tensor)
    ]
    named.extend(
        (f"auxiliary.{name}", value)
        for name, value in trajectory.auxiliary.items()
        if isinstance(value, Tensor)
    )
    for name, value in named:
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise FloatingPointError(f"{context} trajectory tensor {name} contains NaN or Inf")


def _validate_runtime_hypothesis_composition_counts(
    *,
    local_shape: tuple[int, ...],
    candidate_count: Tensor,
    fallback_count: Tensor,
    total_count: Tensor,
    regime_count: Tensor,
    independent_axes: tuple[int, ...],
    candidate_size: int,
    regime_size: int,
) -> None:
    expected_candidate_shape = (*local_shape, candidate_size)
    if (
        candidate_count.shape != expected_candidate_shape
        or candidate_count.dtype != torch.int64
        or torch.any(candidate_count < 0)
    ):
        raise RuntimeError("runtime hypothesis composed candidate counts are invalid")
    for name, value in (("fallback", fallback_count), ("total", total_count)):
        if value.shape != local_shape or value.dtype != torch.int64 or torch.any(value < 0):
            raise RuntimeError(f"runtime hypothesis composed {name} counts are invalid")
    if not torch.equal(candidate_count.sum(dim=-1), total_count):
        raise RuntimeError(
            "runtime hypothesis candidate counts must partition total composed steps"
        )
    if torch.any(fallback_count > candidate_count[..., 0]):
        raise RuntimeError(
            "runtime hypothesis fallback count must be contained in the learned-candidate count"
        )
    expected_regime_shape = (local_shape[0], local_shape[1], local_shape[2], regime_size)
    if (
        regime_count.shape != expected_regime_shape
        or regime_count.dtype != torch.int64
        or torch.any(regime_count < 0)
    ):
        raise RuntimeError("runtime hypothesis composed regime counts are invalid")
    regime_total = regime_count.sum(dim=-1)
    for axis in independent_axes:
        if not torch.equal(regime_total, total_count[..., axis]):
            raise RuntimeError(
                "runtime hypothesis regime counts must partition total composed steps"
            )


def _sum_runtime_hypothesis_counts_on_host(
    value: Tensor,
    *,
    dim: tuple[int, ...],
) -> Tensor:
    """Reduce detached integer diagnostics on the host.

    The custom Aqua-MPS backend can abort while compiling multi-axis integer
    reductions for the batched hypothesis evaluator. These values are report
    counters only and are immediately consumed on the host, so transferring
    before the exact integer reduction avoids that backend kernel without
    changing model execution or floating-point metrics.
    """

    if value.dtype != torch.int64:
        raise TypeError("runtime hypothesis diagnostic counts must use torch.int64")
    return value.detach().cpu().sum(dim=dim)


def _runtime_hypothesis_learned_fallback_diagnostics(
    controller: object,
    belief: WorldBelief,
    trajectory: BeliefTrajectory,
) -> tuple[Tensor, Tensor, Tensor]:
    """Describe a learned fallback before any delayed selector evidence exists."""

    classify_regime = getattr(controller, "_trajectory_regime", None)
    if not callable(classify_regime):
        raise RuntimeError("runtime hypothesis controller cannot classify interaction regimes")
    axis_indices = torch.zeros(
        (*trajectory.active_mask.shape, 3),
        device=belief.device,
        dtype=torch.int64,
    )
    axis_supported = torch.zeros_like(axis_indices, dtype=torch.bool)
    interaction_regime = classify_regime(belief, trajectory)
    return axis_indices, axis_supported, interaction_regime


def _require_finite_measurements(measurements: Any, *, context: str) -> None:
    """Validate measurement diagnostics before metric extraction."""

    named = [
        (name, value) for name, value in vars(measurements).items() if isinstance(value, Tensor)
    ]
    auxiliary = getattr(measurements, "auxiliary", {})
    if isinstance(auxiliary, Mapping):
        named.extend(
            (f"auxiliary.{name}", value)
            for name, value in auxiliary.items()
            if isinstance(value, Tensor)
        )
    for name, value in named:
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise FloatingPointError(f"{context} measurement tensor {name} contains NaN or Inf")


def _require_finite_diagnostics(diagnostics: Any, *, context: str) -> None:
    """Validate tensor diagnostics before their lossy scalar accumulation."""

    for name, value in vars(diagnostics).items():
        if (
            isinstance(value, Tensor)
            and (value.is_floating_point() or value.is_complex())
            and not bool(torch.isfinite(value).all())
        ):
            raise FloatingPointError(f"{context} diagnostic tensor {name} contains NaN or Inf")


def _recovery_persistent_support(
    prior: WorldBelief,
    posterior: WorldBelief,
    posterior_match: Tensor,
) -> Tensor:
    """Support recovery only where one persistent entity spans both states."""

    if posterior_match.shape != prior.objects.active.shape:
        raise ValueError("recovery posterior match must have shape [B,N]")
    return (
        posterior_match
        & prior.objects.active
        & posterior.objects.active
        & (prior.objects.object_id >= 0)
        & (prior.objects.object_id == posterior.objects.object_id)
    )


def _require_finite_metrics(metrics: Mapping[str, Any]) -> None:
    """Reject nonfinite public metrics before hashing or serialization."""

    for name, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"evaluation metric {name!r} must be numeric or null")
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"evaluation metric {name!r} is nonfinite: {value!r}")


@dataclass
class _BinaryAccumulator:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        logits: Tensor,
        target: Tensor,
        mask: Tensor,
        *,
        threshold: float = 0.5,
    ) -> None:
        probability = logits.sigmoid().masked_select(mask).detach().cpu()
        truth = target.bool().masked_select(mask).detach().cpu()
        if probability.numel() == 0:
            return
        prediction = probability >= threshold
        self.true_positive += int((prediction & truth).sum())
        self.false_positive += int((prediction & ~truth).sum())
        self.false_negative += int((~prediction & truth).sum())
        self.true_negative += int((~prediction & ~truth).sum())

    def metrics(self, prefix: str) -> dict[str, float | None]:
        evaluated = (
            self.true_positive + self.false_positive + self.false_negative + self.true_negative
        )
        confusion = {
            f"{prefix}_true_positive_count": float(self.true_positive),
            f"{prefix}_false_positive_count": float(self.false_positive),
            f"{prefix}_false_negative_count": float(self.false_negative),
            f"{prefix}_true_negative_count": float(self.true_negative),
            f"{prefix}_f1_denominator": float(
                2 * self.true_positive + self.false_positive + self.false_negative
            ),
        }
        if evaluated == 0:
            return {
                **confusion,
                f"{prefix}_precision": None,
                f"{prefix}_recall": None,
                f"{prefix}_f1": None,
                f"{prefix}_false_positive_rate": None,
                f"{prefix}_evaluated": 0.0,
            }
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = self.true_positive / precision_denominator if precision_denominator else 0.0
        recall = self.true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        false_positive_denominator = self.false_positive + self.true_negative
        return {
            **confusion,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
            f"{prefix}_false_positive_rate": (
                self.false_positive / false_positive_denominator
                if false_positive_denominator
                else 0.0
            ),
            f"{prefix}_evaluated": float(evaluated),
        }


@dataclass
class _CalibrationAccumulator:
    error: list[Tensor] = field(default_factory=list)
    standard_deviation: list[Tensor] = field(default_factory=list)
    axis_error: list[list[Tensor]] = field(default_factory=lambda: [[], [], []])
    axis_standard_deviation: list[list[Tensor]] = field(default_factory=lambda: [[], [], []])

    def update(
        self,
        mean: Tensor,
        log_variance: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        expanded = mask
        while expanded.ndim < mean.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(mean)
        error = (mean - target).masked_select(expanded)
        standard_deviation = (0.5 * log_variance.clamp(-12.0, 8.0)).exp().masked_select(expanded)
        if error.numel() > 0:
            self.error.append(error.detach().float().cpu())
            self.standard_deviation.append(standard_deviation.detach().float().cpu())
        if mean.ndim == mask.ndim + 1 and mean.shape[-1] == 3:
            for axis in range(3):
                axis_error = (mean[..., axis] - target[..., axis]).masked_select(mask)
                axis_std = (0.5 * log_variance[..., axis].clamp(-12.0, 8.0)).exp()
                axis_std = axis_std.masked_select(mask)
                if axis_error.numel() > 0:
                    self.axis_error[axis].append(axis_error.detach().float().cpu())
                    self.axis_standard_deviation[axis].append(axis_std.detach().float().cpu())

    @staticmethod
    def _metrics_from_values(
        error_values: list[Tensor],
        standard_deviation_values: list[Tensor],
        *,
        prefix: str,
    ) -> dict[str, float | None]:
        if not error_values:
            return {
                f"{prefix}_gaussian_nll": None,
                f"{prefix}_gaussian_nll_sum": 0.0,
                f"{prefix}_sharpness_std": None,
                f"{prefix}_sharpness_std_sum": 0.0,
                f"{prefix}_calibration_coordinate_count": 0.0,
                f"{prefix}_coverage_50": None,
                f"{prefix}_coverage_80": None,
                f"{prefix}_coverage_90": None,
                f"{prefix}_calibration_error90": None,
                f"{prefix}_coverage_95": None,
            }
        error = torch.cat(error_values)
        std = torch.cat(standard_deviation_values).clamp_min(1.0e-8)
        variance = std.square()
        nll = 0.5 * (error.square() / variance + variance.log() + math.log(2.0 * math.pi))
        z = error.abs() / std
        quantiles = {
            50: 0.67448975,
            80: 1.28155157,
            90: 1.64485363,
            95: 1.95996398,
        }
        metrics: dict[str, float | None] = {
            f"{prefix}_gaussian_nll": float(nll.mean()),
            f"{prefix}_gaussian_nll_sum": float(nll.sum()),
            f"{prefix}_sharpness_std": float(std.mean()),
            f"{prefix}_sharpness_std_sum": float(std.sum()),
            f"{prefix}_calibration_coordinate_count": float(error.numel()),
        }
        for level, quantile in quantiles.items():
            metrics[f"{prefix}_coverage_{level}"] = float((z <= quantile).float().mean())
        coverage90 = metrics[f"{prefix}_coverage_90"]
        if coverage90 is None:
            raise AssertionError("supported calibration metrics must include 90% coverage")
        metrics[f"{prefix}_calibration_error90"] = abs(coverage90 - 0.90)
        return metrics

    def metrics(
        self,
        prefix: str = "forecast",
        *,
        include_axes: bool = False,
    ) -> dict[str, float | None]:
        metrics = self._metrics_from_values(
            self.error,
            self.standard_deviation,
            prefix=prefix,
        )
        if include_axes:
            for axis, label in enumerate(("x", "y", "z")):
                metrics.update(
                    self._metrics_from_values(
                        self.axis_error[axis],
                        self.axis_standard_deviation[axis],
                        prefix=f"{prefix}_{label}",
                    )
                )
        return metrics


@dataclass
class _ParameterAccumulator:
    absolute_sum: dict[tuple[str, str], float] = field(default_factory=dict)
    count: dict[tuple[str, str], int] = field(default_factory=dict)

    def update(
        self,
        scope: str,
        name: str,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        if scope not in {"observable", "updated"}:
            raise ValueError(f"unknown parameter metric scope: {scope}")
        expanded = mask
        while expanded.ndim < prediction.ndim:
            expanded = expanded.unsqueeze(-1)
        values = (prediction - target).abs().masked_select(expanded.expand_as(prediction))
        if values.numel() == 0:
            return
        key = (scope, name)
        self.absolute_sum[key] = self.absolute_sum.get(key, 0.0) + float(
            values.detach().float().cpu().sum()
        )
        self.count[key] = self.count.get(key, 0) + int(values.numel())

    def metrics(self) -> dict[str, float | None]:
        results: dict[str, float | None] = {}
        for scope in ("observable", "updated"):
            for name in _IDENTIFIER_PARAMETERS:
                key = (scope, name)
                count = self.count.get(key, 0)
                results[f"{scope}_{name}_mae"] = (
                    self.absolute_sum.get(key, 0.0) / count if count else None
                )
                results[f"{scope}_{name}_count"] = float(count)
        return results


@dataclass
class _IdentifierAccumulator:
    observability_sum: dict[str, float] = field(default_factory=dict)
    observability_max: dict[str, float] = field(default_factory=dict)
    gate_sum: dict[str, float] = field(default_factory=dict)
    gate_max: dict[str, float] = field(default_factory=dict)
    active_count: dict[str, int] = field(default_factory=dict)
    update_count: dict[str, int] = field(default_factory=dict)
    diagnostic_steps: int = 0

    def update(
        self,
        diagnostics: ParameterUpdateDiagnostics,
        active: Tensor,
    ) -> None:
        expected_shape = (*active.shape, len(_IDENTIFIER_PARAMETERS))
        for name, value in (
            ("observability", diagnostics.observability),
            ("gate", diagnostics.gate),
            ("update_count", diagnostics.update_count),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"identifier {name} must have shape {expected_shape}, got {value.shape}"
                )
        self.diagnostic_steps += 1
        for parameter_index, parameter_name in enumerate(_IDENTIFIER_PARAMETERS):
            observability = diagnostics.observability[..., parameter_index].masked_select(active)
            gate = diagnostics.gate[..., parameter_index].masked_select(active)
            updates = diagnostics.update_count[..., parameter_index].masked_select(active)
            if observability.numel() == 0:
                continue
            detached_observability = observability.detach().float().cpu()
            detached_gate = gate.detach().float().cpu()
            self.observability_sum[parameter_name] = self.observability_sum.get(
                parameter_name, 0.0
            ) + float(detached_observability.sum())
            self.observability_max[parameter_name] = max(
                self.observability_max.get(parameter_name, 0.0),
                float(detached_observability.max()),
            )
            self.gate_sum[parameter_name] = self.gate_sum.get(parameter_name, 0.0) + float(
                detached_gate.sum()
            )
            self.gate_max[parameter_name] = max(
                self.gate_max.get(parameter_name, 0.0),
                float(detached_gate.max()),
            )
            self.active_count[parameter_name] = self.active_count.get(parameter_name, 0) + int(
                detached_gate.numel()
            )
            self.update_count[parameter_name] = self.update_count.get(parameter_name, 0) + int(
                updates.detach().cpu().sum()
            )

    def metrics(self) -> dict[str, float | None]:
        results: dict[str, float | None] = {
            "identifier_diagnostic_step_count": float(self.diagnostic_steps)
        }
        for name in _IDENTIFIER_PARAMETERS:
            count = self.active_count.get(name, 0)
            results[f"identifier_{name}_active_object_frame_count"] = float(count)
            results[f"identifier_{name}_observability_mean"] = (
                self.observability_sum.get(name, 0.0) / count if count else None
            )
            results[f"identifier_{name}_observability_max"] = (
                self.observability_max.get(name) if count else None
            )
            results[f"identifier_{name}_gate_mean"] = (
                self.gate_sum.get(name, 0.0) / count if count else None
            )
            results[f"identifier_{name}_gate_max"] = self.gate_max.get(name) if count else None
            results[f"identifier_{name}_update_count"] = float(self.update_count.get(name, 0))
        return results


@dataclass
class _TrackingAccumulator:
    last_prediction: dict[tuple[int, int], int] = field(default_factory=dict)
    switches: int = 0
    associations: int = 0

    def update(
        self,
        predicted_ids: Tensor,
        target_ids: Tensor,
        target_indices: Tensor,
        matched: Tensor,
        *,
        episode_offset: int,
    ) -> None:
        batch = predicted_ids.shape[0]
        for batch_index in range(batch):
            for belief_slot in (
                torch.nonzero(matched[batch_index], as_tuple=False).flatten().tolist()
            ):
                target_slot = int(target_indices[batch_index, belief_slot].detach().cpu())
                if target_slot < 0:
                    continue
                target_id = int(target_ids[batch_index, target_slot].detach().cpu())
                predicted_id = int(predicted_ids[batch_index, belief_slot].detach().cpu())
                if target_id < 0 or predicted_id < 0:
                    continue
                key = (episode_offset + batch_index, target_id)
                previous = self.last_prediction.get(key)
                if previous is not None and previous != predicted_id:
                    self.switches += 1
                self.last_prediction[key] = predicted_id
                self.associations += 1

    def metrics(self) -> dict[str, float | None]:
        return {
            "distance_gated_identity_switches": float(self.switches),
            "distance_gated_object_frame_associations": float(self.associations),
            "distance_gated_identity_switch_rate": (
                self.switches / self.associations if self.associations else None
            ),
        }


@dataclass
class _ForecastIdentityAccumulator:
    """Distance-gated identity consistency against the anchor mapping."""

    mismatches: int = 0
    associations: int = 0
    eligible: int = 0

    def update(
        self,
        anchor_target_ids: Tensor,
        forecast_target_ids: Tensor,
        eligible_mask: Tensor,
        association_mask: Tensor,
    ) -> None:
        if anchor_target_ids.shape != forecast_target_ids.shape:
            raise ValueError("forecast identity inputs must share shape [B,N]")
        if (
            eligible_mask.shape != anchor_target_ids.shape
            or association_mask.shape != anchor_target_ids.shape
            or eligible_mask.dtype is not torch.bool
            or association_mask.dtype is not torch.bool
        ):
            raise ValueError("forecast identity masks must be boolean [B,N]")
        eligible = eligible_mask & (anchor_target_ids >= 0)
        associated = association_mask & (anchor_target_ids >= 0) & (forecast_target_ids >= 0)
        if bool((associated & ~eligible).any()):
            raise ValueError("forecast identity associations must be a subset of eligibility")
        self.eligible += int(eligible.sum().detach().cpu())
        self.associations += int(associated.sum().detach().cpu())
        self.mismatches += int(
            ((anchor_target_ids != forecast_target_ids) & associated).sum().detach().cpu()
        )

    def metrics(self, prefix: str) -> dict[str, float | None]:
        return {
            f"{prefix}_eligible_count": float(self.eligible),
            f"{prefix}_mismatch_count": float(self.mismatches),
            f"{prefix}_association_count": float(self.associations),
            f"{prefix}_association_coverage": (
                self.associations / self.eligible if self.eligible else None
            ),
            f"{prefix}_mismatch_rate": (
                self.mismatches / self.associations if self.associations else None
            ),
        }


@dataclass
class _ScenarioEvaluationAccumulator:
    """Additive clean-pass metrics for one declared simulator scenario.

    The evaluator applies a batch-row scenario mask to the same tensors and
    support masks used by its pooled metrics.  This keeps scenario reporting a
    cheap accounting view of the primary rollout rather than a second model
    pass with potentially different state or randomness.
    """

    episode_count: int = 0
    current_position: _ErrorAccumulator = field(default_factory=_ErrorAccumulator)
    current_velocity: MaskedVelocityErrorAccumulator = field(
        default_factory=MaskedVelocityErrorAccumulator
    )
    current_calibration: _CalibrationAccumulator = field(default_factory=_CalibrationAccumulator)
    forecast_position: dict[str, _ErrorAccumulator] = field(default_factory=dict)
    forecast_velocity: dict[str, MaskedVelocityErrorAccumulator] = field(default_factory=dict)
    collision_events: _BinaryAccumulator = field(default_factory=_BinaryAccumulator)
    collision_events_by_horizon: dict[str, _BinaryAccumulator] = field(default_factory=dict)
    calibration_by_horizon: dict[str, _CalibrationAccumulator] = field(default_factory=dict)
    forecast_identity_by_horizon: dict[str, _ForecastIdentityAccumulator] = field(
        default_factory=dict
    )
    tracking: _TrackingAccumulator = field(default_factory=_TrackingAccumulator)
    target_object_frames: int = 0
    predicted_object_frames: int = 0
    matched_object_frames: int = 0
    distance_gated_matched_object_frames: int = 0
    forecast_target_count: dict[str, int] = field(default_factory=dict)
    forecast_tracked_count: dict[str, int] = field(default_factory=dict)
    forecast_active_count: dict[str, int] = field(default_factory=dict)
    forecast_predictable_target_count: dict[str, int] = field(default_factory=dict)
    forecast_censored_tracked_count: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _increment(values: dict[str, int], horizon: str, count: int) -> None:
        values[horizon] = values.get(horizon, 0) + count

    def metrics(
        self,
        *,
        scenario: str,
        horizons: tuple[str, ...],
        detection_threshold_label: str,
    ) -> dict[str, float | None]:
        """Flatten the scenario view into the public metrics namespace."""

        local: dict[str, float | None] = {
            "episode_count": float(self.episode_count),
            "current_assignment_target_coverage": (
                self.matched_object_frames / self.target_object_frames
                if self.target_object_frames
                else None
            ),
            "current_assignment_prediction_coverage": (
                self.matched_object_frames / self.predicted_object_frames
                if self.predicted_object_frames
                else None
            ),
            f"current_detection_recall@{detection_threshold_label}": (
                self.distance_gated_matched_object_frames / self.target_object_frames
                if self.target_object_frames
                else None
            ),
            f"current_detection_precision@{detection_threshold_label}": (
                self.distance_gated_matched_object_frames / self.predicted_object_frames
                if self.predicted_object_frames
                else None
            ),
            "target_object_frames": float(self.target_object_frames),
            "predicted_object_frames": float(self.predicted_object_frames),
            "assignment_matched_object_frames": float(self.matched_object_frames),
            f"distance_gated_matched_object_frames@{detection_threshold_label}": float(
                self.distance_gated_matched_object_frames
            ),
        }
        local.update(self.current_position.metrics("posterior_current"))
        local.update(self.current_velocity.metrics("posterior_current"))
        local.update(
            self.current_calibration.metrics(
                "posterior_current_position",
                include_axes=True,
            )
        )
        local.update(self.tracking.metrics())
        local.update(self.collision_events.metrics("collision"))
        for horizon in horizons:
            position = self.forecast_position.get(horizon, _ErrorAccumulator())
            velocity = self.forecast_velocity.get(
                horizon,
                MaskedVelocityErrorAccumulator(),
            )
            local.update(position.metrics(f"model@{horizon}"))
            local.update(velocity.metrics(f"model@{horizon}"))
            local.update(
                self.collision_events_by_horizon.get(horizon, _BinaryAccumulator()).metrics(
                    f"collision@{horizon}"
                )
            )
            local.update(
                self.calibration_by_horizon.get(
                    horizon,
                    _CalibrationAccumulator(),
                ).metrics(f"model@{horizon}_position", include_axes=True)
            )
            local.update(
                self.forecast_identity_by_horizon.get(
                    horizon,
                    _ForecastIdentityAccumulator(),
                ).metrics(f"forecast_identity@{horizon}")
            )
            target_count = self.forecast_target_count.get(horizon, 0)
            tracked_count = self.forecast_tracked_count.get(horizon, 0)
            active_count = self.forecast_active_count.get(horizon, 0)
            local.update(
                {
                    f"forecast_target_count@{horizon}": float(target_count),
                    f"forecast_tracked_count@{horizon}": float(tracked_count),
                    f"forecast_active_count@{horizon}": float(active_count),
                    f"forecast_target_coverage@{horizon}": (
                        active_count / target_count if target_count else None
                    ),
                    f"tracked_forecast_active_coverage@{horizon}": (
                        active_count / tracked_count if tracked_count else None
                    ),
                    f"model_dropped_forecast_count@{horizon}": float(tracked_count - active_count),
                    f"forecast_predictable_target_count@{horizon}": float(
                        self.forecast_predictable_target_count.get(horizon, 0)
                    ),
                    f"forecast_censored_tracked_count@{horizon}": float(
                        self.forecast_censored_tracked_count.get(horizon, 0)
                    ),
                    f"forecast_evaluated_object_horizons@{horizon}": float(position.count // 3),
                }
            )
        prefix = f"scenario_{scenario}_"
        return {f"{prefix}{name}": value for name, value in local.items()}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_evaluation_protocol(
    config: OrpheusConfig,
    *,
    checkpoint_sha256: str,
    resolved_seed_protocol: Any,
    batch_size: int,
    runtime_hypothesis_pool: bool,
) -> dict[str, Any]:
    """Canonical, JSON-safe contract for one resolved evaluation pass."""

    return {
        "schema_version": _EVALUATION_PROTOCOL_SCHEMA_VERSION,
        "metric_schema_version": _EVALUATION_METRIC_SCHEMA_VERSION,
        "per_scenario_metric_schema": _PER_SCENARIO_METRIC_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": _canonical_sha256(config.to_dict()),
        "split": resolved_seed_protocol.split,
        "seed_protocol": resolved_seed_protocol.name,
        "seed_manifest": list(resolved_seed_protocol.manifest.seeds),
        "horizons_seconds_requested": list(config.evaluation.horizons_seconds),
        "horizons_observation_grid": list(_configured_horizon_keys(config)),
        "batch_size": batch_size,
        "episode_count": config.evaluation.episodes,
        "runtime_intervention": {
            "evaluator_state_perturbation_in_primary": False,
            "runtime_hypothesis_pool": runtime_hypothesis_pool,
            "recovery_probe_enabled": config.evaluation.recovery_probe_enabled,
            "recovery_probe_position_std": config.evaluation.perturbation_position_std,
            "recovery_probe_velocity_std": config.evaluation.perturbation_velocity_std,
        },
    }


def _future_queries(
    config: OrpheusConfig,
    frame_index: int,
    total_frames: int,
) -> tuple[list[int], list[float]]:
    offsets: list[int] = []
    seconds: list[float] = []
    for horizon in config.evaluation.horizons_seconds:
        offset = max(1, int(round(horizon * config.simulator.frame_rate)))
        if frame_index + offset >= total_frames or offset in offsets:
            continue
        offsets.append(offset)
        seconds.append(offset / config.simulator.frame_rate)
    ordering = sorted(range(len(offsets)), key=offsets.__getitem__)
    return [offsets[index] for index in ordering], [seconds[index] for index in ordering]


def _horizon_key(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _configured_horizon_keys(config: OrpheusConfig) -> tuple[str, ...]:
    """Return the unique observation-grid horizons promised by the config."""

    frame_rate = config.simulator.frame_rate
    seconds = {
        max(1, int(round(horizon * frame_rate))) / frame_rate
        for horizon in config.evaluation.horizons_seconds
    }
    return tuple(_horizon_key(value) for value in sorted(seconds))


def _collision_logits_for_observation_windows(
    event_logits: Tensor,
    query_plan: ObservationWindowQueryPlan,
) -> Tensor:
    """Return collision logits aligned to each target frame's label window."""

    return query_plan.select_target_endpoints(event_logits)[..., MotionMode.COLLISION]


def _distance_gate_matches(
    prediction: Tensor,
    aligned_target: Tensor,
    assignment_mask: Tensor,
    *,
    threshold_m: float,
) -> Tensor:
    if threshold_m <= 0:
        raise ValueError("detection distance threshold must be positive")
    if prediction.shape != aligned_target.shape or prediction.shape[-1] != 3:
        raise ValueError("detection positions must share shape [B,N,3]")
    if assignment_mask.shape != prediction.shape[:-1]:
        raise ValueError("assignment mask must have shape [B,N]")
    distance = torch.linalg.vector_norm(prediction - aligned_target, dim=-1)
    return assignment_mask & torch.isfinite(distance) & (distance <= threshold_m)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _load_evaluation_model(
    config: OrpheusConfig,
    checkpoint: Path,
    *,
    device: torch.device,
    runtime_hypothesis_pool: bool,
) -> tuple[OnlineWorldModel, dict[str, Any]]:
    """Checkpoint-load one independent evaluation runtime."""

    model = OnlineWorldModel.from_config(config, device=device)
    payload = load_checkpoint(
        checkpoint,
        model=model,
        # Keep unused optimizer moments off the accelerator; model loading
        # copies only needed weights to their owning (possibly hybrid) devices.
        map_location="cpu",
        expected_config=config,
    )
    if runtime_hypothesis_pool:
        enable_runtime_hypothesis_pool(model, config)
    if (model.hypothesis_controller is not None) != runtime_hypothesis_pool:
        raise RuntimeError("reported runtime hypothesis policy does not match active controller")
    model.eval()
    return model, payload


def _run_recovery_probe(
    config: OrpheusConfig,
    checkpoint: Path,
    *,
    loader: DataLoader[Any],
    device: torch.device,
    runtime_hypothesis_pool: bool,
    report_progress: Callable[..., None] | None = None,
) -> _RecoveryProbeResult:
    """Replay clean RGB prefixes in a separate runtime, then probe recovery.

    The primary evaluator model is deliberately not an argument. This helper
    checkpoint-loads its own runtime and stops each replay immediately after
    the perturbed observation, so synthetic state can never enter the primary
    online posterior sequence or any later primary forecast anchor.
    """

    seed_everything(
        config.project.seed + 60_000,
        deterministic=config.project.deterministic,
    )
    model, recovery_payload = _load_evaluation_model(
        config,
        checkpoint,
        device=device,
        runtime_hypothesis_pool=runtime_hypothesis_pool,
    )
    del recovery_payload
    result = _RecoveryProbeResult()
    if report_progress is not None:
        report_progress("recovery_probe_started")

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            rgb = batch["rgb"]
            if rgb.ndim != 5:
                raise ValueError("evaluation DataLoader must emit [B,T,3,H,W]")
            batch_size, total_frames = rgb.shape[:2]
            model.reset(batch_size=batch_size)
            perturbation_frame = max(1, total_frames // 3)

            # Reconstruct the same causal RGB history in a fresh runtime. No
            # simulator state participates in these online updates.
            for frame_index in range(perturbation_frame):
                packet = make_rgb_packet(batch, frame_index)
                prepared = (
                    None if model.belief is None else model.prepare_propagation(packet.timestamp)
                )
                prefix_belief = model.ingest(packet, prepared=prepared)
                _require_finite_belief(
                    prefix_belief,
                    context=f"recovery batch {batch_index} prefix frame {frame_index}",
                )

            if model.belief is None:
                raise RuntimeError("recovery probe prefix did not initialize WorldBelief")
            packet = make_rgb_packet(batch, perturbation_frame)
            # Model construction and prefix replay may consume different RNG
            # streams as architectures evolve. Bind the synthetic recovery
            # intervention to the explicit batch manifest immediately before
            # sampling so the same evaluation seed means the same probe.
            seed_everything(
                config.project.seed + 60_000 + batch_index,
                deterministic=config.project.deterministic,
            )
            perturbed = perturb_belief(
                model.belief,
                position_std=config.evaluation.perturbation_position_std,
                velocity_std=config.evaluation.perturbation_velocity_std,
                covariance_log_bias=0.5,
            )
            model.state.belief = perturbed
            if model.hypothesis_controller is not None:
                model.hypothesis_controller.invalidate_pending(
                    reason="evaluation_recovery_probe_perturbation",
                    reset_evidence=True,
                    belief=perturbed,
                )
            prepared = model.prepare_propagation(packet.timestamp)
            prior = prepared.prior
            _require_finite_belief(
                prior,
                context=f"recovery batch {batch_index} perturbed prior",
            )
            frame_offsets, query_seconds = _future_queries(
                config,
                perturbation_frame,
                total_frames,
            )
            prior_rollout = None
            prior_std = None
            if query_seconds:
                prior_rollout = model.dynamics.rollout(
                    prior,
                    query_seconds,
                    return_events=False,
                )
                _require_finite_trajectory(
                    prior_rollout,
                    context=f"recovery batch {batch_index} prior rollout",
                )
                prior_std = prior.objects.fast_log_variance[..., :3].exp().mean(dim=-1).sqrt()
                result.perturbation_updates += batch_size

            posterior = model.ingest(packet, prepared=prepared)
            _require_finite_belief(
                posterior,
                context=f"recovery batch {batch_index} posterior",
            )
            if model.diagnostics.oracle_used:
                raise RuntimeError("oracle diagnostics detected during RGB-only recovery probe")
            target_position = batch["objects"]["position"][:, perturbation_frame]
            target_active = batch["objects"]["active"][:, perturbation_frame].bool()
            target_indices, matched = match_belief_to_targets(
                posterior,
                target_position,
                target_active,
            )
            persistent_support = _recovery_persistent_support(prior, posterior, matched)
            if prior_std is not None:
                posterior_std = (
                    posterior.objects.fast_log_variance[..., :3].exp().mean(dim=-1).sqrt()
                )
                contraction_valid = persistent_support
                result.uncertainty_contraction.extend(
                    (prior_std - posterior_std)
                    .masked_select(contraction_valid)
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )

            if prior_rollout is not None:
                posterior_rollout = model.dynamics.rollout(
                    posterior,
                    query_seconds,
                    return_events=False,
                )
                _require_finite_trajectory(
                    posterior_rollout,
                    context=f"recovery batch {batch_index} posterior rollout",
                )
                for query_index, frame_offset in enumerate(frame_offsets):
                    target_frame = perturbation_frame + frame_offset
                    future_target = gather_target_slots(
                        batch["objects"]["position"][:, target_frame],
                        target_indices,
                    )
                    future_active = (
                        gather_target_slots(
                            batch["objects"]["active"][:, target_frame].unsqueeze(-1),
                            target_indices,
                        )
                        .squeeze(-1)
                        .bool()
                    )
                    valid = persistent_support & future_active
                    valid &= future_predictable_mask(
                        batch,
                        anchor_index=perturbation_frame,
                        target_index=target_frame,
                        target_indices=target_indices,
                    )
                    result.correction.update(
                        prior_rollout.positions[:, query_index],
                        posterior_rollout.positions[:, query_index],
                        future_target,
                        valid,
                    )
            result.evaluated_episodes += batch_size
            if report_progress is not None:
                report_progress(
                    "recovery_probe_batch_complete",
                    recovery_probe_batch=batch_index,
                    recovery_probe_batches=len(loader),
                    recovery_probe_evaluated_episodes=result.evaluated_episodes,
                )

    return result


@dataclass
class _EvaluationProgressSink:
    """Durable progress channel available before model initialization."""

    path: Path | None
    callback: Callable[[dict[str, Any]], None] | None
    last_event: dict[str, Any] | None = None

    def publish(self, event: dict[str, Any]) -> None:
        self.last_event = event
        if self.path is not None:
            atomic_write_text(
                self.path,
                json.dumps(event, allow_nan=False, indent=2, sort_keys=True) + "\n",
            )
        if self.callback is not None:
            self.callback(event)

    def fail(self, error: BaseException) -> None:
        if self.path is None and self.callback is None:
            return
        event: dict[str, Any] = {
            "stage": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        if self.last_event is not None:
            event.update(
                {
                    key: value
                    for key, value in self.last_event.items()
                    if key not in {"stage", "updated_utc", "pid"}
                }
            )
            event["last_stage"] = self.last_event.get("stage")
            event["last_progress"] = self.last_event
        if self.path is not None:
            with contextlib.suppress(Exception):
                atomic_write_text(
                    self.path,
                    json.dumps(event, allow_nan=False, indent=2, sort_keys=True) + "\n",
                )
        if self.callback is not None:
            with contextlib.suppress(Exception):
                self.callback(event)


def _planned_evaluation_output(
    checkpoint_path: str | Path,
    *,
    split: str,
    seed_protocol: str,
    output_dir: str | Path | None,
) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    requested_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent.parent
        / "evaluation"
        / (split if seed_protocol == STANDARD_SEED_PROTOCOL else f"{split}-{seed_protocol}")
    )
    return checkpoint, timestamped_artifact_path(requested_output).resolve()


def evaluate_checkpoint(
    config: OrpheusConfig,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    seed_protocol: str = STANDARD_SEED_PROTOCOL,
    seed_offset: int | None = None,
    output_dir: str | Path | None = None,
    device_info: DeviceInfo | None = None,
    runtime_hypothesis_pool: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trusted checkpoint with progress from initialization onward."""

    checkpoint, planned_output = _planned_evaluation_output(
        checkpoint_path,
        split=split,
        seed_protocol=seed_protocol,
        output_dir=output_dir,
    )
    durable_progress_path = (
        Path(progress_path).expanduser().resolve()
        if progress_path is not None
        else planned_output / "evaluation_progress.json"
    )

    sink = _EvaluationProgressSink(
        path=durable_progress_path,
        callback=progress_callback,
    )
    # These identities must precede the first durable progress write.  In
    # particular, a custom non-ignored output path must not contaminate the
    # source fingerprint that the resulting report claims to have evaluated.
    evaluation_source_provenance = capture_git_metadata(Path(__file__).resolve().parents[2])
    initial_event: dict[str, Any] = {
        "stage": "initializing",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "split": split,
        "seed_protocol": seed_protocol,
        "checkpoint": str(checkpoint),
        "evaluation_source_provenance": evaluation_source_provenance,
        "output_directory": str(planned_output),
        "runtime_hypothesis_pool": runtime_hypothesis_pool,
        "rgb_only": True,
    }
    try:
        # Capture bytes before notifying external callbacks. A callback or
        # concurrent trainer may atomically replace ``last.pt`` immediately
        # after the initializing event; primary and recovery must still use
        # the exact pre-notification snapshot. This copies bytes only—it does
        # not deserialize or construct the model.
        with _capture_checkpoint_snapshot(checkpoint) as captured_checkpoint:
            sink.publish(
                {
                    **initial_event,
                    "checkpoint_capture_status": "captured",
                    "checkpoint_sha256": captured_checkpoint.sha256,
                    "checkpoint_byte_count": captured_checkpoint.byte_count,
                }
            )
            return _evaluate_checkpoint_impl(
                config,
                checkpoint,
                captured_checkpoint=captured_checkpoint,
                evaluation_source_provenance=evaluation_source_provenance,
                split=split,
                seed_protocol=seed_protocol,
                seed_offset=seed_offset,
                output=planned_output,
                device_info=device_info,
                runtime_hypothesis_pool=runtime_hypothesis_pool,
                progress_sink=sink,
            )
    except BaseException as error:
        if sink.last_event is None:
            # A missing, empty, or unreadable checkpoint still gets a durable
            # initializing→failed lifecycle. Callback failure here becomes the
            # terminal error because the requested progress consumer itself
            # prevented delivery.
            try:
                sink.publish(
                    {
                        **initial_event,
                        "checkpoint_capture_status": "failed",
                    }
                )
            except BaseException as progress_error:
                sink.fail(progress_error)
                raise
        sink.fail(error)
        raise


def _evaluate_checkpoint_impl(
    config: OrpheusConfig,
    checkpoint_path: str | Path,
    *,
    captured_checkpoint: _CapturedCheckpoint,
    evaluation_source_provenance: Mapping[str, Any],
    split: str = "test",
    seed_protocol: str = STANDARD_SEED_PROTOCOL,
    seed_offset: int | None = None,
    output: Path,
    device_info: DeviceInfo | None = None,
    runtime_hypothesis_pool: bool = False,
    progress_sink: _EvaluationProgressSink,
) -> dict[str, Any]:
    """Evaluate a trusted local checkpoint on held-out RGB episodes.

    Simulator state and parameters are used for metrics and the explicitly
    labelled oracle-parameter analytic baseline only.  Every model correction
    consumes RGB plus known camera calibration.
    """

    config.validate()
    if not config.evaluation.rgb_only:
        raise ValueError("Milestone 1 evaluation must set evaluation.rgb_only=true")
    if config.runtime.modality != "rgb" or config.runtime.enable_debug_oracle:
        raise ValueError("held-out RGB evaluation forbids debug_oracle runtime input")
    if config.runtime.hypothesis_pool_enabled:
        raise ValueError(
            "checkpoint evaluation config must keep runtime.hypothesis_pool_enabled=false; "
            "use the explicit runtime_hypothesis_pool intervention"
        )
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    resolved_device = device_info or select_device(config.device.preference)
    device = resolved_device.device
    seed_everything(
        config.project.seed + 50_000,
        deterministic=config.project.deterministic,
    )
    model, payload = _load_evaluation_model(
        config,
        captured_checkpoint.snapshot_path,
        device=device,
        runtime_hypothesis_pool=runtime_hypothesis_pool,
    )

    progress_context: dict[str, Any] = {}

    def report_progress(stage: str, **values: Any) -> None:
        event = {
            "stage": stage,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            **progress_context,
            **values,
        }
        progress_sink.publish(event)

    checkpoint_training_config = payload["config"].get("training")
    if not isinstance(checkpoint_training_config, Mapping):
        raise ValueError("checkpoint config is missing its training mapping")
    checkpoint_validation_episodes = checkpoint_training_config.get("validation_episodes")
    if (
        not isinstance(checkpoint_validation_episodes, int)
        or isinstance(checkpoint_validation_episodes, bool)
        or checkpoint_validation_episodes < 0
    ):
        raise ValueError(
            "checkpoint config training.validation_episodes must be a nonnegative integer"
        )
    checkpoint_step = int(payload["step"])
    checkpoint_simulator_version = str(payload["simulator_version"])
    checkpoint_specification_version = str(payload["specification_version"])
    stored_checkpoint_source = payload.get("git")
    checkpoint_source_provenance = (
        dict(stored_checkpoint_source) if isinstance(stored_checkpoint_source, Mapping) else None
    )
    # Loading validates the complete payload, but evaluation needs only these
    # scalar/config values after weights have been copied into the model.
    del payload
    resolved_seed_protocol = make_evaluation_seed_protocol(
        name=seed_protocol,
        split=split,
        episode_count=config.evaluation.episodes,
        training_validation_episodes=checkpoint_validation_episodes,
        seed_offset=seed_offset,
    )
    dataset = SyntheticSphereDataset(
        config,
        split=resolved_seed_protocol.split,
        seeds=resolved_seed_protocol.manifest,
        memory_cache=True,
    )
    evaluation_batch_size = min(
        config.training.batch_size,
        max(1, config.evaluation.episodes),
    )
    loader = DataLoader(
        dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_episodes,
        drop_last=False,
    )
    resolved_evaluation_protocol = _resolved_evaluation_protocol(
        config,
        checkpoint_sha256=captured_checkpoint.sha256,
        resolved_seed_protocol=resolved_seed_protocol,
        batch_size=evaluation_batch_size,
        runtime_hypothesis_pool=runtime_hypothesis_pool,
    )
    resolved_evaluation_protocol_sha256 = _canonical_sha256(resolved_evaluation_protocol)
    progress_context.update(
        {
            "split": resolved_seed_protocol.split,
            "seed_protocol": resolved_seed_protocol.name,
            "device": str(device),
            "precision": resolved_device.precision,
            "episodes": config.evaluation.episodes,
            "batches": len(loader),
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_step,
            "output_directory": str(output),
            "runtime_hypothesis_pool": runtime_hypothesis_pool,
            "rgb_only": True,
        }
    )
    report_progress(
        "started",
    )

    current_error = _ErrorAccumulator()
    current_velocity_error = MaskedVelocityErrorAccumulator()
    ordinary_velocity_correction = OrdinaryVelocityCorrectionAccumulator()
    temporal_velocity_measurements = TemporalVelocityMeasurementAccumulator()
    forecast_errors: dict[tuple[str, str], _ErrorAccumulator] = {}
    forecast_velocity_errors: dict[str, MaskedVelocityErrorAccumulator] = {}
    events = _BinaryAccumulator()
    events_by_horizon: dict[str, _BinaryAccumulator] = {}
    current_calibration = _CalibrationAccumulator()
    calibration = _CalibrationAccumulator()
    calibration_by_horizon: dict[str, _CalibrationAccumulator] = {}
    forecast_identity_by_horizon: dict[str, _ForecastIdentityAccumulator] = {}
    collision_conditioned_forecasts = CollisionConditionedForecastAccumulator()
    parameters = _ParameterAccumulator()
    directional_parameters = OnlineParameterUpdateAccumulator()
    identifier_metrics = _IdentifierAccumulator()
    tracking = _TrackingAccumulator()
    occlusion_transitions = OcclusionTransitionAccumulator()
    global_latencies: list[float] = []
    fast_latencies: list[float] = []
    rollout_latencies: list[float] = []
    uncertainty_visible: list[float] = []
    uncertainty_occluded: list[float] = []
    nonfinite_outputs = 0
    evaluated_episodes = 0
    primary_posterior_trace = _PosteriorTraceHasher()
    target_object_frames = 0
    predicted_object_frames = 0
    matched_object_frames = 0
    distance_gated_matched_object_frames = 0
    trajectory_change_point_count = 0
    trajectory_change_point_inspected_object_frames = 0
    lateral_intervention_gain_sum = 0.0
    lateral_intervention_feature_count = 0
    lateral_intervention_gain_above_half_count = 0
    gravity_intervention_gain_sum = 0.0
    gravity_intervention_feature_count = 0
    gravity_intervention_gain_above_half_count = 0
    simulator_external_actuation_object_event_count = 0
    simulator_external_actuation_interval_count = 0
    simulator_created_object_event_count = 0
    simulator_removed_object_event_count = 0
    runtime_hypothesis_forecast_anchor_count = 0
    runtime_hypothesis_candidates = _runtime_hypothesis_candidate_names(config)
    runtime_hypothesis_axis_selection_count = {
        axis: [0 for _ in runtime_hypothesis_candidates]
        for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_horizon_axis_selection_count: dict[tuple[str, int], list[int]] = {}
    runtime_hypothesis_axis_supported_count = {
        axis: 0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_axis_fallback_count = {
        axis: 0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_horizon_axis_support_count: dict[tuple[str, int], tuple[int, int]] = {}
    runtime_hypothesis_axis_composed_candidate_step_count = {
        axis: [0 for _ in runtime_hypothesis_candidates]
        for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_axis_composed_fallback_step_count = {
        axis: 0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_axis_composed_total_step_count = {
        axis: 0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_axis_composition_grid_fallback_count = {
        axis: 0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_axis_residual_applied_count = {
        axis: 0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_axis_residual_sum = {
        axis: 0.0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_axis_residual_absolute_sum = {
        axis: 0.0 for axis in config.runtime.hypothesis_axis_independent_axes
    }
    runtime_hypothesis_regime_step_count = [0 for _ in _RUNTIME_HYPOTHESIS_REGIMES]
    runtime_hypothesis_regime_query_count = [0 for _ in _RUNTIME_HYPOTHESIS_REGIMES]
    runtime_hypothesis_axis_evidence_summary = {
        axis: {
            "cell_count": 0,
            "support_count_sum": 0,
            "age_seconds_sum": 0.0,
            "age_seconds_max": 0.0,
            "observability_sum": 0.0,
            "observability_min": 1.0,
            "predictive_variance_sum": 0.0,
            "predictive_variance_max": 0.0,
            "confidence_margin_sum": 0.0,
            "confidence_margin_min": 1.0,
        }
        for axis in config.runtime.hypothesis_axis_independent_axes
    }
    forecast_target_count: dict[str, int] = {}
    forecast_tracked_count: dict[str, int] = {}
    forecast_active_count: dict[str, int] = {}
    forecast_predictable_target_count: dict[str, int] = {}
    forecast_censored_tracked_count: dict[str, int] = {}
    configured_horizons = _configured_horizon_keys(config)
    scenario_accumulators = {
        scenario: _ScenarioEvaluationAccumulator() for scenario in config.simulator.scenario_mixture
    }

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            rgb = batch["rgb"]
            if rgb.ndim != 5:
                raise ValueError("evaluation DataLoader must emit [B,T,3,H,W]")
            batch_size, total_frames = rgb.shape[:2]
            scenario_names = batch.get("metadata", {}).get("scenario")
            if (
                not isinstance(scenario_names, list)
                or len(scenario_names) != batch_size
                or not all(isinstance(name, str) for name in scenario_names)
            ):
                raise ValueError("evaluation metadata.scenario must contain one string per episode")
            undeclared_scenarios = set(scenario_names) - set(scenario_accumulators)
            if undeclared_scenarios:
                raise ValueError(
                    "evaluation batch contains undeclared scenarios: "
                    + ", ".join(sorted(undeclared_scenarios))
                )
            scenario_batch_masks = {
                scenario: torch.tensor(
                    [name == scenario for name in scenario_names],
                    device=device,
                    dtype=torch.bool,
                )
                for scenario in scenario_accumulators
            }
            for scenario, row_mask in scenario_batch_masks.items():
                scenario_accumulators[scenario].episode_count += int(row_mask.sum().detach().cpu())
            externally_actuated = batch["events"]["externally_actuated"].bool()
            simulator_external_actuation_object_event_count += int(
                externally_actuated.sum().detach().cpu()
            )
            simulator_external_actuation_interval_count += int(
                externally_actuated.any(dim=-1).sum().detach().cpu()
            )
            simulator_created_object_event_count += int(
                batch["events"]["created"].bool().sum().detach().cpu()
            )
            simulator_removed_object_event_count += int(
                batch["events"]["removed"].bool().sum().detach().cpu()
            )
            model.reset(batch_size=batch_size)
            anchor_stride = max(1, total_frames // 8)
            diagnostic_offset = 0

            for frame_index in range(total_frames):
                packet = make_rgb_packet(batch, frame_index)
                ordinary_velocity_prior = None
                prepared_propagation = None
                propagation_elapsed_ms = 0.0
                if model.belief is not None:
                    synchronize(device)
                    propagation_started = time.perf_counter()
                    prepared_propagation = model.prepare_propagation(packet.timestamp)
                    synchronize(device)
                    propagation_elapsed_ms += (time.perf_counter() - propagation_started) * 1000.0
                    ordinary_velocity_prior = prepared_propagation.prior

                pre_ingest_parameters: dict[str, Tensor] | None = None
                if model.belief is not None:
                    pre_ingest_parameters = {
                        "restitution": model.belief.objects.restitution.clone(),
                        "drag": model.belief.objects.drag.clone(),
                        "active": model.belief.objects.active.clone(),
                        "object_id": model.belief.objects.object_id.clone(),
                    }
                synchronize(device)
                update_started = time.perf_counter()
                belief = model.ingest(packet, prepared=prepared_propagation)
                synchronize(device)
                update_elapsed_ms = (
                    propagation_elapsed_ms + (time.perf_counter() - update_started) * 1000.0
                )
                _require_finite_belief(
                    belief,
                    context=f"primary batch {batch_index} frame {frame_index}",
                )
                primary_posterior_trace.update(
                    batch_index=batch_index,
                    frame_index=frame_index,
                    belief=belief,
                )
                last_measurements = model.last_measurements
                if last_measurements is not None:
                    expected_measurement_timestamp = last_measurements.timestamp.new_full(
                        last_measurements.timestamp.shape,
                        packet.timestamp,
                    )
                    if not torch.allclose(
                        last_measurements.timestamp,
                        expected_measurement_timestamp,
                        atol=1.0e-6,
                        rtol=0.0,
                    ):
                        # SKIP leaves the previous diagnostics snapshot in the
                        # runtime; never count that stale measurement twice.
                        last_measurements = None
                if last_measurements is not None:
                    _require_finite_measurements(
                        last_measurements,
                        context=f"primary batch {batch_index} frame {frame_index}",
                    )
                temporal_velocity_measurements.update(last_measurements)
                if last_measurements is not None:
                    change_point_mask = last_measurements.auxiliary.get(
                        "trajectory_change_point_mask"
                    )
                    if change_point_mask is not None:
                        if change_point_mask.dtype != torch.bool or change_point_mask.ndim != 2:
                            raise ValueError("trajectory_change_point_mask must be boolean [B,N]")
                        trajectory_change_point_count += int(change_point_mask.sum().detach().cpu())
                        eligible_mask = last_measurements.auxiliary.get(
                            "trajectory_change_point_eligible_mask"
                        )
                        if (
                            eligible_mask is None
                            or eligible_mask.shape != change_point_mask.shape
                            or eligible_mask.dtype != torch.bool
                        ):
                            raise ValueError(
                                "trajectory_change_point_eligible_mask must be boolean [B,N]"
                            )
                        trajectory_change_point_inspected_object_frames += int(
                            eligible_mask.sum().detach().cpu()
                        )
                    lateral_gain = last_measurements.auxiliary.get(
                        "trajectory_lateral_intervention_gain"
                    )
                    lateral_valid = last_measurements.auxiliary.get(
                        "trajectory_lateral_intervention_feature_valid_mask"
                    )
                    if lateral_gain is not None or lateral_valid is not None:
                        if (
                            lateral_gain is None
                            or lateral_valid is None
                            or lateral_gain.shape != lateral_valid.shape
                            or lateral_valid.dtype != torch.bool
                        ):
                            raise ValueError(
                                "lateral intervention gain/valid diagnostics "
                                "must be aligned [B,N] tensors"
                            )
                        selected_gain = lateral_gain.masked_select(lateral_valid)
                        lateral_intervention_gain_sum += float(selected_gain.sum().detach().cpu())
                        lateral_intervention_feature_count += int(selected_gain.numel())
                        lateral_intervention_gain_above_half_count += int(
                            (selected_gain >= 0.5).sum().detach().cpu()
                        )
                    gravity_gain = last_measurements.auxiliary.get(
                        "trajectory_gravity_intervention_gain"
                    )
                    gravity_valid = last_measurements.auxiliary.get(
                        "trajectory_gravity_intervention_feature_valid_mask"
                    )
                    if gravity_gain is not None or gravity_valid is not None:
                        if (
                            gravity_gain is None
                            or gravity_valid is None
                            or gravity_gain.shape != gravity_valid.shape
                            or gravity_valid.dtype != torch.bool
                        ):
                            raise ValueError(
                                "gravity intervention gain/valid diagnostics "
                                "must be aligned [B,N] tensors"
                            )
                        selected_gain = gravity_gain.masked_select(gravity_valid)
                        gravity_intervention_gain_sum += float(selected_gain.sum().detach().cpu())
                        gravity_intervention_feature_count += int(selected_gain.numel())
                        gravity_intervention_gain_above_half_count += int(
                            (selected_gain >= 0.5).sum().detach().cpu()
                        )
                if model.diagnostics.oracle_used:
                    raise RuntimeError(
                        "oracle diagnostics detected during claimed RGB-only evaluation"
                    )
                target_position = batch["objects"]["position"][:, frame_index]
                target_active = batch["objects"]["active"][:, frame_index].bool()
                target_indices, matched = match_belief_to_targets(
                    belief, target_position, target_active
                )
                target_object_frames += int(target_active.sum().detach().cpu())
                predicted_object_frames += int(belief.objects.active.sum().detach().cpu())
                matched_object_frames += int(matched.sum().detach().cpu())
                aligned_position = gather_target_slots(target_position, target_indices)
                distance_gated_matched = _distance_gate_matches(
                    belief.objects.position,
                    aligned_position,
                    matched,
                    threshold_m=_CURRENT_DETECTION_DISTANCE_THRESHOLD_M,
                )
                distance_gated_matched_object_frames += int(
                    distance_gated_matched.sum().detach().cpu()
                )
                current_error.update(
                    belief.objects.position,
                    aligned_position,
                    matched,
                )
                target_velocity = batch["objects"]["velocity"][:, frame_index]
                aligned_velocity = gather_target_slots(target_velocity, target_indices)
                current_velocity_error.update(
                    belief.objects.velocity,
                    aligned_velocity,
                    distance_gated_matched,
                )
                current_calibration.update(
                    belief.objects.position,
                    belief.objects.fast_log_variance[..., :3],
                    aligned_position,
                    distance_gated_matched,
                )
                if ordinary_velocity_prior is not None and last_measurements is not None:
                    same_persistent_slot = (
                        ordinary_velocity_prior.objects.active
                        & belief.objects.active
                        & (ordinary_velocity_prior.objects.object_id == belief.objects.object_id)
                    )
                    ordinary_velocity_correction.update(
                        ordinary_velocity_prior.objects.velocity,
                        belief.objects.velocity,
                        aligned_velocity,
                        distance_gated_matched & same_persistent_slot,
                    )
                tracking.update(
                    belief.objects.object_id,
                    batch["objects"]["id"][:, frame_index],
                    target_indices,
                    distance_gated_matched,
                    episode_offset=evaluated_episodes,
                )
                for scenario, row_mask in scenario_batch_masks.items():
                    scenario_metrics = scenario_accumulators[scenario]
                    object_mask = row_mask.unsqueeze(-1)
                    scenario_target_active = target_active & object_mask
                    scenario_predicted_active = belief.objects.active & object_mask
                    scenario_matched = matched & object_mask
                    scenario_distance_gated_matched = distance_gated_matched & object_mask
                    scenario_metrics.target_object_frames += int(
                        scenario_target_active.sum().detach().cpu()
                    )
                    scenario_metrics.predicted_object_frames += int(
                        scenario_predicted_active.sum().detach().cpu()
                    )
                    scenario_metrics.matched_object_frames += int(
                        scenario_matched.sum().detach().cpu()
                    )
                    scenario_metrics.distance_gated_matched_object_frames += int(
                        scenario_distance_gated_matched.sum().detach().cpu()
                    )
                    scenario_metrics.current_position.update(
                        belief.objects.position,
                        aligned_position,
                        scenario_matched,
                    )
                    scenario_metrics.current_velocity.update(
                        belief.objects.velocity,
                        aligned_velocity,
                        scenario_distance_gated_matched,
                    )
                    scenario_metrics.current_calibration.update(
                        belief.objects.position,
                        belief.objects.fast_log_variance[..., :3],
                        aligned_position,
                        scenario_distance_gated_matched,
                    )
                    scenario_metrics.tracking.update(
                        belief.objects.object_id,
                        batch["objects"]["id"][:, frame_index],
                        target_indices,
                        scenario_distance_gated_matched,
                        episode_offset=evaluated_episodes,
                    )

                visible_fraction = gather_target_slots(
                    batch["objects"]["visible_fraction"][:, frame_index].unsqueeze(-1),
                    target_indices,
                ).squeeze(-1)
                position_std = (0.5 * belief.objects.fast_log_variance[..., :3]).exp().mean(dim=-1)
                occlusion_transitions.update_frame(
                    predicted_ids=belief.objects.object_id,
                    predicted_active=belief.objects.active,
                    position_std_m=position_std,
                    target_ids=batch["objects"]["id"][:, frame_index],
                    target_active=target_active,
                    target_visible_fraction=batch["objects"]["visible_fraction"][:, frame_index],
                    matched_target_indices=target_indices,
                    reliable_visible_matches=distance_gated_matched,
                    episode_offset=evaluated_episodes,
                )
                visible_mask = distance_gated_matched & (visible_fraction >= 0.5)
                occluded_mask = distance_gated_matched & (visible_fraction <= 0.05)
                if visible_mask.any():
                    uncertainty_visible.extend(
                        position_std.masked_select(visible_mask).detach().float().cpu().tolist()
                    )
                if occluded_mask.any():
                    uncertainty_occluded.extend(
                        position_std.masked_select(occluded_mask).detach().float().cpu().tolist()
                    )
                aligned_radius = gather_target_slots(
                    batch["objects"]["radius"][:, frame_index],
                    target_indices,
                )
                aligned_mass = gather_target_slots(
                    batch["objects"]["mass"][:, frame_index],
                    target_indices,
                )
                aligned_restitution = gather_target_slots(
                    batch["objects"]["restitution"][:, frame_index],
                    target_indices,
                )
                aligned_drag = gather_target_slots(
                    batch["objects"]["drag"][:, frame_index],
                    target_indices,
                )
                aligned_friction = gather_target_slots(
                    batch["objects"]["friction"][:, frame_index],
                    target_indices,
                )
                identifier_diagnostics = (
                    model.identifier.last_diagnostics if model.identifier is not None else None
                )
                if identifier_diagnostics is not None:
                    _require_finite_diagnostics(
                        identifier_diagnostics,
                        context=f"primary batch {batch_index} frame {frame_index} identifier",
                    )
                    identifier_metrics.update(
                        identifier_diagnostics,
                        belief.objects.active,
                    )
                    predictions = (
                        belief.objects.mass,
                        belief.objects.restitution,
                        belief.objects.drag,
                        belief.objects.friction,
                        belief.objects.radius,
                    )
                    targets = (
                        aligned_mass,
                        aligned_restitution,
                        aligned_drag,
                        aligned_friction,
                        aligned_radius,
                    )
                    for parameter_index, (parameter_name, prediction, target) in enumerate(
                        zip(
                            _IDENTIFIER_PARAMETERS,
                            predictions,
                            targets,
                            strict=True,
                        )
                    ):
                        runtime_observable = (
                            identifier_diagnostics.observability[..., parameter_index] > 0.0
                        )
                        runtime_updated = identifier_diagnostics.update_count[
                            ..., parameter_index
                        ].bool()
                        parameters.update(
                            "observable",
                            parameter_name,
                            prediction,
                            target,
                            distance_gated_matched & runtime_observable,
                        )
                        parameters.update(
                            "updated",
                            parameter_name,
                            prediction,
                            target,
                            distance_gated_matched & runtime_updated,
                        )
                    if pre_ingest_parameters is not None:
                        same_persistent_slot = (
                            pre_ingest_parameters["active"]
                            & belief.objects.active
                            & (pre_ingest_parameters["object_id"] == belief.objects.object_id)
                        )
                        collision_informative = gather_target_slots(
                            batch["events"]["collision"][:, frame_index].unsqueeze(-1),
                            target_indices,
                        ).squeeze(-1)
                        free_steps = model.observability.config.minimum_free_steps
                        free_window_start = frame_index - free_steps + 1
                        drag_informative_target = torch.zeros_like(target_active)
                        if free_window_start >= 0:
                            active_history = batch["objects"]["active"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            contact_history = batch["events"]["contact"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            collision_history = batch["events"]["collision"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            actuation_history = batch["events"]["externally_actuated"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            speed = torch.linalg.vector_norm(
                                batch["objects"]["velocity"][:, frame_index],
                                dim=-1,
                            )
                            drag_informative_target = (
                                active_history.all(dim=1)
                                & ~contact_history.any(dim=1)
                                & ~collision_history.any(dim=1)
                                & ~actuation_history.any(dim=1)
                                & (speed >= config.model.identification.drag_speed_threshold)
                                & (batch["objects"]["visible_fraction"][:, frame_index] >= 0.5)
                            )
                        drag_informative = gather_target_slots(
                            drag_informative_target.unsqueeze(-1),
                            target_indices,
                        ).squeeze(-1)
                        common_directional_mask = distance_gated_matched & same_persistent_slot
                        directional_parameters.update(
                            "restitution",
                            pre_ingest_parameters["restitution"],
                            belief.objects.restitution,
                            aligned_restitution,
                            common_directional_mask
                            & identifier_diagnostics.update_count[..., 1].bool()
                            & collision_informative,
                        )
                        directional_parameters.update(
                            "drag",
                            pre_ingest_parameters["drag"],
                            belief.objects.drag,
                            aligned_drag,
                            common_directional_mask
                            & identifier_diagnostics.update_count[..., 2].bool()
                            & drag_informative,
                        )

                run_forecast = frame_index % anchor_stride == 0
                frame_offsets, query_seconds = _future_queries(config, frame_index, total_frames)
                if run_forecast and query_seconds:
                    event_query_plan = observation_window_query_plan(
                        frame_offsets,
                        frame_rate=config.simulator.frame_rate,
                    )
                    synchronize(device)
                    rollout_started = time.perf_counter()
                    # Route scored forecasts through the online runtime.  This
                    # is normally equivalent to the learned rollout, but is
                    # essential for an explicitly enabled runtime hypothesis
                    # policy: calling ``model.dynamics`` here would attach the
                    # policy yet silently score only its learned candidate.
                    trajectory = model.predict(event_query_plan.query_seconds)
                    _require_finite_trajectory(
                        trajectory,
                        context=f"primary batch {batch_index} frame {frame_index}",
                    )
                    if runtime_hypothesis_pool:
                        axis_indices = trajectory.auxiliary.get("hypothesis_axis_index")
                        axis_supported = trajectory.auxiliary.get("hypothesis_axis_supported")
                        interaction_regime = trajectory.auxiliary.get(
                            "hypothesis_interaction_regime"
                        )
                        # Before the first exact-due RGB observation there is
                        # deliberately no selector evidence, and normal
                        # runtime falls back to the learned rollout. Record
                        # that explicit default rather than pretending an
                        # axis-selection tensor already exists.
                        if axis_indices is None:
                            assert model.hypothesis_controller is not None
                            (
                                axis_indices,
                                axis_supported,
                                interaction_regime,
                            ) = _runtime_hypothesis_learned_fallback_diagnostics(
                                model.hypothesis_controller,
                                belief,
                                trajectory,
                            )
                        if isinstance(axis_indices, Tensor) and axis_indices.shape == (
                            batch_size,
                            3,
                        ):
                            axis_indices = (
                                axis_indices.unsqueeze(1)
                                .unsqueeze(2)
                                .expand(
                                    batch_size,
                                    len(event_query_plan.query_seconds),
                                    belief.objects.max_objects,
                                    3,
                                )
                            )
                        if isinstance(axis_indices, Tensor) and axis_indices.shape == (
                            batch_size,
                            len(event_query_plan.query_seconds),
                            3,
                        ):
                            axis_indices = axis_indices.unsqueeze(2).expand(
                                batch_size,
                                len(event_query_plan.query_seconds),
                                belief.objects.max_objects,
                                3,
                            )
                        if isinstance(axis_supported, Tensor) and axis_supported.shape == (
                            batch_size,
                            3,
                        ):
                            axis_supported = (
                                axis_supported.unsqueeze(1)
                                .unsqueeze(2)
                                .expand(
                                    batch_size,
                                    len(event_query_plan.query_seconds),
                                    belief.objects.max_objects,
                                    3,
                                )
                            )
                        if isinstance(axis_supported, Tensor) and axis_supported.shape == (
                            batch_size,
                            len(event_query_plan.query_seconds),
                            3,
                        ):
                            axis_supported = axis_supported.unsqueeze(2).expand(
                                batch_size,
                                len(event_query_plan.query_seconds),
                                belief.objects.max_objects,
                                3,
                            )
                        if not isinstance(axis_indices, Tensor) or axis_indices.shape != (
                            batch_size,
                            len(event_query_plan.query_seconds),
                            belief.objects.max_objects,
                            3,
                        ):
                            raise RuntimeError(
                                "runtime hypothesis forecast must expose "
                                "hypothesis_axis_index [B,Q,N,3]"
                            )
                        if (
                            not isinstance(axis_supported, Tensor)
                            or axis_supported.shape != axis_indices.shape
                            or axis_supported.dtype != torch.bool
                        ):
                            raise RuntimeError(
                                "runtime hypothesis forecast must expose boolean "
                                "hypothesis_axis_supported [B,Q,N,3]"
                            )
                        if (
                            axis_indices.dtype != torch.int64
                            or torch.any(axis_indices < 0)
                            or torch.any(axis_indices >= len(runtime_hypothesis_candidates))
                        ):
                            raise RuntimeError("runtime hypothesis axis index is invalid")
                        if (
                            not isinstance(interaction_regime, Tensor)
                            or interaction_regime.shape != axis_indices.shape[:3]
                            or interaction_regime.dtype != torch.int64
                            or torch.any(interaction_regime < 0)
                            or torch.any(interaction_regime >= len(_RUNTIME_HYPOTHESIS_REGIMES))
                        ):
                            raise RuntimeError(
                                "runtime hypothesis interaction regime must be int64 [B,Q,N]; "
                                f"got {getattr(interaction_regime, 'shape', None)} / "
                                f"{getattr(interaction_regime, 'dtype', None)} for "
                                f"{axis_indices.shape[:3]}"
                            )
                        axis_support_count = trajectory.auxiliary.get(
                            "hypothesis_axis_support_count"
                        )
                        axis_age = trajectory.auxiliary.get("hypothesis_axis_evidence_age_seconds")
                        axis_observability = trajectory.auxiliary.get(
                            "hypothesis_axis_observability"
                        )
                        axis_predictive_variance = trajectory.auxiliary.get(
                            "hypothesis_axis_predictive_variance"
                        )
                        axis_confidence = trajectory.auxiliary.get(
                            "hypothesis_axis_confidence_margin"
                        )
                        composed_candidate_count = trajectory.auxiliary.get(
                            "hypothesis_composed_candidate_step_count"
                        )
                        composed_fallback_count = trajectory.auxiliary.get(
                            "hypothesis_composed_fallback_step_count"
                        )
                        composed_total_count = trajectory.auxiliary.get(
                            "hypothesis_composed_total_step_count"
                        )
                        composed_regime_count = trajectory.auxiliary.get(
                            "hypothesis_composed_regime_step_count"
                        )
                        composition_grid_fallback = trajectory.auxiliary.get(
                            "hypothesis_composition_grid_fallback"
                        )
                        position_residual = trajectory.auxiliary.get("hypothesis_position_residual")
                        residual_applied = trajectory.auxiliary.get(
                            "hypothesis_position_residual_applied"
                        )
                        local_shape = axis_indices.shape
                        if position_residual is None and residual_applied is None:
                            position_residual = torch.zeros_like(
                                axis_indices, dtype=trajectory.positions.dtype
                            )
                            residual_applied = torch.zeros_like(axis_indices, dtype=torch.bool)
                        elif position_residual is None or residual_applied is None:
                            raise RuntimeError(
                                "runtime hypothesis residual diagnostics are incomplete"
                            )
                        applicability_values = (
                            axis_support_count,
                            axis_age,
                            axis_observability,
                            axis_predictive_variance,
                            axis_confidence,
                        )
                        if any(value is not None for value in applicability_values) and not all(
                            isinstance(value, Tensor) for value in applicability_values
                        ):
                            raise RuntimeError(
                                "runtime hypothesis applicability tensors are incomplete"
                            )
                        if axis_support_count is None:
                            axis_support_count = torch.zeros_like(axis_indices)
                            axis_age = torch.zeros_like(
                                axis_indices, dtype=trajectory.positions.dtype
                            )
                            axis_observability = torch.zeros_like(
                                axis_indices, dtype=trajectory.positions.dtype
                            )
                            axis_predictive_variance = torch.zeros_like(
                                axis_indices, dtype=trajectory.positions.dtype
                            )
                            axis_confidence = torch.zeros_like(
                                axis_indices, dtype=trajectory.positions.dtype
                            )
                        if composition_grid_fallback is None:
                            composition_grid_fallback = torch.zeros_like(
                                axis_supported, dtype=torch.bool
                            )
                        if (
                            not isinstance(composition_grid_fallback, Tensor)
                            or composition_grid_fallback.shape != local_shape
                            or composition_grid_fallback.dtype != torch.bool
                        ):
                            raise RuntimeError(
                                "runtime hypothesis composition grid fallback must be boolean "
                                "[B,Q,N,3]"
                            )
                        if (
                            not isinstance(position_residual, Tensor)
                            or position_residual.shape != local_shape
                            or not position_residual.is_floating_point()
                            or not torch.isfinite(position_residual).all()
                        ):
                            raise RuntimeError(
                                "runtime hypothesis position residual must be finite [B,Q,N,3]"
                            )
                        if (
                            not isinstance(residual_applied, Tensor)
                            or residual_applied.shape != local_shape
                            or residual_applied.dtype != torch.bool
                        ):
                            raise RuntimeError(
                                "runtime hypothesis residual-applied mask must be boolean [B,Q,N,3]"
                            )
                        if (
                            not isinstance(axis_support_count, Tensor)
                            or axis_support_count.shape != local_shape
                            or axis_support_count.dtype != torch.int64
                        ):
                            raise RuntimeError(
                                "runtime hypothesis support count must be int64 [B,Q,N,3]"
                            )
                        for name, value in (
                            ("evidence age", axis_age),
                            ("observability", axis_observability),
                            ("predictive variance", axis_predictive_variance),
                            ("confidence margin", axis_confidence),
                        ):
                            if (
                                not isinstance(value, Tensor)
                                or value.shape != local_shape
                                or not value.is_floating_point()
                                or not torch.isfinite(value).all()
                                or torch.any(value < 0)
                            ):
                                raise RuntimeError(
                                    f"runtime hypothesis {name} must be finite nonnegative [B,Q,N,3]"
                                )
                        composition_values = (
                            composed_candidate_count,
                            composed_fallback_count,
                            composed_total_count,
                            composed_regime_count,
                        )
                        if any(value is not None for value in composition_values):
                            if not all(isinstance(value, Tensor) for value in composition_values):
                                raise RuntimeError(
                                    "runtime hypothesis composed count tensors are incomplete"
                                )
                            assert isinstance(composed_candidate_count, Tensor)
                            assert isinstance(composed_fallback_count, Tensor)
                            assert isinstance(composed_total_count, Tensor)
                            assert isinstance(composed_regime_count, Tensor)
                            _validate_runtime_hypothesis_composition_counts(
                                local_shape=local_shape,
                                candidate_count=composed_candidate_count,
                                fallback_count=composed_fallback_count,
                                total_count=composed_total_count,
                                regime_count=composed_regime_count,
                                independent_axes=config.runtime.hypothesis_axis_independent_axes,
                                candidate_size=len(runtime_hypothesis_candidates),
                                regime_size=len(_RUNTIME_HYPOTHESIS_REGIMES),
                            )
                        runtime_hypothesis_forecast_anchor_count += batch_size
                        target_axis_indices = event_query_plan.select_target_endpoints(axis_indices)
                        target_axis_supported = event_query_plan.select_target_endpoints(
                            axis_supported
                        )
                        target_interaction_regime = event_query_plan.select_target_endpoints(
                            interaction_regime
                        )
                        target_axis_support_count = event_query_plan.select_target_endpoints(
                            axis_support_count
                        )
                        target_axis_age = event_query_plan.select_target_endpoints(axis_age)
                        target_axis_observability = event_query_plan.select_target_endpoints(
                            axis_observability
                        )
                        target_axis_predictive_variance = event_query_plan.select_target_endpoints(
                            axis_predictive_variance
                        )
                        target_axis_confidence = event_query_plan.select_target_endpoints(
                            axis_confidence
                        )
                        target_position_residual = event_query_plan.select_target_endpoints(
                            position_residual
                        )
                        target_residual_applied = event_query_plan.select_target_endpoints(
                            residual_applied
                        )
                        target_composed_candidate_count = (
                            event_query_plan.select_target_endpoints(composed_candidate_count)
                            if isinstance(composed_candidate_count, Tensor)
                            else None
                        )
                        target_composed_fallback_count = (
                            event_query_plan.select_target_endpoints(composed_fallback_count)
                            if isinstance(composed_fallback_count, Tensor)
                            else None
                        )
                        target_composed_total_count = (
                            event_query_plan.select_target_endpoints(composed_total_count)
                            if isinstance(composed_total_count, Tensor)
                            else None
                        )
                        target_composed_regime_count = (
                            event_query_plan.select_target_endpoints(composed_regime_count)
                            if isinstance(composed_regime_count, Tensor)
                            else None
                        )
                        target_composition_grid_fallback = event_query_plan.select_target_endpoints(
                            composition_grid_fallback
                        )
                        target_hypothesis_active = event_query_plan.select_target_endpoints(
                            trajectory.active_mask
                        )
                        query_regime_counts = (
                            torch.bincount(
                                target_interaction_regime.masked_select(target_hypothesis_active),
                                minlength=len(_RUNTIME_HYPOTHESIS_REGIMES),
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        for regime_index, count in enumerate(query_regime_counts):
                            runtime_hypothesis_regime_query_count[regime_index] += int(count)
                        if target_composed_regime_count is not None:
                            regime_counts = _sum_runtime_hypothesis_counts_on_host(
                                target_composed_regime_count
                                * target_hypothesis_active.unsqueeze(-1).to(torch.int64),
                                dim=(0, 1, 2),
                            )
                            for regime_index, count in enumerate(regime_counts.tolist()):
                                runtime_hypothesis_regime_step_count[regime_index] += int(count)
                        for axis in config.runtime.hypothesis_axis_independent_axes:
                            applied_cells = (
                                target_residual_applied[..., axis] & target_hypothesis_active
                            )
                            runtime_hypothesis_axis_residual_applied_count[axis] += int(
                                applied_cells.sum().detach().cpu()
                            )
                            if bool(applied_cells.any()):
                                applied_residual = target_position_residual[
                                    ..., axis
                                ].masked_select(applied_cells)
                                runtime_hypothesis_axis_residual_sum[axis] += float(
                                    applied_residual.sum().detach().cpu()
                                )
                                runtime_hypothesis_axis_residual_absolute_sum[axis] += float(
                                    applied_residual.abs().sum().detach().cpu()
                                )
                            runtime_hypothesis_axis_composition_grid_fallback_count[axis] += int(
                                (
                                    target_composition_grid_fallback[..., axis]
                                    & target_hypothesis_active
                                )
                                .sum()
                                .detach()
                                .cpu()
                            )
                            evidence_cells = (
                                target_axis_supported[..., axis] & target_hypothesis_active
                            )
                            summary = runtime_hypothesis_axis_evidence_summary[axis]
                            cell_count = int(evidence_cells.sum().detach().cpu())
                            summary["cell_count"] += cell_count
                            if cell_count:
                                summary["support_count_sum"] += int(
                                    target_axis_support_count[..., axis]
                                    .masked_select(evidence_cells)
                                    .sum()
                                    .detach()
                                    .cpu()
                                )
                                age_values = target_axis_age[..., axis].masked_select(
                                    evidence_cells
                                )
                                observability_values = target_axis_observability[
                                    ..., axis
                                ].masked_select(evidence_cells)
                                confidence_values = target_axis_confidence[..., axis].masked_select(
                                    evidence_cells
                                )
                                predictive_variance_values = target_axis_predictive_variance[
                                    ..., axis
                                ].masked_select(evidence_cells)
                                summary["age_seconds_sum"] += float(age_values.sum().detach().cpu())
                                summary["age_seconds_max"] = max(
                                    float(summary["age_seconds_max"]),
                                    float(age_values.max().detach().cpu()),
                                )
                                summary["observability_sum"] += float(
                                    observability_values.sum().detach().cpu()
                                )
                                summary["observability_min"] = min(
                                    float(summary["observability_min"]),
                                    float(observability_values.min().detach().cpu()),
                                )
                                summary["predictive_variance_sum"] += float(
                                    predictive_variance_values.sum().detach().cpu()
                                )
                                summary["predictive_variance_max"] = max(
                                    float(summary["predictive_variance_max"]),
                                    float(predictive_variance_values.max().detach().cpu()),
                                )
                                summary["confidence_margin_sum"] += float(
                                    confidence_values.sum().detach().cpu()
                                )
                                summary["confidence_margin_min"] = min(
                                    float(summary["confidence_margin_min"]),
                                    float(confidence_values.min().detach().cpu()),
                                )
                            if target_composed_candidate_count is not None:
                                composed_counts = _sum_runtime_hypothesis_counts_on_host(
                                    target_composed_candidate_count[..., axis, :]
                                    * target_hypothesis_active.unsqueeze(-1).to(torch.int64),
                                    dim=(0, 1, 2),
                                )
                                for candidate_index, count in enumerate(composed_counts.tolist()):
                                    runtime_hypothesis_axis_composed_candidate_step_count[axis][
                                        candidate_index
                                    ] += int(count)
                                assert target_composed_fallback_count is not None
                                assert target_composed_total_count is not None
                                runtime_hypothesis_axis_composed_fallback_step_count[axis] += int(
                                    (
                                        target_composed_fallback_count[..., axis]
                                        * target_hypothesis_active.to(torch.int64)
                                    )
                                    .sum()
                                    .detach()
                                    .cpu()
                                )
                                runtime_hypothesis_axis_composed_total_step_count[axis] += int(
                                    (
                                        target_composed_total_count[..., axis]
                                        * target_hypothesis_active.to(torch.int64)
                                    )
                                    .sum()
                                    .detach()
                                    .cpu()
                                )
                            supported_count = int(
                                (target_axis_supported[..., axis] & target_hypothesis_active)
                                .sum()
                                .detach()
                                .cpu()
                            )
                            active_count_for_axis = int(
                                target_hypothesis_active.sum().detach().cpu()
                            )
                            fallback_count = active_count_for_axis - supported_count
                            runtime_hypothesis_axis_supported_count[axis] += supported_count
                            runtime_hypothesis_axis_fallback_count[axis] += fallback_count
                            selected_counts = (
                                torch.bincount(
                                    target_axis_indices[..., axis].masked_select(
                                        target_hypothesis_active & target_axis_supported[..., axis]
                                    ),
                                    minlength=len(runtime_hypothesis_candidates),
                                )
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            for candidate_index, count in enumerate(selected_counts):
                                runtime_hypothesis_axis_selection_count[axis][candidate_index] += (
                                    int(count)
                                )
                            for query_index, query_seconds_value in enumerate(query_seconds):
                                horizon = _horizon_key(query_seconds_value)
                                key = (horizon, axis)
                                horizon_counts = (
                                    runtime_hypothesis_horizon_axis_selection_count.setdefault(
                                        key,
                                        [0 for _ in runtime_hypothesis_candidates],
                                    )
                                )
                                selected_counts = (
                                    torch.bincount(
                                        target_axis_indices[:, query_index, :, axis].masked_select(
                                            target_hypothesis_active[:, query_index]
                                            & target_axis_supported[:, query_index, :, axis]
                                        ),
                                        minlength=len(runtime_hypothesis_candidates),
                                    )
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )
                                for candidate_index, count in enumerate(selected_counts):
                                    horizon_counts[candidate_index] += int(count)
                                query_supported_count = int(
                                    (
                                        target_axis_supported[:, query_index, :, axis]
                                        & target_hypothesis_active[:, query_index]
                                    )
                                    .sum()
                                    .detach()
                                    .cpu()
                                )
                                query_active_count = int(
                                    target_hypothesis_active[:, query_index].sum().detach().cpu()
                                )
                                prior_supported, prior_fallback = (
                                    runtime_hypothesis_horizon_axis_support_count.get(
                                        key,
                                        (0, 0),
                                    )
                                )
                                runtime_hypothesis_horizon_axis_support_count[key] = (
                                    prior_supported + query_supported_count,
                                    prior_fallback + query_active_count - query_supported_count,
                                )
                    synchronize(device)
                    rollout_latencies.append((time.perf_counter() - rollout_started) * 1000.0)
                    model_positions = event_query_plan.select_target_endpoints(trajectory.positions)
                    model_velocities = event_query_plan.select_target_endpoints(
                        trajectory.velocities
                    )
                    model_log_variance = event_query_plan.select_target_endpoints(
                        trajectory.fast_log_variance
                    )
                    model_active_mask = event_query_plan.select_target_endpoints(
                        trajectory.active_mask
                    )
                    model_collision_logits = (
                        None
                        if trajectory.event_logits is None
                        else _collision_logits_for_observation_windows(
                            trajectory.event_logits,
                            event_query_plan,
                        )
                    )
                    oracle_drag = gather_target_slots(
                        batch["objects"]["drag"][:, frame_index],
                        target_indices,
                    )
                    query_tensor = (
                        belief.timestamp.new_tensor(query_seconds)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )
                    baselines = baseline_bundle(
                        belief.objects.position,
                        belief.objects.velocity,
                        query_tensor,
                        gravity=belief.gravity,
                        default_drag=sum(config.simulator.drag_range) / 2.0,
                        oracle_drag=oracle_drag,
                    )
                    for query_index, frame_offset in enumerate(frame_offsets):
                        target_frame = frame_index + frame_offset
                        future_target = gather_target_slots(
                            batch["objects"]["position"][:, target_frame],
                            target_indices,
                        )
                        future_target_velocity = gather_target_slots(
                            batch["objects"]["velocity"][:, target_frame],
                            target_indices,
                        )
                        future_active = (
                            gather_target_slots(
                                batch["objects"]["active"][:, target_frame].unsqueeze(-1),
                                target_indices,
                            )
                            .squeeze(-1)
                            .bool()
                        )
                        horizon = _horizon_key(query_seconds[query_index])
                        common_valid = matched & future_active
                        predictable = future_predictable_mask(
                            batch,
                            anchor_index=frame_index,
                            target_index=target_frame,
                            target_indices=target_indices,
                        )
                        scene_predictable = future_scene_predictable_mask(
                            batch,
                            anchor_index=frame_index,
                            target_index=target_frame,
                        )
                        point_valid = common_valid & predictable
                        forecast_target_count[horizon] = forecast_target_count.get(
                            horizon, 0
                        ) + int(batch["objects"]["active"][:, target_frame].sum().detach().cpu())
                        forecast_tracked_count[horizon] = forecast_tracked_count.get(
                            horizon, 0
                        ) + int(common_valid.sum().detach().cpu())
                        forecast_active_count[horizon] = forecast_active_count.get(
                            horizon, 0
                        ) + int(
                            (common_valid & model_active_mask[:, query_index]).sum().detach().cpu()
                        )
                        forecast_predictable_target_count[horizon] = (
                            forecast_predictable_target_count.get(horizon, 0)
                            + int(
                                (
                                    batch["objects"]["active"][:, target_frame].bool()
                                    & scene_predictable[:, None]
                                )
                                .sum()
                                .detach()
                                .cpu()
                            )
                        )
                        forecast_censored_tracked_count[horizon] = (
                            forecast_censored_tracked_count.get(horizon, 0)
                            + int((common_valid & ~predictable).sum().detach().cpu())
                        )
                        forecast_errors.setdefault(("model", horizon), _ErrorAccumulator()).update(
                            model_positions[:, query_index],
                            future_target,
                            point_valid,
                        )
                        forecast_velocity_errors.setdefault(
                            horizon,
                            MaskedVelocityErrorAccumulator(),
                        ).update(
                            model_velocities[:, query_index],
                            future_target_velocity,
                            point_valid,
                        )
                        for baseline_name, positions in baselines.items():
                            forecast_errors.setdefault(
                                (baseline_name, horizon),
                                _ErrorAccumulator(),
                            ).update(
                                positions[:, query_index],
                                future_target,
                                point_valid,
                            )
                        # Calibration remains a proper stochastic diagnostic:
                        # unseen interventions are omitted from point RMSE but
                        # still test whether predictive uncertainty widens.
                        calibration.update(
                            model_positions[:, query_index],
                            model_log_variance[:, query_index, :, :3],
                            future_target,
                            common_valid,
                        )
                        calibration_by_horizon.setdefault(
                            horizon,
                            _CalibrationAccumulator(),
                        ).update(
                            model_positions[:, query_index],
                            model_log_variance[:, query_index, :, :3],
                            future_target,
                            common_valid,
                        )
                        forecast_target_indices, forecast_matched = _match_positions_to_targets(
                            model_positions[:, query_index],
                            model_active_mask[:, query_index].bool(),
                            batch["objects"]["position"][:, target_frame],
                            batch["objects"]["active"][:, target_frame].bool(),
                        )
                        forecast_aligned_position = gather_target_slots(
                            batch["objects"]["position"][:, target_frame],
                            forecast_target_indices,
                        )
                        forecast_distance_gated = _distance_gate_matches(
                            model_positions[:, query_index],
                            forecast_aligned_position,
                            forecast_matched,
                            threshold_m=_CURRENT_DETECTION_DISTANCE_THRESHOLD_M,
                        )
                        anchor_target_ids = gather_target_slots(
                            batch["objects"]["id"][:, frame_index],
                            target_indices,
                        )
                        forecast_target_ids = gather_target_slots(
                            batch["objects"]["id"][:, target_frame],
                            forecast_target_indices,
                        )
                        forecast_identity_eligible = (
                            distance_gated_matched
                            & common_valid
                            & model_active_mask[:, query_index].bool()
                        )
                        forecast_identity_associated = (
                            forecast_identity_eligible & forecast_distance_gated
                        )
                        forecast_identity_by_horizon.setdefault(
                            horizon,
                            _ForecastIdentityAccumulator(),
                        ).update(
                            anchor_target_ids,
                            forecast_target_ids,
                            forecast_identity_eligible,
                            forecast_identity_associated,
                        )
                        collision_class_masks = collision_class_masks_for_forecast_window(
                            batch["events"],
                            anchor_frame=frame_index,
                            target_frame=target_frame,
                        )
                        collision_during_window = ~collision_class_masks["no_collision"]
                        aligned_collision_during_window = (
                            gather_target_slots(
                                collision_during_window.unsqueeze(-1),
                                target_indices,
                            )
                            .squeeze(-1)
                            .bool()
                        )
                        collision_classes = {
                            class_name: (
                                gather_target_slots(class_mask.unsqueeze(-1), target_indices)
                                .squeeze(-1)
                                .bool()
                            )
                            for class_name, class_mask in collision_class_masks.items()
                        }
                        collision_conditioned_forecasts.update(
                            horizon=horizon,
                            predictions={
                                "model": model_positions[:, query_index],
                                **{
                                    baseline_name: positions[:, query_index]
                                    for baseline_name, positions in baselines.items()
                                },
                            },
                            target=future_target,
                            valid_mask=point_valid,
                            collision_mask=aligned_collision_during_window,
                            collision_classes=collision_classes,
                        )
                        if model_collision_logits is not None:
                            # The selected endpoint logit covers exactly
                            # [target_frame - 1, target_frame], matching the
                            # simulator event label rather than a horizon
                            # prefix or gap between requested horizons.
                            collision_target = gather_target_slots(
                                batch["events"]["collision"][:, target_frame].unsqueeze(-1),
                                target_indices,
                            ).squeeze(-1)
                            events.update(
                                model_collision_logits[:, query_index],
                                collision_target,
                                point_valid,
                            )
                            events_by_horizon.setdefault(
                                horizon,
                                _BinaryAccumulator(),
                            ).update(
                                model_collision_logits[:, query_index],
                                collision_target,
                                point_valid,
                            )
                        for scenario, row_mask in scenario_batch_masks.items():
                            scenario_metrics = scenario_accumulators[scenario]
                            object_mask = row_mask.unsqueeze(-1)
                            scenario_point_valid = point_valid & object_mask
                            scenario_common_valid = common_valid & object_mask
                            scenario_metrics.forecast_position.setdefault(
                                horizon,
                                _ErrorAccumulator(),
                            ).update(
                                model_positions[:, query_index],
                                future_target,
                                scenario_point_valid,
                            )
                            scenario_metrics.forecast_velocity.setdefault(
                                horizon,
                                MaskedVelocityErrorAccumulator(),
                            ).update(
                                model_velocities[:, query_index],
                                future_target_velocity,
                                scenario_point_valid,
                            )
                            scenario_metrics.calibration_by_horizon.setdefault(
                                horizon,
                                _CalibrationAccumulator(),
                            ).update(
                                model_positions[:, query_index],
                                model_log_variance[:, query_index, :, :3],
                                future_target,
                                scenario_common_valid,
                            )
                            scenario_metrics.forecast_identity_by_horizon.setdefault(
                                horizon,
                                _ForecastIdentityAccumulator(),
                            ).update(
                                anchor_target_ids,
                                forecast_target_ids,
                                forecast_identity_eligible & object_mask,
                                forecast_identity_associated & object_mask,
                            )
                            scenario_metrics._increment(
                                scenario_metrics.forecast_target_count,
                                horizon,
                                int(
                                    (
                                        batch["objects"]["active"][:, target_frame].bool()
                                        & object_mask
                                    )
                                    .sum()
                                    .detach()
                                    .cpu()
                                ),
                            )
                            scenario_metrics._increment(
                                scenario_metrics.forecast_tracked_count,
                                horizon,
                                int(scenario_common_valid.sum().detach().cpu()),
                            )
                            scenario_metrics._increment(
                                scenario_metrics.forecast_active_count,
                                horizon,
                                int(
                                    (scenario_common_valid & model_active_mask[:, query_index])
                                    .sum()
                                    .detach()
                                    .cpu()
                                ),
                            )
                            scenario_metrics._increment(
                                scenario_metrics.forecast_predictable_target_count,
                                horizon,
                                int(
                                    (
                                        batch["objects"]["active"][:, target_frame].bool()
                                        & scene_predictable[:, None]
                                        & object_mask
                                    )
                                    .sum()
                                    .detach()
                                    .cpu()
                                ),
                            )
                            scenario_metrics._increment(
                                scenario_metrics.forecast_censored_tracked_count,
                                horizon,
                                int((scenario_common_valid & ~predictable).sum().detach().cpu()),
                            )
                            if model_collision_logits is not None:
                                scenario_metrics.collision_events.update(
                                    model_collision_logits[:, query_index],
                                    collision_target,
                                    scenario_point_valid,
                                )
                                scenario_metrics.collision_events_by_horizon.setdefault(
                                    horizon,
                                    _BinaryAccumulator(),
                                ).update(
                                    model_collision_logits[:, query_index],
                                    collision_target,
                                    scenario_point_valid,
                                )
                new_diagnostics = model.diagnostics.records[diagnostic_offset:]
                diagnostic_offset = len(model.diagnostics.records)
                for diagnostic in new_diagnostics:
                    if diagnostic.observation_mode in {
                        "GLOBAL_DISCOVERY",
                        "RECOVERY",
                    }:
                        global_latencies.append(update_elapsed_ms)
                    elif diagnostic.observation_mode == "FAST_ROI":
                        fast_latencies.append(update_elapsed_ms)
                # Forecast anchors are the expensive part of this evaluator.
                # Persist only at those coarse milestones (plus the final
                # frame) so a detached MPS run is diagnosable without turning
                # every observation update into a status write.
                if run_forecast or frame_index == total_frames - 1:
                    report_progress(
                        "anchor_complete",
                        batch=batch_index,
                        batches=len(loader),
                        frame=frame_index + 1,
                        total_frames=total_frames,
                        evaluated_episodes=evaluated_episodes,
                        batch_episode_count=batch_size,
                    )
            evaluated_episodes += batch_size
            report_progress(
                "batch_complete",
                batch=batch_index,
                batches=len(loader),
                evaluated_episodes=evaluated_episodes,
            )

    metrics: dict[str, Any] = {}
    metrics.update(current_error.metrics("posterior_current"))
    metrics.update(current_velocity_error.metrics("posterior_current"))
    metrics.update(
        current_calibration.metrics(
            "posterior_current_position",
            include_axes=True,
        )
    )
    metrics.update(ordinary_velocity_correction.metrics())
    metrics.update(temporal_velocity_measurements.metrics())
    for (method, horizon), accumulator in sorted(forecast_errors.items()):
        metrics.update(accumulator.metrics(f"{method}@{horizon}"))
    for horizon in configured_horizons:
        metrics.update(
            forecast_velocity_errors.get(
                horizon,
                MaskedVelocityErrorAccumulator(),
            ).metrics(f"model@{horizon}")
        )
        metrics.update(
            calibration_by_horizon.get(
                horizon,
                _CalibrationAccumulator(),
            ).metrics(f"model@{horizon}_position", include_axes=True)
        )
        metrics.update(
            forecast_identity_by_horizon.get(
                horizon,
                _ForecastIdentityAccumulator(),
            ).metrics(f"forecast_identity@{horizon}")
        )
    metrics.update(collision_conditioned_forecasts.metrics())
    metrics.update(events.metrics("collision"))
    for horizon in configured_horizons:
        metrics.update(
            events_by_horizon.get(horizon, _BinaryAccumulator()).metrics(f"collision@{horizon}")
        )
    metrics.update(calibration.metrics())
    metrics.update(parameters.metrics())
    directional_parameter_metrics = directional_parameters.metrics()
    metrics.update(directional_parameter_metrics)
    metrics.update(identifier_metrics.metrics())
    metrics.update(tracking.metrics())
    metrics.update(occlusion_transitions.metrics())
    detection_threshold_label = f"{_CURRENT_DETECTION_DISTANCE_THRESHOLD_M:.3f}m"
    metrics.update(
        {
            "current_assignment_target_coverage": (
                matched_object_frames / target_object_frames if target_object_frames else None
            ),
            "current_assignment_prediction_coverage": (
                matched_object_frames / predicted_object_frames if predicted_object_frames else None
            ),
            f"current_detection_recall@{detection_threshold_label}": (
                distance_gated_matched_object_frames / target_object_frames
                if target_object_frames
                else None
            ),
            f"current_detection_precision@{detection_threshold_label}": (
                distance_gated_matched_object_frames / predicted_object_frames
                if predicted_object_frames
                else None
            ),
            "target_object_frames": float(target_object_frames),
            "predicted_object_frames": float(predicted_object_frames),
            "assignment_matched_object_frames": float(matched_object_frames),
            "trajectory_change_point_count": float(trajectory_change_point_count),
            "trajectory_change_point_inspected_object_frames": float(
                trajectory_change_point_inspected_object_frames
            ),
            "trajectory_change_point_rate": (
                trajectory_change_point_count / trajectory_change_point_inspected_object_frames
                if trajectory_change_point_inspected_object_frames
                else None
            ),
            "lateral_intervention_feature_count": float(lateral_intervention_feature_count),
            "lateral_intervention_mean_soft_gain": (
                lateral_intervention_gain_sum / lateral_intervention_feature_count
                if lateral_intervention_feature_count
                else None
            ),
            "lateral_intervention_gain_above_half_count": float(
                lateral_intervention_gain_above_half_count
            ),
            "gravity_intervention_feature_count": float(gravity_intervention_feature_count),
            "gravity_intervention_mean_soft_gain": (
                gravity_intervention_gain_sum / gravity_intervention_feature_count
                if gravity_intervention_feature_count
                else None
            ),
            "gravity_intervention_gain_above_half_count": float(
                gravity_intervention_gain_above_half_count
            ),
            f"distance_gated_matched_object_frames@{detection_threshold_label}": float(
                distance_gated_matched_object_frames
            ),
        }
    )
    if runtime_hypothesis_pool:
        metrics["runtime_hypothesis_forecast_anchor_count"] = float(
            runtime_hypothesis_forecast_anchor_count
        )
        for axis, counts in runtime_hypothesis_axis_selection_count.items():
            for candidate, count in zip(runtime_hypothesis_candidates, counts, strict=True):
                metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_{candidate}_count"] = float(count)
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_supported_count"] = float(
                runtime_hypothesis_axis_supported_count[axis]
            )
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_fallback_count"] = float(
                runtime_hypothesis_axis_fallback_count[axis]
            )
            for candidate, count in zip(
                runtime_hypothesis_candidates,
                runtime_hypothesis_axis_composed_candidate_step_count[axis],
                strict=True,
            ):
                metrics[
                    f"runtime_hypothesis_axis_{'xyz'[axis]}_{candidate}_composed_step_count"
                ] = float(count)
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_composed_fallback_step_count"] = float(
                runtime_hypothesis_axis_composed_fallback_step_count[axis]
            )
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_composed_total_step_count"] = float(
                runtime_hypothesis_axis_composed_total_step_count[axis]
            )
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_composition_grid_fallback_count"] = (
                float(runtime_hypothesis_axis_composition_grid_fallback_count[axis])
            )
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_residual_applied_count"] = float(
                runtime_hypothesis_axis_residual_applied_count[axis]
            )
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_residual_sum"] = float(
                runtime_hypothesis_axis_residual_sum[axis]
            )
            metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_residual_absolute_sum"] = float(
                runtime_hypothesis_axis_residual_absolute_sum[axis]
            )
            summary = runtime_hypothesis_axis_evidence_summary[axis]
            evidence_cell_count = int(summary["cell_count"])
            prefix = f"runtime_hypothesis_axis_{'xyz'[axis]}_evidence"
            metrics[f"{prefix}_cell_count"] = float(evidence_cell_count)
            metrics[f"{prefix}_support_count_sum"] = float(summary["support_count_sum"])
            metrics[f"{prefix}_age_seconds_sum"] = float(summary["age_seconds_sum"])
            metrics[f"{prefix}_age_seconds_max"] = (
                float(summary["age_seconds_max"]) if evidence_cell_count else None
            )
            metrics[f"{prefix}_observability_sum"] = float(summary["observability_sum"])
            metrics[f"{prefix}_observability_min"] = (
                float(summary["observability_min"]) if evidence_cell_count else None
            )
            metrics[f"{prefix}_predictive_variance_sum"] = float(summary["predictive_variance_sum"])
            metrics[f"{prefix}_predictive_variance_max"] = (
                float(summary["predictive_variance_max"]) if evidence_cell_count else None
            )
            metrics[f"{prefix}_confidence_margin_sum"] = float(summary["confidence_margin_sum"])
            metrics[f"{prefix}_confidence_margin_min"] = (
                float(summary["confidence_margin_min"]) if evidence_cell_count else None
            )
        for regime, count in zip(
            _RUNTIME_HYPOTHESIS_REGIMES,
            runtime_hypothesis_regime_step_count,
            strict=True,
        ):
            metrics[f"runtime_hypothesis_regime_{regime}_composed_step_count"] = float(count)
        for regime, count in zip(
            _RUNTIME_HYPOTHESIS_REGIMES,
            runtime_hypothesis_regime_query_count,
            strict=True,
        ):
            metrics[f"runtime_hypothesis_regime_{regime}_query_count"] = float(count)
        for (horizon, axis), counts in sorted(
            runtime_hypothesis_horizon_axis_selection_count.items()
        ):
            for candidate, count in zip(runtime_hypothesis_candidates, counts, strict=True):
                metrics[f"runtime_hypothesis@{horizon}_axis_{'xyz'[axis]}_{candidate}_count"] = (
                    float(count)
                )
            supported_count, fallback_count = runtime_hypothesis_horizon_axis_support_count[
                (horizon, axis)
            ]
            metrics[f"runtime_hypothesis@{horizon}_axis_{'xyz'[axis]}_supported_count"] = float(
                supported_count
            )
            metrics[f"runtime_hypothesis@{horizon}_axis_{'xyz'[axis]}_fallback_count"] = float(
                fallback_count
            )
    for horizon, target_count in sorted(forecast_target_count.items()):
        tracked_count = forecast_tracked_count.get(horizon, 0)
        active_count = forecast_active_count.get(horizon, 0)
        metrics[f"forecast_target_count@{horizon}"] = float(target_count)
        metrics[f"forecast_tracked_count@{horizon}"] = float(tracked_count)
        metrics[f"forecast_active_count@{horizon}"] = float(active_count)
        metrics[f"forecast_target_coverage@{horizon}"] = (
            active_count / target_count if target_count else None
        )
        metrics[f"tracked_forecast_active_coverage@{horizon}"] = (
            active_count / tracked_count if tracked_count else None
        )
        metrics[f"model_dropped_forecast_count@{horizon}"] = float(tracked_count - active_count)
        metrics[f"forecast_predictable_target_count@{horizon}"] = float(
            forecast_predictable_target_count.get(horizon, 0)
        )
        metrics[f"forecast_censored_tracked_count@{horizon}"] = float(
            forecast_censored_tracked_count.get(horizon, 0)
        )
    for scenario, accumulator in scenario_accumulators.items():
        metrics.update(
            accumulator.metrics(
                scenario=scenario,
                horizons=configured_horizons,
                detection_threshold_label=detection_threshold_label,
            )
        )
    metrics.update(
        {
            "rgb_global_update_latency_mean_ms": _mean_or_none(global_latencies),
            "rgb_global_update_latency_sum_ms": float(sum(global_latencies)),
            "rgb_global_update_latency_sample_count": float(len(global_latencies)),
            "rgb_fast_update_latency_mean_ms": _mean_or_none(fast_latencies),
            "rgb_fast_update_latency_sum_ms": float(sum(fast_latencies)),
            "rgb_fast_update_latency_sample_count": float(len(fast_latencies)),
            "future_rollout_latency_mean_ms": _mean_or_none(rollout_latencies),
            "future_rollout_latency_sum_ms": float(sum(rollout_latencies)),
            "future_rollout_latency_sample_count": float(len(rollout_latencies)),
            "visible_position_std_mean_m": _mean_or_none(uncertainty_visible),
            "occluded_position_std_mean_m": _mean_or_none(uncertainty_occluded),
            "nonfinite_output_count": float(nonfinite_outputs),
            "evaluated_episodes": float(evaluated_episodes),
            "simulator_external_actuation_object_event_count": float(
                simulator_external_actuation_object_event_count
            ),
            "simulator_external_actuation_interval_count": float(
                simulator_external_actuation_interval_count
            ),
            "simulator_created_object_event_count": float(simulator_created_object_event_count),
            "simulator_removed_object_event_count": float(simulator_removed_object_event_count),
        }
    )
    # Freeze a canonical digest before the optional probe executes. Recovery
    # metrics are appended under a disjoint schema and cannot overwrite the
    # clean primary measurements represented by this digest.
    primary_physical_metrics = _primary_physical_metrics(metrics)
    _require_finite_metrics(primary_physical_metrics)
    primary_physical_metrics_sha256 = hashlib.sha256(
        json.dumps(
            primary_physical_metrics,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    recovery_probe = _RecoveryProbeResult()
    if config.evaluation.recovery_probe_enabled:
        # Release the completed primary runtime before loading the independent
        # probe copy; no primary state or module object crosses this boundary.
        del model
        recovery_probe = _run_recovery_probe(
            config,
            captured_checkpoint.snapshot_path,
            loader=loader,
            device=device,
            runtime_hypothesis_pool=runtime_hypothesis_pool,
            report_progress=report_progress,
        )
    recovery_metrics = recovery_probe.metrics()
    overlapping_metrics = set(metrics) & set(recovery_metrics)
    if overlapping_metrics:
        raise RuntimeError(
            "recovery probe attempted to overwrite primary metrics: "
            + ", ".join(sorted(overlapping_metrics))
        )
    metrics.update(recovery_metrics)
    _require_finite_metrics(metrics)

    limitations = [
        (
            "Evaluation uses the synthetic sphere-world split and does not "
            "establish real-video generalisation."
        ),
        (
            "The analytic_oracle_parameter baseline reads simulator drag only "
            "for comparison; the evaluated model runtime receives RGB and "
            "known calibration only."
        ),
        (
            "Object alignment for metrics uses held-out simulator positions; "
            "alignment is never fed back into the runtime. Ungated Hungarian "
            "assignments are retained for forecast/baseline error comparability "
            "and are reported only as assignment coverage."
        ),
    ]
    if runtime_hypothesis_pool:
        limitations.append(
            "The runtime hypothesis pool is an explicit post-load evaluation "
            "intervention. Its delayed evidence uses RGB associations only; "
            "this report does not change checkpoint or deployment defaults."
        )
    if config.evaluation.recovery_probe_enabled and recovery_probe.correction.count == 0:
        limitations.append(
            "No active matched object/horizon was available for perturbation correction metrics."
        )
    if ordinary_velocity_correction.object_update_count == 0:
        limitations.append(
            "No persistent, distance-gated object was available across an ordinary "
            "non-perturbed prior-to-posterior observation update; ordinary velocity "
            "correction metrics are null."
        )
    if temporal_velocity_measurements.explicit_field_update_count == 0:
        limitations.append(
            "The evaluated runtime exposed no complete explicit temporal velocity "
            "measurement fields; temporal velocity measurement counts are zero and "
            "reported measurement variance is null."
        )
    elif temporal_velocity_measurements.valid_object_count == 0:
        limitations.append(
            "Explicit temporal velocity measurement fields were exposed, but no "
            "distance-independent measurement proposal was marked valid; reported "
            "measurement variance is null."
        )
    if collision_conditioned_forecasts.total_collision_object_horizons == 0:
        limitations.append(
            "No tracked object had a simulator-labelled collision in any future "
            "forecast window; collision-conditioned forecast errors are null."
        )
    if distance_gated_matched_object_frames == 0:
        limitations.append(
            "No current assignment met the 0.5 m detection-distance gate; "
            "distance-gated detection and identity metrics have no true positives."
        )
    if occlusion_transitions.metrics()["occlusion_qualifying_sequence_count"] == 0.0:
        limitations.append(
            "No reliably anchored visible-to-fully-occluded-to-visible target "
            "sequence occurred; sequence-aware occlusion identity and uncertainty "
            "rates are null."
        )
    zero_update_parameters = [
        name
        for name in ("restitution", "drag")
        if identifier_metrics.update_count.get(name, 0) == 0
    ]
    if zero_update_parameters:
        limitations.append(
            "The RGB runtime produced zero identifier updates above the "
            "1e-3 gate threshold for "
            + ", ".join(zero_update_parameters)
            + "; updated-parameter MAE is unavailable for those parameters."
        )
    no_informative_directional_updates = [
        name
        for name in ("restitution", "drag")
        if directional_parameter_metrics[f"informative_{name}_update_count"] == 0.0
    ]
    if no_informative_directional_updates:
        limitations.append(
            "No persistent, distance-gated runtime identifier update coincided "
            "with ground-truth informative evidence for "
            + ", ".join(no_informative_directional_updates)
            + "; directional before/after parameter metrics are unavailable."
        )
    runtime_hypothesis_policy: dict[str, Any] | None = None
    if runtime_hypothesis_pool:
        runtime_hypothesis_policy = {
            "policy_version": (
                "evidence_bounded_entity_axis_regime_horizon_v6"
                if config.runtime.hypothesis_online_acceleration_enabled
                else "evidence_bounded_entity_axis_regime_horizon_v5"
            ),
            "candidates": [
                {"name": "learned", "parameters": {}},
                {"name": "constant_velocity", "parameters": {"damping": 0.0}},
                {
                    "name": "damped_constant_velocity",
                    "parameters": {"damping": 0.05},
                },
                {
                    "name": "ballistic_contact",
                    "parameters": {"ground_height": 0.0, "event_logit": 5.0},
                },
            ]
            + (
                [
                    {
                        "name": "online_local_acceleration",
                        "parameters": {
                            "minimum_support_count": (
                                config.runtime.hypothesis_online_acceleration_minimum_support_count
                            ),
                            "maximum_acceleration_mps2": (
                                config.runtime.hypothesis_online_acceleration_maximum_mps2
                            ),
                            "minimum_delta_time_seconds": (
                                config.model.rgb.temporal_velocity_min_dt
                            ),
                        },
                    }
                ]
                if config.runtime.hypothesis_online_acceleration_enabled
                else []
            ),
            "evidence_horizons_seconds": list(config.runtime.hypothesis_evidence_horizons_seconds),
            "axis_independent_axes": list(config.runtime.hypothesis_axis_independent_axes),
            "axis_prior_strength": config.runtime.hypothesis_axis_prior_strength,
            "evidence_decay": config.runtime.hypothesis_evidence_decay,
            "temperature": 1.0,
            "score": (
                "gaussian_nll_position_plus_optional_rgb_temporal_velocity_"
                "with_predictive_and_measurement_variance"
            ),
            "selection_locality": "persistent_entity_axis_interaction_regime_exact_horizon",
            "local_applicability_enabled": (config.runtime.hypothesis_local_applicability_enabled),
            "minimum_support_count": config.runtime.hypothesis_minimum_support_count,
            "maximum_evidence_age_seconds": (
                config.runtime.hypothesis_maximum_evidence_age_seconds
            ),
            "minimum_observability": config.runtime.hypothesis_minimum_observability,
            "minimum_confidence_margin": (config.runtime.hypothesis_minimum_confidence_margin),
            "velocity_evidence_weight": (config.runtime.hypothesis_velocity_evidence_weight),
            "velocity_nonregression_gate_enabled": (
                config.runtime.hypothesis_velocity_nonregression_gate_enabled
            ),
            "residual_correction_gain_by_axis": list(
                config.runtime.hypothesis_residual_correction_gain_by_axis
            ),
            "robust_influence_delta": config.runtime.hypothesis_robust_influence_delta,
            "composition_step_seconds": config.runtime.hypothesis_composition_step_seconds,
            **(
                {"online_acceleration_enabled": True}
                if config.runtime.hypothesis_online_acceleration_enabled
                else {}
            ),
            "unsupported_query_policy": "learned_fallback",
            "composition": (
                (
                    "bounded_short_step_coherent_state_plus_output_only_causal_residual"
                    if any(config.runtime.hypothesis_residual_correction_gain_by_axis)
                    else "bounded_short_step_coherent_state"
                )
                if config.runtime.hypothesis_composition_step_seconds is not None
                else (
                    "coherent_axis_endpoint_plus_output_only_causal_residual"
                    if any(config.runtime.hypothesis_residual_correction_gain_by_axis)
                    else "coherent_axis_state_endpoint_splice_diagnostic_only"
                )
            ),
            "timestamp_tolerance_seconds": (config.runtime.hypothesis_timestamp_tolerance_seconds),
        }
        runtime_hypothesis_policy["fingerprint_sha256"] = hashlib.sha256(
            json.dumps(
                runtime_hypothesis_policy,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": captured_checkpoint.sha256,
        "checkpoint_byte_count": captured_checkpoint.byte_count,
        "checkpoint_identity_source": "captured_pre_evaluation_immutable_byte_snapshot",
        "checkpoint_step": checkpoint_step,
        "checkpoint_simulator_version": checkpoint_simulator_version,
        "evaluation_simulator_version": SIMULATOR_VERSION,
        "checkpoint_specification_version": checkpoint_specification_version,
        "evaluation_specification_version": SPECIFICATION_VERSION,
        "checkpoint_source_provenance": checkpoint_source_provenance,
        "evaluation_source_provenance": dict(evaluation_source_provenance),
        "evaluation_metric_schema_version": _EVALUATION_METRIC_SCHEMA_VERSION,
        "resolved_evaluation_config_sha256": resolved_evaluation_protocol["resolved_config_sha256"],
        "resolved_evaluation_protocol": resolved_evaluation_protocol,
        "resolved_evaluation_protocol_sha256": resolved_evaluation_protocol_sha256,
        # Backwards-compatible alias for consumers that predate the explicit
        # checkpoint-versus-evaluation protocol split.
        "simulator_version": SIMULATOR_VERSION,
        "scenario_mixture": list(config.simulator.scenario_mixture),
        "per_scenario_metrics_schema": _PER_SCENARIO_METRIC_SCHEMA,
        "per_scenario_metrics_status": "diagnostic_only_not_checkpoint_promotion_complete",
        "per_scenario_metrics_known_omissions": [
            "nonfinite_evidence",
            "physical_baselines",
            "configured_support_floor_markers",
        ],
        "per_scenario_metrics_scenarios": list(config.simulator.scenario_mixture),
        "per_scenario_metrics_horizons": list(configured_horizons),
        "per_scenario_metrics_source": (
            "same_primary_rollout_tensors_and_support_masks_as_pooled_metrics"
        ),
        "resolved_scenarios": {
            scenario: asdict(
                SphereWorldConfig.from_config(config)
                .for_scenario(scenario)
                .for_distribution("ood" if split == "ood" else "in_distribution")
            )
            for scenario in config.simulator.scenario_mixture
        },
        "evaluation_episode_scenarios": [
            config.simulator.scenario_mixture[int(seed) % len(config.simulator.scenario_mixture)]
            for seed in resolved_seed_protocol.manifest.seeds
        ],
        "split": split,
        "episodes": evaluated_episodes,
        "batches": len(loader),
        "device": str(device),
        "precision": resolved_device.precision,
        "evaluation_runtime_environment": {
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "platform_node": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": resolved_device.torch_version,
            "requested_device": resolved_device.requested,
            "resolved_device": str(device),
            "precision": resolved_device.precision,
            "mps_built": resolved_device.mps_built,
            "mps_available": resolved_device.mps_available,
            "cuda_available": resolved_device.cuda_available,
        },
        "rgb_only": True,
        "primary_online_pass_evaluator_state_perturbation_free": True,
        # Deprecated compatibility alias.  Its explicit scope prevents this
        # from being read as a claim that simulator interventions were absent.
        "primary_online_pass_intervention_free": True,
        "primary_online_pass_intervention_free_scope": (
            "evaluator_injected_state_perturbations_only"
        ),
        "primary_online_pass_simulator_external_actuation_present": bool(
            simulator_external_actuation_object_event_count
        ),
        "primary_online_pass_simulator_external_actuation_object_event_count": (
            simulator_external_actuation_object_event_count
        ),
        "primary_online_pass_simulator_external_actuation_interval_count": (
            simulator_external_actuation_interval_count
        ),
        "primary_posterior_trace_sha256": primary_posterior_trace.hexdigest(),
        "primary_posterior_trace_frame_count": primary_posterior_trace.frame_count,
        "primary_posterior_trace_schema": "world_belief_tensor_fields_v1",
        "primary_physical_metrics_sha256": primary_physical_metrics_sha256,
        "primary_physical_metrics_hashed_keys": sorted(primary_physical_metrics),
        "primary_physical_metrics_scope": (
            "clean_primary_metrics_before_isolated_recovery_probe_append"
        ),
        "primary_physical_metrics_hash_excludes": (
            _primary_physical_metrics_hash_exclusion_declaration()
        ),
        "recovery_probe_enabled": config.evaluation.recovery_probe_enabled,
        "recovery_probe_runtime_isolation": (
            "independent_runtime_shared_immutable_checkpoint_snapshot_clean_prefix_replay"
            if config.evaluation.recovery_probe_enabled
            else None
        ),
        # Compatibility field: the primary pass can no longer apply synthetic
        # perturbations, regardless of whether the separate probe is enabled.
        "evaluation_perturbations_applied": False,
        "oracle_runtime_input_used": False,
        "runtime_hypothesis_pool_enabled": runtime_hypothesis_pool,
        "runtime_hypothesis_pool_policy": runtime_hypothesis_policy,
        "simulator_state_usage": "metrics_and_baselines_only",
        "deterministic_forecast_support_mask_source": (
            "persistent_target_match_and_target_active_and_no_unseen_external_"
            "actuation_anywhere_in_coupled_scene_over_(anchor_frame,target_frame]"
        ),
        "stochastic_calibration_support_mask_source": (
            "persistent_target_match_and_target_active_including_hidden_external_actuation_outcomes"
        ),
        "collision_conditioned_mask_source": (
            "deterministic_forecast_support_and_evaluation_only_simulator_"
            "collision_any_in_(anchor_frame,target_frame]"
        ),
        "collision_class_conditioned_schema": [
            "pair_only",
            "ground_only",
            "wall_only",
            "other_only",
            "compound",
            "no_collision",
        ],
        "collision_class_conditioned_mask_source": (
            "deterministic_forecast_support_and_mutually_exclusive_evaluation_only_"
            "simulator_event_kinds_any_in_(anchor_frame,target_frame]"
        ),
        "parameter_metric_mask_source": "runtime_identifier_diagnostics",
        "directional_parameter_metric_mask_source": (
            "persistent_distance_gated_runtime_update_and_evaluation_only_"
            "ground_truth_informative_evidence"
        ),
        "current_detection_distance_threshold_m": (_CURRENT_DETECTION_DISTANCE_THRESHOLD_M),
        "identity_metric_match_source": "distance_gated_current_detection",
        "forecast_identity_metric_match_source": (
            "distance_gated_anchor_persistent_target_mapping_compared_with_"
            "distance_gated_forecast_hungarian_target_assignment_per_horizon"
        ),
        "current_velocity_metric_match_source": "distance_gated_current_detection",
        "velocity_metric_additive_evidence": (
            "pooled_and_xyz_sse_coordinate_counts_current_and_each_horizon"
        ),
        "uncertainty_metric_additive_evidence": (
            "gaussian_nll_sharpness_sums_and_coordinate_counts_pooled_and_xyz_"
            "current_and_each_horizon"
        ),
        "ordinary_velocity_correction_scope": (
            "timestamp_advanced_prior_to_posterior_nonperturbed_fresh_observation_"
            "same_persistent_slot_and_distance_gated_current_detection"
        ),
        "temporal_velocity_measurement_metric_source": (
            "fresh_runtime_last_measurements_explicit_auxiliary_fields_only"
        ),
        "occlusion_visible_fraction_threshold": (occlusion_transitions.visible_fraction_threshold),
        "occlusion_fully_hidden_fraction_threshold": (
            occlusion_transitions.fully_occluded_fraction_threshold
        ),
        "occlusion_tracking_key": "pre_occlusion_target_to_persistent_prediction_id",
        "occlusion_hidden_tracking_localization_gate_m": None,
        **resolved_seed_protocol.metadata(),
    }
    json_path, markdown_path = write_evaluation_report(
        output,
        metadata=metadata,
        metrics=metrics,
        limitations=limitations,
    )
    report_progress(
        "completed",
        evaluated_episodes=evaluated_episodes,
        output_directory=str(output),
    )
    return {
        "output_directory": str(output),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "split": split,
        "seed_protocol": resolved_seed_protocol.name,
        "episode_seeds": list(resolved_seed_protocol.manifest.seeds),
        "device": str(device),
        "rgb_only": True,
        "oracle_runtime_input_used": False,
        "runtime_hypothesis_pool_enabled": runtime_hypothesis_pool,
        "metrics": metrics,
        "limitations": limitations,
    }


__all__ = ["evaluate_checkpoint"]
