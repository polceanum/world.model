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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from world_model.belief import MotionMode, WorldBelief
from world_model.observations import (
    MeasurementSet,
    ObservationContext,
    ObservationPacket,
    SensorContext,
)
from world_model.runtime import OnlineWorldModel
from world_model.training.event_windows import observation_window_query_plan
from world_model.training.losses import (
    balanced_binary_cross_entropy,
    gaussian_nll,
    masked_huber,
    posterior_improvement_hinge,
    weighted_total,
)
from world_model.training.matching import match_measurements_to_targets
from world_model.training.perturbations import perturb_belief
from world_model.utils.config import OrpheusConfig

_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M = 0.5


@dataclass
class TrainingBatchResult:
    """Losses and detached diagnostics for one optimiser step."""

    total_loss: Tensor
    loss_terms: dict[str, Tensor]
    metrics: dict[str, float]
    phase: str


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

    def required(name: str) -> float:
        if name not in additive_metrics:
            raise RuntimeError(f"missing additive physical validation metric {name!r}")
        value = float(additive_metrics[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"additive physical validation metric {name!r} is invalid")
        return value

    def rmse(sse_name: str, count_name: str) -> float:
        squared_error = required(sse_name)
        count = required(count_name)
        if count <= 0:
            raise RuntimeError(f"physical validation metric {count_name!r} has no support")
        return math.sqrt(squared_error / count)

    def bounded_ratio(numerator_name: str, denominator_name: str) -> float:
        numerator = required(numerator_name)
        denominator = required(denominator_name)
        if denominator <= 0:
            return 0.0
        return min(1.0, max(0.0, numerator / denominator))

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
    }
    true_positive = required("physical_collision_true_positive_count")
    false_positive = required("physical_collision_false_positive_count")
    false_negative = required("physical_collision_false_negative_count")
    output["validation_collision_f1"] = _f1_from_confusion(
        true_positive,
        false_positive,
        false_negative,
    )[0]
    seen_offsets: set[int] = set()
    for horizon in config.evaluation.horizons_seconds:
        frame_offset = max(1, int(round(float(horizon) * config.simulator.frame_rate)))
        if frame_offset in seen_offsets:
            continue
        seen_offsets.add(frame_offset)
        physical_seconds = frame_offset / config.simulator.frame_rate
        physical_suffix = f"@{physical_seconds:.3f}s"
        output[f"validation_position_rmse{physical_suffix}"] = rmse(
            f"physical_rollout_position{physical_suffix}_sse",
            f"physical_rollout_position{physical_suffix}_coordinate_count",
        )
        output[f"validation_forecast_target_coverage{physical_suffix}"] = bounded_ratio(
            f"physical_forecast_active_count{physical_suffix}",
            f"physical_forecast_target_count{physical_suffix}",
        )
    return output


@dataclass
class PersistentTargetMatcher:
    """Keep training-only simulator targets aligned to persistent belief IDs.

    A fresh positional Hungarian match bootstraps new runtime tracks. Once an
    internal object ID is mapped, that target slot is retained while both
    remain active. This prevents close contacts from silently swapping the
    velocity/event supervision of two nearby objects.
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
                rows, columns = linear_sum_assignment(np.asarray(cost))
                for row, column in zip(rows, columns, strict=True):
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
    masks = {
        "matched": matched,
        "existence": matched,
    }
    losses = module.training_losses(outputs, targets, masks)
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

    target_values, target_mask, visibility = _target_measurement_values(batch, frame_index)
    aligned = gather_target_slots(target_values, target_indices)
    aligned_target_mask = (
        gather_target_slots(target_mask.unsqueeze(-1), target_indices).squeeze(-1).bool()
    )
    aligned_visibility = gather_target_slots(
        visibility.unsqueeze(-1),
        target_indices,
    ).squeeze(-1)
    supervised = matched_slots & aligned_target_mask & measurements.measurement_mask

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
    masks = {
        "matched": supervised,
        "existence": supervised,
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
    existence_threshold: float,
    distance_threshold_m: float = 0.5,
) -> dict[str, float]:
    """Report calibrated proposal accuracy in sensor and world coordinates.

    The training objective contains differently scaled terms and a Gaussian NLL
    that may legitimately be negative.  Those values are useful optimisation
    diagnostics, but they are not a truthful checkpoint-selection proxy for
    localization.  This metric uses the same transparent Hungarian alignment
    and the calibrated RGB backprojection used by the runtime.
    """

    target_values, target_mask, _ = _target_measurement_values(batch, frame_index)
    aligned, matched, target_indices = match_measurements_to_targets(
        measurements.values,
        target_values,
        target_mask,
        existence_logits=measurements.existence_logits,
    )
    if not matched.any():
        return {
            "rgb_centre_mae_normalized": math.inf,
            "rgb_inverse_depth_mae": math.inf,
            "rgb_world_position_mae_m": math.inf,
            "rgb_detection_recall_at_0_5m": 0.0,
            "rgb_detection_precision_at_0_5m": 0.0,
        }
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
        measurements.existence_logits.sigmoid() >= existence_threshold
    )
    close = matched & (world_error <= distance_threshold_m)
    true_positive = (close & confident).sum()
    target_count = target_mask.sum().clamp_min(1)
    proposal_count = confident.sum().clamp_min(1)
    return {
        "rgb_centre_mae_normalized": float(sensor_error[..., :2][matched].mean().cpu()),
        "rgb_inverse_depth_mae": float(sensor_error[..., 3][matched].mean().cpu()),
        "rgb_world_position_mae_m": float(world_error[matched].mean().cpu()),
        "rgb_detection_recall_at_0_5m": float((true_positive / target_count).cpu()),
        "rgb_detection_precision_at_0_5m": float((true_positive / proposal_count).cpu()),
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


def pretrain_rgb_measurements(
    model: OnlineWorldModel,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    frame_index: int,
) -> TrainingBatchResult:
    """Train the real global RGB proposal path from simulator labels."""

    if "rgb" not in model.observation_modules:
        raise ValueError("RGB measurement pretraining requires the RGB module")
    packet = make_rgb_packet(batch, frame_index)
    module = model.observation_modules["rgb"]
    measurements = module.initialise_measurements(
        [packet],
        _observation_context(packet, config, training=True),
    )
    details = supervised_measurement_losses(module, measurements, batch, frame_index)
    measurement = _weighted_measurement_total(
        details,
        config.training.measurement_loss_weights,
    )
    terms = {"measurement": measurement}
    total = weighted_total(terms, config.training.loss_weights)
    metrics = {name: float(value.detach().cpu()) for name, value in details.items()}
    metrics.update(
        measurement_localization_metrics(
            measurements,
            batch,
            frame_index,
            existence_threshold=config.model.rgb.existence_threshold,
        )
    )
    metrics["proposals_above_birth_threshold"] = float(
        (measurements.existence_logits.sigmoid().detach() >= config.model.rgb.existence_threshold)
        .sum()
        .cpu()
    )
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
    batch, belief_count = objects.active.shape
    if target_position.ndim != 3 or target_position.shape[0] != batch:
        raise ValueError("target_position must have shape [B,N,3]")
    if target_active.shape != target_position.shape[:2]:
        raise ValueError("target_active must have shape [B,N]")
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
    for batch_index in range(batch):
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
            torch.as_tensor(rows, device=belief_slots.device, dtype=torch.int64)
        ]
        selected_targets = target_slots[
            torch.as_tensor(columns, device=target_slots.device, dtype=torch.int64)
        ]
        indices[batch_index, selected_beliefs] = selected_targets
        matched[batch_index, selected_beliefs] = True
    return indices, matched


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

    for name in ("rollout_position", "rollout_velocity", "correction_future"):
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


def _belief_state_losses(
    belief: WorldBelief,
    batch: Mapping[str, Any],
    frame_index: int,
    *,
    indices: Tensor | None = None,
    matched: Tensor | None = None,
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
    state_position = masked_huber(objects.position, aligned_position, matched)
    state_velocity = masked_huber(objects.velocity, aligned_velocity, matched)
    uncertainty = gaussian_nll(
        objects.position,
        aligned_position,
        objects.fast_log_variance[..., :3],
        matched,
    )
    existence_target = matched.to(objects.existence_logit.dtype)
    existence = F.binary_cross_entropy_with_logits(
        objects.existence_logit,
        existence_target,
    )

    speed = torch.linalg.vector_norm(aligned_velocity, dim=-1)
    drag_observable = matched & (speed >= 0.25)
    collision_seen = batch["events"]["collision"][:, : frame_index + 1].any(dim=1)
    aligned_collision_seen = (
        gather_target_slots(collision_seen.unsqueeze(-1), indices).squeeze(-1).bool()
    )
    restitution_observable = matched & aligned_collision_seen
    aligned_drag = gather_target_slots(target_objects["drag"][:, frame_index], indices)
    aligned_restitution = gather_target_slots(
        target_objects["restitution"][:, frame_index], indices
    )
    parameter_drag = masked_huber(objects.drag, aligned_drag, drag_observable)
    parameter_restitution = masked_huber(
        objects.restitution,
        aligned_restitution,
        restitution_observable,
    )
    return (
        {
            "state_position": state_position,
            "state_velocity": state_velocity,
            "uncertainty_position_nll": uncertainty,
            "existence_belief": existence,
            "parameter_drag": parameter_drag,
            "parameter_restitution": parameter_restitution,
        },
        indices,
        matched,
    )


def _rollout_loss_result(
    model: OnlineWorldModel,
    belief: WorldBelief,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    frame_index: int,
    indices: Tensor,
    matched: Tensor,
) -> _RolloutLossResult:
    total_frames = int(batch["rgb"].shape[1])
    frame_offsets, query_seconds, horizon_weights = _valid_rollout_offsets(
        config,
        frame_index,
        total_frames,
    )
    reference = belief.objects.position
    if not frame_offsets:
        return _RolloutLossResult(
            losses={
                "rollout_position": reference.sum() * 0,
                "rollout_velocity": reference.sum() * 0,
                "event_collision": reference.sum() * 0,
            },
            frame_offsets=[],
            query_seconds=[],
            positions=None,
            velocities=None,
            position_log_variance=None,
            active_mask=None,
            physical_metrics={},
        )
    event_query_plan = observation_window_query_plan(
        frame_offsets,
        frame_rate=config.simulator.frame_rate,
    )
    trajectory = model.dynamics.rollout(
        belief,
        event_query_plan.query_seconds,
        return_events=True,
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
    position_losses: list[Tensor] = []
    position_axis_losses: dict[str, list[Tensor]] = {
        "x": [],
        "y": [],
        "z": [],
    }
    velocity_losses: list[Tensor] = []
    horizon_losses: dict[str, Tensor] = {}
    event_losses: list[Tensor] = []
    event_weights: list[float] = []
    physical_metrics: dict[str, float] = {}
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
        target_position = gather_target_slots(
            batch["objects"]["position"][:, target_index], indices
        )
        target_velocity = gather_target_slots(
            batch["objects"]["velocity"][:, target_index], indices
        )
        position_loss = masked_huber(
            target_positions[:, query_index],
            target_position,
            valid,
        )
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            position_axis_losses[axis_name].append(
                masked_huber(
                    target_positions[:, query_index, :, axis_index],
                    target_position[:, :, axis_index],
                    valid,
                )
            )
        velocity_loss = masked_huber(
            target_velocities[:, query_index],
            target_velocity,
            valid,
        )
        position_losses.append(position_loss)
        velocity_losses.append(velocity_loss)
        seconds = query_seconds[query_index]
        horizon_losses[rollout_horizon_loss_key("rollout_position", seconds)] = position_loss
        horizon_losses[rollout_horizon_loss_key("rollout_velocity", seconds)] = velocity_loss
        horizon_suffix = f"@{seconds:.3f}s"
        _add_squared_error_metrics(
            physical_metrics,
            prefix=f"physical_rollout_position{horizon_suffix}",
            prediction=target_positions[:, query_index],
            target=target_position,
            mask=valid,
        )
        _add_gaussian_coverage_metrics(
            physical_metrics,
            prefix=f"physical_rollout_position_coverage90{horizon_suffix}",
            mean=target_positions[:, query_index],
            target=target_position,
            log_variance=target_position_log_variance[:, query_index],
            mask=valid,
        )
        _add_squared_error_metrics(
            physical_metrics,
            prefix=f"physical_rollout_velocity{horizon_suffix}",
            prediction=target_velocities[:, query_index],
            target=target_velocity,
            mask=valid,
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
            if valid.any():
                event_losses.append(
                    balanced_binary_cross_entropy(
                        event_scores,
                        event_target,
                        valid,
                        maximum_positive_weight=(config.training.collision_positive_weight_max),
                    )
                )
                event_weights.append(horizon_weights[query_index])
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
                    ] = float((confusion_mask & valid).sum().cpu())

    def weighted_mean(losses: list[Tensor], weights: list[float]) -> Tensor:
        if not losses:
            return reference.sum() * 0
        weight = reference.new_tensor(weights)
        return (torch.stack(losses) * weight).sum() / weight.sum().clamp_min(1.0e-8)

    return _RolloutLossResult(
        losses={
            "rollout_position": weighted_mean(position_losses, horizon_weights),
            **{
                f"rollout_position_{axis_name}": weighted_mean(
                    axis_losses,
                    horizon_weights,
                )
                for axis_name, axis_losses in position_axis_losses.items()
            },
            "rollout_velocity": weighted_mean(velocity_losses, horizon_weights),
            "event_collision": weighted_mean(event_losses, event_weights),
            **horizon_losses,
        },
        frame_offsets=frame_offsets,
        query_seconds=query_seconds,
        positions=target_positions,
        velocities=target_velocities,
        position_log_variance=target_position_log_variance,
        active_mask=target_active_mask,
        physical_metrics=physical_metrics,
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
    def total(*names: str) -> Tensor:
        selected = [details[name] for name in names if name in details]
        return _mean_losses(selected, reference)

    state_position = details.get("state_position", reference.sum() * 0)
    state_velocity = details.get("state_velocity", reference.sum() * 0)
    rollout_position = details.get("rollout_position", reference.sum() * 0)
    rollout_position_axes = {
        name: details.get(name, reference.sum() * 0)
        for name in (
            "rollout_position_x",
            "rollout_position_y",
            "rollout_position_z",
        )
    }
    rollout_velocity = details.get("rollout_velocity", reference.sum() * 0)
    return {
        "measurement": details.get("measurement", reference.sum() * 0),
        "state_position": state_position,
        "state_velocity": state_velocity,
        "state": total("state_position", "state_velocity"),
        "rollout_position": rollout_position,
        **rollout_position_axes,
        "rollout_velocity": rollout_velocity,
        "rollout": total("rollout_position", "rollout_velocity"),
        "event": details.get("event_collision", reference.sum() * 0),
        "parameter": total("parameter_drag", "parameter_restitution"),
        "existence": details.get("existence_belief", reference.sum() * 0),
        "uncertainty": details.get("uncertainty_position_nll", reference.sum() * 0),
        # Retain the specification's small correction-sparsity regulariser,
        # but pair it with explicit guards against harmful posterior updates.
        "correction": total(
            "correction_magnitude",
            "correction_current",
            "correction_future",
        ),
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
    numerically unchanged.
    """

    aggregate_families = {
        "state": ("state_position", "state_velocity"),
        "rollout": ("rollout_position", "rollout_velocity"),
    }
    rollout_position_axes = (
        "rollout_position_x",
        "rollout_position_y",
        "rollout_position_z",
    )
    use_axis_position = any(name in weights for name in rollout_position_axes)
    selected: dict[str, Tensor] = {
        name: value
        for name, value in terms.items()
        if name
        not in {component for components in aggregate_families.values() for component in components}
        and name not in aggregate_families
        and name not in rollout_position_axes
    }
    for aggregate, components in aggregate_families.items():
        if any(component in weights for component in components):
            for component in components:
                if component == "rollout_position" and use_axis_position:
                    for axis_name in rollout_position_axes:
                        selected[axis_name] = terms[axis_name]
                else:
                    selected[component] = terms[component]
        else:
            selected[aggregate] = terms[aggregate]
    return weighted_total(selected, weights)


def select_closed_loop_window(
    batch: Mapping[str, Any],
    window_steps: int,
    *,
    event_condition_probability: float = 0.5,
    maximum_rollout_frame_offset: int | None = None,
    minimum_rollout_frame_offset: int | None = None,
    long_horizon_probability: float = 0.0,
) -> int:
    """Sample a valid TBPTT window, preferentially covering collision frames.

    The returned start is stochastic under the trainer's seeded Python RNG.
    Collision conditioning has first priority so late events remain trainable.
    Among the remaining windows, long-horizon conditioning can require the
    first anchor to support the maximum configured rollout. Labels select only
    the loss window; they are never passed to the RGB runtime.
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
    collision_frames: list[int] = []
    if isinstance(collision, Tensor):
        if collision.ndim < 2 or tuple(collision.shape[:2]) != tuple(rgb.shape[:2]):
            raise ValueError("events.collision must begin with batch/time axes [B,T]")
        collision_by_frame = (
            collision.bool()
            .reshape(
                collision.shape[0],
                collision.shape[1],
                -1,
            )
            .any(dim=(0, 2))
        )
        collision_frames = (
            torch.nonzero(collision_by_frame, as_tuple=False).flatten().detach().cpu().tolist()
        )
    if collision_frames and random.random() < event_condition_probability:
        event_frame = int(random.choice(collision_frames))
        minimum_start = max(0, event_frame - window_steps + 1)
        maximum_event_start = min(maximum_start, event_frame)
        if event_frame > 0:
            maximum_event_start = min(maximum_event_start, event_frame - 1)
        if minimum_start <= maximum_event_start:
            return random.randint(minimum_start, maximum_event_start)
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
) -> None:
    """Advance the real RGB filter to a mid-episode loss window.

    Prefix frames update the persistent belief, lifecycle, modality caches,
    association state, and scheduler exactly as online inference would.  They
    are deliberately outside the autograd graph; the selected TBPTT window
    starts from their detached numerical posterior instead of a cold reset.
    """

    if window_start < 0:
        raise ValueError("window_start must be nonnegative")
    with torch.no_grad():
        for frame_index in range(window_start):
            belief = model.ingest(make_rgb_packet(batch, frame_index))
            target_matcher.match(
                belief,
                batch["objects"]["position"][:, frame_index],
                batch["objects"]["active"][:, frame_index].bool(),
            )
    if window_start:
        model.detach_state()


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
    compute_future_correction: bool = True,
) -> TrainingBatchResult:
    """Run one causal RGB-only sequence window through the real runtime.

    The belief is never reset to simulator state.  Labels are read only after
    each RGB ingest to compute supervised losses. Validation may disable the
    extra prior future rollout used only by the correction-improvement guard;
    current correction and every posterior physical forecast remain measured.
    """

    if config.runtime.modality != "rgb":
        raise ValueError("closed-loop milestone training requires runtime.modality=rgb")
    rgb = batch["rgb"]
    if not isinstance(rgb, Tensor) or rgb.ndim != 5:
        raise ValueError("closed-loop batch must contain rgb [B,T,3,H,W]")
    batch_size, total_frames = rgb.shape[:2]
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
    _burn_in_causal_prefix(model, batch, window_start, target_matcher)
    detail_lists: dict[str, list[Tensor]] = {}
    current_correction_improvements: list[float] = []
    future_correction_improvements: list[float] = []
    perturbed_updates = 0
    matched_count = 0
    target_object_frames = 0
    predicted_object_frames = 0
    fast_supervised_frames = 0
    physical_metrics: dict[str, float] = {}
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

    for frame_index in range(window_start, window_stop):
        packet = make_rgb_packet(batch, frame_index)
        score_rollout = frame_index in rollout_anchor_set
        prior_belief: WorldBelief | None = None
        prior_rollout = None
        frame_offsets: list[int] = []
        query_seconds: list[float] = []
        if model.belief is not None:
            source_belief = model.belief
            requested = source_belief.timestamp.new_full(
                source_belief.timestamp.shape, packet.timestamp
            )
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
            prior = model.dynamics.predict(source_belief, requested - source_belief.timestamp)
            prior_belief = prior
            if score_rollout:
                frame_offsets, query_seconds, _ = _valid_rollout_offsets(
                    config,
                    frame_index,
                    total_frames,
                )
                if frame_offsets and compute_future_correction:
                    prior_rollout = model.dynamics.rollout(
                        prior,
                        query_seconds,
                        return_events=False,
                    )
            if include_measurement_supervision:
                module = model.observation_modules["rgb"]
                predicted = module.project(
                    prior,
                    SensorContext(
                        sensor_id=packet.sensor_id,
                        timestamp=packet.timestamp,
                        calibration=packet.calibration,
                        frame_id=packet.frame_id,
                        image_size=packet.metadata["image_size"],
                        metadata=packet.metadata,
                    ),
                )
                if bool(predicted.valid_mask.any()):
                    fast_measurements, _ = module.encode_measurements(
                        [packet],
                        prior,
                        predicted,
                        model.state.caches.get(packet.sensor_id),
                    )
                    belief_target_indices, belief_matched_slots = target_matcher.match(
                        prior,
                        batch["objects"]["position"][:, frame_index],
                        batch["objects"]["active"][:, frame_index].bool(),
                    )
                    valid_belief_indices = predicted.belief_indices >= 0
                    target_indices = (
                        gather_target_slots(
                            belief_target_indices.unsqueeze(-1),
                            predicted.belief_indices,
                        )
                        .squeeze(-1)
                        .to(torch.int64)
                    )
                    target_indices = torch.where(
                        valid_belief_indices,
                        target_indices,
                        torch.full_like(target_indices, -1),
                    )
                    matched_slots = (
                        gather_target_slots(
                            belief_matched_slots.unsqueeze(-1),
                            predicted.belief_indices,
                        )
                        .squeeze(-1)
                        .bool()
                        & valid_belief_indices
                    )
                    fast_supervised = supervised_slot_measurement_losses(
                        module,
                        fast_measurements,
                        batch,
                        frame_index,
                        target_indices=target_indices,
                        matched_slots=matched_slots,
                    )
                    add(
                        "measurement",
                        _weighted_measurement_total(
                            fast_supervised,
                            config.training.measurement_loss_weights,
                        ),
                    )
                    for name, value in fast_supervised.items():
                        add(f"fast_{name}", value)
                    fast_supervised_frames += 1

        belief = model.ingest(packet)
        indices, matched = target_matcher.match(
            belief,
            batch["objects"]["position"][:, frame_index],
            batch["objects"]["active"][:, frame_index].bool(),
        )
        current, indices, matched = _belief_state_losses(
            belief,
            batch,
            frame_index,
            indices=indices,
            matched=matched,
        )
        matched_count += int(matched.sum().detach().cpu())
        frame_target_count = int(batch["objects"]["active"][:, frame_index].sum().detach().cpu())
        frame_predicted_count = int(belief.objects.active.sum().detach().cpu())
        target_object_frames += frame_target_count
        predicted_object_frames += frame_predicted_count
        distance_gated_target_object_frames += frame_target_count
        distance_gated_predicted_object_frames += frame_predicted_count
        aligned_position = gather_target_slots(
            batch["objects"]["position"][:, frame_index],
            indices,
        )
        distance_gated_matched = _distance_gate_physical_matches(
            belief.objects.position,
            aligned_position,
            matched,
        )
        distance_gated_matched_count += int(distance_gated_matched.sum().detach().cpu())
        aligned_velocity = gather_target_slots(
            batch["objects"]["velocity"][:, frame_index],
            indices,
        )
        state_physical: dict[str, float] = {}
        _add_squared_error_metrics(
            state_physical,
            prefix="physical_state_position",
            prediction=belief.objects.position,
            target=aligned_position,
            mask=matched,
        )
        _add_squared_error_metrics(
            state_physical,
            prefix="physical_state_velocity",
            prediction=belief.objects.velocity,
            target=aligned_velocity,
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
        _accumulate_float_metrics(physical_metrics, state_physical)
        target_ids = batch["objects"].get("id")
        if isinstance(target_ids, Tensor):
            for batch_index in range(batch_size):
                belief_slots = torch.nonzero(
                    matched[batch_index],
                    as_tuple=False,
                ).flatten()
                for belief_slot_tensor in belief_slots:
                    belief_slot = int(belief_slot_tensor)
                    target_slot = int(indices[batch_index, belief_slot])
                    target_id = int(target_ids[batch_index, frame_index, target_slot])
                    predicted_id = int(
                        belief.objects.object_id[batch_index, belief_slot].detach().cpu()
                    )
                    if target_id < 0 or predicted_id < 0:
                        continue
                    object_frame_associations += 1
                    previous = last_predicted_id_for_target[batch_index].get(target_id)
                    if previous is not None and previous != predicted_id:
                        identity_switches += 1
                    last_predicted_id_for_target[batch_index][target_id] = predicted_id
                distance_gated_belief_slots = torch.nonzero(
                    distance_gated_matched[batch_index],
                    as_tuple=False,
                ).flatten()
                for belief_slot_tensor in distance_gated_belief_slots:
                    belief_slot = int(belief_slot_tensor)
                    target_slot = int(indices[batch_index, belief_slot])
                    target_id = int(target_ids[batch_index, frame_index, target_slot])
                    predicted_id = int(
                        belief.objects.object_id[batch_index, belief_slot].detach().cpu()
                    )
                    if target_id < 0 or predicted_id < 0:
                        continue
                    distance_gated_object_frame_associations += 1
                    previous = last_distance_gated_predicted_id_for_target[batch_index].get(
                        target_id
                    )
                    if previous is not None and previous != predicted_id:
                        distance_gated_identity_switches += 1
                    last_distance_gated_predicted_id_for_target[batch_index][target_id] = (
                        predicted_id
                    )
        for name, value in current.items():
            add(name, value)
        if prior_belief is not None:
            correction_valid = matched & prior_belief.objects.active
            prior_current_error = torch.linalg.vector_norm(
                prior_belief.objects.position - aligned_position,
                dim=-1,
            )
            posterior_current_error = torch.linalg.vector_norm(
                belief.objects.position - aligned_position,
                dim=-1,
            )
            add(
                "correction_current",
                posterior_improvement_hinge(
                    posterior_current_error,
                    prior_current_error,
                    correction_valid,
                ),
            )
            if correction_valid.any():
                current_correction_improvements.append(
                    float(
                        (prior_current_error - posterior_current_error)
                        .masked_select(correction_valid)
                        .mean()
                        .detach()
                        .cpu()
                    )
                )
        rollout_result: _RolloutLossResult | None = None
        if score_rollout:
            rollout_result = _rollout_loss_result(
                model,
                belief,
                batch,
                config,
                frame_index,
                indices,
                matched,
            )
            for name, value in rollout_result.losses.items():
                add(name, value)
            _accumulate_float_metrics(
                physical_metrics,
                rollout_result.physical_metrics,
            )

        if (
            prior_rollout is not None
            and rollout_result is not None
            and rollout_result.positions is not None
            and frame_offsets
        ):
            deltas: list[Tensor] = []
            correction_losses: list[Tensor] = []
            for query_index, frame_offset in enumerate(frame_offsets):
                target_index = frame_index + frame_offset
                target_position = gather_target_slots(
                    batch["objects"]["position"][:, target_index], indices
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
                if valid.any():
                    prior_error = torch.linalg.vector_norm(
                        prior_rollout.positions[:, query_index] - target_position,
                        dim=-1,
                    )
                    posterior_error = torch.linalg.vector_norm(
                        rollout_result.positions[:, query_index] - target_position,
                        dim=-1,
                    )
                    deltas.append((prior_error - posterior_error).masked_select(valid).mean())
                    correction_losses.append(
                        posterior_improvement_hinge(
                            posterior_error,
                            prior_error,
                            valid,
                        )
                    )
                    add(
                        rollout_horizon_loss_key(
                            "correction_future",
                            query_seconds[query_index],
                        ),
                        correction_losses[-1],
                    )
            if deltas:
                future_correction_improvements.append(
                    float(torch.stack(deltas).mean().detach().cpu())
                )
                add("correction_future", torch.stack(correction_losses).mean())

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
            add(
                "measurement",
                _weighted_measurement_total(
                    supervised,
                    config.training.measurement_loss_weights,
                ),
            )
            for name, value in supervised.items():
                add(name, value)

        if (
            frame_index - window_start + 1
        ) % config.training.tbptt_steps == 0 and frame_index + 1 < window_stop:
            model.detach_state()

    reference = rgb
    details = {name: _mean_losses(values, reference) for name, values in detail_lists.items()}
    details = _globally_weight_horizon_details(details, config, reference)
    terms = _group_closed_loop_terms(details, reference)
    total = _weighted_closed_loop_total(terms, config.training.loss_weights)
    metrics = {name: float(value.detach().cpu()) for name, value in details.items()}
    metrics.update(physical_metrics)
    metrics.update(
        {
            "rollout_anchor_count": float(len(rollout_anchor_frames)),
            "rollout_anchor_candidate_count": float(
                sum(
                    bool(_valid_rollout_offsets(config, frame_index, total_frames)[0])
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
        else 0.0
    )
    metrics["physical_state_velocity_rmse_mps"] = (
        math.sqrt(metrics["physical_state_velocity_sse"] / state_velocity_count)
        if state_velocity_count
        else 0.0
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
            metrics[f"physical_rollout_position_rmse_m{horizon_suffix}"] = 0.0
        if velocity_count:
            metrics[f"physical_rollout_velocity_rmse_mps{horizon_suffix}"] = math.sqrt(
                metrics[f"physical_rollout_velocity{horizon_suffix}_sse"] / velocity_count
            )
        else:
            metrics[f"physical_rollout_velocity_rmse_mps{horizon_suffix}"] = 0.0
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
            "perturbed_updates": float(perturbed_updates),
            "fast_path_supervised": float(fast_supervised_frames > 0),
            "fast_supervised_frames": float(fast_supervised_frames),
            "current_correction_improvement_m": (
                float(sum(current_correction_improvements) / len(current_correction_improvements))
                if current_correction_improvements
                else math.nan
            ),
            "future_correction_improvement_m": (
                float(sum(future_correction_improvements) / len(future_correction_improvements))
                if future_correction_improvements
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
    )


__all__ = [
    "TrainingBatchResult",
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
