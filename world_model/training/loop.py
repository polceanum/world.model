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
from dataclasses import dataclass
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


@dataclass
class TrainingBatchResult:
    """Losses and detached diagnostics for one optimiser step."""

    total_loss: Tensor
    loss_terms: dict[str, Tensor]
    metrics: dict[str, float]
    phase: str


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
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    objects = belief.objects
    target_objects = batch["objects"]
    target_position = target_objects["position"][:, frame_index]
    target_active = target_objects["active"][:, frame_index].bool()
    indices, matched = match_belief_to_targets(belief, target_position, target_active)
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


def _rollout_losses(
    model: OnlineWorldModel,
    belief: WorldBelief,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    frame_index: int,
    indices: Tensor,
    matched: Tensor,
) -> dict[str, Tensor]:
    total_frames = int(batch["rgb"].shape[1])
    frame_offsets, query_seconds, horizon_weights = _valid_rollout_offsets(
        config,
        frame_index,
        total_frames,
    )
    reference = belief.objects.position
    if not frame_offsets:
        return {
            "rollout_position": reference.sum() * 0,
            "rollout_velocity": reference.sum() * 0,
            "event_collision": reference.sum() * 0,
        }
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

    def weighted_mean(losses: list[Tensor], weights: list[float]) -> Tensor:
        if not losses:
            return reference.sum() * 0
        weight = reference.new_tensor(weights)
        return (torch.stack(losses) * weight).sum() / weight.sum().clamp_min(1.0e-8)

    return {
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
    }


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
    maximum_start = total_frames - window_steps
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
            model.ingest(make_rgb_packet(batch, frame_index))
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
) -> TrainingBatchResult:
    """Run one causal RGB-only sequence window through the real runtime.

    The belief is never reset to simulator state.  Labels are read only after
    each RGB ingest to compute supervised losses.
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

    model.reset(batch_size=batch_size)
    _burn_in_causal_prefix(model, batch, window_start)
    detail_lists: dict[str, list[Tensor]] = {}
    current_correction_improvements: list[float] = []
    future_correction_improvements: list[float] = []
    perturbed_updates = 0
    matched_count = 0
    fast_supervised_frames = 0

    def add(name: str, value: Tensor) -> None:
        detail_lists.setdefault(name, []).append(value)

    for frame_index in range(window_start, window_stop):
        packet = make_rgb_packet(batch, frame_index)
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
            frame_offsets, query_seconds, _ = _valid_rollout_offsets(
                config,
                frame_index,
                total_frames,
            )
            if frame_offsets:
                prior_rollout = model.dynamics.rollout(prior, query_seconds, return_events=False)
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
                    belief_target_indices, belief_matched_slots = match_belief_to_targets(
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
        current, indices, matched = _belief_state_losses(belief, batch, frame_index)
        matched_count += int(matched.sum().detach().cpu())
        for name, value in current.items():
            add(name, value)
        if prior_belief is not None:
            aligned_position = gather_target_slots(
                batch["objects"]["position"][:, frame_index],
                indices,
            )
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
        rollout = _rollout_losses(
            model,
            belief,
            batch,
            config,
            frame_index,
            indices,
            matched,
        )
        for name, value in rollout.items():
            add(name, value)

        if prior_rollout is not None and frame_offsets:
            posterior_rollout = model.dynamics.rollout(belief, query_seconds, return_events=False)
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
                        posterior_rollout.positions[:, query_index] - target_position,
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
    "pretrain_rgb_measurements",
    "run_closed_loop_batch",
    "select_closed_loop_window",
    "supervised_measurement_losses",
    "supervised_slot_measurement_losses",
]
