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
from world_model.training.losses import gaussian_nll, masked_huber, weighted_total
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
    visibility_logits = measurements.auxiliary.get("visibility_logits")
    if visibility_logits is None:
        visibility_logits = measurements.auxiliary.get("visibility_logit")
    if visibility_logits is not None:
        outputs["visibility_logits"] = visibility_logits
    targets = {
        "values": aligned,
        "visibility": aligned_visibility,
    }
    masks = {
        "matched": matched,
        "existence": matched,
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
    measurement = torch.stack(tuple(details.values())).sum()
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
    trajectory = model.dynamics.rollout(belief, query_seconds, return_events=True)
    position_losses: list[Tensor] = []
    velocity_losses: list[Tensor] = []
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
        position_losses.append(
            masked_huber(
                trajectory.positions[:, query_index],
                target_position,
                valid,
            )
        )
        velocity_losses.append(
            masked_huber(
                trajectory.velocities[:, query_index],
                target_velocity,
                valid,
            )
        )
        if trajectory.event_logits is not None:
            event_target = (
                gather_target_slots(
                    batch["events"]["collision"][:, target_index].unsqueeze(-1),
                    indices,
                )
                .squeeze(-1)
                .to(reference.dtype)
            )
            event_scores = trajectory.event_logits[:, query_index, :, MotionMode.COLLISION]
            if valid.any():
                event_losses.append(
                    F.binary_cross_entropy_with_logits(event_scores[valid], event_target[valid])
                )
                event_weights.append(horizon_weights[query_index])

    def weighted_mean(losses: list[Tensor], weights: list[float]) -> Tensor:
        if not losses:
            return reference.sum() * 0
        weight = reference.new_tensor(weights)
        return (torch.stack(losses) * weight).sum() / weight.sum().clamp_min(1.0e-8)

    return {
        "rollout_position": weighted_mean(position_losses, horizon_weights),
        "rollout_velocity": weighted_mean(velocity_losses, horizon_weights),
        "event_collision": weighted_mean(event_losses, event_weights),
    }


def _group_closed_loop_terms(
    details: dict[str, Tensor],
    reference: Tensor,
) -> dict[str, Tensor]:
    def total(*names: str) -> Tensor:
        selected = [details[name] for name in names if name in details]
        return _mean_losses(selected, reference)

    return {
        "measurement": details.get("measurement", reference.sum() * 0),
        "state": total("state_position", "state_velocity"),
        "rollout": total("rollout_position", "rollout_velocity"),
        "event": details.get("event_collision", reference.sum() * 0),
        "parameter": total("parameter_drag", "parameter_restitution"),
        "existence": details.get("existence_belief", reference.sum() * 0),
        "uncertainty": details.get("uncertainty_position_nll", reference.sum() * 0),
        "correction": details.get("correction_magnitude", reference.sum() * 0),
    }


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
    detail_lists: dict[str, list[Tensor]] = {}
    correction_improvements: list[float] = []
    perturbed_updates = 0
    matched_count = 0
    fast_path_supervised = False

    def add(name: str, value: Tensor) -> None:
        detail_lists.setdefault(name, []).append(value)

    for frame_index in range(window_start, window_stop):
        packet = make_rgb_packet(batch, frame_index)
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
                    velocity_std=0.5 * config.training.perturbation_position_std,
                    covariance_log_bias=0.25,
                )
                # Preserve the previous timestamp.  ``ingest`` remains the
                # authoritative predictor, sees the real dt, and can therefore
                # infer velocity from consecutive RGB position measurements.
                model.state.belief = source_belief
                perturbed_updates += 1
            prior = model.dynamics.predict(source_belief, requested - source_belief.timestamp)
            frame_offsets, query_seconds, _ = _valid_rollout_offsets(
                config,
                frame_index,
                total_frames,
            )
            if frame_offsets:
                prior_rollout = model.dynamics.rollout(prior, query_seconds, return_events=False)
            if include_measurement_supervision and not fast_path_supervised:
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
                        None,
                    )
                    fast_supervised = supervised_measurement_losses(
                        module,
                        fast_measurements,
                        batch,
                        frame_index,
                    )
                    add(
                        "measurement",
                        torch.stack(tuple(fast_supervised.values())).sum(),
                    )
                    for name, value in fast_supervised.items():
                        add(f"fast_{name}", value)
                    fast_path_supervised = True

        belief = model.ingest(packet)
        current, indices, matched = _belief_state_losses(belief, batch, frame_index)
        matched_count += int(matched.sum().detach().cpu())
        for name, value in current.items():
            add(name, value)
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
                valid = matched & future_active
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
            if deltas:
                correction_improvements.append(float(torch.stack(deltas).mean().detach().cpu()))

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
            add("measurement", torch.stack(tuple(supervised.values())).sum())
            for name, value in supervised.items():
                add(name, value)

        if (
            frame_index - window_start + 1
        ) % config.training.tbptt_steps == 0 and frame_index + 1 < window_stop:
            model.detach_state()

    reference = rgb
    details = {name: _mean_losses(values, reference) for name, values in detail_lists.items()}
    terms = _group_closed_loop_terms(details, reference)
    total = weighted_total(terms, config.training.loss_weights)
    metrics = {name: float(value.detach().cpu()) for name, value in details.items()}
    metrics.update(
        {
            "matched_object_frames": float(matched_count),
            "perturbed_updates": float(perturbed_updates),
            "fast_path_supervised": float(fast_path_supervised),
            "correction_improvement_m": (
                float(sum(correction_improvements) / len(correction_improvements))
                if correction_improvements
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
    "supervised_measurement_losses",
]
