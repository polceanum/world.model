"""Concrete public-episode objective adapter for specification 1.61.

The adapter has two deliberately separate paths:

* RGB-D tensors and calibrated camera transforms enter the set proposer.  A
  detached Hungarian assignment is used only to arrange training labels in
  proposal order.
* Simulator state creates an oracle-state dynamics anchor and future target.
  It is never placed in an observation packet or an ``OnlineWorldModel``
  runtime belief.  This is ordinary supervised dynamics training, not an
  inference shortcut.

Planning candidates, oracle winners, regret, and task success are absent from
this module.  They remain downstream qualification evidence only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from world_model.belief import MotionMode, WorldBelief
from world_model.dynamics.actions import WorldImpulseAction
from world_model.observations import ObservationPacket
from world_model.observations.measurements import MeasurementSet
from world_model.training.dynamic_set_objectives import (
    DynamicsLossInputs,
    PerceptionLossInputs,
)
from world_model.training.dynamic_set_scene import (
    DYNAMIC_SET_FRAMES,
    DYNAMIC_SET_IMAGE_SIZE,
    DYNAMIC_SET_MAX_OBJECTS,
)
from world_model.training.dynamic_set_trainer import (
    DynamicSetCausalSupport,
    DynamicSetObjectiveInputs,
    DynamicSetTrainingMicrobatch,
    dynamic_set_perception_frame_index,
)

_SCHEMA = "dynamic_set_episode_objective_adapter_v2"
_PROPOSAL_COUNT = 8
_APPEARANCE_DIM = 8
_STATE_DIM = 6
_DYNAMICS_TRAJECTORY_STEPS = 7
_OBSERVATION_VARIANCE = 1.0e-6
_IMPULSE_MULTIPLIER_BOUND = 0.25
_IMPULSE_ADDITIVE_BOUND = 0.10


def _require_tensor(
    mapping: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    value = mapping.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"episode field {name!r} must be a tensor")
    if shape is not None and value.shape != shape:
        raise ValueError(f"episode field {name!r} has shape {tuple(value.shape)}, expected {shape}")
    if dtype is not None and value.dtype is not dtype:
        raise TypeError(f"episode field {name!r} must use {dtype}")
    if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
        raise ValueError(f"episode field {name!r} contains NaN or Inf")
    return value


def _nested_mapping(episode: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = episode.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"episode {name!r} must be a mapping")
    return value


def _episode(
    materialization: object,
    *,
    perception_frame_index: int | None = None,
) -> Mapping[str, Any]:
    value = getattr(materialization, "episode", None)
    if not isinstance(value, Mapping):
        raise TypeError("materialization must expose an episode mapping")
    _require_tensor(value, "timestamps", shape=(DYNAMIC_SET_FRAMES,))
    metadata = value.get("metadata")
    is_lean = (
        isinstance(metadata, Mapping)
        and metadata.get("training_materialization_schema")
        == "dynamic_set_lean_training_episode_v1"
    )
    if is_lean:
        if perception_frame_index is None:
            raise ValueError("lean training episode requires its requested perception frame")
        if metadata.get("perception_frame_index") != perception_frame_index:
            raise ValueError("lean training episode carries a different perception frame")
        _require_tensor(value, "rgb", shape=(3, *DYNAMIC_SET_IMAGE_SIZE))
        _require_tensor(value, "depth", shape=(1, *DYNAMIC_SET_IMAGE_SIZE))
        labels = _nested_mapping(value, "labels")
        _require_tensor(
            labels,
            "segmentation_mask",
            shape=(DYNAMIC_SET_MAX_OBJECTS, *DYNAMIC_SET_IMAGE_SIZE),
            dtype=torch.bool,
        )
    else:
        _require_tensor(value, "rgb", shape=(DYNAMIC_SET_FRAMES, 3, *DYNAMIC_SET_IMAGE_SIZE))
        _require_tensor(value, "depth", shape=(DYNAMIC_SET_FRAMES, 1, *DYNAMIC_SET_IMAGE_SIZE))
    return value


def _is_lean_episode(episode: Mapping[str, Any]) -> bool:
    metadata = episode.get("metadata")
    return (
        isinstance(metadata, Mapping)
        and metadata.get("training_materialization_schema")
        == "dynamic_set_lean_training_episode_v1"
    )


def _perception_tensor(
    episode: Mapping[str, Any],
    name: str,
    frame_index: int,
) -> Tensor:
    value = _require_tensor(episode, name)
    if _is_lean_episode(episode):
        metadata = _nested_mapping(episode, "metadata")
        if metadata.get("perception_frame_index") != frame_index:
            raise ValueError("lean episode perception frame differs from the requested frame")
        return value
    return value[frame_index]


def _segmentation_mask(episode: Mapping[str, Any], frame_index: int) -> Tensor:
    labels = _nested_mapping(episode, "labels")
    value = _require_tensor(labels, "segmentation_mask")
    if _is_lean_episode(episode):
        metadata = _nested_mapping(episode, "metadata")
        if metadata.get("perception_frame_index") != frame_index:
            raise ValueError("lean episode label frame differs from the requested frame")
        return value
    return value[frame_index]


def _model_components(model: nn.Module) -> tuple[nn.Module, nn.Module, object]:
    try:
        observation = model.observation_modules["rgbd"]  # type: ignore[attr-defined]
        dynamics = model.dynamics  # type: ignore[attr-defined]
        factory = model.belief_factory  # type: ignore[attr-defined]
    except (AttributeError, KeyError) as error:
        raise TypeError(
            "dynamic-set adapter requires an OnlineWorldModel with RGB-D and dynamics"
        ) from error
    if not isinstance(observation, nn.Module) or not isinstance(dynamics, nn.Module):
        raise TypeError("online RGB-D and dynamics owners must be torch modules")
    if getattr(observation, "set_proposer", None) is None:
        raise ValueError("dynamic-set adapter requires explicit set-observation mode")
    if not callable(getattr(factory, "create", None)):
        raise TypeError("online model belief_factory must expose create()")
    return observation, dynamics, factory


def _perception_frame(microbatch: DynamicSetTrainingMicrobatch) -> int:
    # One complete frame per episode keeps the CPU campaign bounded.  The
    # absolute update/microbatch clock visits all 56 frames without adapter
    # mutable state, so checkpoint continuation is exact.
    return dynamic_set_perception_frame_index(
        microbatch.update_index,
        microbatch.microbatch_index,
    )


def _batched_measurement(
    observation: nn.Module,
    episodes: Sequence[Mapping[str, Any]],
    frame_index: int,
) -> MeasurementSet:
    rgb = torch.stack([_perception_tensor(episode, "rgb", frame_index) for episode in episodes])
    depth = torch.stack([_perception_tensor(episode, "depth", frame_index) for episode in episodes])
    world_from_camera = torch.stack(
        [
            _nested_mapping(episode, "camera")["world_from_camera"][frame_index]
            for episode in episodes
        ]
    )
    intrinsics = torch.stack(
        [_nested_mapping(episode, "camera")["intrinsics"][frame_index] for episode in episodes]
    )
    timestamps = torch.stack([episode["timestamps"][frame_index] for episode in episodes])
    if not bool(timestamps.eq(timestamps[:1]).all()):
        raise ValueError("batched perception frames must share one timestamp")
    packet = ObservationPacket(
        modality="rgbd",
        sensor_id="dynamic-set:camera0:rgbd",
        timestamp=float(timestamps[0]),
        payload={"rgb": rgb, "depth": depth},
        calibration={
            "world_from_camera": world_from_camera,
            "intrinsics": intrinsics,
        },
        frame_id=f"dynamic-set:{frame_index}",
        metadata={"image_size": DYNAMIC_SET_IMAGE_SIZE},
    )
    initialise = getattr(observation, "initialise_measurements", None)
    if not callable(initialise):
        raise TypeError("RGB-D observation module lacks initialise_measurements()")
    measurement = initialise([packet], context=object())
    if not isinstance(measurement, MeasurementSet):
        raise TypeError("RGB-D observation did not return a MeasurementSet")
    measurement.validate()
    return measurement


def _observable_appearance(image: Tensor, masks: Tensor) -> Tensor:
    """Return the same observable eight-value colour statistic as the model."""

    if image.shape != (3, *DYNAMIC_SET_IMAGE_SIZE):
        raise ValueError("appearance image must have shape [3,64,64]")
    if masks.shape != (DYNAMIC_SET_MAX_OBJECTS, *DYNAMIC_SET_IMAGE_SIZE):
        raise ValueError("appearance masks must have shape [6,64,64]")
    weights = masks.to(image.dtype)
    epsilon = torch.finfo(image.dtype).eps
    mass = weights.sum(dim=(-2, -1)).clamp_min(epsilon)
    mean_rgb = torch.einsum("nhw,chw->nc", weights, image) / mass.unsqueeze(-1)
    second_rgb = torch.einsum("nhw,chw->nc", weights, image.square()) / mass.unsqueeze(-1)
    std_rgb = (second_rgb - mean_rgb.square() + epsilon).clamp_min(epsilon).sqrt()
    intensity = image.mean(dim=0)
    mean_intensity = torch.einsum("nhw,hw->n", weights, intensity) / mass
    second_intensity = torch.einsum("nhw,hw->n", weights, intensity.square()) / mass
    std_intensity = (second_intensity - mean_intensity.square() + epsilon).clamp_min(epsilon).sqrt()
    descriptor = torch.cat(
        (
            mean_rgb,
            std_rgb,
            mean_intensity.unsqueeze(-1),
            std_intensity.unsqueeze(-1),
        ),
        dim=-1,
    )
    return F.normalize(descriptor, dim=-1, eps=epsilon)


def _hungarian_target_indices(
    measurement: MeasurementSet,
    episodes: Sequence[Mapping[str, Any]],
    frame_index: int,
) -> Tensor:
    """Match active truth objects to all eight proposals on detached evidence."""

    full_mask_logits = measurement.auxiliary.get("set_full_mask_logits")
    if not isinstance(full_mask_logits, Tensor) or full_mask_logits.shape != (
        len(episodes),
        _PROPOSAL_COUNT + 1,
        *DYNAMIC_SET_IMAGE_SIZE,
    ):
        raise ValueError("set proposer did not expose [B,9,64,64] mask logits")
    # The probabilities are a transient matching value.  Reconstructing them
    # here keeps one full-resolution representation in the runtime rather than
    # retaining both logits and probabilities across observation steps.
    full_probability = full_mask_logits.detach().softmax(dim=1)
    result = torch.full(
        (len(episodes), _PROPOSAL_COUNT),
        -1,
        dtype=torch.int64,
        device=measurement.values.device,
    )
    for batch_index, episode in enumerate(episodes):
        objects = _nested_mapping(episode, "objects")
        active = _require_tensor(objects, "active")[frame_index]
        truth_position = _require_tensor(objects, "position")[frame_index]
        truth_masks = _segmentation_mask(episode, frame_index)
        valid_targets = torch.nonzero(active, as_tuple=False).flatten()
        if not valid_targets.numel():
            continue
        predicted = measurement.values[batch_index].detach().cpu()
        target = truth_position[valid_targets].detach().cpu()
        position_cost = torch.cdist(predicted, target, p=2) / 0.21
        predicted_masks = full_probability[batch_index, 1:].cpu()
        selected_masks = truth_masks[valid_targets].to(predicted_masks.dtype).detach().cpu()
        intersection = torch.einsum("phw,nhw->pn", predicted_masks, selected_masks)
        denominator = (
            predicted_masks.sum(dim=(-2, -1))[:, None] + selected_masks.sum(dim=(-2, -1))[None, :]
        )
        dice_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        confidence_cost = (
            -0.05 * measurement.existence_logits[batch_index].detach().sigmoid().cpu()[:, None]
        )
        cost = position_cost + 2.0 * dice_cost + confidence_cost
        proposal_rows, target_columns = linear_sum_assignment(np.asarray(cost))
        proposals = torch.as_tensor(proposal_rows, dtype=torch.int64, device=result.device)
        targets = valid_targets.to(result.device).index_select(
            0,
            torch.as_tensor(target_columns, dtype=torch.int64, device=result.device),
        )
        result[batch_index, proposals] = targets
    return result


def _perception_inputs(
    measurement: MeasurementSet,
    episodes: Sequence[Mapping[str, Any]],
    frame_index: int,
) -> PerceptionLossInputs:
    target_indices = _hungarian_target_indices(measurement, episodes, frame_index)
    batch = len(episodes)
    target_masks = measurement.values.new_zeros(
        batch,
        _PROPOSAL_COUNT + 1,
        *DYNAMIC_SET_IMAGE_SIZE,
    )
    target_exists = torch.zeros(
        batch,
        _PROPOSAL_COUNT,
        dtype=torch.bool,
        device=measurement.values.device,
    )
    target_position = torch.zeros_like(measurement.values)
    if measurement.appearance is None or measurement.appearance.shape != (
        batch,
        _PROPOSAL_COUNT,
        _APPEARANCE_DIM,
    ):
        raise ValueError("set measurement must expose [B,8,8] appearance")
    target_appearance = torch.zeros_like(measurement.appearance)
    for batch_index, episode in enumerate(episodes):
        objects = _nested_mapping(episode, "objects")
        truth_masks = _segmentation_mask(episode, frame_index).to(measurement.values.device)
        truth_position = _require_tensor(objects, "position")[frame_index].to(
            measurement.values.device
        )
        image = _perception_tensor(episode, "rgb", frame_index).to(measurement.values.device)
        appearance = _observable_appearance(image, truth_masks)
        occupied = truth_masks.any(dim=0)
        target_masks[batch_index, 0] = (~occupied).to(target_masks.dtype)
        for proposal in range(_PROPOSAL_COUNT):
            target = int(target_indices[batch_index, proposal])
            if target < 0:
                continue
            target_exists[batch_index, proposal] = True
            target_masks[batch_index, proposal + 1] = truth_masks[target].to(target_masks.dtype)
            target_position[batch_index, proposal] = truth_position[target]
            target_appearance[batch_index, proposal] = appearance[target]
    mask_logits = measurement.auxiliary.get("set_full_mask_logits")
    position_log_variance = measurement.auxiliary.get("world_position_log_variance")
    if not isinstance(mask_logits, Tensor) or mask_logits.shape != target_masks.shape:
        raise ValueError("set measurement must expose aligned full-mask logits")
    if (
        not isinstance(position_log_variance, Tensor)
        or position_log_variance.shape != measurement.values.shape
    ):
        raise ValueError("set measurement must expose position log variance")
    return PerceptionLossInputs(
        mask_logits=mask_logits,
        target_masks=target_masks,
        existence_logits=measurement.existence_logits,
        target_exists=target_exists,
        metric_position=measurement.values,
        target_position=target_position,
        position_log_variance=position_log_variance,
        appearance=measurement.appearance,
        target_appearance=target_appearance,
    )


def _stable_interval(episode: Mapping[str, Any], anchor: int, target: int) -> bool:
    objects = _nested_mapping(episode, "objects")
    active = _require_tensor(objects, "active")
    object_id = _require_tensor(objects, "id")
    if not (0 <= anchor < target < DYNAMIC_SET_FRAMES):
        return False
    anchor_active = active[anchor]
    if not bool(anchor_active.any()):
        return False
    return bool(
        active[anchor : target + 1].eq(anchor_active).all()
        and object_id[anchor : target + 1].eq(object_id[anchor]).all()
    )


def _event_frame(events: Mapping[str, Any], name: str) -> int | None:
    value = _require_tensor(events, name)
    if value.ndim < 1 or value.shape[0] != DYNAMIC_SET_FRAMES:
        raise ValueError(f"event {name!r} must begin with the 56-frame axis")
    collapsed = value.reshape(DYNAMIC_SET_FRAMES, -1).any(dim=-1)
    indices = torch.nonzero(collapsed, as_tuple=False).flatten()
    return None if not indices.numel() else int(indices[0])


def _action_application_time(episode: Mapping[str, Any], event_frame: int) -> Tensor:
    """Return the public absolute time at which the impulse entered physics."""

    events = _nested_mapping(episode, "events")
    explicit = events.get("known_action_timestamp")
    if isinstance(explicit, Tensor):
        if explicit.shape == (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS):
            observed = _require_tensor(events, "known_action_observed")
            values = explicit[event_frame][observed[event_frame]]
        elif explicit.shape == (DYNAMIC_SET_FRAMES,):
            values = explicit[event_frame : event_frame + 1]
        else:
            raise ValueError("known_action_timestamp has an unsupported shape")
        if values.numel() != 1 or not bool(torch.isfinite(values).all()):
            raise ValueError("public known action must expose one finite absolute timestamp")
        return values.reshape(())
    # Canonical episode event tensors describe the interval ending at their
    # frame.  The current materializer injects the impulse at that interval's
    # start.  This fallback is retained for old frozen materializations; new
    # ones should carry the explicit public timestamp above.
    if event_frame <= 0:
        raise ValueError("a known impulse cannot precede the first observation")
    return _require_tensor(episode, "timestamps")[event_frame - 1]


def _select_dynamics_window(
    episode: Mapping[str, Any],
    *,
    known_action: bool,
    contact: bool,
    selector: int,
) -> tuple[int, int, int | None]:
    events = _nested_mapping(episode, "events")
    action_event = _event_frame(events, "known_action_observed")
    collision_event = _event_frame(events, "pair_collision")
    if known_action != (action_event is not None):
        raise ValueError("manifest known-action flag disagrees with public episode")
    if contact and collision_event is None:
        raise ValueError("contact-cell episode has no certified pair collision")
    if not contact and collision_event is not None:
        raise ValueError("contact-free episode contains a pair collision")

    if action_event is not None:
        action_time = _action_application_time(episode, action_event)
        timestamps = _require_tensor(episode, "timestamps")
        earlier = torch.nonzero(timestamps < action_time, as_tuple=False).flatten()
        if not earlier.numel():
            raise ValueError("known action lacks a strictly earlier dynamics anchor")
        anchor = int(earlier[-1])
        target = anchor + _DYNAMICS_TRAJECTORY_STEPS
        if target >= DYNAMIC_SET_FRAMES:
            raise ValueError("known-action trajectory exceeds the episode")
        if _stable_interval(episode, anchor, target):
            return anchor, target, action_event
        raise ValueError("known-action supervision crosses a lifecycle transition")

    if collision_event is not None:
        anchor = collision_event - 3
        target = anchor + _DYNAMICS_TRAJECTORY_STEPS
        if _stable_interval(episode, anchor, target):
            return anchor, target, None
        raise ValueError("contact supervision crosses a lifecycle transition")

    available_anchors = DYNAMIC_SET_FRAMES - _DYNAMICS_TRAJECTORY_STEPS
    desired_anchor = selector % available_anchors
    for offset in range(available_anchors):
        anchor = (desired_anchor + offset) % available_anchors
        target = anchor + _DYNAMICS_TRAJECTORY_STEPS
        if _stable_interval(episode, anchor, target):
            return anchor, target, None
    raise ValueError("episode contains no stable seven-step dynamics interval")


def _truth_belief_batch(
    model: nn.Module,
    episodes: Sequence[Mapping[str, Any]],
    frame_indices: Sequence[int],
) -> WorldBelief:
    if not episodes or len(episodes) != len(frame_indices):
        raise ValueError("truth-belief batch requires matching nonempty episodes and frames")
    _, _, factory = _model_components(model)
    object_rows = tuple(_nested_mapping(episode, "objects") for episode in episodes)
    camera_rows = tuple(_nested_mapping(episode, "camera") for episode in episodes)

    def stack_episode(name: str) -> Tensor:
        return torch.stack(
            [
                _require_tensor(objects, name)[frame_index]
                for objects, frame_index in zip(object_rows, frame_indices, strict=True)
            ]
        )

    timestamp = torch.stack(
        [
            _require_tensor(episode, "timestamps")[frame_index]
            for episode, frame_index in zip(episodes, frame_indices, strict=True)
        ]
    )
    world_from_camera = torch.stack(
        [
            _require_tensor(camera, "world_from_camera")[frame_index]
            for camera, frame_index in zip(camera_rows, frame_indices, strict=True)
        ]
    )
    intrinsics = torch.stack(
        [
            _require_tensor(camera, "intrinsics")[frame_index]
            for camera, frame_index in zip(camera_rows, frame_indices, strict=True)
        ]
    )
    belief = factory.create(  # type: ignore[attr-defined]
        batch_size=len(episodes),
        timestamp=timestamp,
        device=timestamp.device,
        dtype=timestamp.dtype,
        gravity=(0.0, 0.0, 0.0),
        world_from_camera=world_from_camera,
        intrinsics=intrinsics,
        active_modalities=("rgbd",),
    )
    active = stack_episode("active")
    object_id = stack_episode("id")
    position = stack_episode("position")
    velocity = stack_episode("velocity")
    orientation = stack_episode("orientation")
    angular_velocity = stack_episode("angular_velocity")
    radius = stack_episode("radius")
    mass = stack_episode("mass")
    restitution = stack_episode("restitution")
    drag = stack_episode("drag")
    friction = stack_episode("friction")
    albedo = stack_episode("albedo")
    appearance = torch.zeros_like(belief.objects.appearance)
    appearance[..., : min(3, appearance.shape[-1])] = albedo[..., : min(3, appearance.shape[-1])]
    motion_logits = torch.zeros_like(belief.objects.motion_mode_logits)
    motion_logits[..., MotionMode.FREE] = 4.0
    log_variance = torch.full_like(
        belief.objects.fast_log_variance,
        math.log(_OBSERVATION_VARIANCE),
    )
    truth_objects = belief.objects.replace(
        object_id=object_id,
        active=active,
        existence_logit=torch.where(
            active,
            position.new_full(active.shape, 12.0),
            position.new_full(active.shape, -12.0),
        ),
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
        geometry=radius,
        appearance=appearance,
        log_mass=mass.clamp_min(1.0e-8).log(),
        restitution_logit=torch.logit(restitution.clamp(1.0e-6, 1.0 - 1.0e-6)),
        log_drag=drag.clamp_min(1.0e-8).log(),
        friction_logit=torch.logit(friction.clamp(1.0e-6, 1.0 - 1.0e-6)),
        motion_mode_logits=motion_logits,
        visibility_logit=torch.where(
            active,
            position.new_full(active.shape, 12.0),
            position.new_full(active.shape, -12.0),
        ),
        fast_log_variance=log_variance,
    )
    next_id = torch.where(active, object_id, object_id.new_full((), -1)).amax(dim=-1) + 1
    return belief.replace(
        objects=truth_objects,
        next_object_id=next_id.clamp_min(0),
        metadata={
            "initialised": True,
            "supervision_anchor": True,
            "simulator_truth_not_observed": True,
        },
    ).validate()


def _truth_belief(
    model: nn.Module,
    episode: Mapping[str, Any],
    frame_index: int,
) -> WorldBelief:
    """Retain the B1 construction as a parity oracle for focused tests."""

    return _truth_belief_batch(model, (episode,), (frame_index,))


def _known_action(
    episode: Mapping[str, Any],
    action_event: int,
    belief: WorldBelief,
) -> WorldImpulseAction:
    events = _nested_mapping(episode, "events")
    observed = _require_tensor(events, "known_action_observed")
    target_slots = torch.nonzero(observed[action_event], as_tuple=False).flatten()
    if target_slots.numel() != 1:
        raise ValueError("known-action episode must declare exactly one target")
    slot = int(target_slots[0])
    action_object_id = _require_tensor(events, "known_action_object_id")[action_event, slot]
    impulse = _require_tensor(events, "known_impulse_world")[action_event, slot]
    action = WorldImpulseAction(
        timestamp=_action_application_time(episode, action_event).reshape(1).to(belief.timestamp),
        object_id=action_object_id.reshape(1).to(device=belief.device, dtype=torch.int64),
        impulse_world=impulse.reshape(1, 3).to(belief.objects.position),
    )
    action.validate_for(belief)
    return action


def _known_action_batch(
    episodes: Sequence[Mapping[str, Any]],
    action_events: Sequence[int],
    belief: WorldBelief,
) -> WorldImpulseAction:
    if not episodes or len(episodes) != len(action_events) or len(episodes) != belief.batch_size:
        raise ValueError("known-action batch requires one event for every belief row")
    timestamps: list[Tensor] = []
    object_ids: list[Tensor] = []
    impulses: list[Tensor] = []
    for episode, action_event in zip(episodes, action_events, strict=True):
        events = _nested_mapping(episode, "events")
        observed = _require_tensor(events, "known_action_observed")
        target_slots = torch.nonzero(observed[action_event], as_tuple=False).flatten()
        if target_slots.numel() != 1:
            raise ValueError("known-action episode must declare exactly one target")
        slot = int(target_slots[0])
        timestamps.append(_action_application_time(episode, action_event))
        object_ids.append(_require_tensor(events, "known_action_object_id")[action_event, slot])
        impulses.append(_require_tensor(events, "known_impulse_world")[action_event, slot])
    action = WorldImpulseAction(
        timestamp=torch.stack(timestamps).to(belief.timestamp),
        object_id=torch.stack(object_ids).to(device=belief.device, dtype=torch.int64),
        impulse_world=torch.stack(impulses).to(belief.objects.position),
    )
    action.validate_for(belief)
    return action


def _dynamics_group(
    model: nn.Module,
    episodes: Sequence[Mapping[str, Any]],
    windows: Sequence[tuple[int, int, int | None]],
) -> tuple[Tensor, ...]:
    """Evaluate one homogeneous action/no-action group over seven recursive steps."""

    if not episodes or len(episodes) != len(windows):
        raise ValueError("dynamics group requires matching nonempty episodes and windows")
    action_flags = tuple(window[2] is not None for window in windows)
    if len(set(action_flags)) != 1:
        raise ValueError("dynamics group must be homogeneous in known-action presence")
    _, dynamics, _ = _model_components(model)
    anchors = tuple(window[0] for window in windows)
    targets = tuple(window[1] for window in windows)
    if any(
        target - anchor != _DYNAMICS_TRAJECTORY_STEPS
        for anchor, target in zip(anchors, targets, strict=True)
    ):
        raise ValueError("dynamics windows must contain exactly seven prediction steps")
    belief = _truth_belief_batch(model, episodes, anchors)
    anchor_active = belief.objects.active
    anchor_object_id = belief.objects.object_id
    action = None
    if action_flags[0]:
        action = _known_action_batch(
            episodes,
            tuple(int(window[2]) for window in windows),
            belief,
        )
        first_targets = torch.stack(
            [
                _require_tensor(episode, "timestamps")[anchor + 1]
                for episode, anchor in zip(episodes, anchors, strict=True)
            ]
        ).to(action.timestamp)
        if not torch.equal(action.timestamp, first_targets):
            raise ValueError("known action must occur exactly in the first trajectory interval")
    predict_interval = getattr(dynamics, "predict_state_only_step", None)
    if not callable(predict_interval):
        raise TypeError("dynamic-set dynamics must expose predict_state_only_step()")
    predicted_steps: list[Tensor] = []
    target_steps: list[Tensor] = []
    process_steps: list[Tensor] = []
    collision_steps: list[Tensor] = []
    residual_steps: list[Tensor] = []
    current = belief
    for step_index in range(_DYNAMICS_TRAJECTORY_STEPS):
        target_frames = tuple(anchor + step_index + 1 for anchor in anchors)
        interaction = dynamics.interactions(  # type: ignore[attr-defined]
            current.objects,
            current.global_code,
        )
        bounded_pair_residual = torch.stack(
            (
                _IMPULSE_MULTIPLIER_BOUND * torch.tanh(interaction.impulse_multiplier_raw),
                _IMPULSE_ADDITIVE_BOUND * torch.tanh(interaction.impulse_additive_raw),
            ),
            dim=-1,
        )
        residual_steps.append(
            bounded_pair_residual.reshape(
                len(episodes),
                DYNAMIC_SET_MAX_OBJECTS,
                DYNAMIC_SET_MAX_OBJECTS * 2,
            )
        )
        elapsed = torch.stack(
            [
                _require_tensor(episode, "timestamps")[target_frame] - current.timestamp[index]
                for index, (episode, target_frame) in enumerate(
                    zip(episodes, target_frames, strict=True)
                )
            ]
        )
        step = predict_interval(
            current,
            elapsed,
            action=action if step_index == 0 else None,
        )
        current = step.belief
        predicted_steps.append(
            torch.cat((current.objects.position, current.objects.velocity), dim=-1)
        )
        target_steps.append(
            torch.stack(
                [
                    torch.cat(
                        (
                            _require_tensor(_nested_mapping(episode, "objects"), "position")[
                                target_frame
                            ],
                            _require_tensor(_nested_mapping(episode, "objects"), "velocity")[
                                target_frame
                            ],
                        ),
                        dim=-1,
                    )
                    for episode, target_frame in zip(
                        episodes,
                        target_frames,
                        strict=True,
                    )
                ]
            )
        )
        process_steps.append(current.objects.fast_log_variance[..., :_STATE_DIM])
        pair_event_logits = step.auxiliary.get("pair_event_logits")
        expected_pair_shape = (
            len(episodes),
            DYNAMIC_SET_MAX_OBJECTS,
            DYNAMIC_SET_MAX_OBJECTS,
            2,
        )
        if (
            not isinstance(pair_event_logits, Tensor)
            or pair_event_logits.shape != expected_pair_shape
        ):
            raise ValueError("dynamics did not expose pair collision logits")
        collision_steps.append(pair_event_logits[..., 1])

    predicted_state = torch.stack(predicted_steps, dim=1)
    target_state = torch.stack(target_steps, dim=1)
    process_log_variance = torch.stack(process_steps, dim=1)
    collision_logits = torch.stack(collision_steps, dim=1)
    relation_residual = torch.stack(residual_steps, dim=1)
    return (
        predicted_state,
        target_state,
        process_log_variance,
        collision_logits,
        relation_residual,
        anchor_active,
        anchor_object_id,
    )


def _dynamics_row(
    model: nn.Module,
    episode: Mapping[str, Any],
    *,
    anchor: int,
    target: int,
    action_event: int | None,
) -> tuple[Tensor, ...]:
    return _dynamics_group(model, (episode,), ((anchor, target, action_event),))


def _dynamics_inputs(
    model: nn.Module,
    microbatch: DynamicSetTrainingMicrobatch,
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[DynamicsLossInputs, DynamicSetCausalSupport]:
    windows: list[tuple[int, int, int | None]] = []
    rows: list[tuple[Tensor, ...] | None] = [None] * len(episodes)
    for batch_index, (manifest_row, episode) in enumerate(
        zip(microbatch.rows, episodes, strict=True)
    ):
        window = _select_dynamics_window(
            episode,
            known_action=manifest_row.known_action,
            contact=manifest_row.contact,
            selector=(microbatch.update_index * 24 + microbatch.microbatch_index * 4 + batch_index),
        )
        windows.append(window)
    # Keep contact and contact-free rows separate.  The certified event-driven
    # path deliberately falls the whole batch back to fixed microsteps when
    # any row exhausts its event proof.  Mixing those regimes would therefore
    # change an otherwise fast row's numerical path relative to the B1 oracle.
    for action_present, contact_present in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        group_indices = tuple(
            index
            for index, (manifest_row, window) in enumerate(
                zip(microbatch.rows, windows, strict=True)
            )
            if (window[2] is not None) is action_present and manifest_row.contact is contact_present
        )
        if not group_indices:
            continue
        group = _dynamics_group(
            model,
            tuple(episodes[index] for index in group_indices),
            tuple(windows[index] for index in group_indices),
        )
        for local_index, batch_index in enumerate(group_indices):
            rows[batch_index] = tuple(value[local_index : local_index + 1] for value in group)
    if any(row is None for row in rows):
        raise RuntimeError("batched dynamics did not produce every microbatch row")
    complete_rows = tuple(row for row in rows if row is not None)

    predicted_state = torch.cat([row[0] for row in complete_rows], dim=0)
    target_state = torch.cat([row[1] for row in complete_rows], dim=0)
    process_log_variance = torch.cat([row[2] for row in complete_rows], dim=0)
    collision_logits = torch.cat([row[3] for row in complete_rows], dim=0)
    relation_residual = torch.cat([row[4] for row in complete_rows], dim=0)
    contact_window_mask = torch.zeros(
        len(complete_rows),
        _DYNAMICS_TRAJECTORY_STEPS,
        DYNAMIC_SET_MAX_OBJECTS,
        dtype=torch.bool,
    )
    collision_target = torch.zeros(
        len(complete_rows),
        _DYNAMICS_TRAJECTORY_STEPS,
        DYNAMIC_SET_MAX_OBJECTS,
        DYNAMIC_SET_MAX_OBJECTS,
        dtype=torch.bool,
    )
    collision_support = torch.zeros_like(collision_target)
    known_action_predictable_mask = torch.zeros_like(contact_window_mask)
    unaffected_object_mask = torch.zeros_like(contact_window_mask)
    anchor_indices: list[int] = []
    target_indices: list[tuple[int, ...]] = []
    external_rows: list[Tensor] = []
    known_rows: list[Tensor] = []
    identity = torch.eye(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.bool)

    for batch_index, (_manifest_row, episode, window, output) in enumerate(
        zip(microbatch.rows, episodes, windows, complete_rows, strict=True)
    ):
        anchor, target, action_event = window
        frame_indices = tuple(range(anchor + 1, target + 1))
        if len(frame_indices) != _DYNAMICS_TRAJECTORY_STEPS:
            raise RuntimeError("selected dynamics window lost its seven-step invariant")
        events = _nested_mapping(episode, "events")
        external = _require_tensor(events, "externally_actuated", dtype=torch.bool)
        known = _require_tensor(events, "known_action_observed", dtype=torch.bool)
        external_rows.append(external)
        known_rows.append(known)
        anchor_indices.append(anchor)
        target_indices.append(frame_indices)
        active = output[5].squeeze(0).detach().cpu()
        object_id = output[6].squeeze(0).detach().cpu()
        target_objects = _nested_mapping(episode, "objects")
        target_active = _require_tensor(target_objects, "active")[list(frame_indices)]
        target_id = _require_tensor(target_objects, "id")[list(frame_indices)]
        stable = active.unsqueeze(0) & target_active & object_id.unsqueeze(0).eq(target_id)
        pair_stable = stable[:, :, None] & stable[:, None, :] & ~identity.unsqueeze(0)
        collision_support[batch_index] = pair_stable
        target_collision = _require_tensor(events, "pair_collision", dtype=torch.bool)[
            list(frame_indices)
        ]
        collision_target[batch_index] = target_collision & pair_stable
        target_contact = _require_tensor(events, "pair_contact", dtype=torch.bool)[
            list(frame_indices)
        ]
        contact_objects = (target_collision | target_contact).any(dim=0).any(dim=-1)
        contact_window_mask[batch_index] = stable & contact_objects.unsqueeze(0)
        if action_event is not None:
            known_action_predictable_mask[batch_index] = stable
        affected = contact_objects | external[anchor + 1 : target + 1].any(dim=0)
        unaffected_object_mask[batch_index] = stable & ~affected.unsqueeze(0)

    causal_support = DynamicSetCausalSupport(
        externally_actuated=torch.stack(external_rows),
        known_action_observed=torch.stack(known_rows),
        anchor_frame_index=torch.tensor(anchor_indices, dtype=torch.int64),
        target_frame_index=torch.tensor(target_indices, dtype=torch.int64),
    )
    return (
        DynamicsLossInputs(
            predicted_state=predicted_state,
            target_state=target_state.to(predicted_state),
            process_log_variance=process_log_variance,
            contact_window_mask=contact_window_mask.to(predicted_state.device),
            collision_logits=collision_logits,
            collision_target=collision_target.to(collision_logits.device),
            collision_support=collision_support.to(collision_logits.device),
            known_action_predictable_mask=known_action_predictable_mask.to(predicted_state.device),
            relation_residual=relation_residual,
            unaffected_object_mask=unaffected_object_mask.to(relation_residual.device),
        ),
        causal_support,
    )


@dataclass
class DynamicSetEpisodeObjectiveAdapter:
    """Stateless deterministic adapter for accepted specification-1.61 episodes."""

    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("dynamic-set objective adapter schema differs")

    def build_objective_inputs(
        self,
        model: nn.Module,
        microbatch: DynamicSetTrainingMicrobatch,
    ) -> DynamicSetObjectiveInputs:
        if not isinstance(microbatch, DynamicSetTrainingMicrobatch):
            raise TypeError("microbatch must be DynamicSetTrainingMicrobatch")
        if len(microbatch.rows) != 4 or len(microbatch.materializations) != 4:
            raise ValueError("dynamic-set objective adapter requires exact B4")
        observation, _, _ = _model_components(model)
        frame_index = _perception_frame(microbatch)
        episodes = tuple(
            _episode(value, perception_frame_index=frame_index)
            for value in microbatch.materializations
        )
        measurement = _batched_measurement(observation, episodes, frame_index)
        perception = _perception_inputs(measurement, episodes, frame_index)
        dynamics, causal_support = _dynamics_inputs(model, microbatch, episodes)
        return DynamicSetObjectiveInputs(
            perception=perception,
            dynamics=dynamics,
            causal_support=causal_support,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"schema": self.schema}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {"schema"}:
            raise ValueError("dynamic-set objective adapter state schema differs")
        if state["schema"] != self.schema:
            raise ValueError("dynamic-set objective adapter checkpoint schema differs")


__all__ = ["DynamicSetEpisodeObjectiveAdapter"]
