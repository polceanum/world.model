"""Episode generation and validation for the labelled synthetic sphere world."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from world_model.simulator.collisions import BOUNDARY_NAMES
from world_model.simulator.labels import (
    make_perception_labels,
    validate_perception_labels,
)
from world_model.simulator.physics import PhysicsStepEvents, empty_physics_events
from world_model.simulator.sphere_world import SphereWorld, SphereWorldConfig

Episode = dict[str, Any]


def _stack_records(records: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot stack an empty record list")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise ValueError("record keys changed within an episode")
    return {key: torch.stack([record[key] for record in records]) for key in keys}


def _state_record(world: SphereWorld, visible_fraction: Tensor) -> dict[str, Tensor]:
    state = world.state
    return {
        "id": state.object_id.clone(),
        "active": state.active.clone(),
        "position": state.position.clone(),
        "velocity": state.velocity.clone(),
        "orientation": state.orientation.clone(),
        "angular_velocity": state.angular_velocity.clone(),
        "radius": state.radius.clone(),
        "mass": state.mass.clone(),
        "restitution": state.restitution.clone(),
        "drag": state.drag.clone(),
        "friction": state.friction.clone(),
        "albedo": state.albedo.clone(),
        "visible_fraction": visible_fraction.clone(),
        "sleeping": state.sleeping.clone(),
    }


def _event_record(
    physics: PhysicsStepEvents,
    *,
    created: Tensor,
    removed: Tensor,
    interval_start: float,
) -> dict[str, Tensor]:
    first_event_time = torch.where(
        physics.first_event_offset >= 0,
        physics.first_event_offset + interval_start,
        physics.first_event_offset,
    )
    floor_index = BOUNDARY_NAMES.index("floor")
    wall_indices = [
        index for index, name in enumerate(BOUNDARY_NAMES) if name not in {"floor", "ceiling"}
    ]
    return {
        "pair_contact": physics.pair_contact.clone(),
        "pair_collision": physics.pair_collision.clone(),
        "sphere_sphere": physics.pair_collision.clone(),
        "pair_impulse": physics.pair_impulse.clone(),
        "pair_penetration": physics.pair_penetration.clone(),
        "boundary_contact": physics.boundary_contact.clone(),
        "boundary_collision": physics.boundary_collision.clone(),
        "boundary_impulse": physics.boundary_impulse.clone(),
        "boundary_penetration": physics.boundary_penetration.clone(),
        "ground_contact": physics.boundary_contact[:, floor_index].clone(),
        "ground_collision": physics.boundary_collision[:, floor_index].clone(),
        "wall_collision": physics.boundary_collision[:, wall_indices].clone(),
        "collision": physics.collision.clone(),
        "contact": physics.contact.clone(),
        "sleeping": physics.sleeping.clone(),
        "external_impulse": physics.external_impulse.clone(),
        "externally_actuated": (torch.linalg.vector_norm(physics.external_impulse, dim=-1) > 0),
        "created": created.clone(),
        "removed": removed.clone(),
        "first_event_time": first_event_time,
    }


def _camera_velocities(
    world_from_camera: Tensor,
    timestamps: Tensor,
) -> tuple[Tensor, Tensor]:
    """Finite-difference camera translation and world-frame angular velocity."""

    count = world_from_camera.shape[0]
    position = world_from_camera[:, :3, 3]
    linear = torch.zeros_like(position)
    angular = torch.zeros_like(position)
    if count <= 1:
        return linear, angular
    dt = (timestamps[1:] - timestamps[:-1]).clamp_min(1.0e-8)
    linear[1:] = (position[1:] - position[:-1]) / dt[:, None]
    linear[0] = linear[1]
    rotation = world_from_camera[:, :3, :3]
    delta = rotation[1:] @ rotation[:-1].transpose(-1, -2)
    skew = 0.5 * (delta - delta.transpose(-1, -2))
    rotation_vector = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=-1)
    angular[1:] = rotation_vector / dt[:, None]
    angular[0] = angular[1]
    return linear, angular


def generate_episode(
    config: SphereWorldConfig | Mapping[str, Any] | Any,
    seed: int,
) -> Episode:
    """Generate one deterministic, padded, fully labelled RGB episode."""

    resolved = SphereWorldConfig.from_config(config)
    world = SphereWorld(resolved, seed)
    rgb_frames: list[Tensor] = []
    state_records: list[dict[str, Tensor]] = []
    label_records: list[dict[str, Tensor]] = []
    event_records: list[dict[str, Tensor]] = []
    camera_world_from: list[Tensor] = []
    camera_from_world: list[Tensor] = []
    camera_intrinsics: list[Tensor] = []
    camera_position: list[Tensor] = []
    camera_target: list[Tensor] = []
    timestamps = torch.arange(resolved.sequence_frames, dtype=torch.float32) / resolved.frame_rate
    pending_physics = empty_physics_events(resolved.n_max)

    for frame_index, timestamp_tensor in enumerate(timestamps):
        timestamp = float(timestamp_tensor)
        lifecycle = world.apply_lifecycle(frame_index)
        camera = world.camera_frame(timestamp)
        rendered = world.render(camera=camera)
        labels = make_perception_labels(world.state, rendered, resolved.image_size)
        validate_perception_labels(
            labels,
            max_objects=resolved.n_max,
            image_size=resolved.image_size,
        )
        rgb_frames.append(rendered.rgb)
        state_records.append(_state_record(world, rendered.visible_fraction))
        label_records.append(labels)
        interval_start = max(0.0, timestamp - resolved.observation_dt)
        event_records.append(
            _event_record(
                pending_physics,
                created=lifecycle.created,
                removed=lifecycle.removed,
                interval_start=interval_start,
            )
        )
        camera_world_from.append(camera.world_from_camera)
        camera_from_world.append(camera.camera_from_world)
        camera_intrinsics.append(camera.intrinsics)
        camera_position.append(camera.position)
        camera_target.append(camera.target)
        if frame_index + 1 < resolved.sequence_frames:
            pending_physics = world.step(resolved.observation_dt)

    world_from_camera = torch.stack(camera_world_from)
    camera_from_world_tensor = torch.stack(camera_from_world)
    intrinsics = torch.stack(camera_intrinsics)
    linear_velocity, angular_velocity = _camera_velocities(world_from_camera, timestamps)
    objects = _stack_records(state_records)
    labels = _stack_records(label_records)
    # Perception labels are duplicated into the object record only for the
    # common compact fields specified by the canonical episode contract.
    for key in (
        "projected_center",
        "projected_center_pixels",
        "apparent_radius",
        "apparent_radius_normalized",
        "inverse_depth",
        "camera_depth",
        "projected_valid",
    ):
        objects[key] = labels[key]

    episode: Episode = {
        "rgb": torch.stack(rgb_frames).to(torch.float32),
        "timestamps": timestamps,
        "frame_mask": torch.ones(resolved.sequence_frames, dtype=torch.bool),
        "camera": {
            "world_from_camera": world_from_camera,
            "camera_from_world": camera_from_world_tensor,
            "intrinsics": intrinsics,
            "position": torch.stack(camera_position),
            "target": torch.stack(camera_target),
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "calibrated": torch.ones(resolved.sequence_frames, dtype=torch.bool),
        },
        "objects": objects,
        "events": _stack_records(event_records),
        "labels": labels,
        "seed": int(seed),
        "num_objects": int((world._spawn_frame >= 0).sum()),
        "metadata": {
            "simulator": "sphere_world",
            "simulator_version": 1,
            "distribution": resolved.distribution,
            "scenario": world.scenario_name,
            "camera_trajectory": world.camera.mode,
            "frame_rate": resolved.frame_rate,
            "physics_rate": resolved.physics_rate,
            "boundary_names": BOUNDARY_NAMES,
        },
    }
    validate_episode(episode, resolved)
    return episode


def validate_episode(
    episode: Mapping[str, Any],
    config: SphereWorldConfig | Mapping[str, Any] | Any | None = None,
) -> None:
    """Validate canonical sequence dimensions, padding, transforms, and ranges."""

    required = {
        "rgb",
        "timestamps",
        "frame_mask",
        "camera",
        "objects",
        "events",
        "labels",
        "seed",
    }
    missing = required.difference(episode)
    if missing:
        raise ValueError(f"episode is missing required keys: {sorted(missing)}")
    rgb = episode["rgb"]
    timestamps = episode["timestamps"]
    objects = episode["objects"]
    if not isinstance(rgb, Tensor) or rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [T, 3, H, W]")
    time, _, height, width = rgb.shape
    if timestamps.shape != (time,):
        raise ValueError("timestamps must have shape [T]")
    if time > 1 and bool(torch.any(timestamps[1:] <= timestamps[:-1])):
        raise ValueError("episode timestamps must be strictly increasing")
    if rgb.dtype != torch.float32 or not bool(torch.isfinite(rgb).all()):
        raise ValueError("rgb must be finite torch.float32")
    if bool(torch.any((rgb < 0) | (rgb > 1))):
        raise ValueError("rgb values must lie in [0, 1]")
    active = objects["active"]
    object_id = objects["id"]
    if active.ndim != 2 or active.shape[0] != time:
        raise ValueError("objects.active must have shape [T, N]")
    n_max = active.shape[1]
    expected_object_shapes = {
        "id": (time, n_max),
        "position": (time, n_max, 3),
        "velocity": (time, n_max, 3),
        "radius": (time, n_max, 1),
        "mass": (time, n_max, 1),
        "restitution": (time, n_max, 1),
        "drag": (time, n_max, 1),
        "friction": (time, n_max, 1),
        "visible_fraction": (time, n_max),
    }
    for key, shape in expected_object_shapes.items():
        if key not in objects or tuple(objects[key].shape) != shape:
            actual = None if key not in objects else tuple(objects[key].shape)
            raise ValueError(f"objects.{key} expected {shape}, got {actual}")
    if bool(torch.any(active & (object_id < 0))):
        raise ValueError("active simulator objects need nonnegative IDs")
    if bool(torch.any((~active) & (object_id != -1))):
        raise ValueError("inactive/padded slots need ID -1")
    if not bool(torch.isfinite(objects["position"]).all()):
        raise ValueError("object positions contain NaN or Inf")
    camera = episode["camera"]
    for key, shape in {
        "world_from_camera": (time, 4, 4),
        "camera_from_world": (time, 4, 4),
        "intrinsics": (time, 3, 3),
    }.items():
        if key not in camera or tuple(camera[key].shape) != shape:
            raise ValueError(f"camera.{key} must have shape {shape}")
    identity = camera["world_from_camera"] @ camera["camera_from_world"]
    if not torch.allclose(
        identity,
        torch.eye(4).expand_as(identity),
        atol=3.0e-5,
        rtol=3.0e-5,
    ):
        raise ValueError("camera transforms do not compose to identity")
    labels = episode["labels"]
    expected_mask_shape = (time, n_max, height, width)
    if tuple(labels["segmentation_mask"].shape) != expected_mask_shape:
        raise ValueError(f"labels.segmentation_mask must have shape {expected_mask_shape}")
    if config is not None:
        resolved = SphereWorldConfig.from_config(config)
        if (height, width) != resolved.image_size:
            raise ValueError("episode image size differs from simulator config")
        if time != resolved.sequence_frames or n_max != resolved.n_max:
            raise ValueError("episode sequence/object padding differs from config")
