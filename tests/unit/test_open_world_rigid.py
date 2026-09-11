from __future__ import annotations

import inspect
import math
from dataclasses import replace

import torch

from world_model.belief import RigidPrimitive
from world_model.observations.rgbd import (
    DiscoveredRigidObject,
    OpenWorldRigidFrame,
    OpenWorldRigidTracker,
    discover_rigid_objects_from_rgbd,
    tracked_objects_to_belief,
)
from world_model.observations.rgbd.rigid_geometry import ObservableRigidGeometry
from world_model.simulator import (
    CameraFrame,
    RigidBodyState,
    SphereState,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    render_rigid_bodies,
)

DTYPE = torch.float64
IMAGE_SIZE = (96, 96)


def _state() -> RigidBodyState:
    sphere = SphereState(
        object_id=torch.tensor([91, 37, 58]),
        active=torch.ones(3, dtype=torch.bool),
        position=torch.tensor(
            [[-0.85, -0.48, 4.0], [0.85, -0.46, 4.05], [0.0, 0.75, 4.1]],
            dtype=DTYPE,
        ),
        velocity=torch.zeros(3, 3, dtype=DTYPE),
        radius=torch.tensor([[0.28], [0.30], [0.27]], dtype=DTYPE),
        mass=torch.tensor([[1.3], [0.8], [1.7]], dtype=DTYPE),
        restitution=torch.tensor([[0.62], [0.47], [0.71]], dtype=DTYPE),
        drag=torch.tensor([[0.03], [0.07], [0.02]], dtype=DTYPE),
        friction=torch.tensor([[0.28], [0.41], [0.35]], dtype=DTYPE),
        albedo=torch.tensor(
            [[0.73, 0.21, 0.56], [0.18, 0.79, 0.44], [0.31, 0.38, 0.91]],
            dtype=DTYPE,
        ),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=DTYPE),
        angular_velocity=torch.zeros(3, 3, dtype=DTYPE),
        sleeping=torch.zeros(3, dtype=torch.bool),
        sleep_counter=torch.zeros(3, dtype=torch.int64),
    )
    rigid = RigidBodyState.from_spheres(sphere)
    box_extent = torch.tensor([0.32, 0.23, 0.19], dtype=DTYPE)
    return replace(
        rigid,
        primitive=torch.tensor(
            [int(RigidPrimitive.BOX), int(RigidPrimitive.SPHERE), int(RigidPrimitive.BOX)]
        ),
        half_extents=torch.stack(
            (box_extent, torch.full((3,), 0.30, dtype=DTYPE), box_extent * 0.9)
        ),
        radius=torch.tensor(
            [
                [float(torch.linalg.vector_norm(box_extent))],
                [0.30],
                [float(torch.linalg.vector_norm(box_extent * 0.9))],
            ],
            dtype=DTYPE,
        ),
    )


def _cameras() -> tuple[CameraFrame, ...]:
    target = torch.tensor([0.0, 0.0, 4.0], dtype=DTYPE)
    frames = []
    for index, offset in enumerate(
        (
            (0.0, 0.0, -4.0),
            (0.0, 0.0, 4.0),
            (-4.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (0.0, -4.0, 0.0),
            (0.0, 4.0, 0.0),
        )
    ):
        position = target + target.new_tensor(offset)
        world_up = target.new_tensor([0.0, 0.0, 1.0]) if abs(offset[1]) > 3.0 else None
        world_from_camera = look_at_world_from_camera(position, target, world_up=world_up)
        frames.append(
            CameraFrame(
                timestamp=float(index),
                world_from_camera=world_from_camera,
                camera_from_world=invert_rigid_transform(world_from_camera),
                intrinsics=make_intrinsics(IMAGE_SIZE, 56.0, dtype=DTYPE),
                position=position,
                target=target,
            )
        )
    return tuple(frames)


def _discovery(state: RigidBodyState, timestamp: float = 0.0) -> OpenWorldRigidFrame:
    cameras = _cameras()
    rendered = [render_rigid_bodies(state, camera, IMAGE_SIZE) for camera in cameras]
    return discover_rigid_objects_from_rgbd(
        torch.stack([item.rgb.to(DTYPE).permute(1, 2, 0) for item in rendered]),
        torch.stack([item.depth_buffer for item in rendered]),
        torch.stack([item.world_from_camera for item in cameras]),
        torch.stack([item.intrinsics for item in cameras]),
        timestamp=timestamp,
    )


def _geometry(position: tuple[float, float, float], angle: float) -> ObservableRigidGeometry:
    return ObservableRigidGeometry(
        primitive=torch.tensor(int(RigidPrimitive.BOX)),
        world_position=torch.tensor(position, dtype=DTYPE),
        orientation=torch.tensor(
            [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)], dtype=DTYPE
        ),
        radius=torch.tensor(0.5, dtype=DTYPE),
        half_extents=torch.tensor([0.4, 0.3, 0.2], dtype=DTYPE),
        sphere_residual=torch.tensor(0.1, dtype=DTYPE),
        box_residual=torch.tensor(0.001, dtype=DTYPE),
        point_count=torch.tensor(500),
        valid=torch.tensor(True),
    )


def _detection(position: tuple[float, float, float], angle: float) -> DiscoveredRigidObject:
    return DiscoveredRigidObject(
        geometry=_geometry(position, angle),
        appearance=torch.nn.functional.normalize(
            torch.tensor([0.7, 0.2, 0.4, 0.7, 0.5, 0.3, 0.0, 1.0], dtype=DTYPE),
            dim=0,
        ),
        support_views=3,
        confidence=1.0,
    )


def test_public_discovery_has_no_prototype_or_private_identity_input() -> None:
    parameters = inspect.signature(discover_rigid_objects_from_rgbd).parameters
    assert "prototype" not in parameters
    assert "instance_map" not in parameters
    discovered = _discovery(_state())

    assert len(discovered.objects) == 3
    assert {int(item.geometry.primitive) for item in discovered.objects} == {
        int(RigidPrimitive.SPHERE),
        int(RigidPrimitive.BOX),
    }
    assert all(item.support_views >= 3 for item in discovered.objects)
    observed = torch.stack([item.geometry.world_position for item in discovered.objects])
    truth = _state().position
    distances = torch.cdist(observed, truth)
    assert torch.all(distances.min(dim=1).values < 0.04)


def test_tracker_recovers_id_and_angular_velocity_after_one_missing_frame() -> None:
    tracker = OpenWorldRigidTracker(max_missed_steps=2)
    first = tracker.update(OpenWorldRigidFrame(0.0, (_detection((0.0, 0.0, 4.0), 0.0),)))
    missing = tracker.update(OpenWorldRigidFrame(0.05, ()))
    recovered = tracker.update(OpenWorldRigidFrame(0.10, (_detection((0.02, 0.0, 4.0), 0.10),)))

    assert first[0].object_id == missing[0].object_id == recovered[0].object_id
    assert not missing[0].observed and missing[0].missed_steps == 1
    assert recovered[0].observed and recovered[0].missed_steps == 0
    assert abs(float(recovered[0].angular_velocity[2]) - 1.0) < 0.08


def test_belief_uses_explicit_neutral_parameter_priors() -> None:
    tracker = OpenWorldRigidTracker()
    tracked = tracker.update(OpenWorldRigidFrame(0.0, (_detection((0.0, 0.0, 4.0), 0.0),)))
    belief = tracked_objects_to_belief(
        tracked,
        timestamp=0.0,
        max_objects=4,
        initial_mass=1.0,
        initial_restitution=0.5,
        initial_drag=0.05,
        initial_friction=0.25,
    )

    assert belief.active_modalities == ("rgbd_open_world",)
    assert belief.objects.active.sum() == 1
    torch.testing.assert_close(belief.objects.mass[0, 0], torch.tensor([1.0], dtype=DTYPE))
    torch.testing.assert_close(belief.objects.restitution[0, 0], torch.tensor([0.5], dtype=DTYPE))
    torch.testing.assert_close(belief.objects.drag[0, 0], torch.tensor([0.05], dtype=DTYPE))
    torch.testing.assert_close(belief.objects.friction[0, 0], torch.tensor([0.25], dtype=DTYPE))
