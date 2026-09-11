from __future__ import annotations

import inspect
import math
from dataclasses import replace

import torch

from world_model.belief import RigidPrimitive
from world_model.observations.rgbd import fit_rigid_geometry_from_rgbd
from world_model.simulator import (
    CameraFrame,
    RigidBodyState,
    SphereState,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    render_rigid_bodies,
)

IMAGE_SIZE = (81, 81)


def _sphere_state() -> SphereState:
    return SphereState(
        object_id=torch.tensor([3], dtype=torch.int64),
        active=torch.tensor([True]),
        position=torch.zeros(1, 3, dtype=torch.float64),
        velocity=torch.zeros(1, 3, dtype=torch.float64),
        radius=torch.tensor([[0.32]], dtype=torch.float64),
        mass=torch.ones(1, 1, dtype=torch.float64),
        restitution=torch.full((1, 1), 0.7, dtype=torch.float64),
        drag=torch.zeros(1, 1, dtype=torch.float64),
        friction=torch.full((1, 1), 0.2, dtype=torch.float64),
        albedo=torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float64),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
        angular_velocity=torch.zeros(1, 3, dtype=torch.float64),
        sleeping=torch.zeros(1, dtype=torch.bool),
        sleep_counter=torch.zeros(1, dtype=torch.int64),
    )


def _cameras() -> list[CameraFrame]:
    positions = (
        (0.0, 0.0, -3.0),
        (0.0, 0.0, 3.0),
        (-3.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, -3.0, 0.0),
        (0.0, 3.0, 0.0),
    )
    frames = []
    for timestamp, values in enumerate(positions):
        position = torch.tensor(values, dtype=torch.float64)
        world_up = (
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) if abs(values[1]) > 2.0 else None
        )
        world_from_camera = look_at_world_from_camera(
            position,
            torch.zeros(3, dtype=torch.float64),
            world_up=world_up,
        )
        frames.append(
            CameraFrame(
                timestamp=float(timestamp),
                world_from_camera=world_from_camera,
                camera_from_world=invert_rigid_transform(world_from_camera),
                intrinsics=make_intrinsics(IMAGE_SIZE, 38.0, dtype=torch.float64),
                position=position,
                target=torch.zeros(3, dtype=torch.float64),
            )
        )
    return frames


def _public_surfaces(state: RigidBodyState) -> tuple[torch.Tensor, ...]:
    rendered = [render_rigid_bodies(state, camera, IMAGE_SIZE) for camera in _cameras()]
    depth = torch.stack([frame.depth_buffer for frame in rendered])
    # A single isolated foreground body needs no private instance label: valid
    # measured depth is exactly the public segmentation signal.
    mask = (depth > 0.0).to(depth.dtype)
    transforms = torch.stack([camera.world_from_camera for camera in _cameras()])
    intrinsics = torch.stack([camera.intrinsics for camera in _cameras()])
    return depth, mask, transforms, intrinsics


def test_multi_view_public_depth_recovers_box_shape_and_centre() -> None:
    sphere = _sphere_state()
    half_extents = torch.tensor([[0.46, 0.29, 0.18]], dtype=torch.float64)
    angle = math.radians(18.0)
    rigid = replace(
        RigidBodyState.from_spheres(sphere),
        primitive=torch.tensor([int(RigidPrimitive.BOX)], dtype=torch.int64),
        half_extents=half_extents,
        radius=torch.linalg.vector_norm(half_extents, dim=-1, keepdim=True),
        orientation=torch.tensor(
            [[0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]],
            dtype=torch.float64,
        ),
    )
    measured = fit_rigid_geometry_from_rgbd(*_public_surfaces(rigid))

    assert measured.valid
    assert measured.primitive.item() == int(RigidPrimitive.BOX)
    assert measured.point_count.item() >= 256
    torch.testing.assert_close(
        measured.world_position,
        torch.zeros(3, dtype=torch.float64),
        atol=0.025,
        rtol=0.0,
    )
    torch.testing.assert_close(
        measured.half_extents.sort().values,
        half_extents[0].sort().values,
        atol=0.035,
        rtol=0.0,
    )
    assert measured.box_residual < 0.72 * measured.sphere_residual


def test_multi_view_public_depth_retains_sphere_classification() -> None:
    rigid = RigidBodyState.from_spheres(_sphere_state())
    measured = fit_rigid_geometry_from_rgbd(*_public_surfaces(rigid))

    assert measured.valid
    assert measured.primitive.item() == int(RigidPrimitive.SPHERE)
    torch.testing.assert_close(
        measured.world_position,
        torch.zeros(3, dtype=torch.float64),
        atol=0.02,
        rtol=0.0,
    )
    torch.testing.assert_close(
        measured.radius,
        torch.tensor(0.32, dtype=torch.float64),
        atol=0.025,
        rtol=0.0,
    )


def test_observable_fitter_cannot_accept_private_truth_or_identity() -> None:
    parameters = set(inspect.signature(fit_rigid_geometry_from_rgbd).parameters)
    assert not parameters & {
        "object_id",
        "primitive",
        "half_extents",
        "instance_map",
        "simulator_state",
        "truth",
    }
