from __future__ import annotations

import math
from dataclasses import replace

import torch

from world_model.belief import RigidPrimitive
from world_model.observations import ObservationPacket
from world_model.simulator import (
    CameraFrame,
    RigidBodyState,
    SphereState,
    make_intrinsics,
    render_rigid_bodies,
    render_spheres,
)


def _camera() -> CameraFrame:
    identity = torch.eye(4, dtype=torch.float64)
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics((65, 65), 48.0, dtype=torch.float64),
        position=torch.zeros(3, dtype=torch.float64),
        target=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    )


def _sphere() -> SphereState:
    return SphereState(
        object_id=torch.tensor([7], dtype=torch.int64),
        active=torch.tensor([True]),
        position=torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float64),
        velocity=torch.zeros(1, 3, dtype=torch.float64),
        radius=torch.tensor([[0.3]], dtype=torch.float64),
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


def _box(*, quarter_turn: bool = False) -> RigidBodyState:
    state = RigidBodyState.from_spheres(_sphere())
    half_angle = math.pi / 4.0 if quarter_turn else 0.0
    return replace(
        state,
        primitive=torch.tensor([int(RigidPrimitive.BOX)], dtype=torch.int64),
        radius=torch.tensor([[math.sqrt(0.4**2 + 0.15**2 + 0.3**2)]], dtype=torch.float64),
        half_extents=torch.tensor([[0.4, 0.15, 0.3]], dtype=torch.float64),
        orientation=torch.tensor(
            [[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]],
            dtype=torch.float64,
        ),
    )


def test_all_sphere_rigid_render_is_exact_legacy_oracle() -> None:
    sphere = _sphere()

    expected = render_spheres(sphere, _camera(), (65, 65))
    actual = render_rigid_bodies(RigidBodyState.from_spheres(sphere), _camera(), (65, 65))

    for field in expected.__dataclass_fields__:
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field


def test_axis_aligned_box_has_metric_front_depth_and_private_labels() -> None:
    output = render_rigid_bodies(_box(), _camera(), (65, 65))

    assert output.projected_valid.tolist() == [True]
    torch.testing.assert_close(
        output.depth_buffer[32, 32],
        torch.tensor(3.7, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert output.instance_map[32, 32].item() == 7
    packet = ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=0.0,
        payload={
            "rgb": output.rgb.unsqueeze(0),
            "depth": output.depth_buffer[None, None],
        },
        calibration={
            "world_from_camera": _camera().world_from_camera.unsqueeze(0),
            "intrinsics": _camera().intrinsics.unsqueeze(0),
        },
        frame_id="camera:camera0:rgbd",
        metadata={"image_size": (65, 65)},
    )
    assert set(packet.payload) == {"rgb", "depth"}
    assert "instance_map" not in packet.payload


def test_oriented_box_silhouette_tracks_public_orientation() -> None:
    horizontal = render_rigid_bodies(_box(), _camera(), (65, 65)).full_mask[0]
    vertical = render_rigid_bodies(_box(quarter_turn=True), _camera(), (65, 65)).full_mask[0]

    horizontal_rows, horizontal_columns = torch.where(horizontal)
    vertical_rows, vertical_columns = torch.where(vertical)
    horizontal_width = int(horizontal_columns.max() - horizontal_columns.min() + 1)
    horizontal_height = int(horizontal_rows.max() - horizontal_rows.min() + 1)
    vertical_width = int(vertical_columns.max() - vertical_columns.min() + 1)
    vertical_height = int(vertical_rows.max() - vertical_rows.min() + 1)
    assert horizontal_width > horizontal_height
    assert vertical_height > vertical_width
    assert abs(horizontal_width - vertical_height) <= 1
    assert abs(horizontal_height - vertical_width) <= 1
