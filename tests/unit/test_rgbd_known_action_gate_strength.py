"""Ungoverned metric-geometry screen for the specification-1.60.2 profile.

The fixtures are public synthetic scenes.  They exercise the formal absolute
position, palette-invariance, and RGB-connectivity thresholds without opening
or deriving any governed qualification row.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.observations.rgbd.two_disc_geometry import two_disc_geometry_from_rgbd
from world_model.simulator import CameraFrame, SphereState, make_intrinsics, render_spheres
from world_model.training import rgbd_known_action_qualification as qualification

IMAGE_SIZE = (64, 64)
RADIUS_M = 0.21
POSITION_GATE_M = 2.0e-6
PALETTE_GATE_M = 2.0e-6
RGB_VJP_GATE_L1 = 1.0e-8

SCENES = (
    (
        ((-0.55, 0.25, 3.0), (0.70, -0.20, 5.8)),
        ((0.12, 0.90, 0.20), (0.86, 0.12, 0.78)),
    ),
    (
        ((-0.72, -0.20, 5.4), (0.62, 0.24, 3.2)),
        ((0.12, 0.25, 0.92), (0.92, 0.78, 0.12)),
    ),
    (
        ((-0.82, 0.18, 4.8), (0.74, -0.22, 3.5)),
        ((0.20, 0.88, 0.84), (0.91, 0.25, 0.12)),
    ),
)


def _camera() -> CameraFrame:
    identity = torch.eye(4, dtype=torch.float32)
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics(IMAGE_SIZE, 50.0),
        position=torch.zeros(3),
        target=torch.tensor([0.0, 0.0, 1.0]),
    )


def _state(
    positions: tuple[tuple[float, float, float], tuple[float, float, float]],
    colours: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> SphereState:
    count = 2
    return SphereState(
        object_id=torch.arange(count, dtype=torch.int64),
        active=torch.ones(count, dtype=torch.bool),
        position=torch.tensor(positions, dtype=torch.float32),
        velocity=torch.zeros((count, 3)),
        radius=torch.full((count, 1), RADIUS_M),
        mass=torch.ones((count, 1)),
        restitution=torch.zeros((count, 1)),
        drag=torch.zeros((count, 1)),
        friction=torch.zeros((count, 1)),
        albedo=torch.tensor(colours, dtype=torch.float32),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(count, -1).clone(),
        angular_velocity=torch.zeros((count, 3)),
        sleeping=torch.zeros(count, dtype=torch.bool),
        sleep_counter=torch.zeros(count, dtype=torch.int64),
    )


def _measure(
    state: SphereState,
    camera: CameraFrame,
    *,
    blend: float,
    require_rgb_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = render_spheres(state, camera, IMAGE_SIZE)
    assert rendered.visible_fraction.tolist() == [1.0, 1.0]
    assert not bool((rendered.full_mask[0] & rendered.full_mask[1]).any())
    rgb = rendered.rgb.unsqueeze(0)
    if require_rgb_grad:
        rgb = rgb.requires_grad_(True)
    measured = two_disc_geometry_from_rgbd(
        rgb,
        rendered.depth_buffer[None, None],
        RADIUS_M,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
        chromatic_centre_blend=blend,
    )
    assert measured.pair_valid_mask.tolist() == [True]
    return measured.world_position[0], rgb


def _set_max_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    direct = torch.linalg.vector_norm(actual - expected, dim=-1)
    swapped = torch.linalg.vector_norm(actual - expected.flip(0), dim=-1)
    return direct.max() if direct.square().sum() <= swapped.square().sum() else swapped.max()


@pytest.mark.parametrize(("positions", "colours"), SCENES)
def test_known_action_profile_passes_ungoverned_gate_strength_screen(
    positions: tuple[tuple[float, float, float], tuple[float, float, float]],
    colours: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> None:
    qualification._activate_runtime_dependencies()
    config = qualification.require_frozen_config(
        qualification.REPOSITORY_ROOT / qualification.CONFIG_RELATIVE_PATH
    )
    blend = config.model.rgbd.chromatic_centre_blend
    camera = _camera()
    state = _state(positions, colours)

    measured, rgb = _measure(state, camera, blend=blend, require_rgb_grad=True)
    swapped, _ = _measure(
        replace(state, albedo=state.albedo.flip(0).clone()),
        camera,
        blend=blend,
    )
    rgb_vjp = torch.autograd.grad(measured.square().sum(), rgb)[0]

    assert float(_set_max_error(measured, state.position).detach()) <= POSITION_GATE_M
    assert float(torch.cdist(measured, swapped).amin(dim=-1).max().detach()) <= PALETTE_GATE_M
    assert torch.isfinite(rgb_vjp).all()
    assert float(rgb_vjp.abs().sum()) >= RGB_VJP_GATE_L1


@pytest.mark.parametrize(("positions", "colours"), SCENES)
def test_gate_strength_screen_rejects_superseded_forward_blend(
    positions: tuple[tuple[float, float, float], tuple[float, float, float]],
    colours: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> None:
    camera = _camera()
    state = _state(positions, colours)
    measured, _ = _measure(state, camera, blend=0.0025)
    swapped, _ = _measure(
        replace(state, albedo=state.albedo.flip(0).clone()),
        camera,
        blend=0.0025,
    )

    truth_error = float(_set_max_error(measured, state.position))
    palette_error = float(torch.cdist(measured, swapped).amin(dim=-1).max())
    assert truth_error > POSITION_GATE_M or palette_error > PALETTE_GATE_M
