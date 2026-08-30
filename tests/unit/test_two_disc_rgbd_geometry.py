"""Seed-free tests for the fully visible two-sphere RGB-D primitive."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from world_model.observations.rgbd.two_disc_geometry import (
    two_disc_geometry_from_rgbd,
)
from world_model.simulator import CameraFrame, SphereState, make_intrinsics, render_spheres

IMAGE_SIZE = (64, 80)
RADIUS_M = 0.3
PALETTE_SET_INVARIANCE_M = 5.0e-4


def _camera(dtype: torch.dtype = torch.float32) -> CameraFrame:
    identity = torch.eye(4, dtype=dtype)
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics(IMAGE_SIZE, 50.0).to(dtype=dtype),
        position=torch.zeros(3, dtype=dtype),
        target=torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
    )


def _state(
    colours: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> SphereState:
    count = 2
    return SphereState(
        object_id=torch.arange(count, dtype=torch.int64),
        active=torch.ones(count, dtype=torch.bool),
        position=torch.tensor(
            [[-0.72, -0.12, 4.0], [0.78, 0.16, 4.35]],
            dtype=torch.float32,
        ),
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


def _render(
    colours: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.90, 0.20, 0.18),
        (0.18, 0.82, 0.90),
    ),
) -> tuple[torch.Tensor, torch.Tensor, CameraFrame, SphereState]:
    camera = _camera()
    state = _state(colours)
    rendered = render_spheres(state, camera, IMAGE_SIZE)
    return rendered.rgb.unsqueeze(0), rendered.depth_buffer[None, None], camera, state


def _render_custom(
    positions: tuple[tuple[float, float, float], tuple[float, float, float]],
    colours: tuple[tuple[float, float, float], tuple[float, float, float]],
    radius: float,
):
    camera = _camera()
    state = replace(
        _state(colours),
        position=torch.tensor(positions, dtype=torch.float32),
        radius=torch.full((2, 1), radius),
    )
    rendered = render_spheres(state, camera, IMAGE_SIZE)
    measured = two_disc_geometry_from_rgbd(
        rendered.rgb.unsqueeze(0),
        rendered.depth_buffer[None, None],
        radius,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
    )
    return rendered, measured, state


def _render_variable_radius(
    radii: tuple[float, float] = (0.24, 0.18),
) -> tuple[torch.Tensor, torch.Tensor, CameraFrame, SphereState]:
    camera = _camera()
    state = replace(
        _state(((0.90, 0.20, 0.18), (0.18, 0.82, 0.90))),
        radius=torch.tensor(radii, dtype=torch.float32).unsqueeze(-1),
    )
    rendered = render_spheres(state, camera, IMAGE_SIZE)
    return rendered.rgb.unsqueeze(0), rendered.depth_buffer[None, None], camera, state


def _measurement(
    image: torch.Tensor,
    depth: torch.Tensor,
    camera: CameraFrame,
):
    return two_disc_geometry_from_rgbd(
        image,
        depth,
        RADIUS_M,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
    )


def _best_set_error(measured: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    direct = torch.linalg.vector_norm(measured - expected, dim=-1)
    swapped = torch.linalg.vector_norm(measured - expected.flip(0), dim=-1)
    return direct if direct.square().sum() <= swapped.square().sum() else swapped


def _assert_tensor_dataclass_bitwise_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    assert is_dataclass(left)
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value), field.name
        else:
            _assert_tensor_dataclass_bitwise_equal(left_value, right_value)


def test_two_visible_discs_recover_one_unordered_metric_measurement_each() -> None:
    image, depth, camera, state = _render()

    measured = _measurement(image, depth, camera)
    error = _best_set_error(measured.world_position[0], state.position)

    assert measured.pair_valid_mask.tolist() == [True]
    assert measured.valid_mask.tolist() == [[True, True]]
    assert float(error.max()) < 0.008
    assert float(torch.sqrt(error.square().mean())) < 0.005
    assert float(measured.chromatic_eigengap[0]) >= 0.01
    assert torch.linalg.vector_norm(measured.appearance, dim=-1).allclose(
        torch.ones((1, 2)),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_unknown_metric_radii_are_recovered_without_using_the_fixed_prior() -> None:
    image, depth, camera, state = _render_variable_radius()

    measured = two_disc_geometry_from_rgbd(
        image,
        depth,
        0.12,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
        estimate_world_radius=True,
        minimum_world_radius=0.10,
        maximum_world_radius=0.35,
    )
    alternate_prior = two_disc_geometry_from_rgbd(
        image,
        depth,
        0.34,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
        estimate_world_radius=True,
        minimum_world_radius=0.10,
        maximum_world_radius=0.35,
    )
    direct_error = (measured.world_position[0] - state.position).square().sum()
    swapped_error = (measured.world_position[0] - state.position.flip(0)).square().sum()
    expected_radius = state.radius[:, 0]
    if swapped_error < direct_error:
        expected_radius = expected_radius.flip(0)

    assert measured.valid_mask.tolist() == [[True, True]]
    torch.testing.assert_close(
        measured.surface_fit_radius[0],
        expected_radius,
        atol=2.0e-5,
        rtol=0.0,
    )
    assert float(measured.surface_fit_radius_relative_error.max()) < 1.0e-4
    torch.testing.assert_close(
        alternate_prior.surface_fit_radius,
        measured.surface_fit_radius,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        alternate_prior.world_position,
        measured.world_position,
        atol=0.0,
        rtol=0.0,
    )


def test_disabled_radius_estimator_is_bitwise_legacy_and_ignores_new_bounds() -> None:
    image, depth, camera, _ = _render()

    default = _measurement(image, depth, camera)
    explicit_disabled = two_disc_geometry_from_rgbd(
        image,
        depth,
        RADIUS_M,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
        estimate_world_radius=False,
        minimum_world_radius=0.30,
        maximum_world_radius=0.10,
    )

    _assert_tensor_dataclass_bitwise_equal(default, explicit_disabled)


def test_unknown_radius_depth_and_intrinsics_vjps_match_central_differences() -> None:
    image, depth, camera, _ = _render_variable_radius()
    image = image.to(torch.float64)
    depth = depth.to(torch.float64).requires_grad_(True)
    world_from_camera = camera.world_from_camera.unsqueeze(0).to(torch.float64)
    intrinsics = camera.intrinsics.unsqueeze(0).to(torch.float64).requires_grad_(True)

    def objective(depth_value: torch.Tensor, intrinsics_value: torch.Tensor) -> torch.Tensor:
        result = two_disc_geometry_from_rgbd(
            image,
            depth_value,
            0.21,
            world_from_camera,
            intrinsics_value,
            estimate_world_radius=True,
            minimum_world_radius=0.10,
            maximum_world_radius=0.35,
        )
        assert result.valid_mask.all()
        return result.surface_fit_radius.sum()

    value = objective(depth, intrinsics)
    depth_gradient, intrinsics_gradient = torch.autograd.grad(
        value,
        (depth, intrinsics),
    )
    depth_flat_index = int(depth_gradient.abs().argmax())
    intrinsics_flat_index = int(intrinsics_gradient.abs().argmax())

    def central_difference(
        source: torch.Tensor,
        flat_index: int,
        step: float,
        *,
        depth_source: bool,
    ) -> torch.Tensor:
        positive = source.detach().clone()
        negative = source.detach().clone()
        positive.reshape(-1)[flat_index] += step
        negative.reshape(-1)[flat_index] -= step
        if depth_source:
            return (
                objective(positive, intrinsics.detach()) - objective(negative, intrinsics.detach())
            ) / (2.0 * step)
        return (objective(depth.detach(), positive) - objective(depth.detach(), negative)) / (
            2.0 * step
        )

    depth_fd = central_difference(depth, depth_flat_index, 1.0e-5, depth_source=True)
    intrinsics_fd = central_difference(
        intrinsics,
        intrinsics_flat_index,
        1.0e-5,
        depth_source=False,
    )
    torch.testing.assert_close(
        depth_gradient.reshape(-1)[depth_flat_index],
        depth_fd,
        atol=1.0e-6,
        rtol=2.0e-4,
    )
    torch.testing.assert_close(
        intrinsics_gradient.reshape(-1)[intrinsics_flat_index],
        intrinsics_fd,
        atol=1.0e-7,
        rtol=2.0e-4,
    )


def test_unknown_metric_radius_retains_rgb_depth_and_intrinsics_gradients() -> None:
    image, depth, camera, _ = _render_variable_radius()
    image = image.requires_grad_(True)
    depth = depth.requires_grad_(True)
    intrinsics = camera.intrinsics.unsqueeze(0).clone().requires_grad_(True)

    measured = two_disc_geometry_from_rgbd(
        image,
        depth,
        0.21,
        camera.world_from_camera.unsqueeze(0),
        intrinsics,
        estimate_world_radius=True,
        minimum_world_radius=0.10,
        maximum_world_radius=0.35,
    )
    gradients = torch.autograd.grad(
        measured.surface_fit_radius.sum(),
        (image, depth, intrinsics),
    )

    assert measured.valid_mask.all()
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_unknown_metric_radius_outside_declared_bounds_fails_closed() -> None:
    image, depth, camera, _ = _render_variable_radius()

    measured = two_disc_geometry_from_rgbd(
        image,
        depth,
        0.21,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
        estimate_world_radius=True,
        minimum_world_radius=0.25,
        maximum_world_radius=0.35,
    )

    assert not measured.pair_valid_mask.any()
    assert torch.equal(measured.world_position, torch.zeros_like(measured.world_position))
    assert torch.equal(
        measured.surface_fit_radius,
        torch.zeros_like(measured.surface_fit_radius),
    )


@pytest.mark.parametrize(
    ("positions", "colours"),
    [
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
    ],
)
def test_spatial_refinement_generalises_across_depth_size_and_palette(
    positions: tuple[tuple[float, float, float], tuple[float, float, float]],
    colours: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> None:
    rendered, measured, state = _render_custom(positions, colours, radius=0.21)
    error = _best_set_error(measured.world_position[0], state.position)

    assert rendered.visible_fraction.tolist() == [1.0, 1.0]
    assert not bool((rendered.full_mask[0] & rendered.full_mask[1]).any())
    assert measured.pair_valid_mask.tolist() == [True]
    assert float(error.max()) < 0.008
    assert float(torch.sqrt(error.square().mean())) < 0.005

    _, swapped, _ = _render_custom(positions, tuple(reversed(colours)), radius=0.21)
    pairwise = torch.cdist(measured.world_position[0], swapped.world_position[0])
    assert swapped.pair_valid_mask.tolist() == [True]
    assert float(pairwise.amin(dim=-1).max()) < PALETTE_SET_INVARIANCE_M


def test_two_disc_output_is_a_set_under_palette_swap() -> None:
    first_image, first_depth, camera, _ = _render()
    second_image, second_depth, _, _ = _render(((0.18, 0.82, 0.90), (0.90, 0.20, 0.18)))

    first = _measurement(first_image, first_depth, camera)
    second = _measurement(second_image, second_depth, camera)
    pairwise = torch.cdist(first.world_position[0], second.world_position[0])

    assert first.pair_valid_mask.all() and second.pair_valid_mask.all()
    # Palette changes alter the continuous soft foreground evidence slightly,
    # but cannot materially move the unordered physical set.
    assert float(pairwise.amin(dim=-1).max()) < PALETTE_SET_INVARIANCE_M


def test_two_disc_geometry_retains_rgb_and_depth_gradients() -> None:
    image, depth, camera, _ = _render()
    image = image.requires_grad_(True)
    depth = depth.requires_grad_(True)

    measured = _measurement(image, depth, camera)
    world_objective = measured.world_position.square().sum()
    appearance_weights = image.new_tensor([0.37, -0.23, 0.51])
    appearance_objective = (measured.appearance * appearance_weights).sum()
    rgb_world_gradient = torch.autograd.grad(
        world_objective,
        image,
        retain_graph=True,
    )[0]
    rgb_appearance_gradient, depth_gradient = torch.autograd.grad(
        world_objective + appearance_objective,
        (image, depth),
    )

    assert measured.pair_valid_mask.all()
    for gradient in (rgb_world_gradient, rgb_appearance_gradient, depth_gradient):
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def _double_objective(
    image: torch.Tensor,
    depth: torch.Tensor,
    radius: torch.Tensor,
    camera: CameraFrame,
) -> torch.Tensor:
    measured = two_disc_geometry_from_rgbd(
        image,
        depth,
        radius,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
    )
    assert measured.pair_valid_mask.tolist() == [True]
    # Squared set norm is invariant to the arbitrary spectral slot sign.
    return measured.world_position.square().sum()


def _central_difference(
    positive: torch.Tensor,
    negative: torch.Tensor,
    step: float,
) -> torch.Tensor:
    return (positive - negative) / (2.0 * step)


def _relative_derivative_error(analytic: torch.Tensor, finite_difference: torch.Tensor) -> float:
    scale = analytic.abs() + finite_difference.abs() + analytic.new_tensor(1.0e-12)
    return float((analytic - finite_difference).abs() / scale)


def test_two_disc_metric_state_rgb_depth_and_radius_gradients_match_central_fd() -> None:
    image, depth, _, _ = _render()
    camera = _camera(torch.float64)
    image = image.to(torch.float64).requires_grad_(True)
    depth = depth.to(torch.float64).requires_grad_(True)
    radius = torch.tensor(RADIUS_M, dtype=torch.float64, requires_grad=True)
    objective = _double_objective(image, depth, radius, camera)
    image_gradient, depth_gradient, radius_gradient = torch.autograd.grad(
        objective,
        (image, depth, radius),
    )

    rgb_step = 1.0e-4
    rgb_admissible = (image.detach() > 10.0 * rgb_step) & (image.detach() < 1.0 - 10.0 * rgb_step)
    rgb_scores = torch.where(
        rgb_admissible,
        image_gradient.abs(),
        torch.full_like(image_gradient, -1.0),
    )
    rgb_index = tuple(torch.unravel_index(rgb_scores.argmax(), image.shape))
    rgb_positive = image.detach().clone()
    rgb_negative = image.detach().clone()
    rgb_positive[rgb_index] += rgb_step
    rgb_negative[rgb_index] -= rgb_step
    rgb_fd = _central_difference(
        _double_objective(rgb_positive, depth.detach(), radius.detach(), camera),
        _double_objective(rgb_negative, depth.detach(), radius.detach(), camera),
        rgb_step,
    )
    assert float(image_gradient[rgb_index].abs()) > 1.0e-8
    assert _relative_derivative_error(image_gradient[rgb_index], rgb_fd) < 5.0e-3

    depth_step = 1.0e-5
    depth_admissible = depth.detach() > 10.0 * depth_step
    depth_scores = torch.where(
        depth_admissible,
        depth_gradient.abs(),
        torch.full_like(depth_gradient, -1.0),
    )
    depth_index = tuple(torch.unravel_index(depth_scores.argmax(), depth.shape))
    depth_positive = depth.detach().clone()
    depth_negative = depth.detach().clone()
    depth_positive[depth_index] += depth_step
    depth_negative[depth_index] -= depth_step
    depth_fd = _central_difference(
        _double_objective(image.detach(), depth_positive, radius.detach(), camera),
        _double_objective(image.detach(), depth_negative, radius.detach(), camera),
        depth_step,
    )
    assert _relative_derivative_error(depth_gradient[depth_index], depth_fd) < 1.0e-4

    radius_step = 1.0e-5
    radius_fd = _central_difference(
        _double_objective(
            image.detach(),
            depth.detach(),
            radius.detach() + radius_step,
            camera,
        ),
        _double_objective(
            image.detach(),
            depth.detach(),
            radius.detach() - radius_step,
            camera,
        ),
        radius_step,
    )
    assert _relative_derivative_error(radius_gradient, radius_fd) < 1.0e-4


def test_degenerate_colour_pair_fails_closed_with_finite_zero_gradients() -> None:
    image, depth, camera, _ = _render(((0.75, 0.25, 0.20), (0.75, 0.25, 0.20)))
    image = image.requires_grad_(True)
    depth = depth.requires_grad_(True)

    measured = _measurement(image, depth, camera)
    gradients = torch.autograd.grad(
        measured.world_position.sum() + measured.appearance.sum() + measured.slot_logits.sum(),
        (image, depth),
        allow_unused=True,
    )

    assert not measured.pair_valid_mask.any()
    assert not measured.valid_mask.any()
    assert torch.equal(measured.world_position, torch.zeros_like(measured.world_position))
    assert torch.equal(measured.slot_logits, torch.zeros_like(measured.slot_logits))
    for source, gradient in zip((image, depth), gradients, strict=True):
        resolved = torch.zeros_like(source) if gradient is None else gradient
        assert torch.isfinite(resolved).all()
        assert torch.equal(resolved, torch.zeros_like(resolved))


def test_fixed_radius_prior_owns_the_continuous_surface_fit() -> None:
    image, depth, camera, _ = _render()
    radius = torch.tensor(RADIUS_M, requires_grad=True)

    measured = two_disc_geometry_from_rgbd(
        image,
        depth,
        radius,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
    )
    (gradient,) = torch.autograd.grad(measured.world_position.sum(), (radius,))

    assert measured.pair_valid_mask.all()
    assert torch.isfinite(gradient)
    assert float(gradient.abs()) > 0.0


def test_extreme_finite_depth_fails_closed_before_linear_algebra() -> None:
    image, depth, camera, _ = _render()
    depth = torch.full_like(depth, torch.finfo(depth.dtype).max).requires_grad_(True)

    measured = _measurement(image, depth, camera)
    (gradient,) = torch.autograd.grad(
        measured.world_position.sum(),
        (depth,),
        allow_unused=True,
    )

    assert not measured.pair_valid_mask.any()
    assert torch.equal(measured.world_position, torch.zeros_like(measured.world_position))
    resolved = torch.zeros_like(depth) if gradient is None else gradient
    assert torch.isfinite(resolved).all()
    assert torch.equal(resolved, torch.zeros_like(resolved))


def test_observably_overlapping_pair_fails_closed_before_state_publication() -> None:
    camera = _camera()
    state = replace(
        _state(((0.90, 0.20, 0.18), (0.18, 0.82, 0.90))),
        position=torch.tensor([[-0.08, 0.0, 4.0], [0.08, 0.0, 4.2]]),
    )
    rendered = render_spheres(state, camera, IMAGE_SIZE)
    image = rendered.rgb.unsqueeze(0).requires_grad_(True)
    depth = rendered.depth_buffer[None, None].requires_grad_(True)

    measured = _measurement(image, depth, camera)
    gradients = torch.autograd.grad(
        measured.world_position.sum() + measured.slot_logits.sum(),
        (image, depth),
        allow_unused=True,
    )

    assert bool((rendered.full_mask[0] & rendered.full_mask[1]).any())
    assert float(rendered.visible_fraction.min()) < 1.0
    assert float(measured.silhouette_gap_pixels[0].detach()) < 0.0
    assert not measured.pair_valid_mask.any()
    assert torch.equal(measured.world_position, torch.zeros_like(measured.world_position))
    for source, gradient in zip((image, depth), gradients, strict=True):
        resolved = torch.zeros_like(source) if gradient is None else gradient
        assert torch.isfinite(resolved).all()
        assert torch.equal(resolved, torch.zeros_like(resolved))
