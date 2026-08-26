"""Seed-free tests for differentiable observable RGB-D sphere geometry."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from world_model.observations.rgbd import (
    RGBDSphereCentreMeasurementModule,
    metric_sphere_centres_from_surface_depth,
)
from world_model.simulator import (
    CameraFrame,
    SphereState,
    make_intrinsics,
    render_spheres,
)

IMAGE_SIZE = (64, 80)
WORLD_RADIUS_M = 0.3


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


def _sphere(
    position: tuple[float, float, float], albedo: tuple[float, float, float]
) -> SphereState:
    return _spheres((position,), (WORLD_RADIUS_M,), (albedo,))


def _spheres(
    positions: tuple[tuple[float, float, float], ...],
    radii: tuple[float, ...],
    albedos: tuple[tuple[float, float, float], ...],
) -> SphereState:
    count = len(positions)
    if len(radii) != count or len(albedos) != count:
        raise ValueError("sphere test fields must have equal lengths")
    return SphereState(
        object_id=torch.arange(count, dtype=torch.int64),
        active=torch.ones(count, dtype=torch.bool),
        position=torch.tensor(positions, dtype=torch.float32),
        velocity=torch.zeros((count, 3)),
        radius=torch.tensor(radii, dtype=torch.float32).unsqueeze(-1),
        mass=torch.ones((count, 1)),
        restitution=torch.zeros((count, 1)),
        drag=torch.zeros((count, 1)),
        friction=torch.zeros((count, 1)),
        albedo=torch.tensor(albedos, dtype=torch.float32),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(count, -1).clone(),
        angular_velocity=torch.zeros((count, 3)),
        sleeping=torch.zeros(count, dtype=torch.bool),
        sleep_counter=torch.zeros(count, dtype=torch.int64),
    )


def _cartesian_renderer_grid() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    camera = _camera()
    positions = [
        (x, y, z) for z in (3.2, 4.8, 6.0) for x in (-0.8, 0.0, 0.8) for y in (-0.45, 0.45)
    ]
    albedos = (
        (0.82, 0.23, 0.14),
        (0.18, 0.76, 0.31),
        (0.20, 0.38, 0.91),
    )
    rendered = [
        render_spheres(
            _sphere(position, albedos[index % len(albedos)]),
            camera,
            IMAGE_SIZE,
        )
        for index, position in enumerate(positions)
    ]
    batch = len(positions)
    return (
        torch.stack([output.rgb for output in rendered]),
        torch.stack([output.depth_buffer for output in rendered]).unsqueeze(1),
        torch.full((batch, 1), WORLD_RADIUS_M),
        camera.world_from_camera.expand(batch, -1, -1).clone(),
        camera.intrinsics.expand(batch, -1, -1).clone(),
        torch.tensor(positions, dtype=torch.float32).unsqueeze(1),
    )


def test_public_renderer_depth_is_exact_camera_z_at_the_centre_ray() -> None:
    camera = _camera()
    intrinsics = camera.intrinsics
    pixel_x = 24
    pixel_y = 27
    centre_z = 4.0
    ray_x = (pixel_x - float(intrinsics[0, 2])) / float(intrinsics[0, 0])
    ray_y = (pixel_y - float(intrinsics[1, 2])) / float(intrinsics[1, 1])
    centre = (ray_x * centre_z, ray_y * centre_z, centre_z)

    output = render_spheres(
        _sphere(centre, (0.82, 0.23, 0.14)),
        camera,
        IMAGE_SIZE,
    )

    expected_surface_z = centre_z - WORLD_RADIUS_M / (1.0 + ray_x * ray_x + ray_y * ray_y) ** 0.5
    assert output.depth_buffer.dtype == torch.float32
    assert torch.isfinite(output.depth_buffer).all()
    assert float(output.depth_buffer[pixel_y, pixel_x]) == pytest.approx(
        expected_surface_z,
        abs=2.0e-6,
    )
    assert output.depth_buffer[0, 0] == 0.0


def test_depth_instance_and_visibility_share_exact_metric_ordering() -> None:
    camera = _camera()
    intrinsics = camera.intrinsics
    pixel_x = 70
    pixel_y = 32
    ray_x = (pixel_x - float(intrinsics[0, 2])) / float(intrinsics[0, 0])
    ray_y = (pixel_y - float(intrinsics[1, 2])) / float(intrinsics[1, 1])
    ray_norm = (1.0 + ray_x * ray_x + ray_y * ray_y) ** 0.5
    near_radius = 0.2
    far_radius = 0.5
    radius_difference = far_radius - near_radius
    near_z = 4.0
    # The old z-radius approximation picks the larger, farther sphere; exact
    # perspective depth picks the smaller, nearer sphere.  This makes the test
    # sensitive to which ordering owns depth, instances, and RGB visibility.
    far_z = near_z + 0.5 * (radius_difference + radius_difference / ray_norm)
    state = _spheres(
        (
            (ray_x * far_z, ray_y * far_z, far_z),
            (ray_x * near_z, ray_y * near_z, near_z),
        ),
        (far_radius, near_radius),
        ((0.82, 0.23, 0.14), (0.18, 0.76, 0.31)),
    )

    output = render_spheres(state, camera, IMAGE_SIZE)

    expected_near_surface = near_z - near_radius / ray_norm
    assert output.full_mask[:, pixel_y, pixel_x].all()
    assert int(output.instance_slot_map[pixel_y, pixel_x]) == 1
    assert int(output.instance_map[pixel_y, pixel_x]) == 1
    assert not bool(output.visible_mask[0, pixel_y, pixel_x])
    assert bool(output.visible_mask[1, pixel_y, pixel_x])
    assert float(output.depth_buffer[pixel_y, pixel_x]) == pytest.approx(
        expected_near_surface,
        abs=2.0e-6,
    )


@pytest.mark.parametrize("centre_x", [0.8, 1.2, 1.5])
def test_exact_off_axis_silhouette_is_not_clipped_by_projected_disc(
    centre_x: float,
) -> None:
    camera = _camera()
    centre = torch.tensor([centre_x, 0.0, 3.2], dtype=torch.float32)
    output = render_spheres(
        _sphere(tuple(float(value) for value in centre), (0.82, 0.23, 0.14)),
        camera,
        IMAGE_SIZE,
    )

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(IMAGE_SIZE[0], dtype=torch.float32),
        torch.arange(IMAGE_SIZE[1], dtype=torch.float32),
        indexing="ij",
    )
    ray_x = (pixel_x - camera.intrinsics[0, 2]) / camera.intrinsics[0, 0]
    ray_y = (pixel_y - camera.intrinsics[1, 2]) / camera.intrinsics[1, 1]
    ray_norm_squared = 1.0 + ray_x.square() + ray_y.square()
    ray_dot_centre = ray_x * centre[0] + ray_y * centre[1] + centre[2]
    discriminant = ray_dot_centre.square() - ray_norm_squared * (
        centre.square().sum() - WORLD_RADIUS_M**2
    )
    expected_hits = discriminant >= 0.0

    assert torch.equal(output.full_mask[0], expected_hits)
    assert output.depth_buffer[expected_hits].gt(0.0).all()
    assert output.soft_support[0, expected_hits].gt(0.0).all()


def test_rgbd_measurement_is_accurate_on_seed_free_cartesian_renderer_grid() -> None:
    image, depth, radius, world_from_camera, intrinsics, expected_position = (
        _cartesian_renderer_grid()
    )

    measurement = RGBDSphereCentreMeasurementModule()(
        image,
        depth,
        radius,
        world_from_camera,
        intrinsics,
    )

    error = torch.linalg.vector_norm(measurement.world_position - expected_position, dim=-1)
    assert measurement.valid_mask.all()
    assert torch.isfinite(measurement.world_position).all()
    assert float(error.max()) < 0.007
    assert float(torch.sqrt(error.square().mean())) < 0.004


def test_metric_sampling_preserves_centre_and_depth_gradients() -> None:
    image, depth, radius, world_from_camera, intrinsics, _ = _cartesian_renderer_grid()
    rgb_measurement = RGBDSphereCentreMeasurementModule()(
        image[:1],
        depth[:1],
        radius[:1],
        world_from_camera[:1],
        intrinsics[:1],
    )
    centres = rgb_measurement.centres.detach().requires_grad_(True)
    differentiable_depth = depth[:1].clone().requires_grad_(True)

    metric = metric_sphere_centres_from_surface_depth(
        centres,
        differentiable_depth,
        radius[:1],
        world_from_camera[:1],
        intrinsics[:1],
    )
    loss = metric.world_position.square().sum()
    loss.backward()

    assert metric.valid_mask.all()
    assert centres.grad is not None
    assert torch.isfinite(centres.grad).all()
    assert float(centres.grad.abs().sum()) > 0.0
    assert differentiable_depth.grad is not None
    assert torch.isfinite(differentiable_depth.grad).all()
    assert float(differentiable_depth.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("invalid_kind", ["skew", "scaled_transform", "zero_radius"])
def test_metric_measurement_fails_closed_on_invalid_geometry_or_calibration(
    invalid_kind: str,
) -> None:
    image, depth, radius, world_from_camera, intrinsics, _ = _cartesian_renderer_grid()
    measurement = RGBDSphereCentreMeasurementModule()(
        image[:1], depth[:1], radius[:1], world_from_camera[:1], intrinsics[:1]
    )
    centres = measurement.centres.detach()
    test_radius = radius[:1].clone()
    test_world_from_camera = world_from_camera[:1].clone()
    test_intrinsics = intrinsics[:1].clone()
    if invalid_kind == "skew":
        test_intrinsics[0, 0, 1] = 0.1
    elif invalid_kind == "scaled_transform":
        test_world_from_camera[0, 0, 0] = 2.0
    else:
        test_radius.zero_()

    metric = metric_sphere_centres_from_surface_depth(
        centres,
        depth[:1],
        test_radius,
        test_world_from_camera,
        test_intrinsics,
    )

    assert not metric.valid_mask.any()
    assert torch.equal(metric.world_position, torch.zeros_like(metric.world_position))
    assert torch.isfinite(metric.world_position).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metric_measurement_rejects_unsupported_low_precision(dtype: torch.dtype) -> None:
    centres = torch.zeros((1, 1, 2), dtype=dtype)
    depth = torch.ones((1, 1, 4, 4), dtype=dtype)

    with pytest.raises(TypeError, match="only float32 and float64"):
        metric_sphere_centres_from_surface_depth(
            centres,
            depth,
            0.3,
            torch.eye(4, dtype=dtype).unsqueeze(0),
            torch.eye(3, dtype=dtype).unsqueeze(0),
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("input_kind", ["radius", "world_from_camera", "intrinsics"])
def test_metric_measurement_rejects_low_precision_auxiliary_geometry(
    dtype: torch.dtype,
    input_kind: str,
) -> None:
    centres = torch.zeros((1, 1, 2), dtype=torch.float32)
    depth = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    radius = torch.full((1, 1), 0.3, dtype=torch.float32)
    world_from_camera = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    if input_kind == "radius":
        radius = radius.to(dtype)
    elif input_kind == "world_from_camera":
        world_from_camera = world_from_camera.to(dtype)
    else:
        intrinsics = intrinsics.to(dtype)

    with pytest.raises(TypeError, match="only float32 and float64"):
        metric_sphere_centres_from_surface_depth(
            centres,
            depth,
            radius,
            world_from_camera,
            intrinsics,
        )


@pytest.mark.parametrize(
    "invalid_kind",
    ["maximum_centre", "maximum_radius", "maximum_transform", "subnormal_focal", "maximum_depth"],
)
def test_extreme_invalid_metric_rows_have_zero_finite_outputs_and_gradients(
    invalid_kind: str,
) -> None:
    maximum = torch.finfo(torch.float32).max
    centres = torch.zeros((1, 1, 2), dtype=torch.float32)
    depth = torch.full((1, 1, 4, 4), 2.0, dtype=torch.float32)
    radius = torch.tensor([[0.3]], dtype=torch.float32)
    world_from_camera = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    intrinsics = torch.tensor(
        [[[10.0, 0.0, 1.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    if invalid_kind == "maximum_centre":
        centres.fill_(maximum)
    elif invalid_kind == "maximum_radius":
        radius.fill_(maximum)
    elif invalid_kind == "maximum_transform":
        world_from_camera[0, 0, 3] = maximum
    elif invalid_kind == "subnormal_focal":
        subnormal = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        intrinsics[0, 0, 0] = subnormal
        intrinsics[0, 1, 1] = subnormal
    else:
        depth.fill_(maximum)
    inputs = tuple(
        value.requires_grad_(True)
        for value in (centres, depth, radius, world_from_camera, intrinsics)
    )

    metric = metric_sphere_centres_from_surface_depth(*inputs)
    continuous_outputs = (
        metric.world_position,
        metric.camera_position,
        metric.surface_depth,
        metric.centre_depth,
        metric.ray_xy,
        metric.depth_support,
    )
    loss = sum(output.sum() for output in continuous_outputs)
    loss.backward()

    assert not metric.valid_mask.any()
    for output in continuous_outputs:
        assert torch.isfinite(output).all()
        assert not output.any()
    for value in inputs:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert not value.grad.any()


def test_rgbd_composite_preserves_gradients_to_rgb_and_depth() -> None:
    image, depth, radius, world_from_camera, intrinsics, _ = _cartesian_renderer_grid()
    differentiable_image = image[:1].clone().requires_grad_(True)
    differentiable_depth = depth[:1].clone().requires_grad_(True)

    measurement = RGBDSphereCentreMeasurementModule()(
        differentiable_image,
        differentiable_depth,
        radius[:1],
        world_from_camera[:1],
        intrinsics[:1],
    )
    measurement.centres.retain_grad()
    measurement.world_position.square().sum().backward()

    assert measurement.valid_mask.all()
    assert measurement.centres.grad is not None
    assert torch.isfinite(measurement.centres.grad).all()
    assert float(measurement.centres.grad.abs().sum()) > 0.0
    assert differentiable_image.grad is not None
    assert torch.isfinite(differentiable_image.grad).all()
    assert float(differentiable_image.grad.abs().sum()) > 0.0
    assert differentiable_depth.grad is not None
    assert torch.isfinite(differentiable_depth.grad).all()
    assert float(differentiable_depth.grad.abs().sum()) > 0.0


def test_extreme_invalid_composite_row_has_zero_finite_deployed_gradients() -> None:
    image, depth, radius, world_from_camera, intrinsics, _ = _cartesian_renderer_grid()
    maximum = torch.finfo(torch.float32).max
    test_image = image[:1].clone().requires_grad_(True)
    test_depth = depth[:1].clone()
    test_depth.fill_(maximum).requires_grad_(True)
    test_radius = radius[:1].clone()
    test_radius.fill_(maximum).requires_grad_(True)
    test_world_from_camera = world_from_camera[:1].clone()
    test_world_from_camera[0, 0, 3] = maximum
    test_world_from_camera.requires_grad_(True)
    test_intrinsics = intrinsics[:1].clone()
    subnormal = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    test_intrinsics[0, 0, 0] = subnormal
    test_intrinsics[0, 1, 1] = subnormal
    test_intrinsics.requires_grad_(True)
    slot_mask_logits = torch.zeros(
        (1, 1, *IMAGE_SIZE),
        dtype=torch.float32,
        requires_grad=True,
    )

    measurement = RGBDSphereCentreMeasurementModule()(
        test_image,
        test_depth,
        test_radius,
        test_world_from_camera,
        test_intrinsics,
        slot_mask_logits,
    )
    deployed_outputs = (
        measurement.world_position,
        measurement.camera_position,
        measurement.surface_depth,
        measurement.centre_depth,
        measurement.confidence,
    )
    loss = sum(output.sum() for output in deployed_outputs)
    loss.backward()

    assert not measurement.valid_mask.any()
    for output in deployed_outputs:
        assert torch.isfinite(output).all()
        assert not output.any()
    for value in (
        test_image,
        test_depth,
        test_radius,
        test_world_from_camera,
        test_intrinsics,
        slot_mask_logits,
    ):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert not value.grad.any()


@pytest.mark.parametrize("invalid_value", [0.0, -1.0, torch.nan, torch.inf])
def test_metric_depth_fails_closed_on_invalid_sampling_support(invalid_value: float) -> None:
    image, depth, radius, world_from_camera, intrinsics, _ = _cartesian_renderer_grid()
    measurement = RGBDSphereCentreMeasurementModule()(
        image[:1],
        depth[:1],
        radius[:1],
        world_from_camera[:1],
        intrinsics[:1],
    )
    centres = measurement.centres.detach()
    pixel_x = int(round(float(0.5 * (centres[0, 0, 0] + 1.0) * (IMAGE_SIZE[1] - 1))))
    pixel_y = int(round(float(0.5 * (centres[0, 0, 1] + 1.0) * (IMAGE_SIZE[0] - 1))))
    corrupted_depth = depth[:1].clone()
    corrupted_depth[0, 0, pixel_y - 1 : pixel_y + 2, pixel_x - 1 : pixel_x + 2] = invalid_value
    corrupted_depth.requires_grad_(True)

    metric = metric_sphere_centres_from_surface_depth(
        centres,
        corrupted_depth,
        radius[:1],
        world_from_camera[:1],
        intrinsics[:1],
    )

    assert not metric.valid_mask.any()
    assert torch.equal(metric.world_position, torch.zeros_like(metric.world_position))
    assert torch.equal(metric.camera_position, torch.zeros_like(metric.camera_position))
    assert torch.equal(metric.surface_depth, torch.zeros_like(metric.surface_depth))
    assert torch.equal(metric.centre_depth, torch.zeros_like(metric.centre_depth))
    assert torch.isfinite(metric.world_position).all()
    metric.world_position.sum().backward()
    assert corrupted_depth.grad is not None
    assert torch.isfinite(corrupted_depth.grad).all()
    assert not corrupted_depth.grad.any()
