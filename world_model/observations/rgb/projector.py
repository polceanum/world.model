"""Structured RGB measurement projector and calibrated back-projection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from world_model.observations.context import SensorContext
from world_model.observations.measurements import PredictedMeasurements
from world_model.utils.transforms import invert_transform, transform_points

if TYPE_CHECKING:
    from world_model.belief.world_belief import WorldBelief


def _batch_calibration_tensor(
    value: Tensor | float | int | str,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    name: str,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"calibration {name!r} must be a Tensor")
    tensor = value.to(device=device, dtype=dtype)
    if tensor.shape == shape:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != (batch, *shape):
        raise ValueError(f"calibration {name!r} must be {shape} or [B,{','.join(map(str, shape))}]")
    return tensor


def calibration_tensors(
    calibration: Mapping[str, Tensor | float | int | str],
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    fallback_world_from_camera: Tensor | None = None,
    fallback_intrinsics: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    world_value = calibration.get("world_from_camera", fallback_world_from_camera)
    intrinsics_value = calibration.get("intrinsics", fallback_intrinsics)
    if world_value is None:
        raise ValueError("RGB calibration requires world_from_camera")
    if intrinsics_value is None:
        raise ValueError("RGB calibration requires intrinsics")
    world_from_camera = _batch_calibration_tensor(
        world_value,
        batch=batch,
        device=device,
        dtype=dtype,
        shape=(4, 4),
        name="world_from_camera",
    )
    intrinsics = _batch_calibration_tensor(
        intrinsics_value,
        batch=batch,
        device=device,
        dtype=dtype,
        shape=(3, 3),
        name="intrinsics",
    )
    return world_from_camera, intrinsics


def world_to_camera(world_position: Tensor, world_from_camera: Tensor) -> Tensor:
    camera_from_world = invert_transform(world_from_camera)
    return transform_points(camera_from_world[:, None], world_position)


def camera_to_world(camera_position: Tensor, world_from_camera: Tensor) -> Tensor:
    return transform_points(world_from_camera[:, None], camera_position)


def project_world_points(
    world_position: Tensor,
    world_radius: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return centre ``[-1,1]``, radius-normalized, invdepth, camera xyz."""

    height, width = image_size
    camera_position = world_to_camera(world_position, world_from_camera)
    depth = camera_position[..., 2].clamp_min(1.0e-4)
    fx = intrinsics[:, None, 0, 0]
    fy = intrinsics[:, None, 1, 1]
    cx = intrinsics[:, None, 0, 2]
    cy = intrinsics[:, None, 1, 2]
    pixel_x = fx * camera_position[..., 0] / depth + cx
    pixel_y = fy * camera_position[..., 1] / depth + cy
    normalised_x = 2.0 * pixel_x / max(width - 1, 1) - 1.0
    normalised_y = 2.0 * pixel_y / max(height - 1, 1) - 1.0
    focal = 0.5 * (fx + fy)
    radius_pixels = focal * world_radius.squeeze(-1).clamp_min(1.0e-4) / depth
    radius_normalised = radius_pixels / (0.5 * min(height, width))
    return (
        torch.stack((normalised_x, normalised_y), dim=-1),
        radius_normalised.clamp_min(1.0e-5),
        depth.reciprocal(),
        camera_position,
    )


def depth_ordered_circle_occlusion(
    centres: Tensor,
    radii: Tensor,
    depths: Tensor,
    projectable_mask: Tensor,
    *,
    depth_epsilon: float = 1.0e-4,
) -> tuple[Tensor, Tensor]:
    """Estimate visible fractions from depth-ordered projected circle overlap.

    Axes of the returned pairwise tensor are ``[batch, target, occluder]``.
    Multiple nearer occluders are combined with a bounded independent-union
    approximation. The overlap calculation is piecewise differentiable with
    respect to projected centres and radii away from visibility boundaries.
    """

    if centres.ndim != 3 or centres.shape[-1] != 2:
        raise ValueError("centres must have shape [B,N,2]")
    if radii.shape != centres.shape[:2] or depths.shape != centres.shape[:2]:
        raise ValueError("radii and depths must have shape [B,N]")
    if projectable_mask.shape != centres.shape[:2] or projectable_mask.dtype is not torch.bool:
        raise ValueError("projectable_mask must be bool [B,N]")
    if depth_epsilon <= 0:
        raise ValueError("depth_epsilon must be positive")

    target_centre = centres[:, :, None, :]
    occluder_centre = centres[:, None, :, :]
    distance = torch.linalg.vector_norm(target_centre - occluder_centre, dim=-1)
    safe_distance = distance.clamp_min(depth_epsilon)
    safe_radii = radii.clamp_min(depth_epsilon)
    target_radius = safe_radii[:, :, None]
    occluder_radius = safe_radii[:, None, :]
    target_area = math.pi * target_radius.square()
    occluder_area = math.pi * occluder_radius.square()

    cosine_target = (distance.square() + target_radius.square() - occluder_radius.square()) / (
        2.0 * safe_distance * target_radius
    )
    cosine_occluder = (distance.square() + occluder_radius.square() - target_radius.square()) / (
        2.0 * safe_distance * occluder_radius
    )
    angular_epsilon = 1.0e-6
    target_angle = torch.acos(cosine_target.clamp(-1.0 + angular_epsilon, 1.0 - angular_epsilon))
    occluder_angle = torch.acos(
        cosine_occluder.clamp(-1.0 + angular_epsilon, 1.0 - angular_epsilon)
    )
    radicand = (
        (-distance + target_radius + occluder_radius)
        * (distance + target_radius - occluder_radius)
        * (distance - target_radius + occluder_radius)
        * (distance + target_radius + occluder_radius)
    )
    partial_area = (
        target_radius.square() * target_angle
        + occluder_radius.square() * occluder_angle
        - 0.5 * radicand.clamp_min(depth_epsilon**4).sqrt()
    )
    no_overlap = distance >= target_radius + occluder_radius
    target_inside_occluder = distance + target_radius <= occluder_radius
    occluder_inside_target = distance + occluder_radius <= target_radius
    overlap_area = torch.where(
        no_overlap,
        torch.zeros_like(partial_area),
        torch.where(
            target_inside_occluder,
            target_area,
            torch.where(occluder_inside_target, occluder_area, partial_area),
        ),
    )
    overlap_fraction = (overlap_area / target_area.clamp_min(depth_epsilon**2)).clamp(
        0.0,
        1.0,
    )

    target_depth = depths[:, :, None]
    occluder_depth = depths[:, None, :]
    nearer = occluder_depth + depth_epsilon < target_depth
    valid_pair = projectable_mask[:, :, None] & projectable_mask[:, None, :] & nearer
    pairwise_occlusion = torch.where(
        valid_pair,
        overlap_fraction,
        torch.zeros_like(overlap_fraction),
    )
    visible_fraction = (1.0 - pairwise_occlusion).clamp(0.0, 1.0).prod(dim=-1)
    visible_fraction = torch.where(
        projectable_mask,
        visible_fraction,
        torch.zeros_like(visible_fraction),
    )
    return visible_fraction, pairwise_occlusion


def backproject_rgb_measurements(
    values: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
) -> Tensor:
    """Back-project ``[u,v,log-radius,invdepth,...]`` to world position."""

    height, width = image_size
    pixel_x = 0.5 * (values[..., 0] + 1.0) * max(width - 1, 1)
    pixel_y = 0.5 * (values[..., 1] + 1.0) * max(height - 1, 1)
    depth = values[..., 3].clamp_min(1.0e-4).reciprocal()
    fx = intrinsics[:, None, 0, 0]
    fy = intrinsics[:, None, 1, 1]
    cx = intrinsics[:, None, 0, 2]
    cy = intrinsics[:, None, 1, 2]
    camera_x = (pixel_x - cx) * depth / fx.clamp_min(1.0e-4)
    camera_y = (pixel_y - cy) * depth / fy.clamp_min(1.0e-4)
    camera_position = torch.stack((camera_x, camera_y, depth), dim=-1)
    return camera_to_world(camera_position, world_from_camera)


def backproject_rgb_log_variance(
    values: Tensor,
    measurement_log_variance: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    minimum_log_variance: float = -12.0,
    maximum_log_variance: float = 8.0,
) -> Tensor:
    """Propagate RGB uncertainty to diagonal world XYZ log variance.

    The first-order pinhole Jacobian includes normalized centre and inverse
    depth. Log-radius uncertainty is conservatively folded into inverse-depth
    uncertainty because global sphere depth is derived from apparent radius.
    Squared camera rotation coefficients convert the camera-frame diagonal
    approximation to a world-frame diagonal approximation.
    """

    if values.ndim != 3 or values.shape[-1] < 4:
        raise ValueError("RGB values must have shape [B,M,D>=4]")
    log_variance = torch.broadcast_to(measurement_log_variance, values.shape)
    height, width = image_size
    inverse_depth = values[..., 3].clamp_min(1.0e-4)
    depth = inverse_depth.reciprocal()
    pixel_x = 0.5 * (values[..., 0] + 1.0) * max(width - 1, 1)
    pixel_y = 0.5 * (values[..., 1] + 1.0) * max(height - 1, 1)
    fx = intrinsics[:, None, 0, 0].clamp_min(1.0e-4)
    fy = intrinsics[:, None, 1, 1].clamp_min(1.0e-4)
    cx = intrinsics[:, None, 0, 2]
    cy = intrinsics[:, None, 1, 2]

    variance_u = log_variance[..., 0].exp()
    variance_v = log_variance[..., 1].exp()
    variance_inverse_depth = log_variance[..., 3].exp()
    variance_inverse_depth = variance_inverse_depth + (
        inverse_depth.square() * log_variance[..., 2].exp()
    )
    dx_du = 0.5 * max(width - 1, 1) * depth / fx
    dy_dv = 0.5 * max(height - 1, 1) * depth / fy
    dx_dinvdepth = -(pixel_x - cx) / (fx * inverse_depth.square())
    dy_dinvdepth = -(pixel_y - cy) / (fy * inverse_depth.square())
    dz_dinvdepth = -inverse_depth.reciprocal().square()
    camera_variance = torch.stack(
        (
            dx_du.square() * variance_u + dx_dinvdepth.square() * variance_inverse_depth,
            dy_dv.square() * variance_v + dy_dinvdepth.square() * variance_inverse_depth,
            dz_dinvdepth.square() * variance_inverse_depth,
        ),
        dim=-1,
    )
    rotation = world_from_camera[:, :3, :3]
    world_variance = torch.einsum(
        "bij,bmj->bmi",
        rotation.square(),
        camera_variance,
    )
    minimum_variance = values.new_tensor(minimum_log_variance).exp()
    maximum_variance = values.new_tensor(maximum_log_variance).exp()
    return world_variance.clamp(
        min=minimum_variance,
        max=maximum_variance,
    ).log()


def project_world_position_log_variance(
    world_position_log_variance: Tensor,
    camera_position: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    output_dimensions: int,
    variance_floor: float = 1.0e-4,
    maximum_log_variance: float = 5.0,
) -> Tensor:
    """Propagate diagonal world XYZ uncertainty into RGB measurement units.

    Association compares normalized image centres, log projected radius, and
    inverse depth. Copying metre-squared state variance into those coordinates
    makes Mahalanobis gates depend on arbitrary units. This first-order pinhole
    Jacobian keeps every predicted covariance in the units of its measurement.
    Appearance dimensions have no kinematic covariance and receive the
    configured numerical floor.
    """

    if world_position_log_variance.shape != camera_position.shape:
        raise ValueError("world position log variance and camera position must both be [B,N,3]")
    if output_dimensions < 4:
        raise ValueError("RGB predicted measurement must contain at least four dimensions")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    world_variance = world_position_log_variance.exp()
    camera_from_world = invert_transform(world_from_camera)
    camera_rotation = camera_from_world[:, :3, :3]
    camera_variance = torch.einsum(
        "bij,bnj->bni",
        camera_rotation.square(),
        world_variance,
    )
    height, width = image_size
    depth = camera_position[..., 2].clamp_min(1.0e-4)
    camera_x = camera_position[..., 0]
    camera_y = camera_position[..., 1]
    fx = intrinsics[:, None, 0, 0]
    fy = intrinsics[:, None, 1, 1]
    x_scale = 2.0 * fx / max(width - 1, 1)
    y_scale = 2.0 * fy / max(height - 1, 1)
    inverse_depth = depth.reciprocal()
    inverse_depth_squared = inverse_depth.square()
    variance_x = camera_variance[..., 0]
    variance_y = camera_variance[..., 1]
    variance_z = camera_variance[..., 2]
    centre_x_variance = (x_scale * inverse_depth).square() * variance_x + (
        x_scale * camera_x * inverse_depth_squared
    ).square() * variance_z
    centre_y_variance = (y_scale * inverse_depth).square() * variance_y + (
        y_scale * camera_y * inverse_depth_squared
    ).square() * variance_z
    log_radius_variance = inverse_depth_squared * variance_z
    inverse_depth_variance = inverse_depth_squared.square() * variance_z
    geometric = torch.stack(
        (
            centre_x_variance,
            centre_y_variance,
            log_radius_variance,
            inverse_depth_variance,
        ),
        dim=-1,
    )
    if output_dimensions > 4:
        appearance = geometric.new_full(
            (*geometric.shape[:-1], output_dimensions - 4),
            variance_floor,
        )
        geometric = torch.cat((geometric, appearance), dim=-1)
    return geometric.clamp_min(variance_floor).log().clamp_max(maximum_log_variance)


@dataclass(frozen=True)
class RGBProjectorConfig:
    default_radius: float = 0.15
    uncertainty_roi_scale: float = 2.5
    minimum_roi_radius: float = 0.04
    maximum_roi_radius: float = 0.8
    measurement_variance_floor: float = 1.0e-4
    full_occlusion_visible_fraction: float = 0.05


class RGBMeasurementProjector(nn.Module):
    """Differentiable measurement-space projector, not an RGB renderer."""

    def __init__(self, config: RGBProjectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or RGBProjectorConfig()

    def forward(
        self,
        belief: WorldBelief,
        sensor_context: SensorContext,
    ) -> PredictedMeasurements:
        objects = belief.objects
        position = objects.position
        batch, object_count, _ = position.shape
        if sensor_context.image_size is None:
            raise ValueError("RGB projection requires SensorContext.image_size")
        fallback_world = getattr(belief.camera, "world_from_camera", None)
        fallback_intrinsics = getattr(belief.camera, "intrinsics", None)
        world_from_camera, intrinsics = calibration_tensors(
            sensor_context.calibration,
            batch=batch,
            device=position.device,
            dtype=position.dtype,
            fallback_world_from_camera=fallback_world,
            fallback_intrinsics=fallback_intrinsics,
        )
        if objects.geometry.shape[-1] > 0:
            radius = objects.geometry[..., :1].abs().clamp_min(0.02)
        else:
            radius = position.new_full((batch, object_count, 1), self.config.default_radius)
        centre, normalised_radius, inverse_depth, camera_position = project_world_points(
            position,
            radius,
            world_from_camera,
            intrinsics,
            sensor_context.image_size,
        )
        if objects.appearance.shape[-1] >= 3:
            colour = objects.appearance[..., :3].clamp(0.0, 1.0)
        else:
            colour = position.new_full((batch, object_count, 3), 0.5)
        values = torch.cat(
            (
                centre,
                normalised_radius.log().unsqueeze(-1),
                inverse_depth.unsqueeze(-1),
                colour,
            ),
            dim=-1,
        )
        if objects.fast_log_variance.shape[-1] >= 3:
            position_log_variance = objects.fast_log_variance[..., :3]
        else:
            position_log_variance = position.new_full(
                (batch, object_count, 3),
                math.log(self.config.measurement_variance_floor),
            )
        log_variance = project_world_position_log_variance(
            position_log_variance,
            camera_position,
            world_from_camera,
            intrinsics,
            sensor_context.image_size,
            output_dimensions=values.shape[-1],
            variance_floor=self.config.measurement_variance_floor,
        )
        # ROI coordinates are normalized image coordinates, so expand them
        # using projected centre uncertainty in those same units. A
        # metre-space standard deviation times inverse depth omits focal length
        # and silently under-covers tracks as camera intrinsics change.
        centre_standard_deviation = log_variance[..., :2].exp().sqrt().amax(dim=-1)
        roi_uncertainty = self.config.uncertainty_roi_scale * centre_standard_deviation
        roi_radius = (normalised_radius + roi_uncertainty).clamp(
            self.config.minimum_roi_radius,
            self.config.maximum_roi_radius,
        )
        rois = torch.cat(
            (
                centre - roi_radius.unsqueeze(-1),
                centre + roi_radius.unsqueeze(-1),
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        in_front = camera_position[..., 2] > 1.0e-4
        near_image = (centre.abs() <= 1.0 + roi_radius.unsqueeze(-1)).all(dim=-1)
        projectable = (
            in_front
            & near_image
            & torch.isfinite(centre).all(dim=-1)
            & torch.isfinite(inverse_depth)
        )
        height, width = sensor_context.image_size
        pixel_centres = torch.stack(
            (
                0.5 * (centre[..., 0] + 1.0) * max(width - 1, 1),
                0.5 * (centre[..., 1] + 1.0) * max(height - 1, 1),
            ),
            dim=-1,
        )
        pixel_radii = normalised_radius * (0.5 * min(height, width))
        geometric_visible_fraction, pairwise_occlusion = depth_ordered_circle_occlusion(
            pixel_centres,
            pixel_radii,
            camera_position[..., 2],
            objects.active & projectable,
        )
        occlusion_fraction = torch.where(
            objects.active & projectable,
            (1.0 - geometric_visible_fraction).clamp(0.0, 1.0),
            torch.zeros_like(geometric_visible_fraction),
        )
        fully_occluded = (
            objects.active
            & projectable
            & (geometric_visible_fraction <= self.config.full_occlusion_visible_fraction)
        )
        unobservable = objects.active & (~projectable | fully_occluded)
        valid_mask = objects.active & projectable & ~fully_occluded
        rois = torch.where(valid_mask.unsqueeze(-1), rois, torch.zeros_like(rois))
        expected_visibility = (
            objects.visibility_logit.sigmoid()
            * geometric_visible_fraction
            * (objects.active & projectable).to(position.dtype)
        )
        belief_indices = (
            torch.arange(object_count, device=position.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        predicted = PredictedMeasurements(
            modality="rgb",
            sensor_id=sensor_context.sensor_id,
            timestamp=belief.timestamp,
            values=values,
            log_variance=log_variance,
            object_ids=objects.object_id,
            belief_indices=belief_indices,
            valid_mask=valid_mask,
            visibility=expected_visibility,
            rois=rois,
            appearance=objects.appearance,
            auxiliary={
                "world_position": position,
                "camera_position": camera_position,
                "world_radius": radius,
                "world_from_camera": world_from_camera[:, None].expand(-1, object_count, -1, -1),
                "intrinsics": intrinsics[:, None].expand(-1, object_count, -1, -1),
                "projectable_mask": objects.active & projectable,
                "expected_visibility": expected_visibility,
                "visible_fraction": geometric_visible_fraction,
                "occlusion_fraction": occlusion_fraction,
                "occluded_mask": fully_occluded,
                "fully_occluded_mask": fully_occluded,
                "unobservable_mask": unobservable,
                "pairwise_occlusion_fraction": pairwise_occlusion,
            },
        )
        predicted.validate()
        return predicted

    project = forward
