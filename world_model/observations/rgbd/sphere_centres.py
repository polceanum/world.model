"""Differentiable metric sphere-centre measurements from observable RGB-D.

This is the structural measurement primitive used by the public composite
RGB-D observation modality: RGB supplies a subpixel image centre, the depth
image supplies metric surface camera-z, and declared sphere radius plus
calibration recover the metric centre.  No simulator state, renderer label,
instance map, or detached assignment enters the forward path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from world_model.observations.rgb.projector import camera_to_world
from world_model.observations.rgb.soft_geometry import (
    SoftDiscGeometryOutput,
    SoftPhotometricRadiusOutput,
    soft_disc_geometry_from_rgb,
    soft_photometric_disc_radius,
)

MINIMUM_FOCAL_LENGTH_PIXELS = 1.0e-3
MAXIMUM_METRIC_DISTANCE_M = 1.0e4
MAXIMUM_CALIBRATION_MAGNITUDE = 1.0e6


@dataclass(frozen=True)
class MetricSphereCentreOutput:
    """Metric centre recovered from subpixel centres and surface depth."""

    world_position: Tensor
    camera_position: Tensor
    surface_depth: Tensor
    centre_depth: Tensor
    ray_xy: Tensor
    depth_support: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class RGBDSphereCentreMeasurement:
    """Observable RGB-D sphere measurement and its differentiable evidence."""

    world_position: Tensor
    camera_position: Tensor
    centres: Tensor
    surface_depth: Tensor
    centre_depth: Tensor
    confidence: Tensor
    valid_mask: Tensor
    geometry: SoftDiscGeometryOutput
    photometric_geometry: SoftPhotometricRadiusOutput


def _broadcast_radius(
    world_radius: Tensor | float,
    *,
    batch: int,
    slots: int,
    reference: Tensor,
) -> Tensor:
    if isinstance(world_radius, Tensor):
        if not world_radius.is_floating_point():
            raise TypeError("world_radius must be floating point")
        if world_radius.dtype not in {torch.float32, torch.float64}:
            raise TypeError("RGB-D metric geometry supports only float32 and float64")
        if world_radius.dtype != reference.dtype:
            raise TypeError("world_radius and centres must use the same dtype")
        if world_radius.device != reference.device:
            raise ValueError("world_radius and centres must be on the same device")
    elif isinstance(world_radius, bool) or not isinstance(world_radius, (int, float)):
        raise TypeError("world_radius must be a floating-point tensor or real scalar")
    radius = torch.as_tensor(world_radius, dtype=reference.dtype, device=reference.device)
    if radius.ndim > 0 and radius.shape[-1:] == (1,):
        radius = radius.squeeze(-1)
    if radius.ndim == 0:
        return radius.expand(batch, slots)
    if radius.shape == (batch,) and slots == 1:
        return radius.unsqueeze(-1)
    try:
        return torch.broadcast_to(radius, (batch, slots))
    except RuntimeError as error:
        raise ValueError("world_radius must be broadcastable to [B,S]") from error


def _validate_metric_inputs(
    centres: Tensor,
    depth: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
) -> tuple[int, int, int, int]:
    if centres.ndim != 3 or centres.shape[-1] != 2:
        raise ValueError("centres must have shape [B,S,2] in [-1,1] image coordinates")
    if not centres.is_floating_point():
        raise TypeError("centres must be floating point")
    if centres.dtype not in {torch.float32, torch.float64}:
        raise TypeError("RGB-D metric geometry supports only float32 and float64")
    batch, slots = centres.shape[:2]
    if depth.ndim != 4 or depth.shape[:2] != (batch, 1):
        raise ValueError("depth must have shape [B,1,H,W]")
    if not depth.is_floating_point():
        raise TypeError("depth must be floating point")
    if depth.dtype != centres.dtype:
        raise TypeError("centres and depth must use the same dtype")
    if depth.device != centres.device:
        raise ValueError("centres and depth must be on the same device")
    height, width = depth.shape[-2:]
    if height < 2 or width < 2:
        raise ValueError("depth image dimensions must be at least two pixels")
    if world_from_camera.shape != (batch, 4, 4):
        raise ValueError("world_from_camera must have shape [B,4,4]")
    if intrinsics.shape != (batch, 3, 3):
        raise ValueError("intrinsics must have shape [B,3,3]")
    if world_from_camera.device != centres.device or intrinsics.device != centres.device:
        raise ValueError("centres, depth, and calibration must be on the same device")
    if not world_from_camera.is_floating_point() or not intrinsics.is_floating_point():
        raise TypeError("camera calibration must be floating point")
    if world_from_camera.dtype not in {torch.float32, torch.float64} or intrinsics.dtype not in {
        torch.float32,
        torch.float64,
    }:
        raise TypeError("RGB-D metric geometry supports only float32 and float64")
    if world_from_camera.dtype != centres.dtype or intrinsics.dtype != centres.dtype:
        raise TypeError("centres, depth, and calibration must use the same dtype")
    return batch, slots, height, width


def metric_sphere_centres_from_surface_depth(
    centres: Tensor,
    depth: Tensor,
    world_radius: Tensor | float,
    world_from_camera: Tensor,
    intrinsics: Tensor,
) -> MetricSphereCentreOutput:
    """Recover metric sphere centres from image centres and camera-z depth.

    ``depth`` stores the camera-z coordinate of the visible surface, with zero
    denoting no return.  Sampling uses native differentiable bilinear
    interpolation with ``align_corners=True``, matching the project's
    ``[-1,1]`` centre convention.  For the ray
    ``(qx, qy, 1)`` through the sphere centre, the centre camera-z is

    ``surface_z + radius / sqrt(1 + qx**2 + qy**2)``.

    A measurement is valid only when every bilinear contributor is finite and
    positive and the radius/calibration/centre are finite and physically
    admissible.  Invalid rows return finite zero metric values behind an
    explicit mask; valid rows retain ordinary gradients to centres, depth,
    radius, and calibration.
    """

    batch, slots, height, width = _validate_metric_inputs(
        centres,
        depth,
        world_from_camera,
        intrinsics,
    )
    radius = _broadcast_radius(
        world_radius,
        batch=batch,
        slots=slots,
        reference=centres,
    )

    centre_admissible = torch.isfinite(centres).all(dim=-1) & (centres.abs() <= 1.0).all(dim=-1)
    safe_centres = torch.where(
        centre_admissible.unsqueeze(-1),
        centres,
        torch.zeros_like(centres),
    )
    depth_valid_pixels = (
        torch.isfinite(depth) & (depth > 0.0) & (depth <= MAXIMUM_METRIC_DISTANCE_M)
    )
    safe_depth = torch.where(depth_valid_pixels, depth, torch.zeros_like(depth))
    sample_grid = safe_centres.unsqueeze(-2)
    sampled_depth = F.grid_sample(
        safe_depth.to(dtype=centres.dtype),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, :, 0]
    depth_support = F.grid_sample(
        depth_valid_pixels.to(dtype=centres.dtype),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, :, 0]

    intrinsics_typed = intrinsics.to(dtype=centres.dtype)
    world_from_camera_typed = world_from_camera.to(dtype=centres.dtype)
    identity_intrinsics = torch.eye(
        3,
        dtype=centres.dtype,
        device=centres.device,
    ).expand(batch, -1, -1)
    identity_transform = torch.eye(
        4,
        dtype=centres.dtype,
        device=centres.device,
    ).expand(batch, -1, -1)
    intrinsics_bounded = torch.isfinite(intrinsics_typed).all(dim=(-2, -1)) & (
        intrinsics_typed.abs().amax(dim=(-2, -1)) <= MAXIMUM_CALIBRATION_MAGNITUDE
    )
    transform_bounded = torch.isfinite(world_from_camera_typed).all(dim=(-2, -1)) & (
        world_from_camera_typed.abs().amax(dim=(-2, -1)) <= MAXIMUM_CALIBRATION_MAGNITUDE
    )
    bounded_intrinsics = torch.where(
        intrinsics_bounded[:, None, None],
        intrinsics_typed,
        identity_intrinsics,
    )
    bounded_transform = torch.where(
        transform_bounded[:, None, None],
        world_from_camera_typed,
        identity_transform,
    )
    fx = bounded_intrinsics[:, 0, 0]
    fy = bounded_intrinsics[:, 1, 1]
    rotation = bounded_transform[:, :3, :3]
    identity_rotation = torch.eye(
        3,
        dtype=centres.dtype,
        device=centres.device,
    ).expand(batch, -1, -1)
    homogeneous_row = bounded_transform.new_tensor([0.0, 0.0, 0.0, 1.0])
    calibration_tolerance = max(2.0e-5, 64.0 * torch.finfo(centres.dtype).eps)
    canonical_intrinsics = (bounded_intrinsics[:, 0, 1].abs() <= calibration_tolerance) & (
        bounded_intrinsics[:, 1, 0].abs() <= calibration_tolerance
    )
    canonical_intrinsics &= (
        bounded_intrinsics[:, 2] - bounded_intrinsics.new_tensor([0.0, 0.0, 1.0])
    ).abs().amax(dim=-1) <= calibration_tolerance
    orthonormal_rotation = (rotation.transpose(-1, -2) @ rotation - identity_rotation).abs().amax(
        dim=(-2, -1)
    ) <= calibration_tolerance
    right_handed_rotation = (
        torch.linalg.cross(rotation[:, :, 0], rotation[:, :, 1], dim=-1) * rotation[:, :, 2]
    ).sum(dim=-1) > 0.0
    homogeneous_transform = (bounded_transform[:, 3] - homogeneous_row).abs().amax(
        dim=-1
    ) <= calibration_tolerance
    calibration_valid = (
        intrinsics_bounded
        & transform_bounded
        & (fx >= MINIMUM_FOCAL_LENGTH_PIXELS)
        & (fy >= MINIMUM_FOCAL_LENGTH_PIXELS)
        & canonical_intrinsics
        & orthonormal_rotation
        & right_handed_rotation
        & homogeneous_transform
    )
    safe_intrinsics = torch.where(
        calibration_valid[:, None, None],
        bounded_intrinsics,
        identity_intrinsics,
    )
    safe_world_from_camera = torch.where(
        calibration_valid[:, None, None],
        bounded_transform,
        identity_transform,
    )
    safe_fx = safe_intrinsics[:, 0, 0]
    safe_fy = safe_intrinsics[:, 1, 1]
    safe_cx = safe_intrinsics[:, 0, 2]
    safe_cy = safe_intrinsics[:, 1, 2]

    pixel_x = 0.5 * (safe_centres[..., 0] + 1.0) * (width - 1)
    pixel_y = 0.5 * (safe_centres[..., 1] + 1.0) * (height - 1)
    ray_x = (pixel_x - safe_cx[:, None]) / safe_fx[:, None]
    ray_y = (pixel_y - safe_cy[:, None]) / safe_fy[:, None]
    ray_xy = torch.stack((ray_x, ray_y), dim=-1)
    ray_norm = torch.sqrt(1.0 + ray_x.square() + ray_y.square())

    radius_valid = torch.isfinite(radius) & (radius > 0.0) & (radius <= MAXIMUM_METRIC_DISTANCE_M)
    safe_radius = torch.where(radius_valid, radius, torch.zeros_like(radius))
    centre_depth = sampled_depth + safe_radius / ray_norm
    camera_position = torch.stack(
        (ray_x * centre_depth, ray_y * centre_depth, centre_depth),
        dim=-1,
    )
    world_position = camera_to_world(camera_position, safe_world_from_camera)

    support_tolerance = 8.0 * torch.finfo(centres.dtype).eps
    valid_mask = (
        centre_admissible
        & radius_valid
        & calibration_valid[:, None]
        & (depth_support >= 1.0 - support_tolerance)
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & torch.isfinite(world_position).all(dim=-1)
    )
    valid = valid_mask.unsqueeze(-1)
    return MetricSphereCentreOutput(
        world_position=torch.where(valid, world_position, torch.zeros_like(world_position)),
        camera_position=torch.where(valid, camera_position, torch.zeros_like(camera_position)),
        surface_depth=torch.where(valid_mask, sampled_depth, torch.zeros_like(sampled_depth)),
        centre_depth=torch.where(valid_mask, centre_depth, torch.zeros_like(centre_depth)),
        ray_xy=torch.where(valid, ray_xy, torch.zeros_like(ray_xy)),
        depth_support=torch.where(valid_mask, depth_support, torch.zeros_like(depth_support)),
        valid_mask=valid_mask,
    )


class RGBDSphereCentreMeasurementModule(nn.Module):
    """Parameter-free RGB-to-centre plus metric-depth sphere measurement."""

    def __init__(
        self,
        *,
        foreground_threshold: float = 0.04,
        foreground_temperature: float = 0.01,
        minimum_mass: float = 4.0,
    ) -> None:
        super().__init__()
        controls = {
            "foreground_threshold": foreground_threshold,
            "foreground_temperature": foreground_temperature,
            "minimum_mass": minimum_mass,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in controls.values()
        ):
            raise ValueError("RGB-D foreground controls must be finite and positive")
        self.foreground_threshold = float(foreground_threshold)
        self.foreground_temperature = float(foreground_temperature)
        self.minimum_mass = float(minimum_mass)

    def forward(
        self,
        image: Tensor,
        depth: Tensor,
        world_radius: Tensor | float,
        world_from_camera: Tensor,
        intrinsics: Tensor,
        slot_mask_logits: Tensor | None = None,
    ) -> RGBDSphereCentreMeasurement:
        """Measure sphere centres without state labels or simulator truth."""

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B,3,H,W]")
        if image.dtype not in {torch.float32, torch.float64}:
            raise TypeError("RGB-D sphere measurement supports only float32 and float64")
        if image.shape[0] != depth.shape[0] or image.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB and depth batch/image dimensions must match")
        geometry = soft_disc_geometry_from_rgb(
            image,
            slot_mask_logits,
            foreground_threshold=self.foreground_threshold,
            foreground_temperature=self.foreground_temperature,
            minimum_mass=self.minimum_mass,
        )
        photometric_geometry = soft_photometric_disc_radius(
            image,
            geometry.centres,
            geometry.radius_pixels,
        )
        metric = metric_sphere_centres_from_surface_depth(
            photometric_geometry.centres,
            depth,
            world_radius,
            world_from_camera,
            intrinsics,
        )
        # A finite positive depth return is not sufficient evidence that the
        # RGB foreground exists.  Keep the deployed centre fully continuous
        # on admissible rows, but fail closed when the image-moment foreground
        # mass does not reach the declared minimum.  Photometric convergence
        # diagnostics remain non-owning, as in the accepted monocular inverse.
        valid_mask = metric.valid_mask & geometry.valid_mask
        valid = valid_mask.unsqueeze(-1)
        confidence = (
            geometry.confidence * photometric_geometry.confidence * metric.depth_support
        ).clamp(0.0, 1.0)
        return RGBDSphereCentreMeasurement(
            world_position=torch.where(
                valid,
                metric.world_position,
                torch.zeros_like(metric.world_position),
            ),
            camera_position=torch.where(
                valid,
                metric.camera_position,
                torch.zeros_like(metric.camera_position),
            ),
            centres=photometric_geometry.centres,
            surface_depth=torch.where(
                valid_mask,
                metric.surface_depth,
                torch.zeros_like(metric.surface_depth),
            ),
            centre_depth=torch.where(
                valid_mask,
                metric.centre_depth,
                torch.zeros_like(metric.centre_depth),
            ),
            confidence=torch.where(valid_mask, confidence, torch.zeros_like(confidence)),
            valid_mask=valid_mask,
            geometry=geometry,
            photometric_geometry=photometric_geometry,
        )
