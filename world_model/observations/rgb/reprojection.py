"""Differentiable RGB reprojection evidence for sphere beliefs."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from world_model.belief.world_belief import WorldBelief
from world_model.observations.packets import CalibrationValue
from world_model.observations.rgb.projector import calibration_tensors, project_world_points


def soft_sphere_silhouette_reprojection(
    belief: WorldBelief,
    image: Tensor,
    calibration: Mapping[str, CalibrationValue],
    *,
    foreground_threshold: float,
    edge_softness_pixels: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compare a projected sphere belief with foreground evidence from RGB.

    This is a training objective, not another physical state.  The target mask
    is derived only from the supplied RGB frame by the same row-background
    assumption as structured sphere discovery.  The prediction is a smooth
    union of analytic perspective-projected sphere silhouettes, so gradients
    reach continuous belief position and radius without differentiating
    identity, lifecycle, visibility order, or simulator labels.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("RGB reprojection image must have shape [B,3,H,W]")
    if image.shape[0] != belief.batch_size:
        raise ValueError("RGB reprojection image and belief batch sizes must match")
    if not image.is_floating_point():
        raise TypeError("RGB reprojection image must be floating point")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("RGB reprojection image contains NaN or Inf")
    if (
        isinstance(foreground_threshold, bool)
        or not isinstance(foreground_threshold, (int, float))
        or not math.isfinite(foreground_threshold)
        or not 0.0 < foreground_threshold < 2.0
    ):
        raise ValueError("foreground_threshold must be finite and lie in (0, 2)")
    if (
        isinstance(edge_softness_pixels, bool)
        or not isinstance(edge_softness_pixels, (int, float))
        or not math.isfinite(edge_softness_pixels)
        or edge_softness_pixels <= 0.0
    ):
        raise ValueError("edge_softness_pixels must be finite and positive")

    objects = belief.objects
    dtype = objects.position.dtype
    device = objects.position.device
    image = image.to(device=device, dtype=dtype)
    batch, _, height, width = image.shape
    world_from_camera, intrinsics = calibration_tensors(
        calibration,
        batch=batch,
        device=device,
        dtype=dtype,
        fallback_world_from_camera=belief.camera.world_from_camera,
        fallback_intrinsics=belief.camera.intrinsics,
    )
    centres, normalised_radius, _, camera_position = project_world_points(
        objects.position,
        objects.radius,
        world_from_camera,
        intrinsics,
        (height, width),
    )
    centre_x = 0.5 * (centres[..., 0] + 1.0) * max(width - 1, 1)
    centre_y = 0.5 * (centres[..., 1] + 1.0) * max(height - 1, 1)
    radius_pixels = normalised_radius * (0.5 * min(height, width))

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    distance = (
        (pixel_x[None, None] - centre_x[..., None, None]).square()
        + (pixel_y[None, None] - centre_y[..., None, None]).square()
        + torch.finfo(dtype).eps
    ).sqrt()
    per_sphere_alpha = torch.sigmoid(
        (radius_pixels[..., None, None] - distance) / float(edge_softness_pixels)
    )
    finite_projection = (
        torch.isfinite(centres).all(dim=-1)
        & torch.isfinite(radius_pixels)
        & torch.isfinite(camera_position).all(dim=-1)
    )
    positive_depth = camera_position[..., 2] > objects.radius.squeeze(-1) + 1.0e-4
    near_image = (
        (centre_x + radius_pixels >= -float(edge_softness_pixels))
        & (centre_x - radius_pixels <= width - 1 + float(edge_softness_pixels))
        & (centre_y + radius_pixels >= -float(edge_softness_pixels))
        & (centre_y - radius_pixels <= height - 1 + float(edge_softness_pixels))
    )
    projectable = objects.active & finite_projection & positive_depth & near_image
    per_sphere_alpha = per_sphere_alpha * projectable[..., None, None].to(dtype=dtype)
    predicted_foreground = 1.0 - (1.0 - per_sphere_alpha).prod(dim=1)

    # The target carries no graph and uses no simulator segmentation. Fewer
    # than half of each row is occupied in the sphere-world contract, so the
    # row median recovers the sky/floor background even as the camera moves.
    detached_image = image.detach()
    row_background = detached_image.median(dim=-1, keepdim=True).values
    foreground_strength = torch.linalg.vector_norm(detached_image - row_background, dim=1)
    target_softness = max(float(foreground_threshold) * 0.25, 1.0e-4)
    target_foreground = (
        (foreground_strength - (float(foreground_threshold) - target_softness))
        / (2.0 * target_softness)
    ).clamp(0.0, 1.0)

    square_error = (predicted_foreground - target_foreground).square()
    foreground_mass = target_foreground.sum(dim=(-2, -1))
    background = 1.0 - target_foreground
    background_mass = background.sum(dim=(-2, -1))
    positive_loss = (square_error * target_foreground).sum(dim=(-2, -1)) / (
        foreground_mass.clamp_min(1.0)
    )
    negative_loss = (square_error * background).sum(dim=(-2, -1)) / (background_mass.clamp_min(1.0))
    supported = projectable.any(dim=-1) & (foreground_mass >= 1.0)
    supported_weight = supported.to(dtype=dtype)
    row_loss = 0.5 * (positive_loss + negative_loss)
    loss = (row_loss * supported_weight).sum() / supported_weight.sum().clamp_min(1.0)

    predicted_binary = predicted_foreground.detach() >= 0.5
    target_binary = target_foreground >= 0.5
    intersection = (predicted_binary & target_binary).sum(dim=(-2, -1))
    union = (predicted_binary | target_binary).sum(dim=(-2, -1))
    supported_iou = torch.where(
        union > 0,
        intersection.to(dtype=dtype) / union.clamp_min(1).to(dtype=dtype),
        torch.zeros_like(row_loss),
    )
    iou_sum = (supported_iou * supported_weight).sum()
    metrics = {
        "rgb_reprojection_supported_row_count": float(supported.sum().detach().cpu()),
        "rgb_reprojection_projectable_object_count": float(projectable.sum().detach().cpu()),
        "rgb_reprojection_foreground_pixel_count": float(target_binary.sum().detach().cpu()),
        "rgb_reprojection_silhouette_iou_sum": float(iou_sum.detach().cpu()),
        "rgb_reprojection_silhouette_iou_count": float(supported.sum().detach().cpu()),
    }
    return loss, metrics


__all__ = ["soft_sphere_silhouette_reprojection"]
