"""Differentiable unordered RGB-D geometry for two coloured spheres.

This primitive is intentionally narrower than an instance segmenter.  It owns
the first multi-object rung's separated family and an opt-in bounded-partial
extension: exactly two fixed-radius spheres with distinct chromatic appearance.
A weighted colour-covariance eigenvector supplies symmetric ``+/-`` soft slot
logits, so its arbitrary sign can only permute the unordered output set.

No instance map, object ID, simulator state, connected component, or detached
continuous replacement enters the forward path.  Boolean mass/eigengap/depth
checks are fail-closed diagnostics around the ordinary PyTorch graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.observations.rgb.projector import camera_to_world
from world_model.observations.rgb.soft_geometry import (
    SoftDiscGeometryOutput,
    soft_disc_geometry_from_rgb,
)
from world_model.observations.rgbd.sphere_centres import (
    MAXIMUM_CALIBRATION_MAGNITUDE,
    MAXIMUM_METRIC_DISTANCE_M,
    MINIMUM_FOCAL_LENGTH_PIXELS,
    metric_sphere_centres_from_surface_depth,
)

SURFACE_SPHERE_GN_ITERATIONS = 2
SURFACE_SPHERE_TRUST_RADIUS_FRACTION = 0.25
TWO_DISC_ARCHITECTURE_ATTEMPT = 2


@dataclass(frozen=True)
class TwoDiscRGBDGeometryOutput:
    """Two unordered metric sphere measurements and observable diagnostics."""

    world_position: Tensor
    camera_position: Tensor
    centres: Tensor
    radius_pixels: Tensor
    appearance: Tensor
    surface_depth: Tensor
    centre_depth: Tensor
    confidence: Tensor
    valid_mask: Tensor
    pair_valid_mask: Tensor
    chromatic_eigengap: Tensor
    surface_fit_condition_number: Tensor
    surface_fit_radius: Tensor
    surface_fit_radius_relative_error: Tensor
    surface_fit_residual_relative_rms: Tensor
    observed_support_fraction: Tensor
    silhouette_gap_pixels: Tensor
    boundary_clearance_pixels: Tensor
    full_silhouette_radius_pixels: Tensor
    full_silhouette_gap_pixels: Tensor
    full_boundary_clearance_pixels: Tensor
    full_silhouette_overlap_fraction: Tensor
    chromatic_world_position: Tensor
    slot_logits: Tensor
    provisional_centres: Tensor
    geometry: SoftDiscGeometryOutput


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def _fraction(name: str, value: float, *, upper_inclusive: bool) -> float:
    resolved = _positive_finite(name, value)
    if resolved > 1.0 or (resolved == 1.0 and not upper_inclusive):
        interval = "(0,1]" if upper_inclusive else "(0,1)"
        raise ValueError(f"{name} must lie in {interval}")
    return resolved


def _broadcast_world_radius(
    world_radius: Tensor | float,
    *,
    batch: int,
    slots: int,
    reference: Tensor,
) -> Tensor:
    """Broadcast the declared fixed radius without importing hidden state."""

    if isinstance(world_radius, Tensor):
        if not world_radius.is_floating_point():
            raise TypeError("world_radius must be floating point")
        if world_radius.dtype != reference.dtype:
            raise TypeError("world_radius and RGB-D evidence must share dtype")
        if world_radius.device != reference.device:
            raise ValueError("world_radius and RGB-D evidence must share device")
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


def _project_full_silhouettes(
    camera_position: Tensor,
    radius: Tensor,
    intrinsics: Tensor,
    valid_mask: Tensor,
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project fitted fixed-radius spheres and measure conservative overlap.

    The public renderer uses the same mean-focal ``f * radius / z`` apparent
    radius. Circle-intersection area is normalized by the smaller full disc,
    so containment is one and separated silhouettes are zero. These values
    are observation-derived diagnostics and admissibility gates; renderer
    masks or labels never enter this path.
    """

    if camera_position.shape[-2:] != (2, 3) or radius.shape != camera_position.shape[:2]:
        raise ValueError("full-silhouette projection requires [B,2,3] centres and [B,2] radii")
    dtype = camera_position.dtype
    epsilon = max(float(torch.finfo(dtype).eps), 1.0e-8)
    raw_fx = intrinsics[:, 0, 0]
    raw_fy = intrinsics[:, 1, 1]
    calibration_valid = (
        torch.isfinite(intrinsics).all(dim=(-2, -1))
        & (intrinsics.abs().amax(dim=(-2, -1)) <= MAXIMUM_CALIBRATION_MAGNITUDE)
        & (raw_fx >= MINIMUM_FOCAL_LENGTH_PIXELS)
        & (raw_fy >= MINIMUM_FOCAL_LENGTH_PIXELS)
    )
    radius_valid = torch.isfinite(radius) & (radius > 0.0) & (radius <= MAXIMUM_METRIC_DISTANCE_M)
    depth = camera_position[..., 2]
    projection_valid = (
        valid_mask
        & calibration_valid[:, None]
        & radius_valid
        & torch.isfinite(camera_position).all(dim=-1)
        & (depth > radius + epsilon)
    )
    safe_depth = torch.where(projection_valid, depth, torch.ones_like(depth))
    safe_radius = torch.where(projection_valid, radius, torch.ones_like(radius))
    safe_fx = torch.where(calibration_valid, raw_fx, torch.ones_like(raw_fx))
    safe_fy = torch.where(calibration_valid, raw_fy, torch.ones_like(raw_fy))
    safe_cx = torch.where(calibration_valid, intrinsics[:, 0, 2], torch.zeros_like(raw_fx))
    safe_cy = torch.where(calibration_valid, intrinsics[:, 1, 2], torch.zeros_like(raw_fy))
    centre_x = safe_fx[:, None] * camera_position[..., 0] / safe_depth + safe_cx[:, None]
    centre_y = safe_fy[:, None] * camera_position[..., 1] / safe_depth + safe_cy[:, None]
    centres = torch.stack((centre_x, centre_y), dim=-1)
    focal = 0.5 * (safe_fx + safe_fy)
    radius_pixels = focal[:, None] * safe_radius / safe_depth
    separation = torch.linalg.vector_norm(centres[:, 0] - centres[:, 1], dim=-1)
    silhouette_gap = separation - radius_pixels.sum(dim=-1)
    boundary_clearance = torch.stack(
        (
            centre_x - radius_pixels,
            (width - 1) - centre_x - radius_pixels,
            centre_y - radius_pixels,
            (height - 1) - centre_y - radius_pixels,
        ),
        dim=-1,
    ).amin(dim=-1)

    radius_first = radius_pixels[:, 0]
    radius_second = radius_pixels[:, 1]
    minimum_radius = torch.minimum(radius_first, radius_second).clamp_min(epsilon)
    safe_separation = separation.clamp_min(epsilon)
    cosine_first = (safe_separation.square() + radius_first.square() - radius_second.square()) / (
        2.0 * safe_separation * radius_first.clamp_min(epsilon)
    )
    cosine_second = (safe_separation.square() + radius_second.square() - radius_first.square()) / (
        2.0 * safe_separation * radius_second.clamp_min(epsilon)
    )
    partial_area = (
        radius_first.square() * torch.acos(cosine_first.clamp(-1.0, 1.0))
        + radius_second.square() * torch.acos(cosine_second.clamp(-1.0, 1.0))
        - 0.5
        * (
            (-safe_separation + radius_first + radius_second)
            * (safe_separation + radius_first - radius_second)
            * (safe_separation - radius_first + radius_second)
            * (safe_separation + radius_first + radius_second)
        )
        .clamp_min(0.0)
        .sqrt()
    )
    no_overlap = separation >= radius_first + radius_second
    containment = separation <= (radius_first - radius_second).abs()
    intersection_area = torch.where(
        no_overlap,
        torch.zeros_like(partial_area),
        torch.where(containment, math.pi * minimum_radius.square(), partial_area),
    )
    overlap_fraction = (intersection_area / (math.pi * minimum_radius.square())).clamp(0.0, 1.0)
    pair_projection_valid = projection_valid.all(dim=-1)
    slot_gate = projection_valid.unsqueeze(-1)
    return (
        torch.where(slot_gate, centres, torch.zeros_like(centres)),
        torch.where(projection_valid, radius_pixels, torch.zeros_like(radius_pixels)),
        torch.where(pair_projection_valid, silhouette_gap, torch.zeros_like(silhouette_gap)),
        torch.where(projection_valid, boundary_clearance, torch.zeros_like(boundary_clearance)),
        torch.where(pair_projection_valid, overlap_fraction, torch.ones_like(overlap_fraction)),
        projection_valid,
    )


def _masked_geometry(
    geometry: SoftDiscGeometryOutput,
    slot_valid: Tensor,
) -> SoftDiscGeometryOutput:
    if slot_valid.shape != geometry.valid_mask.shape or slot_valid.dtype is not torch.bool:
        raise ValueError("geometry slot validity must be boolean [B,S]")
    slot_gate = slot_valid.unsqueeze(-1)
    image_gate = slot_valid.unsqueeze(-1).unsqueeze(-1)
    foreground_gate = slot_valid.any(dim=-1)[:, None, None, None]
    return SoftDiscGeometryOutput(
        centres=torch.where(slot_gate, geometry.centres, torch.zeros_like(geometry.centres)),
        radius_pixels=torch.where(
            slot_valid,
            geometry.radius_pixels,
            torch.zeros_like(geometry.radius_pixels),
        ),
        confidence=torch.where(
            slot_valid,
            geometry.confidence,
            torch.zeros_like(geometry.confidence),
        ),
        valid_mask=slot_valid,
        mass=torch.where(slot_valid, geometry.mass, torch.zeros_like(geometry.mass)),
        foreground_probability=torch.where(
            foreground_gate,
            geometry.foreground_probability,
            torch.zeros_like(geometry.foreground_probability),
        ),
        effective_masks=torch.where(
            image_gate,
            geometry.effective_masks,
            torch.zeros_like(geometry.effective_masks),
        ),
    )


def _fit_visible_sphere_surfaces(
    depth: Tensor,
    masks: Tensor,
    expected_radius: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    *,
    conditioning_limit: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fit sphere centres to observable surface points with weighted WLS."""

    batch, slots, height, width = masks.shape
    dtype = depth.dtype
    device = depth.device
    calibration_finite = torch.isfinite(world_from_camera).all(dim=(-2, -1))
    calibration_finite &= torch.isfinite(intrinsics).all(dim=(-2, -1))
    calibration_finite &= (
        world_from_camera.abs().amax(dim=(-2, -1)) <= MAXIMUM_CALIBRATION_MAGNITUDE
    )
    calibration_finite &= intrinsics.abs().amax(dim=(-2, -1)) <= MAXIMUM_CALIBRATION_MAGNITUDE
    calibration_finite &= (intrinsics[:, 0, 0] >= 1.0e-3) & (intrinsics[:, 1, 1] >= 1.0e-3)
    safe_intrinsics = torch.where(
        calibration_finite[:, None, None],
        intrinsics,
        torch.eye(3, dtype=dtype, device=device).expand(batch, -1, -1),
    )
    safe_transform = torch.where(
        calibration_finite[:, None, None],
        world_from_camera,
        torch.eye(4, dtype=dtype, device=device).expand(batch, -1, -1),
    )
    valid_depth = (
        torch.isfinite(depth[:, 0])
        & (depth[:, 0] > 0.0)
        & (depth[:, 0] <= MAXIMUM_METRIC_DISTANCE_M)
    )
    safe_depth = torch.where(valid_depth, depth[:, 0], torch.zeros_like(depth[:, 0]))
    y_pixel, x_pixel = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    ray_x = (x_pixel[None] - safe_intrinsics[:, 0, 2, None, None]) / safe_intrinsics[
        :, 0, 0, None, None
    ]
    ray_y = (y_pixel[None] - safe_intrinsics[:, 1, 2, None, None]) / safe_intrinsics[
        :, 1, 1, None, None
    ]
    points = torch.stack(
        (ray_x * safe_depth, ray_y * safe_depth, safe_depth),
        dim=-1,
    )
    weights = masks * valid_depth[:, None].to(dtype)
    support = weights.sum(dim=(-2, -1))
    epsilon = max(float(torch.finfo(dtype).eps), 1.0e-8)
    safe_support = support.clamp_min(epsilon)
    mean_point = torch.einsum("bshw,bhwc->bsc", weights, points) / safe_support.unsqueeze(-1)
    squared_norm = points.square().sum(dim=-1)
    mean_squared_norm = (
        torch.einsum(
            "bshw,bhw->bs",
            weights,
            squared_norm,
        )
        / safe_support
    )
    centred_points = points[:, None] - mean_point[:, :, None, None]
    centred_norm = squared_norm[:, None] - mean_squared_norm[:, :, None, None]
    normal = torch.einsum(
        "bshw,bshwi,bshwj->bsij",
        weights / safe_support[:, :, None, None],
        centred_points,
        centred_points,
    )
    right = 0.5 * torch.einsum(
        "bshw,bshwi,bshw->bsi",
        weights / safe_support[:, :, None, None],
        centred_points,
        centred_norm,
    )
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(normal.detach())
        condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(epsilon)
        radius_admissible = (
            torch.isfinite(expected_radius.detach())
            & (expected_radius.detach() > 0.0)
            & (expected_radius.detach() <= MAXIMUM_METRIC_DISTANCE_M)
        )
        fit_admissible = (
            torch.isfinite(eigenvalues).all(dim=-1)
            & (eigenvalues[..., 0] > epsilon)
            & (condition <= conditioning_limit)
            & (support.detach() >= 4.0)
            & radius_admissible
            & calibration_finite[:, None]
        )
    safe_normal = torch.where(
        fit_admissible[..., None, None],
        normal,
        torch.eye(3, dtype=dtype, device=device).expand(batch, slots, -1, -1),
    )
    safe_right = torch.where(fit_admissible.unsqueeze(-1), right, torch.zeros_like(right))
    camera_centre = torch.linalg.solve(safe_normal, safe_right.unsqueeze(-1)).squeeze(-1)
    safe_radius = torch.where(
        fit_admissible,
        expected_radius,
        torch.ones_like(expected_radius),
    )
    maximum_condition = condition
    for _ in range(SURFACE_SPHERE_GN_ITERATIONS):
        delta = camera_centre[:, :, None, None] - points[:, None]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(epsilon)
        jacobian = delta / distance.unsqueeze(-1)
        residual = distance - safe_radius[:, :, None, None]
        gn_normal = torch.einsum(
            "bshw,bshwi,bshwj->bsij",
            weights / safe_support[:, :, None, None],
            jacobian,
            jacobian,
        )
        gn_right = torch.einsum(
            "bshw,bshwi,bshw->bsi",
            weights / safe_support[:, :, None, None],
            jacobian,
            residual,
        )
        with torch.no_grad():
            gn_eigenvalues = torch.linalg.eigvalsh(gn_normal.detach())
            gn_condition = gn_eigenvalues[..., -1] / gn_eigenvalues[..., 0].clamp_min(epsilon)
            iteration_admissible = (
                fit_admissible
                & torch.isfinite(gn_eigenvalues).all(dim=-1)
                & (gn_eigenvalues[..., 0] > epsilon)
                & (gn_condition <= conditioning_limit)
            )
        maximum_condition = torch.maximum(maximum_condition, gn_condition)
        safe_gn_normal = torch.where(
            iteration_admissible[..., None, None],
            gn_normal,
            torch.eye(3, dtype=dtype, device=device).expand(batch, slots, -1, -1),
        )
        safe_gn_right = torch.where(
            iteration_admissible.unsqueeze(-1),
            gn_right,
            torch.zeros_like(gn_right),
        )
        raw_step = -torch.linalg.solve(
            safe_gn_normal,
            safe_gn_right.unsqueeze(-1),
        ).squeeze(-1)
        trust_radius = (safe_radius * SURFACE_SPHERE_TRUST_RADIUS_FRACTION).clamp_min(epsilon)
        step = trust_radius.unsqueeze(-1) * torch.tanh(raw_step / trust_radius.unsqueeze(-1))
        camera_centre = camera_centre + step
        fit_admissible = iteration_admissible
    world_centre = camera_to_world(camera_centre, safe_transform)
    diagnostic_delta = points[:, None] - camera_centre[:, :, None, None]
    diagnostic_delta = torch.where(
        fit_admissible[:, :, None, None, None],
        diagnostic_delta,
        torch.ones_like(diagnostic_delta),
    )
    residual_radius = torch.linalg.vector_norm(diagnostic_delta, dim=-1)
    fitted_radius = (
        torch.einsum(
            "bshw,bshw->bs",
            weights,
            residual_radius,
        )
        / safe_support
    )
    residual_mean_square = (
        torch.einsum(
            "bshw,bshw->bs",
            weights,
            (residual_radius - safe_radius[:, :, None, None]).square(),
        )
        / safe_support
    ).clamp_min(0.0)
    # The epsilon-squared floor has negligible diagnostic scale but prevents
    # the undefined derivative of ``sqrt(0)`` on an invalid or exact-fit row.
    residual_rms = (residual_mean_square + epsilon**2).sqrt()
    valid = fit_admissible & torch.isfinite(camera_centre.detach()).all(dim=-1)
    value_gate = valid.unsqueeze(-1)
    return (
        torch.where(value_gate, world_centre, torch.zeros_like(world_centre)),
        torch.where(value_gate, camera_centre, torch.zeros_like(camera_centre)),
        torch.where(valid, fitted_radius, torch.zeros_like(fitted_radius)),
        torch.where(
            valid,
            maximum_condition.to(dtype),
            torch.zeros_like(maximum_condition, dtype=dtype),
        ),
        torch.where(valid, residual_rms, torch.zeros_like(residual_rms)),
        torch.where(valid, support, torch.zeros_like(support)),
        valid,
    )


def two_disc_geometry_from_rgbd(
    image: Tensor,
    depth: Tensor,
    world_radius: Tensor | float,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    *,
    foreground_threshold: float = 0.04,
    foreground_temperature: float = 0.01,
    minimum_mass: float = 4.0,
    chromatic_temperature: float = 0.05,
    minimum_chromatic_eigengap: float = 0.01,
    spatial_temperature_pixels: float = 1.0,
    chromatic_centre_blend: float = 0.0025,
    minimum_silhouette_gap_pixels: float = 2.0,
    minimum_boundary_clearance_pixels: float = 2.0,
    maximum_surface_radius_relative_error: float = 0.05,
    surface_fit_conditioning_limit: float = 100.0,
    bounded_partial_visibility: bool = False,
    minimum_observed_support_fraction: float = 0.35,
    maximum_surface_residual_relative_rms: float = 0.05,
    maximum_full_silhouette_overlap_fraction: float = 0.60,
) -> TwoDiscRGBDGeometryOutput:
    """Recover two unordered sphere centres from bounded RGB-D evidence.

    The fixed ``world_radius`` is the same explicit checkpointed prior as the
    accepted one-object bridge.  Radius estimation is deliberately deferred to
    a later scale-identification rung. The opt-in partial-visibility branch
    uses chromatic ownership of visible depth surfaces and a fitted full-disc
    projection; the default separated branch is unchanged.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("two-disc RGB image must have shape [B,3,H,W]")
    if image.dtype not in {torch.float32, torch.float64}:
        raise TypeError("two-disc RGB-D geometry supports only float32 and float64")
    if depth.shape != (image.shape[0], 1, *image.shape[-2:]):
        raise ValueError("two-disc depth must have shape [B,1,H,W]")
    if depth.dtype != image.dtype or depth.device != image.device:
        raise ValueError("two-disc RGB and depth must share dtype and device")
    if not isinstance(bounded_partial_visibility, bool):
        raise TypeError("bounded_partial_visibility must be boolean")
    if bounded_partial_visibility:
        batch = image.shape[0]
        if world_from_camera.shape != (batch, 4, 4):
            raise ValueError("world_from_camera must have shape [B,4,4]")
        if intrinsics.shape != (batch, 3, 3):
            raise ValueError("intrinsics must have shape [B,3,3]")
        if not world_from_camera.is_floating_point() or not intrinsics.is_floating_point():
            raise TypeError("camera calibration must be floating point")
        if world_from_camera.dtype != image.dtype or intrinsics.dtype != image.dtype:
            raise TypeError("RGB-D evidence and calibration must share dtype")
        if world_from_camera.device != image.device or intrinsics.device != image.device:
            raise ValueError("RGB-D evidence and calibration must share device")

    slot_temperature = _positive_finite("chromatic_temperature", chromatic_temperature)
    minimum_gap = _positive_finite(
        "minimum_chromatic_eigengap",
        minimum_chromatic_eigengap,
    )
    spatial_temperature = _positive_finite(
        "spatial_temperature_pixels",
        spatial_temperature_pixels,
    )
    if isinstance(chromatic_centre_blend, bool) or not isinstance(
        chromatic_centre_blend,
        (int, float),
    ):
        raise TypeError("chromatic_centre_blend must be a real number")
    resolved_chromatic_blend = float(chromatic_centre_blend)
    if not math.isfinite(resolved_chromatic_blend) or not 0.0 < resolved_chromatic_blend <= 1.0:
        raise ValueError("chromatic_centre_blend must lie in (0,1]")
    minimum_silhouette_gap = _positive_finite(
        "minimum_silhouette_gap_pixels",
        minimum_silhouette_gap_pixels,
    )
    minimum_boundary_clearance = _positive_finite(
        "minimum_boundary_clearance_pixels",
        minimum_boundary_clearance_pixels,
    )
    maximum_radius_error = _positive_finite(
        "maximum_surface_radius_relative_error",
        maximum_surface_radius_relative_error,
    )
    surface_conditioning_limit = _positive_finite(
        "surface_fit_conditioning_limit",
        surface_fit_conditioning_limit,
    )
    if surface_conditioning_limit <= 1.0:
        raise ValueError("surface_fit_conditioning_limit must be greater than one")
    minimum_support_fraction = _fraction(
        "minimum_observed_support_fraction",
        minimum_observed_support_fraction,
        upper_inclusive=True,
    )
    maximum_residual_rms = _fraction(
        "maximum_surface_residual_relative_rms",
        maximum_surface_residual_relative_rms,
        upper_inclusive=True,
    )
    maximum_overlap_fraction = _fraction(
        "maximum_full_silhouette_overlap_fraction",
        maximum_full_silhouette_overlap_fraction,
        upper_inclusive=False,
    )
    required_mass = _positive_finite("minimum_mass", minimum_mass)

    foreground = soft_disc_geometry_from_rgb(
        image,
        foreground_threshold=foreground_threshold,
        foreground_temperature=foreground_temperature,
        minimum_mass=minimum_mass,
    )
    batch, _, height, width = image.shape
    probability = foreground.foreground_probability[:, 0]
    epsilon = max(float(torch.finfo(image.dtype).eps), 1.0e-8)

    channel_sum = image.sum(dim=1, keepdim=True).clamp_min(epsilon)
    chromaticity = image / channel_sum
    mass = probability.sum(dim=(-2, -1)).clamp_min(epsilon)
    mean = (chromaticity * probability.unsqueeze(1)).sum(dim=(-2, -1)) / mass.unsqueeze(-1)
    centred = chromaticity - mean[:, :, None, None]
    flattened = centred.flatten(start_dim=2).transpose(1, 2)
    weights = probability.flatten(start_dim=1)
    covariance = torch.einsum(
        "bp,bpi,bpj->bij",
        weights / mass.unsqueeze(-1),
        flattened,
        flattened,
    )

    # Degenerate colour rows must have finite zero gradients.  Determine the
    # discrete eigengap admissibility on detached covariance, substitute a
    # benign matrix before the differentiable eigenvector calculation, and
    # retain the real covariance unchanged on admissible rows.
    with torch.no_grad():
        diagnostic_eigenvalues = torch.linalg.eigvalsh(covariance.detach())
        diagnostic_gap = diagnostic_eigenvalues[:, -1] - diagnostic_eigenvalues[:, -2]
        spectral_admissible = (
            torch.isfinite(diagnostic_eigenvalues).all(dim=-1)
            & (diagnostic_gap >= minimum_gap)
            & (mass.detach() >= 2.0 * required_mass)
        )
    safe_diagonal = image.new_tensor([0.0, minimum_gap, 2.0 * minimum_gap])
    safe_covariance = torch.where(
        spectral_admissible[:, None, None],
        covariance,
        torch.diag(safe_diagonal).expand(batch, -1, -1),
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(safe_covariance)
    axis = eigenvectors[..., -1]
    score = torch.einsum("bchw,bc->bhw", centred, axis)
    positive_logits = score / image.new_tensor(slot_temperature)
    chromatic_slot_logits = torch.stack((positive_logits, -positive_logits), dim=1)
    provisional_geometry = soft_disc_geometry_from_rgb(
        image,
        chromatic_slot_logits,
        foreground_threshold=foreground_threshold,
        foreground_temperature=foreground_temperature,
        minimum_mass=minimum_mass,
    )
    if bounded_partial_visibility:
        geometry = provisional_geometry
        selected_slot_logits = chromatic_slot_logits
        # The sign of the symmetric chromatic score is the parameter-free
        # ownership boundary. Retain ordinary ReLU derivatives on its owning
        # side while removing the low-probability cross-colour tails that can
        # mix the front sphere's depth surface into the rear sphere fit.
        chromatic_owner_strength = F.relu(2.0 * chromatic_slot_logits.sigmoid() - 1.0)
        surface_masks = probability[:, None] * chromatic_owner_strength
    else:
        centre_x = 0.5 * (provisional_geometry.centres[..., 0] + 1.0) * (width - 1)
        centre_y = 0.5 * (provisional_geometry.centres[..., 1] + 1.0) * (height - 1)
        y_pixel, x_pixel = torch.meshgrid(
            torch.arange(height, dtype=image.dtype, device=image.device),
            torch.arange(width, dtype=image.dtype, device=image.device),
            indexing="ij",
        )
        distance_squared = (x_pixel - centre_x[..., None, None]).square() + (
            y_pixel - centre_y[..., None, None]
        ).square()
        spatial_score = (distance_squared[:, 1] - distance_squared[:, 0]) / image.new_tensor(
            spatial_temperature**2
        )
        raw_slot_logits = torch.stack((spatial_score, -spatial_score), dim=1)
        geometry = soft_disc_geometry_from_rgb(
            image,
            raw_slot_logits,
            foreground_threshold=foreground_threshold,
            foreground_temperature=foreground_temperature,
            minimum_mass=minimum_mass,
        )
        selected_slot_logits = raw_slot_logits
        surface_masks = geometry.effective_masks
    # Preserve a small, explicit RGB-centre owner in the deployed metric
    # estimate.  The exact fixed-radius surface fit below is intentionally
    # nearly invariant to mask weights on noiseless spheres, so without this
    # physical ray estimate the RGB branch can become numerically detached
    # even though it owns the unordered slot partition.  The refined metric
    # depth remains the range owner; only the chromatic image-centre ray is
    # blended, and the bounded coefficient is a checkpointed semantic.
    raw_fx = intrinsics[:, 0, 0]
    raw_fy = intrinsics[:, 1, 1]
    focal_valid = (
        torch.isfinite(raw_fx)
        & torch.isfinite(raw_fy)
        & (raw_fx >= MINIMUM_FOCAL_LENGTH_PIXELS)
        & (raw_fy >= MINIMUM_FOCAL_LENGTH_PIXELS)
        & (raw_fx <= MAXIMUM_CALIBRATION_MAGNITUDE)
        & (raw_fy <= MAXIMUM_CALIBRATION_MAGNITUDE)
    )
    safe_fx = torch.where(focal_valid, raw_fx, torch.ones_like(raw_fx))
    safe_fy = torch.where(focal_valid, raw_fy, torch.ones_like(raw_fy))
    raw_cx = intrinsics[:, 0, 2]
    raw_cy = intrinsics[:, 1, 2]
    principal_valid = (
        torch.isfinite(raw_cx)
        & torch.isfinite(raw_cy)
        & (raw_cx.abs() <= MAXIMUM_CALIBRATION_MAGNITUDE)
        & (raw_cy.abs() <= MAXIMUM_CALIBRATION_MAGNITUDE)
    )
    safe_cx = torch.where(principal_valid, raw_cx, torch.zeros_like(raw_cx))
    safe_cy = torch.where(principal_valid, raw_cy, torch.zeros_like(raw_cy))
    provisional_pixel_x = 0.5 * (provisional_geometry.centres[..., 0] + 1.0) * (width - 1)
    provisional_pixel_y = 0.5 * (provisional_geometry.centres[..., 1] + 1.0) * (height - 1)
    provisional_ray_x = (provisional_pixel_x - safe_cx[:, None]) / safe_fx[:, None]
    provisional_ray_y = (provisional_pixel_y - safe_cy[:, None]) / safe_fy[:, None]
    transform_valid = torch.isfinite(world_from_camera).all(dim=(-2, -1)) & (
        world_from_camera.abs().amax(dim=(-2, -1)) <= MAXIMUM_CALIBRATION_MAGNITUDE
    )
    safe_transform = torch.where(
        transform_valid[:, None, None],
        world_from_camera,
        torch.eye(4, dtype=image.dtype, device=image.device).expand(batch, -1, -1),
    )
    if bounded_partial_visibility:
        expected_radius = _broadcast_world_radius(
            world_radius,
            batch=batch,
            slots=2,
            reference=image,
        )
    else:
        metric = metric_sphere_centres_from_surface_depth(
            geometry.centres,
            depth,
            world_radius,
            world_from_camera,
            intrinsics,
        )
        chromatic_camera_position = torch.stack(
            (
                provisional_ray_x * metric.centre_depth,
                provisional_ray_y * metric.centre_depth,
                metric.centre_depth,
            ),
            dim=-1,
        )
        chromatic_world_position = camera_to_world(chromatic_camera_position, safe_transform)
        expected_radius = (metric.centre_depth - metric.surface_depth) * torch.sqrt(
            1.0 + metric.ray_xy.square().sum(dim=-1)
        )
    (
        fitted_world_position,
        fitted_camera_position,
        fitted_radius,
        fit_condition,
        surface_residual_rms,
        surface_support,
        surface_fit_valid,
    ) = _fit_visible_sphere_surfaces(
        depth,
        surface_masks,
        expected_radius,
        world_from_camera,
        intrinsics,
        conditioning_limit=surface_conditioning_limit,
    )
    if bounded_partial_visibility:
        fitted_depth = fitted_camera_position[..., 2]
        chromatic_camera_position = torch.stack(
            (
                provisional_ray_x * fitted_depth,
                provisional_ray_y * fitted_depth,
                fitted_depth,
            ),
            dim=-1,
        )
        chromatic_world_position = camera_to_world(chromatic_camera_position, safe_transform)

    appearance_mass = surface_masks.sum(dim=(-2, -1)).clamp_min(epsilon)
    appearance = torch.einsum(
        "bshw,bchw->bsc",
        surface_masks,
        chromaticity,
    ) / appearance_mass.unsqueeze(-1)
    appearance = F.normalize(appearance, dim=-1, eps=epsilon)
    appearance_separation = 1.0 - F.cosine_similarity(
        appearance[:, 0],
        appearance[:, 1],
        dim=-1,
        eps=epsilon,
    )
    radius_relative_error = (fitted_radius - expected_radius).abs() / expected_radius.clamp_min(
        epsilon
    )
    residual_relative_rms = surface_residual_rms / expected_radius.clamp_min(epsilon)
    (
        full_centres_pixels,
        full_radius_pixels,
        full_silhouette_gap_pixels,
        full_boundary_clearance_pixels,
        full_silhouette_overlap_fraction,
        full_projection_valid,
    ) = _project_full_silhouettes(
        fitted_camera_position,
        expected_radius,
        intrinsics,
        surface_fit_valid,
        height=height,
        width=width,
    )
    del full_centres_pixels
    full_disc_area = math.pi * full_radius_pixels.square()
    observed_support_fraction = (surface_support / full_disc_area.clamp_min(epsilon)).clamp(
        0.0, 1.0
    )
    fitted_depth = fitted_camera_position[..., 2]
    safe_fitted_depth = fitted_depth.clamp_min(epsilon)
    fitted_ray_xy = fitted_camera_position[..., :2] / safe_fitted_depth.unsqueeze(-1)
    fitted_ray_norm = torch.sqrt(1.0 + fitted_ray_xy.square().sum(dim=-1))
    fitted_surface_depth = fitted_depth - expected_radius / fitted_ray_norm
    reported_surface_depth = (
        fitted_surface_depth if bounded_partial_visibility else metric.surface_depth
    )
    reported_centre_depth = fitted_depth if bounded_partial_visibility else metric.centre_depth
    refined_centre_x = 0.5 * (geometry.centres[..., 0] + 1.0) * (width - 1)
    refined_centre_y = 0.5 * (geometry.centres[..., 1] + 1.0) * (height - 1)
    refined_centres_pixels = torch.stack((refined_centre_x, refined_centre_y), dim=-1)
    centre_separation_pixels = torch.linalg.vector_norm(
        refined_centres_pixels[:, 0] - refined_centres_pixels[:, 1],
        dim=-1,
    )
    silhouette_gap_pixels = centre_separation_pixels - geometry.radius_pixels.sum(dim=-1)
    boundary_clearance_pixels = torch.stack(
        (
            refined_centre_x - geometry.radius_pixels,
            (width - 1) - refined_centre_x - geometry.radius_pixels,
            refined_centre_y - geometry.radius_pixels,
            (height - 1) - refined_centre_y - geometry.radius_pixels,
        ),
        dim=-1,
    ).amin(dim=(-2, -1))
    observable_separation_admissible = (
        torch.isfinite(silhouette_gap_pixels.detach())
        & torch.isfinite(boundary_clearance_pixels.detach())
        & (silhouette_gap_pixels.detach() >= minimum_silhouette_gap)
        & (boundary_clearance_pixels.detach() >= minimum_boundary_clearance)
    )
    if bounded_partial_visibility:
        both_projected = full_projection_valid.all(dim=-1)
        overlap_admissible = ~both_projected | (
            torch.isfinite(full_silhouette_overlap_fraction.detach())
            & (full_silhouette_overlap_fraction.detach() <= maximum_overlap_fraction)
        )
        slot_valid = (
            spectral_admissible.unsqueeze(-1)
            & geometry.valid_mask
            & surface_fit_valid
            & full_projection_valid
            & (radius_relative_error.detach() <= maximum_radius_error)
            & (residual_relative_rms.detach() <= maximum_residual_rms)
            & (observed_support_fraction.detach() >= minimum_support_fraction)
            & torch.isfinite(full_boundary_clearance_pixels.detach())
            & (full_boundary_clearance_pixels.detach() >= minimum_boundary_clearance)
            & torch.isfinite(appearance.detach()).all(dim=-1)
            & (appearance_separation.detach() > 0.0).unsqueeze(-1)
            & overlap_admissible.unsqueeze(-1)
        )
        pair_valid = slot_valid.all(dim=-1)
    else:
        pair_valid = (
            spectral_admissible
            & observable_separation_admissible
            & geometry.valid_mask.all(dim=-1)
            & metric.valid_mask.all(dim=-1)
            & surface_fit_valid.all(dim=-1)
            & (radius_relative_error.detach() <= maximum_radius_error).all(dim=-1)
            & torch.isfinite(appearance.detach()).all(dim=(-2, -1))
            & (appearance_separation.detach() > 0.0)
        )
        slot_valid = pair_valid.unsqueeze(-1).expand(batch, 2)
    value_gate = slot_valid.unsqueeze(-1)
    blended_world_position = (
        1.0 - resolved_chromatic_blend
    ) * fitted_world_position + resolved_chromatic_blend * chromatic_world_position
    blended_camera_position = (
        1.0 - resolved_chromatic_blend
    ) * fitted_camera_position + resolved_chromatic_blend * chromatic_camera_position
    masked_geometry = _masked_geometry(geometry, slot_valid)
    differentiable_gap = eigenvalues[:, -1] - eigenvalues[:, -2]
    gap_confidence = differentiable_gap / (differentiable_gap + image.new_tensor(minimum_gap))
    if bounded_partial_visibility:
        confidence = (
            geometry.confidence
            * observed_support_fraction.clamp(0.0, 1.0)
            * gap_confidence.unsqueeze(-1)
        ).clamp(0.0, 1.0)
    else:
        confidence = (
            geometry.confidence * metric.depth_support * gap_confidence.unsqueeze(-1)
        ).clamp(0.0, 1.0)
    logit_gate = (
        slot_valid[:, :, None, None]
        if bounded_partial_visibility
        else pair_valid[:, None, None, None]
    )
    slot_logits = torch.where(
        logit_gate, selected_slot_logits, torch.zeros_like(selected_slot_logits)
    )

    return TwoDiscRGBDGeometryOutput(
        world_position=torch.where(
            value_gate,
            blended_world_position,
            torch.zeros_like(blended_world_position),
        ),
        camera_position=torch.where(
            value_gate,
            blended_camera_position,
            torch.zeros_like(blended_camera_position),
        ),
        centres=masked_geometry.centres,
        radius_pixels=masked_geometry.radius_pixels,
        appearance=torch.where(value_gate, appearance, torch.zeros_like(appearance)),
        surface_depth=torch.where(
            slot_valid,
            reported_surface_depth,
            torch.zeros_like(reported_surface_depth),
        ),
        centre_depth=torch.where(
            slot_valid,
            reported_centre_depth,
            torch.zeros_like(reported_centre_depth),
        ),
        confidence=torch.where(slot_valid, confidence, torch.zeros_like(confidence)),
        valid_mask=slot_valid,
        pair_valid_mask=pair_valid,
        chromatic_eigengap=torch.where(
            spectral_admissible,
            diagnostic_gap.to(dtype=image.dtype),
            torch.zeros_like(diagnostic_gap, dtype=image.dtype),
        ),
        surface_fit_condition_number=torch.where(
            slot_valid,
            fit_condition,
            torch.zeros_like(fit_condition),
        ),
        surface_fit_radius=torch.where(
            slot_valid,
            fitted_radius,
            torch.zeros_like(fitted_radius),
        ),
        surface_fit_radius_relative_error=torch.where(
            slot_valid,
            radius_relative_error,
            torch.zeros_like(radius_relative_error),
        ),
        surface_fit_residual_relative_rms=torch.where(
            slot_valid,
            residual_relative_rms,
            torch.zeros_like(residual_relative_rms),
        ),
        observed_support_fraction=torch.where(
            slot_valid,
            observed_support_fraction,
            torch.zeros_like(observed_support_fraction),
        ),
        silhouette_gap_pixels=torch.where(
            torch.isfinite(silhouette_gap_pixels),
            silhouette_gap_pixels,
            torch.zeros_like(silhouette_gap_pixels),
        ),
        boundary_clearance_pixels=torch.where(
            torch.isfinite(boundary_clearance_pixels),
            boundary_clearance_pixels,
            torch.zeros_like(boundary_clearance_pixels),
        ),
        full_silhouette_radius_pixels=torch.where(
            slot_valid,
            full_radius_pixels,
            torch.zeros_like(full_radius_pixels),
        ),
        full_silhouette_gap_pixels=torch.where(
            pair_valid,
            full_silhouette_gap_pixels,
            torch.zeros_like(full_silhouette_gap_pixels),
        ),
        full_boundary_clearance_pixels=torch.where(
            slot_valid,
            full_boundary_clearance_pixels,
            torch.zeros_like(full_boundary_clearance_pixels),
        ),
        full_silhouette_overlap_fraction=torch.where(
            pair_valid,
            full_silhouette_overlap_fraction,
            torch.zeros_like(full_silhouette_overlap_fraction),
        ),
        chromatic_world_position=torch.where(
            value_gate,
            chromatic_world_position,
            torch.zeros_like(chromatic_world_position),
        ),
        slot_logits=slot_logits,
        provisional_centres=torch.where(
            value_gate,
            provisional_geometry.centres,
            torch.zeros_like(provisional_geometry.centres),
        ),
        geometry=masked_geometry,
    )


__all__ = [
    "TWO_DISC_ARCHITECTURE_ATTEMPT",
    "TwoDiscRGBDGeometryOutput",
    "two_disc_geometry_from_rgbd",
]
