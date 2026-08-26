"""Differentiable image-moment geometry for the synthetic sphere world.

The functions in this module use RGB pixels and optional learned per-slot mask
logits only.  They do not consume renderer labels, simulator state, connected
components, or CPU-only assignment.  Every continuous output follows the
ordinary PyTorch graph; the boolean validity mask is diagnostic rather than a
straight-through substitute for another forward computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SoftDiscGeometryOutput:
    """Soft per-slot disc geometry derived from RGB evidence.

    ``centres`` use the project's ``[-1, 1]`` image coordinates and
    ``radius_pixels`` uses pixels.  ``confidence`` is continuous and can be
    used to weight a measurement loss or update.  ``valid_mask`` only records
    whether the foreground mass reaches ``minimum_mass``; callers must not use
    it to pretend that an unrelated learned value produced the returned
    geometry.
    """

    centres: Tensor
    radius_pixels: Tensor
    confidence: Tensor
    valid_mask: Tensor
    mass: Tensor
    foreground_probability: Tensor
    effective_masks: Tensor


def _validate_positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def soft_disc_geometry_from_rgb(
    image: Tensor,
    slot_mask_logits: Tensor | None = None,
    *,
    foreground_threshold: float = 0.04,
    foreground_temperature: float = 0.01,
    minimum_mass: float = 4.0,
) -> SoftDiscGeometryOutput:
    """Estimate disc centres and apparent radii with differentiable moments.

    Args:
        image: Floating RGB tensor with shape ``[B, 3, H, W]``.
        slot_mask_logits: Optional learned logits with shape ``[B, S, H, W]``.
            Their sigmoid probabilities softly assign foreground evidence to
            slots.  When omitted, a single slot receives all foreground.
        foreground_threshold: RGB distance from the row-wise background at
            which foreground probability is one half before zero-background
            debiasing.
        foreground_temperature: Soft-threshold temperature in RGB units.
        minimum_mass: Foreground pixel mass required by ``valid_mask`` and the
            scale of the continuous mass-confidence term.

    The row median is valid for the toy renderer contract in which objects
    occupy less than half of any row.  It is computed on the input device and
    is not detached.  Radius is derived from the radial second moment: a
    uniformly filled disc of radius ``r`` has ``E[distance**2] = r**2 / 2``.
    Instance separation is deliberately delegated to learned soft masks; this
    primitive does not hide a hard assignment or connected-component step.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("soft disc geometry image must have shape [B,3,H,W]")
    if not image.is_floating_point():
        raise TypeError("soft disc geometry image must be floating point")
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("soft disc geometry requires image dimensions of at least two pixels")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("soft disc geometry image contains NaN or Inf")

    threshold = _validate_positive_finite("foreground_threshold", foreground_threshold)
    if threshold >= 2.0:
        raise ValueError("foreground_threshold must lie in (0, 2)")
    temperature = _validate_positive_finite("foreground_temperature", foreground_temperature)
    required_mass = _validate_positive_finite("minimum_mass", minimum_mass)

    batch, _, height, width = image.shape
    if slot_mask_logits is None:
        slot_probability = image.new_ones((batch, 1, height, width))
    else:
        if slot_mask_logits.ndim != 4:
            raise ValueError("slot_mask_logits must have shape [B,S,H,W]")
        if slot_mask_logits.shape[0] != batch or slot_mask_logits.shape[-2:] != (height, width):
            raise ValueError("slot_mask_logits batch and image dimensions must match the RGB image")
        if slot_mask_logits.shape[1] <= 0:
            raise ValueError("slot_mask_logits must contain at least one slot")
        if not slot_mask_logits.is_floating_point():
            raise TypeError("slot_mask_logits must be floating point")
        if slot_mask_logits.device != image.device:
            raise ValueError("slot_mask_logits and image must be on the same device")
        if not bool(torch.isfinite(slot_mask_logits).all()):
            raise ValueError("slot_mask_logits contains NaN or Inf")
        slot_probability = slot_mask_logits.to(dtype=image.dtype).sigmoid()

    # The renderer background is constant along each row.  Median is robust to
    # the stipulated minority foreground occupancy and remains a native tensor
    # operation on CPU, CUDA, and MPS.
    row_background = image.median(dim=-1, keepdim=True).values
    foreground_strength = torch.linalg.vector_norm(image - row_background, dim=1)
    threshold_tensor = image.new_tensor(threshold)
    temperature_tensor = image.new_tensor(temperature)
    background_probability = torch.sigmoid(-threshold_tensor / temperature_tensor)
    foreground_probability = (
        (torch.sigmoid((foreground_strength - threshold_tensor) / temperature_tensor)
         - background_probability)
        / (1.0 - background_probability).clamp_min(torch.finfo(image.dtype).eps)
    ).clamp(0.0, 1.0)
    foreground_probability = foreground_probability.unsqueeze(1)
    effective_masks = foreground_probability * slot_probability

    dtype = image.dtype
    device = image.device
    y_pixels, x_pixels = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    y_normalised, x_normalised = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    epsilon = max(float(torch.finfo(dtype).eps), 1.0e-8)
    mass = effective_masks.sum(dim=(-2, -1))
    safe_mass = mass.clamp_min(epsilon)
    centre_x_normalised = (effective_masks * x_normalised).sum(dim=(-2, -1)) / safe_mass
    centre_y_normalised = (effective_masks * y_normalised).sum(dim=(-2, -1)) / safe_mass
    centres = torch.stack((centre_x_normalised, centre_y_normalised), dim=-1)

    centre_x_pixels = 0.5 * (centre_x_normalised + 1.0) * (width - 1)
    centre_y_pixels = 0.5 * (centre_y_normalised + 1.0) * (height - 1)
    delta_x = x_pixels - centre_x_pixels[..., None, None]
    delta_y = y_pixels - centre_y_pixels[..., None, None]
    variance_x = (effective_masks * delta_x.square()).sum(dim=(-2, -1)) / safe_mass
    variance_y = (effective_masks * delta_y.square()).sum(dim=(-2, -1)) / safe_mass
    covariance_xy = (effective_masks * delta_x * delta_y).sum(dim=(-2, -1)) / safe_mass
    radial_second_moment = variance_x + variance_y
    radius_pixels = (2.0 * radial_second_moment).clamp_min(epsilon).sqrt()

    # Confidence has three smooth, observable factors: enough foreground
    # evidence, approximately circular second moments, and a silhouette that
    # clears the image boundary.  It neither imports labels nor masks a value
    # with a fabricated derivative.
    mass_confidence = 1.0 - torch.exp(-mass / required_mass)
    covariance_determinant = (variance_x * variance_y - covariance_xy.square()).clamp_min(0.0)
    circularity = (
        4.0 * covariance_determinant
        / (radial_second_moment.square() + epsilon)
    ).clamp(0.0, 1.0)
    edge_clearance = torch.stack(
        (
            centre_x_pixels,
            width - 1 - centre_x_pixels,
            centre_y_pixels,
            height - 1 - centre_y_pixels,
        ),
        dim=-1,
    ).amin(dim=-1) - radius_pixels
    boundary_confidence = torch.sigmoid(edge_clearance)
    confidence = (mass_confidence * circularity * boundary_confidence).clamp(0.0, 1.0)
    valid_mask = (mass.detach() >= required_mass) & torch.isfinite(centres.detach()).all(dim=-1)

    return SoftDiscGeometryOutput(
        centres=centres,
        radius_pixels=radius_pixels,
        confidence=confidence,
        valid_mask=valid_mask,
        mass=mass,
        foreground_probability=foreground_probability,
        effective_masks=effective_masks,
    )


__all__ = ["SoftDiscGeometryOutput", "soft_disc_geometry_from_rgb"]
