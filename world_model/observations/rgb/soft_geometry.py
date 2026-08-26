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
import torch.nn.functional as F
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


@dataclass(frozen=True)
class SoftPhotometricRadiusOutput:
    """Differentiable finite-difference Gauss--Newton disc geometry.

    ``stage_fit_mse`` includes the fit before each of four updates followed by
    the final fit.  Boolean masks are diagnostics only and never replace the
    deployed continuous centre or radius.
    """

    centres: Tensor
    radius_pixels: Tensor
    fit_mse: Tensor
    confidence: Tensor
    valid_mask: Tensor
    stage_fit_mse: Tensor
    stage_trust_components: Tensor
    stage_max_trust_fraction: Tensor
    stage_damping: Tensor
    stage_fit_ratio: Tensor
    stage_monotonic_mask: Tensor
    support_fraction: Tensor
    support_valid_mask: Tensor
    all_stage_finite_mask: Tensor
    final_fit_valid_mask: Tensor


PHOTOMETRIC_CENTRE_TRUST_STEPS_PIXELS = (0.5, 0.25, 0.125, 0.0625)
PHOTOMETRIC_RADIUS_SCALE_BRACKET = (0.8, 1.2)
PHOTOMETRIC_CANDIDATES_PER_STAGE = 7
PHOTOMETRIC_EDGE_TEMPERATURE_PIXELS = 1.0e-3
PHOTOMETRIC_RADIAL_TEMPERATURE = 1.0e-3
PHOTOMETRIC_MAXIMUM_FIT_RMS = 0.035
PHOTOMETRIC_DAMPING_FORMULA = "sqrt(dtype_epsilon)*mean(diag(J^T_J))+dtype_epsilon"
PHOTOMETRIC_TRUST_TRANSFORM = "componentwise_tanh"
PHOTOMETRIC_RESIDUAL_NORMALIZATION = "sqrt(3*height*width)"


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
        (
            torch.sigmoid((foreground_strength - threshold_tensor) / temperature_tensor)
            - background_probability
        )
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
    circularity = (4.0 * covariance_determinant / (radial_second_moment.square() + epsilon)).clamp(
        0.0, 1.0
    )
    edge_clearance = (
        torch.stack(
            (
                centre_x_pixels,
                width - 1 - centre_x_pixels,
                centre_y_pixels,
                height - 1 - centre_y_pixels,
            ),
            dim=-1,
        ).amin(dim=-1)
        - radius_pixels
    )
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


def _smooth_clip(
    value: Tensor,
    lower: float,
    upper: float,
    temperature: float,
) -> Tensor:
    """Smoothly approximate ``value.clamp(lower, upper)``."""

    scale = value.new_tensor(temperature)
    return lower + scale * (
        F.softplus((value - lower) / scale) - F.softplus((value - upper) / scale)
    )


def soft_photometric_disc_radius(
    image: Tensor,
    centres: Tensor,
    proposal_radius_pixels: Tensor,
) -> SoftPhotometricRadiusOutput:
    """Jointly refine disc centre and radius with differentiable Gauss--Newton.

    The initial geometry comes from soft RGB moments.  Four fixed trust-region
    stages operate on theta = (centre_x_pixels, centre_y_pixels, log_radius).
    Each stage renders theta and symmetric positive/negative perturbations for
    all three coordinates, forms a finite-difference Jacobian, and applies a
    damped Gauss--Newton step through torch.linalg.solve.  A componentwise tanh
    keeps each update inside its declared trust box.

    The renderer surrogate follows the public one-pixel silhouette and radial
    shading equations.  Row-wise background is measured from RGB and RGB
    albedo is eliminated analytically for every candidate.  Residuals always
    use the same full-frame normalization.  There is no argmin, learned
    calibrator, candidate-dependent support, detached replacement, or label
    input in the deployed path.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("soft photometric geometry image must have shape [B,3,H,W]")
    if not image.is_floating_point():
        raise TypeError("soft photometric geometry image must be floating point")
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("soft photometric geometry requires dimensions of at least two pixels")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("soft photometric geometry image contains NaN or Inf")
    if centres.ndim != 3 or centres.shape[-1] != 2:
        raise ValueError("centres must have shape [B,S,2]")
    if proposal_radius_pixels.shape != centres.shape[:2]:
        raise ValueError("proposal_radius_pixels must have shape [B,S]")
    if centres.shape[0] != image.shape[0]:
        raise ValueError("image and geometry batch dimensions must match")
    if centres.device != image.device or proposal_radius_pixels.device != image.device:
        raise ValueError("image, centres, and proposal radii must share a device")
    if not centres.is_floating_point() or not proposal_radius_pixels.is_floating_point():
        raise TypeError("centres and proposal radii must be floating point")
    if not bool(torch.isfinite(centres).all()) or not bool(
        torch.isfinite(proposal_radius_pixels).all()
    ):
        raise ValueError("centres and proposal radii must be finite")
    if bool(proposal_radius_pixels.le(0.0).any()):
        raise ValueError("proposal radii must be positive")

    batch, _, height, width = image.shape
    dtype = image.dtype
    device = image.device
    dtype_epsilon = torch.finfo(dtype).eps
    residual_dimension = 3 * height * width
    residual_normalizer = math.sqrt(residual_dimension)

    proposal_centre_x = 0.5 * (centres[..., 0] + 1.0) * (width - 1)
    proposal_centre_y = 0.5 * (centres[..., 1] + 1.0) * (height - 1)
    initial_theta = torch.stack(
        (
            proposal_centre_x,
            proposal_centre_y,
            proposal_radius_pixels.to(dtype=dtype).log(),
        ),
        dim=-1,
    )
    theta = initial_theta

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    row_background = image.median(dim=-1, keepdim=True).values
    observed = image[:, None, None]
    background = row_background[:, None, None]

    def render_residual(candidate_theta: Tensor) -> Tensor:
        """Return normalized full-frame residuals for [B,S,C,3] candidates."""

        centre_x = candidate_theta[..., 0, None, None]
        centre_y = candidate_theta[..., 1, None, None]
        radius = candidate_theta[..., 2].exp()[..., None, None]
        delta_x = pixel_x - centre_x
        delta_y = pixel_y - centre_y
        distance = (delta_x.square() + delta_y.square() + dtype_epsilon).sqrt()
        signed_edge_distance = radius - distance

        edge_ramp = _smooth_clip(
            signed_edge_distance + 0.5,
            0.0,
            1.0,
            PHOTOMETRIC_EDGE_TEMPERATURE_PIXELS,
        )
        inside_probability = torch.sigmoid(
            (signed_edge_distance + 5.0 * PHOTOMETRIC_EDGE_TEMPERATURE_PIXELS)
            / PHOTOMETRIC_EDGE_TEMPERATURE_PIXELS
        )
        alpha = (edge_ramp * inside_probability).clamp(0.0, 1.0)
        radial_squared = distance.square() / radius.square().clamp_min(1.0e-8)
        radial_support = radial_squared.new_tensor(PHOTOMETRIC_RADIAL_TEMPERATURE) * F.softplus(
            (1.0 - radial_squared) / PHOTOMETRIC_RADIAL_TEMPERATURE
        )
        front_shape = (radial_support + dtype_epsilon).sqrt()
        shading_coefficient = alpha * (0.48 + 0.52 * front_shape)

        base = background * (1.0 - alpha.unsqueeze(3))
        coefficient_rgb = shading_coefficient.unsqueeze(3)
        denominator = shading_coefficient.square().sum(dim=(-2, -1)).clamp_min(1.0e-8)
        albedo = (coefficient_rgb * (observed - base)).sum(dim=(-2, -1)) / denominator.unsqueeze(-1)
        albedo = albedo.clamp(0.0, 1.0)
        rendered = base + albedo[..., None, None] * coefficient_rgb
        raw_residual = rendered - observed
        return raw_residual.flatten(start_dim=-3) / residual_normalizer

    unit_perturbations = image.new_tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    identity = torch.eye(3, dtype=dtype, device=device)
    log_radius_step = max(
        abs(math.log(PHOTOMETRIC_RADIUS_SCALE_BRACKET[0])),
        abs(math.log(PHOTOMETRIC_RADIUS_SCALE_BRACKET[1])),
    )

    stage_fit_values: list[Tensor] = []
    stage_trust_values: list[Tensor] = []
    stage_damping_values: list[Tensor] = []
    stage_finite_values: list[Tensor] = []
    for stage_index, centre_step in enumerate(PHOTOMETRIC_CENTRE_TRUST_STEPS_PIXELS):
        stage_scale = image.new_tensor(
            (
                centre_step,
                centre_step,
                log_radius_step / (2**stage_index),
            )
        )
        candidate_theta = theta.unsqueeze(-2) + unit_perturbations * stage_scale
        candidate_residual = render_residual(candidate_theta)
        centre_residual = candidate_residual[..., 0, :]
        jacobian = torch.stack(
            (
                0.5 * (candidate_residual[..., 1, :] - candidate_residual[..., 2, :]),
                0.5 * (candidate_residual[..., 3, :] - candidate_residual[..., 4, :]),
                0.5 * (candidate_residual[..., 5, :] - candidate_residual[..., 6, :]),
            ),
            dim=-1,
        )
        normal_matrix = torch.einsum(
            "...pi,...pj->...ij",
            jacobian,
            jacobian,
        )
        scaled_gradient = torch.einsum(
            "...pi,...p->...i",
            jacobian,
            centre_residual,
        )
        mean_diagonal = normal_matrix.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        damping = math.sqrt(dtype_epsilon) * mean_diagonal + dtype_epsilon
        damped_normal_matrix = normal_matrix + damping[..., None, None] * identity
        raw_step = torch.linalg.solve(
            damped_normal_matrix,
            -scaled_gradient.unsqueeze(-1),
        ).squeeze(-1)
        trust_components = torch.tanh(raw_step)
        updated_theta = theta + trust_components * stage_scale

        stage_fit_values.append(centre_residual.square().sum(dim=-1))
        stage_trust_values.append(trust_components)
        stage_damping_values.append(damping)
        stage_finite_values.append(
            torch.isfinite(candidate_residual).all(dim=(-1, -2))
            & torch.isfinite(normal_matrix).all(dim=(-1, -2))
            & torch.isfinite(scaled_gradient).all(dim=-1)
            & torch.isfinite(damping)
            & torch.isfinite(raw_step).all(dim=-1)
            & torch.isfinite(updated_theta).all(dim=-1)
        )
        theta = updated_theta

    final_residual = render_residual(theta.unsqueeze(-2)).squeeze(-2)
    final_fit_mse = final_residual.square().sum(dim=-1)
    stage_fit_mse = torch.stack((*stage_fit_values, final_fit_mse), dim=-1)
    stage_trust_components = torch.stack(stage_trust_values, dim=-2)
    stage_max_trust_fraction = stage_trust_components.abs().amax(dim=-1)
    stage_damping = torch.stack(stage_damping_values, dim=-1)
    stage_fit_ratio = stage_fit_mse[..., 1:] / stage_fit_mse[..., :-1].clamp_min(dtype_epsilon**2)
    monotonic_tolerance = (
        stage_fit_mse[..., :-1] * residual_dimension * dtype_epsilon + dtype_epsilon**2
    )
    stage_monotonic_mask = stage_fit_mse[..., 1:] <= stage_fit_mse[..., :-1] + monotonic_tolerance

    final_centre_pixels = theta[..., :2]
    fitted_centres = torch.stack(
        (
            2.0 * final_centre_pixels[..., 0] / (width - 1) - 1.0,
            2.0 * final_centre_pixels[..., 1] / (height - 1) - 1.0,
        ),
        dim=-1,
    )
    fitted_radius = theta[..., 2].exp()
    correction = theta - initial_theta
    negative_log_support = abs(math.log(PHOTOMETRIC_RADIUS_SCALE_BRACKET[0]))
    positive_log_support = math.log(PHOTOMETRIC_RADIUS_SCALE_BRACKET[1])
    radius_support = torch.where(
        correction[..., 2] < 0.0,
        correction[..., 2].new_tensor(negative_log_support),
        correction[..., 2].new_tensor(positive_log_support),
    )
    support_fraction = torch.stack(
        (
            correction[..., 0].abs() / PHOTOMETRIC_CENTRE_TRUST_STEPS_PIXELS[0],
            correction[..., 1].abs() / PHOTOMETRIC_CENTRE_TRUST_STEPS_PIXELS[0],
            correction[..., 2].abs() / radius_support,
        ),
        dim=-1,
    )
    support_valid_mask = support_fraction.le(1.0).all(dim=-1)
    final_finite = (
        torch.isfinite(final_residual).all(dim=-1)
        & torch.isfinite(theta).all(dim=-1)
        & torch.isfinite(final_fit_mse)
    )
    all_stage_finite_mask = torch.stack((*stage_finite_values, final_finite), dim=-1).all(
        dim=-1
    ) & torch.isfinite(stage_fit_mse).all(dim=-1)
    final_fit_valid_mask = final_fit_mse.sqrt() <= PHOTOMETRIC_MAXIMUM_FIT_RMS
    valid_mask = all_stage_finite_mask & final_fit_valid_mask
    fit_confidence = torch.exp(
        -final_fit_mse / (PHOTOMETRIC_MAXIMUM_FIT_RMS * PHOTOMETRIC_MAXIMUM_FIT_RMS)
    )
    trust_confidence = (1.0 - stage_max_trust_fraction[..., -1]).clamp(0.0, 1.0)
    confidence = (fit_confidence * trust_confidence).clamp(0.0, 1.0)

    return SoftPhotometricRadiusOutput(
        centres=fitted_centres,
        radius_pixels=fitted_radius,
        fit_mse=final_fit_mse,
        confidence=confidence,
        valid_mask=valid_mask,
        stage_fit_mse=stage_fit_mse,
        stage_trust_components=stage_trust_components,
        stage_max_trust_fraction=stage_max_trust_fraction,
        stage_damping=stage_damping,
        stage_fit_ratio=stage_fit_ratio,
        stage_monotonic_mask=stage_monotonic_mask,
        support_fraction=support_fraction,
        support_valid_mask=support_valid_mask,
        all_stage_finite_mask=all_stage_finite_mask,
        final_fit_valid_mask=final_fit_valid_mask,
    )


__all__ = [
    "SoftDiscGeometryOutput",
    "SoftPhotometricRadiusOutput",
    "soft_disc_geometry_from_rgb",
    "soft_photometric_disc_radius",
]
