"""Differentiable bounded identification of constant linear drag.

The estimator profiles out anchor position and velocity with the public exact
free-motion fitter on a fixed coarse quadrature grid in log-drag, followed by
a bounded adaptive local quadrature.  A fixed broad uniform-in-log prior and a
declared metric position-noise floor turn the profile residuals into a smooth
posterior.  The deployed state is then refit at the posterior-mean log-drag,
so no hard grid winner owns the result.

This module owns no learned parameters, buffers, simulator labels, or custom
autograd path.  It is a pure tensor function over observable position history,
timestamps, public gravity, fixed bounds, and optional support/confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.dynamics.free_motion_fit import fit_free_motion


@dataclass(frozen=True)
class AnalyticDragFitResult:
    """Anchor-state fit, drag posterior, and identifiability diagnostics.

    Parameter-valued outputs ``log_drag`` and ``raw_log_drag_variance`` have
    shape ``[B,S,1]``.  Scalar diagnostics and ``valid`` have shape ``[B,S]``.
    ``excitation`` is the weighted RMS Euclidean displacement from the refit
    anchor position, in metres.  ``profile_information`` is the non-negative
    posterior precision gain over the fixed uniform-grid prior, in inverse
    squared log-drag units.  ``boundary_mass`` is posterior probability on the
    two endpoint nodes.

    Invalid rows have exactly zero inferred state, drag, prediction, residual,
    and covariance outputs.  Support, conditioning, excitation, information,
    and boundary diagnostics remain finite and truthful for auditability.
    ``raw_log_drag_variance`` includes the adaptive local grid-cell variance
    and is an uncalibrated estimator diagnostic, not a claimed universal
    posterior.
    """

    position: Tensor
    velocity: Tensor
    log_drag: Tensor
    raw_log_drag_variance: Tensor
    anchor_time: Tensor
    predicted_positions: Tensor
    residuals: Tensor
    residual_covariance: Tensor
    condition_number: Tensor
    support_count: Tensor
    support_weight: Tensor
    excitation: Tensor
    profile_information: Tensor
    boundary_mass: Tensor
    valid: Tensor

    @property
    def anchor_position(self) -> Tensor:
        """Alias spelling out that ``position`` is evaluated at the anchor."""

        return self.position

    @property
    def anchor_velocity(self) -> Tensor:
        """Alias spelling out that ``velocity`` is evaluated at the anchor."""

        return self.velocity


def _require_floating_tensor(name: str, value: Tensor, positions: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.dtype != positions.dtype or value.device != positions.device:
        raise ValueError(f"{name} must have the same dtype and device as positions")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _normalise_timestamps(timestamps: Tensor, positions: Tensor) -> Tensor:
    _require_floating_tensor("timestamps", timestamps, positions)
    batch, time, slots = positions.shape[:3]
    if timestamps.shape == (batch, time):
        return timestamps[:, :, None].expand(batch, time, slots)
    if timestamps.shape == (batch, time, slots):
        return timestamps
    raise ValueError("timestamps must have shape [B,T] or [B,T,S]")


def _normalise_anchor_time(
    anchor_time: float | Tensor | None,
    timestamps: Tensor,
    positions: Tensor,
) -> Tensor:
    batch, slots = positions.shape[0], positions.shape[2]
    if anchor_time is None:
        return timestamps[:, -1, :]
    value = torch.as_tensor(anchor_time, dtype=positions.dtype, device=positions.device)
    if value.ndim == 0:
        value = value.expand(batch, slots)
    elif value.shape == (batch,):
        value = value[:, None].expand(batch, slots)
    elif value.shape != (batch, slots):
        raise ValueError("anchor_time must be scalar, [B], or [B,S]")
    if not torch.isfinite(value).all():
        raise ValueError("anchor_time must contain only finite values")
    return value


def _normalise_gravity(gravity: Tensor, positions: Tensor) -> Tensor:
    _require_floating_tensor("gravity", gravity, positions)
    batch, slots = positions.shape[0], positions.shape[2]
    if gravity.shape == (3,):
        return gravity[None, None, :].expand(batch, slots, 3)
    if gravity.shape == (batch, 3):
        return gravity[:, None, :].expand(batch, slots, 3)
    if gravity.shape == (batch, slots, 3):
        return gravity
    raise ValueError("gravity must have shape [3], [B,3], or [B,S,3]")


def _normalise_support(support: Tensor | None, positions: Tensor) -> Tensor:
    shape = positions.shape[:3]
    if support is None:
        return torch.ones(shape, dtype=torch.bool, device=positions.device)
    if not isinstance(support, Tensor):
        raise TypeError("support must be a torch.Tensor")
    if support.dtype != torch.bool:
        raise TypeError("support must have boolean dtype; use weights for confidence")
    if support.device != positions.device:
        raise ValueError("support must be on the same device as positions")
    if support.shape != shape:
        raise ValueError("support must have shape [B,T,S]")
    return support


def _normalise_weights(weights: Tensor | None, positions: Tensor) -> Tensor:
    shape = positions.shape[:3]
    if weights is None:
        return torch.ones(shape, dtype=positions.dtype, device=positions.device)
    _require_floating_tensor("weights", weights, positions)
    if weights.shape != shape:
        raise ValueError("weights must have shape [B,T,S]")
    if torch.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    return weights


def _require_real_scalar(name: str, value: float, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        comparator = "positive" if minimum > 0 else "nonnegative"
        raise ValueError(f"{name} must be finite and {comparator}")
    return result


def fit_free_motion_with_drag(
    positions: Tensor,
    timestamps: Tensor,
    *,
    gravity: Tensor,
    drag_bounds: tuple[float, float] = (0.01, 0.36),
    anchor_time: float | Tensor | None = None,
    support: Tensor | None = None,
    weights: Tensor | None = None,
    grid_points: int = 257,
    position_noise_floor: float = 2.0e-5,
    minimum_support: int = 3,
    conditioning_limit: float = 1.0e8,
    minimum_excitation: float = 0.015,
    maximum_boundary_mass: float = 0.01,
    minimum_profile_information: float = 1.0,
) -> AnalyticDragFitResult:
    """Identify per-slot constant drag by bounded differentiable profiling.

    The fixed coarse grid is uniform in log-drag and uses trapezoidal
    quadrature weights.  Its smooth posterior mean and variance define a
    bounded same-sized local grid, whose posterior owns the returned moments.
    For every node, :func:`fit_free_motion` exactly profiles out anchor position
    and velocity.  Weighted residual SSE and the declared
    ``position_noise_floor`` define the likelihood.  Both grid construction
    and posterior moments remain ordinary Torch expressions, and a final exact
    fit at the posterior mean defines the returned state and trajectory.

    ``drag_bounds`` are positive physical coefficients in inverse seconds.
    Every supported continuous input is differentiable.  Boolean validity
    decisions fail closed and never create a surrogate gradient.
    """

    if not isinstance(positions, Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if not positions.is_floating_point():
        raise TypeError("positions must have a floating-point dtype")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must use float32 or float64")
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,T,S,3]")
    if min(positions.shape[:3]) <= 0:
        raise ValueError("positions batch, time, and slot dimensions must be nonempty")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must contain only finite values")

    if (
        not isinstance(drag_bounds, tuple)
        or len(drag_bounds) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in drag_bounds
        )
    ):
        raise TypeError("drag_bounds must be a two-element tuple of real scalars")
    lower_drag, upper_drag = (float(value) for value in drag_bounds)
    if not math.isfinite(lower_drag) or not math.isfinite(upper_drag):
        raise ValueError("drag_bounds must be finite")
    if lower_drag <= 0 or upper_drag <= lower_drag:
        raise ValueError("drag_bounds must satisfy 0 < lower < upper")
    if isinstance(grid_points, bool) or not isinstance(grid_points, int):
        raise TypeError("grid_points must be an integer")
    if grid_points < 3:
        raise ValueError("grid_points must be at least three")
    if isinstance(minimum_support, bool) or not isinstance(minimum_support, int):
        raise TypeError("minimum_support must be an integer")
    if minimum_support < 3:
        raise ValueError("minimum_support must be at least three for drag identification")
    dtype_limits = torch.finfo(positions.dtype)
    noise_floor = _require_real_scalar(
        "position_noise_floor",
        position_noise_floor,
        minimum=math.sqrt(dtype_limits.tiny),
    )
    if noise_floor > math.sqrt(dtype_limits.max):
        raise ValueError("position_noise_floor is too large for the positions dtype")
    condition_limit = _require_real_scalar(
        "conditioning_limit",
        conditioning_limit,
        minimum=1.0 + torch.finfo(positions.dtype).eps,
    )
    excitation_floor = _require_real_scalar("minimum_excitation", minimum_excitation, minimum=0.0)
    boundary_limit = _require_real_scalar(
        "maximum_boundary_mass", maximum_boundary_mass, minimum=0.0
    )
    if boundary_limit > 1.0:
        raise ValueError("maximum_boundary_mass must be at most one")
    information_floor = _require_real_scalar(
        "minimum_profile_information", minimum_profile_information, minimum=0.0
    )

    times = _normalise_timestamps(timestamps, positions)
    anchor = _normalise_anchor_time(anchor_time, times, positions)
    acceleration = _normalise_gravity(gravity, positions)
    admissible = _normalise_support(support, positions)
    confidence = _normalise_weights(weights, positions)
    batch, time, slots = positions.shape[:3]

    # Protect the inner exact solve from finite-but-destructive magnitudes and
    # backward exponential overflow.  Ordinary world-scale rows are untouched;
    # rejected rows are replaced only on the branch that is later invalidated.
    numeric_limit = float(dtype_limits.max**0.125)
    exponent_limit = 0.25 * math.log(dtype_limits.max)
    relative_time = times - anchor[:, None, :]
    numeric_valid = (
        (positions.abs().amax(dim=(1, 3)) <= numeric_limit)
        & (times.abs().amax(dim=1) <= numeric_limit)
        & (anchor.abs() <= numeric_limit)
        & (acceleration.abs().amax(dim=-1) <= numeric_limit)
        & (confidence.abs().amax(dim=1) <= numeric_limit)
        & (upper_drag * relative_time.abs().amax(dim=1) <= exponent_limit)
    )
    safe_positions = torch.where(
        numeric_valid[:, None, :, None], positions, torch.zeros_like(positions)
    )
    safe_times = torch.where(numeric_valid[:, None, :], times, torch.zeros_like(times))
    safe_anchor = torch.where(numeric_valid, anchor, torch.zeros_like(anchor))
    safe_acceleration = torch.where(
        numeric_valid[..., None], acceleration, torch.zeros_like(acceleration)
    )
    safe_support = admissible & numeric_valid[:, None, :]
    safe_confidence = torch.where(
        numeric_valid[:, None, :], confidence, torch.zeros_like(confidence)
    )

    lower_log = math.log(lower_drag)
    upper_log = math.log(upper_drag)
    log_grid = torch.linspace(
        lower_log,
        upper_log,
        grid_points,
        dtype=positions.dtype,
        device=positions.device,
    )
    effective_weight = safe_confidence * safe_support.to(dtype=positions.dtype)
    noise_variance = positions.new_tensor(noise_floor).square()

    def profile_at_log_grid(candidate_log_grid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if candidate_log_grid.ndim == 1:
            candidate_count = candidate_log_grid.shape[0]
            candidate_log_drag = candidate_log_grid[None, :, None].expand(
                batch,
                candidate_count,
                slots,
            )
        elif candidate_log_grid.ndim == 3 and candidate_log_grid.shape[:2] == (batch, slots):
            candidate_count = candidate_log_grid.shape[-1]
            candidate_log_drag = candidate_log_grid.permute(0, 2, 1)
        else:
            raise ValueError("candidate log grid must have shape [G] or [B,S,G]")
        candidate_positions = (
            safe_positions[:, None]
            .expand(batch, candidate_count, time, slots, 3)
            .reshape(batch * candidate_count, time, slots, 3)
        )
        candidate_times = (
            safe_times[:, None]
            .expand(batch, candidate_count, time, slots)
            .reshape(batch * candidate_count, time, slots)
        )
        candidate_gravity = (
            safe_acceleration[:, None]
            .expand(batch, candidate_count, slots, 3)
            .reshape(batch * candidate_count, slots, 3)
        )
        candidate_anchor = (
            safe_anchor[:, None]
            .expand(batch, candidate_count, slots)
            .reshape(batch * candidate_count, slots)
        )
        candidate_support = (
            safe_support[:, None]
            .expand(batch, candidate_count, time, slots)
            .reshape(batch * candidate_count, time, slots)
        )
        candidate_weights = (
            safe_confidence[:, None]
            .expand(batch, candidate_count, time, slots)
            .reshape(batch * candidate_count, time, slots)
        )
        candidate_drag = candidate_log_drag.exp().reshape(batch * candidate_count, slots)
        candidate_fit = fit_free_motion(
            candidate_positions,
            candidate_times,
            gravity=candidate_gravity,
            drag=candidate_drag,
            anchor_time=candidate_anchor,
            support=candidate_support,
            weights=candidate_weights,
            minimum_support=minimum_support,
            conditioning_limit=condition_limit,
        )
        candidate_residuals = candidate_fit.residuals.reshape(
            batch,
            candidate_count,
            time,
            slots,
            3,
        )
        profile_sse = torch.einsum(
            "bgtsc,bts->bgs",
            candidate_residuals.square(),
            effective_weight,
        ).permute(0, 2, 1)
        candidate_valid = candidate_fit.valid.reshape(
            batch,
            candidate_count,
            slots,
        ).permute(0, 2, 1)
        candidate_valid = candidate_valid & torch.isfinite(profile_sse)
        candidate_condition = candidate_fit.condition_number.reshape(
            batch,
            candidate_count,
            slots,
        ).permute(0, 2, 1)
        return profile_sse, candidate_valid, candidate_condition

    def posterior_on_log_grid(
        profile_sse: Tensor,
        candidate_valid: Tensor,
        candidate_log_grid: Tensor,
        *,
        fallback_mean: Tensor,
        fallback_variance: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        candidate_count = profile_sse.shape[-1]
        if candidate_log_grid.ndim == 1:
            grid_value = candidate_log_grid[None, None, :]
        else:
            grid_value = candidate_log_grid
        maximum_sse = torch.full_like(profile_sse, torch.finfo(positions.dtype).max)
        masked_sse = torch.where(candidate_valid, profile_sse, maximum_sse)
        minimum_sse = masked_sse.amin(dim=-1, keepdim=True)
        profile_delta = torch.where(
            candidate_valid,
            (profile_sse - minimum_sse).clamp_min(0.0),
            torch.zeros_like(profile_sse),
        )
        relative_log_likelihood = -0.5 * profile_delta / noise_variance
        quadrature_weight = torch.ones(
            candidate_count,
            dtype=positions.dtype,
            device=positions.device,
        )
        quadrature_weight[0] = 0.5
        quadrature_weight[-1] = 0.5
        unnormalised_probability = (
            quadrature_weight[None, None, :]
            * torch.exp(relative_log_likelihood)
            * candidate_valid.to(dtype=positions.dtype)
        )
        probability_sum = unnormalised_probability.sum(dim=-1, keepdim=True)
        posterior_probability = unnormalised_probability / torch.where(
            probability_sum > 0,
            probability_sum,
            torch.ones_like(probability_sum),
        )
        posterior_available = probability_sum.squeeze(-1) > 0
        raw_mean = (posterior_probability * grid_value).sum(dim=-1)
        mean = torch.where(posterior_available, raw_mean, fallback_mean)
        grid_cell_variance = (
            (grid_value[..., -1] - grid_value[..., 0]) / (candidate_count - 1)
        ).square() / 12.0
        raw_variance = (posterior_probability * (grid_value - mean[..., None]).square()).sum(
            dim=-1
        ) + grid_cell_variance
        variance = torch.where(
            posterior_available,
            raw_variance,
            fallback_variance,
        )
        return posterior_probability, posterior_available, mean, variance

    quadrature_weight = torch.ones_like(log_grid)
    quadrature_weight[0] = 0.5
    quadrature_weight[-1] = 0.5
    prior_probability = quadrature_weight / quadrature_weight.sum()
    prior_mean = (prior_probability * log_grid).sum()
    coarse_grid_cell_variance = positions.new_tensor(
        ((upper_log - lower_log) / (grid_points - 1)) ** 2 / 12.0
    )
    prior_variance = (
        prior_probability * (log_grid - prior_mean).square()
    ).sum() + coarse_grid_cell_variance

    coarse_sse, coarse_valid, coarse_condition = profile_at_log_grid(log_grid)
    coarse_probability, coarse_available, coarse_mean, coarse_variance = posterior_on_log_grid(
        coarse_sse,
        coarse_valid,
        log_grid,
        fallback_mean=prior_mean,
        fallback_variance=prior_variance,
    )

    coarse_step = positions.new_tensor((upper_log - lower_log) / (grid_points - 1))
    coarse_standard_deviation = coarse_variance.clamp_min(dtype_limits.tiny).sqrt()
    local_half_width = torch.maximum(
        4.0 * coarse_standard_deviation,
        2.0 * coarse_step,
    )
    local_lower = (coarse_mean - local_half_width).clamp(min=lower_log, max=upper_log)
    local_upper = (coarse_mean + local_half_width).clamp(min=lower_log, max=upper_log)
    local_fraction = torch.linspace(
        0.0,
        1.0,
        grid_points,
        dtype=positions.dtype,
        device=positions.device,
    )
    local_log_grid = (
        local_lower[..., None] + (local_upper - local_lower)[..., None] * local_fraction
    )
    local_sse, local_valid, local_condition = profile_at_log_grid(local_log_grid)
    _, local_available, mean_log_drag, raw_log_drag_variance = posterior_on_log_grid(
        local_sse,
        local_valid,
        local_log_grid,
        fallback_mean=prior_mean,
        fallback_variance=prior_variance,
    )
    posterior_available = coarse_available & local_available
    variance_contraction = prior_variance - raw_log_drag_variance
    contraction_tolerance = 16.0 * torch.finfo(positions.dtype).eps * prior_variance
    profile_information = torch.where(
        variance_contraction > contraction_tolerance,
        variance_contraction
        / (raw_log_drag_variance * prior_variance).clamp_min(torch.finfo(positions.dtype).tiny),
        torch.zeros_like(variance_contraction),
    )
    prior_boundary_mass = prior_probability[0] + prior_probability[-1]
    boundary_mass = torch.where(
        coarse_available,
        coarse_probability[..., 0] + coarse_probability[..., -1],
        prior_boundary_mass,
    )

    final_fit = fit_free_motion(
        safe_positions,
        safe_times,
        gravity=safe_acceleration,
        drag=mean_log_drag.exp(),
        anchor_time=safe_anchor,
        support=safe_support,
        weights=safe_confidence,
        minimum_support=minimum_support,
        conditioning_limit=condition_limit,
    )
    excitation_energy = (
        effective_weight[..., None] * (safe_positions - final_fit.position[:, None, :, :]).square()
    ).sum(dim=(1, 3)) / effective_weight.sum(dim=1).clamp_min(torch.finfo(positions.dtype).eps)
    excitation_epsilon = positions.new_tensor(torch.finfo(positions.dtype).eps)
    excitation = torch.sqrt(excitation_energy.clamp_min(0.0) + excitation_epsilon)
    excitation = excitation - torch.sqrt(excitation_epsilon)
    excitation = torch.where(
        final_fit.valid,
        excitation,
        torch.zeros_like(excitation),
    )

    condition_number = torch.maximum(
        torch.maximum(coarse_condition.amax(dim=-1), local_condition.amax(dim=-1)),
        final_fit.condition_number,
    )
    all_candidates_valid = coarse_valid.all(dim=-1) & local_valid.all(dim=-1)
    valid = (
        numeric_valid
        & posterior_available
        & all_candidates_valid
        & final_fit.valid
        & (excitation >= excitation_floor)
        & (boundary_mass <= boundary_limit)
        & (profile_information >= information_floor)
        & torch.isfinite(raw_log_drag_variance)
        & torch.isfinite(profile_information)
        & torch.isfinite(boundary_mass)
    )

    state_mask = valid[..., None]
    history_mask = valid[:, None, :, None]
    covariance_mask = valid[..., None, None]
    result = AnalyticDragFitResult(
        position=torch.where(state_mask, final_fit.position, torch.zeros_like(final_fit.position)),
        velocity=torch.where(state_mask, final_fit.velocity, torch.zeros_like(final_fit.velocity)),
        log_drag=torch.where(
            state_mask,
            mean_log_drag[..., None],
            torch.zeros_like(mean_log_drag[..., None]),
        ),
        raw_log_drag_variance=torch.where(
            state_mask,
            raw_log_drag_variance[..., None],
            torch.zeros_like(raw_log_drag_variance[..., None]),
        ),
        anchor_time=anchor,
        predicted_positions=torch.where(
            history_mask,
            final_fit.predicted_positions,
            torch.zeros_like(final_fit.predicted_positions),
        ),
        residuals=torch.where(
            history_mask,
            final_fit.residuals,
            torch.zeros_like(final_fit.residuals),
        ),
        residual_covariance=torch.where(
            covariance_mask,
            final_fit.residual_covariance,
            torch.zeros_like(final_fit.residual_covariance),
        ),
        condition_number=condition_number,
        support_count=final_fit.support_count,
        support_weight=final_fit.support_weight,
        excitation=excitation,
        profile_information=profile_information,
        boundary_mass=boundary_mass,
        valid=valid,
    )
    continuous_outputs = (
        result.position,
        result.velocity,
        result.log_drag,
        result.raw_log_drag_variance,
        result.anchor_time,
        result.predicted_positions,
        result.residuals,
        result.residual_covariance,
        result.condition_number,
        result.support_weight,
        result.excitation,
        result.profile_information,
        result.boundary_mass,
    )
    if not all(torch.isfinite(value).all() for value in continuous_outputs):
        raise ValueError("analytic drag fit produced a nonfinite output")
    return result


__all__ = ["AnalyticDragFitResult", "fit_free_motion_with_drag"]
