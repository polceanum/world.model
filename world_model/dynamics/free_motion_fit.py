"""Differentiable closed-form fitting for collision-free linear-drag motion.

The fitter estimates position and velocity at a caller-visible anchor time from
timestamped position measurements.  It is intentionally a small equation-led
component: known gravity and per-slot drag define the temporal basis, and a
batched ``2 x 2`` weighted normal equation estimates only the anchor state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FreeMotionFitResult:
    """Anchor-state fit and transparent per-slot diagnostics.

    ``position`` and ``velocity`` are the fitted state at ``anchor_time``.
    Invalid rows have finite zero-valued state, prediction, residual, and
    covariance outputs and must be selected using ``valid``.  Support and
    conditioning fields remain truthful diagnostics.  ``normal_matrix`` uses
    per-row sum-normalized weights, while ``support_weight`` retains their
    original total scale.  ``residual_covariance`` is the weighted residual
    second moment (maximum-likelihood convention), not a
    degrees-of-freedom-corrected sample estimate.
    """

    position: Tensor
    velocity: Tensor
    anchor_time: Tensor
    predicted_positions: Tensor
    residuals: Tensor
    residual_covariance: Tensor
    normal_matrix: Tensor
    condition_number: Tensor
    support_count: Tensor
    support_weight: Tensor
    valid: Tensor

    @property
    def anchor_position(self) -> Tensor:
        """Alias spelling out that ``position`` is evaluated at the anchor."""

        return self.position

    @property
    def anchor_velocity(self) -> Tensor:
        """Alias spelling out that ``velocity`` is evaluated at the anchor."""

        return self.velocity


def _require_floating_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _require_same_tensor_context(name: str, value: Tensor, reference: Tensor) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} must be on the same device as positions")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} must have the same dtype as positions")


def _linear_drag_basis(delta_time: Tensor, drag: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return displacement bases ``A``, ``B`` and velocity decay.

    For ``dv/dt = gravity - drag * v`` and signed elapsed time ``t``::

        x(t) = x(0) + A(t) v(0) + B(t) gravity
        v(t) = exp(-drag * t) v(0) + A(t) gravity

    The polynomial branch is the exact Taylor limit around ``drag * t = 0``.
    It avoids both division by zero and the cancellation in
    ``z + expm1(-z)`` while retaining derivatives with respect to drag and
    time at zero.
    """

    z = drag * delta_time
    # ``B`` contains the cancellation-prone difference
    # ``z - (1 - exp(-z))``.  Keeping both phi functions in a sufficiently
    # deep Horner series through |z|=0.5 makes the float32 path continuous
    # enough for long-horizon semigroup checks while the direct expression is
    # well conditioned outside that interval.
    small = z.abs() <= 0.5
    safe_z = torch.where(small, torch.ones_like(z), z)

    negative_z = -z
    series_order = 16
    a_taylor = torch.full_like(z, 1.0 / math.factorial(series_order + 1))
    b_taylor = torch.full_like(z, 1.0 / math.factorial(series_order + 2))
    for order in range(series_order - 1, -1, -1):
        a_taylor = a_taylor * negative_z + 1.0 / math.factorial(order + 1)
        b_taylor = b_taylor * negative_z + 1.0 / math.factorial(order + 2)

    negative_expm1 = -torch.expm1(-safe_z)
    a_scaled = torch.where(small, a_taylor, negative_expm1 / safe_z)
    b_scaled = torch.where(
        small,
        b_taylor,
        (safe_z - negative_expm1) / safe_z.square(),
    )
    a = delta_time * a_scaled
    b = delta_time.square() * b_scaled
    decay = torch.exp(-z)
    if not all(torch.isfinite(value).all() for value in (a, b, decay)):
        raise ValueError("drag and elapsed time produce a nonfinite free-motion basis")
    return a, b, decay


def _normalise_step_time(delta_time: float | Tensor, position: Tensor) -> Tensor:
    batch, slots = position.shape[:2]
    value = torch.as_tensor(delta_time, dtype=position.dtype, device=position.device)
    if value.ndim == 0:
        value = value.expand(batch, slots)
    elif value.shape == (batch,):
        value = value[:, None].expand(batch, slots)
    elif value.shape != (batch, slots):
        raise ValueError("delta_time must be scalar, [B], or [B,S]")
    if not torch.isfinite(value).all():
        raise ValueError("delta_time must contain only finite values")
    return value


def _normalise_drag(drag: Tensor, position: Tensor) -> Tensor:
    _require_floating_tensor("drag", drag)
    _require_same_tensor_context("drag", drag, position)
    batch, slots = position.shape[:2]
    if drag.shape == (batch, slots, 1):
        value = drag.squeeze(-1)
    elif drag.shape == (batch, slots):
        value = drag
    else:
        raise ValueError("drag must have shape [B,S] or [B,S,1]")
    if torch.any(value < 0):
        raise ValueError("drag must be nonnegative")
    return value


def _normalise_gravity(gravity: Tensor, position: Tensor) -> Tensor:
    _require_floating_tensor("gravity", gravity)
    _require_same_tensor_context("gravity", gravity, position)
    batch, slots = position.shape[:2]
    if gravity.shape == (3,):
        return gravity[None, None, :].expand(batch, slots, 3)
    if gravity.shape == (batch, 3):
        return gravity[:, None, :].expand(batch, slots, 3)
    if gravity.shape == (batch, slots, 3):
        return gravity
    raise ValueError("gravity must have shape [3], [B,3], or [B,S,3]")


def free_motion_position_velocity(
    position: Tensor,
    velocity: Tensor,
    delta_time: float | Tensor,
    *,
    gravity: Tensor,
    drag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Propagate a ``[B,S,3]`` anchor state by signed elapsed seconds.

    This is the exact constant-gravity, linear-drag solution used by the fit.
    Signed time is supported so the same equation can generate histories and
    future rollouts.  All arithmetic remains in Torch and differentiable.
    """

    _require_floating_tensor("position", position)
    _require_floating_tensor("velocity", velocity)
    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError("position must have shape [B,S,3]")
    if velocity.shape != position.shape:
        raise ValueError("velocity must have the same [B,S,3] shape as position")
    _require_same_tensor_context("velocity", velocity, position)
    elapsed = _normalise_step_time(delta_time, position)
    coefficient = _normalise_drag(drag, position)
    acceleration = _normalise_gravity(gravity, position)
    a, b, decay = _linear_drag_basis(elapsed, coefficient)
    next_position = position + a[..., None] * velocity + b[..., None] * acceleration
    next_velocity = decay[..., None] * velocity + a[..., None] * acceleration
    if not torch.isfinite(next_position).all() or not torch.isfinite(next_velocity).all():
        raise ValueError("free-motion propagation produced a nonfinite state")
    return next_position, next_velocity


def _normalise_timestamps(timestamps: Tensor, positions: Tensor) -> Tensor:
    _require_floating_tensor("timestamps", timestamps)
    _require_same_tensor_context("timestamps", timestamps, positions)
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


def _normalise_support(
    support: Tensor | None,
    positions: Tensor,
) -> Tensor:
    shape = positions.shape[:3]
    if support is None:
        return torch.ones(shape, dtype=torch.bool, device=positions.device)
    if not isinstance(support, Tensor):
        raise TypeError("support must be a torch.Tensor")
    if support.dtype != torch.bool:
        raise TypeError("support must have boolean dtype; use weights for soft confidence")
    if support.device != positions.device:
        raise ValueError("support must be on the same device as positions")
    if support.shape != shape:
        raise ValueError("support must have shape [B,T,S]")
    return support


def _normalise_weights(weights: Tensor | None, positions: Tensor) -> Tensor:
    shape = positions.shape[:3]
    if weights is None:
        return torch.ones(shape, dtype=positions.dtype, device=positions.device)
    _require_floating_tensor("weights", weights)
    _require_same_tensor_context("weights", weights, positions)
    if weights.shape != shape:
        raise ValueError("weights must have shape [B,T,S]")
    if torch.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    return weights


def _normal_condition_number(normal_matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Return finite 2-norm condition number and the smaller eigenvalue."""

    diagonal_0 = normal_matrix[..., 0, 0]
    off_diagonal = normal_matrix[..., 0, 1]
    diagonal_1 = normal_matrix[..., 1, 1]
    trace = diagonal_0 + diagonal_1
    discriminant = (diagonal_0 - diagonal_1).square() + 4.0 * off_diagonal.square()
    epsilon = torch.finfo(normal_matrix.dtype).eps
    root = torch.sqrt(discriminant.clamp_min(0.0))
    eigenvalue_max = 0.5 * (trace + root)
    eigenvalue_min = 0.5 * (trace - root)
    spectral_scale = eigenvalue_max.abs()
    nonzero_scale = spectral_scale > 0
    spectral_floor = torch.where(
        nonzero_scale,
        epsilon * spectral_scale,
        torch.ones_like(spectral_scale),
    )
    resolved = spectral_scale / eigenvalue_min.clamp_min(spectral_floor)
    condition = torch.where(
        nonzero_scale,
        resolved.clamp_min(1.0),
        torch.full_like(resolved, torch.finfo(normal_matrix.dtype).max),
    )
    return condition, eigenvalue_min


def fit_free_motion(
    positions: Tensor,
    timestamps: Tensor,
    *,
    gravity: Tensor,
    drag: Tensor,
    anchor_time: float | Tensor | None = None,
    support: Tensor | None = None,
    weights: Tensor | None = None,
    minimum_support: int = 2,
    conditioning_limit: float = 1.0e8,
) -> FreeMotionFitResult:
    """Fit anchor position and velocity from timestamped positions.

    Args:
        positions: RGB-derived world positions with shape ``[B,T,S,3]``.
        timestamps: Seconds with shape ``[B,T]`` or asynchronous ``[B,T,S]``.
        gravity: Known constant acceleration as ``[3]``, ``[B,3]``, or
            ``[B,S,3]``.
        drag: Known nonnegative linear-drag coefficient as ``[B,S]`` or
            ``[B,S,1]``.
        anchor_time: State time as a scalar, ``[B]``, or ``[B,S]``.  The final
            provided timestamp is used by default; this choice does not inspect
            or discretely select from support.
        support: Boolean admissibility mask ``[B,T,S]``.
        weights: Differentiable nonnegative confidence ``[B,T,S]``.
        minimum_support: Minimum positive-weight supported observations.
        conditioning_limit: Maximum accepted normal-matrix 2-norm condition.

    Returns:
        A :class:`FreeMotionFitResult`.  Continuous valid-row outputs retain
        gradients to positions, timestamps, gravity, drag, anchor time, and
        weights.  Rows lacking enough independent temporal support are finite
        neutral values with ``valid=False``.
    """

    _require_floating_tensor("positions", positions)
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,T,S,3]")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must use float32 or float64 for the normal solve")
    if min(positions.shape[:3]) <= 0:
        raise ValueError("positions batch, time, and slot dimensions must be nonempty")
    if isinstance(minimum_support, bool) or not isinstance(minimum_support, int):
        raise TypeError("minimum_support must be an integer")
    if minimum_support < 2:
        raise ValueError("minimum_support must be at least two")
    if not math.isfinite(conditioning_limit) or conditioning_limit <= 1.0:
        raise ValueError("conditioning_limit must be finite and greater than one")

    times = _normalise_timestamps(timestamps, positions)
    anchor = _normalise_anchor_time(anchor_time, times, positions)
    admissible = _normalise_support(support, positions)
    confidence = _normalise_weights(weights, positions)

    batch, _, slots = positions.shape[:3]
    anchor_reference = positions.new_empty((batch, slots, 3))
    coefficient = _normalise_drag(drag, anchor_reference)
    acceleration = _normalise_gravity(gravity, anchor_reference)
    relative_time = times - anchor[:, None, :]
    a, b, _ = _linear_drag_basis(relative_time, coefficient[:, None, :])

    effective_weight = confidence * admissible.to(dtype=positions.dtype)
    support_weight = effective_weight.sum(dim=1)
    positive_weight = support_weight > 0
    weight_scale = torch.where(
        positive_weight,
        support_weight,
        torch.ones_like(support_weight),
    )
    solve_weight = effective_weight / weight_scale[:, None, :]
    corrected_position = positions - b[..., None] * acceleration[:, None, :, :]
    design = torch.stack((torch.ones_like(a), a), dim=-1)
    normal_matrix = torch.einsum(
        "btsk,bts,btsl->bskl",
        design,
        solve_weight,
        design,
    )
    right_hand_side = torch.einsum(
        "btsk,bts,btsc->bskc",
        design,
        solve_weight,
        corrected_position,
    )

    support_count = (admissible & (confidence > 0)).sum(dim=1)
    condition_number, minimum_eigenvalue = _normal_condition_number(normal_matrix)
    epsilon = torch.finfo(positions.dtype).eps
    eigenvalue_floor = epsilon * normal_matrix.diagonal(dim1=-2, dim2=-1).sum(-1).abs()
    valid = (
        (support_count >= minimum_support)
        & (support_weight > 0)
        & (minimum_eigenvalue > eigenvalue_floor)
        & (condition_number <= conditioning_limit)
    )

    identity = torch.eye(2, dtype=positions.dtype, device=positions.device)
    solve_matrix = torch.where(valid[..., None, None], normal_matrix, identity)
    solve_rhs = torch.where(valid[..., None, None], right_hand_side, 0.0)
    solution = torch.linalg.solve(solve_matrix, solve_rhs)
    fitted_position = solution[..., 0, :]
    fitted_velocity = solution[..., 1, :]

    raw_predicted_positions = (
        fitted_position[:, None, :, :]
        + a[..., None] * fitted_velocity[:, None, :, :]
        + b[..., None] * acceleration[:, None, :, :]
    )
    continuous_mask = valid[:, None, :, None]
    predicted_positions = torch.where(
        continuous_mask,
        raw_predicted_positions,
        torch.zeros_like(raw_predicted_positions),
    )
    residuals = torch.where(
        continuous_mask,
        positions - raw_predicted_positions,
        torch.zeros_like(positions),
    )
    residual_covariance = (
        torch.einsum(
            "btsc,bts,btsd->bscd",
            residuals,
            solve_weight,
            residuals,
        )
        / solve_weight.sum(dim=1).clamp_min(epsilon)[..., None, None]
    )
    residual_covariance = torch.where(
        valid[..., None, None],
        residual_covariance,
        torch.zeros_like(residual_covariance),
    )

    outputs = (
        fitted_position,
        fitted_velocity,
        predicted_positions,
        residuals,
        residual_covariance,
        normal_matrix,
        condition_number,
        support_weight,
    )
    if not all(torch.isfinite(value).all() for value in outputs):
        raise ValueError("free-motion fit produced a nonfinite output")
    return FreeMotionFitResult(
        position=fitted_position,
        velocity=fitted_velocity,
        anchor_time=anchor,
        predicted_positions=predicted_positions,
        residuals=residuals,
        residual_covariance=residual_covariance,
        normal_matrix=normal_matrix,
        condition_number=condition_number,
        support_count=support_count,
        support_weight=support_weight,
        valid=valid,
    )


__all__ = [
    "FreeMotionFitResult",
    "fit_free_motion",
    "free_motion_position_velocity",
]
