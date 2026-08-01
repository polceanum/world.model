"""Analytic sphere contact resolution used by the synthetic world.

The routines in this module operate on one padded scene at a time.  Object
arrays use ``[N, ...]`` shapes and an explicit boolean ``active`` mask.  Pair
impulses are computed once per unordered pair and applied with equal and
opposite signs, which keeps linear momentum conservation visible and testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

BOUNDARY_NAMES = ("x_min", "x_max", "floor", "ceiling", "z_min", "z_max")


@dataclass(frozen=True)
class PairCollisionResult:
    """Result of resolving all sphere--sphere contacts in a scene."""

    position: Tensor
    velocity: Tensor
    contact: Tensor
    collision: Tensor
    impulse_magnitude: Tensor
    penetration: Tensor


@dataclass(frozen=True)
class BoundaryCollisionResult:
    """Result of resolving contacts with an axis-aligned world box."""

    position: Tensor
    velocity: Tensor
    contact: Tensor
    collision: Tensor
    impulse_magnitude: Tensor
    penetration: Tensor


def _as_object_scalar(value: Tensor, count: int, *, name: str) -> Tensor:
    """Return a scalar object property as ``[N]`` without changing dtype/device."""

    if value.shape == (count, 1):
        return value[:, 0]
    if value.shape == (count,):
        return value
    raise ValueError(f"{name} must have shape [N] or [N, 1], got {tuple(value.shape)}")


def resolve_sphere_sphere_collisions(
    position: Tensor,
    velocity: Tensor,
    radius: Tensor,
    mass: Tensor,
    restitution: Tensor,
    *,
    friction: Tensor | None = None,
    active: Tensor | None = None,
    position_correction: float = 0.8,
    penetration_slop: float = 1.0e-4,
    max_position_correction: float = 0.1,
    velocity_epsilon: float = 1.0e-7,
) -> PairCollisionResult:
    """Resolve simultaneous contacts between padded spheres.

    Restitution for a pair is the smaller object restitution.  Tangential
    impulses use Coulomb friction and are bounded by ``mu * normal_impulse``.
    Position projection is inverse-mass weighted and may occur for separating
    overlapping bodies, while a collision event is emitted only when a positive
    normal impulse is applied.
    """

    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"position must have shape [N, 3], got {tuple(position.shape)}")
    if velocity.shape != position.shape:
        raise ValueError("velocity must have the same shape as position")
    count = position.shape[0]
    radius_1d = _as_object_scalar(radius, count, name="radius")
    mass_1d = _as_object_scalar(mass, count, name="mass")
    restitution_1d = _as_object_scalar(restitution, count, name="restitution")
    if active is None:
        active = torch.ones(count, dtype=torch.bool, device=position.device)
    if active.shape != (count,) or active.dtype != torch.bool:
        raise ValueError("active must be a boolean tensor with shape [N]")
    if torch.any(radius_1d[active] <= 0):
        raise ValueError("active sphere radii must be positive")
    if torch.any(mass_1d[active] <= 0):
        raise ValueError("active sphere masses must be positive")
    if friction is None:
        friction_1d = torch.zeros_like(radius_1d)
    else:
        friction_1d = _as_object_scalar(friction, count, name="friction")

    updated_position = position.clone()
    updated_velocity = velocity.clone()
    contact_matrix = torch.zeros((count, count), dtype=torch.bool, device=position.device)
    collision_matrix = torch.zeros_like(contact_matrix)
    impulse_matrix = position.new_zeros((count, count))
    penetration_matrix = position.new_zeros((count, count))
    if count < 2:
        return PairCollisionResult(
            updated_position,
            updated_velocity,
            contact_matrix,
            collision_matrix,
            impulse_matrix,
            penetration_matrix,
        )

    pair_indices = torch.triu_indices(count, count, offset=1, device=position.device)
    first, second = pair_indices[0], pair_indices[1]
    relative_position = position[second] - position[first]
    distance = torch.linalg.vector_norm(relative_position, dim=-1)
    minimum_distance = radius_1d[first] + radius_1d[second]
    pair_active = active[first] & active[second]
    in_contact = pair_active & (distance < minimum_distance)

    safe_distance = distance.clamp_min(torch.finfo(position.dtype).eps)
    normal = relative_position / safe_distance.unsqueeze(-1)
    coincident = distance <= torch.finfo(position.dtype).eps
    if bool(coincident.any()):
        fallback = torch.zeros_like(normal)
        fallback[:, 0] = 1.0
        normal = torch.where(coincident.unsqueeze(-1), fallback, normal)

    relative_velocity = velocity[second] - velocity[first]
    relative_normal_speed = (relative_velocity * normal).sum(dim=-1)
    approaching = in_contact & (relative_normal_speed < -velocity_epsilon)
    inverse_mass_first = mass_1d[first].reciprocal()
    inverse_mass_second = mass_1d[second].reciprocal()
    inverse_mass_sum = inverse_mass_first + inverse_mass_second
    pair_restitution = torch.minimum(restitution_1d[first], restitution_1d[second]).clamp(0.0, 1.0)
    normal_impulse = torch.where(
        approaching,
        -(1.0 + pair_restitution)
        * relative_normal_speed
        / inverse_mass_sum.clamp_min(torch.finfo(position.dtype).eps),
        torch.zeros_like(relative_normal_speed),
    )
    normal_impulse_vector = normal_impulse.unsqueeze(-1) * normal

    # Coulomb friction opposes relative tangential velocity.  It remains
    # equal-and-opposite because one pair impulse is scattered to both bodies.
    relative_tangent = relative_velocity - relative_normal_speed.unsqueeze(-1) * normal
    desired_tangent_impulse = -relative_tangent / inverse_mass_sum.clamp_min(1.0e-12).unsqueeze(-1)
    tangent_norm = torch.linalg.vector_norm(desired_tangent_impulse, dim=-1)
    pair_friction = torch.sqrt(
        friction_1d[first].clamp_min(0.0) * friction_1d[second].clamp_min(0.0)
    )
    max_tangent_impulse = pair_friction * normal_impulse
    tangent_scale = torch.minimum(
        torch.ones_like(tangent_norm),
        max_tangent_impulse / tangent_norm.clamp_min(1.0e-12),
    )
    tangent_impulse_vector = desired_tangent_impulse * tangent_scale.unsqueeze(-1)
    tangent_impulse_vector = torch.where(
        approaching.unsqueeze(-1),
        tangent_impulse_vector,
        torch.zeros_like(tangent_impulse_vector),
    )
    total_impulse = normal_impulse_vector + tangent_impulse_vector

    delta_velocity = torch.zeros_like(updated_velocity)
    delta_velocity.index_add_(0, first, -total_impulse * inverse_mass_first.unsqueeze(-1))
    delta_velocity.index_add_(0, second, total_impulse * inverse_mass_second.unsqueeze(-1))
    updated_velocity = updated_velocity + delta_velocity

    penetration = torch.where(in_contact, (minimum_distance - distance).clamp_min(0.0), 0.0)
    correction_magnitude = (
        position_correction
        * (penetration - penetration_slop).clamp_min(0.0)
        / inverse_mass_sum.clamp_min(1.0e-12)
    ).clamp_max(max_position_correction)
    correction = correction_magnitude.unsqueeze(-1) * normal
    delta_position = torch.zeros_like(updated_position)
    delta_position.index_add_(0, first, -correction * inverse_mass_first.unsqueeze(-1))
    delta_position.index_add_(0, second, correction * inverse_mass_second.unsqueeze(-1))
    updated_position = updated_position + delta_position

    contact_matrix[first, second] = in_contact
    contact_matrix[second, first] = in_contact
    collision_matrix[first, second] = approaching
    collision_matrix[second, first] = approaching
    impulse_matrix[first, second] = normal_impulse
    impulse_matrix[second, first] = normal_impulse
    penetration_matrix[first, second] = penetration
    penetration_matrix[second, first] = penetration
    return PairCollisionResult(
        position=updated_position,
        velocity=updated_velocity,
        contact=contact_matrix,
        collision=collision_matrix,
        impulse_magnitude=impulse_matrix,
        penetration=penetration_matrix,
    )


def resolve_axis_aligned_boundaries(
    position: Tensor,
    velocity: Tensor,
    radius: Tensor,
    mass: Tensor,
    restitution: Tensor,
    bounds: Tensor | Sequence[Sequence[float]],
    *,
    friction: Tensor | None = None,
    active: Tensor | None = None,
    contact_tolerance: float = 1.0e-4,
    collision_speed_epsilon: float = 0.1,
) -> BoundaryCollisionResult:
    """Resolve sphere contacts with an axis-aligned 3-D box.

    ``bounds`` has shape ``[3, 2]`` in ``(x, y, z)`` order.  The lower y plane
    is the ground.  Event columns follow :data:`BOUNDARY_NAMES`.
    """

    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"position must have shape [N, 3], got {tuple(position.shape)}")
    if velocity.shape != position.shape:
        raise ValueError("velocity must have the same shape as position")
    count = position.shape[0]
    radius_1d = _as_object_scalar(radius, count, name="radius")
    mass_1d = _as_object_scalar(mass, count, name="mass")
    restitution_1d = _as_object_scalar(restitution, count, name="restitution")
    friction_1d = (
        torch.zeros_like(radius_1d)
        if friction is None
        else _as_object_scalar(friction, count, name="friction")
    )
    if active is None:
        active = torch.ones(count, dtype=torch.bool, device=position.device)
    if active.shape != (count,) or active.dtype != torch.bool:
        raise ValueError("active must be a boolean tensor with shape [N]")
    bounds_tensor = torch.as_tensor(bounds, dtype=position.dtype, device=position.device)
    if bounds_tensor.shape != (3, 2):
        raise ValueError("bounds must have shape [3, 2] in x/y/z order")
    if bool(torch.any(bounds_tensor[:, 1] <= bounds_tensor[:, 0])):
        raise ValueError("every upper world bound must exceed its lower bound")
    if collision_speed_epsilon < 0:
        raise ValueError("collision_speed_epsilon must be nonnegative")

    updated_position = position.clone()
    updated_velocity = velocity.clone()
    contact = torch.zeros((count, 6), dtype=torch.bool, device=position.device)
    collision = torch.zeros_like(contact)
    impulse = position.new_zeros((count, 6))
    penetration = position.new_zeros((count, 6))

    # Stable public column order differs from raw axis order so the floor and
    # ceiling remain adjacent and easy to inspect.
    plane_specs = (
        (0, 0, 1.0),  # x_min
        (0, 1, -1.0),  # x_max
        (1, 0, 1.0),  # floor
        (1, 1, -1.0),  # ceiling
        (2, 0, 1.0),  # z_min
        (2, 1, -1.0),  # z_max
    )
    for plane_index, (axis, side, inward_sign) in enumerate(plane_specs):
        plane_coordinate = bounds_tensor[axis, side]
        if inward_sign > 0:
            signed_distance = updated_position[:, axis] - plane_coordinate
        else:
            signed_distance = plane_coordinate - updated_position[:, axis]
        plane_penetration = torch.where(
            active,
            (radius_1d - signed_distance).clamp_min(0.0),
            torch.zeros_like(signed_distance),
        )
        plane_contact = active & (signed_distance <= radius_1d + contact_tolerance)
        normal = torch.zeros_like(updated_velocity)
        normal[:, axis] = inward_sign
        normal_speed = (updated_velocity * normal).sum(dim=-1)
        plane_collision = plane_contact & (normal_speed < -collision_speed_epsilon)
        resting_inward = plane_contact & (normal_speed < 0.0) & ~plane_collision
        normal_impulse = torch.where(
            plane_collision,
            -(1.0 + restitution_1d.clamp(0.0, 1.0)) * normal_speed * mass_1d,
            torch.zeros_like(normal_speed),
        )
        updated_velocity = (
            updated_velocity + (normal_impulse / mass_1d.clamp_min(1.0e-12)).unsqueeze(-1) * normal
        )
        # Gravity introduces a small inward speed on every semi-implicit
        # substep. Treat sub-threshold inward motion as a resting constraint,
        # not a fresh restitution impact, while leaving tangential sliding
        # untouched.
        updated_velocity = updated_velocity - (
            torch.where(resting_inward, normal_speed, torch.zeros_like(normal_speed)).unsqueeze(-1)
            * normal
        )

        tangent_velocity = updated_velocity - (
            (updated_velocity * normal).sum(dim=-1).unsqueeze(-1) * normal
        )
        desired_tangent_impulse = -mass_1d.unsqueeze(-1) * tangent_velocity
        desired_tangent_norm = torch.linalg.vector_norm(desired_tangent_impulse, dim=-1)
        max_tangent = friction_1d.clamp_min(0.0) * normal_impulse
        tangent_scale = torch.minimum(
            torch.ones_like(desired_tangent_norm),
            max_tangent / desired_tangent_norm.clamp_min(1.0e-12),
        )
        tangent_impulse = desired_tangent_impulse * tangent_scale.unsqueeze(-1)
        tangent_impulse = torch.where(
            plane_collision.unsqueeze(-1),
            tangent_impulse,
            torch.zeros_like(tangent_impulse),
        )
        updated_velocity = updated_velocity + tangent_impulse / mass_1d.clamp_min(
            1.0e-12
        ).unsqueeze(-1)

        # Project only actual penetration; touching bodies are left unchanged.
        updated_position = updated_position + (plane_penetration.unsqueeze(-1) * normal)
        contact[:, plane_index] = plane_contact
        collision[:, plane_index] = plane_collision
        impulse[:, plane_index] = normal_impulse
        penetration[:, plane_index] = plane_penetration

    return BoundaryCollisionResult(
        position=updated_position,
        velocity=updated_velocity,
        contact=contact,
        collision=collision,
        impulse_magnitude=impulse,
        penetration=penetration,
    )


def sphere_sphere_relative_restitution(
    velocity_before: Tensor,
    velocity_after: Tensor,
    normal: Tensor,
) -> Tensor:
    """Return ``-v_rel_after / v_rel_before`` along a collision normal."""

    if velocity_before.shape != (2, 3) or velocity_after.shape != (2, 3):
        raise ValueError("velocity tensors must have shape [2, 3]")
    unit_normal = normal / torch.linalg.vector_norm(normal).clamp_min(1.0e-12)
    before = ((velocity_before[1] - velocity_before[0]) * unit_normal).sum()
    after = ((velocity_after[1] - velocity_after[0]) * unit_normal).sum()
    return -after / before
