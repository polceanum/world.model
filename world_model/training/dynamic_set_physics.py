"""Prevalidated simulator fast paths owned only by specification 1.61.

The accepted known-action scene binds :mod:`world_model.simulator.physics`
byte-for-byte.  Dynamic-set data generation therefore keeps its optional
prevalidation/cache optimization outside that frozen public simulator module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.simulator.collisions import PairCollisionResult
from world_model.simulator.physics import (
    PhysicsConfig,
    PhysicsStepEvents,
    SphereState,
    _integrate_free_motion_exact,
    _replace_state,
    empty_physics_events,
)


@dataclass(frozen=True)
class PrevalidatedSpherePairCache:
    """Static pair terms for repeated stepping of one padded scene schema."""

    count: int
    first: Tensor
    second: Tensor
    minimum_distance: Tensor
    inverse_mass_first: Tensor
    inverse_mass_second: Tensor
    inverse_mass_sum: Tensor
    pair_restitution: Tensor
    pair_friction: Tensor


def prevalidated_sphere_pair_cache(state: SphereState) -> PrevalidatedSpherePairCache:
    """Validate once and cache immutable pair properties for a scene stream."""

    state.validate()
    count = state.max_objects
    radius = state.radius[:, 0]
    mass = state.mass[:, 0]
    restitution = state.restitution[:, 0]
    friction = state.friction[:, 0]
    pair_indices = torch.triu_indices(
        count,
        count,
        offset=1,
        device=state.position.device,
    )
    first, second = pair_indices[0], pair_indices[1]
    inverse_mass_first = mass[first].reciprocal()
    inverse_mass_second = mass[second].reciprocal()
    return PrevalidatedSpherePairCache(
        count=count,
        first=first,
        second=second,
        minimum_distance=radius[first] + radius[second],
        inverse_mass_first=inverse_mass_first,
        inverse_mass_second=inverse_mass_second,
        inverse_mass_sum=inverse_mass_first + inverse_mass_second,
        pair_restitution=torch.minimum(
            restitution[first],
            restitution[second],
        ).clamp(0.0, 1.0),
        pair_friction=torch.sqrt(friction[first].clamp_min(0.0) * friction[second].clamp_min(0.0)),
    )


def _resolve_sphere_pairs_prevalidated(
    position: Tensor,
    velocity: Tensor,
    active: Tensor,
    cache: PrevalidatedSpherePairCache,
    config: PhysicsConfig,
) -> PairCollisionResult:
    """Equivalent sphere-pair solve using immutable terms cached once."""

    count = cache.count
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

    first, second = cache.first, cache.second
    relative_position = position[second] - position[first]
    distance = torch.linalg.vector_norm(relative_position, dim=-1)
    pair_active = active[first] & active[second]
    in_contact = pair_active & (distance < cache.minimum_distance)
    epsilon = torch.finfo(position.dtype).eps
    normal = relative_position / distance.clamp_min(epsilon).unsqueeze(-1)
    coincident = distance <= epsilon
    if bool(coincident.any()):
        fallback = torch.zeros_like(normal)
        fallback[:, 0] = 1.0
        normal = torch.where(coincident.unsqueeze(-1), fallback, normal)

    relative_velocity = velocity[second] - velocity[first]
    relative_normal_speed = (relative_velocity * normal).sum(dim=-1)
    approaching = in_contact & (relative_normal_speed < -1.0e-7)
    normal_impulse = torch.where(
        approaching,
        -(1.0 + cache.pair_restitution)
        * relative_normal_speed
        / cache.inverse_mass_sum.clamp_min(epsilon),
        torch.zeros_like(relative_normal_speed),
    )
    normal_impulse_vector = normal_impulse.unsqueeze(-1) * normal
    relative_tangent = relative_velocity - relative_normal_speed.unsqueeze(-1) * normal
    desired_tangent_impulse = -relative_tangent / cache.inverse_mass_sum.clamp_min(
        1.0e-12
    ).unsqueeze(-1)
    tangent_norm = torch.linalg.vector_norm(desired_tangent_impulse, dim=-1)
    max_tangent_impulse = cache.pair_friction * normal_impulse
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
    delta_velocity.index_add_(
        0,
        first,
        -total_impulse * cache.inverse_mass_first.unsqueeze(-1),
    )
    delta_velocity.index_add_(
        0,
        second,
        total_impulse * cache.inverse_mass_second.unsqueeze(-1),
    )
    updated_velocity = updated_velocity + delta_velocity
    penetration = torch.where(
        in_contact,
        (cache.minimum_distance - distance).clamp_min(0.0),
        0.0,
    )
    correction_magnitude = (
        config.position_correction
        * (penetration - config.penetration_slop).clamp_min(0.0)
        / cache.inverse_mass_sum.clamp_min(1.0e-12)
    ).clamp_max(config.max_position_correction)
    correction = correction_magnitude.unsqueeze(-1) * normal
    delta_position = torch.zeros_like(updated_position)
    delta_position.index_add_(
        0,
        first,
        -correction * cache.inverse_mass_first.unsqueeze(-1),
    )
    delta_position.index_add_(
        0,
        second,
        correction * cache.inverse_mass_second.unsqueeze(-1),
    )
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


def advance_spheres_no_boundary_contacts_prevalidated(
    state: SphereState,
    dt: float,
    config: PhysicsConfig,
    *,
    external_impulse: Tensor | None = None,
    contact_tolerance: float = 1.0e-4,
    pair_cache: PrevalidatedSpherePairCache | None = None,
) -> tuple[SphereState, PhysicsStepEvents]:
    """Advance a prevalidated scene whose contract forbids boundary contact."""

    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    if not math.isfinite(contact_tolerance) or contact_tolerance < 0:
        raise ValueError("contact_tolerance must be finite and nonnegative")
    count = state.max_objects
    if pair_cache is None:
        pair_cache = prevalidated_sphere_pair_cache(state)
    elif pair_cache.count != count:
        raise ValueError("pair cache object count differs from state")
    if dt == 0:
        return state.clone(), empty_physics_events(
            count,
            dtype=state.position.dtype,
            device=state.position.device,
            substeps=0,
        )
    num_substeps = max(1, math.ceil(dt / config.max_substep))
    sub_dt = dt / num_substeps
    events = empty_physics_events(
        count,
        dtype=state.position.dtype,
        device=state.position.device,
        substeps=num_substeps,
    )

    position = state.position.clone()
    velocity = state.velocity.clone()
    sleeping = state.sleeping.clone()
    sleep_counter = state.sleep_counter.clone()
    active = state.active
    if external_impulse is None:
        external_impulse = torch.zeros_like(velocity)
    if external_impulse.shape != velocity.shape:
        raise ValueError("external_impulse must have shape [N, 3]")
    external_impulse = torch.where(
        active.unsqueeze(-1), external_impulse, torch.zeros_like(external_impulse)
    )
    impulse_norm = torch.linalg.vector_norm(external_impulse, dim=-1)
    externally_acted = active & (impulse_norm > 0)
    velocity = velocity + external_impulse / state.mass.clamp_min(1.0e-12)
    sleeping = sleeping & ~externally_acted
    sleep_counter = torch.where(externally_acted, torch.zeros_like(sleep_counter), sleep_counter)

    pair_contact = events.pair_contact
    pair_collision = events.pair_collision
    pair_impulse = events.pair_impulse
    pair_penetration = events.pair_penetration
    first_event_offset = events.first_event_offset
    gravity = torch.as_tensor(config.gravity, dtype=position.dtype, device=position.device)
    bounds = torch.as_tensor(config.bounds, dtype=position.dtype, device=position.device)
    radius = state.radius[:, 0]
    contact_radius = radius.unsqueeze(-1) + contact_tolerance

    def assert_boundary_clearance(current_position: Tensor) -> None:
        boundary_contact = active.unsqueeze(-1) & (
            (current_position - bounds[:, 0].unsqueeze(0) <= contact_radius)
            | (bounds[:, 1].unsqueeze(0) - current_position <= contact_radius)
        )
        if bool(boundary_contact.any()):
            raise ValueError("no-boundary-contact fast path reached a world boundary")

    for substep_index in range(num_substeps):
        movable = active & ~sleeping
        position, velocity = _integrate_free_motion_exact(
            position,
            velocity,
            state.drag,
            gravity,
            sub_dt,
            movable,
        )
        assert_boundary_clearance(position)

        substep_collision_objects = torch.zeros(
            count,
            dtype=torch.bool,
            device=position.device,
        )
        relative_position = position[pair_cache.second] - position[pair_cache.first]
        pair_distance = torch.linalg.vector_norm(relative_position, dim=-1)
        pair_active = active[pair_cache.first] & active[pair_cache.second]
        has_pair_contact = bool((pair_active & (pair_distance < pair_cache.minimum_distance)).any())
        if has_pair_contact:
            for solver_index in range(config.solver_iterations):
                pair_result = _resolve_sphere_pairs_prevalidated(
                    position,
                    velocity,
                    active,
                    pair_cache,
                    config,
                )
                position, velocity = pair_result.position, pair_result.velocity
                pair_contact |= pair_result.contact
                pair_collision |= pair_result.collision
                pair_impulse = torch.maximum(pair_impulse, pair_result.impulse_magnitude)
                pair_penetration = torch.maximum(pair_penetration, pair_result.penetration)
                substep_collision_objects |= pair_result.collision.any(dim=-1)
                assert_boundary_clearance(position)
                if solver_index == 0 and not bool(pair_result.contact.any()):
                    break

        woke_by_collision = substep_collision_objects & (
            pair_impulse.max(dim=-1).values > config.wake_impulse
        )
        sleeping = sleeping & ~woke_by_collision
        # The general solver resets the counter when floor contact is absent.
        sleep_counter = torch.zeros_like(sleep_counter)
        velocity = torch.where(sleeping.unsqueeze(-1), torch.zeros_like(velocity), velocity)
        first_event_offset = torch.where(
            substep_collision_objects & (first_event_offset < 0),
            torch.full_like(first_event_offset, (substep_index + 1) * sub_dt),
            first_event_offset,
        )

    collision_objects = pair_collision.any(dim=-1)
    contact_objects = pair_contact.any(dim=-1)
    updated = _replace_state(
        state,
        position=position,
        velocity=velocity,
        sleeping=sleeping,
        sleep_counter=sleep_counter,
    )
    return updated, PhysicsStepEvents(
        pair_contact=pair_contact,
        pair_collision=pair_collision,
        pair_impulse=pair_impulse,
        pair_penetration=pair_penetration,
        boundary_contact=events.boundary_contact,
        boundary_collision=events.boundary_collision,
        boundary_impulse=events.boundary_impulse,
        boundary_penetration=events.boundary_penetration,
        collision=collision_objects,
        contact=contact_objects,
        sleeping=sleeping.clone(),
        external_impulse=external_impulse.clone(),
        first_event_offset=first_event_offset,
        substeps=num_substeps,
    )


__all__ = [
    "PrevalidatedSpherePairCache",
    "advance_spheres_no_boundary_contacts_prevalidated",
    "prevalidated_sphere_pair_cache",
]
