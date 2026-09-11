"""Small independent linear rigid-body reference integrator.

This module intentionally does not call the world model's contact resolver.  It
is a deterministic CPU oracle for central, frictionless sphere/box impacts and
oriented support against axis-aligned boundaries.  Angular state is carried
through but contact torque is outside this phase's declared capability.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor

from world_model.belief import RigidPrimitive
from world_model.simulator.physics import PhysicsConfig, PhysicsStepEvents, empty_physics_events
from world_model.simulator.rigid import RigidBodyState, _rotation_matrix


def _box_box_contact(
    first_position: Tensor,
    first_extent: Tensor,
    first_rotation: Tensor,
    second_position: Tensor,
    second_extent: Tensor,
    second_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    axes_first = first_rotation.transpose(0, 1)
    axes_second = second_rotation.transpose(0, 1)
    candidates = [*axes_first, *axes_second]
    for first_axis in axes_first:
        for second_axis in axes_second:
            cross = torch.linalg.cross(first_axis, second_axis)
            if float(torch.linalg.vector_norm(cross)) > 1.0e-8:
                candidates.append(cross)
    delta = second_position - first_position
    best_gap = first_position.new_tensor(-torch.inf)
    best_axis = first_position.new_tensor([1.0, 0.0, 0.0])
    for candidate in candidates:
        axis = candidate / torch.linalg.vector_norm(candidate).clamp_min(1.0e-12)
        first_support = (axes_first @ axis).abs().dot(first_extent)
        second_support = (axes_second @ axis).abs().dot(second_extent)
        gap = delta.dot(axis).abs() - first_support - second_support
        if bool(gap > best_gap):
            best_gap = gap
            best_axis = axis if delta.dot(axis) >= 0.0 else -axis
    return best_axis, best_gap


def _sphere_box_contact(
    sphere_position: Tensor,
    sphere_radius: Tensor,
    box_position: Tensor,
    box_extent: Tensor,
    box_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    local = box_rotation.transpose(0, 1) @ (sphere_position - box_position)
    closest = local.clamp(-box_extent, box_extent)
    sphere_to_box = box_rotation @ (closest - local)
    distance = torch.linalg.vector_norm(sphere_to_box)
    if bool(distance > 1.0e-8):
        return sphere_to_box / distance, distance - sphere_radius
    clearance = box_extent - local.abs()
    axis = int(clearance.argmin())
    outward = torch.zeros_like(local)
    outward[axis] = -1.0 if local[axis] < 0.0 else 1.0
    return -(box_rotation @ outward), -(clearance[axis] + sphere_radius)


def _pair_contact(
    state: RigidBodyState,
    rotations: Tensor,
    first: int,
    second: int,
) -> tuple[Tensor, Tensor]:
    first_box = int(state.primitive[first]) == int(RigidPrimitive.BOX)
    second_box = int(state.primitive[second]) == int(RigidPrimitive.BOX)
    if first_box and second_box:
        return _box_box_contact(
            state.position[first],
            state.half_extents[first],
            rotations[first],
            state.position[second],
            state.half_extents[second],
            rotations[second],
        )
    if not first_box and second_box:
        return _sphere_box_contact(
            state.position[first],
            state.radius[first, 0],
            state.position[second],
            state.half_extents[second],
            rotations[second],
        )
    if first_box and not second_box:
        normal, gap = _sphere_box_contact(
            state.position[second],
            state.radius[second, 0],
            state.position[first],
            state.half_extents[first],
            rotations[first],
        )
        return -normal, gap
    delta = state.position[second] - state.position[first]
    distance = torch.linalg.vector_norm(delta)
    normal = delta / distance.clamp_min(1.0e-8)
    return normal, distance - state.radius[first, 0] - state.radius[second, 0]


def _plane_support(state: RigidBodyState, rotations: Tensor, index: int, normal: Tensor) -> Tensor:
    if int(state.primitive[index]) == int(RigidPrimitive.SPHERE):
        return state.radius[index, 0]
    local_axes = rotations[index].transpose(0, 1)
    return (local_axes @ normal).abs().dot(state.half_extents[index])


def _integrate_free(state: RigidBodyState, dt: float, gravity: Tensor) -> RigidBodyState:
    coefficient = state.drag.clamp_min(0.0)
    decay = torch.exp(-coefficient * dt)
    one_minus_decay = -torch.expm1(-coefficient * dt)
    safe = coefficient.clamp_min(1.0e-5)
    velocity_drag = state.velocity * decay + gravity * one_minus_decay / safe
    position_drag = (
        state.position
        + state.velocity * one_minus_decay / safe
        + gravity * (dt / safe - one_minus_decay / safe.square())
    )
    velocity_free = state.velocity + gravity * dt
    position_free = state.position + state.velocity * dt + 0.5 * gravity * dt**2
    use_drag = coefficient >= 1.0e-5
    movable = state.active & ~state.sleeping
    return replace(
        state,
        position=torch.where(
            movable[:, None], torch.where(use_drag, position_drag, position_free), state.position
        ),
        velocity=torch.where(
            movable[:, None], torch.where(use_drag, velocity_drag, velocity_free), state.velocity
        ),
    )


def _resolve_contacts(
    state: RigidBodyState,
    config: PhysicsConfig,
) -> tuple[RigidBodyState, PhysicsStepEvents]:
    count = state.max_objects
    events = empty_physics_events(count, dtype=state.position.dtype, device=state.position.device)
    position = state.position.clone()
    velocity = state.velocity.clone()
    pair_contact = events.pair_contact
    pair_collision = events.pair_collision
    pair_impulse = events.pair_impulse
    pair_penetration = events.pair_penetration
    boundary_contact = events.boundary_contact
    boundary_collision = events.boundary_collision
    boundary_impulse = events.boundary_impulse
    boundary_penetration = events.boundary_penetration
    lower = state.position.new_tensor([bound[0] for bound in config.bounds])
    upper = state.position.new_tensor([bound[1] for bound in config.bounds])
    plane_normals = state.position.new_tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    plane_offsets = state.position.new_tensor(
        [lower[0], -upper[0], lower[1], -upper[1], lower[2], -upper[2]]
    )
    for _ in range(config.solver_iterations):
        current = replace(state, position=position, velocity=velocity)
        rotations = _rotation_matrix(current.orientation)
        # Match the established simulator/model solver order: every boundary
        # is resolved before unordered body pairs on each iteration.
        for index in range(count):
            if not bool(state.active[index]):
                continue
            for plane, (normal, offset) in enumerate(
                zip(plane_normals, plane_offsets, strict=True)
            ):
                support = _plane_support(current, rotations, index, normal)
                gap = position[index].dot(normal) - offset - support
                penetration = (-gap).clamp_min(0.0)
                if bool(gap <= 1.0e-5):
                    boundary_contact[index, plane] = True
                    boundary_penetration[index, plane] = torch.maximum(
                        boundary_penetration[index, plane], penetration
                    )
                position[index] += normal * penetration
                normal_speed = velocity[index].dot(normal)
                if bool((gap <= 1.0e-5) & (normal_speed < 0.0)):
                    impulse = (
                        -(1.0 + state.restitution[index, 0]) * normal_speed * state.mass[index, 0]
                    )
                    velocity[index] += normal * impulse / state.mass[index, 0]
                    boundary_collision[index, plane] = True
                    boundary_impulse[index, plane] += impulse
                current = replace(current, position=position, velocity=velocity)
        for first in range(count):
            if not bool(current.active[first]):
                continue
            for second in range(first + 1, count):
                if not bool(current.active[second]):
                    continue
                normal, gap = _pair_contact(current, rotations, first, second)
                penetration = (-gap).clamp_min(0.0)
                if bool(gap <= 1.0e-5):
                    pair_contact[first, second] = pair_contact[second, first] = True
                    pair_penetration[first, second] = pair_penetration[second, first] = (
                        torch.maximum(pair_penetration[first, second], penetration)
                    )
                inverse_first = state.mass[first, 0].reciprocal()
                inverse_second = state.mass[second, 0].reciprocal()
                inverse_sum = inverse_first + inverse_second
                correction = (
                    config.position_correction
                    * (penetration - config.penetration_slop).clamp_min(0.0)
                ).clamp_max(config.max_position_correction)
                position[first] -= normal * correction * inverse_first / inverse_sum
                position[second] += normal * correction * inverse_second / inverse_sum
                closing = (velocity[second] - velocity[first]).dot(normal)
                if bool((gap <= 1.0e-5) & (closing < 0.0)):
                    restitution = torch.minimum(
                        state.restitution[first, 0], state.restitution[second, 0]
                    )
                    impulse = -(1.0 + restitution) * closing / inverse_sum
                    velocity[first] -= normal * impulse * inverse_first
                    velocity[second] += normal * impulse * inverse_second
                    pair_collision[first, second] = pair_collision[second, first] = True
                    pair_impulse[first, second] += impulse
                    pair_impulse[second, first] += impulse
                current = replace(current, position=position, velocity=velocity)
    collision = pair_collision.any(dim=-1) | boundary_collision.any(dim=-1)
    contact = pair_contact.any(dim=-1) | boundary_contact.any(dim=-1)
    return replace(state, position=position, velocity=velocity), replace(
        events,
        pair_contact=pair_contact,
        pair_collision=pair_collision,
        pair_impulse=pair_impulse,
        pair_penetration=pair_penetration,
        boundary_contact=boundary_contact,
        boundary_collision=boundary_collision,
        boundary_impulse=boundary_impulse,
        boundary_penetration=boundary_penetration,
        collision=collision,
        contact=contact,
        sleeping=state.sleeping,
    )


def advance_rigid_bodies(
    state: RigidBodyState,
    dt: float,
    config: PhysicsConfig,
    *,
    external_impulse: Tensor | None = None,
) -> tuple[RigidBodyState, PhysicsStepEvents]:
    """Advance a rigid state with independent high-rate central impulses."""

    state.validate()
    config.validate()
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    impulse = torch.zeros_like(state.velocity) if external_impulse is None else external_impulse
    if impulse.shape != state.velocity.shape or not torch.isfinite(impulse).all():
        raise ValueError("external_impulse must be finite [N,3]")
    velocity = state.velocity + torch.where(
        state.active[:, None], impulse / state.mass, torch.zeros_like(impulse)
    )
    current = replace(state, velocity=velocity)
    substeps = max(1, math.ceil(dt / config.max_substep))
    substep_dt = dt / substeps
    accumulated = empty_physics_events(
        state.max_objects,
        dtype=state.position.dtype,
        device=state.position.device,
        substeps=substeps,
    )
    gravity = state.position.new_tensor(config.gravity).unsqueeze(0)
    for substep in range(substeps):
        current = _integrate_free(current, substep_dt, gravity)
        current, step_events = _resolve_contacts(current, config)
        first_offset = accumulated.first_event_offset
        new_event = step_events.collision & (first_offset < 0.0)
        first_offset = torch.where(
            new_event,
            first_offset.new_full(first_offset.shape, substep * substep_dt),
            first_offset,
        )
        accumulated = replace(
            accumulated,
            pair_contact=accumulated.pair_contact | step_events.pair_contact,
            pair_collision=accumulated.pair_collision | step_events.pair_collision,
            pair_impulse=accumulated.pair_impulse + step_events.pair_impulse,
            pair_penetration=torch.maximum(
                accumulated.pair_penetration, step_events.pair_penetration
            ),
            boundary_contact=accumulated.boundary_contact | step_events.boundary_contact,
            boundary_collision=accumulated.boundary_collision | step_events.boundary_collision,
            boundary_impulse=accumulated.boundary_impulse + step_events.boundary_impulse,
            boundary_penetration=torch.maximum(
                accumulated.boundary_penetration, step_events.boundary_penetration
            ),
            collision=accumulated.collision | step_events.collision,
            contact=accumulated.contact | step_events.contact,
            sleeping=current.sleeping,
            external_impulse=impulse,
            first_event_offset=first_offset,
        )
    return current, accumulated


__all__ = ["advance_rigid_bodies"]
