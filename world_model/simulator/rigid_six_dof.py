"""Independent six-degree-of-freedom rigid-body reference physics.

This CPU oracle deliberately does not call world-model contact code.  It
extends the independent rigid simulator with local-frame angular velocity,
shape-derived inertia, off-centre impulses, Coulomb friction, and quaternion
integration while retaining the existing ``PhysicsStepEvents`` contract.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor

from world_model.belief import RigidPrimitive
from world_model.simulator.physics import PhysicsConfig, PhysicsStepEvents, empty_physics_events
from world_model.simulator.rigid import RigidBodyState, _rotation_matrix
from world_model.simulator.rigid_physics import _integrate_free, _pair_contact, _plane_support


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:]
    right_xyz, right_w = right[..., :3], right[..., 3:]
    return torch.cat(
        (
            left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz),
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
        ),
        dim=-1,
    )


def _integrate_orientation(orientation: Tensor, angular_velocity: Tensor, dt: float) -> Tensor:
    rotation = angular_velocity * dt
    angle = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-8,
        torch.sin(half) / angle.clamp_min(1.0e-8),
        0.5 - angle.square() / 48.0,
    )
    delta = torch.cat((rotation * scale, torch.cos(half)), dim=-1)
    updated = _quaternion_multiply(orientation, delta)
    return updated / torch.linalg.vector_norm(updated, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _inverse_inertia_body(state: RigidBodyState) -> Tensor:
    mass = state.mass[:, 0]
    radius = state.radius[:, 0]
    sphere = (0.4 * mass * radius.square()).unsqueeze(-1).expand(-1, 3)
    extent = state.half_extents
    box = (mass / 3.0).unsqueeze(-1) * torch.stack(
        (
            extent[:, 1].square() + extent[:, 2].square(),
            extent[:, 0].square() + extent[:, 2].square(),
            extent[:, 0].square() + extent[:, 1].square(),
        ),
        dim=-1,
    )
    inertia = torch.where(
        (state.primitive == int(RigidPrimitive.BOX)).unsqueeze(-1),
        box,
        sphere,
    )
    return inertia.clamp_min(1.0e-8).reciprocal()


def _inverse_world_inertia(rotation: Tensor, inverse_body: Tensor, vector: Tensor) -> Tensor:
    return rotation @ (inverse_body * (rotation.transpose(-1, -2) @ vector))


def _support_point(
    state: RigidBodyState,
    rotations: Tensor,
    index: int,
    direction: Tensor,
) -> Tensor:
    if int(state.primitive[index]) == int(RigidPrimitive.SPHERE):
        return state.position[index] + direction * state.radius[index, 0]
    local = rotations[index].transpose(-1, -2) @ direction
    sign = torch.where(local < 0.0, -torch.ones_like(local), torch.ones_like(local))
    return state.position[index] + rotations[index] @ (sign * state.half_extents[index])


def _contact_point(
    state: RigidBodyState,
    rotations: Tensor,
    first: int,
    second: int,
    normal: Tensor,
) -> Tensor:
    first_box = int(state.primitive[first]) == int(RigidPrimitive.BOX)
    second_box = int(state.primitive[second]) == int(RigidPrimitive.BOX)
    if not first_box and second_box:
        local = rotations[second].transpose(-1, -2) @ (
            state.position[first] - state.position[second]
        )
        closest = local.clamp(-state.half_extents[second], state.half_extents[second])
        return state.position[second] + rotations[second] @ closest
    if first_box and not second_box:
        local = rotations[first].transpose(-1, -2) @ (
            state.position[second] - state.position[first]
        )
        closest = local.clamp(-state.half_extents[first], state.half_extents[first])
        return state.position[first] + rotations[first] @ closest
    return 0.5 * (
        _support_point(state, rotations, first, normal)
        + _support_point(state, rotations, second, -normal)
    )


def _resolve_six_dof_contacts(
    state: RigidBodyState,
    config: PhysicsConfig,
) -> tuple[RigidBodyState, PhysicsStepEvents]:
    count = state.max_objects
    events = empty_physics_events(count, dtype=state.position.dtype, device=state.position.device)
    position = state.position.clone()
    velocity = state.velocity.clone()
    angular = state.angular_velocity.clone()
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
    normals = state.position.new_tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    offsets = state.position.new_tensor(
        [lower[0], -upper[0], lower[1], -upper[1], lower[2], -upper[2]]
    )
    inverse_mass = state.mass[:, 0].reciprocal()
    inverse_inertia = _inverse_inertia_body(state)

    for _ in range(config.solver_iterations):
        current = replace(
            state,
            position=position,
            velocity=velocity,
            angular_velocity=angular,
        )
        rotations = _rotation_matrix(current.orientation)
        for index in range(count):
            if not bool(current.active[index]):
                continue
            for plane, (normal, offset) in enumerate(zip(normals, offsets, strict=True)):
                support = _plane_support(current, rotations, index, normal)
                gap = position[index].dot(normal) - offset - support
                penetration = (-gap).clamp_min(0.0)
                if bool(gap <= 1.0e-5):
                    boundary_contact[index, plane] = True
                    boundary_penetration[index, plane] = torch.maximum(
                        boundary_penetration[index, plane], penetration
                    )
                position[index] += penetration * normal
                point = _support_point(current, rotations, index, -normal)
                arm = point - position[index]
                omega_world = rotations[index] @ angular[index]
                point_velocity = velocity[index] + torch.linalg.cross(omega_world, arm)
                normal_speed = point_velocity.dot(normal)
                if not bool((gap <= 1.0e-5) & (normal_speed < 0.0)):
                    continue
                normal_cross = torch.linalg.cross(arm, normal)
                denominator = (
                    inverse_mass[index]
                    + normal.dot(
                        torch.linalg.cross(
                            _inverse_world_inertia(
                                rotations[index], inverse_inertia[index], normal_cross
                            ),
                            arm,
                        )
                    )
                ).clamp_min(1.0e-8)
                normal_impulse = (
                    -(1.0 + state.restitution[index, 0]) * normal_speed / denominator
                ).clamp_min(0.0)
                tangent_velocity = point_velocity - normal_speed * normal
                tangent_speed = torch.linalg.vector_norm(tangent_velocity)
                tangent = tangent_velocity / tangent_speed.clamp_min(1.0e-8)
                tangent_cross = torch.linalg.cross(arm, tangent)
                tangent_denominator = (
                    inverse_mass[index]
                    + tangent.dot(
                        torch.linalg.cross(
                            _inverse_world_inertia(
                                rotations[index], inverse_inertia[index], tangent_cross
                            ),
                            arm,
                        )
                    )
                ).clamp_min(1.0e-8)
                tangent_impulse = torch.minimum(
                    state.friction[index, 0] * normal_impulse,
                    tangent_speed / tangent_denominator,
                )
                impulse = normal_impulse * normal - tangent_impulse * tangent
                velocity[index] += impulse * inverse_mass[index]
                torque = torch.linalg.cross(arm, impulse)
                angular[index] += inverse_inertia[index] * (
                    rotations[index].transpose(-1, -2) @ torque
                )
                boundary_collision[index, plane] = True
                boundary_impulse[index, plane] += normal_impulse
                current = replace(
                    current,
                    position=position,
                    velocity=velocity,
                    angular_velocity=angular,
                )

        current = replace(
            state,
            position=position,
            velocity=velocity,
            angular_velocity=angular,
        )
        rotations = _rotation_matrix(current.orientation)
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
                correction = (
                    config.position_correction
                    * (penetration - config.penetration_slop).clamp_min(0.0)
                ).clamp_max(config.max_position_correction)
                inverse_sum = inverse_mass[first] + inverse_mass[second]
                position[first] -= normal * correction * inverse_mass[first] / inverse_sum
                position[second] += normal * correction * inverse_mass[second] / inverse_sum
                if not bool(gap <= 1.0e-5):
                    continue
                point = _contact_point(current, rotations, first, second, normal)
                first_arm = point - position[first]
                second_arm = point - position[second]
                first_contact_velocity = velocity[first] + torch.linalg.cross(
                    rotations[first] @ angular[first], first_arm
                )
                second_contact_velocity = velocity[second] + torch.linalg.cross(
                    rotations[second] @ angular[second], second_arm
                )
                relative = second_contact_velocity - first_contact_velocity
                closing = relative.dot(normal)
                if not bool(closing < 0.0):
                    continue
                first_cross = torch.linalg.cross(first_arm, normal)
                second_cross = torch.linalg.cross(second_arm, normal)
                angular_denominator = normal.dot(
                    torch.linalg.cross(
                        _inverse_world_inertia(
                            rotations[first], inverse_inertia[first], first_cross
                        ),
                        first_arm,
                    )
                    + torch.linalg.cross(
                        _inverse_world_inertia(
                            rotations[second], inverse_inertia[second], second_cross
                        ),
                        second_arm,
                    )
                )
                denominator = (inverse_sum + angular_denominator).clamp_min(1.0e-8)
                restitution = torch.minimum(
                    state.restitution[first, 0], state.restitution[second, 0]
                )
                normal_impulse = (-(1.0 + restitution) * closing / denominator).clamp_min(0.0)
                tangent_velocity = relative - closing * normal
                tangent_speed = torch.linalg.vector_norm(tangent_velocity)
                tangent = tangent_velocity / tangent_speed.clamp_min(1.0e-8)
                first_tangent_cross = torch.linalg.cross(first_arm, tangent)
                second_tangent_cross = torch.linalg.cross(second_arm, tangent)
                tangent_denominator = (
                    inverse_sum
                    + tangent.dot(
                        torch.linalg.cross(
                            _inverse_world_inertia(
                                rotations[first], inverse_inertia[first], first_tangent_cross
                            ),
                            first_arm,
                        )
                        + torch.linalg.cross(
                            _inverse_world_inertia(
                                rotations[second], inverse_inertia[second], second_tangent_cross
                            ),
                            second_arm,
                        )
                    )
                ).clamp_min(1.0e-8)
                friction = torch.sqrt(
                    state.friction[first, 0].clamp_min(0.0)
                    * state.friction[second, 0].clamp_min(0.0)
                )
                tangent_impulse = torch.minimum(
                    friction * normal_impulse,
                    tangent_speed / tangent_denominator,
                )
                impulse = normal_impulse * normal - tangent_impulse * tangent
                velocity[first] -= impulse * inverse_mass[first]
                velocity[second] += impulse * inverse_mass[second]
                angular[first] += inverse_inertia[first] * (
                    rotations[first].transpose(-1, -2) @ torch.linalg.cross(first_arm, -impulse)
                )
                angular[second] += inverse_inertia[second] * (
                    rotations[second].transpose(-1, -2) @ torch.linalg.cross(second_arm, impulse)
                )
                pair_collision[first, second] = pair_collision[second, first] = True
                pair_impulse[first, second] += normal_impulse
                pair_impulse[second, first] += normal_impulse

    collision = pair_collision.any(dim=-1) | boundary_collision.any(dim=-1)
    contact = pair_contact.any(dim=-1) | boundary_contact.any(dim=-1)
    return replace(
        state,
        position=position,
        velocity=velocity,
        angular_velocity=angular,
    ), replace(
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


def advance_rigid_bodies_6dof(
    state: RigidBodyState,
    dt: float,
    config: PhysicsConfig,
    *,
    external_impulse: Tensor | None = None,
) -> tuple[RigidBodyState, PhysicsStepEvents]:
    """Advance rigid state using an independent fixed-substep 6-DoF solver."""

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
        current = replace(
            current,
            orientation=_integrate_orientation(
                current.orientation,
                current.angular_velocity,
                substep_dt,
            ),
        )
        current, step = _resolve_six_dof_contacts(current, config)
        first_offset = accumulated.first_event_offset
        new_event = step.collision & (first_offset < 0.0)
        first_offset = torch.where(
            new_event,
            first_offset.new_full(first_offset.shape, substep * substep_dt),
            first_offset,
        )
        accumulated = replace(
            accumulated,
            pair_contact=accumulated.pair_contact | step.pair_contact,
            pair_collision=accumulated.pair_collision | step.pair_collision,
            pair_impulse=accumulated.pair_impulse + step.pair_impulse,
            pair_penetration=torch.maximum(accumulated.pair_penetration, step.pair_penetration),
            boundary_contact=accumulated.boundary_contact | step.boundary_contact,
            boundary_collision=accumulated.boundary_collision | step.boundary_collision,
            boundary_impulse=accumulated.boundary_impulse + step.boundary_impulse,
            boundary_penetration=torch.maximum(
                accumulated.boundary_penetration, step.boundary_penetration
            ),
            collision=accumulated.collision | step.collision,
            contact=accumulated.contact | step.contact,
            sleeping=current.sleeping,
            external_impulse=impulse,
            first_event_offset=first_offset,
        )
    return current, accumulated


__all__ = ["advance_rigid_bodies_6dof"]
