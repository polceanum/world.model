"""Opt-in six-degree-of-freedom contact resolution for rigid primitives.

The historical resolver remains the exact default.  This implementation only
changes execution when a caller explicitly enables six-DoF contacts and at
least one active object is an oriented box.  It uses a single observable
contact point per pair/manifold, applies angular impulse through analytic body
inertia, and retains the established hard-contact/event result contract.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from world_model.belief import ObjectBeliefTensor, RigidPrimitive
from world_model.dynamics.contacts import (
    SphereContactResolver,
    _quaternion_rotation_matrix,
    _safe_tangent_direction,
)
from world_model.dynamics.graph import InteractionOutput


def _inverse_inertia_body(objects: ObjectBeliefTensor) -> Tensor:
    """Return diagonal inverse inertia in each object's local frame."""

    mass = objects.mass.squeeze(-1)
    radius = objects.radius.squeeze(-1)
    sphere = (0.4 * mass * radius.square()).unsqueeze(-1).expand(-1, -1, 3)
    extent = objects.geometry_half_extents
    box = (mass / 3.0).unsqueeze(-1) * torch.stack(
        (
            extent[..., 1].square() + extent[..., 2].square(),
            extent[..., 0].square() + extent[..., 2].square(),
            extent[..., 0].square() + extent[..., 1].square(),
        ),
        dim=-1,
    )
    primitive = objects.geometry_primitive
    inertia = torch.where(
        (primitive == int(RigidPrimitive.BOX)).unsqueeze(-1),
        box,
        sphere,
    )
    return inertia.clamp_min(1.0e-8).reciprocal()


def _apply_inverse_world_inertia(
    rotation: Tensor,
    inverse_body: Tensor,
    vector_world: Tensor,
) -> Tensor:
    local = torch.matmul(rotation.transpose(-1, -2), vector_world.unsqueeze(-1)).squeeze(-1)
    return torch.matmul(rotation, (inverse_body * local).unsqueeze(-1)).squeeze(-1)


def _support_points(
    position: Tensor,
    rotation: Tensor,
    primitive: Tensor,
    radius: Tensor,
    half_extents: Tensor,
    direction: Tensor,
) -> Tensor:
    """Return support points for any shared leading batch/object shape."""

    direction = torch.broadcast_to(direction, position.shape)
    sphere = position + direction * radius.unsqueeze(-1)
    local_direction = torch.matmul(rotation.transpose(-1, -2), direction.unsqueeze(-1)).squeeze(-1)
    sign = torch.where(
        local_direction < 0.0,
        -torch.ones_like(local_direction),
        torch.ones_like(local_direction),
    )
    box = position + torch.matmul(rotation, (sign * half_extents).unsqueeze(-1)).squeeze(-1)
    return torch.where((primitive == int(RigidPrimitive.BOX)).unsqueeze(-1), box, sphere)


def _batched_contact_point(
    objects: ObjectBeliefTensor,
    rotations: Tensor,
    first: int,
    second: int,
    normal: Tensor,
) -> Tensor:
    """Vectorize one pair's public contact point over independent batch rows."""

    primitive = objects.geometry_primitive
    half_extents = objects.geometry_half_extents
    radius = objects.radius.squeeze(-1)
    first_position = objects.position[:, first]
    second_position = objects.position[:, second]
    first_rotation = rotations[:, first]
    second_rotation = rotations[:, second]
    first_box = primitive[:, first] == int(RigidPrimitive.BOX)
    second_box = primitive[:, second] == int(RigidPrimitive.BOX)

    local_on_second = torch.matmul(
        second_rotation.transpose(-1, -2),
        (first_position - second_position).unsqueeze(-1),
    ).squeeze(-1)
    closest_on_second = torch.minimum(
        torch.maximum(local_on_second, -half_extents[:, second]),
        half_extents[:, second],
    )
    sphere_box = second_position + torch.matmul(
        second_rotation, closest_on_second.unsqueeze(-1)
    ).squeeze(-1)

    local_on_first = torch.matmul(
        first_rotation.transpose(-1, -2),
        (second_position - first_position).unsqueeze(-1),
    ).squeeze(-1)
    closest_on_first = torch.minimum(
        torch.maximum(local_on_first, -half_extents[:, first]),
        half_extents[:, first],
    )
    box_sphere = first_position + torch.matmul(
        first_rotation, closest_on_first.unsqueeze(-1)
    ).squeeze(-1)

    first_support = _support_points(
        first_position,
        first_rotation,
        primitive[:, first],
        radius[:, first],
        half_extents[:, first],
        normal,
    )
    second_support = _support_points(
        second_position,
        second_rotation,
        primitive[:, second],
        radius[:, second],
        half_extents[:, second],
        -normal,
    )
    symmetric = 0.5 * (first_support + second_support)
    return torch.where(
        ((~first_box) & second_box).unsqueeze(-1),
        sphere_box,
        torch.where((first_box & (~second_box)).unsqueeze(-1), box_sphere, symmetric),
    )


def _support_point(
    objects: ObjectBeliefTensor,
    rotation: Tensor,
    batch_index: int,
    object_index: int,
    direction: Tensor,
) -> Tensor:
    position = objects.position[batch_index, object_index]
    primitive = int(objects.geometry_primitive[batch_index, object_index])
    if primitive == int(RigidPrimitive.SPHERE):
        return position + direction * objects.radius[batch_index, object_index, 0]
    local_direction = rotation.transpose(-1, -2) @ direction
    sign = torch.where(local_direction < 0.0, -torch.ones_like(local_direction), 1.0)
    return position + rotation @ (sign * objects.geometry_half_extents[batch_index, object_index])


def _contact_point(
    objects: ObjectBeliefTensor,
    rotations: Tensor,
    batch_index: int,
    first: int,
    second: int,
    normal: Tensor,
) -> Tensor:
    """Return one public-geometry contact point for an unordered pair."""

    primitive = objects.geometry_primitive[batch_index]
    first_box = int(primitive[first]) == int(RigidPrimitive.BOX)
    second_box = int(primitive[second]) == int(RigidPrimitive.BOX)
    if not first_box and second_box:
        box_rotation = rotations[batch_index, second]
        local = box_rotation.transpose(-1, -2) @ (
            objects.position[batch_index, first] - objects.position[batch_index, second]
        )
        closest = torch.minimum(
            torch.maximum(local, -objects.geometry_half_extents[batch_index, second]),
            objects.geometry_half_extents[batch_index, second],
        )
        return objects.position[batch_index, second] + box_rotation @ closest
    if first_box and not second_box:
        box_rotation = rotations[batch_index, first]
        local = box_rotation.transpose(-1, -2) @ (
            objects.position[batch_index, second] - objects.position[batch_index, first]
        )
        closest = torch.minimum(
            torch.maximum(local, -objects.geometry_half_extents[batch_index, first]),
            objects.geometry_half_extents[batch_index, first],
        )
        return objects.position[batch_index, first] + box_rotation @ closest
    first_support = _support_point(
        objects,
        rotations[batch_index, first],
        batch_index,
        first,
        normal,
    )
    second_support = _support_point(
        objects,
        rotations[batch_index, second],
        batch_index,
        second,
        -normal,
    )
    return 0.5 * (first_support + second_support)


class RigidContactResolver6D(SphereContactResolver):
    """Hard rigid contacts with angular velocity and frictional torque."""

    def _resolve_pairs(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None,
    ) -> tuple[ObjectBeliefTensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self._has_box_geometry(objects):
            return super()._resolve_pairs(objects, graph)

        batch, count = objects.active.shape
        normal, gap, penetration = self._pair_geometry(objects)
        upper = torch.triu(
            torch.ones(count, count, device=objects.active.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        active_pair = objects.active[:, :, None] & objects.active[:, None, :]
        position_variance = objects.fast_log_variance[..., :3].exp()
        relative_variance = position_variance[:, :, None, :] + position_variance[:, None, :, :]
        gap_sigma = (relative_variance * normal.square()).sum(dim=-1).sqrt()
        confident_gap = gap + self.contact_confidence_sigma * gap_sigma
        contact_upper = active_pair & upper & (confident_gap < self.contact_margin)
        collision_upper = torch.zeros_like(contact_upper)
        pair_impulse = objects.position.new_zeros(batch, count, count)
        symmetric_contact = contact_upper | contact_upper.transpose(1, 2)
        if not bool(contact_upper.any()):
            return (
                objects,
                symmetric_contact,
                collision_upper,
                pair_impulse,
                penetration * symmetric_contact,
                objects.position.new_zeros(batch),
            )
        velocity = objects.velocity.clone()
        angular_velocity = objects.angular_velocity.clone()
        position = objects.position.clone()
        rotations = _quaternion_rotation_matrix(objects.orientation)
        inverse_inertia = _inverse_inertia_body(objects)
        inverse_mass = objects.mass.squeeze(-1).reciprocal()
        pair_momentum = objects.position.new_zeros(batch, count, count, 3)

        # Preserve the established lexicographic pair order within every
        # scene, but evaluate independent batch/candidate rows together. This
        # removes the B x K Python loop from counterfactual rigid planning
        # without changing any row's sequential multi-contact resolution.
        for first in range(count):
            for second in range(first + 1, count):
                candidate = contact_upper[:, first, second]
                if not bool(candidate.any()):
                    continue
                pair_normal = normal[:, first, second]
                point = _batched_contact_point(
                    objects,
                    rotations,
                    first,
                    second,
                    pair_normal,
                )
                first_arm = point - position[:, first]
                second_arm = point - position[:, second]
                first_rotation = rotations[:, first]
                second_rotation = rotations[:, second]
                first_omega_world = torch.matmul(
                    first_rotation, angular_velocity[:, first].unsqueeze(-1)
                ).squeeze(-1)
                second_omega_world = torch.matmul(
                    second_rotation, angular_velocity[:, second].unsqueeze(-1)
                ).squeeze(-1)
                first_contact_velocity = velocity[:, first] + torch.linalg.cross(
                    first_omega_world, first_arm
                )
                second_contact_velocity = velocity[:, second] + torch.linalg.cross(
                    second_omega_world, second_arm
                )
                relative_velocity = second_contact_velocity - first_contact_velocity
                closing = (relative_velocity * pair_normal).sum(dim=-1)
                colliding = candidate & (closing < -self.collision_speed_epsilon)
                if not bool(colliding.any()):
                    continue
                first_cross = torch.linalg.cross(first_arm, pair_normal)
                second_cross = torch.linalg.cross(second_arm, pair_normal)
                angular_denominator = (
                    pair_normal
                    * (
                        torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                first_rotation,
                                inverse_inertia[:, first],
                                first_cross,
                            ),
                            first_arm,
                        )
                        + torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                second_rotation,
                                inverse_inertia[:, second],
                                second_cross,
                            ),
                            second_arm,
                        )
                    )
                ).sum(dim=-1)
                denominator = (
                    inverse_mass[:, first] + inverse_mass[:, second] + angular_denominator
                ).clamp_min(1.0e-8)
                restitution = torch.minimum(
                    objects.restitution[:, first, 0], objects.restitution[:, second, 0]
                )
                normal_impulse = (-(1.0 + restitution) * closing / denominator).clamp_min(0.0)
                if graph is not None:
                    multiplier = 1.0 + self.max_impulse_multiplier_residual * torch.tanh(
                        graph.impulse_multiplier_raw[:, first, second]
                    )
                    additive = self.max_impulse_additive_residual * torch.tanh(
                        graph.impulse_additive_raw[:, first, second]
                    )
                    normal_impulse = (normal_impulse * multiplier + additive).clamp_min(0.0)

                tangent_velocity = relative_velocity - closing.unsqueeze(-1) * pair_normal
                tangent_speed = torch.linalg.vector_norm(tangent_velocity, dim=-1)
                tangent_direction = _safe_tangent_direction(tangent_velocity, tangent_speed)
                tangent_first_cross = torch.linalg.cross(first_arm, tangent_direction)
                tangent_second_cross = torch.linalg.cross(second_arm, tangent_direction)
                tangent_angular = (
                    tangent_direction
                    * (
                        torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                first_rotation,
                                inverse_inertia[:, first],
                                tangent_first_cross,
                            ),
                            first_arm,
                        )
                        + torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                second_rotation,
                                inverse_inertia[:, second],
                                tangent_second_cross,
                            ),
                            second_arm,
                        )
                    )
                ).sum(dim=-1)
                tangent_denominator = (
                    inverse_mass[:, first] + inverse_mass[:, second] + tangent_angular
                ).clamp_min(1.0e-8)
                friction = torch.sqrt(
                    objects.friction[:, first, 0].clamp_min(0.0)
                    * objects.friction[:, second, 0].clamp_min(0.0)
                )
                tangent_impulse = torch.minimum(
                    friction * normal_impulse,
                    tangent_speed / tangent_denominator,
                )
                impulse_world = normal_impulse.unsqueeze(-1) * pair_normal - (
                    tangent_impulse.unsqueeze(-1) * tangent_direction
                )
                collision_mask = colliding.unsqueeze(-1)
                velocity[:, first] = torch.where(
                    collision_mask,
                    velocity[:, first] - impulse_world * inverse_mass[:, first].unsqueeze(-1),
                    velocity[:, first],
                )
                velocity[:, second] = torch.where(
                    collision_mask,
                    velocity[:, second] + impulse_world * inverse_mass[:, second].unsqueeze(-1),
                    velocity[:, second],
                )
                first_torque = torch.linalg.cross(first_arm, -impulse_world)
                second_torque = torch.linalg.cross(second_arm, impulse_world)
                first_delta = inverse_inertia[:, first] * torch.matmul(
                    first_rotation.transpose(-1, -2), first_torque.unsqueeze(-1)
                ).squeeze(-1)
                second_delta = inverse_inertia[:, second] * torch.matmul(
                    second_rotation.transpose(-1, -2), second_torque.unsqueeze(-1)
                ).squeeze(-1)
                angular_velocity[:, first] = torch.where(
                    collision_mask,
                    angular_velocity[:, first] + first_delta,
                    angular_velocity[:, first],
                )
                angular_velocity[:, second] = torch.where(
                    collision_mask,
                    angular_velocity[:, second] + second_delta,
                    angular_velocity[:, second],
                )
                collision_upper[:, first, second] = colliding
                pair_impulse[:, first, second] = torch.where(
                    colliding, normal_impulse, pair_impulse[:, first, second]
                )
                pair_momentum[:, first, second] = torch.where(
                    collision_mask, -impulse_world, pair_momentum[:, first, second]
                )
                pair_momentum[:, second, first] = torch.where(
                    collision_mask, impulse_world, pair_momentum[:, second, first]
                )

        inverse_mass_sum = (inverse_mass[:, :, None] + inverse_mass[:, None, :]).clamp_min(1.0e-8)
        correction_scale = (
            self.penetration_fraction
            * (penetration - self.penetration_slop).clamp_min(0.0)
            / inverse_mass_sum
        ).clamp_max(self.max_position_correction)
        position_first = (
            -correction_scale.unsqueeze(-1)
            * inverse_mass[:, :, None, None]
            * normal
            * contact_upper.unsqueeze(-1)
        )
        position_second = (
            correction_scale.unsqueeze(-1)
            * inverse_mass[:, None, :, None]
            * normal
            * contact_upper.unsqueeze(-1)
        )
        position += position_first.sum(dim=2) + position_second.sum(dim=1)
        active = objects.active.unsqueeze(-1)
        updated = replace(
            objects,
            position=torch.where(active, position, objects.position),
            velocity=torch.where(active, velocity, objects.velocity),
            angular_velocity=torch.where(active, angular_velocity, objects.angular_velocity),
        )
        symmetric_collision = collision_upper | collision_upper.transpose(1, 2)
        symmetric_impulse = pair_impulse + pair_impulse.transpose(1, 2)
        residual = torch.linalg.vector_norm(pair_momentum.sum(dim=(1, 2)), dim=-1)
        return (
            updated,
            symmetric_contact,
            symmetric_collision,
            symmetric_impulse,
            penetration * symmetric_contact,
            residual,
        )

    def _resolve_planes(
        self,
        objects: ObjectBeliefTensor,
    ) -> tuple[ObjectBeliefTensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self._has_box_geometry(objects):
            return super()._resolve_planes(objects)

        normals = self.plane_normals.to(objects.position)
        offsets = self.plane_offsets.to(objects.position)
        batch, count = objects.active.shape
        plane_count = normals.shape[0]
        position = objects.position.clone()
        velocity = objects.velocity.clone()
        angular_velocity = objects.angular_velocity.clone()
        rotations = _quaternion_rotation_matrix(objects.orientation)
        inverse_inertia = _inverse_inertia_body(objects)
        inverse_mass = objects.mass.squeeze(-1).reciprocal()
        contact = torch.zeros(
            batch, count, plane_count, dtype=torch.bool, device=objects.active.device
        )
        collision = torch.zeros_like(contact)
        penetration = objects.position.new_zeros(batch, count, plane_count)
        signed_center = torch.einsum("bnc,pc->bnp", objects.position, normals)
        support_all = self._plane_support(objects, normals)
        initial_gap = signed_center - offsets[None, None, :] - support_all
        if not bool(
            (objects.active.unsqueeze(-1) & (initial_gap <= self.boundary_contact_tolerance)).any()
        ):
            return (
                objects,
                contact,
                collision,
                objects.active.new_zeros(batch, count),
                objects.active.new_zeros(batch, count),
                penetration,
            )

        primitive = objects.geometry_primitive
        half_extents = objects.geometry_half_extents
        radius = objects.radius.squeeze(-1)
        for plane_index in range(plane_count):
            normal = normals[plane_index]
            offset = offsets[plane_index]
            support = self._plane_support(objects, normal.unsqueeze(0)).squeeze(-1)
            gap = (position * normal).sum(dim=-1) - offset - support
            touching = objects.active & (gap <= self.boundary_contact_tolerance)
            if not bool(touching.any()):
                continue
            contact[:, :, plane_index] = touching
            depth = (-gap).clamp_min(0.0)
            penetration[:, :, plane_index] = torch.where(
                touching, depth, penetration[:, :, plane_index]
            )
            position = position + touching.unsqueeze(-1) * depth.unsqueeze(-1) * normal
            point = _support_points(
                objects.position,
                rotations,
                primitive,
                radius,
                half_extents,
                -normal,
            )
            arm = point - position
            omega_world = torch.matmul(rotations, angular_velocity.unsqueeze(-1)).squeeze(-1)
            point_velocity = velocity + torch.linalg.cross(omega_world, arm)
            normal_speed = (point_velocity * normal).sum(dim=-1)
            colliding = touching & (normal_speed < -self.boundary_collision_speed_epsilon)
            if not bool(colliding.any()):
                continue
            cross_normal = torch.linalg.cross(arm, normal.expand_as(arm))
            angular_term = (
                normal
                * torch.linalg.cross(
                    _apply_inverse_world_inertia(rotations, inverse_inertia, cross_normal),
                    arm,
                )
            ).sum(dim=-1)
            denominator = (inverse_mass + angular_term).clamp_min(1.0e-8)
            normal_impulse = (
                -(1.0 + objects.restitution[..., 0]) * normal_speed / denominator
            ).clamp_min(0.0)
            tangent_velocity = point_velocity - normal_speed.unsqueeze(-1) * normal
            tangent_speed = torch.linalg.vector_norm(tangent_velocity, dim=-1)
            tangent_direction = _safe_tangent_direction(tangent_velocity, tangent_speed)
            cross_tangent = torch.linalg.cross(arm, tangent_direction)
            tangent_angular = (
                tangent_direction
                * torch.linalg.cross(
                    _apply_inverse_world_inertia(rotations, inverse_inertia, cross_tangent),
                    arm,
                )
            ).sum(dim=-1)
            tangent_denominator = (inverse_mass + tangent_angular).clamp_min(1.0e-8)
            tangent_impulse = torch.minimum(
                objects.friction[..., 0] * normal_impulse,
                tangent_speed / tangent_denominator,
            )
            impulse_world = normal_impulse.unsqueeze(-1) * normal - (
                tangent_impulse.unsqueeze(-1) * tangent_direction
            )
            collision_mask = colliding.unsqueeze(-1)
            velocity = torch.where(
                collision_mask,
                velocity + impulse_world * inverse_mass.unsqueeze(-1),
                velocity,
            )
            torque = torch.linalg.cross(arm, impulse_world)
            angular_delta = inverse_inertia * torch.matmul(
                rotations.transpose(-1, -2), torque.unsqueeze(-1)
            ).squeeze(-1)
            angular_velocity = torch.where(
                collision_mask,
                angular_velocity + angular_delta,
                angular_velocity,
            )
            collision[:, :, plane_index] = colliding

        active = objects.active.unsqueeze(-1)
        updated = replace(
            objects,
            position=torch.where(active, position, objects.position),
            velocity=torch.where(active, velocity, objects.velocity),
            angular_velocity=torch.where(active, angular_velocity, objects.angular_velocity),
        )
        ground_mask = self.ground_plane_mask.to(device=objects.active.device)
        return (
            updated,
            contact,
            collision,
            (contact & ground_mask).any(dim=-1),
            (collision & ground_mask).any(dim=-1),
            penetration,
        )


__all__ = ["RigidContactResolver6D"]
