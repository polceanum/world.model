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
    local = rotation.transpose(-1, -2) @ vector_world
    return rotation @ (inverse_body * local)


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

        for batch_index in range(batch):
            for first in range(count):
                for second in range(first + 1, count):
                    if not bool(contact_upper[batch_index, first, second]):
                        continue
                    pair_normal = normal[batch_index, first, second]
                    point = _contact_point(
                        objects,
                        rotations,
                        batch_index,
                        first,
                        second,
                        pair_normal,
                    )
                    first_arm = point - position[batch_index, first]
                    second_arm = point - position[batch_index, second]
                    first_rotation = rotations[batch_index, first]
                    second_rotation = rotations[batch_index, second]
                    first_omega_world = first_rotation @ angular_velocity[batch_index, first]
                    second_omega_world = second_rotation @ angular_velocity[batch_index, second]
                    first_contact_velocity = velocity[batch_index, first] + torch.linalg.cross(
                        first_omega_world,
                        first_arm,
                    )
                    second_contact_velocity = velocity[batch_index, second] + torch.linalg.cross(
                        second_omega_world,
                        second_arm,
                    )
                    relative_velocity = second_contact_velocity - first_contact_velocity
                    closing = relative_velocity.dot(pair_normal)
                    if not bool(closing < -self.collision_speed_epsilon):
                        continue
                    first_cross = torch.linalg.cross(first_arm, pair_normal)
                    second_cross = torch.linalg.cross(second_arm, pair_normal)
                    angular_denominator = pair_normal.dot(
                        torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                first_rotation,
                                inverse_inertia[batch_index, first],
                                first_cross,
                            ),
                            first_arm,
                        )
                        + torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                second_rotation,
                                inverse_inertia[batch_index, second],
                                second_cross,
                            ),
                            second_arm,
                        )
                    )
                    denominator = (
                        inverse_mass[batch_index, first]
                        + inverse_mass[batch_index, second]
                        + angular_denominator
                    ).clamp_min(1.0e-8)
                    restitution = torch.minimum(
                        objects.restitution[batch_index, first, 0],
                        objects.restitution[batch_index, second, 0],
                    )
                    normal_impulse = (-(1.0 + restitution) * closing / denominator).clamp_min(0.0)
                    if graph is not None:
                        multiplier = 1.0 + self.max_impulse_multiplier_residual * torch.tanh(
                            graph.impulse_multiplier_raw[batch_index, first, second]
                        )
                        additive = self.max_impulse_additive_residual * torch.tanh(
                            graph.impulse_additive_raw[batch_index, first, second]
                        )
                        normal_impulse = (normal_impulse * multiplier + additive).clamp_min(0.0)

                    tangent_velocity = relative_velocity - closing * pair_normal
                    tangent_speed = torch.linalg.vector_norm(tangent_velocity)
                    tangent_direction = _safe_tangent_direction(
                        tangent_velocity.unsqueeze(0), tangent_speed.unsqueeze(0)
                    )[0]
                    tangent_first_cross = torch.linalg.cross(first_arm, tangent_direction)
                    tangent_second_cross = torch.linalg.cross(second_arm, tangent_direction)
                    tangent_angular = tangent_direction.dot(
                        torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                first_rotation,
                                inverse_inertia[batch_index, first],
                                tangent_first_cross,
                            ),
                            first_arm,
                        )
                        + torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                second_rotation,
                                inverse_inertia[batch_index, second],
                                tangent_second_cross,
                            ),
                            second_arm,
                        )
                    )
                    tangent_denominator = (
                        inverse_mass[batch_index, first]
                        + inverse_mass[batch_index, second]
                        + tangent_angular
                    ).clamp_min(1.0e-8)
                    friction = torch.sqrt(
                        objects.friction[batch_index, first, 0].clamp_min(0.0)
                        * objects.friction[batch_index, second, 0].clamp_min(0.0)
                    )
                    tangent_impulse = torch.minimum(
                        friction * normal_impulse,
                        tangent_speed / tangent_denominator,
                    )
                    impulse_world = (
                        normal_impulse * pair_normal - tangent_impulse * tangent_direction
                    )
                    velocity[batch_index, first] -= impulse_world * inverse_mass[batch_index, first]
                    velocity[batch_index, second] += (
                        impulse_world * inverse_mass[batch_index, second]
                    )
                    first_torque = torch.linalg.cross(first_arm, -impulse_world)
                    second_torque = torch.linalg.cross(second_arm, impulse_world)
                    angular_velocity[batch_index, first] += inverse_inertia[batch_index, first] * (
                        first_rotation.transpose(-1, -2) @ first_torque
                    )
                    angular_velocity[batch_index, second] += inverse_inertia[
                        batch_index, second
                    ] * (second_rotation.transpose(-1, -2) @ second_torque)
                    collision_upper[batch_index, first, second] = True
                    pair_impulse[batch_index, first, second] = normal_impulse
                    pair_momentum[batch_index, first, second] = -impulse_world
                    pair_momentum[batch_index, second, first] = impulse_world

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

        for plane_index in range(plane_count):
            normal = normals[plane_index]
            offset = offsets[plane_index]
            support = self._plane_support(objects, normal.unsqueeze(0)).squeeze(-1)
            for batch_index in range(batch):
                for object_index in range(count):
                    if not bool(objects.active[batch_index, object_index]):
                        continue
                    gap = (
                        position[batch_index, object_index].dot(normal)
                        - offset
                        - support[batch_index, object_index]
                    )
                    if not bool(gap <= self.boundary_contact_tolerance):
                        continue
                    contact[batch_index, object_index, plane_index] = True
                    depth = (-gap).clamp_min(0.0)
                    penetration[batch_index, object_index, plane_index] = depth
                    position[batch_index, object_index] += depth * normal
                    point = _support_point(
                        objects,
                        rotations[batch_index, object_index],
                        batch_index,
                        object_index,
                        -normal,
                    )
                    arm = point - position[batch_index, object_index]
                    rotation = rotations[batch_index, object_index]
                    omega_world = rotation @ angular_velocity[batch_index, object_index]
                    point_velocity = velocity[batch_index, object_index] + torch.linalg.cross(
                        omega_world, arm
                    )
                    normal_speed = point_velocity.dot(normal)
                    if not bool(normal_speed < -self.boundary_collision_speed_epsilon):
                        continue
                    cross_normal = torch.linalg.cross(arm, normal)
                    angular_term = normal.dot(
                        torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                rotation,
                                inverse_inertia[batch_index, object_index],
                                cross_normal,
                            ),
                            arm,
                        )
                    )
                    denominator = (
                        inverse_mass[batch_index, object_index] + angular_term
                    ).clamp_min(1.0e-8)
                    normal_impulse = (
                        -(1.0 + objects.restitution[batch_index, object_index, 0])
                        * normal_speed
                        / denominator
                    ).clamp_min(0.0)
                    tangent_velocity = point_velocity - normal_speed * normal
                    tangent_speed = torch.linalg.vector_norm(tangent_velocity)
                    tangent_direction = _safe_tangent_direction(
                        tangent_velocity.unsqueeze(0), tangent_speed.unsqueeze(0)
                    )[0]
                    cross_tangent = torch.linalg.cross(arm, tangent_direction)
                    tangent_angular = tangent_direction.dot(
                        torch.linalg.cross(
                            _apply_inverse_world_inertia(
                                rotation,
                                inverse_inertia[batch_index, object_index],
                                cross_tangent,
                            ),
                            arm,
                        )
                    )
                    tangent_denominator = (
                        inverse_mass[batch_index, object_index] + tangent_angular
                    ).clamp_min(1.0e-8)
                    tangent_impulse = torch.minimum(
                        objects.friction[batch_index, object_index, 0] * normal_impulse,
                        tangent_speed / tangent_denominator,
                    )
                    impulse_world = normal_impulse * normal - tangent_impulse * tangent_direction
                    velocity[batch_index, object_index] += (
                        impulse_world * inverse_mass[batch_index, object_index]
                    )
                    torque = torch.linalg.cross(arm, impulse_world)
                    angular_velocity[batch_index, object_index] += inverse_inertia[
                        batch_index, object_index
                    ] * (rotation.transpose(-1, -2) @ torque)
                    collision[batch_index, object_index, plane_index] = True

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
