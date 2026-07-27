"""Structured sphere-pair and sphere-plane contact resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from world_model.belief import ObjectBeliefTensor
from world_model.dynamics.graph import InteractionOutput


@dataclass(frozen=True)
class ContactPlane:
    """Plane whose valid half-space satisfies ``normal·position >= offset``."""

    normal: tuple[float, float, float]
    offset: float = 0.0
    name: str = "ground"


@dataclass
class ContactResult:
    objects: ObjectBeliefTensor
    pair_contact: Tensor
    pair_collision: Tensor
    ground_contact: Tensor
    ground_collision: Tensor
    pair_impulse: Tensor
    max_penetration: Tensor
    mean_penetration: Tensor
    action_reaction_residual: Tensor


class SphereContactResolver(nn.Module):
    """Analytic impulses with bounded learned scalar corrections."""

    def __init__(
        self,
        planes: Sequence[ContactPlane] | None = None,
        *,
        contact_margin: float = 1e-3,
        collision_speed_epsilon: float = 1e-4,
        penetration_fraction: float = 0.8,
        penetration_slop: float = 1e-4,
        max_position_correction: float = 0.05,
        max_impulse_multiplier_residual: float = 0.25,
        max_impulse_additive_residual: float = 0.1,
        contact_confidence_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        selected = tuple(planes or (ContactPlane((0.0, 1.0, 0.0)),))
        if not selected:
            raise ValueError("at least one environment plane is required")
        normals = torch.tensor([item.normal for item in selected], dtype=torch.float32)
        norms = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        if torch.any(norms < 1e-8):
            raise ValueError("contact plane normals must be nonzero")
        self.register_buffer("plane_normals", normals / norms)
        self.register_buffer(
            "plane_offsets",
            torch.tensor([item.offset for item in selected], dtype=torch.float32),
        )
        self.plane_names = tuple(item.name for item in selected)
        self.contact_margin = contact_margin
        self.collision_speed_epsilon = collision_speed_epsilon
        self.penetration_fraction = penetration_fraction
        self.penetration_slop = penetration_slop
        self.max_position_correction = max_position_correction
        self.max_impulse_multiplier_residual = max_impulse_multiplier_residual
        self.max_impulse_additive_residual = max_impulse_additive_residual
        if contact_confidence_sigma < 0:
            raise ValueError("contact_confidence_sigma must be nonnegative")
        self.contact_confidence_sigma = contact_confidence_sigma

    def forward(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None = None,
    ) -> ContactResult:
        return self.resolve(objects, graph)

    def resolve(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None = None,
    ) -> ContactResult:
        pair = self._resolve_pairs(objects, graph)
        pair_objects, pair_contact, pair_collision, pair_impulse, pair_penetration, residual = pair
        plane = self._resolve_planes(pair_objects)
        plane_objects, ground_contact, ground_collision, plane_penetration = plane
        all_penetration = torch.cat(
            (
                pair_penetration.flatten(start_dim=1),
                plane_penetration.flatten(start_dim=1),
            ),
            dim=-1,
        )
        return ContactResult(
            objects=plane_objects,
            pair_contact=pair_contact,
            pair_collision=pair_collision,
            ground_contact=ground_contact,
            ground_collision=ground_collision,
            pair_impulse=pair_impulse,
            max_penetration=all_penetration.max(dim=-1).values,
            mean_penetration=all_penetration.mean(dim=-1),
            action_reaction_residual=residual,
        )

    def _resolve_pairs(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None,
    ) -> tuple[ObjectBeliefTensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, count = objects.active.shape
        rel_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]
        distance = torch.linalg.vector_norm(rel_position, dim=-1).clamp_min(1e-7)
        normal = rel_position / distance.unsqueeze(-1)
        rel_velocity = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]
        relative_normal_velocity = (rel_velocity * normal).sum(dim=-1)
        radius = objects.radius.squeeze(-1)
        penetration = (radius[:, :, None] + radius[:, None, :] - distance).clamp_min(0.0)
        active_pair = objects.active[:, :, None] & objects.active[:, None, :]
        upper = torch.triu(
            torch.ones(count, count, device=objects.active.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        gap = distance - radius[:, :, None] - radius[:, None, :]
        position_variance = objects.fast_log_variance[..., :3].exp()
        relative_variance = position_variance[:, :, None, :] + position_variance[:, None, :, :]
        gap_sigma = (relative_variance * normal.square()).sum(dim=-1).sqrt()
        confident_gap = gap + self.contact_confidence_sigma * gap_sigma
        contact_upper = active_pair & upper & (confident_gap <= self.contact_margin)
        collision_upper = contact_upper & (relative_normal_velocity < -self.collision_speed_epsilon)

        inverse_mass = objects.mass.squeeze(-1).reciprocal()
        inverse_mass_sum = (inverse_mass[:, :, None] + inverse_mass[:, None, :]).clamp_min(1e-8)
        restitution = 0.5 * (
            objects.restitution.squeeze(-1)[:, :, None]
            + objects.restitution.squeeze(-1)[:, None, :]
        )
        impulse = (-(1.0 + restitution) * relative_normal_velocity / inverse_mass_sum).clamp_min(
            0.0
        )
        if graph is not None:
            multiplier = 1.0 + self.max_impulse_multiplier_residual * torch.tanh(
                graph.impulse_multiplier_raw
            )
            additive = self.max_impulse_additive_residual * torch.tanh(graph.impulse_additive_raw)
            impulse = (impulse * multiplier + additive).clamp_min(0.0)
        impulse = impulse * collision_upper

        rel_tangent = rel_velocity - relative_normal_velocity.unsqueeze(-1) * normal
        tangent_speed = torch.linalg.vector_norm(rel_tangent, dim=-1)
        tangent_direction = rel_tangent / tangent_speed.clamp_min(1e-7).unsqueeze(-1)
        friction = 0.5 * (
            objects.friction.squeeze(-1)[:, :, None] + objects.friction.squeeze(-1)[:, None, :]
        )
        friction_impulse = torch.minimum(
            friction * impulse,
            tangent_speed / inverse_mass_sum,
        )
        # Momentum applied to i for each upper pair.
        momentum_i_upper = (
            -impulse.unsqueeze(-1) * normal + friction_impulse.unsqueeze(-1) * tangent_direction
        )
        pair_momentum = momentum_i_upper - momentum_i_upper.transpose(1, 2)
        momentum_change = pair_momentum.sum(dim=2)
        velocity = objects.velocity + momentum_change * inverse_mass.unsqueeze(-1)

        correction_depth = (
            (penetration - self.penetration_slop).clamp_min(0.0) * self.penetration_fraction
        ).clamp_max(self.max_position_correction)
        correction_scale = correction_depth / inverse_mass_sum
        position_i_upper = (
            -correction_scale.unsqueeze(-1)
            * inverse_mass[:, :, None, None]
            * normal
            * contact_upper.unsqueeze(-1)
        )
        pair_position_change = position_i_upper - position_i_upper.transpose(1, 2)
        position = objects.position + pair_position_change.sum(dim=2)
        active_f = objects.active.unsqueeze(-1)
        updated = replace(
            objects,
            position=torch.where(active_f, position, objects.position),
            velocity=torch.where(active_f, velocity, objects.velocity),
        )
        pair_contact = contact_upper | contact_upper.transpose(1, 2)
        pair_collision = collision_upper | collision_upper.transpose(1, 2)
        pair_impulse = impulse + impulse.transpose(1, 2)
        action_reaction = pair_momentum.sum(dim=(1, 2))
        residual = torch.linalg.vector_norm(action_reaction, dim=-1)
        return (
            updated,
            pair_contact,
            pair_collision,
            pair_impulse,
            penetration * (contact_upper | contact_upper.transpose(1, 2)),
            residual,
        )

    def _resolve_planes(
        self,
        objects: ObjectBeliefTensor,
    ) -> tuple[ObjectBeliefTensor, Tensor, Tensor, Tensor]:
        normals = self.plane_normals.to(
            device=objects.position.device,
            dtype=objects.position.dtype,
        )
        offsets = self.plane_offsets.to(
            device=objects.position.device,
            dtype=objects.position.dtype,
        )
        signed_center = torch.einsum("bnc,pc->bnp", objects.position, normals)
        gap = signed_center - offsets[None, None, :] - objects.radius
        position_variance = objects.fast_log_variance[..., :3].exp()
        gap_sigma = torch.einsum(
            "bnc,pc->bnp",
            position_variance,
            normals.square(),
        ).sqrt()
        confident_gap = gap + self.contact_confidence_sigma * gap_sigma
        normal_velocity = torch.einsum("bnc,pc->bnp", objects.velocity, normals)
        contact = objects.active.unsqueeze(-1) & (confident_gap <= self.contact_margin)
        collision = contact & (normal_velocity < -self.collision_speed_epsilon)
        penetration = (-gap).clamp_min(0.0) * contact

        restitution = objects.restitution
        delta_normal_speed = (-(1.0 + restitution) * normal_velocity).clamp_min(0.0) * collision
        delta_velocity_normal = torch.einsum("bnp,pc->bnc", delta_normal_speed, normals)
        velocity_after_normal = objects.velocity + delta_velocity_normal

        # Coulomb-like bounded tangential damping for simultaneous planes.
        projected = torch.einsum("bnc,pc->bnp", velocity_after_normal, normals)
        tangential = (
            velocity_after_normal[:, :, None, :]
            - projected.unsqueeze(-1) * normals[None, None, :, :]
        )
        tangent_speed = torch.linalg.vector_norm(tangential, dim=-1)
        normal_impulse_speed = delta_normal_speed
        reduction = torch.minimum(
            tangent_speed,
            objects.friction * normal_impulse_speed,
        )
        delta_tangent = (
            -tangential
            / tangent_speed.clamp_min(1e-7).unsqueeze(-1)
            * reduction.unsqueeze(-1)
            * collision.unsqueeze(-1)
        ).sum(dim=2)
        velocity = velocity_after_normal + delta_tangent

        correction = (
            (penetration - self.penetration_slop).clamp_min(0.0) * self.penetration_fraction
        ).clamp_max(self.max_position_correction)
        position = objects.position + torch.einsum("bnp,pc->bnc", correction, normals)
        active_f = objects.active.unsqueeze(-1)
        updated = replace(
            objects,
            position=torch.where(active_f, position, objects.position),
            velocity=torch.where(active_f, velocity, objects.velocity),
        )
        return (
            updated,
            contact.any(dim=-1),
            collision.any(dim=-1),
            penetration,
        )
