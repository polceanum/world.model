"""Structured sphere-pair and sphere-plane contact resolution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from world_model.belief import ObjectBeliefTensor
from world_model.dynamics.graph import InteractionOutput

_TANGENT_DIRECTION_EPSILON = 1.0e-7


def _safe_tangent_direction(tangential: Tensor, tangent_speed: Tensor) -> Tensor:
    """Return a finite tangent direction, including an exact rest contact.

    A zero tangential speed has no friction direction.  Dividing by a tiny
    clamped epsilon is algebraically harmless on CPU, but some MPS kernels
    flush that subnormal denominator and form ``0 / 0`` before a later
    collision mask can remove the term.  Selecting a unit denominator for the
    zero/near-zero branch retains a zero vector there while preserving the
    ordinary normalized direction everywhere it is physically meaningful.
    """

    if tangential.shape != tangent_speed.shape + (3,):
        raise ValueError("tangential vector and speed shapes are inconsistent")
    denominator = torch.where(
        tangent_speed > _TANGENT_DIRECTION_EPSILON,
        tangent_speed,
        torch.ones_like(tangent_speed),
    )
    return tangential / denominator.unsqueeze(-1)


@dataclass(frozen=True)
class ContactPlane:
    """Plane whose valid half-space satisfies ``normal·position >= offset``."""

    normal: tuple[float, float, float]
    offset: float = 0.0
    name: str = "ground"
    is_ground: bool | None = None


@dataclass
class ContactResult:
    objects: ObjectBeliefTensor
    # Endpoint contact from the final solver iteration drives persistent
    # motion modes. The interval fields retain any contact encountered while
    # resolving the substep for diagnostics/labels.
    pair_contact: Tensor
    interval_pair_contact: Tensor
    pair_collision: Tensor
    boundary_contact: Tensor
    interval_boundary_contact: Tensor
    boundary_collision: Tensor
    ground_contact: Tensor
    interval_ground_contact: Tensor
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
        contact_margin: float = 0.0,
        boundary_contact_tolerance: float | None = 1.0e-4,
        collision_speed_epsilon: float = 1e-7,
        boundary_collision_speed_epsilon: float = 0.1,
        solver_iterations: int = 2,
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
        self.register_buffer(
            "ground_plane_mask",
            torch.tensor(
                [
                    (
                        item.is_ground
                        if item.is_ground is not None
                        else item.name in {"ground", "floor", "y_minimum"}
                    )
                    for item in selected
                ],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.plane_names = tuple(item.name for item in selected)
        self.contact_margin = contact_margin
        self.boundary_contact_tolerance = (
            contact_margin if boundary_contact_tolerance is None else boundary_contact_tolerance
        )
        self.collision_speed_epsilon = collision_speed_epsilon
        self.boundary_collision_speed_epsilon = boundary_collision_speed_epsilon
        self.solver_iterations = solver_iterations
        self.penetration_fraction = penetration_fraction
        self.penetration_slop = penetration_slop
        self.max_position_correction = max_position_correction
        self.max_impulse_multiplier_residual = max_impulse_multiplier_residual
        self.max_impulse_additive_residual = max_impulse_additive_residual
        if contact_confidence_sigma < 0:
            raise ValueError("contact_confidence_sigma must be nonnegative")
        if collision_speed_epsilon < 0 or not math.isfinite(collision_speed_epsilon):
            raise ValueError("collision_speed_epsilon must be finite and nonnegative")
        if boundary_collision_speed_epsilon < 0 or not math.isfinite(
            boundary_collision_speed_epsilon
        ):
            raise ValueError("boundary_collision_speed_epsilon must be finite and nonnegative")
        if self.boundary_contact_tolerance < 0 or not math.isfinite(
            self.boundary_contact_tolerance
        ):
            raise ValueError("boundary_contact_tolerance must be finite and nonnegative")
        if contact_margin < 0 or not math.isfinite(contact_margin):
            raise ValueError("contact_margin must be finite and nonnegative")
        if solver_iterations < 1:
            raise ValueError("solver_iterations must be at least one")
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
        """Resolve contacts with the same iteration order as the simulator.

        The reference simulator resolves every boundary before sphere pairs and
        repeats that sequence for each solver iteration.  Matching that order is
        important for compound contacts: a wall projection can create or remove
        a pair overlap, while a pair impulse can push a body into a boundary for
        the next iteration.
        """

        batch, count = objects.active.shape
        plane_count = len(self.plane_names)
        pair_contact = torch.zeros(
            batch,
            count,
            count,
            dtype=torch.bool,
            device=objects.position.device,
        )
        interval_pair_contact = torch.zeros_like(pair_contact)
        pair_collision = torch.zeros_like(pair_contact)
        boundary_contact = torch.zeros(
            batch,
            count,
            plane_count,
            dtype=torch.bool,
            device=objects.position.device,
        )
        interval_boundary_contact = torch.zeros_like(boundary_contact)
        boundary_collision = torch.zeros_like(boundary_contact)
        pair_impulse = objects.position.new_zeros(batch, count, count)
        pair_penetration = objects.position.new_zeros(batch, count, count)
        plane_penetration = objects.position.new_zeros(batch, count, plane_count)
        residual = objects.position.new_zeros(batch)
        updated = objects

        for _ in range(self.solver_iterations):
            (
                updated,
                iteration_boundary_contact,
                iteration_boundary_collision,
                _,
                _,
                iteration_plane_penetration,
            ) = self._resolve_planes(updated)
            (
                updated,
                iteration_pair_contact,
                iteration_pair_collision,
                iteration_pair_impulse,
                iteration_pair_penetration,
                iteration_residual,
            ) = self._resolve_pairs(updated, graph)
            boundary_contact = iteration_boundary_contact
            interval_boundary_contact = interval_boundary_contact | iteration_boundary_contact
            boundary_collision = boundary_collision | iteration_boundary_collision
            pair_contact = iteration_pair_contact
            interval_pair_contact = interval_pair_contact | iteration_pair_contact
            pair_collision = pair_collision | iteration_pair_collision
            pair_impulse = torch.maximum(pair_impulse, iteration_pair_impulse)
            pair_penetration = torch.maximum(pair_penetration, iteration_pair_penetration)
            plane_penetration = torch.maximum(
                plane_penetration,
                iteration_plane_penetration,
            )
            residual = torch.maximum(residual, iteration_residual)

        # Solver-occurrence contact is interval evidence. Persistent motion
        # modes need contact at the fully resolved endpoint, after the final
        # pair correction has had a chance to move an object onto or away from
        # a boundary.
        pair_contact = self._measure_pair_contact(updated)
        boundary_contact = self._measure_boundary_contact(updated)
        ground_plane_mask = self.ground_plane_mask.to(device=objects.position.device)
        ground_contact = (boundary_contact & ground_plane_mask).any(dim=-1)
        interval_ground_contact = (interval_boundary_contact & ground_plane_mask).any(dim=-1)
        ground_collision = (boundary_collision & ground_plane_mask).any(dim=-1)
        all_penetration = torch.cat(
            (
                pair_penetration.flatten(start_dim=1),
                plane_penetration.flatten(start_dim=1),
            ),
            dim=-1,
        )
        return ContactResult(
            objects=updated,
            pair_contact=pair_contact,
            interval_pair_contact=interval_pair_contact,
            pair_collision=pair_collision,
            boundary_contact=boundary_contact,
            interval_boundary_contact=interval_boundary_contact,
            boundary_collision=boundary_collision,
            ground_contact=ground_contact,
            interval_ground_contact=interval_ground_contact,
            ground_collision=ground_collision,
            pair_impulse=pair_impulse,
            max_penetration=all_penetration.max(dim=-1).values,
            mean_penetration=all_penetration.mean(dim=-1),
            action_reaction_residual=residual,
        )

    def _measure_pair_contact(self, objects: ObjectBeliefTensor) -> Tensor:
        """Measure symmetric pair contact without applying another jump."""

        _, count = objects.active.shape
        rel_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]
        distance = torch.linalg.vector_norm(rel_position, dim=-1).clamp_min(1e-7)
        normal = rel_position / distance.unsqueeze(-1)
        radius = objects.radius.squeeze(-1)
        gap = distance - radius[:, :, None] - radius[:, None, :]
        position_variance = objects.fast_log_variance[..., :3].exp()
        relative_variance = position_variance[:, :, None, :] + position_variance[:, None, :, :]
        gap_sigma = (relative_variance * normal.square()).sum(dim=-1).sqrt()
        confident_gap = gap + self.contact_confidence_sigma * gap_sigma
        upper = torch.triu(
            torch.ones(count, count, device=objects.active.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        active_pair = objects.active[:, :, None] & objects.active[:, None, :]
        contact_upper = active_pair & upper & (confident_gap < self.contact_margin)
        return contact_upper | contact_upper.transpose(1, 2)

    def _measure_boundary_contact(self, objects: ObjectBeliefTensor) -> Tensor:
        """Measure all endpoint plane contacts without applying another jump."""

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
        return objects.active.unsqueeze(-1) & (confident_gap <= self.boundary_contact_tolerance)

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
        contact_upper = active_pair & upper & (confident_gap < self.contact_margin)
        collision_upper = contact_upper & (relative_normal_velocity < -self.collision_speed_epsilon)

        inverse_mass = objects.mass.squeeze(-1).reciprocal()
        inverse_mass_sum = (inverse_mass[:, :, None] + inverse_mass[:, None, :]).clamp_min(1e-8)
        restitution = torch.minimum(
            objects.restitution.squeeze(-1)[:, :, None],
            objects.restitution.squeeze(-1)[:, None, :],
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
        tangent_direction = _safe_tangent_direction(rel_tangent, tangent_speed)
        friction = torch.sqrt(
            objects.friction.squeeze(-1)[:, :, None].clamp_min(0.0)
            * objects.friction.squeeze(-1)[:, None, :].clamp_min(0.0)
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

        correction_scale = (
            self.penetration_fraction
            * (penetration - self.penetration_slop).clamp_min(0.0)
            / inverse_mass_sum
        ).clamp_max(self.max_position_correction)
        position_first_upper = (
            -correction_scale.unsqueeze(-1)
            * inverse_mass[:, :, None, None]
            * normal
            * contact_upper.unsqueeze(-1)
        )
        position_second_upper = (
            correction_scale.unsqueeze(-1)
            * inverse_mass[:, None, :, None]
            * normal
            * contact_upper.unsqueeze(-1)
        )
        # Unlike an impulse, positional projection is not equal-and-opposite:
        # each body's displacement is weighted by its own inverse mass.
        position_change = position_first_upper.sum(dim=2) + position_second_upper.sum(dim=1)
        position = objects.position + position_change
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
    ) -> tuple[ObjectBeliefTensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        normals = self.plane_normals.to(
            device=objects.position.device,
            dtype=objects.position.dtype,
        )
        offsets = self.plane_offsets.to(
            device=objects.position.device,
            dtype=objects.position.dtype,
        )
        position_variance = objects.fast_log_variance[..., :3].exp()
        position = objects.position
        velocity = objects.velocity
        contacts: list[Tensor] = []
        collisions: list[Tensor] = []
        penetrations: list[Tensor] = []

        # Resolve planes sequentially.  This duplicates the simulator's stable
        # x-min/x-max/floor/ceiling/z-min/z-max behaviour and avoids summing
        # multiple independently computed friction impulses at corners.
        for plane_index in range(normals.shape[0]):
            normal = normals[plane_index]
            offset = offsets[plane_index]
            signed_center = (position * normal).sum(dim=-1)
            gap = signed_center - offset - objects.radius.squeeze(-1)
            gap_sigma = (position_variance * normal.square()).sum(dim=-1).sqrt()
            confident_gap = gap + self.contact_confidence_sigma * gap_sigma
            contact = objects.active & (confident_gap <= self.boundary_contact_tolerance)
            normal_velocity = (velocity * normal).sum(dim=-1)
            collision = contact & (normal_velocity < -self.boundary_collision_speed_epsilon)
            resting_inward = contact & (normal_velocity < 0.0) & ~collision
            penetration = (-gap).clamp_min(0.0) * contact

            collision_delta = (
                -(1.0 + objects.restitution.squeeze(-1)) * normal_velocity
            ).clamp_min(0.0) * collision
            resting_delta = (-normal_velocity).clamp_min(0.0) * resting_inward
            delta_normal_speed = collision_delta + resting_delta
            velocity = velocity + delta_normal_speed.unsqueeze(-1) * normal

            tangential = velocity - (velocity * normal).sum(dim=-1).unsqueeze(-1) * normal
            tangent_speed = torch.linalg.vector_norm(tangential, dim=-1)
            reduction = torch.minimum(
                tangent_speed,
                objects.friction.squeeze(-1).clamp_min(0.0) * delta_normal_speed,
            )
            velocity = velocity + (
                -_safe_tangent_direction(tangential, tangent_speed)
                * reduction.unsqueeze(-1)
                * collision.unsqueeze(-1)
            )

            # Simulator boundaries are hard constraints: project the complete
            # penetration, rather than applying the pair solver's soft,
            # inverse-mass-weighted correction.
            position = position + penetration.unsqueeze(-1) * normal
            contacts.append(contact)
            collisions.append(collision)
            penetrations.append(penetration)

        contact = torch.stack(contacts, dim=-1)
        collision = torch.stack(collisions, dim=-1)
        penetration = torch.stack(penetrations, dim=-1)
        active_f = objects.active.unsqueeze(-1)
        updated = replace(
            objects,
            position=torch.where(active_f, position, objects.position),
            velocity=torch.where(active_f, velocity, objects.velocity),
        )
        ground_plane_mask = self.ground_plane_mask.to(device=objects.position.device)
        ground_contact = (contact & ground_plane_mask).any(dim=-1)
        ground_collision = (collision & ground_plane_mask).any(dim=-1)
        return (
            updated,
            contact,
            collision,
            ground_contact,
            ground_collision,
            penetration,
        )
