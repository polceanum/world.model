"""Bounded online identification from public rigid-body motion evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from world_model.belief import WorldBelief, slow_packing_map
from world_model.identification.parameters import ParameterBounds


@dataclass(frozen=True)
class KnownImpulseEvidence:
    object_id: Tensor
    velocity_before: Tensor
    velocity_after: Tensor
    impulse_world: Tensor


@dataclass(frozen=True)
class FreeMotionEvidence:
    object_id: Tensor
    velocity_before: Tensor
    velocity_after: Tensor
    duration_seconds: Tensor


@dataclass(frozen=True)
class FreeMotionPositionEvidence:
    """Public metric positions for an uninterrupted force-free interval."""

    object_id: Tensor
    positions_world: Tensor
    timestamps: Tensor


@dataclass(frozen=True)
class KnownImpulsePositionEvidence:
    """Public position traces immediately before and after a known impulse."""

    object_id: Tensor
    positions_before_world: Tensor
    timestamps_before: Tensor
    positions_after_world: Tensor
    timestamps_after: Tensor
    action_timestamp: Tensor
    impulse_world: Tensor


@dataclass(frozen=True)
class PairCollisionEvidence:
    first_object_id: Tensor
    second_object_id: Tensor
    normal_world: Tensor
    first_velocity_before: Tensor
    first_velocity_after: Tensor
    second_velocity_before: Tensor
    second_velocity_after: Tensor
    relative_contact_velocity_before: Tensor | None = None
    relative_contact_velocity_after: Tensor | None = None


@dataclass(frozen=True)
class BoundaryCollisionEvidence:
    """Public before/after motion around one stationary boundary impact."""

    object_id: Tensor
    normal_world: Tensor
    velocity_before: Tensor
    velocity_after: Tensor
    relative_contact_velocity_before: Tensor | None = None
    relative_contact_velocity_after: Tensor | None = None


@dataclass(frozen=True)
class BoundaryCollisionPositionEvidence:
    """Public position traces bracketing one stationary-boundary impact."""

    object_id: Tensor
    normal_world: Tensor
    positions_before_world: Tensor
    timestamps_before: Tensor
    positions_after_world: Tensor
    timestamps_after: Tensor
    collision_timestamp: Tensor


@dataclass(frozen=True)
class RigidParameterUpdate:
    parameter: str
    object_ids: Tensor
    estimate: Tensor
    accepted: Tensor


class OnlineRigidParameterEstimator:
    """Analytic belief updates gated by identifiable public interventions."""

    def __init__(
        self,
        *,
        gain: float = 0.65,
        variance_contraction: float = 1.0,
        bounds: ParameterBounds | None = None,
    ) -> None:
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must lie in (0,1]")
        if not math.isfinite(variance_contraction) or variance_contraction <= 0.0:
            raise ValueError("variance_contraction must be finite and positive")
        self.gain = float(gain)
        self.variance_contraction = float(variance_contraction)
        self.bounds = bounds or ParameterBounds()

    @staticmethod
    def _validate_vector(name: str, value: Tensor, belief: WorldBelief) -> None:
        if value.shape != (belief.batch_size, 3):
            raise ValueError(f"{name} must have shape [B,3]")
        if value.device != belief.device or value.dtype != belief.dtype:
            raise ValueError(f"{name} must match belief dtype and device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")

    @staticmethod
    def _validate_position_trace(
        name: str,
        positions: Tensor,
        timestamps: Tensor,
        belief: WorldBelief,
        *,
        minimum_samples: int,
    ) -> None:
        if (
            positions.ndim != 3
            or positions.shape[0] != belief.batch_size
            or positions.shape[2] != 3
        ):
            raise ValueError(f"{name} positions must have shape [B,T,3]")
        if positions.shape[1] < minimum_samples:
            raise ValueError(f"{name} requires at least {minimum_samples} samples")
        if timestamps.shape != positions.shape[:2]:
            raise ValueError(f"{name} timestamps must have shape [B,T]")
        for value_name, value in (("positions", positions), ("timestamps", timestamps)):
            if value.device != belief.device or value.dtype != belief.dtype:
                raise ValueError(f"{name} {value_name} must match belief dtype and device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} {value_name} must be finite")
        if torch.any(timestamps[:, 1:] <= timestamps[:, :-1]):
            raise ValueError(f"{name} timestamps must be strictly increasing")

    @staticmethod
    def _fit_velocity_at_timestamp(
        positions: Tensor,
        timestamps: Tensor,
        reference_timestamp: Tensor,
        drag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fit exponential-drag motion and return velocity at ``reference``."""

        velocities = []
        residuals = []
        for row in range(positions.shape[0]):
            relative = timestamps[row] - reference_timestamp[row]
            row_drag = drag[row].clamp_min(1.0e-8)
            displacement_scale = -torch.expm1(-row_drag * relative) / row_drag
            design = torch.stack((torch.ones_like(relative), displacement_scale), dim=-1)
            solution = torch.linalg.lstsq(design, positions[row]).solution
            fitted = design @ solution
            velocities.append(solution[1])
            residuals.append((fitted - positions[row]).square().mean().sqrt())
        return torch.stack(velocities), torch.stack(residuals)

    @staticmethod
    def _validate_fit_residual(maximum_fit_residual_m: float) -> None:
        if not math.isfinite(maximum_fit_residual_m) or maximum_fit_residual_m <= 0.0:
            raise ValueError("maximum_fit_residual_m must be finite and positive")

    @staticmethod
    def _slots(object_id: Tensor, belief: WorldBelief) -> Tensor:
        if object_id.shape != (belief.batch_size,) or object_id.dtype is not torch.int64:
            raise ValueError("object_id must have shape [B] and dtype int64")
        if object_id.device != belief.device:
            raise ValueError("object_id must match belief device")
        matches = belief.objects.active & (belief.objects.object_id == object_id.unsqueeze(-1))
        if not torch.all(matches.sum(dim=-1) == 1):
            raise ValueError("each evidence object_id must resolve exactly once")
        return matches.to(torch.int64).argmax(dim=-1)

    def _update_scalar(
        self,
        belief: WorldBelief,
        *,
        slots: Tensor,
        estimate: Tensor,
        accepted: Tensor,
        parameter: str,
    ) -> WorldBelief:
        objects = belief.objects.clone()
        batch_index = torch.arange(belief.batch_size, device=belief.device)
        if parameter == "mass":
            lower, upper = self.bounds.mass
            transformed = estimate.clamp(lower, upper).log()
            field = objects.log_mass
            variance_slice = slow_packing_map(objects)["log_mass"]
        elif parameter == "drag":
            lower, upper = self.bounds.drag
            transformed = estimate.clamp(lower, upper).log()
            field = objects.log_drag
            variance_slice = slow_packing_map(objects)["log_drag"]
        elif parameter == "restitution":
            lower, upper = self.bounds.restitution
            transformed = torch.logit(estimate.clamp(lower, upper))
            field = objects.restitution_logit
            variance_slice = slow_packing_map(objects)["restitution_logit"]
        elif parameter == "friction":
            lower, upper = self.bounds.friction
            transformed = torch.logit(estimate.clamp(lower, upper))
            field = objects.friction_logit
            variance_slice = slow_packing_map(objects)["friction_logit"]
        else:  # pragma: no cover - private dispatch guard
            raise ValueError(f"unsupported parameter {parameter}")
        current = field[batch_index, slots, 0]
        updated = torch.where(
            accepted,
            (1.0 - self.gain) * current + self.gain * transformed,
            current,
        )
        field[batch_index, slots, 0] = updated
        variance = objects.slow_log_variance[
            batch_index,
            slots,
            variance_slice,
        ]
        contraction = torch.where(
            accepted.unsqueeze(-1),
            variance.new_full(variance.shape, self.variance_contraction),
            torch.zeros_like(variance),
        )
        objects.slow_log_variance[batch_index, slots, variance_slice] = (
            variance - contraction
        ).clamp_min(-20.0)
        return replace(belief, objects=objects).validate()

    def update_known_impulse(
        self,
        belief: WorldBelief,
        evidence: KnownImpulseEvidence,
    ) -> tuple[WorldBelief, RigidParameterUpdate]:
        """Estimate mass from an isolated, known centre-of-mass impulse."""

        slots = self._slots(evidence.object_id, belief)
        for name, value in (
            ("velocity_before", evidence.velocity_before),
            ("velocity_after", evidence.velocity_after),
            ("impulse_world", evidence.impulse_world),
        ):
            self._validate_vector(name, value, belief)
        delta = evidence.velocity_after - evidence.velocity_before
        denominator = delta.square().sum(dim=-1)
        estimate = (evidence.impulse_world * delta).sum(dim=-1) / denominator.clamp_min(1.0e-10)
        residual = torch.linalg.vector_norm(
            evidence.impulse_world - estimate.unsqueeze(-1) * delta,
            dim=-1,
        )
        impulse_norm = torch.linalg.vector_norm(evidence.impulse_world, dim=-1)
        accepted = (
            (denominator > 1.0e-8)
            & (estimate > 0.0)
            & (residual <= 0.05 * impulse_norm.clamp_min(1.0e-8))
        )
        updated = self._update_scalar(
            belief,
            slots=slots,
            estimate=estimate,
            accepted=accepted,
            parameter="mass",
        )
        return updated, RigidParameterUpdate("mass", evidence.object_id.clone(), estimate, accepted)

    def update_free_motion(
        self,
        belief: WorldBelief,
        evidence: FreeMotionEvidence,
    ) -> tuple[WorldBelief, RigidParameterUpdate]:
        """Estimate linear drag from a contact-free, force-free interval."""

        slots = self._slots(evidence.object_id, belief)
        self._validate_vector("velocity_before", evidence.velocity_before, belief)
        self._validate_vector("velocity_after", evidence.velocity_after, belief)
        duration = evidence.duration_seconds
        if duration.shape != (belief.batch_size,) or duration.dtype != belief.dtype:
            raise ValueError("duration_seconds must have shape [B] and belief dtype")
        if duration.device != belief.device or not torch.isfinite(duration).all():
            raise ValueError("duration_seconds must be finite on the belief device")
        before = torch.linalg.vector_norm(evidence.velocity_before, dim=-1)
        after = torch.linalg.vector_norm(evidence.velocity_after, dim=-1)
        ratio = after / before.clamp_min(1.0e-10)
        estimate = -torch.log(ratio.clamp_min(1.0e-10)) / duration.clamp_min(1.0e-10)
        direction_before = evidence.velocity_before / before.clamp_min(1.0e-10).unsqueeze(-1)
        direction_after = evidence.velocity_after / after.clamp_min(1.0e-10).unsqueeze(-1)
        direction_error = 1.0 - (direction_before * direction_after).sum(dim=-1)
        accepted = (
            (duration > 0.0)
            & (before > 1.0e-4)
            & (after > 0.0)
            & (after <= before)
            & (direction_error.abs() < 0.02)
            & torch.isfinite(estimate)
        )
        updated = self._update_scalar(
            belief,
            slots=slots,
            estimate=estimate,
            accepted=accepted,
            parameter="drag",
        )
        return updated, RigidParameterUpdate("drag", evidence.object_id.clone(), estimate, accepted)

    def update_free_motion_positions(
        self,
        belief: WorldBelief,
        evidence: FreeMotionPositionEvidence,
        *,
        maximum_fit_residual_m: float = 0.01,
    ) -> tuple[WorldBelief, RigidParameterUpdate]:
        """Estimate drag from a public position series with bounded grid fitting."""

        slots = self._slots(evidence.object_id, belief)
        self._validate_position_trace(
            "free motion",
            evidence.positions_world,
            evidence.timestamps,
            belief,
            minimum_samples=4,
        )
        self._validate_fit_residual(maximum_fit_residual_m)
        estimates = []
        accepted_rows = []
        lower, upper = self.bounds.drag
        candidates = torch.logspace(
            math.log10(lower),
            math.log10(upper),
            257,
            device=belief.device,
            dtype=belief.dtype,
        )
        for row in range(belief.batch_size):
            positions = evidence.positions_world[row]
            timestamps = evidence.timestamps[row]
            reference = timestamps[0]
            losses = []
            velocities = []
            for candidate in candidates:
                velocity, residual = self._fit_velocity_at_timestamp(
                    positions.unsqueeze(0),
                    timestamps.unsqueeze(0),
                    reference.unsqueeze(0),
                    candidate.unsqueeze(0),
                )
                losses.append(residual[0])
                velocities.append(velocity[0])
            loss = torch.stack(losses)
            best = int(loss.argmin())
            duration = timestamps[-1] - timestamps[0]
            accepted = bool(
                duration >= 0.25
                and torch.linalg.vector_norm(velocities[best]) > 1.0e-4
                and loss[best] <= maximum_fit_residual_m
            )
            estimates.append(candidates[best])
            accepted_rows.append(accepted)
        estimate = torch.stack(estimates)
        accepted = torch.tensor(accepted_rows, device=belief.device, dtype=torch.bool)
        updated = self._update_scalar(
            belief,
            slots=slots,
            estimate=estimate,
            accepted=accepted,
            parameter="drag",
        )
        return updated, RigidParameterUpdate(
            "drag",
            evidence.object_id.clone(),
            estimate,
            accepted,
        )

    def update_known_impulse_positions(
        self,
        belief: WorldBelief,
        evidence: KnownImpulsePositionEvidence,
        *,
        maximum_fit_residual_m: float = 0.01,
    ) -> tuple[WorldBelief, RigidParameterUpdate]:
        """Estimate mass from public position traces around a known impulse."""

        slots = self._slots(evidence.object_id, belief)
        self._validate_fit_residual(maximum_fit_residual_m)
        for name, positions, timestamps in (
            ("pre-impulse", evidence.positions_before_world, evidence.timestamps_before),
            ("post-impulse", evidence.positions_after_world, evidence.timestamps_after),
        ):
            self._validate_position_trace(
                name,
                positions,
                timestamps,
                belief,
                minimum_samples=3,
            )
        action_timestamp = evidence.action_timestamp
        if (
            action_timestamp.shape != (belief.batch_size,)
            or action_timestamp.device != belief.device
            or action_timestamp.dtype != belief.dtype
            or not torch.isfinite(action_timestamp).all()
        ):
            raise ValueError("action_timestamp must be finite [B] on the belief device")
        if torch.any(evidence.timestamps_before[:, -1] > action_timestamp) or torch.any(
            evidence.timestamps_after[:, 0] < action_timestamp
        ):
            raise ValueError("known-impulse traces must bracket the action timestamp")
        self._validate_vector("impulse_world", evidence.impulse_world, belief)
        batch_index = torch.arange(belief.batch_size, device=belief.device)
        drag = belief.objects.drag[batch_index, slots, 0]
        before, before_residual = self._fit_velocity_at_timestamp(
            evidence.positions_before_world,
            evidence.timestamps_before,
            action_timestamp,
            drag,
        )
        after, after_residual = self._fit_velocity_at_timestamp(
            evidence.positions_after_world,
            evidence.timestamps_after,
            action_timestamp,
            drag,
        )
        fit_accepted = (before_residual <= maximum_fit_residual_m) & (
            after_residual <= maximum_fit_residual_m
        )
        safe_after = torch.where(fit_accepted.unsqueeze(-1), after, before)
        return self.update_known_impulse(
            belief,
            KnownImpulseEvidence(
                evidence.object_id,
                before,
                safe_after,
                evidence.impulse_world,
            ),
        )

    def update_pair_collision(
        self,
        belief: WorldBelief,
        evidence: PairCollisionEvidence,
    ) -> tuple[WorldBelief, tuple[RigidParameterUpdate, RigidParameterUpdate]]:
        """Estimate effective restitution and sliding friction from one impact."""

        first_slots = self._slots(evidence.first_object_id, belief)
        second_slots = self._slots(evidence.second_object_id, belief)
        if torch.any(first_slots == second_slots):
            raise ValueError("pair evidence must address distinct objects")
        for name, value in (
            ("normal_world", evidence.normal_world),
            ("first_velocity_before", evidence.first_velocity_before),
            ("first_velocity_after", evidence.first_velocity_after),
            ("second_velocity_before", evidence.second_velocity_before),
            ("second_velocity_after", evidence.second_velocity_after),
        ):
            self._validate_vector(name, value, belief)
        normal_norm = torch.linalg.vector_norm(evidence.normal_world, dim=-1)
        if torch.any(normal_norm <= 1.0e-8):
            raise ValueError("collision normal must be nonzero")
        normal = evidence.normal_world / normal_norm.unsqueeze(-1)
        relative_before = (
            evidence.relative_contact_velocity_before
            if evidence.relative_contact_velocity_before is not None
            else evidence.second_velocity_before - evidence.first_velocity_before
        )
        relative_after = (
            evidence.relative_contact_velocity_after
            if evidence.relative_contact_velocity_after is not None
            else evidence.second_velocity_after - evidence.first_velocity_after
        )
        self._validate_vector("relative_contact_velocity_before", relative_before, belief)
        self._validate_vector("relative_contact_velocity_after", relative_after, belief)
        before_normal = (relative_before * normal).sum(dim=-1)
        after_normal = (relative_after * normal).sum(dim=-1)
        restitution = -after_normal / before_normal.clamp(max=-1.0e-8)
        collision_accepted = (
            (before_normal < -1.0e-4) & (after_normal >= 0.0) & torch.isfinite(restitution)
        )
        object_ids = torch.cat(
            (evidence.first_object_id, evidence.second_object_id),
            dim=0,
        )
        slots = torch.cat((first_slots, second_slots), dim=0)
        restitution_values = torch.cat((restitution, restitution), dim=0)
        restitution_accepted = torch.cat((collision_accepted, collision_accepted), dim=0)
        repeated = _repeat_belief_rows(belief, 2)
        repeated = self._update_scalar(
            repeated,
            slots=slots,
            estimate=restitution_values,
            accepted=restitution_accepted,
            parameter="restitution",
        )
        belief = _merge_repeated_belief(repeated, belief, parameter="restitution")

        batch_index = torch.arange(belief.batch_size, device=belief.device)
        first_mass = belief.objects.mass[batch_index, first_slots, 0]
        impulse_on_first = first_mass.unsqueeze(-1) * (
            evidence.first_velocity_after - evidence.first_velocity_before
        )
        normal_impulse = (impulse_on_first * normal).sum(dim=-1).abs()
        tangent_impulse = torch.linalg.vector_norm(
            impulse_on_first - (impulse_on_first * normal).sum(dim=-1, keepdim=True) * normal,
            dim=-1,
        )
        friction = tangent_impulse / normal_impulse.clamp_min(1.0e-8)
        tangential_speed = torch.linalg.vector_norm(
            relative_before - before_normal.unsqueeze(-1) * normal,
            dim=-1,
        )
        friction_accepted = collision_accepted & (tangential_speed > 1.0e-3)
        friction_values = torch.cat((friction, friction), dim=0)
        friction_accept_rows = torch.cat((friction_accepted, friction_accepted), dim=0)
        repeated = _repeat_belief_rows(belief, 2)
        repeated = self._update_scalar(
            repeated,
            slots=slots,
            estimate=friction_values,
            accepted=friction_accept_rows,
            parameter="friction",
        )
        belief = _merge_repeated_belief(repeated, belief, parameter="friction")
        return belief, (
            RigidParameterUpdate(
                "restitution",
                object_ids,
                restitution_values,
                restitution_accepted,
            ),
            RigidParameterUpdate(
                "friction",
                object_ids,
                friction_values,
                friction_accept_rows,
            ),
        )

    def update_boundary_collision(
        self,
        belief: WorldBelief,
        evidence: BoundaryCollisionEvidence,
    ) -> tuple[WorldBelief, tuple[RigidParameterUpdate, RigidParameterUpdate]]:
        """Estimate one object's restitution and friction at a fixed boundary.

        The boundary normal and before/after velocities are observable metric
        quantities.  A caller may provide contact-point relative velocities
        when angular motion is observable; otherwise the centre-of-mass
        velocities provide the conservative linear estimate.
        """

        slots = self._slots(evidence.object_id, belief)
        for name, value in (
            ("normal_world", evidence.normal_world),
            ("velocity_before", evidence.velocity_before),
            ("velocity_after", evidence.velocity_after),
        ):
            self._validate_vector(name, value, belief)
        normal_norm = torch.linalg.vector_norm(evidence.normal_world, dim=-1)
        if torch.any(normal_norm <= 1.0e-8):
            raise ValueError("boundary normal must be nonzero")
        normal = evidence.normal_world / normal_norm.unsqueeze(-1)
        relative_before = (
            evidence.relative_contact_velocity_before
            if evidence.relative_contact_velocity_before is not None
            else evidence.velocity_before
        )
        relative_after = (
            evidence.relative_contact_velocity_after
            if evidence.relative_contact_velocity_after is not None
            else evidence.velocity_after
        )
        self._validate_vector("relative_contact_velocity_before", relative_before, belief)
        self._validate_vector("relative_contact_velocity_after", relative_after, belief)
        before_normal = (relative_before * normal).sum(dim=-1)
        after_normal = (relative_after * normal).sum(dim=-1)
        restitution = -after_normal / before_normal.clamp(max=-1.0e-8)
        collision_accepted = (
            (before_normal < -1.0e-4) & (after_normal >= 0.0) & torch.isfinite(restitution)
        )
        belief = self._update_scalar(
            belief,
            slots=slots,
            estimate=restitution,
            accepted=collision_accepted,
            parameter="restitution",
        )

        batch_index = torch.arange(belief.batch_size, device=belief.device)
        mass = belief.objects.mass[batch_index, slots, 0]
        impulse = mass.unsqueeze(-1) * (evidence.velocity_after - evidence.velocity_before)
        normal_impulse = (impulse * normal).sum(dim=-1).abs()
        tangent_impulse = torch.linalg.vector_norm(
            impulse - (impulse * normal).sum(dim=-1, keepdim=True) * normal,
            dim=-1,
        )
        friction = tangent_impulse / normal_impulse.clamp_min(1.0e-8)
        tangent_before = relative_before - before_normal.unsqueeze(-1) * normal
        tangential_speed = torch.linalg.vector_norm(tangent_before, dim=-1)
        friction_accepted = (
            collision_accepted
            & (tangential_speed > 1.0e-3)
            & (normal_impulse > 1.0e-8)
            & torch.isfinite(friction)
        )
        belief = self._update_scalar(
            belief,
            slots=slots,
            estimate=friction,
            accepted=friction_accepted,
            parameter="friction",
        )
        return belief, (
            RigidParameterUpdate(
                "restitution",
                evidence.object_id.clone(),
                restitution,
                collision_accepted,
            ),
            RigidParameterUpdate(
                "friction",
                evidence.object_id.clone(),
                friction,
                friction_accepted,
            ),
        )

    def update_boundary_collision_positions(
        self,
        belief: WorldBelief,
        evidence: BoundaryCollisionPositionEvidence,
        *,
        maximum_fit_residual_m: float = 0.01,
    ) -> tuple[WorldBelief, tuple[RigidParameterUpdate, RigidParameterUpdate]]:
        """Estimate boundary response directly from public position traces."""

        slots = self._slots(evidence.object_id, belief)
        for name, positions, timestamps in (
            ("pre-collision", evidence.positions_before_world, evidence.timestamps_before),
            ("post-collision", evidence.positions_after_world, evidence.timestamps_after),
        ):
            self._validate_position_trace(
                name,
                positions,
                timestamps,
                belief,
                minimum_samples=3,
            )
        collision_timestamp = evidence.collision_timestamp
        if (
            collision_timestamp.shape != (belief.batch_size,)
            or collision_timestamp.device != belief.device
            or collision_timestamp.dtype != belief.dtype
            or not torch.isfinite(collision_timestamp).all()
        ):
            raise ValueError("collision_timestamp must be finite [B] on the belief device")
        if torch.any(evidence.timestamps_before[:, -1] > collision_timestamp) or torch.any(
            evidence.timestamps_after[:, 0] < collision_timestamp
        ):
            raise ValueError("boundary traces must bracket the collision timestamp")
        self._validate_vector("normal_world", evidence.normal_world, belief)
        self._validate_fit_residual(maximum_fit_residual_m)
        batch_index = torch.arange(belief.batch_size, device=belief.device)
        drag = belief.objects.drag[batch_index, slots, 0]
        before, before_residual = self._fit_velocity_at_timestamp(
            evidence.positions_before_world,
            evidence.timestamps_before,
            collision_timestamp,
            drag,
        )
        after, after_residual = self._fit_velocity_at_timestamp(
            evidence.positions_after_world,
            evidence.timestamps_after,
            collision_timestamp,
            drag,
        )
        fit_accepted = (before_residual <= maximum_fit_residual_m) & (
            after_residual <= maximum_fit_residual_m
        )
        safe_after = torch.where(fit_accepted.unsqueeze(-1), after, before)
        return self.update_boundary_collision(
            belief,
            BoundaryCollisionEvidence(
                evidence.object_id,
                evidence.normal_world,
                before,
                safe_after,
            ),
        )


def _repeat_belief_rows(belief: WorldBelief, count: int) -> WorldBelief:
    """Repeat a belief by row for paired per-object scalar updates."""

    from dataclasses import fields

    objects = belief.objects.replace(
        **{
            item.name: torch.cat([getattr(belief.objects, item.name)] * count, dim=0)
            for item in fields(belief.objects)
        }
    )
    camera = belief.camera.replace(
        **{
            item.name: torch.cat([getattr(belief.camera, item.name)] * count, dim=0)
            for item in fields(belief.camera)
        }
    )
    return belief.replace(
        timestamp=torch.cat([belief.timestamp] * count),
        objects=objects,
        camera=camera,
        gravity=torch.cat([belief.gravity] * count),
        global_code=torch.cat([belief.global_code] * count),
        global_log_variance=torch.cat([belief.global_log_variance] * count),
        next_object_id=torch.cat([belief.next_object_id] * count),
    )


def _merge_repeated_belief(
    repeated: WorldBelief,
    original: WorldBelief,
    *,
    parameter: str,
) -> WorldBelief:
    """Merge first/second-object updates back into each original batch row."""

    batch = original.batch_size
    objects = original.objects.clone()
    field_name = {
        "restitution": "restitution_logit",
        "friction": "friction_logit",
    }[parameter]
    source_field = getattr(repeated.objects, field_name)
    target_field = getattr(objects, field_name)
    first_changed = source_field[:batch] != getattr(original.objects, field_name)
    second_changed = source_field[batch:] != getattr(original.objects, field_name)
    target_field.copy_(torch.where(first_changed, source_field[:batch], target_field))
    target_field.copy_(torch.where(second_changed, source_field[batch:], target_field))
    source_variance = repeated.objects.slow_log_variance
    original_variance = original.objects.slow_log_variance
    objects.slow_log_variance.copy_(
        torch.minimum(
            torch.minimum(source_variance[:batch], source_variance[batch:]),
            original_variance,
        )
    )
    return replace(original, objects=objects).validate()


__all__ = [
    "BoundaryCollisionEvidence",
    "BoundaryCollisionPositionEvidence",
    "FreeMotionEvidence",
    "FreeMotionPositionEvidence",
    "KnownImpulseEvidence",
    "KnownImpulsePositionEvidence",
    "OnlineRigidParameterEstimator",
    "PairCollisionEvidence",
    "RigidParameterUpdate",
]
