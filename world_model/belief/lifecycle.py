"""Functional object lifecycle transitions for padded belief slots."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from world_model.belief.object_belief import MotionMode
from world_model.belief.packing import slow_packing_map
from world_model.belief.tentative import TentativeBirthState
from world_model.belief.world_belief import WorldBelief

if TYPE_CHECKING:
    from world_model.observations.measurements import MeasurementSet


@dataclass(frozen=True)
class LifecycleConfig:
    max_missed_steps: int = 15
    missed_existence_delta: float = -0.25
    observed_existence_delta: float = 0.5
    missed_visibility_delta: float = -0.75
    observed_visibility_delta: float = 0.75
    removal_existence_logit: float = -6.0
    # Kept last so existing positional construction remains compatible.
    occluded_existence_delta: float = -0.04
    initial_radius: float = 0.1
    initial_mass: float = 1.0
    initial_restitution: float = 0.7
    initial_drag: float = 0.05
    initial_friction: float = 0.2
    initial_log_variance: float = 0.0
    max_occluded_steps: int = 60
    birth_confirmations: int = 1
    birth_confirmation_distance_m: float = 0.5
    minimum_log_variance: float = -12.0
    maximum_log_variance: float = 8.0


@dataclass(frozen=True)
class BirthAssignments:
    """Explicit correspondence for permanent IDs allocated on one birth pass.

    Every field is a flat ``torch.int64`` tensor with one entry per successful
    allocation.  Keeping the discrete lifecycle correspondence explicit lets
    an observation module move its current raw measurement history from a
    proposal index to the new persistent belief slot without treating the
    tentative-birth buffer as differentiable physical state.
    """

    batch_indices: Tensor
    measurement_indices: Tensor
    belief_indices: Tensor
    object_ids: Tensor

    def validate(self) -> BirthAssignments:
        fields = (
            ("batch_indices", self.batch_indices),
            ("measurement_indices", self.measurement_indices),
            ("belief_indices", self.belief_indices),
            ("object_ids", self.object_ids),
        )
        expected_shape = self.batch_indices.shape
        expected_device = self.batch_indices.device
        for name, value in fields:
            if value.ndim != 1 or value.shape != expected_shape:
                raise ValueError("birth-assignment fields must share flat shape [K]")
            if value.dtype is not torch.int64:
                raise TypeError(f"{name} must use torch.int64")
            if value.device != expected_device:
                raise ValueError("birth-assignment fields must share one device")
            if torch.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
        return self

    @property
    def count(self) -> int:
        return self.batch_indices.numel()


class ObjectLifecycle:
    """Apply cheap seen/missed/birth/death transitions without neural updates."""

    def __init__(self, config: LifecycleConfig | None = None) -> None:
        self.config = config or LifecycleConfig()
        if self.config.max_missed_steps < 1:
            raise ValueError("max_missed_steps must be positive")
        if self.config.max_occluded_steps < self.config.max_missed_steps:
            raise ValueError("max_occluded_steps must be no smaller than max_missed_steps")
        if not (self.config.missed_existence_delta <= self.config.occluded_existence_delta <= 0.0):
            raise ValueError(
                "occluded_existence_delta must be nonpositive and no stronger "
                "than missed_existence_delta"
            )
        for name, value in (
            ("initial_radius", self.config.initial_radius),
            ("initial_mass", self.config.initial_mass),
            ("initial_drag", self.config.initial_drag),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("initial_restitution", self.config.initial_restitution),
            ("initial_friction", self.config.initial_friction),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 < value < 1.0
            ):
                raise ValueError(f"{name} must lie strictly in (0,1)")
        if (
            isinstance(self.config.initial_log_variance, bool)
            or not isinstance(self.config.initial_log_variance, (int, float))
            or not math.isfinite(self.config.initial_log_variance)
        ):
            raise ValueError("initial log variance must be finite")
        if (
            isinstance(self.config.minimum_log_variance, bool)
            or isinstance(self.config.maximum_log_variance, bool)
            or not isinstance(self.config.minimum_log_variance, (int, float))
            or not isinstance(self.config.maximum_log_variance, (int, float))
            or not math.isfinite(self.config.minimum_log_variance)
            or not math.isfinite(self.config.maximum_log_variance)
            or self.config.minimum_log_variance > self.config.maximum_log_variance
        ):
            raise ValueError("lifecycle log-variance bounds must be finite and ordered")
        if not (
            self.config.minimum_log_variance
            <= self.config.initial_log_variance
            <= self.config.maximum_log_variance
        ):
            raise ValueError("initial log variance must lie within lifecycle bounds")
        if (
            isinstance(self.config.birth_confirmations, bool)
            or not isinstance(self.config.birth_confirmations, int)
            or self.config.birth_confirmations < 1
        ):
            raise ValueError("birth_confirmations must be a positive integer")
        if (
            not math.isfinite(self.config.birth_confirmation_distance_m)
            or self.config.birth_confirmation_distance_m <= 0.0
        ):
            raise ValueError("birth_confirmation_distance_m must be finite and positive")

    def confirm_tentative_births(
        self,
        measurements: MeasurementSet,
        unmatched_measurements: Tensor,
        previous: TentativeBirthState | None,
        *,
        confidence_threshold: float,
    ) -> tuple[Tensor, TentativeBirthState]:
        """Require consistent unmatched detections before permanent ID birth.

        Matching occurs only in world coordinates and only across strictly
        increasing observation timestamps. The previous tentative set is
        replaced on every discovery observation, so an unobserved candidate
        cannot accumulate non-consecutive confirmations. Tensors are detached:
        this bounded history selects lifecycle evidence but is not a second
        differentiable physical state.
        """

        batch, proposals = measurements.measurement_mask.shape
        if unmatched_measurements.shape != (batch, proposals):
            raise ValueError("unmatched_measurements must have shape [B,M]")
        if unmatched_measurements.dtype is not torch.bool:
            raise TypeError("unmatched_measurements must be torch.bool")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0,1]")
        world_position = measurements.auxiliary.get("world_position")
        if not isinstance(world_position, Tensor) or world_position.shape != (
            batch,
            proposals,
            3,
        ):
            raise ValueError("birth measurements require world_position [B,M,3]")
        if measurements.timestamp.shape != (batch,):
            raise ValueError("measurement timestamp must have shape [B]")

        detached_position = world_position.detach()
        finite_position = torch.isfinite(detached_position).all(dim=-1)
        confidence = measurements.existence_logits.detach().sigmoid()
        eligible = (
            unmatched_measurements
            & measurements.measurement_mask
            & finite_position
            & (confidence >= confidence_threshold)
        )
        counts = torch.zeros(
            (batch, proposals),
            device=world_position.device,
            dtype=torch.int64,
        )
        if self.config.birth_confirmations == 1:
            counts[eligible] = 1
        else:
            if previous is not None:
                previous.validate()
                if previous.world_position.shape[0] != batch:
                    raise ValueError("tentative birth batch size changed without reset")
                previous_position = previous.world_position.to(
                    device=world_position.device,
                    dtype=world_position.dtype,
                )
                previous_active = previous.active.to(device=world_position.device)
                previous_count = previous.confirmation_count.to(device=world_position.device)
                previous_timestamp = previous.timestamp.to(
                    device=world_position.device,
                    dtype=world_position.dtype,
                )
                for batch_index in range(batch):
                    old_indices = torch.nonzero(
                        previous_active[batch_index]
                        & (measurements.timestamp[batch_index] > previous_timestamp[batch_index]),
                        as_tuple=False,
                    ).flatten()
                    new_indices = torch.nonzero(
                        eligible[batch_index],
                        as_tuple=False,
                    ).flatten()
                    if old_indices.numel() == 0 or new_indices.numel() == 0:
                        continue
                    distances = torch.cdist(
                        previous_position[batch_index, old_indices],
                        detached_position[batch_index, new_indices],
                    )
                    admissible = torch.isfinite(distances) & (
                        distances <= self.config.birth_confirmation_distance_m
                    )
                    maximum_assignment_count = min(distances.shape)
                    invalid_cost = (
                        maximum_assignment_count + 1
                    ) * self.config.birth_confirmation_distance_m + 1.0
                    assignment_cost = torch.where(
                        admissible,
                        distances,
                        torch.full_like(distances, invalid_cost),
                    )
                    rows, columns = linear_sum_assignment(
                        np.asarray(
                            assignment_cost.detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            )
                        )
                    )
                    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                        if not bool(admissible[row, column]):
                            continue
                        current_index = int(new_indices[column])
                        prior_index = int(old_indices[row])
                        counts[batch_index, current_index] = (
                            previous_count[batch_index, prior_index] + 1
                        )
            counts = torch.where(
                eligible & (counts == 0),
                torch.ones_like(counts),
                counts,
            )

        confirmed = eligible & (counts >= self.config.birth_confirmations)
        active = eligible & ~confirmed
        safe_position = torch.where(
            finite_position.unsqueeze(-1),
            detached_position,
            torch.zeros_like(detached_position),
        )
        timestamp = (
            measurements.timestamp.detach()
            .to(device=world_position.device, dtype=world_position.dtype)
            .unsqueeze(-1)
            .expand(batch, proposals)
            .clone()
        )
        state = TentativeBirthState(
            world_position=safe_position,
            active=active,
            confirmation_count=torch.where(
                active,
                counts,
                torch.zeros_like(counts),
            ),
            timestamp=timestamp,
        ).validate()
        return confirmed, state

    def update_visibility(
        self,
        belief: WorldBelief,
        observed_mask: Tensor,
        *,
        occluded_mask: Tensor | None = None,
    ) -> WorldBelief:
        objects = belief.objects
        if observed_mask.shape != objects.active.shape or observed_mask.dtype is not torch.bool:
            raise ValueError("observed_mask must be bool [B,N]")
        if occluded_mask is None:
            occluded_mask = torch.zeros_like(observed_mask)
        if occluded_mask.shape != objects.active.shape or occluded_mask.dtype is not torch.bool:
            raise ValueError("occluded_mask must be bool [B,N]")
        observed = objects.active & observed_mask
        missed = objects.active & ~observed_mask
        occluded_miss = missed & occluded_mask
        visible_miss = missed & ~occluded_mask
        was_occluded = objects.mode == MotionMode.OCCLUDED
        observability_transition = missed & (occluded_miss != was_occluded)
        incremented_missed_steps = objects.missed_steps + missed.to(objects.missed_steps.dtype)
        missed_steps = torch.where(
            observed,
            torch.zeros_like(objects.missed_steps),
            torch.where(
                observability_transition,
                torch.ones_like(objects.missed_steps),
                incremented_missed_steps,
            ),
        )
        age_steps = objects.age_steps + objects.active.to(objects.age_steps.dtype)
        existence = (
            objects.existence_logit
            + observed.to(objects.existence_logit.dtype) * self.config.observed_existence_delta
            + visible_miss.to(objects.existence_logit.dtype) * self.config.missed_existence_delta
            + occluded_miss.to(objects.existence_logit.dtype) * self.config.occluded_existence_delta
        )
        visibility = (
            objects.visibility_logit
            + observed.to(objects.visibility_logit.dtype) * self.config.observed_visibility_delta
            + missed.to(objects.visibility_logit.dtype) * self.config.missed_visibility_delta
        )
        remove = objects.active & (
            (visible_miss & (missed_steps > self.config.max_missed_steps))
            | (occluded_miss & (missed_steps > self.config.max_occluded_steps))
            | (existence < self.config.removal_existence_logit)
        )
        active = objects.active & ~remove
        object_id = torch.where(active, objects.object_id, -torch.ones_like(objects.object_id))
        modes = objects.motion_mode_logits.clone()
        occluded_logits = torch.full_like(modes, -4.0)
        occluded_logits[..., MotionMode.OCCLUDED] = 4.0
        modes = torch.where(occluded_miss.unsqueeze(-1), occluded_logits, modes)
        left_occlusion = was_occluded & ~occluded_miss
        free_logits = torch.full_like(modes, -4.0)
        free_logits[..., MotionMode.FREE] = 4.0
        modes = torch.where(left_occlusion.unsqueeze(-1), free_logits, modes)
        removed_logits = torch.full_like(modes, -4.0)
        removed_logits[..., MotionMode.REMOVED] = 4.0
        modes = torch.where(remove.unsqueeze(-1), removed_logits, modes)
        updated = replace(
            objects,
            object_id=object_id,
            active=active,
            existence_logit=existence,
            visibility_logit=visibility,
            missed_steps=missed_steps,
            age_steps=age_steps,
            motion_mode_logits=modes,
        )
        return replace(belief, objects=updated)

    mark_seen_and_missed = update_visibility

    def birth_from_measurements(
        self,
        belief: WorldBelief,
        measurements: MeasurementSet,
        unmatched_measurements: Tensor,
        *,
        confidence_threshold: float = 0.5,
        initial_velocity_variance: float = 1.0,
    ) -> WorldBelief:
        """Allocate unmatched world-space measurements into inactive slots.

        The observation module must explicitly provide ``auxiliary["world_position"]``.
        Optional ``world_velocity`` and ``world_radius`` fields initialise richer
        state.  Simulator state is never consulted.
        """

        born, _ = self.birth_from_measurements_with_assignments(
            belief,
            measurements,
            unmatched_measurements,
            confidence_threshold=confidence_threshold,
            initial_velocity_variance=initial_velocity_variance,
        )
        return born

    def birth_from_measurements_with_assignments(
        self,
        belief: WorldBelief,
        measurements: MeasurementSet,
        unmatched_measurements: Tensor,
        *,
        confidence_threshold: float = 0.5,
        initial_velocity_variance: float = 1.0,
    ) -> tuple[WorldBelief, BirthAssignments]:
        """Allocate births and return their measurement-to-ID correspondence.

        This is an opt-in extension of :meth:`birth_from_measurements`; the
        legacy method still returns only the updated belief.  Assignment
        tensors are discrete metadata, while the born state continues to copy
        differentiable measurement tensors exactly as before.
        """

        objects = belief.objects
        batch, proposals = measurements.measurement_mask.shape
        if batch != belief.batch_size:
            raise ValueError("measurement and belief batch sizes differ")
        if unmatched_measurements.shape != (batch, proposals):
            raise ValueError("unmatched_measurements must have shape [B,M]")
        if unmatched_measurements.dtype is not torch.bool:
            raise TypeError("unmatched_measurements must be torch.bool")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0,1]")
        if initial_velocity_variance <= 0:
            raise ValueError("initial_velocity_variance must be positive")
        if "world_position" not in measurements.auxiliary:
            raise ValueError("birth measurements require auxiliary['world_position'] [B,M,3]")
        world_position = measurements.auxiliary["world_position"]
        if world_position.shape != (batch, proposals, 3):
            raise ValueError("world_position must have shape [B,M,3]")
        world_velocity = measurements.auxiliary.get(
            "world_velocity",
            torch.zeros_like(world_position),
        )
        if world_velocity.shape != world_position.shape:
            raise ValueError("world_velocity must have shape [B,M,3]")
        world_radius = measurements.auxiliary.get("world_radius")
        world_radius_was_flat = world_radius is not None and world_radius.shape == (
            batch,
            proposals,
        )
        if world_radius is not None:
            if world_radius_was_flat:
                world_radius = world_radius.unsqueeze(-1)
            if world_radius.shape != (batch, proposals, 1):
                raise ValueError("world_radius must have shape [B,M,1] or [B,M]")
        world_radius_log_variance = measurements.auxiliary.get("world_radius_log_variance")
        world_radius_valid = measurements.auxiliary.get("world_radius_valid_mask")
        radius_is_supported = "radius" in measurements.supported_state_fields
        radius_group_present = (
            world_radius_log_variance is not None or world_radius_valid is not None
        )
        if radius_group_present and not radius_is_supported:
            raise ValueError("complete world radius birth evidence must declare radius support")
        if radius_is_supported and not radius_group_present:
            raise ValueError("radius-supported births require the complete radius group")
        if radius_group_present and world_radius_was_flat:
            raise ValueError("typed world radius birth evidence must have shape [B,M,1]")
        if world_radius is not None and not radius_group_present:
            if not world_radius.is_floating_point() or world_radius.dtype != objects.position.dtype:
                raise TypeError("world_radius must use the belief floating dtype")
            if world_radius.device != objects.position.device:
                raise ValueError("world_radius must use the belief device")
            if not torch.isfinite(world_radius).all():
                raise ValueError("world_radius must be finite")
            if torch.any(measurements.measurement_mask.unsqueeze(-1) & (world_radius <= 0.0)):
                raise ValueError("valid world_radius values must be positive")
        if world_radius_log_variance is not None or world_radius_valid is not None:
            if (
                world_radius is None
                or world_radius_log_variance is None
                or world_radius_valid is None
            ):
                raise ValueError("world radius birth evidence must be provided as one group")
            if world_radius_log_variance.shape != (batch, proposals, 1):
                raise ValueError("world_radius_log_variance must have shape [B,M,1]")
            if world_radius_valid.shape != (batch, proposals):
                raise ValueError("world_radius_valid_mask must have shape [B,M]")
            if world_radius_valid.dtype is not torch.bool:
                raise TypeError("world_radius_valid_mask must use torch.bool")
            if (
                not world_radius.is_floating_point()
                or not world_radius_log_variance.is_floating_point()
                or world_radius.dtype != objects.position.dtype
                or world_radius_log_variance.dtype != objects.position.dtype
            ):
                raise TypeError("world radius birth evidence must use the belief floating dtype")
            if (
                world_radius.device != objects.position.device
                or world_radius_log_variance.device != objects.position.device
                or world_radius_valid.device != objects.position.device
            ):
                raise ValueError("world radius birth evidence must use the belief device")
            if not torch.isfinite(world_radius).all():
                raise ValueError("world_radius must be finite")
            if torch.any(world_radius_valid.unsqueeze(-1) & (world_radius <= 0.0)):
                raise ValueError("valid world_radius values must be positive")
            if not torch.isfinite(world_radius_log_variance).all():
                raise ValueError("world_radius_log_variance must be finite")
        world_log_variance = measurements.auxiliary.get("world_log_variance")
        if world_log_variance is not None and (
            world_log_variance.shape[:2] != (batch, proposals) or world_log_variance.shape[-1] < 3
        ):
            raise ValueError("world_log_variance must begin [B,M] and end with >=3")
        if measurements.appearance is not None and (
            measurements.appearance.shape[-1] != objects.appearance_dim
        ):
            raise ValueError("measurement appearance dimension differs from belief")

        confidence = measurements.existence_logits.sigmoid()
        candidates = (
            unmatched_measurements
            & measurements.measurement_mask
            & (confidence >= confidence_threshold)
        )
        if radius_is_supported:
            if world_radius_valid is None:
                raise ValueError("radius-supported births require radius validity")
            candidates &= world_radius_valid
        updated = objects.clone()
        next_id = belief.next_object_id.clone()
        velocity_log_variance = math.log(initial_velocity_variance)
        birth_batch_indices: list[int] = []
        birth_measurement_indices: list[int] = []
        birth_belief_indices: list[int] = []
        birth_object_ids: list[Tensor] = []
        for batch_index in range(batch):
            slots = torch.nonzero(
                ~updated.active[batch_index],
                as_tuple=False,
            ).flatten()
            candidates_b = torch.nonzero(
                candidates[batch_index],
                as_tuple=False,
            ).flatten()
            if candidates_b.numel() > 1:
                # Slot capacity is deliberately bounded.  When discovery
                # produces more qualified unmatched measurements than free
                # slots, retain the strongest evidence rather than whichever
                # proposal happened to have the lowest query index.  Stable
                # sorting keeps equal-confidence allocation deterministic.
                candidate_order = torch.argsort(
                    confidence[batch_index, candidates_b],
                    descending=True,
                    stable=True,
                )
                candidates_b = candidates_b[candidate_order]
            for slot, measurement_index in zip(
                slots.tolist(),
                candidates_b.tolist(),
                strict=False,
            ):
                # An inactive slot may have belonged to an entirely different
                # object. Reset every identity-specific fast, slow, learned, and
                # uncertainty field before applying the new measurement so no
                # physical parameter or recurrent memory crosses identities.
                updated.position[batch_index, slot].zero_()
                updated.velocity[batch_index, slot].zero_()
                updated.orientation[batch_index, slot].zero_()
                updated.orientation[batch_index, slot, 3] = 1.0
                updated.angular_velocity[batch_index, slot].zero_()
                updated.geometry[batch_index, slot].zero_()
                updated.geometry[batch_index, slot, 0] = self.config.initial_radius
                updated.appearance[batch_index, slot].zero_()
                updated.residual_dynamics[batch_index, slot].zero_()
                updated.modal_state[batch_index, slot].zero_()
                updated.modal_frequency[batch_index, slot].zero_()
                updated.modal_decay_raw[batch_index, slot].zero_()
                updated.log_mass[batch_index, slot].fill_(math.log(self.config.initial_mass))
                updated.restitution_logit[batch_index, slot].fill_(
                    math.log(
                        self.config.initial_restitution / (1.0 - self.config.initial_restitution)
                    )
                )
                updated.log_drag[batch_index, slot].fill_(math.log(self.config.initial_drag))
                updated.friction_logit[batch_index, slot].fill_(
                    math.log(self.config.initial_friction / (1.0 - self.config.initial_friction))
                )
                updated.motion_mode_logits[batch_index, slot].fill_(-4.0)
                updated.visibility_logit[batch_index, slot] = 0.0
                updated.age_steps[batch_index, slot] = 0
                updated.missed_steps[batch_index, slot] = 0
                updated.fast_log_variance[batch_index, slot].fill_(self.config.initial_log_variance)
                updated.slow_log_variance[batch_index, slot].fill_(self.config.initial_log_variance)
                updated.parameter_memory[batch_index, slot].zero_()

                updated.active[batch_index, slot] = True
                assigned_object_id = next_id[batch_index].clone()
                updated.object_id[batch_index, slot] = assigned_object_id
                next_id[batch_index] += 1
                birth_batch_indices.append(batch_index)
                birth_measurement_indices.append(measurement_index)
                birth_belief_indices.append(slot)
                birth_object_ids.append(assigned_object_id)
                updated.existence_logit[batch_index, slot] = measurements.existence_logits[
                    batch_index, measurement_index
                ]
                updated.visibility_logit[batch_index, slot] = measurements.auxiliary.get(
                    "visibility_logit",
                    measurements.existence_logits,
                )[batch_index, measurement_index]
                updated.position[batch_index, slot] = world_position[batch_index, measurement_index]
                updated.velocity[batch_index, slot] = world_velocity[batch_index, measurement_index]
                if world_radius is not None:
                    radius_is_valid = world_radius_valid is None or bool(
                        world_radius_valid[batch_index, measurement_index]
                    )
                    if radius_is_valid:
                        updated.geometry[batch_index, slot, :1] = world_radius[
                            batch_index, measurement_index
                        ].clamp_min(1e-6)
                        if world_radius_log_variance is not None:
                            geometry_slice = slow_packing_map(updated)["geometry"]
                            radius_variance_index = geometry_slice.start
                            updated.slow_log_variance[
                                batch_index,
                                slot,
                                radius_variance_index : radius_variance_index + 1,
                            ] = world_radius_log_variance[
                                batch_index,
                                measurement_index,
                            ].clamp(
                                self.config.minimum_log_variance,
                                self.config.maximum_log_variance,
                            )
                if measurements.appearance is not None:
                    appearance = measurements.appearance[batch_index, measurement_index]
                    updated.appearance[batch_index, slot] = appearance / torch.linalg.vector_norm(
                        appearance
                    ).clamp_min(1e-8)
                updated.motion_mode_logits[batch_index, slot, MotionMode.CREATED] = 4.0
                if world_log_variance is not None:
                    updated.fast_log_variance[batch_index, slot, :3] = world_log_variance[
                        batch_index, measurement_index, :3
                    ]
                updated.fast_log_variance[batch_index, slot, 3:6] = velocity_log_variance
        metadata = belief.metadata.copy()
        metadata["initialised"] = bool(updated.active.any())
        born = replace(
            belief,
            objects=updated,
            next_object_id=next_id,
            metadata=metadata,
        )
        assignment_device = objects.object_id.device
        assignments = BirthAssignments(
            batch_indices=torch.tensor(
                birth_batch_indices,
                device=assignment_device,
                dtype=torch.int64,
            ),
            measurement_indices=torch.tensor(
                birth_measurement_indices,
                device=assignment_device,
                dtype=torch.int64,
            ),
            belief_indices=torch.tensor(
                birth_belief_indices,
                device=assignment_device,
                dtype=torch.int64,
            ),
            object_ids=(
                torch.stack(birth_object_ids)
                if birth_object_ids
                else objects.object_id.new_empty((0,))
            ),
        ).validate()
        return born, assignments


def birth_from_measurements(
    belief: WorldBelief,
    measurements: MeasurementSet,
    unmatched_measurements: Tensor,
    *,
    confidence_threshold: float = 0.5,
    initial_velocity_variance: float = 1.0,
) -> WorldBelief:
    """Functional convenience wrapper around :class:`ObjectLifecycle`."""

    return ObjectLifecycle().birth_from_measurements(
        belief,
        measurements,
        unmatched_measurements,
        confidence_threshold=confidence_threshold,
        initial_velocity_variance=initial_velocity_variance,
    )
