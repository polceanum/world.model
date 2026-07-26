"""Functional object lifecycle transitions for padded belief slots."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from world_model.belief.object_belief import MotionMode
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
    missed_log_variance_growth: float = 0.08
    maximum_fast_log_variance: float = 8.0


class ObjectLifecycle:
    """Apply cheap seen/missed/birth/death transitions without neural updates."""

    def __init__(self, config: LifecycleConfig | None = None) -> None:
        self.config = config or LifecycleConfig()
        if self.config.max_missed_steps < 1:
            raise ValueError("max_missed_steps must be positive")
        if not (self.config.missed_existence_delta <= self.config.occluded_existence_delta <= 0.0):
            raise ValueError(
                "occluded_existence_delta must be nonpositive and no stronger "
                "than missed_existence_delta"
            )
        if self.config.missed_log_variance_growth < 0:
            raise ValueError("missed_log_variance_growth must be nonnegative")

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
        missed_steps = torch.where(
            observed,
            torch.zeros_like(objects.missed_steps),
            objects.missed_steps + missed.to(objects.missed_steps.dtype),
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
        fast_log_variance = (
            objects.fast_log_variance
            + missed.unsqueeze(-1).to(objects.fast_log_variance.dtype)
            * self.config.missed_log_variance_growth
        ).clamp_max(self.config.maximum_fast_log_variance)
        remove = objects.active & (
            (missed_steps > self.config.max_missed_steps)
            | (existence < self.config.removal_existence_logit)
        )
        active = objects.active & ~remove
        object_id = torch.where(active, objects.object_id, -torch.ones_like(objects.object_id))
        modes = objects.motion_mode_logits.clone()
        occluded_logits = torch.full_like(modes, -4.0)
        occluded_logits[..., MotionMode.OCCLUDED] = 4.0
        modes = torch.where(occluded_miss.unsqueeze(-1), occluded_logits, modes)
        was_occluded = objects.mode == MotionMode.OCCLUDED
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
            fast_log_variance=fast_log_variance,
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
        if world_radius is not None:
            if world_radius.shape == (batch, proposals):
                world_radius = world_radius.unsqueeze(-1)
            if world_radius.shape != (batch, proposals, 1):
                raise ValueError("world_radius must have shape [B,M,1] or [B,M]")
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
        updated = objects.clone()
        next_id = belief.next_object_id.clone()
        velocity_log_variance = math.log(initial_velocity_variance)
        for batch_index in range(batch):
            slots = torch.nonzero(
                ~updated.active[batch_index],
                as_tuple=False,
            ).flatten()
            candidates_b = torch.nonzero(
                candidates[batch_index],
                as_tuple=False,
            ).flatten()
            for slot, measurement_index in zip(
                slots.tolist(),
                candidates_b.tolist(),
                strict=False,
            ):
                updated.active[batch_index, slot] = True
                updated.object_id[batch_index, slot] = next_id[batch_index]
                next_id[batch_index] += 1
                updated.existence_logit[batch_index, slot] = measurements.existence_logits[
                    batch_index, measurement_index
                ]
                updated.visibility_logit[batch_index, slot] = measurements.auxiliary.get(
                    "visibility_logit",
                    measurements.existence_logits,
                )[batch_index, measurement_index]
                updated.position[batch_index, slot] = world_position[batch_index, measurement_index]
                updated.velocity[batch_index, slot] = world_velocity[batch_index, measurement_index]
                updated.orientation[batch_index, slot].zero_()
                updated.orientation[batch_index, slot, 3] = 1.0
                updated.angular_velocity[batch_index, slot].zero_()
                if world_radius is not None:
                    updated.geometry[batch_index, slot, :1] = world_radius[
                        batch_index, measurement_index
                    ].clamp_min(1e-6)
                if measurements.appearance is not None:
                    appearance = measurements.appearance[batch_index, measurement_index]
                    updated.appearance[batch_index, slot] = appearance / torch.linalg.vector_norm(
                        appearance
                    ).clamp_min(1e-8)
                updated.modal_state[batch_index, slot].zero_()
                updated.motion_mode_logits[batch_index, slot].fill_(-4.0)
                updated.motion_mode_logits[batch_index, slot, MotionMode.CREATED] = 4.0
                updated.age_steps[batch_index, slot] = 0
                updated.missed_steps[batch_index, slot] = 0
                if world_log_variance is not None:
                    updated.fast_log_variance[batch_index, slot, :3] = world_log_variance[
                        batch_index, measurement_index, :3
                    ]
                updated.fast_log_variance[batch_index, slot, 3:6] = velocity_log_variance
        metadata = belief.metadata.copy()
        metadata["initialised"] = bool(updated.active.any())
        return replace(
            belief,
            objects=updated,
            next_object_id=next_id,
            metadata=metadata,
        )


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
