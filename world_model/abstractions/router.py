"""Deterministic first abstraction router over the current world belief."""

from __future__ import annotations

import torch

from world_model.abstractions.contracts import (
    AbstractionAssignment,
    AbstractionKind,
    AbstractionReason,
)
from world_model.abstractions.registry import (
    AbstractionRegistry,
    default_abstraction_registry,
)
from world_model.belief import MotionMode, WorldBelief


class PredictiveAbstractionRouter:
    """Choose the cheapest implemented model family needed by each entity.

    Free-moving spheres can be executed as point trajectories while their
    radius and physical parameters remain available in ``WorldBelief``.
    Contact-like modes refine execution to the rigid-sphere operator.  The
    router is deliberately deterministic until prediction-evidence-based model
    selection is trained and validated.
    """

    _CONTACT_MODES = (
        MotionMode.GROUND_CONTACT,
        MotionMode.PAIR_CONTACT,
        MotionMode.COLLISION,
        MotionMode.ROLLING,
        MotionMode.SLIDING,
    )

    def __init__(self, registry: AbstractionRegistry | None = None) -> None:
        self.registry = registry or default_abstraction_registry()
        self.registry.resolve(AbstractionKind.POINT_TRAJECTORY)
        self.registry.resolve(AbstractionKind.RIGID_SPHERE)

    def route(self, belief: WorldBelief) -> AbstractionAssignment:
        belief.validate()
        objects = belief.objects
        mode = objects.mode
        rigid_mask = torch.zeros_like(objects.active)
        for contact_mode in self._CONTACT_MODES:
            rigid_mask |= mode == int(contact_mode)
        rigid_mask &= objects.active

        point_spec = self.registry.resolve(AbstractionKind.POINT_TRAJECTORY)
        rigid_spec = self.registry.resolve(AbstractionKind.RIGID_SPHERE)
        kind = torch.full_like(
            objects.object_id,
            int(AbstractionKind.POINT_TRAJECTORY),
        )
        kind[rigid_mask] = int(AbstractionKind.RIGID_SPHERE)
        reason = torch.full_like(
            objects.object_id,
            int(AbstractionReason.FREE_MOTION),
        )
        reason[rigid_mask] = int(AbstractionReason.CONTACT_OR_EVENT)
        confidence = objects.position.new_zeros(objects.active.shape)
        confidence[objects.active & ~rigid_mask] = 0.85
        confidence[rigid_mask] = 0.95
        complexity = objects.position.new_zeros(objects.active.shape)
        complexity[objects.active & ~rigid_mask] = point_spec.complexity_cost
        complexity[rigid_mask] = rigid_spec.complexity_cost
        return AbstractionAssignment(
            kind=kind,
            confidence=confidence,
            complexity_cost=complexity,
            reason=reason,
            active_mask=objects.active.clone(),
        ).validate()
