"""Per-parameter observability gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.belief import MotionMode, WorldBelief
from world_model.fusion import AssociationResult, SurpriseAssessment
from world_model.observations import InnovationSet


@dataclass
class Observability:
    mass_ratio: Tensor
    restitution: Tensor
    drag: Tensor
    friction: Tensor
    geometry: Tensor

    def stacked(self) -> Tensor:
        return torch.stack(
            (
                self.mass_ratio,
                self.restitution,
                self.drag,
                self.friction,
                self.geometry,
            ),
            dim=-1,
        )

    def validate(self, expected_shape: tuple[int, int]) -> None:
        for name, value in (
            ("mass_ratio", self.mass_ratio),
            ("restitution", self.restitution),
            ("drag", self.drag),
            ("friction", self.friction),
            ("geometry", self.geometry),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} observability must be [B,N]")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} observability contains NaN or Inf")
            if torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} observability must lie in [0,1]")


@dataclass(frozen=True)
class ObservabilityConfig:
    minimum_drag_speed: float = 0.25
    drag_speed_scale: float = 0.2
    minimum_free_steps: int = 3
    association_cost_scale: float = 5.0


class ObservabilityEstimator:
    """Encode physical identifiability rather than generic confidence."""

    def __init__(self, config: ObservabilityConfig | None = None) -> None:
        self.config = config or ObservabilityConfig()

    def __call__(
        self,
        belief: WorldBelief,
        innovation: InnovationSet,
        association: AssociationResult,
        cause: SurpriseAssessment | None = None,
    ) -> Observability:
        objects = belief.objects
        batch, object_count = objects.active.shape
        device = objects.position.device
        dtype = objects.position.dtype
        zeros = torch.zeros(batch, object_count, device=device, dtype=dtype)
        mass = zeros.clone()
        restitution = zeros.clone()
        drag = zeros.clone()
        friction = zeros.clone()
        geometry = zeros.clone()
        batch_index, pair_index = torch.nonzero(association.pair_mask, as_tuple=True)
        if batch_index.numel() == 0:
            result = Observability(mass, restitution, drag, friction, geometry)
            result.validate((batch, object_count))
            return result
        object_index = association.belief_indices[batch_index, pair_index]
        ambiguity = association.ambiguous[batch_index, pair_index]
        cost = association.pair_cost[batch_index, pair_index]
        association_confidence = torch.exp(-cost / self.config.association_cost_scale) * (
            ~ambiguity
        ).to(dtype)
        visibility = objects.visibility_logit[batch_index, object_index].sigmoid()
        confidence = association_confidence * visibility
        modes = objects.motion_mode_logits[batch_index, object_index].softmax(dim=-1)
        speed = torch.linalg.vector_norm(objects.velocity[batch_index, object_index], dim=-1)
        free = modes[..., MotionMode.FREE]
        ground = modes[..., MotionMode.GROUND_CONTACT]
        pair = modes[..., MotionMode.PAIR_CONTACT]
        collision = modes[..., MotionMode.COLLISION]
        sliding = modes[..., MotionMode.SLIDING]
        if cause is not None:
            # Cause index 2 is PHYSICAL_EVENT by the public enum.
            physical_event = cause.cause_probabilities[batch_index, pair_index, 2]
            collision_evidence = torch.maximum(collision, physical_event)
        else:
            collision_evidence = collision
        interaction = torch.maximum(pair, collision_evidence)
        free_history = (
            objects.age_steps[batch_index, object_index] >= self.config.minimum_free_steps
        ).to(dtype)
        speed_gate = torch.sigmoid(
            (speed - self.config.minimum_drag_speed) / self.config.drag_speed_scale
        )
        values = (
            confidence * interaction,
            confidence * collision_evidence,
            confidence * free * free_history * speed_gate,
            confidence * torch.maximum(ground, sliding) * speed_gate,
            confidence,
        )
        targets = (mass, restitution, drag, friction, geometry)
        for target, value in zip(targets, values, strict=True):
            # A slot has at most one assignment, so direct scatter is safe.
            target[batch_index, object_index] = value.clamp(0.0, 1.0)
        result = Observability(*targets)
        result.validate((batch, object_count))
        return result
