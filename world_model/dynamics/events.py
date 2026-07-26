"""Explicit event probabilities and structured state jumps."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from world_model.belief import NUM_MOTION_MODES, MotionMode, ObjectBeliefTensor
from world_model.dynamics.contacts import ContactResult, SphereContactResolver
from world_model.dynamics.graph import InteractionOutput


@dataclass
class EventOutput:
    objects: ObjectBeliefTensor
    event_logits: Tensor
    pair_event_logits: Tensor
    contacts: ContactResult


class EventModel(nn.Module):
    """Hybrid categorical mode model backed by analytic sphere contact jumps."""

    def __init__(
        self,
        resolver: SphereContactResolver | None = None,
        *,
        contact_logit_scale: float = 0.02,
        sleep_speed_threshold: float = 0.02,
    ) -> None:
        super().__init__()
        self.resolver = resolver or SphereContactResolver()
        self.contact_logit_scale = contact_logit_scale
        self.sleep_speed_threshold = sleep_speed_threshold

    def forward(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None = None,
    ) -> EventOutput:
        contacts = self.resolver(objects, graph)
        updated = contacts.objects
        dtype = updated.position.dtype
        batch, count = updated.active.shape
        logits = updated.motion_mode_logits.new_full(
            (batch, count, NUM_MOTION_MODES),
            -4.0,
        )
        logits[..., MotionMode.FREE] = 2.0
        logits[..., MotionMode.OCCLUDED] = -updated.visibility_logit

        ground_score = torch.where(
            contacts.ground_contact,
            torch.full_like(updated.visibility_logit, 4.0),
            torch.full_like(updated.visibility_logit, -4.0),
        )
        pair_contact_node = contacts.pair_contact.any(dim=-1)
        pair_collision_node = contacts.pair_collision.any(dim=-1)
        pair_score = torch.where(
            pair_contact_node,
            torch.full_like(updated.visibility_logit, 4.0),
            torch.full_like(updated.visibility_logit, -4.0),
        )
        collision_score = torch.where(
            pair_collision_node | contacts.ground_collision,
            torch.full_like(updated.visibility_logit, 6.0),
            torch.full_like(updated.visibility_logit, -4.0),
        )
        if graph is not None:
            pair_score = pair_score + graph.contact_logits.max(dim=-1).values
            collision_score = collision_score + graph.collision_logits.max(dim=-1).values
        logits[..., MotionMode.GROUND_CONTACT] = ground_score
        logits[..., MotionMode.PAIR_CONTACT] = pair_score
        logits[..., MotionMode.COLLISION] = collision_score

        speed = torch.linalg.vector_norm(updated.velocity, dim=-1)
        sleeping = (
            contacts.ground_contact & (speed < self.sleep_speed_threshold) & ~pair_collision_node
        )
        logits[..., MotionMode.SLEEPING] = torch.where(
            sleeping,
            torch.full_like(speed, 5.0),
            torch.full_like(speed, -4.0),
        )
        velocity = torch.where(
            sleeping.unsqueeze(-1),
            torch.zeros_like(updated.velocity),
            updated.velocity,
        )
        # Preserve inactive padding values and modes.
        logits = torch.where(
            updated.active.unsqueeze(-1),
            logits,
            objects.motion_mode_logits,
        )
        updated = replace(
            updated,
            velocity=velocity,
            motion_mode_logits=logits,
        )
        pair_logits = torch.stack(
            (
                torch.where(
                    contacts.pair_contact,
                    torch.full_like(contacts.pair_impulse, 4.0),
                    torch.full_like(contacts.pair_impulse, -4.0),
                ),
                torch.where(
                    contacts.pair_collision,
                    torch.full_like(contacts.pair_impulse, 6.0),
                    torch.full_like(contacts.pair_impulse, -4.0),
                ),
            ),
            dim=-1,
        ).to(dtype)
        return EventOutput(
            objects=updated,
            event_logits=logits,
            pair_event_logits=pair_logits,
            contacts=contacts,
        )
