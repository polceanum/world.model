"""Explicit event probabilities and structured state jumps."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from world_model.belief import NUM_MOTION_MODES, MotionMode, ObjectBeliefTensor
from world_model.dynamics.contacts import ContactResult, SphereContactResolver
from world_model.dynamics.graph import InteractionOutput


@dataclass
class EventOutput:
    objects: ObjectBeliefTensor
    event_logits: Tensor
    pair_event_logits: Tensor
    boundary_event_logits: Tensor
    contacts: ContactResult


def _max_valid_edge_residual(edge_logits: Tensor, edge_mask: Tensor) -> Tensor:
    """Max-pool learned edge logits without treating missing edges as zero."""
    if edge_logits.shape != edge_mask.shape:
        raise ValueError("edge logits and edge mask must have identical shapes")
    if edge_mask.dtype != torch.bool:
        raise ValueError("edge mask must be boolean")
    masked_logits = edge_logits.masked_fill(
        ~edge_mask,
        torch.finfo(edge_logits.dtype).min,
    )
    pooled = masked_logits.max(dim=-1).values
    return torch.where(
        edge_mask.any(dim=-1),
        pooled,
        torch.zeros_like(pooled),
    )


def _max_valid_logits(
    logits: Tensor,
    valid_mask: Tensor,
    *,
    default: float,
) -> Tensor:
    """Max-pool logits while keeping invalid entries out of the reduction."""

    if logits.shape != valid_mask.shape:
        raise ValueError("logits and validity mask must have identical shapes")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("validity mask must be boolean")
    masked_logits = logits.masked_fill(
        ~valid_mask,
        torch.finfo(logits.dtype).min,
    )
    pooled = masked_logits.max(dim=-1).values
    return torch.where(
        valid_mask.any(dim=-1),
        pooled,
        logits.new_full((), default),
    )


def _straight_through_floor(logits: Tensor, mask: Tensor, minimum: float) -> Tensor:
    """Guarantee an analytic event in the forward pass without killing its gradient.

    Hard contact resolution remains the fail-safe source of physical jumps.  Its
    boolean occurrence can therefore impose a positive event-logit floor, while
    the straight-through residual keeps calibration gradients flowing into the
    smooth analytic hazard and learned relation proposal.
    """

    if logits.shape != mask.shape:
        raise ValueError("event logits and resolved-event mask must have identical shapes")
    floor = logits.new_tensor(minimum)
    floored = torch.maximum(logits, floor)
    straight_through = logits + (floored - logits).detach()
    return torch.where(mask, straight_through, logits)


def _safe_projected_standard_deviation(directional_variance: Tensor) -> Tensor:
    """Return a finite standard deviation, including zero self-pair projections.

    Pair geometry contains an explicit diagonal. Its relative direction is the
    zero vector, so the projected variance is exactly zero even though those
    self-pairs are masked later. Backpropagating through ``sqrt(0)`` can form
    ``0 * inf = NaN`` before that mask removes the diagonal. A dtype-scaled
    positive floor leaves every physical off-diagonal projection unchanged at
    ordinary precision while making the complete dense graph differentiable.
    """

    if not directional_variance.is_floating_point():
        raise ValueError("directional variance must be floating point")
    standard_deviation_floor = torch.finfo(directional_variance.dtype).eps
    return directional_variance.clamp_min(standard_deviation_floor**2).sqrt()


class EventModel(nn.Module):
    """Hybrid categorical mode model backed by analytic sphere contact jumps."""

    def __init__(
        self,
        resolver: SphereContactResolver | None = None,
        *,
        smooth_hazard_enabled: bool = False,
        contact_logit_scale: float = 0.02,
        collision_velocity_logit_scale: float = 0.10,
        resolved_event_logit_floor: float = 2.0,
        sleep_speed_threshold: float = 0.02,
    ) -> None:
        super().__init__()
        self.resolver = resolver or SphereContactResolver()
        if not isinstance(smooth_hazard_enabled, bool):
            raise ValueError("smooth_hazard_enabled must be boolean")
        for name, value in (
            ("contact_logit_scale", contact_logit_scale),
            ("collision_velocity_logit_scale", collision_velocity_logit_scale),
            ("resolved_event_logit_floor", resolved_event_logit_floor),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.smooth_hazard_enabled = smooth_hazard_enabled
        self.contact_logit_scale = float(contact_logit_scale)
        self.collision_velocity_logit_scale = float(collision_velocity_logit_scale)
        self.resolved_event_logit_floor = float(resolved_event_logit_floor)
        self.sleep_speed_threshold = sleep_speed_threshold

    def forward(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None = None,
    ) -> EventOutput:
        if self.smooth_hazard_enabled:
            return self._smooth_hazard_forward(objects, graph)
        return self._legacy_forward(objects, graph)

    def _legacy_forward(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None,
    ) -> EventOutput:
        """Preserve the exact hard-logit behavior of historical checkpoints."""

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
        boundary_collision_node = contacts.boundary_collision.any(dim=-1)
        pair_score = torch.where(
            pair_contact_node,
            torch.full_like(updated.visibility_logit, 4.0),
            torch.full_like(updated.visibility_logit, -4.0),
        )
        collision_score = torch.where(
            pair_collision_node | boundary_collision_node,
            torch.full_like(updated.visibility_logit, 6.0),
            torch.full_like(updated.visibility_logit, -4.0),
        )
        if graph is not None:
            pair_score = pair_score + _max_valid_edge_residual(
                graph.contact_logits,
                graph.edge_mask,
            )
            collision_score = collision_score + _max_valid_edge_residual(
                graph.collision_logits,
                graph.edge_mask,
            )
        logits[..., MotionMode.GROUND_CONTACT] = ground_score
        logits[..., MotionMode.PAIR_CONTACT] = pair_score
        logits[..., MotionMode.COLLISION] = collision_score

        speed = torch.linalg.vector_norm(updated.velocity, dim=-1)
        # The simulator requires sustained floor support before entering
        # sleep. The belief has no simulator-only substep counter, so dynamics
        # may preserve a supported SLEEPING posterior but must not invent one
        # from a single slow contact. Ground-contact constraints still remove
        # inward normal speed while low tangential motion remains observable.
        sleeping = (
            (objects.mode == int(MotionMode.SLEEPING))
            & contacts.ground_contact
            & (speed < self.sleep_speed_threshold)
            & ~pair_collision_node
            & ~boundary_collision_node
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
        endpoint_logits = logits.clone()
        # Collision is an interval event, not a persistent endpoint mode after
        # the impulse has been resolved. ``event_logits`` below retains it for
        # observation-window supervision and online history resets.
        endpoint_logits[..., MotionMode.COLLISION] = torch.where(
            updated.active,
            endpoint_logits.new_full((), -4.0),
            objects.motion_mode_logits[..., MotionMode.COLLISION],
        )
        updated = replace(
            updated,
            velocity=velocity,
            motion_mode_logits=endpoint_logits,
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
            boundary_event_logits=torch.stack(
                (
                    torch.where(
                        contacts.boundary_contact,
                        torch.full_like(contacts.boundary_contact, 4.0, dtype=dtype),
                        torch.full_like(contacts.boundary_contact, -4.0, dtype=dtype),
                    ),
                    torch.where(
                        contacts.boundary_collision,
                        torch.full_like(contacts.boundary_collision, 6.0, dtype=dtype),
                        torch.full_like(contacts.boundary_collision, -4.0, dtype=dtype),
                    ),
                ),
                dim=-1,
            ),
            contacts=contacts,
        )

    def _pair_geometry(self, objects: ObjectBeliefTensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return uncertainty-aware gap, normal speed, and active pair mask."""

        _, count = objects.active.shape
        relative_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]
        distance = torch.linalg.vector_norm(relative_position, dim=-1).clamp_min(1.0e-7)
        normal = relative_position / distance.unsqueeze(-1)
        relative_velocity = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]
        normal_velocity = (relative_velocity * normal).sum(dim=-1)
        radius = objects.radius.squeeze(-1)
        gap = distance - radius[:, :, None] - radius[:, None, :]
        position_variance = objects.fast_log_variance[..., :3].exp()
        relative_variance = position_variance[:, :, None, :] + position_variance[:, None, :, :]
        gap_sigma = _safe_projected_standard_deviation(
            (relative_variance * normal.square()).sum(dim=-1)
        )
        confident_gap = gap + self.resolver.contact_confidence_sigma * gap_sigma
        identity = torch.eye(
            count,
            device=objects.position.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        valid = objects.active[:, :, None] & objects.active[:, None, :] & ~identity
        return confident_gap, normal_velocity, valid

    def _boundary_geometry(self, objects: ObjectBeliefTensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return uncertainty-aware plane gap, normal speed, and validity."""

        normals = self.resolver.plane_normals.to(
            device=objects.position.device,
            dtype=objects.position.dtype,
        )
        offsets = self.resolver.plane_offsets.to(
            device=objects.position.device,
            dtype=objects.position.dtype,
        )
        signed_center = torch.einsum("bnc,pc->bnp", objects.position, normals)
        gap = signed_center - offsets[None, None, :] - objects.radius
        position_variance = objects.fast_log_variance[..., :3].exp()
        gap_sigma = _safe_projected_standard_deviation(
            torch.einsum(
                "bnc,pc->bnp",
                position_variance,
                normals.square(),
            )
        )
        confident_gap = gap + self.resolver.contact_confidence_sigma * gap_sigma
        normal_velocity = torch.einsum("bnc,pc->bnp", objects.velocity, normals)
        valid = objects.active.unsqueeze(-1).expand_as(confident_gap)
        return confident_gap, normal_velocity, valid

    @staticmethod
    def _smooth_conjunction(first: Tensor, second: Tensor) -> Tensor:
        """A finite differentiable logit-space approximation to ``min``/AND."""

        # Algebraically this is ``-logaddexp(-first, -second)``.  Keep the
        # explicit stable form because the custom MPS build's logaddexp kernel
        # overflows for ordinary finite inputs above roughly 88, turning a
        # distant pair's strong negative contact hazard into ``-inf``.
        return torch.minimum(first, second) - F.softplus(-torch.abs(first - second))

    def _learned_pair_residual(
        self,
        value: Tensor,
        graph: InteractionOutput | None,
    ) -> Tensor:
        if graph is None:
            return torch.zeros_like(value)
        if graph.edge_mask.shape != value.shape:
            raise ValueError("interaction graph edge mask has incompatible event shape")
        # ``where`` rather than multiplication prevents an invalid/non-edge NaN
        # from contaminating the analytic hazard. Valid learned edges remain
        # fully differentiable.
        return torch.where(graph.edge_mask, value, torch.zeros_like(value))

    def _smooth_hazard_forward(
        self,
        objects: ObjectBeliefTensor,
        graph: InteractionOutput | None,
    ) -> EventOutput:
        """Emit calibrated smooth hazards while retaining hard analytic jumps."""

        # Collision hazards use the incoming pre-resolution geometry and motion;
        # otherwise an analytic bounce would erase the very approaching evidence
        # that caused the event. Contact modes use the resolved endpoint below.
        incoming_pair_gap, incoming_pair_speed, pair_valid = self._pair_geometry(objects)
        incoming_boundary_gap, incoming_boundary_speed, boundary_valid = self._boundary_geometry(
            objects
        )
        contacts = self.resolver(objects, graph)
        updated = contacts.objects
        endpoint_pair_gap, _, endpoint_pair_valid = self._pair_geometry(updated)
        endpoint_boundary_gap, _, endpoint_boundary_valid = self._boundary_geometry(updated)

        pair_contact = -endpoint_pair_gap / self.contact_logit_scale
        pair_contact = pair_contact + self._learned_pair_residual(
            graph.contact_logits if graph is not None else pair_contact,
            graph,
        )
        pair_approach = (
            -incoming_pair_speed - self.resolver.collision_speed_epsilon
        ) / self.collision_velocity_logit_scale
        pair_collision = self._smooth_conjunction(
            -incoming_pair_gap / self.contact_logit_scale,
            pair_approach,
        )
        pair_collision = pair_collision + self._learned_pair_residual(
            graph.collision_logits if graph is not None else pair_collision,
            graph,
        )
        pair_contact = _straight_through_floor(
            pair_contact,
            contacts.pair_contact,
            self.resolved_event_logit_floor,
        )
        pair_collision = _straight_through_floor(
            pair_collision,
            contacts.pair_collision,
            self.resolved_event_logit_floor,
        )
        pair_contact = torch.where(
            endpoint_pair_valid,
            pair_contact,
            pair_contact.new_full((), -4.0),
        )
        pair_collision = torch.where(
            pair_valid,
            pair_collision,
            pair_collision.new_full((), -4.0),
        )
        pair_logits = torch.stack((pair_contact, pair_collision), dim=-1)

        boundary_contact = -endpoint_boundary_gap / self.contact_logit_scale
        boundary_approach = (
            -incoming_boundary_speed - self.resolver.boundary_collision_speed_epsilon
        ) / self.collision_velocity_logit_scale
        boundary_collision = self._smooth_conjunction(
            -incoming_boundary_gap / self.contact_logit_scale,
            boundary_approach,
        )
        boundary_contact = _straight_through_floor(
            boundary_contact,
            contacts.boundary_contact,
            self.resolved_event_logit_floor,
        )
        boundary_collision = _straight_through_floor(
            boundary_collision,
            contacts.boundary_collision,
            self.resolved_event_logit_floor,
        )
        boundary_contact = torch.where(
            endpoint_boundary_valid,
            boundary_contact,
            boundary_contact.new_full((), -4.0),
        )
        boundary_collision = torch.where(
            boundary_valid,
            boundary_collision,
            boundary_collision.new_full((), -4.0),
        )
        boundary_logits = torch.stack(
            (boundary_contact, boundary_collision),
            dim=-1,
        )

        batch, count = updated.active.shape
        logits = updated.motion_mode_logits.new_full(
            (batch, count, NUM_MOTION_MODES),
            -4.0,
        )
        logits[..., MotionMode.FREE] = 2.0
        logits[..., MotionMode.OCCLUDED] = -updated.visibility_logit
        ground_plane_mask = self.resolver.ground_plane_mask.to(
            device=updated.position.device,
        )
        ground_valid = endpoint_boundary_valid & ground_plane_mask[None, None, :]
        logits[..., MotionMode.GROUND_CONTACT] = _max_valid_logits(
            boundary_contact,
            ground_valid,
            default=-4.0,
        )
        logits[..., MotionMode.PAIR_CONTACT] = _max_valid_logits(
            pair_contact,
            endpoint_pair_valid,
            default=-4.0,
        )
        pair_collision_node = _max_valid_logits(
            pair_collision,
            pair_valid,
            default=torch.finfo(pair_collision.dtype).min,
        )
        boundary_collision_node = _max_valid_logits(
            boundary_collision,
            boundary_valid,
            default=torch.finfo(boundary_collision.dtype).min,
        )
        logits[..., MotionMode.COLLISION] = torch.maximum(
            pair_collision_node,
            boundary_collision_node,
        )

        pair_collision_resolved = contacts.pair_collision.any(dim=-1)
        boundary_collision_resolved = contacts.boundary_collision.any(dim=-1)
        speed = torch.linalg.vector_norm(updated.velocity, dim=-1)
        sleeping = (
            (objects.mode == int(MotionMode.SLEEPING))
            & contacts.ground_contact
            & (speed < self.sleep_speed_threshold)
            & ~pair_collision_resolved
            & ~boundary_collision_resolved
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
        logits = torch.where(
            updated.active.unsqueeze(-1),
            logits,
            objects.motion_mode_logits,
        )
        endpoint_logits = logits.clone()
        endpoint_logits[..., MotionMode.COLLISION] = torch.where(
            updated.active,
            endpoint_logits.new_full((), -4.0),
            objects.motion_mode_logits[..., MotionMode.COLLISION],
        )
        updated = replace(
            updated,
            velocity=velocity,
            motion_mode_logits=endpoint_logits,
        )
        return EventOutput(
            objects=updated,
            event_logits=logits,
            pair_event_logits=pair_logits,
            boundary_event_logits=boundary_logits,
            contacts=contacts,
        )
