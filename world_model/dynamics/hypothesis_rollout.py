"""Short-step multi-hypothesis rollouts and evidence-based selection.

The selector is deliberately independent of how candidate dynamics are
constructed.  A candidate is simply a callable with the existing
``RolloutStep`` contract.  This keeps the persistent ``WorldBelief`` as the
source of truth while allowing analytic, learned, or hybrid hypotheses to be
compared on the same future observations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from numbers import Real

import torch
from torch import Tensor

from world_model.belief import BeliefTrajectory, MotionMode, WorldBelief
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.rollout import RolloutEngine, RolloutStep


class HypothesisRegime(IntEnum):
    """Causal interaction regime used to scope local model evidence.

    The regime is derived from the accepted learned/structured prediction,
    never from simulator truth or from the candidate being scored.  Keeping a
    compact taxonomy makes the applicability tensor bounded while separating
    free motion from the contact/event cases in which a transparent
    kinematic fallback is not interchangeable with the structured model.
    """

    FREE = 0
    GROUND_CONTACT = 1
    PAIR_CONTACT = 2
    COLLISION = 3
    OCCLUDED = 4
    EXTERNALLY_ACTUATED = 5


NUM_HYPOTHESIS_REGIMES = len(HypothesisRegime)


@dataclass(frozen=True)
class HypothesisApplicability:
    """One query's fail-closed local applicability decision."""

    selected_index: Tensor
    supported: Tensor
    support_count: Tensor
    age_seconds: Tensor
    observability: Tensor
    predictive_variance: Tensor
    confidence_margin: Tensor
    regime: Tensor
    position_residual: Tensor
    position_residual_supported: Tensor

    def validate(self, *, candidate_count: int) -> HypothesisApplicability:
        expected = self.selected_index.shape
        for name, value in (
            ("supported", self.supported),
            ("support_count", self.support_count),
            ("age_seconds", self.age_seconds),
            ("observability", self.observability),
            ("predictive_variance", self.predictive_variance),
            ("confidence_margin", self.confidence_margin),
            ("position_residual", self.position_residual),
            ("position_residual_supported", self.position_residual_supported),
        ):
            if value.shape != expected:
                raise ValueError(f"hypothesis applicability {name} must match selected_index")
        if self.regime.shape != expected[:2]:
            raise ValueError("hypothesis applicability regime must have shape [B,N]")
        if self.selected_index.dtype is not torch.int64 or self.regime.dtype is not torch.int64:
            raise TypeError("hypothesis applicability indices must use torch.int64")
        if self.supported.dtype is not torch.bool or self.support_count.dtype is not torch.int64:
            raise TypeError("hypothesis applicability masks/counts use bool/int64")
        if self.position_residual_supported.dtype is not torch.bool:
            raise TypeError("hypothesis residual support must use bool")
        if torch.any(self.selected_index < 0) or torch.any(self.selected_index >= candidate_count):
            raise ValueError("hypothesis applicability selected index is out of range")
        if torch.any(self.regime < 0) or torch.any(self.regime >= NUM_HYPOTHESIS_REGIMES):
            raise ValueError("hypothesis applicability regime is out of range")
        if torch.any(self.support_count < 0):
            raise ValueError("hypothesis applicability support count must be nonnegative")
        for name, value in (
            ("age_seconds", self.age_seconds),
            ("observability", self.observability),
            ("predictive_variance", self.predictive_variance),
            ("confidence_margin", self.confidence_margin),
        ):
            if not torch.isfinite(value).all() or torch.any(value < 0):
                raise ValueError(f"hypothesis applicability {name} must be finite and nonnegative")
        if not torch.isfinite(self.position_residual).all():
            raise ValueError("hypothesis applicability position residual must be finite")
        return self


@dataclass(frozen=True)
class HypothesisSelection:
    """Evidence and posterior choice for a batch of candidate rollouts."""

    scores: Tensor
    selected_index: Tensor
    posterior_weights: Tensor
    axis_scores: Tensor | None = None
    axis_posterior: Tensor | None = None
    entity_axis_scores: Tensor | None = None
    entity_axis_posterior: Tensor | None = None
    score_spread: Tensor | None = None
    axis_score_spread: Tensor | None = None
    evidence_mask: Tensor | None = None
    axis_evidence_mask: Tensor | None = None
    entity_axis_evidence_mask: Tensor | None = None
    sample_count: int = 1

    def axis_posterior_weights(self, *, temperature: float = 1.0) -> Tensor:
        """Return per-axis hypothesis weights from delayed position evidence.

        The result has shape ``[B,D,H]``.  It is separate from the joint
        posterior because a downstream runtime may compose coordinates from
        different hypotheses while keeping the joint choice for lifecycle and
        event state.  No simulator state is involved in this operation.
        """

        if self.axis_posterior is not None:
            return self.axis_posterior
        if self.axis_scores is None:
            raise RuntimeError("axis scores are unavailable for this selection")
        if temperature <= 0 or not torch.isfinite(torch.as_tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        return torch.softmax(-self.axis_scores / temperature, dim=-1)

    @property
    def axis_selected_index(self) -> Tensor | None:
        """Return the minimum delayed position-loss hypothesis per axis."""

        if self.axis_posterior is not None:
            return self.axis_posterior.argmax(dim=-1).to(torch.int64)
        if self.axis_scores is None:
            return None
        return self.axis_scores.argmin(dim=-1).to(torch.int64)

    def validate(self) -> HypothesisSelection:
        if self.scores.ndim != 2:
            raise ValueError("hypothesis scores must have shape [B,H]")
        if self.selected_index.shape != (self.scores.shape[0],):
            raise ValueError("selected_index must have shape [B]")
        if self.posterior_weights.shape != self.scores.shape:
            raise ValueError("posterior_weights must match scores")
        if self.axis_scores is not None and (
            self.axis_scores.ndim != 3
            or self.axis_scores.shape[0] != self.scores.shape[0]
            or self.axis_scores.shape[2] != self.scores.shape[1]
        ):
            raise ValueError("axis_scores must have shape [B,D,H]")
        if self.axis_posterior is not None and (
            self.axis_posterior.ndim != 3
            or self.axis_posterior.shape[0] != self.scores.shape[0]
            or self.axis_posterior.shape[2] != self.scores.shape[1]
        ):
            raise ValueError("axis_posterior must have shape [B,D,H]")
        if self.entity_axis_scores is not None and (
            self.entity_axis_scores.ndim != 4
            or self.entity_axis_scores.shape[0] != self.scores.shape[0]
            or self.entity_axis_scores.shape[-1] != self.scores.shape[1]
        ):
            raise ValueError("entity_axis_scores must have shape [B,N,D,H]")
        if self.entity_axis_posterior is not None and (
            self.entity_axis_scores is None
            or self.entity_axis_posterior.shape != self.entity_axis_scores.shape
        ):
            raise ValueError("entity_axis_posterior must match entity_axis_scores")
        if self.score_spread is not None and self.score_spread.shape != self.scores.shape:
            raise ValueError("score_spread must match scores")
        if self.axis_score_spread is not None and (
            self.axis_scores is None or self.axis_score_spread.shape != self.axis_scores.shape
        ):
            raise ValueError("axis_score_spread must match axis_scores")
        if self.evidence_mask is not None and (
            self.evidence_mask.shape != (self.scores.shape[0],)
            or self.evidence_mask.dtype is not torch.bool
        ):
            raise ValueError("evidence_mask must be boolean [B]")
        if self.axis_evidence_mask is not None and (
            self.axis_scores is None
            or self.axis_evidence_mask.shape != self.axis_scores.shape[:2]
            or self.axis_evidence_mask.dtype is not torch.bool
        ):
            raise ValueError("axis_evidence_mask must be boolean [B,D]")
        if self.entity_axis_evidence_mask is not None and (
            self.entity_axis_scores is None
            or self.entity_axis_evidence_mask.shape != self.entity_axis_scores.shape[:3]
            or self.entity_axis_evidence_mask.dtype is not torch.bool
        ):
            raise ValueError("entity_axis_evidence_mask must be boolean [B,N,D]")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.selected_index.dtype != torch.int64:
            raise TypeError("selected_index must use torch.int64")
        if (
            not torch.isfinite(self.scores).all()
            or not torch.isfinite(self.posterior_weights).all()
        ):
            raise ValueError("hypothesis selection contains NaN or Inf")
        if self.axis_posterior is not None:
            if not torch.isfinite(self.axis_posterior).all():
                raise ValueError("axis posterior contains NaN or Inf")
            if not torch.allclose(
                self.axis_posterior.sum(dim=-1),
                torch.ones(
                    self.axis_posterior.shape[:2],
                    device=self.axis_posterior.device,
                    dtype=self.axis_posterior.dtype,
                ),
                atol=1e-5,
            ):
                raise ValueError("axis posterior weights must sum to one")
        if self.entity_axis_posterior is not None:
            if not torch.isfinite(self.entity_axis_posterior).all():
                raise ValueError("entity-axis posterior contains NaN or Inf")
            if not torch.allclose(
                self.entity_axis_posterior.sum(dim=-1),
                torch.ones(
                    self.entity_axis_posterior.shape[:3],
                    device=self.entity_axis_posterior.device,
                    dtype=self.entity_axis_posterior.dtype,
                ),
                atol=1e-5,
            ):
                raise ValueError("entity-axis posterior weights must sum to one")
        for name, value in (
            ("score_spread", self.score_spread),
            ("axis_score_spread", self.axis_score_spread),
        ):
            if value is not None and (not torch.isfinite(value).all() or torch.any(value < 0)):
                raise ValueError(f"{name} must be finite and nonnegative")
        if torch.any(self.selected_index < 0) or torch.any(
            self.selected_index >= self.scores.shape[1]
        ):
            raise ValueError("selected hypothesis index is out of range")
        if not torch.allclose(
            self.posterior_weights.sum(dim=-1),
            torch.ones(self.scores.shape[0], device=self.scores.device, dtype=self.scores.dtype),
            atol=1e-5,
        ):
            raise ValueError("hypothesis posterior weights must sum to one")
        return self


class ConstantVelocityDynamics:
    """Transparent short-horizon baseline used as a selectable hypothesis."""

    def __init__(self, damping: float = 0.0) -> None:
        if damping < 0 or not torch.isfinite(torch.as_tensor(damping)):
            raise ValueError("damping must be finite and nonnegative")
        self.damping = float(damping)
        self.supported_hypothesis_regimes = (HypothesisRegime.FREE,)
        self.shared_horizon_rollout_safe = True

    def predict_step(self, belief: WorldBelief, delta_time: Tensor) -> RolloutStep:
        if delta_time.shape != belief.timestamp.shape:
            raise ValueError("delta_time must have shape [B]")
        objects = belief.objects.clone()
        active = objects.active.unsqueeze(-1)
        if self.damping:
            # Integrate ``dv/dt = -d v`` exactly.  Advancing by ``v * dt``
            # and then decaying velocity overstates displacement whenever the
            # selectable damped hypothesis is used over a non-trivial query.
            # The zero-damping branch below keeps the transparent CV case
            # exact without a numerically fragile division by ``d``.
            decay = torch.exp(-self.damping * delta_time[:, None, None])
            displacement_scale = (1.0 - decay) / self.damping
            objects.position = objects.position + objects.velocity * displacement_scale * active
            objects.velocity = objects.velocity * decay
        else:
            objects.position = (
                objects.position + objects.velocity * delta_time[:, None, None] * active
            )
        objects.fast_log_variance = (
            objects.fast_log_variance + delta_time[:, None, None] * 1.0e-3
        ).clamp(-20.0, 10.0)
        endpoint = belief.replace(
            timestamp=belief.timestamp + delta_time,
            objects=objects,
        )
        return RolloutStep(
            belief=endpoint,
            event_logits=belief.timestamp.new_full(
                (
                    belief.batch_size,
                    objects.max_objects,
                    objects.motion_mode_logits.shape[-1],
                ),
                -4.0,
            ),
            auxiliary={},
        )


class OnlineLocalAccelerationDynamics:
    """Runtime-local constant-acceleration candidate fit from RGB velocity.

    The candidate owns no persistent physical state.  It keeps only bounded
    sufficient statistics keyed by the current persistent object ID and uses
    causal temporal-velocity measurements supplied by the RGB observation
    path.  Until enough same-identity free-motion evidence exists for one
    entity-axis cell, :meth:`applicability_mask` keeps that cell on the learned
    fallback.
    """

    def __init__(
        self,
        *,
        minimum_support_count: int = 4,
        maximum_acceleration: float = 20.0,
        minimum_delta_time: float = 1.0e-3,
    ) -> None:
        if (
            not isinstance(minimum_support_count, int)
            or isinstance(minimum_support_count, bool)
            or minimum_support_count <= 0
        ):
            raise ValueError("minimum_support_count must be a positive integer")
        for name, value in (
            ("maximum_acceleration", maximum_acceleration),
            ("minimum_delta_time", minimum_delta_time),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not torch.isfinite(torch.as_tensor(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.minimum_support_count = int(minimum_support_count)
        self.maximum_acceleration = float(maximum_acceleration)
        self.minimum_delta_time = float(minimum_delta_time)
        self.supported_hypothesis_regimes = (HypothesisRegime.FREE,)
        self.shared_horizon_rollout_safe = True
        self.object_ids: Tensor | None = None
        self.acceleration: Tensor | None = None
        self.acceleration_weight: Tensor | None = None
        self.support_count: Tensor | None = None
        self.last_velocity: Tensor | None = None
        self.last_velocity_log_variance: Tensor | None = None
        self.last_velocity_valid: Tensor | None = None
        self.last_timestamp: Tensor | None = None

    def reset_runtime_state(
        self,
        batch_size: int,
        *,
        max_objects: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if batch_size <= 0 or max_objects <= 0:
            raise ValueError("runtime-local candidate dimensions must be positive")
        shape = (batch_size, max_objects, 3)
        self.object_ids = torch.full(
            (batch_size, max_objects),
            -1,
            device=device,
            dtype=torch.int64,
        )
        self.acceleration = torch.zeros(shape, device=device, dtype=dtype)
        self.acceleration_weight = torch.zeros(shape, device=device, dtype=dtype)
        self.support_count = torch.zeros(shape, device=device, dtype=torch.int64)
        self.last_velocity = torch.zeros(shape, device=device, dtype=dtype)
        self.last_velocity_log_variance = torch.zeros(shape, device=device, dtype=dtype)
        self.last_velocity_valid = torch.zeros(shape, device=device, dtype=torch.bool)
        self.last_timestamp = torch.zeros(
            (batch_size, max_objects),
            device=device,
            dtype=dtype,
        )

    def clear_runtime_state(self) -> None:
        """Drop all episode-local sufficient statistics."""

        self.object_ids = None
        self.acceleration = None
        self.acceleration_weight = None
        self.support_count = None
        self.last_velocity = None
        self.last_velocity_log_variance = None
        self.last_velocity_valid = None
        self.last_timestamp = None

    def _ensure_state(self, belief: WorldBelief) -> None:
        expected = (belief.batch_size, belief.objects.max_objects)
        if self.object_ids is None or self.object_ids.shape != expected:
            self.reset_runtime_state(
                belief.batch_size,
                max_objects=belief.objects.max_objects,
                device=belief.device,
                dtype=belief.dtype,
            )
        assert self.object_ids is not None
        assert self.acceleration is not None
        assert self.acceleration_weight is not None
        assert self.support_count is not None
        assert self.last_velocity is not None
        assert self.last_velocity_log_variance is not None
        assert self.last_velocity_valid is not None
        assert self.last_timestamp is not None
        changed = self.object_ids != belief.objects.object_id
        if bool(changed.any()):
            axis_changed = changed.unsqueeze(-1)
            self.acceleration = torch.where(
                axis_changed, torch.zeros_like(self.acceleration), self.acceleration
            )
            self.acceleration_weight = torch.where(
                axis_changed,
                torch.zeros_like(self.acceleration_weight),
                self.acceleration_weight,
            )
            self.support_count = torch.where(
                axis_changed, torch.zeros_like(self.support_count), self.support_count
            )
            self.last_velocity = torch.where(
                axis_changed, torch.zeros_like(self.last_velocity), self.last_velocity
            )
            self.last_velocity_log_variance = torch.where(
                axis_changed,
                torch.zeros_like(self.last_velocity_log_variance),
                self.last_velocity_log_variance,
            )
            self.last_velocity_valid = torch.where(
                axis_changed,
                torch.zeros_like(self.last_velocity_valid),
                self.last_velocity_valid,
            )
            self.last_timestamp = torch.where(
                changed, torch.zeros_like(self.last_timestamp), self.last_timestamp
            )
            self.object_ids = belief.objects.object_id.detach().clone()

    def assimilate_velocity_observation(
        self,
        belief: WorldBelief,
        velocity: Tensor,
        valid_axis_mask: Tensor,
        log_variance: Tensor,
        timestamp: Tensor,
    ) -> None:
        """Update bounded acceleration statistics from one associated RGB packet."""

        self._ensure_state(belief)
        expected = belief.objects.velocity.shape
        if velocity.shape != expected or log_variance.shape != expected:
            raise ValueError("local acceleration evidence must have shape [B,N,3]")
        if valid_axis_mask.shape != expected or valid_axis_mask.dtype is not torch.bool:
            raise ValueError("local acceleration validity must be boolean [B,N,3]")
        if timestamp.shape != belief.timestamp.shape:
            raise ValueError("local acceleration timestamp must have shape [B]")
        if (
            not torch.isfinite(velocity).all()
            or not torch.isfinite(log_variance).all()
            or not torch.isfinite(timestamp).all()
        ):
            raise ValueError("local acceleration evidence must be finite")
        assert self.acceleration is not None
        assert self.acceleration_weight is not None
        assert self.support_count is not None
        assert self.last_velocity is not None
        assert self.last_velocity_log_variance is not None
        assert self.last_velocity_valid is not None
        assert self.last_timestamp is not None
        mode = belief.objects.motion_mode_logits.argmax(dim=-1)
        free = (mode == int(MotionMode.FREE)) & belief.objects.active
        reset_entity = ~free
        reset_axis = reset_entity.unsqueeze(-1)
        self.acceleration = torch.where(
            reset_axis,
            torch.zeros_like(self.acceleration),
            self.acceleration,
        )
        self.acceleration_weight = torch.where(
            reset_axis,
            torch.zeros_like(self.acceleration_weight),
            self.acceleration_weight,
        )
        self.support_count = torch.where(
            reset_axis,
            torch.zeros_like(self.support_count),
            self.support_count,
        )
        self.last_velocity_valid = torch.where(
            reset_axis,
            torch.zeros_like(self.last_velocity_valid),
            self.last_velocity_valid,
        )
        self.last_timestamp = torch.where(
            reset_entity,
            torch.zeros_like(self.last_timestamp),
            self.last_timestamp,
        )
        valid = valid_axis_mask & free.unsqueeze(-1)
        delta_time = timestamp[:, None] - self.last_timestamp
        eligible = (
            valid & self.last_velocity_valid & (delta_time.unsqueeze(-1) >= self.minimum_delta_time)
        )
        safe_dt = delta_time.clamp_min(self.minimum_delta_time).unsqueeze(-1)
        raw_acceleration = ((velocity - self.last_velocity) / safe_dt).clamp(
            min=-self.maximum_acceleration,
            max=self.maximum_acceleration,
        )
        current_variance = log_variance.clamp(-20.0, 10.0).exp()
        previous_variance = self.last_velocity_log_variance.clamp(-20.0, 10.0).exp()
        observation_weight = (
            safe_dt.square() / (current_variance + previous_variance).clamp_min(1.0e-8)
        ).clamp(max=1.0e3)
        proposed_weight = self.acceleration_weight + observation_weight
        proposed_acceleration = (
            self.acceleration * self.acceleration_weight + raw_acceleration * observation_weight
        ) / proposed_weight.clamp_min(1.0e-8)
        self.acceleration = torch.where(eligible, proposed_acceleration, self.acceleration).detach()
        self.acceleration_weight = torch.where(
            eligible, proposed_weight, self.acceleration_weight
        ).detach()
        self.support_count = torch.where(
            eligible, self.support_count + 1, self.support_count
        ).detach()
        self.last_velocity = torch.where(valid, velocity, self.last_velocity).detach()
        self.last_velocity_log_variance = torch.where(
            valid, log_variance, self.last_velocity_log_variance
        ).detach()
        self.last_velocity_valid = valid.detach()
        observed_entity = valid.any(dim=-1)
        self.last_timestamp = torch.where(
            observed_entity,
            timestamp[:, None].expand_as(self.last_timestamp),
            self.last_timestamp,
        ).detach()

    def applicability_mask(self, belief: WorldBelief) -> Tensor:
        """Return entity-axis cells with enough current-identity observations."""

        self._ensure_state(belief)
        assert self.support_count is not None
        assert self.object_ids is not None
        return (
            belief.objects.active.unsqueeze(-1)
            & (self.object_ids == belief.objects.object_id).unsqueeze(-1)
            & (self.support_count >= self.minimum_support_count)
        )

    def predict_step(self, belief: WorldBelief, delta_time: Tensor) -> RolloutStep:
        if delta_time.shape != belief.timestamp.shape:
            raise ValueError("delta_time must have shape [B]")
        self._ensure_state(belief)
        assert self.acceleration is not None
        supported = self.applicability_mask(belief)
        objects = belief.objects.clone()
        active = objects.active.unsqueeze(-1)
        dt = delta_time[:, None, None]
        applied_acceleration = torch.where(
            supported, self.acceleration, torch.zeros_like(self.acceleration)
        )
        objects.position = objects.position + objects.velocity * dt * active
        objects.position = objects.position + 0.5 * applied_acceleration * dt.square() * active
        objects.velocity = objects.velocity + applied_acceleration * dt * active
        objects.fast_log_variance = (
            objects.fast_log_variance + delta_time[:, None, None] * 1.0e-3
        ).clamp(-20.0, 10.0)
        endpoint = belief.replace(
            timestamp=belief.timestamp + delta_time,
            objects=objects,
        )
        return RolloutStep(
            belief=endpoint,
            event_logits=belief.timestamp.new_full(
                (
                    belief.batch_size,
                    objects.max_objects,
                    objects.motion_mode_logits.shape[-1],
                ),
                -4.0,
            ),
            auxiliary={
                "online_local_acceleration": applied_acceleration,
                "online_local_acceleration_supported": supported,
            },
        )


class BallisticContactDynamics:
    """Analytic gravity/drag hypothesis with explicit contact-event logits.

    This candidate deliberately contains no learned interaction weights. It is
    useful as a heterogeneous alternative for locally ballistic motion while
    still exposing conservative ground and sphere-contact event evidence.
    """

    def __init__(self, *, ground_height: float = 0.0, event_logit: float = 5.0) -> None:
        if not torch.isfinite(torch.as_tensor(ground_height)):
            raise ValueError("ground_height must be finite")
        if event_logit <= 0 or not torch.isfinite(torch.as_tensor(event_logit)):
            raise ValueError("event_logit must be finite and positive")
        self.ground_height = float(ground_height)
        self.event_logit = float(event_logit)
        self.analytic = AnalyticKinematics()
        self.supported_hypothesis_regimes = (
            HypothesisRegime.FREE,
            HypothesisRegime.GROUND_CONTACT,
            HypothesisRegime.PAIR_CONTACT,
        )
        # Contact resolution is path dependent at query boundaries, so this
        # candidate must retain independent source-to-horizon evaluation.
        self.shared_horizon_rollout_safe = False

    def predict_step(self, belief: WorldBelief, delta_time: Tensor) -> RolloutStep:
        if delta_time.shape != belief.timestamp.shape:
            raise ValueError("delta_time must have shape [B]")
        before = belief.objects
        after = self.analytic(before, belief.gravity, delta_time)
        radius = before.geometry[..., :1].clamp_min(1.0e-5)
        before_ground = before.position[..., 1] - radius[..., 0]
        after_ground = after.position[..., 1] - radius[..., 0]
        ground_event = (
            before.active
            & after.active
            & (before_ground > self.ground_height)
            & (after_ground <= self.ground_height)
            & (before.velocity[..., 1] < 0)
        )
        before_delta = before.position[:, :, None, :] - before.position[:, None, :, :]
        after_delta = after.position[:, :, None, :] - after.position[:, None, :, :]
        before_distance = torch.linalg.vector_norm(before_delta, dim=-1)
        after_distance = torch.linalg.vector_norm(after_delta, dim=-1)
        contact_distance = radius[:, :, None, 0] + radius[:, None, :, 0]
        pair_event = (
            before.active[:, :, None]
            & before.active[:, None, :]
            & (before_distance > contact_distance)
            & (after_distance <= contact_distance)
        )
        normal = before_delta / before_distance.clamp_min(1.0e-6).unsqueeze(-1)
        relative_velocity = after.velocity[:, :, None, :] - after.velocity[:, None, :, :]
        approaching = (relative_velocity * normal).sum(dim=-1) < 0
        pair_contact = pair_event & approaching
        collision = ground_event | pair_contact.any(dim=-1)
        # Keep the analytic event hypothesis physically coherent: a detected
        # ground crossing produces an explicit restitution jump and clamps the
        # contact point to the surface. Pair-contact events remain observable
        # but are not resolved here because their normals require a full
        # collision solver.
        contact_position = after.position.clone()
        contact_position[..., 1] = torch.where(
            ground_event,
            torch.as_tensor(
                self.ground_height, device=after.position.device, dtype=after.position.dtype
            )
            + radius[..., 0],
            contact_position[..., 1],
        )
        contact_velocity = after.velocity.clone()
        contact_velocity[..., 1] = torch.where(
            ground_event,
            after.velocity[..., 1].abs() * before.restitution[..., 0],
            contact_velocity[..., 1],
        )
        after = after.replace(position=contact_position, velocity=contact_velocity)
        # Resolve approaching sphere contacts as a one-step equal-and-opposite
        # impulse. The symmetric pair matrix makes momentum exchange explicit
        # while preserving the persistent belief contract.
        pair_restitution = (
            before.restitution[..., 0][:, :, None] + before.restitution[..., 0][:, None, :]
        ) * 0.5
        impulse = -(1.0 + pair_restitution) * (relative_velocity * normal).sum(dim=-1)
        inverse_mass = before.mass[..., 0].reciprocal()
        impulse = torch.where(
            pair_contact,
            impulse / (inverse_mass[:, :, None] + inverse_mass[:, None, :]).clamp_min(1.0e-6),
            torch.zeros_like(impulse),
        )
        impulse_delta = (impulse.unsqueeze(-1) * normal).sum(dim=2)
        pair_velocity = after.velocity + impulse_delta * inverse_mass.unsqueeze(-1)
        after = after.replace(velocity=pair_velocity)
        event_logits = torch.full_like(before.motion_mode_logits, -4.0)
        event_logits[..., MotionMode.COLLISION] = torch.where(
            collision,
            event_logits.new_full((), self.event_logit),
            event_logits[..., MotionMode.COLLISION],
        )
        after = after.replace(
            fast_log_variance=(after.fast_log_variance + delta_time[:, None, None] * 1.0e-3).clamp(
                -20.0, 10.0
            )
        )
        return RolloutStep(
            belief=belief.replace(timestamp=belief.timestamp + delta_time, objects=after),
            event_logits=event_logits,
            auxiliary={"collision_event": collision},
        )


class HypothesisRolloutEngine:
    """Run short-step candidate rollouts and score them against observations.

    ``RolloutEngine`` already advances a cloned belief in chronological query
    order.  This wrapper runs that exact contract once per candidate, then
    computes a batch-wise score.  It does not mutate the source belief or
    update model parameters.  The caller can use ``selected_index`` to choose
    the candidate for the next receding-horizon cycle.
    """

    def __init__(self, rollout_engine: RolloutEngine | None = None) -> None:
        self.rollout_engine = rollout_engine or RolloutEngine()

    def rollout(
        self,
        predictors: Sequence[Callable[[WorldBelief, Tensor], RolloutStep]],
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
    ) -> list[BeliefTrajectory]:
        if not predictors:
            raise ValueError("at least one hypothesis predictor is required")
        return [
            self.rollout_engine.rollout(predictor, belief, query_times) for predictor in predictors
        ]

    def rollout_dynamics(
        self,
        dynamics_models: Sequence[object],
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
    ) -> list[BeliefTrajectory]:
        """Roll out models exposing the standard ``predict_step`` method.

        Keeping this adapter structural avoids coupling the selector to one
        neural architecture and permits analytic, learned, or physics-engine
        backed candidates to coexist in the same pool.
        """

        predictors: list[Callable[[WorldBelief, Tensor], RolloutStep]] = []
        for model in dynamics_models:
            predict_step = getattr(model, "predict_step", None)
            if not callable(predict_step):
                raise TypeError("every dynamics hypothesis must expose predict_step")
            predictors.append(predict_step)
        return self.rollout(predictors, belief, query_times)

    @staticmethod
    def score(
        trajectories: Sequence[BeliefTrajectory],
        target_positions: Tensor,
        target_mask: Tensor,
        *,
        target_position_log_variance: Tensor | None = None,
        target_velocities: Tensor | None = None,
        target_velocity_axis_mask: Tensor | None = None,
        target_velocity_log_variance: Tensor | None = None,
        target_collision: Tensor | None = None,
        position_weight: float = 1.0,
        velocity_weight: float = 0.0,
        velocity_nonregression_gate_enabled: bool = False,
        lifecycle_weight: float = 0.0,
        event_weight: float = 0.0,
        position_gate_ratio: float = 0.0,
        axis_gate_ratio: float = 0.0,
        event_gate_ratio: float = 0.0,
        axis_weights: Sequence[float] | Tensor | None = None,
        uncertainty_aware: bool = True,
        temperature: float = 1.0,
    ) -> HypothesisSelection:
        """Score candidates by masked physical NLL and select per batch item.

        ``target_mask`` is ``[B,T,N]`` and permits asynchronous/occluded
        observations.  With uncertainty enabled, the score is the Gaussian
        diagonal NLL up to a constant using predictive plus measurement
        variance.  This prevents either a deliberately over-wide candidate or
        a noisy RGB localization from dominating the selector incorrectly.
        """

        if not trajectories:
            raise ValueError("at least one trajectory is required")
        reference = trajectories[0]
        if target_positions.shape != reference.positions.shape:
            raise ValueError("target_positions must match trajectory positions")
        if target_mask.shape != reference.active_mask.shape:
            raise ValueError("target_mask must have shape [B,T,N]")
        if target_mask.dtype is not torch.bool:
            raise TypeError("target_mask must use torch.bool")
        if target_position_log_variance is not None:
            if target_position_log_variance.shape != target_positions.shape:
                raise ValueError("target_position_log_variance must match target_positions")
            if not torch.isfinite(target_position_log_variance).all():
                raise ValueError("target_position_log_variance contains NaN or Inf")
        velocity_fields = (
            target_velocities,
            target_velocity_axis_mask,
            target_velocity_log_variance,
        )
        if any(value is not None for value in velocity_fields):
            if not all(isinstance(value, Tensor) for value in velocity_fields):
                raise ValueError("velocity targets require values, axis mask, and log variance")
            assert target_velocities is not None
            assert target_velocity_axis_mask is not None
            assert target_velocity_log_variance is not None
            if target_velocities.shape != target_positions.shape:
                raise ValueError("target_velocities must match target_positions")
            if target_velocity_axis_mask.shape != target_positions.shape:
                raise ValueError("target_velocity_axis_mask must match target_positions")
            if target_velocity_axis_mask.dtype is not torch.bool:
                raise TypeError("target_velocity_axis_mask must use torch.bool")
            if target_velocity_log_variance.shape != target_positions.shape:
                raise ValueError("target_velocity_log_variance must match target_positions")
            if (
                not torch.isfinite(target_velocities).all()
                or not torch.isfinite(target_velocity_log_variance).all()
            ):
                raise ValueError("velocity targets contain NaN or Inf")
        elif velocity_weight or velocity_nonregression_gate_enabled:
            raise ValueError("velocity scoring requires explicit velocity evidence")
        if not isinstance(velocity_nonregression_gate_enabled, bool):
            raise TypeError("velocity_nonregression_gate_enabled must use bool")
        if temperature <= 0 or not torch.isfinite(torch.as_tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        for name, value in (
            ("position_weight", position_weight),
            ("velocity_weight", velocity_weight),
            ("lifecycle_weight", lifecycle_weight),
            ("event_weight", event_weight),
            ("position_gate_ratio", position_gate_ratio),
            ("axis_gate_ratio", axis_gate_ratio),
            ("event_gate_ratio", event_gate_ratio),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or value < 0
                or not torch.isfinite(torch.as_tensor(value))
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if position_weight + velocity_weight + lifecycle_weight + event_weight <= 0:
            raise ValueError("at least one hypothesis score weight must be positive")
        if position_gate_ratio and position_weight <= 0:
            raise ValueError("position_gate_ratio requires a positive position_weight")
        if axis_gate_ratio and position_weight <= 0:
            raise ValueError("axis_gate_ratio requires a positive position_weight")
        if event_gate_ratio and event_weight <= 0:
            raise ValueError("event_gate_ratio requires a positive event_weight")
        if axis_weights is None:
            resolved_axis_weights = target_positions.new_ones((target_positions.shape[-1],))
        else:
            resolved_axis_weights = torch.as_tensor(
                axis_weights, device=target_positions.device, dtype=target_positions.dtype
            )
            if resolved_axis_weights.shape != (target_positions.shape[-1],):
                raise ValueError("axis_weights must have one finite nonnegative value per axis")
            if not torch.isfinite(resolved_axis_weights).all() or torch.any(
                resolved_axis_weights < 0
            ):
                raise ValueError("axis_weights must have finite nonnegative values")
            if not torch.any(resolved_axis_weights > 0):
                raise ValueError("axis_weights must contain at least one positive value")
        if not torch.isfinite(target_positions).all():
            raise ValueError("target_positions contains NaN or Inf")
        if target_collision is not None:
            if target_collision.shape != reference.active_mask.shape:
                raise ValueError("target_collision must have shape [B,T,N]")
            if target_collision.dtype is not torch.bool:
                raise TypeError("target_collision must use torch.bool")

        mask = target_mask.unsqueeze(-1)
        evidence_mask = target_mask.any(dim=(1, 2))
        axis_evidence_mask = evidence_mask.unsqueeze(-1).expand(-1, target_positions.shape[-1])
        entity_axis_evidence_mask = (
            target_mask.any(dim=1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                target_positions.shape[-1],
            )
        )
        uses_velocity_evidence = bool(velocity_weight or velocity_nonregression_gate_enabled)
        if uses_velocity_evidence:
            assert target_velocity_axis_mask is not None
            velocity_axis_evidence_mask = target_velocity_axis_mask.any(dim=(1, 2))
            velocity_entity_axis_evidence_mask = target_velocity_axis_mask.any(dim=1)
            axis_evidence_mask = axis_evidence_mask & velocity_axis_evidence_mask
            entity_axis_evidence_mask = (
                entity_axis_evidence_mask & velocity_entity_axis_evidence_mask
            )
        valid_count = mask.sum(dim=(1, 2, 3)).clamp_min(1).to(target_positions.dtype)
        axis_valid_count = target_mask.sum(dim=(1, 2)).clamp_min(1).to(target_positions.dtype)
        entity_valid_count = target_mask.sum(dim=1).clamp_min(1).to(target_positions.dtype)
        measurement_variance = (
            target_position_log_variance.clamp(-20.0, 10.0).exp()
            if target_position_log_variance is not None
            else None
        )
        velocity_measurement_variance = (
            target_velocity_log_variance.clamp(-20.0, 10.0).exp()
            if target_velocity_log_variance is not None
            else None
        )
        candidate_scores: list[Tensor] = []
        position_scores: list[Tensor] = []
        axis_position_scores: list[Tensor] = []
        entity_axis_position_scores: list[Tensor] = []
        axis_velocity_scores: list[Tensor] = []
        entity_axis_velocity_scores: list[Tensor] = []
        event_scores: list[Tensor] = []
        for trajectory in trajectories:
            if trajectory.positions.shape != target_positions.shape:
                raise ValueError("all trajectories must share target position shape")
            residual = trajectory.positions - target_positions
            if uncertainty_aware:
                if trajectory.fast_log_variance.shape[:3] != trajectory.positions.shape[:3]:
                    raise ValueError("trajectory uncertainty shape does not match positions")
                log_variance = trajectory.fast_log_variance[..., :3].clamp(-20.0, 10.0)
                variance = log_variance.exp()
                if measurement_variance is not None:
                    variance = variance + measurement_variance
                    log_variance = variance.log()
                point_loss = residual.square() / variance + log_variance
            else:
                point_loss = residual.square()
            axis_position_scores.append(
                (point_loss * mask).sum(dim=(1, 2)) / axis_valid_count.unsqueeze(-1)
            )
            entity_axis_position_scores.append(
                (point_loss * mask).sum(dim=1) / entity_valid_count.unsqueeze(-1)
            )
            point_loss = point_loss * resolved_axis_weights.view(1, 1, 1, -1)
            position_score = (point_loss * mask).sum(dim=(1, 2, 3)) / valid_count
            position_scores.append(position_score)
            score = position_weight * position_score
            if uses_velocity_evidence:
                assert target_velocities is not None
                assert target_velocity_axis_mask is not None
                velocity_mask = target_velocity_axis_mask & target_mask.unsqueeze(-1)
                velocity_count = (
                    velocity_mask.sum(dim=(1, 2, 3)).clamp_min(1).to(target_positions.dtype)
                )
                velocity_axis_count = (
                    velocity_mask.sum(dim=(1, 2)).clamp_min(1).to(target_positions.dtype)
                )
                velocity_entity_axis_count = (
                    velocity_mask.sum(dim=1).clamp_min(1).to(target_positions.dtype)
                )
                velocity_residual = trajectory.velocities - target_velocities
                if uncertainty_aware:
                    velocity_log_variance = trajectory.fast_log_variance[..., 3:6].clamp(
                        -20.0, 10.0
                    )
                    velocity_variance = velocity_log_variance.exp()
                    if velocity_measurement_variance is not None:
                        velocity_variance = velocity_variance + velocity_measurement_variance
                        velocity_log_variance = velocity_variance.log()
                    velocity_point_loss = (
                        velocity_residual.square() / velocity_variance + velocity_log_variance
                    )
                else:
                    velocity_point_loss = velocity_residual.square()
                velocity_axis_score = (velocity_point_loss * velocity_mask).sum(
                    dim=(1, 2)
                ) / velocity_axis_count
                velocity_entity_axis_score = (velocity_point_loss * velocity_mask).sum(
                    dim=1
                ) / velocity_entity_axis_count
                axis_velocity_scores.append(velocity_axis_score)
                entity_axis_velocity_scores.append(velocity_entity_axis_score)
                if velocity_weight:
                    axis_position_scores[-1] = (
                        position_weight * axis_position_scores[-1]
                        + velocity_weight * velocity_axis_score
                    )
                    entity_axis_position_scores[-1] = (
                        position_weight * entity_axis_position_scores[-1]
                        + velocity_weight * velocity_entity_axis_score
                    )
                velocity_score = (velocity_point_loss * velocity_mask).sum(
                    dim=(1, 2, 3)
                ) / velocity_count
                score = score + velocity_weight * velocity_score
            if lifecycle_weight:
                lifecycle_loss = (
                    (trajectory.active_mask != target_mask)
                    .to(target_positions.dtype)
                    .mean(dim=(1, 2))
                )
                score = score + lifecycle_weight * lifecycle_loss
            if event_weight:
                if target_collision is None or trajectory.event_logits is None:
                    raise ValueError(
                        "event_weight requires target_collision and trajectory event_logits"
                    )
                event_logits = trajectory.event_logits[..., MotionMode.COLLISION]
                event_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    event_logits,
                    target_collision.to(event_logits.dtype),
                    reduction="none",
                ).mean(dim=(1, 2))
                event_scores.append(event_loss)
                score = score + event_weight * event_loss
            candidate_scores.append(score)

        scores = torch.stack(candidate_scores, dim=-1)
        if position_gate_ratio:
            position_matrix = torch.stack(position_scores, dim=-1)
            best_position = position_matrix.amin(dim=-1, keepdim=True)
            allowed = position_matrix <= best_position * (1.0 + position_gate_ratio) + 1.0e-8
            scores = scores + torch.where(
                allowed,
                torch.zeros_like(scores),
                scores.new_full((), 1.0e6),
            )
        if axis_gate_ratio:
            axis_matrix = torch.stack(axis_position_scores, dim=-1)
            best_axis = axis_matrix.amin(dim=-1, keepdim=True)
            axis_allowed = axis_matrix <= best_axis * (1.0 + axis_gate_ratio) + 1.0e-8
            scores = scores + torch.where(
                axis_allowed.all(dim=1),
                torch.zeros_like(scores),
                scores.new_full((), 1.0e6),
            )
        if event_gate_ratio:
            if not event_scores:
                raise ValueError("event_gate_ratio requires event scoring")
            event_matrix = torch.stack(event_scores, dim=-1)
            best_event = event_matrix.amin(dim=-1, keepdim=True)
            event_allowed = event_matrix <= best_event * (1.0 + event_gate_ratio) + 1.0e-8
            scores = scores + torch.where(
                event_allowed,
                torch.zeros_like(scores),
                scores.new_full((), 1.0e6),
            )
        axis_score_matrix = torch.stack(axis_position_scores, dim=-1)
        entity_axis_score_matrix = torch.stack(entity_axis_position_scores, dim=-1)
        if velocity_nonregression_gate_enabled:
            axis_velocity_matrix = torch.stack(axis_velocity_scores, dim=-1)
            entity_axis_velocity_matrix = torch.stack(entity_axis_velocity_scores, dim=-1)
            axis_velocity_allowed = axis_velocity_matrix <= axis_velocity_matrix[..., :1] + 1.0e-8
            entity_axis_velocity_allowed = (
                entity_axis_velocity_matrix <= entity_axis_velocity_matrix[..., :1] + 1.0e-8
            )
            penalty = scores.new_full((), 1.0e6)
            axis_score_matrix = axis_score_matrix + torch.where(
                axis_velocity_allowed,
                torch.zeros_like(axis_score_matrix),
                penalty,
            )
            entity_axis_score_matrix = entity_axis_score_matrix + torch.where(
                entity_axis_velocity_allowed,
                torch.zeros_like(entity_axis_score_matrix),
                penalty,
            )
        scale = torch.as_tensor(temperature, device=scores.device, dtype=scores.dtype)
        posterior_weights = torch.softmax(-scores / scale, dim=-1)
        selected_index = scores.argmin(dim=-1).to(torch.int64)
        return HypothesisSelection(
            scores,
            selected_index,
            posterior_weights,
            axis_scores=axis_score_matrix,
            entity_axis_scores=entity_axis_score_matrix,
            evidence_mask=evidence_mask,
            axis_evidence_mask=axis_evidence_mask,
            entity_axis_evidence_mask=entity_axis_evidence_mask,
        ).validate()

    @staticmethod
    def _robust_aggregate(sample_values: Tensor, risk_penalty: float) -> tuple[Tensor, Tensor]:
        """Aggregate candidate losses over nearby mental-simulation samples.

        The mean captures expected delayed prediction error.  The optional
        standard-deviation penalty rejects a candidate whose apparent success
        depends on one brittle imagined world, while preserving the exact
        single-sample score when ``risk_penalty`` is zero.
        """

        if sample_values.ndim < 2:
            raise ValueError("sample values must begin with [S,B,...]")
        if risk_penalty < 0 or not torch.isfinite(torch.as_tensor(risk_penalty)):
            raise ValueError("risk_penalty must be finite and nonnegative")
        mean = sample_values.mean(dim=0)
        spread = sample_values.std(dim=0, unbiased=False)
        return mean + float(risk_penalty) * spread, spread

    def score_ensemble(
        self,
        trajectory_samples: Sequence[Sequence[BeliefTrajectory]],
        target_positions: Tensor,
        target_mask: Tensor,
        *,
        risk_penalty: float = 0.0,
        temperature: float = 1.0,
        **score_kwargs: object,
    ) -> HypothesisSelection:
        """Score a set of nearby short-horizon rollout worlds robustly.

        Each sample contains one trajectory per candidate in the same order.
        Samples can represent belief uncertainty, action perturbations, or
        alternate model parameters, but are always evaluated only when real
        delayed evidence is available.  This preserves asynchronous online
        correction and does not alter ``WorldBelief``.
        """

        if not trajectory_samples:
            raise ValueError("trajectory_samples must be nonempty")
        selections = [
            self.score(sample, target_positions, target_mask, **score_kwargs)
            for sample in trajectory_samples
        ]
        candidate_count = selections[0].scores.shape[-1]
        if any(selection.scores.shape != selections[0].scores.shape for selection in selections):
            raise ValueError("all ensemble samples must share score shape")
        if any(selection.scores.shape[-1] != candidate_count for selection in selections):
            raise ValueError("all ensemble samples must share candidate order")
        if any(
            not torch.equal(selection.evidence_mask, selections[0].evidence_mask)
            or not torch.equal(
                selection.axis_evidence_mask,
                selections[0].axis_evidence_mask,
            )
            or not torch.equal(
                selection.entity_axis_evidence_mask,
                selections[0].entity_axis_evidence_mask,
            )
            for selection in selections[1:]
        ):
            raise ValueError("all ensemble samples must share evidence masks")
        scores, score_spread = self._robust_aggregate(
            torch.stack([selection.scores for selection in selections]), risk_penalty
        )
        axis_scores = None
        axis_score_spread = None
        entity_axis_scores = None
        if selections[0].axis_scores is not None:
            if any(selection.axis_scores is None for selection in selections):
                raise ValueError("all ensemble samples must expose axis scores")
            axis_scores, axis_score_spread = self._robust_aggregate(
                torch.stack(
                    [
                        selection.axis_scores
                        for selection in selections
                        if selection.axis_scores is not None
                    ]
                ),
                risk_penalty,
            )
        if selections[0].entity_axis_scores is not None:
            if any(selection.entity_axis_scores is None for selection in selections):
                raise ValueError("all ensemble samples must expose entity-axis scores")
            entity_axis_scores, _ = self._robust_aggregate(
                torch.stack(
                    [
                        selection.entity_axis_scores
                        for selection in selections
                        if selection.entity_axis_scores is not None
                    ]
                ),
                risk_penalty,
            )
        if temperature <= 0 or not torch.isfinite(torch.as_tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        scale = torch.as_tensor(temperature, device=scores.device, dtype=scores.dtype)
        return HypothesisSelection(
            scores=scores,
            selected_index=scores.argmin(dim=-1).to(torch.int64),
            posterior_weights=torch.softmax(-scores / scale, dim=-1),
            axis_scores=axis_scores,
            entity_axis_scores=entity_axis_scores,
            score_spread=score_spread,
            axis_score_spread=axis_score_spread,
            evidence_mask=selections[0].evidence_mask,
            axis_evidence_mask=selections[0].axis_evidence_mask,
            entity_axis_evidence_mask=selections[0].entity_axis_evidence_mask,
            sample_count=len(selections),
        ).validate()


class HypothesisDynamicsPool:
    """Persistent candidate pool for receding-horizon model selection.

    The pool owns only candidate dynamics and their evidence weights.  The
    caller continues to own the authoritative ``WorldBelief``.  Evidence can
    arrive asynchronously: a rollout may be made first and scored later when
    matching observations become available.
    """

    def __init__(
        self,
        dynamics_models: Sequence[object],
        *,
        rollout_engine: HypothesisRolloutEngine | None = None,
        temperature: float = 1.0,
        evidence_decay: float = 1.0,
    ) -> None:
        if not dynamics_models:
            raise ValueError("HypothesisDynamicsPool requires at least one model")
        if temperature <= 0 or not torch.isfinite(torch.as_tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        if (
            evidence_decay <= 0
            or evidence_decay > 1
            or not torch.isfinite(torch.as_tensor(evidence_decay))
        ):
            raise ValueError("evidence_decay must lie in (0,1]")
        for model in dynamics_models:
            if not callable(getattr(model, "predict_step", None)):
                raise TypeError("every hypothesis model must expose predict_step")
        self.dynamics_models = tuple(dynamics_models)
        self.rollout_engine = rollout_engine or HypothesisRolloutEngine()
        self.temperature = float(temperature)
        self.evidence_decay = float(evidence_decay)
        self.log_weights: Tensor | None = None
        self.axis_log_weights: Tensor | None = None
        self.evidence_seen: Tensor | None = None
        self.axis_evidence_seen: Tensor | None = None
        self.entity_axis_log_weights: Tensor | None = None
        self.entity_axis_evidence_seen: Tensor | None = None
        self.entity_axis_regime_log_weights: Tensor | None = None
        self.entity_axis_regime_support_count: Tensor | None = None
        self.entity_axis_regime_last_timestamp: Tensor | None = None
        self.entity_axis_regime_observability: Tensor | None = None
        self.entity_axis_regime_predictive_variance: Tensor | None = None
        self.entity_axis_regime_position_residual: Tensor | None = None
        self.entity_axis_regime_position_residual_supported: Tensor | None = None
        self.entity_object_ids: Tensor | None = None
        self.entity_active: Tensor | None = None
        self.last_selection: HypothesisSelection | None = None

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.log_weights = torch.zeros(
            batch_size,
            len(self.dynamics_models),
            device=device,
            dtype=dtype,
        )
        self.axis_log_weights = torch.zeros(
            batch_size,
            3,
            len(self.dynamics_models),
            device=device,
            dtype=dtype,
        )
        self.evidence_seen = torch.zeros(batch_size, device=device, dtype=torch.bool)
        self.axis_evidence_seen = torch.zeros(batch_size, 3, device=device, dtype=torch.bool)
        self.entity_axis_log_weights = None
        self.entity_axis_evidence_seen = None
        self.entity_axis_regime_log_weights = None
        self.entity_axis_regime_support_count = None
        self.entity_axis_regime_last_timestamp = None
        self.entity_axis_regime_observability = None
        self.entity_axis_regime_predictive_variance = None
        self.entity_axis_regime_position_residual = None
        self.entity_axis_regime_position_residual_supported = None
        self.entity_object_ids = None
        self.entity_active = None
        self.last_selection = None

    def _ensure_weights(self, belief: WorldBelief) -> Tensor:
        if self.log_weights is None:
            self.reset(belief.batch_size, device=belief.device, dtype=belief.dtype)
        assert self.log_weights is not None
        if self.log_weights.shape[0] != belief.batch_size:
            raise ValueError("belief batch size changed; reset the hypothesis pool")
        if self.log_weights.device != belief.device or self.log_weights.dtype != belief.dtype:
            raise ValueError("hypothesis weights must share belief device and dtype")
        assert self.axis_log_weights is not None
        assert self.evidence_seen is not None
        assert self.axis_evidence_seen is not None
        if self.axis_log_weights.shape != (
            belief.batch_size,
            3,
            len(self.dynamics_models),
        ):
            raise ValueError("axis hypothesis weights have incompatible shape")
        if (
            self.axis_log_weights.device != belief.device
            or self.axis_log_weights.dtype != belief.dtype
            or self.evidence_seen.device != belief.device
            or self.axis_evidence_seen.device != belief.device
        ):
            raise ValueError("hypothesis evidence state must share belief device and dtype")
        entity_shape = (
            belief.batch_size,
            belief.objects.max_objects,
            3,
            len(self.dynamics_models),
        )
        if self.entity_axis_log_weights is None:
            self.entity_axis_log_weights = belief.objects.position.new_zeros(entity_shape)
            self.entity_axis_evidence_seen = torch.zeros(
                entity_shape[:3],
                device=belief.device,
                dtype=torch.bool,
            )
            self.entity_object_ids = belief.objects.object_id.detach().clone()
            self.entity_active = belief.objects.active.detach().clone()
            regime_shape = (*entity_shape[:3], NUM_HYPOTHESIS_REGIMES)
            self.entity_axis_regime_log_weights = belief.objects.position.new_zeros(
                (*regime_shape, len(self.dynamics_models))
            )
            self.entity_axis_regime_support_count = torch.zeros(
                regime_shape,
                device=belief.device,
                dtype=torch.int64,
            )
            self.entity_axis_regime_last_timestamp = belief.objects.position.new_zeros(regime_shape)
            self.entity_axis_regime_observability = belief.objects.position.new_zeros(regime_shape)
            self.entity_axis_regime_predictive_variance = belief.objects.position.new_zeros(
                (*regime_shape, len(self.dynamics_models))
            )
            self.entity_axis_regime_position_residual = belief.objects.position.new_zeros(
                regime_shape
            )
            self.entity_axis_regime_position_residual_supported = torch.zeros(
                regime_shape,
                device=belief.device,
                dtype=torch.bool,
            )
        assert self.entity_axis_evidence_seen is not None
        assert self.entity_axis_regime_log_weights is not None
        assert self.entity_axis_regime_support_count is not None
        assert self.entity_axis_regime_last_timestamp is not None
        assert self.entity_axis_regime_observability is not None
        assert self.entity_axis_regime_predictive_variance is not None
        assert self.entity_axis_regime_position_residual is not None
        assert self.entity_axis_regime_position_residual_supported is not None
        assert self.entity_object_ids is not None
        assert self.entity_active is not None
        if self.entity_axis_log_weights.shape != entity_shape:
            raise ValueError("entity-axis hypothesis weights have incompatible shape")
        expected_regime_shape = (*entity_shape[:3], NUM_HYPOTHESIS_REGIMES)
        if self.entity_axis_regime_log_weights.shape != (
            *expected_regime_shape,
            len(self.dynamics_models),
        ):
            raise ValueError("entity-axis-regime hypothesis weights have incompatible shape")
        for name, value in (
            ("support count", self.entity_axis_regime_support_count),
            ("last timestamp", self.entity_axis_regime_last_timestamp),
            ("observability", self.entity_axis_regime_observability),
        ):
            if value.shape != expected_regime_shape:
                raise ValueError(f"entity-axis-regime {name} has incompatible shape")
        if self.entity_axis_regime_predictive_variance.shape != (
            *expected_regime_shape,
            len(self.dynamics_models),
        ):
            raise ValueError("entity-axis-regime predictive variance has incompatible shape")
        if self.entity_axis_regime_position_residual.shape != expected_regime_shape:
            raise ValueError("entity-axis-regime position residual has incompatible shape")
        if self.entity_axis_regime_position_residual_supported.shape != expected_regime_shape:
            raise ValueError("entity-axis-regime residual support has incompatible shape")
        if self.entity_axis_regime_position_residual_supported.dtype != torch.bool:
            raise TypeError("entity-axis-regime residual support must use torch.bool")
        if self.entity_axis_regime_support_count.dtype != torch.int64:
            raise TypeError("entity-axis-regime support count must use torch.int64")
        if (
            self.entity_axis_log_weights.device != belief.device
            or self.entity_axis_log_weights.dtype != belief.dtype
            or self.entity_axis_evidence_seen.device != belief.device
            or self.entity_axis_regime_log_weights.device != belief.device
            or self.entity_axis_regime_log_weights.dtype != belief.dtype
            or self.entity_axis_regime_support_count.device != belief.device
            or self.entity_axis_regime_last_timestamp.device != belief.device
            or self.entity_axis_regime_last_timestamp.dtype != belief.dtype
            or self.entity_axis_regime_observability.device != belief.device
            or self.entity_axis_regime_observability.dtype != belief.dtype
            or self.entity_axis_regime_predictive_variance.device != belief.device
            or self.entity_axis_regime_predictive_variance.dtype != belief.dtype
            or self.entity_axis_regime_position_residual.device != belief.device
            or self.entity_axis_regime_position_residual.dtype != belief.dtype
            or self.entity_axis_regime_position_residual_supported.device != belief.device
            or self.entity_object_ids.device != belief.device
            or self.entity_active.device != belief.device
        ):
            raise ValueError("entity hypothesis state must share belief execution device")
        changed_entity = (self.entity_object_ids != belief.objects.object_id) | (
            self.entity_active != belief.objects.active
        )
        if bool(changed_entity.any()):
            self.entity_axis_log_weights = torch.where(
                changed_entity.unsqueeze(-1).unsqueeze(-1),
                torch.zeros_like(self.entity_axis_log_weights),
                self.entity_axis_log_weights,
            )
            self.entity_axis_evidence_seen = torch.where(
                changed_entity.unsqueeze(-1),
                torch.zeros_like(self.entity_axis_evidence_seen),
                self.entity_axis_evidence_seen,
            )
            regime_reset_mask = changed_entity.unsqueeze(-1).unsqueeze(-1)
            self.entity_axis_regime_log_weights = torch.where(
                regime_reset_mask.unsqueeze(-1),
                torch.zeros_like(self.entity_axis_regime_log_weights),
                self.entity_axis_regime_log_weights,
            )
            self.entity_axis_regime_support_count = torch.where(
                regime_reset_mask,
                torch.zeros_like(self.entity_axis_regime_support_count),
                self.entity_axis_regime_support_count,
            )
            self.entity_axis_regime_last_timestamp = torch.where(
                regime_reset_mask,
                torch.zeros_like(self.entity_axis_regime_last_timestamp),
                self.entity_axis_regime_last_timestamp,
            )
            self.entity_axis_regime_observability = torch.where(
                regime_reset_mask,
                torch.zeros_like(self.entity_axis_regime_observability),
                self.entity_axis_regime_observability,
            )
            self.entity_axis_regime_predictive_variance = torch.where(
                regime_reset_mask.unsqueeze(-1),
                torch.zeros_like(self.entity_axis_regime_predictive_variance),
                self.entity_axis_regime_predictive_variance,
            ).detach()
            self.entity_axis_regime_position_residual = torch.where(
                regime_reset_mask,
                torch.zeros_like(self.entity_axis_regime_position_residual),
                self.entity_axis_regime_position_residual,
            ).detach()
            self.entity_axis_regime_position_residual_supported = torch.where(
                regime_reset_mask,
                torch.zeros_like(self.entity_axis_regime_position_residual_supported),
                self.entity_axis_regime_position_residual_supported,
            ).detach()
            self.entity_object_ids = belief.objects.object_id.detach().clone()
            self.entity_active = belief.objects.active.detach().clone()
        return self.log_weights

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
    ) -> list[BeliefTrajectory]:
        self._ensure_weights(belief)
        return self.rollout_engine.rollout_dynamics(self.dynamics_models, belief, query_times)

    @staticmethod
    def _sample_noise(reference: Tensor, generator: torch.Generator | None) -> Tensor:
        if generator is None:
            return torch.randn_like(reference)
        if reference.device.type == "cpu":
            return torch.randn(
                reference.shape,
                device=reference.device,
                dtype=reference.dtype,
                generator=generator,
            )
        return torch.randn(
            reference.shape,
            device="cpu",
            dtype=reference.dtype,
            generator=generator,
        ).to(reference.device)

    def rollout_ensemble(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        *,
        sample_count: int,
        position_std_scale: float = 0.0,
        velocity_std_scale: float = 0.0,
        max_std: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> list[list[BeliefTrajectory]]:
        """Roll out deterministic uncertainty-scaled nearby belief samples.

        Sample zero is always the exact persistent belief.  Later samples are
        Gaussian local alternatives using only the belief's explicit fast
        state uncertainty; inactive slots are never perturbed.  The method is
        evaluation/planning-only and does not mutate the source belief or pool
        evidence.
        """

        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        for name, value in (
            ("position_std_scale", position_std_scale),
            ("velocity_std_scale", velocity_std_scale),
            ("max_std", max_std),
        ):
            if value < 0 or not torch.isfinite(torch.as_tensor(value)):
                raise ValueError(f"{name} must be finite and nonnegative")
        self._ensure_weights(belief)
        samples: list[list[BeliefTrajectory]] = []
        active = belief.objects.active.unsqueeze(-1)
        position_std = belief.objects.fast_log_variance[..., :3].clamp(-20.0, 10.0).mul(0.5).exp()
        velocity_std = belief.objects.fast_log_variance[..., 3:6].clamp(-20.0, 10.0).mul(0.5).exp()
        for sample_index in range(sample_count):
            if sample_index == 0 or (position_std_scale == 0 and velocity_std_scale == 0):
                sample_belief = belief
            else:
                objects = belief.objects.clone()
                position_delta = self._sample_noise(objects.position, generator) * (
                    (position_std * float(position_std_scale)).clamp_max(float(max_std))
                )
                velocity_delta = self._sample_noise(objects.velocity, generator) * (
                    (velocity_std * float(velocity_std_scale)).clamp_max(float(max_std))
                )
                objects = objects.replace(
                    position=objects.position
                    + torch.where(active, position_delta, torch.zeros_like(position_delta)),
                    velocity=objects.velocity
                    + torch.where(active, velocity_delta, torch.zeros_like(velocity_delta)),
                )
                sample_belief = belief.replace(objects=objects)
            samples.append(self.rollout(sample_belief, query_times))
        return samples

    def assimilate(
        self,
        belief: WorldBelief,
        target_positions: Tensor,
        target_mask: Tensor,
        *,
        trajectories: Sequence[BeliefTrajectory] | None = None,
        target_position_log_variance: Tensor | None = None,
        target_velocities: Tensor | None = None,
        target_velocity_axis_mask: Tensor | None = None,
        target_velocity_log_variance: Tensor | None = None,
        target_collision: Tensor | None = None,
        position_weight: float = 1.0,
        velocity_weight: float = 0.0,
        velocity_nonregression_gate_enabled: bool = False,
        lifecycle_weight: float = 0.0,
        event_weight: float = 0.0,
        position_gate_ratio: float = 0.0,
        axis_gate_ratio: float = 0.0,
        event_gate_ratio: float = 0.0,
        axis_weights: Sequence[float] | Tensor | None = None,
        uncertainty_aware: bool = True,
        evidence_decay_override: float | None = None,
        axis_prior_strength: float = 0.0,
        entity_regime: Tensor | None = None,
        evidence_timestamp: Tensor | None = None,
        entity_axis_observability: Tensor | None = None,
        entity_axis_predictive_variance: Tensor | None = None,
        robust_influence_delta: float = 0.0,
        learned_position_residual: Tensor | None = None,
    ) -> HypothesisSelection:
        prior = self._ensure_weights(belief)
        if trajectories is None:
            raise ValueError("assimilate requires trajectories for explicit asynchronous evidence")
        selection = self.rollout_engine.score(
            trajectories,
            target_positions,
            target_mask,
            target_position_log_variance=target_position_log_variance,
            target_velocities=target_velocities,
            target_velocity_axis_mask=target_velocity_axis_mask,
            target_velocity_log_variance=target_velocity_log_variance,
            target_collision=target_collision,
            position_weight=position_weight,
            velocity_weight=velocity_weight,
            velocity_nonregression_gate_enabled=velocity_nonregression_gate_enabled,
            lifecycle_weight=lifecycle_weight,
            event_weight=event_weight,
            position_gate_ratio=position_gate_ratio,
            axis_gate_ratio=axis_gate_ratio,
            event_gate_ratio=event_gate_ratio,
            axis_weights=axis_weights,
            uncertainty_aware=uncertainty_aware,
            temperature=self.temperature,
        )
        updated = self._update_evidence(
            prior,
            selection,
            evidence_decay_override=evidence_decay_override,
            axis_prior_strength=axis_prior_strength,
        )
        if entity_regime is not None:
            if entity_axis_predictive_variance is None:
                predictive_variance = torch.stack(
                    [
                        trajectory.fast_log_variance[..., :3].clamp(-20.0, 10.0).exp()
                        for trajectory in trajectories
                    ],
                    dim=-1,
                )
                predictive_mask = target_mask.unsqueeze(-1).unsqueeze(-1)
                predictive_count = target_mask.sum(dim=1).clamp_min(1).to(target_positions.dtype)
                entity_axis_predictive_variance = (predictive_variance * predictive_mask).sum(
                    dim=1
                ) / predictive_count.unsqueeze(-1).unsqueeze(-1)
            self._update_regime_evidence(
                belief,
                selection,
                entity_regime=entity_regime,
                evidence_timestamp=evidence_timestamp,
                entity_axis_observability=entity_axis_observability,
                entity_axis_predictive_variance=entity_axis_predictive_variance,
                evidence_decay_override=evidence_decay_override,
                axis_prior_strength=axis_prior_strength,
                robust_influence_delta=robust_influence_delta,
                learned_position_residual=learned_position_residual,
            )
        return updated

    def assimilate_ensemble(
        self,
        belief: WorldBelief,
        target_positions: Tensor,
        target_mask: Tensor,
        *,
        trajectory_samples: Sequence[Sequence[BeliefTrajectory]],
        risk_penalty: float = 0.0,
        target_position_log_variance: Tensor | None = None,
        target_collision: Tensor | None = None,
        position_weight: float = 1.0,
        lifecycle_weight: float = 0.0,
        event_weight: float = 0.0,
        position_gate_ratio: float = 0.0,
        axis_gate_ratio: float = 0.0,
        event_gate_ratio: float = 0.0,
        axis_weights: Sequence[float] | Tensor | None = None,
        uncertainty_aware: bool = True,
        evidence_decay_override: float | None = None,
        axis_prior_strength: float = 0.0,
    ) -> HypothesisSelection:
        """Assimilate robust delayed evidence from nearby imagined worlds."""

        prior = self._ensure_weights(belief)
        selection = self.rollout_engine.score_ensemble(
            trajectory_samples,
            target_positions,
            target_mask,
            risk_penalty=risk_penalty,
            temperature=self.temperature,
            target_position_log_variance=target_position_log_variance,
            target_collision=target_collision,
            position_weight=position_weight,
            lifecycle_weight=lifecycle_weight,
            event_weight=event_weight,
            position_gate_ratio=position_gate_ratio,
            axis_gate_ratio=axis_gate_ratio,
            event_gate_ratio=event_gate_ratio,
            axis_weights=axis_weights,
            uncertainty_aware=uncertainty_aware,
        )
        return self._update_evidence(
            prior,
            selection,
            evidence_decay_override=evidence_decay_override,
            axis_prior_strength=axis_prior_strength,
        )

    def _update_evidence(
        self,
        prior: Tensor,
        selection: HypothesisSelection,
        *,
        evidence_decay_override: float | None,
        axis_prior_strength: float,
    ) -> HypothesisSelection:
        decay = (
            self.evidence_decay
            if evidence_decay_override is None
            else float(evidence_decay_override)
        )
        if not 0.0 < decay <= 1.0 or not torch.isfinite(torch.as_tensor(decay)):
            raise ValueError("evidence_decay_override must lie in (0,1]")
        if not 0.0 <= axis_prior_strength <= 1.0 or not torch.isfinite(
            torch.as_tensor(axis_prior_strength)
        ):
            raise ValueError("axis_prior_strength must lie in [0,1]")
        evidence_mask = (
            selection.evidence_mask
            if selection.evidence_mask is not None
            else torch.ones(prior.shape[0], device=prior.device, dtype=torch.bool)
        )
        proposed_log_weights = decay * prior - selection.scores / self.temperature
        proposed_log_weights = proposed_log_weights - torch.logsumexp(
            proposed_log_weights, dim=-1, keepdim=True
        )
        posterior_log_weights = torch.where(
            evidence_mask.unsqueeze(-1),
            proposed_log_weights,
            prior,
        )
        posterior = torch.softmax(posterior_log_weights, dim=-1)
        assert self.evidence_seen is not None
        self.evidence_seen = self.evidence_seen | evidence_mask
        posterior_selected_index = posterior.argmax(dim=-1).to(torch.int64)
        posterior_selected_index = torch.where(
            self.evidence_seen,
            posterior_selected_index,
            torch.zeros_like(posterior_selected_index),
        )
        axis_posterior = None
        if selection.axis_scores is not None:
            assert self.axis_log_weights is not None
            assert self.axis_evidence_seen is not None
            axis_evidence_mask = (
                selection.axis_evidence_mask
                if selection.axis_evidence_mask is not None
                else torch.ones(
                    selection.axis_scores.shape[:2],
                    device=prior.device,
                    dtype=torch.bool,
                )
            )
            axis_logits = decay * self.axis_log_weights - selection.axis_scores / self.temperature
            if axis_prior_strength:
                axis_logits = axis_logits + axis_prior_strength * prior.unsqueeze(1)
            axis_logits = axis_logits - torch.logsumexp(axis_logits, dim=-1, keepdim=True)
            self.axis_log_weights = torch.where(
                axis_evidence_mask.unsqueeze(-1),
                axis_logits,
                self.axis_log_weights,
            ).detach()
            self.axis_evidence_seen = self.axis_evidence_seen | axis_evidence_mask
            axis_posterior = torch.softmax(self.axis_log_weights, dim=-1)
        entity_axis_posterior = None
        if selection.entity_axis_scores is not None:
            assert self.entity_axis_log_weights is not None
            assert self.entity_axis_evidence_seen is not None
            entity_axis_evidence_mask = (
                selection.entity_axis_evidence_mask
                if selection.entity_axis_evidence_mask is not None
                else torch.ones(
                    selection.entity_axis_scores.shape[:3],
                    device=prior.device,
                    dtype=torch.bool,
                )
            )
            entity_axis_logits = (
                decay * self.entity_axis_log_weights
                - selection.entity_axis_scores / self.temperature
            )
            if axis_prior_strength:
                entity_axis_logits = entity_axis_logits + axis_prior_strength * prior.unsqueeze(
                    1
                ).unsqueeze(1)
            entity_axis_logits = entity_axis_logits - torch.logsumexp(
                entity_axis_logits,
                dim=-1,
                keepdim=True,
            )
            self.entity_axis_log_weights = torch.where(
                entity_axis_evidence_mask.unsqueeze(-1),
                entity_axis_logits,
                self.entity_axis_log_weights,
            ).detach()
            self.entity_axis_evidence_seen = (
                self.entity_axis_evidence_seen | entity_axis_evidence_mask
            )
            entity_axis_posterior = torch.softmax(
                self.entity_axis_log_weights,
                dim=-1,
            )
        self.log_weights = posterior_log_weights.detach()
        self.last_selection = HypothesisSelection(
            selection.scores,
            posterior_selected_index,
            posterior,
            axis_scores=selection.axis_scores,
            axis_posterior=axis_posterior,
            entity_axis_scores=selection.entity_axis_scores,
            entity_axis_posterior=entity_axis_posterior,
            score_spread=selection.score_spread,
            axis_score_spread=selection.axis_score_spread,
            evidence_mask=selection.evidence_mask,
            axis_evidence_mask=selection.axis_evidence_mask,
            entity_axis_evidence_mask=selection.entity_axis_evidence_mask,
            sample_count=selection.sample_count,
        ).validate()
        return self.last_selection

    def _update_regime_evidence(
        self,
        belief: WorldBelief,
        selection: HypothesisSelection,
        *,
        entity_regime: Tensor,
        evidence_timestamp: Tensor | None,
        entity_axis_observability: Tensor | None,
        entity_axis_predictive_variance: Tensor | None,
        evidence_decay_override: float | None,
        axis_prior_strength: float,
        robust_influence_delta: float,
        learned_position_residual: Tensor | None,
    ) -> None:
        """Update only the exact entity/axis/regime cells with RGB evidence."""

        self._ensure_weights(belief)
        if selection.entity_axis_scores is None:
            raise ValueError("regime evidence requires entity-axis scores")
        expected_regime_shape = belief.objects.active.shape
        if entity_regime.shape != expected_regime_shape or entity_regime.dtype is not torch.int64:
            raise ValueError("entity_regime must use int64 with shape [B,N]")
        if torch.any(entity_regime < 0) or torch.any(entity_regime >= NUM_HYPOTHESIS_REGIMES):
            raise ValueError("entity_regime contains an out-of-range value")
        if evidence_timestamp is None or evidence_timestamp.shape != belief.timestamp.shape:
            raise ValueError("regime evidence requires timestamp shape [B]")
        if not torch.isfinite(evidence_timestamp).all():
            raise ValueError("regime evidence timestamp must be finite")
        expected_observability_shape = (*expected_regime_shape, 3)
        if (
            entity_axis_observability is None
            or entity_axis_observability.shape != expected_observability_shape
        ):
            raise ValueError("regime evidence observability must have shape [B,N,3]")
        if learned_position_residual is None:
            learned_position_residual = belief.objects.position.new_zeros(
                expected_observability_shape
            )
        if (
            learned_position_residual.shape != expected_observability_shape
            or not torch.isfinite(learned_position_residual).all()
        ):
            raise ValueError("learned position residual must be finite with shape [B,N,3]")
        expected_predictive_shape = (*expected_regime_shape, 3, len(self.dynamics_models))
        if (
            entity_axis_predictive_variance is None
            or entity_axis_predictive_variance.shape != expected_predictive_shape
        ):
            raise ValueError("regime evidence predictive variance must have shape [B,N,3,H]")
        if not torch.isfinite(entity_axis_predictive_variance).all() or torch.any(
            entity_axis_predictive_variance < 0
        ):
            raise ValueError("regime evidence predictive variance must be finite nonnegative")
        if (
            not torch.isfinite(entity_axis_observability).all()
            or torch.any(entity_axis_observability < 0)
            or torch.any(entity_axis_observability > 1)
        ):
            raise ValueError("regime evidence observability must lie in [0,1]")
        if (
            isinstance(robust_influence_delta, bool)
            or not isinstance(robust_influence_delta, Real)
            or robust_influence_delta < 0
            or not torch.isfinite(torch.as_tensor(robust_influence_delta))
        ):
            raise ValueError("robust_influence_delta must be finite and nonnegative")
        decay = (
            self.evidence_decay
            if evidence_decay_override is None
            else float(evidence_decay_override)
        )
        if not 0.0 < decay <= 1.0 or not torch.isfinite(torch.as_tensor(decay)):
            raise ValueError("evidence_decay_override must lie in (0,1]")
        if not 0.0 <= axis_prior_strength <= 1.0:
            raise ValueError("axis_prior_strength must lie in [0,1]")
        assert self.entity_axis_regime_log_weights is not None
        assert self.entity_axis_regime_support_count is not None
        assert self.entity_axis_regime_last_timestamp is not None
        assert self.entity_axis_regime_observability is not None
        assert self.entity_axis_regime_predictive_variance is not None
        assert self.entity_axis_regime_position_residual is not None
        assert self.entity_axis_regime_position_residual_supported is not None
        assert self.log_weights is not None
        evidence_mask = selection.entity_axis_evidence_mask
        if evidence_mask is None:
            evidence_mask = torch.ones(
                selection.entity_axis_scores.shape[:3],
                device=belief.device,
                dtype=torch.bool,
            )
        regime_one_hot = torch.nn.functional.one_hot(
            entity_regime,
            num_classes=NUM_HYPOTHESIS_REGIMES,
        ).to(torch.bool)
        cell_mask = evidence_mask.unsqueeze(-1) & regime_one_hot.unsqueeze(2)
        gather_index = entity_regime[:, :, None, None, None].expand(
            -1,
            -1,
            3,
            1,
            len(self.dynamics_models),
        )
        prior_cell = torch.gather(
            self.entity_axis_regime_log_weights,
            dim=3,
            index=gather_index,
        ).squeeze(3)
        relative_scores = selection.entity_axis_scores - selection.entity_axis_scores[..., :1]
        if robust_influence_delta:
            relative_scores = relative_scores.clamp(
                min=-float(robust_influence_delta),
                max=float(robust_influence_delta),
            )
        proposed = decay * prior_cell - relative_scores / self.temperature
        if axis_prior_strength:
            proposed = proposed + axis_prior_strength * self.log_weights.unsqueeze(1).unsqueeze(1)
        proposed = proposed - torch.logsumexp(proposed, dim=-1, keepdim=True)
        expanded_proposed = proposed.unsqueeze(3).expand_as(self.entity_axis_regime_log_weights)
        self.entity_axis_regime_log_weights = torch.where(
            cell_mask.unsqueeze(-1),
            expanded_proposed,
            self.entity_axis_regime_log_weights,
        ).detach()

        old_count = self.entity_axis_regime_support_count
        new_count = old_count + cell_mask.to(torch.int64)
        expanded_observability = entity_axis_observability.unsqueeze(-1).expand_as(old_count)
        running_observability = old_count.to(belief.dtype) * self.entity_axis_regime_observability
        running_observability = (
            running_observability + expanded_observability
        ) / new_count.clamp_min(1).to(belief.dtype)
        self.entity_axis_regime_support_count = new_count
        self.entity_axis_regime_observability = torch.where(
            cell_mask,
            running_observability,
            self.entity_axis_regime_observability,
        ).detach()
        expanded_predictive_variance = entity_axis_predictive_variance.unsqueeze(3).expand_as(
            self.entity_axis_regime_predictive_variance
        )
        self.entity_axis_regime_predictive_variance = torch.where(
            cell_mask.unsqueeze(-1),
            expanded_predictive_variance,
            self.entity_axis_regime_predictive_variance,
        ).detach()
        expanded_timestamp = evidence_timestamp[:, None, None, None].expand_as(
            self.entity_axis_regime_last_timestamp
        )
        self.entity_axis_regime_last_timestamp = torch.where(
            cell_mask,
            expanded_timestamp,
            self.entity_axis_regime_last_timestamp,
        ).detach()
        expanded_residual = learned_position_residual.unsqueeze(-1).expand_as(
            self.entity_axis_regime_position_residual
        )
        consistent_residual = (old_count > 0) & (
            self.entity_axis_regime_position_residual * expanded_residual > 0
        )
        self.entity_axis_regime_position_residual_supported = torch.where(
            cell_mask,
            consistent_residual,
            self.entity_axis_regime_position_residual_supported,
        ).detach()
        self.entity_axis_regime_position_residual = torch.where(
            cell_mask,
            expanded_residual,
            self.entity_axis_regime_position_residual,
        ).detach()

    def selected_entity_axis_applicability(
        self,
        belief: WorldBelief,
        *,
        entity_regime: Tensor,
        current_timestamp: Tensor,
        minimum_support_count: int,
        maximum_age_seconds: float,
        minimum_observability: float,
        minimum_confidence_margin: float,
        candidate_regime_mask: Tensor | None = None,
        candidate_entity_axis_support: Tensor | None = None,
    ) -> HypothesisApplicability:
        """Resolve local choices, falling back to candidate zero when unsupported."""

        self._ensure_weights(belief)
        if (
            not isinstance(minimum_support_count, int)
            or isinstance(minimum_support_count, bool)
            or minimum_support_count <= 0
        ):
            raise ValueError("minimum_support_count must be a positive integer")
        for name, value in (
            ("maximum_age_seconds", maximum_age_seconds),
            ("minimum_observability", minimum_observability),
            ("minimum_confidence_margin", minimum_confidence_margin),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not torch.isfinite(torch.as_tensor(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if minimum_observability > 1 or minimum_confidence_margin > 1:
            raise ValueError("observability and confidence thresholds must lie in [0,1]")
        if (
            entity_regime.shape != belief.objects.active.shape
            or entity_regime.dtype is not torch.int64
        ):
            raise ValueError("entity_regime must use int64 with shape [B,N]")
        if (
            current_timestamp.shape != belief.timestamp.shape
            or not torch.isfinite(current_timestamp).all()
        ):
            raise ValueError("current_timestamp must be finite with shape [B]")
        if torch.any(entity_regime < 0) or torch.any(entity_regime >= NUM_HYPOTHESIS_REGIMES):
            raise ValueError("entity_regime contains an out-of-range value")
        assert self.entity_axis_regime_log_weights is not None
        assert self.entity_axis_regime_support_count is not None
        assert self.entity_axis_regime_last_timestamp is not None
        assert self.entity_axis_regime_observability is not None
        assert self.entity_axis_regime_predictive_variance is not None
        assert self.entity_axis_regime_position_residual is not None
        assert self.entity_axis_regime_position_residual_supported is not None
        candidate_count = len(self.dynamics_models)
        gather_index = entity_regime[:, :, None, None, None].expand(
            -1,
            -1,
            3,
            1,
            candidate_count,
        )
        log_weights = torch.gather(
            self.entity_axis_regime_log_weights,
            dim=3,
            index=gather_index,
        ).squeeze(3)
        if candidate_regime_mask is not None:
            if candidate_regime_mask.shape != (NUM_HYPOTHESIS_REGIMES, candidate_count):
                raise ValueError("candidate_regime_mask must have shape [R,H]")
            if candidate_regime_mask.dtype is not torch.bool:
                raise TypeError("candidate_regime_mask must be boolean")
            applicable = candidate_regime_mask[entity_regime].unsqueeze(2).expand_as(log_weights)
            if not torch.all(applicable[..., 0]):
                raise ValueError("learned fallback candidate zero must support every regime")
            log_weights = log_weights.masked_fill(~applicable, torch.finfo(log_weights.dtype).min)
        if candidate_entity_axis_support is not None:
            if candidate_entity_axis_support.shape != log_weights.shape:
                raise ValueError("candidate entity-axis support must have shape [B,N,3,H]")
            if candidate_entity_axis_support.dtype is not torch.bool:
                raise TypeError("candidate entity-axis support must be boolean")
            if not torch.all(candidate_entity_axis_support[..., 0]):
                raise ValueError("learned fallback candidate zero must support every entity-axis")
            log_weights = log_weights.masked_fill(
                ~candidate_entity_axis_support,
                torch.finfo(log_weights.dtype).min,
            )
        posterior = torch.softmax(log_weights, dim=-1)
        selected = posterior.argmax(dim=-1).to(torch.int64)
        if candidate_count == 1:
            margin = torch.ones_like(posterior[..., 0])
        else:
            top_two = posterior.topk(k=2, dim=-1).values
            margin = top_two[..., 0] - top_two[..., 1]
        scalar_index = entity_regime[:, :, None, None].expand(-1, -1, 3, 1)
        support_count = torch.gather(
            self.entity_axis_regime_support_count,
            dim=3,
            index=scalar_index,
        ).squeeze(3)
        last_timestamp = torch.gather(
            self.entity_axis_regime_last_timestamp,
            dim=3,
            index=scalar_index,
        ).squeeze(3)
        observability = torch.gather(
            self.entity_axis_regime_observability,
            dim=3,
            index=scalar_index,
        ).squeeze(3)
        candidate_predictive_variance = torch.gather(
            self.entity_axis_regime_predictive_variance,
            dim=3,
            index=gather_index,
        ).squeeze(3)
        predictive_variance = torch.gather(
            candidate_predictive_variance,
            dim=-1,
            index=selected.unsqueeze(-1),
        ).squeeze(-1)
        position_residual = torch.gather(
            self.entity_axis_regime_position_residual,
            dim=3,
            index=scalar_index,
        ).squeeze(3)
        position_residual_supported = torch.gather(
            self.entity_axis_regime_position_residual_supported,
            dim=3,
            index=scalar_index,
        ).squeeze(3)
        age = (current_timestamp[:, None, None] - last_timestamp).clamp_min(0)
        supported = (
            belief.objects.active.unsqueeze(-1)
            & (support_count >= minimum_support_count)
            & (age <= maximum_age_seconds)
            & (observability >= minimum_observability)
            & (margin >= minimum_confidence_margin)
        )
        selected = torch.where(supported, selected, torch.zeros_like(selected))
        return HypothesisApplicability(
            selected_index=selected,
            supported=supported,
            support_count=support_count,
            age_seconds=age,
            observability=observability,
            predictive_variance=predictive_variance,
            confidence_margin=margin,
            regime=entity_regime,
            position_residual=position_residual,
            position_residual_supported=position_residual_supported,
        ).validate(candidate_count=candidate_count)

    def selected_index(self, belief: WorldBelief) -> Tensor:
        weights = self._ensure_weights(belief)
        return weights.argmax(dim=-1).to(torch.int64)

    def selected_axis_index(self, belief: WorldBelief) -> Tensor:
        """Return per-axis choices from the latest asynchronous evidence."""

        self._ensure_weights(belief)
        if self.last_selection is None or self.last_selection.axis_selected_index is None:
            raise RuntimeError("axis selection is unavailable before assimilation")
        assert self.axis_log_weights is not None
        assert self.axis_evidence_seen is not None
        selected = self.axis_log_weights.argmax(dim=-1).to(torch.int64)
        return torch.where(
            self.axis_evidence_seen,
            selected,
            torch.zeros_like(selected),
        )

    def selected_entity_axis_index(self, belief: WorldBelief) -> Tensor:
        """Return candidate choices independently for each persistent entity."""

        self._ensure_weights(belief)
        if self.last_selection is None:
            raise RuntimeError("entity-axis selection is unavailable before assimilation")
        assert self.entity_axis_log_weights is not None
        assert self.entity_axis_evidence_seen is not None
        selected = self.entity_axis_log_weights.argmax(dim=-1).to(torch.int64)
        return torch.where(
            self.entity_axis_evidence_seen,
            selected,
            torch.zeros_like(selected),
        )


@dataclass(frozen=True)
class PendingHypothesisEvidence:
    """One RGB-only forecast awaiting an observation at its due timestamp."""

    horizon_index: int
    due_timestamp: Tensor
    source: WorldBelief
    source_object_ids: Tensor
    trajectories: tuple[BeliefTrajectory, ...]
    learned_step: RolloutStep
    entity_regime: Tensor | None = None
    source_revision: int | None = None
    source_tensor_signature: object | None = None
    dynamics_tensor_signature: object | None = None
    dynamics_training: bool | None = None
    learned_result_tensor_signature: object | None = None


class RuntimeHypothesisController:
    """Causal adapter from associated RGB positions to a candidate pool.

    This controller is deliberately runtime-local rather than belief state. It
    remembers only pending imagined rollouts and their evidence weights.  At a
    later packet it uses *associated RGB world-position measurements* to score
    a due rollout, then future calls can compose only explicitly configured
    coordinate axes.  Simulator state, ground-truth IDs, and posterior object
    positions are never used as selector targets.
    """

    def __init__(
        self,
        pool: HypothesisDynamicsPool,
        *,
        evidence_horizons_seconds: Sequence[float],
        axis_independent_axes: Sequence[int],
        axis_prior_strength: float = 0.0,
        timestamp_tolerance_seconds: float = 1.0e-5,
        local_applicability_enabled: bool = False,
        minimum_support_count: int = 1,
        maximum_evidence_age_seconds: float = 1.0,
        minimum_observability: float = 0.0,
        minimum_confidence_margin: float = 0.0,
        velocity_evidence_weight: float = 0.0,
        velocity_nonregression_gate_enabled: bool = False,
        residual_correction_gain_by_axis: Sequence[float] = (0.0, 0.0, 0.0),
        robust_influence_delta: float = 0.0,
        composition_step_seconds: float | None = None,
        shared_horizon_rollout_enabled: bool = False,
    ) -> None:
        if not evidence_horizons_seconds or any(
            horizon <= 0 or not torch.isfinite(torch.as_tensor(horizon))
            for horizon in evidence_horizons_seconds
        ):
            raise ValueError("evidence horizons must be finite and positive")
        if any(axis not in (0, 1, 2) for axis in axis_independent_axes):
            raise ValueError("axis_independent_axes must contain only 0, 1, or 2")
        if not 0.0 <= axis_prior_strength <= 1.0:
            raise ValueError("axis_prior_strength must lie in [0,1]")
        if timestamp_tolerance_seconds < 0 or not torch.isfinite(
            torch.as_tensor(timestamp_tolerance_seconds)
        ):
            raise ValueError("timestamp_tolerance_seconds must be finite and nonnegative")
        if not isinstance(local_applicability_enabled, bool):
            raise ValueError("local_applicability_enabled must be boolean")
        if not isinstance(velocity_nonregression_gate_enabled, bool):
            raise ValueError("velocity_nonregression_gate_enabled must be boolean")
        if not isinstance(shared_horizon_rollout_enabled, bool):
            raise ValueError("shared_horizon_rollout_enabled must be boolean")
        if len(residual_correction_gain_by_axis) != 3:
            raise ValueError("residual_correction_gain_by_axis must contain exactly three values")
        for value in residual_correction_gain_by_axis:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not torch.isfinite(torch.as_tensor(value))
                or value < 0
                or value > 1
            ):
                raise ValueError("residual correction gains must lie in [0,1]")
        if (
            not isinstance(minimum_support_count, int)
            or isinstance(minimum_support_count, bool)
            or minimum_support_count <= 0
        ):
            raise ValueError("minimum_support_count must be a positive integer")
        for name, value in (
            ("maximum_evidence_age_seconds", maximum_evidence_age_seconds),
            ("minimum_observability", minimum_observability),
            ("minimum_confidence_margin", minimum_confidence_margin),
            ("velocity_evidence_weight", velocity_evidence_weight),
            ("robust_influence_delta", robust_influence_delta),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or value < 0
                or not torch.isfinite(torch.as_tensor(value))
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if minimum_observability > 1 or minimum_confidence_margin > 1:
            raise ValueError("observability and confidence thresholds must lie in [0,1]")
        residual_axes = {
            axis for axis, value in enumerate(residual_correction_gain_by_axis) if value > 0
        }
        if residual_axes and not local_applicability_enabled:
            raise ValueError("residual correction requires local applicability")
        if not residual_axes.issubset(set(axis_independent_axes)):
            raise ValueError("residual correction axes must be independently configured")
        if composition_step_seconds is not None:
            if (
                isinstance(composition_step_seconds, bool)
                or not isinstance(composition_step_seconds, Real)
                or composition_step_seconds <= 0
                or not torch.isfinite(torch.as_tensor(composition_step_seconds))
            ):
                raise ValueError("composition_step_seconds must be null or finite and positive")
            if not local_applicability_enabled:
                raise ValueError("short-step composition requires local applicability")
        self.evidence_horizons_seconds = tuple(
            sorted(set(float(value) for value in evidence_horizons_seconds))
        )
        # Model accuracy is horizon-dependent.  Each evidence horizon owns an
        # independent posterior over the same replaceable candidate models;
        # otherwise a model that wins at one short endpoint can silently take
        # over an unsupported long rollout.  Keep ``pool`` as the first member
        # for backwards-compatible diagnostics and single-horizon callers.
        self.pools = tuple(
            pool
            if horizon_index == 0
            else HypothesisDynamicsPool(
                pool.dynamics_models,
                rollout_engine=pool.rollout_engine,
                temperature=pool.temperature,
                evidence_decay=pool.evidence_decay,
            )
            for horizon_index, _ in enumerate(self.evidence_horizons_seconds)
        )
        self.pool = self.pools[0]
        self.axis_independent_axes = tuple(sorted(set(int(axis) for axis in axis_independent_axes)))
        self.axis_prior_strength = float(axis_prior_strength)
        self.timestamp_tolerance_seconds = float(timestamp_tolerance_seconds)
        self.local_applicability_enabled = local_applicability_enabled
        self.minimum_support_count = int(minimum_support_count)
        self.maximum_evidence_age_seconds = float(maximum_evidence_age_seconds)
        self.minimum_observability = float(minimum_observability)
        self.minimum_confidence_margin = float(minimum_confidence_margin)
        self.velocity_evidence_weight = float(velocity_evidence_weight)
        self.velocity_nonregression_gate_enabled = velocity_nonregression_gate_enabled
        self.residual_correction_gain_by_axis = tuple(
            float(value) for value in residual_correction_gain_by_axis
        )
        self.robust_influence_delta = float(robust_influence_delta)
        self.composition_step_seconds = (
            None if composition_step_seconds is None else float(composition_step_seconds)
        )
        self.shared_horizon_rollout_enabled = shared_horizon_rollout_enabled
        self.composition_horizon_index: int | None = None
        if self.composition_step_seconds is not None:
            matching = [
                index
                for index, horizon in enumerate(self.evidence_horizons_seconds)
                if abs(horizon - self.composition_step_seconds) <= self.timestamp_tolerance_seconds
            ]
            if len(matching) != 1:
                raise ValueError(
                    "composition step must match exactly one configured evidence horizon"
                )
            self.composition_horizon_index = matching[0]
        self.candidate_regime_mask = self._candidate_regime_mask(pool)
        self.pending: list[PendingHypothesisEvidence] = []
        self.runtime_dynamics_signature: object | None = None
        self.runtime_dynamics_training: bool | None = None
        self.pending_invalidation_counts: dict[str, int] = {}

    @staticmethod
    def _candidate_regime_mask(pool: HypothesisDynamicsPool) -> Tensor:
        mask = torch.zeros(
            NUM_HYPOTHESIS_REGIMES,
            len(pool.dynamics_models),
            dtype=torch.bool,
        )
        mask[:, 0] = True
        for candidate_index, model in enumerate(pool.dynamics_models[1:], start=1):
            regimes = getattr(model, "supported_hypothesis_regimes", (HypothesisRegime.FREE,))
            for regime in regimes:
                regime_index = int(regime)
                if regime_index < 0 or regime_index >= NUM_HYPOTHESIS_REGIMES:
                    raise ValueError("candidate exposes an out-of-range hypothesis regime")
                mask[regime_index, candidate_index] = True
        return mask

    @staticmethod
    def _candidate_entity_axis_support(
        pool: HypothesisDynamicsPool,
        belief: WorldBelief,
    ) -> Tensor:
        values: list[Tensor] = []
        expected = (*belief.objects.active.shape, 3)
        for candidate_index, model in enumerate(pool.dynamics_models):
            accessor = getattr(model, "applicability_mask", None)
            if accessor is None:
                support = torch.ones(expected, device=belief.device, dtype=torch.bool)
            else:
                support = accessor(belief)
                if not isinstance(support, Tensor) or support.shape != expected:
                    raise ValueError("candidate applicability_mask must return boolean [B,N,3]")
                if support.dtype is not torch.bool:
                    raise TypeError("candidate applicability_mask must return torch.bool")
                support = support.to(device=belief.device)
            if candidate_index == 0 and not bool(support.all()):
                raise ValueError("learned fallback candidate zero must support every entity-axis")
            values.append(support)
        return torch.stack(values, dim=-1)

    @staticmethod
    def _trajectory_regime(
        source: WorldBelief,
        trajectory: BeliefTrajectory,
    ) -> Tensor:
        """Classify each learned interval without candidate or simulator truth."""

        shape = trajectory.active_mask.shape
        regime = torch.full(
            shape,
            int(HypothesisRegime.FREE),
            device=trajectory.active_mask.device,
            dtype=torch.int64,
        )
        source_mode = source.objects.motion_mode_logits.argmax(dim=-1)
        source_mode = source_mode.unsqueeze(1).expand(shape)
        ground_source = (
            (source_mode == int(MotionMode.GROUND_CONTACT))
            | (source_mode == int(MotionMode.ROLLING))
            | (source_mode == int(MotionMode.SLIDING))
            | (source_mode == int(MotionMode.SLEEPING))
        )
        regime = torch.where(
            ground_source,
            torch.full_like(regime, int(HypothesisRegime.GROUND_CONTACT)),
            regime,
        )
        regime = torch.where(
            source_mode == int(MotionMode.PAIR_CONTACT),
            torch.full_like(regime, int(HypothesisRegime.PAIR_CONTACT)),
            regime,
        )
        regime = torch.where(
            source_mode == int(MotionMode.OCCLUDED),
            torch.full_like(regime, int(HypothesisRegime.OCCLUDED)),
            regime,
        )
        regime = torch.where(
            source_mode == int(MotionMode.EXTERNALLY_ACTUATED),
            torch.full_like(regime, int(HypothesisRegime.EXTERNALLY_ACTUATED)),
            regime,
        )

        def node_any(name: str) -> Tensor | None:
            value = trajectory.auxiliary.get(name)
            if not isinstance(value, Tensor) or value.shape[:3] != shape:
                return None
            while value.ndim > 3:
                value = value.any(dim=-1)
            return value.to(torch.bool)

        ground_contact = node_any("interval_ground_contact")
        pair_contact = node_any("interval_pair_contact")
        collision = node_any("pair_collision")
        boundary_collision = node_any("boundary_collision")
        if ground_contact is not None:
            regime = torch.where(
                ground_contact,
                torch.full_like(regime, int(HypothesisRegime.GROUND_CONTACT)),
                regime,
            )
        if pair_contact is not None:
            regime = torch.where(
                pair_contact,
                torch.full_like(regime, int(HypothesisRegime.PAIR_CONTACT)),
                regime,
            )
        predicted_collision = trajectory.event_logits[..., MotionMode.COLLISION] > 0
        if collision is not None:
            predicted_collision = predicted_collision | collision
        if boundary_collision is not None:
            predicted_collision = predicted_collision | boundary_collision
        regime = torch.where(
            predicted_collision,
            torch.full_like(regime, int(HypothesisRegime.COLLISION)),
            regime,
        )
        return torch.where(
            trajectory.active_mask,
            regime,
            torch.full_like(regime, int(HypothesisRegime.OCCLUDED)),
        )

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        for pool in self.pools:
            pool.reset(batch_size, device=device, dtype=dtype)
        seen_models: set[int] = set()
        for model in self.pool.dynamics_models:
            clear_runtime_state = getattr(model, "clear_runtime_state", None)
            if clear_runtime_state is None or id(model) in seen_models:
                continue
            seen_models.add(id(model))
            clear_runtime_state()
        self.pending.clear()
        self.runtime_dynamics_signature = None
        self.runtime_dynamics_training = None
        self.pending_invalidation_counts.clear()

    def _record_invalidations(self, reason: str, count: int) -> None:
        if count <= 0:
            return
        self.pending_invalidation_counts[reason] = (
            self.pending_invalidation_counts.get(reason, 0) + count
        )

    def _reset_evidence(self, belief: WorldBelief) -> None:
        for pool in self.pools:
            pool.reset(
                belief.batch_size,
                device=belief.device,
                dtype=belief.dtype,
            )

    def invalidate_pending(
        self,
        *,
        reason: str = "external_revision",
        reset_evidence: bool = False,
        belief: WorldBelief | None = None,
    ) -> None:
        """Discard forecasts whose source belief was revised out of band."""

        if not reason:
            raise ValueError("pending invalidation reason must be nonempty")
        self._record_invalidations(reason, len(self.pending))
        self.pending.clear()
        if reset_evidence:
            if belief is None:
                raise ValueError("belief is required when resetting hypothesis evidence")
            self._reset_evidence(belief)

    def synchronize_runtime_context(
        self,
        belief: WorldBelief,
        *,
        dynamics_tensor_signature: object,
        dynamics_training: bool,
        tensor_signature: Callable[[object], object],
    ) -> None:
        """Invalidate stale evidence before propagation or observation scoring."""

        revision_changed = (
            self.runtime_dynamics_signature is not None
            and self.runtime_dynamics_signature != dynamics_tensor_signature
        )
        mode_changed = (
            self.runtime_dynamics_training is not None
            and self.runtime_dynamics_training != dynamics_training
        )
        if revision_changed or mode_changed:
            reason = "dynamics_revision" if revision_changed else "dynamics_mode"
            self.invalidate_pending(
                reason=reason,
                reset_evidence=True,
                belief=belief,
            )
        else:
            retained: list[PendingHypothesisEvidence] = []
            for pending in self.pending:
                if (
                    pending.source_tensor_signature is not None
                    and tensor_signature(pending.source) != pending.source_tensor_signature
                ):
                    self._record_invalidations("source_tensor_revision", 1)
                    continue
                if (
                    pending.learned_result_tensor_signature is not None
                    and tensor_signature(pending.learned_step)
                    != pending.learned_result_tensor_signature
                ):
                    self._record_invalidations("scheduled_result_revision", 1)
                    continue
                if (
                    pending.dynamics_tensor_signature is not None
                    and pending.dynamics_tensor_signature != dynamics_tensor_signature
                ):
                    self._record_invalidations("dynamics_revision", 1)
                    continue
                if (
                    pending.dynamics_training is not None
                    and pending.dynamics_training != dynamics_training
                ):
                    self._record_invalidations("dynamics_mode", 1)
                    continue
                retained.append(pending)
            self.pending = retained
        self.runtime_dynamics_signature = dynamics_tensor_signature
        self.runtime_dynamics_training = dynamics_training

    @staticmethod
    def _trajectory_from_step(step: RolloutStep) -> BeliefTrajectory:
        objects = step.belief.objects
        return BeliefTrajectory(
            timestamps=step.belief.timestamp.unsqueeze(1),
            positions=objects.position.unsqueeze(1),
            velocities=objects.velocity.unsqueeze(1),
            orientations=objects.orientation.unsqueeze(1),
            motion_mode_logits=objects.motion_mode_logits.unsqueeze(1),
            fast_log_variance=objects.fast_log_variance.unsqueeze(1),
            active_mask=objects.active.unsqueeze(1),
            event_logits=step.event_logits.unsqueeze(1),
            auxiliary={name: value.unsqueeze(1) for name, value in step.auxiliary.items()},
        ).validate()

    @staticmethod
    def _shared_horizon_safe(model: object) -> bool:
        capability = getattr(model, "shared_horizon_rollout_safe", False)
        if callable(capability):
            capability = capability()
        return capability is True

    @staticmethod
    def _prefix_step_from_segments(
        segments: Sequence[RolloutStep],
        horizon_index: int,
    ) -> RolloutStep:
        """Reconstruct one source-to-horizon step from chronological segments."""

        prefix = segments[: horizon_index + 1]
        endpoint = prefix[-1]
        event_logits = endpoint.event_logits.clone()
        event_logits[..., MotionMode.COLLISION] = torch.stack(
            [item.event_logits[..., MotionMode.COLLISION] for item in prefix],
            dim=1,
        ).amax(dim=1)

        interval_or = {
            "interval_pair_contact",
            "pair_collision",
            "interval_boundary_contact",
            "boundary_collision",
            "interval_ground_contact",
            "ground_collision",
            "collision_event",
        }
        interval_max = {"pair_impulse", "max_penetration"}
        endpoint_collision_logits = {"pair_event_logits", "boundary_event_logits"}
        auxiliary: dict[str, Tensor] = {}
        for name in endpoint.auxiliary:
            values = torch.stack([item.auxiliary[name] for item in prefix], dim=1)
            if name in interval_or:
                auxiliary[name] = values.any(dim=1)
            elif name in interval_max:
                auxiliary[name] = values.amax(dim=1)
            elif name in endpoint_collision_logits:
                endpoint_value = values[:, -1]
                auxiliary[name] = torch.stack(
                    (
                        endpoint_value[..., 0],
                        values[..., 1].amax(dim=1),
                    ),
                    dim=-1,
                )
            elif name == "learned_effect_evaluation_count":
                auxiliary[name] = values.sum(dim=1)
            else:
                auxiliary[name] = values[:, -1]
        return RolloutStep(
            belief=endpoint.belief,
            event_logits=event_logits,
            auxiliary=auxiliary,
        )

    def _scheduled_steps_by_candidate(
        self,
        belief: WorldBelief,
    ) -> tuple[tuple[RolloutStep, ...], ...]:
        models = self.pool.dynamics_models
        scheduled: list[tuple[RolloutStep, ...]] = []
        for model in models:
            if self._shared_horizon_safe(model):
                current = belief.clone()
                previous_horizon = 0.0
                segments: list[RolloutStep] = []
                for horizon in self.evidence_horizons_seconds:
                    delta_time = current.timestamp.new_full(
                        current.timestamp.shape,
                        horizon - previous_horizon,
                    )
                    segment = model.predict_step(current, delta_time)
                    current = segment.belief
                    segments.append(segment)
                    previous_horizon = horizon
                scheduled.append(
                    tuple(
                        self._prefix_step_from_segments(segments, horizon_index)
                        for horizon_index in range(len(self.evidence_horizons_seconds))
                    )
                )
            else:
                scheduled.append(
                    tuple(
                        model.predict_step(
                            belief.clone(),
                            belief.timestamp.new_full(belief.timestamp.shape, horizon),
                        )
                        for horizon in self.evidence_horizons_seconds
                    )
                )
        return tuple(scheduled)

    def schedule(
        self,
        belief: WorldBelief,
        *,
        source_revision: int | None = None,
        source_tensor_signature: object | None = None,
        dynamics_tensor_signature: object | None = None,
        dynamics_training: bool | None = None,
        tensor_signature: Callable[[object], object] | None = None,
    ) -> None:
        """Issue small candidate rollouts after a corrected posterior."""

        for pool in self.pools:
            pool._ensure_weights(belief)
        # A non-RGB asynchronous packet may advance time beyond a pending
        # endpoint. Such a forecast cannot later acquire an exact RGB target,
        # so bound runtime-local memory without inventing interpolation.
        self.pending = [
            pending
            for pending in self.pending
            if bool(
                torch.all(
                    belief.timestamp <= pending.due_timestamp + self.timestamp_tolerance_seconds
                )
            )
        ]
        shared_steps = (
            self._scheduled_steps_by_candidate(belief)
            if self.shared_horizon_rollout_enabled
            else None
        )
        for horizon_index, (horizon, pool) in enumerate(
            zip(self.evidence_horizons_seconds, self.pools, strict=True)
        ):
            learned_step = (
                shared_steps[0][horizon_index]
                if shared_steps is not None
                else pool.dynamics_models[0].predict_step(
                    belief.clone(),
                    belief.timestamp.new_full(belief.timestamp.shape, horizon),
                )
            )
            learned_trajectory = self._trajectory_from_step(learned_step)
            learned_regime = self._trajectory_regime(belief, learned_trajectory)[:, 0]
            alternative_trajectories = (
                [
                    self._trajectory_from_step(candidate_steps[horizon_index])
                    for candidate_steps in shared_steps[1:]
                ]
                if shared_steps is not None
                else (
                    pool.rollout_engine.rollout_dynamics(
                        pool.dynamics_models[1:],
                        belief,
                        [horizon],
                    )
                    if len(pool.dynamics_models) > 1
                    else []
                )
            )
            trajectories = (learned_trajectory, *alternative_trajectories)
            self.pending.append(
                PendingHypothesisEvidence(
                    horizon_index=horizon_index,
                    due_timestamp=belief.timestamp.detach().clone() + horizon,
                    source=belief,
                    source_object_ids=belief.objects.object_id.detach().clone(),
                    trajectories=tuple(trajectories),
                    learned_step=learned_step,
                    entity_regime=learned_regime.detach().clone(),
                    source_revision=source_revision,
                    source_tensor_signature=source_tensor_signature,
                    dynamics_tensor_signature=dynamics_tensor_signature,
                    dynamics_training=dynamics_training,
                    learned_result_tensor_signature=(
                        tensor_signature(learned_step) if tensor_signature is not None else None
                    ),
                )
            )

    @staticmethod
    def _associated_rgb_targets(
        belief: WorldBelief,
        measured: object,
        association: object,
        source_object_ids: Tensor,
    ) -> (
        tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            Tensor | None,
        ]
        | None
    ):
        """Map associated RGB back-projections to persistent candidate slots."""

        auxiliary = getattr(measured, "auxiliary", None)
        positions = auxiliary.get("world_position") if isinstance(auxiliary, dict) else None
        position_log_variance = (
            auxiliary.get("world_position_log_variance") if isinstance(auxiliary, dict) else None
        )
        velocity = auxiliary.get("world_velocity") if isinstance(auxiliary, dict) else None
        velocity_log_variance = (
            auxiliary.get("world_velocity_log_variance") if isinstance(auxiliary, dict) else None
        )
        velocity_valid = (
            auxiliary.get("world_velocity_valid_mask") if isinstance(auxiliary, dict) else None
        )
        velocity_axis_valid = (
            auxiliary.get("world_velocity_axis_valid_mask") if isinstance(auxiliary, dict) else None
        )
        measurement_mask = getattr(measured, "measurement_mask", None)
        pair_mask = getattr(association, "pair_mask", None)
        belief_indices = getattr(association, "belief_indices", None)
        measurement_indices = getattr(association, "measurement_indices", None)
        if not all(
            isinstance(value, Tensor)
            for value in (
                positions,
                measurement_mask,
                pair_mask,
                belief_indices,
                measurement_indices,
            )
        ):
            return None
        assert isinstance(positions, Tensor)
        assert isinstance(measurement_mask, Tensor)
        assert isinstance(pair_mask, Tensor)
        assert isinstance(belief_indices, Tensor)
        assert isinstance(measurement_indices, Tensor)
        expected = (*measurement_mask.shape, 3)
        if positions.shape != expected or source_object_ids.shape != belief.objects.object_id.shape:
            return None
        if position_log_variance is not None and (
            not isinstance(position_log_variance, Tensor)
            or position_log_variance.shape != expected
            or not torch.isfinite(position_log_variance).all()
        ):
            return None
        velocity_fields = (
            velocity,
            velocity_log_variance,
            velocity_valid,
            velocity_axis_valid,
        )
        if any(value is not None for value in velocity_fields):
            if not all(isinstance(value, Tensor) for value in velocity_fields):
                return None
            assert isinstance(velocity, Tensor)
            assert isinstance(velocity_log_variance, Tensor)
            assert isinstance(velocity_valid, Tensor)
            assert isinstance(velocity_axis_valid, Tensor)
            if (
                velocity.shape != expected
                or velocity_log_variance.shape != expected
                or velocity_valid.shape != measurement_mask.shape
                or velocity_axis_valid.shape != expected
                or velocity_valid.dtype is not torch.bool
                or velocity_axis_valid.dtype is not torch.bool
                or not torch.isfinite(velocity).all()
                or not torch.isfinite(velocity_log_variance).all()
            ):
                return None
        target = belief.objects.position.new_zeros(belief.objects.position.shape)
        target_log_variance = (
            belief.objects.position.new_zeros(belief.objects.position.shape)
            if position_log_variance is not None
            else None
        )
        target_mask = torch.zeros_like(belief.objects.active)
        target_velocity = (
            belief.objects.velocity.new_zeros(belief.objects.velocity.shape)
            if isinstance(velocity, Tensor)
            else None
        )
        target_velocity_log_variance = (
            belief.objects.velocity.new_zeros(belief.objects.velocity.shape)
            if isinstance(velocity_log_variance, Tensor)
            else None
        )
        target_velocity_axis_mask = (
            torch.zeros_like(belief.objects.velocity, dtype=torch.bool)
            if isinstance(velocity_axis_valid, Tensor)
            else None
        )
        for batch_index in range(belief.batch_size):
            for pair_index in (
                torch.nonzero(pair_mask[batch_index], as_tuple=False).flatten().tolist()
            ):
                slot = int(belief_indices[batch_index, pair_index])
                measurement_index = int(measurement_indices[batch_index, pair_index])
                if not (
                    0 <= slot < belief.objects.max_objects
                    and 0 <= measurement_index < positions.shape[1]
                ):
                    continue
                if not bool(measurement_mask[batch_index, measurement_index]):
                    continue
                # A slot reused by lifecycle birth is never evidence about an
                # older candidate trajectory.
                if (
                    source_object_ids[batch_index, slot]
                    != belief.objects.object_id[batch_index, slot]
                ):
                    continue
                target[batch_index, slot] = positions[batch_index, measurement_index]
                if target_log_variance is not None:
                    assert isinstance(position_log_variance, Tensor)
                    target_log_variance[batch_index, slot] = position_log_variance[
                        batch_index,
                        measurement_index,
                    ]
                if target_velocity is not None:
                    assert isinstance(velocity, Tensor)
                    assert isinstance(velocity_log_variance, Tensor)
                    assert isinstance(velocity_valid, Tensor)
                    assert isinstance(velocity_axis_valid, Tensor)
                    assert target_velocity_log_variance is not None
                    assert target_velocity_axis_mask is not None
                    target_velocity[batch_index, slot] = velocity[
                        batch_index,
                        measurement_index,
                    ]
                    target_velocity_log_variance[batch_index, slot] = velocity_log_variance[
                        batch_index,
                        measurement_index,
                    ]
                    target_velocity_axis_mask[batch_index, slot] = (
                        velocity_valid[batch_index, measurement_index]
                        & velocity_axis_valid[batch_index, measurement_index]
                    )
                target_mask[batch_index, slot] = True
        return (
            target,
            target_mask,
            target_log_variance,
            target_velocity,
            target_velocity_axis_mask,
            target_velocity_log_variance,
        )

    def assimilate_observation(
        self,
        belief: WorldBelief,
        measured: object,
        association: object,
    ) -> HypothesisSelection | None:
        """Score only due forecasts using the packet's own RGB evidence."""

        timestamp = getattr(measured, "timestamp", None)
        if not isinstance(timestamp, Tensor) or timestamp.shape != belief.timestamp.shape:
            return None
        current_targets = self._associated_rgb_targets(
            belief,
            measured,
            association,
            belief.objects.object_id.detach().clone(),
        )
        retained: list[PendingHypothesisEvidence] = []
        latest: HypothesisSelection | None = None
        for pending in self.pending:
            difference = timestamp - pending.due_timestamp
            if bool(torch.all(difference < -self.timestamp_tolerance_seconds)):
                retained.append(pending)
                continue
            # An asynchronous packet that arrives after the forecast's exact
            # endpoint cannot be used as an interpolated target silently.
            if not bool(torch.all(difference.abs() <= self.timestamp_tolerance_seconds)):
                continue
            targets = self._associated_rgb_targets(
                belief, measured, association, pending.source_object_ids
            )
            if targets is None:
                continue
            (
                target_positions,
                target_mask,
                target_position_log_variance,
                target_velocities,
                target_velocity_axis_mask,
                target_velocity_log_variance,
            ) = targets
            if not bool(target_mask.any()):
                continue
            uses_velocity_evidence = bool(
                self.velocity_evidence_weight or self.velocity_nonregression_gate_enabled
            )
            if uses_velocity_evidence and (
                target_velocity_axis_mask is None or not bool(target_velocity_axis_mask.any())
            ):
                continue
            if self.local_applicability_enabled and pending.entity_regime is None:
                raise RuntimeError("local hypothesis evidence is missing its scheduled regime")
            entity_axis_observability = (
                torch.ones_like(target_positions)
                if target_position_log_variance is None
                else 1.0 / (1.0 + target_position_log_variance.clamp(-20.0, 10.0).exp())
            )
            entity_axis_observability = torch.where(
                target_mask.unsqueeze(-1),
                entity_axis_observability,
                torch.zeros_like(entity_axis_observability),
            )
            if uses_velocity_evidence:
                assert target_velocity_axis_mask is not None
                assert target_velocity_log_variance is not None
                velocity_observability = 1.0 / (
                    1.0 + target_velocity_log_variance.clamp(-20.0, 10.0).exp()
                )
                entity_axis_observability = torch.where(
                    target_velocity_axis_mask,
                    torch.minimum(entity_axis_observability, velocity_observability),
                    torch.zeros_like(entity_axis_observability),
                )
            latest = self.pools[pending.horizon_index].assimilate(
                belief,
                target_positions.unsqueeze(1),
                target_mask.unsqueeze(1),
                trajectories=pending.trajectories,
                target_position_log_variance=(
                    target_position_log_variance.unsqueeze(1)
                    if target_position_log_variance is not None
                    else None
                ),
                target_velocities=(
                    target_velocities.unsqueeze(1) if target_velocities is not None else None
                ),
                target_velocity_axis_mask=(
                    target_velocity_axis_mask.unsqueeze(1)
                    if target_velocity_axis_mask is not None
                    else None
                ),
                target_velocity_log_variance=(
                    target_velocity_log_variance.unsqueeze(1)
                    if target_velocity_log_variance is not None
                    else None
                ),
                velocity_weight=self.velocity_evidence_weight,
                velocity_nonregression_gate_enabled=(self.velocity_nonregression_gate_enabled),
                axis_prior_strength=self.axis_prior_strength,
                entity_regime=(pending.entity_regime if self.local_applicability_enabled else None),
                evidence_timestamp=(timestamp if self.local_applicability_enabled else None),
                entity_axis_observability=(
                    entity_axis_observability if self.local_applicability_enabled else None
                ),
                robust_influence_delta=(
                    self.robust_influence_delta if self.local_applicability_enabled else 0.0
                ),
                learned_position_residual=(
                    torch.where(
                        target_mask.unsqueeze(-1),
                        target_positions - pending.trajectories[0].positions[:, 0],
                        torch.zeros_like(target_positions),
                    )
                    if self.local_applicability_enabled
                    else None
                ),
            )
        self.pending = retained
        if current_targets is not None:
            (
                _,
                _,
                _,
                target_velocities,
                target_velocity_axis_mask,
                target_velocity_log_variance,
            ) = current_targets
            if (
                target_velocities is not None
                and target_velocity_axis_mask is not None
                and target_velocity_log_variance is not None
            ):
                seen_models: set[int] = set()
                for model in self.pool.dynamics_models:
                    assimilate_velocity = getattr(
                        model,
                        "assimilate_velocity_observation",
                        None,
                    )
                    if assimilate_velocity is None or id(model) in seen_models:
                        continue
                    seen_models.add(id(model))
                    assimilate_velocity(
                        belief,
                        target_velocities,
                        target_velocity_axis_mask,
                        target_velocity_log_variance,
                        timestamp,
                    )
        return latest

    def _learned_fallback_trajectory(
        self,
        belief: WorldBelief,
        offsets: Tensor,
        *,
        composition_grid_fallback: bool,
    ) -> BeliefTrajectory:
        learned = self.pool.rollout_engine.rollout_dynamics(
            (self.pool.dynamics_models[0],), belief, offsets
        )[0]
        local_shape = (*learned.active_mask.shape, 3)
        indices = torch.zeros(local_shape, device=belief.device, dtype=torch.int64)
        supported = torch.zeros_like(indices, dtype=torch.bool)
        diagnostics = torch.zeros_like(indices, dtype=belief.dtype)
        grid_fallback = learned.active_mask.unsqueeze(-1).expand(local_shape)
        if not composition_grid_fallback:
            grid_fallback = torch.zeros_like(grid_fallback)
        return BeliefTrajectory(
            timestamps=learned.timestamps,
            positions=learned.positions,
            velocities=learned.velocities,
            orientations=learned.orientations,
            motion_mode_logits=learned.motion_mode_logits,
            fast_log_variance=learned.fast_log_variance,
            active_mask=learned.active_mask,
            event_logits=learned.event_logits,
            auxiliary={
                **learned.auxiliary,
                "hypothesis_axis_index": indices,
                "hypothesis_axis_supported": supported,
                "hypothesis_axis_support_count": indices.clone(),
                "hypothesis_axis_evidence_age_seconds": diagnostics.clone(),
                "hypothesis_axis_observability": diagnostics.clone(),
                "hypothesis_axis_predictive_variance": diagnostics.clone(),
                "hypothesis_axis_confidence_margin": diagnostics.clone(),
                "hypothesis_position_residual": diagnostics.clone(),
                "hypothesis_position_residual_applied": supported.clone(),
                "hypothesis_interaction_regime": self._trajectory_regime(belief, learned),
                "hypothesis_composition_grid_fallback": grid_fallback,
                "hypothesis_rollout_candidate_indices": torch.zeros(
                    1, device=belief.device, dtype=torch.int64
                ),
            },
        ).validate()

    def _predict_composed(
        self,
        belief: WorldBelief,
        offsets: Tensor,
    ) -> BeliefTrajectory:
        """Compose applicable local effects through bounded coherent steps."""

        assert self.composition_step_seconds is not None
        assert self.composition_horizon_index is not None
        if offsets.shape[1] == 0:
            return self._learned_fallback_trajectory(
                belief, offsets, composition_grid_fallback=False
            )
        if not torch.isfinite(offsets).all() or torch.any(offsets < 0):
            raise ValueError("query_times must be finite nonnegative offsets")
        if offsets.shape[1] > 1 and torch.any(offsets[:, 1:] < offsets[:, :-1]):
            raise ValueError("query_times must be sorted for every batch element")
        # A shared query grid is the normal online/evaluator contract. Mixed
        # per-row grids fall back to the learned model rather than coupling
        # rows through a Python substep schedule or transferring evidence.
        if not torch.equal(offsets, offsets[:1].expand_as(offsets)):
            return self._learned_fallback_trajectory(
                belief, offsets, composition_grid_fallback=True
            )
        step_seconds = self.composition_step_seconds
        query_values = [float(value) for value in offsets[0].detach().cpu().tolist()]
        previous = 0.0
        substep_counts: list[int] = []
        for query in query_values:
            interval = query - previous
            count = int(round(interval / step_seconds))
            if count < 0 or abs(interval - count * step_seconds) > self.timestamp_tolerance_seconds:
                return self._learned_fallback_trajectory(
                    belief, offsets, composition_grid_fallback=True
                )
            substep_counts.append(count)
            previous = query

        current = belief.clone()
        pool = self.pools[self.composition_horizon_index]
        candidate_count = len(pool.dynamics_models)
        beliefs: list[WorldBelief] = []
        event_values: list[Tensor] = []
        choice_values: list[Tensor] = []
        supported_values: list[Tensor] = []
        support_count_values: list[Tensor] = []
        age_values: list[Tensor] = []
        observability_values: list[Tensor] = []
        predictive_variance_values: list[Tensor] = []
        confidence_values: list[Tensor] = []
        residual_values: list[Tensor] = []
        residual_applied_values: list[Tensor] = []
        regime_values: list[Tensor] = []
        candidate_step_count_values: list[Tensor] = []
        fallback_step_count_values: list[Tensor] = []
        total_step_count_values: list[Tensor] = []
        regime_step_count_values: list[Tensor] = []
        for query_index, substep_count in enumerate(substep_counts):
            segment_candidate_count = torch.zeros(
                belief.batch_size,
                belief.objects.max_objects,
                3,
                candidate_count,
                device=belief.device,
                dtype=torch.int64,
            )
            segment_fallback_count = torch.zeros(
                segment_candidate_count.shape[:-1],
                device=belief.device,
                dtype=torch.int64,
            )
            segment_total_count = torch.zeros_like(segment_fallback_count)
            segment_supported_count = torch.zeros_like(segment_fallback_count)
            segment_min_support = torch.full_like(
                segment_fallback_count, torch.iinfo(torch.int64).max
            )
            segment_max_age = torch.zeros_like(segment_fallback_count, dtype=belief.dtype)
            segment_min_observability = torch.ones_like(segment_fallback_count, dtype=belief.dtype)
            segment_max_predictive_variance = torch.zeros_like(
                segment_fallback_count, dtype=belief.dtype
            )
            segment_min_confidence = torch.ones_like(segment_fallback_count, dtype=belief.dtype)
            segment_regime_count = torch.zeros(
                belief.batch_size,
                belief.objects.max_objects,
                NUM_HYPOTHESIS_REGIMES,
                device=belief.device,
                dtype=torch.int64,
            )
            last_choice = torch.zeros_like(segment_fallback_count)
            last_residual = torch.zeros_like(segment_max_age)
            last_residual_applied = torch.zeros_like(segment_fallback_count, dtype=torch.bool)
            last_regime = torch.full(
                belief.objects.active.shape,
                int(HypothesisRegime.FREE),
                device=belief.device,
                dtype=torch.int64,
            )
            interval_collision_logits: Tensor | None = None
            last_step: RolloutStep | None = None
            for _ in range(substep_count):
                delta_time = current.timestamp.new_full(
                    current.timestamp.shape,
                    step_seconds,
                )
                learned_step = pool.dynamics_models[0].predict_step(
                    current.clone(),
                    delta_time,
                )
                learned_trajectory = self._trajectory_from_step(learned_step)
                regime = self._trajectory_regime(current, learned_trajectory)[:, 0]
                applicability = pool.selected_entity_axis_applicability(
                    belief,
                    entity_regime=regime,
                    current_timestamp=current.timestamp,
                    minimum_support_count=self.minimum_support_count,
                    maximum_age_seconds=self.maximum_evidence_age_seconds,
                    minimum_observability=self.minimum_observability,
                    minimum_confidence_margin=self.minimum_confidence_margin,
                    candidate_regime_mask=self.candidate_regime_mask.to(belief.device),
                    candidate_entity_axis_support=(
                        self._candidate_entity_axis_support(pool, belief)
                    ),
                )
                active = learned_step.belief.objects.active
                step_choice = torch.where(
                    applicability.supported & active.unsqueeze(-1),
                    applicability.selected_index,
                    torch.zeros_like(applicability.selected_index),
                )
                for axis in self.axis_independent_axes:
                    if self.residual_correction_gain_by_axis[axis] > 0:
                        step_choice[..., axis] = torch.where(
                            applicability.supported[..., axis] & active,
                            torch.zeros_like(step_choice[..., axis]),
                            step_choice[..., axis],
                        )
                candidate_steps: dict[int, RolloutStep] = {0: learned_step}
                selected_candidates = {
                    int(value)
                    for value in step_choice[..., self.axis_independent_axes]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                    if int(value) != 0
                }
                for candidate_index in sorted(selected_candidates):
                    candidate_steps[candidate_index] = pool.dynamics_models[
                        candidate_index
                    ].predict_step(current.clone(), delta_time)

                position = learned_step.belief.objects.position.clone()
                velocity = learned_step.belief.objects.velocity.clone()
                for axis in self.axis_independent_axes:
                    axis_active = active
                    axis_supported = applicability.supported[..., axis] & axis_active
                    axis_choice = step_choice[..., axis]
                    if self.residual_correction_gain_by_axis[axis] > 0:
                        residual_axis_supported = (
                            axis_supported & applicability.position_residual_supported[..., axis]
                        )
                        last_residual[..., axis] = torch.where(
                            axis_supported,
                            applicability.position_residual[..., axis],
                            torch.zeros_like(last_residual[..., axis]),
                        )
                        last_residual_applied[..., axis] = residual_axis_supported
                    segment_total_count[..., axis] += axis_active.to(torch.int64)
                    segment_supported_count[..., axis] += axis_supported.to(torch.int64)
                    segment_fallback_count[..., axis] += (axis_active & ~axis_supported).to(
                        torch.int64
                    )
                    for candidate_index in range(candidate_count):
                        segment_candidate_count[..., axis, candidate_index] += (
                            axis_active & (axis_choice == candidate_index)
                        ).to(torch.int64)
                    segment_min_support[..., axis] = torch.where(
                        axis_supported,
                        torch.minimum(
                            segment_min_support[..., axis],
                            applicability.support_count[..., axis],
                        ),
                        segment_min_support[..., axis],
                    )
                    segment_max_age[..., axis] = torch.where(
                        axis_supported,
                        torch.maximum(
                            segment_max_age[..., axis],
                            applicability.age_seconds[..., axis],
                        ),
                        segment_max_age[..., axis],
                    )
                    segment_min_observability[..., axis] = torch.where(
                        axis_supported,
                        torch.minimum(
                            segment_min_observability[..., axis],
                            applicability.observability[..., axis],
                        ),
                        segment_min_observability[..., axis],
                    )
                    segment_max_predictive_variance[..., axis] = torch.where(
                        axis_supported,
                        torch.maximum(
                            segment_max_predictive_variance[..., axis],
                            applicability.predictive_variance[..., axis],
                        ),
                        segment_max_predictive_variance[..., axis],
                    )
                    segment_min_confidence[..., axis] = torch.where(
                        axis_supported,
                        torch.minimum(
                            segment_min_confidence[..., axis],
                            applicability.confidence_margin[..., axis],
                        ),
                        segment_min_confidence[..., axis],
                    )
                    for candidate_index, candidate_step in candidate_steps.items():
                        if candidate_index == 0:
                            continue
                        selected = axis_supported & (axis_choice == candidate_index)
                        selected = selected & candidate_step.belief.objects.active
                        position[..., axis] = torch.where(
                            selected,
                            candidate_step.belief.objects.position[..., axis],
                            position[..., axis],
                        )
                        velocity[..., axis] = torch.where(
                            selected,
                            candidate_step.belief.objects.velocity[..., axis],
                            velocity[..., axis],
                        )
                regime_one_hot = torch.nn.functional.one_hot(
                    regime,
                    num_classes=NUM_HYPOTHESIS_REGIMES,
                ).to(torch.int64)
                segment_regime_count += regime_one_hot * active.unsqueeze(-1).to(torch.int64)
                last_choice = step_choice
                last_regime = regime
                composed_objects = learned_step.belief.objects.replace(
                    position=position,
                    velocity=velocity,
                )
                current = learned_step.belief.replace(objects=composed_objects)
                collision_logits = learned_step.event_logits[..., MotionMode.COLLISION]
                interval_collision_logits = (
                    collision_logits
                    if interval_collision_logits is None
                    else torch.maximum(interval_collision_logits, collision_logits)
                )
                last_step = learned_step

            if last_step is None:
                zero_delta = current.timestamp.new_zeros(current.timestamp.shape)
                last_step = pool.dynamics_models[0].predict_step(current.clone(), zero_delta)
                last_regime = self._trajectory_regime(
                    current,
                    self._trajectory_from_step(last_step),
                )[:, 0]
                current = last_step.belief
                interval_collision_logits = last_step.event_logits[..., MotionMode.COLLISION]
            current = current.replace(timestamp=belief.timestamp + offsets[:, query_index])
            event_logits = last_step.event_logits.clone()
            assert interval_collision_logits is not None
            event_logits[..., MotionMode.COLLISION] = interval_collision_logits
            fully_supported = (segment_total_count > 0) & (
                segment_supported_count == segment_total_count
            )
            minimum_support = torch.where(
                segment_supported_count > 0,
                segment_min_support,
                torch.zeros_like(segment_min_support),
            )
            minimum_observability = torch.where(
                segment_supported_count > 0,
                segment_min_observability,
                torch.zeros_like(segment_min_observability),
            )
            minimum_confidence = torch.where(
                segment_supported_count > 0,
                segment_min_confidence,
                torch.zeros_like(segment_min_confidence),
            )
            dominant_choice = segment_candidate_count.argmax(dim=-1).to(torch.int64)
            beliefs.append(current)
            event_values.append(event_logits)
            choice_values.append(dominant_choice if substep_count else last_choice)
            supported_values.append(fully_supported)
            support_count_values.append(minimum_support)
            age_values.append(segment_max_age)
            observability_values.append(minimum_observability)
            predictive_variance_values.append(segment_max_predictive_variance)
            confidence_values.append(minimum_confidence)
            residual_values.append(last_residual)
            residual_applied_values.append(last_residual_applied)
            regime_values.append(last_regime)
            candidate_step_count_values.append(segment_candidate_count)
            fallback_step_count_values.append(segment_fallback_count)
            total_step_count_values.append(segment_total_count)
            regime_step_count_values.append(segment_regime_count)

        candidate_step_counts = torch.stack(candidate_step_count_values, dim=1)
        residual_applied = torch.stack(residual_applied_values, dim=1)
        scene_intervened = candidate_step_counts[..., 1:].sum(dim=(2, 3, 4)) > 0
        composed_positions = torch.stack([item.objects.position for item in beliefs], dim=1)
        composed_velocities = torch.stack([item.objects.velocity for item in beliefs], dim=1)
        composed_orientations = torch.stack([item.objects.orientation for item in beliefs], dim=1)
        composed_modes = torch.stack([item.objects.motion_mode_logits for item in beliefs], dim=1)
        composed_variance = torch.stack([item.objects.fast_log_variance for item in beliefs], dim=1)
        composed_active = torch.stack([item.objects.active for item in beliefs], dim=1)
        composed_events = torch.stack(event_values, dim=1)
        trajectory_timestamps = belief.timestamp.unsqueeze(1) + offsets
        if not bool(scene_intervened.all()):
            canonical = pool.rollout_engine.rollout_dynamics(
                (pool.dynamics_models[0],),
                belief,
                offsets,
            )[0]
            state_mask = scene_intervened.unsqueeze(-1).unsqueeze(-1)
            node_mask = scene_intervened.unsqueeze(-1)
            composed_positions = torch.where(state_mask, composed_positions, canonical.positions)
            composed_velocities = torch.where(state_mask, composed_velocities, canonical.velocities)
            composed_orientations = torch.where(
                state_mask, composed_orientations, canonical.orientations
            )
            composed_modes = torch.where(state_mask, composed_modes, canonical.motion_mode_logits)
            composed_variance = torch.where(
                state_mask, composed_variance, canonical.fast_log_variance
            )
            composed_active = torch.where(node_mask, composed_active, canonical.active_mask)
            composed_events = torch.where(state_mask, composed_events, canonical.event_logits)
            trajectory_timestamps = canonical.timestamps
        stacked_residual = torch.stack(residual_values, dim=1)
        for axis in self.axis_independent_axes:
            gain = self.residual_correction_gain_by_axis[axis]
            if gain > 0:
                composed_positions[..., axis] = torch.where(
                    residual_applied[..., axis],
                    composed_positions[..., axis] + float(gain) * stacked_residual[..., axis],
                    composed_positions[..., axis],
                )

        return BeliefTrajectory(
            timestamps=trajectory_timestamps,
            positions=composed_positions,
            velocities=composed_velocities,
            orientations=composed_orientations,
            motion_mode_logits=composed_modes,
            fast_log_variance=composed_variance,
            active_mask=composed_active,
            event_logits=composed_events,
            auxiliary={
                "hypothesis_axis_index": torch.stack(choice_values, dim=1),
                "hypothesis_axis_supported": torch.stack(supported_values, dim=1),
                "hypothesis_axis_support_count": torch.stack(support_count_values, dim=1),
                "hypothesis_axis_evidence_age_seconds": torch.stack(age_values, dim=1),
                "hypothesis_axis_observability": torch.stack(observability_values, dim=1),
                "hypothesis_axis_predictive_variance": torch.stack(
                    predictive_variance_values, dim=1
                ),
                "hypothesis_axis_confidence_margin": torch.stack(confidence_values, dim=1),
                "hypothesis_position_residual": stacked_residual,
                "hypothesis_position_residual_applied": residual_applied,
                "hypothesis_interaction_regime": torch.stack(regime_values, dim=1),
                "hypothesis_composed_candidate_step_count": candidate_step_counts,
                "hypothesis_composed_fallback_step_count": torch.stack(
                    fallback_step_count_values, dim=1
                ),
                "hypothesis_composed_total_step_count": torch.stack(total_step_count_values, dim=1),
                "hypothesis_composed_regime_step_count": torch.stack(
                    regime_step_count_values, dim=1
                ),
                "hypothesis_composition_grid_fallback": torch.zeros_like(
                    torch.stack(supported_values, dim=1), dtype=torch.bool
                ),
                "hypothesis_rollout_candidate_indices": torch.arange(
                    candidate_count,
                    device=belief.device,
                    dtype=torch.int64,
                ),
            },
        ).validate()

    def predict(
        self, belief: WorldBelief, query_times: Tensor | Sequence[float]
    ) -> BeliefTrajectory | None:
        """Compose only horizon-supported axes; other outputs stay learned."""

        if not any(pool.last_selection is not None for pool in self.pools):
            return None
        offsets = torch.as_tensor(query_times, device=belief.device, dtype=belief.dtype)
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0).expand(belief.batch_size, -1)
        elif offsets.ndim != 2 or offsets.shape[0] != belief.batch_size:
            raise ValueError("query_times must have shape [T] or [B,T]")
        if self.composition_step_seconds is not None:
            return self._predict_composed(belief, offsets)
        choices = torch.zeros(
            belief.batch_size,
            offsets.shape[1],
            belief.objects.max_objects,
            3,
            device=belief.device,
            dtype=torch.int64,
        )
        supported = torch.zeros_like(choices, dtype=torch.bool)
        support_count = torch.zeros_like(choices, dtype=torch.int64)
        age_seconds = torch.zeros_like(choices, dtype=belief.dtype)
        observability = torch.zeros_like(choices, dtype=belief.dtype)
        predictive_variance = torch.zeros_like(choices, dtype=belief.dtype)
        confidence_margin = torch.zeros_like(choices, dtype=belief.dtype)
        position_residual = torch.zeros_like(choices, dtype=belief.dtype)
        residual_supported = torch.zeros_like(choices, dtype=torch.bool)
        query_regime = torch.full(
            choices.shape[:3],
            int(HypothesisRegime.FREE),
            device=belief.device,
            dtype=torch.int64,
        )
        learned: BeliefTrajectory | None = None
        if self.local_applicability_enabled:
            learned = self.pool.rollout_engine.rollout_dynamics(
                (self.pool.dynamics_models[0],),
                belief,
                query_times,
            )[0]
            query_regime = self._trajectory_regime(belief, learned)
        for horizon, pool in zip(self.evidence_horizons_seconds, self.pools, strict=True):
            if pool.last_selection is None:
                continue
            supported_queries = (offsets - float(horizon)).abs() <= self.timestamp_tolerance_seconds
            if self.local_applicability_enabled:
                for query_index in range(offsets.shape[1]):
                    applicability = pool.selected_entity_axis_applicability(
                        belief,
                        entity_regime=query_regime[:, query_index],
                        current_timestamp=belief.timestamp,
                        minimum_support_count=self.minimum_support_count,
                        maximum_age_seconds=self.maximum_evidence_age_seconds,
                        minimum_observability=self.minimum_observability,
                        minimum_confidence_margin=self.minimum_confidence_margin,
                        candidate_regime_mask=self.candidate_regime_mask.to(belief.device),
                        candidate_entity_axis_support=(
                            self._candidate_entity_axis_support(pool, belief)
                        ),
                    )
                    query_supported = supported_queries[:, query_index].view(-1, 1, 1)
                    for axis in self.axis_independent_axes:
                        axis_supported = (
                            query_supported[..., 0] & applicability.supported[..., axis]
                        )
                        uses_residual = self.residual_correction_gain_by_axis[axis] > 0
                        selected_index = (
                            torch.zeros_like(applicability.selected_index[..., axis])
                            if uses_residual
                            else applicability.selected_index[..., axis]
                        )
                        choices[:, query_index, :, axis] = torch.where(
                            axis_supported,
                            selected_index,
                            choices[:, query_index, :, axis],
                        )
                        supported[:, query_index, :, axis] |= axis_supported
                        support_count[:, query_index, :, axis] = torch.where(
                            axis_supported,
                            applicability.support_count[..., axis],
                            support_count[:, query_index, :, axis],
                        )
                        age_seconds[:, query_index, :, axis] = torch.where(
                            axis_supported,
                            applicability.age_seconds[..., axis],
                            age_seconds[:, query_index, :, axis],
                        )
                        observability[:, query_index, :, axis] = torch.where(
                            axis_supported,
                            applicability.observability[..., axis],
                            observability[:, query_index, :, axis],
                        )
                        predictive_variance[:, query_index, :, axis] = torch.where(
                            axis_supported,
                            applicability.predictive_variance[..., axis],
                            predictive_variance[:, query_index, :, axis],
                        )
                        confidence_margin[:, query_index, :, axis] = torch.where(
                            axis_supported,
                            applicability.confidence_margin[..., axis],
                            confidence_margin[:, query_index, :, axis],
                        )
                        if uses_residual:
                            position_residual[:, query_index, :, axis] = torch.where(
                                axis_supported,
                                applicability.position_residual[..., axis],
                                position_residual[:, query_index, :, axis],
                            )
                            residual_supported[:, query_index, :, axis] |= (
                                axis_supported
                                & applicability.position_residual_supported[..., axis]
                            )
            else:
                horizon_choices = pool.selected_entity_axis_index(belief)
                assert pool.entity_axis_evidence_seen is not None
                for axis in self.axis_independent_axes:
                    axis_supported = supported_queries.unsqueeze(
                        -1
                    ) & pool.entity_axis_evidence_seen[:, :, axis].unsqueeze(1)
                    choices[..., axis] = torch.where(
                        axis_supported,
                        horizon_choices[:, :, axis].unsqueeze(1),
                        choices[..., axis],
                    )
                    supported[..., axis] = supported[..., axis] | axis_supported
        # The forecast retains learned lifecycle, event, identity, and
        # uncertainty outputs.  Only candidates selected for configured axes
        # need a fresh long-horizon rollout.  In the common learned-selection
        # case this avoids needlessly evaluating every analytic alternative at
        # every forecast anchor, which otherwise multiplies MPS work without
        # changing any emitted tensor.
        candidate_indices = {0}
        for axis in self.axis_independent_axes:
            candidate_indices.update(
                int(index) for index in choices[..., axis].detach().cpu().reshape(-1).tolist()
            )
        ordered_indices = tuple(sorted(candidate_indices))
        if learned is None:
            trajectories = self.pool.rollout_engine.rollout_dynamics(
                tuple(self.pool.dynamics_models[index] for index in ordered_indices),
                belief,
                query_times,
            )
            learned = trajectories[0]
        else:
            alternative_models = tuple(
                self.pool.dynamics_models[index] for index in ordered_indices if index != 0
            )
            alternatives = (
                self.pool.rollout_engine.rollout_dynamics(
                    alternative_models,
                    belief,
                    query_times,
                )
                if alternative_models
                else []
            )
            trajectories = [learned, *alternatives]
        positions = learned.positions.clone()
        velocities = learned.velocities.clone()
        candidate_positions = torch.stack([item.positions for item in trajectories], dim=-1)
        candidate_velocities = torch.stack([item.velocities for item in trajectories], dim=-1)
        local_index = torch.empty(
            len(self.pool.dynamics_models),
            device=belief.device,
            dtype=torch.int64,
        )
        for local, external in enumerate(ordered_indices):
            local_index[external] = local
        for axis in self.axis_independent_axes:
            selected = local_index[choices[..., axis]].unsqueeze(-1)
            positions[..., axis] = torch.gather(
                candidate_positions[..., axis, :],
                -1,
                selected.expand(
                    belief.batch_size,
                    positions.shape[1],
                    positions.shape[2],
                    1,
                ),
            ).squeeze(-1)
            velocities[..., axis] = torch.gather(
                candidate_velocities[..., axis, :],
                -1,
                selected.expand(
                    belief.batch_size,
                    velocities.shape[1],
                    velocities.shape[2],
                    1,
                ),
            ).squeeze(-1)
            gain = self.residual_correction_gain_by_axis[axis]
            if gain > 0:
                positions[..., axis] = torch.where(
                    residual_supported[..., axis],
                    positions[..., axis] + float(gain) * position_residual[..., axis],
                    positions[..., axis],
                )
        return BeliefTrajectory(
            timestamps=learned.timestamps,
            positions=positions,
            velocities=velocities,
            orientations=learned.orientations,
            motion_mode_logits=learned.motion_mode_logits,
            # Analytic alternatives do not own a calibrated predictive
            # uncertainty model.  Keep the learned trajectory's uncertainty
            # while composing only the explicitly selected physical axes.
            fast_log_variance=learned.fast_log_variance,
            active_mask=learned.active_mask,
            event_logits=learned.event_logits,
            auxiliary={
                **learned.auxiliary,
                "hypothesis_axis_index": choices.detach().clone(),
                "hypothesis_axis_supported": supported.detach().clone(),
                "hypothesis_axis_support_count": support_count.detach().clone(),
                "hypothesis_axis_evidence_age_seconds": age_seconds.detach().clone(),
                "hypothesis_axis_observability": observability.detach().clone(),
                "hypothesis_axis_predictive_variance": predictive_variance.detach().clone(),
                "hypothesis_axis_confidence_margin": confidence_margin.detach().clone(),
                "hypothesis_position_residual": position_residual.detach().clone(),
                "hypothesis_position_residual_applied": residual_supported.detach().clone(),
                "hypothesis_interaction_regime": query_regime.detach().clone(),
                "hypothesis_composition_grid_fallback": torch.zeros_like(
                    supported, dtype=torch.bool
                ),
                "hypothesis_rollout_candidate_indices": torch.tensor(
                    ordered_indices,
                    dtype=torch.int64,
                    device=belief.device,
                ),
            },
        ).validate()
