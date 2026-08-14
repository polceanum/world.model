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

import torch
from torch import Tensor

from world_model.belief import BeliefTrajectory, MotionMode, WorldBelief
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.rollout import RolloutEngine, RolloutStep


@dataclass(frozen=True)
class HypothesisSelection:
    """Evidence and posterior choice for a batch of candidate rollouts."""

    scores: Tensor
    selected_index: Tensor
    posterior_weights: Tensor
    axis_scores: Tensor | None = None
    axis_posterior: Tensor | None = None

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
        if self.selected_index.dtype != torch.int64:
            raise TypeError("selected_index must use torch.int64")
        if not torch.isfinite(self.scores).all() or not torch.isfinite(
            self.posterior_weights
        ).all():
            raise ValueError("hypothesis selection contains NaN or Inf")
        if self.axis_posterior is not None:
            if not torch.isfinite(self.axis_posterior).all():
                raise ValueError("axis posterior contains NaN or Inf")
            if not torch.allclose(
                self.axis_posterior.sum(dim=-1),
                torch.ones(self.axis_posterior.shape[:2], device=self.axis_posterior.device, dtype=self.axis_posterior.dtype),
                atol=1e-5,
            ):
                raise ValueError("axis posterior weights must sum to one")
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

    def predict_step(self, belief: WorldBelief, delta_time: Tensor) -> RolloutStep:
        if delta_time.shape != belief.timestamp.shape:
            raise ValueError("delta_time must have shape [B]")
        objects = belief.objects.clone()
        active = objects.active.unsqueeze(-1)
        objects.position = objects.position + objects.velocity * delta_time[:, None, None] * active
        if self.damping:
            objects.velocity = objects.velocity * torch.exp(
                -self.damping * delta_time[:, None, None]
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

    def predict_step(self, belief: WorldBelief, delta_time: Tensor) -> RolloutStep:
        if delta_time.shape != belief.timestamp.shape:
            raise ValueError("delta_time must have shape [B]")
        before = belief.objects
        after = self.analytic(before, belief.gravity, delta_time)
        radius = before.geometry[..., :1].clamp_min(1.0e-5)
        before_ground = before.position[..., 1] - radius[..., 0]
        after_ground = after.position[..., 1] - radius[..., 0]
        ground_event = before.active & after.active & (before_ground > self.ground_height) & (
            after_ground <= self.ground_height
        ) & (before.velocity[..., 1] < 0)
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
            torch.as_tensor(self.ground_height, device=after.position.device, dtype=after.position.dtype)
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
        pair_restitution = (before.restitution[..., 0][:, :, None] + before.restitution[..., 0][:, None, :]) * 0.5
        impulse = -(1.0 + pair_restitution) * (relative_velocity * normal).sum(dim=-1)
        inverse_mass = before.mass[..., 0].reciprocal()
        impulse = torch.where(pair_contact, impulse / (inverse_mass[:, :, None] + inverse_mass[:, None, :]).clamp_min(1.0e-6), torch.zeros_like(impulse))
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
            self.rollout_engine.rollout(predictor, belief, query_times)
            for predictor in predictors
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
        target_collision: Tensor | None = None,
        position_weight: float = 1.0,
        lifecycle_weight: float = 0.0,
        event_weight: float = 0.0,
        position_gate_ratio: float = 0.0,
        axis_gate_ratio: float = 0.0,
        event_gate_ratio: float = 0.0,
        axis_weights: Sequence[float] | Tensor | None = None,
        uncertainty_aware: bool = True,
        temperature: float = 1.0,
    ) -> HypothesisSelection:
        """Score candidates by masked position NLL and select per batch item.

        ``target_mask`` is ``[B,T,N]`` and permits asynchronous/occluded
        observations.  With uncertainty enabled, the score is the Gaussian
        diagonal NLL up to a constant; this prevents a deliberately over-wide
        candidate from winning solely through residual magnitude.
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
        if temperature <= 0 or not torch.isfinite(torch.as_tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        for name, value in (
            ("position_weight", position_weight),
            ("lifecycle_weight", lifecycle_weight),
            ("event_weight", event_weight),
            ("position_gate_ratio", position_gate_ratio),
            ("axis_gate_ratio", axis_gate_ratio),
            ("event_gate_ratio", event_gate_ratio),
        ):
            if value < 0 or not torch.isfinite(torch.as_tensor(value)):
                raise ValueError(f"{name} must be finite and nonnegative")
        if position_weight + lifecycle_weight + event_weight <= 0:
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
            if not torch.isfinite(resolved_axis_weights).all() or torch.any(resolved_axis_weights < 0):
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
        valid_count = mask.sum(dim=(1, 2, 3)).clamp_min(1).to(target_positions.dtype)
        axis_valid_count = target_mask.sum(dim=(1, 2)).clamp_min(1).to(target_positions.dtype)
        candidate_scores: list[Tensor] = []
        position_scores: list[Tensor] = []
        axis_position_scores: list[Tensor] = []
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
                point_loss = residual.square() / variance + log_variance
            else:
                point_loss = residual.square()
            axis_position_scores.append(
                (point_loss * mask).sum(dim=(1, 2)) / axis_valid_count.unsqueeze(-1)
            )
            point_loss = point_loss * resolved_axis_weights.view(1, 1, 1, -1)
            position_score = (point_loss * mask).sum(dim=(1, 2, 3)) / valid_count
            position_scores.append(position_score)
            score = position_weight * position_score
            if lifecycle_weight:
                lifecycle_loss = (
                    trajectory.active_mask != target_mask
                ).to(target_positions.dtype).mean(dim=(1, 2))
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
        scale = torch.as_tensor(temperature, device=scores.device, dtype=scores.dtype)
        posterior_weights = torch.softmax(-scores / scale, dim=-1)
        selected_index = scores.argmin(dim=-1).to(torch.int64)
        return HypothesisSelection(
            scores,
            selected_index,
            posterior_weights,
            axis_scores=torch.stack(axis_position_scores, dim=-1),
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
        if evidence_decay <= 0 or evidence_decay > 1 or not torch.isfinite(
            torch.as_tensor(evidence_decay)
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
        self.last_selection = None

    def _ensure_weights(self, belief: WorldBelief) -> Tensor:
        if self.log_weights is None:
            self.reset(belief.batch_size, device=belief.device, dtype=belief.dtype)
        assert self.log_weights is not None
        if self.log_weights.shape[0] != belief.batch_size:
            raise ValueError("belief batch size changed; reset the hypothesis pool")
        if self.log_weights.device != belief.device or self.log_weights.dtype != belief.dtype:
            raise ValueError("hypothesis weights must share belief device and dtype")
        return self.log_weights

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
    ) -> list[BeliefTrajectory]:
        self._ensure_weights(belief)
        return self.rollout_engine.rollout_dynamics(self.dynamics_models, belief, query_times)

    def assimilate(
        self,
        belief: WorldBelief,
        target_positions: Tensor,
        target_mask: Tensor,
        *,
        trajectories: Sequence[BeliefTrajectory] | None = None,
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
        prior = self._ensure_weights(belief)
        if trajectories is None:
            raise ValueError("assimilate requires trajectories for explicit asynchronous evidence")
        selection = self.rollout_engine.score(
            trajectories,
            target_positions,
            target_mask,
            target_collision=target_collision,
            position_weight=position_weight,
            lifecycle_weight=lifecycle_weight,
            event_weight=event_weight,
            position_gate_ratio=position_gate_ratio,
            axis_gate_ratio=axis_gate_ratio,
            event_gate_ratio=event_gate_ratio,
            axis_weights=axis_weights,
            uncertainty_aware=uncertainty_aware,
            temperature=self.temperature,
        )
        decay = self.evidence_decay if evidence_decay_override is None else float(evidence_decay_override)
        if not 0.0 < decay <= 1.0 or not torch.isfinite(torch.as_tensor(decay)):
            raise ValueError("evidence_decay_override must lie in (0,1]")
        if not 0.0 <= axis_prior_strength <= 1.0 or not torch.isfinite(torch.as_tensor(axis_prior_strength)):
            raise ValueError("axis_prior_strength must lie in [0,1]")
        posterior_log_weights = (
            decay * prior - selection.scores / self.temperature
        )
        posterior_log_weights = posterior_log_weights - torch.logsumexp(
            posterior_log_weights, dim=-1, keepdim=True
        )
        posterior = torch.softmax(posterior_log_weights, dim=-1)
        posterior_selected_index = posterior.argmax(dim=-1).to(torch.int64)
        axis_posterior = None
        if selection.axis_scores is not None:
            axis_logits = -selection.axis_scores / self.temperature
            if axis_prior_strength:
                axis_logits = axis_logits + axis_prior_strength * prior.unsqueeze(1)
            axis_posterior = torch.softmax(axis_logits, dim=-1)
        self.log_weights = posterior_log_weights.detach()
        self.last_selection = HypothesisSelection(
            selection.scores,
            posterior_selected_index,
            posterior,
            axis_scores=selection.axis_scores,
            axis_posterior=axis_posterior,
        ).validate()
        return self.last_selection

    def selected_index(self, belief: WorldBelief) -> Tensor:
        weights = self._ensure_weights(belief)
        return weights.argmax(dim=-1).to(torch.int64)

    def selected_axis_index(self, belief: WorldBelief) -> Tensor:
        """Return per-axis choices from the latest asynchronous evidence."""

        self._ensure_weights(belief)
        if self.last_selection is None or self.last_selection.axis_selected_index is None:
            raise RuntimeError("axis selection is unavailable before assimilation")
        return self.last_selection.axis_selected_index
