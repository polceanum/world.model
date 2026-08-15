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
    score_spread: Tensor | None = None
    axis_score_spread: Tensor | None = None
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
        if self.score_spread is not None and self.score_spread.shape != self.scores.shape:
            raise ValueError("score_spread must match scores")
        if self.axis_score_spread is not None and (
            self.axis_scores is None or self.axis_score_spread.shape != self.axis_scores.shape
        ):
            raise ValueError("axis_score_spread must match axis_scores")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
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
            objects.position = objects.position + objects.velocity * delta_time[:, None, None] * active
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
        scores, score_spread = self._robust_aggregate(
            torch.stack([selection.scores for selection in selections]), risk_penalty
        )
        axis_scores = None
        axis_score_spread = None
        if selections[0].axis_scores is not None:
            if any(selection.axis_scores is None for selection in selections):
                raise ValueError("all ensemble samples must expose axis scores")
            axis_scores, axis_score_spread = self._robust_aggregate(
                torch.stack([selection.axis_scores for selection in selections if selection.axis_scores is not None]),
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
            score_spread=score_spread,
            axis_score_spread=axis_score_spread,
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
                    position=objects.position + torch.where(active, position_delta, torch.zeros_like(position_delta)),
                    velocity=objects.velocity + torch.where(active, velocity_delta, torch.zeros_like(velocity_delta)),
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
        return self._update_evidence(
            prior,
            selection,
            evidence_decay_override=evidence_decay_override,
            axis_prior_strength=axis_prior_strength,
        )

    def assimilate_ensemble(
        self,
        belief: WorldBelief,
        target_positions: Tensor,
        target_mask: Tensor,
        *,
        trajectory_samples: Sequence[Sequence[BeliefTrajectory]],
        risk_penalty: float = 0.0,
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
            score_spread=selection.score_spread,
            axis_score_spread=selection.axis_score_spread,
            sample_count=selection.sample_count,
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


@dataclass(frozen=True)
class PendingHypothesisEvidence:
    """One RGB-only forecast awaiting an observation at its due timestamp."""

    due_timestamp: Tensor
    source_object_ids: Tensor
    trajectories: tuple[BeliefTrajectory, ...]


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
        self.pool = pool
        self.evidence_horizons_seconds = tuple(float(value) for value in evidence_horizons_seconds)
        self.axis_independent_axes = tuple(sorted(set(int(axis) for axis in axis_independent_axes)))
        self.axis_prior_strength = float(axis_prior_strength)
        self.timestamp_tolerance_seconds = float(timestamp_tolerance_seconds)
        self.pending: list[PendingHypothesisEvidence] = []

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        self.pool.reset(batch_size, device=device, dtype=dtype)
        self.pending.clear()

    def schedule(self, belief: WorldBelief) -> None:
        """Issue small candidate rollouts after a corrected posterior."""

        self.pool._ensure_weights(belief)
        # A non-RGB asynchronous packet may advance time beyond a pending
        # endpoint. Such a forecast cannot later acquire an exact RGB target,
        # so bound runtime-local memory without inventing interpolation.
        self.pending = [
            pending
            for pending in self.pending
            if bool(
                torch.all(
                    belief.timestamp
                    <= pending.due_timestamp + self.timestamp_tolerance_seconds
                )
            )
        ]
        for horizon in self.evidence_horizons_seconds:
            trajectories = self.pool.rollout(belief, [horizon])
            self.pending.append(
                PendingHypothesisEvidence(
                    due_timestamp=belief.timestamp.detach().clone() + horizon,
                    source_object_ids=belief.objects.object_id.detach().clone(),
                    trajectories=tuple(trajectories),
                )
            )

    @staticmethod
    def _associated_rgb_targets(
        belief: WorldBelief,
        measured: object,
        association: object,
        source_object_ids: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        """Map associated RGB back-projections to persistent candidate slots."""

        auxiliary = getattr(measured, "auxiliary", None)
        positions = auxiliary.get("world_position") if isinstance(auxiliary, dict) else None
        measurement_mask = getattr(measured, "measurement_mask", None)
        pair_mask = getattr(association, "pair_mask", None)
        belief_indices = getattr(association, "belief_indices", None)
        measurement_indices = getattr(association, "measurement_indices", None)
        if not all(isinstance(value, Tensor) for value in (
            positions, measurement_mask, pair_mask, belief_indices, measurement_indices,
        )):
            return None
        assert isinstance(positions, Tensor)
        assert isinstance(measurement_mask, Tensor)
        assert isinstance(pair_mask, Tensor)
        assert isinstance(belief_indices, Tensor)
        assert isinstance(measurement_indices, Tensor)
        expected = (*measurement_mask.shape, 3)
        if positions.shape != expected or source_object_ids.shape != belief.objects.object_id.shape:
            return None
        target = belief.objects.position.new_zeros(belief.objects.position.shape)
        target_mask = torch.zeros_like(belief.objects.active)
        for batch_index in range(belief.batch_size):
            for pair_index in torch.nonzero(pair_mask[batch_index], as_tuple=False).flatten().tolist():
                slot = int(belief_indices[batch_index, pair_index])
                measurement_index = int(measurement_indices[batch_index, pair_index])
                if not (0 <= slot < belief.objects.max_objects and 0 <= measurement_index < positions.shape[1]):
                    continue
                if not bool(measurement_mask[batch_index, measurement_index]):
                    continue
                # A slot reused by lifecycle birth is never evidence about an
                # older candidate trajectory.
                if source_object_ids[batch_index, slot] != belief.objects.object_id[batch_index, slot]:
                    continue
                target[batch_index, slot] = positions[batch_index, measurement_index]
                target_mask[batch_index, slot] = True
        return target, target_mask

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
            target_positions, target_mask = targets
            if not bool(target_mask.any()):
                continue
            latest = self.pool.assimilate(
                belief,
                target_positions.unsqueeze(1),
                target_mask.unsqueeze(1),
                trajectories=pending.trajectories,
                axis_prior_strength=self.axis_prior_strength,
            )
        self.pending = retained
        return latest

    def predict(self, belief: WorldBelief, query_times: Tensor | Sequence[float]) -> BeliefTrajectory | None:
        """Compose only validated axes; lifecycle/events stay learned-model outputs."""

        if self.pool.last_selection is None:
            return None
        choices = self.pool.selected_axis_index(belief)
        # The forecast retains learned lifecycle, event, identity, and
        # uncertainty outputs.  Only candidates selected for configured axes
        # need a fresh long-horizon rollout.  In the common learned-selection
        # case this avoids needlessly evaluating every analytic alternative at
        # every forecast anchor, which otherwise multiplies MPS work without
        # changing any emitted tensor.
        candidate_indices = {0}
        for axis in self.axis_independent_axes:
            candidate_indices.update(int(index) for index in choices[:, axis].detach().cpu().tolist())
        ordered_indices = tuple(sorted(candidate_indices))
        trajectories = self.pool.rollout_engine.rollout_dynamics(
            tuple(self.pool.dynamics_models[index] for index in ordered_indices),
            belief,
            query_times,
        )
        learned = trajectories[0]
        positions = learned.positions.clone()
        candidate_positions = torch.stack([item.positions for item in trajectories], dim=-1)
        local_index = torch.empty(
            len(self.pool.dynamics_models),
            device=belief.device,
            dtype=torch.int64,
        )
        for local, external in enumerate(ordered_indices):
            local_index[external] = local
        for axis in self.axis_independent_axes:
            selected = local_index[choices[:, axis]].view(belief.batch_size, 1, 1, 1, 1)
            positions[..., axis] = torch.gather(
                candidate_positions[..., axis, :], -1, selected.expand(
                    belief.batch_size, positions.shape[1], positions.shape[2], 1, 1
                ).squeeze(-2),
            ).squeeze(-1)
        return BeliefTrajectory(
            timestamps=learned.timestamps,
            positions=positions,
            velocities=learned.velocities,
            orientations=learned.orientations,
            motion_mode_logits=learned.motion_mode_logits,
            fast_log_variance=learned.fast_log_variance,
            active_mask=learned.active_mask,
            event_logits=learned.event_logits,
            auxiliary={
                **learned.auxiliary,
                "hypothesis_axis_index": choices.detach().clone(),
                "hypothesis_rollout_candidate_indices": torch.tensor(
                    ordered_indices,
                    dtype=torch.int64,
                    device=belief.device,
                ),
            },
        ).validate()
