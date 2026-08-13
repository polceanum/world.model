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

from world_model.belief import BeliefTrajectory, WorldBelief
from world_model.dynamics.rollout import RolloutEngine, RolloutStep


@dataclass(frozen=True)
class HypothesisSelection:
    """Evidence and posterior choice for a batch of candidate rollouts."""

    scores: Tensor
    selected_index: Tensor
    posterior_weights: Tensor

    def validate(self) -> HypothesisSelection:
        if self.scores.ndim != 2:
            raise ValueError("hypothesis scores must have shape [B,H]")
        if self.selected_index.shape != (self.scores.shape[0],):
            raise ValueError("selected_index must have shape [B]")
        if self.posterior_weights.shape != self.scores.shape:
            raise ValueError("posterior_weights must match scores")
        if self.selected_index.dtype != torch.int64:
            raise TypeError("selected_index must use torch.int64")
        if not torch.isfinite(self.scores).all() or not torch.isfinite(
            self.posterior_weights
        ).all():
            raise ValueError("hypothesis selection contains NaN or Inf")
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

    @staticmethod
    def score(
        trajectories: Sequence[BeliefTrajectory],
        target_positions: Tensor,
        target_mask: Tensor,
        *,
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
        if not torch.isfinite(target_positions).all():
            raise ValueError("target_positions contains NaN or Inf")

        mask = target_mask.unsqueeze(-1)
        valid_count = mask.sum(dim=(1, 2, 3)).clamp_min(1).to(target_positions.dtype)
        candidate_scores: list[Tensor] = []
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
            candidate_scores.append((point_loss * mask).sum(dim=(1, 2, 3)) / valid_count)

        scores = torch.stack(candidate_scores, dim=-1)
        scale = torch.as_tensor(temperature, device=scores.device, dtype=scores.dtype)
        posterior_weights = torch.softmax(-scores / scale, dim=-1)
        selected_index = scores.argmin(dim=-1).to(torch.int64)
        return HypothesisSelection(scores, selected_index, posterior_weights).validate()

