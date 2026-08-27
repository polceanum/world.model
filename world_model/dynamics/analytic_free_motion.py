"""Parameter-free analytic-only belief dynamics for the RGB-D runtime rung."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import torch
from torch import Tensor, nn

from world_model.belief import BeliefTrajectory, MotionMode, WorldBelief
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.rollout import RolloutEngine, RolloutStep


class AnalyticFreeMotionDynamics(nn.Module):
    """Exact gravity-and-linear-drag propagation with no learned parameters.

    This intentionally excludes contacts, learned residuals, event transitions,
    and process-noise inflation.  It is the explicit runtime semantics for a
    contact-free world-model rung whose only trainable path is measurement and
    differentiable temporal estimation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.analytic = AnalyticKinematics()
        self.rollout_engine = RolloutEngine()

    def predict_step(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
    ) -> RolloutStep:
        elapsed = torch.as_tensor(
            dt,
            device=belief.timestamp.device,
            dtype=belief.timestamp.dtype,
        )
        if elapsed.ndim == 0:
            elapsed = elapsed.expand_as(belief.timestamp)
        if elapsed.shape != belief.timestamp.shape:
            raise ValueError("dt must be scalar or have shape [B]")
        after = self.analytic(belief.objects, belief.gravity, elapsed)
        propagated = after.active & (elapsed > 0.0).unsqueeze(-1)
        free_logits = after.motion_mode_logits.new_full(
            after.motion_mode_logits.shape,
            -4.0,
        )
        free_logits[..., MotionMode.FREE] = 4.0
        after = after.replace(
            motion_mode_logits=torch.where(
                propagated.unsqueeze(-1),
                free_logits,
                after.motion_mode_logits,
            )
        )
        endpoint = belief.replace(
            timestamp=belief.timestamp + elapsed,
            objects=after,
        )
        event_logits = after.motion_mode_logits.new_full(
            after.motion_mode_logits.shape,
            -4.0,
        )
        event_logits = torch.where(
            propagated.unsqueeze(-1),
            free_logits,
            event_logits,
        )
        return RolloutStep(
            belief=endpoint,
            event_logits=event_logits,
            auxiliary={},
        )

    def predict(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
    ) -> WorldBelief:
        return self.predict_step(belief, dt).belief

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        *,
        return_events: bool = True,
        return_auxiliary: bool = True,
        auxiliary_names: Collection[str] | None = None,
    ) -> BeliefTrajectory:
        return self.rollout_engine.rollout(
            self.predict_step,
            belief,
            query_times,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
            auxiliary_names=auxiliary_names,
        )
