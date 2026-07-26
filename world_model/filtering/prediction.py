"""Timestamp-aware filter prediction wrapper."""

from __future__ import annotations

from torch import Tensor, nn

from world_model.belief import WorldBelief


class BeliefPredictor(nn.Module):
    """Delegates state and covariance prediction to the hybrid dynamics model."""

    def __init__(self, dynamics: nn.Module) -> None:
        super().__init__()
        self.dynamics = dynamics

    def forward(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
    ) -> WorldBelief:
        predicted = self.dynamics.predict(belief, dt)
        if predicted is belief:
            raise RuntimeError("dynamics.predict must return a new WorldBelief")
        return predicted

    predict = forward
