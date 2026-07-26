"""Filter-side uncertainty expansion and clamping."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model.belief import WorldBelief, clamp_log_variance


@dataclass(frozen=True)
class FilterUncertaintyConfig:
    missed_fast_variance_increment: float = 0.05
    ambiguous_variance_increment: float = 0.02
    minimum_log_variance: float = -12.0
    maximum_log_variance: float = 8.0


class FilterUncertainty:
    def __init__(self, config: FilterUncertaintyConfig | None = None) -> None:
        self.config = config or FilterUncertaintyConfig()

    def missed(self, belief: WorldBelief, missed_mask: torch.Tensor) -> WorldBelief:
        if missed_mask.shape != belief.objects.active.shape:
            raise ValueError("missed_mask must have shape [B,N]")
        variance = belief.objects.fast_log_variance.exp()
        increment = (
            missed_mask.unsqueeze(-1).to(variance.dtype)
            * self.config.missed_fast_variance_increment
        )
        log_variance = (variance + increment).clamp_min(1.0e-10).log()
        log_variance = clamp_log_variance(
            log_variance,
            (
                self.config.minimum_log_variance,
                self.config.maximum_log_variance,
            ),
        )
        objects = belief.objects.replace(fast_log_variance=log_variance)
        return belief.replace(objects=objects)

    def clamp(self, belief: WorldBelief) -> WorldBelief:
        objects = belief.objects
        objects = objects.replace(
            fast_log_variance=clamp_log_variance(
                objects.fast_log_variance,
                (
                    self.config.minimum_log_variance,
                    self.config.maximum_log_variance,
                ),
            ),
            slow_log_variance=clamp_log_variance(
                objects.slow_log_variance,
                (
                    self.config.minimum_log_variance,
                    self.config.maximum_log_variance,
                ),
            ),
        )
        return belief.replace(objects=objects)
