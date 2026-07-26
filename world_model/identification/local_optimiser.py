"""Optional bounded recent-window parameter refinement, disabled by default."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LocalOptimiserConfig:
    enabled: bool = False
    iterations: int = 3
    learning_rate: float = 0.02
    prior_weight: float = 0.1


class LocalParameterOptimiser:
    """Refine only a supplied small parameter tensor with first-order updates."""

    def __init__(self, config: LocalOptimiserConfig | None = None) -> None:
        self.config = config or LocalOptimiserConfig()

    def refine(
        self,
        initial: Tensor,
        objective: Callable[[Tensor], Tensor],
        *,
        lower: Tensor | float,
        upper: Tensor | float,
    ) -> Tensor:
        if not self.config.enabled:
            return initial
        parameter = initial.detach().clone().requires_grad_(True)
        prior = initial.detach()
        for _ in range(self.config.iterations):
            loss = (
                objective(parameter)
                + self.config.prior_weight * (parameter - prior).square().mean()
            )
            gradient = torch.autograd.grad(loss, parameter, create_graph=False)[0]
            parameter = (
                (parameter - self.config.learning_rate * gradient)
                .clamp(min=lower, max=upper)
                .detach()
                .requires_grad_(True)
            )
        return parameter.detach()
