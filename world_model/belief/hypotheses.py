"""Runtime wrapper for alternate world-belief hypotheses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.belief.world_belief import WorldBelief


@dataclass
class HypothesisSet:
    """A small list of beliefs with batch-wise log weights ``[B,H]``."""

    beliefs: list[WorldBelief]
    log_weights: Tensor

    def validate(self) -> HypothesisSet:
        if not self.beliefs:
            raise ValueError("HypothesisSet requires at least one belief")
        batch = self.beliefs[0].batch_size
        if self.log_weights.shape != (batch, len(self.beliefs)):
            raise ValueError("log_weights must have shape [B,H]")
        if not torch.isfinite(self.log_weights).all():
            raise ValueError("hypothesis log weights must be finite")
        if not self.log_weights.is_floating_point():
            raise TypeError("hypothesis log weights must be floating point")
        for belief in self.beliefs:
            belief.validate()
            if belief.batch_size != batch:
                raise ValueError("all hypotheses must have the same batch size")
            if belief.device != self.log_weights.device:
                raise ValueError("hypothesis weights and beliefs must share a device")
            if belief.dtype != self.log_weights.dtype:
                raise ValueError("hypothesis weights and beliefs must share a dtype")
        return self

    @classmethod
    def singleton(cls, belief: WorldBelief) -> HypothesisSet:
        return cls(
            beliefs=[belief],
            log_weights=torch.zeros(
                belief.batch_size,
                1,
                device=belief.device,
                dtype=belief.dtype,
            ),
        )

    @property
    def normalized_weights(self) -> Tensor:
        return torch.softmax(self.log_weights, dim=-1)

    def reweight(self, log_likelihood: Tensor) -> HypothesisSet:
        if log_likelihood.shape != self.log_weights.shape:
            raise ValueError("log_likelihood must have shape [B,H]")
        updated = self.log_weights + log_likelihood
        updated = updated - torch.logsumexp(updated, dim=-1, keepdim=True)
        return HypothesisSet(self.beliefs.copy(), updated)

    def map(self, function: Callable[[WorldBelief], WorldBelief]) -> HypothesisSet:
        return HypothesisSet([function(item) for item in self.beliefs], self.log_weights)

    def clone(self) -> HypothesisSet:
        return HypothesisSet(
            [belief.clone() for belief in self.beliefs],
            self.log_weights.clone(),
        )

    def detach(self) -> HypothesisSet:
        return HypothesisSet(
            [belief.detach() for belief in self.beliefs],
            self.log_weights.detach(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HypothesisSet:
        return HypothesisSet(
            [belief.to(device=device, dtype=dtype) for belief in self.beliefs],
            self.log_weights.to(device=device, dtype=dtype),
        )

    def best(self, batch_index: int = 0) -> WorldBelief:
        """Return the highest-weight hypothesis for one batch element.

        Hypotheses contain whole batched beliefs, so this chooses one common
        hypothesis object.  Per-example branching can be added without changing
        the wrapper contract.
        """

        index = int(self.log_weights[batch_index].argmax().item())
        return self.beliefs[index]
