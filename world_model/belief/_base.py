"""Shared copy/device helpers for tensor-bearing belief dataclasses."""

from __future__ import annotations

from typing import TypeVar

import torch

from world_model.utils.tensors import clone_tensors, detach_tensors, move_tensors

BeliefT = TypeVar("BeliefT")


class TensorDataclassMixin:
    """Functional helpers used by the persistent belief contracts.

    The methods deliberately return a new dataclass.  Runtime code can therefore
    carry a numerical belief across truncated training windows without exposing
    accidental in-place mutation.
    """

    def clone(self: BeliefT) -> BeliefT:
        return clone_tensors(self)

    def detach(self: BeliefT) -> BeliefT:
        return detach_tensors(self)

    def to(
        self: BeliefT,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BeliefT:
        return move_tensors(self, device=device, dtype=dtype)
