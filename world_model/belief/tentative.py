"""Transient evidence for lifecycle birth confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TentativeBirthState:
    """Bounded sensor-local evidence for objects not yet in ``WorldBelief``.

    Tentative detections are observation history, not confirmed physical
    state. They carry no permanent object ID and never participate in
    dynamics, filtering, or rollouts. Once confirmed, the lifecycle allocates
    a normal monotonic ID in ``WorldBelief`` and removes the tentative record.
    """

    world_position: Tensor
    active: Tensor
    confirmation_count: Tensor
    timestamp: Tensor

    def validate(self) -> TentativeBirthState:
        if self.world_position.ndim != 3 or self.world_position.shape[-1] != 3:
            raise ValueError("tentative world_position must have shape [B,M,3]")
        shape = self.world_position.shape[:2]
        if self.active.shape != shape or self.active.dtype is not torch.bool:
            raise ValueError("tentative active must be boolean [B,M]")
        if (
            self.confirmation_count.shape != shape
            or self.confirmation_count.dtype is not torch.int64
        ):
            raise ValueError("tentative confirmation_count must be int64 [B,M]")
        if self.timestamp.shape != shape:
            raise ValueError("tentative timestamp must have shape [B,M]")
        if self.timestamp.dtype != self.world_position.dtype:
            raise TypeError("tentative timestamp and world_position dtypes must match")
        return self

    def detach(self) -> TentativeBirthState:
        return TentativeBirthState(
            world_position=self.world_position.detach(),
            active=self.active.detach(),
            confirmation_count=self.confirmation_count.detach(),
            timestamp=self.timestamp.detach(),
        )
