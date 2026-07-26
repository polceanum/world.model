"""Rollout query alignment for frame-labelled event supervision.

Simulator event labels at frame ``t`` describe the observation interval
``[t - dt_obs, t]``.  Rollout event logits describe the segment since the
previous rollout query.  The query plan below inserts the start of every
labelled observation interval so selecting its endpoint yields an exactly
aligned event forecast.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ObservationWindowQueryPlan:
    """Expanded rollout queries plus indices for requested target frames."""

    target_frame_offsets: tuple[int, ...]
    target_seconds: tuple[float, ...]
    query_frame_offsets: tuple[int, ...]
    query_seconds: tuple[float, ...]
    target_query_indices: tuple[int, ...]

    def select_target_endpoints(self, values: Tensor) -> Tensor:
        """Select values at target-window endpoints from a rollout time axis.

        ``values`` must begin with ``[B,Q,...]``, where ``Q`` is the expanded
        query count.  For event logits, each selected value therefore covers
        exactly the preceding observation interval rather than a cumulative
        prefix or the arbitrary gap between requested forecast horizons.
        """

        if values.ndim < 2:
            raise ValueError("rollout values must begin with [B,Q] axes")
        if values.shape[1] != len(self.query_frame_offsets):
            raise ValueError("rollout query axis does not match the observation-window query plan")
        indices = torch.as_tensor(
            self.target_query_indices,
            device=values.device,
            dtype=torch.int64,
        )
        return values.index_select(1, indices)


def observation_window_query_plan(
    target_frame_offsets: Sequence[int],
    *,
    frame_rate: float,
) -> ObservationWindowQueryPlan:
    """Bracket each target frame by its exact preceding observation window."""

    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate must be finite and positive")
    try:
        targets = tuple(operator.index(offset) for offset in target_frame_offsets)
    except TypeError as error:
        raise TypeError("target frame offsets must be integers") from error
    if any(offset < 1 for offset in targets):
        raise ValueError("target frame offsets must be positive")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("target frame offsets must be unique and sorted")

    query_offsets = tuple(
        sorted({boundary for target in targets for boundary in (target - 1, target)})
    )
    target_indices = tuple(query_offsets.index(target) for target in targets)
    for target, query_index in zip(targets, target_indices, strict=True):
        if query_index == 0 or query_offsets[query_index - 1] != target - 1:
            raise AssertionError("target query is not preceded by its observation-window start")

    return ObservationWindowQueryPlan(
        target_frame_offsets=targets,
        target_seconds=tuple(offset / frame_rate for offset in targets),
        query_frame_offsets=query_offsets,
        query_seconds=tuple(offset / frame_rate for offset in query_offsets),
        target_query_indices=target_indices,
    )


__all__ = [
    "ObservationWindowQueryPlan",
    "observation_window_query_plan",
]
