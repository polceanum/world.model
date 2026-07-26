"""Batch collation for nested fixed-shape synthetic episode records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

import torch
from torch import Tensor


def _collate(values: Sequence[Any], path: str) -> Any:
    first = values[0]
    if isinstance(first, Tensor):
        if any(not isinstance(value, Tensor) or value.shape != first.shape for value in values[1:]):
            raise ValueError(f"inconsistent tensor shapes at {path}")
        return torch.stack(tuple(values), dim=0)
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(tuple(value) != keys for value in values[1:]):
            raise ValueError(f"inconsistent mapping keys at {path}")
        return {
            key: _collate(
                [value[key] for value in values],
                f"{path}.{key}" if path else str(key),
            )
            for key in keys
        }
    if isinstance(first, bool):
        return torch.tensor(values, dtype=torch.bool)
    if isinstance(first, int):
        return torch.tensor(values, dtype=torch.int64)
    if isinstance(first, float):
        return torch.tensor(values, dtype=torch.float32)
    if isinstance(first, str):
        return list(values)
    if isinstance(first, tuple):
        if not all(value == first for value in values):
            raise ValueError(f"metadata tuple differs across batch at {path}")
        return first
    if first is None:
        if not all(value is None for value in values):
            raise ValueError(f"mixed None/non-None values at {path}")
        return None
    if isinstance(first, Number):
        return torch.as_tensor(values)
    return list(values)


def collate_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack a non-empty sequence of canonical episodes batch-major.

    Tensor fields become ``[B, T, ...]``.  Numeric scalar fields become
    ``[B]`` tensors, while descriptive metadata strings remain Python lists.
    """

    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    return _collate(list(episodes), "")
