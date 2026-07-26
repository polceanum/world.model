"""Utilities for nested tensor-bearing dataclasses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from typing import TypeVar

import torch

T = TypeVar("T")


def map_tensors(value: T, function: Callable[[torch.Tensor], torch.Tensor]) -> T:
    """Recursively transform tensors while preserving dataclass/value structure."""

    if isinstance(value, torch.Tensor):
        return function(value)  # type: ignore[return-value]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            item.name: map_tensors(getattr(value, item.name), function) for item in fields(value)
        }
        return replace(value, **updates)
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}  # type: ignore[return-value]
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)  # type: ignore[return-value]
    return value


def detach_tensors(value: T) -> T:
    """Detach all tensors without altering numerical values or masks."""

    return map_tensors(value, torch.Tensor.detach)


def clone_tensors(value: T) -> T:
    """Deep-clone every tensor in a structured value."""

    return map_tensors(value, torch.Tensor.clone)


def move_tensors(
    value: T,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> T:
    """Move floating tensors while retaining integer and boolean dtypes."""

    def move(tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = dtype if tensor.is_floating_point() else None
        return tensor.to(device=device, dtype=target_dtype)

    return map_tensors(value, move)


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    """Raise an actionable error when a tensor contains NaN or infinity."""

    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
