"""Rigid-transform helpers using ``T_target_from_source`` matrices."""

from __future__ import annotations

import torch


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert batched homogeneous rigid transforms ``[..., 4, 4]``."""

    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [...,4,4] transform, got {tuple(transform.shape)}")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    rotation_t = rotation.transpose(-1, -2)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3] = -(rotation_t @ translation.unsqueeze(-1)).squeeze(-1)
    inverse[..., 3, 3] = 1
    return inverse


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a broadcast-compatible homogeneous transform to 3-D points."""

    if transform.shape[-2:] != (4, 4) or points.shape[-1] != 3:
        raise ValueError("transform_points expects [...,4,4] and [...,3]")
    return (transform[..., :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + transform[..., :3, 3]
