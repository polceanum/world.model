"""Transparent state baselines using the shared physical tensor contract."""

from __future__ import annotations

import torch
from torch import Tensor


def static_positions(position: Tensor, query_times: Tensor) -> Tensor:
    """Repeat current ``[B,N,3]`` positions at ``[B,T]`` relative times."""

    return position[:, None].expand(-1, query_times.shape[1], -1, -1).clone()


def constant_velocity_positions(
    position: Tensor,
    velocity: Tensor,
    query_times: Tensor,
) -> Tensor:
    """Constant-velocity rollout in physical units."""

    return position[:, None] + query_times[:, :, None, None] * velocity[:, None]


def analytic_gravity_drag_positions(
    position: Tensor,
    velocity: Tensor,
    query_times: Tensor,
    *,
    gravity: Tensor,
    drag: Tensor,
) -> Tensor:
    """Closed-form free-motion baseline; collisions are intentionally absent."""

    time = query_times[:, :, None, None]
    coefficient = drag[:, None].clamp_min(1.0e-6)
    decay = torch.exp(-coefficient * time)
    terminal = gravity[:, None, None] / coefficient
    displacement = (velocity[:, None] - terminal) * (1.0 - decay) / coefficient
    displacement = displacement + terminal * time
    return position[:, None] + displacement


def baseline_bundle(
    position: Tensor,
    velocity: Tensor,
    query_times: Tensor,
    *,
    gravity: Tensor,
    default_drag: float = 0.05,
    oracle_drag: Tensor | None = None,
) -> dict[str, Tensor]:
    batch, objects = position.shape[:2]
    default = position.new_full((batch, objects, 1), default_drag)
    bundle = {
        "static": static_positions(position, query_times),
        "constant_velocity": constant_velocity_positions(position, velocity, query_times),
        "analytic_default": analytic_gravity_drag_positions(
            position,
            velocity,
            query_times,
            gravity=gravity,
            drag=default,
        ),
    }
    if oracle_drag is not None:
        bundle["analytic_oracle_parameter"] = analytic_gravity_drag_positions(
            position,
            velocity,
            query_times,
            gravity=gravity,
            drag=oracle_drag,
        )
    return bundle
