"""Causal state, rollout, uncertainty, and parameter losses."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    return value.masked_select(expanded).mean() if expanded.any() else value.sum() * 0


def masked_huber(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return masked_mean(F.smooth_l1_loss(prediction, target, reduction="none"), mask)


def gaussian_nll(
    mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
) -> Tensor:
    log_variance = log_variance.clamp(-12.0, 8.0)
    term = 0.5 * ((mean - target).square() * (-log_variance).exp() + log_variance)
    return masked_mean(term, mask)


def state_losses(
    position: Tensor,
    velocity: Tensor,
    target_position: Tensor,
    target_velocity: Tensor,
    active_mask: Tensor,
) -> dict[str, Tensor]:
    return {
        "state_position": masked_huber(position, target_position, active_mask),
        "state_velocity": masked_huber(velocity, target_velocity, active_mask),
    }


def weighted_total(
    terms: dict[str, Tensor],
    weights: dict[str, float],
) -> Tensor:
    """Combine named terms by exact name or their prefix before the first `_`."""

    if not terms:
        raise ValueError("at least one loss term is required")
    total = next(iter(terms.values())).new_zeros(())
    for name, value in terms.items():
        prefix = name.split("_", 1)[0]
        weight = weights.get(name, weights.get(prefix, 1.0))
        total = total + float(weight) * value
    return total
