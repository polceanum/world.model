"""Causal state, rollout, uncertainty, and parameter losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mean(
    value: Tensor,
    mask: Tensor,
    *,
    batch_macro: bool = False,
) -> Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    if batch_macro:
        if value.ndim == 0:
            raise ValueError("batch-macro masked mean requires a batch dimension")
        flat_value = value.reshape(value.shape[0], -1)
        flat_mask = expanded.reshape(value.shape[0], -1)
        row_count = flat_mask.sum(dim=-1)
        supported_row = row_count > 0
        if not supported_row.any():
            return value.sum() * 0
        row_sum = torch.where(flat_mask, flat_value, torch.zeros_like(flat_value)).sum(dim=-1)
        row_mean = row_sum / row_count.clamp_min(1).to(value.dtype)
        return row_mean.masked_select(supported_row).mean()
    return value.masked_select(expanded).mean() if expanded.any() else value.sum() * 0


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_macro: bool = False,
) -> Tensor:
    return masked_mean(
        F.smooth_l1_loss(prediction, target, reduction="none"),
        mask,
        batch_macro=batch_macro,
    )


def gaussian_nll(
    mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
    *,
    detach_mean_error: bool = False,
    batch_macro: bool = False,
) -> Tensor:
    log_variance = log_variance.clamp(-12.0, 8.0)
    squared_error = (mean - target).square()
    if detach_mean_error:
        # Calibration losses should not duplicate an already-supervised mean
        # gradient with an inverse-variance multiplier. The observed error is
        # retained for the variance gradient; only its path into ``mean`` is
        # stopped.
        squared_error = squared_error.detach()
    term = 0.5 * (squared_error * (-log_variance).exp() + log_variance)
    return masked_mean(term, mask, batch_macro=batch_macro)


def correction_error(
    prediction: Tensor,
    target: Tensor,
    *,
    axiswise: bool = False,
) -> Tensor:
    """Return legacy vector error or absolute per-coordinate correction error."""

    if prediction.shape != target.shape:
        raise ValueError("correction prediction and target must have matching shapes")
    if prediction.ndim == 0:
        raise ValueError("correction error requires a coordinate dimension")
    difference = prediction - target
    if axiswise:
        return difference.abs()
    return torch.linalg.vector_norm(difference, dim=-1)


def posterior_improvement_hinge(
    posterior_error: Tensor,
    prior_error: Tensor,
    mask: Tensor,
    *,
    margin: float = 0.0,
    batch_macro: bool = False,
) -> Tensor:
    """Penalise corrections that fail to improve on the incoming prior.

    The prior is a fixed reference for this objective. Detaching it prevents
    the model from satisfying the relative loss by deliberately making its
    pre-observation prediction worse. A positive margin requests a minimum
    improvement while retaining the absolute posterior state/rollout losses.
    """

    if margin < 0:
        raise ValueError("posterior improvement margin must be nonnegative")
    if posterior_error.shape != prior_error.shape:
        raise ValueError("posterior and prior errors must have matching shapes")
    return masked_mean(
        F.relu(posterior_error - prior_error.detach() + float(margin)),
        mask,
        batch_macro=batch_macro,
    )


def balanced_binary_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    maximum_positive_weight: float = 10.0,
) -> Tensor:
    """BCE with a bounded, batch-observed weight for rare positive events."""

    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError("logits, target, and mask must have matching shapes")
    if maximum_positive_weight < 1:
        raise ValueError("maximum_positive_weight must be at least one")
    selected_logits = logits.masked_select(mask)
    selected_target = target.to(logits.dtype).masked_select(mask)
    if selected_logits.numel() == 0:
        return logits.sum() * 0
    positive_count = selected_target.sum()
    negative_count = selected_target.numel() - positive_count
    positive_weight = torch.where(
        positive_count > 0,
        (negative_count / positive_count.clamp_min(1)).clamp(
            min=1.0,
            max=float(maximum_positive_weight),
        ),
        positive_count.new_ones(()),
    )
    return F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_target,
        pos_weight=positive_weight,
    )


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
