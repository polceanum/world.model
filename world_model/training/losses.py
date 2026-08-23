"""Causal state, rollout, uncertainty, and parameter losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _validated_batch_tail_fraction(batch_tail_fraction: float) -> float:
    if isinstance(batch_tail_fraction, bool) or not isinstance(
        batch_tail_fraction,
        (float, int),
    ):
        raise TypeError("batch_tail_fraction must be a real number")
    fraction = float(batch_tail_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("batch_tail_fraction must lie in (0, 1]")
    return fraction


def _validated_positive_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a real number")
    typed = float(value)
    if not math.isfinite(typed) or typed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return typed


def _supported_row_tail_mean(
    row_mean: Tensor,
    supported_row: Tensor,
    *,
    batch_tail_fraction: float,
) -> Tensor:
    """Average the highest-loss supported batch rows.

    Scenario-balanced causal batches place one episode from each declared
    regime on the batch axis.  A tail mean therefore prevents an improvement
    in several easy regimes from cancelling a regression in the hardest
    supported regimes.  Unsupported rows are omitted before the deterministic
    top-k selection.
    """

    fraction = _validated_batch_tail_fraction(batch_tail_fraction)
    selected = row_mean.masked_select(supported_row)
    if selected.numel() == 0:
        return row_mean.sum() * 0
    tail_count = max(1, math.ceil(selected.numel() * fraction))
    return selected.topk(tail_count, largest=True, sorted=False).values.mean()


def masked_mean(
    value: Tensor,
    mask: Tensor,
    *,
    batch_macro: bool = False,
    batch_tail_fraction: float | None = None,
) -> Tensor:
    if batch_tail_fraction is not None:
        batch_tail_fraction = _validated_batch_tail_fraction(batch_tail_fraction)
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    if batch_macro or batch_tail_fraction is not None:
        if value.ndim == 0:
            raise ValueError("batch-row masked mean requires a batch dimension")
        flat_value = value.reshape(value.shape[0], -1)
        flat_mask = expanded.reshape(value.shape[0], -1)
        row_count = flat_mask.sum(dim=-1)
        supported_row = row_count > 0
        if not supported_row.any():
            return value.sum() * 0
        row_sum = torch.where(flat_mask, flat_value, torch.zeros_like(flat_value)).sum(dim=-1)
        row_mean = row_sum / row_count.clamp_min(1).to(value.dtype)
        if batch_tail_fraction is not None:
            return _supported_row_tail_mean(
                row_mean,
                supported_row,
                batch_tail_fraction=batch_tail_fraction,
            )
        return row_mean.masked_select(supported_row).mean()
    return value.masked_select(expanded).mean() if expanded.any() else value.sum() * 0


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    batch_macro: bool = False,
    batch_tail_fraction: float | None = None,
) -> Tensor:
    return masked_mean(
        F.smooth_l1_loss(prediction, target, reduction="none"),
        mask,
        batch_macro=batch_macro,
        batch_tail_fraction=batch_tail_fraction,
    )


def gaussian_nll(
    mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
    *,
    detach_mean_error: bool = False,
    batch_macro: bool = False,
    batch_tail_fraction: float | None = None,
    standardized_error_gradient_cap: float | None = None,
) -> Tensor:
    log_variance = log_variance.clamp(-12.0, 8.0)
    squared_error = (mean - target).square()
    if detach_mean_error:
        # Calibration losses should not duplicate an already-supervised mean
        # gradient with an inverse-variance multiplier. The observed error is
        # retained for the variance gradient; only its path into ``mean`` is
        # stopped.
        squared_error = squared_error.detach()
    standardized_error = squared_error * (-log_variance).exp()
    if standardized_error_gradient_cap is not None:
        cap = _validated_positive_finite_real(
            standardized_error_gradient_cap,
            name="standardized_error_gradient_cap",
        )
        # Keep the exact proper-score value while robustifying only backward.
        # Above the cap, the logarithmic surrogate has derivative cap/x, so
        # the standardized-error contribution to the log-variance gradient is
        # bounded by cap instead of growing without limit.
        surrogate = torch.where(
            standardized_error <= cap,
            standardized_error,
            cap + cap * torch.log(standardized_error.clamp_min(cap) / cap),
        )
        standardized_error = standardized_error.detach() + surrogate - surrogate.detach()
    term = 0.5 * (standardized_error + log_variance)
    return masked_mean(
        term,
        mask,
        batch_macro=batch_macro,
        batch_tail_fraction=batch_tail_fraction,
    )


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
    batch_tail_fraction: float | None = None,
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
        batch_tail_fraction=batch_tail_fraction,
    )


def balanced_binary_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    maximum_positive_weight: float = 10.0,
    batch_tail_fraction: float | None = None,
) -> Tensor:
    """BCE with a bounded, batch-observed weight for rare positive events."""

    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError("logits, target, and mask must have matching shapes")
    if maximum_positive_weight < 1:
        raise ValueError("maximum_positive_weight must be at least one")
    if batch_tail_fraction is not None:
        batch_tail_fraction = _validated_batch_tail_fraction(batch_tail_fraction)
    typed_target = target.to(logits.dtype)
    selected_logits = logits.masked_select(mask)
    selected_target = typed_target.masked_select(mask)
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
    if batch_tail_fraction is None:
        return F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_target,
            pos_weight=positive_weight,
        )
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        typed_target,
        pos_weight=positive_weight,
        reduction="none",
    )
    return masked_mean(
        elementwise,
        mask,
        batch_tail_fraction=batch_tail_fraction,
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
