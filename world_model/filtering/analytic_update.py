"""Closed-form diagonal Gaussian measurement proposals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DiagonalUpdateResult:
    mean: Tensor
    log_variance: Tensor
    gain: Tensor
    correction: Tensor


def robust_influence(
    whitened_residual: Tensor,
    *,
    clip_norm: float = 8.0,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Huber-like vector influence in ``[0,1]``."""

    norm = torch.linalg.vector_norm(whitened_residual, dim=-1, keepdim=True)
    return torch.minimum(
        torch.ones_like(norm),
        torch.as_tensor(
            clip_norm,
            device=norm.device,
            dtype=norm.dtype,
        )
        / norm.clamp_min(epsilon),
    )


def diagonal_kalman_update(
    prior_mean: Tensor,
    prior_log_variance: Tensor,
    measurement: Tensor,
    measurement_log_variance: Tensor,
    *,
    confidence: Tensor | float = 1.0,
    robust_clip_norm: float = 8.0,
    minimum_log_variance: float = -12.0,
    maximum_log_variance: float = 8.0,
) -> DiagonalUpdateResult:
    """Apply an uncertainty-weighted direct measurement update.

    Shapes are broadcast-compatible ``[...,D]``.  Robust influence only scales
    the mean correction; covariance contracts according to the effective gain.
    """

    if prior_mean.shape != measurement.shape:
        raise ValueError("prior mean and direct measurement shapes must match")
    prior_variance = prior_log_variance.exp().clamp_min(1.0e-10)
    measurement_variance = measurement_log_variance.exp().clamp_min(1.0e-10)
    residual = measurement - prior_mean
    total_variance = prior_variance + measurement_variance
    whitened = residual / total_variance.sqrt()
    influence = robust_influence(whitened, clip_norm=robust_clip_norm)
    confidence_tensor = torch.as_tensor(
        confidence,
        device=prior_mean.device,
        dtype=prior_mean.dtype,
    )
    if confidence_tensor.ndim == prior_mean.ndim - 1:
        confidence_tensor = confidence_tensor.unsqueeze(-1)
    confidence_tensor = confidence_tensor.clamp(0.0, 1.0)
    gain = prior_variance / total_variance
    effective_gain = gain * confidence_tensor * influence
    correction = effective_gain * residual
    posterior_mean = prior_mean + correction
    posterior_variance = (prior_variance * (1.0 - effective_gain)).clamp_min(1.0e-10)
    posterior_log_variance = posterior_variance.log().clamp(
        minimum_log_variance,
        maximum_log_variance,
    )
    return DiagonalUpdateResult(
        mean=posterior_mean,
        log_variance=posterior_log_variance,
        gain=effective_gain,
        correction=correction,
    )
