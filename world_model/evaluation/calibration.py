"""Diagonal-Gaussian uncertainty calibration diagnostics."""

from __future__ import annotations

import math

import torch
from torch import Tensor

_NORMAL_QUANTILES = {
    0.50: 0.67448975,
    0.80: 1.28155157,
    0.90: 1.64485363,
    0.95: 1.95996398,
}


def gaussian_calibration_metrics(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
    mask: Tensor,
) -> dict[str, float]:
    log_variance = log_variance.clamp(-12.0, 8.0)
    standard_deviation = (0.5 * log_variance).exp()
    absolute_z = (mean - target).abs() / standard_deviation.clamp_min(1.0e-8)
    expanded = mask
    while expanded.ndim < mean.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(mean)
    z = absolute_z.masked_select(expanded)
    errors = (mean - target).masked_select(expanded)
    std = standard_deviation.masked_select(expanded)
    if z.numel() == 0:
        return {"gaussian_nll": float("nan"), "sharpness_std": float("nan")}
    variance = std.square()
    nll = 0.5 * (
        errors.square() / variance.clamp_min(1.0e-8)
        + variance.clamp_min(1.0e-8).log()
        + math.log(2 * math.pi)
    )
    result = {
        "gaussian_nll": float(nll.mean()),
        "sharpness_std": float(std.mean()),
    }
    for level, quantile in _NORMAL_QUANTILES.items():
        result[f"coverage_{int(level * 100)}"] = float((z <= quantile).float().mean())
    if errors.numel() > 1 and std.std() > 0 and errors.abs().std() > 0:
        result["uncertainty_error_correlation"] = float(
            torch.corrcoef(torch.stack((std, errors.abs())))[0, 1]
        )
    else:
        result["uncertainty_error_correlation"] = float("nan")
    return result
