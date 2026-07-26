"""State and prior/posterior forecast metrics."""

from __future__ import annotations

from torch import Tensor


def _masked_values(value: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return value.masked_select(expanded.expand_as(value))


def masked_position_metrics(
    prediction: Tensor,
    target: Tensor,
    active_mask: Tensor,
) -> dict[str, float]:
    """Return physical-unit position RMSE/MAE over active object-times."""

    error = prediction - target
    values = _masked_values(error, active_mask)
    if values.numel() == 0:
        return {"position_rmse_m": float("nan"), "position_mae_m": float("nan")}
    return {
        "position_rmse_m": float(values.square().mean().sqrt()),
        "position_mae_m": float(values.abs().mean()),
    }


def metrics_by_horizon(
    prediction: Tensor,
    target: Tensor,
    active_mask: Tensor,
    horizons: Tensor,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for index in range(prediction.shape[1]):
        metrics = masked_position_metrics(
            prediction[:, index],
            target[:, index],
            active_mask[:, index],
        )
        horizon = float(horizons[0, index])
        for name, value in metrics.items():
            results[f"{name}@{horizon:.3f}s"] = value
    return results


def correction_improvement(
    prior_prediction: Tensor,
    posterior_prediction: Tensor,
    target: Tensor,
    active_mask: Tensor,
) -> dict[str, float]:
    """Measure whether a new observation improved future position error."""

    prior_per_item = (prior_prediction - target).square().sum(dim=-1).sqrt()
    posterior_per_item = (posterior_prediction - target).square().sum(dim=-1).sqrt()
    valid_prior = prior_per_item.masked_select(active_mask)
    valid_posterior = posterior_per_item.masked_select(active_mask)
    if valid_prior.numel() == 0:
        return {
            "prior_position_error_m": float("nan"),
            "posterior_position_error_m": float("nan"),
            "correction_improvement_m": float("nan"),
            "correction_improvement_fraction": float("nan"),
            "positive_correction_rate": float("nan"),
        }
    delta = valid_prior - valid_posterior
    prior_mean = valid_prior.mean()
    return {
        "prior_position_error_m": float(prior_mean),
        "posterior_position_error_m": float(valid_posterior.mean()),
        "correction_improvement_m": float(delta.mean()),
        "correction_improvement_fraction": float(delta.mean() / prior_mean.clamp_min(1.0e-8)),
        "positive_correction_rate": float((delta > 0).float().mean()),
    }
