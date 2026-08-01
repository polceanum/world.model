"""Association cost construction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.observations.measurements import MeasurementSet, PredictedMeasurements


def _expanded_log_variance(log_variance: Tensor, values: Tensor) -> Tensor:
    try:
        return torch.broadcast_to(log_variance, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"log variance shape {tuple(log_variance.shape)} cannot broadcast to "
            f"values shape {tuple(values.shape)}"
        ) from exc


def geometry_mahalanobis_cost(
    measured: MeasurementSet,
    predicted: PredictedMeasurements,
    *,
    dimensions: int = 4,
    variance_floor: float = 1.0e-5,
) -> tuple[Tensor, Tensor]:
    """Return ``[B,N,M]`` diagonal Mahalanobis cost and squared distance."""

    dims = min(dimensions, measured.values.shape[-1], predicted.values.shape[-1])
    if dims <= 0:
        raise ValueError("at least one common measurement dimension is required")
    measured_values = measured.values[..., :dims]
    predicted_values = predicted.values[..., :dims]
    measured_lv = _expanded_log_variance(measured.log_variance, measured.values)[..., :dims]
    predicted_lv = _expanded_log_variance(predicted.log_variance, predicted.values)[..., :dims]
    residual = measured_values[:, None, :, :] - predicted_values[:, :, None, :]
    variance = (measured_lv[:, None, :, :].exp() + predicted_lv[:, :, None, :].exp()).clamp_min(
        variance_floor
    )
    squared = (residual.square() / variance).sum(dim=-1)
    # Mean rather than sum keeps thresholds comparable if dimensions change.
    return squared / float(dims), squared


def appearance_cosine_cost(
    measured: MeasurementSet,
    predicted: PredictedMeasurements,
) -> Tensor:
    batch, objects = predicted.valid_mask.shape
    measurements = measured.measurement_mask.shape[1]
    if measured.appearance is None or predicted.appearance is None:
        return predicted.values.new_zeros((batch, objects, measurements))
    dims = min(measured.appearance.shape[-1], predicted.appearance.shape[-1])
    measured_appearance = F.normalize(measured.appearance[..., :dims], dim=-1)
    predicted_appearance = F.normalize(predicted.appearance[..., :dims], dim=-1)
    similarity = torch.einsum("bnd,bmd->bnm", predicted_appearance, measured_appearance)
    return 1.0 - similarity


def existence_cost(
    measured: MeasurementSet,
    predicted: PredictedMeasurements,
) -> Tensor:
    del predicted
    probability = measured.existence_logits.sigmoid().clamp_min(1.0e-6)
    return -probability.log()[:, None, :]


def build_cost_matrix(
    measured: MeasurementSet,
    predicted: PredictedMeasurements,
    *,
    geometry_weight: float = 1.0,
    appearance_weight: float = 0.25,
    existence_weight: float = 0.05,
    geometry_dimensions: int = 4,
    mahalanobis_gate: float = 25.0,
    minimum_measurement_confidence: float = 0.0,
) -> tuple[Tensor, Tensor]:
    geometry, squared_mahalanobis = geometry_mahalanobis_cost(
        measured,
        predicted,
        dimensions=geometry_dimensions,
    )
    appearance = appearance_cosine_cost(measured, predicted)
    existence = existence_cost(measured, predicted)
    cost = (
        geometry_weight * geometry + appearance_weight * appearance + existence_weight * existence
    )
    possible = (
        predicted.valid_mask[:, :, None]
        & measured.measurement_mask[:, None, :]
        & (measured.existence_logits.sigmoid()[:, None, :] >= minimum_measurement_confidence)
        & (squared_mahalanobis <= mahalanobis_gate)
    )
    return cost.masked_fill(~possible, torch.inf), squared_mahalanobis
