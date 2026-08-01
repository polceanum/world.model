"""Innovation construction shared across observation modalities."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from world_model.fusion.association import AssociationResult
from world_model.observations.measurements import (
    InnovationSet,
    MeasurementSet,
    PredictedMeasurements,
)


def gather_pairs(values: Tensor, indices: Tensor, pair_mask: Tensor) -> Tensor:
    """Gather ``[B,L,...]`` values using ``[B,P]`` indices."""

    batch, pairs = indices.shape
    safe_indices = indices.clamp_min(0)
    batch_indices = torch.arange(batch, device=values.device)[:, None]
    gathered = values[batch_indices, safe_indices]
    extra_dims = (1,) * (gathered.ndim - 2)
    return gathered * pair_mask.reshape(batch, pairs, *extra_dims).to(gathered.dtype)


def _expand_variance(log_variance: Tensor, values: Tensor) -> Tensor:
    return torch.broadcast_to(log_variance, values.shape)


def _prediction_rows_for_associations(
    predicted: PredictedMeasurements,
    association: AssociationResult,
) -> Tensor:
    """Map persistent belief-slot assignments back to prediction-row order."""

    matches = predicted.belief_indices[:, :, None] == association.belief_indices[:, None, :]
    matches = matches & predicted.valid_mask[:, :, None] & association.pair_mask[:, None, :]
    match_count = matches.sum(dim=1)
    invalid = association.pair_mask & (match_count != 1)
    if bool(invalid.any()):
        raise ValueError("each associated belief slot must map to exactly one valid predicted row")
    return matches.to(torch.int64).argmax(dim=1)


def build_innovation(
    *,
    measured: MeasurementSet,
    predicted: PredictedMeasurements,
    association: AssociationResult,
    modality_index: int,
    clip_whitened: float = 20.0,
) -> InnovationSet:
    dims = min(measured.values.shape[-1], predicted.values.shape[-1])
    pair_mask = association.pair_mask
    prediction_indices = _prediction_rows_for_associations(predicted, association)
    measured_values = gather_pairs(
        measured.values[..., :dims],
        association.measurement_indices,
        pair_mask,
    )
    predicted_values = gather_pairs(
        predicted.values[..., :dims],
        prediction_indices,
        pair_mask,
    )
    measured_lv = gather_pairs(
        _expand_variance(measured.log_variance, measured.values)[..., :dims],
        association.measurement_indices,
        pair_mask,
    )
    predicted_lv = gather_pairs(
        _expand_variance(predicted.log_variance, predicted.values)[..., :dims],
        prediction_indices,
        pair_mask,
    )
    variance = (measured_lv.exp() + predicted_lv.exp()).clamp_min(1.0e-8)
    residual = measured_values - predicted_values
    whitened = residual / variance.sqrt()
    whitened = whitened.clamp(min=-clip_whitened, max=clip_whitened)
    residual = residual * pair_mask.unsqueeze(-1)
    whitened = whitened * pair_mask.unsqueeze(-1)
    norm = torch.linalg.vector_norm(whitened, dim=-1)
    log_likelihood = -0.5 * (whitened.square() + variance.log() + math.log(2.0 * math.pi)).sum(
        dim=-1
    )
    log_likelihood = torch.where(pair_mask, log_likelihood, torch.zeros_like(log_likelihood))
    event_features = torch.stack(
        (
            norm,
            whitened.abs().amax(dim=-1),
            whitened.abs().mean(dim=-1),
            association.pair_cost.nan_to_num(posinf=0.0),
            association.ambiguous.to(whitened.dtype),
        ),
        dim=-1,
    )
    auxiliary: dict[str, Tensor] = {
        "measured_values": measured_values,
        "predicted_values": predicted_values,
        "measurement_log_variance": measured_lv,
        "predicted_log_variance": predicted_lv,
        "association_cost": association.pair_cost,
        "ambiguous": association.ambiguous,
    }
    for key, value in measured.auxiliary.items():
        if value.ndim >= 2 and value.shape[:2] == measured.values.shape[:2]:
            auxiliary[f"measured_{key}"] = gather_pairs(
                value, association.measurement_indices, pair_mask
            )
    for key, value in predicted.auxiliary.items():
        if value.ndim >= 2 and value.shape[:2] == predicted.values.shape[:2]:
            auxiliary[f"predicted_{key}"] = gather_pairs(value, prediction_indices, pair_mask)
    modality_indices = torch.full(
        pair_mask.shape,
        modality_index,
        dtype=torch.int64,
        device=pair_mask.device,
    )
    result = InnovationSet(
        modality=measured.modality,
        residual=residual,
        whitened_residual=whitened,
        innovation_norm=norm,
        belief_indices=association.belief_indices,
        measurement_indices=association.measurement_indices,
        pair_mask=pair_mask,
        log_likelihood=log_likelihood,
        modality_index=modality_indices,
        event_features=event_features,
        auxiliary=auxiliary,
    )
    result.validate()
    return result
