"""Robust innovation-cause assessment used for recovery scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from world_model.fusion.association import AssociationResult
from world_model.observations.measurements import InnovationSet


class InnovationCause(IntEnum):
    NOISE = 0
    STATE_DRIFT = 1
    PHYSICAL_EVENT = 2
    ASSOCIATION_ERROR = 3
    NEW_OBJECT = 4
    CAMERA_OR_SENSOR_SHIFT = 5
    UNKNOWN_MODEL_ERROR = 6


@dataclass
class SurpriseAssessment:
    cause_probabilities: Tensor
    robust_weight: Tensor
    trigger_global: Tensor
    aggregate_surprise: Tensor


class SurpriseClassifier:
    """Conservative deterministic classifier with no single hard decision."""

    def __init__(
        self,
        *,
        robust_clip: float = 8.0,
        global_threshold: float = 10.0,
    ) -> None:
        self.robust_clip = robust_clip
        self.global_threshold = global_threshold

    def __call__(
        self,
        innovation: InnovationSet,
        association: AssociationResult,
    ) -> SurpriseAssessment:
        norm = innovation.innovation_norm
        mask = innovation.pair_mask
        ambiguous = association.ambiguous.to(norm.dtype)
        probabilities = norm.new_zeros((*norm.shape, len(InnovationCause)))
        probabilities[..., InnovationCause.NOISE] = torch.exp(-0.5 * norm)
        probabilities[..., InnovationCause.STATE_DRIFT] = torch.sigmoid(norm - 2.0)
        probabilities[..., InnovationCause.PHYSICAL_EVENT] = torch.sigmoid(norm - 5.0)
        probabilities[..., InnovationCause.ASSOCIATION_ERROR] = ambiguous * torch.sigmoid(
            norm - 1.0
        )
        probabilities[..., InnovationCause.NEW_OBJECT] = 0.0
        probabilities[..., InnovationCause.CAMERA_OR_SENSOR_SHIFT] = torch.sigmoid(norm - 12.0)
        probabilities[..., InnovationCause.UNKNOWN_MODEL_ERROR] = torch.sigmoid(norm - 8.0)
        probabilities = probabilities + 1.0e-6
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        probabilities = probabilities * mask.unsqueeze(-1)
        robust_weight = torch.minimum(
            torch.ones_like(norm),
            self.robust_clip / norm.clamp_min(1.0e-6),
        )
        robust_weight = robust_weight * mask
        denominator = mask.sum(dim=-1).clamp_min(1)
        aggregate = (norm * mask).sum(dim=-1) / denominator
        unmatched = association.unmatched_measurements.sum(dim=-1)
        trigger = (aggregate >= self.global_threshold) | (unmatched > 0)
        return SurpriseAssessment(
            cause_probabilities=probabilities,
            robust_weight=robust_weight,
            trigger_global=trigger,
            aggregate_surprise=aggregate,
        )
