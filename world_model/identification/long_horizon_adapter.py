"""Stable neural residual around public causal physical estimates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.identification.evidence_transformer import (
    EvidenceTransformerConfig,
    NeuralPhysicsAdapter,
    PhysicsParameterPrediction,
    _parameter_bounds,
    normalized_parameter_targets,
)

_CANDIDATE_START = 28
_SUPPORT_START = 32


@dataclass(frozen=True, slots=True)
class LongHorizonAdapterConfig:
    """Configuration for a bounded correction around causal token estimates."""

    transformer: EvidenceTransformerConfig = EvidenceTransformerConfig()
    max_normalized_residual: float = 0.20

    def validate(self) -> LongHorizonAdapterConfig:
        self.transformer.validate()
        if not 0.0 < self.max_normalized_residual <= 0.5:
            raise ValueError("long-horizon residual bound must lie in (0,0.5]")
        return self


def causal_parameter_baseline(
    evidence: Tensor,
    valid: Tensor,
    config: EvidenceTransformerConfig,
) -> Tensor:
    """Aggregate public sufficient statistics into normalized parameter means."""

    if evidence.ndim != 3 or evidence.shape[-1] < _SUPPORT_START + 4:
        raise ValueError("causal parameter evidence must have shape [B,T,36]")
    if valid.shape != evidence.shape[:2] or valid.dtype is not torch.bool:
        raise ValueError("causal parameter validity must be boolean [B,T]")
    candidates = evidence[..., _CANDIDATE_START:_SUPPORT_START]
    support = evidence[..., _SUPPORT_START : _SUPPORT_START + 4]
    supported = support * valid.unsqueeze(-1)
    totals = (candidates * supported).sum(dim=1)
    counts = supported.sum(dim=1)
    mean = totals / counts.clamp_min(1.0)
    drag = (mean[:, 0] * 0.35).clamp(*config.drag_bounds)
    mass = (mean[:, 1] * 2.20).clamp(*config.mass_bounds)
    restitution = mean[:, 2].clamp(*config.restitution_bounds)
    friction = (mean[:, 3] * 0.75).clamp(*config.friction_bounds)
    belief_values = torch.stack(
        (
            mass.log(),
            drag.log(),
            torch.logit(restitution),
            torch.logit(friction),
        ),
        dim=-1,
    )
    normalized = normalized_parameter_targets(belief_values, config)
    return torch.where(counts > 0.0, normalized, torch.full_like(normalized, 0.5))


class LongHorizonPhysicsAdapter(NeuralPhysicsAdapter):
    """Evidence transformer that learns only a bounded causal-prior correction."""

    def __init__(self, config: LongHorizonAdapterConfig | None = None) -> None:
        self.long_horizon_config = (config or LongHorizonAdapterConfig()).validate()
        super().__init__(self.long_horizon_config.transformer)

    def forward(self, evidence: Tensor, valid: Tensor) -> PhysicsParameterPrediction:
        summary = self._summarize(evidence, valid)
        baseline = causal_parameter_baseline(evidence, valid, self.config)
        residual = self.long_horizon_config.max_normalized_residual * torch.tanh(
            self.mean_head(summary)
        )
        normalized_mean = (baseline + residual).clamp(0.0, 1.0)
        normalized_log_variance = self.log_variance_head(summary).clamp(-8.0, 2.0)
        lower, upper = _parameter_bounds(self.config, normalized_mean)
        span = upper - lower
        belief_values = lower + span * normalized_mean
        belief_log_variance = normalized_log_variance + 2.0 * span.log()
        return PhysicsParameterPrediction(
            normalized_mean=normalized_mean,
            normalized_log_variance=normalized_log_variance,
            belief_values=belief_values,
            belief_log_variance=belief_log_variance,
        )


__all__ = [
    "LongHorizonAdapterConfig",
    "LongHorizonPhysicsAdapter",
    "causal_parameter_baseline",
]
