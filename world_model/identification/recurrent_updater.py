"""Bounded recurrent online restitution/drag/parameter identification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from world_model.belief import WorldBelief, slow_packing_map
from world_model.fusion import AssociationResult
from world_model.identification.observability import Observability
from world_model.identification.parameters import (
    ParameterBounds,
    project_parameter_tensors,
)
from world_model.observations import InnovationSet


@dataclass(frozen=True)
class ParameterUpdaterConfig:
    hidden_dim: int = 32
    slow_learning_rate: float = 0.05
    analytic_signal_scale: float = 0.1
    minimum_log_variance: float = -10.0
    maximum_log_variance: float = 6.0
    bounds: ParameterBounds = ParameterBounds()


@dataclass
class ParameterUpdateDiagnostics:
    observability: Tensor
    gate: Tensor
    delta: Tensor
    update_count: Tensor


class RecurrentParameterUpdater(nn.Module):
    """Inference-time GRU accumulator; no network-weight update is performed."""

    feature_dim = 22
    parameter_count = 5

    def __init__(self, config: ParameterUpdaterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ParameterUpdaterConfig()
        self.gru = nn.GRUCell(self.feature_dim, self.config.hidden_dim)
        self.delta_head = nn.Linear(self.config.hidden_dim, self.parameter_count)
        self.variance_head = nn.Linear(self.config.hidden_dim, self.parameter_count)
        self.evidence_head = nn.Linear(self.config.hidden_dim, self.parameter_count)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.zeros_(self.variance_head.bias)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.constant_(self.evidence_head.bias, -2.5)
        self.last_diagnostics: ParameterUpdateDiagnostics | None = None

    @staticmethod
    def _scatter_pair_features(
        belief: WorldBelief,
        innovation: InnovationSet,
        association: AssociationResult,
    ) -> tuple[Tensor, Tensor, Tensor]:
        objects = belief.objects
        batch, object_count = objects.active.shape
        summary = objects.position.new_zeros((batch, object_count, 4))
        cost_ambiguity = objects.position.new_zeros((batch, object_count, 2))
        velocity_residual = objects.position.new_zeros((batch, object_count, 3))
        batch_index, pair_index = torch.nonzero(association.pair_mask, as_tuple=True)
        if batch_index.numel() == 0:
            return summary, cost_ambiguity, velocity_residual
        object_index = association.belief_indices[batch_index, pair_index]
        whitened = innovation.whitened_residual[batch_index, pair_index]
        summary[batch_index, object_index] = torch.stack(
            (
                whitened.mean(dim=-1),
                whitened.abs().mean(dim=-1),
                whitened.abs().amax(dim=-1),
                torch.linalg.vector_norm(whitened, dim=-1),
            ),
            dim=-1,
        )
        cost_ambiguity[batch_index, object_index, 0] = association.pair_cost[
            batch_index, pair_index
        ].nan_to_num(posinf=100.0)
        cost_ambiguity[batch_index, object_index, 1] = association.ambiguous[
            batch_index, pair_index
        ].to(summary.dtype)
        if innovation.modality == "debug_oracle" and innovation.residual.shape[-1] >= 6:
            velocity_residual[batch_index, object_index] = innovation.residual[
                batch_index, pair_index, 3:6
            ]
        return summary, cost_ambiguity, velocity_residual

    def _features(
        self,
        belief: WorldBelief,
        innovation: InnovationSet,
        association: AssociationResult,
        observability: Observability,
    ) -> tuple[Tensor, Tensor]:
        objects = belief.objects
        innovation_summary, cost_ambiguity, velocity_residual = self._scatter_pair_features(
            belief, innovation, association
        )
        speed = torch.linalg.vector_norm(objects.velocity, dim=-1, keepdim=True)
        modes = objects.motion_mode_logits.softmax(dim=-1)
        selected_modes = modes[
            ...,
            [0, 1, 2, 3, 5],
        ]
        radius = objects.geometry[..., :1]
        current = torch.cat(
            (
                objects.log_mass,
                objects.restitution_logit,
                objects.log_drag,
                objects.friction_logit,
                radius,
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                innovation_summary,
                speed,
                selected_modes,
                observability.stacked(),
                current,
                cost_ambiguity,
            ),
            dim=-1,
        )
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError("parameter-updater feature contract changed")
        return features, velocity_residual

    @staticmethod
    def _analytic_signals(
        belief: WorldBelief,
        innovation: InnovationSet,
        association: AssociationResult,
        velocity_residual: Tensor,
    ) -> Tensor:
        objects = belief.objects
        batch, object_count = objects.active.shape
        signals = objects.position.new_zeros(
            (batch, object_count, RecurrentParameterUpdater.parameter_count)
        )
        batch_index, pair_index = torch.nonzero(association.pair_mask, as_tuple=True)
        if batch_index.numel() == 0:
            return signals
        object_index = association.belief_indices[batch_index, pair_index]
        velocity = objects.velocity[batch_index, object_index]
        speed_squared = velocity.square().sum(dim=-1).clamp_min(1.0e-4)
        if innovation.modality == "debug_oracle" and innovation.residual.shape[-1] >= 6:
            dynamic_residual = velocity_residual[batch_index, object_index]
        else:
            dynamic_residual = (
                innovation.auxiliary["measured_world_position"][batch_index, pair_index]
                - objects.position[batch_index, object_index]
            )
        along_motion = (dynamic_residual * velocity).sum(dim=-1) / speed_squared
        # A measured state lagging the prediction supports greater drag.  A
        # post-impact velocity with greater same-direction magnitude supports
        # greater restitution.
        signals[batch_index, object_index, 1] = along_motion.clamp(-1.0, 1.0)
        signals[batch_index, object_index, 2] = (-along_motion).clamp(-1.0, 1.0)
        return signals

    def update(
        self,
        belief: WorldBelief,
        innovation: InnovationSet,
        association: AssociationResult,
        observability: Observability,
    ) -> WorldBelief:
        objects = belief.objects
        if objects.parameter_memory.shape[-1] != self.config.hidden_dim:
            raise ValueError("belief parameter_memory dimension must equal identifier hidden_dim")
        features, velocity_residual = self._features(belief, innovation, association, observability)
        flat_features = features.reshape(-1, self.feature_dim)
        flat_memory = objects.parameter_memory.reshape(-1, self.config.hidden_dim)
        candidate_memory = self.gru(flat_features, flat_memory).reshape_as(objects.parameter_memory)
        memory_gate = observability.stacked().amax(dim=-1, keepdim=True)
        new_memory = objects.parameter_memory + memory_gate * (
            candidate_memory - objects.parameter_memory
        )
        learned_delta = torch.tanh(self.delta_head(new_memory))
        variance_delta = torch.tanh(self.variance_head(new_memory))
        evidence = torch.sigmoid(self.evidence_head(new_memory))
        observability_tensor = observability.stacked()
        gate = evidence * observability_tensor * objects.active.unsqueeze(-1)
        analytic_signal = self._analytic_signals(
            belief,
            innovation,
            association,
            velocity_residual,
        )
        delta = (
            self.config.slow_learning_rate
            * gate
            * (learned_delta + self.config.analytic_signal_scale * analytic_signal)
        )
        log_mass = objects.log_mass + delta[..., 0:1]
        restitution = objects.restitution_logit + delta[..., 1:2]
        log_drag = objects.log_drag + delta[..., 2:3]
        friction = objects.friction_logit + delta[..., 3:4]
        radius = objects.geometry[..., :1] + delta[..., 4:5]
        log_mass, restitution, log_drag, friction, radius = project_parameter_tensors(
            log_mass=log_mass,
            restitution_logit=restitution,
            log_drag=log_drag,
            friction_logit=friction,
            radius=radius,
            bounds=self.config.bounds,
        )
        geometry = objects.geometry.clone()
        geometry[..., :1] = radius
        slow_log_variance = objects.slow_log_variance.clone()
        packing = slow_packing_map(objects)
        parameter_slices = (
            packing["log_mass"],
            packing["restitution_logit"],
            packing["log_drag"],
            packing["friction_logit"],
            slice(packing["geometry"].start, packing["geometry"].start + 1),
        )
        for parameter_index, parameter_slice in enumerate(parameter_slices):
            current = slow_log_variance[..., parameter_slice]
            contraction = (
                0.1
                * gate[..., parameter_index : parameter_index + 1]
                * (0.5 + 0.5 * variance_delta[..., parameter_index : parameter_index + 1])
            )
            slow_log_variance[..., parameter_slice] = (current - contraction).clamp(
                self.config.minimum_log_variance,
                self.config.maximum_log_variance,
            )
        updated = objects.replace(
            log_mass=log_mass,
            restitution_logit=restitution,
            log_drag=log_drag,
            friction_logit=friction,
            geometry=geometry,
            slow_log_variance=slow_log_variance,
            parameter_memory=new_memory,
        )
        update_count = (gate > 1.0e-3).to(torch.int64)
        self.last_diagnostics = ParameterUpdateDiagnostics(
            observability=observability_tensor,
            gate=gate,
            delta=delta,
            update_count=update_count,
        )
        return belief.replace(objects=updated)


ParameterIdentifier = RecurrentParameterUpdater
