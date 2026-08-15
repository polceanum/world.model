"""Bounded recurrent online restitution/drag/parameter identification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from world_model.belief import WorldBelief, fast_packing_map, slow_packing_map
from world_model.fusion import AssociationResult
from world_model.identification.observability import Observability
from world_model.identification.parameters import (
    ParameterBounds,
    project_parameter_tensors,
)
from world_model.observations import DirectVelocityEvidence, InnovationSet


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
        *,
        elapsed_seconds: Tensor | None = None,
        predicted_belief: WorldBelief | None = None,
        direct_velocity_evidence: DirectVelocityEvidence | None = None,
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
        source = belief if predicted_belief is None else predicted_belief
        if source.objects.active.shape != objects.active.shape:
            raise ValueError("predicted belief must match identifier belief shape")
        velocity = source.objects.velocity[batch_index, object_index]
        supported_axis = torch.ones_like(velocity, dtype=torch.bool)
        signal_confidence = velocity.new_ones(velocity.shape[:-1])
        signal_variance = torch.zeros_like(velocity)
        restitution_supported = innovation.modality == "debug_oracle"
        if innovation.modality == "debug_oracle" and innovation.residual.shape[-1] >= 6:
            dynamic_residual = velocity_residual[batch_index, object_index]
        elif direct_velocity_evidence is not None:
            direct_velocity_evidence.validate()
            if direct_velocity_evidence.velocity.shape[:2] != objects.active.shape:
                raise ValueError("direct velocity evidence must be in belief-slot order")
            supported_axis = direct_velocity_evidence.resolved_axis_valid_mask()[
                batch_index,
                object_index,
            ]
            dynamic_residual = (
                direct_velocity_evidence.velocity[batch_index, object_index] - velocity
            )
            velocity_slice = fast_packing_map(source.objects)["velocity"]
            signal_variance = (
                direct_velocity_evidence.log_variance[batch_index, object_index].exp()
                + source.objects.fast_log_variance[
                    batch_index,
                    object_index,
                    velocity_slice,
                ].exp()
            )
            signal_confidence = direct_velocity_evidence.confidence[
                batch_index,
                object_index,
            ]
            restitution_supported = True
        else:
            measured_world_position = innovation.auxiliary.get("measured_world_position")
            predicted_world_position = innovation.auxiliary.get("predicted_world_position")
            if measured_world_position is None or predicted_world_position is None:
                # Modalities without an explicit world-space projection still
                # contribute through the learned innovation summary, but they
                # cannot support this directional analytic heuristic.
                return signals
            # Parameter evidence is the causal prior prediction error.  Using
            # the already-corrected posterior position here would feed the
            # identifier a residual that the fast filter has deliberately
            # removed, weakening (and sometimes reversing) slow adaptation.
            position_residual = (
                measured_world_position[batch_index, pair_index]
                - predicted_world_position[batch_index, pair_index]
            )
            if elapsed_seconds is None:
                elapsed = velocity.new_ones((belief.batch_size,))[batch_index]
            else:
                elapsed = torch.as_tensor(
                    elapsed_seconds,
                    device=velocity.device,
                    dtype=velocity.dtype,
                )
                if elapsed.ndim == 0:
                    elapsed = elapsed.expand(belief.batch_size)
                if elapsed.shape != belief.timestamp.shape:
                    raise ValueError("identifier elapsed_seconds must be scalar or shape [B]")
                elapsed = elapsed[batch_index]
            valid_elapsed = torch.isfinite(elapsed) & (elapsed > 0.0)
            safe_elapsed = elapsed.clamp_min(1.0e-6)
            supported_axis = supported_axis & valid_elapsed.unsqueeze(-1)
            independent_axis = innovation.auxiliary.get(
                "measured_world_position_independent_axis_mask"
            )
            source_bound = innovation.auxiliary.get("measured_source_bound")
            if independent_axis is not None:
                if (
                    independent_axis.shape != (*association.pair_mask.shape, 3)
                    or independent_axis.dtype != torch.bool
                ):
                    raise ValueError("measured world-position independence must be boolean [B,P,3]")
                supported_axis = supported_axis & independent_axis[batch_index, pair_index]
            elif source_bound is not None:
                if (
                    source_bound.shape != association.pair_mask.shape
                    or source_bound.dtype != torch.bool
                ):
                    raise ValueError("measured source-bound mask must be boolean [B,P]")
                # A prior-conditioned ROI without explicit raw-axis provenance
                # cannot become physical-parameter evidence. Global or legacy
                # modality rows remain all-axis compatible when unbound.
                supported_axis = supported_axis & ~source_bound[
                    batch_index,
                    pair_index,
                ].unsqueeze(-1)
            dynamic_residual = position_residual / safe_elapsed.unsqueeze(-1)
            measured_position_lv = innovation.auxiliary.get("measured_world_position_log_variance")
            predicted_position_lv = innovation.auxiliary.get(
                "predicted_world_position_log_variance"
            )
            if measured_position_lv is not None:
                if predicted_position_lv is None:
                    position_slice = fast_packing_map(source.objects)["position"]
                    selected_predicted_variance = source.objects.fast_log_variance[
                        batch_index,
                        object_index,
                        position_slice,
                    ].exp()
                else:
                    selected_predicted_variance = predicted_position_lv[
                        batch_index,
                        pair_index,
                    ].exp()
                signal_variance = (
                    measured_position_lv[batch_index, pair_index].exp()
                    + selected_predicted_variance
                ) / safe_elapsed.unsqueeze(-1).square()
            position_confidence = innovation.auxiliary.get("measured_position_confidence")
            if position_confidence is not None:
                selected_confidence = position_confidence[batch_index, pair_index]
                if selected_confidence.ndim == signal_confidence.ndim + 1:
                    selected_confidence = selected_confidence.mean(dim=-1)
                signal_confidence = selected_confidence.clamp(0.0, 1.0)

        supported = supported_axis.to(velocity.dtype)
        supported_speed_squared = (velocity.square() * supported).sum(dim=-1).clamp_min(1.0e-4)
        along_motion = (dynamic_residual * velocity * supported).sum(
            dim=-1
        ) / supported_speed_squared
        # The variance of dot(residual, velocity) / ||velocity||^2 provides a
        # dimensionless reliability penalty. Ambiguous monocular depth can
        # therefore contribute through the learned innovation summary without
        # saturating the analytic slow-parameter heuristic.
        projection_variance = (
            velocity.square()
            * supported
            * signal_variance
            / supported_speed_squared.unsqueeze(-1).square()
        ).sum(dim=-1)
        reliability = signal_confidence / (1.0 + projection_variance.clamp_min(0.0))
        along_motion = along_motion * reliability
        # A measured state lagging the prediction supports greater drag.  A
        # directly observed post-impact velocity with greater same-direction
        # magnitude supports greater restitution. Position displacement alone
        # is rate-normalized drag evidence, not a fabricated impact velocity.
        if restitution_supported:
            signals[batch_index, object_index, 1] = along_motion.clamp(-1.0, 1.0)
        signals[batch_index, object_index, 2] = (-along_motion).clamp(-1.0, 1.0)
        return signals

    def update(
        self,
        belief: WorldBelief,
        innovation: InnovationSet,
        association: AssociationResult,
        observability: Observability,
        *,
        elapsed_seconds: Tensor | None = None,
        predicted_belief: WorldBelief | None = None,
        direct_velocity_evidence: DirectVelocityEvidence | None = None,
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
            elapsed_seconds=elapsed_seconds,
            predicted_belief=predicted_belief,
            direct_velocity_evidence=direct_velocity_evidence,
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
