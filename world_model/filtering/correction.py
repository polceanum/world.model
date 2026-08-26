"""Analytic-plus-gated posterior correction for persistent beliefs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from world_model.belief import (
    WorldBelief,
    fast_packing_map,
    pack_fast_state,
    unpack_fast_state,
)
from world_model.filtering.analytic_update import diagonal_kalman_update
from world_model.filtering.learned_update import LearnedFastCorrector
from world_model.filtering.uncertainty import (
    FilterUncertainty,
    FilterUncertaintyConfig,
)
from world_model.fusion import AssociationResult, SurpriseAssessment
from world_model.observations import (
    DirectVelocityEvidence,
    InnovationSet,
    MeasurementSet,
    PredictedMeasurements,
)


@dataclass(frozen=True)
class BeliefUpdaterConfig:
    robust_clip_norm: float = 8.0
    minimum_log_variance: float = -12.0
    maximum_log_variance: float = 8.0
    ambiguous_confidence: float = 0.25
    appearance_ema: float = 0.15
    learned_residual_scale: float = 0.1
    enable_learned_corrector: bool = True
    innovation_anchored_correction: bool = False
    velocity_from_position_coupling: float = 0.5
    velocity_from_position_variance_scale: float = 2.0
    maximum_velocity_from_position_delta: float = 6.0
    minimum_velocity_dt: float = 1.0e-4
    missed_fast_variance_increment: float = 0.05
    observed_confidence_threshold: float = 0.5


@dataclass
class CorrectionDiagnostics:
    analytic_gain: Tensor
    correction_norm: Tensor
    observed_mask: Tensor
    robust_weight: Tensor


class BeliefUpdater(nn.Module):
    """Correct supported fast fields; never resets or re-encodes history.

    A modality may explicitly declare ``velocity_from_position`` support. For
    such measurements, a positive elapsed time permits a conservative temporal
    coupling from world-position innovation to velocity. This is not enabled
    implicitly for other modalities and is skipped exactly at ``dt=0``.
    """

    def __init__(
        self,
        *,
        fast_state_dim: int,
        num_motion_modes: int,
        hidden_dim: int = 128,
        config: BeliefUpdaterConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or BeliefUpdaterConfig()
        if not 0.0 <= self.config.observed_confidence_threshold <= 1.0:
            raise ValueError("observed_confidence_threshold must lie in [0,1]")
        self.learned_corrector = (
            LearnedFastCorrector(
                fast_state_dim=fast_state_dim,
                num_motion_modes=num_motion_modes,
                hidden_dim=hidden_dim,
            )
            if self.config.enable_learned_corrector
            else None
        )
        self.uncertainty = FilterUncertainty(
            FilterUncertaintyConfig(
                missed_fast_variance_increment=(self.config.missed_fast_variance_increment),
                minimum_log_variance=self.config.minimum_log_variance,
                maximum_log_variance=self.config.maximum_log_variance,
            )
        )
        self.last_diagnostics: CorrectionDiagnostics | None = None

    @staticmethod
    def _associated_indices(
        association: AssociationResult,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_index, pair_index = torch.nonzero(association.pair_mask, as_tuple=True)
        belief_index = association.belief_indices[batch_index, pair_index]
        return batch_index, pair_index, belief_index

    def _measurement_confidence(
        self,
        measured: MeasurementSet,
        association: AssociationResult,
        batch_index: Tensor,
        pair_index: Tensor,
    ) -> Tensor:
        measurement_index = association.measurement_indices[batch_index, pair_index]
        confidence = measured.existence_logits[batch_index, measurement_index].sigmoid()
        ambiguous = association.ambiguous[batch_index, pair_index]
        return confidence * torch.where(
            ambiguous,
            confidence.new_full((), self.config.ambiguous_confidence),
            confidence.new_ones(()),
        )

    @staticmethod
    def _position_causal_axis_support(
        measured: MeasurementSet,
        innovation: InnovationSet,
        association: AssociationResult,
        batch_index: Tensor,
        pair_index: Tensor,
    ) -> Tensor:
        """Return independently observed world axes for associated rows.

        The typed measurement is authoritative. Source-conditioned rows that
        omit axis provenance fail closed because a copied prior coordinate is
        not a new observation. Legacy unbound/global measurements retain
        all-axis support.
        """

        independent_axis = measured.auxiliary.get("world_position_independent_axis_mask")
        if independent_axis is not None:
            expected_shape = (*measured.values.shape[:2], 3)
            if independent_axis.shape != expected_shape or independent_axis.dtype != torch.bool:
                raise ValueError("measured world-position independence must be boolean [B,M,3]")
            measurement_index = association.measurement_indices[
                batch_index,
                pair_index,
            ]
            return independent_axis[batch_index, measurement_index]

        source_bound = innovation.auxiliary.get("measured_source_bound")
        if source_bound is not None:
            if (
                source_bound.shape != association.pair_mask.shape
                or source_bound.dtype != torch.bool
            ):
                raise ValueError("measured source-bound mask must be boolean [B,P]")
            selected_source_bound = source_bound[batch_index, pair_index]
        else:
            selected_source_bound = torch.zeros(
                (batch_index.numel(),),
                dtype=torch.bool,
                device=batch_index.device,
            )

        source_fields_present = (
            measured.source_belief_indices is not None or measured.source_object_ids is not None
        )
        if source_fields_present:
            if measured.source_belief_indices is None or measured.source_object_ids is None:
                raise ValueError(
                    "source-conditioned measurements require both source identity fields"
                )
            selected_source_bound = torch.ones_like(selected_source_bound)

        if source_bound is not None or source_fields_present:
            return ~selected_source_bound.unsqueeze(-1).expand(-1, 3)
        return torch.ones(
            (batch_index.numel(), 3),
            dtype=torch.bool,
            device=batch_index.device,
        )

    @staticmethod
    def _normalise_dt(prior: WorldBelief, dt: float | Tensor) -> Tensor:
        value = torch.as_tensor(
            dt,
            device=prior.device,
            dtype=prior.dtype,
        )
        if value.ndim == 0:
            value = value.expand(prior.batch_size).clone()
        if value.shape != prior.timestamp.shape:
            raise ValueError("filter correction dt must be scalar or shape [B]")
        if not torch.isfinite(value).all() or torch.any(value < 0):
            raise ValueError("filter correction dt must be finite and nonnegative")
        return value

    def correct(
        self,
        *,
        prior: WorldBelief,
        measured: MeasurementSet,
        predicted: PredictedMeasurements,
        association: AssociationResult,
        innovation: InnovationSet,
        dt: float | Tensor = 0.0,
        cause: SurpriseAssessment | None = None,
    ) -> WorldBelief:
        del predicted
        elapsed_by_batch = self._normalise_dt(prior, dt)
        packed = pack_fast_state(prior.objects)
        log_variance = prior.objects.fast_log_variance
        if packed.shape != log_variance.shape:
            raise ValueError("packed fast state and fast_log_variance must match")
        updated_packed = packed.clone()
        updated_log_variance = log_variance.clone()
        batch_index, pair_index, belief_index = self._associated_indices(association)
        observed_mask = torch.zeros_like(prior.objects.active)
        if batch_index.numel() == 0:
            posterior = self.uncertainty.missed(prior, prior.objects.active)
            self.last_diagnostics = CorrectionDiagnostics(
                analytic_gain=packed.new_zeros((0, 3)),
                correction_norm=packed.new_zeros((0,)),
                observed_mask=observed_mask,
                robust_weight=packed.new_zeros((0,)),
            )
            return posterior

        confidence = self._measurement_confidence(
            measured,
            association,
            batch_index,
            pair_index,
        )
        learned_confidence = confidence
        if cause is not None:
            # The analytic diagonal update applies its own robust influence.
            # Applying the surprise weight here as well would suppress large,
            # valid event corrections twice. Keep the extra conservative gate
            # only for the unconstrained learned residual path.
            learned_confidence = (
                confidence
                * cause.robust_weight[
                    batch_index,
                    pair_index,
                ]
            )
        position_confidence = confidence
        reported_position_confidence = measured.auxiliary.get("position_confidence")
        if reported_position_confidence is not None:
            scalar_shape = measured.measurement_mask.shape
            axis_shapes = {
                (*scalar_shape, 1),
                (*scalar_shape, 3),
            }
            if (
                reported_position_confidence.shape != scalar_shape
                and reported_position_confidence.shape not in axis_shapes
            ):
                raise ValueError(
                    "auxiliary.position_confidence must have shape [B,M], [B,M,1], or [B,M,3]"
                )
            if not torch.isfinite(reported_position_confidence).all():
                raise ValueError("auxiliary.position_confidence must be finite")
            measurement_index = association.measurement_indices[batch_index, pair_index]
            position_quality = reported_position_confidence[
                batch_index,
                measurement_index,
            ].clamp(0.0, 1.0)
            # Localization quality is a conservative cap: it may reduce the
            # existence/association confidence but can never promote it. Using
            # a minimum also avoids squaring confidence for observers whose
            # default quality estimate aliases existence confidence. The
            # optional trailing dimension permits an observer to downweight a
            # noisy depth/world axis without suppressing precise lateral axes.
            if position_quality.ndim == confidence.ndim + 1:
                position_confidence = torch.minimum(
                    confidence.unsqueeze(-1),
                    position_quality,
                )
            else:
                position_confidence = torch.minimum(confidence, position_quality)
        position = innovation.auxiliary.get("measured_world_position")
        if position is None:
            raise ValueError(
                f"{measured.modality} measurements must provide world_position "
                "for position correction"
            )
        position_measurement = position[batch_index, pair_index]
        position_lv = innovation.auxiliary.get("measured_world_position_log_variance")
        if position_lv is None:
            measured_lv = innovation.auxiliary["measurement_log_variance"]
            position_lv = measured_lv[..., :3]
        position_measurement_lv = position_lv[batch_index, pair_index]
        packing = fast_packing_map(prior.objects)
        correction_evidence = packed.new_zeros((batch_index.numel(), packed.shape[-1]))
        correction_confidence = packed.new_zeros(correction_evidence.shape)
        position_slice = packing["position"]
        position_causal_axis_support = self._position_causal_axis_support(
            measured,
            innovation,
            association,
            batch_index,
            pair_index,
        )
        prior_position = packed[batch_index, belief_index, position_slice]
        prior_position_lv = log_variance[batch_index, belief_index, position_slice]
        causal_position_measurement = torch.where(
            position_causal_axis_support,
            position_measurement,
            prior_position,
        )
        causal_position_measurement_lv = torch.where(
            position_causal_axis_support,
            position_measurement_lv,
            prior_position_lv,
        )
        position_standard_deviation = (
            (prior_position_lv.exp() + causal_position_measurement_lv.exp())
            .clamp_min(1.0e-8)
            .sqrt()
        )
        correction_evidence[..., position_slice] = (
            (causal_position_measurement - prior_position) / position_standard_deviation
        ).clamp(-self.config.robust_clip_norm, self.config.robust_clip_norm)
        position_learned_confidence = position_confidence
        if position_learned_confidence.ndim == confidence.ndim:
            position_learned_confidence = position_learned_confidence.unsqueeze(-1)
        analytic_position_confidence = position_learned_confidence * (
            position_causal_axis_support.to(position_learned_confidence.dtype)
        )
        # The same causal-axis support owns both analytic and learned
        # innovation-anchored updates. A copied/source-bound coordinate is not
        # evidence for either a learned mean shift or variance contraction.
        correction_confidence[..., position_slice] = analytic_position_confidence
        analytic_position = diagonal_kalman_update(
            prior_position,
            prior_position_lv,
            causal_position_measurement,
            causal_position_measurement_lv,
            confidence=analytic_position_confidence,
            robust_clip_norm=self.config.robust_clip_norm,
            minimum_log_variance=self.config.minimum_log_variance,
            maximum_log_variance=self.config.maximum_log_variance,
        )
        updated_packed[batch_index, belief_index, position_slice] = analytic_position.mean
        updated_log_variance[batch_index, belief_index, position_slice] = (
            analytic_position.log_variance
        )

        velocity_slice = packing["velocity"]
        if "velocity" in measured.supported_state_fields:
            velocity = innovation.auxiliary.get("measured_world_velocity")
            if velocity is None:
                raise ValueError("direct velocity measurements require auxiliary.world_velocity")
            velocity_lv = innovation.auxiliary.get("measured_world_velocity_log_variance")
            if velocity_lv is None:
                raise ValueError(
                    "direct velocity measurements require auxiliary.world_velocity_log_variance"
                )
            velocity_valid = innovation.auxiliary.get("measured_world_velocity_valid_mask")
            if velocity_valid is None:
                raise ValueError(
                    "direct velocity measurements require auxiliary.world_velocity_valid_mask"
                )
            if velocity.shape[-1] != 3 or velocity_lv.shape != velocity.shape:
                raise ValueError("direct world velocity and log variance must end with three")
            if velocity_valid.shape != velocity.shape[:2] or velocity_valid.dtype != torch.bool:
                raise ValueError("direct world velocity validity must be boolean [B,P]")
            velocity_measurement = velocity[batch_index, pair_index]
            velocity_measurement_lv = velocity_lv[batch_index, pair_index]
            velocity_confidence = confidence * velocity_valid[
                batch_index,
                pair_index,
            ].to(confidence.dtype)
            analytic_velocity = diagonal_kalman_update(
                packed[batch_index, belief_index, velocity_slice],
                log_variance[batch_index, belief_index, velocity_slice],
                velocity_measurement,
                velocity_measurement_lv,
                confidence=velocity_confidence,
                robust_clip_norm=self.config.robust_clip_norm,
                minimum_log_variance=self.config.minimum_log_variance,
                maximum_log_variance=self.config.maximum_log_variance,
            )
            updated_packed[batch_index, belief_index, velocity_slice] = analytic_velocity.mean
            updated_log_variance[batch_index, belief_index, velocity_slice] = (
                analytic_velocity.log_variance
            )
            velocity_standard_deviation = (
                (
                    log_variance[batch_index, belief_index, velocity_slice].exp()
                    + velocity_measurement_lv.exp()
                )
                .clamp_min(1.0e-8)
                .sqrt()
            )
            velocity_evidence = (
                (velocity_measurement - packed[batch_index, belief_index, velocity_slice])
                / velocity_standard_deviation
            ).clamp(-self.config.robust_clip_norm, self.config.robust_clip_norm)
            velocity_supported = velocity_valid[batch_index, pair_index].unsqueeze(-1)
            correction_evidence[..., velocity_slice] = torch.where(
                velocity_supported,
                velocity_evidence,
                torch.zeros_like(velocity_evidence),
            )
            correction_confidence[..., velocity_slice] = velocity_confidence.unsqueeze(
                -1
            ) * velocity_supported.to(confidence.dtype)
        elif "velocity_from_position" in measured.supported_state_fields:
            elapsed = elapsed_by_batch[batch_index]
            valid_elapsed = elapsed > self.config.minimum_velocity_dt
            safe_elapsed = elapsed.clamp_min(self.config.minimum_velocity_dt)
            position_residual = causal_position_measurement - prior_position
            raw_velocity_delta = position_residual / safe_elapsed.unsqueeze(-1)
            delta_norm = torch.linalg.vector_norm(
                raw_velocity_delta,
                dim=-1,
                keepdim=True,
            )
            bounded_scale = torch.minimum(
                torch.ones_like(delta_norm),
                raw_velocity_delta.new_tensor(self.config.maximum_velocity_from_position_delta)
                / delta_norm.clamp_min(1.0e-8),
            )
            bounded_delta = raw_velocity_delta * bounded_scale
            prior_velocity = packed[batch_index, belief_index, velocity_slice]
            velocity_measurement = prior_velocity + (
                self.config.velocity_from_position_coupling * bounded_delta
            )
            # A finite difference amplifies position noise by 1/dt². The
            # additional scale keeps this indirect observation conservative.
            velocity_measurement_variance = (
                causal_position_measurement_lv.exp()
                / safe_elapsed.unsqueeze(-1).square()
                * self.config.velocity_from_position_variance_scale
            )
            velocity_measurement_lv = velocity_measurement_variance.clamp_min(1.0e-10).log()
            elapsed_confidence = valid_elapsed.to(confidence.dtype)
            if analytic_position_confidence.ndim == elapsed_confidence.ndim + 1:
                elapsed_confidence = elapsed_confidence.unsqueeze(-1)
            velocity_confidence = analytic_position_confidence * elapsed_confidence
            analytic_velocity = diagonal_kalman_update(
                prior_velocity,
                log_variance[batch_index, belief_index, velocity_slice],
                velocity_measurement,
                velocity_measurement_lv,
                confidence=velocity_confidence,
                robust_clip_norm=self.config.robust_clip_norm,
                minimum_log_variance=self.config.minimum_log_variance,
                maximum_log_variance=self.config.maximum_log_variance,
            )
            updated_packed[batch_index, belief_index, velocity_slice] = analytic_velocity.mean
            updated_log_variance[batch_index, belief_index, velocity_slice] = (
                analytic_velocity.log_variance
            )
            velocity_standard_deviation = (
                (
                    log_variance[batch_index, belief_index, velocity_slice].exp()
                    + velocity_measurement_lv.exp()
                )
                .clamp_min(1.0e-8)
                .sqrt()
            )
            velocity_evidence = (
                (velocity_measurement - prior_velocity) / velocity_standard_deviation
            ).clamp(-self.config.robust_clip_norm, self.config.robust_clip_norm)
            velocity_supported = valid_elapsed.unsqueeze(-1)
            correction_evidence[..., velocity_slice] = torch.where(
                velocity_supported,
                velocity_evidence,
                torch.zeros_like(velocity_evidence),
            )
            if velocity_confidence.ndim == confidence.ndim:
                velocity_confidence = velocity_confidence.unsqueeze(-1)
            correction_confidence[..., velocity_slice] = (
                velocity_confidence * velocity_supported.to(confidence.dtype)
            )

        objects = unpack_fast_state(updated_packed, prior.objects)
        orientation = F.normalize(objects.orientation, dim=-1)
        objects = objects.replace(
            orientation=orientation,
            fast_log_variance=updated_log_variance,
        )

        if self.learned_corrector is not None:
            prior_pair = packed[batch_index, belief_index]
            lv_pair = log_variance[batch_index, belief_index]
            whitened = innovation.whitened_residual[batch_index, pair_index]
            cost = association.pair_cost[batch_index, pair_index]
            ambiguity = association.ambiguous[batch_index, pair_index]
            visibility = prior.objects.visibility_logit[batch_index, belief_index].sigmoid()
            elapsed = elapsed_by_batch[batch_index]
            modes = prior.objects.motion_mode_logits[batch_index, belief_index]
            modality = innovation.modality_index[batch_index, pair_index]
            learned = self.learned_corrector(
                prior_fast_state=prior_pair,
                prior_log_variance=lv_pair,
                whitened_innovation=whitened,
                association_cost=cost,
                ambiguity=ambiguity,
                visibility=visibility,
                elapsed_time=elapsed,
                motion_mode_logits=modes,
                modality_index=modality,
            )
            confidence_full = learned_confidence.unsqueeze(-1)
            if self.config.innovation_anchored_correction:
                # The corrector learns a bounded gain on explicit, supported
                # world-state innovation.  This prevents a camera-space
                # summary from inventing an unrelated axis/state correction
                # while still allowing context to modulate or oppose the
                # analytic proposal.  Surprise robustness is applied once to
                # the learned path, after per-axis measurement confidence.
                if cause is not None:
                    correction_confidence = correction_confidence * cause.robust_weight[
                        batch_index,
                        pair_index,
                    ].unsqueeze(-1)
                learned_state_factor = correction_evidence.tanh() * correction_confidence
            else:
                # Preserve exact historical checkpoint semantics unless the
                # corrected protocol is selected explicitly in configuration.
                learned_state_factor = confidence_full
            learned_delta = (
                learned.state_gate
                * learned.mean_delta
                * self.config.learned_residual_scale
                * learned_state_factor
            )
            updated_packed = pack_fast_state(objects).clone()
            updated_packed[batch_index, belief_index] = (
                updated_packed[batch_index, belief_index] + learned_delta
            )
            updated_lv = objects.fast_log_variance.clone()
            variance_factor = (
                correction_confidence
                if self.config.innovation_anchored_correction
                else confidence_full
            )
            updated_lv[batch_index, belief_index] = (
                updated_lv[batch_index, belief_index] + variance_factor * learned.log_variance_delta
            ).clamp(
                self.config.minimum_log_variance,
                self.config.maximum_log_variance,
            )
            objects = unpack_fast_state(updated_packed, objects)
            objects = objects.replace(
                orientation=F.normalize(objects.orientation, dim=-1),
                fast_log_variance=updated_lv,
            )
            motion_logits = objects.motion_mode_logits.clone()
            motion_logits[batch_index, belief_index] = (
                motion_logits[batch_index, belief_index]
                + confidence_full * learned.mode_logit_delta
            )
            existence_logit = objects.existence_logit.clone()
            visibility_logit = objects.visibility_logit.clone()
            existence_logit[batch_index, belief_index] += confidence * learned.existence_delta
            visibility_logit[batch_index, belief_index] += confidence * learned.visibility_delta
            objects = objects.replace(
                motion_mode_logits=motion_logits,
                existence_logit=existence_logit,
                visibility_logit=visibility_logit,
            )

        existence = objects.existence_logit.clone()
        visibility_logits = objects.visibility_logit.clone()
        measurement_indices = association.measurement_indices[batch_index, pair_index]
        measured_existence = measured.existence_logits[batch_index, measurement_indices]
        existence[batch_index, belief_index] = torch.maximum(
            existence[batch_index, belief_index],
            measured_existence,
        )
        measured_visibility = measured.auxiliary.get("visibility_logits")
        if measured_visibility is not None:
            visibility_logits[batch_index, belief_index] = measured_visibility[
                batch_index, measurement_indices
            ]
        observed = confidence >= self.config.observed_confidence_threshold
        observed_mask[batch_index[observed], belief_index[observed]] = True

        if measured.appearance is not None:
            appearance = objects.appearance.clone()
            measured_appearance = measured.appearance[batch_index, measurement_indices]
            dimensions = min(appearance.shape[-1], measured_appearance.shape[-1])
            non_ambiguous = ~association.ambiguous[batch_index, pair_index]
            appearance_confidence = (confidence * non_ambiguous.to(confidence.dtype)).unsqueeze(-1)
            rate = self.config.appearance_ema * appearance_confidence
            old = appearance[batch_index, belief_index, :dimensions]
            new = measured_appearance[..., :dimensions]
            appearance[batch_index, belief_index, :dimensions] = F.normalize(
                old + rate * (new - old), dim=-1
            )
            objects = objects.replace(appearance=appearance)
        objects = objects.replace(
            existence_logit=existence,
            visibility_logit=visibility_logits,
        )
        posterior = prior.replace(objects=objects)
        posterior = self.uncertainty.missed(posterior, posterior.objects.active & ~observed_mask)
        posterior = self.uncertainty.clamp(posterior)
        correction_norm = torch.linalg.vector_norm(analytic_position.correction, dim=-1)
        robust_weight = confidence
        self.last_diagnostics = CorrectionDiagnostics(
            analytic_gain=analytic_position.gain,
            correction_norm=correction_norm,
            observed_mask=observed_mask,
            robust_weight=robust_weight,
        )
        return posterior

    def correct_direct_velocity(
        self,
        prior: WorldBelief,
        evidence: DirectVelocityEvidence,
    ) -> WorldBelief:
        """Apply explicit post-association kinematic evidence in belief-slot order.

        This second analytic update intentionally leaves ``last_diagnostics``
        describing the ordinary measurement correction and observed mask.
        """

        evidence.validate()
        expected = (*prior.objects.active.shape, 3)
        if evidence.velocity.shape != expected:
            raise ValueError(f"direct velocity evidence must have shape {expected}")
        packed = pack_fast_state(prior.objects)
        log_variance = prior.objects.fast_log_variance
        updated_packed = packed.clone()
        updated_log_variance = log_variance.clone()
        position_slice = fast_packing_map(prior.objects)["position"]
        position_update_count = 0
        if evidence.position is not None:
            assert evidence.position_log_variance is not None
            assert evidence.position_valid_mask is not None
            position_valid = evidence.position_valid_mask & prior.objects.active
            position_batch, position_belief = torch.nonzero(position_valid, as_tuple=True)
            if position_batch.numel():
                position_update_count = int(position_batch.numel())
                analytic_position = diagonal_kalman_update(
                    packed[position_batch, position_belief, position_slice],
                    log_variance[position_batch, position_belief, position_slice],
                    evidence.position[position_batch, position_belief],
                    evidence.position_log_variance[position_batch, position_belief],
                    confidence=evidence.confidence[position_batch, position_belief],
                    robust_clip_norm=self.config.robust_clip_norm,
                    minimum_log_variance=self.config.minimum_log_variance,
                    maximum_log_variance=self.config.maximum_log_variance,
                )
                updated_packed[position_batch, position_belief, position_slice] = (
                    analytic_position.mean
                )
                updated_log_variance[position_batch, position_belief, position_slice] = (
                    analytic_position.log_variance
                )

        axis_valid = evidence.resolved_axis_valid_mask() & prior.objects.active.unsqueeze(-1)
        valid = axis_valid.any(dim=-1)
        batch_index, belief_index = torch.nonzero(valid, as_tuple=True)
        velocity_slice = fast_packing_map(prior.objects)["velocity"]
        if batch_index.numel():
            component_valid = axis_valid[batch_index, belief_index]
            prior_velocity = packed[batch_index, belief_index, velocity_slice]
            prior_velocity_log_variance = log_variance[
                batch_index,
                belief_index,
                velocity_slice,
            ]
            component_confidence = evidence.confidence[
                batch_index,
                belief_index,
            ].unsqueeze(-1) * component_valid.to(evidence.confidence.dtype)
            # Unsupported components must not participate in the vector
            # robust-influence norm.  Merely assigning them zero confidence
            # after computing that norm would let an arbitrary unobserved
            # value suppress the correction of a genuinely observed axis.
            component_measurement = torch.where(
                component_valid,
                evidence.velocity[batch_index, belief_index],
                prior_velocity,
            )
            analytic_velocity = diagonal_kalman_update(
                prior_velocity,
                prior_velocity_log_variance,
                component_measurement,
                evidence.log_variance[batch_index, belief_index],
                confidence=component_confidence,
                robust_clip_norm=self.config.robust_clip_norm,
                minimum_log_variance=self.config.minimum_log_variance,
                maximum_log_variance=self.config.maximum_log_variance,
            )
            updated_packed[batch_index, belief_index, velocity_slice] = torch.where(
                component_valid,
                analytic_velocity.mean,
                prior_velocity,
            )
            updated_log_variance[batch_index, belief_index, velocity_slice] = torch.where(
                component_valid,
                analytic_velocity.log_variance,
                prior_velocity_log_variance,
            )
        if batch_index.numel() == 0 and position_update_count == 0:
            return prior
        objects = unpack_fast_state(updated_packed, prior.objects).replace(
            fast_log_variance=updated_log_variance
        )
        return prior.replace(objects=objects)
