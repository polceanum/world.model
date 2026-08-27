"""Modality-independent observation module interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from torch import Tensor, nn

from world_model.observations.context import ObservationContext, SensorContext
from world_model.observations.measurements import (
    DirectVelocityEvidence,
    InnovationSet,
    MeasurementSet,
    PredictedMeasurements,
)
from world_model.observations.packets import ObservationPacket

if TYPE_CHECKING:
    from world_model.belief.lifecycle import BirthAssignments
    from world_model.belief.world_belief import WorldBelief
    from world_model.fusion.association import AssociationResult


class ModalityCache:
    """Marker base class for sensor-local, non-physical cached state."""

    def detach(self) -> ModalityCache:
        return self


class ModalityHistory:
    """Marker base for bounded sensor-local causal histories."""

    def detach(self) -> ModalityHistory:
        return self


class ObservationModule(nn.Module, ABC):
    """Stable interface implemented by every observation modality."""

    modality_name: str
    requires_post_birth_temporal_history: bool = False

    @abstractmethod
    def validate_packet(self, packet: ObservationPacket) -> None:
        """Raise an actionable error if a packet cannot be consumed."""

    @abstractmethod
    def initialise_measurements(
        self,
        packets: Sequence[ObservationPacket],
        context: ObservationContext,
    ) -> MeasurementSet:
        """Run a slow/global measurement pass without a reliable prior."""

    @abstractmethod
    def encode_measurements(
        self,
        packets: Sequence[ObservationPacket],
        prior: WorldBelief,
        predicted: PredictedMeasurements,
        cache: ModalityCache | None,
    ) -> tuple[MeasurementSet, ModalityCache]:
        """Run the ordinary prior-conditioned measurement path."""

    @abstractmethod
    def project(
        self,
        belief: WorldBelief,
        sensor_context: SensorContext,
    ) -> PredictedMeasurements:
        """Project the prior into this modality's measurement space."""

    @abstractmethod
    def innovation(
        self,
        measured: MeasurementSet,
        predicted: PredictedMeasurements,
        association: AssociationResult,
    ) -> InnovationSet:
        """Build typed residual evidence after association."""

    def measurement_likelihood(self, innovation: InnovationSet) -> Tensor:
        return innovation.log_likelihood

    def training_losses(
        self,
        outputs: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
    ) -> Mapping[str, Tensor]:
        del outputs, targets, masks
        return {}

    def update_temporal_history(
        self,
        *,
        posterior: WorldBelief,
        measured: MeasurementSet,
        association: AssociationResult,
        history: ModalityHistory | None,
    ) -> tuple[DirectVelocityEvidence | None, ModalityHistory | None]:
        """Optionally derive causal state evidence after ordinary correction."""

        del posterior, measured, association
        return None, history

    def validate_temporal_history_packet(
        self,
        *,
        posterior: WorldBelief,
        packet: ObservationPacket,
        history: ModalityHistory | None,
    ) -> None:
        """Preflight a causal history append before runtime state can mutate."""

        del posterior, packet, history

    def update_temporal_history_after_births(
        self,
        *,
        posterior: WorldBelief,
        measured: MeasurementSet,
        birth_assignments: BirthAssignments,
        history: ModalityHistory | None,
    ) -> ModalityHistory | None:
        """Optionally seed raw history after permanent lifecycle allocation.

        The ordinary association callback runs before births exist.  Modules
        that opt into this second callback can use explicit discrete birth
        assignments to retain the current raw measurement under its permanent
        ID.  The hook deliberately cannot emit correction evidence: a single
        birth frame must not immediately correct position or velocity twice.
        """

        del posterior, measured, birth_assignments
        return history

    def observe(
        self,
        *,
        packets: Sequence[ObservationPacket],
        prior: WorldBelief | None,
        predicted: PredictedMeasurements | None,
        cache: ModalityCache | None,
        mode: str,
        context: ObservationContext,
    ) -> tuple[MeasurementSet, ModalityCache | None]:
        """Dispatch a scheduler choice without exposing modality branches."""

        if mode in {"GLOBAL_DISCOVERY", "RECOVERY"} or prior is None:
            return self.initialise_measurements(packets, context), cache
        if mode == "SKIP":
            raise ValueError("SKIP mode must be handled by the runtime")
        if predicted is None:
            raise ValueError("FAST_ROI observation requires predicted measurements")
        return self.encode_measurements(packets, prior, predicted, cache)
