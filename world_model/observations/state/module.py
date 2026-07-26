"""Clearly-labelled simulator-state observation module for debugging only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from world_model.fusion.innovation import build_innovation
from world_model.observations.base import ModalityCache, ObservationModule
from world_model.observations.context import ObservationContext, SensorContext
from world_model.observations.measurements import (
    InnovationSet,
    MeasurementSet,
    PredictedMeasurements,
)
from world_model.observations.packets import ObservationPacket
from world_model.observations.registry import register_observation_module

if TYPE_CHECKING:
    from world_model.belief.world_belief import WorldBelief
    from world_model.fusion.association import AssociationResult


@dataclass(frozen=True)
class StateObservationConfig:
    position_variance: float = 1.0e-4
    velocity_variance: float = 1.0e-3
    noise_standard_deviation: float = 0.0
    appearance_dim: int | None = None


def _as_batched(value: Tensor, dimensions: int) -> Tensor:
    if value.ndim == dimensions - 1:
        return value.unsqueeze(0)
    if value.ndim != dimensions:
        raise ValueError(f"oracle payload tensor must have {dimensions - 1} or {dimensions} axes")
    return value


@register_observation_module("debug_oracle")
class StateObservationModule(ObservationModule):
    """Privileged state measurements for tests and dynamics ablations.

    This module is deliberately registered as ``debug_oracle`` so reports and
    configs cannot accidentally describe it as an RGB result.
    """

    modality_name = "debug_oracle"
    modality_index = 1

    def __init__(self, config: StateObservationConfig | None = None) -> None:
        super().__init__()
        self.config = config or StateObservationConfig()

    def validate_packet(self, packet: ObservationPacket) -> None:
        if packet.modality != self.modality_name:
            raise ValueError("StateObservationModule only accepts modality='debug_oracle'")
        if not isinstance(packet.payload, Mapping):
            raise TypeError("debug_oracle payload must be a mapping")
        if "position" not in packet.payload:
            raise ValueError("debug_oracle payload requires position")
        position = packet.payload["position"]
        if not isinstance(position, Tensor) or position.shape[-1] != 3:
            raise ValueError("debug_oracle position must be a Tensor [...,N,3]")
        if not torch.isfinite(position).all():
            raise ValueError("debug_oracle position contains NaN or Inf")

    def _measure(self, packet: ObservationPacket) -> MeasurementSet:
        self.validate_packet(packet)
        payload = packet.payload
        position = _as_batched(payload["position"], 3).to(torch.float32)
        batch, objects, _ = position.shape
        velocity_value = payload.get("velocity")
        if velocity_value is None:
            velocity = torch.zeros_like(position)
        else:
            velocity = _as_batched(velocity_value, 3).to(
                device=position.device,
                dtype=position.dtype,
            )
        values = torch.cat((position, velocity), dim=-1)
        if self.config.noise_standard_deviation > 0:
            values = values + torch.randn_like(values) * (self.config.noise_standard_deviation)
        active_value = payload.get("active")
        if active_value is None:
            active = torch.ones(batch, objects, dtype=torch.bool, device=position.device)
        else:
            active = _as_batched(active_value, 2).to(device=position.device, dtype=torch.bool)
        object_id_value = payload.get("object_id", payload.get("id"))
        if object_id_value is None:
            object_id = (
                torch.arange(objects, dtype=torch.int64, device=position.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )
        else:
            object_id = _as_batched(object_id_value, 2).to(
                device=position.device,
                dtype=torch.int64,
            )
        variances = position.new_tensor(
            [self.config.position_variance] * 3 + [self.config.velocity_variance] * 3
        )
        log_variance = variances.log().reshape(1, 1, 6).expand(batch, objects, -1)
        appearance_value = payload.get("appearance", payload.get("albedo"))
        appearance = (
            None
            if appearance_value is None
            else _as_batched(appearance_value, 3).to(
                device=position.device,
                dtype=position.dtype,
            )
        )
        if appearance is not None and self.config.appearance_dim is not None:
            target_dim = self.config.appearance_dim
            if appearance.shape[-1] < target_dim:
                appearance = torch.nn.functional.pad(
                    appearance, (0, target_dim - appearance.shape[-1])
                )
            else:
                appearance = appearance[..., :target_dim]
        radius_value = payload.get("radius")
        auxiliary: dict[str, Tensor] = {
            "world_position": values[..., :3],
            "world_velocity": values[..., 3:6],
            "world_log_variance": log_variance[..., :3],
            "object_id": object_id,
        }
        if radius_value is not None:
            radius = _as_batched(radius_value, 3).to(
                device=position.device,
                dtype=position.dtype,
            )
            auxiliary["world_radius"] = radius
        return MeasurementSet(
            modality=self.modality_name,
            sensor_id=packet.sensor_id,
            timestamp=position.new_full((batch,), packet.timestamp),
            values=values,
            log_variance=log_variance,
            existence_logits=torch.where(
                active,
                position.new_full((batch, objects), 8.0),
                position.new_full((batch, objects), -8.0),
            ),
            measurement_mask=active,
            appearance=appearance,
            class_logits=None,
            frame_id=packet.frame_id,
            supported_state_fields=("position", "velocity"),
            auxiliary=auxiliary,
        )

    def initialise_measurements(
        self,
        packets: Sequence[ObservationPacket],
        context: ObservationContext,
    ) -> MeasurementSet:
        del context
        if len(packets) != 1:
            raise ValueError("debug_oracle expects one packet whose payload may be batched")
        result = self._measure(packets[0])
        result.validate()
        return result

    def encode_measurements(
        self,
        packets: Sequence[ObservationPacket],
        prior: WorldBelief,
        predicted: PredictedMeasurements,
        cache: ModalityCache | None,
    ) -> tuple[MeasurementSet, ModalityCache]:
        del prior, predicted
        result = self.initialise_measurements(
            packets,
            ObservationContext(
                timestamp=packets[0].timestamp,
                calibration=packets[0].calibration,
                frame_id=packets[0].frame_id,
                max_objects=0,
                device=packets[0].payload["position"].device,
            ),
        )
        return result, cache or ModalityCache()

    def project(
        self,
        belief: WorldBelief,
        sensor_context: SensorContext,
    ) -> PredictedMeasurements:
        objects = belief.objects
        values = torch.cat((objects.position, objects.velocity), dim=-1)
        batch, object_count, _ = values.shape
        if objects.fast_log_variance.shape[-1] >= 6:
            log_variance = objects.fast_log_variance[..., :6]
        else:
            log_variance = values.new_full(values.shape, -4.0)
        indices = (
            torch.arange(object_count, device=values.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        return PredictedMeasurements(
            modality=self.modality_name,
            sensor_id=sensor_context.sensor_id,
            timestamp=belief.timestamp,
            values=values,
            log_variance=log_variance,
            object_ids=objects.object_id,
            belief_indices=indices,
            valid_mask=objects.active,
            visibility=objects.visibility_logit.sigmoid(),
            rois=None,
            appearance=objects.appearance,
            auxiliary={
                "world_position": objects.position,
                "world_velocity": objects.velocity,
            },
        )

    def innovation(
        self,
        measured: MeasurementSet,
        predicted: PredictedMeasurements,
        association: AssociationResult,
    ) -> InnovationSet:
        return build_innovation(
            measured=measured,
            predicted=predicted,
            association=association,
            modality_index=self.modality_index,
        )
