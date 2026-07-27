"""Shared structured measurement data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def _move_tensor(value: Tensor, *args: object, **kwargs: object) -> Tensor:
    return value.to(*args, **kwargs)


@dataclass
class MeasurementSet:
    """Unordered measurement proposals emitted by an observation module."""

    modality: str
    sensor_id: str
    timestamp: Tensor
    values: Tensor
    log_variance: Tensor
    existence_logits: Tensor
    measurement_mask: Tensor
    appearance: Tensor | None
    class_logits: Tensor | None
    frame_id: str
    supported_state_fields: tuple[str, ...]
    auxiliary: dict[str, Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if self.timestamp.ndim != 1:
            raise ValueError("measurement timestamp must have shape [B]")
        if self.values.ndim != 3:
            raise ValueError("measurement values must have shape [B, M, D]")
        batch, measurements, dimensions = self.values.shape
        if self.timestamp.shape[0] != batch:
            raise ValueError("measurement timestamp batch does not match values")
        if self.log_variance.shape not in {
            self.values.shape,
            (batch, measurements, 1),
            (1, 1, dimensions),
        }:
            raise ValueError("measurement log_variance must be [B,M,D], [B,M,1], or [1,1,D]")
        if self.existence_logits.shape != (batch, measurements):
            raise ValueError("existence_logits must have shape [B, M]")
        if self.measurement_mask.shape != (batch, measurements):
            raise ValueError("measurement_mask must have shape [B, M]")
        if self.measurement_mask.dtype != torch.bool:
            raise TypeError("measurement_mask must be torch.bool")
        if self.appearance is not None and self.appearance.shape[:2] != (
            batch,
            measurements,
        ):
            raise ValueError("appearance must begin with shape [B, M]")
        if self.class_logits is not None and self.class_logits.shape[:2] != (
            batch,
            measurements,
        ):
            raise ValueError("class_logits must begin with shape [B, M]")
        if not torch.isfinite(self.values).all():
            raise ValueError("measurement values contain NaN or Inf")
        if not torch.isfinite(self.log_variance).all():
            raise ValueError("measurement log variances contain NaN or Inf")

    def to(self, *args: object, **kwargs: object) -> MeasurementSet:
        """Return a device/dtype converted copy without changing integer masks."""

        def floating(value: Tensor) -> Tensor:
            if value.is_floating_point():
                return _move_tensor(value, *args, **kwargs)
            device = kwargs.get("device")
            if device is None and args:
                device = args[0]
            return value.to(device=device) if device is not None else value

        return MeasurementSet(
            modality=self.modality,
            sensor_id=self.sensor_id,
            timestamp=floating(self.timestamp),
            values=floating(self.values),
            log_variance=floating(self.log_variance),
            existence_logits=floating(self.existence_logits),
            measurement_mask=floating(self.measurement_mask),
            appearance=None if self.appearance is None else floating(self.appearance),
            class_logits=None if self.class_logits is None else floating(self.class_logits),
            frame_id=self.frame_id,
            supported_state_fields=self.supported_state_fields,
            auxiliary={key: floating(value) for key, value in self.auxiliary.items()},
        )

    def detach(self) -> MeasurementSet:
        """Detach tensor fields for ephemeral runtime diagnostics."""

        return MeasurementSet(
            modality=self.modality,
            sensor_id=self.sensor_id,
            timestamp=self.timestamp.detach(),
            values=self.values.detach(),
            log_variance=self.log_variance.detach(),
            existence_logits=self.existence_logits.detach(),
            measurement_mask=self.measurement_mask.detach(),
            appearance=None if self.appearance is None else self.appearance.detach(),
            class_logits=(None if self.class_logits is None else self.class_logits.detach()),
            frame_id=self.frame_id,
            supported_state_fields=self.supported_state_fields,
            auxiliary={key: value.detach() for key, value in self.auxiliary.items()},
        )


@dataclass
class PredictedMeasurements:
    """Expected sensor-space evidence projected from a prior belief."""

    modality: str
    sensor_id: str
    timestamp: Tensor
    values: Tensor
    log_variance: Tensor
    object_ids: Tensor
    belief_indices: Tensor
    valid_mask: Tensor
    visibility: Tensor
    rois: Tensor | None = None
    appearance: Tensor | None = None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("predicted values must have shape [B, N, D]")
        batch, objects, _ = self.values.shape
        for name, tensor in (
            ("object_ids", self.object_ids),
            ("belief_indices", self.belief_indices),
            ("valid_mask", self.valid_mask),
            ("visibility", self.visibility),
        ):
            if tensor.shape != (batch, objects):
                raise ValueError(f"{name} must have shape [B, N]")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("predicted valid_mask must be torch.bool")
        if self.rois is not None and self.rois.shape != (batch, objects, 4):
            raise ValueError("predicted rois must have shape [B, N, 4]")
        if not torch.isfinite(self.values).all():
            raise ValueError("predicted measurement contains NaN or Inf")


@dataclass
class InnovationSet:
    """Residual evidence for associated measurement/prior pairs."""

    modality: str
    residual: Tensor
    whitened_residual: Tensor
    innovation_norm: Tensor
    belief_indices: Tensor
    measurement_indices: Tensor
    pair_mask: Tensor
    log_likelihood: Tensor
    modality_index: Tensor
    event_features: Tensor
    auxiliary: dict[str, Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if self.residual.ndim != 3:
            raise ValueError("innovation residual must have shape [B, P, D]")
        batch, pairs, _ = self.residual.shape
        expected = (batch, pairs)
        for name, tensor in (
            ("innovation_norm", self.innovation_norm),
            ("belief_indices", self.belief_indices),
            ("measurement_indices", self.measurement_indices),
            ("pair_mask", self.pair_mask),
            ("log_likelihood", self.log_likelihood),
        ):
            if tensor.shape != expected:
                raise ValueError(f"{name} must have shape [B, P]")
        if self.pair_mask.dtype != torch.bool:
            raise TypeError("innovation pair_mask must be torch.bool")


@dataclass
class DirectVelocityEvidence:
    """Explicit world-frame velocity evidence in persistent belief-slot order."""

    velocity: Tensor
    log_variance: Tensor
    valid_mask: Tensor
    confidence: Tensor

    def validate(self) -> None:
        if self.velocity.ndim != 3 or self.velocity.shape[-1] != 3:
            raise ValueError("direct velocity must have shape [B,N,3]")
        if self.log_variance.shape != self.velocity.shape:
            raise ValueError("direct velocity log_variance must match velocity")
        if self.valid_mask.shape != self.velocity.shape[:2]:
            raise ValueError("direct velocity valid_mask must have shape [B,N]")
        if self.confidence.shape != self.valid_mask.shape:
            raise ValueError("direct velocity confidence must have shape [B,N]")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("direct velocity valid_mask must be torch.bool")
        if not torch.isfinite(self.velocity).all():
            raise ValueError("direct velocity contains NaN or Inf")
        if not torch.isfinite(self.log_variance).all():
            raise ValueError("direct velocity log_variance contains NaN or Inf")
        if not torch.isfinite(self.confidence).all():
            raise ValueError("direct velocity confidence contains NaN or Inf")
        if torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("direct velocity confidence must lie in [0,1]")
