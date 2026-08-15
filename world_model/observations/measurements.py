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
    source_belief_indices: Tensor | None = None
    source_object_ids: Tensor | None = None

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
        for key in (
            "position_independent_camera_axis_mask",
            "world_position_independent_axis_mask",
        ):
            axis_mask = self.auxiliary.get(key)
            if axis_mask is None:
                continue
            if axis_mask.shape != (batch, measurements, 3):
                raise ValueError(f"auxiliary.{key} must have shape [B,M,3]")
            if axis_mask.dtype != torch.bool:
                raise TypeError(f"auxiliary.{key} must use torch.bool")
        if "world_position_independent_axis_mask" in self.auxiliary and (
            "world_position" not in self.auxiliary
            or "world_position_log_variance" not in self.auxiliary
        ):
            raise ValueError(
                "auxiliary.world_position_independent_axis_mask requires world position and variance"
            )
        source_fields = (
            self.source_belief_indices,
            self.source_object_ids,
        )
        if any(source is not None for source in source_fields):
            if any(source is None for source in source_fields):
                raise ValueError(
                    "measurement source belief indices and object IDs must be provided together"
                )
            assert self.source_belief_indices is not None
            assert self.source_object_ids is not None
            for name, source in (
                ("source_belief_indices", self.source_belief_indices),
                ("source_object_ids", self.source_object_ids),
            ):
                if source.shape != (batch, measurements):
                    raise ValueError(f"{name} must have shape [B, M]")
                if source.dtype is not torch.int64:
                    raise TypeError(f"{name} must use torch.int64")
            valid_source = self.measurement_mask
            if bool(
                torch.any(
                    valid_source & ((self.source_belief_indices < 0) | (self.source_object_ids < 0))
                )
            ):
                raise ValueError(
                    "valid source-conditioned measurements require nonnegative identity"
                )
        if not torch.isfinite(self.values).all():
            raise ValueError("measurement values contain NaN or Inf")
        if not torch.isfinite(self.timestamp).all():
            raise ValueError("measurement timestamp contains NaN or Inf")
        if not torch.isfinite(self.log_variance).all():
            raise ValueError("measurement log variances contain NaN or Inf")
        if not torch.isfinite(self.existence_logits).all():
            raise ValueError("measurement existence logits contain NaN or Inf")
        if self.appearance is not None and not torch.isfinite(self.appearance).all():
            raise ValueError("measurement appearance contains NaN or Inf")
        if self.class_logits is not None and not torch.isfinite(self.class_logits).all():
            raise ValueError("measurement class logits contain NaN or Inf")

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
            source_belief_indices=(
                None if self.source_belief_indices is None else floating(self.source_belief_indices)
            ),
            source_object_ids=(
                None if self.source_object_ids is None else floating(self.source_object_ids)
            ),
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
            source_belief_indices=(
                None if self.source_belief_indices is None else self.source_belief_indices.detach()
            ),
            source_object_ids=(
                None if self.source_object_ids is None else self.source_object_ids.detach()
            ),
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
        batch, objects, dimensions = self.values.shape
        if self.timestamp.shape != (batch,):
            raise ValueError("predicted timestamp must have shape [B]")
        if self.log_variance.shape not in {
            self.values.shape,
            (batch, objects, 1),
            (1, 1, dimensions),
        }:
            raise ValueError("predicted log_variance must be [B,N,D], [B,N,1], or [1,1,D]")
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
        if self.object_ids.dtype != torch.int64:
            raise TypeError("predicted object_ids must use torch.int64")
        if self.belief_indices.dtype != torch.int64:
            raise TypeError("predicted belief_indices must use torch.int64")
        if self.rois is not None and self.rois.shape != (batch, objects, 4):
            raise ValueError("predicted rois must have shape [B, N, 4]")
        if not torch.isfinite(self.values).all():
            raise ValueError("predicted measurement contains NaN or Inf")
        if not torch.isfinite(self.timestamp).all():
            raise ValueError("predicted timestamp contains NaN or Inf")
        if not torch.isfinite(self.log_variance).all():
            raise ValueError("predicted log variances contain NaN or Inf")
        if not torch.isfinite(self.visibility).all():
            raise ValueError("predicted visibility contains NaN or Inf")
        if self.appearance is not None and not torch.isfinite(self.appearance).all():
            raise ValueError("predicted appearance contains NaN or Inf")


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
    """Explicit world-frame kinematic evidence in persistent belief-slot order.

    Velocity remains required for compatibility with the original temporal
    observer.  A modality may additionally provide a position estimate derived
    from a bounded causal trajectory history.  The optional position fields
    are kept on the same typed evidence object so the runtime applies both
    corrections atomically to ``WorldBelief`` rather than maintaining a second
    physical state.
    """

    velocity: Tensor
    log_variance: Tensor
    valid_mask: Tensor
    confidence: Tensor
    position: Tensor | None = None
    position_log_variance: Tensor | None = None
    position_valid_mask: Tensor | None = None
    axis_valid_mask: Tensor | None = None

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
        if self.axis_valid_mask is not None:
            if self.axis_valid_mask.shape != self.velocity.shape:
                raise ValueError("direct velocity axis_valid_mask must have shape [B,N,3]")
            if self.axis_valid_mask.dtype != torch.bool:
                raise TypeError("direct velocity axis_valid_mask must be torch.bool")
        if not torch.isfinite(self.velocity).all():
            raise ValueError("direct velocity contains NaN or Inf")
        if not torch.isfinite(self.log_variance).all():
            raise ValueError("direct velocity log_variance contains NaN or Inf")
        if not torch.isfinite(self.confidence).all():
            raise ValueError("direct velocity confidence contains NaN or Inf")
        if torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("direct velocity confidence must lie in [0,1]")
        position_fields = (
            self.position,
            self.position_log_variance,
            self.position_valid_mask,
        )
        if any(field is not None for field in position_fields):
            if any(field is None for field in position_fields):
                raise ValueError("direct position evidence fields must be provided together")
            assert self.position is not None
            assert self.position_log_variance is not None
            assert self.position_valid_mask is not None
            if self.position.shape != self.velocity.shape:
                raise ValueError("direct position must match direct velocity shape")
            if self.position_log_variance.shape != self.position.shape:
                raise ValueError("direct position log_variance must match position")
            if self.position_valid_mask.shape != self.valid_mask.shape:
                raise ValueError("direct position valid_mask must have shape [B,N]")
            if self.position_valid_mask.dtype != torch.bool:
                raise TypeError("direct position valid_mask must be torch.bool")
            if not torch.isfinite(self.position).all():
                raise ValueError("direct position contains NaN or Inf")
            if not torch.isfinite(self.position_log_variance).all():
                raise ValueError("direct position log_variance contains NaN or Inf")

    def resolved_axis_valid_mask(self) -> Tensor:
        """Return component support while preserving the legacy object mask.

        Historical callers supplied only ``valid_mask`` and therefore support
        all three velocity components for each valid object.  New observers
        may additionally restrict evidence to explicit world-frame axes.  The
        object mask remains authoritative in both cases.
        """

        object_valid = self.valid_mask.unsqueeze(-1)
        if self.axis_valid_mask is None:
            return object_valid.expand_as(self.velocity)
        return object_valid & self.axis_valid_mask
