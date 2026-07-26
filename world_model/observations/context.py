"""Observation and sensor context objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from world_model.observations.packets import ObservationPacket


@dataclass(frozen=True)
class ObservationContext:
    timestamp: float
    calibration: Mapping[str, Tensor | float | int | str]
    frame_id: str
    max_objects: int
    device: torch.device
    dtype: torch.dtype = torch.float32
    training: bool = False
    predicted_regions: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorContext:
    sensor_id: str
    timestamp: float
    calibration: Mapping[str, Tensor | float | int | str]
    frame_id: str
    image_size: tuple[int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def context_from_packet(
    packet: ObservationPacket,
    *,
    max_objects: int,
    device: torch.device,
    dtype: torch.dtype,
    training: bool,
) -> ObservationContext:
    if not isinstance(packet, ObservationPacket):
        raise TypeError("packet must be an ObservationPacket")
    return ObservationContext(
        timestamp=packet.timestamp,
        calibration=packet.calibration,
        frame_id=packet.frame_id,
        max_objects=max_objects,
        device=device,
        dtype=dtype,
        training=training,
        metadata=packet.metadata,
    )
