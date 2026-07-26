"""Immutable timestamped sensor packets.

Packets intentionally carry no world-model assumptions.  Payload validation is
delegated to the registered observation module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

CalibrationValue = Tensor | float | int | str


@dataclass(frozen=True)
class ObservationPacket:
    """One asynchronous sensor observation.

    ``timestamp`` is expressed in monotonically increasing seconds.  The online
    runtime accepts a single packet or a same-timestamp group.
    """

    modality: str
    sensor_id: str
    timestamp: float
    payload: Any
    calibration: Mapping[str, CalibrationValue]
    frame_id: str
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.modality:
            raise ValueError("ObservationPacket.modality must be non-empty")
        if not self.sensor_id:
            raise ValueError("ObservationPacket.sensor_id must be non-empty")
        if not self.frame_id:
            raise ValueError("ObservationPacket.frame_id must be non-empty")
        if not math.isfinite(self.timestamp):
            raise ValueError("ObservationPacket.timestamp must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ObservationPacket.confidence must be finite and in [0, 1]")
