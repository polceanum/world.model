"""Deterministic compute scheduler for global and residual observation paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from world_model.observations.measurements import PredictedMeasurements
from world_model.observations.packets import ObservationPacket

if TYPE_CHECKING:
    from world_model.belief.world_belief import WorldBelief


class ObservationMode(str, Enum):
    FAST_ROI = "FAST_ROI"
    GLOBAL_DISCOVERY = "GLOBAL_DISCOVERY"
    RECOVERY = "RECOVERY"
    SKIP = "SKIP"


@dataclass
class SchedulerSensorState:
    steps_since_global: int = 0
    association_failures: int = 0
    last_surprise: float = 0.0
    last_mode: ObservationMode = ObservationMode.GLOBAL_DISCOVERY


class ObservationScheduler:
    """Threshold scheduler; state is sensor-local and resettable."""

    def __init__(
        self,
        *,
        global_every_steps: int = 15,
        uncertainty_threshold: float = 4.0,
        surprise_threshold: float = 8.0,
        failure_threshold: int = 2,
    ) -> None:
        if global_every_steps <= 0:
            raise ValueError("global_every_steps must be positive")
        if uncertainty_threshold <= 0:
            raise ValueError("uncertainty_threshold must be positive metres")
        self.global_every_steps = global_every_steps
        self.uncertainty_threshold = uncertainty_threshold
        self.surprise_threshold = surprise_threshold
        self.failure_threshold = failure_threshold
        self._sensor_state: dict[str, SchedulerSensorState] = {}

    def reset(self) -> None:
        self._sensor_state.clear()

    def state_for(self, sensor_id: str) -> SchedulerSensorState:
        return self._sensor_state.setdefault(sensor_id, SchedulerSensorState())

    def choose(
        self,
        *,
        packet: ObservationPacket,
        belief: WorldBelief | None,
        predicted: PredictedMeasurements | None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> ObservationMode:
        state = self.state_for(packet.sensor_id)
        if belief is None or predicted is None or not predicted.valid_mask.any():
            return ObservationMode.GLOBAL_DISCOVERY
        diagnostics = diagnostics or {}
        surprise = float(diagnostics.get("surprise", state.last_surprise))
        failures = max(
            int(diagnostics.get("association_failures", 0)),
            state.association_failures,
        )
        if surprise >= self.surprise_threshold or failures >= self.failure_threshold:
            return ObservationMode.RECOVERY
        objects = belief.objects
        if hasattr(objects, "fast_log_variance"):
            # Localization drives whether projected ROIs remain trustworthy.
            # Velocity/modal uncertainty may legitimately stay high at birth.
            # The public threshold is expressed in metres, so compare it with
            # positional standard deviation rather than variance (m²).
            position_std = (0.5 * objects.fast_log_variance[..., :3]).exp()
            active = objects.active.unsqueeze(-1)
            active_position_std = position_std.masked_fill(~active, 0.0)
            max_position_std = float(active_position_std.max().detach().cpu())
            if max_position_std > self.uncertainty_threshold:
                return ObservationMode.RECOVERY
        # ``global_every_steps`` is the distance between global frames, not
        # the number of FAST frames allowed after one.  The counter stores
        # completed FAST frames since the last global frame, so cadence three
        # must emit GLOBAL, FAST, FAST, GLOBAL rather than inserting a third
        # FAST frame.
        if state.steps_since_global >= self.global_every_steps - 1:
            return ObservationMode.GLOBAL_DISCOVERY
        return ObservationMode.FAST_ROI

    def record(
        self,
        sensor_id: str,
        mode: ObservationMode,
        *,
        surprise: float = 0.0,
        association_failures: int = 0,
    ) -> None:
        state = self.state_for(sensor_id)
        state.last_mode = mode
        state.last_surprise = surprise
        state.association_failures = (
            state.association_failures + association_failures if association_failures > 0 else 0
        )
        if mode in {ObservationMode.GLOBAL_DISCOVERY, ObservationMode.RECOVERY}:
            state.steps_since_global = 0
        elif mode == ObservationMode.FAST_ROI:
            state.steps_since_global += 1
