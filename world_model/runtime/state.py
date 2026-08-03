"""Explicit resettable runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field

from world_model.belief import TentativeBirthState, WorldBelief
from world_model.observations.base import ModalityCache, ModalityHistory


@dataclass
class RuntimeState:
    belief: WorldBelief | None = None
    caches: dict[str, ModalityCache] = field(default_factory=dict)
    temporal_histories: dict[str, ModalityHistory] = field(default_factory=dict)
    tentative_births: dict[tuple[str, str], TentativeBirthState] = field(default_factory=dict)
    batch_size: int = 1
    ingest_count: int = 0

    def detach(self) -> RuntimeState:
        return RuntimeState(
            belief=None if self.belief is None else self.belief.detach(),
            caches={sensor_id: cache.detach() for sensor_id, cache in self.caches.items()},
            temporal_histories={
                sensor_id: history.detach()
                for sensor_id, history in self.temporal_histories.items()
            },
            tentative_births={key: state.detach() for key, state in self.tentative_births.items()},
            batch_size=self.batch_size,
            ingest_count=self.ingest_count,
        )
