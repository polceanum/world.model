"""Explicit resettable runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field

from world_model.belief import WorldBelief
from world_model.observations.base import ModalityCache


@dataclass
class RuntimeState:
    belief: WorldBelief | None = None
    caches: dict[str, ModalityCache] = field(default_factory=dict)
    batch_size: int = 1
    ingest_count: int = 0

    def detach(self) -> RuntimeState:
        return RuntimeState(
            belief=None if self.belief is None else self.belief.detach(),
            caches={sensor_id: cache.detach() for sensor_id, cache in self.caches.items()},
            batch_size=self.batch_size,
            ingest_count=self.ingest_count,
        )
