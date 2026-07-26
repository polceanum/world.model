"""Truthful per-ingest runtime diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeStepDiagnostics:
    timestamp: float
    modality: str
    sensor_id: str
    observation_mode: str
    active_objects_before: int
    active_objects_after: int
    matched_pairs: int
    unmatched_measurements: int
    ambiguous_pairs: int
    aggregate_surprise: float
    correction_norm: float
    elapsed_milliseconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeDiagnostics:
    """In-memory diagnostics; external loggers may serialize ``records``."""

    def __init__(self) -> None:
        self.records: list[RuntimeStepDiagnostics] = []

    def reset(self) -> None:
        self.records.clear()

    def record(self, diagnostics: RuntimeStepDiagnostics) -> None:
        self.records.append(diagnostics)

    def scheduler_context(self, sensor_id: str) -> dict[str, float | int]:
        matching = [record for record in reversed(self.records) if record.sensor_id == sensor_id]
        if not matching:
            return {}
        latest = matching[0]
        return {
            "surprise": latest.aggregate_surprise,
            "association_failures": int(latest.matched_pairs == 0),
        }

    @property
    def oracle_used(self) -> bool:
        return any(record.modality == "debug_oracle" for record in self.records)

    @property
    def latest(self) -> RuntimeStepDiagnostics | None:
        return self.records[-1] if self.records else None
