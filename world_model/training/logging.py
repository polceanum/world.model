"""Local console/JSONL run logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Append human-readable-safe scalar metrics to a local JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.perf_counter()

    def log(self, *, step: int, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "step": int(step),
            "split": split,
            "elapsed_seconds": time.perf_counter() - self.started_at,
        }
        for key, value in metrics.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            if isinstance(value, (int, float, bool, str)) or value is None:
                record[key] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record
