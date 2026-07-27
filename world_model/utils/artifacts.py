"""Timestamp-first naming for generated artifact directories."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_TIMESTAMP_PREFIX = re.compile(r"^\d{8}-\d{6}-")


def timestamped_artifact_path(
    path: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Prefix an artifact directory basename with a sortable UTC timestamp."""

    target = Path(path).expanduser()
    if _TIMESTAMP_PREFIX.match(target.name):
        return target
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    stamp = current.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return target.with_name(f"{stamp}-{target.name}")
