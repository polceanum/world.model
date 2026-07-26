"""Truthful JSON and Markdown evaluation report emission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from world_model.utils.io import atomic_write_text


def write_evaluation_report(
    output_dir: str | Path,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    limitations: list[str],
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "evaluation.json"
    markdown_path = target / "report.md"
    payload = {"metadata": metadata, "metrics": metrics, "limitations": limitations}
    atomic_write_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Orpheus evaluation report",
        "",
        "## Protocol",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Metrics", ""])
    for key, value in sorted(metrics.items()):
        formatted = f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append(f"- {key}: `{formatted}`")
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    lines.append("")
    atomic_write_text(markdown_path, "\n".join(lines))
    return json_path, markdown_path
