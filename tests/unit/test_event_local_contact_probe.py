from __future__ import annotations

import runpy
from pathlib import Path

import torch

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/probe_event_local_contact.py"


def test_event_local_contact_probe_has_stable_defaults_and_horizon_indexing() -> None:
    probe = runpy.run_path(str(_SCRIPT))

    parsed = probe["arguments"](["--dry-run"])
    curve = probe["_curve"](torch.arange(240, dtype=torch.float64))

    assert parsed.object_count == 4
    assert Path(parsed.checkpoint).is_absolute()
    assert parsed.checkpoint.name == "model.pt"
    assert curve == {"2.00": 39.0, "4.00": 79.0, "8.00": 159.0, "12.00": 239.0}
