from __future__ import annotations

import runpy
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_extended_horizon.py"


def test_extended_horizon_probe_has_fixed_long_horizon_contract() -> None:
    probe = runpy.run_path(str(_SCRIPT))
    assert probe["_display_horizons"](24.0) == (2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0)
    assert probe["_display_horizons"](48.0)[-1] == 48.0
    parsed = probe["arguments"](["--object-count", "6", "--forecast-seconds", "48"])
    assert parsed.object_count == 6
    assert parsed.forecast_seconds == 48.0
