from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from world_model.evaluation.neural_adaptive_physics import train_neural_physics_adapter

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "screen_extended_horizon_training.py"


def test_tail_training_screen_declares_long_tail_only() -> None:
    screen = runpy.run_path(str(_SCRIPT))
    assert screen["_TAIL_HORIZONS_SECONDS"] == (12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0)
    parsed = screen["arguments"](["--training-steps", "8", "--batch-size", "4"])
    assert parsed.training_steps == 8
    assert parsed.batch_size == 4


@pytest.mark.parametrize("horizons", [(), (12.0, 8.0), (12.0, float("nan"))])
def test_training_rejects_invalid_stability_horizons(horizons: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="stability horizons"):
        train_neural_physics_adapter(
            steps=2,
            batch_size=2,
            stability_loss_weight=1.0,
            stability_horizons=horizons,
        )
