from __future__ import annotations

import math

import pytest
import torch

from world_model.evaluation.evaluator import _ErrorAccumulator


def test_position_accumulator_reports_each_world_axis() -> None:
    accumulator = _ErrorAccumulator()
    accumulator.update(
        prediction=torch.tensor(
            [
                [
                    [1.0, -2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ]
        ),
        target=torch.zeros(1, 2, 3),
        mask=torch.tensor([[True, False]]),
    )

    metrics = accumulator.metrics("model@1.000s")
    assert metrics["model@1.000s_position_rmse_m"] == pytest.approx(math.sqrt(14.0 / 3.0))
    assert metrics["model@1.000s_position_x_rmse_m"] == 1.0
    assert metrics["model@1.000s_position_y_rmse_m"] == 2.0
    assert metrics["model@1.000s_position_z_rmse_m"] == 3.0
    assert metrics["model@1.000s_position_x_count"] == 1.0


def test_position_accumulator_reports_null_axes_without_matches() -> None:
    metrics = _ErrorAccumulator().metrics("posterior_current")

    assert metrics["posterior_current_position_rmse_m"] is None
    assert metrics["posterior_current_position_x_rmse_m"] is None
    assert metrics["posterior_current_position_x_count"] == 0.0
