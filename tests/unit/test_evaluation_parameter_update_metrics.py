from __future__ import annotations

import pytest
import torch

from world_model.evaluation.parameter_metrics import OnlineParameterUpdateAccumulator


def test_directional_parameter_metrics_report_helpful_harmful_and_zero_updates() -> None:
    accumulator = OnlineParameterUpdateAccumulator()
    target = torch.full((1, 3, 1), 0.5)
    mask = torch.ones(1, 3, dtype=torch.bool)

    accumulator.update(
        "restitution",
        torch.tensor([[[0.4], [0.6], [0.8]]]),
        torch.tensor([[[0.5], [0.7], [0.8]]]),
        target,
        mask,
    )

    metrics = accumulator.metrics()
    assert metrics["informative_restitution_pre_update_mae"] == pytest.approx(1.0 / 6.0)
    assert metrics["informative_restitution_post_update_mae"] == pytest.approx(1.0 / 6.0)
    assert metrics["informative_restitution_signed_error_reduction_mean"] == pytest.approx(
        0.0, abs=1.0e-7
    )
    assert metrics["informative_restitution_positive_error_reduction_rate"] == pytest.approx(
        1.0 / 3.0
    )
    assert metrics["informative_restitution_absolute_update_mean"] == pytest.approx(1.0 / 15.0)
    assert metrics["informative_restitution_update_count"] == 3.0


def test_directional_parameter_metrics_mask_uninformative_updates() -> None:
    accumulator = OnlineParameterUpdateAccumulator()
    accumulator.update(
        "drag",
        torch.tensor([[[0.02], [0.10], [0.30]]]),
        torch.tensor([[[0.04], [0.08], [0.10]]]),
        torch.tensor([[[0.05], [0.05], [0.05]]]),
        torch.tensor([[True, False, False]]),
    )

    metrics = accumulator.metrics()
    assert metrics["informative_drag_pre_update_mae"] == pytest.approx(0.03)
    assert metrics["informative_drag_post_update_mae"] == pytest.approx(0.01)
    assert metrics["informative_drag_signed_error_reduction_mean"] == pytest.approx(0.02)
    assert metrics["informative_drag_positive_error_reduction_rate"] == 1.0
    assert metrics["informative_drag_absolute_update_mean"] == pytest.approx(0.02)
    assert metrics["informative_drag_update_count"] == 1.0
    assert metrics["informative_restitution_pre_update_mae"] is None
    assert metrics["informative_restitution_update_count"] == 0.0


def test_directional_parameter_metrics_validate_scalar_contract() -> None:
    accumulator = OnlineParameterUpdateAccumulator()
    values = torch.zeros(1, 2, 1)

    with pytest.raises(ValueError, match="mask"):
        accumulator.update(
            "drag",
            values,
            values,
            values,
            torch.ones(1, 2),
        )
    with pytest.raises(ValueError, match="trailing dimension"):
        accumulator.update(
            "drag",
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, dtype=torch.bool),
        )
