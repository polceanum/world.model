from __future__ import annotations

import pytest
import torch

from world_model.evaluation.collision_conditioned import (
    CollisionConditionedForecastAccumulator,
    collision_mask_for_forecast_window,
)


def test_collision_window_excludes_anchor_and_includes_target() -> None:
    events = torch.zeros(1, 6, 2, dtype=torch.bool)
    events[0, 1, 0] = True
    events[0, 3, 1] = True
    events[0, 5, 0] = True

    mask = collision_mask_for_forecast_window(
        events,
        anchor_frame=1,
        target_frame=3,
    )

    torch.testing.assert_close(mask, torch.tensor([[False, True]]))


def test_collision_conditioned_metrics_use_common_model_baseline_mask() -> None:
    accumulator = CollisionConditionedForecastAccumulator()
    target = torch.zeros(1, 2, 3)
    accumulator.update(
        horizon="0.500s",
        predictions={
            "model": torch.tensor([[[1.0, 1.0, 1.0], [9.0, 9.0, 9.0]]]),
            "constant_velocity": torch.tensor([[[2.0, 2.0, 2.0], [10.0, 10.0, 10.0]]]),
            "static": torch.tensor([[[3.0, 3.0, 3.0], [11.0, 11.0, 11.0]]]),
        },
        target=target,
        valid_mask=torch.tensor([[True, True]]),
        collision_mask=torch.tensor([[True, False]]),
    )

    metrics = accumulator.metrics()
    assert metrics["collision_conditioned_eligible_object_horizons@0.500s"] == 2.0
    assert metrics["collision_conditioned_object_horizons@0.500s"] == 1.0
    assert metrics["collision_conditioned_fraction@0.500s"] == 0.5
    assert metrics["collision_conditioned_model@0.500s_position_rmse_m"] == 1.0
    assert metrics["collision_conditioned_constant_velocity@0.500s_position_rmse_m"] == 2.0
    assert metrics["collision_conditioned_model@0.500s_position_coordinate_count"] == 3.0
    assert metrics[
        "collision_conditioned_model_vs_constant_velocity@0.500s_position_rmse_reduction_fraction"
    ] == pytest.approx(0.5)


def test_collision_conditioned_metrics_report_null_for_no_collision_samples() -> None:
    accumulator = CollisionConditionedForecastAccumulator()
    values = torch.zeros(1, 2, 3)
    accumulator.update(
        horizon="0.250s",
        predictions={"model": values, "constant_velocity": values},
        target=values,
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        collision_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    metrics = accumulator.metrics()
    assert metrics["collision_conditioned_object_horizons@0.250s"] == 0.0
    assert metrics["collision_conditioned_model@0.250s_position_rmse_m"] is None
    assert (
        metrics[
            "collision_conditioned_model_vs_constant_velocity"
            "@0.250s_position_rmse_reduction_fraction"
        ]
        is None
    )


def test_collision_conditioned_metrics_validate_shapes_and_mask_types() -> None:
    accumulator = CollisionConditionedForecastAccumulator()
    target = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match="collision_mask"):
        accumulator.update(
            horizon="0.100s",
            predictions={"model": target},
            target=target,
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            collision_mask=torch.ones(1, 2),
        )

    with pytest.raises(ValueError, match="prediction"):
        accumulator.update(
            horizon="0.100s",
            predictions={"model": torch.zeros(1, 3, 3)},
            target=target,
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            collision_mask=torch.ones(1, 2, dtype=torch.bool),
        )
