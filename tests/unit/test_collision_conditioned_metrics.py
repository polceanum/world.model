from __future__ import annotations

import pytest
import torch

from world_model.evaluation.collision_conditioned import (
    CollisionConditionedForecastAccumulator,
    collision_class_masks_for_forecast_window,
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


def test_collision_class_masks_are_exhaustive_and_distinguish_compound_windows() -> None:
    batch, frames, objects = 1, 3, 6
    collision = torch.zeros(batch, frames, objects, dtype=torch.bool)
    pair = torch.zeros(batch, frames, objects, objects, dtype=torch.bool)
    ground = torch.zeros_like(collision)
    wall = torch.zeros(batch, frames, objects, 4, dtype=torch.bool)
    boundary = torch.zeros(batch, frames, objects, 6, dtype=torch.bool)

    pair[0, 1, 0, 1] = True
    ground[0, 1, 1] = True
    boundary[0, 1, 1, 2] = True
    wall[0, 1, 2, 0] = True
    boundary[0, 1, 2, 0] = True
    pair[0, 1, 3, 0] = True
    ground[0, 2, 3] = True
    boundary[0, 2, 3, 2] = True
    boundary[0, 1, 4, 3] = True
    collision[0, :, :] = pair.any(dim=-1) | ground | boundary.any(dim=-1)

    classes = collision_class_masks_for_forecast_window(
        {
            "collision": collision,
            "pair_collision": pair,
            "ground_collision": ground,
            "wall_collision": wall,
            "boundary_collision": boundary,
        },
        anchor_frame=0,
        target_frame=2,
    )

    assert list(classes) == [
        "pair_only",
        "ground_only",
        "wall_only",
        "other_only",
        "compound",
        "no_collision",
    ]
    torch.testing.assert_close(
        classes["pair_only"], torch.tensor([[True, False, False, False, False, False]])
    )
    torch.testing.assert_close(
        classes["ground_only"], torch.tensor([[False, True, False, False, False, False]])
    )
    torch.testing.assert_close(
        classes["wall_only"], torch.tensor([[False, False, True, False, False, False]])
    )
    torch.testing.assert_close(
        classes["other_only"], torch.tensor([[False, False, False, False, True, False]])
    )
    torch.testing.assert_close(
        classes["compound"], torch.tensor([[False, False, False, True, False, False]])
    )
    torch.testing.assert_close(
        classes["no_collision"], torch.tensor([[False, False, False, False, False, True]])
    )


def test_collision_class_metrics_share_the_same_valid_mask_and_predictions() -> None:
    accumulator = CollisionConditionedForecastAccumulator()
    target = torch.zeros(1, 6, 3)
    model = torch.arange(1.0, 7.0).view(1, 6, 1).expand_as(target)
    constant_velocity = 2.0 * model
    class_indices = {
        "pair_only": 0,
        "ground_only": 1,
        "wall_only": 2,
        "other_only": 3,
        "compound": 4,
        "no_collision": 5,
    }
    classes = {
        name: torch.nn.functional.one_hot(
            torch.tensor(index),
            num_classes=6,
        )
        .bool()
        .unsqueeze(0)
        for name, index in class_indices.items()
    }
    collision = ~classes["no_collision"]
    accumulator.update(
        horizon="1.000s",
        predictions={"model": model, "constant_velocity": constant_velocity},
        target=target,
        valid_mask=torch.ones(1, 6, dtype=torch.bool),
        collision_mask=collision,
        collision_classes=classes,
    )

    metrics = accumulator.metrics()
    assert metrics["collision_class_pair_only_object_horizons@1.000s"] == 1.0
    assert metrics["collision_class_ground_only_model@1.000s_position_rmse_m"] == 2.0
    assert metrics["collision_class_wall_only_model@1.000s_position_coordinate_count"] == 3.0
    assert metrics["collision_class_compound_model@1.000s_position_rmse_m"] == 5.0
    assert metrics["collision_class_no_collision_model@1.000s_position_rmse_m"] == 6.0
    assert metrics[
        "collision_class_compound_model_vs_constant_velocity"
        "@1.000s_position_rmse_reduction_fraction"
    ] == pytest.approx(0.5)
