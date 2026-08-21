from __future__ import annotations

import json
import math

import pytest
import torch

from world_model.evaluation.evaluator import (
    _BinaryAccumulator,
    _CalibrationAccumulator,
    _ErrorAccumulator,
    _ForecastIdentityAccumulator,
    _ScenarioEvaluationAccumulator,
)
from world_model.evaluation.reports import write_evaluation_report
from world_model.evaluation.velocity_metrics import MaskedVelocityErrorAccumulator


def test_scenario_metrics_preserve_axis_event_identity_and_horizon_support() -> None:
    accumulator = _ScenarioEvaluationAccumulator(episode_count=1)
    object_mask = torch.tensor([[True, False]])
    zero = torch.zeros(1, 2, 3)

    accumulator.current_position.update(
        torch.tensor([[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]]),
        zero,
        object_mask,
    )
    accumulator.current_velocity.update(
        torch.tensor([[[3.0, 4.0, 5.0], [99.0, 99.0, 99.0]]]),
        zero,
        object_mask,
    )
    accumulator.current_calibration.update(
        torch.tensor([[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]]),
        torch.zeros_like(zero),
        zero,
        object_mask,
    )
    accumulator.target_object_frames = 5
    accumulator.predicted_object_frames = 4
    accumulator.matched_object_frames = 3
    accumulator.distance_gated_matched_object_frames = 2

    target_ids = torch.tensor([[11, -1]])
    target_indices = torch.tensor([[0, -1]])
    accumulator.tracking.update(
        torch.tensor([[7, -1]]),
        target_ids,
        target_indices,
        object_mask,
        episode_offset=0,
    )
    accumulator.tracking.update(
        torch.tensor([[8, -1]]),
        target_ids,
        target_indices,
        object_mask,
        episode_offset=0,
    )

    horizon = "0.100s"
    accumulator.forecast_position[horizon] = _ErrorAccumulator()
    accumulator.forecast_position[horizon].update(
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        zero,
        object_mask,
    )
    accumulator.forecast_velocity[horizon] = MaskedVelocityErrorAccumulator()
    accumulator.forecast_velocity[horizon].update(
        torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]]),
        zero,
        object_mask,
    )
    accumulator.calibration_by_horizon[horizon] = _CalibrationAccumulator()
    accumulator.calibration_by_horizon[horizon].update(
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        torch.zeros_like(zero),
        zero,
        object_mask,
    )
    accumulator.forecast_identity_by_horizon[horizon] = _ForecastIdentityAccumulator()
    accumulator.forecast_identity_by_horizon[horizon].update(
        torch.tensor([[11, -1]]),
        torch.tensor([[12, -1]]),
        object_mask,
        object_mask,
    )
    accumulator.forecast_identity_by_horizon[horizon].update(
        torch.tensor([[11, 77]]),
        torch.tensor([[11, 88]]),
        object_mask,
        object_mask,
    )
    event_logits = torch.tensor([[10.0, -10.0]])
    event_target = torch.tensor([[True, True]])
    event_mask = torch.tensor([[True, True]])
    accumulator.collision_events.update(event_logits, event_target, event_mask)
    accumulator.collision_events_by_horizon[horizon] = _BinaryAccumulator()
    accumulator.collision_events_by_horizon[horizon].update(
        event_logits,
        event_target,
        event_mask,
    )
    accumulator.forecast_target_count[horizon] = 5
    accumulator.forecast_tracked_count[horizon] = 4
    accumulator.forecast_active_count[horizon] = 3
    accumulator.forecast_predictable_target_count[horizon] = 4
    accumulator.forecast_censored_tracked_count[horizon] = 1

    metrics = accumulator.metrics(
        scenario="elastic_pairs",
        horizons=(horizon, "0.250s"),
        detection_threshold_label="0.500m",
    )
    prefix = "scenario_elastic_pairs_"
    assert metrics[f"{prefix}episode_count"] == 1.0
    assert metrics[f"{prefix}posterior_current_position_rmse_m"] == pytest.approx(
        math.sqrt(14.0 / 3.0)
    )
    assert metrics[f"{prefix}posterior_current_position_x_count"] == 1.0
    assert metrics[f"{prefix}posterior_current_velocity_rmse_mps"] == pytest.approx(
        math.sqrt(50.0 / 3.0)
    )
    assert metrics[f"{prefix}posterior_current_velocity_sse"] == pytest.approx(50.0)
    assert metrics[f"{prefix}posterior_current_velocity_x_sse"] == pytest.approx(9.0)
    assert metrics[f"{prefix}posterior_current_velocity_z_count"] == 1.0
    expected_current_nll_sum = 0.5 * (14.0 + 3.0 * math.log(2.0 * math.pi))
    assert metrics[f"{prefix}posterior_current_position_gaussian_nll_sum"] == pytest.approx(
        expected_current_nll_sum
    )
    assert metrics[f"{prefix}posterior_current_position_sharpness_std_sum"] == 3.0
    assert metrics[f"{prefix}posterior_current_position_calibration_coordinate_count"] == 3.0
    assert metrics[f"{prefix}posterior_current_position_x_gaussian_nll_sum"] == pytest.approx(
        0.5 * (1.0 + math.log(2.0 * math.pi))
    )
    assert metrics[f"{prefix}current_assignment_target_coverage"] == pytest.approx(0.6)
    assert metrics[f"{prefix}current_detection_precision@0.500m"] == pytest.approx(0.5)
    assert metrics[f"{prefix}distance_gated_identity_switches"] == 1.0
    assert metrics[f"{prefix}distance_gated_object_frame_associations"] == 2.0
    assert metrics[f"{prefix}model@{horizon}_position_rmse_m"] == pytest.approx(
        math.sqrt(1.0 / 3.0)
    )
    assert metrics[f"{prefix}model@{horizon}_position_y_count"] == 1.0
    assert metrics[f"{prefix}model@{horizon}_velocity_rmse_mps"] == pytest.approx(
        math.sqrt(4.0 / 3.0)
    )
    assert metrics[f"{prefix}model@{horizon}_velocity_sse"] == pytest.approx(4.0)
    assert metrics[f"{prefix}model@{horizon}_velocity_y_sse"] == pytest.approx(4.0)
    assert metrics[f"{prefix}model@{horizon}_position_gaussian_nll_sum"] == pytest.approx(
        0.5 * (1.0 + 3.0 * math.log(2.0 * math.pi))
    )
    assert metrics[f"{prefix}model@{horizon}_position_calibration_coordinate_count"] == 3.0
    assert metrics[f"{prefix}model@{horizon}_position_coverage_90"] == 1.0
    assert metrics[f"{prefix}model@{horizon}_position_calibration_error90"] == pytest.approx(0.1)
    assert metrics[f"{prefix}forecast_identity@{horizon}_mismatch_count"] == 1.0
    assert metrics[f"{prefix}forecast_identity@{horizon}_eligible_count"] == 2.0
    assert metrics[f"{prefix}forecast_identity@{horizon}_association_count"] == 2.0
    assert metrics[f"{prefix}forecast_identity@{horizon}_association_coverage"] == 1.0
    assert metrics[f"{prefix}forecast_identity@{horizon}_mismatch_rate"] == 0.5
    assert metrics[f"{prefix}collision@{horizon}_true_positive_count"] == 1.0
    assert metrics[f"{prefix}collision@{horizon}_false_negative_count"] == 1.0
    assert metrics[f"{prefix}collision@{horizon}_f1"] == pytest.approx(2.0 / 3.0)
    assert metrics[f"{prefix}forecast_target_count@{horizon}"] == 5.0
    assert metrics[f"{prefix}forecast_tracked_count@{horizon}"] == 4.0
    assert metrics[f"{prefix}forecast_active_count@{horizon}"] == 3.0
    assert metrics[f"{prefix}forecast_evaluated_object_horizons@{horizon}"] == 1.0
    assert metrics[f"{prefix}model@0.250s_position_rmse_m"] is None
    assert metrics[f"{prefix}model@0.250s_position_coordinate_count"] == 0.0
    assert metrics[f"{prefix}model@0.250s_position_gaussian_nll_sum"] == 0.0
    assert metrics[f"{prefix}model@0.250s_position_calibration_error90"] is None
    assert metrics[f"{prefix}forecast_identity@0.250s_association_count"] == 0.0
    assert metrics[f"{prefix}forecast_identity@0.250s_eligible_count"] == 0.0
    assert metrics[f"{prefix}forecast_identity@0.250s_association_coverage"] is None
    assert metrics[f"{prefix}forecast_identity@0.250s_mismatch_rate"] is None
    assert metrics[f"{prefix}collision@0.250s_evaluated"] == 0.0


def test_scenario_metrics_are_persisted_in_json_and_markdown(tmp_path) -> None:
    metric_name = "scenario_baseline_model@1.000s_position_rmse_m"
    json_path, markdown_path = write_evaluation_report(
        tmp_path,
        metadata={"per_scenario_metrics_schema": "clean_primary_additive_support_diagnostic_v3"},
        metrics={metric_name: 0.125},
        limitations=[],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metrics"][metric_name] == 0.125
    assert f"- {metric_name}: `0.125`" in markdown_path.read_text(encoding="utf-8")


def test_two_scenario_slices_are_an_additive_partition_of_pooled_support() -> None:
    """Exercise a genuine mixed batch, not the prior singleton scenario case."""

    prediction = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    target = torch.zeros_like(prediction)
    pooled_mask = torch.tensor([[True, False], [True, True]])
    row_masks = {
        "baseline": torch.tensor([True, False]),
        "elastic_pairs": torch.tensor([False, True]),
    }
    pooled = _ErrorAccumulator()
    pooled.update(prediction, target, pooled_mask)
    pooled_events = _BinaryAccumulator()
    event_logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    event_target = torch.tensor([[True, False], [True, True]])
    pooled_events.update(event_logits, event_target, pooled_mask)
    horizon = "0.100s"
    slices = {name: _ScenarioEvaluationAccumulator(episode_count=1) for name in row_masks}
    for name, row_mask in row_masks.items():
        scenario_mask = pooled_mask & row_mask.unsqueeze(-1)
        slices[name].current_position.update(prediction, target, scenario_mask)
        slices[name].forecast_position[horizon] = _ErrorAccumulator()
        slices[name].forecast_position[horizon].update(
            prediction,
            target,
            scenario_mask,
        )
        slices[name].collision_events.update(
            event_logits,
            event_target,
            scenario_mask,
        )
        slices[name].target_object_frames = int(scenario_mask.sum())
        slices[name].predicted_object_frames = int(scenario_mask.sum())
        slices[name].matched_object_frames = int(scenario_mask.sum())
        slices[name].distance_gated_matched_object_frames = int(scenario_mask.sum())

    assert sum(item.episode_count for item in slices.values()) == 2
    assert sum(item.current_position.count for item in slices.values()) == pooled.count
    assert sum(item.current_position.squared_sum for item in slices.values()) == pytest.approx(
        pooled.squared_sum
    )
    assert sum(item.current_position.absolute_sum for item in slices.values()) == pytest.approx(
        pooled.absolute_sum
    )
    for axis in range(3):
        assert (
            sum(item.current_position.axis_count[axis] for item in slices.values())
            == (pooled.axis_count[axis])
        )
        assert sum(
            item.current_position.axis_squared_sum[axis] for item in slices.values()
        ) == pytest.approx(pooled.axis_squared_sum[axis])
    assert sum(item.target_object_frames for item in slices.values()) == int(pooled_mask.sum())
    assert sum(item.forecast_position[horizon].count for item in slices.values()) == pooled.count
    assert sum(item.collision_events.true_positive for item in slices.values()) == (
        pooled_events.true_positive
    )
    assert sum(item.collision_events.false_positive for item in slices.values()) == (
        pooled_events.false_positive
    )
    assert sum(item.collision_events.false_negative for item in slices.values()) == (
        pooled_events.false_negative
    )
    assert sum(item.collision_events.true_negative for item in slices.values()) == (
        pooled_events.true_negative
    )
