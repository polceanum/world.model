from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch
from torch import nn

from world_model.runtime import OnlineWorldModel
from world_model.training.loop import (
    TrainingBatchResult,
    _distance_gate_physical_matches,
    _globally_weight_horizon_details,
    _group_closed_loop_terms,
    _select_rollout_anchor_frames,
    _weighted_closed_loop_total,
    _weighted_measurement_total,
    physical_validation_metrics,
    rollout_horizon_loss_key,
    select_closed_loop_window,
)
from world_model.training.trainer import (
    _gradient_clip_diagnostics,
    _rollout_selection_improves,
    _rollout_selection_is_compatible,
    _rollout_selection_metrics,
    _validation_loader_result,
    _validation_protocol_checkpoint_metrics,
    _validation_step,
    measurement_pretrain_frame_index,
    set_closed_loop_trainable_scope,
    set_global_perception_trainable,
)
from world_model.utils.config import load_config


def test_fixed_pretraining_sweeps_every_frame_for_every_loader_batch() -> None:
    loader_batches = 4
    total_frames = 16
    visited = {batch_index: [] for batch_index in range(loader_batches)}

    for step in range(loader_batches * total_frames):
        batch_index = step % loader_batches
        visited[batch_index].append(
            measurement_pretrain_frame_index(
                step,
                loader_batches=loader_batches,
                total_frames=total_frames,
                fixed_dataset=True,
            )
        )

    expected = list(range(total_frames))
    assert all(frame_indices == expected for frame_indices in visited.values())


def test_pretraining_frame_index_rejects_empty_axes() -> None:
    for loader_batches, total_frames in ((0, 16), (4, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            measurement_pretrain_frame_index(
                0,
                loader_batches=loader_batches,
                total_frames=total_frames,
                fixed_dataset=True,
            )


def test_streaming_pretraining_samples_a_valid_frame() -> None:
    sampled = {
        measurement_pretrain_frame_index(
            step,
            loader_batches=4,
            total_frames=7,
            fixed_dataset=False,
        )
        for step in range(32)
    }
    assert sampled
    assert sampled <= set(range(7))


def test_global_perception_freeze_leaves_fast_roi_trainable() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_global_perception_trainable(model, trainable=False)

    assert not any(parameter.requires_grad for parameter in rgb.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in rgb.global_detector.parameters())
    assert all(parameter.requires_grad for parameter in rgb.roi_updater.parameters())
    assert isinstance(rgb.roi_updater, nn.Module)

    set_global_perception_trainable(model, trainable=True)
    assert all(parameter.requires_grad for parameter in rgb.backbone.parameters())
    assert all(parameter.requires_grad for parameter in rgb.global_detector.parameters())


def test_dynamics_only_scope_preserves_rgb_and_filter_weights() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    set_closed_loop_trainable_scope(model, scope="dynamics")

    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("dynamics.")
    )

    set_closed_loop_trainable_scope(model, scope="all")
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    ("pre_clip", "maximum", "expected_coefficient", "expected_applied"),
    [
        (0.0, 1.0, 1.0, 0.0),
        (0.5, 1.0, 1.0, 0.5),
        (2.0, 1.0, 1.0 / (2.0 + 1.0e-6), 2.0 / (2.0 + 1.0e-6)),
    ],
)
def test_gradient_clip_diagnostics_match_pytorch_coefficient(
    pre_clip: float,
    maximum: float,
    expected_coefficient: float,
    expected_applied: float,
) -> None:
    coefficient, applied = _gradient_clip_diagnostics(pre_clip, maximum)

    assert coefficient == pytest.approx(expected_coefficient)
    assert applied == pytest.approx(expected_applied)


def test_state_dynamics_scope_freezes_rgb_and_trains_filter_dynamics_identifier() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    set_closed_loop_trainable_scope(model, scope="state_dynamics")

    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert all(parameter.requires_grad for parameter in model.updater.parameters())
    assert model.identifier is not None
    assert all(
        parameter.requires_grad
        for name, parameter in model.identifier.named_parameters()
        if not name.startswith("variance_head.")
    )
    assert not any(
        parameter.requires_grad for parameter in model.identifier.variance_head.parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.observation_modules.parameters())


def test_state_dynamics_roi_scope_trains_fast_rgb_without_global_perception() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_closed_loop_trainable_scope(model, scope="state_dynamics_roi")

    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert all(parameter.requires_grad for parameter in model.updater.parameters())
    assert model.identifier is not None
    assert all(
        parameter.requires_grad
        for name, parameter in model.identifier.named_parameters()
        if not name.startswith("variance_head.")
    )
    assert not any(
        parameter.requires_grad for parameter in model.identifier.variance_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for name, parameter in rgb.roi_updater.named_parameters()
        if not name.startswith("event_head.")
    )
    assert not any(parameter.requires_grad for parameter in rgb.roi_updater.event_head.parameters())
    assert not any(parameter.requires_grad for parameter in rgb.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in rgb.global_detector.parameters())


def test_closed_loop_terms_expose_physical_components_without_double_counting() -> None:
    reference = torch.zeros(())
    terms = _group_closed_loop_terms(
        {
            "state_position": torch.tensor(1.0),
            "state_velocity": torch.tensor(3.0),
            "rollout_position": torch.tensor(2.0),
            "rollout_position_x": torch.tensor(1.0),
            "rollout_position_y": torch.tensor(2.0),
            "rollout_position_z": torch.tensor(3.0),
            "rollout_velocity": torch.tensor(6.0),
        },
        reference,
    )

    assert terms["state_position"].item() == 1.0
    assert terms["state_velocity"].item() == 3.0
    assert terms["state"].item() == 2.0
    assert terms["rollout_position"].item() == 2.0
    assert terms["rollout_position_x"].item() == 1.0
    assert terms["rollout_velocity"].item() == 6.0
    assert terms["rollout"].item() == 4.0
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {"state": 2.0, "rollout": 3.0},
        ),
        torch.tensor(16.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {
                "state": 100.0,
                "state_position": 5.0,
                "state_velocity": 0.5,
                "rollout_position": 2.0,
                "rollout_velocity": 0.25,
            },
        ),
        torch.tensor(12.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {
                "rollout_position_x": 2.0,
                "rollout_position_y": 1.0,
                "rollout_position_z": 1.0,
                "rollout_velocity": 0.25,
            },
        ),
        torch.tensor(10.5),
    )


def test_measurement_weights_keep_metric_position_primary() -> None:
    losses = {
        "rgb_world_position": torch.tensor(0.2),
        "rgb_raw_centre": torch.tensor(0.1),
        "rgb_nll": torch.tensor(-3.0),
        "future_term": torch.tensor(0.1),
    }

    total = _weighted_measurement_total(
        losses,
        {
            "rgb_world_position": 8.0,
            "rgb_raw_centre": 2.0,
            "rgb_nll": 0.05,
        },
    )

    torch.testing.assert_close(total, torch.tensor(1.75))


def test_rollout_horizons_are_weighted_after_per_horizon_averaging() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    details = {
        rollout_horizon_loss_key("rollout_position", 0.1): torch.tensor(1.0),
        rollout_horizon_loss_key("rollout_position", 0.25): torch.tensor(2.0),
        rollout_horizon_loss_key("rollout_position", 0.5): torch.tensor(4.0),
        rollout_horizon_loss_key("rollout_position_x", 0.1): torch.tensor(2.0),
        rollout_horizon_loss_key("rollout_position_x", 0.25): torch.tensor(4.0),
        rollout_horizon_loss_key("rollout_position_x", 0.5): torch.tensor(8.0),
        rollout_horizon_loss_key("rollout_velocity", 0.1): torch.tensor(3.0),
        rollout_horizon_loss_key("rollout_velocity", 0.25): torch.tensor(3.0),
        rollout_horizon_loss_key("rollout_velocity", 0.5): torch.tensor(3.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))

    torch.testing.assert_close(
        balanced["rollout_position"],
        torch.tensor((1.0 * 1.0 + 1.5 * 2.0 + 2.0 * 4.0) / 4.5),
    )
    torch.testing.assert_close(balanced["rollout_velocity"], torch.tensor(3.0))
    torch.testing.assert_close(
        balanced["rollout_position_x"],
        torch.tensor((1.0 * 2.0 + 1.5 * 4.0 + 2.0 * 8.0) / 4.5),
    )


def test_missing_long_horizon_does_not_renormalize_short_losses() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    details = {
        rollout_horizon_loss_key("rollout_position", 0.1): torch.tensor(1.0),
        rollout_horizon_loss_key("rollout_position", 0.25): torch.tensor(2.0),
        rollout_horizon_loss_key("rollout_position_x", 0.1): torch.tensor(2.0),
        rollout_horizon_loss_key("rollout_position_x", 0.25): torch.tensor(4.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))

    torch.testing.assert_close(
        balanced["rollout_position"],
        torch.tensor((1.0 * 1.0 + 1.5 * 2.0) / 4.5),
    )
    torch.testing.assert_close(
        balanced["rollout_position_x"],
        torch.tensor((1.0 * 2.0 + 1.5 * 4.0) / 4.5),
    )


def test_legacy_axis_horizon_normalization_remains_explicitly_available() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["training.normalize_rollout_axes_over_configured_horizons=false"],
    )
    details = {
        "rollout_position_x": torch.tensor(7.0),
        rollout_horizon_loss_key("rollout_position_x", 0.1): torch.tensor(2.0),
        rollout_horizon_loss_key("rollout_position_x", 0.25): torch.tensor(4.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))

    torch.testing.assert_close(balanced["rollout_position_x"], torch.tensor(7.0))


def test_axis_weighted_total_uses_fixed_configured_horizon_denominator() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    details = {
        "rollout_position_x": torch.tensor(7.0),
        rollout_horizon_loss_key("rollout_position_x", 0.1): torch.tensor(2.0),
        rollout_horizon_loss_key("rollout_position_x", 0.25): torch.tensor(4.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))
    terms = _group_closed_loop_terms(balanced, torch.zeros(()))
    total = _weighted_closed_loop_total(
        terms,
        {
            "rollout_position_x": 1.0,
            "rollout_position_y": 0.0,
            "rollout_position_z": 0.0,
            "rollout_velocity": 0.0,
        },
    )

    expected = torch.tensor((1.0 * 2.0 + 1.5 * 4.0) / 4.5)
    torch.testing.assert_close(balanced["rollout_position_x"], expected)
    torch.testing.assert_close(total, expected)


def test_rollout_anchor_limit_spreads_work_and_preserves_earliest_anchor() -> None:
    config = load_config("configs/tiny_overfit.yaml")

    assert _select_rollout_anchor_frames(
        config,
        window_start=0,
        window_stop=6,
        total_frames=16,
        rollout_anchors_per_window=None,
    ) == tuple(range(6))
    assert _select_rollout_anchor_frames(
        config,
        window_start=0,
        window_stop=6,
        total_frames=16,
        rollout_anchors_per_window=2,
    ) == (0, 5)
    assert _select_rollout_anchor_frames(
        config,
        window_start=0,
        window_stop=6,
        total_frames=16,
        rollout_anchors_per_window=1,
    ) == (0,)


def test_rollout_anchor_limit_rejects_nonpositive_values() -> None:
    config = load_config("configs/tiny_overfit.yaml")

    with pytest.raises(ValueError, match="rollout_anchors_per_window"):
        _select_rollout_anchor_frames(
            config,
            window_start=0,
            window_stop=6,
            total_frames=16,
            rollout_anchors_per_window=0,
        )


def test_additive_physical_metrics_convert_to_selection_metrics() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    additive = {
        "physical_state_position_sse": 12.0,
        "physical_state_position_coordinate_count": 3.0,
        "physical_state_velocity_sse": 27.0,
        "physical_state_velocity_coordinate_count": 3.0,
        "physical_matched_object_frames": 9.0,
        "physical_target_object_frames": 10.0,
        "physical_identity_switches": 1.0,
        "physical_object_frame_associations": 100.0,
        "physical_distance_gated_matched_object_frames": 8.0,
        "physical_distance_gated_target_object_frames": 10.0,
        "physical_distance_gated_predicted_object_frames": 16.0,
        "physical_distance_gated_identity_switches": 1.0,
        "physical_distance_gated_object_frame_associations": 80.0,
        "physical_position_coverage90_hit_count": 85.0,
        "physical_position_coverage90_coordinate_count": 100.0,
        "physical_collision_true_positive_count": 3.0,
        "physical_collision_false_positive_count": 1.0,
        "physical_collision_false_negative_count": 2.0,
        "physical_rollout_position@0.100s_sse": 3.0,
        "physical_rollout_position@0.100s_coordinate_count": 3.0,
        "physical_forecast_active_count@0.100s": 8.0,
        "physical_forecast_target_count@0.100s": 10.0,
        "physical_rollout_position@0.250s_sse": 12.0,
        "physical_rollout_position@0.250s_coordinate_count": 3.0,
        "physical_forecast_active_count@0.250s": 7.0,
        "physical_forecast_target_count@0.250s": 10.0,
        "physical_rollout_position@0.500s_sse": 27.0,
        "physical_rollout_position@0.500s_coordinate_count": 3.0,
        "physical_forecast_active_count@0.500s": 6.0,
        "physical_forecast_target_count@0.500s": 10.0,
    }

    metrics = physical_validation_metrics(additive, config)

    assert metrics["validation_position_rmse_m"] == pytest.approx(2.0)
    assert metrics["validation_velocity_rmse_mps"] == pytest.approx(3.0)
    assert metrics["validation_target_coverage"] == pytest.approx(0.8)
    assert metrics["validation_prediction_precision"] == pytest.approx(0.5)
    assert metrics["validation_collision_f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["validation_id_switch_rate"] == pytest.approx(0.0125)
    assert metrics["validation_position_coverage90"] == pytest.approx(0.85)
    assert metrics["validation_position_rmse@0.100s"] == pytest.approx(1.0)
    assert metrics["validation_forecast_target_coverage@0.100s"] == pytest.approx(0.8)
    assert metrics["validation_position_rmse@0.250s"] == pytest.approx(2.0)
    assert metrics["validation_forecast_target_coverage@0.250s"] == pytest.approx(0.7)
    assert metrics["validation_position_rmse@0.500s"] == pytest.approx(3.0)
    assert metrics["validation_forecast_target_coverage@0.500s"] == pytest.approx(0.6)


def test_physical_selection_distance_gate_matches_evaluator_threshold() -> None:
    prediction = torch.tensor([[[0.50, 0.0, 0.0], [0.5001, 0.0, 0.0], [float("nan"), 0.0, 0.0]]])
    target = torch.zeros_like(prediction)
    assignment = torch.tensor([[True, True, True]])

    gated = _distance_gate_physical_matches(prediction, target, assignment)

    torch.testing.assert_close(gated, torch.tensor([[True, False, False]]))


def test_closed_loop_window_can_be_conditioned_on_collision() -> None:
    batch = {
        "rgb": torch.zeros((2, 10, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((2, 10, 3), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][1, 7, 2] = True
    random.seed(11)

    start = select_closed_loop_window(
        batch,
        4,
        event_condition_probability=1.0,
    )

    assert 0 <= start <= 6
    assert start <= 7 < start + 4


def test_closed_loop_window_can_require_a_maximum_horizon_anchor() -> None:
    batch = {
        "rgb": torch.zeros((2, 16, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((2, 16, 3), dtype=torch.bool),
        },
    }
    random.seed(19)

    starts = {
        select_closed_loop_window(
            batch,
            6,
            event_condition_probability=0.0,
            maximum_rollout_frame_offset=10,
            long_horizon_probability=1.0,
        )
        for _ in range(32)
    }

    assert starts
    assert starts <= set(range(6))


def test_collision_conditioning_takes_priority_over_long_horizon_window() -> None:
    batch = {
        "rgb": torch.zeros((1, 32, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 32, 2), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][0, 31, 0] = True
    random.seed(23)

    start = select_closed_loop_window(
        batch,
        8,
        event_condition_probability=1.0,
        maximum_rollout_frame_offset=20,
        long_horizon_probability=1.0,
    )

    assert start == 24
    assert start <= 31 < start + 8


def test_joint_sampling_preserves_long_horizon_when_collision_is_too_late() -> None:
    batch = {
        "rgb": torch.zeros((1, 32, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 32, 2), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][0, 31, 0] = True
    random.seed(23)

    start = select_closed_loop_window(
        batch,
        8,
        event_condition_probability=1.0,
        maximum_rollout_frame_offset=20,
        long_horizon_probability=1.0,
        joint_collision_long_horizon_sampling=True,
    )

    assert 0 <= start <= 11
    assert not (start <= 31 < start + 8)


def test_joint_sampling_covers_collision_and_long_horizon_when_compatible() -> None:
    batch = {
        "rgb": torch.zeros((1, 32, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 32, 2), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][0, 7, 0] = True
    random.seed(31)

    start = select_closed_loop_window(
        batch,
        8,
        event_condition_probability=1.0,
        maximum_rollout_frame_offset=20,
        long_horizon_probability=1.0,
        joint_collision_long_horizon_sampling=True,
    )

    assert 0 <= start <= 7
    assert start <= 7 < start + 8
    assert start <= 11


def test_joint_sampling_selects_a_compatible_collision_before_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {
        "rgb": torch.zeros((1, 32, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 32, 2), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][0, 7, 0] = True
    batch["events"]["collision"][0, 31, 0] = True
    monkeypatch.setattr(random, "choice", max)

    start = select_closed_loop_window(
        batch,
        8,
        event_condition_probability=1.0,
        maximum_rollout_frame_offset=20,
        long_horizon_probability=1.0,
        joint_collision_long_horizon_sampling=True,
    )

    assert start <= 7 < start + 8
    assert start <= 11


def test_closed_loop_window_always_keeps_a_future_rollout_anchor() -> None:
    batch = {
        "rgb": torch.zeros((1, 24, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 24, 2), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][0, 23, 0] = True
    random.seed(29)

    starts = {
        select_closed_loop_window(
            batch,
            2,
            event_condition_probability=1.0,
            maximum_rollout_frame_offset=20,
            minimum_rollout_frame_offset=2,
            long_horizon_probability=0.0,
        )
        for _ in range(32)
    }

    assert starts
    assert max(starts) <= 21


def _physical_selection_metrics(
    *,
    position: float = 0.4,
    velocity: float = 0.8,
    coverage: float = 0.9,
    precision: float = 0.9,
    collision_f1: float = 0.6,
    id_switch_rate: float = 0.01,
    position_coverage90: float = 0.9,
    forecast_coverage: tuple[float, float, float] = (0.9, 0.9, 0.9),
    horizons: tuple[float, float, float] = (0.4, 0.3, 0.2),
) -> dict[str, float]:
    return {
        "validation_position_rmse_m": position,
        "validation_velocity_rmse_mps": velocity,
        "validation_target_coverage": coverage,
        "validation_prediction_precision": precision,
        "validation_collision_f1": collision_f1,
        "validation_id_switch_rate": id_switch_rate,
        "validation_position_coverage90": position_coverage90,
        "validation_position_rmse@0.100s": horizons[0],
        "validation_position_rmse@0.250s": horizons[1],
        "validation_position_rmse@0.500s": horizons[2],
        "validation_forecast_target_coverage@0.100s": forecast_coverage[0],
        "validation_forecast_target_coverage@0.250s": forecast_coverage[1],
        "validation_forecast_target_coverage@0.500s": forecast_coverage[2],
    }


def _broad_checkpoint_metrics() -> dict[str, float]:
    config = load_config("configs/tiny_overfit.yaml")
    selection = _rollout_selection_metrics(_physical_selection_metrics(), config)
    return {
        "best_rollout_validated": 1.0,
        "rollout_selection_metric_version": 3.0,
        **selection.checkpoint_metrics(),
        **_validation_protocol_checkpoint_metrics(config),
    }


def test_rollout_selection_uses_horizon_weighted_physical_rmse() -> None:
    config = load_config("configs/tiny_overfit.yaml")

    selection = _rollout_selection_metrics(_physical_selection_metrics(), config)

    assert selection.score == pytest.approx((1.0 * 0.4 + 1.5 * 0.3 + 2.0 * 0.2) / 4.5)


@pytest.mark.parametrize(
    "candidate_metrics",
    [
        _physical_selection_metrics(
            velocity=0.817,
            horizons=(0.39, 0.29, 0.19),
        ),
        _physical_selection_metrics(
            coverage=0.894,
            horizons=(0.39, 0.29, 0.19),
        ),
        _physical_selection_metrics(
            collision_f1=0.587,
            horizons=(0.39, 0.29, 0.19),
        ),
        _physical_selection_metrics(
            id_switch_rate=0.016,
            horizons=(0.39, 0.29, 0.19),
        ),
        _physical_selection_metrics(horizons=(0.409, 0.2, 0.1)),
        _physical_selection_metrics(position=0.409, horizons=(0.3, 0.2, 0.1)),
    ],
)
def test_rollout_selection_rejects_broad_regressions(
    candidate_metrics: dict[str, float],
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    incumbent = _rollout_selection_metrics(_physical_selection_metrics(), config)
    candidate = _rollout_selection_metrics(candidate_metrics, config)

    assert candidate.score < incumbent.score
    assert not _rollout_selection_improves(candidate, incumbent)


def test_rollout_selection_accepts_score_gain_within_guardrails() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    incumbent = _rollout_selection_metrics(_physical_selection_metrics(), config)
    candidate = _rollout_selection_metrics(
        _physical_selection_metrics(
            position=0.404,
            velocity=0.81,
            coverage=0.896,
            collision_f1=0.59,
            id_switch_rate=0.014,
            horizons=(0.408, 0.24, 0.12),
        ),
        config,
    )

    assert _rollout_selection_improves(candidate, incumbent)


def test_rollout_selection_requires_complete_finite_physical_metrics() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    missing = _physical_selection_metrics()
    del missing["validation_collision_f1"]
    with pytest.raises(RuntimeError, match="validation_collision_f1"):
        _rollout_selection_metrics(missing, config)

    nonfinite = _physical_selection_metrics()
    nonfinite["validation_velocity_rmse_mps"] = float("nan")
    with pytest.raises(FloatingPointError, match="must all be finite"):
        _rollout_selection_metrics(nonfinite, config)


def test_legacy_rollout_score_is_not_reused_after_objective_fix() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    payload = {
        "config": config.to_dict(),
        "metrics": {
            "best_rollout_validated": 1.0,
            "best_rollout_position_loss": 0.01,
        },
    }

    assert not _rollout_selection_is_compatible(payload, config)
    payload["metrics"]["rollout_selection_metric_version"] = 2.0
    assert not _rollout_selection_is_compatible(payload, config)
    payload["metrics"].update(_broad_checkpoint_metrics())
    assert _rollout_selection_is_compatible(payload, config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("simulator", "scenario_mixture", ["baseline", "elastic_pairs"]),
        ("simulator", "sequence_frames", 17),
        ("simulator", "min_objects", 1),
        ("simulator", "max_objects", 3),
        ("training", "validation_episodes", 7),
        ("project", "seed", 99),
    ],
)
def test_rollout_score_is_not_reused_across_validation_protocols(
    section: str,
    field: str,
    value: object,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    checkpoint_config = config.to_dict()
    checkpoint_config[section][field] = value
    payload = {
        "config": checkpoint_config,
        "metrics": {
            **_broad_checkpoint_metrics(),
        },
    }

    assert not _rollout_selection_is_compatible(payload, config)


class _ModeOnlyModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> _ModeOnlyModel:
        self.training = False
        return self

    def train(self, mode: bool = True) -> _ModeOnlyModel:
        self.training = mode
        return self


def _result(value: float) -> TrainingBatchResult:
    scalar = torch.tensor(value)
    physical_metrics = {
        "physical_state_position_sse": value,
        "physical_state_position_coordinate_count": 1.0,
        "physical_state_velocity_sse": value,
        "physical_state_velocity_coordinate_count": 1.0,
        "physical_target_object_frames": 1.0,
        "physical_matched_object_frames": 1.0,
        "physical_identity_switches": 0.0,
        "physical_object_frame_associations": 1.0,
        "physical_distance_gated_target_object_frames": 1.0,
        "physical_distance_gated_predicted_object_frames": 1.0,
        "physical_distance_gated_matched_object_frames": 1.0,
        "physical_distance_gated_identity_switches": 0.0,
        "physical_distance_gated_object_frame_associations": 1.0,
        "physical_position_coverage90_hit_count": 1.0,
        "physical_position_coverage90_coordinate_count": 1.0,
        "physical_collision_true_positive_count": 1.0,
        "physical_collision_false_positive_count": 0.0,
        "physical_collision_false_negative_count": 0.0,
    }
    for suffix in ("0.100s", "0.250s", "0.500s"):
        physical_metrics[f"physical_rollout_position@{suffix}_sse"] = value
        physical_metrics[f"physical_rollout_position@{suffix}_coordinate_count"] = 1.0
        physical_metrics[f"physical_forecast_active_count@{suffix}"] = 1.0
        physical_metrics[f"physical_forecast_target_count@{suffix}"] = 1.0
    return TrainingBatchResult(
        total_loss=scalar,
        loss_terms={"rollout": scalar},
        metrics={"value": value, **physical_metrics},
        phase="closed_loop_rgb",
    )


def test_closed_loop_validation_uses_the_full_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int | bool] = {}

    def fake_closed_loop(
        model: object,
        batch: object,
        config: object,
        **kwargs: int | bool,
    ) -> TrainingBatchResult:
        del model, batch, config
        observed.update(kwargs)
        return _result(1.0)

    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        fake_closed_loop,
    )
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        training=replace(config.training, tbptt_steps=3),
    )
    model = _ModeOnlyModel()
    batch = {
        "rgb": torch.zeros((2, 9, 3, 8, 8)),
        "timestamps": torch.zeros((2, 9)),
    }

    _validation_step(model, batch, config, closed_loop=True)  # type: ignore[arg-type]

    assert observed["window_start"] == 0
    assert observed["window_steps"] == 9
    assert observed["apply_perturbations"] is False
    assert model.training


def test_validation_aggregates_every_loader_batch_by_episode_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []

    def fake_validation(
        model: object,
        batch: dict[str, torch.Tensor],
        config: object,
        *,
        closed_loop: bool,
    ) -> TrainingBatchResult:
        del model, config
        assert closed_loop
        seen.append(int(batch["rgb"].shape[0]))
        return _result(float(batch["rgb"][0, 0, 0, 0, 0]))

    monkeypatch.setattr(
        "world_model.training.trainer._validation_step",
        fake_validation,
    )
    loader = [
        {
            "rgb": torch.ones((2, 3, 3, 8, 8)),
            "timestamps": torch.zeros((2, 3)),
        },
        {
            "rgb": torch.full((1, 3, 3, 8, 8), 4.0),
            "timestamps": torch.zeros((1, 3)),
        },
    ]

    result = _validation_loader_result(
        _ModeOnlyModel(),  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        load_config("configs/tiny_overfit.yaml"),
        device=torch.device("cpu"),
        closed_loop=True,
    )

    assert seen == [2, 1]
    torch.testing.assert_close(result.total_loss, torch.tensor(2.0))
    torch.testing.assert_close(result.loss_terms["rollout"], torch.tensor(2.0))
    assert result.metrics["value"] == 2.0
    assert result.metrics["validation_position_rmse_m"] == pytest.approx(2.5**0.5)
    assert result.metrics["validation_velocity_rmse_mps"] == pytest.approx(2.5**0.5)
    assert result.metrics["validation_target_coverage"] == 1.0
    assert result.metrics["validation_collision_f1"] == 1.0
    assert result.metrics["validation_id_switch_rate"] == 0.0
    assert result.metrics["validation_position_rmse@0.500s"] == pytest.approx(2.5**0.5)
