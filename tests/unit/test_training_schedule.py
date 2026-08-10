from __future__ import annotations

import json
import math
import random
from dataclasses import replace

import pytest
import torch
from torch import nn

from world_model.runtime import OnlineWorldModel
from world_model.training.loop import (
    TrainingBatchResult,
    _combine_measurement_objectives,
    _distance_gate_physical_matches,
    _fast_measurement_has_trainable_perception_path,
    _global_measurement_has_trainable_path,
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
    _ROLLOUT_SELECTION_METRIC_VERSION,
    _causal_training_support,
    _clip_training_gradients,
    _closed_loop_trainable_scope_for_step,
    _finite_nonnegative_integer,
    _gradient_clip_diagnostics,
    _handoff_training_support_failures,
    _has_effective_gradient,
    _make_loader,
    _mean_batch_results,
    _mutable_causal_training_support_failures,
    _rollout_selection_improves,
    _rollout_selection_is_compatible,
    _rollout_selection_metrics,
    _rollout_selection_passes_guardrails,
    _selection_scenario_slugs,
    _validation_loader_result,
    _validation_protocol_checkpoint_metrics,
    _validation_step,
    measurement_pretrain_frame_index,
    set_closed_loop_trainable_scope,
    set_global_perception_trainable,
)
from world_model.utils.config import load_config


def test_worker_loader_bounds_prefetch_and_does_not_persist_workers() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        training=replace(source.training, num_workers=1),
    )

    loader = _make_loader(config, split="train", episodes=1, shuffle=False)

    assert loader.num_workers == 1
    assert loader.prefetch_factor == 1
    assert loader.persistent_workers is False


def test_training_loader_batches_every_declared_scenario_when_balanced() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    scenarios = ("reference_pairs", "baseline", "elastic_pairs", "damped_contacts")
    config = replace(
        source,
        simulator=replace(source.simulator, scenario_mixture=scenarios),
        training=replace(
            source.training,
            batch_size=4,
            train_episodes=16,
            validation_episodes=4,
            scenario_balanced_batches=True,
            num_workers=0,
        ),
    )
    config.validate()

    loader = _make_loader(
        config,
        split="train",
        episodes=16,
        shuffle=True,
        start_step=0,
        stop_step=5,
    )
    batches = list(loader)

    assert len(batches) == 5
    assert all(
        tuple(sorted(batch["metadata"]["scenario"])) == tuple(sorted(scenarios))
        for batch in batches
    )


def test_fixed_pretraining_sweeps_every_adjacent_pair_for_every_loader_batch() -> None:
    loader_batches = 4
    total_frames = 16
    visited = {batch_index: [] for batch_index in range(loader_batches)}

    for step in range(loader_batches * (total_frames - 1)):
        batch_index = step % loader_batches
        visited[batch_index].append(
            measurement_pretrain_frame_index(
                step,
                loader_batches=loader_batches,
                total_frames=total_frames,
                fixed_dataset=True,
            )
        )

    expected = list(range(total_frames - 1))
    assert all(frame_indices == expected for frame_indices in visited.values())


def test_pretraining_frame_index_requires_a_batch_and_rgb_pair() -> None:
    for loader_batches, total_frames in ((0, 16), (4, 0), (4, 1)):
        with pytest.raises(ValueError, match="RGB pair"):
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
    assert sampled <= set(range(6))


def test_global_perception_freeze_leaves_shared_fast_encoder_trainable() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_global_perception_trainable(model, trainable=False)

    assert all(
        parameter.requires_grad
        for stage in rgb.backbone.stages[:2]
        for parameter in stage.parameters()
    )
    assert all(parameter.requires_grad for parameter in rgb.backbone.fast_projection.parameters())
    assert not any(
        parameter.requires_grad
        for stage in rgb.backbone.stages[2:]
        for parameter in stage.parameters()
    )
    assert not any(
        parameter.requires_grad
        for projection in rgb.backbone.projections
        for parameter in projection.parameters()
    )
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
    for name, parameter in model.named_parameters():
        disconnected = (
            ".roi_updater.event_head." in name
            or "updater.learned_corrector.visibility_head." in name
            or "identifier.variance_head." in name
        )
        assert parameter.requires_grad is (not disconnected), name


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


def test_interaction_gradients_are_bounded_before_global_clipping() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=10.0,
            interaction_grad_clip_norm=1.0,
            closed_loop_perception_grad_clip_norm=10.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    interaction_parameters = list(model.dynamics.interactions.parameters())
    interaction_ids = {id(parameter) for parameter in interaction_parameters}
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in interaction_ids
    ]
    interaction = interaction_parameters[0]
    other = other_parameters[0]
    interaction.grad = torch.ones_like(interaction)
    interaction.grad.mul_(3.0 / interaction.grad.norm())
    other.grad = torch.ones_like(other)
    other.grad.mul_(4.0 / other.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(3.0)
    assert diagnostics["interaction_gradient_norm_applied_before_global_clip"] == pytest.approx(
        1.0, abs=1.0e-6
    )
    assert diagnostics["gradient_norm_pre_clip"] == pytest.approx(5.0, abs=1.0e-5)
    assert diagnostics["gradient_norm_pre_global_clip"] == pytest.approx(
        math.sqrt(17.0),
        abs=1.0e-5,
    )
    assert diagnostics["gradient_norm_applied"] == pytest.approx(
        math.sqrt(17.0),
        abs=1.0e-5,
    )
    assert interaction.grad.norm().item() == pytest.approx(1.0, abs=1.0e-6)
    assert other.grad.norm().item() == pytest.approx(4.0, abs=1.0e-6)


def test_perception_gradients_are_bounded_without_scaling_small_dynamics() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=10.0,
            interaction_grad_clip_norm=1.0,
            closed_loop_perception_grad_clip_norm=1.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    perception_parameters = list(model.observation_modules["rgb"].parameters())
    perception_ids = {id(parameter) for parameter in perception_parameters}
    interaction_parameters = list(model.dynamics.interactions.parameters())
    interaction_ids = {id(parameter) for parameter in interaction_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in perception_ids | interaction_ids
    ]
    perception = perception_parameters[0]
    interaction = interaction_parameters[0]
    other = other_parameters[0]
    perception.grad = torch.ones_like(perception)
    perception.grad.mul_(12.0 / perception.grad.norm())
    interaction.grad = torch.ones_like(interaction)
    interaction.grad.mul_(0.3 / interaction.grad.norm())
    other.grad = torch.ones_like(other)
    other.grad.mul_(0.4 / other.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["perception_gradient_norm_pre_clip"] == pytest.approx(12.0)
    assert diagnostics["perception_gradient_norm_applied_before_global_clip"] == pytest.approx(
        1.0, abs=1.0e-6
    )
    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(0.3)
    assert diagnostics["interaction_gradient_norm_applied_before_global_clip"] == pytest.approx(0.3)
    assert diagnostics["gradient_norm_pre_clip"] == pytest.approx(
        math.sqrt(12.0**2 + 0.3**2 + 0.4**2),
        abs=1.0e-5,
    )
    assert diagnostics["gradient_norm_pre_global_clip"] == pytest.approx(
        math.sqrt(1.0**2 + 0.3**2 + 0.4**2),
        abs=1.0e-5,
    )
    assert diagnostics["gradient_norm_applied"] == pytest.approx(
        math.sqrt(1.0**2 + 0.3**2 + 0.4**2),
        abs=1.0e-5,
    )
    assert perception.grad.norm().item() == pytest.approx(1.0, abs=1.0e-6)
    assert interaction.grad.norm().item() == pytest.approx(0.3, abs=1.0e-6)
    assert other.grad.norm().item() == pytest.approx(0.4, abs=1.0e-6)


def test_measurement_pretraining_disables_closed_loop_perception_local_clip() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=20.0,
            interaction_grad_clip_norm=20.0,
            closed_loop_perception_grad_clip_norm=1.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    perception = next(model.observation_modules["rgb"].parameters())
    perception.grad = torch.ones_like(perception)
    perception.grad.mul_(12.0 / perception.grad.norm())

    diagnostics = _clip_training_gradients(
        model,
        config,
        apply_perception_local_clip=False,
    )

    assert diagnostics["perception_gradient_local_clip_enabled"] == 0.0
    assert diagnostics["perception_gradient_norm_pre_clip"] == pytest.approx(12.0)
    assert diagnostics["perception_gradient_clip_coefficient"] == 1.0
    assert diagnostics["perception_gradient_norm_applied_before_global_clip"] == pytest.approx(12.0)
    assert perception.grad.norm().item() == pytest.approx(12.0, abs=1.0e-5)


def test_effective_gradient_threshold_is_strict() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    threshold = config.training.minimum_effective_gradient_norm

    assert not _has_effective_gradient(0.0, config)
    assert not _has_effective_gradient(threshold, config)
    assert _has_effective_gradient(threshold * 2.0 + 1.0e-15, config)


def test_causal_support_excludes_global_only_measurement_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    global_loss = parameter.square()
    unsupported = TrainingBatchResult(
        total_loss=global_loss,
        loss_terms={"measurement": global_loss},
        metrics={
            "matched_object_frames": 0.0,
            "fast_supervised_frames": 0.0,
            "fast_supervised_slots": 0.0,
        },
        phase="closed_loop_rgb",
    )
    supported_trajectory = TrainingBatchResult(
        total_loss=global_loss,
        loss_terms={"measurement": global_loss, "state": global_loss},
        metrics={
            "matched_object_frames": 2.0,
            "fast_supervised_frames": 0.0,
        },
        phase="closed_loop_rgb",
    )
    supported_fast = TrainingBatchResult(
        total_loss=global_loss,
        loss_terms={"measurement": global_loss},
        metrics={
            "matched_object_frames": 0.0,
            "fast_supervised_frames": 1.0,
            "fast_supervised_slots": 1.0,
        },
        phase="closed_loop_rgb",
        support_terms={"fast_measurement": global_loss},
    )

    assert _causal_training_support(unsupported) == (False, 0.0, 0.0, 0.0)
    assert _causal_training_support(supported_trajectory) == (True, 2.0, 0.0, 1.0)
    assert _causal_training_support(supported_fast) == (True, 0.0, 1.0, 0.0)


def test_zero_weight_trajectory_term_cannot_make_global_auxiliary_causal() -> None:
    global_loss = torch.tensor(1.0, requires_grad=True)
    trajectory_loss = torch.tensor(2.0, requires_grad=True)
    result = TrainingBatchResult(
        total_loss=global_loss + 0.0 * trajectory_loss,
        loss_terms={
            "measurement": global_loss,
            "state_position": trajectory_loss,
        },
        metrics={
            "matched_object_frames": 2.0,
            "fast_supervised_slots": 0.0,
        },
        phase="closed_loop_rgb",
    )

    assert _causal_training_support(result) == (False, 2.0, 0.0, 0.0)


def test_differentiable_trajectory_term_without_physical_support_is_not_causal() -> None:
    trajectory_loss = torch.tensor(2.0, requires_grad=True)
    result = TrainingBatchResult(
        total_loss=trajectory_loss,
        loss_terms={"state_position": trajectory_loss},
        metrics={
            "matched_object_frames": 0.0,
            "fast_supervised_slots": 0.0,
        },
        phase="closed_loop_rgb",
    )

    assert _causal_training_support(result) == (False, 0.0, 0.0, 1.0)


def test_active_false_track_existence_negative_is_explicit_causal_support() -> None:
    existence_loss = torch.tensor(0.5, requires_grad=True)
    result = TrainingBatchResult(
        total_loss=existence_loss,
        loss_terms={"existence": existence_loss},
        metrics={
            "matched_object_frames": 0.0,
            "existence_negative_supervision_object_frames": 3.0,
            "fast_supervised_slots": 0.0,
        },
        phase="closed_loop_rgb",
    )

    assert _causal_training_support(result) == (True, 3.0, 0.0, 1.0)


def test_global_measurement_cannot_impersonate_supported_fast_roi_gradient() -> None:
    global_loss = torch.tensor(1.0, requires_grad=True)
    fast_loss = torch.tensor(2.0, requires_grad=True)
    result = TrainingBatchResult(
        total_loss=global_loss + 0.0 * fast_loss,
        loss_terms={"measurement": global_loss},
        metrics={
            "matched_object_frames": 0.0,
            "fast_supervised_slots": 2.0,
        },
        phase="closed_loop_rgb",
        support_terms={"fast_measurement": fast_loss},
    )

    assert _causal_training_support(result) == (False, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -1, 1.5, "2", True])
def test_resume_counter_parser_rejects_nonfinite_or_noninteger_values(value) -> None:
    with pytest.raises(ValueError, match="finite nonnegative integer"):
        _finite_nonnegative_integer(value, name="counter")


def test_state_dynamics_scope_freezes_rgb_and_trains_filter_dynamics_identifier() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    set_closed_loop_trainable_scope(model, scope="state_dynamics")

    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.updater.named_parameters()
        if not name.startswith("learned_corrector.visibility_head.")
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.updater.learned_corrector.visibility_head.parameters()
    )
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
    assert not _fast_measurement_has_trainable_perception_path(model.observation_modules["rgb"])


def test_updater_scope_isolates_correction_recovery() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    set_closed_loop_trainable_scope(model, scope="updater")

    assert any(parameter.requires_grad for parameter in model.updater.parameters())
    assert not any(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert model.identifier is not None
    assert not any(parameter.requires_grad for parameter in model.identifier.parameters())
    assert not any(parameter.requires_grad for parameter in model.observation_modules.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in model.updater.learned_corrector.visibility_head.parameters()
    )


def test_state_dynamics_roi_scope_trains_fast_rgb_without_global_perception() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_closed_loop_trainable_scope(model, scope="state_dynamics_roi")

    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.updater.named_parameters()
        if not name.startswith("learned_corrector.visibility_head.")
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.updater.learned_corrector.visibility_head.parameters()
    )
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
    assert all(
        parameter.requires_grad
        for stage in rgb.backbone.stages[:2]
        for parameter in stage.parameters()
    )
    assert all(parameter.requires_grad for parameter in rgb.backbone.fast_projection.parameters())
    assert not any(
        parameter.requires_grad
        for stage in rgb.backbone.stages[2:]
        for parameter in stage.parameters()
    )
    assert not any(
        parameter.requires_grad
        for projection in rgb.backbone.projections
        for parameter in projection.parameters()
    )
    assert not any(parameter.requires_grad for parameter in rgb.global_detector.parameters())


def test_state_dynamics_fast_roi_scope_keeps_shared_backbone_frozen() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_closed_loop_trainable_scope(model, scope="state_dynamics_fast_roi")

    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert any(parameter.requires_grad for parameter in model.updater.parameters())
    assert model.identifier is not None
    assert any(parameter.requires_grad for parameter in model.identifier.parameters())
    assert any(parameter.requires_grad for parameter in rgb.roi_updater.parameters())
    assert all(parameter.requires_grad for parameter in rgb.backbone.fast_projection.parameters())
    assert not any(
        parameter.requires_grad for stage in rgb.backbone.stages for parameter in stage.parameters()
    )
    assert not any(
        parameter.requires_grad
        for projection in rgb.backbone.projections
        for parameter in projection.parameters()
    )
    assert not any(parameter.requires_grad for parameter in rgb.global_detector.parameters())
    assert not _global_measurement_has_trainable_path(rgb)
    assert _fast_measurement_has_trainable_perception_path(rgb)


def test_fast_roi_scope_freezes_state_and_global_modules() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_closed_loop_trainable_scope(model, scope="fast_roi")

    assert not any(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert not any(parameter.requires_grad for parameter in model.updater.parameters())
    assert model.identifier is not None
    assert not any(parameter.requires_grad for parameter in model.identifier.parameters())
    assert any(parameter.requires_grad for parameter in rgb.roi_updater.parameters())
    assert all(parameter.requires_grad for parameter in rgb.backbone.fast_projection.parameters())
    assert not any(
        parameter.requires_grad for stage in rgb.backbone.stages for parameter in stage.parameters()
    )
    assert not any(parameter.requires_grad for parameter in rgb.global_detector.parameters())
    assert _fast_measurement_has_trainable_perception_path(rgb)


def test_closed_loop_scope_transition_counts_causal_updates() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        training=replace(
            source.training,
            rgb_pretrain_steps=100,
            closed_loop_trainable_scope="fast_roi",
            closed_loop_late_trainable_scope="state_dynamics",
            closed_loop_scope_transition_steps=512,
        ),
    )

    assert _closed_loop_trainable_scope_for_step(config, completed_step=611) == (
        "fast_roi",
        False,
    )
    assert _closed_loop_trainable_scope_for_step(config, completed_step=612) == (
        "state_dynamics",
        True,
    )


def test_global_measurement_trainability_ignores_roi_only_projection() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_closed_loop_trainable_scope(model, scope="state_dynamics_roi")
    assert _global_measurement_has_trainable_path(rgb)

    set_closed_loop_trainable_scope(model, scope="state_dynamics_fast_roi")
    assert all(parameter.requires_grad for parameter in rgb.backbone.fast_projection.parameters())
    assert not _global_measurement_has_trainable_path(rgb)


def test_measurement_branch_coefficients_do_not_change_with_support() -> None:
    global_loss = torch.tensor(6.0)
    fast_loss = torch.tensor(2.0)

    torch.testing.assert_close(
        _combine_measurement_objectives(
            global_measurement=global_loss,
            fast_measurement=fast_loss,
            fast_weight=1.0,
        ),
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        _combine_measurement_objectives(
            global_measurement=global_loss,
            fast_measurement=None,
            fast_weight=1.0,
        ),
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        _combine_measurement_objectives(
            global_measurement=None,
            fast_measurement=fast_loss,
            fast_weight=1.0,
        ),
        torch.tensor(1.0),
    )


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
    for axis in ("x", "y", "z"):
        additive[f"physical_state_position_{axis}_sse"] = 4.0
        additive[f"physical_state_position_{axis}_coordinate_count"] = 1.0
    for suffix, horizon_sse in (
        ("0.100s", 3.0),
        ("0.250s", 12.0),
        ("0.500s", 27.0),
    ):
        for axis in ("x", "y", "z"):
            additive[f"physical_rollout_position_{axis}@{suffix}_sse"] = horizon_sse / 3.0
            additive[f"physical_rollout_position_{axis}@{suffix}_coordinate_count"] = 1.0
        additive[f"physical_rollout_position_coverage90@{suffix}_hit_count"] = 3.0
        additive[f"physical_rollout_position_coverage90@{suffix}_coordinate_count"] = 3.0
        additive[f"physical_forecast_tracked_count@{suffix}"] = 10.0
        additive[f"physical_forecast_predictable_target_count@{suffix}"] = 10.0
        additive[f"physical_rollout_predictable_target_count@{suffix}"] = 1.0
        additive[f"physical_rollout_censored_external_actuation_count@{suffix}"] = 0.0

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


def test_collision_conditioning_aligns_event_to_shortest_rollout_endpoint() -> None:
    batch = {
        "rgb": torch.zeros((1, 16, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 16, 2), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][0, 7, 0] = True

    start = select_closed_loop_window(
        batch,
        4,
        event_condition_probability=1.0,
        minimum_rollout_frame_offset=2,
        maximum_rollout_frame_offset=10,
    )

    assert start == 5
    assert start + 2 == 7


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


def test_collision_conditioning_prioritizes_scarce_pair_interactions() -> None:
    batch = {
        "rgb": torch.zeros((1, 32, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 32, 2), dtype=torch.bool),
            "pair_collision": torch.zeros((1, 32, 2, 2), dtype=torch.bool),
        },
    }
    # A common boundary event occurs early, while the rare object interaction
    # occurs later. A fully conditioned draw must train on the latter.
    batch["events"]["collision"][0, 5, 0] = True
    batch["events"]["collision"][0, 20, 0] = True
    batch["events"]["pair_collision"][0, 20, 0, 1] = True
    random.seed(37)

    start = select_closed_loop_window(
        batch,
        4,
        event_condition_probability=1.0,
    )

    assert start <= 20 < start + 4
    assert not (start <= 5 < start + 4)


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
    metrics = {
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
    for axis in ("x", "y", "z"):
        metrics[f"validation_position_rmse_{axis}_m"] = position
        metrics[f"validation_position_rmse_{axis}@0.100s"] = horizons[0]
        metrics[f"validation_position_rmse_{axis}@0.250s"] = horizons[1]
        metrics[f"validation_position_rmse_{axis}@0.500s"] = horizons[2]
    return metrics


def _with_scenario_selection_metrics(
    metrics: dict[str, float],
    config,
    *,
    scenario_metrics: dict[str, dict[str, float]] | None = None,
    unsupported: set[str] | None = None,
) -> dict[str, float]:
    """Attach the exact per-scenario fields emitted by broad validation."""

    output = dict(metrics)
    base = {name: value for name, value in metrics.items() if name.startswith("validation_")}
    scenario_metrics = scenario_metrics or {}
    unsupported = unsupported or set()
    for scenario in _selection_scenario_slugs(config):
        prefix = f"scenario_{scenario}_"
        output[f"{prefix}episode_count"] = 1.0
        output[f"{prefix}selection_metric_supported"] = float(scenario not in unsupported)
        if scenario in unsupported:
            continue
        selected = scenario_metrics.get(scenario, base)
        output.update({f"{prefix}{name}": value for name, value in selected.items()})
    return output


def _broad_checkpoint_metrics() -> dict[str, float]:
    config = load_config("configs/tiny_overfit.yaml")
    selection = _rollout_selection_metrics(
        _with_scenario_selection_metrics(_physical_selection_metrics(), config),
        config,
        require_scenarios=True,
    )
    return {
        "best_rollout_validated": 1.0,
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
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


def test_rollout_selection_rejects_aggregate_gain_with_unsupported_scenario() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        simulator=replace(
            source.simulator,
            scenario_mixture=("baseline", "elastic_pairs"),
        ),
    )
    incumbent = _rollout_selection_metrics(
        _with_scenario_selection_metrics(_physical_selection_metrics(), config),
        config,
        require_scenarios=True,
    )
    candidate = _rollout_selection_metrics(
        _with_scenario_selection_metrics(
            _physical_selection_metrics(horizons=(0.30, 0.20, 0.10)),
            config,
            unsupported={"elastic_pairs"},
        ),
        config,
        require_scenarios=True,
    )

    assert candidate.score < incumbent.score
    assert not _rollout_selection_improves(candidate, incumbent)
    assert not _rollout_selection_passes_guardrails(candidate, incumbent)
    failures = _handoff_training_support_failures(candidate, incumbent, config)
    assert "scenario_elastic_pairs_selection_support" in {
        str(failure["metric"]) for failure in failures
    }
    assert not _mutable_causal_training_support_failures(candidate, config)


def test_rollout_selection_rejects_aggregate_gain_with_scenario_regression() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        simulator=replace(
            source.simulator,
            scenario_mixture=("baseline", "elastic_pairs"),
        ),
    )
    incumbent_metrics = _physical_selection_metrics()
    candidate_metrics = _physical_selection_metrics(horizons=(0.30, 0.20, 0.10))
    regressed_scenario = _physical_selection_metrics(
        coverage=0.70,
        horizons=(0.50, 0.40, 0.30),
        forecast_coverage=(0.70, 0.70, 0.70),
    )
    incumbent = _rollout_selection_metrics(
        _with_scenario_selection_metrics(incumbent_metrics, config),
        config,
        require_scenarios=True,
    )
    candidate = _rollout_selection_metrics(
        _with_scenario_selection_metrics(
            candidate_metrics,
            config,
            scenario_metrics={"elastic_pairs": regressed_scenario},
        ),
        config,
        require_scenarios=True,
    )

    assert candidate.score < incumbent.score
    assert not _rollout_selection_improves(candidate, incumbent)


def test_handoff_rejects_conditionally_accurate_candidate_with_collapsed_coverage() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    reference = _rollout_selection_metrics(
        _physical_selection_metrics(
            coverage=0.80,
            forecast_coverage=(0.80, 0.78, 0.75),
        ),
        config,
    )
    candidate = _rollout_selection_metrics(
        _physical_selection_metrics(
            position=0.10,
            coverage=0.01,
            horizons=(0.10, 0.09, 0.08),
            forecast_coverage=(0.01, 0.01, 0.01),
        ),
        config,
    )

    failures = _handoff_training_support_failures(candidate, reference, config)

    assert candidate.score < reference.score
    assert {failure["metric"] for failure in failures} == {
        "target_coverage",
        "forecast_target_coverage@0.100s",
        "forecast_target_coverage@0.250s",
        "forecast_target_coverage@0.500s",
    }
    assert {
        failure["metric"]
        for failure in _mutable_causal_training_support_failures(candidate, config)
    } == {
        "target_coverage",
        "forecast_target_coverage@0.100s",
        "forecast_target_coverage@0.250s",
        "forecast_target_coverage@0.500s",
    }


def test_handoff_accepts_coverage_that_preserves_training_support() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    reference = _rollout_selection_metrics(_physical_selection_metrics(), config)
    candidate = _rollout_selection_metrics(
        _physical_selection_metrics(
            coverage=0.60,
            forecast_coverage=(0.55, 0.50, 0.48),
            horizons=(0.30, 0.20, 0.10),
        ),
        config,
    )

    assert not _handoff_training_support_failures(candidate, reference, config)


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


def test_rollout_checkpoint_without_declared_scenario_support_is_not_reused() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    metrics = _broad_checkpoint_metrics()
    scenario = _selection_scenario_slugs(config)[0]
    del metrics[f"best_rollout_scenario_{scenario}_selection_supported"]
    payload = {
        "config": config.to_dict(),
        "metrics": metrics,
    }

    assert not _rollout_selection_is_compatible(payload, config)


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
        "physical_state_position_sse": 3.0 * value,
        "physical_state_position_coordinate_count": 3.0,
        "physical_state_position_x_sse": value,
        "physical_state_position_x_coordinate_count": 1.0,
        "physical_state_position_y_sse": value,
        "physical_state_position_y_coordinate_count": 1.0,
        "physical_state_position_z_sse": value,
        "physical_state_position_z_coordinate_count": 1.0,
        "physical_state_velocity_sse": 3.0 * value,
        "physical_state_velocity_coordinate_count": 3.0,
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
        physical_metrics[f"physical_rollout_position@{suffix}_sse"] = 3.0 * value
        physical_metrics[f"physical_rollout_position@{suffix}_coordinate_count"] = 3.0
        for axis in ("x", "y", "z"):
            physical_metrics[f"physical_rollout_position_{axis}@{suffix}_sse"] = value
            physical_metrics[f"physical_rollout_position_{axis}@{suffix}_coordinate_count"] = 1.0
        physical_metrics[f"physical_rollout_position_coverage90@{suffix}_hit_count"] = 3.0
        physical_metrics[f"physical_rollout_position_coverage90@{suffix}_coordinate_count"] = 3.0
        physical_metrics[f"physical_forecast_active_count@{suffix}"] = 1.0
        physical_metrics[f"physical_forecast_tracked_count@{suffix}"] = 1.0
        physical_metrics[f"physical_forecast_target_count@{suffix}"] = 1.0
        physical_metrics[f"physical_forecast_predictable_target_count@{suffix}"] = 1.0
        physical_metrics[f"physical_rollout_predictable_target_count@{suffix}"] = 1.0
        physical_metrics[f"physical_rollout_censored_external_actuation_count@{suffix}"] = 0.0
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
        training=replace(
            config.training,
            tbptt_steps=3,
            validation_rollout_anchors_per_episode=5,
        ),
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
    assert observed["rollout_anchors_per_window"] == 5
    assert model.training


def test_validation_aggregates_every_loader_batch_by_episode_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
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

    progress_path = tmp_path / "training_progress.json"
    result = _validation_loader_result(
        _ModeOnlyModel(),  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        load_config("configs/tiny_overfit.yaml"),
        device=torch.device("cpu"),
        closed_loop=True,
        progress_path=progress_path,
        progress_split="validation_test",
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
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["state"] == "validation_complete"
    assert progress["split"] == "validation_test"
    assert progress["validation_kind"] == "closed_loop"
    assert progress["protocol_kind"] == "rollout"
    assert len(progress["protocol_hash"]) == 64
    assert progress["completed_batches"] == 2
    assert progress["total_batches"] == 2
    assert progress["completed_episodes"] == 3
    assert progress["total_episodes"] is None
    output = capsys.readouterr().out
    assert "batches=1/2 episodes=2/?" in output
    assert "batches=2/2 episodes=3/?" in output


def test_validation_progress_records_interrupted_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = 0

    def fake_validation(
        model: object,
        batch: dict[str, torch.Tensor],
        config: object,
        *,
        closed_loop: bool,
    ) -> TrainingBatchResult:
        nonlocal calls
        del model, batch, config, closed_loop
        calls += 1
        if calls == 2:
            raise RuntimeError("validation probe failure")
        return _result(1.0)

    monkeypatch.setattr(
        "world_model.training.trainer._validation_step",
        fake_validation,
    )
    loader = [
        {
            "rgb": torch.ones((1, 3, 3, 8, 8)),
            "timestamps": torch.zeros((1, 3)),
        },
        {
            "rgb": torch.ones((1, 3, 3, 8, 8)),
            "timestamps": torch.zeros((1, 3)),
        },
    ]
    progress_path = tmp_path / "training_progress.json"

    with pytest.raises(RuntimeError, match="validation probe failure"):
        _validation_loader_result(
            _ModeOnlyModel(),  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            load_config("configs/tiny_overfit.yaml"),
            device=torch.device("cpu"),
            closed_loop=False,
            progress_path=progress_path,
            progress_split="validation_measurement_probe",
        )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["state"] == "validation_interrupted"
    assert progress["split"] == "validation_measurement_probe"
    assert progress["validation_kind"] == "measurement"
    assert progress["protocol_kind"] == "measurement"
    assert progress["completed_batches"] == 1
    assert progress["completed_episodes"] == 1
    assert progress["exception_type"] == "RuntimeError"


def test_validation_averages_conditional_diagnostics_over_present_support() -> None:
    first = _result(1.0)
    second = _result(4.0)
    second.loss_terms = {}
    second.metrics.pop("value")
    second.metrics["second_only"] = 6.0
    first.metrics["unsupported"] = float("nan")

    result = _mean_batch_results([first, second], weights=[2.0, 1.0])

    torch.testing.assert_close(result.total_loss, torch.tensor(2.0))
    # A diagnostic omitted because its support is absent is not a zero sample
    # and therefore does not dilute the supported episode.
    torch.testing.assert_close(result.loss_terms["rollout"], torch.tensor(1.0))
    assert result.metrics["value"] == 1.0
    assert result.metrics["second_only"] == 6.0
    assert math.isnan(result.metrics["unsupported"])
