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
    _attention_node_complexity_details,
    _closed_loop_loss_weights_for_scope,
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
    _aggregate_physical_validation_metrics,
    _assert_interaction_gradient_retention,
    _attention_gradient_diagnostics,
    _backward_training_result,
    _causal_training_support,
    _clip_training_gradients,
    _closed_loop_trainable_scope_for_step,
    _core_causal_trajectory_episode_supported,
    _finite_nonnegative_integer,
    _gradient_clip_diagnostics,
    _handoff_training_support_failures,
    _has_effective_gradient,
    _make_loader,
    _mean_batch_results,
    _mutable_causal_training_support_failures,
    _prepare_restricted_attention_collision_update,
    _prepare_restricted_attention_node_update,
    _prepare_restricted_updater_mean_update,
    _restore_restricted_updater_mean_update,
    _rollout_selection_guardrail_failures,
    _rollout_selection_improves,
    _rollout_selection_is_compatible,
    _rollout_selection_metrics,
    _rollout_selection_passes_guardrails,
    _selection_scenario_slugs,
    _validate_validation_support_schema,
    _validation_loader_result,
    _validation_protocol_checkpoint_metrics,
    _validation_step,
    _write_training_update_progress,
    closed_loop_learning_rate_at_update,
    measurement_pretrain_frame_index,
    set_closed_loop_trainable_scope,
    set_global_perception_trainable,
)
from world_model.utils.config import load_config


def test_constant_closed_loop_learning_rate_is_exactly_backward_compatible() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    expected = config.training.learning_rate * config.training.closed_loop_learning_rate_scale

    assert closed_loop_learning_rate_at_update(config, causal_update_index=0) == expected
    assert closed_loop_learning_rate_at_update(config, causal_update_index=100_000) == expected


def test_warmup_cosine_schedule_uses_absolute_causal_update_index() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        training=replace(
            source.training,
            learning_rate=1.0,
            closed_loop_learning_rate_scale=0.5,
            closed_loop_learning_rate_schedule="warmup_cosine",
            closed_loop_learning_rate_warmup_steps=2,
            closed_loop_learning_rate_cosine_decay_steps=4,
            closed_loop_learning_rate_minimum_scale=0.2,
        ),
    )
    config.validate()

    observed = [
        closed_loop_learning_rate_at_update(config, causal_update_index=index) for index in range(8)
    ]
    assert observed[0] == pytest.approx(0.25)
    assert observed[1] == pytest.approx(0.5)
    assert observed[2] == pytest.approx(0.5 * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi / 4))))
    assert observed[5] == pytest.approx(0.1)
    assert observed[6:] == pytest.approx([0.1, 0.1])

    extended = replace(config, training=replace(config.training, steps=100_000))
    assert [
        closed_loop_learning_rate_at_update(extended, causal_update_index=index)
        for index in range(8)
    ] == pytest.approx(observed)


def test_closed_loop_learning_rate_rejects_negative_update_index() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    with pytest.raises(ValueError, match="causal_update_index"):
        closed_loop_learning_rate_at_update(config, causal_update_index=-1)


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


def test_attention_scope_isolates_new_typed_residual() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config)

    set_closed_loop_trainable_scope(model, scope="attention")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("dynamics.attention_interactions.") for name in trainable)
    assert model.dynamics.attention_interactions is not None
    assert len(trainable) == len(list(model.dynamics.attention_interactions.parameters()))


def test_attention_relation_scope_freezes_only_node_decoder() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config)

    set_closed_loop_trainable_scope(model, scope="attention_relation")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    attention_prefix = "dynamics.attention_interactions."
    node_prefix = f"{attention_prefix}node_decoder."
    assert trainable
    assert all(name.startswith(attention_prefix) for name in trainable)
    assert not any(name.startswith(node_prefix) for name in trainable)
    expected = {
        f"{attention_prefix}{name}"
        for name, _ in model.dynamics.attention_interactions.named_parameters()
        if not name.startswith("node_decoder.")
    }
    assert trainable == expected
    assert all(
        parameter.requires_grad is (not name.startswith("node_decoder."))
        for name, parameter in model.dynamics.attention_interactions.named_parameters()
    )
    attention = model.dynamics.attention_interactions
    attention.configure_output_gradient_clipping(
        node=0.1,
        collision=0.1,
        force=0.1,
        impulse=0.1,
    )
    diagnostics = attention.output_gradient_diagnostics()
    assert diagnostics["attention_node_output_backprop_gradient_local_clip_enabled"] == 0.0
    for name in ("collision", "force", "impulse"):
        assert diagnostics[f"attention_{name}_output_backprop_gradient_local_clip_enabled"] == 1.0


def test_attention_node_z_scope_preserves_excluded_rows_through_adamw() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope="attention_node_z")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "dynamics.attention_interactions.node_decoder.weight",
        "dynamics.attention_interactions.node_decoder.bias",
    }
    with torch.no_grad():
        attention.node_decoder.weight.copy_(
            torch.arange(
                attention.node_decoder.weight.numel(),
                dtype=attention.node_decoder.weight.dtype,
            ).reshape_as(attention.node_decoder.weight)
            / 1000
        )
        attention.node_decoder.bias.copy_(
            torch.arange(
                attention.node_decoder.bias.numel(),
                dtype=attention.node_decoder.bias.dtype,
            )
            / 1000
        )
    before_weight = attention.node_decoder.weight.detach().clone()
    before_bias = attention.node_decoder.bias.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    attention.node_decoder.weight.grad = torch.ones_like(attention.node_decoder.weight)
    attention.node_decoder.bias.grad = torch.ones_like(attention.node_decoder.bias)

    snapshots = _prepare_restricted_attention_node_update(
        model,
        optimizer,
        scope="attention_node_z",
    )
    assert torch.count_nonzero(attention.node_decoder.weight.grad[:2]) == 0
    assert torch.count_nonzero(attention.node_decoder.weight.grad[2]) > 0
    optimizer.step()
    _restore_restricted_updater_mean_update(optimizer, snapshots)

    torch.testing.assert_close(attention.node_decoder.weight[:2], before_weight[:2])
    assert not torch.equal(attention.node_decoder.weight[2], before_weight[2])
    torch.testing.assert_close(attention.node_decoder.bias[:2], before_bias[:2])
    assert not torch.equal(attention.node_decoder.bias[2], before_bias[2])


@pytest.mark.parametrize("scope", ("attention_node_x", "attention_node_y", "attention_node_z"))
def test_axis_selective_attention_scopes_are_valid_configurations(scope: str) -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=[
            "model.dynamics.attention_residual_enabled=true",
            f"training.closed_loop_trainable_scope={scope}",
        ],
    )
    assert config.training.closed_loop_trainable_scope == scope


def test_attention_scope_requires_enabled_attention() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    with pytest.raises(ValueError, match="requires typed attention"):
        set_closed_loop_trainable_scope(model, scope="attention")


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


def test_attention_gradients_share_the_interaction_local_clip_and_diagnostics() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
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
    assert model.dynamics.attention_interactions is not None
    attention = next(model.dynamics.attention_interactions.parameters())
    attention.grad = torch.ones_like(attention)
    attention.grad.mul_(3.0 / attention.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(
        3.0,
        rel=1.0e-5,
        abs=1.0e-5,
    )
    assert diagnostics["interaction_gradient_norm_applied_before_global_clip"] == pytest.approx(
        1.0, abs=1.0e-6
    )
    assert attention.grad.norm().item() == pytest.approx(1.0, abs=1.0e-5)


def test_attention_collision_row_is_bounded_before_complete_interaction_clip() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=10.0,
            interaction_grad_clip_norm=10.0,
            attention_collision_grad_clip_norm=1.0,
            closed_loop_perception_grad_clip_norm=10.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    decoder = attention.relation_decoder
    decoder.weight.grad = torch.zeros_like(decoder.weight)
    collision_gradient = decoder.weight.grad[attention.collision_output_index]
    collision_gradient.fill_(1.0)
    collision_gradient.mul_(3.0 / collision_gradient.norm())
    other = attention.scene_projection.bias
    other.grad = torch.ones_like(other)
    other.grad.mul_(4.0 / other.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["attention_collision_gradient_norm_pre_clip"] == pytest.approx(3.0)
    assert diagnostics["attention_collision_gradient_clip_coefficient"] == pytest.approx(
        1.0 / (3.0 + 1.0e-6)
    )
    assert diagnostics[
        "attention_collision_gradient_norm_applied_before_interaction_clip"
    ] == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["interaction_gradient_norm_after_attention_collision_clip"] == pytest.approx(
        math.sqrt(17.0), abs=1.0e-5
    )
    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(5.0, abs=1.0e-5)
    assert diagnostics["interaction_gradient_total_clip_coefficient"] == pytest.approx(
        math.sqrt(17.0) / 5.0,
        abs=1.0e-5,
    )
    assert collision_gradient.norm().item() == pytest.approx(1.0, abs=1.0e-6)
    assert other.grad.norm().item() == pytest.approx(4.0, abs=1.0e-6)


def test_attention_node_rows_are_jointly_bounded_before_complete_interaction_clip() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=20.0,
            interaction_grad_clip_norm=20.0,
            attention_node_grad_clip_norm=1.0,
            closed_loop_perception_grad_clip_norm=20.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    decoder = attention.node_decoder
    decoder.weight.grad = torch.zeros_like(decoder.weight)
    decoder.weight.grad[0].fill_(1.0)
    decoder.weight.grad[0].mul_(3.0 / decoder.weight.grad[0].norm())
    decoder.weight.grad[1].fill_(1.0)
    decoder.weight.grad[1].mul_(4.0 / decoder.weight.grad[1].norm())
    other = attention.scene_projection.bias
    other.grad = torch.ones_like(other)
    other.grad.mul_(12.0 / other.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["attention_node_gradient_norm_pre_clip"] == pytest.approx(5.0)
    assert diagnostics["attention_node_gradient_clip_coefficient"] == pytest.approx(
        1.0 / (5.0 + 1.0e-6)
    )
    assert diagnostics[
        "attention_node_gradient_norm_applied_before_interaction_clip"
    ] == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["interaction_gradient_norm_after_attention_node_clip"] == pytest.approx(
        math.sqrt(145.0), abs=1.0e-5
    )
    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(13.0, abs=1.0e-5)
    assert decoder.weight.grad.norm().item() == pytest.approx(1.0, abs=1.0e-6)
    assert other.grad.norm().item() == pytest.approx(12.0, abs=1.0e-6)


def test_attention_node_and_relation_row_hierarchy_reconstructs_raw_norm() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=20.0,
            interaction_grad_clip_norm=20.0,
            attention_node_grad_clip_norm=1.0,
            attention_collision_grad_clip_norm=1.0,
            closed_loop_perception_grad_clip_norm=20.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    attention.node_decoder.weight.grad = torch.zeros_like(attention.node_decoder.weight)
    node_gradient = attention.node_decoder.weight.grad[0]
    node_gradient.fill_(1.0)
    node_gradient.mul_(3.0 / node_gradient.norm())
    attention.relation_decoder.weight.grad = torch.zeros_like(attention.relation_decoder.weight)
    collision_gradient = attention.relation_decoder.weight.grad[attention.collision_output_index]
    collision_gradient.fill_(1.0)
    collision_gradient.mul_(4.0 / collision_gradient.norm())
    other = attention.scene_projection.bias
    other.grad = torch.ones_like(other)
    other.grad.mul_(12.0 / other.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(13.0, abs=1.0e-5)
    assert diagnostics["interaction_gradient_norm_after_attention_node_clip"] == pytest.approx(
        math.sqrt(161.0), abs=1.0e-5
    )
    assert diagnostics["interaction_gradient_norm_after_attention_collision_clip"] == pytest.approx(
        math.sqrt(146.0), abs=1.0e-5
    )
    assert diagnostics["interaction_gradient_norm_after_attention_row_clips"] == pytest.approx(
        math.sqrt(146.0), abs=1.0e-5
    )


def test_attention_force_rows_are_jointly_bounded_before_complete_interaction_clip() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=20.0,
            interaction_grad_clip_norm=20.0,
            attention_force_grad_clip_norm=1.0,
            closed_loop_perception_grad_clip_norm=20.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    decoder = attention.relation_decoder
    decoder.weight.grad = torch.zeros_like(decoder.weight)
    normal_gradient = decoder.weight.grad[attention.force_output_indices[0]]
    tangent_gradient = decoder.weight.grad[attention.force_output_indices[1]]
    normal_gradient.fill_(1.0)
    normal_gradient.mul_(3.0 / normal_gradient.norm())
    tangent_gradient.fill_(1.0)
    tangent_gradient.mul_(4.0 / tangent_gradient.norm())
    other = attention.scene_projection.bias
    other.grad = torch.ones_like(other)
    other.grad.mul_(12.0 / other.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["attention_force_gradient_norm_pre_clip"] == pytest.approx(5.0)
    assert diagnostics["attention_force_gradient_clip_coefficient"] == pytest.approx(
        1.0 / (5.0 + 1.0e-6)
    )
    assert diagnostics[
        "attention_force_gradient_norm_applied_before_interaction_clip"
    ] == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["interaction_gradient_norm_after_attention_row_clips"] == pytest.approx(
        math.sqrt(145.0), abs=1.0e-5
    )
    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(13.0, abs=1.0e-5)
    assert torch.linalg.vector_norm(
        torch.stack((normal_gradient.norm(), tangent_gradient.norm()))
    ).item() == pytest.approx(1.0, abs=1.0e-6)
    assert other.grad.norm().item() == pytest.approx(12.0, abs=1.0e-6)


def test_attention_impulse_rows_are_jointly_bounded_before_complete_interaction_clip() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=20.0,
            interaction_grad_clip_norm=20.0,
            attention_impulse_grad_clip_norm=1.0,
            closed_loop_perception_grad_clip_norm=20.0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    decoder = attention.relation_decoder
    decoder.weight.grad = torch.zeros_like(decoder.weight)
    multiplier_gradient = decoder.weight.grad[attention.impulse_output_indices[0]]
    additive_gradient = decoder.weight.grad[attention.impulse_output_indices[1]]
    multiplier_gradient.fill_(1.0)
    multiplier_gradient.mul_(6.0 / multiplier_gradient.norm())
    additive_gradient.fill_(1.0)
    additive_gradient.mul_(8.0 / additive_gradient.norm())
    other = attention.scene_projection.bias
    other.grad = torch.ones_like(other)
    other.grad.mul_(12.0 / other.grad.norm())

    diagnostics = _clip_training_gradients(model, config)

    assert diagnostics["attention_impulse_gradient_norm_pre_clip"] == pytest.approx(10.0)
    assert diagnostics["attention_impulse_gradient_clip_coefficient"] == pytest.approx(
        1.0 / (10.0 + 1.0e-6)
    )
    assert diagnostics[
        "attention_impulse_gradient_norm_applied_before_interaction_clip"
    ] == pytest.approx(1.0, abs=1.0e-6)
    assert diagnostics["interaction_gradient_norm_after_attention_row_clips"] == pytest.approx(
        math.sqrt(145.0), abs=1.0e-5
    )
    assert diagnostics["interaction_gradient_norm_pre_clip"] == pytest.approx(
        math.sqrt(244.0), abs=1.0e-5
    )
    assert torch.linalg.vector_norm(
        torch.stack((multiplier_gradient.norm(), additive_gradient.norm()))
    ).item() == pytest.approx(1.0, abs=1.0e-6)
    assert other.grad.norm().item() == pytest.approx(12.0, abs=1.0e-6)


def test_complete_interaction_retention_gate_rejects_starved_update() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["training.minimum_interaction_gradient_retention=0.1"],
    )

    with pytest.raises(
        FloatingPointError,
        match=r"retained only 0\.002.*optimiser step 200",
    ) as captured:
        _assert_interaction_gradient_retention(
            {
                "interaction_gradient_clip_coefficient": 0.002,
                "interaction_gradient_norm_pre_clip": 500.0,
            },
            config,
            optimizer_step=200,
        )
    assert captured.value.diagnostics == {
        "interaction_gradient_clip_coefficient": 0.002,
        "interaction_gradient_norm_pre_clip": 500.0,
        "optimizer_step_attempted": 200.0,
        "minimum_interaction_gradient_retention": 0.1,
        "optimizer_update_applied": 0.0,
    }

    _assert_interaction_gradient_retention(
        {"interaction_gradient_clip_coefficient": 0.1},
        config,
        optimizer_step=200,
    )


def test_attention_typed_output_hooks_clip_before_shared_backpropagation() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    attention.train()
    attention.configure_output_gradient_clipping(
        node=0.1,
        collision=0.1,
        force=0.1,
        impulse=0.1,
    )
    attention.reset_output_gradient_diagnostics()

    node_source = torch.zeros(1, 1, 3, requires_grad=True)
    relation_source = torch.zeros(1, 1, attention.relation_output_dim, requires_grad=True)
    node_values = node_source * 2.0
    relation_values = relation_source * 2.0
    node_values.retain_grad()
    relation_values.retain_grad()
    attention._register_output_gradient_hooks(node_values, relation_values)
    node_signal = torch.tensor([[[3.0, 4.0, 0.0]]])
    relation_signal = torch.zeros_like(relation_values)
    relation_signal[..., 0] = 2.0
    relation_signal[..., attention.collision_output_index] = 3.0
    relation_signal[..., attention.force_output_indices[0]] = 3.0
    relation_signal[..., attention.force_output_indices[1]] = 4.0
    relation_signal[..., attention.impulse_output_indices[0]] = 6.0
    relation_signal[..., attention.impulse_output_indices[1]] = 8.0

    ((node_values * node_signal).sum() + (relation_values * relation_signal).sum()).backward()

    assert node_values.grad is not None
    assert relation_values.grad is not None
    assert node_values.grad.norm().item() == pytest.approx(0.1, abs=1.0e-6)
    assert relation_values.grad[..., 0].item() == pytest.approx(2.0)
    assert relation_values.grad[..., attention.collision_output_index].norm().item() == (
        pytest.approx(0.1, abs=1.0e-6)
    )
    force_gradient = relation_values.grad[..., list(attention.force_output_indices)]
    assert force_gradient.norm().item() == pytest.approx(0.1, abs=1.0e-6)
    assert node_source.grad is not None
    assert relation_source.grad is not None
    assert node_source.grad.norm().item() == pytest.approx(0.2, abs=1.0e-6)
    assert relation_source.grad[..., 0].item() == pytest.approx(4.0)
    assert relation_source.grad[..., attention.collision_output_index].norm().item() == (
        pytest.approx(0.2, abs=1.0e-6)
    )
    source_force_gradient = relation_source.grad[..., list(attention.force_output_indices)]
    assert source_force_gradient.norm().item() == pytest.approx(0.2, abs=1.0e-6)
    impulse_gradient = relation_values.grad[..., list(attention.impulse_output_indices)]
    assert impulse_gradient.norm().item() == pytest.approx(0.1, abs=1.0e-6)
    source_impulse_gradient = relation_source.grad[..., list(attention.impulse_output_indices)]
    assert source_impulse_gradient.norm().item() == pytest.approx(0.2, abs=1.0e-6)

    diagnostics = attention.output_gradient_diagnostics()
    for name, raw in (
        ("node", 5.0),
        ("collision", 3.0),
        ("force", 5.0),
        ("impulse", 10.0),
    ):
        prefix = f"attention_{name}_output_backprop_gradient"
        assert diagnostics[f"{prefix}_local_clip_enabled"] == 1.0
        assert diagnostics[f"{prefix}_invocation_count"] == 1.0
        assert diagnostics[f"{prefix}_norm_pre_clip"] == pytest.approx(raw)
        assert diagnostics[f"{prefix}_norm_applied_before_parameter_clips"] == pytest.approx(
            0.1, abs=1.0e-6
        )


def test_frozen_attention_cannot_configure_or_register_upstream_gradient_hooks() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    set_closed_loop_trainable_scope(model, scope="state_roi")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    assert not attention.has_trainable_parameters()
    attention.train()
    attention.configure_output_gradient_clipping(
        node=0.1,
        collision=0.1,
        force=0.1,
        impulse=0.1,
    )
    attention.reset_output_gradient_diagnostics()

    node_source = torch.zeros(1, 1, 3, requires_grad=True)
    relation_source = torch.zeros(1, 1, attention.relation_output_dim, requires_grad=True)
    node_values = node_source * 2.0
    relation_values = relation_source * 2.0
    node_values.retain_grad()
    relation_values.retain_grad()
    attention._register_output_gradient_hooks(node_values, relation_values)
    node_signal = torch.tensor([[[3.0, 4.0, 0.0]]])
    relation_signal = torch.zeros_like(relation_values)
    relation_signal[..., attention.collision_output_index] = 3.0
    relation_signal[..., attention.force_output_indices[0]] = 3.0
    relation_signal[..., attention.force_output_indices[1]] = 4.0
    relation_signal[..., attention.impulse_output_indices[0]] = 6.0
    relation_signal[..., attention.impulse_output_indices[1]] = 8.0

    ((node_values * node_signal).sum() + (relation_values * relation_signal).sum()).backward()

    assert node_values.grad is not None
    assert relation_values.grad is not None
    torch.testing.assert_close(node_values.grad, node_signal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(relation_values.grad, relation_signal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(node_source.grad, 2.0 * node_signal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        relation_source.grad,
        2.0 * relation_signal,
        rtol=0.0,
        atol=0.0,
    )
    diagnostics = attention.output_gradient_diagnostics()
    for name in ("node", "collision", "force", "impulse"):
        prefix = f"attention_{name}_output_backprop_gradient"
        assert diagnostics[f"{prefix}_local_clip_enabled"] == 0.0
        assert diagnostics[f"{prefix}_invocation_count"] == 0.0
        assert diagnostics[f"{prefix}_norm_pre_clip"] == 0.0


def test_attention_typed_output_hooks_bound_aggregate_recursive_gradient() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    attention.train()
    attention.configure_output_gradient_clipping(
        node=0.1,
        collision=0.1,
        force=0.1,
        impulse=0.1,
    )
    attention.reset_output_gradient_diagnostics()

    invocation_count = 16
    node_values = [torch.zeros(1, 1, 3, requires_grad=True) for _ in range(invocation_count)]
    relation_values = [
        torch.zeros(1, 1, attention.relation_output_dim, requires_grad=True)
        for _ in range(invocation_count)
    ]
    for node_value, relation_value in zip(node_values, relation_values, strict=True):
        attention._register_output_gradient_hooks(node_value, relation_value)

    losses = []
    for node_value, relation_value in zip(node_values, relation_values, strict=True):
        relation_signal = torch.zeros_like(relation_value)
        relation_signal[..., attention.collision_output_index] = 3.0
        relation_signal[..., attention.force_output_indices[0]] = 3.0
        relation_signal[..., attention.force_output_indices[1]] = 4.0
        relation_signal[..., attention.impulse_output_indices[0]] = 6.0
        relation_signal[..., attention.impulse_output_indices[1]] = 8.0
        losses.append((node_value * 5.0).sum() + (relation_value * relation_signal).sum())
    torch.stack(losses).sum().backward()

    diagnostics = attention.output_gradient_diagnostics()
    for name in ("node", "collision", "force", "impulse"):
        prefix = f"attention_{name}_output_backprop_gradient"
        assert diagnostics[f"{prefix}_invocation_count"] == float(invocation_count)
        assert diagnostics[f"{prefix}_norm_applied_before_parameter_clips"] == pytest.approx(
            0.1, abs=2.0e-6
        )


def test_attention_raw_gradient_diagnostics_localize_parameters_and_typed_rows() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None

    scene_gradient = attention.scene_projection.bias
    scene_gradient.grad = torch.ones_like(scene_gradient)
    scene_gradient.grad.mul_(4.0 / scene_gradient.grad.norm())

    relation_weight = attention.relation_decoder.weight
    relation_bias = attention.relation_decoder.bias
    relation_weight.grad = torch.zeros_like(relation_weight)
    relation_bias.grad = torch.zeros_like(relation_bias)
    normal_row = relation_weight.grad[2]
    normal_row.fill_(1.0)
    normal_row.mul_(3.0 / normal_row.norm())
    relation_bias.grad[2] = 4.0

    node_weight = attention.node_decoder.weight
    node_weight.grad = torch.zeros_like(node_weight)
    node_x = node_weight.grad[0]
    node_x.fill_(1.0)
    node_x.mul_(2.0 / node_x.norm())

    before = {
        name: parameter.grad.detach().clone()
        for name, parameter in attention.named_parameters()
        if parameter.grad is not None
    }
    diagnostics = _attention_gradient_diagnostics(model)

    assert diagnostics[
        "attention_parameter_gradient_norm_pre_clip__scene_projection__bias"
    ] == pytest.approx(4.0)
    assert diagnostics[
        "attention_parameter_gradient_norm_pre_clip__relation_decoder__weight"
    ] == pytest.approx(3.0)
    assert diagnostics[
        "attention_parameter_gradient_norm_pre_clip__relation_decoder__bias"
    ] == pytest.approx(4.0)
    assert diagnostics["attention_relation_output_gradient_norm_pre_clip@normal_force"] == (
        pytest.approx(5.0)
    )
    assert diagnostics["attention_node_output_gradient_norm_pre_clip@x"] == pytest.approx(2.0)
    assert diagnostics["attention_relation_output_gradient_norm_pre_clip@collision"] == 0.0
    for name, parameter in attention.named_parameters():
        if name in before:
            assert torch.equal(parameter.grad, before[name])


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


def test_updater_state_heads_scope_has_exact_functional_tensor_boundary() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    set_closed_loop_trainable_scope(model, scope="updater_state_heads")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "updater.learned_corrector.mean_head.weight",
        "updater.learned_corrector.mean_head.bias",
        "updater.learned_corrector.variance_head.weight",
        "updater.learned_corrector.variance_head.bias",
        "updater.learned_corrector.gate_head.weight",
        "updater.learned_corrector.gate_head.bias",
    }


def test_updater_state_heads_adamw_step_cannot_mutate_frozen_siblings() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    set_closed_loop_trainable_scope(model, scope="updater_state_heads")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    corrector = model.updater.learned_corrector
    assert corrector is not None
    batch = 3
    corrector_inputs = {
        "prior_fast_state": torch.zeros(batch, corrector.fast_state_dim),
        "prior_log_variance": torch.zeros(batch, corrector.fast_state_dim),
        "whitened_innovation": torch.ones(batch, corrector.fast_state_dim),
        "association_cost": torch.zeros(batch),
        "ambiguity": torch.zeros(batch, dtype=torch.bool),
        "visibility": torch.ones(batch),
        "elapsed_time": torch.full((batch,), 0.05),
        "motion_mode_logits": torch.zeros(batch, corrector.num_motion_modes),
        "modality_index": torch.zeros(batch, dtype=torch.int64),
    }
    with torch.no_grad():
        output_before = corrector(**corrector_inputs)

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    with torch.no_grad():
        output_after = corrector(**corrector_inputs)

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, before[name]), name
            assert parameter in optimizer.state, name
        else:
            torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)
            assert parameter not in optimizer.state, name
    # The selected state heads move, but the frozen shared representation
    # cannot indirectly alter mode, existence, or visibility semantics.
    assert not torch.equal(output_after.mean_delta, output_before.mean_delta)
    assert not torch.equal(output_after.log_variance_delta, output_before.log_variance_delta)
    assert not torch.equal(output_after.state_gate, output_before.state_gate)
    assert torch.equal(output_after.mode_logit_delta, output_before.mode_logit_delta)
    assert torch.equal(output_after.existence_delta, output_before.existence_delta)
    assert torch.equal(output_after.visibility_delta, output_before.visibility_delta)


def test_updater_state_heads_xy_preserves_z_and_nonkinematic_rows_through_adamw() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    set_closed_loop_trainable_scope(model, scope="updater_state_heads_xy")
    corrector = model.updater.learned_corrector
    assert corrector is not None
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "updater.learned_corrector.mean_head.weight",
        "updater.learned_corrector.mean_head.bias",
        "updater.learned_corrector.variance_head.weight",
        "updater.learned_corrector.variance_head.bias",
        "updater.learned_corrector.gate_head.weight",
        "updater.learned_corrector.gate_head.bias",
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    heads = (corrector.mean_head, corrector.variance_head, corrector.gate_head)
    batch = 2
    corrector_inputs = {
        "prior_fast_state": torch.zeros(batch, corrector.fast_state_dim),
        "prior_log_variance": torch.zeros(batch, corrector.fast_state_dim),
        "whitened_innovation": torch.ones(batch, corrector.fast_state_dim),
        "association_cost": torch.zeros(batch),
        "ambiguity": torch.zeros(batch, dtype=torch.bool),
        "visibility": torch.ones(batch),
        "elapsed_time": torch.full((batch,), 0.05),
        "motion_mode_logits": torch.zeros(batch, corrector.num_motion_modes),
        "modality_index": torch.zeros(batch, dtype=torch.int64),
    }
    with torch.no_grad():
        output_before = corrector(**corrector_inputs)
    before = {
        parameter: parameter.detach().clone()
        for head in heads
        for parameter in (head.weight, head.bias)
    }
    for parameter in before:
        parameter.grad = torch.ones_like(parameter)

    snapshots = _prepare_restricted_updater_mean_update(
        model,
        optimizer,
        scope="updater_state_heads_xy",
    )
    assert len(snapshots) == 6
    for parameter in before:
        assert torch.count_nonzero(parameter.grad[[0, 1, 3, 4]]) > 0
        assert torch.count_nonzero(parameter.grad[2]) == 0
        assert torch.count_nonzero(parameter.grad[5:]) == 0
    optimizer.step()
    _restore_restricted_updater_mean_update(optimizer, snapshots)

    with torch.no_grad():
        output_after = corrector(**corrector_inputs)

    for parameter, original in before.items():
        assert not torch.equal(parameter[[0, 1, 3, 4]], original[[0, 1, 3, 4]])
        torch.testing.assert_close(parameter[2], original[2], rtol=0.0, atol=0.0)
        torch.testing.assert_close(parameter[5:], original[5:], rtol=0.0, atol=0.0)
        state = optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][2]) == 0
        assert torch.count_nonzero(state["exp_avg"][5:]) == 0
    for before_output, after_output in (
        (output_before.mean_delta, output_after.mean_delta),
        (output_before.log_variance_delta, output_after.log_variance_delta),
        (output_before.state_gate, output_after.state_gate),
    ):
        assert not torch.equal(after_output[..., [0, 1, 3, 4]], before_output[..., [0, 1, 3, 4]])
        torch.testing.assert_close(after_output[..., 2], before_output[..., 2], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            after_output[..., 5:],
            before_output[..., 5:],
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "scope",
    (
        "updater_state_heads_xy_collision",
        "updater_state_heads_xy_collision_node",
    ),
)
def test_combined_xy_collision_scope_has_exact_typed_tensor_boundary(scope: str) -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope=scope,
            closed_loop_event_loss_weights={scope: 0.05},
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config)

    set_closed_loop_trainable_scope(model, scope=scope)

    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
        "updater.learned_corrector.mean_head.weight",
        "updater.learned_corrector.mean_head.bias",
        "updater.learned_corrector.variance_head.weight",
        "updater.learned_corrector.variance_head.bias",
        "updater.learned_corrector.gate_head.weight",
        "updater.learned_corrector.gate_head.bias",
        "dynamics.attention_interactions.relation_decoder.weight",
        "dynamics.attention_interactions.relation_decoder.bias",
    }
    attention = model.dynamics.attention_interactions
    assert attention is not None
    assert not attention.zero_output_bypass_eligible()


def test_combined_xy_collision_scope_routes_event_only_to_collision_row() -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope="updater_state_heads_xy_collision",
            closed_loop_event_loss_weights={"updater_state_heads_xy_collision": 0.05},
            loss_weights={"state_position": 2.0, "event": 0.05},
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope="updater_state_heads_xy_collision")
    corrector = model.updater.learned_corrector
    attention = model.dynamics.attention_interactions
    assert corrector is not None
    assert attention is not None

    state_signal = corrector.mean_head.weight.sum()
    relation_values = attention.relation_decoder(
        torch.ones(1, attention.width, dtype=state_signal.dtype)
    )
    other_rows = [
        row
        for row in range(attention.relation_output_dim)
        if row != attention.collision_output_index
    ]
    event = (
        relation_values[..., attention.collision_output_index].sum()
        + 2.0 * relation_values[..., other_rows].sum()
        + state_signal
    )
    terms = {"state_position": state_signal * 3.0, "event": event}
    weights, _ = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="updater_state_heads_xy_collision",
    )
    result = TrainingBatchResult(
        total_loss=_weighted_closed_loop_total(terms, weights),
        loss_terms=terms,
        metrics={},
        phase="closed_loop_rgb",
    )

    _backward_training_result(
        model,
        result,
        config,
        active_scope="updater_state_heads_xy_collision",
    )

    torch.testing.assert_close(
        corrector.mean_head.weight.grad,
        torch.full_like(corrector.mean_head.weight, 6.0),
    )
    collision_row = attention.collision_output_index
    assert torch.count_nonzero(attention.relation_decoder.weight.grad[collision_row]) > 0
    assert torch.count_nonzero(attention.relation_decoder.bias.grad[collision_row]) > 0
    assert torch.count_nonzero(attention.relation_decoder.weight.grad[other_rows]) == 0
    assert torch.count_nonzero(attention.relation_decoder.bias.grad[other_rows]) == 0
    assert result.metrics["direct_collision_event_owner_active"] == 1.0
    assert result.metrics["direct_collision_event_objective_supported"] == 1.0
    assert result.metrics["direct_collision_event_loss_weight"] == 0.05
    assert result.metrics["direct_collision_event_noncollision_gradient_discarded_norm"] > 0.0


def test_node_xy_collision_scope_routes_node_event_not_opposing_pair_event() -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    scope = "updater_state_heads_xy_collision_node"
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope=scope,
            closed_loop_event_loss_weights={scope: 0.0045},
            loss_weights={"state_position": 2.0, "event": 0.0045},
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope=scope)
    corrector = model.updater.learned_corrector
    attention = model.dynamics.attention_interactions
    assert corrector is not None
    assert attention is not None

    state_signal = corrector.mean_head.weight.sum()
    relation_values = attention.relation_decoder(
        torch.ones(1, attention.width, dtype=state_signal.dtype)
    )
    collision_row = attention.collision_output_index
    other_rows = [row for row in range(attention.relation_output_dim) if row != collision_row]
    node_event = relation_values[..., collision_row].sum()
    pair_event = -node_event + 2.0 * relation_values[..., other_rows].sum()
    combined_event = 0.5 * (node_event + pair_event)
    terms = {"state_position": state_signal * 3.0, "event": combined_event}
    weights, _ = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope=scope,
    )
    result = TrainingBatchResult(
        total_loss=_weighted_closed_loop_total(terms, weights),
        loss_terms=terms,
        metrics={},
        phase="closed_loop_rgb",
        support_terms={"event_collision_node": node_event},
    )

    _backward_training_result(model, result, config, active_scope=scope)

    torch.testing.assert_close(
        corrector.mean_head.weight.grad,
        torch.full_like(corrector.mean_head.weight, 6.0),
    )
    torch.testing.assert_close(
        attention.relation_decoder.weight.grad[collision_row],
        torch.full_like(attention.relation_decoder.weight.grad[collision_row], 0.0045),
    )
    torch.testing.assert_close(
        attention.relation_decoder.bias.grad[collision_row],
        torch.full_like(attention.relation_decoder.bias.grad[collision_row], 0.0045),
    )
    assert torch.count_nonzero(attention.relation_decoder.weight.grad[other_rows]) == 0
    assert torch.count_nonzero(attention.relation_decoder.bias.grad[other_rows]) == 0
    assert result.metrics["direct_collision_event_owner_active"] == 1.0
    assert result.metrics["direct_collision_event_node_only_routing_active"] == 1.0
    assert result.metrics["direct_collision_event_loss_weight"] == 0.0045
    assert result.metrics["direct_collision_state_event_routing_active"] == 0.0
    assert result.metrics["direct_collision_state_event_loss_weight"] == 0.0
    assert result.metrics["direct_collision_state_event_gradient_norm_pre_parameter_clip"] == 0.0


def test_node_xy_collision_scope_routes_node_event_to_selected_state_rows() -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    scope = "updater_state_heads_xy_collision_node"
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope=scope,
            closed_loop_event_loss_weights={scope: 0.0045},
            closed_loop_state_event_loss_weights={scope: 0.04},
            loss_weights={"state_position": 2.0, "event": 0.0045},
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope=scope)
    corrector = model.updater.learned_corrector
    attention = model.dynamics.attention_interactions
    assert corrector is not None
    assert attention is not None

    state_signal = corrector.mean_head.weight.sum()
    relation_values = attention.relation_decoder(
        torch.ones(1, attention.width, dtype=state_signal.dtype)
    )
    collision_row = attention.collision_output_index
    node_event = state_signal + relation_values[..., collision_row].sum()
    terms = {"state_position": state_signal * 3.0, "event": node_event.detach()}
    weights, _ = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope=scope,
    )
    result = TrainingBatchResult(
        total_loss=_weighted_closed_loop_total(terms, weights),
        loss_terms=terms,
        metrics={},
        phase="closed_loop_rgb",
        support_terms={"event_collision_node": node_event},
    )

    _backward_training_result(model, result, config, active_scope=scope)

    expected = torch.full_like(corrector.mean_head.weight, 6.0)
    expected[[0, 1, 3, 4]] += 0.04
    torch.testing.assert_close(corrector.mean_head.weight.grad, expected)
    torch.testing.assert_close(
        attention.relation_decoder.weight.grad[collision_row],
        torch.full_like(attention.relation_decoder.weight.grad[collision_row], 0.0045),
    )
    assert result.metrics["direct_collision_state_event_routing_active"] == 1.0
    assert result.metrics["direct_collision_state_event_loss_weight"] == 0.04
    assert result.metrics["direct_collision_state_event_gradient_norm_pre_parameter_clip"] > 0.0
    assert result.metrics["direct_collision_state_event_excluded_gradient_discarded_norm"] > 0.0


def test_combined_xy_collision_scope_allows_sparse_batch_without_event_term() -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope="updater_state_heads_xy_collision",
            closed_loop_event_loss_weights={"updater_state_heads_xy_collision": 0.05},
            loss_weights={"state_position": 2.0, "event": 0.05},
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope="updater_state_heads_xy_collision")
    corrector = model.updater.learned_corrector
    attention = model.dynamics.attention_interactions
    assert corrector is not None
    assert attention is not None

    state_signal = corrector.mean_head.weight.sum()
    terms = {"state_position": state_signal * 3.0}
    weights, _ = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="updater_state_heads_xy_collision",
    )
    result = TrainingBatchResult(
        total_loss=_weighted_closed_loop_total(terms, weights),
        loss_terms=terms,
        metrics={},
        phase="closed_loop_rgb",
    )

    _backward_training_result(
        model,
        result,
        config,
        active_scope="updater_state_heads_xy_collision",
    )

    torch.testing.assert_close(
        corrector.mean_head.weight.grad,
        torch.full_like(corrector.mean_head.weight, 6.0),
    )
    assert attention.relation_decoder.weight.grad is None
    assert attention.relation_decoder.bias.grad is None
    assert result.metrics == {
        "direct_collision_event_owner_active": 1.0,
        "direct_collision_event_node_only_routing_active": 0.0,
        "direct_collision_event_objective_supported": 0.0,
        "direct_collision_event_loss_weight": 0.05,
        "direct_collision_event_gradient_norm_pre_parameter_clip": 0.0,
        "direct_collision_event_noncollision_gradient_discarded_norm": 0.0,
        "direct_collision_state_event_routing_active": 0.0,
        "direct_collision_state_event_loss_weight": 0.0,
        "direct_collision_state_event_gradient_norm_pre_parameter_clip": 0.0,
        "direct_collision_state_event_excluded_gradient_discarded_norm": 0.0,
    }


def test_combined_xy_collision_scope_rejects_present_detached_event_term() -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope="updater_state_heads_xy_collision",
            closed_loop_event_loss_weights={"updater_state_heads_xy_collision": 0.05},
            loss_weights={"state_position": 2.0, "event": 0.05},
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope="updater_state_heads_xy_collision")
    corrector = model.updater.learned_corrector
    assert corrector is not None
    state_signal = corrector.mean_head.weight.sum()
    terms = {"state_position": state_signal * 3.0, "event": state_signal.detach()}
    weights, _ = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="updater_state_heads_xy_collision",
    )
    result = TrainingBatchResult(
        total_loss=_weighted_closed_loop_total(terms, weights),
        loss_terms=terms,
        metrics={},
        phase="closed_loop_rgb",
    )

    with pytest.raises(
        RuntimeError,
        match="direct collision ownership requires a differentiable event loss",
    ):
        _backward_training_result(
            model,
            result,
            config,
            active_scope="updater_state_heads_xy_collision",
        )
    assert corrector.mean_head.weight.grad is None


def test_combined_xy_collision_scope_preserves_noncollision_rows_through_adamw() -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope="updater_state_heads_xy_collision",
            closed_loop_event_loss_weights={"updater_state_heads_xy_collision": 0.05},
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope="updater_state_heads_xy_collision")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    with torch.no_grad():
        attention.relation_decoder.weight.copy_(
            torch.arange(
                attention.relation_decoder.weight.numel(),
                dtype=attention.relation_decoder.weight.dtype,
            ).reshape_as(attention.relation_decoder.weight)
            / 1000
        )
        attention.relation_decoder.bias.copy_(
            torch.arange(
                attention.relation_decoder.bias.numel(),
                dtype=attention.relation_decoder.bias.dtype,
            )
            / 1000
        )
    before_weight = attention.relation_decoder.weight.detach().clone()
    before_bias = attention.relation_decoder.bias.detach().clone()
    attention.relation_decoder.weight.grad = torch.ones_like(attention.relation_decoder.weight)
    attention.relation_decoder.bias.grad = torch.ones_like(attention.relation_decoder.bias)

    snapshots = _prepare_restricted_attention_collision_update(
        model,
        optimizer,
        scope="updater_state_heads_xy_collision",
    )
    optimizer.step()
    _restore_restricted_updater_mean_update(optimizer, snapshots)

    collision_row = attention.collision_output_index
    other_rows = [row for row in range(attention.relation_output_dim) if row != collision_row]
    assert not torch.equal(
        attention.relation_decoder.weight[collision_row],
        before_weight[collision_row],
    )
    assert not torch.equal(
        attention.relation_decoder.bias[collision_row],
        before_bias[collision_row],
    )
    torch.testing.assert_close(
        attention.relation_decoder.weight[other_rows],
        before_weight[other_rows],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        attention.relation_decoder.bias[other_rows],
        before_bias[other_rows],
        rtol=0.0,
        atol=0.0,
    )


def test_updater_mean_scope_isolates_semantically_reset_gain_head() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    set_closed_loop_trainable_scope(model, scope="updater_mean")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "updater.learned_corrector.mean_head.weight",
        "updater.learned_corrector.mean_head.bias",
    }


def test_updater_mean_y_scope_preserves_every_excluded_row_through_adamw() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    set_closed_loop_trainable_scope(model, scope="updater_mean_y")
    corrector = model.updater.learned_corrector
    assert corrector is not None
    with torch.no_grad():
        corrector.mean_head.weight.copy_(
            torch.arange(
                corrector.mean_head.weight.numel(),
                dtype=corrector.mean_head.weight.dtype,
            ).reshape_as(corrector.mean_head.weight)
            / 1000
        )
        corrector.mean_head.bias.copy_(
            torch.arange(
                corrector.mean_head.bias.numel(),
                dtype=corrector.mean_head.bias.dtype,
            )
            / 1000
        )
    before_weight = corrector.mean_head.weight.detach().clone()
    before_bias = corrector.mean_head.bias.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    corrector.mean_head.weight.grad = torch.ones_like(corrector.mean_head.weight)
    corrector.mean_head.bias.grad = torch.ones_like(corrector.mean_head.bias)

    snapshots = _prepare_restricted_updater_mean_update(
        model,
        optimizer,
        scope="updater_mean_y",
    )
    assert torch.count_nonzero(corrector.mean_head.weight.grad[0]) == 0
    assert torch.count_nonzero(corrector.mean_head.weight.grad[1]) > 0
    assert torch.count_nonzero(corrector.mean_head.weight.grad[2:]) == 0
    optimizer.step()
    _restore_restricted_updater_mean_update(optimizer, snapshots)

    torch.testing.assert_close(corrector.mean_head.weight[0], before_weight[0])
    assert not torch.equal(corrector.mean_head.weight[1], before_weight[1])
    torch.testing.assert_close(corrector.mean_head.weight[2:], before_weight[2:])
    torch.testing.assert_close(corrector.mean_head.bias[0], before_bias[0])
    assert not torch.equal(corrector.mean_head.bias[1], before_bias[1])
    torch.testing.assert_close(corrector.mean_head.bias[2:], before_bias[2:])
    for parameter in (corrector.mean_head.weight, corrector.mean_head.bias):
        state = optimizer.state[parameter]
        frozen = torch.cat((state["exp_avg"][:1], state["exp_avg"][2:]), dim=0)
        assert torch.count_nonzero(frozen) == 0


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


def test_state_roi_scope_trains_fast_state_and_shared_roi_without_dynamics() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_closed_loop_trainable_scope(model, scope="state_roi")

    expected_trainable = {
        name
        for name, _ in model.named_parameters()
        if (
            (
                name.startswith("updater.")
                and not name.startswith("updater.learned_corrector.visibility_head.")
            )
            or (name.startswith("identifier.") and not name.startswith("identifier.variance_head."))
            or (
                name.startswith("observation_modules.rgb.roi_updater.")
                and not name.startswith("observation_modules.rgb.roi_updater.event_head.")
            )
            or name.startswith("observation_modules.rgb.backbone.stages.0.")
            or name.startswith("observation_modules.rgb.backbone.stages.1.")
            or name.startswith("observation_modules.rgb.backbone.fast_projection.")
        )
    }
    actual_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert actual_trainable == expected_trainable
    assert not any(parameter.requires_grad for parameter in model.dynamics.parameters())
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
    assert _global_measurement_has_trainable_path(rgb)
    assert _fast_measurement_has_trainable_perception_path(rgb)


def test_state_roi_optimizer_step_cannot_mutate_frozen_dynamics_or_global_heads() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    set_closed_loop_trainable_scope(model, scope="state_roi")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, before[name]), name
        else:
            torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)


def test_state_relation_roi_scope_has_exact_pair_state_and_roi_boundary() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config)

    set_closed_loop_trainable_scope(model, scope="state_relation_roi")

    expected_trainable = {
        name
        for name, _ in model.named_parameters()
        if (
            (
                name.startswith("updater.")
                and not name.startswith("updater.learned_corrector.visibility_head.")
            )
            or (name.startswith("identifier.") and not name.startswith("identifier.variance_head."))
            or (
                name.startswith("observation_modules.rgb.roi_updater.")
                and not name.startswith("observation_modules.rgb.roi_updater.event_head.")
            )
            or name.startswith("observation_modules.rgb.backbone.stages.0.")
            or name.startswith("observation_modules.rgb.backbone.stages.1.")
            or name.startswith("observation_modules.rgb.backbone.fast_projection.")
            or name.startswith("dynamics.interactions.edge_network.")
            or (
                name.startswith("dynamics.attention_interactions.")
                and not name.startswith("dynamics.attention_interactions.node_decoder.")
            )
        )
    }
    actual_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert actual_trainable == expected_trainable
    assert not any(
        parameter.requires_grad
        for parameter in model.dynamics.interactions.node_network.parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.dynamics.modal.parameters())
    assert not any(parameter.requires_grad for parameter in model.dynamics.events.parameters())
    assert not any(parameter.requires_grad for parameter in model.dynamics.uncertainty.parameters())
    assert model.dynamics.attention_interactions is not None
    assert not any(
        parameter.requires_grad
        for parameter in model.dynamics.attention_interactions.node_decoder.parameters()
    )


def test_state_relation_roi_optimizer_step_cannot_mutate_frozen_node_or_global_paths() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope="state_relation_roi")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, before[name]), name
        else:
            torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)


def test_state_relation_roi_scope_requires_enabled_attention() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    with pytest.raises(ValueError, match="requires typed attention"):
        set_closed_loop_trainable_scope(model, scope="state_relation_roi")


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


def test_state_roi_can_transition_to_state_dynamics_roi_at_causal_boundary() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        training=replace(
            source.training,
            rgb_pretrain_steps=100,
            closed_loop_trainable_scope="state_roi",
            closed_loop_late_trainable_scope="state_dynamics_roi",
            closed_loop_scope_transition_steps=512,
        ),
    )
    config.validate()

    early_scope = _closed_loop_trainable_scope_for_step(config, completed_step=611)
    late_scope = _closed_loop_trainable_scope_for_step(config, completed_step=612)

    assert early_scope == (
        "state_roi",
        False,
    )
    assert late_scope == (
        "state_dynamics_roi",
        True,
    )
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope=early_scope[0])
    assert not any(parameter.requires_grad for parameter in model.dynamics.parameters())
    set_closed_loop_trainable_scope(model, scope=late_scope[0])
    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())


def test_state_roi_can_transition_to_state_relation_roi_without_opening_node_dynamics() -> None:
    source = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    config = replace(
        source,
        training=replace(
            source.training,
            rgb_pretrain_steps=100,
            closed_loop_trainable_scope="state_roi",
            closed_loop_late_trainable_scope="state_relation_roi",
            closed_loop_scope_transition_steps=512,
        ),
    )
    config.validate()

    assert _closed_loop_trainable_scope_for_step(config, completed_step=611) == (
        "state_roi",
        False,
    )
    assert _closed_loop_trainable_scope_for_step(config, completed_step=612) == (
        "state_relation_roi",
        True,
    )
    model = OnlineWorldModel.from_config(config)
    set_closed_loop_trainable_scope(model, scope="state_roi")
    assert not any(parameter.requires_grad for parameter in model.dynamics.parameters())
    attention = model.dynamics.attention_interactions
    assert attention is not None
    assert attention.zero_output_bypass_eligible()
    assert all(
        torch.count_nonzero(parameter) == 0
        for decoder in (attention.node_decoder, attention.relation_decoder)
        for parameter in decoder.parameters()
    )
    set_closed_loop_trainable_scope(model, scope="state_relation_roi")
    assert any(
        parameter.requires_grad
        for parameter in model.dynamics.interactions.edge_network.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.dynamics.interactions.node_network.parameters()
    )
    assert not attention.zero_output_bypass_eligible()
    assert not any(parameter.requires_grad for parameter in attention.node_decoder.parameters())


def test_scope_owned_event_weight_omits_early_gradient_and_admits_exact_late_weight() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_event_loss_weights={
                "state_roi": 0.0,
                "updater_state_heads": 0.0,
                "state_relation_roi": 0.05,
            },
        ),
    )
    config.validate()

    early_weights, early_metrics = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="state_roi",
    )
    early_event = torch.tensor(3.0, requires_grad=True)
    early_state = torch.tensor(2.0, requires_grad=True)
    early_event_gradients: list[torch.Tensor] = []
    early_event.register_hook(early_event_gradients.append)
    early_total = _weighted_closed_loop_total(
        {"event": early_event, "state_position": early_state},
        early_weights,
    )
    early_total.backward()

    assert early_metrics == {
        "effective_event_loss_weight": 0.0,
        "event_loss_scope_override_active": 1.0,
        "event_loss_legacy_weight_active": 0.0,
        "event_loss_suppressed_no_trainable_owner": 1.0,
    }
    assert early_event.grad is None
    assert early_event_gradients == []
    torch.testing.assert_close(
        early_state.grad,
        torch.tensor(config.training.loss_weights["state_position"]),
    )

    head_weights, head_metrics = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="updater_state_heads",
    )
    assert head_weights["event"] == 0.0
    assert head_metrics == {
        "effective_event_loss_weight": 0.0,
        "event_loss_scope_override_active": 1.0,
        "event_loss_legacy_weight_active": 0.0,
        "event_loss_suppressed_no_trainable_owner": 1.0,
    }

    late_weights, late_metrics = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="state_relation_roi",
    )
    late_event = torch.tensor(3.0, requires_grad=True)
    late_state = torch.tensor(2.0, requires_grad=True)
    late_total = _weighted_closed_loop_total(
        {"event": late_event, "state_position": late_state},
        late_weights,
    )
    late_total.backward()

    assert late_metrics == {
        "effective_event_loss_weight": 0.05,
        "event_loss_scope_override_active": 1.0,
        "event_loss_legacy_weight_active": 0.0,
        "event_loss_suppressed_no_trainable_owner": 0.0,
    }
    torch.testing.assert_close(late_event.grad, torch.tensor(0.05))


def test_missing_scope_event_override_is_exactly_legacy_weight() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    weights, metrics = _closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="state_roi",
    )
    event = torch.tensor(3.0, requires_grad=True)
    state = torch.tensor(2.0, requires_grad=True)
    total = _weighted_closed_loop_total(
        {"event": event, "state_position": state},
        weights,
    )
    total.backward()

    expected = config.training.loss_weights["event"]
    assert weights["event"] == expected
    assert metrics == {
        "effective_event_loss_weight": expected,
        "event_loss_scope_override_active": 0.0,
        "event_loss_legacy_weight_active": 1.0,
        "event_loss_suppressed_no_trainable_owner": 0.0,
    }
    torch.testing.assert_close(event.grad, torch.tensor(expected))


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


def test_attention_node_complexity_is_axis_neutral_and_opt_in() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["model.dynamics.attention_residual_enabled=true"],
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    attention = model.dynamics.attention_interactions
    assert attention is not None
    with torch.no_grad():
        attention.node_decoder.weight.zero_()
        attention.node_decoder.bias.zero_()
        attention.node_decoder.weight[0, 0] = 3.0
        attention.node_decoder.weight[1, 0] = 4.0

    details = _attention_node_complexity_details(model)
    terms = _group_closed_loop_terms(details, torch.zeros(()))

    torch.testing.assert_close(details["attention_node_complexity_x"], torch.tensor(9.0))
    torch.testing.assert_close(details["attention_node_complexity_y"], torch.tensor(16.0))
    torch.testing.assert_close(details["attention_node_complexity_z"], torch.tensor(0.0))
    torch.testing.assert_close(
        details["attention_node_complexity"],
        torch.tensor(25.0 / 3.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(terms, {}),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(terms, {"attention_node_complexity": 2.0}),
        torch.tensor(50.0 / 3.0),
    )


def test_attention_node_complexity_is_absent_without_attention() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"), device="cpu")

    assert _attention_node_complexity_details(model) == {}


def test_attention_node_activity_is_an_exact_opt_in_term() -> None:
    details = {
        "state_position": torch.tensor(2.0),
        "attention_node_activity": torch.tensor(0.25),
        "attention_node_drift": torch.tensor(0.125),
    }
    terms = _group_closed_loop_terms(details, torch.zeros(()))

    torch.testing.assert_close(
        _weighted_closed_loop_total(terms, {"state_position": 1.0}),
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {"state_position": 1.0, "attention_node_activity": 4.0},
        ),
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {"state_position": 1.0, "attention_node_drift": 8.0},
        ),
        torch.tensor(3.0),
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


def test_event_loss_uses_fixed_global_horizon_denominator() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    details = {
        # The per-anchor aggregate must not survive as the optimized value once
        # comparable horizon-specific means are available.
        "event_collision": torch.tensor(99.0),
        rollout_horizon_loss_key("event_collision", 0.1): torch.tensor(3.0),
        rollout_horizon_loss_key("event_collision", 0.25): torch.tensor(6.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))
    terms = _group_closed_loop_terms(balanced, torch.zeros(()))

    expected = torch.tensor((1.0 * 3.0 + 1.5 * 6.0) / 4.5)
    torch.testing.assert_close(balanced["event_collision"], expected)
    torch.testing.assert_close(terms["event"], expected)


def test_axiswise_correction_hinges_keep_fixed_configured_horizon_denominator() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=["training.closed_loop_axiswise_correction_hinge_enabled=true"],
    )
    details = {
        "correction_future": torch.tensor(99.0),
        "correction_future_velocity": torch.tensor(88.0),
        rollout_horizon_loss_key("correction_future", 0.1): torch.tensor(3.0),
        rollout_horizon_loss_key("correction_future", 0.25): torch.tensor(6.0),
        rollout_horizon_loss_key("correction_future_velocity", 0.1): torch.tensor(2.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))

    torch.testing.assert_close(
        balanced["correction_future"],
        torch.tensor((1.0 * 3.0 + 1.5 * 6.0) / 4.5),
    )
    # Missing velocity horizons contribute no numerator but retain the same
    # complete configured denominator rather than renormalizing short support.
    torch.testing.assert_close(
        balanced["correction_future_velocity"],
        torch.tensor((1.0 * 2.0) / 4.5),
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


def _complete_additive_physical_evidence(
    metrics: dict[str, float],
    *,
    suffixes: tuple[str, ...] = ("0.100s", "0.250s", "0.500s"),
) -> None:
    current_coordinate_count = metrics["physical_state_position_coordinate_count"]
    metrics.setdefault("physical_state_position_coverage90_hit_count", current_coordinate_count)
    metrics.setdefault(
        "physical_state_position_coverage90_coordinate_count", current_coordinate_count
    )
    metrics.setdefault("physical_state_position_gaussian_nll_sum", 0.1 * current_coordinate_count)
    metrics.setdefault("physical_state_position_sharpness_std_sum", current_coordinate_count)
    metrics.setdefault(
        "physical_state_position_calibration_coordinate_count", current_coordinate_count
    )
    for axis in ("x", "y", "z"):
        metrics.setdefault(
            f"physical_state_velocity_{axis}_sse",
            metrics["physical_state_velocity_sse"] / 3.0,
        )
        axis_count = metrics[f"physical_state_position_{axis}_coordinate_count"]
        metrics.setdefault(f"physical_state_velocity_{axis}_coordinate_count", axis_count)
        metrics.setdefault(f"physical_state_position_{axis}_gaussian_nll_sum", 0.1 * axis_count)
        metrics.setdefault(f"physical_state_position_{axis}_sharpness_std_sum", axis_count)
        metrics.setdefault(
            f"physical_state_position_{axis}_calibration_coordinate_count",
            axis_count,
        )
    for suffix in suffixes:
        position_sse = metrics[f"physical_rollout_position@{suffix}_sse"]
        coordinate_count = metrics[f"physical_rollout_position@{suffix}_coordinate_count"]
        metrics.setdefault(f"physical_rollout_velocity@{suffix}_sse", position_sse)
        metrics.setdefault(
            f"physical_rollout_velocity@{suffix}_coordinate_count",
            coordinate_count,
        )
        calibration_count = metrics[
            f"physical_rollout_position_coverage90@{suffix}_coordinate_count"
        ]
        metrics.setdefault(
            f"physical_rollout_position@{suffix}_gaussian_nll_sum",
            0.1 * calibration_count,
        )
        metrics.setdefault(
            f"physical_rollout_position@{suffix}_sharpness_std_sum",
            calibration_count,
        )
        metrics.setdefault(
            f"physical_rollout_position@{suffix}_calibration_coordinate_count",
            calibration_count,
        )
        for axis in ("x", "y", "z"):
            axis_sse = metrics[f"physical_rollout_position_{axis}@{suffix}_sse"]
            axis_count = metrics[f"physical_rollout_position_{axis}@{suffix}_coordinate_count"]
            metrics.setdefault(f"physical_rollout_velocity_{axis}@{suffix}_sse", axis_sse)
            metrics.setdefault(
                f"physical_rollout_velocity_{axis}@{suffix}_coordinate_count",
                axis_count,
            )
            axis_calibration_count = calibration_count / 3.0
            metrics.setdefault(
                f"physical_rollout_position_{axis}@{suffix}_gaussian_nll_sum",
                0.1 * axis_calibration_count,
            )
            metrics.setdefault(
                f"physical_rollout_position_{axis}@{suffix}_sharpness_std_sum",
                axis_calibration_count,
            )
            metrics.setdefault(
                f"physical_rollout_position_{axis}@{suffix}_calibration_coordinate_count",
                axis_calibration_count,
            )
        metrics.setdefault(f"physical_forecast_identity_mismatch_count@{suffix}", 0.0)
        metrics.setdefault(f"physical_forecast_identity_association_count@{suffix}", 1.0)
        metrics.setdefault(
            f"physical_forecast_identity_eligible_count@{suffix}",
            metrics[f"physical_forecast_identity_association_count@{suffix}"],
        )
        divisor = float(len(suffixes))
        metrics.setdefault(
            f"physical_collision_true_positive_count@{suffix}",
            metrics.get("physical_collision_true_positive_count", divisor) / divisor,
        )
        metrics.setdefault(
            f"physical_collision_false_positive_count@{suffix}",
            metrics.get("physical_collision_false_positive_count", 0.0) / divisor,
        )
        metrics.setdefault(
            f"physical_collision_false_negative_count@{suffix}",
            metrics.get("physical_collision_false_negative_count", 0.0) / divisor,
        )
        metrics.setdefault(
            f"physical_collision_true_negative_count@{suffix}",
            metrics.get("physical_collision_true_negative_count", divisor) / divisor,
        )
    metrics.setdefault(
        "physical_collision_true_negative_count",
        sum(metrics[f"physical_collision_true_negative_count@{suffix}"] for suffix in suffixes),
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
        additive[f"physical_rollout_position_coverage90@{suffix}_hit_count"] = 85.0 / 3.0
        additive[f"physical_rollout_position_coverage90@{suffix}_coordinate_count"] = 100.0 / 3.0
        additive[f"physical_forecast_tracked_count@{suffix}"] = 10.0
        additive[f"physical_forecast_predictable_target_count@{suffix}"] = 10.0
        additive[f"physical_rollout_predictable_target_count@{suffix}"] = 1.0
        additive[f"physical_rollout_censored_external_actuation_count@{suffix}"] = 0.0
    _complete_additive_physical_evidence(additive)

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
    assert metrics["validation_current_position_coverage90"] == pytest.approx(1.0)
    assert metrics["validation_current_position_gaussian_nll"] == pytest.approx(0.1)
    assert metrics["validation_current_position_sharpness_std"] == pytest.approx(1.0)
    for axis in ("x", "y", "z"):
        assert metrics[f"validation_velocity_rmse_{axis}_mps"] == pytest.approx(3.0)
        assert metrics[f"validation_current_position_gaussian_nll_{axis}"] == pytest.approx(0.1)
        assert metrics[f"validation_current_position_sharpness_std_{axis}"] == pytest.approx(1.0)
    for suffix, expected_rmse in (
        ("0.100s", 1.0),
        ("0.250s", 2.0),
        ("0.500s", 3.0),
    ):
        assert metrics[f"validation_velocity_rmse@{suffix}"] == pytest.approx(expected_rmse)
        assert metrics[f"validation_collision_f1@{suffix}"] == pytest.approx(2.0 / 3.0)
        assert metrics[f"validation_forecast_identity_association_coverage@{suffix}"] == 1.0
        assert metrics[f"validation_forecast_identity_mismatch_rate@{suffix}"] == 0.0
        assert metrics[f"validation_position_coverage90@{suffix}"] == pytest.approx(0.85)
        assert metrics[f"validation_position_gaussian_nll@{suffix}"] == pytest.approx(0.1)
        assert metrics[f"validation_position_sharpness_std@{suffix}"] == pytest.approx(1.0)
        for axis in ("x", "y", "z"):
            assert metrics[f"validation_velocity_rmse_{axis}@{suffix}"] == pytest.approx(
                expected_rmse
            )
            assert metrics[f"validation_position_gaussian_nll_{axis}@{suffix}"] == pytest.approx(
                0.1
            )


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
        "validation_current_position_coverage90": position_coverage90,
        "validation_current_position_gaussian_nll": 0.1,
        "validation_current_position_sharpness_std": 1.0,
        "validation_position_rmse@0.100s": horizons[0],
        "validation_position_rmse@0.250s": horizons[1],
        "validation_position_rmse@0.500s": horizons[2],
        "validation_forecast_target_coverage@0.100s": forecast_coverage[0],
        "validation_forecast_target_coverage@0.250s": forecast_coverage[1],
        "validation_forecast_target_coverage@0.500s": forecast_coverage[2],
    }
    for suffix in ("0.100s", "0.250s", "0.500s"):
        metrics[f"validation_velocity_rmse@{suffix}"] = velocity
        metrics[f"validation_collision_f1@{suffix}"] = collision_f1
        metrics[f"validation_forecast_identity_association_coverage@{suffix}"] = 1.0
        metrics[f"validation_forecast_identity_mismatch_rate@{suffix}"] = id_switch_rate
        metrics[f"validation_position_coverage90@{suffix}"] = position_coverage90
        metrics[f"validation_position_gaussian_nll@{suffix}"] = 0.1
        metrics[f"validation_position_sharpness_std@{suffix}"] = 1.0
    for axis in ("x", "y", "z"):
        metrics[f"validation_position_rmse_{axis}_m"] = position
        metrics[f"validation_velocity_rmse_{axis}_mps"] = velocity
        metrics[f"validation_current_position_gaussian_nll_{axis}"] = 0.1
        metrics[f"validation_current_position_sharpness_std_{axis}"] = 1.0
        metrics[f"validation_position_rmse_{axis}@0.100s"] = horizons[0]
        metrics[f"validation_position_rmse_{axis}@0.250s"] = horizons[1]
        metrics[f"validation_position_rmse_{axis}@0.500s"] = horizons[2]
        for suffix in ("0.100s", "0.250s", "0.500s"):
            metrics[f"validation_velocity_rmse_{axis}@{suffix}"] = velocity
            metrics[f"validation_position_gaussian_nll_{axis}@{suffix}"] = 0.1
            metrics[f"validation_position_sharpness_std_{axis}@{suffix}"] = 1.0
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


@pytest.mark.parametrize(
    ("metric_name", "regressed_value", "expected_failure"),
    [
        ("validation_velocity_rmse_x@0.500s", 0.817, "velocity_rmse_x@0.500s"),
        ("validation_collision_f1@0.500s", 0.58, "collision_f1@0.500s"),
        (
            "validation_forecast_identity_mismatch_rate@0.500s",
            0.016,
            "forecast_identity_mismatch_rate@0.500s",
        ),
        (
            "validation_position_coverage90@0.500s",
            0.879,
            "position_calibration_error90@0.500s",
        ),
        (
            "validation_position_gaussian_nll_x@0.500s",
            0.121,
            "position_gaussian_nll_x@0.500s",
        ),
    ],
)
def test_rollout_selection_rejects_one_regressed_horizon_cell(
    metric_name: str,
    regressed_value: float,
    expected_failure: str,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    reference = _rollout_selection_metrics(_physical_selection_metrics(), config)
    candidate_metrics = _physical_selection_metrics(horizons=(0.30, 0.20, 0.10))
    candidate_metrics[metric_name] = regressed_value
    candidate = _rollout_selection_metrics(candidate_metrics, config)

    assert candidate.score < reference.score
    failures = _rollout_selection_guardrail_failures(candidate, reference)
    assert expected_failure in {str(failure["metric"]) for failure in failures}
    assert not _rollout_selection_improves(candidate, reference)


def test_rollout_selection_recurses_new_velocity_guardrail_through_scenarios() -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        simulator=replace(
            source.simulator,
            scenario_mixture=("baseline", "elastic_pairs"),
        ),
    )
    reference_metrics = _physical_selection_metrics()
    pooled_candidate = _physical_selection_metrics(horizons=(0.30, 0.20, 0.10))
    regressed_scenario = _physical_selection_metrics()
    regressed_scenario["validation_velocity_rmse_x@0.500s"] = 0.817
    reference = _rollout_selection_metrics(
        _with_scenario_selection_metrics(reference_metrics, config),
        config,
        require_scenarios=True,
    )
    candidate = _rollout_selection_metrics(
        _with_scenario_selection_metrics(
            pooled_candidate,
            config,
            scenario_metrics={"elastic_pairs": regressed_scenario},
        ),
        config,
        require_scenarios=True,
    )

    failures = _rollout_selection_guardrail_failures(candidate, reference)
    assert "scenario_elastic_pairs_velocity_rmse_x@0.500s" in {
        str(failure["metric"]) for failure in failures
    }
    assert not _rollout_selection_improves(candidate, reference)


def test_acceptable_coverage_cannot_hide_worse_gaussian_likelihood() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    reference = _rollout_selection_metrics(_physical_selection_metrics(), config)
    candidate_metrics = _physical_selection_metrics(
        position_coverage90=0.9,
        horizons=(0.30, 0.20, 0.10),
    )
    candidate_metrics["validation_position_gaussian_nll@0.500s"] = 0.121
    candidate = _rollout_selection_metrics(candidate_metrics, config)

    assert candidate.horizon_position_calibration_error90["0.500s"] == 0.0
    assert candidate.score < reference.score
    assert "position_gaussian_nll@0.500s" in {
        str(failure["metric"])
        for failure in _rollout_selection_guardrail_failures(candidate, reference)
    }
    assert not _rollout_selection_improves(candidate, reference)


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

    missing_horizon_axis = _physical_selection_metrics()
    del missing_horizon_axis["validation_velocity_rmse_x@0.500s"]
    with pytest.raises(RuntimeError, match="validation_velocity_rmse_x@0.500s"):
        _rollout_selection_metrics(missing_horizon_axis, config)


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
        "physical_position_coverage90_hit_count": 9.0,
        "physical_position_coverage90_coordinate_count": 9.0,
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
    _complete_additive_physical_evidence(physical_metrics)
    return TrainingBatchResult(
        total_loss=scalar,
        loss_terms={"rollout": scalar},
        metrics={"value": value, **physical_metrics},
        phase="closed_loop_rgb",
    )


_FIXED32_HORIZONS = ("0.100s", "0.250s", "0.500s", "0.750s", "1.000s")


def _fixed32_support_config():
    source = load_config("configs/tiny_all_scenarios.yaml")
    config = replace(
        source,
        training=replace(
            source.training,
            validation_episodes=32,
            validation_minimum_predictable_target_count_per_scenario_horizon=4,
            validation_minimum_matched_target_count_per_scenario_horizon=2,
            validation_minimum_supported_episodes_per_scenario=2,
        ),
    )
    config.validate()
    return config


def _recompute_pooled_event_counts(metrics: dict[str, float]) -> None:
    for kind in ("true_positive", "false_positive", "false_negative", "true_negative"):
        metrics[f"physical_collision_{kind}_count"] = sum(
            metrics[f"physical_collision_{kind}_count@{suffix}"] for suffix in _FIXED32_HORIZONS
        )


def _fixed32_rich_episode_result(seed: int) -> TrainingBatchResult:
    result = _result(1.0)
    metrics = dict(result.metrics)
    for suffix in ("0.750s", "1.000s"):
        metrics.update(
            {
                name.replace("@0.500s", f"@{suffix}"): value
                for name, value in tuple(metrics.items())
                if "@0.500s" in name
            }
        )
    metrics["physical_position_coverage90_hit_count"] = sum(
        metrics[f"physical_rollout_position_coverage90@{suffix}_hit_count"]
        for suffix in _FIXED32_HORIZONS
    )
    metrics["physical_position_coverage90_coordinate_count"] = sum(
        metrics[f"physical_rollout_position_coverage90@{suffix}_coordinate_count"]
        for suffix in _FIXED32_HORIZONS
    )
    for suffix in _FIXED32_HORIZONS:
        metrics[f"physical_collision_true_positive_count@{suffix}"] = 1.0
        metrics[f"physical_collision_false_positive_count@{suffix}"] = 0.0
        metrics[f"physical_collision_false_negative_count@{suffix}"] = 0.0
        metrics[f"physical_collision_true_negative_count@{suffix}"] = 1.0

    # Reproduce the real elastic fixed-manifest shape: every episode lacks a
    # positive collision label at at least one horizon, while the four pooled
    # episodes jointly contain both event classes at every horizon.
    elastic_missing_horizon = {
        100002: "0.250s",
        100010: "0.500s",
        100018: "0.100s",
        100026: "0.750s",
    }.get(seed)
    if elastic_missing_horizon is not None:
        metrics[f"physical_collision_true_positive_count@{elastic_missing_horizon}"] = 0.0
        metrics[f"physical_collision_false_negative_count@{elastic_missing_horizon}"] = 0.0
    _recompute_pooled_event_counts(metrics)
    return replace(result, metrics=metrics)


def test_fixed32_core_episode_support_retains_all_rich_scenario_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixed32_support_config()
    scenarios = tuple(config.simulator.scenario_mixture)

    def fake_validation(
        model: object,
        batch: dict[str, object],
        validation_config: object,
        *,
        closed_loop: bool,
    ) -> TrainingBatchResult:
        del model
        assert validation_config is config
        assert closed_loop
        seed = int(batch["seed"].item())  # type: ignore[union-attr]
        return _fixed32_rich_episode_result(seed)

    monkeypatch.setattr(
        "world_model.training.trainer._validation_step",
        fake_validation,
    )
    loader = [
        {
            "rgb": torch.ones((1, 3, 3, 8, 8)),
            "timestamps": torch.zeros((1, 3)),
            "seed": torch.tensor([seed]),
            "metadata": {"scenario": [scenarios[seed % len(scenarios)]]},
        }
        for seed in range(100000, 100032)
    ]

    result = _validation_loader_result(
        _ModeOnlyModel(),  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        config,
        device=torch.device("cpu"),
        closed_loop=True,
    )

    assert (
        sum(
            result.metrics[f"seed_{seed}_selection_metric_supported"]
            for seed in range(100000, 100032)
        )
        == 32.0
    )
    for scenario in _selection_scenario_slugs(config):
        assert result.metrics[f"scenario_{scenario}_supported_episode_count"] == 4.0
        assert result.metrics[f"scenario_{scenario}_selection_metric_supported"] == 1.0
        assert f"scenario_{scenario}_validation_collision_f1@1.000s" in result.metrics
        assert (
            f"scenario_{scenario}_validation_forecast_identity_mismatch_rate@1.000s"
            in result.metrics
        )
    elastic_seeds = (100002, 100010, 100018, 100026)
    assert all(
        any(
            episode.metrics[f"physical_collision_true_positive_count@{suffix}"]
            + episode.metrics[f"physical_collision_false_negative_count@{suffix}"]
            == 0.0
            for suffix in _FIXED32_HORIZONS
        )
        for episode in (_fixed32_rich_episode_result(seed) for seed in elastic_seeds)
    )
    assert _validate_validation_support_schema(result.metrics, config) == 1.0
    selector = _rollout_selection_metrics(result.metrics, config, require_scenarios=True)
    assert len(selector.scenario_slices) == 8
    assert all(selection is not None for selection in selector.scenario_slices.values())


@pytest.mark.parametrize("missing_support", ("event_positive_class", "forecast_identity"))
def test_scenario_rich_support_remains_fail_closed_after_core_episode_support(
    missing_support: str,
) -> None:
    config = _fixed32_support_config()
    results = [_fixed32_rich_episode_result(seed) for seed in (100002, 100010, 100018, 100026)]
    for result in results:
        if missing_support == "event_positive_class":
            result.metrics["physical_collision_true_positive_count@0.100s"] = 0.0
            result.metrics["physical_collision_false_negative_count@0.100s"] = 0.0
            _recompute_pooled_event_counts(result.metrics)
        else:
            result.metrics["physical_forecast_identity_association_count@0.100s"] = 0.0
            result.metrics["physical_forecast_identity_mismatch_count@0.100s"] = 0.0
        assert _core_causal_trajectory_episode_supported(result, config)

    aggregate = _aggregate_physical_validation_metrics(
        results,
        config,
        minimum_predictable_target_count=(
            config.training.validation_minimum_predictable_target_count_per_scenario_horizon
        ),
        minimum_matched_target_count=(
            config.training.validation_minimum_matched_target_count_per_scenario_horizon
        ),
    )

    assert aggregate["selection_metric_supported"] == 0.0
    assert not any(name.startswith("validation_") for name in aggregate)


def test_core_episode_support_rejects_missing_rich_additive_schema() -> None:
    config = _fixed32_support_config()
    result = _fixed32_rich_episode_result(100002)
    del result.metrics["physical_forecast_identity_eligible_count@0.100s"]

    with pytest.raises(RuntimeError, match="physical_forecast_identity_eligible_count@0.100s"):
        _core_causal_trajectory_episode_supported(result, config)


@pytest.mark.parametrize(
    "metric_name",
    [
        "physical_state_velocity_x_sse",
        "physical_rollout_velocity_x@0.500s_sse",
        "physical_rollout_position_x@0.500s_gaussian_nll_sum",
        "physical_rollout_position_x@0.500s_calibration_coordinate_count",
    ],
)
def test_additive_physical_axis_partitions_must_match_pooled_evidence(
    metric_name: str,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    evidence = dict(_result(1.0).metrics)
    evidence[metric_name] += 1.0

    with pytest.raises(ValueError, match="does not equal its x/y/z"):
        physical_validation_metrics(evidence, config)


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
            validation_rollout_anchor_batch_size=4,
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
    assert observed["validation_rollout_anchor_batch_size"] == 4
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


def test_training_update_progress_atomically_records_stage_and_known_timings(
    tmp_path,
) -> None:
    progress_path = tmp_path / "training_progress.json"

    _write_training_update_progress(
        progress_path,
        stage="backward",
        completed_updates=7,
        target_updates=80,
        attempted_update=8,
        data_draw_step=9,
        elapsed_seconds=123.0,
        phase="closed_loop_rgb",
        active_scope="state_roi",
        no_gradient_attempt=1,
        stage_seconds={"data": 2.0, "forward": 11.5},
        last_completed_stage_seconds={
            "data": 1.0,
            "forward": 10.0,
            "backward": 8.0,
            "optimizer": 0.5,
        },
        last_completed_update_seconds=20.0,
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress == {
        "active_scope": "state_roi",
        "attempted_update": 8,
        "completed_updates": 7,
        "data_draw_step": 9,
        "data_seconds": 2.0,
        "elapsed_seconds": 123.0,
        "forward_seconds": 11.5,
        "last_completed_backward_seconds": 8.0,
        "last_completed_data_seconds": 1.0,
        "last_completed_forward_seconds": 10.0,
        "last_completed_optimizer_seconds": 0.5,
        "last_completed_update_seconds": 20.0,
        "no_gradient_attempt": 1,
        "phase": "closed_loop_rgb",
        "pid": progress["pid"],
        "progress_kind": "optimizer_update",
        "stage": "backward",
        "state": "training_running",
        "target_updates": 80,
        "updated_utc": progress["updated_utc"],
    }


def test_anchor_batching_keeps_episode_scenario_and_seed_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_batch_sizes: list[int] = []

    def fake_validation(
        model: object,
        batch: dict[str, torch.Tensor],
        config: object,
        *,
        closed_loop: bool,
    ) -> TrainingBatchResult:
        del model, config
        assert closed_loop
        seen_batch_sizes.append(int(batch["rgb"].shape[0]))
        return _result(1.0)

    monkeypatch.setattr(
        "world_model.training.trainer._validation_step",
        fake_validation,
    )
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            validation_rollout_anchor_batch_size=8,
        ),
    )
    loader = [
        {
            "rgb": torch.ones((1, 3, 3, 8, 8)),
            "timestamps": torch.zeros((1, 3)),
            "seed": torch.tensor([seed]),
            "metadata": {"scenario": ["baseline"]},
        }
        for seed in (100000, 100001)
    ]

    result = _validation_loader_result(
        _ModeOnlyModel(),  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        config,
        device=torch.device("cpu"),
        closed_loop=True,
    )

    assert seen_batch_sizes == [1, 1]
    assert result.metrics["validation_attribution_available"] == 1.0
    assert result.metrics["scenario_baseline_episode_count"] == 2.0
    assert result.metrics["seed_100000_selection_metric_supported"] == 1.0
    assert result.metrics["seed_100001_selection_metric_supported"] == 1.0


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
