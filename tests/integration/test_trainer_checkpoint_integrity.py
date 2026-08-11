from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.datasets import make_seed_manifest
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import save_checkpoint
from world_model.training.loop import TrainingBatchResult
from world_model.training.trainer import (
    _ROLLOUT_SELECTION_METRIC_VERSION,
    _current_model_state_hash,
    _expected_resume_checkpoint_devices,
    _fresh_causal_optimizer_state,
    _measurement_selection_metrics,
    _measurement_validation_protocol_hash,
    _model_state_hash,
    _preserve_resume_measurement_checkpoint,
    _preserve_resume_selector_checkpoint,
    _preserve_resume_validation_history,
    _resolve_run_directory,
    _resolve_training_devices,
    _rollout_selection_improves,
    _rollout_selection_metrics,
    _rollout_selection_passes_guardrails,
    _rollout_validation_checkpoint_metrics,
    _rollout_validation_protocol_hash,
    _selection_horizon_keys,
    _selection_scenario_slugs,
    _validate_exact_resume_source,
    _validation_protocol_checkpoint_metrics,
    _validation_support_evidence,
    _verified_measurement_checkpoint,
    _verified_selector_checkpoint,
    train_from_config,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.device import select_device


def test_checkpoint_support_evidence_keeps_per_seed_support_markers() -> None:
    evidence = _validation_support_evidence(
        {
            "seed_100004_selection_metric_supported": 1.0,
            "seed_100020_selection_metric_supported": 0.0,
            "seed_100004_validation_position_rmse_m": 0.2,
            "scenario_impulse_perturbation_episode_count": 4.0,
            "scenario_impulse_perturbation_selection_metric_supported": 1.0,
            "scenario_impulse_perturbation_supported_episode_count": 3.0,
            "physical_position_coverage90_hit_count": 8.0,
            "physical_position_coverage90_coordinate_count": 9.0,
            "physical_rollout_position_coverage90@1.000s_hit_count": 2.0,
            "physical_rollout_position_coverage90@1.000s_coordinate_count": 3.0,
        }
    )

    assert evidence == {
        "seed_100004_selection_metric_supported": 1.0,
        "seed_100020_selection_metric_supported": 0.0,
        "scenario_impulse_perturbation_episode_count": 4.0,
        "scenario_impulse_perturbation_selection_metric_supported": 1.0,
        "scenario_impulse_perturbation_supported_episode_count": 3.0,
        "physical_position_coverage90_hit_count": 8.0,
        "physical_position_coverage90_coordinate_count": 9.0,
        "physical_rollout_position_coverage90@1.000s_hit_count": 2.0,
        "physical_rollout_position_coverage90@1.000s_coordinate_count": 3.0,
    }


def test_rejected_equal_weight_candidate_still_reports_incumbent_tensor_linkage() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    selection = _rollout_selection_metrics(_physical_metrics(config), config)
    validation = TrainingBatchResult(
        total_loss=torch.tensor(0.4),
        loss_terms={"rollout": torch.tensor(0.3), "rollout_position": torch.tensor(0.2)},
        metrics={},
        phase="closed_loop_rgb",
    )

    metrics = _rollout_validation_checkpoint_metrics(
        validation,
        selection,
        selection,
        selection,
        config=config,
        accepted=False,
        training_support_required=True,
        training_support_failures=[],
        mutable_training_support_failures=[],
        best_measurement=None,
        checkpoint_model_state_hash="same-tensors",
        incumbent_model_state_hash="same-tensors",
        incumbent_step=4,
        reference_model_state_hash="reference-tensors",
        reference_step=0,
    )

    assert metrics["selection_accepted"] == 0.0
    assert metrics["checkpoint_contains_best_rollout_weights"] == 1.0
    assert metrics["checkpoint_contains_reference_rollout_weights"] == 0.0


def _physical_metrics(
    config: OrpheusConfig,
    *,
    score_scale: float = 1.0,
    velocity: float = 0.8,
    prediction_precision: float = 0.8,
    position_coverage90: float = 0.9,
    target_coverage: float = 0.9,
    forecast_coverage: float = 0.9,
) -> dict[str, float]:
    target_count = 100.0
    matched_count = target_coverage * target_count
    predicted_count = (
        matched_count / prediction_precision if prediction_precision > 0.0 else target_count
    )
    metrics = {
        "selection_metric_supported": 1.0,
        "validation_position_rmse_m": 0.4,
        "validation_velocity_rmse_mps": velocity,
        "validation_target_coverage": target_coverage,
        "validation_prediction_precision": prediction_precision,
        "validation_collision_f1": 0.6,
        "validation_id_switch_rate": 0.01,
        "validation_position_coverage90": position_coverage90,
        "physical_state_position_sse": 0.4**2 * 300.0,
        "physical_state_position_coordinate_count": 300.0,
        "physical_state_velocity_sse": velocity**2 * 300.0,
        "physical_state_velocity_coordinate_count": 300.0,
        "physical_distance_gated_matched_object_frames": matched_count,
        "physical_distance_gated_target_object_frames": target_count,
        "physical_distance_gated_predicted_object_frames": predicted_count,
        "physical_distance_gated_identity_switches": 1.0,
        "physical_distance_gated_object_frame_associations": 100.0,
        "physical_position_coverage90_hit_count": position_coverage90 * 300.0,
        "physical_position_coverage90_coordinate_count": 300.0,
        "physical_collision_true_positive_count": 3.0,
        "physical_collision_false_positive_count": 2.0,
        "physical_collision_false_negative_count": 2.0,
    }
    for axis in ("x", "y", "z"):
        metrics[f"validation_position_rmse_{axis}_m"] = 0.4
        metrics[f"physical_state_position_{axis}_sse"] = 0.4**2 * 100.0
        metrics[f"physical_state_position_{axis}_coordinate_count"] = 100.0
    for index, (suffix, _) in enumerate(_selection_horizon_keys(config)):
        horizon_rmse = score_scale * (0.4 - index * 0.1)
        metrics[f"validation_position_rmse@{suffix}"] = horizon_rmse
        metrics[f"validation_forecast_target_coverage@{suffix}"] = forecast_coverage
        metrics[f"physical_rollout_position@{suffix}_sse"] = horizon_rmse**2 * 30.0
        metrics[f"physical_forecast_predictable_target_count@{suffix}"] = 10.0
        metrics[f"physical_rollout_position@{suffix}_coordinate_count"] = 30.0
        metrics[f"physical_rollout_position_coverage90@{suffix}_hit_count"] = (
            position_coverage90 * 30.0
        )
        metrics[f"physical_rollout_position_coverage90@{suffix}_coordinate_count"] = 30.0
        metrics[f"physical_forecast_active_count@{suffix}"] = forecast_coverage * 10.0
        metrics[f"physical_forecast_tracked_count@{suffix}"] = 10.0
        metrics[f"physical_forecast_target_count@{suffix}"] = 10.0
        metrics[f"physical_rollout_predictable_target_count@{suffix}"] = 10.0
        metrics[f"physical_rollout_censored_external_actuation_count@{suffix}"] = 0.0
        for axis in ("x", "y", "z"):
            metrics[f"validation_position_rmse_{axis}@{suffix}"] = horizon_rmse
            metrics[f"physical_rollout_position_{axis}@{suffix}_sse"] = horizon_rmse**2 * 10.0
            metrics[f"physical_rollout_position_{axis}@{suffix}_coordinate_count"] = 10.0
    base = dict(metrics)
    manifest = make_seed_manifest("validation", config.training.validation_episodes)
    scenario_slugs = _selection_scenario_slugs(config)
    scenario_episode_counts = {scenario: 0 for scenario in scenario_slugs}
    for seed in manifest.seeds:
        scenario_episode_counts[scenario_slugs[int(seed) % len(scenario_slugs)]] += 1
    for scenario in scenario_slugs:
        prefix = f"scenario_{scenario}_"
        episode_count = float(scenario_episode_counts[scenario])
        metrics[f"{prefix}episode_count"] = episode_count
        metrics[f"{prefix}supported_episode_count"] = episode_count
        metrics[f"{prefix}minimum_supported_episode_count"] = float(
            config.training.validation_minimum_supported_episodes_per_scenario
        )
        metrics[f"{prefix}selection_metric_supported"] = 1.0
        metrics.update({f"{prefix}{name}": value for name, value in base.items()})
    for seed in manifest.seeds:
        metrics[f"seed_{seed}_selection_metric_supported"] = 1.0
    return metrics


def _selector_checkpoint_metrics(
    model: OnlineWorldModel,
    config: OrpheusConfig,
    *,
    step: int,
) -> dict[str, object]:
    selection = _rollout_selection_metrics(
        _physical_metrics(config),
        config,
        require_scenarios=True,
    )
    model_hash = _current_model_state_hash(model)
    return {
        **_physical_metrics(config),
        "selection_accepted": 1.0,
        "best_rollout_validated": 1.0,
        "rollout_reference_validated": 1.0,
        "incomplete_reference_comparison_required": 0.0,
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        **selection.validation_metrics(),
        **selection.checkpoint_metrics(),
        **selection.checkpoint_metrics(prefix="reference_rollout"),
        **_validation_protocol_checkpoint_metrics(config),
        "checkpoint_model_state_hash": model_hash,
        "checkpoint_contains_best_rollout_weights": 1.0,
        "best_rollout_model_state_hash": model_hash,
        "best_rollout_checkpoint_step": float(step),
        "checkpoint_contains_reference_rollout_weights": 1.0,
        "reference_rollout_model_state_hash": model_hash,
        "reference_rollout_checkpoint_step": float(step),
    }


def test_branched_resume_preserves_only_verified_accepted_numbered_history(
    tmp_path,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    source = tmp_path / "source" / "checkpoints"
    destination = tmp_path / "branch" / "checkpoints"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)

    accepted_metrics = _selector_checkpoint_metrics(model, config, step=3)
    save_checkpoint(
        source / "validation_step_000003.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=3,
        metrics=accepted_metrics,
        device="cpu",
    )
    corrupted_support_metrics = _selector_checkpoint_metrics(model, config, step=4)
    first_validation_seed = make_seed_manifest(
        "validation",
        config.training.validation_episodes,
    ).seeds[0]
    del corrupted_support_metrics[f"seed_{first_validation_seed}_selection_metric_supported"]
    save_checkpoint(
        source / "validation_step_000004.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=4,
        metrics=corrupted_support_metrics,
        device="cpu",
    )
    rejected_metrics = {
        **_selector_checkpoint_metrics(model, config, step=5),
        "selection_accepted": 0.0,
        "checkpoint_contains_best_rollout_weights": 0.0,
    }
    save_checkpoint(
        source / "validation_step_000005.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=5,
        metrics=rejected_metrics,
        device="cpu",
    )
    save_checkpoint(
        source / "validation_step_000009.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=9,
        metrics=_selector_checkpoint_metrics(model, config, step=9),
        device="cpu",
    )
    resume = save_checkpoint(
        source / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=6,
        metrics={},
        device="cpu",
    )

    copied = _preserve_resume_validation_history(
        resume,
        destination,
        config,
        resume_step=6,
        expected_device=torch.device("cpu"),
    )

    assert [path.name for path in copied] == ["validation_step_000003.pt"]
    assert (destination / "validation_step_000003.pt").read_bytes() == (
        source / "validation_step_000003.pt"
    ).read_bytes()
    assert not (destination / "validation_step_000005.pt").exists()
    assert not (destination / "validation_step_000004.pt").exists()
    assert not (destination / "validation_step_000009.pt").exists()


def test_validation_protocol_allows_only_training_step_extension() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    extended = replace(
        config,
        training=replace(config.training, steps=config.training.steps + 100),
    )
    changed_batch = replace(
        config,
        training=replace(config.training, batch_size=config.training.batch_size + 1),
    )
    changed_physics = replace(
        config,
        simulator=replace(config.simulator, mass_range=(0.7, 1.7)),
    )
    changed_evaluation = replace(
        config,
        evaluation=replace(config.evaluation, confidence_level=0.8),
    )
    changed_validation_device = replace(
        config,
        device=replace(config.device, closed_loop_preference="cpu"),
    )
    changed_measurement_device = replace(
        config,
        device=replace(config.device, preference="mps"),
    )
    changed_detector_execution = replace(
        config,
        device=replace(
            config.device,
            global_detector_cpu_on_mps=not config.device.global_detector_cpu_on_mps,
        ),
    )
    changed_handoff_support = replace(
        config,
        training=replace(
            config.training,
            handoff_minimum_target_coverage=(
                config.training.handoff_minimum_target_coverage + 0.01
            ),
        ),
    )
    changed_gradient_stability = replace(
        config,
        training=replace(
            config.training,
            grad_clip_norm=config.training.grad_clip_norm * 2.0,
        ),
    )
    changed_node_gradient_stability = replace(
        config,
        training=replace(
            config.training,
            attention_node_grad_clip_norm=1.0,
        ),
    )
    changed_perception_gradient_stability = replace(
        config,
        training=replace(
            config.training,
            closed_loop_perception_grad_clip_norm=(
                config.training.closed_loop_perception_grad_clip_norm * 2.0
            ),
        ),
    )
    changed_retry_bound = replace(
        config,
        training=replace(
            config.training,
            maximum_no_gradient_batches_per_update=(
                config.training.maximum_no_gradient_batches_per_update + 1
            ),
        ),
    )

    assert _rollout_validation_protocol_hash(extended) == (
        _rollout_validation_protocol_hash(config)
    )
    # Validation is now always batch-one, so the training batch size cannot
    # change its protocol or make otherwise comparable selectors incompatible.
    assert _rollout_validation_protocol_hash(changed_batch) == (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_physics) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_evaluation) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_validation_device) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_measurement_device) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_detector_execution) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_handoff_support) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_gradient_stability) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_node_gradient_stability) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_perception_gradient_stability) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_retry_bound) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _measurement_validation_protocol_hash(changed_detector_execution) != (
        _measurement_validation_protocol_hash(config)
    )


def test_selector_guards_precision_calibration_and_fixed_reference() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    reference = _rollout_selection_metrics(_physical_metrics(config), config)
    moving = _rollout_selection_metrics(
        _physical_metrics(config, score_scale=0.95, velocity=0.815),
        config,
    )
    ratcheted = _rollout_selection_metrics(
        _physical_metrics(config, score_scale=0.90, velocity=0.831),
        config,
    )
    low_precision = _rollout_selection_metrics(
        _physical_metrics(config, score_scale=0.90, prediction_precision=0.78),
        config,
    )
    miscalibrated = _rollout_selection_metrics(
        _physical_metrics(config, score_scale=0.90, position_coverage90=0.879),
        config,
    )
    low_forecast_coverage = _rollout_selection_metrics(
        _physical_metrics(config, score_scale=0.90, forecast_coverage=0.894),
        config,
    )

    assert _rollout_selection_improves(ratcheted, moving)
    assert not _rollout_selection_passes_guardrails(ratcheted, reference)
    assert not _rollout_selection_improves(low_precision, reference)
    assert not _rollout_selection_improves(miscalibrated, reference)
    assert not _rollout_selection_improves(low_forecast_coverage, reference)


def test_resume_preserves_only_a_real_linked_incumbent(tmp_path) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    source_checkpoints = tmp_path / "source" / "checkpoints"
    source_checkpoints.mkdir(parents=True)
    best_path = source_checkpoints / "best_rollout.pt"
    metrics = _selector_checkpoint_metrics(model, config, step=7)
    save_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        config=config,
        step=7,
        metrics=metrics,  # type: ignore[arg-type]
    )
    resume_path = source_checkpoints / "last.pt"
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        config=config,
        step=8,
        metrics=metrics,  # type: ignore[arg-type]
    )

    destination = tmp_path / "new-run" / "checkpoints" / "best_rollout.pt"
    destination.parent.mkdir(parents=True)
    preserved = _preserve_resume_selector_checkpoint(
        resume_path,
        destination,
        config,
        prefix="best_rollout",
        resume_metrics=metrics,
    )

    assert preserved is not None
    assert destination.is_file()
    assert _verified_selector_checkpoint(
        destination,
        config,
        prefix="best_rollout",
        expected_model_state_hash=str(metrics["best_rollout_model_state_hash"]),
        expected_step=7,
    )
    assert (
        _verified_selector_checkpoint(
            destination,
            config,
            prefix="best_rollout",
            expected_model_state_hash=str(metrics["best_rollout_model_state_hash"]),
            expected_step=7,
            expected_device="mps",
        )
        is None
    )
    wrong_link = dict(metrics)
    wrong_link["best_rollout_model_state_hash"] = "0" * 64
    assert (
        _preserve_resume_selector_checkpoint(
            resume_path,
            tmp_path / "rejected.pt",
            config,
            prefix="best_rollout",
            resume_metrics=wrong_link,
        )
        is None
    )


def test_selector_verification_requires_axis_metadata_and_raw_consistency(
    tmp_path,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    checkpoint = tmp_path / "best_rollout.pt"
    metrics = _selector_checkpoint_metrics(model, config, step=7)
    model_hash = str(metrics["best_rollout_model_state_hash"])
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        step=7,
        metrics=metrics,  # type: ignore[arg-type]
    )

    assert (
        _verified_selector_checkpoint(
            checkpoint,
            config,
            prefix="best_rollout",
            expected_model_state_hash=model_hash,
            expected_step=7,
        )
        is not None
    )

    missing_reference_state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    del missing_reference_state["metrics"]["incomplete_reference_comparison_required"]
    torch.save(missing_reference_state, checkpoint)
    assert (
        _verified_selector_checkpoint(
            checkpoint,
            config,
            prefix="best_rollout",
            expected_model_state_hash=model_hash,
            expected_step=7,
        )
        is None
    )
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        step=7,
        metrics=metrics,  # type: ignore[arg-type]
    )

    missing_axes = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for axis in ("x", "y", "z"):
        del missing_axes["metrics"][f"best_rollout_position_rmse_{axis}_m"]
        for suffix, _ in _selection_horizon_keys(config):
            del missing_axes["metrics"][f"best_rollout_position_rmse_{axis}@{suffix}"]
    torch.save(missing_axes, checkpoint)
    assert (
        _verified_selector_checkpoint(
            checkpoint,
            config,
            prefix="best_rollout",
            expected_model_state_hash=model_hash,
            expected_step=7,
        )
        is None
    )

    contradictory = dict(metrics)
    contradictory["best_rollout_position_rmse_m"] = 0.2
    for axis in ("x", "y", "z"):
        contradictory[f"best_rollout_position_rmse_{axis}_m"] = 0.2
    weighted_score = 0.0
    total_weight = 0.0
    for suffix, weight in _selection_horizon_keys(config):
        value = float(contradictory[f"best_rollout_position_rmse@{suffix}"]) * 0.5
        contradictory[f"best_rollout_position_rmse@{suffix}"] = value
        for axis in ("x", "y", "z"):
            contradictory[f"best_rollout_position_rmse_{axis}@{suffix}"] = value
        weighted_score += value * weight
        total_weight += weight
    contradictory["best_rollout_selection_score"] = weighted_score / total_weight
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        step=7,
        metrics=contradictory,  # type: ignore[arg-type]
    )
    assert (
        _verified_selector_checkpoint(
            checkpoint,
            config,
            prefix="best_rollout",
            expected_model_state_hash=model_hash,
            expected_step=7,
        )
        is None
    )


def test_causal_phase_clears_adam_moments_and_sets_phase_hyperparameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3, weight_decay=0.2)
    parameter.grad = torch.tensor(2.0)
    optimizer.step()
    assert optimizer.state

    _fresh_causal_optimizer_state(
        optimizer,
        learning_rate=5.0e-5,
        weight_decay=1.0e-4,
    )

    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == 5.0e-5
    assert optimizer.param_groups[0]["weight_decay"] == 1.0e-4


def test_causal_only_plan_does_not_resolve_unused_mps(
    monkeypatch,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        device=replace(
            config.device,
            preference="mps",
            closed_loop_preference="cpu",
        ),
        training=replace(
            config.training,
            steps=2,
            rgb_pretrain_steps=0,
        ),
    )

    def guarded_select_device(preference: str):
        if preference == "mps":
            raise AssertionError("unused measurement MPS must not be resolved")
        return select_device(preference)

    monkeypatch.setattr(
        "world_model.training.trainer.select_device",
        guarded_select_device,
    )
    info, measurement, closed_loop, active = _resolve_training_devices(
        config,
        start_step=0,
        initialize_from=False,
    )

    assert info.device.type == "cpu"
    assert measurement.type == "mps"
    assert closed_loop.type == "cpu"
    assert active.type == "cpu"


def test_exact_resume_rejects_changed_source_fingerprint() -> None:
    payload = {
        "git": {
            "commit": "abc123",
            "dirty": True,
            "worktree_fingerprint": "old",
        }
    }

    with pytest.raises(ValueError, match="source worktree differs"):
        _validate_exact_resume_source(
            payload,
            {
                "commit": "abc123",
                "dirty": True,
                "worktree_fingerprint": "new",
            },
        )


def test_exact_resume_uses_runtime_source_before_whole_worktree() -> None:
    payload = {
        "git": {
            "commit": "old-docs-commit",
            "dirty": False,
            "worktree_fingerprint": "old-whole-tree",
            "runtime_source_fingerprint": "same-runtime",
        }
    }

    _validate_exact_resume_source(
        payload,
        {
            "commit": "new-docs-commit",
            "dirty": True,
            "worktree_fingerprint": "new-whole-tree",
            "runtime_source_fingerprint": "same-runtime",
        },
    )
    with pytest.raises(ValueError, match="executable source differs"):
        _validate_exact_resume_source(
            payload,
            {
                "commit": "new-code-commit",
                "dirty": False,
                "worktree_fingerprint": "new-whole-tree",
                "runtime_source_fingerprint": "changed-runtime",
            },
        )


def test_boundary_checkpoint_device_uses_handoff_marker_with_legacy_fallback() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        device=replace(
            config.device,
            preference="mps",
            closed_loop_preference="cpu",
        ),
        training=replace(
            config.training,
            steps=8,
            rgb_pretrain_steps=4,
        ),
    )
    measurement = torch.device("mps")
    closed_loop = torch.device("cpu")

    def expected(metrics: dict[str, object]) -> frozenset[torch.device]:
        return _expected_resume_checkpoint_devices(
            {"step": 4, "metrics": metrics},
            config,
            measurement_device=measurement,
            closed_loop_device=closed_loop,
        )

    assert expected({"measurement_handoff_completed": 0.0}) == frozenset({measurement})
    assert expected({"measurement_handoff_completed": 1.0}) == frozenset({closed_loop})
    assert expected({}) == frozenset({measurement, closed_loop})
    assert expected(
        {
            "measurement_handoff_completed": 0.0,
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            "validation_rollout_selection_score": 0.5,
        }
    ) == frozenset({closed_loop})


def test_resume_copies_only_tensor_verified_measurement_selector(tmp_path) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    selection = _measurement_selection_metrics(
        {
            "rgb_runtime_birth_world_position_mae_m": 0.2,
            "rgb_world_position_mae_m": 0.25,
            "rgb_runtime_birth_recall_at_0_5m": 0.8,
            "rgb_runtime_birth_precision_at_0_5m": 0.75,
            "rgb_runtime_birth_f1_at_0_5m": 0.774,
            "rgb_fast_bootstrap_target_coverage": 0.75,
            "rgb_fast_roi_target_coverage": 0.70,
            "rgb_fast_roi_world_position_mae_m": 0.20,
            "rgb_fast_roi_recall_at_0_5m": 0.65,
            "rgb_fast_roi_precision_at_0_5m": 0.75,
            "rgb_fast_roi_f1_at_0_5m": 0.696,
            "rgb_fast_roi_improvement_m": 0.05,
        }
    )
    assert selection is not None
    model_hash = _current_model_state_hash(model)
    source_checkpoints = tmp_path / "source" / "checkpoints"
    source_checkpoints.mkdir(parents=True)
    metrics = {
        "best_measurement_validated": 1.0,
        **selection.checkpoint_metrics(),
        "measurement_validation_protocol_hash": (_measurement_validation_protocol_hash(config)),
        "checkpoint_model_state_hash": model_hash,
        "checkpoint_contains_best_measurement_weights": 1.0,
        "best_measurement_model_state_hash": model_hash,
        "best_measurement_checkpoint_step": 7.0,
    }
    best_path = save_checkpoint(
        source_checkpoints / "best_measurement.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=7,
        metrics=metrics,
        device="cpu",
    )
    resume_path = save_checkpoint(
        source_checkpoints / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=8,
        metrics=metrics,
        device="cpu",
    )
    destination = tmp_path / "new-run" / "checkpoints" / "best_measurement.pt"
    destination.parent.mkdir(parents=True)

    preserved = _preserve_resume_measurement_checkpoint(
        resume_path,
        destination,
        config,
        resume_metrics=metrics,
        expected_device="cpu",
    )

    assert preserved is not None
    assert destination.is_file()
    assert _verified_measurement_checkpoint(
        destination,
        config,
        expected_model_state_hash=model_hash,
        expected_step=7,
        expected_device="cpu",
    )
    tampered = torch.load(best_path, map_location="cpu", weights_only=False)
    tampered["metrics"]["best_measurement_model_state_hash"] = "0" * 64
    torch.save(tampered, best_path)
    assert (
        _preserve_resume_measurement_checkpoint(
            resume_path,
            tmp_path / "rejected.pt",
            config,
            resume_metrics=metrics,
            expected_device="cpu",
        )
        is None
    )


def test_exact_resume_fails_loudly_when_linked_measurement_artifact_is_missing(
    tmp_path,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            config.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            config.training,
            steps=1,
            rgb_pretrain_steps=0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    selection = _measurement_selection_metrics(
        {
            "rgb_runtime_birth_world_position_mae_m": 0.2,
            "rgb_world_position_mae_m": 0.25,
            "rgb_runtime_birth_recall_at_0_5m": 0.8,
            "rgb_runtime_birth_precision_at_0_5m": 0.75,
            "rgb_runtime_birth_f1_at_0_5m": 0.774,
            "rgb_fast_bootstrap_target_coverage": 0.75,
            "rgb_fast_roi_target_coverage": 0.70,
            "rgb_fast_roi_world_position_mae_m": 0.20,
            "rgb_fast_roi_recall_at_0_5m": 0.65,
            "rgb_fast_roi_precision_at_0_5m": 0.75,
            "rgb_fast_roi_f1_at_0_5m": 0.696,
            "rgb_fast_roi_improvement_m": 0.05,
        }
    )
    assert selection is not None
    model_hash = _current_model_state_hash(model)
    checkpoint = save_checkpoint(
        tmp_path / "missing-selector" / "checkpoints" / "last.pt",
        model=model,
        optimizer=None,
        config=config,
        step=1,
        metrics={
            "best_measurement_validated": 1.0,
            **selection.checkpoint_metrics(),
            "measurement_validation_protocol_hash": (_measurement_validation_protocol_hash(config)),
            "best_measurement_model_state_hash": model_hash,
            "best_measurement_checkpoint_step": 0.0,
            "measurement_handoff_completed": 1.0,
        },
        device="cpu",
    )

    with pytest.raises(ValueError, match="best_measurement.pt is missing"):
        train_from_config(config, resume_path=checkpoint)


def test_exact_resume_fails_loudly_when_linked_rollout_artifact_is_missing(
    tmp_path,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            config.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            config.training,
            steps=1,
            rgb_pretrain_steps=0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    metrics = _selector_checkpoint_metrics(model, config, step=0)
    metrics["rollout_selection_metric_version"] = _ROLLOUT_SELECTION_METRIC_VERSION
    metrics["measurement_handoff_completed"] = 1.0
    checkpoint = save_checkpoint(
        tmp_path / "missing-rollout-selectors" / "checkpoints" / "last.pt",
        model=model,
        optimizer=None,
        config=config,
        step=1,
        metrics=metrics,
        device="cpu",
    )

    with pytest.raises(ValueError, match="reference_rollout.pt is missing"):
        train_from_config(config, resume_path=checkpoint)


def test_in_place_resume_requires_the_exact_last_checkpoint(tmp_path) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
    )
    source_run = tmp_path / "source"
    last_path = source_run / "checkpoints" / "last.pt"

    assert (
        _resolve_run_directory(
            config,
            run_name=None,
            resume_path=last_path,
        )
        == source_run
    )
    for checkpoint_name in ("best_rollout.pt", "validation_step_000100.pt"):
        with pytest.raises(
            ValueError,
            match=r"in-place exact resume requires.*checkpoints/last\.pt",
        ):
            _resolve_run_directory(
                config,
                run_name=None,
                resume_path=source_run / "checkpoints" / checkpoint_name,
            )

    forked = _resolve_run_directory(
        config,
        run_name="selector-fork",
        resume_path=source_run / "checkpoints" / "best_rollout.pt",
    )
    assert forked.parent == (tmp_path / "runs")
    assert forked != source_run


@pytest.mark.parametrize(
    ("rgb_pretrain_steps", "expected_closed_loop", "expected_phase"),
    [
        (1, False, "rgb_pretrain"),
        (0, True, "closed_loop_rgb"),
    ],
)
def test_pending_final_validation_recovers_without_optimizer_update(
    tmp_path,
    monkeypatch,
    rgb_pretrain_steps: int,
    expected_closed_loop: bool,
    expected_phase: str,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            config.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            config.training,
            steps=1,
            rgb_pretrain_steps=rgb_pretrain_steps,
            train_episodes=1,
            validation_episodes=1,
            batch_size=1,
            eval_every=1,
        ),
    )
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
    )
    checkpoint = save_checkpoint(
        tmp_path / f"pending-{expected_phase}" / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={
            "loss_total": 2.5,
            "measurement_handoff_completed": float(rgb_pretrain_steps == 0),
            "final_validation_completed": 0.0,
        },
        device="cpu",
    )
    calls: list[bool] = []

    def fake_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        assert device == torch.device("cpu")
        calls.append(closed_loop)
        if closed_loop:
            return TrainingBatchResult(
                total_loss=torch.tensor(0.4),
                loss_terms={
                    "rollout": torch.tensor(0.3),
                    "rollout_position": torch.tensor(0.2),
                },
                metrics=_physical_metrics(config),
                phase="closed_loop_rgb",
            )
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"measurement": torch.tensor(0.3)},
            metrics={
                "rgb_runtime_birth_world_position_mae_m": 0.2,
                "rgb_world_position_mae_m": 0.25,
                "rgb_runtime_birth_recall_at_0_5m": 0.8,
                "rgb_runtime_birth_precision_at_0_5m": 0.75,
                "rgb_runtime_birth_f1_at_0_5m": 0.774,
                "rgb_fast_bootstrap_target_coverage": 0.75,
                "rgb_fast_roi_target_coverage": 0.70,
                "rgb_fast_roi_world_position_mae_m": 0.20,
                "rgb_fast_roi_recall_at_0_5m": 0.65,
                "rgb_fast_roi_precision_at_0_5m": 0.75,
                "rgb_fast_roi_f1_at_0_5m": 0.696,
                "rgb_fast_roi_improvement_m": 0.05,
            },
            phase="rgb_pretrain",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        fake_validation,
    )

    result = train_from_config(
        config,
        resume_path=checkpoint,
    )

    recovered = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert calls == [expected_closed_loop]
    assert recovered["step"] == 1
    assert recovered["metrics"]["final_validation_completed"] == 1.0
    assert recovered["metrics"]["final_validation_loss_total"] == pytest.approx(0.4)
    assert result["completed_steps"] == 1
    assert result["optimizer_updates_this_invocation"] == 0
    assert result["no_op_exact_resume"] is False
    assert result["final_validation_recovered"] is True
    assert result["last_metrics"]["loss_total"] == 2.5
    assert result["last_metrics"]["final_validation_phase"] == expected_phase


def test_global_only_causal_draw_does_not_consume_optimizer_step(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=1,
            rgb_pretrain_steps=0,
            train_episodes=3,
            validation_episodes=1,
            batch_size=1,
            eval_every=0,
            checkpoint_every=1,
            log_every=1,
            maximum_no_gradient_batches_per_update=2,
        ),
    )
    consumed_seeds: list[int] = []

    def fake_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        assert device == torch.device("cpu")
        assert closed_loop
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=_physical_metrics(config),
            phase="closed_loop_rgb",
        )

    def fake_closed_loop(
        model,
        batch,
        _config,
        **_kwargs,
    ) -> TrainingBatchResult:
        consumed_seeds.append(int(batch["seed"].flatten()[0]))
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        loss_terms = {"measurement": loss}
        if len(consumed_seeds) > 1:
            loss_terms["state"] = loss
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms=loss_terms,
            metrics={"matched_object_frames": float(len(consumed_seeds) > 1)},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        fake_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        fake_closed_loop,
    )

    result = train_from_config(config, run_name="zero-gradient-resample")

    assert len(consumed_seeds) == 2
    assert result["completed_steps"] == 1
    assert result["training_batch_draws_total"] == 2
    assert result["skipped_no_gradient_batches"] == 1
    payload = torch.load(
        result["last_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    assert payload["step"] == 1
    assert payload["metrics"]["training_data_draw_step"] == 2.0
    assert payload["metrics"]["skipped_no_gradient_batches"] == 1.0
    assert payload["metrics"]["final_validation_completed"] == 1.0
    records = [
        json.loads(line)
        for line in Path(result["metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    skipped = [record for record in records if record["split"] == "train_skipped_no_gradient"]
    assert len(skipped) == 1
    assert skipped[0]["optimizer_update_applied"] == 0.0
    assert skipped[0]["causal_training_support_present"] == 0.0
    assert skipped[0]["gradient_norm"] == 0.0


def test_trainer_rejects_nonfinite_parameters_immediately_after_optimizer_step(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=1,
            rgb_pretrain_steps=0,
            train_episodes=1,
            validation_episodes=1,
            batch_size=1,
            eval_every=0,
            checkpoint_every=1,
            log_every=1,
        ),
    )

    def fake_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        assert device == torch.device("cpu")
        assert closed_loop
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=_physical_metrics(config),
            phase="closed_loop_rgb",
        )

    def fake_closed_loop(
        model,
        _batch,
        _config,
        **_kwargs,
    ) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    original_step = torch.optim.AdamW.step

    def corrupting_step(optimizer, closure=None):
        result = original_step(optimizer, closure=closure)
        parameter = optimizer.param_groups[0]["params"][0]
        with torch.no_grad():
            parameter.flatten()[0] = float("nan")
        return result

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        fake_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        fake_closed_loop,
    )
    monkeypatch.setattr(torch.optim.AdamW, "step", corrupting_step)

    with pytest.raises(
        FloatingPointError,
        match=r"model_parameters.*NaN or Inf.*after optimiser step 0",
    ):
        train_from_config(config, run_name="post-step-nonfinite")

    run_directories = list((tmp_path / "runs").glob("*post-step-nonfinite*"))
    assert len(run_directories) == 1
    checkpoint_directory = run_directories[0] / "checkpoints"
    assert not (checkpoint_directory / "last.pt").exists()
    assert not (checkpoint_directory / "validation_step_000001.pt").exists()
    assert (checkpoint_directory / "validation_step_000000.pt").is_file()


def test_fresh_causal_run_records_but_does_not_promote_collapsed_incumbent(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=1,
            rgb_pretrain_steps=0,
            validation_episodes=1,
        ),
    )

    def collapsed_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        assert device == torch.device("cpu")
        assert closed_loop
        return TrainingBatchResult(
            total_loss=torch.tensor(0.1),
            loss_terms={"rollout": torch.tensor(0.1)},
            metrics=_physical_metrics(
                config,
                target_coverage=0.01,
                forecast_coverage=0.01,
            ),
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        collapsed_validation,
    )

    def supported_closed_loop(model, _batch, _config, **_kwargs):
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        supported_closed_loop,
    )
    result = train_from_config(config, run_name="fresh-collapsed")
    numbered = torch.load(
        Path(result["run_directory"]) / "checkpoints" / "validation_step_000000.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert result["best_rollout_validated"] is False
    assert numbered["metrics"]["selection_accepted"] == 0.0
    assert numbered["metrics"]["selection_training_support_failures"]


def test_pooled_unsupported_reference_is_diagnostic_not_validated_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=1,
            rgb_pretrain_steps=0,
            validation_episodes=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    scenario = _selection_scenario_slugs(config)[0]

    def unsupported_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        assert device == torch.device("cpu")
        assert closed_loop
        metrics = {
            "selection_metric_supported": 0.0,
            f"scenario_{scenario}_episode_count": 1.0,
            f"scenario_{scenario}_selection_metric_supported": 0.0,
            f"scenario_{scenario}_supported_episode_count": 0.0,
            f"scenario_{scenario}_minimum_supported_episode_count": 1.0,
            "seed_100000_selection_metric_supported": 0.0,
        }
        for suffix, _ in _selection_horizon_keys(config):
            metrics[f"physical_forecast_predictable_target_count@{suffix}"] = 0.0
            metrics[f"physical_rollout_position@{suffix}_coordinate_count"] = 0.0
            metrics[f"scenario_{scenario}_physical_forecast_predictable_target_count@{suffix}"] = (
                0.0
            )
            metrics[f"scenario_{scenario}_physical_rollout_position@{suffix}_coordinate_count"] = (
                0.0
            )
        return TrainingBatchResult(
            total_loss=torch.tensor(0.1),
            loss_terms={"rollout": torch.tensor(0.1)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def supported_closed_loop(model, _batch, _config, **_kwargs):
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        unsupported_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        supported_closed_loop,
    )

    result = train_from_config(config, run_name="pooled-unsupported-reference")
    checkpoint_directory = Path(result["run_directory"]) / "checkpoints"
    numbered = torch.load(
        checkpoint_directory / "validation_step_000000.pt",
        map_location="cpu",
        weights_only=False,
    )
    reference = torch.load(
        checkpoint_directory / "reference_rollout.pt",
        map_location="cpu",
        weights_only=False,
    )

    for payload in (numbered, reference):
        metrics = payload["metrics"]
        assert metrics["selection_metric_supported"] == 0.0
        assert metrics["rollout_reference_validated"] == 0.0
        assert metrics["checkpoint_contains_reference_rollout_weights"] == 0.0
        assert metrics[f"scenario_{scenario}_episode_count"] == 1.0
        assert metrics[f"scenario_{scenario}_selection_metric_supported"] == 0.0
        assert metrics["seed_100000_selection_metric_supported"] == 0.0
    reference_metrics = reference["metrics"]
    assert (
        reference_metrics["reference_rollout_artifact_model_state_hash"]
        == (reference_metrics["checkpoint_model_state_hash"])
    )
    assert reference_metrics["reference_rollout_artifact_checkpoint_step"] == 0.0
    assert "reference_rollout_model_state_hash" not in reference_metrics
    assert "reference_rollout_checkpoint_step" not in reference_metrics


def test_first_supported_after_pooled_failure_only_reestablishes_reference(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=2,
            rgb_pretrain_steps=0,
            validation_episodes=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    scenario = _selection_scenario_slugs(config)[0]
    validation_calls = 0

    def sequenced_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        nonlocal validation_calls
        assert device == torch.device("cpu")
        assert closed_loop
        validation_calls += 1
        if validation_calls == 1:
            metrics = {
                "selection_metric_supported": 0.0,
                f"scenario_{scenario}_episode_count": 1.0,
                f"scenario_{scenario}_selection_metric_supported": 0.0,
                f"scenario_{scenario}_supported_episode_count": 0.0,
                f"scenario_{scenario}_minimum_supported_episode_count": 1.0,
                "seed_100000_selection_metric_supported": 0.0,
            }
            for suffix, _ in _selection_horizon_keys(config):
                metrics[f"physical_forecast_predictable_target_count@{suffix}"] = 0.0
                metrics[f"physical_rollout_position@{suffix}_coordinate_count"] = 0.0
                metrics[
                    f"scenario_{scenario}_physical_forecast_predictable_target_count@{suffix}"
                ] = 0.0
                metrics[
                    f"scenario_{scenario}_physical_rollout_position@{suffix}_coordinate_count"
                ] = 0.0
        elif validation_calls == 2:
            metrics = _physical_metrics(config, score_scale=1.0)
        else:
            metrics = _physical_metrics(config, score_scale=0.9)
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def supported_closed_loop(model, _batch, _config, **_kwargs) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        sequenced_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        supported_closed_loop,
    )

    result = train_from_config(config, run_name="reference-reestablishment")
    checkpoint_directory = Path(result["run_directory"]) / "checkpoints"
    first_supported = torch.load(
        checkpoint_directory / "validation_step_000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    improved = torch.load(
        checkpoint_directory / "validation_step_000002.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert first_supported["metrics"]["selection_accepted"] == 0.0
    assert first_supported["metrics"]["rollout_reference_validated"] == 1.0
    assert any(
        failure["metric"] == "complete_fixed_reference_comparison"
        for failure in first_supported["metrics"]["selection_reference_guardrail_failures"]
    )
    assert improved["metrics"]["selection_accepted"] == 1.0
    assert result["best_rollout_validated"] is True


def test_branched_resume_retains_incomplete_reference_comparison_requirement(
    tmp_path,
    monkeypatch,
) -> None:
    base = load_config("configs/tiny_overfit.yaml")
    source_config = replace(
        base,
        project=replace(base.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            base.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            base.training,
            steps=1,
            rgb_pretrain_steps=0,
            validation_episodes=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    branch_config = replace(
        source_config,
        training=replace(source_config.training, steps=2),
    )
    scenario = _selection_scenario_slugs(source_config)[0]
    validation_supported = False

    def validation_result(
        _model,
        _loader,
        config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        assert device == torch.device("cpu")
        assert closed_loop
        if validation_supported:
            metrics = _physical_metrics(config)
        else:
            metrics = {
                "selection_metric_supported": 0.0,
                f"scenario_{scenario}_episode_count": 1.0,
                f"scenario_{scenario}_selection_metric_supported": 0.0,
                f"scenario_{scenario}_supported_episode_count": 0.0,
                f"scenario_{scenario}_minimum_supported_episode_count": 1.0,
                "seed_100000_selection_metric_supported": 0.0,
            }
            for suffix, _ in _selection_horizon_keys(config):
                metrics[f"physical_forecast_predictable_target_count@{suffix}"] = 0.0
                metrics[f"physical_rollout_position@{suffix}_coordinate_count"] = 0.0
                metrics[
                    f"scenario_{scenario}_physical_forecast_predictable_target_count@{suffix}"
                ] = 0.0
                metrics[
                    f"scenario_{scenario}_physical_rollout_position@{suffix}_coordinate_count"
                ] = 0.0
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def supported_closed_loop(model, _batch, _config, **_kwargs) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        validation_result,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        supported_closed_loop,
    )

    source_result = train_from_config(source_config, run_name="unsupported-source")
    source_last = Path(source_result["last_checkpoint"])
    source_payload = torch.load(source_last, map_location="cpu", weights_only=False)
    assert source_payload["metrics"]["rollout_reference_validated"] == 0.0
    assert source_payload["metrics"]["incomplete_reference_comparison_required"] == 1.0

    validation_supported = True
    branch_result = train_from_config(
        branch_config,
        run_name="supported-branch",
        resume_path=source_last,
    )
    branch_checkpoint = torch.load(
        Path(branch_result["run_directory"]) / "checkpoints" / "validation_step_000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    branch_metrics = branch_checkpoint["metrics"]
    assert branch_metrics["selection_accepted"] == 0.0
    assert branch_metrics["rollout_reference_validated"] == 1.0
    assert branch_metrics["incomplete_reference_comparison_required"] == 0.0
    assert any(
        failure["metric"] == "complete_fixed_reference_comparison"
        for failure in branch_metrics["selection_reference_guardrail_failures"]
    )
    assert branch_result["best_rollout_validated"] is False


def test_imported_unsupported_reference_continues_without_repeated_validation(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=2,
            rgb_pretrain_steps=0,
            train_episodes=2,
            validation_episodes=1,
            batch_size=1,
            eval_every=2,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    initial_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    initialize_from = save_checkpoint(
        tmp_path / "unsupported-initial.pt",
        model=initial_model,
        optimizer=None,
        config=config,
        step=0,
        device="cpu",
    )
    validation_calls = 0

    def unsupported_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        nonlocal validation_calls
        assert device == torch.device("cpu")
        assert closed_loop
        validation_calls += 1
        metrics = _physical_metrics(config)
        scenario = _selection_scenario_slugs(config)[0]
        prefix = f"scenario_{scenario}_"
        metrics[f"{prefix}selection_metric_supported"] = 0.0
        for name in tuple(metrics):
            if name.startswith(f"{prefix}validation_"):
                del metrics[name]
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def supported_closed_loop(model, _batch, _config, **_kwargs) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        unsupported_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        supported_closed_loop,
    )

    result = train_from_config(
        config,
        run_name="imported-unsupported",
        initialize_from_path=initialize_from,
    )
    records = [
        json.loads(line)
        for line in Path(result["metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]

    # One initialization validation plus the declared step-two/final
    # validation. There is no expensive validation before optimizer step one.
    assert validation_calls == 2
    assert result["optimizer_updates_this_invocation"] == 2
    assert result["best_rollout_validated"] is False
    assert result["best_checkpoint_kind"] == "last_unvalidated"
    initialization_control = [
        record for record in records if record["split"] == "training_control_initialization_support"
    ]
    assert len(initialization_control) == 1
    assert initialization_control[0]["initialization_candidate_accepted"] == 0.0
    assert initialization_control[0]["initialization_reference_established"] == 1.0
    assert initialization_control[0]["initialization_training_continues"] == 1.0


def test_first_supported_incumbent_must_pass_fixed_reference_guardrails(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=2,
            rgb_pretrain_steps=0,
            train_episodes=2,
            validation_episodes=1,
            batch_size=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    initial_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    initialize_from = save_checkpoint(
        tmp_path / "guarded-initial.pt",
        model=initial_model,
        optimizer=None,
        config=config,
        step=0,
        device="cpu",
    )
    validation_calls = 0

    def sequenced_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        nonlocal validation_calls
        assert device == torch.device("cpu")
        assert closed_loop
        validation_calls += 1
        if validation_calls == 1:
            metrics = _physical_metrics(config)
            scenario = _selection_scenario_slugs(config)[0]
            prefix = f"scenario_{scenario}_"
            metrics[f"{prefix}selection_metric_supported"] = 0.0
            for name in tuple(metrics):
                if name.startswith(f"{prefix}validation_"):
                    del metrics[name]
        elif validation_calls == 2:
            metrics = _physical_metrics(config, score_scale=1.2)
        else:
            metrics = _physical_metrics(config, score_scale=0.9)
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def supported_closed_loop(model, _batch, _config, **_kwargs) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        sequenced_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        supported_closed_loop,
    )

    result = train_from_config(
        config,
        run_name="guarded-first-incumbent",
        initialize_from_path=initialize_from,
    )
    run_directory = Path(result["run_directory"])
    regressed = torch.load(
        run_directory / "checkpoints" / "validation_step_000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    repaired = torch.load(
        run_directory / "checkpoints" / "validation_step_000002.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert regressed["metrics"]["selection_accepted"] == 0.0
    assert regressed["metrics"]["selection_training_support_passed"] == 1.0
    assert regressed["metrics"]["selection_reference_guardrail_failures"]
    assert regressed["metrics"]["checkpoint_contains_reference_rollout_weights"] == 0.0
    assert (
        regressed["metrics"]["checkpoint_model_state_hash"]
        != regressed["metrics"]["reference_rollout_model_state_hash"]
    )
    assert repaired["metrics"]["selection_accepted"] == 1.0
    assert result["best_rollout_validated"] is True
    assert repaired["metrics"]["best_rollout_checkpoint_step"] == 2.0


def test_exact_resume_preserves_reference_without_deployable_incumbent(
    tmp_path,
    monkeypatch,
) -> None:
    base = load_config("configs/tiny_overfit.yaml")
    source_config = replace(
        base,
        project=replace(base.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            base.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            base.training,
            steps=1,
            rgb_pretrain_steps=0,
            validation_episodes=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    resumed_config = replace(
        source_config,
        training=replace(source_config.training, steps=2),
    )
    model = OnlineWorldModel.from_config(source_config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=source_config.training.learning_rate)
    reference_selection = _rollout_selection_metrics(
        _physical_metrics(source_config),
        source_config,
        require_scenarios=True,
    )
    checkpoint_directory = tmp_path / "source-run" / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    reference_hash = _current_model_state_hash(model)
    reference_metrics = {
        **_physical_metrics(source_config),
        "selection_accepted": 0.0,
        "best_rollout_validated": 0.0,
        "rollout_reference_validated": 1.0,
        "incomplete_reference_comparison_required": 0.0,
        "best_measurement_validated": 0.0,
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        **reference_selection.validation_metrics(),
        **reference_selection.checkpoint_metrics(prefix="reference_rollout"),
        **_validation_protocol_checkpoint_metrics(source_config),
        "checkpoint_model_state_hash": reference_hash,
        "checkpoint_contains_best_rollout_weights": 0.0,
        "checkpoint_contains_reference_rollout_weights": 1.0,
        "reference_rollout_model_state_hash": reference_hash,
        "reference_rollout_checkpoint_step": 0.0,
    }
    save_checkpoint(
        checkpoint_directory / "reference_rollout.pt",
        model=model,
        optimizer=optimizer,
        config=source_config,
        step=0,
        metrics=reference_metrics,
        device="cpu",
    )
    with torch.no_grad():
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        parameter.reshape(-1)[0].add_(0.01)
    last_hash = _current_model_state_hash(model)
    last_metrics = {
        **_physical_metrics(source_config),
        "best_rollout_validated": 0.0,
        "rollout_reference_validated": 1.0,
        "incomplete_reference_comparison_required": 0.0,
        "best_measurement_validated": 0.0,
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        **reference_selection.checkpoint_metrics(prefix="reference_rollout"),
        **_validation_protocol_checkpoint_metrics(source_config),
        "checkpoint_model_state_hash": last_hash,
        "checkpoint_contains_best_rollout_weights": 0.0,
        "checkpoint_contains_reference_rollout_weights": 0.0,
        "reference_rollout_model_state_hash": reference_hash,
        "reference_rollout_checkpoint_step": 0.0,
        "measurement_handoff_completed": 1.0,
        "training_data_draw_step": 1.0,
        "skipped_no_gradient_batches": 0.0,
        "final_validation_completed": 1.0,
    }
    resume = save_checkpoint(
        checkpoint_directory / "last.pt",
        model=model,
        optimizer=optimizer,
        config=source_config,
        step=1,
        metrics=last_metrics,
        device="cpu",
    )

    def regressed_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        assert device == torch.device("cpu")
        assert closed_loop
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=_physical_metrics(resumed_config, score_scale=1.2),
            phase="closed_loop_rgb",
        )

    def supported_closed_loop(model, _batch, _config, **_kwargs) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        regressed_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        supported_closed_loop,
    )

    result = train_from_config(
        resumed_config,
        resume_path=resume,
    )
    rejected = torch.load(
        checkpoint_directory / "validation_step_000002.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert result["best_rollout_validated"] is False
    assert rejected["metrics"]["selection_accepted"] == 0.0
    assert rejected["metrics"]["selection_reference_guardrail_failures"]
    assert rejected["metrics"]["reference_rollout_model_state_hash"] == reference_hash
    assert rejected["metrics"]["reference_rollout_checkpoint_step"] == 0.0
    assert rejected["metrics"]["checkpoint_contains_reference_rollout_weights"] == 0.0


def test_only_support_collapse_rolls_back_mutable_causal_state(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=2,
            rgb_pretrain_steps=0,
            train_episodes=2,
            validation_episodes=1,
            batch_size=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    initial_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    initial_optimizer = torch.optim.AdamW(
        initial_model.parameters(),
        lr=config.training.learning_rate,
    )
    initialize_from = save_checkpoint(
        tmp_path / "initial.pt",
        model=initial_model,
        optimizer=initial_optimizer,
        config=config,
        step=0,
        device="cpu",
    )
    validation_calls = 0

    def fake_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        nonlocal validation_calls
        assert device == torch.device("cpu")
        assert closed_loop
        validation_calls += 1
        if validation_calls == 1:
            metrics = _physical_metrics(config, score_scale=1.0)
        elif validation_calls == 2:
            metrics = _physical_metrics(
                config,
                score_scale=0.2,
                target_coverage=0.01,
                forecast_coverage=0.01,
            )
        else:
            metrics = _physical_metrics(config, score_scale=1.1)
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def fake_closed_loop(
        model,
        _batch,
        _config,
        **_kwargs,
    ) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        fake_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        fake_closed_loop,
    )

    result = train_from_config(
        config,
        run_name="support-collapse-rollback",
        initialize_from_path=initialize_from,
    )

    records = [
        json.loads(line)
        for line in Path(result["metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    rollbacks = [
        record for record in records if record["split"] == "training_control_support_collapse"
    ]
    assert len(rollbacks) == 1
    assert rollbacks[0]["optimizer_state_reset"] == 1.0
    numbered_one = torch.load(
        Path(result["run_directory"]) / "checkpoints" / "validation_step_000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    numbered_two = torch.load(
        Path(result["run_directory"]) / "checkpoints" / "validation_step_000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert numbered_one["metrics"]["selection_accepted"] == 0.0
    assert numbered_one["metrics"]["selection_training_support_failures"]
    assert numbered_two["metrics"]["selection_accepted"] == 0.0
    assert numbered_two["metrics"]["selection_training_support_failures"] == []
    last = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
    best = torch.load(result["best_rollout_checkpoint"], map_location="cpu", weights_only=False)
    assert (
        _current_model_state_hash(initial_model) == best["metrics"]["best_rollout_model_state_hash"]
    )
    assert _model_state_hash(last["model_state"]) != _model_state_hash(best["model_state"])
    assert last["metrics"]["checkpoint_contains_best_rollout_weights"] == 0.0
    assert (
        last["metrics"]["best_rollout_model_state_hash"]
        == best["metrics"]["best_rollout_model_state_hash"]
    )


def test_scenario_only_support_failure_preserves_mutable_causal_state(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=1,
            rgb_pretrain_steps=0,
            train_episodes=1,
            validation_episodes=1,
            batch_size=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    initial_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    initialize_from = save_checkpoint(
        tmp_path / "scenario-support-initial.pt",
        model=initial_model,
        optimizer=None,
        config=config,
        step=0,
        device="cpu",
    )
    validation_calls = 0

    def fake_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        nonlocal validation_calls
        assert device == torch.device("cpu")
        assert closed_loop
        validation_calls += 1
        metrics = _physical_metrics(config)
        if validation_calls == 2:
            scenario = _selection_scenario_slugs(config)[0]
            prefix = f"scenario_{scenario}_"
            metrics[f"{prefix}selection_metric_supported"] = 0.0
            for name in tuple(metrics):
                if name.startswith(f"{prefix}validation_"):
                    del metrics[name]
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def fake_closed_loop(model, _batch, _config, **_kwargs) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        fake_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        fake_closed_loop,
    )

    result = train_from_config(
        config,
        run_name="scenario-support-preserved",
        initialize_from_path=initialize_from,
    )
    records = [
        json.loads(line)
        for line in Path(result["metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert not [
        record for record in records if record["split"] == "training_control_support_collapse"
    ]
    validation = torch.load(
        Path(result["run_directory"]) / "checkpoints" / "validation_step_000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    last = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
    best = torch.load(result["best_rollout_checkpoint"], map_location="cpu", weights_only=False)
    assert validation["metrics"]["selection_accepted"] == 0.0
    assert validation["metrics"]["selection_training_support_passed"] == 0.0
    assert validation["metrics"]["selection_mutable_training_support_passed"] == 1.0
    assert _model_state_hash(last["model_state"]) != _model_state_hash(best["model_state"])
    assert last["optimizer_state"]["state"]
    assert last["metrics"]["checkpoint_state_role"] == "mutable_training_iterate"


def test_terminal_support_collapse_checkpoint_truthfully_contains_restored_incumbent(
    tmp_path,
    monkeypatch,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=1,
            rgb_pretrain_steps=0,
            train_episodes=1,
            validation_episodes=1,
            batch_size=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
        ),
    )
    initial_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    initialize_from = save_checkpoint(
        tmp_path / "terminal-initial.pt",
        model=initial_model,
        optimizer=None,
        config=config,
        step=0,
        device="cpu",
    )
    validation_calls = 0

    def fake_validation(
        _model,
        _loader,
        _config,
        *,
        device,
        closed_loop,
        **_progress,
    ) -> TrainingBatchResult:
        nonlocal validation_calls
        assert device == torch.device("cpu")
        assert closed_loop
        validation_calls += 1
        metrics = (
            _physical_metrics(config)
            if validation_calls == 1
            else _physical_metrics(
                config,
                score_scale=0.2,
                target_coverage=0.01,
                forecast_coverage=0.01,
            )
        )
        return TrainingBatchResult(
            total_loss=torch.tensor(0.4),
            loss_terms={"rollout": torch.tensor(0.4)},
            metrics=metrics,
            phase="closed_loop_rgb",
        )

    def fake_closed_loop(model, _batch, _config, **_kwargs) -> TrainingBatchResult:
        parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        loss = parameter.reshape(-1)[0] + 10.0
        return TrainingBatchResult(
            total_loss=loss,
            loss_terms={"state": loss},
            metrics={"matched_object_frames": 1.0},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(
        "world_model.training.trainer._validation_loader_result",
        fake_validation,
    )
    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        fake_closed_loop,
    )

    result = train_from_config(
        config,
        run_name="terminal-support-collapse",
        initialize_from_path=initialize_from,
    )
    last = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
    best = torch.load(result["best_rollout_checkpoint"], map_location="cpu", weights_only=False)
    best_hash = _model_state_hash(best["model_state"])

    assert _model_state_hash(last["model_state"]) == best_hash
    assert last["metrics"]["checkpoint_model_state_hash"] == best_hash
    assert last["metrics"]["checkpoint_contains_best_rollout_weights"] == 1.0
    assert last["metrics"]["support_collapse_rollback_applied_at_checkpoint"] == 1.0
    assert last["metrics"]["checkpoint_state_role"] == "restored_best_rollout"
    assert last["metrics"]["final_validation_completed"] == 1.0
    assert last["optimizer_state"]["state"] == {}


@pytest.mark.parametrize(
    ("data_draw_step", "skipped"),
    [
        (1.5, 0.0),
        (1.0, 1.0),
    ],
)
def test_exact_resume_rejects_corrupt_data_progress_counters(
    tmp_path,
    data_draw_step,
    skipped,
) -> None:
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        project=replace(source.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            source.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            source.training,
            steps=2,
            rgb_pretrain_steps=0,
            validation_episodes=1,
        ),
    )
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    checkpoint = save_checkpoint(
        tmp_path / "counter-source" / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={
            "measurement_handoff_completed": 1.0,
            "training_data_draw_step": data_draw_step,
            "skipped_no_gradient_batches": skipped,
        },
        device="cpu",
    )

    with pytest.raises(ValueError, match="finite nonnegative integer|data-progress invariant"):
        train_from_config(config, resume_path=checkpoint)
