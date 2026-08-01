from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import save_checkpoint
from world_model.training.loop import TrainingBatchResult
from world_model.training.trainer import (
    _current_model_state_hash,
    _expected_resume_checkpoint_devices,
    _fresh_causal_optimizer_state,
    _measurement_selection_metrics,
    _measurement_validation_protocol_hash,
    _preserve_resume_measurement_checkpoint,
    _preserve_resume_selector_checkpoint,
    _resolve_run_directory,
    _resolve_training_devices,
    _rollout_selection_improves,
    _rollout_selection_metrics,
    _rollout_selection_passes_guardrails,
    _rollout_validation_protocol_hash,
    _selection_horizon_keys,
    _validate_exact_resume_source,
    _validation_protocol_checkpoint_metrics,
    _verified_measurement_checkpoint,
    _verified_selector_checkpoint,
    train_from_config,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.device import select_device


def _physical_metrics(
    config: OrpheusConfig,
    *,
    score_scale: float = 1.0,
    velocity: float = 0.8,
    prediction_precision: float = 0.8,
    position_coverage90: float = 0.9,
    forecast_coverage: float = 0.9,
) -> dict[str, float]:
    metrics = {
        "validation_position_rmse_m": 0.4,
        "validation_velocity_rmse_mps": velocity,
        "validation_target_coverage": 0.9,
        "validation_prediction_precision": prediction_precision,
        "validation_collision_f1": 0.6,
        "validation_id_switch_rate": 0.01,
        "validation_position_coverage90": position_coverage90,
    }
    for index, (suffix, _) in enumerate(_selection_horizon_keys(config)):
        metrics[f"validation_position_rmse@{suffix}"] = score_scale * (0.4 - index * 0.1)
        metrics[f"validation_forecast_target_coverage@{suffix}"] = forecast_coverage
    return metrics


def _selector_checkpoint_metrics(
    model: OnlineWorldModel,
    config: OrpheusConfig,
    *,
    step: int,
) -> dict[str, object]:
    selection = _rollout_selection_metrics(_physical_metrics(config), config)
    model_hash = _current_model_state_hash(model)
    return {
        "best_rollout_validated": 1.0,
        "rollout_reference_validated": 1.0,
        "rollout_selection_metric_version": 3.0,
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
            "rollout_selection_metric_version": 4.0,
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
    metrics["rollout_selection_metric_version"] = 4.0
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

    with pytest.raises(ValueError, match="best_rollout.pt or reference_rollout.pt"):
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
