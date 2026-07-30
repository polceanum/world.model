from __future__ import annotations

from dataclasses import replace

import torch

from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import save_checkpoint
from world_model.training.trainer import (
    _current_model_state_hash,
    _fresh_causal_optimizer_state,
    _preserve_resume_selector_checkpoint,
    _rollout_selection_improves,
    _rollout_selection_metrics,
    _rollout_selection_passes_guardrails,
    _rollout_validation_protocol_hash,
    _selection_horizon_keys,
    _validation_protocol_checkpoint_metrics,
    _verified_selector_checkpoint,
)
from world_model.utils.config import OrpheusConfig, load_config


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

    assert _rollout_validation_protocol_hash(extended) == (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_batch) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_physics) != (
        _rollout_validation_protocol_hash(config)
    )
    assert _rollout_validation_protocol_hash(changed_evaluation) != (
        _rollout_validation_protocol_hash(config)
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
