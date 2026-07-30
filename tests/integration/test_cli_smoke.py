from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from world_model.utils.config import load_config, save_resolved_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_cli_config(tmp_path: Path) -> Path:
    config = load_config(REPOSITORY_ROOT / "configs" / "tiny_overfit.yaml")
    config = replace(
        config,
        project=replace(
            config.project,
            output_dir=str(tmp_path / "runs"),
            deterministic=True,
        ),
        device=replace(config.device, preference="cpu", cuda_amp=False),
        simulator=replace(
            config.simulator,
            image_size=(32, 32),
            sequence_frames=3,
            min_objects=1,
            max_objects=1,
        ),
        training=replace(
            config.training,
            batch_size=1,
            steps=2,
            rgb_pretrain_steps=1,
            tbptt_steps=2,
            train_episodes=1,
            validation_episodes=1,
            checkpoint_every=2,
            eval_every=2,
            log_every=1,
            horizon_weights=(1.0,),
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.05,),
            episodes=1,
        ),
    )
    config.validate()
    path = tmp_path / "cli_smoke.yaml"
    save_resolved_config(config, path)
    return path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_train_resume_and_evaluate_cli_rgb_only(tmp_path):
    config_path = _write_cli_config(tmp_path)
    dry_run = _run(
        "train.py",
        "--config",
        str(config_path),
        "--run-name",
        "cli-smoke",
        "--dry-run",
    )
    assert '"runtime_modality": "rgb"' in dry_run.stdout
    assert '"nominal_training_episode_draws": 2' in dry_run.stdout
    assert '"nominal_dataset_passes": 2.0' in dry_run.stdout

    _run(
        "train.py",
        "--config",
        str(config_path),
        "--run-name",
        "cli-smoke",
        "--set",
        "training.steps=1",
    )
    run_directories = list((tmp_path / "runs").glob("*-cli-smoke"))
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    checkpoint = run_directory / "checkpoints" / "last.pt"
    assert checkpoint.is_file()
    assert not (run_directory / "checkpoints" / "best_rollout.pt").exists()
    assert (run_directory / "checkpoints" / "best_measurement.pt").is_file()
    assert (run_directory / "metrics.jsonl").is_file()
    assert (run_directory / "config.resolved.yaml").is_file()
    pretrain_summary = json.loads(
        (run_directory / "train_summary.json").read_text(encoding="utf-8")
    )
    assert pretrain_summary["best_checkpoint_kind"] == "best_measurement"
    assert pretrain_summary["best_rollout_checkpoint"] is None
    assert pretrain_summary["best_rollout_validated"] is False
    assert pretrain_summary["best_rollout_loss"] is None
    assert pretrain_summary["model_parameter_count"] > 0
    assert pretrain_summary["planned_training_episode_draws"] == 1

    _run(
        "train.py",
        "--config",
        str(config_path),
        "--run-name",
        "cli-initialize",
        "--initialize-from",
        str(checkpoint),
        "--set",
        "training.steps=1",
    )
    initialized_run = next((tmp_path / "runs").glob("*-cli-initialize"))
    initialized_summary = json.loads(
        (initialized_run / "train_summary.json").read_text(encoding="utf-8")
    )
    assert initialized_summary["initialized_from"] == str(checkpoint.resolve())
    assert initialized_summary["resumed_from"] is None
    assert (
        torch.load(
            initialized_run / "checkpoints" / "last.pt",
            map_location="cpu",
            weights_only=False,
        )["step"]
        == 1
    )

    _run(
        "train.py",
        "--config",
        str(config_path),
        "--resume",
        str(checkpoint),
        "--set",
        "training.steps=2",
    )
    best_rollout_path = run_directory / "checkpoints" / "best_rollout.pt"
    assert best_rollout_path.is_file()
    best_rollout_payload = torch.load(
        best_rollout_path,
        map_location="cpu",
        weights_only=False,
    )
    assert best_rollout_payload["step"] in {1, 2}
    assert best_rollout_payload["metrics"]["best_rollout_validated"] == 1.0
    assert "validation_rollout_loss" in best_rollout_payload["metrics"]
    assert (
        best_rollout_payload["metrics"]["best_rollout_loss"]
        == best_rollout_payload["metrics"]["best_rollout_position_loss"]
        == best_rollout_payload["metrics"]["best_rollout_selection_score"]
    )
    assert best_rollout_payload["metrics"]["rollout_selection_metric_version"] == 3.0
    assert "best_rollout_velocity_rmse_mps" in best_rollout_payload["metrics"]
    assert "best_rollout_target_coverage" in best_rollout_payload["metrics"]
    assert "best_rollout_prediction_precision" in best_rollout_payload["metrics"]
    assert "best_rollout_collision_f1" in best_rollout_payload["metrics"]
    assert "best_rollout_id_switch_rate" in best_rollout_payload["metrics"]
    assert "best_rollout_position_calibration_error90" in best_rollout_payload["metrics"]
    assert len(best_rollout_payload["metrics"]["rollout_validation_protocol_hash"]) == 64
    assert best_rollout_payload["metrics"]["checkpoint_contains_best_rollout_weights"] == 1.0
    assert any(
        name.startswith("best_rollout_position_rmse@") for name in best_rollout_payload["metrics"]
    )
    assert any(
        name.startswith("best_rollout_forecast_target_coverage@")
        for name in best_rollout_payload["metrics"]
    )
    reference_rollout_payload = torch.load(
        run_directory / "checkpoints" / "reference_rollout.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert (
        reference_rollout_payload["metrics"]["checkpoint_contains_reference_rollout_weights"] == 1.0
    )
    resumed_last_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert (
        resumed_last_payload["metrics"]["best_rollout_model_state_hash"]
        == best_rollout_payload["metrics"]["best_rollout_model_state_hash"]
    )
    assert (run_directory / "checkpoints" / "validation_step_000001.pt").is_file()
    assert (run_directory / "checkpoints" / "validation_step_000002.pt").is_file()
    training_records = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    closed_loop_records = [
        record
        for record in training_records
        if record.get("split") == "train" and record.get("phase") == "closed_loop_rgb"
    ]
    assert closed_loop_records
    assert closed_loop_records[-1]["fast_path_supervised"] == 1.0

    _run(
        "train.py",
        "--config",
        str(config_path),
        "--resume",
        str(checkpoint),
        "--set",
        "training.steps=3",
    )
    requested_evaluation_directory = run_directory / "evaluation" / "test"
    _run(
        "evaluate.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "test",
        "--output",
        str(requested_evaluation_directory),
    )
    evaluation_directories = list(requested_evaluation_directory.parent.glob("*-test"))
    assert len(evaluation_directories) == 1
    evaluation_directory = evaluation_directories[0]
    report = json.loads((evaluation_directory / "evaluation.json").read_text(encoding="utf-8"))
    assert report["metadata"]["checkpoint_step"] == 3
    assert report["metadata"]["rgb_only"] is True
    assert report["metadata"]["oracle_runtime_input_used"] is False
    assert report["metadata"]["current_detection_distance_threshold_m"] == 0.5
    assert report["metrics"]["evaluated_episodes"] == 1.0
    assert report["metrics"]["nonfinite_output_count"] == 0.0
    assert report["metrics"]["current_assignment_target_coverage"] is not None
    assert "current_detection_recall@0.500m" in report["metrics"]
    assert "current_detection_precision@0.500m" in report["metrics"]
    assert "forecast_target_coverage@0.050s" in report["metrics"]
    assert "identifier_restitution_gate_mean" in report["metrics"]
    assert "identifier_restitution_update_count" in report["metrics"]
    assert "identifier_drag_gate_mean" in report["metrics"]
    assert "identifier_drag_update_count" in report["metrics"]
    if (
        report["metrics"]["identifier_restitution_update_count"] == 0.0
        or report["metrics"]["identifier_drag_update_count"] == 0.0
    ):
        assert any("zero identifier updates" in limitation for limitation in report["limitations"])
    assert (
        report["metrics"]["model@0.050s_position_coordinate_count"]
        == report["metrics"]["constant_velocity@0.050s_position_coordinate_count"]
    )
    assert (evaluation_directory / "report.md").is_file()
