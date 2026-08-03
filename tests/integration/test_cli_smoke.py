from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
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
    # One update is intentionally too short to prove the adjacent-frame fast
    # RGB path. The trainer must not invent a deployable perception selector.
    assert not (run_directory / "checkpoints" / "best_measurement.pt").exists()
    assert (run_directory / "metrics.jsonl").is_file()
    assert (run_directory / "config.resolved.yaml").is_file()
    pretrain_summary = json.loads(
        (run_directory / "train_summary.json").read_text(encoding="utf-8")
    )
    assert pretrain_summary["best_checkpoint_kind"] == "last_unvalidated"
    assert pretrain_summary["best_rollout_checkpoint"] is None
    assert pretrain_summary["best_rollout_validated"] is False
    assert pretrain_summary["best_measurement_validated"] is False
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
    initialized_checkpoint = initialized_run / "checkpoints" / "last.pt"
    initialized_before_resume = torch.load(
        initialized_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert initialized_before_resume["metrics"]["measurement_handoff_completed"] == 0.0

    _run(
        "train.py",
        "--config",
        str(config_path),
        "--resume",
        str(initialized_checkpoint),
        "--set",
        "training.steps=2",
    )
    initialized_records = [
        json.loads(line)
        for line in (initialized_run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        record.get("split") == "validation_measurement_handoff" for record in initialized_records
    )
    initialized_after_resume = torch.load(
        initialized_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert initialized_after_resume["metrics"]["measurement_handoff_completed"] == 1.0
    resumed_initialized_summary = json.loads(
        (initialized_run / "train_summary.json").read_text(encoding="utf-8")
    )
    assert resumed_initialized_summary["initialized_from"] == str(checkpoint.resolve())
    initialized_metadata = json.loads(
        (initialized_run / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert initialized_metadata["initialize_from_path"] == str(checkpoint.resolve())
    assert len(initialized_metadata["resume_history"]) == 1
    assert initialized_metadata["resume_history"][0]["resume_path"] == str(
        initialized_checkpoint.resolve()
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
    # A single causal update still has zero broad tracking coverage in this
    # intentionally tiny smoke. Record the validation, but do not fabricate a
    # deployable rollout selector merely to satisfy the smoke test.
    assert not best_rollout_path.exists()
    reference_rollout_payload = torch.load(
        run_directory / "checkpoints" / "reference_rollout.pt",
        map_location="cpu",
        weights_only=False,
    )
    # Zero-support validation persists a diagnostic artifact, not a
    # tensor-linked numerical reference. The first later supported candidate
    # must establish a complete reference and wait for another comparison.
    assert (
        reference_rollout_payload["metrics"]["checkpoint_contains_reference_rollout_weights"] == 0.0
    )
    assert reference_rollout_payload["metrics"]["incomplete_reference_comparison_required"] == 1.0
    resumed_last_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_last_payload["metrics"]["best_rollout_validated"] == 0.0
    assert resumed_last_payload["metrics"]["checkpoint_contains_best_rollout_weights"] == 0.0
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
    closed_loop_validation_records = [
        record
        for record in training_records
        if record.get("split", "").startswith("validation")
        and record.get("phase") == "closed_loop_rgb"
    ]
    assert closed_loop_validation_records
    assert any(
        record.get("selection_metric_supported") == 0.0 for record in closed_loop_validation_records
    )
    assert all(
        record["selection_accepted"] in {0.0, 1.0}
        and record["selection_rejection_reason_count"] >= 0.0
        and isinstance(json.loads(record["selection_rejection_reasons_json"]), list)
        for record in closed_loop_validation_records
    )
    assert closed_loop_records
    assert closed_loop_records[-1]["fast_path_supervised"] == 1.0
    assert (
        closed_loop_records[-1]["gradient_norm_pre_clip"]
        == closed_loop_records[-1]["gradient_norm"]
    )
    assert 0.0 < closed_loop_records[-1]["interaction_gradient_clip_coefficient"] <= 1.0
    assert closed_loop_records[-1][
        "interaction_gradient_norm_applied_before_global_clip"
    ] == pytest.approx(
        closed_loop_records[-1]["interaction_gradient_norm_pre_clip"]
        * closed_loop_records[-1]["interaction_gradient_clip_coefficient"]
    )
    assert 0.0 < closed_loop_records[-1]["gradient_clip_coefficient"] <= 1.0
    assert closed_loop_records[-1]["gradient_norm_applied"] == pytest.approx(
        closed_loop_records[-1]["gradient_norm_pre_global_clip"]
        * closed_loop_records[-1]["gradient_clip_coefficient"]
    )
    assert closed_loop_records[-1]["gradient_total_clip_coefficient"] == pytest.approx(
        closed_loop_records[-1]["gradient_norm_applied"]
        / closed_loop_records[-1]["gradient_norm_pre_clip"]
    )
    assert closed_loop_records[-1]["gradient_norm_applied"] <= 1.0 + 1.0e-6

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
    assert "forecast_predictable_target_count@0.050s" in report["metrics"]
    assert report["metrics"]["forecast_censored_tracked_count@0.050s"] == 0.0
    assert (
        report["metrics"]["forecast_calibration_coordinate_count"]
        >= (report["metrics"]["model@0.050s_position_coordinate_count"])
    )
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
    training_state = json.loads((run_directory / "training_state.json").read_text(encoding="utf-8"))
    assert training_state["state"] == "completed"
    assert training_state["completed_steps"] == 3
    assert (evaluation_directory / "report.md").is_file()
