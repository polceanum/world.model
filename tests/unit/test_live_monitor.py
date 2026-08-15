from __future__ import annotations

import fcntl
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import monitor as live_monitor


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _hold_training_lock(run: Path):
    path = run / ".training.lock"
    path.write_text("12345\n", encoding="utf-8")
    handle = path.open("r+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_select_run_prefers_verified_active_resume_over_newer_completed(tmp_path) -> None:
    runs = tmp_path / "runs"
    active = runs / "20260814-235959-active-resume"
    completed = runs / "20260815-000001-completed"
    _write_json(active / "training_state.json", {"state": "running"})
    _write_json(completed / "training_state.json", {"state": "completed"})
    lock = _hold_training_lock(active)
    try:
        assert live_monitor.select_run_directory(runs) == active.resolve()
    finally:
        lock.close()
    assert live_monitor.select_run_directory(runs) == completed.resolve()


def test_discovery_handles_nested_evaluation_and_lexical_synthetic_hour(tmp_path) -> None:
    runs = tmp_path / "runs"
    earlier = runs / "20260814-235959-earlier"
    synthetic = runs / "parent" / "evaluation" / "20260814-240000-later"
    _write_json(earlier / "evaluation.json", {"metadata": {}, "metrics": {}})
    _write_json(synthetic / "evaluation.json", {"metadata": {}, "metrics": {}})

    assert set(live_monitor.discover_run_directories(runs)) == {
        earlier.resolve(),
        synthetic.resolve(),
    }
    assert live_monitor.select_run_directory(runs) == synthetic.resolve()


def test_training_snapshot_shows_progress_robust_loss_trend_and_validation(tmp_path) -> None:
    run = tmp_path / "runs" / "20260815-010000-training"
    _write_json(run / "training_state.json", {"state": "running"})
    _write_json(
        run / "run_metadata.json",
        {
            "device": "cpu",
            "measurement_device": "mps",
            "closed_loop_device": "cpu",
            "precision": "float32",
            "torch_version": "local-test",
        },
    )
    (run / "config.resolved.yaml").write_text(
        "project:\n  name: test\ntraining:\n  steps: 80\n",
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = [
        {
            "step": index * 10,
            "split": "train",
            "phase": "closed_loop_rgb",
            "loss_total": loss,
            "learning_rate": 0.0001,
            "elapsed_seconds": index * 5.0,
            "optimizer_update_applied": 1.0,
            "causal_training_support_present": 1.0,
        }
        for index, loss in enumerate([4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0], start=1)
    ]
    rows.append(
        {
            "step": 80,
            "split": "validation",
            "validation_rollout_selection_score": 0.31,
            "validation_position_rmse_m": 0.25,
            "validation_position_rmse@0.333s": 0.3,
            "validation_position_rmse@1.000s": 0.36,
            "validation_position_rmse@2.000s": 0.51,
            "selection_metric_supported": 1.0,
            "selection_accepted": 0.0,
            "selection_rejection_reason_count": 2.0,
        }
    )
    _write_metrics(run / "metrics.jsonl", rows)
    lock = _hold_training_lock(run)
    try:
        modified = datetime.fromtimestamp(
            (run / "metrics.jsonl").stat().st_mtime,
            timezone.utc,
        )
        snapshot = live_monitor.build_snapshot(run, now=modified + timedelta(seconds=30))
    finally:
        lock.close()

    assert snapshot["status"] == "TRAINING"
    assert snapshot["device"]["measurement_device"] == "mps"
    assert snapshot["training"]["step"] == 80
    assert snapshot["training"]["target_steps"] == 80
    assert snapshot["training"]["rolling_loss_median"] == pytest.approx(2.0)
    assert snapshot["training"]["rolling_loss_delta_percent"] == pytest.approx(-50.0)
    assert snapshot["training"]["validation"]["horizon_rmse_m"] == {
        "0.333": 0.3,
        "1.000": 0.36,
        "2.000": 0.51,
    }
    assert "latest validation candidate was safely rejected (2 guardrails)" in snapshot["warnings"]
    rendered = live_monitor.render_snapshot(snapshot)
    assert "80/80 [####################] 100.0%" in rendered
    assert "val       rollout" in rendered
    assert "1.000s 0.36 m" in rendered
    assert "2.000s 0.51 m" in rendered


def test_failure_and_nonfinite_metrics_take_precedence_over_live_lock(tmp_path) -> None:
    run = tmp_path / "20260815-020000-failed"
    _write_json(run / "training_state.json", {"state": "running"})
    _write_json(
        run / "training_failure.json",
        {
            "state": "failed",
            "exception_type": "FloatingPointError",
            "message": "nonfinite parameter",
        },
    )
    _write_metrics(
        run / "metrics.jsonl",
        [{"step": 16, "split": "train", "loss_total": float("nan")}],
    )
    lock = _hold_training_lock(run)
    try:
        snapshot = live_monitor.build_snapshot(run)
    finally:
        lock.close()

    assert snapshot["status"] == "FAILED"
    assert snapshot["status_detail"] == "FloatingPointError"
    assert "non-finite loss_total at step 16" in snapshot["warnings"]
    assert "nonfinite parameter" in snapshot["warnings"]


def test_completed_evaluation_uses_report_metrics_and_hides_stale_pid(tmp_path) -> None:
    run = tmp_path / "20260815-030000-evaluation"
    _write_json(
        run / "evaluation_progress.json",
        {"stage": "completed", "pid": 999999, "evaluated_episodes": 4},
    )
    _write_json(
        run / "evaluation.json",
        {
            "metadata": {
                "device": "mps",
                "precision": "float32",
                "split": "validation",
                "checkpoint_step": 512,
                "episodes": 4,
            },
            "metrics": {
                "evaluated_episodes": 4.0,
                "posterior_current_position_rmse_m": 0.2,
                "model@0.333s_position_rmse_m": 0.3,
                "model@1.000s_position_rmse_m": 0.4,
                "model@2.000s_position_rmse_m": 0.6,
                "forecast_gaussian_nll": 0.5,
                "nonfinite_output_count": 0.0,
            },
        },
    )

    snapshot = live_monitor.build_snapshot(run)
    assert snapshot["status"] == "COMPLETED"
    assert snapshot["evaluation_result"]["horizon_rmse_m"] == {
        "0.333": 0.3,
        "1.000": 0.4,
        "2.000": 0.6,
    }
    rendered = live_monitor.render_snapshot(snapshot)
    assert "result    current RMSE 0.2 m · NLL 0.5" in rendered
    assert "2.000s 0.6 m" in rendered
    assert "process   pid" not in rendered


def test_active_evaluation_supersedes_completed_training_in_same_directory(
    tmp_path,
    monkeypatch,
) -> None:
    run = tmp_path / "20260815-031000-training-and-evaluation"
    _write_json(run / "training_state.json", {"state": "completed", "completed_steps": 8})
    _write_json(
        run / "evaluation_progress.json",
        {"stage": "anchor_complete", "pid": 12345, "batch": 1, "batches": 2},
    )
    monkeypatch.setattr(live_monitor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(live_monitor, "_pid_command_matches_evaluator", lambda _pid: True)

    snapshot = live_monitor.build_snapshot(run)

    assert snapshot["status"] == "EVALUATING"
    assert snapshot["status_detail"] == "anchor_complete"


def test_active_trainer_supersedes_old_completed_evaluation_in_same_directory(
    tmp_path,
) -> None:
    run = tmp_path / "20260815-032000-evaluation-and-resume"
    _write_json(run / "training_state.json", {"state": "running"})
    _write_json(run / "evaluation_progress.json", {"stage": "completed", "pid": 999999})
    lock = _hold_training_lock(run)
    try:
        snapshot = live_monitor.build_snapshot(run)
    finally:
        lock.close()

    assert snapshot["status"] == "TRAINING"


def test_jsonl_tail_ignores_only_an_unterminated_append(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"step": 1, "split": "train"}\n{"step":')

    rows, errors = live_monitor._read_jsonl_tail(path)

    assert rows == [{"step": 1, "split": "train"}]
    assert errors == []


def test_once_without_runs_is_a_concise_error(tmp_path, capsys) -> None:
    result = live_monitor.main(["--runs-root", str(tmp_path / "missing"), "--once"])

    assert result == 2
    assert "no run artifacts found" in capsys.readouterr().err


def test_stale_running_state_is_not_reported_as_live(tmp_path) -> None:
    run = tmp_path / "20260815-040000-stale"
    _write_json(run / "training_state.json", {"state": "running"})
    updated = datetime.fromtimestamp((run / "training_state.json").stat().st_mtime, timezone.utc)

    snapshot = live_monitor.build_snapshot(
        run,
        now=updated + timedelta(seconds=1),
        stale_after_seconds=60,
    )

    assert snapshot["status"] == "STALE"
    assert "trainer lock is not held" in snapshot["status_detail"]
    assert "authoritative running state has no held trainer lock" in snapshot["warnings"]
    assert "ETA" not in live_monitor.render_snapshot(snapshot)


def test_snapshot_fingerprint_ignores_clock_but_not_stale_transition() -> None:
    baseline = {
        "run_directory": "/tmp/run",
        "revision": "same-artifacts",
        "status": "STARTING",
        "status_detail": "trainer lock is not held",
        "warnings": [],
        "checked_utc": "2026-08-15T00:00:00+00:00",
    }
    later = {
        **baseline,
        "checked_utc": "2026-08-15T00:01:00+00:00",
        "activity": {"age_seconds": 60.0},
    }
    stale = {
        **later,
        "status": "STALE",
        "warnings": ["no artifact update for more than 1m 00s"],
    }

    assert live_monitor._snapshot_fingerprint(baseline) == live_monitor._snapshot_fingerprint(later)
    assert live_monitor._snapshot_fingerprint(later) != live_monitor._snapshot_fingerprint(stale)


def test_old_supervisor_failure_does_not_override_new_locked_resume(tmp_path) -> None:
    run = tmp_path / "20260815-050000-resumed"
    _write_json(
        run / "convergence_supervisor_state.json",
        {
            "status": "initial_trainer_failed",
            "updated_utc": "2026-08-15T00:00:00+00:00",
            "error": "old failure",
        },
    )
    _write_json(
        run / "training_state.json",
        {"state": "running", "updated_utc": "2026-08-15T01:00:00+00:00"},
    )
    lock = _hold_training_lock(run)
    try:
        snapshot = live_monitor.build_snapshot(run)
    finally:
        lock.close()

    assert snapshot["status"] == "TRAINING"
    assert "old failure" not in snapshot["warnings"]


def test_dead_evaluation_pid_is_an_explicit_signal(tmp_path, monkeypatch) -> None:
    run = tmp_path / "20260815-060000-dead-evaluation"
    _write_json(
        run / "evaluation_progress.json",
        {"stage": "anchor_complete", "pid": 999999, "batch": 1, "batches": 2},
    )
    monkeypatch.setattr(live_monitor, "_pid_alive", lambda _pid: False)

    snapshot = live_monitor.build_snapshot(run)

    assert snapshot["status"] == "STALE"
    assert "recorded evaluation process is not alive" in snapshot["warnings"]
    assert "signals   no hard failure" not in live_monitor.render_snapshot(snapshot)


def test_failed_evaluation_progress_is_terminal_and_explains_failure(tmp_path) -> None:
    run = tmp_path / "20260815-065000-failed-evaluation"
    _write_json(
        run / "evaluation_progress.json",
        {
            "stage": "failed",
            "pid": 999999,
            "exception_type": "RuntimeError",
            "message": "forced evaluator failure",
            "last_stage": "anchor_complete",
            "batch": 1,
            "batches": 2,
        },
    )

    snapshot = live_monitor.build_snapshot(run)

    assert snapshot["status"] == "FAILED"
    assert snapshot["terminal"] is True
    assert snapshot["status_detail"] == "RuntimeError"
    assert "forced evaluator failure" in snapshot["warnings"]
    rendered = live_monitor.render_snapshot(snapshot)
    assert "FAILED" in rendered
    assert "forced evaluator failure" in rendered


def test_generic_known_report_is_terminal_instead_of_stale(tmp_path) -> None:
    run = tmp_path / "20260815-070000-hypothesis-report"
    _write_json(run / "report.json", {"candidate_names": ["learned"], "episodes": 1})

    snapshot = live_monitor.build_snapshot(run)

    assert snapshot["status"] == "COMPLETED"
    assert snapshot["status_detail"] == "evaluation report"
    assert snapshot["evaluation_result"]["generic_report"] is True
    assert "report    " in live_monitor.render_snapshot(snapshot)


def test_latest_unsupported_validation_replaces_older_scored_validation(tmp_path) -> None:
    run = tmp_path / "20260815-080000-unsupported"
    _write_metrics(
        run / "metrics.jsonl",
        [
            {
                "step": 16,
                "split": "validation",
                "validation_rollout_selection_score": 0.3,
                "selection_metric_supported": 1.0,
                "selection_accepted": 1.0,
            },
            {
                "step": 32,
                "split": "validation_recovery",
                "selection_metric_supported": 0.0,
                "selection_accepted": 0.0,
                "selection_rejection_reason_count": 1.0,
            },
        ],
    )

    snapshot = live_monitor.build_snapshot(run)

    validation = snapshot["training"]["validation"]
    assert validation["step"] == 32
    assert validation["selection_score"] is None
    assert validation["supported"] == 0.0
    assert "latest validation lacks required selection support" in snapshot["warnings"]
    assert "UNSUPPORTED" in live_monitor.render_snapshot(snapshot)


def test_measurement_validation_is_visible_with_incumbent_comparison(tmp_path) -> None:
    run = tmp_path / "20260815-090000-measurement"
    _write_metrics(
        run / "metrics.jsonl",
        [
            {
                "step": 64,
                "split": "validation",
                "validation_measurement_selection_score": 0.21,
                "best_measurement_selection_score": 0.20,
                "best_measurement_checkpoint_step": 32.0,
                "measurement_selection_usable": 1.0,
                "measurement_selection_accepted": 0.0,
                "measurement_selection_rejection_reason_count": 2.0,
                "validation_runtime_birth_world_position_mae_m": 0.18,
                "validation_runtime_birth_f1_at_0_5m": 0.72,
                "validation_fast_roi_f1_at_0_5m": 0.81,
            }
        ],
    )

    snapshot = live_monitor.build_snapshot(run)

    validation = snapshot["training"]["validation"]
    assert validation["kind"] == "measurement"
    assert validation["score_delta"] == pytest.approx(0.01)
    rendered = live_monitor.render_snapshot(snapshot)
    assert "val       measurement" in rendered
    assert "incumbent score 0.2 @ step 32" in rendered


def test_support_collapse_rollback_is_a_hard_signal(tmp_path) -> None:
    run = tmp_path / "20260815-100000-collapse"
    _write_metrics(
        run / "metrics.jsonl",
        [
            {
                "step": 128,
                "split": "training_control_support_collapse",
                "support_collapse_rollback_applied": 1.0,
                "support_collapse_failure_count": 3.0,
            }
        ],
    )

    snapshot = live_monitor.build_snapshot(run)

    assert (
        "training support collapse triggered incumbent rollback with 3 support failures"
        in (snapshot["warnings"])
    )


def test_live_evaluation_keeps_stable_context_on_anchor(tmp_path, monkeypatch) -> None:
    run = tmp_path / "20260815-110000-live-evaluation"
    _write_json(
        run / "evaluation_progress.json",
        {
            "stage": "anchor_complete",
            "pid": 12345,
            "split": "validation",
            "device": "mps",
            "precision": "float32",
            "episodes": 4,
            "evaluated_episodes": 0,
            "batch": 1,
            "batches": 4,
            "frame": 5,
            "total_frames": 40,
        },
    )
    monkeypatch.setattr(live_monitor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(live_monitor, "_pid_command_matches_evaluator", lambda _pid: True)

    snapshot = live_monitor.build_snapshot(run)

    assert snapshot["status"] == "EVALUATING"
    assert snapshot["device"]["device"] == "mps"
    rendered = live_monitor.render_snapshot(snapshot)
    assert "evaluate  anchor_complete · validation · batch 1/4 · frame 5/40 · episodes 0/4" in (
        rendered
    )


@pytest.mark.parametrize(
    ("campaign_status", "expected_status"),
    [("plateau", "CONVERGED"), ("limit_hit", "LIMIT HIT")],
)
def test_convergence_campaign_outcome_is_explicit(
    tmp_path,
    campaign_status,
    expected_status,
) -> None:
    run = tmp_path / f"20260815-120000-{campaign_status}"
    _write_json(
        run / "training_state.json",
        {"state": "completed", "completed_steps": 1024},
    )
    _write_json(
        run / "convergence_report.json",
        {
            "status": campaign_status,
            "reason": "declared decision",
            "completed_steps": 1024,
            "best_step": 768,
        },
    )

    snapshot = live_monitor.build_snapshot(run)

    assert snapshot["status"] == expected_status
    assert "campaign  completed 1024 steps · best step 768" in live_monitor.render_snapshot(
        snapshot
    )
    if campaign_status == "limit_hit":
        assert any("without plateau" in warning for warning in snapshot["warnings"])


@pytest.mark.parametrize(
    ("option", "value"),
    [("--interval", "nan"), ("--interval", "inf"), ("--stale-after", "nan")],
)
def test_nonfinite_monitor_intervals_are_rejected(option, value) -> None:
    with pytest.raises(SystemExit):
        live_monitor.parse_args([option, value])
