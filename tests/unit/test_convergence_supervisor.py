from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts import supervise_convergence
from scripts.supervise_convergence import (
    _acquire_supervisor_lock,
    _record_external_trainer_failure,
    _wait_for_completed_segment,
    parse_args,
)
from world_model.datasets import make_seed_manifest
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import save_checkpoint
from world_model.training.convergence import (
    CampaignIncompleteError,
    CampaignInspection,
    ValidationCandidate,
    decide_continuation,
    inspect_completed_campaign,
)
from world_model.training.trainer import (
    _ROLLOUT_SELECTION_METRIC_VERSION,
    _current_model_state_hash,
    _rollout_selection_metrics,
    _selection_horizon_keys,
    _selection_scenario_slugs,
    _validation_protocol_checkpoint_metrics,
)
from world_model.utils.config import OrpheusConfig, load_config


def _inspection(
    completed_steps: int,
    candidates: list[tuple[int, float, bool]],
    *,
    support_failed_steps: set[int] | None = None,
) -> CampaignInspection:
    support_failed_steps = support_failed_steps or set()
    validation_candidates = tuple(
        ValidationCandidate(
            step=step,
            score=score,
            accepted=accepted,
            training_support_passed=step not in support_failed_steps,
            model_state_hash=f"hash-{step}",
            checkpoint_path=f"/run/checkpoints/validation_step_{step:06d}.pt",
        )
        for step, score, accepted in candidates
    )
    accepted_candidates = tuple(
        candidate for candidate in validation_candidates if candidate.accepted
    )
    best = min(accepted_candidates, key=lambda candidate: candidate.score)
    return CampaignInspection(
        run_directory="/run",
        completed_steps=completed_steps,
        protocol_hash="protocol",
        best_step=best.step,
        best_score=best.score,
        reference_step=validation_candidates[0].step,
        validation_candidates=validation_candidates,
    )


def _decision(
    inspection: CampaignInspection,
    *,
    maximum_total_steps: int = 24576,
):
    return decide_continuation(
        inspection,
        minimum_total_steps=12288,
        extension_steps=4096,
        tail_steps=1024,
        minimum_relative_gain=0.01,
        maximum_total_steps=maximum_total_steps,
    )


def test_recent_safe_one_percent_gain_extends_a_complete_block() -> None:
    decision = _decision(
        _inspection(
            12288,
            [
                (0, 0.80, True),
                (10752, 0.70, True),
                (11776, 0.68, True),
            ],
        )
    )

    assert decision.status == "continue"
    assert decision.next_total_steps == 16384
    assert decision.tail_best_step == 11776
    assert decision.relative_tail_gain is not None
    assert decision.relative_tail_gain > 0.01


def test_gain_outside_final_tail_is_not_enough_plateau_evidence() -> None:
    decision = _decision(
        _inspection(
            12288,
            [
                (0, 0.80, True),
                (10752, 0.68, True),
                (11264, 0.681, False),
                (11776, 0.682, False),
                (12288, 0.683, False),
            ],
        )
    )

    assert decision.status == "continue"
    assert decision.next_total_steps == 16384
    assert "incomplete or contradictory" in decision.reason


def test_four_consecutive_rejections_with_subthreshold_gain_are_a_plateau() -> None:
    decision = _decision(
        _inspection(
            16384,
            [
                (0, 0.80, True),
                (14336, 0.6800, True),
                (14848, 0.6790, False),
                (15360, 0.6780, False),
                (15872, 0.6770, False),
                (16384, 0.6760, False),
            ],
        )
    )

    assert decision.status == "plateau"
    assert decision.optimization_plateau_reached
    assert decision.trainer_stop_recommended
    assert not decision.comprehensive_promotion_eligible
    assert not decision.converged
    assert decision.plateau_primary_gain is not None
    assert decision.plateau_primary_gain < 0.01
    assert decision.plateau_candidate_accepted == (False, False, False, False)


def test_supported_rejected_raw_gain_prevents_premature_plateau() -> None:
    decision = _decision(
        _inspection(
            16384,
            [
                (0, 0.80, True),
                (14336, 0.68, True),
                (14848, 0.65, False),
                (15360, 0.64, False),
                (15872, 0.63, False),
                (16384, 0.62, False),
            ],
        )
    )

    assert decision.status == "continue"
    assert decision.plateau_primary_gain is not None
    assert decision.plateau_primary_gain > 0.01


def test_support_failed_conditional_score_collapse_is_not_convergence_progress() -> None:
    decision = _decision(
        _inspection(
            16384,
            [
                (0, 0.80, True),
                (14336, 0.68, True),
                # These candidates look dramatically better only because
                # their broad guardrails rejected them.
                (14848, 0.30, False),
                (15360, 0.25, False),
                (15872, 0.20, False),
                (16384, 0.15, False),
            ],
            support_failed_steps={14848, 15360, 15872, 16384},
        )
    )

    assert decision.status == "continue"
    assert decision.plateau_primary_gain is None
    assert decision.plateau_candidate_accepted == (False, False, False, False)
    assert decision.plateau_candidate_training_support_passed == (
        False,
        False,
        False,
        False,
    )


def test_plateau_gain_uses_supported_candidates_only() -> None:
    decision = _decision(
        _inspection(
            16384,
            [
                (0, 0.80, True),
                (14336, 0.68, True),
                (14848, 0.10, False),
                (15360, 0.679, False),
                (15872, 0.678, False),
                (16384, 0.677, False),
            ],
            support_failed_steps={14848},
        )
    )

    assert decision.status == "continue"
    assert decision.plateau_primary_gain is not None
    assert decision.plateau_primary_gain < 0.01
    assert decision.plateau_candidate_training_support_passed == (
        False,
        True,
        True,
        True,
    )


def test_accepted_candidate_at_tail_start_prevents_premature_plateau() -> None:
    decision = _decision(
        _inspection(
            12288,
            [
                (0, 0.80, True),
                (10240, 0.70, True),
                (10752, 0.699, False),
                (11264, 0.695, True),
                (11776, 0.696, False),
                (12288, 0.697, False),
            ],
        )
    )

    assert decision.status == "continue"
    assert decision.next_total_steps == 16384
    assert decision.plateau_candidate_accepted == (False, True, False, False)


def test_hard_limit_is_not_mislabeled_as_convergence() -> None:
    decision = _decision(
        _inspection(
            24576,
            [
                (0, 0.80, True),
                (23040, 0.65, True),
                (24064, 0.62, True),
            ],
        )
    )

    assert decision.status == "limit_hit"
    assert decision.next_total_steps is None
    assert "not an objective-convergence claim" in decision.reason


def test_plateau_at_hard_limit_is_still_reported_as_plateau() -> None:
    decision = _decision(
        _inspection(
            24576,
            [
                (0, 0.80, True),
                (22528, 0.6800, True),
                (23040, 0.6790, False),
                (23552, 0.6780, False),
                (24064, 0.6770, False),
                (24576, 0.6760, False),
            ],
        )
    )

    assert decision.status == "plateau"
    assert decision.next_total_steps is None


def test_plateau_report_stops_optimization_without_claiming_full_convergence(
    tmp_path,
) -> None:
    decision = _decision(
        _inspection(
            16384,
            [
                (0, 0.80, True),
                (14336, 0.6800, True),
                (14848, 0.6790, False),
                (15360, 0.6780, False),
                (15872, 0.6770, False),
                (16384, 0.6760, False),
            ],
        )
    )
    report_path = tmp_path / "convergence_report.json"

    supervise_convergence._write_report(
        report_path,
        decision,
        config_path=tmp_path / "config.yaml",
        run_directory=tmp_path / "run",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "plateau"
    assert report["optimization_plateau_reached"] is True
    assert report["trainer_stop_recommended"] is True
    assert report["latency_guardrail_supported"] is False
    assert report["comprehensive_promotion_eligible"] is False
    assert report["converged"] is False


def test_minimum_causal_coverage_must_finish_before_a_decision() -> None:
    decision = _decision(_inspection(11264, [(0, 0.80, True), (10752, 0.70, True)]))

    assert decision.status == "incomplete"
    assert decision.next_total_steps is None


def test_initial_pid_argument_is_parsed(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "supervise_convergence.py",
            "--config",
            "config.yaml",
            "--run",
            "run",
            "--initial-trainer-pid",
            "37360",
        ],
    )

    assert parse_args().initial_trainer_pid == 37360


def test_initial_wait_fails_when_monitored_trainer_disappears(
    tmp_path,
    monkeypatch,
) -> None:
    def missing_process(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("scripts.supervise_convergence.os.kill", missing_process)

    with pytest.raises(CampaignIncompleteError, match="exited before step 12288"):
        _wait_for_completed_segment(
            tmp_path / "run",
            config=load_config("configs/tiny_overfit.yaml"),
            minimum_expected_steps=12288,
            poll_seconds=0.001,
            event_path=tmp_path / "events.jsonl",
            monitored_pid=37360,
        )


def test_initial_wait_surfaces_terminal_trainer_failure_without_pid(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "training_state.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "exception_type": "RuntimeError",
                "message": "causal support exhausted",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        CampaignIncompleteError,
        match=("training_state.json: RuntimeError: causal support exhausted"),
    ):
        _wait_for_completed_segment(
            run,
            config=load_config("configs/tiny_overfit.yaml"),
            minimum_expected_steps=12288,
            poll_seconds=0.001,
            event_path=tmp_path / "events.jsonl",
        )


def test_initial_wait_prefers_terminal_failure_artifact_over_live_pid(
    tmp_path,
    monkeypatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "training_failure.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "exception_type": "KeyboardInterrupt",
                "message": "operator interrupted training",
            }
        ),
        encoding="utf-8",
    )
    process_probes = 0

    def record_process_probe(_pid: int, _signal: int) -> None:
        nonlocal process_probes
        process_probes += 1

    monkeypatch.setattr(
        "scripts.supervise_convergence.os.kill",
        record_process_probe,
    )

    with pytest.raises(
        CampaignIncompleteError,
        match=("training_failure.json: KeyboardInterrupt: operator interrupted training"),
    ):
        _wait_for_completed_segment(
            run,
            config=load_config("configs/tiny_overfit.yaml"),
            minimum_expected_steps=12288,
            poll_seconds=0.001,
            event_path=tmp_path / "events.jsonl",
            monitored_pid=37360,
        )
    assert process_probes == 0


def test_initial_wait_rejects_reused_unrelated_pid(
    tmp_path,
    monkeypatch,
) -> None:
    def live_process(_pid: int, _signal: int) -> None:
        return None

    def unrelated_process(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["ps"],
            returncode=0,
            stdout="/usr/bin/python /tmp/unrelated.py\n",
            stderr="",
        )

    monkeypatch.setattr(
        "scripts.supervise_convergence.os.kill",
        live_process,
    )
    monkeypatch.setattr(
        "scripts.supervise_convergence.subprocess.run",
        unrelated_process,
    )

    run = tmp_path / "run"
    with pytest.raises(
        CampaignIncompleteError,
        match="no longer identifies the trainer",
    ):
        _wait_for_completed_segment(
            run,
            config=load_config("configs/tiny_overfit.yaml"),
            minimum_expected_steps=12288,
            poll_seconds=0.001,
            event_path=tmp_path / "events.jsonl",
            monitored_pid=37360,
        )


def test_initial_wait_accepts_matching_fresh_run_pid_until_artifacts_complete(
    tmp_path,
    monkeypatch,
) -> None:
    run = tmp_path / "20260803-120000-qualification"
    run.mkdir()
    process_probes = 0

    def live_then_exit(_pid: int, _signal: int) -> None:
        nonlocal process_probes
        process_probes += 1
        if process_probes > 1:
            raise ProcessLookupError

    def matching_process(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["ps"],
            returncode=0,
            stdout=(
                "/usr/bin/python /repo/train.py --config /repo/config.yaml "
                "--run-name qualification\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "scripts.supervise_convergence.os.kill",
        live_then_exit,
    )
    monkeypatch.setattr(
        "scripts.supervise_convergence.subprocess.run",
        matching_process,
    )
    monkeypatch.setattr(
        "scripts.supervise_convergence.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(CampaignIncompleteError, match="exited before step 12288"):
        _wait_for_completed_segment(
            run,
            config=load_config("configs/tiny_overfit.yaml"),
            minimum_expected_steps=12288,
            poll_seconds=0.001,
            event_path=tmp_path / "events.jsonl",
            monitored_pid=37360,
        )
    assert process_probes == 2


def test_main_records_initial_trainer_failure(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    arguments = argparse.Namespace(
        config="configs/sustained_accuracy_mps.yaml",
        run=str(run),
        device="mps",
        initial_launchctl_label=None,
        initial_trainer_pid=37360,
        poll_seconds=0.001,
        minimum_total_steps=None,
        extension_steps=4096,
        tail_steps=1024,
        minimum_relative_gain=0.01,
        maximum_total_steps=24576,
        inspect_only=False,
    )

    def fail_wait(*_args, **_kwargs):
        raise CampaignIncompleteError("trainer PID 37360 exited before step 12288")

    monkeypatch.setattr(supervise_convergence, "parse_args", lambda: arguments)
    monkeypatch.setattr(supervise_convergence, "_wait_for_completed_segment", fail_wait)

    assert supervise_convergence.main() == 2
    state = json.loads((run / "convergence_supervisor_state.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (run / "convergence_supervisor.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert state["status"] == "initial_trainer_failed"
    assert state["trainer_pid"] == 37360
    assert events[-1]["event"] == "initial_segment_failed"


def test_main_boots_out_failed_initial_keepalive_job(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    arguments = argparse.Namespace(
        config="configs/sustained_accuracy_mps.yaml",
        run=str(run),
        device="mps",
        initial_launchctl_label="com.example.failed-trainer",
        initial_trainer_pid=37360,
        poll_seconds=0.001,
        minimum_total_steps=None,
        extension_steps=4096,
        tail_steps=1024,
        minimum_relative_gain=0.01,
        maximum_total_steps=24576,
        inspect_only=False,
    )

    def fail_wait(*_args, **_kwargs):
        raise CampaignIncompleteError("trainer PID 37360 exited before step 12288")

    bootout_calls: list[tuple[str, str]] = []

    def record_bootout(label: str, *, event_path: Path, outcome: str) -> None:
        assert event_path == run / "convergence_supervisor.jsonl"
        bootout_calls.append((label, outcome))

    monkeypatch.setattr(supervise_convergence, "parse_args", lambda: arguments)
    monkeypatch.setattr(supervise_convergence, "_wait_for_completed_segment", fail_wait)
    monkeypatch.setattr(supervise_convergence, "_bootout_initial_job", record_bootout)

    assert supervise_convergence.main() == 2
    assert bootout_calls == [("com.example.failed-trainer", "failure")]


def test_initial_cleanup_failure_cannot_mask_durable_trainer_failure(
    tmp_path,
    monkeypatch,
) -> None:
    run = tmp_path / "run"
    arguments = argparse.Namespace(
        config="configs/sustained_accuracy_mps.yaml",
        run=str(run),
        device="mps",
        initial_launchctl_label="com.example.stuck-trainer",
        initial_trainer_pid=37360,
        poll_seconds=0.001,
        minimum_total_steps=None,
        extension_steps=4096,
        tail_steps=1024,
        minimum_relative_gain=0.01,
        maximum_total_steps=24576,
        inspect_only=False,
    )

    def fail_wait(*_args, **_kwargs):
        raise CampaignIncompleteError("original trainer failure")

    def fail_bootout(*_args, **_kwargs):
        raise RuntimeError("launchctl refused cleanup")

    monkeypatch.setattr(supervise_convergence, "parse_args", lambda: arguments)
    monkeypatch.setattr(supervise_convergence, "_wait_for_completed_segment", fail_wait)
    monkeypatch.setattr(supervise_convergence, "_bootout_initial_job", fail_bootout)

    assert supervise_convergence.main() == 2

    state = json.loads((run / "convergence_supervisor_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "initial_trainer_failed"
    assert state["error"] == "original trainer failure"
    assert state["cleanup_error"] == "launchctl refused cleanup"
    events = [
        json.loads(line)
        for line in (run / "convergence_supervisor.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = [event["event"] for event in events]
    assert names.index("initial_segment_failed") < names.index("initial_job_bootout_failed")


def test_external_trainer_exit_updates_the_primary_training_state(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    running = {
        "state": "running",
        "updated_utc": "2026-08-06T00:00:00+00:00",
        "run_directory": str(run),
    }
    (run / "training_state.json").write_text(json.dumps(running), encoding="utf-8")

    _record_external_trainer_failure(
        run,
        trainer_pid=4321,
        target_steps=12288,
        message="trainer was killed by the operating system",
    )

    state = json.loads((run / "training_state.json").read_text(encoding="utf-8"))
    failure = json.loads((run / "training_failure.json").read_text(encoding="utf-8"))
    history = [
        json.loads(line)
        for line in (run / "training_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert state == failure == history[0]
    assert state["state"] == "failed"
    assert state["exception_type"] == "ExternalTrainerExit"
    assert state["trainer_pid"] == 4321
    assert state["target_steps"] == 12288
    assert state["previous_state"] == running


def test_supervisor_lock_rejects_a_second_owner(tmp_path) -> None:
    path = tmp_path / "supervisor.lock"
    first_owner = _acquire_supervisor_lock(path)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            _acquire_supervisor_lock(path)
    finally:
        first_owner.close()
    next_owner = _acquire_supervisor_lock(path)
    next_owner.close()


def _physical_metrics(config: OrpheusConfig, *, scale: float):
    position_rmse = 0.4 * scale
    metrics = {
        "selection_metric_supported": 1.0,
        "validation_position_rmse_m": position_rmse,
        "validation_velocity_rmse_mps": 0.8,
        "validation_target_coverage": 0.9,
        "validation_prediction_precision": 0.9,
        "validation_collision_f1": 0.6,
        "validation_id_switch_rate": 0.0,
        "validation_position_coverage90": 0.9,
        "validation_current_position_coverage90": 0.9,
        "validation_current_position_gaussian_nll": 0.1,
        "validation_current_position_sharpness_std": 1.0,
        "physical_state_position_sse": position_rmse**2 * 300.0,
        "physical_state_position_coordinate_count": 300.0,
        "physical_state_velocity_sse": 0.8**2 * 300.0,
        "physical_state_velocity_coordinate_count": 300.0,
        "physical_distance_gated_matched_object_frames": 90.0,
        "physical_distance_gated_target_object_frames": 100.0,
        "physical_distance_gated_predicted_object_frames": 100.0,
        "physical_distance_gated_identity_switches": 0.0,
        "physical_distance_gated_object_frame_associations": 100.0,
        "physical_position_coverage90_hit_count": 270.0,
        "physical_position_coverage90_coordinate_count": 300.0,
        "physical_state_position_coverage90_hit_count": 270.0,
        "physical_state_position_coverage90_coordinate_count": 300.0,
        "physical_state_position_gaussian_nll_sum": 30.0,
        "physical_state_position_sharpness_std_sum": 300.0,
        "physical_state_position_calibration_coordinate_count": 300.0,
        "physical_collision_true_positive_count": 3.0,
        "physical_collision_false_positive_count": 2.0,
        "physical_collision_false_negative_count": 2.0,
    }
    for axis in ("x", "y", "z"):
        metrics[f"validation_position_rmse_{axis}_m"] = position_rmse
        metrics[f"validation_velocity_rmse_{axis}_mps"] = 0.8
        metrics[f"validation_current_position_gaussian_nll_{axis}"] = 0.1
        metrics[f"validation_current_position_sharpness_std_{axis}"] = 1.0
        metrics[f"physical_state_position_{axis}_sse"] = position_rmse**2 * 100.0
        metrics[f"physical_state_position_{axis}_coordinate_count"] = 100.0
        metrics[f"physical_state_velocity_{axis}_sse"] = 0.8**2 * 100.0
        metrics[f"physical_state_velocity_{axis}_coordinate_count"] = 100.0
        metrics[f"physical_state_position_{axis}_gaussian_nll_sum"] = 10.0
        metrics[f"physical_state_position_{axis}_sharpness_std_sum"] = 100.0
        metrics[f"physical_state_position_{axis}_calibration_coordinate_count"] = 100.0
    horizon_entries = tuple(_selection_horizon_keys(config))
    for index, (suffix, _) in enumerate(horizon_entries):
        horizon_rmse = scale * (0.4 - 0.05 * index)
        metrics[f"validation_position_rmse@{suffix}"] = horizon_rmse
        metrics[f"validation_forecast_target_coverage@{suffix}"] = 0.9
        metrics[f"validation_velocity_rmse@{suffix}"] = 0.8
        metrics[f"validation_collision_f1@{suffix}"] = 0.6
        metrics[f"validation_forecast_identity_association_coverage@{suffix}"] = 1.0
        metrics[f"validation_forecast_identity_mismatch_rate@{suffix}"] = 0.0
        metrics[f"validation_position_coverage90@{suffix}"] = 0.9
        metrics[f"validation_position_gaussian_nll@{suffix}"] = 0.1
        metrics[f"validation_position_sharpness_std@{suffix}"] = 1.0
        metrics[f"physical_rollout_position@{suffix}_sse"] = horizon_rmse**2 * 30.0
        metrics[f"physical_forecast_predictable_target_count@{suffix}"] = 10.0
        metrics[f"physical_rollout_position@{suffix}_coordinate_count"] = 30.0
        metrics[f"physical_rollout_velocity@{suffix}_sse"] = 0.8**2 * 30.0
        metrics[f"physical_rollout_velocity@{suffix}_coordinate_count"] = 30.0
        metrics[f"physical_rollout_position@{suffix}_gaussian_nll_sum"] = 3.0
        metrics[f"physical_rollout_position@{suffix}_sharpness_std_sum"] = 30.0
        metrics[f"physical_rollout_position@{suffix}_calibration_coordinate_count"] = 30.0
        metrics[f"physical_rollout_position_coverage90@{suffix}_hit_count"] = 27.0
        metrics[f"physical_rollout_position_coverage90@{suffix}_coordinate_count"] = 30.0
        metrics[f"physical_forecast_active_count@{suffix}"] = 9.0
        metrics[f"physical_forecast_tracked_count@{suffix}"] = 10.0
        metrics[f"physical_forecast_target_count@{suffix}"] = 10.0
        metrics[f"physical_rollout_predictable_target_count@{suffix}"] = 10.0
        metrics[f"physical_rollout_censored_external_actuation_count@{suffix}"] = 0.0
        metrics[f"physical_forecast_identity_mismatch_count@{suffix}"] = 0.0
        metrics[f"physical_forecast_identity_eligible_count@{suffix}"] = 9.0
        metrics[f"physical_forecast_identity_association_count@{suffix}"] = 9.0
        metrics[f"physical_collision_true_positive_count@{suffix}"] = 3.0
        metrics[f"physical_collision_false_positive_count@{suffix}"] = 2.0
        metrics[f"physical_collision_false_negative_count@{suffix}"] = 2.0
        metrics[f"physical_collision_true_negative_count@{suffix}"] = 3.0
        for axis in ("x", "y", "z"):
            metrics[f"validation_position_rmse_{axis}@{suffix}"] = horizon_rmse
            metrics[f"validation_velocity_rmse_{axis}@{suffix}"] = 0.8
            metrics[f"validation_position_gaussian_nll_{axis}@{suffix}"] = 0.1
            metrics[f"validation_position_sharpness_std_{axis}@{suffix}"] = 1.0
            metrics[f"physical_rollout_position_{axis}@{suffix}_sse"] = horizon_rmse**2 * 10.0
            metrics[f"physical_rollout_position_{axis}@{suffix}_coordinate_count"] = 10.0
            metrics[f"physical_rollout_velocity_{axis}@{suffix}_sse"] = 0.8**2 * 10.0
            metrics[f"physical_rollout_velocity_{axis}@{suffix}_coordinate_count"] = 10.0
            metrics[f"physical_rollout_position_{axis}@{suffix}_gaussian_nll_sum"] = 1.0
            metrics[f"physical_rollout_position_{axis}@{suffix}_sharpness_std_sum"] = 10.0
            metrics[f"physical_rollout_position_{axis}@{suffix}_calibration_coordinate_count"] = (
                10.0
            )
    horizon_count = float(len(horizon_entries))
    metrics["physical_position_coverage90_hit_count"] = 27.0 * horizon_count
    metrics["physical_position_coverage90_coordinate_count"] = 30.0 * horizon_count
    metrics["physical_collision_true_positive_count"] = 3.0 * horizon_count
    metrics["physical_collision_false_positive_count"] = 2.0 * horizon_count
    metrics["physical_collision_false_negative_count"] = 2.0 * horizon_count
    metrics["physical_collision_true_negative_count"] = 3.0 * horizon_count
    base = dict(metrics)
    manifest = make_seed_manifest("validation", config.training.validation_episodes)
    scenario_slugs = _selection_scenario_slugs(config)
    scenario_counts = {scenario: 0 for scenario in scenario_slugs}
    for seed in manifest.seeds:
        scenario_counts[scenario_slugs[int(seed) % len(scenario_slugs)]] += 1
    for scenario in scenario_slugs:
        prefix = f"scenario_{scenario}_"
        episode_count = float(scenario_counts[scenario])
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


def _selector_metrics(
    config: OrpheusConfig,
    *,
    model_hash: str,
    candidate_scale: float,
    best_scale: float,
    checkpoint_step: int,
    best_step: int,
    reference_step: int,
    accepted: bool = True,
):
    candidate = _rollout_selection_metrics(
        _physical_metrics(config, scale=candidate_scale),
        config,
        require_scenarios=True,
    )
    best = _rollout_selection_metrics(
        _physical_metrics(config, scale=best_scale),
        config,
        require_scenarios=True,
    )
    reference = _rollout_selection_metrics(
        _physical_metrics(config, scale=1.0),
        config,
        require_scenarios=True,
    )
    return {
        **_physical_metrics(config, scale=candidate_scale),
        "latency_guardrail_supported": 0.0,
        "latency_guardrail_passed": 0.0,
        "comprehensive_promotion_eligible": 0.0,
        "selection_scope": "fixed_physical_incumbent_not_comprehensive_promotion",
        "selection_accepted": float(accepted),
        "selection_training_support_required": 1.0,
        "selection_training_support_passed": 1.0,
        "selection_mutable_training_support_passed": 1.0,
        "best_rollout_validated": 1.0,
        "rollout_reference_validated": 1.0,
        "incomplete_reference_comparison_required": 0.0,
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        **candidate.validation_metrics(),
        **best.checkpoint_metrics(),
        **reference.checkpoint_metrics(prefix="reference_rollout"),
        **_validation_protocol_checkpoint_metrics(config),
        "checkpoint_model_state_hash": model_hash,
        "checkpoint_contains_best_rollout_weights": float(checkpoint_step == best_step),
        "best_rollout_model_state_hash": model_hash,
        "best_rollout_checkpoint_step": float(best_step),
        "checkpoint_contains_reference_rollout_weights": float(checkpoint_step == reference_step),
        "reference_rollout_model_state_hash": model_hash,
        "reference_rollout_checkpoint_step": float(reference_step),
    }


def _write_completed_campaign(tmp_path):
    source_config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source_config,
        training=replace(
            source_config.training,
            steps=12288,
            rgb_pretrain_steps=8192,
            eval_every=512,
        ),
    )
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    model_hash = _current_model_state_hash(model)
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    reference_metrics = _selector_metrics(
        config,
        model_hash=model_hash,
        candidate_scale=1.0,
        best_scale=1.0,
        checkpoint_step=0,
        best_step=0,
        reference_step=0,
    )
    best_metrics = _selector_metrics(
        config,
        model_hash=model_hash,
        candidate_scale=0.8,
        best_scale=0.8,
        checkpoint_step=11776,
        best_step=11776,
        reference_step=0,
    )
    rejected_metrics = _selector_metrics(
        config,
        model_hash=model_hash,
        candidate_scale=0.99,
        best_scale=1.0,
        checkpoint_step=11264,
        best_step=0,
        reference_step=0,
        accepted=False,
    )
    last_metrics = {
        key: value
        for key, value in best_metrics.items()
        if not key.startswith("validation_") and key != "selection_accepted"
    }
    for name, step, metrics in (
        ("reference_rollout.pt", 0, reference_metrics),
        ("validation_step_000000.pt", 0, reference_metrics),
        ("validation_step_011264.pt", 11264, rejected_metrics),
        ("best_rollout.pt", 11776, best_metrics),
        ("validation_step_011776.pt", 11776, best_metrics),
        ("last.pt", 12288, last_metrics),
    ):
        save_checkpoint(
            checkpoints / name,
            model=model,
            optimizer=optimizer,
            config=config,
            step=step,
            metrics=metrics,
        )
    summary = {
        "completed_steps": 12288,
        "best_rollout_validated": True,
        "best_rollout_loss": _rollout_selection_metrics(
            _physical_metrics(config, scale=0.8),
            config,
        ).score,
    }
    (run / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run, config


def test_completed_campaign_requires_linked_numbered_selector_provenance(tmp_path) -> None:
    run, config = _write_completed_campaign(tmp_path)

    inspection = inspect_completed_campaign(run, config)

    assert inspection.completed_steps == 12288
    assert inspection.best_step == 11776
    assert not inspection.latency_guardrail_supported
    assert not inspection.latency_guardrail_passed
    assert not inspection.comprehensive_promotion_eligible
    assert [item.step for item in inspection.accepted_validations] == [0, 11776]
    assert [item.step for item in inspection.validation_candidates] == [0, 11264, 11776]
    assert inspection.validation_candidates[1].accepted is False


def test_completed_campaign_rejects_tampered_numbered_tensor_hash(tmp_path) -> None:
    run, config = _write_completed_campaign(tmp_path)
    path = run / "checkpoints" / "validation_step_011776.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metrics"]["checkpoint_model_state_hash"] = "0" * 64
    torch.save(payload, path)

    with pytest.raises(ValueError, match="tensor hash mismatch"):
        inspect_completed_campaign(run, config)


def test_completed_campaign_rejects_comprehensive_claim_without_latency_gate(
    tmp_path,
) -> None:
    run, config = _write_completed_campaign(tmp_path)
    last_path = run / "checkpoints" / "last.pt"
    payload = torch.load(last_path, map_location="cpu", weights_only=False)
    payload["metrics"]["comprehensive_promotion_eligible"] = 1.0
    torch.save(payload, last_path)

    with pytest.raises(ValueError, match="requires a passed paired latency"):
        inspect_completed_campaign(run, config)
