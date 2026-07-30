from __future__ import annotations

import argparse
import json
from dataclasses import replace

import pytest
import torch

from scripts import supervise_convergence
from scripts.supervise_convergence import (
    _acquire_supervisor_lock,
    _wait_for_completed_segment,
    parse_args,
)
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
    _current_model_state_hash,
    _rollout_selection_metrics,
    _selection_horizon_keys,
    _validation_protocol_checkpoint_metrics,
)
from world_model.utils.config import OrpheusConfig, load_config


def _inspection(
    completed_steps: int,
    candidates: list[tuple[int, float, bool]],
) -> CampaignInspection:
    validation_candidates = tuple(
        ValidationCandidate(
            step=step,
            score=score,
            accepted=accepted,
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
    assert decision.plateau_primary_gain is not None
    assert decision.plateau_primary_gain < 0.01
    assert decision.plateau_candidate_accepted == (False, False, False, False)


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
    metrics = {
        "validation_position_rmse_m": 0.4 * scale,
        "validation_velocity_rmse_mps": 0.8,
        "validation_target_coverage": 0.9,
        "validation_prediction_precision": 0.9,
        "validation_collision_f1": 0.6,
        "validation_id_switch_rate": 0.0,
        "validation_position_coverage90": 0.9,
    }
    for index, (suffix, _) in enumerate(_selection_horizon_keys(config)):
        metrics[f"validation_position_rmse@{suffix}"] = scale * (0.4 - 0.05 * index)
        metrics[f"validation_forecast_target_coverage@{suffix}"] = 0.9
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
    )
    best = _rollout_selection_metrics(_physical_metrics(config, scale=best_scale), config)
    reference = _rollout_selection_metrics(_physical_metrics(config, scale=1.0), config)
    return {
        "selection_accepted": float(accepted),
        "best_rollout_validated": 1.0,
        "rollout_reference_validated": 1.0,
        "rollout_selection_metric_version": 3.0,
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
