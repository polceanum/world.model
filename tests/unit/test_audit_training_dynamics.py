from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_training_dynamics import audit_run


def _record(step: int, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "step": step,
        "split": "train",
        "phase": "closed_loop_rgb",
        "loss_total": 1.5,
        "optimizer_update_applied": 1.0,
        "causal_training_support_present": 1.0,
        "causal_objective_term_support_count": 3.0,
        "causal_trajectory_support_count": 12.0,
        "gradient_norm_pre_clip": 0.5,
        "gradient_total_clip_coefficient": 1.0,
        "closed_loop_scope_state_dynamics_only": 1.0,
        "perception_gradient_norm_pre_clip": 0.0,
        "skipped_no_gradient_batches": 0.0,
        "training_data_draw_step": float(step),
        "process_max_rss_bytes": 1000.0,
        "scenario_names": "baseline,elastic_pairs",
        "physical_current_distance_gated_target_coverage": 0.5,
        "physical_current_distance_gated_prediction_precision": 0.4,
        "physical_distance_gated_identity_switch_rate": 0.1,
        "physical_collision_f1_proxy": 0.25,
        "physical_position_coverage90": 0.9,
        "uncertainty_position_nll": -0.5,
        "loss_rollout_position_x": 0.1,
        "physical_rollout_position_rmse_m@1.000s": 0.3,
        "physical_forecast_target_coverage@1.000s": 0.6,
        "physical_distance_gated_identity_switches": 1.0,
        "physical_distance_gated_object_frame_associations": 10.0,
        "perturbed_updates": 1.0,
    }
    record.update(overrides)
    return record


def _write_metrics(run: Path, records: list[dict[str, object]]) -> None:
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_audit_canonicalizes_replayed_tail_without_double_counting(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(8, process_max_rss_bytes=1200.0),
            _record(8, process_max_rss_bytes=900.0),
            _record(16, gradient_total_clip_coefficient=0.5),
        ],
    )

    report = audit_run(run)

    assert report["status"] == "pass"
    assert report["unique_training_blocks"] == 2
    assert report["optimizer_updates_applied"] == 16
    assert report["logged_optimizer_update_confirmations"] == 2.0
    assert report["training_metric_cadence_sparse"] is True
    assert report["training_metric_step_gaps"] == [8]
    assert any("cadence samples" in warning for warning in report["warnings"])
    assert report["duplicate_rows"] == [
        {
            "split": "train",
            "step": 8,
            "count": 2,
            "equivalent_except_process_fields": True,
        }
    ]
    assert report["clipped_block_count"] == 1
    assert report["process_rss_bytes"] == {
        "minimum": 900.0,
        "latest": 1000.0,
        "maximum": 1000.0,
    }
    diagnostics = report["live_physical_diagnostics"]
    assert diagnostics["physical_position_coverage90"]["median"] == 0.9
    assert diagnostics["rollout_position_loss_by_axis"]["x"]["median"] == 0.1
    assert diagnostics["rollout_position_rmse_by_horizon_m"]["1.000s"]["median"] == 0.3
    assert diagnostics["forecast_target_coverage_by_horizon"]["1.000s"]["median"] == 0.6
    assert diagnostics["identity_by_recovery_perturbation"] == {
        "all": {"blocks": 2, "switches": 2.0, "associations": 20.0, "switch_rate": 0.1},
        "perturbed": {
            "blocks": 2,
            "switches": 2.0,
            "associations": 20.0,
            "switch_rate": 0.1,
        },
        "unperturbed": {
            "blocks": 0,
            "switches": 0,
            "associations": 0,
            "switch_rate": 0.0,
        },
    }


def test_audit_reports_numerical_support_and_scope_failures(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                loss_total=float("nan"),
                optimizer_update_applied=0.0,
                causal_training_support_present=0.0,
                causal_objective_term_support_count=0.0,
                gradient_norm_pre_clip=0.0,
                perception_gradient_norm_pre_clip=0.25,
                training_data_draw_step=9.0,
            )
        ],
    )

    report = audit_run(run)

    assert report["status"] == "fail"
    assert len(report["failures"]) == 7
    assert any("nonfinite metrics" in failure for failure in report["failures"])
    assert any("optimizer update was not applied" in failure for failure in report["failures"])
    assert any("causal training support is absent" in failure for failure in report["failures"])
    assert any("frozen state/dynamics scope" in failure for failure in report["failures"])


def test_audit_warns_about_severe_global_or_interaction_clipping(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                gradient_total_clip_coefficient=0.065,
                interaction_gradient_clip_coefficient=0.035,
            ),
            _record(
                16,
                gradient_total_clip_coefficient=1.0,
                interaction_gradient_clip_coefficient=1.0,
            ),
        ],
    )

    report = audit_run(run)

    assert report["status"] == "pass"
    assert report["severe_clipped_steps"] == [
        {
            "step": 8,
            "total_coefficient": 0.065,
            "interaction_coefficient": 0.035,
        }
    ]
    assert (
        "severe gradient clipping retained less than 10% of at least one "
        "raw parameter-group/update gradient"
    ) in report["warnings"]


def test_audit_rejects_divergent_replayed_tail(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(run, [_record(8, loss_total=1.0), _record(8, loss_total=2.0)])

    report = audit_run(run)

    assert report["status"] == "fail"
    assert any("replay rows differ" in failure for failure in report["failures"])


def test_audit_exposes_validation_axes_horizons_and_guardrails(tmp_path) -> None:
    run = tmp_path / "run"
    validation = {
        "step": 512,
        "split": "validation",
        "validation_rollout_selection_score": 0.25,
        "selection_accepted": 0.0,
        "selection_reference_guardrail_failures_json": json.dumps(["x", "coverage"]),
        "validation_id_switch_rate": 0.125,
        "validation_position_rmse_x_m": 0.2,
        "validation_position_rmse_x@1.000s": 0.4,
        "validation_position_rmse@1.000s": 0.3,
        "validation_forecast_target_coverage@1.000s": 0.75,
    }
    _write_metrics(run, [_record(8), validation])

    report = audit_run(run)
    summary = report["validations"][0]

    assert summary["reference_guardrail_failure_count"] == 2
    assert summary["identity_switch_rate"] == 0.125
    assert summary["current_position_rmse_by_axis_m"]["x"] == 0.2
    assert summary["horizon_position_rmse_m"]["1.000s"] == 0.3
    assert summary["horizon_position_rmse_by_axis_m"]["x"]["1.000s"] == 0.4
    assert summary["horizon_target_coverage"]["1.000s"] == 0.75
