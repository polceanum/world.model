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
    assert report["scenario_draw_counts"] == {"baseline": 2, "elastic_pairs": 2}
    assert report["scenario_draw_counts_scope"] == "logged_metric_rows_only"
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


def test_audit_rejects_durable_terminal_optimizer_failure(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(run, [_record(56)])
    (run / "training_failure.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "exception_type": "InteractionGradientRetentionError",
                "message": "complete interaction gradient retained only 0.085",
                "updated_utc": "2026-08-11T00:00:00+00:00",
                "diagnostics": {
                    "optimizer_step_attempted": 60.0,
                    "interaction_gradient_clip_coefficient": 0.085,
                    "optimizer_update_applied": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    report = audit_run(run)

    assert report["status"] == "fail"
    assert report["terminal_optimizer_failure"] == {
        "exception_type": "InteractionGradientRetentionError",
        "message": "complete interaction gradient retained only 0.085",
        "updated_utc": "2026-08-11T00:00:00+00:00",
        "diagnostics": {
            "optimizer_step_attempted": 60.0,
            "interaction_gradient_clip_coefficient": 0.085,
            "optimizer_update_applied": 0.0,
        },
        "nonfinite_diagnostic_fields": [],
    }
    assert any("terminal optimizer failure" in item for item in report["failures"])


def test_audit_rejects_severe_complete_interaction_clipping(tmp_path) -> None:
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

    assert report["status"] == "fail"
    assert report["severe_clipped_steps"] == [
        {
            "step": 8,
            "total_coefficient": 0.065,
            "interaction_coefficient": 0.035,
        }
    ]
    assert (
        "severe gradient clipping retained less than 10% of at least one "
        "raw typed-output/parameter-group update gradient"
    ) in report["warnings"]
    assert report["uncontained_interaction_clipped_steps"] == [
        {"step": 8, "interaction_stage_coefficient": 0.035}
    ]
    assert any("complete-interaction clipping" in failure for failure in report["failures"])


def test_audit_reports_attention_collision_row_clipping(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                gradient_total_clip_coefficient=0.5,
                interaction_gradient_total_clip_coefficient=0.4,
                attention_collision_gradient_clip_coefficient=0.02,
            )
        ],
    )

    report = audit_run(run)

    assert report["status"] == "pass"
    assert report["severe_clipped_steps"] == [
        {
            "step": 8,
            "total_coefficient": 0.5,
            "interaction_coefficient": 0.4,
            "attention_collision_coefficient": 0.02,
        }
    ]


def test_audit_reports_attention_node_row_clipping(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                gradient_total_clip_coefficient=0.5,
                interaction_gradient_total_clip_coefficient=0.4,
                attention_node_gradient_clip_coefficient=0.02,
            )
        ],
    )

    report = audit_run(run)

    assert report["status"] == "pass"
    assert report["severe_clipped_steps"] == [
        {
            "step": 8,
            "total_coefficient": 0.5,
            "interaction_coefficient": 0.4,
            "attention_node_coefficient": 0.02,
        }
    ]


def test_audit_reports_attention_force_row_clipping(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                gradient_total_clip_coefficient=0.5,
                interaction_gradient_total_clip_coefficient=0.4,
                attention_force_gradient_clip_coefficient=0.02,
            )
        ],
    )

    report = audit_run(run)

    assert report["status"] == "pass"
    assert report["severe_clipped_steps"] == [
        {
            "step": 8,
            "total_coefficient": 0.5,
            "interaction_coefficient": 0.4,
            "attention_force_coefficient": 0.02,
        }
    ]


def test_audit_reports_typed_output_backpropagation_clipping(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                attention_node_output_backprop_gradient_minimum_clip_coefficient=0.25,
                attention_collision_output_backprop_gradient_minimum_clip_coefficient=0.08,
                attention_force_output_backprop_gradient_minimum_clip_coefficient=0.04,
                attention_impulse_gradient_clip_coefficient=0.03,
                attention_impulse_output_backprop_gradient_minimum_clip_coefficient=0.02,
            )
        ],
    )

    report = audit_run(run)

    assert report["status"] == "pass"
    assert report["severe_clipped_steps"] == [
        {
            "step": 8,
            "total_coefficient": 1.0,
            "interaction_coefficient": 1.0,
            "attention_node_output_coefficient": 0.25,
            "attention_collision_output_coefficient": 0.08,
            "attention_force_output_coefficient": 0.04,
            "attention_impulse_coefficient": 0.03,
            "attention_impulse_output_coefficient": 0.02,
        }
    ]


def test_audit_ignores_frozen_node_output_clipping_in_relation_only_scope(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                closed_loop_scope_attention_relation_only=1.0,
                attention_node_gradient_clip_coefficient=0.01,
                attention_node_output_backprop_gradient_minimum_clip_coefficient=0.02,
                attention_force_output_backprop_gradient_minimum_clip_coefficient=0.08,
            )
        ],
    )

    report = audit_run(run)

    assert report["status"] == "pass"
    assert report["severe_clipped_steps"] == [
        {
            "step": 8,
            "total_coefficient": 1.0,
            "interaction_coefficient": 1.0,
            "attention_force_output_coefficient": 0.08,
        }
    ]


def test_audit_rejects_uncontained_complete_interaction_clipping(tmp_path) -> None:
    run = tmp_path / "run"
    _write_metrics(
        run,
        [
            _record(
                8,
                gradient_total_clip_coefficient=0.001,
                interaction_gradient_total_clip_coefficient=0.001,
                interaction_gradient_clip_coefficient=0.002,
                attention_impulse_gradient_clip_coefficient=1.0,
            )
        ],
    )

    report = audit_run(run)

    assert report["status"] == "fail"
    assert report["uncontained_interaction_clipped_steps"] == [
        {"step": 8, "interaction_stage_coefficient": 0.002}
    ]
    assert any("complete-interaction clipping" in failure for failure in report["failures"])


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


def test_audit_pools_complete_and_partial_training_trend_windows(tmp_path) -> None:
    run = tmp_path / "run"
    records = []
    for index, step in enumerate((8, 16, 24), start=1):
        records.append(
            _record(
                step,
                causal_trajectory_support_count=float(index * 10),
                physical_state_position_sse=float(index * index * 3),
                physical_state_position_coordinate_count=3.0,
                physical_state_position_x_sse=float(index * index),
                physical_state_position_x_coordinate_count=1.0,
                physical_state_position_y_sse=float(index * index),
                physical_state_position_y_coordinate_count=1.0,
                physical_state_position_z_sse=float(index * index),
                physical_state_position_z_coordinate_count=1.0,
                physical_state_position_coverage90_hit_count=float(index * 2),
                physical_state_position_coverage90_coordinate_count=float(index * 3),
                physical_state_velocity_sse=float(index * index * 3),
                physical_state_velocity_coordinate_count=3.0,
                physical_distance_gated_matched_object_frames=float(index * 4),
                physical_distance_gated_target_object_frames=float(index * 5),
                physical_distance_gated_predicted_object_frames=float(index * 8),
                matched_object_frames=float(index * 6),
                existence_negative_supervision_object_frames=float(index * 7),
                physical_collision_true_positive_count=float(index * 2),
                physical_collision_false_positive_count=float(index),
                physical_collision_false_negative_count=float(index),
                parameter_drag_observable_object_count=float(index * 3),
                parameter_restitution_observable_object_count=float(index),
                **{
                    f"physical_rollout_position@{horizon}_sse": float(index * index * 3)
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_rollout_position@{horizon}_coordinate_count": 3.0
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_forecast_predictable_target_count@{horizon}": float(index * 2)
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_forecast_target_count@{horizon}": float(index * 4)
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_rollout_velocity@{horizon}_sse": float(index * index * 3)
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_rollout_velocity@{horizon}_coordinate_count": 3.0
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_collision_true_positive_count@{horizon}": float(index * 2)
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_collision_false_positive_count@{horizon}": float(index)
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
                **{
                    f"physical_collision_false_negative_count@{horizon}": float(index)
                    for horizon in (
                        "0.100s",
                        "0.250s",
                        "0.500s",
                        "0.750s",
                        "1.000s",
                    )
                },
            )
        )
    _write_metrics(run, records)

    report = audit_run(run, trend_window_blocks=2)
    complete, partial = report["training_trend_windows"]

    assert report["training_trend_window_blocks"] == 2
    assert complete["first_step"] == 8
    assert complete["last_step"] == 16
    assert complete["logged_blocks"] == 2
    assert complete["complete"] is True
    assert complete["causal_trajectory_support_count"] == 30.0
    assert complete["current_position_rmse_m"] == (5.0 / 2.0) ** 0.5
    assert complete["current_position_rmse_by_axis_m"] == {
        "x": (5.0 / 2.0) ** 0.5,
        "y": (5.0 / 2.0) ** 0.5,
        "z": (5.0 / 2.0) ** 0.5,
    }
    assert complete["current_position_coverage90"] == 2.0 / 3.0
    assert complete["current_velocity_rmse_mps"] == (5.0 / 2.0) ** 0.5
    assert complete["lifecycle"] == {
        "matched_object_frames": 18.0,
        "existence_negative_supervision_object_frames": 21.0,
        "distance_gated_target_coverage": 0.8,
        "distance_gated_prediction_precision": 0.5,
    }
    assert complete["horizon_position_rmse_m"]["1.000s"] == (5.0 / 2.0) ** 0.5
    assert complete["horizon_target_coverage"]["1.000s"] == 0.5
    assert complete["horizon_velocity_rmse_mps"]["1.000s"] == (5.0 / 2.0) ** 0.5
    assert complete["collision_f1"] == 2.0 / 3.0
    assert complete["collision_f1_by_horizon"]["1.000s"] == 2.0 / 3.0
    assert complete["parameter_observability"] == {
        "drag_object_count": 9.0,
        "restitution_object_count": 3.0,
    }
    assert complete["identity"]["switch_rate"] == 0.1
    assert partial["first_step"] == 24
    assert partial["last_step"] == 24
    assert partial["logged_blocks"] == 1
    assert partial["complete"] is False


def test_audit_compares_count_pooled_matched_reference_schedule(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    shared = {
        "episode_seeds": "1,2",
        "window_start_frame": 4.0,
        "window_stop_frame": 8.0,
        "physical_state_position_coordinate_count": 3.0,
        "physical_state_velocity_coordinate_count": 3.0,
        "physical_distance_gated_matched_object_frames": 4.0,
        "physical_distance_gated_target_object_frames": 8.0,
        "physical_distance_gated_predicted_object_frames": 10.0,
    }
    _write_metrics(
        candidate,
        [
            _record(
                8,
                physical_state_position_sse=3.0,
                physical_state_velocity_sse=12.0,
                **shared,
            ),
            _record(
                16,
                physical_state_position_sse=12.0,
                physical_state_velocity_sse=27.0,
                **shared,
            ),
        ],
    )
    _write_metrics(
        reference,
        [
            _record(
                8,
                physical_state_position_sse=12.0,
                physical_state_velocity_sse=3.0,
                **shared,
            ),
            _record(
                16,
                physical_state_position_sse=12.0,
                physical_state_velocity_sse=12.0,
                **shared,
            ),
        ],
    )

    report = audit_run(candidate, reference_run_directory=reference)
    comparison = report["matched_reference_comparison"]

    assert report["status"] == "pass"
    assert comparison["matched_steps"] == [8, 16]
    assert comparison["missing_reference_steps"] == []
    assert comparison["schedule_mismatches"] == []
    assert comparison["candidate"]["current_position_rmse_m"] == (15.0 / 6.0) ** 0.5
    assert comparison["reference"]["current_position_rmse_m"] == 2.0
    assert comparison["candidate_minus_reference"]["current_position_rmse_m"] == (
        (15.0 / 6.0) ** 0.5 - 2.0
    )
    assert comparison["candidate_minus_reference"]["current_velocity_rmse_mps"] == (
        (39.0 / 6.0) ** 0.5 - (15.0 / 6.0) ** 0.5
    )


def test_audit_rejects_mismatched_reference_schedule(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    _write_metrics(candidate, [_record(8, episode_seeds="1,2")])
    _write_metrics(reference, [_record(8, episode_seeds="3,4")])

    report = audit_run(candidate, reference_run_directory=reference)

    assert report["status"] == "fail"
    comparison = report["matched_reference_comparison"]
    assert comparison["schedule_mismatches"] == [
        {
            "step": 8,
            "field": "episode_seeds",
            "candidate": "1,2",
            "reference": "3,4",
        }
    ]
    assert any("deterministic training schedule differs" in item for item in report["failures"])


def test_audit_rejects_missing_reference_step(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    _write_metrics(candidate, [_record(8), _record(16)])
    _write_metrics(reference, [_record(8)])

    report = audit_run(candidate, reference_run_directory=reference)

    assert report["status"] == "fail"
    comparison = report["matched_reference_comparison"]
    assert comparison["matched_steps"] == [8]
    assert comparison["missing_reference_steps"] == [16]
    assert any("missing candidate training steps: 16" in item for item in report["failures"])
