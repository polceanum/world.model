from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    read_capability_summary,
    write_capability_summary,
)
from world_model.visualisation.progress import (
    _compact_animation_events,
    _numeric_curve_points,
    build_progress_dashboard,
    discover_run_summaries,
    render_summary_html,
    write_run_report,
)


def test_animation_event_ledger_compacts_repeated_long_horizon_contacts() -> None:
    events = [
        ("known action", "+0.10 s"),
        *(("contact", f"+{index / 10:.2f} s") for index in range(1, 11)),
    ]

    compact = _compact_animation_events(events)

    assert compact == "known action @ +0.10 s · contact ×10 (+0.10 s–+1.00 s)"


def _summary(run_id: str, status: str = "completed") -> CapabilityRunSummary:
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=f"2026-09-08T00:00:0{len(run_id)}+00:00",
        lifecycle_status=status,
        outcome="qualified_convergence" if status == "completed" else status,
        source_format=CAPABILITY_SUMMARY_SCHEMA,
        configuration={"seed": 1},
        provenance={"commit": "abc", "planning_used_as_training_loss": False},
        scores={
            "candidate": {"value": 0.12, "supported_weight": 1.0},
            "incumbent": {"value": 0.15, "supported_weight": 1.0},
            "selected": "candidate",
        },
        factor_metrics={
            "sensor_noise": {
                "status": "measured",
                "score": 0.13,
                "proposal_f1": 0.97,
            },
            "physical_parameters": {"status": "unmeasured"},
        },
        cell_metrics={
            "N1/contact=0/dynamic=0": {"current_position_rmse_m": {"value": 0.01, "support": 12}}
        },
        horizon_curves={
            "candidate_position_rmse_m": {"0.05": 0.012, "2.0": 0.10},
            "candidate_velocity_rmse_mps": {"0.05": 0.02, "2.0": 0.08},
        },
        uncertainty={"coverage_90_range": [0.86, 0.94]},
        planning={
            "serial_vectorized_winner_parity": True,
            "maximum_cost_difference": 0.0,
            "slices": [
                {
                    "object_count": 1,
                    "candidate_count": 8,
                    "oracle_winner_accuracy": {"value": 0.95, "support": 10},
                    "normalized_regret_median": {"value": 0.02, "support": 10},
                    "successful_oracle_goal_success": {"value": 0.96, "support": 10},
                }
            ],
        },
        resources={
            "perception_latency_seconds": 0.1,
            "learned_weight_bytes": 10_000,
            "scalability": {"probes": [{"object_count": 16, "median_latency_seconds": 0.08}]},
        },
        artifacts={"run_bytes": 4_096, "archive_bytes": 8_192},
        selection={"selected": "candidate"},
        failure_attribution={"primary_bottleneck": "camera motion", "ablation_owner": "dynamics"},
        qualitative={
            "best_episode": "episode-1",
            "worst_episode": "episode-2",
            "representative_episode": "episode-3",
        },
        unsupported_claims=("unknown calibration",),
        scope_limitations=("articulated bodies",),
    ).validate()


def test_summary_roundtrip_and_active_refresh_are_versioned(tmp_path: Path) -> None:
    active = _summary("active", "active")
    path = tmp_path / "capability_summary.json"
    write_capability_summary(active, path)
    assert read_capability_summary(path) == active
    active_html = render_summary_html(active)
    assert 'http-equiv="refresh"' in active_html
    assert 'id="capability-run-summary"' in active_html
    assert "Model used for this result" in active_html
    assert "Structured online world model" in active_html
    assert active_html.index("Model used for this result") < active_html.index("Capability error")
    assert "Position RMSE across horizon" in active_html
    assert "Prediction horizon (s)" in active_html
    assert "Position RMSE (m)" in active_html
    assert "Velocity RMSE across horizon" in active_html
    assert "Velocity RMSE (m/s)" in active_html
    assert "N1/contact=0/dynamic=0" in active_html
    assert "Factor performance" in active_html
    assert "Proposal F1" in active_html
    assert "4 s RMSE" in active_html
    assert "Parameter error" in active_html
    assert "90% uncertainty coverage by object count, contact, and membership" in active_html
    assert "closest to 0.9 target" in active_html
    assert "Relative within this run; colour is not a pass/fail gate." in active_html
    assert "K8 latency (s)" in active_html
    assert "Planning quality by cardinality" in active_html
    assert "Worst-cardinality planning quality" in active_html
    assert 'class="metric-pass"' in active_html
    assert "Capability frontier" in active_html
    assert "RGB-D" in active_html
    assert "Ablation attribution" in active_html
    assert "Observed nominal-90% coverage range" in active_html

    completed_html = render_summary_html(replace(active, lifecycle_status="completed"))
    assert 'http-equiv="refresh"' not in completed_html


def test_adaptive_model_overview_reports_size_training_and_belief_evolution() -> None:
    summary = replace(
        _summary("adaptive-model", "completed"),
        source_format="world_model_adaptive_physics_scale_v1",
        configuration={
            "single_model": True,
            "ensemble": False,
            "mixture_of_experts": False,
        },
        provenance={
            "checkpoint_loaded": False,
            "learned_parameters_zero_initialized": True,
        },
        resources={"learned_weight_bytes": 12_984},
        cell_metrics={
            "N4/adaptive_physics/mixed_rigid": {
                "parameter_relative_error": {"value": 0.002, "support": 16}
            },
            "N6/adaptive_physics/mixed_rigid": {
                "parameter_relative_error": {"value": 0.002, "support": 24}
            },
            "N8/adaptive_physics/mixed_rigid": {
                "parameter_relative_error": {"value": 0.002, "support": 32}
            },
        },
        uncertainty={"parameter_contracted_fraction": 1.0},
        qualitative={
            "parameter_convergence": [
                {"stage": "neutral priors", "mean_relative_error": 0.4373506},
                {"stage": "public evidence", "mean_relative_error": 0.0023182},
            ]
        },
    )

    html = render_summary_html(summary)

    assert "Single shared structured analytic model" in html
    assert html.count('class="model-stage"') == 5
    assert "1,623" in html
    assert "12.7 KiB" in html
    assert "0 / 1,623" in html
    assert "0 steps" in html
    assert "0 examples in this run" in html
    assert "72 / 72" in html
    assert "43.74% → 0.23%" in html
    assert "99.47% lower parameter error" in html
    assert "Weights and architecture stayed fixed" in html


def test_neural_model_overview_reports_architecture_and_real_training() -> None:
    summary = replace(
        _summary("neural-model", "completed"),
        source_format="world_model_neural_adaptive_physics_v1",
        configuration={
            "model_profile": {
                "name": "Single shared neural-adaptive structured world model",
                "variant": "2-layer evidence transformer · analytic rigid rollout",
                "stages": [
                    {
                        "role": "Learn",
                        "name": "Evidence transformer",
                        "detail": "2 layers · 4 heads · width 48",
                    }
                ],
                "evolution_note": "Transformer weights learned across streamed episodes.",
            }
        },
        provenance={"checkpoint_loaded": False, "trained_from_scratch": True},
        resources={
            "learned_parameter_count": 48_920,
            "nonzero_learned_parameter_count": 48_920,
            "changed_learned_parameter_count": 48_920,
            "learned_weight_bytes": 195_680,
            "optimizer_updates": 2_048,
            "training_examples": 262_144,
            "training_seconds": 120.5,
            "online_adaptation_updates_accepted": 72,
            "online_adaptation_updates_attempted": 72,
            "adaptation_update_label": "observation-conditioned parameter adaptations",
            "adaptation_update_detail": "neural inference; no online optimizer",
        },
        qualitative={
            "parameter_convergence": [
                {"stage": "neutral", "mean_relative_error": 0.5118},
                {"stage": "neural evidence", "mean_relative_error": 0.0214},
            ],
            "training_curve": [
                {"step": 1.0, "loss": 0.1979, "full_parameter_loss": 0.1979},
                {"step": 2048.0, "loss": 0.0072, "full_parameter_loss": 0.0072},
            ],
        },
    )

    html = render_summary_html(summary)

    assert "Single shared neural-adaptive structured world model" in html
    assert "2 layers · 4 heads · width 48" in html
    assert "48,920 / 48,920" in html
    assert "48,920 changed from initialization" in html
    assert "2,048 steps" in html
    assert "262,144 examples in this run" in html
    assert "observation-conditioned parameter adaptations" in html
    assert "neural inference; no online optimizer" in html
    assert "Neural training convergence" in html
    assert "Optimizer step" in html
    assert "Full normalized parameter loss" in html
    assert "Planning outcomes are excluded from the optimized loss" in html


def test_long_horizon_dashboard_names_comparison_curves_and_training_objective() -> None:
    summary = replace(
        _summary("long-horizon", "completed"),
        horizon_curves={
            "candidate_position_rmse_m": {
                "0.5": 0.01,
                "1": 0.02,
                "10": 0.11,
                "12": 0.14,
                "2": 0.025,
                "4": 0.03,
                "8": 0.08,
            },
            "incumbent_position_rmse_m": {"4": 0.05, "8": 0.16, "12": 0.30},
            "oracle_parameter_position_rmse_m": {"4": 0.02, "8": 0.05, "12": 0.09},
        },
        resources={
            "optimizer_updates": 4096,
            "training_examples": 524288,
            "training_seconds": 250.0,
        },
        qualitative={
            "training_curve": [
                {"step": 1.0, "long_horizon_stability_loss": 0.16},
                {"step": 4096.0, "long_horizon_stability_loss": 0.005},
            ],
            "training_curve_metric": "long_horizon_stability_loss",
            "training_curve_title": "Held multi-horizon physical-response objective",
            "training_curve_y_label": "Multi-horizon stability loss",
        },
    )

    html = render_summary_html(summary)

    assert "Prior neural incumbent position RMSE" in html
    assert "Truth-parameter solver floor" in html
    assert "Prediction horizon (s)" in html
    assert "Held multi-horizon physical-response objective" in html
    assert "Multi-horizon stability loss" in html
    candidate_chart = html[: html.index("Prior neural incumbent position RMSE")]
    assert candidate_chart.index(">2s</text>") < candidate_chart.index(">10s</text>")
    assert _numeric_curve_points({"10": 0.10, "2": 0.02, "1": 0.01}) == [
        ("1s", 0.01),
        ("2s", 0.02),
        ("10s", 0.10),
    ]


def test_parameter_uncertainty_contraction_is_not_mislabeled_as_coverage() -> None:
    summary = replace(
        _summary("parameter-adaptation", "completed"),
        uncertainty={"parameter_contracted_fraction": 1.0},
    )

    html = render_summary_html(summary)

    assert "100.0%" in html
    assert "Accepted object-parameter blocks with contracted uncertainty" in html
    assert "Observed nominal-90% coverage range" not in html


def test_adaptive_truth_state_ablation_derives_endpoint_reduction() -> None:
    summary = replace(
        _summary("adaptive-ablation", "completed"),
        factor_metrics={"adaptive_physics_n8": {"four_second_position_rmse_m": 0.026}},
        failure_attribution={
            "primary_bottleneck": "none",
            "ablation_owner": "shared analytic dynamics",
            "ablations": {
                "truth_state_and_parameters_n8": {
                    "endpoint_position_rmse_m": 0.021,
                    "status": "diagnostic",
                }
            },
        },
    )

    html = render_summary_html(summary)

    assert "truth_state_and_parameters_n8" in html
    assert "0.005" in html


def test_report_labels_n8_rgbd_probe_as_development_only() -> None:
    summary = _summary("n8-development", "completed")
    summary = replace(
        summary,
        resources={
            **summary.resources,
            "scalability": {
                "full_perceptual_qualification": False,
                "perceptual_development_probes": [
                    {
                        "object_count": 8,
                        "observed_active_count": 8,
                        "position_rmse_m": 0.001,
                        "inference_latency_seconds": 0.2,
                        "full_perceptual_qualification": False,
                    }
                ],
            },
        },
    )

    html = render_summary_html(summary)

    assert "N=8 RGB-D development" in html
    assert "N=8 observed objects" in html
    assert "development only" in html


def test_dashboard_handles_current_historical_missing_and_malformed_runs(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    archive = tmp_path / ".archive"
    archive.mkdir()
    (archive / "evidence.bin").write_bytes(b"archive")
    for run_id, status in (("active", "active"), ("completed", "completed"), ("failed", "failed")):
        directory = runs / run_id
        directory.mkdir(parents=True)
        summary = _summary(run_id, status)
        write_capability_summary(summary, directory / "capability_summary.json")
        write_run_report(summary, directory)
    historical = runs / "historical"
    historical.mkdir()
    (historical / "development_report.json").write_text(
        json.dumps({"schema": "legacy_report_v3", "status": "passed"}),
        encoding="utf-8",
    )
    (runs / "missing").mkdir()
    malformed = runs / "malformed"
    malformed.mkdir()
    (malformed / "capability_summary.json").write_text("{broken", encoding="utf-8")

    summaries, notices = discover_run_summaries(runs)
    assert {summary.run_id for summary in summaries} == {
        "active",
        "completed",
        "failed",
        "historical",
    }
    assert any("missing" in notice for notice in notices)
    assert any("malformed" in notice for notice in notices)

    dashboard = build_progress_dashboard(runs, archive_root=archive)
    content = dashboard.read_text(encoding="utf-8")
    assert "Capability trend" in content
    assert "historical" in content
    assert "malformed capability summary" in content
    assert "Portable report" in content
    assert "Run ledger (4 compact reports)" in content
    assert "Latest run" in content
    assert '"managed_budget_bytes": 262144000' in content
    assert "Managed run tree" in content
    assert "Budget used" in content
    assert dashboard.stat().st_size < 1_000_000


def test_dashboard_keeps_latest_measured_factor_across_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first_directory = runs / "first"
    second_directory = runs / "second"
    first_directory.mkdir(parents=True)
    second_directory.mkdir()
    first = replace(
        _summary("first"),
        factor_metrics={
            "sensor_noise": {"status": "failed", "score": 0.4},
            "partial_visibility": {"status": "unmeasured"},
        },
    )
    second = replace(
        _summary("second"),
        created_at_utc="2026-09-08T01:00:00+00:00",
        factor_metrics={
            "sensor_noise": {"status": "unmeasured"},
            "partial_visibility": {"status": "passed", "score": 0.2},
        },
        unsupported_claims=(
            "sensor_noise capability",
            "partial_visibility capability",
            "factor-conditioned planning",
        ),
        cell_metrics={},
        horizon_curves={},
        uncertainty={},
        planning={"status": "unmeasured"},
        resources={},
    )
    write_capability_summary(first, first_directory / "capability_summary.json")
    write_capability_summary(second, second_directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert '"source_run": "first"' in content
    assert '"source_run": "second"' in content
    assert "sensor_noise" in content and "partial_visibility" in content
    assert "<li>sensor_noise capability</li>" not in content
    assert "<li>partial_visibility capability</li>" not in content
    assert "<li>factor-conditioned planning</li>" in content
    assert '"planning_source_run": "first"' in content
    assert '"source_run": "first"' in content
    assert "N1/contact=0/dynamic=0" in content


def test_dashboard_merges_sparse_resource_measurements_with_provenance(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    older_directory = runs / "older"
    latest_directory = runs / "latest"
    older_directory.mkdir(parents=True)
    latest_directory.mkdir()
    older = replace(
        _summary("older"),
        resources={
            "perception_latency_seconds": 0.12,
            "process_rss_bytes": 12_000,
            "scalability": {"probes": [{"object_count": 16, "median_latency_seconds": 0.08}]},
        },
    )
    latest = replace(
        _summary("latest"),
        created_at_utc="2026-09-08T02:00:00+00:00",
        resources={
            "learned_weight_bytes": 20_000,
            "scalability": {
                "perceptual_development_probes": [
                    {
                        "object_count": 8,
                        "observed_active_count": 8,
                        "position_rmse_m": 0.002,
                        "inference_latency_seconds": 0.21,
                        "full_perceptual_qualification": False,
                    }
                ]
            },
        },
    )
    write_capability_summary(older, older_directory / "capability_summary.json")
    write_capability_summary(latest, latest_directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert "0.1200 s" in content
    assert "0.0800 s" in content
    assert "0.2100 s" in content
    assert "2.000e+04 bytes" in content
    assert "source: older" in content
    assert "source: latest" in content
    assert '"perception_latency_seconds": "older"' in content
    assert '"scalability.perceptual_development_probes": "latest"' in content


def test_dashboard_resource_headline_prefers_latest_successful_measurement(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    passed_directory = runs / "passed"
    failed_directory = runs / "failed"
    passed_directory.mkdir(parents=True)
    failed_directory.mkdir()
    passed = replace(
        _summary("passed"),
        resources={"n8_rollout_latency_seconds": 0.18},
    )
    failed = replace(
        _summary("failed", "failed"),
        created_at_utc="2026-09-08T03:00:00+00:00",
        outcome="capability_gate_failed",
        resources={"n8_rollout_latency_seconds": 56.0},
    )
    write_capability_summary(passed, passed_directory / "capability_summary.json")
    write_capability_summary(failed, failed_directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert "0.1800 s" in content
    assert "56.0000 s" not in content
    assert '"n8_rollout_latency_seconds": "passed"' in content


def test_dashboard_carries_forward_newest_per_object_accuracy_frontier(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    visual_directory = runs / "visual"
    aggregate_directory = runs / "aggregate"
    visual_directory.mkdir(parents=True)
    aggregate_directory.mkdir()
    visual = replace(
        _summary("visual", "failed"),
        configuration={
            "maximum_per_object_position_error_m": 0.015,
            "maximum_per_object_velocity_error_mps": 0.075,
            "maximum_per_box_orientation_error_degrees": 6.0,
        },
        qualitative={
            "accuracy_frontier": [
                {
                    "object_count": 8,
                    "objects": {
                        "17": {
                            "position_m": 0.010,
                            "velocity_mps": 0.050,
                            "orientation_degrees": 5.0,
                        }
                    },
                }
            ]
        },
    )
    aggregate = replace(
        _summary("aggregate"),
        created_at_utc="2026-09-08T01:00:00+00:00",
        qualitative={},
    )
    write_capability_summary(visual, visual_directory / "capability_summary.json")
    write_capability_summary(aggregate, aggregate_directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert "Worst-slice accuracy frontier" in content
    assert "Evidence source: visual · failed" in content
    assert "Per-object gates: position ≤ 0.0150 m" in content
    assert '<span class="tag passed">passed</span>' in content


def test_aggregate_report_describes_embedded_rgbd_forecasts_from_card_metadata() -> None:
    summary = replace(
        _summary("aggregate"),
        configuration={"aggregate_protocol": True},
        qualitative={
            "forecast_animations": [
                {
                    "schema": "world_model_compact_animation_v1",
                    "label": "N=6 RGB-D initialized multi-contact forecast",
                    "episode": "visual-dynamic-n6",
                    "object_count": 6,
                    "mode": "forecast",
                    "projection": "world_xy",
                    "axis_labels": ["x", "y"],
                    "bounds": {"horizontal": [-1.0, 1.0], "vertical": [-1.0, 1.0]},
                    "known_actions_in_rollout": True,
                    "frames": [
                        {
                            "frame": 0,
                            "time_s": 0.0,
                            "model": [],
                            "reference": [],
                        }
                    ],
                }
            ]
        },
    )

    content = render_summary_html(summary)

    assert "Public RGB-D initialized rollouts" in content
    assert "State-first scale rollouts" not in content


def test_dashboard_keeps_planning_slices_separate_by_factor(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for index, factor in enumerate(("sensor_noise", "camera_motion")):
        directory = runs / factor
        directory.mkdir(parents=True)
        summary = replace(
            _summary(factor),
            created_at_utc=f"2026-09-08T0{index}:00:00+00:00",
            factor_metrics={
                name: {"status": "unmeasured"} for name in ("sensor_noise", "camera_motion")
            },
            planning={**_summary(factor).planning, "factor": factor},
            cell_metrics={},
            resources={},
            unsupported_claims=(
                "sensor_noise factor-conditioned planning",
                "camera_motion factor-conditioned planning",
            ),
        )
        write_capability_summary(summary, directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert '"by_factor"' in content
    assert '"source_run": "sensor_noise"' in content
    assert '"source_run": "camera_motion"' in content
    assert "<td>sensor_noise</td>" in content
    assert "<td>camera_motion</td>" in content
    assert "sensor_noise factor-conditioned planning</li>" not in content
    assert "camera_motion factor-conditioned planning</li>" not in content


def test_dashboard_keeps_latest_specialized_pose_planning_visible(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    factor_directory = runs / "factor"
    pose_directory = runs / "pose"
    factor_directory.mkdir(parents=True)
    pose_directory.mkdir()
    factor = replace(
        _summary("factor"),
        planning={**_summary("factor").planning, "factor": "sensor_noise"},
    )
    pose = replace(
        _summary("pose"),
        created_at_utc="2026-09-08T02:00:00+00:00",
        planning={
            "status": "passed",
            "goal": "terminal world pose",
            "serial_vectorized_winner_parity": True,
            "maximum_cost_difference": 0.0,
            "by_candidate_count": {
                "8": {
                    "winner_accuracy": 1.0,
                    "median_normalized_regret": 0.0,
                    "goal_success": 1.0,
                },
                "32": {
                    "winner_accuracy": 1.0,
                    "median_normalized_regret": 0.0,
                    "goal_success": 1.0,
                },
            },
        },
    )
    write_capability_summary(factor, factor_directory / "capability_summary.json")
    write_capability_summary(pose, pose_directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert '"latest_specialized"' in content
    assert "Latest specialized planning: terminal world pose" in content
    assert "Source: pose" in content
    assert "separately from the factor-conditioned planning matrix" in content


def test_trends_do_not_mix_physical_and_planning_score_scales() -> None:
    physical = replace(
        _summary("physical"),
        source_format="world_model_capability_factor_report_v1",
        configuration={"factor": "sensor_noise"},
    )
    physical_again = replace(physical, run_id="physical-again")
    planning = replace(
        _summary("planning"),
        source_format="world_model_capability_planning_report_v1",
        configuration={"factor": "sensor_noise"},
    )
    planning_again = replace(planning, run_id="planning-again")

    content = render_summary_html(
        planning,
        history=(physical, planning, physical_again, planning_again),
    )

    assert "Sensor Noise physical score — comparable protocol" in content
    assert "Sensor Noise planning error — comparable protocol" in content
    assert "Different horizons, cardinalities, and capability gates" in content
    assert "Overall capability score across runs" not in content
    assert "Lower-is-better score across runs" not in content


def test_unrelated_specialized_runs_are_not_connected_as_a_trend() -> None:
    first = replace(_summary("first"), source_format="world_model_first_gate_v1")
    second = replace(_summary("second"), source_format="world_model_second_gate_v1")

    content = render_summary_html(second, history=(first, second))

    assert "Comparable trend: insufficient repeated protocol evidence" in content
    assert "Overall capability score across runs" not in content


def test_hardening_report_shows_paired_regressions_and_protected_horizon() -> None:
    base = _summary("hardening")
    summary = replace(
        base,
        horizon_curves={
            **base.horizon_curves,
            "protected_long_horizon_position_rmse_m": {
                "0.05": 0.0001,
                "4": 0.0005,
                "8": 0.0004,
            },
        },
        qualitative={
            **base.qualitative,
            "regression_checks": [
                {
                    "family": "visual_n8",
                    "metric": "maximum_position_rmse_m",
                    "direction": "lower",
                    "baseline": 0.014,
                    "candidate": 0.004,
                    "limit": 0.007,
                    "passed": True,
                }
            ],
        },
    )

    content = render_summary_html(summary)

    assert "Cross-capability regression envelope" in content
    assert "positive deltas improve higher-is-better metrics" in content
    assert "visual_n8" in content
    assert "-71.4%" in content
    assert "≤ 0.0070" in content
    assert "Protected long-horizon position RMSE" in content
    assert "8s" in content


def test_coverage_heatmap_colours_distance_from_target_not_larger_values() -> None:
    summary = replace(
        _summary("coverage"),
        cell_metrics={
            "target": {"uncertainty_90_coverage": {"value": 0.90, "support": 12}},
            "high": {"uncertainty_90_coverage": {"value": 0.97, "support": 12}},
        },
    )

    content = render_summary_html(summary)

    assert 'fill="rgb(54,175,92)"/><title>target: 0.9000</title>' in content
    assert 'fill="rgb(229,75,92)"/><title>high: 0.9700</title>' in content


def test_planning_summary_uses_worst_cardinality_and_keeps_full_ledger() -> None:
    planning = {
        **_summary("planning").planning,
        "slices": [
            {
                "object_count": 1,
                "candidate_count": 8,
                "oracle_winner_accuracy": {"value": 1.0, "support": 10},
                "normalized_regret_median": {"value": 0.01, "support": 10},
                "successful_oracle_goal_success": {"value": 0.98, "support": 10},
            },
            {
                "object_count": 6,
                "candidate_count": 8,
                "oracle_winner_accuracy": {"value": 0.75, "support": 10},
                "normalized_regret_median": {"value": 0.12, "support": 10},
                "successful_oracle_goal_success": {"value": 0.80, "support": 10},
            },
        ],
    }

    content = render_summary_html(replace(_summary("planning"), planning=planning))
    aggregate = content.split("Worst-cardinality planning quality", 1)[1].split(
        'class="planning-slices"', 1
    )[0]

    assert "0.7500" in aggregate
    assert "0.1200" in aggregate
    assert "0.8000" in aggregate
    assert "Planning quality by cardinality (2 detailed slices)" in content
    assert "<td>N1</td>" in content and "<td>N6</td>" in content


def test_malformed_animation_is_ignored_without_breaking_the_report() -> None:
    summary = replace(
        _summary("malformed-animation"),
        qualitative={
            **_summary("malformed-animation").qualitative,
            "animations": [
                {
                    "frames": [{"frame": "not-an-integer", "time_s": 0.0}],
                    "bounds": {"horizontal": [-1.0, 1.0], "vertical": [-1.0, 1.0]},
                    "axis_labels": ["x", "y"],
                }
            ],
        },
    )

    content = render_summary_html(summary)

    assert 'class="animation-card"' not in content
    assert "Portable report" in content


def test_dashboard_keeps_three_lightweight_vector_animations_from_latest_evidence(
    tmp_path: Path,
) -> None:
    animation = {
        "schema": "world_model_compact_animation_v1",
        "label": "best",
        "episode": "development:1/seed=2",
        "object_count": 1,
        "contact": False,
        "dynamic_membership": False,
        "current_position_rmse_m": 0.001,
        "mode": "tracking",
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": [-1.0, 1.0], "vertical": [-1.0, 1.0]},
        "frames": [
            {
                "frame": 0,
                "time_s": 0.0,
                "truth": [[0, -0.5, 0.0]],
                "model": [[0, -0.49, 0.01]],
            },
            {
                "frame": 55,
                "time_s": 2.75,
                "truth": [[0, 0.5, 0.0]],
                "model": [[0, 0.49, -0.01]],
            },
        ],
        "events": [{"frame": 28, "kind": "known action"}],
        "reference": "private simulator truth used only after public inference",
    }
    forecast = {
        **animation,
        "schema": "world_model_compact_forecast_animation_v1",
        "mode": "forecast",
        "anchor_frame": 15,
        "two_second_position_rmse_m": 0.02,
        "rollout_horizons_s": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        "frames": [
            {"frame": 15, "time_s": 0.0, "truth": [[0, -0.5, 0.0]], "model": [[0, -0.49, 0.01]]},
            {"frame": 16, "time_s": 0.05, "truth": [[0, -0.4, 0.1]], "model": [[0, -0.39, 0.11]]},
            {"frame": 55, "time_s": 2.0, "truth": [[0, 0.5, 0.0]], "model": [[0, 0.49, -0.01]]},
        ],
        "reference": "private future truth opened only after the public open-loop rollout",
    }
    runs = tmp_path / "runs"
    physical_directory = runs / "physical"
    planning_directory = runs / "planning"
    physical_directory.mkdir(parents=True)
    planning_directory.mkdir()
    physical = replace(
        _summary("physical"),
        created_at_utc="2026-09-08T00:00:00+00:00",
        source_format="world_model_capability_factor_report_v1",
        qualitative={
            **_summary("physical").qualitative,
            "animations": [animation] * 4,
            "forecast_animations": [forecast] * 4,
        },
    )
    planning = replace(
        _summary("planning"),
        created_at_utc="2026-09-08T01:00:00+00:00",
        source_format="world_model_capability_planning_report_v1",
        qualitative={
            **_summary("planning").qualitative,
            "best_episode": "latest-planning-qualitative-marker",
        },
    )
    assert "Evidence source: physical." in render_summary_html(physical)
    write_capability_summary(physical, physical_directory / "capability_summary.json")
    write_capability_summary(planning, planning_directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert "Observed tracking examples" in content
    assert "Two-second open-loop forecasts" in content
    assert content.count('class="animation-card"') == 6
    assert "Colour = object identity" in content
    assert "● filled / solid = model" in content
    assert "○ open / dashed = private reference" in content
    assert "World X (m)" in content and "World Y (m)" in content
    assert "no contact" in content and "static membership" in content
    assert "known action @ 1.40 s" in content
    assert "Evidence source: physical." in content
    assert "Pause" in content and "Replay" in content
    assert 'type="range"' in content
    assert 'class="animation-trail animation-trail-model"' in content
    assert "requestAnimationFrame" in content
    assert '"animation_source_run": "physical"' in content
    assert '"forecast_animation_source_run": "physical"' in content
    assert "latest-planning-qualitative-marker" in content
    assert "<video" not in content and "<img" not in content
    assert "data:image" not in content
    assert len(content.encode("utf-8")) < 250_000


def test_dashboard_forecast_gallery_covers_touching_scale_and_long_horizon(
    tmp_path: Path,
) -> None:
    base = {
        "schema": "world_model_compact_forecast_animation_v1",
        "object_count": 1,
        "mode": "forecast",
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": [-1.0, 1.0], "vertical": [-1.0, 1.0]},
        "frames": [
            {"frame": 0, "time_s": 0.0, "truth": [[0, 0.0, 0.0]], "model": [[0, 0.0, 0.0]]},
            {"frame": 1, "time_s": 2.0, "truth": [[0, 0.1, 0.0]], "model": [[0, 0.1, 0.0]]},
        ],
    }
    runs = tmp_path / "runs"
    for index, (run_id, animation) in enumerate(
        (
            (
                "long",
                {
                    **base,
                    "episode": "long",
                    "label": "eight-second causal forecast",
                    "long_horizon_endpoint_s": 8.0,
                },
            ),
            (
                "touching",
                {
                    **base,
                    "episode": "touching",
                    "label": "same-appearance touching action-free two-second forecast",
                    "long_horizon_endpoint_s": 2.0,
                },
            ),
            (
                "scale",
                {
                    **base,
                    "episode": "scale-n8",
                    "label": "N=8 RGB-D initialized multi-contact forecast",
                    "object_count": 8,
                    "long_horizon_endpoint_s": 2.0,
                },
            ),
            (
                "adaptive",
                {
                    **base,
                    "episode": "adaptive-n8",
                    "label": "N=8 public RGB-D adaptive-physics four-second forecast",
                    "object_count": 8,
                    "long_horizon_endpoint_s": 4.0,
                },
            ),
        )
    ):
        directory = runs / run_id
        directory.mkdir(parents=True)
        summary = replace(
            _summary(run_id),
            created_at_utc=f"2026-09-08T0{index}:00:00+00:00",
            qualitative={"forecast_animations": [animation]},
        )
        write_capability_summary(summary, directory / "capability_summary.json")

    content = build_progress_dashboard(runs).read_text(encoding="utf-8")

    assert "same-appearance touching action-free two-second forecast" in content
    assert "N=8 public RGB-D adaptive-physics four-second forecast" in content
    assert "N=8 RGB-D initialized multi-contact forecast" not in content
    assert "eight-second causal forecast" in content
    assert content.count('class="animation-card"') == 3


def test_recovery_curve_names_observation_gap_axes() -> None:
    summary = replace(
        _summary("long-recovery"),
        horizon_curves={
            "recovery_position_rmse_m": {
                "0.00": 0.002,
                "0.40": 0.032,
                "0.45": 0.004,
            }
        },
    )

    content = render_summary_html(summary)

    assert "Position RMSE through observation gap and recovery" in content
    assert "Elapsed sequence time (s)" in content
    assert "Prediction horizon (s)" not in content


def test_action_free_forecast_introduction_does_not_invent_anchor_or_horizons() -> None:
    forecast = {
        "schema": "world_model_compact_touching_forecast_animation_v1",
        "label": "action-free forecast",
        "episode": "touching",
        "object_count": 1,
        "mode": "forecast",
        "anchor_frame": 7,
        "long_horizon_endpoint_s": 2.0,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": [-1.0, 1.0], "vertical": [-1.0, 1.0]},
        "frames": [
            {"frame": 7, "time_s": 0.0, "truth": [[0, 0.0, 0.0]], "model": [[0, 0.0, 0.0]]},
            {"frame": 8, "time_s": 0.1, "truth": [[0, 0.1, 0.0]], "model": [[0, 0.1, 0.0]]},
        ],
    }
    summary = replace(
        _summary("action-free"),
        qualitative={"forecast_animations": [forecast]},
    )

    content = render_summary_html(summary)

    assert "labelled public observation-derived belief" in content
    assert "frame-15" not in content
    assert "0.05, 0.10, 0.25" not in content


def test_long_horizon_animation_uses_declared_endpoint_and_frame_rate() -> None:
    long_horizon = {
        "schema": "world_model_compact_long_horizon_animation_v1",
        "label": "eight-second causal scene",
        "episode": "state-first:causal-scene",
        "object_count": 2,
        "contact": True,
        "dynamic_membership": False,
        "mode": "forecast",
        "frame_rate": 10.0,
        "anchor_frame": 0,
        "long_horizon_endpoint_s": 8.0,
        "endpoint_position_rmse_m": 0.12,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": [-2.0, 2.0], "vertical": [0.0, 2.5]},
        "frames": [
            {"frame": 0, "time_s": 0.0, "truth": [[1, 0.0, 1.0]], "model": [[1, 0.0, 1.0]]},
            {"frame": 80, "time_s": 8.0, "truth": [[1, 1.0, 1.0]], "model": [[1, 0.9, 1.0]]},
        ],
        "events": [{"frame": 40, "kind": "known action"}],
    }
    summary = replace(
        _summary("long-horizon"),
        configuration={"state_first": True},
        qualitative={
            **_summary("long-horizon").qualitative,
            "forecast_animations": [long_horizon],
        },
    )

    content = render_summary_html(summary)

    assert "Causal forecasts through 8 seconds" in content
    assert "8 s RMSE 0.1200 m" in content
    assert "known action @ +4.00 s" in content
    assert "state-first or state-only pressure test" in content


def test_pose_animation_and_parameter_convergence_are_self_describing() -> None:
    pose_animation = {
        "schema": "world_model_compact_pose_forecast_animation_v1",
        "label": "pose forecast",
        "episode": "off-centre-contact",
        "object_count": 2,
        "contact": True,
        "dynamic_membership": False,
        "mode": "forecast",
        "frame_rate": 20.0,
        "long_horizon_endpoint_s": 1.2,
        "endpoint_position_rmse_m": 0.001,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": [-1.0, 1.0], "vertical": [-1.0, 1.0]},
        "frames": [
            {
                "frame": 0,
                "time_s": 0.0,
                "truth": [[0, -0.5, 0.0, 0.0]],
                "model": [[0, -0.49, 0.01, 0.02]],
            },
            {
                "frame": 12,
                "time_s": 0.6,
                "truth": [[0, 0.0, 0.2, 1.2]],
                "model": [[0, 0.01, 0.19, 1.18]],
                "contacts": [[0.0, 0.2]],
            },
            {
                "frame": 24,
                "time_s": 1.2,
                "truth": [[0, 0.5, 0.4, 2.4]],
                "model": [[0, 0.49, 0.39, 2.38]],
            },
        ],
        "events": [{"frame": 12, "time_s": 0.6, "kind": "off-centre contact"}],
    }
    summary = replace(
        _summary("pose"),
        horizon_curves={
            **_summary("pose").horizon_curves,
            "candidate_orientation_rmse_degrees": {"0": 1.0, "1.2": 2.5},
        },
        qualitative={
            **_summary("pose").qualitative,
            "forecast_animations": [pose_animation],
            "parameter_convergence": [
                {
                    "stage": "neutral priors",
                    "mean_relative_error": 0.8,
                    "mass_relative_error": 0.3,
                    "restitution_relative_error": 0.1,
                    "drag_relative_error": 2.4,
                    "friction_relative_error": 0.4,
                },
                {
                    "stage": "observed contact",
                    "mean_relative_error": 0.01,
                    "mass_relative_error": 0.0,
                    "restitution_relative_error": 0.01,
                    "drag_relative_error": 0.0,
                    "friction_relative_error": 0.01,
                },
            ],
            "per_object_prediction_errors": [
                {
                    "scenario": "N=8",
                    "runtime_id": "42",
                    "position_m": 0.025,
                    "velocity_mps": 0.10,
                    "orientation_degrees": 4.5,
                }
            ],
        },
    )

    content = render_summary_html(summary)

    assert "Orientation RMSE across horizon" in content
    assert "Orientation RMSE (degrees)" in content
    assert "Online physical identification" in content
    assert "Mean relative parameter error" in content
    assert "observed contact" in content
    assert "Per-object prediction ledger" in content
    assert "Max position error (m)" in content
    assert "Runtime ID" in content
    assert 'data-orientation-role="truth"' in content
    assert "data-contact-marker" in content
    assert "Math.atan2" in content


def test_compositional_metric_matrix_uses_holdout_gates_only() -> None:
    summary = replace(
        _summary("compositional"),
        factor_metrics={
            "compositional_holdout": {
                "status": "passed",
                "proposal_f1": 0.96,
                "identity_accuracy": 0.951,
                "lifecycle_f1": 0.70,
                "current_position_rmse_m": 0.08,
                "two_second_position_rmse_m": 0.14,
                "collision_f1": 0.50,
                "uncertainty_90_coverage": 0.70,
            }
        },
    )

    content = render_summary_html(summary)
    row = content.split("<th>compositional_holdout</th>", 1)[1].split("</tr>", 1)[0]

    assert row.count('class="metric-pass"') == 3
    assert row.count('class="metric-info"') == 4
    assert 'class="metric-fail"' not in row
