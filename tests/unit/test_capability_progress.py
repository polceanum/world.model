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
    build_progress_dashboard,
    discover_run_summaries,
    render_summary_html,
    write_run_report,
)


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
        horizon_curves={"candidate_position_rmse_m": {"0.05": 0.012, "2.0": 0.10}},
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
    assert "Position RMSE across horizon" in active_html
    assert "Prediction horizon (s)" in active_html
    assert "Position RMSE (m)" in active_html
    assert "N1/contact=0/dynamic=0" in active_html
    assert "Factor performance" in active_html
    assert "Proposal F1" in active_html
    assert "90% uncertainty coverage by object count, contact, and membership" in active_html
    assert "closest to 0.9 target" in active_html
    assert "Relative within this run; colour is not a pass/fail gate." in active_html
    assert "K8 latency (s)" in active_html
    assert "Planning quality by cardinality" in active_html
    assert "Worst-cardinality planning quality" in active_html
    assert 'class="metric-pass"' in active_html

    completed_html = render_summary_html(replace(active, lifecycle_status="completed"))
    assert 'http-equiv="refresh"' not in completed_html


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


def test_trends_do_not_mix_physical_and_planning_score_scales() -> None:
    physical = replace(
        _summary("physical"),
        source_format="world_model_capability_factor_report_v1",
    )
    planning = replace(
        _summary("planning"),
        source_format="world_model_capability_planning_report_v1",
    )

    content = render_summary_html(planning, history=(physical, planning))

    assert "Physical factor score across runs" in content
    assert "Downstream planning error across runs" in content
    assert "Lower-is-better score across runs" not in content


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
    assert "● model estimate" in content and "○ private reference" in content
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
