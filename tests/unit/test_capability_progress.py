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
            "sensor_noise": {"status": "measured", "score": 0.13},
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
    assert "N1/contact=0/dynamic=0" in active_html

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
    assert dashboard.stat().st_size < 1_000_000
