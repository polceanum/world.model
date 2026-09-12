from __future__ import annotations

import json
from pathlib import Path

from world_model.evaluation.multicontact_six_dof import (
    MULTICONTACT_SIX_DOF_SCHEMA,
    multicontact_manifest_sha256,
    publish_multicontact_six_dof_scale,
    run_multicontact_six_dof_scale,
)


def test_multicontact_six_dof_scale_qualifies_and_stays_compact(tmp_path: Path) -> None:
    assert multicontact_manifest_sha256() == multicontact_manifest_sha256()
    # Fresh CLI execution owns absolute CPU timing. The in-process test still
    # enforces accuracy, contact attribution, exact serial parity, and relative
    # vectorization speedup after arbitrary earlier test-suite load.
    result = run_multicontact_six_dof_scale(enforce_latency=False)

    assert result.schema == MULTICONTACT_SIX_DOF_SCHEMA
    assert result.qualified, result.gate_failures
    assert [item.object_count for item in result.scenarios] == [4, 6, 8]
    for item in result.scenarios:
        assert item.known_action_count == item.expected_action_count == 3
        assert item.unique_reference_contact_pairs >= item.object_count - 1
        assert item.unique_model_contact_pairs == item.unique_reference_contact_pairs
        assert item.contact_pair_f1 == 1.0
        assert item.first_contact_timing_error_frames is not None
        assert item.first_contact_timing_error_frames <= 1
        assert item.simultaneous_contact_frames >= 1
        assert item.maximum_position_rmse_m <= 0.015
        assert item.maximum_velocity_rmse_mps <= 0.060
        assert item.maximum_orientation_rmse_degrees <= 3.5
        assert item.finite and item.source_unchanged
    for item in result.planning:
        assert item.object_count == 8
        assert item.winner_correct and item.goal_success
        assert item.normalized_regret == 0.0
        assert item.normalized_winner_margin >= 0.05
        assert item.serial_vectorized_parity
        assert item.maximum_cost_difference == 0.0
        assert item.vectorization_speedup >= 5.0
        assert item.source_unchanged

    runs = tmp_path / "runs"
    run = runs / "multicontact-six-dof"
    summary = publish_multicontact_six_dof_scale(
        result,
        run_directory=run,
        runs_root=runs,
        archive_root=tmp_path / ".archive",
    )

    assert summary.outcome == "qualified_convergence"
    assert summary.provenance["belief_initialization"] == "controlled state-first oracle"
    assert summary.provenance["truth_withheld_after_belief_initialization"] is True
    assert summary.planning["goal"] == "N=8 multi-contact terminal position"
    assert summary.artifacts["run_bytes"] < 5 * 1024 * 1024
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert not list(run.rglob("*.png"))
    assert not list(run.rglob("*.gif"))
    assert not list(run.rglob("*.mp4"))
    dashboard = (runs / "progress" / "index.html").read_text(encoding="utf-8")
    assert "N=4 Chained And Simultaneous Six-Dof Contacts" in dashboard
    assert "N=6 Chained And Simultaneous Six-Dof Contacts" in dashboard
    assert "N=8 Chained And Simultaneous Six-Dof Contacts" in dashboard
    assert dashboard.count('class="animation-card"') == 3
    assert "Colour = object identity" in dashboard
    assert "● filled / solid = model" in dashboard
    assert "○ open / dashed = private reference" in dashboard
    assert "Two-second causal multi-contact forecasts" in dashboard
    assert "future actions and membership changes are excluded" not in dashboard
    assert 'data-orientation-role="model"' in dashboard
    assert 'data-orientation-role="truth"' in dashboard
    assert "N=8 multi-contact rollout" in dashboard
    assert "K=32 planning speedup" in dashboard
    assert "Vector latency (s)" in dashboard
    first_action = next(
        event
        for event in result.scenarios[0].animation["events"]
        if event["kind"].startswith("known action")
    )
    assert first_action["time_s"] == 0.10
    assert first_action["frame"] == 1
    assert result.scenarios[0].animation["known_actions_in_rollout"] is True
