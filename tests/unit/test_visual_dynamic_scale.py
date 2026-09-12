from __future__ import annotations

import json
from pathlib import Path

from world_model.evaluation.visual_dynamic_scale import (
    VISUAL_DYNAMIC_SCALE_SCHEMA,
    publish_visual_dynamic_scale,
    run_visual_dynamic_scale,
    visual_dynamic_manifest_sha256,
)


def test_visual_dynamic_scale_qualifies_public_rgbd_to_planning(tmp_path: Path) -> None:
    result = run_visual_dynamic_scale(enforce_latency=False)

    assert result.schema == VISUAL_DYNAMIC_SCALE_SCHEMA
    assert result.manifest_sha256 == visual_dynamic_manifest_sha256()
    assert result.qualified, result.gate_failures
    assert [item.object_count for item in result.scenarios] == [4, 6, 8]
    for item in result.scenarios:
        assert item.proposal_f1 >= 0.95
        assert item.persistent_id_accuracy >= 0.98
        assert item.lifecycle_f1 >= 0.95
        assert item.two_frame_birth_confirmation
        assert item.two_miss_removal
        assert item.partial_visibility_recovered
        assert item.primitive_accuracy == 1.0
        assert item.current_position_rmse_m <= 0.020
        assert item.known_action_count == item.expected_action_count == 3
        assert item.contact_pair_f1 >= 0.90
        assert item.simultaneous_contact_frames >= 1
        assert item.maximum_position_rmse_m <= 0.030
        assert item.maximum_velocity_rmse_mps <= 0.100
        assert item.maximum_orientation_rmse_degrees <= 10.0
        assert item.source_unchanged and item.finite
    for item in result.planning:
        assert item.object_count == 8
        assert item.winner_correct and item.goal_success
        assert item.serial_vectorized_parity
        assert item.maximum_cost_difference <= 1.0e-6
        assert item.vectorization_speedup >= 5.0
        assert item.source_unchanged

    runs = tmp_path / "runs"
    run = runs / "visual-dynamic-scale"
    summary = publish_visual_dynamic_scale(
        result,
        run_directory=run,
        runs_root=runs,
        archive_root=tmp_path / ".archive",
    )

    assert summary.outcome == "qualified_convergence"
    assert summary.provenance["belief_initialization"].startswith("prototype-free public")
    assert summary.provenance["runtime_truth_inputs"] is False
    assert summary.artifacts["run_bytes"] < 5 * 1024 * 1024
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert not list(run.rglob("*.png"))
    assert not list(run.rglob("*.gif"))
    assert not list(run.rglob("*.mp4"))
    dashboard = (runs / "progress" / "index.html").read_text(encoding="utf-8")
    assert "N=4 RGB-D Initialized Multi-Contact Forecast" in dashboard
    assert "N=6 RGB-D Initialized Multi-Contact Forecast" in dashboard
    assert "N=8 RGB-D Initialized Multi-Contact Forecast" in dashboard
    assert "Public RGB-D initialized rollouts" in dashboard
    assert "State-first scale rollouts" not in dashboard
    assert "Colour = object identity" in dashboard
    assert "● filled / solid = model" in dashboard
    assert "○ open / dashed = private reference" in dashboard
