from __future__ import annotations

import json
from pathlib import Path

from world_model.evaluation.open_world_touching_recovery import (
    OPEN_WORLD_TOUCHING_RECOVERY_SCHEMA,
    capability_manifest_sha256,
    publish_open_world_touching_recovery_capability,
    run_open_world_touching_recovery_capability,
)


def test_touching_recovery_milestone_qualifies_and_stays_compact(tmp_path: Path) -> None:
    first_hash = capability_manifest_sha256()
    second_hash = capability_manifest_sha256()
    # Absolute CPU timing is adjudicated by the fresh-process governed runner;
    # a late full-suite process inherits unrelated thread-pool/thermal state.
    result = run_open_world_touching_recovery_capability(enforce_latency=False)

    assert first_hash == second_hash == result.manifest_sha256
    assert result.schema == OPEN_WORLD_TOUCHING_RECOVERY_SCHEMA
    assert result.qualified, result.gate_failures
    assert result.connected_depth_components == 2
    assert result.geometry_split_count == result.object_count == 3
    assert result.identical_appearance
    assert result.appearance_weight == 0.0
    assert result.primitive_weight == 0.0
    assert result.dropout_frames == 8
    assert result.all_tracks_retained
    assert result.recovery_id_accuracy == 1.0
    assert result.duplicate_id_count == 0
    assert result.maximum_gap_position_rmse_m <= 0.04
    assert result.recovery_position_rmse_m <= 0.03
    assert result.recovery_velocity_rmse_mps <= 0.08
    assert all(item.winner_correct for item in result.planning)
    assert all(item.serial_vectorized_parity for item in result.planning)
    assert all(item.latency_seconds > 0.0 for item in result.planning)
    assert all(item.maximum_cost_difference == 0.0 for item in result.planning)

    runs = tmp_path / "runs"
    run = runs / "touching-recovery"
    summary = publish_open_world_touching_recovery_capability(
        result,
        run_directory=run,
        runs_root=runs,
        archive_root=tmp_path / ".archive",
    )

    assert summary.outcome == "qualified_convergence"
    assert summary.artifacts["run_bytes"] < 5 * 1024 * 1024
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert not list(run.rglob("*.png"))
    assert not list(run.rglob("*.gif"))
    assert not list(run.rglob("*.mp4"))
    dashboard = (runs / "progress" / "index.html").read_text(encoding="utf-8")
    assert "same-appearance touching split and eight-frame recovery" in dashboard
    assert "Position RMSE through observation gap and recovery" in dashboard
    assert "Colour = object identity" in dashboard
