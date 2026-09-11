from __future__ import annotations

from pathlib import Path

from world_model.evaluation.open_world_six_dof import (
    OPEN_WORLD_SIX_DOF_SCHEMA,
    capability_manifest_sha256,
    publish_open_world_six_dof_capability,
    run_open_world_six_dof_capability,
)


def test_open_world_six_dof_qualification_is_compact_and_end_to_end(tmp_path: Path) -> None:
    result = run_open_world_six_dof_capability()

    assert result.schema == OPEN_WORLD_SIX_DOF_SCHEMA
    assert result.manifest_sha256 == capability_manifest_sha256()
    assert result.qualified, result.gate_failures
    assert result.prototype_free_runtime
    assert result.discovered_count == 2
    assert result.occlusion_recovered
    assert result.final_parameter_relative_error < result.initial_parameter_relative_error
    assert result.collision_f1 == 1.0
    assert max(result.horizon_orientation_rmse_degrees.values()) < 1.0
    assert all(item.winner_correct for item in result.planning)
    assert all(item.serial_vectorized_parity for item in result.planning)
    assert not result.generated_frames_retained

    runs = tmp_path / "runs"
    run = runs / "open-world-six-dof"
    summary = publish_open_world_six_dof_capability(
        result,
        run_directory=run,
        runs_root=runs,
        archive_root=tmp_path / ".archive",
    )

    assert summary.outcome == "qualified_convergence"
    assert summary.artifacts["run_bytes"] < 5 * 1024 * 1024
    assert not any(path.suffix in {".png", ".gif", ".mp4"} for path in run.iterdir())
    report = (run / "report.html").read_text(encoding="utf-8")
    assert "Orientation RMSE across horizon" in report
    assert "Online physical identification" in report
    assert "data-orientation-role" in report
    assert "data-contact-marker" in report
