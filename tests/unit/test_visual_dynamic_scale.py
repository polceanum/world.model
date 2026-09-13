from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from world_model.evaluation.visual_dynamic_scale import (
    VISUAL_DYNAMIC_SCALE_SCHEMA,
    _gate_failures,
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
        assert item.position_curve.keys() == item.velocity_curve.keys()
        assert item.position_curve.keys() == item.orientation_curve_degrees.keys()
        assert len(item.per_object_maximum_errors) == item.object_count
        assert all(
            metrics["position_m"] >= 0.0 and metrics["velocity_mps"] >= 0.0
            for metrics in item.per_object_maximum_errors.values()
        )
        assert (
            max(metrics["position_m"] for metrics in item.per_object_maximum_errors.values())
            <= 0.015
        )
        assert (
            max(metrics["velocity_mps"] for metrics in item.per_object_maximum_errors.values())
            <= 0.075
        )
        assert (
            max(
                metrics["orientation_degrees"]
                for metrics in item.per_object_maximum_errors.values()
                if "orientation_degrees" in metrics
            )
            <= 6.0
        )
        assert item.source_unchanged and item.finite
    for item in result.planning:
        assert item.object_count == 8
        assert item.winner_correct and item.goal_success
        assert item.serial_vectorized_parity
        assert item.maximum_cost_difference <= 1.0e-6
        assert item.vectorization_speedup >= 5.0
        assert item.source_unchanged

    regressed = replace(
        result.scenarios[0],
        maximum_position_rmse_m=1.0,
    )
    failures = _gate_failures(
        (regressed, *result.scenarios[1:]),
        result.planning,
        enforce_latency=False,
    )
    assert "n4:accepted_position_envelope" in failures

    numeric_regression = replace(
        result.scenarios[2],
        maximum_orientation_rmse_degrees=(
            result.scenarios[2].maximum_orientation_rmse_degrees * (1.0 + 2.0e-6)
        ),
    )
    failures = _gate_failures(
        (*result.scenarios[:2], numeric_regression),
        result.planning,
        enforce_latency=False,
    )
    assert "n8:accepted_orientation_envelope" in failures

    object_regression = replace(
        result.scenarios[0],
        per_object_maximum_errors={
            **result.scenarios[0].per_object_maximum_errors,
            next(iter(result.scenarios[0].per_object_maximum_errors)): {
                "position_m": 1.0,
                "velocity_mps": 0.0,
            },
        },
    )
    failures = _gate_failures(
        (object_regression, *result.scenarios[1:]),
        result.planning,
        enforce_latency=False,
    )
    assert "n4:worst_object_position" in failures

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
    assert summary.provenance["incumbent_remeasured_after_reference_clock_fix"] is True
    assert set(summary.horizon_curves) >= {
        "candidate_position_rmse_m",
        "candidate_velocity_rmse_mps",
        "candidate_orientation_rmse_degrees",
    }
    assert len(summary.qualitative["regression_checks"]) == 12
    assert all(item["passed"] for item in summary.qualitative["regression_checks"])
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
    assert "Worst-slice accuracy frontier" in dashboard
    assert "Max position error (m)" in dashboard
    assert "Per-object gates: position ≤ 0.0150 m" in dashboard
    assert dashboard.count("Per-object gate</th>") == 1
    assert "State-first scale rollouts" not in dashboard
    assert "Colour = object identity" in dashboard
    assert "● filled / solid = model" in dashboard
    assert "○ open / dashed = private reference" in dashboard
