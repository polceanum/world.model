from __future__ import annotations

import json
from pathlib import Path

from world_model.evaluation.capability_hardening import (
    CAPABILITY_HARDENING_SCHEMA,
    CapabilityHardeningResult,
    RegressionCheck,
    capability_hardening_manifest_sha256,
    publish_capability_hardening,
)


def test_hardening_manifest_and_compact_report_are_versioned(tmp_path: Path) -> None:
    check = RegressionCheck(
        family="visual_n8",
        metric="maximum_position_rmse_m",
        direction="lower",
        baseline=0.014,
        candidate=0.004,
        limit=0.007,
        passed=True,
    )
    result = CapabilityHardeningResult(
        schema=CAPABILITY_HARDENING_SCHEMA,
        manifest_sha256=capability_hardening_manifest_sha256(),
        checks=(check,),
        source_gate_failures={"visual_dynamic": ()},
        visual_position_curve={"0.05": 0.001, "2.00": 0.004},
        protected_long_horizon_curve={"0.05": 1.0e-7, "8": 0.0004},
        planning_by_candidate_count={
            "8": {
                "winner_accuracy": 1.0,
                "goal_success": 1.0,
                "median_normalized_regret": 0.0,
                "maximum_cost_difference": 0.0,
                "serial_vectorized_parity": 1.0,
                "task_count": 6.0,
            }
        },
        forecast_animations=(),
        learned_weight_bytes=12_984,
        evaluation_seconds=1.0,
        gate_failures=(),
        qualified=True,
    )
    runs = tmp_path / "runs"
    run = runs / "hardening"

    summary = publish_capability_hardening(
        result,
        run_directory=run,
        runs_root=runs,
        archive_root=tmp_path / ".archive",
    )

    assert summary.outcome == "qualified_convergence"
    assert summary.scores["candidate"]["value"] < summary.scores["incumbent"]["value"]
    assert summary.provenance["paired_accepted_baselines"] is True
    assert summary.selection["promotion_evaluated"] is False
    report = (run / "report.html").read_text(encoding="utf-8")
    assert "Cross-capability regression envelope" in report
    assert "Protected long-horizon position RMSE" in report
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert not any(path.suffix in {".gif", ".mp4", ".png"} for path in run.iterdir())


def test_near_zero_unchanged_baseline_has_neutral_family_score(tmp_path: Path) -> None:
    position_check = RegressionCheck(
        family="open_world",
        metric="maximum_position_rmse_m",
        direction="lower",
        baseline=0.001,
        candidate=0.001,
        limit=0.00102,
        passed=True,
    )
    parameter_check = RegressionCheck(
        family="open_world",
        metric="final_parameter_relative_error",
        direction="lower",
        baseline=3.0e-15,
        candidate=3.0e-15,
        limit=1.0e-10,
        passed=True,
    )
    result = CapabilityHardeningResult(
        schema=CAPABILITY_HARDENING_SCHEMA,
        manifest_sha256=capability_hardening_manifest_sha256(),
        checks=(position_check, parameter_check),
        source_gate_failures={"open_world": ()},
        visual_position_curve={"2.00": 0.004},
        protected_long_horizon_curve={"8": 0.0004},
        planning_by_candidate_count={
            "8": {
                "winner_accuracy": 1.0,
                "goal_success": 1.0,
                "median_normalized_regret": 0.0,
                "maximum_cost_difference": 0.0,
                "serial_vectorized_parity": 1.0,
                "task_count": 1.0,
            }
        },
        forecast_animations=(),
        learned_weight_bytes=12_984,
        evaluation_seconds=1.0,
        gate_failures=(),
        qualified=True,
    )

    summary = publish_capability_hardening(
        result,
        run_directory=tmp_path / "runs" / "hardening",
        runs_root=tmp_path / "runs",
        archive_root=tmp_path / ".archive",
    )

    assert summary.factor_metrics["open_world"]["score"] == 1.0
