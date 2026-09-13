"""Cross-capability non-regression and worst-slice improvement gate."""

from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.evaluation.integrated_capability import run_integrated_capability
from world_model.evaluation.long_horizon import run_long_horizon_evaluation
from world_model.evaluation.multicontact_six_dof import run_multicontact_six_dof_scale
from world_model.evaluation.open_world_six_dof import run_open_world_six_dof_capability
from world_model.evaluation.open_world_touching_recovery import (
    run_open_world_touching_recovery_capability,
)
from world_model.evaluation.visual_dynamic_scale import run_visual_dynamic_scale
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

CAPABILITY_HARDENING_SCHEMA = "world_model_capability_hardening_v1"

_BASELINES = {
    "long_horizon/worst_position_rmse_m": 0.0010364428162574768,
    "long_horizon/eight_second_position_rmse_m": 0.0005994171951897442,
    "long_horizon/worst_collision_f1": 0.9166666666666666,
    "integrated/four_second_position_rmse_m": 2.8240331474675584e-06,
    "integrated/four_second_velocity_rmse_mps": 1.6935332408412029e-06,
    "open_world/maximum_position_rmse_m": 0.0005103963893258285,
    "open_world/maximum_orientation_rmse_degrees": 0.2666923178514557,
    "open_world/final_parameter_relative_error": 3.2425745413578546e-15,
    "recovery/maximum_gap_position_rmse_m": 0.03405210558082378,
    "recovery/recovery_position_rmse_m": 0.003109267799783237,
    "recovery/recovery_velocity_rmse_mps": 0.026675342145522182,
    "multicontact_n4/maximum_position_rmse_m": 0.004860520461510193,
    "multicontact_n4/maximum_velocity_rmse_mps": 0.0036648750847652492,
    "multicontact_n4/maximum_orientation_rmse_degrees": 3.228122353610414,
    "multicontact_n6/maximum_position_rmse_m": 0.00523376872512957,
    "multicontact_n6/maximum_velocity_rmse_mps": 0.038635013312279025,
    "multicontact_n6/maximum_orientation_rmse_degrees": 2.887979771407587,
    "multicontact_n8/maximum_position_rmse_m": 0.003649585997230939,
    "multicontact_n8/maximum_velocity_rmse_mps": 0.04388566763042456,
    "multicontact_n8/maximum_orientation_rmse_degrees": 2.7751405832837652,
    "visual_n4/maximum_position_rmse_m": 0.0063384447408479745,
    "visual_n4/maximum_velocity_rmse_mps": 0.028042441735440335,
    "visual_n4/maximum_orientation_rmse_degrees": 5.177510973549666,
    "visual_n4/repeated_contact_f1": 0.875,
    "visual_n6/maximum_position_rmse_m": 0.009352914348502018,
    "visual_n6/maximum_velocity_rmse_mps": 0.09070518811208458,
    "visual_n6/maximum_orientation_rmse_degrees": 9.278944513978788,
    "visual_n6/repeated_contact_f1": 0.7560975609756098,
    "visual_n8/maximum_position_rmse_m": 0.01378871975459622,
    "visual_n8/maximum_velocity_rmse_mps": 0.0555040703509804,
    "visual_n8/maximum_orientation_rmse_degrees": 9.40562622288037,
    "visual_n8/repeated_contact_f1": 0.6133333333333333,
}


@dataclass(frozen=True, slots=True)
class RegressionCheck:
    family: str
    metric: str
    direction: str
    baseline: float
    candidate: float
    limit: float
    passed: bool


@dataclass(frozen=True, slots=True)
class CapabilityHardeningResult:
    schema: str
    manifest_sha256: str
    checks: tuple[RegressionCheck, ...]
    source_gate_failures: dict[str, tuple[str, ...]]
    visual_position_curve: dict[str, float]
    protected_long_horizon_curve: dict[str, float]
    planning_by_candidate_count: dict[str, dict[str, float]]
    forecast_animations: tuple[dict[str, Any], ...]
    learned_weight_bytes: int
    evaluation_seconds: float
    gate_failures: tuple[str, ...]
    qualified: bool
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capability_hardening_manifest_sha256() -> str:
    payload = {
        "accepted_baselines": _BASELINES,
        "accuracy_regression_ratio": 1.02,
        "visual_n8_required_improvement": {
            "position": 0.50,
            "velocity": 0.50,
            "orientation": 0.60,
        },
        "source_tiers": [
            "long_horizon",
            "integrated",
            "open_world",
            "touching_recovery",
            "multicontact",
            "visual_dynamic",
        ],
        "planning_used_as_training_loss": False,
        "absolute_latency_adjudication": "separate fresh-process source runners",
        "retained_media": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _lower_check(
    family: str,
    metric: str,
    candidate: float,
    *,
    ratio: float = 1.02,
    absolute_tolerance: float = 1.0e-12,
) -> RegressionCheck:
    baseline = _BASELINES[f"{family}/{metric}"]
    limit = baseline * ratio + absolute_tolerance
    return RegressionCheck(family, metric, "lower", baseline, candidate, limit, candidate <= limit)


def _higher_check(
    family: str,
    metric: str,
    candidate: float,
    *,
    absolute_tolerance: float = 0.02,
) -> RegressionCheck:
    baseline = _BASELINES[f"{family}/{metric}"]
    limit = baseline - absolute_tolerance
    return RegressionCheck(family, metric, "higher", baseline, candidate, limit, candidate >= limit)


def _aggregate_curve(scenarios: list[dict[str, Any]]) -> dict[str, float]:
    by_horizon: dict[str, list[float]] = {}
    for scenario in scenarios:
        for horizon, value in scenario["position_rmse_m"].items():
            by_horizon.setdefault(str(horizon), []).append(float(value))
    return {
        horizon: max(values)
        for horizon, values in sorted(by_horizon.items(), key=lambda item: float(item[0]))
    }


def _planning_ledger(*groups: tuple[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    by_count: dict[int, list[dict[str, Any]]] = {}
    for family, tasks in groups:
        for task in tasks:
            candidate_count = int(task["candidate_count"])
            winner = task.get("winner_correct", task.get("serial_vectorized_winner_parity", False))
            goal = task.get("goal_success", task.get("terminal_goal_success", False))
            by_count.setdefault(candidate_count, []).append(
                {
                    "family": family,
                    "winner": float(bool(winner)),
                    "goal": float(bool(goal)),
                    "regret": float(task.get("normalized_regret", 0.0)),
                    "cost": float(task.get("maximum_cost_difference", 0.0)),
                    "parity": float(
                        bool(
                            task.get(
                                "serial_vectorized_parity",
                                task.get("serial_vectorized_winner_parity", False),
                            )
                        )
                    ),
                }
            )
    return {
        str(candidate_count): {
            "winner_accuracy": statistics.fmean(item["winner"] for item in tasks),
            "goal_success": statistics.fmean(item["goal"] for item in tasks),
            "median_normalized_regret": statistics.median(item["regret"] for item in tasks),
            "maximum_cost_difference": max(item["cost"] for item in tasks),
            "serial_vectorized_parity": min(item["parity"] for item in tasks),
            "task_count": float(len(tasks)),
        }
        for candidate_count, tasks in sorted(by_count.items())
    }


def run_capability_hardening(
    *,
    model_config_path: str | Path = "configs/rgbd_dynamic_set_planning_cpu.yaml",
    checkpoint_path: str | Path = "runs/20260907-capability-development-v2/selected_checkpoint.pt",
    enforce_latency: bool = False,
) -> CapabilityHardeningResult:
    """Rerun every accepted behavioral tier and enforce paired non-regression.

    The composite process deliberately disables absolute latency by default:
    source runners adjudicate those bounds in isolated fresh processes, while
    this long sequential process owns numerical and behavioral non-regression.
    """

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="world-model-hardening-") as temporary:
        long_report = run_long_horizon_evaluation(
            model_config_path=model_config_path,
            checkpoint_path=checkpoint_path,
            run_directory=Path(temporary) / "long-horizon",
            seed=0,
            threads=1,
            progress=None,
        )
    integrated = run_integrated_capability()
    open_world = run_open_world_six_dof_capability(enforce_latency=enforce_latency)
    recovery = run_open_world_touching_recovery_capability(enforce_latency=enforce_latency)
    multicontact = run_multicontact_six_dof_scale(enforce_latency=enforce_latency)
    visual = run_visual_dynamic_scale(enforce_latency=enforce_latency)

    long_scenarios = list(long_report["scenarios"])
    long_curve = _aggregate_curve(long_scenarios)
    checks = [
        _lower_check(
            "long_horizon",
            "worst_position_rmse_m",
            max(
                float(value)
                for item in long_scenarios
                for value in item["position_rmse_m"].values()
            ),
        ),
        _lower_check(
            "long_horizon",
            "eight_second_position_rmse_m",
            max(
                float(item["position_rmse_m"]["8"])
                for item in long_scenarios
                if "8" in item["position_rmse_m"]
            ),
        ),
        _higher_check(
            "long_horizon",
            "worst_collision_f1",
            min(float(item["collision_f1"]) for item in long_scenarios),
        ),
        _lower_check(
            "integrated",
            "four_second_position_rmse_m",
            integrated.four_second_position_rmse_m,
            absolute_tolerance=1.0e-8,
        ),
        _lower_check(
            "integrated",
            "four_second_velocity_rmse_mps",
            integrated.four_second_velocity_rmse_mps,
            absolute_tolerance=1.0e-8,
        ),
        _lower_check(
            "open_world",
            "maximum_position_rmse_m",
            max(open_world.horizon_position_rmse_m.values()),
        ),
        _lower_check(
            "open_world",
            "maximum_orientation_rmse_degrees",
            max(open_world.horizon_orientation_rmse_degrees.values()),
        ),
        _lower_check(
            "open_world",
            "final_parameter_relative_error",
            open_world.final_parameter_relative_error,
            absolute_tolerance=1.0e-10,
        ),
        _lower_check(
            "recovery",
            "maximum_gap_position_rmse_m",
            recovery.maximum_gap_position_rmse_m,
        ),
        _lower_check(
            "recovery",
            "recovery_position_rmse_m",
            recovery.recovery_position_rmse_m,
        ),
        _lower_check(
            "recovery",
            "recovery_velocity_rmse_mps",
            recovery.recovery_velocity_rmse_mps,
        ),
    ]
    for item in multicontact.scenarios:
        family = f"multicontact_n{item.object_count}"
        checks.extend(
            (
                _lower_check(family, "maximum_position_rmse_m", item.maximum_position_rmse_m),
                _lower_check(family, "maximum_velocity_rmse_mps", item.maximum_velocity_rmse_mps),
                _lower_check(
                    family,
                    "maximum_orientation_rmse_degrees",
                    item.maximum_orientation_rmse_degrees,
                ),
            )
        )
    for item in visual.scenarios:
        family = f"visual_n{item.object_count}"
        improvement = item.object_count == 8
        checks.extend(
            (
                _lower_check(
                    family,
                    "maximum_position_rmse_m",
                    item.maximum_position_rmse_m,
                    ratio=0.50 if improvement else 1.02,
                ),
                _lower_check(
                    family,
                    "maximum_velocity_rmse_mps",
                    item.maximum_velocity_rmse_mps,
                    ratio=0.50 if improvement else 1.02,
                ),
                _lower_check(
                    family,
                    "maximum_orientation_rmse_degrees",
                    item.maximum_orientation_rmse_degrees,
                    ratio=0.60 if improvement else 1.02,
                ),
                _higher_check(
                    family,
                    "repeated_contact_f1",
                    item.repeated_contact_frame_f1,
                ),
            )
        )

    source_failures = {
        "long_horizon": tuple(str(item) for item in long_report["gate_failures"]),
        "integrated": integrated.gate_failures,
        "open_world": open_world.gate_failures,
        "touching_recovery": recovery.gate_failures,
        "multicontact": multicontact.gate_failures,
        "visual_dynamic": visual.gate_failures,
    }
    failures = [f"{check.family}:{check.metric}" for check in checks if not check.passed]
    failures.extend(
        f"{family}:{failure}"
        for family, family_failures in source_failures.items()
        for failure in family_failures
    )
    visual_curve = {
        horizon: max(item.position_curve[horizon] for item in visual.scenarios)
        for horizon in visual.scenarios[0].position_curve
    }
    planning = _planning_ledger(
        ("long_horizon", list(long_report["planning_tasks"])),
        ("integrated", [asdict(item) for item in integrated.planning]),
        ("open_world", [asdict(item) for item in open_world.planning]),
        ("recovery", [asdict(item) for item in recovery.planning]),
        ("multicontact", [asdict(item) for item in multicontact.planning]),
        ("visual", [asdict(item) for item in visual.planning]),
    )
    if any(
        values["winner_accuracy"] < 1.0
        or values["goal_success"] < 1.0
        or values["median_normalized_regret"] > 0.0
        or values["maximum_cost_difference"] > 1.0e-6
        or values["serial_vectorized_parity"] < 1.0
        for values in planning.values()
    ):
        failures.append("cross_capability_planning")
    learned_bytes = max(
        integrated.learned_weight_bytes,
        open_world.learned_weight_bytes,
        recovery.learned_weight_bytes,
        multicontact.learned_weight_bytes,
        visual.learned_weight_bytes,
    )
    return CapabilityHardeningResult(
        schema=CAPABILITY_HARDENING_SCHEMA,
        manifest_sha256=capability_hardening_manifest_sha256(),
        checks=tuple(checks),
        source_gate_failures=source_failures,
        visual_position_curve=visual_curve,
        protected_long_horizon_curve=long_curve,
        planning_by_candidate_count=planning,
        forecast_animations=tuple(item.animation for item in visual.scenarios),
        learned_weight_bytes=learned_bytes,
        evaluation_seconds=time.perf_counter() - started,
        gate_failures=tuple(failures),
        qualified=not failures,
    )


def _summary(
    result: CapabilityHardeningResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    def paired_ratio(check: RegressionCheck) -> float:
        numerator = check.candidate if check.direction == "lower" else check.baseline
        denominator = check.baseline if check.direction == "lower" else check.candidate
        if denominator == 0.0:
            return 1.0 if numerator == 0.0 else 1.0 + abs(numerator)
        return numerator / denominator

    by_family: dict[str, list[RegressionCheck]] = {}
    for check in result.checks:
        by_family.setdefault(check.family, []).append(check)
    primary_ratios = [
        check.candidate / max(check.baseline, 1.0e-12)
        for check in result.checks
        if check.direction == "lower" and "position_rmse" in check.metric
    ]
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.qualified else "failed",
        outcome="qualified_convergence" if result.qualified else "capability_gate_failed",
        source_format=CAPABILITY_HARDENING_SCHEMA,
        configuration={
            "comparison_protocol": "accepted_behavioral_tiers_v1",
            "accuracy_regression_limit": 0.02,
            "visual_n8_worst_slice_improvement_required": True,
            "planning_used_as_training_loss": False,
            "generated_frames_retained": False,
            "absolute_latency_adjudication": "separate fresh-process source runners",
        },
        provenance={
            "hardening_manifest_sha256": result.manifest_sha256,
            "paired_accepted_baselines": True,
            "public_visual_runtime_truth_inputs": False,
            "private_references_opened_after_inference": True,
            "isolated_latency_attestations": [
                "20260913-multicontact-hardening-v2",
                "20260913-visual-dynamic-hardening-v3",
            ],
        },
        scores={
            "candidate": {
                "value": statistics.fmean(primary_ratios),
                "supported_weight": float(len(primary_ratios)),
            },
            "incumbent": {"value": 1.0, "supported_weight": float(len(primary_ratios))},
            "selected": "hardened_public_state" if result.qualified else "incumbent",
        },
        factor_metrics={
            family: {
                "status": "passed" if all(item.passed for item in checks) else "failed",
                "score": statistics.fmean(paired_ratio(item) for item in checks),
            }
            for family, checks in by_family.items()
        },
        cell_metrics={},
        horizon_curves={
            "candidate_position_rmse_m": result.visual_position_curve,
            "protected_long_horizon_position_rmse_m": result.protected_long_horizon_curve,
        },
        uncertainty={"status": "preserved by source-tier gates"},
        planning={
            "status": "passed"
            if "cross_capability_planning" not in result.gate_failures
            else "failed",
            "goal": "cross-capability action selection",
            "by_candidate_count": result.planning_by_candidate_count,
            "serial_vectorized_winner_parity": all(
                item["serial_vectorized_parity"] == 1.0
                for item in result.planning_by_candidate_count.values()
            ),
            "maximum_cost_difference": max(
                item["maximum_cost_difference"]
                for item in result.planning_by_candidate_count.values()
            ),
        },
        resources={
            "evaluation_seconds": result.evaluation_seconds,
            "learned_weight_bytes": result.learned_weight_bytes,
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "hardened_public_state" if result.qualified else "none",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "public box orientation at the causal anchor",
        },
        qualitative={
            "best_episode": "protected long-horizon causal tiers",
            "representative_episode": "visual-dynamic-n6 unchanged",
            "worst_episode": "visual-dynamic-n8 hardened",
            "forecast_gallery_mode": "latest_run",
            "animations": [],
            "forecast_animations": list(result.forecast_animations),
            "regression_checks": [asdict(item) for item in result.checks],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "unknown camera calibration",
            "hidden actions",
            "visual qualification above eight objects",
        ),
        scope_limitations=(
            "fixed accepted manifests only",
            "known physical parameters in the visual scale tier",
            "planning freezes each active set",
        ),
    ).validate()


def publish_capability_hardening(
    result: CapabilityHardeningResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "capability_hardening.json",
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_bytes = int(inventory_runs(runs_root, archive_root=archive_root)["archive_bytes"])
    summary = _summary(result, run_id=run.name, run_bytes=0, archive_bytes=archive_bytes)
    for _ in range(8):
        write_capability_summary(summary, run / "capability_summary.json")
        write_run_report(summary, run)
        write_run_manifest(
            run,
            role="candidate",
            status="completed" if result.qualified else "failed",
            artifacts={
                "capability_hardening.json": "summary",
                "capability_summary.json": "summary",
                "report.html": "report",
            },
        )
        actual = sum(
            path.stat().st_size
            for path in run.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        if summary.artifacts["run_bytes"] == actual:
            break
        summary = replace(summary, artifacts={**summary.artifacts, "run_bytes": actual})
    else:
        raise RuntimeError("hardening evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "CAPABILITY_HARDENING_SCHEMA",
    "CapabilityHardeningResult",
    "RegressionCheck",
    "capability_hardening_manifest_sha256",
    "publish_capability_hardening",
    "run_capability_hardening",
]
