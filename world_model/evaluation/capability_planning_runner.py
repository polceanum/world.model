"""Compact factor-conditioned downstream planning evaluation.

This module adapts the broad capability manifest to the existing public
planning interfaces. Environment controls affect only task generation and
RGB-D history. Checkpoint inference receives calibrated RGB-D and the frozen
candidate action bank; the private oracle is opened only by the established
post-decision scorer. No planning metric is exposed as a training loss.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world_model.evaluation.capability_factor_runner import (
    SUPPORTED_CAPABILITY_FACTORS,
    CapabilityFactor,
    _jsonable,
    _model_from_workbench_checkpoint,
    _seed_process,
)
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.evaluation.general_capability import (
    ALL_CAPABILITY_FACTORS,
    COMPOSITIONAL_FACTOR,
    CapabilityManifestRow,
    FactorControls,
    capability_manifest,
    manifest_sha256,
)
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_planning_materializer import (
    PlanningEnvironmentControls,
    PlanningPopulationEvaluationConfig,
    PlanningTaskMaterializationError,
    PlanningTaskUnresolvedError,
    evaluate_development_planning_materializations,
    materialize_capability_planning_task,
)
from world_model.training.dynamic_set_protocol import PlanningManifestRow
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import enforce_run_budget, inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

CAPABILITY_PLANNING_REPORT_SCHEMA = "world_model_capability_planning_report_v1"
_ACTION_TIME_STRATUM = {"early": 0, "middle": 2, "late": 3}


def factor_planning_rows(
    factor: CapabilityFactor,
) -> tuple[tuple[CapabilityManifestRow, PlanningManifestRow], ...]:
    """Bind one capability family to every N=1--6/K=8,32 planning slice."""

    if factor not in SUPPORTED_CAPABILITY_FACTORS:
        raise ValueError(f"factor must be one of {SUPPORTED_CAPABILITY_FACTORS}")
    split = "compositional_holdout" if factor == COMPOSITIONAL_FACTOR else "development"
    capability_rows = tuple(
        row for row in capability_manifest(split) if row.kind == "planning" and row.factor == factor
    )
    if len(capability_rows) != 12:
        raise RuntimeError("capability planning family must contain 12 N/K slices")
    result: list[tuple[CapabilityManifestRow, PlanningManifestRow]] = []
    for row in capability_rows:
        result.append(
            (
                row,
                PlanningManifestRow(
                    # Both capability namespaces are public development data.
                    # The separate 92M seed space and provenance retain the
                    # compositional holdout distinction without opening the
                    # historical protected compositional_ood split.
                    split="development",
                    ordinal=row.ordinal,
                    seed=row.seed,
                    object_count=row.object_count,
                    previously_dynamic=row.dynamic_membership,
                    candidate_induced_contact=row.contact,
                    target_rank=row.controls.impulse_target_rank,
                    action_time_stratum=_ACTION_TIME_STRATUM[row.controls.impulse_phase],
                    camera_stratum=row.ordinal % 8,
                    goal_direction=row.ordinal % 6,
                    candidate_count=row.candidate_count,  # type: ignore[arg-type]
                    minimum_normalized_winner_margin=0.05,
                    distribution="in_distribution",
                ),
            )
        )
    return tuple(result)


def _environment(factor: CapabilityFactor, controls: FactorControls) -> PlanningEnvironmentControls:
    sensor = factor in {"sensor_noise", COMPOSITIONAL_FACTOR}
    physics = factor in {"physical_parameters", COMPOSITIONAL_FACTOR}
    camera = factor in {"camera_motion", COMPOSITIONAL_FACTOR}
    visibility = factor in {"partial_visibility", COMPOSITIONAL_FACTOR}
    action = factor in {"known_actions", COMPOSITIONAL_FACTOR}
    return PlanningEnvironmentControls(
        rgb_noise_std=controls.rgb_noise_std if sensor else 0.0,
        exposure_scale=controls.exposure_scale if sensor else 1.0,
        depth_noise_std_m=controls.depth_noise_std_m if sensor else 0.0,
        pixel_dropout_probability=controls.pixel_dropout_probability if sensor else 0.0,
        radius_m=controls.radius_m if physics else 0.21,
        mass_kg=controls.mass_kg if physics else 1.0,
        drag_per_second=controls.drag_per_second if physics else 0.05,
        restitution=controls.restitution if physics else 0.70,
        friction=controls.friction if physics else 0.20,
        camera_motion=controls.camera_motion if camera else "static",
        occlusion_frames=controls.occlusion_frames if visibility else 0,
        candidate_delta_velocity_mps=controls.impulse_magnitude if action else 0.50,
    ).validate()


def _supported_value(value: Any) -> float:
    measured = getattr(value, "value", None)
    support = getattr(value, "support", None)
    if not isinstance(measured, (int, float)) or not isinstance(support, int) or support <= 0:
        raise ValueError("planning gate metric lacks support")
    number = float(measured)
    if not math.isfinite(number):
        raise ValueError("planning gate metric is nonfinite")
    return number


def _planning_gate_failures(result: Any, *, compositional: bool) -> tuple[str, ...]:
    gates = result.reduction.gate_metrics
    failures: list[str] = []
    for count in range(1, 7):
        if _supported_value(gates.handle_resolution_by_object_count[f"N{count}"]) < 0.99:
            unresolved = [
                outcome.failure_reason or ""
                for outcome in result.outcomes
                if outcome.evaluation is None and outcome.row.object_count == count
            ]
            if unresolved and all("not mature" in reason for reason in unresolved):
                failures.append(f"N{count}:mature_state_availability")
            elif unresolved and all("appearance handle" not in reason for reason in unresolved):
                failures.append(f"N{count}:stable_state_availability")
            else:
                failures.append(f"N{count}:target_handle_resolution")
    winner_limits = {8: 0.80 if compositional else 0.90, 32: 0.75 if compositional else 0.85}
    regret_limits = {8: 0.05, 32: 0.07}
    for candidates in (8, 32):
        winner = _supported_value(
            gates.oracle_winner_accuracy_by_candidate_distribution[f"K{candidates}/in_distribution"]
        )
        if winner < winner_limits[candidates]:
            failures.append(f"K{candidates}:oracle_winner_accuracy")
        regret = _supported_value(
            gates.normalized_regret_median_by_candidate_count[f"K{candidates}"]
        )
        if regret > regret_limits[candidates]:
            failures.append(f"K{candidates}:median_normalized_regret")
    if _supported_value(gates.successful_oracle_goal_success) < 0.90:
        failures.append("selected_action_goal_success")
    invariants = result.invariants
    if any(outcome.evaluation is None for outcome in result.outcomes):
        # The established evaluator cannot construct its complete stratified
        # invariant cover after a public task fails before rollout. Record the
        # resulting absence as a required-gate failure without claiming that
        # each individual invariant was observed to be false.
        failures.append("planning_invariants:unmeasured_after_task_failure")
    else:
        for name in (
            "serial_vectorized_winner_parity",
            "pre_action_invariance",
            "exactly_once_impulse",
            "action_target_isolation",
            "conservation",
            "batch_independence",
            "source_belief_unchanged",
        ):
            if not bool(getattr(invariants, name)):
                failures.append(name)
        if float(invariants.maximum_cost_difference) > 1.0e-6:
            failures.append("serial_vectorized_cost_agreement")
    return tuple(failures)


def _summary(
    *,
    run_id: str,
    factor: CapabilityFactor,
    result: Any,
    gate_failures: Sequence[str],
    provenance: Mapping[str, Any],
    created_at_utc: str,
    artifact_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    planning = {
        "status": "passed" if not gate_failures else "failed",
        "factor": factor,
        "task_count": len(result.outcomes),
        "planning_error": result.planning_error,
        "slices": _jsonable(result.reduction.slices),
        "gate_metrics": _jsonable(result.reduction.gate_metrics),
        "serial_vectorized_winner_parity": result.reduction.serial_vectorized_winner_parity,
        "maximum_cost_difference": result.reduction.maximum_cost_difference,
        "invariants": _jsonable(result.invariants),
        "gate_failures": list(gate_failures),
    }
    factor_metrics = {
        name: {"status": "unmeasured", "reason": "physical factor not executed in this run"}
        for name in ALL_CAPABILITY_FACTORS
    }
    unsupported = [f"{name} factor-conditioned planning" for name in ALL_CAPABILITY_FACTORS]
    unsupported.remove(f"{factor} factor-conditioned planning")
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=created_at_utc,
        lifecycle_status="completed",
        outcome="planning_passed" if not gate_failures else "planning_failed",
        source_format=CAPABILITY_PLANNING_REPORT_SCHEMA,
        configuration={"factor": factor, "planning_tasks": len(result.outcomes)},
        provenance=dict(provenance),
        scores={
            "candidate": {"value": result.planning_error, "supported_weight": 0.15},
            "incumbent": {},
            "selected": "calibrated_structured_incumbent",
        },
        factor_metrics=factor_metrics,
        cell_metrics={},
        horizon_curves={},
        uncertainty={},
        planning=planning,
        resources={},
        artifacts={"run_bytes": artifact_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "calibrated_structured_incumbent",
            "promotion_evaluated": False,
            "gate_failures": list(gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": "none" if not gate_failures else ", ".join(gate_failures),
            "ablation_owner": "downstream_planning",
        },
        qualitative={
            "best_episode": "unavailable for categorical task outcome",
            "worst_episode": "unavailable for categorical task outcome",
            "representative_episode": f"{factor}:12 deterministic N/K tasks",
            "preview_image": "not retained",
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=tuple(unsupported),
        scope_limitations=(
            "hidden actions",
            "unknown camera calibration",
            "new modalities",
            "deformable or articulated bodies",
            "long-term scene memory",
        ),
    ).validate()


def _failure_summary(
    *,
    run_id: str,
    factor: CapabilityFactor,
    error: Exception,
    completed_tasks: int,
    provenance: Mapping[str, Any],
    created_at_utc: str,
    artifact_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=created_at_utc,
        lifecycle_status="failed",
        outcome="planning_failed",
        source_format=CAPABILITY_PLANNING_REPORT_SCHEMA,
        configuration={"factor": factor, "completed_planning_tasks": completed_tasks},
        provenance=dict(provenance),
        scores={"candidate": {}, "incumbent": {}, "selected": "calibrated_structured_incumbent"},
        factor_metrics={
            name: {"status": "unmeasured", "reason": "physical factor not executed in this run"}
            for name in ALL_CAPABILITY_FACTORS
        },
        cell_metrics={},
        horizon_curves={},
        uncertainty={},
        planning={
            "status": "failed",
            "factor": factor,
            "completed_tasks": completed_tasks,
            "runtime_error": f"{type(error).__name__}: {error}",
            "gate_failures": ["planning_population_completion"],
        },
        resources={},
        artifacts={"run_bytes": artifact_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "calibrated_structured_incumbent",
            "promotion_evaluated": False,
            "gate_failures": ["planning_population_completion"],
        },
        failure_attribution={
            "primary_bottleneck": "factor-conditioned planning materialization or inference",
            "ablation_owner": "downstream_planning",
            "runtime_error": f"{type(error).__name__}: {error}",
        },
        qualitative={
            "best_episode": "unavailable after fail-closed stop",
            "worst_episode": f"{factor}:task-{completed_tasks + 1}",
            "representative_episode": "unavailable after fail-closed stop",
            "preview_image": "not retained",
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(f"{factor} factor-conditioned planning",),
        scope_limitations=(
            "hidden actions",
            "unknown camera calibration",
            "new modalities",
            "deformable or articulated bodies",
            "long-term scene memory",
        ),
    ).validate()


def run_incumbent_factor_planning(
    *,
    factor: CapabilityFactor,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    run_directory: str | Path,
    seed: int = 0,
    threads: int = 1,
    progress: Any = print,
) -> dict[str, Any]:
    """Run all twelve public factor-conditioned N/K planning slices."""

    if factor not in SUPPORTED_CAPABILITY_FACTORS:
        raise ValueError(f"factor must be one of {SUPPORTED_CAPABILITY_FACTORS}")
    _seed_process(seed, threads)
    output = Path(run_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    config = load_config(model_config_path)
    model, checkpoint = _model_from_workbench_checkpoint(config, checkpoint_path)
    rows = factor_planning_rows(factor)
    capability_rows = tuple(row for row, _ in rows)
    complete_manifest = capability_manifest(
        "compositional_holdout" if factor == COMPOSITIONAL_FACTOR else "development"
    )
    provenance = {
        **checkpoint,
        "public_development_only": True,
        "planning_used_as_training_loss": False,
        "truth_runtime_input_count": 0,
        "private_oracle_opened_after_public_decision": True,
        "factor_controls_generator_only": True,
        "capability_manifest_sha256": manifest_sha256(complete_manifest),
        "factor_planning_rows_sha256": manifest_sha256(capability_rows),
    }

    def update(message: str) -> None:
        if progress is not None:
            progress(message)

    stream_state = {"completed_tasks": 0}

    def materializations():
        for index, (capability_row, planning_row) in enumerate(rows, start=1):
            item = materialize_capability_planning_task(
                planning_row,
                _environment(factor, capability_row.controls),
            )
            update(f"{factor} planning: materialized {index}/{len(rows)}")
            stream_state["completed_tasks"] = index
            yield item

    try:
        result = evaluate_development_planning_materializations(
            model,
            materializations(),
            config=PlanningPopulationEvaluationConfig(
                require_complete_slices=True,
                evaluate_invariants=True,
                latency_warmup_runs=1,
                latency_measured_runs=3,
            ),
        )
    except (
        PlanningTaskMaterializationError,
        PlanningTaskUnresolvedError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report = {
            "schema": CAPABILITY_PLANNING_REPORT_SCHEMA,
            "created_at_utc": created,
            "factor": factor,
            "status": "failed",
            "gate_failures": ["planning_population_completion"],
            "runtime_error": f"{type(error).__name__}: {error}",
            "completed_tasks": stream_state["completed_tasks"],
            "provenance": provenance,
            "timing_seconds": {"total": time.perf_counter() - started},
            "artifact_policy": {
                "generated_episodes_retained": False,
                "raw_frame_directories_retained": False,
                "checkpoint_copied": False,
            },
        }
        atomic_write_text(
            output / "planning_report.json",
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        archive_root = output.parent.parent / ".archive"
        archive_bytes = inventory_runs(output.parent, archive_root=archive_root)["archive_bytes"]
        summary = _failure_summary(
            run_id=output.name,
            factor=factor,
            error=error,
            completed_tasks=stream_state["completed_tasks"],
            provenance=provenance,
            created_at_utc=created,
            artifact_bytes=(output / "planning_report.json").stat().st_size,
            archive_bytes=archive_bytes,
        )
        write_capability_summary(summary, output / "capability_summary.json")
        write_run_report(summary, output)
        write_run_manifest(
            output,
            role="candidate",
            status="failed",
            artifacts={
                "capability_summary.json": "summary",
                "planning_report.json": "summary",
                "report.html": "report",
            },
        )
        cleanup = enforce_run_budget(output.parent, archive_root=archive_root)
        build_progress_dashboard(output.parent, archive_root=archive_root)
        update(f"stopped {factor} planning fail-closed: {error}")
        return {**report, "cleanup": cleanup.to_dict(), "run_directory": str(output)}
    gate_failures = _planning_gate_failures(
        result,
        compositional=factor == COMPOSITIONAL_FACTOR,
    )
    report = {
        "schema": CAPABILITY_PLANNING_REPORT_SCHEMA,
        "created_at_utc": created,
        "factor": factor,
        "status": "passed" if not gate_failures else "failed",
        "gate_failures": list(gate_failures),
        "planning_error": result.planning_error,
        "reduction": _jsonable(result.reduction),
        "invariants": _jsonable(result.invariants),
        "outcomes": _jsonable(result.outcomes),
        "provenance": provenance,
        "timing_seconds": {"total": time.perf_counter() - started},
        "artifact_policy": {
            "generated_episodes_retained": False,
            "raw_frame_directories_retained": False,
            "checkpoint_copied": False,
        },
    }
    atomic_write_text(
        output / "planning_report.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_root = output.parent.parent / ".archive"
    archive_bytes = inventory_runs(output.parent, archive_root=archive_root)["archive_bytes"]
    summary = _summary(
        run_id=output.name,
        factor=factor,
        result=result,
        gate_failures=gate_failures,
        provenance=provenance,
        created_at_utc=created,
        artifact_bytes=(output / "planning_report.json").stat().st_size,
        archive_bytes=archive_bytes,
    )
    write_capability_summary(summary, output / "capability_summary.json")
    write_run_report(summary, output)
    artifact_bytes = sum(
        path.stat().st_size for path in output.iterdir() if path.is_file() and not path.is_symlink()
    )
    summary = replace(
        summary,
        artifacts={"run_bytes": artifact_bytes, "archive_bytes": archive_bytes},
    )
    write_capability_summary(summary, output / "capability_summary.json")
    write_run_report(summary, output)
    write_run_manifest(
        output,
        role="candidate",
        status="completed",
        artifacts={
            "capability_summary.json": "summary",
            "planning_report.json": "summary",
            "report.html": "report",
        },
    )
    cleanup = enforce_run_budget(output.parent, archive_root=archive_root)
    build_progress_dashboard(output.parent, archive_root=archive_root)
    update(f"completed {factor} planning with {len(gate_failures)} gate failure(s)")
    return {**report, "cleanup": cleanup.to_dict(), "run_directory": str(output)}


def default_planning_run_directory(factor: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-capability-{factor.replace('_', '-')}-planning"


__all__ = [
    "CAPABILITY_PLANNING_REPORT_SCHEMA",
    "default_planning_run_directory",
    "factor_planning_rows",
    "run_incumbent_factor_planning",
]
