"""Small, repeatable development loop for the dynamic RGB-D world model.

This module intentionally composes the existing model, simulator, objective,
physical evaluator, and required planning evaluator without the one-shot
qualification coordinator.  It is a development tool: results are honest
seeded measurements, never protected-split or promotion claims.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from world_model.evaluation.capability_summary import (
    summary_from_workbench_report,
    write_capability_summary,
)
from world_model.evaluation.scalability import compare_dense_and_packed_scalability
from world_model.runtime import OnlineWorldModel
from world_model.training.dynamic_set_adapter import DynamicSetEpisodeObjectiveAdapter
from world_model.training.dynamic_set_config import OrpheusConfig, load_config
from world_model.training.dynamic_set_evaluation import (
    DynamicSetEvaluationResult,
    DynamicSetPairedEvaluationResult,
    evaluate_paired_dynamic_set_materializations,
)
from world_model.training.dynamic_set_gates import (
    SupportedScalar,
    physical_cell_gate_failures,
    planning_gate_failures,
)
from world_model.training.dynamic_set_materializer import materialize_dynamic_set_episode
from world_model.training.dynamic_set_planning_materializer import (
    PlanningPairedPopulationEvaluationResult,
    PlanningPopulationEvaluationConfig,
    PlanningPopulationEvaluationResult,
    PlanningTaskMaterialization,
    PlanningTaskMaterializationError,
    evaluate_paired_development_planning_materializations,
    materialize_planning_task,
)
from world_model.training.dynamic_set_protocol import (
    PHYSICAL_CELL_COUNT,
    SELECTION_SCORE_WEIGHTS,
    PhysicalCell,
    PhysicalManifestRow,
    PlanningManifestRow,
    physical_manifest,
    planning_manifest,
)
from world_model.training.dynamic_set_trainer import (
    SCREEN_ROWS,
    DynamicSetTrainer,
    DynamicSetUpdateReport,
)
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import (
    enforce_run_budget,
    inventory_runs,
    write_run_manifest,
)
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

WORKBENCH_SCHEMA = "world_model_capability_workbench_v1"
DEFAULT_CAPABILITY_VELOCITY_VARIANCE_FLOOR = 4.0e-13
DEFAULT_DEVELOPMENT_RETENTION_IMPROVEMENT = 0.03
ProgressHook = Callable[[str], None]


def _checkpoint_config_semantics(value: object) -> object:
    """Fill only parameter-free defaults absent from older workbench runs."""

    if not isinstance(value, Mapping):
        return value
    normalized = deepcopy(dict(value))
    model = normalized.get("model")
    if isinstance(model, dict):
        rgbd = model.get("rgbd")
        if isinstance(rgbd, dict):
            rgbd.setdefault("birth_proposals", 2)
        dynamics = model.get("dynamics")
        if isinstance(dynamics, dict):
            dynamics.setdefault("packed_interactions_enabled", False)
    return normalized


@dataclass(frozen=True)
class CapabilityWorkbenchConfig:
    """Bounded public-development workload.

    ``physical_cycles`` contributes one example from every one of the 22
    count/contact/lifecycle cells per cycle. ``planning_repeats`` contributes
    one example per ``(object count, candidate count)`` slice per repeat.
    """

    profile: Literal["smoke", "development"] = "smoke"
    seed: int = 0
    train_updates: int = 1
    physical_cycles: int = 1
    physical_cycle_offset: int = 0
    planning_repeats: int = 1
    threads: int = 1
    planning_latency_runs: int = 1
    planning_invariants: bool = False
    velocity_variance_floor: float = DEFAULT_CAPABILITY_VELOCITY_VARIANCE_FLOOR

    @classmethod
    def for_profile(
        cls,
        profile: Literal["smoke", "development"],
        *,
        seed: int = 0,
        train_updates: int | None = None,
        physical_cycles: int | None = None,
        physical_cycle_offset: int | None = None,
        planning_repeats: int | None = None,
        threads: int = 1,
        planning_invariants: bool | None = None,
        velocity_variance_floor: float = DEFAULT_CAPABILITY_VELOCITY_VARIANCE_FLOOR,
    ) -> CapabilityWorkbenchConfig:
        defaults = {
            "smoke": (1, 1, 0, 1, 1),
            # Cycle zero is the public calibration slice. Development reports
            # start at cycle one so the fitted variance floor is evaluated on
            # different scenes.
            "development": (32, 3, 1, 2, 3),
        }
        if profile not in defaults:
            raise ValueError("profile must be 'smoke' or 'development'")
        updates, cycles, cycle_offset, repeats, latency_runs = defaults[profile]
        return cls(
            profile=profile,
            seed=seed,
            train_updates=updates if train_updates is None else train_updates,
            physical_cycles=cycles if physical_cycles is None else physical_cycles,
            physical_cycle_offset=(
                cycle_offset if physical_cycle_offset is None else physical_cycle_offset
            ),
            planning_repeats=repeats if planning_repeats is None else planning_repeats,
            threads=threads,
            planning_latency_runs=latency_runs,
            planning_invariants=(
                profile == "development" if planning_invariants is None else planning_invariants
            ),
            velocity_variance_floor=velocity_variance_floor,
        ).validate()

    def validate(self) -> CapabilityWorkbenchConfig:
        if self.profile not in {"smoke", "development"}:
            raise ValueError("profile must be 'smoke' or 'development'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            isinstance(self.physical_cycle_offset, bool)
            or not isinstance(self.physical_cycle_offset, int)
            or self.physical_cycle_offset < 0
        ):
            raise ValueError("physical_cycle_offset must be a nonnegative integer")
        for name in (
            "physical_cycles",
            "planning_repeats",
            "threads",
            "planning_latency_runs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.train_updates, bool)
            or not isinstance(self.train_updates, int)
            or not 0 <= self.train_updates <= 512
        ):
            raise ValueError("train_updates must lie in [0,512]")
        if type(self.planning_invariants) is not bool:
            raise TypeError("planning_invariants must be boolean")
        if (
            isinstance(self.velocity_variance_floor, bool)
            or not isinstance(self.velocity_variance_floor, (int, float))
            or not math.isfinite(float(self.velocity_variance_floor))
            or not 0.0 < float(self.velocity_variance_floor) <= 1.0
        ):
            raise ValueError("velocity_variance_floor must be finite and lie in (0,1]")
        return self


def select_physical_development_rows(
    cycles: int,
    *,
    cycle_offset: int = 0,
) -> tuple[PhysicalManifestRow, ...]:
    """Return complete deterministic 22-cell cycles from the public split."""

    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("physical cycles must be a positive integer")
    if isinstance(cycle_offset, bool) or not isinstance(cycle_offset, int) or cycle_offset < 0:
        raise ValueError("physical cycle offset must be a nonnegative integer")
    start = cycle_offset * PHYSICAL_CELL_COUNT
    stop = start + cycles * PHYSICAL_CELL_COUNT
    rows = physical_manifest("development")[start:stop]
    if len(rows) != cycles * PHYSICAL_CELL_COUNT:
        raise ValueError("requested physical workload exceeds the development manifest")
    counts = Counter(row.cell_index for row in rows)
    if set(counts) != set(range(PHYSICAL_CELL_COUNT)) or set(counts.values()) != {cycles}:
        raise RuntimeError("physical workbench rows are not balanced across all cells")
    return rows


def select_planning_development_rows(repeats: int) -> tuple[PlanningManifestRow, ...]:
    """Select balanced N=1..6 and K=8/32 public planning rows."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("planning repeats must be a positive integer")
    source = planning_manifest("development")
    selected: list[PlanningManifestRow] = []
    for object_count in range(1, 7):
        for candidate_count in (8, 32):
            matches = tuple(
                row
                for row in source
                if row.object_count == object_count and row.candidate_count == candidate_count
            )
            if len(matches) < repeats:
                raise ValueError("requested planning workload exceeds one public N/K slice")
            selected.extend(matches[:repeats])
    return tuple(sorted(selected, key=lambda row: (row.ordinal, row.seed)))


def materialize_planning_development_workload(
    repeats: int,
) -> tuple[tuple[PlanningTaskMaterialization, ...], tuple[dict[str, Any], ...]]:
    """Preflight balanced planning slices and retain row-level failures.

    A deterministic development row may fail its independent scene/oracle
    preflight. The workbench continues with the next row from the same N/K
    slice and records the substitution instead of aborting the whole loop.
    """

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("planning repeats must be a positive integer")
    source = planning_manifest("development")
    accepted: list[PlanningTaskMaterialization] = []
    rejected: list[dict[str, Any]] = []
    for object_count in range(1, 7):
        for candidate_count in (8, 32):
            slice_count = 0
            for row in source:
                if row.object_count != object_count or row.candidate_count != candidate_count:
                    continue
                try:
                    materialization = materialize_planning_task(row)
                except PlanningTaskMaterializationError as error:
                    rejected.append(
                        {
                            "ordinal": row.ordinal,
                            "seed": row.seed,
                            "object_count": object_count,
                            "candidate_count": candidate_count,
                            "reason": str(error),
                        }
                    )
                    continue
                accepted.append(materialization)
                slice_count += 1
                if slice_count == repeats:
                    break
            if slice_count != repeats:
                raise PlanningTaskMaterializationError(
                    f"development planning slice N={object_count}, K={candidate_count} "
                    f"provided only {slice_count}/{repeats} valid tasks"
                )
    accepted.sort(key=lambda item: (item.row.ordinal, item.row.seed))
    return tuple(accepted), tuple(rejected)


def workbench_plan(config: CapabilityWorkbenchConfig) -> dict[str, Any]:
    """Describe the exact work without constructing a model or materializing data."""

    config.validate()
    physical_rows = select_physical_development_rows(
        config.physical_cycles,
        cycle_offset=config.physical_cycle_offset,
    )
    planning_rows = select_planning_development_rows(config.planning_repeats)
    return {
        "profile": config.profile,
        "seed": config.seed,
        "train_updates": config.train_updates,
        "training_examples": config.train_updates * 24,
        "physical_examples": len(physical_rows),
        "physical_cells": len({row.cell_index for row in physical_rows}),
        "physical_cycle_offset": config.physical_cycle_offset,
        "planning_tasks": len(planning_rows),
        "planning_slices": len({(row.object_count, row.candidate_count) for row in planning_rows}),
        "planning_invariants": config.planning_invariants,
        "velocity_variance_floor": config.velocity_variance_floor,
        "threads": config.threads,
    }


def _seed_process(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def _new_model(
    config: OrpheusConfig,
    seed: int,
    *,
    velocity_variance_floor: float | None = None,
) -> OnlineWorldModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = OnlineWorldModel.from_config(config, device="cpu")
    if velocity_variance_floor is not None:
        module = model.observation_modules["rgbd"]
        rgbd_config = getattr(module, "config", None)
        if rgbd_config is None or getattr(rgbd_config, "observation_mode", None) != "set":
            raise ValueError("velocity calibration requires the RGB-D set observer")
        module.config = replace(
            rgbd_config,
            temporal_velocity_variance_floor=max(
                float(rgbd_config.temporal_velocity_variance_floor),
                float(velocity_variance_floor),
            ),
        )
    return model


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize workbench value of type {type(value).__name__}")


def _cell_name(row: PhysicalCell) -> str:
    return f"N{row.object_count}/contact={int(row.contact)}/dynamic={int(row.dynamic_membership)}"


def _physical_payload(result: DynamicSetEvaluationResult) -> dict[str, Any]:
    return {
        "episode_count": result.episode_count,
        "selection_components": dict(result.score.components),
        "resources": _jsonable(result.resources),
        "cells": {
            _cell_name(cell): _jsonable(metrics)
            for cell, metrics in sorted(
                result.by_cell.items(),
                key=lambda item: (
                    item[0].object_count,
                    item[0].contact,
                    item[0].dynamic_membership,
                ),
            )
        },
    }


def _planning_payload(
    result: PlanningPopulationEvaluationResult,
    *,
    invariants_evaluated: bool,
) -> dict[str, Any]:
    outcomes = result.outcomes
    failures = Counter(outcome.failure_reason for outcome in outcomes if outcome.failure_reason)
    return {
        "task_count": len(outcomes),
        "planning_error": float(result.planning_error),
        "full_invariants_evaluated": invariants_evaluated,
        "reduction": _jsonable(result.reduction),
        "invariants": _jsonable(result.invariants) if invariants_evaluated else None,
        "unresolved_count": sum(failures.values()),
        "failure_reasons": dict(sorted(failures.items())),
    }


def supported_capability_score(
    physical_components: Mapping[str, float],
    planning_error: float,
) -> tuple[float, float]:
    """Return a transparent score over only metrics supported by this run.

    The second value is the fraction of the fixed score weight represented by
    the available physical evidence plus required planning evidence.  A smoke
    run can therefore remain useful without pretending to be a full gate.
    """

    values = {**physical_components, "planning_error": float(planning_error)}
    supported = {
        name: float(value)
        for name, value in values.items()
        if name in SELECTION_SCORE_WEIGHTS and math.isfinite(float(value))
    }
    weight = sum(SELECTION_SCORE_WEIGHTS[name] for name in supported)
    if weight <= 0.0:
        raise ValueError("capability score has no supported components")
    score = sum(SELECTION_SCORE_WEIGHTS[name] * value for name, value in supported.items()) / weight
    return score, weight


def _score_payload(
    physical: DynamicSetEvaluationResult,
    planning: PlanningPopulationEvaluationResult,
) -> dict[str, float]:
    score, weight = supported_capability_score(
        physical.score.components,
        float(planning.planning_error),
    )
    return {"value": score, "supported_weight": weight}


def capability_change_status(
    candidate_score: float,
    reference_score: float,
    *,
    minimum_relative_change: float = 1.0e-3,
) -> str:
    """Classify only changes large enough to matter at development scale."""

    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in (candidate_score, reference_score, minimum_relative_change)
    ):
        raise ValueError("capability change inputs must be finite real scalars")
    if reference_score < 0.0 or candidate_score < 0.0:
        raise ValueError("capability scores must be nonnegative")
    if not 0.0 < minimum_relative_change < 1.0:
        raise ValueError("minimum_relative_change must lie in (0,1)")
    threshold = max(1.0e-9, abs(reference_score) * minimum_relative_change)
    delta = candidate_score - reference_score
    return "improved" if delta < -threshold else "regressed" if delta > threshold else "unchanged"


def select_development_incumbent(
    candidate_score: float,
    reference_score: float,
    *,
    physical_failures: Sequence[str] = (),
    planning_failures: Sequence[str] = (),
    minimum_relative_improvement: float = DEFAULT_DEVELOPMENT_RETENTION_IMPROVEMENT,
) -> dict[str, Any]:
    """Keep learned weights only after a material, gate-clean public gain.

    This is deliberately a development safeguard, not a promotion claim.  It
    prevents a training run from replacing a stronger structured incumbent
    merely because optimization completed or produced a numerically tiny win.
    """

    values = (candidate_score, reference_score, minimum_relative_improvement)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("development selection inputs must be finite real scalars")
    if candidate_score < 0.0 or reference_score < 0.0:
        raise ValueError("capability scores must be nonnegative")
    if not 0.0 < minimum_relative_improvement < 1.0:
        raise ValueError("minimum_relative_improvement must lie in (0,1)")
    physical = tuple(str(item) for item in physical_failures)
    planning = tuple(str(item) for item in planning_failures)
    denominator = max(float(reference_score), 1.0e-12)
    improvement = (float(reference_score) - float(candidate_score)) / denominator
    retained = improvement >= float(minimum_relative_improvement) and not physical and not planning
    reasons: list[str] = []
    if improvement < float(minimum_relative_improvement):
        reasons.append(
            f"relative score improvement {improvement:.6f} is below "
            f"{float(minimum_relative_improvement):.6f}"
        )
    if physical:
        reasons.append(f"{len(physical)} sampled physical gate failure(s)")
    if planning:
        reasons.append(f"{len(planning)} required planning gate failure(s)")
    return {
        "selected": "trained_candidate" if retained else "calibrated_structured_baseline",
        "learned_weights_retained": retained,
        "relative_score_improvement": improvement,
        "minimum_relative_improvement": float(minimum_relative_improvement),
        "reasons": reasons,
    }


def _sampled_gate_failures(
    physical: DynamicSetEvaluationResult,
    planning: PlanningPopulationEvaluationResult,
    *,
    invariants_evaluated: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    physical_failures = tuple(
        f"{_cell_name(cell)}/{failure}"
        for cell, metrics in sorted(
            physical.by_cell.items(),
            key=lambda item: (
                item[0].object_count,
                item[0].contact,
                item[0].dynamic_membership,
            ),
        )
        for failure in physical_cell_gate_failures(cell, metrics)
    )
    if not invariants_evaluated:
        return physical_failures, ("required_planning_invariants:not_evaluated",)
    return physical_failures, planning_gate_failures(
        planning.reduction.gate_metrics,
        planning.invariants,
        expected_distributions=("in_distribution",),
    )


def _training_payload(reports: Sequence[DynamicSetUpdateReport]) -> dict[str, Any]:
    if not reports:
        return {
            "completed_updates": 0,
            "materialized_examples": 0,
            "first_objective": None,
            "last_objective": None,
            "minimum_objective": None,
            "median_update_seconds": None,
        }
    return {
        "completed_updates": reports[-1].completed_updates,
        "materialized_examples": sum(report.materialized_examples for report in reports),
        "first_objective": reports[0].objective,
        "last_objective": reports[-1].objective,
        "minimum_objective": min(report.objective for report in reports),
        "minimum_gradient_retention": min(report.complete_gradient_retention for report in reports),
        "maximum_perception_gradient_norm": max(
            report.perception_gradient_norm for report in reports
        ),
        "maximum_relation_gradient_norm": max(report.relation_gradient_norm for report in reports),
    }


def _supported_value(value: SupportedScalar | None) -> float | None:
    if value is None or value.support <= 0:
        return None
    return float(value.value)


def _diagnose(
    candidate_physical: DynamicSetEvaluationResult,
    candidate_planning: PlanningPopulationEvaluationResult,
    *,
    physical_cycles: int,
) -> dict[str, Any]:
    cells = tuple(candidate_physical.by_cell.items())
    proposal_values = [
        (_cell_name(cell), _supported_value(metrics.proposal_f1)) for cell, metrics in cells
    ]
    position_values = [
        (_cell_name(cell), _supported_value(metrics.current_position_rmse_m))
        for cell, metrics in cells
    ]
    coverage_values = [
        (_cell_name(cell), _supported_value(metrics.uncertainty_90_coverage))
        for cell, metrics in cells
    ]
    proposal_values = [(name, value) for name, value in proposal_values if value is not None]
    position_values = [(name, value) for name, value in position_values if value is not None]
    coverage_values = [(name, value) for name, value in coverage_values if value is not None]
    unresolved = sum(not outcome.handle_resolved for outcome in candidate_planning.outcomes)
    worst_proposal = min(proposal_values, key=lambda item: item[1]) if proposal_values else None
    worst_position = max(position_values, key=lambda item: item[1]) if position_values else None
    worst_coverage = (
        max(coverage_values, key=lambda item: abs(item[1] - 0.90)) if coverage_values else None
    )
    if unresolved:
        bottleneck = "observable target resolution"
    elif worst_proposal is not None and worst_proposal[1] < 0.98:
        bottleneck = "set perception"
    elif worst_position is not None and worst_position[1] > 0.010:
        bottleneck = "metric state estimation"
    elif worst_coverage is not None and not 0.85 <= worst_coverage[1] <= 0.95:
        bottleneck = "uncertainty calibration"
    elif float(candidate_planning.planning_error) > 0.0:
        bottleneck = "action-conditioned rollout or action selection"
    else:
        bottleneck = "no dominant failure in the sampled workload"
    return {
        "primary_bottleneck": bottleneck,
        "worst_proposal_f1": worst_proposal,
        "worst_current_position_rmse_m": worst_position,
        "worst_uncertainty_90_coverage": worst_coverage,
        "uncertainty_90_coverage_range": (
            [min(value for _, value in coverage_values), max(value for _, value in coverage_values)]
            if coverage_values
            else None
        ),
        "unresolved_planning_tasks": unresolved,
        "coverage_note": (
            "three physical cycles are required to exercise all lifecycle schedules"
            if physical_cycles < 3
            else "all lifecycle schedules are represented"
        ),
    }


def _atomic_torch_save(value: object, path: Path) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(buffer.getvalue())
    os.replace(temporary, path)


def _markdown(report: Mapping[str, Any]) -> str:
    candidate = report["candidate"]
    reference = report["reference"]
    candidate_score = candidate["capability_score"]
    reference_score = reference["capability_score"]
    lines = [
        "# World-model capability workbench",
        "",
        f"Status: **{report['status']}**",
        "",
        (
            f"The compact model trained for {report['training']['completed_updates']} updates "
            f"and was evaluated on {report['workload']['physical_examples']} physical episodes "
            f"and {report['workload']['planning_tasks']} required planning tasks."
        ),
        "",
        "| Measurement | Reference | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        (
            "| Supported capability score (lower is better) | "
            f"{reference_score['value']:.6f} | {candidate_score['value']:.6f} | "
            f"{candidate_score['value'] - reference_score['value']:+.6f} |"
        ),
        (
            "| Planning error (lower is better) | "
            f"{reference['planning']['planning_error']:.6f} | "
            f"{candidate['planning']['planning_error']:.6f} | "
            f"{candidate['planning']['planning_error'] - reference['planning']['planning_error']:+.6f} |"
        ),
        "",
        f"Supported score weight: {candidate_score['supported_weight']:.2f}/1.00.",
        "",
        f"Primary observed bottleneck: **{report['diagnosis']['primary_bottleneck']}**.",
        "",
        (
            "Retained model: "
            f"**{report['selection']['selected'].replace('_', ' ')}** "
            "(learned weights must clear both the material-improvement threshold "
            "and sampled behavior gates)."
        ),
        "",
        (
            "Observed 90% uncertainty coverage range: "
            f"{report['diagnosis']['uncertainty_90_coverage_range'][0]:.3f}--"
            f"{report['diagnosis']['uncertainty_90_coverage_range'][1]:.3f}."
        ),
        "",
        "## Efficiency",
        "",
        f"- Parameters: {report['capacity']['parameters']:,}",
        f"- Learned weight bytes: {report['capacity']['weight_bytes']:,}",
        f"- Total wall time: {report['timing_seconds']['total']:.3f} s",
        f"- Median optimizer update: {report['timing_seconds']['median_update']:.3f} s",
        "",
        "## Interpretation",
        "",
        f"- {report['diagnosis']['coverage_note']}.",
        "- Planning is evaluated downstream and does not contribute a training loss.",
        "- This public development result is not a protected-test or promotion claim.",
        "",
    ]
    return "\n".join(lines)


def run_capability_workbench(
    *,
    model_config_path: str | Path,
    run_directory: str | Path,
    workbench_config: CapabilityWorkbenchConfig,
    resume_checkpoint_path: str | Path | None = None,
    progress: ProgressHook | None = print,
) -> dict[str, Any]:
    """Train once and compare against the identical zero-residual initializer."""

    settings = workbench_config.validate()
    started = time.perf_counter()
    _seed_process(settings.seed, settings.threads)
    model_config = load_config(model_config_path)
    output = Path(run_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    def update(message: str) -> None:
        if progress is not None:
            progress(message)

    update("building calibrated candidate and calibrated structured reference models")
    candidate = _new_model(
        model_config,
        settings.seed,
        velocity_variance_floor=settings.velocity_variance_floor,
    )
    reference = _new_model(
        model_config,
        settings.seed,
        velocity_variance_floor=settings.velocity_variance_floor,
    )
    if any(
        not torch.equal(left, right)
        for left, right in zip(
            candidate.state_dict().values(), reference.state_dict().values(), strict=True
        )
    ):
        raise RuntimeError("candidate and reference initial states differ")
    resumed_checkpoint: Mapping[str, Any] | None = None
    resumed_from: str | None = None
    if resume_checkpoint_path is not None:
        if settings.train_updates != 0:
            raise ValueError("resumed workbench evaluation requires train_updates=0")
        resumed_path = Path(resume_checkpoint_path).expanduser().resolve()
        loaded = torch.load(resumed_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, Mapping) or loaded.get("schema") != WORKBENCH_SCHEMA:
            raise ValueError("resume checkpoint is not a capability-workbench checkpoint")
        if _checkpoint_config_semantics(loaded.get("model_config")) != (
            _checkpoint_config_semantics(model_config.to_dict())
        ):
            raise ValueError("resume checkpoint model configuration differs")
        saved_settings = loaded.get("workbench_settings")
        if not isinstance(saved_settings, Mapping):
            raise ValueError("resume checkpoint lacks workbench settings")
        if (
            saved_settings.get("seed") != settings.seed
            or saved_settings.get("velocity_variance_floor") != settings.velocity_variance_floor
        ):
            raise ValueError("resume checkpoint seed or calibration differs")
        model_state = loaded.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("resume checkpoint lacks model state")
        candidate.load_state_dict(model_state, strict=True)
        resumed_checkpoint = loaded
        resumed_from = str(resumed_path)
        update(f"loaded {int(loaded.get('completed_updates', 0))} updates from {resumed_path}")
    update_seconds: list[float] = []
    training_reports: list[DynamicSetUpdateReport] = []
    trainer: DynamicSetTrainer | None = None
    if settings.train_updates:
        update(f"training {settings.train_updates} balanced updates")
        trainer = DynamicSetTrainer.from_online_world_model(
            model=candidate,
            training_rows=SCREEN_ROWS,
            objective_adapter=DynamicSetEpisodeObjectiveAdapter(),
            resolved_config=model_config,
            source_provenance={
                "schema": WORKBENCH_SCHEMA,
                "mode": "public_development",
                "seed": settings.seed,
            },
            schedule_seed=settings.seed,
            screen_only=True,
        )
        for index in range(settings.train_updates):
            before = time.perf_counter()
            training_reports.append(trainer.run_update())
            update_seconds.append(time.perf_counter() - before)
            update(
                f"completed update {index + 1}/{settings.train_updates}: "
                f"objective={training_reports[-1].objective:.6f}"
            )

    # This function writes only terminal runs. Optimizer state is useful while
    # a campaign is resumable, but retaining it after successful completion is
    # unnecessary storage and conflicts with the compact artifact policy.
    trainer_checkpoint = None
    completed_updates = (
        int(resumed_checkpoint.get("completed_updates", 0))
        if resumed_checkpoint is not None
        else 0
        if trainer is None
        else trainer.completed_updates
    )
    checkpoint_payload: Mapping[str, Any] = {
        "schema": WORKBENCH_SCHEMA,
        "completed_updates": completed_updates,
        "model_state": candidate.state_dict(),
        "model_config": model_config.to_dict(),
        "workbench_settings": asdict(settings),
        "trainer_checkpoint": trainer_checkpoint,
    }
    _atomic_torch_save(checkpoint_payload, output / "checkpoint.pt")

    desired_planning_rows = select_planning_development_rows(settings.planning_repeats)
    update(f"preflighting {len(desired_planning_rows)} required public planning tasks")
    materialize_planning_started = time.perf_counter()
    planning_materializations, planning_row_rejections = materialize_planning_development_workload(
        settings.planning_repeats
    )
    planning_materialization_seconds = time.perf_counter() - materialize_planning_started
    if planning_row_rejections:
        update(f"replaced {len(planning_row_rejections)} planning rows that failed scene preflight")

    update("evaluating serial/vectorized counterfactual planning")
    planning_started = time.perf_counter()
    planning_pair: PlanningPairedPopulationEvaluationResult = (
        evaluate_paired_development_planning_materializations(
            candidate,
            reference,
            planning_materializations,
            config=PlanningPopulationEvaluationConfig(
                require_complete_slices=True,
                evaluate_invariants=settings.planning_invariants,
                latency_warmup_runs=int(settings.planning_invariants),
                latency_measured_runs=settings.planning_latency_runs,
            ),
        )
    )
    planning_seconds = time.perf_counter() - planning_started

    physical_rows = select_physical_development_rows(
        settings.physical_cycles,
        cycle_offset=settings.physical_cycle_offset,
    )
    update(f"streaming {len(physical_rows)} balanced public physical episodes")
    physical_materialization_seconds = 0.0

    def physical_materializations():
        nonlocal physical_materialization_seconds
        for row in physical_rows:
            row_started = time.perf_counter()
            materialization = materialize_dynamic_set_episode(row)
            physical_materialization_seconds += time.perf_counter() - row_started
            yield materialization

    update("evaluating physical behavior against the identical initializer")
    physical_started = time.perf_counter()
    physical_pair: DynamicSetPairedEvaluationResult = evaluate_paired_dynamic_set_materializations(
        candidate,
        reference,
        physical_materializations(),
        require_all_cells=False,
    )
    physical_total_seconds = time.perf_counter() - physical_started
    physical_seconds = max(0.0, physical_total_seconds - physical_materialization_seconds)

    candidate_score = _score_payload(physical_pair.candidate, planning_pair.candidate)
    reference_score = _score_payload(physical_pair.reference, planning_pair.reference)
    status = capability_change_status(candidate_score["value"], reference_score["value"])
    candidate_physical_failures, candidate_planning_failures = _sampled_gate_failures(
        physical_pair.candidate,
        planning_pair.candidate,
        invariants_evaluated=settings.planning_invariants,
    )
    reference_physical_failures, reference_planning_failures = _sampled_gate_failures(
        physical_pair.reference,
        planning_pair.reference,
        invariants_evaluated=settings.planning_invariants,
    )
    selection = select_development_incumbent(
        candidate_score["value"],
        reference_score["value"],
        physical_failures=candidate_physical_failures,
        planning_failures=candidate_planning_failures,
    )
    selected_model = candidate if selection["learned_weights_retained"] else reference
    selected_checkpoint_payload = {
        **checkpoint_payload,
        "completed_updates": completed_updates if selected_model is candidate else 0,
        "model_state": selected_model.state_dict(),
        "trainer_checkpoint": trainer_checkpoint if selected_model is candidate else None,
        "selection": selection,
        "candidate_checkpoint": "checkpoint.pt",
    }
    _atomic_torch_save(selected_checkpoint_payload, output / "selected_checkpoint.pt")
    capacity_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    capacity_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in candidate.parameters()
    )
    update("probing state-only N=8/12/16 scalability")
    scalability = compare_dense_and_packed_scalability(
        selected_model.dynamics,
        warmup_runs=1,
        measured_runs=3,
    )
    total_seconds = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema": WORKBENCH_SCHEMA,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": str(Path(model_config_path).expanduser().resolve()),
        "settings": asdict(settings),
        "workload": workbench_plan(settings),
        "planning_row_rejections": planning_row_rejections,
        "calibration": {
            "kind": "correlation-safe temporal velocity variance floor",
            "candidate_variance_floor": settings.velocity_variance_floor,
            "reference_variance_floor": settings.velocity_variance_floor,
            "fit_physical_cycle": 0,
            "evaluation_physical_cycle_offset": settings.physical_cycle_offset,
            "learned_parameter_change": False,
        },
        "selection": selection,
        "sampled_gate_evidence": {
            "candidate": {
                "physical_failures": candidate_physical_failures,
                "planning_failures": candidate_planning_failures,
            },
            "reference": {
                "physical_failures": reference_physical_failures,
                "planning_failures": reference_planning_failures,
            },
        },
        "capacity": {"parameters": capacity_parameters, "weight_bytes": capacity_bytes},
        "scalability": scalability,
        "training": {
            **_training_payload(training_reports),
            "completed_updates": completed_updates,
            "resumed_from": resumed_from,
            "optimizer_state_retained": False,
        },
        "candidate": {
            "capability_score": candidate_score,
            "physical": _physical_payload(physical_pair.candidate),
            "planning": _planning_payload(
                planning_pair.candidate,
                invariants_evaluated=settings.planning_invariants,
            ),
        },
        "reference": {
            "capability_score": reference_score,
            "physical": _physical_payload(physical_pair.reference),
            "planning": _planning_payload(
                planning_pair.reference,
                invariants_evaluated=settings.planning_invariants,
            ),
        },
        "diagnosis": _diagnose(
            physical_pair.candidate,
            planning_pair.candidate,
            physical_cycles=settings.physical_cycles,
        ),
        "timing_seconds": {
            "training": sum(update_seconds),
            "median_update": median(update_seconds) if update_seconds else 0.0,
            "physical_materialization": physical_materialization_seconds,
            "physical_evaluation": physical_seconds,
            "planning_materialization": planning_materialization_seconds,
            "planning_evaluation": planning_seconds,
            "total": total_seconds,
        },
        "claims": {
            "public_development_only": True,
            "planning_used_as_training_loss": False,
            "protected_or_promotion_claim": False,
        },
    }
    atomic_write_text(
        output / "report.json",
        json.dumps(_jsonable(report), allow_nan=False, indent=2) + "\n",
    )
    atomic_write_text(output / "report.md", _markdown(report))
    archive_root = output.parent.parent / ".archive"
    summary = summary_from_workbench_report(
        report,
        run_id=output.name,
        artifacts={
            "run_bytes": sum(
                path.stat().st_size
                for path in output.iterdir()
                if path.is_file() and not path.is_symlink()
            ),
            "archive_bytes": inventory_runs(
                output.parent,
                archive_root=archive_root,
            )["archive_bytes"],
        },
    )
    write_capability_summary(summary, output / "capability_summary.json")
    write_run_report(summary, output)
    write_run_manifest(
        output,
        role="candidate",
        status="completed",
        artifacts={
            "capability_summary.json": "summary",
            "report.json": "summary",
            "report.md": "report",
            "report.html": "report",
            "selected_checkpoint.pt": "checkpoint",
            "checkpoint.pt": (
                "checkpoint" if selection["learned_weights_retained"] else "rejected"
            ),
        },
    )
    cleanup = enforce_run_budget(output.parent, archive_root=archive_root)
    build_progress_dashboard(output.parent, archive_root=archive_root)
    if cleanup.warnings:
        update(
            f"artifact policy reported {len(cleanup.warnings)} protected/retention warnings; "
            "use world_model_progress.py list or prune for details"
        )
    update(f"completed with status={status}; report={output / 'report.md'}")
    return report


def default_run_directory(profile: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-capability-{profile}"


__all__ = [
    "CapabilityWorkbenchConfig",
    "DEFAULT_CAPABILITY_VELOCITY_VARIANCE_FLOOR",
    "DEFAULT_DEVELOPMENT_RETENTION_IMPROVEMENT",
    "WORKBENCH_SCHEMA",
    "capability_change_status",
    "default_run_directory",
    "materialize_planning_development_workload",
    "run_capability_workbench",
    "select_development_incumbent",
    "select_physical_development_rows",
    "select_planning_development_rows",
    "supported_capability_score",
    "workbench_plan",
]
