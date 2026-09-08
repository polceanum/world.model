"""Versioned portable summaries for capability runs and historical adapters."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from world_model.evaluation.general_capability import ALL_CAPABILITY_FACTORS
from world_model.utils.io import atomic_write_text

CAPABILITY_SUMMARY_SCHEMA = "world_model_capability_run_summary_v1"
WORKBENCH_SCHEMA = "world_model_capability_workbench_v1"


def _validate_json(value: object, path: str = "summary") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite value")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _supported_value(value: object) -> float | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("value")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
        return float(raw)
    return None


def _horizon_curve(cells: Mapping[str, object]) -> dict[str, float]:
    totals: dict[str, float] = {}
    supports: dict[str, int] = {}
    for metrics in cells.values():
        if not isinstance(metrics, Mapping):
            continue
        horizons = metrics.get("horizon_position_rmse_m")
        if not isinstance(horizons, Mapping):
            continue
        for horizon, evidence in horizons.items():
            value = _supported_value(evidence)
            support = evidence.get("support") if isinstance(evidence, Mapping) else None
            if value is None or not isinstance(support, int) or support <= 0:
                continue
            totals[str(horizon)] = math.fsum((totals.get(str(horizon), 0.0), value * support))
            supports[str(horizon)] = supports.get(str(horizon), 0) + support
    return {horizon: totals[horizon] / supports[horizon] for horizon in sorted(totals, key=float)}


@dataclass(frozen=True)
class CapabilityRunSummary:
    """Small complete evidence record used by both HTML views."""

    schema: str
    run_id: str
    created_at_utc: str
    lifecycle_status: Literal["active", "completed", "failed"]
    outcome: str
    source_format: str
    configuration: dict[str, Any]
    provenance: dict[str, Any]
    scores: dict[str, Any]
    factor_metrics: dict[str, Any]
    cell_metrics: dict[str, Any]
    horizon_curves: dict[str, Any]
    uncertainty: dict[str, Any]
    planning: dict[str, Any]
    resources: dict[str, Any]
    artifacts: dict[str, Any]
    selection: dict[str, Any]
    failure_attribution: dict[str, Any]
    qualitative: dict[str, Any]
    unsupported_claims: tuple[str, ...]
    scope_limitations: tuple[str, ...]

    def validate(self) -> CapabilityRunSummary:
        if self.schema != CAPABILITY_SUMMARY_SCHEMA:
            raise ValueError("unsupported capability summary schema")
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be one safe path component")
        if self.lifecycle_status not in {"active", "completed", "failed"}:
            raise ValueError("invalid capability lifecycle status")
        _validate_json(asdict(self))
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityRunSummary:
        if value.get("schema") != CAPABILITY_SUMMARY_SCHEMA:
            raise ValueError("unsupported capability summary schema")
        payload = dict(value)
        payload["unsupported_claims"] = tuple(payload.get("unsupported_claims", ()))
        payload["scope_limitations"] = tuple(payload.get("scope_limitations", ()))
        return cls(**payload).validate()


def summary_from_workbench_report(
    report: Mapping[str, Any],
    *,
    run_id: str,
    artifacts: Mapping[str, Any] | None = None,
) -> CapabilityRunSummary:
    """Adapt a current workbench report without mutating its evidence."""

    if report.get("schema") != WORKBENCH_SCHEMA:
        raise ValueError("report is not a capability workbench report")
    candidate = report.get("candidate")
    reference = report.get("reference")
    if not isinstance(candidate, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("workbench report lacks candidate/reference evidence")
    candidate_physical = candidate.get("physical")
    candidate_planning = candidate.get("planning")
    if not isinstance(candidate_physical, Mapping) or not isinstance(candidate_planning, Mapping):
        raise ValueError("workbench report lacks physical/planning evidence")
    cells = candidate_physical.get("cells", {})
    if not isinstance(cells, Mapping):
        raise ValueError("workbench cell metrics must be a mapping")
    reduction = candidate_planning.get("reduction", {})
    if not isinstance(reduction, Mapping):
        reduction = {}
    diagnosis = report.get("diagnosis", {})
    if not isinstance(diagnosis, Mapping):
        diagnosis = {}
    resources = candidate_physical.get("resources", {})
    if not isinstance(resources, Mapping):
        resources = {}
    capacity = report.get("capacity", {})
    timing = report.get("timing_seconds", {})
    score = candidate.get("capability_score", {})
    reference_score = reference.get("capability_score", {})
    coverage_range = diagnosis.get("uncertainty_90_coverage_range", ())
    factor_metrics = {
        "nominal_structured": {
            "status": "measured",
            "episode_count": candidate_physical.get("episode_count", 0),
            "score": _supported_value(score),
        }
    }
    factor_metrics.update(
        {
            factor: {"status": "unmeasured", "reason": "broad factor run not executed"}
            for factor in ALL_CAPABILITY_FACTORS
        }
    )
    selected = report.get("selection", {})
    selected_name = selected.get("selected") if isinstance(selected, Mapping) else None
    created = report.get("created_at_utc")
    if not isinstance(created, str):
        created = datetime.now(timezone.utc).isoformat()
    worst = diagnosis.get("worst_current_position_rmse_m")
    summary = CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=created,
        lifecycle_status="completed",
        outcome=str(report.get("status", "unknown")),
        source_format=WORKBENCH_SCHEMA,
        configuration={
            "model_config": report.get("model_config"),
            "settings": report.get("settings", {}),
            "workload": report.get("workload", {}),
        },
        provenance={
            "public_development_only": True,
            "adapter": "read_only",
            "planning_used_as_training_loss": False,
        },
        scores={
            "candidate": score,
            "incumbent": reference_score,
            "selected": selected_name,
        },
        factor_metrics=factor_metrics,
        cell_metrics=dict(cells),
        horizon_curves={"candidate_position_rmse_m": _horizon_curve(cells)},
        uncertainty={"coverage_90_range": coverage_range},
        planning={
            "task_count": candidate_planning.get("task_count", 0),
            "slices": reduction.get("slices", []),
            "gate_metrics": reduction.get("gate_metrics", {}),
            "serial_vectorized_winner_parity": reduction.get("serial_vectorized_winner_parity"),
            "maximum_cost_difference": reduction.get("maximum_cost_difference"),
        },
        resources={
            **dict(resources),
            "capacity": capacity,
            "timing_seconds": timing,
            "scalability": report.get("scalability", {}),
        },
        artifacts=dict(artifacts or {}),
        selection=dict(selected) if isinstance(selected, Mapping) else {},
        failure_attribution={
            "primary_bottleneck": diagnosis.get("primary_bottleneck", "unavailable"),
            "ablation_owner": "unmeasured",
        },
        qualitative={
            "best_episode": "not retained",
            "worst_episode": worst[0]
            if isinstance(worst, list | tuple) and worst
            else "unavailable",
            "representative_episode": "not retained",
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "broadened single-factor capability",
            "compositional holdout capability",
            "full perception above six objects",
        ),
        scope_limitations=(
            "hidden actions",
            "unknown camera calibration",
            "new modalities",
            "deformable or articulated bodies",
            "long-term scene memory",
        ),
    )
    if isinstance(coverage_range, (tuple, list)) and len(coverage_range) not in {0, 2}:
        raise ValueError("uncertainty coverage range must be empty or a pair")
    return summary.validate()


def historical_summary(
    report: Mapping[str, Any],
    *,
    run_id: str,
    source_format: str,
) -> CapabilityRunSummary:
    """Return a minimal read-only adapter for older report schemas."""

    status = report.get("status", report.get("outcome", "historical"))
    created = report.get("created_at_utc", report.get("created_at"))
    if not isinstance(created, str):
        created = "unknown"
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=created,
        lifecycle_status="completed",
        outcome=str(status),
        source_format=source_format,
        configuration={},
        provenance={"adapter": "read_only", "historical": True},
        scores={},
        factor_metrics={
            factor: {"status": "unmeasured", "reason": "historical schema"}
            for factor in ALL_CAPABILITY_FACTORS
        },
        cell_metrics={},
        horizon_curves={},
        uncertainty={},
        planning={},
        resources={},
        artifacts={},
        selection={},
        failure_attribution={"primary_bottleneck": "not available in historical schema"},
        qualitative={},
        unsupported_claims=("general capability",),
        scope_limitations=("historical evidence shown without reinterpretation",),
    ).validate()


def write_capability_summary(
    summary: CapabilityRunSummary,
    path: str | Path,
) -> Path:
    """Atomically persist one canonical summary JSON document."""

    target = Path(path)
    atomic_write_text(
        target,
        json.dumps(summary.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return target


def read_capability_summary(path: str | Path) -> CapabilityRunSummary:
    """Read and validate one summary without accepting partial schemas."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("capability summary root must be a mapping")
    return CapabilityRunSummary.from_dict(raw)


__all__ = [
    "CAPABILITY_SUMMARY_SCHEMA",
    "CapabilityRunSummary",
    "historical_summary",
    "read_capability_summary",
    "summary_from_workbench_report",
    "write_capability_summary",
]
