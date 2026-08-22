#!/usr/bin/env python3
"""Fail-closed learned-only versus runtime-hypothesis evaluator gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

from world_model.evaluation.evaluator import (
    _EVALUATION_METRIC_SCHEMA_VERSION,
    _EVALUATION_PROTOCOL_SCHEMA_VERSION,
    _PER_SCENARIO_METRIC_SCHEMA,
    _primary_physical_metrics,
    _primary_physical_metrics_hash_exclusion_declaration,
)
from world_model.evaluation.latency import paired_latency_guardrail
from world_model.evaluation.seed_protocol import (
    EVALUATION_SEED_PROTOCOLS,
    STANDARD_SEED_PROTOCOL,
    EvaluationSeedProtocol,
    make_evaluation_seed_protocol,
)
from world_model.simulator.sphere_world import SphereWorldConfig
from world_model.training.checkpointing import capture_git_metadata
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.io import atomic_write_text
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION

_CANDIDATES = (
    "learned",
    "constant_velocity",
    "damped_constant_velocity",
    "ballistic_contact",
)
_REGIMES = (
    "free",
    "ground_contact",
    "pair_contact",
    "collision",
    "occluded",
    "externally_actuated",
)
_LATENCY_PREFIXES = ("rgb_global_update", "rgb_fast_update", "future_rollout")
_SCHEMA_VERSION = "runtime_hypothesis_paired_promotion_v2"
Direction = Literal["lower", "higher", "maximum_ratio"]


@dataclass(frozen=True)
class CapturedReport:
    path: Path
    content: bytes
    sha256: str
    byte_count: int
    payload: dict[str, Any]
    device: int
    inode: int
    mode: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: str | Path, *, role: str) -> CapturedReport:
        source = Path(path).expanduser().resolve()
        before = source.stat()
        content = source.read_bytes()
        after = source.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise RuntimeError(f"{role} changed while it was captured")
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{role} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{role} must be a JSON object")
        _require_json_finite(payload, role=role)
        return cls(
            path=source,
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
            payload=payload,
            device=after.st_dev,
            inode=after.st_ino,
            mode=after.st_mode,
            mtime_ns=after.st_mtime_ns,
        )

    def assert_path_identity(self) -> None:
        current = self.path.stat()
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mode,
            current.st_mtime_ns,
        )
        expected = (
            self.device,
            self.inode,
            self.byte_count,
            self.mode,
            self.mtime_ns,
        )
        if identity != expected:
            raise RuntimeError(f"captured report path changed: {self.path}")

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "identity_source": "single_immutable_in_memory_byte_capture",
        }


@dataclass(frozen=True)
class ValidatedArm:
    role: str
    report: CapturedReport
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    primary_metrics: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _require_json_finite(value: Any, *, role: str, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{role} contains a nonfinite value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{role} contains a non-string key at {path}")
            _require_json_finite(item, role=role, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _require_json_finite(item, role=role, path=f"{path}[{index}]")
        return
    raise ValueError(f"{role} contains a non-JSON value at {path}")


def _number(metrics: Mapping[str, Any], name: str, *, role: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} metric {name!r} is missing or nonnumeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{role} metric {name!r} is nonfinite")
    return result


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{role} is not a SHA-256 hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{role} is not a SHA-256 hexadecimal digest") from error
    return value


def _tolerance(reference: float, *, absolute: float, relative: float) -> float:
    return max(absolute, relative * max(abs(reference), 1.0))


def _scenario_prefixes(scenarios: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return (("pooled", ""),) + tuple((scenario, f"scenario_{scenario}_") for scenario in scenarios)


def _guardrail(
    failures: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    reference_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    name: str,
    *,
    direction: Direction,
    absolute_tolerance: float,
    relative_tolerance: float,
    maximum_ratio: float | None = None,
) -> None:
    reference = _number(reference_metrics, name, role="reference")
    candidate = _number(candidate_metrics, name, role="candidate")
    tolerance = _tolerance(
        reference,
        absolute=absolute_tolerance,
        relative=relative_tolerance,
    )
    if direction == "lower":
        limit = reference + tolerance
        passed = candidate <= limit
    elif direction == "higher":
        limit = reference - tolerance
        passed = candidate >= limit
    else:
        if maximum_ratio is None or maximum_ratio < 1.0:
            raise ValueError("maximum-ratio guardrail requires a ratio >= 1")
        limit = reference * maximum_ratio + tolerance
        passed = candidate <= limit
    record = {
        "metric": name,
        "direction": direction,
        "reference": reference,
        "candidate": candidate,
        "delta": candidate - reference,
        "limit": limit,
        "passed": passed,
    }
    deltas.append(record)
    if not passed:
        failures.append(record)


def _require_exact_support(
    failures: list[dict[str, Any]],
    reference_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    name: str,
    *,
    positive: bool,
) -> None:
    reference = _number(reference_metrics, name, role="reference")
    candidate = _number(candidate_metrics, name, role="candidate")
    if reference != candidate or (positive and reference <= 0.0):
        failures.append(
            {
                "metric": name,
                "direction": "exact_positive_support" if positive else "exact_support",
                "reference": reference,
                "candidate": candidate,
                "delta": candidate - reference,
                "limit": reference,
                "passed": False,
            }
        )


def compare_runtime_hypothesis_metrics(
    reference_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    horizons: Sequence[str],
    scenarios: Sequence[str],
    axes: Sequence[int],
    absolute_tolerance: float = 1.0e-9,
    relative_tolerance: float = 1.0e-6,
    sharpness_maximum_ratio: float = 1.05,
    minimum_pooled_position_improvement_m: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare a matched evaluator pair with complete forecast guardrails."""

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("numeric tolerances must be nonnegative")
    if sharpness_maximum_ratio < 1.0:
        raise ValueError("sharpness maximum ratio must be at least 1")
    if minimum_pooled_position_improvement_m <= 0.0:
        raise ValueError("minimum pooled position improvement must be positive")
    if not horizons or not scenarios or not axes:
        raise ValueError("horizons, scenarios, and intervention axes must be nonempty")
    if any(axis not in (0, 1, 2) for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError("intervention axes must be unique members of {0,1,2}")

    failures: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    support_failures: list[dict[str, Any]] = []

    reference_keys = set(reference_metrics)
    candidate_keys = set(candidate_metrics)
    runtime_keys = {name for name in candidate_keys if name.startswith("runtime_hypothesis")}
    if {name for name in reference_keys if name.startswith("runtime_hypothesis")}:
        raise ValueError("learned-only reference contains runtime-hypothesis diagnostics")
    if candidate_keys - runtime_keys != reference_keys:
        missing = sorted(reference_keys - candidate_keys)
        unexpected = sorted((candidate_keys - runtime_keys) - reference_keys)
        raise ValueError(
            f"paired metric schemas differ outside runtime diagnostics: missing={missing}, "
            f"unexpected={unexpected}"
        )

    exact_current_prefixes = (
        "posterior_current_",
        "current_assignment_",
        "current_detection_",
        "distance_gated_",
    )
    exact_current_keys = sorted(
        name
        for name in reference_keys
        if name.startswith(exact_current_prefixes)
        or any(
            name.startswith(f"scenario_{scenario}_{prefix}")
            for scenario in scenarios
            for prefix in exact_current_prefixes
        )
    )
    for name in exact_current_keys:
        reference = reference_metrics[name]
        candidate = candidate_metrics[name]
        if reference is None or candidate is None:
            if reference != candidate:
                support_failures.append(
                    {
                        "metric": name,
                        "direction": "exact_current_posterior",
                        "reference": reference,
                        "candidate": candidate,
                        "passed": False,
                    }
                )
            continue
        ref_number = _number(reference_metrics, name, role="reference")
        cand_number = _number(candidate_metrics, name, role="candidate")
        tolerance = _tolerance(
            ref_number,
            absolute=absolute_tolerance,
            relative=relative_tolerance,
        )
        if abs(cand_number - ref_number) > tolerance:
            support_failures.append(
                {
                    "metric": name,
                    "direction": "exact_current_posterior",
                    "reference": ref_number,
                    "candidate": cand_number,
                    "delta": cand_number - ref_number,
                    "limit": tolerance,
                    "passed": False,
                }
            )

    axis_names = ("x", "y", "z")
    pooled_position_improvement = 0.0
    for slice_name, prefix in _scenario_prefixes(scenarios):
        for horizon in horizons:
            model = f"{prefix}model@{horizon}"
            for suffix in (
                "position_coordinate_count",
                "position_calibration_coordinate_count",
                "velocity_coordinate_count",
                "velocity_object_frame_count",
            ):
                _require_exact_support(
                    support_failures,
                    reference_metrics,
                    candidate_metrics,
                    f"{model}_{suffix}",
                    positive=True,
                )
            for axis_name in axis_names:
                for suffix in (
                    f"position_{axis_name}_count",
                    f"position_{axis_name}_calibration_coordinate_count",
                    f"velocity_{axis_name}_count",
                ):
                    _require_exact_support(
                        support_failures,
                        reference_metrics,
                        candidate_metrics,
                        f"{model}_{suffix}",
                        positive=True,
                    )
            for axis_suffix in ("", "_x", "_y", "_z"):
                _guardrail(
                    failures,
                    deltas,
                    reference_metrics,
                    candidate_metrics,
                    f"{model}_position{axis_suffix}_rmse_m",
                    direction="lower",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                _guardrail(
                    failures,
                    deltas,
                    reference_metrics,
                    candidate_metrics,
                    f"{model}_velocity{axis_suffix}_rmse_mps",
                    direction="lower",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                _guardrail(
                    failures,
                    deltas,
                    reference_metrics,
                    candidate_metrics,
                    f"{model}_position{axis_suffix}_gaussian_nll",
                    direction="lower",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                _guardrail(
                    failures,
                    deltas,
                    reference_metrics,
                    candidate_metrics,
                    f"{model}_position{axis_suffix}_calibration_error90",
                    direction="lower",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                _guardrail(
                    failures,
                    deltas,
                    reference_metrics,
                    candidate_metrics,
                    f"{model}_position{axis_suffix}_sharpness_std",
                    direction="maximum_ratio",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    maximum_ratio=sharpness_maximum_ratio,
                )
            for metric_name, direction in (
                (f"{prefix}forecast_target_coverage@{horizon}", "higher"),
                (f"{prefix}tracked_forecast_active_coverage@{horizon}", "higher"),
                (f"{prefix}forecast_identity@{horizon}_association_coverage", "higher"),
                (f"{prefix}forecast_identity@{horizon}_mismatch_rate", "lower"),
                (f"{prefix}collision@{horizon}_f1", "higher"),
                (f"{prefix}model_dropped_forecast_count@{horizon}", "lower"),
            ):
                _guardrail(
                    failures,
                    deltas,
                    reference_metrics,
                    candidate_metrics,
                    metric_name,
                    direction=direction,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            for suffix in (
                "forecast_target_count",
                "forecast_tracked_count",
                "forecast_active_count",
            ):
                _require_exact_support(
                    support_failures,
                    reference_metrics,
                    candidate_metrics,
                    f"{prefix}{suffix}@{horizon}",
                    positive=True,
                )
            for suffix in ("eligible_count",):
                _require_exact_support(
                    support_failures,
                    reference_metrics,
                    candidate_metrics,
                    f"{prefix}forecast_identity@{horizon}_{suffix}",
                    positive=True,
                )
            reference_positive = _number(
                reference_metrics,
                f"{prefix}collision@{horizon}_true_positive_count",
                role="reference",
            ) + _number(
                reference_metrics,
                f"{prefix}collision@{horizon}_false_negative_count",
                role="reference",
            )
            candidate_positive = _number(
                candidate_metrics,
                f"{prefix}collision@{horizon}_true_positive_count",
                role="candidate",
            ) + _number(
                candidate_metrics,
                f"{prefix}collision@{horizon}_false_negative_count",
                role="candidate",
            )
            reference_negative = _number(
                reference_metrics,
                f"{prefix}collision@{horizon}_true_negative_count",
                role="reference",
            ) + _number(
                reference_metrics,
                f"{prefix}collision@{horizon}_false_positive_count",
                role="reference",
            )
            candidate_negative = _number(
                candidate_metrics,
                f"{prefix}collision@{horizon}_true_negative_count",
                role="candidate",
            ) + _number(
                candidate_metrics,
                f"{prefix}collision@{horizon}_false_positive_count",
                role="candidate",
            )
            if (
                reference_positive <= 0.0
                or reference_negative <= 0.0
                or candidate_positive != reference_positive
                or candidate_negative != reference_negative
            ):
                support_failures.append(
                    {
                        "metric": f"{slice_name}.collision@{horizon}.class_support",
                        "direction": "exact_positive_and_negative_class_support",
                        "reference": [reference_positive, reference_negative],
                        "candidate": [candidate_positive, candidate_negative],
                        "passed": False,
                    }
                )
            if slice_name == "pooled":
                pooled_position_improvement += _number(
                    reference_metrics,
                    f"{model}_position_rmse_m",
                    role="reference",
                ) - _number(
                    candidate_metrics,
                    f"{model}_position_rmse_m",
                    role="candidate",
                )

    runtime_failures: list[dict[str, Any]] = []
    forecast_anchor_count = _number(
        candidate_metrics,
        "runtime_hypothesis_forecast_anchor_count",
        role="candidate",
    )
    if forecast_anchor_count <= 0.0:
        runtime_failures.append(
            {
                "metric": "runtime_hypothesis_forecast_anchor_count",
                "direction": "positive_support_required",
                "candidate": forecast_anchor_count,
                "passed": False,
            }
        )
    total_nonlearned = 0.0
    total_nonlearned_composed = 0.0
    for axis in axes:
        axis_name = axis_names[axis]
        selected = {
            candidate_name: _number(
                candidate_metrics,
                f"runtime_hypothesis_axis_{axis_name}_{candidate_name}_count",
                role="candidate",
            )
            for candidate_name in _CANDIDATES
        }
        supported = _number(
            candidate_metrics,
            f"runtime_hypothesis_axis_{axis_name}_supported_count",
            role="candidate",
        )
        fallback = _number(
            candidate_metrics,
            f"runtime_hypothesis_axis_{axis_name}_fallback_count",
            role="candidate",
        )
        if supported <= 0.0 or fallback < 0.0 or sum(selected.values()) != supported:
            runtime_failures.append(
                {
                    "metric": f"runtime_hypothesis_axis_{axis_name}_selection_partition",
                    "direction": "exact_positive_partition",
                    "candidate": {**selected, "supported": supported, "fallback": fallback},
                    "passed": False,
                }
            )
        composed = {
            candidate_name: _number(
                candidate_metrics,
                f"runtime_hypothesis_axis_{axis_name}_{candidate_name}_composed_step_count",
                role="candidate",
            )
            for candidate_name in _CANDIDATES
        }
        composed_total = _number(
            candidate_metrics,
            f"runtime_hypothesis_axis_{axis_name}_composed_total_step_count",
            role="candidate",
        )
        composed_fallback = _number(
            candidate_metrics,
            f"runtime_hypothesis_axis_{axis_name}_composed_fallback_step_count",
            role="candidate",
        )
        grid_fallback = _number(
            candidate_metrics,
            f"runtime_hypothesis_axis_{axis_name}_composition_grid_fallback_count",
            role="candidate",
        )
        if (
            composed_total <= 0.0
            or sum(composed.values()) != composed_total
            or composed_fallback < 0.0
            or composed_fallback > composed["learned"]
            or grid_fallback != 0.0
        ):
            runtime_failures.append(
                {
                    "metric": f"runtime_hypothesis_axis_{axis_name}_composition_partition",
                    "direction": "exact_aligned_composition_partition",
                    "candidate": {
                        **composed,
                        "total": composed_total,
                        "fallback": composed_fallback,
                        "grid_fallback": grid_fallback,
                    },
                    "passed": False,
                }
            )
        total_nonlearned += sum(selected[name] for name in _CANDIDATES[1:])
        total_nonlearned_composed += sum(composed[name] for name in _CANDIDATES[1:])
        for horizon in horizons:
            horizon_selected = {
                candidate_name: _number(
                    candidate_metrics,
                    f"runtime_hypothesis@{horizon}_axis_{axis_name}_{candidate_name}_count",
                    role="candidate",
                )
                for candidate_name in _CANDIDATES
            }
            horizon_supported = _number(
                candidate_metrics,
                f"runtime_hypothesis@{horizon}_axis_{axis_name}_supported_count",
                role="candidate",
            )
            horizon_fallback = _number(
                candidate_metrics,
                f"runtime_hypothesis@{horizon}_axis_{axis_name}_fallback_count",
                role="candidate",
            )
            if (
                horizon_supported < 0.0
                or horizon_fallback < 0.0
                or sum(horizon_selected.values()) != horizon_supported
            ):
                runtime_failures.append(
                    {
                        "metric": f"runtime_hypothesis@{horizon}_axis_{axis_name}_partition",
                        "direction": "exact_horizon_partition",
                        "candidate": {
                            **horizon_selected,
                            "supported": horizon_supported,
                            "fallback": horizon_fallback,
                        },
                        "passed": False,
                    }
                )
        regime_total = sum(
            _number(
                candidate_metrics,
                f"runtime_hypothesis_regime_{regime}_composed_step_count",
                role="candidate",
            )
            for regime in _REGIMES
        )
        if regime_total != composed_total:
            runtime_failures.append(
                {
                    "metric": f"runtime_hypothesis_axis_{axis_name}_regime_partition",
                    "direction": "exact_regime_partition",
                    "candidate": regime_total,
                    "limit": composed_total,
                    "passed": False,
                }
            )
    if total_nonlearned <= 0.0 or total_nonlearned_composed <= 0.0:
        runtime_failures.append(
            {
                "metric": "runtime_hypothesis_nonlearned_use",
                "direction": "positive_selected_and_composed_use_required",
                "candidate": {
                    "selected": total_nonlearned,
                    "composed_steps": total_nonlearned_composed,
                },
                "passed": False,
            }
        )

    if pooled_position_improvement < minimum_pooled_position_improvement_m:
        failures.append(
            {
                "metric": "pooled_position_rmse_sum_improvement_m",
                "direction": "minimum_improvement",
                "reference": None,
                "candidate": pooled_position_improvement,
                "delta": pooled_position_improvement,
                "limit": minimum_pooled_position_improvement_m,
                "passed": False,
            }
        )

    return {
        "physical_guardrail_passed": not failures,
        "support_guardrail_passed": not support_failures,
        "runtime_usage_guardrail_passed": not runtime_failures,
        "physical_promotion_eligible": not failures
        and not support_failures
        and not runtime_failures,
        "failure_count": len(failures) + len(support_failures) + len(runtime_failures),
        "failures": failures,
        "support_failures": support_failures,
        "runtime_usage_failures": runtime_failures,
        "guardrail_count": len(deltas),
        "guardrails": deltas,
        "exact_current_metric_count": len(exact_current_keys),
        "pooled_position_rmse_sum_improvement_m": pooled_position_improvement,
        "minimum_pooled_position_improvement_m": minimum_pooled_position_improvement_m,
        "runtime_nonlearned_selection_count": total_nonlearned,
        "runtime_nonlearned_composed_step_count": total_nonlearned_composed,
    }


def _expected_protocol(
    config: OrpheusConfig,
    *,
    checkpoint_sha256: str,
    resolved_seed_protocol: EvaluationSeedProtocol,
    runtime_hypothesis_pool: bool,
) -> dict[str, Any]:
    return {
        "schema_version": _EVALUATION_PROTOCOL_SCHEMA_VERSION,
        "metric_schema_version": _EVALUATION_METRIC_SCHEMA_VERSION,
        "per_scenario_metric_schema": _PER_SCENARIO_METRIC_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": _canonical_sha256(config.to_dict()),
        "split": resolved_seed_protocol.split,
        "seed_protocol": resolved_seed_protocol.name,
        "seed_manifest": list(resolved_seed_protocol.manifest.seeds),
        "horizons_seconds_requested": list(config.evaluation.horizons_seconds),
        "horizons_observation_grid": [
            f"{value:.3f}s" for value in config.evaluation.horizons_seconds
        ],
        "batch_size": min(config.training.batch_size, config.evaluation.episodes),
        "episode_count": config.evaluation.episodes,
        "runtime_intervention": {
            "evaluator_state_perturbation_in_primary": False,
            "runtime_hypothesis_pool": runtime_hypothesis_pool,
            "recovery_probe_enabled": False,
            "recovery_probe_position_std": config.evaluation.perturbation_position_std,
            "recovery_probe_velocity_std": config.evaluation.perturbation_velocity_std,
        },
    }


def _expected_runtime_policy(config: OrpheusConfig) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "policy_version": "evidence_bounded_entity_axis_regime_horizon_v2",
        "candidates": [
            {"name": "learned", "parameters": {}},
            {"name": "constant_velocity", "parameters": {"damping": 0.0}},
            {
                "name": "damped_constant_velocity",
                "parameters": {"damping": 0.05},
            },
            {
                "name": "ballistic_contact",
                "parameters": {"ground_height": 0.0, "event_logit": 5.0},
            },
        ],
        "evidence_horizons_seconds": list(config.runtime.hypothesis_evidence_horizons_seconds),
        "axis_independent_axes": list(config.runtime.hypothesis_axis_independent_axes),
        "axis_prior_strength": config.runtime.hypothesis_axis_prior_strength,
        "evidence_decay": config.runtime.hypothesis_evidence_decay,
        "temperature": 1.0,
        "score": "gaussian_nll_predictive_plus_rgb_measurement_variance",
        "selection_locality": "persistent_entity_axis_interaction_regime_exact_horizon",
        "local_applicability_enabled": config.runtime.hypothesis_local_applicability_enabled,
        "minimum_support_count": config.runtime.hypothesis_minimum_support_count,
        "maximum_evidence_age_seconds": (config.runtime.hypothesis_maximum_evidence_age_seconds),
        "minimum_observability": config.runtime.hypothesis_minimum_observability,
        "minimum_confidence_margin": config.runtime.hypothesis_minimum_confidence_margin,
        "robust_influence_delta": config.runtime.hypothesis_robust_influence_delta,
        "composition_step_seconds": config.runtime.hypothesis_composition_step_seconds,
        "unsupported_query_policy": "learned_fallback",
        "composition": "bounded_short_step_coherent_state",
        "timestamp_tolerance_seconds": config.runtime.hypothesis_timestamp_tolerance_seconds,
    }
    policy["fingerprint_sha256"] = _canonical_sha256(policy)
    return policy


def _expected_runtime_environment(expected_device: str) -> dict[str, Any]:
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_node": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "requested_device": expected_device,
        "resolved_device": expected_device,
        "precision": "float32",
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _validate_arm(
    report: CapturedReport,
    *,
    role: str,
    config: OrpheusConfig,
    current_source: Mapping[str, Any],
    expected_device: str,
    resolved_seed_protocol: EvaluationSeedProtocol,
    runtime_hypothesis_pool: bool,
) -> ValidatedArm:
    payload = report.payload
    metadata = payload.get("metadata")
    metrics = payload.get("metrics")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        raise ValueError(f"{role} report requires metadata and metrics objects")
    if config.training.validation_episodes != 32 or config.evaluation.episodes != 32:
        raise ValueError("paired runtime promotion requires the exact fixed-32 episode count")
    if config.evaluation.recovery_probe_enabled:
        raise ValueError("paired runtime promotion requires recovery probe disabled")
    if not config.evaluation.rgb_only or config.runtime.modality != "rgb":
        raise ValueError("paired runtime promotion requires RGB-only evaluation")
    if config.runtime.enable_debug_oracle or config.runtime.hypothesis_pool_enabled:
        raise ValueError("paired runtime promotion forbids default oracle/pool runtime input")
    if not config.runtime.hypothesis_local_applicability_enabled:
        raise ValueError("paired runtime promotion requires local applicability enabled")
    if config.runtime.hypothesis_composition_step_seconds is None:
        raise ValueError("paired runtime promotion requires bounded composition")
    if not config.runtime.hypothesis_axis_independent_axes:
        raise ValueError("paired runtime promotion requires at least one intervention axis")
    if metadata.get("checkpoint_identity_source") != (
        "captured_pre_evaluation_immutable_byte_snapshot"
    ):
        raise ValueError(f"{role} checkpoint was not captured immutably")
    checkpoint_sha = _require_sha256(
        metadata.get("checkpoint_sha256"),
        role=f"{role} checkpoint SHA-256",
    )
    checkpoint_bytes = metadata.get("checkpoint_byte_count")
    if (
        isinstance(checkpoint_bytes, bool)
        or not isinstance(checkpoint_bytes, int)
        or checkpoint_bytes <= 0
    ):
        raise ValueError(f"{role} checkpoint byte count is invalid")
    protocol = metadata.get("resolved_evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{role} resolved protocol is missing")
    expected_protocol = _expected_protocol(
        config,
        checkpoint_sha256=checkpoint_sha,
        resolved_seed_protocol=resolved_seed_protocol,
        runtime_hypothesis_pool=runtime_hypothesis_pool,
    )
    if protocol != expected_protocol:
        raise ValueError(f"{role} resolved protocol does not match the fixed evaluator contract")
    if metadata.get("resolved_evaluation_protocol_sha256") != _canonical_sha256(protocol):
        raise ValueError(f"{role} resolved protocol hash is invalid")
    manifest = list(resolved_seed_protocol.manifest.seeds)
    scenarios = list(config.simulator.scenario_mixture)
    episode_scenarios = [scenarios[int(seed) % len(scenarios)] for seed in manifest]
    resolved_scenarios = _canonical_json_value(
        {
            scenario: asdict(
                SphereWorldConfig.from_config(config)
                .for_scenario(scenario)
                .for_distribution(
                    "ood" if resolved_seed_protocol.split == "ood" else "in_distribution"
                )
            )
            for scenario in scenarios
        }
    )
    expected_metadata = {
        "evaluation_metric_schema_version": _EVALUATION_METRIC_SCHEMA_VERSION,
        "resolved_evaluation_config_sha256": _canonical_sha256(config.to_dict()),
        "simulator_version": SIMULATOR_VERSION,
        "evaluation_simulator_version": SIMULATOR_VERSION,
        "evaluation_specification_version": SPECIFICATION_VERSION,
        "scenario_mixture": scenarios,
        "resolved_scenarios": resolved_scenarios,
        "per_scenario_metrics_schema": _PER_SCENARIO_METRIC_SCHEMA,
        "per_scenario_metrics_status": "diagnostic_only_not_checkpoint_promotion_complete",
        "per_scenario_metrics_known_omissions": [
            "nonfinite_evidence",
            "physical_baselines",
            "configured_support_floor_markers",
        ],
        "per_scenario_metrics_scenarios": scenarios,
        "per_scenario_metrics_horizons": [
            f"{value:.3f}s" for value in config.evaluation.horizons_seconds
        ],
        "evaluation_episode_scenarios": episode_scenarios,
        "split": resolved_seed_protocol.split,
        "episodes": 32,
        "batches": math.ceil(32 / min(config.training.batch_size, 32)),
        "device": expected_device,
        "precision": "float32",
        "evaluation_runtime_environment": _expected_runtime_environment(expected_device),
        "rgb_only": True,
        "oracle_runtime_input_used": False,
        "primary_online_pass_evaluator_state_perturbation_free": True,
        "primary_online_pass_intervention_free_scope": (
            "evaluator_injected_state_perturbations_only"
        ),
        "recovery_probe_enabled": False,
        "evaluation_perturbations_applied": False,
        "runtime_hypothesis_pool_enabled": runtime_hypothesis_pool,
        **resolved_seed_protocol.metadata(),
        "primary_posterior_trace_frame_count": 32 * config.simulator.sequence_frames,
        "primary_posterior_trace_schema": "world_belief_tensor_fields_v1",
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"{role} metadata {name!r} does not match the fixed contract")
    source = metadata.get("evaluation_source_provenance")
    if not isinstance(source, Mapping):
        raise ValueError(f"{role} source provenance is missing")
    for name in ("commit", "runtime_source_fingerprint", "worktree_fingerprint", "dirty"):
        if source.get(name) != current_source.get(name):
            raise ValueError(f"{role} source provenance {name!r} does not match current source")
    if metadata.get("primary_physical_metrics_hash_excludes") != (
        _primary_physical_metrics_hash_exclusion_declaration()
    ):
        raise ValueError(f"{role} primary physical exclusion declaration is invalid")
    if metadata.get("primary_physical_metrics_scope") != (
        "clean_primary_metrics_before_isolated_recovery_probe_append"
    ):
        raise ValueError(f"{role} primary physical scope is invalid")
    primary = _primary_physical_metrics(metrics)
    if metadata.get("primary_physical_metrics_hashed_keys") != sorted(primary):
        raise ValueError(f"{role} primary physical key scope is invalid")
    if metadata.get("primary_physical_metrics_sha256") != _canonical_sha256(primary):
        raise ValueError(f"{role} primary physical hash is invalid")
    if _number(metrics, "nonfinite_output_count", role=role) != 0.0:
        raise ValueError(f"{role} contains nonfinite model output")
    if _number(metrics, "evaluated_episodes", role=role) != 32.0:
        raise ValueError(f"{role} evaluated episode count is invalid")
    for scenario in scenarios:
        if _number(metrics, f"scenario_{scenario}_episode_count", role=role) != 4.0:
            raise ValueError(f"{role} scenario {scenario!r} does not contain four episodes")
    for name in (
        "injected_perturbation_batch_updates",
        "recovery_probe_evaluated_episodes",
        "recovery_probe_nonfinite_output_count",
    ):
        if _number(metrics, name, role=role) != 0.0:
            raise ValueError(f"{role} contains forbidden recovery/intervention evidence")
    _require_sha256(
        metadata.get("primary_posterior_trace_sha256"),
        role=f"{role} posterior trace SHA-256",
    )
    if runtime_hypothesis_pool:
        policy = metadata.get("runtime_hypothesis_pool_policy")
        if not isinstance(policy, dict):
            raise ValueError("candidate runtime policy metadata is missing")
        if policy != _expected_runtime_policy(config):
            raise ValueError("candidate runtime policy does not match the exact config contract")
    elif metadata.get("runtime_hypothesis_pool_policy") is not None:
        raise ValueError("reference unexpectedly records a runtime pool policy")
    return ValidatedArm(
        role=role,
        report=report,
        metadata=metadata,
        metrics=metrics,
        primary_metrics=primary,
    )


def compare_evaluation_reports(
    reference: CapturedReport,
    candidate: CapturedReport,
    *,
    config: OrpheusConfig,
    current_source: Mapping[str, Any],
    expected_device: str = "mps",
    split: str = "validation",
    seed_protocol: str = STANDARD_SEED_PROTOCOL,
    seed_offset: int | None = None,
    absolute_tolerance: float = 1.0e-9,
    relative_tolerance: float = 1.0e-6,
    sharpness_maximum_ratio: float = 1.05,
    latency_maximum_ratio: float = 1.10,
    minimum_pooled_position_improvement_m: float = 1.0e-5,
) -> dict[str, Any]:
    resolved_seed_protocol = make_evaluation_seed_protocol(
        name=seed_protocol,
        split=split,
        episode_count=config.evaluation.episodes,
        training_validation_episodes=config.training.validation_episodes,
        seed_offset=seed_offset,
    )
    reference_arm = _validate_arm(
        reference,
        role="reference",
        config=config,
        current_source=current_source,
        expected_device=expected_device,
        resolved_seed_protocol=resolved_seed_protocol,
        runtime_hypothesis_pool=False,
    )
    candidate_arm = _validate_arm(
        candidate,
        role="candidate",
        config=config,
        current_source=current_source,
        expected_device=expected_device,
        resolved_seed_protocol=resolved_seed_protocol,
        runtime_hypothesis_pool=True,
    )
    for name in ("checkpoint_sha256", "checkpoint_byte_count", "checkpoint_step"):
        if reference_arm.metadata.get(name) != candidate_arm.metadata.get(name):
            raise ValueError(f"paired arms do not share checkpoint field {name!r}")
    for name in (
        "resolved_evaluation_config_sha256",
        "checkpoint_source_provenance",
        "checkpoint_simulator_version",
        "checkpoint_specification_version",
        "scenario_mixture",
        "resolved_scenarios",
        "evaluation_episode_scenarios",
        "device",
        "precision",
        "evaluation_runtime_environment",
        "primary_posterior_trace_frame_count",
        "primary_posterior_trace_schema",
    ):
        if reference_arm.metadata.get(name) != candidate_arm.metadata.get(name):
            raise ValueError(f"paired arms do not share metadata field {name!r}")
    if (
        reference_arm.metadata["primary_posterior_trace_sha256"]
        != (candidate_arm.metadata["primary_posterior_trace_sha256"])
    ):
        raise ValueError("runtime intervention changed the online posterior trace")

    horizons = [f"{value:.3f}s" for value in config.evaluation.horizons_seconds]
    physical = compare_runtime_hypothesis_metrics(
        reference_arm.metrics,
        candidate_arm.metrics,
        horizons=horizons,
        scenarios=config.simulator.scenario_mixture,
        axes=config.runtime.hypothesis_axis_independent_axes,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        sharpness_maximum_ratio=sharpness_maximum_ratio,
        minimum_pooled_position_improvement_m=minimum_pooled_position_improvement_m,
    )
    latency = paired_latency_guardrail(
        reference_metrics=reference_arm.metrics,
        candidate_metrics=candidate_arm.metrics,
        maximum_ratio=latency_maximum_ratio,
    ).metrics()
    comprehensive = bool(
        physical["physical_promotion_eligible"] and latency["latency_guardrail_promotion_eligible"]
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": comprehensive,
        "physical_promotion_eligible": physical["physical_promotion_eligible"],
        "latency_guardrail_supported": latency["latency_guardrail_supported"],
        "latency_guardrail_passed": latency["latency_guardrail_passed"],
        "comprehensive_promotion_eligible": comprehensive,
        "reference_report": reference.identity(),
        "candidate_report": candidate.identity(),
        "checkpoint_sha256": reference_arm.metadata["checkpoint_sha256"],
        "checkpoint_byte_count": reference_arm.metadata["checkpoint_byte_count"],
        "checkpoint_step": reference_arm.metadata["checkpoint_step"],
        "source_provenance": dict(current_source),
        "resolved_evaluation_config_sha256": _canonical_sha256(config.to_dict()),
        "protocol": {
            "split": resolved_seed_protocol.split,
            "seed_protocol": resolved_seed_protocol.name,
            "seed_offset": resolved_seed_protocol.seed_offset,
            "seed_role": resolved_seed_protocol.intended_use,
            "seed_manifest": list(resolved_seed_protocol.manifest.seeds),
            "scenario_mixture": list(config.simulator.scenario_mixture),
            "horizons": horizons,
            "device": expected_device,
            "precision": "float32",
            "reference_runtime_hypothesis_pool": False,
            "candidate_runtime_hypothesis_pool": True,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "sharpness_maximum_ratio": sharpness_maximum_ratio,
            "latency_maximum_ratio": latency_maximum_ratio,
            "minimum_pooled_position_improvement_m": (minimum_pooled_position_improvement_m),
        },
        "primary_physical_metrics_sha256": {
            "reference": reference_arm.metadata["primary_physical_metrics_sha256"],
            "candidate": candidate_arm.metadata["primary_physical_metrics_sha256"],
        },
        "posterior_trace_sha256": reference_arm.metadata["primary_posterior_trace_sha256"],
        "physical": physical,
        "latency": latency,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--split", default="validation", choices=["validation", "test", "ood"])
    parser.add_argument(
        "--seed-protocol",
        default=STANDARD_SEED_PROTOCOL,
        choices=EVALUATION_SEED_PROTOCOLS,
    )
    parser.add_argument("--seed-offset", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--sharpness-maximum-ratio", type=float, default=1.05)
    parser.add_argument("--latency-maximum-ratio", type=float, default=1.10)
    parser.add_argument("--minimum-pooled-position-improvement-m", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"comparison output must be fresh: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"comparison output parent does not exist: {output.parent}")
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("comparison output must remain outside the source repository")
    config_path = Path(args.config).expanduser().resolve()
    try:
        config_path.relative_to(repository)
    except ValueError as error:
        raise ValueError(
            "comparison config must be source-tracked inside the repository"
        ) from error
    config = load_config(args.config, overrides=args.set)
    current_source = capture_git_metadata(repository)
    if current_source.get("dirty") is not False:
        raise ValueError("runtime-hypothesis promotion comparison requires a clean source tree")
    reference = CapturedReport.capture(args.reference_report, role="reference report")
    candidate = CapturedReport.capture(args.candidate_report, role="candidate report")
    if reference.path == candidate.path:
        raise ValueError("reference and candidate reports must be distinct artifacts")
    if output in {reference.path, candidate.path}:
        raise ValueError("comparison output cannot overwrite an input report")
    result = compare_evaluation_reports(
        reference,
        candidate,
        config=config,
        current_source=current_source,
        expected_device=args.device,
        split=args.split,
        seed_protocol=args.seed_protocol,
        seed_offset=args.seed_offset,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        sharpness_maximum_ratio=args.sharpness_maximum_ratio,
        latency_maximum_ratio=args.latency_maximum_ratio,
        minimum_pooled_position_improvement_m=args.minimum_pooled_position_improvement_m,
    )
    reference.assert_path_identity()
    candidate.assert_path_identity()
    final_source = capture_git_metadata(repository)
    if final_source != current_source:
        raise RuntimeError("source provenance changed during comparison")
    encoded = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(output, encoded)
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
