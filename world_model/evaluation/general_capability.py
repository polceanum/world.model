"""Deterministic, truth-isolated protocol for broad world-model capability.

This module owns experiment descriptions and online metric reduction, not a
second simulator or a hidden model input.  Factor controls are consumed only
by data generation.  Runtime observations remain ordinary calibrated RGB-D
packets plus public known actions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar

CAPABILITY_MANIFEST_SCHEMA = "world_model_general_capability_manifest_v1"
CAPABILITY_FACTORS = (
    "sensor_noise",
    "physical_parameters",
    "camera_motion",
    "known_actions",
    "partial_visibility",
)
COMPOSITIONAL_FACTOR = "compositional_holdout"
ALL_CAPABILITY_FACTORS = (*CAPABILITY_FACTORS, COMPOSITIONAL_FACTOR)
PLANNING_TRAINING_LOSS_FIELDS: tuple[str, ...] = ()
_SPLIT_SEED_BASES = {
    "development": 91_000_000,
    "compositional_holdout": 92_000_000,
}
_T = TypeVar("_T")
_U = TypeVar("_U")


@dataclass(frozen=True, slots=True)
class FactorControls:
    """Generator-only controls for one capability row."""

    rgb_noise_std: float = 0.0
    exposure_scale: float = 1.0
    depth_noise_std_m: float = 0.0
    pixel_dropout_probability: float = 0.0
    radius_m: float = 0.21
    mass_kg: float = 1.0
    drag_per_second: float = 0.05
    restitution: float = 0.70
    friction: float = 0.20
    camera_motion: Literal["static", "orbital", "translating"] = "static"
    known_action_enabled: bool = False
    impulse_magnitude: float = 0.30
    impulse_phase: Literal["early", "middle", "late"] = "middle"
    impulse_direction_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    impulse_target_rank: int = 0
    occlusion_frames: int = 0
    recovery_observation_frames: int = 0

    def validate(self) -> FactorControls:
        finite = {name: value for name, value in asdict(self).items() if isinstance(value, float)}
        if any(not math.isfinite(value) for value in finite.values()):
            raise ValueError("factor controls must be finite")
        if not 0.0 <= self.pixel_dropout_probability <= 0.05:
            raise ValueError("pixel dropout must lie in [0,0.05]")
        if not 0 <= self.occlusion_frames <= 8:
            raise ValueError("short occlusion must contain zero through eight frames")
        if self.occlusion_frames and self.recovery_observation_frames < 3:
            raise ValueError("occlusion rows require at least three recovery observations")
        if not self.occlusion_frames and self.recovery_observation_frames:
            raise ValueError("non-occlusion rows cannot request recovery observations")
        if self.camera_motion not in {"static", "orbital", "translating"}:
            raise ValueError("unsupported calibrated camera motion")
        if not isinstance(self.known_action_enabled, bool):
            raise TypeError("known_action_enabled must be boolean")
        if self.impulse_target_rank < 0:
            raise ValueError("impulse target rank must be nonnegative")
        direction_norm = math.sqrt(sum(value * value for value in self.impulse_direction_world))
        if not math.isfinite(direction_norm) or abs(direction_norm - 1.0) > 1.0e-6:
            raise ValueError("impulse direction must be a finite unit vector")
        return self


@dataclass(frozen=True, slots=True)
class CapabilityManifestRow:
    """One deterministic physical or planning evaluation row."""

    schema: str
    split: Literal["development", "compositional_holdout"]
    kind: Literal["physical", "planning"]
    ordinal: int
    seed: int
    factor: str
    object_count: int
    contact: bool
    dynamic_membership: bool
    candidate_count: int | None
    controls: FactorControls

    def validate(self) -> CapabilityManifestRow:
        if self.schema != CAPABILITY_MANIFEST_SCHEMA:
            raise ValueError("unsupported capability manifest schema")
        if self.split not in _SPLIT_SEED_BASES:
            raise ValueError("unsupported capability split")
        expected_factor = (
            COMPOSITIONAL_FACTOR if self.split == "compositional_holdout" else self.factor
        )
        if expected_factor not in ALL_CAPABILITY_FACTORS:
            raise ValueError("unsupported capability factor")
        if not 1 <= self.object_count <= 6:
            raise ValueError("this phase promotes only on one through six objects")
        if self.kind == "physical" and self.candidate_count is not None:
            raise ValueError("physical rows cannot carry candidate_count")
        if self.kind == "planning" and self.candidate_count not in {8, 32}:
            raise ValueError("planning rows require K=8 or K=32")
        if self.object_count == 1 and self.contact:
            raise ValueError("one-object rows cannot contain pair contact")
        if self.controls.impulse_target_rank >= self.object_count:
            raise ValueError("impulse target rank lies outside the active object set")
        if isinstance(self.seed, bool) or self.seed < _SPLIT_SEED_BASES[self.split]:
            raise ValueError("capability seed is outside its split namespace")
        self.controls.validate()
        return self

    def generator_payload(self) -> dict[str, Any]:
        """Return deterministic controls available only to scene generation."""

        return {
            "seed": self.seed,
            "object_count": self.object_count,
            "contact": self.contact,
            "dynamic_membership": self.dynamic_membership,
            "controls": asdict(self.controls),
        }

    def runtime_contract(self) -> dict[str, tuple[str, ...]]:
        """Declare the only observation fields allowed to cross the runtime boundary."""

        return {
            "observation": ("rgb", "depth", "timestamp"),
            "calibration": ("intrinsics", "world_from_camera"),
            "known_action": ("timestamp", "appearance_handle", "impulse_world"),
        }


@dataclass(frozen=True, slots=True)
class CapabilityConvergenceSchedule:
    """Bounded convergence campaign requested by the capability phase."""

    screen_examples: int = 64
    screen_updates: int = 256
    initial_updates: int = 2_048
    validation_every_updates: int = 256
    extension_updates: int = 1_024
    maximum_updates: int = 8_192
    maximum_hours: float = 24.0
    plateau_validations: int = 4
    plateau_relative_improvement: float = 0.01

    def validate(self) -> CapabilityConvergenceSchedule:
        if self.screen_examples != 64 or self.screen_updates != 256:
            raise ValueError("capability screen must remain 64 examples / 256 updates")
        if self.initial_updates % self.validation_every_updates:
            raise ValueError("initial updates must land on a validation boundary")
        if self.extension_updates % self.validation_every_updates:
            raise ValueError("extensions must land on validation boundaries")
        if self.maximum_updates < self.initial_updates or self.maximum_hours <= 0.0:
            raise ValueError("invalid convergence limits")
        return self

    def stop_reason(
        self,
        *,
        completed_updates: int,
        elapsed_hours: float,
        raw_scores: Sequence[float],
        accepted_in_window: bool,
    ) -> str | None:
        """Return a bounded stop reason, or ``None`` while work may continue."""

        self.validate()
        if completed_updates >= self.maximum_updates or elapsed_hours >= self.maximum_hours:
            return "limit_hit"
        if len(raw_scores) < self.plateau_validations or accepted_in_window:
            return None
        window = tuple(float(value) for value in raw_scores[-self.plateau_validations :])
        if any(not math.isfinite(value) or value < 0.0 for value in window):
            raise ValueError("validation scores must be finite and nonnegative")
        incumbent = window[0]
        improvement = (incumbent - min(window[1:], default=incumbent)) / max(
            incumbent,
            1.0e-12,
        )
        return "objective_plateau" if improvement < self.plateau_relative_improvement else None


@dataclass(frozen=True, slots=True)
class CapabilityAcceptanceCriteria:
    """Absolute floors and relative promotion requirements for this phase."""

    proposal_f1: float = 0.95
    identity_accuracy: float = 0.98
    lifecycle_f1: float = 0.95
    current_position_rmse_m: float = 0.020
    two_second_position_rmse_m: float = 0.120
    collision_f1: float = 0.90
    uncertainty_coverage_minimum: float = 0.82
    uncertainty_coverage_maximum: float = 0.97
    planning_k8_winner_accuracy: float = 0.90
    planning_k32_winner_accuracy: float = 0.85
    planning_k8_median_regret: float = 0.05
    planning_k32_median_regret: float = 0.07
    planning_goal_success: float = 0.90
    compositional_proposal_f1: float = 0.90
    compositional_identity_accuracy: float = 0.95
    compositional_two_second_rmse_m: float = 0.150
    compositional_k8_winner_accuracy: float = 0.80
    compositional_k32_winner_accuracy: float = 0.75
    minimum_score_improvement: float = 0.03
    minimum_worst_family_improvement: float = 0.01
    maximum_latency_ratio: float = 1.10
    maximum_accepted_regression: float = 0.02
    maximum_learned_weight_bytes: int = 1 << 20
    maximum_rss_bytes: int = int(2.5 * (1 << 30))
    maximum_n16_rollout_seconds: float = 0.10


DEFAULT_CAPABILITY_ACCEPTANCE_CRITERIA = CapabilityAcceptanceCriteria()


def _number(evidence: Mapping[str, Any], name: str) -> float:
    value = evidence.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"acceptance evidence lacks numeric {name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"acceptance evidence {name} is nonfinite")
    return number


def capability_gate_failures(
    evidence: Mapping[str, Any],
    *,
    criteria: CapabilityAcceptanceCriteria = DEFAULT_CAPABILITY_ACCEPTANCE_CRITERIA,
) -> tuple[str, ...]:
    """Evaluate absolute single-factor, composition, planning, and resource gates."""

    failures: list[str] = []
    factors = evidence.get("single_factors")
    if not isinstance(factors, Mapping):
        return ("single_factors:unmeasured",)
    for factor in CAPABILITY_FACTORS:
        metrics = factors.get(factor)
        if not isinstance(metrics, Mapping):
            failures.append(f"{factor}:unmeasured")
            continue
        lower = {
            "proposal_f1": criteria.proposal_f1,
            "identity_accuracy": criteria.identity_accuracy,
            "lifecycle_f1": criteria.lifecycle_f1,
            "collision_f1": criteria.collision_f1,
        }
        upper = {
            "current_position_rmse_m": criteria.current_position_rmse_m,
            "two_second_position_rmse_m": criteria.two_second_position_rmse_m,
        }
        for name, limit in lower.items():
            if _number(metrics, name) < limit:
                failures.append(f"{factor}/{name}")
        for name, limit in upper.items():
            if _number(metrics, name) > limit:
                failures.append(f"{factor}/{name}")
        coverage = _number(metrics, "uncertainty_90_coverage")
        if (
            not criteria.uncertainty_coverage_minimum
            <= coverage
            <= (criteria.uncertainty_coverage_maximum)
        ):
            failures.append(f"{factor}/uncertainty_90_coverage")
    compositional = evidence.get("compositional_holdout")
    if not isinstance(compositional, Mapping):
        failures.append("compositional_holdout:unmeasured")
    else:
        compositional_lower = {
            "proposal_f1": criteria.compositional_proposal_f1,
            "identity_accuracy": criteria.compositional_identity_accuracy,
            "planning_k8_winner_accuracy": criteria.compositional_k8_winner_accuracy,
            "planning_k32_winner_accuracy": criteria.compositional_k32_winner_accuracy,
        }
        for name, limit in compositional_lower.items():
            if _number(compositional, name) < limit:
                failures.append(f"compositional_holdout/{name}")
        if _number(compositional, "two_second_position_rmse_m") > (
            criteria.compositional_two_second_rmse_m
        ):
            failures.append("compositional_holdout/two_second_position_rmse_m")
    planning = evidence.get("planning")
    if not isinstance(planning, Mapping):
        failures.append("planning:unmeasured")
    else:
        planning_lower = {
            "k8_winner_accuracy": criteria.planning_k8_winner_accuracy,
            "k32_winner_accuracy": criteria.planning_k32_winner_accuracy,
            "goal_success": criteria.planning_goal_success,
        }
        planning_upper = {
            "k8_median_regret": criteria.planning_k8_median_regret,
            "k32_median_regret": criteria.planning_k32_median_regret,
        }
        for name, limit in planning_lower.items():
            if _number(planning, name) < limit:
                failures.append(f"planning/{name}")
        for name, limit in planning_upper.items():
            if _number(planning, name) > limit:
                failures.append(f"planning/{name}")
        if planning.get("serial_vectorized_winner_parity") is not True:
            failures.append("planning/serial_vectorized_winner_parity")
    resources = evidence.get("resources")
    if not isinstance(resources, Mapping):
        failures.append("resources:unmeasured")
    else:
        if _number(resources, "learned_weight_bytes") > criteria.maximum_learned_weight_bytes:
            failures.append("resources/learned_weight_bytes")
        if _number(resources, "rss_bytes") > criteria.maximum_rss_bytes:
            failures.append("resources/rss_bytes")
        if _number(resources, "n16_rollout_seconds") > criteria.maximum_n16_rollout_seconds:
            failures.append("resources/n16_rollout_seconds")
    if _number(evidence, "maximum_accepted_regression") > criteria.maximum_accepted_regression:
        failures.append("accepted_like_regression")
    invariants = evidence.get("invariant_failures")
    if not isinstance(invariants, list):
        failures.append("invariants:unmeasured")
    else:
        failures.extend(f"invariant/{item}" for item in invariants)
    return tuple(failures)


def promotion_decision(
    evidence: Mapping[str, Any],
    *,
    criteria: CapabilityAcceptanceCriteria = DEFAULT_CAPABILITY_ACCEPTANCE_CRITERIA,
) -> dict[str, Any]:
    """Require every absolute and paired relative gate before promotion."""

    failures = list(capability_gate_failures(evidence, criteria=criteria))
    relative_improvement = _number(evidence, "relative_score_improvement")
    bootstrap_lower = _number(evidence, "paired_bootstrap_improvement_lower_95")
    worst_family_improvement = _number(evidence, "worst_family_improvement")
    latency_ratio = _number(evidence, "latency_ratio")
    if relative_improvement < criteria.minimum_score_improvement:
        failures.append("relative_score_improvement")
    if bootstrap_lower <= 0.0:
        failures.append("paired_bootstrap_improvement_lower_95")
    if worst_family_improvement < criteria.minimum_worst_family_improvement:
        failures.append("worst_family_improvement")
    if latency_ratio > criteria.maximum_latency_ratio:
        failures.append("latency_ratio")
    return {"promoted": not failures, "failures": tuple(failures)}


@dataclass
class StreamingMetricReducer:
    """Constant-memory sufficient statistics for scalar validation metrics."""

    count: dict[str, int]
    total: dict[str, float]
    squared_total: dict[str, float]
    minimum: dict[str, float]
    maximum: dict[str, float]

    def __init__(self) -> None:
        self.count = {}
        self.total = {}
        self.squared_total = {}
        self.minimum = {}
        self.maximum = {}

    def update(self, metrics: Mapping[str, float]) -> None:
        for name, raw in metrics.items():
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"metric {name!r} is nonfinite")
            self.count[name] = self.count.get(name, 0) + 1
            self.total[name] = math.fsum((self.total.get(name, 0.0), value))
            self.squared_total[name] = math.fsum((self.squared_total.get(name, 0.0), value * value))
            self.minimum[name] = min(self.minimum.get(name, value), value)
            self.maximum[name] = max(self.maximum.get(name, value), value)

    def merge(self, other: StreamingMetricReducer) -> None:
        for name, count in other.count.items():
            self.count[name] = self.count.get(name, 0) + count
            self.total[name] = math.fsum((self.total.get(name, 0.0), other.total[name]))
            self.squared_total[name] = math.fsum(
                (self.squared_total.get(name, 0.0), other.squared_total[name])
            )
            self.minimum[name] = min(
                self.minimum.get(name, other.minimum[name]), other.minimum[name]
            )
            self.maximum[name] = max(
                self.maximum.get(name, other.maximum[name]), other.maximum[name]
            )

    def result(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for name in sorted(self.count):
            count = self.count[name]
            mean = self.total[name] / count
            variance = max(0.0, self.squared_total[name] / count - mean * mean)
            output[name] = {
                "count": count,
                "mean": mean,
                "standard_deviation": math.sqrt(variance),
                "minimum": self.minimum[name],
                "maximum": self.maximum[name],
            }
        return output


def _controls(factor: str, seed: int, object_count: int) -> FactorControls:
    rng = random.Random(seed)
    sensor = factor in {"sensor_noise", COMPOSITIONAL_FACTOR}
    physics = factor in {"physical_parameters", COMPOSITIONAL_FACTOR}
    camera = factor in {"camera_motion", COMPOSITIONAL_FACTOR}
    action = factor in {"known_actions", COMPOSITIONAL_FACTOR}
    occlusion = factor in {"partial_visibility", COMPOSITIONAL_FACTOR}
    azimuth = rng.uniform(-math.pi, math.pi)
    elevation = rng.uniform(-0.35, 0.35)
    horizontal = math.cos(elevation)
    return FactorControls(
        rgb_noise_std=rng.uniform(0.005, 0.03) if sensor else 0.0,
        exposure_scale=rng.uniform(0.8, 1.2) if sensor else 1.0,
        depth_noise_std_m=rng.uniform(0.001, 0.005) if sensor else 0.0,
        pixel_dropout_probability=rng.uniform(0.01, 0.05) if sensor else 0.0,
        radius_m=rng.uniform(0.16, 0.28) if physics else 0.21,
        mass_kg=rng.uniform(0.6, 1.8) if physics else 1.0,
        drag_per_second=rng.uniform(0.01, 0.16) if physics else 0.05,
        restitution=rng.uniform(0.45, 0.90) if physics else 0.70,
        friction=rng.uniform(0.05, 0.35) if physics else 0.20,
        camera_motion=(rng.choice(("static", "orbital", "translating")) if camera else "static"),
        known_action_enabled=action and (factor == COMPOSITIONAL_FACTOR or seed % 2 == 0),
        impulse_magnitude=rng.uniform(0.15, 0.60) if action else 0.30,
        impulse_phase=rng.choice(("early", "middle", "late")) if action else "middle",
        impulse_direction_world=(
            horizontal * math.cos(azimuth),
            math.sin(elevation),
            horizontal * math.sin(azimuth),
        )
        if action
        else (1.0, 0.0, 0.0),
        impulse_target_rank=seed % object_count if action else 0,
        occlusion_frames=rng.randint(1, 8) if occlusion else 0,
        recovery_observation_frames=3 if occlusion else 0,
    ).validate()


def _physical_cells() -> tuple[tuple[int, bool, bool], ...]:
    cells: list[tuple[int, bool, bool]] = [(1, False, False), (1, False, True)]
    for count in range(2, 7):
        for contact in (False, True):
            for dynamic in (False, True):
                cells.append((count, contact, dynamic))
    assert len(cells) == 22
    return tuple(cells)


def capability_manifest(
    split: Literal["development", "compositional_holdout"],
) -> tuple[CapabilityManifestRow, ...]:
    """Build the complete deterministic physical and planning manifest."""

    if split not in _SPLIT_SEED_BASES:
        raise ValueError("unsupported capability split")
    factors = CAPABILITY_FACTORS if split == "development" else (COMPOSITIONAL_FACTOR,)
    rows: list[CapabilityManifestRow] = []
    ordinal = 0
    base = _SPLIT_SEED_BASES[split]
    for factor in factors:
        for object_count, contact, dynamic in _physical_cells():
            seed = base + ordinal
            rows.append(
                CapabilityManifestRow(
                    schema=CAPABILITY_MANIFEST_SCHEMA,
                    split=split,
                    kind="physical",
                    ordinal=ordinal,
                    seed=seed,
                    factor=factor,
                    object_count=object_count,
                    contact=contact,
                    dynamic_membership=dynamic,
                    candidate_count=None,
                    controls=_controls(factor, seed, object_count),
                ).validate()
            )
            ordinal += 1
        for object_count in range(1, 7):
            for candidate_count in (8, 32):
                seed = base + ordinal
                rows.append(
                    CapabilityManifestRow(
                        schema=CAPABILITY_MANIFEST_SCHEMA,
                        split=split,
                        kind="planning",
                        ordinal=ordinal,
                        seed=seed,
                        factor=factor,
                        object_count=object_count,
                        contact=object_count > 1 and ordinal % 2 == 0,
                        dynamic_membership=ordinal % 3 == 0,
                        candidate_count=candidate_count,
                        controls=_controls(factor, seed, object_count),
                    ).validate()
                )
                ordinal += 1
    return tuple(rows)


def manifest_sha256(rows: Sequence[CapabilityManifestRow]) -> str:
    payload = [asdict(row.validate()) for row in rows]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assert_split_disjoint(*populations: Sequence[CapabilityManifestRow]) -> None:
    seen_seeds: set[int] = set()
    seen_keys: set[tuple[str, int]] = set()
    for population in populations:
        for row in population:
            row.validate()
            key = (row.split, row.ordinal)
            if row.seed in seen_seeds or key in seen_keys:
                raise ValueError("capability manifests overlap")
            seen_seeds.add(row.seed)
            seen_keys.add(key)


def stream_materializations(
    rows: Iterable[_T],
    materialize: Callable[[_T], _U],
) -> Iterator[_U]:
    """Materialize exactly one row at a time without retaining prior tensors."""

    for row in rows:
        yield materialize(row)


def attribute_failure(ablation_errors: Mapping[str, float]) -> dict[str, Any]:
    """Attribute a supported error to the smallest tested subsystem owner."""

    required = {
        "observed",
        "clean_observation",
        "truth_association",
        "truth_parameters",
        "truth_state_dynamics",
    }
    if set(ablation_errors) != required:
        raise ValueError(f"ablation errors must contain exactly {sorted(required)}")
    values = {name: float(value) for name, value in ablation_errors.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("ablation errors must be finite and nonnegative")
    baseline = values["observed"]
    ownership = {
        "perception": baseline - values["clean_observation"],
        "association": baseline - values["truth_association"],
        "physical_parameters": baseline - values["truth_parameters"],
        "dynamics": baseline - values["truth_state_dynamics"],
    }
    owner = max(ownership, key=ownership.get)
    reduction = ownership[owner]
    return {
        "owner": owner if reduction > 0.0 else "unsupported",
        "absolute_error_reduction": max(0.0, reduction),
        "relative_error_reduction": max(0.0, reduction) / max(baseline, 1.0e-12),
        "all_reductions": ownership,
        "capacity_change_supported": reduction > 0.0,
    }


def training_admission(factor_status: Mapping[str, str]) -> tuple[bool, tuple[str, ...]]:
    """Require incumbent evidence for every family before optimization."""

    missing = tuple(
        factor
        for factor in ALL_CAPABILITY_FACTORS
        if factor_status.get(factor) not in {"measured", "passed", "failed"}
    )
    return not missing, missing


__all__ = [
    "ALL_CAPABILITY_FACTORS",
    "CAPABILITY_FACTORS",
    "CAPABILITY_MANIFEST_SCHEMA",
    "COMPOSITIONAL_FACTOR",
    "PLANNING_TRAINING_LOSS_FIELDS",
    "CapabilityConvergenceSchedule",
    "CapabilityAcceptanceCriteria",
    "CapabilityManifestRow",
    "FactorControls",
    "StreamingMetricReducer",
    "assert_split_disjoint",
    "attribute_failure",
    "capability_gate_failures",
    "capability_manifest",
    "manifest_sha256",
    "promotion_decision",
    "stream_materializations",
    "training_admission",
]
