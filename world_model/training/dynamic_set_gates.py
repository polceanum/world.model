"""Absolute promotion gates for the specification-1.61 world model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

from world_model.training.dynamic_set_protocol import PhysicalCell

HORIZON_POSITION_LIMITS_M: Mapping[float, float] = {
    0.05: 0.012,
    0.10: 0.014,
    0.25: 0.018,
    0.50: 0.025,
    1.00: 0.035,
    2.00: 0.050,
}
MAXIMUM_PROCESS_RSS_BYTES = int(2.5 * 1024**3)


@dataclass(frozen=True)
class SupportedScalar:
    value: float
    support: int

    def validate(self, *, name: str) -> SupportedScalar:
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError(f"{name} value must be a real scalar")
        if not math.isfinite(float(self.value)):
            raise ValueError(f"{name} must be finite")
        if isinstance(self.support, bool) or not isinstance(self.support, int) or self.support < 0:
            raise ValueError(f"{name} support must be a nonnegative integer")
        return self


@dataclass(frozen=True)
class PhysicalCellMetrics:
    proposal_precision: SupportedScalar
    proposal_recall: SupportedScalar
    proposal_f1: SupportedScalar
    exact_count_accuracy: SupportedScalar
    current_position_rmse_m: SupportedScalar
    mature_velocity_rmse_mps: SupportedScalar
    post_event_velocity_rmse_mps: SupportedScalar | None
    horizon_position_rmse_m: Mapping[float, SupportedScalar]
    collision_f1: SupportedScalar | None
    collision_timing_error_frames: SupportedScalar | None
    persistent_id_accuracy: SupportedScalar
    identity_switch_rate: SupportedScalar
    birth_precision: SupportedScalar | None
    birth_recall: SupportedScalar | None
    birth_latency_frames: SupportedScalar | None
    removal_precision: SupportedScalar | None
    removal_recall: SupportedScalar | None
    removal_latency_frames: SupportedScalar | None
    uncertainty_90_coverage: SupportedScalar


@dataclass(frozen=True)
class PlanningSliceMetrics:
    object_count: int
    candidate_count: int
    distribution: str
    handle_resolution: SupportedScalar
    oracle_winner_accuracy: SupportedScalar
    normalized_regret_median: SupportedScalar
    normalized_regret_p95: SupportedScalar
    successful_oracle_goal_success: SupportedScalar


@dataclass(frozen=True)
class PlanningPopulationGateMetrics:
    """Exact task-population aggregates at the groupings named by the gates."""

    handle_resolution_by_object_count: Mapping[str, SupportedScalar]
    oracle_winner_accuracy_by_candidate_distribution: Mapping[str, SupportedScalar]
    normalized_regret_median_by_candidate_count: Mapping[str, SupportedScalar]
    normalized_regret_p95_by_candidate_count: Mapping[str, SupportedScalar]
    successful_oracle_goal_success: SupportedScalar


@dataclass(frozen=True)
class PlanningInvariantMetrics:
    serial_vectorized_winner_parity: bool
    maximum_cost_difference: float
    pre_action_invariance: bool
    exactly_once_impulse: bool
    action_target_isolation: bool
    conservation: bool
    batch_independence: bool
    source_belief_unchanged: bool
    latency_k8_seconds: float
    latency_k32_seconds: float


@dataclass(frozen=True)
class ResourceMetrics:
    perception_latency_seconds: float
    six_horizon_rollout_seconds: float
    learned_weight_bytes: int
    persistent_tensor_bytes: int
    process_rss_bytes: int


@dataclass(frozen=True)
class PromotionIntegrityMetrics:
    paired_score_improvement_fraction: float
    paired_bootstrap_lower_bound: float
    maximum_critical_regression_fraction: float
    maximum_accepted_like_regression_fraction: float
    truth_leakage_count: int
    fabricated_target_count: int
    nonfinite_state_count: int
    rejected_optimizer_mutation_count: int
    minimum_complete_gradient_retention: float


def _require_support(metric: SupportedScalar | None, *, name: str) -> list[str]:
    if metric is None:
        return [f"{name}:missing"]
    metric.validate(name=name)
    return [] if metric.support > 0 else [f"{name}:unsupported"]


def _maximum_failure(
    metric: SupportedScalar | None,
    *,
    name: str,
    limit: float,
) -> list[str]:
    failures = _require_support(metric, name=name)
    if metric is not None and metric.support > 0 and metric.value > limit:
        failures.append(f"{name}:{metric.value:.12g}>{limit:.12g}")
    return failures


def _minimum_failure(
    metric: SupportedScalar | None,
    *,
    name: str,
    limit: float,
) -> list[str]:
    failures = _require_support(metric, name=name)
    if metric is not None and metric.support > 0 and metric.value < limit:
        failures.append(f"{name}:{metric.value:.12g}<{limit:.12g}")
    return failures


def physical_cell_gate_failures(
    cell: PhysicalCell,
    metrics: PhysicalCellMetrics,
) -> tuple[str, ...]:
    cell.validate()
    failures: list[str] = []
    for name, metric in (
        ("proposal_precision", metrics.proposal_precision),
        ("proposal_recall", metrics.proposal_recall),
        ("proposal_f1", metrics.proposal_f1),
    ):
        failures.extend(_minimum_failure(metric, name=name, limit=0.98))
    failures.extend(
        _minimum_failure(metrics.exact_count_accuracy, name="exact_count_accuracy", limit=0.95)
    )
    failures.extend(
        _maximum_failure(
            metrics.current_position_rmse_m,
            name="current_position_rmse_m",
            limit=0.010,
        )
    )
    failures.extend(
        _maximum_failure(
            metrics.mature_velocity_rmse_mps,
            name="mature_velocity_rmse_mps",
            limit=0.020,
        )
    )
    if cell.dynamic_membership or cell.contact:
        failures.extend(
            _maximum_failure(
                metrics.post_event_velocity_rmse_mps,
                name="post_event_velocity_rmse_mps",
                limit=0.050,
            )
        )
    elif metrics.post_event_velocity_rmse_mps is not None:
        # Static no-contact cells have no lifecycle/contact reset and therefore
        # no applicable post-event population.  Rejecting a supplied scalar
        # prevents incidental samples from masquerading as complete support.
        failures.append("post_event_velocity_rmse_mps:inapplicable")
    for horizon, limit in HORIZON_POSITION_LIMITS_M.items():
        failures.extend(
            _maximum_failure(
                metrics.horizon_position_rmse_m.get(horizon),
                name=f"horizon_{horizon:.2f}_position_rmse_m",
                limit=limit,
            )
        )
    if set(metrics.horizon_position_rmse_m) != set(HORIZON_POSITION_LIMITS_M):
        failures.append("horizon_position_rmse_m:unexpected_schema")
    failures.extend(_minimum_failure(metrics.collision_f1, name="collision_f1", limit=0.95))
    failures.extend(
        _maximum_failure(
            metrics.collision_timing_error_frames,
            name="collision_timing_error_frames",
            limit=1.0,
        )
    )
    failures.extend(
        _minimum_failure(
            metrics.persistent_id_accuracy,
            name="persistent_id_accuracy",
            limit=0.99,
        )
    )
    failures.extend(
        _maximum_failure(
            metrics.identity_switch_rate,
            name="identity_switch_rate",
            limit=0.005,
        )
    )
    for name, metric in (
        ("birth_precision", metrics.birth_precision),
        ("birth_recall", metrics.birth_recall),
        ("removal_precision", metrics.removal_precision),
        ("removal_recall", metrics.removal_recall),
    ):
        failures.extend(_minimum_failure(metric, name=name, limit=0.98))
    failures.extend(
        _maximum_failure(
            metrics.birth_latency_frames,
            name="birth_latency_frames",
            limit=1.0,
        )
    )
    failures.extend(
        _maximum_failure(
            metrics.removal_latency_frames,
            name="removal_latency_frames",
            limit=2.0,
        )
    )
    failures.extend(
        _minimum_failure(
            metrics.uncertainty_90_coverage,
            name="uncertainty_90_coverage",
            limit=0.85,
        )
    )
    failures.extend(
        _maximum_failure(
            metrics.uncertainty_90_coverage,
            name="uncertainty_90_coverage",
            limit=0.95,
        )
    )
    return tuple(failures)


def planning_gate_failures(
    metrics: PlanningPopulationGateMetrics,
    invariants: PlanningInvariantMetrics,
    *,
    expected_distributions: Sequence[str] = (
        "in_distribution",
        "compositional_ood",
    ),
) -> tuple[str, ...]:
    failures: list[str] = []
    distributions = tuple(expected_distributions)
    if (
        not distributions
        or len(set(distributions)) != len(distributions)
        or any(
            distribution not in {"in_distribution", "compositional_ood"}
            for distribution in distributions
        )
    ):
        raise ValueError("expected_distributions must be a unique nonempty supported subset")
    if not isinstance(metrics, PlanningPopulationGateMetrics):
        raise TypeError("planning gates require exact population aggregates")
    expected_handle_keys = {f"N{object_count}" for object_count in range(1, 7)}
    if set(metrics.handle_resolution_by_object_count) != expected_handle_keys:
        failures.append("handle_resolution:incomplete_cardinalities")
    for object_count in range(1, 7):
        key = f"N{object_count}"
        failures.extend(
            _minimum_failure(
                metrics.handle_resolution_by_object_count.get(key),
                name=f"handle_resolution/{key}",
                limit=0.99,
            )
        )
    expected_winner_keys = {
        f"K{candidate_count}/{distribution}"
        for candidate_count in (8, 32)
        for distribution in distributions
    }
    if set(metrics.oracle_winner_accuracy_by_candidate_distribution) != expected_winner_keys:
        failures.append("oracle_winner_accuracy:incomplete_candidate_distribution_groups")
    for candidate_count in (8, 32):
        for distribution in distributions:
            key = f"K{candidate_count}/{distribution}"
            winner_limit = 0.95 if candidate_count == 8 else 0.90
            if distribution == "compositional_ood":
                winner_limit -= 0.05
            failures.extend(
                _minimum_failure(
                    metrics.oracle_winner_accuracy_by_candidate_distribution.get(key),
                    name=f"oracle_winner_accuracy/{key}",
                    limit=winner_limit,
                )
            )
    expected_candidate_counts = {"K8", "K32"}
    if set(metrics.normalized_regret_median_by_candidate_count) != expected_candidate_counts:
        failures.append("normalized_regret_median:incomplete_candidate_groups")
    if set(metrics.normalized_regret_p95_by_candidate_count) != expected_candidate_counts:
        failures.append("normalized_regret_p95:incomplete_candidate_groups")
    for candidate_count in (8, 32):
        key = f"K{candidate_count}"
        median_limit = 0.02 if candidate_count == 8 else 0.03
        p95_limit = 0.10 if candidate_count == 8 else 0.12
        failures.extend(
            _maximum_failure(
                metrics.normalized_regret_median_by_candidate_count.get(key),
                name=f"normalized_regret_median/{key}",
                limit=median_limit,
            )
        )
        failures.extend(
            _maximum_failure(
                metrics.normalized_regret_p95_by_candidate_count.get(key),
                name=f"normalized_regret_p95/{key}",
                limit=p95_limit,
            )
        )
    failures.extend(
        _minimum_failure(
            metrics.successful_oracle_goal_success,
            name="successful_oracle_goal_success",
            limit=0.95,
        )
    )
    for name, passed in (
        ("serial_vectorized_winner_parity", invariants.serial_vectorized_winner_parity),
        ("pre_action_invariance", invariants.pre_action_invariance),
        ("exactly_once_impulse", invariants.exactly_once_impulse),
        ("action_target_isolation", invariants.action_target_isolation),
        ("conservation", invariants.conservation),
        ("batch_independence", invariants.batch_independence),
        ("source_belief_unchanged", invariants.source_belief_unchanged),
    ):
        if not passed:
            failures.append(f"{name}:false")
    for name, value, limit in (
        ("maximum_cost_difference", invariants.maximum_cost_difference, 1.0e-6),
        ("latency_k8_seconds", invariants.latency_k8_seconds, 0.10),
        ("latency_k32_seconds", invariants.latency_k32_seconds, 0.35),
    ):
        if not math.isfinite(value) or value > limit:
            failures.append(f"{name}:{value}>{limit}")
    return tuple(failures)


def promotion_gate_failures(
    integrity: PromotionIntegrityMetrics,
    resources: ResourceMetrics,
) -> tuple[str, ...]:
    failures: list[str] = []
    valid_integrity_scalars: set[str] = set()
    for name in (
        "paired_score_improvement_fraction",
        "paired_bootstrap_lower_bound",
        "maximum_critical_regression_fraction",
        "maximum_accepted_like_regression_fraction",
        "minimum_complete_gradient_retention",
    ):
        value = getattr(integrity, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            failures.append(f"{name}:nonfinite")
        elif float(value) < 0.0:
            failures.append(f"{name}:negative")
        else:
            valid_integrity_scalars.add(name)
    if (
        "paired_score_improvement_fraction" in valid_integrity_scalars
        and integrity.paired_score_improvement_fraction < 0.03
    ):
        failures.append("paired_score_improvement_fraction:<0.03")
    if (
        "paired_bootstrap_lower_bound" in valid_integrity_scalars
        and integrity.paired_bootstrap_lower_bound <= 0.0
    ):
        failures.append("paired_bootstrap_lower_bound:<=0")
    if (
        "maximum_critical_regression_fraction" in valid_integrity_scalars
        and integrity.maximum_critical_regression_fraction > 0.02
    ):
        failures.append("maximum_critical_regression_fraction:>0.02")
    if (
        "maximum_accepted_like_regression_fraction" in valid_integrity_scalars
        and integrity.maximum_accepted_like_regression_fraction > 0.02
    ):
        failures.append("maximum_accepted_like_regression_fraction:>0.02")
    for name in (
        "truth_leakage_count",
        "fabricated_target_count",
        "nonfinite_state_count",
        "rejected_optimizer_mutation_count",
    ):
        value = getattr(integrity, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            failures.append(f"{name}:invalid_count")
        elif value != 0:
            failures.append(f"{name}:nonzero")
    if (
        "minimum_complete_gradient_retention" in valid_integrity_scalars
        and integrity.minimum_complete_gradient_retention < 0.10
    ):
        failures.append("minimum_complete_gradient_retention:<0.10")
    for name, value, limit in (
        ("perception_latency_seconds", resources.perception_latency_seconds, 0.25),
        ("six_horizon_rollout_seconds", resources.six_horizon_rollout_seconds, 0.025),
        ("learned_weight_bytes", resources.learned_weight_bytes, 1_048_576),
        ("persistent_tensor_bytes", resources.persistent_tensor_bytes, 262_144),
        ("process_rss_bytes", resources.process_rss_bytes, MAXIMUM_PROCESS_RSS_BYTES),
    ):
        invalid_type = isinstance(value, bool) or not isinstance(value, Real)
        if invalid_type or not math.isfinite(float(value)) or value < 0 or value > limit:
            failures.append(f"{name}:{value}>{limit}")
    return tuple(failures)


__all__ = [
    "HORIZON_POSITION_LIMITS_M",
    "MAXIMUM_PROCESS_RSS_BYTES",
    "PhysicalCellMetrics",
    "PlanningInvariantMetrics",
    "PlanningPopulationGateMetrics",
    "PlanningSliceMetrics",
    "PromotionIntegrityMetrics",
    "ResourceMetrics",
    "SupportedScalar",
    "physical_cell_gate_failures",
    "planning_gate_failures",
    "promotion_gate_failures",
]
