from __future__ import annotations

from dataclasses import replace

import pytest

from world_model.training.dynamic_set_bootstrap import (
    PlanningTaskScoreEvidence,
    _pack_physical_statistics,
    _physical_scores_from_sums,
    pooled_paired_improvement_evidence,
)
from world_model.training.dynamic_set_evaluation import (
    HORIZONS_SECONDS,
    BinaryCounts,
    DynamicSetCellAccumulator,
    DynamicSetExampleScoreEvidence,
    RatioCounts,
    SquaredErrorSum,
    SumCount,
    pooled_selection_score_evidence,
)
from world_model.training.dynamic_set_planning import (
    PlanningTaskEvaluation,
    PlanningTaskOutcome,
)
from world_model.training.dynamic_set_planning_materializer import planning_error
from world_model.training.dynamic_set_protocol import (
    SELECTION_SCORE_WEIGHTS,
    physical_manifest,
    planning_manifest,
    selection_score,
)


def _error(normalized_rmse: float, scale: float, support: int) -> SquaredErrorSum:
    return SquaredErrorSum(
        squared_error=(normalized_rmse * scale) ** 2 * support,
        coordinate_count=support,
    )


def _accumulator(
    *, normalized_error: float, support: int, mistakes: int
) -> DynamicSetCellAccumulator:
    return DynamicSetCellAccumulator(
        episode_count=1,
        proposal=BinaryCounts(80, mistakes, mistakes),
        exact_count=RatioCounts(10 - mistakes, 10),
        current_position=_error(normalized_error, 0.010, support),
        mature_velocity=_error(normalized_error, 0.020, support),
        post_event_velocity=_error(normalized_error, 0.050, support),
        horizon_position={
            horizon: _error(normalized_error, limit, support)
            for horizon, limit in (
                (0.05, 0.012),
                (0.10, 0.014),
                (0.25, 0.018),
                (0.50, 0.025),
                (1.00, 0.035),
                (2.00, 0.050),
            )
        },
        horizon_velocity={
            horizon: _error(normalized_error, 0.020, support) for horizon in HORIZONS_SECONDS
        },
        collision=BinaryCounts(20, mistakes, mistakes),
        collision_timing=SumCount(float(mistakes), 20),
        persistent_identity=RatioCounts(100 - mistakes, 100),
        identity_switch=RatioCounts(mistakes, 100),
        birth=BinaryCounts(20, mistakes, mistakes),
        birth_latency=SumCount(float(mistakes), 20),
        removal=BinaryCounts(20, mistakes, mistakes),
        removal_latency=SumCount(float(2 * mistakes), 20),
        uncertainty_90=RatioCounts(90, 100),
    )


def _physical_population(
    *,
    normalized_errors: tuple[float, ...],
    supports: tuple[int, ...],
    mistakes: int,
) -> tuple[DynamicSetExampleScoreEvidence, ...]:
    rows = physical_manifest("development")[: len(normalized_errors)]
    return tuple(
        DynamicSetExampleScoreEvidence.create(
            row,
            _accumulator(
                normalized_error=normalized_error,
                support=support,
                mistakes=mistakes,
            ),
        )
        for row, normalized_error, support in zip(
            rows,
            normalized_errors,
            supports,
            strict=True,
        )
    )


def _successful_outcome(row: object) -> PlanningTaskOutcome:
    evaluation = PlanningTaskEvaluation(
        row=row,
        template_sha256="a" * 64,
        binding_sha256="b" * 64,
        private_oracle_sha256="c" * 64,
        model_winner_index=0,
        serial_winner_index=0,
        oracle_winner_index=0,
        oracle_winner_correct=True,
        normalized_regret=0.0,
        oracle_winner_succeeds=True,
        selected_action_goal_success=True,
        serial_vectorized_winner_parity=True,
        maximum_cost_difference=0.0,
        cost_agreement_within_tolerance=True,
        active_set_frozen=True,
        source_belief_unchanged=True,
    )
    return PlanningTaskOutcome.evaluated(evaluation)


def _planning_populations() -> tuple[
    tuple[PlanningTaskScoreEvidence, ...],
    tuple[PlanningTaskScoreEvidence, ...],
]:
    rows = planning_manifest("development")[:4]
    candidate = tuple(PlanningTaskScoreEvidence.create(_successful_outcome(row)) for row in rows)
    reference = tuple(
        PlanningTaskScoreEvidence.create(
            PlanningTaskOutcome.unresolved(row, reason="baseline handle failure")
        )
        for row in rows
    )
    return candidate, reference


def test_pooled_bootstrap_point_is_exact_full_selection_score_and_deterministic() -> None:
    candidate_physical = _physical_population(
        normalized_errors=(0.20, 0.45, 0.30, 0.40),
        supports=(1, 80, 3, 40),
        mistakes=0,
    )
    reference_physical = _physical_population(
        normalized_errors=(0.90, 0.70, 0.80, 0.60),
        supports=(1, 80, 3, 40),
        mistakes=2,
    )
    candidate_planning, reference_planning = _planning_populations()

    result = pooled_paired_improvement_evidence(
        candidate_physical,
        reference_physical,
        candidate_planning,
        reference_planning,
        bootstrap_samples=1_000,
        bootstrap_seed=19,
    )
    repeated = pooled_paired_improvement_evidence(
        candidate_physical,
        reference_physical,
        candidate_planning,
        reference_planning,
        bootstrap_samples=1_000,
        bootstrap_seed=19,
    )
    candidate_components = {
        **pooled_selection_score_evidence(candidate_physical).components,
        "planning_error": planning_error(tuple(record.outcome for record in candidate_planning)),
    }
    reference_components = {
        **pooled_selection_score_evidence(reference_physical).components,
        "planning_error": planning_error(tuple(record.outcome for record in reference_planning)),
    }
    expected_candidate = selection_score(candidate_components)
    expected_reference = selection_score(reference_components)
    assert result == repeated
    assert result.candidate_aggregate_score == expected_candidate
    assert result.baseline_aggregate_score == expected_reference
    assert (
        result.paired_score_improvement_fraction
        == (expected_reference - expected_candidate) / expected_reference
    )
    assert result.paired_bootstrap_lower_bound > 0.0

    # Heterogeneous supports make the pooled RMSE observably different from a
    # mean of per-row RMSEs; the governed result above follows the former.
    row_mean = sum(
        dict(item.supported_components)["current_position"] for item in reference_physical
    ) / len(reference_physical)
    pooled = pooled_selection_score_evidence(reference_physical).components["current_position"]
    assert pooled != pytest.approx(row_mean)


def test_vectorized_physical_reduction_matches_canonical_pooled_reduction() -> None:
    physical = _physical_population(
        normalized_errors=(0.15, 0.80, 0.35),
        supports=(2, 70, 5),
        mistakes=1,
    )
    resample = (physical[0], physical[1], physical[1], physical[2], physical[2])
    packed = _pack_physical_statistics(physical)
    vectorized = _physical_scores_from_sums(packed[[0, 1, 1, 2, 2]].sum(axis=0, keepdims=True))[0]
    components = pooled_selection_score_evidence(resample).components
    canonical = sum(SELECTION_SCORE_WEIGHTS[name] * components[name] for name in components)
    assert vectorized == pytest.approx(canonical, abs=1.0e-14)


def test_planning_is_an_independent_fixed_fifteen_percent_stratum() -> None:
    physical = _physical_population(
        normalized_errors=(0.50, 0.50),
        supports=(10, 10),
        mistakes=1,
    )
    candidate_planning, reference_planning = _planning_populations()
    result = pooled_paired_improvement_evidence(
        physical,
        physical,
        candidate_planning,
        reference_planning,
        bootstrap_samples=1_000,
        bootstrap_seed=23,
    )
    assert result.baseline_aggregate_score - result.candidate_aggregate_score == pytest.approx(0.15)


def test_pooled_bootstrap_rejects_reordering_and_digest_tampering() -> None:
    candidate_physical = _physical_population(
        normalized_errors=(0.20, 0.30),
        supports=(5, 7),
        mistakes=0,
    )
    reference_physical = _physical_population(
        normalized_errors=(0.50, 0.60),
        supports=(5, 7),
        mistakes=1,
    )
    candidate_planning, reference_planning = _planning_populations()
    with pytest.raises(ValueError, match="row bindings differ"):
        pooled_paired_improvement_evidence(
            candidate_physical,
            tuple(reversed(reference_physical)),
            candidate_planning,
            reference_planning,
            bootstrap_samples=1_000,
        )
    with pytest.raises(ValueError, match="planning task bindings differ"):
        pooled_paired_improvement_evidence(
            candidate_physical,
            reference_physical,
            candidate_planning,
            tuple(reversed(reference_planning)),
            bootstrap_samples=1_000,
        )
    tampered_planning = (
        replace(candidate_planning[0], task_error=0.5),
        *candidate_planning[1:],
    )
    with pytest.raises(ValueError, match="task error differs"):
        pooled_paired_improvement_evidence(
            candidate_physical,
            reference_physical,
            tampered_planning,
            reference_planning,
            bootstrap_samples=1_000,
        )
    tampered = (
        replace(candidate_physical[0], evidence_sha256="0" * 64),
        *candidate_physical[1:],
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        pooled_paired_improvement_evidence(
            tampered,
            reference_physical,
            candidate_planning,
            reference_planning,
            bootstrap_samples=1_000,
        )
