from __future__ import annotations

from dataclasses import replace

import pytest

from world_model.training.dynamic_set_gates import promotion_gate_failures
from world_model.training.dynamic_set_protocol import SELECTION_SCORE_WEIGHTS
from world_model.training.dynamic_set_selection import (
    AggregateScoreComparison,
    DynamicSetCheckpointScore,
    PairedScoreSample,
    build_promotion_integrity_metrics,
    paired_improvement_evidence,
    select_dynamic_set_incumbent,
    selection_guardrail_failures,
)


def _candidate(
    step: int,
    value: float,
    *,
    supported: bool = True,
    guardrails_passed: bool = True,
) -> DynamicSetCheckpointScore:
    return DynamicSetCheckpointScore(
        completed_updates=step,
        model_state_sha256=f"{step % 16:x}" * 64,
        components={name: value for name in SELECTION_SCORE_WEIGHTS},
        training_support_passed=supported,
        selection_guardrails_passed=guardrails_passed,
    )


def test_checkpoint_selection_is_strict_lower_is_better_and_support_gated() -> None:
    first = _candidate(512, 0.8)
    assert select_dynamic_set_incumbent(None, first).incumbent is first
    worse = _candidate(1_024, 0.9)
    rejected = select_dynamic_set_incumbent(first, worse)
    assert not rejected.accepted
    assert rejected.incumbent is first
    better = _candidate(1_536, 0.7)
    accepted = select_dynamic_set_incumbent(first, better)
    assert accepted.accepted
    assert accepted.incumbent is better
    unsupported = _candidate(2_048, 0.1, supported=False)
    assert not select_dynamic_set_incumbent(better, unsupported).accepted
    preminimum = select_dynamic_set_incumbent(None, unsupported)
    assert not preminimum.accepted
    assert preminimum.incumbent is None
    assert "minimum supported update count" in preminimum.reason


def test_checkpoint_selection_rejects_lower_scoring_guardrail_failure() -> None:
    incumbent = _candidate(512, 0.8)
    unsafe = _candidate(1_024, 0.1, guardrails_passed=False)

    decision = select_dynamic_set_incumbent(incumbent, unsafe)

    assert decision.accepted is False
    assert decision.incumbent is incumbent
    assert "safety guardrails" in decision.reason


def test_selection_guardrails_exclude_absolute_quality_and_improvement_gates() -> None:
    assert (
        selection_guardrail_failures(
            (
                "physical/N6/contact=1/dynamic=1/current_position_rmse_m:0.02>0.01",
                "planning/K8/winner_accuracy:0.9<0.95",
                "promotion/paired_score_improvement_fraction:<0.03",
                "promotion/paired_bootstrap_lower_bound:<=0",
            )
        )
        == ()
    )
    assert selection_guardrail_failures(
        (
            "promotion/maximum_critical_regression_fraction:>0.02",
            "promotion/process_rss_bytes:1>0",
            "promotion/future_unknown_guardrail:failed",
        )
    ) == (
        "promotion/maximum_critical_regression_fraction:>0.02",
        "promotion/process_rss_bytes:1>0",
        "promotion/future_unknown_guardrail:failed",
    )


def test_checkpoint_score_rejects_schema_and_nonfinite_components() -> None:
    with pytest.raises(ValueError, match="schema"):
        replace(_candidate(512, 0.5), components={"planning_error": 0.1}).validate()
    values = {name: 0.1 for name in SELECTION_SCORE_WEIGHTS}
    values["planning_error"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        replace(_candidate(512, 0.5), components=values).validate()


def test_paired_bootstrap_is_deterministic_and_preserves_complete_pairs() -> None:
    samples = tuple(
        PairedScoreSample(
            example_key=f"development/{index}",
            baseline_score=1.0 + index / 1_000.0,
            candidate_score=(1.0 + index / 1_000.0) * 0.95,
        )
        for index in range(100)
    )
    first = paired_improvement_evidence(samples, bootstrap_samples=1_000, bootstrap_seed=7)
    second = paired_improvement_evidence(samples, bootstrap_samples=1_000, bootstrap_seed=7)
    assert first == second
    assert first.sample_count == 100
    assert first.paired_score_improvement_fraction == pytest.approx(0.05)
    assert first.paired_bootstrap_lower_bound == pytest.approx(0.05)

    with pytest.raises(ValueError, match="unique"):
        paired_improvement_evidence(
            (samples[0], samples[0]),
            bootstrap_samples=1_000,
        )
    with pytest.raises(ValueError, match="aggregate paired baseline score"):
        paired_improvement_evidence(
            (PairedScoreSample("zero", 0.0, 0.0),),
            bootstrap_samples=1_000,
        )
    mixed = paired_improvement_evidence(
        (
            PairedScoreSample("perfect", 0.0, 0.0),
            PairedScoreSample("improved", 1.0, 0.9),
        ),
        bootstrap_samples=1_000,
    )
    assert mixed.paired_score_improvement_fraction == pytest.approx(0.1)


def test_promotion_integrity_is_derived_from_paired_and_aggregate_evidence() -> None:
    paired = tuple(PairedScoreSample(f"example/{index}", 1.0, 0.96) for index in range(40))
    integrity = build_promotion_integrity_metrics(
        paired,
        critical_comparisons=(AggregateScoreComparison("N6/contact", 1.0, 1.019),),
        accepted_like_comparisons=(AggregateScoreComparison("legacy/N2", 1.0, 1.01),),
        truth_leakage_count=0,
        fabricated_target_count=0,
        nonfinite_state_count=0,
        rejected_optimizer_mutation_count=0,
        minimum_complete_gradient_retention=0.10,
        bootstrap_samples=1_000,
        bootstrap_seed=11,
    )
    assert integrity.paired_score_improvement_fraction == pytest.approx(0.04)
    assert integrity.paired_bootstrap_lower_bound > 0.0
    assert integrity.maximum_critical_regression_fraction == pytest.approx(0.019)
    assert integrity.maximum_accepted_like_regression_fraction == pytest.approx(0.01)

    # Resource checks are orthogonal; an empty failure prefix here establishes
    # that every integrity value reaches the existing absolute gate correctly.
    from world_model.training.dynamic_set_gates import ResourceMetrics

    resources = ResourceMetrics(0.1, 0.01, 100, 100, 100)
    assert promotion_gate_failures(integrity, resources) == ()


def test_regression_evidence_cannot_be_vacuous_or_malformed() -> None:
    paired = (PairedScoreSample("example", 1.0, 0.9),)
    with pytest.raises(ValueError, match="critical comparisons"):
        build_promotion_integrity_metrics(
            paired,
            critical_comparisons=(),
            accepted_like_comparisons=(AggregateScoreComparison("legacy", 1.0, 1.0),),
            truth_leakage_count=0,
            fabricated_target_count=0,
            nonfinite_state_count=0,
            rejected_optimizer_mutation_count=0,
            minimum_complete_gradient_retention=0.1,
            bootstrap_samples=1_000,
        )

    perfect = AggregateScoreComparison("perfect", 0.0, 0.0)
    regressed = AggregateScoreComparison("regressed", 0.0, 0.01)
    assert perfect.regression_fraction == 0.0
    assert regressed.regression_fraction == 1.0
    with pytest.raises(ValueError, match="cannot exceed one"):
        build_promotion_integrity_metrics(
            paired,
            critical_comparisons=(AggregateScoreComparison("cell", 1.0, 1.0),),
            accepted_like_comparisons=(AggregateScoreComparison("legacy", 1.0, 1.0),),
            truth_leakage_count=0,
            fabricated_target_count=0,
            nonfinite_state_count=0,
            rejected_optimizer_mutation_count=0,
            minimum_complete_gradient_retention=1.1,
            bootstrap_samples=1_000,
        )
