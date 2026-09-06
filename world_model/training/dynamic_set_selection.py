"""Deterministic checkpoint selection and promotion metrics for 1.61.

This module keeps the lower-is-better selection rule and integrity-metric
construction small and auditable.  The scalar-row bootstrap remains only as a
diagnostic helper; governed promotion uses the raw pooled implementation in
``dynamic_set_bootstrap``.  Neither path feeds an optimizer objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from world_model.training.dynamic_set_gates import PromotionIntegrityMetrics
from world_model.training.dynamic_set_protocol import (
    SELECTION_SCORE_WEIGHTS,
    selection_score,
)
from world_model.training.qualification_core import validated_sha256

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 161_061

_QUALIFICATION_ONLY_GATE_FAILURE_PREFIXES = (
    "physical/",
    "planning/",
    "promotion/paired_score_improvement_fraction:",
    "promotion/paired_bootstrap_lower_bound:",
)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return resolved


@dataclass(frozen=True)
class PairedScoreSample:
    """One baseline/checkpoint comparison on the same frozen example."""

    example_key: str
    baseline_score: float
    candidate_score: float

    def validate(self) -> PairedScoreSample:
        if type(self.example_key) is not str or not self.example_key:
            raise ValueError("paired example_key must be a nonempty exact string")
        _finite_nonnegative(self.baseline_score, name="baseline_score")
        _finite_nonnegative(self.candidate_score, name="candidate_score")
        return self


@dataclass(frozen=True)
class AggregateScoreComparison:
    """One named critical-cell or accepted-like aggregate comparison."""

    name: str
    baseline_score: float
    candidate_score: float

    def validate(self) -> AggregateScoreComparison:
        if type(self.name) is not str or not self.name:
            raise ValueError("aggregate comparison name must be a nonempty exact string")
        _finite_nonnegative(self.baseline_score, name=f"{self.name}/baseline")
        _finite_nonnegative(self.candidate_score, name=f"{self.name}/candidate")
        return self

    @property
    def regression_fraction(self) -> float:
        self.validate()
        if float(self.baseline_score) == 0.0:
            return 0.0 if float(self.candidate_score) == 0.0 else 1.0
        return max(
            0.0,
            (float(self.candidate_score) - float(self.baseline_score)) / float(self.baseline_score),
        )


@dataclass(frozen=True)
class PairedImprovementEvidence:
    sample_count: int
    baseline_aggregate_score: float
    candidate_aggregate_score: float
    paired_score_improvement_fraction: float
    paired_bootstrap_lower_bound: float
    bootstrap_samples: int
    bootstrap_seed: int

    def validate(self) -> PairedImprovementEvidence:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise ValueError("paired sample_count must be a positive integer")
        baseline = _finite_nonnegative(
            self.baseline_aggregate_score,
            name="baseline_aggregate_score",
        )
        candidate = _finite_nonnegative(
            self.candidate_aggregate_score,
            name="candidate_aggregate_score",
        )
        if baseline <= 0.0:
            raise ValueError("aggregate paired baseline score must be positive")
        expected_improvement = (baseline - candidate) / baseline
        if (
            isinstance(self.paired_score_improvement_fraction, bool)
            or not isinstance(self.paired_score_improvement_fraction, Real)
            or not math.isfinite(float(self.paired_score_improvement_fraction))
            or float(self.paired_score_improvement_fraction) != expected_improvement
        ):
            raise ValueError("paired improvement differs from its aggregate scores")
        if (
            isinstance(self.paired_bootstrap_lower_bound, bool)
            or not isinstance(self.paired_bootstrap_lower_bound, Real)
            or not math.isfinite(float(self.paired_bootstrap_lower_bound))
        ):
            raise ValueError("paired bootstrap lower bound must be finite")
        for name, value in (
            ("bootstrap_samples", self.bootstrap_samples),
            ("bootstrap_seed", self.bootstrap_seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.bootstrap_samples < 1_000:
            raise ValueError("paired bootstrap requires at least 1,000 replicates")
        return self


@dataclass(frozen=True)
class DynamicSetCheckpointScore:
    """A validation candidate under the exact lower-is-better score schema."""

    completed_updates: int
    model_state_sha256: str
    components: Mapping[str, float]
    training_support_passed: bool
    selection_guardrails_passed: bool

    def validate(self) -> DynamicSetCheckpointScore:
        if (
            isinstance(self.completed_updates, bool)
            or not isinstance(self.completed_updates, int)
            or self.completed_updates < 0
        ):
            raise ValueError("completed_updates must be a nonnegative integer")
        validated_sha256(self.model_state_sha256, label="model_state_sha256")
        if type(self.training_support_passed) is not bool:
            raise TypeError("training_support_passed must be boolean")
        if type(self.selection_guardrails_passed) is not bool:
            raise TypeError("selection_guardrails_passed must be boolean")
        if set(self.components) != set(SELECTION_SCORE_WEIGHTS):
            raise ValueError("checkpoint score components differ from the frozen schema")
        selection_score(self.components)
        return self

    @property
    def score(self) -> float:
        self.validate()
        return selection_score(self.components)


@dataclass(frozen=True)
class DynamicSetSelectionDecision:
    incumbent: DynamicSetCheckpointScore | None
    accepted: bool
    reason: str


def selection_guardrail_failures(gate_failures: Sequence[str]) -> tuple[str, ...]:
    """Return failures that make a checkpoint unsafe for score selection.

    Absolute physical/planning quality and paired baseline-improvement failures
    remain qualification criteria.  Every other failure, including any future
    unrecognised promotion failure, fails selection closed.
    """

    unsafe: list[str] = []
    for failure in gate_failures:
        if type(failure) is not str or not failure:
            raise TypeError("gate failures must be nonempty exact strings")
        if failure == "training_support_passed:false" or failure.startswith(
            _QUALIFICATION_ONLY_GATE_FAILURE_PREFIXES
        ):
            continue
        unsafe.append(failure)
    return tuple(unsafe)


def select_dynamic_set_incumbent(
    incumbent: DynamicSetCheckpointScore | None,
    candidate: DynamicSetCheckpointScore,
) -> DynamicSetSelectionDecision:
    """Select only a supported strict score improvement; ties retain safety."""

    candidate.validate()
    if incumbent is not None:
        incumbent.validate()
        if not incumbent.training_support_passed or not incumbent.selection_guardrails_passed:
            raise ValueError(
                "an incumbent must have complete support and pass selection guardrails"
            )
    if not candidate.training_support_passed:
        if incumbent is None:
            return DynamicSetSelectionDecision(
                incumbent=None,
                accepted=False,
                reason="candidate precedes the minimum supported update count",
            )
        return DynamicSetSelectionDecision(
            incumbent=incumbent,
            accepted=False,
            reason="candidate lacks complete causal/gradient training support",
        )
    if not candidate.selection_guardrails_passed:
        return DynamicSetSelectionDecision(
            incumbent=incumbent,
            accepted=False,
            reason="candidate fails selection safety guardrails",
        )
    if incumbent is None:
        return DynamicSetSelectionDecision(
            incumbent=candidate,
            accepted=True,
            reason="first completely supported validation candidate",
        )
    if candidate.score < incumbent.score:
        return DynamicSetSelectionDecision(
            incumbent=candidate,
            accepted=True,
            reason="candidate strictly improves the frozen lower-is-better score",
        )
    return DynamicSetSelectionDecision(
        incumbent=incumbent,
        accepted=False,
        reason="candidate does not strictly improve the frozen lower-is-better score",
    )


def paired_improvement_evidence(
    samples: Sequence[PairedScoreSample],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedImprovementEvidence:
    """Compute a deterministic one-sided 95% paired bootstrap lower bound.

    Each bootstrap replicate resamples complete baseline/candidate pairs and
    computes its relative mean difference.  A replicate containing only
    legitimately perfect zero-baseline rows contributes zero when its paired
    candidate mean is also zero; a regression from such rows is scaled by the
    positive observed aggregate baseline.  The lower bound is the nearest-rank
    fifth percentile.
    """

    resolved = tuple(sample.validate() for sample in samples)
    if not resolved:
        raise ValueError("paired improvement requires at least one sample")
    keys = tuple(sample.example_key for sample in resolved)
    if len(set(keys)) != len(keys):
        raise ValueError("paired example keys must be unique")
    for name, value in (
        ("bootstrap_samples", bootstrap_samples),
        ("bootstrap_seed", bootstrap_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if bootstrap_samples < 1_000:
        raise ValueError("paired bootstrap requires at least 1,000 replicates")

    baseline = np.asarray([sample.baseline_score for sample in resolved], dtype=np.float64)
    candidate = np.asarray([sample.candidate_score for sample in resolved], dtype=np.float64)
    baseline_mean = float(baseline.mean())
    candidate_mean = float(candidate.mean())
    if baseline_mean <= 0.0:
        raise ValueError("aggregate paired baseline score must be positive")
    improvement = (baseline_mean - candidate_mean) / baseline_mean

    generator = np.random.default_rng(bootstrap_seed)
    replicate_values = np.empty(bootstrap_samples, dtype=np.float64)
    # Bound working memory for the full 4,400-example development population.
    batch_size = max(1, min(128, 2_000_000 // len(resolved)))
    for start in range(0, bootstrap_samples, batch_size):
        count = min(batch_size, bootstrap_samples - start)
        indices = generator.integers(
            0,
            len(resolved),
            size=(count, len(resolved)),
            endpoint=False,
        )
        baseline_means = baseline[indices].mean(axis=1)
        candidate_means = candidate[indices].mean(axis=1)
        differences = baseline_means - candidate_means
        values = np.divide(
            differences,
            baseline_means,
            out=differences / baseline_mean,
            where=baseline_means > 0.0,
        )
        replicate_values[start : start + count] = values
    replicate_values.sort()
    lower_index = max(0, math.ceil(0.05 * bootstrap_samples) - 1)
    lower_bound = float(replicate_values[lower_index])
    if not math.isfinite(improvement) or not math.isfinite(lower_bound):
        raise FloatingPointError("paired improvement evidence is nonfinite")
    return PairedImprovementEvidence(
        sample_count=len(resolved),
        baseline_aggregate_score=baseline_mean,
        candidate_aggregate_score=candidate_mean,
        paired_score_improvement_fraction=improvement,
        paired_bootstrap_lower_bound=lower_bound,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    ).validate()


def _maximum_regression(comparisons: Sequence[AggregateScoreComparison], *, label: str) -> float:
    resolved = tuple(comparison.validate() for comparison in comparisons)
    if not resolved:
        raise ValueError(f"{label} comparisons must be nonempty")
    names = tuple(comparison.name for comparison in resolved)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} comparison names must be unique")
    return max(comparison.regression_fraction for comparison in resolved)


def build_promotion_integrity_metrics_from_paired_evidence(
    paired: PairedImprovementEvidence,
    *,
    critical_comparisons: Sequence[AggregateScoreComparison],
    accepted_like_comparisons: Sequence[AggregateScoreComparison],
    truth_leakage_count: int,
    fabricated_target_count: int,
    nonfinite_state_count: int,
    rejected_optimizer_mutation_count: int,
    minimum_complete_gradient_retention: float,
) -> PromotionIntegrityMetrics:
    """Build promotion metrics from an already validated paired population."""

    if type(paired) is not PairedImprovementEvidence:
        raise TypeError("paired evidence must be exact PairedImprovementEvidence")
    paired.validate()
    for name, value in (
        ("truth_leakage_count", truth_leakage_count),
        ("fabricated_target_count", fabricated_target_count),
        ("nonfinite_state_count", nonfinite_state_count),
        ("rejected_optimizer_mutation_count", rejected_optimizer_mutation_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    retention = _finite_nonnegative(
        minimum_complete_gradient_retention,
        name="minimum_complete_gradient_retention",
    )
    if retention > 1.0:
        raise ValueError("minimum_complete_gradient_retention cannot exceed one")
    return PromotionIntegrityMetrics(
        paired_score_improvement_fraction=paired.paired_score_improvement_fraction,
        paired_bootstrap_lower_bound=paired.paired_bootstrap_lower_bound,
        maximum_critical_regression_fraction=_maximum_regression(
            critical_comparisons,
            label="critical",
        ),
        maximum_accepted_like_regression_fraction=_maximum_regression(
            accepted_like_comparisons,
            label="accepted_like",
        ),
        truth_leakage_count=truth_leakage_count,
        fabricated_target_count=fabricated_target_count,
        nonfinite_state_count=nonfinite_state_count,
        rejected_optimizer_mutation_count=rejected_optimizer_mutation_count,
        minimum_complete_gradient_retention=retention,
    )


def build_promotion_integrity_metrics(
    paired_samples: Sequence[PairedScoreSample],
    *,
    critical_comparisons: Sequence[AggregateScoreComparison],
    accepted_like_comparisons: Sequence[AggregateScoreComparison],
    truth_leakage_count: int,
    fabricated_target_count: int,
    nonfinite_state_count: int,
    rejected_optimizer_mutation_count: int,
    minimum_complete_gradient_retention: float,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PromotionIntegrityMetrics:
    """Diagnostic scalar-row bootstrap retained for local component tests.

    Governed specification-1.61 promotion does not call this helper.  It must
    resample raw additive physical evidence and planning-task pairs via
    :mod:`world_model.training.dynamic_set_bootstrap` before constructing the
    integrity metrics.
    """

    paired = paired_improvement_evidence(
        paired_samples,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return build_promotion_integrity_metrics_from_paired_evidence(
        paired,
        critical_comparisons=critical_comparisons,
        accepted_like_comparisons=accepted_like_comparisons,
        truth_leakage_count=truth_leakage_count,
        fabricated_target_count=fabricated_target_count,
        nonfinite_state_count=nonfinite_state_count,
        rejected_optimizer_mutation_count=rejected_optimizer_mutation_count,
        minimum_complete_gradient_retention=minimum_complete_gradient_retention,
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "AggregateScoreComparison",
    "DynamicSetCheckpointScore",
    "DynamicSetSelectionDecision",
    "PairedImprovementEvidence",
    "PairedScoreSample",
    "build_promotion_integrity_metrics",
    "build_promotion_integrity_metrics_from_paired_evidence",
    "paired_improvement_evidence",
    "selection_guardrail_failures",
    "select_dynamic_set_incumbent",
]
