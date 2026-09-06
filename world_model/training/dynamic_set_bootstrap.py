"""Pooled paired bootstrap for governed dynamic-set promotion.

Physical selection components are nonlinear functions of additive sufficient
statistics (for example, a pooled RMSE and pooled F1).  Consequently, averaging
per-row component scores is not the frozen population score.  This module
resamples complete candidate/reference row pairs, pools their raw additive
statistics in NumPy batches, and only then evaluates the physical score.  The
planning population is an independent fixed stratum whose complete task pairs
are resampled separately.  Every replicate retains the protocol's exact
85 percent physical / 15 percent planning weighting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from world_model.training.dynamic_set_evaluation import (
    HORIZON_VELOCITY_SCORE_SCALE_MPS,
    HORIZONS_SECONDS,
    DynamicSetExampleScoreEvidence,
    pair_dynamic_set_example_score_evidence,
    pooled_selection_score_evidence,
)
from world_model.training.dynamic_set_gates import HORIZON_POSITION_LIMITS_M
from world_model.training.dynamic_set_planning import (
    PlanningTaskEvaluation,
    PlanningTaskOutcome,
    reduce_planning_task_outcomes,
    validate_planning_manifest_row,
)
from world_model.training.dynamic_set_planning_materializer import planning_error
from world_model.training.dynamic_set_protocol import (
    SELECTION_SCORE_WEIGHTS,
    PlanningManifestRow,
    selection_score,
)
from world_model.training.dynamic_set_selection import PairedImprovementEvidence
from world_model.training.qualification_core import canonical_sha256, validated_sha256

_PHYSICAL_COMPONENTS = frozenset(SELECTION_SCORE_WEIGHTS) - {"planning_error"}
_PLANNING_WEIGHT = float(SELECTION_SCORE_WEIGHTS["planning_error"])
_PHYSICAL_WEIGHT = math.fsum(
    float(weight) for name, weight in SELECTION_SCORE_WEIGHTS.items() if name != "planning_error"
)
_PACKED_STATISTIC_COUNT = 32
_MAXIMUM_GATHER_VALUES = 8_000_000


@dataclass(frozen=True, slots=True)
class PlanningTaskScoreEvidence:
    """Digest-bound complete planning outcome used by the paired bootstrap."""

    row: PlanningManifestRow
    row_sha256: str
    outcome: PlanningTaskOutcome
    task_error: float
    evidence_sha256: str

    @classmethod
    def create(cls, outcome: PlanningTaskOutcome) -> PlanningTaskScoreEvidence:
        _validate_planning_outcome(outcome)
        row_sha256 = canonical_sha256(asdict(outcome.row))
        task_error = planning_error((outcome,))
        unsigned = {
            "row": asdict(outcome.row),
            "row_sha256": row_sha256,
            "outcome": asdict(outcome),
            "task_error": task_error,
        }
        return cls(
            row=outcome.row,
            row_sha256=row_sha256,
            outcome=outcome,
            task_error=task_error,
            evidence_sha256=canonical_sha256(unsigned),
        )

    @property
    def pair_key(self) -> tuple[str, int, int, str]:
        return (self.row.split, self.row.ordinal, self.row.seed, self.row_sha256)

    def validate(self) -> PlanningTaskScoreEvidence:
        validated_sha256(self.row_sha256, label="planning row")
        validated_sha256(self.evidence_sha256, label="planning task evidence")
        _validate_planning_outcome(self.outcome)
        if self.outcome.row != self.row or canonical_sha256(asdict(self.row)) != self.row_sha256:
            raise ValueError("planning task evidence row binding differs")
        expected_error = planning_error((self.outcome,))
        if (
            isinstance(self.task_error, bool)
            or not isinstance(self.task_error, (int, float))
            or not math.isfinite(float(self.task_error))
            or float(self.task_error) < 0.0
            or float(self.task_error) != expected_error
        ):
            raise ValueError("planning task error differs from its complete outcome")
        unsigned = {
            "row": asdict(self.row),
            "row_sha256": self.row_sha256,
            "outcome": asdict(self.outcome),
            "task_error": self.task_error,
        }
        if canonical_sha256(unsigned) != self.evidence_sha256:
            raise ValueError("planning task score evidence digest mismatch")
        return self


def _validate_bootstrap_parameters(samples: int, seed: int) -> None:
    for name, value in (("bootstrap_samples", samples), ("bootstrap_seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if samples < 1_000:
        raise ValueError("paired bootstrap requires at least 1,000 replicates")


def _validated_physical_pairs(
    candidate: Sequence[DynamicSetExampleScoreEvidence],
    reference: Sequence[DynamicSetExampleScoreEvidence],
) -> tuple[
    tuple[DynamicSetExampleScoreEvidence, ...],
    tuple[DynamicSetExampleScoreEvidence, ...],
]:
    candidate_records = tuple(candidate)
    reference_records = tuple(reference)
    if any(type(record) is not DynamicSetExampleScoreEvidence for record in candidate_records):
        raise TypeError("candidate physical evidence must use exact row evidence")
    if any(type(record) is not DynamicSetExampleScoreEvidence for record in reference_records):
        raise TypeError("reference physical evidence must use exact row evidence")
    # This validates every digest, rebuilds every supported-component tuple,
    # rejects duplicate bindings, and enforces same-order complete pairs.
    pair_dynamic_set_example_score_evidence(candidate_records, reference_records)
    return candidate_records, reference_records


def _validate_planning_outcome(outcome: PlanningTaskOutcome) -> None:
    if type(outcome) is not PlanningTaskOutcome:
        raise TypeError("planning evidence must use exact PlanningTaskOutcome values")
    if type(outcome.handle_resolved) is not bool:
        raise TypeError("planning handle_resolved must be an exact bool")
    validate_planning_manifest_row(outcome.row)
    if outcome.handle_resolved:
        if type(outcome.evaluation) is not PlanningTaskEvaluation:
            raise ValueError("resolved planning evidence requires an exact evaluation")
        evaluation = outcome.evaluation
        if evaluation.row != outcome.row or outcome.failure_reason is not None:
            raise ValueError("resolved planning evidence is not bound to its task row")
        validated_sha256(evaluation.template_sha256, label="planning template")
        validated_sha256(evaluation.binding_sha256, label="planning binding")
        validated_sha256(evaluation.private_oracle_sha256, label="planning private oracle")
        for name in ("model_winner_index", "serial_winner_index", "oracle_winner_index"):
            index = getattr(evaluation, name)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < outcome.row.candidate_count
            ):
                raise ValueError(f"planning {name} is outside the candidate bank")
        for name in (
            "oracle_winner_correct",
            "oracle_winner_succeeds",
            "selected_action_goal_success",
            "serial_vectorized_winner_parity",
            "cost_agreement_within_tolerance",
            "active_set_frozen",
            "source_belief_unchanged",
        ):
            if type(getattr(evaluation, name)) is not bool:
                raise TypeError(f"planning {name} must be an exact bool")
        if evaluation.oracle_winner_correct != (
            evaluation.model_winner_index == evaluation.oracle_winner_index
        ):
            raise ValueError("planning winner correctness differs from its indices")
        if evaluation.serial_vectorized_winner_parity != (
            evaluation.model_winner_index == evaluation.serial_winner_index
        ):
            raise ValueError("planning serial/vectorized parity differs from its indices")
        for name in ("normalized_regret", "maximum_cost_difference"):
            metric = getattr(evaluation, name)
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
                or float(metric) < 0.0
            ):
                raise ValueError(f"planning {name} must be finite and nonnegative")
    elif (
        outcome.evaluation is not None
        or type(outcome.failure_reason) is not str
        or not outcome.failure_reason
    ):
        raise ValueError("unresolved planning evidence must contain only a failure reason")


def _validated_planning_pairs(
    candidate: Sequence[PlanningTaskScoreEvidence],
    reference: Sequence[PlanningTaskScoreEvidence],
) -> tuple[
    tuple[PlanningTaskScoreEvidence, ...],
    tuple[PlanningTaskScoreEvidence, ...],
]:
    candidate_records = tuple(candidate)
    reference_records = tuple(reference)
    if len(candidate_records) != len(reference_records) or not candidate_records:
        raise ValueError("candidate/reference planning evidence lengths differ or are empty")
    seen: set[tuple[str, int, int]] = set()
    for candidate_record, reference_record in zip(
        candidate_records,
        reference_records,
        strict=True,
    ):
        if (
            type(candidate_record) is not PlanningTaskScoreEvidence
            or type(reference_record) is not PlanningTaskScoreEvidence
        ):
            raise TypeError("planning pairs must use exact task score evidence")
        candidate_record.validate()
        reference_record.validate()
        if candidate_record.pair_key != reference_record.pair_key:
            raise ValueError("candidate/reference planning task bindings differ")
        row = candidate_record.row
        key = (row.split, row.ordinal, row.seed)
        if key in seen:
            raise ValueError("paired planning evidence contains a duplicate task binding")
        seen.add(key)
    # Reuse the public reducer's full manifest/outcome consistency checks.  Its
    # output is deliberately discarded: the bootstrap consumes only raw tasks.
    reduce_planning_task_outcomes(tuple(record.outcome for record in candidate_records))
    reduce_planning_task_outcomes(tuple(record.outcome for record in reference_records))
    return candidate_records, reference_records


def _pack_physical_statistics(
    records: Sequence[DynamicSetExampleScoreEvidence],
) -> np.ndarray:
    """Pack only the additive terms needed by the seven physical components."""

    packed = np.empty((len(records), _PACKED_STATISTIC_COUNT), dtype=np.float64)
    for index, record in enumerate(records):
        value = record.additive
        horizon_position = {
            horizon: (error, count) for horizon, error, count in value.horizon_position
        }
        horizon_velocity = {
            horizon: (error, count) for horizon, error, count in value.horizon_velocity
        }
        packed[index] = (
            value.current_position[0] / (0.010**2),
            value.current_position[1],
            value.mature_velocity[0] / (0.020**2) + value.post_event_velocity[0] / (0.050**2),
            value.mature_velocity[1] + value.post_event_velocity[1],
            math.fsum(
                horizon_position[horizon][0] / (HORIZON_POSITION_LIMITS_M[horizon] ** 2)
                for horizon in HORIZONS_SECONDS
            ),
            sum(horizon_position[horizon][1] for horizon in HORIZONS_SECONDS),
            math.fsum(
                horizon_velocity[horizon][0] / (HORIZON_VELOCITY_SCORE_SCALE_MPS**2)
                for horizon in HORIZONS_SECONDS
            ),
            sum(horizon_velocity[horizon][1] for horizon in HORIZONS_SECONDS),
            *value.proposal,
            *value.exact_count,
            *value.persistent_identity,
            *value.identity_switch,
            *value.birth,
            *value.removal,
            *value.birth_latency,
            *value.removal_latency,
            *value.collision,
            *value.collision_timing,
        )
    if packed.shape[1] != _PACKED_STATISTIC_COUNT or not np.isfinite(packed).all():
        raise ValueError("packed physical evidence is malformed or nonfinite")
    if np.any(packed < 0.0):
        raise ValueError("packed physical evidence must be nonnegative")
    return packed


def _ratio(numerator: np.ndarray, denominator: np.ndarray, *, label: str) -> np.ndarray:
    if np.any(denominator <= 0.0):
        raise ValueError(f"bootstrap replicate lacks {label} support")
    return numerator / denominator


def _f1(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    denominator = 2.0 * true_positive + false_positive + false_negative
    return _ratio(2.0 * true_positive, denominator, label=label)


def _physical_scores_from_sums(values: np.ndarray) -> np.ndarray:
    """Evaluate the exact seven-component weighted score on pooled rows."""

    if values.ndim != 2 or values.shape[1] != _PACKED_STATISTIC_COUNT:
        raise ValueError("pooled physical statistic matrix has the wrong shape")
    current_position = np.sqrt(_ratio(values[:, 0], values[:, 1], label="position"))
    current_velocity = np.sqrt(_ratio(values[:, 2], values[:, 3], label="velocity"))
    horizon_position = np.sqrt(_ratio(values[:, 4], values[:, 5], label="horizon-position"))
    horizon_velocity = np.sqrt(_ratio(values[:, 6], values[:, 7], label="horizon-velocity"))
    proposal_f1 = _f1(
        values[:, 8],
        values[:, 9],
        values[:, 10],
        label="proposal-F1",
    )
    exact_count = _ratio(values[:, 11], values[:, 12], label="exact-count")
    proposal_error = ((1.0 - proposal_f1) + (1.0 - exact_count)) / 2.0

    persistent_identity = _ratio(values[:, 13], values[:, 14], label="identity")
    identity_switch = _ratio(values[:, 15], values[:, 16], label="identity-switch")
    birth_f1 = _f1(values[:, 17], values[:, 18], values[:, 19], label="birth-F1")
    removal_f1 = _f1(values[:, 20], values[:, 21], values[:, 22], label="removal-F1")
    birth_latency = _ratio(values[:, 23], values[:, 24], label="birth-latency")
    removal_latency = _ratio(values[:, 25], values[:, 26], label="removal-latency")
    identity_lifecycle_error = (
        (1.0 - persistent_identity)
        + identity_switch
        + (1.0 - birth_f1)
        + (1.0 - removal_f1)
        + birth_latency
        + removal_latency / 2.0
    ) / 6.0

    collision_f1 = _f1(
        values[:, 27],
        values[:, 28],
        values[:, 29],
        label="collision-F1",
    )
    collision_timing = _ratio(values[:, 30], values[:, 31], label="collision-timing")
    contact_error = ((1.0 - collision_f1) + collision_timing) / 2.0
    result = (
        SELECTION_SCORE_WEIGHTS["current_position"] * current_position
        + SELECTION_SCORE_WEIGHTS["current_velocity"] * current_velocity
        + SELECTION_SCORE_WEIGHTS["horizon_position"] * horizon_position
        + SELECTION_SCORE_WEIGHTS["horizon_velocity"] * horizon_velocity
        + SELECTION_SCORE_WEIGHTS["proposal_error"] * proposal_error
        + SELECTION_SCORE_WEIGHTS["identity_lifecycle_error"] * identity_lifecycle_error
        + SELECTION_SCORE_WEIGHTS["contact_error"] * contact_error
    )
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError("pooled physical bootstrap score is invalid")
    return result


def _full_population_score(
    physical: Sequence[DynamicSetExampleScoreEvidence],
    planning: Sequence[PlanningTaskScoreEvidence],
) -> float:
    components = dict(pooled_selection_score_evidence(physical).components)
    if set(components) != _PHYSICAL_COMPONENTS:
        missing = sorted(_PHYSICAL_COMPONENTS - set(components))
        extra = sorted(set(components) - _PHYSICAL_COMPONENTS)
        raise ValueError(f"pooled physical score is incomplete: missing={missing}, extra={extra}")
    # Use the same public functions as checkpoint selection so the point
    # estimate is exactly the aggregate selection-score comparison, rather
    # than a NumPy approximation to it.
    return selection_score(
        {
            **components,
            "planning_error": planning_error(tuple(record.outcome for record in planning)),
        }
    )


def pooled_paired_improvement_evidence(
    candidate_physical: Sequence[DynamicSetExampleScoreEvidence],
    reference_physical: Sequence[DynamicSetExampleScoreEvidence],
    candidate_planning: Sequence[PlanningTaskScoreEvidence],
    reference_planning: Sequence[PlanningTaskScoreEvidence],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 161_061,
) -> PairedImprovementEvidence:
    """Bootstrap raw physical/task pairs under the exact 85/15 score.

    Physical and planning rows are two independently resampled fixed strata.
    Candidate and reference always share indices within a stratum, preserving
    complete pairs.  Batched array gathers replace a per-replicate Python merge
    loop, keeping 10,000 replicates practical for 4,400 + 1,200 rows.
    """

    _validate_bootstrap_parameters(bootstrap_samples, bootstrap_seed)
    candidate_records, reference_records = _validated_physical_pairs(
        candidate_physical,
        reference_physical,
    )
    candidate_tasks, reference_tasks = _validated_planning_pairs(
        candidate_planning,
        reference_planning,
    )
    candidate_score = _full_population_score(candidate_records, candidate_tasks)
    reference_score = _full_population_score(reference_records, reference_tasks)
    if reference_score <= 0.0:
        raise ValueError("aggregate paired baseline score must be positive")
    improvement = (reference_score - candidate_score) / reference_score

    candidate_statistics = _pack_physical_statistics(candidate_records)
    reference_statistics = _pack_physical_statistics(reference_records)
    joint_statistics = np.concatenate(
        (candidate_statistics, reference_statistics),
        axis=1,
    )
    candidate_planning_errors = np.asarray(
        [record.task_error for record in candidate_tasks],
        dtype=np.float64,
    )
    reference_planning_errors = np.asarray(
        [record.task_error for record in reference_tasks],
        dtype=np.float64,
    )
    joint_planning_errors = np.column_stack((candidate_planning_errors, reference_planning_errors))
    if not np.isfinite(joint_planning_errors).all() or np.any(joint_planning_errors < 0.0):
        raise ValueError("planning task errors must be finite and nonnegative")

    # Spawn independent deterministic streams so neither stratum's population
    # size nor batching changes the random draws made for the other stratum.
    physical_seed, planning_seed = np.random.SeedSequence(bootstrap_seed).spawn(2)
    physical_generator = np.random.default_rng(physical_seed)
    planning_generator = np.random.default_rng(planning_seed)
    replicate_values = np.empty(bootstrap_samples, dtype=np.float64)
    physical_count = len(candidate_records)
    planning_count = len(candidate_tasks)
    batch_size = max(
        1,
        min(
            128,
            _MAXIMUM_GATHER_VALUES // (physical_count * joint_statistics.shape[1]),
        ),
    )
    for start in range(0, bootstrap_samples, batch_size):
        count = min(batch_size, bootstrap_samples - start)
        physical_indices = physical_generator.integers(
            0,
            physical_count,
            size=(count, physical_count),
            endpoint=False,
        )
        physical_sums = joint_statistics[physical_indices].sum(axis=1, dtype=np.float64)
        candidate_physical_score = _physical_scores_from_sums(
            physical_sums[:, :_PACKED_STATISTIC_COUNT]
        )
        reference_physical_score = _physical_scores_from_sums(
            physical_sums[:, _PACKED_STATISTIC_COUNT:]
        )

        planning_indices = planning_generator.integers(
            0,
            planning_count,
            size=(count, planning_count),
            endpoint=False,
        )
        planning_means = joint_planning_errors[planning_indices].mean(
            axis=1,
            dtype=np.float64,
        )
        # The physical scorer already applies the seven frozen weights, whose
        # sum is exactly 0.85.  Only the planning 0.15 remains to be added.
        candidate_replicate = candidate_physical_score + _PLANNING_WEIGHT * planning_means[:, 0]
        reference_replicate = reference_physical_score + _PLANNING_WEIGHT * planning_means[:, 1]
        differences = reference_replicate - candidate_replicate
        values = np.divide(
            differences,
            reference_replicate,
            out=differences / reference_score,
            where=reference_replicate > 0.0,
        )
        replicate_values[start : start + count] = values

    if not math.isclose(_PHYSICAL_WEIGHT, 0.85, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("frozen physical selection weight differs from 0.85")
    if not math.isclose(_PLANNING_WEIGHT, 0.15, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("frozen planning selection weight differs from 0.15")
    replicate_values.sort()
    lower_index = max(0, math.ceil(0.05 * bootstrap_samples) - 1)
    lower_bound = float(replicate_values[lower_index])
    if not math.isfinite(improvement) or not math.isfinite(lower_bound):
        raise FloatingPointError("pooled paired improvement evidence is nonfinite")
    return PairedImprovementEvidence(
        sample_count=physical_count + planning_count,
        baseline_aggregate_score=reference_score,
        candidate_aggregate_score=candidate_score,
        paired_score_improvement_fraction=improvement,
        paired_bootstrap_lower_bound=lower_bound,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    ).validate()


__all__ = ["PlanningTaskScoreEvidence", "pooled_paired_improvement_evidence"]
