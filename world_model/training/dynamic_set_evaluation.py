"""Clean, additive development evaluation for specification 1.61.

The public pass in this module accepts :class:`DynamicSetPublicFrame` objects
only.  It snapshots detached inference products before the private scoring
pass is allowed to inspect simulator state.  This split is intentional: a
model, hook, or callback can never receive a ``DynamicSetMaterialization`` or
one of its truth tensors.

Selector, confirmation, final, and OOD populations are deliberately outside
this module.  It accepts only development rows, plus explicitly requested
training rows for the disposable screen.
"""

from __future__ import annotations

import math
import resource
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from world_model.belief import WorldBelief, fast_packing_map
from world_model.dynamics import WorldImpulseAction
from world_model.observations import MeasurementSet, ObservationPacket
from world_model.runtime import OnlineWorldModel
from world_model.runtime.prepared import tensor_identity_version_signature
from world_model.training.checkpointing import load_checkpoint
from world_model.training.dynamic_set_config import OrpheusConfig
from world_model.training.dynamic_set_gates import (
    HORIZON_POSITION_LIMITS_M,
    PhysicalCellMetrics,
    ResourceMetrics,
    SupportedScalar,
)
from world_model.training.dynamic_set_materializer import (
    DynamicSetMaterialization,
    DynamicSetPublicBoundaryEvidence,
    DynamicSetPublicFrame,
    _materialize_protected_dynamic_set_episode,
    _mint_protected_physical_materialization_capability,
    inspect_dynamic_set_public_boundary,
    materialize_dynamic_set_episode,
)
from world_model.training.dynamic_set_protocol import (
    FROZEN_PHYSICAL_MANIFEST_SHA256,
    PHYSICAL_CELLS,
    PHYSICAL_SPLIT_SIZES,
    SELECTION_SCORE_WEIGHTS,
    SIMULATOR_VERSION,
    SPECIFICATION_VERSION,
    PhysicalCell,
    PhysicalManifestRow,
    selection_score,
)
from world_model.training.dynamic_set_scene import (
    DYNAMIC_SET_FRAMES,
    DYNAMIC_SET_IMAGE_SIZE,
    DYNAMIC_SET_MAX_OBJECTS,
)
from world_model.training.qualification_core import (
    OrderedSplitLedger,
    SplitPermit,
    canonical_sha256,
    validated_sha256,
)

HORIZONS_SECONDS: tuple[float, ...] = tuple(HORIZON_POSITION_LIMITS_M)
_Z_90 = 1.6448536269514722
_PROTECTED_SPLITS = frozenset({"selector", "confirmation", "final_test", "compositional_ood"})
_PROTECTED_SPLIT_INDICES = {
    "selector": 0,
    "confirmation": 1,
    "final_test": 2,
    "compositional_ood": 3,
}
PHYSICAL_POPULATION_CLAIM_PURPOSE = "dynamic_set_physical_population_v1"

# Horizon velocity is selection evidence, not an absolute promotion gate.  Its
# fixed scale is the mature-current velocity gate, keeping the dimensionless
# lower-is-better component stable across checkpoints.
HORIZON_VELOCITY_SCORE_SCALE_MPS = 0.020
MAXIMUM_PAIRED_EVALUATION_BATCH_SIZE = 32
# B32 stays below half of the 2.5 GiB qualification ceiling on the reference
# CPU host. Frozen manifest order changes action schedule every 22 rows, so
# governed populations naturally flush at B22 while synthetic homogeneous
# populations may use the full measured-safe ceiling.
DEFAULT_PAIRED_EVALUATION_BATCH_SIZE = MAXIMUM_PAIRED_EVALUATION_BATCH_SIZE
_PUBLIC_ACTION_HANDLE_COSINE_MARGIN = 0.05
_PUBLIC_ACTION_SINGLE_COSINE = 0.95
_MISSING_EVENT_TIMING_PENALTY_FRAMES = float(DYNAMIC_SET_FRAMES)
_MISSING_BIRTH_LATENCY_PENALTY_FRAMES = 2.0
_MISSING_REMOVAL_LATENCY_PENALTY_FRAMES = 3.0
_MISSING_VELOCITY_PENALTY_MPS = 1.0
_STATE_BIRTH_CONFIRMATION_OBSERVATIONS = 2


class DynamicSetEvaluationError(RuntimeError):
    """Raised when clean evaluation cannot produce governed evidence."""


@dataclass(frozen=True)
class DynamicSetEvaluationConfig:
    """Frozen evaluator semantics independent of a learned checkpoint."""

    horizon_anchor_frame: int = 15
    proposal_existence_threshold: float = 0.55
    proposal_match_distance_m: float = 0.105
    belief_match_distance_m: float = 0.21
    collision_probability_threshold: float = 0.5
    uncertainty_z: float = _Z_90

    def validate(self) -> DynamicSetEvaluationConfig:
        if self.horizon_anchor_frame != 15:
            raise ValueError("the 2 s six-horizon evaluator anchor is frozen at frame 15")
        for name in (
            "proposal_existence_threshold",
            "collision_probability_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0,1]")
        for name in ("proposal_match_distance_m", "belief_match_distance_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.uncertainty_z) or self.uncertainty_z <= 0.0:
            raise ValueError("uncertainty_z must be finite and positive")
        return self


DEFAULT_EVALUATION_CONFIG = DynamicSetEvaluationConfig()


@dataclass(frozen=True)
class ProposalFrameTrace:
    position: Tensor
    log_variance: Tensor
    existence_probability: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class BeliefFrameTrace:
    frame_index: int
    timestamp: float
    object_id: Tensor
    active: Tensor
    position: Tensor
    velocity: Tensor
    motion_mode: Tensor
    interval_collision: Tensor
    interval_pair_collision_probability: Tensor
    fast_log_variance: Tensor
    proposal: ProposalFrameTrace


@dataclass(frozen=True)
class HorizonTrace:
    anchor_frame: int
    source_object_id: Tensor
    source_active: Tensor
    timestamps: Tensor
    positions: Tensor
    velocities: Tensor
    fast_log_variance: Tensor
    active_mask: Tensor


@dataclass(frozen=True)
class DynamicSetEpisodeTrace:
    """Detached result of one truth-free public inference pass."""

    frames: tuple[BeliefFrameTrace, ...]
    horizon: HorizonTrace
    perception_latencies_seconds: tuple[float, ...]
    six_horizon_latency_seconds: float
    public_action_count: int
    persistent_tensor_bytes: int
    process_rss_bytes: int
    # Resource gates are explicitly B1. Physical evidence remains valid for
    # every batch width, but only traces marked one contribute latency and
    # persistent-runtime bytes to those gates.
    runtime_batch_size: int = 1
    public_boundary_evidence: DynamicSetPublicBoundaryEvidence | None = None


@dataclass
class SumCount:
    total: float = 0.0
    count: int = 0

    def add(self, value: float, *, count: int = 1) -> None:
        if not math.isfinite(value):
            raise ValueError("an additive metric contribution must be finite")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("an additive metric count must be a nonnegative integer")
        self.total += float(value)
        self.count += count

    def merge(self, other: SumCount) -> None:
        self.add(other.total, count=other.count)

    def mean(self) -> float | None:
        return None if self.count == 0 else self.total / self.count


@dataclass
class SquaredErrorSum:
    squared_error: float = 0.0
    coordinate_count: int = 0

    def add_tensor(self, error: Tensor) -> None:
        value = error.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("state error contains NaN or Inf")
        self.squared_error += float(value.square().sum())
        self.coordinate_count += value.numel()

    def add_missing(self, *, euclidean_penalty: float, dimensions: int = 3) -> None:
        """Add one truth-owned target for which no prediction was available."""

        if (
            isinstance(euclidean_penalty, bool)
            or not isinstance(euclidean_penalty, (int, float))
            or not math.isfinite(float(euclidean_penalty))
            or float(euclidean_penalty) <= 0.0
        ):
            raise ValueError("missing-target penalty must be finite and positive")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("missing-target dimensions must be a positive integer")
        # Existing RMSEs pool coordinate counts. A missing 3-D vector receives
        # one conservative Euclidean error while retaining the same three-axis
        # denominator as an observed vector.
        self.squared_error += float(euclidean_penalty) ** 2
        self.coordinate_count += dimensions

    def merge(self, other: SquaredErrorSum) -> None:
        if not math.isfinite(other.squared_error) or other.squared_error < 0.0:
            raise ValueError("squared-error evidence must be finite and nonnegative")
        if other.coordinate_count < 0:
            raise ValueError("coordinate support must be nonnegative")
        self.squared_error += other.squared_error
        self.coordinate_count += other.coordinate_count

    def rmse(self) -> float | None:
        if self.coordinate_count == 0:
            return None
        return math.sqrt(self.squared_error / self.coordinate_count)


@dataclass
class BinaryCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(
        self, *, true_positive: int = 0, false_positive: int = 0, false_negative: int = 0
    ) -> None:
        values = (true_positive, false_positive, false_negative)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("binary confusion counts must be nonnegative integers")
        self.true_positive += true_positive
        self.false_positive += false_positive
        self.false_negative += false_negative

    def merge(self, other: BinaryCounts) -> None:
        self.add(
            true_positive=other.true_positive,
            false_positive=other.false_positive,
            false_negative=other.false_negative,
        )

    @property
    def precision_support(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def recall_support(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def f1_support(self) -> int:
        return 2 * self.true_positive + self.false_positive + self.false_negative

    def precision(self) -> float | None:
        if self.precision_support == 0:
            return None
        return self.true_positive / self.precision_support

    def recall(self) -> float | None:
        if self.recall_support == 0:
            return None
        return self.true_positive / self.recall_support

    def f1(self) -> float | None:
        if self.f1_support == 0:
            return None
        return 2 * self.true_positive / self.f1_support


@dataclass
class RatioCounts:
    numerator: int = 0
    denominator: int = 0

    def add(self, numerator: int, denominator: int) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (numerator, denominator)
        ):
            raise ValueError("ratio counts must be nonnegative integers")
        if numerator > denominator:
            raise ValueError("a ratio numerator cannot exceed its denominator")
        self.numerator += numerator
        self.denominator += denominator

    def merge(self, other: RatioCounts) -> None:
        self.add(other.numerator, other.denominator)

    def value(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator


@dataclass
class DynamicSetCellAccumulator:
    """Mergeable sufficient statistics for one physical cell."""

    episode_count: int = 0
    proposal: BinaryCounts = field(default_factory=BinaryCounts)
    exact_count: RatioCounts = field(default_factory=RatioCounts)
    current_position: SquaredErrorSum = field(default_factory=SquaredErrorSum)
    mature_velocity: SquaredErrorSum = field(default_factory=SquaredErrorSum)
    post_event_velocity: SquaredErrorSum = field(default_factory=SquaredErrorSum)
    horizon_position: dict[float, SquaredErrorSum] = field(
        default_factory=lambda: {horizon: SquaredErrorSum() for horizon in HORIZONS_SECONDS}
    )
    horizon_velocity: dict[float, SquaredErrorSum] = field(
        default_factory=lambda: {horizon: SquaredErrorSum() for horizon in HORIZONS_SECONDS}
    )
    collision: BinaryCounts = field(default_factory=BinaryCounts)
    collision_timing: SumCount = field(default_factory=SumCount)
    persistent_identity: RatioCounts = field(default_factory=RatioCounts)
    identity_switch: RatioCounts = field(default_factory=RatioCounts)
    birth: BinaryCounts = field(default_factory=BinaryCounts)
    birth_latency: SumCount = field(default_factory=SumCount)
    removal: BinaryCounts = field(default_factory=BinaryCounts)
    removal_latency: SumCount = field(default_factory=SumCount)
    uncertainty_90: RatioCounts = field(default_factory=RatioCounts)

    def merge(self, other: DynamicSetCellAccumulator) -> None:
        if other.episode_count < 0:
            raise ValueError("episode support must be nonnegative")
        self.episode_count += other.episode_count
        for name in (
            "proposal",
            "exact_count",
            "current_position",
            "mature_velocity",
            "post_event_velocity",
            "collision",
            "collision_timing",
            "persistent_identity",
            "identity_switch",
            "birth",
            "birth_latency",
            "removal",
            "removal_latency",
            "uncertainty_90",
        ):
            getattr(self, name).merge(getattr(other, name))
        for horizon in HORIZONS_SECONDS:
            self.horizon_position[horizon].merge(other.horizon_position[horizon])
            self.horizon_velocity[horizon].merge(other.horizon_velocity[horizon])

    @staticmethod
    def _supported(value: float | None, support: int) -> SupportedScalar:
        # ``0`` is an inert serialization sentinel when support is zero.  Gate
        # consumers must (and do) inspect support before value.  Raw evidence
        # and flat reports omit that value entirely, so an unsupported metric
        # can never be mistaken for an observed zero error.
        return SupportedScalar(value=0.0 if value is None else float(value), support=support)

    @classmethod
    def _binary_precision(cls, counts: BinaryCounts) -> SupportedScalar:
        return cls._supported(counts.precision(), counts.precision_support)

    @classmethod
    def _binary_recall(cls, counts: BinaryCounts) -> SupportedScalar:
        return cls._supported(counts.recall(), counts.recall_support)

    @classmethod
    def _binary_f1(cls, counts: BinaryCounts) -> SupportedScalar:
        return cls._supported(counts.f1(), counts.f1_support)

    def physical_metrics(self, cell: PhysicalCell) -> PhysicalCellMetrics:
        """Materialize gate-shaped metrics without averaging episode metrics."""

        cell.validate()
        dynamic = cell.dynamic_membership
        contact = cell.contact

        def perfect_negative(false_positive: int) -> SupportedScalar:
            # Negative cells still have an absolute false-positive contract.
            # Episode support makes the all-clear case observed evidence rather
            # than an unsupported/vacuous metric.
            return SupportedScalar(
                value=1.0 if false_positive == 0 else 0.0,
                support=self.episode_count,
            )

        def zero_negative_latency() -> SupportedScalar:
            return SupportedScalar(value=0.0, support=self.episode_count)

        return PhysicalCellMetrics(
            proposal_precision=self._binary_precision(self.proposal),
            proposal_recall=self._binary_recall(self.proposal),
            proposal_f1=self._binary_f1(self.proposal),
            exact_count_accuracy=self._supported(
                self.exact_count.value(), self.exact_count.denominator
            ),
            current_position_rmse_m=self._supported(
                self.current_position.rmse(), self.current_position.coordinate_count
            ),
            mature_velocity_rmse_mps=self._supported(
                self.mature_velocity.rmse(), self.mature_velocity.coordinate_count
            ),
            post_event_velocity_rmse_mps=(
                self._supported(
                    self.post_event_velocity.rmse(),
                    self.post_event_velocity.coordinate_count,
                )
                if dynamic or contact
                else None
            ),
            horizon_position_rmse_m={
                horizon: self._supported(metric.rmse(), metric.coordinate_count)
                for horizon, metric in self.horizon_position.items()
            },
            collision_f1=(
                self._binary_f1(self.collision)
                if contact
                else perfect_negative(self.collision.false_positive)
            ),
            collision_timing_error_frames=(
                self._supported(self.collision_timing.mean(), self.collision_timing.count)
                if contact
                else zero_negative_latency()
            ),
            persistent_id_accuracy=self._supported(
                self.persistent_identity.value(), self.persistent_identity.denominator
            ),
            identity_switch_rate=self._supported(
                self.identity_switch.value(), self.identity_switch.denominator
            ),
            birth_precision=(
                self._binary_precision(self.birth)
                if dynamic
                else perfect_negative(self.birth.false_positive)
            ),
            birth_recall=(
                self._binary_recall(self.birth)
                if dynamic
                else perfect_negative(self.birth.false_positive)
            ),
            birth_latency_frames=(
                self._supported(self.birth_latency.mean(), self.birth_latency.count)
                if dynamic
                else zero_negative_latency()
            ),
            removal_precision=(
                self._binary_precision(self.removal)
                if dynamic
                else perfect_negative(self.removal.false_positive)
            ),
            removal_recall=(
                self._binary_recall(self.removal)
                if dynamic
                else perfect_negative(self.removal.false_positive)
            ),
            removal_latency_frames=(
                self._supported(self.removal_latency.mean(), self.removal_latency.count)
                if dynamic
                else zero_negative_latency()
            ),
            uncertainty_90_coverage=self._supported(
                self.uncertainty_90.value(), self.uncertainty_90.denominator
            ),
        )


@dataclass(frozen=True, slots=True)
class DynamicSetAdditiveSnapshot:
    """Immutable, tensor-free sufficient statistics for one evaluated row."""

    episode_count: int
    proposal: tuple[int, int, int]
    exact_count: tuple[int, int]
    current_position: tuple[float, int]
    mature_velocity: tuple[float, int]
    post_event_velocity: tuple[float, int]
    horizon_position: tuple[tuple[float, float, int], ...]
    horizon_velocity: tuple[tuple[float, float, int], ...]
    collision: tuple[int, int, int]
    collision_timing: tuple[float, int]
    persistent_identity: tuple[int, int]
    identity_switch: tuple[int, int]
    birth: tuple[int, int, int]
    birth_latency: tuple[float, int]
    removal: tuple[int, int, int]
    removal_latency: tuple[float, int]
    uncertainty_90: tuple[int, int]

    @classmethod
    def from_accumulator(cls, accumulator: DynamicSetCellAccumulator) -> DynamicSetAdditiveSnapshot:
        checked = DynamicSetCellAccumulator()
        checked.merge(accumulator)

        def binary(value: BinaryCounts) -> tuple[int, int, int]:
            return (value.true_positive, value.false_positive, value.false_negative)

        def ratio(value: RatioCounts) -> tuple[int, int]:
            return (value.numerator, value.denominator)

        def error(value: SquaredErrorSum) -> tuple[float, int]:
            return (value.squared_error, value.coordinate_count)

        def summed(value: SumCount) -> tuple[float, int]:
            return (value.total, value.count)

        return cls(
            episode_count=checked.episode_count,
            proposal=binary(checked.proposal),
            exact_count=ratio(checked.exact_count),
            current_position=error(checked.current_position),
            mature_velocity=error(checked.mature_velocity),
            post_event_velocity=error(checked.post_event_velocity),
            horizon_position=tuple(
                (
                    horizon,
                    checked.horizon_position[horizon].squared_error,
                    checked.horizon_position[horizon].coordinate_count,
                )
                for horizon in HORIZONS_SECONDS
            ),
            horizon_velocity=tuple(
                (
                    horizon,
                    checked.horizon_velocity[horizon].squared_error,
                    checked.horizon_velocity[horizon].coordinate_count,
                )
                for horizon in HORIZONS_SECONDS
            ),
            collision=binary(checked.collision),
            collision_timing=summed(checked.collision_timing),
            persistent_identity=ratio(checked.persistent_identity),
            identity_switch=ratio(checked.identity_switch),
            birth=binary(checked.birth),
            birth_latency=summed(checked.birth_latency),
            removal=binary(checked.removal),
            removal_latency=summed(checked.removal_latency),
            uncertainty_90=ratio(checked.uncertainty_90),
        )

    def to_accumulator(self) -> DynamicSetCellAccumulator:
        def binary(value: tuple[int, int, int]) -> BinaryCounts:
            result = BinaryCounts()
            result.add(
                true_positive=value[0],
                false_positive=value[1],
                false_negative=value[2],
            )
            return result

        def ratio(value: tuple[int, int]) -> RatioCounts:
            result = RatioCounts()
            result.add(*value)
            return result

        def error(value: tuple[float, int]) -> SquaredErrorSum:
            result = SquaredErrorSum()
            result.merge(SquaredErrorSum(squared_error=value[0], coordinate_count=value[1]))
            return result

        def summed(value: tuple[float, int]) -> SumCount:
            result = SumCount()
            result.add(value[0], count=value[1])
            return result

        if self.episode_count < 0:
            raise ValueError("snapshot episode support must be nonnegative")
        if tuple(item[0] for item in self.horizon_position) != HORIZONS_SECONDS:
            raise ValueError("snapshot horizon-position schema differs")
        if tuple(item[0] for item in self.horizon_velocity) != HORIZONS_SECONDS:
            raise ValueError("snapshot horizon-velocity schema differs")
        return DynamicSetCellAccumulator(
            episode_count=self.episode_count,
            proposal=binary(self.proposal),
            exact_count=ratio(self.exact_count),
            current_position=error(self.current_position),
            mature_velocity=error(self.mature_velocity),
            post_event_velocity=error(self.post_event_velocity),
            horizon_position={
                horizon: error((squared_error, support))
                for horizon, squared_error, support in self.horizon_position
            },
            horizon_velocity={
                horizon: error((squared_error, support))
                for horizon, squared_error, support in self.horizon_velocity
            },
            collision=binary(self.collision),
            collision_timing=summed(self.collision_timing),
            persistent_identity=ratio(self.persistent_identity),
            identity_switch=ratio(self.identity_switch),
            birth=binary(self.birth),
            birth_latency=summed(self.birth_latency),
            removal=binary(self.removal),
            removal_latency=summed(self.removal_latency),
            uncertainty_90=ratio(self.uncertainty_90),
        )


@dataclass(frozen=True, slots=True)
class DynamicSetExampleScoreEvidence:
    """Digest-bound per-row evidence suitable for paired bootstrap resampling."""

    split: str
    ordinal: int
    seed: int
    cell_index: int
    row_sha256: str
    additive: DynamicSetAdditiveSnapshot
    supported_components: tuple[tuple[str, float], ...]
    public_boundary_evidence: DynamicSetPublicBoundaryEvidence
    evidence_sha256: str

    @classmethod
    def create(
        cls,
        row: PhysicalManifestRow,
        accumulator: DynamicSetCellAccumulator,
        *,
        public_boundary_evidence: DynamicSetPublicBoundaryEvidence | None = None,
    ) -> DynamicSetExampleScoreEvidence:
        row_sha256 = canonical_sha256(asdict(row))
        boundary = public_boundary_evidence
        if boundary is None:
            boundary = inspect_dynamic_set_public_boundary(None, ()).bind_manifest_row(row)
        if type(boundary) is not DynamicSetPublicBoundaryEvidence:
            raise TypeError("per-example boundary must use exact boundary evidence")
        boundary.validate()
        if boundary.row_sha256 != row_sha256:
            raise ValueError("per-example boundary differs from its manifest row")
        additive = DynamicSetAdditiveSnapshot.from_accumulator(accumulator)
        components = tuple(sorted(_score_components(accumulator).components.items()))
        unsigned = {
            "split": row.split,
            "ordinal": row.ordinal,
            "seed": row.seed,
            "cell_index": row.cell_index,
            "row_sha256": row_sha256,
            "additive": asdict(additive),
            "supported_components": components,
            "public_boundary_evidence": asdict(boundary),
        }
        return cls(
            split=row.split,
            ordinal=row.ordinal,
            seed=row.seed,
            cell_index=row.cell_index,
            row_sha256=row_sha256,
            additive=additive,
            supported_components=components,
            public_boundary_evidence=boundary,
            evidence_sha256=canonical_sha256(unsigned),
        )

    @property
    def pair_key(self) -> tuple[str, int, int, int, str]:
        return (self.split, self.ordinal, self.seed, self.cell_index, self.row_sha256)

    def validate(self) -> DynamicSetExampleScoreEvidence:
        validated_sha256(self.row_sha256, label="row_sha256")
        validated_sha256(self.evidence_sha256, label="evidence_sha256")
        if type(self.public_boundary_evidence) is not DynamicSetPublicBoundaryEvidence:
            raise TypeError("per-example boundary must use exact boundary evidence")
        self.public_boundary_evidence.validate()
        if self.public_boundary_evidence.row_sha256 != self.row_sha256:
            raise ValueError("per-example boundary row digest differs")
        reconstructed = self.additive.to_accumulator()
        expected_components = tuple(sorted(_score_components(reconstructed).components.items()))
        if self.supported_components != expected_components:
            raise ValueError("per-example score components differ from additive evidence")
        unsigned = {
            "split": self.split,
            "ordinal": self.ordinal,
            "seed": self.seed,
            "cell_index": self.cell_index,
            "row_sha256": self.row_sha256,
            "additive": asdict(self.additive),
            "supported_components": self.supported_components,
            "public_boundary_evidence": asdict(self.public_boundary_evidence),
        }
        if canonical_sha256(unsigned) != self.evidence_sha256:
            raise ValueError("per-example score evidence digest mismatch")
        return self


@dataclass(frozen=True, slots=True)
class EvaluationPopulationBinding:
    split: str
    protocol_sha256: str | None
    manifest_sha256: str | None
    permit_index: int | None
    row_count: int


@dataclass(frozen=True)
class SelectionScoreEvidence:
    """The seven physical components; planning remains a required input."""

    components: Mapping[str, float]

    def with_planning_error(self, planning_error: float) -> float:
        if not math.isfinite(planning_error) or planning_error < 0.0:
            raise ValueError("planning_error must be finite and nonnegative")
        complete = {**self.components, "planning_error": float(planning_error)}
        return selection_score(complete)


@dataclass(frozen=True)
class DynamicSetEvaluationResult:
    by_cell: Mapping[PhysicalCell, PhysicalCellMetrics]
    evidence_by_cell: Mapping[PhysicalCell, DynamicSetCellAccumulator]
    score: SelectionScoreEvidence
    resources: ResourceMetrics
    episode_count: int
    per_example_score_evidence: tuple[DynamicSetExampleScoreEvidence, ...] = ()
    population_binding: EvaluationPopulationBinding | None = None

    def flat_metrics(self, *, include_resources: bool = True) -> dict[str, float]:
        """Return supported values and explicit supports for validation logs."""

        output: dict[str, float] = {"evaluation/episode_count": float(self.episode_count)}
        for cell, metrics in sorted(
            self.by_cell.items(),
            key=lambda item: (item[0].object_count, item[0].contact, item[0].dynamic_membership),
        ):
            prefix = (
                f"cell/n{cell.object_count}/contact{int(cell.contact)}"
                f"/dynamic{int(cell.dynamic_membership)}"
            )
            _flatten_cell_metrics(output, prefix, metrics)
        for name, value in self.score.components.items():
            output[f"selection_component/{name}"] = float(value)
        if include_resources:
            output.update(
                {
                    "resource/perception_latency_seconds": self.resources.perception_latency_seconds,
                    "resource/six_horizon_rollout_seconds": (
                        self.resources.six_horizon_rollout_seconds
                    ),
                    "resource/learned_weight_bytes": float(self.resources.learned_weight_bytes),
                    "resource/persistent_tensor_bytes": float(
                        self.resources.persistent_tensor_bytes
                    ),
                    "resource/process_rss_bytes": float(self.resources.process_rss_bytes),
                }
            )
        return output


@dataclass(frozen=True)
class DynamicSetPairedEvaluationResult:
    """Candidate/reference results produced from one bounded evidence stream."""

    candidate: DynamicSetEvaluationResult
    reference: DynamicSetEvaluationResult

    def validate(self) -> DynamicSetPairedEvaluationResult:
        pair_dynamic_set_example_score_evidence(
            self.candidate.per_example_score_evidence,
            self.reference.per_example_score_evidence,
        )
        if self.candidate.episode_count != self.reference.episode_count:
            raise ValueError("paired physical evaluation row counts differ")
        if self.candidate.population_binding != self.reference.population_binding:
            raise ValueError("paired physical evaluation population bindings differ")
        return self


def _clone_cpu(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu").clone()


def _exact_symmetric_probability(logits: Tensor) -> Tensor:
    """Apply sigmoid while retaining exact pair-matrix symmetry.

    Some vectorized CPU sigmoid kernels can round equal transposed entries one
    ulp differently because they occupy different SIMD lanes. The learned
    logits have already passed an exact-symmetry boundary, so selecting one
    triangle after the elementwise transform preserves the same mathematical
    probability and the public exact-symmetry contract.
    """

    probability = logits.sigmoid()
    upper = torch.triu(probability)
    return upper + torch.triu(probability, diagonal=1).transpose(-1, -2)


_PUBLIC_BOUNDARY_PROJECTION_FIELDS = (
    "frame_count",
    "exact_frame_type_count",
    "exact_schema_frame_count",
    "public_tensor_count",
    "unexpected_frame_type_count",
    "subclass_frame_count",
    "unexpected_field_count",
    "missing_field_count",
    "invalid_public_field_count",
    "public_storage_alias_count",
    "truth_storage_alias_count",
    "public_payload_sha256",
)


def _validated_public_boundary(
    frames: Sequence[object],
    supplied: DynamicSetPublicBoundaryEvidence | None,
) -> DynamicSetPublicBoundaryEvidence:
    observed = inspect_dynamic_set_public_boundary(None, frames).require_clean(
        require_truth_binding=False
    )
    if supplied is None:
        return observed
    if type(supplied) is not DynamicSetPublicBoundaryEvidence:
        raise TypeError("public boundary must use exact DynamicSetPublicBoundaryEvidence")
    supplied.require_clean(require_truth_binding=supplied.truth_bound)
    if any(
        getattr(supplied, name) != getattr(observed, name)
        for name in _PUBLIC_BOUNDARY_PROJECTION_FIELDS
    ):
        raise DynamicSetEvaluationError(
            "public frames differ from their supplied boundary evidence"
        )
    return supplied


def _public_packet(frame: DynamicSetPublicFrame) -> ObservationPacket:
    if type(frame) is not DynamicSetPublicFrame:
        raise TypeError("public inference accepts exact DynamicSetPublicFrame objects only")
    for name in ("rgb", "depth", "world_from_camera", "intrinsics"):
        value = getattr(frame, name)
        if (
            type(value) is not Tensor
            or value.device.type != "cpu"
            or value.dtype is not torch.float32
        ):
            raise TypeError(f"public frame {name} must be CPU float32")
    return ObservationPacket(
        modality="rgbd",
        sensor_id="dynamic-set:camera0:rgbd",
        timestamp=frame.timestamp,
        payload={"rgb": frame.rgb.unsqueeze(0), "depth": frame.depth.unsqueeze(0)},
        calibration={
            "world_from_camera": frame.world_from_camera.unsqueeze(0),
            "intrinsics": frame.intrinsics.unsqueeze(0),
        },
        frame_id=f"dynamic-set:{frame.frame_index}",
        metadata={"image_size": DYNAMIC_SET_IMAGE_SIZE},
    )


def _public_packet_batch(frames: Sequence[DynamicSetPublicFrame]) -> ObservationPacket:
    """Stack one same-time public frame per independent episode."""

    batch = tuple(frames)
    if len(batch) < 2:
        raise ValueError("batched public inference requires at least two frames")
    first = batch[0]
    # Reuse the B1 validator without exposing its packet to the model.
    _public_packet(first)
    for frame in batch[1:]:
        _public_packet(frame)
        if frame.frame_index != first.frame_index or not math.isclose(
            frame.timestamp,
            first.timestamp,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError("batched public frames must share frame index and timestamp")
    return ObservationPacket(
        modality="rgbd",
        sensor_id="dynamic-set:camera0:rgbd",
        timestamp=first.timestamp,
        payload={
            "rgb": torch.stack([frame.rgb for frame in batch]),
            "depth": torch.stack([frame.depth for frame in batch]),
        },
        calibration={
            "world_from_camera": torch.stack([frame.world_from_camera for frame in batch]),
            "intrinsics": torch.stack([frame.intrinsics for frame in batch]),
        },
        frame_id=f"dynamic-set:{first.frame_index}",
        metadata={"image_size": DYNAMIC_SET_IMAGE_SIZE},
    )


def _validate_public_action_fields(frame: DynamicSetPublicFrame) -> bool:
    if (
        frame.known_action_observed.shape != (1,)
        or frame.known_action_observed.dtype is not torch.bool
        or frame.known_action_observed.device.type != "cpu"
    ):
        raise ValueError("known_action_observed must be boolean CPU [1]")
    expected = (
        ("known_action_timestamp", frame.known_action_timestamp, (1,), torch.float32),
        (
            "known_action_appearance_handle",
            frame.known_action_appearance_handle,
            (1, 8),
            torch.float32,
        ),
        ("known_impulse_world", frame.known_impulse_world, (1, 3), torch.float32),
    )
    for name, value, shape, dtype in expected:
        if value.shape != shape or value.dtype is not dtype or value.device.type != "cpu":
            raise ValueError(f"{name} must be CPU float32 with shape {shape}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite public data")
    observed = bool(frame.known_action_observed[0])
    if not observed:
        if (
            float(frame.known_action_timestamp[0]) != -1.0
            or bool(frame.known_action_appearance_handle.ne(0).any())
            or bool(frame.known_impulse_world.ne(0).any())
        ):
            raise ValueError("an unobserved public action must contain only sentinels")
        return False
    if not math.isclose(
        float(frame.known_action_timestamp[0]),
        frame.timestamp,
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise ValueError("public action timestamp differs from its exact application time")
    if bool(frame.known_action_appearance_handle.eq(0).all()):
        raise ValueError("an observed public action requires a nonzero appearance handle")
    if bool(frame.known_impulse_world.eq(0).all()):
        raise ValueError("an observed public action requires a nonzero impulse")
    return True


def _batched_public_action(
    frames: Sequence[DynamicSetPublicFrame],
    belief: WorldBelief | None,
) -> WorldImpulseAction | None:
    """Resolve a homogeneous public action batch using observable appearance only."""

    batch = tuple(frames)
    observed = torch.tensor(
        [_validate_public_action_fields(frame) for frame in batch],
        dtype=torch.bool,
    )
    if not bool(observed.any()):
        return None
    if not bool(observed.all()):
        raise DynamicSetEvaluationError(
            "a runtime batch may not mix observed and unobserved public actions"
        )
    if not isinstance(belief, WorldBelief) or belief.batch_size != len(batch):
        raise DynamicSetEvaluationError("a public action batch requires its preceding belief")
    belief.validate(log_variance_bounds=(-32.0, 20.0))
    active = belief.objects.active
    active_count = active.sum(dim=-1)
    if torch.any(active_count < 1):
        raise ValueError("public action target requires at least one active object")
    if torch.any((belief.objects.object_id < 0) & active):
        raise ValueError("active public action candidates require persistent IDs")
    duplicate = (
        active.unsqueeze(-1)
        & active.unsqueeze(-2)
        & (belief.objects.object_id.unsqueeze(-1) == belief.objects.object_id.unsqueeze(-2))
        & ~torch.eye(
            belief.objects.max_objects,
            device=belief.device,
            dtype=torch.bool,
        ).unsqueeze(0)
    )
    if torch.any(duplicate):
        raise ValueError("active public action candidates require unique persistent IDs")

    handles = torch.cat([frame.known_action_appearance_handle for frame in batch]).to(
        device=belief.device,
        dtype=belief.dtype,
    )
    appearance = belief.objects.appearance
    handle_norm = torch.linalg.vector_norm(handles, dim=-1)
    appearance_norm = torch.linalg.vector_norm(appearance, dim=-1)
    if torch.any(handle_norm <= 0.0) or torch.any(active & (appearance_norm <= 0.0)):
        raise ValueError("public action appearance evidence must have nonzero norm")
    cosine = torch.einsum(
        "bd,bnd->bn",
        handles / handle_norm.unsqueeze(-1),
        appearance / appearance_norm.clamp_min(torch.finfo(belief.dtype).tiny).unsqueeze(-1),
    ).masked_fill(~active, -torch.inf)
    top_values, top_slots = torch.topk(cosine, k=2, dim=-1, largest=True, sorted=True)
    singleton = active_count == 1
    if torch.any(singleton & (top_values[:, 0] < _PUBLIC_ACTION_SINGLE_COSINE)):
        raise ValueError("single-object public action handle is below the cosine gate")
    margin = top_values[:, 0] - top_values[:, 1]
    if torch.any(~singleton & ((margin <= 0.0) | (margin < _PUBLIC_ACTION_HANDLE_COSINE_MARGIN))):
        raise ValueError("public action appearance handle is ambiguous")
    object_id = torch.gather(
        belief.objects.object_id,
        dim=1,
        index=top_slots[:, :1],
    ).squeeze(1)
    action = WorldImpulseAction(
        timestamp=torch.cat([frame.known_action_timestamp for frame in batch])
        .to(device=belief.device, dtype=belief.dtype)
        .clone(),
        object_id=object_id.detach().clone(),
        impulse_world=torch.cat([frame.known_impulse_world for frame in batch])
        .to(device=belief.device, dtype=belief.dtype)
        .clone(),
    )
    action.validate_for(belief)
    return action


def _validate_cpu_float32_model(model: nn.Module) -> None:
    for kind, tensors in (
        ("parameter", model.parameters()),
        ("buffer", model.buffers()),
    ):
        for value in tensors:
            if value.device.type != "cpu":
                raise ValueError(f"dynamic-set evaluation requires CPU {kind}s")
            if value.is_floating_point() and value.dtype is not torch.float32:
                raise TypeError(f"dynamic-set evaluation requires float32 {kind}s")


def _measurement_trace(measurement: MeasurementSet | None) -> ProposalFrameTrace:
    if measurement is None:
        raise DynamicSetEvaluationError("RGB-D inference did not expose last_measurements")
    measurement.validate()
    if measurement.modality != "rgbd" or measurement.values.shape != (1, 8, 3):
        raise DynamicSetEvaluationError("set evaluation requires RGB-D measurements [1,8,3]")
    position = measurement.auxiliary.get("world_position", measurement.values)
    log_variance = measurement.auxiliary.get("world_log_variance", measurement.log_variance)
    if position.shape != (1, 8, 3) or log_variance.shape not in {(1, 8, 1), (1, 8, 3)}:
        raise DynamicSetEvaluationError("set proposals need metric position and variance")
    if log_variance.shape[-1] == 1:
        log_variance = log_variance.expand(-1, -1, 3)
    return ProposalFrameTrace(
        position=_clone_cpu(position[0]),
        log_variance=_clone_cpu(log_variance[0]),
        existence_probability=_clone_cpu(measurement.existence_logits[0].sigmoid()),
        valid_mask=_clone_cpu(measurement.measurement_mask[0]),
    )


def _belief_frame_trace(
    belief: Any,
    measurement: MeasurementSet | None,
    interval_collision_mask: Tensor | None,
    interval_pair_collision_logits: Tensor | None,
    *,
    frame_index: int,
    timestamp: float,
) -> BeliefFrameTrace:
    if belief.batch_size != 1:
        raise DynamicSetEvaluationError("dynamic-set episode inference requires batch size one")
    objects = belief.objects
    expected = (1, DYNAMIC_SET_MAX_OBJECTS)
    if objects.active.shape != expected or objects.object_id.shape != expected:
        raise DynamicSetEvaluationError("world belief must expose exactly six persistent slots")
    packing = fast_packing_map(objects)
    minimum_fast_width = max(packing["position"].stop, packing["velocity"].stop)
    if (
        objects.fast_log_variance.shape[:2] != expected
        or objects.fast_log_variance.shape[-1] < minimum_fast_width
    ):
        raise DynamicSetEvaluationError("belief fast variance does not cover position/velocity")
    for name, value in (
        ("position", objects.position),
        ("velocity", objects.velocity),
        ("fast_log_variance", objects.fast_log_variance),
        ("motion_mode_logits", objects.motion_mode_logits),
    ):
        if not bool(torch.isfinite(value).all()):
            raise DynamicSetEvaluationError(f"belief {name} contains NaN or Inf")
    if (
        measurement is None
        or measurement.timestamp.shape != (1,)
        or not math.isclose(float(measurement.timestamp[0]), timestamp, rel_tol=0.0, abs_tol=1.0e-7)
    ):
        raise DynamicSetEvaluationError("set evaluation requires a fresh measurement every frame")
    if interval_collision_mask is None:
        interval_collision_mask = torch.zeros_like(objects.active)
    if (
        interval_collision_mask.shape != expected
        or interval_collision_mask.dtype is not torch.bool
        or interval_collision_mask.device != objects.active.device
    ):
        raise DynamicSetEvaluationError(
            "runtime interval collision evidence must be boolean belief-slot [1,6]"
        )
    if interval_pair_collision_logits is None:
        if frame_index != 0:
            raise DynamicSetEvaluationError(
                "runtime did not expose learned pair collision confidence"
            )
        pair_collision_probability = objects.position.new_zeros(
            1,
            DYNAMIC_SET_MAX_OBJECTS,
            DYNAMIC_SET_MAX_OBJECTS,
        )
    else:
        pair_expected = (1, DYNAMIC_SET_MAX_OBJECTS, DYNAMIC_SET_MAX_OBJECTS)
        if (
            interval_pair_collision_logits.shape != pair_expected
            or not interval_pair_collision_logits.is_floating_point()
            or interval_pair_collision_logits.device != objects.active.device
            or not bool(torch.isfinite(interval_pair_collision_logits).all())
            or not torch.equal(
                interval_pair_collision_logits,
                interval_pair_collision_logits.transpose(1, 2),
            )
        ):
            raise DynamicSetEvaluationError(
                "runtime pair collision logits must be finite symmetric [1,6,6]"
            )
        pair_collision_probability = _exact_symmetric_probability(interval_pair_collision_logits)
    return BeliefFrameTrace(
        frame_index=frame_index,
        timestamp=float(timestamp),
        object_id=_clone_cpu(objects.object_id[0]),
        active=_clone_cpu(objects.active[0]),
        position=_clone_cpu(objects.position[0]),
        velocity=_clone_cpu(objects.velocity[0]),
        motion_mode=_clone_cpu(objects.motion_mode_logits[0].argmax(dim=-1)),
        interval_collision=_clone_cpu(interval_collision_mask[0]),
        interval_pair_collision_probability=_clone_cpu(pair_collision_probability[0]),
        fast_log_variance=_clone_cpu(objects.fast_log_variance[0]),
        proposal=_measurement_trace(measurement),
    )


def _horizon_trace(model: OnlineWorldModel, anchor_frame: int) -> tuple[HorizonTrace, float]:
    belief = model.belief
    if belief is None:
        raise DynamicSetEvaluationError("cannot roll out before belief initialization")
    source_signature = tensor_identity_version_signature(belief)
    started = time.perf_counter()
    rollout = getattr(getattr(model, "dynamics", None), "rollout", None)
    if not callable(rollout):
        raise DynamicSetEvaluationError("dynamic-set model lacks a dynamics rollout")
    trajectory = rollout(
        belief,
        HORIZONS_SECONDS,
        return_events=False,
        return_auxiliary=False,
    )
    elapsed = time.perf_counter() - started
    if tensor_identity_version_signature(belief) != source_signature or model.belief is not belief:
        raise DynamicSetEvaluationError("six-horizon rollout mutated its source belief")
    trajectory.validate()
    expected = (1, len(HORIZONS_SECONDS), DYNAMIC_SET_MAX_OBJECTS)
    if trajectory.positions.shape[:3] != expected:
        raise DynamicSetEvaluationError("six-horizon rollout has an unexpected object shape")
    return (
        HorizonTrace(
            anchor_frame=anchor_frame,
            source_object_id=_clone_cpu(belief.objects.object_id[0]),
            source_active=_clone_cpu(belief.objects.active[0]),
            timestamps=_clone_cpu(trajectory.timestamps[0]),
            positions=_clone_cpu(trajectory.positions[0]),
            velocities=_clone_cpu(trajectory.velocities[0]),
            fast_log_variance=_clone_cpu(trajectory.fast_log_variance[0]),
            active_mask=_clone_cpu(trajectory.active_mask[0]),
        ),
        elapsed,
    )


def _measurement_batch_traces(
    measurement: MeasurementSet | None,
    *,
    batch_size: int,
) -> tuple[ProposalFrameTrace, ...]:
    if measurement is None:
        raise DynamicSetEvaluationError("RGB-D inference did not expose last_measurements")
    measurement.validate()
    expected_position = (batch_size, 8, 3)
    if measurement.modality != "rgbd" or measurement.values.shape != expected_position:
        raise DynamicSetEvaluationError(
            f"set batch evaluation requires RGB-D measurements {expected_position}"
        )
    position = measurement.auxiliary.get("world_position", measurement.values)
    log_variance = measurement.auxiliary.get("world_log_variance", measurement.log_variance)
    if position.shape != expected_position or log_variance.shape not in {
        (batch_size, 8, 1),
        expected_position,
    }:
        raise DynamicSetEvaluationError("set batch proposals need metric position and variance")
    if log_variance.shape[-1] == 1:
        log_variance = log_variance.expand(-1, -1, 3)
    return tuple(
        ProposalFrameTrace(
            position=_clone_cpu(position[index]),
            log_variance=_clone_cpu(log_variance[index]),
            existence_probability=_clone_cpu(measurement.existence_logits[index].sigmoid()),
            valid_mask=_clone_cpu(measurement.measurement_mask[index]),
        )
        for index in range(batch_size)
    )


def _belief_batch_frame_traces(
    belief: WorldBelief,
    measurement: MeasurementSet | None,
    interval_collision_mask: Tensor | None,
    interval_pair_collision_logits: Tensor | None,
    *,
    frame_index: int,
    timestamp: float,
) -> tuple[BeliefFrameTrace, ...]:
    batch_size = belief.batch_size
    if batch_size < 2:
        raise DynamicSetEvaluationError("batched trace extraction requires batch size at least two")
    objects = belief.objects
    expected = (batch_size, DYNAMIC_SET_MAX_OBJECTS)
    if objects.active.shape != expected or objects.object_id.shape != expected:
        raise DynamicSetEvaluationError("world belief must expose six persistent slots per row")
    packing = fast_packing_map(objects)
    minimum_fast_width = max(packing["position"].stop, packing["velocity"].stop)
    if (
        objects.fast_log_variance.shape[:2] != expected
        or objects.fast_log_variance.shape[-1] < minimum_fast_width
    ):
        raise DynamicSetEvaluationError("belief fast variance does not cover position/velocity")
    for name, value in (
        ("position", objects.position),
        ("velocity", objects.velocity),
        ("fast_log_variance", objects.fast_log_variance),
        ("motion_mode_logits", objects.motion_mode_logits),
    ):
        if not bool(torch.isfinite(value).all()):
            raise DynamicSetEvaluationError(f"belief {name} contains NaN or Inf")
    if (
        measurement is None
        or measurement.timestamp.shape != (batch_size,)
        or not bool(
            torch.isclose(
                measurement.timestamp,
                measurement.timestamp.new_full((batch_size,), timestamp),
                rtol=0.0,
                atol=1.0e-7,
            ).all()
        )
    ):
        raise DynamicSetEvaluationError("set batch evaluation requires fresh measurements")
    if interval_collision_mask is None:
        interval_collision_mask = torch.zeros_like(objects.active)
    if (
        interval_collision_mask.shape != expected
        or interval_collision_mask.dtype is not torch.bool
        or interval_collision_mask.device != objects.active.device
    ):
        raise DynamicSetEvaluationError(
            "runtime interval collision evidence must be boolean belief-slot [B,6]"
        )
    pair_expected = (
        batch_size,
        DYNAMIC_SET_MAX_OBJECTS,
        DYNAMIC_SET_MAX_OBJECTS,
    )
    if interval_pair_collision_logits is None:
        if frame_index != 0:
            raise DynamicSetEvaluationError(
                "runtime did not expose learned pair collision confidence"
            )
        pair_collision_probability = objects.position.new_zeros(pair_expected)
    elif (
        interval_pair_collision_logits.shape != pair_expected
        or not interval_pair_collision_logits.is_floating_point()
        or interval_pair_collision_logits.device != objects.active.device
        or not bool(torch.isfinite(interval_pair_collision_logits).all())
        or not torch.equal(
            interval_pair_collision_logits,
            interval_pair_collision_logits.transpose(1, 2),
        )
    ):
        raise DynamicSetEvaluationError(
            "runtime pair collision logits must be finite symmetric [B,6,6]"
        )
    else:
        pair_collision_probability = _exact_symmetric_probability(interval_pair_collision_logits)
    proposals = _measurement_batch_traces(measurement, batch_size=batch_size)
    return tuple(
        BeliefFrameTrace(
            frame_index=frame_index,
            timestamp=float(timestamp),
            object_id=_clone_cpu(objects.object_id[index]),
            active=_clone_cpu(objects.active[index]),
            position=_clone_cpu(objects.position[index]),
            velocity=_clone_cpu(objects.velocity[index]),
            motion_mode=_clone_cpu(objects.motion_mode_logits[index].argmax(dim=-1)),
            interval_collision=_clone_cpu(interval_collision_mask[index]),
            interval_pair_collision_probability=_clone_cpu(pair_collision_probability[index]),
            fast_log_variance=_clone_cpu(objects.fast_log_variance[index]),
            proposal=proposals[index],
        )
        for index in range(batch_size)
    )


def _horizon_batch_traces(
    model: OnlineWorldModel,
    anchor_frame: int,
) -> tuple[tuple[HorizonTrace, ...], float]:
    belief = model.belief
    if belief is None or belief.batch_size < 2:
        raise DynamicSetEvaluationError("batched horizon requires an initialized B>1 belief")
    source_signature = tensor_identity_version_signature(belief)
    rollout = getattr(getattr(model, "dynamics", None), "rollout", None)
    if not callable(rollout):
        raise DynamicSetEvaluationError("dynamic-set model lacks a dynamics rollout")
    started = time.perf_counter()
    trajectory = rollout(
        belief,
        HORIZONS_SECONDS,
        return_events=False,
        return_auxiliary=False,
    )
    elapsed = time.perf_counter() - started
    if tensor_identity_version_signature(belief) != source_signature or model.belief is not belief:
        raise DynamicSetEvaluationError("batched horizon rollout mutated its source belief")
    trajectory.validate()
    expected = (belief.batch_size, len(HORIZONS_SECONDS), DYNAMIC_SET_MAX_OBJECTS)
    if trajectory.positions.shape[:3] != expected:
        raise DynamicSetEvaluationError("batched six-horizon rollout has an unexpected shape")
    return (
        tuple(
            HorizonTrace(
                anchor_frame=anchor_frame,
                source_object_id=_clone_cpu(belief.objects.object_id[index]),
                source_active=_clone_cpu(belief.objects.active[index]),
                timestamps=_clone_cpu(trajectory.timestamps[index]),
                positions=_clone_cpu(trajectory.positions[index]),
                velocities=_clone_cpu(trajectory.velocities[index]),
                fast_log_variance=_clone_cpu(trajectory.fast_log_variance[index]),
                active_mask=_clone_cpu(trajectory.active_mask[index]),
            )
            for index in range(belief.batch_size)
        ),
        elapsed,
    )


def run_public_dynamic_set_episode(
    model: OnlineWorldModel,
    public_frames: Iterable[DynamicSetPublicFrame],
    *,
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    public_boundary_evidence: DynamicSetPublicBoundaryEvidence | None = None,
) -> DynamicSetEpisodeTrace:
    """Run one clean public-only episode and return detached inference evidence."""

    config.validate()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    _validate_cpu_float32_model(model)
    frames = tuple(public_frames)
    if len(frames) != DYNAMIC_SET_FRAMES:
        raise ValueError(f"dynamic-set evaluation requires exactly {DYNAMIC_SET_FRAMES} frames")
    boundary = _validated_public_boundary(frames, public_boundary_evidence)
    if tuple(frame.frame_index for frame in frames) != tuple(range(DYNAMIC_SET_FRAMES)):
        raise ValueError("public frames must be in exact contiguous frame order")
    public_signature = tensor_identity_version_signature(frames)
    model.eval()
    model.reset(batch_size=1)
    traces: list[BeliefFrameTrace] = []
    latencies: list[float] = []
    horizon: HorizonTrace | None = None
    rollout_latency: float | None = None
    action_count = 0
    with torch.no_grad():
        for frame in frames:
            packet = _public_packet(frame)
            try:
                # Resolve the public appearance handle against the checkpoint's
                # preceding belief before ingesting the action-time RGB-D frame.
                # The frame carries no simulator ID or padded target slot.
                action = frame.world_impulse_action(model.belief)
            except (TypeError, ValueError) as error:
                raise DynamicSetEvaluationError(
                    f"public action target could not be resolved at frame "
                    f"{frame.frame_index}: {error}"
                ) from error
            action_count += int(action is not None)
            started = time.perf_counter()
            belief = model.ingest(packet, action=action)
            latencies.append(time.perf_counter() - started)
            traces.append(
                _belief_frame_trace(
                    belief,
                    model.last_measurements,
                    getattr(model, "last_interval_collision_mask", None),
                    getattr(model, "last_interval_pair_collision_logits", None),
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                )
            )
            if frame.frame_index == config.horizon_anchor_frame:
                horizon, rollout_latency = _horizon_trace(model, frame.frame_index)
    if horizon is None or rollout_latency is None:
        raise AssertionError("frozen horizon anchor was not evaluated")
    if tensor_identity_version_signature(frames) != public_signature:
        raise DynamicSetEvaluationError("inference mutated its public frame tensors")
    return DynamicSetEpisodeTrace(
        frames=tuple(traces),
        horizon=horizon,
        perception_latencies_seconds=tuple(latencies),
        six_horizon_latency_seconds=rollout_latency,
        public_action_count=action_count,
        persistent_tensor_bytes=_runtime_tensor_bytes(model),
        process_rss_bytes=_process_rss_bytes(),
        public_boundary_evidence=boundary,
    )


def run_public_dynamic_set_batch(
    model: OnlineWorldModel,
    public_episodes: Sequence[Iterable[DynamicSetPublicFrame]],
    *,
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    public_boundary_evidence: Sequence[DynamicSetPublicBoundaryEvidence] | None = None,
) -> tuple[DynamicSetEpisodeTrace, ...]:
    """Run independent public episodes in one runtime batch.

    B1 delegates to the historical evaluator exactly. B>1 is accepted only
    when every row has the same public action schedule; no zero-impulse action
    is fabricated for action-free rows.
    """

    config.validate()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    _validate_cpu_float32_model(model)
    episodes = tuple(tuple(frames) for frames in public_episodes)
    if not episodes:
        raise ValueError("batched public evaluation requires at least one episode")
    supplied_boundaries: tuple[DynamicSetPublicBoundaryEvidence | None, ...]
    if public_boundary_evidence is None:
        supplied_boundaries = (None,) * len(episodes)
    else:
        supplied_boundaries = tuple(public_boundary_evidence)
        if len(supplied_boundaries) != len(episodes):
            raise ValueError("public boundary count differs from public episode count")
    if len(episodes) == 1:
        return (
            run_public_dynamic_set_episode(
                model,
                episodes[0],
                config=config,
                public_boundary_evidence=supplied_boundaries[0],
            ),
        )
    if len(episodes) > MAXIMUM_PAIRED_EVALUATION_BATCH_SIZE:
        raise ValueError(
            f"public evaluation batch exceeds {MAXIMUM_PAIRED_EVALUATION_BATCH_SIZE} rows"
        )
    expected_indices = tuple(range(DYNAMIC_SET_FRAMES))
    boundaries: list[DynamicSetPublicBoundaryEvidence] = []
    for frames, supplied_boundary in zip(episodes, supplied_boundaries, strict=True):
        if len(frames) != DYNAMIC_SET_FRAMES:
            raise ValueError(f"dynamic-set evaluation requires exactly {DYNAMIC_SET_FRAMES} frames")
        boundaries.append(_validated_public_boundary(frames, supplied_boundary))
        if tuple(frame.frame_index for frame in frames) != expected_indices:
            raise ValueError("public frames must be in exact contiguous frame order")

    public_signature = tensor_identity_version_signature(episodes)
    batch_size = len(episodes)
    model.eval()
    model.reset(batch_size=batch_size)
    frame_traces: list[list[BeliefFrameTrace]] = [[] for _ in episodes]
    shared_latencies: list[float] = []
    horizons: tuple[HorizonTrace, ...] | None = None
    rollout_latency: float | None = None
    action_counts = [0] * batch_size
    with torch.no_grad():
        for frame_index in expected_indices:
            frames = tuple(episode[frame_index] for episode in episodes)
            packet = _public_packet_batch(frames)
            try:
                action = _batched_public_action(frames, model.belief)
            except (TypeError, ValueError) as error:
                raise DynamicSetEvaluationError(
                    f"public action batch could not be resolved at frame {frame_index}: {error}"
                ) from error
            if action is not None:
                action_counts = [count + 1 for count in action_counts]
            started = time.perf_counter()
            belief = model.ingest(packet, action=action)
            shared_latencies.append(time.perf_counter() - started)
            extracted = _belief_batch_frame_traces(
                belief,
                model.last_measurements,
                getattr(model, "last_interval_collision_mask", None),
                getattr(model, "last_interval_pair_collision_logits", None),
                frame_index=frame_index,
                timestamp=frames[0].timestamp,
            )
            for batch_index, trace in enumerate(extracted):
                frame_traces[batch_index].append(trace)
            if frame_index == config.horizon_anchor_frame:
                horizons, rollout_latency = _horizon_batch_traces(model, frame_index)
    if horizons is None or rollout_latency is None:
        raise AssertionError("frozen horizon anchor was not evaluated")
    if tensor_identity_version_signature(episodes) != public_signature:
        raise DynamicSetEvaluationError("batched inference mutated its public frame tensors")
    persistent_bytes = _runtime_tensor_bytes(model)
    rss_bytes = _process_rss_bytes()
    return tuple(
        DynamicSetEpisodeTrace(
            frames=tuple(frame_traces[index]),
            horizon=horizons[index],
            # These are measured B-wide call times and deliberately excluded
            # from the B1 resource gate by runtime_batch_size.
            perception_latencies_seconds=tuple(shared_latencies),
            six_horizon_latency_seconds=rollout_latency,
            public_action_count=action_counts[index],
            persistent_tensor_bytes=persistent_bytes,
            process_rss_bytes=rss_bytes,
            runtime_batch_size=batch_size,
            public_boundary_evidence=boundaries[index],
        )
        for index in range(batch_size)
    )


def _hungarian_pairs(
    predicted: Tensor,
    truth: Tensor,
    *,
    maximum_distance: float,
) -> tuple[tuple[int, int], ...]:
    if predicted.ndim != 2 or truth.ndim != 2 or predicted.shape[-1] != 3 or truth.shape[-1] != 3:
        raise ValueError("Hungarian positions must have shape [N,3]")
    if predicted.shape[0] == 0 or truth.shape[0] == 0:
        return ()
    distance = torch.cdist(predicted.to(torch.float64), truth.to(torch.float64), p=2)
    rows, columns = linear_sum_assignment(np.asarray(distance))
    return tuple(
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if float(distance[row, column]) <= maximum_distance
    )


def _frame_associations(
    frame: BeliefFrameTrace,
    truth_active: Tensor,
    truth_position: Tensor,
    *,
    maximum_distance: float,
) -> tuple[tuple[int, int], ...]:
    predicted_indices = torch.nonzero(frame.active, as_tuple=False).flatten()
    truth_indices = torch.nonzero(truth_active, as_tuple=False).flatten()
    local = _hungarian_pairs(
        frame.position.index_select(0, predicted_indices),
        truth_position.index_select(0, truth_indices),
        maximum_distance=maximum_distance,
    )
    return tuple(
        (int(predicted_indices[predicted]), int(truth_indices[target]))
        for predicted, target in local
    )


def _proposal_counts(
    accumulator: DynamicSetCellAccumulator,
    frame: BeliefFrameTrace,
    truth_active: Tensor,
    truth_position: Tensor,
    config: DynamicSetEvaluationConfig,
) -> None:
    eligible = frame.proposal.valid_mask & (
        frame.proposal.existence_probability >= config.proposal_existence_threshold
    )
    predicted_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    truth_indices = torch.nonzero(truth_active, as_tuple=False).flatten()
    matches = _hungarian_pairs(
        frame.proposal.position.index_select(0, predicted_indices),
        truth_position.index_select(0, truth_indices),
        maximum_distance=config.proposal_match_distance_m,
    )
    true_positive = len(matches)
    accumulator.proposal.add(
        true_positive=true_positive,
        false_positive=int(predicted_indices.numel()) - true_positive,
        false_negative=int(truth_indices.numel()) - true_positive,
    )
    accumulator.exact_count.add(
        int(predicted_indices.numel() == truth_indices.numel()),
        1,
    )


def _coverage_update(
    accumulator: DynamicSetCellAccumulator,
    error: Tensor,
    log_variance: Tensor,
    *,
    z: float,
) -> None:
    error64 = error.detach().to(torch.float64)
    variance64 = log_variance.detach().to(torch.float64).exp()
    if error64.shape != variance64.shape:
        raise ValueError("uncertainty and state error shapes differ")
    if not bool(torch.isfinite(error64).all()) or not bool(torch.isfinite(variance64).all()):
        raise DynamicSetEvaluationError("uncertainty coverage received nonfinite evidence")
    covered = error64.abs() <= z * variance64.sqrt()
    accumulator.uncertainty_90.add(int(covered.sum()), covered.numel())


def _truth_segment_metadata(episode: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    objects = episode["objects"]
    events = episode["events"]
    active = objects["active"]
    object_id = objects["id"]
    created = events["created"]
    collision = events["collision"]
    action = events["known_action_observed"]
    counts = torch.zeros_like(object_id, dtype=torch.int64)
    post_event = torch.zeros_like(active)
    previous: dict[int, tuple[int, bool]] = {}
    for frame_index in range(DYNAMIC_SET_FRAMES):
        for slot in torch.nonzero(active[frame_index], as_tuple=False).flatten().tolist():
            identity = int(object_id[frame_index, slot])
            new_identity = identity not in previous
            birth_event = bool(frame_index > 0 and created[frame_index, slot])
            reset = bool(birth_event or collision[frame_index, slot] or action[frame_index, slot])
            old_count, old_had_event = previous.get(identity, (0, False))
            # Runtime births are tentative on their first visible frame.  The
            # persistent-ID history segment begins with sample one only on the
            # second observation that confirms the birth.  Contact/action
            # resets for an existing ID still begin on their actual event
            # frame, matching the estimator's runtime segment reset.
            if new_identity:
                count = 0
                had_event = reset
            else:
                count = 1 if reset or old_count == 0 else old_count + 1
                had_event = old_had_event or reset
            counts[frame_index, slot] = count
            post_event[frame_index, slot] = had_event
            previous[identity] = (count, had_event)
    return counts, post_event


def _nearest_event_timing(
    truth_frames: Sequence[int], predicted_frames: Sequence[int]
) -> tuple[float, int]:
    """Match events one-to-one and penalize every missed truth event.

    Timing support is truth-indexed.  Collision F1 already accounts for false
    positives; the full-episode miss penalty prevents a checkpoint from making
    its timing mean look perfect by detecting only the easiest events.
    """

    truth = tuple(sorted(int(frame) for frame in truth_frames))
    predicted = tuple(sorted(int(frame) for frame in predicted_frames))
    # Monotone minimum-cost matching is sufficient on a one-dimensional time
    # axis.  Unlike a greedy nearest-neighbour pass, this cannot consume the
    # only useful late prediction for an earlier truth event and then charge a
    # larger miss penalty for the event it actually identifies.
    previous = [0.0] * (len(predicted) + 1)
    for truth_index, truth_frame in enumerate(truth, start=1):
        current = [truth_index * _MISSING_EVENT_TIMING_PENALTY_FRAMES]
        for predicted_index, predicted_frame in enumerate(predicted, start=1):
            current.append(
                min(
                    current[predicted_index - 1],
                    previous[predicted_index] + _MISSING_EVENT_TIMING_PENALTY_FRAMES,
                    previous[predicted_index - 1] + abs(predicted_frame - truth_frame),
                )
            )
        previous = current
    return previous[-1], len(truth)


def _score_lifecycle(
    accumulator: DynamicSetCellAccumulator,
    trace: DynamicSetEpisodeTrace,
    episode: Mapping[str, Any],
    associations: Sequence[tuple[tuple[int, int], ...]],
) -> None:
    objects = episode["objects"]
    events = episode["events"]
    truth_active = objects["active"]
    truth_ids = objects["id"]
    initial_truth_ids = {int(value) for value in truth_ids[0, truth_active[0]].tolist()}
    truth_births: dict[int, int] = {}
    truth_removals: dict[int, int] = {}
    for frame_index in range(1, DYNAMIC_SET_FRAMES):
        for slot in (
            torch.nonzero(events["created"][frame_index], as_tuple=False).flatten().tolist()
        ):
            truth_births[int(truth_ids[frame_index, slot])] = frame_index
        for slot in (
            torch.nonzero(events["removed"][frame_index], as_tuple=False).flatten().tolist()
        ):
            truth_removals[int(truth_ids[frame_index - 1, slot])] = frame_index

    runtime_frames: dict[int, list[int]] = {}
    runtime_truth_votes: dict[int, list[int]] = {}
    for frame_index, frame in enumerate(trace.frames):
        for slot in torch.nonzero(frame.active, as_tuple=False).flatten().tolist():
            runtime_frames.setdefault(int(frame.object_id[slot]), []).append(frame_index)
        for predicted_slot, truth_slot in associations[frame_index]:
            runtime_id = int(frame.object_id[predicted_slot])
            truth_id = int(truth_ids[frame_index, truth_slot])
            runtime_truth_votes.setdefault(runtime_id, []).append(truth_id)

    runtime_to_truth: dict[int, int] = {}
    for runtime_id, votes in runtime_truth_votes.items():
        counts: dict[int, int] = {}
        for truth_id in votes:
            counts[truth_id] = counts.get(truth_id, 0) + 1
        runtime_to_truth[runtime_id] = min(counts, key=lambda value: (-counts[value], value))

    predicted_births: list[tuple[int, int | None]] = []
    for runtime_id, frame_indices in runtime_frames.items():
        mapped = runtime_to_truth.get(runtime_id)
        if mapped in initial_truth_ids:
            continue
        first = min(frame_indices)
        # Frame-one allocation is the confirmation of the initial visible set;
        # an unassociated ID there is still a false proposal, not a lifecycle birth.
        if first > 1 or mapped in truth_births:
            predicted_births.append((first, mapped))
    unmatched_births = set(truth_births)
    birth_tp = 0
    birth_fp = 0
    for predicted_frame, mapped in sorted(predicted_births):
        if mapped not in unmatched_births:
            birth_fp += 1
            continue
        truth_frame = truth_births[mapped]
        confirmation_frame = truth_frame + 1
        if predicted_frame < confirmation_frame:
            birth_fp += 1
            continue
        birth_tp += 1
        unmatched_births.remove(mapped)
        accumulator.birth_latency.add(float(predicted_frame - confirmation_frame))
    accumulator.birth.add(
        true_positive=birth_tp,
        false_positive=birth_fp,
        false_negative=len(unmatched_births),
    )
    accumulator.birth_latency.add(
        _MISSING_BIRTH_LATENCY_PENALTY_FRAMES * len(unmatched_births),
        count=len(unmatched_births),
    )

    predicted_removals: list[tuple[int, int | None]] = []
    for runtime_id, frame_indices in runtime_frames.items():
        active_set = set(frame_indices)
        for frame_index in frame_indices:
            removal_frame = frame_index + 1
            if removal_frame >= DYNAMIC_SET_FRAMES:
                continue
            if removal_frame not in active_set:
                predicted_removals.append((removal_frame, runtime_to_truth.get(runtime_id)))
    unmatched_removals = set(truth_removals)
    removal_tp = 0
    removal_fp = 0
    for predicted_frame, mapped in sorted(predicted_removals):
        if mapped not in unmatched_removals:
            removal_fp += 1
            continue
        truth_frame = truth_removals[mapped]
        if predicted_frame < truth_frame:
            removal_fp += 1
            continue
        removal_tp += 1
        unmatched_removals.remove(mapped)
        accumulator.removal_latency.add(float(predicted_frame - truth_frame))
    accumulator.removal.add(
        true_positive=removal_tp,
        false_positive=removal_fp,
        false_negative=len(unmatched_removals),
    )
    accumulator.removal_latency.add(
        _MISSING_REMOVAL_LATENCY_PENALTY_FRAMES * len(unmatched_removals),
        count=len(unmatched_removals),
    )


def score_dynamic_set_trace(
    trace: DynamicSetEpisodeTrace,
    materialization: DynamicSetMaterialization,
    *,
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
) -> DynamicSetCellAccumulator:
    """Read private truth only after public inference has completed."""

    config.validate()
    if not isinstance(materialization, DynamicSetMaterialization):
        raise TypeError("private scoring requires a DynamicSetMaterialization")
    episode = materialization.episode
    objects = episode["objects"]
    events = episode["events"]
    if len(trace.frames) != DYNAMIC_SET_FRAMES:
        raise ValueError("trace and truth frame counts differ")
    truth_active = objects["active"].detach().cpu()
    truth_ids = objects["id"].detach().cpu()
    truth_position = objects["position"].detach().cpu()
    truth_velocity = objects["velocity"].detach().cpu()
    pair_collision_truth = events.get("pair_collision")
    expected_pair_truth = (
        DYNAMIC_SET_FRAMES,
        DYNAMIC_SET_MAX_OBJECTS,
        DYNAMIC_SET_MAX_OBJECTS,
    )
    if (
        not isinstance(pair_collision_truth, Tensor)
        or pair_collision_truth.shape != expected_pair_truth
        or pair_collision_truth.dtype is not torch.bool
        or not torch.equal(pair_collision_truth, pair_collision_truth.transpose(1, 2))
    ):
        raise DynamicSetEvaluationError(
            "private pair collision truth must be symmetric boolean [56,6,6]"
        )
    pair_collision_truth = pair_collision_truth.detach().cpu()
    segment_count, segment_post_event = _truth_segment_metadata(episode)
    segment_count = segment_count.cpu()
    segment_post_event = segment_post_event.cpu()
    truth_action_count = int(events["known_action_observed"].sum())
    if trace.public_action_count != truth_action_count:
        raise DynamicSetEvaluationError("public action trace and private action ledger disagree")
    accumulator = DynamicSetCellAccumulator(episode_count=1)
    associations: list[tuple[tuple[int, int], ...]] = []
    expected_runtime_by_truth: dict[int, int] = {}
    previous_runtime_by_truth: dict[int, int] = {}
    truth_collision_frames: dict[tuple[int, int], list[int]] = {}
    predicted_collision_frames: dict[tuple[int, int], list[int]] = {}
    truth_first_visible_frame: dict[int, int] = {}
    for frame_index in range(DYNAMIC_SET_FRAMES):
        for truth_slot in (
            torch.nonzero(truth_active[frame_index], as_tuple=False).flatten().tolist()
        ):
            truth_first_visible_frame.setdefault(
                int(truth_ids[frame_index, truth_slot]),
                frame_index,
            )

    for frame_index, frame in enumerate(trace.frames):
        _proposal_counts(
            accumulator,
            frame,
            truth_active[frame_index],
            truth_position[frame_index],
            config,
        )
        pairs = _frame_associations(
            frame,
            truth_active[frame_index],
            truth_position[frame_index],
            maximum_distance=config.belief_match_distance_m,
        )
        associations.append(pairs)
        predicted_to_truth = dict(pairs)
        truth_to_predicted = {truth_slot: predicted_slot for predicted_slot, truth_slot in pairs}
        runtime_by_truth_slot: dict[int, int] = {}
        active_truth_slots = (
            torch.nonzero(truth_active[frame_index], as_tuple=False).flatten().tolist()
        )
        packing_position = slice(0, 3)
        packing_velocity = slice(3, 6)
        for truth_slot in active_truth_slots:
            truth_id = int(truth_ids[frame_index, truth_slot])
            first_visible_frame = truth_first_visible_frame[truth_id]
            # Proposals are scored from the first visible frame, but the
            # persistent-state interface deliberately requires two visible
            # observations before confirming a birth.  State, identity, and
            # uncertainty support therefore begins on the confirmation frame;
            # otherwise every conforming checkpoint has an unavoidable
            # missing-state error on frame zero and on each lifecycle birth.
            if frame_index - first_visible_frame + 1 < _STATE_BIRTH_CONFIRMATION_OBSERVATIONS:
                continue
            predicted_slot = truth_to_predicted.get(truth_slot)
            sample_count = int(segment_count[frame_index, truth_slot])
            if predicted_slot is None:
                # State-error and calibration support is truth-indexed. A
                # checkpoint may not improve RMSE or coverage by withholding a
                # difficult object that is still present in private truth.
                accumulator.current_position.add_missing(
                    euclidean_penalty=config.belief_match_distance_m
                )
                accumulator.uncertainty_90.add(0, 3)
                if sample_count >= 16:
                    accumulator.mature_velocity.add_missing(
                        euclidean_penalty=_MISSING_VELOCITY_PENALTY_MPS
                    )
                    accumulator.uncertainty_90.add(0, 3)
                elif sample_count >= 3 and bool(segment_post_event[frame_index, truth_slot]):
                    accumulator.post_event_velocity.add_missing(
                        euclidean_penalty=_MISSING_VELOCITY_PENALTY_MPS
                    )
                    accumulator.uncertainty_90.add(0, 3)
                continue
            runtime_id = int(frame.object_id[predicted_slot])
            runtime_by_truth_slot[truth_slot] = runtime_id
            expected_runtime_by_truth.setdefault(truth_id, runtime_id)
            if truth_id in previous_runtime_by_truth:
                accumulator.identity_switch.add(
                    int(runtime_id != previous_runtime_by_truth[truth_id]), 1
                )
            previous_runtime_by_truth[truth_id] = runtime_id

            position_error = (
                frame.position[predicted_slot] - truth_position[frame_index, truth_slot]
            )
            velocity_error = (
                frame.velocity[predicted_slot] - truth_velocity[frame_index, truth_slot]
            )
            accumulator.current_position.add_tensor(position_error)
            _coverage_update(
                accumulator,
                position_error,
                frame.fast_log_variance[predicted_slot, packing_position],
                z=config.uncertainty_z,
            )
            if sample_count >= 16:
                accumulator.mature_velocity.add_tensor(velocity_error)
                _coverage_update(
                    accumulator,
                    velocity_error,
                    frame.fast_log_variance[predicted_slot, packing_velocity],
                    z=config.uncertainty_z,
                )
            elif sample_count >= 3 and bool(segment_post_event[frame_index, truth_slot]):
                accumulator.post_event_velocity.add_tensor(velocity_error)
                _coverage_update(
                    accumulator,
                    velocity_error,
                    frame.fast_log_variance[predicted_slot, packing_velocity],
                    z=config.uncertainty_z,
                )

        for truth_slot in active_truth_slots:
            truth_id = int(truth_ids[frame_index, truth_slot])
            # Two observations establish an initial ID or a later birth. The
            # proposal/count metrics still score the first visible frame, but
            # persistent identity begins only once confirmation can exist.
            # Every subsequent active frame remains supported, so an ID that
            # is still absent after confirmation is a failure.
            if frame_index <= truth_first_visible_frame[truth_id]:
                continue
            runtime_id = runtime_by_truth_slot.get(truth_slot)
            accumulator.persistent_identity.add(
                int(
                    runtime_id is not None and runtime_id == expected_runtime_by_truth.get(truth_id)
                ),
                1,
            )

        probability = frame.interval_pair_collision_probability
        expected_probability = (
            DYNAMIC_SET_MAX_OBJECTS,
            DYNAMIC_SET_MAX_OBJECTS,
        )
        if (
            probability.shape != expected_probability
            or not probability.is_floating_point()
            or not bool(torch.isfinite(probability).all())
            or bool(torch.any((probability < 0.0) | (probability > 1.0)))
            or not torch.equal(probability, probability.transpose(0, 1))
        ):
            raise DynamicSetEvaluationError(
                "trace pair collision confidence must be finite symmetric [6,6] probabilities"
            )
        predicted_pair_collision = probability >= config.collision_probability_threshold
        active_predicted = torch.nonzero(frame.active, as_tuple=False).flatten().tolist()
        paired_truth_slots: set[int] = set()
        for first_index, predicted_first in enumerate(active_predicted):
            for predicted_second in active_predicted[first_index + 1 :]:
                predicted_collision = bool(
                    predicted_pair_collision[predicted_first, predicted_second]
                )
                truth_first = predicted_to_truth.get(predicted_first)
                truth_second = predicted_to_truth.get(predicted_second)
                if truth_first is None or truth_second is None:
                    if predicted_collision:
                        accumulator.collision.add(false_positive=1)
                    continue
                paired_truth_slots.update((truth_first, truth_second))
                actual_collision = bool(
                    pair_collision_truth[frame_index, truth_first, truth_second]
                )
                accumulator.collision.add(
                    true_positive=int(predicted_collision and actual_collision),
                    false_positive=int(predicted_collision and not actual_collision),
                    false_negative=int(actual_collision and not predicted_collision),
                )
                truth_pair = tuple(
                    sorted(
                        (
                            int(truth_ids[frame_index, truth_first]),
                            int(truth_ids[frame_index, truth_second]),
                        )
                    )
                )
                if actual_collision:
                    truth_collision_frames.setdefault(truth_pair, []).append(frame_index)
                if predicted_collision:
                    predicted_collision_frames.setdefault(truth_pair, []).append(frame_index)

        active_truth = (
            torch.nonzero(
                truth_active[frame_index],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        for first_index, truth_first in enumerate(active_truth):
            for truth_second in active_truth[first_index + 1 :]:
                if truth_first in paired_truth_slots and truth_second in paired_truth_slots:
                    continue
                if not bool(pair_collision_truth[frame_index, truth_first, truth_second]):
                    continue
                accumulator.collision.add(false_negative=1)
                truth_pair = tuple(
                    sorted(
                        (
                            int(truth_ids[frame_index, truth_first]),
                            int(truth_ids[frame_index, truth_second]),
                        )
                    )
                )
                truth_collision_frames.setdefault(truth_pair, []).append(frame_index)

    for truth_pair, event_frames in truth_collision_frames.items():
        timing_sum, support = _nearest_event_timing(
            event_frames, predicted_collision_frames.get(truth_pair, ())
        )
        accumulator.collision_timing.add(timing_sum, count=support)

    anchor = trace.horizon.anchor_frame
    anchor_pairs = associations[anchor]
    anchor_truth_to_predicted = {
        truth_slot: predicted_slot for predicted_slot, truth_slot in anchor_pairs
    }
    anchor_truth_slots = torch.nonzero(truth_active[anchor], as_tuple=False).flatten().tolist()
    timestamps = episode["timestamps"].detach().cpu()
    public_actions = events["known_action_observed"].detach().cpu().any(dim=-1)
    fast_position = slice(0, 3)
    fast_velocity = slice(3, 6)
    expected_horizon_timestamps = timestamps[anchor] + timestamps.new_tensor(HORIZONS_SECONDS)
    if not torch.allclose(
        trace.horizon.timestamps,
        expected_horizon_timestamps,
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise DynamicSetEvaluationError("rollout timestamps differ from the six fixed horizons")
    for horizon_index, horizon_seconds in enumerate(HORIZONS_SECONDS):
        target_time = float(timestamps[anchor]) + horizon_seconds
        target_frame = int(round(target_time * 20.0))
        if target_frame >= DYNAMIC_SET_FRAMES or not math.isclose(
            float(timestamps[target_frame]), target_time, rel_tol=0.0, abs_tol=2.0e-6
        ):
            raise DynamicSetEvaluationError("horizon does not land on the fixed 20 Hz truth grid")
        # A future action has not entered the public stream at this anchor.  It
        # is therefore censored rather than silently supplied from truth.
        if bool(public_actions[anchor + 1 : target_frame + 1].any()):
            continue
        for anchor_truth_slot in anchor_truth_slots:
            truth_id = int(truth_ids[anchor, anchor_truth_slot])
            target_slots = torch.nonzero(
                truth_active[target_frame] & (truth_ids[target_frame] == truth_id),
                as_tuple=False,
            ).flatten()
            if target_slots.numel() != 1:
                continue
            target_slot = int(target_slots[0])
            predicted_slot = anchor_truth_to_predicted.get(anchor_truth_slot)
            if predicted_slot is None:
                accumulator.horizon_position[horizon_seconds].add_missing(
                    euclidean_penalty=config.belief_match_distance_m
                )
                accumulator.horizon_velocity[horizon_seconds].add_missing(
                    euclidean_penalty=_MISSING_VELOCITY_PENALTY_MPS
                )
                accumulator.uncertainty_90.add(0, 6)
                continue
            position_error = (
                trace.horizon.positions[horizon_index, predicted_slot]
                - truth_position[target_frame, target_slot]
            )
            velocity_error = (
                trace.horizon.velocities[horizon_index, predicted_slot]
                - truth_velocity[target_frame, target_slot]
            )
            accumulator.horizon_position[horizon_seconds].add_tensor(position_error)
            accumulator.horizon_velocity[horizon_seconds].add_tensor(velocity_error)
            _coverage_update(
                accumulator,
                position_error,
                trace.horizon.fast_log_variance[horizon_index, predicted_slot, fast_position],
                z=config.uncertainty_z,
            )
            _coverage_update(
                accumulator,
                velocity_error,
                trace.horizon.fast_log_variance[horizon_index, predicted_slot, fast_velocity],
                z=config.uncertainty_z,
            )

    _score_lifecycle(accumulator, trace, episode, associations)
    return accumulator


def _learned_weight_bytes(model: nn.Module) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        key = (parameter.device.type, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def _runtime_tensor_bytes(model: nn.Module) -> int:
    roots = (
        getattr(model, "state", None),
        getattr(model, "last_measurements", None),
        getattr(model, "last_direct_velocity_evidence", None),
        getattr(model, "_last_interval_collision_mask", None),
        getattr(model, "_last_interval_pair_collision_logits", None),
        getattr(model, "diagnostics", None),
    )
    seen_containers: set[int] = set()
    seen_storage: set[tuple[str, int]] = set()
    total = 0

    def visit(value: object) -> None:
        nonlocal total
        if isinstance(value, Tensor):
            storage = value.untyped_storage()
            key = (value.device.type, storage.data_ptr())
            if key not in seen_storage:
                seen_storage.add(key)
                total += storage.nbytes()
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
            return
        identity = id(value)
        if identity in seen_containers:
            return
        seen_containers.add(identity)
        if is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                visit(getattr(value, item.name))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)

    for root in roots:
        visit(root)
    return total


def _process_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other supported CI Unix platforms
    # report KiB.
    return peak if sys.platform == "darwin" else peak * 1024


def _normalized_rmse(sources: Sequence[tuple[SquaredErrorSum, float]]) -> float:
    squared = 0.0
    count = 0
    for source, scale in sources:
        if source.coordinate_count == 0:
            continue
        squared += source.squared_error / (scale * scale)
        count += source.coordinate_count
    if count == 0:
        raise DynamicSetEvaluationError("selection score component has no support")
    return math.sqrt(squared / count)


def _score_components(total: DynamicSetCellAccumulator) -> SelectionScoreEvidence:
    proposal_f1 = total.proposal.f1()
    exact_count = total.exact_count.value()
    identity_accuracy = total.persistent_identity.value()
    switch_rate = total.identity_switch.value()
    collision_f1 = total.collision.f1()
    collision_timing = total.collision_timing.mean()
    components: dict[str, float] = {}
    if total.current_position.coordinate_count:
        components["current_position"] = _normalized_rmse(((total.current_position, 0.010),))
    if total.mature_velocity.coordinate_count or total.post_event_velocity.coordinate_count:
        components["current_velocity"] = _normalized_rmse(
            ((total.mature_velocity, 0.020), (total.post_event_velocity, 0.050))
        )
    if any(item.coordinate_count for item in total.horizon_position.values()):
        components["horizon_position"] = _normalized_rmse(
            tuple(
                (total.horizon_position[horizon], HORIZON_POSITION_LIMITS_M[horizon])
                for horizon in HORIZONS_SECONDS
            )
        )
    if any(item.coordinate_count for item in total.horizon_velocity.values()):
        components["horizon_velocity"] = _normalized_rmse(
            tuple(
                (total.horizon_velocity[horizon], HORIZON_VELOCITY_SCORE_SCALE_MPS)
                for horizon in HORIZONS_SECONDS
            )
        )
    if proposal_f1 is not None and exact_count is not None:
        components["proposal_error"] = (
            (1.0 - float(proposal_f1)) + (1.0 - float(exact_count))
        ) / 2.0
    birth_f1 = total.birth.f1()
    removal_f1 = total.removal.f1()
    if (
        identity_accuracy is not None
        and switch_rate is not None
        and birth_f1 is not None
        and removal_f1 is not None
        and total.birth_latency.count
        and total.removal_latency.count
    ):
        identity_terms = [
            1.0 - float(identity_accuracy),
            float(switch_rate),
            1.0 - birth_f1,
            1.0 - removal_f1,
            float(total.birth_latency.mean()),
            float(total.removal_latency.mean()) / 2.0,
        ]
        components["identity_lifecycle_error"] = sum(identity_terms) / len(identity_terms)
    if collision_f1 is not None and collision_timing is not None:
        components["contact_error"] = ((1.0 - float(collision_f1)) + float(collision_timing)) / 2.0
    if any(not math.isfinite(value) or value < 0.0 for value in components.values()):
        raise DynamicSetEvaluationError("selection components must be finite nonnegative values")
    return SelectionScoreEvidence(components=components)


def merge_dynamic_set_example_score_evidence(
    records: Iterable[DynamicSetExampleScoreEvidence],
) -> DynamicSetCellAccumulator:
    """Rebuild pooled sufficient statistics for a bootstrap resample."""

    total = DynamicSetCellAccumulator()
    count = 0
    for record in records:
        if not isinstance(record, DynamicSetExampleScoreEvidence):
            raise TypeError("paired score evidence must use DynamicSetExampleScoreEvidence")
        record.validate()
        total.merge(record.additive.to_accumulator())
        count += 1
    if count == 0:
        raise ValueError("paired score evidence cannot be empty")
    return total


def pooled_selection_score_evidence(
    records: Iterable[DynamicSetExampleScoreEvidence],
) -> SelectionScoreEvidence:
    """Return the fixed physical score components for one bootstrap sample."""

    return _score_components(merge_dynamic_set_example_score_evidence(records))


def pair_dynamic_set_example_score_evidence(
    candidate: Sequence[DynamicSetExampleScoreEvidence],
    reference: Sequence[DynamicSetExampleScoreEvidence],
) -> tuple[tuple[DynamicSetExampleScoreEvidence, DynamicSetExampleScoreEvidence], ...]:
    """Validate exact row pairing before a paired bootstrap samples indices."""

    candidate_rows = tuple(candidate)
    reference_rows = tuple(reference)
    if len(candidate_rows) != len(reference_rows) or not candidate_rows:
        raise ValueError("candidate/reference per-example evidence lengths differ or are empty")
    pairs: list[tuple[DynamicSetExampleScoreEvidence, DynamicSetExampleScoreEvidence]] = []
    seen: set[tuple[str, int, int, int, str]] = set()
    for candidate_record, reference_record in zip(candidate_rows, reference_rows, strict=True):
        candidate_record.validate()
        reference_record.validate()
        if candidate_record.pair_key != reference_record.pair_key:
            raise ValueError("candidate/reference per-example row bindings differ")
        if candidate_record.public_boundary_evidence != reference_record.public_boundary_evidence:
            raise ValueError("candidate/reference public-boundary evidence differs")
        if candidate_record.pair_key in seen:
            raise ValueError("paired source evidence contains a duplicate row binding")
        seen.add(candidate_record.pair_key)
        pairs.append((candidate_record, reference_record))
    return tuple(pairs)


def dynamic_set_truth_leakage_count(
    records: Iterable[DynamicSetExampleScoreEvidence],
) -> int:
    """Derive the promotion leakage count only from scored boundary evidence."""

    rows = tuple(records)
    if not rows:
        raise ValueError("truth-leakage derivation requires physical row evidence")
    total = 0
    for record in rows:
        if type(record) is not DynamicSetExampleScoreEvidence:
            raise TypeError("truth-leakage rows must use exact score evidence")
        record.validate()
        total += record.public_boundary_evidence.truth_leakage_count
    return total


@dataclass
class _DynamicSetEvaluationAccumulator:
    model: nn.Module
    config: DynamicSetEvaluationConfig
    by_cell: dict[PhysicalCell, DynamicSetCellAccumulator] = field(default_factory=dict)
    per_example: list[DynamicSetExampleScoreEvidence] = field(default_factory=list)
    perception_latencies: list[float] = field(default_factory=list)
    rollout_latencies: list[float] = field(default_factory=list)
    maximum_persistent_bytes: int = 0
    maximum_rss_bytes: int = 0
    episode_count: int = 0

    def add(
        self,
        materialization: DynamicSetMaterialization,
        trace: DynamicSetEpisodeTrace,
    ) -> None:
        # Truth first becomes reachable here, after every output for this
        # episode has been detached and the public model call has returned.
        boundary = trace.public_boundary_evidence
        if type(boundary) is not DynamicSetPublicBoundaryEvidence:
            raise DynamicSetEvaluationError("scored trace lacks exact public-boundary evidence")
        try:
            boundary.require_clean(require_truth_binding=True)
        except (TypeError, ValueError) as error:
            raise DynamicSetEvaluationError(
                f"scored trace has invalid public-boundary evidence: {error}"
            ) from error
        if boundary.row_sha256 != canonical_sha256(asdict(materialization.row)):
            raise DynamicSetEvaluationError(
                "scored trace public boundary differs from its manifest row"
            )
        episode_evidence = score_dynamic_set_trace(trace, materialization, config=self.config)
        cell = PHYSICAL_CELLS[materialization.row.cell_index]
        if (
            cell.object_count,
            cell.contact,
            cell.dynamic_membership,
        ) != (
            materialization.row.object_count,
            materialization.row.contact,
            materialization.row.dynamic_membership,
        ):
            raise DynamicSetEvaluationError("manifest row and physical-cell registry disagree")
        self.by_cell.setdefault(cell, DynamicSetCellAccumulator()).merge(episode_evidence)
        self.per_example.append(
            DynamicSetExampleScoreEvidence.create(
                materialization.row,
                episode_evidence,
                public_boundary_evidence=boundary,
            )
        )
        if (
            isinstance(trace.runtime_batch_size, bool)
            or not isinstance(trace.runtime_batch_size, int)
            or trace.runtime_batch_size <= 0
        ):
            raise DynamicSetEvaluationError("trace runtime_batch_size must be a positive integer")
        if trace.runtime_batch_size == 1:
            self.perception_latencies.extend(trace.perception_latencies_seconds)
            if materialization.row.object_count == DYNAMIC_SET_MAX_OBJECTS:
                self.rollout_latencies.append(trace.six_horizon_latency_seconds)
            self.maximum_persistent_bytes = max(
                self.maximum_persistent_bytes,
                trace.persistent_tensor_bytes,
            )
        # Peak RSS is an evaluator-process bound, so batched calls remain part
        # of this resource gate even though latency/persistent-state gates are B1.
        self.maximum_rss_bytes = max(self.maximum_rss_bytes, trace.process_rss_bytes)
        self.episode_count += 1

    def finish(
        self,
        *,
        require_all_cells: bool,
        population_binding: EvaluationPopulationBinding,
    ) -> DynamicSetEvaluationResult:
        if self.episode_count == 0:
            raise ValueError("dynamic-set evaluation requires at least one materialization")
        if not self.perception_latencies:
            raise DynamicSetEvaluationError("resource evaluation requires B1 perception samples")
        if not self.rollout_latencies:
            raise DynamicSetEvaluationError(
                "resource evaluation requires a B1/N=6 six-horizon rollout"
            )
        if require_all_cells and set(self.by_cell) != set(PHYSICAL_CELLS):
            missing = sorted(set(PHYSICAL_CELLS) - set(self.by_cell))
            raise DynamicSetEvaluationError(f"evaluation lacks physical cells: {missing}")

        total = DynamicSetCellAccumulator()
        for evidence in self.by_cell.values():
            total.merge(evidence)
        metrics_by_cell = {
            cell: evidence.physical_metrics(cell) for cell, evidence in self.by_cell.items()
        }
        resources = ResourceMetrics(
            perception_latency_seconds=float(median(self.perception_latencies)),
            six_horizon_rollout_seconds=float(median(self.rollout_latencies)),
            learned_weight_bytes=_learned_weight_bytes(self.model),
            persistent_tensor_bytes=self.maximum_persistent_bytes,
            process_rss_bytes=self.maximum_rss_bytes,
        )
        score = _score_components(total)
        expected_score_components = set(SELECTION_SCORE_WEIGHTS) - {"planning_error"}
        if require_all_cells and set(score.components) != expected_score_components:
            missing = sorted(expected_score_components - set(score.components))
            raise DynamicSetEvaluationError(
                f"complete validation lacks selection support for: {', '.join(missing)}"
            )
        resolved_binding = EvaluationPopulationBinding(
            split=population_binding.split,
            protocol_sha256=population_binding.protocol_sha256,
            manifest_sha256=population_binding.manifest_sha256,
            permit_index=population_binding.permit_index,
            row_count=self.episode_count,
        )
        return DynamicSetEvaluationResult(
            by_cell=metrics_by_cell,
            evidence_by_cell=self.by_cell,
            score=score,
            resources=resources,
            episode_count=self.episode_count,
            per_example_score_evidence=tuple(self.per_example),
            population_binding=resolved_binding,
        )


def _evaluate_dynamic_set_stream(
    model: OnlineWorldModel,
    materializations: Iterable[DynamicSetMaterialization],
    *,
    config: DynamicSetEvaluationConfig,
    expected_split: str,
    require_all_cells: bool,
    expected_rows: tuple[PhysicalManifestRow, ...] | None,
    population_binding: EvaluationPopulationBinding,
) -> DynamicSetEvaluationResult:
    config.validate()
    accumulator = _DynamicSetEvaluationAccumulator(model=model, config=config)
    for materialization in materializations:
        if not isinstance(materialization, DynamicSetMaterialization):
            raise TypeError("all evaluation inputs must be DynamicSetMaterialization objects")
        if materialization.row.split != expected_split:
            raise ValueError(f"evaluation expected only the {expected_split!r} split")
        if expected_rows is not None:
            if accumulator.episode_count >= len(expected_rows):
                raise DynamicSetEvaluationError("materialization stream exceeds its bound manifest")
            if materialization.row != expected_rows[accumulator.episode_count]:
                raise DynamicSetEvaluationError(
                    f"materialization row {accumulator.episode_count} differs from its bound manifest"
                )
        public_frames, boundary = materialization.public_frames_with_boundary()
        trace = run_public_dynamic_set_episode(
            model,
            public_frames,
            config=config,
            public_boundary_evidence=boundary,
        )
        accumulator.add(materialization, trace)
    if expected_rows is not None and accumulator.episode_count != len(expected_rows):
        raise DynamicSetEvaluationError("materialization stream ended before its bound manifest")
    return accumulator.finish(
        require_all_cells=require_all_cells,
        population_binding=population_binding,
    )


def _evaluate_paired_dynamic_set_stream(
    candidate_model: OnlineWorldModel,
    reference_model: OnlineWorldModel,
    materializations: Iterable[DynamicSetMaterialization],
    *,
    config: DynamicSetEvaluationConfig,
    expected_split: str,
    require_all_cells: bool,
    expected_rows: tuple[PhysicalManifestRow, ...] | None,
    population_binding: EvaluationPopulationBinding,
    batch_size: int,
) -> DynamicSetPairedEvaluationResult:
    config.validate()
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= MAXIMUM_PAIRED_EVALUATION_BATCH_SIZE
    ):
        raise ValueError(
            f"paired physical batch_size must lie in [1,{MAXIMUM_PAIRED_EVALUATION_BATCH_SIZE}]"
        )
    for model in (candidate_model, reference_model):
        if not isinstance(model, nn.Module):
            raise TypeError("paired physical models must be torch.nn.Module objects")
        _validate_cpu_float32_model(model)

    candidate = _DynamicSetEvaluationAccumulator(model=candidate_model, config=config)
    reference = _DynamicSetEvaluationAccumulator(model=reference_model, config=config)
    buffered: list[DynamicSetMaterialization] = []
    buffered_action_signature: tuple[bool, int | None] | None = None
    b1_resource_cells: set[PhysicalCell] = set()
    consumed = 0

    def run_batch(items: Sequence[DynamicSetMaterialization]) -> None:
        if not items:
            return
        public_with_boundaries = tuple(item.public_frames_with_boundary() for item in items)
        public_episodes = tuple(item[0] for item in public_with_boundaries)
        public_boundaries = tuple(item[1] for item in public_with_boundaries)
        public_signature = tensor_identity_version_signature(public_episodes)
        candidate_traces = run_public_dynamic_set_batch(
            candidate_model,
            public_episodes,
            config=config,
            public_boundary_evidence=public_boundaries,
        )
        if tensor_identity_version_signature(public_episodes) != public_signature:
            raise DynamicSetEvaluationError("candidate mutated paired public evidence")
        reference_traces = run_public_dynamic_set_batch(
            reference_model,
            public_episodes,
            config=config,
            public_boundary_evidence=public_boundaries,
        )
        if tensor_identity_version_signature(public_episodes) != public_signature:
            raise DynamicSetEvaluationError("reference mutated paired public evidence")
        if len(candidate_traces) != len(items) or len(reference_traces) != len(items):
            raise DynamicSetEvaluationError("paired runtime returned the wrong trace count")
        # Only after both public passes have returned and detached their traces
        # may the private materialization reach either scorer.
        for item, candidate_trace, reference_trace in zip(
            items,
            candidate_traces,
            reference_traces,
            strict=True,
        ):
            candidate.add(item, candidate_trace)
            reference.add(item, reference_trace)

    def flush() -> None:
        nonlocal buffered, buffered_action_signature
        run_batch(buffered)
        buffered = []
        buffered_action_signature = None

    for materialization in materializations:
        if not isinstance(materialization, DynamicSetMaterialization):
            raise TypeError("all evaluation inputs must be DynamicSetMaterialization objects")
        if materialization.row.split != expected_split:
            raise ValueError(f"evaluation expected only the {expected_split!r} split")
        if expected_rows is not None:
            if consumed >= len(expected_rows):
                raise DynamicSetEvaluationError("materialization stream exceeds its bound manifest")
            if materialization.row != expected_rows[consumed]:
                raise DynamicSetEvaluationError(
                    f"materialization row {consumed} differs from its bound manifest"
                )
        if not 0 <= materialization.row.cell_index < len(PHYSICAL_CELLS):
            raise DynamicSetEvaluationError("manifest row has an invalid physical cell")
        cell = PHYSICAL_CELLS[materialization.row.cell_index]
        action_signature = (
            materialization.row.known_action,
            materialization.row.action_time_stratum,
        )
        # One real B1 observation of every physical cell retains the exact B1
        # latency and persistent-state resource semantics without re-running a
        # row. Remaining homogeneous rows use the bounded throughput batch.
        if cell not in b1_resource_cells:
            flush()
            run_batch((materialization,))
            b1_resource_cells.add(cell)
        else:
            if buffered and (
                len(buffered) >= batch_size or buffered_action_signature != action_signature
            ):
                flush()
            if not buffered:
                buffered_action_signature = action_signature
            buffered.append(materialization)
        consumed += 1
    flush()
    if consumed == 0:
        raise ValueError("dynamic-set evaluation requires at least one materialization")
    if expected_rows is not None and consumed != len(expected_rows):
        raise DynamicSetEvaluationError("materialization stream ended before its bound manifest")
    candidate_result = candidate.finish(
        require_all_cells=require_all_cells,
        population_binding=population_binding,
    )
    reference_result = reference.finish(
        require_all_cells=require_all_cells,
        population_binding=population_binding,
    )
    return DynamicSetPairedEvaluationResult(
        candidate=candidate_result,
        reference=reference_result,
    ).validate()


def evaluate_paired_dynamic_set_materializations(
    candidate_model: OnlineWorldModel,
    reference_model: OnlineWorldModel,
    materializations: Iterable[DynamicSetMaterialization],
    *,
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    allowed_split: Literal["development", "training"] = "development",
    require_all_cells: bool = False,
    batch_size: int = DEFAULT_PAIRED_EVALUATION_BATCH_SIZE,
) -> DynamicSetPairedEvaluationResult:
    """Evaluate a public candidate/reference pair from one bounded row stream."""

    if allowed_split not in {"development", "training"}:
        raise ValueError("only development or disposable training-screen evaluation is public")

    def reject_protected() -> Iterator[DynamicSetMaterialization]:
        for materialization in materializations:
            if not isinstance(materialization, DynamicSetMaterialization):
                raise TypeError("all evaluation inputs must be DynamicSetMaterialization objects")
            if materialization.row.split in _PROTECTED_SPLITS:
                raise PermissionError(
                    "protected dynamic-set populations require the governed opener"
                )
            yield materialization

    return _evaluate_paired_dynamic_set_stream(
        candidate_model,
        reference_model,
        reject_protected(),
        config=config,
        expected_split=allowed_split,
        require_all_cells=require_all_cells,
        expected_rows=None,
        population_binding=EvaluationPopulationBinding(
            split=allowed_split,
            protocol_sha256=None,
            manifest_sha256=None,
            permit_index=None,
            row_count=0,
        ),
        batch_size=batch_size,
    )


def evaluate_dynamic_set_materializations(
    model: OnlineWorldModel,
    materializations: Iterable[DynamicSetMaterialization],
    *,
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    allowed_split: Literal["development", "training"] = "development",
    require_all_cells: bool = False,
) -> DynamicSetEvaluationResult:
    """Evaluate public development/screen rows; protected rows fail closed."""

    if allowed_split not in {"development", "training"}:
        raise ValueError("only development or disposable training-screen evaluation is public")

    def reject_protected() -> Iterable[DynamicSetMaterialization]:
        for materialization in materializations:
            if not isinstance(materialization, DynamicSetMaterialization):
                raise TypeError("all evaluation inputs must be DynamicSetMaterialization objects")
            if materialization.row.split in _PROTECTED_SPLITS:
                raise PermissionError(
                    "protected dynamic-set populations require the governed opener"
                )
            yield materialization

    return _evaluate_dynamic_set_stream(
        model,
        reject_protected(),
        config=config,
        expected_split=allowed_split,
        require_all_cells=require_all_cells,
        expected_rows=None,
        population_binding=EvaluationPopulationBinding(
            split=allowed_split,
            protocol_sha256=None,
            manifest_sha256=None,
            permit_index=None,
            row_count=0,
        ),
    )


def _validated_protected_physical_population(
    *,
    permit: SplitPermit,
    expected_protocol_sha256: str,
    expected_manifest_sha256: str,
    expected_rows: Sequence[PhysicalManifestRow],
    config: DynamicSetEvaluationConfig,
) -> tuple[str, str, tuple[PhysicalManifestRow, ...]]:
    """Validate every public authorization fact before protected generation."""

    if type(permit) is not SplitPermit:
        raise TypeError("protected evaluation requires an exact SplitPermit")
    if config != DEFAULT_EVALUATION_CONFIG:
        raise ValueError("protected physical evaluation requires the frozen evaluator config")
    protocol_sha256 = validated_sha256(expected_protocol_sha256, label="expected_protocol_sha256")
    manifest_sha256 = validated_sha256(expected_manifest_sha256, label="expected_manifest_sha256")
    if permit.split not in _PROTECTED_SPLITS:
        raise PermissionError("split permit is not for a protected dynamic-set split")
    if permit.protocol_sha256 != protocol_sha256:
        raise PermissionError("split permit protocol binding differs")
    if permit.index != _PROTECTED_SPLIT_INDICES[permit.split]:
        raise PermissionError("split permit ordered index differs")
    validated_sha256(permit.nonce, label="split permit nonce")
    rows = tuple(expected_rows)
    if len(rows) != PHYSICAL_SPLIT_SIZES[permit.split]:
        raise ValueError("protected physical rows differ from the frozen split count")
    if any(not isinstance(row, PhysicalManifestRow) or row.split != permit.split for row in rows):
        raise ValueError("protected expected rows differ from the permitted split")
    computed_manifest_sha256 = canonical_sha256([asdict(row) for row in rows])
    if computed_manifest_sha256 != manifest_sha256:
        raise ValueError("protected expected-row digest differs from its manifest binding")
    if FROZEN_PHYSICAL_MANIFEST_SHA256[permit.split] != manifest_sha256:
        raise ValueError("protected manifest digest differs from the source freeze")
    return protocol_sha256, manifest_sha256, rows


def protected_physical_population_claim_binding(
    *,
    split: str,
    protocol_sha256: str,
    manifest_sha256: str,
    row_count: int,
) -> str:
    """Bind a durable ledger claim to one exact physical population."""

    return canonical_sha256(
        {
            "schema": PHYSICAL_POPULATION_CLAIM_PURPOSE,
            "split": split,
            "protocol_sha256": validated_sha256(protocol_sha256, label="physical claim protocol"),
            "manifest_sha256": validated_sha256(manifest_sha256, label="physical claim manifest"),
            "row_count": row_count,
        }
    )


def _claim_protected_physical_population(
    ledger: OrderedSplitLedger,
    permit: SplitPermit,
    *,
    protocol_sha256: str,
    manifest_sha256: str,
    row_count: int,
) -> None:
    if type(ledger) is not OrderedSplitLedger:
        raise TypeError("protected physical evaluation requires its exact durable ledger")
    ledger.claim_active(
        permit,
        purpose=PHYSICAL_POPULATION_CLAIM_PURPOSE,
        binding_sha256=protected_physical_population_claim_binding(
            split=permit.split,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
            row_count=row_count,
        ),
    )


def _protected_physical_materializations(
    rows: tuple[PhysicalManifestRow, ...],
    *,
    permit: SplitPermit,
    protocol_sha256: str,
    manifest_sha256: str,
) -> Iterator[DynamicSetMaterialization]:
    """Own one capability and close it on success, failure, or short consume."""

    capability = _mint_protected_physical_materialization_capability(
        split=permit.split,
        rows=rows,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        permit_index=permit.index,
        permit_nonce=permit.nonce,
    )
    finished = False
    try:
        for row in rows:
            yield _materialize_protected_dynamic_set_episode(row, capability=capability)
        capability.finish()
        finished = True
    finally:
        if not finished:
            capability.close()


def evaluate_authorized_dynamic_set_rows(
    model: OnlineWorldModel,
    *,
    ledger: OrderedSplitLedger,
    permit: SplitPermit,
    expected_protocol_sha256: str,
    expected_manifest_sha256: str,
    expected_rows: Sequence[PhysicalManifestRow],
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    require_all_cells: bool = True,
) -> DynamicSetEvaluationResult:
    """Materialize and evaluate one exact ledger-authorized population.

    No caller-supplied materialization can cross this boundary.  Permit,
    protocol, manifest, row-count, split, and order checks all complete before
    the internal one-shot capability can reach the simulator.
    """

    protocol_sha256, manifest_sha256, rows = _validated_protected_physical_population(
        permit=permit,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_rows=expected_rows,
        config=config,
    )
    _claim_protected_physical_population(
        ledger,
        permit,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        row_count=len(rows),
    )
    return _evaluate_dynamic_set_stream(
        model,
        _protected_physical_materializations(
            rows,
            permit=permit,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
        ),
        config=config,
        expected_split=permit.split,
        require_all_cells=require_all_cells,
        expected_rows=rows,
        population_binding=EvaluationPopulationBinding(
            split=permit.split,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
            permit_index=permit.index,
            row_count=len(rows),
        ),
    )


def evaluate_authorized_paired_dynamic_set_rows(
    candidate_model: OnlineWorldModel,
    reference_model: OnlineWorldModel,
    *,
    ledger: OrderedSplitLedger,
    permit: SplitPermit,
    expected_protocol_sha256: str,
    expected_manifest_sha256: str,
    expected_rows: Sequence[PhysicalManifestRow],
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    require_all_cells: bool = True,
    batch_size: int = DEFAULT_PAIRED_EVALUATION_BATCH_SIZE,
) -> DynamicSetPairedEvaluationResult:
    """Materialize each protected row once for both checkpoint evaluations."""

    protocol_sha256, manifest_sha256, rows = _validated_protected_physical_population(
        permit=permit,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_rows=expected_rows,
        config=config,
    )
    _claim_protected_physical_population(
        ledger,
        permit,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        row_count=len(rows),
    )
    return _evaluate_paired_dynamic_set_stream(
        candidate_model,
        reference_model,
        _protected_physical_materializations(
            rows,
            permit=permit,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
        ),
        config=config,
        expected_split=permit.split,
        require_all_cells=require_all_cells,
        expected_rows=rows,
        population_binding=EvaluationPopulationBinding(
            split=permit.split,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
            permit_index=permit.index,
            row_count=len(rows),
        ),
        batch_size=batch_size,
    )


def evaluate_authorized_dynamic_set_materializations(
    model: OnlineWorldModel,
    materializations: Iterable[DynamicSetMaterialization],
    **kwargs: Any,
) -> DynamicSetEvaluationResult:
    """Reject the obsolete caller-owned protected materialization boundary."""

    del model, materializations, kwargs
    raise PermissionError(
        "caller-supplied protected physical materializations are prohibited; "
        "use evaluate_authorized_dynamic_set_rows"
    )


def evaluate_authorized_paired_dynamic_set_materializations(
    candidate_model: OnlineWorldModel,
    reference_model: OnlineWorldModel,
    materializations: Iterable[DynamicSetMaterialization],
    **kwargs: Any,
) -> DynamicSetPairedEvaluationResult:
    """Reject the obsolete caller-owned paired protected boundary."""

    del candidate_model, reference_model, materializations, kwargs
    raise PermissionError(
        "caller-supplied protected physical materializations are prohibited; "
        "use evaluate_authorized_paired_dynamic_set_rows"
    )


def load_dynamic_set_evaluation_model(
    checkpoint_path: str | Path,
    config: OrpheusConfig,
) -> tuple[OnlineWorldModel, Mapping[str, Any]]:
    """Load one trusted local checkpoint into the frozen CPU inference graph."""

    if not isinstance(config, OrpheusConfig):
        raise TypeError("config must be an OrpheusConfig")
    model = OnlineWorldModel.from_config(config, device="cpu")
    payload = load_checkpoint(
        checkpoint_path,
        model=model,
        map_location="cpu",
        restore_rng=False,
        expected_config=config,
    )
    if payload["specification_version"] != SPECIFICATION_VERSION:
        raise ValueError("checkpoint is not bound to specification 1.61")
    if payload["simulator_version"] != SIMULATOR_VERSION:
        raise ValueError("checkpoint simulator version differs from the frozen protocol")
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    return model, payload


def make_dynamic_set_validation_hook(
    rows: Sequence[PhysicalManifestRow],
    *,
    config: DynamicSetEvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    materializer: Any = materialize_dynamic_set_episode,
) -> Any:
    """Bind small manifest rows and stream materializations during validation."""

    frozen = tuple(rows)
    if not frozen or any(
        not isinstance(row, PhysicalManifestRow) or row.split != "development" for row in frozen
    ):
        raise ValueError("a validation hook must bind only fixed development rows")
    if not callable(materializer):
        raise TypeError("materializer must be callable")

    def validation_hook(model: nn.Module, step: int) -> Mapping[str, float]:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("validation step must be a nonnegative integer")
        if not isinstance(model, OnlineWorldModel):
            raise TypeError("dynamic-set validation requires OnlineWorldModel")
        result = evaluate_dynamic_set_materializations(
            model,
            (materializer(row) for row in frozen),
            config=config,
            allowed_split="development",
            require_all_cells=True,
        )
        metrics = result.flat_metrics(include_resources=False)
        metrics["validation_step"] = float(step)
        return metrics

    return validation_hook


def screen_snapshot_from_result(
    result: DynamicSetEvaluationResult,
    *,
    optimization_objective: float,
) -> Any:
    """Build the trainer's exact 64-example disposable-screen snapshot."""

    from world_model.training.dynamic_set_trainer import DynamicSetScreenSnapshot

    if result.episode_count != 64:
        raise ValueError("the disposable screen requires exactly 64 evaluated examples")
    if not math.isfinite(optimization_objective):
        raise ValueError("screen optimization objective must be finite")
    total = DynamicSetCellAccumulator()
    for evidence in result.evidence_by_cell.values():
        total.merge(evidence)
    proposal_f1 = total.proposal.f1()
    collision_f1 = total.collision.f1()
    if proposal_f1 is None or collision_f1 is None:
        raise DynamicSetEvaluationError("screen proposal/collision F1 lacks support")
    return DynamicSetScreenSnapshot(
        example_count=64,
        optimization_objective=float(optimization_objective),
        proposal_f1=proposal_f1,
        collision_f1=collision_f1,
    )


def _flatten_supported(output: dict[str, float], name: str, metric: SupportedScalar | None) -> None:
    if metric is None:
        return
    metric.validate(name=name)
    output[f"{name}/support"] = float(metric.support)
    if metric.support > 0:
        output[name] = float(metric.value)


def _flatten_cell_metrics(
    output: dict[str, float], prefix: str, metrics: PhysicalCellMetrics
) -> None:
    for name in (
        "proposal_precision",
        "proposal_recall",
        "proposal_f1",
        "exact_count_accuracy",
        "current_position_rmse_m",
        "mature_velocity_rmse_mps",
        "post_event_velocity_rmse_mps",
        "collision_f1",
        "collision_timing_error_frames",
        "persistent_id_accuracy",
        "identity_switch_rate",
        "birth_precision",
        "birth_recall",
        "birth_latency_frames",
        "removal_precision",
        "removal_recall",
        "removal_latency_frames",
        "uncertainty_90_coverage",
    ):
        _flatten_supported(output, f"{prefix}/{name}", getattr(metrics, name))
    for horizon, metric in metrics.horizon_position_rmse_m.items():
        _flatten_supported(
            output,
            f"{prefix}/horizon_{horizon:.2f}_position_rmse_m",
            metric,
        )


__all__ = [
    "BeliefFrameTrace",
    "BinaryCounts",
    "DynamicSetAdditiveSnapshot",
    "DynamicSetCellAccumulator",
    "DynamicSetEpisodeTrace",
    "DEFAULT_EVALUATION_CONFIG",
    "DynamicSetExampleScoreEvidence",
    "DynamicSetEvaluationConfig",
    "DynamicSetEvaluationError",
    "DynamicSetEvaluationResult",
    "DynamicSetPairedEvaluationResult",
    "DEFAULT_PAIRED_EVALUATION_BATCH_SIZE",
    "EvaluationPopulationBinding",
    "HORIZONS_SECONDS",
    "HORIZON_VELOCITY_SCORE_SCALE_MPS",
    "HorizonTrace",
    "ProposalFrameTrace",
    "RatioCounts",
    "SelectionScoreEvidence",
    "SquaredErrorSum",
    "SumCount",
    "evaluate_authorized_dynamic_set_rows",
    "evaluate_authorized_paired_dynamic_set_rows",
    "evaluate_dynamic_set_materializations",
    "evaluate_paired_dynamic_set_materializations",
    "dynamic_set_truth_leakage_count",
    "load_dynamic_set_evaluation_model",
    "make_dynamic_set_validation_hook",
    "merge_dynamic_set_example_score_evidence",
    "pair_dynamic_set_example_score_evidence",
    "pooled_selection_score_evidence",
    "run_public_dynamic_set_episode",
    "run_public_dynamic_set_batch",
    "score_dynamic_set_trace",
    "screen_snapshot_from_result",
]
