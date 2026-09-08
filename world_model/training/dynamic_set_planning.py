"""Public task construction and private scoring for specification 1.61 planning.

The public half of this module first freezes one already-authorised
``PlanningManifestRow`` against ledger-owned public reference data.  That
checkpoint-independent template owns the observable handle, world goal,
query times, and complete candidate action bank.  A separate binding step may
then resolve the handle in a mature checkpoint belief, but it may substitute
only that belief's persistent target ID.  It does not enumerate manifests,
open split data, read simulator state, or define an optimisation loss.

The evaluator-only half binds independently supplied oracle costs and event
labels to the exact public template digest.  Oracle evidence is used only after the
model's serial and vectorised plans have completed; it is never passed to the
dynamics model or planner.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor

from world_model.belief import BeliefTrajectory, MotionMode, WorldBelief
from world_model.dynamics import AnalyticKinematics, WorldImpulseAction
from world_model.planning import (
    CounterfactualCostWeights,
    CounterfactualPlanResult,
    TerminalWorldPositionGoal,
    plan_counterfactual_actions,
    resolve_appearance_handle,
)
from world_model.runtime.prepared import tensor_identity_version_signature
from world_model.training.dynamic_set_gates import (
    PlanningInvariantMetrics,
    PlanningPopulationGateMetrics,
    PlanningSliceMetrics,
    SupportedScalar,
)
from world_model.training.dynamic_set_protocol import (
    PlanningManifestRow,
    PlanningOracleCertificate,
    canonical_sha256,
)

_GOAL_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)


@dataclass(frozen=True)
class PlanningTaskConfig:
    """Frozen public construction semantics for one planning task."""

    query_offsets_seconds: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
    action_offsets_seconds: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    delta_velocity_levels_mps: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65)
    k8_delta_velocity_level: int = 2
    minimum_mature_samples: int = 16
    minimum_handle_cosine_margin: float = 0.05
    minimum_single_handle_cosine: float = 0.95
    goal_success_tolerance_m: float = 0.05
    serial_vectorized_cost_tolerance: float = 1.0e-6
    cost_weights: CounterfactualCostWeights = CounterfactualCostWeights()

    def validate(self) -> PlanningTaskConfig:
        if len(self.query_offsets_seconds) == 0:
            raise ValueError("planning query offsets must be nonempty")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.query_offsets_seconds
        ):
            raise ValueError("planning query offsets must be finite and nonnegative")
        if tuple(sorted(self.query_offsets_seconds)) != self.query_offsets_seconds:
            raise ValueError("planning query offsets must be sorted")
        if len(set(self.query_offsets_seconds)) != len(self.query_offsets_seconds):
            raise ValueError("planning query offsets must be unique")
        horizon = self.query_offsets_seconds[-1]
        if len(self.action_offsets_seconds) != 4:
            raise ValueError("planning action offsets must contain four strata")
        if any(
            not math.isfinite(float(value)) or not 0.0 < float(value) < horizon
            for value in self.action_offsets_seconds
        ):
            raise ValueError("every planning action offset must lie inside the horizon")
        if len(set(self.action_offsets_seconds)) != len(self.action_offsets_seconds):
            raise ValueError("planning action offsets must be unique")
        if len(self.delta_velocity_levels_mps) != 4:
            raise ValueError("K=32 requires exactly four delta-velocity levels")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.delta_velocity_levels_mps
        ):
            raise ValueError("delta-velocity levels must be finite and positive")
        if tuple(sorted(self.delta_velocity_levels_mps)) != self.delta_velocity_levels_mps:
            raise ValueError("delta-velocity levels must be strictly increasing")
        if len(set(self.delta_velocity_levels_mps)) != len(self.delta_velocity_levels_mps):
            raise ValueError("delta-velocity levels must be unique")
        if isinstance(self.k8_delta_velocity_level, bool) or not isinstance(
            self.k8_delta_velocity_level, int
        ):
            raise TypeError("k8_delta_velocity_level must be an integer")
        if not 0 <= self.k8_delta_velocity_level < len(self.delta_velocity_levels_mps):
            raise ValueError("k8_delta_velocity_level is out of range")
        if isinstance(self.minimum_mature_samples, bool) or not isinstance(
            self.minimum_mature_samples, int
        ):
            raise TypeError("minimum_mature_samples must be an integer")
        if self.minimum_mature_samples < 3:
            raise ValueError("minimum_mature_samples must be at least three")
        for name, value, strictly_positive in (
            ("minimum_handle_cosine_margin", self.minimum_handle_cosine_margin, False),
            ("minimum_single_handle_cosine", self.minimum_single_handle_cosine, True),
            ("goal_success_tolerance_m", self.goal_success_tolerance_m, True),
            (
                "serial_vectorized_cost_tolerance",
                self.serial_vectorized_cost_tolerance,
                False,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite real number")
            numeric = float(value)
            if (
                not math.isfinite(numeric)
                or numeric < 0.0
                or (strictly_positive and numeric <= 0.0)
            ):
                qualifier = "positive" if strictly_positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if self.minimum_handle_cosine_margin > 2.0:
            raise ValueError("minimum_handle_cosine_margin cannot exceed two")
        if not 0.0 < self.minimum_single_handle_cosine <= 1.0:
            raise ValueError("minimum_single_handle_cosine must lie in (0,1]")
        if not isinstance(self.cost_weights, CounterfactualCostWeights):
            raise TypeError("cost_weights must be CounterfactualCostWeights")
        return self


DEFAULT_PLANNING_TASK_CONFIG = PlanningTaskConfig().validate()
PLANNING_LOG_VARIANCE_BOUNDS = (-32.0, 20.0)


@dataclass(frozen=True)
class PlanningHistoryEvidence:
    """Observable temporal support proving that the supplied belief is mature."""

    valid_sample_count: Tensor
    previously_dynamic: bool

    def validate(
        self,
        row: PlanningManifestRow,
        belief: WorldBelief,
        *,
        minimum_samples: int,
    ) -> PlanningHistoryEvidence:
        if not isinstance(self.valid_sample_count, Tensor):
            raise TypeError("valid_sample_count must be a torch.Tensor")
        if self.valid_sample_count.shape != belief.objects.active.shape:
            raise ValueError("valid_sample_count must have shape [B,N]")
        if self.valid_sample_count.dtype != torch.int64:
            raise TypeError("valid_sample_count must have dtype torch.int64")
        if self.valid_sample_count.device != belief.device:
            raise ValueError("valid_sample_count must be on the belief device")
        if torch.any(self.valid_sample_count < 0):
            raise ValueError("valid_sample_count must be nonnegative")
        if type(self.previously_dynamic) is not bool:
            raise TypeError("previously_dynamic must be boolean")
        if self.previously_dynamic != row.previously_dynamic:
            raise ValueError("history provenance differs from the planning manifest row")
        active = belief.objects.active
        if torch.any(self.valid_sample_count.masked_select(active) < minimum_samples):
            raise ValueError("every active planning target must have a mature history")
        if torch.any(self.valid_sample_count.masked_select(~active) != 0):
            raise ValueError("inactive slots must not retain temporal history")
        if torch.any(belief.objects.age_steps.masked_select(active) < minimum_samples):
            raise ValueError("active planning objects must be mature")
        return self


@dataclass(frozen=True)
class PlanningCandidateDescriptor:
    """Public, model-independent descriptor of one displayed action candidate."""

    display_index: int
    canonical_index: int
    direction_world: tuple[float, float, float]
    delta_velocity_mps: float
    action_offset_seconds: float


@dataclass(frozen=True)
class PublicPlanningTemplate:
    """Checkpoint-independent task data frozen by the public task ledger."""

    row: PlanningManifestRow
    source_timestamp: Tensor
    appearance_handle: Tensor
    query_offsets: Tensor
    goal_position_world: Tensor
    candidate_timestamps: Tensor
    candidate_impulses_world: Tensor
    candidate_descriptors: tuple[PlanningCandidateDescriptor, ...]
    config: PlanningTaskConfig
    template_sha256: str


@dataclass(frozen=True)
class PublicPlanningTask:
    """One fixed public template bound to a checkpoint's current belief IDs."""

    template: PublicPlanningTemplate
    row: PlanningManifestRow
    source_belief: WorldBelief
    history_evidence: PlanningHistoryEvidence
    appearance_handle: Tensor
    target_object_id: Tensor
    query_offsets: Tensor
    goal: TerminalWorldPositionGoal
    candidates: tuple[WorldImpulseAction, ...]
    candidate_descriptors: tuple[PlanningCandidateDescriptor, ...]
    frozen_active_mask: Tensor
    frozen_object_id_by_slot: Tensor
    minimum_mature_samples: int
    goal_success_tolerance_m: float
    serial_vectorized_cost_tolerance: float
    cost_weights: CounterfactualCostWeights
    template_sha256: str
    binding_sha256: str


@dataclass(frozen=True)
class PrivatePlanningOracleEvidence:
    """Evaluator-only oracle labels bound to one reusable public template."""

    template_sha256: str
    candidate_costs: Tensor
    candidate_terminal_goal_distance_m: Tensor
    candidate_goal_success: Tensor
    candidate_induced_contact: Tensor
    certificate: PlanningOracleCertificate
    regret_scale: float
    evidence_sha256: str


@dataclass(frozen=True)
class PlanningTaskEvaluation:
    """One model decision compared with private oracle evidence."""

    row: PlanningManifestRow
    template_sha256: str
    binding_sha256: str
    private_oracle_sha256: str
    model_winner_index: int
    serial_winner_index: int
    oracle_winner_index: int
    oracle_winner_correct: bool
    normalized_regret: float
    oracle_winner_succeeds: bool
    selected_action_goal_success: bool
    serial_vectorized_winner_parity: bool
    maximum_cost_difference: float
    cost_agreement_within_tolerance: bool
    active_set_frozen: bool
    source_belief_unchanged: bool


@dataclass(frozen=True)
class PlanningTaskOutcome:
    """One attempted task, including observable-handle resolution failures."""

    row: PlanningManifestRow
    handle_resolved: bool
    evaluation: PlanningTaskEvaluation | None = None
    failure_reason: str | None = None

    @classmethod
    def evaluated(cls, evaluation: PlanningTaskEvaluation) -> PlanningTaskOutcome:
        if not isinstance(evaluation, PlanningTaskEvaluation):
            raise TypeError("evaluation must be PlanningTaskEvaluation")
        return cls(row=evaluation.row, handle_resolved=True, evaluation=evaluation)

    @classmethod
    def unresolved(
        cls,
        row: PlanningManifestRow,
        *,
        reason: str,
    ) -> PlanningTaskOutcome:
        _validate_manifest_row(row)
        if not isinstance(reason, str) or not reason:
            raise ValueError("an unresolved planning outcome requires a reason")
        return cls(
            row=row,
            handle_resolved=False,
            evaluation=None,
            failure_reason=reason,
        )


@dataclass(frozen=True)
class PlanningEvaluationReduction:
    """Gate-ready planning slices plus serial/vectorised integrity evidence."""

    slices: tuple[PlanningSliceMetrics, ...]
    gate_metrics: PlanningPopulationGateMetrics
    serial_vectorized_winner_parity: bool
    maximum_cost_difference: float
    cost_agreement_within_tolerance: bool
    active_set_frozen: bool
    source_belief_unchanged: bool


@dataclass(frozen=True)
class _PublicPlanningDecision:
    """Oracle-free serial/vectorized decision retained for later scoring."""

    task: PublicPlanningTask
    vectorized: CounterfactualPlanResult
    serial: CounterfactualPlanResult
    cost_tolerance: float
    source_belief_unchanged: bool


def _validate_manifest_row(row: PlanningManifestRow) -> PlanningManifestRow:
    if not isinstance(row, PlanningManifestRow):
        raise TypeError("row must be a PlanningManifestRow")
    if row.split not in {
        "development",
        "selector",
        "confirmation",
        "final_test",
        "compositional_ood",
    }:
        raise ValueError("unknown planning split")
    for name in (
        "ordinal",
        "seed",
        "object_count",
        "target_rank",
        "action_time_stratum",
        "camera_stratum",
        "goal_direction",
        "candidate_count",
    ):
        value = getattr(row, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"planning {name} must be a nonnegative integer")
    for name in ("previously_dynamic", "candidate_induced_contact"):
        if type(getattr(row, name)) is not bool:
            raise TypeError(f"planning {name} must be boolean")
    if row.object_count not in range(1, 7):
        raise ValueError("planning object_count must lie in [1,6]")
    if row.candidate_count not in (8, 32):
        raise ValueError("planning candidate_count must be 8 or 32")
    if not 0 <= row.target_rank < row.object_count:
        raise ValueError("planning target_rank is out of range")
    if not 0 <= row.action_time_stratum < 4:
        raise ValueError("planning action_time_stratum is out of range")
    if not 0 <= row.camera_stratum < 8:
        raise ValueError("planning camera_stratum is out of range")
    if not 0 <= row.goal_direction < len(_GOAL_DIRECTIONS):
        raise ValueError("planning goal_direction is out of range")
    if isinstance(row.minimum_normalized_winner_margin, bool) or not isinstance(
        row.minimum_normalized_winner_margin, Real
    ):
        raise TypeError("planning oracle margin must be a finite real number")
    if (
        not math.isfinite(float(row.minimum_normalized_winner_margin))
        or row.minimum_normalized_winner_margin < 0.05
    ):
        raise ValueError("planning oracle margin must be at least 0.05")
    if row.object_count == 1 and row.candidate_induced_contact:
        raise ValueError("one-object planning tasks cannot induce pair contact")
    if row.distribution not in ("in_distribution", "compositional_ood"):
        raise ValueError("unknown planning distribution")
    if (row.split == "compositional_ood") != (row.distribution == "compositional_ood"):
        raise ValueError("planning split and distribution disagree")
    return row


def validate_planning_manifest_row(row: PlanningManifestRow) -> PlanningManifestRow:
    """Validate one planning row without materializing data or opening a split."""

    return _validate_manifest_row(row)


def _normalised_active_appearances(belief: WorldBelief) -> tuple[Tensor, Tensor]:
    active_slots = torch.nonzero(belief.objects.active[0], as_tuple=False).flatten()
    appearances = belief.objects.appearance[0, active_slots]
    if appearances.shape[-1] == 0:
        raise ValueError("planning handles require nonempty appearance vectors")
    if not torch.isfinite(appearances).all():
        raise ValueError("active planning appearances must be finite")
    norms = torch.linalg.vector_norm(appearances, dim=-1)
    if torch.any(norms <= 0.0):
        raise ValueError("active planning appearances must have nonzero norm")
    return active_slots, appearances / norms.unsqueeze(-1)


def ranked_observable_appearance_handle(
    belief: WorldBelief,
    target_rank: int,
) -> Tensor:
    """Return a permutation-invariant handle for task-data construction.

    Evaluation must call this on an independently frozen public reference, not
    on the checkpoint belief that will subsequently resolve the handle.
    """

    if belief.batch_size != 1:
        raise ValueError("one PlanningManifestRow materialises exactly one B1 task")
    if isinstance(target_rank, bool) or not isinstance(target_rank, int):
        raise TypeError("target_rank must be an integer")
    active_slots, appearances = _normalised_active_appearances(belief)
    if not 0 <= target_rank < active_slots.numel():
        raise ValueError("target_rank is out of range for the active appearance set")

    # Lexicographic ordering is a discrete public handle convention.  It uses
    # neither padded-slot order nor persistent/simulator identity.
    keys = [tuple(float(value) for value in row.detach().cpu()) for row in appearances]
    if len(set(keys)) != len(keys):
        raise ValueError("active appearance handles are not observably distinct")
    ordered_local = sorted(range(len(keys)), key=keys.__getitem__)
    local_index = ordered_local[target_rank]
    slot = int(active_slots[local_index])
    return belief.objects.appearance[:, slot].detach().clone()


def resolve_planning_target_handle(
    belief: WorldBelief,
    prototype: Tensor,
    *,
    minimum_cosine_margin: float,
    minimum_single_cosine: float,
) -> Tensor:
    """Resolve an observable handle for all 1--6 supported cardinalities."""

    if belief.batch_size != 1:
        raise ValueError("one planning target handle requires a B1 belief")
    if (
        isinstance(minimum_single_cosine, bool)
        or not isinstance(minimum_single_cosine, Real)
        or not math.isfinite(float(minimum_single_cosine))
        or not 0.0 < float(minimum_single_cosine) <= 1.0
    ):
        raise ValueError("minimum_single_cosine must lie in (0,1]")
    active_count = int(belief.objects.active.sum())
    if active_count != 1:
        resolved = resolve_appearance_handle(
            belief,
            prototype,
            minimum_cosine_margin=minimum_cosine_margin,
        )
        _, appearances = _normalised_active_appearances(belief)
        prototype_norm = torch.linalg.vector_norm(prototype, dim=-1)
        normalised_prototype = prototype / prototype_norm.unsqueeze(-1)
        top_cosine = torch.einsum("bd,nd->bn", normalised_prototype, appearances).amax(dim=-1)
        if torch.any(top_cosine < float(minimum_single_cosine)):
            raise ValueError("multi-object appearance handle is below the cosine gate")
        return resolved

    if not isinstance(prototype, Tensor):
        raise TypeError("prototype must be a torch.Tensor")
    expected = (1, belief.objects.appearance_dim)
    if prototype.shape != expected:
        raise ValueError(f"prototype must have shape {expected}")
    if prototype.dtype != belief.dtype:
        raise TypeError("prototype must have the belief dtype")
    if prototype.device != belief.device:
        raise ValueError("prototype must be on the belief device")
    if not torch.isfinite(prototype).all():
        raise ValueError("prototype must be finite")
    slot = int(torch.nonzero(belief.objects.active[0], as_tuple=False).item())
    appearance = belief.objects.appearance[:, slot]
    prototype_norm = torch.linalg.vector_norm(prototype, dim=-1)
    appearance_norm = torch.linalg.vector_norm(appearance, dim=-1)
    if torch.any(prototype_norm <= 0.0) or torch.any(appearance_norm <= 0.0):
        raise ValueError("single-object appearance handle must have nonzero norm")
    cosine = (prototype * appearance).sum(dim=-1) / (prototype_norm * appearance_norm)
    if torch.any(cosine < minimum_single_cosine):
        raise ValueError("single-object appearance handle is below the cosine gate")
    object_id = belief.objects.object_id[:, slot]
    if object_id.dtype != torch.int64 or torch.any(object_id < 0):
        raise ValueError("single active target must have a persistent object ID")
    return object_id.detach().clone()


def _target_slot(belief: WorldBelief, target_object_id: Tensor) -> int:
    matches = belief.objects.active & (belief.objects.object_id == target_object_id.unsqueeze(-1))
    if int(matches.sum()) != 1:
        raise ValueError("planning target must resolve to one active slot")
    return int(torch.nonzero(matches[0], as_tuple=False).item())


def _nearest_object_direction(belief: WorldBelief, target_slot: int) -> Tensor:
    relative = belief.objects.position[0] - belief.objects.position[0, target_slot]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    candidate = belief.objects.active[0].clone()
    candidate[target_slot] = False
    if not torch.any(candidate):
        raise ValueError("contact-directed candidates require at least two objects")
    surface_gap = distance - (
        belief.objects.radius[0, :, 0] + belief.objects.radius[0, target_slot, 0]
    )
    surface_gap = surface_gap.masked_fill(~candidate, torch.inf)
    nearest = int(surface_gap.argmin())
    if not math.isfinite(float(distance[nearest])) or float(distance[nearest]) <= 1.0e-8:
        raise ValueError("contact-directed objects must have distinct finite centres")
    return relative[nearest] / distance[nearest]


def _candidate_directions(
    belief: WorldBelief,
    row: PlanningManifestRow,
    target_slot: int,
) -> tuple[Tensor, ...]:
    goal = belief.objects.position.new_tensor(_GOAL_DIRECTIONS[row.goal_direction])
    goal_axis = row.goal_direction // 2
    orthogonal_axes = [axis for axis in range(3) if axis != goal_axis]
    first = torch.zeros_like(goal)
    second = torch.zeros_like(goal)
    first[orthogonal_axes[0]] = 1.0
    second[orthogonal_axes[1]] = 1.0
    diagonal = goal + first + second
    diagonal = diagonal / torch.linalg.vector_norm(diagonal)
    directions: list[Tensor] = [
        goal,
        -goal,
        first,
        -first,
        second,
        -second,
        diagonal,
        -diagonal,
    ]
    if row.candidate_induced_contact:
        contact = _nearest_object_direction(belief, target_slot)
        duplicate = any(
            abs(float(torch.dot(contact, direction))) > 1.0 - 1.0e-6 for direction in directions
        )
        if not duplicate:
            directions[-1] = contact
    if len(directions) != 8:
        raise RuntimeError("planning candidate direction bank must have eight entries")
    for index, left in enumerate(directions):
        if not torch.isfinite(left).all():
            raise ValueError("planning candidate directions must be finite")
        norm = torch.linalg.vector_norm(left)
        if not torch.isclose(norm, norm.new_ones(()), atol=1.0e-6, rtol=1.0e-6):
            raise RuntimeError("planning candidate directions must be unit vectors")
        for right in directions[:index]:
            if torch.allclose(left, right, atol=1.0e-6, rtol=0.0):
                raise RuntimeError("planning candidate directions must be unique")
    return tuple(direction.detach().clone() for direction in directions)


def _display_order(row: PlanningManifestRow) -> tuple[int, ...]:
    count = row.candidate_count
    multipliers = (1, 3, 5, 7) if count == 8 else tuple(range(1, 32, 2))
    multiplier = multipliers[(row.seed + row.camera_stratum) % len(multipliers)]
    offset = (
        row.seed
        + 3 * row.camera_stratum
        + 5 * row.goal_direction
        + 7 * row.target_rank
        + 11 * row.action_time_stratum
    ) % count
    order = tuple((multiplier * index + offset) % count for index in range(count))
    if len(set(order)) != count:
        raise RuntimeError("planning display order must be a permutation")
    return order


def _selected_goal_level(row: PlanningManifestRow, config: PlanningTaskConfig) -> int:
    if row.candidate_count == 8:
        return config.k8_delta_velocity_level
    return (row.ordinal + row.camera_stratum + row.target_rank + row.action_time_stratum) % len(
        config.delta_velocity_levels_mps
    )


def _free_motion_goal(
    belief: WorldBelief,
    *,
    target_slot: int,
    goal_direction: Tensor,
    action_offset: float,
    delta_velocity: float,
    horizon: float,
) -> Tensor:
    with torch.no_grad():
        baseline_objects = AnalyticKinematics()(
            belief.objects,
            belief.gravity,
            belief.timestamp.new_full((1,), horizon),
        )
        remaining = belief.timestamp.new_tensor(horizon - action_offset)
        drag = belief.objects.drag[0, target_slot, 0].clamp(1.0e-8, 100.0)
        one_minus_decay = -torch.expm1(-drag * remaining)
        response = torch.where(
            drag >= 1.0e-5,
            one_minus_decay / drag.clamp_min(1.0e-5),
            remaining,
        )
        goal = baseline_objects.position[0, target_slot] + (
            goal_direction * float(delta_velocity) * response
        )
        return goal.unsqueeze(0).detach().clone()


def _template_digest_payload(
    row: PlanningManifestRow,
    source_timestamp: Tensor,
    appearance_handle: Tensor,
    query_offsets: Tensor,
    goal_position_world: Tensor,
    candidate_timestamps: Tensor,
    candidate_impulses_world: Tensor,
    descriptors: Sequence[PlanningCandidateDescriptor],
    config: PlanningTaskConfig,
) -> Mapping[str, Any]:
    """Return the ID-free, checkpoint-independent public ledger payload."""

    return {
        "row": asdict(row),
        "source_timestamp": _canonical_public_value(source_timestamp),
        "appearance_handle": _canonical_public_value(appearance_handle),
        "query_offsets": _canonical_public_value(query_offsets),
        "goal_position_world": _canonical_public_value(goal_position_world),
        "candidate_timestamps": _canonical_public_value(candidate_timestamps),
        "candidate_impulses_world": _canonical_public_value(candidate_impulses_world),
        "candidate_descriptors": [asdict(descriptor) for descriptor in descriptors],
        "config": _canonical_public_value(config),
    }


def _binding_digest_payload(
    template_sha256: str,
    source_belief: WorldBelief,
    history_evidence: PlanningHistoryEvidence,
    target_object_id: Tensor,
) -> Mapping[str, Any]:
    return {
        "template_sha256": template_sha256,
        "source_belief": _canonical_public_value(source_belief),
        "history_evidence": _canonical_public_value(history_evidence),
        "target_object_id": _canonical_public_value(target_object_id),
    }


def _canonical_public_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_public_value(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_public_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_public_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported public task digest value {type(value).__name__}")


def materialize_public_planning_template(
    row: PlanningManifestRow,
    reference_belief: WorldBelief,
    appearance_handle: Tensor,
    *,
    config: PlanningTaskConfig = DEFAULT_PLANNING_TASK_CONFIG,
) -> PublicPlanningTemplate:
    """Freeze task data once from a ledger-owned public reference belief.

    ``reference_belief`` is public task-generation data.  It must never be the
    baseline or candidate checkpoint belief under evaluation.  Its metadata,
    gradients, and persistent IDs are deliberately absent from the template
    digest.  Only the resulting observable handle, absolute times, world-frame
    goal, action vectors/order, row, and public configuration are frozen.
    """

    _validate_manifest_row(row)
    if not isinstance(reference_belief, WorldBelief):
        raise TypeError("reference_belief must be a WorldBelief")
    reference_belief.validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)
    if reference_belief.batch_size != 1:
        raise ValueError("one PlanningManifestRow materialises exactly one B1 template")
    if int(reference_belief.objects.active.sum()) != row.object_count:
        raise ValueError("active reference cardinality differs from the planning row")
    if not isinstance(config, PlanningTaskConfig):
        raise TypeError("config must be PlanningTaskConfig")
    config.validate()

    # Stage one is ledger-only.  Strip arbitrary metadata and all gradients
    # before deriving any fixed task value.
    reference = (
        reference_belief.detach()
        .clone()
        .replace(metadata={})
        .validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)
    )
    if not isinstance(appearance_handle, Tensor):
        raise TypeError("appearance_handle must be an independently supplied tensor")
    appearance_handle = appearance_handle.detach().clone()
    reference_target_id = resolve_planning_target_handle(
        reference,
        appearance_handle,
        minimum_cosine_margin=config.minimum_handle_cosine_margin,
        minimum_single_cosine=config.minimum_single_handle_cosine,
    )
    ranked_reference_handle = ranked_observable_appearance_handle(reference, row.target_rank)
    ranked_reference_target_id = resolve_planning_target_handle(
        reference,
        ranked_reference_handle,
        minimum_cosine_margin=config.minimum_handle_cosine_margin,
        minimum_single_cosine=config.minimum_single_handle_cosine,
    )
    if not torch.equal(reference_target_id, ranked_reference_target_id):
        raise ValueError("observable planning handle differs from the manifest target rank")
    target_slot = _target_slot(reference, reference_target_id)
    if int(reference.objects.mode[0, target_slot]) != int(MotionMode.FREE):
        raise ValueError("planning reference target must currently be in FREE mode")

    action_offset = config.action_offsets_seconds[row.action_time_stratum]
    directions = _candidate_directions(reference, row, target_slot)
    levels = (
        (config.delta_velocity_levels_mps[config.k8_delta_velocity_level],)
        if row.candidate_count == 8
        else config.delta_velocity_levels_mps
    )
    canonical: list[tuple[Tensor, float]] = [
        (direction, level) for level in levels for direction in directions
    ]
    if len(canonical) != row.candidate_count:
        raise RuntimeError("candidate construction differs from the manifest count")
    display_order = _display_order(row)
    mass = reference.objects.mass[:, target_slot]
    action_timestamp = reference.timestamp + action_offset
    candidate_timestamps: list[Tensor] = []
    candidate_impulses: list[Tensor] = []
    descriptors: list[PlanningCandidateDescriptor] = []
    for display_index, canonical_index in enumerate(display_order):
        direction, delta_velocity = canonical[canonical_index]
        impulse = mass * direction.unsqueeze(0) * float(delta_velocity)
        candidate_timestamps.append(action_timestamp.detach().clone())
        candidate_impulses.append(impulse.detach().clone())
        descriptors.append(
            PlanningCandidateDescriptor(
                display_index=display_index,
                canonical_index=canonical_index,
                direction_world=tuple(float(value) for value in direction.cpu()),
                delta_velocity_mps=float(delta_velocity),
                action_offset_seconds=float(action_offset),
            )
        )

    selected_level = _selected_goal_level(row, config)
    goal_direction = reference.objects.position.new_tensor(_GOAL_DIRECTIONS[row.goal_direction])
    goal_position = _free_motion_goal(
        reference,
        target_slot=target_slot,
        goal_direction=goal_direction,
        action_offset=action_offset,
        delta_velocity=config.delta_velocity_levels_mps[selected_level],
        horizon=config.query_offsets_seconds[-1],
    )
    query_offsets = reference.timestamp.new_tensor(config.query_offsets_seconds).unsqueeze(0)
    stacked_timestamps = torch.stack(candidate_timestamps)
    stacked_impulses = torch.stack(candidate_impulses)
    payload = _template_digest_payload(
        row,
        reference.timestamp,
        appearance_handle,
        query_offsets,
        goal_position,
        stacked_timestamps,
        stacked_impulses,
        descriptors,
        config,
    )
    template = PublicPlanningTemplate(
        row=row,
        source_timestamp=reference.timestamp.detach().clone(),
        appearance_handle=appearance_handle,
        query_offsets=query_offsets.detach().clone(),
        goal_position_world=goal_position,
        candidate_timestamps=stacked_timestamps,
        candidate_impulses_world=stacked_impulses,
        candidate_descriptors=tuple(descriptors),
        config=config,
        template_sha256=canonical_sha256(payload),
    )
    return _validate_public_template(template)


def _clone_public_template(template: PublicPlanningTemplate) -> PublicPlanningTemplate:
    return PublicPlanningTemplate(
        row=template.row,
        source_timestamp=template.source_timestamp.detach().clone(),
        appearance_handle=template.appearance_handle.detach().clone(),
        query_offsets=template.query_offsets.detach().clone(),
        goal_position_world=template.goal_position_world.detach().clone(),
        candidate_timestamps=template.candidate_timestamps.detach().clone(),
        candidate_impulses_world=template.candidate_impulses_world.detach().clone(),
        candidate_descriptors=template.candidate_descriptors,
        config=template.config,
        template_sha256=template.template_sha256,
    )


def bind_public_planning_task(
    template: PublicPlanningTemplate,
    belief: WorldBelief,
    history_evidence: PlanningHistoryEvidence,
) -> PublicPlanningTask:
    """Bind only a resolved persistent ID and checkpoint state to fixed task data."""

    _validate_public_template(template)
    if not isinstance(belief, WorldBelief):
        raise TypeError("belief must be a WorldBelief")
    belief.validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)
    if belief.batch_size != 1:
        raise ValueError("one public planning template binds exactly one B1 belief")
    if int(belief.objects.active.sum()) != template.row.object_count:
        raise ValueError("active belief cardinality differs from the planning template")
    if not torch.equal(belief.timestamp, template.source_timestamp):
        raise ValueError("checkpoint belief timestamp differs from the fixed template")
    if not isinstance(history_evidence, PlanningHistoryEvidence):
        raise TypeError("history_evidence must be PlanningHistoryEvidence")
    history_evidence.validate(
        template.row,
        belief,
        minimum_samples=template.config.minimum_mature_samples,
    )

    source_belief = (
        belief.detach()
        .clone()
        .replace(metadata={})
        .validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)
    )
    frozen_template = _clone_public_template(template)
    frozen_history = PlanningHistoryEvidence(
        valid_sample_count=history_evidence.valid_sample_count.detach().clone(),
        previously_dynamic=history_evidence.previously_dynamic,
    )
    appearance_handle = frozen_template.appearance_handle.detach().clone()
    target_object_id = resolve_planning_target_handle(
        source_belief,
        appearance_handle,
        minimum_cosine_margin=template.config.minimum_handle_cosine_margin,
        minimum_single_cosine=template.config.minimum_single_handle_cosine,
    )
    target_slot = _target_slot(source_belief, target_object_id)
    if int(source_belief.objects.mode[0, target_slot]) != int(MotionMode.FREE):
        raise ValueError("planning action target must currently be in FREE mode")

    candidates = tuple(
        WorldImpulseAction(
            timestamp=frozen_template.candidate_timestamps[index].detach().clone(),
            object_id=target_object_id.detach().clone(),
            impulse_world=frozen_template.candidate_impulses_world[index].detach().clone(),
        )
        for index in range(template.row.candidate_count)
    )
    latest_timestamp = source_belief.timestamp + frozen_template.query_offsets[:, -1]
    for action in candidates:
        action.validate_for(source_belief, latest_timestamp=latest_timestamp)
    goal = TerminalWorldPositionGoal(
        object_id=target_object_id.detach().clone(),
        position_world=frozen_template.goal_position_world.detach().clone(),
    )
    frozen_active = source_belief.objects.active.detach().clone()
    frozen_ids = source_belief.objects.object_id.detach().clone()
    payload = _binding_digest_payload(
        frozen_template.template_sha256,
        source_belief,
        frozen_history,
        target_object_id,
    )
    return PublicPlanningTask(
        template=frozen_template,
        row=frozen_template.row,
        source_belief=source_belief,
        history_evidence=frozen_history,
        appearance_handle=appearance_handle,
        target_object_id=target_object_id,
        query_offsets=frozen_template.query_offsets.detach().clone(),
        goal=goal,
        candidates=candidates,
        candidate_descriptors=frozen_template.candidate_descriptors,
        frozen_active_mask=frozen_active,
        frozen_object_id_by_slot=frozen_ids,
        minimum_mature_samples=frozen_template.config.minimum_mature_samples,
        goal_success_tolerance_m=frozen_template.config.goal_success_tolerance_m,
        serial_vectorized_cost_tolerance=(frozen_template.config.serial_vectorized_cost_tolerance),
        cost_weights=frozen_template.config.cost_weights,
        template_sha256=frozen_template.template_sha256,
        binding_sha256=canonical_sha256(payload),
    )


def _one_dimensional_tensor(
    name: str,
    value: Tensor | Sequence[float] | Sequence[bool],
    *,
    count: int,
    dtype: torch.dtype,
) -> Tensor:
    if dtype is torch.bool:
        uncast = torch.as_tensor(value, device="cpu")
        if uncast.dtype is not torch.bool:
            raise TypeError(f"{name} must contain boolean evaluator labels")
    tensor = torch.as_tensor(value, device="cpu", dtype=dtype)
    if tensor.shape != (count,):
        raise ValueError(f"{name} must have shape [{count}]")
    if tensor.requires_grad:
        raise ValueError(f"{name} must be detached evaluator evidence")
    return tensor.detach().clone()


def certify_private_planning_oracle(
    template: PublicPlanningTemplate,
    *,
    candidate_costs: Tensor | Sequence[float],
    candidate_terminal_goal_distance_m: Tensor | Sequence[float],
    candidate_induced_contact: Tensor | Sequence[bool],
) -> PrivatePlanningOracleEvidence:
    """Certify one oracle ledger reusable across every checkpoint binding."""

    _validate_public_template(template)
    count = template.row.candidate_count
    costs = _one_dimensional_tensor(
        "candidate_costs",
        candidate_costs,
        count=count,
        dtype=torch.float64,
    )
    goal_distance = _one_dimensional_tensor(
        "candidate_terminal_goal_distance_m",
        candidate_terminal_goal_distance_m,
        count=count,
        dtype=torch.float64,
    )
    contact = _one_dimensional_tensor(
        "candidate_induced_contact",
        candidate_induced_contact,
        count=count,
        dtype=torch.bool,
    )
    if not torch.isfinite(costs).all() or torch.any(costs < 0.0):
        raise ValueError("oracle candidate costs must be finite and nonnegative")
    if not torch.isfinite(goal_distance).all() or torch.any(goal_distance < 0.0):
        raise ValueError("oracle terminal goal distances must be finite and nonnegative")
    goal_success = goal_distance <= template.config.goal_success_tolerance_m
    if template.row.object_count == 1 and torch.any(contact):
        raise ValueError("one-object oracle evidence cannot contain pair contact")
    if template.row.candidate_induced_contact:
        if not torch.any(contact):
            raise ValueError("contact-stratum task has no contact-inducing candidate")
    elif torch.any(contact):
        raise ValueError("no-contact-stratum task contains a contact-inducing candidate")
    ordered = torch.argsort(costs, stable=True)
    winner = int(ordered[0])
    certificate = PlanningOracleCertificate.from_costs(
        costs.tolist(),
        winner_succeeds=bool(goal_success[winner]),
        minimum_margin=template.row.minimum_normalized_winner_margin,
    )
    if certificate.winner_index != winner:
        raise RuntimeError("oracle certificate winner differs from stable cost order")
    second = int(ordered[1])
    scale = max(abs(float(costs[winner])), abs(float(costs[second])), 1.0e-12)
    payload = {
        "template_sha256": template.template_sha256,
        "candidate_costs": costs.tolist(),
        "candidate_terminal_goal_distance_m": goal_distance.tolist(),
        "candidate_goal_success": goal_success.tolist(),
        "candidate_induced_contact": contact.tolist(),
        "certificate": asdict(certificate),
        "regret_scale": scale,
    }
    return PrivatePlanningOracleEvidence(
        template_sha256=template.template_sha256,
        candidate_costs=costs,
        candidate_terminal_goal_distance_m=goal_distance,
        candidate_goal_success=goal_success,
        candidate_induced_contact=contact,
        certificate=certificate,
        regret_scale=scale,
        evidence_sha256=canonical_sha256(payload),
    )


def _tensor_tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left, right)
    if is_dataclass(left) or is_dataclass(right):
        if type(left) is not type(right) or not is_dataclass(left) or not is_dataclass(right):
            return False
        return all(
            _tensor_tree_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _tensor_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_tensor_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _validate_public_template(template: PublicPlanningTemplate) -> PublicPlanningTemplate:
    if not isinstance(template, PublicPlanningTemplate):
        raise TypeError("template must be a PublicPlanningTemplate")
    row = _validate_manifest_row(template.row)
    if not isinstance(template.config, PlanningTaskConfig):
        raise TypeError("planning template config must be PlanningTaskConfig")
    config = template.config.validate()
    tensor_schema = (
        ("source_timestamp", template.source_timestamp, (1,)),
        ("appearance_handle", template.appearance_handle, (1, 8)),
        (
            "query_offsets",
            template.query_offsets,
            (1, len(config.query_offsets_seconds)),
        ),
        ("goal_position_world", template.goal_position_world, (1, 3)),
        (
            "candidate_timestamps",
            template.candidate_timestamps,
            (row.candidate_count, 1),
        ),
        (
            "candidate_impulses_world",
            template.candidate_impulses_world,
            (row.candidate_count, 1, 3),
        ),
    )
    for name, value, shape in tensor_schema:
        if not isinstance(value, Tensor) or value.shape != shape:
            raise ValueError(f"planning template {name} must have shape {shape}")
        if value.device.type != "cpu" or value.dtype is not torch.float32:
            raise ValueError(f"planning template {name} must be CPU float32")
        if value.requires_grad:
            raise ValueError(f"planning template {name} must be detached")
        if not torch.isfinite(value).all():
            raise ValueError(f"planning template {name} must be finite")
    expected_offsets = template.source_timestamp.new_tensor(config.query_offsets_seconds).unsqueeze(
        0
    )
    if not torch.equal(template.query_offsets, expected_offsets):
        raise ValueError("planning template query offsets differ from its public config")
    expected_action_time = (
        template.source_timestamp + config.action_offsets_seconds[row.action_time_stratum]
    )
    if not torch.equal(
        template.candidate_timestamps,
        expected_action_time.expand(row.candidate_count, -1),
    ):
        raise ValueError("planning template candidate timestamps differ from its action stratum")
    if bool(template.candidate_impulses_world.eq(0).all(dim=-1).any()):
        raise ValueError("planning template candidates require nonzero impulses")
    flattened_impulses = template.candidate_impulses_world[:, 0]
    if torch.unique(flattened_impulses, dim=0).shape[0] != row.candidate_count:
        raise ValueError("planning template candidate impulses must be unique")
    if (
        not isinstance(template.candidate_descriptors, tuple)
        or len(template.candidate_descriptors) != row.candidate_count
    ):
        raise ValueError("planning template descriptors differ from its candidate count")
    canonical_indices: set[int] = set()
    expected_action_offset = config.action_offsets_seconds[row.action_time_stratum]
    for display_index, descriptor in enumerate(template.candidate_descriptors):
        if not isinstance(descriptor, PlanningCandidateDescriptor):
            raise TypeError("planning template descriptors have an invalid type")
        if descriptor.display_index != display_index:
            raise ValueError("planning template descriptor display order is inconsistent")
        if not 0 <= descriptor.canonical_index < row.candidate_count:
            raise ValueError("planning template descriptor canonical index is out of range")
        canonical_indices.add(descriptor.canonical_index)
        if descriptor.action_offset_seconds != expected_action_offset:
            raise ValueError("planning template descriptor action time is inconsistent")
        direction = flattened_impulses.new_tensor(descriptor.direction_world)
        if direction.shape != (3,) or not torch.isfinite(direction).all():
            raise ValueError("planning template descriptor direction must be finite world xyz")
        if not torch.isclose(
            torch.linalg.vector_norm(direction),
            direction.new_ones(()),
            rtol=1.0e-6,
            atol=1.0e-6,
        ):
            raise ValueError("planning template descriptor direction must be unit length")
        impulse_direction = flattened_impulses[display_index] / torch.linalg.vector_norm(
            flattened_impulses[display_index]
        )
        if not torch.allclose(impulse_direction, direction, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError("planning template descriptor and impulse directions disagree")
        if (
            isinstance(descriptor.delta_velocity_mps, bool)
            or not isinstance(descriptor.delta_velocity_mps, Real)
            or not math.isfinite(float(descriptor.delta_velocity_mps))
            or descriptor.delta_velocity_mps <= 0.0
        ):
            raise ValueError("planning template descriptor delta velocity must be positive")
    if canonical_indices != set(range(row.candidate_count)):
        raise ValueError("planning template canonical candidate order is not a permutation")
    payload = _template_digest_payload(
        row,
        template.source_timestamp,
        template.appearance_handle,
        template.query_offsets,
        template.goal_position_world,
        template.candidate_timestamps,
        template.candidate_impulses_world,
        template.candidate_descriptors,
        config,
    )
    if canonical_sha256(payload) != template.template_sha256:
        raise ValueError("public planning template differs from its template digest")
    return template


def validate_public_planning_template(
    template: PublicPlanningTemplate,
) -> PublicPlanningTemplate:
    """Revalidate a serialized public template and its checkpoint-free digest."""

    return _validate_public_template(template)


def _validate_public_task(task: PublicPlanningTask) -> PublicPlanningTask:
    if not isinstance(task, PublicPlanningTask):
        raise TypeError("task must be a PublicPlanningTask")
    template = _validate_public_template(task.template)
    if task.row != template.row or task.template_sha256 != template.template_sha256:
        raise ValueError("public planning binding refers to a different fixed template")
    task.source_belief.validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)
    if task.source_belief.batch_size != 1:
        raise ValueError("public planning task must remain B1")
    if not torch.equal(task.source_belief.timestamp, template.source_timestamp):
        raise ValueError("public planning source timestamp differs from its fixed template")
    if len(task.candidates) != task.row.candidate_count:
        raise ValueError("public planning task candidate count changed")
    if task.candidate_descriptors != template.candidate_descriptors:
        raise ValueError("public planning descriptors differ from their fixed template")
    if not torch.equal(task.appearance_handle, template.appearance_handle):
        raise ValueError("public planning handle differs from its fixed template")
    if not torch.equal(task.query_offsets, template.query_offsets):
        raise ValueError("public planning queries differ from their fixed template")
    if not isinstance(task.goal, TerminalWorldPositionGoal) or task.goal.frame != "world":
        raise TypeError("public planning goal must be a world-position goal")
    if not torch.equal(task.goal.position_world, template.goal_position_world):
        raise ValueError("public planning goal position differs from its fixed template")
    if (
        task.minimum_mature_samples != template.config.minimum_mature_samples
        or task.goal_success_tolerance_m != template.config.goal_success_tolerance_m
        or task.serial_vectorized_cost_tolerance != template.config.serial_vectorized_cost_tolerance
        or task.cost_weights != template.config.cost_weights
    ):
        raise ValueError("public planning binding configuration differs from its fixed template")
    if not torch.equal(task.source_belief.objects.active, task.frozen_active_mask):
        raise ValueError("public planning task active set changed")
    if not torch.equal(task.source_belief.objects.object_id, task.frozen_object_id_by_slot):
        raise ValueError("public planning task object IDs changed")
    if task.source_belief.metadata:
        raise ValueError("public planning source metadata must remain empty")
    task.history_evidence.validate(
        task.row,
        task.source_belief,
        minimum_samples=task.minimum_mature_samples,
    )
    expected_target_id = resolve_planning_target_handle(
        task.source_belief,
        task.appearance_handle,
        minimum_cosine_margin=template.config.minimum_handle_cosine_margin,
        minimum_single_cosine=template.config.minimum_single_handle_cosine,
    )
    if not torch.equal(task.target_object_id, expected_target_id):
        raise ValueError("public planning target ID differs from its fixed observable handle")
    target_slot = _target_slot(task.source_belief, task.target_object_id)
    if int(task.source_belief.objects.mode[0, target_slot]) != int(MotionMode.FREE):
        raise ValueError("public planning target must remain in FREE mode")
    for index, action in enumerate(task.candidates):
        if not isinstance(action, WorldImpulseAction):
            raise TypeError("public planning candidates must be WorldImpulseAction values")
        if not torch.equal(action.object_id, task.target_object_id):
            raise ValueError("every candidate action must address the resolved planning target")
        if not torch.equal(action.timestamp, template.candidate_timestamps[index]):
            raise ValueError("candidate action timestamp differs from the fixed template")
        if not torch.equal(action.impulse_world, template.candidate_impulses_world[index]):
            raise ValueError("candidate impulse differs from the fixed template")
        action.validate_for(
            task.source_belief,
            latest_timestamp=task.source_belief.timestamp + task.query_offsets[:, -1],
        )
    if not torch.equal(task.goal.object_id, task.target_object_id):
        raise ValueError("planning goal and candidate actions must address the same target")
    payload = _binding_digest_payload(
        task.template_sha256,
        task.source_belief,
        task.history_evidence,
        task.target_object_id,
    )
    if canonical_sha256(payload) != task.binding_sha256:
        raise ValueError("public planning task differs from its checkpoint binding digest")
    return task


def _validate_private_oracle(
    task: PublicPlanningTask,
    oracle: PrivatePlanningOracleEvidence,
) -> PrivatePlanningOracleEvidence:
    if not isinstance(oracle, PrivatePlanningOracleEvidence):
        raise TypeError("oracle must be PrivatePlanningOracleEvidence")
    if oracle.template_sha256 != task.template_sha256:
        raise ValueError("private oracle evidence belongs to a different public template")
    count = task.row.candidate_count
    for name, value, dtype in (
        ("candidate_costs", oracle.candidate_costs, torch.float64),
        (
            "candidate_terminal_goal_distance_m",
            oracle.candidate_terminal_goal_distance_m,
            torch.float64,
        ),
        ("candidate_goal_success", oracle.candidate_goal_success, torch.bool),
        ("candidate_induced_contact", oracle.candidate_induced_contact, torch.bool),
    ):
        if not isinstance(value, Tensor) or value.shape != (count,) or value.dtype != dtype:
            raise ValueError(f"private oracle {name} differs from its certified schema")
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError(f"private oracle {name} must be detached CPU evidence")
    if not torch.isfinite(oracle.candidate_costs).all() or torch.any(oracle.candidate_costs < 0.0):
        raise ValueError("private oracle costs must be finite and nonnegative")
    if not torch.isfinite(oracle.candidate_terminal_goal_distance_m).all() or torch.any(
        oracle.candidate_terminal_goal_distance_m < 0.0
    ):
        raise ValueError("private oracle goal distances must be finite and nonnegative")
    expected_goal_success = (
        oracle.candidate_terminal_goal_distance_m <= task.template.config.goal_success_tolerance_m
    )
    if not torch.equal(expected_goal_success, oracle.candidate_goal_success):
        raise ValueError("private oracle goal-success labels differ from terminal distances")
    ordered = torch.argsort(oracle.candidate_costs, stable=True)
    winner = int(ordered[0])
    expected_certificate = PlanningOracleCertificate.from_costs(
        oracle.candidate_costs.tolist(),
        winner_succeeds=bool(oracle.candidate_goal_success[winner]),
        minimum_margin=task.row.minimum_normalized_winner_margin,
    )
    if expected_certificate != oracle.certificate:
        raise ValueError("private oracle certificate differs from its candidate evidence")
    if task.row.candidate_induced_contact:
        if not torch.any(oracle.candidate_induced_contact):
            raise ValueError("private contact stratum contains no contact evidence")
    elif torch.any(oracle.candidate_induced_contact):
        raise ValueError("private no-contact stratum contains contact evidence")
    second = int(ordered[1])
    expected_scale = max(
        abs(float(oracle.candidate_costs[winner])),
        abs(float(oracle.candidate_costs[second])),
        1.0e-12,
    )
    if not math.isfinite(oracle.regret_scale) or oracle.regret_scale != expected_scale:
        raise ValueError("private oracle regret scale differs from its candidate evidence")
    payload = {
        "template_sha256": oracle.template_sha256,
        "candidate_costs": oracle.candidate_costs.tolist(),
        "candidate_terminal_goal_distance_m": (oracle.candidate_terminal_goal_distance_m.tolist()),
        "candidate_goal_success": oracle.candidate_goal_success.tolist(),
        "candidate_induced_contact": oracle.candidate_induced_contact.tolist(),
        "certificate": asdict(oracle.certificate),
        "regret_scale": oracle.regret_scale,
    }
    if canonical_sha256(payload) != oracle.evidence_sha256:
        raise ValueError("private oracle evidence differs from its certificate digest")
    return oracle


def _plan_active_set_is_frozen(
    task: PublicPlanningTask,
    result: CounterfactualPlanResult,
) -> bool:
    if not torch.equal(result.object_id_by_slot, task.frozen_object_id_by_slot):
        return False
    for trajectory in result.trajectories:
        expected = task.frozen_active_mask.unsqueeze(1).expand_as(trajectory.active_mask)
        if not torch.equal(trajectory.active_mask, expected):
            return False
    return True


def _diagnostic_query_offsets(task: PublicPlanningTask) -> Tensor:
    action_offset = task.candidates[0].timestamp[0] - task.source_belief.timestamp[0]
    horizon = task.query_offsets[0, -1]
    before = 0.5 * action_offset
    after = action_offset + 0.5 * (horizon - action_offset)
    return torch.stack((before, action_offset, action_offset, after)).unsqueeze(0)


def _allclose(left: Tensor, right: Tensor, tolerance: float) -> bool:
    if left.dtype is torch.bool or not left.is_floating_point():
        return torch.equal(left, right)
    return bool(torch.allclose(left, right, rtol=tolerance, atol=tolerance))


def _pre_action_states_equal(
    acted: BeliefTrajectory,
    baseline: BeliefTrajectory,
) -> bool:
    for name in (
        "timestamps",
        "positions",
        "velocities",
        "orientations",
        "motion_mode_logits",
        "fast_log_variance",
        "active_mask",
    ):
        if not torch.equal(getattr(acted, name)[:, 0], getattr(baseline, name)[:, 0]):
            return False
    if (acted.event_logits is None) != (baseline.event_logits is None):
        return False
    return acted.event_logits is None or torch.equal(
        acted.event_logits[:, 0], baseline.event_logits[:, 0]
    )


def _action_invariant_checks(
    dynamics: Any,
    task: PublicPlanningTask,
    *,
    numerical_tolerance: float,
) -> tuple[bool, bool, bool, bool, bool]:
    """Return pre-action, once, isolation, conservation, and source checks.

    Every candidate is checked independently.  This prevents a direction- or
    magnitude-specific action path from passing merely because the first
    displayed candidate happened to obey the public action contract.
    """

    _validate_public_task(task)
    source_before = task.source_belief.clone()
    query_offsets = _diagnostic_query_offsets(task)
    baseline = dynamics.rollout(task.source_belief, query_offsets, action=None)
    pre_action = True
    exactly_once = True
    target_isolation = True
    conservation = True
    for action in task.candidates:
        acted = dynamics.rollout(task.source_belief, query_offsets, action=action)
        pre_action = pre_action and _pre_action_states_equal(acted, baseline)

        target_mask = action.validate_for(
            task.source_belief,
            latest_timestamp=task.source_belief.timestamp + query_offsets[:, -1],
        )
        target_slot = int(torch.nonzero(target_mask[0], as_tuple=False).item())
        boundary_matches = torch.nonzero(
            acted.timestamps[0] == action.timestamp[0],
            as_tuple=False,
        ).flatten()
        action_has_boundary = boundary_matches.numel() > 0
        boundary_index = int(boundary_matches[0]) if action_has_boundary else 0

        applied = acted.auxiliary.get("known_action_applied")
        known_impulse = acted.auxiliary.get("known_impulse_world")
        candidate_exactly_once = (
            action_has_boundary
            and isinstance(applied, Tensor)
            and applied.shape == acted.active_mask.shape
            and applied.dtype is torch.bool
            and int(applied.sum()) == 1
            and isinstance(known_impulse, Tensor)
            and known_impulse.shape == (*acted.active_mask.shape, 3)
        )
        if candidate_exactly_once:
            assert isinstance(applied, Tensor)
            assert isinstance(known_impulse, Tensor)
            event = torch.nonzero(applied, as_tuple=False)
            event_batch, event_time, event_slot = (int(value) for value in event[0])
            expected_known_impulse = torch.zeros_like(known_impulse)
            expected_known_impulse[event_batch, event_time, event_slot] = action.impulse_world[0]
            candidate_exactly_once = bool(
                event_batch == 0
                and event_slot == target_slot
                and bool(acted.timestamps[event_batch, event_time] == action.timestamp[event_batch])
                and _allclose(known_impulse, expected_known_impulse, numerical_tolerance)
            )
            boundary_index = event_time
        exactly_once = exactly_once and candidate_exactly_once

        position_delta = acted.positions[:, boundary_index] - baseline.positions[:, boundary_index]
        velocity_delta = (
            acted.velocities[:, boundary_index] - baseline.velocities[:, boundary_index]
        )
        expected_velocity_delta = torch.zeros_like(velocity_delta)
        expected_velocity_delta[0, target_slot] = (
            action.impulse_world[0] / task.source_belief.objects.mass[0, target_slot]
        )
        other_slots = torch.ones_like(target_mask)
        other_slots[0, target_slot] = False
        active_set_unchanged = torch.equal(
            acted.active_mask,
            task.frozen_active_mask.unsqueeze(1).expand_as(acted.active_mask),
        )
        candidate_isolated = (
            action_has_boundary
            and _allclose(position_delta, torch.zeros_like(position_delta), numerical_tolerance)
            and _allclose(
                velocity_delta.masked_select(other_slots.unsqueeze(-1)),
                torch.zeros_like(velocity_delta).masked_select(other_slots.unsqueeze(-1)),
                numerical_tolerance,
            )
            and _allclose(velocity_delta, expected_velocity_delta, numerical_tolerance)
            and active_set_unchanged
        )
        target_isolation = target_isolation and candidate_isolated

        mass = task.source_belief.objects.mass[..., 0]
        active = task.source_belief.objects.active
        momentum_delta = (
            (mass.unsqueeze(-1) * velocity_delta).masked_fill(~active.unsqueeze(-1), 0.0).sum(dim=1)
        )
        conservation = (
            conservation
            and action_has_boundary
            and _allclose(
                momentum_delta,
                action.impulse_world,
                numerical_tolerance,
            )
        )
    source_unchanged = _tensor_tree_equal(task.source_belief, source_before)
    return pre_action, exactly_once, target_isolation, conservation, source_unchanged


def _concatenate_beliefs(beliefs: Sequence[WorldBelief]) -> WorldBelief:
    if len(beliefs) < 2:
        raise ValueError("batch-independence diagnostics require at least two beliefs")
    first = beliefs[0]
    for belief in beliefs:
        belief.validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)
        if (
            belief.objects.max_objects != first.objects.max_objects
            or belief.objects.appearance_dim != first.objects.appearance_dim
            or belief.device != first.device
            or belief.dtype != first.dtype
            or belief.active_modalities != first.active_modalities
        ):
            raise ValueError("batch-independence beliefs must share one public schema")
    objects = first.objects.replace(
        **{
            item.name: torch.cat([getattr(belief.objects, item.name) for belief in beliefs], dim=0)
            for item in fields(first.objects)
        }
    )
    camera = first.camera.replace(
        **{
            item.name: torch.cat([getattr(belief.camera, item.name) for belief in beliefs], dim=0)
            for item in fields(first.camera)
        }
    )
    return first.replace(
        timestamp=torch.cat([belief.timestamp for belief in beliefs], dim=0),
        objects=objects,
        camera=camera,
        gravity=torch.cat([belief.gravity for belief in beliefs], dim=0),
        global_code=torch.cat([belief.global_code for belief in beliefs], dim=0),
        global_log_variance=torch.cat([belief.global_log_variance for belief in beliefs], dim=0),
        next_object_id=torch.cat([belief.next_object_id for belief in beliefs], dim=0),
        metadata={},
    ).validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)


def _trajectory_batch_row_matches(
    batched: BeliefTrajectory,
    batch_index: int,
    single: BeliefTrajectory,
    *,
    numerical_tolerance: float,
) -> bool:
    for name in (
        "timestamps",
        "positions",
        "velocities",
        "orientations",
        "motion_mode_logits",
        "fast_log_variance",
        "active_mask",
    ):
        if not _allclose(
            getattr(batched, name)[batch_index : batch_index + 1],
            getattr(single, name),
            numerical_tolerance,
        ):
            return False
    if (batched.event_logits is None) != (single.event_logits is None):
        return False
    if batched.event_logits is not None and not _allclose(
        batched.event_logits[batch_index : batch_index + 1],
        single.event_logits,
        numerical_tolerance,
    ):
        return False
    if batched.auxiliary.keys() != single.auxiliary.keys():
        return False
    return all(
        value.ndim > 0
        and value.shape[0] == batched.timestamps.shape[0]
        and _allclose(
            value[batch_index : batch_index + 1],
            single.auxiliary[name],
            numerical_tolerance,
        )
        for name, value in batched.auxiliary.items()
    )


def _batch_independence_check(
    dynamics: Any,
    tasks: Sequence[PublicPlanningTask],
    *,
    numerical_tolerance: float,
) -> tuple[bool, bool]:
    if len(tasks) < 2:
        raise ValueError("batch independence requires at least two bound planning tasks")
    for task in tasks:
        _validate_public_task(task)
    sources_before = tuple(task.source_belief.clone() for task in tasks)
    beliefs = tuple(task.source_belief for task in tasks)
    offsets = tuple(_diagnostic_query_offsets(task) for task in tasks)
    actions = tuple(task.candidates[0] for task in tasks)
    singles = tuple(
        dynamics.rollout(belief, query, action=action)
        for belief, query, action in zip(beliefs, offsets, actions, strict=True)
    )
    batched_belief = _concatenate_beliefs(beliefs)
    batched_offsets = torch.cat(offsets, dim=0)
    batched_action = WorldImpulseAction(
        timestamp=torch.cat([action.timestamp for action in actions], dim=0),
        object_id=torch.cat([action.object_id for action in actions], dim=0),
        impulse_world=torch.cat([action.impulse_world for action in actions], dim=0),
    )
    batched = dynamics.rollout(batched_belief, batched_offsets, action=batched_action)
    independent = all(
        _trajectory_batch_row_matches(
            batched,
            index,
            single,
            numerical_tolerance=numerical_tolerance,
        )
        for index, single in enumerate(singles)
    )
    sources_unchanged = all(
        _tensor_tree_equal(task.source_belief, before)
        for task, before in zip(tasks, sources_before, strict=True)
    )
    return independent, sources_unchanged


def measure_state_only_planning_latency(
    dynamics: Any,
    task: PublicPlanningTask,
    *,
    warmup_runs: int = 1,
    measured_runs: int = 5,
) -> float:
    """Return median B1/N=6 vectorized planning latency without perception."""

    return measure_state_only_planning_latency_population(
        dynamics,
        (task,),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )


def measure_state_only_planning_latency_population(
    dynamics: Any,
    tasks: Sequence[PublicPlanningTask],
    *,
    warmup_runs: int = 1,
    measured_runs: int = 5,
) -> float:
    """Return the median over a frozen population of B1/N=6 planning calls.

    Every task contributes the same number of timed samples.  This prevents a
    qualification result from depending on whichever N=6 row happened to be
    first in manifest order while preserving the single-task helper above.
    """

    try:
        population = tuple(tasks)
    except TypeError as error:
        raise TypeError("planning latency tasks must be a sequence") from error
    if not population:
        raise ValueError("planning latency requires at least one task")

    for name, value, positive in (
        ("warmup_runs", warmup_runs, False),
        ("measured_runs", measured_runs, True),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be a {qualifier} integer")

    for task in population:
        _validate_public_task(task)
        if task.row.object_count != 6:
            raise ValueError("planning latency must use only N=6 tasks")
        if task.source_belief.device.type != "cpu":
            raise ValueError("planning latency must use CPU tasks")

    def run_once(task: PublicPlanningTask) -> None:
        with torch.inference_mode():
            plan_counterfactual_actions(
                dynamics,
                task.source_belief,
                task.query_offsets,
                task.candidates,
                task.goal,
                weights=task.cost_weights,
                candidate_vectorized=True,
                return_events=False,
                return_auxiliary=False,
            )

    for task in population:
        for _ in range(warmup_runs):
            run_once(task)
    samples: list[float] = []
    for _ in range(measured_runs):
        for task in population:
            started = time.perf_counter()
            run_once(task)
            samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def evaluate_required_planning_invariants(
    dynamics: Any,
    invariant_task: PublicPlanningTask,
    batch_peer_task: PublicPlanningTask,
    latency_k8_task: PublicPlanningTask,
    latency_k32_task: PublicPlanningTask,
    *,
    additional_invariant_tasks: Sequence[PublicPlanningTask] = (),
    additional_latency_k8_tasks: Sequence[PublicPlanningTask] = (),
    additional_latency_k32_tasks: Sequence[PublicPlanningTask] = (),
    numerical_tolerance: float = 1.0e-6,
    latency_warmup_runs: int = 1,
    latency_measured_runs: int = 5,
) -> PlanningInvariantMetrics:
    """Evaluate required integrity and latency over a frozen task cover.

    ``additional_invariant_tasks`` lets a population evaluator cover every
    frozen public stratum.  The two additional latency populations make the
    reported K=8/K=32 values medians over all supplied N=6 tasks rather than
    measurements of a manifest-order exemplar.  Serial/vectorized parity for
    the full population remains owned by :func:`reduce_planning_task_outcomes`.
    """

    if (
        isinstance(numerical_tolerance, bool)
        or not isinstance(numerical_tolerance, Real)
        or not math.isfinite(float(numerical_tolerance))
        or numerical_tolerance < 0.0
    ):
        raise ValueError("numerical_tolerance must be finite and nonnegative")
    tolerance = float(numerical_tolerance)
    additional_groups: list[tuple[PublicPlanningTask, ...]] = []
    for name, tasks in (
        ("additional_invariant_tasks", additional_invariant_tasks),
        ("additional_latency_k8_tasks", additional_latency_k8_tasks),
        ("additional_latency_k32_tasks", additional_latency_k32_tasks),
    ):
        try:
            additional_groups.append(tuple(tasks))
        except TypeError as error:
            raise TypeError(f"{name} must be a sequence") from error
    additional, additional_latency_k8, additional_latency_k32 = additional_groups
    latency_k8_tasks = (latency_k8_task, *additional_latency_k8)
    latency_k32_tasks = (latency_k32_task, *additional_latency_k32)
    diagnostic_tasks = (
        invariant_task,
        batch_peer_task,
        *additional,
        *latency_k8_tasks,
        *latency_k32_tasks,
    )
    for task in diagnostic_tasks:
        _validate_public_task(task)
    diagnostic_sources_before = tuple(task.source_belief.clone() for task in diagnostic_tasks)
    if any(task.row.candidate_count != 8 for task in latency_k8_tasks):
        raise ValueError("latency_k8 tasks must contain eight candidates")
    if any(task.row.candidate_count != 32 for task in latency_k32_tasks):
        raise ValueError("latency_k32 tasks must contain 32 candidates")

    source_before = invariant_task.source_belief.clone()
    vectorized = plan_counterfactual_actions(
        dynamics,
        invariant_task.source_belief,
        invariant_task.query_offsets,
        invariant_task.candidates,
        invariant_task.goal,
        weights=invariant_task.cost_weights,
        candidate_vectorized=True,
        return_events=False,
        return_auxiliary=False,
    )
    serial = plan_counterfactual_actions(
        dynamics,
        invariant_task.source_belief,
        invariant_task.query_offsets,
        invariant_task.candidates,
        invariant_task.goal,
        weights=invariant_task.cost_weights,
        candidate_vectorized=False,
        return_events=False,
        return_auxiliary=False,
    )
    cost_difference = (vectorized.total_cost - serial.total_cost).abs()
    maximum_cost_difference = float(cost_difference.max()) if cost_difference.numel() else 0.0
    winner_parity = torch.equal(vectorized.selected_index, serial.selected_index)
    plans_freeze_active_set = _plan_active_set_is_frozen(
        invariant_task, vectorized
    ) and _plan_active_set_is_frozen(invariant_task, serial)
    source_unchanged = _tensor_tree_equal(invariant_task.source_belief, source_before)

    pre_action = True
    exactly_once = True
    isolation = True
    conservation = True
    action_source_unchanged = True
    for task in (invariant_task, batch_peer_task, *additional):
        task_pre_action, task_exactly_once, task_isolation, task_conservation, unchanged = (
            _action_invariant_checks(
                dynamics,
                task,
                numerical_tolerance=tolerance,
            )
        )
        pre_action = pre_action and task_pre_action
        exactly_once = exactly_once and task_exactly_once
        isolation = isolation and task_isolation
        conservation = conservation and task_conservation
        action_source_unchanged = action_source_unchanged and unchanged
    batch_independence, batch_sources_unchanged = _batch_independence_check(
        dynamics,
        (invariant_task, batch_peer_task, *additional),
        numerical_tolerance=tolerance,
    )
    latency_k8 = measure_state_only_planning_latency_population(
        dynamics,
        latency_k8_tasks,
        warmup_runs=latency_warmup_runs,
        measured_runs=latency_measured_runs,
    )
    latency_k32 = measure_state_only_planning_latency_population(
        dynamics,
        latency_k32_tasks,
        warmup_runs=latency_warmup_runs,
        measured_runs=latency_measured_runs,
    )
    all_sources_unchanged = all(
        _tensor_tree_equal(task.source_belief, before)
        for task, before in zip(
            diagnostic_tasks,
            diagnostic_sources_before,
            strict=True,
        )
    )
    return PlanningInvariantMetrics(
        serial_vectorized_winner_parity=winner_parity,
        maximum_cost_difference=maximum_cost_difference,
        pre_action_invariance=pre_action,
        exactly_once_impulse=exactly_once,
        action_target_isolation=isolation and plans_freeze_active_set,
        conservation=conservation,
        batch_independence=batch_independence,
        source_belief_unchanged=(
            source_unchanged
            and action_source_unchanged
            and batch_sources_unchanged
            and all_sources_unchanged
        ),
        latency_k8_seconds=latency_k8,
        latency_k32_seconds=latency_k32,
    )


def _validated_planning_cost_tolerance(
    task: PublicPlanningTask,
    cost_tolerance: float | None,
) -> float:
    tolerance = task.serial_vectorized_cost_tolerance if cost_tolerance is None else cost_tolerance
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("cost_tolerance must be a finite nonnegative real number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("cost_tolerance must be finite and nonnegative")
    return tolerance


def _evaluate_public_planning_decision(
    dynamics: Any,
    task: PublicPlanningTask,
    *,
    cost_tolerance: float | None = None,
) -> _PublicPlanningDecision:
    """Complete both public planners without accepting private evidence."""

    _validate_public_task(task)
    tolerance = _validated_planning_cost_tolerance(task, cost_tolerance)
    source_before = task.source_belief.clone()
    source_signature = tensor_identity_version_signature(task.source_belief)
    vectorized = plan_counterfactual_actions(
        dynamics,
        task.source_belief,
        task.query_offsets,
        task.candidates,
        task.goal,
        weights=task.cost_weights,
        candidate_vectorized=True,
        return_events=False,
        return_auxiliary=False,
    )
    serial = plan_counterfactual_actions(
        dynamics,
        task.source_belief,
        task.query_offsets,
        task.candidates,
        task.goal,
        weights=task.cost_weights,
        candidate_vectorized=False,
        return_events=False,
        return_auxiliary=False,
    )
    source_unchanged = (
        _tensor_tree_equal(
            task.source_belief,
            source_before,
        )
        and tensor_identity_version_signature(task.source_belief) == source_signature
    )
    return _PublicPlanningDecision(
        task=task,
        vectorized=vectorized,
        serial=serial,
        cost_tolerance=tolerance,
        source_belief_unchanged=source_unchanged,
    )


def _score_public_planning_decision(
    decision: _PublicPlanningDecision,
    oracle: PrivatePlanningOracleEvidence,
) -> PlanningTaskEvaluation:
    """Score one completed public decision after its oracle is authenticated."""

    task = decision.task
    vectorized = decision.vectorized
    serial = decision.serial
    tolerance = decision.cost_tolerance

    winner_parity = torch.equal(vectorized.selected_index, serial.selected_index)
    cost_difference = (vectorized.total_cost - serial.total_cost).abs()
    maximum_cost_difference = float(cost_difference.max()) if cost_difference.numel() else 0.0
    active_set_frozen = _plan_active_set_is_frozen(task, vectorized) and _plan_active_set_is_frozen(
        task, serial
    )
    model_winner = int(vectorized.selected_index.item())
    serial_winner = int(serial.selected_index.item())
    oracle_winner = oracle.certificate.winner_index
    regret = max(
        0.0,
        (float(oracle.candidate_costs[model_winner]) - float(oracle.candidate_costs[oracle_winner]))
        / oracle.regret_scale,
    )
    return PlanningTaskEvaluation(
        row=task.row,
        template_sha256=task.template_sha256,
        binding_sha256=task.binding_sha256,
        private_oracle_sha256=oracle.evidence_sha256,
        model_winner_index=model_winner,
        serial_winner_index=serial_winner,
        oracle_winner_index=oracle_winner,
        oracle_winner_correct=model_winner == oracle_winner,
        normalized_regret=regret,
        oracle_winner_succeeds=oracle.certificate.winner_succeeds,
        selected_action_goal_success=bool(oracle.candidate_goal_success[model_winner]),
        serial_vectorized_winner_parity=winner_parity,
        maximum_cost_difference=maximum_cost_difference,
        cost_agreement_within_tolerance=(
            math.isfinite(maximum_cost_difference) and maximum_cost_difference <= tolerance
        ),
        active_set_frozen=active_set_frozen,
        source_belief_unchanged=decision.source_belief_unchanged,
    )


def evaluate_certified_planning_task(
    dynamics: Any,
    task: PublicPlanningTask,
    oracle: PrivatePlanningOracleEvidence,
    *,
    cost_tolerance: float | None = None,
) -> PlanningTaskEvaluation:
    """Run model plans first, then compare their decision with private truth."""

    decision = _evaluate_public_planning_decision(
        dynamics,
        task,
        cost_tolerance=cost_tolerance,
    )
    # Never inspect private costs, success labels, event labels, or even their
    # template binding until both public planner implementations complete.
    _validate_private_oracle(task, oracle)
    return _score_public_planning_decision(decision, oracle)


def evaluate_certified_planning_task_pair(
    candidate_dynamics: Any,
    candidate_task: PublicPlanningTask,
    reference_dynamics: Any,
    reference_task: PublicPlanningTask,
    oracle: PrivatePlanningOracleEvidence,
    *,
    cost_tolerance: float | None = None,
) -> tuple[PlanningTaskEvaluation, PlanningTaskEvaluation]:
    """Plan both checkpoints publicly before opening their one shared oracle."""

    _validate_public_task(candidate_task)
    _validate_public_task(reference_task)
    if candidate_task.row != reference_task.row:
        raise ValueError("paired planning tasks refer to different manifest rows")
    if candidate_task.template_sha256 != reference_task.template_sha256:
        raise ValueError("paired planning tasks refer to different public templates")

    candidate_decision = _evaluate_public_planning_decision(
        candidate_dynamics,
        candidate_task,
        cost_tolerance=cost_tolerance,
    )
    reference_decision = _evaluate_public_planning_decision(
        reference_dynamics,
        reference_task,
        cost_tolerance=cost_tolerance,
    )

    # This is the paired authorization boundary: candidate and reference have
    # each completed vectorized and serial planning before any private oracle
    # field is authenticated or scored.
    _validate_private_oracle(candidate_task, oracle)
    return (
        _score_public_planning_decision(candidate_decision, oracle),
        _score_public_planning_decision(reference_decision, oracle),
    )


def _quantile_nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _sample_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def reduce_planning_task_outcomes(
    outcomes: Sequence[PlanningTaskOutcome],
    *,
    cost_tolerance: float = 1.0e-6,
) -> PlanningEvaluationReduction:
    """Pool task outcomes into the exact cardinality/K/distribution slices."""

    if isinstance(cost_tolerance, bool) or not isinstance(cost_tolerance, Real):
        raise TypeError("cost_tolerance must be a finite nonnegative real number")
    cost_tolerance = float(cost_tolerance)
    if not math.isfinite(cost_tolerance) or cost_tolerance < 0.0:
        raise ValueError("cost_tolerance must be finite and nonnegative")
    if not outcomes:
        raise ValueError("at least one planning outcome is required")

    grouped: dict[tuple[int, int, str], list[PlanningTaskOutcome]] = {}
    seen: set[tuple[str, int, int]] = set()
    evaluations: list[PlanningTaskEvaluation] = []
    for outcome in outcomes:
        if not isinstance(outcome, PlanningTaskOutcome):
            raise TypeError("outcomes must contain PlanningTaskOutcome values")
        row = _validate_manifest_row(outcome.row)
        key = (row.split, row.ordinal, row.seed)
        if key in seen:
            raise ValueError("planning outcomes contain a duplicate task row")
        seen.add(key)
        if outcome.handle_resolved:
            if outcome.evaluation is None or outcome.evaluation.row != row:
                raise ValueError("resolved outcome must contain its matching evaluation")
            if outcome.failure_reason is not None:
                raise ValueError("resolved outcome cannot contain a failure reason")
            evaluations.append(outcome.evaluation)
        elif outcome.evaluation is not None or not outcome.failure_reason:
            raise ValueError("unresolved outcome must contain only a failure reason")
        grouped.setdefault((row.object_count, row.candidate_count, row.distribution), []).append(
            outcome
        )

    slices: list[PlanningSliceMetrics] = []
    for (object_count, candidate_count, distribution), group in sorted(grouped.items()):
        resolved = [item.evaluation for item in group if item.evaluation is not None]
        unresolved_count = len(group) - len(resolved)
        # Handle failures are supported planning failures, not permission to
        # shrink the winner/regret denominator.  Every accepted 1.61 task has
        # a privately certified successful oracle winner, so unresolved rows
        # conservatively contribute one winner error, unit regret, and one
        # selected-action failure without opening their private oracle.
        regrets = [item.normalized_regret for item in resolved] + [1.0] * unresolved_count
        successful_oracle = [item for item in resolved if item.oracle_winner_succeeds]
        successful_oracle_support = len(successful_oracle) + unresolved_count
        slices.append(
            PlanningSliceMetrics(
                object_count=object_count,
                candidate_count=candidate_count,
                distribution=distribution,
                handle_resolution=SupportedScalar(
                    value=len(resolved) / len(group),
                    support=len(group),
                ),
                oracle_winner_accuracy=SupportedScalar(
                    value=sum(item.oracle_winner_correct for item in resolved) / len(group),
                    support=len(group),
                ),
                normalized_regret_median=SupportedScalar(
                    value=_sample_median(regrets),
                    support=len(regrets),
                ),
                normalized_regret_p95=SupportedScalar(
                    value=_quantile_nearest_rank(regrets, 0.95),
                    support=len(regrets),
                ),
                successful_oracle_goal_success=SupportedScalar(
                    value=(
                        sum(item.selected_action_goal_success for item in successful_oracle)
                        / successful_oracle_support
                        if successful_oracle_support
                        else 0.0
                    ),
                    support=successful_oracle_support,
                ),
            )
        )

    def aggregate(group: Sequence[PlanningTaskOutcome]) -> tuple[list[PlanningTaskEvaluation], int]:
        resolved = [item.evaluation for item in group if item.evaluation is not None]
        return resolved, len(group) - len(resolved)

    handle_resolution_by_object_count: dict[str, SupportedScalar] = {}
    for object_count in sorted({item.row.object_count for item in outcomes}):
        group = [item for item in outcomes if item.row.object_count == object_count]
        resolved, _unresolved = aggregate(group)
        handle_resolution_by_object_count[f"N{object_count}"] = SupportedScalar(
            value=len(resolved) / len(group),
            support=len(group),
        )

    winner_accuracy_by_candidate_distribution: dict[str, SupportedScalar] = {}
    winner_keys = sorted({(item.row.candidate_count, item.row.distribution) for item in outcomes})
    for candidate_count, distribution in winner_keys:
        group = [
            item
            for item in outcomes
            if item.row.candidate_count == candidate_count and item.row.distribution == distribution
        ]
        resolved, _unresolved = aggregate(group)
        winner_accuracy_by_candidate_distribution[f"K{candidate_count}/{distribution}"] = (
            SupportedScalar(
                value=sum(item.oracle_winner_correct for item in resolved) / len(group),
                support=len(group),
            )
        )

    regret_median_by_candidate_count: dict[str, SupportedScalar] = {}
    regret_p95_by_candidate_count: dict[str, SupportedScalar] = {}
    for candidate_count in sorted({item.row.candidate_count for item in outcomes}):
        group = [item for item in outcomes if item.row.candidate_count == candidate_count]
        resolved, unresolved_count = aggregate(group)
        regrets = [item.normalized_regret for item in resolved] + [1.0] * unresolved_count
        regret_median_by_candidate_count[f"K{candidate_count}"] = SupportedScalar(
            value=_sample_median(regrets),
            support=len(regrets),
        )
        regret_p95_by_candidate_count[f"K{candidate_count}"] = SupportedScalar(
            value=_quantile_nearest_rank(regrets, 0.95),
            support=len(regrets),
        )

    all_resolved, unresolved_count = aggregate(outcomes)
    successful_oracle = [item for item in all_resolved if item.oracle_winner_succeeds]
    successful_support = len(successful_oracle) + unresolved_count
    gate_metrics = PlanningPopulationGateMetrics(
        handle_resolution_by_object_count=handle_resolution_by_object_count,
        oracle_winner_accuracy_by_candidate_distribution=(
            winner_accuracy_by_candidate_distribution
        ),
        normalized_regret_median_by_candidate_count=regret_median_by_candidate_count,
        normalized_regret_p95_by_candidate_count=regret_p95_by_candidate_count,
        successful_oracle_goal_success=SupportedScalar(
            value=(
                sum(item.selected_action_goal_success for item in successful_oracle)
                / successful_support
                if successful_support
                else 0.0
            ),
            support=successful_support,
        ),
    )

    maximum_cost_difference = max(
        (item.maximum_cost_difference for item in evaluations),
        default=0.0,
    )
    return PlanningEvaluationReduction(
        slices=tuple(slices),
        gate_metrics=gate_metrics,
        serial_vectorized_winner_parity=bool(evaluations)
        and all(item.serial_vectorized_winner_parity for item in evaluations),
        maximum_cost_difference=maximum_cost_difference,
        cost_agreement_within_tolerance=(
            bool(evaluations)
            and all(item.cost_agreement_within_tolerance for item in evaluations)
            and math.isfinite(maximum_cost_difference)
            and maximum_cost_difference <= cost_tolerance
        ),
        active_set_frozen=bool(evaluations) and all(item.active_set_frozen for item in evaluations),
        source_belief_unchanged=bool(evaluations)
        and all(item.source_belief_unchanged for item in evaluations),
    )


__all__ = [
    "DEFAULT_PLANNING_TASK_CONFIG",
    "PlanningCandidateDescriptor",
    "PlanningEvaluationReduction",
    "PlanningHistoryEvidence",
    "PlanningTaskConfig",
    "PlanningTaskEvaluation",
    "PlanningTaskOutcome",
    "PrivatePlanningOracleEvidence",
    "PublicPlanningTemplate",
    "PublicPlanningTask",
    "bind_public_planning_task",
    "certify_private_planning_oracle",
    "evaluate_certified_planning_task",
    "evaluate_certified_planning_task_pair",
    "evaluate_required_planning_invariants",
    "materialize_public_planning_template",
    "measure_state_only_planning_latency",
    "measure_state_only_planning_latency_population",
    "ranked_observable_appearance_handle",
    "reduce_planning_task_outcomes",
    "resolve_planning_target_handle",
    "validate_public_planning_template",
    "validate_planning_manifest_row",
]
