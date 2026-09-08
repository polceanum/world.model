"""Concrete RGB-D planning populations for specification 1.61.

The materializer has a deliberately narrow trust boundary.  Public history
contains only calibrated RGB-D frames and past, observable membership.  The
future candidate bank is frozen in :class:`PublicPlanningTemplate`; simulator
IDs, padded slots, masks, and future lifecycle labels never cross into either
public object.  A separate private oracle is generated with the data
simulator, independent action timing, and an independently implemented cost;
it is inspected only after a checkpoint has planned.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from world_model.belief import BeliefFactory, MotionMode, WorldBelief
from world_model.observations import ObservationPacket
from world_model.observations.rgbd.sphere_centres import (
    metric_sphere_centres_from_surface_depth,
)
from world_model.observations.rgbd.temporal import RGBDTemporalPositionHistory
from world_model.runtime.prepared import tensor_identity_version_signature
from world_model.runtime.state import runtime_stream_key
from world_model.simulator.camera import (
    CameraFrame,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
)
from world_model.simulator.physics import (
    PhysicsConfig,
    SphereState,
)
from world_model.simulator.renderer import RenderOutput, render_spheres
from world_model.training.dynamic_set_gates import PlanningInvariantMetrics
from world_model.training.dynamic_set_physics import (
    advance_spheres_no_boundary_contacts_prevalidated,
    prevalidated_sphere_pair_cache,
)
from world_model.training.dynamic_set_planning import (
    PLANNING_LOG_VARIANCE_BOUNDS,
    PlanningEvaluationReduction,
    PlanningHistoryEvidence,
    PlanningTaskOutcome,
    PrivatePlanningOracleEvidence,
    PublicPlanningTask,
    PublicPlanningTemplate,
    bind_public_planning_task,
    certify_private_planning_oracle,
    evaluate_certified_planning_task,
    evaluate_certified_planning_task_pair,
    evaluate_required_planning_invariants,
    materialize_public_planning_template,
    ranked_observable_appearance_handle,
    reduce_planning_task_outcomes,
    validate_planning_manifest_row,
    validate_public_planning_template,
)
from world_model.training.dynamic_set_protocol import (
    FROZEN_PLANNING_MANIFEST_SHA256,
    PLANNING_SPLIT_SIZES,
    PlanningManifestRow,
    canonical_sha256,
)
from world_model.training.dynamic_set_scene import (
    DYNAMIC_SET_DRAG,
    DYNAMIC_SET_FRAME_RATE_HZ,
    DYNAMIC_SET_FRICTION,
    DYNAMIC_SET_IMAGE_SIZE,
    DYNAMIC_SET_MASS,
    DYNAMIC_SET_MAX_OBJECTS,
    DYNAMIC_SET_RADIUS_M,
    DYNAMIC_SET_RESTITUTION,
    DYNAMIC_SET_VERTICAL_FOV_DEGREES,
)
from world_model.training.qualification_core import (
    OrderedSplitLedger,
    SplitPermit,
    validated_sha256,
)

PLANNING_HISTORY_FRAMES = 22
PLANNING_DYNAMIC_BIRTH_FRAME = 4
PLANNING_HISTORY_SENSOR_ID = "planning_rgbd"
PLANNING_ORACLE_PHYSICS_HZ = 120.0
DEFAULT_PLANNING_MATERIALIZATION_ATTEMPTS = 8

_PROTECTED_SPLITS = frozenset({"selector", "confirmation", "final_test", "compositional_ood"})
_PUBLIC_MATERIALIZATION_SPLITS = frozenset({"training", "development"})
_PROTECTED_CAPABILITY_AUTHORITY = object()
_PROTECTED_SPLIT_INDICES: Mapping[str, int] = {
    "selector": 0,
    "confirmation": 1,
    "final_test": 2,
    "compositional_ood": 3,
}
PLANNING_POPULATION_CLAIM_PURPOSE = "dynamic_set_planning_population_v1"
_GOAL_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
_ALBEDO_PALETTE = torch.tensor(
    [
        [0.98, 0.02, 0.02],
        [0.02, 0.98, 0.02],
        [0.02, 0.02, 0.98],
        [0.98, 0.98, 0.02],
        [0.02, 0.98, 0.98],
        [0.98, 0.02, 0.98],
    ],
    dtype=torch.float32,
)
_SAFE_CAMERA_XY = (
    (-1.65, -0.95),
    (0.00, -0.95),
    (1.65, -0.95),
    (-1.65, 0.95),
    (0.00, 0.95),
    (1.65, 0.95),
)

# Frozen, lower-is-better population score used for the 15% planning
# winner/regret/task selection component.  An unresolved handle receives one
# in every downstream component without requiring the private oracle to open.
PLANNING_ERROR_WEIGHTS: Mapping[str, float] = {
    "oracle_winner_error": 1.0 / 3.0,
    "normalized_regret": 1.0 / 3.0,
    "successful_oracle_task_error": 1.0 / 3.0,
}


@dataclass(frozen=True)
class PublicPlanningHistoryFrame:
    """One truth-free calibrated observation in a planning prefix."""

    frame_index: int
    timestamp: float
    rgb: Tensor
    depth: Tensor
    world_from_camera: Tensor
    intrinsics: Tensor

    def packet(self) -> ObservationPacket:
        return ObservationPacket(
            modality="rgbd",
            sensor_id=PLANNING_HISTORY_SENSOR_ID,
            timestamp=self.timestamp,
            payload={
                "rgb": self.rgb.detach().clone().unsqueeze(0),
                "depth": self.depth.detach().clone().unsqueeze(0),
            },
            calibration={
                "world_from_camera": self.world_from_camera.detach().clone().unsqueeze(0),
                "intrinsics": self.intrinsics.detach().clone().unsqueeze(0),
            },
            frame_id=f"planning:{self.frame_index}",
            metadata={"image_size": DYNAMIC_SET_IMAGE_SIZE},
        )


@dataclass(frozen=True)
class PublicPlanningHistory:
    """ID-free RGB-D prefix plus observable past-membership provenance."""

    frames: tuple[PublicPlanningHistoryFrame, ...]
    previously_dynamic: bool
    history_sha256: str


@dataclass(frozen=True)
class PlanningTaskMaterialization:
    """Public history/template and separately bound private oracle ledger."""

    public_history: PublicPlanningHistory
    template: PublicPlanningTemplate
    private_oracle: PrivatePlanningOracleEvidence
    accepted_seed: int
    attempt_count: int
    rejection_reasons: tuple[str, ...]
    materialization_sha256: str

    @property
    def row(self) -> PlanningManifestRow:
        return self.template.row


class PlanningTaskMaterializationError(RuntimeError):
    """A deterministic row could not pass visibility/oracle certification."""


class PlanningTaskUnresolvedError(RuntimeError):
    """A public checkpoint history cannot resolve a mature planning target."""


class _ProtectedPlanningMaterializationCapability:
    """Ordered, one-shot authority for one protected planning population."""

    __slots__ = (
        "_closed",
        "_cursor",
        "_manifest_sha256",
        "_permit_index",
        "_permit_nonce",
        "_protocol_sha256",
        "_rows",
        "_split",
    )

    def __init__(
        self,
        *,
        authority: object,
        split: str,
        rows: Sequence[PlanningManifestRow],
        protocol_sha256: str,
        manifest_sha256: str,
        permit_index: int,
        permit_nonce: str,
    ) -> None:
        if authority is not _PROTECTED_CAPABILITY_AUTHORITY:
            raise PermissionError("protected planning capabilities are evaluator-owned")
        resolved = tuple(rows)
        if split not in _PROTECTED_SPLITS:
            raise PermissionError("protected planning capability has a public split")
        if not resolved or any(row.split != split for row in resolved):
            raise ValueError("protected planning capability rows differ from its split")
        self._split = split
        self._rows = resolved
        self._protocol_sha256 = protocol_sha256
        self._manifest_sha256 = manifest_sha256
        self._permit_index = permit_index
        self._permit_nonce = permit_nonce
        self._cursor = 0
        self._closed = False

    def claim(self, row: PlanningManifestRow) -> None:
        if self._closed:
            raise PermissionError("protected planning capability is closed or already consumed")
        if self._cursor >= len(self._rows) or row != self._rows[self._cursor]:
            self._closed = True
            raise PermissionError("protected planning capability row order differs")
        self._cursor += 1

    def finish(self) -> None:
        if self._closed:
            raise PermissionError("protected planning capability is closed")
        if self._cursor != len(self._rows):
            self._closed = True
            raise PermissionError("protected planning capability population is incomplete")
        self._closed = True

    def close(self) -> None:
        self._closed = True


def _mint_protected_planning_materialization_capability(
    *,
    split: str,
    rows: Sequence[PlanningManifestRow],
    protocol_sha256: str,
    manifest_sha256: str,
    permit_index: int,
    permit_nonce: str,
) -> _ProtectedPlanningMaterializationCapability:
    """Mint one internal capability after the evaluator validates bindings."""

    return _ProtectedPlanningMaterializationCapability(
        authority=_PROTECTED_CAPABILITY_AUTHORITY,
        split=split,
        rows=rows,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        permit_index=permit_index,
        permit_nonce=permit_nonce,
    )


@dataclass(frozen=True)
class PlanningPopulationEvaluationConfig:
    """Evaluation-only controls; none participates in optimisation."""

    cost_tolerance: float = 1.0e-6
    require_complete_slices: bool = True
    evaluate_invariants: bool = True
    latency_warmup_runs: int = 1
    latency_measured_runs: int = 5

    def validate(self) -> PlanningPopulationEvaluationConfig:
        if (
            isinstance(self.cost_tolerance, bool)
            or not isinstance(self.cost_tolerance, Real)
            or not math.isfinite(float(self.cost_tolerance))
            or self.cost_tolerance < 0.0
        ):
            raise ValueError("cost_tolerance must be finite and nonnegative")
        for name, value, positive in (
            ("latency_warmup_runs", self.latency_warmup_runs, False),
            ("latency_measured_runs", self.latency_measured_runs, True),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
                qualifier = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be a {qualifier} integer")
        if type(self.require_complete_slices) is not bool:
            raise TypeError("require_complete_slices must be boolean")
        if type(self.evaluate_invariants) is not bool:
            raise TypeError("evaluate_invariants must be boolean")
        return self


DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG = PlanningPopulationEvaluationConfig().validate()


@dataclass(frozen=True)
class PlanningPopulationBinding:
    split: str
    protocol_sha256: str | None
    manifest_sha256: str | None
    permit_index: int | None
    row_count: int


@dataclass(frozen=True)
class PlanningPopulationEvaluationResult:
    outcomes: tuple[PlanningTaskOutcome, ...]
    reduction: PlanningEvaluationReduction
    invariants: PlanningInvariantMetrics
    planning_error: float
    population_binding: PlanningPopulationBinding
    result_sha256: str


@dataclass(frozen=True, slots=True)
class PlanningPairProvenance:
    """Digest-only proof that both checkpoints consumed one exact task row."""

    split: str
    ordinal: int
    seed: int
    row_sha256: str
    public_history_sha256: str
    public_template_sha256: str
    public_materialization_sha256: str
    private_oracle_sha256: str
    complete_materialization_sha256: str
    provenance_sha256: str

    @classmethod
    def create(
        cls,
        materialization: PlanningTaskMaterialization,
    ) -> PlanningPairProvenance:
        """Bind one complete materialization after both public passes return."""

        # Both public checkpoint passes have completed before this constructor
        # is called.  It can therefore authenticate the private oracle and
        # retain only opaque digests proving that candidate and reference were
        # scored (or failed handle resolution) on one complete materialization.
        _validate_complete_materialization(materialization)
        row = materialization.row
        public_materialization_sha256 = canonical_sha256(
            {
                "schema": "dynamic_set_public_planning_materialization_v1",
                "row": asdict(row),
                "public_history_sha256": materialization.public_history.history_sha256,
                "public_template_sha256": materialization.template.template_sha256,
                "accepted_seed": materialization.accepted_seed,
                "attempt_count": materialization.attempt_count,
                "rejection_reasons": list(materialization.rejection_reasons),
            }
        )
        body = {
            "split": row.split,
            "ordinal": row.ordinal,
            "seed": row.seed,
            "row_sha256": canonical_sha256(asdict(row)),
            "public_history_sha256": materialization.public_history.history_sha256,
            "public_template_sha256": materialization.template.template_sha256,
            "public_materialization_sha256": public_materialization_sha256,
            "private_oracle_sha256": materialization.private_oracle.evidence_sha256,
            "complete_materialization_sha256": materialization.materialization_sha256,
        }
        return cls(**body, provenance_sha256=canonical_sha256(body))

    def validate(self, row: PlanningManifestRow) -> PlanningPairProvenance:
        """Validate serialized provenance against its aligned manifest row."""

        _validate_row(row)
        for label, value in (
            ("row_sha256", self.row_sha256),
            ("public_history_sha256", self.public_history_sha256),
            ("public_template_sha256", self.public_template_sha256),
            ("public_materialization_sha256", self.public_materialization_sha256),
            ("private_oracle_sha256", self.private_oracle_sha256),
            ("complete_materialization_sha256", self.complete_materialization_sha256),
            ("provenance_sha256", self.provenance_sha256),
        ):
            validated_sha256(value, label=f"planning pair {label}")
        if (self.split, self.ordinal, self.seed) != (row.split, row.ordinal, row.seed):
            raise ValueError("planning pair provenance differs from its aligned row")
        if self.row_sha256 != canonical_sha256(asdict(row)):
            raise ValueError("planning pair provenance row digest differs")
        body = {
            "split": self.split,
            "ordinal": self.ordinal,
            "seed": self.seed,
            "row_sha256": self.row_sha256,
            "public_history_sha256": self.public_history_sha256,
            "public_template_sha256": self.public_template_sha256,
            "public_materialization_sha256": self.public_materialization_sha256,
            "private_oracle_sha256": self.private_oracle_sha256,
            "complete_materialization_sha256": self.complete_materialization_sha256,
        }
        if canonical_sha256(body) != self.provenance_sha256:
            raise ValueError("planning pair provenance digest differs")
        return self


@dataclass(frozen=True, slots=True)
class PlanningPairedPopulationEvaluationResult:
    """Candidate/reference planning results from one materialization stream."""

    candidate: PlanningPopulationEvaluationResult
    reference: PlanningPopulationEvaluationResult
    provenance: tuple[PlanningPairProvenance, ...]
    result_sha256: str

    def validate(self) -> PlanningPairedPopulationEvaluationResult:
        validate_planning_population_evaluation_result(self.candidate)
        validate_planning_population_evaluation_result(self.reference)
        if self.candidate.population_binding != self.reference.population_binding:
            raise ValueError("paired planning population bindings differ")
        candidate_rows = tuple(outcome.row for outcome in self.candidate.outcomes)
        reference_rows = tuple(outcome.row for outcome in self.reference.outcomes)
        if candidate_rows != reference_rows:
            raise ValueError("paired planning candidate/reference row alignment differs")
        if len(self.provenance) != len(candidate_rows):
            raise ValueError("paired planning provenance count differs from its rows")
        seen: set[tuple[str, int, int, str]] = set()
        for item, candidate_outcome, reference_outcome in zip(
            self.provenance,
            self.candidate.outcomes,
            self.reference.outcomes,
            strict=True,
        ):
            if not isinstance(item, PlanningPairProvenance):
                raise TypeError("paired planning provenance has an invalid type")
            item.validate(candidate_outcome.row)
            key = (item.split, item.ordinal, item.seed, item.row_sha256)
            if key in seen:
                raise ValueError("paired planning provenance contains a duplicate row")
            seen.add(key)
            for outcome in (candidate_outcome, reference_outcome):
                if outcome.evaluation is not None and (
                    outcome.evaluation.template_sha256 != item.public_template_sha256
                    or outcome.evaluation.private_oracle_sha256 != item.private_oracle_sha256
                ):
                    raise ValueError(
                        "paired planning outcome differs from its template/oracle provenance"
                    )
        validated_sha256(self.result_sha256, label="paired planning result digest")
        expected = canonical_sha256(_paired_population_result_payload(self))
        if expected != self.result_sha256:
            raise ValueError("paired planning result differs from its evidence digest")
        return self


def _validate_row(row: PlanningManifestRow) -> PlanningManifestRow:
    return validate_planning_manifest_row(row)


def _candidate_seed(seed: int, attempt_index: int) -> int:
    if attempt_index == 0:
        return int(seed)
    digest = hashlib.sha256(f"planning-1.61:{seed}:{attempt_index}".encode("ascii")).digest()
    return (int.from_bytes(digest[:8], "big") & ((1 << 62) - 1)) | (1 << 62)


def _camera(row: PlanningManifestRow) -> CameraFrame:
    angle = 2.0 * math.pi * row.camera_stratum / 8.0
    position = torch.tensor(
        [10.0 * math.sin(angle), 0.0, -10.0 * math.cos(angle)],
        dtype=torch.float32,
    )
    target = torch.zeros(3, dtype=torch.float32)
    world_from_camera = look_at_world_from_camera(position, target)
    camera = CameraFrame(
        timestamp=0.0,
        world_from_camera=world_from_camera,
        camera_from_world=invert_rigid_transform(world_from_camera),
        intrinsics=make_intrinsics(
            DYNAMIC_SET_IMAGE_SIZE,
            DYNAMIC_SET_VERTICAL_FOV_DEGREES,
        ),
        position=position,
        target=target,
    )
    camera.validate()
    return camera


def _camera_to_world(values: Tensor, camera: CameraFrame, *, point: bool) -> Tensor:
    result = values @ camera.world_from_camera[:3, :3].transpose(0, 1)
    if point:
        result = result + camera.world_from_camera[:3, 3]
    return result


def _base_camera_positions(candidate_seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(candidate_seed & 0x7FFF_FFFF_FFFF_FFFF)
    jitter = 0.12 * (torch.rand(2, generator=generator) - 0.5)
    depth = 10.0 + 0.20 * (float(torch.rand((), generator=generator)) - 0.5)
    positions = torch.zeros((DYNAMIC_SET_MAX_OBJECTS, 3), dtype=torch.float32)
    for slot, (x, y) in enumerate(_SAFE_CAMERA_XY):
        positions[slot] = torch.tensor(
            [x + float(jitter[0]), y + float(jitter[1]), depth],
            dtype=torch.float32,
        )
    return positions


def _make_state(
    row: PlanningManifestRow,
    camera: CameraFrame,
    camera_positions: Tensor,
    *,
    before_birth: bool,
) -> SphereState:
    active = torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.bool)
    active[: row.object_count] = True
    if before_birth and row.previously_dynamic:
        active[row.object_count - 1] = False
    private_ids = torch.arange(
        90_000_000 + row.seed % 1_000_000,
        90_000_000 + row.seed % 1_000_000 + DYNAMIC_SET_MAX_OBJECTS,
        dtype=torch.int64,
    )
    private_ids = torch.where(active, private_ids, torch.full_like(private_ids, -1))
    orientation = torch.zeros(DYNAMIC_SET_MAX_OBJECTS, 4, dtype=torch.float32)
    orientation[:, 3] = 1.0
    state = SphereState(
        object_id=private_ids,
        active=active,
        position=_camera_to_world(camera_positions, camera, point=True),
        velocity=torch.zeros(DYNAMIC_SET_MAX_OBJECTS, 3, dtype=torch.float32),
        radius=torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_RADIUS_M),
        mass=torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_MASS),
        restitution=torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_RESTITUTION),
        drag=torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_DRAG),
        friction=torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_FRICTION),
        albedo=_ALBEDO_PALETTE.clone(),
        orientation=orientation,
        angular_velocity=torch.zeros(DYNAMIC_SET_MAX_OBJECTS, 3, dtype=torch.float32),
        sleeping=torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.bool),
        sleep_counter=torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.int64),
    )
    state.validate()
    return state


def _observable_appearance(image: Tensor, mask: Tensor) -> Tensor:
    """Exact eight-value appearance statistic used by RGB-D set mode."""

    if image.shape != (3, *DYNAMIC_SET_IMAGE_SIZE) or image.dtype is not torch.float32:
        raise ValueError("appearance extraction requires one float32 RGB frame")
    if mask.shape != DYNAMIC_SET_IMAGE_SIZE or mask.dtype is not torch.bool or not bool(mask.any()):
        raise ValueError("appearance extraction requires one nonempty boolean mask")
    weight = mask.to(image.dtype)
    epsilon = torch.finfo(image.dtype).eps
    mass = weight.sum().clamp_min(epsilon)
    mean_rgb = torch.einsum("hw,chw->c", weight, image) / mass
    second_rgb = torch.einsum("hw,chw->c", weight, image.square()) / mass
    std_rgb = (second_rgb - mean_rgb.square() + epsilon).clamp_min(epsilon).sqrt()
    intensity = image.mean(dim=0)
    mean_intensity = torch.einsum("hw,hw->", weight, intensity) / mass
    second_intensity = torch.einsum("hw,hw->", weight, intensity.square()) / mass
    std_intensity = (second_intensity - mean_intensity.square() + epsilon).clamp_min(epsilon).sqrt()
    descriptor = torch.cat(
        (mean_rgb, std_rgb, mean_intensity.unsqueeze(0), std_intensity.unsqueeze(0))
    )
    return F.normalize(descriptor, dim=0, eps=epsilon)


def _appearance_descriptors(rendered: RenderOutput, active: Tensor) -> Tensor:
    result = torch.zeros(DYNAMIC_SET_MAX_OBJECTS, 8, dtype=torch.float32)
    for slot in torch.nonzero(active, as_tuple=False).flatten().tolist():
        result[slot] = _observable_appearance(rendered.rgb, rendered.visible_mask[slot])
    return result


def _ranked_target_slot(descriptors: Tensor, active: Tensor, rank: int) -> int:
    slots = torch.nonzero(active, as_tuple=False).flatten().tolist()
    keys = {slot: tuple(float(value) for value in descriptors[slot]) for slot in slots}
    if len(set(keys.values())) != len(keys):
        raise ValueError("rendered appearance handles are not observably distinct")
    return sorted(slots, key=keys.__getitem__)[rank]


def _contact_direction(row: PlanningManifestRow, camera: CameraFrame) -> Tensor:
    goal = torch.tensor(_GOAL_DIRECTIONS[row.goal_direction], dtype=torch.float32)
    forward = camera.world_from_camera[:3, 2]
    direction = torch.linalg.cross(forward, goal)
    if float(torch.linalg.vector_norm(direction)) < 1.0e-6:
        direction = camera.world_from_camera[:3, 0]
    direction = direction / torch.linalg.vector_norm(direction)
    if abs(float(torch.dot(direction, goal))) > 1.0e-5:
        raise RuntimeError("contact direction must remain orthogonal to the goal")
    return direction


def _layout_for_row(
    row: PlanningManifestRow,
    camera: CameraFrame,
    candidate_seed: int,
) -> tuple[Tensor, int]:
    camera_positions = _base_camera_positions(candidate_seed)
    final = _make_state(row, camera, camera_positions, before_birth=False)
    rendered = render_spheres(final, camera, DYNAMIC_SET_IMAGE_SIZE, noise_std=0.0)
    descriptors = _appearance_descriptors(rendered, final.active)
    target_slot = _ranked_target_slot(descriptors, final.active, row.target_rank)
    if not row.candidate_induced_contact:
        return camera_positions, target_slot

    # Re-evaluate the rank after moving the pair.  Descriptor ordering is
    # colour-dominated, but this bounded fixed point prevents position-dependent
    # shading from ever changing which observable target owns the contact pair.
    for _ in range(DYNAMIC_SET_MAX_OBJECTS + 1):
        companion = next(slot for slot in range(row.object_count) if slot != target_slot)
        reset = _base_camera_positions(candidate_seed)
        non_pair = [
            slot for slot in range(row.object_count) if slot not in (target_slot, companion)
        ]
        if non_pair:
            non_pair_x = torch.linspace(-2.40, 2.40, len(non_pair))
            for index, slot in enumerate(non_pair):
                reset[slot] = torch.tensor(
                    [float(non_pair_x[index]), -1.15, float(reset[0, 2])],
                    dtype=torch.float32,
                )
        pair_centre_camera = torch.tensor([0.0, 0.95, float(reset[0, 2])], dtype=torch.float32)
        pair_centre_world = _camera_to_world(pair_centre_camera[None], camera, point=True)[0]
        direction_world = _contact_direction(row, camera)
        target_world = pair_centre_world - 0.34 * direction_world
        companion_world = pair_centre_world + 0.34 * direction_world
        homogeneous = torch.cat(
            (
                torch.stack((target_world, companion_world)),
                torch.ones(2, 1, dtype=torch.float32),
            ),
            dim=-1,
        )
        reset_pair = homogeneous @ camera.camera_from_world.transpose(0, 1)
        reset[target_slot] = reset_pair[0, :3]
        reset[companion] = reset_pair[1, :3]
        final = _make_state(row, camera, reset, before_birth=False)
        rendered = render_spheres(final, camera, DYNAMIC_SET_IMAGE_SIZE, noise_std=0.0)
        descriptors = _appearance_descriptors(rendered, final.active)
        resolved = _ranked_target_slot(descriptors, final.active, row.target_rank)
        camera_positions = reset
        if resolved == target_slot:
            return camera_positions, target_slot
        target_slot = resolved
    raise ValueError("observable target rank did not stabilize after contact layout")


def _preflight_render(
    state: SphereState,
    rendered: RenderOutput,
    camera: CameraFrame,
) -> None:
    active = state.active
    if not bool(rendered.projected_valid[active].all()):
        raise ValueError("every active planning-history object must be projectable")
    if not torch.equal(
        rendered.visible_fraction[active],
        torch.ones_like(rendered.visible_fraction[active]),
    ):
        raise ValueError("planning histories require zero pixel occlusion")
    if not torch.equal(rendered.visible_mask[active], rendered.full_mask[active]):
        raise ValueError("planning histories require complete silhouettes")
    if bool((rendered.visible_mask[active].sum(dim=(-2, -1)) == 0).any()):
        raise ValueError("every active planning object requires visible pixels")
    centres = rendered.projected_center_pixels
    radii = rendered.apparent_radius
    x_clearance = torch.minimum(centres[:, 0] - radii, 63.0 - centres[:, 0] - radii)
    y_clearance = torch.minimum(centres[:, 1] - radii, 63.0 - centres[:, 1] - radii)
    if bool((torch.minimum(x_clearance, y_clearance)[active] < 0.0).any()):
        raise ValueError("every planning-history silhouette must be image-contained")
    if not bool(torch.isfinite(rendered.rgb).all()) or not bool(
        torch.isfinite(rendered.depth_buffer).all()
    ):
        raise ValueError("planning RGB-D history must be finite")
    surface_fit = metric_sphere_centres_from_surface_depth(
        rendered.projected_center.unsqueeze(0),
        rendered.depth_buffer.unsqueeze(0).unsqueeze(0),
        DYNAMIC_SET_RADIUS_M,
        camera.world_from_camera.unsqueeze(0),
        camera.intrinsics.unsqueeze(0),
    )
    if not bool(surface_fit.valid_mask[0, active].all()):
        raise ValueError("every active planning object must support metric surface fitting")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    payload = hashlib.sha256()
    payload.update(str(tensor.dtype).encode("ascii"))
    payload.update(str(tuple(tensor.shape)).encode("ascii"))
    payload.update(tensor.numpy().tobytes(order="C"))
    return payload.hexdigest()


def _history_payload(
    frames: Sequence[PublicPlanningHistoryFrame],
    previously_dynamic: bool,
) -> Mapping[str, Any]:
    return {
        "schema": "dynamic_set_public_planning_history_v1",
        "previously_dynamic": previously_dynamic,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "rgb_sha256": _tensor_sha256(frame.rgb),
                "depth_sha256": _tensor_sha256(frame.depth),
                "world_from_camera_sha256": _tensor_sha256(frame.world_from_camera),
                "intrinsics_sha256": _tensor_sha256(frame.intrinsics),
            }
            for frame in frames
        ],
    }


def validate_public_planning_history(history: PublicPlanningHistory) -> PublicPlanningHistory:
    if not isinstance(history, PublicPlanningHistory):
        raise TypeError("history must be PublicPlanningHistory")
    if type(history.previously_dynamic) is not bool:
        raise TypeError("history previously_dynamic must be boolean")
    if not isinstance(history.frames, tuple) or len(history.frames) != PLANNING_HISTORY_FRAMES:
        raise ValueError(f"planning history must contain exactly {PLANNING_HISTORY_FRAMES} frames")
    first_camera: Tensor | None = None
    first_intrinsics: Tensor | None = None
    for expected_index, frame in enumerate(history.frames):
        if not isinstance(frame, PublicPlanningHistoryFrame):
            raise TypeError("planning history frames have an invalid type")
        if frame.frame_index != expected_index:
            raise ValueError("planning history frame order is not contiguous")
        if frame.timestamp != expected_index / DYNAMIC_SET_FRAME_RATE_HZ:
            raise ValueError("planning history timestamps must use the exact 20 Hz grid")
        for name, value, shape in (
            ("rgb", frame.rgb, (3, *DYNAMIC_SET_IMAGE_SIZE)),
            ("depth", frame.depth, (1, *DYNAMIC_SET_IMAGE_SIZE)),
            ("world_from_camera", frame.world_from_camera, (4, 4)),
            ("intrinsics", frame.intrinsics, (3, 3)),
        ):
            if not isinstance(value, Tensor) or value.shape != shape:
                raise ValueError(f"planning history {name} must have shape {shape}")
            if value.dtype is not torch.float32 or value.device.type != "cpu":
                raise ValueError(f"planning history {name} must be CPU float32")
            if value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(f"planning history {name} must be detached and finite")
        if first_camera is None:
            first_camera = frame.world_from_camera
            first_intrinsics = frame.intrinsics
        elif not torch.equal(frame.world_from_camera, first_camera) or not torch.equal(
            frame.intrinsics, first_intrinsics
        ):
            raise ValueError("planning history must use one fixed calibrated camera")
    expected = canonical_sha256(_history_payload(history.frames, history.previously_dynamic))
    if expected != history.history_sha256:
        raise ValueError("public planning history differs from its digest")
    return history


def _build_public_history(
    row: PlanningManifestRow,
    camera: CameraFrame,
    camera_positions: Tensor,
) -> tuple[PublicPlanningHistory, SphereState, RenderOutput, Tensor]:
    final_state = _make_state(row, camera, camera_positions, before_birth=False)
    final_render = render_spheres(final_state, camera, DYNAMIC_SET_IMAGE_SIZE, noise_std=0.0)
    _preflight_render(final_state, final_render, camera)
    before_state = _make_state(row, camera, camera_positions, before_birth=True)
    before_render = render_spheres(before_state, camera, DYNAMIC_SET_IMAGE_SIZE, noise_std=0.0)
    _preflight_render(before_state, before_render, camera)
    appearances = _appearance_descriptors(final_render, final_state.active)
    frames: list[PublicPlanningHistoryFrame] = []
    for frame_index in range(PLANNING_HISTORY_FRAMES):
        before_birth = row.previously_dynamic and frame_index < PLANNING_DYNAMIC_BIRTH_FRAME
        rendered = before_render if before_birth else final_render
        timestamp = frame_index / DYNAMIC_SET_FRAME_RATE_HZ
        frames.append(
            PublicPlanningHistoryFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                rgb=rendered.rgb.detach().clone(),
                depth=rendered.depth_buffer.detach().clone().unsqueeze(0),
                world_from_camera=camera.world_from_camera.detach().clone(),
                intrinsics=camera.intrinsics.detach().clone(),
            )
        )
    frozen_frames = tuple(frames)
    history = PublicPlanningHistory(
        frames=frozen_frames,
        previously_dynamic=row.previously_dynamic,
        history_sha256=canonical_sha256(_history_payload(frozen_frames, row.previously_dynamic)),
    )
    return validate_public_planning_history(history), final_state, final_render, appearances


def _reference_belief(
    final_state: SphereState,
    appearances: Tensor,
    camera: CameraFrame,
    *,
    timestamp: float,
) -> WorldBelief:
    factory = BeliefFactory(
        max_objects=DYNAMIC_SET_MAX_OBJECTS,
        appearance_dim=8,
        initial_radius=DYNAMIC_SET_RADIUS_M,
        initial_mass=DYNAMIC_SET_MASS,
        initial_restitution=DYNAMIC_SET_RESTITUTION,
        initial_drag=DYNAMIC_SET_DRAG,
        initial_friction=DYNAMIC_SET_FRICTION,
    )
    belief = factory.create(
        timestamp=timestamp,
        gravity=(0.0, 0.0, 0.0),
        intrinsics=camera.intrinsics.unsqueeze(0),
        world_from_camera=camera.world_from_camera.unsqueeze(0),
        active_modalities=("rgbd",),
    )
    objects = belief.objects.clone()
    objects.active.copy_(final_state.active.unsqueeze(0))
    objects.object_id.copy_(final_state.object_id.unsqueeze(0))
    objects.position.copy_(final_state.position.unsqueeze(0))
    objects.velocity.copy_(final_state.velocity.unsqueeze(0))
    objects.geometry[..., 0].copy_(final_state.radius[:, 0].unsqueeze(0))
    objects.appearance.copy_(appearances.unsqueeze(0))
    objects.log_mass.copy_(final_state.mass.log().unsqueeze(0))
    objects.log_drag.copy_(final_state.drag.log().unsqueeze(0))
    objects.restitution_logit.copy_(torch.logit(final_state.restitution).unsqueeze(0))
    objects.friction_logit.copy_(torch.logit(final_state.friction).unsqueeze(0))
    objects.motion_mode_logits.zero_()
    objects.motion_mode_logits[..., MotionMode.FREE] = 8.0
    objects.existence_logit.masked_fill_(objects.active, 8.0)
    objects.visibility_logit.masked_fill_(objects.active, 8.0)
    objects.age_steps.masked_fill_(objects.active, 16)
    belief = replace(
        belief,
        objects=objects,
        next_object_id=final_state.object_id[final_state.active].max().add(1).reshape(1),
        metadata={},
    )
    return belief.validate()


def _independent_oracle_state(reference: WorldBelief) -> SphereState:
    """Translate public reference truth into the independent simulator schema."""

    if reference.batch_size != 1 or reference.device.type != "cpu":
        raise ValueError("private planning oracle requires one CPU reference belief")
    objects = reference.objects
    count = objects.max_objects
    orientation = torch.zeros((count, 4), dtype=torch.float32)
    orientation[:, 3] = 1.0
    state = SphereState(
        object_id=objects.object_id[0].detach().clone(),
        active=objects.active[0].detach().clone(),
        position=objects.position[0].detach().clone(),
        velocity=objects.velocity[0].detach().clone(),
        radius=objects.radius[0].detach().clone(),
        mass=objects.mass[0].detach().clone(),
        restitution=objects.restitution[0].detach().clone(),
        drag=objects.drag[0].detach().clone(),
        friction=objects.friction[0].detach().clone(),
        albedo=torch.zeros((count, 3), dtype=torch.float32),
        orientation=orientation,
        angular_velocity=torch.zeros((count, 3), dtype=torch.float32),
        sleeping=torch.zeros(count, dtype=torch.bool),
        sleep_counter=torch.zeros(count, dtype=torch.int64),
    )
    state.validate()
    return state


def _independent_oracle_target_slot(
    template: PublicPlanningTemplate,
    reference: WorldBelief,
) -> int:
    active_slots = torch.nonzero(reference.objects.active[0], as_tuple=False).flatten()
    appearances = F.normalize(reference.objects.appearance[0, active_slots], dim=-1)
    handle = F.normalize(template.appearance_handle[0], dim=0)
    similarity = appearances @ handle
    order = torch.argsort(similarity, descending=True, stable=True)
    if (
        order.numel() == 0
        or float(similarity[order[0]]) < template.config.minimum_single_handle_cosine
    ):
        raise ValueError("independent oracle cannot resolve the observable target handle")
    if order.numel() > 1 and float(similarity[order[0]] - similarity[order[1]]) < (
        template.config.minimum_handle_cosine_margin
    ):
        raise ValueError("independent oracle target handle is ambiguous")
    return int(active_slots[order[0]])


def _oracle_configuration_payload() -> Mapping[str, Any]:
    """Frozen private analytic semantics used to certify every task ledger."""

    return {
        "schema": "independent_sphere_simulator_planning_oracle_v1",
        "physics_hz": PLANNING_ORACLE_PHYSICS_HZ,
        "integrator": "sphere_simulator_exact_drag_substeps_v1",
        "action_timing": "independent_prefix_then_endpoint_impulse_v1",
        "contact_evidence": "simulator_pair_contact_or_collision_v1",
        "cost": "independent_terminal_squared_error_plus_effort_v1",
        "deterministic_terminal_variance": 0.0,
        "world_bounds": ((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)),
        "solver_iterations": 2,
    }


def _independent_simulator_oracle(
    template: PublicPlanningTemplate,
    reference: WorldBelief,
) -> tuple[Tensor, Tensor, tuple[bool, ...]]:
    """Roll out all candidates without production dynamics or planner code.

    The data simulator is an independent implementation of sphere physics.
    This routine owns its own absolute-time split and cost calculation; it
    deliberately does not construct a ``DynamicsModel``, a production action,
    a belief trajectory, or a counterfactual-plan result.
    """

    if bool(reference.gravity.ne(0.0).any()):
        raise ValueError("planning oracle requires the frozen zero-gravity family")
    initial = _independent_oracle_state(reference)
    target_slot = _independent_oracle_target_slot(template, reference)
    source_timestamp = float(template.source_timestamp[0])
    action_offsets = template.candidate_timestamps[:, 0].to(torch.float64) - source_timestamp
    if not torch.equal(action_offsets, action_offsets[:1].expand_as(action_offsets)):
        raise ValueError("planning candidates do not share one frozen action timestamp")
    action_offset = float(action_offsets[0])
    horizon = float(template.query_offsets[0, -1])
    if not 0.0 < action_offset < horizon:
        raise ValueError("independent oracle action time must lie inside its horizon")
    quantum = 1.0 / PLANNING_ORACLE_PHYSICS_HZ
    action_steps = round(action_offset * PLANNING_ORACLE_PHYSICS_HZ)
    horizon_steps = round(horizon * PLANNING_ORACLE_PHYSICS_HZ)
    if (
        abs(action_offset - action_steps * quantum) > 2.0e-6
        or abs(horizon - horizon_steps * quantum) > 2.0e-6
    ):
        raise ValueError("planning timestamps do not align to the private oracle clock")
    config = PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)),
        max_substep=quantum,
        solver_iterations=2,
    )
    config.validate()
    pair_cache = prevalidated_sphere_pair_cache(initial)
    prefix, prefix_events = advance_spheres_no_boundary_contacts_prevalidated(
        initial,
        action_steps * quantum,
        config,
        external_impulse=torch.zeros_like(initial.velocity),
        pair_cache=pair_cache,
    )
    if bool(prefix_events.pair_contact.any() or prefix_events.pair_collision.any()):
        raise ValueError("planning oracle prefix contains contact before the candidate action")

    terminal_distances: list[Tensor] = []
    costs: list[Tensor] = []
    contacts: list[bool] = []
    post_duration = (horizon_steps - action_steps) * quantum
    weights = template.config.cost_weights
    for candidate_index in range(template.row.candidate_count):
        impulse = torch.zeros_like(prefix.velocity)
        impulse[target_slot] = template.candidate_impulses_world[candidate_index, 0]
        terminal, events = advance_spheres_no_boundary_contacts_prevalidated(
            prefix,
            post_duration,
            config,
            external_impulse=impulse,
            pair_cache=pair_cache,
        )
        displacement = terminal.position[target_slot].to(torch.float64) - (
            template.goal_position_world[0].to(torch.float64)
        )
        squared_error = displacement.square().sum()
        effort = impulse[target_slot].to(torch.float64).square().sum()
        # The simulator has no epistemic/process variance.  The explicit zero
        # term is kept in the independent formula so any future nonzero public
        # variance weight cannot silently invoke production uncertainty code.
        total = (
            float(weights.terminal_position) * squared_error
            + float(weights.terminal_variance) * squared_error.new_zeros(())
            + float(weights.impulse_effort) * effort
        )
        terminal_distances.append(squared_error.clamp_min(0.0).sqrt())
        costs.append(total)
        contacts.append(bool(events.pair_contact.any() or events.pair_collision.any()))
    return torch.stack(costs), torch.stack(terminal_distances), tuple(contacts)


def _oracle_for_template(
    template: PublicPlanningTemplate,
    reference: WorldBelief,
) -> PrivatePlanningOracleEvidence:
    with torch.inference_mode():
        costs, distances, contacts = _independent_simulator_oracle(template, reference)
    return certify_private_planning_oracle(
        template,
        candidate_costs=costs.detach().to(torch.float64),
        candidate_terminal_goal_distance_m=distances.detach().to(torch.float64),
        candidate_induced_contact=contacts,
    )


def _materialization_digest(
    history: PublicPlanningHistory,
    template: PublicPlanningTemplate,
    oracle: PrivatePlanningOracleEvidence,
    accepted_seed: int,
    attempt_count: int,
    rejection_reasons: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "schema": "dynamic_set_planning_materialization_v3",
            "row": asdict(template.row),
            "public_history_sha256": history.history_sha256,
            "public_template_sha256": template.template_sha256,
            "private_oracle_sha256": oracle.evidence_sha256,
            "private_oracle_configuration": _oracle_configuration_payload(),
            "accepted_seed": accepted_seed,
            "attempt_count": attempt_count,
            "rejection_reasons": list(rejection_reasons),
        }
    )


def _materialize_planning_task_core(
    row: PlanningManifestRow,
    *,
    maximum_attempts: int = DEFAULT_PLANNING_MATERIALIZATION_ATTEMPTS,
) -> PlanningTaskMaterialization:
    """Generate one already-authorized planning row deterministically."""

    if (
        isinstance(maximum_attempts, bool)
        or not isinstance(maximum_attempts, int)
        or maximum_attempts <= 0
    ):
        raise ValueError("maximum_attempts must be a positive integer")
    rejection_reasons: list[str] = []
    for attempt_index in range(maximum_attempts):
        accepted_seed = _candidate_seed(row.seed, attempt_index)
        try:
            camera = _camera(row)
            positions, expected_target_slot = _layout_for_row(row, camera, accepted_seed)
            history, final_state, _rendered, appearances = _build_public_history(
                row, camera, positions
            )
            reference = _reference_belief(
                final_state,
                appearances,
                camera,
                timestamp=history.frames[-1].timestamp,
            )
            handle = ranked_observable_appearance_handle(reference, row.target_rank)
            # This private assertion guards the materializer only.  Neither
            # simulator slot nor persistent ID is stored in public output.
            active_slots = torch.nonzero(reference.objects.active[0], as_tuple=False).flatten()
            similarities = F.cosine_similarity(
                reference.objects.appearance[0, active_slots],
                handle.expand_as(reference.objects.appearance[0, active_slots]),
                dim=-1,
            )
            resolved_slot = int(active_slots[int(similarities.argmax())])
            if resolved_slot != expected_target_slot:
                raise RuntimeError("rendered observable rank and layout target disagree")
            template = materialize_public_planning_template(row, reference, handle)
            oracle = _oracle_for_template(template, reference)
            if not oracle.certificate.winner_succeeds:
                raise ValueError("independent oracle winner does not reach the public goal")
            attempt_count = attempt_index + 1
            rejection_audit = tuple(rejection_reasons)
            digest = _materialization_digest(
                history,
                template,
                oracle,
                accepted_seed,
                attempt_count,
                rejection_audit,
            )
            return PlanningTaskMaterialization(
                public_history=history,
                template=template,
                private_oracle=oracle,
                accepted_seed=accepted_seed,
                attempt_count=attempt_count,
                rejection_reasons=rejection_audit,
                materialization_sha256=digest,
            )
        except (RuntimeError, ValueError) as error:
            rejection_reasons.append(f"{type(error).__name__}: {error}")
    last = rejection_reasons[-1] if rejection_reasons else "no candidate was evaluated"
    raise PlanningTaskMaterializationError(
        f"failed to materialize planning row {row.split}:{row.ordinal} after "
        f"{maximum_attempts} deterministic attempts; last rejection: {last}"
    )


def _materialize_planning_task(
    row: PlanningManifestRow,
    *,
    maximum_attempts: int = DEFAULT_PLANNING_MATERIALIZATION_ATTEMPTS,
) -> PlanningTaskMaterialization:
    """Materialize only a public row at the deepest unguarded entry."""

    _validate_row(row)
    if row.split not in _PUBLIC_MATERIALIZATION_SPLITS:
        raise PermissionError("protected planning rows require a claimed capability")
    return _materialize_planning_task_core(row, maximum_attempts=maximum_attempts)


def materialize_planning_task(
    row: PlanningManifestRow,
    *,
    maximum_attempts: int = DEFAULT_PLANNING_MATERIALIZATION_ATTEMPTS,
) -> PlanningTaskMaterialization:
    """Generate one public training/development planning task.

    The complete private oracle for a protected row is available only within
    an authorized evaluator after it has validated the one-shot split permit
    and exact frozen manifest.
    """

    _validate_row(row)
    if row.split not in _PUBLIC_MATERIALIZATION_SPLITS:
        raise PermissionError("protected planning rows require the governed evaluator")
    return _materialize_planning_task(row, maximum_attempts=maximum_attempts)


def _materialize_protected_planning_task(
    row: PlanningManifestRow,
    *,
    capability: _ProtectedPlanningMaterializationCapability,
    maximum_attempts: int = DEFAULT_PLANNING_MATERIALIZATION_ATTEMPTS,
) -> PlanningTaskMaterialization:
    """Generate one protected task after its ordered one-shot claim."""

    _validate_row(row)
    if not isinstance(capability, _ProtectedPlanningMaterializationCapability):
        raise PermissionError("protected planning materialization lacks evaluator authority")
    if row.split not in _PROTECTED_SPLITS:
        capability.close()
        raise PermissionError("protected planning capability cannot open a public row")
    capability.claim(row)
    return _materialize_planning_task_core(row, maximum_attempts=maximum_attempts)


def iter_planning_task_materializations(
    rows: Iterable[PlanningManifestRow],
) -> Iterator[PlanningTaskMaterialization]:
    """Lazily materialize public training/development rows only."""

    for row in rows:
        _validate_row(row)
        if row.split not in _PUBLIC_MATERIALIZATION_SPLITS:
            raise PermissionError("protected planning rows require the governed evaluator")
        yield materialize_planning_task(row)


def _validate_public_materialization(
    materialization: PlanningTaskMaterialization,
) -> PlanningTaskMaterialization:
    if not isinstance(materialization, PlanningTaskMaterialization):
        raise TypeError("materialization must be PlanningTaskMaterialization")
    history = validate_public_planning_history(materialization.public_history)
    template = validate_public_planning_template(materialization.template)
    if (
        isinstance(materialization.attempt_count, bool)
        or not isinstance(materialization.attempt_count, int)
        or materialization.attempt_count <= 0
    ):
        raise ValueError("planning materialization attempt_count is invalid")
    if (
        isinstance(materialization.accepted_seed, bool)
        or not isinstance(materialization.accepted_seed, int)
        or materialization.accepted_seed
        != _candidate_seed(template.row.seed, materialization.attempt_count - 1)
    ):
        raise ValueError("planning materialization accepted seed is not reproducible")
    if (
        not isinstance(materialization.rejection_reasons, tuple)
        or len(materialization.rejection_reasons) != materialization.attempt_count - 1
        or any(
            type(reason) is not str or not reason for reason in materialization.rejection_reasons
        )
    ):
        raise ValueError("planning materialization rejection audit is malformed")
    if history.previously_dynamic != template.row.previously_dynamic:
        raise ValueError("public planning history provenance differs from its template")
    expected_camera = _camera(template.row)
    if any(
        not torch.equal(frame.world_from_camera, expected_camera.world_from_camera)
        or not torch.equal(frame.intrinsics, expected_camera.intrinsics)
        for frame in history.frames
    ):
        raise ValueError("public planning history differs from its camera stratum")
    final_timestamp = torch.tensor([history.frames[-1].timestamp], dtype=torch.float32)
    if not torch.equal(final_timestamp, template.source_timestamp):
        raise ValueError("public history endpoint differs from the template timestamp")
    return materialization


def _validate_complete_materialization(
    materialization: PlanningTaskMaterialization,
) -> PlanningTaskMaterialization:
    _validate_public_materialization(materialization)
    expected = _materialization_digest(
        materialization.public_history,
        materialization.template,
        materialization.private_oracle,
        materialization.accepted_seed,
        materialization.attempt_count,
        materialization.rejection_reasons,
    )
    if expected != materialization.materialization_sha256:
        raise ValueError("planning materialization differs from its combined digest")
    return materialization


def _model_belief(model: Any) -> WorldBelief:
    belief = getattr(model, "belief", None)
    if not isinstance(belief, WorldBelief):
        raise PlanningTaskUnresolvedError("checkpoint did not emit a WorldBelief")
    return belief


def _history_evidence_from_runtime(
    model: Any,
    beliefs: Sequence[WorldBelief],
    template: PublicPlanningTemplate,
) -> PlanningHistoryEvidence:
    final = beliefs[-1]
    state = getattr(model, "state", None)
    histories = getattr(state, "temporal_histories", None)
    stream_key = runtime_stream_key("rgbd", PLANNING_HISTORY_SENSOR_ID)
    if not isinstance(histories, Mapping):
        raise PlanningTaskUnresolvedError("checkpoint exposes no runtime temporal histories")
    temporal = histories.get(stream_key)
    if not isinstance(temporal, RGBDTemporalPositionHistory):
        raise PlanningTaskUnresolvedError("checkpoint has no RGB-D metric-position history")
    try:
        temporal._validate_storage()
    except (TypeError, ValueError) as error:
        raise PlanningTaskUnresolvedError(
            f"invalid checkpoint temporal history: {error}"
        ) from error
    if temporal.object_ids.shape != final.objects.object_id.shape:
        raise PlanningTaskUnresolvedError("checkpoint history has the wrong padded object shape")
    expected_ids = torch.where(
        final.objects.active,
        final.objects.object_id,
        torch.full_like(final.objects.object_id, -1),
    )
    if not torch.equal(temporal.object_ids, expected_ids):
        raise PlanningTaskUnresolvedError(
            "checkpoint history is not aligned to final persistent IDs"
        )
    valid_count = temporal.valid_mask.sum(dim=-1, dtype=torch.int64)
    valid_count = torch.where(final.objects.active, valid_count, torch.zeros_like(valid_count))

    final_ids = final.objects.object_id[0, final.objects.active[0]].tolist()
    first_seen: list[int] = []
    for object_id in final_ids:
        observed = [
            index
            for index, belief in enumerate(beliefs)
            if bool(
                (belief.objects.active[0] & (belief.objects.object_id[0] == int(object_id))).any()
            )
        ]
        if not observed:
            raise PlanningTaskUnresolvedError("a final persistent ID has no observable history")
        first_seen.append(observed[0])
    if template.row.previously_dynamic:
        late = [index for index in first_seen if index >= PLANNING_DYNAMIC_BIRTH_FRAME]
        early = [index for index in first_seen if index < PLANNING_DYNAMIC_BIRTH_FRAME]
        if len(late) != 1 or any(index > 2 for index in early):
            raise PlanningTaskUnresolvedError(
                "checkpoint did not preserve one stable observed post-start birth"
            )
    elif any(index > 2 for index in first_seen):
        raise PlanningTaskUnresolvedError(
            "checkpoint introduced or switched a static-history target too late"
        )
    evidence = PlanningHistoryEvidence(
        valid_sample_count=valid_count,
        previously_dynamic=template.row.previously_dynamic,
    )
    try:
        evidence.validate(
            template.row,
            final,
            minimum_samples=template.config.minimum_mature_samples,
        )
    except (TypeError, ValueError) as error:
        raise PlanningTaskUnresolvedError(f"checkpoint history is not mature: {error}") from error
    return evidence


def infer_and_bind_public_planning_task(
    model: nn.Module,
    public_history: PublicPlanningHistory,
    template: PublicPlanningTemplate,
) -> PublicPlanningTask:
    """Infer a checkpoint state from frames only, then bind its real history."""

    validate_public_planning_history(public_history)
    validate_public_planning_template(template)
    if public_history.previously_dynamic != template.row.previously_dynamic:
        raise ValueError("public history and template lifecycle provenance differ")
    if not isinstance(model, nn.Module):
        raise TypeError("planning inference requires a torch module")
    reset = getattr(model, "reset", None)
    ingest = getattr(model, "ingest", None)
    if not callable(reset) or not callable(ingest):
        raise TypeError("planning inference model must expose reset and ingest")
    reset(batch_size=1)
    model.eval()
    beliefs: list[WorldBelief] = []
    try:
        # Prepared propagation validates tensor version counters so that a
        # stale or mutated belief cannot be replayed. ``inference_mode``
        # removes those counters and therefore made every real
        # ``OnlineWorldModel`` planning history fail before it could resolve a
        # target. Gradients are unnecessary here, but version tracking is a
        # correctness property, so use the lighter-weight no-grad context.
        with torch.no_grad():
            for frame in public_history.frames:
                belief = ingest(frame.packet())
                if not isinstance(belief, WorldBelief):
                    raise PlanningTaskUnresolvedError(
                        "checkpoint ingest did not return WorldBelief"
                    )
                beliefs.append(belief.detach().clone())
    except PlanningTaskUnresolvedError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        raise PlanningTaskUnresolvedError(f"checkpoint RGB-D inference failed: {error}") from error
    returned_final = beliefs[-1]
    final = (
        _model_belief(model)
        .detach()
        .clone()
        .replace(metadata={})
        .validate(log_variance_bounds=PLANNING_LOG_VARIANCE_BOUNDS)
    )
    if final.device.type != "cpu" or final.dtype is not torch.float32:
        raise PlanningTaskUnresolvedError("planning checkpoint must execute CPU float32")
    if (
        not torch.equal(final.timestamp, returned_final.timestamp)
        or not torch.equal(final.objects.active, returned_final.objects.active)
        or not torch.equal(final.objects.object_id, returned_final.objects.object_id)
    ):
        raise PlanningTaskUnresolvedError(
            "checkpoint persistent state differs from its final ingest result"
        )
    if int(final.objects.active.sum()) != template.row.object_count:
        raise PlanningTaskUnresolvedError(
            "checkpoint final cardinality differs from the planning manifest"
        )
    if not torch.equal(final.timestamp, template.source_timestamp.to(final.device)):
        raise PlanningTaskUnresolvedError("checkpoint endpoint timestamp differs from template")
    evidence = _history_evidence_from_runtime(model, beliefs, template)
    try:
        return bind_public_planning_task(template, final, evidence)
    except (RuntimeError, TypeError, ValueError) as error:
        raise PlanningTaskUnresolvedError(
            f"mature target handle did not resolve: {error}"
        ) from error


def _evaluate_one_materialization(
    model: nn.Module,
    materialization: PlanningTaskMaterialization,
) -> tuple[PlanningTaskOutcome, PublicPlanningTask | None]:
    _validate_public_materialization(materialization)
    try:
        task = infer_and_bind_public_planning_task(
            model,
            materialization.public_history,
            materialization.template,
        )
    except PlanningTaskUnresolvedError as error:
        # Do not touch private oracle evidence when observable inference fails.
        return PlanningTaskOutcome.unresolved(materialization.row, reason=str(error)), None
    dynamics = getattr(model, "dynamics", None)
    if dynamics is None:
        raise TypeError("planning evaluation model must expose dynamics")
    with torch.no_grad():
        evaluation = evaluate_certified_planning_task(
            dynamics,
            task,
            materialization.private_oracle,
        )
    _validate_complete_materialization(materialization)
    return PlanningTaskOutcome.evaluated(evaluation), task


def _infer_one_paired_public_task(
    model: nn.Module,
    materialization: PlanningTaskMaterialization,
) -> tuple[PlanningTaskOutcome | None, PublicPlanningTask | None]:
    """Return an observable failure or one oracle-free checkpoint binding."""

    try:
        task = infer_and_bind_public_planning_task(
            model,
            materialization.public_history,
            materialization.template,
        )
    except PlanningTaskUnresolvedError as error:
        return PlanningTaskOutcome.unresolved(materialization.row, reason=str(error)), None
    return None, task


def _evaluate_one_paired_materialization(
    candidate_model: nn.Module,
    reference_model: nn.Module,
    materialization: PlanningTaskMaterialization,
) -> tuple[
    PlanningTaskOutcome,
    PlanningTaskOutcome,
    PublicPlanningTask | None,
    PublicPlanningTask | None,
    PlanningPairProvenance,
]:
    """Evaluate one immutable public row before opening its shared oracle."""

    _validate_public_materialization(materialization)
    public_evidence = (materialization.public_history, materialization.template)
    public_signature = tensor_identity_version_signature(public_evidence)

    candidate_failure, candidate_task = _infer_one_paired_public_task(
        candidate_model,
        materialization,
    )
    if tensor_identity_version_signature(public_evidence) != public_signature:
        raise PlanningTaskMaterializationError("candidate mutated paired public planning evidence")

    reference_failure, reference_task = _infer_one_paired_public_task(
        reference_model,
        materialization,
    )
    if tensor_identity_version_signature(public_evidence) != public_signature:
        raise PlanningTaskMaterializationError("reference mutated paired public planning evidence")

    # Both observable inference passes are now complete.  Authenticate and
    # bind one opaque complete-materialization/oracle commitment even when
    # neither checkpoint resolved the handle, so durable paired evidence
    # cannot silently substitute a different oracle later.
    provenance = PlanningPairProvenance.create(materialization)
    if candidate_task is None and reference_task is None:
        if candidate_failure is None or reference_failure is None:
            raise RuntimeError("paired planning resolution state is inconsistent")
        return candidate_failure, reference_failure, None, None, provenance

    candidate_evaluation = None
    reference_evaluation = None
    with torch.no_grad():
        if candidate_task is not None and reference_task is not None:
            candidate_dynamics = getattr(candidate_model, "dynamics", None)
            reference_dynamics = getattr(reference_model, "dynamics", None)
            if candidate_dynamics is None or reference_dynamics is None:
                raise TypeError("paired planning models must expose dynamics")
            candidate_evaluation, reference_evaluation = evaluate_certified_planning_task_pair(
                candidate_dynamics,
                candidate_task,
                reference_dynamics,
                reference_task,
                materialization.private_oracle,
            )
        elif candidate_task is not None:
            candidate_dynamics = getattr(candidate_model, "dynamics", None)
            if candidate_dynamics is None:
                raise TypeError("candidate planning model must expose dynamics")
            candidate_evaluation = evaluate_certified_planning_task(
                candidate_dynamics,
                candidate_task,
                materialization.private_oracle,
            )
        elif reference_task is not None:
            reference_dynamics = getattr(reference_model, "dynamics", None)
            if reference_dynamics is None:
                raise TypeError("reference planning model must expose dynamics")
            reference_evaluation = evaluate_certified_planning_task(
                reference_dynamics,
                reference_task,
                materialization.private_oracle,
            )

    # The oracle is now authenticated independently by the scorer; bind it
    # back to the exact materialization only after all possible public passes.
    _validate_complete_materialization(materialization)
    if tensor_identity_version_signature(public_evidence) != public_signature:
        raise PlanningTaskMaterializationError("paired planning evaluation mutated public evidence")

    candidate_outcome = (
        PlanningTaskOutcome.evaluated(candidate_evaluation)
        if candidate_evaluation is not None
        else candidate_failure
    )
    reference_outcome = (
        PlanningTaskOutcome.evaluated(reference_evaluation)
        if reference_evaluation is not None
        else reference_failure
    )
    if candidate_outcome is None or reference_outcome is None:
        raise RuntimeError("paired planning evaluation produced an incomplete outcome")
    return (
        candidate_outcome,
        reference_outcome,
        candidate_task,
        reference_task,
        provenance,
    )


def evaluate_planning_task_materialization(
    model: nn.Module,
    materialization: PlanningTaskMaterialization,
) -> PlanningTaskOutcome:
    """Evaluate one task, returning handle/maturity failure as supported data."""

    outcome, _task = _evaluate_one_materialization(model, materialization)
    return outcome


def planning_error(outcomes: Sequence[PlanningTaskOutcome]) -> float:
    """Return the frozen nonnegative planning component for checkpoint selection."""

    if not outcomes:
        raise ValueError("planning_error requires at least one outcome")
    if not math.isclose(sum(PLANNING_ERROR_WEIGHTS.values()), 1.0, abs_tol=1.0e-12):
        raise RuntimeError("planning error weights must sum to one")
    errors: list[float] = []
    for outcome in outcomes:
        if not isinstance(outcome, PlanningTaskOutcome):
            raise TypeError("planning_error requires PlanningTaskOutcome values")
        if outcome.evaluation is None:
            if outcome.handle_resolved or not outcome.failure_reason:
                raise ValueError("unresolved planning outcome is malformed")
            components = {name: 1.0 for name in PLANNING_ERROR_WEIGHTS}
        else:
            evaluation = outcome.evaluation
            regret = min(max(float(evaluation.normalized_regret), 0.0), 1.0)
            components = {
                "oracle_winner_error": float(not evaluation.oracle_winner_correct),
                "normalized_regret": regret,
                "successful_oracle_task_error": float(
                    evaluation.oracle_winner_succeeds
                    and not evaluation.selected_action_goal_success
                ),
            }
        errors.append(sum(PLANNING_ERROR_WEIGHTS[name] * components[name] for name in components))
    result = sum(errors) / len(errors)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError("planning_error must remain finite and nonnegative")
    return result


def _failed_invariants() -> PlanningInvariantMetrics:
    return PlanningInvariantMetrics(
        serial_vectorized_winner_parity=False,
        maximum_cost_difference=math.inf,
        pre_action_invariance=False,
        exactly_once_impulse=False,
        action_target_isolation=False,
        conservation=False,
        batch_independence=False,
        source_belief_unchanged=False,
        latency_k8_seconds=math.inf,
        latency_k32_seconds=math.inf,
    )


def _planning_population_result_payload(
    outcomes: Sequence[PlanningTaskOutcome],
    reduction: PlanningEvaluationReduction,
    invariants: PlanningInvariantMetrics,
    error: float,
    binding: PlanningPopulationBinding,
) -> Mapping[str, Any]:
    return {
        "schema": "dynamic_set_planning_evaluation_v2",
        "population_binding": asdict(binding),
        "planning_error": error,
        "outcomes": [
            {
                "row": asdict(outcome.row),
                "handle_resolved": outcome.handle_resolved,
                "failure_reason": outcome.failure_reason,
                "evaluation": (None if outcome.evaluation is None else asdict(outcome.evaluation)),
            }
            for outcome in outcomes
        ],
        "reduction": asdict(reduction),
        "invariants": {
            name: (
                value if type(value) is not float or math.isfinite(value) else "nonfinite_failure"
            )
            for name, value in asdict(invariants).items()
        },
    }


def validate_planning_population_evaluation_result(
    result: PlanningPopulationEvaluationResult,
) -> PlanningPopulationEvaluationResult:
    """Authenticate a serialized population result against all reported evidence."""

    if not isinstance(result, PlanningPopulationEvaluationResult):
        raise TypeError("result must be PlanningPopulationEvaluationResult")
    if not isinstance(result.outcomes, tuple) or not result.outcomes:
        raise ValueError("planning population result requires a nonempty outcome tuple")
    if not isinstance(result.reduction, PlanningEvaluationReduction):
        raise TypeError("planning population result reduction has an invalid type")
    if not isinstance(result.invariants, PlanningInvariantMetrics):
        raise TypeError("planning population result invariants have an invalid type")
    if not isinstance(result.population_binding, PlanningPopulationBinding):
        raise TypeError("planning population result binding has an invalid type")
    if result.population_binding.row_count != len(result.outcomes):
        raise ValueError("planning population result row count differs from its outcomes")
    if any(outcome.row.split != result.population_binding.split for outcome in result.outcomes):
        raise ValueError("planning population result contains a differently bound split")
    expected_reduction = reduce_planning_task_outcomes(result.outcomes)
    if result.reduction != expected_reduction:
        raise ValueError("planning population reduction differs from its raw outcomes")
    expected_error = planning_error(result.outcomes)
    if result.planning_error != expected_error:
        raise ValueError("planning population result error differs from its outcomes")
    validated_sha256(result.result_sha256, label="planning population result digest")
    expected = canonical_sha256(
        _planning_population_result_payload(
            result.outcomes,
            result.reduction,
            result.invariants,
            result.planning_error,
            result.population_binding,
        )
    )
    if expected != result.result_sha256:
        raise ValueError("planning population result differs from its evidence digest")
    return result


def _paired_population_result_payload(
    result: PlanningPairedPopulationEvaluationResult,
) -> Mapping[str, Any]:
    return {
        "schema": "dynamic_set_paired_planning_evaluation_v1",
        "candidate_result_sha256": result.candidate.result_sha256,
        "reference_result_sha256": result.reference.result_sha256,
        "population_binding": asdict(result.candidate.population_binding),
        "provenance": [asdict(item) for item in result.provenance],
    }


def validate_paired_planning_population_evaluation_result(
    result: PlanningPairedPopulationEvaluationResult,
) -> PlanningPairedPopulationEvaluationResult:
    """Authenticate a paired population and its exact shared-row provenance."""

    if not isinstance(result, PlanningPairedPopulationEvaluationResult):
        raise TypeError("result must be PlanningPairedPopulationEvaluationResult")
    return result.validate()


def _invariant_features(row: PlanningManifestRow) -> frozenset[tuple[str, int]]:
    return frozenset(
        {
            ("object_count", row.object_count),
            ("candidate_count", row.candidate_count),
            ("previously_dynamic", int(row.previously_dynamic)),
            ("candidate_induced_contact", int(row.candidate_induced_contact)),
            ("target_rank", row.target_rank),
            ("action_time_stratum", row.action_time_stratum),
            ("camera_stratum", row.camera_stratum),
            ("goal_direction", row.goal_direction),
            ("distribution", int(row.distribution == "compositional_ood")),
        }
    )


def _stratified_invariant_cover(
    tasks: Sequence[PublicPlanningTask],
) -> tuple[PublicPlanningTask, ...]:
    """Greedily freeze a small deterministic cover of every public stratum."""

    if not tasks:
        return ()

    ordered = sorted(tasks, key=lambda task: (task.row.ordinal, task.row.seed))
    uncovered = set().union(*(_invariant_features(task.row) for task in ordered))
    selected: list[PublicPlanningTask] = []
    remaining = list(ordered)
    while uncovered:
        best = max(
            remaining,
            key=lambda task: (
                len(_invariant_features(task.row) & uncovered),
                -task.row.ordinal,
                -task.row.seed,
            ),
        )
        gain = _invariant_features(best.row) & uncovered
        if not gain:
            raise RuntimeError("planning invariant cover stopped before covering its strata")
        selected.append(best)
        uncovered.difference_update(gain)
        remaining.remove(best)
    return tuple(selected)


def _state_only_latency_population(
    tasks: Sequence[PublicPlanningTask],
    *,
    candidate_count: int,
) -> tuple[PublicPlanningTask, ...]:
    """Keep every B1/N=6 row in one candidate-count latency population."""

    if candidate_count not in {8, 32}:
        raise ValueError("planning latency candidate_count must be eight or 32")
    return tuple(
        task
        for task in tasks
        if task.row.object_count == DYNAMIC_SET_MAX_OBJECTS
        and task.row.candidate_count == candidate_count
    )


def _finish_population_result(
    model: nn.Module,
    outcomes: Sequence[PlanningTaskOutcome],
    tasks: Sequence[PublicPlanningTask],
    *,
    expected_split: str,
    binding: PlanningPopulationBinding,
    config: PlanningPopulationEvaluationConfig,
) -> PlanningPopulationEvaluationResult:
    """Reduce one already-sealed public planning population."""

    if not outcomes:
        raise ValueError("planning evaluation requires at least one task")
    reduction = reduce_planning_task_outcomes(outcomes, cost_tolerance=config.cost_tolerance)
    distribution = (
        "compositional_ood" if expected_split == "compositional_ood" else "in_distribution"
    )
    expected_slices = {
        (object_count, candidate_count, distribution)
        for object_count in range(1, DYNAMIC_SET_MAX_OBJECTS + 1)
        for candidate_count in (8, 32)
    }
    actual_slices = {
        (item.object_count, item.candidate_count, item.distribution) for item in reduction.slices
    }
    if config.require_complete_slices and actual_slices != expected_slices:
        raise ValueError("planning population lacks an N/K/distribution slice")

    invariants = _failed_invariants()
    if config.evaluate_invariants:
        latency_k8_tasks = _state_only_latency_population(tasks, candidate_count=8)
        latency_k32_tasks = _state_only_latency_population(tasks, candidate_count=32)
        required_features = set().union(*(_invariant_features(outcome.row) for outcome in outcomes))
        resolved_features = (
            set().union(*(_invariant_features(task.row) for task in tasks)) if tasks else set()
        )
        if (
            tasks
            and required_features <= resolved_features
            and latency_k8_tasks
            and latency_k32_tasks
        ):
            cover = _stratified_invariant_cover(tasks)
            peer = cover[1] if len(cover) > 1 else cover[0]
            dynamics = getattr(model, "dynamics", None)
            if dynamics is None:
                raise TypeError("planning evaluation model must expose dynamics")
            with torch.no_grad():
                measured = evaluate_required_planning_invariants(
                    dynamics,
                    cover[0],
                    peer,
                    latency_k8_tasks[0],
                    latency_k32_tasks[0],
                    additional_invariant_tasks=cover[2:],
                    additional_latency_k8_tasks=latency_k8_tasks[1:],
                    additional_latency_k32_tasks=latency_k32_tasks[1:],
                    numerical_tolerance=config.cost_tolerance,
                    latency_warmup_runs=config.latency_warmup_runs,
                    latency_measured_runs=config.latency_measured_runs,
                )
            invariants = replace(
                measured,
                serial_vectorized_winner_parity=(
                    measured.serial_vectorized_winner_parity
                    and reduction.serial_vectorized_winner_parity
                ),
                maximum_cost_difference=max(
                    measured.maximum_cost_difference,
                    reduction.maximum_cost_difference,
                ),
                action_target_isolation=(
                    measured.action_target_isolation and reduction.active_set_frozen
                ),
                source_belief_unchanged=(
                    measured.source_belief_unchanged and reduction.source_belief_unchanged
                ),
            )
    error = planning_error(outcomes)
    resolved_binding = replace(binding, row_count=len(outcomes))
    result = PlanningPopulationEvaluationResult(
        outcomes=tuple(outcomes),
        reduction=reduction,
        invariants=invariants,
        planning_error=error,
        population_binding=resolved_binding,
        result_sha256=canonical_sha256(
            _planning_population_result_payload(
                outcomes,
                reduction,
                invariants,
                error,
                resolved_binding,
            )
        ),
    )
    return validate_planning_population_evaluation_result(result)


def _population_result(
    model: nn.Module,
    materializations: Iterable[PlanningTaskMaterialization],
    *,
    expected_split: str,
    expected_rows: Sequence[PlanningManifestRow] | None,
    binding: PlanningPopulationBinding,
    config: PlanningPopulationEvaluationConfig,
) -> PlanningPopulationEvaluationResult:
    config.validate()
    outcomes: list[PlanningTaskOutcome] = []
    tasks: list[PublicPlanningTask] = []
    rows = tuple(expected_rows) if expected_rows is not None else None
    for index, materialization in enumerate(materializations):
        if rows is not None and index >= len(rows):
            raise ValueError("planning materialization stream contains extra rows")
        _validate_public_materialization(materialization)
        row = materialization.row
        if row.split != expected_split:
            raise ValueError("planning materialization belongs to a different split")
        if rows is not None and row != rows[index]:
            raise ValueError("planning materialization stream differs from frozen row order")
        outcome, task = _evaluate_one_materialization(model, materialization)
        outcomes.append(outcome)
        if task is not None:
            tasks.append(task)
    if not outcomes:
        raise ValueError("planning evaluation requires at least one task")
    if rows is not None and len(outcomes) != len(rows):
        raise ValueError("planning materialization stream ended before the frozen manifest")
    return _finish_population_result(
        model,
        outcomes,
        tasks,
        expected_split=expected_split,
        binding=binding,
        config=config,
    )


def _paired_population_result(
    candidate_model: nn.Module,
    reference_model: nn.Module,
    materializations: Iterable[PlanningTaskMaterialization],
    *,
    expected_split: str,
    expected_rows: Sequence[PlanningManifestRow] | None,
    binding: PlanningPopulationBinding,
    config: PlanningPopulationEvaluationConfig,
) -> PlanningPairedPopulationEvaluationResult:
    """Consume one task stream into aligned candidate/reference evidence."""

    config.validate()
    for model in (candidate_model, reference_model):
        if not isinstance(model, nn.Module):
            raise TypeError("paired planning models must be torch modules")

    candidate_outcomes: list[PlanningTaskOutcome] = []
    reference_outcomes: list[PlanningTaskOutcome] = []
    candidate_tasks: list[PublicPlanningTask] = []
    reference_tasks: list[PublicPlanningTask] = []
    provenance: list[PlanningPairProvenance] = []
    rows = tuple(expected_rows) if expected_rows is not None else None

    for index, materialization in enumerate(materializations):
        if rows is not None and index >= len(rows):
            raise ValueError("planning materialization stream contains extra rows")
        _validate_public_materialization(materialization)
        row = materialization.row
        if row.split != expected_split:
            raise ValueError("planning materialization belongs to a different split")
        if rows is not None and row != rows[index]:
            raise ValueError("planning materialization stream differs from frozen row order")
        (
            candidate_outcome,
            reference_outcome,
            candidate_task,
            reference_task,
            row_provenance,
        ) = _evaluate_one_paired_materialization(
            candidate_model,
            reference_model,
            materialization,
        )
        candidate_outcomes.append(candidate_outcome)
        reference_outcomes.append(reference_outcome)
        provenance.append(row_provenance)
        if candidate_task is not None:
            candidate_tasks.append(candidate_task)
        if reference_task is not None:
            reference_tasks.append(reference_task)

    if not candidate_outcomes:
        raise ValueError("planning evaluation requires at least one task")
    if rows is not None and len(candidate_outcomes) != len(rows):
        raise ValueError("planning materialization stream ended before the frozen manifest")

    candidate = _finish_population_result(
        candidate_model,
        candidate_outcomes,
        candidate_tasks,
        expected_split=expected_split,
        binding=binding,
        config=config,
    )
    reference = _finish_population_result(
        reference_model,
        reference_outcomes,
        reference_tasks,
        expected_split=expected_split,
        binding=binding,
        config=config,
    )
    draft = PlanningPairedPopulationEvaluationResult(
        candidate=candidate,
        reference=reference,
        provenance=tuple(provenance),
        result_sha256="0" * 64,
    )
    result = replace(
        draft,
        result_sha256=canonical_sha256(_paired_population_result_payload(draft)),
    )
    return validate_paired_planning_population_evaluation_result(result)


def evaluate_development_planning_materializations(
    model: nn.Module,
    materializations: Iterable[PlanningTaskMaterialization],
    *,
    config: PlanningPopulationEvaluationConfig = DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG,
) -> PlanningPopulationEvaluationResult:
    """Stream public development tasks; every protected row fails closed."""

    def development_only() -> Iterator[PlanningTaskMaterialization]:
        for materialization in materializations:
            if not isinstance(materialization, PlanningTaskMaterialization):
                raise TypeError("planning stream values must be materializations")
            if materialization.row.split in _PROTECTED_SPLITS:
                raise PermissionError("protected planning rows require the governed opener")
            if materialization.row.split != "development":
                raise ValueError("development planning evaluation accepts only development rows")
            yield materialization

    return _population_result(
        model,
        development_only(),
        expected_split="development",
        expected_rows=None,
        binding=PlanningPopulationBinding(
            split="development",
            protocol_sha256=None,
            manifest_sha256=None,
            permit_index=None,
            row_count=0,
        ),
        config=config,
    )


def evaluate_paired_development_planning_materializations(
    candidate_model: nn.Module,
    reference_model: nn.Module,
    materializations: Iterable[PlanningTaskMaterialization],
    *,
    config: PlanningPopulationEvaluationConfig = DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG,
) -> PlanningPairedPopulationEvaluationResult:
    """Evaluate two checkpoints from each public development task exactly once."""

    def development_only() -> Iterator[PlanningTaskMaterialization]:
        for materialization in materializations:
            if not isinstance(materialization, PlanningTaskMaterialization):
                raise TypeError("planning stream values must be materializations")
            if materialization.row.split in _PROTECTED_SPLITS:
                raise PermissionError("protected planning rows require the governed opener")
            if materialization.row.split != "development":
                raise ValueError("development planning evaluation accepts only development rows")
            yield materialization

    return _paired_population_result(
        candidate_model,
        reference_model,
        development_only(),
        expected_split="development",
        expected_rows=None,
        binding=PlanningPopulationBinding(
            split="development",
            protocol_sha256=None,
            manifest_sha256=None,
            permit_index=None,
            row_count=0,
        ),
        config=config,
    )


def _validated_protected_planning_population(
    *,
    permit: SplitPermit,
    expected_protocol_sha256: str,
    expected_manifest_sha256: str,
    expected_rows: Sequence[PlanningManifestRow],
    config: PlanningPopulationEvaluationConfig,
) -> tuple[str, str, tuple[PlanningManifestRow, ...]]:
    """Validate all public authorization facts before oracle generation."""

    if type(permit) is not SplitPermit:
        raise TypeError("protected planning evaluation requires an exact SplitPermit")
    if config != DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG:
        raise ValueError(
            "protected planning evaluation requires the frozen complete-slice, "
            "invariant, tolerance, and latency configuration"
        )
    protocol_sha256 = validated_sha256(expected_protocol_sha256, label="expected_protocol_sha256")
    manifest_sha256 = validated_sha256(expected_manifest_sha256, label="expected_manifest_sha256")
    if permit.split not in _PROTECTED_SPLITS:
        raise PermissionError("split permit is not for a protected planning split")
    if permit.protocol_sha256 != protocol_sha256:
        raise PermissionError("split permit protocol binding differs")
    if permit.index != _PROTECTED_SPLIT_INDICES[permit.split]:
        raise PermissionError("split permit ordered index differs")
    validated_sha256(permit.nonce, label="split permit nonce")
    rows = tuple(expected_rows)
    if len(rows) != PLANNING_SPLIT_SIZES[permit.split]:
        raise ValueError("protected planning rows differ from the frozen split count")
    if any(not isinstance(row, PlanningManifestRow) or row.split != permit.split for row in rows):
        raise ValueError("protected expected rows differ from the permitted split")
    computed = canonical_sha256([asdict(row) for row in rows])
    if computed != manifest_sha256:
        raise ValueError("protected expected-row digest differs from its manifest binding")
    if FROZEN_PLANNING_MANIFEST_SHA256[permit.split] != manifest_sha256:
        raise ValueError("protected planning manifest differs from the source freeze")
    return protocol_sha256, manifest_sha256, rows


def protected_planning_population_claim_binding(
    *,
    split: str,
    protocol_sha256: str,
    manifest_sha256: str,
    row_count: int,
) -> str:
    """Bind a durable ledger claim to one exact planning population."""

    return canonical_sha256(
        {
            "schema": PLANNING_POPULATION_CLAIM_PURPOSE,
            "split": split,
            "protocol_sha256": validated_sha256(protocol_sha256, label="planning claim protocol"),
            "manifest_sha256": validated_sha256(manifest_sha256, label="planning claim manifest"),
            "row_count": row_count,
        }
    )


def _claim_protected_planning_population(
    ledger: OrderedSplitLedger,
    permit: SplitPermit,
    *,
    protocol_sha256: str,
    manifest_sha256: str,
    row_count: int,
) -> None:
    if type(ledger) is not OrderedSplitLedger:
        raise TypeError("protected planning evaluation requires its exact durable ledger")
    ledger.claim_active(
        permit,
        purpose=PLANNING_POPULATION_CLAIM_PURPOSE,
        binding_sha256=protected_planning_population_claim_binding(
            split=permit.split,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
            row_count=row_count,
        ),
    )


def _protected_planning_materializations(
    rows: tuple[PlanningManifestRow, ...],
    *,
    permit: SplitPermit,
    protocol_sha256: str,
    manifest_sha256: str,
) -> Iterator[PlanningTaskMaterialization]:
    """Own and close the only capability able to generate this population."""

    capability = _mint_protected_planning_materialization_capability(
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
            yield _materialize_protected_planning_task(row, capability=capability)
        capability.finish()
        finished = True
    finally:
        if not finished:
            capability.close()


def evaluate_authorized_planning_rows(
    model: nn.Module,
    *,
    ledger: OrderedSplitLedger,
    permit: SplitPermit,
    expected_protocol_sha256: str,
    expected_manifest_sha256: str,
    expected_rows: Sequence[PlanningManifestRow],
    config: PlanningPopulationEvaluationConfig = DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG,
) -> PlanningPopulationEvaluationResult:
    """Materialize and evaluate one exact protected planning population."""

    protocol_sha256, manifest_sha256, rows = _validated_protected_planning_population(
        permit=permit,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_rows=expected_rows,
        config=config,
    )
    _claim_protected_planning_population(
        ledger,
        permit,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        row_count=len(rows),
    )
    return _population_result(
        model,
        _protected_planning_materializations(
            rows,
            permit=permit,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
        ),
        expected_split=permit.split,
        expected_rows=rows,
        binding=PlanningPopulationBinding(
            split=permit.split,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
            permit_index=permit.index,
            row_count=0,
        ),
        config=config,
    )


def evaluate_authorized_paired_planning_rows(
    candidate_model: nn.Module,
    reference_model: nn.Module,
    *,
    ledger: OrderedSplitLedger,
    permit: SplitPermit,
    expected_protocol_sha256: str,
    expected_manifest_sha256: str,
    expected_rows: Sequence[PlanningManifestRow],
    config: PlanningPopulationEvaluationConfig = DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG,
) -> PlanningPairedPopulationEvaluationResult:
    """Materialize each protected task once for both checkpoint evaluations."""

    protocol_sha256, manifest_sha256, rows = _validated_protected_planning_population(
        permit=permit,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_rows=expected_rows,
        config=config,
    )
    _claim_protected_planning_population(
        ledger,
        permit,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        row_count=len(rows),
    )
    return _paired_population_result(
        candidate_model,
        reference_model,
        _protected_planning_materializations(
            rows,
            permit=permit,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
        ),
        expected_split=permit.split,
        expected_rows=rows,
        binding=PlanningPopulationBinding(
            split=permit.split,
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_sha256,
            permit_index=permit.index,
            row_count=0,
        ),
        config=config,
    )


def evaluate_authorized_planning_materializations(
    model: nn.Module,
    materializations: Iterable[PlanningTaskMaterialization],
    **kwargs: Any,
) -> PlanningPopulationEvaluationResult:
    """Reject the obsolete caller-owned protected materialization boundary."""

    del model, materializations, kwargs
    raise PermissionError(
        "caller-supplied protected planning materializations are prohibited; "
        "use evaluate_authorized_planning_rows"
    )


def evaluate_authorized_paired_planning_materializations(
    candidate_model: nn.Module,
    reference_model: nn.Module,
    materializations: Iterable[PlanningTaskMaterialization],
    **kwargs: Any,
) -> PlanningPairedPopulationEvaluationResult:
    """Reject the obsolete caller-owned paired protected boundary."""

    del candidate_model, reference_model, materializations, kwargs
    raise PermissionError(
        "caller-supplied protected planning materializations are prohibited; "
        "use evaluate_authorized_paired_planning_rows"
    )


__all__ = [
    "DEFAULT_PLANNING_MATERIALIZATION_ATTEMPTS",
    "DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG",
    "PLANNING_DYNAMIC_BIRTH_FRAME",
    "PLANNING_ERROR_WEIGHTS",
    "PLANNING_HISTORY_FRAMES",
    "PLANNING_HISTORY_SENSOR_ID",
    "PlanningPopulationBinding",
    "PlanningPopulationEvaluationConfig",
    "PlanningPopulationEvaluationResult",
    "PlanningPairProvenance",
    "PlanningPairedPopulationEvaluationResult",
    "PlanningTaskMaterialization",
    "PlanningTaskMaterializationError",
    "PlanningTaskUnresolvedError",
    "PublicPlanningHistory",
    "PublicPlanningHistoryFrame",
    "evaluate_authorized_paired_planning_rows",
    "evaluate_authorized_planning_rows",
    "evaluate_development_planning_materializations",
    "evaluate_paired_development_planning_materializations",
    "evaluate_planning_task_materialization",
    "infer_and_bind_public_planning_task",
    "iter_planning_task_materializations",
    "materialize_planning_task",
    "planning_error",
    "validate_paired_planning_population_evaluation_result",
    "validate_planning_population_evaluation_result",
    "validate_public_planning_history",
]
