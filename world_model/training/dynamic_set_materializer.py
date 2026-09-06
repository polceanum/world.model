"""Deterministic public RGB-D episode materialization for specification 1.61.

The frozen manifest chooses the experimental cell.  This module realizes that
choice with the public sphere solver and renderer, then accepts the episode
only through :func:`preflight_dynamic_set_episode`.  Candidate seeds advance
on a separate deterministic stride, so a rejected visual/physical layout is
reproducible and cannot perturb the manifest or any other row.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.belief import WorldBelief
from world_model.dynamics import WorldImpulseAction
from world_model.simulator.camera import (
    CameraFrame,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
)
from world_model.simulator.collisions import BOUNDARY_NAMES
from world_model.simulator.episode import Episode, validate_episode
from world_model.simulator.labels import make_perception_labels, validate_perception_labels
from world_model.simulator.physics import (
    PhysicsConfig,
    PhysicsStepEvents,
    SphereState,
    empty_physics_events,
)
from world_model.training.dynamic_set_physics import (
    advance_spheres_no_boundary_contacts_prevalidated,
    prevalidated_sphere_pair_cache,
)
from world_model.training.dynamic_set_planning import resolve_planning_target_handle
from world_model.training.dynamic_set_protocol import (
    PHYSICAL_CELLS,
    SIMULATOR_VERSION,
    PhysicalManifestRow,
)
from world_model.training.dynamic_set_rendering import (
    prevalidated_sphere_render_cache,
    render_spheres,
)
from world_model.training.dynamic_set_scene import (
    DYNAMIC_SET_DRAG,
    DYNAMIC_SET_FRAME_RATE_HZ,
    DYNAMIC_SET_FRAMES,
    DYNAMIC_SET_FRICTION,
    DYNAMIC_SET_IMAGE_SIZE,
    DYNAMIC_SET_MASS,
    DYNAMIC_SET_MAX_OBJECTS,
    DYNAMIC_SET_RADIUS_M,
    DYNAMIC_SET_RESTITUTION,
    DYNAMIC_SET_VERTICAL_FOV_DEGREES,
    DynamicSetSceneCertificate,
    preflight_dynamic_set_episode,
)
from world_model.training.qualification_core import canonical_sha256

DEFAULT_MATERIALIZATION_ATTEMPTS = 32
_PUBLIC_MATERIALIZATION_SPLITS = frozenset({"training", "development"})
_PROTECTED_MATERIALIZATION_SPLITS = frozenset(
    {"selector", "confirmation", "final_test", "compositional_ood"}
)
_PROTECTED_CAPABILITY_AUTHORITY = object()
_PHYSICS_RATE_HZ = 120.0
_ACTION_FRAMES = (10, 15, 34, 44)
_BIRTH_FRAME = 4
_REMOVAL_FRAME = 54
_EARLY_REPLACEMENT_REMOVAL_FRAME = 3
_EARLY_REPLACEMENT_BIRTH_FRAME = 7
_NATURAL_REPLACEMENT_REMOVAL_FRAME = 22
_NATURAL_REPLACEMENT_BIRTH_FRAME = 28
_WORLD_BOUNDS = ((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0))
_ALBEDO_PALETTE = torch.tensor(
    [
        [0.92, 0.18, 0.14],
        [0.12, 0.72, 0.94],
        [0.96, 0.72, 0.12],
        [0.42, 0.86, 0.24],
        [0.72, 0.30, 0.92],
        [0.96, 0.42, 0.68],
    ],
    dtype=torch.float32,
)
_REPLACEMENT_ALBEDO = torch.tensor([0.14, 0.92, 0.62], dtype=torch.float32)
_PUBLIC_ACTION_HANDLE_COSINE_MARGIN = 0.05
_PUBLIC_ACTION_SINGLE_COSINE = 0.95
_PUBLIC_BOUNDARY_SCHEMA = "dynamic_set_public_boundary_v1"
_UNBOUND_ROW_SHA256 = "0" * 64
_PUBLIC_FRAME_FIELD_NAMES = (
    "frame_index",
    "timestamp",
    "rgb",
    "depth",
    "world_from_camera",
    "intrinsics",
    "known_action_observed",
    "known_action_timestamp",
    "known_action_appearance_handle",
    "known_impulse_world",
)
_PUBLIC_TENSOR_SCHEMA = {
    "rgb": ((3, *DYNAMIC_SET_IMAGE_SIZE), torch.float32),
    "depth": ((1, *DYNAMIC_SET_IMAGE_SIZE), torch.float32),
    "world_from_camera": ((4, 4), torch.float32),
    "intrinsics": ((3, 3), torch.float32),
    "known_action_observed": ((1,), torch.bool),
    "known_action_timestamp": ((1,), torch.float32),
    "known_action_appearance_handle": ((1, 8), torch.float32),
    "known_impulse_world": ((1, 3), torch.float32),
}


@dataclass(frozen=True, slots=True)
class DynamicSetPublicFrame:
    """One truth-free public RGB-D/action frame from an accepted episode."""

    frame_index: int
    timestamp: float
    rgb: Tensor
    depth: Tensor
    world_from_camera: Tensor
    intrinsics: Tensor
    known_action_observed: Tensor
    known_action_timestamp: Tensor
    known_action_appearance_handle: Tensor
    known_impulse_world: Tensor

    def world_impulse_action(
        self,
        pre_action_belief: WorldBelief | None = None,
    ) -> WorldImpulseAction | None:
        """Bind this public appearance-addressed command to a checkpoint ID.

        The public frame intentionally carries neither a simulator object ID
        nor a padded target slot.  An observed command must be resolved against
        the checkpoint's belief *before* the current RGB-D frame is ingested,
        so the absolute-timestamp action can be applied causally while the
        model propagates to this frame.
        """

        if (
            self.known_action_observed.shape != (1,)
            or self.known_action_observed.dtype is not torch.bool
            or self.known_action_observed.device.type != "cpu"
        ):
            raise ValueError("known_action_observed must be boolean CPU [1]")
        if (
            self.known_action_timestamp.shape != (1,)
            or self.known_action_timestamp.dtype is not torch.float32
        ):
            raise ValueError("known_action_timestamp must be float32 [1]")
        if (
            self.known_action_appearance_handle.shape != (1, 8)
            or self.known_action_appearance_handle.dtype is not torch.float32
        ):
            raise ValueError("known_action_appearance_handle must be float32 [1,8]")
        if (
            self.known_impulse_world.shape != (1, 3)
            or self.known_impulse_world.dtype is not torch.float32
        ):
            raise ValueError("known_impulse_world must be float32 [1,3]")
        for name in (
            "known_action_timestamp",
            "known_action_appearance_handle",
            "known_impulse_world",
        ):
            value = getattr(self, name)
            if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite CPU public data")
        observed = bool(self.known_action_observed[0])
        if not observed:
            if (
                float(self.known_action_timestamp[0]) != -1.0
                or bool(self.known_action_appearance_handle.ne(0).any())
                or bool(self.known_impulse_world.ne(0).any())
            ):
                raise ValueError("an unobserved public action must contain only sentinels")
            return None
        if float(self.known_action_timestamp[0]) != self.timestamp:
            raise ValueError("public action timestamp differs from its exact application time")
        if bool(self.known_action_appearance_handle.eq(0).all()):
            raise ValueError("an observed public action requires a nonzero appearance handle")
        if bool(self.known_impulse_world.eq(0).all()):
            raise ValueError("an observed public action requires a nonzero impulse")
        if not isinstance(pre_action_belief, WorldBelief):
            raise TypeError("an observed public action requires the pre-action WorldBelief")
        # Specification 1.61 explicitly calibrates fast uncertainty down to
        # log variance -32. Keep this public handle resolver aligned with that
        # frozen profile instead of applying the generic legacy -30 guard.
        pre_action_belief.validate(log_variance_bounds=(-32.0, 20.0))
        if pre_action_belief.batch_size != 1:
            raise ValueError("a public dynamic-set action requires a B1 pre-action belief")
        handle = self.known_action_appearance_handle.to(
            device=pre_action_belief.device,
            dtype=pre_action_belief.dtype,
        )
        object_id = resolve_planning_target_handle(
            pre_action_belief,
            handle,
            minimum_cosine_margin=_PUBLIC_ACTION_HANDLE_COSINE_MARGIN,
            minimum_single_cosine=_PUBLIC_ACTION_SINGLE_COSINE,
        )
        action = WorldImpulseAction(
            timestamp=self.known_action_timestamp.to(
                device=pre_action_belief.device,
                dtype=pre_action_belief.dtype,
            ).clone(),
            object_id=object_id,
            impulse_world=self.known_impulse_world.to(
                device=pre_action_belief.device,
                dtype=pre_action_belief.dtype,
            ).clone(),
        )
        action.validate_for(pre_action_belief)
        return action


@dataclass(frozen=True, slots=True)
class DynamicSetPublicBoundaryEvidence:
    """Digest-bound proof that a public frame stream is truth-isolated.

    The evidence deliberately contains only structural counts and hashes. It
    never serializes simulator truth, storage addresses, labels, or private
    event-ledger values.
    """

    schema: str
    row_sha256: str
    truth_bound: bool
    frame_count: int
    exact_frame_type_count: int
    exact_schema_frame_count: int
    public_tensor_count: int
    truth_tensor_count: int
    unexpected_frame_type_count: int
    subclass_frame_count: int
    unexpected_field_count: int
    missing_field_count: int
    invalid_public_field_count: int
    missing_truth_root_count: int
    public_storage_alias_count: int
    truth_storage_alias_count: int
    public_payload_sha256: str
    boundary_sha256: str

    @property
    def truth_leakage_count(self) -> int:
        """Return the conservative count consumed by the promotion gate."""

        return self._boundary_violation_count(require_truth_binding=True)

    def _boundary_violation_count(self, *, require_truth_binding: bool) -> int:
        expected_tensor_count = DYNAMIC_SET_FRAMES * len(_PUBLIC_TENSOR_SCHEMA)
        return (
            int(self.frame_count != DYNAMIC_SET_FRAMES)
            + int(self.exact_frame_type_count != DYNAMIC_SET_FRAMES)
            + int(self.exact_schema_frame_count != DYNAMIC_SET_FRAMES)
            + int(self.public_tensor_count != expected_tensor_count)
            + self.unexpected_frame_type_count
            + self.unexpected_field_count
            + self.missing_field_count
            + self.invalid_public_field_count
            + self.public_storage_alias_count
            + self.truth_storage_alias_count
            + int(require_truth_binding and not self.truth_bound)
            + int(require_truth_binding and self.row_sha256 == _UNBOUND_ROW_SHA256)
            + (self.missing_truth_root_count if require_truth_binding else 0)
            + int(require_truth_binding and self.truth_tensor_count == 0)
        )

    def _unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "row_sha256": self.row_sha256,
            "truth_bound": self.truth_bound,
            "frame_count": self.frame_count,
            "exact_frame_type_count": self.exact_frame_type_count,
            "exact_schema_frame_count": self.exact_schema_frame_count,
            "public_tensor_count": self.public_tensor_count,
            "truth_tensor_count": self.truth_tensor_count,
            "unexpected_frame_type_count": self.unexpected_frame_type_count,
            "subclass_frame_count": self.subclass_frame_count,
            "unexpected_field_count": self.unexpected_field_count,
            "missing_field_count": self.missing_field_count,
            "invalid_public_field_count": self.invalid_public_field_count,
            "missing_truth_root_count": self.missing_truth_root_count,
            "public_storage_alias_count": self.public_storage_alias_count,
            "truth_storage_alias_count": self.truth_storage_alias_count,
            "public_payload_sha256": self.public_payload_sha256,
        }

    def validate(self) -> DynamicSetPublicBoundaryEvidence:
        if self.schema != _PUBLIC_BOUNDARY_SCHEMA:
            raise ValueError("public-boundary evidence schema differs")
        for name in ("row_sha256", "public_payload_sha256", "boundary_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.truth_bound) is not bool:
            raise TypeError("public-boundary truth_bound must be an exact boolean")
        for name in (
            "frame_count",
            "exact_frame_type_count",
            "exact_schema_frame_count",
            "public_tensor_count",
            "truth_tensor_count",
            "unexpected_frame_type_count",
            "subclass_frame_count",
            "unexpected_field_count",
            "missing_field_count",
            "invalid_public_field_count",
            "missing_truth_root_count",
            "public_storage_alias_count",
            "truth_storage_alias_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"public-boundary {name} must be a nonnegative integer")
        if self.exact_frame_type_count > self.frame_count:
            raise ValueError("exact public-frame type count exceeds frame count")
        if self.exact_schema_frame_count > self.frame_count:
            raise ValueError("exact public-frame schema count exceeds frame count")
        if self.subclass_frame_count > self.unexpected_frame_type_count:
            raise ValueError("public-frame subclass count exceeds unexpected-type count")
        if canonical_sha256(self._unsigned()) != self.boundary_sha256:
            raise ValueError("public-boundary evidence digest mismatch")
        return self

    def require_clean(self, *, require_truth_binding: bool) -> DynamicSetPublicBoundaryEvidence:
        """Reject every malformed, extensible, aliased, or unbound boundary."""

        self.validate()
        if require_truth_binding and not self.truth_bound:
            raise ValueError("public-boundary evidence is not bound to simulator truth")
        if self._boundary_violation_count(require_truth_binding=require_truth_binding):
            raise ValueError("public-frame boundary evidence contains a leakage violation")
        return self

    def bind_manifest_row(
        self,
        row: PhysicalManifestRow,
    ) -> DynamicSetPublicBoundaryEvidence:
        """Bind frame-only evidence to a row without asserting truth isolation."""

        if type(row) is not PhysicalManifestRow:
            raise TypeError("public-boundary row binding requires exact PhysicalManifestRow")
        self.validate()
        if self.truth_bound or self.row_sha256 != _UNBOUND_ROW_SHA256:
            raise ValueError("only unbound frame evidence may receive a manifest-row binding")
        return _make_public_boundary_evidence(
            row_sha256=canonical_sha256(asdict(row)),
            truth_bound=self.truth_bound,
            frame_count=self.frame_count,
            exact_frame_type_count=self.exact_frame_type_count,
            exact_schema_frame_count=self.exact_schema_frame_count,
            public_tensor_count=self.public_tensor_count,
            truth_tensor_count=self.truth_tensor_count,
            unexpected_frame_type_count=self.unexpected_frame_type_count,
            subclass_frame_count=self.subclass_frame_count,
            unexpected_field_count=self.unexpected_field_count,
            missing_field_count=self.missing_field_count,
            invalid_public_field_count=self.invalid_public_field_count,
            missing_truth_root_count=self.missing_truth_root_count,
            public_storage_alias_count=self.public_storage_alias_count,
            truth_storage_alias_count=self.truth_storage_alias_count,
            public_payload_sha256=self.public_payload_sha256,
        )


@dataclass(frozen=True)
class DynamicSetMaterialization:
    """Accepted episode plus deterministic rejection audit information."""

    row: PhysicalManifestRow
    episode: Episode
    known_action_observed: Tensor
    certificate: DynamicSetSceneCertificate
    accepted_seed: int
    attempt_count: int
    rejection_reasons: tuple[str, ...]

    @property
    def rejection_count(self) -> int:
        return len(self.rejection_reasons)

    @property
    def rejection_rate(self) -> float:
        return self.rejection_count / self.attempt_count

    def public_frames(self) -> Iterator[DynamicSetPublicFrame]:
        """Yield the exact 56-frame observable stream in temporal order."""

        yield from iter_dynamic_set_public_frames(self)

    def public_frames_with_boundary(
        self,
    ) -> tuple[tuple[DynamicSetPublicFrame, ...], DynamicSetPublicBoundaryEvidence]:
        """Return public frames with their row-bound truth-isolation proof."""

        return _materialize_dynamic_set_public_frames_with_boundary(self)


class DynamicSetMaterializationError(RuntimeError):
    """Raised after a row exhausts its bounded deterministic candidate set."""


class _ProtectedPhysicalMaterializationCapability:
    """Ephemeral, ordered authority for one protected physical population.

    The constructor is intentionally module-private and authority-guarded.  A
    governed evaluator mints the capability only after validating the ledger
    permit, protocol, frozen manifest digest, and exact row order.  Each row is
    claimed before any simulator truth is generated; an out-of-order claim
    poisons the capability and a claimed row can never be generated twice.
    """

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
        rows: Sequence[PhysicalManifestRow],
        protocol_sha256: str,
        manifest_sha256: str,
        permit_index: int,
        permit_nonce: str,
    ) -> None:
        if authority is not _PROTECTED_CAPABILITY_AUTHORITY:
            raise PermissionError("protected physical capabilities are evaluator-owned")
        resolved = tuple(rows)
        if split not in _PROTECTED_MATERIALIZATION_SPLITS:
            raise PermissionError("protected physical capability has a public split")
        if not resolved or any(row.split != split for row in resolved):
            raise ValueError("protected physical capability rows differ from its split")
        self._split = split
        self._rows = resolved
        self._protocol_sha256 = protocol_sha256
        self._manifest_sha256 = manifest_sha256
        self._permit_index = permit_index
        self._permit_nonce = permit_nonce
        self._cursor = 0
        self._closed = False

    def claim(self, row: PhysicalManifestRow) -> None:
        if self._closed:
            raise PermissionError("protected physical capability is closed or already consumed")
        if self._cursor >= len(self._rows) or row != self._rows[self._cursor]:
            self._closed = True
            raise PermissionError("protected physical capability row order differs")
        self._cursor += 1

    def finish(self) -> None:
        if self._closed:
            raise PermissionError("protected physical capability is closed")
        if self._cursor != len(self._rows):
            self._closed = True
            raise PermissionError("protected physical capability population is incomplete")
        self._closed = True

    def close(self) -> None:
        self._closed = True


def _mint_protected_physical_materialization_capability(
    *,
    split: str,
    rows: Sequence[PhysicalManifestRow],
    protocol_sha256: str,
    manifest_sha256: str,
    permit_index: int,
    permit_nonce: str,
) -> _ProtectedPhysicalMaterializationCapability:
    """Mint one internal capability after the evaluator validates bindings."""

    return _ProtectedPhysicalMaterializationCapability(
        authority=_PROTECTED_CAPABILITY_AUTHORITY,
        split=split,
        rows=rows,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        permit_index=permit_index,
        permit_nonce=permit_nonce,
    )


@dataclass(frozen=True)
class _LifecyclePlan:
    slot: int | None
    birth_frame: int | None
    removal_frame: int | None
    birth_object_id: int | None
    birth_position: Tensor | None
    birth_velocity: Tensor | None
    birth_albedo: Tensor | None


@dataclass(frozen=True)
class _Candidate:
    episode: Episode
    known_action_observed: Tensor
    counterfactual_no_action_pair_collision: Tensor | None


def _validate_row(row: PhysicalManifestRow) -> None:
    if not isinstance(row, PhysicalManifestRow):
        raise TypeError("row must be a PhysicalManifestRow")
    if row.split not in {
        "training",
        "development",
        "selector",
        "confirmation",
        "final_test",
        "compositional_ood",
    }:
        raise ValueError("manifest row has an invalid physical split")
    for name in ("ordinal", "seed", "cell_index", "object_count"):
        value = getattr(row, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"manifest row {name} must be a nonnegative integer")
    for name in ("contact", "dynamic_membership", "known_action"):
        if type(getattr(row, name)) is not bool:
            raise TypeError(f"manifest row {name} must be boolean")
    if row.cell_index not in range(len(PHYSICAL_CELLS)):
        raise ValueError("manifest row has an invalid physical cell index")
    if row.lifecycle_schedule not in {"none", "birth", "removal", "remove_then_birth"}:
        raise ValueError("manifest row has an invalid lifecycle schedule")
    if row.distribution not in {"in_distribution", "compositional_ood"}:
        raise ValueError("manifest row has an invalid distribution")
    if (row.split == "compositional_ood") != (row.distribution == "compositional_ood"):
        raise ValueError("manifest row split and distribution disagree")
    cell = PHYSICAL_CELLS[row.cell_index]
    if (
        row.object_count,
        row.contact,
        row.dynamic_membership,
    ) != (cell.object_count, cell.contact, cell.dynamic_membership):
        raise ValueError("manifest row fields disagree with its frozen physical cell")
    expected_dynamic = row.lifecycle_schedule != "none"
    if expected_dynamic != row.dynamic_membership:
        raise ValueError("manifest lifecycle schedule disagrees with dynamic membership")
    if row.known_action:
        if type(row.action_target_rank) is not int or row.action_target_rank not in range(
            row.object_count
        ):
            raise ValueError("known-action target rank lies outside the manifest cardinality")
        if type(row.action_time_stratum) is not int or row.action_time_stratum not in range(
            len(_ACTION_FRAMES)
        ):
            raise ValueError("known-action time stratum lies outside [0,3]")
    elif row.action_target_rank is not None or row.action_time_stratum is not None:
        raise ValueError("action-free rows cannot declare an action target or time")
    if type(row.camera_stratum) is not int or row.camera_stratum not in range(8):
        raise ValueError("camera stratum lies outside [0,7]")
    if row.contact:
        if row.contact_origin == "action_induced" and not row.known_action:
            raise ValueError("action-induced contact requires a known action")
        if row.contact_origin not in {"natural", "action_induced"}:
            raise ValueError("contact rows require natural or action-induced provenance")
        if row.contact_geometry not in {"head_on", "glancing"}:
            raise ValueError("contact rows require head-on or glancing geometry")
    elif row.contact_origin != "none" or row.contact_geometry != "none":
        raise ValueError("contact-free rows cannot declare contact origin or geometry")


def _candidate_seed(seed: int, attempt_index: int) -> int:
    if attempt_index == 0:
        return int(seed)
    digest = hashlib.sha256(f"dynamic-set-1.61:{seed}:{attempt_index}".encode("ascii")).digest()
    # Retry candidates live in a high deterministic namespace rather than
    # colliding with the frozen 70M--80M manifest seed ranges.
    return (int.from_bytes(digest[:8], byteorder="big") & ((1 << 62) - 1)) | (1 << 62)


def _camera(row: PhysicalManifestRow) -> CameraFrame:
    angle = 2.0 * math.pi * row.camera_stratum / 8.0
    camera_distance = 10.0
    position = torch.tensor(
        [
            camera_distance * math.sin(angle),
            0.0,
            -camera_distance * math.cos(angle),
        ],
        dtype=torch.float32,
    )
    target = torch.zeros(3, dtype=torch.float32)
    world_from_camera = look_at_world_from_camera(position, target)
    return CameraFrame(
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


def _camera_to_world(values: Tensor, camera: CameraFrame, *, point: bool) -> Tensor:
    rotation = camera.world_from_camera[:3, :3]
    transformed = values @ rotation.transpose(0, 1)
    if point:
        transformed = transformed + camera.world_from_camera[:3, 3]
    return transformed


def _interaction_pair(row: PhysicalManifestRow) -> tuple[int, int] | None:
    if not row.contact:
        return None
    if row.contact_origin == "action_induced":
        assert row.action_target_rank is not None
        target = row.action_target_rank
        companion = 1 if target == 0 else 0
        return target, companion
    return 0, 1


def _layout(
    row: PhysicalManifestRow,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, tuple[int, int] | None]:
    count = row.object_count
    position = torch.zeros((DYNAMIC_SET_MAX_OBJECTS, 3), dtype=torch.float32)
    velocity = torch.zeros_like(position)
    global_x = 0.18 * (float(torch.rand((), generator=generator)) - 0.5)
    global_y = 0.18 * (float(torch.rand((), generator=generator)) - 0.5)
    camera_depth = 10.0 + 0.8 * (float(torch.rand((), generator=generator)) - 0.5)
    pair = _interaction_pair(row)

    if pair is None:
        safe_locations = (
            (-1.65, -0.90),
            (0.00, -0.90),
            (1.65, -0.90),
            (-1.65, 0.90),
            (0.00, 0.90),
            (1.65, 0.90),
        )
        for slot in range(count):
            x, y = safe_locations[slot]
            position[slot] = torch.tensor(
                [x + global_x, y + global_y, camera_depth],
                dtype=torch.float32,
            )
            velocity[slot, 2] = 0.025 if slot % 2 == 0 else -0.025
        return position, velocity, None

    first, second = pair
    non_pair = [slot for slot in range(count) if slot not in pair]
    if non_pair:
        x_locations = torch.linspace(-2.10, 2.10, len(non_pair))
        for offset, slot in enumerate(non_pair):
            position[slot] = torch.tensor(
                [
                    float(x_locations[offset]) + global_x,
                    -1.20 + global_y,
                    camera_depth + 0.15 * (offset % 2),
                ],
                dtype=torch.float32,
            )
            velocity[slot, 2] = 0.02 if offset % 2 == 0 else -0.02

    separation = 0.82 if row.contact_origin == "action_induced" else 1.10
    lateral = 0.18 if row.contact_geometry == "glancing" else 0.0
    longitudinal = math.sqrt(separation * separation - lateral * lateral)
    pair_center = torch.tensor(
        [global_x, 1.05 + global_y, camera_depth],
        dtype=torch.float32,
    )
    displacement = torch.tensor([longitudinal, lateral, 0.0], dtype=torch.float32)
    position[first] = pair_center - 0.5 * displacement
    position[second] = pair_center + 0.5 * displacement
    if row.contact_origin == "natural":
        direction = displacement / torch.linalg.vector_norm(displacement)
        lifecycle_slot = count - 1
        if row.lifecycle_schedule == "birth" and lifecycle_slot in pair:
            moving = lifecycle_slot
            companion = second if moving == first else first
            velocity[moving] = (position[companion] - position[moving]) / torch.linalg.vector_norm(
                position[companion] - position[moving]
            )
            velocity[moving] *= 1.05
            velocity[companion].zero_()
        else:
            velocity[first] = 0.50 * direction
            velocity[second] = -0.50 * direction
    else:
        velocity[first].zero_()
        velocity[second].zero_()
    return position, velocity, pair


def _fresh_replacement_id(candidate_seed: int) -> int:
    return DYNAMIC_SET_MAX_OBJECTS + candidate_seed % 2_000_000_000


def _lifecycle_plan(
    row: PhysicalManifestRow,
    candidate_seed: int,
    position: Tensor,
    velocity: Tensor,
    pair: tuple[int, int] | None,
) -> _LifecyclePlan:
    if row.lifecycle_schedule == "none":
        return _LifecyclePlan(None, None, None, None, None, None, None)
    slot = row.object_count - 1
    birth_position = position[slot].clone()
    birth_velocity = velocity[slot].clone()
    if row.lifecycle_schedule == "birth":
        return _LifecyclePlan(
            slot,
            _BIRTH_FRAME,
            None,
            slot,
            birth_position,
            birth_velocity,
            _ALBEDO_PALETTE[slot].clone(),
        )
    if row.lifecycle_schedule == "removal":
        return _LifecyclePlan(slot, None, _REMOVAL_FRAME, None, None, None, None)

    natural_pair_replacement = row.contact_origin == "natural" and pair is not None and slot in pair
    if natural_pair_replacement:
        removal_frame = _NATURAL_REPLACEMENT_REMOVAL_FRAME
        birth_frame = _NATURAL_REPLACEMENT_BIRTH_FRAME
        # The original member owns the natural collision.  Its replacement is
        # deliberately born into the otherwise unused upper lane.
        camera_depth = float(position[slot, 2])
        birth_position = position.new_tensor([0.0, -1.20, camera_depth])
        birth_velocity = torch.zeros(3, dtype=position.dtype)
    else:
        removal_frame = _EARLY_REPLACEMENT_REMOVAL_FRAME
        birth_frame = _EARLY_REPLACEMENT_BIRTH_FRAME
    return _LifecyclePlan(
        slot,
        birth_frame,
        removal_frame,
        _fresh_replacement_id(candidate_seed),
        birth_position,
        birth_velocity,
        _REPLACEMENT_ALBEDO.clone(),
    )


def _make_state(
    row: PhysicalManifestRow,
    candidate_seed: int,
    camera: CameraFrame,
) -> tuple[SphereState, _LifecyclePlan, tuple[int, int] | None]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(candidate_seed)
    local_position, local_velocity, pair = _layout(row, generator)
    position = _camera_to_world(local_position, camera, point=True)
    velocity = _camera_to_world(local_velocity, camera, point=False)
    lifecycle = _lifecycle_plan(
        row,
        candidate_seed,
        local_position,
        local_velocity,
        pair,
    )
    if lifecycle.birth_position is not None:
        lifecycle = replace(
            lifecycle,
            birth_position=_camera_to_world(
                lifecycle.birth_position.unsqueeze(0), camera, point=True
            ).squeeze(0),
            birth_velocity=_camera_to_world(
                lifecycle.birth_velocity.unsqueeze(0), camera, point=False
            ).squeeze(0),
        )

    active = torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.bool)
    active[: row.object_count] = True
    object_id = torch.full((DYNAMIC_SET_MAX_OBJECTS,), -1, dtype=torch.int64)
    object_id[: row.object_count] = torch.arange(row.object_count, dtype=torch.int64)
    if row.lifecycle_schedule == "birth":
        assert lifecycle.slot is not None
        active[lifecycle.slot] = False
        object_id[lifecycle.slot] = -1
        velocity[lifecycle.slot].zero_()

    albedo = _ALBEDO_PALETTE.clone()
    position = torch.where(active.unsqueeze(-1), position, torch.zeros_like(position))
    velocity = torch.where(active.unsqueeze(-1), velocity, torch.zeros_like(velocity))
    albedo = torch.where(active.unsqueeze(-1), albedo, torch.zeros_like(albedo))
    radius = torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_RADIUS_M)
    mass = torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_MASS)
    restitution = torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_RESTITUTION)
    drag = torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_DRAG)
    friction = torch.full((DYNAMIC_SET_MAX_OBJECTS, 1), DYNAMIC_SET_FRICTION)
    orientation = torch.zeros((DYNAMIC_SET_MAX_OBJECTS, 4), dtype=torch.float32)
    orientation[:, 3] = 1.0
    state = SphereState(
        object_id=object_id,
        active=active,
        position=position,
        velocity=velocity,
        radius=radius,
        mass=mass,
        restitution=restitution,
        drag=drag,
        friction=friction,
        albedo=albedo,
        orientation=orientation,
        angular_velocity=torch.zeros((DYNAMIC_SET_MAX_OBJECTS, 3), dtype=torch.float32),
        sleeping=torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.bool),
        sleep_counter=torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.int64),
    )
    state.validate()
    return state, lifecycle, pair


def _apply_lifecycle(
    state: SphereState,
    lifecycle: _LifecyclePlan,
    frame_index: int,
) -> tuple[SphereState, Tensor, Tensor]:
    created = torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.bool)
    removed = torch.zeros_like(created)
    if frame_index == 0:
        created.copy_(state.active)
    if lifecycle.slot is None:
        return state, created, removed

    is_removal = frame_index == lifecycle.removal_frame
    is_birth = frame_index == lifecycle.birth_frame
    if not is_removal and not is_birth:
        return state, created, removed

    slot = lifecycle.slot
    active = state.active.clone()
    object_id = state.object_id.clone()
    position = state.position.clone()
    velocity = state.velocity.clone()
    radius = state.radius.clone()
    mass = state.mass.clone()
    restitution = state.restitution.clone()
    drag = state.drag.clone()
    friction = state.friction.clone()
    albedo = state.albedo.clone()
    orientation = state.orientation.clone()
    angular_velocity = state.angular_velocity.clone()
    sleeping = state.sleeping.clone()
    sleep_counter = state.sleep_counter.clone()
    if is_removal:
        removed[slot] = True
        active[slot] = False
        object_id[slot] = -1
        position[slot].zero_()
        velocity[slot].zero_()
        albedo[slot].zero_()
        orientation[slot].zero_()
        orientation[slot, 3] = 1.0
        angular_velocity[slot].zero_()
        sleeping[slot] = False
        sleep_counter[slot] = 0
    if is_birth:
        assert lifecycle.birth_object_id is not None
        assert lifecycle.birth_position is not None
        assert lifecycle.birth_velocity is not None
        assert lifecycle.birth_albedo is not None
        created[slot] = True
        active[slot] = True
        object_id[slot] = lifecycle.birth_object_id
        position[slot] = lifecycle.birth_position
        velocity[slot] = lifecycle.birth_velocity
        radius[slot] = DYNAMIC_SET_RADIUS_M
        mass[slot] = DYNAMIC_SET_MASS
        restitution[slot] = DYNAMIC_SET_RESTITUTION
        drag[slot] = DYNAMIC_SET_DRAG
        friction[slot] = DYNAMIC_SET_FRICTION
        albedo[slot] = lifecycle.birth_albedo
        orientation[slot].zero_()
        orientation[slot, 3] = 1.0
        angular_velocity[slot].zero_()
        sleeping[slot] = False
        sleep_counter[slot] = 0
    state = replace(
        state,
        object_id=object_id,
        active=active,
        position=position,
        velocity=velocity,
        radius=radius,
        mass=mass,
        restitution=restitution,
        drag=drag,
        friction=friction,
        albedo=albedo,
        orientation=orientation,
        angular_velocity=angular_velocity,
        sleeping=sleeping,
        sleep_counter=sleep_counter,
    )
    state.validate()
    return state, created, removed


def _state_record(state: SphereState, visible_fraction: Tensor) -> dict[str, Tensor]:
    return {
        "id": state.object_id.clone(),
        "active": state.active.clone(),
        "position": state.position.clone(),
        "velocity": state.velocity.clone(),
        "orientation": state.orientation.clone(),
        "angular_velocity": state.angular_velocity.clone(),
        "radius": state.radius.clone(),
        "mass": state.mass.clone(),
        "restitution": state.restitution.clone(),
        "drag": state.drag.clone(),
        "friction": state.friction.clone(),
        "albedo": state.albedo.clone(),
        "visible_fraction": visible_fraction.clone(),
        "sleeping": state.sleeping.clone(),
    }


def _event_record(
    physics: PhysicsStepEvents,
    *,
    created: Tensor,
    removed: Tensor,
    interval_start: float,
) -> dict[str, Tensor]:
    first_event_time = torch.where(
        physics.first_event_offset >= 0,
        physics.first_event_offset + interval_start,
        physics.first_event_offset,
    )
    floor_index = BOUNDARY_NAMES.index("floor")
    wall_indices = [
        index for index, name in enumerate(BOUNDARY_NAMES) if name not in {"floor", "ceiling"}
    ]
    return {
        "pair_contact": physics.pair_contact.clone(),
        "pair_collision": physics.pair_collision.clone(),
        "sphere_sphere": physics.pair_collision.clone(),
        "pair_impulse": physics.pair_impulse.clone(),
        "pair_penetration": physics.pair_penetration.clone(),
        "boundary_contact": physics.boundary_contact.clone(),
        "boundary_collision": physics.boundary_collision.clone(),
        "boundary_impulse": physics.boundary_impulse.clone(),
        "boundary_penetration": physics.boundary_penetration.clone(),
        "ground_contact": physics.boundary_contact[:, floor_index].clone(),
        "ground_collision": physics.boundary_collision[:, floor_index].clone(),
        "wall_collision": physics.boundary_collision[:, wall_indices].clone(),
        "collision": physics.collision.clone(),
        "contact": physics.contact.clone(),
        "sleeping": physics.sleeping.clone(),
        "external_impulse": physics.external_impulse.clone(),
        "externally_actuated": torch.linalg.vector_norm(physics.external_impulse, dim=-1) > 0,
        "created": created.clone(),
        "removed": removed.clone(),
        "first_event_time": first_event_time,
    }


def _stack_records(records: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot stack an empty record sequence")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise ValueError("record keys changed within a dynamic-set episode")
    return {key: torch.stack([record[key] for record in records]) for key in keys}


def _camera_velocities(
    world_from_camera: Tensor,
    timestamps: Tensor,
) -> tuple[Tensor, Tensor]:
    position = world_from_camera[:, :3, 3]
    linear = torch.zeros_like(position)
    angular = torch.zeros_like(position)
    dt = (timestamps[1:] - timestamps[:-1]).clamp_min(1.0e-8)
    linear[1:] = (position[1:] - position[:-1]) / dt[:, None]
    linear[0] = linear[1]
    rotation = world_from_camera[:, :3, :3]
    delta = rotation[1:] @ rotation[:-1].transpose(-1, -2)
    skew = 0.5 * (delta - delta.transpose(-1, -2))
    rotation_vector = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=-1)
    angular[1:] = rotation_vector / dt[:, None]
    angular[0] = angular[1]
    return linear, angular


def _known_action_impulse(
    row: PhysicalManifestRow,
    pair: tuple[int, int] | None,
    camera: CameraFrame,
) -> Tensor:
    """Construct the public command from frozen scene factors, never truth state."""

    impulse = torch.zeros((DYNAMIC_SET_MAX_OBJECTS, 3), dtype=torch.float32)
    if not row.known_action:
        return impulse
    assert row.action_target_rank is not None
    target = row.action_target_rank
    if row.contact_origin == "action_induced":
        assert pair is not None
        if pair[0] != target:
            raise ValueError("action-induced layout must place the declared target first")
        separation = 0.82
        lateral = 0.18 if row.contact_geometry == "glancing" else 0.0
        longitudinal = math.sqrt(separation * separation - lateral * lateral)
        local_direction = impulse.new_tensor([longitudinal, lateral, 0.0]) / separation
        direction = _camera_to_world(local_direction.unsqueeze(0), camera, point=False).squeeze(0)
        impulse[target] = 1.35 * direction
    else:
        # Contact-free actions move along the camera ray, preserving the
        # image-plane separation designed into the materialized layout.
        impulse[target] = 0.18 * camera.world_from_camera[:3, 2]
    return impulse


def _known_action_object_id(
    row: PhysicalManifestRow,
    lifecycle: _LifecyclePlan,
    action_frame: int,
) -> int:
    """Resolve the commanded public ID from the frozen lifecycle schedule."""

    assert row.action_target_rank is not None
    target = row.action_target_rank
    if lifecycle.slot != target:
        return target
    if lifecycle.removal_frame is not None:
        if action_frame < lifecycle.removal_frame:
            return target
        if lifecycle.birth_frame is None or action_frame < lifecycle.birth_frame:
            raise ValueError("known action is scheduled while its target is absent")
        if lifecycle.birth_object_id is None:
            raise RuntimeError("replacement lifecycle lacks its frozen persistent ID")
        return lifecycle.birth_object_id
    if lifecycle.birth_frame is not None:
        if action_frame < lifecycle.birth_frame:
            raise ValueError("known action is scheduled before its target birth")
        if lifecycle.birth_object_id is None:
            raise RuntimeError("birth lifecycle lacks its frozen persistent ID")
        return lifecycle.birth_object_id
    return target


def _build_candidate_episode(row: PhysicalManifestRow, candidate_seed: int) -> _Candidate:
    camera = _camera(row)
    state, lifecycle, pair = _make_state(row, candidate_seed, camera)
    physics = PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=_WORLD_BOUNDS,
        max_substep=1.0 / _PHYSICS_RATE_HZ,
        solver_iterations=2,
    )
    physics.validate()
    pair_cache = prevalidated_sphere_pair_cache(state)
    # Advance an exact paired control from the same initial state and lifecycle
    # while withholding only the declared impulse.  Its collision ledger is
    # private preflight evidence: it is never exposed by ``public_frames``.
    requires_causal_control = row.known_action and row.contact
    no_action_state = state.clone() if requires_causal_control else None
    no_action_pair_cache = (
        prevalidated_sphere_pair_cache(no_action_state) if no_action_state is not None else None
    )
    render_cache = prevalidated_sphere_render_cache(
        state,
        camera,
        DYNAMIC_SET_IMAGE_SIZE,
    )
    timestamps = torch.arange(DYNAMIC_SET_FRAMES, dtype=torch.float32) / (DYNAMIC_SET_FRAME_RATE_HZ)
    rgb_frames: list[Tensor] = []
    depth_frames: list[Tensor] = []
    state_records: list[dict[str, Tensor]] = []
    label_records: list[dict[str, Tensor]] = []
    event_records: list[dict[str, Tensor]] = []
    no_action_pair_collision_records: list[Tensor] = []
    pending_physics = empty_physics_events(DYNAMIC_SET_MAX_OBJECTS)
    no_action_pending_physics = (
        empty_physics_events(DYNAMIC_SET_MAX_OBJECTS) if requires_causal_control else None
    )
    action_frame = (
        None if row.action_time_stratum is None else _ACTION_FRAMES[row.action_time_stratum]
    )
    # These are the public action declarations, constructed from the manifest
    # before inspecting any simulator event output.  Keeping this ledger
    # independent makes the hidden-impulse equality check in scene preflight
    # meaningful rather than tautological.
    declared_action_observed = torch.zeros(
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), dtype=torch.bool
    )
    declared_action_object_id = torch.full(
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), -1, dtype=torch.int64
    )
    declared_action_timestamp = torch.full(
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), -1.0, dtype=torch.float32
    )
    declared_impulse_world = torch.zeros(
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 3), dtype=torch.float32
    )
    if action_frame is not None:
        assert row.action_target_rank is not None
        target = row.action_target_rank
        declared_action_observed[action_frame, target] = True
        declared_action_object_id[action_frame, target] = _known_action_object_id(
            row,
            lifecycle,
            action_frame,
        )
        declared_action_timestamp[action_frame, target] = timestamps[action_frame]
        declared_impulse_world[action_frame] = _known_action_impulse(row, pair, camera)
        if bool(
            declared_impulse_world[action_frame, ~declared_action_observed[action_frame]]
            .ne(0)
            .any()
        ):
            raise ValueError("simulator impulse targets an undeclared public object")

    for frame_index, timestamp_tensor in enumerate(timestamps):
        state, created, removed = _apply_lifecycle(state, lifecycle, frame_index)
        if no_action_state is not None:
            no_action_state, no_action_created, no_action_removed = _apply_lifecycle(
                no_action_state,
                lifecycle,
                frame_index,
            )
            if not torch.equal(created, no_action_created) or not torch.equal(
                removed, no_action_removed
            ):
                raise RuntimeError("paired no-action lifecycle diverged from the acted scene")
        frame_camera = replace(camera, timestamp=float(timestamp_tensor))
        rendered = render_spheres(
            state,
            frame_camera,
            DYNAMIC_SET_IMAGE_SIZE,
            edge_softness_pixels=1.0,
            noise_std=0.0,
            _prevalidated_cache=render_cache,
        )
        labels = make_perception_labels(state, rendered, DYNAMIC_SET_IMAGE_SIZE)
        validate_perception_labels(
            labels,
            max_objects=DYNAMIC_SET_MAX_OBJECTS,
            image_size=DYNAMIC_SET_IMAGE_SIZE,
        )
        rgb_frames.append(rendered.rgb)
        depth_frames.append(rendered.depth_buffer.unsqueeze(0))
        state_records.append(_state_record(state, rendered.visible_fraction))
        label_records.append(labels)
        interval_start = max(0.0, float(timestamp_tensor) - 1.0 / DYNAMIC_SET_FRAME_RATE_HZ)
        event_records.append(
            _event_record(
                pending_physics,
                created=created,
                removed=removed,
                interval_start=interval_start,
            )
        )
        if no_action_pending_physics is not None:
            no_action_pair_collision_records.append(
                no_action_pending_physics.pair_collision.clone()
            )
        if frame_index + 1 < DYNAMIC_SET_FRAMES:
            next_frame = frame_index + 1
            if next_frame == action_frame:
                assert row.action_target_rank is not None
                target = row.action_target_rank
                if not bool(state.active[target]):
                    raise ValueError("known-action target is not active at its public action frame")
                external_impulse = declared_impulse_world[next_frame]
                action_object_id = int(declared_action_object_id[next_frame, target])
                if int(state.object_id[target]) != action_object_id:
                    raise ValueError("simulator target ID differs from the frozen public command")
            else:
                external_impulse = torch.zeros_like(state.velocity)
            state, pending_physics = advance_spheres_no_boundary_contacts_prevalidated(
                state,
                1.0 / DYNAMIC_SET_FRAME_RATE_HZ,
                physics,
                external_impulse=torch.zeros_like(external_impulse),
                pair_cache=pair_cache,
            )
            if no_action_state is not None:
                no_action_state, no_action_pending_physics = (
                    advance_spheres_no_boundary_contacts_prevalidated(
                        no_action_state,
                        1.0 / DYNAMIC_SET_FRAME_RATE_HZ,
                        physics,
                        external_impulse=torch.zeros_like(no_action_state.velocity),
                        pair_cache=no_action_pair_cache,
                    )
                )
            if next_frame == action_frame:
                acted = declared_action_observed[next_frame]
                velocity = state.velocity + external_impulse / state.mass.clamp_min(1.0e-12)
                state = replace(
                    state,
                    velocity=torch.where(acted.unsqueeze(-1), velocity, state.velocity),
                    sleeping=state.sleeping & ~acted,
                    sleep_counter=torch.where(
                        acted,
                        torch.zeros_like(state.sleep_counter),
                        state.sleep_counter,
                    ),
                )
                state.validate()
                pending_physics = replace(
                    pending_physics,
                    external_impulse=external_impulse.clone(),
                    sleeping=state.sleeping.clone(),
                )

    objects = _stack_records(state_records)
    labels = _stack_records(label_records)
    events = _stack_records(event_records)
    counterfactual_no_action_pair_collision = (
        torch.stack(no_action_pair_collision_records) if no_action_pair_collision_records else None
    )
    known_action_observed = declared_action_observed.clone()
    events["known_action_observed"] = known_action_observed.clone()
    events["known_action_timestamp"] = declared_action_timestamp.clone()
    events["known_impulse_world"] = declared_impulse_world.clone()
    events["known_action_object_id"] = declared_action_object_id.clone()
    if not torch.equal(events["external_impulse"], declared_impulse_world):
        raise ValueError("simulator impulses differ from the independent public action ledger")
    if not torch.equal(
        declared_action_object_id[known_action_observed],
        objects["id"][known_action_observed],
    ):
        raise ValueError("public action target ID differs from the observed persistent object")
    for key in (
        "projected_center",
        "projected_center_pixels",
        "apparent_radius",
        "apparent_radius_normalized",
        "inverse_depth",
        "camera_depth",
        "projected_valid",
    ):
        objects[key] = labels[key]

    world_from_camera = camera.world_from_camera.expand(DYNAMIC_SET_FRAMES, -1, -1).clone()
    camera_from_world = camera.camera_from_world.expand(DYNAMIC_SET_FRAMES, -1, -1).clone()
    intrinsics = camera.intrinsics.expand(DYNAMIC_SET_FRAMES, -1, -1).clone()
    camera_linear, camera_angular = _camera_velocities(world_from_camera, timestamps)
    unique_ids = objects["id"][objects["id"] >= 0].unique()
    episode: Episode = {
        "rgb": torch.stack(rgb_frames).to(torch.float32),
        "depth": torch.stack(depth_frames).to(torch.float32),
        "timestamps": timestamps,
        "frame_mask": torch.ones(DYNAMIC_SET_FRAMES, dtype=torch.bool),
        "camera": {
            "world_from_camera": world_from_camera,
            "camera_from_world": camera_from_world,
            "intrinsics": intrinsics,
            "position": camera.position.expand(DYNAMIC_SET_FRAMES, -1).clone(),
            "target": camera.target.expand(DYNAMIC_SET_FRAMES, -1).clone(),
            "linear_velocity": camera_linear,
            "angular_velocity": camera_angular,
            "calibrated": torch.ones(DYNAMIC_SET_FRAMES, dtype=torch.bool),
        },
        "objects": objects,
        "events": events,
        "labels": labels,
        # The root episode seed identifies the accepted deterministic sample.
        # The frozen manifest seed remains separately bound in metadata so a
        # rejection-sampled retry cannot masquerade as the first candidate.
        "seed": int(candidate_seed),
        "num_objects": int(unique_ids.numel()),
        "metadata": {
            "simulator": "sphere_world",
            "simulator_version": SIMULATOR_VERSION,
            "specification_version": "1.61",
            "manifest_seed": int(row.seed),
            "candidate_seed": int(candidate_seed),
            "manifest_ordinal": int(row.ordinal),
            "physical_cell_index": int(row.cell_index),
            "distribution": row.distribution,
            "scenario": "dynamic_set_public",
            "camera_trajectory": "fixed_stratified",
            "camera_stratum": int(row.camera_stratum),
            "frame_rate": DYNAMIC_SET_FRAME_RATE_HZ,
            "physics_rate": _PHYSICS_RATE_HZ,
            "gravity": (0.0, 0.0, 0.0),
            "world_bounds": _WORLD_BOUNDS,
            "boundary_names": BOUNDARY_NAMES,
            "action_frame": -1 if action_frame is None else action_frame,
            "contact_causality_preflight": "paired_no_action_v1",
        },
    }
    validate_episode(episode)
    return _Candidate(
        episode=episode,
        known_action_observed=known_action_observed,
        counterfactual_no_action_pair_collision=counterfactual_no_action_pair_collision,
    )


def _preflight_lifecycle_visibility(episode: Mapping[str, object]) -> None:
    """Bind lifecycle labels to complete observations on both sides of an event."""

    objects = episode.get("objects")
    events = episode.get("events")
    labels = episode.get("labels")
    if (
        not isinstance(objects, Mapping)
        or not isinstance(events, Mapping)
        or not isinstance(labels, Mapping)
    ):
        raise TypeError("lifecycle visibility preflight requires object/event/label mappings")
    active = objects.get("active")
    object_id = objects.get("id")
    visible_fraction = objects.get("visible_fraction")
    created = events.get("created")
    removed = events.get("removed")
    projected_valid = labels.get("projected_valid")
    values = (active, object_id, visible_fraction, created, removed, projected_valid)
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("lifecycle visibility preflight requires tensor fields")
    assert isinstance(active, Tensor)
    assert isinstance(object_id, Tensor)
    assert isinstance(visible_fraction, Tensor)
    assert isinstance(created, Tensor)
    assert isinstance(removed, Tensor)
    assert isinstance(projected_valid, Tensor)
    if any(value.shape != active.shape for value in values[1:]):
        raise ValueError("lifecycle visibility fields must share shape [T,N]")
    if (
        active.dtype is not torch.bool
        or created.dtype is not torch.bool
        or removed.dtype is not torch.bool
    ):
        raise TypeError("active, created, and removed lifecycle fields must be boolean")
    if bool((created & removed).any()):
        raise ValueError("one object cannot be created and removed in the same frame")

    for frame, slot in torch.nonzero(created, as_tuple=False).tolist():
        if not bool(active[frame, slot]) or int(object_id[frame, slot]) < 0:
            raise ValueError("every creation must produce an active persistent object")
        if not bool(projected_valid[frame, slot]) or float(visible_fraction[frame, slot]) != 1.0:
            raise ValueError("every creation must be fully visible at its event frame")
        if frame > 0:
            if bool(active[frame - 1, slot]) or int(object_id[frame - 1, slot]) != -1:
                raise ValueError("a lifecycle birth must transition from an inactive slot")
            if (
                bool(projected_valid[frame - 1, slot])
                or float(visible_fraction[frame - 1, slot]) != 0.0
            ):
                raise ValueError("a lifecycle birth cannot retain a pre-event projection")

    for frame, slot in torch.nonzero(removed, as_tuple=False).tolist():
        if frame == 0:
            raise ValueError("a lifecycle removal cannot occur before an observed active frame")
        if bool(active[frame, slot]) or int(object_id[frame, slot]) != -1:
            raise ValueError("every removal must leave an inactive padded slot")
        if bool(projected_valid[frame, slot]) or float(visible_fraction[frame, slot]) != 0.0:
            raise ValueError("a removed object cannot retain an event-frame projection")
        if not bool(active[frame - 1, slot]) or int(object_id[frame - 1, slot]) < 0:
            raise ValueError("every removal must follow an active persistent object")
        if (
            not bool(projected_valid[frame - 1, slot])
            or float(visible_fraction[frame - 1, slot]) != 1.0
        ):
            raise ValueError("every removed object must be fully visible immediately beforehand")


def _materialize_dynamic_set_episode_core(
    row: PhysicalManifestRow,
    *,
    maximum_attempts: int = DEFAULT_MATERIALIZATION_ATTEMPTS,
) -> DynamicSetMaterialization:
    """Materialize one already-authorized row after common validation."""

    if (
        isinstance(maximum_attempts, bool)
        or not isinstance(maximum_attempts, int)
        or maximum_attempts <= 0
    ):
        raise ValueError("maximum_attempts must be a positive integer")
    rejection_reasons: list[str] = []
    for attempt_index in range(maximum_attempts):
        candidate_seed = _candidate_seed(row.seed, attempt_index)
        try:
            candidate = _build_candidate_episode(row, candidate_seed)
            certificate = preflight_dynamic_set_episode(
                candidate.episode,
                row,
                known_action_observed=candidate.known_action_observed,
                counterfactual_no_action_pair_collision=(
                    candidate.counterfactual_no_action_pair_collision
                ),
            )
            _preflight_lifecycle_visibility(candidate.episode)
        except (RuntimeError, ValueError) as error:
            rejection_reasons.append(f"{type(error).__name__}: {error}")
            continue
        return DynamicSetMaterialization(
            row=row,
            episode=candidate.episode,
            known_action_observed=candidate.known_action_observed,
            certificate=certificate,
            accepted_seed=candidate_seed,
            attempt_count=attempt_index + 1,
            rejection_reasons=tuple(rejection_reasons),
        )
    last_reason = rejection_reasons[-1] if rejection_reasons else "no candidate was evaluated"
    raise DynamicSetMaterializationError(
        f"failed to materialize row {row.split}:{row.ordinal} after "
        f"{maximum_attempts} deterministic attempts; last rejection: {last_reason}"
    )


def _materialize_dynamic_set_episode(
    row: PhysicalManifestRow,
    *,
    maximum_attempts: int = DEFAULT_MATERIALIZATION_ATTEMPTS,
) -> DynamicSetMaterialization:
    """Materialize only a public row at the deepest unguarded entry."""

    _validate_row(row)
    if row.split not in _PUBLIC_MATERIALIZATION_SPLITS:
        raise PermissionError("protected physical rows require a claimed capability")
    return _materialize_dynamic_set_episode_core(row, maximum_attempts=maximum_attempts)


def materialize_dynamic_set_episode(
    row: PhysicalManifestRow,
    *,
    maximum_attempts: int = DEFAULT_MATERIALIZATION_ATTEMPTS,
) -> DynamicSetMaterialization:
    """Materialize one public training/development row deterministically.

    Protected selector, confirmation, final, and OOD truth is deliberately not
    available from this public function.  It can be generated only inside an
    authorized evaluator after that evaluator validates the ledger permit and
    exact frozen population binding.
    """

    _validate_row(row)
    if row.split not in _PUBLIC_MATERIALIZATION_SPLITS:
        raise PermissionError("protected physical rows require the governed evaluator")
    return _materialize_dynamic_set_episode(row, maximum_attempts=maximum_attempts)


def _materialize_protected_dynamic_set_episode(
    row: PhysicalManifestRow,
    *,
    capability: _ProtectedPhysicalMaterializationCapability,
    maximum_attempts: int = DEFAULT_MATERIALIZATION_ATTEMPTS,
) -> DynamicSetMaterialization:
    """Generate one protected row after an ordered one-shot claim."""

    _validate_row(row)
    if not isinstance(capability, _ProtectedPhysicalMaterializationCapability):
        raise PermissionError("protected physical materialization lacks evaluator authority")
    if row.split not in _PROTECTED_MATERIALIZATION_SPLITS:
        capability.close()
        raise PermissionError("protected physical capability cannot open a public row")
    capability.claim(row)
    return _materialize_dynamic_set_episode_core(row, maximum_attempts=maximum_attempts)


def _observable_action_appearance(image: Tensor, mask: Tensor) -> Tensor:
    """Return the set-mode eight-value appearance statistic for one object."""

    if image.shape != (3, *DYNAMIC_SET_IMAGE_SIZE) or image.dtype is not torch.float32:
        raise ValueError("public action appearance requires one float32 RGB frame")
    if mask.shape != DYNAMIC_SET_IMAGE_SIZE or mask.dtype is not torch.bool:
        raise ValueError("public action appearance requires one boolean object mask")
    if not bool(mask.any()):
        raise ValueError("public action appearance requires a visible target mask")
    weights = mask.to(image.dtype)
    epsilon = torch.finfo(image.dtype).eps
    mass = weights.sum().clamp_min(epsilon)
    mean_rgb = torch.einsum("hw,chw->c", weights, image) / mass
    second_rgb = torch.einsum("hw,chw->c", weights, image.square()) / mass
    std_rgb = (second_rgb - mean_rgb.square() + epsilon).clamp_min(epsilon).sqrt()
    intensity = image.mean(dim=0)
    mean_intensity = torch.einsum("hw,hw->", weights, intensity) / mass
    second_intensity = torch.einsum("hw,hw->", weights, intensity.square()) / mass
    std_intensity = (second_intensity - mean_intensity.square() + epsilon).clamp_min(epsilon).sqrt()
    descriptor = torch.cat(
        (
            mean_rgb,
            std_rgb,
            mean_intensity.unsqueeze(0),
            std_intensity.unsqueeze(0),
        )
    )
    return F.normalize(descriptor, dim=0, eps=epsilon).unsqueeze(0)


def _public_action_fields(
    episode: Mapping[str, object],
    frame_index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project the private target ledger into an ID- and slot-free command."""

    events = episode["events"]
    labels = episode["labels"]
    if not isinstance(events, Mapping) or not isinstance(labels, Mapping):
        raise TypeError("materialized action projection requires event and label mappings")
    private_target = events["known_action_observed"][frame_index]
    if not isinstance(private_target, Tensor):
        raise TypeError("private known-action target ledger must be a tensor")
    target_count = int(private_target.sum())
    if target_count == 0:
        return (
            torch.zeros(1, dtype=torch.bool),
            torch.full((1,), -1.0, dtype=torch.float32),
            torch.zeros(1, 8, dtype=torch.float32),
            torch.zeros(1, 3, dtype=torch.float32),
        )
    if target_count != 1 or frame_index == 0:
        raise ValueError("one public action must address one previously observed object")
    private_slot = int(torch.nonzero(private_target, as_tuple=False).item())
    previous_mask = labels["segmentation_mask"][frame_index - 1, private_slot]
    image = episode["rgb"][frame_index - 1]
    handle = _observable_action_appearance(image, previous_mask).clone()
    timestamp = events["known_action_timestamp"][frame_index, private_slot].reshape(1).clone()
    impulse = events["known_impulse_world"][frame_index, private_slot].reshape(1, 3).clone()
    return torch.ones(1, dtype=torch.bool), timestamp, handle, impulse


def _update_boundary_hash(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _tensor_payload_hash(digest: hashlib._Hash, name: str, value: Tensor) -> None:
    _update_boundary_hash(digest, name.encode("utf-8"))
    _update_boundary_hash(digest, str(value.dtype).encode("ascii"))
    _update_boundary_hash(digest, str(value.device).encode("ascii"))
    _update_boundary_hash(digest, repr(tuple(value.shape)).encode("ascii"))
    _update_boundary_hash(digest, repr(tuple(value.stride())).encode("ascii"))
    if value.device.type == "meta":
        _update_boundary_hash(digest, b"meta:no-payload")
        return
    # Hash a private clone: NumPy export makes its backing storage
    # non-resizable, and boundary inspection must not mutate public ownership
    # metadata as a side effect.
    contiguous = value.detach().to(device="cpu").contiguous().clone()
    _update_boundary_hash(digest, contiguous.view(torch.uint8).numpy().tobytes())


def _tensor_storage_key(value: Tensor) -> tuple[str, int | None, int] | None:
    if value.device.type == "meta":
        return None
    storage = value.untyped_storage()
    # ``_cdata`` is used only for equality inside this process and is never
    # serialized. Unlike data_ptr(), it distinguishes independent empty CPU
    # storages while still identifying disjoint views of one backing store.
    return value.device.type, value.device.index, int(storage._cdata)


def _iter_truth_tensors(value: object, *, active: set[int] | None = None) -> Iterator[Tensor]:
    if type(value) is Tensor:
        yield value
        return
    if value is None or type(value) in {str, bytes, bool, int, float}:
        return
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("truth-only materialization tree contains a cycle")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("truth-only materialization mappings require exact string keys")
        active.add(identity)
        try:
            for key in sorted(value):
                yield from _iter_truth_tensors(value[key], active=active)
        finally:
            active.remove(identity)
        return
    if type(value) in {list, tuple}:
        active.add(identity)
        try:
            for item in value:
                yield from _iter_truth_tensors(item, active=active)
        finally:
            active.remove(identity)
        return
    if is_dataclass(value) and not isinstance(value, type):
        active.add(identity)
        try:
            for item in fields(value):
                yield from _iter_truth_tensors(getattr(value, item.name), active=active)
        finally:
            active.remove(identity)
        return
    raise TypeError(f"truth-only materialization tree contains unsupported {type(value).__name__}")


def _declared_class_field_names(value_type: type[object]) -> tuple[str, ...]:
    names: set[str] = set()
    if is_dataclass(value_type):
        names.update(item.name for item in fields(value_type))
    for owner in value_type.__mro__:
        slots = owner.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.update(str(name) for name in slots if name not in {"__dict__", "__weakref__"})
        annotations = owner.__dict__.get("__annotations__", {})
        if isinstance(annotations, dict):
            names.update(str(name) for name in annotations)
    return tuple(sorted(names))


def _make_public_boundary_evidence(
    *,
    row_sha256: str,
    truth_bound: bool,
    frame_count: int,
    exact_frame_type_count: int,
    exact_schema_frame_count: int,
    public_tensor_count: int,
    truth_tensor_count: int,
    unexpected_frame_type_count: int,
    subclass_frame_count: int,
    unexpected_field_count: int,
    missing_field_count: int,
    invalid_public_field_count: int,
    missing_truth_root_count: int,
    public_storage_alias_count: int,
    truth_storage_alias_count: int,
    public_payload_sha256: str,
) -> DynamicSetPublicBoundaryEvidence:
    unsigned: dict[str, object] = {
        "schema": _PUBLIC_BOUNDARY_SCHEMA,
        "row_sha256": row_sha256,
        "truth_bound": truth_bound,
        "frame_count": frame_count,
        "exact_frame_type_count": exact_frame_type_count,
        "exact_schema_frame_count": exact_schema_frame_count,
        "public_tensor_count": public_tensor_count,
        "truth_tensor_count": truth_tensor_count,
        "unexpected_frame_type_count": unexpected_frame_type_count,
        "subclass_frame_count": subclass_frame_count,
        "unexpected_field_count": unexpected_field_count,
        "missing_field_count": missing_field_count,
        "invalid_public_field_count": invalid_public_field_count,
        "missing_truth_root_count": missing_truth_root_count,
        "public_storage_alias_count": public_storage_alias_count,
        "truth_storage_alias_count": truth_storage_alias_count,
        "public_payload_sha256": public_payload_sha256,
    }
    return DynamicSetPublicBoundaryEvidence(
        **unsigned,
        boundary_sha256=canonical_sha256(unsigned),
    ).validate()


def inspect_dynamic_set_public_boundary(
    materialization: DynamicSetMaterialization | None,
    public_frames: Sequence[object],
) -> DynamicSetPublicBoundaryEvidence:
    """Inspect one public stream without exposing private values in evidence."""

    if materialization is not None and type(materialization) is not DynamicSetMaterialization:
        raise TypeError("boundary materialization must use exact DynamicSetMaterialization")
    frames_value = tuple(public_frames)
    payload_digest = hashlib.sha256()
    _update_boundary_hash(payload_digest, _PUBLIC_BOUNDARY_SCHEMA.encode("ascii"))

    truth_tensors: tuple[Tensor, ...] = ()
    missing_truth_roots = 0
    if materialization is not None:
        episode = materialization.episode
        if not isinstance(episode, Mapping):
            raise TypeError("boundary materialization episode must be a mapping")
        roots: list[object] = []
        for name in ("objects", "labels", "events"):
            root = episode.get(name)
            if not isinstance(root, Mapping):
                missing_truth_roots += 1
            else:
                roots.append(root)
        if not isinstance(materialization.known_action_observed, Tensor):
            missing_truth_roots += 1
        else:
            roots.append(materialization.known_action_observed)
        truth_tensors = tuple(tensor for root in roots for tensor in _iter_truth_tensors(root))
    truth_storage = {
        key for tensor in truth_tensors if (key := _tensor_storage_key(tensor)) is not None
    }

    exact_types = 0
    exact_schemas = 0
    public_tensor_count = 0
    unexpected_types = 0
    subclasses = 0
    unexpected_fields = 0
    missing_fields = 0
    invalid_fields = 0
    aliased_public_tensors = 0
    duplicate_public_storages = 0
    public_storage: set[tuple[str, int | None, int]] = set()
    expected_fields = set(_PUBLIC_FRAME_FIELD_NAMES)

    for sequence_index, frame in enumerate(frames_value):
        _update_boundary_hash(payload_digest, sequence_index.to_bytes(8, "big", signed=False))
        if type(frame) is not DynamicSetPublicFrame:
            unexpected_types += 1
            subclasses += int(isinstance(frame, DynamicSetPublicFrame))
            actual_fields = set(_declared_class_field_names(type(frame)))
            unexpected_fields += len(actual_fields - expected_fields)
            missing_fields += len(expected_fields - actual_fields)
            _update_boundary_hash(
                payload_digest,
                f"{type(frame).__module__}.{type(frame).__qualname__}".encode(),
            )
            _update_boundary_hash(payload_digest, repr(sorted(actual_fields)).encode("utf-8"))
            continue
        exact_types += 1
        actual_fields = set(_PUBLIC_FRAME_FIELD_NAMES)
        if tuple(item.name for item in fields(frame)) == _PUBLIC_FRAME_FIELD_NAMES:
            exact_schemas += 1
        else:
            invalid_fields += 1

        if type(frame.frame_index) is not int or frame.frame_index != sequence_index:
            invalid_fields += 1
        if (
            type(frame.timestamp) is not float
            or not math.isfinite(frame.timestamp)
            or not math.isclose(
                frame.timestamp,
                sequence_index / DYNAMIC_SET_FRAME_RATE_HZ,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            invalid_fields += 1
        _update_boundary_hash(payload_digest, repr(frame.frame_index).encode("ascii"))
        _update_boundary_hash(payload_digest, repr(frame.timestamp).encode("ascii"))
        for name, (expected_shape, expected_dtype) in _PUBLIC_TENSOR_SCHEMA.items():
            value = getattr(frame, name)
            if type(value) is not Tensor:
                invalid_fields += 1
                _update_boundary_hash(
                    payload_digest,
                    f"{name}:{type(value).__module__}.{type(value).__qualname__}".encode(),
                )
                continue
            public_tensor_count += 1
            valid_tensor = (
                tuple(value.shape) == expected_shape
                and value.dtype is expected_dtype
                and value.device.type == "cpu"
                and value.layout is torch.strided
                and not value.requires_grad
                and value.grad_fn is None
                and value._base is None
                and value.storage_offset() == 0
                and value.is_contiguous()
            )
            if value.device.type != "meta":
                storage = value.untyped_storage()
                valid_tensor = (
                    valid_tensor
                    and storage.nbytes() == value.numel() * value.element_size()
                    and storage.resizable()
                    and not storage.is_shared()
                )
            if expected_dtype is not torch.bool and value.device.type != "meta":
                valid_tensor = valid_tensor and bool(torch.isfinite(value).all())
            invalid_fields += int(not valid_tensor)
            try:
                _tensor_payload_hash(payload_digest, name, value)
            except (RuntimeError, TypeError, ValueError):
                invalid_fields += 1
                _update_boundary_hash(payload_digest, f"{name}:unhashable".encode("ascii"))
            storage_key = _tensor_storage_key(value)
            if storage_key is not None:
                duplicate_public_storages += int(storage_key in public_storage)
                public_storage.add(storage_key)
            aliased_public_tensors += int(storage_key is not None and storage_key in truth_storage)

    row_sha256 = (
        _UNBOUND_ROW_SHA256
        if materialization is None
        else canonical_sha256(asdict(materialization.row))
    )
    return _make_public_boundary_evidence(
        row_sha256=row_sha256,
        truth_bound=materialization is not None,
        frame_count=len(frames_value),
        exact_frame_type_count=exact_types,
        exact_schema_frame_count=exact_schemas,
        public_tensor_count=public_tensor_count,
        truth_tensor_count=len(truth_tensors),
        unexpected_frame_type_count=unexpected_types,
        subclass_frame_count=subclasses,
        unexpected_field_count=unexpected_fields,
        missing_field_count=missing_fields,
        invalid_public_field_count=invalid_fields,
        missing_truth_root_count=missing_truth_roots,
        public_storage_alias_count=duplicate_public_storages,
        truth_storage_alias_count=aliased_public_tensors,
        public_payload_sha256=payload_digest.hexdigest(),
    )


def certify_dynamic_set_public_boundary(
    materialization: DynamicSetMaterialization,
    public_frames: Sequence[object],
) -> DynamicSetPublicBoundaryEvidence:
    """Return row-bound boundary evidence or fail before public inference."""

    return inspect_dynamic_set_public_boundary(materialization, public_frames).require_clean(
        require_truth_binding=True
    )


def _materialize_dynamic_set_public_frames_with_boundary(
    materialization: DynamicSetMaterialization,
) -> tuple[tuple[DynamicSetPublicFrame, ...], DynamicSetPublicBoundaryEvidence]:
    if type(materialization) is not DynamicSetMaterialization:
        raise TypeError("materialization must be an exact DynamicSetMaterialization")
    episode = materialization.episode
    camera = episode["camera"]
    frames_value: list[DynamicSetPublicFrame] = []
    for frame_index in range(DYNAMIC_SET_FRAMES):
        observed, action_timestamp, action_handle, action_impulse = _public_action_fields(
            episode,
            frame_index,
        )
        frames_value.append(
            DynamicSetPublicFrame(
                frame_index=frame_index,
                timestamp=float(episode["timestamps"][frame_index]),
                rgb=episode["rgb"][frame_index].clone(),
                depth=episode["depth"][frame_index].clone(),
                world_from_camera=camera["world_from_camera"][frame_index].clone(),
                intrinsics=camera["intrinsics"][frame_index].clone(),
                known_action_observed=observed,
                known_action_timestamp=action_timestamp,
                known_action_appearance_handle=action_handle,
                known_impulse_world=action_impulse,
            )
        )
    frames_tuple = tuple(frames_value)
    return frames_tuple, certify_dynamic_set_public_boundary(materialization, frames_tuple)


def iter_dynamic_set_public_frames(
    materialization: DynamicSetMaterialization,
) -> Iterator[DynamicSetPublicFrame]:
    """Yield observable RGB-D, calibration, and declared actions only."""

    frames_value, _ = _materialize_dynamic_set_public_frames_with_boundary(materialization)
    yield from frames_value


__all__ = [
    "DEFAULT_MATERIALIZATION_ATTEMPTS",
    "DynamicSetMaterialization",
    "DynamicSetMaterializationError",
    "DynamicSetPublicBoundaryEvidence",
    "DynamicSetPublicFrame",
    "certify_dynamic_set_public_boundary",
    "inspect_dynamic_set_public_boundary",
    "iter_dynamic_set_public_frames",
    "materialize_dynamic_set_episode",
]
