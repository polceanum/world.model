"""Seedless formal scene family for known-action counterfactual planning.

This module owns immutable ordinal-addressed descriptions and independent
analytic truth only.  It never constructs a simulator episode and never calls
the public camera, renderer, physics, dynamics, runtime, or planner paths.
Formal candidate schedules may be converted into the inert public
WorldImpulseAction value, but their state consequences are computed here from
the closed-form linear-drag law.

The family crosses sixteen split-disjoint rational physical primitives with
two observable target-handle roles and eight accepted orbital-camera strata.
Every bundle exposes the same eight equal-energy single-impulse candidates in
two independently frozen display orders.  Role, palette, camera, and candidate
order are balanced independently of the winning action.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import struct
import sys
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from world_model.dynamics.actions import WorldImpulseAction

SceneSplit = Literal["development", "selector", "confirmation", "final_test"]
EvidenceRole = Literal[
    "governed_development",
    "protected_selector",
    "protected_confirmation",
    "protected_final_test",
]
CandidateOrder = Literal["canonical", "pi", "rho"]

SPLITS: tuple[SceneSplit, ...] = (
    "development",
    "selector",
    "confirmation",
    "final_test",
)
SPLIT_INDEX: dict[SceneSplit, int] = {split: index for index, split in enumerate(SPLITS)}
SPLIT_EVIDENCE_ROLE: dict[SceneSplit, EvidenceRole] = {
    "development": "governed_development",
    "selector": "protected_selector",
    "confirmation": "protected_confirmation",
    "final_test": "protected_final_test",
}

COEFFICIENT_LEVELS = (-7, -5, 5, 7)
COEFFICIENT_RANK = {value: rank for rank, value in enumerate(COEFFICIENT_LEVELS)}
SPLIT_PRIMITIVE_PAIRS: dict[SceneSplit, tuple[tuple[int, int], ...]] = {
    "development": ((-7, -7), (-5, 5), (5, -5), (7, 7)),
    "selector": ((-7, 7), (-5, -5), (5, 5), (7, -7)),
    "confirmation": ((-7, -5), (-5, 7), (5, -7), (7, 5)),
    "final_test": ((-7, 5), (-5, -7), (5, 7), (7, -5)),
}

PRIMITIVES_PER_SPLIT = 4
HANDLE_ROLES = 2
CAMERA_STRATA = 8
CANDIDATES_PER_BUNDLE = 8
BUNDLES_PER_SPLIT = PRIMITIVES_PER_SPLIT * HANDLE_ROLES * CAMERA_STRATA
TOTAL_PRIMITIVE_PROFILES = len(SPLITS) * PRIMITIVES_PER_SPLIT
TOTAL_BUNDLES = len(SPLITS) * BUNDLES_PER_SPLIT
TOTAL_CANDIDATES = TOTAL_BUNDLES * CANDIDATES_PER_BUNDLE

FRAME_COUNT = 56
FRAME_RATE_HZ = 20
PHYSICS_RATE_HZ = 120
SUBSTEPS_PER_FRAME = PHYSICS_RATE_HZ // FRAME_RATE_HZ
PHYSICAL_SUBSTEP_COUNT = (FRAME_COUNT - 1) * SUBSTEPS_PER_FRAME
HISTORY_FRAME_COUNT = 16
ANCHOR_FRAME_INDEX = 15
ANCHOR_SUBSTEP_INDEX = ANCHOR_FRAME_INDEX * SUBSTEPS_PER_FRAME
ANCHOR_TIME_SECONDS = ANCHOR_FRAME_INDEX / FRAME_RATE_HZ
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
QUERY_FRAME_INDICES = (17, 20, 25, 35, 55)

POSITION_DENOMINATOR = 1000
VELOCITY_DENOMINATOR = 4000
IMPULSE_DENOMINATOR = 4000
FIXED_DRAG_NUMERATOR = 1
FIXED_DRAG_DENOMINATOR = 20
FIXED_RADIUS_NUMERATOR = 21
FIXED_RADIUS_DENOMINATOR = 100
FIXED_MASS_NUMERATOR = 1
FIXED_MASS_DENOMINATOR = 1

IMAGE_SIZE = (64, 64)
WORLD_BOUNDS = ((-2.25, 2.25), (0.0, 3.25), (-1.5, 1.5))
CAMERA_PHASES_RADIANS = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
CAMERA_DIRECTIONS = (-1, 1)
CAMERA_TARGET = (0.0, 0.95, 0.0)
CAMERA_RADIUS_M = 4.6
CAMERA_HEIGHT_M = 2.15
CAMERA_ANGULAR_SPEED_RAD_S = 0.24
CAMERA_VERTICAL_FOV_DEGREES = 48.0
PALETTE = ((0.92, 0.20, 0.14), (0.14, 0.84, 0.30))
HANDLE_PROTOTYPES = PALETTE

# Source-time safety gates inherited from the accepted fully-visible orbital
# rung.  They certify every candidate future, not only the winning future.
MINIMUM_FULL_SUPPORT_PIXELS = 20
MINIMUM_CONIC_GAP_LOWER_BOUND_PIXELS = 4.0
MINIMUM_CONIC_BOUNDARY_CLEARANCE_PIXELS = 6.0
MINIMUM_WORLD_SURFACE_GAP_M = 1.0
MINIMUM_WORLD_BOUNDARY_M = 0.15
MAXIMUM_CAMERA_CALIBRATION_ERROR = 2.0e-5
MINIMUM_CAMERA_STEP_ANGLE_RADIANS = 0.01198
MAXIMUM_CAMERA_STEP_ANGLE_RADIANS = 0.01202
MINIMUM_CAMERA_TRANSLATION_STEP_M = 0.0551
MAXIMUM_CAMERA_TRANSLATION_STEP_M = 0.0553
MAXIMUM_PROJECTED_CENTRE_STEP_PIXELS = 0.13
MAXIMUM_ACTED_PROJECTED_CENTRE_STEP_PIXELS = 0.15
MINIMUM_ACTION_AFTER_ANCHOR_SECONDS = 0.30
MINIMUM_ACTION_QUERY_SEPARATION_SECONDS = 0.05
MINIMUM_MATCHED_HANDLE_COSINE = 0.999999
MINIMUM_CROSS_HANDLE_COSINE_DISTANCE = 0.25
BOUNDARY_CONTACT_TOLERANCE_M = 1.0e-4
FROZEN_SIMULATOR_SLEEP_SPEED_MPS = 0.035
FROZEN_SIMULATOR_SLEEP_AFTER_SECONDS = 0.35
FROZEN_ORPHEUS_CONFIG_SLEEP_SPEED_MPS = 0.05
CONIC_PIXEL_SAFETY_TOLERANCE = 1.0e-7
CONIC_RELATIVE_ALGEBRA_TOLERANCE = 1.0e-12
CONIC_RELATIVE_BOUNDARY_RESIDUAL_TOLERANCE = 1.0e-12

PUBLIC_CAMERA_SOURCE_SHA256 = "23c9798d412a44e9f8b7bea57ef7598e469dfeea087e1515a6c27d51e53caa27"
PUBLIC_PHYSICS_SOURCE_SHA256 = "99a69c80ef87ce15a783a43b1342112600431a6b33d0aa95dacaac148202c02f"
PUBLIC_RENDERER_SOURCE_SHA256 = "76ae74a9c0da3f002b4e2b2234228f5dff8c1117965721bf2328df658e548876"
PUBLIC_COLLISIONS_SOURCE_SHA256 = "187c76ecf5c8d082a523c2c32fc6b64eecca52c005694a9040f74756684dc9e8"
PUBLIC_CONFIG_SOURCE_SHA256 = "bcc271e070143c53a320904c3832177a3bc5f9aee0105ab62a2031846f01d70e"
PUBLIC_DEFAULT_CONFIG_SHA256 = "25b1da2c16d768af846d1cbbd5953135f504102e8fc3ae27534cd75f42a8329c"
ACCEPTED_ORBITAL_SOURCE_SHA256 = "02e75b325bdf7bad310f8973a786a396b8762104261702b299a9f8103748e569"
ACCEPTED_ORBITAL_CERTIFICATE_SHA256 = (
    "7832ddb49081292d0f50a5eb63edb38fefb49d136d7e1757c73d9c658e42a36f"
)

FROZEN_TORCH_VERSION = "2.9.0a0+gitcbe1a35"
FROZEN_PYTHON_VERSION = "3.10.20"
FROZEN_PLATFORM_SYSTEM = "Darwin"
FROZEN_PLATFORM_MACHINE = "x86_64"
FROZEN_BYTEORDER = "little"

# BEGIN GENERATED CERTIFICATE FREEZE
# Filled only after the repaired exhaustive pure tests and explicit hash pass.
FROZEN_NORMALIZED_SCENE_SOURCE_SHA256 = (
    "51133013d7f89dfe2876f815fa9781a3e10a866c2b4a3da2b3826b4b6d904f64"
)
FROZEN_INPUT_BINDING_SHA256 = "c43269c3972b58e66fc003e5005bbba9fbdb292b196de9ba7331a6bc696ad2dd"
FROZEN_MANIFEST_SHA256: dict[str, str] = {
    "development": "4d1637c681229e117a2228a059649d220ee1995dd4b21bc4b6b3715e88828b1d",
    "selector": "173b3d99bcac14d23c2f6c1c857b81d12d95c2c90bc2eaf73dd9b58e95f121b3",
    "confirmation": "661af7a7f921751ce54f1286df65cd5e706b15560220c5d8d11f9700fe7ff232",
    "final_test": "aba3cdb33b9a995a3dfc4f4f1a4630b0ad96b2dafc4f551be196abbae7a721e9",
}
FROZEN_TRACE_SHA256: dict[str, str] = {
    "metadata": "8e9a7f83ebe2c2c162a18c5551bc78ab9ed29998af47c95d757aa97f8d052133",
    "prefix": "ce7f374582fc29170ec188cb06f9e2c7dfd6f27de326b3c5af90248b83d0b78a",
    "camera": "43c6ec1857c49c82baa9a8e187428c4722a0a6ecf9de3283753a2c67d13be164",
    "ordered_schedules": "47b63cec32797f3bfd246334f1a703364e319dfbe87c9ae0e5d1121e55e75451",
    "unordered_schedules": "373fb69de944801507f9d7ab51df7202323bfd81c4207bb7318e8f5b874e7b46",
    "candidate_state": "039c9d1fac8a8cf33fbb7cc0a2805bcb72c547a91a4ba7d211cee07f52fdbf03",
    "substep_contact_proof": "954cf36e47b211c4f79f7c2c6cf902d1ead186f6540143e4822fedbd4ccafd35",
    "goals_costs": "dc66e141ad8814c75b6a7a3a7b69f9376a287bd7fb039758950b7452d8b1579a",
    "conic_raster": "0c0f24f6ff7e1bb3c974a65018bb8602b42494b71c66c41ad0e2a6ac56a6a902",
    "combined": "eae5b983a204eca463be569670c9dec2772a50cc7d2e1f3137e59eaf2a1e4157",
}
FROZEN_DESCRIPTOR_SHA256 = "26ece5e0e3e709a10dc7d0f59c2ab28acb8ccd58b8138450c637cad4db609f75"
FROZEN_CERTIFICATE_SHA256 = "6f3f566a6ca76b7c417ead6452b9ec5e8bc8ec1b69e5085199035aefc836a3d2"
# END GENERATED CERTIFICATE FREEZE


@dataclass(frozen=True, slots=True)
class KnownActionSceneSpecification:
    """One immutable rational bundle descriptor."""

    split: SceneSplit
    ordinal: int
    split_index: int
    evidence_role: EvidenceRole
    primitive_index: int
    handle_role: int
    camera_stratum: int
    phase_index: int
    direction_index: int
    direction: int
    a: int
    b: int
    position_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    velocity_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    palette_swapped: bool
    albedo: tuple[tuple[float, float, float], tuple[float, float, float]]
    physical_controlled_row: int
    action_delay_numerator: int
    impulse_magnitude_numerators: tuple[int, int, int]
    optimal_canonical_index: int

    def position_tensor(self) -> Tensor:
        return torch.tensor(self.position_numerators, dtype=torch.float32) / POSITION_DENOMINATOR

    def velocity_tensor(self) -> Tensor:
        return torch.tensor(self.velocity_numerators, dtype=torch.float32) / VELOCITY_DENOMINATOR

    def albedo_tensor(self) -> Tensor:
        return torch.tensor(self.albedo, dtype=torch.float32)

    @property
    def action_delay_seconds(self) -> float:
        return self.action_delay_numerator / FRAME_RATE_HZ

    @property
    def action_timestamp_seconds(self) -> float:
        return ANCHOR_TIME_SECONDS + self.action_delay_seconds


@dataclass(frozen=True, slots=True)
class KnownActionCandidateSpecification:
    """One canonical single-impulse schedule."""

    canonical_index: int
    timestamp_numerator: int
    timestamp_denominator: int
    observable_handle_role: int
    observable_handle_prototype: tuple[float, float, float]
    physical_target_row: int
    impulse_numerators: tuple[int, int, int]
    impulse_denominator: int

    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp_numerator / self.timestamp_denominator

    def impulse_tensor(self, *, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.tensor(self.impulse_numerators, dtype=dtype) / self.impulse_denominator


@dataclass(frozen=True, slots=True)
class PureActionTrajectory:
    """Independent right-continuous fixed-drag trajectory."""

    positions: Tensor
    velocities: Tensor
    substep_positions: Tensor
    substep_velocities: Tensor
    action_events: Tensor
    substep_action_events: Tensor


@dataclass(frozen=True, slots=True)
class PureCameraFrame:
    """Independent accepted orbital calibration value."""

    timestamp: float
    position: Tensor
    target: Tensor
    world_from_camera: Tensor
    camera_from_world: Tensor
    intrinsics: Tensor


def _normalise_split(split: str) -> SceneSplit:
    if type(split) is not str:
        raise TypeError("known-action split must be a string")
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown known-action split {split!r}")
    return split  # type: ignore[return-value]


def _ordinal_components(ordinal: int) -> tuple[int, int, int]:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("known-action bundle ordinal must be an integer")
    if not 0 <= ordinal < BUNDLES_PER_SPLIT:
        raise IndexError(ordinal)
    primitive_index, remainder = divmod(ordinal, HANDLE_ROLES * CAMERA_STRATA)
    handle_role, camera_stratum = divmod(remainder, CAMERA_STRATA)
    return primitive_index, handle_role, camera_stratum


def _rational_geometry(
    a: int,
    b: int,
) -> tuple[
    tuple[tuple[int, int, int], tuple[int, int, int]],
    tuple[tuple[int, int, int], tuple[int, int, int]],
]:
    if a not in COEFFICIENT_RANK or b not in COEFFICIENT_RANK:
        raise ValueError("physical coefficients must belong to the frozen level set")
    position = (
        (-450 + 6 * a, 400 + 2 * b, -300 + 4 * b),
        (450 + 5 * b, 1750 + 2 * a, 300 - 4 * a),
    )
    velocity = (
        (180 + 2 * a, 48 + b, 16 + a),
        (-172 + 2 * b, -32 + a, -12 + b),
    )
    return position, velocity


def scene_specification(split: str, ordinal: int) -> KnownActionSceneSpecification:
    """Resolve one bundle without a seed, RNG, artifact, or public runtime."""

    canonical_split = _normalise_split(split)
    primitive_index, handle_role, camera_stratum = _ordinal_components(ordinal)
    phase_index, direction_index = divmod(camera_stratum, 2)
    direction = CAMERA_DIRECTIONS[direction_index]
    a, b = SPLIT_PRIMITIVE_PAIRS[canonical_split][primitive_index]
    position, velocity = _rational_geometry(a, b)
    palette_swapped = bool((primitive_index + phase_index + direction_index) & 1)
    albedo = PALETTE[::-1] if palette_swapped else PALETTE
    physical_controlled_row = handle_role ^ int(palette_swapped)
    rank_a = COEFFICIENT_RANK[a]
    rank_b = COEFFICIENT_RANK[b]
    impulse_magnitudes = (
        21 + rank_b,
        19 + rank_a,
        23 + ((rank_a + rank_b) % 4),
    )
    return KnownActionSceneSpecification(
        split=canonical_split,
        ordinal=ordinal,
        split_index=SPLIT_INDEX[canonical_split],
        evidence_role=SPLIT_EVIDENCE_ROLE[canonical_split],
        primitive_index=primitive_index,
        handle_role=handle_role,
        camera_stratum=camera_stratum,
        phase_index=phase_index,
        direction_index=direction_index,
        direction=direction,
        a=a,
        b=b,
        position_numerators=position,
        velocity_numerators=velocity,
        palette_swapped=palette_swapped,
        albedo=albedo,
        physical_controlled_row=physical_controlled_row,
        action_delay_numerator=6 + rank_a,
        impulse_magnitude_numerators=impulse_magnitudes,
        optimal_canonical_index=(2 * primitive_index + handle_role + camera_stratum) % 8,
    )


def role_twin_ordinal(ordinal: int) -> int:
    """Toggle only the observable task handle role."""

    _ordinal_components(ordinal)
    return ordinal ^ CAMERA_STRATA


def _canonical_index(value: int, *, label: str = "canonical candidate index") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 0 <= value < CANDIDATES_PER_BUNDLE:
        raise IndexError(value)
    return value


def candidate_specification(
    specification: KnownActionSceneSpecification,
    canonical_index: int,
) -> KnownActionCandidateSpecification:
    """Return one canonical right-continuous world impulse."""

    if not isinstance(specification, KnownActionSceneSpecification):
        raise TypeError("candidate_specification requires a known-action scene")
    q = _canonical_index(canonical_index)
    signs = tuple(1 if q & (1 << axis) else -1 for axis in range(3))
    impulse = tuple(
        sign * magnitude
        for sign, magnitude in zip(
            signs,
            specification.impulse_magnitude_numerators,
            strict=True,
        )
    )
    return KnownActionCandidateSpecification(
        canonical_index=q,
        timestamp_numerator=ANCHOR_FRAME_INDEX + specification.action_delay_numerator,
        timestamp_denominator=FRAME_RATE_HZ,
        observable_handle_role=specification.handle_role,
        observable_handle_prototype=HANDLE_PROTOTYPES[specification.handle_role],
        physical_target_row=specification.physical_controlled_row,
        impulse_numerators=impulse,  # type: ignore[arg-type]
        impulse_denominator=IMPULSE_DENOMINATOR,
    )


def display_to_canonical(
    camera_stratum: int,
    display_index: int,
    *,
    order: CandidateOrder = "pi",
) -> int:
    """Resolve a display index under one frozen role-independent order."""

    if isinstance(camera_stratum, bool) or not isinstance(camera_stratum, int):
        raise TypeError("camera stratum must be an integer")
    if not 0 <= camera_stratum < CAMERA_STRATA:
        raise IndexError(camera_stratum)
    d = _canonical_index(display_index, label="display candidate index")
    if order == "canonical":
        return d
    if order == "pi":
        return (d + 2 * camera_stratum) % 8
    if order == "rho":
        return (5 * d + 3 + 2 * camera_stratum) % 8
    raise ValueError(f"unknown candidate order {order!r}")


def canonical_to_display(
    camera_stratum: int,
    canonical_index: int,
    *,
    order: CandidateOrder = "pi",
) -> int:
    """Invert a frozen display permutation exactly."""

    q = _canonical_index(canonical_index)
    if order == "canonical":
        return q
    if order == "pi":
        return (q - 2 * camera_stratum) % 8
    if order == "rho":
        return (5 * (q - 3 - 2 * camera_stratum)) % 8
    raise ValueError(f"unknown candidate order {order!r}")


def candidate_order(
    camera_stratum: int,
    *,
    order: CandidateOrder = "pi",
) -> tuple[int, ...]:
    return tuple(
        display_to_canonical(camera_stratum, display, order=order)
        for display in range(CANDIDATES_PER_BUNDLE)
    )


def world_impulse_action(
    specification: KnownActionSceneSpecification,
    display_index: int,
    *,
    resolved_persistent_object_id: Tensor,
    order: CandidateOrder = "pi",
    dtype: torch.dtype = torch.float32,
) -> WorldImpulseAction:
    """Materialize an action only after caller-owned appearance resolution.

    The formal handle role, palette slot, physical row, and evaluator truth are
    never interpreted as a persistent identity.  The caller must supply a
    separately resolved int64 ID tensor, normally produced by matching the
    observable handle prototype to the public belief appearance.
    """

    if not isinstance(resolved_persistent_object_id, Tensor):
        raise TypeError("resolved_persistent_object_id must be a torch.Tensor")
    if resolved_persistent_object_id.ndim != 1 or resolved_persistent_object_id.numel() < 1:
        raise ValueError("resolved_persistent_object_id must have nonempty shape [B]")
    if resolved_persistent_object_id.dtype != torch.int64:
        raise TypeError("resolved_persistent_object_id must have dtype torch.int64")
    if bool((resolved_persistent_object_id < 0).any()):
        raise ValueError("resolved_persistent_object_id must contain persistent nonnegative IDs")
    q = display_to_canonical(specification.camera_stratum, display_index, order=order)
    candidate = candidate_specification(specification, q)
    batch_size = resolved_persistent_object_id.shape[0]
    resolved_device = resolved_persistent_object_id.device
    return WorldImpulseAction(
        timestamp=torch.full(
            (batch_size,),
            candidate.timestamp_seconds,
            dtype=dtype,
            device=resolved_device,
        ),
        object_id=resolved_persistent_object_id.clone(),
        impulse_world=(
            torch.tensor(
                candidate.impulse_numerators,
                dtype=dtype,
                device=resolved_device,
            )
            .expand(batch_size, -1)
            .clone()
            / candidate.impulse_denominator
        ),
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def candidate_metadata(candidate: KnownActionCandidateSpecification) -> dict[str, Any]:
    if not isinstance(candidate, KnownActionCandidateSpecification):
        raise TypeError("candidate_metadata requires a candidate specification")
    return {
        "canonical_index": candidate.canonical_index,
        "timestamp_rational_seconds": {
            "numerator": candidate.timestamp_numerator,
            "denominator": candidate.timestamp_denominator,
        },
        "observable_handle": {
            "role": candidate.observable_handle_role,
            "prototype": candidate.observable_handle_prototype,
        },
        "evaluator_only": {
            "physical_target_row": candidate.physical_target_row,
        },
        "impulse_world_rational": {
            "numerators": candidate.impulse_numerators,
            "denominator": candidate.impulse_denominator,
            "units": "belief_mass_unit_m_per_s",
        },
        "frame": "world",
        "right_continuous": True,
    }


def scene_metadata(specification: KnownActionSceneSpecification) -> dict[str, Any]:
    if not isinstance(specification, KnownActionSceneSpecification):
        raise TypeError("scene_metadata requires a known-action scene")
    return {
        "split": specification.split,
        "ordinal": specification.ordinal,
        "split_index": specification.split_index,
        "evidence_role": specification.evidence_role,
        "primitive_index": specification.primitive_index,
        "observable_handle": {
            "role": specification.handle_role,
            "prototype": HANDLE_PROTOTYPES[specification.handle_role],
        },
        "camera_stratum": specification.camera_stratum,
        "phase_index": specification.phase_index,
        "direction_index": specification.direction_index,
        "direction": specification.direction,
        "a": specification.a,
        "b": specification.b,
        "position_rational": {
            "numerators": specification.position_numerators,
            "denominator": POSITION_DENOMINATOR,
        },
        "velocity_rational": {
            "numerators": specification.velocity_numerators,
            "denominator": VELOCITY_DENOMINATOR,
        },
        "radius_rational": {
            "numerator": FIXED_RADIUS_NUMERATOR,
            "denominator": FIXED_RADIUS_DENOMINATOR,
        },
        "mass_rational": {
            "numerator": FIXED_MASS_NUMERATOR,
            "denominator": FIXED_MASS_DENOMINATOR,
        },
        "drag_rational": {
            "numerator": FIXED_DRAG_NUMERATOR,
            "denominator": FIXED_DRAG_DENOMINATOR,
        },
        "palette_swapped": specification.palette_swapped,
        "albedo": specification.albedo,
        "observable_handle_prototypes": HANDLE_PROTOTYPES,
        "action_delay_rational_seconds": {
            "numerator": specification.action_delay_numerator,
            "denominator": FRAME_RATE_HZ,
        },
        "impulse_magnitude_rational": {
            "numerators": specification.impulse_magnitude_numerators,
            "denominator": IMPULSE_DENOMINATOR,
        },
        "evaluator_only": {
            "physical_controlled_row": specification.physical_controlled_row,
            "optimal_canonical_index": specification.optimal_canonical_index,
            "opposite_goal_canonical_index": specification.optimal_canonical_index ^ 7,
        },
        "pi_display_to_canonical": candidate_order(
            specification.camera_stratum,
            order="pi",
        ),
        "rho_display_to_canonical": candidate_order(
            specification.camera_stratum,
            order="rho",
        ),
    }


def scene_signature(specification: KnownActionSceneSpecification) -> str:
    return canonical_sha256(scene_metadata(specification))


def split_manifest(split: str) -> tuple[dict[str, Any], ...]:
    canonical_split = _normalise_split(split)
    return tuple(
        scene_metadata(scene_specification(canonical_split, ordinal))
        for ordinal in range(BUNDLES_PER_SPLIT)
    )


def split_scene_signatures(split: str) -> tuple[str, ...]:
    return tuple(canonical_sha256(row) for row in split_manifest(split))


def family_scene_signature() -> str:
    return canonical_sha256({split: split_scene_signatures(split) for split in SPLITS})


@lru_cache(maxsize=16)
def _prefix_for_profile(split: SceneSplit, primitive_index: int) -> PureActionTrajectory:
    specification = scene_specification(split, primitive_index * 16)
    position = specification.position_tensor()
    velocity = specification.velocity_tensor()
    drag = torch.full((2, 1), FIXED_DRAG_NUMERATOR / FIXED_DRAG_DENOMINATOR)
    dt = 1.0 / PHYSICS_RATE_HZ
    decay = torch.exp(-drag * dt)
    displacement = -torch.expm1(-drag * dt) / drag
    substep_positions = [position.clone()]
    substep_velocities = [velocity.clone()]
    for _ in range(PHYSICAL_SUBSTEP_COUNT):
        position = position + velocity * displacement
        velocity = velocity * decay
        substep_positions.append(position.clone())
        substep_velocities.append(velocity.clone())
    positions = torch.stack(substep_positions)
    velocities = torch.stack(substep_velocities)
    # Frames 0..15 are the byte-exact accepted recurrence.  All futures share
    # one analytic source state at frame 15, matching the public state-to-
    # rollout boundary and making the no-action control exactly identical to
    # every candidate before its impulse.
    relative_times = (
        torch.arange(
            PHYSICAL_SUBSTEP_COUNT - ANCHOR_SUBSTEP_INDEX + 1,
            dtype=torch.float32,
        )
        / PHYSICS_RATE_HZ
    )
    future_decay = torch.exp(-(FIXED_DRAG_NUMERATOR / FIXED_DRAG_DENOMINATOR) * relative_times)
    future_displacement = -torch.expm1(
        -(FIXED_DRAG_NUMERATOR / FIXED_DRAG_DENOMINATOR) * relative_times
    ) / (FIXED_DRAG_NUMERATOR / FIXED_DRAG_DENOMINATOR)
    anchor_position = positions[ANCHOR_SUBSTEP_INDEX].clone()
    anchor_velocity = velocities[ANCHOR_SUBSTEP_INDEX].clone()
    positions[ANCHOR_SUBSTEP_INDEX:] = (
        anchor_position[None] + anchor_velocity[None] * future_displacement[:, None, None]
    )
    velocities[ANCHOR_SUBSTEP_INDEX:] = anchor_velocity[None] * future_decay[:, None, None]
    frame_indices = torch.arange(0, PHYSICAL_SUBSTEP_COUNT + 1, SUBSTEPS_PER_FRAME)
    no_events = torch.zeros((PHYSICAL_SUBSTEP_COUNT + 1, 2), dtype=torch.bool)
    return PureActionTrajectory(
        positions=positions[frame_indices],
        velocities=velocities[frame_indices],
        substep_positions=positions,
        substep_velocities=velocities,
        action_events=no_events[frame_indices],
        substep_action_events=no_events,
    )


def manual_prefix_trajectory(
    specification: KnownActionSceneSpecification,
) -> PureActionTrajectory:
    """Return a defensive copy of the accepted no-action float32 recurrence."""

    if not isinstance(specification, KnownActionSceneSpecification):
        raise TypeError("manual_prefix_trajectory requires a known-action scene")
    value = _prefix_for_profile(specification.split, specification.primitive_index)
    return PureActionTrajectory(
        positions=value.positions.clone(),
        velocities=value.velocities.clone(),
        substep_positions=value.substep_positions.clone(),
        substep_velocities=value.substep_velocities.clone(),
        action_events=value.action_events.clone(),
        substep_action_events=value.substep_action_events.clone(),
    )


def manual_candidate_trajectory(
    specification: KnownActionSceneSpecification,
    canonical_index: int,
) -> PureActionTrajectory:
    """Evaluate one exact single-jump candidate without public dynamics."""

    if not isinstance(specification, KnownActionSceneSpecification):
        raise TypeError("manual_candidate_trajectory requires a known-action scene")
    candidate = candidate_specification(specification, canonical_index)
    prefix = _prefix_for_profile(specification.split, specification.primitive_index)
    positions = prefix.substep_positions.clone()
    velocities = prefix.substep_velocities.clone()
    relative_times = (
        torch.arange(
            PHYSICAL_SUBSTEP_COUNT - ANCHOR_SUBSTEP_INDEX + 1,
            dtype=torch.float32,
        )
        / PHYSICS_RATE_HZ
    )
    drag = FIXED_DRAG_NUMERATOR / FIXED_DRAG_DENOMINATOR
    decay = torch.exp(-drag * relative_times)
    displacement = -torch.expm1(-drag * relative_times) / drag
    anchor_position = prefix.substep_positions[ANCHOR_SUBSTEP_INDEX]
    anchor_velocity = prefix.substep_velocities[ANCHOR_SUBSTEP_INDEX]
    future_position = anchor_position[None] + anchor_velocity[None] * displacement[:, None, None]
    future_velocity = anchor_velocity[None] * decay[:, None, None]
    action_delay = specification.action_delay_seconds
    after_action = relative_times >= action_delay
    elapsed = (relative_times - action_delay).clamp_min(0.0)
    action_decay = torch.exp(-drag * elapsed)
    action_displacement = -torch.expm1(-drag * elapsed) / drag
    impulse = candidate.impulse_tensor()[None, :]
    target_mask = torch.zeros((2, 1), dtype=torch.float32)
    target_mask[candidate.physical_target_row] = 1.0
    delta_velocity = action_decay[:, None, None] * target_mask[None] * impulse[None]
    delta_position = action_displacement[:, None, None] * target_mask[None] * impulse[None]
    future_position = future_position + torch.where(
        after_action[:, None, None],
        delta_position,
        torch.zeros_like(delta_position),
    )
    future_velocity = future_velocity + torch.where(
        after_action[:, None, None],
        delta_velocity,
        torch.zeros_like(delta_velocity),
    )
    positions[ANCHOR_SUBSTEP_INDEX:] = future_position
    velocities[ANCHOR_SUBSTEP_INDEX:] = future_velocity
    event_index = ANCHOR_SUBSTEP_INDEX + specification.action_delay_numerator * SUBSTEPS_PER_FRAME
    events = torch.zeros((PHYSICAL_SUBSTEP_COUNT + 1, 2), dtype=torch.bool)
    events[event_index, candidate.physical_target_row] = True
    frame_indices = torch.arange(0, PHYSICAL_SUBSTEP_COUNT + 1, SUBSTEPS_PER_FRAME)
    return PureActionTrajectory(
        positions=positions[frame_indices],
        velocities=velocities[frame_indices],
        substep_positions=positions,
        substep_velocities=velocities,
        action_events=events[frame_indices],
        substep_action_events=events,
    )


def terminal_goal(
    specification: KnownActionSceneSpecification,
    *,
    opposite: bool = False,
) -> Tensor:
    """Return the exact controlled-object terminal position target."""

    q = specification.optimal_canonical_index
    if opposite:
        q ^= 7
    trajectory = manual_candidate_trajectory(specification, q)
    return trajectory.positions[-1, specification.physical_controlled_row].clone()


def candidate_costs(
    specification: KnownActionSceneSpecification,
    *,
    order: CandidateOrder = "canonical",
    opposite_goal: bool = False,
) -> Tensor:
    """Squared terminal-position costs in one declared display order."""

    goal = terminal_goal(specification, opposite=opposite_goal)
    costs = []
    for display in range(CANDIDATES_PER_BUNDLE):
        q = display_to_canonical(specification.camera_stratum, display, order=order)
        terminal = manual_candidate_trajectory(specification, q).positions[
            -1, specification.physical_controlled_row
        ]
        costs.append((terminal - goal).square().sum())
    return torch.stack(costs)


def _make_intrinsics() -> Tensor:
    height, width = IMAGE_SIZE
    focal = (0.5 * height) / math.tan(0.5 * math.radians(CAMERA_VERTICAL_FOV_DEGREES))
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = 0.5 * (width - 1)
    intrinsics[1, 2] = 0.5 * (height - 1)
    return intrinsics


def pure_orbital_camera_frame(camera_stratum: int, timestamp: float) -> PureCameraFrame:
    """Evaluate the accepted orbit with local float32 matrix algebra."""

    if isinstance(camera_stratum, bool) or not isinstance(camera_stratum, int):
        raise TypeError("camera stratum must be an integer")
    if not 0 <= camera_stratum < CAMERA_STRATA:
        raise IndexError(camera_stratum)
    if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
        raise TypeError("camera timestamp must be a real scalar")
    resolved_timestamp = float(timestamp)
    if not math.isfinite(resolved_timestamp) or resolved_timestamp < 0.0:
        raise ValueError("camera timestamp must be finite and nonnegative")
    phase_index, direction_index = divmod(camera_stratum, 2)
    theta = (
        CAMERA_PHASES_RADIANS[phase_index]
        + CAMERA_DIRECTIONS[direction_index] * CAMERA_ANGULAR_SPEED_RAD_S * resolved_timestamp
    )
    position = torch.tensor(
        (
            CAMERA_RADIUS_M * math.sin(theta),
            CAMERA_HEIGHT_M,
            CAMERA_RADIUS_M * math.cos(theta),
        ),
        dtype=torch.float32,
    )
    target = torch.tensor(CAMERA_TARGET, dtype=torch.float32)
    forward = target - position
    forward = forward / torch.linalg.vector_norm(forward).clamp_min(1.0e-12)
    world_up = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32)
    right = torch.linalg.cross(forward, world_up)
    right = right / torch.linalg.vector_norm(right).clamp_min(1.0e-12)
    down = torch.linalg.cross(forward, right)
    down = down / torch.linalg.vector_norm(down).clamp_min(1.0e-12)
    world_from_camera = torch.eye(4, dtype=torch.float32)
    world_from_camera[:3, :3] = torch.stack((right, down, forward), dim=-1)
    world_from_camera[:3, 3] = position
    rotation = world_from_camera[:3, :3]
    camera_from_world = torch.eye(4, dtype=torch.float32)
    camera_from_world[:3, :3] = rotation.T
    camera_from_world[:3, 3] = -(rotation.T @ position)
    return PureCameraFrame(
        timestamp=resolved_timestamp,
        position=position,
        target=target,
        world_from_camera=world_from_camera,
        camera_from_world=camera_from_world,
        intrinsics=_make_intrinsics(),
    )


def _digest_field(digest: Any, label: str, payload: bytes) -> None:
    """Frame one field with canonical little-endian lengths."""

    label_bytes = label.encode("utf-8")
    digest.update(b"orpheus-known-action-v1\x00")
    digest.update(struct.pack("<I", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _update_tensor_digest(digest: Any, label: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    metadata = {
        "byte_order": "little",
        "dtype": str(value.dtype).removeprefix("torch."),
        "order": "C",
        "shape": list(value.shape),
    }
    _digest_field(digest, f"{label}/metadata", _canonical_json_bytes(metadata))
    array = value.numpy()
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    _digest_field(digest, f"{label}/data", little_endian.tobytes(order="C"))


def _tensor_sequence_sha256(values: tuple[tuple[str, Tensor], ...]) -> str:
    digest = hashlib.sha256()
    for label, value in values:
        _update_tensor_digest(digest, label, value)
    return digest.hexdigest()


def _stack_camera_frames(camera_stratum: int) -> dict[str, Tensor]:
    frames = tuple(
        pure_orbital_camera_frame(camera_stratum, frame / FRAME_RATE_HZ)
        for frame in range(FRAME_COUNT)
    )
    return {
        "position": torch.stack([frame.position for frame in frames]),
        "target": torch.stack([frame.target for frame in frames]),
        "world_from_camera": torch.stack([frame.world_from_camera for frame in frames]),
        "camera_from_world": torch.stack([frame.camera_from_world for frame in frames]),
        "intrinsics": torch.stack([frame.intrinsics for frame in frames]),
    }


def action_schedule_metadata(
    candidate: KnownActionCandidateSpecification,
) -> dict[str, Any]:
    """Return one formal schedule without inventing a persistent identity."""

    if not isinstance(candidate, KnownActionCandidateSpecification):
        raise TypeError("action_schedule_metadata requires a candidate specification")
    return {
        "canonical_index": candidate.canonical_index,
        "timestamp_rational_seconds": {
            "numerator": candidate.timestamp_numerator,
            "denominator": candidate.timestamp_denominator,
        },
        "observable_handle": {
            "role": candidate.observable_handle_role,
            "prototype": candidate.observable_handle_prototype,
        },
        "impulse_world_rational": {
            "numerators": candidate.impulse_numerators,
            "denominator": candidate.impulse_denominator,
            "units": "belief_mass_unit_m_per_s",
        },
        "frame": "world",
    }


def observable_task_metadata(
    specification: KnownActionSceneSpecification,
    *,
    order: CandidateOrder = "pi",
) -> dict[str, Any]:
    """Return the literal task surface allowed to reach runtime collation.

    The target handle is appearance-defined.  Persistent IDs are deliberately
    absent because they must be resolved against the current public belief.
    Physical rows, primitive state, canonical winners, and evaluator labels are
    likewise absent.  The terminal goal is task input, not simulator state.
    """

    if not isinstance(specification, KnownActionSceneSpecification):
        raise TypeError("observable_task_metadata requires a known-action scene")
    goal = terminal_goal(specification)
    candidates: list[dict[str, Any]] = []
    for display_index in range(CANDIDATES_PER_BUNDLE):
        q = display_to_canonical(
            specification.camera_stratum,
            display_index,
            order=order,
        )
        candidate = candidate_specification(specification, q)
        candidates.append(
            {
                "display_index": display_index,
                "timestamp_seconds": candidate.timestamp_seconds,
                "impulse_world": tuple(
                    value / candidate.impulse_denominator for value in candidate.impulse_numerators
                ),
                "frame": "world",
            }
        )
    return {
        "observable_handle": {
            "role": specification.handle_role,
            "prototype": HANDLE_PROTOTYPES[specification.handle_role],
        },
        "candidate_order": order,
        "candidate_actions": candidates,
        "goal_position_world": tuple(float(value) for value in goal),
        "goal_horizon_seconds": HORIZONS_SECONDS[-1],
    }


def collate_observable_tasks(
    specifications: list[KnownActionSceneSpecification] | tuple[KnownActionSceneSpecification, ...],
    *,
    order: CandidateOrder = "pi",
) -> dict[str, Tensor]:
    """Collate only public task literals across heterogeneous formal bundles."""

    if not isinstance(specifications, (list, tuple)) or not specifications:
        raise ValueError("observable task collation requires a nonempty list or tuple")
    if any(
        not isinstance(specification, KnownActionSceneSpecification)
        for specification in specifications
    ):
        raise TypeError("every observable task row must be a known-action scene")
    rows = [
        observable_task_metadata(specification, order=order) for specification in specifications
    ]
    return {
        "observable_handle_role": torch.tensor(
            [row["observable_handle"]["role"] for row in rows],
            dtype=torch.int64,
        ),
        "observable_handle_prototype": torch.tensor(
            [row["observable_handle"]["prototype"] for row in rows],
            dtype=torch.float32,
        ),
        "candidate_timestamps": torch.tensor(
            [
                [candidate["timestamp_seconds"] for candidate in row["candidate_actions"]]
                for row in rows
            ],
            dtype=torch.float32,
        ),
        "candidate_impulses_world": torch.tensor(
            [
                [candidate["impulse_world"] for candidate in row["candidate_actions"]]
                for row in rows
            ],
            dtype=torch.float32,
        ),
        "goal_positions_world": torch.tensor(
            [row["goal_position_world"] for row in rows],
            dtype=torch.float32,
        ),
        "goal_horizons": torch.tensor(
            [row["goal_horizon_seconds"] for row in rows],
            dtype=torch.float32,
        ),
    }


def _pinhole_sphere_conic_batch(
    points_camera: Tensor,
    intrinsics: Tensor,
) -> dict[str, Tensor]:
    """Certify exact float64 pinhole-sphere silhouette ellipses.

    Inputs are [K,F,2,3] camera-space centres and [F,3,3] intrinsics.  The
    homogeneous tangency form is solved independently of the public renderer.
    """

    centre = points_camera.to(torch.float64)
    calibration = intrinsics.to(torch.float64).unsqueeze(0).expand(centre.shape[0], -1, -1, -1)
    radius = torch.full(
        centre.shape[:-1],
        FIXED_RADIUS_NUMERATOR / FIXED_RADIUS_DENOMINATOR,
        dtype=torch.float64,
    )
    if not (
        bool(torch.isfinite(centre).all())
        and bool(torch.isfinite(calibration).all())
        and bool((centre[..., 2] > radius + 1.0e-8).all())
    ):
        raise RuntimeError("conic certification requires finite spheres in front of the camera")
    identity3 = torch.eye(3, dtype=torch.float64).expand(*calibration.shape[:-2], -1, -1)
    inverse = torch.linalg.solve(calibration, identity3)
    rho = centre.square().sum(dim=-1) - radius.square()
    if not bool((rho > 1.0e-12).all()):
        raise RuntimeError("conic certification found a camera inside a sphere")
    tangent = centre[..., :, None] * centre[..., None, :] - rho[..., None, None] * torch.eye(
        3,
        dtype=torch.float64,
    )
    conic = torch.einsum(
        "bfij,bfojk,bfkl->bfoil",
        inverse.transpose(-1, -2),
        tangent,
        inverse,
    )
    scale = conic.abs().amax(dim=(-2, -1))
    symmetry_error = (conic - conic.transpose(-1, -2)).abs().amax(dim=(-2, -1)) / scale
    conic = 0.5 * (conic + conic.transpose(-1, -2)) / scale[..., None, None]
    eigenvalues = torch.linalg.eigvalsh(conic)
    signature_valid = (
        (eigenvalues[..., 0] < -CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (eigenvalues[..., 1] < -CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (eigenvalues[..., 2] > CONIC_RELATIVE_ALGEBRA_TOLERANCE)
    )
    projected = torch.einsum("bfij,bfoj->bfoi", calibration, centre)
    projected = projected / projected[..., 2:]
    projected_value = torch.einsum("bfoi,bfoij,bfoj->bfo", projected, conic, projected)
    quadratic = conic[..., :2, :2]
    linear = conic[..., :2, 2]
    constant = conic[..., 2, 2]
    ellipse_centre = torch.linalg.solve(quadratic, -linear.unsqueeze(-1)).squeeze(-1)
    gamma = constant + torch.einsum("bfoi,bfoi->bfo", linear, ellipse_centre)
    negative_quadratic = -quadratic
    negative_eigenvalues = torch.linalg.eigvalsh(negative_quadratic)
    if not bool(
        signature_valid.all()
        and (projected_value > CONIC_RELATIVE_ALGEBRA_TOLERANCE).all()
        and (gamma > CONIC_RELATIVE_ALGEBRA_TOLERANCE).all()
        and (negative_eigenvalues[..., 0] > CONIC_RELATIVE_ALGEBRA_TOLERANCE).all()
    ):
        raise RuntimeError("conic certification produced an invalid ellipse")
    ellipse_shape = negative_quadratic / gamma[..., None, None]
    identity2 = torch.eye(2, dtype=torch.float64).expand(*ellipse_shape.shape[:-2], -1, -1)
    covariance = torch.linalg.solve(ellipse_shape, identity2)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    coordinate_radius = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
    covariance_trace = covariance[..., 0, 0] + covariance[..., 1, 1]
    covariance_discriminant = torch.hypot(
        covariance[..., 0, 0] - covariance[..., 1, 1],
        2.0 * covariance[..., 0, 1],
    )
    enclosing_radius = (0.5 * (covariance_trace + covariance_discriminant)).sqrt()
    coordinate_extrema = torch.stack(
        (
            ellipse_centre[..., 0] - coordinate_radius[..., 0],
            ellipse_centre[..., 0] + coordinate_radius[..., 0],
            ellipse_centre[..., 1] - coordinate_radius[..., 1],
            ellipse_centre[..., 1] + coordinate_radius[..., 1],
        ),
        dim=-1,
    )
    axis0_delta = covariance[..., :, 0] / coordinate_radius[..., 0, None]
    axis1_delta = covariance[..., :, 1] / coordinate_radius[..., 1, None]
    boundary_points = torch.stack(
        (
            ellipse_centre - axis0_delta,
            ellipse_centre + axis0_delta,
            ellipse_centre - axis1_delta,
            ellipse_centre + axis1_delta,
        ),
        dim=-2,
    )
    homogeneous_boundary = torch.cat(
        (boundary_points, torch.ones_like(boundary_points[..., :1])),
        dim=-1,
    )
    boundary_value = torch.einsum(
        "bfopi,bfoij,bfopj->bfop",
        homogeneous_boundary,
        conic,
        homogeneous_boundary,
    )
    boundary_denominator = torch.linalg.matrix_norm(conic, ord="fro")[..., None] * (
        homogeneous_boundary.square().sum(dim=-1)
    )
    boundary_residual = boundary_value.abs() / boundary_denominator.clamp_min(1.0e-30)
    centre_residual = torch.linalg.vector_norm(
        torch.einsum("bfoij,bfoj->bfoi", quadratic, ellipse_centre) + linear,
        dim=-1,
    ) / (
        torch.linalg.matrix_norm(quadratic, ord="fro")
        * torch.linalg.vector_norm(ellipse_centre, dim=-1)
        + torch.linalg.vector_norm(linear, dim=-1)
    ).clamp_min(1.0e-30)
    shape_residual = torch.linalg.matrix_norm(
        ellipse_shape @ covariance - identity2,
        ord="fro",
    )
    valid = (
        signature_valid
        & torch.isfinite(coordinate_extrema).all(dim=-1)
        & torch.isfinite(enclosing_radius)
        & (symmetry_error <= CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (boundary_residual.amax(dim=-1) <= CONIC_RELATIVE_BOUNDARY_RESIDUAL_TOLERANCE)
        & (centre_residual <= CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (shape_residual <= CONIC_RELATIVE_ALGEBRA_TOLERANCE)
    )
    return {
        "conic_matrix": conic,
        "conic_eigenvalues": eigenvalues,
        "ellipse_centre": ellipse_centre,
        "ellipse_coordinate_extrema": coordinate_extrema,
        "ellipse_enclosing_radius": enclosing_radius,
        "conic_boundary_residual": boundary_residual,
        "conic_centre_residual": centre_residual,
        "conic_shape_residual": shape_residual,
        "conic_valid": valid,
    }


def _independent_geometry_batch(
    positions: Tensor,
    camera: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Trace [K,56,2] spheres with independent conics and stable ray roots."""

    if positions.ndim != 4 or positions.shape[1:] != (FRAME_COUNT, 2, 3):
        raise ValueError("geometry positions must have shape [K,56,2,3]")
    relative = positions - camera["position"][None, :, None, :]
    rotation = camera["world_from_camera"][:, :3, :3]
    points_camera = torch.einsum("bfoi,fij->bfoj", relative, rotation)
    conic = _pinhole_sphere_conic_batch(points_camera, camera["intrinsics"])
    height, width = IMAGE_SIZE
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    intrinsics = camera["intrinsics"]
    ray_x = (pixel_x[None] - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0]
    ray_y = (pixel_y[None] - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1]
    ray_norm_squared = 1.0 + ray_x.square() + ray_y.square()
    ray_dot_centre = (
        ray_x[None, :, None] * points_camera[..., 0, None, None]
        + ray_y[None, :, None] * points_camera[..., 1, None, None]
        + points_camera[..., 2, None, None]
    )
    centre_cross_ray = torch.stack(
        (
            points_camera[..., 1, None, None]
            - points_camera[..., 2, None, None] * ray_y[None, :, None],
            points_camera[..., 2, None, None] * ray_x[None, :, None]
            - points_camera[..., 0, None, None],
            points_camera[..., 0, None, None] * ray_y[None, :, None]
            - points_camera[..., 1, None, None] * ray_x[None, :, None],
        ),
        dim=-1,
    )
    radius = FIXED_RADIUS_NUMERATOR / FIXED_RADIUS_DENOMINATOR
    discriminant = ray_norm_squared[
        None, :, None
    ] * radius * radius - centre_cross_ray.square().sum(dim=-1)
    square_root = discriminant.clamp_min(0.0).sqrt()
    denominator = ray_dot_centre + square_root
    constant = points_camera.square().sum(dim=-1)[..., None, None] - radius * radius
    surface_depth = constant / denominator.clamp_min(1.0e-12)
    full_mask = (
        (points_camera[..., 2] > radius + 1.0e-4)[..., None, None]
        & (discriminant >= 0.0)
        & (denominator > 0.0)
        & (surface_depth > 0.0)
        & torch.isfinite(surface_depth)
    )
    ordered = torch.where(full_mask, surface_depth, torch.full_like(surface_depth, torch.inf))
    depth_buffer, winner = ordered.min(dim=2)
    has_object = torch.isfinite(depth_buffer)
    winner = torch.where(has_object, winner.to(torch.int64), torch.full_like(winner, -1))
    visible_mask = full_mask & (
        winner[:, :, None] == torch.arange(2, dtype=torch.int64)[None, None, :, None, None]
    )
    support = full_mask.sum(dim=(-2, -1))
    visible = visible_mask.sum(dim=(-2, -1))
    return {
        "points_camera": points_camera,
        "discriminant": discriminant,
        "surface_depth": surface_depth,
        "full_mask": full_mask,
        "visible_mask": visible_mask,
        "winner": winner,
        "depth_buffer": torch.where(has_object, depth_buffer, torch.zeros_like(depth_buffer)),
        "support": support,
        "visible": visible,
        **conic,
    }


def _continuous_candidate_margins(trajectory: PureActionTrajectory) -> dict[str, float]:
    """Compute exact per-substep closed-form contact and bound margins."""

    position = trajectory.substep_positions.to(torch.float64)
    velocity = trajectory.substep_velocities.to(torch.float64)
    relative_position = position[:-1, 0] - position[:-1, 1]
    relative_velocity = velocity[:-1, 0] - velocity[:-1, 1]
    drag = FIXED_DRAG_NUMERATOR / FIXED_DRAG_DENOMINATOR
    displacement = -math.expm1(-drag / PHYSICS_RATE_HZ) / drag
    denominator = relative_velocity.square().sum(dim=-1)
    projection = torch.where(
        denominator > 0.0,
        -(relative_position * relative_velocity).sum(dim=-1) / denominator,
        torch.zeros_like(denominator),
    ).clamp(0.0, displacement)
    closest = relative_position + relative_velocity * projection[:, None]
    surface_gap = (
        torch.linalg.vector_norm(closest, dim=-1)
        - 2.0 * FIXED_RADIUS_NUMERATOR / FIXED_RADIUS_DENOMINATOR
    )
    bounds = torch.tensor(WORLD_BOUNDS, dtype=torch.float64)
    world_boundary = torch.minimum(
        position - FIXED_RADIUS_NUMERATOR / FIXED_RADIUS_DENOMINATOR - bounds[:, 0],
        bounds[:, 1] - position - FIXED_RADIUS_NUMERATOR / FIXED_RADIUS_DENOMINATOR,
    )
    speeds = torch.linalg.vector_norm(velocity, dim=-1)
    floor_clearance = (
        position[..., 1] - FIXED_RADIUS_NUMERATOR / FIXED_RADIUS_DENOMINATOR - WORLD_BOUNDS[1][0]
    )
    floor_contact = floor_clearance <= BOUNDARY_CONTACT_TOLERANCE_M
    sleep_candidate = floor_contact & (speeds < FROZEN_ORPHEUS_CONFIG_SLEEP_SPEED_MPS)
    return {
        "minimum_world_surface_gap_m": float(surface_gap.min()),
        "minimum_world_boundary_clearance_m": float(world_boundary.min()),
        "minimum_floor_clearance_m": float(floor_clearance.min()),
        "minimum_speed_mps": float(speeds.min()),
        "maximum_speed_mps": float(speeds.max()),
        "floor_contact_count": float(floor_contact.sum()),
        "sleep_candidate_count": float(sleep_candidate.sum()),
    }


def _validate_public_source_bindings() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    actual = {
        "accepted_orbital_qualification": hashlib.sha256(
            Path(__file__)
            .with_name("rgbd_two_visible_orbital_camera_qualification.py")
            .read_bytes()
        ).hexdigest(),
        "camera": hashlib.sha256((package / "simulator" / "camera.py").read_bytes()).hexdigest(),
        "physics": hashlib.sha256((package / "simulator" / "physics.py").read_bytes()).hexdigest(),
        "renderer": hashlib.sha256(
            (package / "simulator" / "renderer.py").read_bytes()
        ).hexdigest(),
        "collisions": hashlib.sha256(
            (package / "simulator" / "collisions.py").read_bytes()
        ).hexdigest(),
        "config": hashlib.sha256((package / "utils" / "config.py").read_bytes()).hexdigest(),
        "default_config": hashlib.sha256(
            (package.parent / "configs" / "default.yaml").read_bytes()
        ).hexdigest(),
    }
    expected = {
        "accepted_orbital_qualification": ACCEPTED_ORBITAL_SOURCE_SHA256,
        "camera": PUBLIC_CAMERA_SOURCE_SHA256,
        "physics": PUBLIC_PHYSICS_SOURCE_SHA256,
        "renderer": PUBLIC_RENDERER_SOURCE_SHA256,
        "collisions": PUBLIC_COLLISIONS_SOURCE_SHA256,
        "config": PUBLIC_CONFIG_SOURCE_SHA256,
        "default_config": PUBLIC_DEFAULT_CONFIG_SHA256,
    }
    if actual != expected:
        raise RuntimeError("public source differs from the frozen independent certificate binding")
    return actual


def _validated_determinism_scope() -> dict[str, str]:
    actual = {
        "byteorder": sys.byteorder,
        "device": "cpu",
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": platform.python_version(),
        "state_dtype": "torch.float32",
        "conic_dtype": "torch.float64",
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_version": torch.__version__,
    }
    expected = {
        "byteorder": FROZEN_BYTEORDER,
        "device": "cpu",
        "platform_machine": FROZEN_PLATFORM_MACHINE,
        "platform_system": FROZEN_PLATFORM_SYSTEM,
        "python_version": FROZEN_PYTHON_VERSION,
        "state_dtype": "torch.float32",
        "conic_dtype": "torch.float64",
        "torch_default_dtype": "torch.float32",
        "torch_version": FROZEN_TORCH_VERSION,
    }
    if actual != expected:
        raise RuntimeError("known-action certificate is outside its frozen determinism scope")
    return actual


def _normalised_scene_source_sha256() -> str:
    """Hash this source while excluding only generated freeze literal values."""

    contents = Path(__file__).read_bytes()
    begin = b"# BEGIN GENERATED CERTIFICATE FREEZE\n"
    end = b"# END GENERATED CERTIFICATE FREEZE\n"
    start = contents.find(begin)
    finish = contents.find(end)
    if start < 0 or finish < 0 or finish <= start:
        raise RuntimeError("known-action source lacks exact generated-freeze markers")
    normalised = (
        contents[:start]
        + begin
        + b"<generated-certificate-freeze-values-omitted>\n"
        + contents[finish:]
    )
    return hashlib.sha256(normalised).hexdigest()


def _input_binding_literal(value: Any) -> Any:
    """Encode semantic constants without losing container or key types."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError("certificate constants must be finite")
        return value
    if type(value) is tuple:
        return {
            "container": "tuple",
            "items": [_input_binding_literal(member) for member in value],
        }
    if type(value) is list:
        return {
            "container": "list",
            "items": [_input_binding_literal(member) for member in value],
        }
    if type(value) is dict:
        items = [
            [_input_binding_literal(key), _input_binding_literal(member)]
            for key, member in value.items()
        ]
        items.sort(key=lambda item: _canonical_json_bytes(item[0]))
        return {"container": "dict", "items": items}
    raise TypeError(f"unsupported certificate constant binding {type(value)!r}")


def _semantic_constant_bindings() -> dict[str, Any]:
    generated = {
        "FROZEN_NORMALIZED_SCENE_SOURCE_SHA256",
        "FROZEN_INPUT_BINDING_SHA256",
        "FROZEN_MANIFEST_SHA256",
        "FROZEN_TRACE_SHA256",
        "FROZEN_DESCRIPTOR_SHA256",
        "FROZEN_CERTIFICATE_SHA256",
    }
    return {
        name: _input_binding_literal(value)
        for name, value in globals().items()
        if name and name[0].isalpha() and name.isupper() and name not in generated
    }


def _fresh_certificate_input_binding() -> dict[str, Any]:
    """Freshly bind source, host, descriptor, manifests, and every constant."""

    source_bindings = _validate_public_source_bindings()
    determinism_scope = _validated_determinism_scope()
    manifest_sha256 = {split: canonical_sha256(split_manifest(split)) for split in SPLITS}
    unsigned = {
        "schema": "rgbd_known_action_certificate_input_binding_v2",
        "normalised_scene_source_sha256": _normalised_scene_source_sha256(),
        "source_bindings": source_bindings,
        "determinism_scope": determinism_scope,
        "descriptor_sha256": canonical_sha256(_descriptor_unsigned()),
        "manifest_sha256": manifest_sha256,
        "semantic_constants": _semantic_constant_bindings(),
    }
    return {
        **unsigned,
        "binding_sha256": canonical_sha256(unsigned),
    }


def _descriptor_unsigned() -> dict[str, Any]:
    return {
        "name": "rgbd_known_action_counterfactual_planning_scene_v1",
        "authority": "literal_seedless_source_descriptor_and_independent_truth_only",
        "normalised_scene_source_sha256": _normalised_scene_source_sha256(),
        "split_order": list(SPLITS),
        "evidence_roles": dict(SPLIT_EVIDENCE_ROLE),
        "scene_axes": {
            "coefficient_levels": list(COEFFICIENT_LEVELS),
            "primitives_per_split": PRIMITIVES_PER_SPLIT,
            "handle_roles": HANDLE_ROLES,
            "camera_strata": CAMERA_STRATA,
            "candidates_per_bundle": CANDIDATES_PER_BUNDLE,
            "bundles_per_split": BUNDLES_PER_SPLIT,
            "total_primitive_profiles": TOTAL_PRIMITIVE_PROFILES,
            "total_bundles": TOTAL_BUNDLES,
            "total_candidates": TOTAL_CANDIDATES,
        },
        "ordinal_mapping": {
            "formula": "16*primitive_index+8*handle_role+camera_stratum",
            "primitive_index": "ordinal//16",
            "handle_role": "(ordinal%16)//8",
            "camera_stratum": "ordinal%8",
            "role_twin": "ordinal xor 8",
        },
        "primitive_mapping": {
            "pairs_by_split": {
                split: [list(pair) for pair in SPLIT_PRIMITIVE_PAIRS[split]] for split in SPLITS
            },
            "position_denominator": POSITION_DENOMINATOR,
            "position_numerator_rows": [
                "-450+6*a,400+2*b,-300+4*b",
                "450+5*b,1750+2*a,300-4*a",
            ],
            "velocity_denominator": VELOCITY_DENOMINATOR,
            "velocity_numerator_rows": [
                "180+2*a,48+b,16+a",
                "-172+2*b,-32+a,-12+b",
            ],
        },
        "known_physical_controls": {
            "fixed_radius_rational_m": [
                FIXED_RADIUS_NUMERATOR,
                FIXED_RADIUS_DENOMINATOR,
            ],
            "fixed_mass_rational_belief_unit": [
                FIXED_MASS_NUMERATOR,
                FIXED_MASS_DENOMINATOR,
            ],
            "fixed_drag_rational_per_s": [
                FIXED_DRAG_NUMERATOR,
                FIXED_DRAG_DENOMINATOR,
            ],
            "gravity_mps2": [0, 0, 0],
            "contact_free": True,
        },
        "timing": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FRAME_RATE_HZ,
            "physics_rate_hz": PHYSICS_RATE_HZ,
            "substeps_per_frame": SUBSTEPS_PER_FRAME,
            "history_frame_count": HISTORY_FRAME_COUNT,
            "anchor_frame_index": ANCHOR_FRAME_INDEX,
            "anchor_time_seconds": ANCHOR_TIME_SECONDS,
            "query_horizons_seconds": list(HORIZONS_SECONDS),
            "query_frame_indices": list(QUERY_FRAME_INDICES),
            "action_delay_law": "(6+rank(a))/20 seconds after anchor",
            "action_delay_numerators": [6, 7, 8, 9],
            "action_is_strictly_after_anchor": True,
            "action_is_distinct_from_every_query": True,
            "right_continuous_semantics": (
                "position continuous; velocity includes impulse at exact action timestamp"
            ),
        },
        "action_candidates": {
            "canonical_indices": list(range(CANDIDATES_PER_BUNDLE)),
            "sign_bits": {
                "x": "bit0:0->-1,1->+1",
                "y": "bit1:0->-1,1->+1",
                "z": "bit2:0->-1,1->+1",
            },
            "impulse_denominator": IMPULSE_DENOMINATOR,
            "magnitude_numerators": [
                "jx=21+rank(b)",
                "jy=19+rank(a)",
                "jz=23+((rank(a)+rank(b))%4)",
            ],
            "target": "observable_handle_H_r",
            "persistent_id_policy": ("caller_must_resolve_handle_prototype_against_public_belief"),
            "scene_role_or_physical_row_used_as_persistent_id": False,
            "frame": "world",
            "candidate_count": CANDIDATES_PER_BUNDLE,
            "equal_energy_within_bundle": True,
            "single_action_per_candidate": True,
            "old_two_action_schedule_absent": True,
        },
        "candidate_permutations": {
            "canonical_display_to_q": "q=d",
            "pi_display_to_q": "q=(d+2*s)%8",
            "pi_q_to_display": "d=(q-2*s)%8",
            "rho_display_to_q": "q=(5*d+3+2*s)%8",
            "rho_q_to_display": "d=5*(q-3-2*s)%8",
            "depends_only_on_camera_stratum": True,
            "independent_of_role_goal_and_optimum": True,
        },
        "planning_task": {
            "optimal_canonical_index": "q*=(2*p+r+s)%8",
            "goal": "controlled_object_exact_position_at_anchor_plus_2_seconds_under_q*",
            "cost": "terminal_position_squared_euclidean_error",
            "opposite_goal_index": "q_alt=q* xor 7",
            "unique_winner_required": True,
        },
        "appearance_and_handles": {
            "palette": [list(colour) for colour in PALETTE],
            "handle_prototypes": [list(colour) for colour in HANDLE_PROTOTYPES],
            "palette_swap": "w=(primitive+phase_index+direction_index)&1",
            "physical_row_for_handle": "c=handle_role xor w",
            "palette_is_independent_of_handle_role": True,
            "candidate_order_is_independent_of_handle_role": True,
            "palette_direction_term": "direction_index_not_signed_direction",
        },
        "observable_runtime_surface": {
            "allowed": [
                "observable_handle_role",
                "observable_handle_prototype",
                "candidate_timestamps",
                "candidate_impulses_world",
                "goal_positions_world",
                "goal_horizons",
            ],
            "resolved_persistent_id": "required_separately_at_action_materialization",
            "physical_controlled_row_exposed": False,
            "canonical_winner_exposed": False,
            "primitive_position_or_velocity_exposed": False,
        },
        "camera": {
            "phase_radians": ["0", "pi/2", "pi", "3pi/2"],
            "directions": list(CAMERA_DIRECTIONS),
            "theta_law": "theta0+direction*0.24*t",
            "position_law": ["4.6*sin(theta)", "2.15", "4.6*cos(theta)"],
            "target": list(CAMERA_TARGET),
            "vertical_fov_degrees": CAMERA_VERTICAL_FOV_DEGREES,
            "image_size": list(IMAGE_SIZE),
        },
        "independent_truth": {
            "no_action_law": "D(u)=exp(-0.05*u),G(u)=(1-D(u))/0.05",
            "action_velocity_delta": "indicator(u>=tau)*D(u-tau)*J/mass",
            "action_position_delta": "indicator(u>=tau)*G(u-tau)*J/mass",
            "public_camera_calls": False,
            "public_renderer_calls": False,
            "public_physics_calls": False,
            "public_dynamics_calls": False,
            "public_runtime_calls": False,
            "public_planner_calls": False,
            "independent_float32_drag_state": True,
            "independent_float64_pinhole_sphere_conic": True,
            "independent_stable_near_root_raster": True,
            "substep_position_velocity_action_event_trace_hashed": True,
        },
        "sleep_semantics": {
            "public_law": (
                "sleep_candidate=active&floor_contact&(speed<sleep_speed); "
                "sleep requires consecutive candidate substeps"
            ),
            "simulator_default_sleep_speed_mps": FROZEN_SIMULATOR_SLEEP_SPEED_MPS,
            "simulator_default_sleep_after_seconds": (FROZEN_SIMULATOR_SLEEP_AFTER_SECONDS),
            "orpheus_config_default_sleep_speed_mps": (FROZEN_ORPHEUS_CONFIG_SLEEP_SPEED_MPS),
            "boundary_contact_tolerance_m": BOUNDARY_CONTACT_TOLERANCE_M,
            "proof": (
                "minimum_floor_clearance_exceeds_contact_tolerance_so_"
                "floor_contact_and_sleep_candidate_are_exactly_false"
            ),
        },
        "source_bindings": {
            "accepted_orbital_source_sha256": ACCEPTED_ORBITAL_SOURCE_SHA256,
            "accepted_orbital_certificate_sha256": ACCEPTED_ORBITAL_CERTIFICATE_SHA256,
            "camera_source_sha256": PUBLIC_CAMERA_SOURCE_SHA256,
            "physics_source_sha256": PUBLIC_PHYSICS_SOURCE_SHA256,
            "renderer_source_sha256": PUBLIC_RENDERER_SOURCE_SHA256,
            "collisions_source_sha256": PUBLIC_COLLISIONS_SOURCE_SHA256,
            "config_source_sha256": PUBLIC_CONFIG_SOURCE_SHA256,
            "default_config_sha256": PUBLIC_DEFAULT_CONFIG_SHA256,
        },
        "determinism_scope": {
            "torch_version": FROZEN_TORCH_VERSION,
            "python_version": FROZEN_PYTHON_VERSION,
            "platform_system": FROZEN_PLATFORM_SYSTEM,
            "platform_machine": FROZEN_PLATFORM_MACHINE,
            "byteorder": FROZEN_BYTEORDER,
            "device": "cpu",
            "state_dtype": "torch.float32",
            "conic_dtype": "torch.float64",
            "cross_build_digest_portability_claim": False,
        },
        "gates": {
            "minimum_full_support_pixels": MINIMUM_FULL_SUPPORT_PIXELS,
            "minimum_conic_gap_lower_bound_pixels": MINIMUM_CONIC_GAP_LOWER_BOUND_PIXELS,
            "minimum_conic_boundary_clearance_pixels": (MINIMUM_CONIC_BOUNDARY_CLEARANCE_PIXELS),
            "minimum_world_surface_gap_m": MINIMUM_WORLD_SURFACE_GAP_M,
            "minimum_world_boundary_m": MINIMUM_WORLD_BOUNDARY_M,
            "maximum_camera_calibration_error": MAXIMUM_CAMERA_CALIBRATION_ERROR,
            "minimum_camera_step_angle_radians": MINIMUM_CAMERA_STEP_ANGLE_RADIANS,
            "maximum_camera_step_angle_radians": MAXIMUM_CAMERA_STEP_ANGLE_RADIANS,
            "minimum_camera_translation_step_m": MINIMUM_CAMERA_TRANSLATION_STEP_M,
            "maximum_camera_translation_step_m": MAXIMUM_CAMERA_TRANSLATION_STEP_M,
            "maximum_no_action_projected_centre_step_pixels": (
                MAXIMUM_PROJECTED_CENTRE_STEP_PIXELS
            ),
            "maximum_acted_projected_centre_step_pixels": (
                MAXIMUM_ACTED_PROJECTED_CENTRE_STEP_PIXELS
            ),
            "minimum_action_after_anchor_seconds": MINIMUM_ACTION_AFTER_ANCHOR_SECONDS,
            "minimum_action_query_separation_seconds": (MINIMUM_ACTION_QUERY_SEPARATION_SECONDS),
            "minimum_matched_handle_cosine": MINIMUM_MATCHED_HANDLE_COSINE,
            "minimum_cross_handle_cosine_distance": MINIMUM_CROSS_HANDLE_COSINE_DISTANCE,
            "minimum_floor_clearance_m": BOUNDARY_CONTACT_TOLERANCE_M,
            "floor_contact_count": 0,
            "sleep_candidate_count": 0,
        },
        "cache_validation": {
            "cache_key": "fresh_complete_input_binding_sha256",
            "fresh_validation_before_lookup": True,
            "fresh_validation_after_lookup": True,
            "input_binding_components": [
                "normalised_scene_source_sha256",
                "external_source_bindings",
                "determinism_scope",
                "descriptor_sha256",
                "manifest_sha256",
                "all_semantic_constants_and_tables",
            ],
        },
        "digest_recipe": {
            "algorithm": "sha256",
            "domain_tag": "orpheus-known-action-v1 followed by NUL",
            "field_framing": [
                "domain_tag",
                "label_utf8_length_u32_little_endian",
                "label_utf8_bytes",
                "payload_length_u64_little_endian",
                "payload_bytes",
            ],
            "scalar_tables": "canonical_json_sorted_keys_compact_no_nan",
            "tensor_metadata": "canonical_json_dtype_shape_C_order_little_endian",
            "tensor_bytes": "cpu_contiguous_canonical_little_endian_C_order",
            "scene_order": "split_order_then_ordinal_then_canonical_q",
            "unordered_schedule_sets": "lexicographically_sorted_framed_row_sha256",
            "continuous_contact_proof_fields": [
                "substep_positions",
                "substep_velocities",
                "substep_action_events",
            ],
        },
    }


def certificate_descriptor() -> dict[str, Any]:
    unsigned = _descriptor_unsigned()
    return {
        **copy.deepcopy(unsigned),
        "descriptor_sha256": canonical_sha256(unsigned),
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }


def _camera_certificate(
    camera_tensors: dict[int, dict[str, Tensor]],
    camera_digest: Any,
) -> dict[str, float]:
    maximum_inverse_error = 0.0
    maximum_orthonormality_error = 0.0
    maximum_radius_error = 0.0
    maximum_height_error = 0.0
    maximum_target_error = 0.0
    maximum_intrinsics_error = 0.0
    maximum_position_binding_error = 0.0
    minimum_step_angle = math.inf
    maximum_step_angle = 0.0
    minimum_translation = math.inf
    maximum_translation = 0.0
    expected_intrinsics = _make_intrinsics()
    expected_target = torch.tensor(CAMERA_TARGET, dtype=torch.float32)
    signatures: set[str] = set()
    for stratum, values in camera_tensors.items():
        fields = tuple((name, tensor) for name, tensor in values.items())
        signatures.add(_tensor_sequence_sha256(fields))
        for name, tensor in fields:
            _update_tensor_digest(camera_digest, f"camera_{stratum}/{name}", tensor)
        identity = values["world_from_camera"] @ values["camera_from_world"]
        rotation = values["world_from_camera"][:, :3, :3]
        maximum_inverse_error = max(
            maximum_inverse_error,
            float((identity - torch.eye(4)).abs().max()),
        )
        maximum_orthonormality_error = max(
            maximum_orthonormality_error,
            float((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().max()),
        )
        maximum_radius_error = max(
            maximum_radius_error,
            float(
                (torch.linalg.vector_norm(values["position"][:, [0, 2]], dim=-1) - CAMERA_RADIUS_M)
                .abs()
                .max()
            ),
        )
        maximum_height_error = max(
            maximum_height_error,
            float((values["position"][:, 1] - CAMERA_HEIGHT_M).abs().max()),
        )
        maximum_target_error = max(
            maximum_target_error,
            float((values["target"] - expected_target).abs().max()),
        )
        maximum_intrinsics_error = max(
            maximum_intrinsics_error,
            float((values["intrinsics"] - expected_intrinsics).abs().max()),
        )
        maximum_position_binding_error = max(
            maximum_position_binding_error,
            float((values["world_from_camera"][:, :3, 3] - values["position"]).abs().max()),
        )
        orbit = values["position"][:, [0, 2]]
        cosine = (orbit[:-1] * orbit[1:]).sum(dim=-1) / (
            torch.linalg.vector_norm(orbit[:-1], dim=-1)
            * torch.linalg.vector_norm(orbit[1:], dim=-1)
        )
        angles = torch.acos(cosine.clamp(-1.0, 1.0))
        translations = torch.linalg.vector_norm(
            values["position"][1:] - values["position"][:-1],
            dim=-1,
        )
        minimum_step_angle = min(minimum_step_angle, float(angles.min()))
        maximum_step_angle = max(maximum_step_angle, float(angles.max()))
        minimum_translation = min(minimum_translation, float(translations.min()))
        maximum_translation = max(maximum_translation, float(translations.max()))
    calibration_error = max(
        maximum_inverse_error,
        maximum_orthonormality_error,
        maximum_radius_error,
        maximum_height_error,
        maximum_target_error,
        maximum_intrinsics_error,
        maximum_position_binding_error,
    )
    if len(signatures) != CAMERA_STRATA:
        raise RuntimeError("certificate requires eight distinct orbital camera traces")
    if calibration_error > MAXIMUM_CAMERA_CALIBRATION_ERROR:
        raise RuntimeError("camera calibration exceeds the frozen gate")
    if not (
        MINIMUM_CAMERA_STEP_ANGLE_RADIANS
        <= minimum_step_angle
        <= maximum_step_angle
        <= MAXIMUM_CAMERA_STEP_ANGLE_RADIANS
    ):
        raise RuntimeError("camera angular step differs from the accepted orbit")
    if not (
        MINIMUM_CAMERA_TRANSLATION_STEP_M
        <= minimum_translation
        <= maximum_translation
        <= MAXIMUM_CAMERA_TRANSLATION_STEP_M
    ):
        raise RuntimeError("camera translation step differs from the accepted orbit")
    return {
        "shared_camera_trace_count": float(len(signatures)),
        "maximum_camera_inverse_error": maximum_inverse_error,
        "maximum_camera_orthonormality_error": maximum_orthonormality_error,
        "maximum_camera_radius_error_m": maximum_radius_error,
        "maximum_camera_height_error_m": maximum_height_error,
        "maximum_camera_target_error_m": maximum_target_error,
        "maximum_camera_intrinsics_error": maximum_intrinsics_error,
        "maximum_camera_position_binding_error_m": maximum_position_binding_error,
        "minimum_adjacent_camera_angle_radians": minimum_step_angle,
        "maximum_adjacent_camera_angle_radians": maximum_step_angle,
        "minimum_adjacent_camera_translation_m": minimum_translation,
        "maximum_adjacent_camera_translation_m": maximum_translation,
    }


@lru_cache(maxsize=4)
def _computed_scene_family_certificate_bytes_cached(
    input_binding_sha256: str,
) -> bytes:
    """Exhaustively certify 16 profiles, 256 bundles, and 2048 candidates."""

    fresh_input = _fresh_certificate_input_binding()
    if input_binding_sha256 != fresh_input["binding_sha256"]:
        raise RuntimeError("certificate inputs changed before exhaustive computation")
    source_bindings = fresh_input["source_bindings"]
    determinism_scope = fresh_input["determinism_scope"]
    descriptor_sha256 = fresh_input["descriptor_sha256"]
    manifest_sha256 = fresh_input["manifest_sha256"]
    metadata_table = [
        scene_metadata(scene_specification(split, ordinal))
        for split in SPLITS
        for ordinal in range(BUNDLES_PER_SPLIT)
    ]
    metadata_sha256 = canonical_sha256(metadata_table)
    if len({canonical_sha256(row) for row in metadata_table}) != TOTAL_BUNDLES:
        raise RuntimeError("all 256 bundle descriptors must be unique")
    if set(pair for split in SPLITS for pair in SPLIT_PRIMITIVE_PAIRS[split]) != set(
        (a, b) for a in COEFFICIENT_LEVELS for b in COEFFICIENT_LEVELS
    ):
        raise RuntimeError("split primitive tables must partition the exact 4x4 coefficient grid")

    prefix_digest = hashlib.sha256()
    camera_digest = hashlib.sha256()
    ordered_schedule_digest = hashlib.sha256()
    unordered_schedule_digest = hashlib.sha256()
    candidate_state_digest = hashlib.sha256()
    substep_contact_proof_digest = hashlib.sha256()
    goal_cost_digest = hashlib.sha256()
    conic_raster_digest = hashlib.sha256()
    combined_digest = hashlib.sha256()

    profile_signatures: set[str] = set()
    for split in SPLITS:
        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            specification = scene_specification(split, primitive_index * 16)
            prefix = _prefix_for_profile(split, primitive_index)
            values = (
                ("positions", prefix.positions),
                ("velocities", prefix.velocities),
                ("substep_positions", prefix.substep_positions),
                ("substep_velocities", prefix.substep_velocities),
                ("action_events", prefix.action_events.to(torch.uint8)),
                (
                    "substep_action_events",
                    prefix.substep_action_events.to(torch.uint8),
                ),
            )
            profile_signatures.add(_tensor_sequence_sha256(values))
            for label, value in values:
                _update_tensor_digest(
                    prefix_digest,
                    f"{split}/primitive_{primitive_index}/{label}",
                    value,
                )
    if len(profile_signatures) != TOTAL_PRIMITIVE_PROFILES:
        raise RuntimeError("certificate requires sixteen distinct prefix profiles")

    cameras = {stratum: _stack_camera_frames(stratum) for stratum in range(CAMERA_STRATA)}
    camera_metrics = _camera_certificate(cameras, camera_digest)

    palette = torch.tensor(PALETTE, dtype=torch.float64)
    normalised_palette = palette / torch.linalg.vector_norm(palette, dim=-1, keepdim=True)
    palette_cosine = normalised_palette @ normalised_palette.T
    matched_handle_cosine = math.inf
    cross_handle_cosine_distance = math.inf
    primitive_histogram = {
        split: {str(index): 0 for index in range(PRIMITIVES_PER_SPLIT)} for split in SPLITS
    }
    role_histogram = {split: {str(index): 0 for index in range(HANDLE_ROLES)} for split in SPLITS}
    camera_histogram = {
        split: {str(index): 0 for index in range(CAMERA_STRATA)} for split in SPLITS
    }
    palette_histogram = {split: {"false": 0, "true": 0} for split in SPLITS}
    winner_histogram = {
        split: {str(index): 0 for index in range(CANDIDATES_PER_BUNDLE)} for split in SPLITS
    }
    pi_winner_display_histogram = {
        split: {str(index): 0 for index in range(CANDIDATES_PER_BUNDLE)} for split in SPLITS
    }
    rho_winner_display_histogram = copy.deepcopy(pi_winner_display_histogram)

    minimum_support = math.inf
    minimum_conic_gap = math.inf
    minimum_boundary = math.inf
    minimum_projected_handle_centre_separation = math.inf
    maximum_prefix_projected_centre_step = 0.0
    maximum_no_action_projected_centre_step = 0.0
    maximum_acted_projected_centre_step = 0.0
    maximum_incremental_acted_projected_centre_step = 0.0
    minimum_hit_discriminant = math.inf
    maximum_conic_boundary_residual = 0.0
    maximum_conic_centre_residual = 0.0
    maximum_conic_shape_residual = 0.0
    minimum_world_surface_gap = math.inf
    minimum_world_boundary = math.inf
    minimum_floor_clearance = math.inf
    minimum_speed = math.inf
    maximum_speed = 0.0
    floor_contact_count = 0
    sleep_candidate_count = 0
    minimum_action_after_anchor = math.inf
    minimum_action_query_separation = math.inf
    minimum_action_before_terminal = math.inf
    minimum_impulse_energy = math.inf
    maximum_impulse_energy = 0.0
    minimum_terminal_position_effect_by_axis = [math.inf, math.inf, math.inf]
    minimum_terminal_velocity_effect_by_axis = [math.inf, math.inf, math.inf]
    minimum_unique_winner_sse_gap = math.inf
    minimum_opposite_goal_unique_winner_sse_gap = math.inf
    maximum_pre_action_state_delta = 0.0
    maximum_distractor_state_delta = 0.0
    action_controlled_event_count = 0
    action_distractor_event_count = 0
    zero_impulse_control_event_count = 0
    zero_impulse_control_state_delta = 0.0
    role_twin_pair_count = 0
    permutation_invariance_count = 0
    candidate_signatures: set[str] = set()
    schedule_set_signatures: set[str] = set()
    goal_signatures: set[str] = set()
    no_action_projected_centres: dict[tuple[str, int, int], Tensor] = {}

    for split in SPLITS:
        for ordinal in range(BUNDLES_PER_SPLIT):
            specification = scene_specification(split, ordinal)
            primitive_histogram[split][str(specification.primitive_index)] += 1
            role_histogram[split][str(specification.handle_role)] += 1
            camera_histogram[split][str(specification.camera_stratum)] += 1
            palette_histogram[split][str(specification.palette_swapped).lower()] += 1
            winner_histogram[split][str(specification.optimal_canonical_index)] += 1
            pi_winner = canonical_to_display(
                specification.camera_stratum,
                specification.optimal_canonical_index,
                order="pi",
            )
            rho_winner = canonical_to_display(
                specification.camera_stratum,
                specification.optimal_canonical_index,
                order="rho",
            )
            pi_winner_display_histogram[split][str(pi_winner)] += 1
            rho_winner_display_histogram[split][str(rho_winner)] += 1

            controlled_row = specification.physical_controlled_row
            matched = float(
                palette_cosine[
                    controlled_row ^ int(specification.palette_swapped),
                    specification.handle_role,
                ]
            )
            cross = 1.0 - float(palette_cosine[0, 1])
            matched_handle_cosine = min(matched_handle_cosine, matched)
            cross_handle_cosine_distance = min(cross_handle_cosine_distance, cross)
            if matched < MINIMUM_MATCHED_HANDLE_COSINE:
                raise RuntimeError("observable handle no longer matches its physical palette row")
            if cross < MINIMUM_CROSS_HANDLE_COSINE_DISTANCE:
                raise RuntimeError(
                    "observable handle palette prototypes are insufficiently distinct"
                )

            if specification.handle_role == 0:
                twin = scene_specification(split, role_twin_ordinal(ordinal))
                invariant_fields = (
                    "split",
                    "primitive_index",
                    "camera_stratum",
                    "phase_index",
                    "direction_index",
                    "direction",
                    "a",
                    "b",
                    "position_numerators",
                    "velocity_numerators",
                    "palette_swapped",
                    "albedo",
                    "action_delay_numerator",
                    "impulse_magnitude_numerators",
                )
                if any(
                    getattr(specification, field) != getattr(twin, field)
                    for field in invariant_fields
                ):
                    raise RuntimeError(
                        "role twin changed prefix, camera, palette, or action family"
                    )
                if (
                    twin.handle_role != 1
                    or twin.physical_controlled_row == specification.physical_controlled_row
                ):
                    raise RuntimeError("role twin did not toggle exactly the observable target")
                role_twin_pair_count += 1

            candidates = tuple(
                candidate_specification(specification, q) for q in range(CANDIDATES_PER_BUNDLE)
            )
            trajectories = tuple(
                manual_candidate_trajectory(specification, q) for q in range(CANDIDATES_PER_BUNDLE)
            )
            positions = torch.stack([trajectory.positions for trajectory in trajectories])
            velocities = torch.stack([trajectory.velocities for trajectory in trajectories])
            events = torch.stack([trajectory.action_events for trajectory in trajectories])
            prefix = _prefix_for_profile(split, specification.primitive_index)
            no_action_position = prefix.positions
            no_action_velocity = prefix.velocities

            schedule_rows = {
                order: [
                    action_schedule_metadata(
                        candidates[
                            display_to_canonical(
                                specification.camera_stratum,
                                display,
                                order=order,
                            )
                        ]
                    )
                    for display in range(CANDIDATES_PER_BUNDLE)
                ]
                for order in ("canonical", "pi", "rho")
            }
            prefix_label = f"{split}/ordinal_{ordinal}"
            for order, rows in schedule_rows.items():
                _digest_field(
                    ordered_schedule_digest,
                    f"{prefix_label}/{order}",
                    _canonical_json_bytes(rows),
                )
            schedule_row_hashes = sorted(
                canonical_sha256(action_schedule_metadata(candidate)) for candidate in candidates
            )
            schedule_set_signature = canonical_sha256(schedule_row_hashes)
            schedule_set_signatures.add(schedule_set_signature)
            _digest_field(
                unordered_schedule_digest,
                prefix_label,
                _canonical_json_bytes(schedule_row_hashes),
            )

            energy = torch.stack(
                [
                    candidate.impulse_tensor(dtype=torch.float64).square().sum()
                    for candidate in candidates
                ]
            )
            if not torch.equal(energy, energy[0].expand_as(energy)):
                raise RuntimeError("candidate impulses are not exactly equal energy")
            minimum_impulse_energy = min(minimum_impulse_energy, float(energy.min()))
            maximum_impulse_energy = max(maximum_impulse_energy, float(energy.max()))

            q_star = specification.optimal_canonical_index
            q_alt = q_star ^ 7
            goal = positions[q_star, -1, controlled_row]
            opposite_goal = positions[q_alt, -1, controlled_row]
            canonical_costs = (positions[:, -1, controlled_row] - goal[None]).square().sum(dim=-1)
            opposite_costs = (
                (positions[:, -1, controlled_row] - opposite_goal[None]).square().sum(dim=-1)
            )
            if int(canonical_costs.argmin()) != q_star or int(opposite_costs.argmin()) != q_alt:
                raise RuntimeError("a formal goal lacks its declared canonical winner")
            sorted_costs = canonical_costs.sort().values
            sorted_opposite = opposite_costs.sort().values
            if not (
                float(sorted_costs[0]) == 0.0
                and float(sorted_costs[1]) > 0.0
                and float(sorted_opposite[0]) == 0.0
                and float(sorted_opposite[1]) > 0.0
            ):
                raise RuntimeError("terminal position cost does not have a unique exact winner")
            minimum_unique_winner_sse_gap = min(
                minimum_unique_winner_sse_gap,
                float(sorted_costs[1]),
            )
            minimum_opposite_goal_unique_winner_sse_gap = min(
                minimum_opposite_goal_unique_winner_sse_gap,
                float(sorted_opposite[1]),
            )
            goal_signatures.add(
                _tensor_sequence_sha256(
                    (
                        ("goal", goal),
                        ("opposite_goal", opposite_goal),
                    )
                )
            )
            for order in ("pi", "rho"):
                indices = torch.tensor(
                    candidate_order(specification.camera_stratum, order=order),
                    dtype=torch.int64,
                )
                reordered = canonical_costs[indices]
                opposite_reordered = opposite_costs[indices]
                expected_winner = canonical_to_display(
                    specification.camera_stratum,
                    q_star,
                    order=order,
                )
                expected_opposite = canonical_to_display(
                    specification.camera_stratum,
                    q_alt,
                    order=order,
                )
                if (
                    int(reordered.argmin()) != expected_winner
                    or int(opposite_reordered.argmin()) != expected_opposite
                ):
                    raise RuntimeError("candidate display permutation changed a task winner")
                replayed = tuple(
                    manual_candidate_trajectory(specification, int(q)) for q in indices.tolist()
                )
                replayed_positions = torch.stack([trajectory.positions for trajectory in replayed])
                replayed_velocities = torch.stack(
                    [trajectory.velocities for trajectory in replayed]
                )
                if not (
                    torch.equal(replayed_positions, positions[indices])
                    and torch.equal(replayed_velocities, velocities[indices])
                ):
                    raise RuntimeError("candidate permutation changed schedule-keyed truth")
                permutation_invariance_count += 1

            _update_tensor_digest(goal_cost_digest, f"{prefix_label}/goal", goal)
            _update_tensor_digest(
                goal_cost_digest,
                f"{prefix_label}/opposite_goal",
                opposite_goal,
            )
            _update_tensor_digest(
                goal_cost_digest,
                f"{prefix_label}/canonical_costs",
                canonical_costs,
            )
            _update_tensor_digest(
                goal_cost_digest,
                f"{prefix_label}/opposite_costs",
                opposite_costs,
            )

            action_delay = specification.action_delay_seconds
            minimum_action_after_anchor = min(minimum_action_after_anchor, action_delay)
            minimum_action_before_terminal = min(
                minimum_action_before_terminal,
                HORIZONS_SECONDS[-1] - action_delay,
            )
            minimum_action_query_separation = min(
                minimum_action_query_separation,
                min(abs(action_delay - query) for query in HORIZONS_SECONDS),
            )
            if action_delay < MINIMUM_ACTION_AFTER_ANCHOR_SECONDS:
                raise RuntimeError("an action is too close to the causal anchor")
            if minimum_action_query_separation + 1.0e-12 < MINIMUM_ACTION_QUERY_SEPARATION_SECONDS:
                raise RuntimeError("an action coincides with or approaches a query time")

            action_frame = ANCHOR_FRAME_INDEX + specification.action_delay_numerator
            action_controlled_event_count += int(events[:, :, controlled_row].sum())
            action_distractor_event_count += int(events[:, :, 1 - controlled_row].sum())
            if not bool(events[:, action_frame, controlled_row].all()):
                raise RuntimeError("right-continuous action event is absent at its exact frame")
            if int(events.sum()) != CANDIDATES_PER_BUNDLE:
                raise RuntimeError("each candidate must contain exactly one action event")
            zero_impulse_control_event_count += int(prefix.action_events.sum())
            zero_impulse_control_state_delta = max(
                zero_impulse_control_state_delta,
                float((prefix.positions - no_action_position).abs().max()),
                float((prefix.velocities - no_action_velocity).abs().max()),
            )
            pre_action_slice = slice(0, action_frame)
            maximum_pre_action_state_delta = max(
                maximum_pre_action_state_delta,
                float(
                    (positions[:, pre_action_slice] - no_action_position[None, pre_action_slice])
                    .abs()
                    .max()
                ),
                float(
                    (velocities[:, pre_action_slice] - no_action_velocity[None, pre_action_slice])
                    .abs()
                    .max()
                ),
            )
            maximum_distractor_state_delta = max(
                maximum_distractor_state_delta,
                float(
                    (
                        positions[:, :, 1 - controlled_row]
                        - no_action_position[None, :, 1 - controlled_row]
                    )
                    .abs()
                    .max()
                ),
                float(
                    (
                        velocities[:, :, 1 - controlled_row]
                        - no_action_velocity[None, :, 1 - controlled_row]
                    )
                    .abs()
                    .max()
                ),
            )
            terminal_position_effect = (
                positions[:, -1, controlled_row] - no_action_position[-1, controlled_row]
            ).abs()
            terminal_velocity_effect = (
                velocities[:, -1, controlled_row] - no_action_velocity[-1, controlled_row]
            ).abs()
            for axis in range(3):
                minimum_terminal_position_effect_by_axis[axis] = min(
                    minimum_terminal_position_effect_by_axis[axis],
                    float(terminal_position_effect[:, axis].min()),
                )
                minimum_terminal_velocity_effect_by_axis[axis] = min(
                    minimum_terminal_velocity_effect_by_axis[axis],
                    float(terminal_velocity_effect[:, axis].min()),
                )

            for q, trajectory in enumerate(trajectories):
                candidate_values = (
                    ("positions", trajectory.positions),
                    ("velocities", trajectory.velocities),
                    ("action_events", trajectory.action_events.to(torch.uint8)),
                )
                substep_values = (
                    ("substep_positions", trajectory.substep_positions),
                    ("substep_velocities", trajectory.substep_velocities),
                    (
                        "substep_action_events",
                        trajectory.substep_action_events.to(torch.uint8),
                    ),
                )
                signature = _tensor_sequence_sha256((*candidate_values, *substep_values))
                candidate_signatures.add(
                    canonical_sha256(
                        {
                            "schedule": action_schedule_metadata(candidates[q]),
                            "truth_sha256": signature,
                        }
                    )
                )
                for label, value in candidate_values:
                    _update_tensor_digest(
                        candidate_state_digest,
                        f"{prefix_label}/q_{q}/{label}",
                        value,
                    )
                for label, value in substep_values:
                    _update_tensor_digest(
                        substep_contact_proof_digest,
                        f"{prefix_label}/q_{q}/{label}",
                        value,
                    )
                margins = _continuous_candidate_margins(trajectory)
                minimum_world_surface_gap = min(
                    minimum_world_surface_gap,
                    margins["minimum_world_surface_gap_m"],
                )
                minimum_world_boundary = min(
                    minimum_world_boundary,
                    margins["minimum_world_boundary_clearance_m"],
                )
                minimum_floor_clearance = min(
                    minimum_floor_clearance,
                    margins["minimum_floor_clearance_m"],
                )
                minimum_speed = min(minimum_speed, margins["minimum_speed_mps"])
                maximum_speed = max(maximum_speed, margins["maximum_speed_mps"])
                floor_contact_count += int(margins["floor_contact_count"])
                sleep_candidate_count += int(margins["sleep_candidate_count"])

            if minimum_world_surface_gap < MINIMUM_WORLD_SURFACE_GAP_M:
                raise RuntimeError("a candidate future approaches object contact")
            if minimum_world_boundary < MINIMUM_WORLD_BOUNDARY_M:
                raise RuntimeError("a candidate future approaches a world boundary")
            if (
                minimum_floor_clearance <= BOUNDARY_CONTACT_TOLERANCE_M
                or floor_contact_count
                or sleep_candidate_count
            ):
                raise RuntimeError("a candidate future permits floor contact or sleep")

            control_key = (
                split,
                specification.primitive_index,
                specification.camera_stratum,
            )
            if control_key not in no_action_projected_centres:
                control_geometry = _independent_geometry_batch(
                    no_action_position.unsqueeze(0),
                    cameras[specification.camera_stratum],
                )
                no_action_projected_centres[control_key] = control_geometry["ellipse_centre"][
                    0
                ].clone()
            control_centres = no_action_projected_centres[control_key]

            geometry = _independent_geometry_batch(
                positions,
                cameras[specification.camera_stratum],
            )
            if bool((geometry["full_mask"][:, :, 0] & geometry["full_mask"][:, :, 1]).any()):
                raise RuntimeError("independent rays found overlapping candidate silhouettes")
            if not torch.equal(geometry["visible"], geometry["support"]):
                raise RuntimeError("independent rays found candidate occlusion")
            if int(geometry["support"].min()) < MINIMUM_FULL_SUPPORT_PIXELS:
                raise RuntimeError("a candidate future has insufficient raster support")
            if not bool(geometry["conic_valid"].all()):
                raise RuntimeError("a candidate conic became invalid")
            centres = geometry["ellipse_centre"]
            enclosing = geometry["ellipse_enclosing_radius"]
            centre_separation = torch.linalg.vector_norm(
                centres[:, :, 0] - centres[:, :, 1],
                dim=-1,
            )
            projected_centre_step = torch.linalg.vector_norm(
                centres[:, 1:] - centres[:, :-1],
                dim=-1,
            )
            prefix_projected_centre_step = projected_centre_step[:, : HISTORY_FRAME_COUNT - 1]
            control_centre_delta = control_centres[1:] - control_centres[:-1]
            no_action_projected_centre_step = torch.linalg.vector_norm(
                control_centre_delta,
                dim=-1,
            )
            incremental_acted_projected_centre_step = torch.linalg.vector_norm(
                (centres[:, 1:] - centres[:, :-1]) - control_centre_delta[None],
                dim=-1,
            )
            conic_gap = centre_separation - enclosing.sum(dim=-1)
            extrema = geometry["ellipse_coordinate_extrema"]
            conic_boundary = torch.stack(
                (
                    extrema[..., 0],
                    (IMAGE_SIZE[1] - 1.0) - extrema[..., 1],
                    extrema[..., 2],
                    (IMAGE_SIZE[0] - 1.0) - extrema[..., 3],
                ),
                dim=-1,
            )
            certified_gap = conic_gap - CONIC_PIXEL_SAFETY_TOLERANCE
            certified_boundary = conic_boundary - CONIC_PIXEL_SAFETY_TOLERANCE
            if float(certified_gap.min()) < MINIMUM_CONIC_GAP_LOWER_BOUND_PIXELS:
                raise RuntimeError("a candidate future violates the conic gap gate")
            if float(certified_boundary.min()) < MINIMUM_CONIC_BOUNDARY_CLEARANCE_PIXELS:
                raise RuntimeError("a candidate future violates the conic boundary gate")
            if float(prefix_projected_centre_step.max()) > MAXIMUM_PROJECTED_CENTRE_STEP_PIXELS:
                raise RuntimeError("a causal prefix violates inherited image continuity")
            if float(no_action_projected_centre_step.max()) > MAXIMUM_PROJECTED_CENTRE_STEP_PIXELS:
                raise RuntimeError("a no-action control violates inherited image continuity")
            if float(projected_centre_step.max()) > MAXIMUM_ACTED_PROJECTED_CENTRE_STEP_PIXELS:
                raise RuntimeError("an acted future violates its separate image-continuity gate")
            minimum_support = min(minimum_support, float(geometry["support"].min()))
            minimum_conic_gap = min(minimum_conic_gap, float(certified_gap.min()))
            minimum_boundary = min(minimum_boundary, float(certified_boundary.min()))
            minimum_projected_handle_centre_separation = min(
                minimum_projected_handle_centre_separation,
                float(centre_separation.min()),
            )
            maximum_prefix_projected_centre_step = max(
                maximum_prefix_projected_centre_step,
                float(prefix_projected_centre_step.max()),
            )
            maximum_no_action_projected_centre_step = max(
                maximum_no_action_projected_centre_step,
                float(no_action_projected_centre_step.max()),
            )
            maximum_acted_projected_centre_step = max(
                maximum_acted_projected_centre_step,
                float(projected_centre_step.max()),
            )
            maximum_incremental_acted_projected_centre_step = max(
                maximum_incremental_acted_projected_centre_step,
                float(incremental_acted_projected_centre_step.max()),
            )
            minimum_hit_discriminant = min(
                minimum_hit_discriminant,
                float(geometry["discriminant"][geometry["full_mask"]].min()),
            )
            maximum_conic_boundary_residual = max(
                maximum_conic_boundary_residual,
                float(geometry["conic_boundary_residual"].max()),
            )
            maximum_conic_centre_residual = max(
                maximum_conic_centre_residual,
                float(geometry["conic_centre_residual"].max()),
            )
            maximum_conic_shape_residual = max(
                maximum_conic_shape_residual,
                float(geometry["conic_shape_residual"].max()),
            )
            for q in range(CANDIDATES_PER_BUNDLE):
                geometry_values = (
                    ("full_mask", geometry["full_mask"][q].to(torch.uint8)),
                    ("visible_mask", geometry["visible_mask"][q].to(torch.uint8)),
                    ("winner", geometry["winner"][q]),
                    ("depth_buffer", geometry["depth_buffer"][q]),
                    ("support", geometry["support"][q]),
                    ("conic_matrix", geometry["conic_matrix"][q]),
                    ("ellipse_centre", geometry["ellipse_centre"][q]),
                    (
                        "ellipse_coordinate_extrema",
                        geometry["ellipse_coordinate_extrema"][q],
                    ),
                    (
                        "ellipse_enclosing_radius",
                        geometry["ellipse_enclosing_radius"][q],
                    ),
                    ("conic_valid", geometry["conic_valid"][q].to(torch.uint8)),
                )
                for label, value in geometry_values:
                    _update_tensor_digest(
                        conic_raster_digest,
                        f"{prefix_label}/q_{q}/{label}",
                        value,
                    )
                _digest_field(
                    combined_digest,
                    f"{prefix_label}/q_{q}/metadata",
                    _canonical_json_bytes(
                        {
                            "scene": scene_metadata(specification),
                            "candidate": candidate_metadata(candidates[q]),
                            "state_sha256": _tensor_sequence_sha256(
                                (
                                    ("positions", positions[q]),
                                    ("velocities", velocities[q]),
                                    (
                                        "substep_positions",
                                        trajectories[q].substep_positions,
                                    ),
                                    (
                                        "substep_velocities",
                                        trajectories[q].substep_velocities,
                                    ),
                                    (
                                        "substep_action_events",
                                        trajectories[q].substep_action_events.to(torch.uint8),
                                    ),
                                )
                            ),
                            "geometry_sha256": _tensor_sequence_sha256(geometry_values),
                            "cost": float(canonical_costs[q]),
                            "opposite_cost": float(opposite_costs[q]),
                        }
                    ),
                )

    expected_balance = {
        "primitive": {str(index): 16 for index in range(PRIMITIVES_PER_SPLIT)},
        "role": {str(index): 32 for index in range(HANDLE_ROLES)},
        "camera": {str(index): 8 for index in range(CAMERA_STRATA)},
        "palette": {"false": 32, "true": 32},
        "winner": {str(index): 8 for index in range(CANDIDATES_PER_BUNDLE)},
    }
    for split in SPLITS:
        if primitive_histogram[split] != expected_balance["primitive"]:
            raise RuntimeError("primitive balance differs from the frozen manifest")
        if role_histogram[split] != expected_balance["role"]:
            raise RuntimeError("handle-role balance differs from the frozen manifest")
        if camera_histogram[split] != expected_balance["camera"]:
            raise RuntimeError("camera balance differs from the frozen manifest")
        if palette_histogram[split] != expected_balance["palette"]:
            raise RuntimeError("palette balance differs from the frozen manifest")
        if winner_histogram[split] != expected_balance["winner"]:
            raise RuntimeError("canonical winner balance differs from the frozen manifest")
        if pi_winner_display_histogram[split] != expected_balance["winner"]:
            raise RuntimeError("pi display winner balance differs from the frozen manifest")
        if rho_winner_display_histogram[split] != expected_balance["winner"]:
            raise RuntimeError("rho display winner balance differs from the frozen manifest")

    if maximum_pre_action_state_delta != 0.0 or maximum_distractor_state_delta != 0.0:
        raise RuntimeError("an action changed pre-action or distractor state")
    if zero_impulse_control_event_count != 0 or zero_impulse_control_state_delta != 0.0:
        raise RuntimeError("the no-action control emitted an action effect")
    if action_controlled_event_count != TOTAL_CANDIDATES:
        raise RuntimeError("controlled-object action event count differs from 2048")
    if action_distractor_event_count != 0:
        raise RuntimeError("a candidate emitted a distractor action event")
    if floor_contact_count != 0 or sleep_candidate_count != 0:
        raise RuntimeError("floor-contact-derived sleep proof is not exactly zero")

    trace_sha256 = {
        "metadata": metadata_sha256,
        "prefix": prefix_digest.hexdigest(),
        "camera": camera_digest.hexdigest(),
        "ordered_schedules": ordered_schedule_digest.hexdigest(),
        "unordered_schedules": unordered_schedule_digest.hexdigest(),
        "candidate_state": candidate_state_digest.hexdigest(),
        "substep_contact_proof": substep_contact_proof_digest.hexdigest(),
        "goals_costs": goal_cost_digest.hexdigest(),
        "conic_raster": conic_raster_digest.hexdigest(),
        "combined": combined_digest.hexdigest(),
    }
    unsigned: dict[str, Any] = {
        "input_binding_sha256": input_binding_sha256,
        "normalised_scene_source_sha256": fresh_input["normalised_scene_source_sha256"],
        "descriptor_sha256": descriptor_sha256,
        "manifest_sha256": manifest_sha256,
        "trace_sha256": trace_sha256,
        "source_bindings": source_bindings,
        "determinism_scope": determinism_scope,
        "family_scene_signature_sha256": family_scene_signature(),
        "primitive_profile_count": TOTAL_PRIMITIVE_PROFILES,
        "bundle_count": TOTAL_BUNDLES,
        "candidate_count": TOTAL_CANDIDATES,
        "unique_prefix_profile_count": len(profile_signatures),
        "unique_candidate_schedule_set_count": len(schedule_set_signatures),
        "unique_schedule_keyed_candidate_truth_count": len(candidate_signatures),
        "unique_goal_pair_count": len(goal_signatures),
        "role_twin_pair_count": role_twin_pair_count,
        "permutation_invariance_check_count": permutation_invariance_count,
        "primitive_histogram": primitive_histogram,
        "handle_role_histogram": role_histogram,
        "camera_stratum_histogram": camera_histogram,
        "palette_swap_histogram": palette_histogram,
        "canonical_winner_histogram": winner_histogram,
        "pi_winner_display_histogram": pi_winner_display_histogram,
        "rho_winner_display_histogram": rho_winner_display_histogram,
        "minimum_unique_winner_terminal_position_sse_gap": (minimum_unique_winner_sse_gap),
        "minimum_opposite_goal_unique_winner_terminal_position_sse_gap": (
            minimum_opposite_goal_unique_winner_sse_gap
        ),
        "minimum_terminal_action_position_effect_m_by_axis": (
            minimum_terminal_position_effect_by_axis
        ),
        "minimum_terminal_action_velocity_effect_mps_by_axis": (
            minimum_terminal_velocity_effect_by_axis
        ),
        "minimum_impulse_energy_belief_mass_units_squared_m2ps2": (minimum_impulse_energy),
        "maximum_impulse_energy_belief_mass_units_squared_m2ps2": (maximum_impulse_energy),
        "minimum_action_after_anchor_seconds": minimum_action_after_anchor,
        "minimum_action_query_separation_seconds": minimum_action_query_separation,
        "minimum_action_before_terminal_seconds": minimum_action_before_terminal,
        "maximum_pre_action_state_delta": maximum_pre_action_state_delta,
        "maximum_distractor_state_delta": maximum_distractor_state_delta,
        "right_continuous_controlled_action_event_count": (action_controlled_event_count),
        "right_continuous_distractor_action_event_count": (action_distractor_event_count),
        "zero_impulse_control_action_event_count": zero_impulse_control_event_count,
        "zero_impulse_control_state_delta": zero_impulse_control_state_delta,
        "minimum_matched_handle_prototype_cosine": matched_handle_cosine,
        "minimum_cross_handle_prototype_cosine_distance": (cross_handle_cosine_distance),
        "minimum_projected_handle_centre_separation_pixels": (
            minimum_projected_handle_centre_separation
        ),
        "maximum_causal_prefix_projected_centre_step_pixels": (
            maximum_prefix_projected_centre_step
        ),
        "maximum_no_action_projected_centre_step_pixels": (maximum_no_action_projected_centre_step),
        "maximum_acted_projected_centre_step_pixels": (maximum_acted_projected_centre_step),
        "maximum_incremental_acted_projected_centre_step_pixels": (
            maximum_incremental_acted_projected_centre_step
        ),
        "minimum_full_support_pixels": minimum_support,
        "minimum_conic_enclosing_circle_gap_lower_bound_pixels": minimum_conic_gap,
        "minimum_conic_coordinate_extrema_boundary_clearance_pixels": minimum_boundary,
        "minimum_world_surface_gap_m": minimum_world_surface_gap,
        "minimum_world_boundary_clearance_m": minimum_world_boundary,
        "minimum_floor_clearance_m": minimum_floor_clearance,
        "boundary_contact_tolerance_m": BOUNDARY_CONTACT_TOLERANCE_M,
        "floor_contact_count": floor_contact_count,
        "sleep_candidate_count": sleep_candidate_count,
        "sleep_law": {
            "simulator_default_sleep_speed_mps": FROZEN_SIMULATOR_SLEEP_SPEED_MPS,
            "simulator_default_sleep_after_seconds": (FROZEN_SIMULATOR_SLEEP_AFTER_SECONDS),
            "orpheus_config_default_sleep_speed_mps": (FROZEN_ORPHEUS_CONFIG_SLEEP_SPEED_MPS),
            "requires_floor_contact": True,
        },
        "minimum_episode_speed_mps": minimum_speed,
        "maximum_episode_speed_mps": maximum_speed,
        "minimum_hit_discriminant": minimum_hit_discriminant,
        "maximum_conic_relative_boundary_residual": maximum_conic_boundary_residual,
        "maximum_conic_centre_residual": maximum_conic_centre_residual,
        "maximum_conic_shape_residual": maximum_conic_shape_residual,
        "expected_contact_event_count": 0,
        "expected_boundary_event_count": 0,
        "expected_collision_event_count": 0,
        "expected_removal_event_count": 0,
        "expected_sleep_event_count": 0,
        "public_camera_calls_on_formal_values": 0,
        "public_renderer_calls_on_formal_values": 0,
        "public_physics_calls_on_formal_values": 0,
        "public_dynamics_calls_on_formal_values": 0,
        "public_runtime_calls_on_formal_values": 0,
        "public_planner_calls_on_formal_values": 0,
        "camera_metrics": camera_metrics,
        "digest_recipe": _descriptor_unsigned()["digest_recipe"],
    }
    result = _canonical_json_bytes(
        {
            **unsigned,
            "certificate_sha256": canonical_sha256(unsigned),
        }
    )
    if input_binding_sha256 != _fresh_certificate_input_binding()["binding_sha256"]:
        raise RuntimeError("certificate inputs changed during exhaustive computation")
    return result


def _computed_scene_family_certificate_bytes() -> bytes:
    """Freshly validate the exact cache key before and after every lookup."""

    fresh_input = _fresh_certificate_input_binding()
    binding_sha256 = fresh_input["binding_sha256"]
    result = _computed_scene_family_certificate_bytes_cached(binding_sha256)
    if binding_sha256 != _fresh_certificate_input_binding()["binding_sha256"]:
        raise RuntimeError("certificate inputs changed across the cache lookup")
    return result


def _computed_scene_family_certificate() -> dict[str, Any]:
    return json.loads(_computed_scene_family_certificate_bytes())


def scene_family_certificate() -> dict[str, Any]:
    """Return the frozen exhaustive certificate, failing closed on drift."""

    if (
        FROZEN_NORMALIZED_SCENE_SOURCE_SHA256 == "UNFROZEN"
        or FROZEN_INPUT_BINDING_SHA256 == "UNFROZEN"
        or FROZEN_CERTIFICATE_SHA256 == "UNFROZEN"
        or FROZEN_DESCRIPTOR_SHA256 == "UNFROZEN"
        or any(value == "UNFROZEN" for value in FROZEN_MANIFEST_SHA256.values())
        or any(value == "UNFROZEN" for value in FROZEN_TRACE_SHA256.values())
    ):
        raise RuntimeError("known-action scene certificate has not been frozen")
    fresh_input = _fresh_certificate_input_binding()
    if fresh_input["normalised_scene_source_sha256"] != FROZEN_NORMALIZED_SCENE_SOURCE_SHA256:
        raise RuntimeError("known-action scene source differs from the source freeze")
    if fresh_input["binding_sha256"] != FROZEN_INPUT_BINDING_SHA256:
        raise RuntimeError("known-action complete input binding differs from the source freeze")
    result = _computed_scene_family_certificate()
    if result["input_binding_sha256"] != FROZEN_INPUT_BINDING_SHA256:
        raise RuntimeError("known-action result input binding differs from the source freeze")
    if result["normalised_scene_source_sha256"] != FROZEN_NORMALIZED_SCENE_SOURCE_SHA256:
        raise RuntimeError("known-action result source differs from the source freeze")
    if result["descriptor_sha256"] != FROZEN_DESCRIPTOR_SHA256:
        raise RuntimeError("known-action descriptor differs from the source freeze")
    if result["manifest_sha256"] != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("known-action manifests differ from the source freeze")
    if result["trace_sha256"] != FROZEN_TRACE_SHA256:
        raise RuntimeError("known-action traces differ from the source freeze")
    if result["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256:
        raise RuntimeError("known-action certificate differs from the source freeze")
    return copy.deepcopy(result)


__all__ = [
    "BUNDLES_PER_SPLIT",
    "CAMERA_STRATA",
    "CANDIDATES_PER_BUNDLE",
    "FROZEN_CERTIFICATE_SHA256",
    "FROZEN_DESCRIPTOR_SHA256",
    "FROZEN_INPUT_BINDING_SHA256",
    "FROZEN_MANIFEST_SHA256",
    "FROZEN_NORMALIZED_SCENE_SOURCE_SHA256",
    "FROZEN_TRACE_SHA256",
    "HANDLE_ROLES",
    "HORIZONS_SECONDS",
    "KnownActionCandidateSpecification",
    "KnownActionSceneSpecification",
    "PRIMITIVES_PER_SPLIT",
    "PureActionTrajectory",
    "PureCameraFrame",
    "SPLITS",
    "TOTAL_BUNDLES",
    "TOTAL_CANDIDATES",
    "action_schedule_metadata",
    "candidate_costs",
    "candidate_metadata",
    "candidate_order",
    "candidate_specification",
    "canonical_sha256",
    "canonical_to_display",
    "certificate_descriptor",
    "display_to_canonical",
    "family_scene_signature",
    "manual_candidate_trajectory",
    "manual_prefix_trajectory",
    "observable_task_metadata",
    "collate_observable_tasks",
    "pure_orbital_camera_frame",
    "role_twin_ordinal",
    "scene_family_certificate",
    "scene_metadata",
    "scene_signature",
    "scene_specification",
    "split_manifest",
    "split_scene_signatures",
    "terminal_goal",
    "world_impulse_action",
]
