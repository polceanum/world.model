"""Seedless scene family and certificate for identifiable per-object drag.

This module owns only deterministic scene construction and source-time
certification.  It does not construct observation packets, call the runtime,
read simulator truth into an estimator, select integer seeds, or touch run
artifacts.  A scene is addressed solely by a conceptual split and an ordinal.

The family crosses sixteen rational initial geometries with an exact
low/high-drag slot counterfactual and eight known orbital-camera strata.  The
certificate independently traces every formal raster geometry.  Public
physics/renderer equivalence is exercised only in a separate, already-consumed
old-drag/cardinal feasibility namespace; bound public source hashes connect
those checks to the governed independent recurrence and ray mathematics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from world_model.simulator import (
    CameraFrame,
    PhysicsConfig,
    SphereState,
    advance_spheres,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    render_spheres,
    world_to_camera,
)
from world_model.utils.version import SIMULATOR_VERSION

SceneSplit = Literal["development", "selector", "confirmation", "final_test"]
EvidenceRole = Literal["governed_development", "held_out_preflight_only"]

SPLITS: tuple[SceneSplit, ...] = (
    "development",
    "selector",
    "confirmation",
    "final_test",
)
SPLIT_INDEX: dict[SceneSplit, int] = {split: index for index, split in enumerate(SPLITS)}

PRIMITIVES_PER_SPLIT = 4
COUNTERFACTUALS_PER_PRIMITIVE = 2
CAMERA_STRATA = 8
SCENES_PER_SPLIT = PRIMITIVES_PER_SPLIT * COUNTERFACTUALS_PER_PRIMITIVE * CAMERA_STRATA
TOTAL_SCENES = len(SPLITS) * SCENES_PER_SPLIT
FRAME_COUNT = 56
FRAME_RATE_HZ = 20
PHYSICS_RATE_HZ = 120
SUBSTEPS_PER_FRAME = PHYSICS_RATE_HZ // FRAME_RATE_HZ
PHYSICAL_SUBSTEP_COUNT = (FRAME_COUNT - 1) * SUBSTEPS_PER_FRAME
HISTORY_FRAME_COUNT = 16

IMAGE_SIZE = (64, 64)
WORLD_BOUNDS = ((-2.25, 2.25), (0.0, 3.25), (-1.5, 1.5))
SPHERE_RADIUS_M = 0.21

CAMERA_PHASES_RADIANS = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
CONSUMED_PUBLIC_CAMERA_PHASE_OFFSETS_RADIANS = (0.0, math.pi / 8.0, math.pi / 4.0)
SPLIT_PHASE_OFFSETS_RADIANS = (
    math.pi / 16.0,
    3.0 * math.pi / 16.0,
    5.0 * math.pi / 16.0,
    7.0 * math.pi / 16.0,
)
SPLIT_DRAG_NUMERATOR_SHIFTS = (1, 2, 3, 4)
FORMAL_DEVELOPMENT_SPLIT: SceneSplit = "development"
HELD_OUT_PREFLIGHT_SPLITS: tuple[SceneSplit, ...] = (
    "selector",
    "confirmation",
    "final_test",
)
SPLIT_EVIDENCE_ROLE: dict[SceneSplit, EvidenceRole] = {
    "development": "governed_development",
    "selector": "held_out_preflight_only",
    "confirmation": "held_out_preflight_only",
    "final_test": "held_out_preflight_only",
}
CAMERA_DIRECTIONS = (-1, 1)
CAMERA_TARGET = (0.0, 0.95, 0.0)
CAMERA_RADIUS_M = 4.6
CAMERA_HEIGHT_M = 2.15
CAMERA_ANGULAR_SPEED_RAD_S = 0.24
CAMERA_VERTICAL_FOV_DEGREES = 48.0

PALETTE = ((0.92, 0.20, 0.14), (0.14, 0.84, 0.30))
GF4_M2 = (0, 2, 3, 1)
GF4_M3 = (0, 3, 1, 2)
LOW_DRAG_NUMERATORS = (8, 15, 22, 29)
HIGH_DRAG_NUMERATORS = (45, 52, 59, 66)
SPLIT_LOW_DRAG_NUMERATORS = tuple(
    tuple(value + shift for value in LOW_DRAG_NUMERATORS) for shift in SPLIT_DRAG_NUMERATOR_SHIFTS
)
SPLIT_HIGH_DRAG_NUMERATORS = tuple(
    tuple(value - shift for value in HIGH_DRAG_NUMERATORS) for shift in SPLIT_DRAG_NUMERATOR_SHIFTS
)
DRAG_DENOMINATOR = 200
DRAG_ESTIMATOR_BOUNDS = (0.01, 0.36)
MINIMUM_DRAG_EXCITATION_M = 0.015

MINIMUM_FULL_SUPPORT_PIXELS = 20
MINIMUM_CONTINUOUS_GAP_PIXELS = 4.0
MINIMUM_IMAGE_BOUNDARY_PIXELS = 6.0
MINIMUM_WORLD_SURFACE_GAP_M = 1.0
MINIMUM_WORLD_BOUNDARY_M = 0.15
MINIMUM_INITIAL_SPEED_MPS = 0.05
MINIMUM_EPISODE_SPEED_MPS = 0.02
MAXIMUM_EPISODE_SPEED_MPS = 0.071
MINIMUM_DRAG_SEPARATION_PER_S = 0.049
MAXIMUM_CAMERA_CALIBRATION_ERROR = 2.0e-5
MINIMUM_CAMERA_STEP_ANGLE_RADIANS = 0.01198
MAXIMUM_CAMERA_STEP_ANGLE_RADIANS = 0.01202
MINIMUM_CAMERA_TRANSLATION_STEP_M = 0.0551
MAXIMUM_CAMERA_TRANSLATION_STEP_M = 0.0553
MAXIMUM_PUBLIC_PHYSICS_ERROR = 2.0e-7

PUBLIC_RENDERER_ALGORITHM_VERSION = "sphere_world_v7/stable_metric_ray_sphere_near_root"
PUBLIC_RENDERER_SOURCE_SHA256 = "76ae74a9c0da3f002b4e2b2234228f5dff8c1117965721bf2328df658e548876"
PUBLIC_PHYSICS_ALGORITHM_VERSION = "sphere_world_v7/exact_linear_drag_then_contact_substeps"
PUBLIC_PHYSICS_SOURCE_SHA256 = "99a69c80ef87ce15a783a43b1342112600431a6b33d0aa95dacaac148202c02f"
PUBLIC_CAMERA_ALGORITHM_VERSION = "sphere_world_v7/float32_look_at_and_rigid_inverse"
PUBLIC_CAMERA_SOURCE_SHA256 = "23c9798d412a44e9f8b7bea57ef7598e469dfeea087e1515a6c27d51e53caa27"

# Filled only after the exhaustive independent/public proof passes.  These
# constants bind source metadata and generated traces, never data artifacts.
FROZEN_METADATA_SHA256 = "ad84b9227d7d189d6fa714c3e8366c82300527f57c250994a20a450c77470e2f"
FROZEN_PHYSICAL_TRACE_SHA256 = "7ca9d199a524e23f68ecafbab616b4addd28ca44bf8f37c103339c9b530942cc"
FROZEN_CAMERA_TRACE_SHA256 = "7a035e1df82b0cc6698be03b968c411215c048ba6a64fd65d75e0f66cd8e1db8"
FROZEN_RASTER_TRACE_SHA256 = "59d0284bb15701411c59eccfc48dde3e9d0e042ff013a0a5b6627a45b916be52"
FROZEN_COMBINED_TRACE_SHA256 = "229ce55076ef2a85ca775a736176e40b8202b34c5a5bcf8e719c09a91f18748d"
FROZEN_CERTIFICATE_SHA256 = "588c8fe2e2baa38dcb097a012b5ec6517b3ce9733a7c8d068e71c98a1c5f5f9e"

FROZEN_SPLIT_PHYSICAL_TRACE_SHA256 = {
    "development": "ae8d479f262f35e529a228cee94d78e4ccbbae19593226a3850acbcf67840d25",
    "selector": "26299922ea1f7e8399c9ad755478eb6ecc08be14c2d012e1306a06be10b0c3e6",
    "confirmation": "c21dc7d3df24e0a8ad758a9d6d9fd23508c986986591ffe7129d2b902792de8a",
    "final_test": "095e4aaa4652512bde4a0513f51b9684b10b549cdc6b39dd1edc60feece697e9",
}
FROZEN_SPLIT_CAMERA_TRACE_SHA256 = {
    "development": "bb551dbe498bd644543820829d0d4d6b7a00712e06bbec70265ce7647704a6fc",
    "selector": "e8eb8b8773042b6ceb2f39f07a3d3e2c4171d02410b8d2a1cde574c950ca62c4",
    "confirmation": "e61f45aa989bf393240fc97342e381cb71fb02286d561b0009567ae7fd2392c0",
    "final_test": "4b7d2879e8e667444f1104c73f50d14af54a7ca91da2384ab8ead1d42f2bb751",
}
FROZEN_SPLIT_RASTER_TRACE_SHA256 = {
    "development": "64b57bf2b5ecc686aa87f2bfb3bec5848b4619ff03fcd7b901116c623e8ff453",
    "selector": "70414a2d67d76b56f58b478a5803c42d12c5715c01f8a155ee9f4e1b4c48933c",
    "confirmation": "bd405924f73e1b686eab1e304e190ffeb47c6b99e3b6a66921f4fb73e693de99",
    "final_test": "f6c2f1e9a8b389b2bc64eda1ce58d4ba2883a8c7ef078dd24dd30c50e31e67c3",
}
FROZEN_SPLIT_COMBINED_TRACE_SHA256 = {
    "development": "346d849f69b0bccc5aa2dc85bfe231977ed48a7fcb5cea8bf998ec14848fbd2b",
    "selector": "4634b7926e3a0711cab9201a6d04f1d377c7a3d6fff1f7bdc3141c4a902f39f0",
    "confirmation": "c112e2cc56fbcad67cf8256bf3d88e883ee55f742ebc860288f78c02c1f33714",
    "final_test": "8b653c14dca665c66bfbdb7aad83eac8649c2ed70ccf4393a5ed24043d33d3c9",
}


@dataclass(frozen=True, slots=True)
class IdentifiableDragSceneSpecification:
    """One immutable rational scene description.

    Positions use denominator 1000, velocities denominator 4000, and drag
    coefficients denominator 200.  Storing integer numerators makes the
    constructor metadata exact and keeps tensor materialisation explicit.
    """

    split: SceneSplit
    ordinal: int
    split_index: int
    primitive_index: int
    counterfactual_index: int
    camera_stratum: int
    phase_index: int
    direction_index: int
    direction: int
    phase_offset: float
    theta0: float
    evidence_role: EvidenceRole
    a: int
    b: int
    position_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    velocity_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    low_drag_index: int
    high_drag_index: int
    low_drag_numerator: int
    high_drag_numerator: int
    drag_slot_numerators: tuple[int, int]
    palette_swapped: bool
    albedo: tuple[tuple[float, float, float], tuple[float, float, float]]

    @property
    def position(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Initial positions in metres as immutable Python tuples."""

        return tuple(  # type: ignore[return-value]
            tuple(value / 1000.0 for value in row) for row in self.position_numerators
        )

    @property
    def velocity(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Initial velocities in metres/second as immutable Python tuples."""

        return tuple(  # type: ignore[return-value]
            tuple(value / 4000.0 for value in row) for row in self.velocity_numerators
        )

    @property
    def drag(self) -> tuple[float, float]:
        """Per-object drag coefficients in inverse seconds."""

        return tuple(value / DRAG_DENOMINATOR for value in self.drag_slot_numerators)  # type: ignore[return-value]

    def position_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Materialise initial position without exposing stored mutable state."""

        return torch.tensor(self.position_numerators, dtype=dtype, device=device) / 1000.0

    def velocity_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Materialise initial velocity without exposing stored mutable state."""

        return torch.tensor(self.velocity_numerators, dtype=dtype, device=device) / 4000.0

    def drag_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Materialise per-object ``[2,1]`` drag coefficients."""

        return (
            torch.tensor(self.drag_slot_numerators, dtype=dtype, device=device)[:, None]
            / DRAG_DENOMINATOR
        )

    def albedo_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Materialise the two object colours."""

        return torch.tensor(self.albedo, dtype=dtype, device=device)


@dataclass(frozen=True, slots=True)
class PublicFeasibilitySceneSpecification:
    """Consumed old-drag/cardinal fixture, never a governed scene.

    This nominally separate type prevents public feasibility traces from being
    confused with the four formal split namespaces.
    """

    ordinal: int
    source_family_index: int
    primitive_index: int
    counterfactual_index: int
    camera_stratum: int
    phase_index: int
    direction_index: int
    direction: int
    theta0: float
    position_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    velocity_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    drag_slot_numerators: tuple[int, int]
    palette_swapped: bool
    albedo: tuple[tuple[float, float, float], tuple[float, float, float]]
    evidence_role: Literal["public_feasibility_only"] = "public_feasibility_only"

    def position_tensor(self) -> Tensor:
        return torch.tensor(self.position_numerators, dtype=torch.float32) / 1000.0

    def velocity_tensor(self) -> Tensor:
        return torch.tensor(self.velocity_numerators, dtype=torch.float32) / 4000.0

    def drag_tensor(self) -> Tensor:
        return (
            torch.tensor(self.drag_slot_numerators, dtype=torch.float32)[:, None] / DRAG_DENOMINATOR
        )

    def albedo_tensor(self) -> Tensor:
        return torch.tensor(self.albedo, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class PhysicalTrajectory:
    """Manual float32 free-motion trace at frames and all physics substeps."""

    positions: Tensor
    velocities: Tensor
    substep_positions: Tensor
    substep_velocities: Tensor


def _normalise_split(split: str) -> SceneSplit:
    if type(split) is not str:
        raise TypeError("identifiable-drag split must be a string")
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown identifiable-drag split {split!r}")
    return split  # type: ignore[return-value]


def _ordinal_components(ordinal: int) -> tuple[int, int, int]:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("identifiable-drag scene ordinal must be an integer")
    if not 0 <= ordinal < SCENES_PER_SPLIT:
        raise IndexError(ordinal)
    primitive_index, remainder = divmod(ordinal, COUNTERFACTUALS_PER_PRIMITIVE * CAMERA_STRATA)
    counterfactual_index, camera_stratum = divmod(remainder, CAMERA_STRATA)
    return primitive_index, counterfactual_index, camera_stratum


def _source_geometry(
    source_family_index: int,
    primitive_index: int,
) -> tuple[
    tuple[tuple[int, int, int], tuple[int, int, int]],
    tuple[tuple[int, int, int], tuple[int, int, int]],
    int,
    int,
]:
    permuted_primitive = primitive_index ^ GF4_M3[source_family_index]
    a = 2 * primitive_index - 3
    b = 2 * permuted_primitive - 3
    sign_x = 1 if primitive_index % 2 == 0 else -1
    sign_z = 1 if permuted_primitive % 2 == 0 else -1
    position_numerators = (
        (-650 + 12 * a, 460 + 8 * b, -280 + 8 * b),
        (650 + 10 * b, 1640 + 8 * a, 280 - 8 * a),
    )
    velocity_numerators = (
        (sign_x * (230 + 6 * a), 90 + 5 * b, sign_z * (75 - 4 * a)),
        (sign_x * (205 + 5 * b), 75 + 4 * a, sign_z * (65 - 3 * b)),
    )
    return position_numerators, velocity_numerators, a, b


def scene_specification(split: str, ordinal: int) -> IdentifiableDragSceneSpecification:
    """Construct one scene from ``split`` and ``ordinal`` only; no seed exists."""

    canonical_split = _normalise_split(split)
    primitive_index, counterfactual_index, camera_stratum = _ordinal_components(ordinal)
    split_index = SPLIT_INDEX[canonical_split]
    phase_index, direction_index = divmod(camera_stratum, 2)
    direction = CAMERA_DIRECTIONS[direction_index]

    position_numerators, velocity_numerators, a, b = _source_geometry(split_index, primitive_index)

    low_drag_index = primitive_index ^ split_index
    high_drag_index = primitive_index ^ GF4_M2[split_index]
    low_drag_numerator = SPLIT_LOW_DRAG_NUMERATORS[split_index][low_drag_index]
    high_drag_numerator = SPLIT_HIGH_DRAG_NUMERATORS[split_index][high_drag_index]
    drag_slot_numerators = (
        (low_drag_numerator, high_drag_numerator)
        if counterfactual_index == 0
        else (high_drag_numerator, low_drag_numerator)
    )
    palette_swapped = bool((primitive_index + phase_index + direction_index) & 1)
    albedo = PALETTE[::-1] if palette_swapped else PALETTE
    return IdentifiableDragSceneSpecification(
        split=canonical_split,
        ordinal=ordinal,
        split_index=split_index,
        primitive_index=primitive_index,
        counterfactual_index=counterfactual_index,
        camera_stratum=camera_stratum,
        phase_index=phase_index,
        direction_index=direction_index,
        direction=direction,
        phase_offset=SPLIT_PHASE_OFFSETS_RADIANS[split_index],
        theta0=(CAMERA_PHASES_RADIANS[phase_index] + SPLIT_PHASE_OFFSETS_RADIANS[split_index]),
        evidence_role=SPLIT_EVIDENCE_ROLE[canonical_split],
        a=a,
        b=b,
        position_numerators=position_numerators,
        velocity_numerators=velocity_numerators,
        low_drag_index=low_drag_index,
        high_drag_index=high_drag_index,
        low_drag_numerator=low_drag_numerator,
        high_drag_numerator=high_drag_numerator,
        drag_slot_numerators=drag_slot_numerators,
        palette_swapped=palette_swapped,
        albedo=albedo,
    )


PUBLIC_FEASIBILITY_SCENE_COUNT = len(SPLITS) * SCENES_PER_SPLIT


def public_feasibility_specification(ordinal: int) -> PublicFeasibilitySceneSpecification:
    """Return one consumed old-drag/cardinal fixture in a separate namespace."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("public-feasibility ordinal must be an integer")
    if not 0 <= ordinal < PUBLIC_FEASIBILITY_SCENE_COUNT:
        raise IndexError(ordinal)
    source_family_index, family_ordinal = divmod(ordinal, SCENES_PER_SPLIT)
    primitive_index, counterfactual_index, camera_stratum = _ordinal_components(family_ordinal)
    phase_index, direction_index = divmod(camera_stratum, 2)
    direction = CAMERA_DIRECTIONS[direction_index]
    position_numerators, velocity_numerators, _, _ = _source_geometry(
        source_family_index, primitive_index
    )
    low_index = primitive_index ^ source_family_index
    high_index = primitive_index ^ GF4_M2[source_family_index]
    low = LOW_DRAG_NUMERATORS[low_index]
    high = HIGH_DRAG_NUMERATORS[high_index]
    drag_slot_numerators = (low, high) if counterfactual_index == 0 else (high, low)
    palette_swapped = bool((primitive_index + phase_index + direction_index) & 1)
    return PublicFeasibilitySceneSpecification(
        ordinal=ordinal,
        source_family_index=source_family_index,
        primitive_index=primitive_index,
        counterfactual_index=counterfactual_index,
        camera_stratum=camera_stratum,
        phase_index=phase_index,
        direction_index=direction_index,
        direction=direction,
        theta0=CAMERA_PHASES_RADIANS[phase_index],
        position_numerators=position_numerators,
        velocity_numerators=velocity_numerators,
        drag_slot_numerators=drag_slot_numerators,
        palette_swapped=palette_swapped,
        albedo=PALETTE[::-1] if palette_swapped else PALETTE,
    )


def counterfactual_twin_ordinal(ordinal: int) -> int:
    """Return the ordinal with identical non-drag source geometry and swapped drag."""

    primitive_index, counterfactual_index, camera_stratum = _ordinal_components(ordinal)
    return (
        primitive_index * COUNTERFACTUALS_PER_PRIMITIVE * CAMERA_STRATA
        + (1 - counterfactual_index) * CAMERA_STRATA
        + camera_stratum
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scene_metadata(specification: IdentifiableDragSceneSpecification) -> dict[str, Any]:
    """Return exact JSON-safe constructor metadata for signatures and audits."""

    return {
        "split": specification.split,
        "ordinal": specification.ordinal,
        "split_index": specification.split_index,
        "primitive_index": specification.primitive_index,
        "counterfactual_index": specification.counterfactual_index,
        "camera_stratum": specification.camera_stratum,
        "phase_index": specification.phase_index,
        "direction_index": specification.direction_index,
        "direction": specification.direction,
        "phase_offset": specification.phase_offset,
        "theta0": specification.theta0,
        "evidence_role": specification.evidence_role,
        "a": specification.a,
        "b": specification.b,
        "position_rational": {
            "numerators": specification.position_numerators,
            "denominator": 1000,
        },
        "velocity_rational": {
            "numerators": specification.velocity_numerators,
            "denominator": 4000,
        },
        "drag_rational": {
            "low_index": specification.low_drag_index,
            "high_index": specification.high_drag_index,
            "low_numerator": specification.low_drag_numerator,
            "high_numerator": specification.high_drag_numerator,
            "slot_numerators": specification.drag_slot_numerators,
            "denominator": DRAG_DENOMINATOR,
        },
        "palette_swapped": specification.palette_swapped,
        "albedo": specification.albedo,
    }


def scene_signature(specification: IdentifiableDragSceneSpecification) -> str:
    """Hash one complete source-owned scene specification."""

    if not isinstance(specification, IdentifiableDragSceneSpecification):
        raise TypeError("scene_signature requires an IdentifiableDragSceneSpecification")
    return _canonical_sha256(scene_metadata(specification))


def split_scene_signatures(split: str) -> tuple[str, ...]:
    """Return all 64 ordered scene signatures for one conceptual split."""

    canonical_split = _normalise_split(split)
    return tuple(
        scene_signature(scene_specification(canonical_split, ordinal))
        for ordinal in range(SCENES_PER_SPLIT)
    )


def family_scene_signature() -> str:
    """Hash the complete split-labelled ordered family signature table."""

    return _canonical_sha256({split: split_scene_signatures(split) for split in SPLITS})


def _base_geometry_metadata(specification: IdentifiableDragSceneSpecification) -> dict[str, Any]:
    return {
        "a": specification.a,
        "b": specification.b,
        "position_numerators": specification.position_numerators,
        "velocity_numerators": specification.velocity_numerators,
    }


def _drag_physical_metadata(specification: IdentifiableDragSceneSpecification) -> dict[str, Any]:
    return {
        **_base_geometry_metadata(specification),
        "drag_slot_numerators": specification.drag_slot_numerators,
    }


def _cross_split_match_count(values: dict[str, set[str] | set[int]]) -> int:
    return sum(
        len(values[left] & values[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    )


@lru_cache(maxsize=1)
def _scene_balance_certificate_cached() -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    all_scene_signatures: list[str] = []
    all_base_geometry_signatures: set[str] = set()
    all_drag_physical_signatures: set[str] = set()
    all_split_low_high_pairs: set[tuple[str, int, int]] = set()
    drag_truth_levels_by_split: dict[str, set[int]] = {}
    base_geometry_signatures_by_split: dict[str, set[str]] = {}
    drag_physical_signatures_by_split: dict[str, set[str]] = {}
    counterfactual_pair_count = 0

    for split in SPLITS:
        specifications = [
            scene_specification(split, ordinal) for ordinal in range(SCENES_PER_SPLIT)
        ]
        signatures = [scene_signature(specification) for specification in specifications]
        primitive_histogram = {str(index): 0 for index in range(PRIMITIVES_PER_SPLIT)}
        counterfactual_histogram = {str(index): 0 for index in range(2)}
        camera_histogram = {str(index): 0 for index in range(CAMERA_STRATA)}
        split_index = SPLIT_INDEX[split]
        split_low_levels = SPLIT_LOW_DRAG_NUMERATORS[split_index]
        split_high_levels = SPLIT_HIGH_DRAG_NUMERATORS[split_index]
        low_level_histogram = {str(value): 0 for value in split_low_levels}
        high_level_histogram = {str(value): 0 for value in split_high_levels}
        palette_histogram = {"false": 0, "true": 0}
        slot_drag_histogram = {
            str(slot): {str(value): 0 for value in (*split_low_levels, *split_high_levels)}
            for slot in range(2)
        }
        split_low_high_pairs: set[tuple[int, int]] = set()
        split_drag_truth_levels: set[int] = set()
        split_base_geometry_signatures: set[str] = set()
        split_drag_physical_signatures: set[str] = set()

        for specification in specifications:
            primitive_histogram[str(specification.primitive_index)] += 1
            counterfactual_histogram[str(specification.counterfactual_index)] += 1
            camera_histogram[str(specification.camera_stratum)] += 1
            low_level_histogram[str(specification.low_drag_numerator)] += 1
            high_level_histogram[str(specification.high_drag_numerator)] += 1
            palette_histogram[str(specification.palette_swapped).lower()] += 1
            for slot, value in enumerate(specification.drag_slot_numerators):
                slot_drag_histogram[str(slot)][str(value)] += 1
            base_signature = _canonical_sha256(_base_geometry_metadata(specification))
            drag_physical_signature = _canonical_sha256(_drag_physical_metadata(specification))
            split_drag_truth_levels.update(specification.drag_slot_numerators)
            split_base_geometry_signatures.add(base_signature)
            split_drag_physical_signatures.add(drag_physical_signature)
            all_base_geometry_signatures.add(base_signature)
            all_drag_physical_signatures.add(drag_physical_signature)
            split_low_high_pairs.add(
                (specification.low_drag_numerator, specification.high_drag_numerator)
            )
            all_split_low_high_pairs.add(
                (
                    split,
                    specification.low_drag_numerator,
                    specification.high_drag_numerator,
                )
            )

        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            for camera_stratum in range(CAMERA_STRATA):
                ordinal = primitive_index * 16 + camera_stratum
                first = scene_specification(split, ordinal)
                second = scene_specification(split, counterfactual_twin_ordinal(ordinal))
                if _base_geometry_metadata(first) != _base_geometry_metadata(second):
                    raise RuntimeError("drag counterfactuals do not share exact base geometry")
                if first.drag_slot_numerators != second.drag_slot_numerators[::-1]:
                    raise RuntimeError("drag counterfactuals are not exact slot swaps")
                if (
                    first.camera_stratum != second.camera_stratum
                    or first.phase_offset != second.phase_offset
                    or first.evidence_role != second.evidence_role
                    or first.albedo != second.albedo
                    or first.palette_swapped != second.palette_swapped
                ):
                    raise RuntimeError("drag counterfactuals changed camera or appearance")
                counterfactual_pair_count += 1

        if len(split_drag_truth_levels) != 8:
            raise RuntimeError("each split requires exactly eight distinct numeric drag truths")
        if len(split_base_geometry_signatures) != 4:
            raise RuntimeError("each split requires exactly four distinct base geometries")
        if len(split_drag_physical_signatures) != 8:
            raise RuntimeError("each split requires exactly eight drag-labelled physical sources")
        drag_truth_levels_by_split[split] = split_drag_truth_levels
        base_geometry_signatures_by_split[split] = split_base_geometry_signatures
        drag_physical_signatures_by_split[split] = split_drag_physical_signatures

        per_split[split] = {
            "scene_count": len(specifications),
            "unique_scene_signature_count": len(set(signatures)),
            "primitive_histogram": primitive_histogram,
            "counterfactual_histogram": counterfactual_histogram,
            "camera_stratum_histogram": camera_histogram,
            "low_drag_numerator_histogram": low_level_histogram,
            "high_drag_numerator_histogram": high_level_histogram,
            "slot_drag_numerator_histogram": slot_drag_histogram,
            "palette_swap_histogram": palette_histogram,
            "unique_low_high_pair_count": len(split_low_high_pairs),
            "numeric_drag_truth_count": len(split_drag_truth_levels),
            "numeric_drag_truth_sha256": _canonical_sha256(sorted(split_drag_truth_levels)),
            "base_geometry_count": len(split_base_geometry_signatures),
            "base_geometry_set_sha256": _canonical_sha256(sorted(split_base_geometry_signatures)),
            "drag_labelled_physical_source_count": len(split_drag_physical_signatures),
            "drag_labelled_physical_source_set_sha256": _canonical_sha256(
                sorted(split_drag_physical_signatures)
            ),
            "ordered_scene_signatures_sha256": _canonical_sha256(signatures),
        }
        all_scene_signatures.extend(signatures)

    public_drag_levels = set(LOW_DRAG_NUMERATORS) | set(HIGH_DRAG_NUMERATORS)
    governed_drag_level_public_match_count = sum(
        value in public_drag_levels
        for split_index in range(len(SPLITS))
        for value in (
            *SPLIT_LOW_DRAG_NUMERATORS[split_index],
            *SPLIT_HIGH_DRAG_NUMERATORS[split_index],
        )
    )
    if governed_drag_level_public_match_count:
        raise RuntimeError("a governed drag truth matches a consumed public truth")
    governed_drag_shift_consumed_match_count = sum(
        shift == 0 for shift in SPLIT_DRAG_NUMERATOR_SHIFTS
    )
    if governed_drag_shift_consumed_match_count:
        raise RuntimeError("a governed drag grid uses the consumed public shift")
    governed_drag_truth_count = sum(map(len, drag_truth_levels_by_split.values()))
    governed_drag_truth_unique_count = len(set().union(*drag_truth_levels_by_split.values()))
    base_geometry_source_count = sum(map(len, base_geometry_signatures_by_split.values()))
    drag_physical_source_count = sum(map(len, drag_physical_signatures_by_split.values()))
    drag_truth_cross_split_match_count = _cross_split_match_count(drag_truth_levels_by_split)
    base_geometry_cross_split_match_count = _cross_split_match_count(
        base_geometry_signatures_by_split
    )
    drag_physical_cross_split_match_count = _cross_split_match_count(
        drag_physical_signatures_by_split
    )
    if governed_drag_truth_count != 32 or governed_drag_truth_unique_count != 32:
        raise RuntimeError("governed family requires 32/32 distinct numeric drag truths")
    if base_geometry_source_count != 16 or len(all_base_geometry_signatures) != 16:
        raise RuntimeError("governed family requires 16/16 distinct base geometries")
    if drag_physical_source_count != 32 or len(all_drag_physical_signatures) != 32:
        raise RuntimeError("governed family requires 32/32 drag-labelled physical sources")
    if (
        drag_truth_cross_split_match_count
        or base_geometry_cross_split_match_count
        or drag_physical_cross_split_match_count
    ):
        raise RuntimeError("governed physical source values overlap across splits")
    unsigned = {
        "split_order": SPLITS,
        "per_split": per_split,
        "total_scene_count": len(all_scene_signatures),
        "unique_scene_signature_count": len(set(all_scene_signatures)),
        "base_geometry_count": len(all_base_geometry_signatures),
        "drag_labelled_physical_trajectory_count": len(all_drag_physical_signatures),
        "governed_drag_truth_count": governed_drag_truth_count,
        "governed_drag_truth_unique_count": governed_drag_truth_unique_count,
        "governed_drag_truth_cross_split_match_count": drag_truth_cross_split_match_count,
        "base_geometry_source_count": base_geometry_source_count,
        "base_geometry_cross_split_match_count": base_geometry_cross_split_match_count,
        "drag_labelled_physical_source_count": drag_physical_source_count,
        "drag_labelled_physical_cross_split_match_count": (drag_physical_cross_split_match_count),
        "unique_split_role_low_high_pair_count": len(all_split_low_high_pairs),
        "governed_drag_level_public_match_count": governed_drag_level_public_match_count,
        "governed_drag_shift_consumed_match_count": governed_drag_shift_consumed_match_count,
        "counterfactual_pair_count": counterfactual_pair_count,
        "family_scene_signature_sha256": family_scene_signature(),
    }
    return {**unsigned, "balance_sha256": _canonical_sha256(unsigned)}


def scene_balance_certificate() -> dict[str, Any]:
    """Return an exact balance/uniqueness proof without physics or rendering."""

    return copy.deepcopy(_scene_balance_certificate_cached())


def _camera_frame(theta0: float, direction: int, timestamp: float) -> CameraFrame:
    if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
        raise TypeError("camera timestamp must be a real scalar")
    resolved_timestamp = float(timestamp)
    if not math.isfinite(resolved_timestamp) or resolved_timestamp < 0.0:
        raise ValueError("camera timestamp must be finite and nonnegative")
    theta = theta0 + direction * CAMERA_ANGULAR_SPEED_RAD_S * resolved_timestamp
    position = torch.tensor(
        (
            CAMERA_RADIUS_M * math.sin(theta),
            CAMERA_HEIGHT_M,
            CAMERA_RADIUS_M * math.cos(theta),
        ),
        dtype=torch.float32,
    )
    target = torch.tensor(CAMERA_TARGET, dtype=torch.float32)
    world_from_camera = look_at_world_from_camera(position, target)
    camera_from_world = invert_rigid_transform(world_from_camera)
    frame = CameraFrame(
        timestamp=resolved_timestamp,
        world_from_camera=world_from_camera,
        camera_from_world=camera_from_world,
        intrinsics=make_intrinsics(
            IMAGE_SIZE,
            CAMERA_VERTICAL_FOV_DEGREES,
            dtype=torch.float32,
        ),
        position=position,
        target=target,
    )
    frame.validate()
    return frame


def orbital_camera_frame(
    specification: IdentifiableDragSceneSpecification,
    timestamp: float,
) -> CameraFrame:
    """Evaluate a formal split's fresh known float32 orbital-camera law."""

    if not isinstance(specification, IdentifiableDragSceneSpecification):
        raise TypeError("orbital_camera_frame requires an identifiable-drag scene")
    return _camera_frame(specification.theta0, specification.direction, timestamp)


def public_feasibility_camera_frame(
    specification: PublicFeasibilitySceneSpecification,
    timestamp: float,
) -> CameraFrame:
    """Evaluate a consumed cardinal fixture; this is not a governed scene."""

    if not isinstance(specification, PublicFeasibilitySceneSpecification):
        raise TypeError("public_feasibility_camera_frame requires a feasibility scene")
    return _camera_frame(specification.theta0, specification.direction, timestamp)


def _sphere_state_from_values(
    position: Tensor,
    velocity: Tensor,
    drag: Tensor,
    albedo: Tensor,
) -> SphereState:
    state = SphereState(
        object_id=torch.tensor((0, 1), dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=position,
        velocity=velocity,
        radius=torch.full((2, 1), SPHERE_RADIUS_M, dtype=torch.float32),
        mass=torch.ones((2, 1), dtype=torch.float32),
        restitution=torch.full((2, 1), 0.7, dtype=torch.float32),
        drag=drag,
        friction=torch.full((2, 1), 0.2, dtype=torch.float32),
        albedo=albedo,
        orientation=torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2, dtype=torch.float32),
        angular_velocity=torch.zeros((2, 3), dtype=torch.float32),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    state.validate()
    return state


def initial_sphere_state(specification: IdentifiableDragSceneSpecification) -> SphereState:
    """Materialise the formal two-sphere source state."""

    if not isinstance(specification, IdentifiableDragSceneSpecification):
        raise TypeError("initial_sphere_state requires an identifiable-drag scene")
    return _sphere_state_from_values(
        specification.position_tensor(),
        specification.velocity_tensor(),
        specification.drag_tensor(),
        specification.albedo_tensor(),
    )


def public_feasibility_sphere_state(
    specification: PublicFeasibilitySceneSpecification,
) -> SphereState:
    """Materialise a consumed public-feasibility state in its own namespace."""

    if not isinstance(specification, PublicFeasibilitySceneSpecification):
        raise TypeError("public_feasibility_sphere_state requires a feasibility scene")
    return _sphere_state_from_values(
        specification.position_tensor(),
        specification.velocity_tensor(),
        specification.drag_tensor(),
        specification.albedo_tensor(),
    )


def manual_physical_trajectory(
    specification: IdentifiableDragSceneSpecification,
) -> PhysicalTrajectory:
    """Evaluate the independent exact float32 zero-gravity drag recurrence."""

    if not isinstance(specification, IdentifiableDragSceneSpecification):
        raise TypeError("manual_physical_trajectory requires an identifiable-drag scene")
    return _manual_trajectory(
        specification.position_tensor(),
        specification.velocity_tensor(),
        specification.drag_tensor(),
    )


def _public_feasibility_manual_trajectory(
    specification: PublicFeasibilitySceneSpecification,
) -> PhysicalTrajectory:
    return _manual_trajectory(
        specification.position_tensor(),
        specification.velocity_tensor(),
        specification.drag_tensor(),
    )


def _manual_trajectory(position: Tensor, velocity: Tensor, drag: Tensor) -> PhysicalTrajectory:
    position = position.clone()
    velocity = velocity.clone()
    substep_seconds = 1.0 / PHYSICS_RATE_HZ
    decay = torch.exp(-drag * substep_seconds)
    displacement_coefficient = -torch.expm1(-drag * substep_seconds) / drag
    frame_positions = [position.clone()]
    frame_velocities = [velocity.clone()]
    substep_positions = [position.clone()]
    substep_velocities = [velocity.clone()]
    for substep_index in range(PHYSICAL_SUBSTEP_COUNT):
        position = position + velocity * displacement_coefficient
        velocity = velocity * decay
        substep_positions.append(position.clone())
        substep_velocities.append(velocity.clone())
        if (substep_index + 1) % SUBSTEPS_PER_FRAME == 0:
            frame_positions.append(position.clone())
            frame_velocities.append(velocity.clone())
    if len(frame_positions) != FRAME_COUNT:
        raise AssertionError("manual drag recurrence did not emit exactly 56 frames")
    return PhysicalTrajectory(
        positions=torch.stack(frame_positions),
        velocities=torch.stack(frame_velocities),
        substep_positions=torch.stack(substep_positions),
        substep_velocities=torch.stack(substep_velocities),
    )


def _independent_raster_trace(state: SphereState, camera: CameraFrame) -> dict[str, Tensor]:
    """Trace exact ray/sphere roots independently of the public renderer."""

    height, width = IMAGE_SIZE
    points_camera = world_to_camera(state.position, camera.camera_from_world)
    depth = points_camera[:, 2]
    focal = 0.5 * (camera.intrinsics[0, 0] + camera.intrinsics[1, 1])
    centres = torch.stack(
        (
            camera.intrinsics[0, 0] * points_camera[:, 0] / depth + camera.intrinsics[0, 2],
            camera.intrinsics[1, 1] * points_camera[:, 1] / depth + camera.intrinsics[1, 2],
        ),
        dim=-1,
    )
    centre_normalized = torch.stack(
        (
            2.0 * centres[:, 0] / (width - 1) - 1.0,
            2.0 * centres[:, 1] / (height - 1) - 1.0,
        ),
        dim=-1,
    )
    apparent_radius = focal * state.radius[:, 0] / depth
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ray_x = (pixel_x - camera.intrinsics[0, 2]) / camera.intrinsics[0, 0]
    ray_y = (pixel_y - camera.intrinsics[1, 2]) / camera.intrinsics[1, 1]
    ray_norm_squared = 1.0 + ray_x.square() + ray_y.square()
    ray_dot_center = (
        ray_x.unsqueeze(0) * points_camera[:, 0, None, None]
        + ray_y.unsqueeze(0) * points_camera[:, 1, None, None]
        + points_camera[:, 2, None, None]
    )
    center_cross_ray = torch.stack(
        (
            points_camera[:, 1, None, None] - points_camera[:, 2, None, None] * ray_y.unsqueeze(0),
            points_camera[:, 2, None, None] * ray_x.unsqueeze(0) - points_camera[:, 0, None, None],
            points_camera[:, 0, None, None] * ray_y.unsqueeze(0)
            - points_camera[:, 1, None, None] * ray_x.unsqueeze(0),
        ),
        dim=-1,
    )
    discriminant = ray_norm_squared.unsqueeze(0) * state.radius[
        :, None, None, 0
    ].square() - center_cross_ray.square().sum(dim=-1)
    square_root = discriminant.clamp_min(0.0).sqrt()
    denominator = ray_dot_center + square_root
    constant = (
        points_camera.square().sum(dim=-1)[:, None, None] - state.radius[:, None, None, 0].square()
    )
    surface_depth = constant / denominator.clamp_min(1.0e-12)
    geometric = state.active & (depth > state.radius[:, 0] + 1.0e-4)
    full_mask = (
        geometric[:, None, None]
        & (discriminant >= 0.0)
        & (denominator > 0.0)
        & (surface_depth > 0.0)
        & torch.isfinite(surface_depth)
    )
    ordered = torch.where(full_mask, surface_depth, torch.full_like(surface_depth, torch.inf))
    depth_buffer, winner = ordered.min(dim=0)
    has_object = torch.isfinite(depth_buffer)
    winner = torch.where(has_object, winner.to(torch.int64), torch.full_like(winner, -1))
    visible_mask = full_mask & (winner.unsqueeze(0) == torch.arange(2)[:, None, None])
    support = full_mask.sum(dim=(-2, -1))
    visible_pixels = visible_mask.sum(dim=(-2, -1))
    visible_fraction = visible_pixels.to(torch.float32) / support.clamp_min(1).to(torch.float32)
    return {
        "points_camera": points_camera,
        "centres": centres,
        "centre_normalized": centre_normalized,
        "apparent_radius": apparent_radius,
        "discriminant": discriminant,
        "surface_depth": surface_depth,
        "full_mask": full_mask,
        "winner": winner,
        "visible_mask": visible_mask,
        "depth_buffer": torch.where(has_object, depth_buffer, torch.zeros_like(depth_buffer)),
        "support": support,
        "visible_fraction": visible_fraction,
    }


def _update_tensor_digest(digest: Any, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    array = value.numpy()
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    digest.update(little_endian.tobytes(order="C"))


def _tensor_sequence_sha256(tensors: tuple[Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        _update_tensor_digest(digest, tensor)
    return digest.hexdigest()


def _validate_public_source_bindings() -> tuple[str, str, str]:
    camera_path = Path(__file__).resolve().parents[1] / "simulator" / "camera.py"
    physics_path = Path(__file__).resolve().parents[1] / "simulator" / "physics.py"
    renderer_path = Path(__file__).resolve().parents[1] / "simulator" / "renderer.py"
    camera_sha256 = hashlib.sha256(camera_path.read_bytes()).hexdigest()
    physics_sha256 = hashlib.sha256(physics_path.read_bytes()).hexdigest()
    renderer_sha256 = hashlib.sha256(renderer_path.read_bytes()).hexdigest()
    if SIMULATOR_VERSION != "sphere_world_v7":
        raise RuntimeError("identifiable-drag certificate requires simulator sphere_world_v7")
    if camera_sha256 != PUBLIC_CAMERA_SOURCE_SHA256:
        raise RuntimeError("public camera source differs from the certificate binding")
    if renderer_sha256 != PUBLIC_RENDERER_SOURCE_SHA256:
        raise RuntimeError("public renderer source differs from the certificate binding")
    if physics_sha256 != PUBLIC_PHYSICS_SOURCE_SHA256:
        raise RuntimeError("public physics source differs from the certificate binding")
    return camera_sha256, physics_sha256, renderer_sha256


def _assert_zero_events(events: Any) -> None:
    for name in (
        "pair_contact",
        "pair_collision",
        "pair_impulse",
        "pair_penetration",
        "boundary_contact",
        "boundary_collision",
        "boundary_impulse",
        "boundary_penetration",
        "collision",
        "contact",
        "sleeping",
        "external_impulse",
    ):
        if bool(getattr(events, name).ne(0).any()):
            raise RuntimeError(f"public physics emitted forbidden {name}")
    if bool(events.first_event_offset.ne(-1.0).any()):
        raise RuntimeError("public physics emitted a first-event timestamp")


def _state_with_kinematics(state: SphereState, position: Tensor, velocity: Tensor) -> SphereState:
    return SphereState(
        object_id=state.object_id,
        active=state.active,
        position=position.clone(),
        velocity=velocity.clone(),
        radius=state.radius,
        mass=state.mass,
        restitution=state.restitution,
        drag=state.drag,
        friction=state.friction,
        albedo=state.albedo,
        orientation=state.orientation,
        angular_velocity=state.angular_velocity,
        sleeping=state.sleeping,
        sleep_counter=state.sleep_counter,
    )


def _state_at(
    specification: IdentifiableDragSceneSpecification,
    position: Tensor,
    velocity: Tensor,
) -> SphereState:
    return _state_with_kinematics(initial_sphere_state(specification), position, velocity)


def _public_feasibility_state_at(
    specification: PublicFeasibilitySceneSpecification,
    position: Tensor,
    velocity: Tensor,
) -> SphereState:
    return _state_with_kinematics(
        public_feasibility_sphere_state(specification), position, velocity
    )


def _raster_digest_fields(trace: dict[str, Tensor]) -> tuple[Tensor, ...]:
    return (
        trace["full_mask"].to(torch.uint8),
        trace["winner"],
        trace["visible_mask"].to(torch.uint8),
        trace["depth_buffer"],
        trace["visible_fraction"],
        trace["centres"],
        trace["centre_normalized"],
        trace["apparent_radius"],
    )


def _public_raster_trace(rendered: Any) -> dict[str, Tensor]:
    return {
        "full_mask": rendered.full_mask,
        "winner": rendered.instance_slot_map,
        "visible_mask": rendered.visible_mask,
        "depth_buffer": rendered.depth_buffer,
        "visible_fraction": rendered.visible_fraction,
        "centres": rendered.projected_center_pixels,
        "centre_normalized": rendered.projected_center,
        "apparent_radius": rendered.apparent_radius,
    }


@lru_cache(maxsize=1)
def _scene_family_certificate_cached() -> dict[str, Any]:
    balance = _scene_balance_certificate_cached()
    camera_source_sha256, physics_source_sha256, renderer_source_sha256 = (
        _validate_public_source_bindings()
    )
    metadata_table = [
        scene_metadata(scene_specification(split, ordinal))
        for split in SPLITS
        for ordinal in range(SCENES_PER_SPLIT)
    ]
    metadata_sha256 = _canonical_sha256(metadata_table)
    physical_digest = hashlib.sha256()
    public_feasibility_physical_digest = hashlib.sha256()
    camera_digest = hashlib.sha256()
    public_feasibility_camera_digest = hashlib.sha256()
    independent_raster_digest = hashlib.sha256()
    public_feasibility_independent_raster_digest = hashlib.sha256()
    public_raster_digest = hashlib.sha256()
    public_rgb_digest = hashlib.sha256()
    combined_digest = hashlib.sha256()
    split_physical_digests = {split: hashlib.sha256() for split in SPLITS}
    split_camera_digests = {split: hashlib.sha256() for split in SPLITS}
    split_raster_digests = {split: hashlib.sha256() for split in SPLITS}
    split_combined_digests = {split: hashlib.sha256() for split in SPLITS}
    physical_trace_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    raster_trace_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    combined_trace_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}

    physics_config = PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=WORLD_BOUNDS,
        max_substep=1.0 / PHYSICS_RATE_HZ,
        solver_iterations=2,
    )
    physical_trajectories: dict[tuple[str, int, int], PhysicalTrajectory] = {}
    public_trajectories: dict[tuple[str, int, int], tuple[Tensor, Tensor]] = {}
    public_feasibility_trajectories: dict[tuple[int, int, int], PhysicalTrajectory] = {}
    camera_frames: dict[tuple[SceneSplit, int], tuple[CameraFrame, ...]] = {}
    public_feasibility_camera_frames: dict[int, tuple[CameraFrame, ...]] = {}

    maximum_public_position_error = 0.0
    maximum_public_velocity_error = 0.0
    public_physics_substeps = 0
    minimum_world_surface_gap = math.inf
    minimum_world_boundary = math.inf
    minimum_initial_speed = math.inf
    minimum_episode_speed = math.inf
    maximum_episode_speed = 0.0
    minimum_history_displacement = math.inf
    minimum_excitation = math.inf
    minimum_drag = math.inf
    maximum_drag = 0.0
    minimum_drag_separation = math.inf

    bounds = torch.tensor(WORLD_BOUNDS, dtype=torch.float32)
    for split in SPLITS:
        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            for counterfactual_index in range(COUNTERFACTUALS_PER_PRIMITIVE):
                ordinal = primitive_index * 16 + counterfactual_index * 8
                specification = scene_specification(split, ordinal)
                key = (split, primitive_index, counterfactual_index)
                manual = manual_physical_trajectory(specification)
                physical_trajectories[key] = manual
                # Formal scenes remain behind the governed access boundary:
                # only the independent recurrence is evaluated here.
                public_trajectories[key] = (manual.positions, manual.velocities)

                substep_position = manual.substep_positions
                substep_velocity = manual.substep_velocities
                surface_gap = (
                    torch.linalg.vector_norm(
                        substep_position[:, 0] - substep_position[:, 1], dim=-1
                    )
                    - 2.0 * SPHERE_RADIUS_M
                )
                world_boundary = torch.minimum(
                    substep_position - SPHERE_RADIUS_M - bounds[:, 0],
                    bounds[:, 1] - substep_position - SPHERE_RADIUS_M,
                )
                speeds = torch.linalg.vector_norm(substep_velocity, dim=-1)
                initial_speeds = torch.linalg.vector_norm(substep_velocity[0], dim=-1)
                history_displacement = torch.linalg.vector_norm(
                    manual.positions[HISTORY_FRAME_COUNT - 1] - manual.positions[0], dim=-1
                )
                anchor = manual.positions[HISTORY_FRAME_COUNT - 1]
                excitation = (
                    (manual.positions[:HISTORY_FRAME_COUNT] - anchor)
                    .square()
                    .sum(dim=-1)
                    .mean(dim=0)
                ).sqrt()
                drag = specification.drag_tensor()[:, 0]
                minimum_world_surface_gap = min(minimum_world_surface_gap, float(surface_gap.min()))
                minimum_world_boundary = min(minimum_world_boundary, float(world_boundary.min()))
                minimum_initial_speed = min(minimum_initial_speed, float(initial_speeds.min()))
                minimum_episode_speed = min(minimum_episode_speed, float(speeds.min()))
                maximum_episode_speed = max(maximum_episode_speed, float(speeds.max()))
                minimum_history_displacement = min(
                    minimum_history_displacement, float(history_displacement.min())
                )
                minimum_excitation = min(minimum_excitation, float(excitation.min()))
                minimum_drag = min(minimum_drag, float(drag.min()))
                maximum_drag = max(maximum_drag, float(drag.max()))
                minimum_drag_separation = min(
                    minimum_drag_separation, float((drag[0] - drag[1]).abs())
                )
                if float(surface_gap.min()) < MINIMUM_WORLD_SURFACE_GAP_M:
                    raise RuntimeError("identifiable-drag trajectory approaches object contact")
                if float(world_boundary.min()) < MINIMUM_WORLD_BOUNDARY_M:
                    raise RuntimeError("identifiable-drag trajectory approaches a world boundary")
                if float(initial_speeds.min()) < MINIMUM_INITIAL_SPEED_MPS:
                    raise RuntimeError("identifiable-drag initial speed is too small")
                if (
                    float(speeds.min()) < MINIMUM_EPISODE_SPEED_MPS
                    or float(speeds.max()) > MAXIMUM_EPISODE_SPEED_MPS
                ):
                    raise RuntimeError("identifiable-drag episode speed is outside the rung")
                if float(excitation.min()) < MINIMUM_DRAG_EXCITATION_M:
                    raise RuntimeError("identifiable-drag history lacks excitation")
                if float((drag[0] - drag[1]).abs()) < MINIMUM_DRAG_SEPARATION_PER_S:
                    raise RuntimeError(
                        "identifiable-drag object coefficients are insufficiently separated"
                    )
                if not bool(
                    ((drag > DRAG_ESTIMATOR_BOUNDS[0]) & (drag < DRAG_ESTIMATOR_BOUNDS[1])).all()
                ):
                    raise RuntimeError("identifiable-drag coefficient touches estimator bounds")

                physical_trace_values = (
                    manual.positions,
                    manual.velocities,
                    manual.substep_positions,
                    manual.substep_velocities,
                )
                for value in physical_trace_values:
                    _update_tensor_digest(physical_digest, value)
                    _update_tensor_digest(split_physical_digests[split], value)
                physical_trace_signatures_by_split[split].add(
                    _tensor_sequence_sha256(physical_trace_values)
                )

    if len(physical_trajectories) != 32:
        raise RuntimeError("certificate requires exactly 32 drag-labelled trajectories")
    physical_trace_count_by_split = {
        split: len(signatures) for split, signatures in physical_trace_signatures_by_split.items()
    }
    physical_trace_unique_count = len(set().union(*physical_trace_signatures_by_split.values()))
    physical_trace_cross_split_match_count = _cross_split_match_count(
        physical_trace_signatures_by_split
    )
    if physical_trace_count_by_split != {split: 8 for split in SPLITS}:
        raise RuntimeError("each split requires eight distinct numeric physical traces")
    if physical_trace_unique_count != 32 or physical_trace_cross_split_match_count:
        raise RuntimeError("formal physical trace tensors overlap across splits")

    # Public-API recurrence equivalence is exercised only on the already
    # consumed old-drag/cardinal feasibility namespace.  No formal split state
    # crosses this call boundary.
    for source_family_index in range(len(SPLITS)):
        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            for counterfactual_index in range(COUNTERFACTUALS_PER_PRIMITIVE):
                ordinal = (
                    source_family_index * SCENES_PER_SPLIT
                    + primitive_index * 16
                    + counterfactual_index * 8
                )
                specification = public_feasibility_specification(ordinal)
                key = (source_family_index, primitive_index, counterfactual_index)
                manual = _public_feasibility_manual_trajectory(specification)
                public_feasibility_trajectories[key] = manual
                state = public_feasibility_sphere_state(specification)
                public_positions = [state.position.clone()]
                public_velocities = [state.velocity.clone()]
                for _frame_index in range(1, FRAME_COUNT):
                    state, events = advance_spheres(
                        state,
                        1.0 / FRAME_RATE_HZ,
                        physics_config,
                        external_impulse=torch.zeros((2, 3), dtype=torch.float32),
                    )
                    if events.substeps != SUBSTEPS_PER_FRAME:
                        raise RuntimeError("public physics changed the six-substep protocol")
                    _assert_zero_events(events)
                    public_physics_substeps += events.substeps
                    public_positions.append(state.position.clone())
                    public_velocities.append(state.velocity.clone())
                public_position = torch.stack(public_positions)
                public_velocity = torch.stack(public_velocities)
                maximum_public_position_error = max(
                    maximum_public_position_error,
                    float((public_position - manual.positions).abs().max()),
                )
                maximum_public_velocity_error = max(
                    maximum_public_velocity_error,
                    float((public_velocity - manual.velocities).abs().max()),
                )
                if (
                    maximum_public_position_error > MAXIMUM_PUBLIC_PHYSICS_ERROR
                    or maximum_public_velocity_error > MAXIMUM_PUBLIC_PHYSICS_ERROR
                ):
                    raise RuntimeError("public physics differs from the manual drag recurrence")
                for value in (
                    manual.positions,
                    manual.velocities,
                    public_position,
                    public_velocity,
                ):
                    _update_tensor_digest(public_feasibility_physical_digest, value)

    if len(public_feasibility_trajectories) != 32:
        raise RuntimeError("public feasibility physics did not cover 32 trajectories")

    maximum_camera_inverse_error = 0.0
    maximum_camera_orthonormality_error = 0.0
    maximum_camera_radius_error = 0.0
    maximum_camera_height_error = 0.0
    maximum_camera_target_error = 0.0
    maximum_camera_intrinsics_error = 0.0
    maximum_camera_position_binding_error = 0.0
    minimum_adjacent_camera_angle = math.inf
    maximum_adjacent_camera_angle = 0.0
    minimum_adjacent_camera_translation = math.inf
    maximum_adjacent_camera_translation = 0.0
    reference_intrinsics = make_intrinsics(
        IMAGE_SIZE,
        CAMERA_VERTICAL_FOV_DEGREES,
        dtype=torch.float32,
    )
    reference_target = torch.tensor(CAMERA_TARGET, dtype=torch.float32)
    governed_consumed_phase_match_count = sum(
        math.isclose(
            SPLIT_PHASE_OFFSETS_RADIANS[SPLIT_INDEX[split]],
            consumed_offset,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for split in SPLITS
        for consumed_offset in CONSUMED_PUBLIC_CAMERA_PHASE_OFFSETS_RADIANS
    )
    if governed_consumed_phase_match_count:
        raise RuntimeError("a governed camera phase offset was already publicly consumed")
    camera_trace_digests: dict[tuple[SceneSplit, int], str] = {}
    for split in SPLITS:
        for camera_stratum in range(CAMERA_STRATA):
            specification = scene_specification(split, camera_stratum)
            frames = tuple(
                orbital_camera_frame(specification, frame_index / FRAME_RATE_HZ)
                for frame_index in range(FRAME_COUNT)
            )
            camera_frames[(split, camera_stratum)] = frames
            stratum_digest = hashlib.sha256()
            previous: CameraFrame | None = None
            for camera in frames:
                identity = camera.world_from_camera @ camera.camera_from_world
                rotation = camera.world_from_camera[:3, :3]
                maximum_camera_inverse_error = max(
                    maximum_camera_inverse_error,
                    float((identity - torch.eye(4)).abs().max()),
                )
                maximum_camera_orthonormality_error = max(
                    maximum_camera_orthonormality_error,
                    float((rotation.T @ rotation - torch.eye(3)).abs().max()),
                )
                maximum_camera_radius_error = max(
                    maximum_camera_radius_error,
                    abs(float(torch.linalg.vector_norm(camera.position[[0, 2]])) - CAMERA_RADIUS_M),
                )
                maximum_camera_height_error = max(
                    maximum_camera_height_error,
                    abs(float(camera.position[1]) - CAMERA_HEIGHT_M),
                )
                maximum_camera_target_error = max(
                    maximum_camera_target_error,
                    float((camera.target - reference_target).abs().max()),
                )
                maximum_camera_intrinsics_error = max(
                    maximum_camera_intrinsics_error,
                    float((camera.intrinsics - reference_intrinsics).abs().max()),
                )
                maximum_camera_position_binding_error = max(
                    maximum_camera_position_binding_error,
                    float((camera.world_from_camera[:3, 3] - camera.position).abs().max()),
                )
                if previous is not None:
                    previous_horizontal = previous.position[[0, 2]]
                    current_horizontal = camera.position[[0, 2]]
                    cosine = torch.dot(previous_horizontal, current_horizontal) / (
                        torch.linalg.vector_norm(previous_horizontal)
                        * torch.linalg.vector_norm(current_horizontal)
                    )
                    angle = float(torch.acos(cosine.clamp(-1.0, 1.0)))
                    translation = float(
                        torch.linalg.vector_norm(camera.position - previous.position)
                    )
                    minimum_adjacent_camera_angle = min(minimum_adjacent_camera_angle, angle)
                    maximum_adjacent_camera_angle = max(maximum_adjacent_camera_angle, angle)
                    minimum_adjacent_camera_translation = min(
                        minimum_adjacent_camera_translation, translation
                    )
                    maximum_adjacent_camera_translation = max(
                        maximum_adjacent_camera_translation, translation
                    )
                previous = camera
                for digest in (camera_digest, split_camera_digests[split], stratum_digest):
                    _update_tensor_digest(digest, camera.world_from_camera)
                    _update_tensor_digest(digest, camera.camera_from_world)
                    _update_tensor_digest(digest, camera.intrinsics)
            camera_trace_digests[(split, camera_stratum)] = stratum_digest.hexdigest()

    camera_trace_signatures_by_split = {
        split: {
            camera_trace_digests[(split, camera_stratum)] for camera_stratum in range(CAMERA_STRATA)
        }
        for split in SPLITS
    }
    camera_trace_count_by_split = {
        split: len(signatures) for split, signatures in camera_trace_signatures_by_split.items()
    }
    camera_trace_unique_count = len(set().union(*camera_trace_signatures_by_split.values()))
    camera_trace_cross_split_match_count = _cross_split_match_count(
        camera_trace_signatures_by_split
    )
    if camera_trace_count_by_split != {split: 8 for split in SPLITS}:
        raise RuntimeError("each split requires eight distinct numeric camera traces")
    if camera_trace_unique_count != 32 or camera_trace_cross_split_match_count:
        raise RuntimeError("formal camera trace tensors overlap across splits")

    public_cardinal_trace_digests: set[str] = set()
    for camera_stratum in range(CAMERA_STRATA):
        specification = public_feasibility_specification(camera_stratum)
        frames = tuple(
            public_feasibility_camera_frame(specification, frame_index / FRAME_RATE_HZ)
            for frame_index in range(FRAME_COUNT)
        )
        public_feasibility_camera_frames[camera_stratum] = frames
        stratum_digest = hashlib.sha256()
        for camera in frames:
            for digest in (public_feasibility_camera_digest, stratum_digest):
                _update_tensor_digest(digest, camera.world_from_camera)
                _update_tensor_digest(digest, camera.camera_from_world)
                _update_tensor_digest(digest, camera.intrinsics)
        public_cardinal_trace_digests.add(stratum_digest.hexdigest())
    governed_cardinal_trace_match_count = sum(
        camera_trace_digests[(split, camera_stratum)] in public_cardinal_trace_digests
        for split in SPLITS
        for camera_stratum in range(CAMERA_STRATA)
    )
    if governed_cardinal_trace_match_count:
        raise RuntimeError("a governed offset camera trace matches a public cardinal trace")

    calibration_error = max(
        maximum_camera_inverse_error,
        maximum_camera_orthonormality_error,
        maximum_camera_radius_error,
        maximum_camera_height_error,
        maximum_camera_target_error,
        maximum_camera_intrinsics_error,
        maximum_camera_position_binding_error,
    )
    if calibration_error > MAXIMUM_CAMERA_CALIBRATION_ERROR:
        raise RuntimeError("orbital camera calibration exceeds the frozen tolerance")
    if not (
        minimum_adjacent_camera_angle >= MINIMUM_CAMERA_STEP_ANGLE_RADIANS
        and maximum_adjacent_camera_angle <= MAXIMUM_CAMERA_STEP_ANGLE_RADIANS
        and minimum_adjacent_camera_translation >= MINIMUM_CAMERA_TRANSLATION_STEP_M
        and maximum_adjacent_camera_translation <= MAXIMUM_CAMERA_TRANSLATION_STEP_M
    ):
        raise RuntimeError("orbital camera step lies outside the frozen interval")

    minimum_full_support = math.inf
    minimum_continuous_gap = math.inf
    minimum_image_boundary = math.inf
    minimum_visible_fraction = math.inf
    maximum_projected_centre_step = 0.0
    renderer_mismatch_count = 0
    public_renderer_frame_count = 0
    formal_public_renderer_call_count = 0
    overlap_frame_count = 0
    raster_frame_count = 0
    counterfactual_non_drag_source_mismatch_count = 0
    counterfactual_camera_mismatch_count = 0

    for split in SPLITS:
        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            for camera_stratum in range(CAMERA_STRATA):
                first_ordinal = primitive_index * 16 + camera_stratum
                first = scene_specification(split, first_ordinal)
                second = scene_specification(split, counterfactual_twin_ordinal(first_ordinal))
                if (
                    _base_geometry_metadata(first) != _base_geometry_metadata(second)
                    or first.albedo != second.albedo
                    or first.palette_swapped != second.palette_swapped
                ):
                    counterfactual_non_drag_source_mismatch_count += 1
                for frame_index in range(FRAME_COUNT):
                    first_camera = orbital_camera_frame(first, frame_index / FRAME_RATE_HZ)
                    second_camera = orbital_camera_frame(second, frame_index / FRAME_RATE_HZ)
                    if not (
                        torch.equal(first_camera.world_from_camera, second_camera.world_from_camera)
                        and torch.equal(first_camera.intrinsics, second_camera.intrinsics)
                    ):
                        counterfactual_camera_mismatch_count += 1

        for ordinal in range(SCENES_PER_SPLIT):
            specification = scene_specification(split, ordinal)
            physical_key = (
                split,
                specification.primitive_index,
                specification.counterfactual_index,
            )
            positions, velocities = public_trajectories[physical_key]
            previous_centres: Tensor | None = None
            scene_raster_digest = hashlib.sha256()
            scene_combined_digest = hashlib.sha256()
            for frame_index in range(FRAME_COUNT):
                camera = camera_frames[(split, specification.camera_stratum)][frame_index]
                state = _state_at(specification, positions[frame_index], velocities[frame_index])
                trace = _independent_raster_trace(state, camera)
                formal_public_renderer_call_count += 0
                overlap = bool((trace["full_mask"][0] & trace["full_mask"][1]).any())
                overlap_frame_count += int(overlap)
                if overlap:
                    raise RuntimeError(
                        "identifiable-drag certificate found overlapping silhouettes"
                    )
                if not bool(trace["visible_fraction"].eq(1.0).all()):
                    raise RuntimeError("identifiable-drag certificate found partial visibility")
                if bool((trace["support"] < MINIMUM_FULL_SUPPORT_PIXELS).any()):
                    raise RuntimeError("identifiable-drag certificate found insufficient support")
                centres = trace["centres"]
                radii = trace["apparent_radius"]
                gap = torch.linalg.vector_norm(centres[0] - centres[1]) - radii.sum()
                boundary = torch.stack(
                    (
                        centres[:, 0] - radii,
                        (IMAGE_SIZE[1] - 1) - centres[:, 0] - radii,
                        centres[:, 1] - radii,
                        (IMAGE_SIZE[0] - 1) - centres[:, 1] - radii,
                    )
                )
                if float(gap) < MINIMUM_CONTINUOUS_GAP_PIXELS:
                    raise RuntimeError("identifiable-drag silhouettes lack continuous separation")
                if float(boundary.min()) < MINIMUM_IMAGE_BOUNDARY_PIXELS:
                    raise RuntimeError("identifiable-drag silhouette approaches image boundary")
                minimum_full_support = min(minimum_full_support, float(trace["support"].min()))
                minimum_continuous_gap = min(minimum_continuous_gap, float(gap))
                minimum_image_boundary = min(minimum_image_boundary, float(boundary.min()))
                minimum_visible_fraction = min(
                    minimum_visible_fraction, float(trace["visible_fraction"].min())
                )
                if previous_centres is not None:
                    maximum_projected_centre_step = max(
                        maximum_projected_centre_step,
                        float(torch.linalg.vector_norm(centres - previous_centres, dim=-1).max()),
                    )
                previous_centres = centres
                for value in _raster_digest_fields(trace):
                    _update_tensor_digest(independent_raster_digest, value)
                    _update_tensor_digest(split_raster_digests[split], value)
                    _update_tensor_digest(scene_raster_digest, value)
                combined_values = (
                    state.position,
                    state.velocity,
                    state.drag,
                    state.albedo,
                    camera.world_from_camera,
                    camera.camera_from_world,
                    camera.intrinsics,
                    trace["full_mask"].to(torch.uint8),
                    trace["winner"],
                    trace["depth_buffer"],
                )
                for value in combined_values:
                    _update_tensor_digest(combined_digest, value)
                    _update_tensor_digest(split_combined_digests[split], value)
                    _update_tensor_digest(scene_combined_digest, value)
                raster_frame_count += 1
            raster_trace_signatures_by_split[split].add(scene_raster_digest.hexdigest())
            combined_trace_signatures_by_split[split].add(scene_combined_digest.hexdigest())

    raster_trace_count_by_split = {
        split: len(signatures) for split, signatures in raster_trace_signatures_by_split.items()
    }
    combined_trace_count_by_split = {
        split: len(signatures) for split, signatures in combined_trace_signatures_by_split.items()
    }
    raster_trace_unique_count = len(set().union(*raster_trace_signatures_by_split.values()))
    combined_trace_unique_count = len(set().union(*combined_trace_signatures_by_split.values()))
    raster_trace_cross_split_match_count = _cross_split_match_count(
        raster_trace_signatures_by_split
    )
    combined_trace_cross_split_match_count = _cross_split_match_count(
        combined_trace_signatures_by_split
    )
    expected_scene_trace_counts = {split: SCENES_PER_SPLIT for split in SPLITS}
    if raster_trace_count_by_split != expected_scene_trace_counts:
        raise RuntimeError("each split requires 64 distinct independent raster traces")
    if combined_trace_count_by_split != expected_scene_trace_counts:
        raise RuntimeError("each split requires 64 distinct combined scene traces")
    if raster_trace_unique_count != TOTAL_SCENES or raster_trace_cross_split_match_count:
        raise RuntimeError("formal raster trace tensors overlap across splits")
    if combined_trace_unique_count != TOTAL_SCENES or combined_trace_cross_split_match_count:
        raise RuntimeError("formal combined trace tensors overlap across splits")

    # Renderer equivalence is deliberately confined to the separate consumed
    # old-drag/cardinal feasibility namespace.  Formal development and all
    # protected splits above use independent raster mathematics only.
    public_feasibility_raster_frame_count = 0
    for ordinal in range(PUBLIC_FEASIBILITY_SCENE_COUNT):
        specification = public_feasibility_specification(ordinal)
        physical_key = (
            specification.source_family_index,
            specification.primitive_index,
            specification.counterfactual_index,
        )
        trajectory = public_feasibility_trajectories[physical_key]
        camera_sequence = public_feasibility_camera_frames[specification.camera_stratum]
        for frame_index in range(FRAME_COUNT):
            state = _public_feasibility_state_at(
                specification,
                trajectory.positions[frame_index],
                trajectory.velocities[frame_index],
            )
            camera = camera_sequence[frame_index]
            independent = _independent_raster_trace(state, camera)
            rendered = render_spheres(state, camera, IMAGE_SIZE, noise_std=0.0)
            public_trace = _public_raster_trace(rendered)
            mismatch = any(
                not torch.equal(expected, actual)
                for expected, actual in zip(
                    _raster_digest_fields(independent),
                    _raster_digest_fields(public_trace),
                    strict=True,
                )
            )
            renderer_mismatch_count += int(mismatch)
            if bool((independent["full_mask"][0] & independent["full_mask"][1]).any()):
                raise RuntimeError("public feasibility fixture has overlapping silhouettes")
            if not bool(independent["visible_fraction"].eq(1.0).all()):
                raise RuntimeError("public feasibility fixture is not fully visible")
            if bool((independent["support"] < MINIMUM_FULL_SUPPORT_PIXELS).any()):
                raise RuntimeError("public feasibility fixture has insufficient support")
            for value in _raster_digest_fields(independent):
                _update_tensor_digest(public_feasibility_independent_raster_digest, value)
            for value in _raster_digest_fields(public_trace):
                _update_tensor_digest(public_raster_digest, value)
            _update_tensor_digest(public_rgb_digest, rendered.rgb)
            public_renderer_frame_count += 1
            public_feasibility_raster_frame_count += 1

    if renderer_mismatch_count:
        raise RuntimeError("independent raster trace differs from the public renderer")
    if counterfactual_non_drag_source_mismatch_count or counterfactual_camera_mismatch_count:
        raise RuntimeError("drag-slot counterfactual changed non-drag geometry")
    if public_feasibility_independent_raster_digest.hexdigest() != public_raster_digest.hexdigest():
        raise RuntimeError("public-feasibility independent/public raster hashes differ")
    if raster_frame_count != TOTAL_SCENES * FRAME_COUNT:
        raise RuntimeError("certificate did not visit every joint scene frame")
    if public_renderer_frame_count != PUBLIC_FEASIBILITY_SCENE_COUNT * FRAME_COUNT:
        raise RuntimeError("public renderer equivalence missed a public-feasibility frame")
    if formal_public_renderer_call_count:
        raise RuntimeError("certificate called the public renderer on a formal scene")

    split_physical_sha256 = {split: split_physical_digests[split].hexdigest() for split in SPLITS}
    split_camera_sha256 = {split: split_camera_digests[split].hexdigest() for split in SPLITS}
    split_raster_sha256 = {split: split_raster_digests[split].hexdigest() for split in SPLITS}
    split_combined_sha256 = {split: split_combined_digests[split].hexdigest() for split in SPLITS}

    unsigned: dict[str, Any] = {
        "protocol": {
            "constructor": "scene_specification(split, ordinal)",
            "integer_seed_surface": False,
            "split_evidence_roles": SPLIT_EVIDENCE_ROLE,
            "formal_development_split": FORMAL_DEVELOPMENT_SPLIT,
            "held_out_preflight_only_camera_families": HELD_OUT_PREFLIGHT_SPLITS,
            "split_phase_offsets_radians": SPLIT_PHASE_OFFSETS_RADIANS,
            "split_drag_numerator_shifts": SPLIT_DRAG_NUMERATOR_SHIFTS,
            "consumed_public_camera_phase_offsets_radians": (
                CONSUMED_PUBLIC_CAMERA_PHASE_OFFSETS_RADIANS
            ),
            "consumed_public_drag_numerator_shift": 0,
            "public_feasibility_constructor": "public_feasibility_specification(ordinal)",
            "public_feasibility_scene_count": PUBLIC_FEASIBILITY_SCENE_COUNT,
            "split_order": SPLITS,
            "scenes_per_split": SCENES_PER_SPLIT,
            "total_scenes": TOTAL_SCENES,
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FRAME_RATE_HZ,
            "physics_rate_hz": PHYSICS_RATE_HZ,
            "history_frame_count": HISTORY_FRAME_COUNT,
            "gravity_mps2": (0.0, 0.0, 0.0),
            "sphere_radius_m": SPHERE_RADIUS_M,
            "world_bounds_m": WORLD_BOUNDS,
            "runtime_packets_constructed": 0,
            "truth_routed_to_runtime": False,
            "public_renderer_algorithm_version": PUBLIC_RENDERER_ALGORITHM_VERSION,
            "public_renderer_source_sha256": renderer_source_sha256,
            "public_physics_algorithm_version": PUBLIC_PHYSICS_ALGORITHM_VERSION,
            "public_physics_source_sha256": physics_source_sha256,
            "public_camera_algorithm_version": PUBLIC_CAMERA_ALGORITHM_VERSION,
            "public_camera_source_sha256": camera_source_sha256,
        },
        "balance": balance,
        "exact_metadata_sha256": metadata_sha256,
        "formal_manual_physical_trace_sha256": physical_digest.hexdigest(),
        "public_feasibility_physical_equivalence_trace_sha256": (
            public_feasibility_physical_digest.hexdigest()
        ),
        "camera_trace_sha256": camera_digest.hexdigest(),
        "public_feasibility_camera_trace_sha256": (public_feasibility_camera_digest.hexdigest()),
        "independent_raster_trace_sha256": independent_raster_digest.hexdigest(),
        "public_feasibility_independent_raster_trace_sha256": (
            public_feasibility_independent_raster_digest.hexdigest()
        ),
        "public_feasibility_public_raster_trace_sha256": public_raster_digest.hexdigest(),
        "public_feasibility_public_rgb_trace_sha256": public_rgb_digest.hexdigest(),
        "ordered_combined_scene_trace_sha256": combined_digest.hexdigest(),
        "formal_trace_bindings": {
            "physical": {
                "per_split_sha256": split_physical_sha256,
                "per_split_trace_count": physical_trace_count_by_split,
                "global_trace_count": sum(physical_trace_count_by_split.values()),
                "global_unique_trace_count": physical_trace_unique_count,
                "cross_split_match_count": physical_trace_cross_split_match_count,
            },
            "camera": {
                "per_split_sha256": split_camera_sha256,
                "per_split_trace_count": camera_trace_count_by_split,
                "global_trace_count": sum(camera_trace_count_by_split.values()),
                "global_unique_trace_count": camera_trace_unique_count,
                "cross_split_match_count": camera_trace_cross_split_match_count,
            },
            "raster": {
                "per_split_sha256": split_raster_sha256,
                "per_split_trace_count": raster_trace_count_by_split,
                "global_trace_count": sum(raster_trace_count_by_split.values()),
                "global_unique_trace_count": raster_trace_unique_count,
                "cross_split_match_count": raster_trace_cross_split_match_count,
            },
            "combined": {
                "per_split_sha256": split_combined_sha256,
                "per_split_trace_count": combined_trace_count_by_split,
                "global_trace_count": sum(combined_trace_count_by_split.values()),
                "global_unique_trace_count": combined_trace_unique_count,
                "cross_split_match_count": combined_trace_cross_split_match_count,
            },
        },
        "physics": {
            "unique_drag_labelled_trajectory_count": len(physical_trajectories),
            "physical_instants_per_trajectory": PHYSICAL_SUBSTEP_COUNT + 1,
            "formal_public_physics_call_count": 0,
            "public_feasibility_trajectory_count": len(public_feasibility_trajectories),
            "public_feasibility_evaluated_substep_count": public_physics_substeps,
            "joint_scene_logical_substep_count": TOTAL_SCENES * PHYSICAL_SUBSTEP_COUNT,
            "formal_analytic_event_count": 0,
            "formal_analytic_contact_count": 0,
            "public_feasibility_event_count": 0,
            "public_feasibility_contact_count": 0,
            "maximum_feasibility_public_manual_position_error_m": (maximum_public_position_error),
            "maximum_feasibility_public_manual_velocity_error_mps": (maximum_public_velocity_error),
            "minimum_world_surface_gap_m": minimum_world_surface_gap,
            "minimum_world_boundary_clearance_m": minimum_world_boundary,
            "minimum_initial_speed_mps": minimum_initial_speed,
            "minimum_episode_speed_mps": minimum_episode_speed,
            "maximum_episode_speed_mps": maximum_episode_speed,
            "minimum_history_displacement_m": minimum_history_displacement,
            "minimum_drag_excitation_m": minimum_excitation,
            "minimum_drag_per_s": minimum_drag,
            "maximum_drag_per_s": maximum_drag,
            "minimum_within_scene_drag_separation_per_s": minimum_drag_separation,
            "minimum_lower_drag_bound_margin_per_s": minimum_drag - DRAG_ESTIMATOR_BOUNDS[0],
            "minimum_upper_drag_bound_margin_per_s": DRAG_ESTIMATOR_BOUNDS[1] - maximum_drag,
            "minimum_excitation_margin_m": minimum_excitation - MINIMUM_DRAG_EXCITATION_M,
        },
        "camera": {
            "trace_count": len(camera_frames),
            "strata_per_split": CAMERA_STRATA,
            "governed_trace_cardinal_match_count": governed_cardinal_trace_match_count,
            "governed_consumed_phase_match_count": governed_consumed_phase_match_count,
            "minimum_adjacent_angle_radians": minimum_adjacent_camera_angle,
            "maximum_adjacent_angle_radians": maximum_adjacent_camera_angle,
            "minimum_adjacent_translation_m": minimum_adjacent_camera_translation,
            "maximum_adjacent_translation_m": maximum_adjacent_camera_translation,
            "maximum_inverse_error": maximum_camera_inverse_error,
            "maximum_orthonormality_error": maximum_camera_orthonormality_error,
            "maximum_radius_error_m": maximum_camera_radius_error,
            "maximum_height_error_m": maximum_camera_height_error,
            "maximum_target_error_m": maximum_camera_target_error,
            "maximum_intrinsics_error": maximum_camera_intrinsics_error,
            "maximum_position_binding_error_m": maximum_camera_position_binding_error,
        },
        "raster": {
            "evaluated_frame_count": raster_frame_count,
            "independent_formal_frame_count": raster_frame_count,
            "public_feasibility_independent_frame_count": (public_feasibility_raster_frame_count),
            "public_feasibility_renderer_frame_count": public_renderer_frame_count,
            "formal_public_renderer_call_count": formal_public_renderer_call_count,
            "public_renderer_mismatch_count": renderer_mismatch_count,
            "overlap_frame_count": overlap_frame_count,
            "minimum_visible_fraction": minimum_visible_fraction,
            "minimum_full_support_pixels": minimum_full_support,
            "minimum_continuous_silhouette_gap_pixels": minimum_continuous_gap,
            "minimum_image_boundary_clearance_pixels": minimum_image_boundary,
            "maximum_projected_centre_step_pixels": maximum_projected_centre_step,
        },
        "counterfactual": {
            "pair_count": balance["counterfactual_pair_count"],
            "drag_swap_mismatch_count": 0,
            "non_drag_source_mismatch_count": counterfactual_non_drag_source_mismatch_count,
            "camera_trace_mismatch_count": counterfactual_camera_mismatch_count,
            "palette_depends_on_counterfactual_index": False,
        },
    }
    result = {**unsigned, "certificate_sha256": _canonical_sha256(unsigned)}
    frozen_bindings = {
        "exact_metadata_sha256": FROZEN_METADATA_SHA256,
        "formal_manual_physical_trace_sha256": FROZEN_PHYSICAL_TRACE_SHA256,
        "camera_trace_sha256": FROZEN_CAMERA_TRACE_SHA256,
        "independent_raster_trace_sha256": FROZEN_RASTER_TRACE_SHA256,
        "ordered_combined_scene_trace_sha256": FROZEN_COMBINED_TRACE_SHA256,
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    for name, expected in frozen_bindings.items():
        if not expected:
            raise RuntimeError(f"identifiable-drag frozen {name} binding is empty")
        if result[name] != expected:
            raise RuntimeError(f"identifiable-drag frozen {name} binding changed")
    split_frozen_bindings = {
        "physical": FROZEN_SPLIT_PHYSICAL_TRACE_SHA256,
        "camera": FROZEN_SPLIT_CAMERA_TRACE_SHA256,
        "raster": FROZEN_SPLIT_RASTER_TRACE_SHA256,
        "combined": FROZEN_SPLIT_COMBINED_TRACE_SHA256,
    }
    for trace_name, expected_by_split in split_frozen_bindings.items():
        if set(expected_by_split) != set(SPLITS) or not all(expected_by_split.values()):
            raise RuntimeError(
                f"identifiable-drag frozen per-split {trace_name} binding is incomplete"
            )
        observed = result["formal_trace_bindings"][trace_name]["per_split_sha256"]
        if observed != expected_by_split:
            raise RuntimeError(f"identifiable-drag frozen per-split {trace_name} binding changed")
    return result


def scene_family_certificate() -> dict[str, Any]:
    """Exhaustively certify all 256 seedless scenes across all 56 frames."""

    return copy.deepcopy(_scene_family_certificate_cached())


__all__ = [
    "CAMERA_ANGULAR_SPEED_RAD_S",
    "CAMERA_DIRECTIONS",
    "CAMERA_HEIGHT_M",
    "CAMERA_PHASES_RADIANS",
    "CAMERA_RADIUS_M",
    "CAMERA_STRATA",
    "CAMERA_TARGET",
    "CAMERA_VERTICAL_FOV_DEGREES",
    "COUNTERFACTUALS_PER_PRIMITIVE",
    "CONSUMED_PUBLIC_CAMERA_PHASE_OFFSETS_RADIANS",
    "DRAG_DENOMINATOR",
    "DRAG_ESTIMATOR_BOUNDS",
    "FRAME_COUNT",
    "FRAME_RATE_HZ",
    "FROZEN_CAMERA_TRACE_SHA256",
    "FROZEN_CERTIFICATE_SHA256",
    "FROZEN_COMBINED_TRACE_SHA256",
    "FROZEN_METADATA_SHA256",
    "FROZEN_PHYSICAL_TRACE_SHA256",
    "FROZEN_RASTER_TRACE_SHA256",
    "FROZEN_SPLIT_CAMERA_TRACE_SHA256",
    "FROZEN_SPLIT_COMBINED_TRACE_SHA256",
    "FROZEN_SPLIT_PHYSICAL_TRACE_SHA256",
    "FROZEN_SPLIT_RASTER_TRACE_SHA256",
    "GF4_M2",
    "GF4_M3",
    "HELD_OUT_PREFLIGHT_SPLITS",
    "HIGH_DRAG_NUMERATORS",
    "HISTORY_FRAME_COUNT",
    "IMAGE_SIZE",
    "IdentifiableDragSceneSpecification",
    "LOW_DRAG_NUMERATORS",
    "MINIMUM_DRAG_EXCITATION_M",
    "PALETTE",
    "PHYSICS_RATE_HZ",
    "PRIMITIVES_PER_SPLIT",
    "PhysicalTrajectory",
    "PUBLIC_FEASIBILITY_SCENE_COUNT",
    "PUBLIC_CAMERA_ALGORITHM_VERSION",
    "PUBLIC_CAMERA_SOURCE_SHA256",
    "PUBLIC_PHYSICS_ALGORITHM_VERSION",
    "PUBLIC_PHYSICS_SOURCE_SHA256",
    "PUBLIC_RENDERER_ALGORITHM_VERSION",
    "PUBLIC_RENDERER_SOURCE_SHA256",
    "PublicFeasibilitySceneSpecification",
    "SCENES_PER_SPLIT",
    "SPHERE_RADIUS_M",
    "SPLITS",
    "SPLIT_DRAG_NUMERATOR_SHIFTS",
    "SPLIT_EVIDENCE_ROLE",
    "SPLIT_HIGH_DRAG_NUMERATORS",
    "SPLIT_LOW_DRAG_NUMERATORS",
    "SPLIT_PHASE_OFFSETS_RADIANS",
    "TOTAL_SCENES",
    "WORLD_BOUNDS",
    "counterfactual_twin_ordinal",
    "family_scene_signature",
    "initial_sphere_state",
    "manual_physical_trajectory",
    "orbital_camera_frame",
    "public_feasibility_camera_frame",
    "public_feasibility_specification",
    "public_feasibility_sphere_state",
    "scene_balance_certificate",
    "scene_family_certificate",
    "scene_metadata",
    "scene_signature",
    "scene_specification",
    "split_scene_signatures",
]
