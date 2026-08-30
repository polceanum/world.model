"""Seedless variable-radius scene family and independent source certificate.

This module is deliberately smaller in authority than a qualification harness.
It owns immutable, ordinal-addressed scene descriptions and pure float32
mathematics only.  In particular, it does not construct simulator episodes,
observation packets, runtime inputs, or run artifacts.  Formal scene values
must never be passed to the public physics engine or renderer: the certificate
uses an explicit fixed-drag recurrence and an independent stable ray/sphere
intersection throughout.

The family is a controlled successor to the accepted two-visible orbital RGB-D
rung.  It keeps the accepted position, velocity, fixed-drag, palette, and known
camera laws unchanged while crossing each base geometry with an exact
low/high-radius slot swap.  Thus each counterfactual twin has identical
kinematics, camera calibration, and appearance and differs only in which
physical object owns each radius.
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

SceneSplit = Literal["development", "selector", "confirmation", "final_test"]
EvidenceRole = Literal[
    "governed_development",
    "protected_selector",
    "protected_confirmation",
    "protected_final_test",
]

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

PRIMITIVES_PER_SPLIT = 2
PAIR_VARIANTS_PER_PRIMITIVE = 2
RADIUS_ROLES_PER_PRIMITIVE = 2
CAMERA_STRATA = 8
SCENES_PER_SPLIT = (
    PRIMITIVES_PER_SPLIT * PAIR_VARIANTS_PER_PRIMITIVE * RADIUS_ROLES_PER_PRIMITIVE * CAMERA_STRATA
)
TOTAL_SCENES = len(SPLITS) * SCENES_PER_SPLIT
FRAME_COUNT = 56
FRAME_RATE_HZ = 20
PHYSICS_RATE_HZ = 120
SUBSTEPS_PER_FRAME = PHYSICS_RATE_HZ // FRAME_RATE_HZ
PHYSICAL_SUBSTEP_COUNT = (FRAME_COUNT - 1) * SUBSTEPS_PER_FRAME
HISTORY_FRAME_COUNT = 16

IMAGE_SIZE = (64, 64)
WORLD_BOUNDS = ((-2.25, 2.25), (0.0, 3.25), (-1.5, 1.5))
FIXED_DRAG_NUMERATOR = 1
FIXED_DRAG_DENOMINATOR = 20

CAMERA_PHASES_RADIANS = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
CAMERA_DIRECTIONS = (-1, 1)
CAMERA_TARGET = (0.0, 0.95, 0.0)
CAMERA_RADIUS_M = 4.6
CAMERA_HEIGHT_M = 2.15
CAMERA_ANGULAR_SPEED_RAD_S = 0.24
CAMERA_VERTICAL_FOV_DEGREES = 48.0

PALETTE = ((0.92, 0.20, 0.14), (0.14, 0.84, 0.30))

# These eight physical controls are an explicit balanced subset of the sixteen
# already accepted orbital-rung primitives.  Every a and b level occurs twice.
SPLIT_PRIMITIVE_PAIRS: dict[SceneSplit, tuple[tuple[int, int], ...]] = {
    "development": ((-3, -3), (1, 1)),
    "selector": ((-1, -1), (3, 3)),
    "confirmation": ((-3, 3), (1, -1)),
    "final_test": ((-1, 1), (3, -3)),
}

# Literal copies of accepted rational source rows.  Position denominator is
# 1000 and the normalized velocity denominator is 4000.
ACCEPTED_PRIMITIVE_RATIONAL_ROWS = {
    (-3, -3): (
        ((-468, 394, -312), (435, 1744, 312)),
        ((174, 45, 13), (-178, -35, -15)),
    ),
    (1, 1): (
        ((-444, 402, -296), (455, 1752, 296)),
        ((182, 49, 17), (-170, -31, -11)),
    ),
    (-1, -1): (
        ((-456, 398, -304), (445, 1748, 304)),
        ((178, 47, 15), (-174, -33, -13)),
    ),
    (3, 3): (
        ((-432, 406, -288), (465, 1756, 288)),
        ((186, 51, 19), (-166, -29, -9)),
    ),
    (-3, 3): (
        ((-468, 406, -288), (465, 1744, 312)),
        ((174, 51, 13), (-166, -35, -9)),
    ),
    (1, -1): (
        ((-444, 398, -304), (445, 1752, 296)),
        ((182, 47, 17), (-174, -31, -13)),
    ),
    (-1, 1): (
        ((-456, 402, -296), (455, 1748, 304)),
        ((178, 49, 15), (-170, -33, -11)),
    ),
    (3, -3): (
        ((-432, 394, -312), (435, 1756, 288)),
        ((186, 45, 19), (-178, -29, -15)),
    ),
}

# Radius truths are exact half-millimetre rationals.  The half-millimetre
# offset excludes the already-consumed fixed-radius control 0.210 m.  This
# explicit table intentionally makes the unordered pair variant independent of
# kinematics: each physical primitive occurs with two different pairs.
RADIUS_DENOMINATOR = 2000
RADIUS_PAIR_NUMERATORS: dict[
    SceneSplit,
    tuple[tuple[tuple[int, int], ...], ...],
] = {
    "development": (
        ((411, 447), (421, 457)),
        ((431, 467), (441, 477)),
    ),
    "selector": (
        ((413, 449), (419, 455)),
        ((433, 469), (439, 475)),
    ),
    "confirmation": (
        ((415, 451), (425, 461)),
        ((427, 463), (437, 473)),
    ),
    "final_test": (
        ((417, 453), (423, 459)),
        ((429, 465), (435, 471)),
    ),
}
CONSUMED_FIXED_RADIUS_NUMERATOR = 420
RADIUS_ESTIMATOR_BOUNDS_M = (0.19, 0.25)
MINIMUM_RADIUS_PAIR_SEPARATION_M = 0.018

CONIC_CERTIFICATION_DTYPE = "torch.float64"
CONIC_PIXEL_SAFETY_TOLERANCE = 1.0e-7
CONIC_DEPTH_TOLERANCE_M = 1.0e-8
CONIC_RHO_TOLERANCE_M2 = 1.0e-12
CONIC_RELATIVE_ALGEBRA_TOLERANCE = 1.0e-12
CONIC_RELATIVE_BOUNDARY_RESIDUAL_TOLERANCE = 1.0e-12
FROZEN_TORCH_VERSION = "2.9.0a0+gitcbe1a35"
FROZEN_PLATFORM_SYSTEM = "Darwin"
FROZEN_PLATFORM_MACHINE = "x86_64"
FROZEN_PYTHON_VERSION = "3.10.20"
FROZEN_BYTEORDER = "little"
TORCH_DETERMINISM_SCOPE = "exact_frozen_torch_build_and_platform_cpu_float32_state_float64_conic"

# Conservative gates are inherited from the accepted clean two-visible rung
# where applicable.  Sphere-fit gates are source-time observability checks,
# not calibrated uncertainty claims.
MINIMUM_FULL_SUPPORT_PIXELS = 20
MINIMUM_CONIC_GAP_LOWER_BOUND_PIXELS = 4.0
MINIMUM_CONIC_BOUNDARY_CLEARANCE_PIXELS = 6.0
MINIMUM_WORLD_SURFACE_GAP_M = 1.0
MINIMUM_WORLD_BOUNDARY_M = 0.15
MINIMUM_EPISODE_SPEED_MPS = 0.035
MAXIMUM_EPISODE_SPEED_MPS = 0.065
MINIMUM_HISTORY_DISPLACEMENT_M = 0.025
MINIMUM_RADIUS_BOUND_CLEARANCE_M = 0.01
MAXIMUM_SPHERE_FIT_CONDITION = 20.0
MAXIMUM_SPHERE_FIT_RELATIVE_RESIDUAL = 2.0e-5
MAXIMUM_SPHERE_FIT_RADIUS_ERROR_M = 1.0e-5
MAXIMUM_SPHERE_FIT_CENTRE_ERROR_M = 1.5e-5
MAXIMUM_CAMERA_CALIBRATION_ERROR = 2.0e-5
MINIMUM_CAMERA_STEP_ANGLE_RADIANS = 0.01198
MAXIMUM_CAMERA_STEP_ANGLE_RADIANS = 0.01202
MINIMUM_CAMERA_TRANSLATION_STEP_M = 0.0551
MAXIMUM_CAMERA_TRANSLATION_STEP_M = 0.0553

# These public implementations are never called on a formal scene.  Exact
# source bindings connect the independent mathematics to the already accepted
# public-algorithm control without widening this module's authority.
PUBLIC_CAMERA_ALGORITHM_VERSION = "sphere_world_v7/float32_look_at_and_rigid_inverse"
PUBLIC_RENDERER_ALGORITHM_VERSION = "sphere_world_v7/stable_metric_ray_sphere_near_root"
PUBLIC_PHYSICS_ALGORITHM_VERSION = "sphere_world_v7/exact_linear_drag_then_contact_substeps"
PUBLIC_CAMERA_SOURCE_SHA256 = "23c9798d412a44e9f8b7bea57ef7598e469dfeea087e1515a6c27d51e53caa27"
PUBLIC_PHYSICS_SOURCE_SHA256 = "99a69c80ef87ce15a783a43b1342112600431a6b33d0aa95dacaac148202c02f"
PUBLIC_RENDERER_SOURCE_SHA256 = "76ae74a9c0da3f002b4e2b2234228f5dff8c1117965721bf2328df658e548876"
ACCEPTED_ORBITAL_SOURCE_SHA256 = "02e75b325bdf7bad310f8973a786a396b8762104261702b299a9f8103748e569"
ACCEPTED_ORBITAL_CERTIFICATE_SHA256 = (
    "7832ddb49081292d0f50a5eb63edb38fefb49d136d7e1757c73d9c658e42a36f"
)
ACCEPTED_SELECTED_METADATA_SHA256 = (
    "98102feb73c163c57911ff7d1445831b8822ccccc273ef0f88c8d018fd7036f7"
)

# Filled after the exhaustive pure certificate is reviewed.  These values bind
# source metadata and derived traces, never external data or run artifacts.
FROZEN_TRACE_SHA256: dict[str, str] = {
    "metadata": "c7439be3d453fee83b28615cdf338f750b2700f56b1ea4d165790089661acbc1",
    "balance": "ddc36a238791e567012c06953c79999944e1290ad1d7750a6776f060f8e5088f",
    "kinematic": "7b3abf198b12825c9ff548dfe747dde1d9402f4bf7b1e573e2468004a17df1fa",
    "radius_labelled_physical": "4b812318837751db3d97d3b935c20027bb4202147d80b8cbeab3d06c9a4fe960",
    "camera": "aad97eb3b84b35f1016850b80ff685c456458796a420902ec5457002b66d2b76",
    "raster": "b7990b5d424fdcf2373d9f6a9adca3d6a0182a1d39758d2655f5b650625e3e0d",
    "conic_geometry": "f5239000cf5407af72559c55f4817fa9948d57199a8e30871d57435cb0e3ff6a",
    "fit_observability": "a352b68362a502428c4e6d73f2a5f5d806ee63f80dcfcb48685df5aa841353d0",
    "combined": "a2d73c20f6d5167d35bee3b790ae6959e4dcb51889acd7f4323b027607529c79",
    "expected_lifecycle": "3cf7559ecd1606a03b7c21f0596b14f8ab690e6ec658875040ff7e5d9859b38b",
}
FROZEN_SPLIT_TRACE_SHA256: dict[str, dict[str, str]] = {
    "kinematic": {
        "development": "809e23de612aa17e1f270fc983fec4c56d6196c3cb78ec1fb7f9328a19fa0771",
        "selector": "9205618a6169b596dac863888b2375c50615f79221006acfbe47369e923e31fb",
        "confirmation": "566eace7afd803cfc06ffc7390d7989b094bb5bde43c5c8ee9d84b8b2077029b",
        "final_test": "8bbfb3eb8ddfb35be1b0f1685499179d1cde535ad9c925ceaf4b5d3a4ab31126",
    },
    "radius_labelled_physical": {
        "development": "cbde1225854f9f7b37413f927c9d3af2afb11f1e26cbe02f387c8c9994a37c54",
        "selector": "a3f84a3a5491aa0cd9fe02cf561ca6f9f872b4bc8a3a9a8e4df40becf7bdef31",
        "confirmation": "3e8800d283c5ffe3d42a8166547704dfffa38738c284e6809e476fc752944731",
        "final_test": "d6e7315fa7bf49347be29927b79b2082b80e911e4f73a8dcfd9f07224e84fa0c",
    },
    "raster": {
        "development": "2f165c0c0759fbd07da8e5bd6e88d94d12700bd59a9413a21e26ef9327b8be89",
        "selector": "7238e7d9d7f1b80758438563a1bf24e3ee6a20a75ab7eb6e37d3cfc65d9e1da5",
        "confirmation": "e9f257c62448c793f6099c727a1d2f17ac41b01e3e76bb669861e075c5bc147a",
        "final_test": "89ad46344fe8770d79ab0ebba851ea063078ff9f915f369b8386537a07c78df5",
    },
    "conic_geometry": {
        "development": "dd91ad6696032a6d5f43952fc5eaedbf4591ec91b9f2b23480be1479b668beed",
        "selector": "b5dfbdc9ffbcaae62f807b627ee19262e56786e1b7784b1e6ec2077713f7909e",
        "confirmation": "3836edccbb9b653da26d645e2938bf475db6db10c9216387c75f93e78f444b1b",
        "final_test": "139c605d0bbbb6507fcb81e955e4d94e1ad6fc95f55c2e35b94b22ca893a8107",
    },
    "fit_observability": {
        "development": "a2a3f3ecf7bf487fe2b6de7cd338569e6911729978c4772f6452ae03ad72d39d",
        "selector": "de735397a35a0dabc447082fdb1c7556413c61fe81210549b94044787ddf6e3b",
        "confirmation": "f69851687b3ec0a4447eef08220ba5fea71a24c641b1a96167ad5b3f417b6b44",
        "final_test": "4fa71919b84c3b9bc8c6e2e3cbd1235d07ed77f8baf369c487887f4e0954c50f",
    },
    "combined": {
        "development": "d891e50f0e1e9962c34508e96c0294248f43cd8a6ab14ea4cefae0010b4bbee6",
        "selector": "14e7c852fa3a56abf3d130aea4b6793549c29b3cf9aff7ae64395744d4fc68eb",
        "confirmation": "822ed7b4ff3efe05906c1bf4b1928253c1ebdb5d152cab84dee466789c36170f",
        "final_test": "adbc42f44133eb3826ce644f9c08a2a007257a8c26e791376042bbe2b6c4e12c",
    },
}
FROZEN_ACCEPTED_KINEMATIC_TRACE_SHA256: dict[str, str] = {
    "a_-3_b_-3": "d635a5ddfd7a14a0d6ba8f591bb28880a5c49dcd40e09cd851fec7089fed7a95",
    "a_1_b_1": "db0f9da5e8fb281377b2826b445a9fa06a0a685e156b9189682676b1592a5507",
    "a_-1_b_-1": "c8741f4ff7a3e0bdb89a8f59b0523d8538a8e22f9e5d1e0922b4b398eed9330b",
    "a_3_b_3": "c1b25ca8e5ab80bdccd578cbf7e9315cccb45a933814701d5d5a5e82fcc14b20",
    "a_-3_b_3": "65c9391b1f1643f94a0621114eaee935266ca4d8aaa971295780bd278906bd60",
    "a_1_b_-1": "39ab1295dffb6f666f24f9fb9e1613e02c4c5f5c313ffee2195097b38323fce0",
    "a_-1_b_1": "985aa116cc0542f05499787b0009c018a27505b9e3eee2115b995aa245dc5368",
    "a_3_b_-3": "ceeead2738d04d9913fb81c1635ada86a364d0025a84df0755bbb5d67daa72fc",
}
FROZEN_ACCEPTED_CAMERA_TRACE_SHA256: dict[str, str] = {
    "0": "939c21762ffa4dd810e68f67b9a0744097567b92b994d82e5a48c590fb7d6ff3",
    "1": "aad24c1b2393b8290d663f6c56f8134eaa74d636f4d4c3226e029198ad271ebd",
    "2": "fe76096fb7ff96c3cb8ca8719b34359334f7d7176ab0de8601038005fa077c05",
    "3": "e927c7e014a42aac39b2fd1b6cb0d2d6f4196a66a9b826e5139a2328d916c9d6",
    "4": "aa7c05e183fc047d6501cfaf00f071dfd65a7ae9e45eac75d79a859b7866ba48",
    "5": "9c2bf807b6a6c4ae92debc727227a466802f8cded7dfc9c4e3f4c9d9124c38b3",
    "6": "7701909291e2da4cdf6cbca68d8570c16332e078d1836bf96721eb0f8f122cff",
    "7": "ae52238d813e6319ff75b9a7794c6cf6f974483bb55d4da23517412500ef0656",
}
FROZEN_CERTIFICATE_SHA256 = "473137981e0a6443834c806f9f8792e2fee6a556961e5d977d3c6ae69cc7f0d5"


@dataclass(frozen=True, slots=True)
class VariableRadiusSceneSpecification:
    """One immutable rational formal scene description."""

    split: SceneSplit
    ordinal: int
    split_index: int
    evidence_role: EvidenceRole
    primitive_index: int
    pair_variant: int
    radius_role: int
    camera_stratum: int
    phase_index: int
    direction_index: int
    direction: int
    a: int
    b: int
    position_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    velocity_numerators: tuple[tuple[int, int, int], tuple[int, int, int]]
    radius_pair_index: int
    low_radius_numerator: int
    high_radius_numerator: int
    radius_slot_numerators: tuple[int, int]
    palette_swapped: bool
    albedo: tuple[tuple[float, float, float], tuple[float, float, float]]

    @property
    def position(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return tuple(  # type: ignore[return-value]
            tuple(value / 1000.0 for value in row) for row in self.position_numerators
        )

    @property
    def velocity(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return tuple(  # type: ignore[return-value]
            tuple(value / 4000.0 for value in row) for row in self.velocity_numerators
        )

    @property
    def radius(self) -> tuple[float, float]:
        return tuple(  # type: ignore[return-value]
            value / RADIUS_DENOMINATOR for value in self.radius_slot_numerators
        )

    def position_tensor(self) -> Tensor:
        return torch.tensor(self.position_numerators, dtype=torch.float32) / 1000.0

    def velocity_tensor(self) -> Tensor:
        return torch.tensor(self.velocity_numerators, dtype=torch.float32) / 4000.0

    def radius_tensor(self) -> Tensor:
        return (
            torch.tensor(self.radius_slot_numerators, dtype=torch.float32)[:, None]
            / RADIUS_DENOMINATOR
        )

    def albedo_tensor(self) -> Tensor:
        return torch.tensor(self.albedo, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class PurePhysicalTrajectory:
    """Independent fixed-drag kinematics with no simulator authority."""

    positions: Tensor
    velocities: Tensor
    substep_positions: Tensor
    substep_velocities: Tensor


@dataclass(frozen=True, slots=True)
class PureCameraFrame:
    """Independent known calibration value, not a simulator CameraFrame."""

    timestamp: float
    position: Tensor
    target: Tensor
    world_from_camera: Tensor
    camera_from_world: Tensor
    intrinsics: Tensor


def _normalise_split(split: str) -> SceneSplit:
    if type(split) is not str:
        raise TypeError("variable-radius split must be a string")
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown variable-radius split {split!r}")
    return split  # type: ignore[return-value]


def _ordinal_components(ordinal: int) -> tuple[int, int, int, int]:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("variable-radius scene ordinal must be an integer")
    if not 0 <= ordinal < SCENES_PER_SPLIT:
        raise IndexError(ordinal)
    primitive_index, remainder = divmod(ordinal, 32)
    pair_variant, remainder = divmod(remainder, 16)
    radius_role, camera_stratum = divmod(remainder, CAMERA_STRATA)
    return primitive_index, pair_variant, radius_role, camera_stratum


def _accepted_geometry(
    split: SceneSplit,
    primitive_index: int,
) -> tuple[
    tuple[tuple[int, int, int], tuple[int, int, int]],
    tuple[tuple[int, int, int], tuple[int, int, int]],
    int,
    int,
]:
    a, b = SPLIT_PRIMITIVE_PAIRS[split][primitive_index]
    position_numerators = (
        (-450 + 6 * a, 400 + 2 * b, -300 + 4 * b),
        (450 + 5 * b, 1750 + 2 * a, 300 - 4 * a),
    )
    # The accepted formula's x components used denominator 2000.  Doubling
    # those numerators gives one exact denominator-4000 metadata table.
    velocity_numerators = (
        (180 + 2 * a, 48 + b, 16 + a),
        (-172 + 2 * b, -32 + a, -12 + b),
    )
    if (position_numerators, velocity_numerators) != ACCEPTED_PRIMITIVE_RATIONAL_ROWS[(a, b)]:
        raise RuntimeError("variable-radius geometry differs from the copied accepted row")
    return position_numerators, velocity_numerators, a, b


def scene_specification(split: str, ordinal: int) -> VariableRadiusSceneSpecification:
    """Return a formal scene from a conceptual split and ordinal only."""

    canonical_split = _normalise_split(split)
    primitive_index, pair_variant, radius_role, camera_stratum = _ordinal_components(ordinal)
    split_index = SPLIT_INDEX[canonical_split]
    phase_index, direction_index = divmod(camera_stratum, 2)
    direction = CAMERA_DIRECTIONS[direction_index]
    position_numerators, velocity_numerators, a, b = _accepted_geometry(
        canonical_split,
        primitive_index,
    )
    radius_pair_index = primitive_index * PAIR_VARIANTS_PER_PRIMITIVE + pair_variant
    low_radius_numerator, high_radius_numerator = RADIUS_PAIR_NUMERATORS[canonical_split][
        primitive_index
    ][pair_variant]
    radius_slot_numerators = (
        (low_radius_numerator, high_radius_numerator)
        if radius_role == 0
        else (high_radius_numerator, low_radius_numerator)
    )
    palette_swapped = bool((primitive_index + phase_index + direction_index) & 1)
    albedo = PALETTE[::-1] if palette_swapped else PALETTE
    return VariableRadiusSceneSpecification(
        split=canonical_split,
        ordinal=ordinal,
        split_index=split_index,
        evidence_role=SPLIT_EVIDENCE_ROLE[canonical_split],
        primitive_index=primitive_index,
        pair_variant=pair_variant,
        radius_role=radius_role,
        camera_stratum=camera_stratum,
        phase_index=phase_index,
        direction_index=direction_index,
        direction=direction,
        a=a,
        b=b,
        position_numerators=position_numerators,
        velocity_numerators=velocity_numerators,
        radius_pair_index=radius_pair_index,
        low_radius_numerator=low_radius_numerator,
        high_radius_numerator=high_radius_numerator,
        radius_slot_numerators=radius_slot_numerators,
        palette_swapped=palette_swapped,
        albedo=albedo,
    )


def counterfactual_twin_ordinal(ordinal: int) -> int:
    """Toggle only the radius-slot role for one primitive/camera pair."""

    _ordinal_components(ordinal)
    return ordinal ^ CAMERA_STRATA


def pair_variant_twin_ordinal(ordinal: int) -> int:
    """Toggle only the unordered radius-pair variant for one scene address."""

    _ordinal_components(ordinal)
    return ordinal ^ (RADIUS_ROLES_PER_PRIMITIVE * CAMERA_STRATA)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def scene_metadata(specification: VariableRadiusSceneSpecification) -> dict[str, Any]:
    """Return exact JSON-safe constructor metadata."""

    if not isinstance(specification, VariableRadiusSceneSpecification):
        raise TypeError("scene_metadata requires a VariableRadiusSceneSpecification")
    return {
        "split": specification.split,
        "ordinal": specification.ordinal,
        "split_index": specification.split_index,
        "evidence_role": specification.evidence_role,
        "primitive_index": specification.primitive_index,
        "pair_variant": specification.pair_variant,
        "radius_role": specification.radius_role,
        "camera_stratum": specification.camera_stratum,
        "phase_index": specification.phase_index,
        "direction_index": specification.direction_index,
        "direction": specification.direction,
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
        "radius_rational": {
            "pair_index": specification.radius_pair_index,
            "low_numerator": specification.low_radius_numerator,
            "high_numerator": specification.high_radius_numerator,
            "slot_numerators": specification.radius_slot_numerators,
            "denominator": RADIUS_DENOMINATOR,
        },
        "fixed_drag_rational": {
            "numerator": FIXED_DRAG_NUMERATOR,
            "denominator": FIXED_DRAG_DENOMINATOR,
        },
        "palette_swapped": specification.palette_swapped,
        "albedo": specification.albedo,
    }


def scene_signature(specification: VariableRadiusSceneSpecification) -> str:
    return canonical_sha256(scene_metadata(specification))


def split_scene_signatures(split: str) -> tuple[str, ...]:
    canonical_split = _normalise_split(split)
    return tuple(
        scene_signature(scene_specification(canonical_split, ordinal))
        for ordinal in range(SCENES_PER_SPLIT)
    )


def family_scene_signature() -> str:
    return canonical_sha256({split: split_scene_signatures(split) for split in SPLITS})


def _base_geometry_metadata(specification: VariableRadiusSceneSpecification) -> dict[str, Any]:
    return {
        "a": specification.a,
        "b": specification.b,
        "position_numerators": specification.position_numerators,
        "velocity_numerators": specification.velocity_numerators,
    }


def _radius_labelled_metadata(
    specification: VariableRadiusSceneSpecification,
) -> dict[str, Any]:
    return {
        **_base_geometry_metadata(specification),
        "radius_slot_numerators": specification.radius_slot_numerators,
        "radius_denominator": RADIUS_DENOMINATOR,
    }


def _cross_split_match_count(values: dict[str, set[str] | set[int]]) -> int:
    return sum(
        len(values[left] & values[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    )


def _scene_balance_certificate_bytes() -> bytes:
    per_split: dict[str, Any] = {}
    all_scene_signatures: list[str] = []
    geometry_by_split: dict[str, set[str]] = {}
    radius_source_by_split: dict[str, set[str]] = {}
    truth_by_split: dict[str, set[int]] = {}
    all_geometry: set[str] = set()
    all_radius_sources: set[str] = set()
    counterfactual_pair_count = 0
    pair_variant_counterfactual_count = 0

    for split in SPLITS:
        specifications = [
            scene_specification(split, ordinal) for ordinal in range(SCENES_PER_SPLIT)
        ]
        signatures = [scene_signature(specification) for specification in specifications]
        primitive_histogram = {str(index): 0 for index in range(PRIMITIVES_PER_SPLIT)}
        pair_variant_histogram = {str(index): 0 for index in range(PAIR_VARIANTS_PER_PRIMITIVE)}
        role_histogram = {str(index): 0 for index in range(RADIUS_ROLES_PER_PRIMITIVE)}
        camera_histogram = {str(index): 0 for index in range(CAMERA_STRATA)}
        palette_histogram = {"false": 0, "true": 0}
        slot_truth_histogram: dict[str, dict[str, int]] = {}
        camera_truth_histogram: dict[str, dict[str, int]] = {}
        colour_truth_histogram: dict[str, dict[str, int]] = {}
        split_geometry: set[str] = set()
        split_radius_sources: set[str] = set()
        split_truth: set[int] = set()

        for specification in specifications:
            primitive_histogram[str(specification.primitive_index)] += 1
            pair_variant_histogram[str(specification.pair_variant)] += 1
            role_histogram[str(specification.radius_role)] += 1
            camera_histogram[str(specification.camera_stratum)] += 1
            palette_histogram[str(specification.palette_swapped).lower()] += 1
            geometry_signature = canonical_sha256(_base_geometry_metadata(specification))
            radius_source_signature = canonical_sha256(_radius_labelled_metadata(specification))
            split_geometry.add(geometry_signature)
            split_radius_sources.add(radius_source_signature)
            all_geometry.add(geometry_signature)
            all_radius_sources.add(radius_source_signature)
            split_truth.update(
                (specification.low_radius_numerator, specification.high_radius_numerator)
            )
            for slot, truth in enumerate(specification.radius_slot_numerators):
                truth_key = str(truth)
                slot_truth_histogram.setdefault(truth_key, {"0": 0, "1": 0})[str(slot)] += 1
                camera_truth_histogram.setdefault(
                    truth_key,
                    {str(index): 0 for index in range(CAMERA_STRATA)},
                )[str(specification.camera_stratum)] += 1
                colour = "palette_0" if specification.albedo[slot] == PALETTE[0] else "palette_1"
                colour_truth_histogram.setdefault(
                    truth_key,
                    {"palette_0": 0, "palette_1": 0},
                )[colour] += 1

        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            for pair_variant in range(PAIR_VARIANTS_PER_PRIMITIVE):
                for camera_stratum in range(CAMERA_STRATA):
                    ordinal = primitive_index * 32 + pair_variant * 16 + camera_stratum
                    first = scene_specification(split, ordinal)
                    second = scene_specification(split, counterfactual_twin_ordinal(ordinal))
                    first_metadata = scene_metadata(first)
                    second_metadata = scene_metadata(second)
                    first_metadata["ordinal"] = second_metadata["ordinal"]
                    first_metadata["radius_role"] = second_metadata["radius_role"]
                    first_metadata["radius_rational"]["slot_numerators"] = second_metadata[
                        "radius_rational"
                    ]["slot_numerators"]
                    if first_metadata != second_metadata:
                        raise RuntimeError(
                            "radius-role counterfactual changed a non-radius source field"
                        )
                    if first.radius_slot_numerators != second.radius_slot_numerators[::-1]:
                        raise RuntimeError("radius-role counterfactual is not an exact slot swap")
                    counterfactual_pair_count += 1

            for radius_role in range(RADIUS_ROLES_PER_PRIMITIVE):
                for camera_stratum in range(CAMERA_STRATA):
                    ordinal = primitive_index * 32 + radius_role * 8 + camera_stratum
                    first = scene_specification(split, ordinal)
                    second = scene_specification(split, ordinal ^ 16)
                    if first.position_numerators != second.position_numerators:
                        raise RuntimeError("pair variants changed accepted position controls")
                    if first.velocity_numerators != second.velocity_numerators:
                        raise RuntimeError("pair variants changed accepted velocity controls")
                    if (
                        first.camera_stratum != second.camera_stratum
                        or first.radius_role != second.radius_role
                        or first.albedo != second.albedo
                        or first.palette_swapped != second.palette_swapped
                    ):
                        raise RuntimeError(
                            "pair variants changed camera, role, or appearance controls"
                        )
                    if frozenset(first.radius_slot_numerators) == frozenset(
                        second.radius_slot_numerators
                    ):
                        raise RuntimeError(
                            "pair variants do not select genuinely different unordered pairs"
                        )
                    pair_variant_counterfactual_count += 1

        if len(split_geometry) != 2 or len(split_radius_sources) != 8:
            raise RuntimeError("each split requires two geometries and eight radius sources")
        if len(split_truth) != 8:
            raise RuntimeError("each split requires eight distinct radius truths")
        expected_slot_histogram = {"0": 8, "1": 8}
        expected_camera_histogram = {str(index): 2 for index in range(CAMERA_STRATA)}
        expected_colour_histogram = {"palette_0": 8, "palette_1": 8}
        if any(value != expected_slot_histogram for value in slot_truth_histogram.values()):
            raise RuntimeError("a radius truth is not balanced over physical slots")
        if any(value != expected_camera_histogram for value in camera_truth_histogram.values()):
            raise RuntimeError("a radius truth is not balanced over camera strata")
        if any(value != expected_colour_histogram for value in colour_truth_histogram.values()):
            raise RuntimeError("a radius truth is not balanced over object colours")
        geometry_by_split[split] = split_geometry
        radius_source_by_split[split] = split_radius_sources
        truth_by_split[split] = split_truth
        per_split[split] = {
            "scene_count": len(specifications),
            "unique_scene_signature_count": len(set(signatures)),
            "primitive_histogram": primitive_histogram,
            "pair_variant_histogram": pair_variant_histogram,
            "radius_role_histogram": role_histogram,
            "camera_stratum_histogram": camera_histogram,
            "palette_swap_histogram": palette_histogram,
            "radius_truth_numerator_sha256": canonical_sha256(sorted(split_truth)),
            "radius_truth_count": len(split_truth),
            "slot_truth_histogram": slot_truth_histogram,
            "camera_truth_histogram": camera_truth_histogram,
            "colour_truth_histogram": colour_truth_histogram,
            "base_geometry_count": len(split_geometry),
            "base_geometry_set_sha256": canonical_sha256(sorted(split_geometry)),
            "radius_labelled_source_count": len(split_radius_sources),
            "radius_labelled_source_set_sha256": canonical_sha256(sorted(split_radius_sources)),
            "ordered_scene_signatures_sha256": canonical_sha256(signatures),
        }
        all_scene_signatures.extend(signatures)

    truth_count = sum(map(len, truth_by_split.values()))
    truth_unique_count = len(set().union(*truth_by_split.values()))
    geometry_cross_split = _cross_split_match_count(geometry_by_split)
    radius_source_cross_split = _cross_split_match_count(radius_source_by_split)
    truth_cross_split = _cross_split_match_count(truth_by_split)
    if truth_count != 32 or truth_unique_count != 32 or truth_cross_split:
        raise RuntimeError("formal radius truths must be 32/32 and split-disjoint")
    if CONSUMED_FIXED_RADIUS_NUMERATOR in set().union(*truth_by_split.values()):
        raise RuntimeError("a formal radius truth matches the consumed fixed-radius control")
    if len(all_geometry) != 8 or geometry_cross_split:
        raise RuntimeError("formal base geometries must be 8/8 and split-disjoint")
    if len(all_radius_sources) != 32 or radius_source_cross_split:
        raise RuntimeError("formal radius-labelled sources must be 32/32 and split-disjoint")
    unsigned = {
        "split_order": SPLITS,
        "per_split": per_split,
        "total_scene_count": len(all_scene_signatures),
        "unique_scene_signature_count": len(set(all_scene_signatures)),
        "base_geometry_count": len(all_geometry),
        "base_geometry_cross_split_match_count": geometry_cross_split,
        "radius_labelled_source_count": len(all_radius_sources),
        "radius_labelled_source_cross_split_match_count": radius_source_cross_split,
        "radius_truth_count": truth_count,
        "radius_truth_unique_count": truth_unique_count,
        "radius_truth_cross_split_match_count": truth_cross_split,
        "consumed_fixed_radius_match_count": 0,
        "counterfactual_pair_count": counterfactual_pair_count,
        "pair_variant_counterfactual_count": pair_variant_counterfactual_count,
        "family_scene_signature_sha256": family_scene_signature(),
    }
    return _canonical_json_bytes({**unsigned, "balance_sha256": canonical_sha256(unsigned)})


def _computed_scene_balance_certificate() -> dict[str, Any]:
    return json.loads(_scene_balance_certificate_bytes())


def scene_balance_certificate() -> dict[str, Any]:
    return copy.deepcopy(_computed_scene_balance_certificate())


def manual_kinematic_trajectory(
    specification: VariableRadiusSceneSpecification,
) -> PurePhysicalTrajectory:
    """Evaluate the exact float32 fixed-drag recurrence without public physics."""

    if not isinstance(specification, VariableRadiusSceneSpecification):
        raise TypeError("manual_kinematic_trajectory requires a formal scene")
    position = specification.position_tensor()
    velocity = specification.velocity_tensor()
    coefficient = torch.full(
        (2, 1),
        FIXED_DRAG_NUMERATOR / FIXED_DRAG_DENOMINATOR,
        dtype=torch.float32,
    )
    substep_seconds = 1.0 / PHYSICS_RATE_HZ
    decay = torch.exp(-coefficient * substep_seconds)
    displacement = -torch.expm1(-coefficient * substep_seconds) / coefficient
    frame_positions = [position.clone()]
    frame_velocities = [velocity.clone()]
    substep_positions = [position.clone()]
    substep_velocities = [velocity.clone()]
    for substep_index in range(PHYSICAL_SUBSTEP_COUNT):
        position = position + velocity * displacement
        velocity = velocity * decay
        substep_positions.append(position.clone())
        substep_velocities.append(velocity.clone())
        if (substep_index + 1) % SUBSTEPS_PER_FRAME == 0:
            frame_positions.append(position.clone())
            frame_velocities.append(velocity.clone())
    if len(frame_positions) != FRAME_COUNT:
        raise AssertionError("fixed-drag recurrence did not emit exactly 56 frames")
    return PurePhysicalTrajectory(
        positions=torch.stack(frame_positions),
        velocities=torch.stack(frame_velocities),
        substep_positions=torch.stack(substep_positions),
        substep_velocities=torch.stack(substep_velocities),
    )


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
    """Evaluate the accepted known orbit with local float32 matrix algebra."""

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
    right = right / torch.linalg.vector_norm(right)
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


def reject_formal_public_api_input(value: object, *, api_name: str) -> None:
    """Fail closed before any public physics/render wrapper accepts formal state."""

    if type(api_name) is not str or not api_name:
        raise TypeError("public API name must be a non-empty string")
    if isinstance(
        value,
        (VariableRadiusSceneSpecification, PurePhysicalTrajectory, PureCameraFrame),
    ):
        raise PermissionError(
            f"formal variable-radius values may not cross the public {api_name} boundary"
        )


def _digest_field(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(struct.pack(">I", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack(">Q", len(payload)))
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


def _stack_camera_frames(frames: tuple[PureCameraFrame, ...]) -> dict[str, Tensor]:
    return {
        "position": torch.stack([frame.position for frame in frames]),
        "target": torch.stack([frame.target for frame in frames]),
        "world_from_camera": torch.stack([frame.world_from_camera for frame in frames]),
        "camera_from_world": torch.stack([frame.camera_from_world for frame in frames]),
        "intrinsics": torch.stack([frame.intrinsics for frame in frames]),
    }


def _pinhole_sphere_conic_batch(
    points_camera: Tensor,
    radius: Tensor,
    intrinsics: Tensor,
) -> dict[str, Tensor]:
    """Certify float64 pinhole-sphere silhouette ellipses.

    For homogeneous pixel ``p`` and ray ``d=K^-1 p``, tangency is
    ``p^T C p=0`` where
    ``C=K^-T[c c^T-(||c||^2-r^2)I]K^-1``.  Coordinate extrema are exact for
    this ellipse.  Its enclosing circle supplies a conservative two-ellipse
    separation lower bound without claiming the silhouette is a circle.
    """

    centre = points_camera.to(torch.float64)
    resolved_radius = radius.to(torch.float64)
    calibration = intrinsics.to(torch.float64)
    if not (
        torch.isfinite(centre).all()
        and torch.isfinite(resolved_radius).all()
        and torch.isfinite(calibration).all()
    ):
        raise RuntimeError("conic certification requires finite inputs")
    if not bool((resolved_radius > 0.0).all()):
        raise RuntimeError("conic certification requires positive radii")
    if not bool((centre[..., 2] > resolved_radius[None] + CONIC_DEPTH_TOLERANCE_M).all()):
        raise RuntimeError("conic certification requires spheres in front of the camera")

    identity3 = torch.eye(3, dtype=torch.float64).expand(calibration.shape[0], -1, -1)
    calibration_singular_values = torch.linalg.svdvals(calibration)
    if not bool(
        (
            calibration_singular_values[..., -1]
            > CONIC_RELATIVE_ALGEBRA_TOLERANCE * calibration_singular_values[..., 0]
        ).all()
    ):
        raise RuntimeError("conic certification requires nonsingular calibration")
    inverse_intrinsics = torch.linalg.solve(calibration, identity3)
    rho = centre.square().sum(dim=-1) - resolved_radius[None].square()
    if not bool((rho > CONIC_RHO_TOLERANCE_M2).all()):
        raise RuntimeError("conic certification found a camera inside a sphere")
    tangent_form = centre[..., :, None] * centre[..., None, :] - rho[..., None, None] * torch.eye(
        3, dtype=torch.float64
    )
    conic = torch.einsum(
        "fij,fojk,fkl->foil",
        inverse_intrinsics.transpose(-1, -2),
        tangent_form,
        inverse_intrinsics,
    )
    homogeneous_scale = conic.abs().amax(dim=(-2, -1))
    if not bool(torch.isfinite(homogeneous_scale).all() and (homogeneous_scale > 0.0).all()):
        raise RuntimeError("conic certification produced a degenerate scale")
    symmetry_error = (conic - conic.transpose(-1, -2)).abs().amax(dim=(-2, -1)) / (
        homogeneous_scale
    )
    conic = 0.5 * (conic + conic.transpose(-1, -2)) / homogeneous_scale[..., None, None]

    conic_eigenvalues = torch.linalg.eigvalsh(conic)
    signature_valid = (
        (conic_eigenvalues[..., 0] < -CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (conic_eigenvalues[..., 1] < -CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (conic_eigenvalues[..., 2] > CONIC_RELATIVE_ALGEBRA_TOLERANCE)
    )
    projected_homogeneous = torch.einsum("fij,foj->foi", calibration, centre)
    projected_homogeneous = projected_homogeneous / projected_homogeneous[..., 2:]
    projected_value = torch.einsum(
        "foi,foij,foj->fo",
        projected_homogeneous,
        conic,
        projected_homogeneous,
    )
    if not bool(
        signature_valid.all() and (projected_value > CONIC_RELATIVE_ALGEBRA_TOLERANCE).all()
    ):
        raise RuntimeError("conic certification produced the wrong ellipse signature")

    quadratic = conic[..., :2, :2]
    linear = conic[..., :2, 2]
    constant = conic[..., 2, 2]
    ellipse_centre = torch.linalg.solve(quadratic, -linear.unsqueeze(-1)).squeeze(-1)
    gamma = constant + torch.einsum("foi,foi->fo", linear, ellipse_centre)
    negative_quadratic = -quadratic
    negative_eigenvalues = torch.linalg.eigvalsh(negative_quadratic)
    if not bool(
        (gamma > CONIC_RELATIVE_ALGEBRA_TOLERANCE).all()
        and (
            negative_eigenvalues[..., 0]
            > CONIC_RELATIVE_ALGEBRA_TOLERANCE * negative_eigenvalues[..., -1]
        ).all()
    ):
        raise RuntimeError("conic certification produced an unbounded ellipse")
    ellipse_shape = negative_quadratic / gamma[..., None, None]
    identity2 = torch.eye(2, dtype=torch.float64).expand(*ellipse_shape.shape[:-2], -1, -1)
    ellipse_covariance = torch.linalg.solve(ellipse_shape, identity2)
    ellipse_covariance = 0.5 * (ellipse_covariance + ellipse_covariance.transpose(-1, -2))
    covariance_eigenvalues = torch.linalg.eigvalsh(ellipse_covariance)
    if not bool(
        (
            covariance_eigenvalues[..., 0]
            > CONIC_RELATIVE_ALGEBRA_TOLERANCE * covariance_eigenvalues[..., -1]
        ).all()
    ):
        raise RuntimeError("conic certification produced a non-positive ellipse shape")

    coordinate_radius = ellipse_covariance.diagonal(dim1=-2, dim2=-1).sqrt()
    covariance_trace = ellipse_covariance[..., 0, 0] + ellipse_covariance[..., 1, 1]
    covariance_discriminant = torch.hypot(
        ellipse_covariance[..., 0, 0] - ellipse_covariance[..., 1, 1],
        2.0 * ellipse_covariance[..., 0, 1],
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

    axis0_delta = ellipse_covariance[..., :, 0] / coordinate_radius[..., 0, None]
    axis1_delta = ellipse_covariance[..., :, 1] / coordinate_radius[..., 1, None]
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
        "fopi,foij,fopj->fop",
        homogeneous_boundary,
        conic,
        homogeneous_boundary,
    )
    boundary_denominator = torch.linalg.matrix_norm(conic, ord="fro")[:, :, None] * (
        homogeneous_boundary.square().sum(dim=-1)
    )
    relative_boundary_residual = boundary_value.abs() / boundary_denominator.clamp_min(1.0e-30)
    centre_residual = torch.linalg.vector_norm(
        torch.einsum("foij,foj->foi", quadratic, ellipse_centre) + linear,
        dim=-1,
    ) / (
        torch.linalg.matrix_norm(quadratic, ord="fro")
        * torch.linalg.vector_norm(ellipse_centre, dim=-1)
        + torch.linalg.vector_norm(linear, dim=-1)
    ).clamp_min(1.0e-30)
    shape_residual = torch.linalg.matrix_norm(
        ellipse_shape @ ellipse_covariance - identity2,
        ord="fro",
    )
    valid = (
        signature_valid
        & torch.isfinite(ellipse_centre).all(dim=-1)
        & torch.isfinite(ellipse_shape).all(dim=(-2, -1))
        & torch.isfinite(ellipse_covariance).all(dim=(-2, -1))
        & torch.isfinite(coordinate_extrema).all(dim=-1)
        & torch.isfinite(enclosing_radius)
        & (symmetry_error <= CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (centre_residual <= CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (shape_residual <= CONIC_RELATIVE_ALGEBRA_TOLERANCE)
        & (relative_boundary_residual.amax(dim=-1) <= CONIC_RELATIVE_BOUNDARY_RESIDUAL_TOLERANCE)
    )
    return {
        "conic_matrix": conic,
        "conic_eigenvalues": conic_eigenvalues,
        "conic_projected_centre_value": projected_value,
        "conic_input_symmetry_error": symmetry_error,
        "ellipse_centre": ellipse_centre,
        "ellipse_shape": ellipse_shape,
        "ellipse_covariance": ellipse_covariance,
        "ellipse_coordinate_radius": coordinate_radius,
        "ellipse_coordinate_extrema": coordinate_extrema,
        "ellipse_enclosing_radius": enclosing_radius,
        "conic_relative_boundary_residual": relative_boundary_residual,
        "conic_centre_residual": centre_residual,
        "conic_shape_residual": shape_residual,
        "conic_valid": valid,
    }


def _independent_raster_and_fit_batch(
    positions: Tensor,
    radius: Tensor,
    camera: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Trace 56 frames and fit spheres with no renderer or estimator call."""

    height, width = IMAGE_SIZE
    relative = positions - camera["position"][:, None, :]
    rotation = camera["world_from_camera"][:, :3, :3]
    points_camera = torch.einsum("foi,fij->foj", relative, rotation)
    depth = points_camera[..., 2]
    intrinsics = camera["intrinsics"]
    conic = _pinhole_sphere_conic_batch(points_camera, radius, intrinsics)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ray_x = (pixel_x[None] - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0]
    ray_y = (pixel_y[None] - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1]
    ray_norm_squared = 1.0 + ray_x.square() + ray_y.square()
    ray_dot_centre = (
        ray_x[:, None] * points_camera[..., 0, None, None]
        + ray_y[:, None] * points_camera[..., 1, None, None]
        + points_camera[..., 2, None, None]
    )
    centre_cross_ray = torch.stack(
        (
            points_camera[..., 1, None, None] - points_camera[..., 2, None, None] * ray_y[:, None],
            points_camera[..., 2, None, None] * ray_x[:, None] - points_camera[..., 0, None, None],
            points_camera[..., 0, None, None] * ray_y[:, None]
            - points_camera[..., 1, None, None] * ray_x[:, None],
        ),
        dim=-1,
    )
    discriminant = ray_norm_squared[:, None] * radius[
        None, :, None, None
    ].square() - centre_cross_ray.square().sum(dim=-1)
    square_root = discriminant.clamp_min(0.0).sqrt()
    denominator = ray_dot_centre + square_root
    constant = (
        points_camera.square().sum(dim=-1)[..., None, None] - radius[None, :, None, None].square()
    )
    surface_depth = constant / denominator.clamp_min(1.0e-12)
    full_mask = (
        (depth > radius[None, :] + 1.0e-4)[..., None, None]
        & (discriminant >= 0.0)
        & (denominator > 0.0)
        & (surface_depth > 0.0)
        & torch.isfinite(surface_depth)
    )
    ordered = torch.where(full_mask, surface_depth, torch.full_like(surface_depth, torch.inf))
    depth_buffer, winner = ordered.min(dim=1)
    has_object = torch.isfinite(depth_buffer)
    winner = torch.where(has_object, winner.to(torch.int64), torch.full_like(winner, -1))
    visible_mask = full_mask & (winner[:, None] == torch.arange(2)[None, :, None, None])
    support = full_mask.sum(dim=(-2, -1))
    visible = visible_mask.sum(dim=(-2, -1))

    safe_surface_depth = torch.where(full_mask, surface_depth, torch.zeros_like(surface_depth))
    rays = torch.stack(
        (ray_x, ray_y, torch.ones_like(ray_x)),
        dim=-1,
    )
    points = rays[:, None] * safe_surface_depth[..., None]
    weights = full_mask.to(torch.float32)
    safe_support = support.to(torch.float32).clamp_min(1.0)
    mean = torch.einsum("fohw,fohwi->foi", weights, points) / safe_support[..., None]
    centred_points = points - mean[..., None, None, :]
    normal = torch.einsum(
        "fohw,fohwi,fohwj->foij",
        weights / safe_support[..., None, None],
        centred_points,
        centred_points,
    )
    norm_squared = points.square().sum(dim=-1)
    mean_norm_squared = torch.einsum("fohw,fohw->fo", weights, norm_squared) / safe_support
    centred_norm = norm_squared - mean_norm_squared[..., None, None]
    right = 0.5 * torch.einsum(
        "fohw,fohwi,fohw->foi",
        weights / safe_support[..., None, None],
        centred_points,
        centred_norm,
    )
    eigenvalues = torch.linalg.eigvalsh(normal)
    condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(1.0e-8)
    fitted_centre = torch.linalg.solve(normal, right.unsqueeze(-1)).squeeze(-1)
    point_distance = torch.linalg.vector_norm(
        points - fitted_centre[..., None, None, :],
        dim=-1,
    )
    fitted_radius = torch.einsum("fohw,fohw->fo", weights, point_distance) / safe_support
    residual = point_distance - fitted_radius[..., None, None]
    relative_residual = (
        torch.einsum("fohw,fohw->fo", weights, residual.square()) / safe_support
    ).clamp_min(0.0).sqrt() / fitted_radius.clamp_min(1.0e-8)
    fit_valid = (
        (support >= 4)
        & torch.isfinite(eigenvalues).all(dim=-1)
        & (eigenvalues[..., 0] > 1.0e-8)
        & torch.isfinite(fitted_centre).all(dim=-1)
        & torch.isfinite(fitted_radius)
        & torch.isfinite(relative_residual)
    )
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
        "fit_eigenvalues": eigenvalues,
        "fit_condition": condition,
        "fitted_centre": fitted_centre,
        "fitted_radius": fitted_radius,
        "fit_relative_residual": relative_residual,
        "fit_valid": fit_valid,
        **conic,
    }


def _validate_public_source_bindings() -> dict[str, str]:
    simulator_root = Path(__file__).resolve().parents[1] / "simulator"
    actual = {
        "accepted_orbital_qualification": hashlib.sha256(
            Path(__file__)
            .with_name("rgbd_two_visible_orbital_camera_qualification.py")
            .read_bytes()
        ).hexdigest(),
        "camera": hashlib.sha256((simulator_root / "camera.py").read_bytes()).hexdigest(),
        "physics": hashlib.sha256((simulator_root / "physics.py").read_bytes()).hexdigest(),
        "renderer": hashlib.sha256((simulator_root / "renderer.py").read_bytes()).hexdigest(),
    }
    expected = {
        "accepted_orbital_qualification": ACCEPTED_ORBITAL_SOURCE_SHA256,
        "camera": PUBLIC_CAMERA_SOURCE_SHA256,
        "physics": PUBLIC_PHYSICS_SOURCE_SHA256,
        "renderer": PUBLIC_RENDERER_SOURCE_SHA256,
    }
    if actual != expected:
        raise RuntimeError(
            "public simulator source differs from the frozen pure-certificate binding"
        )
    return actual


def _validated_determinism_scope() -> dict[str, Any]:
    actual = {
        "byteorder": sys.byteorder,
        "conic_dtype": CONIC_CERTIFICATION_DTYPE,
        "device": "cpu",
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": platform.python_version(),
        "state_dtype": "torch.float32",
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_version": torch.__version__,
    }
    expected = {
        "byteorder": FROZEN_BYTEORDER,
        "conic_dtype": "torch.float64",
        "device": "cpu",
        "platform_machine": FROZEN_PLATFORM_MACHINE,
        "platform_system": FROZEN_PLATFORM_SYSTEM,
        "python_version": FROZEN_PYTHON_VERSION,
        "state_dtype": "torch.float32",
        "torch_default_dtype": "torch.float32",
        "torch_version": FROZEN_TORCH_VERSION,
    }
    if actual != expected:
        raise RuntimeError("variable-radius certificate is outside its determinism scope")
    return actual


def _input_binding_literal(value: Any) -> Any:
    """Encode source constants without losing container/key type distinctions."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError("certificate input bindings must be finite")
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
        encoded_items = [
            [_input_binding_literal(key), _input_binding_literal(member)]
            for key, member in value.items()
        ]
        encoded_items.sort(key=lambda item: _canonical_json_bytes(item[0]))
        return {"container": "dict", "items": encoded_items}
    raise TypeError(f"unsupported certificate input binding {type(value)!r}")


def _fresh_certificate_input_binding_sha256() -> str:
    """Validate current pins and hash every source/table/environment input."""

    source_bindings = _validate_public_source_bindings()
    determinism_scope = _validated_determinism_scope()
    public_constants = {
        name: _input_binding_literal(value) for name, value in globals().items() if name.isupper()
    }
    payload = {
        "schema": "variable_radius_certificate_input_binding_v1",
        "scene_module_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bound_source_sha256": source_bindings,
        "determinism_scope": determinism_scope,
        "all_uppercase_public_constants_and_tables": public_constants,
    }
    return canonical_sha256(payload)


def _digest_recipe() -> dict[str, Any]:
    return {
        "algorithm": "sha256",
        "field_framing": [
            "label_utf8_length_u32_big_endian",
            "label_utf8_bytes",
            "payload_length_u64_big_endian",
            "payload_bytes",
        ],
        "tensor_metadata": "canonical_json_sorted_keys_compact_no_nan",
        "tensor_data": "cpu_contiguous_little_endian_C_order",
        "tensor_metadata_fields": ["byte_order", "dtype", "order", "shape"],
        "metadata_table_encoding": "canonical_json_sorted_keys_compact_no_nan",
        "trace_field_order": {
            "kinematic": [
                "positions",
                "velocities",
                "substep_positions",
                "substep_velocities",
            ],
            "radius_labelled_physical": ["positions", "velocities", "radius"],
            "camera": [
                "position",
                "target",
                "world_from_camera",
                "camera_from_world",
                "intrinsics",
            ],
            "raster": [
                "full_mask",
                "visible_mask",
                "winner",
                "depth_buffer",
                "support",
            ],
            "conic_geometry": [
                "conic_matrix",
                "conic_eigenvalues",
                "conic_projected_centre_value",
                "conic_input_symmetry_error",
                "ellipse_centre",
                "ellipse_shape",
                "ellipse_covariance",
                "ellipse_coordinate_radius",
                "ellipse_coordinate_extrema",
                "ellipse_enclosing_radius",
                "conic_relative_boundary_residual",
                "conic_centre_residual",
                "conic_shape_residual",
                "conic_valid",
            ],
            "fit_observability": [
                "fit_eigenvalues",
                "fit_condition",
                "fitted_centre",
                "fitted_radius",
                "fit_relative_residual",
                "fit_valid",
            ],
            "combined": [
                "metadata",
                "positions",
                "velocities",
                "radius",
                "albedo",
                "world_from_camera",
                "camera_from_world",
                "intrinsics",
                "raster_fields_in_declared_order",
                "conic_geometry_fields_in_declared_order",
                "fit_observability_fields_in_declared_order",
            ],
            "expected_lifecycle": ["created", "removed", "physical_events"],
        },
        "scene_order": "split_order_then_ordinal_ascending",
        "frame_order": "frame_0_through_55",
        "object_order": "physical_object_0_then_1",
        "split_order": list(SPLITS),
    }


def _descriptor_unsigned() -> dict[str, Any]:
    return {
        "name": "rgbd_two_visible_variable_radius_scene_v1",
        "authority": "literal_source_descriptor_only",
        "split_order": list(SPLITS),
        "scene_axes": {
            "primitives_per_split": PRIMITIVES_PER_SPLIT,
            "pair_variants_per_primitive": PAIR_VARIANTS_PER_PRIMITIVE,
            "radius_roles_per_primitive": RADIUS_ROLES_PER_PRIMITIVE,
            "camera_strata": CAMERA_STRATA,
            "scenes_per_split": SCENES_PER_SPLIT,
            "total_scenes": TOTAL_SCENES,
        },
        "ordinal_mapping": {
            "primitive_index": "ordinal//32",
            "pair_variant": "(ordinal%32)//16",
            "radius_role": "(ordinal%16)//8",
            "camera_stratum": "ordinal%8",
            "radius_role_twin": "ordinal xor 8",
            "unordered_pair_variant_twin": "ordinal xor 16",
        },
        "base_geometry_mapping": {
            "pairs_by_split": {
                split: [list(pair) for pair in SPLIT_PRIMITIVE_PAIRS[split]] for split in SPLITS
            },
            "accepted_rational_rows": [
                {
                    "a": a,
                    "b": b,
                    "position_numerators": [
                        list(row) for row in ACCEPTED_PRIMITIVE_RATIONAL_ROWS[(a, b)][0]
                    ],
                    "velocity_numerators": [
                        list(row) for row in ACCEPTED_PRIMITIVE_RATIONAL_ROWS[(a, b)][1]
                    ],
                }
                for split in SPLITS
                for a, b in SPLIT_PRIMITIVE_PAIRS[split]
            ],
            "position_denominator": 1000,
            "velocity_denominator": 4000,
            "accepted_formula_unchanged": True,
            "accepted_orbital_source_sha256": ACCEPTED_ORBITAL_SOURCE_SHA256,
            "accepted_orbital_certificate_sha256": ACCEPTED_ORBITAL_CERTIFICATE_SHA256,
            "selected_rational_metadata_sha256": ACCEPTED_SELECTED_METADATA_SHA256,
        },
        "radius_mapping": {
            "denominator": RADIUS_DENOMINATOR,
            "explicit_pairs_by_split_primitive_variant": {
                split: [
                    [list(pair) for pair in primitive_pairs]
                    for primitive_pairs in RADIUS_PAIR_NUMERATORS[split]
                ]
                for split in SPLITS
            },
            "role_0_slots": ["low", "high"],
            "role_1_slots": ["high", "low"],
            "pair_variant_shares_exact_kinematics_camera_and_palette": True,
            "minimum_pair_separation_m": MINIMUM_RADIUS_PAIR_SEPARATION_M,
            "consumed_fixed_radius_numerator": CONSUMED_FIXED_RADIUS_NUMERATOR,
            "consumed_fixed_radius_excluded": True,
            "estimator_bounds_m": list(RADIUS_ESTIMATOR_BOUNDS_M),
        },
        "camera": {
            "phase_policy": "shared_across_splits_to_isolate_radius",
            "theta0": ["0", "pi/2", "pi", "3pi/2"],
            "directions": list(CAMERA_DIRECTIONS),
            "theta_law": "theta0+direction*0.24*t",
            "position_law": ["4.6*sin(theta)", "2.15", "4.6*cos(theta)"],
            "target": list(CAMERA_TARGET),
            "vertical_fov_degrees": CAMERA_VERTICAL_FOV_DEGREES,
            "image_size": list(IMAGE_SIZE),
            "calibration_max_abs_error": MAXIMUM_CAMERA_CALIBRATION_ERROR,
            "adjacent_angle_gate_radians": [
                MINIMUM_CAMERA_STEP_ANGLE_RADIANS,
                MAXIMUM_CAMERA_STEP_ANGLE_RADIANS,
            ],
            "adjacent_translation_gate_m": [
                MINIMUM_CAMERA_TRANSLATION_STEP_M,
                MAXIMUM_CAMERA_TRANSLATION_STEP_M,
            ],
        },
        "scene_timing_and_bounds": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FRAME_RATE_HZ,
            "physics_rate_hz": PHYSICS_RATE_HZ,
            "substeps_per_frame": SUBSTEPS_PER_FRAME,
            "physical_substep_count": PHYSICAL_SUBSTEP_COUNT,
            "episode_duration_seconds": (FRAME_COUNT - 1) / FRAME_RATE_HZ,
            "history_frame_count": HISTORY_FRAME_COUNT,
            "history_span_seconds": (HISTORY_FRAME_COUNT - 1) / FRAME_RATE_HZ,
            "world_bounds_m": [list(axis) for axis in WORLD_BOUNDS],
        },
        "controls": {
            "fixed_drag_rational": [FIXED_DRAG_NUMERATOR, FIXED_DRAG_DENOMINATOR],
            "palette": [list(colour) for colour in PALETTE],
            "palette_swap": "(primitive+phase+direction_index)%2==1",
            "radius_role_twin_changes_only_radius_slots": True,
            "pair_variant_twin_changes_only_both_radius_truths": True,
        },
        "continuous_silhouette_certificate": {
            "kind": "pinhole_sphere_homogeneous_conic",
            "formula": "C=K^-T*(c*c^T-(dot(c,c)-r^2)*I)*K^-1",
            "interior_sign": "p^T*C*p>=0",
            "ellipse_coordinate_extrema": "mu_i plus_or_minus sqrt(S_ii)",
            "gap_claim": "enclosing_circle_guaranteed_lower_bound_not_exact_ellipse_distance",
            "image_rectangle": "pixel_centres_[0,width-1]x[0,height-1]",
            "certification_dtype": CONIC_CERTIFICATION_DTYPE,
            "pixel_safety_tolerance": CONIC_PIXEL_SAFETY_TOLERANCE,
            "front_depth_tolerance_m": CONIC_DEPTH_TOLERANCE_M,
            "rho_tolerance_m2": CONIC_RHO_TOLERANCE_M2,
            "relative_algebra_tolerance": CONIC_RELATIVE_ALGEBRA_TOLERANCE,
            "relative_boundary_residual_tolerance": (CONIC_RELATIVE_BOUNDARY_RESIDUAL_TOLERANCE),
        },
        "formal_public_api_policy": {
            "public_physics_on_formal_values": False,
            "public_renderer_on_formal_values": False,
            "public_camera_constructor_on_formal_values": False,
            "independent_fixed_drag_recurrence": True,
            "independent_stable_near_root_raster": True,
            "independent_pinhole_sphere_conic_geometry": True,
            "independent_centered_algebraic_sphere_fit": True,
        },
        "source_bindings": {
            "camera_algorithm": PUBLIC_CAMERA_ALGORITHM_VERSION,
            "camera_sha256": PUBLIC_CAMERA_SOURCE_SHA256,
            "physics_algorithm": PUBLIC_PHYSICS_ALGORITHM_VERSION,
            "physics_sha256": PUBLIC_PHYSICS_SOURCE_SHA256,
            "renderer_algorithm": PUBLIC_RENDERER_ALGORITHM_VERSION,
            "renderer_sha256": PUBLIC_RENDERER_SOURCE_SHA256,
            "accepted_orbital_qualification_sha256": ACCEPTED_ORBITAL_SOURCE_SHA256,
            "accepted_orbital_certificate_sha256": ACCEPTED_ORBITAL_CERTIFICATE_SHA256,
        },
        "accepted_orbital_law_reproduction": {
            "selected_rational_metadata_sha256": ACCEPTED_SELECTED_METADATA_SHA256,
            "selected_kinematic_trace_sha256": dict(FROZEN_ACCEPTED_KINEMATIC_TRACE_SHA256),
            "shared_camera_trace_sha256": dict(FROZEN_ACCEPTED_CAMERA_TRACE_SHA256),
            "exact_copied_metadata_and_trace_assertions": True,
        },
        "determinism_scope": {
            "scope": TORCH_DETERMINISM_SCOPE,
            "torch_version": FROZEN_TORCH_VERSION,
            "python_version": FROZEN_PYTHON_VERSION,
            "platform_system": FROZEN_PLATFORM_SYSTEM,
            "platform_machine": FROZEN_PLATFORM_MACHINE,
            "byteorder": FROZEN_BYTEORDER,
            "device": "cpu",
            "state_dtype": "torch.float32",
            "conic_dtype": CONIC_CERTIFICATION_DTYPE,
            "cross_build_or_cross_platform_digest_portability_claim": False,
        },
        "cache_validation": {
            "balance_result_cached": False,
            "certificate_cache_key": "fresh_exact_input_binding_sha256",
            "fresh_source_pin_validation_before_cache_lookup": True,
            "fresh_source_pin_validation_after_cache_lookup": True,
            "input_binding_components": [
                "scene_module_source_sha256",
                "all_bound_source_sha256",
                "validated_determinism_and_platform_fingerprint",
                "all_uppercase_public_constants_tables_and_frozen_bindings",
            ],
        },
        "gates": {
            "minimum_full_support_pixels": MINIMUM_FULL_SUPPORT_PIXELS,
            "minimum_conic_gap_lower_bound_pixels": (MINIMUM_CONIC_GAP_LOWER_BOUND_PIXELS),
            "minimum_conic_boundary_clearance_pixels": (MINIMUM_CONIC_BOUNDARY_CLEARANCE_PIXELS),
            "minimum_world_surface_gap_m": MINIMUM_WORLD_SURFACE_GAP_M,
            "minimum_world_boundary_m": MINIMUM_WORLD_BOUNDARY_M,
            "minimum_episode_speed_mps": MINIMUM_EPISODE_SPEED_MPS,
            "maximum_episode_speed_mps": MAXIMUM_EPISODE_SPEED_MPS,
            "minimum_history_displacement_m": MINIMUM_HISTORY_DISPLACEMENT_M,
            "minimum_radius_bound_clearance_m": MINIMUM_RADIUS_BOUND_CLEARANCE_M,
            "minimum_radius_pair_separation_m": MINIMUM_RADIUS_PAIR_SEPARATION_M,
            "conic_pixel_safety_tolerance": CONIC_PIXEL_SAFETY_TOLERANCE,
            "conic_depth_tolerance_m": CONIC_DEPTH_TOLERANCE_M,
            "conic_rho_tolerance_m2": CONIC_RHO_TOLERANCE_M2,
            "conic_relative_algebra_tolerance": CONIC_RELATIVE_ALGEBRA_TOLERANCE,
            "conic_relative_boundary_residual_tolerance": (
                CONIC_RELATIVE_BOUNDARY_RESIDUAL_TOLERANCE
            ),
            "maximum_sphere_fit_condition": MAXIMUM_SPHERE_FIT_CONDITION,
            "maximum_sphere_fit_relative_residual": MAXIMUM_SPHERE_FIT_RELATIVE_RESIDUAL,
            "maximum_sphere_fit_radius_error_m": MAXIMUM_SPHERE_FIT_RADIUS_ERROR_M,
            "maximum_sphere_fit_centre_error_m": MAXIMUM_SPHERE_FIT_CENTRE_ERROR_M,
        },
        "digest_recipe": _digest_recipe(),
    }


def certificate_descriptor() -> dict[str, Any]:
    """Return a JSON-literal-only harness provenance descriptor."""

    unsigned = _descriptor_unsigned()
    return {
        **copy.deepcopy(unsigned),
        "descriptor_sha256": canonical_sha256(unsigned),
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }


@lru_cache(maxsize=4)
def _computed_scene_family_certificate_bytes_cached(
    input_binding_sha256: str,
) -> bytes:
    """Compute all 256 formal scenes with independent pure mathematics."""

    if input_binding_sha256 != _fresh_certificate_input_binding_sha256():
        raise RuntimeError("certificate inputs changed before exhaustive computation")
    source_bindings = _validate_public_source_bindings()
    determinism_scope = _validated_determinism_scope()
    balance = _computed_scene_balance_certificate()
    metadata_table = [
        scene_metadata(scene_specification(split, ordinal))
        for split in SPLITS
        for ordinal in range(SCENES_PER_SPLIT)
    ]
    metadata_sha256 = canonical_sha256(metadata_table)

    kinematic_digest = hashlib.sha256()
    radius_physical_digest = hashlib.sha256()
    camera_digest = hashlib.sha256()
    raster_digest = hashlib.sha256()
    conic_digest = hashlib.sha256()
    fit_digest = hashlib.sha256()
    combined_digest = hashlib.sha256()
    lifecycle_digest = hashlib.sha256()
    split_kinematic = {split: hashlib.sha256() for split in SPLITS}
    split_radius_physical = {split: hashlib.sha256() for split in SPLITS}
    split_raster = {split: hashlib.sha256() for split in SPLITS}
    split_conic = {split: hashlib.sha256() for split in SPLITS}
    split_fit = {split: hashlib.sha256() for split in SPLITS}
    split_combined = {split: hashlib.sha256() for split in SPLITS}

    trajectories: dict[tuple[str, int], PurePhysicalTrajectory] = {}
    kinematic_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    unordered_pair_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    radius_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    accepted_kinematic_trace_sha256: dict[str, str] = {}
    accepted_metadata_table: list[dict[str, Any]] = []
    bounds = torch.tensor(WORLD_BOUNDS, dtype=torch.float32)
    minimum_world_surface_gap = math.inf
    minimum_world_boundary = math.inf
    minimum_speed = math.inf
    maximum_speed = 0.0
    minimum_history_displacement = math.inf
    minimum_radius_bound_clearance = math.inf
    minimum_radius_pair_separation_numerator = math.inf

    for split in SPLITS:
        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            base = scene_specification(split, primitive_index * 32)
            trajectory = manual_kinematic_trajectory(base)
            trajectories[(split, primitive_index)] = trajectory
            kinematic_values = (
                ("positions", trajectory.positions),
                ("velocities", trajectory.velocities),
                ("substep_positions", trajectory.substep_positions),
                ("substep_velocities", trajectory.substep_velocities),
            )
            kinematic_signature = _tensor_sequence_sha256(kinematic_values)
            kinematic_signatures_by_split[split].add(kinematic_signature)
            accepted_key = f"a_{base.a}_b_{base.b}"
            accepted_kinematic_trace_sha256[accepted_key] = _tensor_sequence_sha256(
                kinematic_values[:2]
            )
            accepted_metadata_table.append(
                {
                    "a": base.a,
                    "b": base.b,
                    "position_denominator": 1000,
                    "position_numerators": base.position_numerators,
                    "velocity_denominator": 4000,
                    "velocity_numerators": base.velocity_numerators,
                }
            )
            for label, value in kinematic_values:
                qualified = f"{split}/primitive_{primitive_index}/{label}"
                _update_tensor_digest(kinematic_digest, qualified, value)
                _update_tensor_digest(split_kinematic[split], qualified, value)
            speeds = torch.linalg.vector_norm(trajectory.substep_velocities, dim=-1)
            history_displacement = torch.linalg.vector_norm(
                trajectory.positions[HISTORY_FRAME_COUNT - 1] - trajectory.positions[0],
                dim=-1,
            )
            minimum_speed = min(minimum_speed, float(speeds.min()))
            maximum_speed = max(maximum_speed, float(speeds.max()))
            minimum_history_displacement = min(
                minimum_history_displacement,
                float(history_displacement.min()),
            )

            for pair_variant in range(PAIR_VARIANTS_PER_PRIMITIVE):
                pair_base = scene_specification(split, primitive_index * 32 + pair_variant * 16)
                pair_separation_numerator = (
                    pair_base.high_radius_numerator - pair_base.low_radius_numerator
                )
                minimum_radius_pair_separation_numerator = min(
                    minimum_radius_pair_separation_numerator,
                    pair_separation_numerator,
                )
                unordered_pair_values = (
                    ("positions", trajectory.positions),
                    ("velocities", trajectory.velocities),
                    (
                        "unordered_radius",
                        torch.tensor(
                            (
                                pair_base.low_radius_numerator,
                                pair_base.high_radius_numerator,
                            ),
                            dtype=torch.float32,
                        )
                        / RADIUS_DENOMINATOR,
                    ),
                )
                unordered_pair_signatures_by_split[split].add(
                    _tensor_sequence_sha256(unordered_pair_values)
                )
                for radius_role in range(RADIUS_ROLES_PER_PRIMITIVE):
                    specification = scene_specification(
                        split,
                        primitive_index * 32 + pair_variant * 16 + radius_role * 8,
                    )
                    radius = specification.radius_tensor()[:, 0]
                    surface_gap = (
                        torch.linalg.vector_norm(
                            trajectory.substep_positions[:, 0] - trajectory.substep_positions[:, 1],
                            dim=-1,
                        )
                        - radius.sum()
                    )
                    world_boundary = torch.minimum(
                        trajectory.substep_positions - radius[None, :, None] - bounds[:, 0],
                        bounds[:, 1] - trajectory.substep_positions - radius[None, :, None],
                    )
                    minimum_world_surface_gap = min(
                        minimum_world_surface_gap,
                        float(surface_gap.min()),
                    )
                    minimum_world_boundary = min(
                        minimum_world_boundary,
                        float(world_boundary.min()),
                    )
                    minimum_radius_bound_clearance = min(
                        minimum_radius_bound_clearance,
                        float(radius.min()) - RADIUS_ESTIMATOR_BOUNDS_M[0],
                        RADIUS_ESTIMATOR_BOUNDS_M[1] - float(radius.max()),
                    )
                    radius_values = (
                        ("positions", trajectory.positions),
                        ("velocities", trajectory.velocities),
                        ("radius", radius),
                    )
                    radius_signature = _tensor_sequence_sha256(radius_values)
                    radius_signatures_by_split[split].add(radius_signature)
                    for label, value in radius_values:
                        qualified = (
                            f"{split}/primitive_{primitive_index}"
                            f"/pair_variant_{pair_variant}/role_{radius_role}/{label}"
                        )
                        _update_tensor_digest(radius_physical_digest, qualified, value)
                        _update_tensor_digest(split_radius_physical[split], qualified, value)

    kinematic_unique_count = len(set().union(*kinematic_signatures_by_split.values()))
    unordered_pair_unique_count = len(set().union(*unordered_pair_signatures_by_split.values()))
    radius_unique_count = len(set().union(*radius_signatures_by_split.values()))
    if kinematic_unique_count != 8:
        raise RuntimeError("certificate requires eight unique kinematic trajectories")
    if unordered_pair_unique_count != 16:
        raise RuntimeError("certificate requires sixteen unordered-pair trajectories")
    if radius_unique_count != 32:
        raise RuntimeError("certificate requires thirty-two radius-labelled trajectories")
    if _cross_split_match_count(kinematic_signatures_by_split):
        raise RuntimeError("kinematic traces overlap across formal splits")
    if _cross_split_match_count(unordered_pair_signatures_by_split):
        raise RuntimeError("unordered-pair traces overlap across formal splits")
    if _cross_split_match_count(radius_signatures_by_split):
        raise RuntimeError("radius-labelled traces overlap across formal splits")
    if accepted_kinematic_trace_sha256 != FROZEN_ACCEPTED_KINEMATIC_TRACE_SHA256:
        raise RuntimeError("copied accepted kinematic traces differ from their freeze")
    accepted_metadata_sha256 = canonical_sha256(accepted_metadata_table)
    if accepted_metadata_sha256 != ACCEPTED_SELECTED_METADATA_SHA256:
        raise RuntimeError("copied accepted rational metadata differs from its freeze")
    minimum_radius_pair_separation = minimum_radius_pair_separation_numerator / RADIUS_DENOMINATOR
    if minimum_radius_pair_separation < MINIMUM_RADIUS_PAIR_SEPARATION_M:
        raise RuntimeError("a formal unordered radius pair is insufficiently separated")
    if minimum_world_surface_gap < MINIMUM_WORLD_SURFACE_GAP_M:
        raise RuntimeError("a formal trajectory approaches object contact")
    if minimum_world_boundary < MINIMUM_WORLD_BOUNDARY_M:
        raise RuntimeError("a formal trajectory approaches a world boundary")
    if minimum_speed < MINIMUM_EPISODE_SPEED_MPS or maximum_speed > MAXIMUM_EPISODE_SPEED_MPS:
        raise RuntimeError("a formal trajectory speed is outside the accepted control")
    if minimum_history_displacement < MINIMUM_HISTORY_DISPLACEMENT_M:
        raise RuntimeError("a formal history lacks observable displacement")
    if minimum_radius_bound_clearance < MINIMUM_RADIUS_BOUND_CLEARANCE_M:
        raise RuntimeError("a formal radius is too close to an estimator bound")

    camera_tensors: dict[int, dict[str, Tensor]] = {}
    camera_trace_signatures: set[str] = set()
    accepted_camera_trace_sha256: dict[str, str] = {}
    maximum_camera_inverse_error = 0.0
    maximum_camera_orthonormality_error = 0.0
    maximum_camera_radius_error = 0.0
    maximum_camera_height_error = 0.0
    maximum_camera_target_error = 0.0
    maximum_camera_intrinsics_error = 0.0
    maximum_camera_position_binding_error = 0.0
    minimum_camera_step_angle = math.inf
    maximum_camera_step_angle = 0.0
    minimum_camera_translation = math.inf
    maximum_camera_translation = 0.0
    reference_intrinsics = _make_intrinsics()
    reference_target = torch.tensor(CAMERA_TARGET, dtype=torch.float32)

    for camera_stratum in range(CAMERA_STRATA):
        frames = tuple(
            pure_orbital_camera_frame(camera_stratum, frame_index / FRAME_RATE_HZ)
            for frame_index in range(FRAME_COUNT)
        )
        values = _stack_camera_frames(frames)
        camera_tensors[camera_stratum] = values
        signature_values = tuple((label, value) for label, value in values.items())
        camera_signature = _tensor_sequence_sha256(signature_values)
        camera_trace_signatures.add(camera_signature)
        accepted_camera_trace_sha256[str(camera_stratum)] = camera_signature
        for label, value in values.items():
            _update_tensor_digest(
                camera_digest,
                f"camera_stratum_{camera_stratum}/{label}",
                value,
            )
        identity = values["world_from_camera"] @ values["camera_from_world"]
        rotation = values["world_from_camera"][:, :3, :3]
        maximum_camera_inverse_error = max(
            maximum_camera_inverse_error,
            float((identity - torch.eye(4)).abs().max()),
        )
        maximum_camera_orthonormality_error = max(
            maximum_camera_orthonormality_error,
            float((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().max()),
        )
        maximum_camera_radius_error = max(
            maximum_camera_radius_error,
            float(
                (torch.linalg.vector_norm(values["position"][:, [0, 2]], dim=-1) - CAMERA_RADIUS_M)
                .abs()
                .max()
            ),
        )
        maximum_camera_height_error = max(
            maximum_camera_height_error,
            float((values["position"][:, 1] - CAMERA_HEIGHT_M).abs().max()),
        )
        maximum_camera_target_error = max(
            maximum_camera_target_error,
            float((values["target"] - reference_target).abs().max()),
        )
        maximum_camera_intrinsics_error = max(
            maximum_camera_intrinsics_error,
            float((values["intrinsics"] - reference_intrinsics).abs().max()),
        )
        maximum_camera_position_binding_error = max(
            maximum_camera_position_binding_error,
            float((values["world_from_camera"][:, :3, 3] - values["position"]).abs().max()),
        )
        adjacent_position = values["position"][:, [0, 2]]
        cosine = (adjacent_position[:-1] * adjacent_position[1:]).sum(dim=-1) / (
            torch.linalg.vector_norm(adjacent_position[:-1], dim=-1)
            * torch.linalg.vector_norm(adjacent_position[1:], dim=-1)
        )
        angle = torch.acos(cosine.clamp(-1.0, 1.0))
        translation = torch.linalg.vector_norm(
            values["position"][1:] - values["position"][:-1],
            dim=-1,
        )
        minimum_camera_step_angle = min(minimum_camera_step_angle, float(angle.min()))
        maximum_camera_step_angle = max(maximum_camera_step_angle, float(angle.max()))
        minimum_camera_translation = min(minimum_camera_translation, float(translation.min()))
        maximum_camera_translation = max(maximum_camera_translation, float(translation.max()))

    if len(camera_trace_signatures) != 8:
        raise RuntimeError("certificate requires exactly eight shared camera traces")
    if accepted_camera_trace_sha256 != FROZEN_ACCEPTED_CAMERA_TRACE_SHA256:
        raise RuntimeError("copied accepted camera traces differ from their freeze")
    camera_calibration_error = max(
        maximum_camera_inverse_error,
        maximum_camera_orthonormality_error,
        maximum_camera_radius_error,
        maximum_camera_height_error,
        maximum_camera_target_error,
        maximum_camera_intrinsics_error,
        maximum_camera_position_binding_error,
    )
    if camera_calibration_error > MAXIMUM_CAMERA_CALIBRATION_ERROR:
        raise RuntimeError("pure camera calibration differs from the accepted law")
    if not (
        MINIMUM_CAMERA_STEP_ANGLE_RADIANS
        <= minimum_camera_step_angle
        <= maximum_camera_step_angle
        <= MAXIMUM_CAMERA_STEP_ANGLE_RADIANS
    ):
        raise RuntimeError("pure camera angular steps differ from the accepted law")
    if not (
        MINIMUM_CAMERA_TRANSLATION_STEP_M
        <= minimum_camera_translation
        <= maximum_camera_translation
        <= MAXIMUM_CAMERA_TRANSLATION_STEP_M
    ):
        raise RuntimeError("pure camera translation steps differ from the accepted law")

    expected_created = torch.zeros((FRAME_COUNT, 2), dtype=torch.bool)
    expected_created[0] = True
    expected_removed = torch.zeros((FRAME_COUNT, 2), dtype=torch.bool)
    expected_physical_events = torch.zeros((FRAME_COUNT, 2), dtype=torch.bool)
    for label, value in (
        ("created", expected_created),
        ("removed", expected_removed),
        ("physical_events", expected_physical_events),
    ):
        _update_tensor_digest(lifecycle_digest, label, value)

    minimum_support = math.inf
    minimum_raw_conic_gap_lower_bound = math.inf
    minimum_certified_conic_gap_lower_bound = math.inf
    minimum_raw_conic_boundary_clearance = math.inf
    minimum_certified_conic_boundary_clearance = math.inf
    minimum_hit_discriminant = math.inf
    minimum_conic_enclosing_radius = math.inf
    maximum_conic_enclosing_radius = 0.0
    maximum_conic_input_symmetry_error = 0.0
    maximum_conic_relative_boundary_residual = 0.0
    maximum_conic_centre_residual = 0.0
    maximum_conic_shape_residual = 0.0
    minimum_fit_condition = math.inf
    maximum_fit_condition = 0.0
    maximum_fit_relative_residual = 0.0
    maximum_fit_radius_error = 0.0
    maximum_fit_centre_error = 0.0
    raster_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    conic_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    combined_signatures_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}

    for split in SPLITS:
        for ordinal in range(SCENES_PER_SPLIT):
            specification = scene_specification(split, ordinal)
            trajectory = trajectories[(split, specification.primitive_index)]
            radius = specification.radius_tensor()[:, 0]
            camera = camera_tensors[specification.camera_stratum]
            trace = _independent_raster_and_fit_batch(
                trajectory.positions,
                radius,
                camera,
            )
            if bool((trace["full_mask"][:, 0] & trace["full_mask"][:, 1]).any()):
                raise RuntimeError("independent rays found overlapping silhouettes")
            if not torch.equal(trace["visible"], trace["support"]):
                raise RuntimeError("independent rays found an occluded sphere")
            if int(trace["support"].min()) < MINIMUM_FULL_SUPPORT_PIXELS:
                raise RuntimeError("independent rays found insufficient surface support")
            if not bool(trace["conic_valid"].all()):
                raise RuntimeError("independent conic geometry became invalid")
            if not bool(trace["fit_valid"].all()):
                raise RuntimeError("independent sphere observability fit became invalid")

            ellipse_centre = trace["ellipse_centre"]
            enclosing_radius = trace["ellipse_enclosing_radius"]
            raw_conic_gap_lower_bound = torch.linalg.vector_norm(
                ellipse_centre[:, 0] - ellipse_centre[:, 1], dim=-1
            ) - enclosing_radius.sum(dim=-1)
            certified_conic_gap_lower_bound = (
                raw_conic_gap_lower_bound - CONIC_PIXEL_SAFETY_TOLERANCE
            )
            coordinate_extrema = trace["ellipse_coordinate_extrema"]
            raw_conic_boundary_clearance = torch.stack(
                (
                    coordinate_extrema[..., 0],
                    (IMAGE_SIZE[1] - 1.0) - coordinate_extrema[..., 1],
                    coordinate_extrema[..., 2],
                    (IMAGE_SIZE[0] - 1.0) - coordinate_extrema[..., 3],
                ),
                dim=-1,
            )
            certified_conic_boundary_clearance = (
                raw_conic_boundary_clearance - CONIC_PIXEL_SAFETY_TOLERANCE
            )
            hit_discriminant = trace["discriminant"][trace["full_mask"]]
            radius_error = (trace["fitted_radius"] - radius[None]).abs()
            centre_error = torch.linalg.vector_norm(
                trace["fitted_centre"] - trace["points_camera"],
                dim=-1,
            )
            minimum_support = min(minimum_support, float(trace["support"].min()))
            minimum_raw_conic_gap_lower_bound = min(
                minimum_raw_conic_gap_lower_bound,
                float(raw_conic_gap_lower_bound.min()),
            )
            minimum_certified_conic_gap_lower_bound = min(
                minimum_certified_conic_gap_lower_bound,
                float(certified_conic_gap_lower_bound.min()),
            )
            minimum_raw_conic_boundary_clearance = min(
                minimum_raw_conic_boundary_clearance,
                float(raw_conic_boundary_clearance.min()),
            )
            minimum_certified_conic_boundary_clearance = min(
                minimum_certified_conic_boundary_clearance,
                float(certified_conic_boundary_clearance.min()),
            )
            minimum_hit_discriminant = min(
                minimum_hit_discriminant,
                float(hit_discriminant.min()),
            )
            minimum_conic_enclosing_radius = min(
                minimum_conic_enclosing_radius,
                float(enclosing_radius.min()),
            )
            maximum_conic_enclosing_radius = max(
                maximum_conic_enclosing_radius,
                float(enclosing_radius.max()),
            )
            maximum_conic_input_symmetry_error = max(
                maximum_conic_input_symmetry_error,
                float(trace["conic_input_symmetry_error"].max()),
            )
            maximum_conic_relative_boundary_residual = max(
                maximum_conic_relative_boundary_residual,
                float(trace["conic_relative_boundary_residual"].max()),
            )
            maximum_conic_centre_residual = max(
                maximum_conic_centre_residual,
                float(trace["conic_centre_residual"].max()),
            )
            maximum_conic_shape_residual = max(
                maximum_conic_shape_residual,
                float(trace["conic_shape_residual"].max()),
            )
            minimum_fit_condition = min(
                minimum_fit_condition,
                float(trace["fit_condition"].min()),
            )
            maximum_fit_condition = max(
                maximum_fit_condition,
                float(trace["fit_condition"].max()),
            )
            maximum_fit_relative_residual = max(
                maximum_fit_relative_residual,
                float(trace["fit_relative_residual"].max()),
            )
            maximum_fit_radius_error = max(
                maximum_fit_radius_error,
                float(radius_error.max()),
            )
            maximum_fit_centre_error = max(
                maximum_fit_centre_error,
                float(centre_error.max()),
            )
            if float(certified_conic_gap_lower_bound.min()) < MINIMUM_CONIC_GAP_LOWER_BOUND_PIXELS:
                raise RuntimeError("a formal scene violates the conic-separation gate")
            if (
                float(certified_conic_boundary_clearance.min())
                < MINIMUM_CONIC_BOUNDARY_CLEARANCE_PIXELS
            ):
                raise RuntimeError("a formal scene violates the conic-boundary gate")
            if float(trace["fit_condition"].max()) > MAXIMUM_SPHERE_FIT_CONDITION:
                raise RuntimeError("a formal sphere surface is poorly conditioned")
            if float(trace["fit_relative_residual"].max()) > MAXIMUM_SPHERE_FIT_RELATIVE_RESIDUAL:
                raise RuntimeError("a formal sphere surface has excessive fit residual")
            if float(radius_error.max()) > MAXIMUM_SPHERE_FIT_RADIUS_ERROR_M:
                raise RuntimeError("a formal sphere surface does not recover its radius")
            if float(centre_error.max()) > MAXIMUM_SPHERE_FIT_CENTRE_ERROR_M:
                raise RuntimeError("a formal sphere surface does not recover its centre")

            prefix = f"{split}/ordinal_{ordinal}"
            raster_values = (
                ("full_mask", trace["full_mask"].to(torch.uint8)),
                ("visible_mask", trace["visible_mask"].to(torch.uint8)),
                ("winner", trace["winner"]),
                ("depth_buffer", trace["depth_buffer"]),
                ("support", trace["support"]),
            )
            conic_values = (
                ("conic_matrix", trace["conic_matrix"]),
                ("conic_eigenvalues", trace["conic_eigenvalues"]),
                (
                    "conic_projected_centre_value",
                    trace["conic_projected_centre_value"],
                ),
                (
                    "conic_input_symmetry_error",
                    trace["conic_input_symmetry_error"],
                ),
                ("ellipse_centre", trace["ellipse_centre"]),
                ("ellipse_shape", trace["ellipse_shape"]),
                ("ellipse_covariance", trace["ellipse_covariance"]),
                ("ellipse_coordinate_radius", trace["ellipse_coordinate_radius"]),
                ("ellipse_coordinate_extrema", trace["ellipse_coordinate_extrema"]),
                ("ellipse_enclosing_radius", trace["ellipse_enclosing_radius"]),
                (
                    "conic_relative_boundary_residual",
                    trace["conic_relative_boundary_residual"],
                ),
                ("conic_centre_residual", trace["conic_centre_residual"]),
                ("conic_shape_residual", trace["conic_shape_residual"]),
                ("conic_valid", trace["conic_valid"].to(torch.uint8)),
            )
            fit_values = (
                ("fit_eigenvalues", trace["fit_eigenvalues"]),
                ("fit_condition", trace["fit_condition"]),
                ("fitted_centre", trace["fitted_centre"]),
                ("fitted_radius", trace["fitted_radius"]),
                ("fit_relative_residual", trace["fit_relative_residual"]),
                ("fit_valid", trace["fit_valid"].to(torch.uint8)),
            )
            raster_signatures_by_split[split].add(_tensor_sequence_sha256(raster_values))
            conic_signatures_by_split[split].add(_tensor_sequence_sha256(conic_values))
            for label, value in raster_values:
                _update_tensor_digest(raster_digest, f"{prefix}/{label}", value)
                _update_tensor_digest(split_raster[split], f"{prefix}/{label}", value)
            for label, value in conic_values:
                _update_tensor_digest(conic_digest, f"{prefix}/{label}", value)
                _update_tensor_digest(split_conic[split], f"{prefix}/{label}", value)
            for label, value in fit_values:
                _update_tensor_digest(fit_digest, f"{prefix}/{label}", value)
                _update_tensor_digest(split_fit[split], f"{prefix}/{label}", value)

            combined_values = (
                ("positions", trajectory.positions),
                ("velocities", trajectory.velocities),
                ("radius", radius),
                ("albedo", specification.albedo_tensor()),
                ("world_from_camera", camera["world_from_camera"]),
                ("camera_from_world", camera["camera_from_world"]),
                ("intrinsics", camera["intrinsics"]),
                *raster_values,
                *conic_values,
                *fit_values,
            )
            combined_signatures_by_split[split].add(_tensor_sequence_sha256(combined_values))
            _digest_field(
                combined_digest,
                f"{prefix}/metadata",
                _canonical_json_bytes(scene_metadata(specification)),
            )
            _digest_field(
                split_combined[split],
                f"{prefix}/metadata",
                _canonical_json_bytes(scene_metadata(specification)),
            )
            for label, value in combined_values:
                _update_tensor_digest(combined_digest, f"{prefix}/{label}", value)
                _update_tensor_digest(split_combined[split], f"{prefix}/{label}", value)

    raster_unique_count = len(set().union(*raster_signatures_by_split.values()))
    conic_unique_count = len(set().union(*conic_signatures_by_split.values()))
    combined_unique_count = len(set().union(*combined_signatures_by_split.values()))
    if (
        raster_unique_count != TOTAL_SCENES
        or conic_unique_count != TOTAL_SCENES
        or combined_unique_count != TOTAL_SCENES
    ):
        raise RuntimeError("formal joint raster/conic/combined traces must be 256/256 unique")
    if _cross_split_match_count(raster_signatures_by_split):
        raise RuntimeError("formal raster traces overlap across splits")
    if _cross_split_match_count(conic_signatures_by_split):
        raise RuntimeError("formal conic traces overlap across splits")
    if _cross_split_match_count(combined_signatures_by_split):
        raise RuntimeError("formal combined traces overlap across splits")

    trace_sha256 = {
        "metadata": metadata_sha256,
        "balance": balance["balance_sha256"],
        "kinematic": kinematic_digest.hexdigest(),
        "radius_labelled_physical": radius_physical_digest.hexdigest(),
        "camera": camera_digest.hexdigest(),
        "raster": raster_digest.hexdigest(),
        "conic_geometry": conic_digest.hexdigest(),
        "fit_observability": fit_digest.hexdigest(),
        "combined": combined_digest.hexdigest(),
        "expected_lifecycle": lifecycle_digest.hexdigest(),
    }
    split_trace_sha256 = {
        "kinematic": {split: split_kinematic[split].hexdigest() for split in SPLITS},
        "radius_labelled_physical": {
            split: split_radius_physical[split].hexdigest() for split in SPLITS
        },
        "raster": {split: split_raster[split].hexdigest() for split in SPLITS},
        "conic_geometry": {split: split_conic[split].hexdigest() for split in SPLITS},
        "fit_observability": {split: split_fit[split].hexdigest() for split in SPLITS},
        "combined": {split: split_combined[split].hexdigest() for split in SPLITS},
    }
    unsigned: dict[str, Any] = {
        "descriptor_contract_sha256": canonical_sha256(_descriptor_unsigned()),
        "trace_sha256": trace_sha256,
        "split_trace_sha256": split_trace_sha256,
        "source_bindings": source_bindings,
        "determinism_scope": determinism_scope,
        "accepted_orbital_law_reproduction": {
            "source_sha256": source_bindings["accepted_orbital_qualification"],
            "accepted_certificate_sha256": ACCEPTED_ORBITAL_CERTIFICATE_SHA256,
            "selected_rational_metadata_sha256": accepted_metadata_sha256,
            "selected_kinematic_trace_sha256": accepted_kinematic_trace_sha256,
            "shared_camera_trace_sha256": accepted_camera_trace_sha256,
            "copied_metadata_rows_exact": True,
            "copied_fixed_drag_trace_exact": True,
            "copied_camera_trace_exact": True,
        },
        "balance": balance,
        "scene_count": TOTAL_SCENES,
        "frame_count": FRAME_COUNT,
        "scene_frame_count": TOTAL_SCENES * FRAME_COUNT,
        "physical_substep_count": PHYSICAL_SUBSTEP_COUNT,
        "kinematic_trajectory_count": kinematic_unique_count,
        "unordered_pair_labelled_trajectory_count": unordered_pair_unique_count,
        "radius_labelled_trajectory_count": radius_unique_count,
        "shared_camera_trace_count": len(camera_trace_signatures),
        "joint_raster_trace_count": raster_unique_count,
        "joint_conic_trace_count": conic_unique_count,
        "joint_combined_trace_count": combined_unique_count,
        "expected_physical_event_count": 0,
        "expected_lifecycle_frame_zero_birth_count": TOTAL_SCENES * 2,
        "expected_lifecycle_non_frame_zero_birth_count": 0,
        "expected_lifecycle_removal_count": 0,
        "minimum_full_support_pixels": minimum_support,
        "minimum_raw_conic_enclosing_circle_gap_lower_bound_pixels": (
            minimum_raw_conic_gap_lower_bound
        ),
        "minimum_conic_enclosing_circle_gap_lower_bound_pixels": (
            minimum_certified_conic_gap_lower_bound
        ),
        "minimum_raw_conic_coordinate_extrema_boundary_clearance_pixels": (
            minimum_raw_conic_boundary_clearance
        ),
        "minimum_conic_coordinate_extrema_boundary_clearance_pixels": (
            minimum_certified_conic_boundary_clearance
        ),
        "minimum_world_surface_gap_m": minimum_world_surface_gap,
        "minimum_world_boundary_clearance_m": minimum_world_boundary,
        "minimum_episode_speed_mps": minimum_speed,
        "maximum_episode_speed_mps": maximum_speed,
        "minimum_history_displacement_m": minimum_history_displacement,
        "minimum_radius_bound_clearance_m": minimum_radius_bound_clearance,
        "minimum_radius_pair_separation_m": minimum_radius_pair_separation,
        "minimum_hit_discriminant": minimum_hit_discriminant,
        "minimum_conic_enclosing_radius_pixels": minimum_conic_enclosing_radius,
        "maximum_conic_enclosing_radius_pixels": maximum_conic_enclosing_radius,
        "maximum_conic_input_symmetry_error": maximum_conic_input_symmetry_error,
        "maximum_conic_relative_boundary_residual": (maximum_conic_relative_boundary_residual),
        "maximum_conic_centre_residual": maximum_conic_centre_residual,
        "maximum_conic_shape_residual": maximum_conic_shape_residual,
        "conic_certification_dtype": CONIC_CERTIFICATION_DTYPE,
        "conic_pixel_safety_tolerance": CONIC_PIXEL_SAFETY_TOLERANCE,
        "conic_relative_algebra_tolerance": CONIC_RELATIVE_ALGEBRA_TOLERANCE,
        "conic_relative_boundary_residual_tolerance": (CONIC_RELATIVE_BOUNDARY_RESIDUAL_TOLERANCE),
        "minimum_sphere_fit_condition": minimum_fit_condition,
        "maximum_sphere_fit_condition": maximum_fit_condition,
        "maximum_sphere_fit_relative_residual": maximum_fit_relative_residual,
        "maximum_sphere_fit_radius_error_m": maximum_fit_radius_error,
        "maximum_sphere_fit_centre_error_m": maximum_fit_centre_error,
        "maximum_camera_inverse_error": maximum_camera_inverse_error,
        "maximum_camera_orthonormality_error": maximum_camera_orthonormality_error,
        "maximum_camera_radius_error_m": maximum_camera_radius_error,
        "maximum_camera_height_error_m": maximum_camera_height_error,
        "maximum_camera_target_error_m": maximum_camera_target_error,
        "maximum_camera_intrinsics_error": maximum_camera_intrinsics_error,
        "maximum_camera_position_binding_error_m": maximum_camera_position_binding_error,
        "minimum_adjacent_camera_angle_radians": minimum_camera_step_angle,
        "maximum_adjacent_camera_angle_radians": maximum_camera_step_angle,
        "minimum_adjacent_camera_translation_m": minimum_camera_translation,
        "maximum_adjacent_camera_translation_m": maximum_camera_translation,
        "public_physics_calls_on_formal_scenes": 0,
        "public_renderer_calls_on_formal_scenes": 0,
        "public_camera_constructor_calls_on_formal_scenes": 0,
        "digest_recipe": _digest_recipe(),
    }
    result = _canonical_json_bytes({**unsigned, "certificate_sha256": canonical_sha256(unsigned)})
    if input_binding_sha256 != _fresh_certificate_input_binding_sha256():
        raise RuntimeError("certificate inputs changed during exhaustive computation")
    return result


def _computed_scene_family_certificate_bytes() -> bytes:
    """Validate fresh inputs around an exact-key certificate cache lookup."""

    input_binding_sha256 = _fresh_certificate_input_binding_sha256()
    result = _computed_scene_family_certificate_bytes_cached(input_binding_sha256)
    if input_binding_sha256 != _fresh_certificate_input_binding_sha256():
        raise RuntimeError("certificate inputs changed across the cache lookup")
    return result


def _computed_scene_family_certificate() -> dict[str, Any]:
    return json.loads(_computed_scene_family_certificate_bytes())


def scene_family_certificate() -> dict[str, Any]:
    """Return the frozen exhaustive independent certificate."""

    result = _computed_scene_family_certificate()
    if result["trace_sha256"] != FROZEN_TRACE_SHA256:
        raise RuntimeError("variable-radius trace hashes differ from the source freeze")
    if result["split_trace_sha256"] != FROZEN_SPLIT_TRACE_SHA256:
        raise RuntimeError("variable-radius split trace hashes differ from the source freeze")
    if result["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256:
        raise RuntimeError("variable-radius certificate differs from the source freeze")
    return copy.deepcopy(result)


__all__ = [
    "CAMERA_STRATA",
    "FROZEN_CERTIFICATE_SHA256",
    "FROZEN_SPLIT_TRACE_SHA256",
    "FROZEN_TRACE_SHA256",
    "PAIR_VARIANTS_PER_PRIMITIVE",
    "PRIMITIVES_PER_SPLIT",
    "RADIUS_DENOMINATOR",
    "RADIUS_ROLES_PER_PRIMITIVE",
    "SCENES_PER_SPLIT",
    "SPLITS",
    "TOTAL_SCENES",
    "VariableRadiusSceneSpecification",
    "certificate_descriptor",
    "counterfactual_twin_ordinal",
    "family_scene_signature",
    "manual_kinematic_trajectory",
    "pair_variant_twin_ordinal",
    "pure_orbital_camera_frame",
    "reject_formal_public_api_input",
    "scene_balance_certificate",
    "scene_family_certificate",
    "scene_metadata",
    "scene_signature",
    "scene_specification",
    "split_scene_signatures",
]
