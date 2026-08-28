"""Frozen bounded-partial-visibility and one-miss RGB-D qualification.

Protocol inspection and configuration validation are seed-free.  Simulator
state can be materialized only by the runner-private manifest evaluator after a
durable, single-use ledger has issued the exact split authorization.  Runtime
packets contain RGB, the deliberately corrupted depth image, calibration, and
time only; renderer masks and the predeclared miss schedule are retained for
preflight/scoring and never enter :class:`OnlineWorldModel`.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.belief import MotionMode
from world_model.datasets import collate_episodes
from world_model.observations import DirectVelocityEvidence
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.runtime.state import runtime_stream_key
from world_model.simulator.camera import (
    CameraFrame,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    world_to_camera,
)
from world_model.simulator.episode import validate_episode
from world_model.simulator.labels import make_perception_labels, validate_perception_labels
from world_model.simulator.physics import _integrate_free_motion_exact, empty_physics_events
from world_model.simulator.sphere_world import SphereWorld, SphereWorldConfig
from world_model.training.checkpointing import capture_git_metadata, checkpoint_payload
from world_model.training.loop import make_rgbd_packet
from world_model.training.rgbd_online_bridge_qualification import (
    _atomic_temporary,
    _durable_create,
    _durable_replace,
    _fsync_parent,
    _lexists,
    canonical_sha256,
    clean_source,
    sha256_bytes,
    stable_read_bytes,
    validate_distinct_paths,
    validated_sha256,
    write_report_fresh,
)
from world_model.training.rgbd_two_visible_free_motion_qualification import (
    TwoVisibleSceneSpecification,
    _birth_physical_mapping,
    _camera_velocities,
    _event_record,
    _gather_physical_by_slot,
    _install_scene,
    _persistent_runtime_tensor_bytes,
    _process_max_rss_bytes,
    _rmse,
    _stack_records,
    _state_record,
    _storage_alias,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION, __version__

DEVELOPMENT_SEEDS = tuple(range(57_000_000, 57_000_032))
SELECTOR_SEEDS = tuple(range(58_000_000, 58_000_024))
CONFIRMATION_SEEDS = tuple(range(59_000_000, 59_000_024))
FINAL_TEST_SEEDS = tuple(range(60_000_000, 60_000_048))
MANIFESTS: dict[str, tuple[int, ...]] = {
    "development": DEVELOPMENT_SEEDS,
    "selector": SELECTOR_SEEDS,
    "confirmation": CONFIRMATION_SEEDS,
    "final_test": FINAL_TEST_SEEDS,
}
MANIFEST_SHA256 = {
    "development": "ded3d75a7d248e3f9746b03b0cf249f32739208713c4287c45deb5eefd11f8e2",
    "selector": "effa598aa07a44c100da115f71828e00754f181729063899353d22b551f7227a",
    "confirmation": "9240a1dd465574de8ac032e318f3cee618909ed6a5b3e91c5fd8c87bad146cec",
    "final_test": "17fdd50896729b981357960ea0db74ef19e059e21bc8d8e41a7048cf237200a6",
}

INGEST_FRAME_INDICES = tuple(range(18))
LIVE_HISTORY_FRAME_INDICES = tuple(range(2, 18))
ANCHOR_FRAME_INDEX = 17
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TARGET_FRAME_INDICES = (19, 22, 27, 37, 57)
MISS_FRAME_INDICES = (15, 16)
RUNTIME_STREAM_KEY = "rgbd:camera0:rgbd"
OBJECT_INDICES = (0, 1)
AXIS_NAMES = ("x", "y", "z")
VJP_COEFFICIENTS = (0.5, -0.75, 1.25)
VJP_OUTPUTS = (
    "current_position",
    "current_velocity",
    *(f"horizon_{horizon:.2f}_position" for horizon in HORIZONS_SECONDS),
    *(f"horizon_{horizon:.2f}_velocity" for horizon in HORIZONS_SECONDS),
)
STRATUM_NAMES = (
    "separated_no_miss",
    "partial_no_miss",
    "separated_one_miss",
    "partial_one_miss",
)

# The raw-byte binding is rechecked at every non-protocol entry point.  A
# changed YAML requires a new source freeze and this constant must change in
# the same reviewed commit before any manifest authorization can be issued.
FROZEN_CONFIG_SHA256 = "b18f787987394f77771dbf31dae1642bd042b81e64b02a3e93b8cd048dd3416b"
ARCHITECTURE_VERSION = 2
ARCHITECTURE_ATTEMPT = 2
MAX_ARCHITECTURE_ATTEMPTS = 2
OPTIMIZER_UPDATES = 0
EMPTY_MODEL_STATE_SHA256 = canonical_sha256([])

ATTEMPT1_RUN_RELATIVE_PATH = Path("runs/rgbd_partial_visibility_recovery_v1")
ATTEMPT1_DEVELOPMENT_LEDGER_BACKLINK = (
    "/Users/mike/Work/world.model/runs/rgbd_partial_visibility_recovery_v1/"
    "development_attempt_1_access.json"
)
ATTEMPT1_REPORT_SHA256 = "7c08c794690a10d46100b8d17ee448e3a83960d265ec7859bb91cd6d2ac9ca9d"
ATTEMPT1_LEDGER_SHA256 = "e4993abefefe07e0b0fb57a65769fa270012524d62c8ebab4b7db0251979aab4"
ATTEMPT1_CONFIG_SHA256 = "7d563382a8f4b6e301ac30510152f1b1409da32248aacf15dff460ea71d29e2c"
ATTEMPT1_PROTOCOL_SHA256 = "e178d572a238c17eaa4c23f1b0942e2c4e70103a73af3ab51736fffe36b0d8fd"
ATTEMPT1_DEVELOPMENT_MANIFEST_SHA256 = (
    "ca1fb17e87df5216c4429342f74dcccd2c31b11b8d48bb3c76eee27e139cf391"
)
ATTEMPT1_SOURCE_PROVENANCE = {
    "commit": "7e67823667769e47bad3678207f2c01bd3edbfe4",
    "dirty": False,
    "runtime_source_fingerprint": (
        "2345bcf6d785cd864301dbcdcb23cc8f7287f1815615fd1e30e6f635084f12c3"
    ),
    "worktree_fingerprint": ("0d44cabadce831238fe1c8c1cda450677b62f20af3fcf9a411fa4ef621b1842f"),
}

PARTIAL_TEMPLATE_RADIAL_RATE = 0.0085
TEMPLATE_COORDINATE_DENOMINATOR = 16
TEMPLATE_SYMMETRY_COUNT = 8
MINIMUM_RENDERER_DISCRIMINANT_MARGIN = 5.0e-5
MINIMUM_OVERLAP_DEPTH_MARGIN_M = 0.80
MAXIMUM_PROJECTED_CENTRE_DRIFT_PIXELS = 2.0e-5
MAXIMUM_CAMERA_CONJUGACY_ERROR_M = 4.0e-6


@dataclass(frozen=True)
class CameraSpaceSceneTemplate:
    """One rational camera-space primitive before an exact D4 transform."""

    name: str
    severity: str
    midpoint_u_sixteenths: int
    midpoint_v_sixteenths: int
    separation_u_sixteenths: int
    separation_v_sixteenths: int
    front_depth_sixteenths: int
    rear_depth_sixteenths: int
    expected_front_support_pixels: int
    expected_rear_support_pixels: int
    expected_rear_visible_pixels: int


# Every value is an exact multiple of 1/16 in camera/pixel coordinates.  The
# eight partial-severity primitives retain inclusive band admissibility under
# every possible one-pixel support/visibility transition.  Eight separated
# primitives supply the other 64 physical records; D4 is applied only after
# choosing a primitive.
ATTEMPT2_TEMPLATE_TABLE = (
    CameraSpaceSceneTemplate("separated_0", "separated", -70, -2, 144, 16, 78, 98, 29, 18, 18),
    CameraSpaceSceneTemplate("separated_1", "separated", -70, -2, 144, 16, 80, 98, 28, 18, 18),
    CameraSpaceSceneTemplate("separated_2", "separated", -70, -4, 144, 20, 76, 98, 32, 18, 18),
    CameraSpaceSceneTemplate("separated_3", "separated", -70, -6, 144, 24, 76, 98, 32, 18, 18),
    CameraSpaceSceneTemplate("separated_4", "separated", -70, -8, 144, 28, 78, 98, 29, 18, 18),
    CameraSpaceSceneTemplate("separated_5", "separated", -70, -10, 144, 32, 78, 98, 29, 18, 18),
    CameraSpaceSceneTemplate("separated_6", "separated", -70, -10, 144, 32, 80, 98, 28, 18, 18),
    CameraSpaceSceneTemplate("separated_7", "separated", -70, -12, 144, 36, 76, 98, 32, 18, 18),
    CameraSpaceSceneTemplate("mild_0", "mild", -16, -2, 60, 40, 78, 94, 29, 20, 17),
    CameraSpaceSceneTemplate("mild_1", "mild", -16, -2, 60, 40, 78, 96, 29, 20, 17),
    CameraSpaceSceneTemplate("mild_2", "mild", -20, 2, 68, 32, 78, 94, 29, 20, 17),
    CameraSpaceSceneTemplate("mild_3", "mild", -20, 2, 68, 32, 78, 96, 29, 20, 17),
    CameraSpaceSceneTemplate("moderate_0", "moderate", -8, -2, 44, 40, 78, 94, 29, 20, 14),
    CameraSpaceSceneTemplate("moderate_1", "moderate", -8, -2, 44, 40, 78, 96, 29, 20, 14),
    CameraSpaceSceneTemplate("moderate_2", "moderate", -12, 2, 52, 32, 78, 94, 29, 20, 14),
    CameraSpaceSceneTemplate("moderate_3", "moderate", -12, 2, 52, 32, 78, 96, 29, 20, 14),
)
ATTEMPT2_TEMPLATE_TABLE_SHA256 = "c3f17e805de234fecb1f1928b47e8fd2127d608447e7b1e87df9a2ec970ce3aa"
ATTEMPT2_ABSOLUTE_PRIMITIVE_TABLE_SHA256 = (
    "f86f218317d656c16f4c85e5b4a75b2e52094724316a3132b0a6e44715bec86e"
)
ATTEMPT2_ORDERED_PHYSICAL_STATE_SHA256 = (
    "bc3e6349fc0d5effecbb53920a9c4224203067f05306330723f8c75dd9f35c57"
)
ATTEMPT2_PHYSICAL_STATE_SET_SHA256 = (
    "96a53595bf7d21b84fed772baef4b754b6e777b7560a8083d303814fa5f611b5"
)
# Filled by the same independent state reconstruction after sorting the two
# per-object [position, velocity] byte records within every geometry record.
ATTEMPT2_UNORDERED_GEOMETRY_SET_SHA256 = (
    "27a8dabb2d9936e635cde5b2155fffa5eddb89679b477175119917627772cafa"
)
ATTEMPT2_WORLD_TRAJECTORY_SHA256 = (
    "32b34e716ec639cabdd5d36f1c0d30fa17b187546bb5653e4fa7d0a9d6af65d4"
)
ATTEMPT2_RENDERER_TRACE_SHA256 = "4362f06929f8e8958c1f12e8d2077dded6f8dda3bfdb99eed425899bb289f412"

_MODERATE_PARTIAL = (False, True, False, True, False, True, False, True)
_MISS_AT_16 = (False, True, True, False, False, True, False, True)
_REAR_SLOT = (0, 1, 0, 1, 0, 1, 1, 0)
_SEPARATED_TARGET = (1, 0, 0, 1, 1, 0, 0, 1)


@dataclass(frozen=True)
class SceneSchedule:
    """Pure manifest arithmetic; it cannot instantiate simulator state."""

    split: str
    index: int
    stratum_index: int
    stratum: str
    replicate: int
    partial: bool
    severity: str
    miss_frame: int | None
    rear_slot: int | None
    missed_slot: int | None
    palette_swapped: bool


@dataclass(frozen=True)
class PartialVisibilityRecoveryGates:
    """Complete predeclared scalar surface, applied per split and stratum."""

    current_position_rmse_m: float = 0.010
    current_velocity_rmse_mps: float = 0.012
    horizon_position_rmse_m: tuple[float, ...] = (0.011, 0.013, 0.016, 0.022, 0.035)
    horizon_velocity_rmse_mps: float = 0.012
    per_object_axis_position_rmse_m: float = 0.014
    per_object_axis_velocity_rmse_mps: float = 0.016
    maximum_position_error_growth_slope_mps: float = 0.015
    early_stationary_additive_margin_m: float = 0.003
    long_stationary_rmse_ratio: float = 0.80
    zero_velocity_rmse_ratio: float = 0.70
    miss_frame_position_rmse_m: float = 0.010

    identity_switch_count: float = 0.0
    persistent_id_mismatch_count: float = 0.0
    association_coverage: float = 1.0
    ambiguous_pair_count: float = 0.0
    false_miss_association_count: float = 0.0
    false_birth_count: float = 0.0
    death_count: float = 0.0
    reacquisition_latency_frames: float = 1.0
    maximum_missed_steps: float = 1.0
    final_missed_steps: float = 0.0
    missed_target_steps_before: float = 0.0
    missed_target_steps_at_miss: float = 1.0
    missed_target_steps_at_recovery: float = 0.0
    missed_coobject_steps: float = 0.0
    free_motion_mode_value: float = 0.0
    missed_step_trace_mismatch_count: float = 0.0
    runtime_free_mode_mismatch_count: float = 0.0
    rollout_free_mode_mismatch_count: float = 0.0
    active_fraction: float = 1.0
    rollout_active_fraction: float = 1.0

    minimum_hungarian_margin: float = 0.02
    minimum_position_assignment_margin_m: float = 0.25
    minimum_matched_appearance_cosine: float = 0.90
    minimum_cross_appearance_cosine_distance: float = 0.10
    physical_palette_swap_fraction: float = 0.50
    unique_scene_specification_fraction: float = 1.0

    history_sample_count: float = 16.0
    no_miss_history_valid_count: float = 16.0
    missed_history_valid_count: float = 15.0
    history_span_seconds: float = 0.75
    history_span_tolerance_seconds: float = 1.0e-6
    expected_velocity_evidence_coverage: float = 1.0
    false_velocity_evidence_count: float = 0.0
    position_owner_count: float = 1.0
    direct_position_field_count: float = 0.0
    direct_velocity_position_change_max_abs_m: float = 0.0
    direct_metric_position_owner_max_abs_m: float = 1.0e-7
    associator_calls_per_batch: float = 18.0
    direct_velocity_calls_per_batch: float = 3.0
    ingest_calls_per_batch: float = 18.0
    public_predict_calls_per_batch: float = 1.0

    minimum_separated_silhouette_gap_pixels: float = 2.0
    mild_silhouette_gap_min_pixels: float = -1.75
    mild_silhouette_gap_max_pixels: float = -0.35
    moderate_silhouette_gap_min_pixels: float = -2.75
    moderate_silhouette_gap_max_pixels: float = -1.25
    mild_rear_visible_fraction_min: float = 0.80
    mild_rear_visible_fraction_max: float = 0.95
    moderate_rear_visible_fraction_min: float = 0.60
    moderate_rear_visible_fraction_max: float = 0.79
    minimum_boundary_clearance_pixels: float = 2.0
    minimum_world_surface_gap_m: float = 0.50
    minimum_world_boundary_clearance_m: float = 0.10
    minimum_observed_support_fraction: float = 0.35
    maximum_surface_residual_relative_rms: float = 0.05
    maximum_full_silhouette_overlap_fraction: float = 0.60
    maximum_surface_radius_relative_error: float = 0.05
    maximum_surface_fit_condition_number: float = 100.0
    minimum_fitted_boundary_clearance_pixels: float = 2.0
    minimum_full_silhouette_radius_pixels: float = 1.0e-6
    maximum_full_silhouette_gap_abs_pixels: float = 128.0
    maximum_predicted_visibility_error: float = 0.15
    preflight_event_count: float = 0.0

    missed_fast_variance_increment: float = 0.08
    missed_variance_increment_tolerance: float = 1.0e-6
    coobject_variance_increment_max_abs: float = 1.0e-6
    missed_variance_inflation_count: float = 1.0

    semigroup_position_max_abs_m: float = 1.0e-5
    semigroup_velocity_max_abs_mps: float = 1.0e-5
    public_direct_position_max_abs_m: float = 1.0e-6
    public_direct_velocity_max_abs_mps: float = 1.0e-6
    analytic_position_agreement_max_abs_m: float = 2.0e-5
    analytic_velocity_agreement_max_abs_mps: float = 2.0e-5
    public_query_time_max_abs_seconds: float = 1.0e-6

    minimum_input_gradient_l1: float = 1.0e-8
    maximum_input_gradient_l1: float = 1.0e8
    minimum_history_frame_gradient_l1: float = 1.0e-8
    minimum_visible_region_gradient_l1: float = 1.0e-8
    current_position_required_frames: float = 1.0
    maximum_zero_gradient_l1: float = 0.0
    maximum_cross_scene_gradient_l1: float = 0.0
    gradient_audit_scene_count: float = 4.0

    perception_latency_seconds: float = 3.0
    state_only_rollout_latency_seconds: float = 0.075
    persistent_runtime_tensor_state_bytes: int = 65_536
    process_max_rss_bytes: int = 2_500_000_000
    process_rss_delta_bytes: int = 1_000_000_000


DEFAULT_GATES = PartialVisibilityRecoveryGates()


def _assert_seed_namespaces() -> None:
    flattened = [seed for values in MANIFESTS.values() for seed in values]
    if any(not values for values in MANIFESTS.values()):
        raise RuntimeError("every partial-visibility namespace must be nonempty")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("partial-visibility namespaces must be disjoint")
    for split, values in MANIFESTS.items():
        if canonical_sha256(list(values)) != MANIFEST_SHA256[split]:
            raise RuntimeError(f"{split} manifest hash differs from the frozen list")
    _assert_scene_parameter_uniqueness()
    assert_attempt2_admissibility()


def _exact_seed_tuple(seeds: Sequence[int], *, label: str) -> tuple[int, ...]:
    resolved = tuple(seeds)
    if any(type(seed) is not int for seed in resolved):
        raise TypeError(f"{label} seeds must be exact integers")
    return resolved


def scene_schedule(seed: int) -> SceneSchedule:
    """Resolve exact stratum/miss arithmetic without constructing an episode."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("partial-visibility scene seed must be an integer")
    matches = [(split, values) for split, values in MANIFESTS.items() if seed in values]
    if len(matches) != 1:
        raise ValueError("seed is outside the exact partial-visibility manifests")
    split, values = matches[0]
    index = seed - values[0]
    stratum_index = index % 4
    replicate = index // 4
    table_index = replicate % 8
    partial = stratum_index in {1, 3}
    missed = stratum_index in {2, 3}
    severity = (
        "moderate"
        if partial and _MODERATE_PARTIAL[table_index]
        else ("mild" if partial else "separated")
    )
    rear_slot = _REAR_SLOT[table_index] if partial else None
    miss_frame = (16 if _MISS_AT_16[table_index] else 15) if missed else None
    missed_slot = (
        (rear_slot if stratum_index == 3 else _SEPARATED_TARGET[table_index]) if missed else None
    )
    return SceneSchedule(
        split=split,
        index=index,
        stratum_index=stratum_index,
        stratum=STRATUM_NAMES[stratum_index],
        replicate=replicate,
        partial=partial,
        severity=severity,
        miss_frame=miss_frame,
        rear_slot=rear_slot,
        missed_slot=missed_slot,
        palette_swapped=bool((replicate + stratum_index) % 2),
    )


def _d4_coordinate(x: float, y: float, symmetry: int) -> tuple[float, float]:
    """Apply one exact square-grid D4 transform around the principal point."""

    transforms = (
        (x, y),
        (-y, x),
        (-x, -y),
        (y, -x),
        (-x, y),
        (x, -y),
        (y, x),
        (-y, -x),
    )
    if isinstance(symmetry, bool) or not isinstance(symmetry, int):
        raise TypeError("template symmetry must be an exact integer")
    if not 0 <= symmetry < len(transforms):
        raise ValueError("template symmetry is outside D4")
    return transforms[symmetry]


def _templates_for_severity(severity: str) -> tuple[CameraSpaceSceneTemplate, ...]:
    templates = tuple(
        template for template in ATTEMPT2_TEMPLATE_TABLE if template.severity == severity
    )
    expected = 8 if severity == "separated" else 4
    if severity not in {"separated", "mild", "moderate"} or len(templates) != expected:
        raise RuntimeError("attempt-two template table has the wrong severity cardinality")
    return templates


@cache
def _seeds_for_severity(severity: str) -> tuple[int, ...]:
    return tuple(
        seed
        for seeds in MANIFESTS.values()
        for seed in seeds
        if scene_schedule(seed).severity == severity
    )


def _template_assignment(seed: int) -> tuple[CameraSpaceSceneTemplate, int]:
    schedule = scene_schedule(seed)
    ordered = _seeds_for_severity(schedule.severity)
    expected = 64 if schedule.severity == "separated" else 32
    if len(ordered) != expected:
        raise RuntimeError("attempt-two severity population differs from the frozen design")
    rank = ordered.index(seed)
    templates = _templates_for_severity(schedule.severity)
    template_index, symmetry = divmod(rank, TEMPLATE_SYMMETRY_COUNT)
    return templates[template_index], symmetry


def _fixed_intrinsics() -> Tensor:
    return make_intrinsics((64, 64), 48.0, dtype=torch.float32)


def _attempt2_absolute_primitive_table() -> list[dict[str, Any]]:
    """Reconstruct the independently searched absolute front/rear payload."""

    intrinsics = _fixed_intrinsics()
    centre_u = float(intrinsics[0, 2])
    centre_v = float(intrinsics[1, 2])
    denominator = float(TEMPLATE_COORDINATE_DENOMINATOR)
    severity_indices: Counter[str] = Counter()
    table: list[dict[str, Any]] = []
    for template in ATTEMPT2_TEMPLATE_TABLE:
        midpoint_u = template.midpoint_u_sixteenths / denominator
        midpoint_v = template.midpoint_v_sixteenths / denominator
        separation_u = template.separation_u_sixteenths / denominator
        separation_v = template.separation_v_sixteenths / denominator
        table.append(
            {
                "kind": template.severity,
                "index": severity_indices[template.severity],
                "front": [
                    centre_u + midpoint_u - 0.5 * separation_u,
                    centre_v + midpoint_v - 0.5 * separation_v,
                    template.front_depth_sixteenths / denominator,
                ],
                "rear": [
                    centre_u + midpoint_u + 0.5 * separation_u,
                    centre_v + midpoint_v + 0.5 * separation_v,
                    template.rear_depth_sixteenths / denominator,
                ],
            }
        )
        severity_indices[template.severity] += 1
    return table


def _template_camera_state(
    template: CameraSpaceSceneTemplate,
    symmetry: int,
    *,
    intrinsics: Tensor,
) -> tuple[Tensor, Tensor]:
    """Resolve rational template coordinates to float32 position/velocity."""

    denominator = float(TEMPLATE_COORDINATE_DENOMINATOR)
    midpoint = (
        template.midpoint_u_sixteenths / denominator,
        template.midpoint_v_sixteenths / denominator,
    )
    separation = (
        template.separation_u_sixteenths / denominator,
        template.separation_v_sixteenths / denominator,
    )
    canonical_offsets = (
        (midpoint[0] - 0.5 * separation[0], midpoint[1] - 0.5 * separation[1]),
        (midpoint[0] + 0.5 * separation[0], midpoint[1] + 0.5 * separation[1]),
    )
    offsets = tuple(_d4_coordinate(*offset, symmetry) for offset in canonical_offsets)
    depths = (
        template.front_depth_sixteenths / denominator,
        template.rear_depth_sixteenths / denominator,
    )
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points_camera = torch.tensor(
        [
            [offset[0] * depth / fx, offset[1] * depth / fy, depth]
            for offset, depth in zip(offsets, depths, strict=True)
        ],
        dtype=torch.float32,
    )
    velocities_camera = points_camera * PARTIAL_TEMPLATE_RADIAL_RATE
    projected = torch.stack(
        (
            intrinsics[0, 0] * points_camera[:, 0] / points_camera[:, 2] + cx,
            intrinsics[1, 1] * points_camera[:, 1] / points_camera[:, 2] + cy,
        ),
        dim=-1,
    )
    expected_pixels = torch.tensor(
        [[cx + offset[0], cy + offset[1]] for offset in offsets], dtype=torch.float32
    )
    if not torch.equal(projected, expected_pixels):
        raise RuntimeError("template float32 backprojection changed its exact pixel phase")
    return points_camera, velocities_camera


def scene_parameter_record(seed: int) -> dict[str, Any]:
    """Return only parameters consumed by the deterministic scene constructor."""

    schedule = scene_schedule(seed)
    template, symmetry = _template_assignment(seed)
    points_camera, velocities_camera = _template_camera_state(
        template,
        symmetry,
        intrinsics=_fixed_intrinsics(),
    )
    if schedule.partial and schedule.rear_slot == 0:
        points_camera = points_camera.flip(0)
        velocities_camera = velocities_camera.flip(0)
    canonical_albedo = torch.tensor([[0.92, 0.20, 0.14], [0.14, 0.84, 0.30]], dtype=torch.float32)
    albedo = canonical_albedo.flip(0) if schedule.palette_swapped else canonical_albedo
    return {
        "stratum": schedule.stratum,
        "severity": schedule.severity,
        "miss_frame": schedule.miss_frame,
        "rear_slot": schedule.rear_slot,
        "missed_slot": schedule.missed_slot,
        "palette_swapped": schedule.palette_swapped,
        "family": "partial" if schedule.partial else "separated",
        "template": asdict(template),
        "template_symmetry": symmetry,
        "template_table_sha256": ATTEMPT2_TEMPLATE_TABLE_SHA256,
        "camera_position": points_camera.tolist(),
        "camera_velocity": velocities_camera.tolist(),
        "albedo": albedo.tolist(),
    }


@cache
def scene_parameter_signature(seed: int) -> str:
    """Hash consumed pure parameters without constructing simulator state."""

    return canonical_sha256(scene_parameter_record(seed))


SCENE_PARAMETER_SIGNATURE_SHA256 = {
    "development": "8426ea4d0a7e1d507c5d7fc825afa8864ee694a04df622cba955b92ffd4350c0",
    "selector": "d421862763a3e0bc0af042fd81704c836c2123ad0fa260130e791cb250c0b2c7",
    "confirmation": "261f975fcd46795ff9f56c94857de69942ea047455f65cc0341bdc515cc76af5",
    "final_test": "1837d40a35ddba88e3a91f74c5b2c398aa01675ad8e84efa2fe660bbf49e34a2",
}


def _assert_scene_parameter_uniqueness() -> None:
    split_signatures = {
        split: [scene_parameter_signature(seed) for seed in seeds]
        for split, seeds in MANIFESTS.items()
    }
    for split, signatures in split_signatures.items():
        if canonical_sha256(signatures) != SCENE_PARAMETER_SIGNATURE_SHA256[split]:
            raise RuntimeError(f"{split} scene-parameter digest differs from the frozen record")
    signatures = [signature for values in split_signatures.values() for signature in values]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("pure scene-parameter arithmetic aliases across qualification splits")


def scene_specification(
    seed: int,
    camera: CameraFrame,
) -> tuple[SceneSchedule, TwoVisibleSceneSpecification]:
    schedule = scene_schedule(seed)
    parameters = scene_parameter_record(seed)
    if not torch.equal(camera.intrinsics.to(torch.float32), _fixed_intrinsics()):
        raise RuntimeError("attempt-two templates require the frozen fixed-camera intrinsics")
    points_camera = torch.tensor(parameters["camera_position"], dtype=torch.float32)
    velocities_camera = torch.tensor(parameters["camera_velocity"], dtype=torch.float32)
    rotation = camera.world_from_camera[:3, :3].to(torch.float32)
    translation = camera.world_from_camera[:3, 3].to(torch.float32)
    return schedule, TwoVisibleSceneSpecification(
        position=points_camera @ rotation.transpose(0, 1) + translation,
        velocity=velocities_camera @ rotation.transpose(0, 1),
        albedo=torch.tensor(parameters["albedo"], dtype=torch.float32),
        palette_swapped=schedule.palette_swapped,
    )


@dataclass(frozen=True)
class Attempt2AdmissibilityCertificate:
    """Seed-free renderer/physics proof summary for the frozen template table."""

    template_cell_count: int
    physical_record_count: int
    physics_substep_count: int
    template_table_sha256: str
    absolute_primitive_table_sha256: str
    ordered_physical_state_sha256: str
    physical_state_set_sha256: str
    unordered_geometry_set_sha256: str
    world_trajectory_sha256: str
    renderer_trace_sha256: str
    minimum_discriminant_abs_margin: float
    minimum_overlap_depth_margin_m: float
    maximum_projected_centre_drift_pixels: float
    maximum_camera_conjugacy_error_m: float
    minimum_initial_speed_mps: float
    maximum_initial_speed_mps: float
    minimum_world_surface_gap_m: float
    minimum_world_boundary_clearance_m: float
    minimum_image_boundary_clearance_pixels: float
    minimum_separated_silhouette_gap_margin_pixels: float
    minimum_partial_silhouette_band_margin_pixels: float
    minimum_current_visibility_band_margin_fraction: float
    minimum_one_pixel_visibility_band_margin_fraction: float
    minimum_full_support_pixels: int
    minimum_visible_support_pixels: int
    minimum_local_miss_pixels: int


def _renderer_ray_geometry(points_camera: Tensor, intrinsics: Tensor) -> dict[str, Tensor]:
    """Apply the committed float32 ray/discriminant/near-root equations exactly."""

    height = width = 64
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ray_x = (pixel_x - intrinsics[0, 2]) / intrinsics[0, 0]
    ray_y = (pixel_y - intrinsics[1, 2]) / intrinsics[1, 1]
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
    radius = points_camera.new_full((2,), 0.21)
    discriminant = ray_norm_squared.unsqueeze(0) * radius[:, None, None].square() - (
        center_cross_ray.square().sum(dim=-1)
    )
    square_root = discriminant.clamp_min(0.0).sqrt()
    near_root_denominator = ray_dot_center + square_root
    quadratic_constant = (
        points_camera.square().sum(dim=-1)[:, None, None] - radius[:, None, None].square()
    )
    metric_surface_depth = quadratic_constant / near_root_denominator.clamp_min(1.0e-12)
    full_mask = (
        (discriminant >= 0.0)
        & (near_root_denominator > 0.0)
        & (metric_surface_depth > 0.0)
        & torch.isfinite(metric_surface_depth)
    )
    ordered_surface_depth = torch.where(
        full_mask,
        metric_surface_depth,
        torch.full_like(metric_surface_depth, torch.inf),
    )
    depth_buffer, winning_slot = ordered_surface_depth.min(dim=0)
    has_object = torch.isfinite(depth_buffer)
    instance_slot_map = torch.where(
        has_object,
        winning_slot.to(torch.int64),
        torch.full_like(winning_slot, -1, dtype=torch.int64),
    )
    slot_indices = torch.arange(2)[:, None, None]
    visible_mask = full_mask & winning_slot.unsqueeze(0).eq(slot_indices) & has_object.unsqueeze(0)
    projected_centres = torch.stack(
        (
            intrinsics[0, 0] * points_camera[:, 0] / points_camera[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * points_camera[:, 1] / points_camera[:, 2] + intrinsics[1, 2],
        ),
        dim=-1,
    )
    apparent_radius = 0.5 * (intrinsics[0, 0] + intrinsics[1, 1]) * radius / points_camera[:, 2]
    return {
        "discriminant": discriminant,
        "near_root_denominator": near_root_denominator,
        "metric_surface_depth": metric_surface_depth,
        "full_mask": full_mask,
        "visible_mask": visible_mask,
        "instance_slot_map": instance_slot_map,
        "projected_centres": projected_centres,
        "apparent_radius": apparent_radius,
    }


def _exact_float32_free_trajectory(
    initial_position: Tensor,
    initial_velocity: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply the exact float32 six-by-1/120 free recurrence in its input frame."""

    observation_dt = 1.0 / 20.0
    substeps = math.ceil(observation_dt / (1.0 / 120.0))
    sub_dt = observation_dt / substeps
    if substeps != 6 or sub_dt != 1.0 / 120.0:
        raise RuntimeError("attempt-two proof requires exactly six physics substeps per frame")
    position = initial_position.clone()
    velocity = initial_velocity.clone()
    positions = [position]
    velocities = [velocity]
    substep_positions = [position]
    substep_velocities = [velocity]
    drag = torch.full((2, 1), 0.05, dtype=torch.float32)
    gravity = torch.zeros(3, dtype=torch.float32)
    movable = torch.ones(2, dtype=torch.bool)
    for _frame in range(57):
        for _substep in range(substeps):
            position, velocity = _integrate_free_motion_exact(
                position,
                velocity,
                drag,
                gravity,
                sub_dt,
                movable,
            )
            substep_positions.append(position)
            substep_velocities.append(velocity)
        positions.append(position)
        velocities.append(velocity)
    if len(substep_positions) != 343 or len(positions) != 58:
        raise AssertionError("attempt-two trajectory retained the wrong recurrence cardinality")
    return (
        torch.stack(positions),
        torch.stack(velocities),
        torch.stack(substep_positions),
        torch.stack(substep_velocities),
    )


def _exact_float32_world_trajectory(
    initial_camera_position: Tensor,
    initial_camera_velocity: Tensor,
    *,
    world_from_camera: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evolve the exact constructor-consumed world tensors for 342 substeps."""

    if initial_camera_position.shape != (2, 3) or initial_camera_velocity.shape != (2, 3):
        raise ValueError("attempt-two initial camera state must have shape [2,3]")
    if initial_camera_position.dtype != torch.float32 or initial_camera_velocity.dtype != (
        torch.float32
    ):
        raise TypeError("attempt-two initial camera state must remain float32")
    if world_from_camera.shape != (4, 4) or world_from_camera.dtype != torch.float32:
        raise TypeError("attempt-two world_from_camera must be float32 [4,4]")
    rotation = world_from_camera[:3, :3]
    translation = world_from_camera[:3, 3]
    initial_world_position = initial_camera_position @ rotation.transpose(0, 1) + translation
    initial_world_velocity = initial_camera_velocity @ rotation.transpose(0, 1)
    return _exact_float32_free_trajectory(initial_world_position, initial_world_velocity)


def _fixed_world_from_camera() -> Tensor:
    return look_at_world_from_camera(
        torch.tensor([0.0, 2.15, 5.6], dtype=torch.float32),
        torch.tensor([0.0, 0.95, 0.0], dtype=torch.float32),
    )


_INDEPENDENT_D4_MATRICES = (
    ((-1.0, 0.0), (0.0, -1.0)),
    ((-1.0, 0.0), (0.0, 1.0)),
    ((1.0, 0.0), (0.0, -1.0)),
    ((1.0, 0.0), (0.0, 1.0)),
    ((0.0, -1.0), (-1.0, 0.0)),
    ((0.0, -1.0), (1.0, 0.0)),
    ((0.0, 1.0), (-1.0, 0.0)),
    ((0.0, 1.0), (1.0, 0.0)),
)


def _native_float32_bytes(value: Tensor) -> bytes:
    if value.dtype != torch.float32:
        raise TypeError("attempt-two physical state records must remain float32")
    return bytes(value.contiguous().view(torch.uint8).flatten().tolist())


def _attempt2_independent_physical_state_records() -> tuple[tuple[bytes, ...], set[bytes]]:
    """Rebuild all 128 states from the absolute payload and explicit D4 matrices."""

    intrinsics = _fixed_intrinsics()
    centre = intrinsics[0, 2]
    focal = intrinsics[0, 0]
    transform = _fixed_world_from_camera()
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    radial_rate = torch.tensor(PARTIAL_TEMPLATE_RADIAL_RATE, dtype=torch.float32)
    ordered_records: list[bytes] = []
    unordered_geometry_records: set[bytes] = set()
    for primitive in _attempt2_absolute_primitive_table():
        for matrix_values in _INDEPENDENT_D4_MATRICES:
            matrix = torch.tensor(matrix_values, dtype=torch.float32)
            points: list[Tensor] = []
            for role in ("front", "rear"):
                triple = torch.tensor(primitive[role], dtype=torch.float32)
                pixel = centre + matrix @ (triple[:2] - centre)
                depth = triple[2]
                points.append(
                    torch.stack(
                        (
                            (pixel[0] - centre) * depth / focal,
                            (pixel[1] - centre) * depth / focal,
                            depth,
                        )
                    )
                )
            camera_position = torch.stack(points)
            world_position = camera_position @ rotation.transpose(0, 1) + translation
            world_velocity = (radial_rate * camera_position) @ rotation.transpose(0, 1)
            ordered_records.append(
                _native_float32_bytes(
                    torch.cat((world_position.reshape(-1), world_velocity.reshape(-1)))
                )
            )
            object_records = tuple(
                _native_float32_bytes(torch.cat((world_position[slot], world_velocity[slot])))
                for slot in OBJECT_INDICES
            )
            unordered_geometry_records.add(b"".join(sorted(object_records)))
    return tuple(ordered_records), unordered_geometry_records


def _attempt2_admissibility_certificate() -> tuple[
    Attempt2AdmissibilityCertificate, dict[str, tuple[int, int, int]]
]:
    """Compute the finite proof without constructing a simulator or episode."""

    template_digest = canonical_sha256([asdict(template) for template in ATTEMPT2_TEMPLATE_TABLE])
    if not ATTEMPT2_TEMPLATE_TABLE_SHA256.startswith("TO_BE_") and (
        template_digest != ATTEMPT2_TEMPLATE_TABLE_SHA256
    ):
        raise RuntimeError("attempt-two template table hash changed")
    if len({template.name for template in ATTEMPT2_TEMPLATE_TABLE}) != 16:
        raise RuntimeError("attempt-two template names must be unique")
    absolute_primitive_digest = canonical_sha256(_attempt2_absolute_primitive_table())
    if absolute_primitive_digest != ATTEMPT2_ABSOLUTE_PRIMITIVE_TABLE_SHA256:
        raise RuntimeError("attempt-two absolute primitive table hash changed")
    ordered_state_records, independent_unordered_geometry_records = (
        _attempt2_independent_physical_state_records()
    )
    ordered_physical_state_digest = sha256_bytes(b"".join(ordered_state_records))
    physical_state_set_digest = sha256_bytes(b"".join(sorted(ordered_state_records)))
    unordered_geometry_set_digest = sha256_bytes(
        b"".join(sorted(independent_unordered_geometry_records))
    )
    if ordered_physical_state_digest != ATTEMPT2_ORDERED_PHYSICAL_STATE_SHA256:
        raise RuntimeError("attempt-two ordered physical-state hash changed")
    if physical_state_set_digest != ATTEMPT2_PHYSICAL_STATE_SET_SHA256:
        raise RuntimeError("attempt-two physical-state set hash changed")
    if unordered_geometry_set_digest != ATTEMPT2_UNORDERED_GEOMETRY_SET_SHA256:
        raise RuntimeError("attempt-two unordered geometry-set hash changed")
    if (
        len(ordered_state_records) != 128
        or len(set(ordered_state_records)) != 128
        or len(independent_unordered_geometry_records) != 128
    ):
        raise RuntimeError("attempt-two independent physical-state reconstruction aliases")

    assignments: dict[tuple[str, int], tuple[int, SceneSchedule]] = {}
    physical_records: set[bytes] = set()
    for seeds in MANIFESTS.values():
        for seed in seeds:
            schedule = scene_schedule(seed)
            template, symmetry = _template_assignment(seed)
            key = (template.name, symmetry)
            if key in assignments:
                raise RuntimeError("attempt-two seed mapping reused a physical template cell")
            assignments[key] = (seed, schedule)
            points, velocities = _template_camera_state(
                template,
                symmetry,
                intrinsics=_fixed_intrinsics(),
            )
            if schedule.partial and schedule.rear_slot == 0:
                points = points.flip(0)
                velocities = velocities.flip(0)
            transform = _fixed_world_from_camera()
            world_position = points @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]
            world_velocity = velocities @ transform[:3, :3].transpose(0, 1)
            object_records = tuple(
                bytes(
                    torch.cat((world_position[slot], world_velocity[slot]))
                    .contiguous()
                    .view(torch.uint8)
                    .tolist()
                )
                for slot in OBJECT_INDICES
            )
            encoded = b"".join(sorted(object_records))
            if encoded in physical_records:
                raise RuntimeError("attempt-two float32 position/velocity geometry aliases")
            physical_records.add(encoded)
    if len(assignments) != len(ATTEMPT2_TEMPLATE_TABLE) * TEMPLATE_SYMMETRY_COUNT:
        raise RuntimeError("attempt-two manifests do not cover every template cell exactly once")
    if physical_records != independent_unordered_geometry_records:
        raise RuntimeError("attempt-two seed mapping differs from the absolute geometry table")

    intrinsics = _fixed_intrinsics()
    transform = _fixed_world_from_camera()
    camera_from_world = invert_rigid_transform(transform)
    bounds = torch.tensor([[-2.25, 2.25], [0.0, 3.25], [-1.5, 1.5]], dtype=torch.float32)
    world_trajectory_digests: dict[str, str] = {}
    trace_digests: dict[str, str] = {}
    support_records: dict[str, tuple[int, int, int]] = {}
    discriminant_margins: list[float] = []
    overlap_depth_margins: list[float] = []
    projected_drifts: list[float] = []
    camera_conjugacy_errors: list[float] = []
    speed_values: list[float] = []
    surface_gaps: list[float] = []
    boundary_clearances: list[float] = []
    image_boundary_clearances: list[float] = []
    separated_silhouette_margins: list[float] = []
    partial_silhouette_margins: list[float] = []
    current_visibility_margins: list[float] = []
    one_pixel_visibility_margins: list[float] = []
    full_supports: list[int] = []
    visible_supports: list[int] = []
    local_miss_pixels: list[int] = []
    for template in ATTEMPT2_TEMPLATE_TABLE:
        per_template_support: set[tuple[int, int, int]] = set()
        for symmetry in range(TEMPLATE_SYMMETRY_COUNT):
            _seed, schedule = assignments[(template.name, symmetry)]
            points, velocities = _template_camera_state(
                template,
                symmetry,
                intrinsics=intrinsics,
            )
            front_slot, rear_slot = 0, 1
            if schedule.partial and schedule.rear_slot == 0:
                points = points.flip(0)
                velocities = velocities.flip(0)
                front_slot, rear_slot = 1, 0
            (
                _conjugate_camera_positions,
                _conjugate_camera_velocities,
                conjugate_camera_substep_positions,
                _conjugate_camera_substep_velocities,
            ) = _exact_float32_free_trajectory(
                points,
                velocities,
            )
            (
                world_positions,
                world_velocities,
                world_substep_positions,
                world_substep_velocities,
            ) = _exact_float32_world_trajectory(
                points,
                velocities,
                world_from_camera=transform,
            )
            if not torch.equal(world_positions, world_substep_positions[::6]) or not torch.equal(
                world_velocities, world_substep_velocities[::6]
            ):
                raise RuntimeError("attempt-two frame states differ from the 342-substep trace")
            world_trajectory_digests[f"{template.name}/d4_{symmetry}"] = sha256_bytes(
                _native_float32_bytes(world_substep_positions)
                + _native_float32_bytes(world_substep_velocities)
            )
            camera_positions = world_to_camera(world_positions, camera_from_world)
            camera_substep_positions = world_to_camera(
                world_substep_positions,
                camera_from_world,
            )
            conjugacy_error = float(
                (camera_substep_positions - conjugate_camera_substep_positions).abs().max()
            )
            if conjugacy_error > MAXIMUM_CAMERA_CONJUGACY_ERROR_M:
                raise RuntimeError("attempt-two world/camera float32 conjugacy error is too large")
            camera_conjugacy_errors.append(conjugacy_error)
            speed = torch.linalg.vector_norm(world_velocities[0], dim=-1)
            if float(speed.min()) < 0.035 or float(speed.max()) > 0.065:
                raise RuntimeError("attempt-two initial speed left the frozen simulator range")
            speed_values.extend(float(value) for value in speed)
            reference_full: Tensor | None = None
            reference_winner: Tensor | None = None
            reference_centres: Tensor | None = None
            trace = bytearray()
            support_tuple: tuple[int, int, int] | None = None
            coordinate_steps = world_substep_positions[1:] - world_substep_positions[:-1]
            expected_sign = torch.sign(world_substep_velocities[0]).unsqueeze(0)
            if bool((coordinate_steps * expected_sign < -1.0e-7).any()):
                raise RuntimeError("attempt-two world coordinates left their monotonic rays")
            lower = world_substep_positions - 0.21 - bounds[:, 0]
            upper = bounds[:, 1] - world_substep_positions - 0.21
            world_boundary = torch.minimum(lower, upper)
            if float(world_boundary.min()) < DEFAULT_GATES.minimum_world_boundary_clearance_m:
                raise RuntimeError("attempt-two template left the frozen world bounds")
            boundary_clearances.append(float(world_boundary.min()))
            pair_distance = torch.linalg.vector_norm(
                world_substep_positions[:, 0] - world_substep_positions[:, 1], dim=-1
            )
            surface_gap = pair_distance - 0.42
            if float(surface_gap.min()) < DEFAULT_GATES.minimum_world_surface_gap_m:
                raise RuntimeError("attempt-two template violates the noncontact margin")
            if bool((surface_gap[1:] < surface_gap[:-1] - 1.0e-7).any()):
                raise RuntimeError("attempt-two noncontact margin is not monotonic")
            surface_gaps.append(float(surface_gap.min()))
            for frame_index, frame_points in enumerate(camera_positions):
                geometry = _renderer_ray_geometry(frame_points, intrinsics)
                full_mask = geometry["full_mask"]
                winner = geometry["instance_slot_map"]
                if reference_full is None:
                    reference_full = full_mask
                    reference_winner = winner
                    reference_centres = geometry["projected_centres"]
                elif not torch.equal(full_mask, reference_full) or not torch.equal(
                    winner, reference_winner
                ):
                    raise RuntimeError("attempt-two renderer mask/winner trace is not invariant")
                projected_drift = float(
                    (geometry["projected_centres"] - reference_centres).abs().max()
                )
                if projected_drift > MAXIMUM_PROJECTED_CENTRE_DRIFT_PIXELS:
                    raise RuntimeError("attempt-two radial motion changed projected pixel phase")
                projected_drifts.append(projected_drift)
                discriminant_margin = float(geometry["discriminant"].abs().min())
                if discriminant_margin < MINIMUM_RENDERER_DISCRIMINANT_MARGIN:
                    raise RuntimeError(
                        "attempt-two template lacks a renderer-cell margin: "
                        f"{template.name}/d4_{symmetry}/frame_{frame_index}="
                        f"{discriminant_margin:.9g}"
                    )
                discriminant_margins.append(discriminant_margin)
                visible_mask = geometry["visible_mask"]
                full_count = full_mask.sum(dim=(-2, -1))
                visible_count = visible_mask.sum(dim=(-2, -1))
                full_supports.extend(int(value) for value in full_count)
                current_support = (
                    int(full_count[front_slot]),
                    int(full_count[rear_slot]),
                    int(visible_count[rear_slot]),
                )
                if support_tuple is None:
                    support_tuple = current_support
                elif current_support != support_tuple:
                    raise RuntimeError("attempt-two exact renderer support counts changed")
                if int(visible_count[front_slot]) != int(full_count[front_slot]):
                    raise RuntimeError("attempt-two front sphere is not exactly fully visible")
                if int(visible_count.min()) < 4:
                    raise RuntimeError("attempt-two local miss has inadequate visible support")
                visible_supports.extend(int(value) for value in visible_count)
                if schedule.miss_frame is not None:
                    if schedule.missed_slot not in OBJECT_INDICES:
                        raise RuntimeError("attempt-two miss schedule lacks an exact target slot")
                    local_miss_pixels.append(int(visible_count[schedule.missed_slot]))
                overlap = full_mask[front_slot] & full_mask[rear_slot]
                centres = geometry["projected_centres"]
                radii = geometry["apparent_radius"]
                image_boundary = torch.stack(
                    (
                        centres[:, 0] - radii,
                        63.0 - centres[:, 0] - radii,
                        centres[:, 1] - radii,
                        63.0 - centres[:, 1] - radii,
                    ),
                    dim=-1,
                )
                if float(image_boundary.min()) < DEFAULT_GATES.minimum_boundary_clearance_pixels:
                    raise RuntimeError("attempt-two template lacks image-boundary clearance")
                image_boundary_clearances.append(float(image_boundary.min()))
                silhouette_gap = float(
                    torch.linalg.vector_norm(centres[front_slot] - centres[rear_slot]) - radii.sum()
                )
                if template.severity == "separated":
                    if bool(overlap.any()) or silhouette_gap < (
                        DEFAULT_GATES.minimum_separated_silhouette_gap_pixels
                    ):
                        raise RuntimeError("attempt-two separated template overlaps")
                    separated_silhouette_margins.append(
                        silhouette_gap - DEFAULT_GATES.minimum_separated_silhouette_gap_pixels
                    )
                else:
                    if not bool(overlap.any()):
                        raise RuntimeError("attempt-two partial template lacks overlap")
                    depth = geometry["metric_surface_depth"]
                    overlap_margin = float((depth[rear_slot] - depth[front_slot])[overlap].min())
                    if overlap_margin < MINIMUM_OVERLAP_DEPTH_MARGIN_M:
                        raise RuntimeError("attempt-two overlap lacks a near-root depth margin")
                    overlap_depth_margins.append(overlap_margin)
                    bounds_for_severity = (
                        (
                            DEFAULT_GATES.mild_rear_visible_fraction_min,
                            DEFAULT_GATES.mild_rear_visible_fraction_max,
                            DEFAULT_GATES.mild_silhouette_gap_min_pixels,
                            DEFAULT_GATES.mild_silhouette_gap_max_pixels,
                        )
                        if template.severity == "mild"
                        else (
                            DEFAULT_GATES.moderate_rear_visible_fraction_min,
                            DEFAULT_GATES.moderate_rear_visible_fraction_max,
                            DEFAULT_GATES.moderate_silhouette_gap_min_pixels,
                            DEFAULT_GATES.moderate_silhouette_gap_max_pixels,
                        )
                    )
                    visible_min, visible_max, gap_min, gap_max = bounds_for_severity
                    rear_support = int(full_count[rear_slot])
                    rear_visible = int(visible_count[rear_slot])
                    current_ratio = rear_visible / rear_support
                    current_visibility_margins.append(
                        min(current_ratio - visible_min, visible_max - current_ratio)
                    )
                    one_pixel_ratios = (
                        (rear_visible - 1) / rear_support,
                        (rear_visible + 1) / rear_support,
                        (rear_visible + 1) / (rear_support + 1),
                        rear_visible / (rear_support + 1),
                        (rear_visible - 1) / (rear_support - 1),
                        rear_visible / (rear_support - 1),
                    )
                    one_pixel_visibility_margins.append(
                        min(
                            min(ratio - visible_min, visible_max - ratio)
                            for ratio in one_pixel_ratios
                        )
                    )
                    if not all(visible_min <= ratio <= visible_max for ratio in one_pixel_ratios):
                        raise RuntimeError(
                            "attempt-two visibility lacks inclusive one-pixel band admissibility: "
                            f"{template.name}/d4_{symmetry}/frame_{frame_index}="
                            f"{rear_visible}/{rear_support}"
                        )
                    if not gap_min <= silhouette_gap <= gap_max:
                        raise RuntimeError("attempt-two silhouette gap left its frozen band")
                    partial_silhouette_margins.append(
                        min(silhouette_gap - gap_min, gap_max - silhouette_gap)
                    )
                trace.extend(bytes(full_mask.to(torch.uint8).contiguous().flatten().tolist()))
                trace.extend(bytes((winner + 1).to(torch.uint8).contiguous().flatten().tolist()))
            if support_tuple is None:
                raise AssertionError("attempt-two template produced no renderer frames")
            expected_support = (
                template.expected_front_support_pixels,
                template.expected_rear_support_pixels,
                template.expected_rear_visible_pixels,
            )
            if expected_support != (0, 0, 0) and support_tuple != expected_support:
                raise RuntimeError("attempt-two support differs from the frozen template table")
            per_template_support.add(support_tuple)
            trace_digests[f"{template.name}/d4_{symmetry}"] = sha256_bytes(bytes(trace))
        if len(per_template_support) != 1:
            raise RuntimeError("attempt-two D4 transforms changed exact support counts")
        support_records[template.name] = next(iter(per_template_support))
    world_trajectory_digest = canonical_sha256(world_trajectory_digests)
    if not ATTEMPT2_WORLD_TRAJECTORY_SHA256.startswith("TO_BE_") and (
        world_trajectory_digest != ATTEMPT2_WORLD_TRAJECTORY_SHA256
    ):
        raise RuntimeError("attempt-two exact world-trajectory hash changed")
    trace_digest = canonical_sha256(trace_digests)
    if not ATTEMPT2_RENDERER_TRACE_SHA256.startswith("TO_BE_") and (
        trace_digest != ATTEMPT2_RENDERER_TRACE_SHA256
    ):
        raise RuntimeError("attempt-two renderer mask/winner trace hash changed")
    certificate = Attempt2AdmissibilityCertificate(
        template_cell_count=len(assignments),
        physical_record_count=len(physical_records),
        physics_substep_count=342,
        template_table_sha256=template_digest,
        absolute_primitive_table_sha256=absolute_primitive_digest,
        ordered_physical_state_sha256=ordered_physical_state_digest,
        physical_state_set_sha256=physical_state_set_digest,
        unordered_geometry_set_sha256=unordered_geometry_set_digest,
        world_trajectory_sha256=world_trajectory_digest,
        renderer_trace_sha256=trace_digest,
        minimum_discriminant_abs_margin=min(discriminant_margins),
        minimum_overlap_depth_margin_m=min(overlap_depth_margins),
        maximum_projected_centre_drift_pixels=max(projected_drifts),
        maximum_camera_conjugacy_error_m=max(camera_conjugacy_errors),
        minimum_initial_speed_mps=min(speed_values),
        maximum_initial_speed_mps=max(speed_values),
        minimum_world_surface_gap_m=min(surface_gaps),
        minimum_world_boundary_clearance_m=min(boundary_clearances),
        minimum_image_boundary_clearance_pixels=min(image_boundary_clearances),
        minimum_separated_silhouette_gap_margin_pixels=min(separated_silhouette_margins),
        minimum_partial_silhouette_band_margin_pixels=min(partial_silhouette_margins),
        minimum_current_visibility_band_margin_fraction=min(current_visibility_margins),
        minimum_one_pixel_visibility_band_margin_fraction=min(one_pixel_visibility_margins),
        minimum_full_support_pixels=min(full_supports),
        minimum_visible_support_pixels=min(visible_supports),
        minimum_local_miss_pixels=min(local_miss_pixels),
    )
    return certificate, support_records


@cache
def attempt2_admissibility_certificate() -> Attempt2AdmissibilityCertificate:
    certificate, _support = _attempt2_admissibility_certificate()
    if certificate.template_table_sha256 != ATTEMPT2_TEMPLATE_TABLE_SHA256:
        raise RuntimeError("attempt-two template table is not frozen")
    if certificate.absolute_primitive_table_sha256 != ATTEMPT2_ABSOLUTE_PRIMITIVE_TABLE_SHA256:
        raise RuntimeError("attempt-two absolute primitive table is not frozen")
    if (
        certificate.ordered_physical_state_sha256 != ATTEMPT2_ORDERED_PHYSICAL_STATE_SHA256
        or certificate.physical_state_set_sha256 != ATTEMPT2_PHYSICAL_STATE_SET_SHA256
        or certificate.unordered_geometry_set_sha256 != ATTEMPT2_UNORDERED_GEOMETRY_SET_SHA256
    ):
        raise RuntimeError("attempt-two physical-state digests are not frozen")
    if certificate.world_trajectory_sha256 != ATTEMPT2_WORLD_TRAJECTORY_SHA256:
        raise RuntimeError("attempt-two exact world trajectory is not frozen")
    if certificate.renderer_trace_sha256 != ATTEMPT2_RENDERER_TRACE_SHA256:
        raise RuntimeError("attempt-two renderer trace is not frozen")
    return certificate


def assert_attempt2_admissibility() -> None:
    certificate = attempt2_admissibility_certificate()
    if certificate.template_cell_count != 128 or certificate.physical_record_count != 128:
        raise RuntimeError("attempt-two admissibility proof lost exact physical coverage")


def bridge_protocol() -> dict[str, Any]:
    """Return the canonical self-hashed protocol without materializing data."""

    _assert_seed_namespaces()
    admissibility = attempt2_admissibility_certificate()
    protocol: dict[str, Any] = {
        "name": "rgbd_partial_visibility_recovery_v2",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
        "optimizer": None,
        "optimizer_updates": 0,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "source_binding": {
            "clean_committed_source_required": True,
            "runtime_and_worktree_fingerprints_required": True,
            "development_checkpoint_report_ledger_must_match": True,
            "attempt1_rejection_archive_required_before_every_authorization": True,
        },
        "attempt1_rejection": {
            "run_relative_path": str(ATTEMPT1_RUN_RELATIVE_PATH),
            "development_ledger_backlink": ATTEMPT1_DEVELOPMENT_LEDGER_BACKLINK,
            "source_provenance": dict(ATTEMPT1_SOURCE_PROVENANCE),
            "config_sha256": ATTEMPT1_CONFIG_SHA256,
            "protocol_sha256": ATTEMPT1_PROTOCOL_SHA256,
            "development_manifest_sha256": ATTEMPT1_DEVELOPMENT_MANIFEST_SHA256,
            "development_report_sha256": ATTEMPT1_REPORT_SHA256,
            "development_ledger_sha256": ATTEMPT1_LEDGER_SHA256,
            "failure": {
                "type": "RuntimeError",
                "message": "partial scene left its declared renderer visibility band",
                "before_model_evaluation": True,
                "protected_data_materialized": False,
            },
            "permanently_unused_protected_namespaces": {
                "selector": "54000000--54000023",
                "confirmation": "55000000--55000023",
                "final_test": "56000000--56000047",
            },
        },
        "manifests": {split: list(values) for split, values in MANIFESTS.items()},
        "manifest_sha256": dict(MANIFEST_SHA256),
        "scene_parameter_signature_sha256": dict(SCENE_PARAMETER_SIGNATURE_SHA256),
        "scene_family": {
            "object_count": 2,
            "strata": list(STRATUM_NAMES),
            "stratum_assignment": "(seed-split_start)%4",
            "severity_table": list(_MODERATE_PARTIAL),
            "miss_at_16_table": list(_MISS_AT_16),
            "rear_slot_table": list(_REAR_SLOT),
            "separated_target_table": list(_SEPARATED_TARGET),
            "palette_swap": "((seed-split_start)//4 + stratum_index)%2",
            "template_table": [asdict(template) for template in ATTEMPT2_TEMPLATE_TABLE],
            "template_table_sha256": ATTEMPT2_TEMPLATE_TABLE_SHA256,
            "template_table_hash_representation": (
                "canonical JSON of CameraSpaceSceneTemplate records including support metadata"
            ),
            "absolute_primitive_table": _attempt2_absolute_primitive_table(),
            "absolute_primitive_table_sha256": ATTEMPT2_ABSOLUTE_PRIMITIVE_TABLE_SHA256,
            "ordered_physical_state_sha256": ATTEMPT2_ORDERED_PHYSICAL_STATE_SHA256,
            "physical_state_set_sha256": ATTEMPT2_PHYSICAL_STATE_SET_SHA256,
            "unordered_geometry_set_sha256": ATTEMPT2_UNORDERED_GEOMETRY_SET_SHA256,
            "world_trajectory_sha256": ATTEMPT2_WORLD_TRAJECTORY_SHA256,
            "physical_state_hash_recipe": {
                "d4_matrix_order": [
                    [list(row) for row in matrix] for matrix in _INDEPENDENT_D4_MATRICES
                ],
                "ordered_record": (
                    "native-little-endian-float32 world_position(front,rear) then "
                    "world_velocity(front,rear)"
                ),
                "physical_state_set": "lexicographically sorted 48-byte ordered records",
                "unordered_geometry_set": (
                    "sort two 24-byte [position,velocity] object records within each "
                    "geometry, then sort 128 geometry records"
                ),
            },
            "renderer_trace_sha256": ATTEMPT2_RENDERER_TRACE_SHA256,
            "template_assignment": ("global_manifest_order_within_severity_then_divmod(rank,8)"),
            "template_coordinate_denominator": TEMPLATE_COORDINATE_DENOMINATOR,
            "template_symmetry_group": "D4",
            "template_symmetry_count": TEMPLATE_SYMMETRY_COUNT,
            "radial_velocity_scale_per_second": PARTIAL_TEMPLATE_RADIAL_RATE,
            "float32_free_motion_substeps": 342,
            "float32_free_motion_coordinate_frame": "constructor_consumed_world_state",
            "renderer_camera_state_transform": "world_to_camera_with_fixed_camera_from_world",
            "maximum_camera_conjugacy_error_m": MAXIMUM_CAMERA_CONJUGACY_ERROR_M,
            "admissibility_certificate": asdict(admissibility),
            "radius_m": 0.21,
            "drag": 0.05,
            "gravity": [0.0, 0.0, 0.0],
            "fixed_camera": True,
            "noncontact": True,
            "partial_visibility": "bounded_never_full_occlusion",
            "miss_operator": "zero_depth_where_instance_slot_map_equals_target",
            "runtime_receives_renderer_truth": False,
            "preflight_all_frames": True,
            "seed_free_renderer_certificate_before_authorization": True,
            "exact_full_mask_and_winner_invariance": True,
            "one_pixel_visibility_band_slack": (
                "inclusive_all_six_single_pixel_support_visibility_transitions"
            ),
            "rejection_sampling": False,
        },
        "runtime": {
            "packet_factory": "world_model.training.loop.make_rgbd_packet",
            "runtime": "world_model.runtime.OnlineWorldModel",
            "ingest_frames": list(INGEST_FRAME_INDICES),
            "live_history_frames": list(LIVE_HISTORY_FRAME_INDICES),
            "anchor_frame": ANCHOR_FRAME_INDEX,
            "miss_frames": list(MISS_FRAME_INDICES),
            "horizons_seconds": list(HORIZONS_SECONDS),
            "target_frames": list(TARGET_FRAME_INDICES),
            "history_size": 16,
            "max_missing_rows": 1,
            "require_latest_valid": True,
            "stream_key": RUNTIME_STREAM_KEY,
            "learned_parameters": 0,
            "association": "hard_Hungarian_discrete_identity_control_only",
        },
        "perception": {
            "bounded_partial_visibility": True,
            "minimum_observed_support_fraction": 0.35,
            "maximum_surface_residual_relative_rms": 0.05,
            "maximum_full_silhouette_overlap_fraction": 0.60,
            "slot_local_validity": True,
            "pair_valid_mask": "valid_mask.all(-1)",
            "renderer_visible_fraction_is_runtime_input": False,
        },
        "differentiability": {
            "inputs": ["rgb", "depth"],
            "outputs": list(VJP_OUTPUTS),
            "coefficients": list(VJP_COEFFICIENTS),
            "audit_offsets": [0, 1, 6, 15],
            "current_position_support": {"frame": 17, "count": 1},
            "no_miss_temporal_support": {"frames": list(range(2, 18)), "count": 16},
            "missed_target_temporal_support": "live_history_except_scheduled_miss_count_15",
            "excluded_frames_exact_zero": [0, 1],
            "scheduled_miss_exact_zero_for_target": True,
            "cross_scene_gradient_exact_zero": True,
        },
        "evidence": {
            "development_attempts_for_architecture_2": 1,
            "architecture_attempts_consumed_after_development": 2,
            "protected_order": ["selector", "confirmation", "final_test"],
            "constructor_single_use_authorization": True,
            "durable_record_before_materialization": True,
            "report_before_terminal_ledger_digest": True,
            "weights_only_empty_state_checkpoint": True,
            "fresh_paths_and_alias_rejection": True,
        },
        "gates": json.loads(json.dumps(asdict(DEFAULT_GATES), allow_nan=False)),
        "execution": {
            "device": "cpu_float32",
            "torch_intraop_threads": 1,
            "batch_size": 4,
            "latency_warmups": 1,
            "perception_latency_repeats": 3,
            "rollout_latency_repeats": 20,
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def assert_rgbd_partial_visibility_config(config: OrpheusConfig) -> None:
    """Reject every semantic deviation from the frozen rung profile."""

    expected_simulator = {
        "image_size": (64, 64),
        "frame_rate": 20,
        "physics_rate": 120,
        "sequence_frames": 58,
        "min_objects": 2,
        "max_objects": 2,
        "gravity": (0.0, 0.0, 0.0),
        "radius_range": (0.21, 0.21),
        "mass_range": (1.0, 1.0),
        "restitution_range": (0.7, 0.7),
        "drag_range": (0.05, 0.05),
        "friction_range": (0.2, 0.2),
        "initial_speed_range": (0.035, 0.065),
        "camera_motion": "fixed",
        "render_noise_std": 0.0,
        "ensure_collision": False,
        "external_impulse_probability": 0.0,
        "scenario_mixture": ("baseline",),
    }
    for name, expected in expected_simulator.items():
        if getattr(config.simulator, name) != expected:
            raise ValueError(f"partial-visibility RGB-D requires simulator.{name}={expected!r}")
    if SphereWorldConfig.from_config(config).camera_fov_degrees != 48.0:
        raise ValueError("partial-visibility RGB-D requires a 48-degree vertical camera FOV")
    if (
        config.project.name != "orpheus-rgbd-partial-visibility-recovery-v2-cpu"
        or config.project.seed != DEVELOPMENT_SEEDS[0]
        or not config.project.deterministic
    ):
        raise ValueError("partial-visibility project seed/determinism differs from protocol")
    if config.device.preference != "cpu" or config.device.cuda_amp or config.device.compile:
        raise ValueError("partial-visibility qualification requires CPU float32")
    if config.model.max_objects != 2 or config.model.state.appearance_dim != 3:
        raise ValueError("partial-visibility qualification requires exactly two RGB-D slots")
    if config.model.rgb.enabled or not config.model.rgbd.enabled:
        raise ValueError("partial-visibility qualification requires composite RGB-D only")
    expected_rgbd = {
        "global_every_steps": 1,
        "proposal_count": 2,
        "world_radius": 0.21,
        "linear_drag": 0.05,
        "foreground_threshold": 0.04,
        "foreground_temperature": 0.01,
        "minimum_mass": 4.0,
        "chromatic_temperature": 0.05,
        "minimum_chromatic_eigengap": 0.01,
        "spatial_temperature_pixels": 1.0,
        "chromatic_centre_blend": 0.0025,
        "minimum_silhouette_gap_pixels": 2.0,
        "minimum_boundary_clearance_pixels": 2.0,
        "maximum_surface_radius_relative_error": 0.05,
        "bounded_partial_visibility": True,
        "minimum_observed_support_fraction": 0.35,
        "maximum_surface_residual_relative_rms": 0.05,
        "maximum_full_silhouette_overlap_fraction": 0.60,
        "measurement_position_variance": 0.000064,
        "temporal_history_size": 16,
        "temporal_min_samples": 16,
        "max_missing_rows": 1,
        "require_latest_valid": True,
        "temporal_min_dt": 0.001,
        "temporal_velocity_variance_floor": 0.000001,
        "temporal_velocity_variance_ceiling": 0.01,
        "fit_conditioning_limit": 100.0,
    }
    for name, expected in expected_rgbd.items():
        if getattr(config.model.rgbd, name) != expected:
            raise ValueError(f"partial-visibility RGB-D requires model.rgbd.{name}={expected!r}")
    if (
        config.model.lifecycle.birth_confidence != 0.5
        or config.model.lifecycle.birth_confirmations != 1
    ):
        raise ValueError("partial-visibility qualification requires one-frame birth semantics")
    association = config.model.association
    for name, expected in {
        "geometry_weight": 1.0,
        "appearance_weight": 0.25,
        "existence_weight": 0.0,
        "mahalanobis_gate": 100.0,
        "maximum_cost": 100.0,
        "ambiguity_margin": 0.02,
        "minimum_measurement_confidence": 0.5,
    }.items():
        if getattr(association, name) != expected:
            raise ValueError(f"partial-visibility RGB-D requires association.{name}={expected!r}")
    if (
        not config.model.dynamics.analytic_free_motion_only
        or config.model.dynamics.attention_residual_enabled
        or config.model.dynamics.max_substep != 1.0 / 120.0
    ):
        raise ValueError("partial-visibility qualification forbids learned dynamics")
    if (
        config.model.filter.enable_learned_corrector
        or config.model.filter.learned_residual_scale != 0.0
        or not config.model.filter.direct_metric_position_update
        or not config.model.filter.innovation_anchored_correction
        or config.model.filter.missed_variance_growth != 0.08
    ):
        raise ValueError("partial-visibility qualification requires the fixed analytic filter")
    if config.model.identification.enabled:
        raise ValueError("partial-visibility qualification forbids parameter identification")
    if (
        config.runtime.modality != "rgbd"
        or tuple(config.runtime.modality_order) != ("debug_oracle", "rgbd")
        or config.runtime.enable_debug_oracle
        or config.runtime.hypothesis_pool_enabled
        or not config.runtime.strict_timestamps
    ):
        raise ValueError("partial-visibility qualification forbids runtime substitution")
    if (
        config.training.batch_size != 4
        or config.training.steps != 1
        or config.training.tbptt_steps != 18
    ):
        raise ValueError(
            "partial-visibility qualification requires batch four/schema step/18 frames"
        )
    if config.training.rgb_pretrain_steps != 0 or config.training.validation_episodes != 32:
        raise ValueError("partial-visibility qualification has no training phase")
    if tuple(config.evaluation.horizons_seconds) != HORIZONS_SECONDS or config.evaluation.rgb_only:
        raise ValueError("partial-visibility horizon/modality protocol changed")
    derived = tuple(
        ANCHOR_FRAME_INDEX + int(round(horizon * config.simulator.frame_rate))
        for horizon in HORIZONS_SECONDS
    )
    if derived != TARGET_FRAME_INDICES or derived[-1] != config.simulator.sequence_frames - 1:
        raise ValueError("partial-visibility target frames differ from the frozen schedule")


def new_public_model(config: OrpheusConfig) -> OnlineWorldModel:
    assert_rgbd_partial_visibility_config(config)
    model = OnlineWorldModel.from_config(config, device="cpu")
    if tuple(model.parameters()) or tuple(model.buffers()) or model.state_dict():
        raise RuntimeError("partial-visibility runtime must own zero parameter/buffer state")
    return model


def _assert_execution_environment() -> None:
    if torch.get_num_threads() != 1:
        raise RuntimeError("partial-visibility qualification requires one Torch intraop thread")


_MANIFEST_ACCESS_AUTHORITY = object()
_EVALUATOR_RESULT_AUTHORITY = object()
_LEDGER_CONSTRUCTION_AUTHORITY = object()


class _ManifestAccessAuthorization:
    """Single-use constructor capability whose receipt is checked per seed."""

    def __init__(
        self,
        authority: object,
        *,
        issuer: object,
        mint: object,
        split: str,
        seeds: Sequence[int],
        ledger_path: Path,
        ledger_kind: str,
        receipt_sha256: str,
    ) -> None:
        if authority is not _MANIFEST_ACCESS_AUTHORITY:
            raise PermissionError("manifest authorization requires durable ledger authority")
        resolved_seeds = _exact_seed_tuple(seeds, label="authorized manifest")
        if split == "development":
            expected_issuer_type = DevelopmentLedger
            expected_path = development_ledger_path()
            expected_kind = DevelopmentLedger.ARTIFACT_KIND
            expected_seeds = DEVELOPMENT_SEEDS
        elif split in ("selector", "confirmation", "final_test"):
            expected_issuer_type = QualificationLedger
            expected_path = qualification_ledger_path()
            expected_kind = QualificationLedger.ARTIFACT_KIND
            expected_seeds = MANIFESTS[split]
        else:
            raise PermissionError("manifest authorization split is not canonical")
        if (
            type(issuer) is not expected_issuer_type
            or getattr(issuer, "_authorization_mint", None) is not mint
        ):
            raise PermissionError("manifest authorization was not minted by its canonical ledger")
        try:
            supplied_path = _absolute_lexical(Path(ledger_path))
            issuer_path = _absolute_lexical(Path(issuer.path))
        except (AttributeError, TypeError, ValueError) as error:
            raise PermissionError("manifest authorization ledger path is not canonical") from error
        if (
            supplied_path != expected_path
            or issuer_path != expected_path
            or type(ledger_kind) is not str
            or ledger_kind != expected_kind
            or resolved_seeds != expected_seeds
        ):
            raise PermissionError("manifest authorization path/kind/split/seeds are not canonical")
        self._issuer = issuer
        self._mint = mint
        self._split = split
        self._seeds = resolved_seeds
        self._ledger_path = expected_path
        self._ledger_kind = ledger_kind
        self._receipt_sha256 = validated_sha256(
            receipt_sha256, label=f"{split} started-ledger receipt SHA-256"
        )
        self._begun = False
        self._finished = False
        self._cursor = 0
        self._result_sha256: str | None = None

    def _require_canonical_mint(self) -> None:
        minted = (
            type(self._issuer) is DevelopmentLedger
            and self._split == "development"
            and self._issuer._authorization is self
        ) or (
            type(self._issuer) is QualificationLedger
            and self._split in QualificationLedger.ORDER
            and self._issuer._authorizations.get(self._split) is self
        )
        if not minted or getattr(self._issuer, "_authorization_mint", None) is not self._mint:
            raise PermissionError("authorization is not the canonical ledger-minted capability")

    def _validate_receipt(self) -> None:
        _require_attempt1_rejection()
        contents = _single_link_read_bytes(self._ledger_path, label=f"{self._split} access ledger")
        if sha256_bytes(contents) != self._receipt_sha256:
            raise RuntimeError("manifest authorization started-ledger receipt bytes changed")
        record = json.loads(contents)
        if not isinstance(record, Mapping) or record.get("artifact_kind") != self._ledger_kind:
            raise RuntimeError("manifest authorization ledger kind changed")
        if self._ledger_kind == DevelopmentLedger.ARTIFACT_KIND:
            valid = (
                record.get("status") == "development_materialization_started"
                and record.get("access_started") is True
            )
        else:
            splits = record.get("splits")
            state = splits.get(self._split) if isinstance(splits, Mapping) else None
            valid = (
                isinstance(state, Mapping)
                and state.get("access_started") is True
                and state.get("status") == "materialization_started"
                and record.get("status") == f"{self._split}_materialization_started"
            )
        if not valid:
            raise RuntimeError("manifest authorization lacks a durable started receipt")

    def begin_manifest(self, split: str, seeds: Sequence[int]) -> None:
        if self._begun or self._finished:
            raise RuntimeError("manifest authorization is single use")
        if split != self._split or _exact_seed_tuple(seeds, label="manifest") != self._seeds:
            raise PermissionError("manifest authorization does not match split/seeds")
        self._validate_receipt()
        self._begun = True

    def authorize_seed(self, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("scene seed must be an exact integer")
        if not self._begun or self._finished or self._cursor >= len(self._seeds):
            raise PermissionError("scene construction lacks an active authorization")
        if seed != self._seeds[self._cursor]:
            raise PermissionError("scene construction differs from authorized manifest order")
        self._validate_receipt()
        self._cursor += 1

    def finish_manifest(self) -> None:
        if not self._begun or self._finished or self._cursor != len(self._seeds):
            raise RuntimeError("authorized manifest was not consumed exactly once")
        self._validate_receipt()
        self._finished = True

    def require_finished(self) -> None:
        if not self._finished or self._cursor != len(self._seeds):
            raise RuntimeError("manifest authorization did not finish")

    def seal_result(self, authority: object, result: Mapping[str, Any]) -> None:
        self.require_finished()
        if authority is not _EVALUATOR_RESULT_AUTHORITY or self._result_sha256 is not None:
            raise PermissionError("manifest result seal requires the single-use evaluator")
        self._result_sha256 = canonical_sha256(result)

    def require_result(self, result: Mapping[str, Any]) -> None:
        self.require_finished()
        if self._result_sha256 is None or self._result_sha256 != canonical_sha256(result):
            raise RuntimeError("ledger result was not sealed by the authorized evaluator")


def _require_ledger_minted_authorization(
    authorization: object,
) -> _ManifestAccessAuthorization:
    if type(authorization) is not _ManifestAccessAuthorization:
        raise PermissionError("materialization requires an exact ledger-minted authorization")
    authorization._require_canonical_mint()
    return authorization


def _numeric(value: Any) -> float | None:
    if type(value) is not float:
        return None
    return value if math.isfinite(value) else None


class _TrackingMetrics(Mapping[str, Any]):
    """Record every gate lookup so persisted metric schemas are exact."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = values
        self.accessed: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.accessed.add(key)
        return self.values[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        self.accessed.add(key)
        return self.values.get(key, default)


class _TrackingGates:
    """Record every dataclass field used by the acceptance implementation."""

    def __init__(self, gates: PartialVisibilityRecoveryGates) -> None:
        self.gates = gates
        self.accessed: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        if name not in asdict(self.gates):
            raise AttributeError(name)
        self.accessed.add(name)
        return getattr(self.gates, name)


def _typed_canonical_equal(actual: Any, expected: Any) -> bool:
    """Compare evidence without Python's bool/int or list/tuple coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False
        return all(_typed_canonical_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _typed_canonical_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    try:
        return actual == expected and canonical_sha256(actual) == canonical_sha256(expected)
    except (TypeError, ValueError):
        return False


def _construct_partial_visibility_episode(
    config: OrpheusConfig,
    seed: int,
    *,
    authorization: _ManifestAccessAuthorization,
) -> dict[str, Any]:
    """Construct, corrupt, and preflight one authorized deterministic episode."""

    _require_attempt1_rejection()
    authorization = _require_ledger_minted_authorization(authorization)
    authorization.authorize_seed(seed)
    assert_rgbd_partial_visibility_config(config)
    resolved = SphereWorldConfig.from_config(config)
    world = SphereWorld(resolved, seed)
    schedule, specification = scene_specification(seed, world.camera_frame(0.0))
    _install_scene(world, specification)
    rgb_frames: list[Tensor] = []
    depth_frames: list[Tensor] = []
    state_records: list[dict[str, Tensor]] = []
    label_records: list[dict[str, Tensor]] = []
    event_records: list[dict[str, Tensor]] = []
    world_from_camera: list[Tensor] = []
    camera_from_world: list[Tensor] = []
    intrinsics: list[Tensor] = []
    camera_position: list[Tensor] = []
    camera_target: list[Tensor] = []
    timestamps = torch.arange(resolved.sequence_frames, dtype=torch.float32) / resolved.frame_rate
    pending = empty_physics_events(resolved.n_max)
    changed_pixels = 0
    unchanged_max_abs = 0.0
    for frame_index, timestamp_tensor in enumerate(timestamps):
        timestamp = float(timestamp_tensor)
        lifecycle = world.apply_lifecycle(frame_index)
        camera = world.camera_frame(timestamp)
        rendered = world.render(camera=camera)
        labels = make_perception_labels(world.state, rendered, resolved.image_size)
        validate_perception_labels(labels, max_objects=2, image_size=resolved.image_size)
        labels = dict(labels)
        labels["qualification_visible_mask"] = torch.stack(
            [rendered.instance_slot_map.eq(slot) for slot in OBJECT_INDICES]
        )
        raw_depth = rendered.depth_buffer.unsqueeze(0)
        depth = raw_depth
        if frame_index == schedule.miss_frame:
            if schedule.missed_slot not in OBJECT_INDICES:
                raise RuntimeError("miss scene omitted a physical target")
            mask = rendered.instance_slot_map.eq(schedule.missed_slot).unsqueeze(0)
            depth = torch.where(mask, torch.zeros_like(raw_depth), raw_depth)
            changed_pixels = int(mask.sum())
            if changed_pixels < int(config.model.rgbd.minimum_mass):
                raise RuntimeError("object-local miss did not remove enough visible target pixels")
            unchanged_max_abs = float(
                torch.where(mask, torch.zeros_like(raw_depth), depth - raw_depth).abs().max()
            )
        rgb_frames.append(rendered.rgb)
        depth_frames.append(depth)
        state_records.append(_state_record(world.state, rendered.visible_fraction))
        label_records.append(labels)
        event_records.append(
            _event_record(
                pending,
                created=lifecycle.created,
                removed=lifecycle.removed,
                interval_start=max(0.0, timestamp - resolved.observation_dt),
            )
        )
        world_from_camera.append(camera.world_from_camera)
        camera_from_world.append(camera.camera_from_world)
        intrinsics.append(camera.intrinsics)
        camera_position.append(camera.position)
        camera_target.append(camera.target)
        if frame_index + 1 < resolved.sequence_frames:
            pending = world.step(resolved.observation_dt)
    world_from = torch.stack(world_from_camera)
    camera_from = torch.stack(camera_from_world)
    intrinsics_tensor = torch.stack(intrinsics)
    linear_camera, angular_camera = _camera_velocities(world_from, timestamps)
    objects = _stack_records(state_records)
    labels = _stack_records(label_records)
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
    episode: dict[str, Any] = {
        "rgb": torch.stack(rgb_frames).to(torch.float32),
        "depth": torch.stack(depth_frames).to(torch.float32),
        "timestamps": timestamps,
        "frame_mask": torch.ones(resolved.sequence_frames, dtype=torch.bool),
        "camera": {
            "world_from_camera": world_from,
            "camera_from_world": camera_from,
            "intrinsics": intrinsics_tensor,
            "position": torch.stack(camera_position),
            "target": torch.stack(camera_target),
            "linear_velocity": linear_camera,
            "angular_velocity": angular_camera,
            "calibrated": torch.ones(resolved.sequence_frames, dtype=torch.bool),
        },
        "objects": objects,
        "events": _stack_records(event_records),
        "labels": labels,
        "seed": int(seed),
        "num_objects": 2,
        "metadata": {
            "simulator": "sphere_world",
            "simulator_version": SIMULATOR_VERSION,
            "distribution": resolved.distribution,
            "scenario": schedule.stratum,
            "camera_trajectory": world.camera.mode,
            "frame_rate": resolved.frame_rate,
            "physics_rate": resolved.physics_rate,
            "palette_swapped": schedule.palette_swapped,
            "qualification": asdict(schedule),
            "miss_changed_pixels": changed_pixels,
            "miss_unchanged_max_abs": unchanged_max_abs,
        },
    }
    validate_episode(episode, resolved)
    preflight_partial_visibility_episode(
        episode,
        config=config,
        specification=specification,
        schedule=schedule,
    )
    return episode


def preflight_partial_visibility_episode(
    episode: Mapping[str, Any],
    *,
    config: OrpheusConfig,
    specification: TwoVisibleSceneSpecification | None = None,
    schedule: SceneSchedule | None = None,
) -> dict[str, float]:
    """Fail closed on geometry, visibility, miss isolation, or physical events."""

    assert_rgbd_partial_visibility_config(config)
    if schedule is None:
        schedule = scene_schedule(int(episode["seed"]))
    objects = episode["objects"]
    labels = episode["labels"]
    events = episode["events"]
    active = objects["active"][:, :2]
    projected = labels["projected_valid"][:, :2]
    visible = objects["visible_fraction"][:, :2]
    if not bool(active.all()) or not bool(projected.all()):
        raise RuntimeError("preflight requires two active, projectable spheres")
    centres = labels["projected_center_pixels"][:, :2]
    radii = labels["apparent_radius"][:, :2]
    silhouette_gap = torch.linalg.vector_norm(centres[:, 0] - centres[:, 1], dim=-1)
    silhouette_gap = silhouette_gap - radii.sum(dim=-1)
    full_mask = labels["full_mask"][:, :2]
    overlap_pixels = (full_mask[:, 0] & full_mask[:, 1]).sum(dim=(-2, -1))
    height, width = config.simulator.image_size
    boundary = torch.stack(
        (
            centres[..., 0] - radii,
            (width - 1) - centres[..., 0] - radii,
            centres[..., 1] - radii,
            (height - 1) - centres[..., 1] - radii,
        ),
        dim=-1,
    )
    camera_depth = labels["camera_depth"][:, :2]
    front_slot = camera_depth.argmin(dim=-1)
    if schedule.partial:
        if schedule.rear_slot not in OBJECT_INDICES:
            raise RuntimeError("partial scene omitted its fixed rear slot")
        expected_front = 1 - schedule.rear_slot
        if not bool(front_slot.eq(expected_front).all()):
            raise RuntimeError("partial scene changed front/rear depth order")
        if not bool(overlap_pixels.gt(0).all()):
            raise RuntimeError("partial scene lost projected overlap")
        if not bool(visible[:, expected_front].eq(1.0).all()):
            raise RuntimeError("partial scene occluded its front sphere")
        rear_visible = visible[:, schedule.rear_slot]
        if schedule.severity == "mild":
            visibility_bounds = (
                DEFAULT_GATES.mild_rear_visible_fraction_min,
                DEFAULT_GATES.mild_rear_visible_fraction_max,
            )
            gap_bounds = (
                DEFAULT_GATES.mild_silhouette_gap_min_pixels,
                DEFAULT_GATES.mild_silhouette_gap_max_pixels,
            )
        else:
            visibility_bounds = (
                DEFAULT_GATES.moderate_rear_visible_fraction_min,
                DEFAULT_GATES.moderate_rear_visible_fraction_max,
            )
            gap_bounds = (
                DEFAULT_GATES.moderate_silhouette_gap_min_pixels,
                DEFAULT_GATES.moderate_silhouette_gap_max_pixels,
            )
        if not bool(
            ((rear_visible >= visibility_bounds[0]) & (rear_visible <= visibility_bounds[1])).all()
        ):
            raise RuntimeError("partial scene left its declared renderer visibility band")
        if not bool(((silhouette_gap >= gap_bounds[0]) & (silhouette_gap <= gap_bounds[1])).all()):
            raise RuntimeError("partial scene left its declared silhouette-gap band")
    else:
        rear_visible = visible.new_ones(visible.shape[0])
        if not bool(visible.eq(1.0).all()) or bool(overlap_pixels.ne(0).any()):
            raise RuntimeError("separated control must retain exact full visibility")
        if float(silhouette_gap.min()) < DEFAULT_GATES.minimum_separated_silhouette_gap_pixels:
            raise RuntimeError("separated control violates the accepted silhouette gap")
    if float(boundary.min()) < DEFAULT_GATES.minimum_boundary_clearance_pixels:
        raise RuntimeError("preflight rejects insufficient image-boundary clearance")
    pair_distance = torch.linalg.vector_norm(
        objects["position"][:, 0] - objects["position"][:, 1], dim=-1
    )
    surface_gap = pair_distance - objects["radius"][:, :2, 0].sum(dim=-1)
    bounds = torch.tensor(config.simulator.world_bounds, dtype=objects["position"].dtype)
    lower = objects["position"][:, :2] - objects["radius"][:, :2] - bounds[:, 0]
    upper = bounds[:, 1] - objects["position"][:, :2] - objects["radius"][:, :2]
    world_boundary = torch.minimum(lower, upper)
    if float(surface_gap.min()) < DEFAULT_GATES.minimum_world_surface_gap_m:
        raise RuntimeError("preflight rejects a near-contact trajectory")
    if float(world_boundary.min()) < DEFAULT_GATES.minimum_world_boundary_clearance_m:
        raise RuntimeError("preflight rejects a near-boundary trajectory")
    event_count = sum(
        int(events[name].ne(0).sum())
        for name in ("collision", "contact", "external_impulse", "removed")
    ) + int(events["created"][1:].sum())
    if event_count or not bool(events["created"][0, :2].all()):
        raise RuntimeError("preflight rejects physical/lifecycle events")
    for name, expected in (("radius", 0.21), ("drag", 0.05)):
        if not bool(objects[name][:, :2].eq(expected).all()):
            raise RuntimeError(f"preflight rejects non-frozen {name}")
    colour_distance = 1.0 - F.cosine_similarity(
        objects["albedo"][:, 0], objects["albedo"][:, 1], dim=-1
    )
    if float(colour_distance.min()) < DEFAULT_GATES.minimum_cross_appearance_cosine_distance:
        raise RuntimeError("preflight rejects weak chromatic separation")
    metadata = episode.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("preflight requires qualification metadata")
    changed = int(metadata.get("miss_changed_pixels", -1))
    unchanged = float(metadata.get("miss_unchanged_max_abs", math.inf))
    if schedule.miss_frame is None:
        if changed != 0 or unchanged != 0.0:
            raise RuntimeError("no-miss stratum contains a sensor intervention")
    elif changed < int(config.model.rgbd.minimum_mass) or unchanged != 0.0:
        raise RuntimeError("miss intervention was absent or changed non-target depth")
    if specification is not None:
        torch.testing.assert_close(objects["position"][0, :2], specification.position)
        torch.testing.assert_close(objects["velocity"][0, :2], specification.velocity)
        torch.testing.assert_close(objects["albedo"][0, :2], specification.albedo)
    return {
        "preflight_minimum_silhouette_gap_pixels": float(silhouette_gap.min()),
        "preflight_maximum_silhouette_gap_pixels": float(silhouette_gap.max()),
        "preflight_minimum_boundary_clearance_pixels": float(boundary.min()),
        "preflight_minimum_world_surface_gap_m": float(surface_gap.min()),
        "preflight_minimum_world_boundary_clearance_m": float(world_boundary.min()),
        "preflight_minimum_rear_visible_fraction": float(rear_visible.min()),
        "preflight_maximum_rear_visible_fraction": float(rear_visible.max()),
        "preflight_minimum_palette_cosine_distance": float(colour_distance.min()),
        "preflight_minimum_overlap_pixels": float(overlap_pixels.min()),
        "preflight_maximum_overlap_pixels": float(overlap_pixels.max()),
        "preflight_event_count": float(event_count),
        "miss_changed_pixels": float(changed),
        "miss_unchanged_max_abs": unchanged,
    }


def _run_public_batch(batch: Mapping[str, Any], config: OrpheusConfig) -> dict[str, Any]:
    """Run the exact 18-frame public path and retain read-only causal audits."""

    batch_size = int(batch["rgb"].shape[0])
    model = new_public_model(config)
    model.eval()
    model.reset(batch_size=batch_size)
    module = model.observation_modules["rgbd"]
    original_project = module.project
    original_match = model.associator.match
    original_velocity = model.updater.correct_direct_velocity
    original_missed = model.updater.uncertainty.missed
    current_frame = -1
    association_records: list[dict[str, Any]] = []
    prediction_records: list[Any] = []
    velocity_records: list[tuple[int, Tensor]] = []
    variance_records: list[tuple[int, Tensor, Tensor]] = []
    correction_position_fields = 0
    correction_position_change_max = 0.0
    ingest_call_count = 0
    public_predict_call_count = 0

    def recording_project(belief: Any, context: Any) -> Any:
        result = original_project(belief, context)
        prediction_records.append(result)
        return result

    def recording_match(belief: Any, measured: Any, predicted: Any) -> Any:
        result = original_match(belief, measured, predicted)
        association_records.append(
            {
                "frame": current_frame,
                "measured": measured,
                "predicted": predicted,
                "result": result,
                "cost": model.associator.cost_matrix(measured, predicted),
            }
        )
        return result

    def recording_velocity(prior: Any, evidence: DirectVelocityEvidence) -> Any:
        nonlocal correction_position_fields, correction_position_change_max
        evidence.validate()
        velocity_records.append((current_frame, evidence.valid_mask.detach().clone()))
        correction_position_fields += sum(
            field is not None
            for field in (
                evidence.position,
                evidence.position_log_variance,
                evidence.position_valid_mask,
            )
        )
        before = prior.objects.position.detach().clone()
        corrected = original_velocity(prior, evidence)
        correction_position_change_max = max(
            correction_position_change_max,
            float((corrected.objects.position - before).abs().max().detach().cpu()),
        )
        return corrected

    def recording_missed(belief: Any, missed_mask: Tensor) -> Any:
        before = belief.objects.fast_log_variance.exp()
        result = original_missed(belief, missed_mask)
        increment = result.objects.fast_log_variance.exp() - before
        variance_records.append(
            (current_frame, missed_mask.detach().clone(), increment.detach().clone())
        )
        return result

    module.project = recording_project  # type: ignore[method-assign]
    model.associator.match = recording_match  # type: ignore[method-assign]
    model.updater.correct_direct_velocity = recording_velocity  # type: ignore[method-assign]
    model.updater.uncertainty.missed = recording_missed  # type: ignore[method-assign]
    beliefs: list[Any] = []
    measurements: list[Any] = []
    for frame_index in INGEST_FRAME_INDICES:
        current_frame = frame_index
        packet = make_rgbd_packet(batch, frame_index)
        if (
            not isinstance(packet.payload, Mapping)
            or set(packet.payload) != {"rgb", "depth"}
            or set(packet.calibration) != {"world_from_camera", "intrinsics"}
            or set(packet.metadata) != {"image_size", "training_frame_index", "depth_semantics"}
        ):
            raise RuntimeError("runtime RGB-D packet schema differs from the public API")
        forbidden_metadata = {
            "labels",
            "instance_map",
            "instance_slot_map",
            "miss_frame",
            "missed_slot",
            "visible_fraction",
            "qualification",
        }
        if forbidden_metadata.intersection(packet.metadata):
            raise RuntimeError("runtime RGB-D packet leaked qualification truth")
        posterior = model.ingest(packet)
        ingest_call_count += 1
        beliefs.append(posterior)
        if model.last_measurements is None:
            raise RuntimeError("partial-visibility runtime omitted a measurement set")
        measurements.append(model.last_measurements)
    trajectory = model.predict(HORIZONS_SECONDS).validate()
    public_predict_call_count += 1
    belief = model.belief
    if belief is None:
        raise RuntimeError("partial-visibility runtime did not retain a belief")
    history = model.state.temporal_histories.get(RUNTIME_STREAM_KEY)
    if not isinstance(history, RGBDTemporalPositionHistory):
        raise RuntimeError("partial-visibility runtime omitted typed temporal history")
    if runtime_stream_key("rgbd", "camera0:rgbd") != RUNTIME_STREAM_KEY:
        raise RuntimeError("partial-visibility runtime stream key changed")
    direct = model.dynamics.predict(belief, 2.0)
    composed = model.dynamics.predict(model.dynamics.predict(belief, 0.75), 1.25)
    expected_times = belief.timestamp[:, None] + belief.timestamp.new_tensor(HORIZONS_SECONDS)
    alias_count = sum(
        (
            _storage_alias(trajectory.positions, belief.objects.position),
            _storage_alias(trajectory.positions, belief.objects.velocity),
            _storage_alias(trajectory.velocities, belief.objects.position),
            _storage_alias(trajectory.velocities, belief.objects.velocity),
        )
    )
    raw = measurements[-1]
    if raw.supported_state_fields != ("position",):
        raise RuntimeError("RGB-D measurement must remain the sole direct position owner")
    raw_position = raw.auxiliary.get("world_position")
    if not isinstance(raw_position, Tensor):
        raise RuntimeError("RGB-D measurement omitted raw world position")
    distance = torch.linalg.vector_norm(
        belief.objects.position[:, :, None] - raw_position[:, None], dim=-1
    )
    direct_cost = distance[:, 0, 0] + distance[:, 1, 1]
    swap_cost = distance[:, 0, 1] + distance[:, 1, 0]
    measurement_map = torch.where(
        (swap_cost < direct_cost)[:, None],
        torch.tensor([1, 0], device=distance.device),
        torch.tensor([0, 1], device=distance.device),
    )
    matched_raw = torch.gather(
        raw_position,
        1,
        measurement_map[..., None].expand(batch_size, 2, 3),
    )
    drag = belief.objects.drag
    horizons = belief.objects.position.new_tensor(HORIZONS_SECONDS)
    decay = torch.exp(-drag[:, None] * horizons[None, :, None, None])
    analytic_velocity = belief.objects.velocity[:, None] * decay
    analytic_position = belief.objects.position[:, None] + belief.objects.velocity[:, None] * (
        (1.0 - decay) / drag[:, None].clamp_min(1.0e-12)
    )
    return {
        "model": model,
        "belief": belief,
        "beliefs": beliefs,
        "trajectory": trajectory,
        "history": history,
        "measurements": measurements,
        "predictions": prediction_records,
        "associations": association_records,
        "velocity_records": velocity_records,
        "variance_records": variance_records,
        "semigroup_position": (composed.objects.position - direct.objects.position).abs(),
        "semigroup_velocity": (composed.objects.velocity - direct.objects.velocity).abs(),
        "public_direct_position": (trajectory.positions[:, -1] - direct.objects.position).abs(),
        "public_direct_velocity": (trajectory.velocities[:, -1] - direct.objects.velocity).abs(),
        "analytic_position_agreement": (trajectory.positions - analytic_position).abs(),
        "analytic_velocity_agreement": (trajectory.velocities - analytic_velocity).abs(),
        "query_time_error": (trajectory.timestamps - expected_times).abs(),
        "output_alias_count": alias_count,
        "position_owner_count": 1 + int(correction_position_fields > 0),
        "direct_position_field_count": correction_position_fields,
        "direct_velocity_position_change_max_abs": correction_position_change_max,
        "direct_metric_position_owner_error": (belief.objects.position - matched_raw).abs(),
        "runtime_tensor_bytes": _persistent_runtime_tensor_bytes(model),
        "ingest_call_count": ingest_call_count,
        "public_predict_call_count": public_predict_call_count,
    }


def _expected_slot_masks(
    schedules: Sequence[SceneSchedule],
    birth_mapping: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return expected per-frame measurement and direct-velocity validity."""

    batch = len(schedules)
    measurement = torch.ones((batch, 18, 2), dtype=torch.bool, device=birth_mapping.device)
    # Frame zero owns two births rather than associations to prior beliefs.
    measurement[:, 0] = False
    velocity = torch.zeros_like(measurement)
    velocity[:, 15:18] = True
    for batch_index, schedule in enumerate(schedules):
        if schedule.miss_frame is None or schedule.missed_slot is None:
            continue
        persistent_target = int(
            torch.nonzero(birth_mapping[batch_index].eq(schedule.missed_slot)).flatten().item()
        )
        measurement[batch_index, schedule.miss_frame, persistent_target] = False
        velocity[batch_index, schedule.miss_frame, persistent_target] = False
    return measurement, velocity


def _association_metrics(
    output: Mapping[str, Any],
    schedules: Sequence[SceneSchedule],
    birth_mapping: Tensor,
) -> dict[str, float]:
    expected_measurement, _ = _expected_slot_masks(schedules, birth_mapping)
    expected_pairs = int(expected_measurement.sum())
    matched = 0
    ambiguous = 0
    false_miss = 0
    hungarian_margins: list[float] = []
    position_margins: list[float] = []
    appearance_cosines: list[float] = []
    cross_appearance: list[float] = []
    for record in output["associations"]:
        frame = int(record["frame"])
        result = record["result"]
        measured = record["measured"]
        predicted = record["predicted"]
        cost = record["cost"]
        for batch_index in range(cost.shape[0]):
            pairs = torch.nonzero(result.pair_mask[batch_index]).flatten()
            matched += int(pairs.numel())
            ambiguous += int(result.ambiguous[batch_index, pairs].sum())
            associated_beliefs = {
                int(result.belief_indices[batch_index, pair]) for pair in pairs.tolist()
            }
            false_miss += sum(
                int(not bool(expected_measurement[batch_index, frame, belief_index]))
                for belief_index in associated_beliefs
            )
            if pairs.numel() != 2:
                continue
            mapping = torch.full((2,), -1, dtype=torch.int64, device=cost.device)
            row = torch.full((2,), -1, dtype=torch.int64, device=cost.device)
            for pair in pairs.tolist():
                mapping[int(result.belief_indices[batch_index, pair])] = result.measurement_indices[
                    batch_index, pair
                ]
            for prediction_row in (
                torch.nonzero(predicted.valid_mask[batch_index]).flatten().tolist()
            ):
                row[int(predicted.belief_indices[batch_index, prediction_row])] = prediction_row
            if bool((mapping < 0).any()) or bool((row < 0).any()):
                continue
            selected = cost[batch_index, row[0], mapping[0]] + cost[batch_index, row[1], mapping[1]]
            alternate = (
                cost[batch_index, row[0], mapping[1]] + cost[batch_index, row[1], mapping[0]]
            )
            margin = alternate - selected
            if not bool(torch.isfinite(margin)):
                margin = selected.new_tensor(200.0) - selected
            hungarian_margins.append(float(margin.detach().cpu()))
            predicted_position = torch.stack(
                [predicted.values[batch_index, row[index]] for index in OBJECT_INDICES]
            )
            measured_position = measured.values[batch_index, mapping]
            correct = torch.linalg.vector_norm(
                predicted_position - measured_position, dim=-1
            ).mean()
            cross = torch.linalg.vector_norm(
                predicted_position - measured_position.flip(0), dim=-1
            ).mean()
            position_margins.append(float((cross - correct).detach().cpu()))
            if predicted.appearance is not None and measured.appearance is not None:
                predicted_appearance = torch.stack(
                    [predicted.appearance[batch_index, row[index]] for index in OBJECT_INDICES]
                )
                measured_appearance = measured.appearance[batch_index, mapping]
                appearance_cosines.extend(
                    F.cosine_similarity(predicted_appearance, measured_appearance, dim=-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                cross_appearance.extend(
                    (
                        1.0
                        - F.cosine_similarity(
                            predicted_appearance, measured_appearance.flip(0), dim=-1
                        )
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
    return {
        "association_matched_count": float(matched),
        "association_expected_count": float(expected_pairs),
        "association_pair_coverage": matched / expected_pairs if expected_pairs else 0.0,
        "association_ambiguous_pair_count": float(ambiguous),
        "false_miss_association_count": float(false_miss),
        "minimum_hungarian_margin": min(hungarian_margins, default=float("nan")),
        "minimum_position_assignment_margin_m": min(position_margins, default=float("nan")),
        "minimum_matched_appearance_cosine": min(appearance_cosines, default=float("nan")),
        "minimum_cross_appearance_cosine_distance": min(cross_appearance, default=float("nan")),
    }


def _recovery_metrics(
    output: Mapping[str, Any],
    schedules: Sequence[SceneSchedule],
    birth_mapping: Tensor,
) -> dict[str, float]:
    expected_measurement, expected_velocity = _expected_slot_masks(schedules, birth_mapping)
    actual_measurement = torch.zeros_like(expected_measurement)
    for record in output["associations"]:
        frame = int(record["frame"])
        result = record["result"]
        for batch_index in range(result.pair_mask.shape[0]):
            for pair in torch.nonzero(result.pair_mask[batch_index]).flatten().tolist():
                belief_index = int(result.belief_indices[batch_index, pair])
                actual_measurement[batch_index, frame, belief_index] = True
    actual_velocity = torch.zeros_like(expected_velocity)
    for frame, valid in output["velocity_records"]:
        actual_velocity[:, frame] |= valid
    expected_total = int(expected_velocity.sum())
    valid_expected = int((actual_velocity & expected_velocity).sum())
    false_velocity = int((actual_velocity & ~expected_velocity).sum())
    association_frame_counts = Counter(int(record["frame"]) for record in output["associations"])
    velocity_frame_counts = Counter(int(frame) for frame, _valid in output["velocity_records"])
    association_frame_mismatches = sum(
        abs(association_frame_counts.get(frame, 0) - 1) for frame in range(18)
    ) + sum(count for frame, count in association_frame_counts.items() if frame not in range(18))
    velocity_frame_mismatches = sum(
        abs(velocity_frame_counts.get(frame, 0) - 1) for frame in range(15, 18)
    ) + sum(count for frame, count in velocity_frame_counts.items() if frame not in range(15, 18))
    latencies: list[int] = []
    max_missed_steps = 0
    final_missed_steps = 0
    identity_mismatches = 0
    active_count = 0
    active_total = 0
    target_steps_before: list[int] = []
    target_steps_at_miss: list[int] = []
    target_steps_at_recovery: list[int] = []
    coobject_steps: list[int] = []
    free_mode_mismatches = 0
    recovery_mode_values: list[int] = []
    missed_step_trace_mismatches = 0
    runtime_free_mode_mismatches = 0
    for batch_index, schedule in enumerate(schedules):
        persistent_target: int | None = None
        if schedule.miss_frame is not None and schedule.missed_slot is not None:
            persistent_target = int(
                torch.nonzero(birth_mapping[batch_index].eq(schedule.missed_slot)).flatten().item()
            )
        for frame, belief in enumerate(output["beliefs"]):
            active_count += int(belief.objects.active[batch_index].sum())
            active_total += 2
            identity_mismatches += int(
                belief.objects.object_id[batch_index]
                .ne(torch.tensor([0, 1], device=belief.device))
                .sum()
            )
            max_missed_steps = max(
                max_missed_steps, int(belief.objects.missed_steps[batch_index].max())
            )
            expected_steps = torch.zeros_like(belief.objects.missed_steps[batch_index])
            if persistent_target is not None and frame == schedule.miss_frame:
                expected_steps[persistent_target] = 1
            missed_step_trace_mismatches += int(
                belief.objects.missed_steps[batch_index].ne(expected_steps).sum()
            )
            if frame > 0:
                runtime_free_mode_mismatches += int(
                    (
                        belief.objects.mode[batch_index].ne(int(MotionMode.FREE))
                        & belief.objects.active[batch_index]
                    ).sum()
                )
            if frame == 17:
                final_missed_steps = max(
                    final_missed_steps, int(belief.objects.missed_steps[batch_index].max())
                )
        if schedule.miss_frame is not None and schedule.missed_slot is not None:
            if persistent_target is None:
                raise AssertionError("miss schedule omitted its persistent target")
            target = persistent_target
            coobject = 1 - target
            miss_frame = schedule.miss_frame
            target_steps_before.append(
                int(output["beliefs"][miss_frame - 1].objects.missed_steps[batch_index, target])
            )
            target_steps_at_miss.append(
                int(output["beliefs"][miss_frame].objects.missed_steps[batch_index, target])
            )
            target_steps_at_recovery.append(
                int(output["beliefs"][miss_frame + 1].objects.missed_steps[batch_index, target])
            )
            coobject_steps.extend(
                int(output["beliefs"][frame].objects.missed_steps[batch_index, coobject])
                for frame in (miss_frame - 1, miss_frame, miss_frame + 1)
            )
            for frame in (miss_frame, miss_frame + 1):
                mode = output["beliefs"][frame].objects.mode[batch_index]
                recovery_mode_values.extend((int(mode[target]), int(mode[coobject])))
                free_mode_mismatches += int(mode[target].ne(int(MotionMode.FREE))) + int(
                    mode[coobject].ne(int(MotionMode.FREE))
                )
            later = torch.nonzero(
                actual_measurement[batch_index, schedule.miss_frame + 1 :, target]
            ).flatten()
            latencies.append(int(later[0]) + 1 if later.numel() else 10_000)
    target_increment_errors: list[float] = []
    coobject_increments: list[float] = []
    inflation_count = 0
    variance_increment_minima: list[float] = []
    variance_increment_maxima: list[float] = []
    missed_mask_mismatches = 0
    for frame, missed_mask, increment in output["variance_records"]:
        for batch_index, schedule in enumerate(schedules):
            expected_mask = torch.zeros_like(missed_mask[batch_index])
            if (
                schedule.miss_frame is not None
                and schedule.missed_slot is not None
                and frame == schedule.miss_frame
            ):
                target = int(
                    torch.nonzero(birth_mapping[batch_index].eq(schedule.missed_slot))
                    .flatten()
                    .item()
                )
                expected_mask[target] = True
                target_increment_errors.append(
                    float(
                        (
                            increment[batch_index, target]
                            - DEFAULT_GATES.missed_fast_variance_increment
                        )
                        .abs()
                        .max()
                    )
                )
                variance_increment_minima.append(float(increment[batch_index, target].min()))
                variance_increment_maxima.append(float(increment[batch_index, target].max()))
            missed_mask_mismatches += int(missed_mask[batch_index].ne(expected_mask).sum())
            inflation_count += int(missed_mask[batch_index].sum())
            unchanged = increment[batch_index, ~expected_mask]
            if unchanged.numel():
                coobject_increments.append(float(unchanged.abs().max()))
    expected_measurement_counts = torch.full(
        (len(schedules), 18), 2, dtype=torch.int64, device=birth_mapping.device
    )
    for batch_index, schedule in enumerate(schedules):
        if schedule.miss_frame is not None:
            expected_measurement_counts[batch_index, schedule.miss_frame] = 1
    actual_measurement_counts = torch.stack(
        [measurements.measurement_mask.sum(dim=-1) for measurements in output["measurements"]],
        dim=1,
    )
    rollout = output["trajectory"]
    rollout_free_mode_mismatches = int(
        (
            rollout.motion_mode_logits.argmax(dim=-1).ne(int(MotionMode.FREE)) & rollout.active_mask
        ).sum()
    )
    return {
        "measurement_validity_mismatch_count": float(
            actual_measurement_counts.ne(expected_measurement_counts).sum()
        ),
        "association_validity_mismatch_count": float(
            (actual_measurement != expected_measurement).sum()
        ),
        "expected_velocity_evidence_coverage": (
            valid_expected / expected_total if expected_total else 0.0
        ),
        "false_velocity_evidence_count": float(false_velocity),
        "associator_call_count": float(len(output["associations"])),
        "associator_call_frame_mismatch_count": float(association_frame_mismatches),
        "direct_velocity_call_count": float(len(output["velocity_records"])),
        "direct_velocity_call_frame_mismatch_count": float(velocity_frame_mismatches),
        "reacquisition_latency_frames_max": float(max(latencies, default=1)),
        "maximum_missed_steps": float(max_missed_steps),
        "final_missed_steps_max": float(final_missed_steps),
        "missed_target_steps_before_min": float(min(target_steps_before)),
        "missed_target_steps_before_max": float(max(target_steps_before)),
        "missed_target_steps_at_miss_min": float(min(target_steps_at_miss)),
        "missed_target_steps_at_miss_max": float(max(target_steps_at_miss)),
        "missed_target_steps_at_recovery_min": float(min(target_steps_at_recovery)),
        "missed_target_steps_at_recovery_max": float(max(target_steps_at_recovery)),
        "missed_coobject_steps_min": float(min(coobject_steps)),
        "missed_coobject_steps_max": float(max(coobject_steps)),
        "recovery_free_mode_mismatch_count": float(free_mode_mismatches),
        "missed_step_trace_mismatch_count": float(missed_step_trace_mismatches),
        "runtime_free_mode_mismatch_count": float(runtime_free_mode_mismatches),
        "rollout_free_mode_mismatch_count": float(rollout_free_mode_mismatches),
        "recovery_mode_value_min": float(min(recovery_mode_values)),
        "recovery_mode_value_max": float(max(recovery_mode_values)),
        "persistent_id_mismatch_count": float(identity_mismatches),
        "active_fraction": active_count / active_total,
        "missed_variance_increment_max_abs_error": max(target_increment_errors, default=0.0),
        "missed_variance_increment_min": min(variance_increment_minima),
        "missed_variance_increment_max": max(variance_increment_maxima),
        "missed_variance_increment_count": float(len(variance_increment_minima)),
        "coobject_variance_increment_max_abs": max(coobject_increments, default=0.0),
        "missed_variance_mask_mismatch_count": float(missed_mask_mismatches),
        "missed_variance_inflation_count": float(inflation_count),
    }


def _gradient_metrics(
    config: OrpheusConfig,
    batch: Mapping[str, Any],
    schedules: Sequence[SceneSchedule],
) -> dict[str, float]:
    """Audit exact per-scene, per-object, per-output RGB/depth VJP support."""

    batch_size = int(batch["rgb"].shape[0])
    if batch_size != 4 or len(schedules) != 4:
        raise ValueError("partial-visibility VJP audit requires exact B4 scenes")
    differentiable = dict(batch)
    differentiable["rgb"] = batch["rgb"][:, :18].clone().requires_grad_(True)
    differentiable["depth"] = batch["depth"][:, :18].clone().requires_grad_(True)
    differentiable["camera"] = {
        name: value[:, :18].clone() if isinstance(value, Tensor) else value
        for name, value in batch["camera"].items()
    }
    differentiable["timestamps"] = batch["timestamps"][:, :18].clone()
    output = _run_public_batch(differentiable, config)
    belief = output["belief"]
    trajectory = output["trajectory"]
    truth0 = batch["objects"]["position"][:, 0, :2]
    birth_mapping, _ = _birth_physical_mapping(output["beliefs"][0].objects.position, truth0)
    coefficients = belief.objects.position.new_tensor(VJP_COEFFICIENTS)
    losses: list[tuple[int, int, str, Tensor]] = []
    for batch_index in range(batch_size):
        for object_index in OBJECT_INDICES:
            losses.extend(
                (
                    (
                        batch_index,
                        object_index,
                        "current_position",
                        (belief.objects.position[batch_index, object_index] * coefficients).mean(),
                    ),
                    (
                        batch_index,
                        object_index,
                        "current_velocity",
                        (belief.objects.velocity[batch_index, object_index] * coefficients).mean(),
                    ),
                )
            )
            for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
                losses.extend(
                    (
                        (
                            batch_index,
                            object_index,
                            f"horizon_{horizon:.2f}_position",
                            (
                                trajectory.positions[batch_index, horizon_index, object_index]
                                * coefficients
                            ).mean(),
                        ),
                        (
                            batch_index,
                            object_index,
                            f"horizon_{horizon:.2f}_velocity",
                            (
                                trajectory.velocities[batch_index, horizon_index, object_index]
                                * coefficients
                            ).mean(),
                        ),
                    )
                )
    visible_masks = batch["labels"]["qualification_visible_mask"][:, :18]
    expected_visible_shape = (4, 18, 2, *batch["rgb"].shape[-2:])
    if (
        not isinstance(visible_masks, Tensor)
        or visible_masks.dtype is not torch.bool
        or tuple(visible_masks.shape) != expected_visible_shape
        or not bool(visible_masks.flatten(3).any(dim=-1).all())
    ):
        raise RuntimeError("VJP audit omitted qualification-only visible-region masks")
    audit_signatures = {
        scene_parameter_signature(MANIFESTS[schedule.split][schedule.index])
        for schedule in schedules
    }
    metrics: dict[str, float] = {
        "gradient_audit_scene_count": 4.0,
        "gradient_audit_unique_scene_fraction": len(audit_signatures) / 4.0,
    }
    for scene_index, schedule in enumerate(schedules):
        metrics[f"gradient_audit_manifest_offset/scene_{scene_index}"] = float(schedule.index)
        metrics[f"gradient_audit_stratum_index/scene_{scene_index}"] = float(schedule.stratum_index)
        metrics[f"gradient_audit_miss_frame/scene_{scene_index}"] = float(
            -1 if schedule.miss_frame is None else schedule.miss_frame
        )
        metrics[f"gradient_audit_missed_physical_slot/scene_{scene_index}"] = float(
            -1 if schedule.missed_slot is None else schedule.missed_slot
        )
        for object_index in OBJECT_INDICES:
            metrics[
                f"gradient_audit_birth_physical_slot/scene_{scene_index}/object_{object_index}"
            ] = float(birth_mapping[scene_index, object_index])
    inputs = (differentiable["rgb"], differentiable["depth"])
    for loss_index, (batch_index, object_index, output_name, loss) in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            inputs,
            retain_graph=loss_index + 1 < len(losses),
            allow_unused=True,
        )
        expected = torch.zeros(18, dtype=torch.bool)
        if output_name == "current_position":
            expected[ANCHOR_FRAME_INDEX] = True
        else:
            expected[list(LIVE_HISTORY_FRAME_INDICES)] = True
            schedule = schedules[batch_index]
            if schedule.miss_frame is not None and schedule.missed_slot is not None:
                target = int(
                    torch.nonzero(birth_mapping[batch_index].eq(schedule.missed_slot))
                    .flatten()
                    .item()
                )
                if target == object_index:
                    expected[schedule.miss_frame] = False
        for modality, source, gradient in zip(("rgb", "depth"), inputs, gradients, strict=True):
            resolved = torch.zeros_like(source) if gradient is None else gradient
            if not bool(torch.isfinite(resolved).all()):
                raise FloatingPointError("partial-visibility VJP is nonfinite")
            per_scene = resolved.abs().flatten(1).sum(dim=-1)
            target_gradient = resolved[batch_index].abs().flatten(1).sum(dim=-1)
            physical_slot = int(birth_mapping[batch_index, object_index])
            visible_region = visible_masks[batch_index, :, physical_slot].to(
                device=resolved.device, dtype=resolved.dtype
            )
            visible_region_gradient = (
                (resolved[batch_index].abs() * visible_region[:, None]).flatten(1).sum(dim=-1)
            )
            cross = torch.cat((per_scene[:batch_index], per_scene[batch_index + 1 :]))
            expected_device = expected.to(target_gradient.device)
            supported = target_gradient >= DEFAULT_GATES.minimum_history_frame_gradient_l1
            suffix = f"scene_{batch_index}/object_{object_index}/{output_name}/{modality}"
            metrics[f"gradient_total_l1/{suffix}"] = float(per_scene[batch_index])
            metrics[f"gradient_expected_min_l1/{suffix}"] = float(
                target_gradient[expected_device].min()
            )
            metrics[f"gradient_visible_region_expected_min_l1/{suffix}"] = float(
                visible_region_gradient[expected_device].min()
            )
            metrics[f"gradient_unexpected_max_l1/{suffix}"] = float(
                target_gradient[~expected_device].max()
            )
            metrics[f"gradient_supported_frames/{suffix}"] = float(supported.sum())
            metrics[f"gradient_expected_frames/{suffix}"] = float(expected.sum())
            metrics[f"gradient_cross_scene_max_l1/{suffix}"] = float(cross.max())
    return metrics


def _select_batch_rows(value: Any, indices: Tensor, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim and value.shape[0] == batch_size:
        return value.index_select(0, indices.to(value.device))
    if isinstance(value, Mapping):
        return {key: _select_batch_rows(item, indices, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        return [value[int(index)] for index in indices.tolist()]
    return value


@torch.no_grad()
def _latency_metrics(
    config: OrpheusConfig,
    batch: Mapping[str, Any],
    *,
    process_rss_start_bytes: int,
) -> dict[str, float]:
    """Take the worse median of separated and partial+miss B1 fixtures."""

    batch_size = int(batch["rgb"].shape[0])
    fixture_indices = (0, 3)
    perception_worst = 0.0
    rollout_worst = 0.0
    runtime_bytes = 0
    for fixture in fixture_indices:
        selected = _select_batch_rows(batch, torch.tensor([fixture], dtype=torch.int64), batch_size)
        model = new_public_model(config)
        model.eval()

        def ingest_history(
            runtime: OnlineWorldModel = model,
            fixture_batch: Mapping[str, Any] = selected,
        ) -> None:
            runtime.reset(batch_size=1)
            for frame in INGEST_FRAME_INDICES:
                runtime.ingest(make_rgbd_packet(fixture_batch, frame))

        ingest_history()
        model.predict(HORIZONS_SECONDS)
        perception: list[float] = []
        for _ in range(3):
            started = time.perf_counter()
            ingest_history()
            perception.append(time.perf_counter() - started)
        rollout: list[float] = []
        for _ in range(20):
            started = time.perf_counter()
            model.predict(HORIZONS_SECONDS)
            rollout.append(time.perf_counter() - started)
        perception_worst = max(perception_worst, median(perception))
        rollout_worst = max(rollout_worst, median(rollout))
        runtime_bytes = max(runtime_bytes, _persistent_runtime_tensor_bytes(model))
    rss_after = _process_max_rss_bytes()
    return {
        "perception_latency_seconds": float(perception_worst),
        "state_only_rollout_latency_seconds": float(rollout_worst),
        "persistent_runtime_tensor_state_bytes_max": float(runtime_bytes),
        "process_max_rss_bytes": float(rss_after),
        "process_rss_delta_bytes": float(max(0, rss_after - process_rss_start_bytes)),
    }


def _accuracy_metrics(
    metrics: dict[str, Any],
    *,
    prefix: str,
    current_position_error: Tensor,
    current_velocity_error: Tensor,
    future_position_error: Tensor,
    future_velocity_error: Tensor,
    stationary_position_error: Tensor,
    zero_velocity_error: Tensor,
    row_mask: Tensor,
    object_mask: Tensor | None = None,
) -> None:
    """Add the same accuracy surface for an overall, stratum, or role slice."""

    if row_mask.dtype is not torch.bool or row_mask.ndim != 1:
        raise ValueError("accuracy row mask must be boolean [episodes]")
    selected = row_mask[:, None].expand(-1, 2).clone()
    if object_mask is not None:
        if object_mask.shape != selected.shape or object_mask.dtype is not torch.bool:
            raise ValueError("accuracy object mask must be boolean [episodes,2]")
        selected = selected & object_mask
    if not bool(selected.any()):
        raise RuntimeError(f"accuracy slice {prefix!r} is empty")
    label = f"{prefix}/" if prefix else ""
    current_position = current_position_error[selected]
    current_velocity = current_velocity_error[selected]
    future_position = future_position_error[selected]
    future_velocity = future_velocity_error[selected]
    stationary = stationary_position_error[selected]
    zero_velocity = zero_velocity_error[selected]
    current_position_rmse = _rmse(current_position)
    current_velocity_rmse = _rmse(current_velocity)
    horizon_position = [_rmse(future_position[:, index]) for index in range(5)]
    horizon_velocity = [_rmse(future_velocity[:, index]) for index in range(5)]
    stationary_rmse = [_rmse(stationary[:, index]) for index in range(5)]
    epsilon = torch.finfo(torch.float64).eps
    metrics[f"{label}current_position_rmse_m"] = current_position_rmse
    metrics[f"{label}current_velocity_rmse_mps"] = current_velocity_rmse
    metrics[f"{label}maximum_position_error_growth_slope_mps"] = max(
        max(0.0, error - current_position_rmse) / horizon
        for error, horizon in zip(horizon_position, HORIZONS_SECONDS, strict=True)
    )
    metrics[f"{label}early_stationary_additive_regression_m"] = max(
        horizon_position[index] - stationary_rmse[index] for index in (0, 1)
    )
    metrics[f"{label}long_stationary_rmse_ratio"] = max(
        horizon_position[index] / max(stationary_rmse[index], epsilon) for index in (2, 3, 4)
    )
    metrics[f"{label}zero_velocity_rmse_ratio"] = current_velocity_rmse / max(
        _rmse(zero_velocity), epsilon
    )
    for axis_index, axis in enumerate(AXIS_NAMES):
        metrics[f"{label}current_position_rmse_m/{axis}"] = _rmse(current_position[:, axis_index])
        metrics[f"{label}current_velocity_rmse_mps/{axis}"] = _rmse(current_velocity[:, axis_index])
    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        horizon_label = f"{horizon:.2f}"
        metrics[f"{label}horizon_{horizon_label}_position_rmse_m"] = horizon_position[horizon_index]
        metrics[f"{label}horizon_{horizon_label}_velocity_rmse_mps"] = horizon_velocity[
            horizon_index
        ]
        for axis_index, axis in enumerate(AXIS_NAMES):
            metrics[f"{label}horizon_{horizon_label}_position_rmse_m/{axis}"] = _rmse(
                future_position[:, horizon_index, axis_index]
            )
            metrics[f"{label}horizon_{horizon_label}_velocity_rmse_mps/{axis}"] = _rmse(
                future_velocity[:, horizon_index, axis_index]
            )


def gate_failures(metrics: Mapping[str, Any], *, split: str = "development") -> list[str]:
    """Independently recompute every declared scalar acceptance comparison."""

    tracked_metrics = _TrackingMetrics(metrics)
    metrics = tracked_metrics
    gates = _TrackingGates(DEFAULT_GATES)
    failures: list[str] = []

    def maximum(key: str, bound: float) -> None:
        value = _numeric(metrics.get(key))
        if value is None:
            failures.append(f"{key}:missing_or_nonfinite")
        elif value > bound:
            failures.append(f"{key}:{value:.9g}>{bound:.9g}")

    def minimum(key: str, bound: float) -> None:
        value = _numeric(metrics.get(key))
        if value is None:
            failures.append(f"{key}:missing_or_nonfinite")
        elif value < bound:
            failures.append(f"{key}:{value:.9g}<{bound:.9g}")

    def equal(key: str, expected: float) -> None:
        value = _numeric(metrics.get(key))
        if value is None:
            failures.append(f"{key}:missing_or_nonfinite")
        elif value != expected:
            failures.append(f"{key}:{value:.9g}!={expected:.9g}")

    accuracy_prefixes = (
        "",
        *(f"stratum/{name}" for name in STRATUM_NAMES),
        *(f"object/{index}" for index in OBJECT_INDICES),
        "role/front",
        "role/rear",
        "role/missed_target",
        "role/coobject",
    )
    for prefix in accuracy_prefixes:
        label = f"{prefix}/" if prefix else ""
        maximum(f"{label}current_position_rmse_m", gates.current_position_rmse_m)
        maximum(f"{label}current_velocity_rmse_mps", gates.current_velocity_rmse_mps)
        maximum(
            f"{label}maximum_position_error_growth_slope_mps",
            gates.maximum_position_error_growth_slope_mps,
        )
        maximum(
            f"{label}early_stationary_additive_regression_m",
            gates.early_stationary_additive_margin_m,
        )
        maximum(f"{label}long_stationary_rmse_ratio", gates.long_stationary_rmse_ratio)
        maximum(f"{label}zero_velocity_rmse_ratio", gates.zero_velocity_rmse_ratio)
        for axis in AXIS_NAMES:
            maximum(
                f"{label}current_position_rmse_m/{axis}",
                gates.per_object_axis_position_rmse_m,
            )
            maximum(
                f"{label}current_velocity_rmse_mps/{axis}",
                gates.per_object_axis_velocity_rmse_mps,
            )
        for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
            horizon_label = f"{horizon:.2f}"
            maximum(
                f"{label}horizon_{horizon_label}_position_rmse_m",
                gates.horizon_position_rmse_m[horizon_index],
            )
            maximum(
                f"{label}horizon_{horizon_label}_velocity_rmse_mps",
                gates.horizon_velocity_rmse_mps,
            )
            for axis in AXIS_NAMES:
                maximum(
                    f"{label}horizon_{horizon_label}_position_rmse_m/{axis}",
                    gates.per_object_axis_position_rmse_m
                    + gates.horizon_position_rmse_m[horizon_index],
                )
                maximum(
                    f"{label}horizon_{horizon_label}_velocity_rmse_mps/{axis}",
                    gates.per_object_axis_velocity_rmse_mps,
                )
    maximum("miss_frame_position_rmse_m", gates.miss_frame_position_rmse_m)
    for key, expected in (
        ("identity_switch_count", gates.identity_switch_count),
        ("persistent_id_mismatch_count", gates.persistent_id_mismatch_count),
        ("association_ambiguous_pair_count", gates.ambiguous_pair_count),
        ("false_miss_association_count", gates.false_miss_association_count),
        ("false_birth_count", gates.false_birth_count),
        ("death_count", gates.death_count),
        ("false_velocity_evidence_count", gates.false_velocity_evidence_count),
        ("direct_position_field_count", gates.direct_position_field_count),
        ("measurement_validity_mismatch_count", 0.0),
        ("association_validity_mismatch_count", 0.0),
        ("public_rollout_output_alias_count", 0.0),
        ("predicted_unobservable_count", 0.0),
        ("missed_variance_mask_mismatch_count", 0.0),
        ("associator_call_frame_mismatch_count", 0.0),
        ("direct_velocity_call_frame_mismatch_count", 0.0),
        ("recovery_free_mode_mismatch_count", 0.0),
        ("missed_step_trace_mismatch_count", gates.missed_step_trace_mismatch_count),
        ("runtime_free_mode_mismatch_count", gates.runtime_free_mode_mismatch_count),
        ("rollout_free_mode_mismatch_count", gates.rollout_free_mode_mismatch_count),
    ):
        equal(key, expected)
    equal("association_pair_coverage", gates.association_coverage)
    minimum("identity_coverage", 1.0)
    minimum("minimum_hungarian_margin", gates.minimum_hungarian_margin)
    minimum("minimum_position_assignment_margin_m", gates.minimum_position_assignment_margin_m)
    minimum("minimum_matched_appearance_cosine", gates.minimum_matched_appearance_cosine)
    minimum(
        "minimum_cross_appearance_cosine_distance",
        gates.minimum_cross_appearance_cosine_distance,
    )
    equal("reacquisition_latency_frames_max", gates.reacquisition_latency_frames)
    maximum("maximum_missed_steps", gates.maximum_missed_steps)
    equal("final_missed_steps_max", gates.final_missed_steps)
    for suffix in ("min", "max"):
        equal(f"missed_target_steps_before_{suffix}", gates.missed_target_steps_before)
        equal(f"missed_target_steps_at_miss_{suffix}", gates.missed_target_steps_at_miss)
        equal(
            f"missed_target_steps_at_recovery_{suffix}",
            gates.missed_target_steps_at_recovery,
        )
        equal(f"missed_coobject_steps_{suffix}", gates.missed_coobject_steps)
        equal(f"recovery_mode_value_{suffix}", gates.free_motion_mode_value)
    minimum("active_fraction", gates.active_fraction)
    minimum("rollout_active_fraction", gates.rollout_active_fraction)
    equal("physical_palette_swap_fraction", gates.physical_palette_swap_fraction)
    equal("birth_slot_physical_zero_fraction", 0.50)
    minimum("unique_scene_specification_fraction", gates.unique_scene_specification_fraction)
    for name in STRATUM_NAMES:
        equal(f"stratum_fraction/{name}", 0.25)
    equal("severity_fraction/separated", 0.50)
    equal("severity_fraction/mild", 0.25)
    equal("severity_fraction/moderate", 0.25)
    if split not in MANIFESTS:
        raise ValueError(f"unknown qualification split {split!r}")
    episode_count = float(len(MANIFESTS[split]))
    equal("manifest_episode_count", episode_count)
    equal("history_sample_count_min", gates.history_sample_count)
    equal("history_sample_count_max", gates.history_sample_count)
    equal("history_no_miss_valid_count_min", gates.no_miss_history_valid_count)
    equal("history_no_miss_valid_count_max", gates.no_miss_history_valid_count)
    equal("history_missed_target_valid_count_min", gates.missed_history_valid_count)
    equal("history_missed_target_valid_count_max", gates.missed_history_valid_count)
    equal("history_missed_coobject_valid_count_min", gates.no_miss_history_valid_count)
    equal("history_missed_coobject_valid_count_max", gates.no_miss_history_valid_count)
    equal("history_no_miss_slot_count", episode_count)
    equal("history_missed_target_slot_count", episode_count / 2.0)
    equal("history_missed_coobject_slot_count", episode_count / 2.0)
    equal("history_sample_mask_mismatch_count", 0.0)
    equal("history_valid_mask_mismatch_count", 0.0)
    equal("history_latest_valid_mismatch_count", 0.0)
    maximum(
        "history_timestamp_max_abs_error_seconds",
        gates.history_span_tolerance_seconds,
    )
    minimum(
        "history_span_seconds_min",
        gates.history_span_seconds - gates.history_span_tolerance_seconds,
    )
    maximum(
        "history_span_seconds_max",
        gates.history_span_seconds + gates.history_span_tolerance_seconds,
    )
    equal("expected_velocity_evidence_coverage", gates.expected_velocity_evidence_coverage)
    for suffix in ("min", "max"):
        equal(
            f"associator_call_count_per_batch_{suffix}",
            gates.associator_calls_per_batch,
        )
        equal(
            f"direct_velocity_call_count_per_batch_{suffix}",
            gates.direct_velocity_calls_per_batch,
        )
    equal("position_owner_count_min", gates.position_owner_count)
    equal("position_owner_count_max", gates.position_owner_count)
    maximum(
        "direct_velocity_position_change_max_abs_m",
        gates.direct_velocity_position_change_max_abs_m,
    )
    maximum(
        "direct_metric_position_owner_max_abs_m",
        gates.direct_metric_position_owner_max_abs_m,
    )
    minimum("minimum_observed_support_fraction", gates.minimum_observed_support_fraction)
    maximum(
        "maximum_surface_fit_residual_relative_rms",
        gates.maximum_surface_residual_relative_rms,
    )
    maximum(
        "maximum_full_silhouette_overlap_fraction",
        gates.maximum_full_silhouette_overlap_fraction,
    )
    maximum(
        "maximum_surface_radius_relative_error",
        gates.maximum_surface_radius_relative_error,
    )
    maximum(
        "maximum_surface_fit_condition_number",
        gates.maximum_surface_fit_condition_number,
    )
    minimum(
        "minimum_fitted_boundary_clearance_pixels",
        gates.minimum_fitted_boundary_clearance_pixels,
    )
    minimum(
        "minimum_full_silhouette_radius_pixels",
        gates.minimum_full_silhouette_radius_pixels,
    )
    maximum(
        "maximum_full_silhouette_gap_abs_pixels",
        gates.maximum_full_silhouette_gap_abs_pixels,
    )
    equal("geometry_nonfinite_count", 0.0)
    equal("invalid_geometry_nonzero_count", 0.0)
    equal("invalid_pair_geometry_nonzero_count", 0.0)
    maximum("maximum_predicted_visibility_error", gates.maximum_predicted_visibility_error)
    preflight_scopes = (
        ("", "overall", None),
        ("severity/separated/", "separated", None),
        ("severity/mild/", "mild", None),
        ("severity/moderate/", "moderate", None),
        ("stratum/separated_no_miss/", "separated", False),
        ("stratum/partial_no_miss/", "partial", False),
        ("stratum/separated_one_miss/", "separated", True),
        ("stratum/partial_one_miss/", "partial", True),
    )
    for prefix, geometry, missed in preflight_scopes:
        minimum(
            f"{prefix}preflight_minimum_boundary_clearance_pixels",
            gates.minimum_boundary_clearance_pixels,
        )
        minimum(
            f"{prefix}preflight_minimum_world_surface_gap_m",
            gates.minimum_world_surface_gap_m,
        )
        minimum(
            f"{prefix}preflight_minimum_world_boundary_clearance_m",
            gates.minimum_world_boundary_clearance_m,
        )
        minimum(
            f"{prefix}preflight_minimum_palette_cosine_distance",
            gates.minimum_cross_appearance_cosine_distance,
        )
        equal(f"{prefix}preflight_event_count", gates.preflight_event_count)
        equal(f"{prefix}miss_unchanged_max_abs", 0.0)
        if geometry == "separated":
            minimum(
                f"{prefix}preflight_minimum_silhouette_gap_pixels",
                gates.minimum_separated_silhouette_gap_pixels,
            )
            maximum(
                f"{prefix}preflight_maximum_silhouette_gap_pixels",
                gates.maximum_full_silhouette_gap_abs_pixels,
            )
            equal(f"{prefix}preflight_minimum_rear_visible_fraction", 1.0)
            equal(f"{prefix}preflight_maximum_rear_visible_fraction", 1.0)
            equal(f"{prefix}preflight_minimum_overlap_pixels", 0.0)
            equal(f"{prefix}preflight_maximum_overlap_pixels", 0.0)
        elif geometry in {"mild", "moderate"}:
            gap_min = (
                gates.mild_silhouette_gap_min_pixels
                if geometry == "mild"
                else gates.moderate_silhouette_gap_min_pixels
            )
            gap_max = (
                gates.mild_silhouette_gap_max_pixels
                if geometry == "mild"
                else gates.moderate_silhouette_gap_max_pixels
            )
            visible_min = (
                gates.mild_rear_visible_fraction_min
                if geometry == "mild"
                else gates.moderate_rear_visible_fraction_min
            )
            visible_max = (
                gates.mild_rear_visible_fraction_max
                if geometry == "mild"
                else gates.moderate_rear_visible_fraction_max
            )
            minimum(f"{prefix}preflight_minimum_silhouette_gap_pixels", gap_min)
            maximum(f"{prefix}preflight_maximum_silhouette_gap_pixels", gap_max)
            minimum(f"{prefix}preflight_minimum_rear_visible_fraction", visible_min)
            maximum(f"{prefix}preflight_maximum_rear_visible_fraction", visible_max)
            minimum(f"{prefix}preflight_minimum_overlap_pixels", 1.0)
            maximum(f"{prefix}preflight_maximum_overlap_pixels", 4096.0)
        else:
            minimum(
                f"{prefix}preflight_minimum_silhouette_gap_pixels",
                gates.moderate_silhouette_gap_min_pixels,
            )
            maximum(
                f"{prefix}preflight_maximum_silhouette_gap_pixels",
                (
                    gates.maximum_full_silhouette_gap_abs_pixels
                    if geometry == "overall"
                    else gates.mild_silhouette_gap_max_pixels
                ),
            )
            minimum(
                f"{prefix}preflight_minimum_rear_visible_fraction",
                gates.moderate_rear_visible_fraction_min,
            )
            maximum(f"{prefix}preflight_maximum_rear_visible_fraction", 1.0)
            minimum(f"{prefix}preflight_minimum_overlap_pixels", 0.0)
            maximum(f"{prefix}preflight_maximum_overlap_pixels", 4096.0)
        if missed is True:
            minimum(f"{prefix}miss_changed_pixels", 4.0)
        else:
            equal(f"{prefix}miss_changed_pixels", 0.0)
    maximum(
        "missed_variance_increment_max_abs_error",
        gates.missed_variance_increment_tolerance,
    )
    minimum(
        "missed_variance_increment_min",
        gates.missed_fast_variance_increment - gates.missed_variance_increment_tolerance,
    )
    maximum(
        "missed_variance_increment_max",
        gates.missed_fast_variance_increment + gates.missed_variance_increment_tolerance,
    )
    maximum(
        "coobject_variance_increment_max_abs",
        gates.coobject_variance_increment_max_abs,
    )
    expected_misses = episode_count / 2.0
    equal("missed_variance_increment_count", expected_misses)
    equal(
        "missed_variance_inflation_count",
        expected_misses * gates.missed_variance_inflation_count,
    )
    maximum("semigroup_position_max_abs_m", gates.semigroup_position_max_abs_m)
    maximum("semigroup_velocity_max_abs_mps", gates.semigroup_velocity_max_abs_mps)
    maximum("public_direct_position_max_abs_m", gates.public_direct_position_max_abs_m)
    maximum("public_direct_velocity_max_abs_mps", gates.public_direct_velocity_max_abs_mps)
    maximum(
        "analytic_position_agreement_max_abs_m",
        gates.analytic_position_agreement_max_abs_m,
    )
    maximum(
        "analytic_velocity_agreement_max_abs_mps",
        gates.analytic_velocity_agreement_max_abs_mps,
    )
    maximum("public_query_time_max_abs_seconds", gates.public_query_time_max_abs_seconds)
    equal("ingested_frame_count_min", gates.ingest_calls_per_batch)
    equal("ingested_frame_count_max", gates.ingest_calls_per_batch)
    equal("public_predict_calls_per_batch_min", gates.public_predict_calls_per_batch)
    equal("public_predict_calls_per_batch_max", gates.public_predict_calls_per_batch)
    equal("gradient_audit_scene_count", gates.gradient_audit_scene_count)
    equal("gradient_audit_unique_scene_fraction", 1.0)
    for scene_index, audit_offset in enumerate((0, 1, 6, 15)):
        schedule = scene_schedule(DEVELOPMENT_SEEDS[audit_offset])
        equal(f"gradient_audit_manifest_offset/scene_{scene_index}", float(audit_offset))
        equal(
            f"gradient_audit_stratum_index/scene_{scene_index}",
            float(schedule.stratum_index),
        )
        equal(
            f"gradient_audit_miss_frame/scene_{scene_index}",
            float(-1 if schedule.miss_frame is None else schedule.miss_frame),
        )
        equal(
            f"gradient_audit_missed_physical_slot/scene_{scene_index}",
            float(-1 if schedule.missed_slot is None else schedule.missed_slot),
        )
        birth_mapping: list[int] = []
        for object_index in OBJECT_INDICES:
            key = f"gradient_audit_birth_physical_slot/scene_{scene_index}/object_{object_index}"
            value = _numeric(metrics.get(key))
            if value not in {0.0, 1.0}:
                failures.append(f"{key}:not_a_physical_slot")
                birth_mapping.append(-1)
            else:
                birth_mapping.append(int(value))
        if set(birth_mapping) != {0, 1}:
            failures.append(f"gradient_audit_birth_mapping/scene_{scene_index}:not_a_permutation")
        for object_index in OBJECT_INDICES:
            for output_name in VJP_OUTPUTS:
                expected_count = gates.current_position_required_frames
                if output_name != "current_position":
                    expected_count = gates.no_miss_history_valid_count
                    if (
                        schedule.miss_frame is not None
                        and schedule.missed_slot is not None
                        and birth_mapping[object_index] == schedule.missed_slot
                    ):
                        expected_count = gates.missed_history_valid_count
                for modality in ("rgb", "depth"):
                    suffix = f"scene_{scene_index}/object_{object_index}/{output_name}/{modality}"
                    minimum(f"gradient_total_l1/{suffix}", gates.minimum_input_gradient_l1)
                    maximum(f"gradient_total_l1/{suffix}", gates.maximum_input_gradient_l1)
                    minimum(
                        f"gradient_expected_min_l1/{suffix}",
                        gates.minimum_history_frame_gradient_l1,
                    )
                    minimum(
                        f"gradient_visible_region_expected_min_l1/{suffix}",
                        gates.minimum_visible_region_gradient_l1,
                    )
                    maximum(f"gradient_unexpected_max_l1/{suffix}", gates.maximum_zero_gradient_l1)
                    maximum(
                        f"gradient_cross_scene_max_l1/{suffix}",
                        gates.maximum_cross_scene_gradient_l1,
                    )
                    equal(f"gradient_expected_frames/{suffix}", expected_count)
                    equal(f"gradient_supported_frames/{suffix}", expected_count)
    maximum("perception_latency_seconds", gates.perception_latency_seconds)
    maximum("state_only_rollout_latency_seconds", gates.state_only_rollout_latency_seconds)
    maximum(
        "persistent_runtime_tensor_state_bytes_max",
        float(gates.persistent_runtime_tensor_state_bytes),
    )
    maximum("process_max_rss_bytes", float(gates.process_max_rss_bytes))
    maximum("process_rss_delta_bytes", float(gates.process_rss_delta_bytes))
    for key in (
        "learned_parameter_count",
        "learned_parameter_bytes",
        "module_tensor_buffer_count",
        "persistent_module_state_key_count",
        "persistent_module_state_bytes",
        "optimizer_updates",
        "optimizer_state_entry_count",
        "scheduler_state_entry_count",
        "rng_state_entry_count",
    ):
        equal(key, 0.0)
    unexpected = sorted(set(tracked_metrics.values) - tracked_metrics.accessed)
    if unexpected:
        failures.extend(f"metric_schema:unexpected:{key}" for key in unexpected)
    unused_gates = sorted(set(asdict(DEFAULT_GATES)) - gates.accessed)
    if unused_gates:
        failures.extend(f"gate_schema:unused:{key}" for key in unused_gates)
    return failures


def _chunks(values: Sequence[int], size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _validate_manifest(split: str, seeds: Sequence[int]) -> tuple[int, ...]:
    assert_attempt2_admissibility()
    requested = _exact_seed_tuple(seeds, label=f"{split} manifest")
    if split not in MANIFESTS or requested != MANIFESTS[split]:
        raise ValueError(f"{split!r} must use its exact frozen partial-visibility manifest")
    if canonical_sha256(list(requested)) != MANIFEST_SHA256[split]:
        raise ValueError("partial-visibility manifest hash differs from protocol")
    return requested


def _validate_manifest_result(result: Mapping[str, Any], *, split: str) -> None:
    expected_keys = {
        "split",
        "seeds",
        "seed_manifest_sha256",
        "scene_parameter_signature_sha256",
        "metrics",
        "failures",
        "passed",
        "optimizer_updates",
        "runtime_api",
        "scene_constructor",
    }
    if type(result) is not dict or set(result) != expected_keys:
        raise ValueError("manifest result must have the exact typed evidence schema")
    if type(result.get("split")) is not str or result.get("split") != split:
        raise ValueError("manifest result split differs from authorization")
    seeds = result.get("seeds")
    if type(seeds) is not list:
        raise ValueError("manifest result seeds must be an exact JSON list")
    requested = _validate_manifest(split, seeds)
    if result.get("seed_manifest_sha256") != MANIFEST_SHA256[split]:
        raise ValueError("manifest result hash differs from the frozen manifest")
    if result.get("scene_parameter_signature_sha256") != SCENE_PARAMETER_SIGNATURE_SHA256[split]:
        raise ValueError("manifest result pure scene-parameter signature hash differs")
    if type(result.get("optimizer_updates")) is not int or result.get("optimizer_updates") != 0:
        raise ValueError("manifest result must prove exact zero optimizer updates")
    expected_api = {
        "packet_factory": "make_rgbd_packet",
        "ingest_frames": list(INGEST_FRAME_INDICES),
        "rollout_method": "OnlineWorldModel.predict",
        "horizons_seconds": list(HORIZONS_SECONDS),
    }
    if not _typed_canonical_equal(result.get("runtime_api"), expected_api):
        raise ValueError("manifest result runtime API differs")
    if result.get("scene_constructor") != (
        "construct_partial_visibility_episode_with_full_frame_preflight_and_local_depth_miss"
    ):
        raise ValueError("manifest result scene constructor differs")
    metrics = result.get("metrics")
    if type(metrics) is not dict:
        raise ValueError("manifest result metrics must be an exact JSON object")
    recomputed = gate_failures(metrics, split=split)
    failures = result.get("failures")
    if type(failures) is not list or any(type(item) is not str for item in failures):
        raise ValueError("manifest result failures must be an exact string list")
    if failures != recomputed:
        raise ValueError("manifest result failures differ from independent recomputation")
    if type(result.get("passed")) is not bool or result.get("passed") is not (not recomputed):
        raise ValueError("manifest result pass flag differs from independent recomputation")
    if tuple(seeds) != requested:
        raise AssertionError("validated manifest unexpectedly changed")


def _scene_signature(episode: Mapping[str, Any]) -> str:
    objects = episode["objects"]
    return canonical_sha256(
        {
            name: objects[name][0, :2].detach().cpu().tolist()
            for name in ("position", "velocity", "albedo")
        }
    )


def _evaluate_seed_manifest(
    config: OrpheusConfig,
    seeds: Sequence[int],
    *,
    split: str,
    authorization: _ManifestAccessAuthorization | None = None,
) -> dict[str, Any]:
    """Evaluate one exact authorized manifest with no optimizer or hidden oracle."""

    _require_attempt1_rejection()
    authorization = _require_ledger_minted_authorization(authorization)
    assert_rgbd_partial_visibility_config(config)
    _assert_execution_environment()
    requested = _validate_manifest(split, seeds)
    authorization.begin_manifest(split, requested)
    process_rss_start_bytes = _process_max_rss_bytes()
    accumulated: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "current_position_error",
            "current_velocity_error",
            "future_position_error",
            "future_velocity_error",
            "stationary_position_error",
            "zero_velocity_error",
            "stratum_index",
            "birth_mapping",
            "rear_mask",
            "front_mask",
            "missed_mask",
            "coobject_mask",
            "history_sample_count",
            "history_valid_count",
            "history_span",
            "history_sample_mask_mismatch",
            "history_valid_mask_mismatch",
            "history_latest_valid_mismatch",
            "history_timestamp_error",
            "miss_frame_position_error",
            "semigroup_position",
            "semigroup_velocity",
            "public_direct_position",
            "public_direct_velocity",
            "analytic_position_agreement",
            "analytic_velocity_agreement",
            "query_time_error",
            "direct_metric_position_owner_error",
        )
    }
    preflight_values: dict[str, list[float]] = {}
    scene_signatures: set[str] = set()
    audit_episodes: dict[int, Mapping[str, Any]] = {}
    audit_schedules: dict[int, SceneSchedule] = {}
    association_batches: list[dict[str, float]] = []
    recovery_batches: list[dict[str, float]] = []
    support_values: list[Tensor] = []
    residual_values: list[Tensor] = []
    overlap_values: list[Tensor] = []
    radius_error_values: list[Tensor] = []
    condition_values: list[Tensor] = []
    fitted_boundary_values: list[Tensor] = []
    silhouette_radius_values: list[Tensor] = []
    silhouette_gap_values: list[Tensor] = []
    geometry_nonfinite_count = 0
    invalid_geometry_nonzero_count = 0
    invalid_pair_geometry_nonzero_count = 0
    predicted_visibility_errors: list[Tensor] = []
    predicted_unobservable_count = 0
    identity_switch_count = 0
    false_birth_count = 0
    death_count = 0
    identity_correct = 0
    identity_total = 0
    birth_slot_zero = 0
    birth_mapping_count = 0
    rollout_active_count = 0
    rollout_active_total = 0
    position_owner_counts: list[int] = []
    direct_position_fields = 0
    direct_velocity_position_change_max = 0.0
    packet_counts: list[int] = []
    predict_counts: list[int] = []
    alias_count = 0
    runtime_state_bytes = 0
    all_schedules: list[SceneSchedule] = []

    for chunk_start, seed_chunk in enumerate(_chunks(requested, config.training.batch_size)):
        episodes = [
            _construct_partial_visibility_episode(config, seed, authorization=authorization)
            for seed in seed_chunk
        ]
        schedules = [scene_schedule(seed) for seed in seed_chunk]
        all_schedules.extend(schedules)
        for local_index, (episode, schedule) in enumerate(zip(episodes, schedules, strict=True)):
            manifest_index = chunk_start * config.training.batch_size + local_index
            scene_signatures.add(_scene_signature(episode))
            evidence = preflight_partial_visibility_episode(
                episode, config=config, schedule=schedule
            )
            for scope in ("", f"stratum/{schedule.stratum}/", f"severity/{schedule.severity}/"):
                for name, value in evidence.items():
                    preflight_values.setdefault(f"{scope}{name}", []).append(value)
            if manifest_index in {0, 1, 6, 15}:
                audit_episodes[manifest_index] = episode
                audit_schedules[manifest_index] = schedule
        batch = collate_episodes(episodes)
        with torch.no_grad():
            output = _run_public_batch(batch, config)
        beliefs = output["beliefs"]
        belief = output["belief"]
        trajectory = output["trajectory"]
        birth_mapping, _ = _birth_physical_mapping(
            beliefs[0].objects.position, batch["objects"]["position"][:, 0, :2]
        )
        association_batches.append(_association_metrics(output, schedules, birth_mapping))
        recovery_batches.append(_recovery_metrics(output, schedules, birth_mapping))
        birth_slot_zero += int(birth_mapping[:, 0].eq(0).sum())
        birth_mapping_count += birth_mapping.shape[0]
        for frame, posterior in enumerate(beliefs):
            frame_mapping, _ = _birth_physical_mapping(
                posterior.objects.position,
                batch["objects"]["position"][:, frame, :2],
            )
            identity_switch_count += int(frame_mapping.ne(birth_mapping).any(dim=-1).sum())
            identity_correct += int(frame_mapping.eq(birth_mapping).all(dim=-1).sum()) * 2
            identity_total += frame_mapping.shape[0] * 2
            if frame:
                previous = beliefs[frame - 1].objects.active
                false_birth_count += int((~previous & posterior.objects.active).sum())
                death_count += int((previous & ~posterior.objects.active).sum())
        anchor_position = _gather_physical_by_slot(
            batch["objects"]["position"][:, ANCHOR_FRAME_INDEX, :2], birth_mapping
        )
        anchor_velocity = _gather_physical_by_slot(
            batch["objects"]["velocity"][:, ANCHOR_FRAME_INDEX, :2], birth_mapping
        )
        targets = torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64)
        future_position = _gather_physical_by_slot(
            batch["objects"]["position"][:, :, :2].index_select(1, targets), birth_mapping
        )
        future_velocity = _gather_physical_by_slot(
            batch["objects"]["velocity"][:, :, :2].index_select(1, targets), birth_mapping
        )
        accumulated["current_position_error"].append(
            (belief.objects.position - anchor_position).cpu()
        )
        accumulated["current_velocity_error"].append(
            (belief.objects.velocity - anchor_velocity).cpu()
        )
        accumulated["future_position_error"].append(
            (trajectory.positions - future_position).permute(0, 2, 1, 3).cpu()
        )
        accumulated["future_velocity_error"].append(
            (trajectory.velocities - future_velocity).permute(0, 2, 1, 3).cpu()
        )
        accumulated["stationary_position_error"].append(
            (belief.objects.position[:, None] - future_position).permute(0, 2, 1, 3).cpu()
        )
        accumulated["zero_velocity_error"].append((-anchor_velocity).cpu())
        accumulated["stratum_index"].append(
            torch.tensor([schedule.stratum_index for schedule in schedules], dtype=torch.int64)
        )
        accumulated["birth_mapping"].append(birth_mapping.cpu())
        rear_mask = torch.zeros((len(schedules), 2), dtype=torch.bool)
        missed_mask = torch.zeros_like(rear_mask)
        miss_errors: list[Tensor] = []
        for batch_index, schedule in enumerate(schedules):
            if schedule.rear_slot is not None:
                rear_mask[batch_index] = birth_mapping[batch_index].cpu().eq(schedule.rear_slot)
            if schedule.missed_slot is not None and schedule.miss_frame is not None:
                persistent_target = int(
                    torch.nonzero(birth_mapping[batch_index].eq(schedule.missed_slot))
                    .flatten()
                    .item()
                )
                missed_mask[batch_index, persistent_target] = True
                truth = batch["objects"]["position"][
                    batch_index, schedule.miss_frame, schedule.missed_slot
                ]
                miss_errors.append(
                    (
                        beliefs[schedule.miss_frame].objects.position[
                            batch_index, persistent_target
                        ]
                        - truth
                    )
                    .detach()
                    .cpu()
                )
        accumulated["rear_mask"].append(rear_mask)
        accumulated["front_mask"].append((~rear_mask) & rear_mask.any(dim=-1, keepdim=True))
        accumulated["missed_mask"].append(missed_mask)
        accumulated["coobject_mask"].append((~missed_mask) & missed_mask.any(dim=-1, keepdim=True))
        if miss_errors:
            accumulated["miss_frame_position_error"].append(torch.stack(miss_errors))
        history = output["history"]
        sample_count = history.sample_mask.sum(dim=-1)
        valid_count = history.valid_mask.sum(dim=-1)
        expected_sample_mask = torch.ones_like(history.sample_mask)
        expected_valid_mask = torch.ones_like(history.valid_mask)
        for batch_index, schedule in enumerate(schedules):
            if schedule.missed_slot is not None and schedule.miss_frame is not None:
                persistent_target = int(
                    torch.nonzero(birth_mapping[batch_index].eq(schedule.missed_slot))
                    .flatten()
                    .item()
                )
                expected_valid_mask[
                    batch_index,
                    persistent_target,
                    schedule.miss_frame - LIVE_HISTORY_FRAME_INDICES[0],
                ] = False
        expected_timestamps = history.timestamps.new_tensor(LIVE_HISTORY_FRAME_INDICES) / 20.0
        expected_timestamps = expected_timestamps.expand_as(history.timestamps)
        span = history.timestamps[..., -1] - history.timestamps[..., 0]
        accumulated["history_sample_count"].append(sample_count.cpu())
        accumulated["history_valid_count"].append(valid_count.cpu())
        accumulated["history_span"].append(span.cpu())
        accumulated["history_sample_mask_mismatch"].append(
            history.sample_mask.ne(expected_sample_mask).sum().reshape(1).cpu()
        )
        accumulated["history_valid_mask_mismatch"].append(
            history.valid_mask.ne(expected_valid_mask).sum().reshape(1).cpu()
        )
        accumulated["history_latest_valid_mismatch"].append(
            (~history.valid_mask[..., -1]).sum().reshape(1).cpu()
        )
        accumulated["history_timestamp_error"].append(
            (history.timestamps - expected_timestamps).abs().max().reshape(1).cpu()
        )
        for name in (
            "semigroup_position",
            "semigroup_velocity",
            "public_direct_position",
            "public_direct_velocity",
            "analytic_position_agreement",
            "analytic_velocity_agreement",
            "query_time_error",
            "direct_metric_position_owner_error",
        ):
            accumulated[name].append(output[name].cpu())
        for measurements in output["measurements"]:
            mask = measurements.measurement_mask
            support = measurements.auxiliary.get("observed_support_fraction")
            residual = measurements.auxiliary.get("surface_fit_residual_relative_rms")
            overlap = measurements.auxiliary.get("full_silhouette_overlap_fraction")
            radius_error = measurements.auxiliary.get("surface_fit_radius_relative_error")
            condition = measurements.auxiliary.get("surface_fit_condition_number")
            fitted_boundary = measurements.auxiliary.get("full_boundary_clearance_pixels")
            silhouette_radius = measurements.auxiliary.get("full_silhouette_radius_pixels")
            silhouette_gap = measurements.auxiliary.get("full_silhouette_gap_pixels")
            slot_values = (
                support,
                residual,
                radius_error,
                condition,
                silhouette_radius,
                fitted_boundary,
            )
            pair_values = (silhouette_gap, overlap)
            if not all(isinstance(value, Tensor) for value in (*slot_values, *pair_values)):
                raise RuntimeError("partial measurement omitted fitted-geometry diagnostics")
            if any(value.shape != mask.shape for value in slot_values) or any(
                value.shape != mask.shape[:-1] for value in pair_values
            ):
                raise RuntimeError("partial measurement fitted-geometry shapes differ")
            geometry_nonfinite_count += sum(
                int((~torch.isfinite(value)).sum()) for value in (*slot_values, *pair_values)
            )
            invalid_geometry_nonzero_count += sum(
                int(value[~mask].ne(0).sum()) for value in slot_values
            )
            pair_mask = mask.all(dim=-1)
            invalid_pair_geometry_nonzero_count += sum(
                int(value[~pair_mask].ne(0).sum()) for value in pair_values
            )
            support_values.append(support[mask].detach().cpu())
            residual_values.append(residual[mask].detach().cpu())
            overlap_values.append(overlap[pair_mask].detach().cpu())
            radius_error_values.append(radius_error[mask].detach().cpu())
            condition_values.append(condition[mask].detach().cpu())
            fitted_boundary_values.append(fitted_boundary[mask].detach().cpu())
            silhouette_radius_values.append(silhouette_radius[mask].detach().cpu())
            silhouette_gap_values.append(silhouette_gap[pair_mask].detach().cpu())
        for frame in range(1, 18):
            predicted = output["predictions"][frame]
            predicted_visible = predicted.auxiliary.get("visible_fraction")
            unobservable = predicted.auxiliary.get("unobservable_mask")
            if not isinstance(predicted_visible, Tensor) or not isinstance(unobservable, Tensor):
                raise RuntimeError("RGB-D projection omitted visibility diagnostics")
            truth_visible = _gather_physical_by_slot(
                batch["objects"]["visible_fraction"][:, frame, :2].unsqueeze(-1),
                birth_mapping,
            ).squeeze(-1)
            predicted_visibility_errors.append(
                (predicted_visible - truth_visible).abs().detach().cpu()
            )
            predicted_unobservable_count += int(unobservable.sum())
        rollout_active_count += int(trajectory.active_mask.sum())
        rollout_active_total += trajectory.active_mask.numel()
        position_owner_counts.append(int(output["position_owner_count"]))
        direct_position_fields += int(output["direct_position_field_count"])
        direct_velocity_position_change_max = max(
            direct_velocity_position_change_max,
            float(output["direct_velocity_position_change_max_abs"]),
        )
        packet_counts.append(int(output["ingest_call_count"]))
        predict_counts.append(int(output["public_predict_call_count"]))
        alias_count += int(output["output_alias_count"])
        runtime_state_bytes = max(runtime_state_bytes, int(output["runtime_tensor_bytes"]))
    authorization.finish_manifest()

    if set(audit_episodes) != {0, 1, 6, 15}:
        raise RuntimeError("manifest omitted a predeclared B4 gradient scene")
    audit_order = (0, 1, 6, 15)
    audit_batch = collate_episodes([audit_episodes[index] for index in audit_order])
    audit_schedule_values = [audit_schedules[index] for index in audit_order]
    tensors = {
        name: torch.cat(values) if values else torch.empty(0)
        for name, values in accumulated.items()
    }
    episode_count = len(requested)
    row_all = torch.ones(episode_count, dtype=torch.bool)
    metrics: dict[str, Any] = {}
    accuracy_kwargs = {
        "current_position_error": tensors["current_position_error"],
        "current_velocity_error": tensors["current_velocity_error"],
        "future_position_error": tensors["future_position_error"],
        "future_velocity_error": tensors["future_velocity_error"],
        "stationary_position_error": tensors["stationary_position_error"],
        "zero_velocity_error": tensors["zero_velocity_error"],
    }
    _accuracy_metrics(metrics, prefix="", row_mask=row_all, **accuracy_kwargs)
    for stratum_index, name in enumerate(STRATUM_NAMES):
        rows = tensors["stratum_index"].eq(stratum_index)
        _accuracy_metrics(metrics, prefix=f"stratum/{name}", row_mask=rows, **accuracy_kwargs)
        metrics[f"stratum_fraction/{name}"] = float(rows.to(torch.float32).mean())
    for object_index in OBJECT_INDICES:
        object_mask = torch.zeros_like(tensors["front_mask"])
        object_mask[:, object_index] = True
        _accuracy_metrics(
            metrics,
            prefix=f"object/{object_index}",
            row_mask=row_all,
            object_mask=object_mask,
            **accuracy_kwargs,
        )
    partial_rows = tensors["rear_mask"].any(dim=-1)
    miss_rows = tensors["missed_mask"].any(dim=-1)
    for role, rows, object_mask in (
        ("front", partial_rows, tensors["front_mask"]),
        ("rear", partial_rows, tensors["rear_mask"]),
        ("missed_target", miss_rows, tensors["missed_mask"]),
        ("coobject", miss_rows, tensors["coobject_mask"]),
    ):
        _accuracy_metrics(
            metrics,
            prefix=f"role/{role}",
            row_mask=rows,
            object_mask=object_mask,
            **accuracy_kwargs,
        )
    metrics["miss_frame_position_rmse_m"] = _rmse(tensors["miss_frame_position_error"])
    association_matched = sum(item["association_matched_count"] for item in association_batches)
    association_expected = sum(item["association_expected_count"] for item in association_batches)
    metrics.update(
        {
            "association_pair_coverage": association_matched / association_expected,
            "association_ambiguous_pair_count": sum(
                item["association_ambiguous_pair_count"] for item in association_batches
            ),
            "false_miss_association_count": sum(
                item["false_miss_association_count"] for item in association_batches
            ),
            "minimum_hungarian_margin": min(
                item["minimum_hungarian_margin"] for item in association_batches
            ),
            "minimum_position_assignment_margin_m": min(
                item["minimum_position_assignment_margin_m"] for item in association_batches
            ),
            "minimum_matched_appearance_cosine": min(
                item["minimum_matched_appearance_cosine"] for item in association_batches
            ),
            "minimum_cross_appearance_cosine_distance": min(
                item["minimum_cross_appearance_cosine_distance"] for item in association_batches
            ),
        }
    )
    recovery_sum_keys = (
        "measurement_validity_mismatch_count",
        "association_validity_mismatch_count",
        "false_velocity_evidence_count",
        "persistent_id_mismatch_count",
        "missed_variance_inflation_count",
        "missed_variance_increment_count",
        "missed_variance_mask_mismatch_count",
        "associator_call_frame_mismatch_count",
        "direct_velocity_call_frame_mismatch_count",
        "recovery_free_mode_mismatch_count",
        "missed_step_trace_mismatch_count",
        "runtime_free_mode_mismatch_count",
        "rollout_free_mode_mismatch_count",
    )
    for key in recovery_sum_keys:
        metrics[key] = sum(item[key] for item in recovery_batches)
    for key in (
        "reacquisition_latency_frames_max",
        "maximum_missed_steps",
        "final_missed_steps_max",
        "missed_variance_increment_max_abs_error",
        "coobject_variance_increment_max_abs",
        "missed_variance_increment_max",
        "missed_target_steps_before_max",
        "missed_target_steps_at_miss_max",
        "missed_target_steps_at_recovery_max",
        "missed_coobject_steps_max",
        "recovery_mode_value_max",
    ):
        metrics[key] = max(item[key] for item in recovery_batches)
    for key in (
        "missed_variance_increment_min",
        "missed_target_steps_before_min",
        "missed_target_steps_at_miss_min",
        "missed_target_steps_at_recovery_min",
        "missed_coobject_steps_min",
        "recovery_mode_value_min",
    ):
        metrics[key] = min(item[key] for item in recovery_batches)
    for key in ("associator_call_count", "direct_velocity_call_count"):
        metrics[f"{key}_per_batch_min"] = min(item[key] for item in recovery_batches)
        metrics[f"{key}_per_batch_max"] = max(item[key] for item in recovery_batches)
    expected_velocity_slots = sum(
        int(
            _expected_slot_masks(
                [scene_schedule(seed) for seed in chunk],
                tensors["birth_mapping"][start : start + len(chunk)],
            )[1].sum()
        )
        for start, chunk in (
            (start, tuple(requested[start : start + config.training.batch_size]))
            for start in range(0, len(requested), config.training.batch_size)
        )
    )
    # Batches are all B4 and have the same schedule cardinality; use their
    # exact expected-weighted coverage rather than averaging percentages.
    metrics["expected_velocity_evidence_coverage"] = (
        sum(
            item["expected_velocity_evidence_coverage"]
            * int(
                _expected_slot_masks(
                    [scene_schedule(seed) for seed in requested[index : index + 4]],
                    tensors["birth_mapping"][index : index + 4],
                )[1].sum()
            )
            for item, index in zip(recovery_batches, range(0, len(requested), 4), strict=True)
        )
        / expected_velocity_slots
    )
    no_miss_rows = ~tensors["missed_mask"].any(dim=-1)
    no_miss_history = tensors["history_valid_count"][no_miss_rows]
    missed_target_history = tensors["history_valid_count"][tensors["missed_mask"]]
    missed_coobject_history = tensors["history_valid_count"][tensors["coobject_mask"]]
    if not all(
        value.numel() for value in (no_miss_history, missed_target_history, missed_coobject_history)
    ):
        raise RuntimeError("history role audit lost a predeclared support slice")
    metrics.update(
        {
            "manifest_episode_count": float(episode_count),
            "identity_switch_count": float(identity_switch_count),
            "identity_coverage": identity_correct / identity_total,
            "false_birth_count": float(false_birth_count),
            "death_count": float(death_count),
            "active_fraction": sum(item["active_fraction"] for item in recovery_batches)
            / len(recovery_batches),
            "rollout_active_fraction": rollout_active_count / rollout_active_total,
            "physical_palette_swap_fraction": sum(
                schedule.palette_swapped for schedule in all_schedules
            )
            / len(all_schedules),
            "severity_fraction/separated": sum(
                schedule.severity == "separated" for schedule in all_schedules
            )
            / len(all_schedules),
            "severity_fraction/mild": sum(schedule.severity == "mild" for schedule in all_schedules)
            / len(all_schedules),
            "severity_fraction/moderate": sum(
                schedule.severity == "moderate" for schedule in all_schedules
            )
            / len(all_schedules),
            "birth_slot_physical_zero_fraction": birth_slot_zero / birth_mapping_count,
            "unique_scene_specification_fraction": len(scene_signatures) / len(requested),
            "history_sample_count_min": float(tensors["history_sample_count"].min()),
            "history_sample_count_max": float(tensors["history_sample_count"].max()),
            "history_no_miss_valid_count_min": float(no_miss_history.min()),
            "history_no_miss_valid_count_max": float(no_miss_history.max()),
            "history_no_miss_slot_count": float(no_miss_history.numel()),
            "history_missed_target_valid_count_min": float(missed_target_history.min()),
            "history_missed_target_valid_count_max": float(missed_target_history.max()),
            "history_missed_target_slot_count": float(missed_target_history.numel()),
            "history_missed_coobject_valid_count_min": float(missed_coobject_history.min()),
            "history_missed_coobject_valid_count_max": float(missed_coobject_history.max()),
            "history_missed_coobject_slot_count": float(missed_coobject_history.numel()),
            "history_sample_mask_mismatch_count": float(
                tensors["history_sample_mask_mismatch"].sum()
            ),
            "history_valid_mask_mismatch_count": float(
                tensors["history_valid_mask_mismatch"].sum()
            ),
            "history_latest_valid_mismatch_count": float(
                tensors["history_latest_valid_mismatch"].sum()
            ),
            "history_timestamp_max_abs_error_seconds": float(
                tensors["history_timestamp_error"].max()
            ),
            "history_span_seconds_min": float(tensors["history_span"].min()),
            "history_span_seconds_max": float(tensors["history_span"].max()),
            "position_owner_count_min": float(min(position_owner_counts)),
            "position_owner_count_max": float(max(position_owner_counts)),
            "direct_position_field_count": float(direct_position_fields),
            "direct_velocity_position_change_max_abs_m": direct_velocity_position_change_max,
            "direct_metric_position_owner_max_abs_m": float(
                tensors["direct_metric_position_owner_error"].max()
            ),
            "minimum_observed_support_fraction": float(torch.cat(support_values).min()),
            "maximum_surface_fit_residual_relative_rms": float(torch.cat(residual_values).max()),
            "maximum_full_silhouette_overlap_fraction": float(torch.cat(overlap_values).max()),
            "maximum_surface_radius_relative_error": float(torch.cat(radius_error_values).max()),
            "maximum_surface_fit_condition_number": float(torch.cat(condition_values).max()),
            "minimum_fitted_boundary_clearance_pixels": float(
                torch.cat(fitted_boundary_values).min()
            ),
            "minimum_full_silhouette_radius_pixels": float(
                torch.cat(silhouette_radius_values).min()
            ),
            "maximum_full_silhouette_gap_abs_pixels": float(
                torch.cat(silhouette_gap_values).abs().max()
            ),
            "geometry_nonfinite_count": float(geometry_nonfinite_count),
            "invalid_geometry_nonzero_count": float(invalid_geometry_nonzero_count),
            "invalid_pair_geometry_nonzero_count": float(invalid_pair_geometry_nonzero_count),
            "maximum_predicted_visibility_error": float(
                torch.cat(predicted_visibility_errors).max()
            ),
            "predicted_unobservable_count": float(predicted_unobservable_count),
            "semigroup_position_max_abs_m": float(tensors["semigroup_position"].max()),
            "semigroup_velocity_max_abs_mps": float(tensors["semigroup_velocity"].max()),
            "public_direct_position_max_abs_m": float(tensors["public_direct_position"].max()),
            "public_direct_velocity_max_abs_mps": float(tensors["public_direct_velocity"].max()),
            "analytic_position_agreement_max_abs_m": float(
                tensors["analytic_position_agreement"].max()
            ),
            "analytic_velocity_agreement_max_abs_mps": float(
                tensors["analytic_velocity_agreement"].max()
            ),
            "public_rollout_output_alias_count": float(alias_count),
            "public_query_time_max_abs_seconds": float(tensors["query_time_error"].max()),
            "ingested_frame_count_min": float(min(packet_counts)),
            "ingested_frame_count_max": float(max(packet_counts)),
            "public_predict_calls_per_batch_min": float(min(predict_counts)),
            "public_predict_calls_per_batch_max": float(max(predict_counts)),
            "persistent_runtime_tensor_state_bytes_max": float(runtime_state_bytes),
        }
    )
    for name, values in preflight_values.items():
        if (
            "maximum" in name
            or name.endswith("unchanged_max_abs")
            or name.endswith("preflight_event_count")
        ):
            metrics[name] = max(values)
        else:
            metrics[name] = min(values)
    model = new_public_model(config)
    learned = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    buffers = tuple(model.buffers())
    state = tuple(model.state_dict().values())
    metrics.update(
        {
            "learned_parameter_count": float(sum(value.numel() for value in learned)),
            "learned_parameter_bytes": float(
                sum(value.numel() * value.element_size() for value in learned)
            ),
            "module_tensor_buffer_count": float(len(buffers)),
            "persistent_module_state_key_count": float(len(state)),
            "persistent_module_state_bytes": float(
                sum(value.numel() * value.element_size() for value in state)
            ),
            "optimizer_updates": 0.0,
            "optimizer_state_entry_count": 0.0,
            "scheduler_state_entry_count": 0.0,
            "rng_state_entry_count": 0.0,
        }
    )
    metrics.update(_gradient_metrics(config, audit_batch, audit_schedule_values))
    latency = _latency_metrics(
        config,
        audit_batch,
        process_rss_start_bytes=process_rss_start_bytes,
    )
    latency["persistent_runtime_tensor_state_bytes_max"] = max(
        latency["persistent_runtime_tensor_state_bytes_max"],
        metrics["persistent_runtime_tensor_state_bytes_max"],
    )
    metrics.update(latency)
    for name, value in metrics.items():
        if _numeric(value) is None:
            raise FloatingPointError(f"partial-visibility metric {name!r} is nonfinite")
    failures = gate_failures(metrics, split=split)
    result = {
        "split": split,
        "seeds": list(requested),
        "seed_manifest_sha256": canonical_sha256(list(requested)),
        "scene_parameter_signature_sha256": SCENE_PARAMETER_SIGNATURE_SHA256[split],
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
        "optimizer_updates": 0,
        "runtime_api": {
            "packet_factory": "make_rgbd_packet",
            "ingest_frames": list(INGEST_FRAME_INDICES),
            "rollout_method": "OnlineWorldModel.predict",
            "horizons_seconds": list(HORIZONS_SECONDS),
        },
        "scene_constructor": (
            "construct_partial_visibility_episode_with_full_frame_preflight_and_local_depth_miss"
        ),
    }
    _validate_manifest_result(result, split=split)
    authorization.seal_result(_EVALUATOR_RESULT_AUTHORITY, result)
    return result


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_RELATIVE_PATH = Path("runs/rgbd_partial_visibility_recovery_v2")
DEVELOPMENT_REPORT_NAME = "development_report.json"
DEVELOPMENT_CHECKPOINT_NAME = "development_model.pt"
QUALIFICATION_REPORT_NAME = "qualification_report.json"


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without following any symbolic links."""

    return Path(os.path.abspath(os.fspath(path)))


def _canonical_run_directory() -> Path:
    root = _absolute_lexical(REPOSITORY_ROOT)
    if RUN_RELATIVE_PATH.is_absolute() or ".." in RUN_RELATIVE_PATH.parts:
        raise RuntimeError("qualification run path must remain repository-relative")
    directory = _absolute_lexical(root / RUN_RELATIVE_PATH)
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise RuntimeError("qualification run path escaped the repository") from error
    current = root
    for component in RUN_RELATIVE_PATH.parts:
        current /= component
        if _lexists(current) and current.is_symlink():
            raise ValueError(f"qualification run path component must not be a symlink: {current}")
    if directory.resolve() != directory:
        raise ValueError("qualification run directory resolves outside its lexical path")
    return directory


def development_ledger_path() -> Path:
    return _canonical_run_directory() / (f"development_attempt_{ARCHITECTURE_ATTEMPT}_access.json")


def qualification_ledger_path() -> Path:
    return _canonical_run_directory() / (
        f"qualification_attempt_{ARCHITECTURE_ATTEMPT}_access.json"
    )


def canonical_artifact_paths() -> dict[str, Path]:
    directory = _canonical_run_directory()
    return {
        "development_report": directory / DEVELOPMENT_REPORT_NAME,
        "development_checkpoint": directory / DEVELOPMENT_CHECKPOINT_NAME,
        "development_ledger": development_ledger_path(),
        "qualification_report": directory / QUALIFICATION_REPORT_NAME,
        "qualification_ledger": qualification_ledger_path(),
    }


def _require_canonical_path(path: Path, *, artifact: str) -> None:
    expected = canonical_artifact_paths()[artifact]
    if _absolute_lexical(Path(path)) != expected:
        raise ValueError(f"{artifact} must use the canonical qualification path {expected}")


def _require_single_link(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    metadata = path.stat()
    if not path.is_file() or metadata.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file: {path}")


def _single_link_read_bytes(path: Path, *, label: str) -> bytes:
    _require_single_link(path, label=label)
    contents = stable_read_bytes(path, label=label)
    _require_single_link(path, label=label)
    return contents


def _attempt1_run_directory() -> Path:
    root = _absolute_lexical(REPOSITORY_ROOT)
    if ATTEMPT1_RUN_RELATIVE_PATH.is_absolute() or ".." in ATTEMPT1_RUN_RELATIVE_PATH.parts:
        raise RuntimeError("attempt-one archive path must remain repository-relative")
    directory = _absolute_lexical(root / ATTEMPT1_RUN_RELATIVE_PATH)
    directory.relative_to(root)
    current = root
    for component in ATTEMPT1_RUN_RELATIVE_PATH.parts:
        current /= component
        if _lexists(current) and current.is_symlink():
            raise ValueError("attempt-one archive path must not contain symbolic links")
    if directory.resolve() != directory:
        raise ValueError("attempt-one archive resolves outside its lexical path")
    return directory


def _validate_attempt1_rejection_bytes(
    report_contents: bytes,
    ledger_contents: bytes,
) -> None:
    """Validate the exact attempt-one rejection from bytes alone."""

    if sha256_bytes(report_contents) != ATTEMPT1_REPORT_SHA256:
        raise RuntimeError("attempt-one failed report bytes changed")
    if sha256_bytes(ledger_contents) != ATTEMPT1_LEDGER_SHA256:
        raise RuntimeError("attempt-one failed ledger bytes changed")
    try:
        report = json.loads(report_contents)
        ledger = json.loads(ledger_contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("attempt-one rejection evidence is not canonical JSON") from exc
    if not isinstance(report, Mapping) or not isinstance(ledger, Mapping):
        raise RuntimeError("attempt-one rejection evidence must contain JSON objects")
    expected_error = {
        "type": "RuntimeError",
        "message": "partial scene left its declared renderer visibility band",
    }
    expected_report_fields = {
        "artifact_kind": "rgbd_partial_visibility_development",
        "config_sha256": ATTEMPT1_CONFIG_SHA256,
        "optimizer_updates": 0,
        "passed": False,
        "protected_data_materialized": False,
        "review_ready": False,
        "stopped_after": "development",
        "error": expected_error,
        "source_provenance": ATTEMPT1_SOURCE_PROVENANCE,
    }
    for name, expected in expected_report_fields.items():
        if not _typed_canonical_equal(report.get(name), expected):
            raise RuntimeError(f"attempt-one failed report field changed: {name}")
    if report.get("development_ledger") != ATTEMPT1_DEVELOPMENT_LEDGER_BACKLINK:
        raise RuntimeError("attempt-one report-to-ledger backlink changed")
    if set(report) != {
        "artifact_kind",
        "config_sha256",
        "development_ledger",
        "error",
        "optimizer_updates",
        "passed",
        "protected_data_materialized",
        "protocol",
        "review_ready",
        "source_provenance",
        "stopped_after",
    }:
        raise RuntimeError("attempt-one failed report schema changed")
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError("attempt-one failed report lacks its protocol")
    unsigned_protocol = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    expected_manifests = {
        "development": list(range(53_000_000, 53_000_032)),
        "selector": list(range(54_000_000, 54_000_024)),
        "confirmation": list(range(55_000_000, 55_000_024)),
        "final_test": list(range(56_000_000, 56_000_048)),
    }
    expected_manifest_hashes = {
        "development": ATTEMPT1_DEVELOPMENT_MANIFEST_SHA256,
        "selector": "1b1e6ef6938705bcc7e2a66ad5ee4622860c9ea9ec3e6c19c86e8a8534209b28",
        "confirmation": "72d7c922029d300e3d28409bcb55a843633caac10b482f680ae769a442739e9f",
        "final_test": "70b60f48769a26c5587febf778443fd38f5814a39e80ec7da1c98dea9c389ded",
    }
    protocol_fields = {
        "name": "rgbd_partial_visibility_recovery_v1",
        "architecture_version": 1,
        "architecture_attempt": 1,
        "maximum_architecture_attempts": 2,
        "resolved_config_sha256": ATTEMPT1_CONFIG_SHA256,
        "protocol_sha256": ATTEMPT1_PROTOCOL_SHA256,
        "manifests": expected_manifests,
        "manifest_sha256": expected_manifest_hashes,
    }
    for name, expected in protocol_fields.items():
        if not _typed_canonical_equal(protocol.get(name), expected):
            raise RuntimeError(f"attempt-one protocol binding changed: {name}")
    if canonical_sha256(unsigned_protocol) != ATTEMPT1_PROTOCOL_SHA256:
        raise RuntimeError("attempt-one protocol self-hash changed")
    expected_ledger_fields = {
        "artifact_kind": "rgbd_partial_visibility_development_access_ledger",
        "architecture_attempt": 1,
        "maximum_architecture_attempts": 2,
        "attempt_reserved": True,
        "access_started": True,
        "development_data_materialized": True,
        "result_sha256": None,
        "status": "error",
        "error": expected_error,
        "report_sha256": ATTEMPT1_REPORT_SHA256,
        "bindings": {
            "config_sha256": ATTEMPT1_CONFIG_SHA256,
            "development_manifest_sha256": ATTEMPT1_DEVELOPMENT_MANIFEST_SHA256,
            "protocol_sha256": ATTEMPT1_PROTOCOL_SHA256,
            "source_provenance": ATTEMPT1_SOURCE_PROVENANCE,
        },
    }
    if set(ledger) != set(expected_ledger_fields):
        raise RuntimeError("attempt-one failed ledger schema changed")
    for name, expected in expected_ledger_fields.items():
        if not _typed_canonical_equal(ledger.get(name), expected):
            raise RuntimeError(f"attempt-one failed ledger field changed: {name}")


def _require_attempt1_rejection() -> None:
    """Require the exact immutable failed attempt-one archive before access."""

    directory = _attempt1_run_directory()
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError("attempt-one rejection archive is absent or not a real directory")
    report_path = directory / DEVELOPMENT_REPORT_NAME
    ledger_path = directory / "development_attempt_1_access.json"
    expected_inventory = {report_path, ledger_path}
    if set(directory.iterdir()) != expected_inventory:
        raise RuntimeError(
            "attempt-one rejection archive must contain exactly its failed report and ledger"
        )
    report_contents = _single_link_read_bytes(report_path, label="attempt-one failed report")
    ledger_contents = _single_link_read_bytes(ledger_path, label="attempt-one failed ledger")
    _validate_attempt1_rejection_bytes(report_contents, ledger_contents)
    if set(directory.iterdir()) != expected_inventory:
        raise RuntimeError("attempt-one rejection archive changed while it was checked")


def _validate_artifact_inventory(*, allowed_existing: Sequence[str]) -> None:
    paths = canonical_artifact_paths()
    directory = next(iter(paths.values())).parent
    allowed = {paths[name] for name in allowed_existing}
    if _lexists(directory):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("qualification artifact directory must be a real directory")
        for child in directory.iterdir():
            if child not in set(paths.values()):
                raise ValueError(f"unexpected qualification artifact inventory entry: {child}")
            if child not in allowed:
                raise FileExistsError(f"qualification artifact must remain fresh: {child}")
            _require_single_link(child, label="existing qualification artifact")
    for name in allowed_existing:
        path = paths[name]
        if not _lexists(path):
            raise FileNotFoundError(f"required canonical qualification artifact is absent: {path}")
        _require_single_link(path, label=f"canonical {name}")


def require_frozen_config(path: str | Path) -> OrpheusConfig:
    source = Path(path)
    contents = stable_read_bytes(source, label="partial-visibility frozen config")
    digest = sha256_bytes(contents)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "partial-visibility RGB-D requires exact frozen config bytes: "
            f"expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    config = load_config(source)
    assert_rgbd_partial_visibility_config(config)
    return config


def _require_config_matches(config: OrpheusConfig, path: Path) -> None:
    before = stable_read_bytes(path, label="partial-visibility config binding")
    if sha256_bytes(before) != FROZEN_CONFIG_SHA256:
        raise ValueError("executed config path differs from frozen bytes")
    parsed = load_config(path)
    after = stable_read_bytes(path, label="partial-visibility config binding recheck")
    if before != after:
        raise RuntimeError("frozen config changed while it was parsed")
    if canonical_sha256(parsed.to_dict()) != canonical_sha256(config.to_dict()):
        raise ValueError("executed config object differs from frozen bytes")
    assert_rgbd_partial_visibility_config(parsed)


class DevelopmentLedger:
    """Fresh sole-attempt-two development receipt and authorization owner."""

    ARTIFACT_KIND = "rgbd_partial_visibility_development_access_ledger"

    def __init__(self, bindings: Mapping[str, Any], *, authority: object) -> None:
        if authority is not _LEDGER_CONSTRUCTION_AUTHORITY:
            raise PermissionError("development ledger requires the frozen runner authority")
        _require_attempt1_rejection()
        self.path = development_ledger_path()
        _require_canonical_path(self.path, artifact="development_ledger")
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
            "bindings": dict(bindings),
            "attempt_reserved": True,
            "access_started": True,
            "development_data_materialized": True,
            "result_sha256": None,
            "status": "development_materialization_started",
        }
        self._authorization: _ManifestAccessAuthorization | None = None
        self._authorization_mint = object()
        self._issued = False
        _durable_create(self.path, self._serialized())
        self._started_receipt_sha256 = sha256_bytes(
            _single_link_read_bytes(self.path, label="development started-ledger receipt")
        )

    def _serialized(self) -> bytes:
        return json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"

    def _replace(self) -> None:
        _require_single_link(self.path, label="development access ledger")
        _durable_replace(self.path, self._serialized())

    def authorization(self) -> _ManifestAccessAuthorization:
        _require_attempt1_rejection()
        if self._issued:
            raise RuntimeError("development authorization cannot be issued twice")
        self._issued = True
        self._authorization = _ManifestAccessAuthorization(
            _MANIFEST_ACCESS_AUTHORITY,
            issuer=self,
            mint=self._authorization_mint,
            split="development",
            seeds=DEVELOPMENT_SEEDS,
            ledger_path=self.path,
            ledger_kind=self.ARTIFACT_KIND,
            receipt_sha256=self._started_receipt_sha256,
        )
        return self._authorization

    def complete_evaluation(self, result: Mapping[str, Any]) -> None:
        if self.record["status"] != "development_materialization_started":
            raise RuntimeError("development was not durably opened")
        if self._authorization is None:
            raise RuntimeError("development authorization was not issued")
        _validate_manifest_result(result, split="development")
        self._authorization.require_result(result)
        passed = result.get("passed") is True
        self.record["result_sha256"] = canonical_sha256(result)
        self.record["outcome"] = "passed" if passed else "failed"
        self.record["status"] = "development_artifacts_pending"
        self._replace()

    def finish(self, *, report_sha256: str, checkpoint_sha256: str | None) -> None:
        if self.record["status"] != "development_artifacts_pending":
            raise RuntimeError("development artifacts were not pending")
        self.record["report_sha256"] = validated_sha256(
            report_sha256, label="development report SHA-256"
        )
        self.record["checkpoint_sha256"] = (
            None
            if checkpoint_sha256 is None
            else validated_sha256(checkpoint_sha256, label="development checkpoint SHA-256")
        )
        self.record["status"] = "complete" if self.record.get("outcome") == "passed" else "failed"
        self._replace()

    def record_error(self, error: BaseException, *, report_sha256: str | None = None) -> None:
        if self.record.get("status") == "complete":
            raise RuntimeError("complete development ledger cannot be downgraded")
        self.record["status"] = "error"
        self.record["error"] = {"type": type(error).__name__, "message": str(error)}
        if report_sha256 is not None:
            self.record["report_sha256"] = validated_sha256(
                report_sha256, label="failed development report SHA-256"
            )
        self._replace()


class QualificationLedger:
    """Exclusive selector -> confirmation -> final receipt."""

    ARTIFACT_KIND = "rgbd_partial_visibility_exactly_once_access_ledger"
    ORDER = ("selector", "confirmation", "final_test")

    def __init__(self, bindings: Mapping[str, Any], *, authority: object) -> None:
        if authority is not _LEDGER_CONSTRUCTION_AUTHORITY:
            raise PermissionError("qualification ledger requires the frozen runner authority")
        _require_attempt1_rejection()
        self.path = qualification_ledger_path()
        _require_canonical_path(self.path, artifact="qualification_ledger")
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
            "order": list(self.ORDER),
            "bindings": dict(bindings),
            "splits": {
                split: {"access_started": False, "status": "unopened", "result_sha256": None}
                for split in self.ORDER
            },
            "attempt_reserved": True,
            "protected_data_materialized": False,
            "status": "reserved_before_protected_access",
        }
        self._authorizations: dict[str, _ManifestAccessAuthorization] = {}
        self._authorization_mint = object()
        _durable_create(self.path, self._serialized())

    def _serialized(self) -> bytes:
        return json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"

    def _replace(self) -> None:
        _require_single_link(self.path, label="qualification access ledger")
        _durable_replace(self.path, self._serialized())

    def begin_access(self, split: str) -> _ManifestAccessAuthorization:
        _require_attempt1_rejection()
        if split not in self.ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        index = self.ORDER.index(split)
        for predecessor in self.ORDER[:index]:
            if self.record["splits"][predecessor]["status"] != "passed":
                raise RuntimeError(f"{split} must remain unopened until {predecessor} passes")
        state = self.record["splits"][split]
        if state["status"] != "unopened" or state["access_started"] is not False:
            raise RuntimeError(f"protected split {split!r} cannot be opened twice")
        state["access_started"] = True
        state["status"] = "materialization_started"
        self.record["protected_data_materialized"] = True
        self.record["status"] = f"{split}_materialization_started"
        self._replace()
        receipt_sha256 = sha256_bytes(
            _single_link_read_bytes(self.path, label=f"{split} started-ledger receipt")
        )
        authorization = _ManifestAccessAuthorization(
            _MANIFEST_ACCESS_AUTHORITY,
            issuer=self,
            mint=self._authorization_mint,
            split=split,
            seeds=MANIFESTS[split],
            ledger_path=self.path,
            ledger_kind=self.ARTIFACT_KIND,
            receipt_sha256=receipt_sha256,
        )
        self._authorizations[split] = authorization
        return authorization

    def complete_split(self, split: str, result: Mapping[str, Any]) -> None:
        state = self.record["splits"].get(split)
        if not isinstance(state, Mapping) or state.get("status") != "materialization_started":
            raise RuntimeError(f"protected split {split!r} was not durably opened")
        authorization = self._authorizations.get(split)
        if authorization is None:
            raise RuntimeError(f"protected split {split!r} lacks authorization")
        _validate_manifest_result(result, split=split)
        authorization.require_result(result)
        passed = result.get("passed") is True
        state["status"] = "passed" if passed else "failed"
        state["result_sha256"] = canonical_sha256(result)
        self.record["status"] = f"{split}_{state['status']}"
        self._replace()

    def prepare_report(self, *, passed: bool, stopped_after: str) -> None:
        if passed and any(
            self.record["splits"][split]["status"] != "passed" for split in self.ORDER
        ):
            raise RuntimeError("qualification cannot pass before every split passes")
        self.record["outcome"] = "passed" if passed else "failed"
        self.record["stopped_after"] = stopped_after
        self.record["status"] = "qualification_report_write_pending"
        self._replace()

    def finish(self, *, report_sha256: str) -> None:
        if self.record["status"] != "qualification_report_write_pending":
            raise RuntimeError("qualification report was not pending")
        self.record["report_sha256"] = validated_sha256(
            report_sha256, label="qualification report SHA-256"
        )
        self.record["status"] = "complete" if self.record.get("outcome") == "passed" else "failed"
        self._replace()

    def record_error(
        self,
        error: BaseException,
        *,
        stopped_after: str,
        report_sha256: str | None = None,
    ) -> None:
        if self.record.get("status") == "complete":
            raise RuntimeError("complete qualification ledger cannot be downgraded")
        self.record["status"] = "error"
        self.record["stopped_after"] = stopped_after
        self.record["error"] = {"type": type(error).__name__, "message": str(error)}
        if report_sha256 is not None:
            self.record["report_sha256"] = validated_sha256(
                report_sha256, label="failed qualification report SHA-256"
            )
        self._replace()


def _model_state_sha256(model: OnlineWorldModel) -> str:
    if model.state_dict():
        raise RuntimeError("partial-visibility runtime state_dict must remain empty")
    return EMPTY_MODEL_STATE_SHA256


def _save_review_checkpoint(
    path: Path,
    *,
    model: OnlineWorldModel,
    config: OrpheusConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    payload = checkpoint_payload(
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        step=0,
        metrics=metrics,
        device="cpu",
        source_provenance=source,
    )
    payload.pop("rng", None)
    temporary = _atomic_temporary(path)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _durable_create(temporary, buffer.getvalue())
    try:
        # Linking is an atomic, exclusive publication: unlike replace(), it
        # cannot overwrite a concurrently created reviewed checkpoint.
        os.link(temporary, path)
        _fsync_parent(path)
        os.unlink(temporary)
        _fsync_parent(path)
        _require_single_link(path, label="development checkpoint")
    except BaseException:
        # Retain any temporary or multiply-linked file as ambiguous evidence.
        raise


def _load_checkpoint_payload(contents: bytes) -> Mapping[str, Any]:
    payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("reviewed checkpoint payload must be a mapping")
    return payload


def validate_checkpoint_evidence(
    payload: Mapping[str, Any],
    *,
    config: OrpheusConfig,
    source: Mapping[str, Any],
    development: Mapping[str, Any],
) -> None:
    expected_keys = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "step",
        "config",
        "metrics",
        "project_version",
        "specification_version",
        "simulator_version",
        "device",
        "precision",
        "git",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("reviewed checkpoint must have the exact typed payload schema")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or model_state:
        raise ValueError("reviewed partial-visibility checkpoint model state must be empty")
    if type(payload.get("step")) is not int or payload.get("step") != 0:
        raise ValueError("reviewed checkpoint must be exact step zero")
    if payload.get("optimizer_state") is not None or payload.get("scheduler_state") is not None:
        raise ValueError("reviewed checkpoint must be optimizer/scheduler-free")
    if "rng" in payload:
        raise ValueError("reviewed checkpoint must be RNG-free")
    for key, expected in {
        "project_version": __version__,
        "specification_version": SPECIFICATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "device": "cpu",
        "precision": "float32",
    }.items():
        if type(payload.get(key)) is not str or payload.get(key) != expected:
            raise ValueError(f"reviewed checkpoint {key} differs")
    if not _typed_canonical_equal(payload.get("config"), config.to_dict()):
        raise ValueError("reviewed checkpoint config differs")
    if not _typed_canonical_equal(payload.get("git"), dict(source)):
        raise ValueError("reviewed checkpoint source differs")
    expected = {
        "artifact_kind": "rgbd_partial_visibility_empty_model_state",
        "optimizer_updates": 0,
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "protocol": bridge_protocol(),
        "development": development,
    }
    if not _typed_canonical_equal(payload.get("metrics"), expected):
        raise ValueError("reviewed checkpoint evidence differs from frozen protocol")
    roundtrip = new_public_model(config)
    roundtrip.load_state_dict(model_state, strict=True)
    if _model_state_sha256(roundtrip) != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("checkpoint roundtrip changed empty model state")


def _validate_development_split(development: Mapping[str, Any]) -> None:
    _validate_manifest_result(development, split="development")
    if development.get("passed") is not True:
        raise ValueError("reviewed development does not recompute as passing")


def validate_development_evidence(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_keys = {
        "artifact_kind",
        "protocol",
        "source_provenance",
        "config_sha256",
        "development_ledger",
        "optimizer_updates",
        "protected_data_materialized",
        "passed",
        "review_ready",
        "stopped_after",
        "development",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_model_state_sha256",
        "checkpoint_roundtrip_state_sha256",
    }
    if type(report) is not dict or set(report) != expected_keys:
        raise ValueError("reviewed development report must have the exact passed schema")
    if report.get("artifact_kind") != "rgbd_partial_visibility_development":
        raise ValueError("reviewed development report kind differs")
    if report.get("passed") is not True or report.get("review_ready") is not True:
        raise ValueError("reviewed development did not pass")
    if report.get("protected_data_materialized") is not False:
        raise ValueError("reviewed development opened protected data")
    if (
        type(report.get("optimizer_updates")) is not int
        or report.get("optimizer_updates") != 0
        or report.get("stopped_after") != "development"
    ):
        raise ValueError("reviewed development crossed its execution boundary")
    if not _typed_canonical_equal(report.get("protocol"), bridge_protocol()):
        raise ValueError("reviewed protocol differs from frozen source")
    if report.get("config_sha256") != FROZEN_CONFIG_SHA256:
        raise ValueError("reviewed config hash differs")
    if not _typed_canonical_equal(report.get("source_provenance"), dict(source)):
        raise ValueError("reviewed source differs")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reviewed report does not bind checkpoint")
    if report.get("checkpoint") != str(canonical_artifact_paths()["development_checkpoint"]):
        raise ValueError("reviewed report names a noncanonical checkpoint")
    if report.get("development_ledger") != str(canonical_artifact_paths()["development_ledger"]):
        raise ValueError("reviewed report names a noncanonical development ledger")
    if report.get("checkpoint_model_state_sha256") != EMPTY_MODEL_STATE_SHA256:
        raise ValueError("reviewed report does not bind empty model state")
    if report.get("checkpoint_roundtrip_state_sha256") != EMPTY_MODEL_STATE_SHA256:
        raise ValueError("reviewed report does not bind the checkpoint roundtrip state")
    development = report.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("reviewed report omitted development result")
    _validate_development_split(development)
    return development


def validate_development_ledger(
    record: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    report_sha256: str,
    checkpoint_sha256: str,
    source: Mapping[str, Any],
    development: Mapping[str, Any],
) -> None:
    expected_keys = {
        "artifact_kind",
        "architecture_attempt",
        "maximum_architecture_attempts",
        "bindings",
        "attempt_reserved",
        "access_started",
        "development_data_materialized",
        "result_sha256",
        "status",
        "outcome",
        "report_sha256",
        "checkpoint_sha256",
    }
    if type(record) is not dict or set(record) != expected_keys:
        raise ValueError("reviewed development ledger must have the exact completed schema")
    expected_bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": dict(source),
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_manifest_sha256": MANIFEST_SHA256["development"],
    }
    if record.get("artifact_kind") != DevelopmentLedger.ARTIFACT_KIND:
        raise ValueError("reviewed development ledger kind differs")
    if (
        type(record.get("architecture_attempt")) is not int
        or record.get("architecture_attempt") != ARCHITECTURE_ATTEMPT
    ):
        raise ValueError("reviewed development ledger attempt differs")
    if (
        type(record.get("maximum_architecture_attempts")) is not int
        or record.get("maximum_architecture_attempts") != MAX_ARCHITECTURE_ATTEMPTS
    ):
        raise ValueError("reviewed development ledger maximum attempts differ")
    for key in ("attempt_reserved", "access_started", "development_data_materialized"):
        if record.get(key) is not True:
            raise ValueError(f"reviewed development ledger {key} is not exact true")
    if not _typed_canonical_equal(record.get("bindings"), expected_bindings):
        raise ValueError("reviewed development ledger bindings differ")
    if record.get("status") != "complete" or record.get("outcome") != "passed":
        raise ValueError("reviewed development ledger is not complete/passed")
    if record.get("result_sha256") != canonical_sha256(development):
        raise ValueError("reviewed development ledger result hash differs")
    if record.get("report_sha256") != report_sha256:
        raise ValueError("reviewed development ledger report hash differs")
    if record.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reviewed development ledger checkpoint hash differs")
    if report.get("development_ledger") != str(development_ledger_path()):
        raise ValueError("reviewed report names a nonfixed development ledger")


def _guard_frozen_inputs(
    *,
    source: Mapping[str, Any],
    config: OrpheusConfig,
    config_path: Path,
    bound_files: Sequence[tuple[Path, str, str]] = (),
    model: OnlineWorldModel | None = None,
) -> None:
    current = clean_source(
        capture_git_metadata(REPOSITORY_ROOT), label="partial-visibility execution guard"
    )
    if not _typed_canonical_equal(current, dict(source)):
        raise RuntimeError("source provenance changed during qualification")
    _require_config_matches(config, config_path)
    protocol = bridge_protocol()
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol["protocol_sha256"] != canonical_sha256(unsigned):
        raise RuntimeError("protocol self-hash is inconsistent")
    for path, digest, label in bound_files:
        if sha256_bytes(_single_link_read_bytes(path, label=label)) != digest:
            raise RuntimeError(f"{label} changed during qualification")
    if model is not None and _model_state_sha256(model) != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("public model state changed during qualification")


def run_development(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    source_provenance: Mapping[str, Any],
) -> int:
    """Consume the single attempt-two development manifest and stop."""

    _require_attempt1_rejection()
    assert_rgbd_partial_visibility_config(config)
    _assert_execution_environment()
    source = clean_source(source_provenance, label="partial-visibility development")
    _require_config_matches(config, config_path)
    ledger_path = development_ledger_path()
    _require_canonical_path(report_path, artifact="development_report")
    _require_canonical_path(checkpoint_path, artifact="development_checkpoint")
    _validate_artifact_inventory(allowed_existing=())
    validate_distinct_paths(
        {
            "config": config_path,
            "report": report_path,
            "checkpoint": checkpoint_path,
            "development_ledger": ledger_path,
        },
        atomic_writers=("report", "checkpoint", "development_ledger"),
    )
    for path in (
        report_path,
        checkpoint_path,
        ledger_path,
        _atomic_temporary(report_path),
        _atomic_temporary(checkpoint_path),
        _atomic_temporary(ledger_path),
    ):
        if _lexists(path):
            raise FileExistsError(f"development artifact must be fresh: {path}")
    protocol = bridge_protocol()
    model = new_public_model(config)
    _guard_frozen_inputs(source=source, config=config, config_path=config_path, model=model)
    ledger = DevelopmentLedger(
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "source_provenance": source,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "development_manifest_sha256": MANIFEST_SHA256["development"],
        },
        authority=_LEDGER_CONSTRUCTION_AUTHORITY,
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_partial_visibility_development",
        "protocol": protocol,
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_ledger": str(ledger.path),
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "passed": False,
        "review_ready": False,
        "stopped_after": "development",
    }
    try:
        development = _evaluate_seed_manifest(
            config,
            DEVELOPMENT_SEEDS,
            split="development",
            authorization=ledger.authorization(),
        )
        ledger.complete_evaluation(development)
        report["development"] = development
        report["passed"] = development["passed"]
        report["review_ready"] = development["passed"]
        checkpoint_digest: str | None = None
        _guard_frozen_inputs(source=source, config=config, config_path=config_path, model=model)
        if development["passed"]:
            checkpoint_metrics = {
                "artifact_kind": "rgbd_partial_visibility_empty_model_state",
                "optimizer_updates": 0,
                "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
                "protocol": protocol,
                "development": development,
            }
            _save_review_checkpoint(
                checkpoint_path,
                model=model,
                config=config,
                metrics=checkpoint_metrics,
                source=source,
            )
            checkpoint_contents = _single_link_read_bytes(
                checkpoint_path, label="development checkpoint"
            )
            checkpoint_payload_contents = _load_checkpoint_payload(checkpoint_contents)
            validate_checkpoint_evidence(
                checkpoint_payload_contents,
                config=config,
                source=source,
                development=development,
            )
            checkpoint_digest = sha256_bytes(checkpoint_contents)
            report["checkpoint"] = str(checkpoint_path.resolve())
            report["checkpoint_sha256"] = checkpoint_digest
            report["checkpoint_model_state_sha256"] = EMPTY_MODEL_STATE_SHA256
            roundtrip_model = new_public_model(config)
            roundtrip_model.load_state_dict(checkpoint_payload_contents["model_state"], strict=True)
            report["checkpoint_roundtrip_state_sha256"] = _model_state_sha256(roundtrip_model)
        _guard_frozen_inputs(source=source, config=config, config_path=config_path, model=model)
        write_report_fresh(report_path, report)
        report_digest = sha256_bytes(
            _single_link_read_bytes(report_path, label="development report after write")
        )
        if development["passed"]:
            _validate_artifact_inventory(
                allowed_existing=(
                    "development_report",
                    "development_checkpoint",
                    "development_ledger",
                )
            )
        ledger.finish(
            report_sha256=report_digest,
            checkpoint_sha256=checkpoint_digest,
        )
        return 0 if development["passed"] else 1
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report_digest = None
        if not _lexists(report_path):
            try:
                write_report_fresh(report_path, report)
                report_digest = sha256_bytes(
                    _single_link_read_bytes(report_path, label="failed development report")
                )
            except BaseException:
                report_digest = None
        else:
            report_digest = sha256_bytes(
                _single_link_read_bytes(report_path, label="already-created development report")
            )
        ledger.record_error(error, report_sha256=report_digest)
        raise


def run_qualification(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    reviewed_checkpoint_sha256: str | None,
    reviewed_report_sha256: str | None,
    reviewed_development_ledger_sha256: str | None,
    source_provenance: Mapping[str, Any],
) -> int:
    """Consume selector, confirmation, and final once after exact review."""

    _require_attempt1_rejection()
    assert_rgbd_partial_visibility_config(config)
    _assert_execution_environment()
    source = clean_source(source_provenance, label="partial-visibility qualification")
    _require_config_matches(config, config_path)
    ledger_path = qualification_ledger_path()
    development_ledger = development_ledger_path()
    checkpoint_digest = validated_sha256(
        reviewed_checkpoint_sha256, label="reviewed checkpoint SHA-256"
    )
    development_report_digest = validated_sha256(
        reviewed_report_sha256, label="reviewed development report SHA-256"
    )
    development_ledger_digest = validated_sha256(
        reviewed_development_ledger_sha256,
        label="reviewed development ledger SHA-256",
    )
    _require_canonical_path(report_path, artifact="qualification_report")
    _require_canonical_path(checkpoint_path, artifact="development_checkpoint")
    _require_canonical_path(development_report_path, artifact="development_report")
    _validate_artifact_inventory(
        allowed_existing=(
            "development_report",
            "development_checkpoint",
            "development_ledger",
        )
    )
    validate_distinct_paths(
        {
            "config": config_path,
            "report": report_path,
            "checkpoint": checkpoint_path,
            "development_report": development_report_path,
            "development_ledger": development_ledger,
            "qualification_ledger": ledger_path,
        },
        atomic_writers=("report", "qualification_ledger"),
    )
    for path in (
        report_path,
        ledger_path,
        _atomic_temporary(report_path),
        _atomic_temporary(ledger_path),
    ):
        if _lexists(path):
            raise FileExistsError(f"qualification artifact must be fresh: {path}")
    checkpoint_contents = _single_link_read_bytes(checkpoint_path, label="reviewed checkpoint")
    if sha256_bytes(checkpoint_contents) != checkpoint_digest:
        raise ValueError("reviewed checkpoint hash does not match bytes")
    development_report_contents = _single_link_read_bytes(
        development_report_path, label="reviewed development report"
    )
    if sha256_bytes(development_report_contents) != development_report_digest:
        raise ValueError("reviewed development report hash does not match bytes")
    development_report = json.loads(development_report_contents)
    if not isinstance(development_report, Mapping):
        raise ValueError("reviewed development report must be a JSON object")
    development_ledger_contents = _single_link_read_bytes(
        development_ledger, label="reviewed development access ledger"
    )
    if sha256_bytes(development_ledger_contents) != development_ledger_digest:
        raise ValueError("reviewed development ledger hash does not match bytes")
    development_ledger_record = json.loads(development_ledger_contents)
    if not isinstance(development_ledger_record, Mapping):
        raise ValueError("reviewed development ledger must be a JSON object")
    development = validate_development_evidence(
        development_report,
        checkpoint_sha256=checkpoint_digest,
        source=source,
    )
    validate_development_ledger(
        development_ledger_record,
        report=development_report,
        report_sha256=development_report_digest,
        checkpoint_sha256=checkpoint_digest,
        source=source,
        development=development,
    )
    payload = _load_checkpoint_payload(checkpoint_contents)
    validate_checkpoint_evidence(payload, config=config, source=source, development=development)
    model = new_public_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    initial_state = _model_state_sha256(model)
    bound_files = (
        (development_report_path, development_report_digest, "reviewed development report"),
        (development_ledger, development_ledger_digest, "reviewed development ledger"),
        (checkpoint_path, checkpoint_digest, "reviewed checkpoint"),
    )
    _guard_frozen_inputs(
        source=source,
        config=config,
        config_path=config_path,
        bound_files=bound_files,
        model=model,
    )
    ledger = QualificationLedger(
        {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": source,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "reviewed_checkpoint_sha256": checkpoint_digest,
            "reviewed_development_report_sha256": development_report_digest,
            "reviewed_development_ledger_sha256": development_ledger_digest,
            "model_state_sha256": initial_state,
        },
        authority=_LEDGER_CONSTRUCTION_AUTHORITY,
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_partial_visibility_protected_qualification",
        "protocol": bridge_protocol(),
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": checkpoint_digest,
        "reviewed_development_report_sha256": development_report_digest,
        "reviewed_development_ledger_sha256": development_ledger_digest,
        "initial_model_state_sha256": initial_state,
        "optimizer_updates": 0,
        "passed": False,
        "protected_data_materialized": False,
        "stopped_after": "reviewed_development",
    }
    try:
        for split in QualificationLedger.ORDER:
            _guard_frozen_inputs(
                source=source,
                config=config,
                config_path=config_path,
                bound_files=bound_files,
                model=model,
            )
            authorization = ledger.begin_access(split)
            report["protected_data_materialized"] = True
            report["stopped_after"] = split
            result = _evaluate_seed_manifest(
                config,
                MANIFESTS[split],
                split=split,
                authorization=authorization,
            )
            report[split] = result
            ledger.complete_split(split, result)
            _guard_frozen_inputs(
                source=source,
                config=config,
                config_path=config_path,
                bound_files=bound_files,
                model=model,
            )
            if not result["passed"]:
                report["failures"] = list(result["failures"])
                break
        else:
            report["passed"] = True
            report["failures"] = []
        report["final_model_state_sha256"] = _model_state_sha256(model)
        if report["final_model_state_sha256"] != initial_state:
            raise RuntimeError("public model state changed during protected qualification")
        ledger.prepare_report(
            passed=bool(report["passed"]), stopped_after=str(report["stopped_after"])
        )
        report["qualification_ledger"] = str(ledger.path)
        _guard_frozen_inputs(
            source=source,
            config=config,
            config_path=config_path,
            bound_files=bound_files,
            model=model,
        )
        write_report_fresh(report_path, report)
        qualification_digest = sha256_bytes(
            _single_link_read_bytes(report_path, label="completed qualification report")
        )
        _validate_artifact_inventory(allowed_existing=tuple(canonical_artifact_paths()))
        ledger.finish(report_sha256=qualification_digest)
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        error_digest = None
        if not _lexists(report_path):
            try:
                write_report_fresh(report_path, report)
                error_digest = sha256_bytes(
                    _single_link_read_bytes(report_path, label="failed qualification report")
                )
            except BaseException:
                error_digest = None
        else:
            error_digest = sha256_bytes(
                _single_link_read_bytes(
                    report_path, label="qualification report before ledger error"
                )
            )
        ledger.record_error(
            error,
            stopped_after=str(report["stopped_after"]),
            report_sha256=error_digest,
        )
        raise
    return 0 if report["passed"] else 1


__all__ = [
    "ANCHOR_FRAME_INDEX",
    "ARCHITECTURE_ATTEMPT",
    "ARCHITECTURE_VERSION",
    "CONFIRMATION_SEEDS",
    "DEFAULT_GATES",
    "DEVELOPMENT_SEEDS",
    "EMPTY_MODEL_STATE_SHA256",
    "FINAL_TEST_SEEDS",
    "FROZEN_CONFIG_SHA256",
    "HORIZONS_SECONDS",
    "INGEST_FRAME_INDICES",
    "LIVE_HISTORY_FRAME_INDICES",
    "MANIFEST_SHA256",
    "MISS_FRAME_INDICES",
    "PartialVisibilityRecoveryGates",
    "SELECTOR_SEEDS",
    "STRATUM_NAMES",
    "TARGET_FRAME_INDICES",
    "bridge_protocol",
    "development_ledger_path",
    "gate_failures",
    "new_public_model",
    "preflight_partial_visibility_episode",
    "qualification_ledger_path",
    "require_frozen_config",
    "run_development",
    "run_qualification",
    "scene_schedule",
    "scene_specification",
    "validate_checkpoint_evidence",
    "validate_development_evidence",
    "validate_development_ledger",
]
