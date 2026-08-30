"""Hardened qualification for the seedless identifiable-drag RGB-D rung.

The governed family is addressed only by ``{split, ordinal}`` rows.  Scene
materialisation is private and is possible only through a live, ledger-owned,
single-use nominal capability.  Development is consumed once, caches typed
sufficient evidence in memory, and calibrates the three parameter-free RGB-D
uncertainty buffers.  The protected ladder is then strictly selector,
confirmation, and final-test, using externally reviewed development bytes.

This module deliberately contains no retry namespace and no scene-selection
seed.  Reconstruction diagnostics are secondary; the frozen gates measure
behaviour, calibration, identity, counterfactual structure, differentiability,
and resource/state bounds.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import resource
import stat
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from world_model.belief import fast_packing_map, slow_packing_map
from world_model.datasets import collate_episodes
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.runtime.state import runtime_stream_key
from world_model.simulator import SphereState, render_spheres
from world_model.simulator.labels import make_perception_labels, validate_perception_labels
from world_model.training.checkpointing import capture_git_metadata, checkpoint_payload
from world_model.training.loop import make_rgbd_packet
from world_model.training.rgbd_identifiable_drag_scene import (
    CAMERA_STRATA,
    FRAME_COUNT,
    FRAME_RATE_HZ,
    FROZEN_CAMERA_TRACE_SHA256,
    FROZEN_CERTIFICATE_SHA256,
    FROZEN_COMBINED_TRACE_SHA256,
    FROZEN_METADATA_SHA256,
    FROZEN_PHYSICAL_TRACE_SHA256,
    FROZEN_RASTER_TRACE_SHA256,
    FROZEN_SPLIT_CAMERA_TRACE_SHA256,
    FROZEN_SPLIT_COMBINED_TRACE_SHA256,
    FROZEN_SPLIT_PHYSICAL_TRACE_SHA256,
    FROZEN_SPLIT_RASTER_TRACE_SHA256,
    HISTORY_FRAME_COUNT,
    SCENES_PER_SPLIT,
    SPLITS,
    counterfactual_twin_ordinal,
    initial_sphere_state,
    manual_physical_trajectory,
    orbital_camera_frame,
    scene_metadata,
    scene_signature,
    scene_specification,
)
from world_model.training.rgbd_online_bridge_qualification import (
    clean_source,
    stable_read_bytes,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION, __version__

Split = Literal["development", "selector", "confirmation", "final_test"]
_NATIVE_PATH_TYPE = type(Path())

ARCHITECTURE_VERSION = 1
ARCHITECTURE_ATTEMPT = 1
MAX_ARCHITECTURE_ATTEMPTS = 1
OPTIMIZER_UPDATES = 0
ORDINALS = tuple(range(64))
if len(ORDINALS) != SCENES_PER_SPLIT:
    raise RuntimeError("identifiable-drag scene count and ordinal manifest differ")

HISTORY_FRAME_INDICES = tuple(range(HISTORY_FRAME_COUNT))
ANCHOR_FRAME_INDEX = HISTORY_FRAME_COUNT - 1
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TARGET_FRAME_INDICES = tuple(
    ANCHOR_FRAME_INDEX + round(horizon * FRAME_RATE_HZ) for horizon in HORIZONS_SECONDS
)
OBJECT_INDICES = (0, 1)
AXIS_NAMES = ("x", "y", "z")
RUNTIME_STREAM_KEY = "rgbd:camera0:rgbd"
CALIBRATION_CONFIDENCE = 0.90
CALIBRATION_Z = 1.6448536269514722
CALIBRATION_RANK = 59
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024

REPOSITORY_ROOT = Path(__file__).parents[2]
RUN_RELATIVE_PATH = Path("runs/rgbd_two_visible_orbital_camera_identifiable_drag_v1")
DEVELOPMENT_REPORT_NAME = "development_report.json"
CHECKPOINT_NAME = "development_model.pt"
DEVELOPMENT_LEDGER_NAME = f"development_attempt_{ARCHITECTURE_ATTEMPT}_access.json"
QUALIFICATION_REPORT_NAME = "qualification_report.json"
QUALIFICATION_LEDGER_NAME = f"qualification_attempt_{ARCHITECTURE_ATTEMPT}_access.json"
DEVELOPMENT_ARTIFACT_NAMES = frozenset(
    {DEVELOPMENT_REPORT_NAME, CHECKPOINT_NAME, DEVELOPMENT_LEDGER_NAME}
)
QUALIFICATION_ARTIFACT_NAMES = frozenset(
    {*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_REPORT_NAME, QUALIFICATION_LEDGER_NAME}
)
if len(QUALIFICATION_ARTIFACT_NAMES) != 5:
    raise RuntimeError("identifiable-drag qualification owns exactly five artifacts")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def validated_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{label} must be an exact SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


_FROZEN_SCENE_CERTIFICATE_BINDING = {
    "artifact_kind": "rgbd_identifiable_drag_scene_family_offline_source_freeze",
    "runtime_recomputation_permitted": False,
    "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    "scenes_per_split": SCENES_PER_SPLIT,
    "splits": list(SPLITS),
    "metadata_sha256": FROZEN_METADATA_SHA256,
    "physical_trace_sha256": FROZEN_PHYSICAL_TRACE_SHA256,
    "camera_trace_sha256": FROZEN_CAMERA_TRACE_SHA256,
    "raster_trace_sha256": FROZEN_RASTER_TRACE_SHA256,
    "combined_trace_sha256": FROZEN_COMBINED_TRACE_SHA256,
    "split_physical_trace_sha256": dict(FROZEN_SPLIT_PHYSICAL_TRACE_SHA256),
    "split_camera_trace_sha256": dict(FROZEN_SPLIT_CAMERA_TRACE_SHA256),
    "split_raster_trace_sha256": dict(FROZEN_SPLIT_RASTER_TRACE_SHA256),
    "split_combined_trace_sha256": dict(FROZEN_SPLIT_COMBINED_TRACE_SHA256),
}
_FROZEN_SCENE_CERTIFICATE_BINDING_SCHEMA = frozenset(
    {
        "artifact_kind",
        "runtime_recomputation_permitted",
        "certificate_sha256",
        "scenes_per_split",
        "splits",
        "metadata_sha256",
        "physical_trace_sha256",
        "camera_trace_sha256",
        "raster_trace_sha256",
        "combined_trace_sha256",
        "split_physical_trace_sha256",
        "split_camera_trace_sha256",
        "split_raster_trace_sha256",
        "split_combined_trace_sha256",
    }
)


def _frozen_scene_certificate_binding() -> dict[str, Any]:
    """Return literal-only offline certificate bindings without scene access."""

    result = copy.deepcopy(_FROZEN_SCENE_CERTIFICATE_BINDING)
    if (
        type(result) is not dict
        or set(result) != _FROZEN_SCENE_CERTIFICATE_BINDING_SCHEMA
        or result["artifact_kind"] != "rgbd_identifiable_drag_scene_family_offline_source_freeze"
        or result["runtime_recomputation_permitted"] is not False
        or type(result["scenes_per_split"]) is not int
        or result["scenes_per_split"] != SCENES_PER_SPLIT
        or type(result["splits"]) is not list
        or result["splits"] != list(SPLITS)
    ):
        raise RuntimeError("frozen scene certificate binding schema changed")
    for name in (
        "certificate_sha256",
        "metadata_sha256",
        "physical_trace_sha256",
        "camera_trace_sha256",
        "raster_trace_sha256",
        "combined_trace_sha256",
    ):
        validated_sha256(result[name], label=f"frozen scene certificate {name}")
    for name in (
        "split_physical_trace_sha256",
        "split_camera_trace_sha256",
        "split_raster_trace_sha256",
        "split_combined_trace_sha256",
    ):
        values = result[name]
        if type(values) is not dict or list(values) != list(SPLITS):
            raise RuntimeError(f"frozen scene certificate {name} schema changed")
        for split in SPLITS:
            validated_sha256(
                values[split],
                label=f"frozen scene certificate {name} {split}",
            )
    return result


def _manifest_rows(split: str) -> tuple[dict[str, Any], ...]:
    if type(split) is not str or split not in SPLITS:
        raise ValueError(f"unknown identifiable-drag split {split!r}")
    return tuple({"split": split, "ordinal": ordinal} for ordinal in ORDINALS)


def _validate_manifest_rows(split: str, rows: Any) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != 64:
        raise ValueError(f"{split} manifest must be an exact 64-row list")
    for ordinal, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"split", "ordinal"}:
            raise ValueError(f"{split} manifest row {ordinal} has the wrong exact schema")
        if type(row["split"]) is not str or row["split"] != split:
            raise ValueError(f"{split} manifest row {ordinal} has the wrong split")
        if type(row["ordinal"]) is not int or row["ordinal"] != ordinal:
            raise ValueError(f"{split} manifest row {ordinal} has the wrong exact ordinal")
    if canonical_sha256(rows) != MANIFEST_SHA256[split]:
        raise ValueError(f"{split} manifest actual row hash differs")
    return rows


MANIFEST_SHA256 = {split: canonical_sha256(list(_manifest_rows(split))) for split in SPLITS}

# These bindings are source constants, not generated evidence.  They are
# intentionally kept explicit so source review can update them once before the
# eventual clean published-source freeze.
FROZEN_CONFIG_SHA256 = "a22f364601b8f87cdec3fd6bff7d757f134867bf66d9fa176c1f2d881a700c45"
FROZEN_SOURCE_SHA256 = {
    "analytic_drag": "7d63dfef35a33f0c8d5a4343c99112ea549978e085981583cab78533b98099f8",
    "analytic_free_motion": "b47f02e331a3470b8eb6f22df1843951ecc6318fdc02febe5ba949807c5eeb7d",
    "free_motion_fit": "86cecf192130a2c13c2f1fd74f8df9fe0ff375dbe4c25a90ee63b3c24cb1b1d4",
    "measurements": "6934de34260d7b2b6576ee8c49e7058282c4b15600655e0fab804a481eba9bc6",
    "rgbd_temporal": "f63d94afc5b71d8bed10696ea4d4a1fb7e8670db92a00f0915a415ec83fa4afa",
    "rgbd_observation": "f6d15d3311e690582744d50f72049c5150f5e12977d3a4f1d3a77bee1af763d5",
    "filter_correction": "deded5bc7f317db97f8a1782ff9d1e3ebf3769b8481d04c47395e5f4d76e2d45",
    "online_world_model": "539c189bff451a423b17caacb6b85eb8fcb727ac10d4033110689af16d2838e5",
    "checkpointing": "2e6bd0f8d360afd17baee8286686d44203024c287031e6bf32265827ee137fb1",
    "scene": "b3b2e9a71c4020b27cb502f5b0a33b4e4d0174cba2073693b67c9c26c44a9bfd",
}

PUBLIC_CALIBRATION_REGRESSION = {
    "position": {"float32_bits": "0x4127aa75", "additional_ulps": 0},
    "velocity": {
        "initial_float32_bits": "0x41249854",
        "deployed_float32_bits": "0x41249858",
        "additional_ulps": 4,
    },
    "drag": {"float32_bits": "0x3fa419c1", "additional_ulps": 0},
}


@dataclass(frozen=True, slots=True)
class IdentifiableDragGates:
    """Predeclared broad gates, frozen from public feasibility only."""

    object_fit_count: float = 128.0
    scene_count: float = 64.0
    position_scale_min: float = 0.0
    position_scale_max: float = 32.0
    velocity_scale_min: float = 0.0
    velocity_scale_max: float = 32.0
    drag_scale_min: float = 1.0
    drag_scale_max: float = 5.0
    current_position_rmse_m: float = 7.5e-5
    current_velocity_rmse_mps: float = 1.5e-4
    current_drag_rmse_per_s: float = 0.01
    current_log_drag_rmse: float = 0.08
    log_drag_gaussian_nll: float = -1.0
    horizon_position_rmse_m: tuple[float, ...] = (1.5e-4, 2.0e-4, 3.0e-4, 4.5e-4, 7.5e-4)
    horizon_velocity_rmse_mps: float = 5.0e-4
    coverage_min: float = 0.85
    coverage_max: float = 0.995
    position_joint_coverage_min: float = 0.60
    velocity_joint_coverage_min: float = 0.60
    drag_joint_coverage_min: float = 0.80
    rms_z_min: float = 0.25
    rms_z_max: float = 1.75
    mean_h2_position_width_m: float = 0.003
    mean_h2_velocity_width_mps: float = 0.002
    mean_h2_log_drag_width: float = 0.30
    max_h2_position_width_m: float = 0.01
    max_h2_velocity_width_mps: float = 0.005
    max_h2_log_drag_width: float = 1.0
    identity_coverage: float = 1.0
    counterfactual_pair_count: float = 32.0
    minimum_drag_excitation_m: float = 0.015
    minimum_profile_information: float = 1.0
    maximum_boundary_mass: float = 0.01
    drag_grid_point_count: float = 257.0
    minimum_input_gradient_l1: float = 1.0e-8
    maximum_input_gradient_l1: float = 1.0e8
    vjp_required_history_frames: float = 16.0
    evaluation_latency_seconds: float = 90.0
    vjp_latency_seconds: float = 30.0
    process_max_rss_bytes: float = float(1.5 * 1024**3)
    persistent_module_state_bytes: float = 12.0
    analytic_agreement_max_abs: float = 2.0e-5
    semigroup_max_abs: float = 1.0e-5


DEFAULT_GATES = IdentifiableDragGates()


def _gate_surface(metrics: Mapping[str, Any], *, schema_only: bool) -> tuple[list[str], set[str]]:
    gates = DEFAULT_GATES
    failures: list[str] = []
    required: set[str] = set()

    def value(key: str) -> float | None:
        required.add(key)
        if schema_only:
            return None
        candidate = metrics.get(key)
        if type(candidate) is not float or not math.isfinite(candidate):
            failures.append(f"{key}:missing_nonfinite_or_nonfloat")
            return None
        return candidate

    def maximum(key: str, limit: float) -> None:
        candidate = value(key)
        if candidate is not None and candidate > limit:
            failures.append(f"{key}:{candidate:.9g}>{limit:.9g}")

    def minimum(key: str, limit: float) -> None:
        candidate = value(key)
        if candidate is not None and candidate < limit:
            failures.append(f"{key}:{candidate:.9g}<{limit:.9g}")

    def equal(key: str, expected: float) -> None:
        candidate = value(key)
        if candidate is not None and candidate != expected:
            failures.append(f"{key}:{candidate:.9g}!={expected:.9g}")

    def positive(key: str) -> None:
        candidate = value(key)
        if candidate is not None and candidate <= 0.0:
            failures.append(f"{key}:{candidate:.9g}<=0")

    def strict_less(left: str, right: str) -> None:
        left_value = value(left)
        right_value = value(right)
        if left_value is not None and right_value is not None and not left_value < right_value:
            failures.append(f"{left}:{left_value:.9g}>={right}:{right_value:.9g}")

    equal("scene_count", gates.scene_count)
    equal("object_fit_count", gates.object_fit_count)
    positive("position_uncertainty_scale")
    maximum("position_uncertainty_scale", gates.position_scale_max)
    positive("velocity_uncertainty_scale")
    maximum("velocity_uncertainty_scale", gates.velocity_scale_max)
    minimum("drag_uncertainty_scale", gates.drag_scale_min)
    maximum("drag_uncertainty_scale", gates.drag_scale_max)
    maximum("current_position_rmse_m", gates.current_position_rmse_m)
    maximum("current_velocity_rmse_mps", gates.current_velocity_rmse_mps)
    # The drag point-estimate gate is explicitly disjunctive.
    drag_rmse = value("current_drag_rmse_per_s")
    log_drag_rmse = value("current_log_drag_rmse")
    if (
        drag_rmse is not None
        and log_drag_rmse is not None
        and drag_rmse > gates.current_drag_rmse_per_s
        and log_drag_rmse > gates.current_log_drag_rmse
    ):
        failures.append("drag_point_accuracy:neither_linear_nor_log_gate_passed")
    maximum("log_drag_gaussian_nll", gates.log_drag_gaussian_nll)
    for key in (
        "position_variance_floor_clamp_count",
        "position_variance_ceiling_clamp_count",
        "velocity_variance_floor_clamp_count",
        "velocity_variance_ceiling_clamp_count",
        "drag_variance_floor_clamp_count",
        "drag_variance_ceiling_clamp_count",
        "identity_switch_count",
        "persistent_id_mismatch_count",
        "association_ambiguous_pair_count",
        "counterfactual_identity_mismatch_count",
        "counterfactual_structure_mismatch_count",
        "drag_grid_boundary_selection_count",
        "invalid_fit_count",
        "public_rollout_output_alias_count",
        "world_from_camera_homogeneous_last_row_gradient_max_abs",
        "optimizer_updates",
        "optimizer_state_entry_count",
        "rng_state_entry_count",
        "learned_parameter_count",
        "learned_parameter_bytes",
    ):
        equal(key, 0.0)
    equal("identity_coverage", gates.identity_coverage)
    equal("association_pair_coverage", 1.0)
    equal("persistent_object_id_min", 0.0)
    equal("persistent_object_id_max", 1.0)
    equal("counterfactual_pair_count", gates.counterfactual_pair_count)
    equal("counterfactual_drag_swap_fraction", 1.0)
    minimum("minimum_drag_excitation_m", gates.minimum_drag_excitation_m)
    minimum("minimum_profile_information", gates.minimum_profile_information)
    maximum("maximum_boundary_mass", gates.maximum_boundary_mass)
    equal("drag_grid_point_count", gates.drag_grid_point_count)
    maximum("semigroup_position_max_abs_m", gates.semigroup_max_abs)
    maximum("semigroup_velocity_max_abs_mps", gates.semigroup_max_abs)
    maximum("analytic_position_agreement_max_abs_m", gates.analytic_agreement_max_abs)
    maximum("analytic_velocity_agreement_max_abs_mps", gates.analytic_agreement_max_abs)
    maximum(
        "future_position_variance_partition_max_abs",
        gates.analytic_agreement_max_abs,
    )
    maximum(
        "future_velocity_variance_partition_max_abs",
        gates.analytic_agreement_max_abs,
    )
    maximum("public_direct_position_max_abs_m", gates.analytic_agreement_max_abs)
    maximum("public_direct_velocity_max_abs_mps", gates.analytic_agreement_max_abs)
    maximum("public_query_time_max_abs_seconds", 1.0e-6)
    equal("ingested_frame_count_min", 16.0)
    equal("ingested_frame_count_max", 16.0)
    equal("state_ingest_count_min", 16.0)
    equal("state_ingest_count_max", 16.0)
    equal("history_sample_count_per_scene_min", 32.0)
    equal("history_sample_count_per_scene_max", 32.0)
    equal("history_valid_count_per_scene_min", 32.0)
    equal("history_valid_count_per_scene_max", 32.0)
    equal("public_predict_calls_per_scene_min", 1.0)
    equal("public_predict_calls_per_scene_max", 1.0)
    equal("direct_velocity_calls_per_scene_min", 1.0)
    equal("direct_velocity_calls_per_scene_max", 1.0)
    equal("direct_velocity_valid_fraction", 1.0)
    maximum("direct_velocity_position_change_max_abs_m", 0.01)
    maximum("direct_fit_position_owner_max_abs_m", 1.0e-7)
    maximum("direct_fit_velocity_owner_max_abs_mps", 1.0e-7)
    maximum("direct_fit_log_drag_owner_max_abs", 1.0e-7)
    equal("module_tensor_buffer_count", 3.0)
    equal("persistent_module_state_key_count", 3.0)
    equal("persistent_module_state_bytes", gates.persistent_module_state_bytes)
    maximum("evaluation_latency_seconds", gates.evaluation_latency_seconds)
    maximum("vjp_latency_seconds", gates.vjp_latency_seconds)
    maximum("process_max_rss_bytes", gates.process_max_rss_bytes)

    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        adaptive_position = f"horizon_{label}_position_rmse_m"
        adaptive_velocity = f"horizon_{label}_velocity_rmse_mps"
        minimum(adaptive_position, 0.0)
        minimum(adaptive_velocity, 0.0)
        maximum(adaptive_position, gates.horizon_position_rmse_m[horizon_index])
        maximum(adaptive_velocity, gates.horizon_velocity_rmse_mps)
        for fixed in ("0.05", "0.185"):
            fixed_position = f"fixed_{fixed}_horizon_{label}_position_rmse_m"
            fixed_velocity = f"fixed_{fixed}_horizon_{label}_velocity_rmse_mps"
            minimum(fixed_position, 0.0)
            minimum(fixed_velocity, 0.0)
            strict_less(adaptive_position, fixed_position)
            strict_less(adaptive_velocity, fixed_velocity)
        for quantity in ("position", "velocity", "log_drag"):
            prefix = f"horizon_{label}_{quantity}"
            minimum(f"{prefix}_marginal_coverage_90", gates.coverage_min)
            maximum(f"{prefix}_marginal_coverage_90", gates.coverage_max)
            minimum(f"{prefix}_rms_z", gates.rms_z_min)
            maximum(f"{prefix}_rms_z", gates.rms_z_max)
        minimum(
            f"horizon_{label}_position_joint_coverage_90",
            gates.position_joint_coverage_min,
        )
        maximum(f"horizon_{label}_position_joint_coverage_90", 1.0)
        minimum(
            f"horizon_{label}_velocity_joint_coverage_90",
            gates.velocity_joint_coverage_min,
        )
        maximum(f"horizon_{label}_velocity_joint_coverage_90", 1.0)
        minimum(
            f"horizon_{label}_log_drag_joint_coverage_90",
            gates.drag_joint_coverage_min,
        )
        maximum(f"horizon_{label}_log_drag_joint_coverage_90", 1.0)
    maximum("horizon_2.00_position_mean_width_90_m", gates.mean_h2_position_width_m)
    maximum("horizon_2.00_velocity_mean_width_90_mps", gates.mean_h2_velocity_width_mps)
    maximum("horizon_2.00_log_drag_mean_width_90", gates.mean_h2_log_drag_width)
    maximum("horizon_2.00_position_max_width_90_m", gates.max_h2_position_width_m)
    maximum("horizon_2.00_velocity_max_width_90_mps", gates.max_h2_velocity_width_mps)
    maximum("horizon_2.00_log_drag_max_width_90", gates.max_h2_log_drag_width)

    vjp_outputs = (
        "current_log_drag",
        "current_log_drag_log_variance",
        "horizon_2.00_position_log_variance",
    )
    for object_index in OBJECT_INDICES:
        for output in vjp_outputs:
            for modality in ("rgb", "depth", "world_from_camera"):
                suffix = f"object_{object_index}/{output}/{modality}"
                minimum(f"gradient_l1/{suffix}", gates.minimum_input_gradient_l1)
                maximum(f"gradient_max_l1/{suffix}", gates.maximum_input_gradient_l1)
                equal(f"gradient_cross_scene_max_l1/{suffix}", 0.0)
                equal(f"gradient_post_history_max_l1/{suffix}", 0.0)
                minimum(f"gradient_min_history_frame_l1/{suffix}", gates.minimum_input_gradient_l1)
                equal(
                    f"gradient_supported_history_frames/{suffix}",
                    gates.vjp_required_history_frames,
                )
    equal("gradient_audit_scene_count", 4.0)
    equal("gradient_audit_unique_scene_fraction", 1.0)

    # Reject physically impossible negative scalars even when an upper gate is
    # the scientifically interesting side.  NLL is deliberately excluded.
    nonnegative = {
        "scene_count",
        "object_fit_count",
        "position_uncertainty_scale",
        "velocity_uncertainty_scale",
        "drag_uncertainty_scale",
        "current_position_rmse_m",
        "current_velocity_rmse_mps",
        "current_drag_rmse_per_s",
        "current_log_drag_rmse",
        "minimum_drag_excitation_m",
        "minimum_profile_information",
        "maximum_boundary_mass",
        "drag_grid_point_count",
        "semigroup_position_max_abs_m",
        "semigroup_velocity_max_abs_mps",
        "analytic_position_agreement_max_abs_m",
        "analytic_velocity_agreement_max_abs_mps",
        "future_position_variance_partition_max_abs",
        "future_velocity_variance_partition_max_abs",
        "public_direct_position_max_abs_m",
        "public_direct_velocity_max_abs_mps",
        "public_query_time_max_abs_seconds",
        "direct_velocity_position_change_max_abs_m",
        "direct_fit_position_owner_max_abs_m",
        "direct_fit_velocity_owner_max_abs_mps",
        "direct_fit_log_drag_owner_max_abs",
        "module_tensor_buffer_count",
        "persistent_module_state_key_count",
        "persistent_module_state_bytes",
        "evaluation_latency_seconds",
        "vjp_latency_seconds",
        "process_max_rss_bytes",
        "horizon_2.00_position_mean_width_90_m",
        "horizon_2.00_velocity_mean_width_90_mps",
        "horizon_2.00_log_drag_mean_width_90",
        "horizon_2.00_position_max_width_90_m",
        "horizon_2.00_velocity_max_width_90_mps",
        "horizon_2.00_log_drag_max_width_90",
    }
    for horizon in HORIZONS_SECONDS:
        label = f"{horizon:.2f}"
        nonnegative.update(
            {
                f"horizon_{label}_position_rmse_m",
                f"horizon_{label}_velocity_rmse_mps",
                *(f"fixed_{fixed}_horizon_{label}_position_rmse_m" for fixed in ("0.05", "0.185")),
                *(
                    f"fixed_{fixed}_horizon_{label}_velocity_rmse_mps"
                    for fixed in ("0.05", "0.185")
                ),
            }
        )
    for key in sorted(nonnegative):
        minimum(key, 0.0)
    return failures, required


def gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    if not isinstance(metrics, Mapping):
        return ["metric_schema:not_a_mapping"]
    failures, required = _gate_surface(metrics, schema_only=False)
    actual = set(metrics)
    if actual != required:
        failures.append(
            "metric_schema:"
            f"missing={sorted(required - actual)!r}:extra={sorted(actual - required)!r}"
        )
    return failures


GATE_METRIC_SCHEMA = tuple(sorted(_gate_surface({}, schema_only=True)[1]))


@dataclass(frozen=True, slots=True)
class SceneSufficientEvidence:
    """The only per-scene values retained by development in memory."""

    split: str
    ordinal: int
    scene_sha256: str
    current_position_truth: Tensor
    current_position_mean: Tensor
    current_position_raw_variance: Tensor
    current_velocity_truth: Tensor
    current_velocity_mean: Tensor
    current_velocity_raw_variance: Tensor
    log_drag_truth: Tensor
    log_drag_mean: Tensor
    log_drag_raw_variance: Tensor
    future_position_truth: Tensor
    future_position_mean: Tensor
    future_position_raw_variance: Tensor
    future_velocity_truth: Tensor
    future_velocity_mean: Tensor
    future_velocity_raw_variance: Tensor
    fixed_position_mean: Tensor
    fixed_velocity_mean: Tensor
    diagnostics: tuple[tuple[str, float], ...]


_EVIDENCE_TENSOR_SHAPES = {
    "current_position_truth": (2, 3),
    "current_position_mean": (2, 3),
    "current_position_raw_variance": (2, 3),
    "current_velocity_truth": (2, 3),
    "current_velocity_mean": (2, 3),
    "current_velocity_raw_variance": (2, 3),
    "log_drag_truth": (2, 1),
    "log_drag_mean": (2, 1),
    "log_drag_raw_variance": (2, 1),
    "future_position_truth": (5, 2, 3),
    "future_position_mean": (5, 2, 3),
    "future_position_raw_variance": (5, 2, 3),
    "future_velocity_truth": (5, 2, 3),
    "future_velocity_mean": (5, 2, 3),
    "future_velocity_raw_variance": (5, 2, 3),
    "fixed_position_mean": (2, 5, 2, 3),
    "fixed_velocity_mean": (2, 5, 2, 3),
}


def _validated_evidence(
    evidence: SceneSufficientEvidence, *, split: str, ordinal: int
) -> SceneSufficientEvidence:
    if type(evidence) is not SceneSufficientEvidence:
        raise TypeError("ordinal evaluator must return exact typed sufficient evidence")
    if evidence.split != split or type(evidence.ordinal) is not int or evidence.ordinal != ordinal:
        raise ValueError("sufficient evidence split/ordinal differs from the manifest row")
    validated_sha256(evidence.scene_sha256, label="scene evidence SHA-256")
    for name, shape in _EVIDENCE_TENSOR_SHAPES.items():
        tensor = getattr(evidence, name)
        if (
            type(tensor) is not Tensor
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
        ):
            raise TypeError(f"sufficient evidence {name} must be an exact CPU float32 tensor")
        if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"sufficient evidence {name} has invalid shape or values")
        if "variance" in name and not bool((tensor > 0.0).all()):
            raise ValueError(f"sufficient evidence {name} must be strictly positive")
    if type(evidence.diagnostics) is not tuple:
        raise TypeError("sufficient evidence diagnostics must be an exact tuple")
    names: set[str] = set()
    for item in evidence.diagnostics:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not float
            or not math.isfinite(item[1])
            or item[0] in names
        ):
            raise ValueError("sufficient evidence diagnostics must be unique finite float pairs")
        names.add(item[0])
    return evidence


def _update_tensor_digest(digest: Any, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))


def _evidence_cache_sha256(cache: Sequence[SceneSufficientEvidence]) -> str:
    digest = hashlib.sha256()
    for evidence in cache:
        digest.update(_canonical_json({"split": evidence.split, "ordinal": evidence.ordinal}))
        digest.update(evidence.scene_sha256.encode("ascii"))
        for name in _EVIDENCE_TENSOR_SHAPES:
            _update_tensor_digest(digest, getattr(evidence, name))
        digest.update(_canonical_json(list(evidence.diagnostics)))
    return digest.hexdigest()


def _float32_bits(value: Tensor) -> int:
    if type(value) is not Tensor or value.dtype != torch.float32 or value.ndim != 0:
        raise TypeError("float32 bit conversion requires an exact scalar tensor")
    return int(value.view(torch.int32).item()) & 0xFFFFFFFF


def _float32_from_bits(bits: int) -> Tensor:
    if type(bits) is not int or not 0 < bits < 0x7F800000:
        raise ValueError("float32 scale bits must encode a finite positive value")
    signed = bits if bits < 0x80000000 else bits - 0x100000000
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32)


def _ceil_positive_float32(target: float) -> Tensor:
    if type(target) is not float or not math.isfinite(target) or target <= 0.0:
        raise ValueError("calibration target must be an exact finite positive float")
    candidate = torch.tensor(target, dtype=torch.float32)
    if float(candidate) < target:
        candidate = torch.nextafter(candidate, torch.tensor(math.inf, dtype=torch.float32))
    if not bool(torch.isfinite(candidate)) or not bool(candidate > 0.0):
        raise ValueError("calibration target cannot be represented by finite positive float32")
    return candidate


def _deployed_scene_coverage(
    errors: Sequence[Tensor], raw_variances: Sequence[Tensor], scale: Tensor
) -> int:
    """Count scenes covered through the exact cached float32 deployment path."""

    if len(errors) != 64 or len(raw_variances) != 64:
        raise ValueError("calibration coverage requires exactly 64 cached scenes")
    scale_squared = scale.square()
    covered = 0
    z = torch.tensor(CALIBRATION_Z, dtype=torch.float32)
    for error, variance in zip(errors, raw_variances, strict=True):
        if error.dtype != torch.float32 or variance.dtype != torch.float32:
            raise TypeError("deployed calibration path requires float32 cached evidence")
        radius = z * torch.sqrt(variance * scale_squared)
        covered += int(bool((error.abs() <= radius).all()))
    return covered


@dataclass(frozen=True, slots=True)
class _ScaleCalibration:
    scale: Tensor
    target: float
    initial_bits: int
    deployed_bits: int
    additional_ulps: int
    coverage: int
    predecessor_bits: int
    predecessor_coverage: int
    predecessor_admissible: bool


def _calibrate_one_scale(
    errors: Sequence[Tensor],
    raw_variances: Sequence[Tensor],
    *,
    lower_bound: float,
) -> _ScaleCalibration:
    if len(errors) != 64 or len(raw_variances) != 64:
        raise ValueError("one calibration component requires exactly 64 scenes")
    scores: list[float] = []
    for error, variance in zip(errors, raw_variances, strict=True):
        if (
            type(error) is not Tensor
            or type(variance) is not Tensor
            or error.dtype != torch.float32
            or variance.dtype != torch.float32
            or error.shape != variance.shape
            or not bool(torch.isfinite(error).all())
            or not bool(torch.isfinite(variance).all())
            or not bool((variance > 0.0).all())
        ):
            raise ValueError("calibration component received malformed cached evidence")
        score = float((error.to(torch.float64).abs() / variance.to(torch.float64).sqrt()).max())
        if not math.isfinite(score):
            raise FloatingPointError("calibration score is nonfinite")
        scores.append(score)
    rank_score = sorted(scores)[CALIBRATION_RANK - 1]
    target = max(float(lower_bound), rank_score / CALIBRATION_Z)
    scale = _ceil_positive_float32(float(target))
    initial_bits = _float32_bits(scale)
    coverage = _deployed_scene_coverage(errors, raw_variances, scale)
    additional_ulps = 0
    while coverage < CALIBRATION_RANK:
        scale = torch.nextafter(scale, torch.tensor(math.inf, dtype=torch.float32))
        additional_ulps += 1
        if additional_ulps > 1_000_000 or not bool(torch.isfinite(scale)):
            raise RuntimeError("float32 deployment calibration failed to reach rank coverage")
        coverage = _deployed_scene_coverage(errors, raw_variances, scale)
    predecessor = torch.nextafter(scale, torch.tensor(0.0, dtype=torch.float32))
    predecessor_coverage = _deployed_scene_coverage(errors, raw_variances, predecessor)
    predecessor_admissible = float(predecessor) >= lower_bound
    if predecessor_admissible and predecessor_coverage >= CALIBRATION_RANK:
        raise RuntimeError("deployed scale is not the smallest float32 meeting rank-59 coverage")
    return _ScaleCalibration(
        scale=scale,
        target=float(target),
        initial_bits=initial_bits,
        deployed_bits=_float32_bits(scale),
        additional_ulps=additional_ulps,
        coverage=coverage,
        predecessor_bits=_float32_bits(predecessor),
        predecessor_coverage=predecessor_coverage,
        predecessor_admissible=predecessor_admissible,
    )


def _calibrate_development_cache(
    cache: Sequence[SceneSufficientEvidence],
) -> tuple[_ScaleCalibration, _ScaleCalibration, _ScaleCalibration]:
    if type(cache) not in {list, tuple} or len(cache) != 64:
        raise ValueError("development calibration requires exactly 64 cached evidence rows")
    if [evidence.ordinal for evidence in cache] != list(ORDINALS):
        raise ValueError("development cache must retain exact ordinal order 0..63")
    position = _calibrate_one_scale(
        [row.current_position_mean - row.current_position_truth for row in cache],
        [row.current_position_raw_variance for row in cache],
        lower_bound=0.0,
    )
    velocity = _calibrate_one_scale(
        [row.current_velocity_mean - row.current_velocity_truth for row in cache],
        [row.current_velocity_raw_variance for row in cache],
        lower_bound=0.0,
    )
    drag = _calibrate_one_scale(
        [row.log_drag_mean - row.log_drag_truth for row in cache],
        [row.log_drag_raw_variance for row in cache],
        lower_bound=1.0,
    )
    return position, velocity, drag


def bridge_protocol() -> dict[str, Any]:
    """Return the canonical self-hashed seedless qualification contract."""

    protocol: dict[str, Any] = {
        "name": "rgbd_two_visible_orbital_camera_identifiable_drag_v1",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
        "terminal_after_attempt": True,
        "optimizer": None,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "scene_certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        "manifests": {
            split: {
                "rows": list(_manifest_rows(split)),
                "sha256": MANIFEST_SHA256[split],
            }
            for split in SPLITS
        },
        "scene_family": {
            "address": "exact conceptual split plus ordinal",
            "ordinals": list(ORDINALS),
            "scenes_per_split": 64,
            "objects_per_scene": 2,
            "camera_strata": CAMERA_STRATA,
            "fully_visible": True,
            "image_separated": True,
            "non_contact": True,
            "known_orbital_extrinsics": True,
            "unknown_distinct_per_object_drag": True,
            "gravity": [0.0, 0.0, 0.0],
        },
        "runtime": {
            "observation_factory": "make_rgbd_packet",
            "runtime": "OnlineWorldModel",
            "ingested_frame_indices": list(HISTORY_FRAME_INDICES),
            "anchor_frame_index": ANCHOR_FRAME_INDEX,
            "horizons_seconds": list(HORIZONS_SECONDS),
            "target_frame_indices": list(TARGET_FRAME_INDICES),
            "learned_parameters": 0,
            "persistent_float32_scalar_buffers": 3,
            "persistent_module_state_bytes": 12,
            "calibration_buffers": [
                "position_uncertainty_scale",
                "velocity_uncertainty_scale",
                "drag_uncertainty_scale",
            ],
        },
        "development_calibration": {
            "confidence": CALIBRATION_CONFIDENCE,
            "normal_z": CALIBRATION_Z,
            "scene_max_score": True,
            "scene_count": 64,
            "rank": CALIBRATION_RANK,
            "rank_definition": "sorted ascending one-indexed rank 59",
            "position_axes_per_scene": 6,
            "velocity_axes_per_scene": 6,
            "drag_values_per_scene": 2,
            "position_velocity_deflation_allowed": True,
            "drag_scale_lower_bound": 1.0,
            "float32_rule": (
                "smallest finite positive float32 at or above the float64 target, then "
                "nextafter toward positive infinity until the cached deployed float32 path "
                "covers at least 59 of 64 scenes"
            ),
            "predecessor_minimality_required": True,
            "atomic_setter_calls": 1,
            "evidence_replay_count": 0,
            "development_vjp_stage": "scale_one_precalibration_same_single_pass_cache",
            "protected_vjp_stage": "reviewed_deployed_three_buffer_state",
            "public_regression_fixture": PUBLIC_CALIBRATION_REGRESSION,
        },
        "access": {
            "development": {
                "fixed_exclusive_durable_ledger": True,
                "receipt_before_materialisation": True,
                "single_manifest_pass": True,
            },
            "protected": {
                "order": ["selector", "confirmation", "final_test"],
                "receipt_before_materialisation": True,
                "later_split_unopened_after_any_failure": True,
                "external_review_sha256s": [
                    "development checkpoint",
                    "development report",
                    "development ledger",
                ],
                "reviewed_checkpoint_strictly_loaded_for_every_batch": True,
            },
        },
        "gates": asdict(DEFAULT_GATES),
        "gate_metric_schema": list(GATE_METRIC_SCHEMA),
        "execution": {
            "device": "cpu_float32",
            "torch_intraop_threads": 1,
            "evaluation_seconds_max": DEFAULT_GATES.evaluation_latency_seconds,
            "vjp_seconds_max": DEFAULT_GATES.vjp_latency_seconds,
            "rss_bytes_max": DEFAULT_GATES.process_max_rss_bytes,
        },
        "scientific_limitations": [
            "The deterministic designed family is not iid and the rank statistic is not a distribution-free conformal guarantee.",
            "Calibration coverage is finite-sample designed-family evidence only.",
            "The rung contains exactly two separated fixed-radius spheres, zero gravity, known camera extrinsics, and no contacts or interventions.",
            "Fixed-drag controls use the same fitted public position/velocity anchor; simulator truth is never a control anchor.",
            "Direct-anchor multi-query uncertainty is tested; sequential external re-anchoring is outside this rung.",
            "Development VJP evidence is captured on the mandatory scale-one single pass and is explicitly bound to the raw model-state hash; protected VJP evidence uses the reviewed deployed state.",
            "Public feasibility informed broad thresholds but is not development or protected evidence.",
        ],
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def _exact_equal(actual: Any, expected: Any, *, label: str) -> None:
    """Type-strict recursive equality (``False`` is never accepted as zero)."""

    if type(actual) is not type(expected):
        raise ValueError(
            f"{label} has type {type(actual).__name__}, expected {type(expected).__name__}"
        )
    if isinstance(expected, tuple):
        if len(actual) != len(expected):
            raise ValueError(f"{label} has the wrong length")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _exact_equal(left, right, label=f"{label}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{label}={actual!r}, expected {expected!r}")


def assert_rgbd_identifiable_drag_config(config: OrpheusConfig) -> None:
    """Reject every semantic change to the dedicated resolved profile.

    Common configuration may contain fields used by unrelated tooling, but no
    such field selects or transforms a governed manifest row.
    """

    if type(config) is not OrpheusConfig:
        raise TypeError("identifiable-drag execution requires exact OrpheusConfig")
    simulator = {
        "image_size": (64, 64),
        "frame_rate": 20,
        "physics_rate": 120,
        "sequence_frames": 56,
        "min_objects": 2,
        "max_objects": 2,
        "gravity": (0.0, 0.0, 0.0),
        "radius_range": (0.21, 0.21),
        "mass_range": (1.0, 1.0),
        "restitution_range": (0.7, 0.7),
        "drag_range": (0.045, 0.325),
        "friction_range": (0.2, 0.2),
        "initial_speed_range": (0.05, 0.071),
        "camera_motion": "orbit",
        "known_camera_pose": True,
        "render_noise_std": 0.0,
        "ensure_collision": False,
        "external_impulse_probability": 0.0,
        "scenario_mixture": ("baseline",),
    }
    for name, expected in simulator.items():
        _exact_equal(getattr(config.simulator, name), expected, label=f"simulator.{name}")
    if config.project.deterministic is not True:
        raise ValueError("identifiable-drag execution requires deterministic common tooling")
    if (
        config.device.preference != "cpu"
        or config.device.cuda_amp is not False
        or config.device.compile is not False
    ):
        raise ValueError("identifiable-drag execution requires CPU float32 without compile")
    if config.model.max_objects != 2 or config.model.state.appearance_dim != 3:
        raise ValueError("identifiable-drag execution requires exactly two three-colour slots")
    if config.model.rgb.enabled is not False or config.model.rgbd.enabled is not True:
        raise ValueError("identifiable-drag execution requires only composite RGB-D")
    rgbd = {
        "global_every_steps": 1,
        "proposal_count": 2,
        "world_radius": 0.21,
        "linear_drag": 0.185,
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
        "measurement_position_variance": 0.000064,
        "temporal_history_size": 16,
        "temporal_min_samples": 16,
        "temporal_min_dt": 0.001,
        "temporal_velocity_variance_floor": 0.000001,
        "temporal_velocity_variance_ceiling": 0.01,
        "fit_conditioning_limit": 100.0,
        "temporal_drag_estimation_enabled": True,
        "temporal_drag_minimum": 0.01,
        "temporal_drag_maximum": 0.36,
        "temporal_drag_grid_points": 257,
        "temporal_drag_noise_floor_m": 0.00002,
        "temporal_drag_minimum_excitation_m": 0.015,
        "temporal_drag_minimum_profile_information": 1.0,
        "temporal_drag_maximum_boundary_mass": 0.01,
        "temporal_drag_log_parameter_variance_floor": 0.0001,
        "temporal_drag_log_parameter_variance_ceiling": 0.25,
    }
    for name, expected in rgbd.items():
        _exact_equal(getattr(config.model.rgbd, name), expected, label=f"model.rgbd.{name}")
    association = {
        "geometry_weight": 1.0,
        "appearance_weight": 0.25,
        "existence_weight": 0.0,
        "mahalanobis_gate": 100.0,
        "maximum_cost": 100.0,
        "ambiguity_margin": 0.02,
        "minimum_measurement_confidence": 0.5,
    }
    for name, expected in association.items():
        _exact_equal(
            getattr(config.model.association, name), expected, label=f"model.association.{name}"
        )
    if (
        config.model.dynamics.analytic_free_motion_only is not True
        or config.model.dynamics.attention_residual_enabled is not False
        or config.model.dynamics.max_substep != 1.0 / 120.0
    ):
        raise ValueError("identifiable-drag execution requires only analytic free motion")
    if (
        config.model.filter.min_log_variance != -30.0
        or config.model.filter.max_log_variance != 8.0
        or config.model.filter.enable_learned_corrector is not False
        or config.model.filter.learned_residual_scale != 0.0
        or config.model.filter.direct_metric_position_update is not True
        or config.model.filter.innovation_anchored_correction is not True
    ):
        raise ValueError("identifiable-drag filter semantics differ from the frozen rung")
    if config.model.identification.enabled is not False:
        raise ValueError("identifiable-drag execution forbids recurrent learned identification")
    if (
        config.runtime.modality != "rgbd"
        or tuple(config.runtime.modality_order) != ("debug_oracle", "rgbd")
        or config.runtime.enable_debug_oracle is not False
        or config.runtime.hypothesis_pool_enabled is not False
        or config.runtime.strict_timestamps is not True
    ):
        raise ValueError("identifiable-drag runtime modality semantics differ")
    if (
        config.training.batch_size != 4
        or config.training.steps != 1
        or config.training.rgb_pretrain_steps != 0
        or config.training.validation_episodes != 64
        or config.evaluation.episodes != 64
    ):
        raise ValueError("identifiable-drag common training/evaluation shape differs")
    if tuple(config.evaluation.horizons_seconds) != HORIZONS_SECONDS:
        raise ValueError("identifiable-drag horizons differ from the protocol")
    if TARGET_FRAME_INDICES[-1] != FRAME_COUNT - 1:
        raise RuntimeError("identifiable-drag final target must be frame 55")


def _frozen_config_path() -> Path:
    return REPOSITORY_ROOT / "configs" / "rgbd_two_visible_orbital_camera_identifiable_drag_v1.yaml"


def require_frozen_config(path: str | Path) -> OrpheusConfig:
    if type(path) is not _NATIVE_PATH_TYPE:
        raise TypeError("identifiable-drag frozen config requires exact native Path")
    source = path
    if source != _frozen_config_path():
        raise ValueError("identifiable-drag execution requires the canonical frozen config path")
    contents = stable_read_bytes(source, label="identifiable-drag frozen config")
    digest = sha256_bytes(contents)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            f"identifiable-drag frozen config hash differs: expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    config = load_config(source)
    assert_rgbd_identifiable_drag_config(config)
    return config


def _require_config_matches_frozen_path(config: OrpheusConfig, path: Path) -> None:
    if type(path) is not _NATIVE_PATH_TYPE or path != _frozen_config_path():
        raise ValueError("identifiable-drag config binding path is not canonical")
    before_metadata = _require_single_link_regular(
        path, label="identifiable-drag frozen config binding"
    )
    before = stable_read_bytes(path, label="identifiable-drag config binding")
    if sha256_bytes(before) != FROZEN_CONFIG_SHA256:
        raise ValueError("identifiable-drag config binding bytes differ")
    parsed = load_config(path)
    after = stable_read_bytes(path, label="identifiable-drag config binding recheck")
    after_metadata = _require_single_link_regular(
        path, label="identifiable-drag frozen config binding recheck"
    )
    if after != before or (
        before_metadata.st_dev,
        before_metadata.st_ino,
        before_metadata.st_size,
    ) != (after_metadata.st_dev, after_metadata.st_ino, after_metadata.st_size):
        raise RuntimeError("identifiable-drag config bytes changed while parsing")
    if canonical_sha256(config.to_dict()) != canonical_sha256(parsed.to_dict()):
        raise ValueError("executed config object differs from exact frozen config bytes")
    assert_rgbd_identifiable_drag_config(parsed)


_SCALE_STATE_LEAVES = (
    "position_uncertainty_scale",
    "velocity_uncertainty_scale",
    "drag_uncertainty_scale",
)


def _scale_state(model: OnlineWorldModel) -> dict[str, Tensor]:
    state = model.state_dict()
    expected = {f"observation_modules.rgbd.{leaf}" for leaf in _SCALE_STATE_LEAVES}
    if set(state) != expected:
        raise RuntimeError(
            "identifiable-drag model state must contain exactly the reviewed three-scale group"
        )
    result: dict[str, Tensor] = {}
    for name in sorted(expected):
        value = state[name]
        if value.dtype != torch.float32 or value.ndim != 0:
            raise RuntimeError(f"identifiable-drag state {name!r} is not scalar float32")
        squared = value.square()
        if (
            not bool(torch.isfinite(value))
            or not bool(value > 0.0)
            or not bool(torch.isfinite(squared))
            or not bool(squared > 0.0)
        ):
            raise RuntimeError(f"identifiable-drag state {name!r} is not safely positive")
        result[name] = value
    return result


def _model_state_sha256(model: OnlineWorldModel) -> str:
    digest = hashlib.sha256()
    for name, value in _scale_state(model).items():
        digest.update(name.encode("utf-8"))
        _update_tensor_digest(digest, value)
    return digest.hexdigest()


def new_public_model(config: OrpheusConfig) -> OnlineWorldModel:
    assert_rgbd_identifiable_drag_config(config)
    model = OnlineWorldModel.from_config(config, device="cpu")
    learned = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if learned:
        raise RuntimeError("identifiable-drag public runtime must remain parameter-free")
    state = _scale_state(model)
    if (
        len(tuple(model.buffers())) != 3
        or sum(value.numel() * value.element_size() for value in state.values()) != 12
    ):
        raise RuntimeError(
            "identifiable-drag public runtime must own exactly three scalar buffers/12B"
        )
    if any(float(value) != 1.0 for value in state.values()):
        raise RuntimeError("fresh identifiable-drag runtime must begin at three scale-one buffers")
    return model


_CRITICAL_SOURCE_PATHS = {
    "analytic_drag": Path("world_model/identification/analytic_drag.py"),
    "analytic_free_motion": Path("world_model/dynamics/analytic_free_motion.py"),
    "free_motion_fit": Path("world_model/dynamics/free_motion_fit.py"),
    "measurements": Path("world_model/observations/measurements.py"),
    "rgbd_temporal": Path("world_model/observations/rgbd/temporal.py"),
    "rgbd_observation": Path("world_model/observations/rgbd/module.py"),
    "filter_correction": Path("world_model/filtering/correction.py"),
    "online_world_model": Path("world_model/runtime/online_world_model.py"),
    "checkpointing": Path("world_model/training/checkpointing.py"),
    "scene": Path("world_model/training/rgbd_identifiable_drag_scene.py"),
}
_MODULE_RELATIVE_PATH = Path("world_model/training/rgbd_identifiable_drag_qualification.py")


def _validate_repository_identity() -> None:
    """Bind the imported source file separately from lexical artifact paths."""

    imported = Path(__file__)
    expected = REPOSITORY_ROOT / _MODULE_RELATIVE_PATH
    for ancestor in (REPOSITORY_ROOT, *REPOSITORY_ROOT.parents):
        metadata = os.lstat(ancestor)
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError("qualification repository ancestry cannot contain symlinks")
    imported_metadata = os.stat(imported, follow_symlinks=True)
    expected_metadata = os.lstat(expected)
    if (
        stat.S_ISLNK(expected_metadata.st_mode)
        or not stat.S_ISREG(expected_metadata.st_mode)
        or expected_metadata.st_nlink != 1
    ):
        raise PermissionError("identifiable-drag qualification source must be a real file")
    if (imported_metadata.st_dev, imported_metadata.st_ino) != (
        expected_metadata.st_dev,
        expected_metadata.st_ino,
    ):
        raise PermissionError("imported qualification source differs from repository source")
    # Resolution is permitted only for source/repository identity.  Artifact
    # containment below remains purely lexical and never normalises a target.
    if imported.resolve(strict=True) != expected.resolve(strict=True):
        raise PermissionError("qualification source identity is ambiguous")


def _validate_frozen_critical_sources() -> None:
    if set(FROZEN_SOURCE_SHA256) != set(_CRITICAL_SOURCE_PATHS):
        raise RuntimeError("critical-source binding schema differs from the protocol")
    for name, relative in _CRITICAL_SOURCE_PATHS.items():
        expected = validated_sha256(FROZEN_SOURCE_SHA256[name], label=f"frozen {name} source")
        path = REPOSITORY_ROOT / relative
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise PermissionError(f"critical source {name!r} must be a single-link regular file")
        contents = stable_read_bytes(path, label=f"critical source {name}")
        if sha256_bytes(contents) != expected:
            raise PermissionError(f"critical source {name!r} differs from its frozen hash")


def capture_published_source(root: Path) -> dict[str, Any]:
    """Capture clean HEAD/upstream equality without network access."""

    try:
        upstream_ref = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        upstream_commit = subprocess.run(
            ["git", "rev-parse", "@{upstream}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        counts = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        if len(counts) != 2:
            raise RuntimeError("git returned malformed ahead/behind counts")
        ahead, behind = (int(value) for value in counts)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
        return {
            "upstream_ref": None,
            "head_commit": None,
            "upstream_commit": None,
            "ahead": None,
            "behind": None,
        }
    return {
        "upstream_ref": upstream_ref,
        "head_commit": head_commit,
        "upstream_commit": upstream_commit,
        "ahead": ahead,
        "behind": behind,
    }


PUBLICATION_PROVENANCE_SCHEMA = frozenset(
    {"upstream_ref", "head_commit", "upstream_commit", "ahead", "behind"}
)


def _validated_published_source(
    value: Mapping[str, Any], *, source: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(PUBLICATION_PROVENANCE_SCHEMA):
        raise ValueError(f"{label} publication provenance has the wrong exact schema")
    result = dict(value)
    if (
        type(result["upstream_ref"]) is not str
        or not result["upstream_ref"]
        or result["upstream_ref"] == "HEAD"
    ):
        raise ValueError(f"{label} requires a configured branch upstream")
    for name in ("head_commit", "upstream_commit"):
        commit = result[name]
        if type(commit) is not str or len(commit) != 40:
            raise ValueError(f"{label} {name} must be an exact Git SHA")
        try:
            int(commit, 16)
        except ValueError as error:
            raise ValueError(f"{label} {name} must be hexadecimal") from error
    if (
        result["head_commit"] != source.get("commit")
        or result["upstream_commit"] != result["head_commit"]
    ):
        raise ValueError(f"{label} requires clean HEAD equal to published upstream")
    if type(result["ahead"]) is not int or type(result["behind"]) is not int:
        raise TypeError(f"{label} ahead/behind counts must be exact integers")
    if result["ahead"] != 0 or result["behind"] != 0:
        raise ValueError(f"{label} requires zero commits ahead and behind")
    return result


def _canonical_run_directory() -> Path:
    return REPOSITORY_ROOT / RUN_RELATIVE_PATH


def canonical_development_report_path() -> Path:
    return _canonical_run_directory() / DEVELOPMENT_REPORT_NAME


def canonical_checkpoint_path() -> Path:
    return _canonical_run_directory() / CHECKPOINT_NAME


def canonical_qualification_report_path() -> Path:
    return _canonical_run_directory() / QUALIFICATION_REPORT_NAME


def development_ledger_path() -> Path:
    return _canonical_run_directory() / DEVELOPMENT_LEDGER_NAME


def qualification_ledger_path() -> Path:
    return _canonical_run_directory() / QUALIFICATION_LEDGER_NAME


def _require_canonical_path(actual: Path, expected: Path, *, label: str) -> None:
    if type(actual) is not _NATIVE_PATH_TYPE or type(expected) is not _NATIVE_PATH_TYPE:
        raise TypeError(f"{label} must use exact native Path values")
    if actual != expected:
        raise ValueError(f"{label} must use canonical fixed path {expected}")
    try:
        relative = actual.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must remain lexically inside the repository") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} has a non-canonical lexical path")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_nonlink_directory(path: Path, *, label: str) -> os.stat_result:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"{label} must be a real directory, not a link")
    return metadata


def _require_single_link_regular(path: Path, *, label: str) -> os.stat_result:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise PermissionError(f"{label} must be a single-link regular file")
    return metadata


def _validate_run_tree(expected_names: frozenset[str], *, stage: str) -> None:
    if type(expected_names) is not frozenset or not expected_names <= QUALIFICATION_ARTIFACT_NAMES:
        raise ValueError(f"{stage} requested an unknown artifact inventory")
    runs_root = REPOSITORY_ROOT / "runs"
    run_directory = _canonical_run_directory()
    if _lexists(runs_root):
        _require_nonlink_directory(runs_root, label=f"{stage} runs root")
    elif expected_names:
        raise FileNotFoundError(f"{stage} requires the canonical runs root")
    else:
        return
    if _lexists(run_directory):
        _require_nonlink_directory(run_directory, label=f"{stage} run directory")
    elif expected_names:
        raise FileNotFoundError(f"{stage} requires the canonical run directory")
    else:
        return
    with os.scandir(run_directory) as entries:
        materialised = list(entries)
    names = {entry.name for entry in materialised}
    if names != set(expected_names):
        raise PermissionError(
            f"{stage} inventory differs; missing={sorted(expected_names - names)}, "
            f"unexpected={sorted(names - expected_names)}"
        )
    for entry in materialised:
        if entry.is_symlink():
            raise PermissionError(f"{stage} artifact cannot be a symlink: {entry.name}")
        _require_single_link_regular(run_directory / entry.name, label=f"{stage} {entry.name}")


def _atomic_temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _validate_distinct_canonical_paths(
    paths: Mapping[str, Path], *, atomic_writers: Sequence[str]
) -> None:
    if type(paths) is not dict:
        raise TypeError("artifact path map must be an exact dict")
    expanded: dict[str, Path] = {}
    for name, path in paths.items():
        if type(name) is not str or type(path) is not _NATIVE_PATH_TYPE:
            raise TypeError("artifact paths require exact names and native Paths")
        expanded[name] = path
    for name in atomic_writers:
        if name not in expanded:
            raise ValueError(f"unknown atomic artifact {name!r}")
        expanded[f"{name}_atomic_temporary"] = _atomic_temporary(expanded[name])
    lexical: dict[Path, list[str]] = {}
    for name, path in expanded.items():
        lexical.setdefault(path, []).append(name)
    collisions = [names for names in lexical.values() if len(names) > 1]
    if collisions:
        raise ValueError(f"artifact paths lexically alias: {collisions!r}")
    identities: dict[tuple[int, int], str] = {}
    for name, path in expanded.items():
        if not _lexists(path):
            continue
        metadata = os.lstat(path)
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in identities:
            raise ValueError(f"artifact paths hard-link alias: {identities[identity]}, {name}")
        identities[identity] = name


def _ensure_canonical_run_directory() -> None:
    runs_root = REPOSITORY_ROOT / "runs"
    run_directory = _canonical_run_directory()
    if not _lexists(runs_root):
        os.mkdir(runs_root, 0o700)
        _fsync_parent(runs_root)
    _require_nonlink_directory(runs_root, label="identifiable-drag runs root")
    if not _lexists(run_directory):
        os.mkdir(run_directory, 0o700)
        _fsync_parent(run_directory)
    _require_nonlink_directory(run_directory, label="identifiable-drag run directory")


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_descriptor(descriptor: int, contents: bytes) -> None:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_artifact_paths() -> frozenset[Path]:
    return frozenset(
        {
            canonical_development_report_path(),
            canonical_checkpoint_path(),
            development_ledger_path(),
            canonical_qualification_report_path(),
            qualification_ledger_path(),
        }
    )


def _durable_create(path: Path, contents: bytes, *, mode: int = 0o600) -> None:
    if type(path) is not _NATIVE_PATH_TYPE or path not in _canonical_artifact_paths():
        raise ValueError("identifiable-drag artifacts have no arbitrary write path")
    _ensure_canonical_run_directory()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    _write_descriptor(descriptor, contents)
    _fsync_parent(path)


class _PublishedReplacementError(RuntimeError):
    """A replacement reached the namespace but a later durability check failed."""

    def __init__(self, path: Path, metadata: os.stat_result, cause: BaseException) -> None:
        super().__init__(f"published replacement requires fail-closed reconciliation: {path}")
        self.path = path
        self.metadata = metadata
        self.cause = cause


def _reconcile_published_replacement(
    path: Path,
    contents: bytes,
    expected_metadata: os.stat_result,
) -> os.stat_result | None:
    """Independently prove exact target bytes after a post-replace exception."""

    descriptor: int | None = None
    result: os.stat_result | None = None
    try:
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(contents)
            or (metadata.st_dev, metadata.st_ino)
            != (expected_metadata.st_dev, expected_metadata.st_ino)
        ):
            return None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
        ):
            return None
        chunks: list[bytes] = []
        remaining = len(contents) + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if b"".join(chunks) != contents:
            return None
        _fsync_parent(path)
        result = metadata
    except BaseException:
        result = None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException:
                result = None
    return result


def _durable_replace(path: Path, contents: bytes, *, mode: int = 0o600) -> os.stat_result:
    if type(path) is not _NATIVE_PATH_TYPE or path not in _canonical_artifact_paths():
        raise ValueError("identifiable-drag artifacts have no arbitrary replacement path")
    _ensure_canonical_run_directory()
    _require_single_link_regular(path, label="identifiable-drag replacement target")
    temporary = _atomic_temporary(path)
    if _lexists(temporary):
        raise FileExistsError(f"atomic temporary must be fresh: {temporary}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        _write_descriptor(descriptor, contents)
        metadata = _require_single_link_regular(
            temporary, label="identifiable-drag replacement temporary"
        )
        os.replace(temporary, path)
        try:
            _fsync_parent(path)
            published_metadata = _require_single_link_regular(
                path, label="identifiable-drag replacement publication"
            )
            published = stable_read_bytes(path, label="identifiable-drag replacement publication")
            if published != contents or (
                published_metadata.st_dev,
                published_metadata.st_ino,
            ) != (metadata.st_dev, metadata.st_ino):
                raise RuntimeError("durable replacement publication differs from intended bytes")
            return published_metadata
        except BaseException as error:
            reconciled = _reconcile_published_replacement(path, contents, metadata)
            if reconciled is not None:
                return reconciled
            raise _PublishedReplacementError(path, metadata, error) from error
    except BaseException:
        # A leftover temporary is terminal evidence of an ambiguous write.
        raise


def _report_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(report), allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _write_report_fresh(path: Path, report: Mapping[str, Any]) -> None:
    if type(path) is not _NATIVE_PATH_TYPE or path not in {
        canonical_development_report_path(),
        canonical_qualification_report_path(),
    }:
        raise ValueError("report path is not one of the two fixed artifacts")
    _durable_create(path, _report_bytes(report))


def _persist_failed_report(path: Path, report: Mapping[str, Any], *, label: str) -> str | None:
    """Return a digest only when the exact intended failed bytes are durable."""

    if report.get("passed") is not False:
        raise ValueError("failed-report persistence requires passed=false")
    stopped_after = report.get("stopped_after")
    if type(stopped_after) is not str or not stopped_after:
        raise ValueError("failed-report persistence requires durable stopped_after")
    intended = _report_bytes(report)
    try:
        if _lexists(path):
            _require_single_link_regular(path, label=f"{label} existing report")
            _durable_replace(path, intended)
        else:
            _write_report_fresh(path, report)
    except BaseException:
        pass
    try:
        if not _lexists(path):
            return None
        _require_single_link_regular(path, label=f"{label} failed report")
        persisted = stable_read_bytes(path, label=f"{label} failed report")
        if persisted != intended:
            return None
        return sha256_bytes(persisted)
    except BaseException:
        return None


SOURCE_PROVENANCE_SCHEMA = frozenset(
    {"commit", "dirty", "worktree_fingerprint", "runtime_source_fingerprint"}
)
RUNTIME_API_SCHEMA = frozenset(
    {"packet_factory", "ingest_frames", "rollout_method", "horizons_seconds"}
)
SPLIT_RESULT_SCHEMA = frozenset(
    {
        "split",
        "manifest",
        "manifest_sha256",
        "metrics",
        "failures",
        "passed",
        "optimizer_updates",
        "runtime_api",
        "scene_constructor",
        "model_state_sha256",
    }
)
SCALE_CALIBRATION_SCHEMA = frozenset(
    {
        "target_float64",
        "initial_float32_bits",
        "deployed_float32_bits",
        "additional_ulps",
        "coverage_count",
        "predecessor_float32_bits",
        "predecessor_coverage_count",
        "predecessor_admissible",
    }
)
CALIBRATION_SCHEMA = frozenset(
    {
        "method",
        "confidence",
        "normal_z",
        "rank",
        "scene_count",
        "evidence_ingest_count",
        "evidence_replay_count",
        "atomic_setter_calls",
        "evidence_cache_sha256",
        "raw_model_state_sha256",
        "calibrated_model_state_sha256",
        "gradient_audit_model_state_sha256",
        "position",
        "velocity",
        "drag",
        "variance_floor_clamp_count",
        "variance_ceiling_clamp_count",
    }
)
CHECKPOINT_SCHEMA = frozenset(
    {
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
)
CHECKPOINT_METRICS_SCHEMA = frozenset(
    {
        "artifact_kind",
        "optimizer_updates",
        "optimizer_state_entry_count",
        "rng_state_entry_count",
        "model_state_sha256",
        "protocol",
        "publication_provenance",
        "calibration",
        "development",
    }
)
DEVELOPMENT_REPORT_SCHEMA = frozenset(
    {
        "artifact_kind",
        "protocol",
        "source_provenance",
        "publication_provenance",
        "config_sha256",
        "critical_source_sha256",
        "scene_family_certificate",
        "development_ledger",
        "optimizer_updates",
        "protected_data_materialized",
        "development",
        "calibration",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_model_state_sha256",
        "passed",
        "review_ready",
        "stopped_after",
    }
)
QUALIFICATION_REPORT_SCHEMA = frozenset(
    {
        "artifact_kind",
        "protocol",
        "source_provenance",
        "publication_provenance",
        "config_sha256",
        "critical_source_sha256",
        "scene_family_certificate",
        "qualification_ledger",
        "reviewed_checkpoint_sha256",
        "reviewed_development_report_sha256",
        "reviewed_development_ledger_sha256",
        "model_state_sha256",
        "optimizer_updates",
        "development",
        "calibration",
        "selector",
        "confirmation",
        "final_test",
        "protected_data_materialized",
        "passed",
        "stopped_after",
    }
)
DEVELOPMENT_ERROR_REPORT_SCHEMA = frozenset(
    {
        "artifact_kind",
        "protocol",
        "source_provenance",
        "publication_provenance",
        "config_sha256",
        "critical_source_sha256",
        "scene_family_certificate",
        "development_ledger",
        "optimizer_updates",
        "protected_data_materialized",
        "development",
        "calibration",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_model_state_sha256",
        "passed",
        "review_ready",
        "stopped_after",
        "error",
    }
)
QUALIFICATION_ERROR_REPORT_SCHEMA = frozenset({*QUALIFICATION_REPORT_SCHEMA, "error"})
LEDGER_SPLIT_STATE_SCHEMA = frozenset(
    {
        "access_started",
        "status",
        "result_sha256",
        "completed_ordinal_count",
        "materialized_ordinal_count",
        "active_ordinal",
        "ordinal_evidence_sha256s",
        "active_batch_ordinals",
        "completed_batch_count",
        "batch_evidence_sha256s",
    }
)


def _require_exact_keys(value: Any, expected: frozenset[str], *, label: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _validate_metrics(metrics: Any) -> None:
    _require_exact_keys(metrics, frozenset(GATE_METRIC_SCHEMA), label="split metrics")
    for name, value in metrics.items():
        if type(name) is not str or type(value) is not float or not math.isfinite(value):
            raise ValueError(f"split metric {name!r} must be an exact finite float")


def _validate_split_result(result: Any, *, split: str) -> None:
    _require_exact_keys(result, SPLIT_RESULT_SCHEMA, label=f"{split} result")
    if split not in SPLITS or result["split"] != split:
        raise ValueError("split result names the wrong conceptual split")
    manifest = _validate_manifest_rows(split, result["manifest"])
    if result["manifest_sha256"] != canonical_sha256(manifest):
        raise ValueError(f"{split} result manifest hash differs")
    _validate_metrics(result["metrics"])
    failures = result["failures"]
    if type(failures) is not list or any(type(item) is not str for item in failures):
        raise TypeError(f"{split} failures must be an exact list of strings")
    recomputed = gate_failures(result["metrics"])
    if failures != recomputed or result["passed"] is not (not recomputed):
        raise ValueError(f"{split} pass/failure evidence does not recompute exactly")
    if type(result["optimizer_updates"]) is not int or result["optimizer_updates"] != 0:
        raise ValueError(f"{split} result must remain optimizer-free")
    _require_exact_keys(result["runtime_api"], RUNTIME_API_SCHEMA, label=f"{split} runtime API")
    expected_runtime = {
        "packet_factory": "make_rgbd_packet",
        "ingest_frames": list(HISTORY_FRAME_INDICES),
        "rollout_method": "OnlineWorldModel.predict",
        "horizons_seconds": list(HORIZONS_SECONDS),
    }
    if result["runtime_api"] != expected_runtime:
        raise ValueError(f"{split} runtime API differs from the public path")
    if result["scene_constructor"] != "private_capability_owned_identifiable_drag_episode":
        raise ValueError(f"{split} result names the wrong private constructor")
    validated_sha256(result["model_state_sha256"], label=f"{split} model state")


def _scale_calibration_record(value: _ScaleCalibration) -> dict[str, Any]:
    return {
        "target_float64": value.target,
        "initial_float32_bits": f"0x{value.initial_bits:08x}",
        "deployed_float32_bits": f"0x{value.deployed_bits:08x}",
        "additional_ulps": value.additional_ulps,
        "coverage_count": value.coverage,
        "predecessor_float32_bits": f"0x{value.predecessor_bits:08x}",
        "predecessor_coverage_count": value.predecessor_coverage,
        "predecessor_admissible": value.predecessor_admissible,
    }


def _validate_calibration(
    value: Any, *, cache: Sequence[SceneSufficientEvidence] | None = None
) -> None:
    _require_exact_keys(value, CALIBRATION_SCHEMA, label="development calibration")
    if (
        value["method"] != "designed_family_scene_max_rank_59_float32_minimal"
        or type(value["confidence"]) is not float
        or value["confidence"] != CALIBRATION_CONFIDENCE
        or type(value["normal_z"]) is not float
        or value["normal_z"] != CALIBRATION_Z
        or type(value["rank"]) is not int
        or value["rank"] != CALIBRATION_RANK
        or type(value["scene_count"]) is not int
        or value["scene_count"] != 64
        or type(value["evidence_ingest_count"]) is not int
        or value["evidence_ingest_count"] != 64
        or type(value["evidence_replay_count"]) is not int
        or value["evidence_replay_count"] != 0
        or type(value["atomic_setter_calls"]) is not int
        or value["atomic_setter_calls"] != 1
    ):
        raise ValueError("development calibration method/count contract differs")
    validated_sha256(value["evidence_cache_sha256"], label="development evidence cache")
    validated_sha256(value["raw_model_state_sha256"], label="raw model state")
    validated_sha256(value["calibrated_model_state_sha256"], label="calibrated model state")
    validated_sha256(
        value["gradient_audit_model_state_sha256"],
        label="development gradient-audit model state",
    )
    if value["gradient_audit_model_state_sha256"] != value["raw_model_state_sha256"]:
        raise ValueError("development gradient audit must bind the scale-one raw state")
    for name in ("position", "velocity", "drag"):
        record = value[name]
        _require_exact_keys(record, SCALE_CALIBRATION_SCHEMA, label=f"{name} calibration")
        if (
            type(record["target_float64"]) is not float
            or not math.isfinite(record["target_float64"])
            or record["target_float64"] <= 0.0
        ):
            raise ValueError(f"{name} calibration target must be finite float")
        for key in ("initial_float32_bits", "deployed_float32_bits", "predecessor_float32_bits"):
            encoded = record[key]
            if type(encoded) is not str or len(encoded) != 10 or not encoded.startswith("0x"):
                raise ValueError(f"{name} {key} must encode exact float32 bits")
            try:
                bits = int(encoded[2:], 16)
            except ValueError as error:
                raise ValueError(f"{name} {key} must be hexadecimal") from error
            _float32_from_bits(bits)
        initial_bits = int(record["initial_float32_bits"][2:], 16)
        deployed_bits = int(record["deployed_float32_bits"][2:], 16)
        predecessor_bits = int(record["predecessor_float32_bits"][2:], 16)
        expected_initial = _ceil_positive_float32(
            max(1.0, record["target_float64"]) if name == "drag" else record["target_float64"]
        )
        if initial_bits != _float32_bits(expected_initial):
            raise ValueError(f"{name} calibration initial bits do not ceil its target")
        if name == "drag" and (
            record["target_float64"] < 1.0 or float(_float32_from_bits(deployed_bits)) < 1.0
        ):
            raise ValueError("drag calibration target/deployed scale must be at least one")
        reconstructed = expected_initial
        if type(record["additional_ulps"]) is not int or record["additional_ulps"] < 0:
            raise ValueError(f"{name} calibration ULP count must be a nonnegative integer")
        positive_infinity = torch.tensor(math.inf, dtype=torch.float32)
        for _ in range(record["additional_ulps"]):
            reconstructed = torch.nextafter(reconstructed, positive_infinity)
        if _float32_bits(reconstructed) != deployed_bits:
            raise ValueError(f"{name} deployed bits do not follow exact +inf nextafter steps")
        if (
            type(record["additional_ulps"]) is not int
            or record["additional_ulps"] < 0
            or type(record["coverage_count"]) is not int
            or record["coverage_count"] < CALIBRATION_RANK
            or record["coverage_count"] > 64
            or type(record["predecessor_coverage_count"]) is not int
            or type(record["predecessor_admissible"]) is not bool
            or (
                record["predecessor_admissible"]
                and record["predecessor_coverage_count"] >= CALIBRATION_RANK
            )
        ):
            raise ValueError(f"{name} calibration lacks float32 minimality evidence")
        if deployed_bits != initial_bits + record["additional_ulps"]:
            raise ValueError(f"{name} calibration ULP accounting differs")
        if predecessor_bits + 1 != deployed_bits:
            raise ValueError(f"{name} predecessor is not exactly one float32 ULP below")
        expected_predecessor_admissible = not (
            name == "drag" and float(_float32_from_bits(predecessor_bits)) < 1.0
        )
        if record["predecessor_admissible"] is not expected_predecessor_admissible:
            raise ValueError(f"{name} predecessor admissibility differs from the lower bound")
    for key in ("variance_floor_clamp_count", "variance_ceiling_clamp_count"):
        if type(value[key]) is not int or value[key] != 0:
            raise ValueError("development calibration requires zero variance clamps")
    if cache is not None:
        if _evidence_cache_sha256(cache) != value["evidence_cache_sha256"]:
            raise ValueError("development calibration cache hash differs from source evidence")
        expected_records = tuple(
            _scale_calibration_record(item) for item in _calibrate_development_cache(cache)
        )
        for name, expected in zip(("position", "velocity", "drag"), expected_records, strict=True):
            if value[name] != expected:
                raise ValueError(f"{name} calibration does not recompute from cached evidence")


def _validate_report_root(report: Any, *, qualification: bool, error: bool) -> None:
    expected = (
        QUALIFICATION_ERROR_REPORT_SCHEMA
        if qualification and error
        else QUALIFICATION_REPORT_SCHEMA
        if qualification
        else DEVELOPMENT_ERROR_REPORT_SCHEMA
        if error
        else DEVELOPMENT_REPORT_SCHEMA
    )
    _require_exact_keys(
        report, expected, label="qualification report" if qualification else "development report"
    )
    expected_kind = (
        "rgbd_identifiable_drag_qualification"
        if qualification
        else "rgbd_identifiable_drag_development"
    )
    if report["artifact_kind"] != expected_kind:
        raise ValueError("report artifact kind differs")
    if canonical_sha256(report["protocol"]) != canonical_sha256(bridge_protocol()):
        raise ValueError("report protocol differs from frozen source")
    _require_exact_keys(
        report["source_provenance"], SOURCE_PROVENANCE_SCHEMA, label="report source"
    )
    _require_exact_keys(
        report["publication_provenance"], PUBLICATION_PROVENANCE_SCHEMA, label="report publication"
    )
    if report["config_sha256"] != FROZEN_CONFIG_SHA256:
        raise ValueError("report config hash differs")
    if report["critical_source_sha256"] != FROZEN_SOURCE_SHA256:
        raise ValueError("report critical source hashes differ")
    certificate = report["scene_family_certificate"]
    if (
        type(certificate) is not dict
        or certificate.get("certificate_sha256") != FROZEN_CERTIFICATE_SHA256
    ):
        raise ValueError("report certificate binding differs")
    if type(report["optimizer_updates"]) is not int or report["optimizer_updates"] != 0:
        raise ValueError("report optimizer updates differ")
    if type(report["protected_data_materialized"]) is not bool:
        raise TypeError("report protected materialization flag must be bool")
    if type(report["passed"]) is not bool or type(report["stopped_after"]) is not str:
        raise TypeError("report outcome fields have wrong types")
    if error:
        _require_exact_keys(report["error"], frozenset({"type", "message"}), label="report error")
        if report["passed"] is not False:
            raise ValueError("error report cannot pass")


def _checkpoint_payload_from_bytes(contents: bytes) -> Mapping[str, Any]:
    if type(contents) is not bytes:
        raise TypeError("checkpoint contents must be exact bytes")
    if len(contents) <= 0 or len(contents) > MAX_CHECKPOINT_BYTES:
        raise ValueError("reviewed checkpoint size is outside the frozen bound")
    payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise ValueError("reviewed checkpoint payload must be an exact dict")
    return payload


def _validate_checkpoint_model_state(state: Any) -> dict[str, Tensor]:
    if type(state) is not dict:
        raise TypeError("checkpoint model state must be an exact dict")
    expected = {f"observation_modules.rgbd.{leaf}" for leaf in _SCALE_STATE_LEAVES}
    if set(state) != expected:
        raise ValueError("checkpoint model state must contain exactly the three reviewed buffers")
    result: dict[str, Tensor] = {}
    for name in sorted(expected):
        tensor = state[name]
        if (
            type(tensor) is not Tensor
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.ndim != 0
        ):
            raise ValueError(f"checkpoint state {name!r} must be one CPU scalar float32 tensor")
        squared = tensor.square()
        if (
            not bool(torch.isfinite(tensor))
            or not bool(tensor > 0.0)
            or not bool(torch.isfinite(squared))
            or not bool(squared > 0.0)
        ):
            raise ValueError(f"checkpoint state {name!r} is not safely finite and positive")
        result[name] = tensor
    return result


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        _update_tensor_digest(digest, state[name])
    return digest.hexdigest()


def _current_execution_provenance(
    *, label: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _validate_repository_identity()
    _validate_frozen_critical_sources()
    source = clean_source(capture_git_metadata(REPOSITORY_ROOT), label=label)
    publication = _validated_published_source(
        capture_published_source(REPOSITORY_ROOT), source=source, label=label
    )
    config_contents = stable_read_bytes(_frozen_config_path(), label=f"{label} config")
    if sha256_bytes(config_contents) != FROZEN_CONFIG_SHA256:
        raise PermissionError(f"{label} observed changed frozen config bytes")
    certificate = _frozen_scene_certificate_binding()
    if (
        type(certificate) is not dict
        or certificate.get("certificate_sha256") != FROZEN_CERTIFICATE_SHA256
    ):
        raise PermissionError(f"{label} observed changed scene certificate")
    return source, publication, certificate


def _guard_frozen_inputs(
    *,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    config: OrpheusConfig,
    config_path: Path,
    expected_inventory: frozenset[str],
    label: str,
) -> dict[str, Any]:
    current_source, current_publication, certificate = _current_execution_provenance(label=label)
    if current_source != source:
        raise RuntimeError(f"{label} source provenance changed")
    if current_publication != publication:
        raise RuntimeError(f"{label} publication provenance changed")
    _require_config_matches_frozen_path(config, config_path)
    _validate_run_tree(expected_inventory, stage=label)
    return certificate


def _require_single_thread_execution() -> None:
    if torch.get_num_threads() != 1 or threading.active_count() != 1:
        raise RuntimeError("identifiable-drag qualification requires exactly one active CPU thread")


_RUN_AUTHORITY = object()
_CAPABILITY_AUTHORITY = object()
_LIVE_RUN_AUTHORIZATIONS: dict[int, tuple[object, ...]] = {}
_LIVE_REVIEWED_DEVELOPMENT_SEALS: dict[int, tuple[object, ...]] = {}
_LIVE_PRIVATE_LEDGERS: dict[int, tuple[object, ...]] = {}
_LIVE_LEDGER_RECEIPTS: dict[int, tuple[object, ...]] = {}
_LIVE_MANIFEST_CAPABILITIES: dict[int, tuple[object, ...]] = {}
_LIVE_MANIFEST_BINDINGS: dict[tuple[int, str], tuple[object, object]] = {}
_LIVE_ORDINAL_CAPABILITIES: dict[int, tuple[object, ...]] = {}
_LIVE_BATCH_CAPABILITIES: dict[int, tuple[object, ...]] = {}


class _RunAuthorization:
    """Unconstructable nominal proof of one exact preflight."""

    def __init__(self) -> None:
        raise PermissionError("run authorizations are minted only by exact preflight")


class _ReviewedDevelopmentSeal:
    """Unconstructable nominal proof of all three externally reviewed bytes."""

    def __init__(self) -> None:
        raise PermissionError("review seals are minted only by exact development validation")


def _mint_run_authorization(
    kind: str,
    bindings: Mapping[str, Any],
    *,
    reviewed_seal: _ReviewedDevelopmentSeal | None = None,
) -> _RunAuthorization:
    if kind not in {"development", "qualification"} or type(bindings) is not dict:
        raise PermissionError("run authorization requires exact kind/bindings")
    if kind == "qualification" and type(reviewed_seal) is not _ReviewedDevelopmentSeal:
        raise PermissionError("qualification authorization requires reviewed development")
    _require_single_thread_execution()
    source, publication, _ = _current_execution_provenance(
        label=f"identifiable-drag {kind} authorization"
    )
    expected_inventory = frozenset() if kind == "development" else DEVELOPMENT_ARTIFACT_NAMES
    _validate_run_tree(expected_inventory, stage=f"{kind} authorization")
    if kind == "development":
        expected = {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": source,
            "publication_provenance": publication,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "critical_source_sha256": dict(FROZEN_SOURCE_SHA256),
            "development_manifest_sha256": MANIFEST_SHA256["development"],
            "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        }
    else:
        seal_registration = _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(id(reviewed_seal))
        if (
            type(seal_registration) is not tuple
            or len(seal_registration) != 3
            or seal_registration[0] is not reviewed_seal
            or seal_registration[2] is not None
        ):
            raise PermissionError("qualification authorization lacks a fresh live review seal")
        expected = {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": source,
            "publication_provenance": publication,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "critical_source_sha256": dict(FROZEN_SOURCE_SHA256),
            "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
            "reviewed_checkpoint_sha256": bindings.get("reviewed_checkpoint_sha256"),
            "reviewed_development_report_sha256": bindings.get(
                "reviewed_development_report_sha256"
            ),
            "reviewed_development_ledger_sha256": bindings.get(
                "reviewed_development_ledger_sha256"
            ),
            "model_state_sha256": bindings.get("model_state_sha256"),
            "calibration_sha256": bindings.get("calibration_sha256"),
        }
        for key in (
            "reviewed_checkpoint_sha256",
            "reviewed_development_report_sha256",
            "reviewed_development_ledger_sha256",
            "model_state_sha256",
            "calibration_sha256",
        ):
            validated_sha256(expected[key], label=key)
        if seal_registration[1] != canonical_sha256(expected):
            raise PermissionError("qualification bindings differ from reviewed seal")
    if bindings != expected:
        raise PermissionError("run authorization binding values differ from live frozen inputs")
    binding_sha256 = canonical_sha256(bindings)
    if any(
        registration[2] == kind and registration[3] == binding_sha256
        for registration in _LIVE_RUN_AUTHORIZATIONS.values()
    ):
        raise PermissionError("an identical live run authorization already exists")
    authorization = object.__new__(_RunAuthorization)
    _LIVE_RUN_AUTHORIZATIONS[id(authorization)] = (
        authorization,
        _RUN_AUTHORITY,
        kind,
        binding_sha256,
        reviewed_seal,
    )
    return authorization


def _consume_run_authorization(
    authorization: _RunAuthorization,
    *,
    kind: str,
    bindings: Mapping[str, Any],
    reviewed_seal: _ReviewedDevelopmentSeal | None = None,
) -> None:
    _require_single_thread_execution()
    if type(authorization) is not _RunAuthorization:
        raise PermissionError("run authorization has the wrong nominal type")
    expected = (
        authorization,
        _RUN_AUTHORITY,
        kind,
        canonical_sha256(dict(bindings)),
        reviewed_seal,
    )
    if _LIVE_RUN_AUTHORIZATIONS.get(id(authorization)) != expected:
        raise PermissionError("run authorization is forged, stale, or bound differently")
    _LIVE_RUN_AUTHORIZATIONS.pop(id(authorization), None)


def _mint_reviewed_development_seal(bindings: Mapping[str, Any]) -> _ReviewedDevelopmentSeal:
    _require_single_thread_execution()
    if type(bindings) is not dict:
        raise PermissionError("review seal requires exact bindings")
    for name in (
        "reviewed_checkpoint_sha256",
        "reviewed_development_report_sha256",
        "reviewed_development_ledger_sha256",
        "model_state_sha256",
        "calibration_sha256",
    ):
        validated_sha256(bindings.get(name), label=name)
    binding_sha256 = canonical_sha256(dict(bindings))
    if any(
        registration[1] == binding_sha256
        for registration in _LIVE_REVIEWED_DEVELOPMENT_SEALS.values()
    ):
        raise PermissionError("an identical live reviewed-development seal already exists")
    seal = object.__new__(_ReviewedDevelopmentSeal)
    _LIVE_REVIEWED_DEVELOPMENT_SEALS[id(seal)] = (
        seal,
        binding_sha256,
        None,
    )
    return seal


def _bind_reviewed_development_seal(
    seal: _ReviewedDevelopmentSeal, *, bindings: Mapping[str, Any], ledger: object
) -> None:
    _require_single_thread_execution()
    registration = _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(id(seal))
    if registration != (seal, canonical_sha256(dict(bindings)), None):
        raise PermissionError("reviewed-development seal is stale or already bound")
    _LIVE_REVIEWED_DEVELOPMENT_SEALS[id(seal)] = (
        seal,
        canonical_sha256(dict(bindings)),
        ledger,
    )


def _mint_owned_qualification_authorization(
    seal: _ReviewedDevelopmentSeal,
    bindings: Mapping[str, Any],
) -> _RunAuthorization:
    """Mint protected authority or revoke every authority created by the attempt."""

    prior_authorizations = frozenset(_LIVE_RUN_AUTHORIZATIONS)
    try:
        return _mint_run_authorization("qualification", bindings, reviewed_seal=seal)
    except BaseException:
        for authorization_id in set(_LIVE_RUN_AUTHORIZATIONS) - prior_authorizations:
            registration = _LIVE_RUN_AUTHORIZATIONS.get(authorization_id)
            if type(registration) is tuple and len(registration) == 5 and registration[4] is seal:
                _LIVE_RUN_AUTHORIZATIONS.pop(authorization_id, None)
        seal_registration = _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(id(seal))
        if seal_registration == (seal, canonical_sha256(dict(bindings)), None):
            _LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(seal), None)
        raise


class _OrdinalCapability:
    """One exact constructor/evaluator use, owned by a live manifest capability."""

    def __init__(self) -> None:
        raise PermissionError("ordinal capabilities are minted only by a live manifest")


class _BatchCapability:
    """One exact aligned four-ordinal constructor/evaluator transaction."""

    def __init__(self) -> None:
        raise PermissionError("batch capabilities are minted only by a live manifest")


def _require_exact_governed_ordinal(split: object, ordinal: object) -> None:
    if type(split) is not str or split not in SPLITS:
        raise ValueError("governed split must be an exact known string")
    if type(ordinal) is not int or not 0 <= ordinal < 64:
        raise TypeError("governed ordinal must be an exact bounded integer")


def _require_exact_governed_batch(split: object, ordinals: object) -> None:
    if type(split) is not str or split not in SPLITS:
        raise ValueError("governed split must be an exact known string")
    if (
        type(ordinals) is not tuple
        or len(ordinals) != 4
        or any(type(ordinal) is not int for ordinal in ordinals)
        or any(not 0 <= ordinal < 64 for ordinal in ordinals)
        or ordinals != tuple(range(ordinals[0], ordinals[0] + 4))
        or ordinals[0] % 4 != 0
    ):
        raise TypeError("governed batch must be an exact aligned four-integer tuple")


class _ManifestCapability:
    """Exact ordered 0..63 authority owned by one durable private ledger."""

    def __init__(
        self,
        authority: object,
        *,
        ledger: object,
        ledger_mint_identity: object,
        split: str,
    ) -> None:
        _require_single_thread_execution()
        if authority is not _CAPABILITY_AUTHORITY:
            raise PermissionError("manifest capability authority is private")
        if type(ledger) not in {_DevelopmentLedger, _QualificationLedger}:
            raise PermissionError("manifest capability requires exact private ledger type")
        registration = _LIVE_PRIVATE_LEDGERS.get(id(ledger))
        if (
            type(registration) is not tuple
            or len(registration) != 5
            or registration[0] is not ledger
            or registration[1] is not ledger_mint_identity
            or registration[4] != canonical_sha256(getattr(ledger, "_bindings", None))
        ):
            raise PermissionError("manifest capability ledger is not live registered")
        if split not in SPLITS:
            raise ValueError("manifest capability split is unknown")
        binding_key = (id(ledger), split)
        if binding_key in _LIVE_MANIFEST_BINDINGS:
            raise PermissionError("ledger split already owns a live manifest capability")
        self._ledger = ledger
        self._ledger_mint_identity = ledger_mint_identity
        self._split = split
        self._next_ordinal = 0
        self._active: _OrdinalCapability | None = None
        self._pending: dict[int, _OrdinalCapability] = {}
        self._active_batch: _BatchCapability | None = None
        self._finished = False
        _LIVE_MANIFEST_CAPABILITIES[id(self)] = (self, ledger, ledger_mint_identity, split)
        _LIVE_MANIFEST_BINDINGS[binding_key] = (ledger, self)

    def begin_ordinal(self, ordinal: int) -> _OrdinalCapability:
        _validate_manifest_capability(self, split=self._split, operation="begin")
        if type(ordinal) is not int:
            raise TypeError("governed ordinal must be an exact integer")
        if (
            self._finished
            or self._active is not None
            or self._pending
            or self._active_batch is not None
            or ordinal != self._next_ordinal
        ):
            raise RuntimeError("governed ordinal order/replay differs from exact 0..63")
        self._ledger._begin_ordinal(self._split, ordinal)
        capability = object.__new__(_OrdinalCapability)
        self._active = capability
        self._pending[ordinal] = capability
        _LIVE_ORDINAL_CAPABILITIES[id(capability)] = (
            capability,
            self,
            self._ledger,
            self._split,
            ordinal,
            "issued",
        )
        return capability

    def begin_batch(self, ordinals: Sequence[int]) -> _BatchCapability:
        _validate_manifest_capability(self, split=self._split, operation="begin")
        if type(ordinals) is not tuple or len(ordinals) != 4:
            raise TypeError("governed batch must be an exact four-ordinal tuple")
        expected = tuple(range(self._next_ordinal, self._next_ordinal + 4))
        if (
            any(type(ordinal) is not int for ordinal in ordinals)
            or tuple(ordinals) != expected
            or self._next_ordinal % 4 != 0
            or self._finished
            or self._active is not None
            or self._pending
            or self._active_batch is not None
            or expected[-1] >= 64
        ):
            raise RuntimeError("governed batch must be the next aligned consecutive four rows")
        self._ledger._begin_batch(self._split, expected)
        batch = object.__new__(_BatchCapability)
        tokens: list[_OrdinalCapability] = []
        for ordinal in expected:
            token = object.__new__(_OrdinalCapability)
            tokens.append(token)
            self._pending[ordinal] = token
            _LIVE_ORDINAL_CAPABILITIES[id(token)] = (
                token,
                self,
                self._ledger,
                self._split,
                ordinal,
                "issued",
            )
        batch._manifest = self
        batch._ledger = self._ledger
        batch._split = self._split
        batch._ordinals = expected
        batch._tokens = tuple(tokens)
        batch._next_constructor = 0
        batch._evaluated = False
        self._active_batch = batch
        _LIVE_BATCH_CAPABILITIES[id(batch)] = (
            batch,
            self,
            self._ledger,
            self._split,
            expected,
            tuple(tokens),
        )
        return batch

    def complete_batch(
        self,
        batch: _BatchCapability,
        *,
        evidence_sha256s: Sequence[str],
    ) -> None:
        _validate_batch_capability(batch, split=self._split, operation="complete")
        if type(evidence_sha256s) is not tuple or len(evidence_sha256s) != 4:
            raise TypeError("batch evidence hashes must be an exact four-tuple")
        for digest in evidence_sha256s:
            validated_sha256(digest, label="batch ordinal evidence")
        if not batch._evaluated or batch._next_constructor != 4:
            raise RuntimeError("batch cannot complete before exact construction/evaluation")
        for ordinal, token in zip(batch._ordinals, batch._tokens, strict=True):
            registration = _LIVE_ORDINAL_CAPABILITIES.get(id(token))
            if registration != (
                token,
                self,
                self._ledger,
                self._split,
                ordinal,
                "evaluated",
            ):
                raise PermissionError("batch completion has partial/reordered token state")
        self._ledger._complete_batch(self._split, batch._ordinals, tuple(evidence_sha256s))
        for ordinal, token in zip(batch._ordinals, batch._tokens, strict=True):
            _LIVE_ORDINAL_CAPABILITIES.pop(id(token), None)
            self._pending.pop(ordinal, None)
        _LIVE_BATCH_CAPABILITIES.pop(id(batch), None)
        self._active_batch = None
        self._next_ordinal += 4

    def complete_ordinal(
        self, capability: _OrdinalCapability, *, ordinal: int, evidence_sha256: str
    ) -> None:
        _require_exact_governed_ordinal(self._split, ordinal)
        _validate_manifest_capability(self, split=self._split, operation="finish")
        validated_sha256(evidence_sha256, label="ordinal evidence")
        registration = _LIVE_ORDINAL_CAPABILITIES.get(id(capability))
        expected = (
            capability,
            self,
            self._ledger,
            self._split,
            ordinal,
            "evaluated",
        )
        if type(capability) is not _OrdinalCapability or registration != expected:
            raise PermissionError("ordinal completion lacks exact evaluated nominal capability")
        if self._active is not capability or self._pending.get(ordinal) is not capability:
            raise RuntimeError("ordinal completion differs from live manifest pending set")
        self._ledger._complete_ordinal(self._split, ordinal, evidence_sha256)
        _LIVE_ORDINAL_CAPABILITIES.pop(id(capability), None)
        self._pending.pop(ordinal, None)
        self._active = None
        self._next_ordinal += 1

    def finish_manifest(self) -> None:
        _validate_manifest_capability(self, split=self._split, operation="complete")
        if (
            self._active is not None
            or self._pending
            or self._active_batch is not None
            or self._next_ordinal != 64
            or self._finished
        ):
            raise RuntimeError("manifest cannot finish before exact single use of ordinals 0..63")
        self._finished = True

    def require_finished(self) -> None:
        _validate_manifest_capability(self, split=self._split, operation="complete")
        if (
            not self._finished
            or self._next_ordinal != 64
            or self._active is not None
            or self._pending
            or self._active_batch is not None
        ):
            raise RuntimeError("manifest capability is not exactly finished")


def _retire_manifest_capability(capability: _ManifestCapability) -> None:
    registration = _LIVE_MANIFEST_CAPABILITIES.get(id(capability))
    if type(registration) is not tuple or len(registration) != 4:
        raise PermissionError("cannot retire a non-live manifest capability")
    ledger = registration[1]
    split = registration[3]
    binding_key = (id(ledger), split)
    if _LIVE_MANIFEST_BINDINGS.get(binding_key) != (ledger, capability):
        raise PermissionError("manifest binding registry differs during retirement")
    _LIVE_MANIFEST_BINDINGS.pop(binding_key)
    _LIVE_MANIFEST_CAPABILITIES.pop(id(capability))


def _revoke_ledger_governed_access(ledger: object) -> None:
    """Irrevocably remove every data-access token without fallible I/O."""

    for capability_id, registration in tuple(_LIVE_ORDINAL_CAPABILITIES.items()):
        if type(registration) is tuple and len(registration) >= 3 and registration[2] is ledger:
            _LIVE_ORDINAL_CAPABILITIES.pop(capability_id, None)
    for capability_id, registration in tuple(_LIVE_BATCH_CAPABILITIES.items()):
        if type(registration) is tuple and len(registration) >= 3 and registration[2] is ledger:
            _LIVE_BATCH_CAPABILITIES.pop(capability_id, None)
    for capability_id, registration in tuple(_LIVE_MANIFEST_CAPABILITIES.items()):
        if type(registration) is tuple and len(registration) >= 2 and registration[1] is ledger:
            _LIVE_MANIFEST_CAPABILITIES.pop(capability_id, None)
    for binding_key, registration in tuple(_LIVE_MANIFEST_BINDINGS.items()):
        if registration[0] is ledger:
            _LIVE_MANIFEST_BINDINGS.pop(binding_key, None)


def _consume_ordinal_constructor_capability(
    capability: _OrdinalCapability, *, split: str, ordinal: int
) -> None:
    _require_exact_governed_ordinal(split, ordinal)
    if type(capability) is not _OrdinalCapability:
        raise PermissionError("constructor requires exact ordinal nominal capability")
    registration = _LIVE_ORDINAL_CAPABILITIES.get(id(capability))
    if (
        type(registration) is not tuple
        or len(registration) != 6
        or registration[0] is not capability
        or registration[3] != split
        or registration[4] != ordinal
        or registration[5] != "issued"
    ):
        raise PermissionError("constructor capability is forged, replayed, or bound differently")
    manifest = registration[1]
    _validate_manifest_capability(manifest, split=split, operation="constructor")
    batch = manifest._active_batch
    if batch is not None:
        _validate_batch_capability(batch, split=split, operation="constructor")
        index = batch._next_constructor
        if (
            index >= 4
            or batch._tokens[index] is not capability
            or batch._ordinals[index] != ordinal
        ):
            raise PermissionError("batch constructor order is partial, duplicated, or reordered")
        batch._ledger._begin_batch_ordinal(split, batch._ordinals, ordinal)
    _LIVE_ORDINAL_CAPABILITIES[id(capability)] = (*registration[:5], "constructed")


def _mark_ordinal_evaluated(capability: _OrdinalCapability, *, split: str, ordinal: int) -> None:
    _require_exact_governed_ordinal(split, ordinal)
    if type(capability) is not _OrdinalCapability:
        raise PermissionError("evaluator requires exact ordinal nominal capability")
    registration = _LIVE_ORDINAL_CAPABILITIES.get(id(capability))
    if (
        type(registration) is not tuple
        or len(registration) != 6
        or registration[0] is not capability
        or registration[3] != split
        or registration[4] != ordinal
        or registration[5] != "constructed"
    ):
        raise PermissionError("evaluator capability lacks one exact constructor use")
    manifest = registration[1]
    _validate_manifest_capability(manifest, split=split, operation="evaluator")
    _LIVE_ORDINAL_CAPABILITIES[id(capability)] = (*registration[:5], "evaluated")


def _mark_ordinal_constructed(capability: _OrdinalCapability, *, split: str, ordinal: int) -> None:
    _require_exact_governed_ordinal(split, ordinal)
    if type(capability) is not _OrdinalCapability:
        raise PermissionError("constructor completion requires exact ordinal capability")
    registration = _LIVE_ORDINAL_CAPABILITIES.get(id(capability))
    if (
        type(registration) is not tuple
        or len(registration) != 6
        or registration[0] is not capability
        or registration[3] != split
        or registration[4] != ordinal
        or registration[5] != "constructed"
    ):
        raise PermissionError("constructor completion lacks exact consumed capability")
    manifest = registration[1]
    _validate_manifest_capability(manifest, split=split, operation="constructor")
    if manifest._active is not capability or manifest._next_ordinal != ordinal:
        batch = manifest._active_batch
        if batch is None:
            raise RuntimeError("constructor completion differs from live manifest position")
        _validate_batch_capability(batch, split=split, operation="constructor")
        if batch._tokens[batch._next_constructor] is not capability:
            raise RuntimeError("batch constructor completion differs from exact tuple position")
        batch._ledger._mark_batch_ordinal_constructed(split, batch._ordinals, ordinal)
        batch._next_constructor += 1
        return
    manifest._ledger._mark_ordinal_constructed(split, ordinal)


def _validate_batch_capability(batch: _BatchCapability, *, split: str, operation: str) -> None:
    if type(batch) is not _BatchCapability:
        raise PermissionError("batch capability has the wrong nominal type")
    registration = _LIVE_BATCH_CAPABILITIES.get(id(batch))
    if registration != (
        batch,
        getattr(batch, "_manifest", None),
        getattr(batch, "_ledger", None),
        split,
        getattr(batch, "_ordinals", None),
        getattr(batch, "_tokens", None),
    ):
        raise PermissionError("batch capability is forged, stale, or bound differently")
    manifest = batch._manifest
    _validate_manifest_capability(manifest, split=split, operation="evaluator")
    if manifest._active_batch is not batch:
        raise PermissionError("batch capability is not uniquely manifest-owned")
    if operation not in {"constructor", "evaluator", "complete"}:
        raise ValueError("unknown batch capability operation")


def _mark_batch_evaluated(batch: _BatchCapability) -> None:
    _validate_batch_capability(batch, split=batch._split, operation="evaluator")
    if batch._next_constructor != 4 or batch._evaluated:
        raise RuntimeError("batch evaluator requires exactly four constructed rows once")
    for ordinal, token in zip(batch._ordinals, batch._tokens, strict=True):
        registration = _LIVE_ORDINAL_CAPABILITIES.get(id(token))
        if registration != (
            token,
            batch._manifest,
            batch._ledger,
            batch._split,
            ordinal,
            "constructed",
        ):
            raise PermissionError("batch evaluator token tuple is partial or reordered")
    for token in batch._tokens:
        registration = _LIVE_ORDINAL_CAPABILITIES[id(token)]
        _LIVE_ORDINAL_CAPABILITIES[id(token)] = (*registration[:5], "evaluated")
    batch._evaluated = True


def _validate_ledger_record_shape(record: Any, *, qualification: bool) -> None:
    if type(record) is not dict:
        raise TypeError("private ledger root must be an exact dict")
    if qualification:
        base = set(QUALIFICATION_LEDGER_COMPLETE_SCHEMA) - {
            "outcome",
            "stopped_after",
            "report_sha256",
        }
        allowed = (
            base,
            set(QUALIFICATION_LEDGER_COMPLETE_SCHEMA),
            set(QUALIFICATION_LEDGER_COMPLETE_SCHEMA) | {"status_before_error", "error"},
        )
        if set(record) not in allowed:
            raise ValueError("qualification ledger root has a non-exact stage schema")
        _require_exact_keys(
            record.get("splits"),
            frozenset(_QualificationLedger.ORDER),
            label="qualification ledger splits",
        )
        split_records = record["splits"].values()
    else:
        base = set(DEVELOPMENT_LEDGER_COMPLETE_SCHEMA) - {
            "outcome",
            "report_sha256",
            "checkpoint_sha256",
        }
        allowed = (
            base,
            base | {"outcome"},
            set(DEVELOPMENT_LEDGER_COMPLETE_SCHEMA),
            set(DEVELOPMENT_LEDGER_COMPLETE_SCHEMA) | {"status_before_error", "error"},
        )
        if set(record) not in allowed:
            raise ValueError("development ledger root has a non-exact stage schema")
        split_records = (record,)
    for split_record in split_records:
        if qualification:
            _require_exact_keys(
                split_record,
                LEDGER_SPLIT_STATE_SCHEMA,
                label="qualification ledger split state",
            )
        for name in (
            "completed_ordinal_count",
            "materialized_ordinal_count",
            "completed_batch_count",
        ):
            value = split_record[name]
            if type(value) is not int or not 0 <= value <= 64:
                raise ValueError(f"ledger {name} must be an exact bounded integer")
        completed = split_record["completed_ordinal_count"]
        materialized = split_record["materialized_ordinal_count"]
        completed_batches = split_record["completed_batch_count"]
        if not 0 <= completed <= materialized <= 64 or not 0 <= completed_batches <= 16:
            raise ValueError("ledger ordinal/batch counters are inconsistent")
        ordinal_hashes = split_record["ordinal_evidence_sha256s"]
        batch_hashes = split_record["batch_evidence_sha256s"]
        if (
            type(ordinal_hashes) is not list
            or len(ordinal_hashes) != completed
            or type(batch_hashes) is not list
            or len(batch_hashes) != completed_batches
        ):
            raise ValueError("ledger evidence hash counts differ from durable counters")
        for digest in (*ordinal_hashes, *batch_hashes):
            validated_sha256(digest, label="ledger evidence SHA-256")
        active = split_record["active_ordinal"]
        if active is not None and (type(active) is not int or not 0 <= active < 64):
            raise ValueError("ledger active ordinal is not exact")
        active_batch = split_record["active_batch_ordinals"]
        if active_batch is not None and (
            type(active_batch) is not list
            or len(active_batch) != 4
            or any(type(item) is not int for item in active_batch)
            or active_batch != list(range(active_batch[0], active_batch[0] + 4))
            or active_batch[0] % 4 != 0
        ):
            raise ValueError("ledger active batch is not an exact aligned four-row list")


def _refresh_ledger_receipt(ledger: object) -> None:
    path = getattr(ledger, "path", None)
    record = getattr(ledger, "record", None)
    if type(path) is not _NATIVE_PATH_TYPE or type(record) is not dict:
        raise PermissionError("private ledger has the wrong nominal state")
    metadata = _require_single_link_regular(path, label="identifiable-drag ledger")
    contents = stable_read_bytes(path, label="identifiable-drag live ledger")
    parsed = json.loads(contents)
    if type(parsed) is not dict or canonical_sha256(parsed) != canonical_sha256(record):
        raise RuntimeError("ledger memory and durable receipt differ")
    _validate_ledger_record_shape(parsed, qualification=type(ledger) is _QualificationLedger)
    ledger._receipt_digest = sha256_bytes(contents)
    ledger._receipt_device = metadata.st_dev
    ledger._receipt_inode = metadata.st_ino
    ledger._durable_record = parsed


def _ledger_receipt_pin(ledger: object, *, generation: int) -> tuple[object, ...]:
    if type(generation) is not int or generation < 0:
        raise ValueError("ledger receipt generation must be an exact nonnegative integer")
    return (
        ledger,
        ledger.path,
        ledger._receipt_device,
        ledger._receipt_inode,
        ledger._receipt_digest,
        canonical_sha256(ledger._durable_record),
        canonical_sha256(ledger._bindings),
        generation,
    )


def _register_ledger_receipt(ledger: object) -> None:
    if id(ledger) in _LIVE_LEDGER_RECEIPTS:
        raise PermissionError("ledger receipt identity is already registered")
    ledger._receipt_generation = 0
    _LIVE_LEDGER_RECEIPTS[id(ledger)] = _ledger_receipt_pin(ledger, generation=0)


def _validate_live_ledger_receipt(
    ledger: object,
    *,
    split: str,
    expected_inventory: frozenset[str] | None = None,
) -> None:
    _require_single_thread_execution()
    registration = _LIVE_PRIVATE_LEDGERS.get(id(ledger))
    if (
        type(registration) is not tuple
        or len(registration) != 5
        or registration[0] is not ledger
        or registration[1] is not getattr(ledger, "_mint_identity", None)
    ):
        raise PermissionError("private ledger is not live issuer registered")
    receipt_pin = _LIVE_LEDGER_RECEIPTS.get(id(ledger))
    if (
        type(receipt_pin) is not tuple
        or len(receipt_pin) != 8
        or receipt_pin[0] is not ledger
        or receipt_pin[1] != getattr(ledger, "path", None)
        or receipt_pin[2] != getattr(ledger, "_receipt_device", None)
        or receipt_pin[3] != getattr(ledger, "_receipt_inode", None)
        or receipt_pin[4] != getattr(ledger, "_receipt_digest", None)
        or receipt_pin[5] != canonical_sha256(getattr(ledger, "_durable_record", None))
        or receipt_pin[6] != canonical_sha256(getattr(ledger, "_bindings", None))
        or receipt_pin[7] != getattr(ledger, "_receipt_generation", None)
    ):
        raise PermissionError("private ledger receipt identity differs from trusted registry")
    expected_path = (
        development_ledger_path()
        if type(ledger) is _DevelopmentLedger
        else qualification_ledger_path()
    )
    if type(ledger.path) is not _NATIVE_PATH_TYPE or ledger.path != expected_path:
        raise PermissionError("private ledger path is not canonical")
    _require_config_matches_frozen_path(ledger._config, _frozen_config_path())
    run_directory = _canonical_run_directory()
    _require_nonlink_directory(run_directory, label=f"{split} live capability run directory")
    with os.scandir(run_directory) as entries:
        current_inventory = frozenset(entry.name for entry in entries)
    default_inventory = (
        frozenset({DEVELOPMENT_LEDGER_NAME})
        if type(ledger) is _DevelopmentLedger
        else frozenset({*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_LEDGER_NAME})
    )
    required_inventory = default_inventory if expected_inventory is None else expected_inventory
    if type(required_inventory) is not frozenset or current_inventory != required_inventory:
        raise PermissionError(
            f"{split} live capability has a non-stage artifact inventory: "
            f"expected {sorted(required_inventory)!r}, got {sorted(current_inventory)!r}"
        )
    _validate_run_tree(current_inventory, stage=f"{split} live capability")
    metadata = os.lstat(ledger.path)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_dev != ledger._receipt_device
        or metadata.st_ino != ledger._receipt_inode
    ):
        raise PermissionError("private ledger identity/link count changed")
    contents = stable_read_bytes(ledger.path, label=f"{split} live capability receipt")
    if sha256_bytes(contents) != ledger._receipt_digest:
        raise PermissionError("private ledger receipt bytes changed")
    parsed = json.loads(contents)
    durable_record = getattr(ledger, "_durable_record", None)
    memory_record = getattr(ledger, "record", None)
    bindings = getattr(ledger, "_bindings", None)
    if (
        type(parsed) is not dict
        or type(durable_record) is not dict
        or type(memory_record) is not dict
        or type(bindings) is not dict
        or parsed != durable_record
        or memory_record != durable_record
        or canonical_sha256(parsed) != receipt_pin[5]
    ):
        raise PermissionError("private ledger durable record differs from its pinned receipt")
    if registration[4] != canonical_sha256(bindings) or parsed.get("bindings") != bindings:
        raise PermissionError("private ledger immutable bindings changed")
    _validate_ledger_record_shape(parsed, qualification=type(ledger) is _QualificationLedger)
    source, publication, certificate = _current_execution_provenance(
        label=f"{split} live capability"
    )
    if ledger._bindings.get("source_provenance") != source:
        raise PermissionError("private ledger source binding changed")
    if ledger._bindings.get("publication_provenance") != publication:
        raise PermissionError("private ledger publication binding changed")
    if type(ledger) is _QualificationLedger:
        reviewed_contents: dict[str, bytes] = {}
        for path, binding, label in (
            (canonical_checkpoint_path(), "reviewed_checkpoint_sha256", "checkpoint"),
            (
                canonical_development_report_path(),
                "reviewed_development_report_sha256",
                "development report",
            ),
            (
                development_ledger_path(),
                "reviewed_development_ledger_sha256",
                "development ledger",
            ),
        ):
            _require_single_link_regular(path, label=f"live reviewed {label}")
            contents = stable_read_bytes(path, label=f"live reviewed {label}")
            reviewed_contents[label] = contents
            if sha256_bytes(contents) != ledger._bindings[binding]:
                raise PermissionError(f"live reviewed {label} differs from protected binding")
            if label == "checkpoint":
                payload = _checkpoint_payload_from_bytes(contents)
                _require_exact_keys(payload, CHECKPOINT_SCHEMA, label="live reviewed checkpoint")
                _validate_checkpoint_model_state(payload["model_state"])
            else:
                parsed_review = _strict_json_loads(contents, label=f"live reviewed {label}")
                _require_exact_keys(
                    parsed_review,
                    DEVELOPMENT_REPORT_SCHEMA
                    if label == "development report"
                    else DEVELOPMENT_LEDGER_COMPLETE_SCHEMA,
                    label=f"live reviewed {label}",
                )
        report = _strict_json_loads(
            reviewed_contents["development report"],
            label="live reviewed development report",
        )
        development, calibration = _validate_development_report(
            report,
            checkpoint_sha256=ledger._bindings["reviewed_checkpoint_sha256"],
            source=source,
            publication=publication,
            certificate=certificate,
        )
        development_ledger = _strict_json_loads(
            reviewed_contents["development ledger"],
            label="live reviewed development ledger",
        )
        _validate_development_ledger_record(
            development_ledger,
            report=report,
            development=development,
            report_sha256=ledger._bindings["reviewed_development_report_sha256"],
            checkpoint_sha256=ledger._bindings["reviewed_checkpoint_sha256"],
            source=source,
            publication=publication,
        )
        state, state_sha256 = _validate_checkpoint_evidence(
            _checkpoint_payload_from_bytes(reviewed_contents["checkpoint"]),
            config=ledger._config,
            source=source,
            publication=publication,
            development=development,
            calibration=calibration,
        )
        if (
            state_sha256 != ledger._bindings["model_state_sha256"]
            or canonical_sha256(calibration) != ledger._bindings["calibration_sha256"]
            or set(state) != {f"observation_modules.rgbd.{leaf}" for leaf in _SCALE_STATE_LEAVES}
        ):
            raise PermissionError("live reviewed bundle differs from protected bindings")


def _exact_error_inventory(*, qualification: bool) -> frozenset[str]:
    run_directory = _canonical_run_directory()
    _require_nonlink_directory(run_directory, label="error receipt run directory")
    with os.scandir(run_directory) as entries:
        inventory = frozenset(entry.name for entry in entries)
    allowed = QUALIFICATION_ARTIFACT_NAMES if qualification else DEVELOPMENT_ARTIFACT_NAMES
    required = (
        frozenset({*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_LEDGER_NAME})
        if qualification
        else frozenset({DEVELOPMENT_LEDGER_NAME})
    )
    if not required.issubset(inventory) or not inventory.issubset(allowed):
        raise PermissionError("error receipt has a non-stage artifact inventory")
    _validate_run_tree(inventory, stage="exact error receipt")
    return inventory


def _validate_manifest_capability(
    capability: _ManifestCapability, *, split: str, operation: str
) -> None:
    if type(capability) is not _ManifestCapability:
        raise PermissionError("manifest capability has the wrong nominal type")
    registration = _LIVE_MANIFEST_CAPABILITIES.get(id(capability))
    if registration != (
        capability,
        getattr(capability, "_ledger", None),
        getattr(capability, "_ledger_mint_identity", None),
        split,
    ):
        raise PermissionError("manifest capability is not live ledger-owned")
    ledger = capability._ledger
    if _LIVE_MANIFEST_BINDINGS.get((id(ledger), split)) != (ledger, capability):
        raise PermissionError("manifest capability is not the unique live ledger/split owner")
    _validate_live_ledger_receipt(ledger, split=split)
    if split != capability._split or split not in SPLITS:
        raise PermissionError("manifest capability split differs")
    if MANIFEST_SHA256[split] != canonical_sha256(list(_manifest_rows(split))):
        raise PermissionError("manifest rows/hash differ from frozen source")
    if type(ledger) is _DevelopmentLedger:
        if split != "development" or ledger._capability is not capability:
            raise PermissionError("development capability is not uniquely ledger-owned")
    else:
        if ledger._capabilities.get(split) is not capability:
            raise PermissionError("protected capability is not uniquely ledger-owned")
        seal = ledger._reviewed_seal
        if _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(id(seal)) != (
            seal,
            canonical_sha256(ledger._bindings),
            ledger,
        ):
            raise PermissionError("protected capability lost its review seal")
    if operation not in {"begin", "finish", "complete", "constructor", "evaluator"}:
        raise ValueError("unknown manifest capability operation")


def _ledger_bound_source_guard(ledger: object, *, label: str) -> None:
    source, publication, _ = _current_execution_provenance(label=label)
    if getattr(ledger, "_bindings", {}).get("source_provenance") != source:
        raise RuntimeError(f"{label} source binding changed")
    if getattr(ledger, "_bindings", {}).get("publication_provenance") != publication:
        raise RuntimeError(f"{label} publication binding changed")


class _DevelopmentLedger:
    """One durable, non-retryable development access receipt."""

    ARTIFACT_KIND = "rgbd_identifiable_drag_development_access_ledger"

    def __init__(
        self,
        authorization: _RunAuthorization,
        bindings: Mapping[str, Any],
        *,
        config: OrpheusConfig,
    ) -> None:
        if type(self) is not _DevelopmentLedger or type(bindings) is not dict:
            raise PermissionError("development ledger requires exact nominal construction")
        _require_single_thread_execution()
        _consume_run_authorization(authorization, kind="development", bindings=bindings)
        self.path = development_ledger_path()
        self._bindings = copy.deepcopy(bindings)
        assert_rgbd_identifiable_drag_config(config)
        self._config = config
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
            "bindings": copy.deepcopy(self._bindings),
            "attempt_reserved": True,
            "access_started": True,
            "development_data_materialized": True,
            "active_ordinal": None,
            "materialized_ordinal_count": 0,
            "completed_ordinal_count": 0,
            "ordinal_evidence_sha256s": [],
            "active_batch_ordinals": None,
            "completed_batch_count": 0,
            "batch_evidence_sha256s": [],
            "result_sha256": None,
            "status": "development_materialization_started",
        }
        self._capability: _ManifestCapability | None = None
        self._capability_issued = False
        self._mint_identity = object()
        _durable_create(self.path, self._serialized())
        _refresh_ledger_receipt(self)
        _ledger_bound_source_guard(self, label="development ledger creation")
        _register_ledger_receipt(self)
        _LIVE_PRIVATE_LEDGERS[id(self)] = (
            self,
            self._mint_identity,
            authorization,
            None,
            canonical_sha256(self._bindings),
        )

    def _serialized(self, record: Mapping[str, Any] | None = None) -> bytes:
        value = self.record if record is None else record
        return json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    def _transition_record(
        self, *, expected_inventory: frozenset[str] | None = None
    ) -> dict[str, Any]:
        _validate_live_ledger_receipt(
            self,
            split="development",
            expected_inventory=expected_inventory,
        )
        return copy.deepcopy(self.record)

    def _replace(
        self,
        record: dict[str, Any],
        *,
        label: str,
        expected_inventory: frozenset[str] | None = None,
    ) -> None:
        if type(record) is not dict or record.get("bindings") != self._bindings:
            raise PermissionError("development transition changed immutable bindings")
        _validate_live_ledger_receipt(
            self,
            split="development",
            expected_inventory=expected_inventory,
        )
        _validate_ledger_record_shape(record, qualification=False)
        _ledger_bound_source_guard(self, label=f"{label} before write")
        next_generation = self._receipt_generation + 1
        contents = self._serialized(record)
        receipt_digest = sha256_bytes(contents)
        durable_record = copy.deepcopy(record)
        durable_record_sha256 = canonical_sha256(durable_record)
        bindings_sha256 = canonical_sha256(self._bindings)
        metadata = _durable_replace(self.path, contents)
        self.record = record
        self._receipt_digest = receipt_digest
        self._receipt_device = metadata.st_dev
        self._receipt_inode = metadata.st_ino
        self._durable_record = durable_record
        self._receipt_generation = next_generation
        _LIVE_LEDGER_RECEIPTS[id(self)] = (
            self,
            self.path,
            metadata.st_dev,
            metadata.st_ino,
            receipt_digest,
            durable_record_sha256,
            bindings_sha256,
            next_generation,
        )

    def capability(self) -> _ManifestCapability:
        _validate_live_ledger_receipt(self, split="development")
        if self._capability_issued or self._capability is not None:
            raise RuntimeError("development manifest capability cannot be minted twice")
        self._capability_issued = True
        self._capability = _ManifestCapability(
            _CAPABILITY_AUTHORITY,
            ledger=self,
            ledger_mint_identity=self._mint_identity,
            split="development",
        )
        return self._capability

    def _begin_ordinal(self, split: str, ordinal: int) -> None:
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record()
        if split != "development" or ordinal != record["materialized_ordinal_count"]:
            raise RuntimeError("development ordinal differs from durable receipt position")
        if record["active_ordinal"] is not None or record["status"] not in {
            "development_materialization_started",
            "development_ordinal_complete",
            "development_ordinal_constructed",
        }:
            raise RuntimeError("development ordinal boundary is not open")
        record["active_ordinal"] = ordinal
        record["status"] = "development_ordinal_materialization_started"
        self._replace(record, label=f"development ordinal {ordinal} receipt")

    def _begin_batch(self, split: str, ordinals: tuple[int, ...]) -> None:
        _require_exact_governed_batch(split, ordinals)
        record = self._transition_record()
        if (
            split != "development"
            or ordinals
            != tuple(
                range(
                    record["materialized_ordinal_count"],
                    record["materialized_ordinal_count"] + 4,
                )
            )
            or record["active_batch_ordinals"] is not None
            or record["active_ordinal"] is not None
            or record["materialized_ordinal_count"] != record["completed_ordinal_count"]
        ):
            raise RuntimeError("development batch boundary differs from durable receipt")
        record["active_batch_ordinals"] = list(ordinals)
        record["status"] = "development_batch_reserved"
        self._replace(record, label=f"development batch {ordinals[0] // 4} receipt")

    def _begin_batch_ordinal(self, split: str, ordinals: tuple[int, ...], ordinal: int) -> None:
        _require_exact_governed_batch(split, ordinals)
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record()
        if (
            split != "development"
            or record["active_batch_ordinals"] != list(ordinals)
            or ordinal != record["materialized_ordinal_count"]
            or ordinal not in ordinals
            or record["active_ordinal"] is not None
        ):
            raise RuntimeError("development batch constructor receipt differs")
        record["active_ordinal"] = ordinal
        record["status"] = "development_batch_ordinal_materialization_started"
        self._replace(record, label=f"development batch ordinal {ordinal} receipt")

    def _mark_batch_ordinal_constructed(
        self, split: str, ordinals: tuple[int, ...], ordinal: int
    ) -> None:
        _require_exact_governed_batch(split, ordinals)
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record()
        if (
            split != "development"
            or record["active_batch_ordinals"] != list(ordinals)
            or record["active_ordinal"] != ordinal
            or record["materialized_ordinal_count"] != ordinal
        ):
            raise RuntimeError("development batch constructor completion differs")
        record["materialized_ordinal_count"] = ordinal + 1
        record["active_ordinal"] = None
        record["status"] = "development_batch_ordinal_constructed"
        self._replace(record, label=f"development batch ordinal {ordinal} constructed")

    def _complete_batch(
        self,
        split: str,
        ordinals: tuple[int, ...],
        evidence_sha256s: tuple[str, ...],
    ) -> None:
        _require_exact_governed_batch(split, ordinals)
        record = self._transition_record()
        start = ordinals[0]
        if (
            split != "development"
            or record["active_batch_ordinals"] != list(ordinals)
            or record["active_ordinal"] is not None
            or record["materialized_ordinal_count"] != start + 4
            or record["completed_ordinal_count"] != start
            or len(evidence_sha256s) != 4
        ):
            raise RuntimeError("development batch evidence completion differs")
        record["ordinal_evidence_sha256s"].extend(evidence_sha256s)
        record["batch_evidence_sha256s"].append(canonical_sha256(list(evidence_sha256s)))
        record["completed_ordinal_count"] = start + 4
        record["completed_batch_count"] += 1
        record["active_batch_ordinals"] = None
        record["status"] = "development_batch_complete"
        self._replace(record, label=f"development batch {start // 4} completion")

    def _mark_ordinal_constructed(self, split: str, ordinal: int) -> None:
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record()
        if (
            split != "development"
            or record["status"] != "development_ordinal_materialization_started"
            or record["active_ordinal"] != ordinal
            or record["materialized_ordinal_count"] != ordinal
        ):
            raise RuntimeError("development constructor completion differs from durable receipt")
        record["materialized_ordinal_count"] = ordinal + 1
        record["active_ordinal"] = None
        record["status"] = "development_ordinal_constructed"
        self._replace(record, label=f"development ordinal {ordinal} constructor completion")

    def _complete_ordinal(self, split: str, ordinal: int, evidence_sha256: str) -> None:
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record()
        if (
            split != "development"
            or record["completed_ordinal_count"] != ordinal
            or record["materialized_ordinal_count"] <= ordinal
            or record["active_ordinal"] is not None
        ):
            raise RuntimeError("development ordinal completion differs from durable receipt")
        validated_sha256(evidence_sha256, label="development ordinal evidence")
        record["ordinal_evidence_sha256s"].append(evidence_sha256)
        record["completed_ordinal_count"] = ordinal + 1
        record["active_ordinal"] = None
        record["status"] = "development_ordinal_complete"
        self._replace(record, label=f"development ordinal {ordinal} completion")

    def complete_evaluation(self, result: Mapping[str, Any]) -> None:
        record = self._transition_record()
        if self._capability is None:
            raise RuntimeError("development capability was not issued")
        self._capability.require_finished()
        _validate_split_result(result, split="development")
        if (
            record["completed_ordinal_count"] != 64
            or record["materialized_ordinal_count"] != 64
            or record["active_ordinal"] is not None
            or record["active_batch_ordinals"] is not None
            or record["completed_batch_count"] != 16
            or len(record["batch_evidence_sha256s"]) != 16
        ):
            raise RuntimeError("development ledger did not record all 64 ordinals")
        _retire_manifest_capability(self._capability)
        record["result_sha256"] = canonical_sha256(dict(result))
        record["outcome"] = "passed" if result["passed"] else "failed"
        record["status"] = "development_artifacts_pending"
        self._replace(record, label="development evaluation completion")

    def finish(self, *, report_sha256: str, checkpoint_sha256: str | None) -> None:
        expected_inventory = (
            DEVELOPMENT_ARTIFACT_NAMES
            if checkpoint_sha256 is not None
            else frozenset({DEVELOPMENT_LEDGER_NAME, DEVELOPMENT_REPORT_NAME})
        )
        record = self._transition_record(expected_inventory=expected_inventory)
        if record["status"] != "development_artifacts_pending":
            raise RuntimeError("development artifacts were not durably pending")
        report_digest = validated_sha256(report_sha256, label="development report SHA-256")
        report_path = canonical_development_report_path()
        _require_single_link_regular(report_path, label="development terminal report")
        report_contents = stable_read_bytes(report_path, label="development terminal report")
        if sha256_bytes(report_contents) != report_digest:
            raise PermissionError("development terminal report differs from intended digest")
        report = _strict_json_loads(report_contents, label="development terminal report")
        source = self._bindings["source_provenance"]
        publication = self._bindings["publication_provenance"]
        certificate = _frozen_scene_certificate_binding()
        _development_report_is_valid(
            report,
            source=source,
            publication=publication,
            certificate=certificate,
        )
        passed = record["outcome"] == "passed"
        if report["passed"] is not passed:
            raise PermissionError("development terminal report outcome differs from ledger")
        if passed:
            checkpoint_digest = validated_sha256(
                checkpoint_sha256, label="development checkpoint SHA-256"
            )
            _validate_run_tree(DEVELOPMENT_ARTIFACT_NAMES, stage="development terminal artifacts")
            checkpoint_contents = stable_read_bytes(
                canonical_checkpoint_path(), label="development terminal checkpoint"
            )
            if sha256_bytes(checkpoint_contents) != checkpoint_digest:
                raise PermissionError(
                    "development terminal checkpoint differs from intended digest"
                )
            development, calibration = _validate_development_report(
                report,
                checkpoint_sha256=checkpoint_digest,
                source=source,
                publication=publication,
                certificate=certificate,
            )
            _validate_checkpoint_evidence(
                _checkpoint_payload_from_bytes(checkpoint_contents),
                config=self._config,
                source=source,
                publication=publication,
                development=development,
                calibration=calibration,
            )
        else:
            if checkpoint_sha256 is not None or _lexists(canonical_checkpoint_path()):
                raise PermissionError("failed development cannot publish a checkpoint")
            checkpoint_digest = None
            _validate_run_tree(
                frozenset({DEVELOPMENT_LEDGER_NAME, DEVELOPMENT_REPORT_NAME}),
                stage="failed development terminal artifacts",
            )
        record["report_sha256"] = report_digest
        record["checkpoint_sha256"] = checkpoint_digest
        record["status"] = "complete" if record["outcome"] == "passed" else "failed"
        _validate_terminal_development_ledger_record(
            record,
            report=report,
            report_sha256=report_digest,
            checkpoint_sha256=checkpoint_digest,
            bindings=self._bindings,
        )
        if passed:
            _validate_development_ledger_record(
                record,
                report=report,
                development=development,
                report_sha256=report_digest,
                checkpoint_sha256=checkpoint_digest,
                source=self._bindings["source_provenance"],
                publication=self._bindings["publication_provenance"],
            )
        self._replace(
            record,
            label="development terminal receipt",
            expected_inventory=expected_inventory,
        )
        _LIVE_PRIVATE_LEDGERS.pop(id(self), None)
        _LIVE_LEDGER_RECEIPTS.pop(id(self), None)

    def record_error(self, error: BaseException, *, report_sha256: str | None) -> None:
        _revoke_ledger_governed_access(self)
        expected_inventory = _exact_error_inventory(qualification=False)
        record = self._transition_record(expected_inventory=expected_inventory)
        prior_status = record.get("status")
        if type(prior_status) is not str:
            raise TypeError("development error prior status must be an exact string")
        error_record = {"type": type(error).__name__, "message": str(error)}
        failed: dict[str, Any] | None = None
        report_digest = (
            None
            if report_sha256 is None
            else validated_sha256(report_sha256, label="failed development report")
        )
        if report_digest is not None:
            contents = stable_read_bytes(
                canonical_development_report_path(), label="failed development report receipt"
            )
            if sha256_bytes(contents) != report_digest:
                raise PermissionError(
                    "failed development report digest differs before error ledger"
                )
            failed = _strict_json_loads(contents, label="failed development report receipt")
            _development_error_report_is_valid(
                failed,
                source=self._bindings["source_provenance"],
                publication=self._bindings["publication_provenance"],
                certificate=_frozen_scene_certificate_binding(),
            )
            if failed["error"] != error_record:
                raise PermissionError("development error report exception differs")
        record["status"] = "error"
        record["outcome"] = "failed"
        record["status_before_error"] = prior_status
        record["error"] = error_record
        record["report_sha256"] = report_digest
        record.setdefault("checkpoint_sha256", None)
        record["checkpoint_sha256"] = None
        _validate_development_error_ledger_record(
            record,
            report=failed,
            report_sha256=report_digest,
            bindings=self._bindings,
            prior_status=prior_status,
            error_record=error_record,
        )
        try:
            self._replace(
                record,
                label="development error receipt",
                expected_inventory=expected_inventory,
            )
        finally:
            _LIVE_PRIVATE_LEDGERS.pop(id(self), None)
            _LIVE_LEDGER_RECEIPTS.pop(id(self), None)


class _QualificationLedger:
    """Exclusive selector -> confirmation -> final-test protected receipt."""

    ARTIFACT_KIND = "rgbd_identifiable_drag_exactly_once_access_ledger"
    ORDER = ("selector", "confirmation", "final_test")

    def __init__(
        self,
        authorization: _RunAuthorization,
        reviewed_seal: _ReviewedDevelopmentSeal,
        bindings: Mapping[str, Any],
        *,
        config: OrpheusConfig,
    ) -> None:
        if type(self) is not _QualificationLedger or type(bindings) is not dict:
            raise PermissionError("qualification ledger requires exact nominal construction")
        _require_single_thread_execution()
        # The ledger itself, not merely the wrapper, re-reads and validates all
        # three externally reviewed development artifacts before consuming its
        # one-shot authorization.
        reviewed = _validate_reviewed_bundle_from_disk(
            config=config,
            source=bindings.get("source_provenance"),
            publication=bindings.get("publication_provenance"),
            reviewed_checkpoint_sha256=bindings.get("reviewed_checkpoint_sha256"),
            reviewed_report_sha256=bindings.get("reviewed_development_report_sha256"),
            reviewed_ledger_sha256=bindings.get("reviewed_development_ledger_sha256"),
        )
        if reviewed["model_state_sha256"] != bindings.get("model_state_sha256") or reviewed[
            "calibration_sha256"
        ] != bindings.get("calibration_sha256"):
            raise PermissionError("ledger review result differs from protected bindings")
        _consume_run_authorization(
            authorization,
            kind="qualification",
            bindings=bindings,
            reviewed_seal=reviewed_seal,
        )
        self.path = qualification_ledger_path()
        self._bindings = copy.deepcopy(bindings)
        self._config = config
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
            "order": list(self.ORDER),
            "bindings": copy.deepcopy(self._bindings),
            "splits": {
                split: {
                    "access_started": False,
                    "status": "unopened",
                    "result_sha256": None,
                    "completed_ordinal_count": 0,
                    "materialized_ordinal_count": 0,
                    "active_ordinal": None,
                    "ordinal_evidence_sha256s": [],
                    "active_batch_ordinals": None,
                    "completed_batch_count": 0,
                    "batch_evidence_sha256s": [],
                }
                for split in self.ORDER
            },
            "attempt_reserved": True,
            "protected_data_materialized": False,
            "status": "reserved_before_protected_access",
        }
        self._capabilities: dict[str, _ManifestCapability] = {}
        self._reviewed_seal = reviewed_seal
        self._mint_identity = object()
        _durable_create(self.path, self._serialized())
        _refresh_ledger_receipt(self)
        _ledger_bound_source_guard(self, label="qualification ledger creation")
        _bind_reviewed_development_seal(reviewed_seal, bindings=bindings, ledger=self)
        _register_ledger_receipt(self)
        _LIVE_PRIVATE_LEDGERS[id(self)] = (
            self,
            self._mint_identity,
            authorization,
            reviewed_seal,
            canonical_sha256(self._bindings),
        )

    def _serialized(self, record: Mapping[str, Any] | None = None) -> bytes:
        value = self.record if record is None else record
        return json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    def _transition_record(
        self,
        *,
        split: str,
        expected_inventory: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        _validate_live_ledger_receipt(
            self,
            split=split,
            expected_inventory=expected_inventory,
        )
        return copy.deepcopy(self.record)

    def _replace(
        self,
        record: dict[str, Any],
        *,
        label: str,
        expected_inventory: frozenset[str] | None = None,
    ) -> None:
        if type(record) is not dict or record.get("bindings") != self._bindings:
            raise PermissionError("qualification transition changed immutable bindings")
        _validate_live_ledger_receipt(
            self,
            split="selector",
            expected_inventory=expected_inventory,
        )
        _validate_ledger_record_shape(record, qualification=True)
        _ledger_bound_source_guard(self, label=f"{label} before write")
        next_generation = self._receipt_generation + 1
        contents = self._serialized(record)
        receipt_digest = sha256_bytes(contents)
        durable_record = copy.deepcopy(record)
        durable_record_sha256 = canonical_sha256(durable_record)
        bindings_sha256 = canonical_sha256(self._bindings)
        metadata = _durable_replace(self.path, contents)
        self.record = record
        self._receipt_digest = receipt_digest
        self._receipt_device = metadata.st_dev
        self._receipt_inode = metadata.st_ino
        self._durable_record = durable_record
        self._receipt_generation = next_generation
        _LIVE_LEDGER_RECEIPTS[id(self)] = (
            self,
            self.path,
            metadata.st_dev,
            metadata.st_ino,
            receipt_digest,
            durable_record_sha256,
            bindings_sha256,
            next_generation,
        )

    def begin_access(self, split: str) -> _ManifestCapability:
        record = self._transition_record(split=split)
        if split not in self.ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        index = self.ORDER.index(split)
        states = record["splits"]
        if any(states[prior]["status"] != "passed" for prior in self.ORDER[:index]):
            raise RuntimeError(f"{split} must remain unopened until every predecessor passes")
        state = states[split]
        if state["status"] != "unopened" or state["access_started"] is not False:
            raise RuntimeError(f"protected split {split!r} cannot be opened twice")
        if any(states[later]["access_started"] for later in self.ORDER[index + 1 :]):
            raise RuntimeError("protected split order is inconsistent")
        state["access_started"] = True
        state["status"] = "materialization_started"
        record["protected_data_materialized"] = True
        record["status"] = f"{split}_materialization_started"
        self._replace(record, label=f"{split} access receipt")
        capability = _ManifestCapability(
            _CAPABILITY_AUTHORITY,
            ledger=self,
            ledger_mint_identity=self._mint_identity,
            split=split,
        )
        if split in self._capabilities:
            raise RuntimeError(f"protected split {split!r} capability already exists")
        self._capabilities[split] = capability
        return capability

    def _begin_ordinal(self, split: str, ordinal: int) -> None:
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record(split=split)
        if split not in self.ORDER:
            raise ValueError("protected ordinal split is unknown")
        state = record["splits"][split]
        if (
            ordinal != state["materialized_ordinal_count"]
            or state["active_ordinal"] is not None
            or state["status"]
            not in {"materialization_started", "ordinal_complete", "ordinal_constructed"}
        ):
            raise RuntimeError("protected ordinal differs from durable receipt position")
        state["active_ordinal"] = ordinal
        state["status"] = "ordinal_materialization_started"
        record["status"] = f"{split}_ordinal_{ordinal}_materialization_started"
        self._replace(record, label=f"{split} ordinal {ordinal} receipt")

    def _begin_batch(self, split: str, ordinals: tuple[int, ...]) -> None:
        _require_exact_governed_batch(split, ordinals)
        record = self._transition_record(split=split)
        state = record["splits"][split]
        if (
            split not in self.ORDER
            or ordinals
            != tuple(
                range(state["materialized_ordinal_count"], state["materialized_ordinal_count"] + 4)
            )
            or state["active_batch_ordinals"] is not None
            or state["active_ordinal"] is not None
            or state["materialized_ordinal_count"] != state["completed_ordinal_count"]
        ):
            raise RuntimeError("protected batch boundary differs from durable receipt")
        state["active_batch_ordinals"] = list(ordinals)
        state["status"] = "batch_reserved"
        record["status"] = f"{split}_batch_{ordinals[0] // 4}_reserved"
        self._replace(record, label=f"{split} batch {ordinals[0] // 4} receipt")

    def _begin_batch_ordinal(self, split: str, ordinals: tuple[int, ...], ordinal: int) -> None:
        _require_exact_governed_batch(split, ordinals)
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record(split=split)
        state = record["splits"][split]
        if (
            state["active_batch_ordinals"] != list(ordinals)
            or ordinal != state["materialized_ordinal_count"]
            or ordinal not in ordinals
            or state["active_ordinal"] is not None
        ):
            raise RuntimeError("protected batch constructor receipt differs")
        state["active_ordinal"] = ordinal
        state["status"] = "batch_ordinal_materialization_started"
        record["status"] = f"{split}_batch_ordinal_{ordinal}_materialization_started"
        self._replace(record, label=f"{split} batch ordinal {ordinal} receipt")

    def _mark_batch_ordinal_constructed(
        self, split: str, ordinals: tuple[int, ...], ordinal: int
    ) -> None:
        _require_exact_governed_batch(split, ordinals)
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record(split=split)
        state = record["splits"][split]
        if (
            state["active_batch_ordinals"] != list(ordinals)
            or state["active_ordinal"] != ordinal
            or state["materialized_ordinal_count"] != ordinal
        ):
            raise RuntimeError("protected batch constructor completion differs")
        state["materialized_ordinal_count"] = ordinal + 1
        state["active_ordinal"] = None
        state["status"] = "batch_ordinal_constructed"
        record["status"] = f"{split}_batch_ordinal_{ordinal}_constructed"
        self._replace(record, label=f"{split} batch ordinal {ordinal} constructed")

    def _complete_batch(
        self,
        split: str,
        ordinals: tuple[int, ...],
        evidence_sha256s: tuple[str, ...],
    ) -> None:
        _require_exact_governed_batch(split, ordinals)
        record = self._transition_record(split=split)
        state = record["splits"][split]
        start = ordinals[0]
        if (
            state["active_batch_ordinals"] != list(ordinals)
            or state["active_ordinal"] is not None
            or state["materialized_ordinal_count"] != start + 4
            or state["completed_ordinal_count"] != start
            or len(evidence_sha256s) != 4
        ):
            raise RuntimeError("protected batch evidence completion differs")
        state["ordinal_evidence_sha256s"].extend(evidence_sha256s)
        state["batch_evidence_sha256s"].append(canonical_sha256(list(evidence_sha256s)))
        state["completed_ordinal_count"] = start + 4
        state["completed_batch_count"] += 1
        state["active_batch_ordinals"] = None
        state["status"] = "batch_complete"
        record["status"] = f"{split}_batch_{start // 4}_complete"
        self._replace(record, label=f"{split} batch {start // 4} completion")

    def _mark_ordinal_constructed(self, split: str, ordinal: int) -> None:
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record(split=split)
        state = record["splits"][split]
        if (
            state["status"] != "ordinal_materialization_started"
            or state["active_ordinal"] != ordinal
            or state["materialized_ordinal_count"] != ordinal
        ):
            raise RuntimeError("protected constructor completion differs from durable receipt")
        state["materialized_ordinal_count"] = ordinal + 1
        state["active_ordinal"] = None
        state["status"] = "ordinal_constructed"
        record["status"] = f"{split}_ordinal_{ordinal}_constructed"
        self._replace(record, label=f"{split} ordinal {ordinal} constructor completion")

    def _complete_ordinal(self, split: str, ordinal: int, evidence_sha256: str) -> None:
        _require_exact_governed_ordinal(split, ordinal)
        record = self._transition_record(split=split)
        state = record["splits"][split]
        if (
            state["completed_ordinal_count"] != ordinal
            or state["materialized_ordinal_count"] <= ordinal
            or state["active_ordinal"] is not None
        ):
            raise RuntimeError("protected ordinal completion differs from durable receipt")
        validated_sha256(evidence_sha256, label=f"{split} ordinal evidence")
        state["ordinal_evidence_sha256s"].append(evidence_sha256)
        state["completed_ordinal_count"] = ordinal + 1
        state["active_ordinal"] = None
        state["status"] = "ordinal_complete"
        record["status"] = f"{split}_ordinal_{ordinal}_complete"
        self._replace(record, label=f"{split} ordinal {ordinal} completion")

    def complete_split(self, split: str, result: Mapping[str, Any]) -> None:
        record = self._transition_record(split=split)
        if split not in self.ORDER:
            raise ValueError("protected completion split is unknown")
        capability = self._capabilities.get(split)
        if capability is None:
            raise RuntimeError(f"protected split {split!r} lacks a capability")
        capability.require_finished()
        _validate_split_result(result, split=split)
        state = record["splits"][split]
        if (
            state["completed_ordinal_count"] != 64
            or state["materialized_ordinal_count"] != 64
            or state["active_ordinal"] is not None
            or state["active_batch_ordinals"] is not None
            or state["completed_batch_count"] != 16
            or len(state["batch_evidence_sha256s"]) != 16
        ):
            raise RuntimeError(f"protected split {split!r} did not record 64 ordinals")
        _retire_manifest_capability(capability)
        state["result_sha256"] = canonical_sha256(dict(result))
        state["status"] = "passed" if result["passed"] else "failed"
        record["status"] = f"{split}_{state['status']}"
        self._replace(record, label=f"{split} split completion")

    def prepare_report(self, *, passed: bool, stopped_after: str) -> None:
        record = self._transition_record(split="selector")
        if type(passed) is not bool or type(stopped_after) is not str:
            raise TypeError("qualification report preparation requires exact outcome values")
        if passed and any(record["splits"][split]["status"] != "passed" for split in self.ORDER):
            raise RuntimeError("qualification cannot pass before every protected split passes")
        opened = [
            split for split in self.ORDER if record["splits"][split]["access_started"] is True
        ]
        expected_stopped_after = opened[-1] if opened else "reviewed_development"
        if stopped_after != expected_stopped_after:
            raise ValueError("qualification stopped_after differs from last opened split")
        expected_passed = len(opened) == 3 and all(
            record["splits"][split]["status"] == "passed" for split in self.ORDER
        )
        if passed is not expected_passed:
            raise ValueError("qualification prepared outcome differs from durable split states")
        if not passed and (
            not opened
            or record["splits"][opened[-1]]["status"] != "failed"
            or any(record["splits"][split]["status"] != "passed" for split in opened[:-1])
        ):
            raise ValueError(
                "failed qualification must stop on the exact first failed protected split"
            )
        record["outcome"] = "passed" if passed else "failed"
        record["stopped_after"] = stopped_after
        record["status"] = "qualification_report_write_pending"
        self._replace(record, label="qualification report pending receipt")

    def finish(self, *, report_sha256: str) -> None:
        record = self._transition_record(
            split="selector",
            expected_inventory=QUALIFICATION_ARTIFACT_NAMES,
        )
        if record["status"] != "qualification_report_write_pending":
            raise RuntimeError("qualification report was not durably pending")
        report_digest = validated_sha256(report_sha256, label="qualification report SHA-256")
        _validate_run_tree(QUALIFICATION_ARTIFACT_NAMES, stage="qualification terminal artifacts")
        report_path = canonical_qualification_report_path()
        contents = stable_read_bytes(report_path, label="qualification terminal report")
        if sha256_bytes(contents) != report_digest:
            raise PermissionError("qualification terminal report differs from intended digest")
        report = _strict_json_loads(contents, label="qualification terminal report")
        source = self._bindings["source_provenance"]
        publication = self._bindings["publication_provenance"]
        certificate = _frozen_scene_certificate_binding()
        _qualification_report_is_valid(
            report,
            source=source,
            publication=publication,
            certificate=certificate,
        )
        if report["passed"] is not (record["outcome"] == "passed"):
            raise PermissionError("qualification report outcome differs from ledger")
        for report_key, binding in (
            ("reviewed_checkpoint_sha256", "reviewed_checkpoint_sha256"),
            ("reviewed_development_report_sha256", "reviewed_development_report_sha256"),
            ("reviewed_development_ledger_sha256", "reviewed_development_ledger_sha256"),
            ("model_state_sha256", "model_state_sha256"),
        ):
            if report[report_key] != self._bindings[binding]:
                raise PermissionError(f"qualification terminal {report_key} differs")
        reviewed = _validate_reviewed_bundle_from_disk(
            config=self._config,
            source=source,
            publication=publication,
            reviewed_checkpoint_sha256=self._bindings["reviewed_checkpoint_sha256"],
            reviewed_report_sha256=self._bindings["reviewed_development_report_sha256"],
            reviewed_ledger_sha256=self._bindings["reviewed_development_ledger_sha256"],
            expected_inventory=QUALIFICATION_ARTIFACT_NAMES,
        )
        if (
            report["development"] != reviewed["development"]
            or report["calibration"] != reviewed["calibration"]
            or canonical_sha256(report["calibration"]) != self._bindings["calibration_sha256"]
        ):
            raise PermissionError(
                "qualification terminal embedded development differs from reviewed bytes"
            )
        record["report_sha256"] = report_digest
        record["status"] = "complete" if record["outcome"] == "passed" else "failed"
        _validate_qualification_ledger_record(
            record,
            report=report,
            report_sha256=report_digest,
            bindings=self._bindings,
        )
        self._replace(
            record,
            label="qualification terminal receipt",
            expected_inventory=QUALIFICATION_ARTIFACT_NAMES,
        )
        _LIVE_PRIVATE_LEDGERS.pop(id(self), None)
        _LIVE_LEDGER_RECEIPTS.pop(id(self), None)
        _LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(self._reviewed_seal), None)

    def record_error(
        self, error: BaseException, *, stopped_after: str, report_sha256: str | None
    ) -> None:
        _revoke_ledger_governed_access(self)
        expected_inventory = _exact_error_inventory(qualification=True)
        record = self._transition_record(
            split="selector",
            expected_inventory=expected_inventory,
        )
        prior_status = record.get("status")
        if type(prior_status) is not str or type(stopped_after) is not str:
            raise TypeError("qualification error state requires exact strings")
        started = [
            split for split in self.ORDER if record["splits"][split]["access_started"] is True
        ]
        expected_stopped = started[-1] if started else "reviewed_development"
        if stopped_after != expected_stopped:
            raise ValueError("qualification error stopped_after differs from durable access")
        error_record = {"type": type(error).__name__, "message": str(error)}
        failed: dict[str, Any] | None = None
        report_digest = (
            None
            if report_sha256 is None
            else validated_sha256(report_sha256, label="failed qualification report")
        )
        if report_digest is not None:
            contents = stable_read_bytes(
                canonical_qualification_report_path(),
                label="failed qualification report receipt",
            )
            if sha256_bytes(contents) != report_digest:
                raise PermissionError(
                    "failed qualification report digest differs before error ledger"
                )
            failed = _strict_json_loads(contents, label="failed qualification report receipt")
            reviewed = _validate_reviewed_bundle_from_disk(
                config=self._config,
                source=self._bindings["source_provenance"],
                publication=self._bindings["publication_provenance"],
                reviewed_checkpoint_sha256=self._bindings["reviewed_checkpoint_sha256"],
                reviewed_report_sha256=self._bindings["reviewed_development_report_sha256"],
                reviewed_ledger_sha256=self._bindings["reviewed_development_ledger_sha256"],
                expected_inventory=expected_inventory,
            )
            _qualification_error_report_is_valid(
                failed,
                source=self._bindings["source_provenance"],
                publication=self._bindings["publication_provenance"],
                certificate=_frozen_scene_certificate_binding(),
                bindings=self._bindings,
                reviewed=reviewed,
                ledger_record=record,
            )
            if failed["error"] != error_record:
                raise PermissionError("qualification error report exception differs")
        record["status"] = "error"
        record["outcome"] = "failed"
        record["status_before_error"] = prior_status
        record["stopped_after"] = stopped_after
        record["error"] = error_record
        record["report_sha256"] = report_digest
        _validate_qualification_error_ledger_record(
            record,
            report=failed,
            report_sha256=report_digest,
            bindings=self._bindings,
            prior_status=prior_status,
            stopped_after=stopped_after,
            error_record=error_record,
        )
        try:
            self._replace(
                record,
                label="qualification error receipt",
                expected_inventory=expected_inventory,
            )
        finally:
            _LIVE_PRIVATE_LEDGERS.pop(id(self), None)
            _LIVE_LEDGER_RECEIPTS.pop(id(self), None)
            _LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(self._reviewed_seal), None)


def _validate_checkpoint_evidence(
    payload: Any,
    *,
    config: OrpheusConfig,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    development: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[dict[str, Tensor], str]:
    _require_exact_keys(payload, CHECKPOINT_SCHEMA, label="reviewed checkpoint")
    state = _validate_checkpoint_model_state(payload["model_state"])
    state_sha256 = _state_dict_sha256(state)
    if type(payload["step"]) is not int or payload["step"] != 0:
        raise ValueError("reviewed checkpoint must be exact step zero")
    if payload["optimizer_state"] is not None or payload["scheduler_state"] is not None:
        raise ValueError("reviewed checkpoint must contain no optimizer/scheduler state")
    if (
        payload["project_version"] != __version__
        or payload["specification_version"] != SPECIFICATION_VERSION
        or payload["simulator_version"] != SIMULATOR_VERSION
        or payload["device"] != "cpu"
        or payload["precision"] != "float32"
    ):
        raise ValueError("reviewed checkpoint versions/device/precision differ")
    if canonical_sha256(payload["config"]) != canonical_sha256(config.to_dict()):
        raise ValueError("reviewed checkpoint resolved config differs")
    if canonical_sha256(payload["git"]) != canonical_sha256(source):
        raise ValueError("reviewed checkpoint source differs")
    metrics = payload["metrics"]
    _require_exact_keys(metrics, CHECKPOINT_METRICS_SCHEMA, label="checkpoint metrics")
    expected_metrics = {
        "artifact_kind": "rgbd_identifiable_drag_three_scale_state",
        "optimizer_updates": 0,
        "optimizer_state_entry_count": 0,
        "rng_state_entry_count": 0,
        "model_state_sha256": state_sha256,
        "protocol": bridge_protocol(),
        "publication_provenance": dict(publication),
        "calibration": dict(calibration),
        "development": dict(development),
    }
    if canonical_sha256(metrics) != canonical_sha256(expected_metrics):
        raise ValueError("reviewed checkpoint metric evidence differs")
    _validate_calibration(calibration)
    _validate_split_calibration_binding(
        development,
        calibration,
        expected_state_sha256=calibration["calibrated_model_state_sha256"],
        label="reviewed development",
    )
    if calibration["calibrated_model_state_sha256"] != state_sha256:
        raise ValueError("reviewed checkpoint state differs from calibrated state hash")
    for leaf, component in zip(_SCALE_STATE_LEAVES, ("position", "velocity", "drag"), strict=True):
        tensor = state[f"observation_modules.rgbd.{leaf}"]
        expected_bits = int(calibration[component]["deployed_float32_bits"][2:], 16)
        if _float32_bits(tensor) != expected_bits:
            raise ValueError(f"reviewed checkpoint {component} scale differs from calibration bits")
    roundtrip = new_public_model(config)
    roundtrip.load_state_dict(state, strict=True)
    if _model_state_sha256(roundtrip) != state_sha256:
        raise RuntimeError("reviewed checkpoint strict roundtrip changed model state")
    if tuple(parameter for parameter in roundtrip.parameters() if parameter.requires_grad):
        raise RuntimeError("reviewed checkpoint roundtrip introduced learned parameters")
    return {name: tensor.detach().clone() for name, tensor in state.items()}, state_sha256


def _save_review_checkpoint(
    path: Path,
    *,
    model: OnlineWorldModel,
    config: OrpheusConfig,
    development: Mapping[str, Any],
    calibration: Mapping[str, Any],
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> None:
    _require_canonical_path(path, canonical_checkpoint_path(), label="checkpoint")
    _validate_run_tree(frozenset({DEVELOPMENT_LEDGER_NAME}), stage="pre-checkpoint write")
    if _lexists(path) or _lexists(_atomic_temporary(path)):
        raise FileExistsError("reviewed checkpoint and temporary path must both be fresh")
    state_sha256 = _model_state_sha256(model)
    metrics = {
        "artifact_kind": "rgbd_identifiable_drag_three_scale_state",
        "optimizer_updates": 0,
        "optimizer_state_entry_count": 0,
        "rng_state_entry_count": 0,
        "model_state_sha256": state_sha256,
        "protocol": bridge_protocol(),
        "publication_provenance": dict(publication),
        "calibration": dict(calibration),
        "development": dict(development),
    }
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
    _validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        publication=publication,
        development=development,
        calibration=calibration,
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    contents = buffer.getvalue()
    if len(contents) > MAX_CHECKPOINT_BYTES:
        raise RuntimeError("review checkpoint exceeds frozen size bound")
    # Restricted load before persistence proves no unsafe pickle globals are
    # needed by the exact payload.
    roundtrip_payload = _checkpoint_payload_from_bytes(contents)
    _validate_checkpoint_evidence(
        roundtrip_payload,
        config=config,
        source=source,
        publication=publication,
        development=development,
        calibration=calibration,
    )
    _durable_create(path, contents)


DEVELOPMENT_LEDGER_COMPLETE_SCHEMA = frozenset(
    {
        "artifact_kind",
        "architecture_attempt",
        "maximum_architecture_attempts",
        "bindings",
        "attempt_reserved",
        "access_started",
        "development_data_materialized",
        "active_ordinal",
        "completed_ordinal_count",
        "materialized_ordinal_count",
        "ordinal_evidence_sha256s",
        "active_batch_ordinals",
        "completed_batch_count",
        "batch_evidence_sha256s",
        "result_sha256",
        "status",
        "outcome",
        "report_sha256",
        "checkpoint_sha256",
    }
)
QUALIFICATION_LEDGER_COMPLETE_SCHEMA = frozenset(
    {
        "artifact_kind",
        "architecture_attempt",
        "maximum_architecture_attempts",
        "order",
        "bindings",
        "splits",
        "attempt_reserved",
        "protected_data_materialized",
        "status",
        "outcome",
        "stopped_after",
        "report_sha256",
    }
)


def _strict_json_loads(contents: bytes, *, label: str) -> dict[str, Any]:
    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(contents, object_pairs_hook=exact_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not exact UTF-8 JSON") from error
    if type(parsed) is not dict:
        raise TypeError(f"{label} root must be an exact JSON object")
    return parsed


def _validate_development_ledger_record(
    record: Any,
    *,
    report: Mapping[str, Any],
    development: Mapping[str, Any],
    report_sha256: str,
    checkpoint_sha256: str,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        record, DEVELOPMENT_LEDGER_COMPLETE_SCHEMA, label="reviewed development ledger"
    )
    expected_bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": dict(source),
        "publication_provenance": dict(publication),
        "config_sha256": FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "development_manifest_sha256": MANIFEST_SHA256["development"],
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    if (
        record["artifact_kind"] != _DevelopmentLedger.ARTIFACT_KIND
        or type(record["architecture_attempt"]) is not int
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or type(record["maximum_architecture_attempts"]) is not int
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["attempt_reserved"] is not True
        or record["access_started"] is not True
        or record["development_data_materialized"] is not True
        or record["active_ordinal"] is not None
        or record["active_batch_ordinals"] is not None
        or type(record["completed_ordinal_count"]) is not int
        or record["completed_ordinal_count"] != 64
        or type(record["materialized_ordinal_count"]) is not int
        or record["materialized_ordinal_count"] != 64
        or type(record["completed_batch_count"]) is not int
        or record["completed_batch_count"] != 16
        or record["status"] != "complete"
        or record["outcome"] != "passed"
    ):
        raise ValueError("reviewed development ledger is not exactly terminal/passed")
    if record["bindings"] != expected_bindings:
        raise ValueError("reviewed development ledger bindings differ")
    hashes = record["ordinal_evidence_sha256s"]
    if type(hashes) is not list or len(hashes) != 64:
        raise ValueError("reviewed development ledger lacks 64 ordinal receipts")
    for index, digest in enumerate(hashes):
        validated_sha256(digest, label=f"development ordinal {index} evidence")
    if len(set(hashes)) != 64:
        raise ValueError("reviewed development ordinal evidence hashes are not unique")
    batch_hashes = record["batch_evidence_sha256s"]
    if type(batch_hashes) is not list or len(batch_hashes) != 16:
        raise ValueError("reviewed development ledger lacks 16 batch receipts")
    for batch_index, digest in enumerate(batch_hashes):
        validated_sha256(digest, label=f"development batch {batch_index} evidence")
        if digest != canonical_sha256(hashes[4 * batch_index : 4 * batch_index + 4]):
            raise ValueError("development batch evidence hash differs from ordinal receipts")
    if record["result_sha256"] != canonical_sha256(dict(development)):
        raise ValueError("reviewed development ledger result hash differs")
    if record["report_sha256"] != report_sha256 or record["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("reviewed development ledger artifact hashes differ")
    if report["development_ledger"] != str(development_ledger_path()):
        raise ValueError("reviewed development report names the wrong fixed ledger")


def _validate_terminal_development_ledger_record(
    record: Any,
    *,
    report: Mapping[str, Any],
    report_sha256: str,
    checkpoint_sha256: str | None,
    bindings: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        record,
        DEVELOPMENT_LEDGER_COMPLETE_SCHEMA,
        label="terminal development ledger",
    )
    _validate_ledger_record_shape(record, qualification=False)
    passed = report["passed"] is True
    if (
        record["artifact_kind"] != _DevelopmentLedger.ARTIFACT_KIND
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["bindings"] != dict(bindings)
        or record["attempt_reserved"] is not True
        or record["access_started"] is not True
        or record["development_data_materialized"] is not True
        or record["active_ordinal"] is not None
        or record["active_batch_ordinals"] is not None
        or record["completed_ordinal_count"] != 64
        or record["materialized_ordinal_count"] != 64
        or record["completed_batch_count"] != 16
        or record["result_sha256"] != canonical_sha256(dict(report["development"]))
        or record["outcome"] != ("passed" if passed else "failed")
        or record["status"] != ("complete" if passed else "failed")
        or record["report_sha256"] != report_sha256
        or record["checkpoint_sha256"] != checkpoint_sha256
        or report["development_ledger"] != str(development_ledger_path())
    ):
        raise ValueError("terminal development ledger differs from report/bindings")
    ordinal_hashes = record["ordinal_evidence_sha256s"]
    batch_hashes = record["batch_evidence_sha256s"]
    if len(ordinal_hashes) != 64 or len(set(ordinal_hashes)) != 64 or len(batch_hashes) != 16:
        raise ValueError("terminal development ledger evidence receipts differ")
    for ordinal, digest in enumerate(ordinal_hashes):
        validated_sha256(digest, label=f"terminal development ordinal {ordinal}")
    for batch_index, digest in enumerate(batch_hashes):
        validated_sha256(digest, label=f"terminal development batch {batch_index}")
        if digest != canonical_sha256(ordinal_hashes[4 * batch_index : 4 * batch_index + 4]):
            raise ValueError("terminal development batch receipt differs from ordinals")


def _validate_development_error_ledger_record(
    record: Any,
    *,
    report: Mapping[str, Any] | None,
    report_sha256: str | None,
    bindings: Mapping[str, Any],
    prior_status: str,
    error_record: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        record,
        frozenset({*DEVELOPMENT_LEDGER_COMPLETE_SCHEMA, "status_before_error", "error"}),
        label="development error ledger",
    )
    _validate_ledger_record_shape(record, qualification=False)
    expected_result = (
        None
        if report is None or report["development"] is None
        else canonical_sha256(dict(report["development"]))
    )
    if report is None and record["result_sha256"] is not None:
        validated_sha256(record["result_sha256"], label="development error durable result")
    if (
        record["artifact_kind"] != _DevelopmentLedger.ARTIFACT_KIND
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["bindings"] != dict(bindings)
        or record["attempt_reserved"] is not True
        or record["access_started"] is not True
        or record["development_data_materialized"] is not True
        or record["status"] != "error"
        or record["outcome"] != "failed"
        or record["status_before_error"] != prior_status
        or record["error"] != dict(error_record)
        or record["report_sha256"] != report_sha256
        or record["checkpoint_sha256"] is not None
        or (report is not None and record["result_sha256"] != expected_result)
    ):
        raise ValueError("development error ledger semantics differ")


def _validate_qualification_error_ledger_record(
    record: Any,
    *,
    report: Mapping[str, Any] | None,
    report_sha256: str | None,
    bindings: Mapping[str, Any],
    prior_status: str,
    stopped_after: str,
    error_record: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        record,
        frozenset({*QUALIFICATION_LEDGER_COMPLETE_SCHEMA, "status_before_error", "error"}),
        label="qualification error ledger",
    )
    _validate_ledger_record_shape(record, qualification=True)
    started: list[str] = []
    encountered_unopened = False
    for split in _QualificationLedger.ORDER:
        state = record["splits"][split]
        if state["access_started"] is True:
            if encountered_unopened:
                raise ValueError("qualification error ledger opened a later split")
            started.append(split)
        elif state["access_started"] is False:
            encountered_unopened = True
        else:
            raise TypeError("qualification error ledger access flag must be exact bool")
        if report is not None:
            result = report[split]
            expected_result = None if result is None else canonical_sha256(dict(result))
            if state["result_sha256"] != expected_result:
                raise ValueError("qualification error ledger result differs from report")
    expected_stopped = started[-1] if started else "reviewed_development"
    if (
        record["artifact_kind"] != _QualificationLedger.ARTIFACT_KIND
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["order"] != list(_QualificationLedger.ORDER)
        or record["bindings"] != dict(bindings)
        or record["attempt_reserved"] is not True
        or record["protected_data_materialized"] is not bool(started)
        or record["status"] != "error"
        or record["outcome"] != "failed"
        or record["status_before_error"] != prior_status
        or record["stopped_after"] != expected_stopped
        or stopped_after != expected_stopped
        or record["error"] != dict(error_record)
        or record["report_sha256"] != report_sha256
    ):
        raise ValueError("qualification error ledger semantics differ")


def _validate_qualification_ledger_record(
    record: Any,
    *,
    report: Mapping[str, Any],
    report_sha256: str,
    bindings: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        record,
        QUALIFICATION_LEDGER_COMPLETE_SCHEMA,
        label="terminal qualification ledger",
    )
    if (
        record["artifact_kind"] != _QualificationLedger.ARTIFACT_KIND
        or type(record["architecture_attempt"]) is not int
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or type(record["maximum_architecture_attempts"]) is not int
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["order"] != list(_QualificationLedger.ORDER)
        or record["bindings"] != dict(bindings)
        or record["attempt_reserved"] is not True
        or record["protected_data_materialized"] is not True
        or record["report_sha256"] != report_sha256
        or record["outcome"] not in {"passed", "failed"}
        or record["status"] not in {"complete", "failed"}
        or record["stopped_after"] != report["stopped_after"]
        or report["passed"] is not (record["outcome"] == "passed")
    ):
        raise ValueError("terminal qualification ledger root differs from report/bindings")
    _require_exact_keys(
        record["splits"],
        frozenset(_QualificationLedger.ORDER),
        label="terminal qualification splits",
    )
    encountered_stop = False
    opened: list[str] = []
    for split in _QualificationLedger.ORDER:
        state = record["splits"][split]
        _require_exact_keys(
            state, LEDGER_SPLIT_STATE_SCHEMA, label=f"terminal {split} ledger state"
        )
        result = report[split]
        if result is None:
            encountered_stop = True
            if state != {
                "access_started": False,
                "status": "unopened",
                "result_sha256": None,
                "completed_ordinal_count": 0,
                "materialized_ordinal_count": 0,
                "active_ordinal": None,
                "ordinal_evidence_sha256s": [],
                "active_batch_ordinals": None,
                "completed_batch_count": 0,
                "batch_evidence_sha256s": [],
            }:
                raise ValueError(f"terminal unopened {split} ledger state changed")
            continue
        if encountered_stop:
            raise ValueError("terminal qualification opened a later split after stop")
        _validate_split_result(result, split=split)
        opened.append(split)
        expected_status = "passed" if result["passed"] else "failed"
        if (
            state["access_started"] is not True
            or state["status"] != expected_status
            or state["result_sha256"] != canonical_sha256(dict(result))
            or state["completed_ordinal_count"] != 64
            or state["materialized_ordinal_count"] != 64
            or state["active_ordinal"] is not None
            or state["active_batch_ordinals"] is not None
            or state["completed_batch_count"] != 16
        ):
            raise ValueError(f"terminal {split} ledger completion differs from result")
        ordinal_hashes = state["ordinal_evidence_sha256s"]
        batch_hashes = state["batch_evidence_sha256s"]
        if (
            type(ordinal_hashes) is not list
            or len(ordinal_hashes) != 64
            or len(set(ordinal_hashes)) != 64
            or type(batch_hashes) is not list
            or len(batch_hashes) != 16
        ):
            raise ValueError(f"terminal {split} ledger evidence receipt counts differ")
        for ordinal, digest in enumerate(ordinal_hashes):
            validated_sha256(digest, label=f"terminal {split} ordinal {ordinal} evidence")
        for batch_index, digest in enumerate(batch_hashes):
            validated_sha256(digest, label=f"terminal {split} batch evidence")
            if digest != canonical_sha256(ordinal_hashes[4 * batch_index : 4 * batch_index + 4]):
                raise ValueError(f"terminal {split} batch receipt differs from ordinals")
        if not result["passed"]:
            encountered_stop = True
    expected_passed = len(opened) == 3 and all(report[split]["passed"] for split in opened)
    expected_stopped_after = opened[-1] if opened else "reviewed_development"
    if (
        not opened
        or record["protected_data_materialized"] is not bool(opened)
        or report["passed"] is not expected_passed
        or record["outcome"] != ("passed" if expected_passed else "failed")
        or record["status"] != ("complete" if expected_passed else "failed")
        or record["stopped_after"] != expected_stopped_after
    ):
        raise ValueError("terminal qualification stop/outcome semantics differ")


def _validate_development_report(
    report: Any,
    *,
    checkpoint_sha256: str,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _validate_report_root(report, qualification=False, error=False)
    if (
        report["passed"] is not True
        or report["review_ready"] is not True
        or report["protected_data_materialized"] is not False
        or report["stopped_after"] != "development"
    ):
        raise ValueError("reviewed development report is not exact passed/review-ready evidence")
    if report["source_provenance"] != source or report["publication_provenance"] != publication:
        raise ValueError("reviewed development report provenance differs from live source")
    if canonical_sha256(report["scene_family_certificate"]) != canonical_sha256(certificate):
        raise ValueError("reviewed development report certificate differs")
    development = report["development"]
    calibration = report["calibration"]
    _validate_split_result(development, split="development")
    if development["passed"] is not True:
        raise ValueError("reviewed development split did not pass")
    _validate_calibration(calibration)
    if report["checkpoint"] != str(canonical_checkpoint_path()):
        raise ValueError("reviewed development report names wrong checkpoint path")
    if report["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("reviewed development report checkpoint digest differs")
    if report["checkpoint_model_state_sha256"] != calibration["calibrated_model_state_sha256"]:
        raise ValueError("reviewed development report state hash differs from calibration")
    return development, calibration


def _validate_reviewed_bundle_from_disk(
    *,
    config: OrpheusConfig,
    source: Any,
    publication: Any,
    reviewed_checkpoint_sha256: Any,
    reviewed_report_sha256: Any,
    reviewed_ledger_sha256: Any,
    expected_inventory: frozenset[str] = DEVELOPMENT_ARTIFACT_NAMES,
) -> dict[str, Any]:
    if type(source) is not dict or type(publication) is not dict:
        raise TypeError("reviewed bundle requires exact live provenance dicts")
    checkpoint_digest = validated_sha256(
        reviewed_checkpoint_sha256, label="reviewed checkpoint SHA-256"
    )
    report_digest = validated_sha256(reviewed_report_sha256, label="reviewed report SHA-256")
    ledger_digest = validated_sha256(
        reviewed_ledger_sha256, label="reviewed development ledger SHA-256"
    )
    _validate_run_tree(expected_inventory, stage="reviewed development bundle")
    paths = {
        "checkpoint": canonical_checkpoint_path(),
        "development_report": canonical_development_report_path(),
        "development_ledger": development_ledger_path(),
    }
    _validate_distinct_canonical_paths(paths, atomic_writers=())
    checkpoint_contents = stable_read_bytes(paths["checkpoint"], label="reviewed checkpoint")
    report_contents = stable_read_bytes(paths["development_report"], label="reviewed report")
    ledger_contents = stable_read_bytes(paths["development_ledger"], label="reviewed ledger")
    if sha256_bytes(checkpoint_contents) != checkpoint_digest:
        raise ValueError("reviewed checkpoint bytes differ from external SHA-256")
    if sha256_bytes(report_contents) != report_digest:
        raise ValueError("reviewed report bytes differ from external SHA-256")
    if sha256_bytes(ledger_contents) != ledger_digest:
        raise ValueError("reviewed development ledger bytes differ from external SHA-256")
    current_source, current_publication, certificate = _current_execution_provenance(
        label="reviewed development bundle"
    )
    if current_source != source or current_publication != publication:
        raise RuntimeError("reviewed bundle provenance changed while reading artifacts")
    report = _strict_json_loads(report_contents, label="reviewed development report")
    ledger = _strict_json_loads(ledger_contents, label="reviewed development ledger")
    development, calibration = _validate_development_report(
        report,
        checkpoint_sha256=checkpoint_digest,
        source=source,
        publication=publication,
        certificate=certificate,
    )
    _validate_development_ledger_record(
        ledger,
        report=report,
        development=development,
        report_sha256=report_digest,
        checkpoint_sha256=checkpoint_digest,
        source=source,
        publication=publication,
    )
    payload = _checkpoint_payload_from_bytes(checkpoint_contents)
    model_state, state_sha256 = _validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        publication=publication,
        development=development,
        calibration=calibration,
    )
    # Re-read all bytes after validation so a concurrent replacement cannot
    # create a mixed reviewed bundle.
    for name, path, contents in (
        ("checkpoint", paths["checkpoint"], checkpoint_contents),
        ("report", paths["development_report"], report_contents),
        ("ledger", paths["development_ledger"], ledger_contents),
    ):
        if stable_read_bytes(path, label=f"reviewed {name} recheck") != contents:
            raise RuntimeError(f"reviewed {name} changed during validation")
    return {
        "report": report,
        "ledger": ledger,
        "development": development,
        "calibration": calibration,
        "model_state": model_state,
        "model_state_sha256": state_sha256,
        "calibration_sha256": canonical_sha256(calibration),
    }


def _review_development(
    *,
    config: OrpheusConfig,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    reviewed_checkpoint_sha256: str,
    reviewed_report_sha256: str,
    reviewed_ledger_sha256: str,
) -> tuple[_ReviewedDevelopmentSeal, dict[str, Any], dict[str, Any]]:
    reviewed = _validate_reviewed_bundle_from_disk(
        config=config,
        source=source,
        publication=publication,
        reviewed_checkpoint_sha256=reviewed_checkpoint_sha256,
        reviewed_report_sha256=reviewed_report_sha256,
        reviewed_ledger_sha256=reviewed_ledger_sha256,
    )
    bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": dict(source),
        "publication_provenance": dict(publication),
        "config_sha256": FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        "reviewed_checkpoint_sha256": reviewed_checkpoint_sha256,
        "reviewed_development_report_sha256": reviewed_report_sha256,
        "reviewed_development_ledger_sha256": reviewed_ledger_sha256,
        "model_state_sha256": reviewed["model_state_sha256"],
        "calibration_sha256": reviewed["calibration_sha256"],
    }
    seal = _mint_reviewed_development_seal(bindings)
    return seal, bindings, reviewed


def _stack_records(records: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot stack empty identifiable-drag records")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise RuntimeError("identifiable-drag record schema changed between frames")
    return {key: torch.stack([record[key] for record in records]) for key in keys}


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


def _camera_velocities(world_from_camera: Tensor, timestamps: Tensor) -> tuple[Tensor, Tensor]:
    position = world_from_camera[:, :3, 3]
    linear = torch.zeros_like(position)
    angular = torch.zeros_like(position)
    dt = (timestamps[1:] - timestamps[:-1]).clamp_min(1.0e-8)
    linear[1:] = (position[1:] - position[:-1]) / dt[:, None]
    linear[0] = linear[1]
    rotation = world_from_camera[:, :3, :3]
    delta = rotation[1:] @ rotation[:-1].transpose(-1, -2)
    skew = 0.5 * (delta - delta.transpose(-1, -2))
    vector = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=-1)
    angular[1:] = vector / dt[:, None]
    angular[0] = angular[1]
    return linear, angular


def _construct_identifiable_drag_episode(
    config: OrpheusConfig,
    *,
    split: str,
    ordinal: int,
    capability: _OrdinalCapability,
) -> dict[str, Any]:
    """Materialise one governed scene after a durable one-use receipt."""

    # This check/consume is deliberately the first operation: neither config
    # parsing nor scene metadata is touched through a forged/replayed token.
    _require_exact_governed_ordinal(split, ordinal)
    _consume_ordinal_constructor_capability(capability, split=split, ordinal=ordinal)
    _require_config_matches_frozen_path(config, _frozen_config_path())
    specification = scene_specification(split, ordinal)
    trajectory = manual_physical_trajectory(specification)
    initial = initial_sphere_state(specification)
    timestamps = torch.arange(FRAME_COUNT, dtype=torch.float32) / FRAME_RATE_HZ
    rgb_frames: list[Tensor] = []
    depth_frames: list[Tensor] = []
    state_records: list[dict[str, Tensor]] = []
    world_from_camera: list[Tensor] = []
    camera_from_world: list[Tensor] = []
    intrinsics: list[Tensor] = []
    camera_positions: list[Tensor] = []
    camera_targets: list[Tensor] = []
    minimum_support = math.inf
    minimum_visible_fraction = math.inf
    for frame_index, timestamp_tensor in enumerate(timestamps):
        state = replace(
            initial,
            position=trajectory.positions[frame_index].clone(),
            velocity=trajectory.velocities[frame_index].clone(),
        )
        state.validate()
        camera = orbital_camera_frame(specification, float(timestamp_tensor))
        rendered = render_spheres(state, camera, (64, 64), noise_std=0.0)
        labels = make_perception_labels(state, rendered, (64, 64))
        validate_perception_labels(labels, max_objects=2, image_size=(64, 64))
        support = rendered.full_mask.sum(dim=(-2, -1))
        minimum_support = min(minimum_support, float(support.min()))
        minimum_visible_fraction = min(
            minimum_visible_fraction, float(rendered.visible_fraction.min())
        )
        if bool((rendered.visible_mask.sum(dim=0) > 1).any()):
            raise RuntimeError("governed identifiable-drag silhouettes overlap")
        rgb_frames.append(rendered.rgb)
        depth_frames.append(rendered.depth_buffer.unsqueeze(0))
        state_records.append(_state_record(state, rendered.visible_fraction))
        world_from_camera.append(camera.world_from_camera)
        camera_from_world.append(camera.camera_from_world)
        intrinsics.append(camera.intrinsics)
        camera_positions.append(camera.position)
        camera_targets.append(camera.target)
    if minimum_support < 20.0 or minimum_visible_fraction != 1.0:
        raise RuntimeError("governed identifiable-drag scene failed full-visibility preflight")
    world_from = torch.stack(world_from_camera)
    camera_from = torch.stack(camera_from_world)
    timestamps = timestamps.contiguous()
    linear_camera, angular_camera = _camera_velocities(world_from, timestamps)
    episode: dict[str, Any] = {
        "rgb": torch.stack(rgb_frames).to(torch.float32),
        "depth": torch.stack(depth_frames).to(torch.float32),
        "timestamps": timestamps,
        "frame_mask": torch.ones(FRAME_COUNT, dtype=torch.bool),
        "camera": {
            "world_from_camera": world_from,
            "camera_from_world": camera_from,
            "intrinsics": torch.stack(intrinsics),
            "position": torch.stack(camera_positions),
            "target": torch.stack(camera_targets),
            "linear_velocity": linear_camera,
            "angular_velocity": angular_camera,
            "calibrated": torch.ones(FRAME_COUNT, dtype=torch.bool),
        },
        "objects": _stack_records(state_records),
        "num_objects": 2,
        "metadata": {
            **scene_metadata(specification),
            "scene_sha256": scene_signature(specification),
            "image_size": (64, 64),
            "frame_rate": FRAME_RATE_HZ,
            "minimum_support_pixels": float(minimum_support),
            "minimum_visible_fraction": float(minimum_visible_fraction),
        },
    }
    if "seed" in episode or "seed" in episode["metadata"]:
        raise RuntimeError("governed identifiable-drag episode exposed forbidden selection state")
    _mark_ordinal_constructed(capability, split=split, ordinal=ordinal)
    return episode


def _validate_ordinal_evaluator_capability(
    capability: _OrdinalCapability, *, split: str, ordinal: int
) -> None:
    _require_exact_governed_ordinal(split, ordinal)
    if type(capability) is not _OrdinalCapability:
        raise PermissionError("evaluator requires exact ordinal capability")
    registration = _LIVE_ORDINAL_CAPABILITIES.get(id(capability))
    if (
        type(registration) is not tuple
        or len(registration) != 6
        or registration[0] is not capability
        or registration[3] != split
        or registration[4] != ordinal
        or registration[5] != "constructed"
    ):
        raise PermissionError("evaluator capability is forged, replayed, or bound differently")
    _validate_manifest_capability(registration[1], split=split, operation="evaluator")


def _gather_physical_by_slot(value: Tensor, mapping: Tensor) -> Tensor:
    if value.shape[0] != mapping.shape[0] or mapping.shape[1] != 2:
        raise ValueError("physical gather requires matching batch and two slots")
    tail = value.shape[2:]
    index = mapping.reshape(mapping.shape[0], 2, *((1,) * len(tail))).expand(
        mapping.shape[0], 2, *tail
    )
    return torch.gather(value, 1, index)


def _birth_mapping(estimate: Tensor, truth: Tensor) -> Tensor:
    if estimate.shape != truth.shape or estimate.shape[1:] != (2, 3):
        raise ValueError("birth mapping requires [B,2,3] estimate/truth")
    direct = torch.linalg.vector_norm(estimate - truth, dim=-1).sum(dim=-1)
    swapped = torch.linalg.vector_norm(estimate - truth.flip(1), dim=-1).sum(dim=-1)
    direct_map = torch.tensor((0, 1), dtype=torch.int64, device=estimate.device)
    swap_map = torch.tensor((1, 0), dtype=torch.int64, device=estimate.device)
    return torch.where((swapped < direct)[:, None], swap_map, direct_map)


def _new_strict_reviewed_model(
    config: OrpheusConfig,
    *,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
) -> OnlineWorldModel:
    model = new_public_model(config)
    if reviewed_state is None:
        if expected_state_sha256 is not None:
            raise ValueError("raw development model cannot claim reviewed state")
        return model
    state = _validate_checkpoint_model_state(dict(reviewed_state))
    if expected_state_sha256 != _state_dict_sha256(state):
        raise ValueError("protected reviewed state differs from exact expected hash")
    model.load_state_dict(state, strict=True)
    if _model_state_sha256(model) != expected_state_sha256:
        raise RuntimeError("protected model did not strictly load reviewed three-buffer state")
    return model


def _vjp_metrics(
    *,
    batch: Mapping[str, Any],
    belief: Any,
    trajectory: Any,
    scene_sha256s: Sequence[str],
) -> dict[str, float]:
    batch_size = int(batch["rgb"].shape[0])
    if batch_size != 4 or len(scene_sha256s) != batch_size or len(set(scene_sha256s)) != batch_size:
        raise RuntimeError("VJP audit requires unique governed rows")
    sources = (
        batch["rgb"],
        batch["depth"],
        batch["camera"]["world_from_camera"],
    )
    if any(not source.requires_grad for source in sources):
        raise RuntimeError("VJP audit inputs must retain the public differentiable graph")
    fast = fast_packing_map(belief.objects)
    slow = slow_packing_map(belief.objects)
    coefficients = belief.objects.position.new_tensor((0.5, -0.75, 1.25))
    losses: list[tuple[int, int, str, Tensor]] = []
    for batch_index in range(batch_size):
        for object_index in OBJECT_INDICES:
            losses.extend(
                (
                    (
                        batch_index,
                        object_index,
                        "current_log_drag",
                        belief.objects.log_drag[batch_index, object_index].mean(),
                    ),
                    (
                        batch_index,
                        object_index,
                        "current_log_drag_log_variance",
                        belief.objects.slow_log_variance[
                            batch_index, object_index, slow["log_drag"]
                        ].mean(),
                    ),
                    (
                        batch_index,
                        object_index,
                        "horizon_2.00_position_log_variance",
                        (
                            trajectory.fast_log_variance[
                                batch_index, -1, object_index, fast["position"]
                            ]
                            * coefficients
                        ).mean(),
                    ),
                )
            )
    metrics: dict[str, float] = {
        "gradient_audit_scene_count": float(batch_size),
        "gradient_audit_unique_scene_fraction": 1.0,
        "world_from_camera_homogeneous_last_row_gradient_max_abs": 0.0,
    }
    for loss_index, (batch_index, object_index, output, loss) in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            sources,
            retain_graph=loss_index + 1 < len(losses),
            allow_unused=True,
        )
        for modality, source, gradient in zip(
            ("rgb", "depth", "world_from_camera"), sources, gradients, strict=True
        ):
            resolved = torch.zeros_like(source) if gradient is None else gradient
            if not bool(torch.isfinite(resolved).all()):
                raise FloatingPointError("VJP audit produced nonfinite gradients")
            per_scene = resolved.abs().reshape(batch_size, -1).sum(dim=-1)
            target = resolved[batch_index, :16]
            post_history = resolved[batch_index, 16:]
            per_frame = target.abs().reshape(16, -1).sum(dim=-1)
            suffix = f"object_{object_index}/{output}/{modality}"
            total = float(per_scene[batch_index])
            cross = torch.cat((per_scene[:batch_index], per_scene[batch_index + 1 :]))
            values = {
                f"gradient_l1/{suffix}": total,
                f"gradient_max_l1/{suffix}": total,
                f"gradient_cross_scene_max_l1/{suffix}": (
                    float(cross.max()) if cross.numel() else 0.0
                ),
                f"gradient_post_history_max_l1/{suffix}": (
                    float(post_history.abs().max()) if post_history.numel() else 0.0
                ),
                f"gradient_min_history_frame_l1/{suffix}": float(per_frame.min()),
                f"gradient_supported_history_frames/{suffix}": float(
                    (per_frame >= DEFAULT_GATES.minimum_input_gradient_l1).sum()
                ),
            }
            for key, value in values.items():
                if key.startswith(
                    (
                        "gradient_l1/",
                        "gradient_min_history_frame_l1/",
                        "gradient_supported_history_frames/",
                    )
                ):
                    metrics[key] = min(metrics.get(key, value), value)
                else:
                    metrics[key] = max(metrics.get(key, value), value)
            if modality == "world_from_camera":
                metrics["world_from_camera_homogeneous_last_row_gradient_max_abs"] = max(
                    metrics["world_from_camera_homogeneous_last_row_gradient_max_abs"],
                    float(resolved[..., 3, :].abs().max()),
                )
    return metrics


def _storage_alias(left: Tensor, right: Tensor) -> bool:
    return (
        left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _process_max_rss_bytes() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024.0


def _atomic_direct_valid_mask(evidence: Any) -> Tensor:
    valid = getattr(evidence, "valid_mask", None)
    drag_valid = getattr(evidence, "drag_valid_mask", None)
    position_valid = getattr(evidence, "position_valid_mask", None)
    if (
        type(valid) is not Tensor
        or valid.dtype != torch.bool
        or getattr(evidence, "log_drag", None) is None
        or type(drag_valid) is not Tensor
        or getattr(evidence, "position", None) is None
        or type(position_valid) is not Tensor
        or not torch.equal(drag_valid, valid)
        or not torch.equal(position_valid, valid)
        or not torch.equal(evidence.resolved_axis_valid_mask().all(dim=-1), valid)
    ):
        raise RuntimeError("direct runtime evidence must atomically own position/velocity/log-drag")
    return valid & drag_valid


def _evaluate_authorized_batch(
    config: OrpheusConfig,
    *,
    split: str,
    ordinals: tuple[int, int, int, int],
    manifest_capability: _ManifestCapability,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
    audit_vjp: bool,
) -> tuple[list[SceneSufficientEvidence], dict[str, float]]:
    """Atomically consume four exact rows and return only sufficient evidence."""

    # Revalidate the live durable capability before config or scene work.
    _require_exact_governed_batch(split, ordinals)
    _validate_manifest_capability(manifest_capability, split=split, operation="evaluator")
    batch_capability = manifest_capability.begin_batch(ordinals)
    episodes: list[dict[str, Any]] = []
    for ordinal, ordinal_capability in zip(
        batch_capability._ordinals, batch_capability._tokens, strict=True
    ):
        episodes.append(
            _construct_identifiable_drag_episode(
                config,
                split=split,
                ordinal=ordinal,
                capability=ordinal_capability,
            )
        )
        _validate_ordinal_evaluator_capability(ordinal_capability, split=split, ordinal=ordinal)
    _validate_batch_capability(batch_capability, split=split, operation="evaluator")
    batch = collate_episodes(episodes)
    if audit_vjp:
        batch["rgb"] = batch["rgb"].clone().requires_grad_(True)
        batch["depth"] = batch["depth"].clone().requires_grad_(True)
        camera = dict(batch["camera"])
        camera["world_from_camera"] = camera["world_from_camera"].clone().requires_grad_(True)
        batch["camera"] = camera
    model = _new_strict_reviewed_model(
        config,
        reviewed_state=reviewed_state,
        expected_state_sha256=expected_state_sha256,
    )
    model.eval()
    model.reset(batch_size=4)
    original_match = model.associator.match
    original_direct = model.updater.correct_direct_velocity
    association = {
        "matched": torch.zeros(4, dtype=torch.int64),
        "opportunities": torch.zeros(4, dtype=torch.int64),
        "ambiguous": torch.zeros(4, dtype=torch.int64),
    }
    direct = {
        "calls": torch.zeros(4, dtype=torch.int64),
        "valid": torch.zeros(4, dtype=torch.int64),
        "total": torch.zeros(4, dtype=torch.int64),
        "position_change": torch.zeros(4, dtype=torch.float32),
        "position_owner_error": torch.zeros(4, dtype=torch.float32),
        "velocity_owner_error": torch.zeros(4, dtype=torch.float32),
        "drag_owner_error": torch.zeros(4, dtype=torch.float32),
    }

    def recording_match(belief: Any, measurements: Any, predicted: Any) -> Any:
        result = original_match(belief, measurements, predicted)
        association["opportunities"] += predicted.valid_mask.detach().cpu().sum(dim=-1)
        association["matched"] += result.pair_mask.detach().cpu().sum(dim=-1)
        association["ambiguous"] += (result.ambiguous & result.pair_mask).detach().cpu().sum(dim=-1)
        return result

    def recording_direct(prior: Any, evidence: Any) -> Any:
        direct["calls"] += 1
        atomic_valid = _atomic_direct_valid_mask(evidence)
        direct["valid"] += atomic_valid.detach().cpu().sum(dim=-1)
        direct["total"] += torch.full((4,), evidence.valid_mask.shape[-1], dtype=torch.int64)
        before = prior.objects.position
        posterior = original_direct(prior, evidence)
        direct["position_change"] = torch.maximum(
            direct["position_change"],
            (posterior.objects.position - before).detach().abs().reshape(4, -1).amax(dim=-1).cpu(),
        )
        if evidence.log_drag is not None:
            valid = atomic_valid & prior.objects.active
            for batch_index in range(4):
                selected = valid[batch_index]
                if not bool(selected.any()):
                    continue
                direct["position_owner_error"][batch_index] = max(
                    direct["position_owner_error"][batch_index],
                    (
                        posterior.objects.position[batch_index, selected]
                        - evidence.position[batch_index, selected]
                    )
                    .detach()
                    .abs()
                    .max()
                    .cpu(),
                )
                direct["velocity_owner_error"][batch_index] = max(
                    direct["velocity_owner_error"][batch_index],
                    (
                        posterior.objects.velocity[batch_index, selected]
                        - evidence.velocity[batch_index, selected]
                    )
                    .detach()
                    .abs()
                    .max()
                    .cpu(),
                )
                direct["drag_owner_error"][batch_index] = max(
                    direct["drag_owner_error"][batch_index],
                    (
                        posterior.objects.log_drag[batch_index, selected]
                        - evidence.log_drag[batch_index, selected]
                    )
                    .detach()
                    .abs()
                    .max()
                    .cpu(),
                )
        return posterior

    model.associator.match = recording_match  # type: ignore[method-assign]
    model.updater.correct_direct_velocity = recording_direct  # type: ignore[method-assign]
    identities: list[Tensor] = []
    observed_positions: list[Tensor] = []
    ingest_count = 0
    predict_count = 0
    runtime_fit_calls: list[tuple[int, int, int]] = []
    original_ingest = model.ingest
    original_predict = model.predict
    original_fit_with_drag = RGBDTemporalPositionHistory.fit_with_drag

    def counted_ingest(packet: Any) -> Any:
        nonlocal ingest_count
        ingest_count += 1
        return original_ingest(packet)

    def counted_predict(query_times: Any) -> Any:
        nonlocal predict_count
        predict_count += 1
        return original_predict(query_times)

    def recording_fit_with_drag(
        history: RGBDTemporalPositionHistory, *args: Any, **kwargs: Any
    ) -> Any:
        grid_points = kwargs.get("grid_points")
        minimum_support = kwargs.get("minimum_support")
        if type(grid_points) is not int or type(minimum_support) is not int:
            raise RuntimeError("runtime drag fit omitted exact grid/support arguments")
        runtime_fit_calls.append((grid_points, minimum_support, history.history_size))
        return original_fit_with_drag(history, *args, **kwargs)

    model.ingest = counted_ingest  # type: ignore[method-assign]
    model.predict = counted_predict  # type: ignore[method-assign]
    start = time.perf_counter()
    context = torch.enable_grad() if audit_vjp else torch.no_grad()
    RGBDTemporalPositionHistory.fit_with_drag = recording_fit_with_drag
    try:
        with context:
            for frame_index in HISTORY_FRAME_INDICES:
                posterior = model.ingest(make_rgbd_packet(batch, frame_index))
                identities.append(posterior.objects.object_id)
                observed_positions.append(posterior.objects.position)
            trajectory = model.predict(HORIZONS_SECONDS).validate()
            belief = model.belief
            if belief is None:
                raise RuntimeError("identifiable-drag runtime failed to retain a belief")
            vjp_start = time.perf_counter()
            vjp = (
                _vjp_metrics(
                    batch=batch,
                    belief=belief,
                    trajectory=trajectory,
                    scene_sha256s=tuple(
                        episode["metadata"]["scene_sha256"] for episode in episodes
                    ),
                )
                if audit_vjp
                else {}
            )
            vjp_seconds = time.perf_counter() - vjp_start if audit_vjp else 0.0
    finally:
        RGBDTemporalPositionHistory.fit_with_drag = original_fit_with_drag
    evaluation_seconds = time.perf_counter() - start - vjp_seconds
    history = model.state.temporal_histories.get(RUNTIME_STREAM_KEY)
    if not isinstance(history, RGBDTemporalPositionHistory):
        raise RuntimeError("identifiable-drag runtime omitted typed 16-row history")
    if runtime_stream_key("rgbd", "camera0:rgbd") != RUNTIME_STREAM_KEY:
        raise RuntimeError("identifiable-drag runtime stream key changed")
    expected_object_ids = torch.tensor(
        OBJECT_INDICES, dtype=torch.int64, device=history.object_ids.device
    ).expand(4, -1)
    expected_history_timestamps = (
        batch["timestamps"][:, :HISTORY_FRAME_COUNT]
        .to(device=history.timestamps.device, dtype=history.timestamps.dtype)
        .unsqueeze(1)
        .expand(-1, 2, -1)
    )
    if (
        ingest_count != HISTORY_FRAME_COUNT
        or model.state.ingest_count != HISTORY_FRAME_COUNT
        or predict_count != 1
        or model.state.batch_size != 4
        or history.history_size != HISTORY_FRAME_COUNT
        or set(model.state.temporal_histories) != {RUNTIME_STREAM_KEY}
        or tuple(history.sample_mask.shape) != (4, 2, HISTORY_FRAME_COUNT)
        or tuple(history.valid_mask.shape) != (4, 2, HISTORY_FRAME_COUNT)
        or not bool(history.sample_mask.all())
        or not bool(history.valid_mask.all())
        or not torch.equal(history.object_ids, expected_object_ids)
        or not torch.equal(history.timestamps, expected_history_timestamps)
        or not torch.equal(belief.objects.object_id, expected_object_ids)
    ):
        raise RuntimeError(
            "runtime history/counters differ from exact per-ID 16-row public schedule"
        )
    expected_fit_call = (
        config.model.rgbd.temporal_drag_grid_points,
        config.model.rgbd.temporal_min_samples,
        HISTORY_FRAME_COUNT,
    )
    if runtime_fit_calls != [expected_fit_call]:
        raise RuntimeError("runtime drag fit was not invoked once with the exact frozen grid")
    with torch.no_grad():
        fit, fit_valid = history.fit_with_drag(
            gravity=belief.gravity,
            drag_bounds=(
                config.model.rgbd.temporal_drag_minimum,
                config.model.rgbd.temporal_drag_maximum,
            ),
            grid_points=config.model.rgbd.temporal_drag_grid_points,
            position_noise_floor=config.model.rgbd.temporal_drag_noise_floor_m,
            minimum_support=config.model.rgbd.temporal_min_samples,
            minimum_dt=config.model.rgbd.temporal_min_dt,
            conditioning_limit=config.model.rgbd.fit_conditioning_limit,
            minimum_excitation=config.model.rgbd.temporal_drag_minimum_excitation_m,
            maximum_boundary_mass=config.model.rgbd.temporal_drag_maximum_boundary_mass,
            minimum_profile_information=(
                config.model.rgbd.temporal_drag_minimum_profile_information
            ),
        )
    direct_evidence = model.last_direct_velocity_evidence
    if direct_evidence is None or not bool(fit_valid.all()):
        raise RuntimeError("identifiable-drag scene did not produce two complete valid fits")
    if not bool(_atomic_direct_valid_mask(direct_evidence).all()):
        raise RuntimeError("identifiable-drag runtime omitted an atomic p/v/log-drag fit")
    fast = fast_packing_map(belief.objects)
    slow = slow_packing_map(belief.objects)
    identity_tensor = torch.stack(identities, dim=1)
    observed = torch.stack(observed_positions, dim=1)
    truth_birth = batch["objects"]["position"][:, 0]
    mapping = _birth_mapping(observed[:, 0], truth_birth)
    current_position_truth = _gather_physical_by_slot(
        batch["objects"]["position"][:, ANCHOR_FRAME_INDEX], mapping
    )
    current_velocity_truth = _gather_physical_by_slot(
        batch["objects"]["velocity"][:, ANCHOR_FRAME_INDEX], mapping
    )
    log_drag_truth = _gather_physical_by_slot(
        batch["objects"]["drag"][:, ANCHOR_FRAME_INDEX], mapping
    ).log()
    target_index = torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64)
    future_position_truth = _gather_physical_by_slot(
        batch["objects"]["position"][:, target_index].transpose(1, 2), mapping
    ).transpose(1, 2)
    future_velocity_truth = _gather_physical_by_slot(
        batch["objects"]["velocity"][:, target_index].transpose(1, 2), mapping
    ).transpose(1, 2)
    current_position_mean = belief.objects.position
    current_velocity_mean = belief.objects.velocity
    log_drag_mean = belief.objects.log_drag
    current_position_variance = belief.objects.fast_log_variance[:, :, fast["position"]].exp()
    current_velocity_variance = belief.objects.fast_log_variance[:, :, fast["velocity"]].exp()
    log_drag_variance = belief.objects.slow_log_variance[:, :, slow["log_drag"]].exp()
    future_position_variance = trajectory.fast_log_variance[:, :, :, fast["position"]].exp()
    future_velocity_variance = trajectory.fast_log_variance[:, :, :, fast["velocity"]].exp()
    horizons = belief.objects.position.new_tensor(HORIZONS_SECONDS)[None, :, None, None]
    fixed_position: list[Tensor] = []
    fixed_velocity: list[Tensor] = []
    for coefficient in (0.05, 0.185):
        decay = torch.exp(-coefficient * horizons)
        fixed_velocity.append(current_velocity_mean[:, None] * decay)
        fixed_position.append(
            current_position_mean[:, None]
            + current_velocity_mean[:, None] * ((1.0 - decay) / coefficient)
        )
    fixed_position_tensor = torch.stack(fixed_position)
    fixed_velocity_tensor = torch.stack(fixed_velocity)
    direct_two = model.dynamics.predict(belief, 2.0)
    composed = model.dynamics.predict(model.dynamics.predict(belief, 0.75), 1.25)
    expected_times = belief.timestamp[:, None] + belief.timestamp.new_tensor(HORIZONS_SECONDS)
    analytic_horizons = horizons
    analytic_drag = belief.objects.drag[:, None]
    analytic_decay = torch.exp(-analytic_drag * analytic_horizons)
    analytic_velocity = belief.objects.velocity[:, None] * analytic_decay
    analytic_position = belief.objects.position[:, None] + belief.objects.velocity[:, None] * (
        -torch.expm1(-analytic_drag * analytic_horizons) / analytic_drag
    )
    analytic_position_factor = -torch.expm1(-analytic_drag * analytic_horizons) / analytic_drag
    analytic_position_drag_jacobian = belief.objects.velocity[:, None] * (
        analytic_horizons * analytic_decay - analytic_position_factor
    )
    analytic_velocity_drag_jacobian = (
        -analytic_drag * analytic_horizons * analytic_decay * belief.objects.velocity[:, None]
    )
    reconstructed_position_variance = (
        current_position_variance[:, None]
        + analytic_position_factor.square() * current_velocity_variance[:, None]
        + analytic_position_drag_jacobian.square() * log_drag_variance[:, None]
    )
    reconstructed_velocity_variance = (
        analytic_decay.square() * current_velocity_variance[:, None]
        + analytic_velocity_drag_jacobian.square() * log_drag_variance[:, None]
    )
    alias_count = sum(
        (
            _storage_alias(trajectory.positions, belief.objects.position),
            _storage_alias(trajectory.positions, belief.objects.velocity),
            _storage_alias(trajectory.velocities, belief.objects.position),
            _storage_alias(trajectory.velocities, belief.objects.velocity),
        )
    )
    expected_ids = torch.tensor(
        OBJECT_INDICES, dtype=torch.int64, device=identity_tensor.device
    ).view(1, 1, 2)
    identity_switches = identity_tensor[:, 1:].ne(identity_tensor[:, :-1]).any(dim=-1).sum(dim=-1)
    persistent_mismatch = identity_tensor.ne(expected_ids).sum(dim=(1, 2))
    position_floor = float(math.exp(config.model.filter.min_log_variance))
    position_ceiling = float(math.exp(config.model.filter.max_log_variance))
    drag_floor = config.model.rgbd.temporal_drag_log_parameter_variance_floor
    drag_ceiling = config.model.rgbd.temporal_drag_log_parameter_variance_ceiling
    state_dict = model.state_dict()
    state_bytes = sum(item.numel() * item.element_size() for item in state_dict.values())
    parameters = tuple(model.parameters())
    learned_parameter_count = sum(parameter.numel() for parameter in parameters)
    learned_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    process_rss = _process_max_rss_bytes()
    rows: list[SceneSufficientEvidence] = []
    for batch_index, (ordinal, episode) in enumerate(
        zip(batch_capability._ordinals, episodes, strict=True)
    ):
        valid_row = fit_valid[batch_index]
        diagnostics = (
            ("fit_valid_count", float(valid_row.sum())),
            ("minimum_drag_excitation_m", float(fit.excitation[batch_index, valid_row].min())),
            (
                "minimum_profile_information",
                float(fit.profile_information[batch_index, valid_row].min()),
            ),
            ("maximum_boundary_mass", float(fit.boundary_mass[batch_index, valid_row].max())),
            ("identity_switch_count", float(identity_switches[batch_index])),
            ("persistent_id_mismatch_count", float(persistent_mismatch[batch_index])),
            ("persistent_object_id_min", float(identity_tensor[batch_index].min())),
            ("persistent_object_id_max", float(identity_tensor[batch_index].max())),
            ("association_matched", float(association["matched"][batch_index])),
            (
                "association_opportunities",
                float(association["opportunities"][batch_index]),
            ),
            (
                "association_ambiguous_pair_count",
                float(association["ambiguous"][batch_index]),
            ),
            ("direct_velocity_calls", float(direct["calls"][batch_index])),
            ("direct_velocity_valid", float(direct["valid"][batch_index])),
            ("direct_velocity_total", float(direct["total"][batch_index])),
            (
                "direct_velocity_position_change_max_abs_m",
                float(direct["position_change"][batch_index]),
            ),
            (
                "direct_fit_position_owner_max_abs_m",
                float(direct["position_owner_error"][batch_index]),
            ),
            (
                "direct_fit_velocity_owner_max_abs_mps",
                float(direct["velocity_owner_error"][batch_index]),
            ),
            (
                "direct_fit_log_drag_owner_max_abs",
                float(direct["drag_owner_error"][batch_index]),
            ),
            (
                "semigroup_position_max_abs_m",
                float(
                    (
                        composed.objects.position[batch_index]
                        - direct_two.objects.position[batch_index]
                    )
                    .abs()
                    .max()
                ),
            ),
            (
                "semigroup_velocity_max_abs_mps",
                float(
                    (
                        composed.objects.velocity[batch_index]
                        - direct_two.objects.velocity[batch_index]
                    )
                    .abs()
                    .max()
                ),
            ),
            (
                "analytic_position_agreement_max_abs_m",
                float(
                    (trajectory.positions[batch_index] - analytic_position[batch_index]).abs().max()
                ),
            ),
            (
                "analytic_velocity_agreement_max_abs_mps",
                float(
                    (trajectory.velocities[batch_index] - analytic_velocity[batch_index])
                    .abs()
                    .max()
                ),
            ),
            (
                "future_position_variance_partition_max_abs",
                float(
                    (
                        future_position_variance[batch_index]
                        - reconstructed_position_variance[batch_index]
                    )
                    .abs()
                    .max()
                ),
            ),
            (
                "future_velocity_variance_partition_max_abs",
                float(
                    (
                        future_velocity_variance[batch_index]
                        - reconstructed_velocity_variance[batch_index]
                    )
                    .abs()
                    .max()
                ),
            ),
            (
                "public_direct_position_max_abs_m",
                float(
                    (
                        trajectory.positions[batch_index, -1]
                        - direct_two.objects.position[batch_index]
                    )
                    .abs()
                    .max()
                ),
            ),
            (
                "public_direct_velocity_max_abs_mps",
                float(
                    (
                        trajectory.velocities[batch_index, -1]
                        - direct_two.objects.velocity[batch_index]
                    )
                    .abs()
                    .max()
                ),
            ),
            (
                "public_query_time_max_abs_seconds",
                float(
                    (trajectory.timestamps[batch_index] - expected_times[batch_index]).abs().max()
                ),
            ),
            ("public_rollout_output_alias_count", float(alias_count)),
            ("ingested_frame_count", float(ingest_count)),
            ("state_ingest_count", float(model.state.ingest_count)),
            ("history_sample_count", float(history.sample_mask[batch_index].sum())),
            ("history_valid_count", float(history.valid_mask[batch_index].sum())),
            ("public_predict_calls", float(predict_count)),
            (
                "position_variance_floor_clamp_count",
                float((current_position_variance[batch_index] == position_floor).sum()),
            ),
            (
                "position_variance_ceiling_clamp_count",
                float((current_position_variance[batch_index] == position_ceiling).sum()),
            ),
            (
                "velocity_variance_floor_clamp_count",
                float((current_velocity_variance[batch_index] == position_floor).sum()),
            ),
            (
                "velocity_variance_ceiling_clamp_count",
                float((current_velocity_variance[batch_index] == position_ceiling).sum()),
            ),
            (
                "drag_variance_floor_clamp_count",
                float((log_drag_variance[batch_index] == drag_floor).sum()),
            ),
            (
                "drag_variance_ceiling_clamp_count",
                float((log_drag_variance[batch_index] == drag_ceiling).sum()),
            ),
            (
                "drag_grid_boundary_selection_count",
                float(
                    (
                        (
                            log_drag_mean[batch_index].exp()
                            <= config.model.rgbd.temporal_drag_minimum
                        )
                        | (
                            log_drag_mean[batch_index].exp()
                            >= config.model.rgbd.temporal_drag_maximum
                        )
                    ).sum()
                ),
            ),
            ("drag_grid_point_count", float(runtime_fit_calls[0][0])),
            (
                "evaluation_latency_seconds",
                float(evaluation_seconds) if batch_index == 0 else 0.0,
            ),
            ("vjp_latency_seconds", float(vjp_seconds) if batch_index == 0 else 0.0),
            ("process_max_rss_bytes", process_rss),
            ("module_tensor_buffer_count", float(len(tuple(model.buffers())))),
            ("persistent_module_state_key_count", float(len(state_dict))),
            ("persistent_module_state_bytes", float(state_bytes)),
            ("learned_parameter_count", float(learned_parameter_count)),
            ("learned_parameter_bytes", float(learned_parameter_bytes)),
        )
        sufficient = SceneSufficientEvidence(
            split=split,
            ordinal=ordinal,
            scene_sha256=episode["metadata"]["scene_sha256"],
            current_position_truth=current_position_truth[batch_index].detach().cpu().contiguous(),
            current_position_mean=current_position_mean[batch_index].detach().cpu().contiguous(),
            current_position_raw_variance=current_position_variance[batch_index]
            .detach()
            .cpu()
            .contiguous(),
            current_velocity_truth=current_velocity_truth[batch_index].detach().cpu().contiguous(),
            current_velocity_mean=current_velocity_mean[batch_index].detach().cpu().contiguous(),
            current_velocity_raw_variance=current_velocity_variance[batch_index]
            .detach()
            .cpu()
            .contiguous(),
            log_drag_truth=log_drag_truth[batch_index].detach().cpu().contiguous(),
            log_drag_mean=log_drag_mean[batch_index].detach().cpu().contiguous(),
            log_drag_raw_variance=log_drag_variance[batch_index].detach().cpu().contiguous(),
            future_position_truth=future_position_truth[batch_index].detach().cpu().contiguous(),
            future_position_mean=trajectory.positions[batch_index].detach().cpu().contiguous(),
            future_position_raw_variance=future_position_variance[batch_index]
            .detach()
            .cpu()
            .contiguous(),
            future_velocity_truth=future_velocity_truth[batch_index].detach().cpu().contiguous(),
            future_velocity_mean=trajectory.velocities[batch_index].detach().cpu().contiguous(),
            future_velocity_raw_variance=future_velocity_variance[batch_index]
            .detach()
            .cpu()
            .contiguous(),
            fixed_position_mean=fixed_position_tensor[:, batch_index].detach().cpu().contiguous(),
            fixed_velocity_mean=fixed_velocity_tensor[:, batch_index].detach().cpu().contiguous(),
            diagnostics=tuple((name, float(value)) for name, value in diagnostics),
        )
        rows.append(_validated_evidence(sufficient, split=split, ordinal=ordinal))
    _mark_batch_evaluated(batch_capability)
    evidence_sha256s = tuple(_evidence_cache_sha256((row,)) for row in rows)
    manifest_capability.complete_batch(
        batch_capability,
        evidence_sha256s=evidence_sha256s,
    )
    return rows, vjp


@dataclass(frozen=True, slots=True)
class _DeployedVarianceEvidence:
    current_position: Tensor
    current_velocity: Tensor
    log_drag: Tensor
    future_position: Tensor
    future_velocity: Tensor
    position_floor_clamps: int
    position_ceiling_clamps: int
    velocity_floor_clamps: int
    velocity_ceiling_clamps: int
    drag_floor_clamps: int
    drag_ceiling_clamps: int


def _exact_float32_scale(value: Tensor, *, label: str) -> Tensor:
    if (
        type(value) is not Tensor
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.ndim != 0
        or not bool(torch.isfinite(value))
        or not bool(value > 0.0)
        or not bool(torch.isfinite(value.square()))
    ):
        raise ValueError(f"{label} must be an exact finite positive CPU float32 scalar")
    return value


def _deployed_variances_from_cache(
    row: SceneSufficientEvidence,
    *,
    position_scale: Tensor,
    velocity_scale: Tensor,
    drag_scale: Tensor,
    config: OrpheusConfig,
) -> _DeployedVarianceEvidence:
    """Reconstruct the exact direct-anchor variance partition without replay."""

    _validated_evidence(row, split=row.split, ordinal=row.ordinal)
    position_scale = _exact_float32_scale(position_scale, label="position scale")
    velocity_scale = _exact_float32_scale(velocity_scale, label="velocity scale")
    drag_scale = _exact_float32_scale(drag_scale, label="drag scale")
    minimum_log_variance = float(config.model.filter.min_log_variance)
    maximum_log_variance = float(config.model.filter.max_log_variance)

    def scaled_filter_variance(raw: Tensor, scale: Tensor) -> tuple[Tensor, int, int]:
        unbounded = raw * scale.square()
        log_unbounded = unbounded.log()
        # Equality is counted conservatively: cached values at a saturation
        # boundary cannot prove that the runtime path did not clamp there.
        floor = int((log_unbounded <= minimum_log_variance).sum())
        ceiling = int((log_unbounded >= maximum_log_variance).sum())
        return (
            log_unbounded.clamp(
                min=minimum_log_variance,
                max=maximum_log_variance,
            ).exp(),
            floor,
            ceiling,
        )

    current_position, position_floor, position_ceiling = scaled_filter_variance(
        row.current_position_raw_variance, position_scale
    )
    current_velocity, velocity_floor, velocity_ceiling = scaled_filter_variance(
        row.current_velocity_raw_variance, velocity_scale
    )
    raw_drag = row.log_drag_raw_variance * drag_scale.square()
    drag_floor_bound = float(config.model.rgbd.temporal_drag_log_parameter_variance_floor)
    drag_ceiling_bound = float(config.model.rgbd.temporal_drag_log_parameter_variance_ceiling)
    drag_floor = int((raw_drag <= drag_floor_bound).sum())
    drag_ceiling = int((raw_drag >= drag_ceiling_bound).sum())
    log_drag = (
        raw_drag.clamp(min=drag_floor_bound, max=drag_ceiling_bound)
        .log()
        .clamp(min=minimum_log_variance, max=maximum_log_variance)
        .exp()
    )
    horizons = row.current_position_mean.new_tensor(HORIZONS_SECONDS)[:, None, None]
    drag = row.log_drag_mean.exp()[None]
    decay = torch.exp(-drag * horizons)
    position_factor = -torch.expm1(-drag * horizons) / drag
    position_drag_jacobian = row.current_velocity_mean[None] * (horizons * decay - position_factor)
    velocity_drag_jacobian = -drag * horizons * decay * row.current_velocity_mean[None]
    future_position = (
        current_position[None]
        + position_factor.square() * current_velocity[None]
        + position_drag_jacobian.square() * log_drag[None]
    )
    future_velocity = (
        decay.square() * current_velocity[None] + velocity_drag_jacobian.square() * log_drag[None]
    )
    for name, value in (
        ("current position", current_position),
        ("current velocity", current_velocity),
        ("log drag", log_drag),
        ("future position", future_position),
        ("future velocity", future_velocity),
    ):
        if (
            value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
            or not bool((value > 0.0).all())
        ):
            raise FloatingPointError(f"cached deployed {name} variance is invalid")
    return _DeployedVarianceEvidence(
        current_position=current_position,
        current_velocity=current_velocity,
        log_drag=log_drag,
        future_position=future_position,
        future_velocity=future_velocity,
        position_floor_clamps=position_floor,
        position_ceiling_clamps=position_ceiling,
        velocity_floor_clamps=velocity_floor,
        velocity_ceiling_clamps=velocity_ceiling,
        drag_floor_clamps=drag_floor,
        drag_ceiling_clamps=drag_ceiling,
    )


def _diagnostic_map(row: SceneSufficientEvidence) -> dict[str, float]:
    result = dict(row.diagnostics)
    if len(result) != len(row.diagnostics):
        raise ValueError("cached evidence diagnostics contain duplicate names")
    return result


def _finite_rmse(error: Tensor, *, label: str) -> float:
    result = float(error.square().mean().sqrt())
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError(f"{label} RMSE is invalid")
    return result


def _coverage_statistics(error: Tensor, variance: Tensor) -> tuple[float, float, float]:
    if error.shape != variance.shape or error.shape[0] != 64:
        raise ValueError("coverage requires matching 64-scene tensors")
    if not bool(torch.isfinite(error).all()) or not bool(torch.isfinite(variance).all()):
        raise FloatingPointError("coverage evidence is nonfinite")
    if not bool((variance > 0.0).all()):
        raise ValueError("coverage variance must be strictly positive")
    radius = error.new_tensor(CALIBRATION_Z) * variance.sqrt()
    covered = error.abs() <= radius
    marginal = float(covered.to(torch.float64).mean())
    joint = float(covered.reshape(64, -1).all(dim=-1).to(torch.float64).mean())
    rms_z = float((error.square() / variance).to(torch.float64).mean().sqrt())
    if not all(math.isfinite(value) for value in (marginal, joint, rms_z)):
        raise FloatingPointError("coverage statistics are nonfinite")
    return marginal, joint, rms_z


def _aggregate_split_metrics(
    cache: Sequence[SceneSufficientEvidence],
    *,
    split: str,
    position_scale: Tensor,
    velocity_scale: Tensor,
    drag_scale: Tensor,
    config: OrpheusConfig,
    vjp: Mapping[str, float],
    cache_is_scale_one: bool,
) -> dict[str, float]:
    """Aggregate one exact cache; no scene, packet, or runtime is replayed."""

    if type(cache) not in {list, tuple} or len(cache) != 64:
        raise ValueError("split aggregation requires exactly 64 cached rows")
    rows = [
        _validated_evidence(row, split=split, ordinal=ordinal) for ordinal, row in enumerate(cache)
    ]
    if type(cache_is_scale_one) is not bool:
        raise TypeError("cache scale-one provenance flag must be exact bool")
    variance_position_scale = (
        position_scale if cache_is_scale_one else torch.tensor(1.0, dtype=torch.float32)
    )
    variance_velocity_scale = (
        velocity_scale if cache_is_scale_one else torch.tensor(1.0, dtype=torch.float32)
    )
    variance_drag_scale = (
        drag_scale if cache_is_scale_one else torch.tensor(1.0, dtype=torch.float32)
    )
    deployed = [
        _deployed_variances_from_cache(
            row,
            position_scale=variance_position_scale,
            velocity_scale=variance_velocity_scale,
            drag_scale=variance_drag_scale,
            config=config,
        )
        for row in rows
    ]
    diagnostics = [_diagnostic_map(row) for row in rows]

    def stack(name: str) -> Tensor:
        return torch.stack([getattr(row, name) for row in rows])

    def deployed_stack(name: str) -> Tensor:
        return torch.stack([getattr(item, name) for item in deployed])

    def diagnostic_values(name: str) -> list[float]:
        try:
            values = [item[name] for item in diagnostics]
        except KeyError as error:
            raise ValueError(f"cached evidence omitted diagnostic {name!r}") from error
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ValueError(f"cached diagnostic {name!r} must remain finite floats")
        return values

    position_truth = stack("current_position_truth")
    position_mean = stack("current_position_mean")
    velocity_truth = stack("current_velocity_truth")
    velocity_mean = stack("current_velocity_mean")
    log_drag_truth = stack("log_drag_truth")
    log_drag_mean = stack("log_drag_mean")
    future_position_truth = stack("future_position_truth")
    future_position_mean = stack("future_position_mean")
    future_velocity_truth = stack("future_velocity_truth")
    future_velocity_mean = stack("future_velocity_mean")
    fixed_position_mean = stack("fixed_position_mean")
    fixed_velocity_mean = stack("fixed_velocity_mean")
    position_error = position_mean - position_truth
    velocity_error = velocity_mean - velocity_truth
    log_drag_error = log_drag_mean - log_drag_truth
    current_log_drag_variance = deployed_stack("log_drag")
    future_position_variance = deployed_stack("future_position")
    future_velocity_variance = deployed_stack("future_velocity")

    metrics: dict[str, float] = {
        "scene_count": float(len(rows)),
        "position_uncertainty_scale": float(position_scale),
        "velocity_uncertainty_scale": float(velocity_scale),
        "drag_uncertainty_scale": float(drag_scale),
        "current_position_rmse_m": _finite_rmse(position_error, label="current position"),
        "current_velocity_rmse_mps": _finite_rmse(velocity_error, label="current velocity"),
        "current_drag_rmse_per_s": _finite_rmse(
            log_drag_mean.exp() - log_drag_truth.exp(), label="current drag"
        ),
        "current_log_drag_rmse": _finite_rmse(log_drag_error, label="current log drag"),
        "log_drag_gaussian_nll": float(
            (
                0.5
                * (
                    math.log(2.0 * math.pi)
                    + current_log_drag_variance.log()
                    + log_drag_error.square() / current_log_drag_variance
                )
            )
            .to(torch.float64)
            .mean()
        ),
    }
    fit_valid_count = sum(diagnostic_values("fit_valid_count"))
    metrics["object_fit_count"] = float(fit_valid_count)
    metrics["invalid_fit_count"] = float(128.0 - fit_valid_count)

    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        position_horizon_error = (
            future_position_mean[:, horizon_index] - future_position_truth[:, horizon_index]
        )
        velocity_horizon_error = (
            future_velocity_mean[:, horizon_index] - future_velocity_truth[:, horizon_index]
        )
        metrics[f"horizon_{label}_position_rmse_m"] = _finite_rmse(
            position_horizon_error, label=f"horizon {label} position"
        )
        metrics[f"horizon_{label}_velocity_rmse_mps"] = _finite_rmse(
            velocity_horizon_error, label=f"horizon {label} velocity"
        )
        for fixed_index, coefficient in enumerate(("0.05", "0.185")):
            metrics[f"fixed_{coefficient}_horizon_{label}_position_rmse_m"] = _finite_rmse(
                fixed_position_mean[:, fixed_index, horizon_index]
                - future_position_truth[:, horizon_index],
                label=f"fixed {coefficient} horizon {label} position",
            )
            metrics[f"fixed_{coefficient}_horizon_{label}_velocity_rmse_mps"] = _finite_rmse(
                fixed_velocity_mean[:, fixed_index, horizon_index]
                - future_velocity_truth[:, horizon_index],
                label=f"fixed {coefficient} horizon {label} velocity",
            )
        for quantity, error, variance in (
            (
                "position",
                position_horizon_error,
                future_position_variance[:, horizon_index],
            ),
            (
                "velocity",
                velocity_horizon_error,
                future_velocity_variance[:, horizon_index],
            ),
            ("log_drag", log_drag_error, current_log_drag_variance),
        ):
            marginal, joint, rms_z = _coverage_statistics(error, variance)
            prefix = f"horizon_{label}_{quantity}"
            metrics[f"{prefix}_marginal_coverage_90"] = marginal
            metrics[f"{prefix}_joint_coverage_90"] = joint
            metrics[f"{prefix}_rms_z"] = rms_z

    position_width = 2.0 * CALIBRATION_Z * future_position_variance[:, -1].sqrt()
    velocity_width = 2.0 * CALIBRATION_Z * future_velocity_variance[:, -1].sqrt()
    drag_width = 2.0 * CALIBRATION_Z * current_log_drag_variance.sqrt()
    for quantity, width, unit in (
        ("position", position_width, "m"),
        ("velocity", velocity_width, "mps"),
        ("log_drag", drag_width, ""),
    ):
        suffix = f"_{unit}" if unit else ""
        metrics[f"horizon_2.00_{quantity}_mean_width_90{suffix}"] = float(
            width.to(torch.float64).mean()
        )
        metrics[f"horizon_2.00_{quantity}_max_width_90{suffix}"] = float(width.max())

    sum_names = (
        "identity_switch_count",
        "persistent_id_mismatch_count",
        "association_ambiguous_pair_count",
        "drag_grid_boundary_selection_count",
        "public_rollout_output_alias_count",
    )
    for name in sum_names:
        metrics[name] = float(sum(diagnostic_values(name)))
    mismatch = metrics["persistent_id_mismatch_count"]
    metrics["identity_coverage"] = float(1.0 - mismatch / (64.0 * 16.0 * 2.0))
    matched = sum(diagnostic_values("association_matched"))
    opportunities = sum(diagnostic_values("association_opportunities"))
    if opportunities <= 0.0:
        raise ValueError("association evidence has no measured opportunities")
    metrics["association_pair_coverage"] = float(matched / opportunities)
    direct_valid = sum(diagnostic_values("direct_velocity_valid"))
    direct_total = sum(diagnostic_values("direct_velocity_total"))
    if direct_total <= 0.0:
        raise ValueError("direct evidence has no measured ownership denominator")
    metrics["direct_velocity_valid_fraction"] = float(direct_valid / direct_total)
    for diagnostic, minimum_name, maximum_name in (
        (
            "direct_velocity_calls",
            "direct_velocity_calls_per_scene_min",
            "direct_velocity_calls_per_scene_max",
        ),
        ("ingested_frame_count", "ingested_frame_count_min", "ingested_frame_count_max"),
        ("state_ingest_count", "state_ingest_count_min", "state_ingest_count_max"),
        (
            "history_sample_count",
            "history_sample_count_per_scene_min",
            "history_sample_count_per_scene_max",
        ),
        (
            "history_valid_count",
            "history_valid_count_per_scene_min",
            "history_valid_count_per_scene_max",
        ),
        (
            "public_predict_calls",
            "public_predict_calls_per_scene_min",
            "public_predict_calls_per_scene_max",
        ),
    ):
        values = diagnostic_values(diagnostic)
        metrics[minimum_name] = float(min(values))
        metrics[maximum_name] = float(max(values))
    metrics["persistent_object_id_min"] = float(min(diagnostic_values("persistent_object_id_min")))
    metrics["persistent_object_id_max"] = float(max(diagnostic_values("persistent_object_id_max")))
    for diagnostic in (
        "direct_velocity_position_change_max_abs_m",
        "direct_fit_position_owner_max_abs_m",
        "direct_fit_velocity_owner_max_abs_mps",
        "direct_fit_log_drag_owner_max_abs",
        "semigroup_position_max_abs_m",
        "semigroup_velocity_max_abs_mps",
        "analytic_position_agreement_max_abs_m",
        "analytic_velocity_agreement_max_abs_mps",
        "future_position_variance_partition_max_abs",
        "future_velocity_variance_partition_max_abs",
        "public_direct_position_max_abs_m",
        "public_direct_velocity_max_abs_mps",
        "public_query_time_max_abs_seconds",
        "maximum_boundary_mass",
        "process_max_rss_bytes",
    ):
        metrics[diagnostic] = float(max(diagnostic_values(diagnostic)))
    metrics["minimum_drag_excitation_m"] = float(
        min(diagnostic_values("minimum_drag_excitation_m"))
    )
    metrics["minimum_profile_information"] = float(
        min(diagnostic_values("minimum_profile_information"))
    )
    for diagnostic in (
        "module_tensor_buffer_count",
        "persistent_module_state_key_count",
        "persistent_module_state_bytes",
        "learned_parameter_count",
        "learned_parameter_bytes",
    ):
        values = diagnostic_values(diagnostic)
        if len(set(values)) != 1:
            raise RuntimeError(f"model-state diagnostic {diagnostic!r} changed across scenes")
        metrics[diagnostic] = values[0]
    grid_points = diagnostic_values("drag_grid_point_count")
    if len(set(grid_points)) != 1:
        raise RuntimeError("runtime drag grid changed across governed scenes")
    metrics["drag_grid_point_count"] = grid_points[0]
    metrics["evaluation_latency_seconds"] = float(
        sum(diagnostic_values("evaluation_latency_seconds"))
    )
    metrics["vjp_latency_seconds"] = float(sum(diagnostic_values("vjp_latency_seconds")))

    metrics["position_variance_floor_clamp_count"] = float(
        sum(item.position_floor_clamps for item in deployed)
        + sum(diagnostic_values("position_variance_floor_clamp_count"))
    )
    metrics["position_variance_ceiling_clamp_count"] = float(
        sum(item.position_ceiling_clamps for item in deployed)
        + sum(diagnostic_values("position_variance_ceiling_clamp_count"))
    )
    metrics["velocity_variance_floor_clamp_count"] = float(
        sum(item.velocity_floor_clamps for item in deployed)
        + sum(diagnostic_values("velocity_variance_floor_clamp_count"))
    )
    metrics["velocity_variance_ceiling_clamp_count"] = float(
        sum(item.velocity_ceiling_clamps for item in deployed)
        + sum(diagnostic_values("velocity_variance_ceiling_clamp_count"))
    )
    metrics["drag_variance_floor_clamp_count"] = float(
        sum(item.drag_floor_clamps for item in deployed)
        + sum(diagnostic_values("drag_variance_floor_clamp_count"))
    )
    metrics["drag_variance_ceiling_clamp_count"] = float(
        sum(item.drag_ceiling_clamps for item in deployed)
        + sum(diagnostic_values("drag_variance_ceiling_clamp_count"))
    )

    pair_count = 0
    drag_swap_count = 0
    counterfactual_identity_mismatch = 0
    counterfactual_structure_mismatch = 0
    for ordinal in ORDINALS:
        twin = counterfactual_twin_ordinal(ordinal)
        if ordinal >= twin:
            continue
        pair_count += 1
        first = rows[ordinal]
        second = rows[twin]
        truth_structure = torch.equal(
            first.log_drag_truth.sort(dim=0).values,
            second.log_drag_truth.sort(dim=0).values,
        )
        if not truth_structure or first.scene_sha256 == second.scene_sha256:
            counterfactual_structure_mismatch += 1
        first_truth_low = int(first.log_drag_truth.argmin(dim=0).item())
        second_truth_low = int(second.log_drag_truth.argmin(dim=0).item())
        first_mean_low = int(first.log_drag_mean.argmin(dim=0).item())
        second_mean_low = int(second.log_drag_mean.argmin(dim=0).item())
        if first_mean_low != first_truth_low or second_mean_low != second_truth_low:
            counterfactual_identity_mismatch += 1
        if first_mean_low != second_mean_low:
            drag_swap_count += 1
    metrics["counterfactual_pair_count"] = float(pair_count)
    metrics["counterfactual_drag_swap_fraction"] = float(drag_swap_count / pair_count)
    metrics["counterfactual_identity_mismatch_count"] = float(counterfactual_identity_mismatch)
    metrics["counterfactual_structure_mismatch_count"] = float(counterfactual_structure_mismatch)

    metrics.update(
        {
            "optimizer_updates": 0.0,
            "optimizer_state_entry_count": 0.0,
            "rng_state_entry_count": 0.0,
        }
    )
    if type(vjp) is not dict:
        raise TypeError("VJP evidence must be an exact dict")
    for key, value in vjp.items():
        if type(key) is not str or type(value) is not float or not math.isfinite(value):
            raise ValueError("VJP evidence must contain exact finite float metrics")
        if key in metrics:
            raise ValueError(f"VJP evidence duplicates aggregate metric {key!r}")
        metrics[key] = value
    _validate_metrics(metrics)
    return metrics


def _split_result(
    *, split: str, metrics: Mapping[str, float], model_state_sha256: str
) -> dict[str, Any]:
    _validate_metrics(dict(metrics))
    failures = gate_failures(metrics)
    result = {
        "split": split,
        "manifest": list(_manifest_rows(split)),
        "manifest_sha256": MANIFEST_SHA256[split],
        "metrics": dict(metrics),
        "failures": failures,
        "passed": not failures,
        "optimizer_updates": 0,
        "runtime_api": {
            "packet_factory": "make_rgbd_packet",
            "ingest_frames": list(HISTORY_FRAME_INDICES),
            "rollout_method": "OnlineWorldModel.predict",
            "horizons_seconds": list(HORIZONS_SECONDS),
        },
        "scene_constructor": "private_capability_owned_identifiable_drag_episode",
        "model_state_sha256": validated_sha256(model_state_sha256, label=f"{split} model state"),
    }
    _validate_split_result(result, split=split)
    return result


def _collect_manifest_once(
    config: OrpheusConfig,
    *,
    split: str,
    manifest_capability: _ManifestCapability,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
    boundary_guard: Any,
) -> tuple[tuple[SceneSufficientEvidence, ...], dict[str, float]]:
    """Consume the exact 0..63 manifest in sixteen atomic four-row batches."""

    if not callable(boundary_guard):
        raise TypeError("manifest evaluation requires a callable source boundary guard")
    cache: list[SceneSufficientEvidence] = []
    vjp: dict[str, float] | None = None
    boundary_guard(f"{split} manifest before access")
    for start in range(0, 64, 4):
        boundary_guard(f"{split} batch {start // 4} before access")
        rows, batch_vjp = _evaluate_authorized_batch(
            config,
            split=split,
            ordinals=(start, start + 1, start + 2, start + 3),
            manifest_capability=manifest_capability,
            reviewed_state=reviewed_state,
            expected_state_sha256=expected_state_sha256,
            audit_vjp=start == 0,
        )
        if len(rows) != 4 or [row.ordinal for row in rows] != list(range(start, start + 4)):
            raise RuntimeError("authorized batch returned reordered or partial evidence")
        cache.extend(rows)
        if start == 0:
            if not batch_vjp:
                raise RuntimeError("first governed batch omitted mandatory VJP evidence")
            vjp = dict(batch_vjp)
        elif batch_vjp:
            raise RuntimeError("later governed batch unexpectedly repeated VJP access")
        boundary_guard(f"{split} batch {start // 4} after access")
    manifest_capability.finish_manifest()
    manifest_capability.require_finished()
    boundary_guard(f"{split} manifest after access")
    if vjp is None:
        raise RuntimeError("governed manifest omitted mandatory VJP evidence")
    return tuple(cache), vjp


def _evaluate_manifest_once(
    config: OrpheusConfig,
    *,
    split: str,
    manifest_capability: _ManifestCapability,
    reviewed_state: Mapping[str, Tensor],
    expected_state_sha256: str,
    position_scale: Tensor,
    velocity_scale: Tensor,
    drag_scale: Tensor,
    boundary_guard: Any,
) -> tuple[dict[str, Any], tuple[SceneSufficientEvidence, ...]]:
    cache, vjp = _collect_manifest_once(
        config,
        split=split,
        manifest_capability=manifest_capability,
        reviewed_state=reviewed_state,
        expected_state_sha256=expected_state_sha256,
        boundary_guard=boundary_guard,
    )
    metrics = _aggregate_split_metrics(
        cache,
        split=split,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
        drag_scale=drag_scale,
        config=config,
        vjp=vjp,
        cache_is_scale_one=False,
    )
    return _split_result(
        split=split,
        metrics=metrics,
        model_state_sha256=expected_state_sha256,
    ), tuple(cache)


def _calibrated_development_evidence(
    config: OrpheusConfig,
    *,
    cache: Sequence[SceneSufficientEvidence],
    vjp: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any], OnlineWorldModel]:
    """Calibrate once and derive all development metrics from cached evidence."""

    raw_model = new_public_model(config)
    raw_state_sha256 = _model_state_sha256(raw_model)
    position, velocity, drag = _calibrate_development_cache(cache)
    calibrated_model = new_public_model(config)
    observer = calibrated_model.observation_modules.get("rgbd")
    if observer is None or not hasattr(observer, "set_development_uncertainty_scales"):
        raise RuntimeError("public RGB-D model omitted the atomic development scale setter")
    setter_calls = 0
    observer.set_development_uncertainty_scales(
        position=position.scale,
        velocity=velocity.scale,
        drag=drag.scale,
    )
    setter_calls += 1
    calibrated_state_sha256 = _model_state_sha256(calibrated_model)
    calibrated_state = _scale_state(calibrated_model)
    for leaf, scale in zip(
        _SCALE_STATE_LEAVES,
        (position.scale, velocity.scale, drag.scale),
        strict=True,
    ):
        actual = calibrated_state[f"observation_modules.rgbd.{leaf}"]
        if _float32_bits(actual) != _float32_bits(scale):
            raise RuntimeError(f"atomic setter changed reviewed {leaf} bits")
    metrics = _aggregate_split_metrics(
        cache,
        split="development",
        position_scale=position.scale,
        velocity_scale=velocity.scale,
        drag_scale=drag.scale,
        config=config,
        vjp=vjp,
        cache_is_scale_one=True,
    )
    floor_clamps = int(
        metrics["position_variance_floor_clamp_count"]
        + metrics["velocity_variance_floor_clamp_count"]
        + metrics["drag_variance_floor_clamp_count"]
    )
    ceiling_clamps = int(
        metrics["position_variance_ceiling_clamp_count"]
        + metrics["velocity_variance_ceiling_clamp_count"]
        + metrics["drag_variance_ceiling_clamp_count"]
    )
    calibration = {
        "method": "designed_family_scene_max_rank_59_float32_minimal",
        "confidence": float(CALIBRATION_CONFIDENCE),
        "normal_z": float(CALIBRATION_Z),
        "rank": CALIBRATION_RANK,
        "scene_count": 64,
        "evidence_ingest_count": 64,
        "evidence_replay_count": 0,
        "atomic_setter_calls": setter_calls,
        "evidence_cache_sha256": _evidence_cache_sha256(cache),
        "raw_model_state_sha256": raw_state_sha256,
        "calibrated_model_state_sha256": calibrated_state_sha256,
        "gradient_audit_model_state_sha256": raw_state_sha256,
        "position": _scale_calibration_record(position),
        "velocity": _scale_calibration_record(velocity),
        "drag": _scale_calibration_record(drag),
        "variance_floor_clamp_count": floor_clamps,
        "variance_ceiling_clamp_count": ceiling_clamps,
    }
    _validate_calibration(calibration, cache=cache)
    development = _split_result(
        split="development",
        metrics=metrics,
        model_state_sha256=calibrated_state_sha256,
    )
    return development, calibration, calibrated_model


def _validate_split_calibration_binding(
    result: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    expected_state_sha256: str,
    label: str,
) -> None:
    if result["model_state_sha256"] != expected_state_sha256:
        raise ValueError(f"{label} split model state differs from reviewed state")
    if calibration["calibrated_model_state_sha256"] != expected_state_sha256:
        raise ValueError(f"{label} calibration state differs from reviewed state")
    for component, metric_name in (
        ("position", "position_uncertainty_scale"),
        ("velocity", "velocity_uncertainty_scale"),
        ("drag", "drag_uncertainty_scale"),
    ):
        bits = int(calibration[component]["deployed_float32_bits"][2:], 16)
        expected = float(_float32_from_bits(bits))
        actual = result["metrics"][metric_name]
        if type(actual) is not float or actual != expected:
            raise ValueError(f"{label} {component} scale differs from exact deployed bits")


def _development_report_is_valid(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> None:
    _validate_report_root(report, qualification=False, error=False)
    if report["source_provenance"] != source or report["publication_provenance"] != publication:
        raise ValueError("development report provenance differs from execution")
    if canonical_sha256(report["scene_family_certificate"]) != canonical_sha256(certificate):
        raise ValueError("development report certificate differs from execution")
    _validate_split_result(report["development"], split="development")
    _validate_calibration(report["calibration"])
    _validate_split_calibration_binding(
        report["development"],
        report["calibration"],
        expected_state_sha256=report["calibration"]["calibrated_model_state_sha256"],
        label="development",
    )
    if report["passed"] is not report["development"]["passed"]:
        raise ValueError("development report outcome differs from split gates")
    if report["review_ready"] is not report["passed"]:
        raise ValueError("development review readiness differs from passed state")
    if report["protected_data_materialized"] is not False:
        raise ValueError("development report cannot claim protected access")
    if report["stopped_after"] != "development":
        raise ValueError("development report stopped_after differs")
    if report["passed"]:
        if report["checkpoint"] != str(canonical_checkpoint_path()):
            raise ValueError("passed development report names wrong checkpoint")
        validated_sha256(report["checkpoint_sha256"], label="development checkpoint")
        if (
            report["checkpoint_model_state_sha256"]
            != report["calibration"]["calibrated_model_state_sha256"]
        ):
            raise ValueError("development report checkpoint state differs")
    elif any(
        report[name] is not None
        for name in ("checkpoint", "checkpoint_sha256", "checkpoint_model_state_sha256")
    ):
        raise ValueError("failed development report cannot publish checkpoint evidence")


def _qualification_report_is_valid(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> None:
    _validate_report_root(report, qualification=True, error=False)
    if report["source_provenance"] != source or report["publication_provenance"] != publication:
        raise ValueError("qualification report provenance differs from execution")
    if canonical_sha256(report["scene_family_certificate"]) != canonical_sha256(certificate):
        raise ValueError("qualification report certificate differs from execution")
    if report["qualification_ledger"] != str(qualification_ledger_path()):
        raise ValueError("qualification report names wrong ledger")
    _validate_split_result(report["development"], split="development")
    _validate_calibration(report["calibration"])
    _validate_split_calibration_binding(
        report["development"],
        report["calibration"],
        expected_state_sha256=report["model_state_sha256"],
        label="reviewed development",
    )
    opened: list[str] = []
    encountered_stop = False
    for split in _QualificationLedger.ORDER:
        result = report[split]
        if result is None:
            encountered_stop = True
            continue
        if encountered_stop:
            raise ValueError("qualification report opened a later split after a stop")
        _validate_split_result(result, split=split)
        _validate_split_calibration_binding(
            result,
            report["calibration"],
            expected_state_sha256=report["model_state_sha256"],
            label=split,
        )
        opened.append(split)
        if not result["passed"]:
            encountered_stop = True
    expected_passed = len(opened) == 3 and all(report[split]["passed"] for split in opened)
    if report["passed"] is not expected_passed:
        raise ValueError("qualification report outcome differs from protected gates")
    expected_stopped = opened[-1] if opened else "reviewed_development"
    if report["stopped_after"] != expected_stopped:
        raise ValueError("qualification stopped_after differs from protected access")
    if report["protected_data_materialized"] is not bool(opened):
        raise ValueError("qualification protected materialization flag differs")


def _development_error_report_is_valid(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> None:
    _validate_report_root(report, qualification=False, error=True)
    if (
        report["source_provenance"] != source
        or report["publication_provenance"] != publication
        or canonical_sha256(report["scene_family_certificate"]) != canonical_sha256(certificate)
        or report["development_ledger"] != str(development_ledger_path())
        or report["passed"] is not False
        or report["review_ready"] is not False
        or report["protected_data_materialized"] is not False
        or report["stopped_after"] != "development"
        or any(
            report[name] is not None
            for name in (
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_model_state_sha256",
            )
        )
    ):
        raise ValueError("development error report execution semantics differ")
    development = report["development"]
    calibration = report["calibration"]
    if (development is None) != (calibration is None):
        raise ValueError("development error report has partial scientific evidence")
    if development is not None:
        _validate_split_result(development, split="development")
        _validate_calibration(calibration)
        _validate_split_calibration_binding(
            development,
            calibration,
            expected_state_sha256=calibration["calibrated_model_state_sha256"],
            label="development error",
        )


def _qualification_error_report_is_valid(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    certificate: Mapping[str, Any],
    bindings: Mapping[str, Any],
    reviewed: Mapping[str, Any],
    ledger_record: Mapping[str, Any],
) -> None:
    _validate_report_root(report, qualification=True, error=True)
    if (
        report["source_provenance"] != source
        or report["publication_provenance"] != publication
        or canonical_sha256(report["scene_family_certificate"]) != canonical_sha256(certificate)
        or report["qualification_ledger"] != str(qualification_ledger_path())
        or report["passed"] is not False
        or report["development"] != reviewed["development"]
        or report["calibration"] != reviewed["calibration"]
        or canonical_sha256(report["calibration"]) != bindings["calibration_sha256"]
    ):
        raise ValueError("qualification error report reviewed evidence differs")
    for report_key, binding in (
        ("reviewed_checkpoint_sha256", "reviewed_checkpoint_sha256"),
        ("reviewed_development_report_sha256", "reviewed_development_report_sha256"),
        ("reviewed_development_ledger_sha256", "reviewed_development_ledger_sha256"),
        ("model_state_sha256", "model_state_sha256"),
    ):
        if report[report_key] != bindings[binding]:
            raise ValueError(f"qualification error {report_key} differs")
    _validate_split_calibration_binding(
        report["development"],
        report["calibration"],
        expected_state_sha256=report["model_state_sha256"],
        label="qualification error reviewed development",
    )
    started: list[str] = []
    encountered_unopened = False
    for split in _QualificationLedger.ORDER:
        state = ledger_record["splits"][split]
        if state["access_started"] is True:
            if encountered_unopened:
                raise ValueError("qualification error ledger opened a later split")
            started.append(split)
        elif state["access_started"] is False:
            encountered_unopened = True
        else:
            raise TypeError("qualification error access_started must be exact bool")
        result = report[split]
        if state["result_sha256"] is None:
            if result is not None:
                raise ValueError("qualification error report invents a split result")
            continue
        if result is None:
            raise ValueError("qualification error report omits a durable split result")
        _validate_split_result(result, split=split)
        _validate_split_calibration_binding(
            result,
            report["calibration"],
            expected_state_sha256=report["model_state_sha256"],
            label=f"qualification error {split}",
        )
        if state["result_sha256"] != canonical_sha256(dict(result)) or state["status"] != (
            "passed" if result["passed"] else "failed"
        ):
            raise ValueError("qualification error result differs from durable split")
    expected_stopped = started[-1] if started else "reviewed_development"
    if report["stopped_after"] != expected_stopped or report[
        "protected_data_materialized"
    ] is not bool(started):
        raise ValueError("qualification error stop/materialization semantics differ")


def _terminal_commit_matches_disk(
    ledger: object,
    *,
    qualification: bool,
) -> bool:
    """Detect a fully published terminal ledger before any report rewrite."""

    try:
        bindings = getattr(ledger, "_bindings", None)
        config = getattr(ledger, "_config", None)
        if type(bindings) is not dict:
            return False
        source = bindings.get("source_provenance")
        publication = bindings.get("publication_provenance")
        if type(source) is not dict or type(publication) is not dict:
            return False
        _require_config_matches_frozen_path(config, _frozen_config_path())
        current_source, current_publication, certificate = _current_execution_provenance(
            label="terminal commit reconciliation"
        )
        if (
            current_source != source
            or current_publication != publication
            or bindings.get("certificate_sha256") != certificate.get("certificate_sha256")
        ):
            return False
        ledger_path = qualification_ledger_path() if qualification else development_ledger_path()
        report_path = (
            canonical_qualification_report_path()
            if qualification
            else canonical_development_report_path()
        )
        _require_single_link_regular(ledger_path, label="terminal reconciliation ledger")
        _require_single_link_regular(report_path, label="terminal reconciliation report")
        ledger_contents = stable_read_bytes(ledger_path, label="terminal reconciliation ledger")
        report_contents = stable_read_bytes(report_path, label="terminal reconciliation report")
        record = _strict_json_loads(ledger_contents, label="terminal reconciliation ledger")
        report = _strict_json_loads(report_contents, label="terminal reconciliation report")
        report_digest = sha256_bytes(report_contents)
        if record.get("report_sha256") != report_digest:
            return False
        if qualification:
            _validate_run_tree(
                QUALIFICATION_ARTIFACT_NAMES,
                stage="qualification terminal reconciliation",
            )
            _qualification_report_is_valid(
                report,
                source=source,
                publication=publication,
                certificate=certificate,
            )
            for report_key, binding in (
                ("reviewed_checkpoint_sha256", "reviewed_checkpoint_sha256"),
                (
                    "reviewed_development_report_sha256",
                    "reviewed_development_report_sha256",
                ),
                (
                    "reviewed_development_ledger_sha256",
                    "reviewed_development_ledger_sha256",
                ),
                ("model_state_sha256", "model_state_sha256"),
            ):
                if report[report_key] != ledger._bindings[binding]:
                    return False
            if canonical_sha256(report["calibration"]) != bindings["calibration_sha256"]:
                return False
            reviewed = _validate_reviewed_bundle_from_disk(
                config=config,
                source=source,
                publication=publication,
                reviewed_checkpoint_sha256=bindings["reviewed_checkpoint_sha256"],
                reviewed_report_sha256=bindings["reviewed_development_report_sha256"],
                reviewed_ledger_sha256=bindings["reviewed_development_ledger_sha256"],
                expected_inventory=QUALIFICATION_ARTIFACT_NAMES,
            )
            if (
                report["development"] != reviewed["development"]
                or report["calibration"] != reviewed["calibration"]
                or reviewed["model_state_sha256"] != bindings["model_state_sha256"]
                or reviewed["calibration_sha256"] != bindings["calibration_sha256"]
            ):
                return False
            _validate_qualification_ledger_record(
                record,
                report=report,
                report_sha256=report_digest,
                bindings=bindings,
            )
        else:
            _development_report_is_valid(
                report,
                source=source,
                publication=publication,
                certificate=certificate,
            )
            passed = report["passed"] is True
            checkpoint_digest = record.get("checkpoint_sha256")
            expected_inventory = (
                DEVELOPMENT_ARTIFACT_NAMES
                if passed
                else frozenset({DEVELOPMENT_LEDGER_NAME, DEVELOPMENT_REPORT_NAME})
            )
            _validate_run_tree(
                expected_inventory,
                stage="development terminal reconciliation",
            )
            if report["checkpoint_sha256"] != checkpoint_digest:
                return False
            if passed:
                checkpoint_digest = validated_sha256(
                    checkpoint_digest,
                    label="terminal reconciliation checkpoint SHA-256",
                )
                checkpoint_path = canonical_checkpoint_path()
                _require_single_link_regular(
                    checkpoint_path,
                    label="terminal reconciliation checkpoint",
                )
                checkpoint_contents = stable_read_bytes(
                    checkpoint_path,
                    label="terminal reconciliation checkpoint",
                )
                if sha256_bytes(checkpoint_contents) != checkpoint_digest:
                    return False
                development, calibration = _validate_development_report(
                    report,
                    checkpoint_sha256=checkpoint_digest,
                    source=source,
                    publication=publication,
                    certificate=certificate,
                )
                _validate_checkpoint_evidence(
                    _checkpoint_payload_from_bytes(checkpoint_contents),
                    config=config,
                    source=source,
                    publication=publication,
                    development=development,
                    calibration=calibration,
                )
                _validate_development_ledger_record(
                    record,
                    report=report,
                    development=development,
                    report_sha256=report_digest,
                    checkpoint_sha256=checkpoint_digest,
                    source=source,
                    publication=publication,
                )
            elif checkpoint_digest is not None or _lexists(canonical_checkpoint_path()):
                return False
            _validate_terminal_development_ledger_record(
                record,
                report=report,
                report_sha256=report_digest,
                checkpoint_sha256=checkpoint_digest,
                bindings=bindings,
            )
        _fsync_parent(ledger_path)
        if (
            stable_read_bytes(
                ledger_path,
                label="terminal reconciliation ledger recheck",
            )
            != ledger_contents
            or stable_read_bytes(
                report_path,
                label="terminal reconciliation report recheck",
            )
            != report_contents
        ):
            return False
        checkpoint_recheck_passed = not (
            not qualification
            and report["passed"] is True
            and stable_read_bytes(
                canonical_checkpoint_path(),
                label="terminal reconciliation checkpoint recheck",
            )
            != checkpoint_contents
        )
        return checkpoint_recheck_passed
    except BaseException:
        return False


def _persist_development_error(
    *,
    report_path: Path,
    report: dict[str, Any],
    ledger: _DevelopmentLedger,
    error: BaseException,
) -> None:
    _revoke_ledger_governed_access(ledger)
    if ledger.record["result_sha256"] is None:
        report["development"] = None
        report["calibration"] = None
    elif (
        report["development"] is None
        or canonical_sha256(dict(report["development"])) != ledger.record["result_sha256"]
    ):
        raise PermissionError("failed development report lost durable result evidence")
    report["checkpoint"] = None
    report["checkpoint_sha256"] = None
    report["checkpoint_model_state_sha256"] = None
    report["passed"] = False
    report["review_ready"] = False
    report["protected_data_materialized"] = False
    report["stopped_after"] = "development"
    report["error"] = {"type": type(error).__name__, "message": str(error)}
    _development_error_report_is_valid(
        report,
        source=ledger._bindings["source_provenance"],
        publication=ledger._bindings["publication_provenance"],
        certificate=_frozen_scene_certificate_binding(),
    )
    digest = _persist_failed_report(report_path, report, label="identifiable-drag development")
    ledger.record_error(error, report_sha256=digest)


def _persist_qualification_error(
    *,
    report_path: Path,
    report: dict[str, Any],
    ledger: _QualificationLedger,
    error: BaseException,
) -> None:
    _revoke_ledger_governed_access(ledger)
    for split in _QualificationLedger.ORDER:
        durable_digest = ledger.record["splits"][split]["result_sha256"]
        if durable_digest is None:
            report[split] = None
        elif report[split] is None or canonical_sha256(dict(report[split])) != durable_digest:
            raise PermissionError(
                f"failed qualification report lost durable {split} result evidence"
            )
    started = [
        split
        for split in _QualificationLedger.ORDER
        if ledger.record["splits"][split]["access_started"] is True
    ]
    report["passed"] = False
    report["protected_data_materialized"] = bool(started)
    report["stopped_after"] = started[-1] if started else "reviewed_development"
    report["error"] = {"type": type(error).__name__, "message": str(error)}
    expected_inventory = _exact_error_inventory(qualification=True)
    reviewed = _validate_reviewed_bundle_from_disk(
        config=ledger._config,
        source=ledger._bindings["source_provenance"],
        publication=ledger._bindings["publication_provenance"],
        reviewed_checkpoint_sha256=ledger._bindings["reviewed_checkpoint_sha256"],
        reviewed_report_sha256=ledger._bindings["reviewed_development_report_sha256"],
        reviewed_ledger_sha256=ledger._bindings["reviewed_development_ledger_sha256"],
        expected_inventory=expected_inventory,
    )
    _qualification_error_report_is_valid(
        report,
        source=ledger._bindings["source_provenance"],
        publication=ledger._bindings["publication_provenance"],
        certificate=_frozen_scene_certificate_binding(),
        bindings=ledger._bindings,
        reviewed=reviewed,
        ledger_record=ledger.record,
    )
    digest = _persist_failed_report(report_path, report, label="identifiable-drag qualification")
    ledger.record_error(
        error,
        stopped_after=report["stopped_after"],
        report_sha256=digest,
    )


def run_development(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    source_provenance: Mapping[str, Any],
) -> int:
    """Consume development once, calibrate from cache, and publish review bytes."""

    _require_single_thread_execution()
    _require_canonical_path(config_path, _frozen_config_path(), label="config")
    _require_canonical_path(
        report_path, canonical_development_report_path(), label="development report"
    )
    _require_canonical_path(checkpoint_path, canonical_checkpoint_path(), label="checkpoint")
    if type(source_provenance) is not dict:
        raise TypeError("development source provenance must be an exact dict")
    source = clean_source(source_provenance, label="identifiable-drag development source")
    current_source, publication, certificate = _current_execution_provenance(
        label="identifiable-drag development preflight"
    )
    if source != current_source:
        raise ValueError("development source argument differs from live clean source")
    _require_config_matches_frozen_path(config, config_path)
    _require_single_link_regular(config_path, label="identifiable-drag frozen config")
    _validate_run_tree(frozenset(), stage="identifiable-drag pre-development")
    ledger_path = development_ledger_path()
    _validate_distinct_canonical_paths(
        {
            "config": config_path,
            "development_report": report_path,
            "checkpoint": checkpoint_path,
            "development_ledger": ledger_path,
        },
        atomic_writers=("development_report", "checkpoint", "development_ledger"),
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
    bindings = {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "development_manifest_sha256": MANIFEST_SHA256["development"],
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    authorization = _mint_run_authorization("development", bindings)
    ledger: _DevelopmentLedger | None = None
    try:
        ledger = _DevelopmentLedger(authorization, bindings, config=config)
        report: dict[str, Any] = {
            "artifact_kind": "rgbd_identifiable_drag_development",
            "protocol": protocol,
            "source_provenance": source,
            "publication_provenance": publication,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "critical_source_sha256": dict(FROZEN_SOURCE_SHA256),
            "scene_family_certificate": certificate,
            "development_ledger": str(ledger.path),
            "optimizer_updates": 0,
            "protected_data_materialized": False,
            "development": None,
            "calibration": None,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "checkpoint_model_state_sha256": None,
            "passed": False,
            "review_ready": False,
            "stopped_after": "development",
        }
    except BaseException:
        _LIVE_RUN_AUTHORIZATIONS.pop(id(authorization), None)
        if ledger is not None:
            _revoke_ledger_governed_access(ledger)
            _LIVE_PRIVATE_LEDGERS.pop(id(ledger), None)
            _LIVE_LEDGER_RECEIPTS.pop(id(ledger), None)
        raise

    def boundary_guard(label: str) -> None:
        _guard_frozen_inputs(
            source=source,
            publication=publication,
            config=config,
            config_path=config_path,
            expected_inventory=frozenset({DEVELOPMENT_LEDGER_NAME}),
            label=label,
        )

    try:
        _validate_run_tree(frozenset({DEVELOPMENT_LEDGER_NAME}), stage="development ledger created")
        cache, vjp = _collect_manifest_once(
            config,
            split="development",
            manifest_capability=ledger.capability(),
            reviewed_state=None,
            expected_state_sha256=None,
            boundary_guard=boundary_guard,
        )
        development, calibration, calibrated_model = _calibrated_development_evidence(
            config, cache=cache, vjp=vjp
        )
        report["development"] = development
        report["calibration"] = calibration
        report["passed"] = development["passed"]
        report["review_ready"] = development["passed"]
        ledger.complete_evaluation(development)
        checkpoint_digest: str | None = None
        if development["passed"]:
            _guard_frozen_inputs(
                source=source,
                publication=publication,
                config=config,
                config_path=config_path,
                expected_inventory=frozenset({DEVELOPMENT_LEDGER_NAME}),
                label="development before checkpoint write",
            )
            _save_review_checkpoint(
                checkpoint_path,
                model=calibrated_model,
                config=config,
                development=development,
                calibration=calibration,
                source=source,
                publication=publication,
            )
            _guard_frozen_inputs(
                source=source,
                publication=publication,
                config=config,
                config_path=config_path,
                expected_inventory=frozenset({DEVELOPMENT_LEDGER_NAME, CHECKPOINT_NAME}),
                label="development after checkpoint write",
            )
            checkpoint_contents = stable_read_bytes(
                checkpoint_path, label="development checkpoint receipt"
            )
            _validate_checkpoint_evidence(
                _checkpoint_payload_from_bytes(checkpoint_contents),
                config=config,
                source=source,
                publication=publication,
                development=development,
                calibration=calibration,
            )
            checkpoint_digest = sha256_bytes(checkpoint_contents)
            report["checkpoint"] = str(checkpoint_path)
            report["checkpoint_sha256"] = checkpoint_digest
            report["checkpoint_model_state_sha256"] = _model_state_sha256(calibrated_model)
        expected_before_report = (
            frozenset({DEVELOPMENT_LEDGER_NAME, CHECKPOINT_NAME})
            if development["passed"]
            else frozenset({DEVELOPMENT_LEDGER_NAME})
        )
        _guard_frozen_inputs(
            source=source,
            publication=publication,
            config=config,
            config_path=config_path,
            expected_inventory=expected_before_report,
            label="development before report write",
        )
        _development_report_is_valid(
            report,
            source=source,
            publication=publication,
            certificate=certificate,
        )
        _write_report_fresh(report_path, report)
        expected_terminal = (
            DEVELOPMENT_ARTIFACT_NAMES
            if development["passed"]
            else frozenset({DEVELOPMENT_LEDGER_NAME, DEVELOPMENT_REPORT_NAME})
        )
        _guard_frozen_inputs(
            source=source,
            publication=publication,
            config=config,
            config_path=config_path,
            expected_inventory=expected_terminal,
            label="development after report write",
        )
        written = stable_read_bytes(report_path, label="development report receipt")
        parsed = _strict_json_loads(written, label="development report receipt")
        _development_report_is_valid(
            parsed,
            source=source,
            publication=publication,
            certificate=certificate,
        )
        ledger.finish(
            report_sha256=sha256_bytes(written),
            checkpoint_sha256=checkpoint_digest,
        )
        return 0 if development["passed"] else 1
    except BaseException as error:
        _revoke_ledger_governed_access(ledger)
        terminal_committed = _terminal_commit_matches_disk(ledger, qualification=False)
        try:
            if (
                not terminal_committed
                and not isinstance(error, _PublishedReplacementError)
                and id(ledger) in _LIVE_PRIVATE_LEDGERS
            ):
                _persist_development_error(
                    report_path=report_path,
                    report=report,
                    ledger=ledger,
                    error=error,
                )
        finally:
            _revoke_ledger_governed_access(ledger)
            _LIVE_PRIVATE_LEDGERS.pop(id(ledger), None)
            _LIVE_LEDGER_RECEIPTS.pop(id(ledger), None)
        raise


def run_qualification(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    reviewed_checkpoint_sha256: str,
    reviewed_report_sha256: str,
    reviewed_development_ledger_sha256: str,
    source_provenance: Mapping[str, Any],
) -> int:
    """Consume selector, confirmation, then final once using reviewed state."""

    _require_single_thread_execution()
    _require_canonical_path(config_path, _frozen_config_path(), label="config")
    _require_canonical_path(
        report_path, canonical_qualification_report_path(), label="qualification report"
    )
    _require_canonical_path(checkpoint_path, canonical_checkpoint_path(), label="checkpoint")
    _require_canonical_path(
        development_report_path,
        canonical_development_report_path(),
        label="development report",
    )
    if type(source_provenance) is not dict:
        raise TypeError("qualification source provenance must be an exact dict")
    source = clean_source(source_provenance, label="identifiable-drag qualification source")
    current_source, publication, certificate = _current_execution_provenance(
        label="identifiable-drag qualification preflight"
    )
    if source != current_source:
        raise ValueError("qualification source argument differs from live clean source")
    _require_config_matches_frozen_path(config, config_path)
    _require_single_link_regular(config_path, label="identifiable-drag frozen config")
    _validate_run_tree(DEVELOPMENT_ARTIFACT_NAMES, stage="pre-qualification")
    qualification_path = qualification_ledger_path()
    _validate_distinct_canonical_paths(
        {
            "config": config_path,
            "qualification_report": report_path,
            "checkpoint": checkpoint_path,
            "development_report": development_report_path,
            "development_ledger": development_ledger_path(),
            "qualification_ledger": qualification_path,
        },
        atomic_writers=("qualification_report", "qualification_ledger"),
    )
    for path in (
        report_path,
        qualification_path,
        _atomic_temporary(report_path),
        _atomic_temporary(qualification_path),
    ):
        if _lexists(path):
            raise FileExistsError(f"qualification artifact must be fresh: {path}")
    checkpoint_digest = validated_sha256(
        reviewed_checkpoint_sha256, label="externally reviewed checkpoint"
    )
    development_report_digest = validated_sha256(
        reviewed_report_sha256, label="externally reviewed development report"
    )
    development_ledger_digest = validated_sha256(
        reviewed_development_ledger_sha256,
        label="externally reviewed development ledger",
    )
    seal, bindings, reviewed = _review_development(
        config=config,
        source=source,
        publication=publication,
        reviewed_checkpoint_sha256=checkpoint_digest,
        reviewed_report_sha256=development_report_digest,
        reviewed_ledger_sha256=development_ledger_digest,
    )
    authorization: _RunAuthorization | None = None
    ledger: _QualificationLedger | None = None
    try:
        authorization = _mint_owned_qualification_authorization(seal, bindings)
        ledger = _QualificationLedger(
            authorization,
            seal,
            bindings,
            config=config,
        )
        state = _validate_checkpoint_model_state(dict(reviewed["model_state"]))
        position_scale = state["observation_modules.rgbd.position_uncertainty_scale"]
        velocity_scale = state["observation_modules.rgbd.velocity_uncertainty_scale"]
        drag_scale = state["observation_modules.rgbd.drag_uncertainty_scale"]
        report: dict[str, Any] = {
            "artifact_kind": "rgbd_identifiable_drag_qualification",
            "protocol": bridge_protocol(),
            "source_provenance": source,
            "publication_provenance": publication,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "critical_source_sha256": dict(FROZEN_SOURCE_SHA256),
            "scene_family_certificate": certificate,
            "qualification_ledger": str(ledger.path),
            "reviewed_checkpoint_sha256": checkpoint_digest,
            "reviewed_development_report_sha256": development_report_digest,
            "reviewed_development_ledger_sha256": development_ledger_digest,
            "model_state_sha256": reviewed["model_state_sha256"],
            "optimizer_updates": 0,
            "development": reviewed["development"],
            "calibration": reviewed["calibration"],
            "selector": None,
            "confirmation": None,
            "final_test": None,
            "protected_data_materialized": False,
            "passed": False,
            "stopped_after": "reviewed_development",
        }
    except BaseException:
        if authorization is not None:
            _LIVE_RUN_AUTHORIZATIONS.pop(id(authorization), None)
        if ledger is not None:
            _revoke_ledger_governed_access(ledger)
            _LIVE_PRIVATE_LEDGERS.pop(id(ledger), None)
            _LIVE_LEDGER_RECEIPTS.pop(id(ledger), None)
        _LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(seal), None)
        raise

    def boundary_guard(label: str) -> None:
        _guard_frozen_inputs(
            source=source,
            publication=publication,
            config=config,
            config_path=config_path,
            expected_inventory=frozenset({*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_LEDGER_NAME}),
            label=label,
        )

    try:
        _validate_run_tree(
            frozenset({*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_LEDGER_NAME}),
            stage="qualification ledger created",
        )
        all_passed = True
        for split in _QualificationLedger.ORDER:
            result, _ = _evaluate_manifest_once(
                config,
                split=split,
                manifest_capability=ledger.begin_access(split),
                reviewed_state=state,
                expected_state_sha256=reviewed["model_state_sha256"],
                position_scale=position_scale,
                velocity_scale=velocity_scale,
                drag_scale=drag_scale,
                boundary_guard=boundary_guard,
            )
            report[split] = result
            report["protected_data_materialized"] = True
            report["stopped_after"] = split
            ledger.complete_split(split, result)
            if not result["passed"]:
                all_passed = False
                break
        report["passed"] = all_passed and report["final_test"] is not None
        ledger.prepare_report(passed=report["passed"], stopped_after=report["stopped_after"])
        boundary_guard("qualification before report write")
        _qualification_report_is_valid(
            report,
            source=source,
            publication=publication,
            certificate=certificate,
        )
        _write_report_fresh(report_path, report)
        _guard_frozen_inputs(
            source=source,
            publication=publication,
            config=config,
            config_path=config_path,
            expected_inventory=QUALIFICATION_ARTIFACT_NAMES,
            label="qualification after report write",
        )
        written = stable_read_bytes(report_path, label="qualification report receipt")
        parsed = _strict_json_loads(written, label="qualification report receipt")
        _qualification_report_is_valid(
            parsed,
            source=source,
            publication=publication,
            certificate=certificate,
        )
        ledger.finish(report_sha256=sha256_bytes(written))
        return 0 if report["passed"] else 1
    except BaseException as error:
        _revoke_ledger_governed_access(ledger)
        terminal_committed = _terminal_commit_matches_disk(ledger, qualification=True)
        try:
            if (
                not terminal_committed
                and not isinstance(error, _PublishedReplacementError)
                and id(ledger) in _LIVE_PRIVATE_LEDGERS
            ):
                _persist_qualification_error(
                    report_path=report_path,
                    report=report,
                    ledger=ledger,
                    error=error,
                )
        finally:
            _revoke_ledger_governed_access(ledger)
            _LIVE_PRIVATE_LEDGERS.pop(id(ledger), None)
            _LIVE_LEDGER_RECEIPTS.pop(id(ledger), None)
            _LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(seal), None)
        raise


__all__ = [
    "ARCHITECTURE_ATTEMPT",
    "ARCHITECTURE_VERSION",
    "DEFAULT_GATES",
    "FROZEN_CONFIG_SHA256",
    "FROZEN_SOURCE_SHA256",
    "GATE_METRIC_SCHEMA",
    "MANIFEST_SHA256",
    "PUBLIC_CALIBRATION_REGRESSION",
    "bridge_protocol",
    "canonical_checkpoint_path",
    "canonical_development_report_path",
    "canonical_qualification_report_path",
    "development_ledger_path",
    "gate_failures",
    "new_public_model",
    "qualification_ledger_path",
    "require_frozen_config",
    "run_development",
    "run_qualification",
]
