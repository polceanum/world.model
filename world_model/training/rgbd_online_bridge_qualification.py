"""Frozen qualification harness for the public one-slot RGB-D runtime bridge.

This module deliberately separates *qualification source* from *qualification
execution*.  Importing it, printing :func:`bridge_protocol`, validating a
config, and running its unit tests never materialize a simulator episode.  The
only function that calls the episode generator is :func:`evaluate_seed_manifest`,
after it has verified an exact predeclared manifest.  Protected-manifest
authorization remains the runner's responsibility and is durably recorded
before this function is called.

The qualified implementation owns no optimizer, learned parameters, or module
state.  Frames 0..15 enter only through ``make_rgbd_packet`` and
``OnlineWorldModel.ingest``; every future is queried through the public
``OnlineWorldModel.predict`` API.  The already-qualified standalone estimator
is retained solely as a same-input bridge-agreement oracle.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import resource
import stat
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import Tensor

from world_model.datasets import collate_episodes
from world_model.observations import DirectVelocityEvidence
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.runtime.state import runtime_stream_key
from world_model.training.checkpointing import capture_git_metadata, save_checkpoint
from world_model.training.loop import make_rgbd_packet
from world_model.training.rgbd_temporal_free_motion import (
    RGBDTemporalFreeMotionEstimator,
)
from world_model.utils.config import OrpheusConfig, load_config

DEVELOPMENT_SEEDS = tuple(range(45_000_000, 45_000_024))
SELECTOR_SEEDS = tuple(range(46_000_000, 46_000_016))
CONFIRMATION_SEEDS = tuple(range(47_000_000, 47_000_016))
FINAL_TEST_SEEDS = tuple(range(48_000_000, 48_000_032))

HISTORY_FRAME_INDICES = tuple(range(16))
ANCHOR_FRAME_INDEX = 15
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TARGET_FRAME_INDICES = (17, 20, 25, 35, 55)
RUNTIME_STREAM_KEY = "rgbd:camera0:rgbd"
HISTORY_GRADIENT_TARGETS = (
    "current_velocity",
    *(f"horizon_{horizon:.2f}_position" for horizon in HORIZONS_SECONDS),
    *(f"horizon_{horizon:.2f}_velocity" for horizon in HORIZONS_SECONDS),
)

ARCHITECTURE_VERSION = 1
ARCHITECTURE_ATTEMPT = 1
OPTIMIZER_UPDATES = 0

# Raw bytes, not merely parsed values, are part of the reviewed protocol.
FROZEN_CONFIG_SHA256 = "c40b3438c7fd60646d356db3fe54050039912ace288d9db89620b626106993a3"


@dataclass(frozen=True)
class RGBDOnlineBridgeGates:
    """Predeclared scalar gates independently applied to every split."""

    current_position_rmse_m: float = 0.010
    current_position_axis_rmse_m: float = 0.012
    current_velocity_rmse_mps: float = 0.010
    current_velocity_axis_rmse_mps: float = 0.015
    horizon_position_rmse_m: tuple[float, ...] = (0.011, 0.012, 0.015, 0.020, 0.030)
    horizon_position_axis_rmse_m: tuple[float, ...] = (
        0.014,
        0.015,
        0.018,
        0.024,
        0.035,
    )
    horizon_velocity_rmse_mps: float = 0.010
    horizon_velocity_axis_rmse_mps: float = 0.015
    horizon_position_error_growth_m: tuple[float, ...] = (
        0.004,
        0.006,
        0.009,
        0.014,
        0.024,
    )
    maximum_position_error_growth_slope_mps: float = 0.0125
    early_stationary_additive_margin_m: float = 0.003
    long_stationary_rmse_ratio: float = 0.75
    zero_velocity_rmse_ratio: float = 0.60

    public_standalone_current_position_rmse_m: float = 0.012
    public_standalone_current_velocity_rmse_mps: float = 0.015
    public_standalone_horizon_position_rmse_m: tuple[float, ...] = (
        0.013,
        0.015,
        0.020,
        0.028,
        0.045,
    )
    public_standalone_horizon_velocity_rmse_mps: float = 0.015
    history_standalone_measurement_max_abs_m: float = 1.0e-6
    fixed_prior_max_abs: float = 1.0e-7

    identity_change_count: float = 0.0
    persistent_object_id: float = 0.0
    active_fraction: float = 1.0
    rollout_active_fraction: float = 1.0
    history_sample_count: float = 16.0
    history_valid_count: float = 16.0
    history_span_seconds: float = 0.75
    history_span_tolerance_seconds: float = 1.0e-6
    direct_velocity_calls_per_batch: float = 1.0
    direct_velocity_valid_fraction: float = 1.0
    position_owner_count: float = 1.0
    direct_metric_position_owner_max_abs_m: float = 1.0e-7
    direct_position_field_count: float = 0.0
    direct_velocity_position_change_max_abs_m: float = 0.0
    public_rollout_output_alias_count: float = 0.0
    public_query_time_max_abs_seconds: float = 1.0e-6

    missing_depth_last_measurement_valid_fraction: float = 0.0
    missing_depth_fit_valid_fraction: float = 0.0
    missing_depth_direct_velocity_calls: float = 0.0
    missing_depth_history_sample_count: float = 16.0
    missing_depth_history_valid_count: float = 15.0
    missing_depth_finite_fraction: float = 1.0
    no_foreground_last_measurement_valid_fraction: float = 0.0
    no_foreground_fit_valid_fraction: float = 0.0
    no_foreground_direct_velocity_calls: float = 0.0
    no_foreground_history_sample_count: float = 16.0
    no_foreground_history_valid_count: float = 15.0
    no_foreground_finite_fraction: float = 1.0

    semigroup_position_max_abs_m: float = 1.0e-5
    semigroup_velocity_max_abs_mps: float = 1.0e-5
    public_direct_position_max_abs_m: float = 1.0e-6
    public_direct_velocity_max_abs_mps: float = 1.0e-6
    minimum_input_gradient_l1: float = 1.0e-12
    maximum_input_gradient_l1: float = 1.0e8
    minimum_history_frame_gradient_l1: float = 1.0e-14
    required_history_gradient_frames: float = 16.0

    perception_latency_seconds: float = 2.0
    state_only_rollout_latency_seconds: float = 0.05
    # Full batch-four persistent tensor state is counted recursively and
    # storage-uniquely across runtime state and retained measurement/update
    # diagnostics. The seed-free architecture fixture uses 25,364 B.
    persistent_runtime_tensor_state_bytes: int = 32_768
    process_max_rss_bytes: int = 2_500_000_000
    process_rss_delta_bytes: int = 1_000_000_000


DEFAULT_GATES = RGBDOnlineBridgeGates()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _assert_seed_namespaces() -> None:
    namespaces = (
        DEVELOPMENT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    flattened = [seed for namespace in namespaces for seed in namespace]
    if any(not namespace for namespace in namespaces):
        raise RuntimeError("every RGB-D online bridge namespace must be nonempty")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("RGB-D online bridge namespaces must be disjoint")


def bridge_protocol() -> dict[str, Any]:
    """Return the immutable, canonical-hashable bridge qualification contract."""

    _assert_seed_namespaces()
    protocol: dict[str, Any] = {
        "name": "rgbd_online_free_motion_bridge_v1",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "optimizer": None,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "manifests": {
            "development": list(DEVELOPMENT_SEEDS),
            "selector": list(SELECTOR_SEEDS),
            "confirmation": list(CONFIRMATION_SEEDS),
            "final_test": list(FINAL_TEST_SEEDS),
        },
        "runtime": {
            "observation_factory": "world_model.training.loop.make_rgbd_packet",
            "runtime": "world_model.runtime.OnlineWorldModel",
            "ingested_frame_indices": list(HISTORY_FRAME_INDICES),
            "ingested_frame_count": 16,
            "anchor_frame_index": ANCHOR_FRAME_INDEX,
            "public_rollout_method": "OnlineWorldModel.predict",
            "evaluation_rgb_only": False,
            "horizon_offsets_seconds": list(HORIZONS_SECONDS),
            "target_frame_indices": list(TARGET_FRAME_INDICES),
            "stream_key": RUNTIME_STREAM_KEY,
            "history_ownership": "raw_metric_associated_positions_bounded_16",
            "position_owner": "single_direct_metric_measurement_update",
            "direct_kinematic_evidence": "velocity_only_no_position_fields",
            "learned_parameters": 0,
            "persistent_module_tensor_state": 0,
        },
        "fixed_priors": {
            "radius_source": "model.rgbd.world_radius",
            "drag_source": "model.rgbd.linear_drag",
            "required_radius_m": 0.21,
            "required_linear_drag": 0.05,
            "simulator_drag_must_equal_model_prior": True,
            "standalone_and_public_must_share_prior": True,
        },
        "agreement_oracle": {
            "class": (
                "world_model.training.rgbd_temporal_free_motion.RGBDTemporalFreeMotionEstimator"
            ),
            "inputs": "the_exact_same_development_rgb_depth_calibration_and_timestamps",
            "role": "bridge_agreement_only_not_a_second_runtime",
        },
        "differentiability": {
            "kind": "fixed_output_vector_jacobian_products",
            "coefficients": [0.5, -0.75, 1.25],
            "inputs": ["rgb", "depth"],
            "outputs": [
                "current_position",
                "current_velocity",
                "position_and_velocity_at_every_declared_horizon",
            ],
            "history_support_gate": {
                "targets": list(HISTORY_GRADIENT_TARGETS),
                "per_target_and_modality_required_nonzero_frames": 16,
                "reason": "aggregate_VJP_must_not_hide_a_detached_history_slice",
            },
        },
        "missing_depth": {
            "ablation": "zero_the_sixteenth_depth_frame",
            "required_history_samples": 16,
            "required_valid_samples": 15,
            "required_temporal_fit": "invalid",
            "fallback": None,
        },
        "no_foreground": {
            "ablation": "zero_the_sixteenth_RGB_frame_while_depth_remains_positive",
            "required_history_samples": 16,
            "required_valid_samples": 15,
            "required_temporal_fit": "invalid",
            "depth_without_foreground_is_valid": False,
            "fallback": None,
        },
        "protected_access": {
            "order": ["selector", "confirmation", "final_test"],
            "record_before_materialization": True,
            "exactly_once_exclusive_durable_ledger": True,
            "final_unopened_until_both_predecessors_pass": True,
            "reviewed_development_report_and_checkpoint_hashes_required": True,
            "clean_source_config_protocol_and_model_state_required": True,
            "artifact_and_atomic_temporary_aliases_rejected": True,
        },
        "gates": asdict(DEFAULT_GATES),
        "execution": {
            "device": "cpu_float32",
            "torch_intraop_threads": 1,
            "batch_size": 4,
            "latency_warmups": 1,
            "perception_latency_repeats": 3,
            "state_only_rollout_latency_repeats": 20,
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def assert_rgbd_online_bridge_config(config: OrpheusConfig) -> None:
    """Reject every silent change to the first public RGB-D bridge contract."""

    simulator = config.simulator
    simulator_expected: dict[str, Any] = {
        "image_size": (64, 64),
        "frame_rate": 20,
        "physics_rate": 120,
        "sequence_frames": 56,
        "min_objects": 1,
        "max_objects": 1,
        "gravity": (0.0, 0.0, 0.0),
        "radius_range": (0.21, 0.21),
        "mass_range": (1.0, 1.0),
        "restitution_range": (0.7, 0.7),
        "drag_range": (0.05, 0.05),
        "friction_range": (0.2, 0.2),
        "initial_speed_range": (0.035, 0.035),
        "camera_motion": "fixed",
        "render_noise_std": 0.0,
        "ensure_collision": False,
        "external_impulse_probability": 0.0,
        "scenario_mixture": ("baseline",),
    }
    for name, required in simulator_expected.items():
        actual = getattr(simulator, name)
        if actual != required:
            raise ValueError(
                f"RGB-D online bridge requires simulator.{name}={required!r}, got {actual!r}"
            )

    if config.project.seed != DEVELOPMENT_SEEDS[0] or not config.project.deterministic:
        raise ValueError("RGB-D online bridge project seed/determinism differs from protocol")
    if config.device.preference != "cpu" or config.device.cuda_amp or config.device.compile:
        raise ValueError("RGB-D online bridge qualification requires CPU float32 without compile")
    if config.model.max_objects != 1:
        raise ValueError("RGB-D online bridge requires exactly one persistent object slot")
    if config.model.rgb.enabled or not config.model.rgbd.enabled:
        raise ValueError("RGB-D online bridge requires only the composite RGB-D observation path")

    rgbd = config.model.rgbd
    rgbd_expected: dict[str, Any] = {
        "global_every_steps": 1,
        "world_radius": 0.21,
        "linear_drag": 0.05,
        "foreground_threshold": 0.04,
        "foreground_temperature": 0.01,
        "minimum_mass": 4.0,
        "measurement_position_variance": 0.000064,
        "temporal_history_size": 16,
        "temporal_min_samples": 16,
        "temporal_min_dt": 0.001,
        "temporal_velocity_variance_floor": 0.000001,
        "temporal_velocity_variance_ceiling": 0.01,
        "fit_conditioning_limit": 100.0,
    }
    for name, required in rgbd_expected.items():
        actual = getattr(rgbd, name)
        if actual != required:
            raise ValueError(
                f"RGB-D online bridge requires model.rgbd.{name}={required!r}, got {actual!r}"
            )
    if simulator.radius_range != (rgbd.world_radius, rgbd.world_radius):
        raise ValueError("public and standalone radius priors must equal the simulator radius")
    if simulator.drag_range != (rgbd.linear_drag, rgbd.linear_drag):
        raise ValueError("public and standalone drag priors must equal simulator linear drag")

    if not config.model.dynamics.analytic_free_motion_only:
        raise ValueError("RGB-D online bridge requires analytic-free-motion-only dynamics")
    if config.model.dynamics.attention_residual_enabled:
        raise ValueError("RGB-D online bridge forbids learned dynamics residuals")
    if (
        config.model.filter.enable_learned_corrector
        or config.model.filter.learned_residual_scale != 0.0
    ):
        raise ValueError("RGB-D online bridge forbids a learned filter corrector")
    if not config.model.filter.direct_metric_position_update:
        raise ValueError("RGB-D online bridge requires the direct metric position update")
    if config.model.identification.enabled:
        raise ValueError("RGB-D online bridge forbids online parameter identification")
    if config.runtime.modality != "rgbd" or config.runtime.enable_debug_oracle:
        raise ValueError("RGB-D online bridge forbids oracle/runtime modality substitution")
    if config.runtime.hypothesis_pool_enabled:
        raise ValueError("RGB-D online bridge forbids a hypothesis-pool dynamics substitution")
    if not config.runtime.strict_timestamps:
        raise ValueError("RGB-D online bridge requires strict timestamps")
    if config.training.batch_size != 4 or config.training.steps != 1:
        raise ValueError("shared bridge config must retain batch four and one schema-only step")
    if config.training.rgb_pretrain_steps != 0:
        raise ValueError("RGB-D online bridge has no RGB pretraining or optimizer phase")
    if config.training.validation_episodes != len(DEVELOPMENT_SEEDS):
        raise ValueError("validation_episodes must match the development manifest")
    if tuple(config.evaluation.horizons_seconds) != HORIZONS_SECONDS:
        raise ValueError(f"evaluation horizons must be exactly {HORIZONS_SECONDS!r}")
    if config.evaluation.rgb_only:
        raise ValueError("RGB-D online bridge requires evaluation.rgb_only=false")

    derived_targets = tuple(
        ANCHOR_FRAME_INDEX + int(round(horizon * simulator.frame_rate))
        for horizon in HORIZONS_SECONDS
    )
    if derived_targets != TARGET_FRAME_INDICES:
        raise RuntimeError("target frames do not match the exact public rollout horizons")
    if TARGET_FRAME_INDICES[-1] != simulator.sequence_frames - 1:
        raise ValueError("the two-second target must remain the final episode frame")


def new_public_model(config: OrpheusConfig) -> OnlineWorldModel:
    """Construct the sole public runtime admitted by the frozen config."""

    assert_rgbd_online_bridge_config(config)
    model = OnlineWorldModel.from_config(config, device="cpu")
    if model.belief_factory.initial_radius != config.model.rgbd.world_radius:
        raise RuntimeError("public birth radius was not seeded from model.rgbd.world_radius")
    if model.belief_factory.initial_drag != config.model.rgbd.linear_drag:
        raise RuntimeError("public birth drag was not seeded from model.rgbd.linear_drag")
    if tuple(model.parameters()) or tuple(model.buffers()) or model.state_dict():
        raise RuntimeError("public RGB-D bridge must own no parameter or tensor module state")
    return model


def new_standalone_estimator(config: OrpheusConfig) -> RGBDTemporalFreeMotionEstimator:
    """Construct the same-prior standalone agreement oracle."""

    assert_rgbd_online_bridge_config(config)
    estimator = RGBDTemporalFreeMotionEstimator(
        image_size=config.simulator.image_size,
        world_radius_m=config.model.rgbd.world_radius,
        gravity=config.simulator.gravity,
        drag=config.model.rgbd.linear_drag,
        horizons_seconds=HORIZONS_SECONDS,
    )
    if tuple(estimator.parameters()) or tuple(estimator.buffers()) or estimator.state_dict():
        raise RuntimeError("standalone agreement estimator must remain parameter-free")
    return estimator


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Recompute every independent split gate from scalar report evidence."""

    gates = DEFAULT_GATES
    failures: list[str] = []

    def require_max(key: str, maximum: float) -> None:
        value = _numeric(metrics.get(key))
        if value is None:
            failures.append(f"{key}:missing_or_nonfinite")
        elif value > maximum:
            failures.append(f"{key}:{value:.9g}>{maximum:.9g}")

    def require_min(key: str, minimum: float) -> None:
        value = _numeric(metrics.get(key))
        if value is None:
            failures.append(f"{key}:missing_or_nonfinite")
        elif value < minimum:
            failures.append(f"{key}:{value:.9g}<{minimum:.9g}")

    def require_equal(key: str, expected: float) -> None:
        value = _numeric(metrics.get(key))
        if value is None:
            failures.append(f"{key}:missing_or_nonfinite")
        elif value != expected:
            failures.append(f"{key}:{value:.9g}!={expected:.9g}")

    require_max("current_position_rmse_m", gates.current_position_rmse_m)
    require_max("current_position_axis_rmse_m", gates.current_position_axis_rmse_m)
    require_max("current_velocity_rmse_mps", gates.current_velocity_rmse_mps)
    require_max("current_velocity_axis_rmse_mps", gates.current_velocity_axis_rmse_mps)
    require_max("horizon_velocity_rmse_mps", gates.horizon_velocity_rmse_mps)
    require_max("horizon_velocity_axis_rmse_mps", gates.horizon_velocity_axis_rmse_mps)
    require_max(
        "maximum_position_error_growth_slope_mps",
        gates.maximum_position_error_growth_slope_mps,
    )
    for index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        require_max(
            f"horizon_{label}_position_rmse_m",
            gates.horizon_position_rmse_m[index],
        )
        require_max(
            f"horizon_{label}_position_axis_rmse_m",
            gates.horizon_position_axis_rmse_m[index],
        )
        require_max(
            f"horizon_{label}_velocity_rmse_mps",
            gates.horizon_velocity_rmse_mps,
        )
        require_max(
            f"horizon_{label}_velocity_axis_rmse_mps",
            gates.horizon_velocity_axis_rmse_mps,
        )
        require_max(
            f"horizon_{label}_position_error_growth_m",
            gates.horizon_position_error_growth_m[index],
        )
        require_max(
            f"public_standalone_horizon_{label}_position_rmse_m",
            gates.public_standalone_horizon_position_rmse_m[index],
        )
        require_max(
            f"public_standalone_horizon_{label}_velocity_rmse_mps",
            gates.public_standalone_horizon_velocity_rmse_mps,
        )

    require_max(
        "early_stationary_additive_regression_m",
        gates.early_stationary_additive_margin_m,
    )
    require_max("long_stationary_rmse_ratio", gates.long_stationary_rmse_ratio)
    require_max("zero_velocity_rmse_ratio", gates.zero_velocity_rmse_ratio)
    require_max(
        "public_standalone_current_position_rmse_m",
        gates.public_standalone_current_position_rmse_m,
    )
    require_max(
        "public_standalone_current_velocity_rmse_mps",
        gates.public_standalone_current_velocity_rmse_mps,
    )
    require_max(
        "history_standalone_measurement_max_abs_m",
        gates.history_standalone_measurement_max_abs_m,
    )
    require_max("fixed_prior_max_abs", gates.fixed_prior_max_abs)

    require_equal("identity_change_count", gates.identity_change_count)
    require_equal("persistent_object_id_min", gates.persistent_object_id)
    require_equal("persistent_object_id_max", gates.persistent_object_id)
    require_min("active_fraction", gates.active_fraction)
    require_min("rollout_active_fraction", gates.rollout_active_fraction)
    require_equal("history_sample_count_min", gates.history_sample_count)
    require_equal("history_sample_count_max", gates.history_sample_count)
    require_equal("history_valid_count_min", gates.history_valid_count)
    require_equal("history_valid_count_max", gates.history_valid_count)
    require_max(
        "history_span_max_abs_error_seconds",
        gates.history_span_tolerance_seconds,
    )
    require_equal(
        "direct_velocity_calls_per_batch_min",
        gates.direct_velocity_calls_per_batch,
    )
    require_equal(
        "direct_velocity_calls_per_batch_max",
        gates.direct_velocity_calls_per_batch,
    )
    require_min("direct_velocity_valid_fraction", gates.direct_velocity_valid_fraction)
    require_equal("position_owner_count_min", gates.position_owner_count)
    require_equal("position_owner_count_max", gates.position_owner_count)
    require_max(
        "direct_metric_position_owner_max_abs_m",
        gates.direct_metric_position_owner_max_abs_m,
    )
    require_equal("direct_position_field_count", gates.direct_position_field_count)
    require_max(
        "direct_velocity_position_change_max_abs_m",
        gates.direct_velocity_position_change_max_abs_m,
    )
    require_equal(
        "public_rollout_output_alias_count",
        gates.public_rollout_output_alias_count,
    )
    require_max(
        "public_query_time_max_abs_seconds",
        gates.public_query_time_max_abs_seconds,
    )
    require_equal("ingested_frame_count_min", 16.0)
    require_equal("ingested_frame_count_max", 16.0)
    require_equal("public_predict_calls_per_batch_min", 1.0)
    require_equal("public_predict_calls_per_batch_max", 1.0)

    require_equal(
        "missing_depth_last_measurement_valid_fraction",
        gates.missing_depth_last_measurement_valid_fraction,
    )
    require_equal(
        "missing_depth_fit_valid_fraction",
        gates.missing_depth_fit_valid_fraction,
    )
    require_equal(
        "missing_depth_direct_velocity_calls",
        gates.missing_depth_direct_velocity_calls,
    )
    require_equal(
        "missing_depth_history_sample_count_min",
        gates.missing_depth_history_sample_count,
    )
    require_equal(
        "missing_depth_history_sample_count_max",
        gates.missing_depth_history_sample_count,
    )
    require_equal(
        "missing_depth_history_valid_count_min",
        gates.missing_depth_history_valid_count,
    )
    require_equal(
        "missing_depth_history_valid_count_max",
        gates.missing_depth_history_valid_count,
    )
    require_min("missing_depth_finite_fraction", gates.missing_depth_finite_fraction)
    require_equal(
        "no_foreground_last_measurement_valid_fraction",
        gates.no_foreground_last_measurement_valid_fraction,
    )
    require_equal(
        "no_foreground_fit_valid_fraction",
        gates.no_foreground_fit_valid_fraction,
    )
    require_equal(
        "no_foreground_direct_velocity_calls",
        gates.no_foreground_direct_velocity_calls,
    )
    require_equal(
        "no_foreground_history_sample_count_min",
        gates.no_foreground_history_sample_count,
    )
    require_equal(
        "no_foreground_history_sample_count_max",
        gates.no_foreground_history_sample_count,
    )
    require_equal(
        "no_foreground_history_valid_count_min",
        gates.no_foreground_history_valid_count,
    )
    require_equal(
        "no_foreground_history_valid_count_max",
        gates.no_foreground_history_valid_count,
    )
    require_min("no_foreground_finite_fraction", gates.no_foreground_finite_fraction)

    require_max("semigroup_position_max_abs_m", gates.semigroup_position_max_abs_m)
    require_max("semigroup_velocity_max_abs_mps", gates.semigroup_velocity_max_abs_mps)
    require_max("public_direct_position_max_abs_m", gates.public_direct_position_max_abs_m)
    require_max("public_direct_velocity_max_abs_mps", gates.public_direct_velocity_max_abs_mps)
    for output_name in (
        "current_position",
        "current_velocity",
        *(f"horizon_{horizon:.2f}_position" for horizon in HORIZONS_SECONDS),
        *(f"horizon_{horizon:.2f}_velocity" for horizon in HORIZONS_SECONDS),
    ):
        for modality in ("rgb", "depth"):
            key = f"gradient_l1/{output_name}/{modality}"
            require_min(key, gates.minimum_input_gradient_l1)
            require_max(key, gates.maximum_input_gradient_l1)
    for output_name in HISTORY_GRADIENT_TARGETS:
        for modality in ("rgb", "depth"):
            require_min(
                f"gradient_min_history_frame_l1/{output_name}/{modality}",
                gates.minimum_history_frame_gradient_l1,
            )
            require_equal(
                f"gradient_supported_history_frames/{output_name}/{modality}",
                gates.required_history_gradient_frames,
            )

    require_max("perception_latency_seconds", gates.perception_latency_seconds)
    require_max(
        "state_only_rollout_latency_seconds",
        gates.state_only_rollout_latency_seconds,
    )
    require_max(
        "persistent_runtime_tensor_state_bytes_max",
        float(gates.persistent_runtime_tensor_state_bytes),
    )
    require_max("process_max_rss_bytes", float(gates.process_max_rss_bytes))
    require_max("process_rss_delta_bytes", float(gates.process_rss_delta_bytes))
    for key in (
        "learned_parameter_count",
        "learned_parameter_bytes",
        "module_tensor_buffer_count",
        "persistent_module_state_key_count",
        "persistent_module_state_bytes",
        "optimizer_updates",
        "optimizer_state_entry_count",
    ):
        require_equal(key, 0.0)
    return failures


def _chunks(values: Sequence[int], size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield tuple(int(value) for value in values[start : start + size])


def _rmse(error: Tensor) -> float:
    return float(error.to(dtype=torch.float64).square().mean().sqrt())


def _axis_rmse(error: Tensor) -> Tensor:
    return error.to(dtype=torch.float64).reshape(-1, 3).square().mean(dim=0).sqrt()


def _process_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _history_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    indices = torch.tensor(HISTORY_FRAME_INDICES, dtype=torch.int64)
    return {
        "images": batch["rgb"].index_select(1, indices),
        "depth": batch["depth"].index_select(1, indices),
        "world_from_camera": batch["camera"]["world_from_camera"].index_select(1, indices),
        "intrinsics": batch["camera"]["intrinsics"].index_select(1, indices),
        "timestamps": batch["timestamps"].index_select(1, indices),
    }


def _target_tensors(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    future = torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64)
    return {
        "anchor_position": batch["objects"]["position"][:, ANCHOR_FRAME_INDEX, :1],
        "anchor_velocity": batch["objects"]["velocity"][:, ANCHOR_FRAME_INDEX, :1],
        "future_position": batch["objects"]["position"][:, :, :1].index_select(1, future),
        "future_velocity": batch["objects"]["velocity"][:, :, :1].index_select(1, future),
    }


def _assert_free_motion_batch(batch: Mapping[str, Any], seeds: Sequence[int]) -> None:
    events = batch["events"]
    if (
        bool(events["collision"].any())
        or bool(events["contact"].any())
        or bool(events["external_impulse"].ne(0).any())
        or bool(events["created"][:, 1:].any())
        or bool(events["removed"].any())
    ):
        raise RuntimeError(f"RGB-D online bridge manifest contains an event: {tuple(seeds)!r}")
    if not bool(batch["objects"]["active"][:, :, :1].all()):
        raise RuntimeError("the one qualified sphere must remain active")
    if not bool(batch["labels"]["projected_valid"][:, :, :1].all()):
        raise RuntimeError("the one qualified sphere must remain projectable")


def _storage_alias(left: Tensor, right: Tensor) -> bool:
    if left.numel() == 0 or right.numel() == 0:
        return False
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _persistent_runtime_tensor_bytes(model: OnlineWorldModel) -> int:
    """Count unique tensor storage retained by the public online runtime."""

    roots = {
        "state": model.state,
        "last_measurements": model.last_measurements,
        "last_direct_velocity_evidence": model.last_direct_velocity_evidence,
        "updater_diagnostics": model.updater.last_diagnostics,
        "scheduler_state": model.scheduler._sensor_state,
        "runtime_diagnostics": model.diagnostics.records,
    }
    pending: list[Any] = [roots]
    visited_objects: set[int] = set()
    storages: dict[tuple[str, int | None, int, int], int] = {}
    while pending:
        value = pending.pop()
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            continue
        if isinstance(value, Tensor):
            storage = value.untyped_storage()
            byte_count = int(storage.nbytes())
            if byte_count:
                key = (
                    value.device.type,
                    value.device.index,
                    int(storage.data_ptr()),
                    byte_count,
                )
                storages[key] = byte_count
            continue
        identity = id(value)
        if identity in visited_objects:
            continue
        visited_objects.add(identity)
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
    return sum(storages.values())


def _run_public_batch(
    batch: Mapping[str, Any],
    config: OrpheusConfig,
) -> dict[str, Any]:
    """Run exactly the admitted 16-packet public observation/prediction path."""

    batch_size = int(batch["rgb"].shape[0])
    model = new_public_model(config)
    model.eval()
    model.reset(batch_size=batch_size)
    original_correction = model.updater.correct_direct_velocity
    correction_audit: dict[str, Any] = {
        "calls": 0,
        "valid": 0,
        "total": 0,
        "position_fields": 0,
        "position_change_max_abs": 0.0,
    }

    def recording_correction(
        prior: Any,
        evidence: DirectVelocityEvidence,
    ) -> Any:
        evidence.validate()
        correction_audit["calls"] += 1
        correction_audit["valid"] += int(evidence.valid_mask.sum().detach().cpu())
        correction_audit["total"] += int(evidence.valid_mask.numel())
        correction_audit["position_fields"] += sum(
            field is not None
            for field in (
                evidence.position,
                evidence.position_log_variance,
                evidence.position_valid_mask,
            )
        )
        before = prior.objects.position
        corrected = original_correction(prior, evidence)
        change = (corrected.objects.position - before).abs().max()
        correction_audit["position_change_max_abs"] = max(
            float(correction_audit["position_change_max_abs"]),
            float(change.detach().cpu()),
        )
        return corrected

    # Instance-level instrumentation leaves production code untouched while
    # proving exactly what the public updater consumed on these inputs.
    model.updater.correct_direct_velocity = recording_correction  # type: ignore[method-assign]

    identities: list[Tensor] = []
    active_masks: list[Tensor] = []
    packet_count = 0
    for frame_index in HISTORY_FRAME_INDICES:
        packet = make_rgbd_packet(batch, frame_index)
        packet_count += 1
        posterior = model.ingest(packet)
        identities.append(posterior.objects.object_id)
        active_masks.append(posterior.objects.active)
    trajectory = model.predict(HORIZONS_SECONDS).validate()
    predict_count = 1

    belief = model.belief
    if belief is None:
        raise RuntimeError("public runtime failed to retain a belief after ingestion")
    history = model.state.temporal_histories.get(runtime_stream_key("rgbd", "camera0:rgbd"))
    if not isinstance(history, RGBDTemporalPositionHistory):
        raise RuntimeError("public runtime failed to retain the typed RGB-D history")
    if runtime_stream_key("rgbd", "camera0:rgbd") != RUNTIME_STREAM_KEY:
        raise RuntimeError("public RGB-D runtime stream key changed")

    standalone = new_standalone_estimator(config)
    standalone.eval()
    standalone_output = standalone(**_history_inputs(batch))

    direct = model.dynamics.predict(belief, 2.0)
    first = model.dynamics.predict(belief, 0.75)
    composed = model.dynamics.predict(first, 1.25)
    expected_times = belief.timestamp[:, None] + belief.timestamp.new_tensor(HORIZONS_SECONDS)
    alias_count = sum(
        (
            _storage_alias(trajectory.positions, belief.objects.position),
            _storage_alias(trajectory.positions, belief.objects.velocity),
            _storage_alias(trajectory.velocities, belief.objects.position),
            _storage_alias(trajectory.velocities, belief.objects.velocity),
        )
    )
    last_measurements = model.last_measurements
    if last_measurements is None:
        raise RuntimeError("public runtime did not expose final RGB-D measurements")
    if last_measurements.supported_state_fields != ("position",):
        raise RuntimeError("RGB-D metric measurement must be the sole public position owner")
    raw_position = last_measurements.auxiliary.get("world_position")
    if not isinstance(raw_position, Tensor):
        raise RuntimeError("RGB-D metric measurement did not expose its raw world position")
    position_owner_count = 1 + int(correction_audit["position_fields"] > 0)
    fixed_prior_max_abs = max(
        abs(model.belief_factory.initial_radius - config.model.rgbd.world_radius),
        abs(model.belief_factory.initial_drag - config.model.rgbd.linear_drag),
        abs(standalone.world_radius_m - config.model.rgbd.world_radius),
        abs(standalone.drag - config.model.rgbd.linear_drag),
        float((belief.objects.radius - config.model.rgbd.world_radius).abs().max()),
        float((belief.objects.drag - config.model.rgbd.linear_drag).abs().max()),
    )

    return {
        "model": model,
        "belief": belief,
        "trajectory": trajectory,
        "standalone": standalone_output,
        "history": history,
        "identities": torch.stack(identities, dim=1),
        "active_masks": torch.stack(active_masks, dim=1),
        "packet_count": packet_count,
        "predict_count": predict_count,
        "correction_audit": correction_audit,
        "semigroup_position": (composed.objects.position - direct.objects.position).abs(),
        "semigroup_velocity": (composed.objects.velocity - direct.objects.velocity).abs(),
        "public_direct_position": (trajectory.positions[:, -1] - direct.objects.position).abs(),
        "public_direct_velocity": (trajectory.velocities[:, -1] - direct.objects.velocity).abs(),
        "query_time_error": (trajectory.timestamps - expected_times).abs(),
        "output_alias_count": alias_count,
        "last_measurement_valid": last_measurements.measurement_mask,
        "position_owner_count": position_owner_count,
        "direct_metric_position_owner_error": (belief.objects.position - raw_position).abs(),
        "fixed_prior_max_abs": fixed_prior_max_abs,
        "runtime_tensor_bytes": _persistent_runtime_tensor_bytes(model),
    }


def _history_gradient_diagnostics(
    gradient: Tensor,
    *,
    output_name: str,
    modality: str,
) -> dict[str, float]:
    """Expose the weakest frame contribution so aggregate VJPs cannot mask detachments."""

    if gradient.ndim < 3 or gradient.shape[1] != len(HISTORY_FRAME_INDICES):
        raise ValueError("history input gradient must have explicit [B,16,...] frame slices")
    reduction_dimensions = (0, *range(2, gradient.ndim))
    per_frame_l1 = gradient.abs().sum(dim=reduction_dimensions)
    if per_frame_l1.shape != (len(HISTORY_FRAME_INDICES),):
        raise RuntimeError("history-frame gradient reduction produced the wrong shape")
    if not bool(torch.isfinite(per_frame_l1).all()):
        raise FloatingPointError(f"{output_name} has nonfinite {modality} frame VJPs")
    supported = per_frame_l1 >= DEFAULT_GATES.minimum_history_frame_gradient_l1
    return {
        f"gradient_min_history_frame_l1/{output_name}/{modality}": float(per_frame_l1.min()),
        f"gradient_supported_history_frames/{output_name}/{modality}": float(supported.sum()),
    }


def _gradient_metrics(config: OrpheusConfig, batch: Mapping[str, Any]) -> dict[str, float]:
    history_indices = torch.tensor(
        HISTORY_FRAME_INDICES,
        dtype=torch.int64,
        device=batch["rgb"].device,
    )
    differentiable_batch = dict(batch)
    differentiable_batch["rgb"] = (
        batch["rgb"][:1].index_select(1, history_indices).clone().requires_grad_(True)
    )
    differentiable_batch["depth"] = (
        batch["depth"][:1].index_select(1, history_indices).clone().requires_grad_(True)
    )
    differentiable_batch["camera"] = {
        name: value[:1].index_select(1, history_indices.to(value.device)).clone()
        if isinstance(value, Tensor)
        else value
        for name, value in batch["camera"].items()
    }
    differentiable_batch["timestamps"] = (
        batch["timestamps"][:1].index_select(1, history_indices).clone()
    )
    output = _run_public_batch(differentiable_batch, config)
    belief = output["belief"]
    trajectory = output["trajectory"]
    coefficients = belief.objects.position.new_tensor((0.5, -0.75, 1.25))

    def probe(value: Tensor) -> Tensor:
        return (value * coefficients).mean()

    losses: list[tuple[str, Tensor]] = [
        ("current_position", probe(belief.objects.position)),
        ("current_velocity", probe(belief.objects.velocity)),
    ]
    for index, horizon in enumerate(HORIZONS_SECONDS):
        losses.extend(
            (
                (f"horizon_{horizon:.2f}_position", probe(trajectory.positions[:, index])),
                (f"horizon_{horizon:.2f}_velocity", probe(trajectory.velocities[:, index])),
            )
        )

    inputs = (differentiable_batch["rgb"], differentiable_batch["depth"])
    metrics: dict[str, float] = {}
    for index, (name, loss) in enumerate(losses):
        if loss.requires_grad:
            gradients = torch.autograd.grad(
                loss,
                inputs,
                retain_graph=index + 1 < len(losses),
                allow_unused=True,
            )
        else:
            gradients = (None, None)
        for modality, source, gradient in zip(
            ("rgb", "depth"),
            inputs,
            gradients,
            strict=True,
        ):
            resolved = torch.zeros_like(source) if gradient is None else gradient
            if not bool(torch.isfinite(resolved).all()):
                raise FloatingPointError(f"{name} has a nonfinite {modality} VJP")
            metrics[f"gradient_l1/{name}/{modality}"] = float(resolved.abs().sum())
            if name in HISTORY_GRADIENT_TARGETS:
                metrics.update(
                    _history_gradient_diagnostics(
                        resolved,
                        output_name=name,
                        modality=modality,
                    )
                )
    return metrics


def _fail_closed_ablation_metrics(
    config: OrpheusConfig,
    batch: Mapping[str, Any],
    *,
    ablation: str,
) -> dict[str, float]:
    if ablation not in {"missing_depth", "no_foreground"}:
        raise ValueError(f"unknown RGB-D fail-closed ablation {ablation!r}")
    ablated_batch = dict(batch)
    ablated_rgb = batch["rgb"][:1].clone()
    ablated_depth = batch["depth"][:1].clone()
    if ablation == "missing_depth":
        ablated_depth[:, ANCHOR_FRAME_INDEX].zero_()
    else:
        ablated_rgb[:, ANCHOR_FRAME_INDEX].zero_()
    ablated_batch["rgb"] = ablated_rgb
    ablated_batch["depth"] = ablated_depth
    ablated_batch["camera"] = {
        name: value[:1].clone() if isinstance(value, Tensor) else value
        for name, value in batch["camera"].items()
    }
    ablated_batch["timestamps"] = batch["timestamps"][:1].clone()
    output = _run_public_batch(ablated_batch, config)
    belief = output["belief"]
    trajectory = output["trajectory"]
    history = output["history"]
    fit, fit_valid = history.fit(
        gravity=belief.gravity,
        drag=belief.objects.drag,
        minimum_support=config.model.rgbd.temporal_min_samples,
        minimum_dt=config.model.rgbd.temporal_min_dt,
        conditioning_limit=config.model.rgbd.fit_conditioning_limit,
    )
    finite_tensors = (
        belief.objects.position,
        belief.objects.velocity,
        trajectory.positions,
        trajectory.velocities,
        history.positions,
        fit.position,
        fit.velocity,
    )
    finite_fraction = sum(bool(torch.isfinite(value).all()) for value in finite_tensors) / len(
        finite_tensors
    )
    sample_counts = history.sample_mask.sum(dim=-1)
    valid_counts = history.valid_mask.sum(dim=-1)
    audit = output["correction_audit"]
    return {
        f"{ablation}_last_measurement_valid_fraction": float(
            output["last_measurement_valid"].float().mean()
        ),
        f"{ablation}_fit_valid_fraction": float(fit_valid.float().mean()),
        f"{ablation}_direct_velocity_calls": float(audit["calls"]),
        f"{ablation}_history_sample_count_min": float(sample_counts.min()),
        f"{ablation}_history_sample_count_max": float(sample_counts.max()),
        f"{ablation}_history_valid_count_min": float(valid_counts.min()),
        f"{ablation}_history_valid_count_max": float(valid_counts.max()),
        f"{ablation}_finite_fraction": float(finite_fraction),
    }


def _missing_depth_metrics(config: OrpheusConfig, batch: Mapping[str, Any]) -> dict[str, float]:
    return _fail_closed_ablation_metrics(
        config,
        batch,
        ablation="missing_depth",
    )


def _no_foreground_metrics(config: OrpheusConfig, batch: Mapping[str, Any]) -> dict[str, float]:
    return _fail_closed_ablation_metrics(
        config,
        batch,
        ablation="no_foreground",
    )


@torch.no_grad()
def _latency_metrics(config: OrpheusConfig, batch: Mapping[str, Any]) -> dict[str, float]:
    latency_batch = dict(batch)
    latency_batch["rgb"] = batch["rgb"][:1]
    latency_batch["depth"] = batch["depth"][:1]
    latency_batch["camera"] = {
        name: value[:1] if isinstance(value, Tensor) else value
        for name, value in batch["camera"].items()
    }
    latency_batch["timestamps"] = batch["timestamps"][:1]
    model = new_public_model(config)
    model.eval()

    def ingest_history() -> None:
        model.reset(batch_size=1)
        for frame_index in HISTORY_FRAME_INDICES:
            model.ingest(make_rgbd_packet(latency_batch, frame_index))

    rss_before = _process_max_rss_bytes()
    ingest_history()
    model.predict(HORIZONS_SECONDS)
    rss_after_warmup = _process_max_rss_bytes()

    perception: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        ingest_history()
        perception.append(time.perf_counter() - started)
    rss_after_perception = _process_max_rss_bytes()

    rollout: list[float] = []
    for _ in range(20):
        started = time.perf_counter()
        model.predict(HORIZONS_SECONDS)
        rollout.append(time.perf_counter() - started)
    rss_after = _process_max_rss_bytes()
    return {
        "perception_latency_seconds": float(median(perception)),
        "state_only_rollout_latency_seconds": float(median(rollout)),
        "process_max_rss_bytes": float(rss_after),
        "process_rss_delta_bytes": float(max(0, rss_after - rss_before)),
        "warmup_rss_delta_bytes": float(max(0, rss_after_warmup - rss_before)),
        "perception_rss_delta_bytes": float(max(0, rss_after_perception - rss_after_warmup)),
    }


def _validate_manifest(split: str, seeds: Sequence[int]) -> tuple[int, ...]:
    manifests = {
        "development": DEVELOPMENT_SEEDS,
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }
    requested = tuple(int(seed) for seed in seeds)
    if split not in manifests or requested != manifests[split]:
        raise ValueError(f"{split!r} must use its exact frozen RGB-D online bridge manifest")
    if len(requested) != len(set(requested)):
        raise ValueError("RGB-D online bridge manifests must contain unique seeds")
    return requested


def evaluate_seed_manifest(
    config: OrpheusConfig,
    seeds: Sequence[int],
    *,
    split: str,
) -> dict[str, Any]:
    """Materialize one already-authorized exact manifest and evaluate it once.

    Manifest validation happens before the simulator generator is called and
    before any episode is materialized.  This boundary contains no optimizer
    construction or update.
    """

    assert_rgbd_online_bridge_config(config)
    requested = _validate_manifest(split, seeds)
    # The call remains below exact manifest validation: protocol/config/tests
    # cannot accidentally materialize a seed.
    from world_model.simulator import generate_episode

    accumulated: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "current_position_error",
            "current_velocity_error",
            "future_position_error",
            "future_velocity_error",
            "stationary_position_error",
            "zero_velocity_error",
            "public_standalone_current_position_error",
            "public_standalone_current_velocity_error",
            "public_standalone_future_position_error",
            "public_standalone_future_velocity_error",
            "history_measurement_difference",
            "semigroup_position",
            "semigroup_velocity",
            "public_direct_position",
            "public_direct_velocity",
            "query_time_error",
            "history_sample_count",
            "history_valid_count",
            "history_span_error",
            "direct_metric_position_owner_error",
        )
    }
    first_batch: Mapping[str, Any] | None = None
    identities_changed = 0
    persistent_ids: list[Tensor] = []
    active_count = 0
    active_total = 0
    rollout_active_count = 0
    rollout_active_total = 0
    direct_calls: list[int] = []
    direct_valid = 0
    direct_total = 0
    direct_position_fields = 0
    position_owner_counts: list[int] = []
    direct_position_change_max = 0.0
    packet_counts: list[int] = []
    predict_counts: list[int] = []
    alias_count = 0
    runtime_state_bytes_max = 0
    fixed_prior_max_abs = 0.0

    for seed_chunk in _chunks(requested, config.training.batch_size):
        batch = collate_episodes([generate_episode(config, seed) for seed in seed_chunk])
        _assert_free_motion_batch(batch, seed_chunk)
        if first_batch is None:
            first_batch = batch
        with torch.no_grad():
            output = _run_public_batch(batch, config)
        belief = output["belief"]
        trajectory = output["trajectory"]
        standalone = output["standalone"]
        history = output["history"]
        targets = _target_tensors(batch)

        accumulated["current_position_error"].append(
            (belief.objects.position - targets["anchor_position"]).cpu()
        )
        accumulated["current_velocity_error"].append(
            (belief.objects.velocity - targets["anchor_velocity"]).cpu()
        )
        accumulated["future_position_error"].append(
            (trajectory.positions - targets["future_position"]).cpu()
        )
        accumulated["future_velocity_error"].append(
            (trajectory.velocities - targets["future_velocity"]).cpu()
        )
        accumulated["stationary_position_error"].append(
            (
                belief.objects.position[:, None].expand_as(targets["future_position"])
                - targets["future_position"]
            ).cpu()
        )
        accumulated["zero_velocity_error"].append((-targets["anchor_velocity"]).cpu())
        accumulated["public_standalone_current_position_error"].append(
            (belief.objects.position - standalone.fit.position).cpu()
        )
        accumulated["public_standalone_current_velocity_error"].append(
            (belief.objects.velocity - standalone.fit.velocity).cpu()
        )
        accumulated["public_standalone_future_position_error"].append(
            (trajectory.positions - standalone.rollout_positions).cpu()
        )
        accumulated["public_standalone_future_velocity_error"].append(
            (trajectory.velocities - standalone.rollout_velocities).cpu()
        )
        raw_history = history.positions.permute(0, 2, 1, 3)
        accumulated["history_measurement_difference"].append(
            (raw_history - standalone.measured_positions).cpu()
        )
        for key in (
            "semigroup_position",
            "semigroup_velocity",
            "public_direct_position",
            "public_direct_velocity",
            "query_time_error",
            "direct_metric_position_owner_error",
        ):
            accumulated[key].append(output[key].cpu())
        sample_count = history.sample_mask.sum(dim=-1)
        valid_count = history.valid_mask.sum(dim=-1)
        history_span = history.timestamps[..., -1] - history.timestamps[..., 0]
        accumulated["history_sample_count"].append(sample_count.cpu())
        accumulated["history_valid_count"].append(valid_count.cpu())
        accumulated["history_span_error"].append(
            (history_span - DEFAULT_GATES.history_span_seconds).abs().cpu()
        )

        identities = output["identities"]
        identities_changed += int((identities[:, 1:] != identities[:, :-1]).sum())
        identities_changed += int((history.object_ids != identities[:, -1]).sum().detach().cpu())
        persistent_ids.append(identities.cpu())
        active = output["active_masks"]
        active_count += int(active.sum().detach().cpu())
        active_total += active.numel()
        rollout_active_count += int(trajectory.active_mask.sum().detach().cpu())
        rollout_active_total += trajectory.active_mask.numel()
        audit = output["correction_audit"]
        direct_calls.append(int(audit["calls"]))
        direct_valid += int(audit["valid"])
        direct_total += int(audit["total"])
        direct_position_fields += int(audit["position_fields"])
        position_owner_counts.append(int(output["position_owner_count"]))
        direct_position_change_max = max(
            direct_position_change_max,
            float(audit["position_change_max_abs"]),
        )
        packet_counts.append(int(output["packet_count"]))
        predict_counts.append(int(output["predict_count"]))
        alias_count += int(output["output_alias_count"])
        runtime_state_bytes_max = max(runtime_state_bytes_max, int(output["runtime_tensor_bytes"]))
        fixed_prior_max_abs = max(
            fixed_prior_max_abs,
            float(output["fixed_prior_max_abs"]),
        )

    if first_batch is None:
        raise RuntimeError("RGB-D online bridge manifest unexpectedly produced no batches")
    tensors = {name: torch.cat(values) for name, values in accumulated.items()}
    future_position_rmse = [
        _rmse(tensors["future_position_error"][:, index]) for index in range(len(HORIZONS_SECONDS))
    ]
    future_velocity_rmse = [
        _rmse(tensors["future_velocity_error"][:, index]) for index in range(len(HORIZONS_SECONDS))
    ]
    future_position_axis = [
        float(_axis_rmse(tensors["future_position_error"][:, index]).max())
        for index in range(len(HORIZONS_SECONDS))
    ]
    future_velocity_axis = [
        float(_axis_rmse(tensors["future_velocity_error"][:, index]).max())
        for index in range(len(HORIZONS_SECONDS))
    ]
    stationary_rmse = [
        _rmse(tensors["stationary_position_error"][:, index])
        for index in range(len(HORIZONS_SECONDS))
    ]
    public_standalone_future_position = [
        _rmse(tensors["public_standalone_future_position_error"][:, index])
        for index in range(len(HORIZONS_SECONDS))
    ]
    public_standalone_future_velocity = [
        _rmse(tensors["public_standalone_future_velocity_error"][:, index])
        for index in range(len(HORIZONS_SECONDS))
    ]
    current_position_rmse = _rmse(tensors["current_position_error"])
    current_velocity_rmse = _rmse(tensors["current_velocity_error"])
    zero_velocity_rmse = _rmse(tensors["zero_velocity_error"])
    epsilon = torch.finfo(torch.float64).eps

    model = new_public_model(config)
    estimator = new_standalone_estimator(config)
    learned_parameters = tuple(
        parameter
        for owner in (model, estimator)
        for parameter in owner.parameters()
        if parameter.requires_grad
    )
    all_buffers = tuple(buffer for owner in (model, estimator) for buffer in owner.buffers())
    state_tensors = tuple(
        tensor for owner in (model, estimator) for tensor in owner.state_dict().values()
    )
    metrics: dict[str, Any] = {
        "current_position_rmse_m": current_position_rmse,
        "current_position_axis_rmse_m": float(_axis_rmse(tensors["current_position_error"]).max()),
        "current_velocity_rmse_mps": current_velocity_rmse,
        "current_velocity_axis_rmse_mps": float(
            _axis_rmse(tensors["current_velocity_error"]).max()
        ),
        "horizon_velocity_rmse_mps": max(future_velocity_rmse),
        "horizon_velocity_axis_rmse_mps": max(future_velocity_axis),
        "maximum_position_error_growth_slope_mps": max(
            max(0.0, error - current_position_rmse) / horizon
            for horizon, error in zip(HORIZONS_SECONDS, future_position_rmse, strict=True)
        ),
        "early_stationary_additive_regression_m": max(
            future_position_rmse[index] - stationary_rmse[index] for index in (0, 1)
        ),
        "long_stationary_rmse_ratio": max(
            future_position_rmse[index] / max(stationary_rmse[index], epsilon)
            for index in (2, 3, 4)
        ),
        "zero_velocity_rmse_ratio": current_velocity_rmse / max(zero_velocity_rmse, epsilon),
        "public_standalone_current_position_rmse_m": _rmse(
            tensors["public_standalone_current_position_error"]
        ),
        "public_standalone_current_velocity_rmse_mps": _rmse(
            tensors["public_standalone_current_velocity_error"]
        ),
        "history_standalone_measurement_max_abs_m": float(
            tensors["history_measurement_difference"].abs().max()
        ),
        "fixed_prior_max_abs": fixed_prior_max_abs,
        "identity_change_count": float(identities_changed),
        "persistent_object_id_min": float(torch.cat(persistent_ids).min()),
        "persistent_object_id_max": float(torch.cat(persistent_ids).max()),
        "active_fraction": active_count / active_total,
        "rollout_active_fraction": rollout_active_count / rollout_active_total,
        "history_sample_count_min": float(tensors["history_sample_count"].min()),
        "history_sample_count_max": float(tensors["history_sample_count"].max()),
        "history_valid_count_min": float(tensors["history_valid_count"].min()),
        "history_valid_count_max": float(tensors["history_valid_count"].max()),
        "history_span_max_abs_error_seconds": float(tensors["history_span_error"].max()),
        "direct_velocity_calls_per_batch_min": float(min(direct_calls)),
        "direct_velocity_calls_per_batch_max": float(max(direct_calls)),
        "direct_velocity_valid_fraction": direct_valid / direct_total if direct_total else 0.0,
        "position_owner_count_min": float(min(position_owner_counts)),
        "position_owner_count_max": float(max(position_owner_counts)),
        "direct_metric_position_owner_max_abs_m": float(
            tensors["direct_metric_position_owner_error"].max()
        ),
        "direct_position_field_count": float(direct_position_fields),
        "direct_velocity_position_change_max_abs_m": direct_position_change_max,
        "public_rollout_output_alias_count": float(alias_count),
        "public_query_time_max_abs_seconds": float(tensors["query_time_error"].max()),
        "ingested_frame_count_min": float(min(packet_counts)),
        "ingested_frame_count_max": float(max(packet_counts)),
        "public_predict_calls_per_batch_min": float(min(predict_counts)),
        "public_predict_calls_per_batch_max": float(max(predict_counts)),
        "semigroup_position_max_abs_m": float(tensors["semigroup_position"].max()),
        "semigroup_velocity_max_abs_mps": float(tensors["semigroup_velocity"].max()),
        "public_direct_position_max_abs_m": float(tensors["public_direct_position"].max()),
        "public_direct_velocity_max_abs_mps": float(tensors["public_direct_velocity"].max()),
        "learned_parameter_count": float(sum(value.numel() for value in learned_parameters)),
        "learned_parameter_bytes": float(
            sum(value.numel() * value.element_size() for value in learned_parameters)
        ),
        "module_tensor_buffer_count": float(len(all_buffers)),
        "persistent_module_state_key_count": float(len(state_tensors)),
        "persistent_module_state_bytes": float(
            sum(value.numel() * value.element_size() for value in state_tensors)
        ),
        "persistent_runtime_tensor_state_bytes_max": float(runtime_state_bytes_max),
        "optimizer_updates": 0.0,
        "optimizer_state_entry_count": 0.0,
    }
    for index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = future_position_rmse[index]
        metrics[f"horizon_{label}_position_axis_rmse_m"] = future_position_axis[index]
        metrics[f"horizon_{label}_velocity_rmse_mps"] = future_velocity_rmse[index]
        metrics[f"horizon_{label}_velocity_axis_rmse_mps"] = future_velocity_axis[index]
        metrics[f"horizon_{label}_position_error_growth_m"] = (
            future_position_rmse[index] - current_position_rmse
        )
        metrics[f"public_standalone_horizon_{label}_position_rmse_m"] = (
            public_standalone_future_position[index]
        )
        metrics[f"public_standalone_horizon_{label}_velocity_rmse_mps"] = (
            public_standalone_future_velocity[index]
        )
    metrics.update(_missing_depth_metrics(config, first_batch))
    metrics.update(_no_foreground_metrics(config, first_batch))
    metrics.update(_gradient_metrics(config, first_batch))
    metrics.update(_latency_metrics(config, first_batch))
    for name, value in metrics.items():
        if _numeric(value) is None:
            raise FloatingPointError(f"RGB-D online bridge metric {name!r} is nonfinite")
    failures = gate_failures(metrics)
    return {
        "split": split,
        "seeds": list(requested),
        "seed_manifest_sha256": canonical_sha256(list(requested)),
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
        "optimizer_updates": 0,
        "runtime_api": {
            "packet_factory": "make_rgbd_packet",
            "ingest_frames": list(HISTORY_FRAME_INDICES),
            "rollout_method": "OnlineWorldModel.predict",
            "horizons_seconds": list(HORIZONS_SECONDS),
        },
        "standalone_role": "same_input_same_fixed_prior_agreement_oracle_only",
    }


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EMPTY_MODEL_STATE_SHA256 = canonical_sha256([])


def validated_sha256(value: str | None, *, label: str) -> str:
    """Return one normalized explicit SHA-256 or fail closed."""

    if value is None or len(value) != 64:
        raise ValueError(f"qualification requires a 64-character {label}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"qualification {label} must be hexadecimal") from error
    return value.lower()


def clean_source(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Require exact clean committed source provenance."""

    required = {
        "commit",
        "dirty",
        "worktree_fingerprint",
        "runtime_source_fingerprint",
    }
    if set(value) != required:
        raise ValueError(f"{label} source provenance must contain exactly {sorted(required)}")
    normalized = dict(value)
    if normalized["dirty"] is not False:
        raise ValueError(f"{label} requires a clean committed worktree")
    commit = normalized["commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{label} requires an exact 40-character Git commit")
    try:
        int(commit, 16)
    except ValueError as error:
        raise ValueError(f"{label} Git commit must be hexadecimal") from error
    validated_sha256(
        normalized["worktree_fingerprint"],
        label=f"{label} worktree fingerprint",
    )
    validated_sha256(
        normalized["runtime_source_fingerprint"],
        label=f"{label} runtime source fingerprint",
    )
    return normalized


def require_frozen_config(path: str | Path) -> OrpheusConfig:
    """Load only the exact reviewed raw config bytes and semantic profile."""

    source = Path(path)
    contents = stable_read_bytes(source, label="frozen config")
    digest = sha256_bytes(contents)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "RGB-D online bridge requires the exact frozen config bytes: "
            f"expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    config = load_config(source)
    assert_rgbd_online_bridge_config(config)
    return config


def _atomic_temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_regular_nonlink(path: Path, *, label: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} does not exist: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return metadata


def stable_read_bytes(path: str | Path, *, label: str) -> bytes:
    """Read one non-link regular file and reject concurrent replacement."""

    source = Path(path)
    before = _require_regular_nonlink(source, label=label)
    contents = source.read_bytes()
    after = _require_regular_nonlink(source, label=label)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(contents) != after.st_size:
        raise RuntimeError(f"{label} changed while it was being read")
    return contents


def validate_distinct_paths(
    paths: Mapping[str, Path],
    *,
    atomic_writers: Sequence[str],
) -> None:
    """Reject resolved, hard-link, and atomic-temporary artifact aliases."""

    expanded: dict[str, Path] = {name: Path(path) for name, path in paths.items()}
    for name in atomic_writers:
        if name not in expanded:
            raise ValueError(f"unknown atomic artifact {name!r}")
        expanded[f"{name}_atomic_temporary"] = _atomic_temporary(expanded[name])
    resolved: dict[Path, list[str]] = {}
    for name, path in expanded.items():
        resolved.setdefault(path.resolve(), []).append(name)
    collisions = [names for names in resolved.values() if len(names) > 1]
    if collisions:
        details = "; ".join(", ".join(names) for names in collisions)
        raise ValueError("RGB-D online bridge artifact paths alias: " + details)

    existing = [(name, path) for name, path in expanded.items() if _lexists(path)]
    for index, (left_name, left_path) in enumerate(existing):
        for right_name, right_path in existing[index + 1 :]:
            try:
                alias = os.path.samefile(left_path, right_path)
            except (FileNotFoundError, OSError):
                alias = False
            if alias:
                raise ValueError(
                    f"RGB-D online bridge artifact paths hard-link alias: {left_name}, {right_name}"
                )


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_descriptor(descriptor: int, contents: bytes) -> None:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())


def _durable_create(path: Path, contents: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    _write_descriptor(descriptor, contents)
    _fsync_parent(path)


def _durable_replace(path: Path, contents: bytes, *, mode: int = 0o600) -> None:
    temporary = _atomic_temporary(path)
    if _lexists(temporary):
        raise FileExistsError(f"atomic temporary must be fresh: {temporary}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        _write_descriptor(descriptor, contents)
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        # A leftover temporary is evidence of an ambiguous durable update.  Do
        # not remove it and do not permit a second qualification attempt.
        raise


def write_report_fresh(path: Path, report: Mapping[str, Any]) -> None:
    """Create one finite JSON report without overwriting any prior evidence."""

    contents = (
        json.dumps(
            dict(report),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    _durable_create(path, contents, mode=0o600)


class QualificationLedger:
    """Exactly-once durable selector -> confirmation -> final access receipt."""

    ORDER = ("selector", "confirmation", "final_test")

    def __init__(self, path: str | Path, bindings: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.record: dict[str, Any] = {
            "artifact_kind": "rgbd_online_bridge_exactly_once_access_ledger",
            "order": list(self.ORDER),
            "bindings": dict(bindings),
            "splits": {
                split: {
                    "access_started": False,
                    "status": "unopened",
                    "result_sha256": None,
                }
                for split in self.ORDER
            },
            "attempt_reserved": True,
            "protected_data_materialized": False,
            "status": "reserved_before_protected_access",
        }
        _durable_create(self.path, self._serialized())

    def _serialized(self) -> bytes:
        return (
            json.dumps(
                self.record,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )

    def _replace(self) -> None:
        _durable_replace(self.path, self._serialized())

    def begin_access(self, split: str) -> None:
        if split not in self.ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        index = self.ORDER.index(split)
        for predecessor in self.ORDER[:index]:
            if self.record["splits"][predecessor]["status"] != "passed":
                raise RuntimeError(f"{split} must remain unopened until {predecessor} passes")
        state = self.record["splits"][split]
        if state["status"] != "unopened" or state["access_started"] is not False:
            raise RuntimeError(f"protected split {split!r} cannot be opened twice")
        later_started = any(
            self.record["splits"][later]["access_started"] for later in self.ORDER[index + 1 :]
        )
        if later_started:
            raise RuntimeError("protected access order is inconsistent")
        state["access_started"] = True
        state["status"] = "materialization_started"
        self.record["protected_data_materialized"] = True
        self.record["status"] = f"{split}_materialization_started"
        self._replace()

    def complete_split(self, split: str, result: Mapping[str, Any]) -> None:
        if split not in self.ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        state = self.record["splits"][split]
        if state["status"] != "materialization_started":
            raise RuntimeError(f"protected split {split!r} was not durably opened")
        passed = result.get("passed") is True
        state["status"] = "passed" if passed else "failed"
        state["result_sha256"] = canonical_sha256(result)
        self.record["status"] = f"{split}_{state['status']}"
        self._replace()

    def finish(self, *, passed: bool, stopped_after: str) -> None:
        final_status = "complete" if passed else "failed"
        if passed and any(
            self.record["splits"][split]["status"] != "passed" for split in self.ORDER
        ):
            raise RuntimeError("qualification cannot complete before all protected splits pass")
        self.record["status"] = final_status
        self.record["stopped_after"] = stopped_after
        self._replace()

    def record_error(self, error: BaseException, *, stopped_after: str) -> None:
        self.record["status"] = "error"
        self.record["stopped_after"] = stopped_after
        self.record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._replace()


def _model_state_sha256(model: OnlineWorldModel) -> str:
    state = model.state_dict()
    if state:
        raise RuntimeError("RGB-D online bridge model state must be exactly empty")
    return EMPTY_MODEL_STATE_SHA256


def _validate_development_split(development: Mapping[str, Any]) -> None:
    if development.get("split") != "development":
        raise ValueError("reviewed evidence has the wrong development split name")
    expected_seeds = list(DEVELOPMENT_SEEDS)
    if development.get("seeds") != expected_seeds:
        raise ValueError("reviewed evidence has the wrong development seed manifest")
    if development.get("seed_manifest_sha256") != canonical_sha256(expected_seeds):
        raise ValueError("reviewed evidence has the wrong development manifest hash")
    if development.get("optimizer_updates") != 0:
        raise ValueError("reviewed development must prove zero optimizer updates")
    runtime_api = development.get("runtime_api")
    expected_api = {
        "packet_factory": "make_rgbd_packet",
        "ingest_frames": list(HISTORY_FRAME_INDICES),
        "rollout_method": "OnlineWorldModel.predict",
        "horizons_seconds": list(HORIZONS_SECONDS),
    }
    if runtime_api != expected_api:
        raise ValueError("reviewed development did not use the exact public runtime APIs")
    metrics = development.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("reviewed development is missing scalar metrics")
    failures = gate_failures(metrics)
    if development.get("failures") != failures or failures:
        raise ValueError("reviewed development gates do not recompute as passed")
    if development.get("passed") is not True:
        raise ValueError("reviewed development did not pass")


def validate_development_evidence(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind reviewed passing development evidence to current frozen source."""

    if report.get("artifact_kind") != "rgbd_online_bridge_development":
        raise ValueError("reviewed development report has the wrong artifact kind")
    if report.get("passed") is not True or report.get("review_ready") is not True:
        raise ValueError("reviewed development evidence did not pass")
    if report.get("protected_data_materialized") is not False:
        raise ValueError("development evidence must leave all protected data unopened")
    if report.get("optimizer_updates") != 0:
        raise ValueError("development evidence must prove zero optimizer updates")
    if report.get("stopped_after") != "development":
        raise ValueError("reviewed evidence must stop after development")
    if canonical_sha256(report.get("protocol")) != canonical_sha256(bridge_protocol()):
        raise ValueError("reviewed development protocol differs from frozen source")
    if report.get("config_sha256") != FROZEN_CONFIG_SHA256:
        raise ValueError("reviewed development config hash differs from frozen bytes")
    if report.get("source_provenance") != source:
        raise ValueError("reviewed development source differs from current clean source")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reviewed development report does not bind the checkpoint")
    if report.get("checkpoint_model_state_sha256") != EMPTY_MODEL_STATE_SHA256:
        raise ValueError("reviewed checkpoint did not bind an empty model state")
    development = report.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("reviewed report is missing development split evidence")
    _validate_development_split(development)
    return development


def _load_checkpoint_payload(contents: bytes) -> Mapping[str, Any]:
    payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False)
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
    """Require a step-zero, optimizer-free, exactly empty model checkpoint."""

    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or model_state:
        raise ValueError("reviewed bridge checkpoint model state must be exactly empty")
    if payload.get("step") != 0:
        raise ValueError("reviewed bridge checkpoint must be step zero")
    if payload.get("optimizer_state") is not None or payload.get("scheduler_state") is not None:
        raise ValueError("reviewed bridge checkpoint must contain no optimizer/scheduler state")
    if payload.get("config") != config.to_dict():
        raise ValueError("reviewed bridge checkpoint config differs from frozen config")
    if payload.get("git") != source:
        raise ValueError("reviewed bridge checkpoint source differs from current source")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("reviewed bridge checkpoint is missing evidence metrics")
    if metrics.get("artifact_kind") != "rgbd_online_bridge_empty_model_state":
        raise ValueError("reviewed bridge checkpoint has the wrong artifact kind")
    if metrics.get("optimizer_updates") != 0:
        raise ValueError("reviewed bridge checkpoint must prove zero optimizer updates")
    if metrics.get("model_state_sha256") != EMPTY_MODEL_STATE_SHA256:
        raise ValueError("reviewed bridge checkpoint has the wrong empty-state hash")
    if metrics.get("protocol") != bridge_protocol():
        raise ValueError("reviewed bridge checkpoint protocol differs from frozen source")
    if metrics.get("development") != development:
        raise ValueError("reviewed bridge checkpoint does not bind development evidence")


def _guard_frozen_inputs(
    *,
    source: Mapping[str, Any],
    config_path: Path,
    development_report_path: Path | None = None,
    development_report_sha256: str | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    model: OnlineWorldModel | None = None,
) -> None:
    current_source = clean_source(
        capture_git_metadata(REPOSITORY_ROOT),
        label="RGB-D online bridge execution guard",
    )
    if current_source != source:
        raise RuntimeError("source provenance changed during bridge execution")
    if (
        sha256_bytes(stable_read_bytes(config_path, label="guarded frozen config"))
        != FROZEN_CONFIG_SHA256
    ):
        raise RuntimeError("frozen config bytes changed during bridge execution")
    if bridge_protocol()["protocol_sha256"] != canonical_sha256(
        {key: value for key, value in bridge_protocol().items() if key != "protocol_sha256"}
    ):
        raise RuntimeError("bridge protocol self-hash is inconsistent")
    if (
        development_report_path is not None
        and sha256_bytes(
            stable_read_bytes(
                development_report_path,
                label="guarded reviewed development report",
            )
        )
        != development_report_sha256
    ):
        raise RuntimeError("reviewed development report changed during qualification")
    if (
        checkpoint_path is not None
        and sha256_bytes(stable_read_bytes(checkpoint_path, label="guarded reviewed checkpoint"))
        != checkpoint_sha256
    ):
        raise RuntimeError("reviewed checkpoint changed during qualification")
    if model is not None and _model_state_sha256(model) != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("public bridge model state changed during qualification")


def run_development(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    source_provenance: Mapping[str, Any],
) -> int:
    """Evaluate development only and emit reviewable empty-state evidence."""

    assert_rgbd_online_bridge_config(config)
    source = clean_source(source_provenance, label="RGB-D online bridge development")
    validate_distinct_paths(
        {
            "config": config_path,
            "report": report_path,
            "checkpoint": checkpoint_path,
        },
        atomic_writers=("report", "checkpoint"),
    )
    if _lexists(report_path) or _lexists(checkpoint_path):
        raise FileExistsError("development report and checkpoint paths must both be fresh")
    if _lexists(_atomic_temporary(report_path)) or _lexists(_atomic_temporary(checkpoint_path)):
        raise FileExistsError("development atomic temporary paths must be fresh")
    if (
        sha256_bytes(stable_read_bytes(config_path, label="development frozen config"))
        != FROZEN_CONFIG_SHA256
    ):
        raise ValueError("development config bytes differ from the frozen protocol")

    protocol = bridge_protocol()
    model = new_public_model(config)
    development = evaluate_seed_manifest(config, DEVELOPMENT_SEEDS, split="development")
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_online_bridge_development",
        "protocol": protocol,
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development": development,
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "passed": development["passed"],
        "review_ready": development["passed"],
        "stopped_after": "development",
    }
    _guard_frozen_inputs(source=source, config_path=config_path, model=model)
    if not development["passed"]:
        write_report_fresh(report_path, report)
        return 1

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        step=0,
        metrics={
            "artifact_kind": "rgbd_online_bridge_empty_model_state",
            "optimizer_updates": 0,
            "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
            "protocol": protocol,
            "development": development,
        },
        device="cpu",
        source_provenance=source,
    )
    checkpoint_contents = stable_read_bytes(checkpoint_path, label="development checkpoint")
    payload = _load_checkpoint_payload(checkpoint_contents)
    validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        development=development,
    )
    report["checkpoint"] = str(checkpoint_path.resolve())
    report["checkpoint_sha256"] = sha256_bytes(checkpoint_contents)
    report["checkpoint_model_state_sha256"] = EMPTY_MODEL_STATE_SHA256
    _guard_frozen_inputs(source=source, config_path=config_path, model=model)
    write_report_fresh(report_path, report)
    return 0


def run_qualification(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    ledger_path: Path,
    reviewed_checkpoint_sha256: str | None,
    reviewed_report_sha256: str | None,
    source_provenance: Mapping[str, Any],
) -> int:
    """Consume protected splits once, in order, after reviewed development."""

    assert_rgbd_online_bridge_config(config)
    source = clean_source(source_provenance, label="RGB-D online bridge qualification")
    checkpoint_digest = validated_sha256(
        reviewed_checkpoint_sha256,
        label="reviewed checkpoint SHA-256",
    )
    development_report_digest = validated_sha256(
        reviewed_report_sha256,
        label="reviewed development report SHA-256",
    )
    validate_distinct_paths(
        {
            "config": config_path,
            "report": report_path,
            "checkpoint": checkpoint_path,
            "development_report": development_report_path,
            "qualification_ledger": ledger_path,
        },
        atomic_writers=("report", "qualification_ledger"),
    )
    for name, path in (
        ("qualification report", report_path),
        ("qualification ledger", ledger_path),
        ("qualification report temporary", _atomic_temporary(report_path)),
        ("qualification ledger temporary", _atomic_temporary(ledger_path)),
    ):
        if _lexists(path):
            raise FileExistsError(f"{name} path must be fresh: {path}")

    checkpoint_contents = stable_read_bytes(checkpoint_path, label="reviewed checkpoint")
    if sha256_bytes(checkpoint_contents) != checkpoint_digest:
        raise ValueError("reviewed checkpoint SHA-256 does not match bytes read")
    development_report_contents = stable_read_bytes(
        development_report_path,
        label="reviewed development report",
    )
    if sha256_bytes(development_report_contents) != development_report_digest:
        raise ValueError("reviewed development report SHA-256 does not match bytes read")
    development_report = json.loads(development_report_contents)
    if not isinstance(development_report, Mapping):
        raise ValueError("reviewed development report must be a JSON object")
    development = validate_development_evidence(
        development_report,
        checkpoint_sha256=checkpoint_digest,
        source=source,
    )
    payload = _load_checkpoint_payload(checkpoint_contents)
    validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        development=development,
    )
    model = new_public_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    initial_model_state_sha256 = _model_state_sha256(model)
    if initial_model_state_sha256 != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("reviewed public model state is not empty")

    _guard_frozen_inputs(
        source=source,
        config_path=config_path,
        development_report_path=development_report_path,
        development_report_sha256=development_report_digest,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_digest,
        model=model,
    )
    ledger = QualificationLedger(
        ledger_path,
        {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": source,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "reviewed_checkpoint_sha256": checkpoint_digest,
            "reviewed_development_report_sha256": development_report_digest,
            "model_state_sha256": initial_model_state_sha256,
        },
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_online_bridge_protected_qualification",
        "protocol": bridge_protocol(),
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": checkpoint_digest,
        "reviewed_development_report_sha256": development_report_digest,
        "initial_model_state_sha256": initial_model_state_sha256,
        "optimizer_updates": 0,
        "passed": False,
        "protected_data_materialized": False,
        "stopped_after": "reviewed_development",
    }
    try:
        for split, seeds in (
            ("selector", SELECTOR_SEEDS),
            ("confirmation", CONFIRMATION_SEEDS),
            ("final_test", FINAL_TEST_SEEDS),
        ):
            _guard_frozen_inputs(
                source=source,
                config_path=config_path,
                development_report_path=development_report_path,
                development_report_sha256=development_report_digest,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_digest,
                model=model,
            )
            ledger.begin_access(split)
            report["protected_data_materialized"] = True
            result = evaluate_seed_manifest(config, seeds, split=split)
            report[split] = result
            report["stopped_after"] = split
            ledger.complete_split(split, result)
            _guard_frozen_inputs(
                source=source,
                config_path=config_path,
                development_report_path=development_report_path,
                development_report_sha256=development_report_digest,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_digest,
                model=model,
            )
            if not result["passed"]:
                report["failures"] = list(result["failures"])
                break
        else:
            report["passed"] = True
            report["failures"] = []
        report["final_model_state_sha256"] = _model_state_sha256(model)
        if report["final_model_state_sha256"] != initial_model_state_sha256:
            raise RuntimeError("public model state changed during protected qualification")
        ledger.finish(
            passed=bool(report["passed"]),
            stopped_after=str(report["stopped_after"]),
        )
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            ledger.record_error(error, stopped_after=str(report["stopped_after"]))
        finally:
            write_report_fresh(report_path, report)
        raise

    ledger_contents = stable_read_bytes(ledger_path, label="completed qualification ledger")
    report["qualification_ledger"] = str(ledger_path.resolve())
    report["qualification_ledger_sha256"] = sha256_bytes(ledger_contents)
    _guard_frozen_inputs(
        source=source,
        config_path=config_path,
        development_report_path=development_report_path,
        development_report_sha256=development_report_digest,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_digest,
        model=model,
    )
    write_report_fresh(report_path, report)
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
    "HISTORY_FRAME_INDICES",
    "HISTORY_GRADIENT_TARGETS",
    "HORIZONS_SECONDS",
    "OPTIMIZER_UPDATES",
    "QualificationLedger",
    "RGBDOnlineBridgeGates",
    "SELECTOR_SEEDS",
    "TARGET_FRAME_INDICES",
    "assert_rgbd_online_bridge_config",
    "bridge_protocol",
    "canonical_sha256",
    "clean_source",
    "evaluate_seed_manifest",
    "gate_failures",
    "new_public_model",
    "new_standalone_estimator",
    "require_frozen_config",
    "run_development",
    "run_qualification",
    "sha256_file",
    "stable_read_bytes",
    "validate_checkpoint_evidence",
    "validate_development_evidence",
    "validate_distinct_paths",
    "validated_sha256",
    "write_report_fresh",
]
