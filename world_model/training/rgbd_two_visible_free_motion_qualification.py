"""Frozen qualification for the two-visible-sphere public RGB-D rung.

Protocol inspection, configuration validation, and unit tests are deliberately
seed-free.  Simulator state is materialized only by :func:`evaluate_seed_manifest`
after an exact predeclared manifest has passed validation.  Protected manifests
additionally require an exactly-once durable access receipt written by the thin
runner before this boundary is entered.

The admitted runtime is parameter-free ``OnlineWorldModel``.  It consumes
frames 0..15 through ``make_rgbd_packet``/``ingest`` and exposes every future
through ``predict``.  The scene family is constructed here, then preflighted
frame-by-frame for visibility, image separation, boundary clearance, and the
absence of physical events; ``ensure_collision=false`` is never accepted as
evidence of those properties.
"""

from __future__ import annotations

import io
import json
import math
import os
import resource
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.datasets import collate_episodes
from world_model.observations import DirectVelocityEvidence
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.runtime.state import runtime_stream_key
from world_model.simulator.episode import validate_episode
from world_model.simulator.labels import make_perception_labels, validate_perception_labels
from world_model.simulator.physics import PhysicsStepEvents, SphereState, empty_physics_events
from world_model.simulator.sphere_world import SphereWorld, SphereWorldConfig
from world_model.training.checkpointing import capture_git_metadata, checkpoint_payload
from world_model.training.loop import make_rgbd_packet
from world_model.training.rgbd_online_bridge_qualification import (
    canonical_sha256,
    clean_source,
    sha256_bytes,
    stable_read_bytes,
    validate_distinct_paths,
    validated_sha256,
    write_report_fresh,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.version import SIMULATOR_VERSION

DEVELOPMENT_SEEDS = tuple(range(49_000_000, 49_000_032))
SELECTOR_SEEDS = tuple(range(50_000_000, 50_000_024))
CONFIRMATION_SEEDS = tuple(range(51_000_000, 51_000_024))
FINAL_TEST_SEEDS = tuple(range(52_000_000, 52_000_048))

HISTORY_FRAME_INDICES = tuple(range(16))
ANCHOR_FRAME_INDEX = 15
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TARGET_FRAME_INDICES = (17, 20, 25, 35, 55)
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

ARCHITECTURE_VERSION = 1
ARCHITECTURE_ATTEMPT = 2
OPTIMIZER_UPDATES = 0
FROZEN_CONFIG_SHA256 = "84e6f44b818bb9323a774bdba9492ef056e2a2747b93517fa38497ba83218bba"
EMPTY_MODEL_STATE_SHA256 = canonical_sha256([])


@dataclass(frozen=True)
class TwoVisibleRGBDGates:
    """Predeclared gates independently recomputed for every split."""

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

    identity_switch_count: float = 0.0
    persistent_id_mismatch_count: float = 0.0
    identity_coverage: float = 1.0
    persistent_object_id_min: float = 0.0
    persistent_object_id_max: float = 1.0
    association_pair_coverage: float = 1.0
    association_ambiguous_pair_count: float = 0.0
    minimum_hungarian_margin: float = 0.02
    minimum_position_assignment_margin_m: float = 0.25
    minimum_matched_appearance_cosine: float = 0.90
    minimum_cross_appearance_cosine_distance: float = 0.10
    physical_palette_swap_fraction: float = 0.50
    birth_slot_physical_zero_fraction_min: float = 0.25
    birth_slot_physical_zero_fraction_max: float = 0.75
    unique_scene_specification_fraction: float = 1.0
    gradient_audit_scene_count: float = 4.0
    gradient_audit_unique_scene_fraction: float = 1.0

    active_fraction: float = 1.0
    rollout_active_fraction: float = 1.0
    history_sample_count: float = 16.0
    history_valid_count: float = 16.0
    history_span_seconds: float = 0.75
    history_span_tolerance_seconds: float = 1.0e-6
    direct_velocity_calls_per_batch: float = 1.0
    direct_velocity_valid_fraction: float = 1.0
    position_owner_count: float = 1.0
    direct_position_field_count: float = 0.0
    direct_velocity_position_change_max_abs_m: float = 0.0
    direct_metric_position_owner_max_abs_m: float = 1.0e-7
    ambiguity_direct_position_write_count: float = 0.0
    ambiguity_direct_velocity_write_count: float = 0.0

    minimum_silhouette_gap_pixels: float = 2.0
    minimum_boundary_clearance_pixels: float = 2.0
    minimum_world_surface_gap_m: float = 0.50
    minimum_world_boundary_clearance_m: float = 0.10
    minimum_visible_fraction: float = 1.0
    preflight_event_count: float = 0.0

    semigroup_position_max_abs_m: float = 1.0e-5
    semigroup_velocity_max_abs_mps: float = 1.0e-5
    public_direct_position_max_abs_m: float = 1.0e-6
    public_direct_velocity_max_abs_mps: float = 1.0e-6
    analytic_position_agreement_max_abs_m: float = 2.0e-5
    analytic_velocity_agreement_max_abs_mps: float = 2.0e-5
    public_rollout_output_alias_count: float = 0.0
    public_query_time_max_abs_seconds: float = 1.0e-6

    minimum_input_gradient_l1: float = 1.0e-8
    maximum_input_gradient_l1: float = 1.0e8
    minimum_history_frame_gradient_l1: float = 1.0e-8
    required_history_gradient_frames: float = 16.0
    current_position_required_history_gradient_frames: float = 1.0
    current_position_nonanchor_gradient_max_l1: float = 0.0
    maximum_cross_scene_gradient_l1: float = 0.0

    perception_latency_seconds: float = 3.0
    state_only_rollout_latency_seconds: float = 0.075
    persistent_runtime_tensor_state_bytes: int = 65_536
    process_max_rss_bytes: int = 2_500_000_000
    process_rss_delta_bytes: int = 1_000_000_000


DEFAULT_GATES = TwoVisibleRGBDGates()


@dataclass(frozen=True)
class TwoVisibleSceneSpecification:
    """Exact initial conditions derived without sampling generic placement."""

    position: Tensor
    velocity: Tensor
    albedo: Tensor
    palette_swapped: bool


_MANIFEST_ACCESS_AUTHORITY = object()


class _ManifestAccessAuthorization:
    """Single-use manifest capability backed by an on-disk durable receipt."""

    def __init__(
        self,
        authority: object,
        *,
        split: str,
        seeds: Sequence[int],
        ledger_path: Path,
        ledger_kind: str,
    ) -> None:
        if authority is not _MANIFEST_ACCESS_AUTHORITY:
            raise PermissionError("manifest authorization may only be issued by a durable ledger")
        self._split = split
        self._seeds = tuple(int(seed) for seed in seeds)
        self._ledger_path = ledger_path
        self._ledger_kind = ledger_kind
        self._begun = False
        self._finished = False
        self._cursor = 0

    def _validate_receipt(self) -> None:
        contents = stable_read_bytes(self._ledger_path, label=f"{self._split} access ledger")
        record = json.loads(contents)
        if not isinstance(record, Mapping) or record.get("artifact_kind") != self._ledger_kind:
            raise RuntimeError("manifest authorization ledger has the wrong artifact kind")
        if self._ledger_kind == "rgbd_two_visible_development_access_ledger":
            if (
                record.get("status") != "development_materialization_started"
                or record.get("access_started") is not True
            ):
                raise RuntimeError("development receipt is not durably materialization-started")
        else:
            splits = record.get("splits")
            state = splits.get(self._split) if isinstance(splits, Mapping) else None
            if (
                not isinstance(state, Mapping)
                or state.get("access_started") is not True
                or state.get("status") != "materialization_started"
                or record.get("status") != f"{self._split}_materialization_started"
            ):
                raise RuntimeError("protected receipt is not durably materialization-started")

    def begin_manifest(self, split: str, seeds: Sequence[int]) -> None:
        if self._begun or self._finished:
            raise RuntimeError("manifest authorization is single use")
        if split != self._split or tuple(int(seed) for seed in seeds) != self._seeds:
            raise PermissionError("manifest authorization does not match requested split/seeds")
        self._validate_receipt()
        self._begun = True

    def authorize_seed(self, seed: int) -> None:
        if not self._begun or self._finished or self._cursor >= len(self._seeds):
            raise PermissionError("scene construction lacks an active manifest authorization")
        if seed != self._seeds[self._cursor]:
            raise PermissionError("scene construction order differs from authorized manifest")
        self._validate_receipt()
        self._cursor += 1

    def finish_manifest(self) -> None:
        if not self._begun or self._finished or self._cursor != len(self._seeds):
            raise RuntimeError("authorized manifest did not materialize exactly once in order")
        self._validate_receipt()
        self._finished = True

    def require_finished(self) -> None:
        if not self._finished or self._cursor != len(self._seeds):
            raise RuntimeError("manifest authorization was not fully consumed")


def _assert_seed_namespaces() -> None:
    namespaces = (
        DEVELOPMENT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    flattened = [seed for namespace in namespaces for seed in namespace]
    if any(not namespace for namespace in namespaces):
        raise RuntimeError("every two-visible RGB-D namespace must be nonempty")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("two-visible RGB-D namespaces must be disjoint")


_SEED_FRACTION_BITS = 24
_SEED_FRACTION_MASK = (1 << _SEED_FRACTION_BITS) - 1
_SEED_STREAM_MULTIPLIERS = (
    0x9E3779,
    0x7F4A7D,
    0x6A09E7,
    0xBB67AF,
    0x3C6EF3,
    0xA54FF5,
    0x510E53,
    0x1F83D9,
    0x5BE0CD,
    0xC2B2AF,
    0x27D4EB,
    0x165667,
    0xD3A265,
    0xFD7047,
    0xB55A4F,
    0x94D049,
    0x369DEB,
    0xDB4F0B,
)


def _seed_mixed_value(seed: int, stream: int) -> int:
    if not 0 <= stream < len(_SEED_STREAM_MULTIPLIERS):
        raise ValueError("two-visible seed stream is out of range")
    multiplier = _SEED_STREAM_MULTIPLIERS[stream]
    if multiplier % 2 != 1:
        raise AssertionError("two-visible seed stream multiplier must be odd")
    return (
        (seed & _SEED_FRACTION_MASK) * multiplier + (stream + 1) * 0x45D9F3
    ) & _SEED_FRACTION_MASK


def _seed_unit_interval(seed: int, stream: int) -> float:
    """Map an integer seed to a deterministic non-period-eight unit scalar."""

    mixed = _seed_mixed_value(seed, stream)
    return (mixed + 0.5) / float(1 << _SEED_FRACTION_BITS)


def _seed_unique_unit_pair(seed: int) -> tuple[float, float]:
    """Return two exactly separated 12-bit coordinates for scene identity.

    The first mixed stream is a permutation modulo ``2**24``.  Frozen seeds
    are distinct modulo that value, and splitting all 24 bits into two 12-bit
    coordinates preserves injectivity.  Their physical increments are much
    larger than float32 spacing, so uniqueness is not merely probabilistic.
    """

    mixed = _seed_mixed_value(seed, 0)
    denominator = float(1 << 12)
    return (
        ((mixed >> 12) + 0.5) / denominator,
        ((mixed & ((1 << 12) - 1)) + 0.5) / denominator,
    )


def scene_specification(seed: int) -> TwoVisibleSceneSpecification:
    """Return one bounded deterministic scene specification.

    This pure arithmetic helper does not instantiate a simulator.  The public
    manifest boundary is responsible for deciding which seeds may be passed.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("two-visible scene seed must be an integer")
    unit_values = [_seed_unit_interval(seed, stream) for stream in range(18)]
    unit_values[0], unit_values[1] = _seed_unique_unit_pair(seed)
    unit = tuple(unit_values)
    signed = tuple(2.0 * value - 1.0 for value in unit)
    position = torch.tensor(
        [
            [
                -0.74 + 0.035 * signed[0],
                0.86 + 0.025 * signed[1],
                -0.04 + 0.020 * signed[2],
            ],
            [
                0.74 + 0.035 * signed[3],
                1.18 + 0.025 * signed[4],
                0.03 + 0.020 * signed[5],
            ],
        ],
        dtype=torch.float32,
    )
    velocity = torch.tensor(
        [
            [
                0.055 + 0.006 * signed[6],
                0.015 + 0.004 * signed[7],
                0.004 + 0.002 * signed[8],
            ],
            [
                -0.043 + 0.006 * signed[9],
                -0.011 + 0.004 * signed[10],
                -0.003 + 0.002 * signed[11],
            ],
        ],
        dtype=torch.float32,
    )
    palette = torch.tensor(
        [
            [
                0.92 + 0.025 * signed[12],
                0.20 + 0.025 * signed[13],
                0.14 + 0.020 * signed[14],
            ],
            [
                0.14 + 0.020 * signed[15],
                0.84 + 0.035 * signed[16],
                0.30 + 0.030 * signed[17],
            ],
        ],
        dtype=torch.float32,
    )
    swapped = bool(seed % 2)
    if swapped:
        palette = palette.flip(0)
    return TwoVisibleSceneSpecification(
        position=position,
        velocity=velocity,
        albedo=palette,
        palette_swapped=swapped,
    )


def bridge_protocol() -> dict[str, Any]:
    """Return the canonical, self-hashed qualification contract."""

    _assert_seed_namespaces()
    protocol: dict[str, Any] = {
        "name": "rgbd_two_visible_free_motion_bridge_v1",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "optimizer": None,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "source_binding": {
            "commit": "captured_at_execution_from_eventual_clean_harness_source",
            "dirty": False,
            "worktree_and_runtime_fingerprints_required": True,
            "development_checkpoint_report_and_protected_ledger_must_match": True,
        },
        "manifests": {
            "development": list(DEVELOPMENT_SEEDS),
            "selector": list(SELECTOR_SEEDS),
            "confirmation": list(CONFIRMATION_SEEDS),
            "final_test": list(FINAL_TEST_SEEDS),
        },
        "scene_family": {
            "constructor": (
                "world_model.training.rgbd_two_visible_free_motion_qualification."
                "construct_two_visible_episode"
            ),
            "object_count": 2,
            "world_radius_m": 0.21,
            "linear_drag": 0.05,
            "gravity": [0.0, 0.0, 0.0],
            "initial_speed_norm_bounds_mps": [0.035, 0.065],
            "frame_rate_hz": 20,
            "fully_visible": True,
            "image_separated": True,
            "non_contact": True,
            "palette_swapped_on_seed_parity": True,
            "continuous_seed_mapping": "24_bit_bijective_integer_streams",
            "no_split_local_scene_period": True,
            "preflight_before_return": True,
            "generic_ensure_collision_false_is_not_evidence": True,
        },
        "runtime": {
            "observation_factory": "world_model.training.loop.make_rgbd_packet",
            "runtime": "world_model.runtime.OnlineWorldModel",
            "ingested_frame_indices": list(HISTORY_FRAME_INDICES),
            "anchor_frame_index": ANCHOR_FRAME_INDEX,
            "public_rollout_method": "OnlineWorldModel.predict",
            "horizon_offsets_seconds": list(HORIZONS_SECONDS),
            "target_frame_indices": list(TARGET_FRAME_INDICES),
            "stream_key": RUNTIME_STREAM_KEY,
            "learned_parameters": 0,
            "persistent_module_tensor_state": 0,
            "association": "hard_Hungarian_discrete_stable_branch",
            "ambiguity_direct_writes": "fail_closed",
        },
        "perception": {
            "kind": "differentiable_symmetric_two_slot_RGB_D_geometry",
            "appearance_dim": 3,
            "chromatic_temperature": 0.05,
            "chromatic_centre_blend": 0.0025,
            "spatial_temperature_pixels": 1.0,
            "minimum_silhouette_gap_pixels": 2.0,
            "minimum_boundary_clearance_pixels": 2.0,
            "maximum_surface_radius_relative_error": 0.05,
        },
        "differentiability": {
            "kind": "per_object_fixed_output_vector_jacobian_products",
            "coefficients": list(VJP_COEFFICIENTS),
            "coefficient_reduction": "(output * coefficients).mean()",
            "inputs": ["rgb", "depth"],
            "outputs": list(VJP_OUTPUTS),
            "minimum_total_l1_per_object_target_modality": 1.0e-8,
            "current_position_owner": "anchor_frame_15_only",
            "current_position_required_history_frames": 1,
            "current_position_nonanchor_gradient_max_l1": 0.0,
            "temporal_output_minimum_l1_per_object_target_modality_frame": 1.0e-8,
            "temporal_output_required_history_frames": 16,
            "audit_scenes_per_split": 4,
            "floor_values_are_minima_and_ceiling_values_are_maxima_across_audit_scenes": True,
            "cross_scene_gradient_max_l1": 0.0,
        },
        "development_access": {
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "fixed_exclusive_durable_ledger": True,
            "record_before_materialization": True,
            "no_renamed_development_retry": True,
        },
        "protected_access": {
            "order": ["selector", "confirmation", "final_test"],
            "record_before_materialization": True,
            "exactly_once_exclusive_durable_ledger": True,
            "final_unopened_until_both_predecessors_pass": True,
            "clean_committed_source_and_reviewed_development_required": True,
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


def assert_rgbd_two_visible_config(config: OrpheusConfig) -> None:
    """Reject every semantic change to the frozen two-visible profile."""

    simulator_expected: dict[str, Any] = {
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
        "drag_range": (0.05, 0.05),
        "friction_range": (0.2, 0.2),
        "initial_speed_range": (0.035, 0.065),
        "camera_motion": "fixed",
        "render_noise_std": 0.0,
        "ensure_collision": False,
        "external_impulse_probability": 0.0,
        "scenario_mixture": ("baseline",),
    }
    for name, required in simulator_expected.items():
        actual = getattr(config.simulator, name)
        if actual != required:
            raise ValueError(f"two-visible RGB-D requires simulator.{name}={required!r}")
    if config.project.seed != DEVELOPMENT_SEEDS[0] or not config.project.deterministic:
        raise ValueError("two-visible RGB-D project seed/determinism differs from protocol")
    if config.device.preference != "cpu" or config.device.cuda_amp or config.device.compile:
        raise ValueError("two-visible RGB-D requires CPU float32 without compile")
    if config.model.max_objects != 2 or config.model.state.appearance_dim != 3:
        raise ValueError("two-visible RGB-D requires exactly two slots and appearance_dim three")
    if (
        config.model.lifecycle.birth_confidence != 0.5
        or config.model.lifecycle.birth_confirmations != 1
    ):
        raise ValueError("two-visible RGB-D requires exact lifecycle birth semantics")
    if config.model.rgb.enabled or not config.model.rgbd.enabled:
        raise ValueError("two-visible RGB-D requires only the composite RGB-D path")
    rgbd_expected: dict[str, Any] = {
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
        "measurement_position_variance": 0.000064,
        "temporal_history_size": 16,
        "temporal_min_samples": 16,
        "temporal_min_dt": 0.001,
        "temporal_velocity_variance_floor": 0.000001,
        "temporal_velocity_variance_ceiling": 0.01,
        "fit_conditioning_limit": 100.0,
    }
    for name, required in rgbd_expected.items():
        if getattr(config.model.rgbd, name) != required:
            raise ValueError(f"two-visible RGB-D requires model.rgbd.{name}={required!r}")
    association_expected = {
        "geometry_weight": 1.0,
        "appearance_weight": 0.25,
        "existence_weight": 0.0,
        "mahalanobis_gate": 100.0,
        "maximum_cost": 100.0,
        "ambiguity_margin": 0.02,
        "minimum_measurement_confidence": 0.5,
    }
    for name, required in association_expected.items():
        if getattr(config.model.association, name) != required:
            raise ValueError(f"two-visible RGB-D requires model.association.{name}={required!r}")
    if config.model.association.ambiguity_margin <= 0.0:
        raise ValueError("two-visible RGB-D requires a positive ambiguity margin")
    if not config.model.dynamics.analytic_free_motion_only:
        raise ValueError("two-visible RGB-D requires analytic-free-motion-only dynamics")
    if (
        config.model.dynamics.attention_residual_enabled
        or config.model.dynamics.max_substep != 1.0 / 120.0
    ):
        raise ValueError("two-visible RGB-D forbids learned dynamics residuals")
    if (
        config.model.filter.enable_learned_corrector
        or config.model.filter.learned_residual_scale != 0.0
        or not config.model.filter.direct_metric_position_update
        or not config.model.filter.innovation_anchored_correction
    ):
        raise ValueError("two-visible RGB-D requires only direct metric position correction")
    if config.model.identification.enabled:
        raise ValueError("two-visible RGB-D forbids online parameter identification")
    if (
        config.runtime.modality != "rgbd"
        or tuple(config.runtime.modality_order) != ("debug_oracle", "rgbd")
        or config.runtime.enable_debug_oracle
    ):
        raise ValueError("two-visible RGB-D forbids oracle/runtime modality substitution")
    if config.runtime.hypothesis_pool_enabled or not config.runtime.strict_timestamps:
        raise ValueError("two-visible RGB-D requires strict single-hypothesis execution")
    if config.training.batch_size != 4 or config.training.steps != 1:
        raise ValueError("two-visible shared config requires batch four and one schema step")
    if config.training.rgb_pretrain_steps != 0:
        raise ValueError("two-visible RGB-D has no RGB pretraining phase")
    if config.training.validation_episodes != len(DEVELOPMENT_SEEDS):
        raise ValueError("validation_episodes must match development manifest")
    if tuple(config.evaluation.horizons_seconds) != HORIZONS_SECONDS:
        raise ValueError("two-visible RGB-D horizons differ from protocol")
    if config.evaluation.rgb_only:
        raise ValueError("two-visible RGB-D evaluation requires RGB and depth")
    if config.simulator.radius_range != (
        config.model.rgbd.world_radius,
        config.model.rgbd.world_radius,
    ):
        raise ValueError("fixed radius prior differs from simulator")
    if config.simulator.drag_range != (
        config.model.rgbd.linear_drag,
        config.model.rgbd.linear_drag,
    ):
        raise ValueError("fixed drag prior differs from simulator")
    derived = tuple(
        ANCHOR_FRAME_INDEX + int(round(horizon * config.simulator.frame_rate))
        for horizon in HORIZONS_SECONDS
    )
    if derived != TARGET_FRAME_INDICES or derived[-1] != config.simulator.sequence_frames - 1:
        raise ValueError("two-visible target frame indices differ from declared horizons")


def new_public_model(config: OrpheusConfig) -> OnlineWorldModel:
    """Construct the only admitted public runtime and prove zero module state."""

    assert_rgbd_two_visible_config(config)
    model = OnlineWorldModel.from_config(config, device="cpu")
    if model.belief_factory.initial_radius != 0.21 or model.belief_factory.initial_drag != 0.05:
        raise RuntimeError("public runtime did not receive the frozen radius/drag priors")
    if tuple(model.parameters()) or tuple(model.buffers()) or model.state_dict():
        raise RuntimeError("two-visible public runtime must own zero parameter/buffer state")
    return model


def _stack_records(records: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot stack empty episode records")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise RuntimeError("two-visible episode record schema changed between frames")
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


def _event_record(
    physics: PhysicsStepEvents,
    *,
    created: Tensor,
    removed: Tensor,
    interval_start: float,
) -> dict[str, Tensor]:
    from world_model.simulator.collisions import BOUNDARY_NAMES

    first_event_time = torch.where(
        physics.first_event_offset >= 0,
        physics.first_event_offset + interval_start,
        physics.first_event_offset,
    )
    floor = BOUNDARY_NAMES.index("floor")
    walls = [index for index, name in enumerate(BOUNDARY_NAMES) if name not in {"floor", "ceiling"}]
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
        "ground_contact": physics.boundary_contact[:, floor].clone(),
        "ground_collision": physics.boundary_collision[:, floor].clone(),
        "wall_collision": physics.boundary_collision[:, walls].clone(),
        "collision": physics.collision.clone(),
        "contact": physics.contact.clone(),
        "sleeping": physics.sleeping.clone(),
        "external_impulse": physics.external_impulse.clone(),
        "externally_actuated": torch.linalg.vector_norm(physics.external_impulse, dim=-1) > 0,
        "created": created.clone(),
        "removed": removed.clone(),
        "first_event_time": first_event_time,
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


def _install_scene(world: SphereWorld, specification: TwoVisibleSceneSpecification) -> None:
    state = world.state
    if state.max_objects != 2:
        raise ValueError("two-visible constructor requires exactly two simulator slots")
    replacement = replace(
        state,
        object_id=torch.tensor([0, 1], dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=specification.position.clone(),
        velocity=specification.velocity.clone(),
        radius=torch.full((2, 1), 0.21, dtype=torch.float32),
        mass=torch.ones((2, 1), dtype=torch.float32),
        restitution=torch.full((2, 1), 0.7, dtype=torch.float32),
        drag=torch.full((2, 1), 0.05, dtype=torch.float32),
        friction=torch.full((2, 1), 0.2, dtype=torch.float32),
        albedo=specification.albedo.clone(),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1).clone(),
        angular_velocity=torch.zeros((2, 3), dtype=torch.float32),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    replacement.validate()
    world.state = replacement
    world._simulator_ids = torch.tensor([0, 1], dtype=torch.int64)
    world._spawn_frame = torch.tensor([0, 0], dtype=torch.int64)
    world._remove_frame = torch.tensor([-1, -1], dtype=torch.int64)
    world._spawn_position = specification.position.clone()
    world._spawn_velocity = specification.velocity.clone()


def construct_two_visible_episode(
    config: OrpheusConfig,
    seed: int,
    *,
    authorization: _ManifestAccessAuthorization,
) -> dict[str, Any]:
    """Construct and preflight one exact scene after manifest authorization."""

    authorization.authorize_seed(seed)
    assert_rgbd_two_visible_config(config)
    resolved = SphereWorldConfig.from_config(config)
    world = SphereWorld(resolved, seed)
    specification = scene_specification(seed)
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
    for frame_index, timestamp_tensor in enumerate(timestamps):
        timestamp = float(timestamp_tensor)
        lifecycle = world.apply_lifecycle(frame_index)
        camera = world.camera_frame(timestamp)
        rendered = world.render(camera=camera)
        labels = make_perception_labels(world.state, rendered, resolved.image_size)
        validate_perception_labels(labels, max_objects=2, image_size=resolved.image_size)
        rgb_frames.append(rendered.rgb)
        depth_frames.append(rendered.depth_buffer.unsqueeze(0))
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
            "scenario": "two_visible_free_motion",
            "camera_trajectory": world.camera.mode,
            "frame_rate": resolved.frame_rate,
            "physics_rate": resolved.physics_rate,
            "palette_swapped": specification.palette_swapped,
        },
    }
    validate_episode(episode, resolved)
    preflight_two_visible_episode(episode, config=config, specification=specification)
    return episode


def preflight_two_visible_episode(
    episode: Mapping[str, Any],
    *,
    config: OrpheusConfig,
    specification: TwoVisibleSceneSpecification | None = None,
) -> dict[str, float]:
    """Fail closed unless every frame belongs to the frozen observable family."""

    assert_rgbd_two_visible_config(config)
    objects = episode["objects"]
    labels = episode["labels"]
    events = episode["events"]
    active = objects["active"][:, :2]
    projected = labels["projected_valid"][:, :2]
    visible = objects["visible_fraction"][:, :2]
    if not bool(active.all()) or not bool(projected.all()):
        raise RuntimeError("two-visible preflight requires both spheres active and projectable")
    if not bool(visible.eq(1.0).all()):
        raise RuntimeError("two-visible preflight requires exact full visibility in every frame")
    full_mask = labels["full_mask"][:, :2]
    if bool((full_mask[:, 0] & full_mask[:, 1]).any()):
        raise RuntimeError("two-visible preflight rejects overlapping silhouettes")
    centres = labels["projected_center_pixels"][:, :2]
    radius_pixels = labels["apparent_radius"][:, :2]
    silhouette_gap = torch.linalg.vector_norm(centres[:, 0] - centres[:, 1], dim=-1)
    silhouette_gap = silhouette_gap - radius_pixels.sum(dim=-1)
    height, width = config.simulator.image_size
    boundary_clearance = torch.stack(
        (
            centres[..., 0] - radius_pixels,
            (width - 1) - centres[..., 0] - radius_pixels,
            centres[..., 1] - radius_pixels,
            (height - 1) - centres[..., 1] - radius_pixels,
        ),
        dim=-1,
    )
    if float(silhouette_gap.min()) < config.model.rgbd.minimum_silhouette_gap_pixels:
        raise RuntimeError("two-visible preflight rejects insufficient silhouette separation")
    if float(boundary_clearance.min()) < config.model.rgbd.minimum_boundary_clearance_pixels:
        raise RuntimeError("two-visible preflight rejects insufficient image-boundary clearance")
    pair_distance = torch.linalg.vector_norm(
        objects["position"][:, 0] - objects["position"][:, 1], dim=-1
    )
    surface_gap = pair_distance - objects["radius"][:, :2, 0].sum(dim=-1)
    bounds = torch.tensor(config.simulator.world_bounds, dtype=objects["position"].dtype)
    lower = objects["position"][:, :2] - objects["radius"][:, :2] - bounds[:, 0]
    upper = bounds[:, 1] - objects["position"][:, :2] - objects["radius"][:, :2]
    world_boundary = torch.minimum(lower, upper)
    if float(surface_gap.min()) < DEFAULT_GATES.minimum_world_surface_gap_m:
        raise RuntimeError("two-visible preflight rejects a near-contact trajectory")
    if float(world_boundary.min()) < DEFAULT_GATES.minimum_world_boundary_clearance_m:
        raise RuntimeError("two-visible preflight rejects a near-boundary trajectory")
    event_count = sum(
        int(events[name].ne(0).sum())
        for name in (
            "collision",
            "contact",
            "external_impulse",
            "removed",
        )
    )
    event_count += int(events["created"][1:].sum())
    if event_count:
        raise RuntimeError("two-visible preflight rejects collisions/contact/lifecycle events")
    if not bool(events["created"][0, :2].all()):
        raise RuntimeError("two-visible preflight requires both births at frame zero")
    for name, expected in (("radius", 0.21), ("drag", 0.05)):
        if not bool(objects[name][:, :2].eq(expected).all()):
            raise RuntimeError(f"two-visible preflight rejects non-frozen {name}")
    albedo = objects["albedo"][:, :2]
    cross_colour_distance = 1.0 - F.cosine_similarity(albedo[:, 0], albedo[:, 1], dim=-1)
    if float(cross_colour_distance.min()) < DEFAULT_GATES.minimum_cross_appearance_cosine_distance:
        raise RuntimeError("two-visible preflight rejects weak chromatic separation")
    if specification is not None:
        torch.testing.assert_close(objects["position"][0, :2], specification.position)
        torch.testing.assert_close(objects["velocity"][0, :2], specification.velocity)
        torch.testing.assert_close(objects["albedo"][0, :2], specification.albedo)
    return {
        "preflight_minimum_silhouette_gap_pixels": float(silhouette_gap.min()),
        "preflight_minimum_boundary_clearance_pixels": float(boundary_clearance.min()),
        "preflight_minimum_world_surface_gap_m": float(surface_gap.min()),
        "preflight_minimum_world_boundary_clearance_m": float(world_boundary.min()),
        "preflight_minimum_visible_fraction": float(visible.min()),
        "preflight_event_count": float(event_count),
        "preflight_minimum_palette_cosine_distance": float(cross_colour_distance.min()),
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Recompute the complete scalar gate surface from report evidence."""

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
    require_max("current_velocity_rmse_mps", gates.current_velocity_rmse_mps)
    require_max(
        "maximum_position_error_growth_slope_mps",
        gates.maximum_position_error_growth_slope_mps,
    )
    require_max("early_stationary_additive_regression_m", gates.early_stationary_additive_margin_m)
    require_max("long_stationary_rmse_ratio", gates.long_stationary_rmse_ratio)
    require_max("zero_velocity_rmse_ratio", gates.zero_velocity_rmse_ratio)
    for object_index in OBJECT_INDICES:
        for axis in AXIS_NAMES:
            require_max(
                f"current_position_rmse_m/object_{object_index}/{axis}",
                gates.per_object_axis_position_rmse_m,
            )
            require_max(
                f"current_velocity_rmse_mps/object_{object_index}/{axis}",
                gates.per_object_axis_velocity_rmse_mps,
            )
    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        require_max(
            f"horizon_{label}_position_rmse_m",
            gates.horizon_position_rmse_m[horizon_index],
        )
        require_max(
            f"horizon_{label}_velocity_rmse_mps",
            gates.horizon_velocity_rmse_mps,
        )
        for object_index in OBJECT_INDICES:
            for axis in AXIS_NAMES:
                require_max(
                    f"horizon_{label}_position_rmse_m/object_{object_index}/{axis}",
                    gates.per_object_axis_position_rmse_m
                    + gates.horizon_position_rmse_m[horizon_index],
                )
                require_max(
                    f"horizon_{label}_velocity_rmse_mps/object_{object_index}/{axis}",
                    gates.per_object_axis_velocity_rmse_mps,
                )

    require_equal("identity_switch_count", gates.identity_switch_count)
    require_equal("persistent_id_mismatch_count", gates.persistent_id_mismatch_count)
    require_min("identity_coverage", gates.identity_coverage)
    require_equal("persistent_object_id_min", gates.persistent_object_id_min)
    require_equal("persistent_object_id_max", gates.persistent_object_id_max)
    require_min("association_pair_coverage", gates.association_pair_coverage)
    require_equal(
        "association_ambiguous_pair_count",
        gates.association_ambiguous_pair_count,
    )
    require_min("minimum_hungarian_margin", gates.minimum_hungarian_margin)
    require_min(
        "minimum_position_assignment_margin_m",
        gates.minimum_position_assignment_margin_m,
    )
    require_min(
        "minimum_matched_appearance_cosine",
        gates.minimum_matched_appearance_cosine,
    )
    require_min(
        "minimum_cross_appearance_cosine_distance",
        gates.minimum_cross_appearance_cosine_distance,
    )
    require_equal("physical_palette_swap_fraction", gates.physical_palette_swap_fraction)
    require_min(
        "birth_slot_physical_zero_fraction",
        gates.birth_slot_physical_zero_fraction_min,
    )
    require_max(
        "birth_slot_physical_zero_fraction",
        gates.birth_slot_physical_zero_fraction_max,
    )
    require_min(
        "unique_scene_specification_fraction",
        gates.unique_scene_specification_fraction,
    )
    require_equal("gradient_audit_scene_count", gates.gradient_audit_scene_count)
    require_min(
        "gradient_audit_unique_scene_fraction",
        gates.gradient_audit_unique_scene_fraction,
    )

    require_min("active_fraction", gates.active_fraction)
    require_min("rollout_active_fraction", gates.rollout_active_fraction)
    require_equal("history_sample_count_min", gates.history_sample_count)
    require_equal("history_sample_count_max", gates.history_sample_count)
    require_equal("history_valid_count_min", gates.history_valid_count)
    require_equal("history_valid_count_max", gates.history_valid_count)
    require_max("history_span_max_abs_error_seconds", gates.history_span_tolerance_seconds)
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
    require_equal("direct_position_field_count", gates.direct_position_field_count)
    require_max(
        "direct_velocity_position_change_max_abs_m",
        gates.direct_velocity_position_change_max_abs_m,
    )
    require_max(
        "direct_metric_position_owner_max_abs_m",
        gates.direct_metric_position_owner_max_abs_m,
    )
    require_equal(
        "ambiguity_direct_position_write_count",
        gates.ambiguity_direct_position_write_count,
    )
    require_equal(
        "ambiguity_direct_velocity_write_count",
        gates.ambiguity_direct_velocity_write_count,
    )

    require_min(
        "preflight_minimum_silhouette_gap_pixels",
        gates.minimum_silhouette_gap_pixels,
    )
    require_min(
        "preflight_minimum_boundary_clearance_pixels",
        gates.minimum_boundary_clearance_pixels,
    )
    require_min("preflight_minimum_world_surface_gap_m", gates.minimum_world_surface_gap_m)
    require_min(
        "preflight_minimum_world_boundary_clearance_m",
        gates.minimum_world_boundary_clearance_m,
    )
    require_min("preflight_minimum_visible_fraction", gates.minimum_visible_fraction)
    require_equal("preflight_event_count", gates.preflight_event_count)
    require_min(
        "preflight_minimum_palette_cosine_distance",
        gates.minimum_cross_appearance_cosine_distance,
    )

    require_max("semigroup_position_max_abs_m", gates.semigroup_position_max_abs_m)
    require_max("semigroup_velocity_max_abs_mps", gates.semigroup_velocity_max_abs_mps)
    require_max("public_direct_position_max_abs_m", gates.public_direct_position_max_abs_m)
    require_max("public_direct_velocity_max_abs_mps", gates.public_direct_velocity_max_abs_mps)
    require_max(
        "analytic_position_agreement_max_abs_m",
        gates.analytic_position_agreement_max_abs_m,
    )
    require_max(
        "analytic_velocity_agreement_max_abs_mps",
        gates.analytic_velocity_agreement_max_abs_mps,
    )
    require_equal("public_rollout_output_alias_count", 0.0)
    require_max("public_query_time_max_abs_seconds", gates.public_query_time_max_abs_seconds)
    require_equal("ingested_frame_count_min", 16.0)
    require_equal("ingested_frame_count_max", 16.0)
    require_equal("public_predict_calls_per_batch_min", 1.0)
    require_equal("public_predict_calls_per_batch_max", 1.0)

    for object_index in OBJECT_INDICES:
        for output_name in VJP_OUTPUTS:
            for modality in ("rgb", "depth"):
                prefix = f"object_{object_index}/{output_name}/{modality}"
                require_min(
                    f"gradient_l1/{prefix}",
                    gates.minimum_input_gradient_l1,
                )
                require_max(
                    f"gradient_max_l1/{prefix}",
                    gates.maximum_input_gradient_l1,
                )
                require_max(
                    f"gradient_cross_scene_max_l1/{prefix}",
                    gates.maximum_cross_scene_gradient_l1,
                )
                if output_name == "current_position":
                    require_min(
                        f"gradient_anchor_history_frame_l1/{prefix}",
                        gates.minimum_history_frame_gradient_l1,
                    )
                    require_max(
                        f"gradient_nonanchor_max_history_frame_l1/{prefix}",
                        gates.current_position_nonanchor_gradient_max_l1,
                    )
                    require_equal(
                        f"gradient_supported_history_frames/{prefix}",
                        gates.current_position_required_history_gradient_frames,
                    )
                else:
                    require_min(
                        f"gradient_min_history_frame_l1/{prefix}",
                        gates.minimum_history_frame_gradient_l1,
                    )
                    require_equal(
                        f"gradient_supported_history_frames/{prefix}",
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


def _process_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _storage_alias(left: Tensor, right: Tensor) -> bool:
    return bool(
        left.numel()
        and right.numel()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _persistent_runtime_tensor_bytes(model: OnlineWorldModel) -> int:
    roots = {
        "state": model.state,
        "last_measurements": model.last_measurements,
        "last_direct_velocity_evidence": model.last_direct_velocity_evidence,
        "updater_diagnostics": model.updater.last_diagnostics,
        "scheduler_state": model.scheduler._sensor_state,
        "runtime_diagnostics": model.diagnostics.records,
    }
    pending: list[Any] = [roots]
    visited: set[int] = set()
    storages: dict[tuple[str, int | None, int, int], int] = {}
    while pending:
        value = pending.pop()
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            continue
        if isinstance(value, Tensor):
            storage = value.untyped_storage()
            byte_count = int(storage.nbytes())
            if byte_count:
                storages[
                    (
                        value.device.type,
                        value.device.index,
                        int(storage.data_ptr()),
                        byte_count,
                    )
                ] = byte_count
            continue
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
    return sum(storages.values())


def _gather_physical_by_slot(value: Tensor, physical_by_slot: Tensor) -> Tensor:
    if value.shape[0] != physical_by_slot.shape[0] or value.shape[-2:] != (2, 3):
        raise ValueError("truth gather expects [...,2,3] and [B,2] mapping")
    shape = (value.shape[0],) + (1,) * (value.ndim - 3) + (2, 1)
    index = physical_by_slot.reshape(shape).expand(*value.shape[:-2], 2, 3)
    return torch.gather(value, dim=-2, index=index)


def _birth_physical_mapping(estimate: Tensor, truth: Tensor) -> tuple[Tensor, Tensor]:
    if estimate.shape != truth.shape or estimate.shape[-2:] != (2, 3):
        raise ValueError("birth mapping requires matching [B,2,3] tensors")
    distance = torch.linalg.vector_norm(estimate[:, :, None] - truth[:, None], dim=-1)
    direct = distance[:, 0, 0] + distance[:, 1, 1]
    swapped = distance[:, 0, 1] + distance[:, 1, 0]
    use_swap = swapped < direct
    direct_map = torch.tensor([0, 1], dtype=torch.int64, device=estimate.device)
    swap_map = torch.tensor([1, 0], dtype=torch.int64, device=estimate.device)
    mapping = torch.where(use_swap[:, None], swap_map[None], direct_map[None])
    margin = (direct - swapped).abs()
    return mapping, margin


def _run_public_batch(batch: Mapping[str, Any], config: OrpheusConfig) -> dict[str, Any]:
    """Run the exact 16-frame public path with read-only association audits."""

    batch_size = int(batch["rgb"].shape[0])
    model = new_public_model(config)
    model.eval()
    model.reset(batch_size=batch_size)
    original_match = model.associator.match
    original_velocity = model.updater.correct_direct_velocity
    association_audit: dict[str, Any] = {
        "matched": 0,
        "opportunities": 0,
        "ambiguous": 0,
        "hungarian_margins": [],
        "position_margins": [],
        "appearance_cosines": [],
        "cross_appearance_distances": [],
        "last_measurement_by_belief": None,
    }
    correction_audit: dict[str, Any] = {
        "calls": 0,
        "valid": 0,
        "total": 0,
        "position_fields": 0,
        "position_change_max_abs": 0.0,
    }

    def recording_match(belief: Any, measurements: Any, predicted: Any) -> Any:
        result = original_match(belief, measurements, predicted)
        cost = model.associator.cost_matrix(measurements, predicted)
        for batch_index in range(cost.shape[0]):
            valid_predictions = torch.nonzero(predicted.valid_mask[batch_index]).flatten()
            valid_measurements = torch.nonzero(measurements.measurement_mask[batch_index]).flatten()
            if valid_predictions.numel() != 2 or valid_measurements.numel() != 2:
                continue
            association_audit["opportunities"] += 2
            pair_mask = result.pair_mask[batch_index]
            association_audit["matched"] += int(pair_mask.sum().detach().cpu())
            association_audit["ambiguous"] += int(
                result.ambiguous[batch_index, pair_mask].sum().detach().cpu()
            )
            if int(pair_mask.sum()) != 2:
                continue
            belief_to_measurement = torch.full((2,), -1, dtype=torch.int64, device=cost.device)
            for pair_index in torch.nonzero(pair_mask).flatten().tolist():
                belief_index = int(result.belief_indices[batch_index, pair_index])
                belief_to_measurement[belief_index] = result.measurement_indices[
                    batch_index, pair_index
                ]
            if bool((belief_to_measurement < 0).any()):
                continue
            row_for_belief = torch.full((2,), -1, dtype=torch.int64, device=cost.device)
            for row in valid_predictions.tolist():
                row_for_belief[int(predicted.belief_indices[batch_index, row])] = row
            selected = sum(
                cost[batch_index, row_for_belief[belief], belief_to_measurement[belief]]
                for belief in OBJECT_INDICES
            )
            alternate = sum(
                cost[
                    batch_index,
                    row_for_belief[belief],
                    belief_to_measurement[1 - belief],
                ]
                for belief in OBJECT_INDICES
            )
            margin = alternate - selected
            if not bool(torch.isfinite(margin)):
                margin = selected.new_tensor(2.0 * config.model.association.maximum_cost) - selected
            association_audit["hungarian_margins"].append(float(margin.detach().cpu()))
            predicted_position = torch.stack(
                [predicted.values[batch_index, row_for_belief[index]] for index in OBJECT_INDICES]
            )
            measured_position = measurements.values[batch_index, belief_to_measurement]
            correct_distance = torch.linalg.vector_norm(
                predicted_position - measured_position, dim=-1
            ).mean()
            cross_distance = torch.linalg.vector_norm(
                predicted_position - measured_position.flip(0), dim=-1
            ).mean()
            association_audit["position_margins"].append(
                float((cross_distance - correct_distance).detach().cpu())
            )
            if predicted.appearance is not None and measurements.appearance is not None:
                predicted_appearance = torch.stack(
                    [
                        predicted.appearance[batch_index, row_for_belief[index]]
                        for index in OBJECT_INDICES
                    ]
                )
                measured_appearance = measurements.appearance[batch_index, belief_to_measurement]
                matched_cosine = F.cosine_similarity(
                    predicted_appearance, measured_appearance, dim=-1
                )
                cross_cosine = F.cosine_similarity(
                    predicted_appearance, measured_appearance.flip(0), dim=-1
                )
                association_audit["appearance_cosines"].extend(
                    matched_cosine.detach().cpu().tolist()
                )
                association_audit["cross_appearance_distances"].extend(
                    (1.0 - cross_cosine).detach().cpu().tolist()
                )
            association_audit["last_measurement_by_belief"] = belief_to_measurement.detach().clone()
        return result

    def recording_velocity(prior: Any, evidence: DirectVelocityEvidence) -> Any:
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
        corrected = original_velocity(prior, evidence)
        correction_audit["position_change_max_abs"] = max(
            correction_audit["position_change_max_abs"],
            float((corrected.objects.position - before).abs().max().detach().cpu()),
        )
        return corrected

    model.associator.match = recording_match  # type: ignore[method-assign]
    model.updater.correct_direct_velocity = recording_velocity  # type: ignore[method-assign]
    identities: list[Tensor] = []
    active_masks: list[Tensor] = []
    observed_positions: list[Tensor] = []
    packet_count = 0
    for frame_index in HISTORY_FRAME_INDICES:
        posterior = model.ingest(make_rgbd_packet(batch, frame_index))
        packet_count += 1
        identities.append(posterior.objects.object_id)
        active_masks.append(posterior.objects.active)
        observed_positions.append(posterior.objects.position)
    trajectory = model.predict(HORIZONS_SECONDS).validate()
    belief = model.belief
    if belief is None:
        raise RuntimeError("two-visible runtime failed to retain a belief")
    history = model.state.temporal_histories.get(runtime_stream_key("rgbd", "camera0:rgbd"))
    if not isinstance(history, RGBDTemporalPositionHistory):
        raise RuntimeError("two-visible runtime failed to retain typed temporal histories")
    if runtime_stream_key("rgbd", "camera0:rgbd") != RUNTIME_STREAM_KEY:
        raise RuntimeError("two-visible runtime stream key changed")
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
    raw = model.last_measurements
    if raw is None or raw.supported_state_fields != ("position",):
        raise RuntimeError("two-visible RGB-D measurement must be the sole position owner")
    raw_position = raw.auxiliary.get("world_position")
    if not isinstance(raw_position, Tensor):
        raise RuntimeError("two-visible measurement omitted raw world position")
    measurement_by_belief = association_audit["last_measurement_by_belief"]
    if not isinstance(measurement_by_belief, Tensor):
        raise RuntimeError("two-visible final frame did not produce a complete association")
    if batch_size != 1:
        # The audit retains one common mapping only when batch rows agree.  For
        # report batches, reconstruct the nearest raw mapping row by row; this
        # is a diagnostic equality check, not an input to the runtime path.
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
    else:
        measurement_map = measurement_by_belief[None]
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
        "trajectory": trajectory,
        "history": history,
        "identities": torch.stack(identities, dim=1),
        "active_masks": torch.stack(active_masks, dim=1),
        "observed_positions": torch.stack(observed_positions, dim=1),
        "packet_count": packet_count,
        "predict_count": 1,
        "association_audit": association_audit,
        "correction_audit": correction_audit,
        "semigroup_position": (composed.objects.position - direct.objects.position).abs(),
        "semigroup_velocity": (composed.objects.velocity - direct.objects.velocity).abs(),
        "public_direct_position": (trajectory.positions[:, -1] - direct.objects.position).abs(),
        "public_direct_velocity": (trajectory.velocities[:, -1] - direct.objects.velocity).abs(),
        "analytic_position_agreement": (trajectory.positions - analytic_position).abs(),
        "analytic_velocity_agreement": (trajectory.velocities - analytic_velocity).abs(),
        "query_time_error": (trajectory.timestamps - expected_times).abs(),
        "output_alias_count": alias_count,
        "position_owner_count": 1 + int(correction_audit["position_fields"] > 0),
        "direct_metric_position_owner_error": (belief.objects.position - matched_raw).abs(),
        "runtime_tensor_bytes": _persistent_runtime_tensor_bytes(model),
    }


def _history_gradient_diagnostics(
    gradient: Tensor,
    *,
    object_index: int,
    output_name: str,
    modality: str,
) -> dict[str, float]:
    if gradient.ndim < 3 or gradient.shape[1] != len(HISTORY_FRAME_INDICES):
        raise ValueError("history VJP must retain an explicit [B,16,...] frame axis")
    reduction_dimensions = (0, *range(2, gradient.ndim))
    per_frame_l1 = gradient.abs().sum(dim=reduction_dimensions)
    if per_frame_l1.shape != (16,) or not bool(torch.isfinite(per_frame_l1).all()):
        raise FloatingPointError("two-visible history VJP is malformed or nonfinite")
    supported = per_frame_l1 >= DEFAULT_GATES.minimum_history_frame_gradient_l1
    suffix = f"object_{object_index}/{output_name}/{modality}"
    if output_name == "current_position":
        nonanchor = torch.cat((per_frame_l1[:ANCHOR_FRAME_INDEX], per_frame_l1[16:]))
        return {
            f"gradient_anchor_history_frame_l1/{suffix}": float(per_frame_l1[ANCHOR_FRAME_INDEX]),
            f"gradient_nonanchor_max_history_frame_l1/{suffix}": float(nonanchor.max()),
            f"gradient_supported_history_frames/{suffix}": float(supported.sum()),
        }
    return {
        f"gradient_min_history_frame_l1/{suffix}": float(per_frame_l1.min()),
        f"gradient_supported_history_frames/{suffix}": float(supported.sum()),
    }


def _gradient_metrics(config: OrpheusConfig, batch: Mapping[str, Any]) -> dict[str, float]:
    indices = torch.tensor(HISTORY_FRAME_INDICES, dtype=torch.int64)
    batch_size = int(batch["rgb"].shape[0])
    if batch_size != config.training.batch_size:
        raise ValueError("two-visible VJP audit requires one complete frozen batch")
    differentiable = dict(batch)
    differentiable["rgb"] = batch["rgb"].index_select(1, indices).clone().requires_grad_(True)
    differentiable["depth"] = batch["depth"].index_select(1, indices).clone().requires_grad_(True)
    differentiable["camera"] = {
        name: value.index_select(1, indices.to(value.device)).clone()
        if isinstance(value, Tensor)
        else value
        for name, value in batch["camera"].items()
    }
    differentiable["timestamps"] = batch["timestamps"].index_select(1, indices).clone()
    output = _run_public_batch(differentiable, config)
    belief = output["belief"]
    trajectory = output["trajectory"]
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
    inputs = (differentiable["rgb"], differentiable["depth"])
    objects = batch["objects"]
    scene_signatures = {
        canonical_sha256(
            {
                name: objects[name][batch_index, 0, :2].detach().cpu().tolist()
                for name in ("position", "velocity", "albedo")
            }
        )
        for batch_index in range(batch_size)
    }
    metrics: dict[str, float] = {
        "gradient_audit_scene_count": float(batch_size),
        "gradient_audit_unique_scene_fraction": len(scene_signatures) / batch_size,
    }
    for loss_index, (batch_index, object_index, output_name, loss) in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            inputs,
            retain_graph=loss_index + 1 < len(losses),
            allow_unused=True,
        )
        for modality, source, gradient in zip(("rgb", "depth"), inputs, gradients, strict=True):
            resolved = torch.zeros_like(source) if gradient is None else gradient
            if not bool(torch.isfinite(resolved).all()):
                raise FloatingPointError(
                    f"object {object_index} {output_name} has nonfinite {modality} VJP"
                )
            suffix = f"object_{object_index}/{output_name}/{modality}"
            per_scene_l1 = resolved.abs().reshape(batch_size, -1).sum(dim=-1)
            target = resolved[batch_index : batch_index + 1]
            gradient_l1 = float(per_scene_l1[batch_index])
            cross_scene = torch.cat((per_scene_l1[:batch_index], per_scene_l1[batch_index + 1 :]))
            cross_scene_max_l1 = float(cross_scene.max()) if cross_scene.numel() else 0.0
            diagnostics = {
                f"gradient_l1/{suffix}": gradient_l1,
                **_history_gradient_diagnostics(
                    target,
                    object_index=object_index,
                    output_name=output_name,
                    modality=modality,
                ),
            }
            for key, value in diagnostics.items():
                if key.startswith("gradient_nonanchor_max_history_frame_l1/"):
                    metrics[key] = max(metrics.get(key, value), value)
                else:
                    metrics[key] = min(metrics.get(key, value), value)
            maximum_key = f"gradient_max_l1/{suffix}"
            metrics[maximum_key] = max(metrics.get(maximum_key, gradient_l1), gradient_l1)
            cross_scene_key = f"gradient_cross_scene_max_l1/{suffix}"
            metrics[cross_scene_key] = max(
                metrics.get(cross_scene_key, cross_scene_max_l1),
                cross_scene_max_l1,
            )
    return metrics


def _ambiguity_fail_closed_metrics(
    config: OrpheusConfig,
    batch: Mapping[str, Any],
) -> dict[str, float]:
    """Prove an explicitly ambiguous pair cannot directly write position/history."""

    model = new_public_model(config)
    model.eval()
    model.reset(batch_size=1)
    model.ingest(make_rgbd_packet(batch, 0))
    history_before = model.state.temporal_histories[RUNTIME_STREAM_KEY]
    if not isinstance(history_before, RGBDTemporalPositionHistory):
        raise RuntimeError("ambiguity fixture did not establish RGB-D history")
    valid_before = history_before.valid_mask.sum()
    original_match = model.associator.match

    def force_ambiguous(belief: Any, measurements: Any, predicted: Any) -> Any:
        result = original_match(belief, measurements, predicted)
        return replace(result, ambiguous=result.pair_mask.clone())

    model.associator.match = force_ambiguous  # type: ignore[method-assign]
    model.ingest(make_rgbd_packet(batch, 1))
    diagnostics = model.updater.last_diagnostics
    history_after = model.state.temporal_histories[RUNTIME_STREAM_KEY]
    if diagnostics is None or not isinstance(history_after, RGBDTemporalPositionHistory):
        raise RuntimeError("ambiguity fixture omitted correction/history diagnostics")
    direct_position_writes = int(diagnostics.analytic_gain.ne(0).sum().detach().cpu())
    newly_valid_history = int((history_after.valid_mask.sum() - valid_before).detach().cpu())
    return {
        "ambiguity_direct_position_write_count": float(direct_position_writes),
        "ambiguity_direct_velocity_write_count": float(
            int(model.last_direct_velocity_evidence is not None) + newly_valid_history
        ),
    }


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
    rss_after = _process_max_rss_bytes()
    return {
        "perception_latency_seconds": float(median(perception)),
        "state_only_rollout_latency_seconds": float(median(rollout)),
        "process_max_rss_bytes": float(rss_after),
        "process_rss_delta_bytes": float(max(0, rss_after - rss_before)),
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
        raise ValueError(f"{split!r} must use its exact frozen two-visible RGB-D manifest")
    if len(requested) != len(set(requested)):
        raise ValueError("two-visible RGB-D manifest contains duplicate seeds")
    return requested


def _episode_scene_signature(episode: Mapping[str, Any]) -> str:
    objects = episode["objects"]
    return canonical_sha256(
        {
            name: objects[name][0, :2].detach().cpu().tolist()
            for name in ("position", "velocity", "albedo")
        }
    )


def evaluate_seed_manifest(
    config: OrpheusConfig,
    seeds: Sequence[int],
    *,
    split: str,
    authorization: _ManifestAccessAuthorization | None = None,
) -> dict[str, Any]:
    """Evaluate one exact already-authorized manifest with zero optimizer work."""

    assert_rgbd_two_visible_config(config)
    requested = _validate_manifest(split, seeds)
    if authorization is None:
        raise PermissionError("exact manifest evaluation requires a durable access authorization")
    authorization.begin_manifest(split, requested)
    accumulated: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "current_position_error",
            "current_velocity_error",
            "future_position_error",
            "future_velocity_error",
            "stationary_position_error",
            "zero_velocity_error",
            "semigroup_position",
            "semigroup_velocity",
            "public_direct_position",
            "public_direct_velocity",
            "analytic_position_agreement",
            "analytic_velocity_agreement",
            "query_time_error",
            "history_sample_count",
            "history_valid_count",
            "history_span_error",
            "direct_metric_position_owner_error",
        )
    }
    first_batch: Mapping[str, Any] | None = None
    identity_switch_count = 0
    identity_correct = 0
    identity_total = 0
    persistent_ids: list[Tensor] = []
    persistent_id_mismatch_count = 0
    birth_slot_zero = 0
    birth_mapping_count = 0
    active_count = 0
    active_total = 0
    rollout_active_count = 0
    rollout_active_total = 0
    association_matched = 0
    association_opportunities = 0
    association_ambiguous = 0
    hungarian_margins: list[float] = []
    position_margins: list[float] = []
    matched_appearance: list[float] = []
    cross_appearance_distance: list[float] = []
    direct_calls: list[int] = []
    direct_valid = 0
    direct_total = 0
    direct_position_fields = 0
    direct_position_change_max = 0.0
    position_owner_counts: list[int] = []
    packet_counts: list[int] = []
    predict_counts: list[int] = []
    alias_count = 0
    runtime_state_bytes_max = 0
    preflight_values: dict[str, list[float]] = {}
    scene_signatures: set[str] = set()

    for seed_chunk in _chunks(requested, config.training.batch_size):
        episodes = [
            construct_two_visible_episode(config, seed, authorization=authorization)
            for seed in seed_chunk
        ]
        for episode in episodes:
            scene_signatures.add(_episode_scene_signature(episode))
            evidence = preflight_two_visible_episode(episode, config=config)
            for name, value in evidence.items():
                preflight_values.setdefault(name, []).append(value)
        batch = collate_episodes(episodes)
        if first_batch is None:
            first_batch = batch
        with torch.no_grad():
            output = _run_public_batch(batch, config)
        belief = output["belief"]
        trajectory = output["trajectory"]
        history = output["history"]
        observed = output["observed_positions"]
        birth_mapping, _ = _birth_physical_mapping(
            observed[:, 0], batch["objects"]["position"][:, 0, :2]
        )
        birth_slot_zero += int((birth_mapping[:, 0] == 0).sum().detach().cpu())
        birth_mapping_count += birth_mapping.shape[0]
        for frame_offset, frame_index in enumerate(HISTORY_FRAME_INDICES):
            frame_mapping, _ = _birth_physical_mapping(
                observed[:, frame_offset], batch["objects"]["position"][:, frame_index, :2]
            )
            identity_switch_count += int(
                (frame_mapping != birth_mapping).any(dim=-1).sum().detach().cpu()
            )
            identity_correct += int((frame_mapping == birth_mapping).all(dim=-1).sum()) * 2
            identity_total += frame_mapping.shape[0] * 2
        anchor_position = _gather_physical_by_slot(
            batch["objects"]["position"][:, ANCHOR_FRAME_INDEX, :2], birth_mapping
        )
        anchor_velocity = _gather_physical_by_slot(
            batch["objects"]["velocity"][:, ANCHOR_FRAME_INDEX, :2], birth_mapping
        )
        target_indices = torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64)
        future_position = _gather_physical_by_slot(
            batch["objects"]["position"][:, :, :2].index_select(1, target_indices),
            birth_mapping,
        )
        future_velocity = _gather_physical_by_slot(
            batch["objects"]["velocity"][:, :, :2].index_select(1, target_indices),
            birth_mapping,
        )
        accumulated["current_position_error"].append(
            (belief.objects.position - anchor_position).cpu()
        )
        accumulated["current_velocity_error"].append(
            (belief.objects.velocity - anchor_velocity).cpu()
        )
        accumulated["future_position_error"].append((trajectory.positions - future_position).cpu())
        accumulated["future_velocity_error"].append((trajectory.velocities - future_velocity).cpu())
        accumulated["stationary_position_error"].append(
            (belief.objects.position[:, None] - future_position).cpu()
        )
        accumulated["zero_velocity_error"].append((-anchor_velocity).cpu())
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
        sample_count = history.sample_mask.sum(dim=-1)
        valid_count = history.valid_mask.sum(dim=-1)
        span = history.timestamps[..., -1] - history.timestamps[..., 0]
        accumulated["history_sample_count"].append(sample_count.cpu())
        accumulated["history_valid_count"].append(valid_count.cpu())
        accumulated["history_span_error"].append(
            (span - DEFAULT_GATES.history_span_seconds).abs().cpu()
        )
        identities = output["identities"]
        persistent_ids.append(identities.cpu())
        expected_ids = torch.tensor([0, 1], dtype=torch.int64, device=identities.device)
        persistent_id_mismatch_count += int(
            identities.ne(expected_ids.view(1, 1, 2)).sum().detach().cpu()
        )
        active = output["active_masks"]
        active_count += int(active.sum().detach().cpu())
        active_total += active.numel()
        rollout_active_count += int(trajectory.active_mask.sum().detach().cpu())
        rollout_active_total += trajectory.active_mask.numel()
        association = output["association_audit"]
        association_matched += int(association["matched"])
        association_opportunities += int(association["opportunities"])
        association_ambiguous += int(association["ambiguous"])
        hungarian_margins.extend(association["hungarian_margins"])
        position_margins.extend(association["position_margins"])
        matched_appearance.extend(association["appearance_cosines"])
        cross_appearance_distance.extend(association["cross_appearance_distances"])
        correction = output["correction_audit"]
        direct_calls.append(int(correction["calls"]))
        direct_valid += int(correction["valid"])
        direct_total += int(correction["total"])
        direct_position_fields += int(correction["position_fields"])
        direct_position_change_max = max(
            direct_position_change_max,
            float(correction["position_change_max_abs"]),
        )
        position_owner_counts.append(int(output["position_owner_count"]))
        packet_counts.append(int(output["packet_count"]))
        predict_counts.append(int(output["predict_count"]))
        alias_count += int(output["output_alias_count"])
        runtime_state_bytes_max = max(runtime_state_bytes_max, int(output["runtime_tensor_bytes"]))

    authorization.finish_manifest()
    if first_batch is None:
        raise RuntimeError("two-visible manifest unexpectedly produced no batches")
    tensors = {name: torch.cat(values) for name, values in accumulated.items()}
    current_position_rmse = _rmse(tensors["current_position_error"])
    current_velocity_rmse = _rmse(tensors["current_velocity_error"])
    future_position_rmse = [
        _rmse(tensors["future_position_error"][:, index]) for index in range(len(HORIZONS_SECONDS))
    ]
    future_velocity_rmse = [
        _rmse(tensors["future_velocity_error"][:, index]) for index in range(len(HORIZONS_SECONDS))
    ]
    stationary_rmse = [
        _rmse(tensors["stationary_position_error"][:, index])
        for index in range(len(HORIZONS_SECONDS))
    ]
    zero_velocity_rmse = _rmse(tensors["zero_velocity_error"])
    epsilon = torch.finfo(torch.float64).eps
    model = new_public_model(config)
    learned = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    buffers = tuple(model.buffers())
    state_tensors = tuple(model.state_dict().values())
    metrics: dict[str, Any] = {
        "current_position_rmse_m": current_position_rmse,
        "current_velocity_rmse_mps": current_velocity_rmse,
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
        "identity_switch_count": float(identity_switch_count),
        "persistent_id_mismatch_count": float(persistent_id_mismatch_count),
        "identity_coverage": identity_correct / identity_total,
        "persistent_object_id_min": float(torch.cat(persistent_ids).min()),
        "persistent_object_id_max": float(torch.cat(persistent_ids).max()),
        "association_pair_coverage": (
            association_matched / association_opportunities if association_opportunities else 0.0
        ),
        "association_ambiguous_pair_count": float(association_ambiguous),
        "minimum_hungarian_margin": min(hungarian_margins, default=float("nan")),
        "minimum_position_assignment_margin_m": min(position_margins, default=float("nan")),
        "minimum_matched_appearance_cosine": min(matched_appearance, default=float("nan")),
        "minimum_cross_appearance_cosine_distance": min(
            cross_appearance_distance, default=float("nan")
        ),
        "physical_palette_swap_fraction": sum(seed % 2 for seed in requested) / len(requested),
        "birth_slot_physical_zero_fraction": birth_slot_zero / birth_mapping_count,
        "unique_scene_specification_fraction": len(scene_signatures) / len(requested),
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
        "direct_position_field_count": float(direct_position_fields),
        "direct_velocity_position_change_max_abs_m": direct_position_change_max,
        "direct_metric_position_owner_max_abs_m": float(
            tensors["direct_metric_position_owner_error"].max()
        ),
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
        "persistent_runtime_tensor_state_bytes_max": float(runtime_state_bytes_max),
        "learned_parameter_count": float(sum(value.numel() for value in learned)),
        "learned_parameter_bytes": float(
            sum(value.numel() * value.element_size() for value in learned)
        ),
        "module_tensor_buffer_count": float(len(buffers)),
        "persistent_module_state_key_count": float(len(state_tensors)),
        "persistent_module_state_bytes": float(
            sum(value.numel() * value.element_size() for value in state_tensors)
        ),
        "optimizer_updates": 0.0,
        "optimizer_state_entry_count": 0.0,
    }
    for object_index in OBJECT_INDICES:
        for axis_index, axis in enumerate(AXIS_NAMES):
            metrics[f"current_position_rmse_m/object_{object_index}/{axis}"] = _rmse(
                tensors["current_position_error"][:, object_index, axis_index]
            )
            metrics[f"current_velocity_rmse_mps/object_{object_index}/{axis}"] = _rmse(
                tensors["current_velocity_error"][:, object_index, axis_index]
            )
    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = future_position_rmse[horizon_index]
        metrics[f"horizon_{label}_velocity_rmse_mps"] = future_velocity_rmse[horizon_index]
        for object_index in OBJECT_INDICES:
            for axis_index, axis in enumerate(AXIS_NAMES):
                metrics[f"horizon_{label}_position_rmse_m/object_{object_index}/{axis}"] = _rmse(
                    tensors["future_position_error"][:, horizon_index, object_index, axis_index]
                )
                metrics[f"horizon_{label}_velocity_rmse_mps/object_{object_index}/{axis}"] = _rmse(
                    tensors["future_velocity_error"][:, horizon_index, object_index, axis_index]
                )
    for name, values in preflight_values.items():
        metrics[name] = min(values)
    metrics.update(_ambiguity_fail_closed_metrics(config, first_batch))
    metrics.update(_gradient_metrics(config, first_batch))
    metrics.update(_latency_metrics(config, first_batch))
    for name, value in metrics.items():
        if _numeric(value) is None:
            raise FloatingPointError(f"two-visible metric {name!r} is missing or nonfinite")
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
        "scene_constructor": "construct_two_visible_episode_with_full_frame_preflight",
    }


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_RELATIVE_PATH = Path("runs/rgbd_two_visible_bridge_v1")


def development_ledger_path() -> Path:
    return (
        REPOSITORY_ROOT
        / RUN_RELATIVE_PATH
        / f"development_attempt_{ARCHITECTURE_ATTEMPT}_access.json"
    ).resolve()


def qualification_ledger_path() -> Path:
    return (
        REPOSITORY_ROOT
        / RUN_RELATIVE_PATH
        / f"qualification_attempt_{ARCHITECTURE_ATTEMPT}_access.json"
    ).resolve()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def require_frozen_config(path: str | Path) -> OrpheusConfig:
    """Load only the exact frozen raw profile and semantic contract."""

    source = Path(path)
    contents = stable_read_bytes(source, label="two-visible frozen config")
    digest = sha256_bytes(contents)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "two-visible RGB-D requires exact frozen config bytes: "
            f"expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    config = load_config(source)
    assert_rgbd_two_visible_config(config)
    return config


def _require_config_matches_frozen_path(config: OrpheusConfig, path: Path) -> None:
    """Bind the exact executed config object to the immutable profile bytes."""

    before = stable_read_bytes(path, label="two-visible frozen config binding")
    if sha256_bytes(before) != FROZEN_CONFIG_SHA256:
        raise ValueError("two-visible config path differs from frozen bytes")
    parsed = load_config(path)
    after = stable_read_bytes(path, label="two-visible frozen config binding recheck")
    if after != before:
        raise RuntimeError("two-visible config bytes changed while being parsed")
    if canonical_sha256(config.to_dict()) != canonical_sha256(parsed.to_dict()):
        raise ValueError("executed config object differs from exact frozen config bytes")
    assert_rgbd_two_visible_config(parsed)


def _atomic_temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


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
        # A leftover temporary makes durability ambiguous and permanently
        # blocks re-entry; never silently delete this evidence.
        raise


class DevelopmentLedger:
    """Fixed attempt-scoped receipt preventing repeated development access."""

    ARTIFACT_KIND = "rgbd_two_visible_development_access_ledger"

    def __init__(self, bindings: Mapping[str, Any]) -> None:
        self.path = development_ledger_path()
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "bindings": dict(bindings),
            "attempt_reserved": True,
            "access_started": True,
            "development_data_materialized": True,
            "result_sha256": None,
            "status": "development_materialization_started",
        }
        self._authorization_issued = False
        self._authorization: _ManifestAccessAuthorization | None = None
        _durable_create(self.path, self._serialized())

    def _serialized(self) -> bytes:
        return (
            json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )

    def _replace(self) -> None:
        _durable_replace(self.path, self._serialized())

    def authorization(self) -> _ManifestAccessAuthorization:
        if self._authorization_issued:
            raise RuntimeError("development manifest authorization cannot be issued twice")
        self._authorization_issued = True
        self._authorization = _ManifestAccessAuthorization(
            _MANIFEST_ACCESS_AUTHORITY,
            split="development",
            seeds=DEVELOPMENT_SEEDS,
            ledger_path=self.path,
            ledger_kind=self.ARTIFACT_KIND,
        )
        return self._authorization

    def complete_evaluation(self, result: Mapping[str, Any]) -> None:
        if self.record["status"] != "development_materialization_started":
            raise RuntimeError("development evaluation was not durably opened")
        if self._authorization is None:
            raise RuntimeError("development manifest authorization was not issued")
        self._authorization.require_finished()
        passed = result.get("passed") is True
        self.record["result_sha256"] = canonical_sha256(result)
        self.record["outcome"] = "passed" if passed else "failed"
        self.record["status"] = "development_artifacts_pending"
        self._replace()

    def finish(
        self,
        *,
        report_sha256: str,
        checkpoint_sha256: str | None,
    ) -> None:
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

    def record_error(
        self,
        error: BaseException,
        *,
        report_sha256: str | None = None,
    ) -> None:
        if self.record.get("status") == "complete":
            raise RuntimeError("completed development ledger cannot be downgraded")
        self.record["status"] = "error"
        self.record["error"] = {"type": type(error).__name__, "message": str(error)}
        if report_sha256 is not None:
            self.record["report_sha256"] = validated_sha256(
                report_sha256, label="failed development report SHA-256"
            )
        self._replace()


class QualificationLedger:
    """Fixed exclusive selector -> confirmation -> one-shot-final receipt."""

    ARTIFACT_KIND = "rgbd_two_visible_exactly_once_access_ledger"
    ORDER = ("selector", "confirmation", "final_test")
    MANIFESTS = {
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }

    def __init__(self, bindings: Mapping[str, Any]) -> None:
        self.path = qualification_ledger_path()
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
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
        self._authorizations: dict[str, _ManifestAccessAuthorization] = {}
        _durable_create(self.path, self._serialized())

    def _serialized(self) -> bytes:
        return (
            json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )

    def _replace(self) -> None:
        _durable_replace(self.path, self._serialized())

    def begin_access(self, split: str) -> _ManifestAccessAuthorization:
        if split not in self.ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        index = self.ORDER.index(split)
        for predecessor in self.ORDER[:index]:
            if self.record["splits"][predecessor]["status"] != "passed":
                raise RuntimeError(f"{split} must remain unopened until {predecessor} passes")
        state = self.record["splits"][split]
        if state["status"] != "unopened" or state["access_started"] is not False:
            raise RuntimeError(f"protected split {split!r} cannot be opened twice")
        if any(self.record["splits"][later]["access_started"] for later in self.ORDER[index + 1 :]):
            raise RuntimeError("protected access order is inconsistent")
        state["access_started"] = True
        state["status"] = "materialization_started"
        self.record["protected_data_materialized"] = True
        self.record["status"] = f"{split}_materialization_started"
        self._replace()
        authorization = _ManifestAccessAuthorization(
            _MANIFEST_ACCESS_AUTHORITY,
            split=split,
            seeds=self.MANIFESTS[split],
            ledger_path=self.path,
            ledger_kind=self.ARTIFACT_KIND,
        )
        self._authorizations[split] = authorization
        return authorization

    def complete_split(self, split: str, result: Mapping[str, Any]) -> None:
        if split not in self.ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        state = self.record["splits"][split]
        if state["status"] != "materialization_started":
            raise RuntimeError(f"protected split {split!r} was not durably opened")
        authorization = self._authorizations.get(split)
        if authorization is None:
            raise RuntimeError(f"protected split {split!r} lacks manifest authorization")
        authorization.require_finished()
        passed = result.get("passed") is True
        state["status"] = "passed" if passed else "failed"
        state["result_sha256"] = canonical_sha256(result)
        self.record["status"] = f"{split}_{state['status']}"
        self._replace()

    def prepare_report(self, *, passed: bool, stopped_after: str) -> None:
        if passed and any(
            self.record["splits"][split]["status"] != "passed" for split in self.ORDER
        ):
            raise RuntimeError("qualification cannot prepare report before all splits pass")
        self.record["outcome"] = "passed" if passed else "failed"
        self.record["stopped_after"] = stopped_after
        self.record["status"] = "qualification_report_write_pending"
        self._replace()

    def finish(self, *, report_sha256: str) -> None:
        if self.record["status"] != "qualification_report_write_pending":
            raise RuntimeError("qualification report was not durably pending")
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
            raise RuntimeError("completed qualification ledger cannot be downgraded")
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
        raise RuntimeError("two-visible public runtime state_dict must remain empty")
    return EMPTY_MODEL_STATE_SHA256


def _validate_development_split(development: Mapping[str, Any]) -> None:
    if development.get("split") != "development":
        raise ValueError("reviewed evidence has the wrong development split")
    expected_seeds = list(DEVELOPMENT_SEEDS)
    if development.get("seeds") != expected_seeds:
        raise ValueError("reviewed evidence has the wrong development manifest")
    if development.get("seed_manifest_sha256") != canonical_sha256(expected_seeds):
        raise ValueError("reviewed evidence has the wrong development manifest hash")
    if (
        type(development.get("optimizer_updates")) is not int
        or development.get("optimizer_updates") != 0
    ):
        raise ValueError("reviewed development must prove zero optimizer updates")
    expected_api = {
        "packet_factory": "make_rgbd_packet",
        "ingest_frames": list(HISTORY_FRAME_INDICES),
        "rollout_method": "OnlineWorldModel.predict",
        "horizons_seconds": list(HORIZONS_SECONDS),
    }
    if canonical_sha256(development.get("runtime_api")) != canonical_sha256(expected_api):
        raise ValueError("reviewed development did not use exact public runtime APIs")
    if development.get("scene_constructor") != (
        "construct_two_visible_episode_with_full_frame_preflight"
    ):
        raise ValueError("reviewed development did not use the frozen scene constructor")
    metrics = development.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("reviewed development is missing metrics")
    failures = gate_failures(metrics)
    if failures or development.get("failures") != failures:
        raise ValueError("reviewed development gates do not recompute as passed")
    if development.get("passed") is not True:
        raise ValueError("reviewed development did not pass")


def validate_development_evidence(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind reviewed development to exact eventual clean harness source."""

    if report.get("artifact_kind") != "rgbd_two_visible_development":
        raise ValueError("reviewed development report has the wrong artifact kind")
    if report.get("passed") is not True or report.get("review_ready") is not True:
        raise ValueError("reviewed development evidence did not pass")
    if report.get("protected_data_materialized") is not False:
        raise ValueError("reviewed development must leave protected data unopened")
    if (
        type(report.get("optimizer_updates")) is not int
        or report.get("optimizer_updates") != 0
        or report.get("stopped_after") != "development"
    ):
        raise ValueError("reviewed development has an invalid execution boundary")
    if canonical_sha256(report.get("protocol")) != canonical_sha256(bridge_protocol()):
        raise ValueError("reviewed development protocol differs from frozen source")
    if report.get("config_sha256") != FROZEN_CONFIG_SHA256:
        raise ValueError("reviewed development config hash differs from frozen bytes")
    if canonical_sha256(report.get("source_provenance")) != canonical_sha256(source):
        raise ValueError("reviewed development source differs from current clean source")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reviewed development report does not bind checkpoint")
    if report.get("checkpoint_model_state_sha256") != EMPTY_MODEL_STATE_SHA256:
        raise ValueError("reviewed development checkpoint did not bind empty model state")
    development = report.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("reviewed report is missing development evidence")
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
    expected_bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_manifest_sha256": canonical_sha256(list(DEVELOPMENT_SEEDS)),
    }
    if record.get("artifact_kind") != DevelopmentLedger.ARTIFACT_KIND:
        raise ValueError("reviewed development ledger has the wrong artifact kind")
    if (
        type(record.get("architecture_attempt")) is not int
        or record.get("architecture_attempt") != ARCHITECTURE_ATTEMPT
    ):
        raise ValueError("reviewed development ledger has the wrong architecture attempt")
    if record.get("status") != "complete" or record.get("outcome") != "passed":
        raise ValueError("reviewed development ledger is not terminally passed")
    if record.get("attempt_reserved") is not True or record.get("access_started") is not True:
        raise ValueError("reviewed development ledger lacks an exclusive access receipt")
    if record.get("development_data_materialized") is not True:
        raise ValueError("reviewed development ledger lacks materialization evidence")
    if canonical_sha256(record.get("bindings")) != canonical_sha256(expected_bindings):
        raise ValueError("reviewed development ledger bindings differ")
    if record.get("result_sha256") != canonical_sha256(development):
        raise ValueError("reviewed development ledger result hash differs")
    if record.get("report_sha256") != report_sha256:
        raise ValueError("reviewed development ledger report hash differs")
    if record.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reviewed development ledger checkpoint hash differs")
    if report.get("development_ledger") != str(development_ledger_path()):
        raise ValueError("reviewed development report names the wrong fixed ledger")


def _load_checkpoint_payload(contents: bytes) -> Mapping[str, Any]:
    payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("reviewed checkpoint payload must be a mapping")
    return payload


def _save_review_checkpoint(
    path: Path,
    *,
    model: OnlineWorldModel,
    config: OrpheusConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    """Atomically write a weight-only-safe, project-compatible checkpoint."""

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
    # This qualification is deliberately non-resumable and optimizer-free.
    # Removing Python/NumPy RNG tuples keeps the reviewed payload loadable by
    # PyTorch's restricted ``weights_only`` unpickler.
    payload.pop("rng", None)
    temporary = _atomic_temporary(path)
    if _lexists(path) or _lexists(temporary):
        raise FileExistsError(f"two-visible checkpoint path must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_parent(path)


def validate_checkpoint_evidence(
    payload: Mapping[str, Any],
    *,
    config: OrpheusConfig,
    source: Mapping[str, Any],
    development: Mapping[str, Any],
) -> None:
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or model_state:
        raise ValueError("reviewed two-visible checkpoint model state must be empty")
    if type(payload.get("step")) is not int or payload.get("step") != 0:
        raise ValueError("reviewed two-visible checkpoint must be step zero")
    if payload.get("optimizer_state") is not None or payload.get("scheduler_state") is not None:
        raise ValueError("reviewed two-visible checkpoint must be optimizer-free")
    if "rng" in payload:
        raise ValueError("reviewed two-visible checkpoint must be non-resumable and RNG-free")
    if canonical_sha256(payload.get("config")) != canonical_sha256(
        config.to_dict()
    ) or canonical_sha256(payload.get("git")) != canonical_sha256(source):
        raise ValueError("reviewed two-visible checkpoint config/source differs")
    metrics = payload.get("metrics")
    expected = {
        "artifact_kind": "rgbd_two_visible_empty_model_state",
        "optimizer_updates": 0,
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "protocol": bridge_protocol(),
        "development": development,
    }
    if canonical_sha256(metrics) != canonical_sha256(expected):
        raise ValueError("reviewed two-visible checkpoint evidence differs from protocol")
    roundtrip = new_public_model(config)
    roundtrip.load_state_dict(model_state, strict=True)
    if _model_state_sha256(roundtrip) != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("checkpoint roundtrip changed empty model state")


def _guard_frozen_inputs(
    *,
    source: Mapping[str, Any],
    config: OrpheusConfig,
    config_path: Path,
    development_report_path: Path | None = None,
    development_report_sha256: str | None = None,
    development_ledger_path_value: Path | None = None,
    development_ledger_sha256: str | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    model: OnlineWorldModel | None = None,
) -> None:
    current = clean_source(
        capture_git_metadata(REPOSITORY_ROOT),
        label="two-visible RGB-D execution guard",
    )
    if current != source:
        raise RuntimeError("source provenance changed during two-visible execution")
    _require_config_matches_frozen_path(config, config_path)
    protocol = bridge_protocol()
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol["protocol_sha256"] != canonical_sha256(unsigned):
        raise RuntimeError("two-visible protocol self-hash is inconsistent")
    if (
        development_report_path is not None
        and sha256_bytes(
            stable_read_bytes(development_report_path, label="guarded development report")
        )
        != development_report_sha256
    ):
        raise RuntimeError("reviewed development report changed during qualification")
    if (
        checkpoint_path is not None
        and sha256_bytes(stable_read_bytes(checkpoint_path, label="guarded checkpoint"))
        != checkpoint_sha256
    ):
        raise RuntimeError("reviewed checkpoint changed during qualification")
    if (
        development_ledger_path_value is not None
        and sha256_bytes(
            stable_read_bytes(
                development_ledger_path_value, label="guarded development access ledger"
            )
        )
        != development_ledger_sha256
    ):
        raise RuntimeError("reviewed development access ledger changed during qualification")
    if model is not None and _model_state_sha256(model) != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("two-visible public model state changed during execution")


def run_development(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    source_provenance: Mapping[str, Any],
) -> int:
    """Evaluate development only and emit reviewable zero-state evidence."""

    assert_rgbd_two_visible_config(config)
    source = clean_source(source_provenance, label="two-visible RGB-D development")
    _require_config_matches_frozen_path(config, config_path)
    ledger_path = development_ledger_path()
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
            raise FileExistsError(f"two-visible development artifact must be fresh: {path}")
    protocol = bridge_protocol()
    model = new_public_model(config)
    ledger = DevelopmentLedger(
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "source_provenance": source,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "development_manifest_sha256": canonical_sha256(list(DEVELOPMENT_SEEDS)),
        }
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_two_visible_development",
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
        development = evaluate_seed_manifest(
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
        _guard_frozen_inputs(
            source=source,
            config=config,
            config_path=config_path,
            model=model,
        )
        if development["passed"]:
            checkpoint_metrics = {
                "artifact_kind": "rgbd_two_visible_empty_model_state",
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
            checkpoint_contents = stable_read_bytes(checkpoint_path, label="development checkpoint")
            payload = _load_checkpoint_payload(checkpoint_contents)
            validate_checkpoint_evidence(
                payload,
                config=config,
                source=source,
                development=development,
            )
            checkpoint_digest = sha256_bytes(checkpoint_contents)
            report["checkpoint"] = str(checkpoint_path.resolve())
            report["checkpoint_sha256"] = checkpoint_digest
            report["checkpoint_model_state_sha256"] = EMPTY_MODEL_STATE_SHA256
            report["checkpoint_roundtrip_state_sha256"] = _model_state_sha256(
                new_public_model(config)
            )
        _guard_frozen_inputs(
            source=source,
            config=config,
            config_path=config_path,
            model=model,
        )
        write_report_fresh(report_path, report)
        report_digest = sha256_bytes(
            stable_read_bytes(report_path, label="development report after write")
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
                    stable_read_bytes(report_path, label="failed development report")
                )
            except BaseException:
                report_digest = None
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
    source_provenance: Mapping[str, Any],
) -> int:
    """Consume protected splits exactly once after reviewed development."""

    assert_rgbd_two_visible_config(config)
    source = clean_source(source_provenance, label="two-visible RGB-D qualification")
    _require_config_matches_frozen_path(config, config_path)
    ledger_path = qualification_ledger_path()
    reviewed_development_ledger_path = development_ledger_path()
    checkpoint_digest = validated_sha256(
        reviewed_checkpoint_sha256, label="reviewed checkpoint SHA-256"
    )
    report_digest = validated_sha256(
        reviewed_report_sha256, label="reviewed development report SHA-256"
    )
    validate_distinct_paths(
        {
            "config": config_path,
            "report": report_path,
            "checkpoint": checkpoint_path,
            "development_report": development_report_path,
            "development_ledger": reviewed_development_ledger_path,
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
            raise FileExistsError(f"two-visible qualification artifact must be fresh: {path}")
    checkpoint_contents = stable_read_bytes(checkpoint_path, label="reviewed checkpoint")
    if sha256_bytes(checkpoint_contents) != checkpoint_digest:
        raise ValueError("reviewed checkpoint hash does not match bytes")
    report_contents = stable_read_bytes(
        development_report_path, label="reviewed development report"
    )
    if sha256_bytes(report_contents) != report_digest:
        raise ValueError("reviewed development report hash does not match bytes")
    development_report = json.loads(report_contents)
    if not isinstance(development_report, Mapping):
        raise ValueError("reviewed development report must be a JSON object")
    development_ledger_contents = stable_read_bytes(
        reviewed_development_ledger_path, label="reviewed development access ledger"
    )
    development_ledger_digest = sha256_bytes(development_ledger_contents)
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
        report_sha256=report_digest,
        checkpoint_sha256=checkpoint_digest,
        source=source,
        development=development,
    )
    payload = _load_checkpoint_payload(checkpoint_contents)
    validate_checkpoint_evidence(payload, config=config, source=source, development=development)
    model = new_public_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    initial_state = _model_state_sha256(model)
    _guard_frozen_inputs(
        source=source,
        config=config,
        config_path=config_path,
        development_report_path=development_report_path,
        development_report_sha256=report_digest,
        development_ledger_path_value=reviewed_development_ledger_path,
        development_ledger_sha256=development_ledger_digest,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_digest,
        model=model,
    )
    ledger = QualificationLedger(
        {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": source,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "reviewed_checkpoint_sha256": checkpoint_digest,
            "reviewed_development_report_sha256": report_digest,
            "reviewed_development_ledger_sha256": development_ledger_digest,
            "model_state_sha256": initial_state,
        },
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_two_visible_protected_qualification",
        "protocol": bridge_protocol(),
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": checkpoint_digest,
        "reviewed_development_report_sha256": report_digest,
        "reviewed_development_ledger_sha256": development_ledger_digest,
        "initial_model_state_sha256": initial_state,
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
                config=config,
                config_path=config_path,
                development_report_path=development_report_path,
                development_report_sha256=report_digest,
                development_ledger_path_value=reviewed_development_ledger_path,
                development_ledger_sha256=development_ledger_digest,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_digest,
                model=model,
            )
            authorization = ledger.begin_access(split)
            report["protected_data_materialized"] = True
            result = evaluate_seed_manifest(
                config,
                seeds,
                split=split,
                authorization=authorization,
            )
            report[split] = result
            report["stopped_after"] = split
            ledger.complete_split(split, result)
            _guard_frozen_inputs(
                source=source,
                config=config,
                config_path=config_path,
                development_report_path=development_report_path,
                development_report_sha256=report_digest,
                development_ledger_path_value=reviewed_development_ledger_path,
                development_ledger_sha256=development_ledger_digest,
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
            development_report_path=development_report_path,
            development_report_sha256=report_digest,
            development_ledger_path_value=reviewed_development_ledger_path,
            development_ledger_sha256=development_ledger_digest,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_digest,
            model=model,
        )
        write_report_fresh(report_path, report)
        qualification_report_digest = sha256_bytes(
            stable_read_bytes(report_path, label="completed qualification report")
        )
        ledger.finish(report_sha256=qualification_report_digest)
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        error_report_digest = None
        if not _lexists(report_path):
            try:
                write_report_fresh(report_path, report)
                error_report_digest = sha256_bytes(
                    stable_read_bytes(report_path, label="failed qualification report")
                )
            except BaseException:
                error_report_digest = None
        else:
            error_report_digest = sha256_bytes(
                stable_read_bytes(report_path, label="qualification report before ledger error")
            )
        ledger.record_error(
            error,
            stopped_after=str(report["stopped_after"]),
            report_sha256=error_report_digest,
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
    "DevelopmentLedger",
    "EMPTY_MODEL_STATE_SHA256",
    "FINAL_TEST_SEEDS",
    "FROZEN_CONFIG_SHA256",
    "HISTORY_FRAME_INDICES",
    "HORIZONS_SECONDS",
    "OBJECT_INDICES",
    "OPTIMIZER_UPDATES",
    "QualificationLedger",
    "SELECTOR_SEEDS",
    "TARGET_FRAME_INDICES",
    "TwoVisibleRGBDGates",
    "TwoVisibleSceneSpecification",
    "VJP_COEFFICIENTS",
    "VJP_OUTPUTS",
    "assert_rgbd_two_visible_config",
    "bridge_protocol",
    "development_ledger_path",
    "gate_failures",
    "new_public_model",
    "preflight_two_visible_episode",
    "qualification_ledger_path",
    "require_frozen_config",
    "run_development",
    "run_qualification",
    "scene_specification",
    "sha256_file",
    "validate_checkpoint_evidence",
    "validate_development_evidence",
]
