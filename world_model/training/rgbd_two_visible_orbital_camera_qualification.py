"""Frozen qualification for the two-visible orbital-camera-sphere public RGB-D rung.

Protocol inspection, configuration validation, and unit tests are deliberately
seed-free.  Simulator state is materialized only by
:func:`_evaluate_seed_manifest`
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

import contextlib
import hashlib
import io
import json
import math
import os
import resource
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.datasets import collate_episodes
from world_model.dynamics.free_motion_fit import fit_free_motion, free_motion_position_velocity
from world_model.observations import DirectVelocityEvidence
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.runtime.state import runtime_stream_key
from world_model.simulator.camera import (
    CameraFrame,
    camera_to_world,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    world_to_camera,
)
from world_model.simulator.episode import validate_episode
from world_model.simulator.labels import make_perception_labels, validate_perception_labels
from world_model.simulator.physics import PhysicsStepEvents, SphereState, empty_physics_events
from world_model.simulator.renderer import render_spheres
from world_model.simulator.sphere_world import SphereWorld, SphereWorldConfig
from world_model.training.checkpointing import capture_git_metadata, checkpoint_payload
from world_model.training.loop import make_rgbd_packet
from world_model.training.rgbd_online_bridge_qualification import (
    canonical_sha256,
    clean_source,
    sha256_bytes,
    stable_read_bytes,
    validated_sha256,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION, __version__

_NATIVE_PATH_TYPE = type(Path())

DEVELOPMENT_SEEDS = tuple(range(61_000_000, 61_000_032))
SELECTOR_SEEDS = tuple(range(62_000_000, 62_000_024))
CONFIRMATION_SEEDS = tuple(range(63_000_000, 63_000_024))
FINAL_TEST_SEEDS = tuple(range(64_000_000, 64_000_048))

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
ARCHITECTURE_ATTEMPT = 1
MAX_ARCHITECTURE_ATTEMPTS = 1
OPTIMIZER_UPDATES = 0
FROZEN_CONFIG_SHA256 = "a9c348ea54b168ec78780d59d3b3eb066344d3a7551464b9aad1e5b9ac6d6cbd"
EMPTY_MODEL_STATE_SHA256 = canonical_sha256([])

MANIFEST_SHA256 = {
    "development": "eb558805c2974302c33abef4531e142bb60e8f20045d8530330838223a6899a0",
    "selector": "c97fff97459ee9962b972cb7905887c2b2ed6eb5a1837d908f1512ce77e6d97f",
    "confirmation": "b47f03633732fc2986939e71007a0a79b12db2b42f0b5261b4ebd2d0a304f544",
    "final_test": "82927d192b53f2e4af11491f53039c145acfd8e0401a3e2b0b1e974591ee4174",
}

SPLIT_PHYSICAL_PAIRS: dict[str, tuple[tuple[int, int], ...]] = {
    "development": ((-3, -3), (-1, 1), (1, -1), (3, 3)),
    "selector": ((-3, 3), (-1, -1), (3, -3)),
    "confirmation": ((-3, -1), (1, 3), (3, -1)),
    "final_test": ((-3, 1), (-1, -3), (-1, 3), (1, -3), (1, 1), (3, 1)),
}
PHYSICAL_PRIMITIVE_PAIRS = tuple(
    pair
    for split in ("development", "selector", "confirmation", "final_test")
    for pair in SPLIT_PHYSICAL_PAIRS[split]
)
CAMERA_PHASES_RADIANS = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
CAMERA_DIRECTIONS = (-1, 1)
CAMERA_TARGET = (0.0, 0.95, 0.0)
CAMERA_RADIUS_M = 4.6
CAMERA_HEIGHT_M = 2.15
CAMERA_ANGULAR_SPEED_RAD_S = 0.24
CAMERA_VERTICAL_FOV_DEGREES = 48.0
PALETTE = ((0.92, 0.20, 0.14), (0.14, 0.84, 0.30))

FROZEN_CERTIFICATE_SHA256 = "7832ddb49081292d0f50a5eb63edb38fefb49d136d7e1757c73d9c658e42a36f"


def capture_published_source(root: Path) -> dict[str, Any]:
    """Capture local HEAD/upstream equality without network access."""

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


def _validated_published_source(
    value: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    required = {"upstream_ref", "head_commit", "upstream_commit", "ahead", "behind"}
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{label} publication provenance must contain exactly {sorted(required)}")
    result = dict(value)
    upstream_ref = result["upstream_ref"]
    head_commit = result["head_commit"]
    upstream_commit = result["upstream_commit"]
    if not isinstance(upstream_ref, str) or not upstream_ref or upstream_ref == "HEAD":
        raise ValueError(f"{label} requires a configured branch upstream")
    for name, commit in (("head", head_commit), ("upstream", upstream_commit)):
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError(f"{label} {name} commit must be an exact Git SHA")
        try:
            int(commit, 16)
        except ValueError as error:
            raise ValueError(f"{label} {name} commit must be hexadecimal") from error
    if head_commit != source.get("commit") or upstream_commit != head_commit:
        raise ValueError(f"{label} requires upstream commit == clean HEAD")
    if type(result["ahead"]) is not int or type(result["behind"]) is not int:
        raise TypeError(f"{label} ahead/behind counts must be exact integers")
    if result["ahead"] != 0 or result["behind"] != 0:
        raise ValueError(f"{label} requires zero commits ahead and behind upstream")
    return result


@dataclass(frozen=True)
class OrbitalCameraRGBDGates:
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
    physical_trajectory_count: float = 16.0
    camera_appearance_combination_count: float = 128.0
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

    minimum_silhouette_gap_pixels: float = 4.0
    minimum_boundary_clearance_pixels: float = 6.0
    minimum_world_surface_gap_m: float = 1.0
    minimum_world_boundary_clearance_m: float = 0.15
    minimum_visible_fraction: float = 1.0
    minimum_full_support_pixels: float = 20.0
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
    world_from_camera_homogeneous_last_row_gradient_max_abs: float = 0.0

    stale_camera_current_position_rmse_m: float = 0.045
    correct_to_stale_current_position_rmse_ratio: float = 0.25
    stale_camera_current_velocity_rmse_mps: float = 0.050
    correct_to_stale_current_velocity_rmse_ratio: float = 0.25
    stale_camera_horizon_2_00_position_rmse_m: float = 0.080
    correct_to_stale_horizon_2_00_position_rmse_ratio: float = 0.45
    ideal_wls_stale_camera_current_position_rmse_m: float = 0.045
    ideal_wls_stale_camera_velocity_rmse_mps: float = 0.065
    ideal_wls_stale_camera_horizon_2_00_position_rmse_m: float = 0.120
    stale_camera_identity_switch_count: float = 0.0
    stale_camera_association_ambiguous_pair_count: float = 0.0
    stale_camera_history_valid_count: float = 16.0

    minimum_camera_adjacent_angle_radians: float = 0.01198
    maximum_camera_adjacent_angle_radians: float = 0.01202
    minimum_camera_translation_step_m: float = 0.0551
    maximum_camera_translation_step_m: float = 0.0553
    maximum_projected_centre_step_pixels: float = 0.13
    camera_calibration_max_abs_error: float = 2.0e-5
    final_belief_camera_max_abs_error: float = 1.0e-6

    perception_latency_seconds: float = 3.0
    state_only_rollout_latency_seconds: float = 0.075
    persistent_runtime_tensor_state_bytes: int = 65_536
    process_max_rss_bytes: int = 2_500_000_000
    process_rss_delta_bytes: int = 1_000_000_000


DEFAULT_GATES = OrbitalCameraRGBDGates()


@dataclass(frozen=True)
class OrbitalCameraSceneSpecification:
    """One explicit physical primitive crossed with one camera stratum."""

    position: Tensor
    velocity: Tensor
    albedo: Tensor
    palette_swapped: bool
    split: str
    split_primitive_index: int
    physical_index: int
    a: int
    b: int
    camera_stratum: int
    phase_index: int
    direction_index: int
    direction: int
    theta0: float


_MANIFEST_CAPABILITY_AUTHORITY = object()
_LIVE_MANIFEST_CAPABILITIES: dict[int, tuple[object, object]] = {}
_LIVE_PRIVATE_LEDGERS: dict[int, tuple[object, ...]] = {}
_LIVE_RUN_AUTHORIZATIONS: dict[int, tuple[object, ...]] = {}
_LIVE_REVIEWED_DEVELOPMENT_SEALS: dict[int, tuple[object, ...]] = {}


class _RunAuthorization:
    """Unconstructable nominal proof that one run API completed preflight."""

    def __init__(self) -> None:
        raise PermissionError("run authorizations are minted only after exact run preflight")


class _ReviewedDevelopmentSeal:
    """Unconstructable nominal proof of externally reviewed development bytes."""

    def __init__(self) -> None:
        raise PermissionError(
            "reviewed-development seals are minted only after every exact validator passes"
        )


def _mint_run_authorization(
    kind: str,
    bindings: Mapping[str, Any],
    *,
    reviewed_seal: _ReviewedDevelopmentSeal | None = None,
) -> _RunAuthorization:
    if kind not in {"development", "qualification"} or type(bindings) is not dict:
        raise PermissionError("run authorization requires exact preflight bindings")
    if kind == "qualification" and type(reviewed_seal) is not _ReviewedDevelopmentSeal:
        raise PermissionError("qualification authorization requires reviewed development")
    _require_single_thread_execution()
    current_source = clean_source(
        capture_git_metadata(REPOSITORY_ROOT),
        label=f"orbital-camera {kind} authorization",
    )
    current_publication = _validated_published_source(
        capture_published_source(REPOSITORY_ROOT),
        source=current_source,
        label=f"orbital-camera {kind} authorization",
    )
    config_contents = stable_read_bytes(
        _frozen_config_path(),
        label=f"orbital-camera {kind} authorization config",
    )
    if sha256_bytes(config_contents) != FROZEN_CONFIG_SHA256:
        raise PermissionError("run authorization observed non-frozen config bytes")
    if scene_family_certificate()["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256:
        raise PermissionError("run authorization observed a changed scene certificate")
    if kind == "development":
        _validate_run_tree(frozenset(), stage="development authorization")
        expected_keys = {
            "protocol_sha256",
            "source_provenance",
            "publication_provenance",
            "config_sha256",
            "development_manifest_sha256",
            "certificate_sha256",
        }
        expected_bindings = {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": current_source,
            "publication_provenance": current_publication,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "development_manifest_sha256": MANIFEST_SHA256["development"],
            "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        }
    else:
        _validate_run_tree(DEVELOPMENT_ARTIFACT_NAMES, stage="qualification authorization")
        expected_keys = {
            "protocol_sha256",
            "source_provenance",
            "publication_provenance",
            "config_sha256",
            "reviewed_checkpoint_sha256",
            "reviewed_development_report_sha256",
            "reviewed_development_ledger_sha256",
            "model_state_sha256",
            "certificate_sha256",
        }
        seal_registration = _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(id(reviewed_seal))
        if seal_registration != (
            reviewed_seal,
            canonical_sha256(dict(bindings)),
            None,
        ):
            raise PermissionError("qualification authorization lacks a fresh exact review seal")
        if getattr(reviewed_seal, "_bindings", None) != dict(bindings):
            raise PermissionError("qualification authorization review seal is misbound")
        for binding in (
            "reviewed_checkpoint_sha256",
            "reviewed_development_report_sha256",
            "reviewed_development_ledger_sha256",
        ):
            validated_sha256(bindings.get(binding), label=binding)
        expected_bindings = {
            **{
                key: bindings[key]
                for key in (
                    "reviewed_checkpoint_sha256",
                    "reviewed_development_report_sha256",
                    "reviewed_development_ledger_sha256",
                )
            },
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": current_source,
            "publication_provenance": current_publication,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
            "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        }
    if set(bindings) != expected_keys:
        raise PermissionError("run authorization binding schema differs from protocol")
    if dict(bindings) != expected_bindings:
        raise PermissionError("run authorization binding values differ from exact preflight")
    authorization = object.__new__(_RunAuthorization)
    authorization._kind = kind
    authorization._bindings = dict(bindings)
    authorization._reviewed_seal = reviewed_seal
    _LIVE_RUN_AUTHORIZATIONS[id(authorization)] = (
        authorization,
        kind,
        canonical_sha256(authorization._bindings),
        reviewed_seal,
    )
    if kind == "qualification":
        reviewed_seal._authorization = authorization
        _LIVE_REVIEWED_DEVELOPMENT_SEALS[id(reviewed_seal)] = (
            reviewed_seal,
            canonical_sha256(dict(bindings)),
            authorization,
        )
    return authorization


def _mint_reviewed_development_seal(
    bindings: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    ledger_record: Mapping[str, Any],
    checkpoint_payload_value: Mapping[str, Any],
    config: OrpheusConfig,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> _ReviewedDevelopmentSeal:
    if type(bindings) is not dict:
        raise PermissionError("reviewed-development seal requires exact validated bindings")
    expected_keys = {
        "protocol_sha256",
        "source_provenance",
        "publication_provenance",
        "config_sha256",
        "reviewed_checkpoint_sha256",
        "reviewed_development_report_sha256",
        "reviewed_development_ledger_sha256",
        "model_state_sha256",
        "certificate_sha256",
    }
    if set(bindings) != expected_keys:
        raise PermissionError("reviewed-development seal binding schema differs")
    for binding in (
        "reviewed_checkpoint_sha256",
        "reviewed_development_report_sha256",
        "reviewed_development_ledger_sha256",
    ):
        validated_sha256(bindings[binding], label=binding)
    expected_known = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": dict(source),
        "publication_provenance": dict(publication),
        "config_sha256": FROZEN_CONFIG_SHA256,
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    if any(bindings[key] != value for key, value in expected_known.items()):
        raise PermissionError("reviewed-development seal known binding values differ")
    development = validate_development_evidence(
        report,
        checkpoint_sha256=bindings.get("reviewed_checkpoint_sha256"),
        source=source,
        publication=publication,
    )
    validate_development_ledger(
        ledger_record,
        report=report,
        report_sha256=bindings.get("reviewed_development_report_sha256"),
        checkpoint_sha256=bindings.get("reviewed_checkpoint_sha256"),
        source=source,
        publication=publication,
        development=development,
    )
    validate_checkpoint_evidence(
        checkpoint_payload_value,
        config=config,
        source=source,
        publication=publication,
        development=development,
    )
    seal = object.__new__(_ReviewedDevelopmentSeal)
    seal._bindings = dict(bindings)
    _LIVE_REVIEWED_DEVELOPMENT_SEALS[id(seal)] = (
        seal,
        canonical_sha256(seal._bindings),
        None,
    )
    return seal


def _consume_run_authorization(
    authorization: _RunAuthorization,
    *,
    kind: str,
    bindings: Mapping[str, Any],
    reviewed_seal: _ReviewedDevelopmentSeal | None = None,
) -> None:
    if type(authorization) is not _RunAuthorization or type(bindings) is not dict:
        raise PermissionError("private ledger requires an exact run authorization")
    registration = _LIVE_RUN_AUTHORIZATIONS.pop(id(authorization), None)
    expected = (
        authorization,
        kind,
        canonical_sha256(dict(bindings)),
        reviewed_seal,
    )
    if registration != expected:
        raise PermissionError("run authorization is fake, stale, replayed, or misbound")
    if (
        getattr(authorization, "_kind", None) != kind
        or getattr(authorization, "_bindings", None) != dict(bindings)
        or getattr(authorization, "_reviewed_seal", None) is not reviewed_seal
    ):
        raise PermissionError("run authorization state differs from exact preflight")


def _bind_reviewed_development_seal(
    seal: _ReviewedDevelopmentSeal,
    *,
    bindings: Mapping[str, Any],
    ledger: object,
) -> None:
    if type(seal) is not _ReviewedDevelopmentSeal or type(bindings) is not dict:
        raise PermissionError("qualification ledger requires exact reviewed-development seal")
    registration = _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(id(seal))
    authorization = getattr(seal, "_authorization", None)
    expected = (seal, canonical_sha256(dict(bindings)), authorization)
    if (
        type(authorization) is not _RunAuthorization
        or registration != expected
        or getattr(seal, "_bindings", None) != dict(bindings)
    ):
        raise PermissionError("reviewed-development seal is fake, stale, replayed, or misbound")
    _LIVE_REVIEWED_DEVELOPMENT_SEALS[id(seal)] = (
        seal,
        canonical_sha256(dict(bindings)),
        ledger,
    )


class _ManifestCapability:
    """Nominal, ledger-minted, single-use authority for one exact manifest."""

    def __init__(
        self,
        authority: object,
        *,
        ledger: object,
        ledger_mint_identity: object,
        split: str,
        seeds: Sequence[int],
    ) -> None:
        if authority is not _MANIFEST_CAPABILITY_AUTHORITY:
            raise PermissionError("manifest capabilities may only be minted by the live ledger")
        if ledger_mint_identity is not getattr(ledger, "_mint_identity", None):
            raise PermissionError("manifest capability lacks this ledger's mint identity")
        self._ledger = ledger
        self._split = split
        self._seeds = tuple(int(seed) for seed in seeds)
        self._begun = False
        self._finished = False
        self._cursor = 0
        _LIVE_MANIFEST_CAPABILITIES[id(self)] = (self, ledger)

    def begin_manifest(self, split: str, seeds: Sequence[int]) -> None:
        _validate_manifest_capability(self, split=split, seeds=seeds, operation="begin")
        if self._begun or self._finished:
            raise RuntimeError("manifest capability is single use")
        self._begun = True

    def authorize_seed(self, seed: int) -> None:
        _validate_manifest_capability(
            self,
            split=self._split,
            seeds=self._seeds,
            operation="seed",
        )
        if not self._begun or self._finished or self._cursor >= len(self._seeds):
            raise PermissionError("scene construction lacks an active manifest capability")
        if type(seed) is not int or seed != self._seeds[self._cursor]:
            raise PermissionError("scene construction order differs from authorized manifest")
        self._cursor += 1

    def finish_manifest(self) -> None:
        _validate_manifest_capability(
            self,
            split=self._split,
            seeds=self._seeds,
            operation="finish",
        )
        if not self._begun or self._finished or self._cursor != len(self._seeds):
            raise RuntimeError("authorized manifest did not materialize exactly once in order")
        self._finished = True

    def require_finished(self) -> None:
        _validate_manifest_capability(
            self,
            split=self._split,
            seeds=self._seeds,
            operation="complete",
        )
        if not self._finished or self._cursor != len(self._seeds):
            raise RuntimeError("manifest capability was not fully consumed")


def _assert_seed_namespaces() -> None:
    namespaces = (
        DEVELOPMENT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    flattened = [seed for namespace in namespaces for seed in namespace]
    if any(not namespace for namespace in namespaces):
        raise RuntimeError("every two-visible orbital-camera RGB-D namespace must be nonempty")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("two-visible orbital-camera RGB-D namespaces must be disjoint")


def _manifest_for_split(split: str) -> tuple[int, ...]:
    manifests = {
        "development": DEVELOPMENT_SEEDS,
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }
    try:
        return manifests[split]
    except KeyError as error:
        raise ValueError(f"unknown orbital-camera split {split!r}") from error


def _rational_primitive_metadata(a: int, b: int) -> dict[str, object]:
    """Return the literal rational table row from which float32 state is made."""

    return {
        "a": a,
        "b": b,
        "position": (
            ((-450 + 6 * a, 1000), (400 + 2 * b, 1000), (-300 + 4 * b, 1000)),
            ((450 + 5 * b, 1000), (1750 + 2 * a, 1000), (300 - 4 * a, 1000)),
        ),
        "velocity": (
            ((90 + a, 2000), (48 + b, 4000), (16 + a, 4000)),
            ((-86 + b, 2000), (-32 + a, 4000), (-12 + b, 4000)),
        ),
    }


def _float32_primitive(a: int, b: int) -> tuple[Tensor, Tensor]:
    row = _rational_primitive_metadata(a, b)

    def resolve(table: object) -> Tensor:
        if not isinstance(table, tuple):
            raise TypeError("rational primitive table is malformed")
        return torch.tensor(
            [[numerator / denominator for numerator, denominator in vector] for vector in table],
            dtype=torch.float32,
        )

    return resolve(row["position"]), resolve(row["velocity"])


def scene_specification(split: str, ordinal: int) -> OrbitalCameraSceneSpecification:
    """Return one pure table/camera combination without touching a seed or run."""

    manifest = _manifest_for_split(split)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError("orbital-camera scene ordinal must be an integer")
    if not 0 <= ordinal < len(manifest):
        raise IndexError(ordinal)
    split_primitive_index = ordinal // 8
    camera_stratum = ordinal % 8
    phase_index = camera_stratum // 2
    direction_index = camera_stratum % 2
    direction = CAMERA_DIRECTIONS[direction_index]
    a, b = SPLIT_PHYSICAL_PAIRS[split][split_primitive_index]
    position, velocity = _float32_primitive(a, b)
    palette_swapped = bool((split_primitive_index + phase_index + direction_index) % 2)
    albedo = torch.tensor(PALETTE, dtype=torch.float32)
    if palette_swapped:
        albedo = albedo.flip(0)
    return OrbitalCameraSceneSpecification(
        position=position,
        velocity=velocity,
        albedo=albedo,
        palette_swapped=palette_swapped,
        split=split,
        split_primitive_index=split_primitive_index,
        physical_index=PHYSICAL_PRIMITIVE_PAIRS.index((a, b)),
        a=a,
        b=b,
        camera_stratum=camera_stratum,
        phase_index=phase_index,
        direction_index=direction_index,
        direction=direction,
        theta0=CAMERA_PHASES_RADIANS[phase_index],
    )


def _scene_specification_for_seed(split: str, seed: int) -> OrbitalCameraSceneSpecification:
    manifest = _manifest_for_split(split)
    if type(seed) is not int:
        raise TypeError("orbital-camera scene seed must be an integer")
    try:
        ordinal = manifest.index(seed)
    except ValueError as error:
        raise PermissionError("seed is outside the authorized split") from error
    return scene_specification(split, ordinal)


def orbital_camera_frame(
    specification: OrbitalCameraSceneSpecification,
    timestamp: float,
) -> CameraFrame:
    """Evaluate the single qualification-owned float32 extrinsics law."""

    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("camera timestamp must be finite and nonnegative")
    theta = specification.theta0 + specification.direction * CAMERA_ANGULAR_SPEED_RAD_S * timestamp
    position = torch.tensor(
        [
            CAMERA_RADIUS_M * math.sin(theta),
            CAMERA_HEIGHT_M,
            CAMERA_RADIUS_M * math.cos(theta),
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(CAMERA_TARGET, dtype=torch.float32)
    world_from_camera = look_at_world_from_camera(position, target)
    camera_from_world = invert_rigid_transform(world_from_camera)
    intrinsics = make_intrinsics(
        (64, 64),
        CAMERA_VERTICAL_FOV_DEGREES,
        dtype=torch.float32,
    )
    frame = CameraFrame(
        timestamp=timestamp,
        world_from_camera=world_from_camera,
        camera_from_world=camera_from_world,
        intrinsics=intrinsics,
        position=position,
        target=target,
    )
    frame.validate()
    return frame


def _exact_physical_trajectory(
    specification: OrbitalCameraSceneSpecification,
) -> tuple[Tensor, Tensor]:
    """Reproduce the 330 float32 drag substeps used by the simulator."""

    position = specification.position.clone()
    velocity = specification.velocity.clone()
    positions = [position.clone()]
    velocities = [velocity.clone()]
    drag = torch.full((2, 1), 0.05, dtype=torch.float32)
    substep_seconds = 1.0 / 120.0
    for substep_index in range(330):
        coefficient = drag.clamp_min(0.0)
        decay = torch.exp(-coefficient * substep_seconds)
        one_minus_decay = -torch.expm1(-coefficient * substep_seconds)
        safe_coefficient = coefficient.clamp_min(1.0e-5)
        acceleration = torch.zeros((1, 3), dtype=torch.float32)
        position = (
            position
            + velocity * one_minus_decay / safe_coefficient
            + acceleration
            * (substep_seconds / safe_coefficient - one_minus_decay / safe_coefficient.square())
        )
        velocity = velocity * decay + acceleration * one_minus_decay / safe_coefficient
        if (substep_index + 1) % 6 == 0:
            positions.append(position.clone())
            velocities.append(velocity.clone())
    if len(positions) != 56 or len(velocities) != 56:
        raise AssertionError("orbital-camera recurrence must emit exactly 56 frames")
    return torch.stack(positions), torch.stack(velocities)


def _certificate_state(
    specification: OrbitalCameraSceneSpecification,
    position: Tensor,
    velocity: Tensor,
) -> SphereState:
    state = SphereState(
        object_id=torch.tensor([0, 1], dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=position.clone(),
        velocity=velocity.clone(),
        radius=torch.full((2, 1), 0.21, dtype=torch.float32),
        mass=torch.ones((2, 1), dtype=torch.float32),
        restitution=torch.full((2, 1), 0.7, dtype=torch.float32),
        drag=torch.full((2, 1), 0.05, dtype=torch.float32),
        friction=torch.full((2, 1), 0.2, dtype=torch.float32),
        albedo=specification.albedo.clone(),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).expand(2, -1).clone(),
        angular_velocity=torch.zeros((2, 3), dtype=torch.float32),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    state.validate()
    return state


def _independent_raster_trace(
    state: SphereState,
    camera: CameraFrame,
) -> dict[str, Tensor]:
    """Independent exact ray/discriminant/stable-near-root visibility trace."""

    height = width = 64
    points_camera = world_to_camera(state.position, camera.camera_from_world)
    depth = points_camera[:, 2]
    focal = 0.5 * (camera.intrinsics[0, 0] + camera.intrinsics[1, 1])
    centres = torch.stack(
        (
            focal * points_camera[:, 0] / depth + camera.intrinsics[0, 2],
            focal * points_camera[:, 1] / depth + camera.intrinsics[1, 2],
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
    discriminant = ray_norm_squared.unsqueeze(0) * state.radius[:, None, None, 0].square()
    discriminant = discriminant - center_cross_ray.square().sum(dim=-1)
    square_root = discriminant.clamp_min(0.0).sqrt()
    denominator = ray_dot_center + square_root
    constant = points_camera.square().sum(dim=-1)[:, None, None]
    constant = constant - state.radius[:, None, None, 0].square()
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
    winner = torch.where(has_object, winner, torch.full_like(winner, -1))
    visible_mask = full_mask & (winner.unsqueeze(0) == torch.arange(2)[:, None, None])
    support = full_mask.sum(dim=(-2, -1))
    visible = visible_mask.sum(dim=(-2, -1))
    return {
        "points_camera": points_camera,
        "centres": centres,
        "apparent_radius": apparent_radius,
        "discriminant": discriminant,
        "surface_depth": surface_depth,
        "full_mask": full_mask,
        "winner": winner,
        "depth_buffer": torch.where(has_object, depth_buffer, torch.zeros_like(depth_buffer)),
        "support": support,
        "visible": visible,
    }


def _update_tensor_digest(digest: Any, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _ideal_stale_camera_metrics(
    specification: OrbitalCameraSceneSpecification,
    positions: Tensor,
    velocities: Tensor,
) -> dict[str, float]:
    """Exact uniform-WLS sensitivity control with frame-zero WFC held stale."""

    stale_world_from = orbital_camera_frame(specification, 0.0).world_from_camera
    reconstructed: list[Tensor] = []
    for frame_index in HISTORY_FRAME_INDICES:
        camera = orbital_camera_frame(specification, frame_index / 20.0)
        camera_position = world_to_camera(positions[frame_index], camera.camera_from_world)
        reconstructed.append(camera_to_world(camera_position, stale_world_from))
    history = torch.stack(reconstructed).unsqueeze(0)
    timestamps = (torch.arange(16, dtype=torch.float32) / 20.0).unsqueeze(0)
    gravity = torch.zeros(3, dtype=torch.float32)
    drag = torch.full((1, 2), 0.05, dtype=torch.float32)
    fit = fit_free_motion(
        history,
        timestamps,
        gravity=gravity,
        drag=drag,
        anchor_time=0.75,
        support=None,
        weights=None,
        minimum_support=16,
        conditioning_limit=100.0,
    )
    if not bool(fit.valid.all()):
        raise RuntimeError("ideal stale-camera WLS certificate became invalid")
    future_position, _ = free_motion_position_velocity(
        fit.position,
        fit.velocity,
        2.0,
        gravity=gravity,
        drag=drag,
    )
    return {
        "current_position_rmse_m": float(
            (fit.position[0] - positions[ANCHOR_FRAME_INDEX]).square().mean().sqrt()
        ),
        "current_velocity_rmse_mps": float(
            (fit.velocity[0] - velocities[ANCHOR_FRAME_INDEX]).square().mean().sqrt()
        ),
        "horizon_2_00_position_rmse_m": float(
            (future_position[0] - positions[TARGET_FRAME_INDICES[-1]]).square().mean().sqrt()
        ),
    }


@lru_cache(maxsize=2)
def scene_family_certificate(*, verify_public_renderer: bool = False) -> dict[str, Any]:
    """Audit all 16 physical trajectories x 8 camera strata without seeds/runs."""

    metadata_table = [
        {"physical_index": index, **_rational_primitive_metadata(a, b)}
        for index, (a, b) in enumerate(PHYSICAL_PRIMITIVE_PAIRS)
    ]
    metadata_sha = canonical_sha256(metadata_table)
    state_digest = hashlib.sha256()
    geometry_digests: set[str] = set()
    combined_digest = hashlib.sha256()
    camera_digest = hashlib.sha256()
    raster_digest = hashlib.sha256()
    lifecycle_digest = hashlib.sha256()
    split_signatures: dict[str, list[str]] = {split: [] for split in SPLIT_PHYSICAL_PAIRS}
    camera_histogram: dict[str, int] = {}
    palette_histogram = {False: 0, True: 0}
    minimum_support = math.inf
    minimum_gap = math.inf
    minimum_boundary = math.inf
    minimum_surface_gap = math.inf
    minimum_world_boundary = math.inf
    minimum_speed = math.inf
    maximum_speed = 0.0
    minimum_discriminant = math.inf
    minimum_adjacent_angle = math.inf
    maximum_adjacent_angle = 0.0
    minimum_translation = math.inf
    maximum_translation = 0.0
    maximum_projected_centre_step = 0.0
    maximum_inverse_error = 0.0
    maximum_orthonormality_error = 0.0
    maximum_radius_error = 0.0
    maximum_height_error = 0.0
    maximum_target_error = 0.0
    maximum_intrinsics_error = 0.0
    maximum_camera_position_binding_error = 0.0
    public_renderer_mismatches = 0
    ideal_stale_values: dict[str, list[float]] = {
        "current_position_rmse_m": [],
        "current_velocity_rmse_mps": [],
        "horizon_2_00_position_rmse_m": [],
    }
    ideal_stale_by_stratum: dict[int, dict[str, list[float]]] = {
        stratum: {name: [] for name in ideal_stale_values} for stratum in range(8)
    }
    ideal_stale_by_split: dict[str, dict[str, list[float]]] = {
        split: {name: [] for name in ideal_stale_values} for split in SPLIT_PHYSICAL_PAIRS
    }
    ideal_stale_by_split_stratum: dict[str, dict[int, dict[str, list[float]]]] = {
        split: {stratum: {name: [] for name in ideal_stale_values} for stratum in range(8)}
        for split in SPLIT_PHYSICAL_PAIRS
    }
    physical_trajectories: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
    frozen_target = torch.tensor(CAMERA_TARGET, dtype=torch.float32)
    frozen_intrinsics = make_intrinsics((64, 64), CAMERA_VERTICAL_FOV_DEGREES, dtype=torch.float32)

    for split, manifest in (
        ("development", DEVELOPMENT_SEEDS),
        ("selector", SELECTOR_SEEDS),
        ("confirmation", CONFIRMATION_SEEDS),
        ("final_test", FINAL_TEST_SEEDS),
    ):
        for ordinal in range(len(manifest)):
            specification = scene_specification(split, ordinal)
            physical_key = (specification.a, specification.b)
            if physical_key not in physical_trajectories:
                physical_trajectories[physical_key] = _exact_physical_trajectory(specification)
                physical_digest = hashlib.sha256()
                _update_tensor_digest(physical_digest, physical_trajectories[physical_key][0])
                _update_tensor_digest(physical_digest, physical_trajectories[physical_key][1])
                geometry_digests.add(physical_digest.hexdigest())
            positions, velocities = physical_trajectories[physical_key]
            ideal_stale = _ideal_stale_camera_metrics(specification, positions, velocities)
            for name, value in ideal_stale.items():
                ideal_stale_values[name].append(value)
                ideal_stale_by_stratum[specification.camera_stratum][name].append(value)
                ideal_stale_by_split[split][name].append(value)
                ideal_stale_by_split_stratum[split][specification.camera_stratum][name].append(
                    value
                )
            if specification.camera_stratum == 0:
                _update_tensor_digest(state_digest, positions)
                _update_tensor_digest(state_digest, velocities)
            palette_histogram[specification.palette_swapped] += 1
            camera_key = f"phase_{specification.phase_index}/direction_{specification.direction:+d}"
            camera_histogram[camera_key] = camera_histogram.get(camera_key, 0) + 1
            joint = hashlib.sha256()
            joint.update(
                json.dumps(
                    {
                        "split": split,
                        "ordinal": ordinal,
                        "physical_index": specification.physical_index,
                        "a": specification.a,
                        "b": specification.b,
                        "camera_stratum": specification.camera_stratum,
                        "phase_index": specification.phase_index,
                        "direction": specification.direction,
                        "palette_swapped": specification.palette_swapped,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            expected_created = torch.zeros((56, 2), dtype=torch.bool)
            expected_created[0] = True
            expected_removed = torch.zeros((56, 2), dtype=torch.bool)
            expected_physical_events = torch.zeros((56, 2), dtype=torch.bool)
            for expected in (expected_created, expected_removed, expected_physical_events):
                _update_tensor_digest(lifecycle_digest, expected)
            previous_camera: CameraFrame | None = None
            previous_centres: Tensor | None = None
            for frame_index in range(56):
                timestamp = frame_index / 20.0
                camera = orbital_camera_frame(specification, timestamp)
                state = _certificate_state(
                    specification,
                    positions[frame_index],
                    velocities[frame_index],
                )
                trace = _independent_raster_trace(state, camera)
                if bool((trace["full_mask"][0] & trace["full_mask"][1]).any()):
                    raise RuntimeError("certificate found overlapping silhouettes")
                if not bool(trace["visible"].eq(trace["support"]).all()):
                    raise RuntimeError("certificate found an occluded sphere")
                if bool((trace["support"] < 20).any()):
                    raise RuntimeError("certificate found insufficient raster support")
                centres = trace["centres"]
                radii = trace["apparent_radius"]
                gap = torch.linalg.vector_norm(centres[0] - centres[1]) - radii.sum()
                boundary = torch.stack(
                    (
                        centres[:, 0] - radii,
                        63.0 - centres[:, 0] - radii,
                        centres[:, 1] - radii,
                        63.0 - centres[:, 1] - radii,
                    )
                )
                surface_gap = torch.linalg.vector_norm(state.position[0] - state.position[1]) - 0.42
                bounds = torch.tensor(
                    ((-2.25, 2.25), (0.0, 3.25), (-1.5, 1.5)),
                    dtype=torch.float32,
                )
                world_boundary = torch.minimum(
                    state.position - 0.21 - bounds[:, 0],
                    bounds[:, 1] - state.position - 0.21,
                )
                speeds = torch.linalg.vector_norm(state.velocity, dim=-1)
                if float(gap) < 4.0 or float(boundary.min()) < 6.0:
                    raise RuntimeError("certificate found an image geometry outside the rung")
                if float(surface_gap) < 1.0 or float(world_boundary.min()) < 0.15:
                    raise RuntimeError("certificate found a world geometry outside the rung")
                if float(speeds.min()) < 0.035 or float(speeds.max()) > 0.065:
                    raise RuntimeError("certificate found a speed outside the frozen range")
                minimum_support = min(minimum_support, float(trace["support"].min()))
                minimum_gap = min(minimum_gap, float(gap))
                minimum_boundary = min(minimum_boundary, float(boundary.min()))
                minimum_surface_gap = min(minimum_surface_gap, float(surface_gap))
                minimum_world_boundary = min(minimum_world_boundary, float(world_boundary.min()))
                minimum_speed = min(minimum_speed, float(speeds.min()))
                maximum_speed = max(maximum_speed, float(speeds.max()))
                hits = trace["discriminant"][trace["full_mask"]]
                minimum_discriminant = min(minimum_discriminant, float(hits.min()))
                identity = camera.world_from_camera @ camera.camera_from_world
                rotation = camera.world_from_camera[:3, :3]
                maximum_inverse_error = max(
                    maximum_inverse_error,
                    float((identity - torch.eye(4)).abs().max()),
                )
                maximum_orthonormality_error = max(
                    maximum_orthonormality_error,
                    float((rotation.T @ rotation - torch.eye(3)).abs().max()),
                )
                maximum_radius_error = max(
                    maximum_radius_error,
                    abs(float(torch.linalg.vector_norm(camera.position[[0, 2]])) - CAMERA_RADIUS_M),
                )
                maximum_height_error = max(
                    maximum_height_error,
                    abs(float(camera.position[1]) - CAMERA_HEIGHT_M),
                )
                maximum_target_error = max(
                    maximum_target_error,
                    float((camera.target - frozen_target).abs().max()),
                )
                maximum_intrinsics_error = max(
                    maximum_intrinsics_error,
                    float((camera.intrinsics - frozen_intrinsics).abs().max()),
                )
                maximum_camera_position_binding_error = max(
                    maximum_camera_position_binding_error,
                    float((camera.world_from_camera[:3, 3] - camera.position).abs().max()),
                )
                if previous_camera is not None:
                    previous_relative = previous_camera.position[[0, 2]]
                    current_relative = camera.position[[0, 2]]
                    cosine = torch.dot(previous_relative, current_relative) / (
                        torch.linalg.vector_norm(previous_relative)
                        * torch.linalg.vector_norm(current_relative)
                    )
                    angle = float(torch.acos(cosine.clamp(-1.0, 1.0)))
                    translation = float(
                        torch.linalg.vector_norm(camera.position - previous_camera.position)
                    )
                    minimum_adjacent_angle = min(minimum_adjacent_angle, angle)
                    maximum_adjacent_angle = max(maximum_adjacent_angle, angle)
                    minimum_translation = min(minimum_translation, translation)
                    maximum_translation = max(maximum_translation, translation)
                if previous_centres is not None:
                    maximum_projected_centre_step = max(
                        maximum_projected_centre_step,
                        float(torch.linalg.vector_norm(centres - previous_centres, dim=-1).max()),
                    )
                previous_camera = camera
                previous_centres = centres
                for digest in (joint, combined_digest):
                    _update_tensor_digest(digest, state.position)
                    _update_tensor_digest(digest, state.velocity)
                    _update_tensor_digest(digest, state.albedo)
                    _update_tensor_digest(digest, camera.world_from_camera)
                    _update_tensor_digest(digest, camera.camera_from_world)
                    _update_tensor_digest(digest, camera.intrinsics)
                _update_tensor_digest(camera_digest, camera.world_from_camera)
                _update_tensor_digest(camera_digest, camera.camera_from_world)
                _update_tensor_digest(camera_digest, camera.intrinsics)
                _update_tensor_digest(raster_digest, trace["full_mask"].to(torch.uint8))
                _update_tensor_digest(raster_digest, trace["winner"])
                _update_tensor_digest(raster_digest, trace["depth_buffer"])
                if verify_public_renderer:
                    rendered = render_spheres(state, camera, (64, 64), noise_std=0.0)
                    mismatch = not torch.equal(rendered.full_mask, trace["full_mask"])
                    mismatch |= not torch.equal(rendered.instance_slot_map, trace["winner"])
                    mismatch |= not torch.equal(rendered.depth_buffer, trace["depth_buffer"])
                    public_renderer_mismatches += int(mismatch)
            split_signatures[split].append(joint.hexdigest())

    if len(physical_trajectories) != 16 or len(geometry_digests) != 16:
        raise RuntimeError("certificate requires exactly 16 distinct physical trajectories")
    if public_renderer_mismatches:
        raise RuntimeError("independent raster trace differs from the public renderer")
    camera_calibration_error = max(
        maximum_inverse_error,
        maximum_orthonormality_error,
        maximum_radius_error,
        maximum_height_error,
        maximum_target_error,
        maximum_intrinsics_error,
        maximum_camera_position_binding_error,
    )
    if camera_calibration_error > DEFAULT_GATES.camera_calibration_max_abs_error:
        raise RuntimeError("certificate found calibration outside the exact orbital camera law")
    unsigned: dict[str, Any] = {
        "rational_metadata_table_sha256": metadata_sha,
        "ordered_float32_physical_states_sha256": state_digest.hexdigest(),
        "unordered_physical_geometry_set_sha256": canonical_sha256(sorted(geometry_digests)),
        "ordered_combined_scene_trace_sha256": combined_digest.hexdigest(),
        "camera_transform_trace_sha256": camera_digest.hexdigest(),
        "independent_raster_trace_sha256": raster_digest.hexdigest(),
        "event_lifecycle_trace_sha256": lifecycle_digest.hexdigest(),
        "split_scene_signature_sha256": {
            split: canonical_sha256(signatures) for split, signatures in split_signatures.items()
        },
        "physical_trajectory_count": len(physical_trajectories),
        "camera_appearance_combination_count": sum(map(len, split_signatures.values())),
        "frame_count": 56,
        "physical_substep_count": 330,
        "physical_event_count": 0,
        "lifecycle_frame_zero_birth_count": 256,
        "lifecycle_non_frame_zero_birth_count": 0,
        "lifecycle_removal_count": 0,
        "camera_stratum_histogram": camera_histogram,
        "palette_swap_histogram": {
            str(key).lower(): value for key, value in palette_histogram.items()
        },
        "minimum_full_support_pixels": minimum_support,
        "minimum_continuous_silhouette_gap_pixels": minimum_gap,
        "minimum_image_boundary_clearance_pixels": minimum_boundary,
        "minimum_world_surface_gap_m": minimum_surface_gap,
        "minimum_world_boundary_clearance_m": minimum_world_boundary,
        "minimum_speed_mps": minimum_speed,
        "maximum_speed_mps": maximum_speed,
        "minimum_hit_discriminant": minimum_discriminant,
        "minimum_adjacent_camera_angle_radians": minimum_adjacent_angle,
        "maximum_adjacent_camera_angle_radians": maximum_adjacent_angle,
        "minimum_adjacent_camera_translation_m": minimum_translation,
        "maximum_adjacent_camera_translation_m": maximum_translation,
        "maximum_projected_centre_step_pixels": maximum_projected_centre_step,
        "maximum_camera_inverse_error": maximum_inverse_error,
        "maximum_camera_orthonormality_error": maximum_orthonormality_error,
        "maximum_camera_radius_error_m": maximum_radius_error,
        "maximum_camera_height_error_m": maximum_height_error,
        "maximum_camera_target_error_m": maximum_target_error,
        "maximum_camera_intrinsics_error": maximum_intrinsics_error,
        "maximum_camera_position_binding_error_m": maximum_camera_position_binding_error,
        "ideal_stale_camera_control": {
            "pooled_rmse": {
                name: math.sqrt(sum(value * value for value in values) / len(values))
                for name, values in ideal_stale_values.items()
            },
            "minimum_joint_rmse": {
                name: min(values) for name, values in ideal_stale_values.items()
            },
            "per_camera_stratum_minimum_joint_rmse": {
                str(stratum): {name: min(values) for name, values in strata_values.items()}
                for stratum, strata_values in ideal_stale_by_stratum.items()
            },
            "per_split_pooled_rmse": {
                split: {
                    name: math.sqrt(sum(value * value for value in values) / len(values))
                    for name, values in split_values.items()
                }
                for split, split_values in ideal_stale_by_split.items()
            },
            "per_split_camera_stratum_minimum_joint_rmse": {
                split: {
                    str(stratum): {name: min(values) for name, values in stratum_values.items()}
                    for stratum, stratum_values in split_strata.items()
                }
                for split, split_strata in ideal_stale_by_split_stratum.items()
            },
            "fit": "public_uniform_exact_free_motion_wls_16_frames",
        },
        "public_renderer_equivalence": "required_by_seed_free_test",
    }
    return {**unsigned, "certificate_sha256": canonical_sha256(unsigned)}


def bridge_protocol() -> dict[str, Any]:
    """Return the canonical, self-hashed qualification contract."""

    _assert_seed_namespaces()
    protocol: dict[str, Any] = {
        "name": "rgbd_two_visible_orbital_camera_bridge_v1",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
        "terminal_after_attempt": True,
        "optimizer": None,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "source_binding": {
            "commit": "captured_at_execution_from_eventual_clean_harness_source",
            "dirty": False,
            "upstream_commit_must_equal_head": True,
            "ahead_commits": 0,
            "behind_commits": 0,
            "worktree_and_runtime_fingerprints_required": True,
            "development_checkpoint_report_and_protected_ledger_must_match": True,
        },
        "manifests": {
            split: {
                "seeds": list(_manifest_for_split(split)),
                "sha256": MANIFEST_SHA256[split],
            }
            for split in ("development", "selector", "confirmation", "final_test")
        },
        "scene_family": {
            "constructor": (
                "world_model.training.rgbd_two_visible_orbital_camera_qualification."
                "_construct_two_visible_orbital_camera_episode"
            ),
            "physical_primitive_pairs_by_split": {
                split: [list(pair) for pair in pairs]
                for split, pairs in SPLIT_PHYSICAL_PAIRS.items()
            },
            "physical_formula": {
                "p0": ["-.450+.006*a", ".400+.002*b", "-.300+.004*b"],
                "p1": [".450+.005*b", "1.750+.002*a", ".300-.004*a"],
                "v0": [".045+.0005*a", ".012+.00025*b", ".004+.00025*a"],
                "v1": ["-.043+.0005*b", "-.008+.00025*a", "-.003+.00025*b"],
                "construction_dtype": "torch.float32",
            },
            "physical_trajectory_count": 16,
            "camera_appearance_combination_count": 128,
            "physical_geometry_count_claim": 16,
            "joint_mapping": {
                "primitive_index": "split_offset//8",
                "camera_stratum": "split_offset%8",
                "phase_index": "camera_stratum//2",
                "direction_index": "camera_stratum%2",
                "direction_by_index": list(CAMERA_DIRECTIONS),
                "palette_swap": "(primitive_index+phase_index+direction_index)%2==1",
                "palette": [list(colour) for colour in PALETTE],
            },
            "object_count": 2,
            "world_radius_m": 0.21,
            "linear_drag": 0.05,
            "gravity": [0.0, 0.0, 0.0],
            "initial_speed_norm_bounds_mps": [0.035, 0.065],
            "frame_rate_hz": 20,
            "fully_visible": True,
            "image_separated": True,
            "non_contact": True,
            "balanced_palette_swap_per_split": True,
            "preflight_before_return": True,
            "generic_ensure_collision_false_is_not_evidence": True,
        },
        "camera": {
            "ownership": "qualification_exact_known_extrinsics_not_generic_seeded_camera",
            "theta_law": "theta(t)=theta0+direction*0.24*t",
            "theta0_by_phase_index": ["0", "pi/2", "pi", "3pi/2"],
            "position_law": ["4.6*sin(theta)", "2.15", "4.6*cos(theta)"],
            "target": list(CAMERA_TARGET),
            "vertical_fov_degrees": CAMERA_VERTICAL_FOV_DEGREES,
            "image_size": [64, 64],
            "world_from_camera": "look_at_world_from_camera(position,target)",
            "camera_from_world": "invert_rigid_transform(world_from_camera)",
            "frame_count": 56,
            "frame_rate_hz": 20,
            "calibration_known": True,
        },
        "certificate": {
            "sha256": FROZEN_CERTIFICATE_SHA256,
            "seed_manifest_and_run_materialization": False,
            "physical_substeps": 330,
            "joint_scenes": 128,
            "frames_per_joint_scene": 56,
            "independent_ray_discriminant_near_root_trace": True,
            "public_renderer_equivalence_required": True,
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
            "inputs": [
                "rgb",
                "depth",
                "timestamp",
                "world_from_camera",
                "intrinsics",
                "image_metadata",
            ],
            "excluded_inputs": [
                "camera_position",
                "camera_target",
                "camera_linear_velocity",
                "camera_angular_velocity",
                "camera_from_world",
                "truth_objects",
                "truth_labels",
                "events",
            ],
            "final_belief_camera_owns_frame_15_calibration": True,
        },
        "perception": {
            "kind": "differentiable_symmetric_two_slot_RGB_D_geometry",
            "appearance_dim": 3,
            "chromatic_temperature": 0.05,
            "chromatic_centre_blend": 0.0025,
            "spatial_temperature_pixels": 1.0,
            "scene_qualification_minimum_silhouette_gap_pixels": 4.0,
            "scene_qualification_minimum_boundary_clearance_pixels": 6.0,
            "runtime_module_minimum_silhouette_gap_pixels": 2.0,
            "runtime_module_minimum_boundary_clearance_pixels": 2.0,
            "maximum_surface_radius_relative_error": 0.05,
        },
        "differentiability": {
            "kind": "per_object_fixed_output_vector_jacobian_products",
            "coefficients": list(VJP_COEFFICIENTS),
            "coefficient_reduction": "(output * coefficients).mean()",
            "inputs": ["rgb", "depth", "world_from_camera"],
            "intrinsics": "diagnostic_only_not_a_qualification_floor",
            "world_from_camera_gradient": "ambient_matrix_vjp_not_se3_or_pose_estimation",
            "homogeneous_last_row_gradient": "exact_zero",
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
        "stale_camera_negative_control": {
            "model": "fresh_public_runtime",
            "unchanged": ["rgb", "depth", "timestamps", "intrinsics", "truth", "order"],
            "mutation": "world_from_camera_frames_1_through_15_equal_frame_0",
            "identity_and_history_must_remain_valid": True,
            "pooled_and_per_camera_stratum_gates": True,
            "ideal_exact_wls_control": True,
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


def assert_rgbd_two_visible_orbital_camera_config(config: OrpheusConfig) -> None:
    """Reject every semantic change to the frozen two-visible orbital-camera profile."""

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
        "camera_motion": "orbit",
        "known_camera_pose": True,
        "render_noise_std": 0.0,
        "ensure_collision": False,
        "external_impulse_probability": 0.0,
        "scenario_mixture": ("baseline",),
    }
    for name, required in simulator_expected.items():
        actual = getattr(config.simulator, name)
        if actual != required:
            raise ValueError(
                f"two-visible orbital-camera RGB-D requires simulator.{name}={required!r}"
            )
    if config.project.seed != DEVELOPMENT_SEEDS[0] or not config.project.deterministic:
        raise ValueError(
            "two-visible orbital-camera RGB-D project seed/determinism differs from protocol"
        )
    if config.device.preference != "cpu" or config.device.cuda_amp or config.device.compile:
        raise ValueError("two-visible orbital-camera RGB-D requires CPU float32 without compile")
    if config.model.max_objects != 2 or config.model.state.appearance_dim != 3:
        raise ValueError(
            "two-visible orbital-camera RGB-D requires exactly two slots and appearance_dim three"
        )
    if (
        config.model.lifecycle.birth_confidence != 0.5
        or config.model.lifecycle.birth_confirmations != 1
    ):
        raise ValueError(
            "two-visible orbital-camera RGB-D requires exact lifecycle birth semantics"
        )
    if config.model.rgb.enabled or not config.model.rgbd.enabled:
        raise ValueError("two-visible orbital-camera RGB-D requires only the composite RGB-D path")
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
            raise ValueError(
                f"two-visible orbital-camera RGB-D requires model.rgbd.{name}={required!r}"
            )
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
            raise ValueError(
                f"two-visible orbital-camera RGB-D requires model.association.{name}={required!r}"
            )
    if config.model.association.ambiguity_margin <= 0.0:
        raise ValueError("two-visible orbital-camera RGB-D requires a positive ambiguity margin")
    if not config.model.dynamics.analytic_free_motion_only:
        raise ValueError(
            "two-visible orbital-camera RGB-D requires analytic-free-motion-only dynamics"
        )
    if (
        config.model.dynamics.attention_residual_enabled
        or config.model.dynamics.max_substep != 1.0 / 120.0
    ):
        raise ValueError("two-visible orbital-camera RGB-D forbids learned dynamics residuals")
    if (
        config.model.filter.enable_learned_corrector
        or config.model.filter.learned_residual_scale != 0.0
        or not config.model.filter.direct_metric_position_update
        or not config.model.filter.innovation_anchored_correction
    ):
        raise ValueError(
            "two-visible orbital-camera RGB-D requires only direct metric position correction"
        )
    if config.model.identification.enabled:
        raise ValueError("two-visible orbital-camera RGB-D forbids online parameter identification")
    if (
        config.runtime.modality != "rgbd"
        or tuple(config.runtime.modality_order) != ("debug_oracle", "rgbd")
        or config.runtime.enable_debug_oracle
    ):
        raise ValueError(
            "two-visible orbital-camera RGB-D forbids oracle/runtime modality substitution"
        )
    if config.runtime.hypothesis_pool_enabled or not config.runtime.strict_timestamps:
        raise ValueError(
            "two-visible orbital-camera RGB-D requires strict single-hypothesis execution"
        )
    if config.training.batch_size != 4 or config.training.steps != 1:
        raise ValueError(
            "two-visible orbital-camera shared config requires batch four and one schema step"
        )
    if config.training.rgb_pretrain_steps != 0:
        raise ValueError("two-visible orbital-camera RGB-D has no RGB pretraining phase")
    if config.training.validation_episodes != len(DEVELOPMENT_SEEDS):
        raise ValueError("validation_episodes must match development manifest")
    if tuple(config.evaluation.horizons_seconds) != HORIZONS_SECONDS:
        raise ValueError("two-visible orbital-camera RGB-D horizons differ from protocol")
    if config.evaluation.rgb_only:
        raise ValueError("two-visible orbital-camera RGB-D evaluation requires RGB and depth")
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
        raise ValueError(
            "two-visible orbital-camera target frame indices differ from declared horizons"
        )


def new_public_model(config: OrpheusConfig) -> OnlineWorldModel:
    """Construct the only admitted public runtime and prove zero module state."""

    assert_rgbd_two_visible_orbital_camera_config(config)
    model = OnlineWorldModel.from_config(config, device="cpu")
    if model.belief_factory.initial_radius != 0.21 or model.belief_factory.initial_drag != 0.05:
        raise RuntimeError("public runtime did not receive the frozen radius/drag priors")
    if tuple(model.parameters()) or tuple(model.buffers()) or model.state_dict():
        raise RuntimeError(
            "two-visible orbital-camera public runtime must own zero parameter/buffer state"
        )
    return model


def _stack_records(records: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot stack empty episode records")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise RuntimeError(
            "two-visible orbital-camera episode record schema changed between frames"
        )
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


def _install_scene(world: SphereWorld, specification: OrbitalCameraSceneSpecification) -> None:
    state = world.state
    if state.max_objects != 2:
        raise ValueError(
            "two-visible orbital-camera constructor requires exactly two simulator slots"
        )
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


def _construct_two_visible_orbital_camera_episode(
    config: OrpheusConfig,
    seed: int,
    *,
    split: str,
    capability: _ManifestCapability,
) -> dict[str, Any]:
    """Construct and preflight one exact scene after manifest authorization."""

    if type(capability) is not _ManifestCapability:
        raise PermissionError("constructor requires an exact nominal manifest capability")
    capability.authorize_seed(seed)
    _require_config_matches_frozen_path(config, _frozen_config_path())
    resolved = SphereWorldConfig.from_config(config)
    world = SphereWorld(resolved, seed)
    specification = _scene_specification_for_seed(split, seed)
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
        camera = orbital_camera_frame(specification, timestamp)
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
            "scenario": "two_visible_orbital_camera_free_motion",
            "camera_trajectory": "qualification_owned_exact_orbit",
            "camera_calibration_owner": "qualification_known_extrinsics",
            "frame_rate": resolved.frame_rate,
            "physics_rate": resolved.physics_rate,
            "palette_swapped": specification.palette_swapped,
            "split": specification.split,
            "physical_index": specification.physical_index,
            "split_primitive_index": specification.split_primitive_index,
            "primitive_a": specification.a,
            "primitive_b": specification.b,
            "camera_stratum": specification.camera_stratum,
            "camera_phase_index": specification.phase_index,
            "camera_direction_index": specification.direction_index,
            "camera_direction": specification.direction,
        },
    }
    validate_episode(episode, resolved)
    preflight_two_visible_orbital_camera_episode(
        episode, config=config, specification=specification
    )
    return episode


def preflight_two_visible_orbital_camera_episode(
    episode: Mapping[str, Any],
    *,
    config: OrpheusConfig,
    specification: OrbitalCameraSceneSpecification | None = None,
) -> dict[str, float]:
    """Fail closed unless every frame belongs to the frozen observable family."""

    _require_config_matches_frozen_path(config, _frozen_config_path())
    if specification is None:
        metadata = episode.get("metadata")
        if not isinstance(metadata, Mapping):
            raise RuntimeError("orbital-camera preflight requires exact scene metadata")
        split = metadata.get("split")
        primitive_index = metadata.get("split_primitive_index")
        camera_stratum = metadata.get("camera_stratum")
        if (
            type(split) is not str
            or type(primitive_index) is not int
            or type(camera_stratum) is not int
        ):
            raise RuntimeError("orbital-camera scene metadata is malformed")
        specification = scene_specification(split, primitive_index * 8 + camera_stratum)
    objects = episode["objects"]
    labels = episode["labels"]
    events = episode["events"]
    active = objects["active"][:, :2]
    projected = labels["projected_valid"][:, :2]
    visible = objects["visible_fraction"][:, :2]
    if not bool(active.all()) or not bool(projected.all()):
        raise RuntimeError(
            "two-visible orbital-camera preflight requires both spheres active and projectable"
        )
    if not bool(visible.eq(1.0).all()):
        raise RuntimeError(
            "two-visible orbital-camera preflight requires exact full visibility in every frame"
        )
    full_mask = labels["full_mask"][:, :2]
    if bool((full_mask[:, 0] & full_mask[:, 1]).any()):
        raise RuntimeError("two-visible orbital-camera preflight rejects overlapping silhouettes")
    support_pixels = full_mask.sum(dim=(-2, -1))
    if int(support_pixels.min()) < int(DEFAULT_GATES.minimum_full_support_pixels):
        raise RuntimeError(
            "two-visible orbital-camera preflight rejects insufficient raster support"
        )
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
    if float(silhouette_gap.min()) < DEFAULT_GATES.minimum_silhouette_gap_pixels:
        raise RuntimeError(
            "two-visible orbital-camera preflight rejects insufficient silhouette separation"
        )
    if float(boundary_clearance.min()) < DEFAULT_GATES.minimum_boundary_clearance_pixels:
        raise RuntimeError(
            "two-visible orbital-camera preflight rejects insufficient image-boundary clearance"
        )
    pair_distance = torch.linalg.vector_norm(
        objects["position"][:, 0] - objects["position"][:, 1], dim=-1
    )
    surface_gap = pair_distance - objects["radius"][:, :2, 0].sum(dim=-1)
    bounds = torch.tensor(config.simulator.world_bounds, dtype=objects["position"].dtype)
    lower = objects["position"][:, :2] - objects["radius"][:, :2] - bounds[:, 0]
    upper = bounds[:, 1] - objects["position"][:, :2] - objects["radius"][:, :2]
    world_boundary = torch.minimum(lower, upper)
    if float(surface_gap.min()) < DEFAULT_GATES.minimum_world_surface_gap_m:
        raise RuntimeError("two-visible orbital-camera preflight rejects a near-contact trajectory")
    if float(world_boundary.min()) < DEFAULT_GATES.minimum_world_boundary_clearance_m:
        raise RuntimeError(
            "two-visible orbital-camera preflight rejects a near-boundary trajectory"
        )
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
        raise RuntimeError(
            "two-visible orbital-camera preflight rejects collisions/contact/lifecycle events"
        )
    if not bool(events["created"][0, :2].all()):
        raise RuntimeError(
            "two-visible orbital-camera preflight requires both births at frame zero"
        )
    for name, expected in (("radius", 0.21), ("drag", 0.05)):
        if not bool(objects[name][:, :2].eq(expected).all()):
            raise RuntimeError(f"two-visible orbital-camera preflight rejects non-frozen {name}")
    albedo = objects["albedo"][:, :2]
    cross_colour_distance = 1.0 - F.cosine_similarity(albedo[:, 0], albedo[:, 1], dim=-1)
    if float(cross_colour_distance.min()) < DEFAULT_GATES.minimum_cross_appearance_cosine_distance:
        raise RuntimeError("two-visible orbital-camera preflight rejects weak chromatic separation")
    if specification is not None:
        torch.testing.assert_close(objects["position"][0, :2], specification.position)
        torch.testing.assert_close(objects["velocity"][0, :2], specification.velocity)
        torch.testing.assert_close(objects["albedo"][0, :2], specification.albedo)
        expected_position, expected_velocity = _exact_physical_trajectory(specification)
        torch.testing.assert_close(
            objects["position"][:, :2], expected_position, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            objects["velocity"][:, :2], expected_velocity, atol=0.0, rtol=0.0
        )
    camera = episode["camera"]
    world_from_camera = camera["world_from_camera"]
    camera_from_world = camera["camera_from_world"]
    intrinsics = camera["intrinsics"]
    expected_frames = [
        orbital_camera_frame(specification, frame_index / config.simulator.frame_rate)
        for frame_index in range(config.simulator.sequence_frames)
    ]
    expected_world_from = torch.stack([frame.world_from_camera for frame in expected_frames])
    expected_camera_from = torch.stack([frame.camera_from_world for frame in expected_frames])
    expected_intrinsics = torch.stack([frame.intrinsics for frame in expected_frames])
    expected_camera_position = torch.stack([frame.position for frame in expected_frames])
    expected_camera_target = torch.stack([frame.target for frame in expected_frames])
    camera_law_error = max(
        float((world_from_camera - expected_world_from).abs().max()),
        float((camera_from_world - expected_camera_from).abs().max()),
        float((intrinsics - expected_intrinsics).abs().max()),
        float((camera["position"] - expected_camera_position).abs().max()),
        float((camera["target"] - expected_camera_target).abs().max()),
    )
    identity = world_from_camera @ camera_from_world
    inverse_error = float((identity - torch.eye(4)).abs().max())
    rotation = world_from_camera[:, :3, :3]
    orthonormality_error = float((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().max())
    camera_positions = world_from_camera[:, :3, 3]
    radii = torch.linalg.vector_norm(camera_positions[:, [0, 2]], dim=-1)
    radius_error = float((radii - CAMERA_RADIUS_M).abs().max())
    height_error = float((camera_positions[:, 1] - CAMERA_HEIGHT_M).abs().max())
    translation = torch.linalg.vector_norm(camera_positions[1:] - camera_positions[:-1], dim=-1)
    horizontal = camera_positions[:, [0, 2]]
    cosine = (horizontal[1:] * horizontal[:-1]).sum(dim=-1) / (
        torch.linalg.vector_norm(horizontal[1:], dim=-1)
        * torch.linalg.vector_norm(horizontal[:-1], dim=-1)
    )
    adjacent_angle = torch.acos(cosine.clamp(-1.0, 1.0))
    projected_step = torch.linalg.vector_norm(centres[1:] - centres[:-1], dim=-1)
    if float(adjacent_angle.min()) < DEFAULT_GATES.minimum_camera_adjacent_angle_radians:
        raise RuntimeError("orbital-camera preflight found too little angular camera motion")
    if float(adjacent_angle.max()) > DEFAULT_GATES.maximum_camera_adjacent_angle_radians:
        raise RuntimeError("orbital-camera preflight found too much angular camera motion")
    if float(translation.min()) < DEFAULT_GATES.minimum_camera_translation_step_m:
        raise RuntimeError("orbital-camera preflight found too little camera translation")
    if float(translation.max()) > DEFAULT_GATES.maximum_camera_translation_step_m:
        raise RuntimeError("orbital-camera preflight found too much camera translation")
    if float(projected_step.max()) > DEFAULT_GATES.maximum_projected_centre_step_pixels:
        raise RuntimeError("orbital-camera preflight found excessive projected centre motion")
    calibration_error = max(
        camera_law_error,
        inverse_error,
        orthonormality_error,
        radius_error,
        height_error,
    )
    if calibration_error > DEFAULT_GATES.camera_calibration_max_abs_error:
        raise RuntimeError("orbital-camera preflight rejects calibration drift")
    return {
        "preflight_minimum_silhouette_gap_pixels": float(silhouette_gap.min()),
        "preflight_minimum_boundary_clearance_pixels": float(boundary_clearance.min()),
        "preflight_minimum_world_surface_gap_m": float(surface_gap.min()),
        "preflight_minimum_world_boundary_clearance_m": float(world_boundary.min()),
        "preflight_minimum_visible_fraction": float(visible.min()),
        "preflight_minimum_full_support_pixels": float(support_pixels.min()),
        "preflight_event_count": float(event_count),
        "preflight_minimum_palette_cosine_distance": float(cross_colour_distance.min()),
        "preflight_minimum_camera_adjacent_angle_radians": float(adjacent_angle.min()),
        "preflight_maximum_camera_adjacent_angle_radians": float(adjacent_angle.max()),
        "preflight_minimum_camera_translation_step_m": float(translation.min()),
        "preflight_maximum_camera_translation_step_m": float(translation.max()),
        "preflight_maximum_projected_centre_step_pixels": float(projected_step.max()),
        "preflight_camera_calibration_max_abs_error": calibration_error,
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _gate_surface(
    metrics: Mapping[str, Any],
    *,
    schema_only: bool,
) -> tuple[list[str], set[str]]:
    """Evaluate or collect the one canonical scalar gate surface."""

    gates = DEFAULT_GATES
    failures: list[str] = []
    required: set[str] = set()

    def require_max(key: str, maximum: float) -> None:
        required.add(key)
        if schema_only:
            return
        raw = metrics.get(key)
        value = float(raw) if type(raw) is float and math.isfinite(raw) else None
        if value is None:
            failures.append(f"{key}:missing_nonfinite_or_nonfloat")
        elif value > maximum:
            failures.append(f"{key}:{value:.9g}>{maximum:.9g}")

    def require_min(key: str, minimum: float) -> None:
        required.add(key)
        if schema_only:
            return
        raw = metrics.get(key)
        value = float(raw) if type(raw) is float and math.isfinite(raw) else None
        if value is None:
            failures.append(f"{key}:missing_nonfinite_or_nonfloat")
        elif value < minimum:
            failures.append(f"{key}:{value:.9g}<{minimum:.9g}")

    def require_equal(key: str, expected: float) -> None:
        required.add(key)
        if schema_only:
            return
        raw = metrics.get(key)
        value = float(raw) if type(raw) is float and math.isfinite(raw) else None
        if value is None:
            failures.append(f"{key}:missing_nonfinite_or_nonfloat")
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
    require_min(
        "stale_camera_current_position_rmse_m",
        gates.stale_camera_current_position_rmse_m,
    )
    require_max(
        "correct_to_stale_current_position_rmse_ratio",
        gates.correct_to_stale_current_position_rmse_ratio,
    )
    require_min(
        "stale_camera_current_velocity_rmse_mps",
        gates.stale_camera_current_velocity_rmse_mps,
    )
    require_max(
        "correct_to_stale_current_velocity_rmse_ratio",
        gates.correct_to_stale_current_velocity_rmse_ratio,
    )
    require_min(
        "stale_camera_horizon_2_00_position_rmse_m",
        gates.stale_camera_horizon_2_00_position_rmse_m,
    )
    require_max(
        "correct_to_stale_horizon_2_00_position_rmse_ratio",
        gates.correct_to_stale_horizon_2_00_position_rmse_ratio,
    )
    require_equal(
        "stale_camera_identity_switch_count",
        gates.stale_camera_identity_switch_count,
    )
    require_equal(
        "stale_camera_association_ambiguous_pair_count",
        gates.stale_camera_association_ambiguous_pair_count,
    )
    require_equal(
        "stale_camera_history_valid_count_min",
        gates.stale_camera_history_valid_count,
    )
    for camera_stratum in range(8):
        phase_index = camera_stratum // 2
        direction = CAMERA_DIRECTIONS[camera_stratum % 2]
        suffix = f"phase_{phase_index}/direction_{direction:+d}"
        require_min(
            f"stale_camera_current_position_rmse_m/{suffix}",
            gates.stale_camera_current_position_rmse_m,
        )
        require_max(
            f"correct_to_stale_current_position_rmse_ratio/{suffix}",
            gates.correct_to_stale_current_position_rmse_ratio,
        )
        require_min(
            f"stale_camera_current_velocity_rmse_mps/{suffix}",
            gates.stale_camera_current_velocity_rmse_mps,
        )
        require_max(
            f"correct_to_stale_current_velocity_rmse_ratio/{suffix}",
            gates.correct_to_stale_current_velocity_rmse_ratio,
        )
        require_min(
            f"stale_camera_horizon_2_00_position_rmse_m/{suffix}",
            gates.stale_camera_horizon_2_00_position_rmse_m,
        )
        require_max(
            f"correct_to_stale_horizon_2_00_position_rmse_ratio/{suffix}",
            gates.correct_to_stale_horizon_2_00_position_rmse_ratio,
        )
    require_min(
        "certificate_ideal_stale_camera_current_position_rmse_m",
        gates.ideal_wls_stale_camera_current_position_rmse_m,
    )
    require_min(
        "certificate_ideal_stale_camera_current_velocity_rmse_mps",
        gates.ideal_wls_stale_camera_velocity_rmse_mps,
    )
    require_min(
        "certificate_ideal_stale_camera_horizon_2_00_position_rmse_m",
        gates.ideal_wls_stale_camera_horizon_2_00_position_rmse_m,
    )
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
    for camera_stratum in range(8):
        phase_index = camera_stratum // 2
        direction = CAMERA_DIRECTIONS[camera_stratum % 2]
        suffix = f"phase_{phase_index}/direction_{direction:+d}"
        require_max(f"current_position_rmse_m/{suffix}", gates.current_position_rmse_m)
        require_max(f"current_velocity_rmse_mps/{suffix}", gates.current_velocity_rmse_mps)
        for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
            label = f"{horizon:.2f}"
            require_max(
                f"horizon_{label}_position_rmse_m/{suffix}",
                gates.horizon_position_rmse_m[horizon_index],
            )
            require_max(
                f"horizon_{label}_velocity_rmse_mps/{suffix}",
                gates.horizon_velocity_rmse_mps,
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
    require_equal("certificate_physical_trajectory_count", gates.physical_trajectory_count)
    require_equal(
        "certificate_camera_appearance_combination_count",
        gates.camera_appearance_combination_count,
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
    require_min(
        "preflight_minimum_full_support_pixels",
        gates.minimum_full_support_pixels,
    )
    require_equal("preflight_event_count", gates.preflight_event_count)
    require_min(
        "preflight_minimum_palette_cosine_distance",
        gates.minimum_cross_appearance_cosine_distance,
    )
    require_min(
        "preflight_minimum_camera_adjacent_angle_radians",
        gates.minimum_camera_adjacent_angle_radians,
    )
    require_max(
        "preflight_maximum_camera_adjacent_angle_radians",
        gates.maximum_camera_adjacent_angle_radians,
    )
    require_min(
        "preflight_minimum_camera_translation_step_m",
        gates.minimum_camera_translation_step_m,
    )
    require_max(
        "preflight_maximum_camera_translation_step_m",
        gates.maximum_camera_translation_step_m,
    )
    require_max(
        "preflight_maximum_projected_centre_step_pixels",
        gates.maximum_projected_centre_step_pixels,
    )
    require_max(
        "preflight_camera_calibration_max_abs_error",
        gates.camera_calibration_max_abs_error,
    )
    require_max(
        "final_belief_camera_max_abs_error",
        gates.final_belief_camera_max_abs_error,
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
            for modality in ("rgb", "depth", "world_from_camera"):
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
    require_max(
        "world_from_camera_homogeneous_last_row_gradient_max_abs",
        gates.world_from_camera_homogeneous_last_row_gradient_max_abs,
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
    return failures, required


def gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Recompute the complete exact scalar gate schema from report evidence."""

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
if len(GATE_METRIC_SCHEMA) != 685:
    raise RuntimeError("orbital-camera exact gate metric schema must contain 685 scalars")


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
        raise RuntimeError("two-visible orbital-camera runtime failed to retain a belief")
    history = model.state.temporal_histories.get(runtime_stream_key("rgbd", "camera0:rgbd"))
    if not isinstance(history, RGBDTemporalPositionHistory):
        raise RuntimeError(
            "two-visible orbital-camera runtime failed to retain typed temporal histories"
        )
    if runtime_stream_key("rgbd", "camera0:rgbd") != RUNTIME_STREAM_KEY:
        raise RuntimeError("two-visible orbital-camera runtime stream key changed")
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
        raise RuntimeError(
            "two-visible orbital-camera RGB-D measurement must be the sole position owner"
        )
    raw_position = raw.auxiliary.get("world_position")
    if not isinstance(raw_position, Tensor):
        raise RuntimeError("two-visible orbital-camera measurement omitted raw world position")
    measurement_by_belief = association_audit["last_measurement_by_belief"]
    if not isinstance(measurement_by_belief, Tensor):
        raise RuntimeError(
            "two-visible orbital-camera final frame did not produce a complete association"
        )
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
    expected_final_camera = batch["camera"]["world_from_camera"][:, ANCHOR_FRAME_INDEX]
    final_belief_camera_error = (belief.camera.world_from_camera - expected_final_camera).abs()
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
        "final_belief_camera_error": final_belief_camera_error,
        "runtime_tensor_bytes": _persistent_runtime_tensor_bytes(model),
    }


def _stale_world_from_camera_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Clone only WFC and freeze history frames 1..15 to exact frame zero."""

    stale = dict(batch)
    stale_camera = dict(batch["camera"])
    world_from_camera = batch["camera"]["world_from_camera"]
    stale_world_from_camera = world_from_camera.clone()
    stale_world_from_camera[:, 1:16] = stale_world_from_camera[:, :1]
    stale_camera["world_from_camera"] = stale_world_from_camera
    stale["camera"] = stale_camera
    for key in ("rgb", "depth", "timestamps", "objects", "labels", "events", "metadata"):
        if stale[key] is not batch[key]:
            raise AssertionError(f"stale-camera control unexpectedly copied {key}")
    for key, value in batch["camera"].items():
        if key != "world_from_camera" and stale_camera[key] is not value:
            raise AssertionError(f"stale-camera control unexpectedly changed camera.{key}")
    if not torch.equal(stale_world_from_camera[:, 0], world_from_camera[:, 0]):
        raise AssertionError("stale-camera control changed frame-zero calibration")
    if not torch.equal(stale_world_from_camera[:, 16:], world_from_camera[:, 16:]):
        raise AssertionError("stale-camera control changed non-history calibration")
    return stale


def _stale_camera_batch_metrics(
    batch: Mapping[str, Any],
    config: OrpheusConfig,
) -> dict[str, Tensor | float]:
    """Run the fresh-runtime stale-extrinsics negative control."""

    stale_batch = _stale_world_from_camera_batch(batch)
    with torch.no_grad():
        output = _run_public_batch(stale_batch, config)
    observed = output["observed_positions"]
    mapping, _ = _birth_physical_mapping(observed[:, 0], batch["objects"]["position"][:, 0, :2])
    anchor_position = _gather_physical_by_slot(
        batch["objects"]["position"][:, ANCHOR_FRAME_INDEX, :2], mapping
    )
    anchor_velocity = _gather_physical_by_slot(
        batch["objects"]["velocity"][:, ANCHOR_FRAME_INDEX, :2], mapping
    )
    final_position = _gather_physical_by_slot(
        batch["objects"]["position"][:, TARGET_FRAME_INDICES[-1], :2], mapping
    )
    identity_switches = 0
    for frame_offset, frame_index in enumerate(HISTORY_FRAME_INDICES):
        frame_mapping, _ = _birth_physical_mapping(
            observed[:, frame_offset], batch["objects"]["position"][:, frame_index, :2]
        )
        identity_switches += int(frame_mapping.ne(mapping).any(dim=-1).sum().detach().cpu())
    history = output["history"]
    return {
        "current_position_error": (output["belief"].objects.position - anchor_position).cpu(),
        "current_velocity_error": (output["belief"].objects.velocity - anchor_velocity).cpu(),
        "horizon_2_00_position_error": (
            output["trajectory"].positions[:, -1] - final_position
        ).cpu(),
        "identity_switch_count": float(identity_switches),
        "association_ambiguous_pair_count": float(output["association_audit"]["ambiguous"]),
        "history_valid_count_min": float(history.valid_mask.sum(dim=-1).min().detach().cpu()),
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
        raise FloatingPointError("two-visible orbital-camera history VJP is malformed or nonfinite")
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
        raise ValueError("two-visible orbital-camera VJP audit requires one complete frozen batch")
    differentiable = dict(batch)
    differentiable["rgb"] = batch["rgb"].index_select(1, indices).clone().requires_grad_(True)
    differentiable["depth"] = batch["depth"].index_select(1, indices).clone().requires_grad_(True)
    differentiable_camera: dict[str, Any] = {}
    for name, value in batch["camera"].items():
        resolved = (
            value.index_select(1, indices.to(value.device)).clone()
            if isinstance(value, Tensor)
            else value
        )
        if name in {"world_from_camera", "intrinsics"}:
            resolved.requires_grad_(True)
        differentiable_camera[name] = resolved
    differentiable["camera"] = differentiable_camera
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
    inputs = (
        differentiable["rgb"],
        differentiable["depth"],
        differentiable["camera"]["world_from_camera"],
        differentiable["camera"]["intrinsics"],
    )
    objects = batch["objects"]
    scene_signatures = {
        canonical_sha256(
            {
                "physical": {
                    name: objects[name][batch_index, 0, :2].detach().cpu().tolist()
                    for name in ("position", "velocity", "albedo")
                },
                "world_from_camera": batch["camera"]["world_from_camera"][batch_index]
                .detach()
                .cpu()
                .tolist(),
            }
        )
        for batch_index in range(batch_size)
    }
    metrics: dict[str, float] = {
        "gradient_audit_scene_count": float(batch_size),
        "gradient_audit_unique_scene_fraction": len(scene_signatures) / batch_size,
        "world_from_camera_homogeneous_last_row_gradient_max_abs": 0.0,
    }
    for loss_index, (batch_index, object_index, output_name, loss) in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            inputs,
            retain_graph=loss_index + 1 < len(losses),
            allow_unused=True,
        )
        for modality, source, gradient in zip(
            ("rgb", "depth", "world_from_camera", "intrinsics"),
            inputs,
            gradients,
            strict=True,
        ):
            resolved = torch.zeros_like(source) if gradient is None else gradient
            if not bool(torch.isfinite(resolved).all()):
                raise FloatingPointError(
                    f"object {object_index} {output_name} has nonfinite {modality} VJP"
                )
            # Intrinsics are known calibration and remain a finite diagnostic,
            # not part of the exact qualification gate schema.  The runtime's
            # robust qualification claims are RGB, depth, and ambient WFC VJPs.
            if modality == "intrinsics":
                continue
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
            if modality == "world_from_camera":
                metrics["world_from_camera_homogeneous_last_row_gradient_max_abs"] = max(
                    metrics["world_from_camera_homogeneous_last_row_gradient_max_abs"],
                    float(resolved[..., 3, :].abs().max()),
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
    if any(type(seed) is not int for seed in seeds):
        raise TypeError("orbital-camera manifests require exact integer seeds")
    requested = tuple(seeds)
    if split not in manifests or requested != manifests[split]:
        raise ValueError(
            f"{split!r} must use its exact frozen two-visible orbital-camera RGB-D manifest"
        )
    if len(requested) != len(set(requested)):
        raise ValueError("two-visible orbital-camera RGB-D manifest contains duplicate seeds")
    if canonical_sha256(list(requested)) != MANIFEST_SHA256[split]:
        raise RuntimeError("orbital-camera manifest hash differs from the frozen protocol")
    return requested


def _episode_scene_signature(episode: Mapping[str, Any]) -> str:
    objects = episode["objects"]
    return canonical_sha256(
        {
            "physical": {
                name: objects[name][0, :2].detach().cpu().tolist()
                for name in ("position", "velocity", "albedo")
            },
            "world_from_camera": episode["camera"]["world_from_camera"].detach().cpu().tolist(),
            "primitive_a": episode["metadata"]["primitive_a"],
            "primitive_b": episode["metadata"]["primitive_b"],
            "phase_index": episode["metadata"]["camera_phase_index"],
            "direction": episode["metadata"]["camera_direction"],
        }
    )


def _evaluate_seed_manifest(
    config: OrpheusConfig,
    seeds: Sequence[int],
    *,
    split: str,
    capability: _ManifestCapability,
) -> dict[str, Any]:
    """Evaluate one exact already-authorized manifest with zero optimizer work."""

    _require_config_matches_frozen_path(config, _frozen_config_path())
    requested = _validate_manifest(split, seeds)
    if type(capability) is not _ManifestCapability:
        raise PermissionError("exact manifest evaluation requires a nominal ledger capability")
    capability.begin_manifest(split, requested)
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
            "final_belief_camera_error",
            "stale_camera_current_position_error",
            "stale_camera_current_velocity_error",
            "stale_camera_horizon_2_00_position_error",
        )
    }
    camera_strata: list[Tensor] = []
    stale_identity_switch_count = 0.0
    stale_association_ambiguous_pair_count = 0.0
    stale_history_valid_count_min = math.inf
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
            _construct_two_visible_orbital_camera_episode(
                config,
                seed,
                split=split,
                capability=capability,
            )
            for seed in seed_chunk
        ]
        for episode in episodes:
            scene_signatures.add(_episode_scene_signature(episode))
            evidence = preflight_two_visible_orbital_camera_episode(episode, config=config)
            for name, value in evidence.items():
                preflight_values.setdefault(name, []).append(value)
        batch = collate_episodes(episodes)
        if first_batch is None:
            first_batch = batch
        with torch.no_grad():
            output = _run_public_batch(batch, config)
        stale = _stale_camera_batch_metrics(batch, config)
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
            "final_belief_camera_error",
        ):
            accumulated[name].append(output[name].cpu())
        accumulated["stale_camera_current_position_error"].append(stale["current_position_error"])
        accumulated["stale_camera_current_velocity_error"].append(stale["current_velocity_error"])
        accumulated["stale_camera_horizon_2_00_position_error"].append(
            stale["horizon_2_00_position_error"]
        )
        stale_identity_switch_count += float(stale["identity_switch_count"])
        stale_association_ambiguous_pair_count += float(stale["association_ambiguous_pair_count"])
        stale_history_valid_count_min = min(
            stale_history_valid_count_min,
            float(stale["history_valid_count_min"]),
        )
        stratum = batch["metadata"]["camera_stratum"]
        if not isinstance(stratum, Tensor) or stratum.shape != (len(seed_chunk),):
            raise RuntimeError("heterogeneous camera-stratum metadata was not row-preserved")
        camera_strata.append(stratum.to(torch.int64).cpu())
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

    capability.finish_manifest()
    if first_batch is None:
        raise RuntimeError("two-visible orbital-camera manifest unexpectedly produced no batches")
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
    stale_current_position_rmse = _rmse(tensors["stale_camera_current_position_error"])
    stale_current_velocity_rmse = _rmse(tensors["stale_camera_current_velocity_error"])
    stale_horizon_2_00_position_rmse = _rmse(tensors["stale_camera_horizon_2_00_position_error"])
    strata = torch.cat(camera_strata)
    epsilon = torch.finfo(torch.float64).eps
    model = new_public_model(config)
    learned = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    buffers = tuple(model.buffers())
    state_tensors = tuple(model.state_dict().values())
    certificate = scene_family_certificate()
    if certificate["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256:
        raise RuntimeError("orbital-camera scene certificate differs from frozen source")
    ideal_minimum = certificate["ideal_stale_camera_control"]["minimum_joint_rmse"]
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
        "stale_camera_current_position_rmse_m": stale_current_position_rmse,
        "correct_to_stale_current_position_rmse_ratio": current_position_rmse
        / max(stale_current_position_rmse, epsilon),
        "stale_camera_current_velocity_rmse_mps": stale_current_velocity_rmse,
        "correct_to_stale_current_velocity_rmse_ratio": current_velocity_rmse
        / max(stale_current_velocity_rmse, epsilon),
        "stale_camera_horizon_2_00_position_rmse_m": stale_horizon_2_00_position_rmse,
        "correct_to_stale_horizon_2_00_position_rmse_ratio": future_position_rmse[-1]
        / max(stale_horizon_2_00_position_rmse, epsilon),
        "stale_camera_identity_switch_count": stale_identity_switch_count,
        "stale_camera_association_ambiguous_pair_count": (stale_association_ambiguous_pair_count),
        "stale_camera_history_valid_count_min": stale_history_valid_count_min,
        "certificate_ideal_stale_camera_current_position_rmse_m": ideal_minimum[
            "current_position_rmse_m"
        ],
        "certificate_ideal_stale_camera_current_velocity_rmse_mps": ideal_minimum[
            "current_velocity_rmse_mps"
        ],
        "certificate_ideal_stale_camera_horizon_2_00_position_rmse_m": ideal_minimum[
            "horizon_2_00_position_rmse_m"
        ],
        "certificate_physical_trajectory_count": float(certificate["physical_trajectory_count"]),
        "certificate_camera_appearance_combination_count": float(
            certificate["camera_appearance_combination_count"]
        ),
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
        "physical_palette_swap_fraction": sum(
            scene_specification(split, ordinal).palette_swapped for ordinal in range(len(requested))
        )
        / len(requested),
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
        "final_belief_camera_max_abs_error": float(tensors["final_belief_camera_error"].max()),
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
    for camera_stratum in range(8):
        mask = strata == camera_stratum
        if not bool(mask.any()):
            raise RuntimeError(f"camera stratum {camera_stratum} is absent from {split}")
        phase_index = camera_stratum // 2
        direction = CAMERA_DIRECTIONS[camera_stratum % 2]
        suffix = f"phase_{phase_index}/direction_{direction:+d}"
        correct_position = _rmse(tensors["current_position_error"][mask])
        correct_velocity = _rmse(tensors["current_velocity_error"][mask])
        correct_future = _rmse(tensors["future_position_error"][mask, -1])
        stale_position = _rmse(tensors["stale_camera_current_position_error"][mask])
        stale_velocity = _rmse(tensors["stale_camera_current_velocity_error"][mask])
        stale_future = _rmse(tensors["stale_camera_horizon_2_00_position_error"][mask])
        metrics[f"stale_camera_current_position_rmse_m/{suffix}"] = stale_position
        metrics[f"correct_to_stale_current_position_rmse_ratio/{suffix}"] = correct_position / max(
            stale_position, epsilon
        )
        metrics[f"stale_camera_current_velocity_rmse_mps/{suffix}"] = stale_velocity
        metrics[f"correct_to_stale_current_velocity_rmse_ratio/{suffix}"] = correct_velocity / max(
            stale_velocity, epsilon
        )
        metrics[f"stale_camera_horizon_2_00_position_rmse_m/{suffix}"] = stale_future
        metrics[f"correct_to_stale_horizon_2_00_position_rmse_ratio/{suffix}"] = (
            correct_future / max(stale_future, epsilon)
        )
        metrics[f"current_position_rmse_m/{suffix}"] = correct_position
        metrics[f"current_velocity_rmse_mps/{suffix}"] = correct_velocity
        for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
            label = f"{horizon:.2f}"
            metrics[f"horizon_{label}_position_rmse_m/{suffix}"] = _rmse(
                tensors["future_position_error"][mask, horizon_index]
            )
            metrics[f"horizon_{label}_velocity_rmse_mps/{suffix}"] = _rmse(
                tensors["future_velocity_error"][mask, horizon_index]
            )
    for name, values in preflight_values.items():
        metrics[name] = (
            max(values)
            if "maximum" in name or "max_abs" in name or name == "preflight_event_count"
            else min(values)
        )
    metrics.update(_ambiguity_fail_closed_metrics(config, first_batch))
    metrics.update(_gradient_metrics(config, first_batch))
    metrics.update(_latency_metrics(config, first_batch))
    if set(metrics) != set(GATE_METRIC_SCHEMA):
        raise RuntimeError("evaluator metric keys differ from the exact 685-scalar schema")
    for name, value in metrics.items():
        if type(value) is not float or not math.isfinite(value):
            raise FloatingPointError(
                f"two-visible orbital-camera metric {name!r} is not an exact finite float"
            )
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
        "scene_constructor": "private_two_visible_orbital_camera_episode_with_full_frame_preflight",
    }


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_RELATIVE_PATH = Path("runs/rgbd_two_visible_orbital_camera_v1")
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


def _frozen_config_path() -> Path:
    return REPOSITORY_ROOT / "configs" / "rgbd_two_visible_orbital_camera_cpu.yaml"


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
        raise TypeError(f"{label} must use the exact native Path type")
    if actual != expected:
        raise ValueError(f"{label} must use canonical fixed path {expected}")
    try:
        relative = actual.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must remain lexically inside the repository") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} has a non-canonical lexical path")


def _require_nonlink_directory(path: Path, *, label: str) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"{label} must be a real directory, not a link")


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
    """Validate one lexical, non-link, stage-exact fixed-run inventory."""

    allowed = QUALIFICATION_ARTIFACT_NAMES
    if not expected_names <= allowed:
        raise ValueError(f"{stage} requested an unknown orbital-camera artifact inventory")
    runs_root = REPOSITORY_ROOT / "runs"
    run_directory = _canonical_run_directory()
    if _lexists(runs_root):
        _require_nonlink_directory(runs_root, label=f"{stage} runs root")
    elif expected_names:
        raise FileNotFoundError(f"{stage} requires the fixed runs root")
    else:
        return
    if _lexists(run_directory):
        _require_nonlink_directory(run_directory, label=f"{stage} fixed run directory")
    elif expected_names:
        raise FileNotFoundError(f"{stage} requires the fixed run directory")
    else:
        return
    with os.scandir(run_directory) as entries:
        materialized = list(entries)
    names = {entry.name for entry in materialized}
    if names != set(expected_names):
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        raise PermissionError(
            f"{stage} fixed run inventory differs; missing={missing}, unexpected={extra}"
        )
    for entry in materialized:
        if entry.is_symlink():
            raise PermissionError(f"{stage} artifact must not be a symbolic link: {entry.name}")
        _require_single_link_regular(
            run_directory / entry.name,
            label=f"{stage} artifact {entry.name}",
        )


def _validate_distinct_canonical_paths(
    paths: Mapping[str, Path],
    *,
    atomic_writers: Sequence[str],
) -> None:
    """Reject lexical and inode aliases without resolving through run parents."""

    if type(paths) is not dict:
        raise TypeError("orbital-camera artifact path map must be an exact dict")
    expanded: dict[str, Path] = {}
    for name, path in paths.items():
        if type(name) is not str or type(path) is not _NATIVE_PATH_TYPE:
            raise TypeError("orbital-camera artifact paths require exact names/native Paths")
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
        details = "; ".join(", ".join(names) for names in collisions)
        raise ValueError("orbital-camera artifact paths lexically alias: " + details)
    identities: dict[tuple[int, int], str] = {}
    for name, path in expanded.items():
        if not _lexists(path):
            continue
        metadata = os.lstat(path)
        identity = (metadata.st_dev, metadata.st_ino)
        prior = identities.get(identity)
        if prior is not None:
            raise ValueError(f"orbital-camera artifact paths hard-link alias: {prior}, {name}")
        identities[identity] = name


def _ensure_canonical_run_directory() -> None:
    """Create only the two fixed lexical directories, then reject link retargeting."""

    runs_root = REPOSITORY_ROOT / "runs"
    run_directory = _canonical_run_directory()
    if not _lexists(runs_root):
        os.mkdir(runs_root, 0o700)
    _require_nonlink_directory(runs_root, label="orbital-camera runs root")
    if not _lexists(run_directory):
        os.mkdir(run_directory, 0o700)
    _require_nonlink_directory(run_directory, label="orbital-camera fixed run directory")


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def require_frozen_config(path: str | Path) -> OrpheusConfig:
    """Load only the exact frozen raw profile and semantic contract."""

    source = Path(path)
    contents = stable_read_bytes(source, label="two-visible orbital-camera frozen config")
    digest = sha256_bytes(contents)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "two-visible orbital-camera RGB-D requires exact frozen config bytes: "
            f"expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    config = load_config(source)
    assert_rgbd_two_visible_orbital_camera_config(config)
    return config


def _require_config_matches_frozen_path(config: OrpheusConfig, path: Path) -> None:
    """Bind the exact executed config object to the immutable profile bytes."""

    before = stable_read_bytes(path, label="two-visible orbital-camera frozen config binding")
    if sha256_bytes(before) != FROZEN_CONFIG_SHA256:
        raise ValueError("two-visible orbital-camera config path differs from frozen bytes")
    parsed = load_config(path)
    after = stable_read_bytes(
        path, label="two-visible orbital-camera frozen config binding recheck"
    )
    if after != before:
        raise RuntimeError("two-visible orbital-camera config bytes changed while being parsed")
    if canonical_sha256(config.to_dict()) != canonical_sha256(parsed.to_dict()):
        raise ValueError("executed config object differs from exact frozen config bytes")
    assert_rgbd_two_visible_orbital_camera_config(parsed)


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
    if path.parent == _canonical_run_directory():
        _ensure_canonical_run_directory()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    _write_descriptor(descriptor, contents)
    _fsync_parent(path)


def _durable_replace(path: Path, contents: bytes, *, mode: int = 0o600) -> None:
    if path.parent == _canonical_run_directory():
        _ensure_canonical_run_directory()
        _require_single_link_regular(path, label="orbital-camera replacement target")
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


def _report_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(report), allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _write_report_fresh(path: Path, report: Mapping[str, Any]) -> None:
    if type(path) is not _NATIVE_PATH_TYPE or path not in {
        canonical_development_report_path(),
        canonical_qualification_report_path(),
    }:
        raise ValueError("orbital-camera report path is not one of the two fixed artifacts")
    _durable_create(path, _report_bytes(report))


def _persist_failed_report(path: Path, report: Mapping[str, Any], *, label: str) -> str | None:
    """Best-effort failed evidence; bind only the exact intended failed bytes."""

    intended = _report_bytes(report)
    try:
        if _lexists(path):
            _require_single_link_regular(path, label=f"{label} existing report")
            _durable_replace(path, intended)
        else:
            _write_report_fresh(path, report)
    except BaseException:
        # The ledger must never bind stale passing bytes after a failed rewrite.
        # A missing or contradictory report is represented by a null digest and
        # remains terminally failed through the durable ledger.
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


def _refresh_ledger_receipt(ledger: object) -> None:
    path = getattr(ledger, "path", None)
    record = getattr(ledger, "record", None)
    if type(path) is not _NATIVE_PATH_TYPE or type(record) is not dict:
        raise PermissionError("manifest ledger has the wrong nominal state")
    if path.parent == _canonical_run_directory():
        _ensure_canonical_run_directory()
    metadata = _require_single_link_regular(path, label="manifest ledger")
    contents = stable_read_bytes(path, label="orbital-camera live access ledger")
    parsed = json.loads(contents)
    if canonical_sha256(parsed) != canonical_sha256(record):
        raise RuntimeError("manifest ledger memory and durable receipt differ")
    ledger._receipt_digest = sha256_bytes(contents)
    ledger._receipt_device = metadata.st_dev
    ledger._receipt_inode = metadata.st_ino


def _validate_manifest_capability(
    capability: _ManifestCapability,
    *,
    split: str,
    seeds: Sequence[int],
    operation: str,
) -> None:
    """Revalidate exact nominal authority and current durable bytes at every boundary."""

    if type(capability) is not _ManifestCapability:
        raise PermissionError("manifest capability has the wrong nominal type")
    registration = _LIVE_MANIFEST_CAPABILITIES.get(id(capability))
    if (
        type(registration) is not tuple
        or len(registration) != 2
        or registration[0] is not capability
        or registration[1] is not getattr(capability, "_ledger", None)
    ):
        raise PermissionError("manifest capability is not live-ledger registered")
    ledger = registration[1]
    if type(ledger) not in {_DevelopmentLedger, _QualificationLedger}:
        raise PermissionError("manifest capability was not minted by an exact private ledger")
    ledger_registration = _LIVE_PRIVATE_LEDGERS.get(id(ledger))
    if (
        type(ledger_registration) is not tuple
        or len(ledger_registration) != 4
        or ledger_registration[0] is not ledger
        or ledger_registration[1] is not getattr(ledger, "_mint_identity", None)
    ):
        raise PermissionError("manifest capability ledger is not live-issuer registered")
    if type(ledger) is _DevelopmentLedger:
        if getattr(ledger, "_capability", None) is not capability:
            raise PermissionError("development capability is not the ledger-owned capability")
        expected_inventory = frozenset({DEVELOPMENT_LEDGER_NAME})
    else:
        reviewed_seal = getattr(ledger, "_reviewed_seal", None)
        if ledger_registration[3] is not reviewed_seal or _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(
            id(reviewed_seal)
        ) != (reviewed_seal, canonical_sha256(ledger._bindings), ledger):
            raise PermissionError("protected capability lost its reviewed-development seal")
        capabilities = getattr(ledger, "_capabilities", None)
        if type(capabilities) is not dict or capabilities.get(split) is not capability:
            raise PermissionError("protected capability is not the ledger-owned split capability")
        expected_inventory = frozenset({*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_LEDGER_NAME})
    _validate_run_tree(expected_inventory, stage=f"{split} materialization boundary")
    expected_path = (
        development_ledger_path()
        if type(ledger) is _DevelopmentLedger
        else qualification_ledger_path()
    )
    if type(ledger.path) is not _NATIVE_PATH_TYPE or ledger.path != expected_path:
        raise PermissionError("manifest capability ledger path is not canonical")
    metadata = os.lstat(ledger.path)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_dev != ledger._receipt_device
        or metadata.st_ino != ledger._receipt_inode
    ):
        raise PermissionError("manifest ledger identity/link count changed")
    contents = stable_read_bytes(ledger.path, label=f"{split} access capability receipt")
    if sha256_bytes(contents) != ledger._receipt_digest:
        raise PermissionError("manifest ledger receipt digest changed")
    parsed = json.loads(contents)
    if type(parsed) is not dict or canonical_sha256(parsed) != canonical_sha256(ledger.record):
        raise PermissionError("manifest ledger receipt no longer matches live registration")
    if (
        type(ledger.record.get("bindings")) is not dict
        or ledger.record["bindings"] != ledger._bindings
    ):
        raise PermissionError("manifest ledger full bindings changed")
    current_source = clean_source(
        capture_git_metadata(REPOSITORY_ROOT),
        label="orbital-camera live manifest capability",
    )
    config_path = _frozen_config_path()
    if (
        sha256_bytes(stable_read_bytes(config_path, label="capability frozen config"))
        != FROZEN_CONFIG_SHA256
    ):
        raise PermissionError("manifest capability observed changed config bytes")
    certificate = scene_family_certificate()
    if certificate["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256:
        raise PermissionError("manifest capability observed changed certificate")
    requested = tuple(seeds)
    if split != capability._split or requested != capability._seeds:
        raise PermissionError("manifest capability split/seeds differ from the ledger grant")
    if _manifest_for_split(split) != capability._seeds:
        raise PermissionError("manifest capability is not bound to the canonical seed order")
    if canonical_sha256(list(requested)) != MANIFEST_SHA256[split]:
        raise PermissionError("manifest capability seed receipt hash differs")
    if type(ledger) is _DevelopmentLedger:
        development_record_schema = {
            "artifact_kind",
            "architecture_attempt",
            "maximum_architecture_attempts",
            "bindings",
            "attempt_reserved",
            "access_started",
            "development_data_materialized",
            "result_sha256",
            "status",
        }
        if set(ledger.record) != development_record_schema:
            raise PermissionError("development materialization ledger schema differs")
        expected_keys = {
            "protocol_sha256",
            "source_provenance",
            "publication_provenance",
            "config_sha256",
            "development_manifest_sha256",
            "certificate_sha256",
        }
        if split != "development" or set(ledger._bindings) != expected_keys:
            raise PermissionError("development ledger has incomplete or extra bindings")
        expected_bindings = {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": current_source,
            "publication_provenance": _validated_published_source(
                capture_published_source(REPOSITORY_ROOT),
                source=current_source,
                label="development capability publication",
            ),
            "config_sha256": FROZEN_CONFIG_SHA256,
            "development_manifest_sha256": MANIFEST_SHA256["development"],
            "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        }
        if ledger._bindings != expected_bindings:
            raise PermissionError("development ledger binding values differ from live source")
        if (
            ledger.record.get("artifact_kind") != ledger.ARTIFACT_KIND
            or ledger.record.get("architecture_attempt") != ARCHITECTURE_ATTEMPT
            or ledger.record.get("maximum_architecture_attempts") != MAX_ARCHITECTURE_ATTEMPTS
            or ledger.record.get("attempt_reserved") is not True
            or ledger.record.get("access_started") is not True
            or ledger.record.get("development_data_materialized") is not True
            or ledger.record.get("status") != "development_materialization_started"
        ):
            raise PermissionError("development receipt is not materialization-started")
    else:
        qualification_record_schema = {
            "artifact_kind",
            "architecture_attempt",
            "maximum_architecture_attempts",
            "order",
            "bindings",
            "splits",
            "attempt_reserved",
            "protected_data_materialized",
            "status",
        }
        if set(ledger.record) != qualification_record_schema:
            raise PermissionError("protected materialization ledger schema differs")
        expected_keys = {
            "protocol_sha256",
            "source_provenance",
            "publication_provenance",
            "config_sha256",
            "reviewed_checkpoint_sha256",
            "reviewed_development_report_sha256",
            "reviewed_development_ledger_sha256",
            "model_state_sha256",
            "certificate_sha256",
        }
        if (
            set(ledger._bindings) != expected_keys
            or tuple(ledger.record.get("order", ())) != ledger.ORDER
        ):
            raise PermissionError("qualification ledger has incomplete bindings/order")
        for binding in (
            "reviewed_checkpoint_sha256",
            "reviewed_development_report_sha256",
            "reviewed_development_ledger_sha256",
        ):
            validated_sha256(ledger._bindings.get(binding), label=binding)
        expected_known_bindings = {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": current_source,
            "publication_provenance": _validated_published_source(
                capture_published_source(REPOSITORY_ROOT),
                source=current_source,
                label="qualification capability publication",
            ),
            "config_sha256": FROZEN_CONFIG_SHA256,
            "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
            "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        }
        if any(
            ledger._bindings.get(key) != value for key, value in expected_known_bindings.items()
        ):
            raise PermissionError(
                "qualification ledger known binding values differ from live source"
            )
        for path, binding, label in (
            (
                canonical_checkpoint_path(),
                "reviewed_checkpoint_sha256",
                "capability reviewed checkpoint",
            ),
            (
                canonical_development_report_path(),
                "reviewed_development_report_sha256",
                "capability reviewed development report",
            ),
            (
                development_ledger_path(),
                "reviewed_development_ledger_sha256",
                "capability reviewed development ledger",
            ),
        ):
            _require_single_link_regular(path, label=label)
            if sha256_bytes(stable_read_bytes(path, label=label)) != ledger._bindings[binding]:
                raise PermissionError(f"{label} differs from qualification binding")
        states = ledger.record.get("splits")
        if type(states) is not dict or set(states) != set(ledger.ORDER):
            raise PermissionError("protected split ledger root schema differs")
        if any(
            type(value) is not dict or set(value) != set(LEDGER_SPLIT_STATE_SCHEMA)
            for value in states.values()
        ):
            raise PermissionError("protected split ledger state schema differs")
        state_value = states.get(split) if type(states) is dict else None
        if (
            ledger.record.get("artifact_kind") != ledger.ARTIFACT_KIND
            or ledger.record.get("architecture_attempt") != ARCHITECTURE_ATTEMPT
            or ledger.record.get("maximum_architecture_attempts") != MAX_ARCHITECTURE_ATTEMPTS
            or ledger.record.get("attempt_reserved") is not True
            or ledger.record.get("protected_data_materialized") is not True
            or type(state_value) is not dict
            or state_value.get("access_started") is not True
            or state_value.get("status") != "materialization_started"
            or ledger.record.get("status") != f"{split}_materialization_started"
        ):
            raise PermissionError("protected receipt is not materialization-started")
        index = ledger.ORDER.index(split)
        if any(
            states[predecessor].get("status") != "passed" for predecessor in ledger.ORDER[:index]
        ) or any(
            states[later].get("status") != "unopened"
            or states[later].get("access_started") is not False
            for later in ledger.ORDER[index + 1 :]
        ):
            raise PermissionError("protected receipt split order differs from protocol")
    if operation not in {"begin", "seed", "finish", "complete"}:
        raise ValueError("unknown manifest capability operation")


class _DevelopmentLedger:
    """Fixed attempt-scoped receipt preventing repeated development access."""

    ARTIFACT_KIND = "rgbd_two_visible_orbital_camera_development_access_ledger"

    def __init__(self, authorization: _RunAuthorization, bindings: Mapping[str, Any]) -> None:
        if type(self) is not _DevelopmentLedger or type(bindings) is not dict:
            raise PermissionError("development ledger requires exact nominal construction")
        _consume_run_authorization(
            authorization,
            kind="development",
            bindings=bindings,
        )
        self.path = development_ledger_path()
        self._bindings = dict(bindings)
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
            "bindings": dict(self._bindings),
            "attempt_reserved": True,
            "access_started": True,
            "development_data_materialized": True,
            "result_sha256": None,
            "status": "development_materialization_started",
        }
        self._capability_issued = False
        self._capability: _ManifestCapability | None = None
        self._mint_identity = object()
        _durable_create(self.path, self._serialized())
        _refresh_ledger_receipt(self)
        _LIVE_PRIVATE_LEDGERS[id(self)] = (
            self,
            self._mint_identity,
            authorization,
            None,
        )

    def _serialized(self) -> bytes:
        return (
            json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )

    def _replace(self) -> None:
        _durable_replace(self.path, self._serialized())
        _refresh_ledger_receipt(self)

    def capability(self) -> _ManifestCapability:
        if self._capability_issued:
            raise RuntimeError("development manifest capability cannot be issued twice")
        self._capability_issued = True
        self._capability = _ManifestCapability(
            _MANIFEST_CAPABILITY_AUTHORITY,
            ledger=self,
            ledger_mint_identity=self._mint_identity,
            split="development",
            seeds=DEVELOPMENT_SEEDS,
        )
        return self._capability

    def complete_evaluation(self, result: Mapping[str, Any]) -> None:
        if self.record["status"] != "development_materialization_started":
            raise RuntimeError("development evaluation was not durably opened")
        if self._capability is None:
            raise RuntimeError("development manifest capability was not issued")
        self._capability.require_finished()
        _validate_split_result(result, split="development")
        _LIVE_MANIFEST_CAPABILITIES.pop(id(self._capability), None)
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
        _LIVE_PRIVATE_LEDGERS.pop(id(self), None)

    def record_error(
        self,
        error: BaseException,
        *,
        report_sha256: str | None = None,
    ) -> None:
        prior_status = self.record.get("status")
        if self._capability is not None:
            _LIVE_MANIFEST_CAPABILITIES.pop(id(self._capability), None)
        self.record["status"] = "error"
        self.record["outcome"] = "failed"
        self.record["status_before_error"] = prior_status
        self.record["error"] = {"type": type(error).__name__, "message": str(error)}
        self.record["report_sha256"] = (
            None
            if report_sha256 is None
            else validated_sha256(report_sha256, label="failed development report SHA-256")
        )
        self.record.setdefault("checkpoint_sha256", None)
        _validate_development_error_ledger_schema(self.record)
        try:
            self._replace()
        finally:
            _LIVE_PRIVATE_LEDGERS.pop(id(self), None)


class _QualificationLedger:
    """Fixed exclusive selector -> confirmation -> one-shot-final receipt."""

    ARTIFACT_KIND = "rgbd_two_visible_orbital_camera_exactly_once_access_ledger"
    ORDER = ("selector", "confirmation", "final_test")
    MANIFESTS = {
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }

    def __init__(
        self,
        authorization: _RunAuthorization,
        reviewed_seal: _ReviewedDevelopmentSeal,
        bindings: Mapping[str, Any],
    ) -> None:
        if type(self) is not _QualificationLedger or type(bindings) is not dict:
            raise PermissionError("qualification ledger requires exact nominal construction")
        _consume_run_authorization(
            authorization,
            kind="qualification",
            bindings=bindings,
            reviewed_seal=reviewed_seal,
        )
        self.path = qualification_ledger_path()
        self._bindings = dict(bindings)
        self.record: dict[str, Any] = {
            "artifact_kind": self.ARTIFACT_KIND,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
            "order": list(self.ORDER),
            "bindings": dict(self._bindings),
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
        self._capabilities: dict[str, _ManifestCapability] = {}
        self._reviewed_seal = reviewed_seal
        self._mint_identity = object()
        _durable_create(self.path, self._serialized())
        _refresh_ledger_receipt(self)
        _bind_reviewed_development_seal(reviewed_seal, bindings=bindings, ledger=self)
        _LIVE_PRIVATE_LEDGERS[id(self)] = (
            self,
            self._mint_identity,
            authorization,
            reviewed_seal,
        )

    def _serialized(self) -> bytes:
        return (
            json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )

    def _replace(self) -> None:
        _durable_replace(self.path, self._serialized())
        _refresh_ledger_receipt(self)

    def begin_access(self, split: str) -> _ManifestCapability:
        seal_registration = _LIVE_REVIEWED_DEVELOPMENT_SEALS.get(id(self._reviewed_seal))
        if seal_registration != (
            self._reviewed_seal,
            canonical_sha256(self._bindings),
            self,
        ):
            raise PermissionError("protected access lacks its live reviewed-development seal")
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
        capability = _ManifestCapability(
            _MANIFEST_CAPABILITY_AUTHORITY,
            ledger=self,
            ledger_mint_identity=self._mint_identity,
            split=split,
            seeds=self.MANIFESTS[split],
        )
        self._capabilities[split] = capability
        return capability

    def complete_split(self, split: str, result: Mapping[str, Any]) -> None:
        if split not in self.ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        state = self.record["splits"][split]
        if state["status"] != "materialization_started":
            raise RuntimeError(f"protected split {split!r} was not durably opened")
        capability = self._capabilities.get(split)
        if capability is None:
            raise RuntimeError(f"protected split {split!r} lacks manifest capability")
        capability.require_finished()
        _validate_split_result(result, split=split)
        _LIVE_MANIFEST_CAPABILITIES.pop(id(capability), None)
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
        _LIVE_PRIVATE_LEDGERS.pop(id(self), None)
        _LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(self._reviewed_seal), None)

    def record_error(
        self,
        error: BaseException,
        *,
        stopped_after: str,
        report_sha256: str | None = None,
    ) -> None:
        prior_status = self.record.get("status")
        for capability in self._capabilities.values():
            _LIVE_MANIFEST_CAPABILITIES.pop(id(capability), None)
        self.record["status"] = "error"
        self.record["outcome"] = "failed"
        self.record["status_before_error"] = prior_status
        self.record["stopped_after"] = stopped_after
        self.record["error"] = {"type": type(error).__name__, "message": str(error)}
        self.record["report_sha256"] = (
            None
            if report_sha256 is None
            else validated_sha256(report_sha256, label="failed qualification report SHA-256")
        )
        _validate_qualification_error_ledger_schema(self.record)
        try:
            self._replace()
        finally:
            _LIVE_PRIVATE_LEDGERS.pop(id(self), None)
            _LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(self._reviewed_seal), None)


def _model_state_sha256(model: OnlineWorldModel) -> str:
    if model.state_dict():
        raise RuntimeError("two-visible orbital-camera public runtime state_dict must remain empty")
    return EMPTY_MODEL_STATE_SHA256


SPLIT_RESULT_SCHEMA = frozenset(
    {
        "split",
        "seeds",
        "seed_manifest_sha256",
        "metrics",
        "failures",
        "passed",
        "optimizer_updates",
        "runtime_api",
        "scene_constructor",
    }
)
RUNTIME_API_SCHEMA = frozenset(
    {"packet_factory", "ingest_frames", "rollout_method", "horizons_seconds"}
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
        "model_state_sha256",
        "protocol",
        "publication_provenance",
        "development",
    }
)
DEVELOPMENT_REPORT_BASE_SCHEMA = frozenset(
    {
        "artifact_kind",
        "protocol",
        "source_provenance",
        "publication_provenance",
        "config_sha256",
        "scene_family_certificate",
        "development_ledger",
        "optimizer_updates",
        "protected_data_materialized",
        "passed",
        "review_ready",
        "stopped_after",
    }
)
DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA = frozenset(
    {
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_model_state_sha256",
        "checkpoint_roundtrip_state_sha256",
    }
)
QUALIFICATION_REPORT_BASE_SCHEMA = frozenset(
    {
        "artifact_kind",
        "protocol",
        "source_provenance",
        "publication_provenance",
        "config_sha256",
        "scene_family_certificate",
        "reviewed_checkpoint_sha256",
        "reviewed_development_report_sha256",
        "reviewed_development_ledger_sha256",
        "initial_model_state_sha256",
        "optimizer_updates",
        "passed",
        "protected_data_materialized",
        "stopped_after",
    }
)
LEDGER_SPLIT_STATE_SCHEMA = frozenset({"access_started", "status", "result_sha256"})
SOURCE_PROVENANCE_SCHEMA = frozenset(
    {"commit", "dirty", "worktree_fingerprint", "runtime_source_fingerprint"}
)
PUBLICATION_PROVENANCE_SCHEMA = frozenset(
    {"upstream_ref", "head_commit", "upstream_commit", "ahead", "behind"}
)
DEVELOPMENT_ERROR_LEDGER_SCHEMA = frozenset(
    {
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
        "status_before_error",
        "error",
        "report_sha256",
        "checkpoint_sha256",
    }
)
QUALIFICATION_ERROR_LEDGER_SCHEMA = frozenset(
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
        "status_before_error",
        "error",
        "report_sha256",
    }
)


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    if type(value) is not dict or set(value) != set(expected):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{label} schema differs; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_split_result(result: Mapping[str, Any], *, split: str) -> None:
    _require_exact_keys(result, SPLIT_RESULT_SCHEMA, label=f"{split} result")
    expected_seeds = list(_manifest_for_split(split))
    if result["split"] != split or result["seeds"] != expected_seeds:
        raise ValueError(f"{split} result has the wrong exact manifest")
    if result["seed_manifest_sha256"] != MANIFEST_SHA256[split]:
        raise ValueError(f"{split} result has the wrong manifest SHA-256")
    if type(result["optimizer_updates"]) is not int or result["optimizer_updates"] != 0:
        raise TypeError(f"{split} result optimizer_updates must be exact integer zero")
    if type(result["passed"]) is not bool:
        raise TypeError(f"{split} result passed must be an exact bool")
    if type(result["failures"]) is not list or any(
        type(failure) is not str for failure in result["failures"]
    ):
        raise TypeError(f"{split} result failures must be an exact string list")
    metrics = result["metrics"]
    if type(metrics) is not dict or set(metrics) != set(GATE_METRIC_SCHEMA):
        raise ValueError(f"{split} result metrics differ from the exact gate schema")
    failures = gate_failures(metrics)
    if result["failures"] != failures or result["passed"] is not (not failures):
        raise ValueError(f"{split} result gate evidence is internally inconsistent")
    runtime_api = result["runtime_api"]
    _require_exact_keys(runtime_api, RUNTIME_API_SCHEMA, label=f"{split} runtime API")
    expected_api = {
        "packet_factory": "make_rgbd_packet",
        "ingest_frames": list(HISTORY_FRAME_INDICES),
        "rollout_method": "OnlineWorldModel.predict",
        "horizons_seconds": list(HORIZONS_SECONDS),
    }
    if runtime_api != expected_api:
        raise ValueError(f"{split} result did not use the exact public runtime API")
    if result["scene_constructor"] != (
        "private_two_visible_orbital_camera_episode_with_full_frame_preflight"
    ):
        raise ValueError(f"{split} result names the wrong private constructor")


def _development_report_schema(report: Mapping[str, Any], *, error: bool) -> frozenset[str]:
    expected = set(DEVELOPMENT_REPORT_BASE_SCHEMA)
    if not error:
        expected.add("development")
        expected.update(DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA)
    else:
        if "development" in report:
            expected.add("development")
        checkpoint_present = DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA & set(report)
        if checkpoint_present:
            if checkpoint_present != DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA:
                raise ValueError("development report contains a partial checkpoint binding")
            expected.update(DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA)
    if error:
        expected.add("error")
    return frozenset(expected)


def _validate_development_report_schema(
    report: Mapping[str, Any],
    *,
    error: bool,
) -> None:
    _require_exact_keys(
        report,
        _development_report_schema(report, error=error),
        label="development report",
    )
    if type(report["passed"]) is not bool or type(report["review_ready"]) is not bool:
        raise TypeError("development report pass/review flags must be exact bools")
    _require_exact_keys(
        report["source_provenance"], SOURCE_PROVENANCE_SCHEMA, label="development source"
    )
    _require_exact_keys(
        report["publication_provenance"],
        PUBLICATION_PROVENANCE_SCHEMA,
        label="development publication",
    )
    if (
        report["artifact_kind"] != "rgbd_two_visible_orbital_camera_development"
        or canonical_sha256(report["protocol"]) != canonical_sha256(bridge_protocol())
        or report["config_sha256"] != FROZEN_CONFIG_SHA256
        or canonical_sha256(report["scene_family_certificate"])
        != canonical_sha256(scene_family_certificate())
        or report["development_ledger"] != str(development_ledger_path())
        or report["protected_data_materialized"] is not False
        or report["stopped_after"] != "development"
        or type(report["optimizer_updates"]) is not int
        or report["optimizer_updates"] != 0
    ):
        raise ValueError("development report has the wrong exact root evidence")
    if "development" in report:
        _validate_split_result(report["development"], split="development")
        if not error and (
            report["passed"] is not report["development"]["passed"]
            or report["review_ready"] is not report["development"]["passed"]
        ):
            raise ValueError("development report flags differ from its exact split result")
    if set(report) >= DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA:
        if report["checkpoint"] != str(canonical_checkpoint_path()):
            raise ValueError("development report names the wrong checkpoint path")
        for name in (
            "checkpoint_sha256",
            "checkpoint_model_state_sha256",
            "checkpoint_roundtrip_state_sha256",
        ):
            validated_sha256(report[name], label=f"development {name}")
        if (
            report["checkpoint_model_state_sha256"] != EMPTY_MODEL_STATE_SHA256
            or report["checkpoint_roundtrip_state_sha256"] != EMPTY_MODEL_STATE_SHA256
        ):
            raise ValueError("development report checkpoint state is not exactly empty")
    if error:
        _require_exact_keys(
            report["error"], frozenset({"type", "message"}), label="development error"
        )
        if report["passed"] or report["review_ready"]:
            raise ValueError("failed development report cannot remain passed or review-ready")


def _qualification_report_schema(
    report: Mapping[str, Any],
    *,
    error: bool,
) -> frozenset[str]:
    expected = set(QUALIFICATION_REPORT_BASE_SCHEMA)
    stopped_after = report.get("stopped_after")
    if stopped_after in _QualificationLedger.ORDER:
        index = _QualificationLedger.ORDER.index(stopped_after)
        expected.update(_QualificationLedger.ORDER[:index])
        if not error or stopped_after in report:
            expected.add(stopped_after)
    for optional in ("failures", "final_model_state_sha256", "qualification_ledger"):
        if optional in report:
            expected.add(optional)
    if error:
        expected.add("error")
    return frozenset(expected)


def _validate_qualification_report_schema(
    report: Mapping[str, Any],
    *,
    error: bool,
    terminal: bool,
) -> None:
    _require_exact_keys(
        report,
        _qualification_report_schema(report, error=error),
        label="qualification report",
    )
    if (
        type(report["passed"]) is not bool
        or type(report["protected_data_materialized"]) is not bool
    ):
        raise TypeError("qualification report pass/materialization flags must be exact bools")
    if type(report["optimizer_updates"]) is not int or report["optimizer_updates"] != 0:
        raise TypeError("qualification report optimizer_updates must be exact integer zero")
    _require_exact_keys(
        report["source_provenance"], SOURCE_PROVENANCE_SCHEMA, label="qualification source"
    )
    _require_exact_keys(
        report["publication_provenance"],
        PUBLICATION_PROVENANCE_SCHEMA,
        label="qualification publication",
    )
    if (
        report["artifact_kind"] != "rgbd_two_visible_orbital_camera_protected_qualification"
        or canonical_sha256(report["protocol"]) != canonical_sha256(bridge_protocol())
        or report["config_sha256"] != FROZEN_CONFIG_SHA256
        or canonical_sha256(report["scene_family_certificate"])
        != canonical_sha256(scene_family_certificate())
        or report["initial_model_state_sha256"] != EMPTY_MODEL_STATE_SHA256
    ):
        raise ValueError("qualification report has the wrong exact root evidence")
    for name in (
        "reviewed_checkpoint_sha256",
        "reviewed_development_report_sha256",
        "reviewed_development_ledger_sha256",
        "initial_model_state_sha256",
    ):
        validated_sha256(report[name], label=f"qualification {name}")
    stopped_after = report["stopped_after"]
    if stopped_after != "reviewed_development" and stopped_after not in _QualificationLedger.ORDER:
        raise ValueError("qualification report has an invalid stopped_after")
    if report["protected_data_materialized"] is not (stopped_after in _QualificationLedger.ORDER):
        raise ValueError("qualification materialization flag differs from durable stopped_after")
    if stopped_after in _QualificationLedger.ORDER:
        index = _QualificationLedger.ORDER.index(stopped_after)
        for split in _QualificationLedger.ORDER[:index]:
            _validate_split_result(report[split], split=split)
            if report[split]["passed"] is not True:
                raise ValueError("qualification report opened a split after a failed predecessor")
        if stopped_after in report:
            _validate_split_result(report[stopped_after], split=stopped_after)
        elif not error:
            raise ValueError("non-exception qualification report lacks its stopped split result")
    if report["passed"] and (
        stopped_after != "final_test"
        or any(
            split not in report or report[split]["passed"] is not True
            for split in _QualificationLedger.ORDER
        )
    ):
        raise ValueError("passed qualification report lacks three passed protected splits")
    if terminal:
        if stopped_after not in _QualificationLedger.ORDER:
            raise ValueError("terminal qualification must stop on a protected split")
        if not report["passed"] and report[stopped_after]["passed"] is not False:
            raise ValueError("failed qualification terminal split must itself have failed")
        required_terminal = {"failures", "final_model_state_sha256", "qualification_ledger"}
        if not required_terminal <= set(report):
            raise ValueError("terminal qualification report lacks exact terminal evidence")
        if report["qualification_ledger"] != str(qualification_ledger_path()):
            raise ValueError("qualification report names the wrong fixed ledger")
        if report["final_model_state_sha256"] != EMPTY_MODEL_STATE_SHA256:
            raise ValueError("qualification report final model state is not exactly empty")
        if type(report["failures"]) is not list or any(
            type(failure) is not str for failure in report["failures"]
        ):
            raise TypeError("qualification terminal failures must be an exact string list")
        expected_failures = [] if report["passed"] else list(report[stopped_after]["failures"])
        if report["failures"] != expected_failures:
            raise ValueError("qualification terminal failures differ from stopped split gates")
    if error:
        _require_exact_keys(
            report["error"], frozenset({"type", "message"}), label="qualification error"
        )
        if report["passed"]:
            raise ValueError("exception qualification report cannot remain passed")


def _validate_qualification_ledger_schema(
    record: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    terminal: bool,
) -> None:
    expected = {
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
    }
    if terminal:
        expected.add("report_sha256")
    _require_exact_keys(record, frozenset(expected), label="qualification ledger")
    if (
        record["artifact_kind"] != _QualificationLedger.ARTIFACT_KIND
        or type(record["architecture_attempt"]) is not int
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or type(record["maximum_architecture_attempts"]) is not int
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["order"] != list(_QualificationLedger.ORDER)
        or type(record["attempt_reserved"]) is not bool
        or record["attempt_reserved"] is not True
        or type(record["protected_data_materialized"]) is not bool
        or record["protected_data_materialized"] is not report["protected_data_materialized"]
    ):
        raise ValueError("qualification ledger fixed root values differ")
    bindings = record["bindings"]
    expected_binding_keys = {
        "protocol_sha256",
        "source_provenance",
        "publication_provenance",
        "config_sha256",
        "reviewed_checkpoint_sha256",
        "reviewed_development_report_sha256",
        "reviewed_development_ledger_sha256",
        "model_state_sha256",
        "certificate_sha256",
    }
    _require_exact_keys(bindings, frozenset(expected_binding_keys), label="qualification bindings")
    expected_bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": report["source_provenance"],
        "publication_provenance": report["publication_provenance"],
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": report["reviewed_checkpoint_sha256"],
        "reviewed_development_report_sha256": report["reviewed_development_report_sha256"],
        "reviewed_development_ledger_sha256": report["reviewed_development_ledger_sha256"],
        "model_state_sha256": report["initial_model_state_sha256"],
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    if bindings != expected_bindings:
        raise ValueError("qualification ledger binding values differ from report")
    splits = record["splits"]
    _require_exact_keys(splits, frozenset(_QualificationLedger.ORDER), label="qualification splits")
    for split, state in splits.items():
        _require_exact_keys(state, LEDGER_SPLIT_STATE_SCHEMA, label=f"{split} ledger state")
        if type(state["access_started"]) is not bool or type(state["status"]) is not str:
            raise TypeError(f"{split} ledger state has non-exact scalar types")
        if state["result_sha256"] is not None:
            validated_sha256(state["result_sha256"], label=f"{split} result SHA-256")
        if split in report:
            expected_status = "passed" if report[split]["passed"] else "failed"
            if (
                state["access_started"] is not True
                or state["status"] != expected_status
                or state["result_sha256"] != canonical_sha256(report[split])
            ):
                raise ValueError(f"qualification ledger {split} result differs from report")
        elif state != {
            "access_started": False,
            "status": "unopened",
            "result_sha256": None,
        }:
            raise ValueError(f"qualification ledger unopened {split} state differs")
    if record["outcome"] != ("passed" if report["passed"] else "failed"):
        raise ValueError("qualification ledger outcome differs from report")
    if record["stopped_after"] != report["stopped_after"]:
        raise ValueError("qualification ledger stopped_after differs from report")
    expected_status = "complete" if report["passed"] else "failed"
    if terminal:
        if record["status"] != expected_status:
            raise ValueError("terminal qualification ledger status differs from report")
        validated_sha256(record["report_sha256"], label="qualification report SHA-256")
        if record["report_sha256"] != sha256_bytes(_report_bytes(report)):
            raise ValueError("qualification ledger report digest differs from exact report bytes")
    elif record["status"] != "qualification_report_write_pending":
        raise ValueError("qualification ledger is not pending exact report write")


def _validate_development_live_ledger_schema(
    record: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    terminal: bool,
) -> None:
    expected = {
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
    }
    if terminal:
        expected.update({"report_sha256", "checkpoint_sha256"})
    _require_exact_keys(record, frozenset(expected), label="development ledger")
    if (
        record["artifact_kind"] != _DevelopmentLedger.ARTIFACT_KIND
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["attempt_reserved"] is not True
        or record["access_started"] is not True
        or record["development_data_materialized"] is not True
    ):
        raise ValueError("development ledger fixed root values differ")
    expected_bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": report["source_provenance"],
        "publication_provenance": report["publication_provenance"],
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_manifest_sha256": MANIFEST_SHA256["development"],
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    if record["bindings"] != expected_bindings:
        raise ValueError("development ledger binding values differ from report")
    development = report.get("development")
    if type(development) is not dict or record["result_sha256"] != canonical_sha256(development):
        raise ValueError("development ledger result binding differs from report")
    if record["outcome"] != ("passed" if development["passed"] else "failed"):
        raise ValueError("development ledger outcome differs from split result")
    if terminal:
        expected_status = "complete" if development["passed"] else "failed"
        if record["status"] != expected_status:
            raise ValueError("terminal development ledger status differs from result")
        validated_sha256(record["report_sha256"], label="development report SHA-256")
        validated_sha256(record["checkpoint_sha256"], label="development checkpoint SHA-256")
        if record["report_sha256"] != sha256_bytes(_report_bytes(report)):
            raise ValueError("development ledger report digest differs from exact report bytes")
        if record["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("development ledger checkpoint digest differs from report")
    elif record["status"] != "development_artifacts_pending":
        raise ValueError("development ledger is not pending exact artifacts")


def _validate_development_error_ledger_schema(record: Mapping[str, Any]) -> None:
    _require_exact_keys(
        record,
        DEVELOPMENT_ERROR_LEDGER_SCHEMA,
        label="failed development ledger",
    )
    if (
        record["artifact_kind"] != _DevelopmentLedger.ARTIFACT_KIND
        or type(record["architecture_attempt"]) is not int
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or type(record["maximum_architecture_attempts"]) is not int
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["attempt_reserved"] is not True
        or record["access_started"] is not True
        or record["development_data_materialized"] is not True
        or record["status"] != "error"
        or record["outcome"] != "failed"
        or type(record["status_before_error"]) is not str
    ):
        raise ValueError("failed development ledger fixed values differ from protocol")
    _require_exact_keys(
        record["bindings"],
        frozenset(
            {
                "protocol_sha256",
                "source_provenance",
                "publication_provenance",
                "config_sha256",
                "development_manifest_sha256",
                "certificate_sha256",
            }
        ),
        label="failed development bindings",
    )
    _require_exact_keys(
        record["error"],
        frozenset({"type", "message"}),
        label="failed development error",
    )
    for name in ("result_sha256", "report_sha256", "checkpoint_sha256"):
        if record[name] is not None:
            validated_sha256(record[name], label=f"failed development {name}")


def _validate_qualification_error_ledger_schema(record: Mapping[str, Any]) -> None:
    _require_exact_keys(
        record,
        QUALIFICATION_ERROR_LEDGER_SCHEMA,
        label="failed qualification ledger",
    )
    stopped_after = record["stopped_after"]
    if (
        record["artifact_kind"] != _QualificationLedger.ARTIFACT_KIND
        or type(record["architecture_attempt"]) is not int
        or record["architecture_attempt"] != ARCHITECTURE_ATTEMPT
        or type(record["maximum_architecture_attempts"]) is not int
        or record["maximum_architecture_attempts"] != MAX_ARCHITECTURE_ATTEMPTS
        or record["order"] != list(_QualificationLedger.ORDER)
        or record["attempt_reserved"] is not True
        or type(record["protected_data_materialized"]) is not bool
        or record["status"] != "error"
        or record["outcome"] != "failed"
        or type(record["status_before_error"]) is not str
        or stopped_after != "reviewed_development"
        and stopped_after not in _QualificationLedger.ORDER
        or record["protected_data_materialized"]
        is not (stopped_after in _QualificationLedger.ORDER)
    ):
        raise ValueError("failed qualification ledger fixed values differ from protocol")
    _require_exact_keys(
        record["bindings"],
        frozenset(
            {
                "protocol_sha256",
                "source_provenance",
                "publication_provenance",
                "config_sha256",
                "reviewed_checkpoint_sha256",
                "reviewed_development_report_sha256",
                "reviewed_development_ledger_sha256",
                "model_state_sha256",
                "certificate_sha256",
            }
        ),
        label="failed qualification bindings",
    )
    splits = record["splits"]
    _require_exact_keys(
        splits,
        frozenset(_QualificationLedger.ORDER),
        label="failed qualification splits",
    )
    for split, state in splits.items():
        _require_exact_keys(state, LEDGER_SPLIT_STATE_SCHEMA, label=f"failed {split} state")
        if type(state["access_started"]) is not bool or type(state["status"]) is not str:
            raise TypeError(f"failed {split} ledger state has non-exact scalar types")
        if state["result_sha256"] is not None:
            validated_sha256(state["result_sha256"], label=f"failed {split} result SHA-256")
    _require_exact_keys(
        record["error"],
        frozenset({"type", "message"}),
        label="failed qualification error",
    )
    if record["report_sha256"] is not None:
        validated_sha256(
            record["report_sha256"],
            label="failed qualification report SHA-256",
        )


def _record_development_exception(
    *,
    report_path: Path,
    report: dict[str, Any],
    ledger: _DevelopmentLedger,
    error: BaseException,
) -> None:
    """Persist fail-closed development evidence without skipping the ledger."""

    report["passed"] = False
    report["review_ready"] = False
    report["protected_data_materialized"] = False
    report["stopped_after"] = "development"
    if "development" in report:
        try:
            _validate_split_result(report["development"], split="development")
        except BaseException:
            report.pop("development", None)
    checkpoint_present = DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA & set(report)
    if checkpoint_present != DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA:
        for key in DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA:
            report.pop(key, None)
    elif checkpoint_present:
        try:
            if report["checkpoint"] != str(canonical_checkpoint_path()):
                raise ValueError("non-canonical failed checkpoint binding")
            for key in (
                "checkpoint_sha256",
                "checkpoint_model_state_sha256",
                "checkpoint_roundtrip_state_sha256",
            ):
                validated_sha256(report[key], label=f"failed development {key}")
        except BaseException:
            for key in DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA:
                report.pop(key, None)
    report["error"] = {"type": type(error).__name__, "message": str(error)}
    # Schema checking is diagnostic here: it must never prevent durable failed
    # evidence or the exact ledger error transition.
    with contextlib.suppress(BaseException):
        _validate_development_report_schema(report, error=True)
    report_sha256 = _persist_failed_report(report_path, report, label="development")
    ledger.record_error(error, report_sha256=report_sha256)


def _record_qualification_exception(
    *,
    report_path: Path,
    report: dict[str, Any],
    ledger: _QualificationLedger,
    error: BaseException,
) -> None:
    """Persist fail-closed qualification evidence without erasing access truth."""

    report["passed"] = False
    durable_materialized = ledger.record.get("protected_data_materialized") is True
    started = [
        split
        for split in _QualificationLedger.ORDER
        if ledger.record.get("splits", {}).get(split, {}).get("access_started") is True
    ]
    report["protected_data_materialized"] = durable_materialized
    report["stopped_after"] = started[-1] if started else "reviewed_development"
    for split in _QualificationLedger.ORDER:
        if split not in report:
            continue
        try:
            _validate_split_result(report[split], split=split)
        except BaseException:
            index = _QualificationLedger.ORDER.index(split)
            for discarded in _QualificationLedger.ORDER[index:]:
                report.pop(discarded, None)
            break
    report["error"] = {"type": type(error).__name__, "message": str(error)}
    with contextlib.suppress(BaseException):
        _validate_qualification_report_schema(report, error=True, terminal=False)
    report_sha256 = _persist_failed_report(report_path, report, label="qualification")
    ledger.record_error(
        error,
        stopped_after=str(report["stopped_after"]),
        report_sha256=report_sha256,
    )


def _validate_development_split(development: Mapping[str, Any]) -> None:
    _validate_split_result(development, split="development")
    if development["passed"] is not True:
        raise ValueError("reviewed development did not pass")


def validate_development_evidence(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind reviewed development to exact eventual clean harness source."""

    expected_schema = frozenset(
        {*DEVELOPMENT_REPORT_BASE_SCHEMA, *DEVELOPMENT_CHECKPOINT_REPORT_SCHEMA, "development"}
    )
    _require_exact_keys(report, expected_schema, label="reviewed development report")
    _validate_development_report_schema(report, error=False)
    if report.get("artifact_kind") != "rgbd_two_visible_orbital_camera_development":
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
    if canonical_sha256(report.get("scene_family_certificate")) != canonical_sha256(
        scene_family_certificate()
    ):
        raise ValueError("reviewed development scene certificate differs from frozen source")
    if canonical_sha256(report.get("source_provenance")) != canonical_sha256(source):
        raise ValueError("reviewed development source differs from current clean source")
    if canonical_sha256(report.get("publication_provenance")) != canonical_sha256(publication):
        raise ValueError("reviewed development publication differs from current upstream")
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
    publication: Mapping[str, Any],
    development: Mapping[str, Any],
) -> None:
    expected_schema = frozenset(
        {
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
    )
    _require_exact_keys(record, expected_schema, label="reviewed development ledger")
    expected_bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_manifest_sha256": canonical_sha256(list(DEVELOPMENT_SEEDS)),
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    if record.get("artifact_kind") != _DevelopmentLedger.ARTIFACT_KIND:
        raise ValueError("reviewed development ledger has the wrong artifact kind")
    if (
        type(record.get("architecture_attempt")) is not int
        or record.get("architecture_attempt") != ARCHITECTURE_ATTEMPT
        or record.get("maximum_architecture_attempts") != MAX_ARCHITECTURE_ATTEMPTS
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
    _require_canonical_path(path, canonical_checkpoint_path(), label="checkpoint")
    _validate_run_tree(
        frozenset({DEVELOPMENT_LEDGER_NAME}),
        stage="pre-development-checkpoint",
    )
    if _lexists(path) or _lexists(_atomic_temporary(path)):
        raise FileExistsError(f"two-visible orbital-camera checkpoint path must be fresh: {path}")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _durable_create(path, buffer.getvalue())


def validate_checkpoint_evidence(
    payload: Mapping[str, Any],
    *,
    config: OrpheusConfig,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
    development: Mapping[str, Any],
) -> None:
    _require_exact_keys(payload, CHECKPOINT_SCHEMA, label="reviewed checkpoint")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or model_state:
        raise ValueError("reviewed two-visible orbital-camera checkpoint model state must be empty")
    if type(payload.get("step")) is not int or payload.get("step") != 0:
        raise ValueError("reviewed two-visible orbital-camera checkpoint must be step zero")
    if payload.get("optimizer_state") is not None or payload.get("scheduler_state") is not None:
        raise ValueError("reviewed two-visible orbital-camera checkpoint must be optimizer-free")
    if (
        payload.get("project_version") != __version__
        or payload.get("specification_version") != SPECIFICATION_VERSION
        or payload.get("simulator_version") != SIMULATOR_VERSION
        or payload.get("device") != "cpu"
        or payload.get("precision") != "float32"
    ):
        raise ValueError("reviewed checkpoint versions/device/precision differ from protocol")
    if canonical_sha256(payload.get("config")) != canonical_sha256(
        config.to_dict()
    ) or canonical_sha256(payload.get("git")) != canonical_sha256(source):
        raise ValueError("reviewed two-visible orbital-camera checkpoint config/source differs")
    metrics = payload.get("metrics")
    _require_exact_keys(metrics, CHECKPOINT_METRICS_SCHEMA, label="checkpoint metrics")
    expected = {
        "artifact_kind": "rgbd_two_visible_orbital_camera_empty_model_state",
        "optimizer_updates": 0,
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "protocol": bridge_protocol(),
        "publication_provenance": publication,
        "development": development,
    }
    if canonical_sha256(metrics) != canonical_sha256(expected):
        raise ValueError(
            "reviewed two-visible orbital-camera checkpoint evidence differs from protocol"
        )
    roundtrip = new_public_model(config)
    roundtrip.load_state_dict(model_state, strict=True)
    if _model_state_sha256(roundtrip) != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("checkpoint roundtrip changed empty model state")


def _guard_frozen_inputs(
    *,
    source: Mapping[str, Any],
    publication: Mapping[str, Any],
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
        label="two-visible orbital-camera RGB-D execution guard",
    )
    if current != source:
        raise RuntimeError("source provenance changed during two-visible orbital-camera execution")
    current_publication = _validated_published_source(
        capture_published_source(REPOSITORY_ROOT),
        source=current,
        label="two-visible orbital-camera execution guard",
    )
    if current_publication != publication:
        raise RuntimeError("published upstream provenance changed during execution")
    _require_config_matches_frozen_path(config, config_path)
    protocol = bridge_protocol()
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol["protocol_sha256"] != canonical_sha256(unsigned):
        raise RuntimeError("two-visible orbital-camera protocol self-hash is inconsistent")
    certificate = scene_family_certificate()
    if certificate["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256:
        raise RuntimeError("two-visible orbital-camera certificate changed during execution")
    for split in ("development", "selector", "confirmation", "final_test"):
        if canonical_sha256(list(_manifest_for_split(split))) != MANIFEST_SHA256[split]:
            raise RuntimeError(f"{split} manifest differs from the frozen SHA-256")
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
        raise RuntimeError("two-visible orbital-camera public model state changed during execution")


def _require_single_thread_execution() -> None:
    if torch.get_num_threads() != 1:
        raise RuntimeError("orbital-camera run APIs require torch.get_num_threads()==1")


def run_development(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    source_provenance: Mapping[str, Any],
) -> int:
    """Evaluate development only and emit reviewable zero-state evidence."""

    _require_single_thread_execution()

    _require_canonical_path(
        config_path,
        _frozen_config_path(),
        label="config",
    )
    _require_canonical_path(report_path, canonical_development_report_path(), label="report")
    _require_canonical_path(checkpoint_path, canonical_checkpoint_path(), label="checkpoint")
    assert_rgbd_two_visible_orbital_camera_config(config)
    source = clean_source(source_provenance, label="two-visible orbital-camera RGB-D development")
    publication = _validated_published_source(
        capture_published_source(REPOSITORY_ROOT),
        source=source,
        label="two-visible orbital-camera RGB-D development",
    )
    _require_config_matches_frozen_path(config, config_path)
    _validate_run_tree(frozenset(), stage="pre-development")
    ledger_path = development_ledger_path()
    _validate_distinct_canonical_paths(
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
            raise FileExistsError(
                f"two-visible orbital-camera development artifact must be fresh: {path}"
            )
    protocol = bridge_protocol()
    certificate = scene_family_certificate()
    if certificate["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256:
        raise RuntimeError("orbital-camera development certificate differs from source")
    model = new_public_model(config)
    development_bindings = {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_manifest_sha256": canonical_sha256(list(DEVELOPMENT_SEEDS)),
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    authorization = _mint_run_authorization("development", development_bindings)
    ledger = _DevelopmentLedger(
        authorization,
        development_bindings,
    )
    _validate_run_tree(
        frozenset({DEVELOPMENT_LEDGER_NAME}),
        stage="development-ledger-created",
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_two_visible_orbital_camera_development",
        "protocol": protocol,
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "scene_family_certificate": certificate,
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
            capability=ledger.capability(),
        )
        ledger.complete_evaluation(development)
        report["development"] = development
        report["passed"] = development["passed"]
        report["review_ready"] = development["passed"]
        _guard_frozen_inputs(
            source=source,
            publication=publication,
            config=config,
            config_path=config_path,
            model=model,
        )
        checkpoint_metrics = {
            "artifact_kind": "rgbd_two_visible_orbital_camera_empty_model_state",
            "optimizer_updates": 0,
            "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
            "protocol": protocol,
            "publication_provenance": publication,
            "development": development,
        }
        _save_review_checkpoint(
            checkpoint_path,
            model=model,
            config=config,
            metrics=checkpoint_metrics,
            source=source,
        )
        _validate_run_tree(
            frozenset({DEVELOPMENT_LEDGER_NAME, CHECKPOINT_NAME}),
            stage="development-checkpoint-created",
        )
        checkpoint_contents = stable_read_bytes(checkpoint_path, label="development checkpoint")
        payload = _load_checkpoint_payload(checkpoint_contents)
        validate_checkpoint_evidence(
            payload,
            config=config,
            source=source,
            publication=publication,
            development=development,
        )
        checkpoint_digest = sha256_bytes(checkpoint_contents)
        report["checkpoint"] = str(checkpoint_path)
        report["checkpoint_sha256"] = checkpoint_digest
        report["checkpoint_model_state_sha256"] = EMPTY_MODEL_STATE_SHA256
        report["checkpoint_roundtrip_state_sha256"] = _model_state_sha256(new_public_model(config))
        _guard_frozen_inputs(
            source=source,
            publication=publication,
            config=config,
            config_path=config_path,
            model=model,
        )
        _validate_development_report_schema(report, error=False)
        _validate_development_live_ledger_schema(
            ledger.record,
            report=report,
            terminal=False,
        )
        _write_report_fresh(report_path, report)
        _validate_run_tree(DEVELOPMENT_ARTIFACT_NAMES, stage="post-development")
        written_report = stable_read_bytes(report_path, label="development report after write")
        parsed_report = json.loads(written_report)
        _validate_development_report_schema(parsed_report, error=False)
        report_digest = sha256_bytes(written_report)
        ledger.finish(
            report_sha256=report_digest,
            checkpoint_sha256=checkpoint_digest,
        )
        _validate_development_live_ledger_schema(
            ledger.record,
            report=report,
            terminal=True,
        )
        _validate_run_tree(DEVELOPMENT_ARTIFACT_NAMES, stage="development-terminal")
        return 0 if development["passed"] else 1
    except BaseException as error:
        _record_development_exception(
            report_path=report_path,
            report=report,
            ledger=ledger,
            error=error,
        )
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
    """Consume protected splits exactly once after reviewed development."""

    _require_single_thread_execution()

    _require_canonical_path(
        config_path,
        _frozen_config_path(),
        label="config",
    )
    _require_canonical_path(report_path, canonical_qualification_report_path(), label="report")
    _require_canonical_path(checkpoint_path, canonical_checkpoint_path(), label="checkpoint")
    _require_canonical_path(
        development_report_path,
        canonical_development_report_path(),
        label="development report",
    )
    assert_rgbd_two_visible_orbital_camera_config(config)
    source = clean_source(source_provenance, label="two-visible orbital-camera RGB-D qualification")
    publication = _validated_published_source(
        capture_published_source(REPOSITORY_ROOT),
        source=source,
        label="two-visible orbital-camera RGB-D qualification",
    )
    _require_config_matches_frozen_path(config, config_path)
    _validate_run_tree(DEVELOPMENT_ARTIFACT_NAMES, stage="pre-qualification")
    ledger_path = qualification_ledger_path()
    reviewed_development_ledger_path = development_ledger_path()
    checkpoint_digest = validated_sha256(
        reviewed_checkpoint_sha256, label="reviewed checkpoint SHA-256"
    )
    report_digest = validated_sha256(
        reviewed_report_sha256, label="reviewed development report SHA-256"
    )
    development_ledger_digest = validated_sha256(
        reviewed_development_ledger_sha256,
        label="externally reviewed development ledger SHA-256",
    )
    _validate_distinct_canonical_paths(
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
            raise FileExistsError(
                f"two-visible orbital-camera qualification artifact must be fresh: {path}"
            )
    checkpoint_contents = stable_read_bytes(checkpoint_path, label="reviewed checkpoint")
    if sha256_bytes(checkpoint_contents) != checkpoint_digest:
        raise ValueError("reviewed checkpoint hash does not match bytes")
    report_contents = stable_read_bytes(
        development_report_path, label="reviewed development report"
    )
    if sha256_bytes(report_contents) != report_digest:
        raise ValueError("reviewed development report hash does not match bytes")
    development_ledger_contents = stable_read_bytes(
        reviewed_development_ledger_path, label="reviewed development access ledger"
    )
    if sha256_bytes(development_ledger_contents) != development_ledger_digest:
        raise ValueError("reviewed development ledger hash does not match bytes")
    development_report = json.loads(report_contents)
    if not isinstance(development_report, Mapping):
        raise ValueError("reviewed development report must be a JSON object")
    development_ledger_record = json.loads(development_ledger_contents)
    if not isinstance(development_ledger_record, Mapping):
        raise ValueError("reviewed development ledger must be a JSON object")
    development = validate_development_evidence(
        development_report,
        checkpoint_sha256=checkpoint_digest,
        source=source,
        publication=publication,
    )
    validate_development_ledger(
        development_ledger_record,
        report=development_report,
        report_sha256=report_digest,
        checkpoint_sha256=checkpoint_digest,
        source=source,
        publication=publication,
        development=development,
    )
    payload = _load_checkpoint_payload(checkpoint_contents)
    validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        publication=publication,
        development=development,
    )
    model = new_public_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    initial_state = _model_state_sha256(model)
    _guard_frozen_inputs(
        source=source,
        publication=publication,
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
    qualification_bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": checkpoint_digest,
        "reviewed_development_report_sha256": report_digest,
        "reviewed_development_ledger_sha256": development_ledger_digest,
        "model_state_sha256": initial_state,
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    reviewed_seal = _mint_reviewed_development_seal(
        qualification_bindings,
        report=development_report,
        ledger_record=development_ledger_record,
        checkpoint_payload_value=payload,
        config=config,
        source=source,
        publication=publication,
    )
    authorization = _mint_run_authorization(
        "qualification",
        qualification_bindings,
        reviewed_seal=reviewed_seal,
    )
    ledger = _QualificationLedger(
        authorization,
        reviewed_seal,
        qualification_bindings,
    )
    _validate_run_tree(
        frozenset({*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_LEDGER_NAME}),
        stage="qualification-ledger-created",
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_two_visible_orbital_camera_protected_qualification",
        "protocol": bridge_protocol(),
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "scene_family_certificate": scene_family_certificate(),
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
                publication=publication,
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
            capability = ledger.begin_access(split)
            report["protected_data_materialized"] = True
            report["stopped_after"] = split
            result = _evaluate_seed_manifest(
                config,
                seeds,
                split=split,
                capability=capability,
            )
            report[split] = result
            ledger.complete_split(split, result)
            _guard_frozen_inputs(
                source=source,
                publication=publication,
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
            publication=publication,
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
        _validate_qualification_report_schema(report, error=False, terminal=True)
        _validate_qualification_ledger_schema(
            ledger.record,
            report=report,
            terminal=False,
        )
        _write_report_fresh(report_path, report)
        _validate_run_tree(QUALIFICATION_ARTIFACT_NAMES, stage="qualification-terminal")
        written_report = stable_read_bytes(report_path, label="completed qualification report")
        parsed_report = json.loads(written_report)
        _validate_qualification_report_schema(parsed_report, error=False, terminal=True)
        qualification_report_digest = sha256_bytes(written_report)
        _validate_qualification_ledger_schema(
            ledger.record,
            report=parsed_report,
            terminal=False,
        )
        ledger.finish(report_sha256=qualification_report_digest)
        _validate_qualification_ledger_schema(
            ledger.record,
            report=parsed_report,
            terminal=True,
        )
        _validate_run_tree(QUALIFICATION_ARTIFACT_NAMES, stage="qualification-finished")
    except BaseException as error:
        _record_qualification_exception(
            report_path=report_path,
            report=report,
            ledger=ledger,
            error=error,
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
    "FROZEN_CERTIFICATE_SHA256",
    "FROZEN_CONFIG_SHA256",
    "HISTORY_FRAME_INDICES",
    "HORIZONS_SECONDS",
    "MANIFEST_SHA256",
    "MAX_ARCHITECTURE_ATTEMPTS",
    "OBJECT_INDICES",
    "OPTIMIZER_UPDATES",
    "SELECTOR_SEEDS",
    "TARGET_FRAME_INDICES",
    "OrbitalCameraRGBDGates",
    "OrbitalCameraSceneSpecification",
    "VJP_COEFFICIENTS",
    "VJP_OUTPUTS",
    "assert_rgbd_two_visible_orbital_camera_config",
    "bridge_protocol",
    "development_ledger_path",
    "gate_failures",
    "new_public_model",
    "orbital_camera_frame",
    "preflight_two_visible_orbital_camera_episode",
    "qualification_ledger_path",
    "require_frozen_config",
    "run_development",
    "run_qualification",
    "scene_family_certificate",
    "scene_specification",
    "sha256_file",
    "validate_checkpoint_evidence",
    "validate_development_evidence",
]
