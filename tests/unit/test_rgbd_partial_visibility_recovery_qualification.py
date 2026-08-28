from __future__ import annotations

import inspect
import io
import json
import os
import pickle
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import world_model.simulator.physics as simulator_physics
import world_model.training.rgbd_partial_visibility_recovery_qualification as qualification
from world_model.observations import ObservationPacket
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.simulator.camera import CameraFrame, invert_rigid_transform, world_to_camera
from world_model.simulator.physics import SphereState
from world_model.simulator.renderer import render_spheres
from world_model.training.loop import make_rgbd_packet
from world_model.training.rgbd_partial_visibility_recovery_qualification import (
    AXIS_NAMES,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EMPTY_MODEL_STATE_SHA256,
    FINAL_TEST_SEEDS,
    FROZEN_CONFIG_SHA256,
    HORIZONS_SECONDS,
    INGEST_FRAME_INDICES,
    MANIFEST_SHA256,
    MISS_FRAME_INDICES,
    OBJECT_INDICES,
    SELECTOR_SEEDS,
    STRATUM_NAMES,
    VJP_COEFFICIENTS,
    VJP_OUTPUTS,
    DevelopmentLedger,
    QualificationLedger,
    _expected_slot_masks,
    _load_checkpoint_payload,
    _typed_canonical_equal,
    assert_rgbd_partial_visibility_config,
    bridge_protocol,
    canonical_sha256,
    gate_failures,
    scene_schedule,
    sha256_bytes,
    validate_checkpoint_evidence,
    validate_development_evidence,
    validate_development_ledger,
)
from world_model.utils.config import load_config

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_partial_visibility_recovery_cpu.yaml"
ATTEMPT1_ARCHIVE_FIXTURE = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "rgbd_partial_visibility_recovery" / "attempt-1"
)
_LIVE_ATTEMPT1_REJECTION_GUARD = qualification._require_attempt1_rejection
_EXPECTED_ATTEMPT2_ABSOLUTE_PRIMITIVES = (
    ("separated", 0, (22.625, 30.875, 4.875), (31.625, 31.875, 6.125)),
    ("separated", 1, (22.625, 30.875, 5.0), (31.625, 31.875, 6.125)),
    ("separated", 2, (22.625, 30.625, 4.75), (31.625, 31.875, 6.125)),
    ("separated", 3, (22.625, 30.375, 4.75), (31.625, 31.875, 6.125)),
    ("separated", 4, (22.625, 30.125, 4.875), (31.625, 31.875, 6.125)),
    ("separated", 5, (22.625, 29.875, 4.875), (31.625, 31.875, 6.125)),
    ("separated", 6, (22.625, 29.875, 5.0), (31.625, 31.875, 6.125)),
    ("separated", 7, (22.625, 29.625, 4.75), (31.625, 31.875, 6.125)),
    ("mild", 0, (28.625, 30.125, 4.875), (32.375, 32.625, 5.875)),
    ("mild", 1, (28.625, 30.125, 4.875), (32.375, 32.625, 6.0)),
    ("mild", 2, (28.125, 30.625, 4.875), (32.375, 32.625, 5.875)),
    ("mild", 3, (28.125, 30.625, 4.875), (32.375, 32.625, 6.0)),
    ("moderate", 0, (29.625, 30.125, 4.875), (32.375, 32.625, 5.875)),
    ("moderate", 1, (29.625, 30.125, 4.875), (32.375, 32.625, 6.0)),
    ("moderate", 2, (29.125, 30.625, 4.875), (32.375, 32.625, 5.875)),
    ("moderate", 3, (29.125, 30.625, 4.875), (32.375, 32.625, 6.0)),
)


@pytest.fixture(autouse=True)
def _use_tracked_attempt1_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the production guard against exact portable rejection bytes."""

    monkeypatch.setattr(
        qualification,
        "_attempt1_run_directory",
        lambda: ATTEMPT1_ARCHIVE_FIXTURE,
    )


def _attempt2_camera() -> CameraFrame:
    world_from_camera = qualification._fixed_world_from_camera()
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=world_from_camera,
        camera_from_world=invert_rigid_transform(world_from_camera),
        intrinsics=qualification._fixed_intrinsics(),
        position=world_from_camera[:3, 3],
        target=torch.tensor([0.0, 0.95, 0.0], dtype=torch.float32),
    )


def _attempt2_state(position: torch.Tensor, velocity: torch.Tensor) -> SphereState:
    return SphereState(
        object_id=torch.arange(2, dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=position,
        velocity=velocity,
        radius=torch.full((2, 1), 0.21, dtype=torch.float32),
        mass=torch.ones((2, 1), dtype=torch.float32),
        restitution=torch.full((2, 1), 0.7, dtype=torch.float32),
        drag=torch.full((2, 1), 0.05, dtype=torch.float32),
        friction=torch.full((2, 1), 0.2, dtype=torch.float32),
        albedo=torch.tensor([[0.92, 0.20, 0.14], [0.14, 0.84, 0.30]], dtype=torch.float32),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1).clone(),
        angular_velocity=torch.zeros((2, 3), dtype=torch.float32),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )


class _UnsafeCheckpointValue:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (eval, ("40 + 2",))


class _EmptyModel:
    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: object, *, strict: bool) -> None:
        assert state == {}
        assert strict is True


def _passing_metrics(*, split: str = "development") -> dict[str, float]:
    episode_count = float(len(qualification.MANIFESTS[split]))
    metrics: dict[str, float] = {}
    prefixes = (
        "",
        *(f"stratum/{name}" for name in STRATUM_NAMES),
        *(f"object/{index}" for index in OBJECT_INDICES),
        "role/front",
        "role/rear",
        "role/missed_target",
        "role/coobject",
    )
    for prefix in prefixes:
        label = f"{prefix}/" if prefix else ""
        for name in (
            "current_position_rmse_m",
            "current_velocity_rmse_mps",
            "maximum_position_error_growth_slope_mps",
            "early_stationary_additive_regression_m",
            "long_stationary_rmse_ratio",
            "zero_velocity_rmse_ratio",
        ):
            metrics[f"{label}{name}"] = 0.0
        for axis in AXIS_NAMES:
            metrics[f"{label}current_position_rmse_m/{axis}"] = 0.0
            metrics[f"{label}current_velocity_rmse_mps/{axis}"] = 0.0
        for horizon in HORIZONS_SECONDS:
            horizon_label = f"{horizon:.2f}"
            metrics[f"{label}horizon_{horizon_label}_position_rmse_m"] = 0.0
            metrics[f"{label}horizon_{horizon_label}_velocity_rmse_mps"] = 0.0
            for axis in AXIS_NAMES:
                metrics[f"{label}horizon_{horizon_label}_position_rmse_m/{axis}"] = 0.0
                metrics[f"{label}horizon_{horizon_label}_velocity_rmse_mps/{axis}"] = 0.0

    metrics.update(
        {
            "miss_frame_position_rmse_m": 0.0,
            "identity_switch_count": 0.0,
            "persistent_id_mismatch_count": 0.0,
            "association_ambiguous_pair_count": 0.0,
            "false_miss_association_count": 0.0,
            "false_birth_count": 0.0,
            "death_count": 0.0,
            "measurement_validity_mismatch_count": 0.0,
            "association_validity_mismatch_count": 0.0,
            "false_velocity_evidence_count": 0.0,
            "direct_position_field_count": 0.0,
            "public_rollout_output_alias_count": 0.0,
            "predicted_unobservable_count": 0.0,
            "association_pair_coverage": 1.0,
            "identity_coverage": 1.0,
            "minimum_hungarian_margin": 1.0,
            "minimum_position_assignment_margin_m": 1.0,
            "minimum_matched_appearance_cosine": 1.0,
            "minimum_cross_appearance_cosine_distance": 1.0,
            "reacquisition_latency_frames_max": 1.0,
            "maximum_missed_steps": 1.0,
            "final_missed_steps_max": 0.0,
            "missed_target_steps_before_min": 0.0,
            "missed_target_steps_before_max": 0.0,
            "missed_target_steps_at_miss_min": 1.0,
            "missed_target_steps_at_miss_max": 1.0,
            "missed_target_steps_at_recovery_min": 0.0,
            "missed_target_steps_at_recovery_max": 0.0,
            "missed_coobject_steps_min": 0.0,
            "missed_coobject_steps_max": 0.0,
            "recovery_mode_value_min": 0.0,
            "recovery_mode_value_max": 0.0,
            "recovery_free_mode_mismatch_count": 0.0,
            "missed_step_trace_mismatch_count": 0.0,
            "runtime_free_mode_mismatch_count": 0.0,
            "rollout_free_mode_mismatch_count": 0.0,
            "active_fraction": 1.0,
            "rollout_active_fraction": 1.0,
            "physical_palette_swap_fraction": 0.5,
            "birth_slot_physical_zero_fraction": 0.5,
            "unique_scene_specification_fraction": 1.0,
            "manifest_episode_count": episode_count,
            "history_sample_count_min": 16.0,
            "history_sample_count_max": 16.0,
            "history_no_miss_valid_count_min": 16.0,
            "history_no_miss_valid_count_max": 16.0,
            "history_missed_target_valid_count_min": 15.0,
            "history_missed_target_valid_count_max": 15.0,
            "history_missed_coobject_valid_count_min": 16.0,
            "history_missed_coobject_valid_count_max": 16.0,
            "history_no_miss_slot_count": episode_count,
            "history_missed_target_slot_count": episode_count / 2.0,
            "history_missed_coobject_slot_count": episode_count / 2.0,
            "history_sample_mask_mismatch_count": 0.0,
            "history_valid_mask_mismatch_count": 0.0,
            "history_latest_valid_mismatch_count": 0.0,
            "history_timestamp_max_abs_error_seconds": 0.0,
            "history_span_seconds_min": 0.75,
            "history_span_seconds_max": 0.75,
            "expected_velocity_evidence_coverage": 1.0,
            "associator_call_count_per_batch_min": 18.0,
            "associator_call_count_per_batch_max": 18.0,
            "associator_call_frame_mismatch_count": 0.0,
            "direct_velocity_call_count_per_batch_min": 3.0,
            "direct_velocity_call_count_per_batch_max": 3.0,
            "direct_velocity_call_frame_mismatch_count": 0.0,
            "position_owner_count_min": 1.0,
            "position_owner_count_max": 1.0,
            "direct_velocity_position_change_max_abs_m": 0.0,
            "direct_metric_position_owner_max_abs_m": 0.0,
            "minimum_observed_support_fraction": 1.0,
            "maximum_surface_fit_residual_relative_rms": 0.0,
            "maximum_full_silhouette_overlap_fraction": 0.0,
            "maximum_surface_radius_relative_error": 0.0,
            "maximum_surface_fit_condition_number": 1.0,
            "minimum_fitted_boundary_clearance_pixels": 10.0,
            "minimum_full_silhouette_radius_pixels": 1.0,
            "maximum_full_silhouette_gap_abs_pixels": 0.0,
            "geometry_nonfinite_count": 0.0,
            "invalid_geometry_nonzero_count": 0.0,
            "invalid_pair_geometry_nonzero_count": 0.0,
            "maximum_predicted_visibility_error": 0.0,
            "missed_variance_increment_max_abs_error": 0.0,
            "missed_variance_increment_min": 0.08,
            "missed_variance_increment_max": 0.08,
            "missed_variance_increment_count": episode_count / 2.0,
            "coobject_variance_increment_max_abs": 0.0,
            "missed_variance_mask_mismatch_count": 0.0,
            "missed_variance_inflation_count": episode_count / 2.0,
            "semigroup_position_max_abs_m": 0.0,
            "semigroup_velocity_max_abs_mps": 0.0,
            "public_direct_position_max_abs_m": 0.0,
            "public_direct_velocity_max_abs_mps": 0.0,
            "analytic_position_agreement_max_abs_m": 0.0,
            "analytic_velocity_agreement_max_abs_mps": 0.0,
            "public_query_time_max_abs_seconds": 0.0,
            "ingested_frame_count_min": 18.0,
            "ingested_frame_count_max": 18.0,
            "public_predict_calls_per_batch_min": 1.0,
            "public_predict_calls_per_batch_max": 1.0,
            "gradient_audit_scene_count": 4.0,
            "gradient_audit_unique_scene_fraction": 1.0,
            "perception_latency_seconds": 0.0,
            "state_only_rollout_latency_seconds": 0.0,
            "persistent_runtime_tensor_state_bytes_max": 0.0,
            "process_max_rss_bytes": 0.0,
            "process_rss_delta_bytes": 0.0,
            "learned_parameter_count": 0.0,
            "learned_parameter_bytes": 0.0,
            "module_tensor_buffer_count": 0.0,
            "persistent_module_state_key_count": 0.0,
            "persistent_module_state_bytes": 0.0,
            "optimizer_updates": 0.0,
            "optimizer_state_entry_count": 0.0,
            "scheduler_state_entry_count": 0.0,
            "rng_state_entry_count": 0.0,
        }
    )
    metrics.update({f"stratum_fraction/{name}": 0.25 for name in STRATUM_NAMES})
    metrics.update(
        {
            "severity_fraction/separated": 0.50,
            "severity_fraction/mild": 0.25,
            "severity_fraction/moderate": 0.25,
        }
    )

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
        scoped = {
            f"{prefix}preflight_minimum_boundary_clearance_pixels": 10.0,
            f"{prefix}preflight_minimum_world_surface_gap_m": 1.0,
            f"{prefix}preflight_minimum_world_boundary_clearance_m": 1.0,
            f"{prefix}preflight_minimum_palette_cosine_distance": 1.0,
            f"{prefix}preflight_event_count": 0.0,
            f"{prefix}miss_unchanged_max_abs": 0.0,
            f"{prefix}miss_changed_pixels": 4.0 if missed is True else 0.0,
        }
        if geometry == "separated":
            scoped.update(
                {
                    f"{prefix}preflight_minimum_silhouette_gap_pixels": 2.0,
                    f"{prefix}preflight_maximum_silhouette_gap_pixels": 2.0,
                    f"{prefix}preflight_minimum_rear_visible_fraction": 1.0,
                    f"{prefix}preflight_maximum_rear_visible_fraction": 1.0,
                    f"{prefix}preflight_minimum_overlap_pixels": 0.0,
                    f"{prefix}preflight_maximum_overlap_pixels": 0.0,
                }
            )
        elif geometry == "mild":
            scoped.update(
                {
                    f"{prefix}preflight_minimum_silhouette_gap_pixels": -1.75,
                    f"{prefix}preflight_maximum_silhouette_gap_pixels": -0.35,
                    f"{prefix}preflight_minimum_rear_visible_fraction": 0.80,
                    f"{prefix}preflight_maximum_rear_visible_fraction": 0.95,
                    f"{prefix}preflight_minimum_overlap_pixels": 1.0,
                    f"{prefix}preflight_maximum_overlap_pixels": 1.0,
                }
            )
        elif geometry == "moderate":
            scoped.update(
                {
                    f"{prefix}preflight_minimum_silhouette_gap_pixels": -2.75,
                    f"{prefix}preflight_maximum_silhouette_gap_pixels": -1.25,
                    f"{prefix}preflight_minimum_rear_visible_fraction": 0.60,
                    f"{prefix}preflight_maximum_rear_visible_fraction": 0.79,
                    f"{prefix}preflight_minimum_overlap_pixels": 1.0,
                    f"{prefix}preflight_maximum_overlap_pixels": 1.0,
                }
            )
        elif geometry == "partial":
            scoped.update(
                {
                    f"{prefix}preflight_minimum_silhouette_gap_pixels": -2.75,
                    f"{prefix}preflight_maximum_silhouette_gap_pixels": -0.35,
                    f"{prefix}preflight_minimum_rear_visible_fraction": 0.60,
                    f"{prefix}preflight_maximum_rear_visible_fraction": 0.95,
                    f"{prefix}preflight_minimum_overlap_pixels": 1.0,
                    f"{prefix}preflight_maximum_overlap_pixels": 1.0,
                }
            )
        else:
            scoped.update(
                {
                    f"{prefix}preflight_minimum_silhouette_gap_pixels": -2.75,
                    f"{prefix}preflight_maximum_silhouette_gap_pixels": 2.0,
                    f"{prefix}preflight_minimum_rear_visible_fraction": 0.60,
                    f"{prefix}preflight_maximum_rear_visible_fraction": 1.0,
                    f"{prefix}preflight_minimum_overlap_pixels": 0.0,
                    f"{prefix}preflight_maximum_overlap_pixels": 1.0,
                }
            )
        metrics.update(scoped)

    for scene_index, audit_offset in enumerate((0, 1, 6, 15)):
        schedule = scene_schedule(DEVELOPMENT_SEEDS[audit_offset])
        metrics[f"gradient_audit_manifest_offset/scene_{scene_index}"] = float(audit_offset)
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
            ] = float(object_index)
            expected_frames = 16.0
            if schedule.miss_frame is not None and schedule.missed_slot == object_index:
                expected_frames = 15.0
            for output_name in VJP_OUTPUTS:
                for modality in ("rgb", "depth"):
                    suffix = f"scene_{scene_index}/object_{object_index}/{output_name}/{modality}"
                    metrics[f"gradient_total_l1/{suffix}"] = 1.0
                    metrics[f"gradient_expected_min_l1/{suffix}"] = 1.0
                    metrics[f"gradient_visible_region_expected_min_l1/{suffix}"] = 1.0
                    metrics[f"gradient_unexpected_max_l1/{suffix}"] = 0.0
                    metrics[f"gradient_cross_scene_max_l1/{suffix}"] = 0.0
                    count = 1.0 if output_name == "current_position" else expected_frames
                    metrics[f"gradient_supported_frames/{suffix}"] = count
                    metrics[f"gradient_expected_frames/{suffix}"] = count
    return metrics


def _source() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "dirty": False,
        "worktree_fingerprint": "2" * 64,
        "runtime_source_fingerprint": "3" * 64,
    }


def _development_result() -> dict[str, object]:
    return {
        "split": "development",
        "seeds": list(DEVELOPMENT_SEEDS),
        "seed_manifest_sha256": MANIFEST_SHA256["development"],
        "scene_parameter_signature_sha256": qualification.SCENE_PARAMETER_SIGNATURE_SHA256[
            "development"
        ],
        "metrics": _passing_metrics(),
        "failures": [],
        "passed": True,
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


def _manifest_result(split: str) -> dict[str, object]:
    result = _development_result()
    result["split"] = split
    result["seeds"] = list(qualification.MANIFESTS[split])
    result["seed_manifest_sha256"] = MANIFEST_SHA256[split]
    result["scene_parameter_signature_sha256"] = qualification.SCENE_PARAMETER_SIGNATURE_SHA256[
        split
    ]
    result["metrics"] = _passing_metrics(split=split)
    return result


def _reviewed_report(ledger_path: Path) -> tuple[dict[str, object], dict[str, object], str]:
    source = _source()
    checkpoint_sha256 = "4" * 64
    report: dict[str, object] = {
        "artifact_kind": "rgbd_partial_visibility_development",
        "protocol": json.loads(json.dumps(bridge_protocol())),
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_ledger": str(ledger_path),
        "development": _development_result(),
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "passed": True,
        "review_ready": True,
        "stopped_after": "development",
        "checkpoint": str(qualification.canonical_artifact_paths()["development_checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "checkpoint_roundtrip_state_sha256": EMPTY_MODEL_STATE_SHA256,
    }
    return report, source, checkpoint_sha256


def _consume_authorization(
    authorization: object,
    *,
    split: str,
    seeds: tuple[int, ...],
) -> None:
    authorization.begin_manifest(split, seeds)  # type: ignore[attr-defined]
    for seed in seeds:
        authorization.authorize_seed(seed)  # type: ignore[attr-defined]
    authorization.finish_manifest()  # type: ignore[attr-defined]


def test_protocol_freezes_manifests_schedule_vjp_and_resource_contract() -> None:
    namespaces = (DEVELOPMENT_SEEDS, SELECTOR_SEEDS, CONFIRMATION_SEEDS, FINAL_TEST_SEEDS)
    assert tuple(map(len, namespaces)) == (32, 24, 24, 48)
    assert tuple(range(57_000_000, 57_000_032)) == DEVELOPMENT_SEEDS
    assert tuple(range(58_000_000, 58_000_024)) == SELECTOR_SEEDS
    assert tuple(range(59_000_000, 59_000_024)) == CONFIRMATION_SEEDS
    assert tuple(range(60_000_000, 60_000_048)) == FINAL_TEST_SEEDS
    flattened = tuple(seed for namespace in namespaces for seed in namespace)
    assert len(flattened) == len(set(flattened))
    for split, seeds in qualification.MANIFESTS.items():
        assert canonical_sha256(list(seeds)) == MANIFEST_SHA256[split]

    protocol = bridge_protocol()
    stated = protocol.pop("protocol_sha256")
    assert stated == canonical_sha256(protocol)
    assert protocol["runtime"]["ingest_frames"] == list(range(18))
    assert protocol["runtime"]["live_history_frames"] == list(range(2, 18))
    assert protocol["runtime"]["anchor_frame"] == 17
    assert protocol["runtime"]["miss_frames"] == [15, 16]
    assert protocol["runtime"]["target_frames"] == [19, 22, 27, 37, 57]
    assert protocol["runtime"]["max_missing_rows"] == 1
    assert protocol["runtime"]["require_latest_valid"] is True
    assert protocol["differentiability"]["coefficients"] == list(VJP_COEFFICIENTS)
    assert protocol["differentiability"]["audit_offsets"] == [0, 1, 6, 15]
    assert protocol["scene_parameter_signature_sha256"] == (
        qualification.SCENE_PARAMETER_SIGNATURE_SHA256
    )
    assert protocol["evidence"]["constructor_single_use_authorization"] is True
    assert protocol["evidence"]["report_before_terminal_ledger_digest"] is True
    assert protocol["evidence"]["weights_only_empty_state_checkpoint"] is True
    assert protocol["optimizer_updates"] == 0
    assert protocol["name"] == "rgbd_partial_visibility_recovery_v2"
    assert protocol["architecture_version"] == 2
    assert protocol["architecture_attempt"] == 2
    assert protocol["maximum_architecture_attempts"] == 2


@pytest.mark.parametrize(
    "seeds",
    (DEVELOPMENT_SEEDS, SELECTOR_SEEDS, CONFIRMATION_SEEDS, FINAL_TEST_SEEDS),
)
def test_scene_schedule_is_balanced_and_bounded_for_every_split(
    seeds: tuple[int, ...],
) -> None:
    schedules = [scene_schedule(seed) for seed in seeds]
    assert Counter(item.stratum for item in schedules) == {
        name: len(seeds) // 4 for name in STRATUM_NAMES
    }
    assert sum(item.palette_swapped for item in schedules) == len(seeds) // 2
    missed = [item for item in schedules if item.miss_frame is not None]
    assert len(missed) == len(seeds) // 2
    assert Counter(item.miss_frame for item in missed) == {
        frame: len(missed) // 2 for frame in MISS_FRAME_INDICES
    }
    partial = [item for item in schedules if item.partial]
    assert {item.severity for item in partial} == {"mild", "moderate"}
    assert Counter(item.rear_slot for item in partial) == {
        0: len(partial) // 2,
        1: len(partial) // 2,
    }
    partial_miss = [item for item in schedules if item.stratum == "partial_one_miss"]
    assert all(item.missed_slot == item.rear_slot for item in partial_miss)
    assert all(item.miss_frame is None for item in schedules if "no_miss" in item.stratum)


@pytest.mark.parametrize("bad_seed", (True, 57_000_000.0, 56_999_999, 60_000_048))
def test_scene_schedule_rejects_noncanonical_seed(bad_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        scene_schedule(bad_seed)  # type: ignore[arg-type]


def test_pure_scene_parameter_signatures_are_unique_and_manifest_bound() -> None:
    all_signatures: list[str] = []
    for split, seeds in qualification.MANIFESTS.items():
        signatures = [qualification.scene_parameter_signature(seed) for seed in seeds]
        assert len(signatures) == len(set(signatures))
        assert canonical_sha256(signatures) == qualification.SCENE_PARAMETER_SIGNATURE_SHA256[split]
        all_signatures.extend(signatures)
    assert len(all_signatures) == len(set(all_signatures))


def test_attempt2_admissibility_certificate_freezes_every_template_cell() -> None:
    certificate = qualification.attempt2_admissibility_certificate()
    assert certificate.template_cell_count == 128
    assert certificate.physical_record_count == 128
    assert certificate.physics_substep_count == 342
    assert certificate.template_table_sha256 == qualification.ATTEMPT2_TEMPLATE_TABLE_SHA256
    assert (
        certificate.absolute_primitive_table_sha256
        == qualification.ATTEMPT2_ABSOLUTE_PRIMITIVE_TABLE_SHA256
    )
    assert (
        certificate.ordered_physical_state_sha256
        == qualification.ATTEMPT2_ORDERED_PHYSICAL_STATE_SHA256
    )
    assert certificate.physical_state_set_sha256 == qualification.ATTEMPT2_PHYSICAL_STATE_SET_SHA256
    assert (
        certificate.unordered_geometry_set_sha256
        == qualification.ATTEMPT2_UNORDERED_GEOMETRY_SET_SHA256
    )
    assert certificate.world_trajectory_sha256 == qualification.ATTEMPT2_WORLD_TRAJECTORY_SHA256
    assert certificate.renderer_trace_sha256 == qualification.ATTEMPT2_RENDERER_TRACE_SHA256
    assert certificate.minimum_discriminant_abs_margin >= 5.0e-5
    assert certificate.minimum_overlap_depth_margin_m >= 0.80
    assert certificate.maximum_projected_centre_drift_pixels <= 2.0e-5
    assert (
        0.0
        < certificate.maximum_camera_conjugacy_error_m
        <= (qualification.MAXIMUM_CAMERA_CONJUGACY_ERROR_M)
    )
    assert certificate.minimum_initial_speed_mps >= 0.035
    assert certificate.maximum_initial_speed_mps <= 0.065
    assert certificate.minimum_world_surface_gap_m >= 0.50
    assert certificate.minimum_world_boundary_clearance_m >= 0.10
    assert certificate.minimum_image_boundary_clearance_pixels >= 2.0
    assert certificate.minimum_separated_silhouette_gap_margin_pixels >= 1.0
    assert certificate.minimum_partial_silhouette_band_margin_pixels >= 0.40
    assert certificate.minimum_current_visibility_band_margin_fraction >= 0.049
    assert certificate.minimum_one_pixel_visibility_band_margin_fraction >= 0.0
    assert certificate.minimum_full_support_pixels >= 18
    assert certificate.minimum_visible_support_pixels >= 14
    assert certificate.minimum_local_miss_pixels >= 14


def test_attempt2_absolute_primitives_and_physical_state_digests_are_independent() -> None:
    expected = [
        {
            "kind": kind,
            "index": index,
            "front": list(front),
            "rear": list(rear),
        }
        for kind, index, front, rear in _EXPECTED_ATTEMPT2_ABSOLUTE_PRIMITIVES
    ]
    assert qualification._attempt2_absolute_primitive_table() == expected
    assert canonical_sha256(expected) == qualification.ATTEMPT2_ABSOLUTE_PRIMITIVE_TABLE_SHA256
    ordered_records, unordered_geometry_records = (
        qualification._attempt2_independent_physical_state_records()
    )
    assert len(ordered_records) == len(set(ordered_records)) == 128
    assert len(unordered_geometry_records) == 128
    assert (
        sha256_bytes(b"".join(ordered_records))
        == qualification.ATTEMPT2_ORDERED_PHYSICAL_STATE_SHA256
    )
    assert (
        sha256_bytes(b"".join(sorted(ordered_records)))
        == qualification.ATTEMPT2_PHYSICAL_STATE_SET_SHA256
    )
    assert (
        sha256_bytes(b"".join(sorted(unordered_geometry_records)))
        == qualification.ATTEMPT2_UNORDERED_GEOMETRY_SET_SHA256
    )


def test_attempt2_ray_certificate_matches_public_renderer_for_all_cells_and_frames() -> None:
    camera = _attempt2_camera()
    rotation = camera.world_from_camera[:3, :3]
    translation = camera.world_from_camera[:3, 3]
    for template in qualification.ATTEMPT2_TEMPLATE_TABLE:
        for symmetry in range(qualification.TEMPLATE_SYMMETRY_COUNT):
            points, velocities = qualification._template_camera_state(
                template,
                symmetry,
                intrinsics=camera.intrinsics,
            )
            world_positions, world_velocities, world_substep_positions, world_substep_velocities = (
                qualification._exact_float32_world_trajectory(
                    points,
                    velocities,
                    world_from_camera=camera.world_from_camera,
                )
            )
            assert torch.equal(
                world_positions[0],
                points @ rotation.transpose(0, 1) + translation,
            )
            assert torch.equal(world_velocities[0], velocities @ rotation.transpose(0, 1))
            assert torch.equal(world_positions, world_substep_positions[::6])
            assert torch.equal(world_velocities, world_substep_velocities[::6])
            camera_positions = world_to_camera(world_positions, camera.camera_from_world)
            for frame_points, frame_world_position, frame_world_velocity in zip(
                camera_positions,
                world_positions,
                world_velocities,
                strict=True,
            ):
                state = _attempt2_state(frame_world_position, frame_world_velocity)
                rendered = render_spheres(state, camera, (64, 64))
                certified = qualification._renderer_ray_geometry(
                    frame_points,
                    camera.intrinsics,
                )
                assert torch.equal(rendered.full_mask, certified["full_mask"])
                assert torch.equal(
                    rendered.instance_slot_map,
                    certified["instance_slot_map"],
                )
                assert torch.equal(rendered.visible_mask, certified["visible_mask"])


def test_attempt2_certified_world_trace_is_exact_public_solver_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    resolved = qualification.SphereWorldConfig.from_config(config)
    camera = _attempt2_camera()
    expected_traces: list[tuple[torch.Tensor, torch.Tensor]] = []
    for template in qualification.ATTEMPT2_TEMPLATE_TABLE:
        for symmetry in range(qualification.TEMPLATE_SYMMETRY_COUNT):
            points, velocities = qualification._template_camera_state(
                template,
                symmetry,
                intrinsics=camera.intrinsics,
            )
            _, _, positions, world_velocities = qualification._exact_float32_world_trajectory(
                points,
                velocities,
                world_from_camera=camera.world_from_camera,
            )
            expected_traces.append((positions, world_velocities))

    original_integrate = simulator_physics._integrate_free_motion_exact
    original_boundary = simulator_physics.resolve_axis_aligned_boundaries
    original_pair = simulator_physics.resolve_sphere_sphere_collisions
    active_trace: dict[str, object] = {}

    def tracking_integrate(
        position: torch.Tensor,
        velocity: torch.Tensor,
        drag: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
        movable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_positions = active_trace["positions"]
        expected_velocities = active_trace["velocities"]
        cursor = active_trace["cursor"]
        assert isinstance(expected_positions, torch.Tensor)
        assert isinstance(expected_velocities, torch.Tensor)
        assert isinstance(cursor, int)
        assert torch.equal(position, expected_positions[cursor])
        assert torch.equal(velocity, expected_velocities[cursor])
        result = original_integrate(position, velocity, drag, gravity, dt, movable)
        cursor += 1
        assert torch.equal(result[0], expected_positions[cursor])
        assert torch.equal(result[1], expected_velocities[cursor])
        active_trace["cursor"] = cursor
        return result

    def tracking_boundary(*args: object, **kwargs: object) -> object:
        result = original_boundary(*args, **kwargs)
        position, velocity = args[:2]
        assert isinstance(position, torch.Tensor)
        assert isinstance(velocity, torch.Tensor)
        assert torch.equal(result.position, position)
        assert torch.equal(result.velocity, velocity)
        assert not bool(result.contact.any())
        assert not bool(result.collision.any())
        assert not bool(result.impulse_magnitude.ne(0).any())
        assert not bool(result.penetration.ne(0).any())
        return result

    def tracking_pair(*args: object, **kwargs: object) -> object:
        result = original_pair(*args, **kwargs)
        position, velocity = args[:2]
        assert isinstance(position, torch.Tensor)
        assert isinstance(velocity, torch.Tensor)
        assert torch.equal(result.position, position)
        assert torch.equal(result.velocity, velocity)
        assert not bool(result.contact.any())
        assert not bool(result.collision.any())
        assert not bool(result.impulse_magnitude.ne(0).any())
        assert not bool(result.penetration.ne(0).any())
        return result

    monkeypatch.setattr(simulator_physics, "_integrate_free_motion_exact", tracking_integrate)
    monkeypatch.setattr(simulator_physics, "resolve_axis_aligned_boundaries", tracking_boundary)
    monkeypatch.setattr(simulator_physics, "resolve_sphere_sphere_collisions", tracking_pair)
    physics_config = simulator_physics.PhysicsConfig(
        gravity=resolved.gravity,
        bounds=resolved.world_bounds,
        max_substep=1.0 / resolved.physics_rate,
        solver_iterations=resolved.solver_iterations,
    )
    for expected_positions, expected_velocities in expected_traces:
        state = _attempt2_state(expected_positions[0], expected_velocities[0])
        active_trace.update(
            positions=expected_positions,
            velocities=expected_velocities,
            cursor=0,
        )
        for frame_index in range(57):
            state, events = simulator_physics.advance_spheres(
                state,
                resolved.observation_dt,
                physics_config,
                external_impulse=torch.zeros_like(state.velocity),
            )
            assert events.substeps == 6
            assert not bool(events.pair_contact.any())
            assert not bool(events.pair_collision.any())
            assert not bool(events.pair_impulse.ne(0).any())
            assert not bool(events.pair_penetration.ne(0).any())
            assert not bool(events.boundary_contact.any())
            assert not bool(events.boundary_collision.any())
            assert not bool(events.boundary_impulse.ne(0).any())
            assert not bool(events.boundary_penetration.ne(0).any())
            assert not bool(events.collision.any())
            assert not bool(events.contact.any())
            assert not bool(events.sleeping.any())
            assert not bool(events.external_impulse.ne(0).any())
            assert bool(events.first_event_offset.eq(-1.0).all())
            substep_index = (frame_index + 1) * 6
            assert torch.equal(state.position, expected_positions[substep_index])
            assert torch.equal(state.velocity, expected_velocities[substep_index])
        assert active_trace["cursor"] == 342


def test_all_attempt2_primitives_are_feasible_through_public_rgbd_runtime() -> None:
    config = load_config(CONFIG_PATH)
    camera = _attempt2_camera()
    for template in qualification.ATTEMPT2_TEMPLATE_TABLE:
        model = qualification.new_public_model(config)
        points, velocities = qualification._template_camera_state(
            template,
            0,
            intrinsics=camera.intrinsics,
        )
        world_positions, world_velocities, _, _ = qualification._exact_float32_world_trajectory(
            points,
            velocities,
            world_from_camera=camera.world_from_camera,
        )
        posterior = None
        for frame_index in qualification.INGEST_FRAME_INDICES:
            state = _attempt2_state(
                world_positions[frame_index],
                world_velocities[frame_index],
            )
            rendered = render_spheres(state, camera, (64, 64))
            posterior = model.ingest(
                ObservationPacket(
                    modality="rgbd",
                    sensor_id="camera0:rgbd",
                    timestamp=frame_index / 20.0,
                    payload={
                        "rgb": rendered.rgb.unsqueeze(0),
                        "depth": rendered.depth_buffer[None, None],
                    },
                    calibration={
                        "world_from_camera": camera.world_from_camera.unsqueeze(0),
                        "intrinsics": camera.intrinsics.unsqueeze(0),
                    },
                    frame_id="camera:camera0:rgbd",
                    metadata={"image_size": (64, 64)},
                )
            )
            assert model.last_measurements is not None
            assert bool(model.last_measurements.measurement_mask.all())
        assert posterior is not None
        assert bool(posterior.objects.active.all())
        assert bool(posterior.objects.missed_steps.eq(0).all())
        history = model.state.temporal_histories[qualification.RUNTIME_STREAM_KEY]
        assert isinstance(history, RGBDTemporalPositionHistory)
        assert bool(history.sample_mask.all())
        assert bool(history.valid_mask.all())


def test_attempt1_rejection_archive_is_exact_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_contents = (ATTEMPT1_ARCHIVE_FIXTURE / "development_report.json").read_bytes()
    ledger_contents = (ATTEMPT1_ARCHIVE_FIXTURE / "development_attempt_1_access.json").read_bytes()
    qualification._validate_attempt1_rejection_bytes(report_contents, ledger_contents)

    archive = tmp_path / "attempt1"
    archive.mkdir()
    (archive / "development_report.json").write_bytes(report_contents)
    (archive / "development_attempt_1_access.json").write_bytes(ledger_contents)
    monkeypatch.setattr(qualification, "_attempt1_run_directory", lambda: archive)
    _LIVE_ATTEMPT1_REJECTION_GUARD()
    ledger = archive / "development_attempt_1_access.json"
    ledger.write_bytes(ledger.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="ledger bytes changed"):
        _LIVE_ATTEMPT1_REJECTION_GUARD()


def test_expected_masks_isolate_exactly_one_target_miss() -> None:
    schedules = [scene_schedule(DEVELOPMENT_SEEDS[index]) for index in range(4)]
    mapping = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    measurement, velocity = _expected_slot_masks(schedules, mapping)
    assert measurement.shape == velocity.shape == (4, 18, 2)
    assert not bool(measurement[:, 0].any())
    for row, schedule in enumerate(schedules):
        invalid = torch.nonzero(~measurement[row, 1:], as_tuple=False)
        if schedule.miss_frame is None:
            assert invalid.numel() == 0
            continue
        assert invalid.shape == (1, 2)
        assert int(invalid[0, 0]) + 1 == schedule.miss_frame
        runtime_slot = int(invalid[0, 1])
        assert not bool(velocity[row, schedule.miss_frame, runtime_slot])
        assert bool(velocity[row, schedule.miss_frame, 1 - runtime_slot])


def test_frozen_config_bytes_and_semantics_match_release_candidate() -> None:
    assert sha256_bytes(CONFIG_PATH.read_bytes()) == FROZEN_CONFIG_SHA256
    config = load_config(CONFIG_PATH)
    assert_rgbd_partial_visibility_config(config)
    assert config.model.rgbd.bounded_partial_visibility is True
    assert config.model.rgbd.minimum_observed_support_fraction == 0.35
    assert config.model.rgbd.maximum_surface_residual_relative_rms == 0.05
    assert config.model.rgbd.maximum_full_silhouette_overlap_fraction == 0.60
    assert config.model.rgbd.max_missing_rows == 1
    assert config.model.rgbd.require_latest_valid is True
    assert config.model.filter.missed_variance_growth == 0.08


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("rgbd", "bounded_partial_visibility", False),
        ("rgbd", "minimum_observed_support_fraction", 0.34),
        ("rgbd", "maximum_surface_residual_relative_rms", 0.06),
        ("rgbd", "maximum_full_silhouette_overlap_fraction", 0.61),
        ("rgbd", "max_missing_rows", 0),
        ("rgbd", "require_latest_valid", False),
        ("filter", "missed_variance_growth", 0.07),
    ),
)
def test_config_rejects_partial_recovery_semantic_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    config = load_config(CONFIG_PATH)
    changed = replace(
        config,
        model=replace(
            config.model,
            **{section: replace(getattr(config.model, section), **{field: value})},
        ),
    )
    with pytest.raises(ValueError):
        assert_rgbd_partial_visibility_config(changed)


def test_runtime_packet_excludes_all_renderer_truth_and_schedule_fields() -> None:
    batch: dict[str, Any] = {
        "rgb": torch.zeros(2, 18, 3, 4, 4),
        "depth": torch.zeros(2, 18, 1, 4, 4),
        "timestamps": (torch.arange(18, dtype=torch.float32) / 20.0).expand(2, -1).clone(),
        "camera": {
            "world_from_camera": torch.eye(4).expand(2, 18, 4, 4).clone(),
            "intrinsics": torch.eye(3).expand(2, 18, 3, 3).clone(),
        },
        "instance_slot_map": object(),
        "visible_fraction": object(),
        "labels": object(),
        "miss_schedule": object(),
    }
    packet = make_rgbd_packet(batch, 15)
    assert set(packet.payload) == {"rgb", "depth"}
    assert set(packet.calibration) == {"world_from_camera", "intrinsics"}
    serialized_keys = set(packet.payload) | set(packet.calibration) | set(packet.metadata)
    assert not serialized_keys & {
        "instance_slot_map",
        "visible_fraction",
        "labels",
        "miss_schedule",
    }


def test_accuracy_object_mask_isolated_selection_without_mutating_row_mask() -> None:
    current_position = torch.zeros(2, 2, 3, dtype=torch.float64)
    current_position[0, 0] = 100.0
    current_position[0, 1, 0] = 3.0
    row_mask = torch.tensor([True, False])
    original_row_mask = row_mask.clone()
    object_mask = torch.tensor([[False, True], [True, False]])
    metrics: dict[str, float] = {}

    qualification._accuracy_metrics(
        metrics,
        prefix="synthetic",
        current_position_error=current_position,
        current_velocity_error=torch.zeros_like(current_position),
        future_position_error=torch.zeros(2, 2, 5, 3, dtype=torch.float64),
        future_velocity_error=torch.zeros(2, 2, 5, 3, dtype=torch.float64),
        stationary_position_error=torch.zeros(2, 2, 5, 3, dtype=torch.float64),
        zero_velocity_error=torch.zeros_like(current_position),
        row_mask=row_mask,
        object_mask=object_mask,
    )

    assert torch.equal(row_mask, original_row_mask)
    assert metrics["synthetic/current_position_rmse_m"] == pytest.approx(3.0**0.5)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    (
        ("current_position_rmse_m", 1.0),
        ("stratum/partial_one_miss/horizon_2.00_position_rmse_m", 1.0),
        ("role/rear/current_velocity_rmse_mps/z", 1.0),
        ("role/missed_target/zero_velocity_rmse_ratio", 1.0),
        ("identity_switch_count", 1.0),
        ("identity_coverage", 0.99),
        ("association_pair_coverage", 0.99),
        ("false_miss_association_count", 1.0),
        ("reacquisition_latency_frames_max", 2.0),
        ("maximum_missed_steps", 2.0),
        ("missed_step_trace_mismatch_count", 1.0),
        ("runtime_free_mode_mismatch_count", 1.0),
        ("rollout_free_mode_mismatch_count", 1.0),
        ("final_missed_steps_max", 1.0),
        ("birth_slot_physical_zero_fraction", 0.49),
        ("stratum_fraction/partial_one_miss", 0.24),
        ("history_sample_count_min", 15.0),
        ("history_valid_mask_mismatch_count", 1.0),
        ("associator_call_count_per_batch_min", 17.0),
        ("direct_velocity_call_frame_mismatch_count", 1.0),
        ("expected_velocity_evidence_coverage", 0.99),
        ("minimum_observed_support_fraction", 0.34),
        ("maximum_surface_fit_residual_relative_rms", 0.051),
        ("maximum_full_silhouette_overlap_fraction", 0.61),
        ("maximum_surface_radius_relative_error", 0.051),
        ("maximum_surface_fit_condition_number", 100.1),
        ("geometry_nonfinite_count", 1.0),
        ("maximum_predicted_visibility_error", 0.16),
        ("missed_variance_increment_max_abs_error", 1.0e-5),
        ("coobject_variance_increment_max_abs", 1.1e-6),
        ("semigroup_position_max_abs_m", 1.0e-4),
        ("persistent_runtime_tensor_state_bytes_max", 65_537.0),
        ("learned_parameter_count", 1.0),
    ),
)
def test_gate_recomputation_fails_closed_across_every_surface(
    key: str,
    bad_value: float,
) -> None:
    metrics = _passing_metrics()
    assert gate_failures(metrics) == []
    metrics[key] = bad_value
    assert any(failure.startswith(f"{key}:") for failure in gate_failures(metrics))


def test_evaluator_reads_the_published_surface_fit_radius_diagnostic() -> None:
    evaluator_source = inspect.getsource(qualification._evaluate_seed_manifest)
    assert 'get("surface_fit_radius_relative_error")' in evaluator_source
    assert 'get("surface_radius_relative_error")' not in evaluator_source


@pytest.mark.parametrize("bad_value", (None, True, float("nan"), float("inf"), "0"))
def test_gate_recomputation_rejects_missing_nonfinite_or_nonnumeric(bad_value: object) -> None:
    metrics: dict[str, object] = _passing_metrics()
    if bad_value is None:
        metrics.pop("current_position_rmse_m")
    else:
        metrics["current_position_rmse_m"] = bad_value
    assert gate_failures(metrics)[0].startswith("current_position_rmse_m:")


@pytest.mark.parametrize(
    "missing_key",
    (
        "current_position_rmse_m",
        "history_valid_mask_mismatch_count",
        "severity/mild/preflight_minimum_silhouette_gap_pixels",
        "gradient_audit_manifest_offset/scene_2",
        "gradient_visible_region_expected_min_l1/scene_3/object_1/horizon_2.00_position/depth",
    ),
)
def test_gate_metric_schema_rejects_missing_and_extra_keys(missing_key: str) -> None:
    metrics = _passing_metrics()
    assert gate_failures(metrics, split="development") == []

    missing = dict(metrics)
    missing.pop(missing_key)
    assert any(missing_key in failure for failure in gate_failures(missing, split="development"))

    extra = dict(metrics)
    extra["not_in_frozen_metric_schema"] = 0.0
    assert "metric_schema:unexpected:not_in_frozen_metric_schema" in gate_failures(
        extra, split="development"
    )


@pytest.mark.parametrize("split", tuple(qualification.MANIFESTS))
def test_passing_metric_fixture_is_split_bound(split: str) -> None:
    assert gate_failures(_passing_metrics(split=split), split=split) == []
    wrong_split = "final_test" if split != "final_test" else "development"
    assert gate_failures(_passing_metrics(split=split), split=wrong_split)


@pytest.mark.parametrize(
    "key",
    (
        "gradient_total_l1/scene_3/object_1/horizon_2.00_position/depth",
        "gradient_expected_min_l1/scene_2/object_0/current_velocity/rgb",
    ),
)
def test_vjp_gate_rejects_detached_expected_support(key: str) -> None:
    metrics = _passing_metrics()
    metrics[key] = 0.0
    assert any(key in failure for failure in gate_failures(metrics))


@pytest.mark.parametrize(
    "key",
    (
        "gradient_unexpected_max_l1/scene_0/object_0/current_position/rgb",
        "gradient_cross_scene_max_l1/scene_1/object_1/horizon_0.50_velocity/depth",
    ),
)
def test_vjp_gate_rejects_leakage(key: str) -> None:
    metrics = _passing_metrics()
    metrics[key] = 1.0e-12
    assert any(key in failure for failure in gate_failures(metrics))


def test_vjp_gate_rejects_wrong_frame_count_for_missed_target() -> None:
    metrics = _passing_metrics()
    schedule = scene_schedule(DEVELOPMENT_SEEDS[15])
    assert schedule.missed_slot is not None
    key = (
        "gradient_supported_frames/scene_3/"
        f"object_{schedule.missed_slot}/horizon_2.00_position/depth"
    )
    metrics[key] += 1.0
    assert any(key in failure for failure in gate_failures(metrics))


def test_typed_canonical_comparison_rejects_python_type_coercions() -> None:
    assert _typed_canonical_equal({"value": [0, False]}, {"value": [0, False]})
    assert not _typed_canonical_equal({"value": False}, {"value": 0})
    assert not _typed_canonical_equal({"value": 0.0}, {"value": 0})
    assert not _typed_canonical_equal({"value": (1, 2)}, {"value": [1, 2]})
    assert not _typed_canonical_equal({"value": float("nan")}, {"value": float("nan")})


def test_constructor_authorization_and_evaluator_order_are_static_boundaries() -> None:
    assert "construct_partial_visibility_episode" not in qualification.__all__
    assert "evaluate_seed_manifest" not in qualification.__all__
    constructor_source = inspect.getsource(qualification._construct_partial_visibility_episode)
    assert constructor_source.index("_require_attempt1_rejection()") < constructor_source.index(
        "_require_ledger_minted_authorization"
    )
    assert constructor_source.index(
        "authorization.authorize_seed(seed)"
    ) < constructor_source.index("SphereWorldConfig.from_config")
    evaluator_source = inspect.getsource(qualification._evaluate_seed_manifest)
    assert evaluator_source.index("_require_attempt1_rejection()") < evaluator_source.index(
        "_require_ledger_minted_authorization"
    )
    assert evaluator_source.index("authorization.begin_manifest") < evaluator_source.index(
        "_construct_partial_visibility_episode"
    )
    assert evaluator_source.index("authorization.finish_manifest") > evaluator_source.index(
        "_construct_partial_visibility_episode"
    )


def test_fake_duck_authorizations_fail_before_constructor_or_evaluator_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class DuckAuthorization:
        def begin_manifest(self, *_args: object) -> None:
            events.append("duck:begin")

        def authorize_seed(self, *_args: object) -> None:
            events.append("duck:seed")

        def finish_manifest(self) -> None:
            events.append("duck:finish")

    def rejection_guard() -> None:
        events.append("attempt1:checked")

    def forbidden_work(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("authorization rejection must precede materialization work")

    monkeypatch.setattr(qualification, "_require_attempt1_rejection", rejection_guard)
    monkeypatch.setattr(
        qualification,
        "assert_rgbd_partial_visibility_config",
        forbidden_work,
    )
    duck = DuckAuthorization()
    with pytest.raises(PermissionError, match="exact ledger-minted authorization"):
        qualification._construct_partial_visibility_episode(
            object(),  # type: ignore[arg-type]
            DEVELOPMENT_SEEDS[0],
            authorization=duck,  # type: ignore[arg-type]
        )
    with pytest.raises(PermissionError, match="exact ledger-minted authorization"):
        qualification._evaluate_seed_manifest(
            object(),  # type: ignore[arg-type]
            DEVELOPMENT_SEEDS,
            split="development",
            authorization=duck,  # type: ignore[arg-type]
        )
    assert events == ["attempt1:checked", "attempt1:checked"]


def test_private_token_rejects_noncanonical_authorization_contracts(tmp_path: Path) -> None:
    issuer = object.__new__(DevelopmentLedger)
    issuer.path = qualification.development_ledger_path()
    issuer._authorization_mint = object()
    issuer._authorization = None
    canonical = {
        "issuer": issuer,
        "mint": issuer._authorization_mint,
        "split": "development",
        "seeds": DEVELOPMENT_SEEDS,
        "ledger_path": issuer.path,
        "ledger_kind": DevelopmentLedger.ARTIFACT_KIND,
        "receipt_sha256": "0" * 64,
    }
    invalid_contracts = (
        {"ledger_path": tmp_path / "attacker-ledger.json"},
        {"ledger_kind": QualificationLedger.ARTIFACT_KIND},
        {"split": "selector", "seeds": SELECTOR_SEEDS},
        {"seeds": SELECTOR_SEEDS},
    )
    for invalid in invalid_contracts:
        with pytest.raises(PermissionError, match="canonical"):
            qualification._ManifestAccessAuthorization(
                qualification._MANIFEST_ACCESS_AUTHORITY,
                **(canonical | invalid),
            )


def test_unregistered_exact_authorization_identity_fails_at_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = object.__new__(DevelopmentLedger)
    issuer.path = qualification.development_ledger_path()
    issuer._authorization_mint = object()
    issuer._authorization = None
    authorization = qualification._ManifestAccessAuthorization(
        qualification._MANIFEST_ACCESS_AUTHORITY,
        issuer=issuer,
        mint=issuer._authorization_mint,
        split="development",
        seeds=DEVELOPMENT_SEEDS,
        ledger_path=issuer.path,
        ledger_kind=DevelopmentLedger.ARTIFACT_KIND,
        receipt_sha256="0" * 64,
    )
    events: list[str] = []
    monkeypatch.setattr(
        qualification,
        "_require_attempt1_rejection",
        lambda: events.append("attempt1:checked"),
    )
    monkeypatch.setattr(
        qualification,
        "assert_rgbd_partial_visibility_config",
        lambda *_args: events.append("forbidden:config"),
    )
    with pytest.raises(PermissionError, match="canonical ledger-minted capability"):
        qualification._construct_partial_visibility_episode(
            object(),  # type: ignore[arg-type]
            DEVELOPMENT_SEEDS[0],
            authorization=authorization,
        )
    with pytest.raises(PermissionError, match="canonical ledger-minted capability"):
        qualification._evaluate_seed_manifest(
            object(),  # type: ignore[arg-type]
            DEVELOPMENT_SEEDS,
            split="development",
            authorization=authorization,
        )
    assert events == ["attempt1:checked", "attempt1:checked"]


def test_exactly_once_ledgers_are_durable_ordered_and_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development_path = tmp_path / "development-access.json"
    protected_path = tmp_path / "qualification-access.json"
    monkeypatch.setattr(qualification, "development_ledger_path", lambda: development_path)
    monkeypatch.setattr(qualification, "qualification_ledger_path", lambda: protected_path)

    development = DevelopmentLedger(
        {"protocol_sha256": "0" * 64},
        authority=qualification._LEDGER_CONSTRUCTION_AUTHORITY,
    )
    assert json.loads(development_path.read_text())["status"] == (
        "development_materialization_started"
    )
    authorization = development.authorization()
    with pytest.raises(RuntimeError, match="cannot be issued twice"):
        development.authorization()
    _consume_authorization(authorization, split="development", seeds=DEVELOPMENT_SEEDS)
    development_result = _manifest_result("development")
    authorization.seal_result(qualification._EVALUATOR_RESULT_AUTHORITY, development_result)
    development.complete_evaluation(development_result)
    development.finish(report_sha256="a" * 64, checkpoint_sha256="b" * 64)
    assert json.loads(development_path.read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        DevelopmentLedger(
            {"protocol_sha256": "0" * 64},
            authority=qualification._LEDGER_CONSTRUCTION_AUTHORITY,
        )

    protected = QualificationLedger(
        {"protocol_sha256": "0" * 64},
        authority=qualification._LEDGER_CONSTRUCTION_AUTHORITY,
    )
    with pytest.raises(RuntimeError, match="must remain unopened"):
        protected.begin_access("final_test")
    for split in protected.ORDER:
        authorization = protected.begin_access(split)
        durable = json.loads(protected_path.read_text())
        assert durable["status"] == f"{split}_materialization_started"
        with pytest.raises(RuntimeError, match="cannot be opened twice"):
            protected.begin_access(split)
        _consume_authorization(authorization, split=split, seeds=qualification.MANIFESTS[split])
        result = _manifest_result(split)
        authorization.seal_result(qualification._EVALUATOR_RESULT_AUTHORITY, result)
        protected.complete_split(split, result)
    protected.prepare_report(passed=True, stopped_after="final_test")
    assert json.loads(protected_path.read_text())["status"] == (
        "qualification_report_write_pending"
    )
    protected.finish(report_sha256="c" * 64)
    final = json.loads(protected_path.read_text())
    assert final["status"] == "complete"
    assert final["protected_data_materialized"] is True


def test_authorization_fails_if_durable_receipt_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "qualification-access.json"
    monkeypatch.setattr(qualification, "qualification_ledger_path", lambda: path)
    ledger = QualificationLedger(
        {"protocol_sha256": "0" * 64},
        authority=qualification._LEDGER_CONSTRUCTION_AUTHORITY,
    )
    authorization = ledger.begin_access("selector")
    record = json.loads(path.read_text())
    record["status"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="started-ledger receipt"):
        authorization.begin_manifest("selector", SELECTOR_SEEDS)


def test_alias_symlink_and_hardlink_artifacts_fail_closed(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="artifact paths alias"):
        qualification.validate_distinct_paths(
            {"report": report, "temporary": report.with_suffix(".json.tmp")},
            atomic_writers=("report",),
        )
    source = tmp_path / "source.json"
    alias = tmp_path / "alias.json"
    source.write_text("{}")
    os.link(source, alias)
    with pytest.raises(ValueError, match="hard-link alias"):
        qualification.validate_distinct_paths({"source": source, "alias": alias}, atomic_writers=())
    with pytest.raises(ValueError, match="single-link"):
        qualification._single_link_read_bytes(source, label="hard-linked evidence")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="symbolic link"):
        qualification.stable_read_bytes(symlink, label="adversarial evidence")


def test_canonical_five_artifact_inventory_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    paths = qualification.canonical_artifact_paths()
    expected = ("development_report", "development_checkpoint", "development_ledger")
    for name in expected:
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(name.encode())
    qualification._validate_artifact_inventory(allowed_existing=expected)
    (paths["development_report"].parent / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected qualification artifact"):
        qualification._validate_artifact_inventory(allowed_existing=expected)


def test_canonical_artifact_directory_rejects_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    (repository / "runs").mkdir(parents=True)
    outside.mkdir()
    (repository / qualification.RUN_RELATIVE_PATH).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", repository)
    with pytest.raises(ValueError, match="must not be a symlink"):
        qualification.canonical_artifact_paths()


def test_reviewed_development_uses_typed_exact_canonical_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "development-access.json"
    monkeypatch.setattr(qualification, "development_ledger_path", lambda: ledger_path)
    report, source, checkpoint_sha256 = _reviewed_report(ledger_path)
    development = validate_development_evidence(
        report, checkpoint_sha256=checkpoint_sha256, source=source
    )
    assert development["passed"] is True
    tampered = json.loads(json.dumps(report))
    tampered["optimizer_updates"] = False
    with pytest.raises(ValueError, match="execution boundary"):
        validate_development_evidence(tampered, checkpoint_sha256=checkpoint_sha256, source=source)
    tampered = json.loads(json.dumps(report))
    tampered["protocol"]["architecture_version"] = 1.0
    with pytest.raises(ValueError, match="protocol differs"):
        validate_development_evidence(tampered, checkpoint_sha256=checkpoint_sha256, source=source)
    tampered = json.loads(json.dumps(report))
    tampered["unexpected"] = None
    with pytest.raises(ValueError, match="exact passed schema"):
        validate_development_evidence(tampered, checkpoint_sha256=checkpoint_sha256, source=source)
    tampered = json.loads(json.dumps(report))
    tampered["checkpoint_roundtrip_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="roundtrip state"):
        validate_development_evidence(tampered, checkpoint_sha256=checkpoint_sha256, source=source)


def test_reviewed_development_ledger_binds_exact_report_result_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "development-access.json"
    monkeypatch.setattr(qualification, "development_ledger_path", lambda: ledger_path)
    report, source, checkpoint_sha256 = _reviewed_report(ledger_path)
    development = report["development"]
    assert isinstance(development, dict)
    report_sha256 = "5" * 64
    record = {
        "artifact_kind": DevelopmentLedger.ARTIFACT_KIND,
        "architecture_attempt": 2,
        "maximum_architecture_attempts": 2,
        "bindings": {
            "protocol_sha256": bridge_protocol()["protocol_sha256"],
            "source_provenance": json.loads(json.dumps(source)),
            "config_sha256": FROZEN_CONFIG_SHA256,
            "development_manifest_sha256": MANIFEST_SHA256["development"],
        },
        "attempt_reserved": True,
        "access_started": True,
        "development_data_materialized": True,
        "status": "complete",
        "outcome": "passed",
        "result_sha256": canonical_sha256(development),
        "report_sha256": report_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    validate_development_ledger(
        record,
        report=report,
        report_sha256=report_sha256,
        checkpoint_sha256=checkpoint_sha256,
        source=source,
        development=development,
    )
    wrong_attempt = json.loads(json.dumps(record))
    wrong_attempt["architecture_attempt"] = True
    with pytest.raises(ValueError, match="attempt differs"):
        validate_development_ledger(
            wrong_attempt,
            report=report,
            report_sha256=report_sha256,
            checkpoint_sha256=checkpoint_sha256,
            source=source,
            development=development,
        )
    record["bindings"]["source_provenance"]["dirty"] = 0
    with pytest.raises(ValueError, match="bindings differ"):
        validate_development_ledger(
            record,
            report=report,
            report_sha256=report_sha256,
            checkpoint_sha256=checkpoint_sha256,
            source=source,
            development=development,
        )


def test_checkpoint_is_weights_only_rng_free_empty_state_and_exactly_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    source = _source()
    development = _development_result()
    monkeypatch.setattr(qualification, "new_public_model", lambda _config: _EmptyModel())
    payload: dict[str, object] = {
        "model_state": {},
        "step": 0,
        "optimizer_state": None,
        "scheduler_state": None,
        "config": config.to_dict(),
        "git": source,
        "metrics": {
            "artifact_kind": "rgbd_partial_visibility_empty_model_state",
            "optimizer_updates": 0,
            "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
            "protocol": bridge_protocol(),
            "development": development,
        },
        "project_version": qualification.__version__,
        "specification_version": qualification.SPECIFICATION_VERSION,
        "simulator_version": qualification.SIMULATOR_VERSION,
        "device": "cpu",
        "precision": "float32",
    }
    validate_checkpoint_evidence(payload, config=config, source=source, development=development)
    payload["metrics"]["optimizer_updates"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="evidence differs"):
        validate_checkpoint_evidence(payload, config=config, source=source, development=development)


def test_restricted_checkpoint_loader_rejects_untrusted_pickle_global() -> None:
    buffer = io.BytesIO()
    torch.save({"unsafe": _UnsafeCheckpointValue()}, buffer)
    with pytest.raises(pickle.UnpicklingError):
        _load_checkpoint_payload(buffer.getvalue())


def test_protected_runner_writes_report_before_terminal_ledger_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    digest = sha256_bytes(b"{}")

    class FakeLedger:
        ORDER = ("selector", "confirmation", "final_test")

        def __init__(self, _bindings: object, *, authority: object) -> None:
            assert authority is qualification._LEDGER_CONSTRUCTION_AUTHORITY
            self.path = tmp_path / "qualification-access.json"

        def begin_access(self, split: str) -> str:
            events.append(f"begin:{split}")
            return split

        def complete_split(self, split: str, _result: object) -> None:
            events.append(f"complete:{split}")

        def prepare_report(self, *, passed: bool, stopped_after: str) -> None:
            assert passed and stopped_after == "final_test"
            events.append("ledger:report-pending")

        def finish(self, *, report_sha256: str) -> None:
            assert report_sha256 == digest
            events.append("ledger:terminal")

        def record_error(self, *_args: object, **_kwargs: object) -> None:
            events.append("ledger:error")

    fake_development = _development_result()
    # This test replaces every evidence read with synthetic bytes; the exact
    # archive guard has its own production-path and tamper tests above.
    monkeypatch.setattr(qualification, "_require_attempt1_rejection", lambda: None)
    monkeypatch.setattr(qualification, "assert_rgbd_partial_visibility_config", lambda _: None)
    monkeypatch.setattr(qualification, "_assert_execution_environment", lambda: None)
    monkeypatch.setattr(qualification, "_require_config_matches", lambda *_args: None)
    monkeypatch.setattr(qualification, "_require_canonical_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qualification, "_validate_artifact_inventory", lambda **_kwargs: None)
    monkeypatch.setattr(qualification, "validate_distinct_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qualification, "_lexists", lambda _path: False)
    monkeypatch.setattr(qualification, "stable_read_bytes", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(qualification, "_single_link_read_bytes", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(
        qualification, "validate_development_evidence", lambda *_a, **_k: fake_development
    )
    monkeypatch.setattr(qualification, "validate_development_ledger", lambda *_a, **_k: None)
    monkeypatch.setattr(qualification, "validate_checkpoint_evidence", lambda *_a, **_k: None)
    monkeypatch.setattr(
        qualification, "_load_checkpoint_payload", lambda _contents: {"model_state": {}}
    )
    monkeypatch.setattr(qualification, "new_public_model", lambda _config: _EmptyModel())
    monkeypatch.setattr(qualification, "_guard_frozen_inputs", lambda **_kwargs: None)
    monkeypatch.setattr(qualification, "QualificationLedger", FakeLedger)
    monkeypatch.setattr(qualification, "qualification_ledger_path", lambda: tmp_path / "q.json")
    monkeypatch.setattr(qualification, "development_ledger_path", lambda: tmp_path / "d.json")
    monkeypatch.setattr(
        qualification,
        "_evaluate_seed_manifest",
        lambda _config, seeds, *, split, authorization: {
            "split": split,
            "seeds": list(seeds),
            "passed": True,
            "failures": [],
        },
    )
    monkeypatch.setattr(
        qualification,
        "write_report_fresh",
        lambda _path, _report: events.append("report:durable"),
    )
    result = qualification.run_qualification(
        object(),  # type: ignore[arg-type]
        config_path=tmp_path / "config.yaml",
        report_path=tmp_path / "report.json",
        checkpoint_path=tmp_path / "checkpoint.pt",
        development_report_path=tmp_path / "development.json",
        reviewed_checkpoint_sha256=digest,
        reviewed_report_sha256=digest,
        reviewed_development_ledger_sha256=digest,
        source_provenance=_source(),
    )
    assert result == 0
    assert events.index("ledger:report-pending") < events.index("report:durable")
    assert events.index("report:durable") < events.index("ledger:terminal")
    assert events[-1] == "ledger:terminal"
