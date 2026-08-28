from __future__ import annotations

import io
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import world_model.training.rgbd_two_visible_orbital_camera_qualification as qualification
from scripts.run_rgbd_two_visible_orbital_camera_qualification import arguments as cli_arguments
from world_model.datasets import collate_episodes
from world_model.simulator.renderer import render_spheres
from world_model.simulator.sphere_world import SphereWorld, SphereWorldConfig
from world_model.training.loop import make_rgbd_packet
from world_model.training.rgbd_online_bridge_qualification import canonical_sha256, sha256_bytes
from world_model.training.rgbd_two_visible_orbital_camera_qualification import (
    CAMERA_DIRECTIONS,
    CONFIRMATION_SEEDS,
    DEFAULT_GATES,
    DEVELOPMENT_SEEDS,
    FINAL_TEST_SEEDS,
    FROZEN_CERTIFICATE_SHA256,
    FROZEN_CONFIG_SHA256,
    HORIZONS_SECONDS,
    MANIFEST_SHA256,
    MAX_ARCHITECTURE_ATTEMPTS,
    SELECTOR_SEEDS,
    SPLIT_PHYSICAL_PAIRS,
    VJP_OUTPUTS,
    assert_rgbd_two_visible_orbital_camera_config,
    bridge_protocol,
    gate_failures,
    new_public_model,
    orbital_camera_frame,
    scene_family_certificate,
    scene_specification,
)
from world_model.utils.config import load_config

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_two_visible_orbital_camera_cpu.yaml"


class _UnsafeCheckpointValue:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (eval, ("40 + 2",))


@pytest.fixture(autouse=True)
def _isolate_private_authority_registries() -> Any:
    registries = (
        qualification._LIVE_MANIFEST_CAPABILITIES,
        qualification._LIVE_PRIVATE_LEDGERS,
        qualification._LIVE_RUN_AUTHORIZATIONS,
        qualification._LIVE_REVIEWED_DEVELOPMENT_SEALS,
    )
    for registry in registries:
        registry.clear()
    yield
    for registry in registries:
        registry.clear()


def _source() -> dict[str, Any]:
    return {
        "commit": "1" * 40,
        "dirty": False,
        "worktree_fingerprint": "2" * 64,
        "runtime_source_fingerprint": "3" * 64,
    }


def _publication(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "upstream_ref": "origin/agent/rgbd-moving-camera-rung-1",
        "head_commit": source["commit"],
        "upstream_commit": source["commit"],
        "ahead": 0,
        "behind": 0,
    }


def _handcrafted_row(split: str, ordinal: int) -> dict[str, Any]:
    specification = scene_specification(split, ordinal)
    positions, velocities = qualification._exact_physical_trajectory(specification)
    rgb: list[torch.Tensor] = []
    depth: list[torch.Tensor] = []
    world_from_camera: list[torch.Tensor] = []
    camera_from_world: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []
    camera_position: list[torch.Tensor] = []
    camera_target: list[torch.Tensor] = []
    for frame_index in range(16):
        camera = orbital_camera_frame(specification, frame_index / 20.0)
        state = qualification._certificate_state(
            specification,
            positions[frame_index],
            velocities[frame_index],
        )
        rendered = render_spheres(state, camera, (64, 64), noise_std=0.0)
        rgb.append(rendered.rgb)
        depth.append(rendered.depth_buffer.unsqueeze(0))
        world_from_camera.append(camera.world_from_camera)
        camera_from_world.append(camera.camera_from_world)
        intrinsics.append(camera.intrinsics)
        camera_position.append(camera.position)
        camera_target.append(camera.target)
    return {
        "rgb": torch.stack(rgb),
        "depth": torch.stack(depth),
        "timestamps": torch.arange(16, dtype=torch.float32) / 20.0,
        "camera": {
            "world_from_camera": torch.stack(world_from_camera),
            "camera_from_world": torch.stack(camera_from_world),
            "intrinsics": torch.stack(intrinsics),
            "position": torch.stack(camera_position),
            "target": torch.stack(camera_target),
        },
        "objects": {
            "position": positions,
            "velocity": velocities,
            "albedo": specification.albedo[None].expand(56, -1, -1).clone(),
        },
        "labels": {"sentinel": torch.tensor([ordinal], dtype=torch.int64)},
        "events": {"sentinel": torch.tensor([ordinal], dtype=torch.int64)},
        "metadata": {
            "scenario": "two_visible_orbital_camera_free_motion",
            "camera_calibration_owner": "qualification_known_extrinsics",
            "split": split,
            "physical_index": specification.physical_index,
            "split_primitive_index": specification.split_primitive_index,
            "primitive_a": specification.a,
            "primitive_b": specification.b,
            "camera_stratum": specification.camera_stratum,
            "camera_phase_index": specification.phase_index,
            "camera_direction_index": specification.direction_index,
            "camera_direction": specification.direction,
            "palette_swapped": specification.palette_swapped,
        },
    }


@pytest.fixture(scope="module")
def heterogeneous_batch() -> dict[str, Any]:
    rows = (
        _handcrafted_row("development", 0),
        _handcrafted_row("development", 11),
        _handcrafted_row("selector", 18),
        _handcrafted_row("final_test", 39),
    )
    return collate_episodes(rows)


def _passing_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {
        "current_position_rmse_m": 0.0,
        "current_velocity_rmse_mps": 0.0,
        "maximum_position_error_growth_slope_mps": 0.0,
        "early_stationary_additive_regression_m": 0.0,
        "long_stationary_rmse_ratio": 0.0,
        "zero_velocity_rmse_ratio": 0.0,
        "stale_camera_current_position_rmse_m": 1.0,
        "correct_to_stale_current_position_rmse_ratio": 0.0,
        "stale_camera_current_velocity_rmse_mps": 1.0,
        "correct_to_stale_current_velocity_rmse_ratio": 0.0,
        "stale_camera_horizon_2_00_position_rmse_m": 1.0,
        "correct_to_stale_horizon_2_00_position_rmse_ratio": 0.0,
        "stale_camera_identity_switch_count": 0.0,
        "stale_camera_association_ambiguous_pair_count": 0.0,
        "stale_camera_history_valid_count_min": 16.0,
        "certificate_ideal_stale_camera_current_position_rmse_m": 1.0,
        "certificate_ideal_stale_camera_current_velocity_rmse_mps": 1.0,
        "certificate_ideal_stale_camera_horizon_2_00_position_rmse_m": 1.0,
        "identity_switch_count": 0.0,
        "persistent_id_mismatch_count": 0.0,
        "identity_coverage": 1.0,
        "persistent_object_id_min": 0.0,
        "persistent_object_id_max": 1.0,
        "association_pair_coverage": 1.0,
        "association_ambiguous_pair_count": 0.0,
        "minimum_hungarian_margin": 1.0,
        "minimum_position_assignment_margin_m": 1.0,
        "minimum_matched_appearance_cosine": 1.0,
        "minimum_cross_appearance_cosine_distance": 1.0,
        "physical_palette_swap_fraction": 0.5,
        "birth_slot_physical_zero_fraction": 0.5,
        "unique_scene_specification_fraction": 1.0,
        "certificate_physical_trajectory_count": 16.0,
        "certificate_camera_appearance_combination_count": 128.0,
        "gradient_audit_scene_count": 4.0,
        "gradient_audit_unique_scene_fraction": 1.0,
        "active_fraction": 1.0,
        "rollout_active_fraction": 1.0,
        "history_sample_count_min": 16.0,
        "history_sample_count_max": 16.0,
        "history_valid_count_min": 16.0,
        "history_valid_count_max": 16.0,
        "history_span_max_abs_error_seconds": 0.0,
        "direct_velocity_calls_per_batch_min": 1.0,
        "direct_velocity_calls_per_batch_max": 1.0,
        "direct_velocity_valid_fraction": 1.0,
        "position_owner_count_min": 1.0,
        "position_owner_count_max": 1.0,
        "direct_position_field_count": 0.0,
        "direct_velocity_position_change_max_abs_m": 0.0,
        "direct_metric_position_owner_max_abs_m": 0.0,
        "ambiguity_direct_position_write_count": 0.0,
        "ambiguity_direct_velocity_write_count": 0.0,
        "preflight_minimum_silhouette_gap_pixels": 10.0,
        "preflight_minimum_boundary_clearance_pixels": 10.0,
        "preflight_minimum_world_surface_gap_m": 2.0,
        "preflight_minimum_world_boundary_clearance_m": 1.0,
        "preflight_minimum_visible_fraction": 1.0,
        "preflight_minimum_full_support_pixels": 22.0,
        "preflight_event_count": 0.0,
        "preflight_minimum_palette_cosine_distance": 1.0,
        "preflight_minimum_camera_adjacent_angle_radians": 0.012,
        "preflight_maximum_camera_adjacent_angle_radians": 0.012,
        "preflight_minimum_camera_translation_step_m": 0.0552,
        "preflight_maximum_camera_translation_step_m": 0.0552,
        "preflight_maximum_projected_centre_step_pixels": 0.12,
        "preflight_camera_calibration_max_abs_error": 0.0,
        "final_belief_camera_max_abs_error": 0.0,
        "semigroup_position_max_abs_m": 0.0,
        "semigroup_velocity_max_abs_mps": 0.0,
        "public_direct_position_max_abs_m": 0.0,
        "public_direct_velocity_max_abs_mps": 0.0,
        "analytic_position_agreement_max_abs_m": 0.0,
        "analytic_velocity_agreement_max_abs_mps": 0.0,
        "public_rollout_output_alias_count": 0.0,
        "public_query_time_max_abs_seconds": 0.0,
        "ingested_frame_count_min": 16.0,
        "ingested_frame_count_max": 16.0,
        "public_predict_calls_per_batch_min": 1.0,
        "public_predict_calls_per_batch_max": 1.0,
        "world_from_camera_homogeneous_last_row_gradient_max_abs": 0.0,
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
    }
    for object_index in (0, 1):
        for axis in ("x", "y", "z"):
            metrics[f"current_position_rmse_m/object_{object_index}/{axis}"] = 0.0
            metrics[f"current_velocity_rmse_mps/object_{object_index}/{axis}"] = 0.0
    for horizon in HORIZONS_SECONDS:
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = 0.0
        metrics[f"horizon_{label}_velocity_rmse_mps"] = 0.0
        for object_index in (0, 1):
            for axis in ("x", "y", "z"):
                metrics[f"horizon_{label}_position_rmse_m/object_{object_index}/{axis}"] = 0.0
                metrics[f"horizon_{label}_velocity_rmse_mps/object_{object_index}/{axis}"] = 0.0
    for camera_stratum in range(8):
        phase_index = camera_stratum // 2
        direction = CAMERA_DIRECTIONS[camera_stratum % 2]
        suffix = f"phase_{phase_index}/direction_{direction:+d}"
        metrics[f"current_position_rmse_m/{suffix}"] = 0.0
        metrics[f"current_velocity_rmse_mps/{suffix}"] = 0.0
        metrics[f"stale_camera_current_position_rmse_m/{suffix}"] = 1.0
        metrics[f"correct_to_stale_current_position_rmse_ratio/{suffix}"] = 0.0
        metrics[f"stale_camera_current_velocity_rmse_mps/{suffix}"] = 1.0
        metrics[f"correct_to_stale_current_velocity_rmse_ratio/{suffix}"] = 0.0
        metrics[f"stale_camera_horizon_2_00_position_rmse_m/{suffix}"] = 1.0
        metrics[f"correct_to_stale_horizon_2_00_position_rmse_ratio/{suffix}"] = 0.0
        for horizon in HORIZONS_SECONDS:
            label = f"{horizon:.2f}"
            metrics[f"horizon_{label}_position_rmse_m/{suffix}"] = 0.0
            metrics[f"horizon_{label}_velocity_rmse_mps/{suffix}"] = 0.0
    for object_index in (0, 1):
        for output_name in VJP_OUTPUTS:
            for modality in ("rgb", "depth", "world_from_camera"):
                suffix = f"object_{object_index}/{output_name}/{modality}"
                metrics[f"gradient_l1/{suffix}"] = 1.0
                metrics[f"gradient_max_l1/{suffix}"] = 1.0
                metrics[f"gradient_cross_scene_max_l1/{suffix}"] = 0.0
                if output_name == "current_position":
                    metrics[f"gradient_anchor_history_frame_l1/{suffix}"] = 1.0
                    metrics[f"gradient_nonanchor_max_history_frame_l1/{suffix}"] = 0.0
                    metrics[f"gradient_supported_history_frames/{suffix}"] = 1.0
                else:
                    metrics[f"gradient_min_history_frame_l1/{suffix}"] = 1.0
                    metrics[f"gradient_supported_history_frames/{suffix}"] = 16.0
    return metrics


def _passing_split_result(split: str) -> dict[str, Any]:
    manifests = {
        "development": DEVELOPMENT_SEEDS,
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }
    seeds = manifests[split]
    return {
        "split": split,
        "seeds": list(seeds),
        "seed_manifest_sha256": MANIFEST_SHA256[split],
        "metrics": _passing_metrics(),
        "failures": [],
        "passed": True,
        "optimizer_updates": 0,
        "runtime_api": {
            "packet_factory": "make_rgbd_packet",
            "ingest_frames": list(range(16)),
            "rollout_method": "OnlineWorldModel.predict",
            "horizons_seconds": list(HORIZONS_SECONDS),
        },
        "scene_constructor": (
            "private_two_visible_orbital_camera_episode_with_full_frame_preflight"
        ),
    }


def test_protocol_freezes_one_attempt_manifests_mapping_and_input_boundary() -> None:
    manifests = {
        "development": DEVELOPMENT_SEEDS,
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }
    assert tuple(map(len, manifests.values())) == (32, 24, 24, 48)
    assert tuple(range(61_000_000, 61_000_032)) == DEVELOPMENT_SEEDS
    assert tuple(range(62_000_000, 62_000_024)) == SELECTOR_SEEDS
    assert tuple(range(63_000_000, 63_000_024)) == CONFIRMATION_SEEDS
    assert tuple(range(64_000_000, 64_000_048)) == FINAL_TEST_SEEDS
    flattened = tuple(seed for values in manifests.values() for seed in values)
    assert len(flattened) == len(set(flattened))
    for split, seeds in manifests.items():
        assert canonical_sha256(list(seeds)) == MANIFEST_SHA256[split]
    protocol = bridge_protocol()
    stated = protocol.pop("protocol_sha256")
    assert stated == canonical_sha256(protocol)
    assert protocol["name"] == "rgbd_two_visible_orbital_camera_bridge_v1"
    assert protocol["architecture_attempt"] == 1
    assert protocol["maximum_architecture_attempts"] == MAX_ARCHITECTURE_ATTEMPTS == 1
    assert protocol["terminal_after_attempt"] is True
    assert protocol["scene_family"]["physical_trajectory_count"] == 16
    assert protocol["scene_family"]["camera_appearance_combination_count"] == 128
    assert protocol["camera"]["theta_law"] == "theta(t)=theta0+direction*0.24*t"
    perception = protocol["perception"]
    assert perception["scene_qualification_minimum_silhouette_gap_pixels"] == 4.0
    assert perception["scene_qualification_minimum_boundary_clearance_pixels"] == 6.0
    assert perception["runtime_module_minimum_silhouette_gap_pixels"] == 2.0
    assert perception["runtime_module_minimum_boundary_clearance_pixels"] == 2.0
    assert protocol["runtime"]["inputs"] == [
        "rgb",
        "depth",
        "timestamp",
        "world_from_camera",
        "intrinsics",
        "image_metadata",
    ]
    assert "truth_objects" in protocol["runtime"]["excluded_inputs"]
    assert "camera_from_world" in protocol["runtime"]["excluded_inputs"]
    assert protocol["source_binding"]["upstream_commit_must_equal_head"] is True


def test_exact_config_binds_parameter_free_known_extrinsics_runtime() -> None:
    assert sha256_bytes(CONFIG_PATH.read_bytes()) == FROZEN_CONFIG_SHA256
    config = load_config(CONFIG_PATH)
    assert_rgbd_two_visible_orbital_camera_config(config)
    assert config.project.seed == DEVELOPMENT_SEEDS[0]
    assert config.simulator.camera_motion == "orbit"
    assert config.simulator.known_camera_pose is True
    model = new_public_model(config)
    assert not tuple(model.parameters())
    assert not tuple(model.buffers())
    assert model.state_dict() == {}
    with pytest.raises(ValueError, match="camera_motion"):
        assert_rgbd_two_visible_orbital_camera_config(
            replace(config, simulator=replace(config.simulator, camera_motion="fixed"))
        )
    with pytest.raises(ValueError, match="known_camera_pose"):
        assert_rgbd_two_visible_orbital_camera_config(
            replace(config, simulator=replace(config.simulator, known_camera_pose=False))
        )


def test_raw_boundaries_rebind_config_threads_and_external_cli_review_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    tampered = replace(config, project=replace(config.project, seed=config.project.seed + 1))
    with pytest.raises(ValueError, match="executed config object differs"):
        qualification._require_config_matches_frozen_path(tampered, CONFIG_PATH)
    monkeypatch.setattr(qualification.torch, "get_num_threads", lambda: 2)
    with pytest.raises(RuntimeError, match=r"get_num_threads\(\)==1"):
        qualification._require_single_thread_execution()

    digest = "0" * 64
    with pytest.raises(SystemExit):
        cli_arguments(
            [
                "--phase",
                "qualification",
                "--reviewed-checkpoint-sha256",
                digest,
                "--reviewed-report-sha256",
                digest,
            ]
        )
    parsed = cli_arguments(
        [
            "--phase",
            "qualification",
            "--reviewed-checkpoint-sha256",
            digest,
            "--reviewed-report-sha256",
            digest,
            "--reviewed-development-ledger-sha256",
            digest,
        ]
    )
    assert parsed.reviewed_development_ledger_sha256 == digest


def test_literal_physical_table_joint_mapping_and_palette_balance() -> None:
    all_pairs: list[tuple[int, int]] = []
    combined_signatures: set[str] = set()
    physical_signatures: set[str] = set()
    for split, pairs in SPLIT_PHYSICAL_PAIRS.items():
        all_pairs.extend(pairs)
        manifest_length = len(pairs) * 8
        swapped = 0
        for ordinal in range(manifest_length):
            specification = scene_specification(split, ordinal)
            primitive_index = ordinal // 8
            stratum = ordinal % 8
            assert specification.split_primitive_index == primitive_index
            assert specification.camera_stratum == stratum
            assert specification.phase_index == stratum // 2
            assert specification.direction == CAMERA_DIRECTIONS[stratum % 2]
            a, b = pairs[primitive_index]
            expected_position = torch.tensor(
                [
                    [-0.450 + 0.006 * a, 0.400 + 0.002 * b, -0.300 + 0.004 * b],
                    [0.450 + 0.005 * b, 1.750 + 0.002 * a, 0.300 - 0.004 * a],
                ],
                dtype=torch.float32,
            )
            expected_velocity = torch.tensor(
                [
                    [0.045 + 0.0005 * a, 0.012 + 0.00025 * b, 0.004 + 0.00025 * a],
                    [-0.043 + 0.0005 * b, -0.008 + 0.00025 * a, -0.003 + 0.00025 * b],
                ],
                dtype=torch.float32,
            )
            torch.testing.assert_close(specification.position, expected_position, atol=0, rtol=0)
            torch.testing.assert_close(specification.velocity, expected_velocity, atol=0, rtol=0)
            swapped += int(specification.palette_swapped)
            physical_signatures.add(
                canonical_sha256(
                    {
                        "position": specification.position.tolist(),
                        "velocity": specification.velocity.tolist(),
                    }
                )
            )
            combined_signatures.add(
                canonical_sha256(
                    {
                        "physical": specification.physical_index,
                        "stratum": specification.camera_stratum,
                        "albedo": specification.albedo.tolist(),
                    }
                )
            )
        assert swapped == manifest_length // 2
    assert len(all_pairs) == len(set(all_pairs)) == 16
    assert len(physical_signatures) == 16
    assert len(combined_signatures) == 128


def test_orbital_camera_uses_one_exact_float32_pose_law() -> None:
    specification = scene_specification("development", 0)
    frames = [orbital_camera_frame(specification, index / 20.0) for index in range(56)]
    positions = torch.stack([frame.position for frame in frames])
    transforms = torch.stack([frame.world_from_camera for frame in frames])
    inverses = torch.stack([frame.camera_from_world for frame in frames])
    radii = torch.linalg.vector_norm(positions[:, [0, 2]], dim=-1)
    torch.testing.assert_close(radii, torch.full_like(radii, 4.6), atol=1e-6, rtol=0)
    torch.testing.assert_close(positions[:, 1], torch.full((56,), 2.15), atol=0, rtol=0)
    identity = transforms @ inverses
    torch.testing.assert_close(identity, torch.eye(4).expand(56, -1, -1), atol=2e-5, rtol=0)
    chord = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=-1)
    assert float(chord.min()) >= DEFAULT_GATES.minimum_camera_translation_step_m
    assert float(chord.max()) <= DEFAULT_GATES.maximum_camera_translation_step_m


def test_seed_free_certificate_is_exact_and_public_renderer_equivalent() -> None:
    certificate = scene_family_certificate(verify_public_renderer=True)
    assert certificate["certificate_sha256"] == FROZEN_CERTIFICATE_SHA256
    assert certificate["physical_trajectory_count"] == 16
    assert certificate["camera_appearance_combination_count"] == 128
    assert certificate["minimum_full_support_pixels"] == 22.0
    assert certificate["minimum_continuous_silhouette_gap_pixels"] == pytest.approx(
        9.622230529785156
    )
    assert certificate["minimum_image_boundary_clearance_pixels"] == pytest.approx(
        15.584700584411621
    )
    assert certificate["minimum_world_surface_gap_m"] == pytest.approx(1.144572138786316)
    assert certificate["minimum_world_boundary_clearance_m"] == pytest.approx(0.18400000035762787)
    assert certificate["minimum_speed_mps"] == pytest.approx(0.03676881268620491)
    assert certificate["maximum_speed_mps"] == pytest.approx(0.048449717462062836)
    ideal = certificate["ideal_stale_camera_control"]
    assert ideal["minimum_joint_rmse"] == pytest.approx(
        {
            "current_position_rmse_m": 0.052036356180906296,
            "current_velocity_rmse_mps": 0.06767760217189789,
            "horizon_2_00_position_rmse_m": 0.18084664642810822,
        }
    )
    for values in ideal["per_split_pooled_rmse"].values():
        assert values["current_position_rmse_m"] >= 0.045
        assert values["current_velocity_rmse_mps"] >= 0.065
        assert values["horizon_2_00_position_rmse_m"] >= 0.12


def test_public_sphere_world_matches_exact_float32_physics_with_no_contact_corrections() -> None:
    config = load_config(CONFIG_PATH)
    resolved = SphereWorldConfig.from_config(config)
    specifications = [
        scene_specification(split, primitive_index * 8)
        for split, pairs in SPLIT_PHYSICAL_PAIRS.items()
        for primitive_index in range(len(pairs))
    ]
    assert len(specifications) == 16
    total_substeps = 0
    event_fields = (
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
    )
    for specification in specifications:
        expected_position, expected_velocity = qualification._exact_physical_trajectory(
            specification
        )
        world = SphereWorld(resolved, seed=0)
        qualification._install_scene(world, specification)
        assert torch.equal(world.state.position, expected_position[0])
        assert torch.equal(world.state.velocity, expected_velocity[0])
        for frame_index in range(1, 56):
            events = world.step(
                resolved.observation_dt,
                external_impulse=torch.zeros((2, 3), dtype=torch.float32),
            )
            total_substeps += events.substeps
            assert events.substeps == 6
            assert all(
                torch.count_nonzero(getattr(events, name)).item() == 0 for name in event_fields
            )
            assert torch.equal(events.first_event_offset, torch.full((2,), -1.0))
            assert torch.equal(world.state.position, expected_position[frame_index])
            assert torch.equal(world.state.velocity, expected_velocity[frame_index])
    assert total_substeps == 16 * 330


def test_heterogeneous_metadata_collation_preserves_each_row_calibration(
    heterogeneous_batch: dict[str, Any],
) -> None:
    metadata = heterogeneous_batch["metadata"]
    rows = (("development", 0), ("development", 11), ("selector", 18), ("final_test", 39))
    specifications = [scene_specification(split, ordinal) for split, ordinal in rows]
    assert metadata["scenario"] == ["two_visible_orbital_camera_free_motion"] * 4
    assert metadata["camera_calibration_owner"] == ["qualification_known_extrinsics"] * 4
    assert metadata["split"] == [split for split, _ordinal in rows]
    for key, attribute in (
        ("physical_index", "physical_index"),
        ("split_primitive_index", "split_primitive_index"),
        ("primitive_a", "a"),
        ("primitive_b", "b"),
        ("camera_stratum", "camera_stratum"),
        ("camera_phase_index", "phase_index"),
        ("camera_direction_index", "direction_index"),
        ("camera_direction", "direction"),
        ("palette_swapped", "palette_swapped"),
    ):
        assert metadata[key].tolist() == [
            getattr(specification, attribute) for specification in specifications
        ]
    signatures = {
        canonical_sha256(heterogeneous_batch["camera"]["world_from_camera"][row].tolist())
        for row in range(4)
    }
    assert len(signatures) == 4
    for row, (split, ordinal) in enumerate(rows):
        expected = orbital_camera_frame(scene_specification(split, ordinal), 0.75)
        torch.testing.assert_close(
            heterogeneous_batch["camera"]["world_from_camera"][row, 15],
            expected.world_from_camera,
            atol=0,
            rtol=0,
        )


def test_public_packet_exposes_only_observable_known_calibration(
    heterogeneous_batch: dict[str, Any],
) -> None:
    packet = make_rgbd_packet(heterogeneous_batch, 7)
    assert set(packet.payload) == {"rgb", "depth"}
    assert set(packet.calibration) == {"world_from_camera", "intrinsics"}
    assert set(packet.metadata) == {"image_size", "training_frame_index", "depth_semantics"}
    forbidden = {
        "position",
        "target",
        "linear_velocity",
        "angular_velocity",
        "camera_from_world",
        "objects",
        "labels",
        "truth",
    }
    assert forbidden.isdisjoint(packet.payload)
    assert forbidden.isdisjoint(packet.calibration)
    assert forbidden.isdisjoint(packet.metadata)


def test_stale_control_changes_only_history_world_from_camera(
    heterogeneous_batch: dict[str, Any],
) -> None:
    stale = qualification._stale_world_from_camera_batch(heterogeneous_batch)
    assert stale["rgb"] is heterogeneous_batch["rgb"]
    assert stale["depth"] is heterogeneous_batch["depth"]
    assert stale["timestamps"] is heterogeneous_batch["timestamps"]
    assert stale["objects"] is heterogeneous_batch["objects"]
    assert stale["labels"] is heterogeneous_batch["labels"]
    assert stale["camera"]["intrinsics"] is heterogeneous_batch["camera"]["intrinsics"]
    fresh_wfc = heterogeneous_batch["camera"]["world_from_camera"]
    stale_wfc = stale["camera"]["world_from_camera"]
    assert torch.equal(stale_wfc[:, 0], fresh_wfc[:, 0])
    assert torch.equal(stale_wfc[:, 1:16], fresh_wfc[:, :1].expand(-1, 15, -1, -1))
    assert not torch.equal(stale_wfc[:, 1:16], fresh_wfc[:, 1:16])


def test_seed_free_public_runtime_uses_moving_calibration_and_stale_control_is_causal(
    heterogeneous_batch: dict[str, Any],
) -> None:
    config = load_config(CONFIG_PATH)
    with torch.no_grad():
        correct = qualification._run_public_batch(heterogeneous_batch, config)
    assert float(correct["final_belief_camera_error"].max()) <= 1.0e-6
    assert int(correct["identities"].ne(torch.tensor([0, 1]).view(1, 1, 2)).sum()) == 0
    stale = qualification._stale_camera_batch_metrics(heterogeneous_batch, config)
    assert qualification._rmse(stale["current_position_error"]) >= 0.045
    assert qualification._rmse(stale["current_velocity_error"]) >= 0.050
    assert qualification._rmse(stale["horizon_2_00_position_error"]) >= 0.080
    assert stale["identity_switch_count"] == 0.0
    assert stale["association_ambiguous_pair_count"] == 0.0
    assert stale["history_valid_count_min"] == 16.0


def test_world_from_camera_ambient_vjps_retain_topology_without_pose_claim(
    heterogeneous_batch: dict[str, Any],
) -> None:
    metrics = qualification._gradient_metrics(load_config(CONFIG_PATH), heterogeneous_batch)
    assert metrics["gradient_audit_scene_count"] == 4.0
    assert metrics["gradient_audit_unique_scene_fraction"] == 1.0
    assert metrics["world_from_camera_homogeneous_last_row_gradient_max_abs"] == 0.0
    for object_index in (0, 1):
        for output_name in VJP_OUTPUTS:
            prefix = f"object_{object_index}/{output_name}/world_from_camera"
            assert metrics[f"gradient_l1/{prefix}"] >= DEFAULT_GATES.minimum_input_gradient_l1
            assert metrics[f"gradient_cross_scene_max_l1/{prefix}"] == 0.0
            if output_name == "current_position":
                assert metrics[f"gradient_supported_history_frames/{prefix}"] == 1.0
                assert metrics[f"gradient_nonanchor_max_history_frame_l1/{prefix}"] == 0.0
            else:
                assert metrics[f"gradient_supported_history_frames/{prefix}"] == 16.0
    # Intrinsics VJPs are finite-checked as diagnostics inside the harness, but
    # are intentionally absent from the exact qualification gate surface.
    assert not any("/intrinsics" in key for key in metrics)
    assert bridge_protocol()["differentiability"]["world_from_camera_gradient"] == (
        "ambient_matrix_vjp_not_se3_or_pose_estimation"
    )


def test_complete_gate_schema_passes_and_per_stratum_correct_tamper_fails() -> None:
    metrics = _passing_metrics()
    assert len(qualification.GATE_METRIC_SCHEMA) == 685
    assert set(metrics) == set(qualification.GATE_METRIC_SCHEMA)
    assert gate_failures(metrics) == []
    extra = dict(metrics, undeclared_diagnostic=0.0)
    assert any(failure.startswith("metric_schema:") for failure in gate_failures(extra))
    bool_metric = dict(metrics)
    bool_metric["current_position_rmse_m"] = False
    assert gate_failures(bool_metric)
    key = "horizon_2.00_position_rmse_m/phase_3/direction_+1"
    tampered = dict(metrics)
    tampered[key] = DEFAULT_GATES.horizon_position_rmse_m[-1] + 0.001
    assert any(failure.startswith(f"{key}:") for failure in gate_failures(tampered))
    stale_key = "stale_camera_current_velocity_rmse_mps/phase_0/direction_-1"
    tampered = dict(metrics)
    tampered[stale_key] = DEFAULT_GATES.stale_camera_current_velocity_rmse_mps - 0.001
    assert any(failure.startswith(f"{stale_key}:") for failure in gate_failures(tampered))


def test_exact_split_report_ledger_and_checkpoint_schemas_reject_tampering() -> None:
    source = _source()
    publication = _publication(source)
    development = _passing_split_result("development")
    qualification._validate_split_result(development, split="development")
    extra_result = dict(development, undeclared=True)
    with pytest.raises(ValueError, match="schema differs"):
        qualification._validate_split_result(extra_result, split="development")
    bool_result = dict(development, optimizer_updates=False)
    with pytest.raises(TypeError, match="integer zero"):
        qualification._validate_split_result(bool_result, split="development")

    digest = "0" * 64
    development_report = {
        "artifact_kind": "rgbd_two_visible_orbital_camera_development",
        "protocol": bridge_protocol(),
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "scene_family_certificate": scene_family_certificate(),
        "development_ledger": str(qualification.development_ledger_path()),
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "passed": True,
        "review_ready": True,
        "stopped_after": "development",
        "development": development,
        "checkpoint": str(qualification.canonical_checkpoint_path()),
        "checkpoint_sha256": digest,
        "checkpoint_model_state_sha256": qualification.EMPTY_MODEL_STATE_SHA256,
        "checkpoint_roundtrip_state_sha256": qualification.EMPTY_MODEL_STATE_SHA256,
    }
    qualification._validate_development_report_schema(development_report, error=False)
    missing_development = dict(development_report)
    missing_development.pop("development")
    with pytest.raises(ValueError, match="missing=.*development"):
        qualification._validate_development_report_schema(missing_development, error=False)

    checkpoint_metrics = {
        "artifact_kind": "rgbd_two_visible_orbital_camera_empty_model_state",
        "optimizer_updates": 0,
        "model_state_sha256": qualification.EMPTY_MODEL_STATE_SHA256,
        "protocol": bridge_protocol(),
        "publication_provenance": publication,
        "development": development,
    }
    config = load_config(CONFIG_PATH)
    payload = qualification.checkpoint_payload(
        model=new_public_model(config),
        optimizer=None,
        scheduler=None,
        config=config,
        step=0,
        metrics=checkpoint_metrics,
        device="cpu",
        source_provenance=source,
    )
    payload.pop("rng")
    qualification.validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        publication=publication,
        development=development,
    )
    extra_payload = dict(payload, undeclared=torch.tensor(1.0))
    with pytest.raises(ValueError, match="schema differs"):
        qualification.validate_checkpoint_evidence(
            extra_payload,
            config=config,
            source=source,
            publication=publication,
            development=development,
        )


def test_raw_materialization_boundaries_are_private_and_not_exported() -> None:
    assert "_construct_two_visible_orbital_camera_episode" not in qualification.__all__
    assert "_evaluate_seed_manifest" not in qualification.__all__
    assert "_DevelopmentLedger" not in qualification.__all__
    assert "_QualificationLedger" not in qualification.__all__
    assert not hasattr(qualification, "construct_two_visible_orbital_camera_episode")
    assert not hasattr(qualification, "evaluate_seed_manifest")
    assert set(name for name in qualification.__all__ if name.startswith("run_")) == {
        "run_development",
        "run_qualification",
    }


def test_publication_validation_rejects_missing_upstream_and_divergence() -> None:
    source = _source()
    publication = _publication(source)
    assert (
        qualification._validated_published_source(publication, source=source, label="test")
        == publication
    )
    missing = dict(publication, upstream_ref=None)
    with pytest.raises(ValueError, match="configured branch upstream"):
        qualification._validated_published_source(missing, source=source, label="test")
    diverged = dict(publication, ahead=1)
    with pytest.raises(ValueError, match="zero commits ahead and behind"):
        qualification._validated_published_source(diverged, source=source, label="test")
    wrong_commit = dict(publication, upstream_commit="4" * 40)
    with pytest.raises(ValueError, match="upstream commit == clean HEAD"):
        qualification._validated_published_source(wrong_commit, source=source, label="test")


def _patch_live_provenance(
    monkeypatch: pytest.MonkeyPatch,
    source: dict[str, Any],
    publication: dict[str, Any],
) -> None:
    monkeypatch.setattr(qualification, "capture_git_metadata", lambda _root: dict(source))
    monkeypatch.setattr(qualification, "capture_published_source", lambda _root: dict(publication))


def _development_bindings(source: dict[str, Any], publication: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development_manifest_sha256": MANIFEST_SHA256["development"],
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }


def _patch_private_test_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: dict[str, Any],
    publication: dict[str, Any],
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "configs" / CONFIG_PATH.name
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG_PATH.read_bytes())
    _patch_live_provenance(monkeypatch, source, publication)
    monkeypatch.setattr(qualification.torch, "get_num_threads", lambda: 1)


def _mint_development_ledger(
    source: dict[str, Any], publication: dict[str, Any]
) -> qualification._DevelopmentLedger:
    wrong = _development_bindings(source, publication)
    wrong["protocol_sha256"] = "f" * 64
    with pytest.raises(PermissionError, match="binding values"):
        qualification._mint_run_authorization("development", wrong)

    bindings = _development_bindings(source, publication)
    authorization = qualification._mint_run_authorization("development", bindings)
    return qualification._DevelopmentLedger(authorization, bindings)


def test_nominal_development_capability_checks_receipt_order_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    _patch_private_test_repository(tmp_path, monkeypatch, source, publication)
    ledger = _mint_development_ledger(source, publication)
    capability = ledger.capability()
    capability.begin_manifest("development", DEVELOPMENT_SEEDS)
    with pytest.raises(PermissionError, match="order differs"):
        capability.authorize_seed(DEVELOPMENT_SEEDS[1])
    for seed in DEVELOPMENT_SEEDS:
        capability.authorize_seed(seed)
    capability.finish_manifest()
    capability.require_finished()
    with pytest.raises(RuntimeError, match="single use"):
        capability.begin_manifest("development", DEVELOPMENT_SEEDS)


def test_capability_rejects_direct_ledger_fake_subclass_manual_allocation_and_wrong_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    _patch_private_test_repository(tmp_path, monkeypatch, source, publication)
    with pytest.raises(PermissionError, match="exact run authorization"):
        qualification._DevelopmentLedger(object(), {})
    with pytest.raises(PermissionError, match="only be minted"):
        qualification._ManifestCapability(
            object(),
            ledger=object(),
            ledger_mint_identity=object(),
            split="development",
            seeds=DEVELOPMENT_SEEDS,
        )
    manual = object.__new__(qualification._ManifestCapability)
    with pytest.raises(PermissionError, match="live-ledger registered"):
        qualification._validate_manifest_capability(
            manual,
            split="development",
            seeds=DEVELOPMENT_SEEDS,
            operation="begin",
        )

    manual_ledger = object.__new__(qualification._DevelopmentLedger)
    manual_ledger._mint_identity = object()
    manual_capability = qualification._ManifestCapability(
        qualification._MANIFEST_CAPABILITY_AUTHORITY,
        ledger=manual_ledger,
        ledger_mint_identity=manual_ledger._mint_identity,
        split="development",
        seeds=DEVELOPMENT_SEEDS,
    )
    with pytest.raises(PermissionError, match="live-issuer registered"):
        qualification._validate_manifest_capability(
            manual_capability,
            split="development",
            seeds=DEVELOPMENT_SEEDS,
            operation="begin",
        )

    class _CapabilitySubclass(qualification._ManifestCapability):
        pass

    subclass = object.__new__(_CapabilitySubclass)
    with pytest.raises(PermissionError, match="wrong nominal type"):
        qualification._validate_manifest_capability(
            subclass,
            split="development",
            seeds=DEVELOPMENT_SEEDS,
            operation="begin",
        )

    bindings = _development_bindings(source, publication)
    authorization = qualification._mint_run_authorization("development", bindings)
    ledger = qualification._DevelopmentLedger(authorization, bindings)
    official = ledger.capability()
    duplicate = qualification._ManifestCapability(
        qualification._MANIFEST_CAPABILITY_AUTHORITY,
        ledger=ledger,
        ledger_mint_identity=ledger._mint_identity,
        split="development",
        seeds=DEVELOPMENT_SEEDS,
    )
    with pytest.raises(PermissionError, match="ledger-owned capability"):
        duplicate.begin_manifest("development", DEVELOPMENT_SEEDS)
    with pytest.raises(PermissionError, match="stale, replayed"):
        qualification._DevelopmentLedger(authorization, bindings)
    assert official is ledger._capability


def test_capability_rejects_arbitrary_path_hardlink_and_receipt_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    _patch_private_test_repository(tmp_path, monkeypatch, source, publication)
    ledger = _mint_development_ledger(source, publication)
    canonical = qualification.development_ledger_path()
    capability = ledger.capability()
    ledger.path = tmp_path / "arbitrary.json"
    with pytest.raises(PermissionError, match="path is not canonical"):
        capability.begin_manifest("development", DEVELOPMENT_SEEDS)
    ledger.path = canonical
    hardlink = tmp_path / "receipt-link.json"
    os.link(canonical, hardlink)
    with pytest.raises(PermissionError, match="single-link regular"):
        capability.begin_manifest("development", DEVELOPMENT_SEEDS)


def test_paths_reject_subclass_symlink_parent_retarget_and_unexpected_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    symlink_case = tmp_path / "symlink_parent"
    _patch_private_test_repository(symlink_case, monkeypatch, source, publication)
    outside = tmp_path / "outside"
    outside.mkdir()
    (symlink_case / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="real directory"):
        qualification._mint_run_authorization(
            "development", _development_bindings(source, publication)
        )

    retarget_case = tmp_path / "retarget"
    _patch_private_test_repository(retarget_case, monkeypatch, source, publication)
    ledger = _mint_development_ledger(source, publication)
    capability = ledger.capability()
    canonical = qualification.development_ledger_path()

    class _PathSubclass(type(canonical)):
        pass

    with pytest.raises(TypeError, match="exact native Path"):
        qualification._require_canonical_path(
            _PathSubclass(str(canonical)), canonical, label="subclass"
        )
    run_directory = canonical.parent
    diverted = retarget_case / "diverted-run"
    run_directory.rename(diverted)
    run_directory.symlink_to(diverted, target_is_directory=True)
    with pytest.raises(PermissionError, match="real directory"):
        capability.begin_manifest("development", DEVELOPMENT_SEEDS)

    inventory_case = tmp_path / "inventory"
    _patch_private_test_repository(inventory_case, monkeypatch, source, publication)
    ledger = _mint_development_ledger(source, publication)
    capability = ledger.capability()
    (qualification.development_ledger_path().parent / "unexpected.txt").write_text("x")
    with pytest.raises(PermissionError, match="unexpected"):
        capability.begin_manifest("development", DEVELOPMENT_SEEDS)


def test_qualification_capability_binds_reviewed_bytes_and_detects_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    _patch_private_test_repository(tmp_path, monkeypatch, source, publication)
    run_directory = qualification._canonical_run_directory()
    run_directory.mkdir(parents=True)
    checkpoint = qualification.canonical_checkpoint_path()
    report = qualification.canonical_development_report_path()
    development_ledger = qualification.development_ledger_path()
    checkpoint.write_bytes(b"checkpoint")
    report.write_bytes(b"report")
    development_ledger.write_bytes(b"ledger")
    bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()),
        "reviewed_development_report_sha256": sha256_bytes(report.read_bytes()),
        "reviewed_development_ledger_sha256": sha256_bytes(development_ledger.read_bytes()),
        "model_state_sha256": qualification.EMPTY_MODEL_STATE_SHA256,
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    validator_calls: list[str] = []
    monkeypatch.setattr(
        qualification,
        "validate_development_evidence",
        lambda *_args, **_kwargs: validator_calls.append("report") or {"passed": True},
    )
    monkeypatch.setattr(
        qualification,
        "validate_development_ledger",
        lambda *_args, **_kwargs: validator_calls.append("ledger"),
    )
    monkeypatch.setattr(
        qualification,
        "validate_checkpoint_evidence",
        lambda *_args, **_kwargs: validator_calls.append("checkpoint"),
    )
    reviewed_seal = qualification._mint_reviewed_development_seal(
        bindings,
        report={},
        ledger_record={},
        checkpoint_payload_value={},
        config=load_config(CONFIG_PATH),
        source=source,
        publication=publication,
    )
    authorization = qualification._mint_run_authorization(
        "qualification", bindings, reviewed_seal=reviewed_seal
    )
    with pytest.raises(PermissionError, match="fresh exact review seal"):
        qualification._mint_run_authorization(
            "qualification", bindings, reviewed_seal=reviewed_seal
        )
    ledger = qualification._QualificationLedger(
        authorization,
        reviewed_seal,
        bindings,
    )
    assert validator_calls == ["report", "ledger", "checkpoint"]
    capability = ledger.begin_access("selector")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="reviewed checkpoint differs"):
        capability.begin_manifest("selector", SELECTOR_SEEDS)

    with pytest.raises(PermissionError, match="exact run authorization"):
        qualification._QualificationLedger(object(), object(), bindings)


def _fail_only_report_replacement(
    monkeypatch: pytest.MonkeyPatch,
    report_path: Path,
) -> None:
    durable_replace = qualification._durable_replace

    def replacement(path: Path, contents: bytes, *, mode: int = 0o600) -> None:
        if path == report_path:
            raise OSError("injected failed-report replacement failure")
        durable_replace(path, contents, mode=mode)

    monkeypatch.setattr(qualification, "_durable_replace", replacement)


def test_development_error_ledger_never_binds_stale_passing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    _patch_private_test_repository(tmp_path, monkeypatch, source, publication)
    ledger = _mint_development_ledger(source, publication)
    report_path = qualification.canonical_development_report_path()
    stale_passing = qualification._report_bytes(
        {"artifact_kind": "stale-development", "passed": True, "review_ready": True}
    )
    report_path.write_bytes(stale_passing)
    stale_digest = sha256_bytes(stale_passing)
    _fail_only_report_replacement(monkeypatch, report_path)

    report = {"artifact_kind": "development-in-progress", "passed": True, "review_ready": True}
    qualification._record_development_exception(
        report_path=report_path,
        report=report,
        ledger=ledger,
        error=RuntimeError("injected after-report failure"),
    )

    assert report["passed"] is False
    assert report["review_ready"] is False
    assert report_path.read_bytes() == stale_passing
    assert ledger.record["status"] == "error"
    assert ledger.record["outcome"] == "failed"
    assert ledger.record["report_sha256"] is None
    assert stale_digest not in ledger._serialized().decode("utf-8")


def test_qualification_error_ledger_never_binds_stale_passing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    _patch_private_test_repository(tmp_path, monkeypatch, source, publication)
    run_directory = qualification._canonical_run_directory()
    run_directory.mkdir(parents=True)
    checkpoint = qualification.canonical_checkpoint_path()
    development_report = qualification.canonical_development_report_path()
    development_ledger = qualification.development_ledger_path()
    checkpoint.write_bytes(b"checkpoint")
    development_report.write_bytes(b"development-report")
    development_ledger.write_bytes(b"development-ledger")
    bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()),
        "reviewed_development_report_sha256": sha256_bytes(development_report.read_bytes()),
        "reviewed_development_ledger_sha256": sha256_bytes(development_ledger.read_bytes()),
        "model_state_sha256": qualification.EMPTY_MODEL_STATE_SHA256,
        "certificate_sha256": FROZEN_CERTIFICATE_SHA256,
    }
    monkeypatch.setattr(
        qualification,
        "validate_development_evidence",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        qualification,
        "validate_development_ledger",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        qualification,
        "validate_checkpoint_evidence",
        lambda *_args, **_kwargs: None,
    )
    reviewed_seal = qualification._mint_reviewed_development_seal(
        bindings,
        report={},
        ledger_record={},
        checkpoint_payload_value={},
        config=load_config(CONFIG_PATH),
        source=source,
        publication=publication,
    )
    authorization = qualification._mint_run_authorization(
        "qualification", bindings, reviewed_seal=reviewed_seal
    )
    ledger = qualification._QualificationLedger(authorization, reviewed_seal, bindings)
    ledger.begin_access("selector")
    report_path = qualification.canonical_qualification_report_path()
    stale_passing = qualification._report_bytes(
        {"artifact_kind": "stale-qualification", "passed": True}
    )
    report_path.write_bytes(stale_passing)
    stale_digest = sha256_bytes(stale_passing)
    _fail_only_report_replacement(monkeypatch, report_path)

    report = {"artifact_kind": "qualification-in-progress", "passed": True}
    qualification._record_qualification_exception(
        report_path=report_path,
        report=report,
        ledger=ledger,
        error=RuntimeError("injected after-report failure"),
    )

    assert report["passed"] is False
    assert report["protected_data_materialized"] is True
    assert report["stopped_after"] == "selector"
    assert report_path.read_bytes() == stale_passing
    assert ledger.record["status"] == "error"
    assert ledger.record["outcome"] == "failed"
    assert ledger.record["report_sha256"] is None
    assert stale_digest not in ledger._serialized().decode("utf-8")


def test_failed_report_digest_requires_exact_intended_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    publication = _publication(source)
    _patch_private_test_repository(tmp_path, monkeypatch, source, publication)
    report_path = qualification.canonical_development_report_path()
    report = {"artifact_kind": "failed-development", "passed": False}
    intended = qualification._report_bytes(report)

    created_digest = qualification._persist_failed_report(report_path, report, label="development")
    assert report_path.read_bytes() == intended
    assert created_digest == sha256_bytes(intended)

    replacement = {"artifact_kind": "failed-development-replaced", "passed": False}
    replacement_bytes = qualification._report_bytes(replacement)
    replaced_digest = qualification._persist_failed_report(
        report_path, replacement, label="development"
    )
    assert report_path.read_bytes() == replacement_bytes
    assert replaced_digest == sha256_bytes(replacement_bytes)


def test_restricted_checkpoint_loader_rejects_untrusted_pickle_global() -> None:
    buffer = io.BytesIO()
    torch.save({"unsafe": _UnsafeCheckpointValue()}, buffer)
    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        qualification._load_checkpoint_payload(buffer.getvalue())


def test_canonical_artifact_paths_define_exactly_five_run_files() -> None:
    files = {
        qualification.canonical_development_report_path(),
        qualification.canonical_checkpoint_path(),
        qualification.development_ledger_path(),
        qualification.canonical_qualification_report_path(),
        qualification.qualification_ledger_path(),
    }
    assert len(files) == 5
    expected_parent = REPOSITORY_ROOT / "runs" / "rgbd_two_visible_orbital_camera_v1"
    assert {path.parent for path in files} == {expected_parent}
    assert all(path.is_absolute() for path in files)
