from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import world_model.training.rgbd_known_action_scene as scene
from world_model.dynamics.actions import WorldImpulseAction


def _all_specifications() -> list[scene.KnownActionSceneSpecification]:
    return [
        scene.scene_specification(split, ordinal)
        for split in scene.SPLITS
        for ordinal in range(scene.BUNDLES_PER_SPLIT)
    ]


def test_exact_seedless_family_cardinality_and_split_partition() -> None:
    assert scene.COEFFICIENT_LEVELS == (-7, -5, 5, 7)
    assert scene.PRIMITIVES_PER_SPLIT == 4
    assert scene.HANDLE_ROLES == 2
    assert scene.CAMERA_STRATA == 8
    assert scene.CANDIDATES_PER_BUNDLE == 8
    assert scene.BUNDLES_PER_SPLIT == 64
    assert scene.TOTAL_PRIMITIVE_PROFILES == 16
    assert scene.TOTAL_BUNDLES == 256
    assert scene.TOTAL_CANDIDATES == 2048

    pairs = [pair for split in scene.SPLITS for pair in scene.SPLIT_PRIMITIVE_PAIRS[split]]
    assert len(pairs) == len(set(pairs)) == 16
    assert set(pairs) == {
        (a, b) for a in scene.COEFFICIENT_LEVELS for b in scene.COEFFICIENT_LEVELS
    }
    assert scene.SPLIT_PRIMITIVE_PAIRS == {
        "development": ((-7, -7), (-5, 5), (5, -5), (7, 7)),
        "selector": ((-7, 7), (-5, -5), (5, 5), (7, -7)),
        "confirmation": ((-7, -5), (-5, 7), (5, -7), (7, 5)),
        "final_test": ((-7, 5), (-5, -7), (5, 7), (7, -5)),
    }


@pytest.mark.parametrize("split", scene.SPLITS)
def test_ordinal_mapping_manifests_and_balances(split: str) -> None:
    specifications = [
        scene.scene_specification(split, ordinal) for ordinal in range(scene.BUNDLES_PER_SPLIT)
    ]
    assert [specification.ordinal for specification in specifications] == list(range(64))
    assert len(scene.split_manifest(split)) == 64
    assert len(scene.split_scene_signatures(split)) == 64
    assert len(set(scene.split_scene_signatures(split))) == 64
    for ordinal, specification in enumerate(specifications):
        primitive, remainder = divmod(ordinal, 16)
        role, camera = divmod(remainder, 8)
        assert specification.primitive_index == primitive
        assert specification.handle_role == role
        assert specification.camera_stratum == camera
        assert specification.ordinal == 16 * primitive + 8 * role + camera
        assert specification.a, specification.b
    assert {
        key: sum(spec.primitive_index == int(key) for spec in specifications)
        for key in map(str, range(4))
    } == {str(index): 16 for index in range(4)}
    assert {
        key: sum(spec.handle_role == int(key) for spec in specifications)
        for key in map(str, range(2))
    } == {
        "0": 32,
        "1": 32,
    }
    assert {
        key: sum(spec.camera_stratum == int(key) for spec in specifications)
        for key in map(str, range(8))
    } == {str(index): 8 for index in range(8)}
    assert sum(spec.palette_swapped for spec in specifications) == 32
    assert {
        q: sum(spec.optimal_canonical_index == q for spec in specifications) for q in range(8)
    } == {q: 8 for q in range(8)}


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_scene_rejects_noninteger_ordinals(bad: object) -> None:
    with pytest.raises(TypeError, match="ordinal must be an integer"):
        scene.scene_specification("development", bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1, 64])
def test_scene_rejects_out_of_range_ordinals(bad: int) -> None:
    with pytest.raises(IndexError):
        scene.scene_specification("development", bad)


def test_scene_rejects_invalid_split_without_aliasing() -> None:
    with pytest.raises(TypeError, match="split must be a string"):
        scene.scene_specification(0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown known-action split"):
        scene.scene_specification("test", 0)


def test_rational_geometry_is_literal_and_float32_only_at_materialization() -> None:
    for specification in _all_specifications():
        a, b = specification.a, specification.b
        assert specification.position_numerators == (
            (-450 + 6 * a, 400 + 2 * b, -300 + 4 * b),
            (450 + 5 * b, 1750 + 2 * a, 300 - 4 * a),
        )
        assert specification.velocity_numerators == (
            (180 + 2 * a, 48 + b, 16 + a),
            (-172 + 2 * b, -32 + a, -12 + b),
        )
        assert specification.position_tensor().dtype == torch.float32
        assert specification.velocity_tensor().dtype == torch.float32
        torch.testing.assert_close(
            specification.position_tensor(),
            torch.tensor(specification.position_numerators, dtype=torch.float32) / 1000,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            specification.velocity_tensor(),
            torch.tensor(specification.velocity_numerators, dtype=torch.float32) / 4000,
            rtol=0.0,
            atol=0.0,
        )


def test_role_twins_share_prefix_camera_palette_and_toggle_only_task_target() -> None:
    invariant = (
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
    pairs = 0
    for split in scene.SPLITS:
        for primitive in range(4):
            for camera in range(8):
                first = scene.scene_specification(split, 16 * primitive + camera)
                second = scene.scene_specification(
                    split,
                    scene.role_twin_ordinal(first.ordinal),
                )
                assert all(getattr(first, name) == getattr(second, name) for name in invariant)
                assert first.handle_role == 0
                assert second.handle_role == 1
                assert first.physical_controlled_row != second.physical_controlled_row
                assert torch.equal(
                    scene.manual_prefix_trajectory(first).positions,
                    scene.manual_prefix_trajectory(second).positions,
                )
                first_camera = scene.pure_orbital_camera_frame(
                    first.camera_stratum,
                    1.25,
                )
                second_camera = scene.pure_orbital_camera_frame(
                    second.camera_stratum,
                    1.25,
                )
                assert first_camera.timestamp == second_camera.timestamp
                for field in (
                    "position",
                    "target",
                    "world_from_camera",
                    "camera_from_world",
                    "intrinsics",
                ):
                    assert torch.equal(
                        getattr(first_camera, field),
                        getattr(second_camera, field),
                    )
                pairs += 1
    assert pairs == 128


def test_palette_handle_mapping_and_cosine_margin_are_role_independent() -> None:
    palette = torch.tensor(scene.PALETTE, dtype=torch.float64)
    normalized = palette / torch.linalg.vector_norm(palette, dim=-1, keepdim=True)
    cross_distance = 1.0 - float((normalized[0] * normalized[1]).sum())
    assert cross_distance == pytest.approx(0.6057722264231578)
    assert cross_distance > scene.MINIMUM_CROSS_HANDLE_COSINE_DISTANCE
    for specification in _all_specifications():
        w = (
            specification.primitive_index
            + specification.phase_index
            + specification.direction_index
        ) & 1
        assert specification.palette_swapped is bool(w)
        assert specification.physical_controlled_row == specification.handle_role ^ w
        assert (
            specification.albedo[specification.physical_controlled_row]
            == (scene.HANDLE_PROTOTYPES[specification.handle_role])
        )


def test_candidate_rational_times_sign_bits_equal_energy_and_public_value() -> None:
    for specification in _all_specifications():
        rank_a = scene.COEFFICIENT_RANK[specification.a]
        rank_b = scene.COEFFICIENT_RANK[specification.b]
        expected_magnitudes = (
            21 + rank_b,
            19 + rank_a,
            23 + ((rank_a + rank_b) % 4),
        )
        assert specification.action_delay_numerator == 6 + rank_a
        assert specification.impulse_magnitude_numerators == expected_magnitudes
        assert specification.action_delay_seconds in (0.30, 0.35, 0.40, 0.45)
        assert specification.action_delay_seconds not in scene.HORIZONS_SECONDS
        candidates = [scene.candidate_specification(specification, q) for q in range(8)]
        energies = [
            candidate.impulse_tensor(dtype=torch.float64).square().sum() for candidate in candidates
        ]
        assert all(torch.equal(energies[0], energy) for energy in energies)
        for q, candidate in enumerate(candidates):
            expected_signs = tuple(1 if q & (1 << axis) else -1 for axis in range(3))
            assert candidate.impulse_numerators == tuple(
                sign * magnitude
                for sign, magnitude in zip(
                    expected_signs,
                    expected_magnitudes,
                    strict=True,
                )
            )
            assert candidate.timestamp_numerator == (scene.ANCHOR_FRAME_INDEX + 6 + rank_a)
            assert candidate.timestamp_denominator == scene.FRAME_RATE_HZ
            assert candidate.observable_handle_role == specification.handle_role
            assert (
                candidate.observable_handle_prototype
                == (scene.HANDLE_PROTOTYPES[specification.handle_role])
            )
            assert candidate.physical_target_row == specification.physical_controlled_row

        for order in ("canonical", "pi", "rho"):
            resolved_id = torch.tensor([91], dtype=torch.int64)
            action = scene.world_impulse_action(
                specification,
                0,
                resolved_persistent_object_id=resolved_id,
                order=order,
                dtype=torch.float64,
            )
            q = scene.display_to_canonical(
                specification.camera_stratum,
                0,
                order=order,
            )
            candidate = candidates[q]
            assert isinstance(action, WorldImpulseAction)
            assert action.timestamp.shape == (1,)
            assert action.object_id.tolist() == [91]
            assert action.impulse_world.shape == (1, 3)
            assert action.impulse_world.dtype == torch.float64
            torch.testing.assert_close(
                action.impulse_world[0],
                candidate.impulse_tensor(dtype=torch.float64),
                rtol=0.0,
                atol=0.0,
            )


def test_pi_and_rho_are_exact_distinct_role_independent_permutations() -> None:
    pi_winners = {display: 0 for display in range(8)}
    rho_winners = {display: 0 for display in range(8)}
    for camera in range(8):
        pi = scene.candidate_order(camera, order="pi")
        rho = scene.candidate_order(camera, order="rho")
        assert sorted(pi) == list(range(8))
        assert sorted(rho) == list(range(8))
        assert pi != rho
        for display in range(8):
            assert pi[display] == (display + 2 * camera) % 8
            assert rho[display] == (5 * display + 3 + 2 * camera) % 8
            assert (
                scene.canonical_to_display(
                    camera,
                    pi[display],
                    order="pi",
                )
                == display
            )
            assert (
                scene.canonical_to_display(
                    camera,
                    rho[display],
                    order="rho",
                )
                == display
            )
    for specification in _all_specifications():
        pi_winners[
            scene.canonical_to_display(
                specification.camera_stratum,
                specification.optimal_canonical_index,
                order="pi",
            )
        ] += 1
        rho_winners[
            scene.canonical_to_display(
                specification.camera_stratum,
                specification.optimal_canonical_index,
                order="rho",
            )
        ] += 1
        twin = scene.scene_specification(
            specification.split,
            scene.role_twin_ordinal(specification.ordinal),
        )
        assert scene.candidate_order(
            specification.camera_stratum,
            order="pi",
        ) == scene.candidate_order(twin.camera_stratum, order="pi")
        assert scene.candidate_order(
            specification.camera_stratum,
            order="rho",
        ) == scene.candidate_order(twin.camera_stratum, order="rho")
    assert pi_winners == {display: 32 for display in range(8)}
    assert rho_winners == {display: 32 for display in range(8)}


def test_candidate_truth_is_right_continuous_exactly_once_and_isolated() -> None:
    for specification in _all_specifications():
        baseline = scene.manual_prefix_trajectory(specification)
        for q in range(8):
            candidate = scene.candidate_specification(specification, q)
            trajectory = scene.manual_candidate_trajectory(specification, q)
            frame = scene.ANCHOR_FRAME_INDEX + specification.action_delay_numerator
            target = specification.physical_controlled_row
            distractor = 1 - target
            assert trajectory.positions.shape == (56, 2, 3)
            assert trajectory.velocities.shape == (56, 2, 3)
            assert trajectory.substep_positions.shape == (331, 2, 3)
            assert trajectory.substep_velocities.shape == (331, 2, 3)
            assert int(trajectory.action_events.sum()) == 1
            assert int(trajectory.substep_action_events.sum()) == 1
            assert bool(trajectory.action_events[frame, target])
            assert not bool(trajectory.action_events[:, distractor].any())
            assert torch.equal(
                trajectory.positions[: frame + 1],
                baseline.positions[: frame + 1],
            )
            assert torch.equal(
                trajectory.velocities[:frame],
                baseline.velocities[:frame],
            )
            torch.testing.assert_close(
                trajectory.velocities[frame, target] - baseline.velocities[frame, target],
                candidate.impulse_tensor(),
                rtol=0.0,
                atol=2.0e-9,
            )
            assert torch.equal(
                trajectory.positions[:, distractor],
                baseline.positions[:, distractor],
            )
            assert torch.equal(
                trajectory.velocities[:, distractor],
                baseline.velocities[:, distractor],
            )


def test_terminal_goals_costs_and_two_permutations_are_exhaustively_unique() -> None:
    canonical_gap = math.inf
    opposite_gap = math.inf
    for specification in _all_specifications():
        canonical = scene.candidate_costs(specification)
        opposite = scene.candidate_costs(specification, opposite_goal=True)
        assert int(canonical.argmin()) == specification.optimal_canonical_index
        assert int(opposite.argmin()) == specification.optimal_canonical_index ^ 7
        assert float(canonical.min()) == 0.0
        assert float(opposite.min()) == 0.0
        canonical_gap = min(canonical_gap, float(canonical.sort().values[1]))
        opposite_gap = min(opposite_gap, float(opposite.sort().values[1]))
        for order in ("pi", "rho"):
            indices = torch.tensor(scene.candidate_order(specification.camera_stratum, order=order))
            displayed = scene.candidate_costs(specification, order=order)
            displayed_opposite = scene.candidate_costs(
                specification,
                order=order,
                opposite_goal=True,
            )
            assert torch.equal(displayed, canonical[indices])
            assert torch.equal(displayed_opposite, opposite[indices])
            assert int(displayed.argmin()) == scene.canonical_to_display(
                specification.camera_stratum,
                specification.optimal_canonical_index,
                order=order,
            )
    assert canonical_gap > 0.0
    assert opposite_gap > 0.0


def test_camera_law_is_exact_known_orbit_and_calibration() -> None:
    intrinsics = scene._make_intrinsics()
    for camera in range(8):
        previous = None
        for frame in range(scene.FRAME_COUNT):
            value = scene.pure_orbital_camera_frame(camera, frame / scene.FRAME_RATE_HZ)
            assert value.timestamp == frame / scene.FRAME_RATE_HZ
            assert torch.equal(value.intrinsics, intrinsics)
            assert float(value.position[1]) == pytest.approx(scene.CAMERA_HEIGHT_M)
            assert float(torch.linalg.vector_norm(value.position[[0, 2]])) == pytest.approx(
                scene.CAMERA_RADIUS_M,
                abs=2.0e-6,
            )
            torch.testing.assert_close(
                value.world_from_camera @ value.camera_from_world,
                torch.eye(4),
                rtol=0.0,
                atol=2.0e-6,
            )
            if previous is not None:
                translation = float(torch.linalg.vector_norm(value.position - previous.position))
                assert (
                    scene.MINIMUM_CAMERA_TRANSLATION_STEP_M
                    <= translation
                    <= scene.MAXIMUM_CAMERA_TRANSLATION_STEP_M
                )
            previous = value


def test_formal_values_are_frozen_and_action_ids_are_only_caller_resolved() -> None:
    unswapped = scene.scene_specification("development", 0)
    swapped = scene.scene_specification("development", 2)
    assert unswapped.handle_role == swapped.handle_role == 0
    assert unswapped.palette_swapped is False
    assert swapped.palette_swapped is True
    assert unswapped.physical_controlled_row == 0
    assert swapped.physical_controlled_row == 1
    with pytest.raises(FrozenInstanceError):
        unswapped.handle_role = 1  # type: ignore[misc]

    first = scene.world_impulse_action(
        unswapped,
        0,
        resolved_persistent_object_id=torch.tensor([101], dtype=torch.int64),
        order="canonical",
    )
    second = scene.world_impulse_action(
        swapped,
        0,
        resolved_persistent_object_id=torch.tensor([307], dtype=torch.int64),
        order="canonical",
    )
    assert first.object_id.tolist() == [101]
    assert second.object_id.tolist() == [307]
    assert 101 not in {
        unswapped.handle_role,
        unswapped.physical_controlled_row,
    }
    assert 307 not in {
        swapped.handle_role,
        swapped.physical_controlled_row,
    }
    assert torch.equal(first.impulse_world, second.impulse_world)

    resolved_permutation = torch.tensor([503, 11], dtype=torch.int64)
    batched = scene.world_impulse_action(
        unswapped,
        0,
        resolved_persistent_object_id=resolved_permutation,
        order="canonical",
    )
    assert batched.object_id.tolist() == [503, 11]
    assert batched.object_id.data_ptr() != resolved_permutation.data_ptr()
    assert batched.timestamp.shape == (2,)
    assert batched.impulse_world.shape == (2, 3)
    assert torch.equal(batched.impulse_world[0], batched.impulse_world[1])

    with pytest.raises(TypeError, match="torch.Tensor"):
        scene.world_impulse_action(
            unswapped,
            0,
            resolved_persistent_object_id=7,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="torch.int64"):
        scene.world_impulse_action(
            unswapped,
            0,
            resolved_persistent_object_id=torch.tensor([7], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        scene.world_impulse_action(
            unswapped,
            0,
            resolved_persistent_object_id=torch.tensor([-1], dtype=torch.int64),
        )


def test_observable_metadata_and_heterogeneous_collation_exclude_evaluator_truth() -> None:
    specifications = [
        scene.scene_specification("development", 0),
        scene.scene_specification("selector", 11),
        scene.scene_specification("confirmation", 42),
        scene.scene_specification("final_test", 63),
    ]
    forbidden_fragments = (
        "object_id",
        "persistent_id",
        "physical",
        "controlled_row",
        "canonical",
        "optimal",
        "primitive",
        "truth",
        "position_numerators",
        "velocity_numerators",
    )

    def forbidden_key(key: str) -> bool:
        lowered = key.lower()
        return lowered == "id" or any(fragment in lowered for fragment in forbidden_fragments)

    def all_keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return [key for name, member in value.items() for key in (str(name), *all_keys(member))]
        if isinstance(value, list):
            return [key for member in value for key in all_keys(member)]
        return []

    for specification in specifications:
        metadata = scene.observable_task_metadata(specification, order="rho")
        keys = all_keys(metadata)
        assert not any(forbidden_key(key) for key in keys)
        assert metadata["observable_handle"] == {
            "role": specification.handle_role,
            "prototype": scene.HANDLE_PROTOTYPES[specification.handle_role],
        }
        assert len(metadata["candidate_actions"]) == 8
        assert metadata["goal_horizon_seconds"] == 2.0

    collated = scene.collate_observable_tasks(specifications, order="rho")
    assert set(collated) == {
        "observable_handle_role",
        "observable_handle_prototype",
        "candidate_timestamps",
        "candidate_impulses_world",
        "goal_positions_world",
        "goal_horizons",
    }
    assert collated["observable_handle_role"].shape == (4,)
    assert collated["observable_handle_prototype"].shape == (4, 3)
    assert collated["candidate_timestamps"].shape == (4, 8)
    assert collated["candidate_impulses_world"].shape == (4, 8, 3)
    assert collated["goal_positions_world"].shape == (4, 3)
    assert collated["goal_horizons"].shape == (4,)
    assert not any(forbidden_key(key) for key in collated)


def test_static_ast_has_no_rng_seed_or_public_execution_call() -> None:
    source_path = Path(scene.__file__)
    source = source_path.read_text()
    tree = ast.parse(source)
    forbidden_import_prefixes = (
        "random",
        "numpy",
        "world_model.runtime",
        "world_model.planning",
        "world_model.simulator",
        "world_model.dynamics.analytic_free_motion",
    )
    forbidden_calls = {
        "AnalyticFreeMotionDynamics",
        "OnlineWorldModel",
        "SphereWorld",
        "make_intrinsics",
        "look_at_world_from_camera",
        "render_spheres",
        "rollout",
        "predict",
        "plan_actions",
        "manual_seed",
        "seed",
        "rand",
        "randn",
        "randint",
    }
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                calls.append(function.id)
            elif isinstance(function, ast.Attribute):
                calls.append(function.attr)
    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden_import_prefixes
    )
    assert not (set(calls) & forbidden_calls)
    assert "DEVELOPMENT_SEEDS" not in source
    assert "SELECTOR_SEEDS" not in source
    assert "handle_object_id" not in source
    assert "observable_handle_object_id" not in source
    assert "tau2" not in source
    assert "j2" not in source
    materializer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "world_impulse_action"
    )
    forbidden_identity_attributes = {
        "handle_role",
        "physical_controlled_row",
        "physical_target_row",
    }
    assert (
        not {node.attr for node in ast.walk(materializer) if isinstance(node, ast.Attribute)}
        & forbidden_identity_attributes
    )


def test_digest_field_framing_is_domain_separated_and_little_endian() -> None:
    actual = hashlib.sha256()
    scene._digest_field(actual, "alpha", b"payload")
    expected = hashlib.sha256()
    expected.update(b"orpheus-known-action-v1\x00")
    expected.update(struct.pack("<I", 5))
    expected.update(b"alpha")
    expected.update(struct.pack("<Q", 7))
    expected.update(b"payload")
    assert actual.hexdigest() == expected.hexdigest()


def test_descriptor_is_json_literal_self_hashed_and_explicitly_single_action() -> None:
    descriptor = scene.certificate_descriptor()
    json.loads(json.dumps(descriptor, allow_nan=False))
    unsigned = dict(descriptor)
    descriptor_sha256 = unsigned.pop("descriptor_sha256")
    unsigned.pop("certificate_sha256")
    assert descriptor_sha256 == scene.canonical_sha256(unsigned)
    assert descriptor["action_candidates"]["single_action_per_candidate"] is True
    assert descriptor["action_candidates"]["old_two_action_schedule_absent"] is True
    assert (
        descriptor["action_candidates"]["scene_role_or_physical_row_used_as_persistent_id"] is False
    )
    assert descriptor["observable_runtime_surface"]["physical_controlled_row_exposed"] is False
    assert descriptor["observable_runtime_surface"]["canonical_winner_exposed"] is False
    assert descriptor["independent_truth"]["public_renderer_calls"] is False
    assert descriptor["independent_truth"]["public_physics_calls"] is False
    assert descriptor["independent_truth"]["public_runtime_calls"] is False


def test_exhaustive_certificate_covers_every_bundle_candidate_and_gate() -> None:
    certificate = scene._computed_scene_family_certificate()
    assert certificate["primitive_profile_count"] == 16
    assert certificate["bundle_count"] == 256
    assert certificate["candidate_count"] == 2048
    assert certificate["unique_prefix_profile_count"] == 16
    assert certificate["role_twin_pair_count"] == 128
    assert certificate["permutation_invariance_check_count"] == 512
    assert certificate["right_continuous_controlled_action_event_count"] == 2048
    assert certificate["right_continuous_distractor_action_event_count"] == 0
    assert certificate["zero_impulse_control_action_event_count"] == 0
    assert certificate["zero_impulse_control_state_delta"] == 0.0
    assert certificate["maximum_pre_action_state_delta"] == 0.0
    assert certificate["maximum_distractor_state_delta"] == 0.0
    assert certificate["minimum_unique_winner_terminal_position_sse_gap"] > 0.0
    assert certificate["minimum_opposite_goal_unique_winner_terminal_position_sse_gap"] > 0.0
    assert min(certificate["minimum_terminal_action_position_effect_m_by_axis"]) > 0.0
    assert min(certificate["minimum_terminal_action_velocity_effect_mps_by_axis"]) > 0.0
    assert (
        certificate["minimum_matched_handle_prototype_cosine"]
        >= scene.MINIMUM_MATCHED_HANDLE_COSINE
    )
    assert (
        certificate["minimum_cross_handle_prototype_cosine_distance"]
        >= scene.MINIMUM_CROSS_HANDLE_COSINE_DISTANCE
    )
    assert certificate["minimum_full_support_pixels"] >= scene.MINIMUM_FULL_SUPPORT_PIXELS
    assert (
        certificate["minimum_conic_enclosing_circle_gap_lower_bound_pixels"]
        >= scene.MINIMUM_CONIC_GAP_LOWER_BOUND_PIXELS
    )
    assert (
        certificate["minimum_conic_coordinate_extrema_boundary_clearance_pixels"]
        >= scene.MINIMUM_CONIC_BOUNDARY_CLEARANCE_PIXELS
    )
    assert certificate["minimum_world_surface_gap_m"] >= scene.MINIMUM_WORLD_SURFACE_GAP_M
    assert certificate["minimum_world_boundary_clearance_m"] >= scene.MINIMUM_WORLD_BOUNDARY_M
    assert (
        certificate["maximum_causal_prefix_projected_centre_step_pixels"]
        <= scene.MAXIMUM_PROJECTED_CENTRE_STEP_PIXELS
    )
    assert (
        certificate["maximum_no_action_projected_centre_step_pixels"]
        <= scene.MAXIMUM_PROJECTED_CENTRE_STEP_PIXELS
    )
    assert (
        certificate["maximum_acted_projected_centre_step_pixels"]
        <= scene.MAXIMUM_ACTED_PROJECTED_CENTRE_STEP_PIXELS
    )
    assert certificate["maximum_incremental_acted_projected_centre_step_pixels"] > 0.0
    assert certificate["minimum_floor_clearance_m"] > scene.BOUNDARY_CONTACT_TOLERANCE_M
    assert certificate["floor_contact_count"] == 0
    assert certificate["sleep_candidate_count"] == 0
    assert certificate["sleep_law"]["requires_floor_contact"] is True
    assert "substep_contact_proof" in certificate["trace_sha256"]
    assert certificate["public_renderer_calls_on_formal_values"] == 0
    assert certificate["public_physics_calls_on_formal_values"] == 0
    assert certificate["public_runtime_calls_on_formal_values"] == 0
    assert certificate["expected_contact_event_count"] == 0
    assert certificate["expected_collision_event_count"] == 0
    assert certificate["expected_removal_event_count"] == 0
    assert certificate["expected_sleep_event_count"] == 0

    if scene.FROZEN_CERTIFICATE_SHA256 != "UNFROZEN":
        assert (
            certificate["normalised_scene_source_sha256"]
            == scene.FROZEN_NORMALIZED_SCENE_SOURCE_SHA256
        )
        assert certificate["input_binding_sha256"] == scene.FROZEN_INPUT_BINDING_SHA256
        assert certificate["descriptor_sha256"] == scene.FROZEN_DESCRIPTOR_SHA256
        assert certificate["manifest_sha256"] == scene.FROZEN_MANIFEST_SHA256
        assert certificate["trace_sha256"] == scene.FROZEN_TRACE_SHA256
        assert certificate["certificate_sha256"] == scene.FROZEN_CERTIFICATE_SHA256
        assert scene.scene_family_certificate() == certificate


def test_frozen_certificate_revalidates_source_and_constants_after_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if scene.FROZEN_CERTIFICATE_SHA256 == "UNFROZEN":
        pytest.skip("post-freeze cache regression runs after literal freeze")
    baseline = scene.scene_family_certificate()
    assert baseline["certificate_sha256"] == scene.FROZEN_CERTIFICATE_SHA256

    with monkeypatch.context() as context:
        context.setattr(
            scene,
            "MINIMUM_FULL_SUPPORT_PIXELS",
            scene.MINIMUM_FULL_SUPPORT_PIXELS + 1,
        )
        with pytest.raises(RuntimeError, match="complete input binding"):
            scene.scene_family_certificate()

    source_path = Path(scene.__file__).resolve()
    original_read_bytes = Path.read_bytes

    def drifted_read_bytes(path: Path) -> bytes:
        contents = original_read_bytes(path)
        if path.resolve() == source_path:
            return contents + b"\n# simulated post-cache source drift\n"
        return contents

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", drifted_read_bytes)
        with pytest.raises(RuntimeError, match="scene source"):
            scene.scene_family_certificate()

    assert scene.scene_family_certificate() == baseline
