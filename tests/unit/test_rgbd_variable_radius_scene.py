from __future__ import annotations

import ast
import inspect
import json
import math
from dataclasses import replace
from typing import Any

import pytest
import torch

import world_model.training.rgbd_variable_radius_scene as scene_module
from world_model.training.rgbd_variable_radius_scene import (
    CAMERA_STRATA,
    FROZEN_CERTIFICATE_SHA256,
    FROZEN_SPLIT_TRACE_SHA256,
    FROZEN_TRACE_SHA256,
    PAIR_VARIANTS_PER_PRIMITIVE,
    PRIMITIVES_PER_SPLIT,
    RADIUS_DENOMINATOR,
    RADIUS_ROLES_PER_PRIMITIVE,
    SCENES_PER_SPLIT,
    SPLITS,
    TOTAL_SCENES,
    certificate_descriptor,
    counterfactual_twin_ordinal,
    manual_kinematic_trajectory,
    pair_variant_twin_ordinal,
    pure_orbital_camera_frame,
    reject_formal_public_api_input,
    scene_balance_certificate,
    scene_family_certificate,
    scene_metadata,
    scene_specification,
)

EXPECTED_MAPPING = {
    "development": (
        (-3, -3, ((411, 447), (421, 457))),
        (1, 1, ((431, 467), (441, 477))),
    ),
    "selector": (
        (-1, -1, ((413, 449), (419, 455))),
        (3, 3, ((433, 469), (439, 475))),
    ),
    "confirmation": (
        (-3, 3, ((415, 451), (425, 461))),
        (1, -1, ((427, 463), (437, 473))),
    ),
    "final_test": (
        (-1, 1, ((417, 453), (423, 459))),
        (3, -3, ((429, 465), (435, 471))),
    ),
}


def _literal_only(value: Any) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        assert math.isfinite(value)
        return
    if type(value) is list:
        for member in value:
            _literal_only(member)
        return
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for member in value.values():
            _literal_only(member)
        return
    raise AssertionError(f"non-literal descriptor value {type(value)!r}")


def test_axes_and_exact_split_mapping_are_frozen() -> None:
    assert len(SPLITS) == 4
    assert PRIMITIVES_PER_SPLIT == 2
    assert PAIR_VARIANTS_PER_PRIMITIVE == 2
    assert RADIUS_ROLES_PER_PRIMITIVE == 2
    assert CAMERA_STRATA == 8
    assert SCENES_PER_SPLIT == 64
    assert TOTAL_SCENES == 256
    assert RADIUS_DENOMINATOR == 2000
    for split, expected_rows in EXPECTED_MAPPING.items():
        actual_rows = []
        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            pairs = []
            for pair_variant in range(PAIR_VARIANTS_PER_PRIMITIVE):
                specification = scene_specification(
                    split,
                    primitive_index * 32 + pair_variant * 16,
                )
                assert specification.radius_pair_index == primitive_index * 2 + pair_variant
                pairs.append(
                    (
                        specification.low_radius_numerator,
                        specification.high_radius_numerator,
                    )
                )
            actual_rows.append((specification.a, specification.b, tuple(pairs)))
        assert tuple(actual_rows) == expected_rows


@pytest.mark.parametrize("split", SPLITS)
def test_radius_role_twins_swap_only_radius_slots(split: str) -> None:
    for primitive_index in range(PRIMITIVES_PER_SPLIT):
        for pair_variant in range(PAIR_VARIANTS_PER_PRIMITIVE):
            for camera_stratum in range(CAMERA_STRATA):
                ordinal = primitive_index * 32 + pair_variant * 16 + camera_stratum
                twin_ordinal = counterfactual_twin_ordinal(ordinal)
                assert twin_ordinal == ordinal ^ 8
                assert counterfactual_twin_ordinal(twin_ordinal) == ordinal
                first = scene_specification(split, ordinal)
                second = scene_specification(split, twin_ordinal)
                assert first.radius_role == 0
                assert second.radius_role == 1
                assert first.radius_slot_numerators == second.radius_slot_numerators[::-1]
                first_metadata = scene_metadata(first)
                second_metadata = scene_metadata(second)
                first_metadata["ordinal"] = second_metadata["ordinal"]
                first_metadata["radius_role"] = second_metadata["radius_role"]
                first_metadata["radius_rational"]["slot_numerators"] = second_metadata[
                    "radius_rational"
                ]["slot_numerators"]
                assert first_metadata == second_metadata


@pytest.mark.parametrize("split", SPLITS)
def test_pair_variant_twins_change_both_truths_and_nothing_else(split: str) -> None:
    for primitive_index in range(PRIMITIVES_PER_SPLIT):
        for radius_role in range(RADIUS_ROLES_PER_PRIMITIVE):
            for camera_stratum in range(CAMERA_STRATA):
                ordinal = primitive_index * 32 + radius_role * 8 + camera_stratum
                twin_ordinal = pair_variant_twin_ordinal(ordinal)
                assert twin_ordinal == ordinal ^ 16
                assert pair_variant_twin_ordinal(twin_ordinal) == ordinal
                first = scene_specification(split, ordinal)
                second = scene_specification(split, twin_ordinal)
                assert first.pair_variant == 0
                assert second.pair_variant == 1
                assert set(first.radius_slot_numerators).isdisjoint(second.radius_slot_numerators)
                first_metadata = scene_metadata(first)
                second_metadata = scene_metadata(second)
                first_metadata["ordinal"] = second_metadata["ordinal"]
                first_metadata["pair_variant"] = second_metadata["pair_variant"]
                first_metadata["radius_rational"] = second_metadata["radius_rational"]
                assert first_metadata == second_metadata


def test_family_has_disjoint_truths_geometries_and_radius_labelled_sources() -> None:
    balance = scene_balance_certificate()
    assert balance["total_scene_count"] == 256
    assert balance["unique_scene_signature_count"] == 256
    assert balance["base_geometry_count"] == 8
    assert balance["base_geometry_cross_split_match_count"] == 0
    assert balance["radius_labelled_source_count"] == 32
    assert balance["radius_labelled_source_cross_split_match_count"] == 0
    assert balance["radius_truth_count"] == 32
    assert balance["radius_truth_unique_count"] == 32
    assert balance["radius_truth_cross_split_match_count"] == 0
    assert balance["consumed_fixed_radius_match_count"] == 0
    assert balance["counterfactual_pair_count"] == 128
    assert balance["pair_variant_counterfactual_count"] == 128

    truth_sets = []
    geometry_pairs = set()
    for split in SPLITS:
        truths = set()
        for primitive_index in range(PRIMITIVES_PER_SPLIT):
            for pair_variant in range(PAIR_VARIANTS_PER_PRIMITIVE):
                specification = scene_specification(
                    split,
                    primitive_index * 32 + pair_variant * 16,
                )
                truths.update(
                    (
                        specification.low_radius_numerator,
                        specification.high_radius_numerator,
                    )
                )
                geometry_pairs.add((specification.a, specification.b))
        assert len(truths) == 8
        assert 420 not in truths
        assert all(numerator % 2 == 1 for numerator in truths)
        truth_sets.append(truths)
        per_split = balance["per_split"][split]
        assert per_split["scene_count"] == 64
        assert per_split["unique_scene_signature_count"] == 64
        assert per_split["primitive_histogram"] == {"0": 32, "1": 32}
        assert per_split["pair_variant_histogram"] == {"0": 32, "1": 32}
        assert per_split["radius_role_histogram"] == {"0": 32, "1": 32}
        assert per_split["camera_stratum_histogram"] == {str(index): 8 for index in range(8)}
        assert per_split["palette_swap_histogram"] == {"false": 32, "true": 32}
        assert per_split["base_geometry_count"] == 2
        assert per_split["radius_labelled_source_count"] == 8
        assert all(
            histogram == {"0": 8, "1": 8}
            for histogram in per_split["slot_truth_histogram"].values()
        )
        assert all(
            histogram == {str(index): 2 for index in range(8)}
            for histogram in per_split["camera_truth_histogram"].values()
        )
        assert all(
            histogram == {"palette_0": 8, "palette_1": 8}
            for histogram in per_split["colour_truth_histogram"].values()
        )
    assert len(set().union(*truth_sets)) == 32
    assert all(
        not (truth_sets[left] & truth_sets[right])
        for left in range(len(truth_sets))
        for right in range(left + 1, len(truth_sets))
    )
    assert len(geometry_pairs) == 8


def test_ordinal_and_camera_inputs_are_strict() -> None:
    with pytest.raises(TypeError):
        scene_specification("development", True)
    with pytest.raises(TypeError):
        scene_specification("development", 1.0)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        scene_specification("development", -1)
    with pytest.raises(IndexError):
        scene_specification("development", 64)
    with pytest.raises(TypeError):
        scene_specification(1, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        scene_specification("unknown", 0)
    with pytest.raises(TypeError):
        pair_variant_twin_ordinal(True)
    with pytest.raises(TypeError):
        pure_orbital_camera_frame(True, 0.0)
    with pytest.raises(IndexError):
        pure_orbital_camera_frame(8, 0.0)
    with pytest.raises(TypeError):
        pure_orbital_camera_frame(0, True)
    with pytest.raises(ValueError):
        pure_orbital_camera_frame(0, math.nan)


def test_materialisation_is_defensive_and_twins_share_kinematics_bitwise() -> None:
    first = scene_specification("development", 0)
    role_twin = scene_specification("development", 8)
    pair_twin = scene_specification("development", 16)
    first_position = first.position_tensor()
    first_position[0, 0] = 100.0
    assert scene_specification("development", 0).position_tensor()[0, 0] != 100.0
    first_trace = manual_kinematic_trajectory(first)
    for twin in (role_twin, pair_twin):
        twin_trace = manual_kinematic_trajectory(twin)
        assert torch.equal(first_trace.positions, twin_trace.positions)
        assert torch.equal(first_trace.velocities, twin_trace.velocities)
        assert torch.equal(first_trace.substep_positions, twin_trace.substep_positions)
        assert torch.equal(first_trace.substep_velocities, twin_trace.substep_velocities)
        assert first.albedo == twin.albedo


def test_formal_types_are_rejected_at_public_api_guard() -> None:
    specification = scene_specification("development", 0)
    trajectory = manual_kinematic_trajectory(specification)
    camera = pure_orbital_camera_frame(0, 0.0)
    for value, api_name in (
        (specification, "renderer"),
        (trajectory, "physics"),
        (camera, "camera constructor"),
    ):
        with pytest.raises(PermissionError, match="may not cross"):
            reject_formal_public_api_input(value, api_name=api_name)
    reject_formal_public_api_input(object(), api_name="public feasibility control")
    with pytest.raises(TypeError):
        reject_formal_public_api_input(object(), api_name="")


def test_scene_module_has_no_public_simulator_imports_or_calls() -> None:
    tree = ast.parse(inspect.getsource(scene_module))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name.startswith("world_model.simulator") for name in imported_modules)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint(
        {
            "advance_spheres",
            "render_spheres",
            "look_at_world_from_camera",
            "invert_rigid_transform",
            "make_intrinsics",
        }
    )


def test_pure_certificate_does_not_cross_monkeypatched_public_apis(monkeypatch: Any) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("formal certificate crossed a public simulator API")

    monkeypatch.setattr("world_model.simulator.physics.advance_spheres", forbidden)
    monkeypatch.setattr("world_model.simulator.renderer.render_spheres", forbidden)
    monkeypatch.setattr("world_model.simulator.camera.look_at_world_from_camera", forbidden)
    monkeypatch.setattr("world_model.simulator.camera.invert_rigid_transform", forbidden)
    monkeypatch.setattr("world_model.simulator.camera.make_intrinsics", forbidden)
    scene_module._computed_scene_family_certificate_bytes_cached.cache_clear()
    certificate = scene_family_certificate()
    assert certificate["public_physics_calls_on_formal_scenes"] == 0
    assert certificate["public_renderer_calls_on_formal_scenes"] == 0
    assert certificate["public_camera_constructor_calls_on_formal_scenes"] == 0


def test_cached_certificate_revalidates_bound_source_bytes(monkeypatch: Any) -> None:
    first = scene_family_certificate()
    camera_source = (
        scene_module.Path(scene_module.__file__).resolve().parents[1] / "simulator" / "camera.py"
    )
    original_read_bytes = scene_module.Path.read_bytes

    def mutated_read_bytes(path: Any) -> bytes:
        payload = original_read_bytes(path)
        if path.resolve() == camera_source:
            return payload + b"\n"
        return payload

    with monkeypatch.context() as mutation:
        mutation.setattr(scene_module.Path, "read_bytes", mutated_read_bytes)
        with pytest.raises(RuntimeError, match="source differs"):
            scene_family_certificate()

    second = scene_family_certificate()
    assert second["certificate_sha256"] == first["certificate_sha256"]


def test_balance_and_certificate_recompute_after_nested_table_mutation() -> None:
    first_balance = scene_balance_certificate()
    first_certificate = scene_family_certificate()
    original_development = scene_module.RADIUS_PAIR_NUMERATORS["development"]
    mutated_development = (
        (((420, 447), original_development[0][1])),
        original_development[1],
    )
    cache_before = scene_module._computed_scene_family_certificate_bytes_cached.cache_info()
    scene_module.RADIUS_PAIR_NUMERATORS["development"] = mutated_development
    try:
        with pytest.raises(RuntimeError, match="consumed fixed-radius"):
            scene_balance_certificate()
        with pytest.raises(RuntimeError, match="consumed fixed-radius"):
            scene_family_certificate()
    finally:
        scene_module.RADIUS_PAIR_NUMERATORS["development"] = original_development

    cache_after = scene_module._computed_scene_family_certificate_bytes_cached.cache_info()
    assert cache_after.misses == cache_before.misses + 1
    assert scene_balance_certificate()["balance_sha256"] == first_balance["balance_sha256"]
    assert (
        scene_family_certificate()["certificate_sha256"] == first_certificate["certificate_sha256"]
    )


def test_certificate_hashes_counts_and_observability_margins_are_frozen() -> None:
    certificate = scene_family_certificate()
    assert certificate["certificate_sha256"] == FROZEN_CERTIFICATE_SHA256
    assert certificate["trace_sha256"] == FROZEN_TRACE_SHA256
    assert certificate["split_trace_sha256"] == FROZEN_SPLIT_TRACE_SHA256
    assert set(certificate["source_bindings"]) == {
        "accepted_orbital_qualification",
        "camera",
        "physics",
        "renderer",
    }
    assert all(len(value) == 64 for value in certificate["source_bindings"].values())
    assert certificate["scene_count"] == 256
    assert certificate["scene_frame_count"] == 14_336
    assert certificate["kinematic_trajectory_count"] == 8
    assert certificate["unordered_pair_labelled_trajectory_count"] == 16
    assert certificate["radius_labelled_trajectory_count"] == 32
    assert certificate["shared_camera_trace_count"] == 8
    assert certificate["joint_raster_trace_count"] == 256
    assert certificate["joint_conic_trace_count"] == 256
    assert certificate["joint_combined_trace_count"] == 256
    assert certificate["expected_physical_event_count"] == 0
    assert certificate["minimum_full_support_pixels"] == 21.0
    assert certificate["minimum_conic_enclosing_circle_gap_lower_bound_pixels"] == pytest.approx(
        9.209182018763137, abs=1.0e-12
    )
    assert certificate[
        "minimum_conic_coordinate_extrema_boundary_clearance_pixels"
    ] == pytest.approx(15.101709847131012, abs=1.0e-12)
    assert certificate["minimum_world_surface_gap_m"] == pytest.approx(
        1.107572078704834,
        abs=1.0e-12,
    )
    assert certificate["minimum_world_boundary_clearance_m"] == pytest.approx(
        0.15850000083446503,
        abs=1.0e-12,
    )
    assert certificate["minimum_radius_bound_clearance_m"] == pytest.approx(
        0.011500000953674316,
        abs=1.0e-12,
    )
    assert certificate["minimum_radius_pair_separation_m"] == 0.018
    assert certificate["maximum_conic_relative_boundary_residual"] < 5.0e-19
    assert certificate["maximum_conic_centre_residual"] < 1.5e-16
    assert certificate["maximum_conic_shape_residual"] < 3.6e-16
    assert certificate["maximum_sphere_fit_condition"] < 10.54
    assert certificate["maximum_sphere_fit_relative_residual"] < 7.4e-6
    assert certificate["maximum_sphere_fit_radius_error_m"] < 5.7e-6
    assert certificate["maximum_sphere_fit_centre_error_m"] < 7.3e-6
    reproduction = certificate["accepted_orbital_law_reproduction"]
    assert reproduction["copied_metadata_rows_exact"] is True
    assert reproduction["copied_fixed_drag_trace_exact"] is True
    assert reproduction["copied_camera_trace_exact"] is True
    assert len(reproduction["selected_kinematic_trace_sha256"]) == 8
    assert len(reproduction["shared_camera_trace_sha256"]) == 8


def test_certificate_and_literal_descriptor_are_defensive() -> None:
    descriptor = certificate_descriptor()
    _literal_only(descriptor)
    json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert descriptor["authority"] == "literal_source_descriptor_only"
    assert descriptor["certificate_sha256"] == FROZEN_CERTIFICATE_SHA256
    assert descriptor["scene_axes"] == {
        "primitives_per_split": 2,
        "pair_variants_per_primitive": 2,
        "radius_roles_per_primitive": 2,
        "camera_strata": 8,
        "scenes_per_split": 64,
        "total_scenes": 256,
    }
    assert descriptor["ordinal_mapping"] == {
        "primitive_index": "ordinal//32",
        "pair_variant": "(ordinal%32)//16",
        "radius_role": "(ordinal%16)//8",
        "camera_stratum": "ordinal%8",
        "radius_role_twin": "ordinal xor 8",
        "unordered_pair_variant_twin": "ordinal xor 16",
    }
    assert descriptor["camera"]["phase_policy"] == "shared_across_splits_to_isolate_radius"
    assert descriptor["formal_public_api_policy"] == {
        "public_physics_on_formal_values": False,
        "public_renderer_on_formal_values": False,
        "public_camera_constructor_on_formal_values": False,
        "independent_fixed_drag_recurrence": True,
        "independent_stable_near_root_raster": True,
        "independent_pinhole_sphere_conic_geometry": True,
        "independent_centered_algebraic_sphere_fit": True,
    }
    assert descriptor["determinism_scope"] == {
        "scope": "exact_frozen_torch_build_and_platform_cpu_float32_state_float64_conic",
        "torch_version": "2.9.0a0+gitcbe1a35",
        "python_version": "3.10.20",
        "platform_system": "Darwin",
        "platform_machine": "x86_64",
        "byteorder": "little",
        "device": "cpu",
        "state_dtype": "torch.float32",
        "conic_dtype": "torch.float64",
        "cross_build_or_cross_platform_digest_portability_claim": False,
    }
    descriptor_text = json.dumps(descriptor).lower()
    assert "seed" not in descriptor_text
    assert "manifest" not in descriptor_text

    first = scene_family_certificate()
    first["trace_sha256"]["metadata"] = "0" * 64
    second = scene_family_certificate()
    assert second["trace_sha256"]["metadata"] == FROZEN_TRACE_SHA256["metadata"]
    descriptor["scene_axes"]["total_scenes"] = 0
    assert certificate_descriptor()["scene_axes"]["total_scenes"] == 256


def test_public_guard_rejects_nominally_forged_formal_values() -> None:
    specification = scene_specification("development", 0)
    forged = replace(specification, ordinal=1)
    with pytest.raises(PermissionError):
        reject_formal_public_api_input(forged, api_name="renderer")
