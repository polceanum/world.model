from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
import struct
from pathlib import Path
from typing import Any

import pytest
import torch

import world_model.training.rgbd_identifiable_drag_scene as scene

EXPECTED_DRAG_PAIRS = {
    "development": ((9, 44), (16, 51), (23, 58), (30, 65)),
    "selector": ((17, 57), (10, 64), (31, 43), (24, 50)),
    "confirmation": ((25, 63), (32, 56), (11, 49), (18, 42)),
    "final_test": ((33, 48), (26, 41), (19, 62), (12, 55)),
}
EXPECTED_AB = {
    "development": ((-3, -3), (-1, -1), (1, 1), (3, 3)),
    "selector": ((-3, 3), (-1, 1), (1, -1), (3, -3)),
    "confirmation": ((-3, -1), (-1, -3), (1, 3), (3, 1)),
    "final_test": ((-3, 1), (-1, 3), (1, -3), (3, -1)),
}


def test_formal_constructor_is_seedless_strict_and_immutable() -> None:
    assert tuple(inspect.signature(scene.scene_specification).parameters) == ("split", "ordinal")
    with pytest.raises(TypeError):
        scene.scene_specification(True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        scene.scene_specification("public_feasibility", 0)
    for ordinal in (True, 1.0, "0"):
        with pytest.raises(TypeError):
            scene.scene_specification("development", ordinal)  # type: ignore[arg-type]
    for ordinal in (-1, scene.SCENES_PER_SPLIT):
        with pytest.raises(IndexError):
            scene.scene_specification("development", ordinal)

    specification = scene.scene_specification("development", 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        specification.ordinal = 1  # type: ignore[misc]
    first = specification.position_tensor()
    first.zero_()
    assert not torch.equal(first, specification.position_tensor())


def test_public_feasibility_namespace_is_nominally_separate_from_formal_splits() -> None:
    public = scene.public_feasibility_specification(0)
    formal = scene.scene_specification("development", 0)
    assert type(public) is scene.PublicFeasibilitySceneSpecification
    assert type(formal) is scene.IdentifiableDragSceneSpecification
    assert public.evidence_role == "public_feasibility_only"
    assert formal.evidence_role == "governed_development"
    assert public.theta0 == 0.0
    assert formal.theta0 == math.pi / 16.0
    assert public.drag_slot_numerators == (8, 45)
    assert formal.drag_slot_numerators == (9, 44)
    with pytest.raises(TypeError):
        scene.orbital_camera_frame(public, 0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        scene.initial_sphere_state(public)  # type: ignore[arg-type]


def test_exact_gf4_geometry_drag_tables_and_fresh_formal_roles() -> None:
    expected_offsets = (
        math.pi / 16.0,
        3.0 * math.pi / 16.0,
        5.0 * math.pi / 16.0,
        7.0 * math.pi / 16.0,
    )
    assert expected_offsets == scene.SPLIT_PHASE_OFFSETS_RADIANS
    assert scene.SPLIT_DRAG_NUMERATOR_SHIFTS == (1, 2, 3, 4)
    assert scene.SPLIT_EVIDENCE_ROLE == {
        "development": "governed_development",
        "selector": "held_out_preflight_only",
        "confirmation": "held_out_preflight_only",
        "final_test": "held_out_preflight_only",
    }
    for split_index, split in enumerate(scene.SPLITS):
        for primitive_index in range(scene.PRIMITIVES_PER_SPLIT):
            specification = scene.scene_specification(split, primitive_index * 16)
            a, b = EXPECTED_AB[split][primitive_index]
            assert (specification.a, specification.b) == (a, b)
            assert specification.drag_slot_numerators == EXPECTED_DRAG_PAIRS[split][primitive_index]
            assert specification.position_numerators == (
                (-650 + 12 * a, 460 + 8 * b, -280 + 8 * b),
                (650 + 10 * b, 1640 + 8 * a, 280 - 8 * a),
            )
            permuted = primitive_index ^ scene.GF4_M3[split_index]
            sign_x = 1 if primitive_index % 2 == 0 else -1
            sign_z = 1 if permuted % 2 == 0 else -1
            assert specification.velocity_numerators == (
                (sign_x * (230 + 6 * a), 90 + 5 * b, sign_z * (75 - 4 * a)),
                (sign_x * (205 + 5 * b), 75 + 4 * a, sign_z * (65 - 3 * b)),
            )


def test_all_governed_camera_offsets_and_drag_truths_are_fresh() -> None:
    consumed_offsets = scene.CONSUMED_PUBLIC_CAMERA_PHASE_OFFSETS_RADIANS
    assert consumed_offsets == (0.0, math.pi / 8.0, math.pi / 4.0)
    assert not set(scene.SPLIT_PHASE_OFFSETS_RADIANS) & set(consumed_offsets)
    consumed_initial_phases = {
        (base + offset) % (2.0 * math.pi)
        for base in scene.CAMERA_PHASES_RADIANS
        for offset in consumed_offsets
    }
    governed_initial_phases = {
        (base + offset) % (2.0 * math.pi)
        for base in scene.CAMERA_PHASES_RADIANS
        for offset in scene.SPLIT_PHASE_OFFSETS_RADIANS
    }
    assert consumed_initial_phases.isdisjoint(governed_initial_phases)

    consumed_drag_truths = set(scene.LOW_DRAG_NUMERATORS) | set(scene.HIGH_DRAG_NUMERATORS)
    for lows, highs in zip(
        scene.SPLIT_LOW_DRAG_NUMERATORS,
        scene.SPLIT_HIGH_DRAG_NUMERATORS,
        strict=True,
    ):
        assert consumed_drag_truths.isdisjoint(lows)
        assert consumed_drag_truths.isdisjoint(highs)


def test_drag_slot_twins_change_only_drag_role() -> None:
    signatures: set[str] = set()
    for split in scene.SPLITS:
        for ordinal in range(scene.SCENES_PER_SPLIT):
            specification = scene.scene_specification(split, ordinal)
            signatures.add(scene.scene_signature(specification))
            twin = scene.scene_specification(split, scene.counterfactual_twin_ordinal(ordinal))
            assert specification.position_numerators == twin.position_numerators
            assert specification.velocity_numerators == twin.velocity_numerators
            assert specification.albedo == twin.albedo
            assert specification.palette_swapped == twin.palette_swapped
            assert specification.camera_stratum == twin.camera_stratum
            assert specification.theta0 == twin.theta0
            assert specification.drag_slot_numerators == twin.drag_slot_numerators[::-1]
            assert scene.counterfactual_twin_ordinal(twin.ordinal) == ordinal
    assert len(signatures) == scene.TOTAL_SCENES


def test_family_balance_is_exact_per_split_without_cross_split_cartesian_claim() -> None:
    certificate = scene.scene_balance_certificate()
    assert certificate["total_scene_count"] == 256
    assert certificate["unique_scene_signature_count"] == 256
    assert certificate["base_geometry_count"] == 16
    assert certificate["drag_labelled_physical_trajectory_count"] == 32
    assert certificate["unique_split_role_low_high_pair_count"] == 16
    assert certificate["counterfactual_pair_count"] == 128
    assert certificate["governed_drag_truth_count"] == 32
    assert certificate["governed_drag_truth_unique_count"] == 32
    assert certificate["governed_drag_truth_cross_split_match_count"] == 0
    assert certificate["base_geometry_source_count"] == 16
    assert certificate["base_geometry_cross_split_match_count"] == 0
    assert certificate["drag_labelled_physical_source_count"] == 32
    assert certificate["drag_labelled_physical_cross_split_match_count"] == 0
    assert certificate["governed_drag_level_public_match_count"] == 0
    assert certificate["governed_drag_shift_consumed_match_count"] == 0
    for split_index, split in enumerate(scene.SPLITS):
        values = certificate["per_split"][split]
        assert values["scene_count"] == values["unique_scene_signature_count"] == 64
        assert values["primitive_histogram"] == {str(index): 16 for index in range(4)}
        assert values["counterfactual_histogram"] == {"0": 32, "1": 32}
        assert values["camera_stratum_histogram"] == {str(index): 8 for index in range(8)}
        assert values["palette_swap_histogram"] == {"false": 32, "true": 32}
        assert values["unique_low_high_pair_count"] == 4
        assert values["numeric_drag_truth_count"] == 8
        assert values["base_geometry_count"] == 4
        assert values["drag_labelled_physical_source_count"] == 8
        assert values["low_drag_numerator_histogram"] == {
            str(value): 16 for value in scene.SPLIT_LOW_DRAG_NUMERATORS[split_index]
        }
        assert values["high_drag_numerator_histogram"] == {
            str(value): 16 for value in scene.SPLIT_HIGH_DRAG_NUMERATORS[split_index]
        }
        expected_slot = {
            str(value): 8
            for value in (
                *scene.SPLIT_LOW_DRAG_NUMERATORS[split_index],
                *scene.SPLIT_HIGH_DRAG_NUMERATORS[split_index],
            )
        }
        assert values["slot_drag_numerator_histogram"] == {
            "0": expected_slot,
            "1": expected_slot,
        }


@pytest.mark.parametrize(
    ("tensor", "expected_bytes"),
    (
        (torch.tensor((1.5, -2.25), dtype=torch.float32), struct.pack("<ff", 1.5, -2.25)),
        (torch.tensor((1, -2), dtype=torch.int64), struct.pack("<qq", 1, -2)),
        (torch.tensor((0, 255), dtype=torch.uint8), struct.pack("<BB", 0, 255)),
    ),
)
def test_tensor_digest_is_explicit_contiguous_little_endian(
    tensor: torch.Tensor,
    expected_bytes: bytes,
) -> None:
    digest = hashlib.sha256()
    scene._update_tensor_digest(digest, tensor)
    expected = hashlib.sha256(
        str(tuple(tensor.shape)).encode("ascii")
        + str(tensor.dtype).encode("ascii")
        + expected_bytes
    ).hexdigest()
    assert digest.hexdigest() == expected


def test_camera_law_and_manual_recurrence_are_exact_source_functions() -> None:
    specification = scene.scene_specification("final_test", 63)
    frames = [
        scene.orbital_camera_frame(specification, frame_index / scene.FRAME_RATE_HZ)
        for frame_index in range(scene.FRAME_COUNT)
    ]
    camera_positions = torch.stack([frame.position for frame in frames])
    radii = torch.linalg.vector_norm(camera_positions[:, (0, 2)], dim=-1)
    torch.testing.assert_close(radii, torch.full_like(radii, 4.6), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(
        camera_positions[:, 1], torch.full((scene.FRAME_COUNT,), 2.15), atol=0, rtol=0
    )

    trajectory = scene.manual_physical_trajectory(specification)
    assert trajectory.positions.shape == (56, 2, 3)
    assert trajectory.velocities.shape == (56, 2, 3)
    assert trajectory.substep_positions.shape == (331, 2, 3)
    assert trajectory.substep_velocities.shape == (331, 2, 3)
    assert torch.equal(trajectory.positions, trajectory.substep_positions[::6])
    assert torch.equal(trajectory.velocities, trajectory.substep_velocities[::6])
    times = torch.arange(56, dtype=torch.float32)[:, None, None] / scene.FRAME_RATE_HZ
    drag = specification.drag_tensor()[None]
    decay = torch.exp(-drag * times)
    direct_velocity = specification.velocity_tensor()[None] * decay
    direct_position = specification.position_tensor()[None] + specification.velocity_tensor()[
        None
    ] * (-torch.expm1(-drag * times) / drag)
    torch.testing.assert_close(trajectory.positions, direct_position, atol=3.0e-6, rtol=0)
    torch.testing.assert_close(trajectory.velocities, direct_velocity, atol=5.0e-7, rtol=0)


def test_module_does_not_construct_runtime_or_observation_packets() -> None:
    source_text = inspect.getsource(scene)
    for forbidden in (
        "from world_model.runtime",
        "from world_model.observations",
        "OnlineWorldModel",
        "make_rgbd_packet",
        "SphereWorld(",
    ):
        assert forbidden not in source_text
    assert scene.scene_family_certificate.__doc__ is not None


@pytest.mark.slow
def test_exhaustive_formal_independent_certificate_and_public_feasibility_only_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_advance = scene.advance_spheres
    original_renderer = scene.render_spheres
    physics_calls = 0
    formal_physics_calls = 0
    renderer_calls = 0

    formal_drag_pairs = {
        specification.drag_slot_numerators
        for split in scene.SPLITS
        for primitive_index in range(scene.PRIMITIVES_PER_SPLIT)
        for counterfactual_index in range(scene.COUNTERFACTUALS_PER_PRIMITIVE)
        for specification in (
            scene.scene_specification(
                split,
                primitive_index * 16 + counterfactual_index * 8,
            ),
        )
    }
    public_physical_truth: dict[
        int,
        tuple[
            scene.PublicFeasibilitySceneSpecification,
            torch.Tensor,
            torch.Tensor,
        ],
    ] = {}
    substeps_per_frame = scene.PHYSICS_RATE_HZ // scene.FRAME_RATE_HZ
    substep_seconds = 1.0 / scene.PHYSICS_RATE_HZ
    for source_family_index in range(len(scene.SPLITS)):
        for primitive_index in range(scene.PRIMITIVES_PER_SPLIT):
            for counterfactual_index in range(scene.COUNTERFACTUALS_PER_PRIMITIVE):
                ordinal = (
                    source_family_index * scene.SCENES_PER_SPLIT
                    + primitive_index * 16
                    + counterfactual_index * 8
                )
                specification = scene.public_feasibility_specification(ordinal)
                assert isinstance(
                    specification,
                    scene.PublicFeasibilitySceneSpecification,
                )
                position = specification.position_tensor()
                velocity = specification.velocity_tensor()
                drag = specification.drag_tensor()
                decay = torch.exp(-drag * substep_seconds)
                displacement_coefficient = -torch.expm1(-drag * substep_seconds) / drag
                positions = [position.clone()]
                velocities = [velocity.clone()]
                for substep_index in range((scene.FRAME_COUNT - 1) * substeps_per_frame):
                    position = position + velocity * displacement_coefficient
                    velocity = velocity * decay
                    if (substep_index + 1) % substeps_per_frame == 0:
                        positions.append(position.clone())
                        velocities.append(velocity.clone())
                public_physical_truth[ordinal] = (
                    specification,
                    torch.stack(positions),
                    torch.stack(velocities),
                )
    assert len(public_physical_truth) == 32

    def guarded_advance(*args: Any, **kwargs: Any) -> Any:
        nonlocal formal_physics_calls, physics_calls
        state, dt, config = args[:3]
        trajectory_index, frame_index = divmod(physics_calls, scene.FRAME_COUNT - 1)
        source_family_index, family_index = divmod(
            trajectory_index,
            scene.PRIMITIVES_PER_SPLIT * scene.COUNTERFACTUALS_PER_PRIMITIVE,
        )
        primitive_index, counterfactual_index = divmod(
            family_index,
            scene.COUNTERFACTUALS_PER_PRIMITIVE,
        )
        ordinal = (
            source_family_index * scene.SCENES_PER_SPLIT
            + primitive_index * 16
            + counterfactual_index * 8
        )
        expected_specification, expected_positions, expected_velocities = public_physical_truth[
            ordinal
        ]
        assert isinstance(expected_specification, scene.PublicFeasibilitySceneSpecification)
        observed_drag_numerators = tuple(
            round(float(value) * scene.DRAG_DENOMINATOR) for value in state.drag[:, 0]
        )
        formal_physics_calls += int(observed_drag_numerators in formal_drag_pairs)
        assert observed_drag_numerators == expected_specification.drag_slot_numerators
        assert torch.equal(state.drag, expected_specification.drag_tensor())
        assert torch.equal(state.albedo, expected_specification.albedo_tensor())
        assert torch.equal(state.object_id, torch.tensor((0, 1), dtype=torch.int64))
        torch.testing.assert_close(
            state.position,
            expected_positions[frame_index],
            atol=scene.MAXIMUM_PUBLIC_PHYSICS_ERROR,
            rtol=0,
        )
        torch.testing.assert_close(
            state.velocity,
            expected_velocities[frame_index],
            atol=scene.MAXIMUM_PUBLIC_PHYSICS_ERROR,
            rtol=0,
        )
        assert dt == 1.0 / scene.FRAME_RATE_HZ
        assert config.gravity == (0.0, 0.0, 0.0)
        assert torch.count_nonzero(kwargs["external_impulse"]) == 0
        physics_calls += 1
        return original_advance(*args, **kwargs)

    def guarded_renderer(*args: Any, **kwargs: Any) -> Any:
        nonlocal renderer_calls
        state, camera = args[:2]
        ordinal, frame_index = divmod(renderer_calls, scene.FRAME_COUNT)
        expected_specification = scene.public_feasibility_specification(ordinal)
        expected_camera = scene.public_feasibility_camera_frame(
            expected_specification, frame_index / scene.FRAME_RATE_HZ
        )
        assert torch.equal(state.drag, expected_specification.drag_tensor())
        assert torch.equal(camera.world_from_camera, expected_camera.world_from_camera)
        renderer_calls += 1
        return original_renderer(*args, **kwargs)

    monkeypatch.setattr(scene, "advance_spheres", guarded_advance)
    monkeypatch.setattr(scene, "render_spheres", guarded_renderer)
    certificate = scene.scene_family_certificate()
    assert physics_calls == 32 * (scene.FRAME_COUNT - 1)
    assert formal_physics_calls == 0
    assert renderer_calls == scene.PUBLIC_FEASIBILITY_SCENE_COUNT * scene.FRAME_COUNT
    assert certificate["certificate_sha256"] == scene.FROZEN_CERTIFICATE_SHA256
    assert certificate["exact_metadata_sha256"] == scene.FROZEN_METADATA_SHA256
    assert certificate["formal_manual_physical_trace_sha256"] == (
        scene.FROZEN_PHYSICAL_TRACE_SHA256
    )
    assert certificate["camera_trace_sha256"] == scene.FROZEN_CAMERA_TRACE_SHA256
    assert certificate["independent_raster_trace_sha256"] == scene.FROZEN_RASTER_TRACE_SHA256
    assert certificate["ordered_combined_scene_trace_sha256"] == (
        scene.FROZEN_COMBINED_TRACE_SHA256
    )
    trace_bindings = certificate["formal_trace_bindings"]
    expected_counts = {
        "physical": (8, 32),
        "camera": (8, 32),
        "raster": (64, 256),
        "combined": (64, 256),
    }
    expected_split_hashes = {
        "physical": scene.FROZEN_SPLIT_PHYSICAL_TRACE_SHA256,
        "camera": scene.FROZEN_SPLIT_CAMERA_TRACE_SHA256,
        "raster": scene.FROZEN_SPLIT_RASTER_TRACE_SHA256,
        "combined": scene.FROZEN_SPLIT_COMBINED_TRACE_SHA256,
    }
    for name, (per_split_count, global_count) in expected_counts.items():
        assert trace_bindings[name]["per_split_sha256"] == expected_split_hashes[name]
        assert trace_bindings[name]["per_split_trace_count"] == {
            split: per_split_count for split in scene.SPLITS
        }
        assert trace_bindings[name]["global_trace_count"] == global_count
        assert trace_bindings[name]["global_unique_trace_count"] == global_count
        assert trace_bindings[name]["cross_split_match_count"] == 0
    assert certificate["protocol"]["runtime_packets_constructed"] == 0
    assert certificate["protocol"]["truth_routed_to_runtime"] is False
    assert certificate["protocol"]["public_camera_source_sha256"] == (
        scene.PUBLIC_CAMERA_SOURCE_SHA256
    )
    assert certificate["physics"]["formal_public_physics_call_count"] == 0
    assert certificate["camera"]["governed_consumed_phase_match_count"] == 0
    assert certificate["camera"]["governed_trace_cardinal_match_count"] == 0
    assert certificate["raster"]["evaluated_frame_count"] == 256 * 56
    assert certificate["raster"]["public_feasibility_renderer_frame_count"] == 256 * 56
    assert certificate["raster"]["formal_public_renderer_call_count"] == 0
    assert certificate["raster"]["public_renderer_mismatch_count"] == 0
    assert certificate["raster"]["overlap_frame_count"] == 0
    assert certificate["raster"]["minimum_visible_fraction"] == 1.0
    assert certificate["raster"]["minimum_full_support_pixels"] >= 20
    assert certificate["raster"]["minimum_continuous_silhouette_gap_pixels"] >= 4.0
    assert certificate["raster"]["minimum_image_boundary_clearance_pixels"] >= 6.0
    assert certificate["physics"]["minimum_world_surface_gap_m"] >= 1.0
    assert certificate["physics"]["minimum_world_boundary_clearance_m"] >= 0.15
    assert certificate["physics"]["minimum_drag_excitation_m"] >= 0.015
    assert certificate["physics"]["public_feasibility_event_count"] == 0
    assert certificate["physics"]["public_feasibility_contact_count"] == 0
    assert certificate["counterfactual"] == {
        "pair_count": 128,
        "drag_swap_mismatch_count": 0,
        "non_drag_source_mismatch_count": 0,
        "camera_trace_mismatch_count": 0,
        "palette_depends_on_counterfactual_index": False,
    }


def test_public_source_bindings_are_frozen() -> None:
    root = Path(scene.__file__).resolve().parents[1]
    assert hashlib.sha256((root / "simulator" / "camera.py").read_bytes()).hexdigest() == (
        scene.PUBLIC_CAMERA_SOURCE_SHA256
    )
    assert hashlib.sha256((root / "simulator" / "physics.py").read_bytes()).hexdigest() == (
        scene.PUBLIC_PHYSICS_SOURCE_SHA256
    )
    assert hashlib.sha256((root / "simulator" / "renderer.py").read_bytes()).hexdigest() == (
        scene.PUBLIC_RENDERER_SOURCE_SHA256
    )
