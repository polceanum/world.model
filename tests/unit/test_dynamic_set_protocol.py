from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import pytest

from world_model.training.dynamic_set_protocol import (
    EXAMPLES_PER_UPDATE,
    MACROCYCLE_UPDATES,
    PHYSICAL_CELL_COUNT,
    PHYSICAL_CELLS,
    PHYSICAL_SPLIT_SIZES,
    PLANNING_SPLIT_SIZES,
    SELECTION_SCORE_WEIGHTS,
    PlanningOracleCertificate,
    macrocycle_cell_indices,
    physical_manifest,
    planning_manifest,
    selection_score,
    validate_frozen_manifests,
)


def test_physical_manifest_has_exact_cell_and_split_balance() -> None:
    assert len(PHYSICAL_CELLS) == PHYSICAL_CELL_COUNT == 22
    assert len(set(PHYSICAL_CELLS)) == 22
    assert sum(cell.object_count == 1 for cell in PHYSICAL_CELLS) == 2
    assert all(not cell.contact for cell in PHYSICAL_CELLS if cell.object_count == 1)

    for split, size in PHYSICAL_SPLIT_SIZES.items():
        rows = physical_manifest(split)
        assert len(rows) == size
        counts = Counter(row.cell_index for row in rows)
        assert set(counts) == set(range(22))
        assert set(counts.values()) == {size // 22}
        for cell_index in range(22):
            cell_rows = [row for row in rows if row.cell_index == cell_index]
            action_counts = Counter(row.known_action for row in cell_rows)
            cell = PHYSICAL_CELLS[cell_index]
            if cell.contact:
                regimes = Counter((row.contact_origin, row.known_action) for row in cell_rows)
                assert set(regimes) == {
                    ("natural", False),
                    ("action_induced", True),
                }
                assert len(set(regimes.values())) == 1
                assert [(row.contact_origin, row.known_action) for row in cell_rows[:2]] == [
                    ("natural", False),
                    ("action_induced", True),
                ]
                if cell.dynamic_membership:
                    for regime in regimes:
                        assert {
                            row.lifecycle_schedule
                            for row in cell_rows
                            if (row.contact_origin, row.known_action) == regime
                        } == {"birth", "removal", "remove_then_birth"}
                    causal_lifecycle_geometry = Counter(
                        (
                            row.contact_origin,
                            row.known_action,
                            row.lifecycle_schedule,
                            row.contact_geometry,
                        )
                        for row in cell_rows
                    )
                    assert set(causal_lifecycle_geometry) == {
                        (contact_origin, known_action, lifecycle, geometry)
                        for contact_origin, known_action in (
                            ("natural", False),
                            ("action_induced", True),
                        )
                        for lifecycle in ("birth", "removal", "remove_then_birth")
                        for geometry in ("head_on", "glancing")
                    }
                    assert (
                        max(causal_lifecycle_geometry.values())
                        - min(causal_lifecycle_geometry.values())
                        <= 1
                    )
                else:
                    causal_geometry = Counter(
                        (row.contact_origin, row.known_action, row.contact_geometry)
                        for row in cell_rows
                    )
                    assert set(causal_geometry) == {
                        (contact_origin, known_action, geometry)
                        for contact_origin, known_action in (
                            ("natural", False),
                            ("action_induced", True),
                        )
                        for geometry in ("head_on", "glancing")
                    }
                    assert max(causal_geometry.values()) - min(causal_geometry.values()) <= 1
            else:
                assert action_counts[False] == action_counts[True]
            action_rows = [row for row in cell_rows if row.known_action]
            target_time = Counter(
                (row.action_target_rank, row.action_time_stratum) for row in action_rows
            )
            assert set(target_time) == set(product(range(cell.object_count), range(4)))
            assert max(target_time.values()) - min(target_time.values()) <= 1
            camera_axes = Counter(row.camera_stratum // 2 for row in action_rows)
            assert set(camera_axes) == set(range(4))
            assert max(camera_axes.values()) - min(camera_axes.values()) <= 1
            if cell.dynamic_membership:
                lifecycle_counts = Counter(row.lifecycle_schedule for row in cell_rows)
                assert set(lifecycle_counts) == {"birth", "removal", "remove_then_birth"}
                assert max(lifecycle_counts.values()) - min(lifecycle_counts.values()) <= 1


def test_eleven_update_macrocycle_covers_every_cell_then_equalizes_duplicates() -> None:
    duplicated = Counter()
    for update_index in range(MACROCYCLE_UPDATES):
        groups = macrocycle_cell_indices(update_index)
        assert len(groups) == 6
        assert all(len(group) == 4 for group in groups)
        flat = [cell for group in groups for cell in group]
        assert len(flat) == EXAMPLES_PER_UPDATE == 24
        counts = Counter(flat)
        assert set(counts) == set(range(22))
        assert sorted(counts.values()).count(2) == 2
        duplicated.update(cell for cell, count in counts.items() if count == 2)
    assert duplicated == Counter({cell: 1 for cell in range(22)})


def test_planning_manifests_cross_every_declared_public_stratum() -> None:
    for split, size in PLANNING_SPLIT_SIZES.items():
        rows = planning_manifest(split)
        assert len(rows) == size
        assert {row.object_count for row in rows} == set(range(1, 7))
        assert {row.previously_dynamic for row in rows} == {False, True}
        assert {row.candidate_count for row in rows} == {8, 32}
        assert {row.action_time_stratum for row in rows} == set(range(4))
        assert {row.camera_stratum for row in rows} == set(range(8))
        assert {row.goal_direction for row in rows} == set(range(6))
        assert {row.candidate_induced_contact for row in rows if row.object_count > 1} == {
            False,
            True,
        }
        assert all(0 <= row.target_rank < row.object_count for row in rows)
        assert all(row.minimum_normalized_winner_margin == 0.05 for row in rows)

        for object_count in range(1, 7):
            cardinality_rows = [row for row in rows if row.object_count == object_count]
            assert len(cardinality_rows) == size // 6
            factor_values = {
                "previously_dynamic": (False, True),
                "candidate_induced_contact": ((False,) if object_count == 1 else (False, True)),
                "target_rank": tuple(range(object_count)),
                "action_time_stratum": tuple(range(4)),
                "camera_stratum": tuple(range(8)),
                "goal_direction": tuple(range(6)),
                "candidate_count": (8, 32),
            }
            for factor_name, expected_values in factor_values.items():
                counts = Counter(getattr(row, factor_name) for row in cardinality_rows)
                assert set(counts) == set(expected_values)
                assert max(counts.values()) - min(counts.values()) <= 1
            for left, right in combinations(factor_values, 2):
                observed_pairs = {
                    (getattr(row, left), getattr(row, right)) for row in cardinality_rows
                }
                expected_pairs = set(product(factor_values[left], factor_values[right]))
                assert observed_pairs == expected_pairs, (split, object_count, left, right)


def test_compositional_ood_uses_held_out_primitive_couplings() -> None:
    physical_id = physical_manifest("development")
    physical_ood = physical_manifest("compositional_ood")
    assert len(physical_id) == len(physical_ood)
    id_physical_contexts = {
        (row.cell_index, row.known_action, row.lifecycle_schedule, row.camera_stratum)
        for row in physical_id
    }
    for reference, held_out in zip(physical_id, physical_ood, strict=True):
        assert held_out.cell_index == reference.cell_index
        assert held_out.known_action == reference.known_action
        assert held_out.lifecycle_schedule == reference.lifecycle_schedule
        assert held_out.camera_stratum // 2 == reference.camera_stratum // 2
        assert held_out.camera_stratum % 2 != reference.camera_stratum % 2
        assert (
            held_out.cell_index,
            held_out.known_action,
            held_out.lifecycle_schedule,
            held_out.camera_stratum,
        ) not in id_physical_contexts
    assert {row.camera_stratum for row in physical_ood} == {
        row.camera_stratum for row in physical_id
    }

    planning_id = planning_manifest("development")
    planning_ood = planning_manifest("compositional_ood")
    assert len(planning_id) == len(planning_ood)
    id_planning_contexts = {
        (
            row.object_count,
            row.previously_dynamic,
            row.candidate_induced_contact,
            row.target_rank,
            row.action_time_stratum,
            row.camera_stratum,
            row.goal_direction // 2,
            row.goal_direction % 2,
            row.candidate_count,
        )
        for row in planning_id
    }
    for reference, held_out in zip(planning_id, planning_ood, strict=True):
        assert held_out.object_count == reference.object_count
        assert held_out.previously_dynamic == reference.previously_dynamic
        assert held_out.candidate_induced_contact == reference.candidate_induced_contact
        assert held_out.target_rank == reference.target_rank
        assert held_out.action_time_stratum == reference.action_time_stratum
        assert held_out.camera_stratum == reference.camera_stratum
        assert held_out.goal_direction // 2 == reference.goal_direction // 2
        assert held_out.goal_direction % 2 != reference.goal_direction % 2
        assert held_out.candidate_count == reference.candidate_count
        assert (
            held_out.object_count,
            held_out.previously_dynamic,
            held_out.candidate_induced_contact,
            held_out.target_rank,
            held_out.action_time_stratum,
            held_out.camera_stratum,
            held_out.goal_direction // 2,
            held_out.goal_direction % 2,
            held_out.candidate_count,
        ) not in id_planning_contexts
    assert {row.goal_direction for row in planning_ood} == {
        row.goal_direction for row in planning_id
    }


def test_all_manifest_namespaces_are_source_frozen_and_disjoint() -> None:
    validate_frozen_manifests()


def test_planning_oracle_certificate_requires_finite_unique_margin() -> None:
    certificate = PlanningOracleCertificate.from_costs(
        [1.0, 1.2, 1.5],
        winner_succeeds=True,
    )
    assert certificate.winner_index == 0
    assert certificate.normalized_winner_margin == pytest.approx(1.0 / 6.0)
    assert certificate.winner_succeeds

    with pytest.raises(ValueError, match="unique winner margin"):
        PlanningOracleCertificate.from_costs(
            [1.0, 1.01, 2.0],
            winner_succeeds=False,
        )
    with pytest.raises(ValueError, match="finite"):
        PlanningOracleCertificate.from_costs(
            [1.0, float("nan")],
            winner_succeeds=False,
        )


def test_frozen_selection_score_is_complete_lower_is_better() -> None:
    assert sum(SELECTION_SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    components = {name: float(index + 1) for index, name in enumerate(SELECTION_SCORE_WEIGHTS)}
    expected = sum(
        SELECTION_SCORE_WEIGHTS[name] * components[name] for name in SELECTION_SCORE_WEIGHTS
    )
    assert selection_score(components) == pytest.approx(expected)
    with pytest.raises(ValueError, match="components differ"):
        selection_score({"planning_error": 1.0})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        selection_score({name: -1.0 for name in SELECTION_SCORE_WEIGHTS})
