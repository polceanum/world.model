from __future__ import annotations

from collections import Counter, defaultdict

from world_model.training.dynamic_set_campaign import TRAINING_CACHE_FULL_COVERAGE_UPDATES
from world_model.training.dynamic_set_protocol import physical_manifest
from world_model.training.dynamic_set_sampling import DynamicSetMicrobatchSchedule


def test_microbatch_schedule_is_random_access_balanced_and_resume_exact() -> None:
    rows = physical_manifest("training")
    schedule = DynamicSetMicrobatchSchedule(rows, seed=1_610)
    resumed = DynamicSetMicrobatchSchedule(rows, seed=1_610)

    for update_index in (0, 1, 10, 11, 517, 32_767):
        batches = schedule.microbatches_for_update(update_index)
        assert batches == resumed.microbatches_for_update(update_index)
        assert len(batches) == 6
        assert all(len(batch) == 4 for batch in batches)
        cell_counts: dict[int, int] = defaultdict(int)
        for batch in batches:
            for dataset_index in batch:
                cell_counts[rows[dataset_index].cell_index] += 1
        assert set(cell_counts) == set(range(22))
        assert sorted(cell_counts.values()).count(2) == 2


def test_each_cell_pool_is_shuffled_without_replacement_within_an_epoch() -> None:
    rows = physical_manifest("training")
    schedule = DynamicSetMicrobatchSchedule(rows, seed=42)
    selected: dict[int, list[int]] = defaultdict(list)
    for update_index in range(100):
        for batch in schedule.microbatches_for_update(update_index):
            for dataset_index in batch:
                selected[rows[dataset_index].cell_index].append(dataset_index)

    assert set(selected) == set(range(22))
    assert all(len(indices) == len(set(indices)) for indices in selected.values())


def test_production_cache_fill_bound_visits_every_manifest_row() -> None:
    rows = physical_manifest("training")
    schedule = DynamicSetMicrobatchSchedule(rows, seed=1_610)
    expected_by_cell: dict[int, set[int]] = defaultdict(set)
    selected_by_cell: dict[int, set[int]] = defaultdict(set)
    for dataset_index, row in enumerate(rows):
        expected_by_cell[row.cell_index].add(dataset_index)
    assert {len(pool) for pool in expected_by_cell.values()} == {3_000}

    for update_index in range(TRAINING_CACHE_FULL_COVERAGE_UPDATES):
        for batch in schedule.microbatches_for_update(update_index):
            for dataset_index in batch:
                selected_by_cell[rows[dataset_index].cell_index].add(dataset_index)

    assert selected_by_cell == expected_by_cell


def test_schedule_seed_changes_order_but_not_cell_contract() -> None:
    rows = physical_manifest("training")
    first = DynamicSetMicrobatchSchedule(rows, seed=1).microbatches_for_update(0)
    second = DynamicSetMicrobatchSchedule(rows, seed=2).microbatches_for_update(0)
    assert first != second
    assert [rows[index].cell_index for batch in first for index in batch] == [
        rows[index].cell_index for batch in second for index in batch
    ]


def test_exact_64_row_screen_prefix_supports_every_cell_without_substitution() -> None:
    rows = physical_manifest("training")[:64]
    schedule = DynamicSetMicrobatchSchedule(rows, seed=1_610)

    for update_index in (0, 1, 10, 11, 511):
        batches = schedule.microbatches_for_update(update_index)
        selected = [dataset_index for batch in batches for dataset_index in batch]
        assert len(selected) == 24
        assert all(0 <= dataset_index < 64 for dataset_index in selected)
        counts = Counter(rows[dataset_index].cell_index for dataset_index in selected)
        assert set(counts) == set(range(22))
        assert sorted(counts.values()).count(2) == 2


def test_every_macrocycle_rotates_action_contact_and_lifecycle_strata_evenly() -> None:
    rows = physical_manifest("training")
    schedule = DynamicSetMicrobatchSchedule(rows, seed=1_610)

    for macrocycle in (0, 1, 249, 250):
        selected: dict[int, list] = defaultdict(list)
        for update_index in range(macrocycle * 11, (macrocycle + 1) * 11):
            for batch in schedule.microbatches_for_update(update_index):
                for dataset_index in batch:
                    row = rows[dataset_index]
                    selected[row.cell_index].append(row)

        assert all(len(cell_rows) == 12 for cell_rows in selected.values())
        for cell_rows in selected.values():
            if cell_rows[0].contact:
                assert Counter((row.contact_origin, row.known_action) for row in cell_rows) == {
                    ("natural", False): 6,
                    ("action_induced", True): 6,
                }
                assert Counter(row.contact_geometry for row in cell_rows) == {
                    "head_on": 6,
                    "glancing": 6,
                }
            else:
                assert Counter(row.known_action for row in cell_rows) == {
                    False: 6,
                    True: 6,
                }
            if cell_rows[0].dynamic_membership:
                assert Counter(row.lifecycle_schedule for row in cell_rows) == {
                    "birth": 4,
                    "removal": 4,
                    "remove_then_birth": 4,
                }
                if cell_rows[0].contact:
                    cross = Counter(
                        (
                            row.contact_origin,
                            row.known_action,
                            row.lifecycle_schedule,
                        )
                        for row in cell_rows
                    )
                    assert cross == {
                        (contact_origin, known_action, lifecycle): 2
                        for contact_origin, known_action in (
                            ("natural", False),
                            ("action_induced", True),
                        )
                        for lifecycle in ("birth", "removal", "remove_then_birth")
                    }
                    assert Counter(
                        (
                            row.contact_origin,
                            row.known_action,
                            row.lifecycle_schedule,
                            row.contact_geometry,
                        )
                        for row in cell_rows
                    ) == {
                        (contact_origin, known_action, lifecycle, geometry): 1
                        for contact_origin, known_action in (
                            ("natural", False),
                            ("action_induced", True),
                        )
                        for lifecycle in ("birth", "removal", "remove_then_birth")
                        for geometry in ("head_on", "glancing")
                    }
                else:
                    assert Counter(
                        (row.known_action, row.lifecycle_schedule) for row in cell_rows
                    ) == {
                        (known_action, lifecycle): 2
                        for known_action in (False, True)
                        for lifecycle in ("birth", "removal", "remove_then_birth")
                    }
