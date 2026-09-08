from __future__ import annotations

from pathlib import Path

import pytest

import world_model.training.capability_workbench as workbench
from world_model.training.capability_workbench import (
    CapabilityWorkbenchConfig,
    capability_change_status,
    materialize_planning_development_workload,
    select_development_incumbent,
    select_physical_development_rows,
    select_planning_development_rows,
    supported_capability_score,
    workbench_plan,
)
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_planning_materializer import PlanningTaskMaterializationError

_PROFILE = Path(__file__).parents[2] / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"


def test_smoke_plan_covers_every_physical_and_planning_slice() -> None:
    config = CapabilityWorkbenchConfig.for_profile("smoke", seed=7)
    plan = workbench_plan(config)

    assert plan == {
        "profile": "smoke",
        "seed": 7,
        "train_updates": 1,
        "training_examples": 24,
        "physical_examples": 22,
        "physical_cells": 22,
        "physical_cycle_offset": 0,
        "planning_tasks": 12,
        "planning_slices": 12,
        "planning_invariants": False,
        "velocity_variance_floor": 4.0e-13,
        "threads": 1,
    }


def test_development_rows_are_balanced_without_protected_data() -> None:
    physical = select_physical_development_rows(3, cycle_offset=1)
    planning = select_planning_development_rows(2)

    assert len(physical) == 66
    assert {row.split for row in physical} == {"development"}
    assert {row.cell_index for row in physical} == set(range(22))
    assert min(row.ordinal for row in physical) == 22
    assert len(planning) == 24
    assert {row.split for row in planning} == {"development"}
    assert {(row.object_count, row.candidate_count) for row in planning} == {
        (count, candidates) for count in range(1, 7) for candidates in (8, 32)
    }


def test_candidate_velocity_calibration_does_not_add_parameters() -> None:
    config = load_config(_PROFILE)
    reference = workbench._new_model(config, 17)
    candidate = workbench._new_model(config, 17, velocity_variance_floor=4.0e-13)

    assert candidate.observation_modules["rgbd"].config.temporal_velocity_variance_floor == 4.0e-13
    assert reference.observation_modules["rgbd"].config.temporal_velocity_variance_floor == 1.0e-14
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())


def test_previous_workbench_checkpoint_defaults_remain_compatible() -> None:
    config = load_config(_PROFILE).to_dict()
    historical = {**config, "model": {**config["model"]}}
    historical["model"]["rgbd"] = {**historical["model"]["rgbd"]}
    historical["model"]["dynamics"] = {**historical["model"]["dynamics"]}
    historical["model"]["rgbd"].pop("birth_proposals")
    historical["model"]["dynamics"].pop("packed_interactions_enabled")

    assert workbench._checkpoint_config_semantics(historical) == (
        workbench._checkpoint_config_semantics(config)
    )


def test_planning_preflight_replaces_a_failed_row_in_the_same_slice(monkeypatch) -> None:
    class _Materialization:
        def __init__(self, row):
            self.row = row

    def materialize(row):
        if row.ordinal == 10:
            raise PlanningTaskMaterializationError("invalid deterministic scene")
        return _Materialization(row)

    monkeypatch.setattr(workbench, "materialize_planning_task", materialize)
    materializations, rejected = materialize_planning_development_workload(1)

    assert len(materializations) == 12
    assert all(item.row.ordinal != 10 for item in materializations)
    assert [item["ordinal"] for item in rejected] == [10]


def test_supported_score_reports_coverage_instead_of_fabricating_missing_metrics() -> None:
    score, weight = supported_capability_score(
        {"current_position": 0.2, "proposal_error": 0.4},
        planning_error=0.1,
    )

    assert weight == pytest.approx(0.40)
    assert score == pytest.approx((0.15 * 0.2 + 0.10 * 0.4 + 0.15 * 0.1) / 0.40)


def test_capability_status_ignores_numerically_tiny_changes() -> None:
    assert capability_change_status(0.0021676369, 0.0021676586) == "unchanged"
    assert capability_change_status(0.19, 0.20) == "improved"
    assert capability_change_status(0.21, 0.20) == "regressed"


def test_development_selection_retains_only_material_gate_clean_learning() -> None:
    selected = select_development_incumbent(0.90, 1.0)
    too_small = select_development_incumbent(0.98, 1.0)
    planning_failed = select_development_incumbent(
        0.90,
        1.0,
        planning_failures=("batch_independence:false",),
    )

    assert selected["selected"] == "trained_candidate"
    assert selected["learned_weights_retained"]
    assert too_small["selected"] == "calibrated_structured_baseline"
    assert not too_small["learned_weights_retained"]
    assert planning_failed["selected"] == "calibrated_structured_baseline"
    assert "planning" in planning_failed["reasons"][-1]


@pytest.mark.parametrize(
    ("field", "value"),
    (("train_updates", 513), ("physical_cycles", 0), ("planning_repeats", 0)),
)
def test_workbench_rejects_unbounded_or_empty_workloads(field: str, value: int) -> None:
    settings = CapabilityWorkbenchConfig(**{field: value})
    with pytest.raises(ValueError):
        settings.validate()
