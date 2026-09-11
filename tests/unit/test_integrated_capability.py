from __future__ import annotations

from world_model.evaluation.integrated_capability import (
    _gate_failures,
    _lifecycle_metrics,
    default_integrated_actions,
    integrated_manifest_sha256,
)
from world_model.evaluation.rigid_capability import RigidPlanningResult


def _plan(candidate_count: int) -> RigidPlanningResult:
    return RigidPlanningResult(
        scenario="integrated-mixed-rigid",
        candidate_count=candidate_count,
        winner_correct=True,
        normalized_regret=0.0,
        goal_success=True,
        serial_vectorized_parity=True,
        maximum_cost_difference=0.0,
        known_action_count=1,
        pre_action_invariant=True,
        action_target_isolated=True,
        source_unchanged=True,
        latency_seconds=0.02,
    )


def _values() -> dict[str, object]:
    return {
        "lifecycle_f1": 1.0,
        "persistent_id_accuracy": 1.0,
        "occlusion_recovered": True,
        "primitive_accuracy": 1.0,
        "current_position_rmse_m": 0.001,
        "box_half_extent_rmse_m": 0.002,
        "two_second_position_rmse_m": 0.01,
        "four_second_position_rmse_m": 0.02,
        "collision_f1": 1.0,
        "collision_timing_error_frames": 0,
        "known_action_count": 3,
        "expected_action_count": 3,
        "source_unchanged": True,
        "finite": True,
    }


def test_integrated_manifest_is_deterministic_and_causal() -> None:
    assert integrated_manifest_sha256() == integrated_manifest_sha256()
    actions = default_integrated_actions()
    assert len(actions) == 3
    assert [item.offset_seconds for item in actions] == sorted(
        item.offset_seconds for item in actions
    )
    assert len({item.target_slot for item in actions}) == 3


def test_integrated_public_lifecycle_recovers_identity_through_occlusion() -> None:
    lifecycle_f1, identity_accuracy, recovered = _lifecycle_metrics()

    assert lifecycle_f1 == 1.0
    assert identity_accuracy == 1.0
    assert recovered


def test_integrated_gate_requires_contact_timing_and_both_planning_sizes() -> None:
    plans = (_plan(8), _plan(32))
    assert _gate_failures(_values(), plans) == ()
    assert _gate_failures(_values(), (_plan(8),)) == ("planning_k32",)

    values = {**_values(), "collision_timing_error_frames": 2}
    assert _gate_failures(values, plans) == ("collision_timing_error_frames",)

    failed_plan = RigidPlanningResult(
        **{**_plan(32).to_dict(), "winner_correct": False, "goal_success": False}
    )
    assert _gate_failures(_values(), (_plan(8), failed_plan)) == ("planning_k32",)
