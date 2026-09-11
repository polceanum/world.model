from __future__ import annotations

from world_model.evaluation.rigid_capability import (
    RigidPlanningResult,
    RigidScenarioResult,
    _gate_failures,
    default_rigid_scenarios,
    rigid_manifest_sha256,
)


def _scenario() -> RigidScenarioResult:
    return RigidScenarioResult(
        name="mixed-sphere-box",
        primitive_accuracy=1.0,
        current_position_rmse_m=0.001,
        box_half_extent_rmse_m=0.01,
        persistent_handle_accuracy=1.0,
        two_second_position_rmse_m=0.02,
        two_second_velocity_rmse_mps=0.03,
        collision_f1=1.0,
        collision_timing_error_frames=0,
        predicted_collision_frames=(9,),
        reference_collision_frames=(9,),
        source_unchanged=True,
        finite=True,
        animation={},
    )


def _plan(candidate_count: int) -> RigidPlanningResult:
    return RigidPlanningResult(
        scenario="mixed-sphere-box",
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


def test_rigid_manifest_is_deterministic_and_mixed() -> None:
    first = default_rigid_scenarios()
    second = default_rigid_scenarios()

    assert first == second
    assert rigid_manifest_sha256(first) == rigid_manifest_sha256(second)
    assert {primitive for item in first for primitive in item.primitives} == {"sphere", "box"}
    assert all(any(abs(angle) > 0.0 for angle in item.angles_degrees) for item in first)


def test_rigid_gate_requires_physics_and_action_invariants() -> None:
    plans = (_plan(8), _plan(32))
    assert _gate_failures((_scenario(),), plans) == ()

    failed = RigidPlanningResult(
        **{
            **_plan(8).to_dict(),
            "action_target_isolated": False,
        }
    )
    assert _gate_failures((_scenario(),), (failed,)) == ("mixed-sphere-box:k8:action_invariant",)
