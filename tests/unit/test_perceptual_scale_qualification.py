from __future__ import annotations

import json
import math

from world_model.evaluation.perceptual_scale_qualification import (
    PerceptualScalePlanningResult,
    PerceptualScaleScenarioResult,
    _gate_failures,
    _json_safe,
    default_perceptual_scale_scenarios,
    perceptual_scale_manifest_sha256,
)


def _passing_scenario(count: int) -> PerceptualScaleScenarioResult:
    return PerceptualScaleScenarioResult(
        name=f"n{count}-separated_motion",
        object_count=count,
        family="separated_motion",
        proposal_precision=1.0,
        proposal_recall=1.0,
        proposal_f1=1.0,
        exact_count_accuracy=1.0,
        current_position_rmse_m=0.001,
        two_second_position_rmse_m=0.01,
        persistent_id_accuracy=1.0,
        lifecycle_f1=1.0,
        collision_f1=None,
        collision_timing_error_frames=None,
        occlusion_recovered=None,
        known_action_count=0,
        target_handle_resolved=True,
        finite=True,
        latency_seconds=0.1,
        truth_count_trace=(count,),
        proposal_count_trace=(count,),
        belief_count_trace=(count,),
        animation=None,
    )


def _passing_plan(count: int, candidates: int) -> PerceptualScalePlanningResult:
    return PerceptualScalePlanningResult(
        object_count=count,
        candidate_count=candidates,
        winner_correct=True,
        normalized_regret=0.0,
        goal_success=True,
        serial_vectorized_parity=True,
        maximum_cost_difference=0.0,
        latency_seconds=0.02,
    )


def test_scale_manifest_is_stable_and_covers_each_required_family() -> None:
    first = default_perceptual_scale_scenarios()
    second = default_perceptual_scale_scenarios()

    assert first == second
    assert perceptual_scale_manifest_sha256(first) == perceptual_scale_manifest_sha256(second)
    assert {item.object_count for item in first} == {7, 8}
    assert {item.family for item in first} == {
        "separated_motion",
        "known_action",
        "pair_contact",
        "lifecycle",
        "occlusion_recovery",
        "compositional",
    }
    assert len(first) == 12


def test_scale_gate_accepts_absolute_floors_and_rejects_one_regression() -> None:
    scenarios = (_passing_scenario(7), _passing_scenario(8))
    planning = tuple(_passing_plan(count, candidates) for count in (7, 8) for candidates in (8, 32))
    assert _gate_failures(scenarios, planning) == ()

    failed = PerceptualScaleScenarioResult(
        **{
            **_passing_scenario(8).to_dict(),
            "proposal_f1": 0.949,
            "animation": None,
        }
    )
    assert _gate_failures((failed,), planning) == ("n8-separated_motion:proposal_f1",)


def test_failed_nonfinite_metrics_serialize_as_explicit_null() -> None:
    payload = _json_safe({"error": math.inf, "nested": [1.0, -math.inf]})

    assert payload == {"error": None, "nested": [1.0, None]}
    assert json.dumps(payload, allow_nan=False)
