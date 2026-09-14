from __future__ import annotations

from dataclasses import replace

from world_model.evaluation.multicontact_six_dof import MultiContactPlanningResult
from world_model.evaluation.neural_adaptive_physics import NeuralPhysicsMetrics
from world_model.evaluation.neural_long_horizon import (
    LongHorizonScenarioResult,
    _gate_failures,
)


def _metrics(error: float) -> NeuralPhysicsMetrics:
    return NeuralPhysicsMetrics(
        example_count=32,
        mean_relative_error=error,
        p95_relative_error=error,
        maximum_relative_error=error,
        by_parameter={"mass": error},
        prefix_mean_relative_error={"contact": error},
        permutation_maximum_difference=0.0,
    )


def _scenario(object_count: int) -> LongHorizonScenarioResult:
    incumbent = {"2.00": 0.04, "4.00": 0.06, "8.00": 0.08, "12.00": 0.10}
    candidate = {key: 0.80 * value for key, value in incumbent.items()}
    return LongHorizonScenarioResult(
        object_count=object_count,
        candidate_parameter_error=0.01,
        incumbent_parameter_error=0.02,
        candidate_position_rmse_m=candidate,
        incumbent_position_rmse_m=incumbent,
        oracle_parameter_position_rmse_m=candidate,
        candidate_velocity_rmse_mps=candidate,
        incumbent_velocity_rmse_mps=incumbent,
        oracle_parameter_velocity_rmse_mps=candidate,
        candidate_orientation_rmse_degrees=candidate,
        incumbent_orientation_rmse_degrees=incumbent,
        oracle_parameter_orientation_rmse_degrees=candidate,
        candidate_contact_pair_f1=1.0,
        incumbent_contact_pair_f1=1.0,
        oracle_parameter_contact_pair_f1=1.0,
        candidate_rollout_seconds=0.01,
        incumbent_rollout_seconds=0.01,
        oracle_parameter_rollout_seconds=0.01,
        source_unchanged=True,
        finite=True,
        per_object_maximum_errors={},
        animation={},
    )


def _planning(candidate_count: int) -> MultiContactPlanningResult:
    return MultiContactPlanningResult(
        object_count=8,
        candidate_count=candidate_count,
        oracle_winner=1,
        selected_winner=1,
        winner_correct=True,
        normalized_regret=0.0,
        normalized_winner_margin=0.1,
        goal_success=True,
        serial_vectorized_parity=True,
        maximum_cost_difference=0.0,
        vectorized_latency_seconds=0.01,
        serial_latency_seconds=0.02,
        vectorization_speedup=2.0,
        source_unchanged=True,
    )


def test_long_horizon_gate_is_per_cardinality_and_rejects_one_tail_regression() -> None:
    curve = (
        {"long_horizon_stability_loss": 0.10},
        {"long_horizon_stability_loss": 0.01},
    )
    scenarios = tuple(_scenario(count) for count in (4, 6, 8))
    planning = (_planning(8), _planning(32))

    passing = _gate_failures(
        _metrics(0.08),
        _metrics(0.10),
        _metrics(0.07),
        _metrics(0.10),
        curve,
        scenarios,
        planning,
        learned_bytes=400,
        changed_parameters=100,
        learned_parameters=100,
    )
    assert passing == ()

    regressed_n6 = replace(
        scenarios[1],
        candidate_position_rmse_m={
            **scenarios[1].candidate_position_rmse_m,
            "12.00": 0.19,
        },
    )
    failures = _gate_failures(
        _metrics(0.08),
        _metrics(0.10),
        _metrics(0.07),
        _metrics(0.10),
        curve,
        (scenarios[0], regressed_n6, scenarios[2]),
        planning,
        learned_bytes=400,
        changed_parameters=100,
        learned_parameters=100,
    )

    assert "n6:h12_paired_position" in failures
    assert not any(failure.startswith("n4:") for failure in failures)
    assert not any(failure.startswith("n8:") for failure in failures)
