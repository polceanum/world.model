from __future__ import annotations

from dataclasses import replace

import pytest

from world_model.training.dynamic_set_gates import (
    HORIZON_POSITION_LIMITS_M,
    MAXIMUM_PROCESS_RSS_BYTES,
    PhysicalCellMetrics,
    PlanningInvariantMetrics,
    PlanningPopulationGateMetrics,
    PromotionIntegrityMetrics,
    ResourceMetrics,
    SupportedScalar,
    physical_cell_gate_failures,
    planning_gate_failures,
    promotion_gate_failures,
)
from world_model.training.dynamic_set_protocol import PhysicalCell


def _supported(value: float, support: int = 100) -> SupportedScalar:
    return SupportedScalar(value=value, support=support)


def _physical_metrics(*, contact: bool, dynamic: bool) -> PhysicalCellMetrics:
    return PhysicalCellMetrics(
        proposal_precision=_supported(0.99),
        proposal_recall=_supported(0.99),
        proposal_f1=_supported(0.99),
        exact_count_accuracy=_supported(0.96),
        current_position_rmse_m=_supported(0.009),
        mature_velocity_rmse_mps=_supported(0.019),
        post_event_velocity_rmse_mps=(_supported(0.049) if contact or dynamic else None),
        horizon_position_rmse_m={
            horizon: _supported(limit * 0.99)
            for horizon, limit in HORIZON_POSITION_LIMITS_M.items()
        },
        collision_f1=_supported(0.96),
        collision_timing_error_frames=_supported(0.9),
        persistent_id_accuracy=_supported(0.995),
        identity_switch_rate=_supported(0.004),
        birth_precision=_supported(0.99),
        birth_recall=_supported(0.99),
        birth_latency_frames=_supported(1.0),
        removal_precision=_supported(0.99),
        removal_recall=_supported(0.99),
        removal_latency_frames=_supported(2.0),
        uncertainty_90_coverage=_supported(0.90),
    )


def _planning_metrics(
    distributions: tuple[str, ...] = ("in_distribution", "compositional_ood"),
) -> PlanningPopulationGateMetrics:
    return PlanningPopulationGateMetrics(
        handle_resolution_by_object_count={
            f"N{object_count}": _supported(0.995) for object_count in range(1, 7)
        },
        oracle_winner_accuracy_by_candidate_distribution={
            f"K{candidate_count}/{distribution}": _supported(0.96)
            for candidate_count in (8, 32)
            for distribution in distributions
        },
        normalized_regret_median_by_candidate_count={
            f"K{candidate_count}": _supported(0.01) for candidate_count in (8, 32)
        },
        normalized_regret_p95_by_candidate_count={
            f"K{candidate_count}": _supported(0.05) for candidate_count in (8, 32)
        },
        successful_oracle_goal_success=_supported(0.96),
    )


def _planning_invariants() -> PlanningInvariantMetrics:
    return PlanningInvariantMetrics(
        serial_vectorized_winner_parity=True,
        maximum_cost_difference=1.0e-7,
        pre_action_invariance=True,
        exactly_once_impulse=True,
        action_target_isolation=True,
        conservation=True,
        batch_independence=True,
        source_belief_unchanged=True,
        latency_k8_seconds=0.09,
        latency_k32_seconds=0.34,
    )


def test_every_applicable_physical_gate_passes_only_with_support() -> None:
    for object_count in range(1, 7):
        for contact in (False,) if object_count == 1 else (False, True):
            for dynamic in (False, True):
                cell = PhysicalCell(object_count, contact, dynamic)
                metrics = _physical_metrics(contact=contact, dynamic=dynamic)
                assert physical_cell_gate_failures(cell, metrics) == ()

    cell = PhysicalCell(6, True, True)
    unsupported = replace(
        _physical_metrics(contact=True, dynamic=True),
        collision_f1=_supported(1.0, support=0),
    )
    assert "collision_f1:unsupported" in physical_cell_gate_failures(cell, unsupported)


def test_physical_gate_catches_threshold_and_schema_failures() -> None:
    cell = PhysicalCell(2, True, False)
    metrics = replace(
        _physical_metrics(contact=True, dynamic=False),
        proposal_f1=_supported(0.97),
        uncertainty_90_coverage=_supported(0.96),
        horizon_position_rmse_m={0.05: _supported(0.001)},
    )
    failures = physical_cell_gate_failures(cell, metrics)
    assert any(failure.startswith("proposal_f1:") for failure in failures)
    assert any(failure.startswith("uncertainty_90_coverage:") for failure in failures)
    assert "horizon_position_rmse_m:unexpected_schema" in failures
    assert any(failure.startswith("horizon_2.00_position_rmse_m:missing") for failure in failures)


def test_post_event_velocity_is_required_exactly_for_lifecycle_or_contact_cells() -> None:
    static = PhysicalCell(2, False, False)
    static_metrics = _physical_metrics(contact=False, dynamic=False)
    assert static_metrics.post_event_velocity_rmse_mps is None
    assert physical_cell_gate_failures(static, static_metrics) == ()
    assert "post_event_velocity_rmse_mps:inapplicable" in physical_cell_gate_failures(
        static,
        replace(
            static_metrics,
            post_event_velocity_rmse_mps=_supported(0.0),
        ),
    )

    for cell in (PhysicalCell(2, True, False), PhysicalCell(2, False, True)):
        metrics = _physical_metrics(
            contact=cell.contact,
            dynamic=cell.dynamic_membership,
        )
        assert "post_event_velocity_rmse_mps:missing" in physical_cell_gate_failures(
            cell,
            replace(metrics, post_event_velocity_rmse_mps=None),
        )
        assert "post_event_velocity_rmse_mps:unsupported" in physical_cell_gate_failures(
            cell,
            replace(
                metrics,
                post_event_velocity_rmse_mps=_supported(0.0, support=0),
            ),
        )


def test_negative_physical_cells_still_reject_event_false_positives() -> None:
    cell = PhysicalCell(2, False, False)
    metrics = replace(
        _physical_metrics(contact=False, dynamic=False),
        collision_f1=_supported(0.0),
        birth_precision=_supported(0.0),
        removal_precision=_supported(0.0),
    )

    failures = physical_cell_gate_failures(cell, metrics)
    assert any(failure.startswith("collision_f1:") for failure in failures)
    assert any(failure.startswith("birth_precision:") for failure in failures)
    assert any(failure.startswith("removal_precision:") for failure in failures)


def test_required_planning_gate_is_cardinality_candidate_and_distribution_complete() -> None:
    metrics = _planning_metrics()
    assert planning_gate_failures(metrics, _planning_invariants()) == ()

    incomplete = replace(
        metrics,
        handle_resolution_by_object_count={
            key: value
            for key, value in metrics.handle_resolution_by_object_count.items()
            if key != "N6"
        },
    )
    assert "handle_resolution:incomplete_cardinalities" in planning_gate_failures(
        incomplete,
        _planning_invariants(),
    )
    failed = replace(
        metrics,
        oracle_winner_accuracy_by_candidate_distribution={
            **metrics.oracle_winner_accuracy_by_candidate_distribution,
            "K8/in_distribution": _supported(0.949),
        },
    )
    failures = planning_gate_failures(failed, _planning_invariants())
    assert any(
        failure.startswith("oracle_winner_accuracy/K8/in_distribution") for failure in failures
    )


def test_planning_gate_can_validate_each_governed_distribution_without_opening_the_other() -> None:
    assert (
        planning_gate_failures(
            _planning_metrics(("in_distribution",)),
            _planning_invariants(),
            expected_distributions=("in_distribution",),
        )
        == ()
    )
    assert (
        planning_gate_failures(
            _planning_metrics(("compositional_ood",)),
            _planning_invariants(),
            expected_distributions=("compositional_ood",),
        )
        == ()
    )
    with pytest.raises(ValueError, match="unique nonempty"):
        planning_gate_failures(
            _planning_metrics(),
            _planning_invariants(),
            expected_distributions=(),
        )


def test_planning_invariants_and_latency_are_promotion_blocking() -> None:
    invariants = replace(
        _planning_invariants(),
        exactly_once_impulse=False,
        maximum_cost_difference=1.1e-6,
        latency_k32_seconds=0.351,
    )
    failures = planning_gate_failures(_planning_metrics(), invariants)
    assert "exactly_once_impulse:false" in failures
    assert any(failure.startswith("maximum_cost_difference:") for failure in failures)
    assert any(failure.startswith("latency_k32_seconds:") for failure in failures)


def test_integrity_improvement_gradient_and_resource_gates_are_jointly_required() -> None:
    integrity = PromotionIntegrityMetrics(
        paired_score_improvement_fraction=0.031,
        paired_bootstrap_lower_bound=0.001,
        maximum_critical_regression_fraction=0.019,
        maximum_accepted_like_regression_fraction=0.019,
        truth_leakage_count=0,
        fabricated_target_count=0,
        nonfinite_state_count=0,
        rejected_optimizer_mutation_count=0,
        minimum_complete_gradient_retention=0.10,
    )
    resources = ResourceMetrics(
        perception_latency_seconds=0.25,
        six_horizon_rollout_seconds=0.025,
        learned_weight_bytes=1_048_576,
        persistent_tensor_bytes=262_144,
        process_rss_bytes=MAXIMUM_PROCESS_RSS_BYTES,
    )
    assert promotion_gate_failures(integrity, resources) == ()
    assert promotion_gate_failures(
        integrity,
        replace(resources, process_rss_bytes=MAXIMUM_PROCESS_RSS_BYTES + 1),
    ) == (f"process_rss_bytes:{MAXIMUM_PROCESS_RSS_BYTES + 1}>{MAXIMUM_PROCESS_RSS_BYTES}",)

    failed_integrity = replace(
        integrity,
        paired_score_improvement_fraction=0.029,
        truth_leakage_count=1,
        minimum_complete_gradient_retention=0.099,
    )
    failed_resources = replace(resources, learned_weight_bytes=1_048_577)
    failures = promotion_gate_failures(failed_integrity, failed_resources)
    assert "paired_score_improvement_fraction:<0.03" in failures
    assert "truth_leakage_count:nonzero" in failures
    assert "minimum_complete_gradient_retention:<0.10" in failures
    assert any(failure.startswith("learned_weight_bytes:") for failure in failures)


def test_gate_values_cannot_pass_with_boolean_or_negative_resource_evidence() -> None:
    with pytest.raises(TypeError, match="real scalar"):
        SupportedScalar(value=True, support=1).validate(name="metric")

    integrity = PromotionIntegrityMetrics(
        paired_score_improvement_fraction=0.031,
        paired_bootstrap_lower_bound=0.001,
        maximum_critical_regression_fraction=0.0,
        maximum_accepted_like_regression_fraction=0.0,
        truth_leakage_count=-1,
        fabricated_target_count=0,
        nonfinite_state_count=0,
        rejected_optimizer_mutation_count=0,
        minimum_complete_gradient_retention=0.10,
    )
    resources = ResourceMetrics(
        perception_latency_seconds=0.0,
        six_horizon_rollout_seconds=0.0,
        learned_weight_bytes=-1,
        persistent_tensor_bytes=0,
        process_rss_bytes=0,
    )

    failures = promotion_gate_failures(integrity, resources)
    assert "truth_leakage_count:invalid_count" in failures
    assert any(failure.startswith("learned_weight_bytes:") for failure in failures)
