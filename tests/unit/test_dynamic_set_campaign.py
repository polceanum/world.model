from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from world_model.training.convergence import CampaignInspection, ValidationCandidate
from world_model.training.dynamic_set_campaign import (
    DEFAULT_CAMPAIGN,
    TRAINING_CACHE_FULL_COVERAGE_UPDATES,
    TRAINING_CACHE_TIMING_SUPPORT_UPDATES,
    TRAINING_CACHE_WARM_WINDOW_START_UPDATES,
    DisposableScreenMetrics,
    DynamicSetCampaignConfig,
    SecondAttemptAdmissionEvidence,
    choose_second_attempt,
    configure_second_attempt,
    decide_dynamic_set_campaign,
    disposable_screen_failures,
    learning_rate_multiplier,
    project_minimum_update_feasibility,
)
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_objectives import dynamic_set_objective_regret
from world_model.training.qualification_core import canonical_sha256

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"


def _candidate(step: int, score: float, *, accepted: bool) -> ValidationCandidate:
    return ValidationCandidate(
        step=step,
        score=score,
        accepted=accepted,
        training_support_passed=True,
        model_state_hash=f"hash-{step}",
        checkpoint_path=f"validation_step_{step}.pt",
        selection_guardrails_passed=True,
    )


def _plateau_inspection(*, completed: int = 16_384) -> CampaignInspection:
    candidates = (
        _candidate(completed - 2_048, 1.0, accepted=True),
        _candidate(completed - 1_536, 0.999, accepted=False),
        _candidate(completed - 1_024, 0.998, accepted=False),
        _candidate(completed - 512, 0.997, accepted=False),
        _candidate(completed, 0.996, accepted=False),
    )
    return CampaignInspection(
        run_directory="/tmp/dynamic-set-test",
        completed_steps=completed,
        protocol_hash="protocol",
        best_step=completed - 2_048,
        best_score=1.0,
        reference_step=0,
        validation_candidates=candidates,
    )


def test_learning_rate_has_exact_warmup_and_ten_percent_cosine_floor() -> None:
    assert learning_rate_multiplier(0) == 0.0
    assert learning_rate_multiplier(256) == pytest.approx(0.5)
    assert learning_rate_multiplier(512) == 1.0
    assert learning_rate_multiplier(DEFAULT_CAMPAIGN.maximum_updates) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="outside"):
        learning_rate_multiplier(DEFAULT_CAMPAIGN.maximum_updates + 1)


def _tiny_timing_campaign() -> DynamicSetCampaignConfig:
    return DynamicSetCampaignConfig(
        minimum_updates=4,
        extension_updates=2,
        maximum_updates=8,
        validation_interval_updates=2,
        plateau_validation_count=2,
        warmup_updates=2,
        maximum_training_hours=1.0,
        reserved_audit_hours=0.5,
        minimum_timing_support_updates=2,
        timing_projection_window_updates=2,
        timing_projection_confidence_z=1.0,
        timing_projection_safety_factor=1.0,
        screen_maximum_updates=2,
    ).validate()


def _second_attempt_admission(
    *,
    config: DynamicSetCampaignConfig | None = None,
    update_seconds: tuple[float, ...] = (100.0, 100.0),
    validation_seconds: tuple[float, ...] = (25.0,),
    first_screen_seconds: float = 100.0,
    prior_attempt_seconds: float = 100.0,
    second_screen_seconds: float = 50.0,
    discarded_attempt_seconds: float = 0.0,
    architecture_choice: str = "widen_perception",
    validation_boundary_updates: tuple[int, ...] | None = None,
    base_config_sha256: str = "b" * 64,
    resolved_config_sha256: str = "c" * 64,
    protocol_sha256: str = "1" * 64,
) -> SecondAttemptAdmissionEvidence:
    resolved = _tiny_timing_campaign() if config is None else config
    boundaries = (
        tuple(
            resolved.validation_interval_updates * index
            for index in range(1, len(validation_seconds) + 1)
        )
        if validation_boundary_updates is None
        else validation_boundary_updates
    )
    validation_timing_sha256 = "9" * 64 if validation_seconds else "0" * 64
    timing_body = {
        "schema": "dynamic_set_update_timing_v3",
        "architecture_attempt_index": 2,
        "architecture_choice": architecture_choice,
        "base_config_sha256": base_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "prior_attempt_cumulative_seconds": prior_attempt_seconds,
        "architecture_attempt_sha256": "8" * 64,
        "completed_update_seconds": list(update_seconds),
        "discarded_attempt_seconds": discarded_attempt_seconds,
        "screen_wall_seconds": second_screen_seconds,
        "completed_validation_seconds": list(validation_seconds),
        "validation_timing_sha256": validation_timing_sha256,
        "cumulative_training_seconds": float(
            math.fsum(
                (
                    prior_attempt_seconds,
                    second_screen_seconds,
                    *update_seconds,
                    discarded_attempt_seconds,
                    *validation_seconds,
                )
            )
        ),
    }
    execution_timing = {
        **timing_body,
        "evidence_sha256": canonical_sha256(timing_body),
    }
    return SecondAttemptAdmissionEvidence.from_sealed_execution_timing(
        execution_timing=execution_timing,
        validation_boundary_updates=boundaries,
        protocol_sha256=protocol_sha256,
        source_sha256="2" * 64,
        first_attempt_sha256="3" * 64,
        first_screen_result_sha256="4" * 64,
        first_screen_wall_seconds=first_screen_seconds,
        second_screen_result_sha256="5" * 64,
        execution_progress_record_sha256="6" * 64,
        config=resolved,
    )


def test_sixty_hour_envelope_preserves_twelve_hours_before_mutation() -> None:
    assert DEFAULT_CAMPAIGN.maximum_training_hours == 60.0
    assert DEFAULT_CAMPAIGN.training_mutation_hours == 48.0
    assert DEFAULT_CAMPAIGN.reserved_audit_hours == 12.0
    assert DEFAULT_CAMPAIGN.training_mutation_seconds == 48.0 * 3600.0
    assert TRAINING_CACHE_FULL_COVERAGE_UPDATES == 3_000
    assert TRAINING_CACHE_WARM_WINDOW_START_UPDATES == 3_072
    assert TRAINING_CACHE_TIMING_SUPPORT_UPDATES == 3_584
    assert DEFAULT_CAMPAIGN.minimum_timing_support_updates == TRAINING_CACHE_TIMING_SUPPORT_UPDATES
    assert DEFAULT_CAMPAIGN.timing_projection_window_updates == 512
    assert TRAINING_CACHE_WARM_WINDOW_START_UPDATES >= TRAINING_CACHE_FULL_COVERAGE_UPDATES
    assert TRAINING_CACHE_WARM_WINDOW_START_UPDATES % 512 == 0
    assert TRAINING_CACHE_TIMING_SUPPORT_UPDATES % 512 == 0
    assert (
        TRAINING_CACHE_WARM_WINDOW_START_UPDATES + DEFAULT_CAMPAIGN.timing_projection_window_updates
        == TRAINING_CACHE_TIMING_SUPPORT_UPDATES
    )
    with pytest.raises(ValueError, match="wall-time"):
        replace(DEFAULT_CAMPAIGN, reserved_audit_hours=60.0).validate()


def test_projection_waits_for_complete_support_then_uses_conservative_rate() -> None:
    config = _tiny_timing_campaign()
    unsupported = project_minimum_update_feasibility(
        completed_update_seconds=(1_000.0,),
        config=config,
    )
    assert not unsupported.support_satisfied
    assert unsupported.minimum_update_feasible is None
    assert unsupported.projected_minimum_training_seconds is None
    assert unsupported.limit_hit_reason == "none"

    feasible = project_minimum_update_feasibility(
        completed_update_seconds=(100.0, 100.0),
        completed_validation_seconds=(0.0,),
        config=config,
    )
    assert feasible.support_satisfied
    assert feasible.conservative_update_seconds == 100.0
    assert feasible.projected_minimum_training_seconds == 400.0
    assert feasible.projected_minimum_envelope_seconds == 2_200.0
    assert feasible.envelope_limit_seconds == 3_600.0
    assert feasible.minimum_update_feasible is True
    assert feasible.limit_hit_reason == "none"

    infeasible = project_minimum_update_feasibility(
        completed_update_seconds=(500.0, 500.0),
        completed_validation_seconds=(0.0,),
        config=config,
    )
    assert infeasible.projected_minimum_training_seconds == 2_000.0
    assert infeasible.projected_minimum_envelope_seconds == 3_800.0
    assert infeasible.mutation_limit_seconds == 1_800.0
    assert infeasible.minimum_update_feasible is False
    assert infeasible.limit_hit_reason == "minimum_update_projection_infeasible"


def test_production_projection_charges_cold_fill_but_extrapolates_only_warm_window() -> None:
    # Measured one-thread profile: 12.3 seconds to certify/cache 24 cold rows
    # plus 2.3 seconds of ordinary update compute, versus 2.3 seconds warm.
    cold_update_seconds = 14.6
    warm_update_seconds = 2.3
    cold_seconds = (cold_update_seconds,) * TRAINING_CACHE_WARM_WINDOW_START_UPDATES
    warm_seconds = (warm_update_seconds,) * DEFAULT_CAMPAIGN.timing_projection_window_updates

    old_cold_extrapolation = project_minimum_update_feasibility(
        completed_update_seconds=cold_seconds[:512],
        completed_validation_seconds=(100.0,),
        screen_wall_seconds=300.0,
        config=replace(DEFAULT_CAMPAIGN, minimum_timing_support_updates=512).validate(),
    )
    assert old_cold_extrapolation.support_satisfied
    assert old_cold_extrapolation.minimum_update_feasible is False
    assert old_cold_extrapolation.limit_hit_reason == "minimum_update_projection_infeasible"

    unsupported = project_minimum_update_feasibility(
        completed_update_seconds=(*cold_seconds, *warm_seconds[:-1]),
        completed_validation_seconds=(100.0,) * 6,
        screen_wall_seconds=300.0,
    )
    assert unsupported.completed_updates == TRAINING_CACHE_TIMING_SUPPORT_UPDATES - 1
    assert not unsupported.support_satisfied
    assert unsupported.conservative_update_seconds is None
    assert unsupported.projected_minimum_training_seconds is None
    assert unsupported.cumulative_training_seconds == pytest.approx(
        cold_update_seconds * TRAINING_CACHE_WARM_WINDOW_START_UPDATES
        + warm_update_seconds * 511
        + 600.0
        + 300.0
    )

    supported = project_minimum_update_feasibility(
        completed_update_seconds=(*cold_seconds, *warm_seconds),
        completed_validation_seconds=(100.0,) * 6,
        screen_wall_seconds=300.0,
    )
    assert supported.support_satisfied
    assert supported.conservative_update_seconds == pytest.approx(2.53)
    assert supported.conservative_validation_seconds == pytest.approx(110.0)
    assert supported.projected_remaining_validation_seconds == pytest.approx(2_860.0)
    expected_cumulative = (
        cold_update_seconds * TRAINING_CACHE_WARM_WINDOW_START_UPDATES
        + warm_update_seconds * DEFAULT_CAMPAIGN.timing_projection_window_updates
        + 600.0
        + 300.0
    )
    assert supported.cumulative_training_seconds == pytest.approx(expected_cumulative)
    assert supported.projected_minimum_training_seconds == pytest.approx(
        expected_cumulative
        + 2.53 * (DEFAULT_CAMPAIGN.minimum_updates - TRAINING_CACHE_TIMING_SUPPORT_UPDATES)
        + 2_860.0
    )
    assert supported.minimum_update_feasible is True
    assert supported.limit_hit_reason == "none"


def test_production_warm_projection_still_rejects_a_truly_infeasible_campaign() -> None:
    samples = (1.0,) * TRAINING_CACHE_WARM_WINDOW_START_UPDATES + (12.0,) * 512
    projection = project_minimum_update_feasibility(
        completed_update_seconds=samples,
        completed_validation_seconds=(100.0,) * 6,
        screen_wall_seconds=300.0,
    )
    assert projection.cumulative_training_seconds < DEFAULT_CAMPAIGN.training_mutation_seconds
    assert projection.conservative_update_seconds == pytest.approx(13.2)
    assert projection.minimum_update_feasible is False
    assert projection.limit_hit_reason == "minimum_update_projection_infeasible"


def test_discarded_attempt_time_is_sunk_and_enforces_reserve_boundary() -> None:
    projection = project_minimum_update_feasibility(
        completed_update_seconds=(),
        discarded_attempt_seconds=DEFAULT_CAMPAIGN.training_mutation_seconds,
    )
    assert not projection.support_satisfied
    assert projection.limit_hit
    assert projection.limit_hit_reason == "training_reserve_boundary"


def test_second_attempt_projection_charges_prior_attempt_and_active_screen_once() -> None:
    config = _tiny_timing_campaign()
    projection = project_minimum_update_feasibility(
        completed_update_seconds=(100.0, 100.0),
        completed_validation_seconds=(25.0,),
        discarded_attempt_seconds=50.0,
        prior_attempt_cumulative_seconds=700.0,
        screen_wall_seconds=75.0,
        config=config,
    )
    assert projection.prior_attempt_cumulative_seconds == 700.0
    assert projection.screen_wall_seconds == 75.0
    assert projection.cumulative_training_seconds == 1_050.0
    assert projection.projected_minimum_training_seconds == 1_275.0

    exhausted = project_minimum_update_feasibility(
        completed_update_seconds=(),
        prior_attempt_cumulative_seconds=config.training_mutation_seconds,
        screen_wall_seconds=0.0,
        config=config,
    )
    assert exhausted.limit_hit_reason == "training_reserve_boundary"


def test_disposable_screen_requires_all_declared_signals() -> None:
    passing = DisposableScreenMetrics(
        example_count=64,
        initial_optimization_objective=-0.446,
        final_optimization_objective=-1.40,
        initial_objective_regret=dynamic_set_objective_regret(-0.446),
        final_objective_regret=dynamic_set_objective_regret(-1.40),
        proposal_f1=0.95,
        collision_f1=0.95,
        finite_owner_gradients=True,
        rejected_update_count=0,
        completed_updates=512,
    )
    assert disposable_screen_failures(passing) == ()
    failing = DisposableScreenMetrics(
        example_count=63,
        initial_optimization_objective=10.0,
        final_optimization_objective=2.01,
        initial_objective_regret=dynamic_set_objective_regret(10.0),
        final_objective_regret=dynamic_set_objective_regret(2.01),
        proposal_f1=0.949,
        collision_f1=0.949,
        finite_owner_gradients=False,
        rejected_update_count=1,
        completed_updates=513,
    )
    failures = disposable_screen_failures(failing)
    assert "example_count:!=64" in failures
    assert "objective_reduction:<80%" in failures
    assert "proposal_f1:<0.95" in failures
    assert "collision_f1:<0.95" in failures
    assert "finite_owner_gradients:false" in failures
    assert "rejected_update_count:nonzero" in failures
    assert "completed_updates:outside_screen_budget" in failures

    unbound = DisposableScreenMetrics(
        **{
            **passing.__dict__,
            "final_objective_regret": passing.final_objective_regret + 0.01,
        }
    )
    assert "objective_regret:binding_mismatch" in disposable_screen_failures(unbound)


def test_second_attempt_requires_localization_and_supported_feasibility_evidence() -> None:
    config = _tiny_timing_campaign()
    perception_admission = _second_attempt_admission(
        config=config,
        architecture_choice="widen_perception",
    )
    assert (
        choose_second_attempt(
            first_attempt_failed_early=True,
            oracle_state_dynamics_passed=True,
            perception_gates_passed=False,
            truth_state_contact_owns_error=False,
            admission_evidence=perception_admission,
            admission_evidence_sha256=perception_admission.evidence_sha256,
            config=config,
        )
        == "widen_perception"
    )
    relation_admission = _second_attempt_admission(
        config=config,
        architecture_choice="widen_relation",
    )
    assert (
        choose_second_attempt(
            first_attempt_failed_early=True,
            oracle_state_dynamics_passed=True,
            perception_gates_passed=True,
            truth_state_contact_owns_error=True,
            admission_evidence=relation_admission,
            admission_evidence_sha256=relation_admission.evidence_sha256,
            config=config,
        )
        == "widen_relation"
    )
    assert (
        choose_second_attempt(
            first_attempt_failed_early=False,
            oracle_state_dynamics_passed=True,
            perception_gates_passed=False,
            truth_state_contact_owns_error=False,
            admission_evidence=perception_admission,
            admission_evidence_sha256=perception_admission.evidence_sha256,
            config=config,
        )
        == "none"
    )


def test_positive_remaining_time_alone_cannot_authorize_second_attempt() -> None:
    assert (
        choose_second_attempt(
            first_attempt_failed_early=True,
            oracle_state_dynamics_passed=True,
            perception_gates_passed=False,
            truth_state_contact_owns_error=False,
        )
        == "none"
    )
    admission = _second_attempt_admission()
    assert (
        choose_second_attempt(
            first_attempt_failed_early=True,
            oracle_state_dynamics_passed=True,
            perception_gates_passed=False,
            truth_state_contact_owns_error=False,
            admission_evidence=admission,
            config=_tiny_timing_campaign(),
        )
        == "none"
    )
    with pytest.raises(ValueError, match="campaign binding"):
        choose_second_attempt(
            first_attempt_failed_early=True,
            oracle_state_dynamics_passed=True,
            perception_gates_passed=False,
            truth_state_contact_owns_error=False,
            admission_evidence=admission,
            admission_evidence_sha256="a" * 64,
            config=_tiny_timing_campaign(),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        choose_second_attempt(
            first_attempt_failed_early=True,
            remaining_training_hours=48.0,  # type: ignore[call-arg]
            oracle_state_dynamics_passed=True,
            perception_gates_passed=False,
            truth_state_contact_owns_error=False,
        )


def test_second_attempt_admission_reconstructs_screens_minimum_validations_and_reserve() -> None:
    config = _tiny_timing_campaign()
    evidence = _second_attempt_admission(config=config)

    assert evidence.minimum_updates == 4
    assert evidence.validation_interval_updates == 2
    assert evidence.required_validation_count == 2
    assert evidence.reserved_audit_seconds == 1_800.0
    assert evidence.projection.prior_attempt_cumulative_seconds == 100.0
    assert evidence.projection.screen_wall_seconds == 50.0
    assert evidence.projection.cumulative_training_seconds == 375.0
    assert evidence.projection.projected_remaining_validation_seconds == 25.0
    assert evidence.projection.projected_minimum_training_seconds == 600.0
    assert evidence.projection.projected_minimum_envelope_seconds == 2_400.0
    assert evidence.minimum_campaign_feasible(
        expected_evidence_sha256=evidence.evidence_sha256,
        config=config,
    )


def test_second_attempt_admission_is_none_when_timing_support_is_unavailable_or_infeasible() -> (
    None
):
    config = _tiny_timing_campaign()
    insufficient = _second_attempt_admission(
        config=config,
        update_seconds=(100.0,),
        validation_seconds=(),
    )
    infeasible = _second_attempt_admission(
        config=config,
        update_seconds=(500.0, 500.0),
        validation_seconds=(100.0,),
    )

    for evidence in (insufficient, infeasible):
        assert (
            choose_second_attempt(
                first_attempt_failed_early=True,
                oracle_state_dynamics_passed=True,
                perception_gates_passed=False,
                truth_state_contact_owns_error=False,
                admission_evidence=evidence,
                admission_evidence_sha256=evidence.evidence_sha256,
                config=config,
            )
            == "none"
        )
    assert insufficient.projection.minimum_update_feasible is None
    assert infeasible.projection.minimum_update_feasible is False


def test_second_attempt_admission_rejects_unsealed_omitted_and_tampered_costs() -> None:
    config = _tiny_timing_campaign()
    with pytest.raises(ValueError, match="zero digest"):
        _second_attempt_admission(config=config, protocol_sha256="0" * 64)
    with pytest.raises(ValueError, match="both architecture screens"):
        _second_attempt_admission(config=config, second_screen_seconds=0.0)
    with pytest.raises(ValueError, match="omits its first screen"):
        _second_attempt_admission(
            config=config,
            first_screen_seconds=101.0,
            prior_attempt_seconds=100.0,
        )

    valid = _second_attempt_admission(config=config)
    with pytest.raises(ValueError, match="projection differs|digest mismatch"):
        replace(valid, second_screen_wall_seconds=49.0).validate(
            expected_evidence_sha256=valid.evidence_sha256,
            config=config,
        )
    with pytest.raises(ValueError, match="campaign binding"):
        replace(valid, evidence_sha256="a" * 64).validate(
            expected_evidence_sha256=valid.evidence_sha256,
            config=config,
        )
    with pytest.raises(ValueError, match="validation timing lineage"):
        replace(valid, validation_boundary_updates=(1,)).validate(
            expected_evidence_sha256=valid.evidence_sha256,
            config=config,
        )


@pytest.mark.parametrize(
    ("completed_updates", "validation_count", "expected"),
    [
        (3_583, 6, False),
        (3_584, 6, False),
        (3_585, 7, False),
        (3_584, 7, True),
    ],
)
def test_second_attempt_requires_supported_complete_production_validation_boundary(
    completed_updates: int,
    validation_count: int,
    expected: bool,
) -> None:
    evidence = _second_attempt_admission(
        config=DEFAULT_CAMPAIGN,
        update_seconds=(1.0,) * completed_updates,
        validation_seconds=(1.0,) * validation_count,
        first_screen_seconds=1.0,
        prior_attempt_seconds=1.0,
        second_screen_seconds=1.0,
    )

    assert evidence.projection.support_satisfied is (completed_updates >= 3_584)
    assert (
        evidence.minimum_campaign_feasible(
            expected_evidence_sha256=evidence.evidence_sha256,
        )
        is expected
    )


def test_second_attempt_projection_accepts_exact_envelope_and_rejects_excess() -> None:
    config = _tiny_timing_campaign()
    exact = _second_attempt_admission(
        config=config,
        discarded_attempt_seconds=1_200.0,
    )
    excess = _second_attempt_admission(
        config=config,
        discarded_attempt_seconds=1_200.000_001,
    )

    assert exact.projection.projected_minimum_training_seconds == 1_800.0
    assert exact.projection.projected_minimum_envelope_seconds == 3_600.0
    assert exact.reserved_audit_seconds == 1_800.0
    assert exact.minimum_campaign_feasible(
        expected_evidence_sha256=exact.evidence_sha256,
        config=config,
    )
    assert excess.projection.projected_minimum_training_seconds > 1_800.0
    assert not excess.minimum_campaign_feasible(
        expected_evidence_sha256=excess.evidence_sha256,
        config=config,
    )


@pytest.mark.parametrize(
    "timing_override",
    [
        {"prior_attempt_seconds": 1_300.000_001},
        {"second_screen_seconds": 1_250.000_001},
    ],
)
def test_second_attempt_prior_and_second_screen_costs_can_each_exhaust_budget(
    timing_override: dict[str, float],
) -> None:
    config = _tiny_timing_campaign()
    evidence = _second_attempt_admission(config=config, **timing_override)

    assert evidence.projection.minimum_update_feasible is False
    assert not evidence.minimum_campaign_feasible(
        expected_evidence_sha256=evidence.evidence_sha256,
        config=config,
    )


@pytest.mark.parametrize(
    ("oracle_passed", "perception_passed", "contact_owns", "expected"),
    [
        (True, False, False, "widen_perception"),
        (True, True, True, "widen_relation"),
    ],
)
def test_second_attempt_configuration_changes_only_the_evidence_owned_width(
    oracle_passed: bool,
    perception_passed: bool,
    contact_owns: bool,
    expected: str,
) -> None:
    base = load_config(CONFIG_PATH)
    campaign_config = _tiny_timing_campaign()
    if expected == "widen_perception":
        expected_model = replace(
            base.model,
            rgbd=replace(base.model.rgbd, set_feature_dim=64),
        )
    else:
        expected_model = replace(
            base.model,
            dynamics=replace(base.model.dynamics, relation_hidden_dim=64),
        )
    expected_config = replace(base, model=expected_model)
    admission = _second_attempt_admission(
        config=campaign_config,
        architecture_choice=expected,
        base_config_sha256=canonical_sha256(base.to_dict()),
        resolved_config_sha256=canonical_sha256(expected_config.to_dict()),
    )
    choice, configured = configure_second_attempt(
        base,
        first_attempt_failed_early=True,
        oracle_state_dynamics_passed=oracle_passed,
        perception_gates_passed=perception_passed,
        truth_state_contact_owns_error=contact_owns,
        admission_evidence=admission,
        admission_evidence_sha256=admission.evidence_sha256,
        campaign_config=campaign_config,
    )
    assert choice == expected
    assert configured is not None
    if expected == "widen_perception":
        assert configured.model.rgbd.set_feature_dim == 64
        assert configured.model.dynamics.relation_hidden_dim is None
    else:
        assert configured.model.rgbd.set_feature_dim == 32
        assert configured.model.dynamics.relation_hidden_dim == 64


def test_second_attempt_configuration_returns_no_config_without_authority() -> None:
    base = load_config(CONFIG_PATH)
    choice, configured = configure_second_attempt(
        base,
        first_attempt_failed_early=False,
        oracle_state_dynamics_passed=True,
        perception_gates_passed=False,
        truth_state_contact_owns_error=False,
    )
    assert choice == "none"
    assert configured is None


def test_post_minimum_gate_pass_qualifies_without_waiting_for_plateau() -> None:
    qualified = decide_dynamic_set_campaign(
        _plateau_inspection(),
        absolute_and_promotion_gates_passed=True,
    )
    assert qualified.status == "qualified_convergence"
    assert qualified.next_total_updates is None


def test_authenticated_time_limit_precedes_otherwise_qualified_incumbent() -> None:
    decision = decide_dynamic_set_campaign(
        _plateau_inspection(),
        absolute_and_promotion_gates_passed=True,
        elapsed_training_hours=48.0,
        execution_limit_hit_reason="training_reserve_boundary",
    )

    assert decision.status == "limit_hit"
    assert decision.next_total_updates is None


def test_plateau_without_absolute_gates_is_objective_plateau() -> None:
    unqualified = decide_dynamic_set_campaign(
        _plateau_inspection(),
        absolute_and_promotion_gates_passed=False,
    )
    assert unqualified.status == "objective_plateau"


def test_gate_pass_before_minimum_cannot_be_reported_as_convergence() -> None:
    inspection = _plateau_inspection(completed=8_192)
    decision = decide_dynamic_set_campaign(
        inspection,
        absolute_and_promotion_gates_passed=True,
    )
    assert decision.status == "continue"
    assert decision.next_total_updates == DEFAULT_CAMPAIGN.minimum_updates


@pytest.mark.parametrize(
    ("absolute_gates_passed", "failed_to_improve"),
    [
        (True, False),
        (False, False),
        (False, True),
    ],
)
def test_mid_extension_validation_cannot_terminate_campaign(
    absolute_gates_passed: bool,
    failed_to_improve: bool,
) -> None:
    decision = decide_dynamic_set_campaign(
        _plateau_inspection(completed=16_896),
        absolute_and_promotion_gates_passed=absolute_gates_passed,
        failed_to_improve=failed_to_improve,
    )

    assert decision.status == "continue"
    assert decision.next_total_updates == 20_480
    assert "incomplete 4,096-update extension" in decision.reason


@pytest.mark.parametrize("completed", [16_384, 20_480, 24_576, 28_672, 32_768])
def test_only_complete_campaign_and_extension_boundaries_can_qualify(completed: int) -> None:
    decision = decide_dynamic_set_campaign(
        _plateau_inspection(completed=completed),
        absolute_and_promotion_gates_passed=True,
    )

    assert decision.status == "qualified_convergence"


def test_budget_and_improvement_stops_cannot_be_misreported_as_convergence() -> None:
    inspection = _plateau_inspection()
    budget = decide_dynamic_set_campaign(
        inspection,
        absolute_and_promotion_gates_passed=False,
        elapsed_training_hours=48.0,
    )
    assert budget.status == "limit_hit"
    failed = decide_dynamic_set_campaign(
        inspection,
        absolute_and_promotion_gates_passed=False,
        failed_to_improve=True,
    )
    assert failed.status == "failed_to_improve"


def test_improvement_failure_without_plateau_continues_a_complete_extension() -> None:
    inspection = _plateau_inspection()
    improved_candidate = _candidate(16_384, 0.80, accepted=True)
    inspection = replace(
        inspection,
        best_step=improved_candidate.step,
        best_score=improved_candidate.score,
        validation_candidates=(*inspection.validation_candidates[:-1], improved_candidate),
    )

    decision = decide_dynamic_set_campaign(
        inspection,
        absolute_and_promotion_gates_passed=False,
        failed_to_improve=True,
    )

    assert decision.status == "continue"
    assert decision.next_total_updates == 20_480


def test_plateau_with_improvement_failure_has_deterministic_terminal_classification() -> None:
    decision = decide_dynamic_set_campaign(
        _plateau_inspection(completed=20_480),
        absolute_and_promotion_gates_passed=False,
        failed_to_improve=True,
    )

    assert decision.status == "failed_to_improve"
    assert decision.next_total_updates is None


def test_hard_update_cap_remains_a_limit_when_improvement_still_fails() -> None:
    decision = decide_dynamic_set_campaign(
        _plateau_inspection(completed=32_768),
        absolute_and_promotion_gates_passed=False,
        failed_to_improve=True,
    )

    assert decision.status == "limit_hit"
    assert decision.next_total_updates is None


def test_failed_to_improve_cannot_end_main_campaign_before_minimum_or_override_wall_cap() -> None:
    early = decide_dynamic_set_campaign(
        _plateau_inspection(completed=8_192),
        absolute_and_promotion_gates_passed=False,
        failed_to_improve=True,
    )
    assert early.status == "continue"
    assert early.next_total_updates == DEFAULT_CAMPAIGN.minimum_updates

    capped = decide_dynamic_set_campaign(
        _plateau_inspection(),
        absolute_and_promotion_gates_passed=False,
        failed_to_improve=True,
        elapsed_training_hours=48.0,
    )
    assert capped.status == "limit_hit"


def test_authenticated_projection_reason_is_a_terminal_limit_signal() -> None:
    decision = decide_dynamic_set_campaign(
        _plateau_inspection(completed=512),
        absolute_and_promotion_gates_passed=False,
        elapsed_training_hours=1.0,
        execution_limit_hit_reason="minimum_update_projection_infeasible",
    )
    assert decision.status == "limit_hit"
    assert decision.next_total_updates is None
