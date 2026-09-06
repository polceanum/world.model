from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

import world_model.training.dynamic_set_planning as dynamic_set_planning
from world_model.belief import BeliefFactory
from world_model.dynamics import AnalyticFreeMotionDynamics
from world_model.planning import plan_counterfactual_actions
from world_model.training.dynamic_set_planning import (
    PlanningHistoryEvidence,
    PlanningTaskOutcome,
    bind_public_planning_task,
    certify_private_planning_oracle,
    evaluate_certified_planning_task,
    evaluate_required_planning_invariants,
    materialize_public_planning_template,
    measure_state_only_planning_latency,
    measure_state_only_planning_latency_population,
    ranked_observable_appearance_handle,
    reduce_planning_task_outcomes,
)
from world_model.training.dynamic_set_protocol import PlanningManifestRow


def _row(
    *,
    ordinal: int = 0,
    seed: int = 76_900_000,
    object_count: int = 2,
    candidate_count: int = 8,
    target_rank: int = 0,
    previously_dynamic: bool = False,
    candidate_induced_contact: bool = False,
    goal_direction: int = 0,
) -> PlanningManifestRow:
    return PlanningManifestRow(
        split="development",
        ordinal=ordinal,
        seed=seed,
        object_count=object_count,
        previously_dynamic=previously_dynamic,
        candidate_induced_contact=candidate_induced_contact,
        target_rank=target_rank,
        action_time_stratum=ordinal % 4,
        camera_stratum=ordinal % 8,
        goal_direction=goal_direction,
        candidate_count=candidate_count,  # type: ignore[arg-type]
        minimum_normalized_winner_margin=0.05,
        distribution="in_distribution",
    )


def _belief(object_count: int, *, permutation: tuple[int, ...] | None = None):
    belief = BeliefFactory(max_objects=6, appearance_dim=8).create(
        batch_size=1,
        gravity=(0.0, 0.0, 0.0),
    )
    objects = belief.objects.clone()
    if permutation is None:
        permutation = tuple(range(object_count))
    ids = torch.arange(101, 101 + object_count, dtype=torch.int64)
    positions = torch.stack(
        [
            torch.tensor([float(index) * 1.3 - 2.0, 1.0 + 0.1 * index, 0.2 * index])
            for index in range(object_count)
        ]
    )
    velocities = torch.stack(
        [torch.tensor([0.03 * index, -0.01 * index, 0.02]) for index in range(object_count)]
    )
    appearances = torch.zeros(object_count, 8)
    for index in range(object_count):
        appearances[index, index] = 1.0
        appearances[index, -1] = 0.1 * (index + 1)
    for slot, entity in enumerate(permutation):
        objects.active[0, slot] = True
        objects.object_id[0, slot] = ids[entity]
        objects.position[0, slot] = positions[entity]
        objects.velocity[0, slot] = velocities[entity]
        objects.appearance[0, slot] = appearances[entity]
        objects.age_steps[0, slot] = 24
        objects.existence_logit[0, slot] = 8.0
        objects.visibility_logit[0, slot] = 8.0
        objects.log_drag[0, slot] = math.log(0.05)
    return replace(
        belief,
        objects=objects,
        next_object_id=torch.tensor([101 + object_count], dtype=torch.int64),
        metadata={"simulator_truth_must_not_cross": torch.tensor([123.0])},
    ).validate()


def _history(belief, *, previously_dynamic: bool = False) -> PlanningHistoryEvidence:
    samples = torch.zeros_like(belief.objects.object_id)
    samples.masked_fill_(belief.objects.active, 16)
    return PlanningHistoryEvidence(
        valid_sample_count=samples,
        previously_dynamic=previously_dynamic,
    )


def _handle(row: PlanningManifestRow, belief) -> torch.Tensor:
    """Stand in for a handle frozen by the public task-data materializer."""

    return ranked_observable_appearance_handle(belief, row.target_rank)


def _task(
    row: PlanningManifestRow,
    belief,
    history: PlanningHistoryEvidence | None = None,
    template=None,
):
    if template is None:
        template = materialize_public_planning_template(
            row,
            belief,
            _handle(row, belief),
        )
    return bind_public_planning_task(
        template,
        belief,
        _history(belief, previously_dynamic=row.previously_dynamic) if history is None else history,
    )


def _oracle_from_public_analytic(task):
    plan = plan_counterfactual_actions(
        AnalyticFreeMotionDynamics(),
        task.source_belief,
        task.query_offsets,
        task.candidates,
        task.goal,
        weights=task.cost_weights,
    )
    costs = plan.total_cost[0].detach().to(torch.float64)
    goal_distance = plan.terminal_squared_error[0].sqrt()
    contact = torch.zeros(task.row.candidate_count, dtype=torch.bool)
    return certify_private_planning_oracle(
        task.template,
        candidate_costs=costs,
        candidate_terminal_goal_distance_m=goal_distance,
        candidate_induced_contact=contact,
    )


@pytest.mark.parametrize("object_count", range(1, 7))
@pytest.mark.parametrize("candidate_count", (8, 32))
def test_materializer_covers_all_cardinalities_and_candidate_counts(
    object_count: int,
    candidate_count: int,
) -> None:
    row = _row(
        object_count=object_count,
        candidate_count=candidate_count,
        target_rank=object_count - 1,
        goal_direction=(object_count - 1) % 6,
    )
    belief = _belief(object_count)
    source = belief.clone()

    task = _task(row, belief)

    assert len(task.candidates) == candidate_count
    assert len(task.candidate_descriptors) == candidate_count
    assert task.query_offsets.shape == (1, 6)
    assert task.source_belief.metadata == {}
    assert not task.goal.position_world.requires_grad
    assert all(not action.impulse_world.requires_grad for action in task.candidates)
    assert all(torch.equal(action.object_id, task.target_object_id) for action in task.candidates)
    assert all(
        torch.equal(
            action.timestamp, task.source_belief.timestamp + (row.action_time_stratum + 1) * 0.25
        )
        for action in task.candidates
    )
    impulses = {
        tuple(float(value) for value in action.impulse_world[0]) for action in task.candidates
    }
    assert len(impulses) == candidate_count
    torch.testing.assert_close(belief.objects.position, source.objects.position)
    torch.testing.assert_close(belief.objects.velocity, source.objects.velocity)
    assert torch.equal(belief.objects.object_id, source.objects.object_id)


def test_observable_target_resolution_is_slot_permutation_invariant() -> None:
    row = _row(object_count=4, candidate_count=32, target_rank=2, goal_direction=4)
    first_belief = _belief(4)
    second_belief = _belief(4, permutation=(2, 0, 3, 1))

    fixed_handle = _handle(row, first_belief)
    template = materialize_public_planning_template(row, first_belief, fixed_handle)
    first = bind_public_planning_task(
        template,
        first_belief,
        _history(first_belief),
    )
    second = bind_public_planning_task(
        template,
        second_belief,
        _history(second_belief),
    )

    assert torch.equal(first.target_object_id, second.target_object_id)
    torch.testing.assert_close(first.appearance_handle, second.appearance_handle)
    torch.testing.assert_close(first.goal.position_world, second.goal.position_world)
    for left, right in zip(first.candidates, second.candidates, strict=True):
        assert torch.equal(left.object_id, right.object_id)
        torch.testing.assert_close(left.impulse_world, right.impulse_world)


def test_one_fixed_template_and_oracle_bind_to_different_checkpoint_beliefs() -> None:
    row = _row(object_count=3, candidate_count=32, target_rank=1, goal_direction=4)
    reference = _belief(3)
    handle = _handle(row, reference)
    template = materialize_public_planning_template(row, reference, handle)

    renumbered_reference = _belief(3)
    renumbered_objects = renumbered_reference.objects.clone()
    renumbered_objects.object_id[renumbered_objects.active] += 1_000
    renumbered_reference = replace(
        renumbered_reference,
        objects=renumbered_objects,
        next_object_id=torch.tensor([1_104], dtype=torch.int64),
        metadata={"different_ledger_metadata": torch.tensor([999.0])},
    ).validate()
    same_template = materialize_public_planning_template(
        row,
        renumbered_reference,
        handle,
    )
    assert same_template.template_sha256 == template.template_sha256

    baseline_belief = _belief(3)
    candidate_belief = _belief(3)
    candidate_objects = candidate_belief.objects.clone()
    candidate_objects.object_id[candidate_objects.active] += 2_000
    candidate_objects.position[candidate_objects.active] += torch.tensor([0.15, -0.05, 0.02])
    candidate_objects.velocity[candidate_objects.active] *= 1.5
    candidate_objects.log_mass[candidate_objects.active] = math.log(2.0)
    candidate_objects.log_drag[candidate_objects.active] = math.log(0.2)
    candidate_belief = replace(
        candidate_belief,
        objects=candidate_objects,
        next_object_id=torch.tensor([2_104], dtype=torch.int64),
    ).validate()

    baseline = _task(row, baseline_belief, template=template)
    candidate = _task(row, candidate_belief, template=template)
    assert baseline.template_sha256 == candidate.template_sha256 == template.template_sha256
    assert baseline.binding_sha256 != candidate.binding_sha256
    assert not torch.equal(baseline.target_object_id, candidate.target_object_id)
    torch.testing.assert_close(baseline.query_offsets, candidate.query_offsets, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        baseline.goal.position_world,
        candidate.goal.position_world,
        rtol=0.0,
        atol=0.0,
    )
    for index, (left, right) in enumerate(
        zip(baseline.candidates, candidate.candidates, strict=True)
    ):
        torch.testing.assert_close(left.timestamp, right.timestamp, rtol=0.0, atol=0.0)
        torch.testing.assert_close(left.impulse_world, right.impulse_world, rtol=0.0, atol=0.0)
        assert torch.equal(left.timestamp, template.candidate_timestamps[index])
        assert torch.equal(left.impulse_world, template.candidate_impulses_world[index])

    oracle = _oracle_from_public_analytic(baseline)
    baseline_evaluation = evaluate_certified_planning_task(
        AnalyticFreeMotionDynamics(), baseline, oracle
    )
    candidate_evaluation = evaluate_certified_planning_task(
        AnalyticFreeMotionDynamics(), candidate, oracle
    )
    assert baseline_evaluation.template_sha256 == candidate_evaluation.template_sha256


def test_target_handle_must_be_independently_supplied_and_unambiguous() -> None:
    row = _row(object_count=2)
    belief = _belief(2)
    appearance = belief.objects.appearance[0, :2]
    normalised = appearance / torch.linalg.vector_norm(appearance, dim=-1, keepdim=True)
    ambiguous_handle = (normalised[0] + normalised[1]).unsqueeze(0)

    with pytest.raises(ValueError, match="ambiguous"):
        materialize_public_planning_template(row, belief, ambiguous_handle)

    wrong_rank_handle = _handle(replace(row, target_rank=1), belief)
    with pytest.raises(ValueError, match="manifest target rank"):
        materialize_public_planning_template(row, belief, wrong_rank_handle)

    weak_handle = torch.zeros(1, 8)
    weak_handle[0, 0] = 0.3
    weak_handle[0, 2] = 1.0
    with pytest.raises(ValueError, match="below the cosine gate"):
        materialize_public_planning_template(row, belief, weak_handle)


def test_task_owns_detached_storage_and_rejects_immature_or_stale_history() -> None:
    row = _row(object_count=2, previously_dynamic=True)
    belief = _belief(2)
    evidence = _history(belief, previously_dynamic=True)
    public_handle = _handle(row, belief)
    template = materialize_public_planning_template(row, belief, public_handle)
    task = bind_public_planning_task(template, belief, evidence)
    before = task.source_belief.objects.position.clone()
    fixed_goal = task.template.goal_position_world.clone()

    belief.objects.position.add_(100.0)
    evidence.valid_sample_count.zero_()
    public_handle.zero_()
    template.goal_position_world.zero_()
    torch.testing.assert_close(task.source_belief.objects.position, before)
    assert bool(task.appearance_handle.ne(0).any())
    torch.testing.assert_close(task.template.goal_position_world, fixed_goal)
    assert torch.equal(
        task.history_evidence.valid_sample_count.masked_select(task.frozen_active_mask),
        torch.full((2,), 16, dtype=torch.int64),
    )

    fresh_belief = _belief(2)
    fresh_template = materialize_public_planning_template(
        row,
        fresh_belief,
        _handle(row, fresh_belief),
    )
    immature = _history(fresh_belief, previously_dynamic=True)
    immature.valid_sample_count[0, 0] = 15
    with pytest.raises(ValueError, match="mature history"):
        bind_public_planning_task(fresh_template, fresh_belief, immature)
    with pytest.raises(ValueError, match="provenance"):
        bind_public_planning_task(
            fresh_template,
            fresh_belief,
            _history(fresh_belief, previously_dynamic=False),
        )


def test_private_certificate_requires_margin_and_matching_contact_stratum() -> None:
    row = _row(object_count=2, candidate_induced_contact=True)
    belief = _belief(2)
    task = _task(row, belief)
    costs = torch.arange(1, 9, dtype=torch.float64)
    goal_distance = torch.ones(8, dtype=torch.float64)
    goal_distance[0] = 0.0
    contact = torch.zeros(8, dtype=torch.bool)

    with pytest.raises(ValueError, match="contact"):
        certify_private_planning_oracle(
            task.template,
            candidate_costs=costs,
            candidate_terminal_goal_distance_m=goal_distance,
            candidate_induced_contact=contact,
        )
    with pytest.raises(TypeError, match="boolean evaluator labels"):
        certify_private_planning_oracle(
            task.template,
            candidate_costs=costs,
            candidate_terminal_goal_distance_m=goal_distance,
            candidate_induced_contact=torch.zeros(8),
        )

    contact[0] = True
    oracle = certify_private_planning_oracle(
        task.template,
        candidate_costs=costs,
        candidate_terminal_goal_distance_m=goal_distance,
        candidate_induced_contact=contact,
    )
    assert oracle.certificate.winner_index == 0
    assert oracle.certificate.winner_succeeds
    assert oracle.certificate.normalized_winner_margin >= 0.05

    tied = costs.clone()
    tied[1] = 1.04
    with pytest.raises(ValueError, match="winner margin"):
        certify_private_planning_oracle(
            task.template,
            candidate_costs=tied,
            candidate_terminal_goal_distance_m=goal_distance,
            candidate_induced_contact=contact,
        )


def test_contact_stratum_requires_a_contact_candidate_not_a_contact_winner() -> None:
    row = _row(object_count=2, candidate_induced_contact=True)
    belief = _belief(2)
    task = _task(row, belief)
    costs = torch.arange(1, 9, dtype=torch.float64)
    goal_distance = torch.ones(8, dtype=torch.float64)
    goal_distance[0] = 0.0
    contact = torch.zeros(8, dtype=torch.bool)
    contact[3] = True

    oracle = certify_private_planning_oracle(
        task.template,
        candidate_costs=costs,
        candidate_terminal_goal_distance_m=goal_distance,
        candidate_induced_contact=contact,
    )

    assert oracle.certificate.winner_index == 0
    assert not bool(oracle.candidate_induced_contact[oracle.certificate.winner_index])
    assert bool(oracle.candidate_induced_contact.any())


@pytest.mark.parametrize("candidate_count", (8, 32))
def test_evaluator_matches_private_oracle_and_serial_vectorized_oracle(
    candidate_count: int,
) -> None:
    row = _row(
        ordinal=3,
        object_count=3,
        candidate_count=candidate_count,
        target_rank=1,
        goal_direction=5,
    )
    belief = _belief(3)
    task = _task(row, belief)
    oracle = _oracle_from_public_analytic(task)

    result = evaluate_certified_planning_task(
        AnalyticFreeMotionDynamics(),
        task,
        oracle,
    )

    assert result.oracle_winner_correct
    assert result.normalized_regret == 0.0
    assert result.oracle_winner_succeeds
    assert result.selected_action_goal_success
    assert result.serial_vectorized_winner_parity
    assert result.maximum_cost_difference == 0.0
    assert result.cost_agreement_within_tolerance
    assert result.active_set_frozen
    assert result.source_belief_unchanged


def test_required_planning_invariants_and_cpu_latencies_are_measured() -> None:
    dynamics = AnalyticFreeMotionDynamics()
    belief = _belief(6)
    k8 = _task(
        _row(
            ordinal=1,
            seed=76_900_101,
            object_count=6,
            candidate_count=8,
            target_rank=2,
        ),
        belief,
    )
    peer_belief = _belief(6, permutation=(5, 3, 1, 4, 2, 0))
    peer = _task(
        _row(
            ordinal=2,
            seed=76_900_102,
            object_count=6,
            candidate_count=8,
            target_rank=4,
        ),
        peer_belief,
    )
    k32 = _task(
        _row(
            ordinal=3,
            seed=76_900_103,
            object_count=6,
            candidate_count=32,
            target_rank=5,
        ),
        belief,
    )

    metrics = evaluate_required_planning_invariants(
        dynamics,
        k8,
        peer,
        k8,
        k32,
        latency_warmup_runs=0,
        latency_measured_runs=2,
    )

    assert metrics.serial_vectorized_winner_parity
    assert metrics.maximum_cost_difference == 0.0
    assert metrics.pre_action_invariance
    assert metrics.exactly_once_impulse
    assert metrics.action_target_isolation
    assert metrics.conservation
    assert metrics.batch_independence
    assert metrics.source_belief_unchanged
    assert math.isfinite(metrics.latency_k8_seconds)
    assert metrics.latency_k8_seconds >= 0.0
    assert math.isfinite(metrics.latency_k32_seconds)
    assert metrics.latency_k32_seconds >= 0.0


def test_state_only_latency_rejects_non_n6_and_invalid_sample_counts() -> None:
    dynamics = AnalyticFreeMotionDynamics()
    n5 = _task(_row(object_count=5), _belief(5))
    n6 = _task(_row(object_count=6), _belief(6))

    with pytest.raises(ValueError, match="N=6"):
        measure_state_only_planning_latency(dynamics, n5)
    with pytest.raises(ValueError, match="warmup_runs"):
        measure_state_only_planning_latency(dynamics, n6, warmup_runs=True)
    with pytest.raises(ValueError, match="measured_runs"):
        measure_state_only_planning_latency(dynamics, n6, measured_runs=0)


def test_state_only_latency_disables_event_and_auxiliary_materialization() -> None:
    class RecordingDynamics(AnalyticFreeMotionDynamics):
        def __init__(self) -> None:
            super().__init__()
            self.output_flags: list[tuple[bool, bool]] = []

        def rollout(
            self,
            belief,
            query_times,
            *,
            action=None,
            return_events=True,
            return_auxiliary=True,
        ):
            self.output_flags.append((return_events, return_auxiliary))
            return super().rollout(
                belief,
                query_times,
                action=action,
                return_events=return_events,
                return_auxiliary=return_auxiliary,
            )

    dynamics = RecordingDynamics()
    latency = measure_state_only_planning_latency(
        dynamics,
        _task(_row(object_count=6), _belief(6)),
        warmup_runs=1,
        measured_runs=2,
    )

    assert math.isfinite(latency)
    assert dynamics.output_flags == [(False, False)] * 3


def test_state_only_population_latency_samples_every_n6_task_equally(monkeypatch) -> None:
    first = _task(_row(ordinal=0, object_count=6), _belief(6))
    second = _task(_row(ordinal=1, seed=76_900_001, object_count=6), _belief(6))
    action_times: list[float] = []

    def record_call(_dynamics, _belief, _query_offsets, candidates, _goal, **_kwargs):
        action_times.append(float(candidates[0].timestamp[0]))

    clock = iter((0.0, 0.1, 1.0, 1.9, 2.0, 2.2, 3.0, 3.8))
    monkeypatch.setattr(dynamic_set_planning, "plan_counterfactual_actions", record_call)
    monkeypatch.setattr(dynamic_set_planning.time, "perf_counter", lambda: next(clock))

    latency = measure_state_only_planning_latency_population(
        AnalyticFreeMotionDynamics(),
        (first, second),
        warmup_runs=0,
        measured_runs=2,
    )

    assert (
        action_times
        == [
            float(first.candidates[0].timestamp[0]),
            float(second.candidates[0].timestamp[0]),
        ]
        * 2
    )
    assert latency == pytest.approx(0.5)


def test_private_oracle_is_digest_bound_to_candidate_order() -> None:
    belief = _belief(2)
    first_row = _row(ordinal=0, seed=76_900_001)
    second_row = _row(ordinal=1, seed=76_900_002)
    first = _task(first_row, belief)
    second = _task(second_row, belief)
    oracle = _oracle_from_public_analytic(first)

    with pytest.raises(ValueError, match="different public template"):
        evaluate_certified_planning_task(
            AnalyticFreeMotionDynamics(),
            second,
            oracle,
        )


def test_private_oracle_is_not_inspected_until_both_public_plans_finish() -> None:
    belief = _belief(2)
    row = _row(ordinal=0, seed=76_900_004)
    task = _task(row, belief)

    class CountingDynamics:
        def __init__(self) -> None:
            self.delegate = AnalyticFreeMotionDynamics()
            self.rollout_calls = 0
            self.output_flags: list[tuple[bool, bool]] = []

        def validate_action_rollout(self, belief, query_times, action):
            return self.delegate.validate_action_rollout(belief, query_times, action)

        def rollout(
            self,
            belief,
            query_times,
            *,
            action=None,
            return_events=True,
            return_auxiliary=True,
        ):
            self.rollout_calls += 1
            self.output_flags.append((return_events, return_auxiliary))
            return self.delegate.rollout(
                belief,
                query_times,
                action=action,
                return_events=return_events,
                return_auxiliary=return_auxiliary,
            )

    dynamics = CountingDynamics()
    with pytest.raises(TypeError, match="PrivatePlanningOracleEvidence"):
        evaluate_certified_planning_task(dynamics, task, object())

    # One vectorized rollout plus one serial rollout for each of eight
    # candidates proves both decisions completed before private validation.
    assert dynamics.rollout_calls == 9
    assert dynamics.output_flags == [(False, False)] * 9


def test_public_state_and_private_evidence_are_tamper_evident() -> None:
    belief = _belief(2)
    row = _row(ordinal=2, seed=76_900_006)
    task = _task(row, belief)
    oracle = _oracle_from_public_analytic(task)

    task.source_belief.objects.velocity[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="binding digest"):
        evaluate_certified_planning_task(
            AnalyticFreeMotionDynamics(),
            task,
            oracle,
        )

    fresh = _task(row, belief)
    fresh_oracle = _oracle_from_public_analytic(fresh)
    fresh_oracle.candidate_costs[0] += 0.01
    with pytest.raises(ValueError, match="certificate digest"):
        evaluate_certified_planning_task(
            AnalyticFreeMotionDynamics(),
            fresh,
            fresh_oracle,
        )


def test_goal_and_every_candidate_remain_bound_to_the_resolved_target() -> None:
    belief = _belief(2)
    row = _row(ordinal=2, seed=76_900_007)
    goal_tampered = _task(row, belief)
    goal_oracle = _oracle_from_public_analytic(goal_tampered)
    active_ids = goal_tampered.source_belief.objects.object_id[
        goal_tampered.source_belief.objects.active
    ]
    other_id = active_ids[active_ids != goal_tampered.target_object_id[0]][0].clone()
    goal_tampered.goal.object_id[0] = other_id

    with pytest.raises(ValueError, match="same target"):
        evaluate_certified_planning_task(
            AnalyticFreeMotionDynamics(),
            goal_tampered,
            goal_oracle,
        )

    action_tampered = _task(row, belief)
    action_oracle = _oracle_from_public_analytic(action_tampered)
    action_tampered.candidates[0].object_id[0] = other_id
    with pytest.raises(ValueError, match="resolved planning target"):
        evaluate_certified_planning_task(
            AnalyticFreeMotionDynamics(),
            action_tampered,
            action_oracle,
        )


def test_reducer_counts_resolution_support_and_conditional_goal_success() -> None:
    belief = _belief(2)
    first_row = _row(ordinal=0, seed=76_900_010)
    second_row = _row(ordinal=4, seed=76_900_011)
    task = _task(first_row, belief)
    evaluation = evaluate_certified_planning_task(
        AnalyticFreeMotionDynamics(),
        task,
        _oracle_from_public_analytic(task),
    )
    reduction = reduce_planning_task_outcomes(
        (
            PlanningTaskOutcome.evaluated(evaluation),
            PlanningTaskOutcome.unresolved(second_row, reason="ambiguous appearance"),
        )
    )

    assert len(reduction.slices) == 1
    metrics = reduction.slices[0]
    assert metrics.handle_resolution.value == 0.5
    assert metrics.handle_resolution.support == 2
    assert metrics.oracle_winner_accuracy.value == 0.5
    assert metrics.oracle_winner_accuracy.support == 2
    assert metrics.normalized_regret_median.value == 0.5
    assert metrics.normalized_regret_p95.value == 1.0
    assert metrics.successful_oracle_goal_success.value == 0.5
    assert metrics.successful_oracle_goal_success.support == 2
    assert reduction.serial_vectorized_winner_parity
    assert reduction.cost_agreement_within_tolerance
    assert reduction.active_set_frozen
    assert reduction.source_belief_unchanged


def test_reducer_uses_the_conventional_even_population_median() -> None:
    belief = _belief(2)
    first_row = _row(ordinal=30, seed=76_900_030)
    second_row = _row(ordinal=31, seed=76_900_031)
    first_task = _task(first_row, belief)
    first = evaluate_certified_planning_task(
        AnalyticFreeMotionDynamics(),
        first_task,
        _oracle_from_public_analytic(first_task),
    )
    second = replace(first, row=second_row, normalized_regret=1.0)

    reduction = reduce_planning_task_outcomes(
        (PlanningTaskOutcome.evaluated(first), PlanningTaskOutcome.evaluated(second))
    )

    assert reduction.slices[0].normalized_regret_median.value == 0.5


def test_reducer_aggregates_handle_resolution_independently_by_cardinality() -> None:
    dynamics = AnalyticFreeMotionDynamics()
    outcomes: list[PlanningTaskOutcome] = []
    for object_count in range(1, 7):
        resolved_row = _row(
            ordinal=10 * object_count,
            seed=76_901_000 + 2 * object_count,
            object_count=object_count,
            target_rank=object_count - 1,
        )
        unresolved_row = _row(
            ordinal=10 * object_count + 1,
            seed=76_901_001 + 2 * object_count,
            object_count=object_count,
            target_rank=object_count - 1,
        )
        task = _task(resolved_row, _belief(object_count))
        evaluation = evaluate_certified_planning_task(
            dynamics,
            task,
            _oracle_from_public_analytic(task),
        )
        outcomes.extend(
            (
                PlanningTaskOutcome.evaluated(evaluation),
                PlanningTaskOutcome.unresolved(
                    unresolved_row,
                    reason="synthetic ambiguous handle",
                ),
            )
        )

    reduction = reduce_planning_task_outcomes(tuple(outcomes))

    assert [metrics.object_count for metrics in reduction.slices] == list(range(1, 7))
    assert all(metrics.candidate_count == 8 for metrics in reduction.slices)
    assert all(metrics.handle_resolution.value == 0.5 for metrics in reduction.slices)
    assert all(metrics.handle_resolution.support == 2 for metrics in reduction.slices)
    assert all(metrics.oracle_winner_accuracy.support == 2 for metrics in reduction.slices)
    assert all(metrics.oracle_winner_accuracy.value == 0.5 for metrics in reduction.slices)
    assert all(metrics.normalized_regret_median.value == 0.5 for metrics in reduction.slices)
    assert all(metrics.normalized_regret_p95.value == 1.0 for metrics in reduction.slices)
    assert all(
        metrics.successful_oracle_goal_success.value == 0.5
        and metrics.successful_oracle_goal_success.support == 2
        for metrics in reduction.slices
    )
    assert set(reduction.gate_metrics.handle_resolution_by_object_count) == {
        f"N{object_count}" for object_count in range(1, 7)
    }
    assert all(
        metric.value == 0.5 and metric.support == 2
        for metric in reduction.gate_metrics.handle_resolution_by_object_count.values()
    )
    assert set(reduction.gate_metrics.oracle_winner_accuracy_by_candidate_distribution) == {
        "K8/in_distribution"
    }
    assert reduction.gate_metrics.normalized_regret_median_by_candidate_count["K8"].value == 0.5
    assert reduction.gate_metrics.normalized_regret_p95_by_candidate_count["K8"].value == 1.0
    assert reduction.gate_metrics.successful_oracle_goal_success.support == 12


def test_reducer_rejects_duplicate_task_rows() -> None:
    row = _row()
    outcome = PlanningTaskOutcome.unresolved(row, reason="synthetic failure")
    with pytest.raises(ValueError, match="duplicate"):
        reduce_planning_task_outcomes((outcome, outcome))


def test_planning_manifest_runtime_schema_rejects_boolean_integer_aliases() -> None:
    belief = _belief(2)
    row = _row()
    handle = _handle(row, belief)

    with pytest.raises(TypeError, match="previously_dynamic must be boolean"):
        materialize_public_planning_template(replace(row, previously_dynamic=0), belief, handle)
    with pytest.raises(ValueError, match="candidate_count must be a nonnegative integer"):
        materialize_public_planning_template(replace(row, candidate_count=True), belief, handle)
    with pytest.raises(TypeError, match="oracle margin"):
        materialize_public_planning_template(
            replace(row, minimum_normalized_winner_margin=True), belief, handle
        )
