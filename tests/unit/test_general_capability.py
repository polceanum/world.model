from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.dynamics import DynamicsModel
from world_model.dynamics.graph import InteractionGraph
from world_model.evaluation.general_capability import (
    ALL_CAPABILITY_FACTORS,
    CAPABILITY_FACTORS,
    PLANNING_TRAINING_LOSS_FIELDS,
    CapabilityConvergenceSchedule,
    StreamingMetricReducer,
    assert_split_disjoint,
    attribute_failure,
    capability_manifest,
    manifest_sha256,
    promotion_decision,
    stream_materializations,
    training_admission,
)
from world_model.evaluation.scalability import (
    compare_dense_and_packed_scalability,
    run_state_only_scalability_probes,
)


def _objects(count: int = 6):
    belief = BeliefFactory(
        max_objects=count,
        residual_dynamics_dim=1,
        global_code_dim=1,
    ).create(gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[:, : max(1, count // 2)] = True
    objects.object_id[0, : max(1, count // 2)] = torch.arange(max(1, count // 2))
    objects.position[0, :, 0] = torch.linspace(-2.0, 2.0, count)
    objects.position[0, :, 1] = 1.0
    objects.geometry[..., 0] = 0.21
    objects.fast_log_variance.fill_(-12.0)
    return belief, objects


def test_general_capability_manifests_are_deterministic_balanced_and_disjoint() -> None:
    development = capability_manifest("development")
    regenerated = capability_manifest("development")
    holdout = capability_manifest("compositional_holdout")

    assert development == regenerated
    assert manifest_sha256(development) == manifest_sha256(regenerated)
    assert_split_disjoint(development, holdout)
    physical = [row for row in development if row.kind == "physical"]
    planning = [row for row in development if row.kind == "planning"]
    assert len(physical) == len(CAPABILITY_FACTORS) * 22
    assert len(planning) == len(CAPABILITY_FACTORS) * 12
    for factor in CAPABILITY_FACTORS:
        factor_physical = [row for row in physical if row.factor == factor]
        factor_planning = [row for row in planning if row.factor == factor]
        assert len(factor_physical) == 22
        assert {row.object_count for row in factor_physical} == set(range(1, 7))
        assert {(row.object_count, row.candidate_count) for row in factor_planning} == {
            (count, candidates) for count in range(1, 7) for candidates in (8, 32)
        }
    camera_rows = [row for row in development if row.factor == "camera_motion"]
    assert {row.controls.camera_motion for row in camera_rows} == {
        "static",
        "orbital",
        "translating",
    }
    action_rows = [row for row in development if row.factor == "known_actions"]
    assert {row.controls.known_action_enabled for row in action_rows} == {False, True}
    assert len({row.controls.impulse_direction_world for row in action_rows}) > 6
    assert all(row.controls.impulse_target_rank < row.object_count for row in action_rows)
    assert all(row.factor == "compositional_holdout" for row in holdout)


def test_manifest_runtime_contract_contains_no_generator_truth() -> None:
    forbidden = {"truth", "radius", "mass", "drag", "restitution", "friction", "factor"}
    for row in capability_manifest("development"):
        runtime_names = {name for values in row.runtime_contract().values() for name in values}
        assert runtime_names.isdisjoint(forbidden)
        assert "controls" in row.generator_payload()


def test_streaming_reduction_matches_chunked_reduction_and_is_lazy() -> None:
    visited: list[int] = []

    def materialize(value: int) -> dict[str, float]:
        visited.append(value)
        return {"loss": float(value), "accuracy": 1.0 - value / 10.0}

    stream = stream_materializations(range(5), materialize)
    assert visited == []
    whole = StreamingMetricReducer()
    for metrics in stream:
        whole.update(metrics)
    left = StreamingMetricReducer()
    right = StreamingMetricReducer()
    for value in range(2):
        left.update(materialize(value))
    for value in range(2, 5):
        right.update(materialize(value))
    left.merge(right)
    whole_result = whole.result()
    chunked_result = left.result()
    assert whole_result.keys() == chunked_result.keys()
    for name in whole_result:
        assert whole_result[name]["count"] == chunked_result[name]["count"]
        for statistic in ("mean", "standard_deviation", "minimum", "maximum"):
            assert whole_result[name][statistic] == pytest.approx(
                chunked_result[name][statistic], abs=1.0e-15
            )
    assert visited[:5] == list(range(5))


def test_convergence_and_failure_attribution_are_bounded_and_owner_specific() -> None:
    schedule = CapabilityConvergenceSchedule().validate()
    assert (
        schedule.stop_reason(
            completed_updates=2_048,
            elapsed_hours=1.0,
            raw_scores=(1.0, 0.997, 0.996, 0.995),
            accepted_in_window=False,
        )
        == "objective_plateau"
    )
    assert (
        schedule.stop_reason(
            completed_updates=8_192,
            elapsed_hours=1.0,
            raw_scores=(),
            accepted_in_window=False,
        )
        == "limit_hit"
    )
    attribution = attribute_failure(
        {
            "observed": 0.20,
            "clean_observation": 0.08,
            "truth_association": 0.18,
            "truth_parameters": 0.19,
            "truth_state_dynamics": 0.17,
        }
    )
    assert attribution["owner"] == "perception"
    assert attribution["capacity_change_supported"]
    admitted, missing = training_admission(
        {factor: "measured" for factor in ALL_CAPABILITY_FACTORS}
    )
    assert admitted and not missing
    assert PLANNING_TRAINING_LOSS_FIELDS == ()


def test_promotion_requires_absolute_planning_resource_and_paired_gates() -> None:
    factor = {
        "proposal_f1": 0.96,
        "identity_accuracy": 0.99,
        "lifecycle_f1": 0.96,
        "current_position_rmse_m": 0.015,
        "two_second_position_rmse_m": 0.10,
        "collision_f1": 0.92,
        "uncertainty_90_coverage": 0.90,
    }
    evidence = {
        "single_factors": {name: factor for name in CAPABILITY_FACTORS},
        "compositional_holdout": {
            "proposal_f1": 0.91,
            "identity_accuracy": 0.96,
            "two_second_position_rmse_m": 0.14,
            "planning_k8_winner_accuracy": 0.81,
            "planning_k32_winner_accuracy": 0.76,
        },
        "planning": {
            "k8_winner_accuracy": 0.91,
            "k32_winner_accuracy": 0.86,
            "k8_median_regret": 0.04,
            "k32_median_regret": 0.06,
            "goal_success": 0.91,
            "serial_vectorized_winner_parity": True,
        },
        "resources": {
            "learned_weight_bytes": 100_000,
            "rss_bytes": 1_000_000_000,
            "n16_rollout_seconds": 0.08,
        },
        "maximum_accepted_regression": 0.01,
        "invariant_failures": [],
        "relative_score_improvement": 0.04,
        "paired_bootstrap_improvement_lower_95": 0.005,
        "worst_family_improvement": 0.02,
        "latency_ratio": 1.05,
    }
    assert promotion_decision(evidence) == {"promoted": True, "failures": ()}
    rejected = promotion_decision(
        {**evidence, "planning": {**evidence["planning"], "k8_winner_accuracy": 0.80}}
    )
    assert not rejected["promoted"]
    assert "planning/k8_winner_accuracy" in rejected["failures"]


def test_packed_interactions_match_dense_oracle_and_skip_inactive_pairs() -> None:
    belief, objects = _objects()
    torch.manual_seed(19)
    dense = InteractionGraph(1, 1, hidden_dim=16, interaction_radius=0.5)
    packed = InteractionGraph(
        1,
        1,
        hidden_dim=16,
        interaction_radius=0.5,
        packed_interactions_enabled=True,
    )
    with torch.no_grad():
        dense.edge_network.output.bias.copy_(torch.tensor((0.2, -0.1, 0.3, -0.2, 0.1, -0.4, 0.2)))
        dense.node_network.output.bias.copy_(torch.tensor((0.1, -0.2, 0.3)))
    packed.load_state_dict(dense.state_dict(), strict=True)

    dense_output = dense(objects, belief.global_code)
    packed_output = packed(objects, belief.global_code)
    for field in fields(dense_output):
        left = getattr(dense_output, field.name)
        right = getattr(packed_output, field.name)
        if left.dtype.is_floating_point:
            torch.testing.assert_close(left, right, rtol=0.0, atol=1.0e-7)
        else:
            assert torch.equal(left, right)

    packed_loss = packed_output.residual_acceleration.square().sum()
    dense_loss = dense_output.residual_acceleration.square().sum()
    dense_loss.backward()
    packed_loss.backward()
    for dense_parameter, packed_parameter in zip(
        dense.parameters(), packed.parameters(), strict=True
    ):
        if dense_parameter.grad is None:
            assert packed_parameter.grad is None
        else:
            torch.testing.assert_close(
                dense_parameter.grad,
                packed_parameter.grad,
                rtol=0.0,
                atol=1.0e-6,
            )


@pytest.mark.parametrize("proposal_count", (6, 10, 14, 18))
def test_set_proposer_supports_derived_capacity(proposal_count: int) -> None:
    from world_model.observations.rgbd import RGBDSetProposer

    proposer = RGBDSetProposer(proposal_count=proposal_count)
    assert proposer.anchor_points.shape == (proposal_count, 2)
    assert proposer.parameter_count() <= 100_000


def test_variable_capacity_anchor_permutation_only_permutes_proposals() -> None:
    from world_model.observations.rgbd import RGBDSetProposer

    proposer = RGBDSetProposer(proposal_count=10).eval()
    image = torch.zeros(1, 3, 12, 12)
    log_depth = torch.zeros(1, 1, 12, 12)
    valid = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    foreground = torch.zeros(1, 1, 12, 12)
    axis = torch.linspace(-1.0, 1.0, 12)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((xx, yy)).unsqueeze(0)
    inputs = (image, log_depth, valid, foreground, coordinates)
    with torch.no_grad():
        reference = proposer(*inputs)
        anchors = proposer.anchor_points.clone()
        permutation = torch.tensor((6, 2, 9, 1, 7, 0, 8, 4, 3, 5))
        proposer.anchor_points.copy_(anchors[permutation])
        permuted = proposer(*inputs)
    for name in (
        "slot_mask_logits",
        "base_slot_mask_logits",
        "mask_residual",
        "existence_residual",
        "appearance_residual",
        "log_variance_residual",
        "query_features",
    ):
        assert torch.equal(getattr(permuted, name), getattr(reference, name)[:, permutation])
    assert torch.equal(
        permuted.full_mask_probability[:, 1:],
        reference.full_mask_probability[:, 1:][:, permutation],
    )


def test_state_only_scalability_probes_cover_n8_n12_n16() -> None:
    source = BeliefFactory(
        max_objects=2,
        residual_dynamics_dim=1,
        global_code_dim=1,
    ).create(gravity=(0.0, 0.0, 0.0))
    model = DynamicsModel.from_belief(
        source,
        max_substep=1.0 / 120.0,
        graph_hidden_dim=8,
        graph_relation_hidden_dim=8,
        uncertainty_hidden_dim=8,
        modal_dynamics_enabled=False,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        event_driven_state_only_enabled=True,
        packed_interactions_enabled=True,
        world_bounds=((-30.0, 30.0),) * 3,
        process_noise_position=1.0e-8,
        process_noise_velocity=1.0e-8,
        log_variance_min=-32.0,
    )
    probes = run_state_only_scalability_probes(model, warmup_runs=0, measured_runs=1)
    assert [probe.object_count for probe in probes] == [8, 12, 16]
    assert all(probe.finite and probe.batch_independent for probe in probes)
    assert all(probe.packed_interactions for probe in probes)
    assert all(not probe.full_perceptual_qualification for probe in probes)
    comparison = compare_dense_and_packed_scalability(
        model,
        counts=(8,),
        warmup_runs=0,
        measured_runs=1,
    )
    assert comparison["dense_oracle_probes"][0]["packed_interactions"] is False
    assert comparison["probes"][0]["packed_interactions"] is True
    assert model.interactions.packed_interactions_enabled is True
