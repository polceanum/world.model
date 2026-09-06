from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode, WorldBelief
from world_model.dynamics import DynamicsModel, WorldImpulseAction
from world_model.planning import (
    CounterfactualCostWeights,
    TerminalWorldPositionGoal,
    plan_counterfactual_actions,
)


def _belief(*, requires_grad: bool = False) -> WorldBelief:
    belief = BeliefFactory(max_objects=3).create(
        batch_size=2,
        dtype=torch.float64,
        gravity=(0.0, 0.0, 0.0),
    )
    objects = belief.objects.clone()
    objects.active[:, :2] = True
    objects.object_id[:] = torch.tensor([[7, 41, -1], [41, 7, -1]])
    objects.position[:] = belief.timestamp.new_tensor(
        [
            [[-1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    objects.velocity[:, :2] = belief.timestamp.new_tensor(
        [
            [[0.2, -0.1, 0.05], [-0.1, 0.1, 0.0]],
            [[-0.1, 0.1, 0.0], [0.2, -0.1, 0.05]],
        ]
    )
    objects.log_mass[:, 0, 0] = math.log(2.0)
    objects.log_mass[:, 1, 0] = math.log(4.0)
    objects.log_drag.fill_(math.log(0.2))
    objects.geometry[..., 0] = 0.05
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., MotionMode.FREE] = 4.0
    if requires_grad:
        objects = objects.replace(
            position=objects.position.detach().clone().requires_grad_(),
            velocity=objects.velocity.detach().clone().requires_grad_(),
        )
    return belief.replace(
        objects=objects,
        next_object_id=torch.full_like(belief.next_object_id, 42),
    ).validate()


def _dynamics(belief: WorldBelief) -> DynamicsModel:
    model = DynamicsModel.from_belief(
        belief,
        max_substep=0.05,
        graph_hidden_dim=8,
        uncertainty_hidden_dim=8,
        interaction_radius=0.2,
        base_process_variance_per_second=1.0e-8,
        ground_height=-10.0,
    ).to(dtype=belief.dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _slice_belief_batch(belief: WorldBelief, index: int) -> WorldBelief:
    objects = belief.objects.replace(
        **{
            item.name: getattr(belief.objects, item.name)[index : index + 1].clone()
            for item in fields(belief.objects)
        }
    )
    camera = belief.camera.replace(
        **{
            item.name: getattr(belief.camera, item.name)[index : index + 1].clone()
            for item in fields(belief.camera)
        }
    )
    return belief.replace(
        timestamp=belief.timestamp[index : index + 1].clone(),
        objects=objects,
        camera=camera,
        gravity=belief.gravity[index : index + 1].clone(),
        global_code=belief.global_code[index : index + 1].clone(),
        global_log_variance=belief.global_log_variance[index : index + 1].clone(),
        next_object_id=belief.next_object_id[index : index + 1].clone(),
    ).validate()


def _action(belief: WorldBelief, impulse: torch.Tensor) -> WorldImpulseAction:
    return WorldImpulseAction(
        timestamp=belief.timestamp.new_tensor([0.2, 0.35]),
        object_id=torch.tensor([7, 7], dtype=torch.int64),
        impulse_world=impulse,
    )


def _assert_trajectory_equal(left: object, right: object) -> None:
    for item in fields(left):
        left_value = getattr(left, item.name)
        right_value = getattr(right, item.name)
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value), item.name
        elif isinstance(left_value, dict):
            assert left_value.keys() == right_value.keys()
            for name in left_value:
                assert torch.equal(left_value[name], right_value[name]), name
        else:
            assert left_value is right_value


def test_shared_action_is_absolute_id_addressed_and_exactly_once() -> None:
    belief = _belief(requires_grad=True)
    source = belief.clone()
    dynamics = _dynamics(belief)
    impulse = belief.timestamp.new_tensor([[0.8, -0.4, 0.2], [0.8, -0.4, 0.2]]).requires_grad_()
    action = _action(belief, impulse)
    queries = belief.timestamp.new_tensor([[0.1, 0.2, 0.2, 0.6], [0.1, 0.3, 0.35, 0.6]])

    baseline = dynamics.rollout(belief, queries)
    acted = dynamics.rollout(belief, queries, action=action)
    target_slots = torch.tensor([0, 1])
    boundary_indices = torch.tensor([1, 2])
    batch = torch.arange(2)
    boundary_position_delta = (
        acted.positions[batch, boundary_indices, target_slots]
        - baseline.positions[batch, boundary_indices, target_slots]
    )
    boundary_velocity_delta = (
        acted.velocities[batch, boundary_indices, target_slots]
        - baseline.velocities[batch, boundary_indices, target_slots]
    )

    assert torch.equal(boundary_position_delta, torch.zeros_like(boundary_position_delta))
    torch.testing.assert_close(
        boundary_velocity_delta,
        impulse / belief.objects.mass[batch, target_slots],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    applied = acted.auxiliary["known_action_applied"]
    assert applied.sum(dim=(1, 2)).tolist() == [1, 1]
    assert applied[0, 1, 0]
    assert applied[1, 2, 1]
    assert not applied[0, 2].any()
    modes = acted.event_logits.argmax(dim=-1)
    assert modes[0, 1, 0].item() == MotionMode.EXTERNALLY_ACTUATED
    assert modes[1, 2, 1].item() == MotionMode.EXTERNALLY_ACTUATED

    gradient = torch.autograd.grad(acted.positions[:, -1].sum(), impulse)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == gradient.numel()
    assert torch.equal(belief.timestamp, source.timestamp)
    assert torch.equal(belief.objects.position, source.objects.position)
    assert torch.equal(belief.objects.velocity, source.objects.velocity)


def test_shared_action_none_preserves_exact_values_auxiliaries_and_gradients() -> None:
    belief = _belief(requires_grad=True)
    dynamics = _dynamics(belief)
    queries = belief.timestamp.new_tensor([0.1, 0.25, 0.6])

    implicit = dynamics.rollout(belief, queries)
    explicit = dynamics.rollout(belief, queries, action=None)
    _assert_trajectory_equal(implicit, explicit)

    sources = (belief.objects.position, belief.objects.velocity)
    implicit_gradients = torch.autograd.grad(
        implicit.positions.sum() + implicit.velocities.sum(),
        sources,
        retain_graph=True,
    )
    explicit_gradients = torch.autograd.grad(
        explicit.positions.sum() + explicit.velocities.sum(),
        sources,
    )
    for implicit_gradient, explicit_gradient in zip(
        implicit_gradients,
        explicit_gradients,
        strict=True,
    ):
        assert torch.equal(implicit_gradient, explicit_gradient)


def test_shared_predict_step_matches_one_query_and_rejects_action_reuse() -> None:
    belief = _belief()
    dynamics = _dynamics(belief)
    action = _action(
        belief,
        belief.timestamp.new_tensor([[0.4, -0.2, 0.1], [0.2, 0.3, -0.1]]),
    )

    step = dynamics.predict_step(belief, 0.6, action=action)
    trajectory = dynamics.rollout(belief, [0.6], action=action)
    assert torch.equal(step.belief.timestamp, trajectory.timestamps[:, 0])
    assert torch.equal(step.belief.objects.position, trajectory.positions[:, 0])
    assert torch.equal(step.belief.objects.velocity, trajectory.velocities[:, 0])
    assert torch.equal(step.event_logits, trajectory.event_logits[:, 0])
    for name, value in step.auxiliary.items():
        assert torch.equal(value, trajectory.auxiliary[name][:, 0]), name

    at_action = dynamics.predict(belief, action.timestamp - belief.timestamp, action=action)
    with pytest.raises(ValueError, match="strictly after"):
        dynamics.predict(at_action, 0.1, action=action)


def test_shared_action_keeps_post_action_contact_evidence() -> None:
    belief = BeliefFactory(max_objects=2).create(
        batch_size=1,
        dtype=torch.float64,
        gravity=(0.0, 0.0, 0.0),
    )
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.tensor([[10, 20]], dtype=torch.int64)
    objects.position[:] = belief.timestamp.new_tensor([[[-0.3, 2.0, 0.0], [0.3, 2.0, 0.0]]])
    objects.geometry[..., 0] = 0.1
    objects.log_drag.fill_(-16.0)
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., MotionMode.FREE] = 4.0
    belief = belief.replace(
        objects=objects,
        next_object_id=torch.tensor([21], dtype=torch.int64),
    ).validate()
    dynamics = DynamicsModel.from_belief(
        belief,
        max_substep=0.005,
        graph_hidden_dim=8,
        uncertainty_hidden_dim=8,
        interaction_radius=0.05,
        base_process_variance_per_second=1.0e-8,
        ground_height=-10.0,
    ).to(dtype=belief.dtype)
    with torch.no_grad():
        for parameter in dynamics.parameters():
            parameter.zero_()
    action = WorldImpulseAction(
        timestamp=belief.timestamp.new_tensor([0.05]),
        object_id=torch.tensor([10], dtype=torch.int64),
        impulse_world=belief.timestamp.new_tensor([[4.0, 0.0, 0.0]]),
    )

    # The action and resulting collision deliberately share one public output
    # interval.  The action marker must not replace its collision evidence.
    trajectory = dynamics.rollout(belief, [0.2], action=action)

    assert trajectory.auxiliary["known_action_applied"][0, 0, 0]
    assert trajectory.auxiliary["known_action_applied"].sum().item() == 1
    assert trajectory.auxiliary["pair_collision"][0, 0, 0, 1]
    assert trajectory.event_logits[0, 0, :, MotionMode.COLLISION].gt(0.0).all()
    assert trajectory.event_logits[0, 0, 0, MotionMode.EXTERNALLY_ACTUATED] > 0.0


def test_shared_zero_impulse_is_differentiable_but_not_labelled_applied() -> None:
    belief = _belief()
    dynamics = _dynamics(belief)
    impulse = belief.timestamp.new_zeros(2, 3).requires_grad_()
    action = _action(belief, impulse)

    ordinary = dynamics.rollout(belief, [0.1, 0.6])
    acted = dynamics.rollout(belief, [0.1, 0.6], action=action)
    gradient = torch.autograd.grad(acted.positions[:, -1].sum(), impulse)[0]

    torch.testing.assert_close(acted.positions, ordinary.positions, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        acted.velocities,
        ordinary.velocities,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == gradient.numel()
    assert not acted.auxiliary["known_action_applied"].any()
    assert torch.count_nonzero(acted.auxiliary["known_impulse_world"]) == 0


def test_shared_vectorized_planner_matches_serial_public_result() -> None:
    belief = _belief()
    dynamics = _dynamics(belief)
    first = _action(
        belief,
        belief.timestamp.new_tensor([[0.5, 0.1, 0.0], [0.4, -0.1, 0.0]]),
    )
    second = replace(
        first,
        impulse_world=belief.timestamp.new_tensor([[-0.3, 0.2, 0.0], [-0.6, 0.1, 0.0]]),
    )
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([7, 7], dtype=torch.int64),
        position_world=belief.timestamp.new_tensor([[0.2, 2.0, 0.0], [0.2, 2.0, 0.0]]),
    )
    candidates = (None, first, second)

    vectorized = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.4, 0.6],
        candidates,
        goal,
    )
    serial = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.4, 0.6],
        candidates,
        goal,
        candidate_vectorized=False,
    )

    for name in (
        "terminal_squared_error",
        "terminal_position_variance",
        "impulse_effort",
        "total_cost",
        "selected_index",
        "object_id_by_slot",
    ):
        torch.testing.assert_close(getattr(vectorized, name), getattr(serial, name))
    for vectorized_trajectory, serial_trajectory in zip(
        vectorized.trajectories,
        serial.trajectories,
        strict=True,
    ):
        for item in fields(vectorized_trajectory):
            vectorized_value = getattr(vectorized_trajectory, item.name)
            serial_value = getattr(serial_trajectory, item.name)
            if isinstance(vectorized_value, torch.Tensor):
                torch.testing.assert_close(vectorized_value, serial_value)
            elif isinstance(vectorized_value, dict):
                assert vectorized_value.keys() == serial_value.keys()
                for key in vectorized_value:
                    torch.testing.assert_close(
                        vectorized_value[key],
                        serial_value[key],
                        msg=lambda message, key=key: f"{key}: {message}",
                    )
            else:
                assert vectorized_value is serial_value


def test_shared_planner_is_candidate_order_and_batch_independent_without_cross_boundaries() -> None:
    belief = _belief()
    dynamics = _dynamics(belief)
    template = _action(
        belief,
        belief.timestamp.new_tensor([[0.5, 0.1, 0.0], [0.4, -0.1, 0.0]]),
    )
    early = replace(
        template,
        timestamp=belief.timestamp.new_full((belief.batch_size,), 0.051),
    )
    late = replace(
        template,
        timestamp=belief.timestamp.new_full((belief.batch_size,), 0.349),
        impulse_world=-template.impulse_world,
    )
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([7, 7], dtype=torch.int64),
        position_world=belief.timestamp.new_tensor([[0.2, 2.0, 0.0], [0.2, 2.0, 0.0]]),
    )
    weights = CounterfactualCostWeights(terminal_variance=1.0)
    candidates = (None, early, late)

    vectorized = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.6],
        candidates,
        goal,
        weights=weights,
    )
    serial = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.6],
        candidates,
        goal,
        weights=weights,
        candidate_vectorized=False,
    )

    direct_none = dynamics.rollout(belief, [0.2, 0.6])
    direct_early = dynamics.rollout(belief, [0.2, 0.6], action=early)
    for planned in (vectorized.trajectories[0], serial.trajectories[0]):
        _assert_trajectory_equal(planned, direct_none)
    for planned in (vectorized.trajectories[1], serial.trajectories[1]):
        _assert_trajectory_equal(planned, direct_early)

    assert vectorized.total_cost.shape == (belief.batch_size, len(candidates))
    torch.testing.assert_close(vectorized.total_cost, serial.total_cost, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(
        vectorized.terminal_position_variance,
        serial.terminal_position_variance,
        rtol=0.0,
        atol=1.0e-6,
    )
    for vectorized_trajectory, serial_trajectory in zip(
        vectorized.trajectories,
        serial.trajectories,
        strict=True,
    ):
        assert vectorized_trajectory.timestamps.shape[1] == 2
        for item in fields(vectorized_trajectory):
            vectorized_value = getattr(vectorized_trajectory, item.name)
            serial_value = getattr(serial_trajectory, item.name)
            if isinstance(vectorized_value, torch.Tensor):
                if vectorized_value.dtype.is_floating_point:
                    torch.testing.assert_close(
                        vectorized_value,
                        serial_value,
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                else:
                    assert torch.equal(vectorized_value, serial_value)
            elif isinstance(vectorized_value, dict):
                assert vectorized_value.keys() == serial_value.keys()
                for key in vectorized_value:
                    if vectorized_value[key].dtype.is_floating_point:
                        torch.testing.assert_close(
                            vectorized_value[key],
                            serial_value[key],
                            rtol=0.0,
                            atol=1.0e-6,
                        )
                    else:
                        assert torch.equal(vectorized_value[key], serial_value[key])
            else:
                assert vectorized_value is serial_value

    permutation = (late, None, early)
    permuted = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.6],
        permutation,
        goal,
        weights=weights,
    )
    torch.testing.assert_close(
        permuted.total_cost,
        vectorized.total_cost[:, [2, 0, 1]],
        rtol=0.0,
        atol=1.0e-6,
    )
    for permuted_index, original_index in enumerate((2, 0, 1)):
        torch.testing.assert_close(
            permuted.trajectories[permuted_index].positions,
            vectorized.trajectories[original_index].positions,
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            permuted.trajectories[permuted_index].fast_log_variance,
            vectorized.trajectories[original_index].fast_log_variance,
            rtol=0.0,
            atol=1.0e-6,
        )

    for batch_index in range(belief.batch_size):
        single_belief = _slice_belief_batch(belief, batch_index)
        single_candidates = tuple(
            None
            if action is None
            else replace(
                action,
                timestamp=action.timestamp[batch_index : batch_index + 1].clone(),
                object_id=action.object_id[batch_index : batch_index + 1].clone(),
                impulse_world=action.impulse_world[batch_index : batch_index + 1].clone(),
            )
            for action in candidates
        )
        single_goal = replace(
            goal,
            object_id=goal.object_id[batch_index : batch_index + 1].clone(),
            position_world=goal.position_world[batch_index : batch_index + 1].clone(),
        )
        single = plan_counterfactual_actions(
            dynamics,
            single_belief,
            [0.2, 0.6],
            single_candidates,
            single_goal,
            weights=weights,
        )
        torch.testing.assert_close(
            vectorized.total_cost[batch_index],
            single.total_cost[0],
            rtol=0.0,
            atol=1.0e-6,
        )
        for combined_trajectory, single_trajectory in zip(
            vectorized.trajectories,
            single.trajectories,
            strict=True,
        ):
            torch.testing.assert_close(
                combined_trajectory.positions[batch_index],
                single_trajectory.positions[0],
                rtol=0.0,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                combined_trajectory.fast_log_variance[batch_index],
                single_trajectory.fast_log_variance[0],
                rtol=0.0,
                atol=1.0e-6,
            )
