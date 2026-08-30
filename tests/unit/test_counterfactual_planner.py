from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory
from world_model.dynamics import AnalyticFreeMotionDynamics, WorldImpulseAction
from world_model.planning import (
    CounterfactualCostWeights,
    TerminalWorldPositionGoal,
    plan_counterfactual_actions,
    resolve_appearance_handle,
)


def _two_object_belief(*, appearance_dim: int = 3):
    belief = BeliefFactory(max_objects=3, appearance_dim=appearance_dim).create(batch_size=2)
    objects = belief.objects.clone()
    objects.active[:, :2] = True
    objects.object_id[:] = torch.tensor([[11, 22, -1], [22, 11, -1]])
    objects.position[:] = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    objects.log_drag.fill_(math.log(0.1))
    objects.fast_log_variance[..., :3] = math.log(0.25)
    return replace(
        belief,
        objects=objects,
        gravity=torch.zeros_like(belief.gravity),
        next_object_id=torch.full_like(belief.next_object_id, 23),
    ).validate()


def _action(
    belief,
    impulse_world: torch.Tensor,
    *,
    object_id: torch.Tensor | None = None,
    timestamp: float = 0.25,
) -> WorldImpulseAction:
    if object_id is None:
        object_id = torch.full(
            (belief.batch_size,),
            11,
            device=belief.device,
            dtype=torch.int64,
        )
    return WorldImpulseAction(
        timestamp=belief.timestamp.new_full((belief.batch_size,), timestamp),
        object_id=object_id,
        impulse_world=impulse_world,
    )


class _CountingDynamics(AnalyticFreeMotionDynamics):
    def __init__(self) -> None:
        super().__init__()
        self.rollout_calls = 0

    def rollout(self, *args, **kwargs) -> BeliefTrajectory:
        self.rollout_calls += 1
        return super().rollout(*args, **kwargs)


def test_all_inputs_are_prevalidated_before_any_rollout() -> None:
    belief = _two_object_belief()
    dynamics = _CountingDynamics()
    valid = _action(belief, belief.objects.position.new_ones(2, 3))
    outside_horizon = _action(
        belief,
        belief.objects.position.new_ones(2, 3),
        timestamp=1.25,
    )
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([11, 11]),
        position_world=torch.zeros(2, 3),
    )

    with pytest.raises(ValueError, match="horizon|latest|timestamp"):
        plan_counterfactual_actions(
            dynamics,
            belief,
            [0.5, 1.0],
            (valid, outside_horizon),
            goal,
        )
    assert dynamics.rollout_calls == 0

    invalid_goal = replace(goal, object_id=torch.tensor([11, 999]))
    with pytest.raises(ValueError, match="exactly one active"):
        plan_counterfactual_actions(
            dynamics,
            belief,
            [0.5, 1.0],
            (valid,),
            invalid_goal,
        )
    assert dynamics.rollout_calls == 0

    with pytest.raises(TypeError, match="candidate"):
        plan_counterfactual_actions(
            dynamics,
            belief,
            [0.5, 1.0],
            (valid, object()),  # type: ignore[arg-type]
            goal,
        )
    assert dynamics.rollout_calls == 0


def test_result_components_use_persistent_ids_and_retain_all_candidates() -> None:
    belief = _two_object_belief()
    source = belief.clone()
    dynamics = AnalyticFreeMotionDynamics()
    impulse = belief.objects.position.new_tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    action = _action(belief, impulse)
    query_times = torch.tensor([0.25, 0.5, 1.0])
    acted = dynamics.rollout(belief, query_times, action=action)
    batch = torch.arange(2)
    target_slots = torch.tensor([0, 1])
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([11, 11]),
        position_world=acted.positions[batch, -1, target_slots].detach().clone(),
    )
    weights = CounterfactualCostWeights(
        terminal_position=2.0,
        terminal_variance=3.0,
        impulse_effort=0.4,
    )

    result = plan_counterfactual_actions(
        dynamics,
        belief,
        query_times,
        (None, action),
        goal,
        weights=weights,
    )

    assert result.actions == (None, action)
    assert len(result.trajectories) == 2
    assert result.terminal_squared_error.shape == (2, 2)
    assert result.terminal_position_variance.shape == (2, 2)
    assert result.impulse_effort.shape == (2, 2)
    assert result.total_cost.shape == (2, 2)
    assert result.selected_index.shape == (2,)
    torch.testing.assert_close(result.terminal_squared_error[:, 1], torch.zeros(2))
    torch.testing.assert_close(
        result.terminal_position_variance,
        torch.full((2, 2), 0.75),
    )
    torch.testing.assert_close(result.impulse_effort, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    torch.testing.assert_close(
        result.total_cost,
        2.0 * result.terminal_squared_error
        + 3.0 * result.terminal_position_variance
        + 0.4 * result.impulse_effort,
    )
    assert torch.equal(result.selected_index, torch.ones(2, dtype=torch.int64))
    assert torch.equal(result.object_id_by_slot, belief.objects.object_id)
    assert result.object_id_by_slot.data_ptr() != belief.objects.object_id.data_ptr()
    torch.testing.assert_close(belief.timestamp, source.timestamp)
    torch.testing.assert_close(belief.objects.position, source.objects.position)
    torch.testing.assert_close(belief.objects.velocity, source.objects.velocity)
    torch.testing.assert_close(belief.objects.fast_log_variance, source.objects.fast_log_variance)


def test_none_candidate_matches_ordinary_rollout_exactly() -> None:
    belief = _two_object_belief()
    dynamics = AnalyticFreeMotionDynamics()
    query_times = torch.tensor([0.0, 0.5, 1.0])
    ordinary = dynamics.rollout(belief, query_times)
    target_slots = torch.tensor([0, 1])
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([11, 11]),
        position_world=ordinary.positions[torch.arange(2), -1, target_slots].clone(),
    )

    result = plan_counterfactual_actions(
        dynamics,
        belief,
        query_times,
        (None,),
        goal,
    )
    planned = result.trajectories[0]

    for name in (
        "timestamps",
        "positions",
        "velocities",
        "orientations",
        "motion_mode_logits",
        "fast_log_variance",
        "active_mask",
        "event_logits",
    ):
        expected = getattr(ordinary, name)
        actual = getattr(planned, name)
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(actual, expected)
    assert ordinary.auxiliary.keys() == planned.auxiliary.keys()
    for name, expected in ordinary.auxiliary.items():
        assert torch.equal(planned.auxiliary[name], expected)
    assert torch.equal(result.selected_index, torch.zeros(2, dtype=torch.int64))


def test_first_argmin_wins_ties_and_candidate_permutation_reorders_columns() -> None:
    belief = _two_object_belief()
    dynamics = AnalyticFreeMotionDynamics()
    query_times = [0.5, 1.0]
    right = _action(belief, belief.objects.position.new_tensor([[0.7, 0.0, 0.0]]).expand(2, -1))
    left = _action(belief, belief.objects.position.new_tensor([[-0.7, 0.0, 0.0]]).expand(2, -1))
    right_trajectory = dynamics.rollout(belief, query_times, action=right)
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([11, 11]),
        position_world=right_trajectory.positions[
            torch.arange(2),
            -1,
            torch.tensor([0, 1]),
        ].detach(),
    )

    ties = plan_counterfactual_actions(
        dynamics,
        belief,
        query_times,
        (right, right),
        goal,
    )
    assert torch.equal(ties.selected_index, torch.zeros(2, dtype=torch.int64))

    original = plan_counterfactual_actions(
        dynamics,
        belief,
        query_times,
        (left, None, right),
        goal,
    )
    permuted = plan_counterfactual_actions(
        dynamics,
        belief,
        query_times,
        (right, left, None),
        goal,
    )
    torch.testing.assert_close(permuted.total_cost, original.total_cost[:, [2, 0, 1]])
    assert torch.equal(original.selected_index, torch.full((2,), 2, dtype=torch.int64))
    assert torch.equal(permuted.selected_index, torch.zeros(2, dtype=torch.int64))


def test_every_cost_column_retains_candidate_and_source_gradients() -> None:
    belief = _two_object_belief()
    source_position = belief.objects.position.detach().clone().requires_grad_()
    belief = replace(
        belief,
        objects=belief.objects.replace(position=source_position),
    )
    first_impulse = torch.tensor(
        [[0.4, 0.1, 0.0], [0.5, -0.1, 0.0]],
        requires_grad=True,
    )
    second_impulse = torch.tensor(
        [[-0.3, 0.2, 0.0], [-0.6, 0.1, 0.0]],
        requires_grad=True,
    )
    first = _action(belief, first_impulse)
    second = _action(belief, second_impulse)
    goal_position = torch.tensor(
        [[1.5, 0.25, 0.0], [-0.5, -0.25, 0.0]],
        requires_grad=True,
    )
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([11, 11]),
        position_world=goal_position,
    )

    result = plan_counterfactual_actions(
        AnalyticFreeMotionDynamics(),
        belief,
        [0.25, 0.75],
        (first, second),
        goal,
        weights=CounterfactualCostWeights(impulse_effort=0.2),
    )
    assert result.total_cost[:, 0].requires_grad
    assert result.total_cost[:, 1].requires_grad
    result.total_cost.sum().backward()

    for gradient in (
        first_impulse.grad,
        second_impulse.grad,
        source_position.grad,
        goal_position.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_cost_weight_contract_rejects_boolean_nonfinite_and_negative_values() -> None:
    with pytest.raises(TypeError, match="non-boolean"):
        CounterfactualCostWeights(terminal_position=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        CounterfactualCostWeights(terminal_variance=float("nan"))
    with pytest.raises(ValueError, match="nonnegative"):
        CounterfactualCostWeights(impulse_effort=-0.1)
    with pytest.raises(ValueError, match="strictly positive"):
        CounterfactualCostWeights(terminal_position=0.0)


def _appearance_belief():
    belief = BeliefFactory(max_objects=3, appearance_dim=3).create(batch_size=2)
    objects = belief.objects.clone()
    objects.active[:, :2] = True
    objects.object_id[:] = torch.tensor([[11, 22, -1], [11, 22, -1]])
    # The same red observable belongs to different persistent IDs and slots.
    objects.appearance[:] = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    return replace(
        belief,
        objects=objects,
        next_object_id=torch.full_like(belief.next_object_id, 23),
    ).validate()


def test_appearance_handle_follows_observable_palette_across_slots_and_ids() -> None:
    belief = _appearance_belief()
    red = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    resolved = resolve_appearance_handle(
        belief,
        red,
        minimum_cosine_margin=0.5,
    )
    assert torch.equal(resolved, torch.tensor([11, 22]))

    objects = belief.objects
    order = torch.tensor([1, 0, 2])
    permuted = belief.replace(
        objects=objects.replace(
            object_id=objects.object_id[:, order],
            active=objects.active[:, order],
            appearance=objects.appearance[:, order],
        )
    )
    permuted_resolved = resolve_appearance_handle(
        permuted,
        red,
        minimum_cosine_margin=0.5,
    )
    assert torch.equal(permuted_resolved, resolved)


def test_appearance_handle_rejects_ambiguity_bad_support_and_corrupt_beliefs() -> None:
    belief = _appearance_belief()
    prototype = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    ambiguous_appearance = belief.objects.appearance.clone()
    ambiguous_appearance[:, :2] = prototype[:, None, :]
    ambiguous = belief.replace(objects=belief.objects.replace(appearance=ambiguous_appearance))
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_appearance_handle(ambiguous, prototype, minimum_cosine_margin=0.0)

    unsupported_active = belief.objects.active.clone()
    unsupported_active[:, 1] = False
    unsupported_ids = belief.objects.object_id.clone()
    unsupported_ids[:, 1] = -1
    unsupported = belief.replace(
        objects=belief.objects.replace(active=unsupported_active, object_id=unsupported_ids)
    )
    with pytest.raises(ValueError, match="at least two"):
        resolve_appearance_handle(unsupported, prototype, minimum_cosine_margin=0.1)

    nonfinite_appearance = belief.objects.appearance.clone()
    nonfinite_appearance[0, 0, 0] = float("nan")
    nonfinite = belief.replace(objects=belief.objects.replace(appearance=nonfinite_appearance))
    with pytest.raises(ValueError, match="finite"):
        resolve_appearance_handle(nonfinite, prototype, minimum_cosine_margin=0.1)

    duplicate_ids = belief.objects.object_id.clone()
    duplicate_ids[:, 1] = duplicate_ids[:, 0]
    duplicate = belief.replace(objects=belief.objects.replace(object_id=duplicate_ids))
    with pytest.raises(ValueError, match="unique"):
        resolve_appearance_handle(duplicate, prototype, minimum_cosine_margin=0.1)


def test_appearance_handle_validates_prototype_and_margin_exactly() -> None:
    belief = _appearance_belief()
    prototype = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="shape"):
        resolve_appearance_handle(
            belief,
            prototype[:, :2],
            minimum_cosine_margin=0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        resolve_appearance_handle(
            belief,
            prototype.fill_(float("inf")),
            minimum_cosine_margin=0.1,
        )
    with pytest.raises(TypeError, match="non-boolean"):
        resolve_appearance_handle(
            belief,
            torch.ones(2, 3),
            minimum_cosine_margin=True,  # type: ignore[arg-type]
        )
