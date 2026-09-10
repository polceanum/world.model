from __future__ import annotations

import math

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.dynamics import (
    AnalyticFreeMotionDynamics,
    DynamicsModel,
    WorldImpulseAction,
    WorldImpulseSchedule,
)
from world_model.planning import TerminalWorldPositionGoal, plan_counterfactual_actions


def _belief(*, batch_size: int = 2):
    belief = BeliefFactory(max_objects=2).create(
        batch_size=batch_size,
        dtype=torch.float64,
        gravity=(0.0, 0.0, 0.0),
    )
    objects = belief.objects.clone()
    objects.active[:, 0] = True
    objects.object_id[:, 0] = 7
    objects.position[:, 0] = belief.timestamp.new_tensor([0.0, 1.0, 0.0])
    objects.geometry[..., 0] = 0.05
    objects.log_mass[:, 0, 0] = math.log(2.0)
    objects.log_drag.fill_(-20.0)
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., MotionMode.FREE] = 4.0
    return belief.replace(
        objects=objects,
        next_object_id=torch.full_like(belief.next_object_id, 8),
    ).validate()


def _action(belief, timestamp: float, impulse_x: float) -> WorldImpulseAction:
    return WorldImpulseAction(
        timestamp=belief.timestamp.new_full((belief.batch_size,), timestamp),
        object_id=torch.full(
            (belief.batch_size,),
            7,
            dtype=torch.int64,
            device=belief.device,
        ),
        impulse_world=belief.timestamp.new_tensor([[impulse_x, 0.0, 0.0]] * belief.batch_size),
    )


def _shared_dynamics(belief) -> DynamicsModel:
    dynamics = DynamicsModel.from_belief(
        belief,
        max_substep=0.05,
        graph_hidden_dim=8,
        uncertainty_hidden_dim=8,
        interaction_radius=0.2,
        base_process_variance_per_second=1.0e-8,
        ground_height=-10.0,
    ).to(dtype=belief.dtype)
    with torch.no_grad():
        for parameter in dynamics.parameters():
            parameter.zero_()
    return dynamics


def test_schedule_validates_order_before_rollout() -> None:
    belief = _belief()
    reversed_schedule = WorldImpulseSchedule(
        actions=(_action(belief, 0.4, 1.0), _action(belief, 0.2, -0.5))
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        AnalyticFreeMotionDynamics().rollout(
            belief,
            [0.8],
            action=reversed_schedule,
        )


@pytest.mark.parametrize("shared", [False, True])
def test_schedule_applies_each_impulse_once_and_preserves_source(shared: bool) -> None:
    belief = _belief()
    source = belief.clone()
    schedule = WorldImpulseSchedule(actions=(_action(belief, 0.2, 2.0), _action(belief, 0.5, -1.0)))
    dynamics = _shared_dynamics(belief) if shared else AnalyticFreeMotionDynamics()

    trajectory = dynamics.rollout(
        belief,
        [0.1, 0.2, 0.4, 0.5, 0.8],
        action=schedule,
    )

    assert trajectory.auxiliary["known_action_count"][:, :, 0].tolist() == [
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
    ]
    assert trajectory.auxiliary["known_action_count"].sum(dim=(1, 2)).tolist() == [2, 2]
    torch.testing.assert_close(
        trajectory.velocities[:, -1, 0, 0],
        belief.timestamp.new_full((belief.batch_size,), 0.5),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert torch.equal(belief.timestamp, source.timestamp)
    assert torch.equal(belief.objects.position, source.objects.position)
    assert torch.equal(belief.objects.velocity, source.objects.velocity)


@pytest.mark.parametrize("shared", [False, True])
def test_schedule_planner_vectorization_matches_serial(shared: bool) -> None:
    belief = _belief()
    dynamics = _shared_dynamics(belief) if shared else AnalyticFreeMotionDynamics()
    first = WorldImpulseSchedule(actions=(_action(belief, 0.2, 1.0), _action(belief, 0.5, -0.25)))
    second = WorldImpulseSchedule(actions=(_action(belief, 0.2, -0.5), _action(belief, 0.6, 1.5)))
    goal = TerminalWorldPositionGoal(
        object_id=torch.full((belief.batch_size,), 7, dtype=torch.int64),
        position_world=belief.timestamp.new_tensor([[0.3, 1.0, 0.0]] * belief.batch_size),
    )
    candidates = (None, first, second)

    vectorized = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.5, 0.8],
        candidates,
        goal,
    )
    serial = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.5, 0.8],
        candidates,
        goal,
        candidate_vectorized=False,
    )

    torch.testing.assert_close(vectorized.total_cost, serial.total_cost)
    assert torch.equal(vectorized.selected_index, serial.selected_index)
    for actual, expected in zip(
        vectorized.trajectories,
        serial.trajectories,
        strict=True,
    ):
        torch.testing.assert_close(actual.positions, expected.positions)
        torch.testing.assert_close(actual.velocities, expected.velocities)


def test_mixed_single_and_schedule_candidates_have_exact_parity() -> None:
    belief = _belief()
    dynamics = AnalyticFreeMotionDynamics()
    single = _action(belief, 0.2, 1.0)
    schedule = WorldImpulseSchedule(actions=(single, _action(belief, 0.5, -0.25)))
    goal = TerminalWorldPositionGoal(
        object_id=torch.full((belief.batch_size,), 7, dtype=torch.int64),
        position_world=belief.timestamp.new_tensor([[0.3, 1.0, 0.0]] * belief.batch_size),
    )

    vectorized = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.5, 0.8],
        (None, single, schedule),
        goal,
    )
    serial = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.2, 0.5, 0.8],
        (None, single, schedule),
        goal,
        candidate_vectorized=False,
    )
    torch.testing.assert_close(vectorized.total_cost, serial.total_cost)
    assert torch.equal(vectorized.selected_index, serial.selected_index)
