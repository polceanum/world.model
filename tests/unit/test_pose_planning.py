from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory, WorldBelief
from world_model.dynamics import (
    AnalyticFreeMotionDynamics,
    WorldImpulseAction,
    quaternion_from_rotation_vector,
)
from world_model.planning import (
    CounterfactualCostWeights,
    TerminalWorldPoseGoal,
    plan_counterfactual_actions,
)


class _ImpulseOrientationDynamics(AnalyticFreeMotionDynamics):
    """A minimal planner fixture mapping public impulses to terminal attitude."""

    def rollout(
        self,
        belief: WorldBelief,
        query_times,
        *,
        action=None,
        return_events: bool = True,
        return_auxiliary: bool = True,
    ) -> BeliefTrajectory:
        trajectory = super().rollout(
            belief,
            query_times,
            action=None,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
        )
        angle = belief.timestamp.new_zeros(belief.batch_size)
        if isinstance(action, WorldImpulseAction):
            angle = action.impulse_world[:, 2] * action._application_mask_for(belief)
        rotation_vector = torch.stack((torch.zeros_like(angle), torch.zeros_like(angle), angle), -1)
        orientation = quaternion_from_rotation_vector(rotation_vector)
        orientations = orientation[:, None, None, :].expand_as(trajectory.orientations).clone()
        return replace(trajectory, orientations=orientations).validate()


def _belief() -> WorldBelief:
    belief = BeliefFactory(max_objects=2).create(batch_size=1)
    objects = belief.objects.clone()
    objects.active[:, 0] = True
    objects.object_id[:, 0] = 7
    return replace(
        belief,
        objects=objects,
        gravity=torch.zeros_like(belief.gravity),
        next_object_id=torch.tensor([8]),
    ).validate()


def _action(belief: WorldBelief, angle: float) -> WorldImpulseAction:
    return WorldImpulseAction(
        timestamp=belief.timestamp.new_tensor([0.25]),
        object_id=torch.tensor([7]),
        impulse_world=belief.timestamp.new_tensor([[0.0, 0.0, angle]]),
    )


def test_pose_goal_scores_geodesic_attitude_and_preserves_vectorized_parity() -> None:
    belief = _belief()
    dynamics = _ImpulseOrientationDynamics()
    candidates = (None, _action(belief, 0.4), _action(belief, 1.1), _action(belief, 1.55))
    goal = TerminalWorldPoseGoal(
        object_id=torch.tensor([7]),
        position_world=torch.zeros(1, 3),
        orientation_world=quaternion_from_rotation_vector(torch.tensor([[0.0, 0.0, math.pi / 2]])),
    )
    weights = CounterfactualCostWeights(terminal_orientation=2.0)

    vectorized = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.5, 1.0],
        candidates,
        goal,
        weights=weights,
    )
    serial = plan_counterfactual_actions(
        dynamics,
        belief,
        [0.5, 1.0],
        candidates,
        goal,
        weights=weights,
        candidate_vectorized=False,
    )

    assert vectorized.terminal_orientation_error is not None
    assert vectorized.terminal_orientation_error.shape == (1, 4)
    torch.testing.assert_close(vectorized.total_cost, serial.total_cost, atol=0.0, rtol=0.0)
    assert torch.equal(vectorized.selected_index, torch.tensor([3]))
    torch.testing.assert_close(
        vectorized.total_cost,
        2.0 * vectorized.terminal_orientation_error,
    )


def test_pose_goal_rejects_non_unit_or_nonfinite_orientation_before_rollout() -> None:
    belief = _belief()
    dynamics = _ImpulseOrientationDynamics()
    goal = TerminalWorldPoseGoal(
        object_id=torch.tensor([7]),
        position_world=torch.zeros(1, 3),
        orientation_world=torch.tensor([[0.0, 0.0, 0.0, 2.0]]),
    )

    with pytest.raises(ValueError, match="unit quaternions"):
        plan_counterfactual_actions(dynamics, belief, [1.0], (None,), goal)
    with pytest.raises(ValueError, match="finite"):
        plan_counterfactual_actions(
            dynamics,
            belief,
            [1.0],
            (None,),
            replace(goal, orientation_world=torch.tensor([[0.0, 0.0, float("nan"), 1.0]])),
        )
