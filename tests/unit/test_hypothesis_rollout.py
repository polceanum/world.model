from __future__ import annotations

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory
from world_model.dynamics import (
    ConstantVelocityDynamics,
    DynamicsModel,
    HypothesisDynamicsPool,
    HypothesisRolloutEngine,
    RolloutStep,
)


def _trajectory(position: float, *, variance: float = 1.0) -> BeliefTrajectory:
    return BeliefTrajectory(
        timestamps=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        positions=torch.full((1, 2, 1, 3), position),
        velocities=torch.zeros(1, 2, 1, 3),
        orientations=torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]]),
        motion_mode_logits=torch.zeros(1, 2, 1, 2),
        fast_log_variance=torch.full((1, 2, 1, 13), variance),
        active_mask=torch.ones(1, 2, 1, dtype=torch.bool),
    ).validate()


def test_selector_chooses_best_candidate_per_batch() -> None:
    target = torch.zeros(1, 2, 1, 3)
    selection = HypothesisRolloutEngine.score(
        [_trajectory(0.5), _trajectory(0.0)],
        target,
        torch.ones(1, 2, 1, dtype=torch.bool),
        uncertainty_aware=False,
    )
    assert selection.selected_index.tolist() == [1]
    assert selection.scores[0, 1] < selection.scores[0, 0]
    assert torch.allclose(selection.posterior_weights.sum(-1), torch.ones(1))


def test_selector_ignores_occluded_frames_and_scores_uncertainty() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.tensor([[[True], [False]]])
    narrow = _trajectory(0.2, variance=0.0)
    wide = _trajectory(0.3, variance=2.0)
    selection = HypothesisRolloutEngine.score(
        [narrow, wide], target, mask, uncertainty_aware=True
    )
    # The log-variance penalty prevents a deliberately broad forecast from
    # winning just because its normalized residual is smaller.
    assert selection.selected_index.item() == 0


def test_selector_rejects_empty_or_bad_mask() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one"):
        HypothesisRolloutEngine.score([], target, mask)
    with pytest.raises(TypeError, match="torch.bool"):
        HypothesisRolloutEngine.score([_trajectory(0.0)], target, mask.to(torch.float32))


def test_dynamics_adapter_uses_predict_step_contract() -> None:
    belief = BeliefFactory(max_objects=1).create()
    source_position = belief.objects.position.clone()
    first = DynamicsModel.from_belief(belief, max_substep=0.05)
    second = DynamicsModel.from_belief(belief, max_substep=0.05)
    engine = HypothesisRolloutEngine()
    trajectories = engine.rollout_dynamics([first, second], belief, [0.05, 0.1])
    assert len(trajectories) == 2
    assert trajectories[0].positions.shape == (1, 2, 1, 3)
    torch.testing.assert_close(belief.objects.position, source_position)
    with pytest.raises(TypeError, match="predict_step"):
        engine.rollout_dynamics([object()], belief, [0.1])


class _FixedDynamics:
    def __init__(self, position: float) -> None:
        self.position = position

    def predict_step(self, belief, delta_time):
        objects = belief.objects.clone()
        objects.position[..., 0] = self.position
        endpoint = belief.replace(
            timestamp=belief.timestamp + delta_time,
            objects=objects,
        )
        return RolloutStep(
            belief=endpoint,
            event_logits=torch.zeros(belief.batch_size, objects.max_objects, 2),
            auxiliary={},
        )


def test_pool_assimilates_late_evidence_and_updates_selected_model() -> None:
    belief = BeliefFactory(max_objects=1).create()
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(belief, [0.1, 0.2])
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    selection = pool.assimilate(belief, target, mask, trajectories=trajectories, uncertainty_aware=False)
    assert selection.selected_index.tolist() == [1]
    assert pool.selected_index(belief).tolist() == [1]
    assert pool.last_selection is not None
    assert belief.timestamp.item() == pytest.approx(0.0)


def test_pool_selection_respects_accumulated_prior() -> None:
    belief = BeliefFactory(max_objects=1).create()
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)], temperature=1.0)
    trajectories = pool.rollout(belief, [0.1])
    target = torch.zeros_like(trajectories[0].positions)
    mask = torch.zeros_like(trajectories[0].active_mask)
    mask[:, :, 0] = True
    first = pool.assimilate(
        belief, target, mask, trajectories=trajectories, uncertainty_aware=False
    )
    assert first.selected_index.item() == 1
    # A second observation supports model 0, but not enough to erase the
    # posterior accumulated for model 1; the reported choice must follow the
    # posterior rather than the instantaneous raw score.
    target[..., 0] = 0.75
    second = pool.assimilate(
        belief, target, mask, trajectories=trajectories, uncertainty_aware=False
    )
    assert second.posterior_weights[0, 1] > second.posterior_weights[0, 0]
    assert second.selected_index.item() == 1


def test_constant_velocity_hypothesis_is_transparent_and_non_mutating() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.position[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    objects.velocity[0, 0] = torch.tensor([2.0, -1.0, 0.5])
    source = belief.replace(objects=objects)
    result = ConstantVelocityDynamics().predict_step(source, torch.tensor([0.25]))
    torch.testing.assert_close(
        result.belief.objects.position[0, 0],
        torch.tensor([1.5, 1.75, 3.125]),
    )
    torch.testing.assert_close(source.objects.position[0, 0], torch.tensor([1.0, 2.0, 3.0]))
