from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory
from world_model.dynamics import (
    BallisticContactDynamics,
    ConstantVelocityDynamics,
    DynamicsModel,
    HypothesisDynamicsPool,
    HypothesisRolloutEngine,
    RolloutStep,
    RuntimeHypothesisController,
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
        event_logits=torch.zeros(1, 2, 1, 11),
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
    assert selection.axis_scores is not None
    assert selection.axis_scores.shape == (1, 3, 2)
    assert selection.axis_scores[0, :, 1].lt(selection.axis_scores[0, :, 0]).all()
    assert selection.axis_selected_index.tolist() == [[1, 1, 1]]
    axis_weights = selection.axis_posterior_weights()
    assert axis_weights.shape == (1, 3, 2)
    assert torch.allclose(axis_weights.sum(dim=-1), torch.ones(1, 3))


def test_selector_can_use_collision_evidence() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    collision = torch.ones(1, 2, 1, dtype=torch.bool)
    no_event = _trajectory(0.0)
    event = _trajectory(0.0)
    no_event.event_logits[..., 3] = -6.0
    event.event_logits[..., 3] = 6.0
    selection = HypothesisRolloutEngine.score(
        [no_event, event],
        target,
        mask,
        target_collision=collision,
        event_weight=1.0,
        uncertainty_aware=False,
    )
    assert selection.selected_index.tolist() == [1]


def test_position_gate_blocks_event_candidate_with_large_position_regression() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    collision = torch.ones(1, 2, 1, dtype=torch.bool)
    position_good = _trajectory(0.0)
    event_bad_position = _trajectory(1.0)
    position_good.event_logits[..., 3] = -6.0
    event_bad_position.event_logits[..., 3] = 6.0
    selection = HypothesisRolloutEngine.score(
        [position_good, event_bad_position],
        target,
        mask,
        target_collision=collision,
        event_weight=100.0,
        position_gate_ratio=0.05,
        uncertainty_aware=False,
    )
    assert selection.selected_index.tolist() == [0]


def test_axis_weights_can_prefer_lower_error_on_critical_axis() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    x_better = _trajectory(0.0)
    y_better = _trajectory(0.0)
    x_better.positions[..., 0] = 0.2
    x_better.positions[..., 1] = 1.0
    y_better.positions[..., 0] = 1.0
    y_better.positions[..., 1] = 0.2
    unweighted = HypothesisRolloutEngine.score(
        [x_better, y_better], target, mask, uncertainty_aware=False
    )
    y_weighted = HypothesisRolloutEngine.score(
        [x_better, y_better],
        target,
        mask,
        axis_weights=(1.0, 4.0, 1.0),
        uncertainty_aware=False,
    )
    assert unweighted.selected_index.tolist() == [0]
    assert y_weighted.selected_index.tolist() == [1]


def test_axis_gate_blocks_candidate_with_single_axis_regression() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    learned = _trajectory(0.0)
    alternative = _trajectory(0.0)
    learned.positions[..., 0] = 0.2
    alternative.positions[..., 0] = 0.2
    learned.positions[..., 1] = 0.2
    alternative.positions[..., 1] = 1.0
    selection = HypothesisRolloutEngine.score(
        [learned, alternative],
        target,
        mask,
        event_weight=0.0,
        axis_gate_ratio=0.05,
        uncertainty_aware=False,
    )
    assert selection.selected_index.tolist() == [0]


def test_event_gate_blocks_candidate_with_event_regression() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    collision = torch.ones(1, 2, 1, dtype=torch.bool)
    event_good = _trajectory(0.0)
    event_bad = _trajectory(0.0)
    event_good.event_logits[..., 3] = 4.0
    event_bad.event_logits[..., 3] = -4.0
    selection = HypothesisRolloutEngine.score(
        [event_good, event_bad],
        target,
        mask,
        target_collision=collision,
        event_weight=1.0,
        event_gate_ratio=0.05,
        uncertainty_aware=False,
    )
    assert selection.selected_index.tolist() == [0]


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


def test_ensemble_scoring_penalizes_brittle_nearby_rollouts() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    engine = HypothesisRolloutEngine()
    samples = [
        [_trajectory(0.0), _trajectory(0.45)],
        [_trajectory(0.6), _trajectory(0.45)],
    ]
    expected_error = engine.score_ensemble(
        samples, target, mask, uncertainty_aware=False
    )
    robust_error = engine.score_ensemble(
        samples, target, mask, uncertainty_aware=False, risk_penalty=0.2
    )
    assert expected_error.sample_count == 2
    assert expected_error.selected_index.tolist() == [0]
    assert robust_error.selected_index.tolist() == [1]
    assert robust_error.score_spread is not None
    assert robust_error.score_spread[0, 0] > robust_error.score_spread[0, 1]
    assert robust_error.axis_score_spread is not None


def test_pool_assimilates_robust_ensemble_evidence_without_mutating_belief() -> None:
    belief = BeliefFactory(max_objects=1).create()
    pool = HypothesisDynamicsPool([_FixedDynamics(0.0), _FixedDynamics(0.45)])
    target = torch.zeros(1, 1, 1, 3)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    first = pool.rollout(belief, [0.1])
    nearby_world = HypothesisDynamicsPool([_FixedDynamics(0.6), _FixedDynamics(0.45)])
    second = nearby_world.rollout(belief, [0.1])
    selection = pool.assimilate_ensemble(
        belief,
        target,
        mask,
        trajectory_samples=[first, second],
        risk_penalty=0.2,
        uncertainty_aware=False,
    )
    assert selection.sample_count == 2
    assert selection.selected_index.tolist() == [1]
    assert pool.selected_index(belief).tolist() == [1]
    assert belief.timestamp.item() == pytest.approx(0.0)


def test_pool_rollout_ensemble_uses_belief_uncertainty_without_mutation() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.position[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    objects.velocity[0, 0] = torch.tensor([0.5, 0.0, 0.0])
    objects.fast_log_variance[0, 0, :6] = 0.0
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([ConstantVelocityDynamics()])
    ensembles = pool.rollout_ensemble(
        source,
        [0.1],
        sample_count=2,
        position_std_scale=0.1,
        velocity_std_scale=0.1,
        generator=torch.Generator().manual_seed(7),
    )
    assert len(ensembles) == 2
    assert len(ensembles[0]) == 1
    torch.testing.assert_close(ensembles[0][0].positions[0, 0, 0], torch.tensor([1.05, 2.0, 3.0]))
    assert not torch.equal(ensembles[0][0].positions, ensembles[1][0].positions)
    torch.testing.assert_close(source.objects.position[0, 0], torch.tensor([1.0, 2.0, 3.0]))


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
            event_logits=torch.zeros(
                belief.batch_size, objects.max_objects, objects.motion_mode_logits.shape[-1]
            ),
            auxiliary={},
        )


class _CountingFixedDynamics(_FixedDynamics):
    def __init__(self, position: float) -> None:
        super().__init__(position)
        self.calls = 0

    def predict_step(self, belief, delta_time):
        self.calls += 1
        return super().predict_step(belief, delta_time)


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
    assert pool.selected_axis_index(belief).shape == (1, 3)
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


def test_pool_evidence_decay_allows_local_model_switch() -> None:
    belief = BeliefFactory(max_objects=1).create()
    pool = HypothesisDynamicsPool(
        [_FixedDynamics(1.0), _FixedDynamics(0.0)], evidence_decay=0.1
    )
    trajectories = pool.rollout(belief, [0.1])
    mask = torch.ones_like(trajectories[0].active_mask)
    first = pool.assimilate(
        belief,
        torch.zeros_like(trajectories[0].positions),
        mask,
        trajectories=trajectories,
        uncertainty_aware=False,
    )
    assert first.selected_index.item() == 1
    second = pool.assimilate(
        belief,
        torch.ones_like(trajectories[0].positions),
        mask,
        trajectories=trajectories,
        uncertainty_aware=False,
    )
    assert second.selected_index.item() == 0


def test_runtime_controller_uses_only_associated_measurements_and_splices_x() -> None:
    """A pending rollout is scored from RGB-style associated evidence only."""

    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    at_due_time = source.replace(timestamp=torch.tensor([0.1]))
    measured = SimpleNamespace(
        timestamp=torch.tensor([0.1]),
        measurement_mask=torch.tensor([[True]]),
        auxiliary={"world_position": torch.tensor([[[0.0, 0.0, 0.0]]])},
    )
    association = SimpleNamespace(
        pair_mask=torch.tensor([[True]]),
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
    )

    selection = controller.assimilate_observation(at_due_time, measured, association)

    assert selection is not None
    assert controller.pool.selected_index(at_due_time).tolist() == [1]
    forecast = controller.predict(at_due_time, [0.1])
    assert forecast is not None
    assert forecast.positions[0, 0, 0, 0] == pytest.approx(0.0)
    assert "hypothesis_axis_index" in forecast.auxiliary
    torch.testing.assert_close(source.objects.position, belief.objects.position)


def test_runtime_controller_discards_late_evidence_without_interpolation() -> None:
    belief = BeliefFactory(max_objects=1).create()
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(0.0), _FixedDynamics(1.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=belief.device, dtype=belief.dtype)
    controller.schedule(belief)
    late = belief.replace(timestamp=torch.tensor([0.2]))
    measured = SimpleNamespace(
        timestamp=torch.tensor([0.2]),
        measurement_mask=torch.tensor([[True]]),
        auxiliary={"world_position": torch.zeros(1, 1, 3)},
    )
    association = SimpleNamespace(
        pair_mask=torch.tensor([[True]]),
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
    )
    assert controller.assimilate_observation(late, measured, association) is None
    assert controller.pool.last_selection is None
    assert not controller.pending


def test_runtime_controller_rolls_out_only_selected_axis_candidates() -> None:
    belief = BeliefFactory(max_objects=1).create()
    models = [_CountingFixedDynamics(0.0), _CountingFixedDynamics(1.0), _CountingFixedDynamics(2.0)]
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool(models),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=belief.device, dtype=belief.dtype)
    controller.schedule(belief)
    at_due_time = belief.replace(timestamp=torch.tensor([0.1]))
    measured = SimpleNamespace(
        timestamp=torch.tensor([0.1]),
        measurement_mask=torch.tensor([[True]]),
        auxiliary={"world_position": torch.tensor([[[0.0, 0.0, 0.0]]])},
    )
    association = SimpleNamespace(
        pair_mask=torch.tensor([[True]]),
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
    )
    controller.assimilate_observation(at_due_time, measured, association)
    before = [model.calls for model in models]
    forecast = controller.predict(at_due_time, [0.1])
    assert forecast is not None
    assert [model.calls - prior for model, prior in zip(models, before, strict=True)] == [1, 0, 0]
    assert forecast.auxiliary["hypothesis_rollout_candidate_indices"].tolist() == [0]


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
    damped = ConstantVelocityDynamics(damping=2.0).predict_step(source, torch.tensor([2.0]))
    assert damped.belief.objects.velocity[0, 0, 0].item() == pytest.approx(2.0 * torch.exp(torch.tensor(-4.0)).item())
    expected_displacement = 2.0 * (1.0 - torch.exp(torch.tensor(-4.0)).item()) / 2.0
    assert damped.belief.objects.position[0, 0, 0].item() == pytest.approx(1.0 + expected_displacement)
    trajectory = HypothesisRolloutEngine().rollout_dynamics(
        [ConstantVelocityDynamics()], source, [0.25, 0.5]
    )[0]
    assert trajectory.fast_log_variance.shape[:3] == (1, 2, 1)


def test_ballistic_contact_hypothesis_predicts_gravity_and_ground_event() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.position[0, 0] = torch.tensor([0.0, 0.2, 0.0])
    objects.velocity[0, 0, 1] = -1.0
    objects.geometry[0, 0, 0] = 0.1
    source = belief.replace(objects=objects, gravity=torch.tensor([[0.0, -9.81, 0.0]]))
    step = BallisticContactDynamics().predict_step(source, torch.tensor([0.1]))
    assert step.belief.objects.position[0, 0, 1] == pytest.approx(0.1)
    assert step.belief.objects.velocity[0, 0, 1] > 0
    assert step.event_logits[0, 0, 3] > 0
    torch.testing.assert_close(source.objects.position[0, 0], torch.tensor([0.0, 0.2, 0.0]))


def test_ballistic_contact_hypothesis_resolves_approaching_pair() -> None:
    belief = BeliefFactory(max_objects=2).create()
    objects = belief.objects.clone()
    objects.active[0, :2] = True
    objects.position[0, 0] = torch.tensor([-0.11, 0.2, 0.0])
    objects.position[0, 1] = torch.tensor([0.11, 0.2, 0.0])
    objects.velocity[0, 0, 0] = 1.0
    objects.velocity[0, 1, 0] = -1.0
    objects.geometry[0, :2, 0] = 0.1
    source = belief.replace(objects=objects, gravity=torch.zeros(1, 3))
    step = BallisticContactDynamics().predict_step(source, torch.tensor([0.1]))
    assert step.event_logits[0, 0, 3] > 0
    assert step.event_logits[0, 1, 3] > 0
    assert step.belief.objects.velocity[0, 0, 0] < 0
    assert step.belief.objects.velocity[0, 1, 0] > 0
