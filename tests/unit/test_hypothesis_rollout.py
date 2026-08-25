from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory, MotionMode
from world_model.dynamics import (
    BallisticContactDynamics,
    ConstantVelocityDynamics,
    DynamicsModel,
    HypothesisDynamicsPool,
    HypothesisRegime,
    HypothesisRolloutEngine,
    OnlineLocalAccelerationDynamics,
    RolloutStep,
    RuntimeHypothesisController,
)
from world_model.runtime.prepared import tensor_identity_version_signature


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


def _repeat_batch(trajectory: BeliefTrajectory, batch_size: int) -> BeliefTrajectory:
    def repeat(value):
        if value is None:
            return None
        return value.expand(batch_size, *value.shape[1:]).clone()

    return BeliefTrajectory(
        timestamps=repeat(trajectory.timestamps),
        positions=repeat(trajectory.positions),
        velocities=repeat(trajectory.velocities),
        orientations=repeat(trajectory.orientations),
        motion_mode_logits=repeat(trajectory.motion_mode_logits),
        fast_log_variance=repeat(trajectory.fast_log_variance),
        active_mask=repeat(trajectory.active_mask),
        event_logits=repeat(trajectory.event_logits),
        auxiliary={name: repeat(value) for name, value in trajectory.auxiliary.items()},
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


def test_selector_explicit_all_axis_position_support_is_exact_legacy() -> None:
    target = torch.zeros(1, 2, 1, 3)
    target_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    trajectories = [_trajectory(0.5), _trajectory(0.0)]

    legacy = HypothesisRolloutEngine.score(
        trajectories,
        target,
        target_mask,
        uncertainty_aware=False,
    )
    explicit = HypothesisRolloutEngine.score(
        trajectories,
        target,
        target_mask,
        target_position_axis_mask=target_mask.unsqueeze(-1).expand_as(target),
        uncertainty_aware=False,
    )

    for name in (
        "scores",
        "selected_index",
        "posterior_weights",
        "axis_scores",
        "entity_axis_scores",
        "evidence_mask",
        "axis_evidence_mask",
        "entity_axis_evidence_mask",
    ):
        legacy_value = getattr(legacy, name)
        explicit_value = getattr(explicit, name)
        assert isinstance(legacy_value, torch.Tensor)
        assert isinstance(explicit_value, torch.Tensor)
        assert torch.equal(legacy_value, explicit_value), name


def test_selector_position_axis_support_updates_only_supported_axes() -> None:
    target = torch.zeros(1, 2, 1, 3)
    target_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    position_axis_mask = torch.zeros_like(target, dtype=torch.bool)
    position_axis_mask[..., 0] = True
    x_winner = _trajectory(0.0)
    other_axis_winner = _trajectory(0.0)
    x_winner.positions[..., 1:] = 5.0
    other_axis_winner.positions[..., 0] = 1.0

    selection = HypothesisRolloutEngine.score(
        [x_winner, other_axis_winner],
        target,
        target_mask,
        target_position_axis_mask=position_axis_mask,
        uncertainty_aware=False,
    )

    assert selection.selected_index.tolist() == [0]
    assert selection.axis_evidence_mask is not None
    assert selection.axis_evidence_mask.tolist() == [[True, False, False]]
    assert selection.entity_axis_evidence_mask is not None
    assert selection.entity_axis_evidence_mask[0, 0].tolist() == [True, False, False]
    assert selection.axis_scores is not None
    assert selection.axis_scores[0, 0, 0] < selection.axis_scores[0, 0, 1]
    assert selection.axis_scores[0, 1:].eq(0).all()


def test_selector_position_axis_support_rejects_axes_outside_target_mask() -> None:
    target = torch.zeros(1, 2, 1, 3)
    target_mask = torch.zeros(1, 2, 1, dtype=torch.bool)
    position_axis_mask = torch.zeros_like(target, dtype=torch.bool)
    position_axis_mask[..., 0] = True

    with pytest.raises(ValueError, match="subset of target_mask"):
        HypothesisRolloutEngine.score(
            [_trajectory(0.0)],
            target,
            target_mask,
            target_position_axis_mask=position_axis_mask,
            uncertainty_aware=False,
        )


def test_selector_requires_axis_valid_rgb_velocity_before_replacing_velocity() -> None:
    target_position = torch.zeros(1, 2, 1, 3)
    target_velocity = torch.zeros_like(target_position)
    position_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    velocity_axis_mask = torch.zeros_like(target_position, dtype=torch.bool)
    velocity_axis_mask[..., 0] = True
    position_only_winner = _trajectory(0.0)
    velocity_winner = _trajectory(0.1)
    position_only_winner.velocities[..., 0] = 1.0

    selection = HypothesisRolloutEngine.score(
        [position_only_winner, velocity_winner],
        target_position,
        position_mask,
        target_velocities=target_velocity,
        target_velocity_axis_mask=velocity_axis_mask,
        target_velocity_log_variance=torch.zeros_like(target_velocity),
        velocity_weight=1.0,
        uncertainty_aware=False,
    )

    assert selection.entity_axis_scores is not None
    assert selection.entity_axis_scores.argmin(dim=-1)[0, 0, 0].item() == 1
    assert selection.entity_axis_evidence_mask is not None
    assert selection.entity_axis_evidence_mask[0, 0].tolist() == [True, False, False]


def test_selector_velocity_nonregression_gate_vetoes_position_winner() -> None:
    target_position = torch.zeros(1, 2, 1, 3)
    target_velocity = torch.zeros_like(target_position)
    position_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    velocity_axis_mask = torch.zeros_like(target_position, dtype=torch.bool)
    velocity_axis_mask[..., 0] = True
    learned = _trajectory(0.1)
    position_winner = _trajectory(0.0)
    position_winner.velocities[..., 0] = 2.0

    legacy = HypothesisRolloutEngine.score(
        [learned, position_winner],
        target_position,
        position_mask,
        uncertainty_aware=False,
    )
    gated = HypothesisRolloutEngine.score(
        [learned, position_winner],
        target_position,
        position_mask,
        target_velocities=target_velocity,
        target_velocity_axis_mask=velocity_axis_mask,
        target_velocity_log_variance=torch.zeros_like(target_velocity),
        velocity_nonregression_gate_enabled=True,
        uncertainty_aware=False,
    )

    assert legacy.entity_axis_scores is not None
    assert gated.entity_axis_scores is not None
    assert legacy.entity_axis_scores.argmin(dim=-1)[0, 0, 0].item() == 1
    assert gated.entity_axis_scores.argmin(dim=-1)[0, 0, 0].item() == 0


def test_selector_velocity_nonregression_gate_allows_nonworse_candidate() -> None:
    target_position = torch.zeros(1, 2, 1, 3)
    target_velocity = torch.zeros_like(target_position)
    position_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    velocity_axis_mask = torch.zeros_like(target_position, dtype=torch.bool)
    velocity_axis_mask[..., 0] = True

    selection = HypothesisRolloutEngine.score(
        [_trajectory(0.1), _trajectory(0.0)],
        target_position,
        position_mask,
        target_velocities=target_velocity,
        target_velocity_axis_mask=velocity_axis_mask,
        target_velocity_log_variance=torch.zeros_like(target_velocity),
        velocity_nonregression_gate_enabled=True,
        uncertainty_aware=False,
    )

    assert selection.entity_axis_scores is not None
    assert selection.entity_axis_scores.argmin(dim=-1)[0, 0, 0].item() == 1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_selector_velocity_nonregression_gate_mps_forward() -> None:
    target_position = torch.zeros(1, 2, 1, 3, device="mps")
    target_velocity = torch.zeros_like(target_position)
    velocity_axis_mask = torch.zeros_like(target_position, dtype=torch.bool)
    velocity_axis_mask[..., 0] = True
    learned = _trajectory(0.1).to("mps")
    candidate = _trajectory(0.0).to("mps")
    candidate.velocities[..., 0] = 2.0

    selection = HypothesisRolloutEngine.score(
        [learned, candidate],
        target_position,
        torch.ones(1, 2, 1, dtype=torch.bool, device="mps"),
        target_velocities=target_velocity,
        target_velocity_axis_mask=velocity_axis_mask,
        target_velocity_log_variance=torch.zeros_like(target_velocity),
        velocity_nonregression_gate_enabled=True,
        uncertainty_aware=False,
    )

    assert selection.entity_axis_scores is not None
    assert selection.entity_axis_scores.argmin(dim=-1)[0, 0, 0].item() == 0


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
    selection = HypothesisRolloutEngine.score([narrow, wide], target, mask, uncertainty_aware=True)
    # The log-variance penalty prevents a deliberately broad forecast from
    # winning just because its normalized residual is smaller.
    assert selection.selected_index.item() == 0


def test_selector_combines_rgb_and_predictive_position_variance() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    narrow = _trajectory(0.5, variance=-4.6051702)
    broad = _trajectory(1.0, variance=0.0)

    prediction_only = HypothesisRolloutEngine.score(
        [narrow, broad],
        target,
        mask,
    )
    noisy_rgb = HypothesisRolloutEngine.score(
        [narrow, broad],
        target,
        mask,
        target_position_log_variance=torch.full_like(target, 4.6051702),
    )

    assert prediction_only.selected_index.tolist() == [1]
    assert noisy_rgb.selected_index.tolist() == [0]


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
    expected_error = engine.score_ensemble(samples, target, mask, uncertainty_aware=False)
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
            event_logits=objects.motion_mode_logits.new_zeros(
                belief.batch_size,
                objects.max_objects,
                objects.motion_mode_logits.shape[-1],
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


class _RecordingConstantVelocity(ConstantVelocityDynamics):
    def __init__(self) -> None:
        super().__init__()
        self.elapsed: list[float] = []

    def predict_step(self, belief, delta_time):
        self.elapsed.append(float(delta_time[0]))
        return super().predict_step(belief, delta_time)


class _RecordingUnsafeDynamics(_FixedDynamics):
    def __init__(self) -> None:
        super().__init__(0.0)
        self.elapsed: list[float] = []
        self.shared_horizon_rollout_safe = False

    def predict_step(self, belief, delta_time):
        self.elapsed.append(float(delta_time[0]))
        return super().predict_step(belief, delta_time)


class _IncrementXStateDynamics:
    def __init__(self, increment: float, *, collision: bool = False) -> None:
        self.increment = increment
        self.collision = collision

    def predict_step(self, belief, delta_time):
        objects = belief.objects.clone()
        objects.position[..., 0] = objects.position[..., 0] + self.increment
        objects.velocity[..., 0] = self.increment / delta_time[:, None].clamp_min(1.0e-6)
        objects.fast_log_variance[..., 0] = objects.fast_log_variance[..., 0] + 0.1
        objects.fast_log_variance[..., 3] = objects.fast_log_variance[..., 3] + 0.1
        event_logits = torch.full_like(objects.motion_mode_logits, -4.0)
        if self.collision:
            event_logits[..., 3] = 5.0
        return RolloutStep(
            belief=belief.replace(timestamp=belief.timestamp + delta_time, objects=objects),
            event_logits=event_logits,
            auxiliary={},
        )


def test_pool_assimilates_late_evidence_and_updates_selected_model() -> None:
    belief = BeliefFactory(max_objects=1).create()
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(belief, [0.1, 0.2])
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    selection = pool.assimilate(
        belief, target, mask, trajectories=trajectories, uncertainty_aware=False
    )
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
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)], evidence_decay=0.1)
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


def test_pool_preserves_axis_evidence_for_batch_row_without_new_match() -> None:
    belief = BeliefFactory(max_objects=1).create(batch_size=2)
    pool = HypothesisDynamicsPool(
        [_FixedDynamics(1.0), _FixedDynamics(0.0)],
        temperature=1.0,
    )
    trajectories = [
        _repeat_batch(_trajectory(1.0), 2),
        _repeat_batch(_trajectory(0.0), 2),
    ]
    mask = torch.ones(2, 2, 1, dtype=torch.bool)
    pool.assimilate(
        belief,
        torch.zeros(2, 2, 1, 3),
        mask,
        trajectories=trajectories,
        uncertainty_aware=False,
    )
    assert pool.axis_log_weights is not None
    previous = pool.axis_log_weights[1].clone()

    second_target = torch.ones(2, 2, 1, 3)
    second_mask = mask.clone()
    second_mask[1] = False
    pool.assimilate(
        belief,
        second_target,
        second_mask,
        trajectories=trajectories,
        uncertainty_aware=False,
    )

    torch.testing.assert_close(pool.axis_log_weights[1], previous)
    assert pool.selected_axis_index(belief)[1].tolist() == [1, 1, 1]


def test_pool_resets_entity_evidence_when_lifecycle_reuses_slot() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(source, [0.1])
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
    )
    assert pool.selected_entity_axis_index(source)[0, 0].tolist() == [1, 0, 0]

    reused_objects = source.objects.clone()
    reused_objects.object_id[0, 0] = 8
    reused = source.replace(objects=reused_objects)

    assert pool.selected_entity_axis_index(reused)[0, 0].tolist() == [0, 0, 0]


def test_pool_regime_evidence_is_local_and_exposes_support_state() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(source, [0.1])
    trajectories[0].fast_log_variance[..., :3] = torch.log(torch.tensor(4.0))
    trajectories[1].fast_log_variance[..., :3] = torch.log(torch.tensor(0.25))
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
        entity_regime=torch.tensor([[int(HypothesisRegime.FREE)]], dtype=torch.int64),
        evidence_timestamp=torch.tensor([0.1]),
        entity_axis_observability=torch.full((1, 1, 3), 0.8),
    )

    free = pool.selected_entity_axis_applicability(
        source,
        entity_regime=torch.tensor([[int(HypothesisRegime.FREE)]], dtype=torch.int64),
        current_timestamp=torch.tensor([0.2]),
        minimum_support_count=1,
        maximum_age_seconds=0.2,
        minimum_observability=0.5,
        minimum_confidence_margin=0.0,
    )
    collision = pool.selected_entity_axis_applicability(
        source,
        entity_regime=torch.tensor([[int(HypothesisRegime.COLLISION)]], dtype=torch.int64),
        current_timestamp=torch.tensor([0.2]),
        minimum_support_count=1,
        maximum_age_seconds=0.2,
        minimum_observability=0.5,
        minimum_confidence_margin=0.0,
    )

    assert free.selected_index[0, 0].tolist() == [1, 0, 0]
    assert free.supported[0, 0].tolist() == [True, True, True]
    assert free.support_count[0, 0].tolist() == [1, 1, 1]
    assert free.age_seconds[0, 0].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert free.observability[0, 0].tolist() == pytest.approx([0.8, 0.8, 0.8])
    assert free.predictive_variance[0, 0].tolist() == pytest.approx([0.25, 4.0, 4.0])
    assert collision.selected_index[0, 0].tolist() == [0, 0, 0]
    assert not collision.supported.any()
    assert not collision.support_count.any()
    assert not collision.predictive_variance.any()


def test_pool_regime_position_residual_is_local_and_resets_with_lifecycle() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(source, [0.1])
    regime = torch.tensor([[int(HypothesisRegime.FREE)]], dtype=torch.int64)
    residual = torch.tensor([[[-0.75, 0.25, 9.0]]])
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
        entity_regime=regime,
        evidence_timestamp=torch.tensor([0.1]),
        entity_axis_observability=torch.ones(1, 1, 3),
        learned_position_residual=residual,
    )
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
        entity_regime=regime,
        evidence_timestamp=torch.tensor([0.2]),
        entity_axis_observability=torch.ones(1, 1, 3),
        learned_position_residual=residual,
    )

    applicability = pool.selected_entity_axis_applicability(
        source,
        entity_regime=regime,
        current_timestamp=torch.tensor([0.1]),
        minimum_support_count=1,
        maximum_age_seconds=1.0,
        minimum_observability=0.0,
        minimum_confidence_margin=0.0,
    )
    torch.testing.assert_close(applicability.position_residual, residual)
    assert applicability.position_residual_supported.all()

    reused_objects = source.objects.clone()
    reused_objects.object_id[0, 0] = 8
    reused = source.replace(objects=reused_objects)
    reset = pool.selected_entity_axis_applicability(
        reused,
        entity_regime=regime,
        current_timestamp=torch.tensor([0.1]),
        minimum_support_count=1,
        maximum_age_seconds=1.0,
        minimum_observability=0.0,
        minimum_confidence_margin=0.0,
    )
    assert not reset.supported.any()
    assert not reset.position_residual.any()


@pytest.mark.parametrize(
    ("minimum_support", "maximum_age", "minimum_observability", "minimum_margin"),
    [
        (2, 1.0, 0.0, 0.0),
        (1, 0.05, 0.0, 0.0),
        (1, 1.0, 0.9, 0.0),
        (1, 1.0, 0.0, 0.99),
    ],
)
def test_pool_regime_applicability_falls_back_when_any_gate_fails(
    minimum_support: int,
    maximum_age: float,
    minimum_observability: float,
    minimum_margin: float,
) -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(source, [0.1])
    regime = torch.tensor([[int(HypothesisRegime.FREE)]], dtype=torch.int64)
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
        entity_regime=regime,
        evidence_timestamp=torch.tensor([0.1]),
        entity_axis_observability=torch.full((1, 1, 3), 0.8),
    )

    applicability = pool.selected_entity_axis_applicability(
        source,
        entity_regime=regime,
        current_timestamp=torch.tensor([0.2]),
        minimum_support_count=minimum_support,
        maximum_age_seconds=maximum_age,
        minimum_observability=minimum_observability,
        minimum_confidence_margin=minimum_margin,
    )

    assert applicability.selected_index[0, 0].tolist() == [0, 0, 0]
    assert not applicability.supported.any()


def test_pool_regime_capability_mask_forces_learned_fallback() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(source, [0.1])
    regime = torch.tensor([[int(HypothesisRegime.COLLISION)]], dtype=torch.int64)
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
        entity_regime=regime,
        evidence_timestamp=torch.tensor([0.1]),
        entity_axis_observability=torch.ones(1, 1, 3),
    )
    capability = torch.zeros(6, 2, dtype=torch.bool)
    capability[:, 0] = True
    capability[int(HypothesisRegime.FREE), 1] = True

    applicability = pool.selected_entity_axis_applicability(
        source,
        entity_regime=regime,
        current_timestamp=torch.tensor([0.1]),
        minimum_support_count=1,
        maximum_age_seconds=1.0,
        minimum_observability=0.0,
        minimum_confidence_margin=0.0,
        candidate_regime_mask=capability,
    )

    assert applicability.supported.all()
    assert applicability.selected_index[0, 0].tolist() == [0, 0, 0]


def test_pool_regime_robust_influence_bounds_one_observation() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([_FixedDynamics(0.0), _FixedDynamics(100.0)])
    trajectories = pool.rollout(source, [0.1])
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
        entity_regime=torch.tensor([[int(HypothesisRegime.FREE)]], dtype=torch.int64),
        evidence_timestamp=torch.tensor([0.1]),
        entity_axis_observability=torch.ones(1, 1, 3),
        robust_influence_delta=0.25,
    )

    assert pool.entity_axis_regime_log_weights is not None
    cell = pool.entity_axis_regime_log_weights[0, 0, 0, int(HypothesisRegime.FREE)]
    assert abs(float(cell[0] - cell[1])) == pytest.approx(0.25)


def test_pool_lifecycle_reuse_clears_regime_support_and_freshness() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = pool.rollout(source, [0.1])
    pool.assimilate(
        source,
        torch.zeros_like(trajectories[0].positions),
        torch.ones_like(trajectories[0].active_mask),
        trajectories=trajectories,
        uncertainty_aware=False,
        entity_regime=torch.tensor([[int(HypothesisRegime.FREE)]], dtype=torch.int64),
        evidence_timestamp=torch.tensor([0.1]),
        entity_axis_observability=torch.ones(1, 1, 3),
    )
    reused_objects = source.objects.clone()
    reused_objects.object_id[0, 0] = 8
    reused = source.replace(objects=reused_objects)

    applicability = pool.selected_entity_axis_applicability(
        reused,
        entity_regime=torch.tensor([[int(HypothesisRegime.FREE)]], dtype=torch.int64),
        current_timestamp=torch.tensor([0.2]),
        minimum_support_count=1,
        maximum_age_seconds=1.0,
        minimum_observability=0.0,
        minimum_confidence_margin=0.0,
    )

    assert not applicability.supported.any()
    assert not applicability.support_count.any()
    assert not applicability.observability.any()
    assert not applicability.predictive_variance.any()


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
    assert forecast.auxiliary["hypothesis_axis_index"].shape == (1, 1, 1, 3)
    torch.testing.assert_close(source.objects.position, belief.objects.position)


def test_runtime_controller_source_bound_missing_axis_provenance_abstains() -> None:
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
    due = source.replace(timestamp=torch.tensor([0.1]))
    measured = SimpleNamespace(
        timestamp=due.timestamp,
        measurement_mask=torch.tensor([[True]]),
        auxiliary={"world_position": torch.zeros(1, 1, 3)},
        source_belief_indices=torch.tensor([[0]], dtype=torch.int64),
        source_object_ids=torch.tensor([[7]], dtype=torch.int64),
    )
    association = SimpleNamespace(
        pair_mask=torch.tensor([[True]]),
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
    )

    selection = controller.assimilate_observation(due, measured, association)

    assert selection is None
    assert controller.pool.evidence_seen is not None
    assert not controller.pool.evidence_seen.any()
    assert controller.pool.entity_axis_evidence_seen is not None
    assert not controller.pool.entity_axis_evidence_seen.any()


def test_runtime_controller_source_bound_position_evidence_uses_declared_axes() -> None:
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
    due = source.replace(timestamp=torch.tensor([0.1]))
    measured = SimpleNamespace(
        timestamp=due.timestamp,
        measurement_mask=torch.tensor([[True]]),
        auxiliary={
            "world_position": torch.zeros(1, 1, 3),
            "world_position_independent_axis_mask": torch.tensor([[[True, False, False]]]),
        },
        source_belief_indices=torch.tensor([[0]], dtype=torch.int64),
        source_object_ids=torch.tensor([[7]], dtype=torch.int64),
    )
    association = SimpleNamespace(
        pair_mask=torch.tensor([[True]]),
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
    )

    selection = controller.assimilate_observation(due, measured, association)

    assert selection is not None
    assert selection.entity_axis_evidence_mask is not None
    assert selection.entity_axis_evidence_mask[0, 0].tolist() == [True, False, False]
    assert controller.pool.entity_axis_evidence_seen is not None
    assert controller.pool.entity_axis_evidence_seen[0, 0].tolist() == [True, False, False]


def test_runtime_controller_shares_only_semigroup_safe_horizon_rollouts() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.velocity[0, 0, 0] = 1.0
    source = belief.replace(objects=objects)
    safe = _RecordingConstantVelocity()
    unsafe = _RecordingUnsafeDynamics()
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([safe, unsafe]),
        evidence_horizons_seconds=(0.1, 0.25, 0.5),
        axis_independent_axes=(0,),
        shared_horizon_rollout_enabled=True,
    )
    controller.reset(1, device=source.device, dtype=source.dtype)

    controller.schedule(source)

    assert safe.elapsed == pytest.approx([0.1, 0.15, 0.25])
    assert unsafe.elapsed == pytest.approx([0.1, 0.25, 0.5])
    assert [float(item.trajectories[0].positions[0, 0, 0, 0]) for item in controller.pending] == (
        pytest.approx([0.1, 0.25, 0.5])
    )


def test_shared_horizon_reconstruction_matches_stride_one_dynamics() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    objects.position[0, 0] = torch.tensor([0.0, 1.0, 0.0])
    objects.velocity[0, 0] = torch.tensor([0.4, 0.1, -0.2])
    source = belief.replace(objects=objects)
    model = DynamicsModel.from_belief(
        source,
        max_substep=0.05,
        learned_effect_interval_seconds=None,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    horizons = (0.05, 0.1)
    current = source.clone()
    segments = []
    previous = 0.0
    for horizon in horizons:
        segment = model.predict_step(current, torch.tensor([horizon - previous]))
        segments.append(segment)
        current = segment.belief
        previous = horizon

    assert model.shared_horizon_rollout_safe
    for horizon_index, horizon in enumerate(horizons):
        reconstructed = RuntimeHypothesisController._prefix_step_from_segments(
            segments,
            horizon_index,
        )
        direct = model.predict_step(source.clone(), torch.tensor([horizon]))
        torch.testing.assert_close(
            reconstructed.belief.objects.position,
            direct.belief.objects.position,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            reconstructed.belief.objects.velocity,
            direct.belief.objects.velocity,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            reconstructed.belief.objects.fast_log_variance,
            direct.belief.objects.fast_log_variance,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            reconstructed.belief.objects.modal_state,
            direct.belief.objects.modal_state,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            reconstructed.event_logits,
            direct.event_logits,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        assert reconstructed.auxiliary.keys() == direct.auxiliary.keys()
        for name in direct.auxiliary:
            torch.testing.assert_close(
                reconstructed.auxiliary[name],
                direct.auxiliary[name],
                rtol=1.0e-6,
                atol=1.0e-6,
            )

    multirate = DynamicsModel.from_belief(
        source,
        max_substep=0.05,
        learned_effect_interval_seconds=0.1,
    )
    assert not multirate.shared_horizon_rollout_safe


def test_shared_horizon_reconstruction_preserves_interval_event_semantics() -> None:
    source = BeliefFactory(max_objects=1).create()
    first_belief = source.clone()
    first_belief.objects.modal_state.fill_(1.0)
    second_belief = source.clone()
    second_belief.objects.modal_state.fill_(2.0)
    first_event = source.objects.motion_mode_logits.new_full((1, 1, len(MotionMode)), -4.0)
    second_event = first_event.clone()
    first_event[..., MotionMode.COLLISION] = 5.0
    second_event[..., MotionMode.COLLISION] = -3.0
    first = RolloutStep(
        belief=first_belief,
        event_logits=first_event,
        auxiliary={
            "interval_pair_contact": torch.tensor([[[True]]]),
            "pair_collision": torch.tensor([[[True]]]),
            "pair_impulse": torch.tensor([[[2.0]]]),
            "pair_event_logits": torch.tensor([[[[1.0, 7.0]]]]),
            "learned_effect_evaluation_count": torch.tensor([2]),
            "mean_penetration": torch.tensor([1.0]),
        },
    )
    second = RolloutStep(
        belief=second_belief,
        event_logits=second_event,
        auxiliary={
            "interval_pair_contact": torch.tensor([[[False]]]),
            "pair_collision": torch.tensor([[[False]]]),
            "pair_impulse": torch.tensor([[[1.0]]]),
            "pair_event_logits": torch.tensor([[[[3.0, 4.0]]]]),
            "learned_effect_evaluation_count": torch.tensor([3]),
            "mean_penetration": torch.tensor([2.0]),
        },
    )

    reconstructed = RuntimeHypothesisController._prefix_step_from_segments(
        (first, second),
        1,
    )

    assert reconstructed.event_logits[..., MotionMode.COLLISION].item() == pytest.approx(5.0)
    assert reconstructed.auxiliary["interval_pair_contact"].item()
    assert reconstructed.auxiliary["pair_collision"].item()
    assert reconstructed.auxiliary["pair_impulse"].item() == pytest.approx(2.0)
    assert reconstructed.auxiliary["pair_event_logits"].tolist() == [[[[3.0, 7.0]]]]
    assert reconstructed.auxiliary["learned_effect_evaluation_count"].item() == 5
    assert reconstructed.auxiliary["mean_penetration"].item() == pytest.approx(2.0)
    assert reconstructed.belief.objects.modal_state.eq(2.0).all()


def test_online_local_acceleration_is_support_gated_and_identity_bound() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    candidate = OnlineLocalAccelerationDynamics(
        minimum_support_count=1,
        maximum_acceleration=20.0,
    )

    unsupported = candidate.predict_step(source, torch.tensor([0.1]))
    baseline = ConstantVelocityDynamics().predict_step(source, torch.tensor([0.1]))
    torch.testing.assert_close(
        unsupported.belief.objects.position,
        baseline.belief.objects.position,
        rtol=0,
        atol=0,
    )
    assert not candidate.applicability_mask(source).any()

    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    log_variance = torch.zeros(1, 1, 3)
    candidate.assimilate_velocity_observation(
        source,
        torch.zeros(1, 1, 3),
        valid,
        log_variance,
        torch.tensor([0.0]),
    )
    accelerated_objects = source.objects.clone()
    accelerated_objects.velocity[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    accelerated = source.replace(
        timestamp=torch.tensor([0.1]),
        objects=accelerated_objects,
    )
    candidate.assimilate_velocity_observation(
        accelerated,
        accelerated.objects.velocity,
        valid,
        log_variance,
        accelerated.timestamp,
    )

    assert candidate.applicability_mask(accelerated).all()
    predicted = candidate.predict_step(accelerated, torch.tensor([0.1]))
    assert predicted.belief.objects.velocity[0, 0].tolist() == pytest.approx([2.0, 4.0, 5.0])
    assert predicted.belief.objects.position[0, 0].tolist() == pytest.approx([0.15, 0.3, 0.4])
    assert predicted.auxiliary["online_local_acceleration_supported"].all()

    reused_objects = accelerated.objects.clone()
    reused_objects.object_id[0, 0] = 8
    reused = accelerated.replace(objects=reused_objects)
    assert not candidate.applicability_mask(reused).any()


def test_online_local_acceleration_resets_across_nonfree_motion() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    belief = belief.replace(objects=objects)
    candidate = OnlineLocalAccelerationDynamics(minimum_support_count=1)
    valid = torch.ones_like(belief.objects.velocity, dtype=torch.bool)
    log_variance = torch.zeros_like(belief.objects.velocity)
    candidate.assimilate_velocity_observation(
        belief,
        torch.zeros_like(belief.objects.velocity),
        valid,
        log_variance,
        belief.timestamp,
    )
    candidate.assimilate_velocity_observation(
        belief,
        torch.ones_like(belief.objects.velocity),
        valid,
        log_variance,
        belief.timestamp + 0.1,
    )
    assert candidate.applicability_mask(belief).all()

    contact = belief.clone()
    contact.objects.motion_mode_logits.zero_()
    contact.objects.motion_mode_logits[..., MotionMode.GROUND_CONTACT] = 5.0
    candidate.assimilate_velocity_observation(
        contact,
        torch.ones_like(contact.objects.velocity),
        valid,
        log_variance,
        contact.timestamp + 0.2,
    )
    assert not candidate.applicability_mask(contact).any()

    candidate.assimilate_velocity_observation(
        belief,
        torch.full_like(belief.objects.velocity, 2.0),
        valid,
        log_variance,
        belief.timestamp + 0.3,
    )
    assert not candidate.applicability_mask(belief).any()
    assert candidate.support_count is not None
    assert not candidate.support_count.any()


def test_runtime_controller_updates_online_candidate_after_scoring_due_forecast() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    online = OnlineLocalAccelerationDynamics(minimum_support_count=1)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), online]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    association = SimpleNamespace(
        pair_mask=torch.tensor([[True]]),
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
    )

    controller.assimilate_observation(
        source,
        SimpleNamespace(
            timestamp=torch.tensor([0.0]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={
                "world_position": torch.zeros(1, 1, 3),
                "world_velocity": torch.zeros(1, 1, 3),
                "world_velocity_log_variance": torch.zeros(1, 1, 3),
                "world_velocity_valid_mask": torch.tensor([[True]]),
                "world_velocity_axis_valid_mask": torch.ones(1, 1, 3, dtype=torch.bool),
            },
        ),
        association,
    )
    controller.schedule(source)
    due_objects = source.objects.clone()
    due_objects.velocity[0, 0, 0] = 1.0
    due = source.replace(timestamp=torch.tensor([0.1]), objects=due_objects)
    selection = controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={
                "world_position": torch.zeros(1, 1, 3),
                "world_velocity": due.objects.velocity.clone(),
                "world_velocity_log_variance": torch.zeros(1, 1, 3),
                "world_velocity_valid_mask": torch.tensor([[True]]),
                "world_velocity_axis_valid_mask": torch.ones(1, 1, 3, dtype=torch.bool),
            },
        ),
        association,
    )

    assert selection is not None
    assert online.applicability_mask(due)[0, 0, 0]
    forecast = controller.predict(due, [0.1])
    assert forecast is not None
    assert forecast.auxiliary["hypothesis_axis_index"][0, 0, 0, 0].item() == 1
    assert forecast.positions[0, 0, 0, 0].item() == pytest.approx(0.15)
    assert forecast.velocities[0, 0, 0, 0].item() == pytest.approx(2.0)


def test_runtime_controller_keeps_exact_learned_fallback_without_online_support() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    online = OnlineLocalAccelerationDynamics(minimum_support_count=2)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), online]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    selection = controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=due.timestamp,
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.zeros(1, 1, 3)},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )

    assert selection is not None
    assert selection.entity_axis_scores is not None
    assert selection.entity_axis_scores.argmin(dim=-1)[0, 0, 0].item() == 1
    assert not online.applicability_mask(due).any()
    forecast = controller.predict(due, [0.1])
    assert forecast is not None
    expected = _FixedDynamics(1.0).predict_step(due, torch.tensor([0.1]))
    torch.testing.assert_close(
        forecast.positions[:, 0],
        expected.belief.objects.position,
        rtol=0,
        atol=0,
    )
    assert forecast.auxiliary["hypothesis_axis_index"][0, 0, 0, 0].item() == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_evidence_age_seconds", True),
        ("minimum_observability", "0.5"),
        ("minimum_confidence_margin", None),
        ("velocity_evidence_weight", True),
        ("velocity_nonregression_gate_enabled", 1),
        ("robust_influence_delta", False),
    ),
)
def test_runtime_controller_rejects_non_numeric_local_thresholds(
    field: str,
    value: object,
) -> None:
    arguments = {field: value}
    with pytest.raises(ValueError, match=field):
        RuntimeHypothesisController(
            HypothesisDynamicsPool([_FixedDynamics(0.0), _FixedDynamics(1.0)]),
            evidence_horizons_seconds=(0.1,),
            axis_independent_axes=(0,),
            **arguments,
        )


def test_runtime_controller_local_applicability_reports_exact_cell_evidence() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
        minimum_support_count=1,
        maximum_evidence_age_seconds=0.5,
        minimum_observability=0.5,
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={
                "world_position": torch.tensor([[[0.0, 0.0, 0.0]]]),
                "world_position_log_variance": torch.full((1, 1, 3), -4.0),
            },
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )
    forecast = controller.predict(due, [0.1])

    assert forecast is not None
    assert forecast.positions[0, 0, 0, 0] == pytest.approx(0.0)
    assert forecast.auxiliary["hypothesis_axis_index"][0, 0, 0, 0].item() == 1
    assert forecast.auxiliary["hypothesis_axis_supported"][0, 0, 0, 0].item()
    assert forecast.auxiliary["hypothesis_axis_support_count"][0, 0, 0, 0].item() == 1
    assert forecast.auxiliary["hypothesis_axis_evidence_age_seconds"][
        0, 0, 0, 0
    ].item() == pytest.approx(0.0)
    assert forecast.auxiliary["hypothesis_axis_observability"][0, 0, 0, 0].item() > 0.98
    assert forecast.auxiliary["hypothesis_axis_predictive_variance"][0, 0, 0, 0].item() > 0.0
    assert forecast.auxiliary["hypothesis_axis_confidence_margin"][0, 0, 0, 0].item() > 0.0
    assert forecast.auxiliary["hypothesis_interaction_regime"][0, 0, 0].item() == int(
        HypothesisRegime.FREE
    )


def test_runtime_controller_residual_correction_prefers_learned_and_is_axis_local() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0, 1),
        local_applicability_enabled=True,
        residual_correction_gain_by_axis=(0.25, 0.5, 0.0),
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.tensor([[[0.0, 0.5, 7.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )
    controller.schedule(due)
    due = source.replace(timestamp=torch.tensor([0.2]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.2]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.tensor([[[0.0, 0.5, 7.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )

    forecast = controller.predict(due, [0.1])

    assert forecast is not None
    assert forecast.positions[0, 0, 0].tolist() == pytest.approx([0.75, 0.25, 0.0])
    assert forecast.auxiliary["hypothesis_axis_index"][0, 0, 0].tolist() == [0, 0, 0]
    assert forecast.auxiliary["hypothesis_position_residual"][0, 0, 0].tolist() == pytest.approx(
        [-1.0, 0.5, 0.0]
    )
    assert forecast.auxiliary["hypothesis_position_residual_applied"][0, 0, 0].tolist() == [
        True,
        True,
        False,
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_runtime_controller_residual_correction_mps_is_finite() -> None:
    belief = BeliefFactory(max_objects=1).create(device="mps")
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
        residual_correction_gain_by_axis=(0.25, 0.0, 0.0),
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1], device="mps"))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1], device="mps"),
            measurement_mask=torch.tensor([[True]], device="mps"),
            auxiliary={"world_position": torch.zeros(1, 1, 3, device="mps")},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]], device="mps"),
            belief_indices=torch.tensor([[0]], dtype=torch.int64, device="mps"),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64, device="mps"),
        ),
    )
    controller.schedule(due)
    due = source.replace(timestamp=torch.tensor([0.2], device="mps"))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.2], device="mps"),
            measurement_mask=torch.tensor([[True]], device="mps"),
            auxiliary={"world_position": torch.zeros(1, 1, 3, device="mps")},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]], device="mps"),
            belief_indices=torch.tensor([[0]], dtype=torch.int64, device="mps"),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64, device="mps"),
        ),
    )

    forecast = controller.predict(due, [0.1])

    assert forecast is not None
    assert torch.isfinite(forecast.positions).all()
    assert forecast.positions[0, 0, 0, 0].item() == pytest.approx(0.75)


def test_runtime_controller_composed_residual_is_output_only_not_recursive() -> None:
    belief = BeliefFactory(max_objects=1).create(batch_size=2)
    objects = belief.objects.clone()
    objects.active[:, 0] = True
    objects.object_id[:, 0] = torch.tensor([7, 8])
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_IncrementXStateDynamics(1.0), _IncrementXStateDynamics(0.25)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
        residual_correction_gain_by_axis=(0.5, 0.0, 0.0),
        composition_step_seconds=0.1,
    )
    controller.reset(2, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1, 0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1, 0.1]),
            measurement_mask=torch.tensor([[True], [True]]),
            auxiliary={"world_position": torch.tensor([[[0.25, 0.0, 0.0]], [[0.25, 0.0, 0.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True], [True]]),
            belief_indices=torch.tensor([[0], [0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0], [0]], dtype=torch.int64),
        ),
    )
    controller.schedule(due)
    due = source.replace(timestamp=torch.tensor([0.2, 0.2]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.2, 0.2]),
            measurement_mask=torch.tensor([[True], [True]]),
            auxiliary={"world_position": torch.tensor([[[0.25, 0.0, 0.0]], [[0.25, 0.0, 0.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True], [True]]),
            belief_indices=torch.tensor([[0], [0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0], [0]], dtype=torch.int64),
        ),
    )

    forecast = controller.predict(due, [0.1, 0.3, 0.4])

    assert forecast is not None
    canonical = controller.pool.rollout_engine.rollout_dynamics(
        (controller.pool.dynamics_models[0],), due, [0.1, 0.3, 0.4]
    )[0]
    expected_positions = canonical.positions.clone()
    expected_positions[..., 0] -= 0.375
    assert torch.equal(forecast.positions, expected_positions)
    assert torch.equal(forecast.velocities, canonical.velocities)
    assert torch.equal(forecast.fast_log_variance, canonical.fast_log_variance)
    assert torch.equal(forecast.event_logits, canonical.event_logits)
    assert not forecast.auxiliary["hypothesis_composed_candidate_step_count"][..., 1:].any()
    assert forecast.auxiliary["hypothesis_position_residual_applied"][0, :, 0, 0].tolist() == [
        True,
        True,
        True,
    ]


def test_runtime_controller_composes_selected_effect_in_bounded_steps() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_IncrementXStateDynamics(1.0), _IncrementXStateDynamics(0.25)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
        composition_step_seconds=0.1,
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.tensor([[[0.25, 0.0, 0.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )

    forecast = controller.predict(due, [0.1, 0.3])

    assert forecast is not None
    assert forecast.positions[0, :, 0, 0].tolist() == pytest.approx([0.25, 0.75])
    assert forecast.velocities[0, :, 0, 0].tolist() == pytest.approx([2.5, 2.5])
    assert forecast.fast_log_variance[0, 1, 0, 0] > forecast.fast_log_variance[0, 0, 0, 0]
    assert forecast.auxiliary["hypothesis_composed_candidate_step_count"][
        0, :, 0, 0, 1
    ].tolist() == [1, 2]
    assert forecast.auxiliary["hypothesis_composed_fallback_step_count"][0, :, 0, 0].tolist() == [
        0,
        0,
    ]
    candidate_counts = forecast.auxiliary["hypothesis_composed_candidate_step_count"]
    total_counts = forecast.auxiliary["hypothesis_composed_total_step_count"]
    regime_counts = forecast.auxiliary["hypothesis_composed_regime_step_count"]
    assert torch.equal(candidate_counts.sum(dim=-1), total_counts)
    assert torch.equal(regime_counts.sum(dim=-1), total_counts[..., 0])
    assert total_counts[0, :, 0, 0].tolist() == [1, 2]
    assert forecast.auxiliary["hypothesis_axis_predictive_variance"][
        0, :, 0, 0
    ].tolist() == pytest.approx([torch.exp(torch.tensor(0.1)).item()] * 2)
    assert forecast.auxiliary["hypothesis_axis_supported"][0, :, 0, 0].tolist() == [
        True,
        True,
    ]

    fallback = controller.predict(due, [0.15])
    assert fallback is not None
    assert fallback.positions[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert fallback.auxiliary["hypothesis_axis_index"][0, 0, 0, 0].item() == 0
    assert not fallback.auxiliary["hypothesis_axis_supported"].any()
    assert fallback.auxiliary["hypothesis_composition_grid_fallback"][0, 0, 0, 0]
    assert "hypothesis_composed_total_step_count" not in fallback.auxiliary


def test_runtime_controller_composition_is_exact_when_no_alternative_intervenes() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    learned = _IncrementXStateDynamics(1.0)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([learned, _IncrementXStateDynamics(0.25)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
        composition_step_seconds=0.1,
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.tensor([[[1.0, 0.0, 0.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )

    forecast = controller.predict(due, [0.1, 0.3])
    canonical = controller.pool.rollout_engine.rollout_dynamics((learned,), due, [0.1, 0.3])[0]

    assert forecast is not None
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
        assert torch.equal(getattr(forecast, name), getattr(canonical, name)), name
    assert not forecast.auxiliary["hypothesis_composed_candidate_step_count"][..., 1:].any()


def test_runtime_controller_composition_keeps_joint_collision_regime_learned() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool(
            [_IncrementXStateDynamics(1.0, collision=True), _IncrementXStateDynamics(0.0)]
        ),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
        local_applicability_enabled=True,
        composition_step_seconds=0.1,
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.tensor([[[0.0, 0.0, 0.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )

    forecast = controller.predict(due, [0.1])

    assert forecast is not None
    assert forecast.positions[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert forecast.event_logits[0, 0, 0, 3].item() == pytest.approx(5.0)
    assert forecast.auxiliary["hypothesis_axis_index"][0, 0, 0, 0].item() == 0
    assert forecast.auxiliary["hypothesis_interaction_regime"][0, 0, 0].item() == int(
        HypothesisRegime.COLLISION
    )


def test_runtime_controller_does_not_extrapolate_short_evidence_to_long_horizon() -> None:
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
    controller.assimilate_observation(at_due_time, measured, association)

    forecast = controller.predict(at_due_time, [0.1, 0.5])

    assert forecast is not None
    assert forecast.positions[0, 0, 0, 0] == pytest.approx(0.0)
    assert forecast.positions[0, 1, 0, 0] == pytest.approx(1.0)
    assert forecast.auxiliary["hypothesis_axis_index"][0, :, 0, 0].tolist() == [1, 0]
    assert forecast.auxiliary["hypothesis_axis_supported"][0, :, 0, 0].tolist() == [
        True,
        False,
    ]


def test_runtime_controller_keeps_independent_evidence_per_horizon() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)]),
        evidence_horizons_seconds=(0.1, 0.2),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    association = SimpleNamespace(
        pair_mask=torch.tensor([[True]]),
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
    )
    first_due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        first_due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.tensor([[[0.0, 0.0, 0.0]]])},
        ),
        association,
    )
    second_due = source.replace(timestamp=torch.tensor([0.2]))
    controller.assimilate_observation(
        second_due,
        SimpleNamespace(
            timestamp=torch.tensor([0.2]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.tensor([[[1.0, 0.0, 0.0]]])},
        ),
        association,
    )

    forecast = controller.predict(second_due, [0.1, 0.2])

    assert forecast is not None
    assert forecast.positions[0, :, 0, 0].tolist() == pytest.approx([0.0, 1.0])
    assert controller.pools[0].selected_axis_index(second_due)[0, 0].item() == 1
    assert controller.pools[1].selected_axis_index(second_due)[0, 0].item() == 0


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


def test_runtime_controller_invalidates_predictions_after_external_revision() -> None:
    belief = BeliefFactory(max_objects=1).create()
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(0.0), _FixedDynamics(1.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=belief.device, dtype=belief.dtype)
    controller.schedule(belief)

    controller.invalidate_pending()

    assert not controller.pending


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("source", "source_tensor_revision"),
        ("result", "scheduled_result_revision"),
    ],
)
def test_runtime_controller_rejects_mutated_pending_provenance(
    mutation: str,
    reason: str,
) -> None:
    belief = BeliefFactory(max_objects=1).create()
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(0.0), _FixedDynamics(1.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=belief.device, dtype=belief.dtype)
    controller.schedule(
        belief,
        source_revision=1,
        source_tensor_signature=tensor_identity_version_signature(belief),
        dynamics_tensor_signature=("dynamics",),
        dynamics_training=False,
        tensor_signature=tensor_identity_version_signature,
    )
    if mutation == "source":
        belief.objects.position.add_(0.25)
    else:
        controller.pending[0].learned_step.belief.objects.position.add_(0.25)

    controller.synchronize_runtime_context(
        belief,
        dynamics_tensor_signature=("dynamics",),
        dynamics_training=False,
        tensor_signature=tensor_identity_version_signature,
    )

    assert not controller.pending
    assert controller.pending_invalidation_counts[reason] == 1


def test_runtime_controller_resets_applicability_on_dynamics_revision() -> None:
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
    controller.synchronize_runtime_context(
        source,
        dynamics_tensor_signature=("revision", 1),
        dynamics_training=False,
        tensor_signature=tensor_identity_version_signature,
    )
    controller.schedule(
        source,
        source_revision=1,
        source_tensor_signature=tensor_identity_version_signature(source),
        dynamics_tensor_signature=("revision", 1),
        dynamics_training=False,
        tensor_signature=tensor_identity_version_signature,
    )
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={"world_position": torch.zeros(1, 1, 3)},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )
    assert controller.predict(due, [0.1]) is not None

    controller.synchronize_runtime_context(
        due,
        dynamics_tensor_signature=("revision", 2),
        dynamics_training=False,
        tensor_signature=tensor_identity_version_signature,
    )

    assert controller.predict(due, [0.1]) is None


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


class _FixedXStateDynamics(_FixedDynamics):
    def __init__(self, position: float, velocity: float, log_variance: float) -> None:
        super().__init__(position)
        self.velocity = velocity
        self.log_variance = log_variance

    def predict_step(self, belief, delta_time):
        step = super().predict_step(belief, delta_time)
        objects = step.belief.objects.clone()
        objects.velocity[..., 0] = self.velocity
        objects.fast_log_variance[..., 0] = self.log_variance
        objects.fast_log_variance[..., 3] = self.log_variance
        return RolloutStep(
            belief=step.belief.replace(objects=objects),
            event_logits=step.event_logits,
            auxiliary=step.auxiliary,
        )


def test_runtime_controller_splices_axis_state_but_keeps_learned_uncertainty() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool(
            [
                _FixedXStateDynamics(1.0, 10.0, 1.0),
                _FixedXStateDynamics(0.0, 2.0, -2.0),
            ]
        ),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True]]),
            auxiliary={
                "world_position": torch.tensor([[[0.0, 0.0, 0.0]]]),
                "world_position_log_variance": torch.full((1, 1, 3), -4.0),
            },
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True]]),
            belief_indices=torch.tensor([[0]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        ),
    )

    forecast = controller.predict(due, [0.1])

    assert forecast is not None
    assert forecast.positions[0, 0, 0, 0] == pytest.approx(0.0)
    assert forecast.velocities[0, 0, 0, 0] == pytest.approx(2.0)
    assert forecast.fast_log_variance[0, 0, 0, 0] == pytest.approx(1.0)
    assert forecast.fast_log_variance[0, 0, 0, 3] == pytest.approx(1.0)


def test_runtime_controller_selects_models_per_persistent_entity() -> None:
    belief = BeliefFactory(max_objects=2).create()
    objects = belief.objects.clone()
    objects.active[0, :2] = True
    objects.object_id[0, :2] = torch.tensor([7, 8])
    source = belief.replace(objects=objects)
    controller = RuntimeHypothesisController(
        HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)]),
        evidence_horizons_seconds=(0.1,),
        axis_independent_axes=(0,),
    )
    controller.reset(1, device=source.device, dtype=source.dtype)
    controller.schedule(source)
    due = source.replace(timestamp=torch.tensor([0.1]))
    controller.assimilate_observation(
        due,
        SimpleNamespace(
            timestamp=torch.tensor([0.1]),
            measurement_mask=torch.tensor([[True, True]]),
            auxiliary={"world_position": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])},
        ),
        SimpleNamespace(
            pair_mask=torch.tensor([[True, True]]),
            belief_indices=torch.tensor([[0, 1]], dtype=torch.int64),
            measurement_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        ),
    )

    forecast = controller.predict(due, [0.1])

    assert forecast is not None
    assert forecast.positions[0, 0, :2, 0].tolist() == pytest.approx([0.0, 1.0])
    assert forecast.auxiliary["hypothesis_axis_index"][0, 0, :2, 0].tolist() == [
        1,
        0,
    ]


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
    assert damped.belief.objects.velocity[0, 0, 0].item() == pytest.approx(
        2.0 * torch.exp(torch.tensor(-4.0)).item()
    )
    expected_displacement = 2.0 * (1.0 - torch.exp(torch.tensor(-4.0)).item()) / 2.0
    assert damped.belief.objects.position[0, 0, 0].item() == pytest.approx(
        1.0 + expected_displacement
    )
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
