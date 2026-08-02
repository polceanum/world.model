from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.dynamics import AnalyticKinematics, DynamicsModel


def _active_belief(batch_size: int = 1, max_objects: int = 2):
    belief = BeliefFactory(max_objects=max_objects).create(batch_size=batch_size)
    objects = belief.objects.clone()
    objects.active[:, 0] = True
    objects.object_id[:, 0] = torch.arange(batch_size)
    objects.position[:, 0, 1] = 1.0
    objects.log_drag[:, 0] = -16.0
    return replace(
        belief,
        objects=objects,
        next_object_id=torch.ones(batch_size, dtype=torch.int64),
    )


def _pair_collision_belief(batch_size: int = 1):
    belief = BeliefFactory(max_objects=2).create(batch_size=batch_size)
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.tensor([0, 1])
    objects.position[:, 0] = torch.tensor([-0.15, 1.0, 0.0])
    objects.position[:, 1] = torch.tensor([0.15, 1.0, 0.0])
    objects.velocity[:, 0, 0] = 1.0
    objects.velocity[:, 1, 0] = -1.0
    objects.geometry[..., 0] = 0.1
    objects.log_drag.fill_(-16.0)
    return replace(
        belief,
        objects=objects,
        gravity=torch.zeros_like(belief.gravity),
        next_object_id=torch.full((batch_size,), 2, dtype=torch.int64),
    )


def _deterministic_collision_model(belief):
    model = DynamicsModel.from_belief(belief, max_substep=0.01)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def test_isolated_gravity_trajectory() -> None:
    belief = _active_belief()
    integrated = AnalyticKinematics()(
        belief.objects,
        belief.gravity,
        0.1,
    )

    assert integrated.velocity[0, 0, 1].item() == pytest.approx(-0.981, rel=2e-5)
    assert integrated.position[0, 0, 1].item() == pytest.approx(
        1.0 - 0.5 * 9.81 * 0.1**2,
        rel=2e-5,
    )
    torch.testing.assert_close(integrated.position[0, 1], belief.objects.position[0, 1])


def test_linear_drag_uses_stable_exponential_solution() -> None:
    belief = _active_belief()
    objects = belief.objects.clone()
    objects.velocity[0, 0, 0] = 1.0
    objects.log_drag[0, 0] = math.log(2.0)
    gravity = torch.zeros_like(belief.gravity)

    integrated = AnalyticKinematics()(objects, gravity, 0.5)

    expected_velocity = math.exp(-1.0)
    expected_position = (1.0 - math.exp(-1.0)) / 2.0
    assert integrated.velocity[0, 0, 0].item() == pytest.approx(expected_velocity, rel=1e-6)
    assert integrated.position[0, 0, 0].item() == pytest.approx(expected_position, rel=1e-6)


def test_sleeping_object_is_not_advanced() -> None:
    belief = _active_belief()
    objects = belief.objects.clone()
    objects.motion_mode_logits[0, 0].fill_(-4)
    objects.motion_mode_logits[0, 0, MotionMode.SLEEPING] = 4
    objects.velocity[0, 0] = torch.tensor([1.0, 2.0, 3.0])

    integrated = AnalyticKinematics()(objects, belief.gravity, 0.5)
    torch.testing.assert_close(integrated.position, objects.position)
    torch.testing.assert_close(integrated.velocity, objects.velocity)


def test_irregular_per_batch_elapsed_times() -> None:
    belief = _active_belief(batch_size=2)
    integrated = AnalyticKinematics()(
        belief.objects,
        belief.gravity,
        torch.tensor([0.1, 0.2]),
    )
    assert integrated.velocity[0, 0, 1].item() == pytest.approx(-0.981, rel=2e-5)
    assert integrated.velocity[1, 0, 1].item() == pytest.approx(-1.962, rel=2e-5)


def test_composite_predict_and_rollout_do_not_mutate_source() -> None:
    belief = _active_belief()
    source = belief.clone()
    model = DynamicsModel.from_belief(belief, max_substep=1.0 / 120.0)

    predicted = model.predict(belief, 0.05)
    trajectory = model.rollout(belief, torch.tensor([0.0, 0.03, 0.1]))

    torch.testing.assert_close(belief.timestamp, source.timestamp)
    torch.testing.assert_close(belief.objects.position, source.objects.position)
    torch.testing.assert_close(belief.objects.fast_log_variance, source.objects.fast_log_variance)
    assert predicted.timestamp.item() == pytest.approx(0.05)
    torch.testing.assert_close(
        trajectory.timestamps,
        torch.tensor([[0.0, 0.03, 0.1]]),
    )
    assert trajectory.positions.shape == (1, 3, 2, 3)
    assert trajectory.event_logits is not None
    assert trajectory.auxiliary["process_variance"].shape[:2] == (1, 3)
    assert torch.isfinite(trajectory.positions).all()
    assert (
        predicted.objects.fast_log_variance[0, 0, :6] > belief.objects.fast_log_variance[0, 0, :6]
    ).all()


def test_per_batch_zero_dt_is_an_exact_identity() -> None:
    belief = _active_belief(batch_size=2)
    objects = belief.objects.clone()
    objects.position[0, 0, 1] = 0.05  # penetrating ground, but time does not advance
    belief = replace(belief, objects=objects)
    model = DynamicsModel.from_belief(belief)

    predicted = model.predict(belief, torch.tensor([0.0, 0.02]))

    torch.testing.assert_close(
        predicted.objects.position[0],
        belief.objects.position[0],
    )
    torch.testing.assert_close(
        predicted.objects.velocity[0],
        belief.objects.velocity[0],
    )
    torch.testing.assert_close(
        predicted.objects.motion_mode_logits[0],
        belief.objects.motion_mode_logits[0],
    )
    assert predicted.timestamp[0] == belief.timestamp[0]
    assert predicted.timestamp[1] > belief.timestamp[1]


def test_rollout_collision_logits_cover_each_prediction_segment() -> None:
    belief = _pair_collision_belief()
    model = _deterministic_collision_model(belief)

    trajectory = model.rollout(belief, [0.06, 0.10])

    collision = MotionMode.COLLISION
    assert (trajectory.event_logits[0, 0, :, collision] > 0).all()
    assert (trajectory.event_logits[0, 1, :, collision] < 0).all()
    assert trajectory.auxiliary["pair_collision"][0, 0, 0, 1]
    assert not trajectory.auxiliary["pair_collision"][0, 1].any()
    assert trajectory.auxiliary["pair_impulse"][0, 0, 0, 1] > 0
    # The first endpoint retains a small solver-consistent overlap while the
    # spheres separate, so it is contact but no longer the interval collision
    # mode. The second endpoint has separated fully.
    assert trajectory.motion_mode_logits[0, 0, 0].argmax() == MotionMode.PAIR_CONTACT
    assert trajectory.motion_mode_logits[0, 1, 0].argmax() == MotionMode.FREE


def test_rollout_zero_duration_segment_masks_collision_occurrence_per_batch() -> None:
    belief = _pair_collision_belief(batch_size=2)
    objects = belief.objects.clone()
    objects.motion_mode_logits[0, :, MotionMode.FREE] = -4.0
    objects.motion_mode_logits[0, :, MotionMode.COLLISION] = 8.0
    belief = replace(belief, objects=objects)
    model = _deterministic_collision_model(belief)

    trajectory = model.rollout(belief, torch.tensor([[0.0], [0.06]]))

    collision = MotionMode.COLLISION
    assert (trajectory.event_logits[0, 0, :, collision] < 0).all()
    assert not trajectory.auxiliary["pair_collision"][0, 0].any()
    assert trajectory.auxiliary["pair_impulse"][0, 0].count_nonzero() == 0
    torch.testing.assert_close(trajectory.positions[0, 0], belief.objects.position[0])
    torch.testing.assert_close(
        trajectory.motion_mode_logits[0, 0],
        belief.objects.motion_mode_logits[0],
    )
    assert (trajectory.event_logits[1, 0, :, collision] > 0).all()
    assert trajectory.auxiliary["pair_collision"][1, 0, 0, 1]


def test_rollout_rejects_unsorted_or_negative_offsets() -> None:
    belief = _active_belief()
    model = DynamicsModel.from_belief(belief)
    with pytest.raises(ValueError, match="sorted"):
        model.rollout(belief, [0.2, 0.1])
    with pytest.raises(ValueError, match="nonnegative"):
        model.rollout(belief, [-0.1, 0.1])
