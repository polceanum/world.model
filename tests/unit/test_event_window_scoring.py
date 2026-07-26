from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory, MotionMode
from world_model.evaluation.evaluator import (
    _collision_logits_for_observation_windows,
)
from world_model.training.event_windows import observation_window_query_plan
from world_model.training.loop import _rollout_losses
from world_model.utils.config import load_config


def test_observation_window_plan_brackets_every_target_frame() -> None:
    plan = observation_window_query_plan([1, 3, 4, 9], frame_rate=20.0)

    assert plan.target_frame_offsets == (1, 3, 4, 9)
    assert plan.query_frame_offsets == (0, 1, 2, 3, 4, 8, 9)
    assert plan.target_query_indices == (1, 3, 4, 6)
    assert plan.target_seconds == pytest.approx((0.05, 0.15, 0.20, 0.45))
    for target, query_index in zip(
        plan.target_frame_offsets,
        plan.target_query_indices,
        strict=True,
    ):
        assert plan.query_frame_offsets[query_index - 1 : query_index + 1] == (
            target - 1,
            target,
        )


def test_evaluator_collision_selection_uses_only_preceding_windows() -> None:
    plan = observation_window_query_plan([2, 5, 10], frame_rate=20.0)
    event_logits = torch.zeros(1, len(plan.query_seconds), 1, len(MotionMode))
    event_logits[0, :, 0, MotionMode.COLLISION] = torch.tensor([-8.0, 8.0, 8.0, -8.0, -8.0, 8.0])

    selected = _collision_logits_for_observation_windows(event_logits, plan)

    torch.testing.assert_close(selected[0, :, 0], torch.tensor([8.0, -8.0, 8.0]))


class _RecordingDynamics:
    def __init__(self) -> None:
        self.query_seconds: tuple[float, ...] | None = None

    def rollout(
        self,
        belief,
        query_seconds,
        *,
        return_events: bool,
    ) -> BeliefTrajectory:
        assert return_events
        self.query_seconds = tuple(float(value) for value in query_seconds)
        query = belief.timestamp.new_tensor(self.query_seconds)
        count = len(self.query_seconds)
        objects = belief.objects
        event_logits = objects.motion_mode_logits[:, None].expand(-1, count, -1, -1).clone()
        scores_by_offset = {
            1: -8.0,
            2: 8.0,
            4: 8.0,
            5: -8.0,
            9: -8.0,
            10: 8.0,
        }
        event_logits[0, :, 0, MotionMode.COLLISION] = belief.timestamp.new_tensor(
            [scores_by_offset[round(value * 20.0)] for value in self.query_seconds]
        )
        return BeliefTrajectory(
            timestamps=belief.timestamp[:, None] + query[None],
            positions=objects.position[:, None].expand(-1, count, -1, -1).clone(),
            velocities=objects.velocity[:, None].expand(-1, count, -1, -1).clone(),
            orientations=objects.orientation[:, None].expand(-1, count, -1, -1).clone(),
            motion_mode_logits=objects.motion_mode_logits[:, None]
            .expand(-1, count, -1, -1)
            .clone(),
            fast_log_variance=objects.fast_log_variance[:, None].expand(-1, count, -1, -1).clone(),
            active_mask=objects.active[:, None].expand(-1, count, -1).clone(),
            event_logits=event_logits,
        ).validate()


def test_training_collision_loss_scores_exact_observation_windows() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    belief = replace(belief, objects=objects)
    dynamics = _RecordingDynamics()
    model = SimpleNamespace(dynamics=dynamics)
    frames = config.simulator.sequence_frames
    collision = torch.zeros(1, frames, 1, dtype=torch.bool)
    collision[0, [2, 10], 0] = True
    batch = {
        "rgb": torch.zeros(1, frames, 3, 1, 1),
        "objects": {
            "active": torch.ones(1, frames, 1, dtype=torch.bool),
            "position": torch.zeros(1, frames, 1, 3),
            "velocity": torch.zeros(1, frames, 1, 3),
        },
        "events": {"collision": collision},
    }

    losses = _rollout_losses(
        model,
        belief,
        batch,
        config,
        frame_index=0,
        indices=torch.zeros(1, 1, dtype=torch.int64),
        matched=torch.ones(1, 1, dtype=torch.bool),
    )

    assert dynamics.query_seconds == pytest.approx((0.05, 0.10, 0.20, 0.25, 0.45, 0.50))
    assert losses["event_collision"].item() < 0.001
    assert losses["rollout_position"].item() == 0.0
    assert losses["rollout_velocity"].item() == 0.0
