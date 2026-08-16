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
        return_auxiliary: bool,
    ) -> BeliefTrajectory:
        assert return_events
        assert not return_auxiliary
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
    for seconds in (0.1, 0.25, 0.5):
        assert losses[f"event_collision@{seconds:.3f}s"].item() < 0.001
    assert losses["rollout_position"].item() == 0.0
    assert losses["rollout_velocity"].item() == 0.0


class _PairEventDynamics:
    def __init__(self, pair_collision_logits: torch.Tensor) -> None:
        self.pair_collision_logits = pair_collision_logits

    def rollout(
        self,
        belief,
        query_seconds,
        *,
        return_events: bool,
        return_auxiliary: bool,
        auxiliary_names,
    ) -> BeliefTrajectory:
        assert return_events
        assert return_auxiliary
        assert tuple(auxiliary_names) == ("pair_event_logits",)
        count = len(query_seconds)
        objects = belief.objects
        query = belief.timestamp.new_tensor(query_seconds)
        event_logits = objects.motion_mode_logits[:, None].expand(-1, count, -1, -1).clone()
        event_logits[..., MotionMode.COLLISION] = 8.0
        pair_logits = objects.position.new_full(
            (belief.batch_size, count, objects.max_objects, objects.max_objects, 2),
            -8.0,
        )
        pair_logits[..., 1] = self.pair_collision_logits[:, None]
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
            auxiliary={"pair_event_logits": pair_logits},
        ).validate()


def test_smooth_event_training_supervises_pair_ownership_not_only_node_events() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(
                config.model.dynamics,
                smooth_event_hazard_enabled=True,
            ),
        ),
        training=replace(
            config.training,
            horizon_weights=(1.0,),
            minimum_rollout_age_steps=0,
        ),
        evaluation=replace(config.evaluation, horizons_seconds=(0.05,)),
    )
    belief = BeliefFactory(max_objects=4).create()
    objects = belief.objects.clone()
    objects.active.fill_(True)
    objects.object_id[0] = torch.arange(4)
    belief = replace(belief, objects=objects)
    frames = config.simulator.sequence_frames
    pair_target = torch.zeros(1, frames, 4, 4, dtype=torch.bool)
    for first, second in ((0, 1), (2, 3)):
        pair_target[0, 1, first, second] = True
        pair_target[0, 1, second, first] = True
    node_target = pair_target.any(dim=-1)
    batch = {
        "rgb": torch.zeros(1, frames, 3, 1, 1),
        "objects": {
            "active": torch.ones(1, frames, 4, dtype=torch.bool),
            "position": torch.zeros(1, frames, 4, 3),
            "velocity": torch.zeros(1, frames, 4, 3),
        },
        "events": {
            "collision": node_target,
            "pair_collision": pair_target,
        },
    }

    correct_logits = torch.full((1, 4, 4), -8.0)
    wrong_logits = torch.full((1, 4, 4), -8.0)
    for first, second in ((0, 1), (2, 3)):
        correct_logits[0, first, second] = 8.0
        correct_logits[0, second, first] = 8.0
    for first, second in ((0, 2), (1, 3)):
        wrong_logits[0, first, second] = 8.0
        wrong_logits[0, second, first] = 8.0

    def event_loss(pair_logits: torch.Tensor) -> torch.Tensor:
        losses = _rollout_losses(
            SimpleNamespace(dynamics=_PairEventDynamics(pair_logits)),
            belief,
            batch,
            config,
            frame_index=0,
            indices=torch.arange(4).unsqueeze(0),
            matched=torch.ones(1, 4, dtype=torch.bool),
        )
        return losses["event_collision"]

    correct_loss = event_loss(correct_logits)
    wrong_loss = event_loss(wrong_logits)

    assert correct_loss.item() < 0.001
    assert wrong_loss.item() > 1.0
