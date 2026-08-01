"""Regression tests for stable high-rate simulator integration."""

from __future__ import annotations

import torch

from world_model.simulator import PhysicsConfig, SphereState, advance_spheres


def _single_sphere_state() -> SphereState:
    return SphereState(
        object_id=torch.tensor([0], dtype=torch.int64),
        active=torch.tensor([True]),
        position=torch.tensor([[0.0, 0.65, 0.0]]),
        velocity=torch.tensor([[0.8, -1.0, 0.0]]),
        radius=torch.tensor([[0.2]]),
        mass=torch.tensor([[1.0]]),
        restitution=torch.tensor([[0.02]]),
        drag=torch.tensor([[0.0]]),
        friction=torch.tensor([[0.0]]),
        albedo=torch.ones((1, 3)),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        angular_velocity=torch.zeros((1, 3)),
        sleeping=torch.tensor([False]),
        sleep_counter=torch.zeros(1, dtype=torch.int64),
    )


def test_low_restitution_slider_settles_without_repeated_ground_impacts() -> None:
    state = _single_sphere_state()
    config = PhysicsConfig(
        bounds=((-5.0, 5.0), (0.0, 3.0), (-5.0, 5.0)),
        sleep_speed=0.0,
        sleep_after_seconds=10.0,
    )
    collision_frames: list[int] = []
    tail_positions: list[float] = []
    tail_vertical_speeds: list[float] = []

    for frame_index in range(120):
        state, events = advance_spheres(state, 1.0 / 30.0, config)
        if bool(events.ground_collision[0]):
            collision_frames.append(frame_index)
        if frame_index >= 80:
            tail_positions.append(float(state.position[0, 1]))
            tail_vertical_speeds.append(float(state.velocity[0, 1]))

    assert collision_frames == [6]
    assert max(abs(position - 0.2) for position in tail_positions) < 1.0e-6
    assert max(abs(speed) for speed in tail_vertical_speeds) < 1.0e-7
    # The resting constraint cancels only inward normal speed. Horizontal
    # sliding remains active and is not mistaken for sleep.
    assert state.position[0, 0] > 3.0
    assert state.velocity[0, 0] > 0.79
