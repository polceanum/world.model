"""Regression tests for stable high-rate simulator integration."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.dynamics import AnalyticKinematics
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


def test_collision_free_drag_and_gravity_match_exact_continuous_solution() -> None:
    initial = replace(
        _single_sphere_state(),
        position=torch.tensor([[1.25, 7.0, -0.8]]),
        velocity=torch.tensor([[1.1, -0.4, 0.35]]),
        drag=torch.tensor([[0.27]]),
    )
    gravity = torch.tensor([0.25, -9.81, -0.15])
    dt = 0.73
    config = PhysicsConfig(
        gravity=tuple(float(value) for value in gravity),
        bounds=((-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)),
        max_substep=1.0 / 120.0,
    )

    integrated, events = advance_spheres(initial, dt, config)

    drag = initial.drag
    decay = torch.exp(-drag * dt)
    one_minus_decay = -torch.expm1(-drag * dt)
    expected_velocity = initial.velocity * decay + gravity * one_minus_decay / drag
    expected_position = (
        initial.position
        + initial.velocity * one_minus_decay / drag
        + gravity * (dt / drag - one_minus_decay / drag.square())
    )
    torch.testing.assert_close(integrated.velocity, expected_velocity, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(integrated.position, expected_position, atol=2.0e-5, rtol=2.0e-5)
    assert not events.collision.any()


def test_collision_free_exact_integration_is_substep_invariant() -> None:
    initial = replace(
        _single_sphere_state(),
        position=torch.tensor([[-0.4, 8.0, 0.7]]),
        velocity=torch.tensor([[0.9, 0.3, -0.55]]),
        drag=torch.tensor([[0.19]]),
    )
    common = {
        "gravity": (0.0, -9.81, 0.1),
        "bounds": ((-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)),
    }

    fine, fine_events = advance_spheres(
        initial,
        0.8,
        PhysicsConfig(max_substep=1.0 / 240.0, **common),
    )
    coarse, coarse_events = advance_spheres(
        initial,
        0.8,
        PhysicsConfig(max_substep=0.8, **common),
    )

    torch.testing.assert_close(fine.position, coarse.position, atol=3.0e-5, rtol=3.0e-5)
    torch.testing.assert_close(fine.velocity, coarse.velocity, atol=3.0e-5, rtol=3.0e-5)
    assert fine_events.substeps == 192
    assert coarse_events.substeps == 1
    assert not fine_events.collision.any()
    assert not coarse_events.collision.any()


@pytest.mark.parametrize("max_substep", [1.0 / 120.0, 1.0 / 60.0, 1.0 / 40.0])
def test_simulator_free_motion_matches_analytic_dynamics_model(max_substep: float) -> None:
    initial = replace(
        _single_sphere_state(),
        position=torch.tensor([[0.7, 8.0, -0.45]]),
        velocity=torch.tensor([[1.2, 0.25, -0.3]]),
        drag=torch.tensor([[0.23]]),
    )
    gravity = (0.1, -9.81, 0.2)
    dt = 0.8
    simulated, events = advance_spheres(
        initial,
        dt,
        PhysicsConfig(
            gravity=gravity,
            bounds=((-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)),
            max_substep=max_substep,
        ),
    )

    belief = BeliefFactory(max_objects=1).create(batch_size=1)
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = initial.position[0]
    objects.velocity[0, 0] = initial.velocity[0]
    objects.log_drag[0, 0] = math.log(float(initial.drag[0, 0]))
    expected = AnalyticKinematics()(
        objects,
        torch.tensor([gravity]),
        dt,
    )

    torch.testing.assert_close(
        simulated.position[0],
        expected.position[0, 0],
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    torch.testing.assert_close(
        simulated.velocity[0],
        expected.velocity[0, 0],
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert not events.collision.any()
