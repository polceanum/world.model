from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from world_model.belief import RigidPrimitive
from world_model.simulator import PhysicsConfig, RigidBodyState, SphereState, advance_rigid_bodies


def _base(count: int = 2) -> RigidBodyState:
    sphere = SphereState(
        object_id=torch.arange(count, dtype=torch.int64),
        active=torch.ones(count, dtype=torch.bool),
        position=torch.zeros(count, 3, dtype=torch.float64),
        velocity=torch.zeros(count, 3, dtype=torch.float64),
        radius=torch.full((count, 1), 0.5, dtype=torch.float64),
        mass=torch.ones(count, 1, dtype=torch.float64),
        restitution=torch.ones(count, 1, dtype=torch.float64),
        drag=torch.zeros(count, 1, dtype=torch.float64),
        friction=torch.zeros(count, 1, dtype=torch.float64),
        albedo=torch.full((count, 3), 0.5, dtype=torch.float64),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        .expand(count, -1)
        .clone(),
        angular_velocity=torch.zeros(count, 3, dtype=torch.float64),
        sleeping=torch.zeros(count, dtype=torch.bool),
        sleep_counter=torch.zeros(count, dtype=torch.int64),
    )
    return replace(
        RigidBodyState.from_spheres(sphere),
        primitive=torch.full((count,), int(RigidPrimitive.BOX), dtype=torch.int64),
        half_extents=torch.tensor([[0.4, 0.25, 0.2]], dtype=torch.float64)
        .expand(count, -1)
        .clone(),
    )


def _config() -> PhysicsConfig:
    return PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
        max_substep=0.005,
        solver_iterations=1,
        position_correction=1.0,
        penetration_slop=0.0,
        max_position_correction=1.0,
    )


def test_independent_box_reference_resolves_central_elastic_impact() -> None:
    state = _base()
    state = replace(
        state,
        position=torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64),
        velocity=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64),
    )

    result, events = advance_rigid_bodies(state, 0.2, _config())

    assert events.pair_collision[0, 1]
    torch.testing.assert_close(
        result.velocity[:, 0],
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        atol=1.0e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(result.velocity.sum(dim=0), torch.zeros(3, dtype=torch.float64))


def test_independent_reference_applies_external_impulse_once() -> None:
    state = _base(count=1)
    state = replace(state, mass=torch.full((1, 1), 2.0, dtype=torch.float64))
    impulse = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)

    result, events = advance_rigid_bodies(state, 0.1, _config(), external_impulse=impulse)

    torch.testing.assert_close(
        result.velocity[0], torch.tensor([0.25, 0.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.position[0], torch.tensor([0.025, 0.0, 0.0], dtype=torch.float64)
    )
    assert torch.equal(events.external_impulse, impulse)


def test_independent_reference_uses_rotated_box_boundary_support() -> None:
    state = _base(count=1)
    half_angle = math.pi / 4.0
    state = replace(
        state,
        position=torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float64),
        half_extents=torch.tensor([[0.6, 0.2, 0.2]], dtype=torch.float64),
        orientation=torch.tensor(
            [[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]], dtype=torch.float64
        ),
    )
    config = replace(_config(), bounds=((-10.0, 10.0), (0.0, 10.0), (-10.0, 10.0)))

    result, events = advance_rigid_bodies(state, 0.001, config)

    assert events.boundary_contact[0, 2]
    torch.testing.assert_close(
        result.position[0, 1], torch.tensor(0.6, dtype=torch.float64), atol=1.0e-10, rtol=0.0
    )


def test_rigid_state_rejects_invalid_active_physical_parameters() -> None:
    state = replace(_base(count=1), mass=torch.zeros(1, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="positive mass"):
        state.validate()
