"""Focused analytic contact tests for the deterministic toy simulator."""

from __future__ import annotations

import torch

from world_model.simulator.collisions import (
    BOUNDARY_NAMES,
    resolve_axis_aligned_boundaries,
    resolve_sphere_sphere_collisions,
    sphere_sphere_relative_restitution,
)


def test_elastic_equal_mass_collision_conserves_momentum_and_restitution() -> None:
    position = torch.tensor([[-0.45, 1.0, 0.0], [0.45, 1.0, 0.0], [0.0, 0.0, 0.0]])
    velocity = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # The third slot is deliberately zero-padded.  Collision validation must
    # apply physical positivity checks only to active objects.
    radius = torch.tensor([[0.5], [0.5], [0.0]])
    mass = torch.tensor([[1.0], [1.0], [0.0]])
    restitution = torch.tensor([[1.0], [1.0], [0.0]])
    active = torch.tensor([True, True, False])

    result = resolve_sphere_sphere_collisions(
        position,
        velocity,
        radius,
        mass,
        restitution,
        active=active,
    )

    assert result.collision[0, 1]
    assert not result.contact[2].any()
    torch.testing.assert_close(
        result.velocity[:2],
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    momentum_before = (mass[:2] * velocity[:2]).sum(dim=0)
    momentum_after = (mass[:2] * result.velocity[:2]).sum(dim=0)
    torch.testing.assert_close(momentum_after, momentum_before)
    measured_restitution = sphere_sphere_relative_restitution(
        velocity[:2], result.velocity[:2], torch.tensor([1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(measured_restitution, torch.tensor(1.0))


def test_perfectly_inelastic_equal_mass_collision_shares_velocity() -> None:
    position = torch.tensor([[-0.49, 1.0, 0.0], [0.49, 1.0, 0.0]])
    velocity = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scalar = torch.ones((2, 1))

    result = resolve_sphere_sphere_collisions(
        position,
        velocity,
        0.5 * scalar,
        scalar,
        torch.zeros_like(scalar),
    )

    torch.testing.assert_close(
        result.velocity,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(result.impulse_magnitude[0, 1], torch.tensor(1.0))


def test_separating_overlapping_spheres_receive_no_impulse() -> None:
    position = torch.tensor([[-0.45, 1.0, 0.0], [0.45, 1.0, 0.0]])
    velocity = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scalar = torch.ones((2, 1))

    result = resolve_sphere_sphere_collisions(
        position,
        velocity,
        0.5 * scalar,
        scalar,
        scalar,
    )

    assert result.contact[0, 1]
    assert not result.collision[0, 1]
    torch.testing.assert_close(result.velocity, velocity)
    assert result.penetration[0, 1] > 0
    assert torch.linalg.vector_norm(
        result.position[1] - result.position[0]
    ) > torch.linalg.vector_norm(position[1] - position[0])


def test_floor_bounce_uses_object_restitution_and_projects_penetration() -> None:
    position = torch.tensor([[0.0, 0.45, 0.0]])
    velocity = torch.tensor([[0.6, -2.0, -0.2]])
    radius = torch.tensor([[0.5]])
    mass = torch.tensor([[2.0]])
    restitution = torch.tensor([[0.75]])
    friction = torch.tensor([[0.0]])

    result = resolve_axis_aligned_boundaries(
        position,
        velocity,
        radius,
        mass,
        restitution,
        ((-2.0, 2.0), (0.0, 3.0), (-2.0, 2.0)),
        friction=friction,
    )

    floor = BOUNDARY_NAMES.index("floor")
    assert result.contact[0, floor]
    assert result.collision[0, floor]
    torch.testing.assert_close(result.position[0, 1], torch.tensor(0.5))
    torch.testing.assert_close(result.velocity[0, 1], torch.tensor(1.5))
    torch.testing.assert_close(result.velocity[0, [0, 2]], velocity[0, [0, 2]])


def test_wall_collision_is_permutation_independent_for_object_slots() -> None:
    position = torch.tensor([[-1.9, 1.0, 0.0], [1.9, 1.2, 0.0]])
    velocity = torch.tensor([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radius = torch.full((2, 1), 0.2)
    mass = torch.ones((2, 1))
    restitution = torch.full((2, 1), 0.5)
    bounds = ((-2.0, 2.0), (0.0, 3.0), (-2.0, 2.0))

    direct = resolve_axis_aligned_boundaries(position, velocity, radius, mass, restitution, bounds)
    permuted = resolve_axis_aligned_boundaries(
        position.flip(0),
        velocity.flip(0),
        radius.flip(0),
        mass.flip(0),
        restitution.flip(0),
        bounds,
    )

    torch.testing.assert_close(direct.position, permuted.position.flip(0))
    torch.testing.assert_close(direct.velocity, permuted.velocity.flip(0))
