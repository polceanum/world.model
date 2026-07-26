from __future__ import annotations

import math

import pytest
import torch

from world_model.dynamics.quaternion import (
    integrate_quaternion,
    normalize_quaternion,
    quaternion_from_rotation_vector,
    quaternion_geodesic_distance,
    quaternion_multiply,
)


def test_identity_and_known_axis_rotation() -> None:
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    quarter_turn = quaternion_from_rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2]))

    torch.testing.assert_close(quaternion_multiply(identity, quarter_turn), quarter_turn)
    torch.testing.assert_close(
        quarter_turn,
        torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_and_small_angular_velocity_have_finite_gradients() -> None:
    orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    angular_velocity = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    integrated = integrate_quaternion(orientation, angular_velocity, 0.1)

    torch.testing.assert_close(integrated, orientation)
    integrated[..., :3].sum().backward()
    assert angular_velocity.grad is not None
    assert torch.isfinite(angular_velocity.grad).all()


def test_repeated_integration_preserves_unit_norm() -> None:
    orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    velocity = torch.tensor([[0.3, -0.2, 0.5]])
    for _ in range(1000):
        orientation = integrate_quaternion(orientation, velocity, 0.01)

    torch.testing.assert_close(
        torch.linalg.vector_norm(orientation, dim=-1),
        torch.ones(1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_geodesic_distance_is_sign_invariant() -> None:
    first = normalize_quaternion(torch.randn(8, 4))
    distance = quaternion_geodesic_distance(first, -first)
    assert distance.max().item() == pytest.approx(0.0, abs=1e-6)
