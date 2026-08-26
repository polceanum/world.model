from __future__ import annotations

import math

import pytest
import torch

from world_model.dynamics import fit_free_motion, free_motion_position_velocity
from world_model.simulator.physics import _integrate_free_motion_exact


def _oracle_history(
    anchor_position: torch.Tensor,
    anchor_velocity: torch.Tensor,
    timestamps: torch.Tensor,
    anchor_time: torch.Tensor,
    gravity: torch.Tensor,
    drag: torch.Tensor,
) -> torch.Tensor:
    """Generate exact histories through the public signed-time oracle."""

    frames = []
    for time_index in range(timestamps.shape[1]):
        position, _ = free_motion_position_velocity(
            anchor_position,
            anchor_velocity,
            timestamps[:, time_index] - anchor_time,
            gravity=gravity,
            drag=drag,
        )
        frames.append(position)
    return torch.stack(frames, dim=1)


def test_fit_recovers_batched_slot_states_from_negative_signed_times() -> None:
    dtype = torch.float64
    anchor_position = torch.tensor(
        [
            [[0.4, 1.2, -0.3], [-0.7, 0.8, 0.2], [1.1, -0.5, 0.6]],
            [[-0.2, 0.9, 0.7], [0.5, 1.4, -0.8], [0.3, 0.2, 0.1]],
        ],
        dtype=dtype,
    )
    anchor_velocity = torch.tensor(
        [
            [[1.4, -0.1, 0.3], [-0.8, 0.5, 0.2], [0.1, -1.2, 0.4]],
            [[0.2, 0.7, -0.4], [1.1, -0.6, 0.9], [-0.3, 0.4, 1.3]],
        ],
        dtype=dtype,
    )
    anchor_time = torch.tensor([[1.3, 1.1, 1.5], [0.8, 1.2, 0.9]], dtype=dtype)
    relative = torch.tensor([-0.71, -0.43, -0.24, -0.09, 0.0], dtype=dtype)
    timestamps = anchor_time[:, None, :] + relative[None, :, None]
    gravity = torch.tensor(
        [
            [[0.1, -9.81, 0.0], [0.0, -9.7, 0.2], [-0.1, -9.9, 0.0]],
            [[0.0, -9.6, 0.1], [0.2, -9.81, -0.1], [0.0, -10.0, 0.0]],
        ],
        dtype=dtype,
    )
    drag = torch.tensor([[0.0, 0.07, 0.31], [0.02, 0.15, 0.5]], dtype=dtype)
    positions = _oracle_history(
        anchor_position,
        anchor_velocity,
        timestamps,
        anchor_time,
        gravity,
        drag,
    )

    result = fit_free_motion(
        positions,
        timestamps,
        gravity=gravity,
        drag=drag[..., None],
        anchor_time=anchor_time,
    )

    assert result.valid.all()
    assert result.position.shape == (2, 3, 3)
    assert result.velocity.shape == (2, 3, 3)
    assert result.normal_matrix.shape == (2, 3, 2, 2)
    assert result.residual_covariance.shape == (2, 3, 3, 3)
    assert torch.equal(result.support_count, torch.full((2, 3), 5))
    torch.testing.assert_close(result.anchor_position, anchor_position, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(result.anchor_velocity, anchor_velocity, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(result.predicted_positions, positions, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        result.residual_covariance,
        torch.zeros_like(result.residual_covariance),
        rtol=0,
        atol=1e-22,
    )


@pytest.mark.parametrize(
    "gravity_builder",
    [
        lambda base, batch, slots: base.squeeze(0),
        lambda base, batch, slots: base.expand(batch, 3).clone(),
        lambda base, batch, slots: base.expand(batch, slots, 3).clone(),
    ],
)
def test_fit_accepts_declared_gravity_shapes(gravity_builder) -> None:
    dtype = torch.float64
    batch, slots = 2, 2
    position = torch.tensor(
        [[[0.2, 0.4, 0.1], [-0.3, 0.8, 0.2]], [[0.7, 0.5, -0.1], [0.1, 1.1, 0.3]]],
        dtype=dtype,
    )
    velocity = torch.tensor(
        [[[0.9, 0.2, 0.0], [-0.4, 0.3, 0.1]], [[0.1, -0.2, 0.4], [0.5, 0.2, -0.3]]],
        dtype=dtype,
    )
    times = torch.tensor([[-0.4, -0.2, 0.0], [-0.3, -0.1, 0.0]], dtype=dtype)
    anchor = torch.zeros(batch, dtype=dtype)
    base_gravity = torch.tensor([[0.0, -9.81, 0.0]], dtype=dtype)
    gravity = gravity_builder(base_gravity, batch, slots)
    drag = torch.tensor([[0.0, 0.2], [0.05, 0.4]], dtype=dtype)
    expanded_times = times[:, :, None].expand(batch, times.shape[1], slots)
    positions = _oracle_history(
        position,
        velocity,
        expanded_times,
        anchor[:, None].expand(batch, slots),
        gravity,
        drag,
    )

    result = fit_free_motion(
        positions,
        times,
        gravity=gravity,
        drag=drag,
        anchor_time=anchor,
    )

    assert result.valid.all()
    torch.testing.assert_close(result.position, position, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(result.velocity, velocity, rtol=1e-11, atol=1e-11)


def test_zero_drag_is_exact_ballistic_and_continuous_from_tiny_drag() -> None:
    dtype = torch.float64
    position = torch.tensor([[[0.3, -0.2, 1.1]]], dtype=dtype)
    velocity = torch.tensor([[[1.2, 0.4, -0.7]]], dtype=dtype)
    gravity = torch.tensor([0.1, -9.81, 0.2], dtype=dtype)
    elapsed = torch.tensor([[-0.35]], dtype=dtype)
    zero_drag = torch.zeros((1, 1), dtype=dtype, requires_grad=True)

    next_position, next_velocity = free_motion_position_velocity(
        position,
        velocity,
        elapsed,
        gravity=gravity,
        drag=zero_drag,
    )
    expected_position = position + elapsed[..., None] * velocity
    expected_position = expected_position + 0.5 * elapsed.square()[..., None] * gravity
    expected_velocity = velocity + elapsed[..., None] * gravity

    torch.testing.assert_close(next_position, expected_position, rtol=0, atol=1e-14)
    torch.testing.assert_close(next_velocity, expected_velocity, rtol=0, atol=1e-14)
    drag_gradient = torch.autograd.grad(next_position.sum() + next_velocity.sum(), zero_drag)[0]
    assert torch.isfinite(drag_gradient).all()
    assert torch.count_nonzero(drag_gradient).item() == 1

    tiny_position, tiny_velocity = free_motion_position_velocity(
        position,
        velocity,
        elapsed,
        gravity=gravity,
        drag=torch.full((1, 1), 1.0e-10, dtype=dtype),
    )
    torch.testing.assert_close(tiny_position, expected_position, rtol=0, atol=2e-10)
    torch.testing.assert_close(tiny_velocity, expected_velocity, rtol=0, atol=2e-9)


@pytest.mark.parametrize("drag_value", [0.0, 0.17, 0.83])
def test_exact_propagation_has_semigroup_property(drag_value: float) -> None:
    dtype = torch.float64
    position = torch.tensor(
        [[[0.2, 0.7, -0.1], [-0.4, 1.3, 0.8]]],
        dtype=dtype,
    )
    velocity = torch.tensor(
        [[[1.1, -0.3, 0.2], [0.5, 0.9, -0.6]]],
        dtype=dtype,
    )
    gravity = torch.tensor([[0.0, -9.81, 0.1]], dtype=dtype)
    drag = torch.full((1, 2), drag_value, dtype=dtype)
    first_time = torch.tensor([0.23], dtype=dtype)
    second_time = torch.tensor([-0.08], dtype=dtype)

    first_position, first_velocity = free_motion_position_velocity(
        position,
        velocity,
        first_time,
        gravity=gravity,
        drag=drag,
    )
    composed_position, composed_velocity = free_motion_position_velocity(
        first_position,
        first_velocity,
        second_time,
        gravity=gravity,
        drag=drag,
    )
    direct_position, direct_velocity = free_motion_position_velocity(
        position,
        velocity,
        first_time + second_time,
        gravity=gravity,
        drag=drag,
    )

    torch.testing.assert_close(composed_position, direct_position, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(composed_velocity, direct_velocity, rtol=2e-13, atol=2e-13)


def test_exact_propagation_matches_simulator_free_motion_oracle() -> None:
    dtype = torch.float64
    position = torch.tensor(
        [[[0.2, 2.7, -0.1], [-0.4, 3.3, 0.8]], [[0.1, 4.0, 0.2], [0.7, 2.9, -0.5]]],
        dtype=dtype,
    )
    velocity = torch.tensor(
        [[[1.1, -0.3, 0.2], [0.5, 0.9, -0.6]], [[-0.2, 0.8, 0.1], [0.9, -0.4, 0.3]]],
        dtype=dtype,
    )
    gravity = torch.tensor([[0.0, -9.81, 0.0], [0.1, -9.7, -0.2]], dtype=dtype)
    drag = torch.tensor([[0.0, 0.17], [0.08, 0.63]], dtype=dtype)
    elapsed = 0.037

    actual_position, actual_velocity = free_motion_position_velocity(
        position,
        velocity,
        elapsed,
        gravity=gravity,
        drag=drag,
    )
    expected_positions = []
    expected_velocities = []
    for batch_index in range(position.shape[0]):
        expected_position, expected_velocity = _integrate_free_motion_exact(
            position[batch_index],
            velocity[batch_index],
            drag[batch_index, :, None],
            gravity[batch_index],
            elapsed,
            torch.ones(position.shape[1], dtype=torch.bool),
        )
        expected_positions.append(expected_position)
        expected_velocities.append(expected_velocity)

    torch.testing.assert_close(
        actual_position,
        torch.stack(expected_positions),
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        actual_velocity,
        torch.stack(expected_velocities),
        rtol=1e-13,
        atol=1e-13,
    )


def test_masked_support_cannot_change_valid_fit_and_invalid_rows_are_neutral() -> None:
    dtype = torch.float64
    anchor_position = torch.tensor([[[0.5, 1.0, -0.2], [-0.1, 0.8, 0.4]]], dtype=dtype)
    anchor_velocity = torch.tensor([[[0.9, -0.3, 0.2], [0.2, 0.5, -0.1]]], dtype=dtype)
    timestamps = torch.tensor([[-0.5, -0.3, -0.1, 0.0]], dtype=dtype)
    expanded_times = timestamps[:, :, None].expand(1, 4, 2)
    anchor_time = torch.zeros((1, 2), dtype=dtype)
    gravity = torch.tensor([0.0, -9.81, 0.0], dtype=dtype)
    drag = torch.tensor([[0.1, 0.2]], dtype=dtype)
    positions = _oracle_history(
        anchor_position,
        anchor_velocity,
        expanded_times,
        anchor_time,
        gravity,
        drag,
    )
    support = torch.tensor([[[True, True], [False, False], [True, False], [True, False]]])
    corrupted = positions.clone()
    corrupted[:, 1] = 1.0e6

    result = fit_free_motion(
        corrupted,
        timestamps,
        gravity=gravity,
        drag=drag,
        anchor_time=0.0,
        support=support,
    )

    assert result.valid.tolist() == [[True, False]]
    assert result.support_count.tolist() == [[3, 1]]
    torch.testing.assert_close(result.position[:, 0], anchor_position[:, 0], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(result.velocity[:, 0], anchor_velocity[:, 0], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(result.position[:, 1], torch.zeros((1, 3), dtype=dtype))
    torch.testing.assert_close(result.velocity[:, 1], torch.zeros((1, 3), dtype=dtype))
    torch.testing.assert_close(
        result.residual_covariance[:, 1],
        torch.zeros((1, 3, 3), dtype=dtype),
    )
    assert all(
        torch.isfinite(value).all()
        for value in (
            result.position,
            result.velocity,
            result.predicted_positions,
            result.residuals,
            result.residual_covariance,
            result.condition_number,
        )
    )


def test_fully_invalid_rows_have_zero_continuous_outputs_and_gradients() -> None:
    dtype = torch.float64
    positions = torch.randn((1, 4, 1, 3), dtype=dtype, requires_grad=True)
    timestamps = torch.tensor(
        [[-0.3, -0.2, -0.1, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    gravity = torch.tensor([0.0, -9.81, 0.1], dtype=dtype, requires_grad=True)
    drag = torch.tensor([[0.2]], dtype=dtype, requires_grad=True)
    weights = torch.ones((1, 4, 1), dtype=dtype, requires_grad=True)

    result = fit_free_motion(
        positions,
        timestamps,
        gravity=gravity,
        drag=drag,
        support=torch.zeros((1, 4, 1), dtype=torch.bool),
        weights=weights,
    )

    assert not result.valid.item()
    for value in (
        result.position,
        result.velocity,
        result.predicted_positions,
        result.residuals,
        result.residual_covariance,
    ):
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
    objective = sum(
        value.sum()
        for value in (
            result.position,
            result.velocity,
            result.predicted_positions,
            result.residuals,
            result.residual_covariance,
        )
    )
    gradients = torch.autograd.grad(
        objective,
        (positions, timestamps, gravity, drag, weights),
    )
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)


def test_uniform_soft_weight_scale_cannot_change_fit_validity_or_condition() -> None:
    dtype = torch.float32
    position = torch.tensor([[[0.3, 0.8, -0.1]]], dtype=dtype)
    velocity = torch.tensor([[[0.6, -0.2, 0.4]]], dtype=dtype)
    timestamps = torch.tensor([[-0.75, -0.55, -0.3, -0.1, 0.0]], dtype=dtype)
    gravity = torch.tensor([0.0, -9.81, 0.0], dtype=dtype)
    drag = torch.tensor([[0.12]], dtype=dtype)
    positions = _oracle_history(
        position,
        velocity,
        timestamps[:, :, None],
        torch.zeros((1, 1), dtype=dtype),
        gravity,
        drag,
    )
    relative_weights = torch.tensor(
        [[[0.4], [1.1], [0.7], [1.8], [0.9]]],
        dtype=dtype,
    )

    baseline = fit_free_motion(
        positions,
        timestamps,
        gravity=gravity,
        drag=drag,
        weights=relative_weights,
    )
    assert baseline.valid.item()
    assert baseline.condition_number.item() >= 1.0
    for scale in (1.0e-12, 1.0e-7, 1.0e7, 1.0e12):
        scaled = fit_free_motion(
            positions,
            timestamps,
            gravity=gravity,
            drag=drag,
            weights=relative_weights * scale,
        )
        assert scaled.valid.item()
        torch.testing.assert_close(scaled.position, baseline.position, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(scaled.velocity, baseline.velocity, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(
            scaled.normal_matrix,
            baseline.normal_matrix,
            rtol=2e-6,
            atol=2e-7,
        )
        torch.testing.assert_close(
            scaled.condition_number,
            baseline.condition_number,
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            scaled.support_weight,
            baseline.support_weight * scale,
            rtol=2e-6,
            atol=0,
        )


def test_residual_covariance_uses_only_weighted_supported_residuals() -> None:
    dtype = torch.float64
    position = torch.tensor([[[0.3, 0.8, -0.1]]], dtype=dtype)
    velocity = torch.tensor([[[0.6, -0.2, 0.4]]], dtype=dtype)
    timestamps = torch.tensor([[-0.6, -0.4, -0.2, -0.1, 0.0]], dtype=dtype)
    expanded = timestamps[:, :, None]
    gravity = torch.tensor([0.0, -9.81, 0.0], dtype=dtype)
    drag = torch.tensor([[0.12]], dtype=dtype)
    positions = _oracle_history(
        position,
        velocity,
        expanded,
        torch.zeros((1, 1), dtype=dtype),
        gravity,
        drag,
    )
    positions = positions + torch.tensor(
        [
            [
                [[0.01, -0.02, 0.03]],
                [[-0.02, 0.01, 0.0]],
                [[0.0, 0.02, -0.01]],
                [[0.03, 0.0, 0.02]],
                [[-0.01, -0.01, 0.0]],
            ]
        ],
        dtype=dtype,
    )
    weights = torch.tensor([[[1.0], [0.5], [0.0], [2.0], [1.5]]], dtype=dtype)
    support = torch.tensor([[[True], [True], [True], [False], [True]]])

    result = fit_free_motion(
        positions,
        timestamps,
        gravity=gravity,
        drag=drag,
        anchor_time=0.0,
        support=support,
        weights=weights,
    )
    effective = weights * support.to(dtype)
    expected = (
        torch.einsum(
            "btsc,bts,btsd->bscd",
            result.residuals,
            effective,
            result.residuals,
        )
        / effective.sum(dim=1)[..., None, None]
    )

    assert result.valid.item()
    torch.testing.assert_close(result.residual_covariance, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        result.residual_covariance,
        result.residual_covariance.transpose(-1, -2),
        rtol=0,
        atol=1e-15,
    )
    assert torch.linalg.eigvalsh(result.residual_covariance).amin().item() >= -1e-15


def test_fit_gradients_match_finite_difference_and_reach_soft_weights() -> None:
    dtype = torch.float64
    base_positions = torch.tensor(
        [
            [
                [[-0.21, 0.91, 0.10]],
                [[0.02, 0.72, 0.16]],
                [[0.25, 0.43, 0.23]],
                [[0.43, 0.19, 0.27]],
            ]
        ],
        dtype=dtype,
    )
    positions = base_positions.clone().requires_grad_(True)
    timestamps = torch.tensor([[-0.6, -0.4, -0.2, 0.0]], dtype=dtype, requires_grad=True)
    gravity = torch.tensor([0.0, -9.81, 0.1], dtype=dtype, requires_grad=True)
    drag = torch.tensor([[0.18]], dtype=dtype, requires_grad=True)
    anchor_time = torch.tensor(0.03, dtype=dtype, requires_grad=True)
    weights = torch.tensor([[[0.7], [1.1], [0.9], [1.4]]], dtype=dtype, requires_grad=True)

    def objective(position_value: torch.Tensor) -> torch.Tensor:
        result = fit_free_motion(
            position_value,
            timestamps,
            gravity=gravity,
            drag=drag,
            anchor_time=anchor_time,
            weights=weights,
        )
        return (
            0.7 * result.position.square().sum()
            + 0.2 * result.velocity.square().sum()
            + 0.1 * result.residual_covariance.square().sum()
        )

    loss = objective(positions)
    gradients = torch.autograd.grad(
        loss,
        (positions, timestamps, gravity, drag, anchor_time, weights),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0

    epsilon = 1.0e-6
    plus = base_positions.clone()
    minus = base_positions.clone()
    plus[0, 1, 0, 0] += epsilon
    minus[0, 1, 0, 0] -= epsilon
    finite_difference = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    torch.testing.assert_close(
        gradients[0][0, 1, 0, 0],
        finite_difference,
        rtol=2e-6,
        atol=2e-8,
    )


def test_default_anchor_is_final_provided_timestamp_without_support_selection() -> None:
    dtype = torch.float64
    timestamps = torch.tensor([[[-0.4, -0.3], [-0.2, -0.1], [0.1, 0.2]]], dtype=dtype)
    anchor = timestamps[:, -1]
    position = torch.tensor([[[0.3, 0.9, 0.1], [-0.2, 1.2, 0.4]]], dtype=dtype)
    velocity = torch.tensor([[[0.8, 0.1, -0.2], [0.3, -0.4, 0.5]]], dtype=dtype)
    gravity = torch.tensor([0.0, -9.81, 0.0], dtype=dtype)
    drag = torch.tensor([[0.03, 0.2]], dtype=dtype)
    positions = _oracle_history(position, velocity, timestamps, anchor, gravity, drag)
    support = torch.ones((1, 3, 2), dtype=torch.bool)
    support[:, -1] = False

    result = fit_free_motion(
        positions,
        timestamps,
        gravity=gravity,
        drag=drag,
        support=support,
    )

    torch.testing.assert_close(result.anchor_time, anchor)
    assert result.valid.all()
    torch.testing.assert_close(result.position, position, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(result.velocity, velocity, rtol=1e-11, atol=1e-11)


def test_duplicate_time_support_is_rejected_as_ill_conditioned() -> None:
    positions = torch.ones((1, 3, 1, 3), dtype=torch.float64)
    timestamps = torch.zeros((1, 3), dtype=torch.float64)
    result = fit_free_motion(
        positions,
        timestamps,
        gravity=torch.zeros(3, dtype=torch.float64),
        drag=torch.zeros((1, 1), dtype=torch.float64),
    )

    assert not result.valid.item()
    torch.testing.assert_close(result.position, torch.zeros_like(result.position))
    torch.testing.assert_close(result.velocity, torch.zeros_like(result.velocity))
    assert torch.isfinite(result.condition_number).all()
    assert result.condition_number.item() >= 1.0


def test_float32_two_second_semigroup_grid_meets_long_horizon_gate() -> None:
    dtype = torch.float32
    drag_values = torch.linspace(0.0, 1.0, 101, dtype=dtype)
    split_values = torch.linspace(0.0, 2.0, 101, dtype=dtype)
    drag_grid, split_grid = torch.meshgrid(drag_values, split_values, indexing="ij")
    batch = drag_grid.numel()
    drag = drag_grid.reshape(batch, 1)
    first_dt = split_grid.reshape(batch)
    second_dt = 2.0 - first_dt
    position = torch.tensor([0.2, 0.7, -0.1], dtype=dtype).expand(batch, 1, 3).clone()
    velocity = torch.tensor([1.1, -0.3, 0.2], dtype=dtype).expand(batch, 1, 3).clone()
    gravity = torch.tensor([0.0, -9.81, 0.1], dtype=dtype)

    first_position, first_velocity = free_motion_position_velocity(
        position,
        velocity,
        first_dt,
        gravity=gravity,
        drag=drag,
    )
    composed_position, composed_velocity = free_motion_position_velocity(
        first_position,
        first_velocity,
        second_dt,
        gravity=gravity,
        drag=drag,
    )
    direct_position, direct_velocity = free_motion_position_velocity(
        position,
        velocity,
        2.0,
        gravity=gravity,
        drag=drag,
    )

    assert float((composed_position - direct_position).abs().max()) <= 1.0e-5
    assert float((composed_velocity - direct_velocity).abs().max()) <= 1.0e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fit_rejects_unsupported_low_precision_normal_solve(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        fit_free_motion(
            torch.ones((1, 3, 1, 3), dtype=dtype),
            torch.tensor([[0.0, 0.1, 0.2]], dtype=dtype),
            gravity=torch.zeros(3, dtype=dtype),
            drag=torch.ones((1, 1), dtype=dtype),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "exception", "message"),
    [
        ("positions", torch.ones((1, 3, 1), dtype=torch.float64), ValueError, "positions"),
        ("positions", torch.ones((1, 3, 1, 3), dtype=torch.int64), TypeError, "positions"),
        ("timestamps", torch.ones((1, 3, 1, 1), dtype=torch.float64), ValueError, "timestamps"),
        ("gravity", torch.ones((1, 1), dtype=torch.float64), ValueError, "gravity"),
        ("drag", torch.ones((1,), dtype=torch.float64), ValueError, "drag"),
        ("drag", torch.tensor([[-0.1]], dtype=torch.float64), ValueError, "nonnegative"),
        ("support", torch.ones((1, 3, 1), dtype=torch.float64), TypeError, "boolean"),
        ("weights", torch.ones((1, 3), dtype=torch.float64), ValueError, "weights"),
        ("weights", -torch.ones((1, 3, 1), dtype=torch.float64), ValueError, "nonnegative"),
    ],
)
def test_fit_rejects_invalid_shapes_and_values(
    field: str,
    replacement: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    arguments = {
        "positions": torch.ones((1, 3, 1, 3), dtype=torch.float64),
        "timestamps": torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float64),
        "gravity": torch.zeros(3, dtype=torch.float64),
        "drag": torch.ones((1, 1), dtype=torch.float64),
        "support": None,
        "weights": None,
    }
    arguments[field] = replacement
    positions = arguments.pop("positions")
    timestamps = arguments.pop("timestamps")
    with pytest.raises(exception, match=message):
        fit_free_motion(positions, timestamps, **arguments)


@pytest.mark.parametrize("field", ["positions", "timestamps", "gravity", "drag", "weights"])
def test_fit_rejects_nonfinite_inputs_even_when_an_observation_is_masked(field: str) -> None:
    arguments = {
        "positions": torch.ones((1, 3, 1, 3), dtype=torch.float64),
        "timestamps": torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float64),
        "gravity": torch.zeros(3, dtype=torch.float64),
        "drag": torch.ones((1, 1), dtype=torch.float64),
        "support": torch.tensor([[[False], [True], [True]]]),
        "weights": torch.ones((1, 3, 1), dtype=torch.float64),
    }
    value = arguments[field].clone()
    value.reshape(-1)[0] = math.nan
    arguments[field] = value
    positions = arguments.pop("positions")
    timestamps = arguments.pop("timestamps")
    with pytest.raises(ValueError, match="finite"):
        fit_free_motion(positions, timestamps, **arguments)


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    [
        ("minimum_support", 1, ValueError),
        ("minimum_support", 2.0, TypeError),
        ("conditioning_limit", 1.0, ValueError),
        ("conditioning_limit", math.inf, ValueError),
    ],
)
def test_fit_rejects_invalid_solver_controls(
    keyword: str,
    value: float,
    exception: type[Exception],
) -> None:
    arguments = {
        "gravity": torch.zeros(3, dtype=torch.float64),
        "drag": torch.ones((1, 1), dtype=torch.float64),
        keyword: value,
    }
    with pytest.raises(exception):
        fit_free_motion(
            torch.ones((1, 3, 1, 3), dtype=torch.float64),
            torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float64),
            **arguments,
        )
