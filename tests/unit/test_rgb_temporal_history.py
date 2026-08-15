from __future__ import annotations

import math

import pytest
import torch

from world_model.belief import NUM_MOTION_MODES, BeliefFactory, MotionMode
from world_model.filtering import BeliefUpdater, BeliefUpdaterConfig
from world_model.fusion import AssociationResult
from world_model.observations import MeasurementSet
from world_model.observations.rgb import RGBObservationConfig, RGBObservationModule
from world_model.observations.rgb.temporal import RGBTemporalPositionHistory


def _empty_history(
    object_ids: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    history_size: int = 3,
    dtype: torch.dtype = torch.float64,
) -> RGBTemporalPositionHistory:
    if active_mask is None:
        active_mask = object_ids >= 0
    return RGBTemporalPositionHistory.empty(
        object_ids=object_ids,
        active_mask=active_mask,
        history_size=history_size,
        dtype=dtype,
    )


def _append(
    history: RGBTemporalPositionHistory,
    *,
    object_ids: torch.Tensor,
    timestamp: float,
    positions: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    observed_mask: torch.Tensor | None = None,
    observed_axis_mask: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
    scale_valid_mask: torch.Tensor | None = None,
    position_variance: float = 1.0e-4,
    minimum_dt: float = 1.0e-3,
) -> RGBTemporalPositionHistory:
    if active_mask is None:
        active_mask = object_ids >= 0
    if observed_mask is None:
        observed_mask = active_mask.clone()
    return history.append(
        object_ids=object_ids,
        active_mask=active_mask,
        observed_mask=observed_mask,
        observed_axis_mask=observed_axis_mask,
        scale_valid_mask=scale_valid_mask,
        reset_mask=reset_mask,
        timestamp=torch.tensor([timestamp], dtype=positions.dtype),
        positions=positions,
        position_log_variance=torch.full_like(
            positions,
            math.log(position_variance),
        ),
        minimum_dt=minimum_dt,
    )


def test_robust_point_scale_trajectory_rejects_one_depth_outlier() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids, history_size=5)
    velocity = torch.tensor([[[0.4, -0.2, 0.5]]], dtype=torch.float64)
    initial = torch.tensor([[[0.1, 0.3, 2.0]]], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1, 0.15, 0.2):
        position = initial + velocity * timestamp
        if timestamp == 0.15:
            position = position + torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=position,
            scale_valid_mask=torch.tensor([[True]]),
            position_variance=2.5e-3,
        )

    estimate, log_variance, valid = history.robust_trajectory_position(
        query_timestamp=torch.tensor([0.2], dtype=torch.float64),
        minimum_dt=1.0e-3,
        minimum_samples=3,
        robust_threshold=2.0,
        variance_scale=2.0,
        variance_floor=1.0e-4,
        variance_ceiling=0.1,
    )

    assert valid.item()
    expected = initial + velocity * 0.2
    torch.testing.assert_close(estimate[..., :2], expected[..., :2], atol=1.0e-10, rtol=0.0)
    assert abs(float(estimate[..., 2] - expected[..., 2])) < 0.12
    assert torch.all(log_variance.exp() <= 0.1)


def test_point_scale_trajectory_ignores_centre_only_samples() -> None:
    object_ids = torch.tensor([[9]])
    history = _empty_history(object_ids)
    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor([[[timestamp, 0.0, 2.0]]], dtype=torch.float64),
            scale_valid_mask=torch.tensor([[sample_index != 1]]),
        )

    _, _, valid = history.robust_trajectory_position(
        query_timestamp=torch.tensor([0.1], dtype=torch.float64),
        minimum_dt=1.0e-3,
        minimum_samples=3,
        robust_threshold=2.5,
        variance_scale=2.0,
        variance_floor=1.0e-4,
    )
    assert not valid.item()


def test_scale_anchor_ring_survives_interleaved_fast_centre_samples() -> None:
    object_ids = torch.tensor([[9]])
    history = _empty_history(object_ids, history_size=3)
    for sample_index in range(7):
        timestamp = 0.05 * sample_index
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor([[[timestamp, 0.0, 2.0 + timestamp]]], dtype=torch.float64),
            scale_valid_mask=torch.tensor([[sample_index % 3 == 0]]),
        )

    assert history.valid_mask.sum().item() == 3
    assert history.scale_valid_mask.sum().item() == 3
    estimate, _, valid = history.robust_trajectory_position(
        query_timestamp=torch.tensor([0.3], dtype=torch.float64),
        minimum_dt=1.0e-3,
        minimum_samples=3,
        robust_threshold=2.5,
        variance_scale=2.0,
        variance_floor=1.0e-4,
    )
    assert valid.item()
    torch.testing.assert_close(estimate[..., 2], torch.tensor([[2.3]], dtype=torch.float64))


def test_three_causal_samples_recover_constant_velocity_at_20_hz() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    initial_position = torch.tensor([[[0.4, -0.2, 1.1]]], dtype=torch.float64)
    expected_velocity = torch.tensor([[[1.5, -0.75, 0.25]]], dtype=torch.float64)

    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=initial_position + expected_velocity * timestamp,
        )
        velocity, log_variance, valid = history.least_squares_velocity(
            minimum_dt=1.0e-3,
            variance_scale=1.0,
            variance_floor=1.0e-3,
        )
        assert valid.item() is (sample_index == 2)

    torch.testing.assert_close(velocity, expected_velocity, rtol=0.0, atol=1.0e-12)
    # At 20 Hz, the three-point LS weights are [-10, 0, 10].
    expected_variance = torch.full_like(log_variance, 0.02)
    torch.testing.assert_close(log_variance.exp(), expected_variance)


def test_axis_local_fit_ignores_prior_copied_coordinates() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            # Only x is an RGB observation. The other coordinates deliberately
            # follow large prior-dependent trajectories that must not leak into
            # temporal evidence.
            positions=torch.tensor(
                [[[2.0 * timestamp, 100.0 * sample_index, -75.0 * sample_index]]],
                dtype=torch.float64,
            ),
            observed_axis_mask=torch.tensor([[[True, False, False]]]),
        )

    velocity, _, axis_valid = history.least_squares_velocity_axis_local(
        minimum_dt=1.0e-3,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert axis_valid.tolist() == [[[True, False, False]]]
    torch.testing.assert_close(
        velocity,
        torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    _, _, complete_valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert not complete_valid.item()
    assert history.position_axis_valid_mask.tolist() == [
        [
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
            ]
        ]
    ]


def test_legacy_complete_position_history_resolves_as_all_axis_support() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    for timestamp in (0.0, 0.05):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor(
                [[[timestamp, 2.0 * timestamp, -timestamp]]], dtype=torch.float64
            ),
        )
    del history.position_axis_valid_mask

    torch.testing.assert_close(
        history.resolved_position_axis_valid_mask(),
        history.valid_mask.unsqueeze(-1).expand_as(history.positions),
    )
    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.1,
        positions=torch.tensor([[[0.1, 0.2, -0.1]]], dtype=torch.float64),
    )
    assert history.position_axis_valid_mask.all()
    velocity, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert valid.item()
    torch.testing.assert_close(
        velocity,
        torch.tensor([[[1.0, 2.0, -1.0]]], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_known_acceleration_fit_estimates_velocity_at_query_time() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids, history_size=5)
    acceleration = torch.tensor([[0.0, -10.0, 0.0]], dtype=torch.float64)
    initial_velocity = torch.tensor([[[0.4, 2.0, -0.3]]], dtype=torch.float64)
    initial_position = torch.tensor([[[0.1, 0.4, 2.0]]], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1, 0.15, 0.2):
        position = (
            initial_position
            + initial_velocity * timestamp
            + 0.5 * acceleration[:, None] * timestamp**2
        )
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=position,
        )

    uncompensated, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-4,
    )
    compensated, _, compensated_valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-4,
        query_timestamp=torch.tensor([0.2], dtype=torch.float64),
        known_acceleration=acceleration,
    )

    expected_velocity = initial_velocity + acceleration[:, None] * 0.2
    assert valid.item() and compensated_valid.item()
    assert abs(float(uncompensated[..., 1] - expected_velocity[..., 1])) > 0.9
    torch.testing.assert_close(compensated, expected_velocity, atol=1.0e-10, rtol=0.0)


def test_gravity_aware_fit_recovers_current_velocity_and_exact_diagonal() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids, history_size=5)
    acceleration = torch.tensor([[0.0, -9.0, 0.0]], dtype=torch.float64)
    initial_velocity = torch.tensor([[[1.25, 2.4, -0.75]]], dtype=torch.float64)
    initial_position = torch.tensor([[[0.2, 0.6, 1.7]]], dtype=torch.float64)
    sample_variance = torch.tensor([1.0e-4, 4.0e-4, 9.0e-4], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1, 0.15, 0.2):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=(
                initial_position
                + initial_velocity * timestamp
                + 0.5 * acceleration[:, None] * timestamp**2
            ),
        )
    history.position_log_variance = (
        sample_variance.log().view(1, 1, 1, 3).expand_as(history.positions)
    )

    velocity, log_variance, axis_valid = history.gravity_aware_least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-8,
        query_timestamp=torch.tensor([0.2], dtype=torch.float64),
        known_acceleration=acceleration,
    )

    expected_velocity = initial_velocity + acceleration[:, None] * 0.2
    assert axis_valid.all()
    torch.testing.assert_close(velocity, expected_velocity, atol=1.0e-10, rtol=0.0)
    # Five equally spaced samples have LS weights [-4,-2,0,2,4], so
    # acceleration subtraction changes the mean but not its measurement noise.
    torch.testing.assert_close(
        log_variance.exp(),
        (40.0 * sample_variance).view(1, 1, 3),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_gravity_aware_fit_is_rotation_coherent_and_projects_variance() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids, history_size=5)
    gravity_axis = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    gravity_axis = torch.nn.functional.normalize(gravity_axis, dim=-1)
    acceleration = 7.0 * gravity_axis
    lateral_axis = torch.tensor([[0.8, 0.2, 0.55]], dtype=torch.float64)
    lateral_axis = torch.nn.functional.normalize(lateral_axis, dim=-1)
    initial_velocity = torch.tensor([[[0.7, 1.1, -0.4]]], dtype=torch.float64)
    reference_velocity = torch.tensor([[[-0.2, 0.5, 0.9]]], dtype=torch.float64)
    sample_variance = torch.tensor([2.0e-4, 5.0e-4, 8.0e-4], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1, 0.15, 0.2):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=(initial_velocity * timestamp + 0.5 * acceleration[:, None] * timestamp**2),
        )
    history.position_log_variance = (
        sample_variance.log().view(1, 1, 1, 3).expand_as(history.positions)
    )

    velocity, log_variance, axis_valid = history.gravity_aware_least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-8,
        query_timestamp=torch.tensor([0.2], dtype=torch.float64),
        known_acceleration=acceleration,
        orthogonal_axis=lateral_axis,
        reference_velocity=reference_velocity,
    )

    lateral_orthogonal = lateral_axis - gravity_axis * (lateral_axis * gravity_axis).sum(
        dim=-1, keepdim=True
    )
    lateral_orthogonal = torch.nn.functional.normalize(lateral_orthogonal, dim=-1)
    projection = torch.einsum("bi,bj->bij", lateral_orthogonal, lateral_orthogonal) + torch.einsum(
        "bi,bj->bij", gravity_axis, gravity_axis
    )
    current_velocity = initial_velocity + acceleration[:, None] * 0.2
    expected_velocity = reference_velocity + torch.einsum(
        "bij,bnj->bni",
        projection,
        current_velocity - reference_velocity,
    )
    ls_variance = 40.0 * sample_variance
    expected_variance = torch.einsum(
        "bij,bj->bi",
        projection.square(),
        ls_variance.unsqueeze(0),
    ).unsqueeze(1)
    assert axis_valid.all()
    torch.testing.assert_close(velocity, expected_velocity, atol=1.0e-10, rtol=0.0)
    torch.testing.assert_close(
        log_variance.exp(),
        expected_variance,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_gravity_aware_fit_fails_closed_for_partial_rotated_support() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    acceleration = torch.tensor([[4.0, -4.0, 0.0]], dtype=torch.float64)
    initial_velocity = torch.tensor([[[0.5, 1.0, -0.25]]], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=(initial_velocity * timestamp + 0.5 * acceleration[:, None] * timestamp**2),
            observed_axis_mask=torch.tensor([[[True, False, True]]]),
        )

    velocity, _, axis_valid = history.gravity_aware_least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-6,
        query_timestamp=torch.tensor([0.1], dtype=torch.float64),
        known_acceleration=acceleration,
        reference_velocity=torch.tensor([[[-3.0, 2.0, 5.0]]], dtype=torch.float64),
    )

    assert axis_valid.tolist() == [[[False, False, True]]]
    torch.testing.assert_close(
        velocity,
        torch.tensor([[[-3.0, 2.0, -0.25]]], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_gravity_aware_fit_broadcasts_batch_queries_across_different_object_count() -> None:
    object_ids = torch.tensor([[11, 12, 13], [21, 22, 23]])
    history = _empty_history(object_ids, history_size=5)
    acceleration = torch.tensor(
        [[0.0, -9.0, 0.0], [2.0, -4.0, 1.0]],
        dtype=torch.float64,
    )
    initial_velocity = torch.tensor(
        [
            [[0.4, 2.0, -0.3], [0.1, 1.0, 0.2], [-0.2, 0.5, 0.7]],
            [[-0.5, 1.5, 0.4], [0.8, -0.2, -0.1], [0.3, 0.9, -0.6]],
        ],
        dtype=torch.float64,
    )
    initial_position = torch.tensor(
        [
            [[0.1, 0.4, 2.0], [0.3, 0.2, 2.2], [-0.2, 0.5, 1.8]],
            [[0.6, 0.1, 2.4], [-0.4, 0.7, 2.1], [0.2, -0.1, 1.9]],
        ],
        dtype=torch.float64,
    )
    for base_timestamp in (0.0, 0.05, 0.1, 0.15, 0.2):
        timestamp = torch.tensor(
            [base_timestamp, base_timestamp + 0.03],
            dtype=torch.float64,
        )
        positions = (
            initial_position
            + initial_velocity * timestamp[:, None, None]
            + 0.5 * acceleration[:, None, :] * timestamp[:, None, None].square()
        )
        history = history.append(
            object_ids=object_ids,
            active_mask=torch.ones_like(object_ids, dtype=torch.bool),
            observed_mask=torch.ones_like(object_ids, dtype=torch.bool),
            scale_valid_mask=torch.ones_like(object_ids, dtype=torch.bool),
            timestamp=timestamp,
            positions=positions,
            position_log_variance=torch.full_like(positions, math.log(1.0e-4)),
            minimum_dt=1.0e-3,
        )

    query_timestamp = torch.tensor([0.2, 0.23], dtype=torch.float64)
    velocity, _, axis_valid = history.gravity_aware_least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-6,
        query_timestamp=query_timestamp,
        known_acceleration=acceleration,
    )

    expected = initial_velocity + acceleration[:, None, :] * query_timestamp[:, None, None]
    assert axis_valid.all()
    torch.testing.assert_close(velocity, expected, atol=1.0e-10, rtol=0.0)


def test_gravity_aware_fit_with_zero_gravity_matches_ordinary_fit() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    expected_velocity = torch.tensor([[[1.5, -0.75, 0.25]]], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=expected_velocity * timestamp,
        )

    ordinary = history.least_squares_velocity_axis_local(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-6,
    )
    gravity_aware = history.gravity_aware_least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-6,
        query_timestamp=torch.tensor([0.1], dtype=torch.float64),
        known_acceleration=torch.zeros(1, 3, dtype=torch.float64),
    )

    for actual, expected in zip(gravity_aware, ordinary, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_gravity_aware_fit_has_finite_position_gradients() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    acceleration = torch.tensor([[0.0, -9.81, 0.0]], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor(
                [[[timestamp, 0.5 * timestamp - 4.905 * timestamp**2, -timestamp]]],
                dtype=torch.float64,
            ),
        )
    history.positions = history.positions.detach().requires_grad_(True)

    velocity, _, axis_valid = history.gravity_aware_least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-6,
        query_timestamp=torch.tensor([0.1], dtype=torch.float64),
        known_acceleration=acceleration,
    )
    velocity[axis_valid].sum().backward()

    assert history.positions.grad is not None
    assert torch.isfinite(history.positions.grad).all()
    assert torch.count_nonzero(history.positions.grad) > 0


def test_known_acceleration_fit_broadcasts_query_per_batch_not_per_object() -> None:
    object_ids = torch.tensor([[11, 12, 13], [21, 22, 23]])
    history = _empty_history(object_ids, history_size=5)
    acceleration = torch.tensor(
        [[0.0, -10.0, 0.0], [1.0, -4.0, 0.5]],
        dtype=torch.float64,
    )
    initial_velocity = torch.tensor(
        [
            [[0.4, 2.0, -0.3], [0.1, 1.0, 0.2], [-0.2, 0.5, 0.7]],
            [[-0.5, 1.5, 0.4], [0.8, -0.2, -0.1], [0.3, 0.9, -0.6]],
        ],
        dtype=torch.float64,
    )
    initial_position = torch.tensor(
        [
            [[0.1, 0.4, 2.0], [0.3, 0.2, 2.2], [-0.2, 0.5, 1.8]],
            [[0.6, 0.1, 2.4], [-0.4, 0.7, 2.1], [0.2, -0.1, 1.9]],
        ],
        dtype=torch.float64,
    )
    for base_timestamp in (0.0, 0.05, 0.1, 0.15, 0.2):
        timestamp = torch.tensor(
            [base_timestamp, base_timestamp + 0.03],
            dtype=torch.float64,
        )
        positions = (
            initial_position
            + initial_velocity * timestamp[:, None, None]
            + 0.5 * acceleration[:, None, :] * timestamp[:, None, None].square()
        )
        history = history.append(
            object_ids=object_ids,
            active_mask=torch.ones_like(object_ids, dtype=torch.bool),
            observed_mask=torch.ones_like(object_ids, dtype=torch.bool),
            scale_valid_mask=torch.ones_like(object_ids, dtype=torch.bool),
            reset_mask=None,
            timestamp=timestamp,
            positions=positions,
            position_log_variance=torch.full_like(positions, math.log(1.0e-4)),
            minimum_dt=1.0e-3,
        )

    query_timestamp = torch.tensor([0.2, 0.23], dtype=torch.float64)
    estimate, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=3,
        variance_scale=1.0,
        variance_floor=1.0e-4,
        query_timestamp=query_timestamp,
        known_acceleration=acceleration,
    )

    expected = initial_velocity + acceleration[:, None, :] * query_timestamp[:, None, None]
    assert valid.all()
    torch.testing.assert_close(estimate, expected, atol=1.0e-10, rtol=0.0)


def test_kinematic_change_point_rejects_smooth_ballistic_motion() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    acceleration = torch.tensor([[0.0, -10.0, 0.0]], dtype=torch.float64)
    initial_velocity = torch.tensor([[[0.5, 2.0, 0.0]]], dtype=torch.float64)
    for timestamp in (0.0, 0.05, 0.1):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=(initial_velocity * timestamp + 0.5 * acceleration[:, None] * timestamp**2),
        )

    axes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        dtype=torch.float64,
    )
    changed, score = history.kinematic_change_point(
        observable_axes=axes,
        known_acceleration=acceleration,
        minimum_dt=1.0e-3,
        minimum_speed=0.25,
        minimum_velocity_change=0.75,
        strong_velocity_change=2.0,
    )

    assert not changed.item()
    assert score.item() < 1.0e-10


def test_kinematic_change_point_detects_observable_velocity_reversal() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    for timestamp, vertical_position in ((0.0, 0.0), (0.05, -0.05), (0.1, 0.0)):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor(
                [[[0.0, vertical_position, 0.0]]],
                dtype=torch.float64,
            ),
        )

    axes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        dtype=torch.float64,
    )
    changed, score = history.kinematic_change_point(
        observable_axes=axes,
        known_acceleration=torch.tensor([[0.0, -10.0, 0.0]], dtype=torch.float64),
        minimum_dt=1.0e-3,
        minimum_speed=0.25,
        minimum_velocity_change=0.75,
        strong_velocity_change=2.0,
    )

    assert changed.item()
    assert score.item() > 2.0


def test_kinematic_change_point_ignores_monocular_depth_jump() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    for timestamp, depth in ((0.0, 2.0), (0.05, 2.0), (0.1, 3.0)):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor([[[0.0, 0.0, depth]]], dtype=torch.float64),
        )

    axes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        dtype=torch.float64,
    )
    changed, score = history.kinematic_change_point(
        observable_axes=axes,
        known_acceleration=torch.zeros(1, 3, dtype=torch.float64),
        minimum_dt=1.0e-3,
        minimum_speed=0.25,
        minimum_velocity_change=0.75,
        strong_velocity_change=2.0,
    )

    assert not changed.item()
    assert score.item() == 0.0


def test_change_point_features_standardize_by_rgb_position_uncertainty() -> None:
    object_ids = torch.tensor([[17]])
    low_uncertainty = _empty_history(object_ids)
    high_uncertainty = _empty_history(object_ids)
    for timestamp, vertical_position in ((0.0, 0.0), (0.05, -0.05), (0.1, 0.0)):
        position = torch.tensor([[[0.0, vertical_position, 0.0]]], dtype=torch.float64)
        low_uncertainty = _append(
            low_uncertainty,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=position,
            position_variance=1.0e-4,
        )
        high_uncertainty = _append(
            high_uncertainty,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=position,
            position_variance=0.1,
        )

    axis = torch.tensor([[[0.0], [1.0], [0.0]]], dtype=torch.float64)
    low_features, low_valid = low_uncertainty.kinematic_change_point_features(
        observable_axes=axis,
        known_acceleration=torch.tensor([[0.0, -10.0, 0.0]], dtype=torch.float64),
        minimum_dt=1.0e-3,
    )
    high_features, high_valid = high_uncertainty.kinematic_change_point_features(
        observable_axes=axis,
        known_acceleration=torch.tensor([[0.0, -10.0, 0.0]], dtype=torch.float64),
        minimum_dt=1.0e-3,
    )

    assert low_valid.item() and high_valid.item()
    torch.testing.assert_close(low_features[..., 0], high_features[..., 0])
    assert low_features[..., 2].item() > high_features[..., 2].item()


def test_two_sample_mode_emits_earliest_causal_velocity() -> None:
    object_ids = torch.tensor([[17]])
    history = _empty_history(object_ids)
    expected_velocity = torch.tensor([[[0.8, -0.2, 0.0]]], dtype=torch.float64)

    for sample_index, timestamp in enumerate((0.0, 0.05)):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=expected_velocity * timestamp,
        )
        velocity, _, valid = history.least_squares_velocity(
            minimum_dt=1.0e-3,
            minimum_samples=2,
            variance_scale=1.0,
            variance_floor=1.0e-3,
        )
        assert valid.item() is (sample_index == 1)

    torch.testing.assert_close(velocity, expected_velocity, rtol=0.0, atol=1.0e-12)


def test_reset_mask_discards_pre_event_motion_before_append() -> None:
    object_ids = torch.tensor([[7]])
    history = _empty_history(object_ids)
    for timestamp in (0.0, 0.05):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor([[[timestamp, 0.0, 0.0]]], dtype=torch.float64),
        )

    history = history.append(
        object_ids=object_ids,
        active_mask=torch.tensor([[True]]),
        observed_mask=torch.tensor([[True]]),
        reset_mask=torch.tensor([[True]]),
        timestamp=torch.tensor([0.1], dtype=torch.float64),
        positions=torch.tensor([[[0.0, 0.5, 0.0]]], dtype=torch.float64),
        position_log_variance=torch.full((1, 1, 3), math.log(1.0e-4), dtype=torch.float64),
        minimum_dt=1.0e-3,
    )

    assert history.valid_mask.sum().item() == 1
    assert history.has_reset.item()
    assert history.post_reset_sample_count.item() == 1
    torch.testing.assert_close(
        history.positions[0, 0, 0],
        torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64),
    )


def test_point_reset_can_preserve_independent_scale_anchor_ring() -> None:
    object_ids = torch.tensor([[7]])
    history = _empty_history(object_ids)
    for timestamp in (0.0, 0.05, 0.1):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor([[[timestamp, 0.0, 2.0]]], dtype=torch.float64),
            scale_valid_mask=torch.tensor([[True]]),
        )

    history = history.append(
        object_ids=object_ids,
        active_mask=torch.tensor([[True]]),
        observed_mask=torch.tensor([[True]]),
        scale_valid_mask=torch.tensor([[False]]),
        reset_mask=torch.tensor([[True]]),
        scale_reset_mask=torch.tensor([[False]]),
        timestamp=torch.tensor([0.15], dtype=torch.float64),
        positions=torch.tensor([[[0.15, 0.0, 2.0]]], dtype=torch.float64),
        position_log_variance=torch.full(
            (1, 1, 3),
            math.log(1.0e-4),
            dtype=torch.float64,
        ),
        minimum_dt=1.0e-3,
    )

    assert history.valid_mask.sum().item() == 1
    assert history.scale_valid_mask.sum().item() == 3
    assert history.has_reset.item()
    assert not history.scale_reset_active.item()


def test_post_reset_segment_count_tracks_only_accepted_samples() -> None:
    object_ids = torch.tensor([[7]])
    history = _empty_history(object_ids)
    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.0,
        positions=torch.zeros(1, 1, 3, dtype=torch.float64),
    )
    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.05,
        positions=torch.tensor([[[0.1, 0.0, 0.0]]], dtype=torch.float64),
        reset_mask=torch.tensor([[True]]),
    )
    assert history.post_reset_sample_count.item() == 1

    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.05,
        positions=torch.tensor([[[99.0, 0.0, 0.0]]], dtype=torch.float64),
    )
    assert history.post_reset_sample_count.item() == 1

    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.1,
        positions=torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float64),
    )
    assert history.post_reset_sample_count.item() == 2
    velocity, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=2,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert valid.item()
    torch.testing.assert_close(
        velocity,
        torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float64),
    )


def test_sustained_collision_mode_resets_once_and_learns_outgoing_velocity() -> None:
    object_ids = torch.tensor([[7]])
    history = _empty_history(object_ids)
    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.0,
        positions=torch.zeros(1, 1, 3, dtype=torch.float64),
    )
    collision_active = torch.tensor([[True]])
    for timestamp, x_position in ((0.05, -0.05), (0.10, -0.15)):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=torch.tensor(
                [[[x_position, 0.0, 0.0]]],
                dtype=torch.float64,
            ),
            reset_mask=collision_active,
        )

    assert history.valid_mask.sum().item() == 2
    assert history.post_reset_sample_count.item() == 2
    assert history.reset_active.item()
    velocity, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        minimum_samples=2,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert valid.item()
    torch.testing.assert_close(
        velocity,
        torch.tensor([[[-2.0, 0.0, 0.0]]], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )

    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.15,
        positions=torch.tensor([[[-0.25, 0.0, 0.0]]], dtype=torch.float64),
        reset_mask=torch.tensor([[False]]),
    )
    assert not history.reset_active.item()


def test_nonmonotonic_and_too_close_timestamps_are_skipped() -> None:
    object_ids = torch.tensor([[4]])
    history = _empty_history(object_ids, history_size=4)
    velocity = torch.tensor([[[2.0, 0.5, -1.0]]], dtype=torch.float64)

    for timestamp in (1.0, 1.05):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=velocity * timestamp,
        )

    accepted_timestamps = history.timestamps.clone()
    for rejected_timestamp in (1.05, 1.0505, 0.9):
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=rejected_timestamp,
            positions=velocity * rejected_timestamp,
        )
        assert history.valid_mask.sum().item() == 2
        torch.testing.assert_close(history.timestamps, accepted_timestamps)

    history = _append(
        history,
        object_ids=object_ids,
        timestamp=1.1,
        positions=velocity * 1.1,
    )
    estimate, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert valid.item()
    torch.testing.assert_close(estimate, velocity, rtol=0.0, atol=1.0e-12)


def test_history_follows_ids_across_reorder_birth_death_and_explicit_reset() -> None:
    object_ids = torch.tensor([[10, 20]])
    history = _empty_history(object_ids)
    velocities_by_id = {
        10: torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        20: torch.tensor([0.0, -2.0, 0.5], dtype=torch.float64),
    }

    for timestamp in (0.0, 0.05):
        positions = torch.stack(
            [velocities_by_id[10] * timestamp, velocities_by_id[20] * timestamp]
        ).unsqueeze(0)
        history = _append(
            history,
            object_ids=object_ids,
            timestamp=timestamp,
            positions=positions,
        )

    reordered_ids = torch.tensor([[20, 10]])
    reordered_positions = torch.stack(
        [velocities_by_id[20] * 0.1, velocities_by_id[10] * 0.1]
    ).unsqueeze(0)
    history = _append(
        history,
        object_ids=reordered_ids,
        timestamp=0.1,
        positions=reordered_positions,
    )
    estimate, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert valid.tolist() == [[True, True]]
    torch.testing.assert_close(
        estimate,
        torch.stack([velocities_by_id[20], velocities_by_id[10]]).unsqueeze(0),
        rtol=0.0,
        atol=1.0e-12,
    )

    ids_with_birth = torch.tensor([[20, 30]])
    history = _append(
        history,
        object_ids=ids_with_birth,
        timestamp=0.15,
        positions=torch.tensor([[[0.0, -0.3, 0.075], [4.0, 1.0, 0.0]]], dtype=torch.float64),
    )
    assert history.object_ids.tolist() == [[20, 30]]
    assert history.valid_mask.sum(dim=-1).tolist() == [[3, 1]]
    _, _, valid = history.least_squares_velocity(
        minimum_dt=1.0e-3,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert valid.tolist() == [[True, False]]

    active_after_death = torch.tensor([[False, True]])
    history = _append(
        history,
        object_ids=ids_with_birth,
        active_mask=active_after_death,
        observed_mask=active_after_death,
        timestamp=0.2,
        positions=torch.tensor([[[99.0, 99.0, 99.0], [4.1, 1.0, 0.0]]], dtype=torch.float64),
    )
    assert history.object_ids.tolist() == [[-1, 30]]
    assert history.valid_mask.sum(dim=-1).tolist() == [[0, 2]]

    # Once an ID has left the active set its samples are discarded. Even if an
    # upstream lifecycle later reuses that integer, the new track starts fresh.
    reused_ids = torch.tensor([[20, 30]])
    history = _append(
        history,
        object_ids=reused_ids,
        timestamp=0.25,
        positions=torch.tensor([[[7.0, 7.0, 7.0], [4.2, 1.0, 0.0]]], dtype=torch.float64),
    )
    assert history.valid_mask.sum(dim=-1).tolist() == [[1, 3]]
    torch.testing.assert_close(
        history.timestamps[0, 0, 0],
        torch.tensor(0.25, dtype=torch.float64),
    )

    reset = _empty_history(
        ids_with_birth,
        active_mask=active_after_death,
        history_size=history.history_size,
    )
    assert reset.object_ids.tolist() == [[-1, 30]]
    assert not reset.valid_mask.any()
    estimate, _, valid = reset.least_squares_velocity(
        minimum_dt=1.0e-3,
        variance_scale=1.0,
        variance_floor=1.0e-3,
    )
    assert not valid.any()
    assert not estimate.any()


def test_axis_support_follows_identity_and_does_not_survive_id_reuse() -> None:
    object_ids = torch.tensor([[10, 20]])
    history = _empty_history(object_ids)
    history = _append(
        history,
        object_ids=object_ids,
        timestamp=0.0,
        positions=torch.zeros(1, 2, 3, dtype=torch.float64),
        observed_axis_mask=torch.tensor([[[True, False, False], [False, True, False]]]),
    )
    reordered_ids = torch.tensor([[20, 10]])
    history = _append(
        history,
        object_ids=reordered_ids,
        timestamp=0.05,
        positions=torch.ones(1, 2, 3, dtype=torch.float64),
        observed_axis_mask=torch.tensor([[[False, True, False], [True, False, False]]]),
    )
    assert history.position_axis_valid_mask[0, 0, :2].tolist() == [
        [False, True, False],
        [False, True, False],
    ]
    assert history.position_axis_valid_mask[0, 1, :2].tolist() == [
        [True, False, False],
        [True, False, False],
    ]

    history = _append(
        history,
        object_ids=reordered_ids,
        active_mask=torch.tensor([[False, True]]),
        observed_mask=torch.tensor([[False, True]]),
        observed_axis_mask=torch.tensor([[[False, False, False], [True, False, False]]]),
        timestamp=0.1,
        positions=torch.ones(1, 2, 3, dtype=torch.float64),
    )
    history = _append(
        history,
        object_ids=reordered_ids,
        timestamp=0.15,
        positions=torch.ones(1, 2, 3, dtype=torch.float64),
        observed_axis_mask=torch.tensor([[[False, False, True], [True, False, False]]]),
    )
    assert history.valid_mask.sum(dim=-1).tolist() == [[1, 3]]
    assert history.position_axis_valid_mask[0, 0, 0].tolist() == [False, False, True]
    assert not history.position_axis_valid_mask[0, 0, 1:].any()


def test_detach_removes_temporal_history_graph() -> None:
    object_ids = torch.tensor([[8]])
    history = _empty_history(object_ids)
    positions = torch.tensor(
        [[[0.1, 0.2, 0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    log_variance = torch.full(
        (1, 1, 3),
        math.log(1.0e-3),
        dtype=torch.float64,
        requires_grad=True,
    )
    history = history.append(
        object_ids=object_ids,
        active_mask=torch.tensor([[True]]),
        observed_mask=torch.tensor([[True]]),
        timestamp=torch.tensor([0.0], dtype=torch.float64, requires_grad=True),
        positions=positions,
        position_log_variance=log_variance,
        minimum_dt=1.0e-3,
    )
    assert history.positions.requires_grad
    assert history.position_log_variance.requires_grad
    assert history.timestamps.requires_grad

    detached = history.detach()
    assert detached.history_size == history.history_size
    assert not detached.object_ids.requires_grad
    assert not detached.timestamps.requires_grad
    assert not detached.positions.requires_grad
    assert not detached.position_log_variance.requires_grad
    assert not detached.valid_mask.requires_grad
    assert not detached.position_axis_valid_mask.requires_grad
    assert not detached.post_reset_sample_count.requires_grad
    assert not detached.has_reset.requires_grad
    assert detached.timestamps.grad_fn is None
    assert detached.positions.grad_fn is None
    assert detached.position_log_variance.grad_fn is None


@pytest.mark.parametrize("legacy_blend", (0.0, 0.125, 1.0))
def test_rgb_module_velocity_uses_raw_measurements_not_posterior_positions(
    legacy_blend: float,
) -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    belief = factory.create().replace(
        objects=factory.create().objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            position=torch.zeros(1, 1, 3),
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_variance_scale=1.0,
            temporal_velocity_variance_floor=0.01,
            temporal_velocity_variance_ceiling=2.0,
            temporal_velocity_independent_raw_history_enabled=True,
            temporal_velocity_measurement_position_blend=legacy_blend,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    expected_velocity = torch.tensor([[[1.0, -0.5, 0.25]]])
    raw_log_variance = torch.tensor([[[math.log(0.01), math.log(0.02), math.log(0.03)]]])
    history = None
    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        current = belief.replace(
            objects=belief.objects.replace(
                position=torch.tensor([[[100.0 + 10.0 * timestamp, -50.0, 20.0]]])
            )
        )
        raw_position = expected_velocity * timestamp
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position", "geometry", "appearance"),
            auxiliary={
                "world_position": raw_position,
                "world_position_log_variance": raw_log_variance,
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=current,
            measured=measured,
            association=association,
            history=history,
        )
        assert evidence is not None
        assert evidence.valid_mask.item() is (sample_index == 2)
        assert measured.auxiliary["world_velocity_valid_mask"].item() is (sample_index == 2)

    assert evidence is not None
    assert isinstance(history, RGBTemporalPositionHistory)
    torch.testing.assert_close(evidence.velocity, expected_velocity)
    expected_positions = torch.stack(
        [expected_velocity[0, 0] * timestamp for timestamp in (0.0, 0.05, 0.1)]
    )
    torch.testing.assert_close(history.positions[0, 0], expected_positions)
    torch.testing.assert_close(
        history.position_log_variance[0, 0],
        raw_log_variance[0, 0].expand(3, -1),
    )
    assert torch.all(evidence.log_variance.exp() <= 2.0)
    torch.testing.assert_close(
        measured.auxiliary["world_velocity"],
        expected_velocity,
    )


def test_legacy_temporal_blend_remains_functional_until_raw_mode_is_enabled() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    posterior = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            position=torch.tensor([[[10.0, 20.0, 30.0]]]),
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_measurement_position_blend=0.25,
        )
    )
    measured = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([0.0]),
        values=torch.zeros(1, 1, 7),
        log_variance=torch.zeros(1, 1, 7),
        existence_logits=torch.tensor([[8.0]]),
        measurement_mask=torch.tensor([[True]]),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera",
        supported_state_fields=("position",),
        auxiliary={
            "world_position": torch.tensor([[[2.0, 4.0, 6.0]]]),
            "world_position_log_variance": torch.full((1, 1, 3), math.log(0.04)),
            "world_position_independent_axis_mask": torch.zeros(1, 1, 3, dtype=torch.bool),
        },
        source_belief_indices=torch.tensor([[0]]),
        source_object_ids=torch.tensor([[12]]),
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )

    _, history = module.update_temporal_history(
        posterior=posterior,
        measured=measured,
        association=association,
        history=None,
    )

    assert isinstance(history, RGBTemporalPositionHistory)
    torch.testing.assert_close(history.positions[0, 0, 0], torch.tensor([8.0, 16.0, 24.0]))
    assert history.position_axis_valid_mask[0, 0, 0].all()


def test_prior_copied_fast_roi_rows_never_create_temporal_support() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_variance_floor=0.01,
            temporal_velocity_independent_raw_history_enabled=True,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    for timestamp in (0.0, 0.05, 0.1):
        copied_prior = torch.tensor([[[10.0 * timestamp, -4.0 * timestamp, 2.0]]])
        belief = base.replace(
            objects=base.objects.replace(
                active=torch.tensor([[True]]),
                object_id=torch.tensor([[12]]),
                position=copied_prior,
                existence_logit=torch.tensor([[8.0]]),
            )
        )
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": copied_prior,
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(0.01),
                ),
                "world_position_independent_axis_mask": torch.zeros(
                    1,
                    1,
                    3,
                    dtype=torch.bool,
                ),
            },
            source_belief_indices=torch.tensor([[0]]),
            source_object_ids=torch.tensor([[12]]),
        )
        measured.validate()
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )
        assert evidence is not None
        assert not evidence.valid_mask.item()
        assert evidence.axis_valid_mask is not None and not evidence.axis_valid_mask.any()

    assert isinstance(history, RGBTemporalPositionHistory)
    assert not history.valid_mask.any()
    assert not history.position_axis_valid_mask.any()


@pytest.mark.parametrize(
    ("axis_mask", "error"),
    (
        (torch.ones(1, 1, 2, dtype=torch.bool), "shape"),
        (torch.ones(1, 1, 3), "torch.bool"),
    ),
)
def test_measurement_contract_validates_position_independence_axis_mask(
    axis_mask: torch.Tensor,
    error: str,
) -> None:
    measured = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([0.0]),
        values=torch.zeros(1, 1, 7),
        log_variance=torch.zeros(1, 1, 7),
        existence_logits=torch.zeros(1, 1),
        measurement_mask=torch.tensor([[True]]),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera",
        supported_state_fields=("position",),
        auxiliary={"world_position_independent_axis_mask": axis_mask},
    )
    with pytest.raises((TypeError, ValueError), match=error):
        measured.validate()


def test_fast_roi_independent_lateral_axis_produces_only_lateral_velocity() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_variance_floor=0.01,
            temporal_velocity_independent_raw_history_enabled=True,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    evidence = None
    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        raw_position = torch.tensor(
            [[[2.0 * timestamp, 50.0 * sample_index, -25.0 * sample_index]]]
        )
        belief = base.replace(
            objects=base.objects.replace(
                active=torch.tensor([[True]]),
                object_id=torch.tensor([[12]]),
                position=torch.tensor([[[100.0, -100.0, 20.0]]]),
                velocity=torch.tensor([[[0.0, 3.0, -4.0]]]),
                existence_logit=torch.tensor([[8.0]]),
            )
        )
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": raw_position,
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(0.01),
                ),
                "world_position_independent_axis_mask": torch.tensor([[[True, False, False]]]),
            },
            source_belief_indices=torch.tensor([[0]]),
            source_object_ids=torch.tensor([[12]]),
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )

    assert evidence is not None and evidence.valid_mask.item()
    assert evidence.axis_valid_mask is not None
    assert evidence.axis_valid_mask.tolist() == [[[True, False, False]]]
    torch.testing.assert_close(evidence.velocity, torch.tensor([[[2.0, 3.0, -4.0]]]))
    assert isinstance(history, RGBTemporalPositionHistory)
    assert history.position_axis_valid_mask[..., 0].all()
    assert not history.position_axis_valid_mask[..., 1:].any()


def test_rgb_module_missing_or_ambiguous_raw_rows_do_not_create_velocity_support() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            position=torch.tensor([[[100.0, -50.0, 20.0]]]),
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_reset_on_collision=True,
            temporal_velocity_independent_raw_history_enabled=True,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )

    history = None
    cases = (
        {
            "world_position": torch.zeros(1, 1, 3),
            "world_position_log_variance": torch.full((1, 1, 3), math.log(0.01)),
            "structured_depth_valid": torch.tensor([[True]]),
        },
        {},
        {
            "world_position": torch.tensor([[[0.1, 0.0, 0.0]]]),
            "world_position_log_variance": torch.full((1, 1, 3), math.log(0.01)),
            "structured_centre_ambiguous": torch.tensor([[True]]),
            "structured_depth_valid": torch.tensor([[True]]),
        },
        {
            "world_position": torch.tensor([[[0.15, 0.0, 0.0]]]),
            "world_position_log_variance": torch.full((1, 1, 3), math.log(0.01)),
            "structured_depth_valid": torch.tensor([[True]]),
        },
        {"prior_interval_collision_mask": torch.tensor([[True]])},
    )
    for sample_index, auxiliary in enumerate(cases):
        if sample_index == 3:
            association.ambiguous.fill_(True)
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([0.05 * sample_index]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary=auxiliary,
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )
        assert evidence is not None and not evidence.valid_mask.item()
        assert not measured.auxiliary["world_velocity_valid_mask"].item()
        if sample_index == 3:
            assert isinstance(history, RGBTemporalPositionHistory)
            assert history.valid_mask.sum().item() == 1
            assert history.scale_valid_mask.sum().item() == 1
            torch.testing.assert_close(history.positions[0, 0, 0], torch.zeros(3))

    assert isinstance(history, RGBTemporalPositionHistory)
    assert history.has_reset.item()
    assert history.valid_mask.sum().item() == 0
    assert history.scale_valid_mask.sum().item() == 0
    assert history.post_reset_sample_count.item() == 0


def test_rgb_history_resets_on_prior_interval_collision_not_only_endpoint_mode() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_reset_on_collision=True,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    for timestamp in (0.0, 0.05, 0.1):
        current = belief.replace(
            objects=belief.objects.replace(position=torch.tensor([[[timestamp, 0.0, 0.0]]]))
        )
        auxiliary = {}
        if timestamp == 0.1:
            auxiliary["prior_interval_collision_mask"] = torch.tensor([[True]])
        auxiliary["world_position"] = torch.tensor([[[timestamp, 0.0, 0.0]]])
        auxiliary["world_position_log_variance"] = torch.full(
            (1, 1, 3),
            math.log(0.01),
        )
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary=auxiliary,
        )
        _, history = module.update_temporal_history(
            posterior=current,
            measured=measured,
            association=association,
            history=history,
        )

    assert isinstance(history, RGBTemporalPositionHistory)
    assert history.has_reset.item()
    assert history.valid_mask.sum().item() == 1
    assert history.post_reset_sample_count.item() == 1


def test_rgb_module_emits_depth_only_multiframe_position_evidence() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            position=torch.tensor([[[0.0, 0.0, 2.0]]]),
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=False,
            temporal_velocity_history_size=3,
            temporal_position_enabled=True,
            temporal_position_min_samples=3,
            temporal_position_variance_scale=2.0,
            temporal_position_variance_floor=0.01,
            temporal_position_variance_ceiling=0.1,
            temporal_position_depth_only=True,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    evidence = None
    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        measured_position = torch.tensor([[[0.3, -0.2, 2.0 + timestamp]]])
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": measured_position,
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(0.01),
                ),
                "structured_depth_valid": torch.tensor([[True]]),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )
        assert evidence is not None
        assert evidence.position_valid_mask is not None
        assert evidence.position_valid_mask.item() is (sample_index == 2)

    assert evidence is not None and evidence.position is not None
    torch.testing.assert_close(evidence.position[..., :2], belief.objects.position[..., :2])
    torch.testing.assert_close(evidence.position[..., 2], torch.tensor([[2.1]]))
    assert not evidence.valid_mask.item()
    assert measured.auxiliary["world_trajectory_position_valid_mask"].item()
    torch.testing.assert_close(
        measured.auxiliary["world_trajectory_position"][..., 2],
        torch.tensor([[2.1]]),
    )


def test_rgb_module_lateral_velocity_preserves_analytic_vertical_state() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    prior_velocity = torch.tensor([[[0.1, -2.0, 0.3]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            position=torch.zeros(1, 1, 3),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_min_samples=2,
            temporal_velocity_variance_floor=0.01,
            temporal_velocity_variance_ceiling=0.1,
            temporal_velocity_lateral_only=True,
            temporal_velocity_unobserved_variance=1.0e4,
            temporal_velocity_measurement_position_blend=1.0,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    evidence = None
    for timestamp in (0.0, 0.05):
        current = belief.replace(objects=belief.objects.replace(position=torch.zeros(1, 1, 3)))
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position", "geometry", "appearance"),
            auxiliary={
                "world_position": torch.tensor([[[timestamp, -0.5 * timestamp, 0.0]]]),
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=current,
            measured=measured,
            association=association,
            history=history,
        )

    assert evidence is not None and evidence.valid_mask.item()
    torch.testing.assert_close(evidence.velocity, torch.tensor([[[1.0, -2.0, 0.3]]]))
    assert evidence.log_variance.exp()[0, 0, 0] <= 0.1
    torch.testing.assert_close(
        evidence.log_variance.exp()[0, 0, 1:],
        torch.tensor([1.0e4, 1.0e4]),
    )
    assert evidence.axis_valid_mask is not None
    assert evidence.axis_valid_mask.tolist() == [[[True, False, False]]]

    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    corrected = updater.correct_direct_velocity(belief, evidence)
    assert corrected.objects.velocity[0, 0, 0] > belief.objects.velocity[0, 0, 0]
    assert torch.equal(
        corrected.objects.velocity[..., 1:],
        belief.objects.velocity[..., 1:],
    )
    assert torch.equal(
        corrected.objects.fast_log_variance[..., 4:6],
        belief.objects.fast_log_variance[..., 4:6],
    )


def test_rgb_module_post_event_gravity_compensation_updates_vertical_velocity() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    prior_velocity = torch.tensor([[[0.1, -2.0, 0.3]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            position=torch.zeros(1, 1, 3),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_min_samples=3,
            temporal_velocity_variance_floor=0.01,
            temporal_velocity_variance_ceiling=0.1,
            temporal_velocity_lateral_only=True,
            temporal_velocity_post_event_gravity_axis_enabled=True,
            temporal_velocity_unobserved_variance=1.0e4,
            temporal_velocity_reset_on_collision=True,
            temporal_velocity_max_age_steps=3,
            temporal_velocity_post_event_max_samples=3,
            temporal_velocity_measurement_position_blend=1.0,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    initial_velocity = torch.tensor([[[1.0, 2.0, 5.0]]])
    history = None
    evidence = None
    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        modes = belief.objects.motion_mode_logits.clone()
        modes[..., MotionMode.COLLISION] = 8.0 if sample_index == 0 else -8.0
        current = belief.replace(
            objects=belief.objects.replace(
                motion_mode_logits=modes,
            )
        )
        measured_position = (
            initial_velocity * timestamp + 0.5 * belief.gravity[:, None] * timestamp**2
        )
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": measured_position,
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=current,
            measured=measured,
            association=association,
            history=history,
        )

    assert evidence is not None and evidence.valid_mask.item()
    expected_vertical = initial_velocity[..., 1] + belief.gravity[:, None, 1] * 0.1
    torch.testing.assert_close(evidence.velocity[..., 0], torch.ones(1, 1))
    torch.testing.assert_close(evidence.velocity[..., 1], expected_vertical)
    torch.testing.assert_close(evidence.velocity[..., 2], prior_velocity[..., 2])
    assert torch.all(evidence.log_variance.exp()[..., :2] <= 0.1)
    torch.testing.assert_close(
        evidence.log_variance.exp()[..., 2],
        torch.full((1, 1), 1.0e4),
    )
    assert evidence.axis_valid_mask is not None
    assert evidence.axis_valid_mask.tolist() == [[[True, True, False]]]


@pytest.mark.parametrize("gravity_aware", (False, True))
def test_rgb_module_continuous_gravity_fit_is_opt_in(
    gravity_aware: bool,
) -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    prior_velocity = torch.tensor([[[-3.0, 4.0, 2.0]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=5,
            temporal_velocity_min_samples=3,
            temporal_velocity_variance_floor=1.0e-6,
            temporal_velocity_variance_ceiling=0.2,
            temporal_velocity_independent_raw_history_enabled=True,
            temporal_velocity_continuous_gravity_axis_enabled=gravity_aware,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    initial_velocity = torch.tensor([[[1.25, 2.0, -0.6]]])
    history = None
    evidence = None
    for timestamp in (0.0, 0.05, 0.1, 0.15, 0.2):
        measured_position = (
            initial_velocity * timestamp + 0.5 * belief.gravity[:, None] * timestamp**2
        )
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": measured_position,
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
                "world_position_independent_axis_mask": torch.ones(
                    1,
                    1,
                    3,
                    dtype=torch.bool,
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )

    assert evidence is not None and evidence.valid_mask.item()
    expected_query_velocity = initial_velocity + belief.gravity[:, None] * 0.2
    expected_midpoint_velocity = initial_velocity + belief.gravity[:, None] * 0.1
    expected = expected_query_velocity if gravity_aware else expected_midpoint_velocity
    torch.testing.assert_close(evidence.velocity, expected, atol=2.0e-5, rtol=0.0)
    assert evidence.axis_valid_mask is not None and evidence.axis_valid_mask.all()


def test_continuous_gravity_lateral_mode_uses_rotated_camera_basis() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    angle = math.pi / 4.0
    world_from_camera = torch.eye(4).unsqueeze(0)
    world_from_camera[0, 0, 0] = math.cos(angle)
    world_from_camera[0, 1, 0] = math.sin(angle)
    world_from_camera[0, 0, 1] = -math.sin(angle)
    world_from_camera[0, 1, 1] = math.cos(angle)
    prior_velocity = torch.tensor([[[-2.0, 3.0, 4.5]]])
    belief = base.replace(
        camera=base.camera.replace(world_from_camera=world_from_camera),
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        ),
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=3,
            temporal_velocity_min_samples=3,
            temporal_velocity_variance_floor=1.0e-6,
            temporal_velocity_lateral_only=True,
            temporal_velocity_independent_raw_history_enabled=True,
            temporal_velocity_continuous_gravity_axis_enabled=True,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    initial_velocity = torch.tensor([[[1.0, 2.0, -0.75]]])
    history = None
    evidence = None
    for timestamp in (0.0, 0.05, 0.1):
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": (
                    initial_velocity * timestamp + 0.5 * belief.gravity[:, None] * timestamp**2
                ),
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
                "world_position_independent_axis_mask": torch.ones(
                    1,
                    1,
                    3,
                    dtype=torch.bool,
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )

    assert evidence is not None and evidence.valid_mask.item()
    # Camera x has a world-y component, but orthogonalization keeps the
    # ordinary lateral estimate on world x while gravity supplies current y.
    expected = torch.tensor([[[1.0, 2.0 - 0.981, 4.5]]])
    torch.testing.assert_close(evidence.velocity, expected, atol=2.0e-5, rtol=0.0)
    assert evidence.axis_valid_mask is not None
    assert evidence.axis_valid_mask.tolist() == [[[True, True, False]]]


def test_old_track_can_reinitialize_lateral_velocity_after_collision() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    modes = base.objects.motion_mode_logits.clone()
    modes[..., MotionMode.COLLISION] = 8.0
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            age_steps=torch.tensor([[20]]),
            motion_mode_logits=modes,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_min_samples=2,
            temporal_velocity_lateral_only=True,
            temporal_velocity_reset_on_collision=True,
            temporal_velocity_max_age_steps=3,
            temporal_velocity_post_event_max_samples=3,
            temporal_velocity_measurement_position_blend=1.0,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    for sample_index, timestamp in enumerate((0.0, 0.05)):
        current_modes = modes.clone()
        if sample_index:
            current_modes[..., MotionMode.COLLISION] = -8.0
        current = belief.replace(objects=belief.objects.replace(motion_mode_logits=current_modes))
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": torch.tensor([[[timestamp, 0.0, 0.0]]]),
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=current,
            measured=measured,
            association=association,
            history=history,
        )

    assert history is not None
    assert history.has_reset.item()
    assert history.post_reset_sample_count.item() == 2
    assert evidence is not None and evidence.valid_mask.item()
    torch.testing.assert_close(evidence.velocity[..., 0], torch.ones(1, 1))


def test_rgb_change_point_requires_post_event_gravity_correction() -> None:
    with pytest.raises(
        ValueError,
        match="post-event gravity-axis correction",
    ):
        RGBObservationConfig(
            temporal_velocity_change_point_enabled=True,
            temporal_velocity_reset_on_collision=True,
        )


def test_rgb_change_point_reopens_gravity_velocity_at_contact_mode() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    prior_velocity = torch.tensor([[[0.1, -2.0, 0.3]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            age_steps=torch.tensor([[20]]),
            position=torch.zeros(1, 1, 3),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=5,
            temporal_velocity_min_samples=3,
            temporal_velocity_variance_floor=0.01,
            temporal_velocity_variance_ceiling=0.1,
            temporal_velocity_lateral_only=True,
            temporal_velocity_post_event_gravity_axis_enabled=True,
            temporal_velocity_unobserved_variance=1.0e4,
            temporal_velocity_reset_on_collision=True,
            temporal_velocity_max_age_steps=3,
            temporal_velocity_post_event_max_samples=3,
            temporal_velocity_change_point_enabled=True,
            temporal_velocity_change_point_minimum_speed=0.25,
            temporal_velocity_change_point_minimum_delta=0.75,
            temporal_velocity_change_point_strong_delta=2.0,
            temporal_velocity_measurement_position_blend=1.0,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    evidence = None
    detected = False
    first_post_event_valid_timestamp = None
    positions = (0.0, -0.05, 0.0, 0.0377375, 0.05095)
    for timestamp, vertical_position in zip(
        (0.0, 0.05, 0.1, 0.15, 0.2),
        positions,
        strict=True,
    ):
        modes = belief.objects.motion_mode_logits.clone()
        modes.fill_(-8.0)
        endpoint_mode = (
            MotionMode.GROUND_CONTACT if math.isclose(timestamp, 0.1) else MotionMode.FREE
        )
        modes[..., endpoint_mode] = 8.0
        current = belief.replace(objects=belief.objects.replace(motion_mode_logits=modes))
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": torch.tensor([[[0.0, vertical_position, 0.0]]]),
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=current,
            measured=measured,
            association=association,
            history=history,
        )
        detected = detected or bool(measured.auxiliary["trajectory_change_point_mask"].item())
        if (
            detected
            and evidence is not None
            and evidence.valid_mask.item()
            and first_post_event_valid_timestamp is None
        ):
            first_post_event_valid_timestamp = timestamp

    assert detected
    assert first_post_event_valid_timestamp == 0.15
    assert history is not None and history.has_reset.item()
    assert history.change_point_reset.item()
    assert history.post_reset_sample_count.item() == 3
    assert evidence is not None and evidence.valid_mask.item()
    torch.testing.assert_close(
        evidence.velocity[..., 1],
        torch.tensor([[0.019]]),
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(evidence.velocity[..., 0], prior_velocity[..., 0])
    torch.testing.assert_close(evidence.velocity[..., 2], prior_velocity[..., 2])


def test_learned_outgoing_proposal_is_consumed_on_aligned_trigger_frame() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    prior_velocity = torch.tensor([[[0.1, -2.0, 0.3]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            age_steps=torch.tensor([[20]]),
            position=torch.zeros(1, 1, 3),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=5,
            temporal_velocity_min_samples=3,
            temporal_velocity_lateral_only=True,
            temporal_velocity_post_event_gravity_axis_enabled=True,
            temporal_velocity_reset_on_collision=True,
            temporal_velocity_max_age_steps=3,
            temporal_velocity_post_event_max_samples=3,
            temporal_velocity_change_point_enabled=True,
            temporal_velocity_change_point_gate="linear",
            temporal_velocity_change_point_linear_weights=(0.0,) * 9,
            temporal_velocity_change_point_linear_bias=10.0,
            temporal_velocity_change_point_minimum_interval_samples=3,
            temporal_velocity_change_point_require_contact_mode=False,
            temporal_velocity_outgoing_proposal_enabled=True,
            temporal_velocity_outgoing_proposal_hidden_weights=(0.0,) * 11,
            temporal_velocity_outgoing_proposal_hidden_bias=(0.0,),
            temporal_velocity_outgoing_proposal_output_weights=(0.0,),
            temporal_velocity_outgoing_proposal_output_bias=1.25,
            temporal_velocity_outgoing_proposal_variance=0.25,
            temporal_velocity_measurement_position_blend=1.0,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    evidence = None
    measured = None
    trigger_evidence = None
    trigger_measurement = None
    for timestamp, vertical_position in zip(
        (0.0, 0.05, 0.1, 0.15, 0.2, 0.25),
        (0.0, -0.05, 0.0, 0.04, 0.06, 0.07),
        strict=True,
    ):
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": torch.tensor([[[0.0, vertical_position, 0.0]]]),
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )
        if measured.auxiliary["trajectory_change_point_mask"].item():
            trigger_measurement = measured
            trigger_evidence = evidence

    assert trigger_measurement is not None
    torch.testing.assert_close(
        trigger_measurement.auxiliary["trajectory_outgoing_velocity_delta"],
        torch.tensor([[1.25]]),
    )
    assert trigger_evidence is not None and trigger_evidence.valid_mask.item()
    torch.testing.assert_close(trigger_evidence.velocity[..., 0], prior_velocity[..., 0])
    torch.testing.assert_close(
        trigger_evidence.velocity[..., 1],
        torch.tensor([[-3.25]]),
    )
    torch.testing.assert_close(trigger_evidence.velocity[..., 2], prior_velocity[..., 2])


def test_lateral_intervention_emits_soft_gated_pre_filter_measurement() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    prior_velocity = torch.tensor([[[0.1, -2.0, 0.3]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            age_steps=torch.tensor([[20]]),
            position=torch.zeros(1, 1, 3),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=5,
            temporal_velocity_min_samples=3,
            temporal_velocity_lateral_only=True,
            temporal_velocity_max_age_steps=3,
            temporal_velocity_measurement_position_blend=1.0,
            temporal_velocity_lateral_intervention_enabled=True,
            temporal_velocity_lateral_intervention_hidden_weights=(0.0,) * 19,
            temporal_velocity_lateral_intervention_hidden_bias=(0.0,),
            temporal_velocity_lateral_intervention_output_weights=(0.0,) * 2,
            temporal_velocity_lateral_intervention_output_bias=(1.25, 10.0),
            temporal_velocity_lateral_intervention_variance_floor=0.04,
            temporal_velocity_lateral_intervention_variance_ceiling=25.0,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    evidence = None
    measured = None
    for timestamp in (0.0, 0.05, 0.1):
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": torch.tensor([[[timestamp, 0.0, 0.0]]]),
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )

    assert measured is not None
    assert evidence is not None and evidence.valid_mask.item()
    torch.testing.assert_close(
        measured.auxiliary["trajectory_direct_prior_velocity"],
        prior_velocity,
    )
    assert measured.auxiliary["trajectory_lateral_intervention_gain"].item() > 0.999
    torch.testing.assert_close(evidence.velocity[..., 0], torch.tensor([[1.35]]))
    torch.testing.assert_close(evidence.velocity[..., 1], prior_velocity[..., 1])
    torch.testing.assert_close(evidence.velocity[..., 2], prior_velocity[..., 2])
    assert evidence.log_variance.exp()[0, 0, 0] < 0.041
    assert torch.all(evidence.log_variance.exp()[0, 0, 1:] >= 1.0e4)


def test_gravity_intervention_changes_only_gravity_velocity_component() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    prior_velocity = torch.tensor([[[0.1, -2.0, 0.3]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[12]]),
            age_steps=torch.tensor([[20]]),
            position=torch.zeros(1, 1, 3),
            velocity=prior_velocity,
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=4,
            roi_size=8,
            roi_hidden_dim=16,
            temporal_velocity_enabled=True,
            temporal_velocity_history_size=5,
            temporal_velocity_min_samples=3,
            temporal_velocity_lateral_only=True,
            temporal_velocity_max_age_steps=3,
            temporal_velocity_measurement_position_blend=1.0,
            temporal_velocity_gravity_intervention_enabled=True,
            temporal_velocity_gravity_intervention_hidden_weights=(0.0,) * 21,
            temporal_velocity_gravity_intervention_hidden_bias=(0.0,),
            temporal_velocity_gravity_intervention_output_weights=(0.0,) * 2,
            temporal_velocity_gravity_intervention_output_bias=(1.25, 10.0),
            temporal_velocity_gravity_intervention_variance_floor=0.04,
            temporal_velocity_gravity_intervention_variance_ceiling=25.0,
        )
    )
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[False]]),
    )
    history = None
    evidence = None
    measured = None
    for timestamp in (0.0, 0.05, 0.1):
        measured = MeasurementSet(
            modality="rgb",
            sensor_id="camera",
            timestamp=torch.tensor([timestamp]),
            values=torch.zeros(1, 1, 7),
            log_variance=torch.zeros(1, 1, 7),
            existence_logits=torch.tensor([[8.0]]),
            measurement_mask=torch.tensor([[True]]),
            appearance=None,
            class_logits=None,
            frame_id="camera:camera",
            supported_state_fields=("position",),
            auxiliary={
                "world_position": torch.tensor([[[0.0, -timestamp, 0.0]]]),
                "world_position_log_variance": torch.full(
                    (1, 1, 3),
                    math.log(1.0e-4),
                ),
            },
        )
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=history,
        )

    assert measured is not None
    assert evidence is not None and evidence.valid_mask.item()
    assert measured.auxiliary["trajectory_gravity_intervention_gain"].item() > 0.999
    torch.testing.assert_close(evidence.velocity[..., 0], prior_velocity[..., 0])
    torch.testing.assert_close(evidence.velocity[..., 1], torch.tensor([[-3.25]]))
    torch.testing.assert_close(evidence.velocity[..., 2], prior_velocity[..., 2])
    assert evidence.log_variance.exp()[0, 0, 1] < 0.041
    assert evidence.log_variance.exp()[0, 0, 0] >= 1.0e4
    assert evidence.log_variance.exp()[0, 0, 2] >= 1.0e4
