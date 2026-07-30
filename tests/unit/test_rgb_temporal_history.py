from __future__ import annotations

import math

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode
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
    assert not detached.post_reset_sample_count.requires_grad
    assert not detached.has_reset.requires_grad
    assert detached.timestamps.grad_fn is None
    assert detached.positions.grad_fn is None
    assert detached.position_log_variance.grad_fn is None


def test_rgb_module_emits_post_correction_evidence_and_measurement_annotations() -> None:
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
    history = None
    for sample_index, timestamp in enumerate((0.0, 0.05, 0.1)):
        current = belief.replace(
            objects=belief.objects.replace(position=expected_velocity * timestamp)
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
            supported_state_fields=("position", "geometry", "appearance"),
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
    torch.testing.assert_close(evidence.velocity, expected_velocity)
    assert torch.all(evidence.log_variance.exp() <= 2.0)
    torch.testing.assert_close(
        measured.auxiliary["world_velocity"],
        expected_velocity,
    )


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
