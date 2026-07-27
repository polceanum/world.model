from __future__ import annotations

import math

import torch

from world_model.belief import BeliefFactory
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
        timestamp=torch.tensor([timestamp], dtype=positions.dtype),
        positions=positions,
        position_log_variance=torch.full_like(
            positions,
            math.log(position_variance),
        ),
        minimum_dt=minimum_dt,
    )


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
