from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.belief import NUM_MOTION_MODES, BeliefFactory
from world_model.filtering import (
    BeliefUpdater,
    BeliefUpdaterConfig,
    diagonal_kalman_update,
)
from world_model.fusion import AssociationResult, Associator, build_innovation
from world_model.observations import (
    DirectVelocityEvidence,
    MeasurementSet,
    ObservationPacket,
    PredictedMeasurements,
    SensorContext,
)
from world_model.observations.state import StateObservationModule


def test_diagonal_update_respects_measurement_noise_and_contracts_variance() -> None:
    prior = torch.tensor([[0.0]])
    prior_lv = torch.tensor([[0.0]])
    measurement = torch.tensor([[1.0]])
    low_noise = diagonal_kalman_update(prior, prior_lv, measurement, torch.tensor([[-8.0]]))
    high_noise = diagonal_kalman_update(prior, prior_lv, measurement, torch.tensor([[4.0]]))
    assert low_noise.mean.item() > high_noise.mean.item()
    assert low_noise.log_variance.item() < prior_lv.item()
    assert high_noise.log_variance.item() < prior_lv.item()


def test_zero_innovation_leaves_mean_and_outlier_is_robustly_clipped() -> None:
    prior = torch.zeros(1, 2)
    prior_lv = torch.zeros_like(prior)
    zero = diagonal_kalman_update(prior, prior_lv, prior, prior_lv)
    outlier = diagonal_kalman_update(
        prior,
        prior_lv,
        torch.full_like(prior, 1000.0),
        torch.full_like(prior, -6.0),
        robust_clip_norm=2.0,
    )
    assert torch.equal(zero.mean, prior)
    assert torch.linalg.vector_norm(outlier.mean) < 3.0


def test_oracle_position_update_reduces_error_without_resetting_identity() -> None:
    factory = BeliefFactory(max_objects=2, appearance_dim=4)
    belief = factory.create()
    objects = belief.objects.replace(
        active=torch.tensor([[True, False]]),
        object_id=torch.tensor([[7, -1]]),
        position=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        visibility_logit=torch.tensor([[5.0, 0.0]]),
    )
    belief = belief.replace(objects=objects)
    module = StateObservationModule()
    packet = ObservationPacket(
        modality="debug_oracle",
        sensor_id="state",
        timestamp=0.0,
        payload={
            "position": torch.tensor([[0.0, 0.0, 0.0]]),
            "velocity": torch.tensor([[0.0, 0.0, 0.0]]),
            "active": torch.tensor([True]),
        },
        calibration={},
        frame_id="world",
    )
    measured = module.initialise_measurements(
        [packet],
        context=object(),  # the debug module does not use initialization context
    )
    predicted = module.project(
        belief,
        SensorContext(
            sensor_id="state",
            timestamp=0.0,
            calibration={},
            frame_id="world",
        ),
    )
    association = Associator(
        geometry_dimensions=3,
        mahalanobis_gate=100.0,
    ).match(belief, measured, predicted)
    innovation = module.innovation(measured, predicted, association)
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=0.0,
    )
    assert posterior.objects.object_id[0, 0].item() == 7
    assert posterior.objects.position[0, 0, 0].abs() < 0.01
    assert posterior.objects.fast_log_variance[0, 0, 0] < belief.objects.fast_log_variance[0, 0, 0]


def _rgb_position_update_case() -> tuple[
    object,
    MeasurementSet,
    PredictedMeasurements,
    AssociationResult,
]:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    belief = factory.create()
    objects = belief.objects.replace(
        active=torch.tensor([[True]]),
        object_id=torch.tensor([[3]]),
        position=torch.zeros(1, 1, 3),
        velocity=torch.zeros(1, 1, 3),
        existence_logit=torch.tensor([[8.0]]),
        visibility_logit=torch.tensor([[8.0]]),
    )
    belief = belief.replace(objects=objects)
    measured_values = torch.tensor([[[1.0, 0.0, 0.0]]])
    measurement = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([1.0]),
        values=measured_values,
        log_variance=torch.full_like(measured_values, -8.0),
        existence_logits=torch.tensor([[8.0]]),
        measurement_mask=torch.tensor([[True]]),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera",
        supported_state_fields=("position", "velocity_from_position"),
        auxiliary={
            "world_position": measured_values,
            "world_position_log_variance": torch.full_like(measured_values, -8.0),
        },
    )
    predicted = PredictedMeasurements(
        modality="rgb",
        sensor_id="camera",
        timestamp=belief.timestamp,
        values=belief.objects.position,
        log_variance=belief.objects.fast_log_variance[..., :3],
        object_ids=belief.objects.object_id,
        belief_indices=torch.tensor([[0]]),
        valid_mask=torch.tensor([[True]]),
        visibility=torch.tensor([[1.0]]),
        appearance=None,
        auxiliary={"world_position": belief.objects.position},
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
    return belief, measurement, predicted, association


def test_rgb_temporal_position_coupling_estimates_directional_velocity() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=0,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )
    assert posterior.objects.velocity[0, 0, 0] > 0.25
    assert posterior.objects.velocity[0, 0, 1:].abs().max() < 1.0e-6
    target_future_x = torch.tensor(2.0)
    prior_future_x = belief.objects.position[0, 0, 0] + belief.objects.velocity[0, 0, 0]
    posterior_future_x = posterior.objects.position[0, 0, 0] + posterior.objects.velocity[0, 0, 0]
    assert (posterior_future_x - target_future_x).abs() < (prior_future_x - target_future_x).abs()


def test_same_timestamp_rgb_position_does_not_update_velocity() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=0,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=0.0,
    )
    assert torch.equal(posterior.objects.velocity, belief.objects.velocity)


def _direct_velocity_posterior(
    *,
    velocity_log_variance: float,
    valid: bool = True,
) -> tuple[object, object]:
    belief, measured, predicted, association = _rgb_position_update_case()
    measured = replace(
        measured,
        supported_state_fields=("position", "velocity"),
        auxiliary={
            **measured.auxiliary,
            "world_velocity": torch.tensor([[[2.0, 0.0, 0.0]]]),
            "world_velocity_log_variance": torch.full(
                (1, 1, 3),
                velocity_log_variance,
            ),
            "world_velocity_valid_mask": torch.tensor([[valid]]),
        },
    )
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=0,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=0.0,
    )
    return belief, posterior


def test_direct_velocity_uses_auxiliary_uncertainty_not_rgb_value_dimensions() -> None:
    _, precise = _direct_velocity_posterior(velocity_log_variance=-8.0)
    _, uncertain = _direct_velocity_posterior(velocity_log_variance=8.0)

    # This RGB fixture has only three sensor-value dimensions. Direct world
    # velocity and its uncertainty therefore cannot come from values[...,3:6].
    assert precise.objects.velocity[0, 0, 0] > 1.5
    assert uncertain.objects.velocity[0, 0, 0] < 0.01
    assert (
        precise.objects.fast_log_variance[0, 0, 3] < (uncertain.objects.fast_log_variance[0, 0, 3])
    )


def test_invalid_direct_velocity_evidence_leaves_velocity_unchanged() -> None:
    belief, posterior = _direct_velocity_posterior(
        velocity_log_variance=-8.0,
        valid=False,
    )
    torch.testing.assert_close(posterior.objects.velocity, belief.objects.velocity)
    torch.testing.assert_close(
        posterior.objects.fast_log_variance[..., 3:6],
        belief.objects.fast_log_variance[..., 3:6],
    )


def test_post_association_direct_velocity_updates_only_valid_active_slots() -> None:
    factory = BeliefFactory(max_objects=2, appearance_dim=4)
    belief = factory.create().replace(
        objects=factory.create().objects.replace(
            active=torch.tensor([[True, False]]),
            object_id=torch.tensor([[5, -1]]),
            velocity=torch.zeros(1, 2, 3),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    sentinel = object()
    updater.last_diagnostics = sentinel  # type: ignore[assignment]
    posterior = updater.correct_direct_velocity(
        belief,
        DirectVelocityEvidence(
            velocity=torch.tensor([[[2.0, -1.0, 0.5], [9.0, 9.0, 9.0]]]),
            log_variance=torch.full((1, 2, 3), -8.0),
            valid_mask=torch.tensor([[True, True]]),
            confidence=torch.ones(1, 2),
        ),
    )

    assert posterior.objects.velocity[0, 0, 0] > 1.5
    torch.testing.assert_close(posterior.objects.velocity[0, 1], torch.zeros(3))
    assert updater.last_diagnostics is sentinel


def test_direct_velocity_requires_explicit_auxiliary_log_variance() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    measured = replace(
        measured,
        supported_state_fields=("position", "velocity"),
        auxiliary={
            **measured.auxiliary,
            "world_velocity": torch.zeros(1, 1, 3),
            "world_velocity_valid_mask": torch.tensor([[True]]),
        },
    )
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=0,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    with pytest.raises(ValueError, match="world_velocity_log_variance"):
        updater.correct(
            prior=belief,
            measured=measured,
            predicted=predicted,
            association=association,
            innovation=innovation,
            dt=0.0,
        )
