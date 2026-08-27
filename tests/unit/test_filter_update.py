from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.belief import NUM_MOTION_MODES, BeliefFactory, ObjectLifecycle
from world_model.filtering import (
    BeliefUpdater,
    BeliefUpdaterConfig,
    FilterUncertainty,
    FilterUncertaintyConfig,
    diagonal_kalman_update,
)
from world_model.fusion import (
    AssociationResult,
    Associator,
    SurpriseAssessment,
    build_innovation,
)
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


def test_filter_is_single_authority_for_missed_track_uncertainty() -> None:
    factory = BeliefFactory(max_objects=2)
    belief = factory.create()
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True, False]]),
            object_id=torch.tensor([[4, -1]]),
        )
    )
    increment = 0.125
    uncertainty = FilterUncertainty(
        FilterUncertaintyConfig(missed_fast_variance_increment=increment)
    )

    posterior = uncertainty.missed(belief, torch.tensor([[True, False]]))

    prior_variance = belief.objects.fast_log_variance.exp()
    posterior_variance = posterior.objects.fast_log_variance.exp()
    torch.testing.assert_close(
        posterior_variance[0, 0],
        prior_variance[0, 0] + increment,
    )
    torch.testing.assert_close(
        posterior_variance[0, 1],
        prior_variance[0, 1],
    )

    lifecycle_posterior = ObjectLifecycle().update_visibility(
        posterior,
        observed_mask=torch.tensor([[False, False]]),
    )
    torch.testing.assert_close(
        lifecycle_posterior.objects.fast_log_variance,
        posterior.objects.fast_log_variance,
    )


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


def test_direct_metric_position_update_has_one_differentiable_axis_owner() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    world_position = torch.tensor(
        [[[1.0, 1_000.0, -0.5]]],
        requires_grad=True,
    )
    axis_support = torch.tensor([[[True, False, True]]])
    measurement_log_variance = torch.full_like(world_position, -8.0)
    measured = replace(
        measured,
        modality="rgbd",
        values=world_position,
        log_variance=measurement_log_variance,
        supported_state_fields=("position",),
        auxiliary={
            "world_position": world_position,
            "world_position_log_variance": measurement_log_variance,
            "world_position_independent_axis_mask": axis_support,
        },
    )
    predicted = replace(predicted, modality="rgbd")
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=2,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(
            enable_learned_corrector=False,
            direct_metric_position_update=True,
        ),
    )

    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )

    expected_position = torch.tensor([[[1.0, 0.0, -0.5]]])
    torch.testing.assert_close(posterior.objects.position, expected_position)
    torch.testing.assert_close(
        posterior.objects.fast_log_variance[..., :3],
        torch.where(
            axis_support,
            measurement_log_variance,
            belief.objects.fast_log_variance[..., :3],
        ),
    )
    assert updater.last_diagnostics is not None
    torch.testing.assert_close(
        updater.last_diagnostics.analytic_gain,
        axis_support[0].to(torch.float32),
    )
    posterior.objects.position.sum().backward()
    assert world_position.grad is not None
    torch.testing.assert_close(
        world_position.grad,
        axis_support.to(torch.float32),
    )


def test_ambiguous_direct_metric_position_update_fails_closed() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    world_position = torch.tensor(
        [[[0.1, -0.2, 0.3]]],
        requires_grad=True,
    )
    measurement_log_variance = torch.full_like(world_position, -8.0)
    measured = replace(
        measured,
        modality="rgbd",
        values=world_position,
        log_variance=measurement_log_variance,
        supported_state_fields=("position",),
        auxiliary={
            "world_position": world_position,
            "world_position_log_variance": measurement_log_variance,
            "world_position_independent_axis_mask": torch.ones_like(
                world_position,
                dtype=torch.bool,
            ),
        },
    )
    predicted = replace(predicted, modality="rgbd")
    association = replace(association, ambiguous=torch.tensor([[True]]))
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=2,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(
            enable_learned_corrector=False,
            direct_metric_position_update=True,
        ),
    )

    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )

    torch.testing.assert_close(posterior.objects.position, belief.objects.position)
    missed = updater.uncertainty.missed(belief, belief.objects.active)
    torch.testing.assert_close(
        posterior.objects.fast_log_variance[..., :3],
        missed.objects.fast_log_variance[..., :3],
    )
    assert updater.last_diagnostics is not None
    assert not updater.last_diagnostics.observed_mask.any()
    torch.testing.assert_close(
        updater.last_diagnostics.analytic_gain,
        torch.zeros_like(updater.last_diagnostics.analytic_gain),
    )
    torch.testing.assert_close(
        updater.last_diagnostics.correction_norm,
        torch.zeros_like(updater.last_diagnostics.correction_norm),
    )
    posterior.objects.position.sum().backward()
    assert world_position.grad is not None
    torch.testing.assert_close(world_position.grad, torch.zeros_like(world_position.grad))


def test_source_bound_copied_position_axes_never_fuse_as_new_evidence() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )

    unbound_innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=0,
    )
    unbound = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=unbound_innovation,
        dt=1.0,
    )
    assert not torch.equal(unbound.objects.position, belief.objects.position)
    assert not torch.equal(unbound.objects.velocity, belief.objects.velocity)
    assert not torch.equal(
        unbound.objects.fast_log_variance[..., :6],
        belief.objects.fast_log_variance[..., :6],
    )

    for independent_axis_mask in (
        torch.zeros((1, 1, 3), dtype=torch.bool),
        None,
    ):
        auxiliary = dict(measured.auxiliary)
        if independent_axis_mask is not None:
            auxiliary["world_position_independent_axis_mask"] = independent_axis_mask
        source_bound = replace(
            measured,
            auxiliary=auxiliary,
            source_belief_indices=torch.tensor([[0]]),
            source_object_ids=torch.tensor([[3]]),
        )
        source_innovation = build_innovation(
            measured=source_bound,
            predicted=predicted,
            association=association,
            modality_index=0,
        )
        posterior = updater.correct(
            prior=belief,
            measured=source_bound,
            predicted=predicted,
            association=association,
            innovation=source_innovation,
            dt=1.0,
        )

        assert torch.equal(posterior.objects.position, belief.objects.position)
        assert torch.equal(posterior.objects.velocity, belief.objects.velocity)
        assert torch.equal(
            posterior.objects.fast_log_variance,
            belief.objects.fast_log_variance,
        )


def test_weak_association_does_not_confirm_track_lifecycle() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    measured = replace(
        measured,
        existence_logits=torch.full_like(measured.existence_logits, -20.0),
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
        dt=1.0,
    )
    assert updater.last_diagnostics is not None
    assert not updater.last_diagnostics.observed_mask.any()

    accounted = ObjectLifecycle().update_visibility(
        posterior,
        updater.last_diagnostics.observed_mask,
    )
    assert accounted.objects.missed_steps[0, 0] == 1
    assert accounted.objects.existence_logit[0, 0] < belief.objects.existence_logit[0, 0]


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


def _force_nonzero_learned_fast_outputs(updater: BeliefUpdater) -> None:
    corrector = updater.learned_corrector
    assert corrector is not None
    with torch.no_grad():
        corrector.mean_head.weight.zero_()
        corrector.mean_head.bias.fill_(1.0)
        corrector.variance_head.weight.zero_()
        corrector.variance_head.bias.fill_(1.0)
        corrector.gate_head.weight.zero_()
        corrector.gate_head.bias.fill_(8.0)


@pytest.mark.parametrize("include_axis_mask", [True, False])
def test_source_bound_unsupported_axes_block_learned_mean_and_variance(
    include_axis_mask: bool,
) -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    auxiliary = dict(measured.auxiliary)
    if include_axis_mask:
        auxiliary["world_position_independent_axis_mask"] = torch.zeros(
            (1, 1, 3),
            dtype=torch.bool,
        )
    source_bound = replace(
        measured,
        auxiliary=auxiliary,
        source_belief_indices=torch.tensor([[0]]),
        source_object_ids=torch.tensor([[3]]),
    )
    innovation = build_innovation(
        measured=source_bound,
        predicted=predicted,
        association=association,
        modality_index=0,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(
            innovation_anchored_correction=True,
            learned_residual_scale=0.5,
        ),
    )
    _force_nonzero_learned_fast_outputs(updater)
    corrector = updater.learned_corrector
    assert corrector is not None
    with torch.no_grad():
        corrector.variance_head.bias[:6].fill_(-1.0)

    posterior = updater.correct(
        prior=belief,
        measured=source_bound,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )

    assert torch.equal(posterior.objects.position, belief.objects.position)
    assert torch.equal(posterior.objects.velocity, belief.objects.velocity)
    assert torch.equal(
        posterior.objects.fast_log_variance[..., :6],
        belief.objects.fast_log_variance[..., :6],
    )


def test_unsupported_position_values_cannot_suppress_a_supported_axis() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(
            innovation_anchored_correction=True,
            learned_residual_scale=0.5,
        ),
    )
    _force_nonzero_learned_fast_outputs(updater)

    def corrected(unsupported_y: float, unsupported_z: float):
        world_position = measured.auxiliary["world_position"].clone()
        world_position[..., 1] = unsupported_y
        world_position[..., 2] = unsupported_z
        partial = replace(
            measured,
            auxiliary={
                **measured.auxiliary,
                "world_position": world_position,
                "world_position_independent_axis_mask": torch.tensor([[[True, False, False]]]),
            },
            source_belief_indices=torch.tensor([[0]]),
            source_object_ids=torch.tensor([[3]]),
        )
        innovation = build_innovation(
            measured=partial,
            predicted=predicted,
            association=association,
            modality_index=0,
        )
        return updater.correct(
            prior=belief,
            measured=partial,
            predicted=predicted,
            association=association,
            innovation=innovation,
            dt=1.0,
        )

    ordinary = corrected(0.0, 0.0)
    adversarial = corrected(1000.0, -1000.0)

    assert ordinary.objects.position[0, 0, 0] > 0.5
    assert torch.equal(adversarial.objects.position, ordinary.objects.position)
    assert torch.equal(adversarial.objects.velocity, ordinary.objects.velocity)
    assert torch.equal(
        adversarial.objects.fast_log_variance[..., :6],
        ordinary.objects.fast_log_variance[..., :6],
    )


def test_innovation_anchored_corrector_is_axis_local_and_support_masked() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=association,
        modality_index=0,
    )
    analytic = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    ).correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )
    updater = BeliefUpdater(
        fast_state_dim=belief.objects.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(
            innovation_anchored_correction=True,
            learned_residual_scale=0.5,
        ),
    )
    _force_nonzero_learned_fast_outputs(updater)

    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )

    assert posterior.objects.position[0, 0, 0] > analytic.objects.position[0, 0, 0]
    assert posterior.objects.velocity[0, 0, 0] > analytic.objects.velocity[0, 0, 0]
    torch.testing.assert_close(
        posterior.objects.position[..., 1:],
        analytic.objects.position[..., 1:],
    )
    torch.testing.assert_close(
        posterior.objects.velocity[..., 1:],
        analytic.objects.velocity[..., 1:],
    )
    torch.testing.assert_close(posterior.objects.orientation, analytic.objects.orientation)
    torch.testing.assert_close(
        posterior.objects.angular_velocity,
        analytic.objects.angular_velocity,
    )
    torch.testing.assert_close(posterior.objects.modal_state, analytic.objects.modal_state)
    torch.testing.assert_close(
        posterior.objects.fast_log_variance[..., 6:],
        analytic.objects.fast_log_variance[..., 6:],
    )


def test_innovation_anchored_corrector_cannot_move_mean_without_innovation() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    measured = replace(
        measured,
        values=torch.zeros_like(measured.values),
        auxiliary={
            **measured.auxiliary,
            "world_position": torch.zeros_like(measured.auxiliary["world_position"]),
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
        config=BeliefUpdaterConfig(
            innovation_anchored_correction=True,
            learned_residual_scale=0.5,
        ),
    )
    _force_nonzero_learned_fast_outputs(updater)

    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )

    torch.testing.assert_close(posterior.objects.position, belief.objects.position)
    torch.testing.assert_close(posterior.objects.velocity, belief.objects.velocity)
    torch.testing.assert_close(posterior.objects.orientation, belief.objects.orientation)


def test_innovation_anchored_corrector_keeps_a_finite_trainable_path() -> None:
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
        config=BeliefUpdaterConfig(innovation_anchored_correction=True),
    )

    posterior = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )
    posterior.objects.position[..., 0].sum().backward()

    corrector = updater.learned_corrector
    assert corrector is not None
    gradient = corrector.mean_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_position_quality_conservatively_caps_each_axis() -> None:
    belief, measured, predicted, association = _rgb_position_update_case()
    measured_values = torch.ones_like(measured.values)
    measured = replace(
        measured,
        values=measured_values,
        existence_logits=torch.zeros_like(measured.existence_logits),
        auxiliary={
            **measured.auxiliary,
            "world_position": measured_values,
            "position_confidence": torch.tensor([[[1.0, 0.01, 0.0]]]),
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
        dt=1.0,
    )

    # High quality cannot promote a 0.5 existence confidence.
    assert 0.45 < posterior.objects.position[0, 0, 0] < 0.55
    assert 0.0 < posterior.objects.position[0, 0, 1] < 0.02
    torch.testing.assert_close(
        posterior.objects.position[0, 0, 2],
        torch.tensor(0.0),
    )
    # Indirect position-to-velocity evidence obeys the same position quality.
    assert posterior.objects.velocity[0, 0, 0] > 0.2
    assert posterior.objects.velocity[0, 0, 1].abs() < 0.01
    torch.testing.assert_close(
        posterior.objects.velocity[0, 0, 2],
        torch.tensor(0.0),
    )


def test_surprise_weight_does_not_double_clip_analytic_position_update() -> None:
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
    baseline = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )
    cause = SurpriseAssessment(
        cause_probabilities=torch.zeros(1, 1, 7),
        robust_weight=torch.tensor([[0.01]]),
        trigger_global=torch.tensor([True]),
        aggregate_surprise=torch.tensor([100.0]),
    )
    surprised = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
        cause=cause,
    )

    torch.testing.assert_close(
        surprised.objects.position,
        baseline.objects.position,
    )


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


def test_invalid_temporal_velocity_preserves_position_innovation_velocity_update() -> None:
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
    ordinary = updater.correct(
        prior=belief,
        measured=measured,
        predicted=predicted,
        association=association,
        innovation=innovation,
        dt=1.0,
    )
    assert ordinary.objects.velocity[0, 0, 0] > 0.0

    posterior = updater.correct_direct_velocity(
        ordinary,
        DirectVelocityEvidence(
            velocity=torch.full((1, 1, 3), 99.0),
            log_variance=torch.full((1, 1, 3), -8.0),
            valid_mask=torch.tensor([[False]]),
            confidence=torch.ones(1, 1),
            axis_valid_mask=torch.ones(1, 1, 3, dtype=torch.bool),
        ),
    )

    assert torch.equal(posterior.objects.velocity, ordinary.objects.velocity)
    assert torch.equal(
        posterior.objects.fast_log_variance[..., 3:6],
        ordinary.objects.fast_log_variance[..., 3:6],
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
    assert posterior.objects.velocity[0, 0, 1] < -0.75
    assert posterior.objects.velocity[0, 0, 2] > 0.375
    torch.testing.assert_close(posterior.objects.velocity[0, 1], torch.zeros(3))
    assert updater.last_diagnostics is sentinel


def test_direct_velocity_fifth_positional_argument_remains_position() -> None:
    velocity = torch.zeros(1, 1, 3)
    log_variance = torch.zeros_like(velocity)
    valid_mask = torch.tensor([[False]])
    confidence = torch.ones(1, 1)
    position = torch.tensor([[[0.25, -0.5, 1.0]]])
    position_log_variance = torch.zeros_like(position)
    position_valid_mask = torch.tensor([[True]])
    axis_valid_mask = torch.tensor([[[True, False, True]]])

    evidence = DirectVelocityEvidence(
        velocity,
        log_variance,
        valid_mask,
        confidence,
        position,
        position_log_variance,
        position_valid_mask,
        axis_valid_mask,
    )

    assert evidence.position is position
    assert evidence.position_log_variance is position_log_variance
    assert evidence.position_valid_mask is position_valid_mask
    assert evidence.axis_valid_mask is axis_valid_mask
    evidence.validate()


def test_direct_velocity_axis_mask_leaves_unsupported_components_bitwise_unchanged() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    initial_velocity = torch.tensor([[[0.25, -1.5, 0.75]]])
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[5]]),
            velocity=initial_velocity,
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )

    posterior = updater.correct_direct_velocity(
        belief,
        DirectVelocityEvidence(
            velocity=torch.tensor([[[2.0, 9.0, -4.0]]]),
            log_variance=torch.full((1, 1, 3), -8.0),
            valid_mask=torch.tensor([[True]]),
            confidence=torch.ones(1, 1),
            axis_valid_mask=torch.tensor([[[True, False, False]]]),
        ),
    )

    assert posterior.objects.velocity[0, 0, 0] > 1.0
    assert torch.equal(
        posterior.objects.velocity[..., 1:],
        belief.objects.velocity[..., 1:],
    )
    assert torch.equal(
        posterior.objects.fast_log_variance[..., 4:6],
        belief.objects.fast_log_variance[..., 4:6],
    )
    assert posterior.objects.fast_log_variance[0, 0, 3] < belief.objects.fast_log_variance[0, 0, 3]


def test_unsupported_velocity_components_do_not_reduce_valid_axis_influence() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[5]]),
            velocity=torch.tensor([[[0.25, -1.5, 0.75]]]),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    common = {
        "log_variance": torch.full((1, 1, 3), -8.0),
        "valid_mask": torch.tensor([[True]]),
        "confidence": torch.ones(1, 1),
        "axis_valid_mask": torch.tensor([[[True, False, False]]]),
    }
    neutral = updater.correct_direct_velocity(
        belief,
        DirectVelocityEvidence(
            velocity=torch.tensor([[[2.0, -1.5, 0.75]]]),
            **common,
        ),
    )
    extreme_unsupported = updater.correct_direct_velocity(
        belief,
        DirectVelocityEvidence(
            velocity=torch.tensor([[[2.0, 1.0e6, -1.0e6]]]),
            **common,
        ),
    )

    torch.testing.assert_close(
        extreme_unsupported.objects.velocity,
        neutral.objects.velocity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        extreme_unsupported.objects.fast_log_variance,
        neutral.objects.fast_log_variance,
        rtol=0.0,
        atol=0.0,
    )


def test_post_association_direct_position_can_update_without_velocity() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    belief = factory.create().replace(
        objects=factory.create().objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[5]]),
            position=torch.zeros(1, 1, 3),
            velocity=torch.tensor([[[0.2, -0.1, 0.3]]]),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    posterior = updater.correct_direct_velocity(
        belief,
        DirectVelocityEvidence(
            velocity=torch.zeros(1, 1, 3),
            log_variance=torch.zeros(1, 1, 3),
            valid_mask=torch.tensor([[False]]),
            confidence=torch.ones(1, 1),
            position=torch.tensor([[[0.0, 0.0, 1.0]]]),
            position_log_variance=torch.full((1, 1, 3), -8.0),
            position_valid_mask=torch.tensor([[True]]),
        ),
    )

    assert posterior.objects.position[0, 0, 2] > 0.75
    torch.testing.assert_close(posterior.objects.velocity, belief.objects.velocity)


def test_direct_position_contracts_variance_even_when_mean_is_unchanged() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=4)
    base = factory.create()
    belief = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[5]]),
            position=torch.tensor([[[0.2, -0.1, 1.0]]]),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    posterior = updater.correct_direct_velocity(
        belief,
        DirectVelocityEvidence(
            velocity=torch.zeros(1, 1, 3),
            log_variance=torch.zeros(1, 1, 3),
            valid_mask=torch.tensor([[False]]),
            confidence=torch.ones(1, 1),
            position=belief.objects.position.clone(),
            position_log_variance=torch.full((1, 1, 3), -8.0),
            position_valid_mask=torch.tensor([[True]]),
        ),
    )

    torch.testing.assert_close(posterior.objects.position, belief.objects.position)
    assert torch.all(
        posterior.objects.fast_log_variance[..., :3] < belief.objects.fast_log_variance[..., :3]
    )


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
