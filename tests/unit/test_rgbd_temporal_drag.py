"""Seed-free unit tests for opt-in RGB-D per-object drag evidence."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from world_model.belief import NUM_MOTION_MODES, BeliefFactory, slow_packing_map
from world_model.dynamics import free_motion_position_velocity
from world_model.filtering import BeliefUpdater, BeliefUpdaterConfig
from world_model.fusion import AssociationResult
from world_model.observations import DirectVelocityEvidence, MeasurementSet
from world_model.observations.rgbd import RGBDObservationConfig, RGBDObservationModule


def _drag_module() -> RGBDObservationModule:
    return RGBDObservationModule(RGBDObservationConfig(temporal_drag_estimation_enabled=True))


def _active_two_object_belief():
    factory = BeliefFactory(max_objects=2, appearance_dim=3)
    base = factory.create(gravity=(0.0, 0.0, 0.0))
    return base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True, True]]),
            object_id=torch.tensor([[7, 11]], dtype=torch.int64),
        )
    )


def _measurement(
    positions: torch.Tensor,
    timestamp: float,
    *,
    valid: torch.Tensor | None = None,
) -> MeasurementSet:
    if positions.shape != (1, 2, 3):
        raise ValueError("test positions must be [1,2,3]")
    if valid is None:
        valid = torch.ones((1, 2), dtype=torch.bool, device=positions.device)
    log_variance = positions.new_full(positions.shape, -9.0)
    visible_positions = torch.where(
        valid.unsqueeze(-1),
        positions,
        torch.zeros_like(positions),
    )
    return MeasurementSet(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=positions.new_tensor([timestamp]),
        values=visible_positions,
        log_variance=log_variance,
        existence_logits=torch.where(
            valid,
            positions.new_full(valid.shape, 8.0),
            positions.new_full(valid.shape, -8.0),
        ),
        measurement_mask=valid,
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary={
            "world_position": visible_positions,
            "world_position_log_variance": log_variance,
        },
    )


def _association(*, ambiguous: torch.Tensor | None = None) -> AssociationResult:
    if ambiguous is None:
        ambiguous = torch.zeros((1, 2), dtype=torch.bool)
    return AssociationResult(
        belief_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        pair_mask=torch.ones((1, 2), dtype=torch.bool),
        pair_cost=torch.zeros((1, 2)),
        unmatched_beliefs=torch.zeros((1, 2), dtype=torch.bool),
        unmatched_measurements=torch.zeros((1, 2), dtype=torch.bool),
        ambiguous=ambiguous,
    )


def _trajectory_positions(timestamp: float) -> torch.Tensor:
    position, _ = free_motion_position_velocity(
        torch.tensor([[[0.05, -0.04, 3.8], [-0.12, 0.06, 4.4]]]),
        torch.tensor([[[0.045, 0.012, -0.008], [-0.038, 0.026, 0.011]]]),
        timestamp,
        gravity=torch.zeros(1, 3),
        drag=torch.tensor([[[0.08], [0.28]]]),
    )
    return position


def _run_drag_history(
    *,
    invalid_frame: int | None = None,
    ambiguous_frame: int | None = None,
    constant: bool = False,
    require_position_gradients: bool = False,
    position_scale: float = 1.0,
    velocity_scale: float = 1.0,
    drag_scale: float = 1.0,
):
    module = _drag_module()
    module.set_development_uncertainty_scales(
        position=position_scale,
        velocity=velocity_scale,
        drag=drag_scale,
    )
    belief = _active_two_object_belief()
    history = None
    raw_positions: list[torch.Tensor] = []
    evidence = None
    for frame_index in range(16):
        timestamp = frame_index * 0.05
        belief = belief.with_timestamp(timestamp)
        positions = (
            torch.tensor([[[0.05, -0.04, 3.8], [-0.12, 0.06, 4.4]]])
            if constant
            else _trajectory_positions(timestamp)
        )
        if require_position_gradients:
            positions = positions.detach().clone().requires_grad_(True)
            raw_positions.append(positions)
        valid = torch.ones((1, 2), dtype=torch.bool)
        if invalid_frame == frame_index:
            valid[0, 1] = False
        ambiguous = torch.zeros((1, 2), dtype=torch.bool)
        if ambiguous_frame == frame_index:
            ambiguous[0, 1] = True
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=_measurement(positions, timestamp, valid=valid),
            association=_association(ambiguous=ambiguous),
            history=history,
        )
        if frame_index < 15:
            assert evidence is None
    return module, belief, evidence, history, raw_positions


def test_uncertainty_buffers_are_conditional_and_set_atomically_by_development_api() -> None:
    disabled = RGBDObservationModule()
    enabled = _drag_module()

    assert disabled.state_dict() == {}
    assert list(disabled.parameters()) == []
    assert set(enabled.state_dict()) == {
        "position_uncertainty_scale",
        "velocity_uncertainty_scale",
        "drag_uncertainty_scale",
    }
    assert list(enabled.parameters()) == []
    for name in enabled.state_dict():
        torch.testing.assert_close(
            getattr(enabled, name),
            torch.tensor(1.0),
            rtol=0.0,
            atol=0.0,
        )

    enabled.set_development_uncertainty_scales(
        position=0.125,
        velocity=torch.tensor(0.25, dtype=torch.float64),
        drag=1.75,
    )
    expected = {
        "position_uncertainty_scale": 0.125,
        "velocity_uncertainty_scale": 0.25,
        "drag_uncertainty_scale": 1.75,
    }
    for name, value in expected.items():
        torch.testing.assert_close(
            getattr(enabled, name),
            torch.tensor(value),
            rtol=0.0,
            atol=0.0,
        )

    invalid_cases = (
        (False, TypeError),
        (torch.tensor(1), TypeError),
        (torch.ones(1), ValueError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (1.0e30, ValueError),
        (1.0e-30, ValueError),
    )
    for field in ("position", "velocity", "drag"):
        for invalid, error_type in invalid_cases:
            before = {name: value.clone() for name, value in enabled.state_dict().items()}
            values = {"position": 0.5, "velocity": 0.75, "drag": 1.25}
            values[field] = invalid
            with pytest.raises(error_type):
                enabled.set_development_uncertainty_scales(**values)
            for name, value in before.items():
                assert torch.equal(getattr(enabled, name), value)

    with pytest.raises(RuntimeError, match="enabled drag estimator"):
        disabled.set_development_uncertainty_scales(
            position=1.0,
            velocity=1.0,
            drag=1.0,
        )

    enabled.position_uncertainty_scale = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="position_uncertainty_scale"):
        enabled._validated_uncertainty_scales()


def test_scales_leave_every_pre_anchor_history_row_and_input_bitwise_unchanged() -> None:
    unit = _drag_module()
    scaled = _drag_module()
    scaled.set_development_uncertainty_scales(
        position=0.125,
        velocity=0.25,
        drag=1.75,
    )
    belief = _active_two_object_belief()
    unit_history = None
    scaled_history = None
    association = _association()
    association_before = {
        name: getattr(association, name).clone()
        for name in (
            "belief_indices",
            "measurement_indices",
            "pair_mask",
            "pair_cost",
            "unmatched_beliefs",
            "unmatched_measurements",
            "ambiguous",
        )
    }
    position_before = belief.objects.position.clone()
    velocity_before = belief.objects.velocity.clone()

    for frame_index in range(15):
        timestamp = frame_index * 0.05
        belief = belief.with_timestamp(timestamp)
        measured = _measurement(_trajectory_positions(timestamp), timestamp)
        unit_evidence, unit_history = unit.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=unit_history,
        )
        scaled_evidence, scaled_history = scaled.update_temporal_history(
            posterior=belief,
            measured=measured,
            association=association,
            history=scaled_history,
        )

        assert unit_evidence is None
        assert scaled_evidence is None
        assert unit_history is not None and scaled_history is not None
        assert unit_history.history_size == scaled_history.history_size
        for name in (
            "object_ids",
            "timestamps",
            "positions",
            "sample_mask",
            "valid_mask",
        ):
            assert torch.equal(getattr(scaled_history, name), getattr(unit_history, name))

    for name, before in association_before.items():
        assert torch.equal(getattr(association, name), before)
    assert torch.equal(belief.objects.position, position_before)
    assert torch.equal(belief.objects.velocity, velocity_before)


def test_sixteen_complete_rows_recover_distinct_per_object_drag_atomically() -> None:
    _, _, evidence, history, _ = _run_drag_history()

    assert evidence is not None
    assert evidence.valid_mask.tolist() == [[True, True]]
    assert evidence.drag_valid_mask is not None
    assert evidence.drag_valid_mask.tolist() == [[True, True]]
    assert evidence.log_drag is not None
    assert evidence.log_drag_log_variance is not None
    assert evidence.position is not None
    assert evidence.position_log_variance is not None
    assert evidence.position_valid_mask is not None
    assert evidence.position_valid_mask.tolist() == [[True, True]]
    torch.testing.assert_close(
        evidence.log_drag.exp(),
        torch.tensor([[[0.08], [0.28]]]),
        rtol=0.0,
        atol=8.0e-3,
    )
    assert evidence.log_drag[0, 0, 0] != evidence.log_drag[0, 1, 0]
    assert torch.isfinite(evidence.velocity).all()
    assert torch.isfinite(evidence.log_variance).all()
    assert torch.isfinite(evidence.log_drag_log_variance).all()
    assert history.sample_mask.all()
    assert history.valid_mask.all()


def test_fit_owned_state_variance_uses_exact_48_over_41_joint_dof_factor() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            temporal_drag_estimation_enabled=True,
            temporal_velocity_variance_floor=1.0e-12,
            temporal_velocity_variance_ceiling=1.0,
        )
    )
    residual_covariance = torch.diag_embed(torch.tensor([[[2.0, 3.0, 4.0]]]))
    normal_matrix = torch.eye(2).reshape(1, 1, 2, 2)
    fit = SimpleNamespace(residual_covariance=residual_covariance)
    conditional_fit = SimpleNamespace(
        normal_matrix=normal_matrix,
        valid=torch.tensor([[True]]),
    )

    position_variance, velocity_variance = module._drag_state_variance(
        fit,  # type: ignore[arg-type]
        conditional_fit,  # type: ignore[arg-type]
        torch.tensor([[True]]),
    )

    expected = residual_covariance.diagonal(dim1=-2, dim2=-1) * (48.0 / 41.0) / 16.0
    torch.testing.assert_close(position_variance, expected, rtol=1.0e-6, atol=0.0)
    torch.testing.assert_close(velocity_variance, expected, rtol=1.0e-6, atol=0.0)


def test_joint_fit_velocity_raw_variance_is_not_owned_by_the_legacy_floor() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            temporal_drag_estimation_enabled=True,
            temporal_velocity_variance_floor=1.0e-6,
            temporal_velocity_variance_ceiling=1.0,
        )
    )
    residual_covariance = torch.diag_embed(torch.full((1, 1, 3), 1.0e-12))
    normal_matrix = torch.eye(2).reshape(1, 1, 2, 2)
    valid = torch.tensor([[True]])
    fit = SimpleNamespace(
        residual_covariance=residual_covariance,
        normal_matrix=normal_matrix,
        valid=valid,
    )

    _, joint_velocity_variance = module._drag_state_variance(
        fit,  # type: ignore[arg-type]
        fit,  # type: ignore[arg-type]
        valid,
    )
    legacy_velocity_variance = module._velocity_variance(  # type: ignore[arg-type]
        fit,
        valid,
    )

    assert torch.all(joint_velocity_variance < 1.0e-6)
    torch.testing.assert_close(
        legacy_velocity_variance,
        torch.full_like(legacy_velocity_variance, 1.0e-6),
        rtol=0.0,
        atol=0.0,
    )


def test_declared_uncertainty_scales_change_owned_variances_by_their_squares() -> None:
    _, _, unit_evidence, _, _ = _run_drag_history()
    _, _, scaled_evidence, _, _ = _run_drag_history(
        position_scale=0.125,
        velocity_scale=0.25,
        drag_scale=2.0,
    )
    assert unit_evidence is not None and scaled_evidence is not None
    assert unit_evidence.log_drag is not None and scaled_evidence.log_drag is not None
    assert unit_evidence.log_drag_log_variance is not None
    assert scaled_evidence.log_drag_log_variance is not None
    assert unit_evidence.position is not None and scaled_evidence.position is not None
    assert unit_evidence.position_log_variance is not None
    assert scaled_evidence.position_log_variance is not None

    torch.testing.assert_close(
        scaled_evidence.position,
        unit_evidence.position,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        scaled_evidence.velocity,
        unit_evidence.velocity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        scaled_evidence.log_drag,
        unit_evidence.log_drag,
        rtol=0.0,
        atol=0.0,
    )
    variance_pairs = (
        (
            scaled_evidence.position_log_variance,
            unit_evidence.position_log_variance,
            0.125**2,
        ),
        (scaled_evidence.log_variance, unit_evidence.log_variance, 0.25**2),
        (
            scaled_evidence.log_drag_log_variance,
            unit_evidence.log_drag_log_variance,
            4.0,
        ),
    )
    for scaled_log_variance, unit_log_variance, expected_ratio in variance_pairs:
        torch.testing.assert_close(
            scaled_log_variance.exp() / unit_log_variance.exp(),
            torch.full_like(scaled_log_variance, expected_ratio),
            rtol=5.0e-6,
            atol=0.0,
        )
        torch.testing.assert_close(
            scaled_log_variance,
            unit_log_variance + math.log(expected_ratio),
            rtol=0.0,
            atol=5.0e-6,
        )


def test_calibrated_drag_variance_respects_the_declared_final_ceiling() -> None:
    _, _, evidence, _, _ = _run_drag_history(drag_scale=100.0)
    assert evidence is not None
    assert evidence.log_drag_log_variance is not None

    torch.testing.assert_close(
        evidence.log_drag_log_variance.exp(),
        torch.full_like(evidence.log_drag_log_variance, 0.25),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("invalid_frame", "ambiguous_frame", "constant"),
    [
        (8, None, False),
        (None, 8, False),
        (None, None, True),
    ],
)
def test_incomplete_ambiguous_or_low_excitation_history_fails_closed(
    invalid_frame: int | None,
    ambiguous_frame: int | None,
    constant: bool,
) -> None:
    _, _, evidence, _, _ = _run_drag_history(
        invalid_frame=invalid_frame,
        ambiguous_frame=ambiguous_frame,
        constant=constant,
    )

    if constant:
        assert evidence is None
    else:
        assert evidence is not None
        assert evidence.valid_mask.tolist() == [[True, False]]
        assert evidence.drag_valid_mask is not None
        assert evidence.drag_valid_mask.tolist() == [[True, False]]
        assert torch.equal(evidence.velocity[0, 1], torch.zeros(3))
        assert evidence.log_drag is not None
        assert torch.equal(evidence.log_drag[0, 1], torch.zeros(1))


def test_drag_fit_preserves_finite_nonzero_gradients_to_every_position_row() -> None:
    _, _, evidence, _, raw_positions = _run_drag_history(require_position_gradients=True)
    assert evidence is not None
    assert evidence.log_drag is not None
    assert evidence.log_drag_log_variance is not None
    assert evidence.position is not None
    assert evidence.position_log_variance is not None

    gradients = torch.autograd.grad(
        (
            evidence.position.sum()
            + evidence.velocity.sum()
            + evidence.log_drag.sum()
            + 0.1 * evidence.position_log_variance.sum()
            + 0.1 * evidence.log_drag_log_variance.sum()
        ),
        raw_positions,
    )

    assert len(gradients) == 16
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0.0


def test_persistent_id_replacement_resets_drag_history_without_carry() -> None:
    module = _drag_module()
    belief = _active_two_object_belief()
    history = None
    for frame_index in range(15):
        timestamp = frame_index * 0.05
        belief = belief.with_timestamp(timestamp)
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=_measurement(_trajectory_positions(timestamp), timestamp),
            association=_association(),
            history=history,
        )
        assert evidence is None

    timestamp = 0.75
    belief = belief.replace(
        objects=belief.objects.replace(
            object_id=torch.tensor([[19, 23]], dtype=torch.int64),
        )
    ).with_timestamp(timestamp)
    evidence, history = module.update_temporal_history(
        posterior=belief,
        measured=_measurement(_trajectory_positions(timestamp), timestamp),
        association=_association(),
        history=history,
    )

    assert evidence is None
    assert history.object_ids.tolist() == [[19, 23]]
    assert history.sample_mask.sum(dim=-1).tolist() == [[1, 1]]


def test_direct_drag_fields_are_atomic_and_require_matching_velocity_support() -> None:
    common = {
        "velocity": torch.zeros(1, 1, 3),
        "log_variance": torch.zeros(1, 1, 3),
        "valid_mask": torch.tensor([[True]]),
        "confidence": torch.ones(1, 1),
    }
    with pytest.raises(ValueError, match="provided together"):
        DirectVelocityEvidence(
            **common,
            log_drag=torch.zeros(1, 1, 1),
        ).validate()
    with pytest.raises(ValueError, match="all-or-nothing triple"):
        DirectVelocityEvidence(
            **{**common, "valid_mask": torch.tensor([[False]])},
            log_drag=torch.zeros(1, 1, 1),
            log_drag_log_variance=torch.zeros(1, 1, 1),
            drag_valid_mask=torch.tensor([[True]]),
        ).validate()
    with pytest.raises(ValueError, match="all three velocity axes"):
        DirectVelocityEvidence(
            **common,
            axis_valid_mask=torch.tensor([[[True, False, True]]]),
            position=torch.zeros(1, 1, 3),
            position_log_variance=torch.zeros(1, 1, 3),
            position_valid_mask=torch.tensor([[True]]),
            log_drag=torch.zeros(1, 1, 1),
            log_drag_log_variance=torch.zeros(1, 1, 1),
            drag_valid_mask=torch.tensor([[True]]),
        ).validate()
    with pytest.raises(ValueError, match="all-or-nothing triple"):
        DirectVelocityEvidence(
            **common,
            position=torch.zeros(1, 1, 3),
            position_log_variance=torch.zeros(1, 1, 3),
            position_valid_mask=torch.tensor([[False]]),
            log_drag=torch.zeros(1, 1, 1),
            log_drag_log_variance=torch.zeros(1, 1, 1),
            drag_valid_mask=torch.tensor([[True]]),
        ).validate()


def test_filter_atomically_applies_distinct_velocity_drag_and_drag_variance() -> None:
    factory = BeliefFactory(max_objects=2, appearance_dim=3)
    base = factory.create()
    prior = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True, True]]),
            object_id=torch.tensor([[7, 11]]),
            position=torch.tensor([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]]),
            velocity=torch.zeros(1, 2, 3),
            log_drag=torch.full((1, 2, 1), math.log(0.05)),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(
            enable_learned_corrector=False,
            minimum_log_variance=-30.0,
        ),
    )
    fitted_log_drag = torch.tensor([[[math.log(0.08)], [math.log(0.28)]]])
    fitted_log_variance = torch.tensor([[[math.log(0.002)], [math.log(0.004)]]])
    fitted_position = torch.tensor([[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]])
    fitted_position_log_variance = torch.full((1, 2, 3), -20.0)
    fitted_velocity = torch.tensor([[[0.04, 0.01, -0.02], [-0.03, 0.02, 0.01]]])
    fitted_velocity_log_variance = torch.full((1, 2, 3), -18.0)
    evidence = DirectVelocityEvidence(
        velocity=fitted_velocity,
        log_variance=fitted_velocity_log_variance,
        valid_mask=torch.tensor([[True, True]]),
        confidence=torch.ones(1, 2),
        axis_valid_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        position=fitted_position,
        position_log_variance=fitted_position_log_variance,
        position_valid_mask=torch.tensor([[True, True]]),
        log_drag=fitted_log_drag,
        log_drag_log_variance=fitted_log_variance,
        drag_valid_mask=torch.tensor([[True, True]]),
    )

    posterior = updater.correct_direct_velocity(prior, evidence)

    torch.testing.assert_close(
        posterior.objects.position,
        fitted_position,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        posterior.objects.velocity,
        fitted_velocity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        posterior.objects.fast_log_variance[..., :3],
        fitted_position_log_variance,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        posterior.objects.fast_log_variance[..., 3:6],
        fitted_velocity_log_variance,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        posterior.objects.log_drag,
        fitted_log_drag,
        rtol=0.0,
        atol=0.0,
    )
    drag_slice = slow_packing_map(prior.objects)["log_drag"]
    torch.testing.assert_close(
        posterior.objects.slow_log_variance[..., drag_slice],
        fitted_log_variance,
        rtol=0.0,
        atol=0.0,
    )


def test_complete_drag_state_variances_are_owned_by_the_configured_filter_clamp() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=3)
    base = factory.create()
    prior = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[7]]),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    evidence = DirectVelocityEvidence(
        velocity=torch.tensor([[[0.1, 0.2, 0.3]]]),
        log_variance=torch.full((1, 1, 3), -20.0),
        valid_mask=torch.tensor([[True]]),
        confidence=torch.ones(1, 1),
        position=torch.tensor([[[1.0, 2.0, 3.0]]]),
        position_log_variance=torch.full((1, 1, 3), -21.0),
        position_valid_mask=torch.tensor([[True]]),
        log_drag=torch.tensor([[[math.log(0.12)]]]),
        log_drag_log_variance=torch.full((1, 1, 1), -22.0),
        drag_valid_mask=torch.tensor([[True]]),
    )

    posterior = updater.correct_direct_velocity(prior, evidence)

    torch.testing.assert_close(
        posterior.objects.fast_log_variance[..., :6],
        torch.full_like(posterior.objects.fast_log_variance[..., :6], -12.0),
        rtol=0.0,
        atol=0.0,
    )
    drag_slice = slow_packing_map(prior.objects)["log_drag"]
    torch.testing.assert_close(
        posterior.objects.slow_log_variance[..., drag_slice],
        torch.full_like(posterior.objects.slow_log_variance[..., drag_slice], -12.0),
        rtol=0.0,
        atol=0.0,
    )


def test_complete_drag_state_direct_replacement_preserves_input_gradients() -> None:
    factory = BeliefFactory(max_objects=1, appearance_dim=3)
    base = factory.create()
    prior = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[7]]),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(
            enable_learned_corrector=False,
            minimum_log_variance=-30.0,
        ),
    )
    position = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    velocity = torch.tensor([[[0.1, 0.2, 0.3]]], requires_grad=True)
    position_log_variance = torch.full((1, 1, 3), -20.0, requires_grad=True)
    velocity_log_variance = torch.full((1, 1, 3), -18.0, requires_grad=True)
    log_drag = torch.tensor([[[math.log(0.12)]]], requires_grad=True)
    log_drag_log_variance = torch.full((1, 1, 1), -16.0, requires_grad=True)
    evidence = DirectVelocityEvidence(
        velocity=velocity,
        log_variance=velocity_log_variance,
        valid_mask=torch.tensor([[True]]),
        confidence=torch.ones(1, 1),
        position=position,
        position_log_variance=position_log_variance,
        position_valid_mask=torch.tensor([[True]]),
        log_drag=log_drag,
        log_drag_log_variance=log_drag_log_variance,
        drag_valid_mask=torch.tensor([[True]]),
    )

    posterior = updater.correct_direct_velocity(prior, evidence)
    drag_slice = slow_packing_map(prior.objects)["log_drag"]
    loss = (
        posterior.objects.position.sum()
        + posterior.objects.velocity.sum()
        + posterior.objects.log_drag.sum()
        + posterior.objects.fast_log_variance[..., :6].sum()
        + posterior.objects.slow_log_variance[..., drag_slice].sum()
    )
    gradients = torch.autograd.grad(
        loss,
        (
            position,
            velocity,
            position_log_variance,
            velocity_log_variance,
            log_drag,
            log_drag_log_variance,
        ),
    )

    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0.0


def test_invalid_drag_slot_leaves_slow_state_bitwise_unchanged() -> None:
    factory = BeliefFactory(max_objects=2, appearance_dim=3)
    base = factory.create()
    prior = base.replace(
        objects=base.objects.replace(
            active=torch.tensor([[True, True]]),
            object_id=torch.tensor([[7, 11]]),
            log_drag=torch.tensor([[[math.log(0.05)], [math.log(0.12)]]]),
        )
    )
    updater = BeliefUpdater(
        fast_state_dim=factory.fast_state_dim,
        num_motion_modes=NUM_MOTION_MODES,
        config=BeliefUpdaterConfig(enable_learned_corrector=False),
    )
    evidence = DirectVelocityEvidence(
        velocity=torch.tensor([[[0.2, 0.3, 0.4], [9.0, 9.0, 9.0]]]),
        log_variance=torch.full((1, 2, 3), -8.0),
        valid_mask=torch.tensor([[True, False]]),
        confidence=torch.ones(1, 2),
        position=torch.tensor([[[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]]]),
        position_log_variance=torch.full((1, 2, 3), -9.0),
        position_valid_mask=torch.tensor([[True, False]]),
        log_drag=torch.tensor([[[math.log(0.08)], [math.log(0.28)]]]),
        log_drag_log_variance=torch.tensor([[[math.log(0.002)], [math.log(0.004)]]]),
        drag_valid_mask=torch.tensor([[True, False]]),
    )

    posterior = updater.correct_direct_velocity(prior, evidence)

    torch.testing.assert_close(
        posterior.objects.log_drag[0, 0],
        evidence.log_drag[0, 0],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(posterior.objects.log_drag[0, 1], prior.objects.log_drag[0, 1])
    drag_slice = slow_packing_map(prior.objects)["log_drag"]
    assert torch.equal(
        posterior.objects.slow_log_variance[0, 1, drag_slice],
        prior.objects.slow_log_variance[0, 1, drag_slice],
    )
    assert torch.equal(posterior.objects.position[0, 1], prior.objects.position[0, 1])
    assert torch.equal(posterior.objects.velocity[0, 1], prior.objects.velocity[0, 1])
    assert torch.equal(
        posterior.objects.fast_log_variance[0, 1],
        prior.objects.fast_log_variance[0, 1],
    )
