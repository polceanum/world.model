from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.fusion import AssociationResult
from world_model.observations import MeasurementSet
from world_model.observations.rgbd import RGBDObservationConfig, RGBDObservationModule
from world_model.observations.rgbd.temporal import RGBDTemporalPositionHistory

OBJECT_IDS = torch.tensor([[7]], dtype=torch.int64)
ACTIVE = torch.tensor([[True]])


def _empty() -> RGBDTemporalPositionHistory:
    return RGBDTemporalPositionHistory.empty(
        object_ids=OBJECT_IDS,
        active_mask=ACTIVE,
        history_size=16,
        dtype=torch.float32,
    )


def _append(
    history: RGBDTemporalPositionHistory,
    *,
    timestamp: float,
    x: float,
    valid: bool = True,
    reset: bool = False,
) -> RGBDTemporalPositionHistory:
    return history.append(
        object_ids=OBJECT_IDS,
        active_mask=ACTIVE,
        append_mask=ACTIVE,
        timestamp=torch.tensor([timestamp]),
        positions=torch.tensor([[[x, 0.0, 2.0]]]),
        valid_mask=torch.tensor([[valid]]),
        minimum_dt=1.0e-3,
        reset_mask=torch.tensor([[reset]]),
    )


def _fit(
    history: RGBDTemporalPositionHistory,
    *,
    require_complete_window: bool,
):
    return history.fit(
        gravity=torch.zeros(1, 3),
        drag=torch.zeros(1, 1, 1),
        minimum_support=3,
        minimum_dt=1.0e-3,
        conditioning_limit=1.0e8,
        require_complete_window=require_complete_window,
    )


def test_short_segment_is_opt_in_and_complete_window_semantics_are_unchanged() -> None:
    history = _empty()
    for index in range(3):
        history = _append(
            history,
            timestamp=0.05 * index,
            x=0.1 * index,
        )

    _, complete_valid = _fit(history, require_complete_window=True)
    partial_fit, partial_valid = _fit(history, require_complete_window=False)

    assert not complete_valid.any()
    assert partial_valid.all()
    assert partial_fit.support_count.item() == 3
    torch.testing.assert_close(
        partial_fit.velocity,
        torch.tensor([[[2.0, 0.0, 0.0]]]),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_contact_reset_discards_pre_event_rows_before_partial_fit() -> None:
    history = _empty()
    for index in range(6):
        history = _append(
            history,
            timestamp=0.05 * index,
            x=0.05 * index,
        )

    history = _append(history, timestamp=0.30, x=10.0, reset=True)
    assert history.sample_mask.sum().item() == 1
    history = _append(history, timestamp=0.35, x=10.1)
    before_fit, before_valid = _fit(history, require_complete_window=False)
    assert before_fit.support_count.item() == 2
    assert not before_valid.any()

    history = _append(history, timestamp=0.40, x=10.2)
    after_fit, after_valid = _fit(history, require_complete_window=False)
    assert after_valid.all()
    assert after_fit.support_count.item() == 3
    torch.testing.assert_close(
        after_fit.velocity,
        torch.tensor([[[2.0, 0.0, 0.0]]]),
        rtol=3.0e-5,
        atol=5.0e-5,
    )


def test_partial_segment_does_not_skip_an_invalid_observation() -> None:
    history = _empty()
    history = _append(history, timestamp=0.0, x=0.0)
    history = _append(history, timestamp=0.05, x=0.1, valid=False)
    history = _append(history, timestamp=0.10, x=0.2)

    fit, valid = _fit(history, require_complete_window=False)

    assert fit.support_count.item() == 2
    assert not valid.any()


def test_reset_rejects_inactive_slots() -> None:
    history = _empty()
    with pytest.raises(ValueError, match="only active slots"):
        history.append(
            object_ids=torch.tensor([[-1]], dtype=torch.int64),
            active_mask=torch.tensor([[False]]),
            append_mask=torch.tensor([[False]]),
            timestamp=torch.tensor([0.0]),
            positions=torch.zeros(1, 1, 3),
            valid_mask=torch.tensor([[False]]),
            minimum_dt=1.0e-3,
            reset_mask=torch.tensor([[True]]),
        )


def _set_belief(timestamp: float):
    belief = BeliefFactory(max_objects=6, appearance_dim=8).create(
        timestamp=timestamp,
        gravity=(0.0, 0.0, 0.0),
    )
    active = torch.zeros_like(belief.objects.active)
    active[:, 0] = True
    object_id = torch.full_like(belief.objects.object_id, -1)
    object_id[:, 0] = 7
    mode_logits = belief.objects.motion_mode_logits.new_full(
        belief.objects.motion_mode_logits.shape,
        -4.0,
    )
    mode_logits[..., MotionMode.FREE] = 4.0
    return belief.replace(
        objects=belief.objects.replace(
            active=active,
            object_id=object_id,
            motion_mode_logits=mode_logits,
        )
    )


def _set_measurement(
    timestamp: float,
    x: float,
    *,
    collision: bool = False,
    known_action: bool = False,
) -> MeasurementSet:
    values = torch.zeros(1, 8, 3)
    values[0, 0] = torch.tensor([x, 0.0, 2.0])
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, 0] = True
    variance = torch.full_like(values, -9.0)
    return MeasurementSet(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=torch.tensor([timestamp]),
        values=values,
        log_variance=variance,
        existence_logits=torch.where(mask, torch.tensor(8.0), torch.tensor(-8.0)),
        measurement_mask=mask,
        appearance=torch.zeros(1, 8, 8),
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary={
            "world_position": values,
            "world_position_log_variance": variance,
            "prior_interval_collision_mask": torch.tensor(
                [[collision, False, False, False, False, False]]
            ),
            "prior_interval_known_action_mask": torch.tensor(
                [[known_action, False, False, False, False, False]]
            ),
        },
    )


def _set_association() -> AssociationResult:
    return AssociationResult(
        belief_indices=torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0, -1, -1, -1, -1, -1]], dtype=torch.int64),
        pair_mask=torch.tensor([[True, False, False, False, False, False]]),
        pair_cost=torch.zeros(1, 6),
        unmatched_beliefs=torch.tensor([[False, True, True, True, True, True]]),
        unmatched_measurements=torch.tensor([[False, True, True, True, True, True, True, True]]),
        ambiguous=torch.zeros(1, 6, dtype=torch.bool),
    )


def test_set_module_resets_on_runtime_collision_and_emits_after_three_rows() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=8,
            appearance_dim=8,
            temporal_min_samples=3,
            observation_mode="set",
            max_objects=6,
            fit_conditioning_limit=10_000.0,
        )
    )
    history = RGBDTemporalPositionHistory.empty(
        object_ids=_set_belief(0.0).objects.object_id,
        active_mask=_set_belief(0.0).objects.active,
        history_size=16,
        dtype=torch.float32,
    )
    for index in range(5):
        history = history.append(
            object_ids=_set_belief(0.05 * index).objects.object_id,
            active_mask=_set_belief(0.05 * index).objects.active,
            append_mask=_set_belief(0.05 * index).objects.active,
            timestamp=torch.tensor([0.05 * index]),
            positions=torch.tensor([[[0.1 * index, 0.0, 2.0]]]).expand(1, 6, 3).clone(),
            valid_mask=_set_belief(0.05 * index).objects.active,
            minimum_dt=1.0e-3,
        )

    evidence, history = module.update_temporal_history(
        posterior=_set_belief(0.25),
        measured=_set_measurement(0.25, 4.0, collision=True),
        association=_set_association(),
        history=history,
    )
    assert evidence is None
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.sum() == 1

    evidence, history = module.update_temporal_history(
        posterior=_set_belief(0.30),
        measured=_set_measurement(0.30, 4.1),
        association=_set_association(),
        history=history,
    )
    assert evidence is None
    evidence, history = module.update_temporal_history(
        posterior=_set_belief(0.35),
        measured=_set_measurement(0.35, 4.2),
        association=_set_association(),
        history=history,
    )
    assert evidence is not None
    assert evidence.valid_mask[0, 0]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask[0, 0].sum() == 3


def test_set_module_starts_new_segment_after_known_action() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=8,
            appearance_dim=8,
            temporal_min_samples=3,
            observation_mode="set",
            max_objects=6,
            fit_conditioning_limit=10_000.0,
        )
    )
    history = RGBDTemporalPositionHistory.empty(
        object_ids=_set_belief(0.0).objects.object_id,
        active_mask=_set_belief(0.0).objects.active,
        history_size=16,
        dtype=torch.float32,
    )
    for index in range(5):
        history = history.append(
            object_ids=_set_belief(0.05 * index).objects.object_id,
            active_mask=_set_belief(0.05 * index).objects.active,
            append_mask=_set_belief(0.05 * index).objects.active,
            timestamp=torch.tensor([0.05 * index]),
            positions=torch.tensor([[[0.1 * index, 0.0, 2.0]]]).expand(1, 6, 3).clone(),
            valid_mask=_set_belief(0.05 * index).objects.active,
            minimum_dt=1.0e-3,
        )

    evidence, history = module.update_temporal_history(
        posterior=_set_belief(0.25),
        measured=_set_measurement(0.25, 4.0, known_action=True),
        association=_set_association(),
        history=history,
    )

    assert evidence is None
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.sum() == 1


def test_set_config_requires_three_samples_while_legacy_stays_sixteen() -> None:
    with pytest.raises(ValueError, match="legacy.*equal"):
        RGBDObservationConfig(temporal_min_samples=3)
    with pytest.raises(ValueError, match="set.*three"):
        RGBDObservationConfig(
            proposal_count=8,
            appearance_dim=8,
            temporal_min_samples=16,
            observation_mode="set",
            max_objects=6,
        )


def test_set_velocity_variance_has_exact_sixteen_over_support_inflation() -> None:
    history = _empty()
    for timestamp, x in ((0.0, 0.0), (0.05, 0.11), (0.10, 0.25)):
        history = _append(history, timestamp=timestamp, x=x)
    fit, valid = history.fit(
        gravity=torch.zeros(1, 3),
        drag=torch.zeros(1, 1, 1),
        minimum_support=3,
        minimum_dt=1.0e-3,
        conditioning_limit=10_000.0,
        require_complete_window=False,
    )
    module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=8,
            appearance_dim=8,
            temporal_min_samples=3,
            temporal_velocity_variance_floor=1.0e-12,
            temporal_velocity_variance_ceiling=100.0,
            fit_conditioning_limit=10_000.0,
            observation_mode="set",
            max_objects=6,
        )
    )

    actual = module._velocity_variance(fit, valid)
    support = fit.support_count.to(fit.normal_matrix.dtype)
    inverse_normal = torch.linalg.inv(fit.normal_matrix)
    uninflated = (
        fit.residual_covariance * (support / (support - 2.0)).unsqueeze(-1).unsqueeze(-1)
    ).diagonal(dim1=-2, dim2=-1) * (inverse_normal[..., 1, 1] / support).unsqueeze(-1)
    expected = uninflated.clamp_min(1.0e-12) * (16.0 / support).square().unsqueeze(-1)

    torch.testing.assert_close(actual, expected)
    expected_floor = actual.new_tensor(1.0e-12) * (
        actual.new_tensor(16.0) / support
    ).square().unsqueeze(-1)
    torch.testing.assert_close(
        actual[..., 1:],
        expected_floor.expand_as(actual)[..., 1:],
        rtol=0.0,
        atol=0.0,
    )


def test_mature_set_segment_is_bitwise_the_accepted_sixteen_sample_estimator() -> None:
    history = _empty()
    for index in range(16):
        # Retain a small deterministic residual so covariance arithmetic is
        # compared as well as the fitted mean.
        history = _append(
            history,
            timestamp=0.05 * index,
            x=0.10 * index + (0.001 if index % 3 == 0 else -0.0005),
        )

    complete, complete_valid = _fit(history, require_complete_window=True)
    set_mode, set_valid = _fit(history, require_complete_window=False)

    assert torch.equal(complete_valid, set_valid)
    for item in fields(complete):
        assert torch.equal(getattr(complete, item.name), getattr(set_mode, item.name))

    common = dict(
        temporal_velocity_variance_floor=1.0e-12,
        temporal_velocity_variance_ceiling=100.0,
        fit_conditioning_limit=1.0e8,
    )
    legacy_module = RGBDObservationModule(RGBDObservationConfig(**common))
    set_module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=8,
            appearance_dim=8,
            temporal_min_samples=3,
            observation_mode="set",
            max_objects=6,
            **common,
        )
    )
    assert torch.equal(
        legacy_module._velocity_variance(complete, complete_valid),
        set_module._velocity_variance(set_mode, set_valid),
    )
