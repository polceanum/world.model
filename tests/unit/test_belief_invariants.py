from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.belief import (
    BeliefFactory,
    HypothesisSet,
    LifecycleConfig,
    MotionMode,
    ObjectLifecycle,
)
from world_model.observations import MeasurementSet


def _activate_first(belief: object) -> object:
    world = belief
    objects = world.objects.clone()
    objects.active[:, 0] = True
    objects.object_id[:, 0] = torch.arange(
        objects.batch_size,
        device=objects.object_id.device,
    )
    return replace(
        world,
        objects=objects,
        next_object_id=torch.ones_like(world.next_object_id),
    )


def test_factory_creates_valid_modality_independent_belief() -> None:
    belief = BeliefFactory(max_objects=4).create(
        batch_size=2,
        timestamp=torch.tensor([0.0, 0.25]),
        active_modalities=("rgb",),
    )

    assert belief.validate() is belief
    assert belief.timestamp.shape == (2,)
    assert belief.objects.active.shape == (2, 4)
    assert not belief.objects.active.any()
    assert (belief.objects.object_id == -1).all()
    assert belief.active_modalities == ("rgb",)
    assert not hasattr(belief.objects, "rgb")


def test_validator_rejects_duplicate_and_invalid_object_ids() -> None:
    belief = BeliefFactory(max_objects=3).create()
    objects = belief.objects.clone()
    objects.active[0, :2] = True
    objects.object_id[0, :2] = 7

    with pytest.raises(ValueError, match="unique"):
        replace(belief, objects=objects).validate()

    objects.object_id[0, 1] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        replace(belief, objects=objects).validate()


def test_validator_rejects_nonunit_quaternion_and_nonfinite_variance() -> None:
    belief = BeliefFactory(max_objects=2).create()
    objects = belief.objects.clone()
    objects.orientation[0, 0] = 0
    with pytest.raises(ValueError, match="unit norm"):
        replace(belief, objects=objects).validate()

    objects = belief.objects.clone()
    objects.fast_log_variance[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        replace(belief, objects=objects).validate()


def test_clone_detach_and_to_are_copy_safe() -> None:
    belief = _activate_first(BeliefFactory(max_objects=2).create())
    belief.objects.position.requires_grad_(True)
    cloned = belief.clone()
    detached = belief.detach()
    converted = belief.to(dtype=torch.float64)

    assert cloned.objects.position.data_ptr() != belief.objects.position.data_ptr()
    cloned.objects.position.detach()[0, 0, 0] = 12.0
    assert belief.objects.position[0, 0, 0] == 0
    assert not detached.objects.position.requires_grad
    assert torch.equal(detached.objects.active, belief.objects.active)
    assert converted.objects.position.dtype == torch.float64
    assert converted.objects.object_id.dtype == torch.int64
    assert converted.objects.active.dtype == torch.bool


def test_timestamp_helper_rejects_time_reversal() -> None:
    belief = BeliefFactory(max_objects=1).create(timestamp=1.0)
    updated = belief.with_timestamp(1.25)
    assert updated.timestamp.item() == pytest.approx(1.25)
    assert belief.timestamp.item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="backward"):
        belief.with_timestamp(0.99)


def test_hypothesis_set_normalises_and_reweights() -> None:
    belief = BeliefFactory(max_objects=1).create(batch_size=2)
    hypotheses = HypothesisSet(
        beliefs=[belief.clone(), belief.clone()],
        log_weights=torch.tensor([[0.0, -1.0], [-2.0, 0.0]]),
    ).validate()
    assert torch.allclose(hypotheses.normalized_weights.sum(-1), torch.ones(2))
    updated = hypotheses.reweight(torch.tensor([[0.0, 2.0], [0.0, 0.0]]))
    assert updated.best(0) is updated.beliefs[1]


def test_lifecycle_birth_allocates_monotonic_id_from_measurement() -> None:
    belief = BeliefFactory(max_objects=2, appearance_dim=3).create()
    measurements = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([0.0]),
        values=torch.zeros(1, 1, 4),
        log_variance=torch.zeros(1, 1, 4),
        existence_logits=torch.tensor([[4.0]]),
        measurement_mask=torch.tensor([[True]]),
        appearance=torch.tensor([[[1.0, 2.0, 2.0]]]),
        class_logits=None,
        frame_id="camera:test",
        supported_state_fields=("position",),
        auxiliary={
            "world_position": torch.tensor([[[0.25, 1.0, -0.5]]]),
            "world_radius": torch.tensor([[[0.2]]]),
        },
    )

    born = ObjectLifecycle().birth_from_measurements(
        belief,
        measurements,
        torch.tensor([[True]]),
    )

    assert born.objects.active[0, 0]
    assert born.objects.object_id[0, 0] == 0
    assert born.next_object_id[0] == 1
    torch.testing.assert_close(
        born.objects.position[0, 0],
        torch.tensor([0.25, 1.0, -0.5]),
    )
    assert born.objects.mode[0, 0] == MotionMode.CREATED
    assert torch.linalg.vector_norm(born.objects.appearance[0, 0]).item() == pytest.approx(1.0)
    belief.validate()
    born.validate()


def _tentative_measurement(timestamp: float, positions: torch.Tensor) -> MeasurementSet:
    positions = positions.reshape(1, -1, 3)
    proposals = positions.shape[1]
    return MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([timestamp]),
        values=torch.zeros(1, proposals, 4),
        log_variance=torch.zeros(1, proposals, 4),
        existence_logits=torch.full((1, proposals), 4.0),
        measurement_mask=torch.ones(1, proposals, dtype=torch.bool),
        appearance=None,
        class_logits=None,
        frame_id="camera:test",
        supported_state_fields=("position",),
        auxiliary={"world_position": positions},
    )


def test_tentative_birth_requires_two_consistent_increasing_time_detections() -> None:
    lifecycle = ObjectLifecycle(
        LifecycleConfig(
            birth_confirmations=2,
            birth_confirmation_distance_m=0.5,
        )
    )
    first = _tentative_measurement(0.0, torch.tensor([[0.0, 0.0, 0.0]]))
    confirmed, state = lifecycle.confirm_tentative_births(
        first,
        torch.tensor([[True]]),
        None,
        confidence_threshold=0.55,
    )
    assert not confirmed.any()
    assert state.active[0, 0]
    assert state.confirmation_count[0, 0] == 1

    second = _tentative_measurement(0.1, torch.tensor([[0.2, 0.0, 0.0]]))
    confirmed, state = lifecycle.confirm_tentative_births(
        second,
        torch.tensor([[True]]),
        state,
        confidence_threshold=0.55,
    )
    assert confirmed[0, 0]
    assert not state.active.any()
    assert not state.confirmation_count.any()


def test_tentative_birth_restarts_after_far_or_same_timestamp_detection() -> None:
    lifecycle = ObjectLifecycle(
        LifecycleConfig(
            birth_confirmations=2,
            birth_confirmation_distance_m=0.5,
        )
    )
    first = _tentative_measurement(0.0, torch.tensor([[0.0, 0.0, 0.0]]))
    _, state = lifecycle.confirm_tentative_births(
        first,
        torch.tensor([[True]]),
        None,
        confidence_threshold=0.55,
    )
    same_time = _tentative_measurement(0.0, torch.tensor([[0.1, 0.0, 0.0]]))
    confirmed, state = lifecycle.confirm_tentative_births(
        same_time,
        torch.tensor([[True]]),
        state,
        confidence_threshold=0.55,
    )
    assert not confirmed.any()
    assert state.confirmation_count[0, 0] == 1

    far = _tentative_measurement(0.1, torch.tensor([[0.7, 0.0, 0.0]]))
    confirmed, state = lifecycle.confirm_tentative_births(
        far,
        torch.tensor([[True]]),
        state,
        confidence_threshold=0.55,
    )
    assert not confirmed.any()
    assert state.confirmation_count[0, 0] == 1


def test_tentative_birth_assignment_maximizes_in_gate_confirmations() -> None:
    lifecycle = ObjectLifecycle(
        LifecycleConfig(
            birth_confirmations=2,
            birth_confirmation_distance_m=0.5,
        )
    )
    prior_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.22916667, 0.438309, 0.0],
        ]
    )
    first = _tentative_measurement(0.0, prior_positions)
    _, state = lifecycle.confirm_tentative_births(
        first,
        torch.tensor([[True, True]]),
        None,
        confidence_threshold=0.55,
    )
    # Raw minimum-distance Hungarian prefers 0.01 + 0.51 and would discard the
    # second edge after gating. Cardinality-first gating must instead retain
    # both approximately-0.49 m cross assignments.
    second = _tentative_measurement(
        0.1,
        torch.tensor(
            [
                [0.01, 0.0, 0.0],
                [0.49, 0.0, 0.0],
            ]
        ),
    )
    confirmed, state = lifecycle.confirm_tentative_births(
        second,
        torch.tensor([[True, True]]),
        state,
        confidence_threshold=0.55,
    )

    assert confirmed.all()
    assert not state.active.any()


def test_single_confirmation_preserves_immediate_birth_semantics() -> None:
    lifecycle = ObjectLifecycle(LifecycleConfig(birth_confirmations=1))
    measurement = _tentative_measurement(0.0, torch.tensor([[0.0, 0.0, 0.0]]))
    confirmed, state = lifecycle.confirm_tentative_births(
        measurement,
        torch.tensor([[True]]),
        None,
        confidence_threshold=0.55,
    )
    assert confirmed[0, 0]
    assert not state.active.any()


def test_lifecycle_birth_prioritizes_confidence_when_slots_are_limited() -> None:
    belief = BeliefFactory(max_objects=1).create()
    probabilities = torch.tensor([[0.6, 0.9]])
    measurements = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([0.0]),
        values=torch.zeros(1, 2, 4),
        log_variance=torch.zeros(1, 2, 4),
        existence_logits=torch.logit(probabilities),
        measurement_mask=torch.tensor([[True, True]]),
        appearance=None,
        class_logits=None,
        frame_id="camera:test",
        supported_state_fields=("position",),
        auxiliary={"world_position": torch.tensor([[[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]])},
    )

    born = ObjectLifecycle().birth_from_measurements(
        belief,
        measurements,
        torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(
        born.objects.position[0, 0],
        torch.tensor([0.9, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        born.objects.existence_logit[0, 0],
        torch.logit(torch.tensor(0.9)),
    )
    assert born.next_object_id[0] == 1
    born.validate()


def test_recycled_birth_slot_does_not_inherit_previous_identity_state() -> None:
    belief = BeliefFactory(
        max_objects=1,
        geometry_dim=3,
        appearance_dim=4,
        residual_dynamics_dim=3,
        modal_count=2,
        modal_dim=2,
        parameter_memory_dim=5,
    ).create()
    stale = belief.objects.clone()
    stale.active[0, 0] = False
    stale.object_id[0, 0] = -1
    stale.geometry[0, 0].fill_(9.0)
    stale.appearance[0, 0].fill_(8.0)
    stale.residual_dynamics[0, 0].fill_(7.0)
    stale.modal_state[0, 0].fill_(6.0)
    stale.modal_frequency[0, 0].fill_(5.0)
    stale.modal_decay_raw[0, 0].fill_(4.0)
    stale.log_mass[0, 0].fill_(3.0)
    stale.restitution_logit[0, 0].fill_(-3.0)
    stale.log_drag[0, 0].fill_(2.0)
    stale.friction_logit[0, 0].fill_(-2.0)
    stale.fast_log_variance[0, 0].fill_(1.5)
    stale.slow_log_variance[0, 0].fill_(1.25)
    stale.parameter_memory[0, 0].fill_(11.0)
    belief = belief.replace(
        objects=stale,
        next_object_id=torch.tensor([41]),
    )
    measurements = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([0.0]),
        values=torch.zeros(1, 1, 4),
        log_variance=torch.zeros(1, 1, 4),
        existence_logits=torch.tensor([[8.0]]),
        measurement_mask=torch.tensor([[True]]),
        appearance=None,
        class_logits=None,
        frame_id="camera:test",
        supported_state_fields=("position",),
        auxiliary={"world_position": torch.tensor([[[0.5, 1.0, 2.0]]])},
    )

    born = ObjectLifecycle().birth_from_measurements(
        belief,
        measurements,
        torch.tensor([[True]]),
    )
    objects = born.objects

    assert objects.object_id[0, 0] == 41
    torch.testing.assert_close(objects.geometry[0, 0], torch.tensor([0.1, 0.0, 0.0]))
    assert not objects.appearance[0, 0].any()
    assert not objects.residual_dynamics[0, 0].any()
    assert not objects.modal_state[0, 0].any()
    assert not objects.modal_frequency[0, 0].any()
    assert not objects.modal_decay_raw[0, 0].any()
    assert objects.mass[0, 0].item() == pytest.approx(1.0)
    assert objects.restitution[0, 0].item() == pytest.approx(0.7)
    assert objects.drag[0, 0].item() == pytest.approx(0.05)
    assert objects.friction[0, 0].item() == pytest.approx(0.2)
    assert not objects.fast_log_variance[0, 0].any()
    assert not objects.slow_log_variance[0, 0].any()
    assert not objects.parameter_memory[0, 0].any()
    born.validate()
