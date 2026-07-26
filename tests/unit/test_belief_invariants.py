from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.belief import (
    BeliefFactory,
    HypothesisSet,
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
