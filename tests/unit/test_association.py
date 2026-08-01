from __future__ import annotations

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.fusion import Associator, build_innovation
from world_model.observations import MeasurementSet, PredictedMeasurements


def _belief_and_prediction() -> tuple[object, PredictedMeasurements]:
    belief = BeliefFactory(max_objects=3, appearance_dim=4).create()
    objects = belief.objects.replace(
        active=torch.tensor([[True, True, False]]),
        object_id=torch.tensor([[10, 11, -1]]),
    )
    belief = belief.replace(objects=objects)
    values = torch.tensor([[[0.0, 0.0, -1.0, 0.5], [0.8, 0.2, -1.2, 0.4], [0.0] * 4]])
    predicted = PredictedMeasurements(
        modality="rgb",
        sensor_id="camera",
        timestamp=belief.timestamp,
        values=values,
        log_variance=torch.full_like(values, -4.0),
        object_ids=objects.object_id,
        belief_indices=torch.tensor([[0, 1, 2]]),
        valid_mask=objects.active,
        visibility=torch.tensor([[1.0, 1.0, 0.0]]),
        appearance=None,
    )
    return belief, predicted


def _measurements(values: torch.Tensor) -> MeasurementSet:
    batch, count, _ = values.shape
    return MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.zeros(batch),
        values=values,
        log_variance=torch.full_like(values, -4.0),
        existence_logits=torch.full((batch, count), 5.0),
        measurement_mask=torch.ones(batch, count, dtype=torch.bool),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera",
        supported_state_fields=("position",),
    )


def test_hungarian_association_matches_swapped_obvious_pairs() -> None:
    belief, predicted = _belief_and_prediction()
    measurements = _measurements(torch.tensor([[[0.81, 0.2, -1.2, 0.4], [0.01, 0.0, -1.0, 0.5]]]))
    result = Associator(mahalanobis_gate=25.0).match(belief, measurements, predicted)
    pairs = {
        (int(belief_index), int(measurement_index))
        for belief_index, measurement_index, valid in zip(
            result.belief_indices[0],
            result.measurement_indices[0],
            result.pair_mask[0],
            strict=True,
        )
        if bool(valid)
    }
    assert pairs == {(0, 1), (1, 0)}
    assert not result.unmatched_beliefs[0, :2].any()
    result.validate()


def test_association_gates_impossible_pair_as_unmatched() -> None:
    belief, predicted = _belief_and_prediction()
    measurements = _measurements(torch.tensor([[[10.0, 10.0, 5.0, 5.0]]]))
    result = Associator(mahalanobis_gate=4.0).match(belief, measurements, predicted)
    assert not result.pair_mask.any()
    assert result.unmatched_measurements[0, 0]
    assert result.unmatched_beliefs[0, :2].all()


def test_association_records_close_cost_ambiguity_without_duplicates() -> None:
    belief, predicted = _belief_and_prediction()
    predicted.values[0, 1] = torch.tensor([0.02, 0.0, -1.0, 0.5])
    measurements = _measurements(torch.tensor([[[0.01, 0.0, -1.0, 0.5]]]))
    result = Associator(
        mahalanobis_gate=25.0,
        ambiguity_margin=1.0,
    ).match(belief, measurements, predicted)
    assert result.pair_mask.sum() == 1
    assert result.ambiguous[result.pair_mask].all()
    result.validate()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_association_transfers_cost_to_cpu_without_mps_float64() -> None:
    belief, predicted = _belief_and_prediction()
    belief = belief.to(device="mps")
    predicted = PredictedMeasurements(
        modality=predicted.modality,
        sensor_id=predicted.sensor_id,
        timestamp=predicted.timestamp.to("mps"),
        values=predicted.values.to("mps"),
        log_variance=predicted.log_variance.to("mps"),
        object_ids=predicted.object_ids.to("mps"),
        belief_indices=predicted.belief_indices.to("mps"),
        valid_mask=predicted.valid_mask.to("mps"),
        visibility=predicted.visibility.to("mps"),
    )
    measurements = _measurements(
        torch.tensor([[[0.81, 0.2, -1.2, 0.4], [0.01, 0.0, -1.0, 0.5]]])
    ).to(device="mps")
    result = Associator().match(belief, measurements, predicted)
    assert result.pair_mask.device.type == "mps"
    assert result.pair_mask.sum().item() == 2


def test_association_maps_reordered_prediction_rows_to_persistent_belief_slots() -> None:
    belief = BeliefFactory(max_objects=3, appearance_dim=4).create()
    objects = belief.objects.replace(
        active=torch.tensor([[True, False, True]]),
        object_id=torch.tensor([[10, -1, 12]]),
    )
    belief = belief.replace(objects=objects)
    # Projection rows are deliberately ordered [slot 2, slot 0].
    predicted_values = torch.tensor([[[2.0, 0.0, -1.0, 0.4], [0.0, 0.0, -1.0, 0.5]]])
    predicted = PredictedMeasurements(
        modality="rgb",
        sensor_id="camera",
        timestamp=belief.timestamp,
        values=predicted_values,
        log_variance=torch.full_like(predicted_values, -4.0),
        object_ids=torch.tensor([[12, 10]]),
        belief_indices=torch.tensor([[2, 0]]),
        valid_mask=torch.tensor([[True, True]]),
        visibility=torch.ones(1, 2),
    )
    measured = _measurements(torch.tensor([[[0.0, 0.0, -1.0, 0.5], [2.0, 0.0, -1.0, 0.4]]]))

    result = Associator().match(belief, measured, predicted)
    pairs = {
        (int(belief_index), int(measurement_index))
        for belief_index, measurement_index, valid in zip(
            result.belief_indices[0],
            result.measurement_indices[0],
            result.pair_mask[0],
            strict=True,
        )
        if bool(valid)
    }

    assert pairs == {(0, 0), (2, 1)}
    assert result.unmatched_beliefs.shape == belief.objects.active.shape
    assert not result.unmatched_beliefs.any()
    innovation = build_innovation(
        measured=measured,
        predicted=predicted,
        association=result,
        modality_index=0,
    )
    torch.testing.assert_close(
        innovation.residual[result.pair_mask],
        torch.zeros_like(innovation.residual[result.pair_mask]),
    )


def test_association_rejects_near_zero_existence_even_with_exact_geometry() -> None:
    belief, predicted = _belief_and_prediction()
    measured = _measurements(predicted.values[:, :1].clone())
    measured.existence_logits.fill_(-20.0)

    result = Associator().match(belief, measured, predicted)

    assert not result.pair_mask.any()
    assert result.unmatched_measurements[0, 0]
    assert result.unmatched_beliefs[0, :2].all()
