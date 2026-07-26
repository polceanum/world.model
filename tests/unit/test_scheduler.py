from __future__ import annotations

import math

import torch

from world_model.belief import BeliefFactory
from world_model.fusion import ObservationMode, ObservationScheduler
from world_model.observations import ObservationPacket, PredictedMeasurements


def _packet() -> ObservationPacket:
    return ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.0,
        payload=torch.zeros(3, 8, 8),
        calibration={},
        frame_id="camera:camera",
    )


def _belief_and_prediction(position_std: float) -> tuple[object, PredictedMeasurements]:
    belief = BeliefFactory(max_objects=2).create()
    objects = belief.objects.replace(
        active=torch.tensor([[True, False]]),
        object_id=torch.tensor([[7, -1]]),
        fast_log_variance=torch.full_like(
            belief.objects.fast_log_variance,
            math.log(position_std**2),
        ),
    )
    belief = belief.replace(objects=objects)
    predicted = PredictedMeasurements(
        modality="rgb",
        sensor_id="camera",
        timestamp=belief.timestamp,
        values=torch.zeros(1, 2, 4),
        log_variance=torch.zeros(1, 2, 4),
        object_ids=objects.object_id,
        belief_indices=torch.tensor([[0, 1]]),
        valid_mask=objects.active,
        visibility=objects.active.float(),
    )
    return belief, predicted


def test_scheduler_uncertainty_threshold_is_position_standard_deviation() -> None:
    scheduler = ObservationScheduler(
        uncertainty_threshold=4.0,
        surprise_threshold=8.0,
    )
    assert (
        scheduler.choose(packet=_packet(), belief=None, predicted=None)
        == ObservationMode.GLOBAL_DISCOVERY
    )

    belief, predicted = _belief_and_prediction(position_std=3.0)
    scheduler.record("camera", ObservationMode.GLOBAL_DISCOVERY)
    assert (
        scheduler.choose(packet=_packet(), belief=belief, predicted=predicted)
        == ObservationMode.FAST_ROI
    )

    belief, predicted = _belief_and_prediction(position_std=5.0)
    assert (
        scheduler.choose(packet=_packet(), belief=belief, predicted=predicted)
        == ObservationMode.RECOVERY
    )
