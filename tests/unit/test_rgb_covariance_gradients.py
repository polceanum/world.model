from __future__ import annotations

import torch

from world_model.observations import ObservationContext, ObservationPacket
from world_model.observations.rgb import RGBObservationConfig, RGBObservationModule


def test_world_covariance_calibration_does_not_backpropagate_through_measurement_mean() -> None:
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=2,
            birth_extra_queries=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
            structured_disc_center_enabled=False,
        )
    )
    intrinsics = torch.tensor(
        [
            [30.0, 0.0, 15.5],
            [0.0, 30.0, 15.5],
            [0.0, 0.0, 1.0],
        ]
    )
    world_from_camera = torch.eye(4)
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.0,
        payload=torch.rand(1, 3, 32, 32),
        calibration={
            "intrinsics": intrinsics.unsqueeze(0),
            "world_from_camera": world_from_camera.unsqueeze(0),
        },
        frame_id="camera:camera",
    )
    measurements = module.initialise_measurements(
        [packet],
        ObservationContext(
            timestamp=0.0,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            max_objects=2,
            device=torch.device("cpu"),
        ),
    )
    measurements.values.retain_grad()

    measurements.auxiliary["world_position_log_variance"].mean().backward()

    assert measurements.values.grad is None
    for head in (
        module.global_detector.centre_head,
        module.global_detector.radius_head,
        module.global_detector.depth_head,
    ):
        assert all(parameter.grad is None for parameter in head.parameters())
    variance_gradients = [
        parameter.grad for parameter in module.global_detector.variance_head.parameters()
    ]
    assert all(gradient is not None for gradient in variance_gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in variance_gradients if gradient is not None
    )
    assert any(
        torch.count_nonzero(gradient) > 0 for gradient in variance_gradients if gradient is not None
    )
