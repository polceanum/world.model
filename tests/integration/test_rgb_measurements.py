from __future__ import annotations

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.fusion import Associator
from world_model.observations import (
    MeasurementSet,
    ObservationContext,
    ObservationPacket,
    SensorContext,
)
from world_model.observations.rgb import (
    RGBObservationConfig,
    RGBObservationModule,
    backproject_rgb_log_variance,
    backproject_rgb_measurements,
    project_world_points,
    sample_rois,
)


def _calibration(size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [
            [30.0, 0.0, (size - 1) / 2.0],
            [0.0, 30.0, (size - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world_from_camera = torch.eye(4)
    return intrinsics, world_from_camera


def test_project_backproject_roundtrip_uses_explicit_camera_transform() -> None:
    intrinsics, world_from_camera = _calibration()
    position = torch.tensor([[[0.25, -0.15, 3.0]]])
    radius = torch.tensor([[[0.2]]])
    centre, radius_normalized, inverse_depth, _ = project_world_points(
        position,
        radius,
        world_from_camera.unsqueeze(0),
        intrinsics.unsqueeze(0),
        (32, 32),
    )
    values = torch.cat(
        (
            centre,
            radius_normalized.log().unsqueeze(-1),
            inverse_depth.unsqueeze(-1),
            torch.zeros(1, 1, 3),
        ),
        dim=-1,
    )
    reconstructed = backproject_rgb_measurements(
        values,
        world_from_camera.unsqueeze(0),
        intrinsics.unsqueeze(0),
        (32, 32),
    )
    assert torch.allclose(reconstructed, position, atol=1.0e-5)


def test_backprojection_propagates_finite_depth_dependent_xyz_variance() -> None:
    intrinsics, world_from_camera = _calibration()
    near = torch.tensor([[[0.2, -0.1, -2.0, 0.5, 0.0, 0.0, 0.0]]])
    far = near.clone()
    far[..., 3] = 0.25
    sensor_log_variance = torch.full_like(near, -6.0)
    near_world_lv = backproject_rgb_log_variance(
        near,
        sensor_log_variance,
        world_from_camera.unsqueeze(0),
        intrinsics.unsqueeze(0),
        (32, 32),
    )
    far_world_lv = backproject_rgb_log_variance(
        far,
        sensor_log_variance,
        world_from_camera.unsqueeze(0),
        intrinsics.unsqueeze(0),
        (32, 32),
    )
    noisier_world_lv = backproject_rgb_log_variance(
        near,
        sensor_log_variance + 2.0,
        world_from_camera.unsqueeze(0),
        intrinsics.unsqueeze(0),
        (32, 32),
    )
    assert near_world_lv.shape == (1, 1, 3)
    assert torch.isfinite(near_world_lv).all()
    assert torch.all(far_world_lv > near_world_lv)
    assert torch.all(noisier_world_lv > near_world_lv)


def test_global_rgb_measurements_are_finite_and_state_free() -> None:
    intrinsics, world_from_camera = _calibration()
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=3,
            birth_extra_queries=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
        )
    )
    image = torch.zeros(3, 32, 32)
    image[0, 10:18, 12:20] = 1.0
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.0,
        payload=image,
        calibration={
            "intrinsics": intrinsics,
            "world_from_camera": world_from_camera,
        },
        frame_id="camera:camera",
    )
    measured = module.initialise_measurements(
        [packet],
        ObservationContext(
            timestamp=0.0,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            max_objects=3,
            device=torch.device("cpu"),
        ),
    )
    assert measured.values.shape == (1, 4, 7)
    assert measured.auxiliary["world_position"].shape == (1, 4, 3)
    assert torch.isfinite(measured.values).all()
    assert "object_id" not in measured.auxiliary


def test_grid_sample_roi_path_and_projector_shapes() -> None:
    feature = torch.arange(16.0).reshape(1, 1, 4, 4)
    rois = torch.tensor([[[-1.0, -1.0, 1.0, 1.0]]])
    crop = sample_rois(feature, rois, output_size=3)
    assert crop.shape == (1, 1, 1, 3, 3)
    assert crop[0, 0, 0, 1, 1] == 7.5

    intrinsics, world_from_camera = _calibration()
    belief = BeliefFactory(max_objects=2, geometry_dim=1, appearance_dim=8).create(
        intrinsics=intrinsics.unsqueeze(0),
        world_from_camera=world_from_camera.unsqueeze(0),
    )
    objects = belief.objects.replace(
        active=torch.tensor([[True, False]]),
        object_id=torch.tensor([[2, -1]]),
        position=torch.tensor([[[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]]),
    )
    belief = belief.replace(objects=objects)
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=2,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
        )
    )
    predicted = module.project(
        belief,
        SensorContext(
            sensor_id="camera",
            timestamp=0.0,
            calibration={
                "intrinsics": intrinsics,
                "world_from_camera": world_from_camera,
            },
            frame_id="camera:camera",
            image_size=(32, 32),
        ),
    )
    assert predicted.values.shape == (1, 2, 7)
    assert predicted.rois is not None
    assert predicted.rois.shape == (1, 2, 4)


def test_offscreen_object_is_invalid_and_cannot_be_roi_associated() -> None:
    intrinsics, world_from_camera = _calibration()
    belief = BeliefFactory(max_objects=1, geometry_dim=1, appearance_dim=8).create()
    objects = belief.objects.replace(
        active=torch.tensor([[True]]),
        object_id=torch.tensor([[9]]),
        position=torch.tensor([[[100.0, 0.0, 3.0]]]),
    )
    belief = belief.replace(objects=objects)
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
        )
    )
    predicted = module.project(
        belief,
        SensorContext(
            sensor_id="camera",
            timestamp=0.0,
            calibration={
                "intrinsics": intrinsics,
                "world_from_camera": world_from_camera,
            },
            frame_id="camera:camera",
            image_size=(32, 32),
        ),
    )
    assert not predicted.valid_mask.any()
    assert not predicted.auxiliary["occluded_mask"].any()
    assert not predicted.auxiliary["occlusion_fraction"].any()
    assert predicted.rois is not None
    assert torch.equal(predicted.rois, torch.zeros_like(predicted.rois))

    measurements = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.zeros(1),
        values=predicted.values.clone(),
        log_variance=torch.zeros_like(predicted.values),
        existence_logits=torch.full((1, 1), 8.0),
        measurement_mask=torch.ones(1, 1, dtype=torch.bool),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera",
        supported_state_fields=("position",),
    )
    association = Associator().match(belief, measurements, predicted)
    assert not association.pair_mask.any()
    assert association.unmatched_measurements.all()


def test_roi_sampling_cpu_backward_reaches_features_and_coordinates() -> None:
    feature = torch.randn(1, 2, 8, 8, requires_grad=True)
    rois = torch.tensor(
        [[[-0.75, -0.5, 0.6, 0.8]]],
        requires_grad=True,
    )
    sample_rois(feature, rois, output_size=5, training=True).square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert rois.grad is not None and torch.isfinite(rois.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_roi_sampling_mps_training_cpu_fallback_is_differentiable() -> None:
    feature = torch.randn(
        1,
        2,
        8,
        8,
        device="mps",
        requires_grad=True,
    )
    rois = torch.tensor(
        [[[-0.75, -0.5, 0.6, 0.8]]],
        device="mps",
        requires_grad=True,
    )
    sampled = sample_rois(feature, rois, output_size=5, training=True)
    assert sampled.device.type == "mps"
    sampled.square().mean().backward()
    assert feature.grad is not None
    assert feature.grad.device.type == "mps"
    assert torch.isfinite(feature.grad).all()
    assert rois.grad is not None
    assert rois.grad.device.type == "mps"
    assert torch.isfinite(rois.grad).all()
