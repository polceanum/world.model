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
from world_model.observations.rgb.roi_updater import (
    FastROIUpdater,
    _sample_rois_native_bilinear,
    _uses_native_mps_gradient_sampler,
    make_roi_grid,
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
    assert measured.source_belief_indices is None
    assert measured.source_object_ids is None


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


def test_fast_roi_depth_residual_is_gated_until_explicitly_enabled() -> None:
    intrinsics, world_from_camera = _calibration()
    belief = BeliefFactory(max_objects=1, geometry_dim=1, appearance_dim=8).create(
        intrinsics=intrinsics.unsqueeze(0),
        world_from_camera=world_from_camera.unsqueeze(0),
    )
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[2]]),
            position=torch.tensor([[[0.0, 0.0, 3.0]]]),
        )
    )
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.05,
        payload=torch.zeros(1, 3, 32, 32),
        calibration={
            "intrinsics": intrinsics.unsqueeze(0),
            "world_from_camera": world_from_camera.unsqueeze(0),
        },
        frame_id="camera:camera",
    )

    def measure(enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
        module = RGBObservationModule(
            RGBObservationConfig(
                max_objects=1,
                backbone_channels=(8, 16, 24, 32),
                feature_dim=16,
                appearance_dim=8,
                roi_size=8,
                roi_hidden_dim=16,
                fast_depth_residual_enabled=enabled,
            )
        )
        with torch.no_grad():
            module.roi_updater.delta_head.bias[3] = 2.0
        predicted = module.project(
            belief,
            SensorContext(
                sensor_id=packet.sensor_id,
                timestamp=packet.timestamp,
                calibration=packet.calibration,
                frame_id=packet.frame_id,
                image_size=(32, 32),
            ),
        )
        measured, _ = module.encode_measurements(
            [packet],
            belief,
            predicted,
            None,
        )
        torch.testing.assert_close(
            measured.source_belief_indices,
            predicted.belief_indices,
        )
        torch.testing.assert_close(
            measured.source_object_ids,
            predicted.object_ids,
        )
        return predicted.values[..., 3], measured.values[..., 3]

    gated_prediction, gated_measurement = measure(False)
    enabled_prediction, enabled_measurement = measure(True)
    torch.testing.assert_close(gated_measurement, gated_prediction)
    assert not torch.allclose(enabled_measurement, enabled_prediction)


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


def test_native_bilinear_roi_sampler_matches_grid_sample_with_zero_padding() -> None:
    feature = torch.randn(2, 3, 7, 9, requires_grad=True)
    rois = torch.tensor(
        [
            [[-0.75, -0.5, 0.6, 0.8], [0.7, -1.2, 1.3, -0.4]],
            [[-1.4, 0.2, -0.5, 1.2], [-0.25, -0.8, 0.9, 0.7]],
        ],
        requires_grad=True,
    )
    grid = make_roi_grid(rois, output_size=5)
    expected = sample_rois(feature, rois, output_size=5)
    actual = _sample_rois_native_bilinear(feature, grid)

    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=1.0e-5)
    actual.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert rois.grad is not None and torch.isfinite(rois.grad).all()


def test_native_mps_roi_sampler_is_only_selected_when_gradients_are_enabled() -> None:
    assert _uses_native_mps_gradient_sampler(
        training=True,
        gradient_enabled=True,
        device_type="mps",
    )
    assert not _uses_native_mps_gradient_sampler(
        training=True,
        gradient_enabled=False,
        device_type="mps",
    )
    assert not _uses_native_mps_gradient_sampler(
        training=False,
        gradient_enabled=True,
        device_type="mps",
    )
    assert not _uses_native_mps_gradient_sampler(
        training=True,
        gradient_enabled=True,
        device_type="cpu",
    )


def test_training_mode_no_grad_roi_sampling_matches_inference_path() -> None:
    feature = torch.randn(2, 3, 7, 9)
    rois = torch.tensor(
        [
            [[-0.75, -0.5, 0.6, 0.8], [0.7, -1.2, 1.3, -0.4]],
            [[-1.4, 0.2, -0.5, 1.2], [-0.25, -0.8, 0.9, 0.7]],
        ]
    )

    with torch.no_grad():
        burn_in = sample_rois(feature, rois, output_size=5, training=True)
        inference = sample_rois(feature, rois, output_size=5, training=False)

    torch.testing.assert_close(burn_in, inference, atol=0.0, rtol=0.0)


def test_fast_roi_does_not_clip_a_valid_partially_offscreen_prior() -> None:
    updater = FastROIUpdater(
        feature_dim=8,
        appearance_dim=4,
        roi_size=8,
        hidden_dim=16,
    )
    predicted = torch.tensor([[[1.6, -1.5, -2.0, 0.25, 0.5, 0.5, 0.5]]])
    output = updater(
        torch.zeros(1, 8, 8, 8),
        torch.tensor([[[0.8, -1.0, 1.0, -0.7]]]),
        predicted,
        valid_mask=torch.tensor([[True]]),
    )

    torch.testing.assert_close(output.values[..., :2], predicted[..., :2])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_roi_sampling_mps_training_mode_no_grad_uses_inference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = torch.randn(1, 2, 8, 8, device="mps")
    rois = torch.tensor(
        [[[-0.75, -0.5, 0.6, 0.8]]],
        device="mps",
    )

    def unexpected_native_sampler(feature_map: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        raise AssertionError("no-grad ROI sampling must use the inference grid_sample path")

    monkeypatch.setattr(
        "world_model.observations.rgb.roi_updater._sample_rois_native_bilinear",
        unexpected_native_sampler,
    )
    with torch.no_grad():
        burn_in = sample_rois(feature, rois, output_size=5, training=True)
        inference = sample_rois(feature, rois, output_size=5, training=False)

    torch.testing.assert_close(burn_in, inference, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_roi_sampling_mps_training_native_bilinear_path_is_differentiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    native_sampler_called = False

    def recording_native_sampler(
        feature_map: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal native_sampler_called
        native_sampler_called = True
        return _sample_rois_native_bilinear(feature_map, grid)

    monkeypatch.setattr(
        "world_model.observations.rgb.roi_updater._sample_rois_native_bilinear",
        recording_native_sampler,
    )
    sampled = sample_rois(feature, rois, output_size=5, training=True)
    assert native_sampler_called
    assert sampled.device.type == "mps"
    sampled.square().mean().backward()
    assert feature.grad is not None
    assert feature.grad.device.type == "mps"
    assert torch.isfinite(feature.grad).all()
    assert rois.grad is not None
    assert rois.grad.device.type == "mps"
    assert torch.isfinite(rois.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_global_rgb_cpu_detector_trains_and_roundtrips_with_mps_backbone(
    tmp_path,
) -> None:
    torch.manual_seed(17)
    device = torch.device("mps")
    config = RGBObservationConfig(
        max_objects=3,
        birth_extra_queries=1,
        backbone_channels=(8, 16, 24, 32),
        feature_dim=16,
        appearance_dim=8,
        roi_size=8,
        roi_hidden_dim=16,
        global_detector_cpu_on_mps=True,
        structured_disc_center_enabled=False,
    )
    module = RGBObservationModule(config).to(device)
    module.train()

    assert {parameter.device.type for parameter in module.backbone.parameters()} == {"mps"}
    assert {parameter.device.type for parameter in module.roi_updater.parameters()} == {"mps"}
    assert {parameter.device.type for parameter in module.global_detector.parameters()} == {"cpu"}
    assert {buffer.device.type for buffer in module.global_detector.buffers()} == {"cpu"}

    axis = torch.arange(32, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    images = torch.zeros(2, 3, 32, 32)
    first_disc = (xx - 10.0).square() + (yy - 13.0).square() <= 5.0**2
    second_disc = (xx - 22.0).square() + (yy - 19.0).square() <= 4.0**2
    images[0, 0] = first_disc.to(images.dtype)
    images[0, 1] = 0.25 * first_disc.to(images.dtype)
    images[1, 1] = second_disc.to(images.dtype)
    images[1, 2] = 0.5 * second_disc.to(images.dtype)
    images = images.to(device)
    intrinsics, world_from_camera = _calibration()
    batched_intrinsics = intrinsics.unsqueeze(0).expand(2, -1, -1)
    batched_world_from_camera = world_from_camera.unsqueeze(0).expand(2, -1, -1)
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.0,
        payload=images,
        calibration={
            "intrinsics": batched_intrinsics,
            "world_from_camera": batched_world_from_camera,
        },
        frame_id="camera:camera",
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    measured = module.initialise_measurements(
        [packet],
        ObservationContext(
            timestamp=0.0,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            max_objects=3,
            device=device,
        ),
    )
    assert measured.values.device.type == "mps"
    assert measured.log_variance.device.type == "mps"
    assert measured.existence_logits.device.type == "mps"
    loss = (
        measured.values.square().mean()
        + 0.1 * measured.log_variance.square().mean()
        + 0.1 * measured.existence_logits.square().mean()
        + 0.1 * measured.auxiliary["visibility_logits"].square().mean()
        + 0.01 * measured.auxiliary["query_features"].square().mean()
    )
    assert loss.device.type == "mps"
    assert bool(torch.isfinite(loss))
    loss.backward()

    def finite_gradient_sum(parameters) -> float:
        parameters = tuple(parameters)
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients
        assert all(
            gradient.device == parameter.device
            for gradient, parameter in zip(
                gradients,
                (parameter for parameter in parameters if parameter.grad is not None),
                strict=True,
            )
        )
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        return sum(float(gradient.detach().abs().sum().cpu()) for gradient in gradients)

    assert finite_gradient_sum(module.backbone.parameters()) > 0.0
    assert finite_gradient_sum(module.global_detector.parameters()) > 0.0
    backbone_before = [
        parameter.detach().cpu().clone() for parameter in module.backbone.parameters()
    ]
    detector_before = [
        parameter.detach().cpu().clone() for parameter in module.global_detector.parameters()
    ]
    gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
    assert bool(torch.isfinite(gradient_norm.detach().cpu()))
    assert float(gradient_norm.detach().cpu()) > 0.0
    optimizer.step()

    assert all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())
    assert any(
        not torch.equal(before, parameter.detach().cpu())
        for before, parameter in zip(
            backbone_before,
            module.backbone.parameters(),
            strict=True,
        )
    )
    assert any(
        not torch.equal(before, parameter.detach().cpu())
        for before, parameter in zip(
            detector_before,
            module.global_detector.parameters(),
            strict=True,
        )
    )

    def assert_optimizer_state_ownership(
        candidate: torch.optim.Optimizer,
    ) -> None:
        moment_devices = set()
        for parameter, state in candidate.state.items():
            step = state["step"]
            assert isinstance(step, torch.Tensor)
            assert step.device.type == "cpu"
            assert bool(torch.isfinite(step))
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state[name]
                assert moment.device == parameter.device
                assert bool(torch.isfinite(moment).all())
                moment_devices.add(moment.device.type)
        assert moment_devices == {"cpu", "mps"}

    assert_optimizer_state_ownership(optimizer)

    checkpoint = tmp_path / "mixed-device-rgb.pt"
    torch.save(
        {
            "model_state": module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        checkpoint,
    )
    restored = RGBObservationModule(config).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-3)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["model_state"])
    restored_optimizer.load_state_dict(payload["optimizer_state"])

    assert {parameter.device.type for parameter in restored.backbone.parameters()} == {"mps"}
    assert {parameter.device.type for parameter in restored.roi_updater.parameters()} == {"mps"}
    assert {parameter.device.type for parameter in restored.global_detector.parameters()} == {"cpu"}
    assert_optimizer_state_ownership(restored_optimizer)

    restored_optimizer.zero_grad(set_to_none=True)
    restored_measurements = restored.initialise_measurements(
        [packet],
        ObservationContext(
            timestamp=0.0,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            max_objects=3,
            device=device,
        ),
    )
    restored_loss = (
        restored_measurements.values.square().mean()
        + 0.1 * restored_measurements.log_variance.square().mean()
        + 0.1 * restored_measurements.existence_logits.square().mean()
        + 0.1 * restored_measurements.auxiliary["visibility_logits"].square().mean()
        + 0.01 * restored_measurements.auxiliary["query_features"].square().mean()
    )
    assert bool(torch.isfinite(restored_loss))
    restored_loss.backward()
    restored_gradient_norm = torch.nn.utils.clip_grad_norm_(
        restored.parameters(),
        max_norm=1.0,
    )
    assert bool(torch.isfinite(restored_gradient_norm.detach().cpu()))
    assert float(restored_gradient_norm.detach().cpu()) > 0.0
    restored_optimizer.step()

    assert all(bool(torch.isfinite(parameter).all()) for parameter in restored.parameters())
    assert_optimizer_state_ownership(restored_optimizer)
