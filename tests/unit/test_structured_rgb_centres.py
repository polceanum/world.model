from __future__ import annotations

import math

import pytest
import torch

from world_model.belief import BeliefFactory, ObjectLifecycle
from world_model.fusion import AssociationResult
from world_model.observations import (
    MeasurementSet,
    ObservationContext,
    ObservationPacket,
    PredictedMeasurements,
    SensorContext,
)
from world_model.observations.rgb import (
    GlobalDetectorOutput,
    GlobalObjectDetector,
    RGBObservationConfig,
    RGBObservationModule,
)
from world_model.observations.rgb import module as rgb_module
from world_model.observations.rgb.structured_centres import (
    structured_disc_centres,
    structured_disc_centres_in_rois,
)


def _normalized_pixel(
    *,
    x: float,
    y: float,
    width: int,
    height: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [
            2.0 * x / max(width - 1, 1) - 1.0,
            2.0 * y / max(height - 1, 1) - 1.0,
        ],
        dtype=dtype,
    )


def _paint_block(
    image: torch.Tensor,
    *,
    batch_index: int,
    top: int,
    left: int,
    height: int,
    width: int,
    colour: tuple[float, float, float],
) -> torch.Tensor:
    image[
        batch_index,
        :,
        top : top + height,
        left : left + width,
    ] = torch.tensor(colour, dtype=image.dtype).reshape(3, 1, 1)
    return _normalized_pixel(
        x=left + (width - 1) / 2.0,
        y=top + (height - 1) / 2.0,
        width=image.shape[-1],
        height=image.shape[-2],
        dtype=image.dtype,
    )


def _global_structured_measurement(
    monkeypatch: pytest.MonkeyPatch,
    *,
    image: torch.Tensor,
    proposal_centres: torch.Tensor,
    learned_existence_logit: float = 8.0,
    structured_confidence: float = 0.995,
    packet_confidence: float = 1.0,
    existence_requires_grad: bool = False,
) -> tuple[MeasurementSet, GlobalDetectorOutput]:
    query_count = proposal_centres.shape[1]
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=query_count,
            birth_extra_queries=0,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            structured_disc_center_enabled=True,
            structured_disc_depth_relative_std=0.05,
            structured_disc_position_confidence=structured_confidence,
        )
    )
    learned_log_radius = image.new_full((1, query_count, 1), math.log(0.5))
    learned_depth_residual = image.new_full((1, query_count, 1), 0.2)
    learned_log_variance = (
        image.new_tensor((-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0))
        .view(1, 1, 7)
        .expand(1, query_count, 7)
    )
    existence_logits = image.new_full((1, query_count), learned_existence_logit)
    existence_logits.requires_grad_(existence_requires_grad)
    detector_output = GlobalDetectorOutput(
        centre=proposal_centres,
        log_radius=learned_log_radius,
        inverse_depth_residual=learned_depth_residual,
        colour=image.new_full((1, query_count, 3), 0.5),
        existence_logits=existence_logits,
        visibility_logits=image.new_full((1, query_count), 8.0),
        log_variance=learned_log_variance,
        appearance=image.new_zeros((1, query_count, 8)),
        query_features=image.new_zeros((1, query_count, 16)),
        attention=image.new_zeros((1, query_count, 1)),
    )
    monkeypatch.setattr(
        module.global_detector,
        "forward",
        lambda feature_map: detector_output,
    )
    height, width = image.shape[-2:]
    intrinsics = image.new_tensor(
        (
            (30.0, 0.0, (width - 1) / 2.0),
            (0.0, 30.0, (height - 1) / 2.0),
            (0.0, 0.0, 1.0),
        )
    )
    world_from_camera = torch.eye(4, dtype=image.dtype, device=image.device)
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.1,
        payload=image,
        calibration={
            "intrinsics": intrinsics.unsqueeze(0),
            "world_from_camera": world_from_camera.unsqueeze(0),
        },
        frame_id="camera:camera",
        confidence=packet_confidence,
    )
    measurement = module.initialise_measurements(
        [packet],
        ObservationContext(
            timestamp=packet.timestamp,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            max_objects=query_count,
            device=image.device,
        ),
    )
    return measurement, detector_output


def test_localizes_exact_foreground_component_centres_from_rgb_pixels() -> None:
    image = torch.zeros((1, 3, 9, 13), dtype=torch.float32)
    left_centre = _paint_block(
        image,
        batch_index=0,
        top=2,
        left=1,
        height=3,
        width=3,
        colour=(1.0, 0.25, 0.1),
    )
    right_centre = _paint_block(
        image,
        batch_index=0,
        top=5,
        left=9,
        height=3,
        width=3,
        colour=(0.1, 0.4, 1.0),
    )
    expected = torch.stack((left_centre, right_centre)).to(torch.float64)
    proposals = (expected + torch.tensor([[0.03, -0.02], [-0.04, 0.01]])).unsqueeze(0)

    result = structured_disc_centres(image, proposals)

    assert result.component_count.tolist() == [2]
    assert result.valid_mask.tolist() == [[True, True]]
    assert result.depth_valid_mask.tolist() == [[True, True]]
    assert result.centres.dtype == proposals.dtype
    assert result.centres.device == proposals.device
    torch.testing.assert_close(
        result.radius_pixels[0],
        torch.full((2,), math.sqrt(9.0 / math.pi), dtype=proposals.dtype),
    )
    torch.testing.assert_close(result.centres[0], expected, rtol=0.0, atol=1.0e-6)


def test_hungarian_alignment_preserves_proposal_order_and_unmatched_slot() -> None:
    image = torch.zeros((1, 3, 9, 13), dtype=torch.float32)
    left_centre = _paint_block(
        image,
        batch_index=0,
        top=2,
        left=1,
        height=3,
        width=3,
        colour=(0.9, 0.2, 0.2),
    )
    right_centre = _paint_block(
        image,
        batch_index=0,
        top=5,
        left=9,
        height=3,
        width=3,
        colour=(0.2, 0.9, 0.2),
    )
    unmatched_proposal = torch.tensor([0.0, -0.95])
    proposals = torch.stack(
        (
            right_centre + torch.tensor([0.04, -0.03]),
            left_centre + torch.tensor([-0.03, 0.02]),
            unmatched_proposal,
        )
    ).unsqueeze(0)

    result = structured_disc_centres(
        image,
        proposals,
        maximum_assignment_distance=0.25,
    )

    assert result.valid_mask.tolist() == [[True, True, False]]
    torch.testing.assert_close(result.centres[0, 0], right_centre, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(result.centres[0, 1], left_centre, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(result.centres[0, 2], unmatched_proposal)


def test_splits_touching_disc_silhouettes_at_distance_peaks() -> None:
    height = width = 15
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    centres_pixels = ((5, 7), (9, 7))
    colours = ((0.9, 0.2, 0.1), (0.1, 0.5, 0.9))
    for (centre_x, centre_y), colour in zip(centres_pixels, colours, strict=True):
        disc = (pixel_x - centre_x).square() + (pixel_y - centre_y).square() <= 2**2
        image[0, :, disc] = torch.tensor(colour).unsqueeze(-1)
    expected = torch.stack(
        [
            _normalized_pixel(
                x=centre_x,
                y=centre_y,
                width=width,
                height=height,
            )
            for centre_x, centre_y in centres_pixels
        ]
    )
    proposals = (expected + torch.tensor([[0.02, 0.0], [-0.02, 0.0]])).unsqueeze(0)

    result = structured_disc_centres(image, proposals)

    assert result.component_count.tolist() == [2]
    assert result.valid_mask.tolist() == [[True, True]]
    assert result.ambiguous_mask.tolist() == [[True, True]]
    assert result.depth_valid_mask.tolist() == [[False, False]]
    torch.testing.assert_close(result.centres[0], expected, rtol=0.0, atol=0.055)


def test_global_component_touching_image_boundary_keeps_centre_but_rejects_scale() -> None:
    image = torch.zeros((1, 3, 11, 11), dtype=torch.float32)
    centre = _paint_block(
        image,
        batch_index=0,
        top=3,
        left=0,
        height=5,
        width=4,
        colour=(0.9, 0.2, 0.1),
    )
    result = structured_disc_centres(image, centre.reshape(1, 1, 2))

    assert result.valid_mask.tolist() == [[True]]
    assert result.ambiguous_mask.tolist() == [[False]]
    assert result.depth_valid_mask.tolist() == [[False]]
    torch.testing.assert_close(result.centres[0, 0], centre)


def test_global_rgb_touching_components_use_observed_scale_with_inflated_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height = width = 15
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    centres_pixels = ((5, 7), (9, 7))
    for (centre_x, centre_y), colour in zip(
        centres_pixels,
        ((0.9, 0.2, 0.1), (0.1, 0.5, 0.9)),
        strict=True,
    ):
        disc = (pixel_x - centre_x).square() + (pixel_y - centre_y).square() <= 2**2
        image[0, :, disc] = torch.tensor(colour).unsqueeze(-1)
    proposals = torch.stack(
        [
            _normalized_pixel(
                x=centre_x,
                y=centre_y,
                width=width,
                height=height,
            )
            for centre_x, centre_y in centres_pixels
        ]
    ).unsqueeze(0)
    structured = structured_disc_centres(image, proposals)
    measurement, detector = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposals,
    )

    assert structured.valid_mask.tolist() == [[True, True]]
    assert structured.depth_valid_mask.tolist() == [[False, False]]
    assert measurement.auxiliary["structured_centre_valid"].tolist() == [[True, True]]
    assert measurement.auxiliary["structured_depth_valid"].tolist() == [[False, False]]
    torch.testing.assert_close(measurement.values[..., :2], structured.centres)
    observed_log_radius = (
        (structured.radius_pixels / (0.5 * min(height, width))).log().unsqueeze(-1)
    )
    torch.testing.assert_close(measurement.values[..., 2:3], observed_log_radius)
    expected_inverse_depth = structured.radius_pixels / (30.0 * 0.15)
    torch.testing.assert_close(measurement.values[..., 3], expected_inverse_depth)
    relative_variance = 0.05**2 * 9.0
    expected_log_variance = torch.stack(
        (
            torch.full_like(expected_inverse_depth, math.log(relative_variance)),
            (expected_inverse_depth.square() * relative_variance).log(),
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        measurement.log_variance[..., 2:4],
        expected_log_variance,
    )
    assert not torch.allclose(measurement.values[..., 2:3], detector.log_radius)


def test_global_rgb_truncated_component_uses_observed_scale_with_inflated_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height = width = 21
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    proposal = _paint_block(
        image,
        batch_index=0,
        top=7,
        left=0,
        height=7,
        width=5,
        colour=(0.9, 0.2, 0.1),
    ).reshape(1, 1, 2)
    structured = structured_disc_centres(image, proposal)
    measurement, detector = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposal + torch.tensor([[[0.02, -0.01]]]),
    )

    assert structured.valid_mask.tolist() == [[True]]
    assert structured.depth_valid_mask.tolist() == [[False]]
    assert measurement.auxiliary["structured_centre_valid"].tolist() == [[True]]
    assert measurement.auxiliary["structured_depth_valid"].tolist() == [[False]]
    torch.testing.assert_close(measurement.values[0, 0, :2], structured.centres[0, 0])
    observed_log_radius = (
        (structured.radius_pixels / (0.5 * min(height, width))).log().unsqueeze(-1)
    )
    torch.testing.assert_close(measurement.values[..., 2:3], observed_log_radius)
    expected_inverse_depth = structured.radius_pixels / (30.0 * 0.15)
    torch.testing.assert_close(measurement.values[..., 3], expected_inverse_depth)
    relative_variance = 0.05**2 * 9.0
    expected_log_variance = torch.stack(
        (
            torch.full_like(expected_inverse_depth, math.log(relative_variance)),
            (expected_inverse_depth.square() * relative_variance).log(),
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        measurement.log_variance[..., 2:4],
        expected_log_variance,
    )
    assert not torch.allclose(measurement.values[..., 2:3], detector.log_radius)


def test_global_rgb_isolated_component_uses_observed_depth_scale_and_tight_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height = width = 21
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    proposal = _paint_block(
        image,
        batch_index=0,
        top=8,
        left=8,
        height=5,
        width=5,
        colour=(0.9, 0.2, 0.1),
    ).reshape(1, 1, 2)
    structured = structured_disc_centres(image, proposal)
    measurement, detector = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposal,
    )

    assert structured.valid_mask.tolist() == [[True]]
    assert structured.depth_valid_mask.tolist() == [[True]]
    assert measurement.auxiliary["structured_centre_valid"].tolist() == [[True]]
    assert measurement.auxiliary["structured_depth_valid"].tolist() == [[True]]
    observed_log_radius = (
        (structured.radius_pixels / (0.5 * min(height, width))).log().unsqueeze(-1)
    )
    torch.testing.assert_close(measurement.values[..., 2:3], observed_log_radius)
    expected_inverse_depth = structured.radius_pixels / (30.0 * 0.15)
    torch.testing.assert_close(measurement.values[..., 3], expected_inverse_depth)
    relative_variance = 0.05**2
    expected_log_variance = torch.stack(
        (
            torch.full_like(expected_inverse_depth, math.log(relative_variance)),
            (expected_inverse_depth.square() * relative_variance).log(),
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        measurement.log_variance[..., 2:4],
        expected_log_variance,
    )
    assert not torch.allclose(measurement.values[..., 2:3], detector.log_radius)


def test_global_rgb_structured_confidence_fail_closes_unsupported_queries_with_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height = width = 21
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    observed_centre = _paint_block(
        image,
        batch_index=0,
        top=8,
        left=8,
        height=5,
        width=5,
        colour=(0.9, 0.2, 0.1),
    )
    proposals = torch.stack((observed_centre, torch.tensor([-0.9, -0.9]))).unsqueeze(0)
    measurement, detector = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposals,
        learned_existence_logit=-2.5,
        structured_confidence=0.97,
        existence_requires_grad=True,
    )

    assert measurement.auxiliary["structured_centre_valid"].tolist() == [[True, False]]
    assert measurement.auxiliary["structured_learned_fallback"].tolist() == [[False, False]]
    assert measurement.auxiliary["structured_runtime_supported"].tolist() == [[True, False]]
    torch.testing.assert_close(
        measurement.existence_logits.sigmoid(),
        torch.tensor([[0.97, 1.0e-4]]),
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        measurement.auxiliary["position_confidence"],
        torch.tensor([[0.97, 1.0e-4]]),
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    measurement.existence_logits.sum().backward()
    assert detector.existence_logits.grad is not None
    torch.testing.assert_close(
        detector.existence_logits.grad,
        torch.ones_like(detector.existence_logits),
    )


def test_global_rgb_structured_confidence_is_capped_by_packet_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.zeros((1, 3, 21, 21), dtype=torch.float32)
    observed_centre = _paint_block(
        image,
        batch_index=0,
        top=8,
        left=8,
        height=5,
        width=5,
        colour=(0.9, 0.2, 0.1),
    )
    measurement, _ = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=observed_centre.reshape(1, 1, 2),
        structured_confidence=0.995,
        packet_confidence=0.01,
    )

    torch.testing.assert_close(
        measurement.existence_logits.sigmoid(),
        torch.tensor([[0.00995]]),
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        measurement.auxiliary["position_confidence"],
        torch.tensor([[0.00995]]),
        rtol=1.0e-5,
        atol=1.0e-7,
    )


def test_global_rgb_no_component_retains_learned_discovery_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.zeros((1, 3, 21, 21), dtype=torch.float32)
    proposals = torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]])
    measurement, detector = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposals,
        learned_existence_logit=1.25,
        existence_requires_grad=True,
    )

    assert measurement.auxiliary["structured_component_count"].tolist() == [0]
    assert not measurement.auxiliary["structured_centre_valid"].any()
    assert measurement.auxiliary["structured_learned_fallback"].tolist() == [[True, True]]
    assert measurement.auxiliary["structured_runtime_supported"].tolist() == [[True, True]]
    torch.testing.assert_close(
        measurement.existence_logits,
        torch.full((1, 2), 1.25),
    )
    measurement.existence_logits.sum().backward()
    assert detector.existence_logits.grad is not None
    torch.testing.assert_close(
        detector.existence_logits.grad,
        torch.ones_like(detector.existence_logits),
    )


def test_global_rgb_incomplete_component_assignment_bounds_learned_fallback_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.zeros((1, 3, 25, 25), dtype=torch.float32)
    assigned = _paint_block(
        image,
        batch_index=0,
        top=10,
        left=4,
        height=5,
        width=5,
        colour=(0.9, 0.2, 0.1),
    )
    _paint_block(
        image,
        batch_index=0,
        top=10,
        left=17,
        height=5,
        width=5,
        colour=(0.1, 0.5, 0.9),
    )
    proposals = torch.stack(
        (assigned, torch.tensor([-0.9, -0.9]), torch.tensor([-0.9, 0.9]))
    ).unsqueeze(0)
    measurement, _ = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposals,
        learned_existence_logit=0.25,
        structured_confidence=0.97,
    )

    assert measurement.auxiliary["structured_component_count"].tolist() == [2]
    assert measurement.auxiliary["structured_centre_valid"].tolist() == [[True, False, False]]
    assert measurement.auxiliary["structured_learned_fallback"].tolist() == [[False, True, False]]
    torch.testing.assert_close(
        measurement.existence_logits.sigmoid(),
        torch.tensor([[0.97, torch.sigmoid(torch.tensor(0.25)), 1.0e-4]]),
        rtol=1.0e-5,
        atol=1.0e-7,
    )


def test_global_rgb_touching_components_recover_directly_without_duplicate_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height = width = 15
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    centres_pixels = ((5, 7), (9, 7))
    for (centre_x, centre_y), colour in zip(
        centres_pixels,
        ((0.9, 0.2, 0.1), (0.1, 0.5, 0.9)),
        strict=True,
    ):
        disc = (pixel_x - centre_x).square() + (pixel_y - centre_y).square() <= 2**2
        image[0, :, disc] = torch.tensor(colour).unsqueeze(-1)
    proposals = torch.stack(
        (
            _normalized_pixel(x=5, y=7, width=width, height=height),
            _normalized_pixel(x=9, y=7, width=width, height=height),
            torch.tensor([-0.9, -0.9]),
        )
    ).unsqueeze(0)
    measurement, _ = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposals,
        learned_existence_logit=-0.75,
        structured_confidence=0.97,
    )

    assert measurement.auxiliary["structured_centre_valid"].tolist() == [[True, True, False]]
    assert measurement.auxiliary["structured_centre_ambiguous"].tolist() == [[True, True, False]]
    assert measurement.auxiliary["structured_learned_fallback"].tolist() == [[False, False, False]]
    torch.testing.assert_close(
        measurement.existence_logits.sigmoid(),
        torch.tensor([[0.97, 0.97, 1.0e-4]]),
        rtol=1.0e-5,
        atol=1.0e-7,
    )


def test_global_rgb_isolated_components_birth_no_unsupported_ghost_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.zeros((1, 3, 25, 25), dtype=torch.float32)
    left = _paint_block(
        image,
        batch_index=0,
        top=8,
        left=4,
        height=5,
        width=5,
        colour=(0.9, 0.2, 0.1),
    )
    right = _paint_block(
        image,
        batch_index=0,
        top=13,
        left=16,
        height=5,
        width=5,
        colour=(0.1, 0.5, 0.9),
    )
    proposals = torch.stack(
        (left, right, torch.tensor([-0.9, 0.9]), torch.tensor([0.9, -0.9]))
    ).unsqueeze(0)
    measurement, _ = _global_structured_measurement(
        monkeypatch,
        image=image,
        proposal_centres=proposals,
        learned_existence_logit=8.0,
        structured_confidence=0.995,
    )

    assert measurement.auxiliary["structured_centre_valid"].tolist() == [[True, True, False, False]]
    torch.testing.assert_close(
        measurement.existence_logits.sigmoid(),
        torch.tensor([[0.995, 0.995, 1.0e-4, 1.0e-4]]),
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    belief = BeliefFactory(max_objects=4, geometry_dim=1, appearance_dim=8).create()
    born = ObjectLifecycle().birth_from_measurements(
        belief,
        measurement,
        torch.ones_like(measurement.measurement_mask),
        confidence_threshold=0.55,
    )

    assert int(born.objects.active.sum()) == 2
    assert born.objects.object_id[0].tolist() == [0, 1, -1, -1]


def test_fresh_global_detector_existence_prior_is_not_birth_worthy() -> None:
    detector = GlobalObjectDetector(
        feature_dim=16,
        query_count=4,
        appearance_dim=8,
        attention_heads=4,
        attention_layers=1,
    )

    torch.testing.assert_close(
        detector.existence_head.bias,
        torch.full_like(detector.existence_head.bias, -2.0),
    )
    assert torch.sigmoid(detector.existence_head.bias).item() < 0.5


def test_rejects_bright_speckle_noise_below_minimum_component_size() -> None:
    image = torch.zeros((1, 3, 11, 15), dtype=torch.float32)
    object_centre = _paint_block(
        image,
        batch_index=0,
        top=4,
        left=7,
        height=2,
        width=2,
        colour=(0.8, 0.3, 0.1),
    )
    speckle_pixels = ((1, 1), (1, 13), (9, 1))
    for y, x in speckle_pixels:
        image[0, :, y, x] = torch.tensor([1.0, 0.2, 0.8])

    speckle_centres = [
        _normalized_pixel(x=x, y=y, width=15, height=11) for y, x in speckle_pixels[:2]
    ]
    proposals = torch.stack(
        (
            object_centre + torch.tensor([0.01, -0.01]),
            speckle_centres[0],
            speckle_centres[1],
        )
    ).unsqueeze(0)

    result = structured_disc_centres(image, proposals, minimum_pixels=4)

    assert result.component_count.tolist() == [1]
    assert result.valid_mask.tolist() == [[True, False, False]]
    torch.testing.assert_close(result.centres[0, 0], object_centre, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(result.centres[0, 1:], proposals[0, 1:])


def test_assignment_gate_leaves_distant_proposal_unmodified() -> None:
    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    _paint_block(
        image,
        batch_index=0,
        top=3,
        left=5,
        height=3,
        width=3,
        colour=(0.2, 0.5, 1.0),
    )
    proposals = torch.tensor([[[-0.75, 0.0]]])

    result = structured_disc_centres(
        image,
        proposals,
        maximum_assignment_distance=0.25,
    )

    assert result.component_count.tolist() == [1]
    assert not result.valid_mask.any()
    torch.testing.assert_close(result.centres, proposals)


def test_rejects_invalid_image_and_proposal_shapes() -> None:
    valid_image = torch.zeros((1, 3, 8, 8))
    valid_proposals = torch.zeros((1, 2, 2))

    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        structured_disc_centres(valid_image[0], valid_proposals)
    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        structured_disc_centres(torch.zeros((1, 1, 8, 8)), valid_proposals)
    with pytest.raises(ValueError, match=r"\[B,Q,2\]"):
        structured_disc_centres(valid_image, valid_proposals[0])
    with pytest.raises(ValueError, match=r"\[B,Q,2\]"):
        structured_disc_centres(valid_image, torch.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="batch dimensions"):
        structured_disc_centres(valid_image, torch.zeros((2, 2, 2)))


def test_roi_refinement_localizes_only_the_projected_rgb_crop() -> None:
    height = width = 21
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    target = _paint_block(
        image,
        batch_index=0,
        top=8,
        left=12,
        height=5,
        width=5,
        colour=(0.9, 0.25, 0.1),
    )
    # A second object elsewhere in the frame must not participate in this
    # fast-path refinement.
    _paint_block(
        image,
        batch_index=0,
        top=2,
        left=2,
        height=4,
        width=4,
        colour=(0.1, 0.4, 0.9),
    )
    proposal = (target + torch.tensor([-0.08, 0.04], dtype=torch.float64)).reshape(
        1,
        1,
        2,
    )
    roi = torch.tensor(
        [[[*_normalized_pixel(x=9, y=5, width=width, height=height), 0.8, 0.5]]],
        dtype=torch.float64,
    )

    result = structured_disc_centres_in_rois(
        image,
        proposal,
        roi,
        output_size=25,
    )

    assert result.valid_mask.tolist() == [[True]]
    assert result.depth_valid_mask.tolist() == [[True]]
    assert result.component_pixel_count.item() >= 4
    assert 1.5 < result.radius_pixels.item() < 5.0
    assert result.centres.dtype == proposal.dtype
    assert result.centres.device == proposal.device
    torch.testing.assert_close(
        result.centres[0, 0],
        target.to(torch.float64),
        rtol=0.0,
        atol=0.025,
    )


def test_roi_refinement_rejects_noise_and_leaves_unmatched_slots_unchanged() -> None:
    height = width = 25
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    target = _paint_block(
        image,
        batch_index=0,
        top=10,
        left=15,
        height=4,
        width=4,
        colour=(0.9, 0.2, 0.1),
    )
    # The isolated speckle is closer to the prior than the real component, but
    # it must not become a supported foreground seed.
    image[0, :, 11, 13] = torch.tensor([1.0, 0.1, 0.8])
    proposals = torch.stack(
        (
            _normalized_pixel(x=13, y=11, width=width, height=height),
            torch.tensor([-0.75, -0.75]),
            target,
        )
    ).unsqueeze(0)
    rois = torch.tensor(
        [
            [
                [0.0, -0.35, 0.75, 0.35],
                [-1.0, -1.0, -0.5, -0.5],
                [0.0, -0.35, 0.75, 0.35],
            ]
        ]
    )

    result = structured_disc_centres_in_rois(
        image,
        proposals,
        rois,
        valid_mask=torch.tensor([[True, True, False]]),
        output_size=24,
        minimum_pixels=4,
    )

    assert result.valid_mask.tolist() == [[True, False, False]]
    torch.testing.assert_close(result.centres[0, 0], target, rtol=0.0, atol=0.035)
    torch.testing.assert_close(result.centres[0, 1:], proposals[0, 1:])


def test_roi_refinement_applies_distance_and_degenerate_roi_gates() -> None:
    image = torch.zeros((1, 3, 17, 17))
    target = _paint_block(
        image,
        batch_index=0,
        top=7,
        left=11,
        height=3,
        width=3,
        colour=(0.2, 0.7, 1.0),
    )
    proposals = torch.stack((torch.tensor([-0.5, 0.0]), target)).unsqueeze(0)
    rois = torch.tensor([[[0.0, -0.5, 1.0, 0.5], [0.0, 0.0, 0.0, 0.0]]])

    result = structured_disc_centres_in_rois(
        image,
        proposals,
        rois,
        maximum_assignment_distance=0.2,
    )

    assert result.valid_mask.tolist() == [[False, False]]
    torch.testing.assert_close(result.centres, proposals)


def test_roi_refinement_rejects_near_tied_disconnected_component_ownership() -> None:
    height = width = 25
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    _paint_block(
        image,
        batch_index=0,
        top=10,
        left=5,
        height=5,
        width=5,
        colour=(0.9, 0.2, 0.1),
    )
    _paint_block(
        image,
        batch_index=0,
        top=10,
        left=15,
        height=5,
        width=5,
        colour=(0.1, 0.4, 0.9),
    )
    rois = torch.tensor([[[-1.0, -1.0, 1.0, 1.0]]])

    for offset in (-1.0e-7, 1.0e-7):
        proposals = torch.tensor([[[offset, 0.0]]])
        result = structured_disc_centres_in_rois(
            image,
            proposals,
            rois,
            output_size=25,
        )

        assert result.ambiguous_mask.tolist() == [[True]]
        assert result.valid_mask.tolist() == [[False]]
        assert result.depth_valid_mask.tolist() == [[False]]
        torch.testing.assert_close(result.centres, proposals)


def test_roi_refinement_rejects_invalid_shapes() -> None:
    image = torch.zeros((1, 3, 16, 16))
    proposals = torch.zeros((1, 2, 2))
    rois = torch.zeros((1, 2, 4))

    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        structured_disc_centres_in_rois(image[0], proposals, rois)
    with pytest.raises(ValueError, match=r"\[B,N,2\]"):
        structured_disc_centres_in_rois(image, proposals[..., :1], rois)
    with pytest.raises(ValueError, match=r"\[B,N,4\]"):
        structured_disc_centres_in_rois(image, proposals, rois[..., :3])
    with pytest.raises(ValueError, match="batch/object dimensions"):
        structured_disc_centres_in_rois(image, proposals, torch.zeros((1, 3, 4)))
    with pytest.raises(ValueError, match=r"\[B,N\]"):
        structured_disc_centres_in_rois(
            image,
            proposals,
            rois,
            valid_mask=torch.ones((1, 1), dtype=torch.bool),
        )


def test_rgb_module_keeps_global_and_fast_raw_centres_differentiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 32
    intrinsics = torch.tensor(
        [
            [30.0, 0.0, (size - 1) / 2.0],
            [0.0, 30.0, (size - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world_from_camera = torch.eye(4)
    image = torch.zeros((1, 3, size, size))
    image[:, :, 14:18, 14:18] = torch.tensor([0.9, 0.2, 0.1]).reshape(1, 3, 1, 1)
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.1,
        payload=image,
        calibration={
            "intrinsics": intrinsics.unsqueeze(0),
            "world_from_camera": world_from_camera.unsqueeze(0),
        },
        frame_id="camera:camera",
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            birth_extra_queries=0,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
            structured_disc_center_enabled=True,
            structured_disc_fast_depth_enabled=True,
            structured_disc_depth_relative_std=0.05,
            structured_disc_position_confidence=0.9975,
        )
    )

    global_measurement = module.initialise_measurements(
        [packet],
        ObservationContext(
            timestamp=packet.timestamp,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            max_objects=1,
            device=torch.device("cpu"),
        ),
    )
    global_raw = global_measurement.auxiliary["raw_centre"]
    torch.testing.assert_close(
        global_measurement.auxiliary["position_confidence"],
        torch.tensor([[0.9975]]),
    )
    assert global_measurement.auxiliary["world_position_independent_axis_mask"].tolist() == [
        [[True, True, True]]
    ]
    assert global_raw.requires_grad
    global_raw.sum().backward()
    assert module.global_detector.centre_head.weight.grad is not None

    module.zero_grad(set_to_none=True)
    belief = BeliefFactory(max_objects=1, geometry_dim=1, appearance_dim=8).create(
        intrinsics=intrinsics.unsqueeze(0),
        world_from_camera=world_from_camera.unsqueeze(0),
    )
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[7]]),
            position=torch.tensor([[[0.0, 0.0, 3.0]]]),
            geometry=torch.tensor([[[0.3]]]),
            fast_log_variance=torch.full_like(
                belief.objects.fast_log_variance,
                -8.0,
            ),
        )
    )
    predicted = module.project(
        belief,
        SensorContext(
            sensor_id=packet.sensor_id,
            timestamp=packet.timestamp,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            image_size=(size, size),
        ),
    )

    def fail_global_scan(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("FAST_ROI must not invoke full-frame structured discovery")

    monkeypatch.setattr(rgb_module, "structured_disc_centres", fail_global_scan)
    fast_measurement, _ = module.encode_measurements(
        [packet],
        belief,
        predicted,
        None,
    )
    fast_raw = fast_measurement.auxiliary["raw_centre"]
    assert fast_measurement.auxiliary["structured_centre_valid"].tolist() == [[True]]
    assert fast_measurement.auxiliary["structured_depth_valid"].tolist() == [[True]]
    assert fast_measurement.auxiliary["position_independent_camera_axis_mask"].tolist() == [
        [[True, True, True]]
    ]
    assert fast_measurement.auxiliary["world_position_independent_axis_mask"].tolist() == [
        [[True, True, True]]
    ]
    assert torch.isfinite(fast_measurement.auxiliary["world_position"]).all()
    assert fast_raw.requires_grad
    fast_raw.sum().backward()
    assert module.roi_updater.delta_head.bias.grad is not None


def test_zero_residual_fast_roi_is_not_an_independent_temporal_sample() -> None:
    size = 32
    intrinsics = torch.tensor(
        [
            [30.0, 0.0, (size - 1) / 2.0],
            [0.0, 30.0, (size - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world_from_camera = torch.eye(4)
    image = torch.zeros((1, 3, size, size))
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.1,
        payload=image,
        calibration={
            "intrinsics": intrinsics.unsqueeze(0),
            "world_from_camera": world_from_camera.unsqueeze(0),
        },
        frame_id="camera:camera",
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            birth_extra_queries=0,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
            structured_disc_center_enabled=False,
            fast_depth_residual_enabled=False,
        )
    )
    factory = BeliefFactory(max_objects=1, geometry_dim=1, appearance_dim=8)
    belief = factory.create(
        intrinsics=intrinsics.unsqueeze(0),
        world_from_camera=world_from_camera.unsqueeze(0),
    )
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[7]]),
            position=torch.tensor([[[0.0, 0.0, 3.0]]]),
            geometry=torch.tensor([[[0.3]]]),
        )
    )
    predicted = module.project(
        belief,
        SensorContext(
            sensor_id=packet.sensor_id,
            timestamp=packet.timestamp,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            image_size=(size, size),
        ),
    )
    measured, _ = module.encode_measurements([packet], belief, predicted, None)

    # The neutral residual head copies the prior exactly. It remains a valid,
    # differentiable ordinary filter measurement, but is not new temporal data.
    torch.testing.assert_close(measured.values[..., :4], predicted.values[..., :4])
    assert not measured.auxiliary["position_independent_camera_axis_mask"].any()
    assert not measured.auxiliary["world_position_independent_axis_mask"].any()
    measured.auxiliary["world_position"].sum().backward()
    assert module.roi_updater.delta_head.bias.grad is not None


def test_structured_fast_centre_marks_lateral_but_not_copied_depth_support() -> None:
    size = 32
    intrinsics = torch.tensor(
        [
            [30.0, 0.0, (size - 1) / 2.0],
            [0.0, 30.0, (size - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world_from_camera = torch.eye(4)
    image = torch.zeros((1, 3, size, size))
    image[:, :, 14:18, 14:18] = torch.tensor([0.9, 0.2, 0.1]).reshape(1, 3, 1, 1)
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=0.1,
        payload=image,
        calibration={
            "intrinsics": intrinsics.unsqueeze(0),
            "world_from_camera": world_from_camera.unsqueeze(0),
        },
        frame_id="camera:camera",
    )
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            birth_extra_queries=0,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
            structured_disc_center_enabled=True,
            structured_disc_fast_depth_enabled=False,
        )
    )
    factory = BeliefFactory(max_objects=1, geometry_dim=1, appearance_dim=8)
    belief = factory.create(
        intrinsics=intrinsics.unsqueeze(0),
        world_from_camera=world_from_camera.unsqueeze(0),
    )
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[7]]),
            position=torch.tensor([[[0.0, 0.0, 3.0]]]),
            geometry=torch.tensor([[[0.3]]]),
        )
    )
    predicted = module.project(
        belief,
        SensorContext(
            sensor_id=packet.sensor_id,
            timestamp=packet.timestamp,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            image_size=(size, size),
        ),
    )
    measured, _ = module.encode_measurements([packet], belief, predicted, None)

    assert measured.auxiliary["structured_centre_valid"].tolist() == [[True]]
    assert measured.auxiliary["structured_depth_valid"].tolist() == [[False]]
    assert measured.auxiliary["position_independent_camera_axis_mask"].tolist() == [
        [[True, True, False]]
    ]
    assert measured.auxiliary["world_position_independent_axis_mask"].tolist() == [
        [[True, True, False]]
    ]


def test_rgb_innovation_inflates_only_depth_outlier_correction_variance() -> None:
    module = RGBObservationModule(
        RGBObservationConfig(
            max_objects=1,
            birth_extra_queries=0,
            backbone_channels=(8, 16, 24, 32),
            feature_dim=16,
            appearance_dim=8,
            roi_size=8,
            roi_hidden_dim=16,
            structured_disc_depth_outlier_relative_threshold=0.12,
            structured_disc_depth_outlier_variance_scale=9.0,
        )
    )
    measured_values = torch.tensor([[[0.0, 0.0, -2.0, 0.5, 0.2, 0.3, 0.4]]])
    position_log_variance = torch.full((1, 1, 3), -4.0)
    measured = MeasurementSet(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([0.1]),
        values=measured_values,
        log_variance=torch.full_like(measured_values, -4.0),
        existence_logits=torch.tensor([[8.0]]),
        measurement_mask=torch.tensor([[True]]),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera",
        supported_state_fields=("position",),
        auxiliary={
            "world_position": torch.zeros(1, 1, 3),
            "world_position_log_variance": position_log_variance,
            "structured_centre_valid": torch.tensor([[True]]),
        },
    )
    predicted_values = measured_values.clone()
    predicted_values[..., 3] = 0.4
    predicted = PredictedMeasurements(
        modality="rgb",
        sensor_id="camera",
        timestamp=torch.tensor([0.1]),
        values=predicted_values,
        log_variance=torch.full_like(predicted_values, -4.0),
        object_ids=torch.tensor([[7]]),
        belief_indices=torch.tensor([[0]]),
        valid_mask=torch.tensor([[True]]),
        visibility=torch.tensor([[1.0]]),
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

    innovation = module.innovation(measured, predicted, association)

    assert innovation.auxiliary["measured_depth_outlier_mask"].item()
    torch.testing.assert_close(
        innovation.auxiliary["measured_world_position_log_variance"],
        position_log_variance + math.log(9.0),
    )


def test_global_structured_assignment_skips_nonfinite_proposal_rows() -> None:
    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    expected = _paint_block(
        image,
        batch_index=0,
        top=3,
        left=3,
        height=3,
        width=3,
        colour=(1.0, 0.2, 0.1),
    )
    proposals = torch.tensor([[[float("nan"), 0.0], [0.0, 0.0]]])

    output = structured_disc_centres(image, proposals)

    assert not output.valid_mask[0, 0]
    assert output.valid_mask[0, 1]
    torch.testing.assert_close(output.centres[0, 1], expected)
