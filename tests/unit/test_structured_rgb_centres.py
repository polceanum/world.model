from __future__ import annotations

import math

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.fusion import AssociationResult
from world_model.observations import (
    MeasurementSet,
    ObservationContext,
    ObservationPacket,
    PredictedMeasurements,
    SensorContext,
)
from world_model.observations.rgb import RGBObservationConfig, RGBObservationModule
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
    torch.testing.assert_close(result.centres[0], expected, rtol=0.0, atol=0.055)


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
    assert result.component_pixel_count.item() >= 4
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
    assert fast_raw.requires_grad
    fast_raw.sum().backward()
    assert module.roi_updater.delta_head.bias.grad is not None


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
