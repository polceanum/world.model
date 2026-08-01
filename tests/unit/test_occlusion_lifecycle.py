from __future__ import annotations

import pytest
import torch

from world_model.belief import (
    BeliefFactory,
    LifecycleConfig,
    MotionMode,
    ObjectLifecycle,
)
from world_model.observations import SensorContext
from world_model.observations.rgb import RGBMeasurementProjector


def _sensor_context() -> SensorContext:
    return SensorContext(
        sensor_id="camera",
        timestamp=0.0,
        calibration={
            "intrinsics": torch.tensor(
                [
                    [30.0, 0.0, 15.5],
                    [0.0, 30.0, 15.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "world_from_camera": torch.eye(4),
        },
        frame_id="camera:camera",
        image_size=(32, 32),
    )


def _overlapping_belief(*, far_x: float = 0.0, requires_grad: bool = False):
    belief = BeliefFactory(max_objects=2, geometry_dim=1).create()
    position = torch.tensor(
        [[[0.0, 0.0, 2.0], [far_x, 0.0, 4.0]]],
        requires_grad=requires_grad,
    )
    objects = belief.objects.replace(
        active=torch.tensor([[True, True]]),
        object_id=torch.tensor([[4, 9]]),
        position=position,
        geometry=torch.tensor([[[0.5], [0.5]]]),
    )
    return belief.replace(objects=objects), position


def test_nearer_projected_circle_fully_occludes_farther_identity() -> None:
    belief, _ = _overlapping_belief()
    predicted = RGBMeasurementProjector()(belief, _sensor_context())

    torch.testing.assert_close(
        predicted.auxiliary["pairwise_occlusion_fraction"][0, 1, 0],
        torch.tensor(1.0),
    )
    assert not predicted.auxiliary["occluded_mask"][0, 0]
    assert predicted.auxiliary["occluded_mask"][0, 1]
    assert predicted.valid_mask[0, 0]
    assert not predicted.valid_mask[0, 1]
    assert predicted.visibility[0, 1] == 0
    assert torch.equal(predicted.object_ids, torch.tensor([[4, 9]]))
    assert torch.equal(predicted.rois[0, 1], torch.zeros(4))


def test_partial_circle_occlusion_remains_differentiable() -> None:
    belief, position = _overlapping_belief(far_x=0.7, requires_grad=True)
    predicted = RGBMeasurementProjector()(belief, _sensor_context())
    occlusion = predicted.auxiliary["occlusion_fraction"]
    assert 0.0 < occlusion[0, 1] < 1.0

    occlusion.sum().backward()
    assert position.grad is not None
    assert torch.isfinite(position.grad).all()
    assert position.grad.abs().sum() > 0


def test_out_of_view_track_is_unobservable_not_a_visible_miss() -> None:
    belief = BeliefFactory(max_objects=1, geometry_dim=1).create()
    objects = belief.objects.replace(
        active=torch.tensor([[True]]),
        object_id=torch.tensor([[4]]),
        position=torch.tensor([[[0.0, 0.0, -1.0]]]),
        existence_logit=torch.zeros(1, 1),
    )
    belief = belief.replace(objects=objects)
    predicted = RGBMeasurementProjector()(belief, _sensor_context())

    assert not predicted.valid_mask[0, 0]
    assert predicted.auxiliary["unobservable_mask"][0, 0]
    posterior = ObjectLifecycle().update_visibility(
        belief,
        torch.tensor([[False]]),
        occluded_mask=predicted.auxiliary["unobservable_mask"],
    )
    assert posterior.objects.existence_logit[0, 0].item() == pytest.approx(-0.04)
    assert posterior.objects.mode[0, 0] == MotionMode.OCCLUDED


def test_projected_covariance_uses_rgb_coordinate_units() -> None:
    belief = BeliefFactory(max_objects=1, geometry_dim=1).create()
    position_variance = torch.tensor([0.04, 0.01, 0.01])
    objects = belief.objects.replace(
        active=torch.tensor([[True]]),
        object_id=torch.tensor([[4]]),
        position=torch.tensor([[[0.0, 0.0, 4.0]]]),
        geometry=torch.tensor([[[0.5]]]),
        fast_log_variance=belief.objects.fast_log_variance.clone(),
    )
    objects.fast_log_variance[0, 0, :3] = position_variance.log()
    predicted = RGBMeasurementProjector()(belief.replace(objects=objects), _sensor_context())
    variance = predicted.log_variance[0, 0].exp()

    normalized_focal = 2.0 * 30.0 / 31.0
    assert variance[0].item() == pytest.approx(
        (normalized_focal / 4.0) ** 2 * 0.04,
        rel=1.0e-5,
    )
    assert variance[1].item() == pytest.approx(
        (normalized_focal / 4.0) ** 2 * 0.01,
        rel=1.0e-5,
    )
    assert variance[2].item() == pytest.approx(0.01 / 4.0**2, rel=1.0e-5)
    # Inverse-depth uncertainty falls below the configured numerical floor,
    # and colour dimensions do not inherit metre-squared state covariance.
    torch.testing.assert_close(variance[3:], torch.full_like(variance[3:], 1.0e-4))


def test_roi_expansion_uses_projected_not_metre_space_uncertainty() -> None:
    belief = BeliefFactory(max_objects=1, geometry_dim=1).create()
    objects = belief.objects.replace(
        active=torch.tensor([[True]]),
        object_id=torch.tensor([[4]]),
        position=torch.tensor([[[0.0, 0.0, 4.0]]]),
        geometry=torch.tensor([[[0.1]]]),
        fast_log_variance=belief.objects.fast_log_variance.clone(),
    )
    objects.fast_log_variance[0, 0, :3] = torch.tensor([0.04, 0.01, 0.01]).log()
    projector = RGBMeasurementProjector()
    predicted = projector(belief.replace(objects=objects), _sensor_context())
    assert predicted.rois is not None

    centre_standard_deviation = predicted.log_variance[0, 0, :2].exp().sqrt().amax()
    expected_half_width = (
        predicted.values[0, 0, 2].exp()
        + projector.config.uncertainty_roi_scale * centre_standard_deviation
    )
    actual_half_width = (predicted.rois[0, 0, 2] - predicted.rois[0, 0, 0]) / 2.0
    assert actual_half_width.item() == pytest.approx(
        expected_half_width.item(),
        rel=1.0e-5,
    )


def test_occluded_miss_preserves_identity_longer_than_visible_miss() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.replace(
        active=torch.tensor([[True]]),
        object_id=torch.tensor([[17]]),
        existence_logit=torch.zeros(1, 1),
    )
    belief = belief.replace(objects=objects)
    lifecycle = ObjectLifecycle(
        LifecycleConfig(
            max_missed_steps=5,
            missed_existence_delta=-0.35,
            occluded_existence_delta=-0.04,
            removal_existence_logit=-0.5,
        )
    )
    missed = torch.tensor([[False]])

    visible_missing = belief
    occluded = belief
    for _ in range(2):
        visible_missing = lifecycle.update_visibility(
            visible_missing,
            missed,
            occluded_mask=torch.tensor([[False]]),
        )
        occluded = lifecycle.update_visibility(
            occluded,
            missed,
            occluded_mask=torch.tensor([[True]]),
        )

    assert not visible_missing.objects.active[0, 0]
    assert visible_missing.objects.object_id[0, 0] == -1
    assert occluded.objects.active[0, 0]
    assert occluded.objects.object_id[0, 0] == 17
    assert occluded.objects.existence_logit[0, 0].item() == pytest.approx(-0.08)
    assert occluded.objects.mode[0, 0] == MotionMode.OCCLUDED
    assert occluded.objects.missed_steps[0, 0] == 2
    # The filter owns uncertainty expansion. Lifecycle updates only identity,
    # counts, existence, visibility, and modes, so normal correction followed
    # by lifecycle accounting cannot inflate a missed track twice.
    torch.testing.assert_close(
        occluded.objects.fast_log_variance,
        belief.objects.fast_log_variance,
    )

    reappeared = lifecycle.update_visibility(
        occluded,
        torch.tensor([[True]]),
        occluded_mask=torch.tensor([[False]]),
    )
    assert reappeared.objects.object_id[0, 0] == 17
    assert reappeared.objects.mode[0, 0] == MotionMode.FREE
    assert reappeared.objects.missed_steps[0, 0] == 0


def test_occlusion_uses_its_own_configured_identity_retention_duration() -> None:
    belief = BeliefFactory(max_objects=1).create()
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[17]]),
            existence_logit=torch.tensor([[8.0]]),
        )
    )
    lifecycle = ObjectLifecycle(
        LifecycleConfig(
            max_missed_steps=2,
            max_occluded_steps=5,
            removal_existence_logit=-20.0,
        )
    )

    visible_missing = belief
    occluded = belief
    for _ in range(3):
        visible_missing = lifecycle.update_visibility(
            visible_missing,
            torch.tensor([[False]]),
            occluded_mask=torch.tensor([[False]]),
        )
        occluded = lifecycle.update_visibility(
            occluded,
            torch.tensor([[False]]),
            occluded_mask=torch.tensor([[True]]),
        )

    assert not visible_missing.objects.active[0, 0]
    assert occluded.objects.active[0, 0]
    for _ in range(3):
        occluded = lifecycle.update_visibility(
            occluded,
            torch.tensor([[False]]),
            occluded_mask=torch.tensor([[True]]),
        )
    assert not occluded.objects.active[0, 0]
