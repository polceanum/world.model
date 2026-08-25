from __future__ import annotations

import torch

from world_model.observations.rgb.losses import rgb_measurement_losses


def _loss_inputs() -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    values = torch.tensor(
        [[[0.25, -0.5, -1.0, 0.2, 0.8, 0.3, 0.1]]],
        requires_grad=True,
    )
    target = values.detach().clone()
    return (
        {
            "values": values,
            "raw_centre": torch.tensor(
                [[[-0.4, 0.1]]],
                requires_grad=True,
            ),
            "log_variance": torch.zeros_like(values),
            "existence_logits": torch.ones((1, 1), requires_grad=True),
        },
        {"values": target},
        {
            "matched": torch.ones((1, 1), dtype=torch.bool),
            "existence": torch.ones((1, 1), dtype=torch.bool),
        },
    )


def test_raw_centre_loss_trains_head_when_structured_forward_is_exact() -> None:
    outputs, targets, masks = _loss_inputs()

    losses = rgb_measurement_losses(outputs, targets, masks)
    losses["rgb_raw_centre"].backward()

    assert losses["rgb_geometry"].item() == 0.0
    assert losses["rgb_raw_centre"].item() > 0
    gradient = outputs["raw_centre"].grad
    assert gradient is not None
    assert torch.linalg.vector_norm(gradient) > 0


def test_raw_centre_shape_must_match_measurement_centres() -> None:
    outputs, targets, masks = _loss_inputs()
    outputs["raw_centre"] = torch.zeros((1, 2, 2))

    try:
        rgb_measurement_losses(outputs, targets, masks)
    except ValueError as error:
        assert "raw_centre" in str(error)
    else:
        raise AssertionError("mismatched raw centre shape was accepted")


def test_raw_radius_loss_trains_student_when_structured_forward_is_exact() -> None:
    outputs, targets, masks = _loss_inputs()
    outputs["raw_log_radius"] = torch.tensor([[[-0.25]]], requires_grad=True)
    masks["raw_log_radius"] = torch.ones((1, 1), dtype=torch.bool)

    losses = rgb_measurement_losses(outputs, targets, masks)
    losses["rgb_raw_log_radius"].backward()

    assert losses["rgb_geometry"].item() == 0.0
    assert losses["rgb_raw_log_radius"].item() > 0.0
    gradient = outputs["raw_log_radius"].grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) == 1


def test_raw_radius_output_requires_explicit_structured_support() -> None:
    outputs, targets, masks = _loss_inputs()
    outputs["raw_log_radius"] = torch.tensor([[[-0.25]]], requires_grad=True)

    losses = rgb_measurement_losses(outputs, targets, masks)

    assert "rgb_raw_log_radius" not in losses


def test_measurement_nll_calibrates_variance_without_duplicate_mean_gradient() -> None:
    prediction = torch.tensor(
        [[[1.0, -1.0, 0.5, 0.2, 0.8, 0.3, 0.1]]],
        requires_grad=True,
    )
    log_variance = torch.zeros_like(prediction, requires_grad=True)
    outputs = {
        "values": prediction,
        "log_variance": log_variance,
        "existence_logits": torch.ones((1, 1), requires_grad=True),
    }
    targets = {"values": torch.zeros_like(prediction)}
    masks = {
        "matched": torch.ones((1, 1), dtype=torch.bool),
        "existence": torch.ones((1, 1), dtype=torch.bool),
    }

    losses = rgb_measurement_losses(outputs, targets, masks)
    losses["rgb_nll"].backward()

    assert prediction.grad is None or torch.count_nonzero(prediction.grad) == 0
    assert log_variance.grad is not None
    assert torch.count_nonzero(log_variance.grad) > 0


def test_unsupported_exact_geometry_retains_existence_and_uncertainty_training() -> None:
    prediction = torch.full((1, 1, 7), 2.0, requires_grad=True)
    log_variance = torch.zeros_like(prediction, requires_grad=True)
    predicted_world = torch.full((1, 1, 3), 2.0, requires_grad=True)
    world_log_variance = torch.zeros_like(predicted_world, requires_grad=True)
    existence_logits = torch.zeros((1, 1), requires_grad=True)
    outputs = {
        "values": prediction,
        "raw_centre": prediction[..., :2],
        "log_variance": log_variance,
        "existence_logits": existence_logits,
        "world_position": predicted_world,
        "world_position_log_variance": world_log_variance,
    }
    targets = {
        "values": torch.zeros_like(prediction),
        "world_position": torch.zeros_like(predicted_world),
    }
    masks = {
        "matched": torch.ones((1, 1), dtype=torch.bool),
        "existence": torch.ones((1, 1), dtype=torch.bool),
        "geometry": torch.zeros((1, 1), dtype=torch.bool),
    }

    losses = rgb_measurement_losses(outputs, targets, masks)
    (losses["rgb_nll"] + losses["rgb_world_position_nll"] + losses["rgb_existence"]).backward()

    assert "rgb_geometry" not in losses
    assert "rgb_raw_centre" not in losses
    assert "rgb_world_position" not in losses
    assert losses["rgb_nll"].item() > 0.0
    assert losses["rgb_world_position_nll"].item() > 0.0
    assert losses["rgb_existence"].item() > 0.0
    assert prediction.grad is None
    assert predicted_world.grad is None
    assert torch.count_nonzero(log_variance.grad) > 0
    assert torch.count_nonzero(world_log_variance.grad) > 0
    assert torch.count_nonzero(existence_logits.grad) > 0


def test_invalid_roi_slots_do_not_become_negative_existence_examples() -> None:
    outputs, targets, masks = _loss_inputs()
    masks["matched"] = torch.zeros((1, 1), dtype=torch.bool)
    masks["existence"] = torch.zeros((1, 1), dtype=torch.bool)
    masks["existence_valid"] = torch.zeros((1, 1), dtype=torch.bool)

    losses = rgb_measurement_losses(outputs, targets, masks)

    assert "rgb_existence" not in losses


def test_valid_empty_roi_trains_only_existence_and_visibility_heads() -> None:
    prediction = torch.full((1, 1, 7), 2.0, requires_grad=True)
    log_variance = torch.zeros_like(prediction, requires_grad=True)
    predicted_world = torch.full((1, 1, 3), 2.0, requires_grad=True)
    world_log_variance = torch.zeros_like(predicted_world, requires_grad=True)
    existence_logits = torch.zeros((1, 1), requires_grad=True)
    visibility_logits = torch.zeros((1, 1), requires_grad=True)
    outputs = {
        "values": prediction,
        "raw_centre": prediction[..., :2],
        "log_variance": log_variance,
        "existence_logits": existence_logits,
        "visibility_logits": visibility_logits,
        "world_position": predicted_world,
        "world_position_log_variance": world_log_variance,
    }
    targets = {
        "values": torch.zeros_like(prediction),
        "visibility": torch.zeros((1, 1)),
        "world_position": torch.zeros_like(predicted_world),
    }
    masks = {
        "matched": torch.zeros((1, 1), dtype=torch.bool),
        "existence": torch.zeros((1, 1), dtype=torch.bool),
        "existence_valid": torch.ones((1, 1), dtype=torch.bool),
        "visibility_valid": torch.ones((1, 1), dtype=torch.bool),
        "geometry": torch.zeros((1, 1), dtype=torch.bool),
    }

    losses = rgb_measurement_losses(outputs, targets, masks)
    sum(losses.values()).backward()

    assert losses["rgb_existence"].item() > 0.0
    assert losses["rgb_visibility"].item() > 0.0
    assert "rgb_geometry" not in losses
    assert "rgb_colour" not in losses
    assert "rgb_nll" not in losses
    assert "rgb_world_position" not in losses
    assert "rgb_world_position_nll" not in losses
    assert torch.count_nonzero(existence_logits.grad) > 0
    assert torch.count_nonzero(visibility_logits.grad) > 0
    assert prediction.grad is None
    assert log_variance.grad is None
    assert predicted_world.grad is None
    assert world_log_variance.grad is None
