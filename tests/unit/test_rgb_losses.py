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
