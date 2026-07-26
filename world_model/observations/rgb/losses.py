"""Losses for global and fast RGB structured measurements."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def gaussian_nll(
    prediction: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
) -> Tensor:
    loss = 0.5 * ((prediction - target).square() * (-log_variance).exp() + log_variance)
    expanded_mask = mask.unsqueeze(-1).expand_as(loss)
    return loss.masked_select(expanded_mask).mean() if expanded_mask.any() else loss.sum() * 0


def rgb_measurement_losses(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    masks: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Compute matched proposal losses.

    Training code supplies already Hungarian-aligned targets and ``matched``.
    Keeping matching outside this function makes assignments inspectable.
    """

    predicted = outputs["values"]
    target = targets["values"]
    matched = masks["matched"].bool()
    log_variance = outputs["log_variance"].clamp(-10.0, 5.0)
    existence_target = masks.get("existence", matched).to(predicted.dtype)
    existence = F.binary_cross_entropy_with_logits(outputs["existence_logits"], existence_target)
    if matched.any():
        geometry = F.smooth_l1_loss(
            predicted[..., :4][matched],
            target[..., :4][matched],
        )
        colour = F.smooth_l1_loss(
            predicted[..., 4:7][matched],
            target[..., 4:7][matched],
        )
    else:
        geometry = predicted.sum() * 0
        colour = predicted.sum() * 0
    nll = gaussian_nll(predicted, target, log_variance, matched)
    losses = {
        "rgb_existence": existence,
        "rgb_geometry": geometry,
        "rgb_colour": colour,
        "rgb_nll": nll,
    }
    if "visibility_logits" in outputs and "visibility" in targets:
        losses["rgb_visibility"] = (
            F.binary_cross_entropy_with_logits(
                outputs["visibility_logits"][matched],
                targets["visibility"].to(predicted.dtype)[matched],
            )
            if matched.any()
            else predicted.sum() * 0
        )
    if "appearance" in outputs and "appearance" in targets and matched.any():
        losses["rgb_appearance"] = (
            1.0
            - F.cosine_similarity(
                outputs["appearance"][matched],
                targets["appearance"][matched],
                dim=-1,
            )
        ).mean()
    return losses
