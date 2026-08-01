"""Losses for global and fast RGB structured measurements."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def gaussian_nll(
    prediction: Tensor,
    target: Tensor,
    log_variance: Tensor,
    mask: Tensor,
    *,
    detach_mean_error: bool = False,
) -> Tensor:
    squared_error = (prediction - target).square()
    if detach_mean_error:
        # Geometry/colour/world-position means already have explicit robust
        # objectives.  This term is their uncertainty-calibration objective;
        # allowing its inverse-variance factor to backpropagate into the same
        # mean duplicated and frequently dominated the useful mean gradient on
        # hard tracking frames.
        squared_error = squared_error.detach()
    loss = 0.5 * (squared_error * (-log_variance).exp() + log_variance)
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
    nll = gaussian_nll(
        predicted,
        target,
        log_variance,
        matched,
        detach_mean_error=True,
    )
    losses = {
        "rgb_existence": existence,
        "rgb_geometry": geometry,
        "rgb_colour": colour,
        "rgb_nll": nll,
    }
    raw_centre = outputs.get("raw_centre")
    if raw_centre is not None:
        if raw_centre.shape != predicted[..., :2].shape:
            raise ValueError("raw_centre must match the first two measurement dimensions")
        losses["rgb_raw_centre"] = (
            F.smooth_l1_loss(raw_centre[matched], target[..., :2][matched])
            if matched.any()
            else raw_centre.sum() * 0
        )
    if "world_position" in outputs and "world_position" in targets:
        predicted_world = outputs["world_position"]
        target_world = targets["world_position"]
        if predicted_world.shape != target_world.shape:
            raise ValueError("world_position output and target shapes must match")
        if matched.any():
            losses["rgb_world_position"] = F.smooth_l1_loss(
                predicted_world[matched],
                target_world[matched],
            )
        else:
            losses["rgb_world_position"] = predicted_world.sum() * 0
        if "world_position_log_variance" in outputs:
            world_log_variance = outputs["world_position_log_variance"].clamp(-12.0, 8.0)
            if world_log_variance.shape != predicted_world.shape:
                raise ValueError("world_position_log_variance must match world_position")
            losses["rgb_world_position_nll"] = gaussian_nll(
                predicted_world,
                target_world,
                world_log_variance,
                matched,
                detach_mean_error=True,
            )
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
