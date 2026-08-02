"""Losses for global and fast RGB structured measurements."""

from __future__ import annotations

import torch
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

    def supervision_mask(
        name: str,
        default: Tensor,
        *,
        subset_of_matched: bool = True,
    ) -> Tensor:
        mask = masks.get(name, default).bool()
        if mask.shape != matched.shape:
            raise ValueError(f"{name} mask must match the matched mask")
        if subset_of_matched and torch.any(mask & ~matched):
            raise ValueError(f"{name} mask cannot include unmatched measurements")
        return mask

    geometry_matched = supervision_mask("geometry", matched)
    colour_matched = supervision_mask("colour", matched)
    nll_matched = supervision_mask("nll", matched)
    appearance_matched = supervision_mask("appearance", matched)
    world_matched = supervision_mask("world", geometry_matched)
    world_nll_matched = supervision_mask("world_nll", matched)
    log_variance = outputs["log_variance"].clamp(-10.0, 5.0)
    existence_target = masks.get("existence", matched).to(predicted.dtype)
    if existence_target.shape != matched.shape:
        raise ValueError("existence target must match the matched mask")
    existence_valid = supervision_mask(
        "existence_valid",
        torch.ones_like(matched),
        subset_of_matched=False,
    )
    losses: dict[str, Tensor] = {}
    if geometry_matched.any():
        losses["rgb_geometry"] = F.smooth_l1_loss(
            predicted[..., :4][geometry_matched],
            target[..., :4][geometry_matched],
        )
    if colour_matched.any():
        losses["rgb_colour"] = F.smooth_l1_loss(
            predicted[..., 4:7][colour_matched],
            target[..., 4:7][colour_matched],
        )
    if nll_matched.any():
        losses["rgb_nll"] = gaussian_nll(
            predicted,
            target,
            log_variance,
            nll_matched,
            detach_mean_error=True,
        )
    if existence_valid.any():
        losses["rgb_existence"] = F.binary_cross_entropy_with_logits(
            outputs["existence_logits"][existence_valid],
            existence_target[existence_valid],
        )
    raw_centre = outputs.get("raw_centre")
    if raw_centre is not None:
        if raw_centre.shape != predicted[..., :2].shape:
            raise ValueError("raw_centre must match the first two measurement dimensions")
        if geometry_matched.any():
            losses["rgb_raw_centre"] = F.smooth_l1_loss(
                raw_centre[geometry_matched],
                target[..., :2][geometry_matched],
            )
    if "world_position" in outputs and "world_position" in targets:
        predicted_world = outputs["world_position"]
        target_world = targets["world_position"]
        if predicted_world.shape != target_world.shape:
            raise ValueError("world_position output and target shapes must match")
        if world_matched.any():
            losses["rgb_world_position"] = F.smooth_l1_loss(
                predicted_world[world_matched],
                target_world[world_matched],
            )
        if "world_position_log_variance" in outputs:
            world_log_variance = outputs["world_position_log_variance"].clamp(-12.0, 8.0)
            if world_log_variance.shape != predicted_world.shape:
                raise ValueError("world_position_log_variance must match world_position")
            if world_nll_matched.any():
                losses["rgb_world_position_nll"] = gaussian_nll(
                    predicted_world,
                    target_world,
                    world_log_variance,
                    world_nll_matched,
                    detach_mean_error=True,
                )
    if "visibility_logits" in outputs and "visibility" in targets:
        visibility_valid = supervision_mask(
            "visibility_valid",
            matched,
            subset_of_matched=False,
        )
        if targets["visibility"].shape != matched.shape:
            raise ValueError("visibility target must match the matched mask")
        if visibility_valid.any():
            losses["rgb_visibility"] = F.binary_cross_entropy_with_logits(
                outputs["visibility_logits"][visibility_valid],
                targets["visibility"].to(predicted.dtype)[visibility_valid],
            )
    if "appearance" in outputs and "appearance" in targets and appearance_matched.any():
        losses["rgb_appearance"] = (
            1.0
            - F.cosine_similarity(
                outputs["appearance"][appearance_matched],
                targets["appearance"][appearance_matched],
                dim=-1,
            )
        ).mean()
    return losses
