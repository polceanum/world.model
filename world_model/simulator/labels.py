"""Exact simulator-derived labels kept separate from RGB runtime inputs."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from world_model.simulator.physics import SphereState
from world_model.simulator.renderer import RenderOutput


def make_perception_labels(
    state: SphereState,
    rendered: RenderOutput,
    image_size: tuple[int, int],
) -> dict[str, Tensor]:
    """Build one-frame RGB supervision labels from exact simulator geometry.

    These labels may be used for training/evaluation only.  They are never
    included in an RGB observation packet consumed by the online runtime.
    """

    height, width = image_size
    radius_scale = 0.5 * min(height, width)
    radius_normalized = rendered.apparent_radius / max(radius_scale, 1.0)
    log_radius_normalized = torch.where(
        rendered.projected_valid,
        radius_normalized.clamp_min(1.0e-6).log(),
        torch.zeros_like(radius_normalized),
    )
    return {
        "projected_center": rendered.projected_center.clone(),
        "projected_center_pixels": rendered.projected_center_pixels.clone(),
        "apparent_radius": rendered.apparent_radius.clone(),
        "apparent_radius_normalized": radius_normalized,
        "log_apparent_radius_normalized": log_radius_normalized,
        "inverse_depth": rendered.inverse_depth.clone(),
        "camera_depth": rendered.camera_depth.clone(),
        "visible_fraction": rendered.visible_fraction.clone(),
        "visible": (rendered.visible_fraction > 0),
        "existence": state.active.clone(),
        "projected_valid": rendered.projected_valid.clone(),
        "segmentation_mask": rendered.visible_mask.clone(),
        "full_mask": rendered.full_mask.clone(),
        "soft_support": rendered.soft_support.clone(),
        "instance_map": rendered.instance_map.clone(),
        "instance_slot_map": rendered.instance_slot_map.clone(),
        "association": state.object_id.clone(),
        "albedo": state.albedo.clone(),
    }


def validate_perception_labels(
    labels: dict[str, Tensor],
    *,
    max_objects: int,
    image_size: tuple[int, int],
) -> None:
    """Validate shapes, ranges, and padded masks for RGB supervision."""

    height, width = image_size
    expected = {
        "projected_center": (max_objects, 2),
        "projected_center_pixels": (max_objects, 2),
        "apparent_radius": (max_objects,),
        "apparent_radius_normalized": (max_objects,),
        "log_apparent_radius_normalized": (max_objects,),
        "inverse_depth": (max_objects,),
        "camera_depth": (max_objects,),
        "visible_fraction": (max_objects,),
        "visible": (max_objects,),
        "existence": (max_objects,),
        "projected_valid": (max_objects,),
        "segmentation_mask": (max_objects, height, width),
        "full_mask": (max_objects, height, width),
        "soft_support": (max_objects, height, width),
        "instance_map": (height, width),
        "instance_slot_map": (height, width),
        "association": (max_objects,),
        "albedo": (max_objects, 3),
    }
    for name, shape in expected.items():
        if name not in labels:
            raise ValueError(f"missing perception label {name!r}")
        if tuple(labels[name].shape) != shape:
            raise ValueError(
                f"label {name!r} must have shape {shape}, got {tuple(labels[name].shape)}"
            )
    if not bool(torch.isfinite(labels["soft_support"]).all()):
        raise ValueError("soft support contains NaN or Inf")
    if bool(torch.any((labels["visible_fraction"] < 0) | (labels["visible_fraction"] > 1))):
        raise ValueError("visible_fraction must lie in [0, 1]")
    if bool(torch.any(labels["projected_center"][labels["projected_valid"]].abs() > 8.0)):
        # Off-screen centres can legitimately leave [-1, 1], but huge values
        # indicate a projection/depth failure.
        raise ValueError("projected centres are implausibly far outside the image")
    if not math.isfinite(float(labels["apparent_radius"].max())):
        raise ValueError("apparent radius contains NaN or Inf")
    if bool(torch.any((~labels["existence"]) & (labels["association"] != -1))):
        raise ValueError("nonexistent/padded objects must have association ID -1")
