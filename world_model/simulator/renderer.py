"""Lightweight perspective renderer for labelled RGB sphere episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.simulator.camera import CameraFrame, project_camera_points, world_to_camera
from world_model.simulator.physics import SphereState


@dataclass(frozen=True)
class RenderOutput:
    """RGB frame and exact geometric/visibility labels for every padded slot."""

    rgb: Tensor
    depth_buffer: Tensor
    instance_map: Tensor
    instance_slot_map: Tensor
    visible_mask: Tensor
    full_mask: Tensor
    soft_support: Tensor
    visible_fraction: Tensor
    projected_center: Tensor
    projected_center_pixels: Tensor
    apparent_radius: Tensor
    inverse_depth: Tensor
    camera_depth: Tensor
    projected_valid: Tensor


def _background(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Create a deterministic sky/floor cue in image coordinates."""

    vertical = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
    sky_top = torch.tensor([0.08, 0.11, 0.16], dtype=dtype, device=device)
    sky_bottom = torch.tensor([0.24, 0.29, 0.34], dtype=dtype, device=device)
    gradient = (
        sky_top[:, None] * (1.0 - vertical[None, :]) + sky_bottom[:, None] * vertical[None, :]
    )
    image = gradient[:, :, None].expand(3, height, width).clone()

    # Subtle horizontal bands are a simple, non-semantic depth/orientation cue.
    row = torch.arange(height, device=device)
    bands = ((row % max(6, height // 8)) == 0) & (row > height // 2)
    image[:, bands, :] *= 0.82
    return image


def render_spheres(
    state: SphereState,
    camera: CameraFrame,
    image_size: tuple[int, int],
    *,
    edge_softness_pixels: float = 1.0,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> RenderOutput:
    """Render padded spheres with perspective, depth ordering, and occlusion.

    Rendering is deliberately non-differentiable with respect to visibility
    order; the RGB model is supervised from the returned exact labels.  Pixel
    work is vectorised as ``[N, H, W]`` tensors.
    """

    state.validate()
    camera.validate()
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if edge_softness_pixels <= 0:
        raise ValueError("edge_softness_pixels must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")

    dtype = state.position.dtype
    device = state.position.device
    count = state.max_objects
    points_camera = world_to_camera(state.position, camera.camera_from_world)
    centers_pixels, positive_depth = project_camera_points(points_camera, camera.intrinsics)
    depth = points_camera[:, 2]
    radius = state.radius[:, 0]
    focal = 0.5 * (camera.intrinsics[0, 0] + camera.intrinsics[1, 1])
    safe_depth = depth.clamp_min(1.0e-4)
    apparent_radius = focal * radius / safe_depth
    geometrically_valid = (
        state.active & positive_depth & (depth > radius + 1.0e-4) & torch.isfinite(apparent_radius)
    )
    in_view = (
        (centers_pixels[:, 0] + apparent_radius >= 0)
        & (centers_pixels[:, 0] - apparent_radius <= width - 1)
        & (centers_pixels[:, 1] + apparent_radius >= 0)
        & (centers_pixels[:, 1] - apparent_radius <= height - 1)
    )
    projected_valid = geometrically_valid & in_view
    apparent_radius = torch.where(
        projected_valid, apparent_radius, torch.zeros_like(apparent_radius)
    )
    inverse_depth = torch.where(projected_valid, safe_depth.reciprocal(), torch.zeros_like(depth))
    center_normalized = torch.stack(
        (
            2.0 * centers_pixels[:, 0] / max(width - 1, 1) - 1.0,
            2.0 * centers_pixels[:, 1] / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    center_normalized = torch.where(
        projected_valid.unsqueeze(-1),
        center_normalized,
        torch.zeros_like(center_normalized),
    )
    centers_pixels = torch.where(
        projected_valid.unsqueeze(-1),
        centers_pixels,
        torch.zeros_like(centers_pixels),
    )

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    delta_x = pixel_x.unsqueeze(0) - centers_pixels[:, 0, None, None]
    delta_y = pixel_y.unsqueeze(0) - centers_pixels[:, 1, None, None]
    safe_radius = apparent_radius.clamp_min(1.0e-4)
    radial_squared = (delta_x.square() + delta_y.square()) / safe_radius[:, None, None].square()
    full_mask = (radial_squared <= 1.0) & projected_valid[:, None, None]
    # One-pixel smooth silhouette support for anti-aliased RGB.
    radial_distance = torch.sqrt(radial_squared.clamp_min(0.0))
    soft_width = edge_softness_pixels / safe_radius[:, None, None]
    soft_support = ((1.0 - radial_distance) / soft_width.clamp_min(1.0e-4) + 0.5).clamp(0.0, 1.0)
    soft_support = torch.where(
        projected_valid[:, None, None],
        soft_support,
        torch.zeros_like(soft_support),
    )

    front_shape = torch.sqrt((1.0 - radial_squared).clamp_min(0.0))
    surface_depth = depth[:, None, None] - radius[:, None, None] * front_shape
    infinity = torch.full_like(surface_depth, torch.inf)
    surface_depth = torch.where(full_mask, surface_depth, infinity)
    depth_buffer, winning_slot = surface_depth.min(dim=0)
    has_object = torch.isfinite(depth_buffer)
    instance_slot_map = torch.where(
        has_object,
        winning_slot.to(torch.int64),
        torch.full_like(winning_slot, -1, dtype=torch.int64),
    )
    instance_map = torch.full((height, width), -1, dtype=torch.int64, device=device)
    if count > 0:
        safe_slot = winning_slot.clamp(0, max(count - 1, 0))
        winning_id = state.object_id[safe_slot]
        instance_map = torch.where(has_object, winning_id, instance_map)

    slot_indices = torch.arange(count, device=device)[:, None, None]
    visible_mask = full_mask & (winning_slot.unsqueeze(0) == slot_indices) & has_object.unsqueeze(0)
    support_pixels = full_mask.sum(dim=(-2, -1))
    visible_pixels = visible_mask.sum(dim=(-2, -1))
    visible_fraction = torch.where(
        support_pixels > 0,
        visible_pixels.to(dtype) / support_pixels.clamp_min(1).to(dtype),
        torch.zeros(count, dtype=dtype, device=device),
    )
    visible_fraction = torch.where(
        state.active, visible_fraction, torch.zeros_like(visible_fraction)
    )

    rgb = _background(height, width, dtype=dtype, device=device)
    # A front-facing radial light produces a Lambertian-like sphere cue while
    # retaining exact object albedo as an association label.
    shade = (0.48 + 0.52 * front_shape).clamp(0.0, 1.0)
    for slot in range(count):
        alpha = soft_support[slot] * visible_mask[slot].to(dtype)
        sphere_rgb = state.albedo[slot, :, None, None] * shade[slot][None, :, :]
        rgb = rgb * (1.0 - alpha.unsqueeze(0)) + sphere_rgb * alpha.unsqueeze(0)
    if noise_std > 0:
        noise = torch.randn(
            rgb.shape,
            dtype=rgb.dtype,
            device=rgb.device,
            generator=generator,
        )
        rgb = rgb + noise_std * noise
    rgb = rgb.clamp(0.0, 1.0).to(torch.float32)
    depth_buffer = torch.where(has_object, depth_buffer, torch.zeros_like(depth_buffer))
    return RenderOutput(
        rgb=rgb,
        depth_buffer=depth_buffer,
        instance_map=instance_map,
        instance_slot_map=instance_slot_map,
        visible_mask=visible_mask,
        full_mask=full_mask,
        soft_support=soft_support.to(torch.float32),
        visible_fraction=visible_fraction.to(torch.float32),
        projected_center=center_normalized.to(torch.float32),
        projected_center_pixels=centers_pixels.to(torch.float32),
        apparent_radius=apparent_radius.to(torch.float32),
        inverse_depth=inverse_depth.to(torch.float32),
        camera_depth=depth.to(torch.float32),
        projected_valid=projected_valid,
    )
