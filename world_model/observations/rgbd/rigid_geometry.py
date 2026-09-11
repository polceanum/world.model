"""Parameter-free rigid geometry fitting from public calibrated RGB-D surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.belief import RigidGeometryCodec, RigidPrimitive
from world_model.simulator.camera import backproject_pixels, camera_to_world


@dataclass(frozen=True)
class ObservableRigidGeometry:
    """A geometry estimate derived only from depth, masks, and calibration."""

    primitive: Tensor
    world_position: Tensor
    orientation: Tensor
    radius: Tensor
    half_extents: Tensor
    sphere_residual: Tensor
    box_residual: Tensor
    point_count: Tensor
    valid: Tensor

    def encode(self, *, geometry_dim: int = 5) -> Tensor:
        sphere = RigidGeometryCodec.encode_sphere(
            self.radius.unsqueeze(-1), geometry_dim=geometry_dim
        )
        box = RigidGeometryCodec.encode_box(self.half_extents, geometry_dim=geometry_dim)
        return torch.where((self.primitive == int(RigidPrimitive.BOX)).unsqueeze(-1), box, sphere)


def _matrix_to_quaternion(rotation: Tensor) -> Tensor:
    """Convert a proper 3x3 rotation to scalar-last quaternion."""

    diagonal = torch.diagonal(rotation)
    w = 0.5 * torch.sqrt((1.0 + diagonal.sum()).clamp_min(0.0))
    x = 0.5 * torch.sqrt((1.0 + diagonal[0] - diagonal[1] - diagonal[2]).clamp_min(0.0))
    y = 0.5 * torch.sqrt((1.0 - diagonal[0] + diagonal[1] - diagonal[2]).clamp_min(0.0))
    z = 0.5 * torch.sqrt((1.0 - diagonal[0] - diagonal[1] + diagonal[2]).clamp_min(0.0))
    quaternion = torch.stack(
        (
            torch.copysign(x, rotation[2, 1] - rotation[1, 2]),
            torch.copysign(y, rotation[0, 2] - rotation[2, 0]),
            torch.copysign(z, rotation[1, 0] - rotation[0, 1]),
            w,
        )
    )
    norm = torch.linalg.vector_norm(quaternion)
    identity = quaternion.new_tensor([0.0, 0.0, 0.0, 1.0])
    return torch.where(norm > 1.0e-8, quaternion / norm.clamp_min(1.0e-8), identity)


def _surface_points(
    depth: Tensor,
    mask: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    *,
    mask_threshold: float,
    max_points_per_frame: int,
) -> tuple[Tensor, Tensor]:
    frames, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype, device=depth.device),
        torch.arange(width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    pixels = torch.stack((x, y), dim=-1)
    clouds: list[Tensor] = []
    frame_indices: list[Tensor] = []
    for frame in range(frames):
        valid = (
            (mask[frame] >= mask_threshold) & torch.isfinite(depth[frame]) & (depth[frame] > 0.0)
        )
        selected_pixels = pixels[valid]
        selected_depth = depth[frame][valid]
        if selected_depth.numel() == 0:
            continue
        if selected_depth.numel() > max_points_per_frame:
            stride = max(1, selected_depth.numel() // max_points_per_frame)
            indices = torch.arange(
                0,
                selected_depth.numel(),
                stride,
                device=depth.device,
            )[:max_points_per_frame]
            selected_pixels = selected_pixels[indices]
            selected_depth = selected_depth[indices]
        camera_points = backproject_pixels(
            selected_pixels,
            selected_depth,
            intrinsics[frame],
        )
        world_points = camera_to_world(camera_points, world_from_camera[frame])
        clouds.append(world_points)
        frame_indices.append(
            torch.full(
                (world_points.shape[0],),
                frame,
                dtype=torch.int64,
                device=depth.device,
            )
        )
    if not clouds:
        return depth.new_zeros((0, 3)), torch.zeros(0, dtype=torch.int64, device=depth.device)
    return torch.cat(clouds, dim=0), torch.cat(frame_indices, dim=0)


def _box_from_axes(points: Tensor, axes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    cloud_mean = points.mean(dim=0)
    local = (points - cloud_mean) @ axes
    lower = local.amin(dim=0)
    upper = local.amax(dim=0)
    local_center = 0.5 * (lower + upper)
    center = cloud_mean + axes @ local_center
    half_extents = 0.5 * (upper - lower).clamp_min(1.0e-6)
    centered_local = (points - center) @ axes
    surface_distance = (half_extents - centered_local.abs()).abs().amin(dim=-1)
    residual = surface_distance.square().mean().sqrt()
    return center, half_extents, residual


def _refine_box_axes(
    points: Tensor,
    frame_indices: Tensor,
    pca_axes: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Use public per-view surface planes to refine an oriented-box basis."""

    candidates = [pca_axes]
    normals: list[Tensor] = []
    for frame in torch.unique(frame_indices).tolist():
        cloud = points[frame_indices == frame]
        if cloud.shape[0] < 16:
            continue
        centered = cloud - cloud.mean(dim=0)
        covariance = centered.transpose(0, 1) @ centered / max(cloud.shape[0] - 1, 1)
        _, vectors = torch.linalg.eigh(covariance)
        normals.append(vectors[:, 0])
    for first_index, first in enumerate(normals):
        first = first / torch.linalg.vector_norm(first).clamp_min(1.0e-12)
        for second_index, second in enumerate(normals):
            if first_index == second_index or abs(float(first.dot(second))) > 0.45:
                continue
            second = second - first * first.dot(second)
            second = second / torch.linalg.vector_norm(second).clamp_min(1.0e-12)
            third = torch.linalg.cross(first, second)
            third = third / torch.linalg.vector_norm(third).clamp_min(1.0e-12)
            candidates.append(torch.stack((first, second, third), dim=-1))
    best_axes = candidates[0]
    best_center, best_extents, best_residual = _box_from_axes(points, best_axes)
    for axes in candidates[1:]:
        center, extents, residual = _box_from_axes(points, axes)
        if bool(residual < best_residual):
            best_axes = axes
            best_center = center
            best_extents = extents
            best_residual = residual
    return best_axes, best_center, best_extents, best_residual


def _fit_sphere(points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fit a sphere algebraically to public surface points.

    Every measured surface point satisfies ``2 p·c + (r² - c²) = p²``.
    Solving that overdetermined system avoids the view- and pixel-dependent
    centre jitter of an axis-aligned bounding box while remaining entirely
    parameter free.
    """

    design = torch.cat((2.0 * points, torch.ones_like(points[:, :1])), dim=-1)
    squared_norm = points.square().sum(dim=-1, keepdim=True)
    solution = torch.linalg.lstsq(design, squared_norm).solution.squeeze(-1)
    center = solution[:3]
    radial = torch.linalg.vector_norm(points - center, dim=-1)
    radius = radial.median()
    residual = (radial - radius).square().mean().sqrt()
    return center, radius, residual


def fit_rigid_geometry_from_rgbd(
    depth: Tensor,
    mask: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    *,
    centres_world: Tensor | None = None,
    mask_threshold: float = 0.5,
    max_points_per_frame: int = 2048,
    minimum_points: int = 64,
    box_decision_ratio: float = 0.72,
) -> ObservableRigidGeometry:
    """Fit one sphere or box from one track's public multi-view surfaces.

    Inputs have shapes ``depth[T,H,W]``, ``mask[T,H,W]``, transforms
    ``[T,4,4]``, and intrinsics ``[T,3,3]``.  Optional public centre estimates
    compensate translation before accumulating a moving track.  No object ID,
    simulator state, full silhouette, or shape label is accepted.
    """

    if depth.ndim != 3 or mask.shape != depth.shape:
        raise ValueError("depth and mask must share shape [T,H,W]")
    frames = depth.shape[0]
    if world_from_camera.shape != (frames, 4, 4) or intrinsics.shape != (frames, 3, 3):
        raise ValueError("calibration must have shapes [T,4,4] and [T,3,3]")
    if depth.dtype != mask.dtype or not depth.is_floating_point():
        raise TypeError("depth and mask must share a floating dtype")
    if any(
        value.dtype != depth.dtype or value.device != depth.device
        for value in (world_from_camera, intrinsics)
    ):
        raise ValueError("RGB-D geometry inputs must share dtype and device")
    if centres_world is not None and centres_world.shape != (frames, 3):
        raise ValueError("centres_world must have shape [T,3]")
    if max_points_per_frame < 16 or minimum_points < 16:
        raise ValueError("point limits must be at least 16")
    if not 0.0 < box_decision_ratio < 1.0:
        raise ValueError("box_decision_ratio must lie in (0,1)")

    points, frame_indices = _surface_points(
        depth,
        mask,
        world_from_camera,
        intrinsics,
        mask_threshold=mask_threshold,
        max_points_per_frame=max_points_per_frame,
    )
    point_count = torch.tensor(points.shape[0], dtype=torch.int64, device=depth.device)
    if centres_world is not None and points.numel() > 0:
        reference = centres_world[-1]
        points = points - centres_world[frame_indices] + reference
    if points.shape[0] < minimum_points:
        zero = depth.new_zeros(())
        return ObservableRigidGeometry(
            primitive=torch.tensor(
                int(RigidPrimitive.SPHERE), dtype=torch.int64, device=depth.device
            ),
            world_position=depth.new_zeros(3),
            orientation=depth.new_tensor([0.0, 0.0, 0.0, 1.0]),
            radius=zero,
            half_extents=depth.new_zeros(3),
            sphere_residual=depth.new_full((), torch.inf),
            box_residual=depth.new_full((), torch.inf),
            point_count=point_count,
            valid=torch.tensor(False, dtype=torch.bool, device=depth.device),
        )

    cloud_mean = points.mean(dim=0)
    centered = points - cloud_mean
    covariance = centered.transpose(0, 1) @ centered / max(points.shape[0] - 1, 1)
    _, axes = torch.linalg.eigh(covariance)
    # Eigenvectors are columns and define local-to-world orientation.  Enforce
    # a proper basis without using any private orientation label.
    if torch.linalg.det(axes) < 0.0:
        axes = axes.clone()
        axes[:, 0] = -axes[:, 0]
    axes, box_center, half_extents, box_residual = _refine_box_axes(
        points,
        frame_indices,
        axes,
    )

    sphere_center, sphere_radius, sphere_residual = _fit_sphere(points)
    scale = torch.linalg.vector_norm(half_extents).clamp_min(1.0e-6)
    choose_box = box_residual < box_decision_ratio * sphere_residual
    primitive = torch.where(
        choose_box,
        torch.tensor(int(RigidPrimitive.BOX), dtype=torch.int64, device=depth.device),
        torch.tensor(int(RigidPrimitive.SPHERE), dtype=torch.int64, device=depth.device),
    )
    position = torch.where(choose_box, box_center, sphere_center)
    radius = torch.where(choose_box, scale, sphere_radius)
    orientation = torch.where(
        choose_box,
        _matrix_to_quaternion(axes),
        depth.new_tensor([0.0, 0.0, 0.0, 1.0]),
    )
    valid = (
        torch.isfinite(position).all()
        & torch.isfinite(half_extents).all()
        & torch.isfinite(radius)
        & (radius > 0.0)
    )
    return ObservableRigidGeometry(
        primitive=primitive,
        world_position=position,
        orientation=orientation,
        radius=radius,
        half_extents=half_extents,
        sphere_residual=sphere_residual,
        box_residual=box_residual,
        point_count=point_count,
        valid=valid,
    )


__all__ = ["ObservableRigidGeometry", "fit_rigid_geometry_from_rgbd"]
