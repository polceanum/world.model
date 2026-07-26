"""Numerically stable scalar-last quaternion operations."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_quaternion(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalise ``[...,4]`` quaternions, mapping degenerate values to identity."""

    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    return torch.where(norm > eps, quaternion / norm.clamp_min(eps), identity)


def quaternion_conjugate(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)


def quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product for scalar-last ``[x,y,z,w]`` quaternions."""

    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternion operands must have final dimension 4")
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    scalar = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, scalar), dim=-1)


def quaternion_exp(half_angle_vector: Tensor, eps: float = 1e-8) -> Tensor:
    """Exponential of a pure quaternion.

    ``half_angle_vector`` is axis times half rotation angle.  For a conventional
    rotation vector use :func:`quaternion_from_rotation_vector`.
    """

    if half_angle_vector.shape[-1] != 3:
        raise ValueError("half-angle vector must have final dimension 3")
    angle = torch.linalg.vector_norm(half_angle_vector, dim=-1, keepdim=True)
    angle_sq = angle.square()
    scale = torch.where(
        angle > eps,
        torch.sin(angle) / angle.clamp_min(eps),
        1.0 - angle_sq / 6.0 + angle_sq.square() / 120.0,
    )
    return normalize_quaternion(torch.cat((half_angle_vector * scale, torch.cos(angle)), dim=-1))


def quaternion_from_rotation_vector(rotation_vector: Tensor) -> Tensor:
    """Convert an axis-angle rotation vector in radians to a quaternion."""

    return quaternion_exp(0.5 * rotation_vector)


def integrate_quaternion(
    orientation: Tensor,
    angular_velocity: Tensor,
    dt: float | Tensor,
) -> Tensor:
    """Integrate body-frame angular velocity for real, broadcastable seconds."""

    if orientation.shape[:-1] != angular_velocity.shape[:-1]:
        raise ValueError("orientation and angular velocity leading shapes must match")
    delta_time = torch.as_tensor(
        dt,
        device=angular_velocity.device,
        dtype=angular_velocity.dtype,
    )
    while delta_time.ndim < angular_velocity.ndim - 1:
        delta_time = delta_time.unsqueeze(-1)
    try:
        rotation_vector = angular_velocity * delta_time.unsqueeze(-1)
    except RuntimeError as error:
        raise ValueError("dt is not broadcastable to quaternion batch axes") from error
    delta = quaternion_exp(0.5 * rotation_vector)
    return normalize_quaternion(quaternion_multiply(orientation, delta))


def quaternion_geodesic_distance(
    first: Tensor,
    second: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """Shortest SO(3) geodesic angle in radians."""

    first = normalize_quaternion(first)
    second = normalize_quaternion(second)
    dot = (first * second).sum(dim=-1).abs().clamp(max=1.0)
    # ``atan2`` has better small-angle behaviour than acos(dot).
    sin_half = torch.sqrt((1.0 - dot.square()).clamp_min(0.0))
    return 2.0 * torch.atan2(sin_half, dot.clamp_min(eps))


def geodesic_orientation_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    distance_sq = quaternion_geodesic_distance(predicted, target).square()
    if reduction == "none":
        return distance_sq
    if reduction == "sum":
        return distance_sq.sum()
    if reduction == "mean":
        return distance_sq.mean()
    raise ValueError(f"unsupported reduction: {reduction}")
