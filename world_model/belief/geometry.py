"""Backward-compatible geometry encoding for extensible rigid object state."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor


class RigidPrimitive(IntEnum):
    """Rigid geometry tags stored in optional geometry component one."""

    SPHERE = 0
    BOX = 1


class RigidGeometryCodec:
    """Encode primitive geometry while preserving legacy component zero.

    ``geometry[..., 0]`` remains the conservative bounding radius used by all
    legacy sphere checkpoints.  When capacity exists, component one stores an
    exact primitive tag and components two through four store box half-extents.
    A one-component checkpoint therefore continues to decode as a sphere.

    This is a representation seam, not a claim that boxes are already
    perceptually or dynamically qualified.
    """

    MINIMUM_BOX_DIMENSION = 5

    @staticmethod
    def bounding_radius(geometry: Tensor) -> Tensor:
        RigidGeometryCodec._validate_geometry(geometry)
        return geometry[..., :1].clamp_min(1.0e-6)

    @staticmethod
    def primitive(geometry: Tensor) -> Tensor:
        RigidGeometryCodec._validate_geometry(geometry)
        if geometry.shape[-1] < 2:
            return torch.zeros(
                geometry.shape[:-1],
                device=geometry.device,
                dtype=torch.int64,
            )
        tag = geometry[..., 1]
        rounded = tag.round()
        if torch.any((tag - rounded).abs() > 1.0e-6):
            raise ValueError("geometry primitive tag must be integer-valued")
        primitive = rounded.to(torch.int64)
        if torch.any(
            (primitive != int(RigidPrimitive.SPHERE)) & (primitive != int(RigidPrimitive.BOX))
        ):
            raise ValueError("geometry contains an unsupported rigid primitive tag")
        return primitive

    @staticmethod
    def half_extents(geometry: Tensor) -> Tensor:
        """Return local half-extents for sphere and box rows."""

        primitive = RigidGeometryCodec.primitive(geometry)
        radius = RigidGeometryCodec.bounding_radius(geometry).expand(
            *geometry.shape[:-1],
            3,
        )
        if not bool((primitive == int(RigidPrimitive.BOX)).any()):
            return radius
        if geometry.shape[-1] < RigidGeometryCodec.MINIMUM_BOX_DIMENSION:
            raise ValueError("box geometry requires at least five components")
        box = geometry[..., 2:5]
        if torch.any((primitive == int(RigidPrimitive.BOX)).unsqueeze(-1) & (box <= 0.0)):
            raise ValueError("box half-extents must be positive")
        return torch.where(
            (primitive == int(RigidPrimitive.BOX)).unsqueeze(-1),
            box,
            radius,
        )

    @staticmethod
    def encode_sphere(radius: Tensor, *, geometry_dim: int) -> Tensor:
        RigidGeometryCodec._validate_radius(radius)
        if isinstance(geometry_dim, bool) or not isinstance(geometry_dim, int) or geometry_dim < 1:
            raise ValueError("geometry_dim must be a positive integer")
        output = radius.new_zeros(*radius.shape[:-1], geometry_dim)
        output[..., 0] = radius[..., 0]
        return output

    @staticmethod
    def encode_box(half_extents: Tensor, *, geometry_dim: int) -> Tensor:
        if (
            not isinstance(half_extents, Tensor)
            or half_extents.ndim < 1
            or half_extents.shape[-1] != 3
            or not half_extents.is_floating_point()
        ):
            raise TypeError("box half_extents must be a floating tensor [...,3]")
        if not torch.isfinite(half_extents).all() or torch.any(half_extents <= 0.0):
            raise ValueError("box half_extents must be finite and positive")
        if geometry_dim < RigidGeometryCodec.MINIMUM_BOX_DIMENSION:
            raise ValueError("box geometry requires geometry_dim >= 5")
        output = half_extents.new_zeros(*half_extents.shape[:-1], geometry_dim)
        output[..., 0] = torch.linalg.vector_norm(half_extents, dim=-1)
        output[..., 1] = float(RigidPrimitive.BOX)
        output[..., 2:5] = half_extents
        return output

    @staticmethod
    def _validate_geometry(geometry: Tensor) -> None:
        if (
            not isinstance(geometry, Tensor)
            or geometry.ndim < 1
            or geometry.shape[-1] < 1
            or not geometry.is_floating_point()
        ):
            raise TypeError("geometry must be a floating tensor [...,D] with D >= 1")
        if not torch.isfinite(geometry).all():
            raise ValueError("geometry must be finite")

    @staticmethod
    def _validate_radius(radius: Tensor) -> None:
        if (
            not isinstance(radius, Tensor)
            or radius.ndim < 1
            or radius.shape[-1] != 1
            or not radius.is_floating_point()
        ):
            raise TypeError("sphere radius must be a floating tensor [...,1]")
        if not torch.isfinite(radius).all() or torch.any(radius <= 0.0):
            raise ValueError("sphere radius must be finite and positive")


__all__ = ["RigidGeometryCodec", "RigidPrimitive"]
