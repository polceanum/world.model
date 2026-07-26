"""Timestamp-aware analytic free-motion integration."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from world_model.belief import MotionMode, ObjectBeliefTensor
from world_model.dynamics.quaternion import integrate_quaternion


def _object_dt(dt: float | Tensor, reference: Tensor) -> Tensor:
    """Return elapsed seconds broadcastable as ``[B,1,1]``."""

    value = torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.shape != (reference.shape[0],):
        raise ValueError(f"dt must be scalar or [B], got {tuple(value.shape)}")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError("dt must contain finite nonnegative seconds")
    return value[:, None, None]


class AnalyticKinematics(nn.Module):
    """Closed-form constant-gravity, linear-drag rigid-body integration."""

    def __init__(
        self,
        *,
        small_drag: float = 1e-5,
        min_drag: float = 1e-8,
        max_drag: float = 100.0,
    ) -> None:
        super().__init__()
        self.small_drag = small_drag
        self.min_drag = min_drag
        self.max_drag = max_drag

    def forward(
        self,
        objects: ObjectBeliefTensor,
        gravity: Tensor,
        dt: float | Tensor,
        *,
        residual_acceleration: Tensor | None = None,
        external_acceleration: Tensor | None = None,
    ) -> ObjectBeliefTensor:
        return self.integrate(
            objects,
            gravity,
            dt,
            residual_acceleration=residual_acceleration,
            external_acceleration=external_acceleration,
        )

    def integrate(
        self,
        objects: ObjectBeliefTensor,
        gravity: Tensor,
        dt: float | Tensor,
        *,
        residual_acceleration: Tensor | None = None,
        external_acceleration: Tensor | None = None,
    ) -> ObjectBeliefTensor:
        """Return integrated means without mutating ``objects``.

        Gravity and supplied residuals are accelerations in the world frame.
        Linear drag uses the exact constant-force solution with a safe
        zero-drag branch.
        """

        if gravity.shape != (objects.batch_size, 3):
            raise ValueError("gravity must have shape [B,3]")
        delta_time = _object_dt(dt, objects.position)
        acceleration = gravity[:, None, :].expand_as(objects.position)
        for name, value in (
            ("residual_acceleration", residual_acceleration),
            ("external_acceleration", external_acceleration),
        ):
            if value is not None:
                if value.shape != objects.position.shape:
                    raise ValueError(f"{name} must have shape [B,N,3]")
                acceleration = acceleration + value

        mode = objects.mode
        movable = (
            objects.active & (mode != int(MotionMode.SLEEPING)) & (mode != int(MotionMode.REMOVED))
        )
        movable_f = movable.unsqueeze(-1)
        acceleration = torch.where(movable_f, acceleration, torch.zeros_like(acceleration))

        drag = objects.drag.clamp(min=self.min_drag, max=self.max_drag)
        decay = torch.exp(-drag * delta_time)
        one_minus_decay = -torch.expm1(-drag * delta_time)
        safe_drag = drag.clamp_min(self.small_drag)

        velocity_drag = objects.velocity * decay + acceleration * one_minus_decay / safe_drag
        position_drag = (
            objects.position
            + objects.velocity * one_minus_decay / safe_drag
            + acceleration * (delta_time / safe_drag - one_minus_decay / safe_drag.square())
        )
        velocity_zero_drag = objects.velocity + acceleration * delta_time
        position_zero_drag = (
            objects.position
            + objects.velocity * delta_time
            + 0.5 * acceleration * delta_time.square()
        )
        use_drag = drag >= self.small_drag
        velocity = torch.where(use_drag, velocity_drag, velocity_zero_drag)
        position = torch.where(use_drag, position_drag, position_zero_drag)
        velocity = torch.where(movable_f, velocity, objects.velocity)
        position = torch.where(movable_f, position, objects.position)

        orientation = integrate_quaternion(
            objects.orientation,
            objects.angular_velocity,
            delta_time.squeeze(-1),
        )
        orientation = torch.where(
            movable_f.expand_as(orientation),
            orientation,
            objects.orientation,
        )
        return replace(
            objects,
            position=position,
            velocity=velocity,
            orientation=orientation,
        )
