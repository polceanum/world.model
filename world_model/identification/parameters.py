"""Bounded physical-parameter helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ParameterBounds:
    mass: tuple[float, float] = (0.05, 20.0)
    restitution: tuple[float, float] = (0.02, 0.98)
    drag: tuple[float, float] = (1.0e-4, 5.0)
    friction: tuple[float, float] = (0.01, 0.99)
    radius: tuple[float, float] = (0.02, 1.0)

    def __post_init__(self) -> None:
        for name in ("mass", "restitution", "drag", "friction", "radius"):
            lower, upper = getattr(self, name)
            if not 0 < lower < upper:
                raise ValueError(f"invalid {name} bounds")
        for name in ("restitution", "friction"):
            _, upper = getattr(self, name)
            if upper >= 1:
                raise ValueError(f"{name} upper bound must be below one")


def logit_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = bounds
    return math.log(lower / (1.0 - lower)), math.log(upper / (1.0 - upper))


def project_parameter_tensors(
    *,
    log_mass: Tensor,
    restitution_logit: Tensor,
    log_drag: Tensor,
    friction_logit: Tensor,
    radius: Tensor,
    bounds: ParameterBounds,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    restitution_bounds = logit_bounds(bounds.restitution)
    friction_bounds = logit_bounds(bounds.friction)
    return (
        log_mass.clamp(math.log(bounds.mass[0]), math.log(bounds.mass[1])),
        restitution_logit.clamp(*restitution_bounds),
        log_drag.clamp(math.log(bounds.drag[0]), math.log(bounds.drag[1])),
        friction_logit.clamp(*friction_bounds),
        radius.clamp(*bounds.radius),
    )


def physical_parameter_vector(
    log_mass: Tensor,
    restitution_logit: Tensor,
    log_drag: Tensor,
    friction_logit: Tensor,
    radius: Tensor,
) -> Tensor:
    return torch.cat(
        (
            log_mass.exp(),
            restitution_logit.sigmoid(),
            log_drag.exp(),
            friction_logit.sigmoid(),
            radius,
        ),
        dim=-1,
    )
