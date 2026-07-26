"""Belief corruptions used to train and evaluate online recovery."""

from __future__ import annotations

from dataclasses import replace

import torch

from world_model.belief.world_belief import WorldBelief


def perturb_belief(
    belief: WorldBelief,
    *,
    position_std: float = 0.0,
    velocity_std: float = 0.0,
    covariance_log_bias: float = 0.0,
    generator: torch.Generator | None = None,
) -> WorldBelief:
    """Return a perturbed copy; never mutate the source belief."""

    objects = belief.objects
    active = objects.active.unsqueeze(-1)
    position_noise = torch.randn(
        objects.position.shape,
        device=objects.position.device,
        dtype=objects.position.dtype,
        generator=generator,
    )
    velocity_noise = torch.randn(
        objects.velocity.shape,
        device=objects.velocity.device,
        dtype=objects.velocity.dtype,
        generator=generator,
    )
    position = objects.position + active * position_std * position_noise
    velocity = objects.velocity + active * velocity_std * velocity_noise
    fast_log_variance = objects.fast_log_variance + active * covariance_log_bias
    perturbed_objects = replace(
        objects,
        position=position,
        velocity=velocity,
        fast_log_variance=fast_log_variance,
    )
    return replace(belief, objects=perturbed_objects)
