"""Diagonal uncertainty propagation for hybrid dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from world_model.belief import ObjectBeliefTensor, clamp_log_variance


@dataclass
class UncertaintyOutput:
    objects: ObjectBeliefTensor
    process_variance: Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class UncertaintyDynamics(nn.Module):
    """Closed-form position/velocity propagation plus learned positive noise."""

    feature_dim = 8

    def __init__(
        self,
        fast_state_dim: int,
        *,
        hidden_dim: int = 32,
        base_process_variance_per_second: float = 1e-5,
        position_process_variance_per_second: float | None = None,
        velocity_process_variance_per_second: float | None = None,
        log_variance_bounds: tuple[float, float] = (-20.0, 10.0),
        max_process_variance_per_step: float = 1.0,
    ) -> None:
        super().__init__()
        if fast_state_dim < 6:
            raise ValueError("fast_state_dim must contain position and velocity")
        position_noise = (
            base_process_variance_per_second
            if position_process_variance_per_second is None
            else position_process_variance_per_second
        )
        velocity_noise = (
            base_process_variance_per_second
            if velocity_process_variance_per_second is None
            else velocity_process_variance_per_second
        )
        if min(base_process_variance_per_second, position_noise, velocity_noise) <= 0:
            raise ValueError("base process variances must be positive")
        self.fast_state_dim = fast_state_dim
        self.log_variance_bounds = log_variance_bounds
        self.max_process_variance_per_step = max_process_variance_per_step
        self.process_network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, fast_state_dim),
        )
        output = self.process_network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        baseline = torch.full(
            (fast_state_dim,),
            base_process_variance_per_second,
            dtype=output.bias.dtype,
        )
        baseline[:3] = position_noise
        baseline[3:6] = velocity_noise
        with torch.no_grad():
            output.bias.copy_(
                torch.tensor(
                    [_inverse_softplus(float(item)) for item in baseline],
                    dtype=output.bias.dtype,
                )
            )

    def forward(
        self,
        objects: ObjectBeliefTensor,
        dt: float | Tensor,
        *,
        event_logits: Tensor | None = None,
        interaction_density: Tensor | None = None,
        residual_acceleration: Tensor | None = None,
    ) -> UncertaintyOutput:
        if objects.fast_state_dim != self.fast_state_dim:
            raise ValueError("object fast state does not match uncertainty model")
        delta_time = torch.as_tensor(
            dt,
            device=objects.position.device,
            dtype=objects.position.dtype,
        )
        if delta_time.ndim == 0:
            delta_time = delta_time.expand(objects.batch_size)
        if delta_time.shape != (objects.batch_size,):
            raise ValueError("uncertainty dt must be scalar or [B]")
        if not torch.isfinite(delta_time).all() or torch.any(delta_time < 0):
            raise ValueError("uncertainty dt must be finite and nonnegative")
        dt_object = delta_time[:, None]

        speed = torch.linalg.vector_norm(objects.velocity, dim=-1)
        occlusion = 1.0 - objects.visibility_logit.sigmoid()
        missed = objects.missed_steps.to(objects.position.dtype)
        missed = missed / (1.0 + missed)
        if event_logits is None:
            event_probability = torch.zeros_like(speed)
            event_entropy = torch.zeros_like(speed)
        else:
            probability = torch.softmax(event_logits, dim=-1)
            event_probability = 1.0 - probability[..., 0]
            event_entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
        if interaction_density is None:
            interaction_density = torch.zeros_like(speed)
        interaction_density = interaction_density / (1.0 + interaction_density)
        if residual_acceleration is None:
            residual_magnitude = torch.zeros_like(speed)
        else:
            residual_magnitude = torch.linalg.vector_norm(residual_acceleration, dim=-1)
        mean_uncertainty = objects.fast_log_variance.exp().mean(dim=-1)
        features = torch.stack(
            (
                dt_object.expand_as(speed),
                speed,
                event_probability,
                event_entropy,
                occlusion,
                interaction_density,
                residual_magnitude + missed,
                mean_uncertainty,
            ),
            dim=-1,
        )
        process_variance = F.softplus(self.process_network(features))
        process_variance = (process_variance * dt_object.unsqueeze(-1)).clamp(
            max=self.max_process_variance_per_step
        )
        process_variance = process_variance * objects.active.unsqueeze(-1)

        variance = objects.fast_log_variance.exp()
        updated_variance = variance + process_variance
        dt_squared = dt_object.square().unsqueeze(-1)
        updated_variance = torch.cat(
            (
                variance[..., :3] + dt_squared * variance[..., 3:6] + process_variance[..., :3],
                variance[..., 3:6] + process_variance[..., 3:6],
                updated_variance[..., 6:],
            ),
            dim=-1,
        )
        log_variance = clamp_log_variance(
            updated_variance.clamp_min(1e-12).log(),
            self.log_variance_bounds,
        )
        log_variance = torch.where(
            objects.active.unsqueeze(-1),
            log_variance,
            objects.fast_log_variance,
        )
        updated = replace(objects, fast_log_variance=log_variance)
        return UncertaintyOutput(
            objects=updated,
            process_variance=process_variance,
        )
