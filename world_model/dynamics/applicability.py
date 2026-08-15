"""Causal smooth applicability for learned pair and event residuals.

The analytic integrator and contact resolver remain authoritative.  This
module only scales learned residual proposals using geometry, relative motion,
and uncertainty already present in ``WorldBelief``.  It owns no parameters or
persistent state, so enabling it is an explicit runtime protocol change rather
than an alternate source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from world_model.belief import ObjectBeliefTensor
from world_model.dynamics.graph import InteractionOutput

_LOGISTIC_NORMAL_CDF_SCALE = 1.702


@dataclass(frozen=True)
class PairApplicability:
    """Symmetric probabilities for pair/contact and collision applicability."""

    pair: Tensor
    collision: Tensor

    def validate(self, objects: ObjectBeliefTensor) -> PairApplicability:
        expected = (*objects.active.shape, objects.max_objects)
        if self.pair.shape != expected or self.collision.shape != expected:
            raise ValueError("pair applicability tensors must have shape [B,N,N]")
        for name, value in (("pair", self.pair), ("collision", self.collision)):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} applicability contains NaN or Inf")
            if torch.any(value < 0.0) or torch.any(value > 1.0):
                raise ValueError(f"{name} applicability must lie in [0,1]")
        return self


@dataclass(frozen=True)
class PairApplicabilityConfig:
    """Parameter-free physical scales for the smooth applicability envelope."""

    enabled: bool = False
    lookahead_seconds: float = 0.05
    margin_m: float = 0.05
    gap_temperature_m: float = 0.025
    velocity_temperature_mps: float = 0.10
    collision_speed_epsilon: float = 1.0e-7

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        for name, value in (
            ("lookahead_seconds", self.lookahead_seconds),
            ("margin_m", self.margin_m),
            ("collision_speed_epsilon", self.collision_speed_epsilon),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("gap_temperature_m", self.gap_temperature_m),
            ("velocity_temperature_mps", self.velocity_temperature_mps),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def pair_applicability(
    objects: ObjectBeliefTensor,
    edge_mask: Tensor,
    config: PairApplicabilityConfig,
) -> PairApplicability:
    """Return current smooth pair/event applicability from causal belief state.

    The near/contact term approximates a Gaussian contact probability after a
    short constant-relative-velocity lookahead.  Position and velocity
    variances are projected onto the pair normal.  Closing motion is selected
    smoothly, while separating motion cannot create a spurious future contact.
    The collision term additionally requires probabilistic closing evidence.

    When disabled, every existing candidate edge receives an exact unit gate.
    This is the historical behavior and avoids any arithmetic on model outputs.
    """

    batch, count = objects.active.shape
    if edge_mask.shape != (batch, count, count) or edge_mask.dtype is not torch.bool:
        raise ValueError("edge_mask must be boolean with shape [B,N,N]")
    edge_weight = edge_mask.to(objects.position.dtype)
    if not config.enabled:
        return PairApplicability(pair=edge_weight, collision=edge_weight)

    rel_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]
    rel_velocity = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]
    distance = torch.linalg.vector_norm(rel_position, dim=-1).clamp_min(1.0e-7)
    normal = rel_position / distance.unsqueeze(-1)
    normal_velocity = (rel_velocity * normal).sum(dim=-1)
    radius = objects.radius.squeeze(-1)
    gap = distance - radius[:, :, None] - radius[:, None, :]

    position_variance = objects.fast_log_variance[..., :3].clamp(-30.0, 20.0).exp()
    velocity_variance = objects.fast_log_variance[..., 3:6].clamp(-30.0, 20.0).exp()
    relative_position_variance = position_variance[:, :, None, :] + position_variance[:, None, :, :]
    relative_velocity_variance = velocity_variance[:, :, None, :] + velocity_variance[:, None, :, :]
    gap_variance = (relative_position_variance * normal.square()).sum(dim=-1)
    normal_velocity_variance = (relative_velocity_variance * normal.square()).sum(dim=-1)

    # ``v * sigmoid(-v / temperature)`` is a smooth negative-part
    # approximation. It follows closing motion but tends to zero for clearly
    # separating pairs, avoiding an artificial attraction rule.
    closing_selected_velocity = normal_velocity * torch.sigmoid(
        -normal_velocity / config.velocity_temperature_mps
    )
    lookahead = objects.position.new_tensor(config.lookahead_seconds)
    projected_gap = gap + lookahead * closing_selected_velocity
    projected_gap_variance = gap_variance + lookahead.square() * normal_velocity_variance
    gap_scale = (
        (projected_gap_variance + config.gap_temperature_m * config.gap_temperature_m)
        .clamp_min(1.0e-12)
        .sqrt()
    )
    pair_logit = _LOGISTIC_NORMAL_CDF_SCALE * (config.margin_m - projected_gap) / gap_scale
    pair_gate = torch.sigmoid(pair_logit) * edge_weight

    velocity_scale = (
        (
            normal_velocity_variance
            + config.velocity_temperature_mps * config.velocity_temperature_mps
        )
        .clamp_min(1.0e-12)
        .sqrt()
    )
    collision_logit = (
        _LOGISTIC_NORMAL_CDF_SCALE
        * (-normal_velocity - config.collision_speed_epsilon)
        / velocity_scale
    )
    closing_gate = torch.sigmoid(collision_logit)
    collision_gate = pair_gate * closing_gate
    return PairApplicability(pair=pair_gate, collision=collision_gate)


def apply_pair_applicability(
    objects: ObjectBeliefTensor,
    interaction: InteractionOutput,
    config: PairApplicabilityConfig,
) -> tuple[InteractionOutput, PairApplicability]:
    """Scale learned pair/event effects and restore a coherent acceleration.

    Node acceleration is deliberately preserved.  Pair force is gated first,
    then pair and total residual acceleration are recomputed so conservation,
    mass scaling, and the typed ``InteractionOutput`` contract cannot diverge.
    The analytic contact resolver is downstream and remains fully active.
    """

    applicability = pair_applicability(objects, interaction.edge_mask, config)
    if not config.enabled:
        return interaction, applicability

    pair_force = interaction.pair_force * applicability.pair.unsqueeze(-1)
    pair_acceleration = pair_force.sum(dim=2) / objects.mass
    active = objects.active.unsqueeze(-1)
    node_acceleration = interaction.node_acceleration
    residual_acceleration = (pair_acceleration + node_acceleration) * active
    return (
        replace(
            interaction,
            residual_acceleration=residual_acceleration,
            pair_acceleration=pair_acceleration,
            pair_force=pair_force,
            contact_logits=interaction.contact_logits * applicability.pair,
            collision_logits=interaction.collision_logits * applicability.collision,
            impulse_multiplier_raw=(interaction.impulse_multiplier_raw * applicability.collision),
            impulse_additive_raw=(interaction.impulse_additive_raw * applicability.collision),
            edge_process_noise=interaction.edge_process_noise * applicability.pair,
        ),
        applicability,
    )
