"""Dense, permutation-equivariant interaction graph for small object sets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from world_model.belief import ObjectBeliefTensor


@dataclass
class InteractionOutput:
    """Structured learned interaction outputs.

    ``pair_force[:, i, j]`` is force applied to object ``i`` by object ``j``;
    it is explicitly antisymmetric.  Pair scalar outputs are symmetric.
    """

    residual_acceleration: Tensor
    pair_acceleration: Tensor
    node_acceleration: Tensor
    pair_force: Tensor
    contact_logits: Tensor
    collision_logits: Tensor
    impulse_multiplier_raw: Tensor
    impulse_additive_raw: Tensor
    edge_process_noise: Tensor
    edge_mask: Tensor
    interaction_density: Tensor


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @property
    def output(self) -> nn.Linear:
        layer = self.layers[-1]
        assert isinstance(layer, nn.Linear)
        return layer

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class InteractionGraph(nn.Module):
    """Small learned pair/node residual with conservation-biased pair forces."""

    edge_feature_dim = 13
    edge_output_dim = 7

    def __init__(
        self,
        residual_dynamics_dim: int,
        global_code_dim: int,
        *,
        hidden_dim: int = 64,
        interaction_radius: float = 0.5,
        uncertainty_margin_scale: float = 2.0,
        max_pair_force: float = 2.0,
        max_node_acceleration: float = 2.0,
    ) -> None:
        super().__init__()
        if residual_dynamics_dim < 0 or global_code_dim < 0:
            raise ValueError("graph code dimensions must be nonnegative")
        if hidden_dim <= 0 or interaction_radius <= 0:
            raise ValueError("hidden_dim and interaction_radius must be positive")
        self.residual_dynamics_dim = residual_dynamics_dim
        self.global_code_dim = global_code_dim
        self.interaction_radius = interaction_radius
        self.uncertainty_margin_scale = uncertainty_margin_scale
        self.max_pair_force = max_pair_force
        self.max_node_acceleration = max_node_acceleration
        self.edge_network = _MLP(
            self.edge_feature_dim,
            hidden_dim,
            self.edge_output_dim,
        )
        node_input_dim = (
            3  # velocity
            + 3  # modal acceleration
            + 3  # aggregate pair acceleration
            + residual_dynamics_dim
            + global_code_dim
            + 2  # mean state uncertainty and interaction density
        )
        self.node_network = _MLP(node_input_dim, hidden_dim, 3)
        # A new model starts as analytic physics.  These are still real,
        # trainable residual networks and immediately receive gradients.
        nn.init.zeros_(self.edge_network.output.weight)
        nn.init.zeros_(self.edge_network.output.bias)
        nn.init.zeros_(self.node_network.output.weight)
        nn.init.zeros_(self.node_network.output.bias)

    def forward(
        self,
        objects: ObjectBeliefTensor,
        global_code: Tensor | None = None,
        *,
        modal_acceleration: Tensor | None = None,
    ) -> InteractionOutput:
        batch, count = objects.active.shape
        if objects.residual_dynamics_dim != self.residual_dynamics_dim:
            raise ValueError("Object residual dynamics dimension does not match InteractionGraph")
        if global_code is None:
            global_code = objects.position.new_zeros(batch, self.global_code_dim)
        if global_code.shape != (batch, self.global_code_dim):
            raise ValueError("global_code does not match configured graph dimension")
        if modal_acceleration is None:
            modal_acceleration = torch.zeros_like(objects.position)
        if modal_acceleration.shape != objects.position.shape:
            raise ValueError("modal_acceleration must have shape [B,N,3]")

        rel_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]  # j - i
        rel_velocity = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]
        distance = torch.linalg.vector_norm(rel_position, dim=-1).clamp_min(1e-7)
        normal = rel_position / distance.unsqueeze(-1)
        relative_normal_velocity = (rel_velocity * normal).sum(dim=-1)
        tangential_velocity = rel_velocity - relative_normal_velocity.unsqueeze(-1) * normal
        tangential_speed = torch.linalg.vector_norm(tangential_velocity, dim=-1)
        relative_speed = torch.linalg.vector_norm(rel_velocity, dim=-1)

        radius = objects.radius.squeeze(-1)
        radius_sum = radius[:, :, None] + radius[:, None, :]
        radius_difference = (radius[:, :, None] - radius[:, None, :]).abs()
        inverse_mass = objects.mass.squeeze(-1).reciprocal()
        inverse_mass_sum = inverse_mass[:, :, None] + inverse_mass[:, None, :]
        inverse_mass_difference = (inverse_mass[:, :, None] - inverse_mass[:, None, :]).abs()
        pair_restitution = torch.minimum(
            objects.restitution.squeeze(-1)[:, :, None],
            objects.restitution.squeeze(-1)[:, None, :],
        )
        friction_mean = 0.5 * (
            objects.friction.squeeze(-1)[:, :, None] + objects.friction.squeeze(-1)[:, None, :]
        )
        position_variance = objects.fast_log_variance[..., :3].exp().mean(dim=-1)
        pair_position_variance = position_variance[:, :, None] + position_variance[:, None, :]
        position_std = pair_position_variance.clamp_min(0.0).sqrt()
        gap = distance - radius_sum
        features = torch.stack(
            (
                distance,
                gap,
                relative_normal_velocity,
                tangential_speed,
                relative_speed,
                radius_sum,
                radius_difference,
                inverse_mass_sum,
                inverse_mass_difference,
                pair_restitution,
                friction_mean,
                pair_position_variance,
                position_std,
            ),
            dim=-1,
        )

        active_pair = objects.active[:, :, None] & objects.active[:, None, :]
        identity = torch.eye(count, device=objects.active.device, dtype=torch.bool)
        candidate = distance <= (
            radius_sum + self.interaction_radius + self.uncertainty_margin_scale * position_std
        )
        upper_mask = (
            active_pair
            & ~identity.unsqueeze(0)
            & candidate
            & torch.triu(
                torch.ones(count, count, device=objects.active.device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0)
        )
        edge_values = self.edge_network(features)
        edge_values = edge_values * upper_mask.unsqueeze(-1)
        contact_upper = edge_values[..., 0]
        collision_upper = edge_values[..., 1]
        normal_force = self.max_pair_force * torch.tanh(edge_values[..., 2]) * upper_mask
        tangent_force = self.max_pair_force * torch.tanh(edge_values[..., 3]) * upper_mask
        tangent_direction = tangential_velocity / tangential_speed.clamp_min(1e-7).unsqueeze(-1)
        # Momentum/force on i is opposite the i->j normal.  Tangential force
        # opposes relative motion of j with respect to i.
        force_on_i_upper = (
            -normal_force.unsqueeze(-1) * normal + tangent_force.unsqueeze(-1) * tangent_direction
        )
        pair_force = force_on_i_upper - force_on_i_upper.transpose(1, 2)
        net_force = pair_force.sum(dim=2)
        pair_acceleration = net_force * inverse_mass.unsqueeze(-1)

        symmetric = lambda value: value + value.transpose(1, 2)  # noqa: E731
        contact_logits = symmetric(contact_upper)
        collision_logits = symmetric(collision_upper)
        impulse_multiplier_raw = symmetric(edge_values[..., 4])
        impulse_additive_raw = symmetric(edge_values[..., 5])
        edge_mask = upper_mask | upper_mask.transpose(1, 2)
        # Zero network output must preserve the analytic uncertainty baseline.
        # Represent this as a smooth signed residual around softplus(0), rather
        # than an always-positive softplus whose zero logit would inject 0.693
        # units of untrained noise per edge.
        edge_process_noise = (
            torch.nn.functional.softplus(symmetric(edge_values[..., 6])) - math.log(2.0)
        ) * edge_mask
        interaction_density = edge_mask.sum(dim=-1).to(objects.position.dtype)

        mean_uncertainty = objects.fast_log_variance.exp().mean(dim=-1)
        node_input = torch.cat(
            (
                objects.velocity,
                modal_acceleration,
                pair_acceleration,
                objects.residual_dynamics,
                global_code[:, None, :].expand(-1, count, -1),
                mean_uncertainty.unsqueeze(-1),
                interaction_density.unsqueeze(-1),
            ),
            dim=-1,
        )
        node_acceleration = self.max_node_acceleration * torch.tanh(self.node_network(node_input))
        node_acceleration = node_acceleration * objects.active.unsqueeze(-1)
        residual_acceleration = (pair_acceleration + node_acceleration) * (
            objects.active.unsqueeze(-1)
        )
        return InteractionOutput(
            residual_acceleration=residual_acceleration,
            pair_acceleration=pair_acceleration,
            node_acceleration=node_acceleration,
            pair_force=pair_force,
            contact_logits=contact_logits,
            collision_logits=collision_logits,
            impulse_multiplier_raw=impulse_multiplier_raw,
            impulse_additive_raw=impulse_additive_raw,
            edge_process_noise=edge_process_noise,
            edge_mask=edge_mask,
            interaction_density=interaction_density,
        )
