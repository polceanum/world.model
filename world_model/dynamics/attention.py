"""Typed attention residuals for structured object interactions.

The attention stack consumes a derived set of scene, entity, and relation
tokens.  It never owns persistent state: outputs are decoded immediately into
the same structured interaction contract used by the analytic dynamics and
event resolver.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from world_model.belief import ObjectBeliefTensor
from world_model.dynamics.graph import InteractionOutput


class _SwiGLU(nn.Module):
    """Position-wise gated feed-forward network used by the pilot blocks."""

    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.value = nn.Linear(width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.output(torch.nn.functional.silu(self.gate(value)) * self.value(value))


class _PreNormAttentionBlock(nn.Module):
    """RMS-pre-normalized self-attention plus a SwiGLU residual branch."""

    def __init__(
        self,
        width: int,
        heads: int,
        feed_forward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.RMSNorm(width)
        self.feed_forward = _SwiGLU(width, feed_forward_width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        return tokens + self.dropout(self.feed_forward(self.feed_forward_norm(tokens)))


@dataclass(frozen=True)
class AttentionTokenLayout:
    """Offsets for the reversible current-belief token set."""

    entity_start: int
    entity_count: int
    relation_start: int
    relation_count: int

    @property
    def sequence_length(self) -> int:
        return self.relation_start + self.relation_count


class TypedAttentionInteractionResidual(nn.Module):
    """Optional attention residual around the proven interaction graph.

    There are no learned slot-position embeddings.  Entity and relation tokens
    are sets, so permuting padded object slots permutes outputs rather than
    changing their meaning.  The output decoders start at exact zero, making a
    newly enabled pilot numerically identical to its graph-only initialization
    checkpoint before optimization.
    """

    relation_feature_dim = 13
    relation_output_dim = 7

    def __init__(
        self,
        *,
        modal_count: int,
        modal_dim: int,
        geometry_dim: int,
        appearance_dim: int,
        residual_dynamics_dim: int,
        parameter_memory_dim: int,
        motion_mode_dim: int,
        global_code_dim: int,
        width: int = 128,
        heads: int = 4,
        layers: int = 4,
        feed_forward_width: int = 512,
        dropout: float = 0.0,
        max_pair_force: float = 0.5,
        max_node_acceleration: float = 0.5,
        max_event_logit_residual: float = 2.0,
        max_process_noise_residual: float = 0.25,
    ) -> None:
        super().__init__()
        fast_state_dim = 13 + modal_count * 2 * modal_dim
        slow_state_dim = 4 + geometry_dim + appearance_dim + residual_dynamics_dim
        entity_feature_dim = (
            3  # position
            + 3  # velocity
            + 4  # orientation
            + 3  # angular velocity
            + geometry_dim
            + appearance_dim
            + residual_dynamics_dim
            + modal_count * 2 * modal_dim
            + modal_count * modal_dim  # modal frequency
            + modal_count * modal_dim  # modal decay
            + 4  # physical parameter logits
            + motion_mode_dim
            + 2  # existence and visibility
            + 2  # age and missed age
            + fast_state_dim
            + slow_state_dim
            + parameter_memory_dim
        )
        self.width = width
        self.max_pair_force = max_pair_force
        self.max_node_acceleration = max_node_acceleration
        self.max_event_logit_residual = max_event_logit_residual
        self.max_process_noise_residual = max_process_noise_residual
        self.scene_projection = nn.Linear(global_code_dim, width)
        self.entity_projection = nn.Linear(entity_feature_dim, width)
        self.relation_projection = nn.Linear(self.relation_feature_dim, width)
        self.type_embedding = nn.Embedding(3, width)
        self.blocks = nn.ModuleList(
            _PreNormAttentionBlock(
                width,
                heads,
                feed_forward_width,
                dropout,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.RMSNorm(width)
        self.node_decoder = nn.Linear(width, 3)
        self.relation_decoder = nn.Linear(width, self.relation_output_dim)
        nn.init.zeros_(self.node_decoder.weight)
        nn.init.zeros_(self.node_decoder.bias)
        nn.init.zeros_(self.relation_decoder.weight)
        nn.init.zeros_(self.relation_decoder.bias)

    @staticmethod
    def _entity_features(objects: ObjectBeliefTensor) -> Tensor:
        dtype = objects.position.dtype
        age = torch.log1p(objects.age_steps.to(dtype)).unsqueeze(-1)
        missed = torch.log1p(objects.missed_steps.to(dtype)).unsqueeze(-1)
        return torch.cat(
            (
                objects.position,
                objects.velocity,
                objects.orientation,
                objects.angular_velocity,
                objects.geometry,
                objects.appearance,
                objects.residual_dynamics,
                objects.modal_state.flatten(start_dim=-3),
                objects.modal_frequency.flatten(start_dim=-2),
                objects.modal_decay_raw.flatten(start_dim=-2),
                objects.log_mass,
                objects.restitution_logit,
                objects.log_drag,
                objects.friction_logit,
                objects.motion_mode_logits,
                objects.existence_logit.unsqueeze(-1),
                objects.visibility_logit.unsqueeze(-1),
                age,
                missed,
                objects.fast_log_variance,
                objects.slow_log_variance,
                objects.parameter_memory,
            ),
            dim=-1,
        )

    @staticmethod
    def _relation_features(objects: ObjectBeliefTensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rel_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]
        rel_velocity = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]
        distance = torch.linalg.vector_norm(rel_position, dim=-1).clamp_min(1.0e-7)
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
        tangent_direction = tangential_velocity / tangential_speed.clamp_min(1.0e-7).unsqueeze(-1)
        return features, normal, tangent_direction, inverse_mass

    @staticmethod
    def _layout(object_count: int) -> AttentionTokenLayout:
        relation_count = object_count * (object_count - 1) // 2
        return AttentionTokenLayout(
            entity_start=1,
            entity_count=object_count,
            relation_start=1 + object_count,
            relation_count=relation_count,
        )

    def forward(
        self,
        objects: ObjectBeliefTensor,
        global_code: Tensor,
        base: InteractionOutput,
    ) -> InteractionOutput:
        batch, count = objects.active.shape
        if global_code.shape[0] != batch:
            raise ValueError("global code batch does not match object belief")
        if base.edge_mask.shape != (batch, count, count):
            raise ValueError("base interaction edge mask has incompatible shape")

        pair_indices = torch.triu_indices(count, count, offset=1, device=objects.position.device)
        pair_i, pair_j = pair_indices[0], pair_indices[1]
        layout = self._layout(count)
        relation_features, normal, tangent_direction, inverse_mass = self._relation_features(
            objects
        )

        scene_tokens = self.scene_projection(global_code).unsqueeze(1)
        entity_tokens = self.entity_projection(self._entity_features(objects))
        selected_relation_features = relation_features[:, pair_i, pair_j]
        relation_tokens = self.relation_projection(selected_relation_features)
        tokens = torch.cat((scene_tokens, entity_tokens, relation_tokens), dim=1)
        token_types = torch.cat(
            (
                torch.zeros(1, device=tokens.device, dtype=torch.long),
                torch.ones(count, device=tokens.device, dtype=torch.long),
                torch.full(
                    (layout.relation_count,),
                    2,
                    device=tokens.device,
                    dtype=torch.long,
                ),
            )
        )
        tokens = tokens + self.type_embedding(token_types).unsqueeze(0)
        relation_valid = base.edge_mask[:, pair_i, pair_j]
        valid_mask = torch.cat(
            (
                torch.ones(batch, 1, device=tokens.device, dtype=torch.bool),
                objects.active,
                relation_valid,
            ),
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, valid_mask)
        tokens = self.output_norm(tokens)

        node_values = self.node_decoder(
            tokens[:, layout.entity_start : layout.entity_start + layout.entity_count]
        )
        node_residual = self.max_node_acceleration * torch.tanh(node_values)
        node_residual = node_residual * objects.active.unsqueeze(-1)
        relation_values = self.relation_decoder(tokens[:, layout.relation_start :])
        relation_values = relation_values * relation_valid.unsqueeze(-1)

        upper = objects.position.new_zeros(batch, count, count, self.relation_output_dim)
        upper[:, pair_i, pair_j] = relation_values
        contact_residual = self.max_event_logit_residual * torch.tanh(upper[..., 0])
        collision_residual = self.max_event_logit_residual * torch.tanh(upper[..., 1])
        normal_force = self.max_pair_force * torch.tanh(upper[..., 2])
        tangent_force = self.max_pair_force * torch.tanh(upper[..., 3])
        force_on_i_upper = (
            -normal_force.unsqueeze(-1) * normal + tangent_force.unsqueeze(-1) * tangent_direction
        )
        pair_force_residual = force_on_i_upper - force_on_i_upper.transpose(1, 2)
        pair_force = base.pair_force + pair_force_residual
        pair_acceleration = base.pair_acceleration + pair_force_residual.sum(dim=2) * (
            inverse_mass.unsqueeze(-1)
        )
        node_acceleration = base.node_acceleration + node_residual
        residual_acceleration = (pair_acceleration + node_acceleration) * (
            objects.active.unsqueeze(-1)
        )

        symmetric = lambda value: value + value.transpose(1, 2)  # noqa: E731
        return InteractionOutput(
            residual_acceleration=residual_acceleration,
            pair_acceleration=pair_acceleration,
            node_acceleration=node_acceleration,
            pair_force=pair_force,
            contact_logits=base.contact_logits + symmetric(contact_residual),
            collision_logits=base.collision_logits + symmetric(collision_residual),
            impulse_multiplier_raw=base.impulse_multiplier_raw + symmetric(upper[..., 4]),
            impulse_additive_raw=base.impulse_additive_raw + symmetric(upper[..., 5]),
            edge_process_noise=base.edge_process_noise
            + self.max_process_noise_residual * torch.tanh(symmetric(upper[..., 6])),
            edge_mask=base.edge_mask,
            interaction_density=base.interaction_density,
        )
