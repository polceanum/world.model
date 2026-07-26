"""Global learned object-proposal head for synthetic RGB scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class GlobalDetectorOutput:
    centre: Tensor
    log_radius: Tensor
    inverse_depth_residual: Tensor
    colour: Tensor
    existence_logits: Tensor
    visibility_logits: Tensor
    log_variance: Tensor
    appearance: Tensor
    query_features: Tensor
    attention: Tensor


def _anchor_grid(query_count: int) -> Tensor:
    side = int(math.ceil(math.sqrt(query_count)))
    axis = torch.linspace(-0.75, 0.75, side)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)[:query_count]


class GlobalObjectDetector(nn.Module):
    """A tiny DETR-like proposal head implemented with ``nn.MultiheadAttention``."""

    def __init__(
        self,
        *,
        feature_dim: int,
        query_count: int,
        appearance_dim: int,
        attention_heads: int = 4,
        attention_layers: int = 2,
        minimum_radius: float = 0.015,
        maximum_radius: float = 0.75,
    ) -> None:
        super().__init__()
        if query_count <= 0:
            raise ValueError("query_count must be positive")
        if feature_dim % attention_heads != 0:
            raise ValueError("feature_dim must be divisible by attention_heads")
        self.query_count = query_count
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.queries = nn.Parameter(torch.randn(query_count, feature_dim) * 0.02)
        self.coordinate_projection = nn.Linear(2, feature_dim)
        self.attention_layers = nn.ModuleList(
            nn.MultiheadAttention(
                feature_dim,
                attention_heads,
                batch_first=True,
            )
            for _ in range(attention_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(feature_dim) for _ in range(attention_layers))
        self.feed_forwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.SiLU(),
                nn.Linear(feature_dim * 2, feature_dim),
            )
            for _ in range(attention_layers)
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.centre_head = nn.Linear(feature_dim, 2)
        self.radius_head = nn.Linear(feature_dim, 1)
        self.depth_head = nn.Linear(feature_dim, 1)
        self.colour_head = nn.Linear(feature_dim, 3)
        self.existence_head = nn.Linear(feature_dim, 1)
        self.visibility_head = nn.Linear(feature_dim, 1)
        self.variance_head = nn.Linear(feature_dim, 7)
        self.appearance_head = nn.Linear(feature_dim, appearance_dim)
        self.register_buffer("query_anchors", _anchor_grid(query_count))
        # A mildly positive prior makes an untrained smoke model initialise
        # tentative slots; supervised existence learning quickly overrides it.
        nn.init.constant_(self.existence_head.bias, 0.5)
        nn.init.constant_(self.visibility_head.bias, 0.5)
        nn.init.constant_(self.variance_head.bias, -2.0)

    def _memory_with_position(self, feature_map: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = feature_map.shape
        y_axis = torch.linspace(
            -1.0,
            1.0,
            height,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        x_axis = torch.linspace(
            -1.0,
            1.0,
            width,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        coordinates = coordinates.expand(batch, -1, -1)
        memory = feature_map.flatten(2).transpose(1, 2)
        memory = memory + self.coordinate_projection(coordinates)
        if memory.shape[-1] != channels:
            raise RuntimeError("unexpected detector memory channel mismatch")
        return memory, coordinates

    def forward(self, feature_map: Tensor) -> GlobalDetectorOutput:
        if feature_map.ndim != 4:
            raise ValueError("global detector feature map must be [B,C,H,W]")
        batch = feature_map.shape[0]
        memory, coordinates = self._memory_with_position(feature_map)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        final_attention = feature_map.new_zeros((batch, self.query_count, memory.shape[1]))
        for attention, norm, feed_forward in zip(
            self.attention_layers,
            self.norms,
            self.feed_forwards,
            strict=True,
        ):
            attended, weights = attention(
                queries,
                memory,
                memory,
                need_weights=True,
                average_attn_weights=True,
            )
            queries = norm(queries + attended)
            queries = norm(queries + feed_forward(queries))
            final_attention = weights
        queries = self.output_norm(queries)
        anchor = self.query_anchors.to(dtype=queries.dtype)
        learned_centre = torch.tanh(anchor.unsqueeze(0) + 0.5 * self.centre_head(queries))
        # Attention centroid gives an observation-dependent geometric route in
        # addition to the learned query regression.
        attention_centre = torch.einsum("bql,bld->bqd", final_attention, coordinates)
        centre = (0.75 * learned_centre + 0.25 * attention_centre).clamp(-1.0, 1.0)
        radius = self.minimum_radius + (self.maximum_radius - self.minimum_radius) * torch.sigmoid(
            self.radius_head(queries)
        )
        appearance = F.normalize(self.appearance_head(queries), dim=-1)
        return GlobalDetectorOutput(
            centre=centre,
            log_radius=radius.log(),
            inverse_depth_residual=0.25 * torch.tanh(self.depth_head(queries)),
            colour=torch.sigmoid(self.colour_head(queries)),
            existence_logits=self.existence_head(queries).squeeze(-1),
            visibility_logits=self.visibility_head(queries).squeeze(-1),
            log_variance=self.variance_head(queries).clamp(-8.0, 3.0),
            appearance=appearance,
            query_features=queries,
            attention=final_attention,
        )
