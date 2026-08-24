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
    dense_center_logits: Tensor | None = None

    def to(self, device: torch.device | str) -> GlobalDetectorOutput:
        """Move proposal tensors while preserving the autograd copy path."""

        return GlobalDetectorOutput(
            centre=self.centre.to(device),
            log_radius=self.log_radius.to(device),
            inverse_depth_residual=self.inverse_depth_residual.to(device),
            colour=self.colour.to(device),
            existence_logits=self.existence_logits.to(device),
            visibility_logits=self.visibility_logits.to(device),
            log_variance=self.log_variance.to(device),
            appearance=self.appearance.to(device),
            query_features=self.query_features.to(device),
            attention=self.attention.to(device),
            dense_center_logits=(
                None if self.dense_center_logits is None else self.dense_center_logits.to(device)
            ),
        )


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
        # A fresh learned fallback should require visual evidence before it can
        # create persistent state.  A negative prior stays below the runtime
        # birth gate while supervised positive/negative queries can move it.
        nn.init.constant_(self.existence_head.bias, -2.0)
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


class DenseGlobalObjectDetector(nn.Module):
    """Dense center proposals with typed attributes sampled at local maxima.

    The center branch is the exact architecture qualified by the specification
    1.68 feasibility probe.  Attribute maps complete the existing typed global
    detector contract without changing the center logits or their decoding.
    """

    _ATTRIBUTE_DIM = 1 + 1 + 3 + 1 + 7

    def __init__(
        self,
        *,
        feature_dim: int,
        query_count: int,
        appearance_dim: int,
        hidden_dim: int = 64,
        minimum_radius: float = 0.015,
        maximum_radius: float = 0.75,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or query_count <= 0 or appearance_dim <= 0:
            raise ValueError("dense detector dimensions must be positive")
        if hidden_dim <= 0 or hidden_dim % 8:
            raise ValueError("dense detector hidden_dim must be positive and divisible by 8")
        self.query_count = query_count
        self.appearance_dim = appearance_dim
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.trunk = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.attribute_head = nn.Conv2d(
            hidden_dim,
            self._ATTRIBUTE_DIM + appearance_dim,
            kernel_size=1,
        )
        # Preserve the exact fixed feasibility initialization, independent of
        # unrelated model construction order. Attribute rows use a separate
        # deterministic stream and conservative typed priors.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(168)
            nn.init.kaiming_uniform_(self.trunk[0].weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.trunk[0].weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.trunk[0].bias, -bound, bound)
            nn.init.normal_(self.center_head.weight, std=0.001)
            nn.init.constant_(self.center_head.bias, -2.19)
            torch.manual_seed(169)
            nn.init.normal_(self.attribute_head.weight, std=0.001)
        nn.init.zeros_(self.attribute_head.bias)
        with torch.no_grad():
            self.attribute_head.bias[0] = -2.0
            self.attribute_head.bias[5] = 0.5
            self.attribute_head.bias[6:13] = -2.0

    @staticmethod
    def _gather_map(value: Tensor, indices: Tensor) -> Tensor:
        if value.ndim != 4 or indices.ndim != 2 or value.shape[0] != indices.shape[0]:
            raise ValueError("dense map gathering expects [B,C,H,W] and [B,Q]")
        flattened = value.flatten(2).transpose(1, 2)
        return flattened.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, flattened.shape[-1]),
        )

    def forward(self, feature_map: Tensor) -> GlobalDetectorOutput:
        if feature_map.ndim != 4:
            raise ValueError("dense global detector feature map must be [B,C,H,W]")
        hidden = self.trunk(feature_map)
        center_logits = self.center_head(hidden)
        probability = center_logits.sigmoid()
        retained = probability.eq(F.max_pool2d(probability, kernel_size=3, stride=1, padding=1))
        _, indices = (probability * retained).flatten(1).topk(self.query_count, dim=-1)
        height, width = center_logits.shape[-2:]
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices.remainder(width)
        centre = torch.stack(
            (
                2.0 * x.to(center_logits.dtype) / max(width - 1, 1) - 1.0,
                2.0 * y.to(center_logits.dtype) / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        )
        # Attribute supervision must not perturb the already-qualified center
        # representation. The typed maps retain their own trainable head while
        # consuming a detached copy of the fixed center trunk features.
        attributes = self._gather_map(self.attribute_head(hidden.detach()), indices)
        radius_raw = attributes[..., 0:1]
        radius = self.minimum_radius + (self.maximum_radius - self.minimum_radius) * torch.sigmoid(
            radius_raw
        )
        appearance_start = self._ATTRIBUTE_DIM
        appearance = F.normalize(attributes[..., appearance_start:], dim=-1)
        attention = F.one_hot(
            indices,
            num_classes=height * width,
        ).to(dtype=center_logits.dtype)
        return GlobalDetectorOutput(
            centre=centre,
            log_radius=radius.log(),
            inverse_depth_residual=0.25 * torch.tanh(attributes[..., 1:2]),
            colour=torch.sigmoid(attributes[..., 2:5]),
            existence_logits=self._gather_map(center_logits, indices).squeeze(-1),
            visibility_logits=attributes[..., 5],
            log_variance=attributes[..., 6:13].clamp(-8.0, 3.0),
            appearance=appearance,
            query_features=self._gather_map(hidden, indices),
            attention=attention,
            dense_center_logits=center_logits,
        )

    def center_logits(self, feature_map: Tensor) -> Tensor:
        """Return the unsampled heatmap used by the dense focal objective."""

        if feature_map.ndim != 4:
            raise ValueError("dense global detector feature map must be [B,C,H,W]")
        return self.center_head(self.trunk(feature_map))
