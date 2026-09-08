"""Small permutation-equivariant proposal masks for variable RGB-D sets.

The proposer is deliberately only an observable image partitioner. RGB,
valid/log depth, analytic foreground, and image coordinates form its memory;
it never predicts a metric centre. Downstream RGB-D moments and calibrated
depth remain the sole owners of position. Eight fixed image anchors share
every learned operation, so reordering anchors exactly reorders proposals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from world_model.observations.rgb.structured_centres import structured_disc_centres

SET_MAX_OBJECTS = 6
SET_PROPOSAL_COUNT = 8
SET_APPEARANCE_DIM = 8
SET_FEATURE_DIM = 32
SET_ATTENTION_HEADS = 4
SET_ATTENTION_LAYERS = 2
SET_FEED_FORWARD_DIM = 64
SET_ATTENTION_MEMORY_STRIDE = 4
SET_ANCHOR_TEMPERATURE = 0.25
SET_COMPONENT_TEMPERATURE = 0.05
SET_FOREGROUND_TEMPERATURE_FLOOR = 0.10
SET_MAX_LOG_VARIANCE_RESIDUAL = 4.0
SET_MAX_CONFIGURABLE_LOG_VARIANCE_RESIDUAL = 32.0


@dataclass(frozen=True)
class RGBDSetProposalOutput:
    """Unordered, anchor-indexed learned residual proposal evidence."""

    slot_mask_logits: Tensor
    base_slot_mask_logits: Tensor
    full_mask_logits: Tensor
    full_mask_probability: Tensor
    base_full_mask_logits: Tensor
    background_mask_logits: Tensor
    mask_residual: Tensor
    existence_residual: Tensor
    appearance_residual: Tensor
    log_variance_residual: Tensor
    anchor_points: Tensor
    query_features: Tensor


def _set_anchor_grid(proposal_count: int = SET_PROPOSAL_COUNT) -> Tensor:
    """Return a deterministic near-square image cover in ``[-1,1]``.

    The historical eight-proposal profile remains the exact 4-by-2 grid.  A
    different capacity changes only the non-persistent anchor rows; learned
    operations stay shared and therefore carry no slot identity.
    """

    if isinstance(proposal_count, bool) or not isinstance(proposal_count, int):
        raise TypeError("proposal_count must be an integer")
    if proposal_count <= 0:
        raise ValueError("proposal_count must be positive")
    columns = math.ceil(math.sqrt(2.0 * proposal_count))
    rows = math.ceil(proposal_count / columns)
    x_axis = torch.linspace(-0.75, 0.75, columns)
    y_axis = torch.linspace(-0.50, 0.50, rows)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)[:proposal_count]


class _AnchorCrossAttentionBlock(nn.Module):
    """One shared pre-norm anchor-to-observation cross-attention block."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.memory_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            SET_ATTENTION_HEADS,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(feature_dim)
        feed_forward_dim = 2 * feature_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feed_forward_dim),
            nn.SiLU(),
            nn.Linear(feed_forward_dim, feature_dim),
        )

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.feed_forward(self.feed_forward_norm(queries))


class RGBDSetProposer(nn.Module):
    """A <=100k width-32, two-block shared-anchor set proposer.

    There are no learned slot embeddings or slot-index-specific parameters.
    The fixed anchors are a non-persistent buffer: they follow module device
    moves but add no checkpoint key. Mask, existence, and appearance output
    projections initialize to exact zero, so initial proposals equal the
    deterministic anchor cover while every output projection has a direct
    first-update gradient.
    """

    def __init__(
        self,
        *,
        proposal_count: int = SET_PROPOSAL_COUNT,
        appearance_dim: int = SET_APPEARANCE_DIM,
        feature_dim: int = SET_FEATURE_DIM,
        log_variance_residual_limit: float = SET_MAX_LOG_VARIANCE_RESIDUAL,
    ) -> None:
        super().__init__()
        if isinstance(proposal_count, bool) or not isinstance(proposal_count, int):
            raise TypeError("set proposer proposal_count must be an integer")
        if proposal_count <= 0:
            raise ValueError("set proposer proposal_count must be positive")
        if appearance_dim != SET_APPEARANCE_DIM:
            raise ValueError(f"set proposer requires appearance_dim={SET_APPEARANCE_DIM}")
        if feature_dim not in {SET_FEATURE_DIM, 2 * SET_FEATURE_DIM}:
            raise ValueError("set proposer feature_dim must be 32 or the permitted widened 64")
        if feature_dim % SET_ATTENTION_HEADS:
            raise ValueError("set proposer feature_dim must divide four attention heads")
        if (
            isinstance(log_variance_residual_limit, bool)
            or not isinstance(log_variance_residual_limit, (int, float))
            or not math.isfinite(float(log_variance_residual_limit))
            or not 0.0
            < float(log_variance_residual_limit)
            <= SET_MAX_CONFIGURABLE_LOG_VARIANCE_RESIDUAL
        ):
            raise ValueError(
                "set proposer log_variance_residual_limit must be finite and lie in (0,32]"
            )

        self.proposal_count = proposal_count
        self.appearance_dim = appearance_dim
        self.feature_dim = feature_dim
        self.log_variance_residual_limit = float(log_variance_residual_limit)
        # 3 RGB + log depth + valid depth + analytic foreground + image xy.
        # The observation channels are already spatially registered.  A shared
        # pointwise projection is therefore sufficient to form the learned
        # per-pixel mask memory; spatial mixing is supplied by the two anchor
        # cross-attention blocks.  Attention reads a safely pooled copy while
        # masks continue to use every full-resolution projected pixel.
        self.memory_projection = nn.Conv2d(8, feature_dim, kernel_size=1)
        self.memory_norm = nn.LayerNorm(feature_dim)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.cross_attention_blocks = nn.ModuleList(
            _AnchorCrossAttentionBlock(feature_dim) for _ in range(SET_ATTENTION_LAYERS)
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.mask_residual_projection = nn.Linear(feature_dim, feature_dim)
        self.existence_residual_head = nn.Linear(feature_dim, 1)
        self.appearance_residual_head = nn.Linear(feature_dim, appearance_dim)
        self.log_variance_residual_head = nn.Linear(feature_dim, 3)
        self.register_buffer(
            "anchor_points",
            _set_anchor_grid(proposal_count),
            persistent=False,
        )

        nn.init.zeros_(self.mask_residual_projection.weight)
        nn.init.zeros_(self.mask_residual_projection.bias)
        nn.init.zeros_(self.existence_residual_head.weight)
        nn.init.zeros_(self.existence_residual_head.bias)
        nn.init.zeros_(self.appearance_residual_head.weight)
        nn.init.zeros_(self.appearance_residual_head.bias)
        nn.init.zeros_(self.log_variance_residual_head.weight)
        nn.init.zeros_(self.log_variance_residual_head.bias)

    def parameter_count(self) -> int:
        """Return the complete trainable proposer capacity."""

        return sum(parameter.numel() for parameter in self.parameters())

    def _residual_heads_are_exact_zero(self) -> bool:
        """Return whether all behavior-owning residual projections are zero."""

        residual_modules = (
            self.mask_residual_projection,
            self.existence_residual_head,
            self.appearance_residual_head,
            self.log_variance_residual_head,
        )
        return all(
            not bool(torch.count_nonzero(parameter))
            for module in residual_modules
            for parameter in module.parameters()
        )

    @staticmethod
    def _normalized_full_masks(full_mask_logits: Tensor) -> Tensor:
        """Normalize mask logits in an anchor-order-independent reduction."""

        # Ordinary softmax reductions may change their last bits when callers
        # permute proposal anchors.  Sorting the positive terms gives one
        # canonical reduction order while retaining the caller-visible class
        # order in the numerator.
        full_maximum = full_mask_logits.amax(dim=1, keepdim=True)
        full_weight = torch.exp(full_mask_logits - full_maximum)
        full_denominator = torch.sort(full_weight, dim=1).values.sum(dim=1, keepdim=True)
        return full_weight / full_denominator

    @staticmethod
    def _validate_inputs(
        image: Tensor,
        log_depth: Tensor,
        valid_depth: Tensor,
        analytic_foreground: Tensor,
        image_coordinates: Tensor,
    ) -> tuple[int, int, int]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("set proposer image must have shape [B,3,H,W]")
        if image.dtype not in {torch.float32, torch.float64}:
            raise TypeError("set proposer supports only float32 and float64")
        batch, _, height, width = image.shape
        if min(height, width) < 2:
            raise ValueError("set proposer image dimensions must be at least two pixels")
        scalar_shape = (batch, 1, height, width)
        if log_depth.shape != scalar_shape:
            raise ValueError("set proposer log_depth must have shape [B,1,H,W]")
        if valid_depth.shape != scalar_shape or valid_depth.dtype is not torch.bool:
            raise ValueError("set proposer valid_depth must be boolean [B,1,H,W]")
        if analytic_foreground.shape != scalar_shape:
            raise ValueError("set proposer analytic_foreground must have shape [B,1,H,W]")
        if image_coordinates.shape != (batch, 2, height, width):
            raise ValueError("set proposer image_coordinates must have shape [B,2,H,W]")
        for name, value in (
            ("log_depth", log_depth),
            ("analytic_foreground", analytic_foreground),
            ("image_coordinates", image_coordinates),
        ):
            if not value.is_floating_point():
                raise TypeError(f"set proposer {name} must be floating point")
            if value.dtype != image.dtype or value.device != image.device:
                raise ValueError(f"set proposer {name} must share image dtype and device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"set proposer {name} contains NaN or Inf")
        if valid_depth.device != image.device:
            raise ValueError("set proposer valid_depth must share image device")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("set proposer image contains NaN or Inf")
        if bool(torch.any((analytic_foreground < 0.0) | (analytic_foreground > 1.0))):
            raise ValueError("set proposer analytic_foreground must lie in [0,1]")
        if bool(torch.any((image_coordinates < -1.0) | (image_coordinates > 1.0))):
            raise ValueError("set proposer image_coordinates must lie in [-1,1]")
        return batch, height, width

    @staticmethod
    def _base_logits(image_coordinates: Tensor, anchors: Tensor) -> Tensor:
        coordinates = image_coordinates.permute(0, 2, 3, 1)
        squared_distance = (
            (coordinates[:, None] - anchors[None, :, None, None]).square().sum(dim=-1)
        )
        # This independent Gaussian logit has no reduction over anchors. The
        # arithmetic for any anchor is unchanged by anchor order.
        return 1.0 - squared_distance / image_coordinates.new_tensor(
            2.0 * SET_ANCHOR_TEMPERATURE**2
        )

    @staticmethod
    def _analytic_component_baseline(
        image: Tensor,
        valid_depth: Tensor,
        analytic_foreground: Tensor,
        image_coordinates: Tensor,
        anchors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return an observable component partition aligned to fixed anchors.

        Component discovery is a detached analytic RGB operation.  It supplies
        only the zero-residual proposal prior: learned mask residuals still own
        all trainable corrections, and calibrated depth remains the sole owner
        of metric distance.  Aligning components to fixed anchors gives empty
        birth-candidate rows without introducing learned slot identities.
        """

        batch = image.shape[0]
        aligned = structured_disc_centres(
            image,
            anchors.unsqueeze(0).expand(batch, -1, -1),
            threshold=0.04,
            minimum_pixels=4,
            maximum_assignment_distance=0.80,
        )
        coordinates = image_coordinates.permute(0, 2, 3, 1)
        squared_distance = (
            (coordinates[:, None] - aligned.centres[:, :, None, None]).square().sum(dim=-1)
        )
        component_logits = 1.0 - squared_distance / image.new_tensor(
            2.0 * SET_COMPONENT_TEMPERATURE**2
        )
        # Keep every tensor finite for the objective/audit path.  The value is
        # far below any valid component logit but avoids infinities in saved
        # diagnostics and in the all-background lifecycle prefix.
        invalid_logit = image.new_tensor(-80.0)
        component_logits = torch.where(
            aligned.valid_mask[:, :, None, None],
            component_logits,
            invalid_logit,
        )
        maximum = component_logits.amax(dim=1, keepdim=True)
        component_weight = torch.exp(component_logits - maximum)
        component_weight = component_weight * aligned.valid_mask[:, :, None, None]
        denominator = torch.sort(component_weight, dim=1).values.sum(dim=1, keepdim=True)
        # A scene may legitimately be empty before a visible birth.  Its
        # proposal probabilities must then be exactly zero, not a diffuse set
        # of fabricated foreground objects.
        component_probability = torch.where(
            denominator > 0.0,
            component_weight / denominator.clamp_min(torch.finfo(image.dtype).tiny),
            torch.zeros_like(component_weight),
        )
        has_component = aligned.valid_mask.any(dim=1, keepdim=True)
        effective_foreground = analytic_foreground * has_component[:, :, None, None]
        soft_probability = torch.cat(
            (
                1.0 - effective_foreground,
                effective_foreground * component_probability,
            ),
            dim=1,
        )
        nearest_component = component_logits.argmax(dim=1)
        hard_component = torch.nn.functional.one_hot(
            nearest_component,
            num_classes=anchors.shape[0],
        ).permute(0, 3, 1, 2)
        supported_depth = valid_depth & has_component[:, :, None, None]
        hard_slot_probability = hard_component.to(image.dtype) * supported_depth.to(image.dtype)
        hard_probability = torch.cat(
            (
                (~supported_depth).to(image.dtype),
                hard_slot_probability,
            ),
            dim=1,
        )
        # Exact forward values reproduce the analytic component masks while
        # the observable soft foreground/partition supplies a straight-through
        # image gradient for the learned residual path.
        base_probability = soft_probability + (hard_probability - soft_probability).detach()
        return component_logits, base_probability

    def forward(
        self,
        image: Tensor,
        log_depth: Tensor,
        valid_depth: Tensor,
        analytic_foreground: Tensor,
        image_coordinates: Tensor,
    ) -> RGBDSetProposalOutput:
        batch, height, width = self._validate_inputs(
            image,
            log_depth,
            valid_depth,
            analytic_foreground,
            image_coordinates,
        )
        anchors = self.anchor_points.to(dtype=image.dtype)
        base_logits, base_full_mask_probability = self._analytic_component_baseline(
            image,
            valid_depth,
            analytic_foreground,
            image_coordinates,
            anchors,
        )
        base_full_mask_logits = base_full_mask_probability.clamp_min(
            torch.finfo(image.dtype).tiny
        ).log()

        # The frozen structured baseline has exactly-zero behavior heads.  In
        # evaluation without autograd, none of the learned query features can
        # affect a public measurement, so avoid constructing their large image
        # memory.  The zero diagnostic query tensor deliberately records that
        # the learned path was skipped.  Training and every grad-enabled call
        # still execute the complete two-block architecture.
        if (
            not self.training
            and not torch.is_grad_enabled()
            and self._residual_heads_are_exact_zero()
        ):
            mask_residual = image.new_zeros((batch, self.proposal_count, height, width))
            full_mask_logits = base_full_mask_logits
            full_mask_probability = self._normalized_full_masks(full_mask_logits)
            query_features = image.new_zeros((batch, self.proposal_count, self.feature_dim))
            return RGBDSetProposalOutput(
                slot_mask_logits=base_logits,
                base_slot_mask_logits=base_logits,
                full_mask_logits=full_mask_logits,
                full_mask_probability=full_mask_probability,
                base_full_mask_logits=base_full_mask_logits,
                background_mask_logits=full_mask_logits[:, :1],
                mask_residual=mask_residual,
                existence_residual=image.new_zeros((batch, self.proposal_count)),
                appearance_residual=image.new_zeros(
                    (batch, self.proposal_count, self.appearance_dim)
                ),
                log_variance_residual=image.new_zeros((batch, self.proposal_count, 3)),
                anchor_points=anchors,
                query_features=query_features,
            )

        normalized_rgb = (image - image.new_tensor(0.5)) * image.new_tensor(2.0)
        observable_input = torch.cat(
            (
                normalized_rgb,
                log_depth,
                valid_depth.to(dtype=image.dtype),
                analytic_foreground,
                image_coordinates,
            ),
            dim=1,
        )
        memory_map = self.memory_projection(observable_input)
        memory = self.memory_norm(memory_map.flatten(2).transpose(1, 2))
        attention_memory_map = F.avg_pool2d(
            memory_map,
            kernel_size=SET_ATTENTION_MEMORY_STRIDE,
            stride=SET_ATTENTION_MEMORY_STRIDE,
            ceil_mode=True,
        )
        attention_memory = self.memory_norm(attention_memory_map.flatten(2).transpose(1, 2))
        queries = self.anchor_encoder(anchors).unsqueeze(0).expand(batch, -1, -1)
        for block in self.cross_attention_blocks:
            queries = block(queries, attention_memory)
        queries = self.output_norm(queries)

        mask_queries = self.mask_residual_projection(queries)
        mask_residual = torch.einsum("bpc,blc->bpl", mask_queries, memory)
        mask_residual = mask_residual.reshape(batch, self.proposal_count, height, width)
        mask_residual = mask_residual / math.sqrt(self.feature_dim)
        existence_residual = self.existence_residual_head(queries).squeeze(-1)
        appearance_residual = self.appearance_residual_head(queries)
        log_variance_residual = self.log_variance_residual_limit * torch.tanh(
            self.log_variance_residual_head(queries)
        )
        full_mask_logits = base_full_mask_logits + torch.cat(
            (torch.zeros_like(analytic_foreground), mask_residual),
            dim=1,
        )
        full_mask_probability = self._normalized_full_masks(full_mask_logits)
        return RGBDSetProposalOutput(
            slot_mask_logits=base_logits + mask_residual,
            base_slot_mask_logits=base_logits,
            full_mask_logits=full_mask_logits,
            full_mask_probability=full_mask_probability,
            base_full_mask_logits=base_full_mask_logits,
            background_mask_logits=full_mask_logits[:, :1],
            mask_residual=mask_residual,
            existence_residual=existence_residual,
            appearance_residual=appearance_residual,
            log_variance_residual=log_variance_residual,
            anchor_points=anchors,
            query_features=queries,
        )


__all__ = [
    "RGBDSetProposalOutput",
    "RGBDSetProposer",
    "SET_ANCHOR_TEMPERATURE",
    "SET_APPEARANCE_DIM",
    "SET_ATTENTION_HEADS",
    "SET_ATTENTION_LAYERS",
    "SET_ATTENTION_MEMORY_STRIDE",
    "SET_COMPONENT_TEMPERATURE",
    "SET_FEATURE_DIM",
    "SET_FOREGROUND_TEMPERATURE_FLOOR",
    "SET_MAX_OBJECTS",
    "SET_MAX_LOG_VARIANCE_RESIDUAL",
    "SET_MAX_CONFIGURABLE_LOG_VARIANCE_RESIDUAL",
    "SET_PROPOSAL_COUNT",
]
