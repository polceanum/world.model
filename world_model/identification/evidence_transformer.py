"""Compact neural adaptation from public object-motion evidence.

The adapter learns across episodes, then adapts a particular object's physical
belief at inference time by attending to the evidence accumulated for that
object.  It deliberately predicts interpretable mass, drag, restitution, and
friction logits instead of an unconstrained future trajectory.  Analytic
geometry, action application, and contact resolution therefore remain the
stable inductive-bias scaffold around the learned component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor, nn

from world_model.belief import WorldBelief, slow_packing_map

EVIDENCE_FEATURE_DIM = 36
PREDICTED_PARAMETER_COUNT = 4


class PhysicsEvidenceKind(IntEnum):
    """Observable cause associated with one velocity transition."""

    FREE_MOTION = 0
    KNOWN_IMPULSE = 1
    BOUNDARY_CONTACT = 2


@dataclass(frozen=True, slots=True)
class EvidenceTransformerConfig:
    """Small CPU-friendly set-transformer configuration."""

    width: int = 48
    heads: int = 4
    layers: int = 2
    feed_forward_width: int = 96
    dropout: float = 0.0
    mass_bounds: tuple[float, float] = (0.50, 2.20)
    drag_bounds: tuple[float, float] = (0.03, 0.35)
    restitution_bounds: tuple[float, float] = (0.20, 0.97)
    friction_bounds: tuple[float, float] = (0.02, 0.75)

    def validate(self) -> EvidenceTransformerConfig:
        if self.width <= 0 or self.heads <= 0 or self.layers <= 0:
            raise ValueError("evidence-transformer width, heads, and layers must be positive")
        if self.width % self.heads:
            raise ValueError("evidence-transformer width must divide its head count")
        if self.feed_forward_width < self.width:
            raise ValueError("evidence-transformer feed-forward width must cover model width")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("evidence-transformer dropout must lie in [0,1)")
        for name, bounds in (
            ("mass", self.mass_bounds),
            ("drag", self.drag_bounds),
            ("restitution", self.restitution_bounds),
            ("friction", self.friction_bounds),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) for value in bounds)
                or bounds[0] <= 0.0
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} bounds must be finite, positive, and increasing")
        if self.restitution_bounds[1] >= 1.0 or self.friction_bounds[1] >= 1.0:
            raise ValueError("bounded restitution and friction must remain below one")
        return self


@dataclass(frozen=True, slots=True)
class PhysicsParameterPrediction:
    """Distribution over normalized and belief-native physical parameters."""

    normalized_mean: Tensor
    normalized_log_variance: Tensor
    belief_values: Tensor
    belief_log_variance: Tensor


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _parameter_bounds(
    config: EvidenceTransformerConfig, reference: Tensor
) -> tuple[Tensor, Tensor]:
    lower = reference.new_tensor(
        [
            math.log(config.mass_bounds[0]),
            math.log(config.drag_bounds[0]),
            _logit(config.restitution_bounds[0]),
            _logit(config.friction_bounds[0]),
        ]
    )
    upper = reference.new_tensor(
        [
            math.log(config.mass_bounds[1]),
            math.log(config.drag_bounds[1]),
            _logit(config.restitution_bounds[1]),
            _logit(config.friction_bounds[1]),
        ]
    )
    return lower, upper


def normalized_parameter_targets(
    values: Tensor,
    config: EvidenceTransformerConfig,
) -> Tensor:
    """Map belief-native ``[log mass, log drag, e-logit, mu-logit]`` to [0,1]."""

    if values.shape[-1] != PREDICTED_PARAMETER_COUNT or not values.is_floating_point():
        raise ValueError("physical parameter targets must be floating [...,4]")
    lower, upper = _parameter_bounds(config.validate(), values)
    return ((values - lower) / (upper - lower)).clamp(0.0, 1.0)


def velocity_transition_token(
    kind: PhysicsEvidenceKind | int,
    *,
    time_seconds: Tensor,
    duration_seconds: Tensor,
    duration_before_seconds: Tensor | None = None,
    duration_after_seconds: Tensor | None = None,
    velocity_before: Tensor,
    velocity_after: Tensor,
    known_impulse_world: Tensor | None = None,
    contact_normal_world: Tensor | None = None,
) -> Tensor:
    """Encode a public finite-difference transition into normalized features.

    The token contains no simulator parameter or identity label.  Velocities
    are derived upstream from observed metric positions; the only causal side
    information is a public action impulse or an observed stationary-boundary
    normal.
    """

    if velocity_before.shape != velocity_after.shape or velocity_before.shape[-1] != 3:
        raise ValueError("transition velocities must share shape [...,3]")
    prefix_shape = velocity_before.shape[:-1]
    for name, value in (
        ("time_seconds", time_seconds),
        ("duration_seconds", duration_seconds),
    ):
        if value.shape != prefix_shape:
            raise ValueError(f"{name} must match the transition prefix shape")
    before_duration = (
        duration_seconds if duration_before_seconds is None else duration_before_seconds
    )
    after_duration = duration_seconds if duration_after_seconds is None else duration_after_seconds
    if before_duration.shape != prefix_shape or after_duration.shape != prefix_shape:
        raise ValueError("before/after durations must match the transition prefix shape")
    kind_index = int(kind)
    if kind_index not in {int(item) for item in PhysicsEvidenceKind}:
        raise ValueError("unsupported physical evidence kind")
    impulse = (
        torch.zeros_like(velocity_before) if known_impulse_world is None else known_impulse_world
    )
    normal = (
        torch.zeros_like(velocity_before) if contact_normal_world is None else contact_normal_world
    )
    if impulse.shape != velocity_before.shape or normal.shape != velocity_before.shape:
        raise ValueError("impulse and contact normal must match transition velocity shape")
    if not all(
        torch.isfinite(value).all()
        for value in (
            time_seconds,
            duration_seconds,
            velocity_before,
            velocity_after,
            impulse,
            normal,
        )
    ):
        raise ValueError("physical evidence contains NaN or Inf")
    kind_features = velocity_before.new_zeros((*prefix_shape, len(PhysicsEvidenceKind)))
    kind_features[..., kind_index] = 1.0
    speed_before = torch.linalg.vector_norm(velocity_before, dim=-1, keepdim=True)
    speed_after = torch.linalg.vector_norm(velocity_after, dim=-1, keepdim=True)
    impulse_magnitude = torch.linalg.vector_norm(impulse, dim=-1, keepdim=True)
    impulse_direction = impulse / impulse_magnitude.clamp_min(1.0e-6)
    impulse_before = (velocity_before * impulse_direction).sum(dim=-1, keepdim=True)
    impulse_after = (velocity_after * impulse_direction).sum(dim=-1, keepdim=True)
    normal_before = (velocity_before * normal).sum(dim=-1, keepdim=True)
    normal_after = (velocity_after * normal).sum(dim=-1, keepdim=True)
    tangent_before = torch.linalg.vector_norm(
        velocity_before - normal_before * normal,
        dim=-1,
        keepdim=True,
    )
    tangent_after = torch.linalg.vector_norm(
        velocity_after - normal_after * normal,
        dim=-1,
        keepdim=True,
    )
    safe_speed_ratio = (speed_after / speed_before.clamp_min(1.0e-5)).clamp_min(1.0e-5)
    drag_candidate = (
        -safe_speed_ratio.log() / duration_seconds.unsqueeze(-1).clamp_min(1.0e-5)
    ).clamp(0.0, 0.70) / 0.35
    delta_velocity_along_impulse = ((velocity_after - velocity_before) * impulse_direction).sum(
        dim=-1, keepdim=True
    )
    mass_candidate = (
        impulse_magnitude / delta_velocity_along_impulse.abs().clamp_min(1.0e-5)
    ).clamp(0.0, 4.40) / 2.20
    restitution_candidate = (-normal_after / normal_before.clamp(max=-1.0e-5)).clamp(0.0, 1.0)
    friction_candidate = (
        (tangent_before - tangent_after).clamp_min(0.0)
        / ((1.0 + restitution_candidate) * normal_before.abs()).clamp_min(1.0e-5)
    ).clamp(0.0, 1.50) / 0.75
    candidates = torch.cat(
        (
            drag_candidate,
            mass_candidate,
            restitution_candidate,
            friction_candidate,
        ),
        dim=-1,
    )
    candidate_support = velocity_before.new_zeros((*prefix_shape, PREDICTED_PARAMETER_COUNT))
    if kind_index == int(PhysicsEvidenceKind.FREE_MOTION):
        candidate_support[..., 0] = 1.0
    elif kind_index == int(PhysicsEvidenceKind.KNOWN_IMPULSE):
        candidate_support[..., 1] = 1.0
    else:
        candidate_support[..., 2:] = 1.0
    candidates = candidates * candidate_support
    return torch.cat(
        (
            kind_features,
            (time_seconds / 2.0).unsqueeze(-1),
            (before_duration / 0.5).unsqueeze(-1),
            (after_duration / 0.5).unsqueeze(-1),
            velocity_before / 6.0,
            velocity_after / 6.0,
            impulse / 2.0,
            normal,
            speed_before / 6.0,
            speed_after / 6.0,
            (speed_after - speed_before) / 6.0,
            impulse_magnitude / 2.0,
            impulse_before / 6.0,
            impulse_after / 6.0,
            normal_before / 6.0,
            normal_after / 6.0,
            tangent_before / 6.0,
            tangent_after / 6.0,
            candidates,
            candidate_support,
        ),
        dim=-1,
    )


def free_motion_tokens(positions_world: Tensor, timestamps: Tensor) -> Tensor:
    """Create transition-pair tokens from one public metric position trace."""

    if positions_world.ndim < 2 or positions_world.shape[-1] != 3:
        raise ValueError("free-motion positions must have shape [...,T,3]")
    if timestamps.shape != positions_world.shape[:-1]:
        raise ValueError("free-motion timestamps must match position rows")
    if positions_world.shape[-2] < 3:
        raise ValueError("free-motion evidence requires at least three observations")
    intervals = timestamps[..., 1:] - timestamps[..., :-1]
    if not bool((intervals > 0.0).all()):
        raise ValueError("free-motion timestamps must increase strictly")
    velocity = (positions_world[..., 1:, :] - positions_world[..., :-1, :]) / intervals.unsqueeze(
        -1
    )
    return velocity_transition_token(
        PhysicsEvidenceKind.FREE_MOTION,
        time_seconds=timestamps[..., 1:-1],
        duration_seconds=0.5 * (intervals[..., :-1] + intervals[..., 1:]),
        velocity_before=velocity[..., :-1, :],
        velocity_after=velocity[..., 1:, :],
    )


def event_transition_token(
    kind: PhysicsEvidenceKind | int,
    positions_before_world: Tensor,
    timestamps_before: Tensor,
    positions_after_world: Tensor,
    timestamps_after: Tensor,
    *,
    known_impulse_world: Tensor | None = None,
    contact_normal_world: Tensor | None = None,
) -> Tensor:
    """Create one action/contact token from public positions around an event."""

    if int(kind) == int(PhysicsEvidenceKind.FREE_MOTION):
        raise ValueError("event token kind must be a known impulse or boundary contact")
    for positions, timestamps, label in (
        (positions_before_world, timestamps_before, "before"),
        (positions_after_world, timestamps_after, "after"),
    ):
        if positions.shape[-2:] != (3, 3) or timestamps.shape[-1] != 3:
            raise ValueError(f"{label} event evidence must contain exactly three positions")
        if timestamps.shape != positions.shape[:-1]:
            raise ValueError(f"{label} timestamps must match position rows")
    before_dt = timestamps_before[..., -1] - timestamps_before[..., -2]
    after_dt = timestamps_after[..., 1] - timestamps_after[..., 0]
    if not bool(((before_dt > 0.0) & (after_dt > 0.0)).all()):
        raise ValueError("event timestamps must increase around the event")
    velocity_before = (
        positions_before_world[..., -1, :] - positions_before_world[..., -2, :]
    ) / before_dt.unsqueeze(-1)
    velocity_after = (
        positions_after_world[..., 1, :] - positions_after_world[..., 0, :]
    ) / after_dt.unsqueeze(-1)
    # Absolute scene time is not a causal property of an impulse or contact.
    # Encoding it allowed a synthetic time-zero shortcut that failed when the
    # same evidence occurred later in a public runtime episode.
    event_time = torch.zeros_like(timestamps_before[..., -1])
    return velocity_transition_token(
        kind,
        time_seconds=event_time,
        duration_seconds=0.5 * (before_dt + after_dt),
        duration_before_seconds=before_dt,
        duration_after_seconds=after_dt,
        velocity_before=velocity_before,
        velocity_after=velocity_after,
        known_impulse_world=known_impulse_world,
        contact_normal_world=contact_normal_world,
    ).unsqueeze(-2)


class _SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.value = nn.Linear(width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.output(torch.nn.functional.silu(self.gate(value)) * self.value(value))


class _EvidenceBlock(nn.Module):
    def __init__(self, config: EvidenceTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.RMSNorm(config.width)
        self.feed_forward = _SwiGLU(config.width, config.feed_forward_width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        return tokens + self.dropout(self.feed_forward(self.feed_forward_norm(tokens)))


class NeuralPhysicsAdapter(nn.Module):
    """Shared evidence transformer that adapts interpretable object physics."""

    def __init__(self, config: EvidenceTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = (config or EvidenceTransformerConfig()).validate()
        self.input_projection = nn.Linear(EVIDENCE_FEATURE_DIM, self.config.width)
        self.query = nn.Parameter(torch.zeros(1, 1, self.config.width))
        self.blocks = nn.ModuleList(_EvidenceBlock(self.config) for _ in range(self.config.layers))
        self.output_norm = nn.RMSNorm(self.config.width)
        self.mean_head = nn.Linear(self.config.width, PREDICTED_PARAMETER_COUNT)
        self.log_variance_head = nn.Linear(self.config.width, PREDICTED_PARAMETER_COUNT)
        # The untrained model emits the centre of each declared range.  Only
        # the small decoder is zero; random shared features remain learnable on
        # optimizer step one and avoid the dead all-zero network used by the
        # former analytic evaluation harness.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, evidence: Tensor, valid: Tensor) -> PhysicsParameterPrediction:
        if evidence.ndim != 3 or evidence.shape[-1] != EVIDENCE_FEATURE_DIM:
            raise ValueError(f"physical evidence must have shape [B,T,{EVIDENCE_FEATURE_DIM}]")
        if valid.shape != evidence.shape[:2] or valid.dtype is not torch.bool:
            raise ValueError("physical evidence validity must be boolean [B,T]")
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("every adapter row requires at least one public evidence token")
        if not bool(torch.isfinite(evidence).all()):
            raise ValueError("physical evidence contains NaN or Inf")
        batch = evidence.shape[0]
        # Features are already normalized in physical units by the public
        # token builder.  Per-token LayerNorm would erase absolute speed and
        # impulse magnitude—the very evidence that identifies mass and drag.
        tokens = self.input_projection(evidence)
        query = self.query.expand(batch, -1, -1)
        tokens = torch.cat((query, tokens), dim=1)
        token_valid = torch.cat(
            (torch.ones((batch, 1), dtype=torch.bool, device=valid.device), valid),
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, token_valid)
        summary = self.output_norm(tokens[:, 0])
        normalized_mean = torch.sigmoid(self.mean_head(summary))
        normalized_log_variance = self.log_variance_head(summary).clamp(-8.0, 2.0)
        lower, upper = _parameter_bounds(self.config, normalized_mean)
        span = upper - lower
        belief_values = lower + span * normalized_mean
        belief_log_variance = normalized_log_variance + 2.0 * span.log()
        return PhysicsParameterPrediction(
            normalized_mean=normalized_mean,
            normalized_log_variance=normalized_log_variance,
            belief_values=belief_values,
            belief_log_variance=belief_log_variance,
        )

    def adapt_belief(
        self,
        belief: WorldBelief,
        evidence: Tensor,
        valid: Tensor,
    ) -> WorldBelief:
        """Condition all active object slots on their public evidence histories."""

        batch, object_count = belief.objects.active.shape
        if evidence.ndim != 4 or evidence.shape[:2] != (batch, object_count):
            raise ValueError(f"belief evidence must have shape [B,N,T,{EVIDENCE_FEATURE_DIM}]")
        if valid.shape != evidence.shape[:3]:
            raise ValueError("belief evidence validity must have shape [B,N,T]")
        active = belief.objects.active
        if not bool((valid.any(dim=-1) | ~active).all()):
            raise ValueError("every active object requires public adaptation evidence")
        safe_valid = valid.clone()
        safe_evidence = evidence.clone()
        inactive_without_evidence = ~safe_valid.any(dim=-1)
        first_token = torch.zeros_like(safe_valid)
        first_token[..., 0] = True
        safe_valid = safe_valid | (inactive_without_evidence.unsqueeze(-1) & first_token)
        safe_evidence = torch.where(
            inactive_without_evidence.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(safe_evidence),
            safe_evidence,
        )
        prediction = self(
            safe_evidence.reshape(batch * object_count, evidence.shape[-2], evidence.shape[-1]),
            safe_valid.reshape(batch * object_count, valid.shape[-1]),
        )
        values = prediction.belief_values.reshape(batch, object_count, -1)
        log_variance = prediction.belief_log_variance.reshape(batch, object_count, -1)
        objects = belief.objects.clone()
        mask = active.unsqueeze(-1)
        objects.log_mass = torch.where(mask, values[..., 0:1], objects.log_mass)
        objects.log_drag = torch.where(mask, values[..., 1:2], objects.log_drag)
        objects.restitution_logit = torch.where(
            mask,
            values[..., 2:3],
            objects.restitution_logit,
        )
        objects.friction_logit = torch.where(mask, values[..., 3:4], objects.friction_logit)
        packing = slow_packing_map(objects)
        for parameter_index, name in enumerate(
            ("log_mass", "log_drag", "restitution_logit", "friction_logit")
        ):
            parameter_slice = packing[name]
            proposed = log_variance[..., parameter_index : parameter_index + 1]
            current = objects.slow_log_variance[..., parameter_slice]
            objects.slow_log_variance[..., parameter_slice] = torch.where(
                mask,
                proposed.clamp(-12.0, 8.0),
                current,
            )
        return belief.replace(objects=objects)


__all__ = [
    "EVIDENCE_FEATURE_DIM",
    "EvidenceTransformerConfig",
    "NeuralPhysicsAdapter",
    "PhysicsEvidenceKind",
    "PhysicsParameterPrediction",
    "event_transition_token",
    "free_motion_tokens",
    "normalized_parameter_targets",
    "velocity_transition_token",
]
