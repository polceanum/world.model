"""Fit a tiny causal RGB trajectory change-point classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LinearChangePointGate:
    weights: tuple[float, ...]
    bias: float
    probability_threshold: float

    def logits(self, features: Tensor) -> Tensor:
        if features.shape[-1] != len(self.weights):
            raise ValueError("feature dimension does not match gate weights")
        weights = features.new_tensor(self.weights)
        return torch.einsum("...f,f->...", features, weights) + self.bias


@dataclass(frozen=True)
class MLPChangePointGate:
    hidden_weights: tuple[float, ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float
    probability_threshold: float

    def logits(self, features: Tensor) -> Tensor:
        input_features = features.shape[-1]
        hidden_features = len(self.hidden_bias)
        if len(self.hidden_weights) != input_features * hidden_features:
            raise ValueError("hidden weight dimensions do not match features")
        if len(self.output_weights) != hidden_features:
            raise ValueError("output weight dimensions do not match hidden layer")
        hidden_weights = features.new_tensor(self.hidden_weights).reshape(
            hidden_features,
            input_features,
        )
        hidden_bias = features.new_tensor(self.hidden_bias)
        output_weights = features.new_tensor(self.output_weights)
        hidden = F.silu(F.linear(features, hidden_weights, hidden_bias))
        return F.linear(
            hidden,
            output_weights.unsqueeze(0),
            features.new_tensor([self.output_bias]),
        ).squeeze(-1)


@dataclass(frozen=True)
class MLPOutgoingVelocityProposal:
    hidden_weights: tuple[float, ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float
    variance: float
    maximum_delta: float

    def delta(self, features: Tensor) -> Tensor:
        input_features = features.shape[-1]
        hidden_features = len(self.hidden_bias)
        if len(self.hidden_weights) != input_features * hidden_features:
            raise ValueError("hidden weight dimensions do not match proposal features")
        hidden_weights = features.new_tensor(self.hidden_weights).reshape(
            hidden_features,
            input_features,
        )
        hidden = F.silu(
            F.linear(
                features,
                hidden_weights,
                features.new_tensor(self.hidden_bias),
            )
        )
        output = F.linear(
            hidden,
            features.new_tensor(self.output_weights).unsqueeze(0),
            features.new_tensor([self.output_bias]),
        ).squeeze(-1)
        return output.clamp(-self.maximum_delta, self.maximum_delta)


@dataclass(frozen=True)
class MLPLateralVelocityIntervention:
    """Tiny lateral measurement proposal with a learned soft abstention gain."""

    hidden_weights: tuple[float, ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: tuple[float, float]
    variance_floor: float
    variance_ceiling: float
    gain_power: float
    maximum_delta: float

    def propose(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        input_features = features.shape[-1]
        hidden_features = len(self.hidden_bias)
        if len(self.hidden_weights) != input_features * hidden_features:
            raise ValueError("hidden weight dimensions do not match intervention features")
        if len(self.output_weights) != 2 * hidden_features:
            raise ValueError("output weight dimensions do not match intervention head")
        hidden_weights = features.new_tensor(self.hidden_weights).reshape(
            hidden_features,
            input_features,
        )
        hidden = F.silu(
            F.linear(
                features,
                hidden_weights,
                features.new_tensor(self.hidden_bias),
            )
        )
        output = F.linear(
            hidden,
            features.new_tensor(self.output_weights).reshape(2, hidden_features),
            features.new_tensor(self.output_bias),
        )
        delta = output[..., 0].clamp(-self.maximum_delta, self.maximum_delta)
        gain = output[..., 1].sigmoid()
        variance = (self.variance_floor / gain.clamp_min(1.0e-4).pow(self.gain_power)).clamp(
            max=self.variance_ceiling
        )
        return delta, gain, variance


def binary_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have matching shapes")
    predicted = logits.sigmoid() >= threshold
    target = targets.bool()
    true_positive = int((predicted & target).sum())
    false_positive = int((predicted & ~target).sum())
    false_negative = int((~predicted & target).sum())
    true_negative = int((~predicted & ~target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
        "positive_rate": (true_positive + false_positive) / max(target.numel(), 1),
    }


def select_precision_threshold(
    logits: Tensor,
    targets: Tensor,
    *,
    minimum_precision: float = 0.8,
) -> tuple[float, dict[str, float]]:
    """Select the highest-recall threshold satisfying a precision floor."""

    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must lie in (0, 1]")
    probabilities = logits.sigmoid()
    candidates = (
        torch.unique(
            torch.cat(
                (
                    probabilities.detach().cpu(),
                    probabilities.new_tensor([0.5, 0.9, 0.95, 0.99]).cpu(),
                )
            )
        )
        .sort()
        .values
    )
    selected: tuple[float, dict[str, float]] | None = None
    fallback: tuple[float, dict[str, float], float] | None = None
    for value in candidates:
        threshold = float(value)
        metrics = binary_metrics(logits, targets, threshold=threshold)
        if (
            metrics["precision"] >= minimum_precision
            and metrics["true_positive"] > 0
            and (selected is None or metrics["recall"] > selected[1]["recall"])
        ):
            selected = threshold, metrics
        precision = metrics["precision"]
        recall = metrics["recall"]
        f_half = 1.25 * precision * recall / max(0.25 * precision + recall, 1.0e-12)
        if fallback is None or f_half > fallback[2]:
            fallback = threshold, metrics, f_half
    if selected is not None:
        return selected
    assert fallback is not None
    return fallback[0], fallback[1]


def fit_linear_change_point_gate(
    train_features: Tensor,
    train_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    steps: int = 800,
    learning_rate: float = 0.05,
    weight_decay: float = 1.0e-3,
    maximum_positive_weight: float = 20.0,
    minimum_precision: float = 0.8,
) -> tuple[LinearChangePointGate, dict[str, float]]:
    """Fit balanced logistic regression and select a conservative threshold."""

    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("features must have shape [samples, features]")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("training and validation feature dimensions must match")
    if train_targets.shape != train_features.shape[:1]:
        raise ValueError("train_targets must have shape [samples]")
    if validation_targets.shape != validation_features.shape[:1]:
        raise ValueError("validation_targets must have shape [samples]")
    if steps <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid optimizer settings")
    train_features = train_features.float()
    validation_features = validation_features.float()
    train_targets = train_targets.float()
    validation_targets = validation_targets.float()
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("training data must contain positive and negative examples")
    positive_weight = float((negatives / positives).clamp(min=1.0, max=maximum_positive_weight))

    weights = torch.zeros(train_features.shape[1], requires_grad=True)
    prior_probability = float(positives / train_targets.numel())
    bias = torch.tensor(
        math.log(prior_probability / max(1.0 - prior_probability, 1.0e-8)),
        requires_grad=True,
    )
    optimizer = torch.optim.AdamW(
        (weights, bias),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    positive_weight_tensor = train_features.new_tensor(positive_weight)
    for _ in range(steps):
        logits = train_features @ weights + bias
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_targets,
            pos_weight=positive_weight_tensor,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = train_features @ weights + bias
        validation_logits = validation_features @ weights + bias
        threshold, validation_metrics = select_precision_threshold(
            validation_logits,
            validation_targets,
            minimum_precision=minimum_precision,
        )
        train_metrics = binary_metrics(train_logits, train_targets, threshold=threshold)
    gate = LinearChangePointGate(
        weights=tuple(float(value) for value in weights.detach()),
        bias=float(bias.detach()),
        probability_threshold=threshold,
    )
    metrics = {
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"validation_{name}": value for name, value in validation_metrics.items()},
        "train_examples": float(train_targets.numel()),
        "train_positive_examples": float(train_targets.sum()),
        "validation_examples": float(validation_targets.numel()),
        "validation_positive_examples": float(validation_targets.sum()),
        "positive_weight": positive_weight,
        "probability_threshold": threshold,
    }
    return gate, metrics


def fit_mlp_change_point_gate(
    train_features: Tensor,
    train_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    hidden_features: int = 12,
    steps: int = 2000,
    learning_rate: float = 0.02,
    weight_decay: float = 2.0e-3,
    maximum_positive_weight: float = 20.0,
    minimum_precision: float = 0.8,
    seed: int = 0,
) -> tuple[MLPChangePointGate, dict[str, float]]:
    """Fit a one-hidden-layer gate while keeping inference tiny and explicit."""

    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("features must have shape [samples, features]")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("training and validation feature dimensions must match")
    if train_targets.shape != train_features.shape[:1]:
        raise ValueError("train_targets must have shape [samples]")
    if validation_targets.shape != validation_features.shape[:1]:
        raise ValueError("validation_targets must have shape [samples]")
    if hidden_features <= 0 or steps <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid MLP fitting settings")
    train_features = train_features.float()
    validation_features = validation_features.float()
    train_targets = train_targets.float()
    validation_targets = validation_targets.float()
    positives = train_targets.sum()
    negatives = train_targets.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("training data must contain positive and negative examples")
    positive_weight = float((negatives / positives).clamp(min=1.0, max=maximum_positive_weight))
    generator = torch.Generator().manual_seed(seed)
    hidden_weights = (
        torch.randn(
            hidden_features,
            train_features.shape[1],
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    hidden_bias = torch.zeros(hidden_features, requires_grad=True)
    output_weights = (torch.randn(hidden_features, generator=generator) * 0.1).requires_grad_()
    prior_probability = float(positives / train_targets.numel())
    output_bias = torch.tensor(
        math.log(prior_probability / max(1.0 - prior_probability, 1.0e-8)),
        requires_grad=True,
    )
    optimizer = torch.optim.AdamW(
        (hidden_weights, hidden_bias, output_weights, output_bias),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    positive_weight_tensor = train_features.new_tensor(positive_weight)
    for _ in range(steps):
        hidden = F.silu(F.linear(train_features, hidden_weights, hidden_bias))
        logits = F.linear(
            hidden,
            output_weights.unsqueeze(0),
            output_bias.unsqueeze(0),
        ).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_targets,
            pos_weight=positive_weight_tensor,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = F.linear(
            F.silu(F.linear(train_features, hidden_weights, hidden_bias)),
            output_weights.unsqueeze(0),
            output_bias.unsqueeze(0),
        ).squeeze(-1)
        validation_logits = F.linear(
            F.silu(F.linear(validation_features, hidden_weights, hidden_bias)),
            output_weights.unsqueeze(0),
            output_bias.unsqueeze(0),
        ).squeeze(-1)
        threshold, validation_metrics = select_precision_threshold(
            validation_logits,
            validation_targets,
            minimum_precision=minimum_precision,
        )
        train_metrics = binary_metrics(train_logits, train_targets, threshold=threshold)
    gate = MLPChangePointGate(
        hidden_weights=tuple(float(value) for value in hidden_weights.detach().flatten()),
        hidden_bias=tuple(float(value) for value in hidden_bias.detach()),
        output_weights=tuple(float(value) for value in output_weights.detach()),
        output_bias=float(output_bias.detach()),
        probability_threshold=threshold,
    )
    metrics = {
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"validation_{name}": value for name, value in validation_metrics.items()},
        "train_examples": float(train_targets.numel()),
        "train_positive_examples": float(train_targets.sum()),
        "validation_examples": float(validation_targets.numel()),
        "validation_positive_examples": float(validation_targets.sum()),
        "positive_weight": positive_weight,
        "probability_threshold": threshold,
        "hidden_features": float(hidden_features),
    }
    return gate, metrics


def fit_mlp_outgoing_velocity_proposal(
    train_features: Tensor,
    train_target_delta: Tensor,
    validation_features: Tensor,
    validation_target_delta: Tensor,
    *,
    hidden_features: int = 8,
    steps: int = 2000,
    learning_rate: float = 0.01,
    weight_decay: float = 5.0e-3,
    minimum_variance: float = 0.25,
    maximum_variance: float = 9.0,
    maximum_delta: float = 5.0,
    seed: int = 0,
) -> tuple[MLPOutgoingVelocityProposal, dict[str, float]]:
    """Fit an axis-local outgoing-velocity delta with calibrated variance."""

    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("proposal features must have shape [samples, features]")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("training and validation feature dimensions must match")
    if train_target_delta.shape != train_features.shape[:1]:
        raise ValueError("train_target_delta must have shape [samples]")
    if validation_target_delta.shape != validation_features.shape[:1]:
        raise ValueError("validation_target_delta must have shape [samples]")
    if (
        hidden_features <= 0
        or steps <= 0
        or learning_rate <= 0
        or weight_decay < 0
        or not 0 < minimum_variance <= maximum_variance
        or maximum_delta <= 0
    ):
        raise ValueError("invalid outgoing proposal fitting settings")
    train_features = train_features.float()
    validation_features = validation_features.float()
    train_target_delta = train_target_delta.float()
    validation_target_delta = validation_target_delta.float()
    if train_target_delta.numel() == 0 or validation_target_delta.numel() == 0:
        raise ValueError("proposal training and validation sets cannot be empty")

    generator = torch.Generator().manual_seed(seed)
    hidden_weights = (
        torch.randn(
            hidden_features,
            train_features.shape[1],
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    hidden_bias = torch.zeros(hidden_features, requires_grad=True)
    output_weights = (torch.randn(hidden_features, generator=generator) * 0.1).requires_grad_()
    output_bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.AdamW(
        (hidden_weights, hidden_bias, output_weights, output_bias),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for _ in range(steps):
        prediction = F.linear(
            F.silu(F.linear(train_features, hidden_weights, hidden_bias)),
            output_weights.unsqueeze(0),
            output_bias,
        ).squeeze(-1)
        prediction = prediction.clamp(-maximum_delta, maximum_delta)
        loss = F.smooth_l1_loss(prediction, train_target_delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prediction = F.linear(
            F.silu(F.linear(train_features, hidden_weights, hidden_bias)),
            output_weights.unsqueeze(0),
            output_bias,
        ).squeeze(-1)
        validation_prediction = F.linear(
            F.silu(F.linear(validation_features, hidden_weights, hidden_bias)),
            output_weights.unsqueeze(0),
            output_bias,
        ).squeeze(-1)
        train_prediction = train_prediction.clamp(-maximum_delta, maximum_delta)
        validation_prediction = validation_prediction.clamp(
            -maximum_delta,
            maximum_delta,
        )
        validation_residual = validation_prediction - validation_target_delta
        variance = float(
            validation_residual.square()
            .mean()
            .clamp(
                min=minimum_variance,
                max=maximum_variance,
            )
        )

    def regression_metrics(
        prediction: Tensor,
        target: Tensor,
        prefix: str,
    ) -> dict[str, float]:
        residual = prediction - target
        prior_error = target.abs()
        posterior_error = residual.abs()
        return {
            f"{prefix}_prior_mae_mps": float(prior_error.mean()),
            f"{prefix}_proposal_mae_mps": float(posterior_error.mean()),
            f"{prefix}_prior_rmse_mps": float(target.square().mean().sqrt()),
            f"{prefix}_proposal_rmse_mps": float(residual.square().mean().sqrt()),
            f"{prefix}_positive_improvement_rate": float(
                (posterior_error < prior_error).float().mean()
            ),
            f"{prefix}_examples": float(target.numel()),
        }

    proposal = MLPOutgoingVelocityProposal(
        hidden_weights=tuple(float(value) for value in hidden_weights.detach().flatten()),
        hidden_bias=tuple(float(value) for value in hidden_bias.detach()),
        output_weights=tuple(float(value) for value in output_weights.detach()),
        output_bias=float(output_bias.detach()),
        variance=variance,
        maximum_delta=maximum_delta,
    )
    metrics = {
        **regression_metrics(train_prediction, train_target_delta, "train"),
        **regression_metrics(
            validation_prediction,
            validation_target_delta,
            "validation",
        ),
        "calibrated_variance_mps2": variance,
    }
    return proposal, metrics


def fit_mlp_lateral_velocity_intervention(
    train_features: Tensor,
    train_target_delta: Tensor,
    train_prior_variance: Tensor,
    train_confidence: Tensor,
    validation_features: Tensor,
    validation_target_delta: Tensor,
    validation_prior_variance: Tensor,
    validation_confidence: Tensor,
    *,
    hidden_features: int = 12,
    steps: int = 3000,
    learning_rate: float = 0.01,
    weight_decay: float = 5.0e-3,
    gain_sparsity: float = 0.01,
    variance_floor: float = 0.04,
    variance_ceiling: float = 25.0,
    gain_power: float = 2.0,
    maximum_delta: float = 5.0,
    robust_clip_norm: float = 8.0,
    seed: int = 0,
) -> tuple[MLPLateralVelocityIntervention, dict[str, float]]:
    """Fit the actual post-filter lateral correction, including soft abstention."""

    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("intervention features must have shape [samples, features]")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("training and validation feature dimensions must match")
    train_rows = train_features.shape[:1]
    validation_rows = validation_features.shape[:1]
    if any(
        tensor.shape != train_rows
        for tensor in (
            train_target_delta,
            train_prior_variance,
            train_confidence,
        )
    ):
        raise ValueError("training intervention targets must have shape [samples]")
    if any(
        tensor.shape != validation_rows
        for tensor in (
            validation_target_delta,
            validation_prior_variance,
            validation_confidence,
        )
    ):
        raise ValueError("validation intervention targets must have shape [samples]")
    if (
        hidden_features <= 0
        or steps <= 0
        or learning_rate <= 0
        or weight_decay < 0
        or gain_sparsity < 0
        or not 0 < variance_floor <= variance_ceiling
        or gain_power < 1
        or maximum_delta <= 0
        or robust_clip_norm <= 0
    ):
        raise ValueError("invalid lateral intervention fitting settings")

    train_features = train_features.float()
    validation_features = validation_features.float()
    train_target_delta = train_target_delta.float()
    validation_target_delta = validation_target_delta.float()
    train_prior_variance = train_prior_variance.float().clamp_min(1.0e-8)
    validation_prior_variance = validation_prior_variance.float().clamp_min(1.0e-8)
    train_confidence = train_confidence.float().clamp(0.0, 1.0)
    validation_confidence = validation_confidence.float().clamp(0.0, 1.0)
    if train_target_delta.numel() == 0 or validation_target_delta.numel() == 0:
        raise ValueError("intervention training and validation sets cannot be empty")

    # Optimize in standardized coordinates so low-amplitude uncertainty and
    # timing features are not ignored beside the order-one kinematic features.
    # The normalization is folded back into the first layer below, keeping the
    # deployed runtime contract as a plain MLP over the original 19 features.
    feature_mean = train_features.mean(dim=0)
    # Do not amplify nearly constant diagnostics into enormous deployed
    # coefficients. Their small train-set variation is not evidence of a
    # stable causal effect and is especially brittle under new scenarios.
    feature_scale = train_features.std(dim=0).clamp_min(5.0e-2)
    normalized_train_features = (train_features - feature_mean) / feature_scale
    normalized_validation_features = (validation_features - feature_mean) / feature_scale

    generator = torch.Generator().manual_seed(seed)
    hidden_weights = (
        torch.randn(
            hidden_features,
            train_features.shape[1],
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    hidden_bias = torch.zeros(hidden_features, requires_grad=True)
    output_weights = (torch.randn(2, hidden_features, generator=generator) * 0.1).requires_grad_()
    output_bias = torch.tensor([0.0, -2.0], requires_grad=True)
    optimizer = torch.optim.AdamW(
        (hidden_weights, hidden_bias, output_weights, output_bias),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def intervention(
        features: Tensor,
        prior_variance: Tensor,
        confidence: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = F.linear(
            F.silu(F.linear(features, hidden_weights, hidden_bias)),
            output_weights,
            output_bias,
        )
        proposal_delta = output[..., 0].clamp(-maximum_delta, maximum_delta)
        proposal_gain = output[..., 1].sigmoid()
        measurement_variance = (
            variance_floor / proposal_gain.clamp_min(1.0e-4).pow(gain_power)
        ).clamp(max=variance_ceiling)
        total_variance = prior_variance + measurement_variance
        influence = torch.minimum(
            torch.ones_like(proposal_delta),
            proposal_delta.new_tensor(robust_clip_norm)
            / (proposal_delta.abs() / total_variance.sqrt()).clamp_min(1.0e-6),
        )
        filter_gain = prior_variance / total_variance * confidence * influence
        applied_delta = filter_gain * proposal_delta
        return applied_delta, proposal_delta, proposal_gain, measurement_variance

    for _ in range(steps):
        train_applied, _, train_gain, _ = intervention(
            normalized_train_features,
            train_prior_variance,
            train_confidence,
        )
        velocity_loss = F.smooth_l1_loss(train_applied, train_target_delta)
        # The 0.5 s and 1.0 s terms are a local receding-horizon proxy for the
        # position effect of this intervention before another event occurs.
        future_loss = 0.5 * F.smooth_l1_loss(
            0.5 * train_applied,
            0.5 * train_target_delta,
        ) + F.smooth_l1_loss(train_applied, train_target_delta)
        loss = velocity_loss + future_loss + gain_sparsity * train_gain.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_applied, _, train_gain, _ = intervention(
            normalized_train_features,
            train_prior_variance,
            train_confidence,
        )
        validation_applied, _, validation_gain, validation_variance = intervention(
            normalized_validation_features,
            validation_prior_variance,
            validation_confidence,
        )

    def intervention_metrics(
        applied: Tensor,
        target: Tensor,
        gain: Tensor,
        prefix: str,
    ) -> dict[str, float]:
        prior_error = target.abs()
        posterior_error = (applied - target).abs()
        return {
            f"{prefix}_prior_mae_mps": float(prior_error.mean()),
            f"{prefix}_posterior_mae_mps": float(posterior_error.mean()),
            f"{prefix}_prior_rmse_mps": float(target.square().mean().sqrt()),
            f"{prefix}_posterior_rmse_mps": float((applied - target).square().mean().sqrt()),
            f"{prefix}_positive_improvement_rate": float(
                (posterior_error < prior_error).float().mean()
            ),
            f"{prefix}_mean_soft_gain": float(gain.mean()),
            f"{prefix}_examples": float(target.numel()),
        }

    deployed_hidden_weights = hidden_weights.detach() / feature_scale.unsqueeze(0)
    deployed_hidden_bias = hidden_bias.detach() - deployed_hidden_weights @ feature_mean
    intervention_model = MLPLateralVelocityIntervention(
        hidden_weights=tuple(float(value) for value in deployed_hidden_weights.flatten()),
        hidden_bias=tuple(float(value) for value in deployed_hidden_bias),
        output_weights=tuple(float(value) for value in output_weights.detach().flatten()),
        output_bias=(
            float(output_bias.detach()[0]),
            float(output_bias.detach()[1]),
        ),
        variance_floor=variance_floor,
        variance_ceiling=variance_ceiling,
        gain_power=gain_power,
        maximum_delta=maximum_delta,
    )
    metrics = {
        **intervention_metrics(
            train_applied,
            train_target_delta,
            train_gain,
            "train",
        ),
        **intervention_metrics(
            validation_applied,
            validation_target_delta,
            validation_gain,
            "validation",
        ),
        "validation_measurement_variance_mean_mps2": float(validation_variance.mean()),
    }
    return intervention_model, metrics


__all__ = [
    "LinearChangePointGate",
    "MLPChangePointGate",
    "MLPLateralVelocityIntervention",
    "MLPOutgoingVelocityProposal",
    "binary_metrics",
    "fit_linear_change_point_gate",
    "fit_mlp_lateral_velocity_intervention",
    "fit_mlp_change_point_gate",
    "fit_mlp_outgoing_velocity_proposal",
    "select_precision_threshold",
]
