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


__all__ = [
    "LinearChangePointGate",
    "MLPChangePointGate",
    "binary_metrics",
    "fit_linear_change_point_gate",
    "fit_mlp_change_point_gate",
    "select_precision_threshold",
]
