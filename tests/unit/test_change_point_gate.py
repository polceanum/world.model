from __future__ import annotations

import torch

from world_model.training.change_point_gate import (
    binary_metrics,
    fit_linear_change_point_gate,
    fit_mlp_change_point_gate,
    select_precision_threshold,
)


def test_precision_threshold_prefers_conservative_valid_cutoff() -> None:
    logits = torch.tensor([-4.0, -2.0, 0.2, 1.0, 3.0, 4.0])
    targets = torch.tensor([False, False, False, True, True, True])

    threshold, metrics = select_precision_threshold(
        logits,
        targets,
        minimum_precision=1.0,
    )

    assert threshold > 0.5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_linear_gate_learns_uncertainty_aware_separation() -> None:
    generator = torch.Generator().manual_seed(4)
    negative = torch.randn(160, 3, generator=generator) * 0.15
    positive = torch.randn(40, 3, generator=generator) * 0.15
    positive[:, 0] += 1.5
    train_features = torch.cat((negative[:120], positive[:30]))
    train_targets = torch.cat(
        (torch.zeros(120, dtype=torch.bool), torch.ones(30, dtype=torch.bool))
    )
    validation_features = torch.cat((negative[120:], positive[30:]))
    validation_targets = torch.cat(
        (torch.zeros(40, dtype=torch.bool), torch.ones(10, dtype=torch.bool))
    )

    gate, metrics = fit_linear_change_point_gate(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        steps=250,
        minimum_precision=0.9,
    )

    assert len(gate.weights) == 3
    assert metrics["validation_precision"] >= 0.9
    assert metrics["validation_recall"] >= 0.9
    predicted_metrics = binary_metrics(
        gate.logits(validation_features),
        validation_targets,
        threshold=gate.probability_threshold,
    )
    assert predicted_metrics["f1"] >= 0.9


def test_mlp_gate_learns_nonlinear_separation() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.rand(400, 2, generator=generator) * 2.0 - 1.0
    targets = (features[:, 0] * features[:, 1]) > 0.2
    train_features = features[:300]
    train_targets = targets[:300]
    validation_features = features[300:]
    validation_targets = targets[300:]

    gate, metrics = fit_mlp_change_point_gate(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        hidden_features=8,
        steps=600,
        minimum_precision=0.8,
        seed=3,
    )

    assert len(gate.hidden_bias) == 8
    assert len(gate.hidden_weights) == 16
    assert metrics["validation_precision"] >= 0.8
    assert metrics["validation_recall"] >= 0.7
