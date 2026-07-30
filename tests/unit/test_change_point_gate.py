from __future__ import annotations

import torch

from world_model.training.change_point_gate import (
    binary_metrics,
    fit_linear_change_point_gate,
    fit_mlp_change_point_gate,
    fit_mlp_lateral_velocity_intervention,
    fit_mlp_outgoing_velocity_proposal,
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


def test_outgoing_velocity_proposal_improves_nonlinear_delta() -> None:
    generator = torch.Generator().manual_seed(9)
    features = torch.rand(500, 3, generator=generator) * 2.0 - 1.0
    target_delta = 1.5 * features[:, 0] - features[:, 1].square() + 0.4

    proposal, metrics = fit_mlp_outgoing_velocity_proposal(
        features[:400],
        target_delta[:400],
        features[400:],
        target_delta[400:],
        hidden_features=8,
        steps=700,
        seed=2,
    )

    assert len(proposal.hidden_weights) == 24
    assert proposal.variance > 0
    assert metrics["validation_proposal_rmse_mps"] < metrics["validation_prior_rmse_mps"] * 0.35
    assert metrics["validation_positive_improvement_rate"] > 0.85


def test_lateral_intervention_learns_post_filter_correction_and_abstention() -> None:
    generator = torch.Generator().manual_seed(12)
    features = torch.rand(600, 4, generator=generator) * 2.0 - 1.0
    active = features[:, 0] > 0.1
    target_delta = torch.where(
        active,
        1.2 * features[:, 0] - 0.6 * features[:, 1],
        torch.zeros_like(features[:, 0]),
    )
    prior_variance = torch.full((600,), 1.5)
    confidence = torch.full((600,), 0.95)

    intervention, metrics = fit_mlp_lateral_velocity_intervention(
        features[:480],
        target_delta[:480],
        prior_variance[:480],
        confidence[:480],
        features[480:],
        target_delta[480:],
        prior_variance[480:],
        confidence[480:],
        hidden_features=10,
        steps=1200,
        gain_sparsity=0.02,
        seed=5,
    )
    _, gain, variance = intervention.propose(features[480:])

    assert len(intervention.hidden_weights) == 40
    assert len(intervention.output_weights) == 20
    assert metrics["validation_posterior_rmse_mps"] < (metrics["validation_prior_rmse_mps"] * 0.55)
    assert gain[active[480:]].mean() > gain[~active[480:]].mean()
    assert torch.all(variance >= intervention.variance_floor)
    assert torch.all(variance <= intervention.variance_ceiling)
