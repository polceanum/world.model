from __future__ import annotations

import pytest
import torch

from world_model.training.losses import (
    balanced_binary_cross_entropy,
    posterior_improvement_hinge,
)


def test_posterior_improvement_hinge_only_penalises_insufficient_correction() -> None:
    posterior = torch.tensor([[0.4, 0.7, 2.0]], requires_grad=True)
    prior = torch.tensor([[0.8, 0.6, 1.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])

    loss = posterior_improvement_hinge(
        posterior,
        prior,
        mask,
        margin=0.05,
    )

    assert loss.item() == pytest.approx(0.075)
    loss.backward()
    assert posterior.grad is not None
    assert posterior.grad.tolist() == [[0.0, 0.5, 0.0]]
    assert prior.grad is None


def test_posterior_improvement_hinge_validates_inputs() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        posterior_improvement_hinge(
            torch.ones(1),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            margin=-0.1,
        )


def test_balanced_binary_cross_entropy_upweights_rare_positive_events() -> None:
    logits = torch.zeros(4)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.ones(4, dtype=torch.bool)

    loss = balanced_binary_cross_entropy(logits, target, mask)

    # One positive receives weight three and the three negatives weight one.
    assert float(loss) == pytest.approx(1.5 * torch.log(torch.tensor(2.0)).item())


def test_balanced_binary_cross_entropy_handles_empty_and_all_negative_masks() -> None:
    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.zeros(2)

    empty = balanced_binary_cross_entropy(
        logits,
        target,
        torch.zeros(2, dtype=torch.bool),
    )
    negative = balanced_binary_cross_entropy(
        logits,
        target,
        torch.ones(2, dtype=torch.bool),
    )

    assert float(empty.detach()) == 0.0
    assert torch.isfinite(negative)
    with pytest.raises(ValueError, match="at least one"):
        balanced_binary_cross_entropy(
            logits,
            target,
            torch.ones(2, dtype=torch.bool),
            maximum_positive_weight=0.5,
        )
    with pytest.raises(ValueError, match="matching shapes"):
        posterior_improvement_hinge(
            torch.ones(1),
            torch.ones(2),
            torch.ones(1, dtype=torch.bool),
        )
