from __future__ import annotations

import pytest
import torch

from world_model.training.losses import (
    balanced_binary_cross_entropy,
    correction_error,
    masked_mean,
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


def test_batch_macro_masked_mean_equalizes_unequal_row_support_and_gradients() -> None:
    value = torch.tensor(
        [
            [10.0, 100.0, 100.0, 100.0],
            [2.0, 2.0, 2.0, 2.0],
            [1000.0, 1000.0, 1000.0, 1000.0],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
            [False, False, False, False],
        ]
    )

    pooled = masked_mean(value, mask)
    macro = masked_mean(value, mask, batch_macro=True)

    torch.testing.assert_close(pooled, torch.tensor(18.0 / 5.0))
    torch.testing.assert_close(macro, torch.tensor(6.0))
    macro.backward()
    torch.testing.assert_close(
        value.grad,
        torch.tensor(
            [
                [0.5, 0.0, 0.0, 0.0],
                [0.125, 0.125, 0.125, 0.125],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_scenario_tail_masked_mean_selects_only_worst_supported_rows() -> None:
    value = torch.tensor(
        [
            [1.0, 1.0],
            [8.0, 8.0],
            [4.0, 4.0],
            [100.0, 100.0],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [True, True],
            [True, True],
            [True, True],
            [False, False],
        ]
    )

    loss = masked_mean(value, mask, batch_tail_fraction=0.5)

    torch.testing.assert_close(loss, torch.tensor(6.0))
    loss.backward()
    torch.testing.assert_close(
        value.grad,
        torch.tensor(
            [
                [0.0, 0.0],
                [0.25, 0.25],
                [0.25, 0.25],
                [0.0, 0.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "value",
    [0.0, -0.1, 1.1, float("inf"), float("nan"), True, "0.25"],
)
def test_scenario_tail_masked_mean_rejects_invalid_fraction(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="batch_tail_fraction"):
        masked_mean(
            torch.ones(2, 1),
            torch.ones(2, 1, dtype=torch.bool),
            batch_tail_fraction=value,  # type: ignore[arg-type]
        )


def test_legacy_masked_mean_false_is_bit_and_gradient_identical() -> None:
    mask = torch.tensor([[True, False, True], [True, True, False]])
    implicit_value = torch.tensor([[1.0, 7.0, 3.0], [5.0, 9.0, 11.0]], requires_grad=True)
    explicit_value = implicit_value.detach().clone().requires_grad_()

    implicit = masked_mean(implicit_value, mask)
    explicit = masked_mean(explicit_value, mask, batch_macro=False)
    implicit.backward()
    explicit.backward()

    assert torch.equal(implicit, explicit)
    assert torch.equal(implicit_value.grad, explicit_value.grad)


def test_legacy_correction_error_is_bit_and_gradient_identical_to_vector_norm() -> None:
    target = torch.tensor([[[0.5, -0.25, 1.0]]])
    implicit_prediction = torch.tensor([[[2.0, -1.0, 4.0]]], requires_grad=True)
    explicit_prediction = implicit_prediction.detach().clone().requires_grad_()

    implicit = correction_error(implicit_prediction, target)
    explicit = torch.linalg.vector_norm(explicit_prediction - target, dim=-1)
    implicit.sum().backward()
    explicit.sum().backward()

    assert torch.equal(implicit, explicit)
    assert torch.equal(implicit_prediction.grad, explicit_prediction.grad)


def test_batch_macro_correction_hinge_omits_unsupported_rows() -> None:
    posterior = torch.tensor(
        [[4.0, 100.0, 100.0, 100.0], [2.0, 2.0, 2.0, 2.0], [50.0] * 4],
        requires_grad=True,
    )
    prior = torch.zeros_like(posterior)
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
            [False, False, False, False],
        ]
    )

    pooled = posterior_improvement_hinge(posterior, prior, mask)
    macro = posterior_improvement_hinge(
        posterior,
        prior,
        mask,
        batch_macro=True,
    )

    torch.testing.assert_close(pooled, torch.tensor(12.0 / 5.0))
    torch.testing.assert_close(macro, torch.tensor(3.0))
    macro.backward()
    assert torch.count_nonzero(posterior.grad[2]) == 0


def test_axiswise_error_exposes_xz_regression_hidden_by_y_improvement() -> None:
    target = torch.zeros(1, 1, 3)
    prior = torch.tensor([[[1.0, 3.0, 1.0]]], requires_grad=True)
    posterior = torch.tensor([[[2.0, 0.0, 2.0]]], requires_grad=True)
    object_support = torch.ones(1, 1, dtype=torch.bool)
    coordinate_support = object_support.unsqueeze(-1).expand_as(posterior)

    legacy = posterior_improvement_hinge(
        correction_error(posterior, target),
        correction_error(prior, target),
        object_support,
    )
    axiswise = posterior_improvement_hinge(
        correction_error(posterior, target, axiswise=True),
        correction_error(prior, target, axiswise=True),
        coordinate_support,
    )

    torch.testing.assert_close(legacy, torch.tensor(0.0))
    torch.testing.assert_close(axiswise, torch.tensor(2.0 / 3.0))
    axiswise.backward()
    torch.testing.assert_close(
        posterior.grad,
        torch.tensor([[[1.0 / 3.0, 0.0, 1.0 / 3.0]]]),
    )
    assert prior.grad is None


def test_balanced_binary_cross_entropy_upweights_rare_positive_events() -> None:
    logits = torch.zeros(4)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.ones(4, dtype=torch.bool)

    loss = balanced_binary_cross_entropy(logits, target, mask)

    # One positive receives weight three and the three negatives weight one.
    assert float(loss) == pytest.approx(1.5 * torch.log(torch.tensor(2.0)).item())


def test_balanced_binary_cross_entropy_scenario_tail_selects_hardest_row() -> None:
    logits = torch.tensor([[-4.0, -4.0], [4.0, 4.0]], requires_grad=True)
    target = torch.ones_like(logits)
    mask = torch.ones_like(logits, dtype=torch.bool)

    loss = balanced_binary_cross_entropy(
        logits,
        target,
        mask,
        batch_tail_fraction=0.5,
    )

    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(4.0)))
    loss.backward()
    assert torch.count_nonzero(logits.grad[0]) == 2
    assert torch.count_nonzero(logits.grad[1]) == 0


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
    with pytest.raises(ValueError, match="batch_tail_fraction"):
        balanced_binary_cross_entropy(
            logits,
            target,
            torch.zeros(2, dtype=torch.bool),
            batch_tail_fraction=0.0,
        )
    with pytest.raises(ValueError, match="matching shapes"):
        posterior_improvement_hinge(
            torch.ones(1),
            torch.ones(2),
            torch.ones(1, dtype=torch.bool),
        )
