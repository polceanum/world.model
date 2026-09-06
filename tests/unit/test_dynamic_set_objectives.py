from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from world_model.training.dynamic_set_objectives import (
    DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
    DynamicsLossInputs,
    PerceptionLossInputs,
    detached_mean_gaussian_nll,
    dynamic_set_objective,
    dynamic_set_objective_lower_bound,
    dynamic_set_objective_regret,
    perception_losses,
)


def test_signed_training_objective_has_nonnegative_screen_regret() -> None:
    lower_bound = dynamic_set_objective_lower_bound()

    assert lower_bound < -0.446
    assert dynamic_set_objective_regret(-0.446) > 0.0
    assert dynamic_set_objective_regret(lower_bound) == 0.0
    with pytest.raises(ValueError, match="below"):
        dynamic_set_objective_regret(lower_bound - 1.0e-3)


def _inputs() -> tuple[PerceptionLossInputs, DynamicsLossInputs]:
    generator = torch.Generator().manual_seed(17)
    mask_logits = torch.randn(2, 9, 5, 5, generator=generator, requires_grad=True)
    target_exists = torch.tensor([[True, True, False, False, False, False, False, False]] * 2)
    target_masks = torch.zeros_like(mask_logits)
    target_masks[:, 1, :2, :2] = 1.0
    target_masks[:, 2, 3:, 3:] = 1.0
    target_masks[:, 0] = 1.0 - target_masks[:, 1:].amax(dim=1)
    metric_position = torch.randn(2, 8, 3, generator=generator, requires_grad=True)
    target_position = torch.randn(2, 8, 3, generator=generator)
    position_log_variance = torch.zeros(2, 8, 3, requires_grad=True)
    appearance = torch.randn(2, 8, 8, generator=generator, requires_grad=True)
    target_appearance = torch.randn(2, 8, 8, generator=generator)
    perception = PerceptionLossInputs(
        mask_logits=mask_logits,
        target_masks=target_masks,
        existence_logits=torch.randn(2, 8, generator=generator, requires_grad=True),
        target_exists=target_exists,
        metric_position=metric_position,
        target_position=target_position,
        position_log_variance=position_log_variance,
        appearance=appearance,
        target_appearance=target_appearance,
    )

    predicted_state = torch.randn(2, 3, 6, 6, generator=generator, requires_grad=True)
    target_state = torch.randn(2, 3, 6, 6, generator=generator)
    contact_window = torch.zeros(2, 3, 6, dtype=torch.bool)
    contact_window[:, 1:, :2] = True
    known_action = torch.zeros_like(contact_window)
    known_action[:, :, 2] = True
    collision_support = torch.ones(2, 3, 6, 6, dtype=torch.bool)
    diagonal = torch.eye(6, dtype=torch.bool).view(1, 1, 6, 6)
    collision_support &= ~diagonal
    dynamics = DynamicsLossInputs(
        predicted_state=predicted_state,
        target_state=target_state,
        process_log_variance=torch.zeros_like(predicted_state, requires_grad=True),
        contact_window_mask=contact_window,
        collision_logits=torch.randn(2, 3, 6, 6, generator=generator, requires_grad=True),
        collision_target=torch.zeros(2, 3, 6, 6, dtype=torch.bool),
        collision_support=collision_support,
        known_action_predictable_mask=known_action,
        relation_residual=torch.randn(2, 3, 6, 3, generator=generator, requires_grad=True),
        unaffected_object_mask=~(contact_window | known_action),
    )
    return perception, dynamics


def test_dynamic_set_objective_contains_only_declared_physical_terms() -> None:
    perception, dynamics = _inputs()
    losses = dynamic_set_objective(perception, dynamics)

    assert tuple(item.name for item in fields(losses)) == (
        "total",
        "mask_dice",
        "mask_bce",
        "existence_focal",
        "metric_position_huber",
        "appearance_cosine",
        "measurement_variance_nll",
        "contact_state_huber",
        "collision_bce",
        "process_variance_nll",
        "known_action_trajectory",
        "unaffected_zero_residual",
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    owners = (
        perception.mask_logits,
        perception.existence_logits,
        perception.metric_position,
        perception.position_log_variance,
        perception.appearance,
        dynamics.predicted_state,
        dynamics.process_log_variance,
        dynamics.collision_logits,
        dynamics.relation_residual,
    )
    for owner in owners:
        assert owner.grad is not None
        assert torch.isfinite(owner.grad).all()


def test_detached_mean_nll_only_gives_variance_gradient() -> None:
    mean = torch.tensor([[[1.0, -2.0, 0.5]]], requires_grad=True)
    target = torch.zeros_like(mean)
    log_variance = torch.zeros_like(mean, requires_grad=True)
    support = torch.ones(1, 1, dtype=torch.bool)

    loss = detached_mean_gaussian_nll(mean, target, log_variance, support)
    mean_gradient, variance_gradient = torch.autograd.grad(
        loss,
        (mean, log_variance),
        allow_unused=True,
    )

    assert mean_gradient is None
    assert variance_gradient is not None
    assert torch.isfinite(variance_gradient).all()


def test_exact_appearance_match_cannot_produce_negative_cosine_loss() -> None:
    perception, _ = _inputs()
    exact = PerceptionLossInputs(
        **{
            **vars(perception),
            "target_appearance": perception.appearance.detach().clone(),
        }
    )

    loss = perception_losses(exact)["appearance_cosine"]

    assert float(loss.detach()) >= 0.0


def test_measurement_calibration_weight_preserves_complete_gradient_ownership() -> None:
    # The analytic sensor prior starts at millimetre variance while an
    # untrained proposal can miss by metres.  The NLL stays mathematically
    # exact; its fixed coefficient prevents that initial scale mismatch from
    # consuming essentially all of a clipped optimizer update.
    assert DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS.measurement_variance_nll == 1.0e-5

    residual = torch.tensor([[[1.0, 2.0, 3.0]]])
    mean = torch.zeros_like(residual)
    log_variance = torch.full_like(residual, torch.log(torch.tensor(0.000064)))
    log_variance.requires_grad_()
    support = torch.ones(1, 1, dtype=torch.bool)
    loss = detached_mean_gaussian_nll(mean, residual, log_variance, support)
    (raw_gradient,) = torch.autograd.grad(loss, (log_variance,))

    weighted_norm = (
        raw_gradient * DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS.measurement_variance_nll
    ).norm()
    assert 0.1 < float(weighted_norm) < 1.0


def test_known_actions_stay_supervised_but_hidden_actuation_can_be_excluded() -> None:
    perception, dynamics = _inputs()
    predictable = dynamics.known_action_predictable_mask.clone()
    predictable[:, -1, 2] = False
    dynamics = DynamicsLossInputs(
        **{
            **vars(dynamics),
            "known_action_predictable_mask": predictable,
        }
    )

    base = dynamic_set_objective(perception, dynamics)
    changed_target = dynamics.target_state.detach().clone()
    changed_target[:, -1, 2] += 1_000.0
    changed = dynamic_set_objective(
        perception,
        DynamicsLossInputs(**{**vars(dynamics), "target_state": changed_target}),
    )

    torch.testing.assert_close(base.known_action_trajectory, changed.known_action_trajectory)
