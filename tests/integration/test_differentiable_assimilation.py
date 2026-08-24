from __future__ import annotations

from dataclasses import replace

import torch

from world_model.runtime import OnlineWorldModel
from world_model.training.loop import run_closed_loop_batch
from world_model.training.trainer import _make_loader, set_closed_loop_trainable_scope
from world_model.utils.config import load_config


def test_real_rgb_causal_batch_backpropagates_through_soft_assimilation() -> None:
    torch.manual_seed(17)
    base = load_config("configs/tiny_overfit.yaml")
    loss_weights = {
        **base.training.loss_weights,
        "soft_association_state": 2.0,
        "soft_association_velocity": 0.5,
        "soft_association_exclusivity": 0.05,
    }
    config = replace(
        base,
        training=replace(
            base.training,
            batch_size=2,
            num_workers=0,
            tbptt_steps=4,
            rollout_anchors_per_window=1,
            closed_loop_trainable_scope="differentiable_state_estimator",
            closed_loop_soft_association_temperature=0.5,
            loss_weights=loss_weights,
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.train()
    set_closed_loop_trainable_scope(model, scope="differentiable_state_estimator")
    batch = next(
        iter(
            _make_loader(
                config,
                split="train",
                episodes=2,
                shuffle=False,
            )
        )
    )

    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_start=3,
        window_steps=4,
        active_trainable_scope="differentiable_state_estimator",
    )

    assert result.metrics["soft_association_supported_coordinate_count"] > 0
    assert "soft_association_state" in result.loss_terms
    assert torch.isfinite(result.total_loss)
    result.total_loss.backward()
    rgb_gradients = [
        parameter.grad
        for parameter in model.observation_modules["rgb"].parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    updater_gradients = [
        parameter.grad
        for parameter in model.updater.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    identifier_gradients = [
        parameter.grad
        for parameter in model.identifier.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert rgb_gradients and all(torch.isfinite(gradient).all() for gradient in rgb_gradients)
    assert updater_gradients and all(
        torch.isfinite(gradient).all() for gradient in updater_gradients
    )
    assert identifier_gradients and all(
        torch.isfinite(gradient).all() for gradient in identifier_gradients
    )
    assert not any(parameter.requires_grad for parameter in model.dynamics.parameters())


def test_disabled_soft_assimilation_does_not_call_trace_api(monkeypatch) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.train()
    batch = next(iter(_make_loader(config, split="train", episodes=1, shuffle=False)))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy disabled path must not request an ingest trace")

    monkeypatch.setattr(model, "ingest_with_trace", forbidden)
    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_start=0,
        window_steps=2,
    )
    assert torch.isfinite(result.total_loss)
