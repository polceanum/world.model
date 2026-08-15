from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.dynamics import DynamicsModel, pair_applicability
from world_model.training.checkpointing import (
    validate_checkpoint_config,
    validate_training_resume_config,
)
from world_model.utils.config import load_config
from world_model.utils.version import SIMULATOR_VERSION


def _free_belief(*, batch_size: int = 1):
    belief = BeliefFactory(max_objects=2).create(batch_size=batch_size)
    objects = belief.objects.clone()
    objects.active[:, 0] = True
    objects.object_id[:, 0] = torch.arange(batch_size)
    objects.position[:, 0] = torch.tensor([0.0, 2.0, 0.0])
    objects.velocity[:, 0] = torch.tensor([0.4, 0.0, -0.2])
    objects.geometry[:, 0, 0] = 0.1
    objects.log_drag[:, 0] = -16.0
    return replace(
        belief,
        objects=objects,
        gravity=torch.zeros_like(belief.gravity),
        next_object_id=torch.ones(batch_size, dtype=torch.int64),
    )


def _collision_belief():
    belief = BeliefFactory(max_objects=2).create()
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.tensor([0, 1])
    objects.position[0, 0] = torch.tensor([-0.11, 2.0, 0.0])
    objects.position[0, 1] = torch.tensor([0.11, 2.0, 0.0])
    objects.velocity[0, 0, 0] = 1.0
    objects.velocity[0, 1, 0] = -1.0
    objects.geometry[..., 0] = 0.1
    objects.log_drag.fill_(-16.0)
    return replace(
        belief,
        objects=objects,
        gravity=torch.zeros_like(belief.gravity),
        next_object_id=torch.full((1,), 2, dtype=torch.int64),
    )


def _model(
    belief,
    *,
    interval: float | None,
    attention: bool = False,
    interaction_radius: float = 0.5,
    pair_applicability_enabled: bool = False,
):
    return DynamicsModel.from_belief(
        belief,
        max_substep=0.01,
        learned_effect_interval_seconds=interval,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=16,
        interaction_radius=interaction_radius,
        pair_applicability_enabled=pair_applicability_enabled,
        attention_residual_enabled=attention,
        attention_width=16,
        attention_heads=4,
        attention_layers=1,
        attention_feed_forward_width=32,
    )


def _assert_object_means_equal(first, second) -> None:
    for item in fields(first):
        first_value = getattr(first, item.name)
        second_value = getattr(second, item.name)
        if first_value.dtype.is_floating_point:
            torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)
        else:
            assert torch.equal(first_value, second_value)


def test_default_and_one_tick_interval_have_exact_forward_parity() -> None:
    torch.manual_seed(9)
    belief = _free_belief()
    historical = _model(belief, interval=None, attention=True).eval()
    one_tick = _model(belief, interval=0.01, attention=True).eval()
    one_tick.load_state_dict(historical.state_dict())

    with torch.no_grad():
        historical_step = historical.predict_step(belief, 0.07)
        one_tick_step = one_tick.predict_step(belief, 0.07)

    _assert_object_means_equal(historical_step.belief.objects, one_tick_step.belief.objects)
    torch.testing.assert_close(
        historical_step.event_logits,
        one_tick_step.event_logits,
        rtol=0.0,
        atol=0.0,
    )
    assert historical_step.auxiliary["learned_effect_evaluation_count"].item() == 7
    assert one_tick_step.auxiliary["learned_effect_evaluation_count"].item() == 7


def test_coarse_interval_reduces_graph_and_attention_calls_exactly() -> None:
    belief = _free_belief()
    model = _model(belief, interval=0.04, attention=True).eval()
    calls = {"graph": 0, "attention": 0}

    def count_graph(*_args) -> None:
        calls["graph"] += 1

    def count_attention(*_args) -> None:
        calls["attention"] += 1

    graph_handle = model.interactions.register_forward_hook(count_graph)
    assert model.attention_interactions is not None
    attention_handle = model.attention_interactions.register_forward_hook(count_attention)
    try:
        with torch.no_grad():
            step = model.predict_step(belief, 0.10)
    finally:
        graph_handle.remove()
        attention_handle.remove()

    assert calls == {"graph": 3, "attention": 3}
    assert step.auxiliary["learned_effect_evaluation_count"].item() == 3


def test_collision_invalidates_held_effect_before_the_next_microstep() -> None:
    belief = _collision_belief()
    model = _model(belief, interval=0.10).eval()
    calls = 0

    def count_graph(*_args) -> None:
        nonlocal calls
        calls += 1

    handle = model.interactions.register_forward_hook(count_graph)
    try:
        with torch.no_grad():
            step = model.predict_step(belief, 0.03)
    finally:
        handle.remove()

    assert step.auxiliary["pair_collision"][0, 0, 1]
    assert step.event_logits[0, :, MotionMode.COLLISION].gt(0).all()
    # The first tick collides, so its proposal cannot be reused on tick two.
    assert calls == 2
    assert step.auxiliary["learned_effect_evaluation_count"].item() == 2


@pytest.mark.parametrize("entering", [True, False])
def test_edge_set_change_recomputes_complete_effect_before_substep(entering: bool) -> None:
    belief = _collision_belief()
    objects = belief.objects.clone()
    objects.fast_log_variance[..., :3] = -20.0
    if entering:
        objects.position[0, 0, 0] = -0.30
        objects.position[0, 1, 0] = 0.30
        objects.velocity[0, 0, 0] = 2.0
        objects.velocity[0, 1, 0] = -2.0
    else:
        objects.position[0, 0, 0] = -0.15
        objects.position[0, 1, 0] = 0.15
        objects.velocity[0, 0, 0] = -1.0
        objects.velocity[0, 1, 0] = 1.0
    belief = replace(belief, objects=objects)
    model = _model(
        belief,
        interval=0.10,
        attention=True,
        interaction_radius=0.20,
    ).eval()
    model.interactions.uncertainty_margin_scale = 0.0
    evaluated_masks: list[torch.Tensor] = []

    def record_mask(_module, _inputs, output) -> None:
        evaluated_masks.append(output.edge_mask.detach().clone())

    handle = model.interactions.register_forward_hook(record_mask)
    try:
        with torch.no_grad():
            step = model.predict_step(belief, 0.08)
    finally:
        handle.remove()

    assert len(evaluated_masks) == 2
    assert bool(evaluated_masks[0][0, 0, 1]) is (not entering)
    assert bool(evaluated_masks[1][0, 0, 1]) is entering
    assert not step.auxiliary["pair_collision"].any()
    assert step.auxiliary["learned_effect_evaluation_count"].item() == 2


def test_pair_applicability_refreshes_each_tick_while_raw_effect_is_held() -> None:
    belief = _collision_belief()
    objects = belief.objects.clone()
    objects.position[0, 0, 0] = -0.20
    objects.position[0, 1, 0] = 0.20
    objects.velocity[0, 0, 0] = 0.50
    objects.velocity[0, 1, 0] = -0.50
    objects.fast_log_variance.fill_(-12.0)
    belief = replace(belief, objects=objects)
    model = _model(
        belief,
        interval=0.10,
        interaction_radius=0.5,
        pair_applicability_enabled=True,
    ).eval()
    initial_mask = model.interactions.candidate_edge_mask(belief.objects)
    initial = pair_applicability(
        belief.objects,
        initial_mask,
        model.pair_applicability_config,
    )

    with torch.no_grad():
        step = model.predict_step(belief, 0.04)

    # The broad candidate edge remains present, so the expensive proposal is
    # evaluated once. Its cheap causal envelope nevertheless opens as the pair
    # closes over four analytic microsteps.
    assert step.auxiliary["learned_effect_evaluation_count"].item() == 1
    assert step.auxiliary["pair_applicability"][0, 0, 1] > initial.pair[0, 0, 1]


def test_zero_learned_output_keeps_collision_endpoint_and_uncertainty_exact() -> None:
    torch.manual_seed(3)
    belief = _collision_belief()
    every_tick = _model(belief, interval=None, attention=True).eval()
    with torch.no_grad():
        for parameter in every_tick.interactions.parameters():
            parameter.zero_()
        if isinstance(every_tick.modal.readout, torch.nn.Linear):
            every_tick.modal.readout.weight.zero_()
            every_tick.modal.readout.bias.zero_()
    coarse = _model(belief, interval=0.05, attention=True).eval()
    coarse.load_state_dict(every_tick.state_dict())

    with torch.no_grad():
        reference = every_tick.predict_step(belief, 0.06)
        candidate = coarse.predict_step(belief, 0.06)

    _assert_object_means_equal(reference.belief.objects, candidate.belief.objects)
    torch.testing.assert_close(reference.event_logits, candidate.event_logits, rtol=0.0, atol=0.0)
    for name in (
        "pair_contact",
        "interval_pair_contact",
        "pair_collision",
        "boundary_contact",
        "boundary_collision",
        "ground_contact",
        "ground_collision",
        "pair_impulse",
        "process_variance",
    ):
        first = reference.auxiliary[name]
        second = candidate.auxiliary[name]
        if first.dtype.is_floating_point:
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        else:
            assert torch.equal(first, second)
    assert (
        candidate.auxiliary["learned_effect_evaluation_count"].item()
        < reference.auxiliary["learned_effect_evaluation_count"].item()
    )


def test_held_effect_accumulates_finite_nonzero_gradient() -> None:
    belief = _free_belief()
    model = _model(belief, interval=0.04, attention=True).train()

    step = model.predict_step(belief, 0.08)
    loss = step.belief.objects.position[0, 0, 0] + step.belief.objects.velocity[0, 0, 0]
    loss.backward()

    gradient = model.interactions.node_network.output.bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert model.attention_interactions is not None
    attention_gradient = model.attention_interactions.node_decoder.bias.grad
    assert attention_gradient is not None
    assert torch.isfinite(attention_gradient).all()
    assert torch.count_nonzero(attention_gradient) > 0
    assert step.auxiliary["learned_effect_evaluation_count"].item() == 2


def test_held_reuse_matches_repeated_state_invariant_effect_gradient() -> None:
    belief = _free_belief()
    every_tick = _model(belief, interval=None, attention=True).train()
    with torch.no_grad():
        if isinstance(every_tick.modal.readout, torch.nn.Linear):
            every_tick.modal.readout.weight.zero_()
            every_tick.modal.readout.bias.zero_()
        for parameter in every_tick.interactions.parameters():
            parameter.zero_()
        every_tick.interactions.node_network.output.bias.copy_(
            torch.atanh(torch.tensor([0.10, -0.05, 0.025]))
        )
        assert every_tick.attention_interactions is not None
        for parameter in every_tick.attention_interactions.parameters():
            parameter.zero_()
        every_tick.attention_interactions.node_decoder.bias.copy_(
            torch.atanh(torch.tensor([0.04, 0.02, -0.03]))
        )
    held = _model(belief, interval=0.04, attention=True).train()
    held.load_state_dict(every_tick.state_dict())

    reference = every_tick.predict_step(belief, 0.04)
    candidate = held.predict_step(belief, 0.04)
    reference_loss = (
        reference.belief.objects.position[0, 0].square().sum()
        + 0.5 * reference.belief.objects.velocity[0, 0].square().sum()
    )
    candidate_loss = (
        candidate.belief.objects.position[0, 0].square().sum()
        + 0.5 * candidate.belief.objects.velocity[0, 0].square().sum()
    )
    reference_loss.backward()
    candidate_loss.backward()

    torch.testing.assert_close(
        candidate.belief.objects.position,
        reference.belief.objects.position,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate.belief.objects.velocity,
        reference.belief.objects.velocity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        held.interactions.node_network.output.bias.grad,
        every_tick.interactions.node_network.output.bias.grad,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert held.attention_interactions is not None
    assert every_tick.attention_interactions is not None
    torch.testing.assert_close(
        held.attention_interactions.node_decoder.bias.grad,
        every_tick.attention_interactions.node_decoder.bias.grad,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert candidate.auxiliary["learned_effect_evaluation_count"].item() == 1
    assert reference.auxiliary["learned_effect_evaluation_count"].item() == 4


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_mps_held_attention_effect_is_finite_and_differentiable() -> None:
    belief = _free_belief().to("mps")
    model = _model(belief, interval=0.04, attention=True).to("mps").train()

    step = model.predict_step(belief, 0.08)
    loss = step.belief.objects.position[0, 0].square().sum()
    loss.backward()

    assert torch.isfinite(step.belief.objects.position).all()
    assert step.auxiliary["learned_effect_evaluation_count"].item() == 2
    assert model.attention_interactions is not None
    gradient = model.attention_interactions.node_decoder.bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_batch_elapsed_and_rollout_segments_keep_exact_timestamps_and_local_cadence() -> None:
    belief = _free_belief(batch_size=2)
    belief = replace(belief, timestamp=torch.tensor([1.0, 2.0]))
    model = _model(belief, interval=0.04).eval()

    with torch.no_grad():
        step = model.predict_step(belief, torch.tensor([0.0, 0.055]))
        trajectory = model.rollout(
            belief,
            torch.tensor([[0.02, 0.05], [0.03, 0.06]]),
        )

    torch.testing.assert_close(step.belief.timestamp, torch.tensor([1.0, 2.055]))
    for item in fields(belief.objects):
        actual = getattr(step.belief.objects, item.name)[0]
        expected = getattr(belief.objects, item.name)[0]
        if actual.dtype.is_floating_point:
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        else:
            assert torch.equal(actual, expected)
    torch.testing.assert_close(
        trajectory.timestamps,
        torch.tensor([[1.02, 1.05], [2.03, 2.06]]),
    )
    # Each query segment starts with a fresh local proposal; no effect tensor
    # is cached across ``predict_step`` calls or rollout query boundaries.
    assert torch.equal(
        trajectory.auxiliary["learned_effect_evaluation_count"],
        torch.ones(2, 2, dtype=torch.int64),
    )


def test_multirate_config_is_explicit_validated_and_legacy_checkpoint_compatible() -> None:
    config = load_config("configs/toy_smoke.yaml")
    assert config.model.dynamics.learned_effect_interval_seconds is None

    enabled = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(
                config.model.dynamics,
                learned_effect_interval_seconds=0.05,
            ),
        ),
    )
    enabled.validate()
    invalid = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(
                config.model.dynamics,
                learned_effect_interval_seconds=0.001,
            ),
        ),
    )
    with pytest.raises(ValueError, match="no smaller than max_substep"):
        invalid.validate()

    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    legacy_payload = deepcopy(payload)
    legacy_payload["config"]["model"]["dynamics"].pop("learned_effect_interval_seconds")
    validate_checkpoint_config(legacy_payload, config)
    validate_training_resume_config(legacy_payload, config)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, enabled)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, enabled)
    with pytest.raises(ValueError, match="model.dynamics.learned_effect_interval_seconds"):
        validate_training_resume_config(payload, enabled)
