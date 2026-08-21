from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode, ObjectBeliefTensor, WorldBelief
from world_model.dynamics import DynamicsModel
from world_model.dynamics.rollout import RolloutStep


def _scenario_belief(
    scenario: str,
    *,
    batch_size: int = 2,
    device: str = "cpu",
) -> WorldBelief:
    belief = BeliefFactory(max_objects=2).create(batch_size=batch_size).to(device)
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.tensor([0, 1], device=belief.device)
    objects.fast_log_variance.fill_(-4.0)
    objects.log_drag.fill_(-16.0)
    if scenario == "free":
        objects.position[:, 0] = torch.tensor([-0.45, 1.0, 0.0], device=belief.device)
        objects.position[:, 1] = torch.tensor([0.45, 1.2, 0.1], device=belief.device)
        objects.velocity[:, 0] = torch.tensor([0.2, -0.1, 0.05], device=belief.device)
        objects.velocity[:, 1] = torch.tensor([-0.1, 0.15, -0.02], device=belief.device)
    elif scenario == "pair":
        objects.position[:, 0] = torch.tensor([-0.14, 1.0, 0.0], device=belief.device)
        objects.position[:, 1] = torch.tensor([0.14, 1.0, 0.0], device=belief.device)
        objects.velocity[:, 0, 0] = 1.0
        objects.velocity[:, 1, 0] = -1.0
    elif scenario == "boundary":
        objects.position[:, 0] = torch.tensor([0.84, 1.0, 0.0], device=belief.device)
        objects.position[:, 1] = torch.tensor([-0.45, 1.2, 0.1], device=belief.device)
        objects.velocity[:, 0, 0] = 1.5
        objects.velocity[:, 1, 2] = 0.1
    else:  # pragma: no cover - test helper contract
        raise ValueError(f"unknown scenario {scenario}")
    return replace(
        belief,
        objects=objects,
        gravity=torch.zeros_like(belief.gravity),
        next_object_id=torch.full(
            (batch_size,),
            2,
            dtype=torch.int64,
            device=belief.device,
        ),
    )


def _model(belief: WorldBelief) -> DynamicsModel:
    return DynamicsModel.from_belief(
        belief,
        max_substep=0.01,
        learned_effect_interval_seconds=0.02,
        smooth_event_hazard_enabled=True,
        world_bounds=((-1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)),
    ).to(belief.device)


def _assert_objects_equal(left: ObjectBeliefTensor, right: ObjectBeliefTensor) -> None:
    for item in fields(left):
        assert torch.equal(getattr(left, item.name), getattr(right, item.name)), item.name


def _assert_steps_equal(left: RolloutStep, right: RolloutStep) -> None:
    torch.testing.assert_close(left.belief.timestamp, right.belief.timestamp, rtol=0.0, atol=0.0)
    _assert_objects_equal(left.belief.objects, right.belief.objects)
    assert torch.equal(left.event_logits, right.event_logits)
    assert left.auxiliary.keys() == right.auxiliary.keys()
    for name in left.auxiliary:
        assert torch.equal(left.auxiliary[name], right.auxiliary[name]), name


def _forced_legacy_mask(sub_dt: torch.Tensor) -> torch.Tensor:
    return sub_dt > 0


@pytest.mark.parametrize("scenario", ["free", "pair", "boundary"])
def test_all_positive_fast_path_is_bit_exact_to_legacy_masks(
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    belief = _scenario_belief(scenario)
    model = _model(belief)

    fast = model.predict_step(belief, torch.tensor([0.06, 0.04]))
    monkeypatch.setattr(model, "_optional_advance_mask", _forced_legacy_mask)
    legacy = model.predict_step(belief, torch.tensor([0.06, 0.04]))

    _assert_steps_equal(fast, legacy)
    if scenario == "pair":
        assert fast.auxiliary["pair_collision"].any()
    if scenario == "boundary":
        assert fast.auxiliary["boundary_collision"].any()
    assert fast.auxiliary["learned_effect_evaluation_count"].min() >= 2


def test_all_positive_fast_path_has_exact_gradient_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(37)
    source = _scenario_belief("pair")
    position = source.objects.position.clone().requires_grad_()
    velocity = source.objects.velocity.clone().requires_grad_()
    fast_log_variance = source.objects.fast_log_variance.clone().requires_grad_()
    belief = replace(
        source,
        objects=replace(
            source.objects,
            position=position,
            velocity=velocity,
            fast_log_variance=fast_log_variance,
        ),
    )
    model = _model(belief)

    fast = model.predict_step(belief, torch.tensor([0.06, 0.04]))
    monkeypatch.setattr(model, "_optional_advance_mask", _forced_legacy_mask)
    legacy = model.predict_step(belief, torch.tensor([0.06, 0.04]))
    _assert_steps_equal(fast, legacy)

    def loss(step: RolloutStep) -> torch.Tensor:
        return (
            step.belief.objects.position.square().sum()
            + 0.3 * step.belief.objects.velocity.square().sum()
            + 0.01 * step.belief.objects.fast_log_variance.square().sum()
            + 0.001 * step.event_logits.square().sum()
            + 0.001 * step.auxiliary["residual_acceleration"].square().sum()
            + 0.001 * step.auxiliary["pair_event_logits"].square().sum()
        )

    inputs = (
        position,
        velocity,
        fast_log_variance,
        *tuple(model.parameters()),
    )
    fast_gradients = torch.autograd.grad(loss(fast), inputs, allow_unused=True)
    legacy_gradients = torch.autograd.grad(loss(legacy), inputs, allow_unused=True)
    for fast_gradient, legacy_gradient in zip(
        fast_gradients,
        legacy_gradients,
        strict=True,
    ):
        assert (fast_gradient is None) == (legacy_gradient is None)
        if fast_gradient is not None and legacy_gradient is not None:
            torch.testing.assert_close(
                fast_gradient,
                legacy_gradient,
                rtol=0.0,
                atol=0.0,
            )


def test_segment_selects_none_once_and_reuses_it_for_every_positive_microstep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    belief = _scenario_belief("free", batch_size=1)
    model = _model(belief)
    decisions = 0
    masks: list[torch.Tensor | None] = []
    original_decision = model._optional_advance_mask
    original_substep = model._substep

    def counted_decision(sub_dt: torch.Tensor) -> torch.Tensor | None:
        nonlocal decisions
        decisions += 1
        return original_decision(sub_dt)

    def recording_substep(*args: Any, **kwargs: Any) -> RolloutStep:
        masks.append(kwargs["advance_mask"])
        return original_substep(*args, **kwargs)

    monkeypatch.setattr(model, "_optional_advance_mask", counted_decision)
    monkeypatch.setattr(model, "_substep", recording_substep)

    model.predict_step(belief, torch.tensor([0.04]))

    assert decisions == 1
    assert masks == [None, None, None, None]


def test_none_sentinel_bypasses_object_and_auxiliary_masks() -> None:
    belief = _scenario_belief("free", batch_size=1)
    previous = belief.objects
    updated = previous.clone()
    updated.position.add_(1.0)
    auxiliary = {"value": torch.ones(1, 2, 3)}

    assert DynamicsModel._blend_objects(previous, updated, None) is updated
    assert DynamicsModel._mask_auxiliary(auxiliary, None) is auxiliary


def test_mixed_zero_batch_preserves_legacy_mask_and_zero_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    belief = _scenario_belief("pair")
    model = _model(belief)
    masks: list[torch.Tensor | None] = []
    original_substep = model._substep

    def recording_substep(*args: Any, **kwargs: Any) -> RolloutStep:
        masks.append(kwargs["advance_mask"])
        return original_substep(*args, **kwargs)

    monkeypatch.setattr(model, "_substep", recording_substep)
    result = model.predict_step(belief, torch.tensor([0.0, 0.04]))

    assert len(masks) == 4
    for mask in masks:
        assert mask is not None
        assert torch.equal(mask, torch.tensor([False, True]))
    for item in fields(belief.objects):
        assert torch.equal(
            getattr(result.belief.objects, item.name)[0],
            getattr(belief.objects, item.name)[0],
        ), item.name
    assert result.belief.timestamp[0] == belief.timestamp[0]
    assert result.auxiliary["pair_collision"][0].count_nonzero() == 0
    assert result.auxiliary["pair_impulse"][0].count_nonzero() == 0
    assert (result.event_logits[0, :, MotionMode.COLLISION] < 0).all()


def test_all_zero_segment_does_not_make_positive_path_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    belief = _scenario_belief("free", batch_size=1)
    model = _model(belief)

    def unexpected_decision(sub_dt: torch.Tensor) -> torch.Tensor | None:
        raise AssertionError("zero segment must use the existing zero-step path")

    monkeypatch.setattr(model, "_optional_advance_mask", unexpected_decision)
    result = model.predict_step(belief, torch.tensor([0.0]))

    assert result.belief.timestamp.item() == belief.timestamp.item()
    assert result.auxiliary["pair_collision"].count_nonzero() == 0


@pytest.mark.device
def test_all_positive_fast_path_matches_legacy_masks_and_backpropagates_on_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(41)
    belief = _scenario_belief("pair", batch_size=1, device="mps")
    model = _model(belief)

    fast = model.predict_step(belief, torch.tensor([0.04], device="mps"))
    monkeypatch.setattr(model, "_optional_advance_mask", _forced_legacy_mask)
    legacy = model.predict_step(belief, torch.tensor([0.04], device="mps"))

    _assert_steps_equal(fast, legacy)
    fast_loss = fast.belief.objects.position.square().sum() + (
        fast.belief.objects.fast_log_variance.square().sum()
    )
    legacy_loss = legacy.belief.objects.position.square().sum() + (
        legacy.belief.objects.fast_log_variance.square().sum()
    )
    parameters = tuple(model.parameters())
    fast_gradients = torch.autograd.grad(fast_loss, parameters, allow_unused=True)
    legacy_gradients = torch.autograd.grad(legacy_loss, parameters, allow_unused=True)
    for fast_gradient, legacy_gradient in zip(
        fast_gradients,
        legacy_gradients,
        strict=True,
    ):
        assert (fast_gradient is None) == (legacy_gradient is None)
        if fast_gradient is not None and legacy_gradient is not None:
            torch.testing.assert_close(
                fast_gradient,
                legacy_gradient,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            assert torch.isfinite(fast_gradient).all()
