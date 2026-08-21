from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import pytest
import torch

import world_model.dynamics.analytic as analytic_module
import world_model.dynamics.modal as modal_module
import world_model.dynamics.uncertainty as uncertainty_module
from world_model.belief import BeliefFactory, MotionMode, ObjectBeliefTensor
from world_model.dynamics import (
    AnalyticKinematics,
    DynamicsModel,
    ModalDynamics,
    UncertaintyDynamics,
)


def _active_belief(*, batch_size: int = 2, device: str = "cpu"):
    belief = BeliefFactory(max_objects=2).create(batch_size=batch_size).to(device)
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.tensor([0, 1], device=belief.device)
    objects.position[:, 0] = torch.tensor([-0.45, 1.0, 0.0], device=belief.device)
    objects.position[:, 1] = torch.tensor([0.45, 1.2, 0.1], device=belief.device)
    objects.velocity[:, 0] = torch.tensor([0.2, -0.1, 0.05], device=belief.device)
    objects.velocity[:, 1] = torch.tensor([-0.1, 0.15, -0.02], device=belief.device)
    objects.fast_log_variance.fill_(-2.0)
    objects.log_drag.fill_(-2.5)
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


def _assert_object_tensors_equal(
    left: ObjectBeliefTensor,
    right: ObjectBeliefTensor,
) -> None:
    for item in fields(left):
        assert torch.equal(getattr(left, item.name), getattr(right, item.name)), item.name


def _assert_exact_gradients(
    left_loss: torch.Tensor,
    right_loss: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    left = torch.autograd.grad(left_loss, inputs)
    right = torch.autograd.grad(right_loss, inputs)
    for left_gradient, right_gradient in zip(left, right, strict=True):
        torch.testing.assert_close(left_gradient, right_gradient, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("invalid_dt", [float("nan"), float("inf"), -0.01])
def test_public_dynamics_modules_reject_nonfinite_or_negative_dt(
    invalid_dt: float,
) -> None:
    belief = _active_belief()

    with pytest.raises(ValueError, match="finite nonnegative"):
        AnalyticKinematics()(belief.objects, belief.gravity, invalid_dt)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ModalDynamics(
            belief.objects.modal_count,
            belief.objects.modal_dim,
        )(belief.objects, invalid_dt)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        UncertaintyDynamics(belief.objects.fast_state_dim)(
            belief.objects,
            invalid_dt,
        )
    with pytest.raises(ValueError, match="finite nonnegative"):
        DynamicsModel.from_belief(belief).predict_step(belief, invalid_dt)


def test_public_dynamics_modules_reject_wrong_dt_shape() -> None:
    belief = _active_belief()
    invalid_dt = torch.zeros(2, 1)

    with pytest.raises(ValueError, match=r"scalar or \[B\]"):
        AnalyticKinematics()(belief.objects, belief.gravity, invalid_dt)
    with pytest.raises(ValueError, match=r"scalar or shape \[B\]"):
        ModalDynamics(
            belief.objects.modal_count,
            belief.objects.modal_dim,
        )(belief.objects, invalid_dt)
    with pytest.raises(ValueError, match=r"scalar or \[B\]"):
        UncertaintyDynamics(belief.objects.fast_state_dim)(
            belief.objects,
            invalid_dt,
        )
    with pytest.raises(ValueError, match=r"scalar or shape \[B\]"):
        DynamicsModel.from_belief(belief).predict_step(belief, invalid_dt)


def test_analytic_validated_dt_path_has_exact_output_and_gradient_parity() -> None:
    belief = _active_belief()
    position = belief.objects.position.clone().requires_grad_()
    velocity = belief.objects.velocity.clone().requires_grad_()
    log_drag = belief.objects.log_drag.clone().requires_grad_()
    objects = replace(
        belief.objects,
        position=position,
        velocity=velocity,
        log_drag=log_drag,
    )
    gravity = torch.tensor(
        [[0.1, -9.81, 0.2], [-0.2, -8.0, 0.3]],
        requires_grad=True,
    )
    residual = torch.linspace(-0.2, 0.3, 12).reshape(2, 2, 3).requires_grad_()
    elapsed = torch.tensor([0.013, 0.027])
    model = AnalyticKinematics()

    public = model(
        objects,
        gravity,
        elapsed,
        residual_acceleration=residual,
    )
    internal = model._integrate_validated_dt(
        objects,
        gravity,
        elapsed,
        residual_acceleration=residual,
    )

    _assert_object_tensors_equal(public, internal)
    public_loss = public.position.square().sum() + public.velocity.square().sum()
    internal_loss = internal.position.square().sum() + internal.velocity.square().sum()
    _assert_exact_gradients(
        public_loss,
        internal_loss,
        (position, velocity, log_drag, gravity, residual),
    )


def test_modal_validated_dt_path_has_exact_output_and_gradient_parity() -> None:
    torch.manual_seed(17)
    belief = _active_belief()
    modal_state = belief.objects.modal_state.normal_().requires_grad_()
    modal_frequency = belief.objects.modal_frequency.uniform_(0.1, 3.0).requires_grad_()
    modal_decay_raw = belief.objects.modal_decay_raw.normal_().requires_grad_()
    objects = replace(
        belief.objects,
        modal_state=modal_state,
        modal_frequency=modal_frequency,
        modal_decay_raw=modal_decay_raw,
    )
    elapsed = torch.tensor([0.013, 0.027])
    model = ModalDynamics(objects.modal_count, objects.modal_dim)

    public_objects, public_output = model(objects, elapsed)
    internal_objects, internal_output = model._forward_validated_dt(objects, elapsed)

    _assert_object_tensors_equal(public_objects, internal_objects)
    assert torch.equal(public_output.state, internal_output.state)
    assert torch.equal(
        public_output.residual_acceleration,
        internal_output.residual_acceleration,
    )
    public_loss = public_output.state.square().sum() + (
        public_output.residual_acceleration.square().sum()
    )
    internal_loss = internal_output.state.square().sum() + (
        internal_output.residual_acceleration.square().sum()
    )
    _assert_exact_gradients(
        public_loss,
        internal_loss,
        (
            modal_state,
            modal_frequency,
            modal_decay_raw,
            *tuple(model.parameters()),
        ),
    )


def test_uncertainty_validated_dt_path_has_exact_output_and_gradient_parity() -> None:
    torch.manual_seed(23)
    belief = _active_belief()
    velocity = belief.objects.velocity.clone().requires_grad_()
    visibility = belief.objects.visibility_logit.clone().requires_grad_()
    fast_log_variance = belief.objects.fast_log_variance.clone().requires_grad_()
    objects = replace(
        belief.objects,
        velocity=velocity,
        visibility_logit=visibility,
        fast_log_variance=fast_log_variance,
    )
    event_logits = torch.randn(
        2,
        2,
        len(MotionMode),
        requires_grad=True,
    )
    interaction_density = torch.rand(2, 2, requires_grad=True)
    residual = torch.randn(2, 2, 3, requires_grad=True)
    elapsed = torch.tensor([0.013, 0.027])
    model = UncertaintyDynamics(objects.fast_state_dim)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)

    public = model(
        objects,
        elapsed,
        event_logits=event_logits,
        interaction_density=interaction_density,
        residual_acceleration=residual,
    )
    internal = model._forward_validated_dt(
        objects,
        elapsed,
        event_logits=event_logits,
        interaction_density=interaction_density,
        residual_acceleration=residual,
    )

    _assert_object_tensors_equal(public.objects, internal.objects)
    assert torch.equal(public.process_variance, internal.process_variance)
    public_loss = public.objects.fast_log_variance.square().sum() + (
        public.process_variance.square().sum()
    )
    internal_loss = internal.objects.fast_log_variance.square().sum() + (
        internal.process_variance.square().sum()
    )
    _assert_exact_gradients(
        public_loss,
        internal_loss,
        (
            velocity,
            visibility,
            fast_log_variance,
            event_logits,
            interaction_density,
            residual,
            *tuple(model.parameters()),
        ),
    )


def _assert_composite_uses_one_segment_validation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: str,
) -> None:
    belief = _active_belief(batch_size=1, device=device)
    model = DynamicsModel.from_belief(
        belief,
        max_substep=0.01,
    ).to(device)
    calls = {"dt": 0, "finite": 0, "substep": 0}
    original_normalise = model._normalise_dt
    original_finite = model._validate_finite_segment
    original_substep = model._substep

    def counted_normalise(belief_value: Any, dt: Any) -> torch.Tensor:
        calls["dt"] += 1
        return original_normalise(belief_value, dt)

    def counted_finite(step: Any) -> Any:
        calls["finite"] += 1
        return original_finite(step)

    def counted_substep(*args: Any, **kwargs: Any) -> Any:
        calls["substep"] += 1
        return original_substep(*args, **kwargs)

    def forbidden_child_guard(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a public child guard ran inside the validated hot loop")

    monkeypatch.setattr(model, "_normalise_dt", counted_normalise)
    monkeypatch.setattr(model, "_validate_finite_segment", counted_finite)
    monkeypatch.setattr(model, "_substep", counted_substep)
    monkeypatch.setattr(analytic_module, "_object_dt", forbidden_child_guard)
    monkeypatch.setattr(modal_module, "_modal_dt", forbidden_child_guard)
    monkeypatch.setattr(
        uncertainty_module,
        "clamp_log_variance",
        forbidden_child_guard,
    )

    result = model.predict_step(
        belief,
        torch.tensor([0.04], device=belief.device),
    )

    assert calls == {"dt": 1, "finite": 1, "substep": 4}
    assert torch.isfinite(result.belief.objects.position).all()
    assert torch.isfinite(result.belief.objects.fast_log_variance).all()


def test_composite_dynamics_validates_dt_and_output_once_per_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_composite_uses_one_segment_validation(monkeypatch, device="cpu")


@pytest.mark.device
def test_composite_dynamics_validated_dt_hot_loop_on_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    _assert_composite_uses_one_segment_validation(monkeypatch, device="mps")


def test_composite_segment_boundary_rejects_injected_uncertainty_nan() -> None:
    belief = _active_belief(batch_size=1)
    model = DynamicsModel.from_belief(belief, max_substep=0.01)
    output = model.uncertainty.process_network[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.bias[0] = float("nan")

    with pytest.raises(ValueError, match="cannot clamp a non-finite log variance"):
        model.uncertainty(belief.objects, torch.tensor([0.01]))
    with pytest.raises(ValueError, match="dynamics segment output contains NaN or Inf"):
        model.predict_step(belief, torch.tensor([0.01]))
