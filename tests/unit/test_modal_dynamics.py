from __future__ import annotations

import pytest
import torch

from world_model.dynamics import ModalDynamics


def _modal_tensors(
    *,
    batch: int = 2,
    objects: int = 3,
    modes: int = 4,
    dimensions: int = 2,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.randn(batch, objects, modes, 2, dimensions, dtype=dtype)
    frequency = torch.rand(batch, objects, modes, dimensions, dtype=dtype) * 4
    decay_raw = torch.randn(batch, objects, modes, dimensions, dtype=dtype)
    return state, frequency, decay_raw


def test_modal_identity_and_composition() -> None:
    model = ModalDynamics(4, 2)
    state, frequency, decay = _modal_tensors()
    at_zero = model.evolve(state, frequency, decay, 0.0)
    composed = model.evolve(
        model.evolve(state, frequency, decay, 0.13),
        frequency,
        decay,
        0.27,
    )
    direct = model.evolve(state, frequency, decay, 0.4)

    torch.testing.assert_close(at_zero, state)
    torch.testing.assert_close(composed, direct, atol=2e-6, rtol=2e-6)


def test_positive_decay_never_grows_paired_mode_norm() -> None:
    model = ModalDynamics(4, 2)
    state, frequency, decay = _modal_tensors()
    evolved = model.evolve(state, frequency, decay, 1.0)

    before = torch.linalg.vector_norm(state, dim=-2)
    after = torch.linalg.vector_norm(evolved, dim=-2)
    assert torch.all(after <= before + 1e-6)


def test_constant_modes_are_exactly_preserved() -> None:
    model = ModalDynamics(3, 2, constant_mode_count=1)
    state, frequency, decay = _modal_tensors(modes=3)
    evolved = model.evolve(state, frequency, decay, 20.0)
    torch.testing.assert_close(evolved[:, :, 0], state[:, :, 0])


def test_modal_gradients_and_long_rollout_are_finite() -> None:
    model = ModalDynamics(2, 3).double()
    state, frequency, decay = _modal_tensors(
        batch=1,
        objects=2,
        modes=2,
        dimensions=3,
        dtype=torch.float64,
    )
    state.requires_grad_(True)
    frequency.requires_grad_(True)
    decay.requires_grad_(True)
    evolved = state
    for _ in range(500):
        evolved = model.evolve(evolved, frequency, decay, 0.02)
    loss = evolved.square().mean()
    loss.backward()

    assert torch.isfinite(evolved).all()
    for tensor in (state, frequency, decay):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.device
def test_modal_device_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = ModalDynamics(2, 2).to("mps")
    state, frequency, decay = _modal_tensors(
        batch=1,
        objects=1,
        modes=2,
        dimensions=2,
    )
    output = model.evolve(
        state.to("mps"),
        frequency.to("mps"),
        decay.to("mps"),
        0.1,
    )
    assert output.device.type == "mps"
    assert torch.isfinite(output).all()
