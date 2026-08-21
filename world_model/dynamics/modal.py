"""Stable continuous rotation-decay modal dynamics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from world_model.belief import ObjectBeliefTensor


def _modal_dt(dt: float | Tensor, state: Tensor) -> Tensor:
    value = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    if value.ndim == 0:
        value = value.expand(state.shape[0])
    if value.shape != (state.shape[0],):
        raise ValueError("modal dt must be scalar or shape [B]")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError("modal dt must be finite and nonnegative")
    return value[:, None, None, None]


@dataclass
class ModalOutput:
    state: Tensor
    residual_acceleration: Tensor


class ModalDynamics(nn.Module):
    """Vectorised bounded mode evolution plus a small learned readout."""

    def __init__(
        self,
        modal_count: int,
        modal_dim: int,
        *,
        constant_mode_count: int = 0,
        min_frequency: float = 0.0,
        max_frequency: float = 40.0,
        max_residual_acceleration: float = 5.0,
    ) -> None:
        super().__init__()
        if modal_count < 0 or modal_dim < 0:
            raise ValueError("modal dimensions must be nonnegative")
        if not 0 <= constant_mode_count <= modal_count:
            raise ValueError("constant_mode_count must be within modal_count")
        self.modal_count = modal_count
        self.modal_dim = modal_dim
        self.constant_mode_count = constant_mode_count
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.max_residual_acceleration = max_residual_acceleration
        input_dim = modal_count * 2 * modal_dim
        self.readout: nn.Module
        if input_dim == 0:
            self.readout = _ZeroReadout()
        else:
            self.readout = nn.Linear(input_dim, 3)
            nn.init.normal_(self.readout.weight, std=1e-3)
            nn.init.zeros_(self.readout.bias)

    def evolve(
        self,
        state: Tensor,
        frequency: Tensor,
        decay_raw: Tensor,
        dt: float | Tensor,
    ) -> Tensor:
        self._validate_state_shapes(state, frequency, decay_raw)
        delta_time = _modal_dt(dt, state)
        return self._evolve_with_modal_dt(
            state,
            frequency,
            decay_raw,
            delta_time,
        )

    def _evolve_validated_dt(
        self,
        state: Tensor,
        frequency: Tensor,
        decay_raw: Tensor,
        dt: Tensor,
    ) -> Tensor:
        """Evolve using a parent-validated ``[B]`` elapsed-time tensor."""

        self._validate_state_shapes(state, frequency, decay_raw)
        if dt.shape != (state.shape[0],) or dt.device != state.device or dt.dtype != state.dtype:
            raise ValueError("validated modal dt must match state batch, device, and dtype")
        return self._evolve_with_modal_dt(
            state,
            frequency,
            decay_raw,
            dt[:, None, None, None],
        )

    def _validate_state_shapes(
        self,
        state: Tensor,
        frequency: Tensor,
        decay_raw: Tensor,
    ) -> None:
        if state.ndim != 5 or state.shape[-2] != 2:
            raise ValueError("modal state must have shape [B,N,K,2,Dm]")
        expected = (*state.shape[:3], state.shape[-1])
        if frequency.shape != expected or decay_raw.shape != expected:
            raise ValueError("frequency/decay must have shape [B,N,K,Dm]")
        if state.shape[-3:] != (self.modal_count, 2, self.modal_dim):
            raise ValueError("modal state does not match configured dimensions")

    def _evolve_with_modal_dt(
        self,
        state: Tensor,
        frequency: Tensor,
        decay_raw: Tensor,
        delta_time: Tensor,
    ) -> Tensor:
        """Implement modal evolution for normalized ``[B,1,1,1]`` time."""

        frequency = frequency.clamp(
            min=self.min_frequency,
            max=self.max_frequency,
        )
        angle = frequency * delta_time
        decay_rate = F.softplus(decay_raw)
        decay = torch.exp(-decay_rate * delta_time)
        if self.constant_mode_count:
            constant = (
                torch.arange(self.modal_count, device=state.device) < self.constant_mode_count
            )
            constant = constant.view(1, 1, self.modal_count, 1)
            angle = torch.where(constant, torch.zeros_like(angle), angle)
            decay = torch.where(constant, torch.ones_like(decay), decay)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        x, y = state.unbind(dim=-2)
        x_new = decay * (cosine * x - sine * y)
        y_new = decay * (sine * x + cosine * y)
        return torch.stack((x_new, y_new), dim=-2)

    def acceleration(self, state: Tensor) -> Tensor:
        if self.modal_count == 0 or self.modal_dim == 0:
            return state.new_zeros(*state.shape[:2], 3)
        value = self.readout(state.flatten(start_dim=2))
        return self.max_residual_acceleration * torch.tanh(value)

    def forward(
        self,
        objects: ObjectBeliefTensor,
        dt: float | Tensor,
    ) -> tuple[ObjectBeliefTensor, ModalOutput]:
        state = self.evolve(
            objects.modal_state,
            objects.modal_frequency,
            objects.modal_decay_raw,
            dt,
        )
        state = torch.where(
            objects.active[..., None, None, None],
            state,
            objects.modal_state,
        )
        output = ModalOutput(
            state=state,
            residual_acceleration=self.acceleration(state) * objects.active.unsqueeze(-1),
        )
        return replace(objects, modal_state=state), output

    def _forward_validated_dt(
        self,
        objects: ObjectBeliefTensor,
        dt: Tensor,
    ) -> tuple[ObjectBeliefTensor, ModalOutput]:
        """Apply modal evolution after parent-level elapsed-time validation."""

        state = self._evolve_validated_dt(
            objects.modal_state,
            objects.modal_frequency,
            objects.modal_decay_raw,
            dt,
        )
        state = torch.where(
            objects.active[..., None, None, None],
            state,
            objects.modal_state,
        )
        output = ModalOutput(
            state=state,
            residual_acceleration=self.acceleration(state) * objects.active.unsqueeze(-1),
        )
        return replace(objects, modal_state=state), output


class _ZeroReadout(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value.new_zeros(*value.shape[:-1], 3)
