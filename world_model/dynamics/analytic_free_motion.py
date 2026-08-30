"""Parameter-free analytic-only belief dynamics for the RGB-D runtime rung."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace

import torch
from torch import Tensor, nn

from world_model.belief import (
    BeliefTrajectory,
    MotionMode,
    WorldBelief,
    fast_packing_map,
    slow_packing_map,
)
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.rollout import RolloutEngine, RolloutStep


def _stable_drag_coefficients(
    drag: Tensor,
    elapsed: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``decay, A, dA/dlog(k), dB/dlog(k)`` without small-``k t`` loss."""

    scaled_time = drag * elapsed
    small = scaled_time.abs() < 1.0e-2
    safe_scaled_time = torch.where(
        small,
        torch.ones_like(scaled_time),
        scaled_time,
    )
    decay = torch.exp(-scaled_time)
    expm1_negative = torch.expm1(-scaled_time)
    phi_one_regular = -expm1_negative / safe_scaled_time
    phi_two_regular = (scaled_time + expm1_negative) / safe_scaled_time.square()

    x = scaled_time
    phi_one_series = 1.0 + x * (
        -0.5
        + x * (1.0 / 6.0 + x * (-1.0 / 24.0 + x * (1.0 / 120.0 + x * (-1.0 / 720.0 + x / 5040.0))))
    )
    a_log_derivative_series = x * (
        -0.5
        + x * (1.0 / 3.0 + x * (-1.0 / 8.0 + x * (1.0 / 30.0 + x * (-1.0 / 144.0 + x / 840.0))))
    )
    b_log_derivative_series = x * (
        -1.0 / 6.0
        + x
        * (1.0 / 12.0 + x * (-1.0 / 40.0 + x * (1.0 / 180.0 + x * (-1.0 / 1008.0 + x / 6720.0))))
    )

    phi_one = torch.where(small, phi_one_series, phi_one_regular)
    a_log_derivative_factor = torch.where(
        small,
        a_log_derivative_series,
        decay - phi_one_regular,
    )
    b_log_derivative_factor = torch.where(
        small,
        b_log_derivative_series,
        phi_one_regular - 2.0 * phi_two_regular,
    )
    return (
        decay,
        elapsed * phi_one,
        elapsed * a_log_derivative_factor,
        elapsed.square() * b_log_derivative_factor,
    )


def _log_squared(value: Tensor) -> Tensor:
    """Return ``log(value**2)`` with a finite zero-gradient path at zero."""

    nonzero = value != 0.0
    safe_absolute = torch.where(nonzero, value.abs(), torch.ones_like(value))
    finite_value = 2.0 * safe_absolute.log()
    return torch.where(
        nonzero,
        finite_value,
        torch.full_like(finite_value, -torch.inf),
    )


class AnalyticFreeMotionDynamics(nn.Module):
    """Exact gravity-and-linear-drag propagation with no learned parameters.

    This intentionally excludes contacts, learned residuals, event transitions,
    and process-noise inflation.  It is the explicit runtime semantics for a
    contact-free world-model rung whose only trainable path is measurement and
    differentiable temporal estimation.
    """

    def __init__(self, *, propagate_drag_uncertainty: bool = False) -> None:
        super().__init__()
        if not isinstance(propagate_drag_uncertainty, bool):
            raise TypeError("propagate_drag_uncertainty must be boolean")
        self.propagate_drag_uncertainty = propagate_drag_uncertainty
        self.analytic = AnalyticKinematics()
        self.rollout_engine = RolloutEngine()

    def _direct_fast_log_variance(
        self,
        belief: WorldBelief,
        elapsed: Tensor,
    ) -> Tensor:
        """Propagate diagonal marginals directly from ``belief`` at ``[B,T]`` offsets."""

        if elapsed.ndim != 2 or elapsed.shape[0] != belief.batch_size:
            raise ValueError("elapsed must have shape [B,T]")
        objects = belief.objects
        fast_packing = fast_packing_map(objects)
        slow_packing = slow_packing_map(objects)
        position_slice = fast_packing["position"]
        velocity_slice = fast_packing["velocity"]
        drag_slice = slow_packing["log_drag"]

        time = elapsed[:, :, None, None]
        drag = objects.drag.clamp(
            min=self.analytic.min_drag,
            max=self.analytic.max_drag,
        )[:, None, :, :]
        decay, velocity_coefficient, a_log_derivative, b_log_derivative = _stable_drag_coefficients(
            drag, time
        )

        use_drag = drag >= self.analytic.small_drag
        decay = torch.where(use_drag, decay, torch.ones_like(decay))
        velocity_coefficient = torch.where(
            use_drag,
            velocity_coefficient,
            time,
        )
        log_drag_is_unclamped = (
            (objects.log_drag > -16.0)
            & (objects.log_drag < 8.0)
            & (objects.drag > self.analytic.min_drag)
            & (objects.drag < self.analytic.max_drag)
        )[:, None, :, :]
        drag_is_sensitive = use_drag & log_drag_is_unclamped
        a_log_derivative = torch.where(
            drag_is_sensitive,
            a_log_derivative,
            torch.zeros_like(a_log_derivative),
        )
        b_log_derivative = torch.where(
            drag_is_sensitive,
            b_log_derivative,
            torch.zeros_like(b_log_derivative),
        )

        source_velocity = objects.velocity[:, None, :, :]
        gravity = belief.gravity[:, None, None, :]
        position_drag_jacobian = a_log_derivative * source_velocity + b_log_derivative * gravity
        velocity_drag_jacobian = -drag * time * decay * source_velocity + a_log_derivative * gravity
        velocity_drag_jacobian = torch.where(
            drag_is_sensitive,
            velocity_drag_jacobian,
            torch.zeros_like(velocity_drag_jacobian),
        )

        position_log_variance = objects.fast_log_variance[..., position_slice][:, None, :, :]
        velocity_log_variance = objects.fast_log_variance[..., velocity_slice][:, None, :, :]
        drag_log_variance = objects.slow_log_variance[..., drag_slice][:, None, :, :]
        propagated_position = torch.logaddexp(
            position_log_variance,
            _log_squared(velocity_coefficient) + velocity_log_variance,
        )
        propagated_position = torch.logaddexp(
            propagated_position,
            _log_squared(position_drag_jacobian) + drag_log_variance,
        )
        propagated_velocity = torch.logaddexp(
            _log_squared(decay) + velocity_log_variance,
            _log_squared(velocity_drag_jacobian) + drag_log_variance,
        )

        mode = objects.mode
        movable = (
            objects.active & (mode != int(MotionMode.SLEEPING)) & (mode != int(MotionMode.REMOVED))
        )
        propagate = movable[:, None, :, None] & (time > 0.0)
        propagated_position = torch.where(
            propagate,
            propagated_position,
            position_log_variance,
        )
        propagated_velocity = torch.where(
            propagate,
            propagated_velocity,
            velocity_log_variance,
        )

        result = (
            objects.fast_log_variance[:, None, :, :]
            .expand(
                -1,
                elapsed.shape[1],
                -1,
                -1,
            )
            .clone()
        )
        result[..., position_slice] = propagated_position
        result[..., velocity_slice] = propagated_velocity
        return result

    def predict_step(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
    ) -> RolloutStep:
        elapsed = torch.as_tensor(
            dt,
            device=belief.timestamp.device,
            dtype=belief.timestamp.dtype,
        )
        if elapsed.ndim == 0:
            elapsed = elapsed.expand_as(belief.timestamp)
        if elapsed.shape != belief.timestamp.shape:
            raise ValueError("dt must be scalar or have shape [B]")
        after = self.analytic(belief.objects, belief.gravity, elapsed)
        if self.propagate_drag_uncertainty:
            after = after.replace(
                fast_log_variance=self._direct_fast_log_variance(
                    belief,
                    elapsed[:, None],
                )[:, 0],
            )
        propagated = after.active & (elapsed > 0.0).unsqueeze(-1)
        free_logits = after.motion_mode_logits.new_full(
            after.motion_mode_logits.shape,
            -4.0,
        )
        free_logits[..., MotionMode.FREE] = 4.0
        after = after.replace(
            motion_mode_logits=torch.where(
                propagated.unsqueeze(-1),
                free_logits,
                after.motion_mode_logits,
            )
        )
        endpoint = belief.replace(
            timestamp=belief.timestamp + elapsed,
            objects=after,
        )
        event_logits = after.motion_mode_logits.new_full(
            after.motion_mode_logits.shape,
            -4.0,
        )
        event_logits = torch.where(
            propagated.unsqueeze(-1),
            free_logits,
            event_logits,
        )
        return RolloutStep(
            belief=endpoint,
            event_logits=event_logits,
            auxiliary={},
        )

    def predict(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
    ) -> WorldBelief:
        return self.predict_step(belief, dt).belief

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        *,
        return_events: bool = True,
        return_auxiliary: bool = True,
        auxiliary_names: Collection[str] | None = None,
    ) -> BeliefTrajectory:
        if not self.propagate_drag_uncertainty:
            return self.rollout_engine.rollout(
                self.predict_step,
                belief,
                query_times,
                return_events=return_events,
                return_auxiliary=return_auxiliary,
                auxiliary_names=auxiliary_names,
            )
        absolute_offsets = self.rollout_engine._normalise_query_times(
            belief,
            query_times,
        )
        trajectory = self.rollout_engine.rollout(
            self.predict_step,
            belief,
            absolute_offsets,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
            auxiliary_names=auxiliary_names,
        )
        if trajectory.timestamps.shape[1] == 0:
            return trajectory
        return replace(
            trajectory,
            fast_log_variance=self._direct_fast_log_variance(
                belief,
                absolute_offsets,
            ),
        ).validate()
