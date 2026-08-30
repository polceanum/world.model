"""Parameter-free analytic-only belief dynamics for the RGB-D runtime rung."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace

import torch
from torch import Tensor, nn

from world_model.belief import BeliefTrajectory, MotionMode, WorldBelief
from world_model.dynamics.actions import WorldImpulseAction
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.rollout import RolloutEngine, RolloutStep

_KNOWN_ACTION_AUXILIARIES = frozenset(
    {
        "known_action_applied",
        "known_impulse_world",
    }
)


class AnalyticFreeMotionDynamics(nn.Module):
    """Exact gravity-and-linear-drag propagation with no learned parameters.

    This intentionally excludes contacts, learned residuals, event transitions,
    and process-noise inflation.  It is the explicit runtime semantics for a
    contact-free world-model rung whose only trainable path is measurement and
    differentiable temporal estimation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.analytic = AnalyticKinematics()
        self.rollout_engine = RolloutEngine()

    def predict_step(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
        *,
        action: WorldImpulseAction | None = None,
    ) -> RolloutStep:
        if action is None:
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

        if not isinstance(action, WorldImpulseAction):
            raise TypeError("action must be a WorldImpulseAction or None")
        elapsed = self._normalise_elapsed_time(belief, dt)
        target_mask = action.validate_for(
            belief,
            latest_timestamp=belief.timestamp + elapsed,
        )
        ordinary = self.predict_step(belief, elapsed)
        absolute_times = (belief.timestamp + elapsed).unsqueeze(-1)
        position_delta, velocity_delta = self._action_response(
            belief,
            absolute_times,
            action,
            target_mask,
        )
        applied, known_impulse = self._action_interval_auxiliaries(
            belief,
            absolute_times,
            action,
            target_mask,
        )
        objects = ordinary.belief.objects.replace(
            position=ordinary.belief.objects.position + position_delta[:, 0],
            velocity=ordinary.belief.objects.velocity + velocity_delta[:, 0],
        )
        return RolloutStep(
            belief=ordinary.belief.replace(objects=objects),
            event_logits=self._with_action_events(ordinary.event_logits, applied[:, 0]),
            auxiliary={
                "known_action_applied": applied[:, 0],
                "known_impulse_world": known_impulse[:, 0],
            },
        )

    def predict(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
        *,
        action: WorldImpulseAction | None = None,
    ) -> WorldBelief:
        if action is None:
            return self.predict_step(belief, dt).belief
        return self.predict_step(belief, dt, action=action).belief

    def validate_action_rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        action: WorldImpulseAction | None,
    ) -> Tensor:
        """Purely normalize queries and validate one optional action.

        The returned tensor contains ``[B,T]`` nonnegative offsets.  Planners
        can call this method for every candidate before any propagation, so an
        invalid candidate cannot produce a partial counterfactual result.
        """

        offsets = self.rollout_engine._normalise_query_times(belief, query_times)
        if action is None:
            return offsets
        if not isinstance(action, WorldImpulseAction):
            raise TypeError("action must be a WorldImpulseAction or None")
        if offsets.shape[1] == 0:
            raise ValueError("an action rollout requires at least one query time")
        action.validate_for(
            belief,
            latest_timestamp=belief.timestamp + offsets[:, -1],
        )
        return offsets

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        *,
        action: WorldImpulseAction | None = None,
        return_events: bool = True,
        return_auxiliary: bool = True,
        auxiliary_names: Collection[str] | None = None,
    ) -> BeliefTrajectory:
        if action is None:
            return self.rollout_engine.rollout(
                self.predict_step,
                belief,
                query_times,
                return_events=return_events,
                return_auxiliary=return_auxiliary,
                auxiliary_names=auxiliary_names,
            )

        offsets = self.validate_action_rollout(belief, query_times, action)
        selected_auxiliary = self._validate_action_auxiliary_request(
            return_auxiliary=return_auxiliary,
            auxiliary_names=auxiliary_names,
        )
        ordinary = self.rollout_engine.rollout(
            self.predict_step,
            belief,
            offsets,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
            auxiliary_names=() if return_auxiliary else None,
        )
        target_mask = action.validate_for(
            belief,
            latest_timestamp=belief.timestamp + offsets[:, -1],
        )
        absolute_times = belief.timestamp.unsqueeze(-1) + offsets
        position_delta, velocity_delta = self._action_response(
            belief,
            absolute_times,
            action,
            target_mask,
        )
        applied, known_impulse = self._action_interval_auxiliaries(
            belief,
            absolute_times,
            action,
            target_mask,
        )
        auxiliary = dict(ordinary.auxiliary)
        if return_auxiliary and (
            selected_auxiliary is None or "known_action_applied" in selected_auxiliary
        ):
            auxiliary["known_action_applied"] = applied
        if return_auxiliary and (
            selected_auxiliary is None or "known_impulse_world" in selected_auxiliary
        ):
            auxiliary["known_impulse_world"] = known_impulse
        trajectory = replace(
            ordinary,
            positions=ordinary.positions + position_delta,
            velocities=ordinary.velocities + velocity_delta,
            event_logits=(
                self._with_action_events(ordinary.event_logits, applied)
                if ordinary.event_logits is not None
                else None
            ),
            auxiliary=auxiliary,
        )
        return trajectory.validate()

    @staticmethod
    def _normalise_elapsed_time(belief: WorldBelief, dt: float | Tensor) -> Tensor:
        elapsed = torch.as_tensor(dt, device=belief.device, dtype=belief.dtype)
        if elapsed.ndim == 0:
            elapsed = elapsed.expand_as(belief.timestamp)
        if elapsed.shape != belief.timestamp.shape:
            raise ValueError("dt must be scalar or have shape [B]")
        if not torch.isfinite(elapsed).all() or torch.any(elapsed < 0):
            raise ValueError("dt must contain finite nonnegative seconds")
        return elapsed

    def _action_response(
        self,
        belief: WorldBelief,
        absolute_times: Tensor,
        action: WorldImpulseAction,
        target_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        elapsed = (absolute_times - action.timestamp.unsqueeze(-1)).clamp_min(0.0)
        reached = absolute_times >= action.timestamp.unsqueeze(-1)
        object_elapsed = elapsed.unsqueeze(-1).unsqueeze(-1)
        target = target_mask.unsqueeze(1).unsqueeze(-1)
        reached_target = reached.unsqueeze(-1).unsqueeze(-1) & target

        velocity_jump = action.impulse_world.unsqueeze(1) / belief.objects.mass
        velocity_jump = velocity_jump.unsqueeze(1)
        drag = belief.objects.drag.clamp(
            min=self.analytic.min_drag,
            max=self.analytic.max_drag,
        ).unsqueeze(1)
        decay = torch.exp(-drag * object_elapsed)
        one_minus_decay = -torch.expm1(-drag * object_elapsed)
        safe_drag = drag.clamp_min(self.analytic.small_drag)
        use_drag = drag >= self.analytic.small_drag
        displacement_factor = torch.where(
            use_drag,
            one_minus_decay / safe_drag,
            object_elapsed,
        )
        velocity_factor = torch.where(use_drag, decay, torch.ones_like(decay))
        position_delta = torch.where(
            reached_target,
            velocity_jump * displacement_factor,
            torch.zeros_like(velocity_jump * displacement_factor),
        )
        velocity_delta = torch.where(
            reached_target,
            velocity_jump * velocity_factor,
            torch.zeros_like(velocity_jump * velocity_factor),
        )
        return position_delta, velocity_delta

    @staticmethod
    def _action_interval_auxiliaries(
        belief: WorldBelief,
        absolute_times: Tensor,
        action: WorldImpulseAction,
        target_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        previous_times = torch.cat(
            (belief.timestamp.unsqueeze(-1), absolute_times[:, :-1]),
            dim=1,
        )
        owns_action = (previous_times < action.timestamp.unsqueeze(-1)) & (
            action.timestamp.unsqueeze(-1) <= absolute_times
        )
        nonzero = torch.any(action.impulse_world != 0.0, dim=-1)
        applied = (
            owns_action.unsqueeze(-1)
            & nonzero.unsqueeze(-1).unsqueeze(-1)
            & target_mask.unsqueeze(1)
        )
        impulse = action.impulse_world[:, None, None, :].expand(
            -1,
            absolute_times.shape[1],
            belief.objects.max_objects,
            -1,
        )
        known_impulse = torch.where(
            applied.unsqueeze(-1),
            impulse,
            torch.zeros_like(impulse),
        )
        return applied, known_impulse

    @staticmethod
    def _with_action_events(event_logits: Tensor, applied: Tensor) -> Tensor:
        actuated_logits = event_logits.new_full(event_logits.shape, -4.0)
        actuated_logits[..., MotionMode.EXTERNALLY_ACTUATED] = 4.0
        return torch.where(applied.unsqueeze(-1), actuated_logits, event_logits)

    @staticmethod
    def _validate_action_auxiliary_request(
        *,
        return_auxiliary: bool,
        auxiliary_names: Collection[str] | None,
    ) -> frozenset[str] | None:
        if auxiliary_names is not None and not return_auxiliary:
            raise ValueError("auxiliary_names requires return_auxiliary=True")
        if auxiliary_names is None:
            return None
        selected = frozenset(auxiliary_names)
        if any(not isinstance(name, str) or not name for name in selected):
            raise ValueError("auxiliary_names must contain nonempty strings")
        missing = selected - _KNOWN_ACTION_AUXILIARIES
        if missing:
            raise KeyError(
                "predictor did not emit requested auxiliaries: " + ", ".join(sorted(missing))
            )
        return selected
