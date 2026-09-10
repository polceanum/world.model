"""Typed, deterministic actions for analytic counterfactual rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor

from world_model.belief import MotionMode, WorldBelief
from world_model.belief._base import TensorDataclassMixin


@dataclass(frozen=True)
class WorldImpulseAction(TensorDataclassMixin):
    """One known world-frame momentum impulse at an absolute future time.

    ``impulse_world`` has units of belief-mass-unit times metres per second.
    It is a physical N s impulse only when the belief mass is declared in
    kilograms by the data profile.  The action is deliberately stateless and
    single-use: callers must provide a timestamp strictly after the source
    belief and resolve the target through its persistent object ID.
    """

    timestamp: Tensor
    object_id: Tensor
    impulse_world: Tensor
    frame: Literal["world"] = "world"

    def _application_mask_for(self, belief: WorldBelief) -> Tensor:
        """Return rows on which this action exists.

        Public actions always exist on every batch row.  The counterfactual
        planner overrides this internal hook only for its flattened ``B x K``
        carrier, where a ``None`` candidate must share one rollout with real
        actions without becoming a numerically-zero action boundary.
        """

        return torch.ones(belief.batch_size, dtype=torch.bool, device=belief.device)

    def validate_for(
        self,
        belief: WorldBelief,
        *,
        latest_timestamp: Tensor | None = None,
    ) -> Tensor:
        """Validate against ``belief`` and return the resolved ``[B,N]`` mask.

        This method is pure: it does not clone, cast, detach, propagate, or
        mutate either the action or belief.  ``latest_timestamp`` is an
        optional absolute inclusive rollout horizon used by atomic planners.
        """

        if not isinstance(belief, WorldBelief):
            raise TypeError("belief must be a WorldBelief")
        for name, value in (
            ("timestamp", self.timestamp),
            ("object_id", self.object_id),
            ("impulse_world", self.impulse_world),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"action {name} must be a torch.Tensor")
        if self.frame != "world":
            raise ValueError("WorldImpulseAction frame must be 'world'")

        batch = belief.batch_size
        if self.timestamp.shape != (batch,):
            raise ValueError("action timestamp must have shape [B]")
        if self.object_id.shape != (batch,):
            raise ValueError("action object_id must have shape [B]")
        if self.impulse_world.shape != (batch, 3):
            raise ValueError("action impulse_world must have shape [B,3]")

        for name, value in (
            ("timestamp", self.timestamp),
            ("impulse_world", self.impulse_world),
        ):
            if value.device != belief.device:
                raise ValueError(f"action {name} device must exactly match the belief")
            if value.dtype != belief.dtype:
                raise TypeError(f"action {name} dtype must exactly match the belief")
            if not torch.isfinite(value).all():
                raise ValueError(f"action {name} must contain only finite values")
        if self.object_id.device != belief.device:
            raise ValueError("action object_id device must exactly match the belief")
        if self.object_id.dtype != torch.int64:
            raise TypeError("action object_id must have dtype torch.int64")
        if self.timestamp.requires_grad:
            raise ValueError("action timestamp must not require gradients")
        if torch.any(self.timestamp <= belief.timestamp):
            raise ValueError("action timestamp must be strictly after the belief timestamp")

        if latest_timestamp is not None:
            if not isinstance(latest_timestamp, Tensor):
                raise TypeError("latest_timestamp must be a torch.Tensor")
            if latest_timestamp.shape != (batch,):
                raise ValueError("latest_timestamp must have shape [B]")
            if latest_timestamp.device != belief.device:
                raise ValueError("latest_timestamp device must exactly match the belief")
            if latest_timestamp.dtype != belief.dtype:
                raise TypeError("latest_timestamp dtype must exactly match the belief")
            if not torch.isfinite(latest_timestamp).all():
                raise ValueError("latest_timestamp must contain only finite values")
            if torch.any(latest_timestamp < belief.timestamp):
                raise ValueError("latest_timestamp cannot precede the belief timestamp")
            if torch.any(self.timestamp > latest_timestamp):
                raise ValueError("action timestamp must lie within the rollout horizon")

        persistent = self.object_id >= 0
        matches = (
            belief.objects.active
            & persistent.unsqueeze(-1)
            & (belief.objects.object_id == self.object_id.unsqueeze(-1))
        )
        if torch.any(matches.sum(dim=-1) != 1):
            raise ValueError(
                "action object_id must resolve to exactly one active persistent object"
            )
        target_mode = torch.where(
            matches,
            belief.objects.mode,
            belief.objects.mode.new_full(belief.objects.mode.shape, -1),
        ).amax(dim=-1)
        if torch.any(target_mode != int(MotionMode.FREE)):
            raise ValueError("action target must be in FREE motion mode")
        return matches


@dataclass(frozen=True)
class WorldImpulseSchedule:
    """An ordered sequence of known world-frame impulses.

    The schedule is intentionally a thin immutable container around the
    existing action type.  That keeps one-action callers and checkpoints
    exactly compatible while allowing causal rollouts and planners to express
    several future interventions.  Every action is validated against the
    source belief before propagation starts, and timestamps must be strictly
    increasing on each batch row that owns consecutive actions.
    """

    actions: tuple[WorldImpulseAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple):
            raise TypeError("WorldImpulseSchedule actions must be a tuple")
        if not self.actions:
            raise ValueError("WorldImpulseSchedule requires at least one action")
        if any(not isinstance(action, WorldImpulseAction) for action in self.actions):
            raise TypeError("WorldImpulseSchedule entries must be WorldImpulseAction values")

    @classmethod
    def from_action(cls, action: WorldImpulseAction) -> WorldImpulseSchedule:
        """Wrap one legacy action without changing its tensor identity."""

        if not isinstance(action, WorldImpulseAction):
            raise TypeError("action must be a WorldImpulseAction")
        return cls(actions=(action,))

    def validate_for(
        self,
        belief: WorldBelief,
        *,
        latest_timestamp: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Validate the complete schedule before execution and return targets."""

        target_masks: list[Tensor] = []
        previous_timestamp = belief.timestamp
        previous_active = torch.zeros(
            belief.batch_size,
            dtype=torch.bool,
            device=belief.device,
        )
        for action in self.actions:
            target_masks.append(action.validate_for(belief, latest_timestamp=latest_timestamp))
            active = action._application_mask_for(belief)
            if not isinstance(active, Tensor):
                raise TypeError("action application mask must be a tensor")
            if active.shape != (belief.batch_size,):
                raise ValueError("action application mask must have shape [B]")
            if active.dtype is not torch.bool:
                raise TypeError("action application mask must be boolean")
            if active.device != belief.device:
                raise ValueError("action application mask must match belief device")
            if active.requires_grad:
                raise ValueError("action application mask must not require gradients")
            if torch.any(active & previous_active & (action.timestamp <= previous_timestamp)):
                raise ValueError(
                    "schedule timestamps must be strictly increasing on every active row"
                )
            previous_timestamp = torch.where(active, action.timestamp, previous_timestamp)
            previous_active = previous_active | active
        return tuple(target_masks)


WorldAction: TypeAlias = WorldImpulseAction | WorldImpulseSchedule


__all__ = ["WorldAction", "WorldImpulseAction", "WorldImpulseSchedule"]
