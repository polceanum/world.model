"""Stateless counterfactual planning over known world-frame impulses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch
from torch import Tensor

from world_model.belief import BeliefTrajectory, WorldBelief, fast_packing_map
from world_model.belief._base import TensorDataclassMixin
from world_model.dynamics import AnalyticFreeMotionDynamics, WorldImpulseAction


@dataclass(frozen=True)
class TerminalWorldPositionGoal(TensorDataclassMixin):
    """A terminal world-frame position target addressed by persistent object ID."""

    object_id: Tensor
    position_world: Tensor
    frame: Literal["world"] = "world"


@dataclass(frozen=True)
class CounterfactualCostWeights:
    """Fixed, non-learned weights for counterfactual candidate costs."""

    terminal_position: float = 1.0
    terminal_variance: float = 0.0
    impulse_effort: float = 0.0

    def __post_init__(self) -> None:
        _validate_cost_weights(self)


@dataclass(frozen=True)
class CounterfactualPlanResult:
    """All candidate rollouts and costs, retaining every differentiable branch."""

    actions: tuple[WorldImpulseAction | None, ...]
    trajectories: tuple[BeliefTrajectory, ...]
    object_id_by_slot: Tensor
    terminal_squared_error: Tensor
    terminal_position_variance: Tensor
    impulse_effort: Tensor
    total_cost: Tensor
    selected_index: Tensor


def _validate_real_weight(name: str, value: object, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-boolean real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    if not positive and numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_cost_weights(weights: CounterfactualCostWeights) -> None:
    _validate_real_weight(
        "terminal_position",
        weights.terminal_position,
        positive=True,
    )
    _validate_real_weight(
        "terminal_variance",
        weights.terminal_variance,
        positive=False,
    )
    _validate_real_weight(
        "impulse_effort",
        weights.impulse_effort,
        positive=False,
    )


_DEFAULT_COST_WEIGHTS = CounterfactualCostWeights()


def _validate_goal(goal: TerminalWorldPositionGoal, belief: WorldBelief) -> Tensor:
    if not isinstance(goal, TerminalWorldPositionGoal):
        raise TypeError("goal must be a TerminalWorldPositionGoal")
    if goal.frame != "world":
        raise ValueError("goal frame must be 'world'")
    if not isinstance(goal.object_id, Tensor):
        raise TypeError("goal object_id must be a tensor")
    if not isinstance(goal.position_world, Tensor):
        raise TypeError("goal position_world must be a tensor")

    batch = belief.batch_size
    if goal.object_id.shape != (batch,):
        raise ValueError(f"goal object_id must have shape {(batch,)}")
    if goal.object_id.dtype is not torch.int64:
        raise TypeError("goal object_id must have dtype torch.int64")
    if goal.object_id.device != belief.device:
        raise ValueError("goal object_id must be on the belief device")
    if goal.position_world.shape != (batch, 3):
        raise ValueError(f"goal position_world must have shape {(batch, 3)}")
    if goal.position_world.dtype != belief.dtype:
        raise TypeError("goal position_world must have the belief dtype")
    if goal.position_world.device != belief.device:
        raise ValueError("goal position_world must be on the belief device")
    if not torch.isfinite(goal.position_world).all():
        raise ValueError("goal position_world must be finite")

    matches = belief.objects.active & (belief.objects.object_id == goal.object_id.unsqueeze(-1))
    if not torch.all(matches.sum(dim=-1) == 1):
        raise ValueError("each goal object_id must resolve to exactly one active object")
    return matches


def _target_slots(target_mask: Tensor) -> Tensor:
    return target_mask.to(dtype=torch.int64).argmax(dim=-1)


def resolve_appearance_handle(
    belief: WorldBelief,
    prototype: Tensor,
    *,
    minimum_cosine_margin: float,
) -> Tensor:
    """Resolve observable appearance prototypes to persistent object IDs.

    Resolution uses only the active belief's appearance vectors.  It deliberately
    exposes no slot, simulator-truth, assignment, or evaluation correspondence.
    The returned persistent IDs are a discrete selection; gradients through the
    selection decision are not part of this contract.
    """

    if not isinstance(prototype, Tensor):
        raise TypeError("prototype must be a tensor")
    expected = (belief.batch_size, belief.objects.appearance_dim)
    if prototype.shape != expected:
        raise ValueError(f"prototype must have shape {expected}")
    if prototype.dtype != belief.dtype:
        raise TypeError("prototype must have the belief dtype")
    if prototype.device != belief.device:
        raise ValueError("prototype must be on the belief device")
    if not torch.isfinite(prototype).all():
        raise ValueError("prototype must be finite")
    _validate_real_weight(
        "minimum_cosine_margin",
        minimum_cosine_margin,
        positive=False,
    )

    active = belief.objects.active
    if torch.any(active.sum(dim=-1) < 2):
        raise ValueError("appearance resolution requires at least two active objects")
    if torch.any((belief.objects.object_id < 0) & active):
        raise ValueError("active appearance candidates must have persistent object IDs")
    if torch.any(
        active.unsqueeze(-1)
        & active.unsqueeze(-2)
        & (belief.objects.object_id.unsqueeze(-1) == belief.objects.object_id.unsqueeze(-2))
        & ~torch.eye(
            belief.objects.max_objects,
            device=belief.device,
            dtype=torch.bool,
        ).unsqueeze(0)
    ):
        raise ValueError("active appearance candidates must have unique persistent IDs")
    if not torch.isfinite(belief.objects.appearance.masked_select(active.unsqueeze(-1))).all():
        raise ValueError("active appearance candidates must be finite")

    with torch.no_grad():
        prototype_norm = torch.linalg.vector_norm(prototype, dim=-1)
        appearance_norm = torch.linalg.vector_norm(
            belief.objects.appearance,
            dim=-1,
        )
        if torch.any(prototype_norm <= 0.0):
            raise ValueError("prototype must have nonzero norm")
        if torch.any(active & (appearance_norm <= 0.0)):
            raise ValueError("active appearance candidates must have nonzero norm")

        normalised_prototype = prototype / prototype_norm.unsqueeze(-1)
        normalised_appearance = belief.objects.appearance / appearance_norm.clamp_min(
            torch.finfo(belief.dtype).tiny
        ).unsqueeze(-1)
        cosine = torch.einsum("bd,bnd->bn", normalised_prototype, normalised_appearance)
        cosine = cosine.masked_fill(~active, -torch.inf)
        top_values, top_slots = torch.topk(cosine, k=2, dim=-1, largest=True, sorted=True)
        margin = top_values[:, 0] - top_values[:, 1]
        required_margin = float(minimum_cosine_margin)
        if torch.any((margin <= 0.0) | (margin < required_margin)):
            raise ValueError("appearance handle is ambiguous or below the cosine margin")

        selected_ids = torch.gather(
            belief.objects.object_id,
            dim=1,
            index=top_slots[:, :1],
        ).squeeze(1)
        selected_matches = active & (belief.objects.object_id == selected_ids.unsqueeze(-1))
        if not torch.all(selected_matches.sum(dim=-1) == 1):
            raise ValueError("appearance handle did not resolve to exactly one persistent ID")
        return selected_ids.clone()


def plan_counterfactual_actions(
    dynamics: AnalyticFreeMotionDynamics,
    belief: WorldBelief,
    query_times: Tensor | Sequence[float],
    candidates: Sequence[WorldImpulseAction | None],
    goal: TerminalWorldPositionGoal,
    *,
    weights: CounterfactualCostWeights = _DEFAULT_COST_WEIGHTS,
) -> CounterfactualPlanResult:
    """Roll out and score a finite set of known-action counterfactuals.

    Every query, goal, weight, and candidate is validated before the first
    rollout.  The function is read-only with respect to ``belief`` and retains
    the computation graph for every candidate cost column.
    """

    if not isinstance(weights, CounterfactualCostWeights):
        raise TypeError("weights must be CounterfactualCostWeights")
    _validate_cost_weights(weights)
    try:
        actions = tuple(candidates)
    except TypeError as error:
        raise TypeError("candidates must be a sequence") from error
    if not actions:
        raise ValueError("candidates must be nonempty")

    # This call is a pure normalisation/validation pass, not a rollout.
    offsets = dynamics.validate_action_rollout(belief, query_times, None)
    if offsets.shape[1] == 0:
        raise ValueError("counterfactual planning requires at least one query time")
    target_mask = _validate_goal(goal, belief)
    latest_timestamp = belief.timestamp + offsets[:, -1]
    for action in actions:
        if action is not None and not isinstance(action, WorldImpulseAction):
            raise TypeError("each candidate must be WorldImpulseAction or None")
        if action is not None:
            action.validate_for(belief, latest_timestamp=latest_timestamp)

    target_slot = _target_slots(target_mask)
    batch_index = torch.arange(belief.batch_size, device=belief.device)
    position_slice = fast_packing_map(belief.objects)["position"]
    trajectories: list[BeliefTrajectory] = []
    squared_errors: list[Tensor] = []
    position_variances: list[Tensor] = []
    impulse_efforts: list[Tensor] = []

    for action in actions:
        trajectory = dynamics.rollout(belief, offsets, action=action)
        trajectories.append(trajectory)
        terminal_position = trajectory.positions[batch_index, -1, target_slot]
        terminal_log_variance = trajectory.fast_log_variance[
            batch_index,
            -1,
            target_slot,
            position_slice,
        ]
        squared_errors.append((terminal_position - goal.position_world).square().sum(dim=-1))
        position_variances.append(terminal_log_variance.exp().sum(dim=-1))
        if action is None:
            impulse_efforts.append(belief.objects.position.new_zeros(belief.batch_size))
        else:
            impulse_efforts.append(action.impulse_world.square().sum(dim=-1))

    terminal_squared_error = torch.stack(squared_errors, dim=-1)
    terminal_position_variance = torch.stack(position_variances, dim=-1)
    impulse_effort = torch.stack(impulse_efforts, dim=-1)
    total_cost = (
        weights.terminal_position * terminal_squared_error
        + weights.terminal_variance * terminal_position_variance
        + weights.impulse_effort * impulse_effort
    )
    selected_index = total_cost.argmin(dim=-1)
    return CounterfactualPlanResult(
        actions=actions,
        trajectories=tuple(trajectories),
        object_id_by_slot=belief.objects.object_id.clone(),
        terminal_squared_error=terminal_squared_error,
        terminal_position_variance=terminal_position_variance,
        impulse_effort=impulse_effort,
        total_cost=total_cost,
        selected_index=selected_index,
    )


__all__ = [
    "CounterfactualCostWeights",
    "CounterfactualPlanResult",
    "TerminalWorldPositionGoal",
    "plan_counterfactual_actions",
    "resolve_appearance_handle",
]
