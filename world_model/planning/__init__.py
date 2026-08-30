"""Read-only action planning over persistent world-model beliefs."""

from world_model.planning.counterfactual import (
    CounterfactualCostWeights,
    CounterfactualPlanResult,
    TerminalWorldPositionGoal,
    plan_counterfactual_actions,
    resolve_appearance_handle,
)

__all__ = [
    "CounterfactualCostWeights",
    "CounterfactualPlanResult",
    "TerminalWorldPositionGoal",
    "plan_counterfactual_actions",
    "resolve_appearance_handle",
]
