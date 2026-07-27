"""Forecast metrics conditioned on future simulator collision labels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor


def collision_mask_for_forecast_window(
    collision_events: Tensor,
    *,
    anchor_frame: int,
    target_frame: int,
) -> Tensor:
    """Return ``[B,N]`` labels for any collision in ``(anchor, target]``.

    The simulator event tensor is evaluation-only ground truth with shape
    ``[B,T,N]``.  Excluding the anchor frame prevents an already-observed
    collision from being used to condition a future forecast.
    """

    if collision_events.ndim != 3:
        raise ValueError("collision_events must have shape [B,T,N]")
    if anchor_frame < 0:
        raise ValueError("anchor_frame must be nonnegative")
    if target_frame <= anchor_frame:
        raise ValueError("target_frame must be after anchor_frame")
    if target_frame >= collision_events.shape[1]:
        raise ValueError("target_frame exceeds collision event sequence")
    return collision_events[:, anchor_frame + 1 : target_frame + 1].bool().any(dim=1)


@dataclass
class _PositionError:
    squared_sum: float = 0.0
    absolute_sum: float = 0.0
    coordinate_count: int = 0

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        expanded = mask.unsqueeze(-1).expand_as(prediction)
        values = (prediction - target).masked_select(expanded)
        if values.numel() == 0:
            return
        detached = values.detach().float().cpu()
        self.squared_sum += float(detached.square().sum())
        self.absolute_sum += float(detached.abs().sum())
        self.coordinate_count += int(detached.numel())

    @property
    def rmse(self) -> float | None:
        if self.coordinate_count == 0:
            return None
        return math.sqrt(self.squared_sum / self.coordinate_count)

    @property
    def mae(self) -> float | None:
        if self.coordinate_count == 0:
            return None
        return self.absolute_sum / self.coordinate_count


@dataclass
class CollisionConditionedForecastAccumulator:
    """Accumulate model/baseline errors on identical collision windows."""

    errors: dict[tuple[str, str], _PositionError] = field(default_factory=dict)
    eligible_object_horizons: dict[str, int] = field(default_factory=dict)
    collision_object_horizons: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        *,
        horizon: str,
        predictions: Mapping[str, Tensor],
        target: Tensor,
        valid_mask: Tensor,
        collision_mask: Tensor,
    ) -> None:
        """Accumulate one horizon using simulator labels only as a metric mask."""

        if not horizon:
            raise ValueError("horizon must be nonempty")
        if not predictions:
            raise ValueError("predictions must contain at least one method")
        if target.ndim != 3 or target.shape[-1] != 3:
            raise ValueError("target must have shape [B,N,3]")
        expected_mask_shape = target.shape[:-1]
        for name, mask in (("valid_mask", valid_mask), ("collision_mask", collision_mask)):
            if mask.shape != expected_mask_shape or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be bool with shape [B,N]")
        for method, prediction in predictions.items():
            if not method:
                raise ValueError("prediction method names must be nonempty")
            if prediction.shape != target.shape:
                raise ValueError(
                    f"prediction {method!r} must have shape {tuple(target.shape)}, "
                    f"got {tuple(prediction.shape)}"
                )

        conditioned_mask = valid_mask & collision_mask
        self.eligible_object_horizons[horizon] = self.eligible_object_horizons.get(
            horizon, 0
        ) + int(valid_mask.detach().sum().cpu())
        self.collision_object_horizons[horizon] = self.collision_object_horizons.get(
            horizon, 0
        ) + int(conditioned_mask.detach().sum().cpu())
        for method, prediction in predictions.items():
            self.errors.setdefault((method, horizon), _PositionError()).update(
                prediction,
                target,
                conditioned_mask,
            )

    @property
    def total_collision_object_horizons(self) -> int:
        return sum(self.collision_object_horizons.values())

    def metrics(self) -> dict[str, float | None]:
        """Return physical errors and paired model-vs-CV RMSE reductions."""

        results: dict[str, float | None] = {}
        horizons = sorted(self.eligible_object_horizons)
        methods = sorted({method for method, _ in self.errors})
        for horizon in horizons:
            eligible = self.eligible_object_horizons[horizon]
            conditioned = self.collision_object_horizons.get(horizon, 0)
            results[f"collision_conditioned_eligible_object_horizons@{horizon}"] = float(eligible)
            results[f"collision_conditioned_object_horizons@{horizon}"] = float(conditioned)
            results[f"collision_conditioned_fraction@{horizon}"] = (
                conditioned / eligible if eligible else None
            )
            for method in methods:
                error = self.errors.get((method, horizon), _PositionError())
                prefix = f"collision_conditioned_{method}@{horizon}"
                results[f"{prefix}_position_rmse_m"] = error.rmse
                results[f"{prefix}_position_mae_m"] = error.mae
                results[f"{prefix}_position_coordinate_count"] = float(error.coordinate_count)

            model_error = self.errors.get(("model", horizon))
            constant_velocity_error = self.errors.get(("constant_velocity", horizon))
            reduction_key = (
                "collision_conditioned_model_vs_constant_velocity"
                f"@{horizon}_position_rmse_reduction_fraction"
            )
            if (
                model_error is None
                or constant_velocity_error is None
                or model_error.rmse is None
                or constant_velocity_error.rmse is None
                or constant_velocity_error.rmse <= 0.0
            ):
                results[reduction_key] = None
            else:
                if model_error.coordinate_count != constant_velocity_error.coordinate_count:
                    raise RuntimeError(
                        "collision-conditioned model and constant-velocity masks differ"
                    )
                results[reduction_key] = 1.0 - (model_error.rmse / constant_velocity_error.rmse)
        return results


__all__ = [
    "CollisionConditionedForecastAccumulator",
    "collision_mask_for_forecast_window",
]
