"""Forecast metrics conditioned on future simulator collision labels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from world_model.simulator.collisions import BOUNDARY_NAMES

_COLLISION_CLASSES = (
    "pair_only",
    "ground_only",
    "wall_only",
    "other_only",
    "compound",
    "no_collision",
)


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


def collision_class_masks_for_forecast_window(
    events: Mapping[str, Tensor],
    *,
    anchor_frame: int,
    target_frame: int,
) -> dict[str, Tensor]:
    """Classify each target object by collisions anywhere in ``(anchor, target]``.

    The returned masks are mutually exclusive and exhaustive. ``other_only``
    keeps ceiling or future boundary kinds visible instead of incorrectly
    counting them as no-collision. A window with two or more kinds is
    ``compound`` even when those events occur at different frames.
    """

    required = ("collision", "pair_collision", "ground_collision", "wall_collision")
    missing = [name for name in required if name not in events]
    if missing:
        raise ValueError("collision class labels require event fields: " + ", ".join(missing))
    any_collision = collision_mask_for_forecast_window(
        events["collision"],
        anchor_frame=anchor_frame,
        target_frame=target_frame,
    )
    expected_shape = any_collision.shape
    start = anchor_frame + 1
    stop = target_frame + 1

    pair_events = events["pair_collision"]
    if tuple(pair_events.shape) != (
        expected_shape[0],
        events["collision"].shape[1],
        expected_shape[1],
        expected_shape[1],
    ):
        raise ValueError("pair_collision must have shape [B,T,N,N]")
    pair = pair_events[:, start:stop].bool().any(dim=1).any(dim=-1)

    ground_events = events["ground_collision"]
    if ground_events.shape != events["collision"].shape:
        raise ValueError("ground_collision must have shape [B,T,N]")
    ground = ground_events[:, start:stop].bool().any(dim=1)

    wall_events = events["wall_collision"]
    if tuple(wall_events.shape) != (
        expected_shape[0],
        events["collision"].shape[1],
        expected_shape[1],
        len(BOUNDARY_NAMES) - 2,
    ):
        raise ValueError("wall_collision must have shape [B,T,N,W]")
    wall = wall_events[:, start:stop].bool().any(dim=(1, 3))

    other = torch.zeros_like(any_collision)
    boundary_events = events.get("boundary_collision")
    if boundary_events is not None:
        if boundary_events.ndim != 4 or boundary_events.shape[:3] != (
            expected_shape[0],
            events["collision"].shape[1],
            expected_shape[1],
        ):
            raise ValueError("boundary_collision must have shape [B,T,N,K]")
        if boundary_events.shape[-1] != len(BOUNDARY_NAMES):
            raise ValueError("boundary_collision boundary axis does not match BOUNDARY_NAMES")
        other_indices = [index for index, name in enumerate(BOUNDARY_NAMES) if name == "ceiling"]
        if other_indices:
            other = boundary_events[:, start:stop, :, other_indices].bool().any(dim=(1, 3))
    # Preserve exhaustiveness for old/custom event records that expose only
    # the aggregate collision flag for an otherwise unknown boundary kind.
    other |= any_collision & ~(pair | ground | wall)

    kind_count = pair.to(torch.int8) + ground.to(torch.int8) + wall.to(torch.int8)
    kind_count += other.to(torch.int8)
    masks = {
        "pair_only": pair & (kind_count == 1),
        "ground_only": ground & (kind_count == 1),
        "wall_only": wall & (kind_count == 1),
        "other_only": other & (kind_count == 1),
        "compound": kind_count >= 2,
        "no_collision": ~any_collision,
    }
    class_count = torch.stack(tuple(masks.values()), dim=-1).sum(dim=-1)
    if not bool((class_count == 1).all()):
        raise RuntimeError("collision class masks must be mutually exclusive and exhaustive")
    return masks


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
    class_errors: dict[tuple[str, str, str], _PositionError] = field(default_factory=dict)
    class_object_horizons: dict[tuple[str, str], int] = field(default_factory=dict)
    class_schema_enabled: bool = False

    def update(
        self,
        *,
        horizon: str,
        predictions: Mapping[str, Tensor],
        target: Tensor,
        valid_mask: Tensor,
        collision_mask: Tensor,
        collision_classes: Mapping[str, Tensor] | None = None,
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
        if collision_classes is not None:
            if tuple(collision_classes) != _COLLISION_CLASSES:
                raise ValueError(
                    "collision_classes must contain the canonical ordered class schema"
                )
            class_count = torch.zeros_like(valid_mask, dtype=torch.int8)
            for name, class_mask in collision_classes.items():
                if class_mask.shape != expected_mask_shape or class_mask.dtype != torch.bool:
                    raise ValueError(f"collision class {name!r} must be bool with shape [B,N]")
                class_count += class_mask.to(torch.int8)
            if bool((class_count > 1).any()) or not bool(
                (class_count.masked_select(valid_mask) == 1).all()
            ):
                raise ValueError(
                    "collision class masks must be mutually exclusive and exhaustive "
                    "under the forecast-valid mask"
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
        if collision_classes is not None:
            self.class_schema_enabled = True
            for class_name, class_mask in collision_classes.items():
                selected = valid_mask & class_mask
                key = (class_name, horizon)
                self.class_object_horizons[key] = self.class_object_horizons.get(key, 0) + int(
                    selected.detach().sum().cpu()
                )
                for method, prediction in predictions.items():
                    self.class_errors.setdefault(
                        (class_name, method, horizon),
                        _PositionError(),
                    ).update(prediction, target, selected)

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
            if not self.class_schema_enabled:
                continue
            classified = sum(
                self.class_object_horizons.get((class_name, horizon), 0)
                for class_name in _COLLISION_CLASSES
            )
            if classified != eligible:
                raise RuntimeError(
                    "collision class support must partition eligible object horizons"
                )
            for class_name in _COLLISION_CLASSES:
                class_count = self.class_object_horizons.get((class_name, horizon), 0)
                class_prefix = f"collision_class_{class_name}"
                results[f"{class_prefix}_object_horizons@{horizon}"] = float(class_count)
                results[f"{class_prefix}_fraction@{horizon}"] = (
                    class_count / eligible if eligible else None
                )
                for method in methods:
                    error = self.class_errors.get(
                        (class_name, method, horizon),
                        _PositionError(),
                    )
                    prefix = f"{class_prefix}_{method}@{horizon}"
                    results[f"{prefix}_position_rmse_m"] = error.rmse
                    results[f"{prefix}_position_mae_m"] = error.mae
                    results[f"{prefix}_position_coordinate_count"] = float(error.coordinate_count)

                model_class_error = self.class_errors.get((class_name, "model", horizon))
                constant_velocity_class_error = self.class_errors.get(
                    (class_name, "constant_velocity", horizon)
                )
                reduction_key = (
                    f"{class_prefix}_model_vs_constant_velocity"
                    f"@{horizon}_position_rmse_reduction_fraction"
                )
                if (
                    model_class_error is None
                    or constant_velocity_class_error is None
                    or model_class_error.rmse is None
                    or constant_velocity_class_error.rmse is None
                    or constant_velocity_class_error.rmse <= 0.0
                ):
                    results[reduction_key] = None
                else:
                    if (
                        model_class_error.coordinate_count
                        != constant_velocity_class_error.coordinate_count
                    ):
                        raise RuntimeError(
                            "collision-class model and constant-velocity masks differ"
                        )
                    results[reduction_key] = 1.0 - (
                        model_class_error.rmse / constant_velocity_class_error.rmse
                    )
        return results


__all__ = [
    "CollisionConditionedForecastAccumulator",
    "collision_class_masks_for_forecast_window",
    "collision_mask_for_forecast_window",
]
