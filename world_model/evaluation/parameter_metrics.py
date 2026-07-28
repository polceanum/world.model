"""Directional diagnostics for inference-time physical-parameter updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

_DIRECTIONAL_PARAMETERS = ("restitution", "drag")


@dataclass
class _ParameterUpdateTotals:
    pre_update_absolute_error: float = 0.0
    post_update_absolute_error: float = 0.0
    absolute_update: float = 0.0
    positive_error_reductions: int = 0
    count: int = 0


@dataclass
class OnlineParameterUpdateAccumulator:
    """Measure whether selected online parameter updates move toward truth.

    The evaluator supplies a mask that is already restricted to persistent,
    distance-gated objects with both a runtime identifier update and
    ground-truth informative evidence. Simulator parameters and events are used
    only to score direction after inference; they are never runtime inputs.
    """

    totals: dict[str, _ParameterUpdateTotals] = field(
        default_factory=lambda: {name: _ParameterUpdateTotals() for name in _DIRECTIONAL_PARAMETERS}
    )

    def update(
        self,
        name: str,
        pre_update: Tensor,
        post_update: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        """Accumulate scalar physical-space errors for selected object updates."""

        if name not in self.totals:
            raise ValueError(
                f"directional parameter name must be one of {_DIRECTIONAL_PARAMETERS}, got {name!r}"
            )
        if pre_update.shape != post_update.shape or pre_update.shape != target.shape:
            raise ValueError("pre-update, post-update, and target tensors must share shape")
        if pre_update.ndim < 1 or pre_update.shape[-1] != 1:
            raise ValueError("directional physical parameters must have trailing dimension one")
        if mask.shape != pre_update.shape[:-1] or mask.dtype is not torch.bool:
            raise ValueError("directional parameter mask must be bool and match parameter [B,N]")

        expanded_mask = mask.unsqueeze(-1).expand_as(pre_update)
        selected_pre = pre_update.masked_select(expanded_mask)
        if selected_pre.numel() == 0:
            return
        selected_post = post_update.masked_select(expanded_mask)
        selected_target = target.masked_select(expanded_mask)
        for label, values in (
            ("pre-update", selected_pre),
            ("post-update", selected_post),
            ("target", selected_target),
        ):
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"selected {label} parameter values must be finite")

        pre_error = (selected_pre - selected_target).abs()
        post_error = (selected_post - selected_target).abs()
        error_reduction = pre_error - post_error
        absolute_update = (selected_post - selected_pre).abs()
        # MPS cannot cast directly to float64. Transfer first, then use CPU
        # double precision for stable long-run accumulation.
        detached_pre = pre_error.detach().to(device="cpu").to(dtype=torch.float64)
        detached_post = post_error.detach().to(device="cpu").to(dtype=torch.float64)
        detached_reduction = error_reduction.detach().to(device="cpu").to(dtype=torch.float64)
        detached_update = absolute_update.detach().to(device="cpu").to(dtype=torch.float64)

        totals = self.totals[name]
        totals.pre_update_absolute_error += float(detached_pre.sum())
        totals.post_update_absolute_error += float(detached_post.sum())
        totals.absolute_update += float(detached_update.sum())
        totals.positive_error_reductions += int((detached_reduction > 0.0).sum())
        totals.count += int(detached_reduction.numel())

    def metrics(self) -> dict[str, float | None]:
        """Return explicit before/after, direction, and magnitude metrics."""

        results: dict[str, float | None] = {}
        for name in _DIRECTIONAL_PARAMETERS:
            totals = self.totals[name]
            prefix = f"informative_{name}"
            count = totals.count
            results[f"{prefix}_pre_update_mae"] = (
                totals.pre_update_absolute_error / count if count else None
            )
            results[f"{prefix}_post_update_mae"] = (
                totals.post_update_absolute_error / count if count else None
            )
            results[f"{prefix}_signed_error_reduction_mean"] = (
                (totals.pre_update_absolute_error - totals.post_update_absolute_error) / count
                if count
                else None
            )
            results[f"{prefix}_positive_error_reduction_rate"] = (
                totals.positive_error_reductions / count if count else None
            )
            results[f"{prefix}_absolute_update_mean"] = (
                totals.absolute_update / count if count else None
            )
            results[f"{prefix}_update_count"] = float(count)
        return results


__all__ = ["OnlineParameterUpdateAccumulator"]
