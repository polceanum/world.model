"""World-space trajectory plotting helpers."""

from __future__ import annotations

import matplotlib.axes
import numpy as np
from matplotlib.lines import Line2D


def plot_xy_trajectory(
    axis: matplotlib.axes.Axes,
    positions: np.ndarray,
    active: np.ndarray,
    *,
    color: str,
    label: str | None,
    linestyle: str = "-",
    alpha: float = 1.0,
    linewidth: float = 1.2,
    zorder: float | None = None,
) -> list[Line2D]:
    """Plot x/y tracks from ``[T,N,3]`` state arrays."""

    if positions.ndim != 3:
        return []
    if active.shape != positions.shape[:2]:
        raise ValueError("active must match the [T,N] trajectory axes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("trajectory alpha must lie in [0, 1]")
    labelled = False
    lines: list[Line2D] = []
    for slot in range(positions.shape[1]):
        valid = active[:, slot] & np.isfinite(positions[:, slot]).all(axis=-1)
        if valid.any():
            lines.extend(
                axis.plot(
                    positions[valid, slot, 0],
                    positions[valid, slot, 1],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                    label=label if label is not None and not labelled else None,
                )
            )
            labelled = True
    return lines
