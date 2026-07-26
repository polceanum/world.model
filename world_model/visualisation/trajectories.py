"""World-space trajectory plotting helpers."""

from __future__ import annotations

import matplotlib.axes
import numpy as np


def plot_xy_trajectory(
    axis: matplotlib.axes.Axes,
    positions: np.ndarray,
    active: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    """Plot x/y tracks from ``[T,N,3]`` state arrays."""

    if positions.ndim != 3:
        return
    labelled = False
    for slot in range(positions.shape[1]):
        valid = active[:, slot] & np.isfinite(positions[:, slot]).all(axis=-1)
        if valid.any():
            axis.plot(
                positions[valid, slot, 0],
                positions[valid, slot, 1],
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                label=label if not labelled else None,
            )
            labelled = True
