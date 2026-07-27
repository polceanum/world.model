"""Approximate image-space uncertainty overlays."""

from __future__ import annotations

import matplotlib.axes
from matplotlib.patches import Ellipse


def add_uncertainty_ellipse(
    axis: matplotlib.axes.Axes,
    *,
    x: float,
    y: float,
    sigma_x_pixels: float,
    sigma_y_pixels: float,
    color: str = "lime",
    confidence_scale: float = 1.64,
    angle_degrees: float = 0.0,
) -> None:
    width = max(1.0, 2.0 * confidence_scale * sigma_x_pixels)
    height = max(1.0, 2.0 * confidence_scale * sigma_y_pixels)
    axis.add_patch(
        Ellipse(
            (x, y),
            width=width,
            height=height,
            angle=angle_degrees,
            fill=False,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.7,
        )
    )
