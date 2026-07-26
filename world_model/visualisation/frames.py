"""Image-space overlays for structured measurements and beliefs."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.axes
import numpy as np


def normalized_to_pixels(
    centres: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    height, width = image_size
    pixels = np.empty_like(centres)
    pixels[..., 0] = (centres[..., 0] + 1.0) * 0.5 * (width - 1)
    pixels[..., 1] = (centres[..., 1] + 1.0) * 0.5 * (height - 1)
    return pixels


def overlay_points(
    axis: matplotlib.axes.Axes,
    points: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
    valid: Iterable[bool] | None = None,
    size: float = 40.0,
) -> None:
    if points.size == 0:
        return
    mask = np.ones(points.shape[0], dtype=bool)
    if valid is not None:
        mask = np.asarray(list(valid), dtype=bool)
    finite = np.isfinite(points).all(axis=-1)
    selected = points[mask & finite]
    if selected.size:
        axis.scatter(
            selected[:, 0],
            selected[:, 1],
            s=size,
            facecolors="none" if marker == "o" else color,
            edgecolors=color,
            marker=marker,
            linewidths=1.4,
            label=label,
        )
