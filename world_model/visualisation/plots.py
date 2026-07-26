"""Standalone parameter-history plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_parameter_plot(
    timestamps: np.ndarray,
    estimated_drag: np.ndarray,
    estimated_restitution: np.ndarray,
    target_drag: np.ndarray,
    target_restitution: np.ndarray,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(timestamps, estimated_drag, label="estimated mean")
    axes[0].plot(timestamps, target_drag, "--", label="visible GT mean")
    axes[0].set_ylabel("drag")
    axes[0].legend(fontsize=8)
    axes[1].plot(timestamps, estimated_restitution, label="estimated mean")
    axes[1].plot(timestamps, target_restitution, "--", label="visible GT mean")
    axes[1].set_ylabel("restitution")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target
