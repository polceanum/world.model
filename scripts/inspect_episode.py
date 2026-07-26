#!/usr/bin/env python3
"""Render an annotated contact sheet for one deterministic toy episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from world_model.simulator import generate_episode
from world_model.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    episode = generate_episode(config, args.seed)
    count = min(args.frames, episode["rgb"].shape[0])
    indices = np.linspace(0, episode["rgb"].shape[0] - 1, count).round().astype(int)
    figure, axes = plt.subplots(2, (count + 1) // 2, figsize=(3 * ((count + 1) // 2), 6))
    axes_flat = np.asarray(axes).reshape(-1)
    for axis, frame_index in zip(axes_flat, indices, strict=False):
        image = episode["rgb"][frame_index].permute(1, 2, 0).numpy()
        axis.imshow(image)
        active = episode["objects"]["active"][frame_index]
        centres = episode["labels"]["projected_center_pixels"][frame_index]
        ids = episode["objects"]["id"][frame_index]
        for slot in active.nonzero().flatten().tolist():
            x, y = centres[slot].tolist()
            axis.text(x, y, str(int(ids[slot])), color="white", fontsize=8)
        collision = bool(episode["events"]["collision"][frame_index].any())
        axis.set_title(f"frame {frame_index} collision={collision}")
        axis.axis("off")
    for axis in axes_flat[count:]:
        axis.axis("off")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
