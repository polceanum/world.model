#!/usr/bin/env python3
"""Generate deterministic trusted-local synthetic episode tensor files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from world_model.datasets import SyntheticSphereDataset
from world_model.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "ood"])
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    config = load_config(args.config)
    dataset = SyntheticSphereDataset(
        config,
        split=args.split,
        num_episodes=args.episodes,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(len(dataset)):
        episode = dataset[index]
        path = output / f"{args.split}_{int(episode['seed']):06d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(episode, temporary)
        temporary.replace(path)
        records.append({"seed": int(episode["seed"]), "path": path.name})
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": args.split,
                "config": str(Path(args.config).resolve()),
                "episodes": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
