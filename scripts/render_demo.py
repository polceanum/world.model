#!/usr/bin/env python3
"""Developer wrapper for the public held-out RGB demo renderer."""

from __future__ import annotations

import argparse
import json

from world_model.utils.config import load_config
from world_model.utils.device import select_device
from world_model.visualisation.animation import create_demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    overrides: list[str] = []
    if args.device:
        overrides.append(f"device.preference={args.device}")
    if args.seed is not None:
        overrides.append(f"demo.seed={args.seed}")
    config = load_config(args.config, overrides=overrides)
    device = select_device(config.device.preference)
    result = create_demo(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device_info=device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
