#!/usr/bin/env python3
"""Evaluate a trusted local Orpheus checkpoint."""

from __future__ import annotations

import argparse
import json

from world_model.evaluation.seed_protocol import EVALUATION_SEED_PROTOCOLS
from world_model.utils.config import load_config
from world_model.utils.device import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test", "ood"])
    parser.add_argument(
        "--seed-protocol",
        default="standard",
        choices=EVALUATION_SEED_PROTOCOLS,
        help=(
            "Use fresh_validation with --split validation for checkpoint "
            "selection on seeds disjoint from trainer validation and test."
        ),
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        help=(
            "Explicit offset inside the selected split. For fresh_validation "
            "it must begin after the checkpoint's trainer-validation manifest."
        ),
    )
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = list(args.set)
    if args.device:
        overrides.append(f"device.preference={args.device}")
    config = load_config(args.config, overrides=overrides)
    device = select_device(config.device.preference)

    from world_model.evaluation.evaluator import evaluate_checkpoint

    result = evaluate_checkpoint(
        config=config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        seed_protocol=args.seed_protocol,
        seed_offset=args.seed_offset,
        output_dir=args.output,
        device_info=device,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
