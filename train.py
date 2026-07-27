#!/usr/bin/env python3
"""Train Project Orpheus from a validated YAML configuration."""

from __future__ import annotations

import argparse
import json

from world_model.utils.config import load_config
from world_model.utils.device import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    parser.add_argument(
        "--run-name",
        help="Run label; the directory receives a sortable UTC timestamp prefix",
    )
    parser.add_argument("--resume", help="Trusted local checkpoint to resume")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = list(args.set)
    if args.device:
        overrides.append(f"device.preference={args.device}")
    if args.seed is not None:
        overrides.append(f"project.seed={args.seed}")
    config = load_config(args.config, overrides=overrides)
    device = select_device(config.device.preference)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": "train",
                    "config": config.source_path,
                    "project": config.project.name,
                    "run_name": args.run_name,
                    "resume": args.resume,
                    "device": str(device.device),
                    "torch": device.torch_version,
                    "mps_built": device.mps_built,
                    "mps_available": device.mps_available,
                    "simulator": config.simulator.type,
                    "image_size": config.simulator.image_size,
                    "sequence_frames": config.simulator.sequence_frames,
                    "runtime_modality": config.runtime.modality,
                    "rgb_only_evaluation": config.evaluation.rgb_only,
                    "training_steps": config.training.steps,
                },
                indent=2,
            )
        )
        return 0

    from world_model.training.trainer import train_from_config

    result = train_from_config(
        config,
        run_name=args.run_name,
        resume_path=args.resume,
        device_info=device,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
