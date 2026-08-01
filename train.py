#!/usr/bin/env python3
"""Train Project Orpheus from a validated YAML configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from world_model.training.trainer import _resolve_training_devices
from world_model.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    parser.add_argument(
        "--run-name",
        help="Run label; the directory receives a sortable UTC timestamp prefix",
    )
    parser.add_argument("--resume", help="Trusted local checkpoint to resume")
    parser.add_argument(
        "--initialize-from",
        help=(
            "Trusted local checkpoint providing model weights only; starts a new "
            "run with step/optimizer/RNG reset"
        ),
    )
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

    if args.dry_run:
        resume_step = 0
        if args.resume is not None:
            resume_source = Path(args.resume).expanduser().resolve()
            if not resume_source.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {resume_source}")
            payload = torch.load(
                resume_source,
                map_location="cpu",
                weights_only=False,
            )
            resume_step = int(payload.get("step", -1))
            if resume_step < 0:
                raise ValueError("checkpoint step must be a nonnegative integer")
        device_info, measurement_device, closed_loop_device, active_device = (
            _resolve_training_devices(
                config,
                start_step=resume_step,
                initialize_from=args.initialize_from is not None,
            )
        )
        print(
            json.dumps(
                {
                    "command": "train",
                    "config": config.source_path,
                    "project": config.project.name,
                    "run_name": args.run_name,
                    "resume": args.resume,
                    "initialize_from": args.initialize_from,
                    "device": str(active_device),
                    "measurement_device": str(measurement_device),
                    "closed_loop_device": str(closed_loop_device),
                    "torch": device_info.torch_version,
                    "mps_built": device_info.mps_built,
                    "mps_available": device_info.mps_available,
                    "simulator": config.simulator.type,
                    "image_size": config.simulator.image_size,
                    "sequence_frames": config.simulator.sequence_frames,
                    "runtime_modality": config.runtime.modality,
                    "rgb_only_evaluation": config.evaluation.rgb_only,
                    "training_steps": config.training.steps,
                    "train_episodes": config.training.train_episodes,
                    "validation_episodes": config.training.validation_episodes,
                    "batch_size": config.training.batch_size,
                    "nominal_training_episode_draws": (
                        config.training.steps * config.training.batch_size
                    ),
                    "nominal_dataset_passes": (
                        config.training.steps
                        * config.training.batch_size
                        / config.training.train_episodes
                    ),
                    "scenario_families": list(config.simulator.scenario_mixture),
                    "fixed_dataset": config.training.fixed_dataset,
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
        initialize_from_path=args.initialize_from,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
