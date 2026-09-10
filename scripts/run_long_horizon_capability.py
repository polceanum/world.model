#!/usr/bin/env python3
"""Run the compact four/eight-second causal capability pilot."""

from __future__ import annotations

import argparse
import json

from world_model.evaluation.long_horizon import (
    default_long_horizon_run_directory,
    long_horizon_scenario_sha256,
    run_long_horizon_evaluation,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-action long-horizon contact and boundary behavior."
    )
    parser.add_argument(
        "--config",
        default="configs/rgbd_dynamic_set_planning_cpu.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/20260907-capability-development-v2/selected_checkpoint.pt",
    )
    parser.add_argument("--run-directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "scenario_manifest_sha256": long_horizon_scenario_sha256(),
                    "generated_episodes_retained": False,
                    "animation_media_retained": False,
                    "planning_used_as_training_loss": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = run_long_horizon_evaluation(
        model_config_path=parsed.config,
        checkpoint_path=parsed.checkpoint,
        run_directory=parsed.run_directory or default_long_horizon_run_directory(),
        seed=parsed.seed,
        threads=parsed.threads,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_failures": report["gate_failures"],
                "run_directory": report["run_directory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
