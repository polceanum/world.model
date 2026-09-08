#!/usr/bin/env python3
"""Run the small public-development world-model capability loop."""

from __future__ import annotations

import argparse
import json

from world_model.training.capability_workbench import (
    CapabilityWorkbenchConfig,
    default_run_directory,
    run_capability_workbench,
    workbench_plan,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and directly measure the compact 1-6 object RGB-D world model, "
            "including required counterfactual planning."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/rgbd_dynamic_set_planning_cpu.yaml",
        help="model/simulator configuration",
    )
    parser.add_argument("--profile", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--run-directory")
    parser.add_argument(
        "--checkpoint",
        help="resume a saved workbench model for evaluation; requires --train-updates 0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-updates", type=int)
    parser.add_argument("--physical-cycles", type=int)
    parser.add_argument("--physical-cycle-offset", type=int)
    parser.add_argument("--planning-repeats", type=int)
    parser.add_argument(
        "--velocity-variance-floor",
        type=float,
        default=4.0e-13,
        help="candidate-only floor for correlated RGB-D temporal velocity variance",
    )
    parser.add_argument(
        "--planning-invariants",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run the slower causal/parity/latency stress suite (development default: on)",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact workload without materializing data or writing artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    settings = CapabilityWorkbenchConfig.for_profile(
        parsed.profile,
        seed=parsed.seed,
        train_updates=parsed.train_updates,
        physical_cycles=parsed.physical_cycles,
        physical_cycle_offset=parsed.physical_cycle_offset,
        planning_repeats=parsed.planning_repeats,
        threads=parsed.threads,
        planning_invariants=parsed.planning_invariants,
        velocity_variance_floor=parsed.velocity_variance_floor,
    )
    if parsed.dry_run:
        print(json.dumps(workbench_plan(settings), indent=2, sort_keys=True))
        return 0
    run_directory = parsed.run_directory or default_run_directory(parsed.profile)
    report = run_capability_workbench(
        model_config_path=parsed.config,
        run_directory=run_directory,
        workbench_config=settings,
        resume_checkpoint_path=parsed.checkpoint,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_directory": str(run_directory),
                "candidate_score": report["candidate"]["capability_score"],
                "reference_score": report["reference"]["capability_score"],
                "primary_bottleneck": report["diagnosis"]["primary_bottleneck"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
