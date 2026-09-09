#!/usr/bin/env python3
"""Run one compact factor-conditioned downstream planning screen."""

from __future__ import annotations

import argparse
import json

from world_model.evaluation.capability_factor_runner import SUPPORTED_CAPABILITY_FACTORS
from world_model.evaluation.capability_planning_runner import (
    default_planning_run_directory,
    factor_planning_rows,
    run_incumbent_factor_planning,
)
from world_model.evaluation.general_capability import manifest_sha256


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the calibrated incumbent on factor-conditioned planning tasks."
    )
    parser.add_argument("--factor", choices=SUPPORTED_CAPABILITY_FACTORS, required=True)
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
    rows = factor_planning_rows(parsed.factor)
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "factor": parsed.factor,
                    "planning_tasks": len(rows),
                    "factor_rows_sha256": manifest_sha256(tuple(item[0] for item in rows)),
                    "candidate_counts": [8, 32],
                    "object_counts": list(range(1, 7)),
                    "generated_episodes_retained": False,
                    "planning_used_as_training_loss": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    run_directory = parsed.run_directory or default_planning_run_directory(parsed.factor)
    report = run_incumbent_factor_planning(
        factor=parsed.factor,
        model_config_path=parsed.config,
        checkpoint_path=parsed.checkpoint,
        run_directory=run_directory,
        seed=parsed.seed,
        threads=parsed.threads,
    )
    print(
        json.dumps(
            {
                "factor": report["factor"],
                "status": report["status"],
                "gate_failures": report["gate_failures"],
                "planning_error": report.get("planning_error"),
                "runtime_error": report.get("runtime_error"),
                "run_directory": report["run_directory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
