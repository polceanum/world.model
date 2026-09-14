#!/usr/bin/env python3
"""Train and evaluate the compact twelve-second neural world-model bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.neural_long_horizon import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRAINING_STEPS,
    FORECAST_SECONDS,
    NEURAL_LONG_HORIZON_SCHEMA,
    neural_long_horizon_protocol_sha256,
    publish_neural_long_horizon,
    run_neural_long_horizon,
)
from world_model.identification import LongHorizonPhysicsAdapter


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent-checkpoint",
        default="runs/20260914-neural-adaptive-physics-v2/model.pt",
    )
    parser.add_argument("--run-directory")
    parser.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    incumbent = Path(parsed.incumbent_checkpoint).expanduser().resolve()
    if not incumbent.is_file():
        raise FileNotFoundError(f"incumbent checkpoint does not exist: {incumbent}")
    incumbent_sha = hashlib.sha256(incumbent.read_bytes()).hexdigest()
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": NEURAL_LONG_HORIZON_SCHEMA,
                    "architecture": "2-layer/4-head/width-48 bounded causal-residual transformer",
                    "learned_parameters": LongHorizonPhysicsAdapter().parameter_count(),
                    "forecast_seconds": FORECAST_SECONDS,
                    "object_counts": [4, 6, 8],
                    "training_steps": parsed.training_steps,
                    "batch_size": parsed.batch_size,
                    "training_examples": parsed.training_steps * parsed.batch_size,
                    "planning_used_as_training_loss": False,
                    "incumbent_checkpoint_sha256": incumbent_sha,
                    "protocol_sha256": neural_long_horizon_protocol_sha256(
                        incumbent_checkpoint_sha256=incumbent_sha,
                        training_steps=parsed.training_steps,
                        batch_size=parsed.batch_size,
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result, model = run_neural_long_horizon(
        incumbent_checkpoint=incumbent,
        training_steps=parsed.training_steps,
        batch_size=parsed.batch_size,
    )
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-neural-long-horizon-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_neural_long_horizon(result, model, run_directory=run_directory)
    print(
        json.dumps(
            {
                "outcome": summary.outcome,
                "passed": result.passed,
                "gate_failures": list(result.gate_failures),
                "run_directory": str(run_directory),
                "run_bytes": summary.artifacts["run_bytes"],
                "training_seconds": result.training_seconds,
                "evaluation_seconds": result.evaluation_seconds,
                "twelve_second_position_rmse_m": {
                    str(item.object_count): item.candidate_position_rmse_m["12.00"]
                    for item in result.scenarios
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
