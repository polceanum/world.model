#!/usr/bin/env python3
"""Train and evaluate the compact neural-adaptive world-model bridge."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.neural_adaptive_physics import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRAINING_STEPS,
    NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
    neural_adaptive_protocol_sha256,
    publish_neural_adaptive_physics,
    run_neural_adaptive_physics,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory")
    parser.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--skip-public-scenarios", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
                    "architecture": {
                        "type": "objectwise public-evidence set transformer",
                        "layers": 2,
                        "heads": 4,
                        "width": 48,
                        "feed_forward_width": 96,
                    },
                    "training_steps": parsed.training_steps,
                    "batch_size": parsed.batch_size,
                    "training_examples": parsed.training_steps * parsed.batch_size,
                    "protocol_sha256": neural_adaptive_protocol_sha256(
                        training_steps=parsed.training_steps,
                        batch_size=parsed.batch_size,
                    ),
                    "runtime_adaptation": "observation-conditioned forward pass",
                    "online_backpropagation": False,
                    "planning_used_as_training_loss": False,
                    "public_scenarios": not parsed.skip_public_scenarios,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result, model = run_neural_adaptive_physics(
        training_steps=parsed.training_steps,
        batch_size=parsed.batch_size,
        evaluate_public_scenarios=not parsed.skip_public_scenarios,
    )
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-neural-adaptive-physics-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_neural_adaptive_physics(
        result,
        model,
        run_directory=run_directory,
    )
    print(
        json.dumps(
            {
                "outcome": summary.outcome,
                "passed": result.passed,
                "gate_failures": list(result.gate_failures),
                "run_directory": str(run_directory),
                "run_bytes": summary.artifacts["run_bytes"],
                "learned_parameters": result.learned_parameter_count,
                "nonzero_learned_parameters": result.nonzero_learned_parameter_count,
                "training_loss": [result.training_loss_initial, result.training_loss_final],
                "development_mean_relative_error": result.development.mean_relative_error,
                "ood_mean_relative_error": result.compositional_ood.mean_relative_error,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
