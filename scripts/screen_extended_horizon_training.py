#!/usr/bin/env python3
"""Run a disposable tail-focused training screen for the compact world model.

The screen uses the existing single bounded-residual transformer. Its only
objective change is to weight physical-response consequences at 12--48 seconds
so the long tail, rather than a short-horizon average, drives the experiment.
It writes no checkpoint or media.
"""

from __future__ import annotations

import argparse
import json

from world_model.evaluation.neural_adaptive_physics import train_neural_physics_adapter
from world_model.identification import LongHorizonPhysicsAdapter

_TAIL_HORIZONS_SECONDS = (12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.training_steps <= 0 or parsed.batch_size <= 0:
        raise ValueError("training steps and batch size must be positive")
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": "world_model_extended_horizon_training_screen_v1",
                    "architecture": "single 2-layer/4-head/width-48 bounded causal-residual transformer",
                    "training_steps": parsed.training_steps,
                    "batch_size": parsed.batch_size,
                    "stability_horizons_seconds": _TAIL_HORIZONS_SECONDS,
                    "planning_used_as_training_loss": False,
                    "artifact_policy": "no checkpoint, media, or batch data retained",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    model, trace, training_seconds, changed = train_neural_physics_adapter(
        steps=parsed.training_steps,
        batch_size=parsed.batch_size,
        seed=91_724,
        stability_loss_weight=1.0,
        stability_horizons=_TAIL_HORIZONS_SECONDS,
        model_factory=LongHorizonPhysicsAdapter,
        learning_rate=7.5e-4,
        warmup_steps=min(64, parsed.training_steps - 1),
        cosine_decay_to=0.1,
    )
    print(
        json.dumps(
            {
                "schema": "world_model_extended_horizon_training_screen_v1",
                "stability_horizons_seconds": _TAIL_HORIZONS_SECONDS,
                "training_steps": parsed.training_steps,
                "training_seconds": training_seconds,
                "changed_parameter_count": changed,
                "learned_parameter_count": model.parameter_count(),
                "first": trace[0],
                "last": trace[-1],
                "artifact_policy": "no checkpoint, media, or batch data retained",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
