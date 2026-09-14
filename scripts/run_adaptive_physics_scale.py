#!/usr/bin/env python3
"""Run the single-model N=4/6/8 adaptive-physics scale gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.adaptive_physics_scale import (
    ADAPTIVE_PHYSICS_SCALE_SCHEMA,
    adaptive_physics_manifest_sha256,
    publish_adaptive_physics_scale,
    run_adaptive_physics_scale,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-latency-gates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": ADAPTIVE_PHYSICS_SCALE_SCHEMA,
                    "scenario_manifest_sha256": adaptive_physics_manifest_sha256(),
                    "single_model": True,
                    "ensemble": False,
                    "object_counts": [4, 6, 8],
                    "parameter_evidence": [
                        "public RGB-D free-motion position traces",
                        "public RGB-D traces around known impulses",
                        "public RGB-D traces around stationary-boundary impacts",
                    ],
                    "heterogeneous_parameters": [
                        "mass",
                        "drag",
                        "restitution",
                        "friction",
                    ],
                    "forecast_seconds": 4.0,
                    "planning_candidate_counts": [8, 32],
                    "planning_used_as_training_loss": False,
                    "generated_frames_retained": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_adaptive_physics_scale(enforce_latency=not parsed.no_latency_gates)
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-adaptive-physics-scale-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_adaptive_physics_scale(result, run_directory=run_directory)
    print(
        json.dumps(
            {
                "outcome": summary.outcome,
                "qualified": result.qualified,
                "gate_failures": list(result.gate_failures),
                "run_directory": str(run_directory),
                "run_bytes": summary.artifacts["run_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
