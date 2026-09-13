#!/usr/bin/env python3
"""Run the integrated public RGB-D to N=4/6/8 visual-dynamic scale gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.visual_dynamic_scale import (
    VISUAL_DYNAMIC_SCALE_SCHEMA,
    publish_visual_dynamic_scale,
    run_visual_dynamic_scale,
    visual_dynamic_manifest_sha256,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": VISUAL_DYNAMIC_SCALE_SCHEMA,
                    "scenario_manifest_sha256": visual_dynamic_manifest_sha256(),
                    "object_counts": [4, 6, 8],
                    "runtime_inputs": [
                        "rgb",
                        "depth",
                        "world_from_camera",
                        "intrinsics",
                        "timestamp",
                    ],
                    "runtime_object_prototypes": False,
                    "runtime_owned_ids": True,
                    "birth_confirmation_observations": 2,
                    "retirement_visible_misses": 2,
                    "moving_calibrated_cameras": True,
                    "planning_candidate_counts": [8, 32],
                    "dense_serial_planning_oracle": True,
                    "planning_used_as_training_loss": False,
                    "generated_frames_retained": False,
                    "maximum_per_object_position_error_m": 0.015,
                    "maximum_per_object_velocity_error_mps": 0.075,
                    "maximum_per_box_orientation_error_degrees": 6.0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_visual_dynamic_scale(enforce_latency=True)
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-visual-dynamic-scale-v3"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_visual_dynamic_scale(result, run_directory=run_directory)
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
