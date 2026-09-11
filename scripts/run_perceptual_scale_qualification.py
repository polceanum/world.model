#!/usr/bin/env python3
"""Run compact full RGB-D qualification for visible sets of seven and eight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.perceptual_scale_qualification import (
    default_perceptual_scale_scenarios,
    perceptual_scale_manifest_sha256,
    publish_perceptual_scale_qualification,
    run_perceptual_scale_qualification,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/rgbd_dynamic_set_planning_cpu.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/20260907-capability-development-v2/selected_checkpoint.pt",
    )
    parser.add_argument("--run-directory")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": "world_model_perceptual_scale_qualification_v1",
                    "scenario_count": len(default_perceptual_scale_scenarios()),
                    "scenario_manifest_sha256": perceptual_scale_manifest_sha256(),
                    "object_counts": [7, 8],
                    "generated_episodes_retained": False,
                    "planning_used_as_training_loss": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_perceptual_scale_qualification(
        model_config_path=parsed.config,
        checkpoint_path=parsed.checkpoint,
    )
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-perceptual-scale-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_perceptual_scale_qualification(
        result,
        run_directory=run_directory,
        model_config_path=parsed.config,
        checkpoint_path=parsed.checkpoint,
    )
    print(
        json.dumps(
            {
                "outcome": summary.outcome,
                "full_perceptual_qualification": result.full_perceptual_qualification,
                "gate_failures": list(result.gate_failures),
                "run_directory": str(run_directory),
                "run_bytes": summary.artifacts["run_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.full_perceptual_qualification else 2


if __name__ == "__main__":
    raise SystemExit(main())
