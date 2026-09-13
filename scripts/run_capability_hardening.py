#!/usr/bin/env python3
"""Run the cross-capability accuracy and non-regression hardening gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.capability_hardening import (
    CAPABILITY_HARDENING_SCHEMA,
    capability_hardening_manifest_sha256,
    publish_capability_hardening,
    run_capability_hardening,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory")
    parser.add_argument("--config", default="configs/rgbd_dynamic_set_planning_cpu.yaml")
    parser.add_argument(
        "--checkpoint",
        default="runs/20260907-capability-development-v2/selected_checkpoint.pt",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce-in-process-latency", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": CAPABILITY_HARDENING_SCHEMA,
                    "manifest_sha256": capability_hardening_manifest_sha256(),
                    "source_tiers": [
                        "long_horizon",
                        "integrated",
                        "open_world",
                        "touching_recovery",
                        "multicontact",
                        "visual_dynamic",
                    ],
                    "maximum_accuracy_regression": 0.02,
                    "planning_used_as_training_loss": False,
                    "generated_frames_retained": False,
                    "absolute_latency_adjudication": "separate fresh-process source runners",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_capability_hardening(
        model_config_path=parsed.config,
        checkpoint_path=parsed.checkpoint,
        enforce_latency=parsed.enforce_in_process_latency,
    )
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-capability-hardening-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_capability_hardening(result, run_directory=run_directory)
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
