#!/usr/bin/env python3
"""Run observable sphere/box geometry, contact, and planning qualification."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.rigid_capability import (
    default_rigid_scenarios,
    publish_rigid_capability,
    rigid_manifest_sha256,
    run_rigid_capability,
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
                    "schema": "world_model_observable_rigid_capability_v1",
                    "scenario_count": len(default_rigid_scenarios()),
                    "scenario_manifest_sha256": rigid_manifest_sha256(),
                    "generated_frames_retained": False,
                    "planning_used_as_training_loss": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_rigid_capability()
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-rigid-capability-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_rigid_capability(result, run_directory=run_directory)
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
