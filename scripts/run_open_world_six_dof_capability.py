#!/usr/bin/env python3
"""Run compact prototype-free six-DoF world-model qualification."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.open_world_six_dof import (
    OPEN_WORLD_SIX_DOF_SCHEMA,
    capability_manifest_sha256,
    publish_open_world_six_dof_capability,
    run_open_world_six_dof_capability,
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
                    "schema": OPEN_WORLD_SIX_DOF_SCHEMA,
                    "scenario_manifest_sha256": capability_manifest_sha256(),
                    "runtime_object_prototypes": False,
                    "six_dof_contact": True,
                    "online_parameter_identification": True,
                    "pose_planning_candidate_counts": [8, 32],
                    "generated_frames_retained": False,
                    "planning_used_as_training_loss": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_open_world_six_dof_capability()
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-open-world-six-dof-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_open_world_six_dof_capability(result, run_directory=run_directory)
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
