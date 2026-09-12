#!/usr/bin/env python3
"""Run the compact N=4/6/8 multi-contact six-DoF scale gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from world_model.evaluation.multicontact_six_dof import (
    MULTICONTACT_SIX_DOF_SCHEMA,
    multicontact_manifest_sha256,
    publish_multicontact_six_dof_scale,
    run_multicontact_six_dof_scale,
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
                    "schema": MULTICONTACT_SIX_DOF_SCHEMA,
                    "scenario_manifest_sha256": multicontact_manifest_sha256(),
                    "object_counts": [4, 6, 8],
                    "mixed_rigid_primitives": True,
                    "known_action_count": 3,
                    "planning_candidate_counts": [8, 32],
                    "dense_serial_planning_oracle": True,
                    "planning_used_as_training_loss": False,
                    "state_first": True,
                    "generated_frames_retained": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_multicontact_six_dof_scale(enforce_latency=True)
    default_name = f"{datetime.now(timezone.utc):%Y%m%d}-multicontact-six-dof-v1"
    run_directory = Path(parsed.run_directory or Path("runs") / default_name)
    summary = publish_multicontact_six_dof_scale(
        result,
        run_directory=run_directory,
    )
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
