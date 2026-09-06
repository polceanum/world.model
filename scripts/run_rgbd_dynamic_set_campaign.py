#!/usr/bin/env python3
"""Train/resume an authorized specification-1.61 campaign boundary."""

from __future__ import annotations

import argparse
import json

from world_model.training.dynamic_set_execution import execute_repository_training_block


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an initialized dynamic-set campaign to its next exact 512-update "
            "development-validation boundary."
        )
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    report = execute_repository_training_block(
        run_directory=parsed.run_directory,
        work_directory=parsed.work_directory,
        config_path=parsed.config,
    )
    print(json.dumps(report.to_dict(), allow_nan=False, indent=2, sort_keys=True))
    return 2 if report.status == "limit_hit" else 0


if __name__ == "__main__":
    raise SystemExit(main())
