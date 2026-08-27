#!/usr/bin/env python3
"""Thin CLI for the frozen public RGB-D bridge qualification harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from world_model.training.checkpointing import capture_git_metadata
from world_model.training.rgbd_online_bridge_qualification import (
    ARCHITECTURE_ATTEMPT,
    bridge_protocol,
    require_frozen_config,
    run_development,
    run_qualification,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPOSITORY_ROOT / "runs" / "rgbd_online_bridge_v1"
# Deliberately fixed: changing a CLI path cannot manufacture a second final attempt.
QUALIFICATION_LEDGER = DEFAULT_RUN / f"qualification_attempt_{ARCHITECTURE_ATTEMPT}_access.json"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze/inspect the RGB-D online bridge protocol, evaluate development, "
            "or consume the one-shot reviewed protected ladder."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("protocol", "development", "qualification"),
        default="protocol",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rgbd_online_free_motion_cpu.yaml"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--reviewed-checkpoint-sha256")
    parser.add_argument("--reviewed-report-sha256")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.phase == "protocol":
        print(json.dumps(bridge_protocol(), allow_nan=False, indent=2, sort_keys=True))
        return 0

    config_path = args.config.resolve()
    config = require_frozen_config(config_path)
    torch.set_num_threads(1)
    source = capture_git_metadata(REPOSITORY_ROOT)
    checkpoint_path = (args.checkpoint or (DEFAULT_RUN / "development_model.pt")).resolve()

    if args.phase == "development":
        report_path = (args.report or (DEFAULT_RUN / "development_report.json")).resolve()
        result = run_development(
            config,
            config_path=config_path,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            source_provenance=source,
        )
        print(
            "PASSED: development is review-ready; protected data remains unopened"
            if result == 0
            else "FAILED: development gates; protected data remains unopened"
        )
        print(f"report: {report_path}")
        if result == 0:
            print(f"checkpoint: {checkpoint_path}")
        return result

    if args.development_report is None:
        raise ValueError("qualification requires an explicit --development-report")
    report_path = (args.report or (DEFAULT_RUN / "qualification_report.json")).resolve()
    result = run_qualification(
        config,
        config_path=config_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        development_report_path=args.development_report.resolve(),
        ledger_path=QUALIFICATION_LEDGER.resolve(),
        reviewed_checkpoint_sha256=args.reviewed_checkpoint_sha256,
        reviewed_report_sha256=args.reviewed_report_sha256,
        source_provenance=source,
    )
    print(
        "PASSED: selector, confirmation, and one-shot final bridge gates"
        if result == 0
        else "FAILED: protected bridge qualification stopped before a later split"
    )
    print(f"report: {report_path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
