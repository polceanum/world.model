#!/usr/bin/env python3
"""Thin CLI for the frozen two-visible orbital-camera RGB-D qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from world_model.training.checkpointing import capture_git_metadata
from world_model.training.rgbd_two_visible_orbital_camera_qualification import (
    bridge_protocol,
    canonical_checkpoint_path,
    canonical_development_report_path,
    canonical_qualification_report_path,
    development_ledger_path,
    qualification_ledger_path,
    require_frozen_config,
    run_development,
    run_qualification,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_two_visible_orbital_camera_cpu.yaml"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the two-visible orbital-camera RGB-D protocol, run development only, or "
            "consume its reviewed exactly-once protected ladder."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("protocol", "development", "qualification"),
        default="protocol",
    )
    parser.add_argument("--reviewed-checkpoint-sha256")
    parser.add_argument("--reviewed-report-sha256")
    parser.add_argument("--reviewed-development-ledger-sha256")
    parsed = parser.parse_args(argv)
    if parsed.phase == "qualification":
        missing = [
            option
            for option, value in (
                ("--reviewed-checkpoint-sha256", parsed.reviewed_checkpoint_sha256),
                ("--reviewed-report-sha256", parsed.reviewed_report_sha256),
                (
                    "--reviewed-development-ledger-sha256",
                    parsed.reviewed_development_ledger_sha256,
                ),
            )
            if value is None
        ]
        if missing:
            parser.error("qualification requires " + ", ".join(missing))
    return parsed


def main() -> int:
    args = arguments()
    if args.phase == "protocol":
        print(json.dumps(bridge_protocol(), allow_nan=False, indent=2, sort_keys=True))
        return 0
    config_path = CONFIG_PATH
    config = require_frozen_config(config_path)
    torch.set_num_threads(1)
    source = capture_git_metadata(REPOSITORY_ROOT)
    checkpoint_path = canonical_checkpoint_path()
    if args.phase == "development":
        report_path = canonical_development_report_path()
        result = run_development(
            config,
            config_path=config_path,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            source_provenance=source,
        )
        print(
            "PASSED: two-visible orbital-camera development is review-ready; protected data remains unopened"
            if result == 0
            else "FAILED: two-visible orbital-camera development gates; protected data remains unopened"
        )
        print(f"report: {report_path}")
        print(f"development ledger: {development_ledger_path()}")
        if result == 0:
            print(f"checkpoint: {checkpoint_path}")
        return result
    report_path = canonical_qualification_report_path()
    result = run_qualification(
        config,
        config_path=config_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        development_report_path=canonical_development_report_path(),
        reviewed_checkpoint_sha256=args.reviewed_checkpoint_sha256,
        reviewed_report_sha256=args.reviewed_report_sha256,
        reviewed_development_ledger_sha256=args.reviewed_development_ledger_sha256,
        source_provenance=source,
    )
    print(
        "PASSED: selector, confirmation, and one-shot final two-visible orbital-camera gates"
        if result == 0
        else "FAILED: protected qualification stopped before a later split"
    )
    print(f"report: {report_path}")
    print(f"qualification ledger: {qualification_ledger_path()}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
