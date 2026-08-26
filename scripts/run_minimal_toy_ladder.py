#!/usr/bin/env python3
"""Run the fixed one-sphere differentiable convergence ladder once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from world_model.training.minimal_toy import (
    ConvergenceGateError,
    run_minimal_toy_ladder,
)
from world_model.utils.config import load_config
from world_model.utils.io import atomic_write_text


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared oracle -> RGB measurement -> short rollout "
            "convergence ladder. Later rungs never run after a failed gate."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/minimal_differentiable_toy_cpu.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/minimal_differentiable_toy/report.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/minimal_differentiable_toy/model.pt"),
    )
    return parser.parse_args()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = _arguments()
    existing = [path for path in (args.report, args.checkpoint) if path.exists()]
    if existing:
        joined = ", ".join(str(path.resolve()) for path in existing)
        raise FileExistsError(
            "minimal ladder refuses to overwrite existing evidence: " + joined
        )
    config = load_config(args.config)
    try:
        model, report = run_minimal_toy_ladder(config)
    except ConvergenceGateError as error:
        _write_report(args.report, error.report)
        print(f"FAILED: {error}")
        print(f"report: {args.report.resolve()}")
        return 1

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "protocol": report["protocol"],
            "final_test": report["final_test"],
        },
        args.checkpoint,
    )
    report["checkpoint"] = str(args.checkpoint.resolve())
    _write_report(args.report, report)
    print("PASSED: minimal differentiable toy convergence ladder")
    print(json.dumps(report["final_test"], indent=2, sort_keys=True))
    print(f"report: {args.report.resolve()}")
    print(f"checkpoint: {args.checkpoint.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
