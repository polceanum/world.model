#!/usr/bin/env python3
"""Run the fixed one-sphere differentiable convergence ladder once."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from world_model.training.checkpointing import capture_git_metadata, save_checkpoint
from world_model.training.minimal_toy import (
    MEASUREMENT_UPDATES,
    ROLLOUT_UPDATES,
    ConvergenceGateError,
    DifferentiableToyStateEstimator,
    run_minimal_toy_ladder,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.io import atomic_write_text

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
        default=Path("runs/minimal_differentiable_toy_v2/report.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/minimal_differentiable_toy_v2/model.pt"),
    )
    return parser.parse_args()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"minimal ladder refuses to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_paths(report: Path, checkpoint: Path) -> None:
    resolved_report = report.resolve()
    resolved_checkpoint = checkpoint.resolve()
    if resolved_report == resolved_checkpoint:
        raise ValueError("minimal ladder report and checkpoint paths must be distinct")
    existing = [path for path in (report, checkpoint) if path.exists()]
    if existing:
        joined = ", ".join(str(path.resolve()) for path in existing)
        raise FileExistsError("minimal ladder refuses to overwrite existing evidence: " + joined)


def _write_checkpoint(
    path: Path,
    *,
    model: DifferentiableToyStateEstimator,
    config: OrpheusConfig,
    report: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> Path:
    """Write a project-compatible, weights-only toy initialization atomically."""

    if path.exists():
        raise FileExistsError(f"minimal ladder refuses to overwrite checkpoint: {path}")
    return save_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        step=MEASUREMENT_UPDATES + ROLLOUT_UPDATES,
        metrics={
            "artifact_kind": "minimal_differentiable_toy_weights_only",
            "exact_resume": False,
            "protocol": report["protocol"],
            "final_test": report["final_test"],
        },
        device="cpu",
        source_provenance=source_provenance,
    )


def _write_success_artifacts(
    *,
    report_path: Path,
    checkpoint_path: Path,
    model: DifferentiableToyStateEstimator,
    config: OrpheusConfig,
    report: dict[str, Any],
    source_provenance: Mapping[str, Any],
) -> None:
    _validate_output_paths(report_path, checkpoint_path)
    report["source_provenance"] = dict(source_provenance)
    report["checkpoint"] = str(checkpoint_path.resolve())
    report["checkpoint_kind"] = "project_compatible_weights_only"
    _write_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        report=report,
        source_provenance=source_provenance,
    )
    report["checkpoint_sha256"] = _sha256_file(checkpoint_path)
    _write_report(report_path, report)


def main() -> int:
    args = _arguments()
    _validate_output_paths(args.report, args.checkpoint)
    config = load_config(args.config)
    source_provenance = capture_git_metadata(_REPOSITORY_ROOT)
    try:
        model, report = run_minimal_toy_ladder(config)
    except ConvergenceGateError as error:
        error.report["source_provenance"] = dict(source_provenance)
        _write_report(args.report, error.report)
        print(f"FAILED: {error}")
        print(f"report: {args.report.resolve()}")
        return 1

    _write_success_artifacts(
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        model=model,
        config=config,
        report=report,
        source_provenance=source_provenance,
    )
    print("PASSED: minimal differentiable toy convergence ladder")
    print(json.dumps(report["final_test"], allow_nan=False, indent=2, sort_keys=True))
    print(f"report: {args.report.resolve()}")
    print(f"checkpoint: {args.checkpoint.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
