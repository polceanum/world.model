#!/usr/bin/env python3
"""Develop or qualify the frozen parameter-free RGB-D temporal rung."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from world_model.training.checkpointing import (
    capture_git_metadata,
    load_model_weights,
    save_checkpoint,
)
from world_model.training.rgbd_temporal_free_motion import (
    ARCHITECTURE_ATTEMPT,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    FINAL_TEST_SEEDS,
    FROZEN_CONFIG_SHA256,
    SELECTOR_SEEDS,
    evaluate_seed_manifest,
    gate_failures,
    new_estimator,
    temporal_protocol,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.io import atomic_write_text

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RUN = _REPOSITORY_ROOT / "runs" / "rgbd_temporal_free_motion_v1"
_QUALIFICATION_LEDGER = _DEFAULT_RUN / f"qualification_attempt_{ARCHITECTURE_ATTEMPT}_access.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the zero-optimizer RGB-D development manifest, or pass a "
            "reviewed empty-state checkpoint through selector, confirmation, and final."
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
        default=Path("configs/rgbd_temporal_free_motion_cpu.yaml"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--reviewed-checkpoint-sha256")
    parser.add_argument("--reviewed-report-sha256")
    return parser.parse_args()


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_sha256(value: str | None, *, label: str) -> str:
    if value is None or len(value) != 64:
        raise ValueError(f"qualification requires a 64-character {label}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"qualification {label} must be hexadecimal") from error
    return value.lower()


def _clean_source(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    required = {
        "commit",
        "dirty",
        "worktree_fingerprint",
        "runtime_source_fingerprint",
    }
    if set(value) != required:
        raise ValueError(f"{label} source provenance must contain exactly {sorted(required)}")
    normalized = dict(value)
    if normalized["dirty"] is not False:
        raise ValueError(f"{label} requires a clean committed worktree")
    if not isinstance(normalized["commit"], str) or len(normalized["commit"]) != 40:
        raise ValueError(f"{label} requires an exact Git commit")
    _validated_sha256(normalized["worktree_fingerprint"], label="worktree fingerprint")
    _validated_sha256(
        normalized["runtime_source_fingerprint"],
        label="runtime source fingerprint",
    )
    return normalized


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"RGB-D temporal runner refuses to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        json.dumps(dict(report), allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _validate_distinct_paths(
    paths: Mapping[str, Path],
    *,
    atomic_writers: tuple[str, ...],
) -> None:
    """Reject artifact paths that alias another writer's temporary file."""

    resolved = {name: path.resolve() for name, path in paths.items()}
    for name in atomic_writers:
        if name not in paths:
            raise ValueError(f"unknown atomic artifact {name!r}")
        resolved[f"{name}_atomic_temporary"] = _atomic_temporary(paths[name]).resolve()
    aliases: dict[Path, list[str]] = {}
    for name, path in resolved.items():
        aliases.setdefault(path, []).append(name)
    collisions = [names for names in aliases.values() if len(names) > 1]
    if collisions:
        detail = "; ".join(", ".join(names) for names in collisions)
        raise ValueError("RGB-D temporal artifact paths must be distinct: " + detail)


def _require_frozen_config(path: Path) -> OrpheusConfig:
    digest = _sha256_file(path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "RGB-D temporal qualification requires the exact frozen config bytes: "
            f"expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    return load_config(path)


class _QualificationLedger:
    """Exclusive durable receipt written before each protected materialization."""

    _ORDER = ("selector", "confirmation", "final_test")

    def __init__(self, path: Path, initial: Mapping[str, Any]) -> None:
        self.path = path
        self.record = dict(initial)
        self.record["access_started"] = {split: False for split in self._ORDER}
        self.record["status"] = "reserved_before_protected_access"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(self._serialized())
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            # Ambiguity consumes the one protected attempt; retain the path.
            raise

    def _serialized(self) -> str:
        return json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True) + "\n"

    def _replace(self) -> None:
        atomic_write_text(self.path, self._serialized())

    def record_access(self, split: str) -> None:
        accessed = self.record["access_started"]
        next_index = sum(bool(accessed[name]) for name in self._ORDER)
        expected = self._ORDER[next_index] if next_index < len(self._ORDER) else None
        if split != expected:
            raise RuntimeError(f"protected access order violation: expected {expected!r}")
        accessed[split] = True
        self.record["status"] = f"{split}_materialization_started"
        self._replace()

    def finish(self, report: Mapping[str, Any]) -> None:
        self.record["status"] = "complete" if report.get("passed") is True else "failed"
        self.record["stopped_after"] = report.get("stopped_after")
        self.record["report_summary_sha256"] = _canonical_sha256(report)
        self.record["protected_data_materialized"] = any(
            bool(value) for value in self.record["access_started"].values()
        )
        self._replace()


def _development(
    config: OrpheusConfig,
    *,
    report_path: Path,
    checkpoint_path: Path,
    source: Mapping[str, Any],
) -> int:
    clean_source = _clean_source(source, label="RGB-D temporal development")
    _validate_distinct_paths(
        {"report": report_path, "checkpoint": checkpoint_path},
        atomic_writers=("report", "checkpoint"),
    )
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("development evidence paths must both be fresh")

    estimator = new_estimator(config)
    development = evaluate_seed_manifest(
        estimator,
        config,
        DEVELOPMENT_SEEDS,
        split="development",
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_temporal_free_motion_development",
        "protocol": temporal_protocol(),
        "source_provenance": clean_source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development": development,
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "passed": development["passed"],
        "review_ready": development["passed"],
        "stopped_after": "development",
    }
    source_after = _clean_source(
        capture_git_metadata(_REPOSITORY_ROOT),
        label="RGB-D temporal development completion",
    )
    if source_after != clean_source:
        raise RuntimeError("source provenance changed during development evaluation")
    if not development["passed"]:
        _write_report(report_path, report)
        print("FAILED: RGB-D temporal development gates")
        print(f"report: {report_path.resolve()}")
        return 1

    save_checkpoint(
        checkpoint_path,
        model=estimator,
        optimizer=None,
        scheduler=None,
        config=config,
        step=0,
        metrics={
            "artifact_kind": "rgbd_temporal_parameter_free_empty_state",
            "exact_resume": False,
            "optimizer_updates": 0,
            "protocol": report["protocol"],
            "development": development,
        },
        device="cpu",
        source_provenance=clean_source,
    )
    report["checkpoint"] = str(checkpoint_path.resolve())
    report["checkpoint_sha256"] = _sha256_file(checkpoint_path)
    report["checkpoint_model_state"] = "empty_parameter_free"
    _write_report(report_path, report)
    print("PASSED: RGB-D temporal development gates; protected data remains unopened")
    print(f"report: {report_path.resolve()}")
    print(f"checkpoint: {checkpoint_path.resolve()}")
    return 0


def _validate_development_evidence(
    development_report: Mapping[str, Any],
    *,
    checkpoint_digest: str,
    clean_source: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Fail closed unless reviewed JSON contains complete passing development evidence."""

    if development_report.get("artifact_kind") != "rgbd_temporal_free_motion_development":
        raise ValueError("reviewed development evidence has the wrong artifact kind")
    if (
        development_report.get("passed") is not True
        or development_report.get("review_ready") is not True
    ):
        raise ValueError("reviewed development evidence did not pass")
    if development_report.get("protected_data_materialized") is not False:
        raise ValueError("development evidence must prove protected data remained unopened")
    if development_report.get("optimizer_updates") != 0:
        raise ValueError("development evidence must prove zero optimizer updates")
    if development_report.get("stopped_after") != "development":
        raise ValueError("reviewed evidence must stop after the development split")
    if _canonical_sha256(development_report.get("protocol")) != _canonical_sha256(
        temporal_protocol()
    ):
        raise ValueError("reviewed development protocol differs from frozen source")
    if development_report.get("config_sha256") != FROZEN_CONFIG_SHA256:
        raise ValueError("reviewed development config hash differs from frozen config")
    if development_report.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("reviewed development report does not bind the checkpoint")
    if development_report.get("source_provenance") != clean_source:
        raise ValueError("reviewed development source differs from current clean source")
    development = development_report.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("reviewed report is missing its development split evidence")
    expected_seeds = list(DEVELOPMENT_SEEDS)
    if development.get("split") != "development":
        raise ValueError("reviewed development split has the wrong name")
    if development.get("seeds") != expected_seeds:
        raise ValueError("reviewed development split has the wrong seed manifest")
    if development.get("seed_manifest_sha256") != _canonical_sha256(expected_seeds):
        raise ValueError("reviewed development split has the wrong seed-manifest hash")
    if development.get("optimizer_updates") != 0:
        raise ValueError("reviewed development split must prove zero optimizer updates")
    if development.get("uncertainty_claim") != (
        "iid_ols_residual_diagnostic_not_calibrated_posterior"
    ):
        raise ValueError("reviewed development split overstates its uncertainty claim")
    metrics = development.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("reviewed development split is missing scalar metrics")
    recomputed_failures = gate_failures(metrics)
    if development.get("failures") != recomputed_failures or recomputed_failures:
        raise ValueError("reviewed development gates do not recompute as passed")
    if development.get("passed") is not True:
        raise ValueError("reviewed development split did not pass")
    return development


def _validate_checkpoint_evidence(
    payload: Mapping[str, Any],
    *,
    development: Mapping[str, Any],
    clean_source: Mapping[str, Any],
    expected_config: OrpheusConfig,
) -> None:
    """Bind the empty-state checkpoint to the same protocol, split, and source."""

    if (
        payload.get("step") != 0
        or payload.get("optimizer_state") is not None
        or payload.get("scheduler_state") is not None
    ):
        raise ValueError("parameter-free checkpoint must be step zero without optimizer state")
    checkpoint_metrics = payload.get("metrics")
    if not isinstance(checkpoint_metrics, Mapping):
        raise ValueError("parameter-free checkpoint is missing evidence metrics")
    if checkpoint_metrics.get("artifact_kind") != "rgbd_temporal_parameter_free_empty_state":
        raise ValueError("parameter-free checkpoint has the wrong artifact kind")
    if checkpoint_metrics.get("optimizer_updates") != 0:
        raise ValueError("parameter-free checkpoint must prove zero optimizer updates")
    if checkpoint_metrics.get("protocol") != temporal_protocol():
        raise ValueError("parameter-free checkpoint protocol differs from frozen source")
    if checkpoint_metrics.get("development") != development:
        raise ValueError("parameter-free checkpoint does not bind reviewed development evidence")
    if payload.get("git") != clean_source:
        raise ValueError("parameter-free checkpoint source differs from current clean source")
    if payload.get("config") != expected_config.to_dict():
        raise ValueError("parameter-free checkpoint config differs from exact frozen config")


def _qualification(
    config: OrpheusConfig,
    *,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    reviewed_checkpoint_sha256: str | None,
    reviewed_report_sha256: str | None,
    source: Mapping[str, Any],
) -> int:
    clean_source = _clean_source(source, label="RGB-D temporal qualification")
    _validate_distinct_paths(
        {
            "report": report_path,
            "checkpoint": checkpoint_path,
            "development_report": development_report_path,
            "qualification_ledger": _QUALIFICATION_LEDGER,
        },
        atomic_writers=("report", "qualification_ledger"),
    )
    checkpoint_digest = _validated_sha256(
        reviewed_checkpoint_sha256,
        label="checkpoint SHA-256",
    )
    report_digest = _validated_sha256(reviewed_report_sha256, label="report SHA-256")
    if not checkpoint_path.is_file() or not development_report_path.is_file():
        raise FileNotFoundError("reviewed development checkpoint and report must both exist")
    if _sha256_file(checkpoint_path) != checkpoint_digest:
        raise ValueError("reviewed checkpoint SHA-256 does not match bytes read")
    report_bytes = development_report_path.read_bytes()
    if _sha256_bytes(report_bytes) != report_digest:
        raise ValueError("reviewed development-report SHA-256 does not match bytes read")
    development_report = json.loads(report_bytes)
    if not isinstance(development_report, Mapping):
        raise ValueError("reviewed development report must be a JSON object")
    development = _validate_development_evidence(
        development_report,
        checkpoint_digest=checkpoint_digest,
        clean_source=clean_source,
    )
    if report_path.exists() or _QUALIFICATION_LEDGER.exists():
        raise FileExistsError("qualification report and access ledger must both be fresh")
    estimator = new_estimator(config)
    payload = load_model_weights(
        checkpoint_path,
        model=estimator,
        expected_config=config,
    )
    _validate_checkpoint_evidence(
        payload,
        development=development,
        clean_source=clean_source,
        expected_config=config,
    )
    if estimator.state_dict():
        raise RuntimeError("parameter-free estimator unexpectedly loaded persistent tensors")

    ledger = _QualificationLedger(
        _QUALIFICATION_LEDGER,
        {
            "protocol_sha256": temporal_protocol()["protocol_sha256"],
            "source_provenance": clean_source,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "reviewed_checkpoint_sha256": checkpoint_digest,
            "reviewed_report_sha256": report_digest,
        },
    )
    report: dict[str, Any] = {
        "artifact_kind": "rgbd_temporal_free_motion_protected_qualification",
        "protocol": temporal_protocol(),
        "source_provenance": clean_source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "reviewed_checkpoint_sha256": checkpoint_digest,
        "reviewed_report_sha256": report_digest,
        "optimizer_updates": 0,
        "passed": False,
    }
    try:
        for split, seeds in (
            ("selector", SELECTOR_SEEDS),
            ("confirmation", CONFIRMATION_SEEDS),
            ("final_test", FINAL_TEST_SEEDS),
        ):
            ledger.record_access(split)
            result = evaluate_seed_manifest(estimator, config, seeds, split=split)
            report[split] = result
            report["stopped_after"] = split
            if not result["passed"]:
                report["failures"] = list(result["failures"])
                break
        else:
            report["passed"] = True
            report["failures"] = []
        report["protected_data_materialized"] = True
        report["checkpoint_model_state"] = "empty_parameter_free_unchanged"
        source_after = _clean_source(
            capture_git_metadata(_REPOSITORY_ROOT),
            label="RGB-D temporal qualification completion",
        )
        if source_after != clean_source:
            raise RuntimeError("source provenance changed during protected qualification")
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        ledger.finish(report)
        _write_report(report_path, report)
        raise

    ledger.finish(report)
    _write_report(report_path, report)
    if not report["passed"]:
        print(f"FAILED: RGB-D temporal qualification stopped after {report['stopped_after']}")
        print(f"report: {report_path.resolve()}")
        return 1
    print("PASSED: RGB-D temporal selector, confirmation, and one-shot final gates")
    print(f"report: {report_path.resolve()}")
    return 0


def main() -> int:
    args = _arguments()
    if args.phase == "protocol":
        print(json.dumps(temporal_protocol(), allow_nan=False, indent=2, sort_keys=True))
        return 0

    config = _require_frozen_config(args.config)
    torch.set_num_threads(1)
    source = capture_git_metadata(_REPOSITORY_ROOT)
    if args.phase == "development":
        report = args.report or (_DEFAULT_RUN / "development_report.json")
        checkpoint = args.checkpoint or (_DEFAULT_RUN / "development_model.pt")
        return _development(
            config,
            report_path=report,
            checkpoint_path=checkpoint,
            source=source,
        )

    if args.development_report is None:
        raise ValueError("qualification requires --development-report")
    report = args.report or (_DEFAULT_RUN / "qualification_report.json")
    checkpoint = args.checkpoint or (_DEFAULT_RUN / "development_model.pt")
    return _qualification(
        config,
        report_path=report,
        checkpoint_path=checkpoint,
        development_report_path=args.development_report,
        reviewed_checkpoint_sha256=args.reviewed_checkpoint_sha256,
        reviewed_report_sha256=args.reviewed_report_sha256,
        source=source,
    )


if __name__ == "__main__":
    raise SystemExit(main())
