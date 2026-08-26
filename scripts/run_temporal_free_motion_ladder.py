#!/usr/bin/env python3
"""Train or qualify the fixed temporal free-motion rung."""

from __future__ import annotations

import argparse
import hashlib
import io
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
from world_model.training.temporal_free_motion_toy import (
    DEVELOPMENT_UPDATES,
    MAXIMUM_ARCHITECTURE_ATTEMPTS,
    TemporalFreeMotionEstimator,
    TemporalQualificationError,
    run_development,
    run_protected_qualification,
    temporal_protocol,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.io import atomic_write_text

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_QUALIFICATION_LEDGER_PATH = (
    _REPOSITORY_ROOT
    / "runs"
    / "temporal_free_motion_toy_v1"
    / f"qualification_attempt_{MAXIMUM_ARCHITECTURE_ATTEMPTS}_access.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a development-only temporal review checkpoint, or run the "
            "reviewed checkpoint through selector, confirmation, and one-shot final gates."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("development", "qualification"),
        default="development",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/temporal_free_motion_toy_cpu.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/temporal_free_motion_toy_v1/development_report.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/temporal_free_motion_toy_v1/development_model.pt"),
        help=(
            "Fresh output for development; reviewed development checkpoint input for qualification."
        ),
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        help="Reviewed development report input; required for qualification.",
    )
    parser.add_argument(
        "--reviewed-checkpoint-sha256",
        help="Independent-review checksum; required for qualification.",
    )
    parser.add_argument(
        "--reviewed-report-sha256",
        help="Independent-review development-report checksum; required for qualification.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_sha256(value: str | None, *, name: str) -> str:
    if value is None or len(value) != 64:
        raise ValueError(f"qualification requires a 64-character {name}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"qualification {name} must be hexadecimal") from error
    return value.lower()


def _require_clean_source_provenance(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
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
        raise ValueError(f"{label} requires a clean Git worktree")
    commit = normalized["commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{label} requires an exact 40-character Git commit")
    _validated_sha256(normalized["worktree_fingerprint"], name="worktree fingerprint")
    _validated_sha256(
        normalized["runtime_source_fingerprint"],
        name="runtime source fingerprint",
    )
    return normalized


def _read_bytes_once(path: Path) -> tuple[bytes, str]:
    contents = path.read_bytes()
    return contents, hashlib.sha256(contents).hexdigest()


def _assert_finite_model_state(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint model_state must map strings to tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise FloatingPointError(f"checkpoint model tensor {name!r} is nonfinite")


class _QualificationLedger:
    """Durable, exclusive access receipt for the single protected attempt."""

    _ORDER = ("selector", "confirmation", "final_test")

    def __init__(self, path: Path, initial: Mapping[str, Any]) -> None:
        self.path = path
        self.record = dict(initial)
        self.record["access_started"] = {split: False for split in self._ORDER}
        self.record["status"] = "reserved_before_protected_access"
        self._create_exclusive()

    def _serialized(self) -> str:
        return json.dumps(self.record, allow_nan=False, indent=2, sort_keys=True) + "\n"

    def _create_exclusive(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(self._serialized())
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            # The exclusive path is deliberately retained even if persistence
            # fails: ambiguity must consume the one protected attempt.
            raise

    def _replace(self) -> None:
        atomic_write_text(self.path, self._serialized())

    def record_access(self, split: str) -> None:
        if split not in self._ORDER:
            raise ValueError(f"unknown protected split {split!r}")
        accessed = self.record["access_started"]
        expected_index = sum(bool(accessed[name]) for name in self._ORDER)
        if expected_index >= len(self._ORDER) or self._ORDER[expected_index] != split:
            expected = self._ORDER[expected_index] if expected_index < len(self._ORDER) else None
            raise RuntimeError(f"protected access order violation: expected {expected!r}")
        accessed[split] = True
        self.record["status"] = f"{split}_materialization_started"
        self._replace()

    def finish(self, report: Mapping[str, Any]) -> None:
        self.record["status"] = "complete" if report.get("passed") is True else "failed"
        self.record["stopped_after"] = report.get("stopped_after")
        self.record["protected_data_materialized"] = any(
            bool(value) for value in self.record["access_started"].values()
        )
        self.record["report_summary_sha256"] = _canonical_sha256(report)
        self._replace()


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"temporal ladder refuses to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        json.dumps(dict(report), allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _validate_paths(
    phase: str,
    report: Path,
    checkpoint: Path,
    *,
    development_report: Path | None = None,
    ledger_path: Path = _QUALIFICATION_LEDGER_PATH,
) -> None:
    paths = {"report": report, "checkpoint": checkpoint}
    if phase == "qualification":
        if development_report is None:
            raise ValueError("qualification requires --development-report")
        paths["development_report"] = development_report
        paths["qualification_ledger"] = ledger_path
    resolved: dict[str, Path] = {}
    for name, path in paths.items():
        resolved[name] = path.resolve()
        resolved[f"{name}_atomic_temporary"] = _atomic_temporary(path).resolve()
    aliases: dict[Path, list[str]] = {}
    for name, path in resolved.items():
        aliases.setdefault(path, []).append(name)
    collisions = [names for names in aliases.values() if len(names) > 1]
    if collisions:
        detail = "; ".join(", ".join(names) for names in collisions)
        raise ValueError(
            "temporal ladder artifacts and atomic temporary paths must be distinct: " + detail
        )
    if report.exists():
        raise FileExistsError(f"temporal ladder refuses to overwrite report: {report}")
    if phase == "development" and checkpoint.exists():
        raise FileExistsError(
            f"temporal ladder refuses to overwrite development checkpoint: {checkpoint}"
        )
    if phase == "qualification" and not checkpoint.is_file():
        raise FileNotFoundError(f"reviewed development checkpoint does not exist: {checkpoint}")
    if phase == "qualification":
        assert development_report is not None
        if not development_report.is_file():
            raise FileNotFoundError(
                f"reviewed development report does not exist: {development_report}"
            )
        if ledger_path.exists():
            raise FileExistsError(
                "temporal protected qualification is one-shot and its durable ledger "
                f"already exists: {ledger_path}"
            )


def _new_model(config: OrpheusConfig) -> TemporalFreeMotionEstimator:
    return TemporalFreeMotionEstimator(
        image_size=config.simulator.image_size,
        world_radius_m=config.simulator.radius_range[0],
        gravity=config.simulator.gravity,
        drag=config.simulator.drag_range[0],
    )


def _write_development_checkpoint(
    path: Path,
    *,
    model: TemporalFreeMotionEstimator,
    config: OrpheusConfig,
    report: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> Path:
    return save_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        step=DEVELOPMENT_UPDATES,
        metrics={
            "artifact_kind": "temporal_free_motion_development_weights_only",
            "exact_resume": False,
            "protected_data_materialized": False,
            "development_report_passed": report.get("passed") is True,
            "development_review_ready": report.get("review_ready") is True,
            "protocol": report["protocol"],
            "training": report["training"],
            "development_audit": report["development_audit"],
            "model": report["model"],
            "resource": report["resource"],
        },
        device="cpu",
        source_provenance=source_provenance,
    )


def _develop(
    config: OrpheusConfig,
    *,
    report_path: Path,
    checkpoint_path: Path,
    source_provenance: Mapping[str, Any],
) -> int:
    clean_source = _require_clean_source_provenance(
        source_provenance,
        label="temporal development",
    )
    try:
        model, report = run_development(config)
    except TemporalQualificationError as error:
        failed = dict(error.report)
        failed["source_provenance"] = clean_source
        _write_report(report_path, failed)
        print(f"FAILED: {error}")
        print(f"report: {report_path.resolve()}")
        return 1

    source_after = _require_clean_source_provenance(
        capture_git_metadata(_REPOSITORY_ROOT),
        label="temporal development completion",
    )
    if source_after != clean_source:
        raise RuntimeError("source provenance changed during temporal development")
    report["source_provenance"] = clean_source
    report["checkpoint"] = str(checkpoint_path.resolve())
    report["checkpoint_kind"] = "project_compatible_weights_only"
    _write_development_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        report=report,
        source_provenance=clean_source,
    )
    report["checkpoint_sha256"] = _sha256_file(checkpoint_path)

    reloaded = _new_model(config)
    payload = load_model_weights(
        checkpoint_path,
        model=reloaded,
        expected_config=config,
    )
    if payload["weight_load_missing_keys"]:
        raise RuntimeError("development checkpoint round trip has missing weights")
    reloaded_hash = report["training"]["trained_model_state_sha256"]
    if _sha256_state(reloaded) != reloaded_hash:
        raise RuntimeError("development checkpoint round trip changed model state")
    report["checkpoint_roundtrip"] = {
        "passed": True,
        "step": int(payload["step"]),
        "model_state_sha256": reloaded_hash,
    }
    _write_report(report_path, report)
    print("PASSED: frozen development review checkpoint; protected data remains unopened")
    print(json.dumps(report["development_audit"], allow_nan=False, indent=2, sort_keys=True))
    print(f"report: {report_path.resolve()}")
    print(f"checkpoint: {checkpoint_path.resolve()}")
    return 0


def _sha256_state(model: TemporalFreeMotionEstimator) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _qualify(
    config: OrpheusConfig,
    *,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    reviewed_checkpoint_sha256: str,
    reviewed_report_sha256: str,
    source_provenance: Mapping[str, Any],
    ledger_path: Path = _QUALIFICATION_LEDGER_PATH,
) -> int:
    clean_source = _require_clean_source_provenance(
        source_provenance,
        label="temporal qualification",
    )
    expected_checkpoint_sha256 = _validated_sha256(
        reviewed_checkpoint_sha256,
        name="reviewed checkpoint SHA-256",
    )
    expected_report_sha256 = _validated_sha256(
        reviewed_report_sha256,
        name="reviewed report SHA-256",
    )
    checkpoint_bytes, checkpoint_sha256 = _read_bytes_once(checkpoint_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint bytes do not match the independently reviewed SHA-256")
    development_report_bytes, development_report_sha256 = _read_bytes_once(development_report_path)
    if development_report_sha256 != expected_report_sha256:
        raise ValueError("development report does not match the independently reviewed SHA-256")
    try:
        development_report = json.loads(development_report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("reviewed development report is not valid JSON") from error
    if not isinstance(development_report, Mapping):
        raise ValueError("reviewed development report must contain a JSON object")

    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("reviewed checkpoint payload must be a mapping")
    required = {"model_state", "step", "config", "metrics", "git"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"reviewed checkpoint is missing fields: {sorted(missing)}")
    if payload["config"] != config.to_dict():
        raise ValueError("reviewed checkpoint does not contain the exact frozen resolved config")
    checkpoint_source = _require_clean_source_provenance(
        payload["git"],
        label="reviewed checkpoint",
    )
    if checkpoint_source != clean_source:
        raise ValueError("current source does not exactly match reviewed checkpoint source")
    model = _new_model(config)
    _assert_finite_model_state(payload["model_state"])
    model.load_state_dict(payload["model_state"], strict=True)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("reviewed checkpoint metrics must be a mapping")
    if metrics.get("artifact_kind") != "temporal_free_motion_development_weights_only":
        raise ValueError("qualification requires a temporal development weights-only checkpoint")
    if metrics.get("protected_data_materialized") is not False:
        raise ValueError("development checkpoint must state that protected data was unopened")
    if metrics.get("development_report_passed") is not True:
        raise ValueError("checkpoint must bind a passed development report")
    if metrics.get("development_review_ready") is not True:
        raise ValueError("checkpoint must bind a review-ready development audit")
    if int(payload["step"]) != DEVELOPMENT_UPDATES:
        raise ValueError("development checkpoint has the wrong completed update count")
    expected_protocol = temporal_protocol()
    if metrics.get("protocol") != expected_protocol:
        raise ValueError("checkpoint protocol does not exactly match current frozen protocol")
    if metrics.get("development_audit", {}).get("passed") is not True:
        raise ValueError("checkpoint development audit did not pass")
    if development_report.get("artifact_kind") != "temporal_free_motion_development_review":
        raise ValueError("reviewed development report has the wrong artifact kind")
    if development_report.get("passed") is not True:
        raise ValueError("reviewed development report did not pass")
    if development_report.get("review_ready") is not True:
        raise ValueError("reviewed development report is not review-ready")
    if development_report.get("protected_data_materialized") is not False:
        raise ValueError("reviewed development report must state protected data was unopened")
    if development_report.get("protocol") != expected_protocol:
        raise ValueError("reviewed development report protocol does not exactly match")
    if development_report.get("source_provenance") != clean_source:
        raise ValueError("reviewed development report source does not exactly match")
    if development_report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reviewed development report binds a different checkpoint")
    if development_report.get("checkpoint_roundtrip", {}).get("passed") is not True:
        raise ValueError("reviewed development checkpoint round trip did not pass")
    for field in ("training", "development_audit", "model", "resource"):
        if development_report.get(field) != metrics.get(field):
            raise ValueError(f"checkpoint and development report disagree on {field}")
    before_hash = _sha256_state(model)
    if metrics.get("training", {}).get("trained_model_state_sha256") != before_hash:
        raise ValueError("loaded model does not match the reviewed trained-state hash")
    if development_report.get("checkpoint_roundtrip", {}).get("model_state_sha256") != before_hash:
        raise ValueError("development report round-trip hash does not match loaded model")

    ledger = _QualificationLedger(
        ledger_path,
        {
            "artifact_kind": "temporal_free_motion_protected_access_ledger",
            "architecture_attempt": expected_protocol["architecture_attempt"],
            "maximum_architecture_attempts": expected_protocol["maximum_architecture_attempts"],
            "checkpoint_sha256": checkpoint_sha256,
            "development_report_sha256": development_report_sha256,
            "protocol_sha256": _canonical_sha256(expected_protocol),
            "resolved_config_sha256": expected_protocol["resolved_config_sha256"],
            "source_provenance": clean_source,
            "qualification_report": str(report_path.resolve()),
        },
    )

    def record_access(split: str) -> None:
        current_source = _require_clean_source_provenance(
            capture_git_metadata(_REPOSITORY_ROOT),
            label=f"pre-{split} qualification",
        )
        if current_source != clean_source:
            raise RuntimeError(f"source changed before protected {split} access")
        ledger.record_access(split)

    failed_error: Exception | None = None
    try:
        report = run_protected_qualification(model, config, access_recorder=record_access)
    except TemporalQualificationError as error:
        report = dict(error.report)
        failed_error = error
    except Exception as error:  # pragma: no cover - defensive outer receipt
        report = {
            "artifact_kind": "temporal_free_motion_protected_qualification",
            "protocol": expected_protocol,
            "passed": False,
            "stopped_after": "unexpected_exception",
            "unexpected_error": {"type": type(error).__name__, "message": str(error)},
        }
        failed_error = error
    after_hash = _sha256_state(model)
    if before_hash != after_hash:
        report["passed"] = False
        report["stopped_after"] = "model_mutation_detected"
        report.setdefault("failures", []).append("protected qualification mutated reviewed model")
        failed_error = RuntimeError("protected qualification mutated the reviewed model")
    access_started = dict(ledger.record["access_started"])
    if report.get("passed") is True and not all(access_started.values()):
        report["passed"] = False
        report["stopped_after"] = "incomplete_protected_access"
        report.setdefault("failures", []).append(
            "successful qualification did not record all three protected accesses"
        )
        failed_error = RuntimeError("protected qualification access receipt is incomplete")
    source_after = _require_clean_source_provenance(
        capture_git_metadata(_REPOSITORY_ROOT),
        label="temporal qualification completion",
    )
    if source_after != clean_source:
        report["passed"] = False
        report["stopped_after"] = "source_change_detected"
        report.setdefault("failures", []).append("source changed during qualification")
        failed_error = RuntimeError("source changed during temporal qualification")
    report["access_started"] = access_started
    report["protected_data_materialized"] = any(access_started.values())
    report["source_provenance"] = clean_source
    report["input_checkpoint"] = str(checkpoint_path.resolve())
    report["input_checkpoint_sha256"] = checkpoint_sha256
    report["development_report"] = str(development_report_path.resolve())
    report["development_report_sha256"] = development_report_sha256
    report["model_state_sha256_before"] = before_hash
    report["model_state_sha256_after"] = after_hash
    ledger.finish(report)
    _write_report(report_path, report)
    if failed_error is not None or report.get("passed") is not True:
        print(f"FAILED: {failed_error or 'protected qualification gate failed'}")
        print(f"report: {report_path.resolve()}")
        print(f"durable access ledger: {ledger_path.resolve()}")
        return 1
    print("PASSED: temporal selector, confirmation, and one-shot final")
    print(json.dumps(report["final_test"], allow_nan=False, indent=2, sort_keys=True))
    print(f"report: {report_path.resolve()}")
    print(f"durable access ledger: {ledger_path.resolve()}")
    return 0


def main() -> int:
    args = _arguments()
    _validate_paths(
        args.phase,
        args.report,
        args.checkpoint,
        development_report=args.development_report,
    )
    # Set the frozen CPU execution policy before any tensor work starts. Keeping
    # this at the process boundary avoids mutating global thread state in the
    # reusable training and qualification functions.
    torch.set_num_threads(1)
    config = load_config(args.config)
    source_provenance = capture_git_metadata(_REPOSITORY_ROOT)
    _require_clean_source_provenance(
        source_provenance,
        label=f"temporal {args.phase}",
    )
    if args.phase == "development":
        return _develop(
            config,
            report_path=args.report,
            checkpoint_path=args.checkpoint,
            source_provenance=source_provenance,
        )
    return _qualify(
        config,
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        development_report_path=args.development_report,
        reviewed_checkpoint_sha256=args.reviewed_checkpoint_sha256,
        reviewed_report_sha256=args.reviewed_report_sha256,
        source_provenance=source_provenance,
    )


if __name__ == "__main__":
    raise SystemExit(main())
