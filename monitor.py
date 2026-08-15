#!/usr/bin/env python3
"""Low-noise live monitor for Project Orpheus run artifacts.

The monitor is deliberately read-only: it never imports or executes the model,
loads a checkpoint, or contacts an external service.  It watches the durable
JSON/JSONL files already emitted by training and evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TIMESTAMP_PREFIX = re.compile(r"^(\d{8}-\d{6})(?:-|$)")
RUN_ARTIFACTS = frozenset(
    {
        "training_state.json",
        "training_failure.json",
        "training_progress.json",
        "metrics.jsonl",
        "run_metadata.json",
        "train_summary.json",
        "evaluation_progress.json",
        "evaluation.json",
        "report.json",
        "convergence_supervisor_state.json",
        "convergence_report.json",
    }
)
SKIPPED_DISCOVERY_DIRECTORIES = frozenset(
    {
        ".git",
        "__pycache__",
        "checkpoints",
        "frames",
        "videos",
    }
)
FAILED_SUPERVISOR_STATES = frozenset(
    {"extension_failed", "initial_trainer_failed", "failed", "error"}
)
ACTIVE_SUPERVISOR_STATES = frozenset(
    {"extension_starting", "extension_running", "segment_running", "running"}
)
TERMINAL_STATUSES = frozenset({"COMPLETED", "CONVERGED", "FAILED", "LIMIT HIT"})
TERMINAL_EVALUATION_STAGES = frozenset({"completed", "failed", "interrupted"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {}, f"{path.name} is unreadable: {type(error).__name__}"
    if not isinstance(value, dict):
        return {}, f"{path.name} is not a JSON object"
    return value, None


def _read_jsonl_tail(
    path: Path,
    *,
    limit: int = 64,
    maximum_bytes: int = 32 * 1024 * 1024,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read complete recent JSONL rows without loading a long run in full."""

    if not path.is_file():
        return [], []
    try:
        size = path.stat().st_size
        start = max(0, size - maximum_bytes)
        with path.open("rb") as handle:
            handle.seek(start)
            payload = handle.read()
    except OSError as error:
        return [], [f"metrics.jsonl is unreadable: {type(error).__name__}"]

    lines = payload.splitlines()
    if start > 0 and lines:
        # The first row is generally a partial line when reading a byte tail.
        lines = lines[1:]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # An append may be observed between write and newline. Ignore only
            # the trailing fragment; malformed durable middle rows are visible.
            if index == len(lines) - 1 and not payload.endswith(b"\n"):
                continue
            errors.append("metrics.jsonl contains a malformed complete row")
            continue
        if not isinstance(value, dict):
            errors.append("metrics.jsonl contains a non-object row")
            continue
        rows.append(value)
    if start > 0 and len(rows) < limit:
        errors.append(
            "metrics.jsonl recent rows exceed the 32 MiB monitor tail; trend is truncated"
        )
    return rows[-limit:], list(dict.fromkeys(errors))


def _timestamp_key(path: Path) -> str:
    """Return the nearest sortable timestamp prefix, including synthetic hours."""

    for part in reversed(path.parts):
        match = TIMESTAMP_PREFIX.match(part)
        if match is not None:
            return match.group(1)
    return ""


def discover_run_directories(runs_root: Path) -> list[Path]:
    runs_root = runs_root.expanduser().resolve()
    if not runs_root.is_dir():
        return []
    candidates: list[Path] = []
    for directory, child_directories, filenames in os.walk(runs_root):
        child_directories[:] = [
            name
            for name in child_directories
            if name not in SKIPPED_DISCOVERY_DIRECTORIES and not name.startswith(".")
        ]
        if RUN_ARTIFACTS.intersection(filenames):
            candidates.append(Path(directory).resolve())
    return candidates


def _artifact_mtime_ns(run_directory: Path) -> int:
    mtimes: list[int] = []
    for name in RUN_ARTIFACTS:
        path = run_directory / name
        with contextlib.suppress(OSError):
            mtimes.append(path.stat().st_mtime_ns)
    checkpoint_directory = run_directory / "checkpoints"
    if checkpoint_directory.is_dir():
        for path in checkpoint_directory.glob("*.pt"):
            with contextlib.suppress(OSError):
                mtimes.append(path.stat().st_mtime_ns)
    return max(mtimes, default=0)


def _training_lock_held(path: Path) -> bool:
    """Check the trainer's advisory lock without mutating its durable file."""

    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def _pid_alive(pid: int | None) -> bool | None:
    if pid is None or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_command_matches_evaluator(pid: int | None) -> bool | None:
    if pid is None or pid <= 0:
        return None
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return False
    command = result.stdout.lower()
    return any(
        marker in command
        for marker in (
            "evaluate.py",
            "evaluate_hypothesis_pool.py",
            "evaluate_modular_candidate.py",
            "replay_promotion_mps.py",
        )
    )


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _updated_time(payload: Mapping[str, Any]) -> datetime | None:
    return _parse_utc(payload.get("updated_utc"))


def _read_target_steps(config_path: Path) -> int | None:
    """Read only ``training.steps`` from the canonical resolved YAML."""

    if not config_path.is_file():
        return None
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_training = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            in_training = line.strip() == "training:"
            continue
        if in_training:
            match = re.match(r"^\s+steps:\s*([0-9]+)\s*(?:#.*)?$", line)
            if match is not None:
                return int(match.group(1))
    return None


def _latest_checkpoint(run_directory: Path) -> dict[str, Any] | None:
    checkpoint_directory = run_directory / "checkpoints"
    if not checkpoint_directory.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_directory.glob("*.pt"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size <= 0:
            continue
        candidates.append((stat.st_mtime_ns, path.resolve()))
    if not candidates:
        return None
    mtime_ns, path = max(candidates)
    return {
        "path": str(path),
        "updated_utc": datetime.fromtimestamp(mtime_ns / 1e9, timezone.utc).isoformat(),
    }


def _horizon_metrics(
    payload: Mapping[str, Any],
    *,
    pattern: re.Pattern[str],
) -> dict[str, float]:
    values: list[tuple[float, str, float]] = []
    for key, raw_value in payload.items():
        match = pattern.fullmatch(str(key))
        value = _number(raw_value)
        if match is None or value is None:
            continue
        label = match.group("horizon")
        values.append((float(label), label, value))
    return {label: value for _, label, value in sorted(values)}


def _artifact_revision(run_directory: Path, *, process_state: str) -> str:
    entries: list[str] = [process_state]
    for name in sorted(RUN_ARTIFACTS | {".training.lock"}):
        path = run_directory / name
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(f"{name}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}")
    checkpoint = _latest_checkpoint(run_directory)
    if checkpoint is not None:
        entries.append(f"checkpoint:{checkpoint['path']}:{checkpoint['updated_utc']}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _metric_summary(rows: Sequence[Mapping[str, Any]], target_steps: int | None) -> dict[str, Any]:
    train_rows = [row for row in rows if row.get("split") == "train"]
    skipped_rows = [row for row in rows if row.get("split") == "train_skipped_no_gradient"]
    step_rows = [row for row in rows if _integer(row.get("step")) is not None]
    latest_step = max((_integer(row.get("step")) or 0 for row in step_rows), default=None)
    latest_train = train_rows[-1] if train_rows else None

    losses = [
        value for row in train_rows[-8:] if (value := _number(row.get("loss_total"))) is not None
    ]
    rolling_median = statistics.median(losses[-4:]) if losses else None
    loss_delta_percent: float | None = None
    if len(losses) >= 8:
        previous = statistics.median(losses[-8:-4])
        current = statistics.median(losses[-4:])
        if abs(previous) > 1e-12:
            loss_delta_percent = (current / previous - 1.0) * 100.0

    eta_seconds: float | None = None
    timed_rows = [
        row
        for row in train_rows
        if _integer(row.get("step")) is not None and _number(row.get("elapsed_seconds")) is not None
    ]
    if target_steps is not None and latest_step is not None and len(timed_rows) >= 2:
        first = timed_rows[0]
        last = timed_rows[-1]
        step_delta = (_integer(last.get("step")) or 0) - (_integer(first.get("step")) or 0)
        elapsed_delta = (_number(last.get("elapsed_seconds")) or 0.0) - (
            _number(first.get("elapsed_seconds")) or 0.0
        )
        if step_delta > 0 and elapsed_delta > 0 and latest_step < target_steps:
            eta_seconds = elapsed_delta / step_delta * (target_steps - latest_step)

    validation_rows = [
        row
        for row in rows
        if str(row.get("split") or "").startswith("validation")
        and any(
            key in row
            for key in (
                "validation_rollout_selection_score",
                "selection_metric_supported",
                "selection_accepted",
                "validation_measurement_selection_score",
                "measurement_selection_usable",
                "measurement_selection_accepted",
            )
        )
    ]
    validation: dict[str, Any] | None = None
    if validation_rows:
        row = validation_rows[-1]
        if any(
            key in row
            for key in (
                "validation_measurement_selection_score",
                "measurement_selection_usable",
                "measurement_selection_accepted",
            )
        ):
            validation = {
                "kind": "measurement",
                "step": _integer(row.get("step")),
                "split": row.get("split"),
                "selection_score": _number(row.get("validation_measurement_selection_score")),
                "incumbent_score": _number(row.get("best_measurement_selection_score")),
                "incumbent_step": _integer(row.get("best_measurement_checkpoint_step")),
                "accepted": _number(row.get("measurement_selection_accepted")),
                "supported": _number(row.get("measurement_selection_usable")),
                "rejection_count": _integer(
                    row.get("measurement_selection_rejection_reason_count")
                ),
                "world_position_mae_m": _number(
                    row.get("validation_runtime_birth_world_position_mae_m")
                ),
                "runtime_birth_f1": _number(row.get("validation_runtime_birth_f1_at_0_5m")),
                "fast_roi_f1": _number(row.get("validation_fast_roi_f1_at_0_5m")),
                "horizon_rmse_m": {},
            }
        else:
            validation = {
                "kind": "rollout",
                "step": _integer(row.get("step")),
                "split": row.get("split"),
                "selection_score": _number(row.get("validation_rollout_selection_score")),
                "incumbent_score": _number(row.get("best_rollout_selection_score")),
                "incumbent_step": _integer(row.get("best_rollout_checkpoint_step")),
                "accepted": _number(row.get("selection_accepted")),
                "supported": _number(row.get("selection_metric_supported")),
                "rejection_count": _integer(row.get("selection_rejection_reason_count")),
                "position_rmse_m": _number(row.get("validation_position_rmse_m")),
                "velocity_rmse_mps": _number(row.get("validation_velocity_rmse_mps")),
                "target_coverage": _number(row.get("validation_target_coverage")),
                "collision_f1": _number(row.get("validation_collision_f1")),
                "horizon_rmse_m": _horizon_metrics(
                    row,
                    pattern=re.compile(
                        r"validation_position_rmse@(?P<horizon>[0-9]+(?:\.[0-9]+)?)s"
                    ),
                ),
            }
        candidate_score = _number(validation.get("selection_score"))
        incumbent_score = _number(validation.get("incumbent_score"))
        validation["score_delta"] = (
            candidate_score - incumbent_score
            if candidate_score is not None and incumbent_score is not None
            else None
        )

    return {
        "step": latest_step,
        "target_steps": target_steps,
        "phase": latest_train.get("phase") if latest_train is not None else None,
        "latest_loss": (
            _number(latest_train.get("loss_total")) if latest_train is not None else None
        ),
        "rolling_loss_median": rolling_median,
        "rolling_loss_delta_percent": loss_delta_percent,
        "learning_rate": (
            _number(latest_train.get("learning_rate")) if latest_train is not None else None
        ),
        "elapsed_seconds": (
            _number(latest_train.get("elapsed_seconds")) if latest_train is not None else None
        ),
        "eta_seconds": eta_seconds,
        "recent_train_records": len(train_rows),
        "recent_skipped_no_gradient_records": len(skipped_rows),
        "validation": validation,
    }


def _evaluation_result(run_directory: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    for name in ("evaluation.json", "report.json"):
        path = run_directory / name
        payload, error = _read_json(path)
        if error is not None:
            errors.append(error)
        if not payload:
            continue
        metadata = payload.get("metadata")
        metrics = payload.get("metrics")
        if not isinstance(metadata, Mapping) or not isinstance(metrics, Mapping):
            return (
                {
                    "path": str(path.resolve()),
                    "generic_report": True,
                    "horizon_rmse_m": {},
                },
                errors,
            )
        horizon_rmse = _horizon_metrics(
            metrics,
            pattern=re.compile(r"model@(?P<horizon>[0-9]+(?:\.[0-9]+)?)s_position_rmse_m"),
        )
        return (
            {
                "path": str(path.resolve()),
                "device": metadata.get("device"),
                "precision": metadata.get("precision"),
                "split": metadata.get("split"),
                "checkpoint_step": _integer(metadata.get("checkpoint_step")),
                "episodes": _integer(metadata.get("episodes")),
                "evaluated_episodes": _integer(metrics.get("evaluated_episodes")),
                "current_position_rmse_m": _number(
                    metrics.get("posterior_current_position_rmse_m")
                ),
                "horizon_rmse_m": horizon_rmse,
                "forecast_gaussian_nll": _number(metrics.get("forecast_gaussian_nll")),
                "nonfinite_output_count": _integer(metrics.get("nonfinite_output_count")),
                "collision_f1": _number(metrics.get("collision_f1")),
                "identity_switch_rate": _number(metrics.get("distance_gated_identity_switch_rate")),
            },
            errors,
        )
    return None, errors


def _is_nonfinite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and not math.isfinite(float(value))
    )


def _hard_metric_warnings(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    keys = (
        "loss_total",
        "gradient_norm_pre_clip",
        "gradient_norm_applied",
        "validation_rollout_selection_score",
        "validation_position_rmse_m",
        "validation_velocity_rmse_mps",
    )
    for row in rows[-16:]:
        for key in keys:
            if key in row and _is_nonfinite_number(row[key]):
                warnings.append(f"non-finite {key} at step {row.get('step', '?')}")
    recent_train = [
        row for row in rows[-16:] if row.get("split") in {"train", "train_skipped_no_gradient"}
    ]
    if recent_train:
        skipped = sum(row.get("split") == "train_skipped_no_gradient" for row in recent_train)
        unsupported = sum(
            _number(row.get("causal_training_support_present")) == 0.0 for row in recent_train
        )
        unapplied = sum(_number(row.get("optimizer_update_applied")) == 0.0 for row in recent_train)
        if skipped:
            warnings.append(
                f"{skipped}/{len(recent_train)} recent train records skipped for no gradient"
            )
        if unsupported:
            warnings.append(
                f"{unsupported}/{len(recent_train)} recent train records lacked causal support"
            )
        if unapplied:
            warnings.append(
                f"{unapplied}/{len(recent_train)} recent train records applied no update"
            )
    collapse_rows = [
        row
        for row in rows[-16:]
        if str(row.get("split") or "").startswith("training_control_support_collapse")
        or _number(row.get("support_collapse_rollback_applied")) == 1.0
    ]
    if collapse_rows:
        latest = collapse_rows[-1]
        failure_count = _integer(latest.get("support_collapse_failure_count"))
        count_text = f" with {failure_count} support failures" if failure_count is not None else ""
        warnings.append(f"training support collapse triggered incumbent rollback{count_text}")
    return list(dict.fromkeys(warnings))


def _progress_fraction(completed: Any, total: Any) -> float | None:
    completed_number = _number(completed)
    total_number = _number(total)
    if completed_number is None or total_number is None or total_number <= 0:
        return None
    return min(max(completed_number / total_number, 0.0), 1.0)


def build_snapshot(
    run_directory: Path,
    *,
    now: datetime | None = None,
    stale_after_seconds: float = 1800.0,
) -> dict[str, Any]:
    run_directory = run_directory.expanduser().resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_directory}")
    now = now or _utc_now()
    errors: list[str] = []

    payloads: dict[str, dict[str, Any]] = {}
    for name in (
        "training_state.json",
        "training_failure.json",
        "training_progress.json",
        "evaluation_progress.json",
        "run_metadata.json",
        "train_summary.json",
        "convergence_supervisor_state.json",
        "convergence_report.json",
    ):
        payload, error = _read_json(run_directory / name)
        payloads[name] = payload
        if error is not None:
            errors.append(error)

    rows, metrics_errors = _read_jsonl_tail(run_directory / "metrics.jsonl")
    errors.extend(metrics_errors)
    evaluation_result, evaluation_errors = _evaluation_result(run_directory)
    errors.extend(evaluation_errors)

    state = payloads["training_state.json"]
    failure = payloads["training_failure.json"]
    training_progress = payloads["training_progress.json"]
    evaluation_progress = payloads["evaluation_progress.json"]
    metadata = payloads["run_metadata.json"]
    summary = payloads["train_summary.json"]
    supervisor = payloads["convergence_supervisor_state.json"]
    convergence_report = payloads["convergence_report.json"]

    target_steps = _integer(state.get("target_steps"))
    if target_steps is None:
        target_steps = _read_target_steps(run_directory / "config.resolved.yaml")
    if target_steps is None:
        target_steps = _integer(summary.get("completed_steps"))
    training = _metric_summary(rows, target_steps)
    if training["step"] is None:
        training["step"] = _integer(state.get("completed_steps"))

    lock_active = _training_lock_held(run_directory / ".training.lock")
    lock_pid: int | None = None
    lock_path = run_directory / ".training.lock"
    if lock_path.is_file():
        with contextlib.suppress(OSError, ValueError):
            lock_pid = int(lock_path.read_text(encoding="utf-8").strip())
    progress_pid = _integer(evaluation_progress.get("pid"))
    if progress_pid is None:
        progress_pid = _integer(training_progress.get("pid"))
    if progress_pid is None:
        progress_pid = _integer(supervisor.get("child_pid"))
    if progress_pid is None:
        progress_pid = _integer(supervisor.get("trainer_pid"))
    pid = lock_pid if lock_active and lock_pid is not None else progress_pid
    pid_alive = _pid_alive(pid)
    evaluator_identity = (
        _pid_command_matches_evaluator(pid)
        if evaluation_progress.get("stage") not in ({None} | TERMINAL_EVALUATION_STAGES)
        and pid_alive
        else None
    )

    monitored_paths: list[Path] = []
    for name in RUN_ARTIFACTS:
        path = run_directory / name
        if path.is_file():
            monitored_paths.append(path)
    checkpoint = _latest_checkpoint(run_directory)
    if checkpoint is not None:
        monitored_paths.append(Path(checkpoint["path"]))
    activity_path: Path | None = None
    activity_time: datetime | None = None
    for path in monitored_paths:
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        except OSError:
            continue
        if activity_time is None or modified > activity_time:
            activity_time = modified
            activity_path = path
    activity_age_seconds = (
        max(0.0, (now - activity_time).total_seconds()) if activity_time is not None else None
    )
    stale = activity_age_seconds is not None and activity_age_seconds > stale_after_seconds

    state_name = str(state.get("state") or "").lower()
    evaluation_stage = str(evaluation_progress.get("stage") or "").lower()
    supervisor_status = str(supervisor.get("status") or "").lower()
    convergence_status = str(
        convergence_report.get("status") or supervisor.get("status") or ""
    ).lower()
    supervisor_failed = supervisor_status in FAILED_SUPERVISOR_STATES
    state_updated = _updated_time(state)
    supervisor_updated = _updated_time(supervisor)
    if state_name in {"running", "completed"} and supervisor_failed:
        supervisor_failed = bool(
            supervisor_updated is not None
            and (state_updated is None or supervisor_updated > state_updated)
        )

    kind = "evaluation" if evaluation_progress or evaluation_result is not None else "training"
    status_detail = ""
    if failure:
        status = "FAILED"
        status_detail = str(failure.get("exception_type") or failure.get("message") or "training")
    elif state_name == "failed":
        status = "FAILED"
        status_detail = str(state.get("exception_type") or state.get("message") or "training")
    elif supervisor_failed:
        status = "FAILED"
        status_detail = str(supervisor.get("error_type") or supervisor_status)
    elif lock_active:
        if training_progress.get("state") == "validation_running":
            status = "VALIDATING"
            status_detail = str(training_progress.get("split") or "validation")
        else:
            status = "TRAINING"
            status_detail = str(training.get("phase") or state_name or "running")
    elif evaluation_stage in {"failed", "interrupted"}:
        status = "FAILED"
        status_detail = str(
            evaluation_progress.get("exception_type") or f"evaluation {evaluation_stage}"
        )
    elif evaluation_stage == "completed":
        status = "COMPLETED"
        status_detail = "evaluation"
    elif evaluation_stage and evaluation_stage not in TERMINAL_EVALUATION_STAGES:
        if pid_alive and evaluator_identity is not False:
            status = "EVALUATING"
            status_detail = evaluation_stage
        else:
            status = "STALE"
            status_detail = "evaluation process is not verified live"
    elif evaluation_result is not None and not evaluation_progress:
        status = "COMPLETED"
        status_detail = "evaluation report"
    elif convergence_status == "plateau":
        status = "CONVERGED"
        status_detail = str(
            convergence_report.get("reason") or supervisor.get("reason") or "plateau"
        )
    elif convergence_status == "limit_hit":
        status = "LIMIT HIT"
        status_detail = str(
            convergence_report.get("reason") or supervisor.get("reason") or "not converged"
        )
    elif state_name == "completed":
        status = "COMPLETED"
        status_detail = "training"
    elif state_name in {"starting", "running"}:
        status = "STALE"
        status_detail = "trainer lock is not held"
    elif supervisor_status in ACTIVE_SUPERVISOR_STATES and pid_alive:
        status = "TRAINING"
        status_detail = supervisor_status
    elif rows or summary:
        status = "STALE" if stale else "UNKNOWN"
        status_detail = "no authoritative terminal state"
    else:
        status = "STARTING" if not stale else "STALE"
        status_detail = "waiting for run artifacts"

    warnings = _hard_metric_warnings(rows)
    warnings.extend(errors)
    if stale and status not in TERMINAL_STATUSES:
        warnings.append(f"no artifact update for more than {_format_duration(stale_after_seconds)}")
    if evaluator_identity is False and status not in TERMINAL_STATUSES:
        warnings.append("recorded evaluation PID belongs to another command or has exited")
    if (
        evaluation_stage
        and evaluation_stage not in TERMINAL_EVALUATION_STAGES
        and pid_alive is not True
    ):
        warnings.append("recorded evaluation process is not alive")
    if state_name in {"starting", "running"} and not lock_active:
        warnings.append("authoritative running state has no held trainer lock")
    if failure:
        message = failure.get("message")
        if message:
            warnings.append(str(message))
    elif state_name == "failed" and state.get("message"):
        warnings.append(str(state["message"]))
    elif supervisor_failed and supervisor.get("error"):
        warnings.append(str(supervisor["error"]))
    if evaluation_stage in {"failed", "interrupted"} and evaluation_progress.get("message"):
        warnings.append(str(evaluation_progress["message"]))
    if status == "LIMIT HIT":
        warnings.append("convergence campaign reached its configured limit without plateau")

    validation = training.get("validation")
    if isinstance(validation, Mapping):
        if validation.get("supported") == 0.0:
            warnings.append("latest validation lacks required selection support")
        elif validation.get("accepted") == 0.0:
            count = validation.get("rejection_count")
            suffix = f" ({count} guardrails)" if count is not None else ""
            warnings.append(f"latest validation candidate was safely rejected{suffix}")
    delta = training.get("rolling_loss_delta_percent")
    if isinstance(delta, (int, float)) and delta > 100.0:
        warnings.append(f"recent rolling train loss rose {delta:.1f}%")
    if evaluation_result is not None:
        nonfinite = evaluation_result.get("nonfinite_output_count")
        if isinstance(nonfinite, int) and nonfinite > 0:
            warnings.append(f"evaluation recorded {nonfinite} non-finite outputs")

    best_checkpoint = state.get("best_checkpoint") or summary.get("best_checkpoint")
    device = {
        "device": metadata.get("device"),
        "measurement_device": metadata.get("measurement_device"),
        "closed_loop_device": metadata.get("closed_loop_device"),
        "precision": metadata.get("precision"),
        "torch_version": metadata.get("torch_version"),
    }
    if evaluation_result is not None:
        device["device"] = evaluation_result.get("device") or device["device"]
        device["precision"] = evaluation_result.get("precision") or device["precision"]
    if evaluation_progress:
        device["device"] = evaluation_progress.get("device") or device["device"]
        device["precision"] = evaluation_progress.get("precision") or device["precision"]

    process_state = (
        "terminal"
        if status in TERMINAL_STATUSES
        else f"lock={lock_active};pid={pid};alive={pid_alive};eval={evaluator_identity}"
    )
    return {
        "run_directory": str(run_directory),
        "kind": kind,
        "status": status,
        "status_detail": status_detail,
        "terminal": status in TERMINAL_STATUSES,
        "checked_utc": now.isoformat(),
        "activity": {
            "path": str(activity_path) if activity_path is not None else None,
            "updated_utc": activity_time.isoformat() if activity_time is not None else None,
            "age_seconds": activity_age_seconds,
            "stale_after_seconds": stale_after_seconds,
        },
        "process": {
            "pid": pid,
            "alive": pid_alive,
            "training_lock_held": lock_active,
            "evaluator_command_matches": evaluator_identity,
        },
        "device": device,
        "training": training,
        "validation_progress": training_progress or None,
        "evaluation_progress": evaluation_progress or None,
        "evaluation_result": evaluation_result,
        "checkpoint": checkpoint,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "supervisor": supervisor or None,
        "convergence": convergence_report or supervisor or None,
        "warnings": list(dict.fromkeys(warnings)),
        "revision": _artifact_revision(run_directory, process_state=process_state),
    }


def _candidate_is_active(run_directory: Path) -> bool:
    if _training_lock_held(run_directory / ".training.lock"):
        return True
    progress, _ = _read_json(run_directory / "evaluation_progress.json")
    stage = progress.get("stage")
    if stage and stage not in TERMINAL_EVALUATION_STAGES:
        pid = _integer(progress.get("pid"))
        return bool(_pid_alive(pid) and _pid_command_matches_evaluator(pid) is not False)
    supervisor, _ = _read_json(run_directory / "convergence_supervisor_state.json")
    if str(supervisor.get("status") or "").lower() in ACTIVE_SUPERVISOR_STATES:
        pid = _integer(supervisor.get("child_pid")) or _integer(supervisor.get("trainer_pid"))
        return bool(_pid_alive(pid))
    return False


def select_run_directory(runs_root: Path) -> Path | None:
    candidates = discover_run_directories(runs_root)
    if not candidates:
        return None
    active = [path for path in candidates if _candidate_is_active(path)]
    pool = active or candidates
    return max(
        pool,
        key=lambda path: (_timestamp_key(path), _artifact_mtime_ns(path), str(path)),
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _format_number(value: Any, *, digits: int = 5) -> str:
    number = _number(value)
    return "?" if number is None else f"{number:.{digits}g}"


def _short_path(value: str | None, *, cwd: Path | None = None) -> str:
    if value is None:
        return "?"
    path = Path(value)
    cwd = cwd or Path.cwd()
    with contextlib.suppress(ValueError):
        return str(path.resolve().relative_to(cwd.resolve()))
    return str(path)


def _bar(fraction: float | None, width: int = 20) -> str:
    if fraction is None:
        return "[" + "?" * width + "]"
    filled = min(width, max(0, round(fraction * width)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render_snapshot(snapshot: Mapping[str, Any]) -> str:
    run_directory = str(snapshot["run_directory"])
    status = str(snapshot["status"])
    detail = snapshot.get("status_detail")
    lines = [
        f"Orpheus monitor  {snapshot['checked_utc']}",
        f"run       {_short_path(run_directory)}",
        f"status    {status}{f' · {detail}' if detail else ''}",
    ]

    process = snapshot.get("process")
    if (
        isinstance(process, Mapping)
        and status not in TERMINAL_STATUSES
        and (process.get("pid") is not None or process.get("training_lock_held"))
    ):
        pieces: list[str] = []
        if process.get("pid") is not None:
            pieces.append(f"pid {process['pid']}")
        if process.get("training_lock_held"):
            pieces.append("trainer lock held")
        elif process.get("alive") is True:
            pieces.append("process alive")
        elif process.get("alive") is False and status not in TERMINAL_STATUSES:
            pieces.append("process not alive")
        lines.append("process   " + " · ".join(pieces))

    activity = snapshot.get("activity")
    if isinstance(activity, Mapping) and activity.get("updated_utc"):
        lines.append(
            "activity  "
            f"{_format_duration(_number(activity.get('age_seconds')))} ago · "
            f"{Path(str(activity.get('path'))).name}"
        )

    device = snapshot.get("device")
    if isinstance(device, Mapping) and any(device.values()):
        pieces = []
        if device.get("measurement_device"):
            pieces.append(f"RGB={device['measurement_device']}")
        if device.get("closed_loop_device"):
            pieces.append(f"closed-loop={device['closed_loop_device']}")
        elif device.get("device"):
            pieces.append(str(device["device"]))
        if device.get("precision"):
            pieces.append(str(device["precision"]))
        if device.get("torch_version"):
            pieces.append(f"torch {device['torch_version']}")
        lines.append("device    " + " · ".join(pieces))

    training = snapshot.get("training")
    if isinstance(training, Mapping) and training.get("step") is not None:
        step = _integer(training.get("step"))
        target = _integer(training.get("target_steps"))
        fraction = _progress_fraction(step, target)
        target_text = str(target) if target is not None else "?"
        percent = f" {fraction * 100:5.1f}%" if fraction is not None else ""
        phase = f" · {training['phase']}" if training.get("phase") else ""
        eta = (
            f" · ETA {_format_duration(_number(training.get('eta_seconds')))}"
            if status in {"TRAINING", "VALIDATING"} and training.get("eta_seconds") is not None
            else ""
        )
        lines.append(f"train     {step}/{target_text} {_bar(fraction)}{percent}{phase}{eta}")
        if training.get("latest_loss") is not None:
            loss_line = (
                f"loss      latest {_format_number(training.get('latest_loss'))}"
                f" · median {_format_number(training.get('rolling_loss_median'))}"
            )
            delta = _number(training.get("rolling_loss_delta_percent"))
            if delta is not None:
                loss_line += f" · {delta:+.1f}% vs prior 4 logs"
            if training.get("learning_rate") is not None:
                loss_line += f" · lr {_format_number(training.get('learning_rate'), digits=3)}"
            lines.append(loss_line)

    validation_progress = snapshot.get("validation_progress")
    if (
        isinstance(validation_progress, Mapping)
        and validation_progress.get("state") == "validation_running"
    ):
        completed = validation_progress.get("completed_episodes")
        total = validation_progress.get("total_episodes")
        lines.append(
            "validate  "
            f"{completed}/{total if total is not None else '?'} episodes "
            f"{_bar(_progress_fraction(completed, total))} · "
            f"{validation_progress.get('validation_kind', '?')} · "
            f"last {validation_progress.get('last_scenario', '?')}"
        )

    validation = training.get("validation") if isinstance(training, Mapping) else None
    if isinstance(validation, Mapping):
        acceptance = ""
        if validation.get("supported") == 0.0:
            acceptance = " · UNSUPPORTED"
        elif validation.get("accepted") == 1.0:
            acceptance = " · ACCEPTED"
        elif validation.get("accepted") == 0.0:
            acceptance = " · rejected"
        kind = str(validation.get("kind") or "rollout")
        if kind == "measurement":
            lines.append(
                "val       "
                f"measurement · step {validation.get('step', '?')} · score "
                f"{_format_number(validation.get('selection_score'))}{acceptance} · "
                f"birth MAE {_format_number(validation.get('world_position_mae_m'))} m · "
                f"birth F1 {_format_number(validation.get('runtime_birth_f1'))}"
            )
        else:
            lines.append(
                "val       "
                f"rollout · step {validation.get('step', '?')} · score "
                f"{_format_number(validation.get('selection_score'))}{acceptance} · "
                f"current RMSE {_format_number(validation.get('position_rmse_m'))} m"
            )
        if validation.get("incumbent_score") is not None:
            incumbent_step = validation.get("incumbent_step")
            step_text = f" @ step {incumbent_step}" if incumbent_step is not None else ""
            delta = _number(validation.get("score_delta"))
            delta_text = f" · candidate Δ {delta:+.6g}" if delta is not None else ""
            lines.append(
                "incumbent "
                f"score {_format_number(validation.get('incumbent_score'))}{step_text}"
                f"{delta_text}"
            )
        horizons = validation.get("horizon_rmse_m")
        if isinstance(horizons, Mapping) and horizons:
            lines.append(
                "horizon   "
                + " · ".join(f"{key}s {_format_number(value)} m" for key, value in horizons.items())
            )

    evaluation_progress = snapshot.get("evaluation_progress")
    if isinstance(evaluation_progress, Mapping) and evaluation_progress.get("stage") != "completed":
        stage = evaluation_progress.get("stage", "?")
        pieces = [str(stage)]
        if evaluation_progress.get("split") is not None:
            pieces.append(str(evaluation_progress["split"]))
        if evaluation_progress.get("batch") is not None:
            pieces.append(
                f"batch {evaluation_progress['batch']}/{evaluation_progress.get('batches', '?')}"
            )
        if evaluation_progress.get("frame") is not None:
            pieces.append(
                f"frame {evaluation_progress['frame']}/{evaluation_progress.get('total_frames', '?')}"
            )
        if evaluation_progress.get("evaluated_episodes") is not None:
            pieces.append(
                f"episodes {evaluation_progress['evaluated_episodes']}/"
                f"{evaluation_progress.get('episodes', '?')}"
            )
        lines.append("evaluate  " + " · ".join(pieces))

    evaluation_result = snapshot.get("evaluation_result")
    if isinstance(evaluation_result, Mapping) and not evaluation_result.get("generic_report"):
        lines.append(
            "result    "
            f"current RMSE {_format_number(evaluation_result.get('current_position_rmse_m'))} m"
            f" · NLL {_format_number(evaluation_result.get('forecast_gaussian_nll'))}"
        )
        horizons = evaluation_result.get("horizon_rmse_m")
        if isinstance(horizons, Mapping) and horizons:
            lines.append(
                "horizon   "
                + " · ".join(f"{key}s {_format_number(value)} m" for key, value in horizons.items())
            )

    convergence = snapshot.get("convergence")
    if isinstance(convergence, Mapping) and status in {"CONVERGED", "LIMIT HIT"}:
        completed_steps = convergence.get("completed_steps")
        best_step = convergence.get("best_step")
        pieces = []
        if completed_steps is not None:
            pieces.append(f"completed {completed_steps} steps")
        if best_step is not None:
            pieces.append(f"best step {best_step}")
        if pieces:
            lines.append("campaign  " + " · ".join(pieces))

    checkpoint = snapshot.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        checkpoint_time = _parse_utc(checkpoint.get("updated_utc"))
        checked = _parse_utc(snapshot.get("checked_utc"))
        age = (
            max(0.0, (checked - checkpoint_time).total_seconds())
            if checkpoint_time is not None and checked is not None
            else None
        )
        lines.append(
            f"checkpoint {_short_path(str(checkpoint.get('path')))} · {_format_duration(age)} old"
        )
    if snapshot.get("best_checkpoint"):
        lines.append(f"best      {_short_path(str(snapshot['best_checkpoint']))}")

    warnings = snapshot.get("warnings")
    if isinstance(warnings, Sequence) and warnings:
        lines.append("signals   " + str(warnings[0]))
        lines.extend(f"          {warning}" for warning in warnings[1:])
    else:
        lines.append("signals   no hard failure/collapse signal detected")
    if isinstance(evaluation_result, Mapping) and evaluation_result.get("path"):
        lines.append(f"report    {_short_path(str(evaluation_result['path']))}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        help="Explicit run/evaluation directory; otherwise follow the newest verified-active run",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root searched recursively when --run is omitted (default: runs)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=1800.0,
        help="Warn after this many seconds without an artifact update (default: 1800)",
    )
    parser.add_argument(
        "--heartbeat-intervals",
        type=int,
        default=10,
        help="Print an unchanged heartbeat after this many polls; 0 disables it (default: 10)",
    )
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON snapshots")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive")
    if not math.isfinite(args.stale_after) or args.stale_after <= 0:
        parser.error("--stale-after must be positive")
    if args.heartbeat_intervals < 0:
        parser.error("--heartbeat-intervals must be nonnegative")
    return args


def _emit(snapshot: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(snapshot, indent=2, sort_keys=True, default=str), flush=True)
    else:
        print(render_snapshot(snapshot), flush=True)


def _snapshot_fingerprint(snapshot: Mapping[str, Any]) -> tuple[str, str]:
    """Track artifact/process changes plus derived failure/staleness transitions."""

    derived = json.dumps(
        {
            "revision": snapshot.get("revision"),
            "status": snapshot.get("status"),
            "status_detail": snapshot.get("status_detail"),
            "warnings": snapshot.get("warnings"),
        },
        sort_keys=True,
        default=str,
    )
    return str(snapshot.get("run_directory")), hashlib.sha256(derived.encode()).hexdigest()


def monitor(
    *,
    explicit_run: Path | None,
    runs_root: Path,
    interval_seconds: float,
    stale_after_seconds: float,
    heartbeat_intervals: int,
    once: bool,
    as_json: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    last_fingerprint: tuple[str, str] | None = None
    unchanged_polls = 0
    missing_reported = False
    while True:
        run_directory = explicit_run or select_run_directory(runs_root)
        if run_directory is None:
            if once:
                print(
                    f"no run artifacts found under {runs_root.expanduser().resolve()}",
                    file=sys.stderr,
                )
                return 2
            if not missing_reported:
                print(
                    f"Orpheus monitor: waiting for a run under {runs_root.expanduser().resolve()} "
                    f"(polling every {_format_duration(interval_seconds)})",
                    flush=True,
                )
                missing_reported = True
            sleep(interval_seconds)
            continue
        missing_reported = False
        try:
            snapshot = build_snapshot(
                run_directory,
                stale_after_seconds=stale_after_seconds,
            )
        except FileNotFoundError as error:
            if explicit_run is not None or once:
                print(str(error), file=sys.stderr)
                return 2
            sleep(interval_seconds)
            continue
        fingerprint = _snapshot_fingerprint(snapshot)
        if fingerprint != last_fingerprint:
            if last_fingerprint is not None and not as_json:
                print("", flush=True)
            _emit(snapshot, as_json=as_json)
            last_fingerprint = fingerprint
            unchanged_polls = 0
        else:
            unchanged_polls += 1
            if heartbeat_intervals and unchanged_polls >= heartbeat_intervals:
                if as_json:
                    _emit(snapshot, as_json=True)
                else:
                    age = snapshot.get("activity", {}).get("age_seconds")
                    print(
                        f"heartbeat  {snapshot['status']} · {_short_path(str(run_directory))} · "
                        f"last artifact {_format_duration(_number(age))} ago",
                        flush=True,
                    )
                unchanged_polls = 0
        if once:
            return 0
        sleep(interval_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_run = Path(args.run).expanduser().resolve() if args.run else None
    try:
        return monitor(
            explicit_run=explicit_run,
            runs_root=Path(args.runs_root),
            interval_seconds=args.interval,
            stale_after_seconds=args.stale_after,
            heartbeat_intervals=args.heartbeat_intervals,
            once=args.once,
            as_json=args.json,
        )
    except KeyboardInterrupt:
        print("\nOrpheus monitor stopped.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
