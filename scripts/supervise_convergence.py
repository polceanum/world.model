#!/usr/bin/env python3
"""Continue one completed sustained run until its broad objective plateaus."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from world_model.training.convergence import (
    CampaignIncompleteError,
    CampaignInspection,
    ConvergenceDecision,
    decide_continuation,
    inspect_completed_campaign,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.io import atomic_write_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(path: Path, event: str, **fields: Any) -> None:
    record = {"timestamp_utc": _utc_now(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def _write_report(
    path: Path,
    decision: ConvergenceDecision,
    *,
    config_path: Path,
    run_directory: Path,
) -> None:
    payload = {
        "updated_utc": _utc_now(),
        "config": str(config_path),
        "run_directory": str(run_directory),
        **decision.to_dict(),
        "converged": decision.status == "plateau",
        "limit_hit": decision.status == "limit_hit",
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_state(path: Path, *, status: str, **fields: Any) -> None:
    payload = {"updated_utc": _utc_now(), "status": status, **fields}
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"supervisor state is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"supervisor state must be a JSON object: {path}")
    return value


def _acquire_supervisor_lock(path: Path) -> TextIO:
    """Hold one non-blocking process lock for this run until process exit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("another convergence supervisor already owns this run") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _bootout_initial_job(
    label: str,
    *,
    event_path: Path,
    outcome: str,
) -> None:
    """Remove the initial KeepAlive job after verified completion or failure."""

    if outcome not in {"completion", "failure"}:
        raise ValueError("initial job outcome must be 'completion' or 'failure'")

    domain_target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["launchctl", "bootout", domain_target],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        probe = subprocess.run(
            ["launchctl", "print", domain_target],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            raise RuntimeError(
                f"could not boot out {outcome} KeepAlive job {domain_target}: "
                f"{result.stderr.strip()}"
            )
        _append_event(
            event_path,
            "initial_job_already_absent",
            launchctl_target=domain_target,
            outcome=outcome,
        )
        return
    _append_event(
        event_path,
        f"initial_job_booted_out_after_verified_{outcome}",
        launchctl_target=domain_target,
    )


def _bootout_initial_job_safely(
    label: str,
    *,
    event_path: Path,
    outcome: str,
) -> str | None:
    """Attempt cleanup without masking already-durable trainer state."""

    try:
        _bootout_initial_job(
            label,
            event_path=event_path,
            outcome=outcome,
        )
    except Exception as error:
        _append_event(
            event_path,
            "initial_job_bootout_failed",
            outcome=outcome,
            error_type=type(error).__name__,
            error=str(error),
        )
        return str(error)
    return None


def _raise_on_terminal_trainer_failure(
    run_directory: Path,
    *,
    minimum_expected_steps: int,
) -> None:
    """Surface the trainer's atomic terminal failure instead of polling forever."""

    failure_path = run_directory / "training_failure.json"
    state_path = run_directory / "training_state.json"
    for path in (failure_path, state_path):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise CampaignIncompleteError(
                f"trainer state artifact is unreadable before step {minimum_expected_steps}: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise CampaignIncompleteError(
                f"trainer state artifact is not a JSON object before step "
                f"{minimum_expected_steps}: {path}"
            )
        state = payload.get("state")
        if path == failure_path and state != "failed":
            raise CampaignIncompleteError(
                f"trainer failure artifact has invalid state {state!r} before "
                f"step {minimum_expected_steps}: {path}"
            )
        if state != "failed":
            continue
        exception_type = str(payload.get("exception_type") or "UnknownError")
        message = str(payload.get("message") or "no failure message recorded")
        raise CampaignIncompleteError(
            f"trainer reported terminal failure before step "
            f"{minimum_expected_steps} via {path.name}: "
            f"{exception_type}: {message}"
        )


def _pid_matches_training_run(pid: int, run_directory: Path) -> bool:
    """Prove that a live PID is still the trainer for the requested run."""

    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return False
    try:
        arguments = shlex.split(result.stdout.strip())
    except ValueError:
        return False
    if not any(Path(argument).name == "train.py" for argument in arguments):
        return False

    run_directory = run_directory.resolve()
    for index, argument in enumerate(arguments[:-1]):
        value = arguments[index + 1]
        if argument == "--run-name":
            return value == run_directory.name or run_directory.name.endswith(f"-{value}")
        if argument == "--resume":
            checkpoint = Path(value).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).resolve().parents[1] / checkpoint
            checkpoint = checkpoint.resolve()
            if (
                checkpoint.name == "last.pt"
                and checkpoint.parent.name == "checkpoints"
                and checkpoint.parent.parent == run_directory
            ):
                return True
    return False


def _wait_for_completed_segment(
    run_directory: Path,
    *,
    config: OrpheusConfig,
    minimum_expected_steps: int,
    poll_seconds: float,
    event_path: Path,
    monitored_pid: int | None = None,
) -> CampaignInspection:
    """Wait for at least one target, then verify all selector links."""

    summary_path = run_directory / "train_summary.json"
    last_wait_log = 0.0
    while True:
        _raise_on_terminal_trainer_failure(
            run_directory,
            minimum_expected_steps=minimum_expected_steps,
        )
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                completed = int(summary.get("completed_steps", -1))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                completed = -1
            if completed >= minimum_expected_steps:
                inspection = inspect_completed_campaign(run_directory, config)
                return inspection
        if monitored_pid is not None:
            try:
                os.kill(monitored_pid, 0)
            except ProcessLookupError as error:
                raise CampaignIncompleteError(
                    f"trainer PID {monitored_pid} exited before step {minimum_expected_steps}"
                ) from error
            except PermissionError:
                pass
            if not _pid_matches_training_run(monitored_pid, run_directory):
                raise CampaignIncompleteError(
                    f"trainer PID {monitored_pid} no longer identifies the "
                    f"trainer for {run_directory} before step "
                    f"{minimum_expected_steps}"
                )
        now = time.monotonic()
        if now - last_wait_log >= 600.0:
            _append_event(
                event_path,
                "waiting_for_segment",
                minimum_expected_steps=minimum_expected_steps,
            )
            last_wait_log = now
        time.sleep(poll_seconds)


def _archive_segment_summary(run_directory: Path, *, completed_steps: int) -> None:
    source = run_directory / "train_summary.json"
    archive = run_directory / "convergence" / f"train_summary_step_{completed_steps:06d}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive)


def _resume_command(
    *,
    repository_root: Path,
    config_path: Path,
    run_directory: Path,
    device: str,
    next_total_steps: int,
) -> list[str]:
    return [
        sys.executable,
        str(repository_root / "train.py"),
        "--config",
        str(config_path),
        "--resume",
        str(run_directory / "checkpoints" / "last.pt"),
        "--device",
        device,
        "--set",
        f"training.steps={next_total_steps}",
    ]


def _matching_trainer_pid(
    *,
    stored_pid: int | None,
    run_directory: Path,
    target_steps: int,
) -> int | None:
    """Find only the exact in-place extension trainer, avoiding PID-reuse races."""

    required = (
        str(Path(__file__).resolve().parents[1] / "train.py"),
        str(run_directory / "checkpoints" / "last.pt"),
        f"training.steps={target_steps}",
    )
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not inspect trainer processes: {result.stderr.strip()}")
    matches: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if all(fragment in command for fragment in required):
            matches.append(pid)
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple trainers target the same run and step {target_steps}: {matches}"
        )
    if stored_pid is not None and stored_pid in matches:
        return stored_pid
    return matches[0] if matches else None


def _launch_extension(
    *,
    command: list[str],
    repository_root: Path,
    environment: dict[str, str],
    state_path: Path,
    event_path: Path,
    previous_steps: int,
    target_steps: int,
    recovered: bool,
) -> int:
    event = "extension_recovered" if recovered else "extension_started"
    _append_event(
        event_path,
        event,
        command=command,
        previous_steps=previous_steps,
        target_steps=target_steps,
    )
    _write_state(
        state_path,
        status="extension_starting",
        previous_steps=previous_steps,
        target_steps=target_steps,
        command=command,
    )
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=environment,
    )
    _write_state(
        state_path,
        status="extension_running",
        previous_steps=previous_steps,
        target_steps=target_steps,
        child_pid=process.pid,
        command=command,
    )
    returncode = process.wait()
    if returncode != 0:
        _append_event(
            event_path,
            "extension_failed",
            returncode=returncode,
            target_steps=target_steps,
        )
        _write_state(
            state_path,
            status="extension_failed",
            target_steps=target_steps,
            returncode=returncode,
        )
    return returncode


def _record_extension_artifact_failure(
    *,
    state_path: Path,
    event_path: Path,
    target_steps: int,
    error: Exception,
) -> int:
    _append_event(
        event_path,
        "extension_artifact_failure",
        target_steps=target_steps,
        error_type=type(error).__name__,
        error=str(error),
    )
    _write_state(
        state_path,
        status="extension_failed",
        target_steps=target_steps,
        returncode=2,
        error_type=type(error).__name__,
        error=str(error),
    )
    return 2


def _validate_completed_step_grid(
    completed_steps: int,
    *,
    minimum_total_steps: int,
    extension_steps: int,
    maximum_total_steps: int,
) -> None:
    if completed_steps < minimum_total_steps:
        return
    if completed_steps > maximum_total_steps:
        raise RuntimeError("campaign completed beyond the supervisor hard limit")
    if (completed_steps - minimum_total_steps) % extension_steps != 0:
        raise RuntimeError("completed trainer target is not aligned to a complete extension block")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True, help="Existing timestamped run directory")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument(
        "--initial-launchctl-label",
        help=(
            "Initial job to boot out after verified completion or a monitored "
            "initial-trainer failure"
        ),
    )
    parser.add_argument(
        "--initial-trainer-pid",
        type=int,
        help=(
            "PID of the already-running initial trainer; fail explicitly if it "
            "exits before the initial segment verifies"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-total-steps", type=int)
    parser.add_argument("--extension-steps", type=int, default=4096)
    parser.add_argument("--tail-steps", type=int, default=1024)
    parser.add_argument("--minimum-relative-gain", type=float, default=0.01)
    parser.add_argument(
        "--maximum-total-steps",
        type=int,
        default=24576,
        help=(
            "Hard safety limit; reaching it without demonstrated plateau is reported as limit-hit"
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect a completed segment once without waiting, booting out, or resuming",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser().resolve()
    run_directory = Path(args.run).expanduser().resolve()
    config = load_config(config_path)
    minimum_total_steps = (
        config.training.steps if args.minimum_total_steps is None else int(args.minimum_total_steps)
    )
    if minimum_total_steps - config.training.rgb_pretrain_steps < 4096:
        raise ValueError("the convergence supervisor requires the full 4,096-window minimum")
    if config.training.eval_every != 512:
        raise ValueError(
            "the sustained plateau protocol requires validation every 512 optimizer steps"
        )
    if args.maximum_total_steps < minimum_total_steps:
        raise ValueError("maximum-total-steps cannot be below the declared minimum")
    if args.extension_steps <= 0 or args.tail_steps <= 0:
        raise ValueError("extension-steps and tail-steps must be positive")
    if (args.maximum_total_steps - minimum_total_steps) % args.extension_steps != 0:
        raise ValueError("the hard limit must contain a whole number of extension blocks")
    if not 0.0 < args.minimum_relative_gain < 1.0:
        raise ValueError("minimum-relative-gain must lie in (0, 1)")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.initial_trainer_pid is not None and args.initial_trainer_pid <= 0:
        raise ValueError("initial-trainer-pid must be positive")

    event_path = run_directory / "convergence_supervisor.jsonl"
    report_path = run_directory / "convergence_report.json"
    state_path = run_directory / "convergence_supervisor_state.json"
    if args.inspect_only:
        inspection = inspect_completed_campaign(run_directory, config)
        _validate_completed_step_grid(
            inspection.completed_steps,
            minimum_total_steps=minimum_total_steps,
            extension_steps=args.extension_steps,
            maximum_total_steps=args.maximum_total_steps,
        )
        decision = decide_continuation(
            inspection,
            minimum_total_steps=minimum_total_steps,
            extension_steps=args.extension_steps,
            tail_steps=args.tail_steps,
            minimum_relative_gain=args.minimum_relative_gain,
            maximum_total_steps=args.maximum_total_steps,
        )
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
        return 0

    try:
        # Keep this handle live for all waits and child processes. The kernel
        # releases the advisory lock if the supervisor is terminated.
        _supervisor_lock_handle = _acquire_supervisor_lock(
            run_directory / ".convergence_supervisor.lock"
        )
    except RuntimeError as error:
        _append_event(
            event_path,
            "supervisor_already_running",
            error=str(error),
        )
        return 0

    _append_event(
        event_path,
        "supervisor_started",
        config=str(config_path),
        run_directory=str(run_directory),
        minimum_total_steps=minimum_total_steps,
        extension_steps=args.extension_steps,
        tail_steps=args.tail_steps,
        minimum_relative_gain=args.minimum_relative_gain,
        maximum_total_steps=args.maximum_total_steps,
    )
    state = _read_state(state_path)
    if state.get("status") in {"extension_failed", "initial_trainer_failed"}:
        recorded_status = str(state["status"])
        if recorded_status == "initial_trainer_failed" and args.initial_launchctl_label:
            _bootout_initial_job_safely(
                args.initial_launchctl_label,
                event_path=event_path,
                outcome="failure",
            )
        _append_event(
            event_path,
            "supervisor_stopped_on_recorded_failure",
            failure_status=recorded_status,
            target_steps=state.get("target_steps"),
            returncode=state.get("returncode"),
            trainer_pid=state.get("trainer_pid"),
        )
        # A LaunchAgent may restart this supervisor after an abnormal child
        # exit. Exit successfully on the second invocation so a deterministic
        # failure cannot become an infinite automatic retry loop.
        return 0
    pending_target_value = (
        state.get("target_steps")
        if state.get("status") in {"extension_starting", "extension_running"}
        else None
    )
    pending_target = int(pending_target_value) if pending_target_value is not None else None
    if pending_target is None:
        try:
            inspection = _wait_for_completed_segment(
                run_directory,
                config=config,
                minimum_expected_steps=minimum_total_steps,
                poll_seconds=args.poll_seconds,
                event_path=event_path,
                monitored_pid=args.initial_trainer_pid,
            )
        except Exception as error:
            _append_event(
                event_path,
                "initial_segment_failed",
                trainer_pid=args.initial_trainer_pid,
                minimum_expected_steps=minimum_total_steps,
                error_type=type(error).__name__,
                error=str(error),
            )
            _write_state(
                state_path,
                status="initial_trainer_failed",
                trainer_pid=args.initial_trainer_pid,
                target_steps=minimum_total_steps,
                returncode=2,
                error_type=type(error).__name__,
                error=str(error),
            )
            cleanup_error = None
            if args.initial_launchctl_label:
                cleanup_error = _bootout_initial_job_safely(
                    args.initial_launchctl_label,
                    event_path=event_path,
                    outcome="failure",
                )
                if cleanup_error is not None:
                    _write_state(
                        state_path,
                        status="initial_trainer_failed",
                        trainer_pid=args.initial_trainer_pid,
                        target_steps=minimum_total_steps,
                        returncode=2,
                        error_type=type(error).__name__,
                        error=str(error),
                        cleanup_error=cleanup_error,
                    )
            return 2
    else:
        stored_pid_value = state.get("child_pid")
        stored_pid = int(stored_pid_value) if stored_pid_value is not None else None
        active_pid = _matching_trainer_pid(
            stored_pid=stored_pid,
            run_directory=run_directory,
            target_steps=pending_target,
        )
        summary_path = run_directory / "train_summary.json"
        summary_steps = -1
        if summary_path.is_file():
            try:
                summary_steps = int(
                    json.loads(summary_path.read_text(encoding="utf-8")).get(
                        "completed_steps",
                        -1,
                    )
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                summary_steps = -1
        if summary_steps < pending_target and active_pid is None:
            command = _resume_command(
                repository_root=repository_root,
                config_path=config_path,
                run_directory=run_directory,
                device=args.device,
                next_total_steps=pending_target,
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repository_root)
            returncode = _launch_extension(
                command=command,
                repository_root=repository_root,
                environment=environment,
                state_path=state_path,
                event_path=event_path,
                previous_steps=max(minimum_total_steps, summary_steps),
                target_steps=pending_target,
                recovered=True,
            )
            if returncode != 0:
                return returncode
            try:
                inspection = inspect_completed_campaign(run_directory, config)
                if inspection.completed_steps < pending_target:
                    raise CampaignIncompleteError(
                        "recovered extension exited successfully before its target "
                        f"step {pending_target}"
                    )
            except Exception as error:
                return _record_extension_artifact_failure(
                    state_path=state_path,
                    event_path=event_path,
                    target_steps=pending_target,
                    error=error,
                )
        elif active_pid is not None:
            _append_event(
                event_path,
                "reattached_to_extension",
                child_pid=active_pid,
                target_steps=pending_target,
            )
            try:
                inspection = _wait_for_completed_segment(
                    run_directory,
                    config=config,
                    minimum_expected_steps=pending_target,
                    poll_seconds=args.poll_seconds,
                    event_path=event_path,
                    monitored_pid=active_pid,
                )
            except Exception as error:
                return _record_extension_artifact_failure(
                    state_path=state_path,
                    event_path=event_path,
                    target_steps=pending_target,
                    error=error,
                )
        else:
            try:
                inspection = inspect_completed_campaign(run_directory, config)
            except Exception as error:
                return _record_extension_artifact_failure(
                    state_path=state_path,
                    event_path=event_path,
                    target_steps=pending_target,
                    error=error,
                )
    _validate_completed_step_grid(
        inspection.completed_steps,
        minimum_total_steps=minimum_total_steps,
        extension_steps=args.extension_steps,
        maximum_total_steps=args.maximum_total_steps,
    )
    _write_state(
        state_path,
        status="segment_completed",
        completed_steps=inspection.completed_steps,
    )
    if args.initial_launchctl_label:
        _bootout_initial_job(
            args.initial_launchctl_label,
            event_path=event_path,
            outcome="completion",
        )

    while True:
        decision = decide_continuation(
            inspection,
            minimum_total_steps=minimum_total_steps,
            extension_steps=args.extension_steps,
            tail_steps=args.tail_steps,
            minimum_relative_gain=args.minimum_relative_gain,
            maximum_total_steps=args.maximum_total_steps,
        )
        _write_report(
            report_path,
            decision,
            config_path=config_path,
            run_directory=run_directory,
        )
        _append_event(event_path, "convergence_decision", **decision.to_dict())
        if decision.status != "continue":
            _write_state(
                state_path,
                status=decision.status,
                completed_steps=inspection.completed_steps,
                reason=decision.reason,
            )
            return 0
        if decision.next_total_steps is None:
            raise AssertionError("continue decision lacks a next step target")
        _archive_segment_summary(
            run_directory,
            completed_steps=inspection.completed_steps,
        )
        command = _resume_command(
            repository_root=repository_root,
            config_path=config_path,
            run_directory=run_directory,
            device=args.device,
            next_total_steps=decision.next_total_steps,
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository_root)
        returncode = _launch_extension(
            command=command,
            repository_root=repository_root,
            environment=environment,
            state_path=state_path,
            event_path=event_path,
            previous_steps=inspection.completed_steps,
            target_steps=decision.next_total_steps,
            recovered=False,
        )
        if returncode != 0:
            return returncode
        try:
            inspection = inspect_completed_campaign(run_directory, config)
            if inspection.completed_steps != decision.next_total_steps:
                raise CampaignIncompleteError(
                    "completed extension does not match its declared target"
                )
        except Exception as error:
            return _record_extension_artifact_failure(
                state_path=state_path,
                event_path=event_path,
                target_steps=decision.next_total_steps,
                error=error,
            )
        _write_state(
            state_path,
            status="segment_completed",
            completed_steps=inspection.completed_steps,
        )


if __name__ == "__main__":
    raise SystemExit(main())
