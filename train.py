#!/usr/bin/env python3
"""Train Project Orpheus from a validated YAML configuration."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch

from world_model.training.trainer import _resolve_run_directory, _resolve_training_devices
from world_model.utils.config import load_config
from world_model.utils.io import append_jsonl, atomic_write_text


def _acquire_training_lock(path: Path):
    """Own one non-blocking trainer lock for a run until invocation exit."""

    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another trainer already owns this run: {path.parent}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    parser.add_argument(
        "--run-name",
        help="Run label; the directory receives a sortable UTC timestamp prefix",
    )
    parser.add_argument("--resume", help="Trusted local checkpoint to resume")
    parser.add_argument(
        "--initialize-from",
        help=(
            "Trusted local checkpoint providing model weights only; starts a new "
            "run with step/optimizer/RNG reset"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = list(args.set)
    if args.device:
        overrides.append(f"device.preference={args.device}")
    if args.seed is not None:
        overrides.append(f"project.seed={args.seed}")
    config = load_config(args.config, overrides=overrides)

    if args.dry_run:
        resume_step = 0
        if args.resume is not None:
            resume_source = Path(args.resume).expanduser().resolve()
            if not resume_source.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {resume_source}")
            payload = torch.load(
                resume_source,
                map_location="cpu",
                weights_only=False,
            )
            resume_step = int(payload.get("step", -1))
            if resume_step < 0:
                raise ValueError("checkpoint step must be a nonnegative integer")
        device_info, measurement_device, closed_loop_device, active_device = (
            _resolve_training_devices(
                config,
                start_step=resume_step,
                initialize_from=args.initialize_from is not None,
            )
        )
        print(
            json.dumps(
                {
                    "command": "train",
                    "config": config.source_path,
                    "project": config.project.name,
                    "run_name": args.run_name,
                    "resume": args.resume,
                    "initialize_from": args.initialize_from,
                    "device": str(active_device),
                    "measurement_device": str(measurement_device),
                    "closed_loop_device": str(closed_loop_device),
                    "torch": device_info.torch_version,
                    "mps_built": device_info.mps_built,
                    "mps_available": device_info.mps_available,
                    "simulator": config.simulator.type,
                    "image_size": config.simulator.image_size,
                    "sequence_frames": config.simulator.sequence_frames,
                    "runtime_modality": config.runtime.modality,
                    "rgb_only_evaluation": config.evaluation.rgb_only,
                    "training_steps": config.training.steps,
                    "train_episodes": config.training.train_episodes,
                    "validation_episodes": config.training.validation_episodes,
                    "batch_size": config.training.batch_size,
                    "scenario_balanced_batches": (config.training.scenario_balanced_batches),
                    "nominal_training_episode_draws": (
                        config.training.steps * config.training.batch_size
                    ),
                    "nominal_dataset_passes": (
                        config.training.steps
                        * config.training.batch_size
                        / config.training.train_episodes
                    ),
                    "scenario_families": list(config.simulator.scenario_mixture),
                    "fixed_dataset": config.training.fixed_dataset,
                },
                indent=2,
            )
        )
        return 0

    from world_model.training.trainer import train_from_config

    effective_run_name = args.run_name
    planned_run_directory = _resolve_run_directory(
        config,
        run_name=args.run_name,
        resume_path=args.resume,
    )
    starts_new_directory = args.resume is None or args.run_name is not None
    if starts_new_directory:
        # Claim the complete run directory, including early-failure state.
        # Treat even a state-only directory as occupied so an accidental retry
        # cannot overwrite the original diagnostic or leave stale failure
        # evidence beside a later successful run.
        planned_run_directory.mkdir(parents=True, exist_ok=False)
        effective_run_name = planned_run_directory.name
    training_lock = _acquire_training_lock(planned_run_directory / ".training.lock")
    state_path = planned_run_directory / "training_state.json"
    failure_path = planned_run_directory / "training_failure.json"
    previous_state: dict[str, object] | None = None
    if not starts_new_directory and state_path.is_file():
        try:
            decoded_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            decoded_state = None
        if isinstance(decoded_state, dict):
            previous_state = {
                key: decoded_state[key]
                for key in (
                    "state",
                    "updated_utc",
                    "run_directory",
                    "completed_steps",
                    "best_checkpoint",
                    "best_checkpoint_kind",
                    "exception_type",
                    "message",
                )
                if key in decoded_state
            }
    # ``training_failure.json`` describes only the current terminal attempt.
    # Its append-only history is retained separately, while a new starting
    # state must not coexist with a stale current-failure marker.
    failure_path.unlink(missing_ok=True)
    atomic_write_text(
        state_path,
        json.dumps(
            {
                "state": "starting",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "config": str(Path(args.config).expanduser().resolve()),
                "run_directory": str(planned_run_directory),
                "resume": args.resume,
                "initialize_from": args.initialize_from,
                "previous_state": previous_state,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    try:
        atomic_write_text(
            state_path,
            json.dumps(
                {
                    "state": "running",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "config": str(Path(args.config).expanduser().resolve()),
                    "run_directory": str(planned_run_directory),
                    "resume": args.resume,
                    "initialize_from": args.initialize_from,
                    "previous_state": previous_state,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        result = train_from_config(
            config,
            run_name=effective_run_name,
            resume_path=args.resume,
            initialize_from_path=args.initialize_from,
            _run_lock_handle=training_lock,
            _cli_claimed_empty_run_directory=starts_new_directory,
        )
    except BaseException as error:
        failure = {
            "state": "failed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(planned_run_directory),
            "resume": args.resume,
            "initialize_from": args.initialize_from,
            "previous_state": previous_state,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        diagnostics = getattr(error, "diagnostics", None)
        if isinstance(diagnostics, Mapping):
            failure["diagnostics"] = dict(diagnostics)
        encoded_failure = json.dumps(failure, indent=2, sort_keys=True) + "\n"
        atomic_write_text(failure_path, encoded_failure)
        atomic_write_text(state_path, encoded_failure)
        with contextlib.suppress(OSError):
            append_jsonl(planned_run_directory / "training_failures.jsonl", failure)
        # The atomic current-state artifacts above are authoritative. A
        # best-effort historical append must never mask the trainer error.
        training_lock.close()
        raise
    failure_path.unlink(missing_ok=True)
    atomic_write_text(
        state_path,
        json.dumps(
            {
                "state": "completed",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "run_directory": result["run_directory"],
                "completed_steps": result["completed_steps"],
                "best_checkpoint": result["best_checkpoint"],
                "best_checkpoint_kind": result["best_checkpoint_kind"],
                "previous_state": previous_state,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    training_lock.close()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
