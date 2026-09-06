#!/usr/bin/env python3
"""Explicit, split-safe CLI for the compact specification-1.61 campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from world_model.training.dynamic_set_qualification import (
    DynamicSetQualification,
    DynamicSetSourceFreeze,
    authenticate_dynamic_set_source_freeze,
    build_dynamic_set_protocol_binding,
    capture_dynamic_set_source_freeze,
    validate_known_action_foundation,
)
from world_model.training.qualification_core import sha256_bytes


def _json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON object from {path}") from error
    if type(value) is not dict:
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _source(path: str | Path) -> DynamicSetSourceFreeze:
    value = _json_object(path)
    if set(value) != {
        "source_sha256",
        "commit",
        "tree",
        "upstream_commit",
        "clean",
        "published",
    }:
        raise ValueError("source JSON differs from the frozen source-binding schema")
    return authenticate_dynamic_set_source_freeze(DynamicSetSourceFreeze(**value))


def _print(value: object) -> None:
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight, initialise, inspect, or explicitly consume one phase of the "
            "specification-1.61 dynamic-set qualification."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser(
        "capture-source",
        help="read-only capture of the current clean, upstream-equal source",
    )
    capture.add_argument("--repository-root")

    preflight = commands.add_parser("preflight", help="read-only protocol/foundation check")
    preflight.add_argument("--known-action-directory", required=True)
    preflight.add_argument("--source-json", required=True)
    preflight.add_argument("--config", required=True)

    initialise = commands.add_parser("init", help="create a fresh development campaign")
    initialise.add_argument("--run-directory", required=True)
    initialise.add_argument("--known-action-directory", required=True)
    initialise.add_argument("--source-json", required=True)
    initialise.add_argument("--config", required=True)

    status = commands.add_parser("status", help="read status without opening a split")
    status.add_argument("--run-directory", required=True)

    screen = commands.add_parser("execute-screen", help="consume the disposable screen once")
    screen.add_argument("--run-directory", required=True)
    screen.add_argument("--work-directory", required=True)
    screen.add_argument("--config", required=True)
    screen.add_argument("--updates", type=int, default=512)

    begin = commands.add_parser("begin-campaign", help="begin training after a passed screen")
    begin.add_argument("--run-directory", required=True)

    validation = commands.add_parser(
        "execute-validation", help="evaluate one exact 512-update development candidate"
    )
    validation.add_argument("--run-directory", required=True)
    validation.add_argument("--completed-updates", required=True, type=int)
    validation.add_argument("--checkpoint", required=True)
    validation.add_argument("--execution-progress", required=True)
    validation.add_argument("--model-state-sha256", required=True)
    validation.add_argument("--config", required=True)

    finish = commands.add_parser("finish-campaign", help="publish one four-way campaign result")
    finish.add_argument("--run-directory", required=True)
    finish.add_argument(
        "--status",
        required=True,
        choices=(
            "qualified_convergence",
            "objective_plateau",
            "failed_to_improve",
            "limit_hit",
        ),
    )
    finish.add_argument("--execution-progress", required=True)

    review = commands.add_parser(
        "review-development", help="record an independently created digest receipt"
    )
    review.add_argument("--run-directory", required=True)
    review.add_argument("--receipt-json", required=True)

    protected = commands.add_parser(
        "init-protected", help="create the protected ledger after development review"
    )
    protected.add_argument("--run-directory", required=True)

    execute = commands.add_parser(
        "execute-protected", help="consume exactly the next protected split"
    )
    execute.add_argument("--run-directory", required=True)
    execute.add_argument(
        "--split",
        required=True,
        choices=("selector", "confirmation", "final_test", "compositional_ood"),
    )
    execute.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.command == "capture-source":
        source = (
            capture_dynamic_set_source_freeze()
            if parsed.repository_root is None
            else capture_dynamic_set_source_freeze(parsed.repository_root)
        )
        _print(asdict(source))
        return 0
    if parsed.command == "preflight":
        foundation = validate_known_action_foundation(parsed.known_action_directory)
        source = _source(parsed.source_json)
        config_sha256 = sha256_bytes(Path(parsed.config).read_bytes())
        protocol = build_dynamic_set_protocol_binding(
            foundation=foundation,
            source=source,
            config_sha256=config_sha256,
        )
        _print(protocol)
        return 0
    if parsed.command == "init":
        source = _source(parsed.source_json)
        config_sha256 = sha256_bytes(Path(parsed.config).read_bytes())
        qualification = DynamicSetQualification.create_fresh(
            parsed.run_directory,
            known_action_directory=parsed.known_action_directory,
            source=source,
            config_sha256=config_sha256,
        )
        _print(qualification.status())
        return 0

    qualification = DynamicSetQualification.attach(parsed.run_directory)
    if parsed.command == "status":
        _print(qualification.status())
    elif parsed.command == "execute-screen":
        result = qualification.execute_screen(
            config_path=parsed.config,
            work_directory=parsed.work_directory,
            updates=parsed.updates,
        )
        _print(result)
        return 0 if result.get("passed") is True else 2
    elif parsed.command == "begin-campaign":
        _print({"permit": asdict(qualification.begin_campaign())})
    elif parsed.command == "execute-validation":
        _print(
            qualification.execute_validation_candidate(
                completed_updates=parsed.completed_updates,
                checkpoint_path=parsed.checkpoint,
                execution_progress_path=parsed.execution_progress,
                model_state_sha256=parsed.model_state_sha256,
                config_path=parsed.config,
            )
        )
    elif parsed.command == "finish-campaign":
        result = qualification.finish_campaign(
            requested_status=parsed.status,
            execution_progress_path=parsed.execution_progress,
        )
        _print(result)
        return 0 if result.get("status") == "qualified_convergence" else 2
    elif parsed.command == "review-development":
        _print(
            qualification.record_independent_development_review(_json_object(parsed.receipt_json))
        )
    elif parsed.command == "init-protected":
        _print(qualification.create_protected_ledger())
    elif parsed.command == "execute-protected":
        result = qualification.execute_protected_split(
            parsed.split,
            config_path=parsed.config,
        )
        _print(result)
        return 0 if result.get("passed") is True else 2
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(f"unhandled command {parsed.command!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
