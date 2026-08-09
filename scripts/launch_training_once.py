#!/usr/bin/env python3
"""Launch one persistent macOS training process without automatic retries."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from world_model.utils.io import atomic_write_text

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Unique launchd job label")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-name",
        help=(
            "Run name for a new run. Omit this for an exact in-place resume "
            "from the source run's checkpoints/last.pt."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--initialize-from")
    source.add_argument("--resume")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--stdout")
    parser.add_argument("--stderr")
    parser.add_argument("--plist")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch specification without writing or loading it",
    )
    return parser.parse_args()


def build_launchd_payload(
    *,
    label: str,
    repository_root: Path,
    python_executable: Path,
    config_path: Path,
    run_name: str | None,
    device: str,
    initialize_from: Path | None,
    resume: Path | None,
    overrides: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    """Build a one-shot LaunchAgent payload suitable for ``launchctl bootstrap``."""

    if not _SAFE_LABEL.fullmatch(label):
        raise ValueError("launch label contains unsupported characters")
    if initialize_from is not None and resume is not None:
        raise ValueError("initialize_from and resume are mutually exclusive")
    program_arguments = [
        "/usr/bin/caffeinate",
        "-dimsu",
        str(python_executable),
        str(repository_root / "train.py"),
        "--config",
        str(config_path),
    ]
    if run_name is not None:
        program_arguments.extend(("--run-name", run_name))
    program_arguments.extend(("--device", device))
    if initialize_from is not None:
        program_arguments.extend(("--initialize-from", str(initialize_from)))
    if resume is not None:
        program_arguments.extend(("--resume", str(resume)))
    for override in overrides:
        program_arguments.extend(("--set", override))
    return {
        "Label": label,
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(repository_root),
        "EnvironmentVariables": {
            "PYTHONPATH": str(repository_root),
        },
        "RunAtLoad": True,
        # A failed or completed trainer must remain terminal. ``launchctl
        # submit`` inferred KeepAlive and previously retried one failed
        # initialization thousands of times.
        "KeepAlive": False,
        # Do not classify explicitly requested training as ``Background``:
        # macOS then applies resource limits that can reduce a multi-core
        # trainer to roughly one core and make healthy validation look
        # stalled.  Omitting ProcessType selects launchd's portable Standard
        # classification; caffeinate still preserves the one-shot process
        # across idle sleep.
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    initialize_from = (
        None if args.initialize_from is None else Path(args.initialize_from).expanduser().resolve()
    )
    resume = None if args.resume is None else Path(args.resume).expanduser().resolve()
    for name, source in (("initialization checkpoint", initialize_from), ("resume", resume)):
        if source is not None and not source.is_file():
            raise FileNotFoundError(f"{name} not found: {source}")
    if args.run_name is None and resume is None:
        raise ValueError("--run-name is required unless performing an exact --resume")
    artifact_name = args.run_name
    if artifact_name is None:
        assert resume is not None
        artifact_name = resume.parent.parent.name
    stdout_path = (
        Path(args.stdout or f"/private/tmp/{artifact_name}.stdout.log").expanduser().resolve()
    )
    stderr_path = (
        Path(args.stderr or f"/private/tmp/{artifact_name}.stderr.log").expanduser().resolve()
    )
    plist_path = Path(args.plist or f"/private/tmp/{args.label}.plist").expanduser().resolve()
    payload = build_launchd_payload(
        label=args.label,
        repository_root=repository_root,
        python_executable=Path(sys.executable).resolve(),
        config_path=config_path,
        run_name=args.run_name,
        device=args.device,
        initialize_from=initialize_from,
        resume=resume,
        overrides=list(args.set),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    summary = {
        "label": args.label,
        "launchctl_target": f"gui/{os.getuid()}/{args.label}",
        "plist": str(plist_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "keep_alive": False,
        "process_type": payload.get("ProcessType", "Standard (launchd default)"),
        "program_arguments": payload["ProgramArguments"],
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    probe = subprocess.run(
        ["launchctl", "print", summary["launchctl_target"]],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        raise RuntimeError(f"launchd job is already loaded: {summary['launchctl_target']}")
    plist_text = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True).decode("utf-8")
    atomic_write_text(plist_path, plist_text)
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)],
        check=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
