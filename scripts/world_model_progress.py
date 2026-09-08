#!/usr/bin/env python3
"""Build capability reports and safely inspect/prune compact run artifacts."""

from __future__ import annotations

import argparse
import json

from world_model.utils.run_artifacts import apply_cleanup, inventory_runs, plan_cleanup
from world_model.visualisation.progress import build_progress_dashboard


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--archive-root", default=".archive")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="rebuild the self-contained static dashboard")
    commands.add_parser("list", help="list run roles, sizes, and pruning eligibility")
    prune = commands.add_parser("prune", help="enforce the rolling budget (dry-run default)")
    prune.add_argument("--apply", action="store_true")
    clear = commands.add_parser("clear", help="clear one safe manifest-scoped category")
    clear.add_argument("--category", choices=("transient", "media", "rejected"), required=True)
    clear.add_argument("--apply", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    if parsed.command == "build":
        path = build_progress_dashboard(parsed.runs_root, archive_root=parsed.archive_root)
        output: object = {"dashboard": str(path)}
    elif parsed.command == "list":
        output = inventory_runs(parsed.runs_root, archive_root=parsed.archive_root)
    else:
        category = parsed.category if parsed.command == "clear" else None
        plan = plan_cleanup(
            parsed.runs_root,
            archive_root=parsed.archive_root,
            category=category,
        )
        output = apply_cleanup(plan, parsed.runs_root).to_dict() if parsed.apply else plan.to_dict()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
