#!/usr/bin/env python3
"""Compare hypothesis reports and fail on guarded regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mean(values: Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _aggregate(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    episodes = report["episode_results"]
    result: dict[str, dict[str, Any]] = {}
    for horizon in report["horizons_seconds"]:
        key = str(horizon)
        rmse = [episode["selected_rmse_m"][key] for episode in episodes]
        result[key] = {
            "rmse": [_mean([row[axis] for row in rmse]) for axis in range(3)],
            "lifecycle_mismatch": sum(
                episode["selected_lifecycle_mismatch"][key] for episode in episodes
            ),
            "identity_coverage": sum(
                episode["selected_identity_coverage"][key] for episode in episodes
            ),
            "event_f1": _mean(
                episode["selected_event_metrics"][key]["collision_f1"] for episode in episodes
            ),
            "uncertainty": _mean(
                episode["selected_mean_position_std_m"][key] for episode in episodes
            ),
        }
    return result


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any], *, tolerance: float = 1.0e-6
) -> dict[str, Any]:
    """Return deltas and a strict all-metric non-regression decision."""
    base = _aggregate(baseline)
    trial = _aggregate(candidate)
    if set(base) != set(trial):
        raise ValueError("reports must contain the same horizons")
    horizons: dict[str, Any] = {}
    regressions: list[str] = []
    for horizon in base:
        b, c = base[horizon], trial[horizon]
        rmse_delta = [c["rmse"][axis] - b["rmse"][axis] for axis in range(3)]
        delta = {
            "rmse_delta": rmse_delta,
            "lifecycle_delta": c["lifecycle_mismatch"] - b["lifecycle_mismatch"],
            "identity_delta": c["identity_coverage"] - b["identity_coverage"],
            "event_f1_delta": c["event_f1"] - b["event_f1"],
            "uncertainty_delta": c["uncertainty"] - b["uncertainty"],
        }
        horizons[horizon] = delta
        for axis, value in enumerate(rmse_delta):
            if value > tolerance:
                regressions.append(f"{horizon}.rmse_axis_{axis}")
        if delta["lifecycle_delta"] > 0:
            regressions.append(f"{horizon}.lifecycle")
        if delta["identity_delta"] < 0:
            regressions.append(f"{horizon}.identity")
        if delta["event_f1_delta"] < -tolerance:
            regressions.append(f"{horizon}.event_f1")
        if delta["uncertainty_delta"] > tolerance:
            regressions.append(f"{horizon}.uncertainty")
    return {"horizons": horizons, "regressions": regressions, "passed": not regressions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tolerance < 0:
        raise ValueError("--tolerance must be nonnegative")
    result = compare_reports(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        tolerance=args.tolerance,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
