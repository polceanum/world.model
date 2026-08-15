#!/usr/bin/env python3
"""Sweep event thresholds from evaluator probability histograms.

The evaluator stores ten fixed probability bins, split by positive and
negative targets.  This utility estimates precision/recall/F1 at bin-aligned
thresholds without rerunning the rollout.  It is deliberately conservative:
thresholds between bin boundaries are not reported because the histogram does
not contain enough information to distinguish them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metrics(positive: int, negative: int, threshold_bin: int) -> dict[str, float]:
    tp = sum(positive[threshold_bin:]) if threshold_bin < len(positive) else 0
    fp = sum(negative[threshold_bin:]) if threshold_bin < len(negative) else 0
    fn = sum(positive[:threshold_bin])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold_bin / 10.0,
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sweep_report(report: dict[str, Any]) -> dict[str, dict[str, list[dict[str, float]]]]:
    """Return threshold metrics keyed by horizon and candidate name."""
    names = report.get("candidate_names")
    if not names:
        names = [
            str(i)
            for i in range(
                len(
                    next(iter(report["episode_results"])["event_probability_histograms"].values())[
                        0
                    ]
                )
            )
        ]
    aggregate: dict[str, dict[str, list[list[int]]]] = {}
    for episode in report["episode_results"]:
        for horizon, candidates in episode["event_probability_positive_histograms"].items():
            neg_candidates = episode["event_probability_negative_histograms"][horizon]
            slot = aggregate.setdefault(horizon, {})
            for index, (pos, neg) in enumerate(zip(candidates, neg_candidates, strict=True)):
                key = names[index]
                acc = slot.setdefault(key, [[0] * 10, [0] * 10])
                acc[0] = [a + b for a, b in zip(acc[0], pos, strict=True)]
                acc[1] = [a + b for a, b in zip(acc[1], neg, strict=True)]
    return {
        horizon: {
            candidate: [_metrics(pos, neg, threshold_bin) for threshold_bin in range(11)]
            for candidate, (pos, neg) in candidates.items()
        }
        for horizon, candidates in aggregate.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = sweep_report(json.loads(args.report.read_text()))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
