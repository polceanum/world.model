#!/usr/bin/env python3
"""Run one development-only N=8 RGB-D capacity probe."""

from __future__ import annotations

import argparse
import json

from world_model.evaluation.perceptual_scaling import run_n8_perception_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/rgbd_dynamic_set_planning_cpu.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/20260907-capability-development-v2/selected_checkpoint.pt",
    )
    parsed = parser.parse_args(argv)
    result = run_n8_perception_probe(
        model_config_path=parsed.config,
        checkpoint_path=parsed.checkpoint,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
