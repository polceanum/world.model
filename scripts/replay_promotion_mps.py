#!/usr/bin/env python3
"""Replay a candidate/reference selector pair on the active-Aqua MPS backend.

This is deliberately a promotion gate rather than a generic evaluation tool.
It reruns the trainer's exact fixed validation manifest for both immutable
checkpoints, derives the existing physical selector metrics from raw additive
evidence, and applies the same lifecycle, identity, event, horizon, axis, and
calibration guardrails used during training.  It never mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import load_model_weights
from world_model.training.trainer import (
    _handoff_training_support_failures,
    _make_loader,
    _mutable_causal_training_support_failures,
    _rollout_selection_guardrail_failures,
    _rollout_selection_improves,
    _rollout_selection_metrics,
    _validate_validation_support_schema,
    _validation_loader_result,
)
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import load_config
from world_model.utils.device import select_device
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--reference",
        required=True,
        help="Protected incumbent/reference checkpoint to replay on MPS.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate checkpoint to replay on the same MPS manifest.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Report label/path; its basename receives a UTC timestamp prefix.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Validation workers; zero keeps the MPS replay maximally auditable.",
    )
    return parser.parse_args()


def _replay_checkpoint(
    checkpoint: Path,
    *,
    config: Any,
    device: torch.device,
    output: Path,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    """Return selector metrics plus raw evidence from one isolated runtime."""

    model = OnlineWorldModel.from_config(config, device=device)
    load_model_weights(checkpoint, model=model, expected_config=config)
    model.eval()
    model.reset()
    loader = _make_loader(
        config,
        split="validation",
        episodes=config.training.validation_episodes,
        shuffle=False,
        batch_size_override=1,
    )
    validation = _validation_loader_result(
        model,
        loader,
        config,
        device=device,
        closed_loop=True,
        progress_path=output / f"{role}_validation_progress.json",
        progress_split=f"mps_{role}",
    )
    _validate_validation_support_schema(validation.metrics, config)
    selector = _rollout_selection_metrics(
        validation.metrics,
        config,
        require_scenarios=True,
    )
    raw_metrics = dict(validation.metrics)
    atomic_write_text(
        output / f"{role}_validation_metrics.json",
        json.dumps(raw_metrics, indent=2, sort_keys=True) + "\n",
    )
    return selector, raw_metrics


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    config = load_config(
        args.config,
        overrides=[
            "device.preference=mps",
            "device.closed_loop_preference=mps",
            f"training.num_workers={args.num_workers}",
        ],
    )
    config.validate()
    device_info = select_device("mps")
    if device_info.device.type != "mps":
        raise RuntimeError("promotion replay requires an active Aqua MPS device")
    device = device_info.device
    seed_everything(config.project.seed, deterministic=config.project.deterministic)

    reference_path = Path(args.reference).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    if reference_path == candidate_path:
        raise ValueError("reference and candidate must be distinct checkpoints")
    output = timestamped_artifact_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    reference, _ = _replay_checkpoint(
        reference_path,
        config=config,
        device=device,
        output=output,
        role="reference",
    )
    candidate, _ = _replay_checkpoint(
        candidate_path,
        config=config,
        device=device,
        output=output,
        role="candidate",
    )
    reference_guardrail_failures = _rollout_selection_guardrail_failures(candidate, reference)
    training_support_failures = _handoff_training_support_failures(
        candidate,
        reference,
        config,
    )
    mutable_support_failures = _mutable_causal_training_support_failures(candidate, config)
    accepted = (
        not reference_guardrail_failures
        and not training_support_failures
        and not mutable_support_failures
        and _rollout_selection_improves(candidate, reference)
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_backend": str(device),
        "promotion_eligible": bool(accepted),
        "reference_checkpoint": str(reference_path),
        "candidate_checkpoint": str(candidate_path),
        "validation_episode_count": config.training.validation_episodes,
        "scenario_mixture": list(config.simulator.scenario_mixture),
        "reference": reference.validation_metrics(),
        "candidate": candidate.validation_metrics(),
        "reference_guardrail_failures": reference_guardrail_failures,
        "training_support_failures": training_support_failures,
        "mutable_support_failures": mutable_support_failures,
    }
    atomic_write_text(output / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
