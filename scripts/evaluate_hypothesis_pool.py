#!/usr/bin/env python3
"""Evaluate a heterogeneous short-step hypothesis pool on RGB episodes.

Simulator object state is used only as evaluation supervision.  The runtime
receives RGB packets, predicts from ``WorldBelief``, and the pool is updated
with delayed future targets.  This is intentionally an evaluation tool, not a
training shortcut or an oracle runtime path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from world_model.dynamics import ConstantVelocityDynamics, HypothesisDynamicsPool
from world_model.observations import ObservationPacket
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.checkpointing import load_checkpoint
from world_model.utils.config import load_config
from world_model.utils.device import select_device


def _packet(episode: dict[str, Any], index: int, image_size: tuple[int, int]) -> ObservationPacket:
    return ObservationPacket(
        modality="rgb",
        sensor_id="camera0",
        timestamp=float(episode["timestamps"][index]),
        payload=episode["rgb"][index],
        calibration={
            "intrinsics": episode["camera"]["intrinsics"][index],
            "world_from_camera": episode["camera"]["world_from_camera"][index],
        },
        frame_id="camera:camera0",
        metadata={"image_size": image_size},
    )


def _future_targets_aligned_to_belief(
    model: OnlineWorldModel,
    episode: dict[str, Any],
    future_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map simulator supervision onto persistent runtime object identities."""

    assert model.belief is not None
    objects = model.belief.objects
    positions = episode["objects"]["position"][future_index]
    active = episode["objects"]["active"][future_index]
    ids = episode["objects"]["id"][future_index]
    target = objects.position.new_zeros(1, 1, objects.max_objects, 3)
    mask = torch.zeros(1, 1, objects.max_objects, dtype=torch.bool, device=objects.active.device)
    for slot, object_id in enumerate(objects.object_id[0].detach().cpu().tolist()):
        if object_id < 0:
            continue
        matches = torch.where(ids == object_id)[0]
        if matches.numel() != 1:
            continue
        source = int(matches[0])
        target[0, 0, slot] = positions[source].to(device=target.device, dtype=target.dtype)
        mask[0, 0, slot] = bool(active[source])
    return target, mask


def _rmse(sum_of_squares: list[float], count: list[int]) -> list[float | None]:
    return [
        math.sqrt(total / number) if number else None
        for total, number in zip(sum_of_squares, count, strict=True)
    ]


def evaluate_episode(
    model: OnlineWorldModel,
    episode: dict[str, Any],
    horizons: tuple[float, ...],
    frame_rate: int,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    model.reset(batch_size=1)
    pool: HypothesisDynamicsPool | None = None
    candidate_squares = [[[0.0, 0.0, 0.0] for _ in range(2)] for _ in horizons]
    selected_squares = [[0.0, 0.0, 0.0] for _ in horizons]
    candidate_counts = [[[0, 0, 0] for _ in range(2)] for _ in horizons]
    selected_counts = [[0, 0, 0] for _ in horizons]
    choice_counts = [0, 0]
    timestamps = episode["timestamps"]
    with torch.no_grad():
        for frame_index, timestamp in enumerate(timestamps.tolist()):
            model.ingest(_packet(episode, frame_index, image_size))
            if model.belief is None:
                continue
            if pool is None:
                pool = HypothesisDynamicsPool(
                    [model.dynamics, ConstantVelocityDynamics(damping=0.05)]
                )
            for horizon_index, horizon in enumerate(horizons):
                future_index = frame_index + round(horizon * frame_rate)
                if future_index >= len(timestamps):
                    continue
                offset = float(timestamps[future_index] - timestamp)
                trajectories = model.predict_hypotheses(pool, [offset])
                target, mask = _future_targets_aligned_to_belief(model, episode, future_index)
                selection = model.assimilate_hypotheses(
                    pool,
                    target,
                    mask,
                    trajectories,
                    uncertainty_aware=False,
                )
                selected = int(selection.selected_index[0])
                choice_counts[selected] += 1
                for candidate_index, trajectory in enumerate(trajectories):
                    residual = trajectory.positions[:, 0] - target
                    for axis in range(3):
                        values = residual[..., axis].masked_select(mask)
                        candidate_squares[horizon_index][candidate_index][axis] += float(
                            values.square().sum().cpu()
                        )
                        candidate_counts[horizon_index][candidate_index][axis] += int(values.numel())
                residual = trajectories[selected].positions[:, 0] - target
                for axis in range(3):
                    values = residual[..., axis].masked_select(mask)
                    selected_squares[horizon_index][axis] += float(values.square().sum().cpu())
                    selected_counts[horizon_index][axis] += int(values.numel())
    return {
        "candidate_rmse_m": {
            str(horizon): [
                _rmse(candidate_squares[index][candidate], candidate_counts[index][candidate])
                for candidate in range(2)
            ]
            for index, horizon in enumerate(horizons)
        },
        "selected_rmse_m": {
            str(horizon): _rmse(selected_squares[index], selected_counts[index])
            for index, horizon in enumerate(horizons)
        },
        "selection_counts": choice_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    config = load_config(args.config, overrides=[f"device.preference={args.device}"])
    device = select_device(config.device.preference).device
    model = OnlineWorldModel.from_config(config, device=device).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=model, map_location=device, expected_config=config)
        model.eval()
    horizons = tuple(float(value) for value in config.evaluation.horizons_seconds)
    episodes = [
        evaluate_episode(
            model,
            generate_episode(config, args.seed + index),
            horizons,
            config.simulator.frame_rate,
            tuple(config.simulator.image_size),
        )
        for index in range(args.episodes)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "device": str(device),
                "episodes": args.episodes,
                "seed_start": args.seed,
                "horizons_seconds": horizons,
                "episode_results": episodes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
