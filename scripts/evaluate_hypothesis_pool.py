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

from world_model.belief import MotionMode
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
    reference_index: int,
    future_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map simulator supervision onto persistent runtime object identities."""

    assert model.belief is not None
    objects = model.belief.objects
    positions = episode["objects"]["position"][future_index]
    active = episode["objects"]["active"][future_index]
    ids = episode["objects"]["id"][future_index]
    collision = episode["events"]["collision"][future_index]
    reference_positions = episode["objects"]["position"][reference_index]
    reference_active = episode["objects"]["active"][reference_index]
    belief_positions = objects.position[0].detach().cpu()
    belief_active = objects.active[0].detach().cpu()
    active_sources = torch.where(reference_active)[0].tolist()
    active_slots = torch.where(belief_active)[0].tolist()
    source_to_slot: dict[int, int] = {}
    remaining_slots = set(active_slots)
    for source in active_sources:
        if not remaining_slots:
            break
        candidates = sorted(
            remaining_slots,
            key=lambda slot: float(
                torch.linalg.vector_norm(
                    belief_positions[slot].cpu() - reference_positions[source].cpu()
                )
            ),
        )
        slot = candidates[0]
        source_to_slot[int(episode["objects"]["id"][reference_index, source].item())] = slot
        remaining_slots.remove(slot)
    target = objects.position.new_zeros(1, 1, objects.max_objects, 3)
    mask = torch.zeros(1, 1, objects.max_objects, dtype=torch.bool, device=objects.active.device)
    collision_target = torch.zeros_like(mask)
    for source, object_id in enumerate(ids.detach().cpu().tolist()):
        slot = source_to_slot.get(int(object_id))
        if slot is None:
            continue
        target[0, 0, slot] = positions[source].to(device=target.device, dtype=target.dtype)
        mask[0, 0, slot] = bool(active[source])
        collision_target[0, 0, slot] = bool(collision[source])
    return target, mask, collision_target


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
    evidence_decay: float,
) -> dict[str, Any]:
    model.reset(batch_size=1)
    pool: HypothesisDynamicsPool | None = None
    candidate_squares = [[[0.0, 0.0, 0.0] for _ in range(2)] for _ in horizons]
    selected_squares = [[0.0, 0.0, 0.0] for _ in horizons]
    candidate_counts = [[[0, 0, 0] for _ in range(2)] for _ in horizons]
    selected_counts = [[0, 0, 0] for _ in horizons]
    lifecycle_mismatch = [[0, 0] for _ in horizons]
    identity_covered = [[0, 0] for _ in horizons]
    selected_lifecycle_mismatch = [0 for _ in horizons]
    selected_identity_covered = [0 for _ in horizons]
    event_counts = [
        [{"tp": 0, "fp": 0, "fn": 0} for _ in range(2)] for _ in horizons
    ]
    selected_event_counts = [{"tp": 0, "fp": 0, "fn": 0} for _ in horizons]
    uncertainty_sum = [[0.0, 0] for _ in horizons]
    choice_counts = [0, 0]
    timestamps = episode["timestamps"]
    with torch.no_grad():
        for frame_index, timestamp in enumerate(timestamps.tolist()):
            model.ingest(_packet(episode, frame_index, image_size))
            if model.belief is None:
                continue
            if pool is None:
                pool = HypothesisDynamicsPool(
                    [model.dynamics, ConstantVelocityDynamics(damping=0.05)],
                    evidence_decay=evidence_decay,
                )
            for horizon_index, horizon in enumerate(horizons):
                future_index = frame_index + round(horizon * frame_rate)
                if future_index >= len(timestamps):
                    continue
                offset = float(timestamps[future_index] - timestamp)
                trajectories = model.predict_hypotheses(pool, [offset])
                target, mask, collision_target = _future_targets_aligned_to_belief(
                    model, episode, frame_index, future_index
                )
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
                    active_prediction = trajectory.active_mask[:, 0]
                    lifecycle_mismatch[horizon_index][candidate_index] += int(
                        (active_prediction != mask[:, 0]).sum().cpu()
                    )
                    identity_covered[horizon_index][candidate_index] += int(
                        (active_prediction & mask[:, 0]).sum().cpu()
                    )
                    event_prediction = (
                        trajectory.event_logits[:, 0, :, MotionMode.COLLISION].sigmoid() >= 0.5
                    )
                    truth = collision_target[:, 0]
                    event_counts[horizon_index][candidate_index]["tp"] += int(
                        (event_prediction & truth).sum().cpu()
                    )
                    event_counts[horizon_index][candidate_index]["fp"] += int(
                        (event_prediction & ~truth).sum().cpu()
                    )
                    event_counts[horizon_index][candidate_index]["fn"] += int(
                        ((~event_prediction) & truth).sum().cpu()
                    )
                residual = trajectories[selected].positions[:, 0] - target
                for axis in range(3):
                    values = residual[..., axis].masked_select(mask)
                    selected_squares[horizon_index][axis] += float(values.square().sum().cpu())
                    selected_counts[horizon_index][axis] += int(values.numel())
                selected_active = trajectories[selected].active_mask[:, 0]
                selected_lifecycle_mismatch[horizon_index] += int(
                    (selected_active != mask[:, 0]).sum().cpu()
                )
                selected_identity_covered[horizon_index] += int(
                    (selected_active & mask[:, 0]).sum().cpu()
                )
                selected_event = (
                    trajectories[selected].event_logits[:, 0, :, MotionMode.COLLISION].sigmoid()
                    >= 0.5
                )
                truth = collision_target[:, 0]
                selected_event_counts[horizon_index]["tp"] += int((selected_event & truth).sum().cpu())
                selected_event_counts[horizon_index]["fp"] += int((selected_event & ~truth).sum().cpu())
                selected_event_counts[horizon_index]["fn"] += int(
                    ((~selected_event) & truth).sum().cpu()
                )
                selected_variance = trajectories[selected].fast_log_variance[:, 0, ..., :3].exp()
                active_variance = selected_variance.masked_select(mask[:, 0].unsqueeze(-1))
                if active_variance.numel():
                    uncertainty_sum[horizon_index][0] += float(active_variance.mean().sqrt().cpu())
                    uncertainty_sum[horizon_index][1] += 1

    def event_metrics(counts: dict[str, int]) -> dict[str, float]:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "collision_precision": precision,
            "collision_recall": recall,
            "collision_f1": (2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0,
            "collision_true_positive": float(tp),
            "collision_false_positive": float(fp),
            "collision_false_negative": float(fn),
        }
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
        "candidate_lifecycle_mismatch": {
            str(horizon): lifecycle_mismatch[index] for index, horizon in enumerate(horizons)
        },
        "candidate_identity_coverage": {
            str(horizon): identity_covered[index] for index, horizon in enumerate(horizons)
        },
        "selected_lifecycle_mismatch": {
            str(horizon): selected_lifecycle_mismatch[index]
            for index, horizon in enumerate(horizons)
        },
        "selected_identity_coverage": {
            str(horizon): selected_identity_covered[index]
            for index, horizon in enumerate(horizons)
        },
        "candidate_event_metrics": {
            str(horizon): [event_metrics(item) for item in event_counts[index]]
            for index, horizon in enumerate(horizons)
        },
        "selected_event_metrics": {
            str(horizon): event_metrics(selected_event_counts[index])
            for index, horizon in enumerate(horizons)
        },
        "selected_mean_position_std_m": {
            str(horizon): (total / count if count else None)
            for horizon, (total, count) in zip(horizons, uncertainty_sum, strict=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-decay", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not 0.0 < args.evidence_decay <= 1.0:
        raise ValueError("--evidence-decay must lie in (0,1]")
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
            args.evidence_decay,
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
                "evidence_decay": args.evidence_decay,
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
