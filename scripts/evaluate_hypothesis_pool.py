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
from world_model.dynamics import (
    BallisticContactDynamics,
    ConstantVelocityDynamics,
    HypothesisDynamicsPool,
)
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
    event_weight: float,
    lifecycle_weight: float,
    position_gate_ratio: float,
    axis_gate_ratio: float,
    event_gate_ratio: float,
    axis_weights: tuple[float, float, float],
    blend_positions: bool,
    temperature: float,
    event_threshold: float,
    uncertainty_aware: bool,
    horizon_decay_scale: float,
    independent_horizons: bool,
    axis_independent: bool,
    axis_independent_axes: tuple[int, ...] | None,
    axis_prior_strength: float,
) -> dict[str, Any]:
    model.reset(batch_size=1)
    pool: HypothesisDynamicsPool | None = None
    horizon_pools: list[HypothesisDynamicsPool] | None = None
    candidate_count = 4
    candidate_squares = [[[0.0, 0.0, 0.0] for _ in range(candidate_count)] for _ in horizons]
    selected_squares = [[0.0, 0.0, 0.0] for _ in horizons]
    candidate_counts = [[[0, 0, 0] for _ in range(candidate_count)] for _ in horizons]
    selected_counts = [[0, 0, 0] for _ in horizons]
    lifecycle_mismatch = [[0 for _ in range(candidate_count)] for _ in horizons]
    identity_covered = [[0 for _ in range(candidate_count)] for _ in horizons]
    selected_lifecycle_mismatch = [0 for _ in horizons]
    selected_identity_covered = [0 for _ in horizons]
    event_counts = [
        [{"tp": 0, "fp": 0, "fn": 0} for _ in range(candidate_count)] for _ in horizons
    ]
    event_probability_histograms = [
        [[0 for _ in range(10)] for _ in range(candidate_count)] for _ in horizons
    ]
    event_probability_positive_histograms = [
        [[0 for _ in range(10)] for _ in range(candidate_count)] for _ in horizons
    ]
    event_probability_negative_histograms = [
        [[0 for _ in range(10)] for _ in range(candidate_count)] for _ in horizons
    ]
    selected_event_counts = [{"tp": 0, "fp": 0, "fn": 0} for _ in horizons]
    uncertainty_sum = [[0.0, 0] for _ in horizons]
    choice_counts = [0 for _ in range(candidate_count)]
    horizon_choice_counts = [[0 for _ in range(candidate_count)] for _ in horizons]
    axis_choice_counts = [
        [[0 for _ in range(candidate_count)] for _ in range(3)] for _ in horizons
    ]
    timestamps = episode["timestamps"]
    # Evaluation never backpropagates or mutates tensors through autograd.
    # Inference mode also disables version-counter bookkeeping, which matters
    # here because every frame runs the learned dynamics for all hypotheses.
    with torch.inference_mode():
        for frame_index, timestamp in enumerate(timestamps.tolist()):
            model.ingest(_packet(episode, frame_index, image_size))
            if model.belief is None:
                continue
            if pool is None:
                candidate_models = [
                    model.dynamics,
                    ConstantVelocityDynamics(damping=0.0),
                    ConstantVelocityDynamics(damping=0.05),
                    BallisticContactDynamics(),
                ]
                pool = HypothesisDynamicsPool(
                    candidate_models,
                    evidence_decay=evidence_decay,
                    temperature=temperature,
                )
                if independent_horizons:
                    horizon_pools = [
                        HypothesisDynamicsPool(
                            candidate_models,
                            evidence_decay=evidence_decay,
                            temperature=temperature,
                        )
                        for _ in horizons
                    ]
            valid_queries: list[tuple[int, int]] = []
            query_offsets: list[float] = []
            for horizon_index, horizon in enumerate(horizons):
                future_index = frame_index + round(horizon * frame_rate)
                if future_index < len(timestamps):
                    valid_queries.append((horizon_index, future_index))
                    query_offsets.append(float(timestamps[future_index] - timestamp))
            if not valid_queries:
                continue
            trajectories = model.predict_hypotheses(pool, query_offsets)
            for query_index, (horizon_index, future_index) in enumerate(valid_queries):
                single_step_trajectories = []
                for trajectory in trajectories:
                    single_step_trajectories.append(
                        trajectory.__class__(
                            timestamps=trajectory.timestamps[:, query_index : query_index + 1],
                            positions=trajectory.positions[:, query_index : query_index + 1],
                            velocities=trajectory.velocities[:, query_index : query_index + 1],
                            orientations=trajectory.orientations[:, query_index : query_index + 1],
                            motion_mode_logits=trajectory.motion_mode_logits[:, query_index : query_index + 1],
                            fast_log_variance=trajectory.fast_log_variance[:, query_index : query_index + 1],
                            active_mask=trajectory.active_mask[:, query_index : query_index + 1],
                            event_logits=(
                                trajectory.event_logits[:, query_index : query_index + 1]
                                if trajectory.event_logits is not None
                                else None
                            ),
                            auxiliary={
                                name: value[:, query_index : query_index + 1]
                                for name, value in trajectory.auxiliary.items()
                            },
                        )
                    )
                target, mask, collision_target = _future_targets_aligned_to_belief(
                    model, episode, frame_index, future_index
                )
                decay_exponent = 1.0 + horizon_decay_scale * max(query_offsets[query_index], 0.0)
                if decay_exponent <= 0:
                    raise ValueError("horizon decay exponent must remain positive")
                evidence_pool = horizon_pools[horizon_index] if horizon_pools is not None else pool
                selection = model.assimilate_hypotheses(
                    evidence_pool,
                    target,
                    mask,
                    single_step_trajectories,
                    target_collision=collision_target,
                    event_weight=event_weight,
                    lifecycle_weight=lifecycle_weight,
                    position_gate_ratio=position_gate_ratio,
                    axis_gate_ratio=axis_gate_ratio,
                    event_gate_ratio=event_gate_ratio,
                    axis_weights=axis_weights,
                    uncertainty_aware=uncertainty_aware,
                    axis_prior_strength=axis_prior_strength,
                    evidence_decay_override=(
                        evidence_decay**decay_exponent
                        if horizon_decay_scale
                        else None
                    ),
                )
                selected = int(selection.selected_index[0])
                choice_counts[selected] += 1
                horizon_choice_counts[horizon_index][selected] += 1
                axis_selected = None
                if axis_independent:
                    if selection.axis_scores is None:
                        raise RuntimeError("axis-independent selection requires axis scores")
                    axis_selected = selection.selected_index[0].repeat(3)
                    independent_axes = (
                        tuple(range(3)) if axis_independent_axes is None else axis_independent_axes
                    )
                    axis_scores = selection.axis_selected_index[0]
                    if axis_scores is None:
                        raise RuntimeError("axis-independent selection requires axis indices")
                    for axis in independent_axes:
                        axis_selected[axis] = axis_scores[axis]
                    for axis, candidate in enumerate(axis_selected.tolist()):
                        axis_choice_counts[horizon_index][axis][candidate] += 1
                if blend_positions:
                    posterior = selection.posterior_weights[0]
                    predicted_positions = torch.stack(
                        [trajectory.positions[0, 0] for trajectory in single_step_trajectories]
                    )
                    blended_position = torch.einsum("h,hnd->nd", posterior, predicted_positions)
                    predicted_variances = torch.stack(
                        [trajectory.fast_log_variance[0, 0, ..., :3].exp() for trajectory in single_step_trajectories]
                    )
                    blended_variance = torch.einsum(
                        "h,hnd->nd",
                        posterior,
                        predicted_variances
                        + (predicted_positions - blended_position.unsqueeze(0)).square(),
                    )
                elif axis_selected is not None:
                    predicted_positions = torch.stack(
                        [trajectory.positions[0, 0] for trajectory in single_step_trajectories]
                    )
                    predicted_variances = torch.stack(
                        [trajectory.fast_log_variance[0, 0, ..., :3].exp() for trajectory in single_step_trajectories]
                    )
                    blended_position = torch.stack(
                        [predicted_positions[int(axis_selected[axis]), :, axis] for axis in range(3)],
                        dim=-1,
                    )
                    blended_variance = torch.stack(
                        [predicted_variances[int(axis_selected[axis]), :, axis] for axis in range(3)],
                        dim=-1,
                    )
                else:
                    blended_position = single_step_trajectories[selected].positions[0, 0]
                    blended_variance = single_step_trajectories[selected].fast_log_variance[0, 0, ..., :3].exp()
                truth = collision_target[:, 0]
                for candidate_index, trajectory in enumerate(single_step_trajectories):
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
                        trajectory.event_logits[:, 0, :, MotionMode.COLLISION].sigmoid() >= event_threshold
                    )
                    event_probability = trajectory.event_logits[:, 0, :, MotionMode.COLLISION].sigmoid()
                    bins = (event_probability.clamp(0.0, 1.0 - 1.0e-7) * 10).to(torch.int64)
                    histogram = torch.bincount(bins.flatten(), minlength=10).tolist()
                    event_probability_histograms[horizon_index][candidate_index] = [
                        previous + current
                        for previous, current in zip(
                            event_probability_histograms[horizon_index][candidate_index], histogram, strict=True
                        )
                    ]
                    for destination, selector in (
                        (event_probability_positive_histograms, truth),
                        (event_probability_negative_histograms, ~truth),
                    ):
                        selected_bins = bins.masked_select(selector)
                        selected_histogram = torch.bincount(selected_bins, minlength=10).tolist()
                        destination[horizon_index][candidate_index] = [
                            previous + current
                            for previous, current in zip(
                                destination[horizon_index][candidate_index],
                                selected_histogram,
                                strict=True,
                            )
                        ]
                    event_counts[horizon_index][candidate_index]["tp"] += int(
                        (event_prediction & truth).sum().cpu()
                    )
                    event_counts[horizon_index][candidate_index]["fp"] += int(
                        (event_prediction & ~truth).sum().cpu()
                    )
                    event_counts[horizon_index][candidate_index]["fn"] += int(
                        ((~event_prediction) & truth).sum().cpu()
                    )
                residual = blended_position.unsqueeze(0) - target
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
                    single_step_trajectories[selected].event_logits[:, 0, :, MotionMode.COLLISION].sigmoid()
                    >= event_threshold
                )
                truth = collision_target[:, 0]
                selected_event_counts[horizon_index]["tp"] += int((selected_event & truth).sum().cpu())
                selected_event_counts[horizon_index]["fp"] += int((selected_event & ~truth).sum().cpu())
                selected_event_counts[horizon_index]["fn"] += int(
                    ((~selected_event) & truth).sum().cpu()
                )
                active_variance = blended_variance.masked_select(mask[0, 0].unsqueeze(-1))
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
                for candidate in range(candidate_count)
            ]
            for index, horizon in enumerate(horizons)
        },
        "selected_rmse_m": {
            str(horizon): _rmse(selected_squares[index], selected_counts[index])
            for index, horizon in enumerate(horizons)
        },
        "selection_counts": choice_counts,
        "selection_counts_by_horizon": {
            str(horizon): counts for horizon, counts in zip(horizons, horizon_choice_counts, strict=True)
        },
        "axis_selection_counts_by_horizon": {
            str(horizon): counts for horizon, counts in zip(horizons, axis_choice_counts, strict=True)
        },
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
        "event_probability_histograms": {
            str(horizon): histograms
            for horizon, histograms in zip(horizons, event_probability_histograms, strict=True)
        },
        "event_probability_positive_histograms": {
            str(horizon): histograms
            for horizon, histograms in zip(
                horizons, event_probability_positive_histograms, strict=True
            )
        },
        "event_probability_negative_histograms": {
            str(horizon): histograms
            for horizon, histograms in zip(
                horizons, event_probability_negative_histograms, strict=True
            )
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
    parser.add_argument("--event-weight", type=float, default=0.0)
    parser.add_argument("--lifecycle-weight", type=float, default=0.0)
    parser.add_argument("--position-gate-ratio", type=float, default=0.0)
    parser.add_argument("--axis-gate-ratio", type=float, default=0.0)
    parser.add_argument("--event-gate-ratio", type=float, default=0.0)
    parser.add_argument("--axis-weights", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--blend-positions", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--event-threshold", type=float, default=0.5)
    parser.add_argument("--uncertainty-aware", action="store_true")
    parser.add_argument("--horizon-decay-scale", type=float, default=0.0)
    parser.add_argument("--independent-horizons", action="store_true")
    axis_group = parser.add_mutually_exclusive_group()
    axis_group.add_argument(
        "--axis-independent",
        action="store_true",
        default=None,
        help="select delayed-evidence hypotheses independently for x/y/z",
    )
    axis_group.add_argument(
        "--no-axis-independent",
        dest="axis_independent",
        action="store_false",
        help="override config and keep joint hypothesis selection",
    )
    parser.add_argument(
        "--axis-independent-axes",
        type=int,
        nargs="+",
        choices=(0, 1, 2),
        help="subset of coordinates for --axis-independent (default: all)",
    )
    parser.add_argument("--axis-prior-strength", type=float, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not 0.0 < args.evidence_decay <= 1.0:
        raise ValueError("--evidence-decay must lie in (0,1]")
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        raise ValueError("--temperature must be finite and positive")
    if not math.isfinite(args.horizon_decay_scale):
        raise ValueError("--horizon-decay-scale must be finite")
    if not 0.0 <= args.event_threshold <= 1.0 or not math.isfinite(args.event_threshold):
        raise ValueError("--event-threshold must lie in [0,1]")
    if (
        args.event_weight < 0
        or args.lifecycle_weight < 0
        or args.position_gate_ratio < 0
        or args.axis_gate_ratio < 0
        or args.event_gate_ratio < 0
    ):
        raise ValueError("score weights must be nonnegative")
    if any((weight < 0 or not math.isfinite(weight)) for weight in args.axis_weights) or not any(
        weight > 0 for weight in args.axis_weights
    ):
        raise ValueError("--axis-weights must contain finite nonnegative values and one positive value")
    config = load_config(args.config, overrides=[f"device.preference={args.device}"])
    axis_independent = (
        config.evaluation.hypothesis_axis_independent
        if args.axis_independent is None
        else args.axis_independent
    )
    axis_independent_axes = (
        tuple(config.evaluation.hypothesis_axis_independent_axes)
        if args.axis_independent_axes is None
        else tuple(args.axis_independent_axes)
    )
    axis_prior_strength = (
        config.evaluation.hypothesis_axis_prior_strength
        if args.axis_prior_strength is None
        else args.axis_prior_strength
    )
    if not 0.0 <= axis_prior_strength <= 1.0 or not math.isfinite(axis_prior_strength):
        raise ValueError("axis prior strength must lie in [0,1]")
    if args.axis_independent_axes and not axis_independent:
        raise ValueError("--axis-independent-axes requires axis-independent selection")
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
            args.event_weight,
            args.lifecycle_weight,
            args.position_gate_ratio,
            args.axis_gate_ratio,
            args.event_gate_ratio,
            tuple(args.axis_weights),
            args.blend_positions,
            args.temperature,
            args.event_threshold,
            args.uncertainty_aware,
            args.horizon_decay_scale,
            args.independent_horizons,
            axis_independent,
            axis_independent_axes if axis_independent else None,
            axis_prior_strength,
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
                "event_weight": args.event_weight,
                "lifecycle_weight": args.lifecycle_weight,
                "position_gate_ratio": args.position_gate_ratio,
                "axis_gate_ratio": args.axis_gate_ratio,
                "event_gate_ratio": args.event_gate_ratio,
                "axis_weights": list(args.axis_weights),
                "blend_positions": args.blend_positions,
                "temperature": args.temperature,
                "event_threshold": args.event_threshold,
                "uncertainty_aware": args.uncertainty_aware,
                "horizon_decay_scale": args.horizon_decay_scale,
                "independent_horizons": args.independent_horizons,
                "axis_independent": axis_independent,
                "axis_independent_axes": list(axis_independent_axes) if axis_independent else None,
                "axis_prior_strength": axis_prior_strength,
                "candidate_names": [
                    "learned",
                    "constant_velocity_damped_0.0",
                    "constant_velocity_damped_0.05",
                    "ballistic_contact",
                ],
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
