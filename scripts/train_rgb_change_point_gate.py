#!/usr/bin/env python3
"""Fit a causal RGB trajectory gate and write a weights-identical checkpoint."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from world_model.datasets import SyntheticSphereDataset, collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.training.change_point_gate import (
    fit_linear_change_point_gate,
    fit_mlp_change_point_gate,
    fit_mlp_lateral_velocity_intervention,
    fit_mlp_outgoing_velocity_proposal,
)
from world_model.training.checkpointing import load_checkpoint
from world_model.training.loop import (
    PersistentTargetMatcher,
    make_rgb_packet,
    move_batch_to_device,
)
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import load_config, save_resolved_config
from world_model.utils.device import select_device
from world_model.utils.seeds import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/rgb-change-point-gate")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--train-seed-offset", type=int, default=0)
    parser.add_argument("--validation-seed-offset", type=int, default=0)
    parser.add_argument("--minimum-velocity-jump", type=float, default=0.75)
    parser.add_argument("--minimum-precision", type=float, default=0.8)
    parser.add_argument("--fit-steps", type=int, default=800)
    parser.add_argument("--gate-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-features", type=int, default=12)
    parser.add_argument("--train-cache")
    parser.add_argument("--validation-cache")
    parser.add_argument("--fit-outgoing-proposal", action="store_true")
    parser.add_argument("--proposal-hidden-features", type=int, default=8)
    parser.add_argument("--proposal-fit-steps", type=int, default=2000)
    parser.add_argument("--proposal-gate-focus-weight", type=int, default=20)
    parser.add_argument("--proposal-maximum-delta", type=float, default=3.0)
    parser.add_argument("--fit-lateral-intervention", action="store_true")
    parser.add_argument("--lateral-hidden-features", type=int, default=12)
    parser.add_argument("--lateral-fit-steps", type=int, default=3000)
    parser.add_argument("--lateral-learning-rate", type=float, default=0.01)
    parser.add_argument("--lateral-weight-decay", type=float, default=5.0e-3)
    parser.add_argument("--lateral-gain-sparsity", type=float, default=0.01)
    parser.add_argument("--lateral-variance-floor", type=float, default=0.04)
    parser.add_argument("--lateral-variance-ceiling", type=float, default=25.0)
    parser.add_argument("--lateral-gain-power", type=float, default=2.0)
    parser.add_argument("--lateral-maximum-delta", type=float, default=5.0)
    parser.add_argument("--fit-gravity-intervention", action="store_true")
    parser.add_argument("--gravity-hidden-features", type=int, default=4)
    parser.add_argument("--gravity-fit-steps", type=int, default=2000)
    parser.add_argument("--gravity-learning-rate", type=float, default=0.005)
    parser.add_argument("--gravity-weight-decay", type=float, default=0.2)
    parser.add_argument("--gravity-gain-sparsity", type=float, default=0.1)
    parser.add_argument("--gravity-variance-floor", type=float, default=0.04)
    parser.add_argument("--gravity-variance-ceiling", type=float, default=25.0)
    parser.add_argument("--gravity-gain-power", type=float, default=2.0)
    parser.add_argument("--gravity-maximum-delta", type=float, default=5.0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


@torch.no_grad()
def collect_examples(
    model: OnlineWorldModel,
    config: Any,
    *,
    split: str,
    episodes: int,
    seed_offset: int,
    device: torch.device,
    minimum_velocity_jump: float,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episode count must be positive")
    dataset = SyntheticSphereDataset(
        config.simulator,
        split=split,
        num_episodes=episodes,
        seed_offset=seed_offset,
        memory_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_episodes,
    )
    features: list[Tensor] = []
    targets: list[Tensor] = []
    outgoing_prior: list[Tensor] = []
    outgoing_target: list[Tensor] = []
    lateral_features: list[Tensor] = []
    lateral_prior: list[Tensor] = []
    lateral_target: list[Tensor] = []
    lateral_prior_variance: list[Tensor] = []
    lateral_confidence: list[Tensor] = []
    gravity_features: list[Tensor] = []
    gravity_prior: list[Tensor] = []
    gravity_target: list[Tensor] = []
    gravity_prior_variance: list[Tensor] = []
    gravity_confidence: list[Tensor] = []
    eligible_count = 0
    collision_count = 0
    jump_count = 0
    gravity = torch.tensor(config.simulator.gravity, device=device, dtype=torch.float32)
    gravity_axis = F.normalize(gravity, dim=-1)
    model.eval()
    for episode_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch_to_device(raw_batch, device)
        model.reset(batch_size=1)
        matcher = PersistentTargetMatcher()
        total_frames = int(batch["rgb"].shape[1])
        for frame_index in range(total_frames):
            packet = make_rgb_packet(batch, frame_index)
            belief = model.ingest(packet)
            indices, matched = matcher.match(
                belief,
                batch["objects"]["position"][:, frame_index],
                batch["objects"]["active"][:, frame_index].bool(),
            )
            measurements = model.last_measurements
            if measurements is None or abs(float(measurements.timestamp) - packet.timestamp) > 1e-7:
                continue
            gate_features = measurements.auxiliary.get("trajectory_change_point_features")
            feature_valid = measurements.auxiliary.get("trajectory_change_point_feature_valid_mask")
            feature_timestamps = measurements.auxiliary.get(
                "trajectory_change_point_feature_timestamps"
            )
            current_lateral_features = measurements.auxiliary.get(
                "trajectory_lateral_intervention_features"
            )
            lateral_feature_valid = measurements.auxiliary.get(
                "trajectory_lateral_intervention_feature_valid_mask"
            )
            direct_prior_velocity = measurements.auxiliary.get("trajectory_direct_prior_velocity")
            direct_prior_velocity_log_variance = measurements.auxiliary.get(
                "trajectory_direct_prior_velocity_log_variance"
            )
            direct_confidence = measurements.auxiliary.get("trajectory_direct_confidence")
            camera_lateral_axis = measurements.auxiliary.get("trajectory_camera_lateral_axis")
            current_gravity_features = measurements.auxiliary.get(
                "trajectory_gravity_intervention_features"
            )
            gravity_feature_valid = measurements.auxiliary.get(
                "trajectory_gravity_intervention_feature_valid_mask"
            )
            gravity_axis = measurements.auxiliary.get("trajectory_gravity_axis")
            if (
                not isinstance(gate_features, Tensor)
                or not isinstance(feature_valid, Tensor)
                or not isinstance(feature_timestamps, Tensor)
                or not isinstance(current_lateral_features, Tensor)
                or not isinstance(lateral_feature_valid, Tensor)
                or not isinstance(direct_prior_velocity, Tensor)
                or not isinstance(direct_prior_velocity_log_variance, Tensor)
                or not isinstance(direct_confidence, Tensor)
                or not isinstance(camera_lateral_axis, Tensor)
                or not isinstance(current_gravity_features, Tensor)
                or not isinstance(gravity_feature_valid, Tensor)
                or not isinstance(gravity_axis, Tensor)
            ):
                raise RuntimeError(
                    "RGB runtime did not expose aligned intervention training features"
                )
            valid = feature_valid.bool() & matched & belief.objects.active
            lateral_valid = lateral_feature_valid.bool() & matched & belief.objects.active
            gravity_valid = gravity_feature_valid.bool() & matched & belief.objects.active
            if (valid | lateral_valid | gravity_valid).any():
                aligned_target = torch.zeros_like(valid)
                aligned_collision = torch.zeros_like(valid)
                aligned_outgoing_prior = torch.zeros_like(
                    valid,
                    dtype=belief.objects.position.dtype,
                )
                aligned_outgoing_target = torch.zeros_like(
                    aligned_outgoing_prior,
                )
                aligned_lateral_prior = (
                    direct_prior_velocity * camera_lateral_axis[:, None, :]
                ).sum(dim=-1)
                aligned_lateral_target = torch.zeros_like(aligned_lateral_prior)
                aligned_gravity_prior = (direct_prior_velocity * gravity_axis[:, None, :]).sum(
                    dim=-1
                )
                aligned_gravity_target = torch.zeros_like(aligned_gravity_prior)
                aligned_lateral_prior_variance = (
                    direct_prior_velocity_log_variance.exp()
                    * camera_lateral_axis[:, None, :].square()
                ).sum(dim=-1)
                # Gravity features use the same exact timestamp triplet.
                # Include their rows even when another experimental path has a
                # stricter validity mask.
                for belief_slot_tensor in torch.nonzero(
                    valid[0] | lateral_valid[0] | gravity_valid[0],
                    as_tuple=False,
                ).flatten():
                    belief_slot = int(belief_slot_tensor)
                    target_slot = int(indices[0, belief_slot])
                    frame_times = batch["timestamps"][0]
                    window_indices = torch.stack(
                        [
                            (frame_times - timestamp).abs().argmin()
                            for timestamp in feature_timestamps[0, belief_slot]
                        ]
                    )
                    start_index = int(window_indices[0])
                    end_index = int(window_indices[-1])
                    if end_index <= start_index:
                        continue
                    has_collision = bool(
                        batch["events"]["collision"][
                            0,
                            start_index + 1 : end_index + 1,
                            target_slot,
                        ]
                        .bool()
                        .any()
                    )
                    dt = frame_times[end_index] - frame_times[start_index]
                    velocity_jump = (
                        batch["objects"]["velocity"][0, end_index, target_slot]
                        - batch["objects"]["velocity"][0, start_index, target_slot]
                        - gravity * dt
                    )
                    observable_jump = (velocity_jump * gravity_axis).sum().abs()
                    aligned_collision[0, belief_slot] = has_collision
                    aligned_target[0, belief_slot] = has_collision and bool(
                        observable_jump >= minimum_velocity_jump
                    )
                    aligned_outgoing_prior[0, belief_slot] = (
                        direct_prior_velocity[0, belief_slot] * gravity_axis
                    ).sum()
                    aligned_outgoing_target[0, belief_slot] = (
                        batch["objects"]["velocity"][0, end_index, target_slot] * gravity_axis
                    ).sum()
                    aligned_lateral_target[0, belief_slot] = (
                        batch["objects"]["velocity"][0, end_index, target_slot]
                        * camera_lateral_axis[0]
                    ).sum()
                    aligned_gravity_target[0, belief_slot] = (
                        batch["objects"]["velocity"][0, end_index, target_slot] * gravity_axis[0]
                    ).sum()
                if valid.any():
                    features.append(
                        gate_features.masked_select(valid.unsqueeze(-1)).reshape(
                            -1,
                            9,
                        )
                    )
                    targets.append(aligned_target.masked_select(valid))
                    outgoing_prior.append(aligned_outgoing_prior.masked_select(valid))
                    outgoing_target.append(aligned_outgoing_target.masked_select(valid))
                if lateral_valid.any():
                    lateral_features.append(
                        current_lateral_features.masked_select(lateral_valid.unsqueeze(-1)).reshape(
                            -1, 19
                        )
                    )
                    lateral_prior.append(aligned_lateral_prior.masked_select(lateral_valid))
                    lateral_target.append(aligned_lateral_target.masked_select(lateral_valid))
                    lateral_prior_variance.append(
                        aligned_lateral_prior_variance.masked_select(lateral_valid)
                    )
                    lateral_confidence.append(direct_confidence.masked_select(lateral_valid))
                if gravity_valid.any():
                    gravity_features.append(
                        current_gravity_features.masked_select(gravity_valid.unsqueeze(-1)).reshape(
                            -1, 21
                        )
                    )
                    gravity_prior.append(aligned_gravity_prior.masked_select(gravity_valid))
                    gravity_target.append(aligned_gravity_target.masked_select(gravity_valid))
                    gravity_prior_variance.append(
                        (
                            direct_prior_velocity_log_variance.exp()
                            * gravity_axis[:, None, :].square()
                        )
                        .sum(dim=-1)
                        .masked_select(gravity_valid)
                    )
                    gravity_confidence.append(direct_confidence.masked_select(gravity_valid))
                eligible_count += int(valid.sum().cpu())
                collision_count += int(aligned_collision.masked_select(valid).sum().cpu())
                jump_count += int(aligned_target.masked_select(valid).sum().cpu())
        print(
            f"[{split}] episode {episode_index}/{episodes}: "
            f"eligible={eligible_count} positive={jump_count}",
            flush=True,
        )
    if not features:
        raise RuntimeError("no eligible RGB trajectory windows were collected")
    concatenated_prior = torch.cat(outgoing_prior).cpu()
    concatenated_target = torch.cat(outgoing_target).cpu()
    examples = {
        "features": torch.cat(features).cpu(),
        "targets": torch.cat(targets).cpu(),
        "outgoing_prior": concatenated_prior,
        "outgoing_target": concatenated_target,
        "outgoing_delta": concatenated_target - concatenated_prior,
        "collection": {
            "episodes": float(episodes),
            "eligible_windows": float(eligible_count),
            "collision_windows": float(collision_count),
            "observable_jump_windows": float(jump_count),
        },
    }
    if lateral_features:
        concatenated_lateral_prior = torch.cat(lateral_prior).cpu()
        concatenated_lateral_target = torch.cat(lateral_target).cpu()
        examples.update(
            {
                "lateral_features": torch.cat(lateral_features).cpu(),
                "lateral_prior": concatenated_lateral_prior,
                "lateral_target": concatenated_lateral_target,
                "lateral_target_delta": (concatenated_lateral_target - concatenated_lateral_prior),
                "lateral_prior_variance": torch.cat(lateral_prior_variance).cpu(),
                "lateral_confidence": torch.cat(lateral_confidence).cpu(),
            }
        )
    if gravity_features:
        concatenated_gravity_prior = torch.cat(gravity_prior).cpu()
        concatenated_gravity_target = torch.cat(gravity_target).cpu()
        examples.update(
            {
                "gravity_features": torch.cat(gravity_features).cpu(),
                "gravity_prior": concatenated_gravity_prior,
                "gravity_target": concatenated_gravity_target,
                "gravity_target_delta": (concatenated_gravity_target - concatenated_gravity_prior),
                "gravity_prior_variance": torch.cat(gravity_prior_variance).cpu(),
                "gravity_confidence": torch.cat(gravity_confidence).cpu(),
            }
        )
    return examples


def main() -> int:
    args = parse_args()
    if args.minimum_velocity_jump <= 0:
        raise ValueError("--minimum-velocity-jump must be positive")
    overrides = list(args.set)
    overrides.append(f"device.preference={args.device}")
    config = load_config(args.config, overrides=overrides)
    device_info = select_device(config.device.preference)
    seed_everything(config.project.seed, deterministic=config.project.deterministic)
    output = timestamped_artifact_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = OnlineWorldModel.from_config(config, device=device_info.device)
    payload = load_checkpoint(
        args.checkpoint,
        model=model,
        map_location="cpu",
        expected_config=config,
    )
    if args.train_cache:
        train_examples = torch.load(
            args.train_cache,
            map_location="cpu",
            weights_only=False,
        )
    else:
        train_examples = collect_examples(
            model,
            config,
            split="train",
            episodes=args.train_episodes,
            seed_offset=args.train_seed_offset,
            device=device_info.device,
            minimum_velocity_jump=args.minimum_velocity_jump,
        )
    train_features = train_examples["features"]
    train_targets = train_examples["targets"]
    train_collection = train_examples["collection"]
    torch.save(train_examples, output / "train_features.pt")
    if args.validation_cache:
        validation_examples = torch.load(
            args.validation_cache,
            map_location="cpu",
            weights_only=False,
        )
    else:
        validation_examples = collect_examples(
            model,
            config,
            split="validation",
            episodes=args.validation_episodes,
            seed_offset=args.validation_seed_offset,
            device=device_info.device,
            minimum_velocity_jump=args.minimum_velocity_jump,
        )
    validation_features = validation_examples["features"]
    validation_targets = validation_examples["targets"]
    validation_collection = validation_examples["collection"]
    torch.save(validation_examples, output / "validation_features.pt")
    if args.gate_type == "mlp":
        gate, metrics = fit_mlp_change_point_gate(
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            hidden_features=args.hidden_features,
            steps=args.fit_steps,
            minimum_precision=args.minimum_precision,
            seed=config.project.seed,
        )
        gate_updates = {
            "temporal_velocity_change_point_gate": "mlp",
            "temporal_velocity_change_point_mlp_hidden_weights": gate.hidden_weights,
            "temporal_velocity_change_point_mlp_hidden_bias": gate.hidden_bias,
            "temporal_velocity_change_point_mlp_output_weights": gate.output_weights,
            "temporal_velocity_change_point_mlp_output_bias": gate.output_bias,
        }
    else:
        gate, metrics = fit_linear_change_point_gate(
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            steps=args.fit_steps,
            minimum_precision=args.minimum_precision,
        )
        gate_updates = {
            "temporal_velocity_change_point_gate": "linear",
            "temporal_velocity_change_point_linear_weights": gate.weights,
            "temporal_velocity_change_point_linear_bias": gate.bias,
        }

    proposal_metrics: dict[str, float] = {}
    proposal_updates: dict[str, Any] = {}
    if args.fit_outgoing_proposal:
        missing_proposal_fields = {
            "outgoing_prior",
            "outgoing_delta",
        } - set(train_examples)
        missing_proposal_fields |= {
            "outgoing_prior",
            "outgoing_delta",
        } - set(validation_examples)
        if missing_proposal_fields:
            raise ValueError(
                "cached features do not contain outgoing targets: "
                + ", ".join(sorted(missing_proposal_fields))
            )
        train_gate_probability = gate.logits(train_features).sigmoid()
        validation_gate_probability = gate.logits(validation_features).sigmoid()
        train_proposal_features = torch.cat(
            (
                train_features,
                train_examples["outgoing_prior"].unsqueeze(-1) / 5.0,
                train_gate_probability.unsqueeze(-1),
            ),
            dim=-1,
        )
        validation_proposal_features = torch.cat(
            (
                validation_features,
                validation_examples["outgoing_prior"].unsqueeze(-1) / 5.0,
                validation_gate_probability.unsqueeze(-1),
            ),
            dim=-1,
        )
        base_train_proposal_features = train_proposal_features
        train_proposal_target = torch.where(
            train_targets.bool(),
            train_examples["outgoing_delta"],
            torch.zeros_like(train_examples["outgoing_delta"]),
        )
        validation_proposal_target = torch.where(
            validation_targets.bool(),
            validation_examples["outgoing_delta"],
            torch.zeros_like(validation_examples["outgoing_delta"]),
        )
        train_gate_selected = train_gate_probability >= gate.probability_threshold
        if args.proposal_gate_focus_weight > 1 and train_gate_selected.any():
            repeat_count = args.proposal_gate_focus_weight - 1
            train_proposal_features = torch.cat(
                (
                    train_proposal_features,
                    train_proposal_features[train_gate_selected].repeat(
                        repeat_count,
                        1,
                    ),
                ),
                dim=0,
            )
            train_proposal_target = torch.cat(
                (
                    train_proposal_target,
                    train_proposal_target[train_gate_selected].repeat(
                        repeat_count,
                    ),
                ),
                dim=0,
            )
        proposal, proposal_metrics = fit_mlp_outgoing_velocity_proposal(
            train_proposal_features,
            train_proposal_target,
            validation_proposal_features,
            validation_proposal_target,
            hidden_features=args.proposal_hidden_features,
            steps=args.proposal_fit_steps,
            maximum_delta=args.proposal_maximum_delta,
            seed=config.project.seed,
        )
        validation_selected = validation_gate_probability >= gate.probability_threshold
        if validation_selected.any():
            selected_target = validation_proposal_target[validation_selected]
            selected_residual = (
                proposal.delta(
                    validation_proposal_features[validation_selected],
                )
                - selected_target
            )
            proposal_metrics.update(
                {
                    "validation_gate_selected_examples": float(validation_selected.sum()),
                    "validation_gate_selected_prior_mae_mps": float(selected_target.abs().mean()),
                    "validation_gate_selected_proposal_mae_mps": float(
                        selected_residual.abs().mean()
                    ),
                    "validation_gate_selected_prior_rmse_mps": float(
                        selected_target.square().mean().sqrt()
                    ),
                    "validation_gate_selected_proposal_rmse_mps": float(
                        selected_residual.square().mean().sqrt()
                    ),
                }
            )
        proposal_metrics["gate_focus_weight"] = float(args.proposal_gate_focus_weight)
        proposal_metrics["maximum_delta_mps"] = float(args.proposal_maximum_delta)
        proposal_metrics["unreplicated_train_examples"] = float(
            base_train_proposal_features.shape[0]
        )
        proposal_updates = {
            "temporal_velocity_outgoing_proposal_enabled": True,
            "temporal_velocity_outgoing_proposal_hidden_weights": (proposal.hidden_weights),
            "temporal_velocity_outgoing_proposal_hidden_bias": proposal.hidden_bias,
            "temporal_velocity_outgoing_proposal_output_weights": (proposal.output_weights),
            "temporal_velocity_outgoing_proposal_output_bias": proposal.output_bias,
            "temporal_velocity_outgoing_proposal_variance": proposal.variance,
            "temporal_velocity_outgoing_proposal_maximum_delta": (proposal.maximum_delta),
        }

    lateral_metrics: dict[str, float] = {}
    lateral_updates: dict[str, Any] = {}
    if args.fit_lateral_intervention:
        required_lateral_fields = {
            "lateral_features",
            "lateral_target_delta",
            "lateral_prior_variance",
            "lateral_confidence",
        }
        missing_lateral_fields = required_lateral_fields - set(train_examples)
        missing_lateral_fields |= required_lateral_fields - set(validation_examples)
        if missing_lateral_fields:
            raise ValueError(
                "cached features do not contain lateral intervention targets: "
                + ", ".join(sorted(missing_lateral_fields))
            )
        lateral, lateral_metrics = fit_mlp_lateral_velocity_intervention(
            train_examples["lateral_features"],
            train_examples["lateral_target_delta"],
            train_examples["lateral_prior_variance"],
            train_examples["lateral_confidence"],
            validation_examples["lateral_features"],
            validation_examples["lateral_target_delta"],
            validation_examples["lateral_prior_variance"],
            validation_examples["lateral_confidence"],
            hidden_features=args.lateral_hidden_features,
            steps=args.lateral_fit_steps,
            learning_rate=args.lateral_learning_rate,
            weight_decay=args.lateral_weight_decay,
            gain_sparsity=args.lateral_gain_sparsity,
            variance_floor=args.lateral_variance_floor,
            variance_ceiling=args.lateral_variance_ceiling,
            gain_power=args.lateral_gain_power,
            maximum_delta=args.lateral_maximum_delta,
            robust_clip_norm=config.model.filter.robust_clip,
            seed=config.project.seed,
        )
        lateral_metrics["gain_sparsity"] = args.lateral_gain_sparsity
        lateral_metrics["learning_rate"] = args.lateral_learning_rate
        lateral_metrics["weight_decay"] = args.lateral_weight_decay
        lateral_updates = {
            "temporal_velocity_lateral_intervention_enabled": True,
            "temporal_velocity_lateral_intervention_hidden_weights": (lateral.hidden_weights),
            "temporal_velocity_lateral_intervention_hidden_bias": (lateral.hidden_bias),
            "temporal_velocity_lateral_intervention_output_weights": (lateral.output_weights),
            "temporal_velocity_lateral_intervention_output_bias": (lateral.output_bias),
            "temporal_velocity_lateral_intervention_variance_floor": (lateral.variance_floor),
            "temporal_velocity_lateral_intervention_variance_ceiling": (lateral.variance_ceiling),
            "temporal_velocity_lateral_intervention_gain_power": lateral.gain_power,
            "temporal_velocity_lateral_intervention_maximum_delta": (lateral.maximum_delta),
        }

    gravity_metrics: dict[str, float] = {}
    gravity_updates: dict[str, Any] = {}
    if args.fit_gravity_intervention:
        required_gravity_fields = {
            "gravity_features",
            "gravity_target_delta",
            "gravity_prior_variance",
            "gravity_confidence",
        }
        missing_gravity_fields = required_gravity_fields - set(train_examples)
        missing_gravity_fields |= required_gravity_fields - set(validation_examples)
        if missing_gravity_fields:
            raise ValueError(
                "cached features do not contain gravity intervention targets: "
                + ", ".join(sorted(missing_gravity_fields))
            )
        gravity_intervention, gravity_metrics = fit_mlp_lateral_velocity_intervention(
            train_examples["gravity_features"],
            train_examples["gravity_target_delta"],
            train_examples["gravity_prior_variance"],
            train_examples["gravity_confidence"],
            validation_examples["gravity_features"],
            validation_examples["gravity_target_delta"],
            validation_examples["gravity_prior_variance"],
            validation_examples["gravity_confidence"],
            hidden_features=args.gravity_hidden_features,
            steps=args.gravity_fit_steps,
            learning_rate=args.gravity_learning_rate,
            weight_decay=args.gravity_weight_decay,
            gain_sparsity=args.gravity_gain_sparsity,
            variance_floor=args.gravity_variance_floor,
            variance_ceiling=args.gravity_variance_ceiling,
            gain_power=args.gravity_gain_power,
            maximum_delta=args.gravity_maximum_delta,
            robust_clip_norm=config.model.filter.robust_clip,
            seed=config.project.seed,
        )
        gravity_metrics.update(
            {
                "gain_sparsity": args.gravity_gain_sparsity,
                "learning_rate": args.gravity_learning_rate,
                "weight_decay": args.gravity_weight_decay,
            }
        )
        gravity_updates = {
            "temporal_velocity_gravity_intervention_enabled": True,
            "temporal_velocity_gravity_intervention_hidden_weights": (
                gravity_intervention.hidden_weights
            ),
            "temporal_velocity_gravity_intervention_hidden_bias": (
                gravity_intervention.hidden_bias
            ),
            "temporal_velocity_gravity_intervention_output_weights": (
                gravity_intervention.output_weights
            ),
            "temporal_velocity_gravity_intervention_output_bias": (
                gravity_intervention.output_bias
            ),
            "temporal_velocity_gravity_intervention_variance_floor": (
                gravity_intervention.variance_floor
            ),
            "temporal_velocity_gravity_intervention_variance_ceiling": (
                gravity_intervention.variance_ceiling
            ),
            "temporal_velocity_gravity_intervention_gain_power": (gravity_intervention.gain_power),
            "temporal_velocity_gravity_intervention_maximum_delta": (
                gravity_intervention.maximum_delta
            ),
        }

    enable_change_point_runtime = args.fit_outgoing_proposal or not (
        args.fit_lateral_intervention or args.fit_gravity_intervention
    )
    gate_rgb = replace(
        config.model.rgb,
        temporal_velocity_post_event_gravity_axis_enabled=(enable_change_point_runtime),
        temporal_velocity_change_point_enabled=enable_change_point_runtime,
        temporal_velocity_change_point_require_contact_mode=False,
        temporal_velocity_change_point_probability_threshold=(gate.probability_threshold),
        **gate_updates,
        **proposal_updates,
        **lateral_updates,
        **gravity_updates,
    )
    gate_config = replace(config, model=replace(config.model, rgb=gate_rgb))
    gate_config.validate()

    checkpoint_path = output / "checkpoints" / "change_point_gate.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = dict(payload)
    checkpoint_config = deepcopy(payload["config"])
    checkpoint_config["model"]["rgb"] = gate_config.to_dict()["model"]["rgb"]
    output_payload["config"] = checkpoint_config
    output_payload["metrics"] = {
        **dict(payload.get("metrics", {})),
        **{f"change_point_gate_{name}": value for name, value in metrics.items()},
        **{f"outgoing_velocity_proposal_{name}": value for name, value in proposal_metrics.items()},
        **{
            f"lateral_velocity_intervention_{name}": value
            for name, value in lateral_metrics.items()
        },
        **{
            f"gravity_velocity_intervention_{name}": value
            for name, value in gravity_metrics.items()
        },
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(output_payload, temporary)
    temporary.replace(checkpoint_path)
    save_resolved_config(gate_config, output / "config.resolved.yaml")

    report = {
        "source_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "device": str(device_info.device),
        "mps_available": device_info.mps_available,
        "minimum_velocity_jump": args.minimum_velocity_jump,
        "minimum_precision": args.minimum_precision,
        "train_feature_cache": (
            str(Path(args.train_cache).expanduser().resolve()) if args.train_cache else None
        ),
        "validation_feature_cache": (
            str(Path(args.validation_cache).expanduser().resolve())
            if args.validation_cache
            else None
        ),
        "gate_type": args.gate_type,
        "gate": {
            **{
                key.removeprefix("temporal_velocity_change_point_"): (
                    list(value) if isinstance(value, tuple) else value
                )
                for key, value in gate_updates.items()
            },
            "probability_threshold": gate.probability_threshold,
        },
        "outgoing_proposal": {
            key.removeprefix("temporal_velocity_outgoing_proposal_"): (
                list(value) if isinstance(value, tuple) else value
            )
            for key, value in proposal_updates.items()
        },
        "lateral_intervention": {
            key.removeprefix("temporal_velocity_lateral_intervention_"): (
                list(value) if isinstance(value, tuple) else value
            )
            for key, value in lateral_updates.items()
        },
        "gravity_intervention": {
            key.removeprefix("temporal_velocity_gravity_intervention_"): (
                list(value) if isinstance(value, tuple) else value
            )
            for key, value in gravity_updates.items()
        },
        "train_collection": train_collection,
        "validation_collection": validation_collection,
        "metrics": metrics,
        "proposal_metrics": proposal_metrics,
        "lateral_metrics": lateral_metrics,
        "gravity_metrics": gravity_metrics,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "\n".join(
            (
                "# RGB trajectory change-point gate and outgoing velocity proposal",
                "",
                f"- source checkpoint: `{report['source_checkpoint']}`",
                f"- trained checkpoint: `{checkpoint_path}`",
                f"- device: `{device_info.device}`",
                f"- train eligible/positive: `{int(train_collection['eligible_windows'])}` / "
                f"`{int(train_collection['observable_jump_windows'])}`",
                "- validation eligible/positive: "
                f"`{int(validation_collection['eligible_windows'])}` / "
                f"`{int(validation_collection['observable_jump_windows'])}`",
                f"- validation precision: `{metrics['validation_precision']:.6f}`",
                f"- validation recall: `{metrics['validation_recall']:.6f}`",
                f"- validation F1: `{metrics['validation_f1']:.6f}`",
                f"- probability threshold: `{gate.probability_threshold:.6f}`",
                *(
                    (
                        "- proposal validation prior/proposal RMSE: "
                        f"`{proposal_metrics['validation_prior_rmse_mps']:.6f}` / "
                        f"`{proposal_metrics['validation_proposal_rmse_mps']:.6f}` m/s",
                        "- proposal validation positive-improvement rate: "
                        f"`{proposal_metrics['validation_positive_improvement_rate']:.6f}`",
                        "- proposal gate-selected validation prior/proposal RMSE: "
                        f"`{proposal_metrics.get('validation_gate_selected_prior_rmse_mps', float('nan')):.6f}` / "
                        f"`{proposal_metrics.get('validation_gate_selected_proposal_rmse_mps', float('nan')):.6f}` m/s",
                        "- proposal calibrated variance: "
                        f"`{proposal_metrics['calibrated_variance_mps2']:.6f}` m2/s2",
                    )
                    if proposal_metrics
                    else ()
                ),
                *(
                    (
                        "- gravity validation prior/posterior RMSE: "
                        f"`{gravity_metrics['validation_prior_rmse_mps']:.6f}` / "
                        f"`{gravity_metrics['validation_posterior_rmse_mps']:.6f}` m/s",
                        "- gravity validation positive-improvement rate: "
                        f"`{gravity_metrics['validation_positive_improvement_rate']:.6f}`",
                        "- gravity validation mean soft gain: "
                        f"`{gravity_metrics['validation_mean_soft_gain']:.6f}`",
                    )
                    if gravity_metrics
                    else ()
                ),
                *(
                    (
                        "- lateral validation prior/posterior RMSE: "
                        f"`{lateral_metrics['validation_prior_rmse_mps']:.6f}` / "
                        f"`{lateral_metrics['validation_posterior_rmse_mps']:.6f}` m/s",
                        "- lateral validation positive-improvement rate: "
                        f"`{lateral_metrics['validation_positive_improvement_rate']:.6f}`",
                        "- lateral validation mean soft gain: "
                        f"`{lateral_metrics['validation_mean_soft_gain']:.6f}`",
                    )
                    if lateral_metrics
                    else ()
                ),
                "",
                "This is a local supervised gate fit. Simulator state supplied labels only;",
                "runtime gate features remain causal and RGB-derived.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
