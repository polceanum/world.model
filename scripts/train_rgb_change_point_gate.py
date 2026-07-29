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
) -> tuple[Tensor, Tensor, dict[str, float]]:
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
            if (
                not isinstance(gate_features, Tensor)
                or not isinstance(feature_valid, Tensor)
                or not isinstance(feature_timestamps, Tensor)
            ):
                raise RuntimeError("RGB runtime did not expose change-point training features")
            valid = feature_valid.bool() & matched & belief.objects.active
            if valid.any():
                aligned_target = torch.zeros_like(valid)
                aligned_collision = torch.zeros_like(valid)
                for belief_slot_tensor in torch.nonzero(valid[0], as_tuple=False).flatten():
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
                features.append(gate_features.masked_select(valid.unsqueeze(-1)).reshape(-1, 9))
                targets.append(aligned_target.masked_select(valid))
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
    return (
        torch.cat(features).cpu(),
        torch.cat(targets).cpu(),
        {
            "episodes": float(episodes),
            "eligible_windows": float(eligible_count),
            "collision_windows": float(collision_count),
            "observable_jump_windows": float(jump_count),
        },
    )


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
        map_location=device_info.device,
        expected_config=config,
    )
    if args.train_cache:
        cached_train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
        train_features = cached_train["features"]
        train_targets = cached_train["targets"]
        train_collection = cached_train["collection"]
    else:
        train_features, train_targets, train_collection = collect_examples(
            model,
            config,
            split="train",
            episodes=args.train_episodes,
            seed_offset=args.train_seed_offset,
            device=device_info.device,
            minimum_velocity_jump=args.minimum_velocity_jump,
        )
    torch.save(
        {"features": train_features, "targets": train_targets, "collection": train_collection},
        output / "train_features.pt",
    )
    if args.validation_cache:
        cached_validation = torch.load(
            args.validation_cache,
            map_location="cpu",
            weights_only=False,
        )
        validation_features = cached_validation["features"]
        validation_targets = cached_validation["targets"]
        validation_collection = cached_validation["collection"]
    else:
        validation_features, validation_targets, validation_collection = collect_examples(
            model,
            config,
            split="validation",
            episodes=args.validation_episodes,
            seed_offset=args.validation_seed_offset,
            device=device_info.device,
            minimum_velocity_jump=args.minimum_velocity_jump,
        )
    torch.save(
        {
            "features": validation_features,
            "targets": validation_targets,
            "collection": validation_collection,
        },
        output / "validation_features.pt",
    )
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

    gate_rgb = replace(
        config.model.rgb,
        temporal_velocity_post_event_gravity_axis_enabled=True,
        temporal_velocity_change_point_enabled=True,
        temporal_velocity_change_point_require_contact_mode=False,
        temporal_velocity_change_point_probability_threshold=(gate.probability_threshold),
        **gate_updates,
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
        "train_collection": train_collection,
        "validation_collection": validation_collection,
        "metrics": metrics,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "\n".join(
            (
                "# RGB trajectory change-point gate",
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
