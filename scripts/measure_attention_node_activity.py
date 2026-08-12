#!/usr/bin/env python3
"""Measure functional node-residual activity on one deterministic causal draw."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import load_model_weights
from world_model.training.loop import (
    move_batch_to_device,
    run_closed_loop_batch,
    select_closed_loop_window,
)
from world_model.training.trainer import _make_loader
from world_model.utils.config import load_config
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gradient_norm(gradients: tuple[Tensor | None, ...]) -> float:
    return math.sqrt(
        sum(
            float(gradient.detach().square().sum())
            for gradient in gradients
            if gradient is not None
        )
    )


def _gradient_enabled_active_residual(
    output: Tensor,
    active: Tensor,
    *,
    maximum: float,
) -> Tensor | None:
    """Select exactly the calls included by ``node_activity_details``.

    Prepared propagation and other runtime bookkeeping may invoke the
    attention residual under ``torch.no_grad()`` during a causal batch.  The
    differentiable activity/drift objective deliberately excludes those
    calls, so the emitted-value trace must exclude them as well.
    """

    if not torch.is_grad_enabled():
        return None
    if output.ndim != 3 or active.shape != output.shape[:2]:
        raise ValueError("node output/active mask shapes are incompatible")
    residual = float(maximum) * torch.tanh(output)
    return residual[active].detach().to(device="cpu", dtype=torch.float64)


def measure_activity(
    *,
    config_path: Path,
    checkpoint_path: Path,
    step_index: int | None,
    device: torch.device,
) -> dict[str, object]:
    """Run one balanced causal draw and compare functional/parameter priors."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    config = load_config(config_path)
    # This is a single-process diagnostic. Avoid a spawn boundary so the
    # command is deterministic and works from an ordinary __main__ entrypoint.
    config = replace(config, training=replace(config.training, num_workers=0))
    model = OnlineWorldModel.from_config(config, device=device)
    payload = load_model_weights(checkpoint_path, model=model)
    checkpoint_step = int(payload["step"])
    if step_index is None:
        step_index = max(0, checkpoint_step - 1)
    if step_index < 0:
        raise ValueError("step index must be nonnegative")

    # Step-indexed batches are independent of preceding loader iteration. Use
    # a named seed for perturbation/window sampling and report it explicitly;
    # this is reproducible calibration, not a claim of exact trainer RNG replay.
    sampling_seed = config.project.seed + step_index
    seed_everything(sampling_seed, deterministic=config.project.deterministic)
    loader = _make_loader(
        config,
        split="train",
        episodes=config.training.train_episodes,
        shuffle=True,
        start_step=step_index,
        stop_step=step_index + 1,
    )
    batch = move_batch_to_device(next(iter(loader)), device)

    model.train()
    attention = model.dynamics.attention_interactions
    if attention is None:
        raise ValueError("configured model has no typed attention residual")
    attention.reset_output_gradient_diagnostics()
    current_active: Tensor | None = None
    emitted_sum = torch.zeros(3, dtype=torch.float64)
    emitted_squared_sum = torch.zeros(3, dtype=torch.float64)
    emitted_minimum = torch.full((3,), math.inf, dtype=torch.float64)
    emitted_maximum = torch.full((3,), -math.inf, dtype=torch.float64)
    emitted_count = 0
    invocation_count = 0

    def capture_active(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        nonlocal current_active
        objects = inputs[0]
        active = getattr(objects, "active", None)
        if not isinstance(active, Tensor):
            raise TypeError("attention input does not expose a tensor active mask")
        current_active = active

    def capture_node_values(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: Tensor,
    ) -> None:
        nonlocal emitted_count, invocation_count
        if current_active is None:
            raise RuntimeError("node decoder ran without an attention active mask")
        selected = _gradient_enabled_active_residual(
            output,
            current_active,
            maximum=attention.max_node_acceleration,
        )
        if selected is None:
            return
        invocation_count += 1
        if selected.numel() == 0:
            return
        emitted_count += int(selected.shape[0])
        emitted_sum.add_(selected.sum(dim=0))
        emitted_squared_sum.add_(selected.square().sum(dim=0))
        emitted_minimum.copy_(torch.minimum(emitted_minimum, selected.amin(dim=0)))
        emitted_maximum.copy_(torch.maximum(emitted_maximum, selected.amax(dim=0)))

    attention_handle = attention.register_forward_pre_hook(capture_active)
    decoder_handle = attention.node_decoder.register_forward_hook(capture_node_values)
    total_frames = int(batch["rgb"].shape[1])
    window_steps = min(total_frames, config.training.tbptt_steps)
    frame_offsets = [
        max(1, int(round(horizon * config.simulator.frame_rate)))
        for horizon in config.evaluation.horizons_seconds
    ]
    window_start = select_closed_loop_window(
        batch,
        window_steps,
        event_condition_probability=config.training.collision_window_probability,
        maximum_rollout_frame_offset=max(frame_offsets),
        minimum_rollout_frame_offset=min(frame_offsets),
        long_horizon_probability=config.training.long_horizon_window_probability,
        joint_collision_long_horizon_sampling=(
            config.training.joint_collision_long_horizon_sampling
        ),
    )
    try:
        result = run_closed_loop_batch(
            model,
            batch,
            config,
            window_start=window_start,
            window_steps=window_steps,
            apply_perturbations=True,
            include_measurement_supervision=True,
            rollout_anchors_per_window=config.training.rollout_anchors_per_window,
        )
    finally:
        decoder_handle.remove()
        attention_handle.remove()
    activity = result.loss_terms.get("attention_node_activity")
    drift = result.loss_terms.get("attention_node_drift")
    complexity = result.loss_terms.get("attention_node_complexity")
    if activity is None or drift is None or complexity is None:
        raise RuntimeError("closed-loop result omitted attention node diagnostics")
    differentiable_records = attention._node_activity_records
    recorded_active_count = int(
        torch.stack([record_count for _, _, record_count in differentiable_records])
        .sum()
        .detach()
        .cpu()
    )
    if invocation_count != len(differentiable_records) or emitted_count != recorded_active_count:
        raise RuntimeError(
            "emitted node trace does not match differentiable activity population "
            f"(calls {invocation_count} != {len(differentiable_records)}, "
            f"objects {emitted_count} != {recorded_active_count})"
        )

    parameters = tuple(parameter for parameter in attention.parameters() if parameter.requires_grad)
    complexity_gradients = torch.autograd.grad(
        complexity,
        parameters,
        allow_unused=True,
        retain_graph=True,
    )
    drift_gradients = torch.autograd.grad(
        drift,
        parameters,
        allow_unused=True,
        retain_graph=True,
    )
    activity_gradients = torch.autograd.grad(
        activity,
        parameters,
        allow_unused=True,
    )
    complexity_gradient_norm = _gradient_norm(complexity_gradients)
    drift_gradient_norm = _gradient_norm(drift_gradients)
    activity_gradient_norm = _gradient_norm(activity_gradients)
    node_parameter_ids = {id(parameter) for parameter in attention.node_decoder.parameters()}
    activity_node_gradient_norm = _gradient_norm(
        tuple(
            gradient if id(parameter) in node_parameter_ids else None
            for parameter, gradient in zip(parameters, activity_gradients, strict=True)
        )
    )
    equal_gradient_weight = (
        complexity_gradient_norm / activity_gradient_norm if activity_gradient_norm > 0.0 else None
    )
    equal_drift_gradient_weight = (
        complexity_gradient_norm / drift_gradient_norm if drift_gradient_norm > 0.0 else None
    )
    if emitted_count <= 0:
        raise RuntimeError("causal draw emitted no active-object node residuals")
    emitted_mean = emitted_sum / emitted_count
    emitted_variance = (emitted_squared_sum / emitted_count - emitted_mean.square()).clamp_min(0.0)

    scenario_names = batch["metadata"]["scenario"]
    return {
        "status": "pass",
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "step_index": step_index,
        "sampling_seed": sampling_seed,
        "device": str(device),
        "scenario_names": list(scenario_names),
        "window_start": window_start,
        "window_steps": window_steps,
        "attention_node_activity": float(activity.detach()),
        "attention_node_activity_x": result.metrics["attention_node_activity_x"],
        "attention_node_activity_y": result.metrics["attention_node_activity_y"],
        "attention_node_activity_z": result.metrics["attention_node_activity_z"],
        "attention_node_activity_rms_acceleration": math.sqrt(float(activity.detach())),
        "attention_node_activity_gradient_norm": activity_gradient_norm,
        "attention_node_activity_node_decoder_gradient_norm": (activity_node_gradient_norm),
        "attention_node_drift": float(drift.detach()),
        "attention_node_drift_gradient_norm": drift_gradient_norm,
        "attention_node_emitted_acceleration_invocation_count": invocation_count,
        "attention_node_emitted_acceleration_active_object_count": emitted_count,
        "attention_node_emitted_acceleration_trace_scope": (
            "gradient_enabled_causal_attention_calls"
        ),
        "attention_node_emitted_acceleration_mean": emitted_mean.tolist(),
        "attention_node_emitted_acceleration_std": emitted_variance.sqrt().tolist(),
        "attention_node_emitted_acceleration_minimum": emitted_minimum.tolist(),
        "attention_node_emitted_acceleration_maximum": emitted_maximum.tolist(),
        "attention_node_complexity": float(complexity.detach()),
        "attention_node_complexity_gradient_norm": complexity_gradient_norm,
        "activity_weight_matching_unit_complexity_gradient": equal_gradient_weight,
        "drift_weight_matching_unit_complexity_gradient": equal_drift_gradient_weight,
        "configured_attention_node_activity_weight": (
            config.training.loss_weights.get("attention_node_activity")
        ),
        "configured_attention_node_drift_weight": (
            config.training.loss_weights.get("attention_node_drift")
        ),
        "configured_attention_node_complexity_weight": (
            config.training.loss_weights.get("attention_node_complexity")
        ),
        "configured_total_loss": float(result.total_loss.detach()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--step-index", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = measure_activity(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        step_index=args.step_index,
        device=torch.device(args.device),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        atomic_write_text(args.output, encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
