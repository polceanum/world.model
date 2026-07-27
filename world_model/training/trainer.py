"""Local AdamW trainer for the first complete Orpheus vertical slice."""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from world_model.datasets import SyntheticSphereDataset, collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import load_checkpoint, save_checkpoint
from world_model.training.logging import MetricsLogger
from world_model.training.loop import (
    TrainingBatchResult,
    move_batch_to_device,
    pretrain_rgb_measurements,
    run_closed_loop_batch,
    select_closed_loop_window,
)
from world_model.utils.config import OrpheusConfig, save_resolved_config
from world_model.utils.device import DeviceInfo, select_device
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything

_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ROLLOUT_SELECTION_MIN_DELTA = 1.0e-5
_ROLLOUT_SELECTION_METRIC_VERSION = 2.0


def _repository_root(config: OrpheusConfig) -> Path:
    if config.source_path:
        source = Path(config.source_path).resolve()
        if source.parent.name == "configs":
            return source.parent.parent
        return source.parent
    return Path.cwd().resolve()


def _output_root(config: OrpheusConfig) -> Path:
    output = Path(config.project.output_dir).expanduser()
    if not output.is_absolute():
        output = _repository_root(config) / output
    return output.resolve()


def _new_run_name(config: OrpheusConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    project = re.sub(r"[^A-Za-z0-9._-]+", "-", config.project.name).strip("-")
    return f"{project or 'orpheus'}-{stamp}"


def _resolve_run_directory(
    config: OrpheusConfig,
    *,
    run_name: str | None,
    resume_path: str | Path | None,
) -> Path:
    if resume_path is not None and run_name is None:
        checkpoint = Path(resume_path).expanduser().resolve()
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent
    selected = run_name or _new_run_name(config)
    if not _SAFE_RUN_NAME.fullmatch(selected):
        raise ValueError(
            "run_name must contain only letters, digits, '.', '_', and '-' "
            "and cannot begin with punctuation"
        )
    return _output_root(config) / selected


def _next_batch(
    loader: DataLoader[dict[str, Any]],
    iterator: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _make_loader(
    config: OrpheusConfig,
    *,
    split: str,
    episodes: int,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    dataset = SyntheticSphereDataset(
        config,
        split=split,
        num_episodes=episodes,
        memory_cache=config.training.fixed_dataset,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.project.seed + (0 if split == "train" else 10_000))
    return DataLoader(
        dataset,
        batch_size=min(config.training.batch_size, max(1, episodes)),
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        collate_fn=collate_episodes,
        drop_last=False,
        generator=generator,
    )


def _check_batch_major(batch: Mapping[str, Any]) -> None:
    rgb = batch.get("rgb")
    timestamps = batch.get("timestamps")
    if not isinstance(rgb, Tensor) or rgb.ndim != 5:
        raise ValueError("DataLoader must emit rgb with shape [B,T,3,H,W]")
    if not isinstance(timestamps, Tensor) or timestamps.shape != rgb.shape[:2]:
        raise ValueError("DataLoader must emit timestamps with shape [B,T]")


def _result_metrics(
    result: TrainingBatchResult,
    *,
    learning_rate: float,
    gradient_norm: float | None = None,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "phase": result.phase,
        "loss_total": float(result.total_loss.detach().cpu()),
        "learning_rate": float(learning_rate),
    }
    for name, value in result.loss_terms.items():
        metrics[f"loss_{name}"] = float(value.detach().cpu())
    for name, value in result.metrics.items():
        if math.isfinite(value):
            metrics[name] = float(value)
    if gradient_norm is not None:
        metrics["gradient_norm"] = float(gradient_norm)
    return metrics


def _rollout_selection_is_compatible(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> bool:
    """Return whether a resumed best score uses this objective's semantics."""

    metrics = payload.get("metrics")
    checkpoint_config = payload.get("config")
    if not isinstance(metrics, Mapping) or not isinstance(checkpoint_config, Mapping):
        return False
    try:
        version = float(metrics.get("rollout_selection_metric_version", 0.0))
    except (TypeError, ValueError):
        return False
    if version != _ROLLOUT_SELECTION_METRIC_VERSION:
        return False
    checkpoint_training = checkpoint_config.get("training")
    checkpoint_evaluation = checkpoint_config.get("evaluation")
    if not isinstance(checkpoint_training, Mapping) or not isinstance(
        checkpoint_evaluation,
        Mapping,
    ):
        return False
    requested = config.to_dict()
    return (
        checkpoint_training.get("horizon_weights")
        == requested["training"]["horizon_weights"]
        and checkpoint_evaluation.get("horizons_seconds")
        == requested["evaluation"]["horizons_seconds"]
    )


def measurement_pretrain_frame_index(
    step: int,
    *,
    loader_batches: int,
    total_frames: int,
    fixed_dataset: bool,
) -> int:
    """Choose an RGB pretraining frame without coupling batches to parity.

    For a fixed dataset, every batch sees frame 0 before every batch sees frame
    1, and so on.  The previous ``step % total_frames`` rule coupled loader
    position and frame parity, so some episodes could never see half of their
    frames.  Streaming/shuffled datasets retain independently sampled frames.
    """

    if loader_batches <= 0 or total_frames <= 0:
        raise ValueError("loader_batches and total_frames must be positive")
    if fixed_dataset:
        return (step // loader_batches) % total_frames
    return random.randrange(total_frames)


def set_global_perception_trainable(
    model: OnlineWorldModel,
    *,
    trainable: bool,
) -> None:
    """Freeze or unfreeze full-frame RGB discovery without disabling fast ROI."""

    module = model.observation_modules["rgb"]
    for component_name in ("backbone", "global_detector"):
        component = getattr(module, component_name, None)
        if component is None:
            raise TypeError(f"RGB module is missing {component_name}")
        component.requires_grad_(trainable)


def _mean_batch_results(
    results: list[TrainingBatchResult],
    *,
    weights: list[float] | None = None,
) -> TrainingBatchResult:
    if not results:
        raise ValueError("cannot average an empty validation result list")
    if weights is None:
        weights = [1.0] * len(results)
    if len(weights) != len(results):
        raise ValueError("validation result weights must match the result count")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("validation result weights must be finite and positive")
    total_weight = float(sum(weights))
    phase = results[0].phase
    if any(result.phase != phase for result in results):
        raise ValueError("validation results must share a phase")
    term_names = set(results[0].loss_terms)
    if any(set(result.loss_terms) != term_names for result in results):
        raise ValueError("validation results must share loss terms")
    metric_names = set(results[0].metrics)
    if any(set(result.metrics) != metric_names for result in results):
        raise ValueError("validation results must share metrics")
    return TrainingBatchResult(
        total_loss=(
            torch.stack(
                [
                    result.total_loss * float(weight)
                    for result, weight in zip(results, weights, strict=True)
                ]
            ).sum()
            / total_weight
        ),
        loss_terms={
            name: (
                torch.stack(
                    [
                        result.loss_terms[name] * float(weight)
                        for result, weight in zip(results, weights, strict=True)
                    ]
                ).sum()
                / total_weight
            )
            for name in sorted(term_names)
        },
        metrics={
            name: float(
                sum(
                    result.metrics[name] * float(weight)
                    for result, weight in zip(results, weights, strict=True)
                )
                / total_weight
            )
            for name in sorted(metric_names)
        },
        phase=phase,
    )


@torch.no_grad()
def _validation_step(
    model: OnlineWorldModel,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    closed_loop: bool,
) -> TrainingBatchResult:
    was_training = model.training
    model.eval()
    try:
        if closed_loop:
            result = run_closed_loop_batch(
                model,
                batch,
                config,
                window_start=0,
                window_steps=int(batch["rgb"].shape[1]),
                apply_perturbations=False,
                include_measurement_supervision=True,
            )
        else:
            total_frames = int(batch["rgb"].shape[1])
            frame_count = min(
                total_frames,
                config.training.measurement_validation_frames,
            )
            if frame_count == total_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = (
                    torch.linspace(0, total_frames - 1, frame_count)
                    .round()
                    .to(dtype=torch.int64)
                    .unique(sorted=True)
                    .tolist()
                )
            result = _mean_batch_results(
                [
                    pretrain_rgb_measurements(
                        model,
                        batch,
                        config,
                        frame_index=int(frame_index),
                    )
                    for frame_index in frame_indices
                ]
            )
    finally:
        model.train(was_training)
    return result


@torch.no_grad()
def _validation_loader_result(
    model: OnlineWorldModel,
    loader: DataLoader[dict[str, Any]],
    config: OrpheusConfig,
    *,
    device: torch.device,
    closed_loop: bool,
) -> TrainingBatchResult:
    """Evaluate every configured validation episode exactly once."""

    results: list[TrainingBatchResult] = []
    batch_sizes: list[float] = []
    for raw_batch in loader:
        _check_batch_major(raw_batch)
        batch = move_batch_to_device(raw_batch, device)
        results.append(
            _validation_step(
                model,
                batch,
                config,
                closed_loop=closed_loop,
            )
        )
        batch_sizes.append(float(batch["rgb"].shape[0]))
    return _mean_batch_results(results, weights=batch_sizes)


def _write_run_metadata(
    path: Path,
    *,
    config: OrpheusConfig,
    device_info: DeviceInfo,
    resume_path: str | Path | None,
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": config.project.name,
        "seed": config.project.seed,
        "device": str(device_info.device),
        "torch_version": device_info.torch_version,
        "mps_built": device_info.mps_built,
        "mps_available": device_info.mps_available,
        "cuda_available": device_info.cuda_available,
        "precision": device_info.precision,
        "runtime_modality": config.runtime.modality,
        "debug_oracle_enabled": config.runtime.enable_debug_oracle,
        "resume_path": None if resume_path is None else str(Path(resume_path).resolve()),
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_from_config(
    config: OrpheusConfig,
    *,
    run_name: str | None = None,
    resume_path: str | Path | None = None,
    device_info: DeviceInfo | None = None,
) -> dict[str, Any]:
    """Train RGB measurements, then the causal RGB-only online loop.

    This signature is the public contract used by :mod:`train.py`.
    """

    config.validate()
    if config.runtime.modality != "rgb":
        raise ValueError(
            "the primary trainer requires runtime.modality=rgb; "
            "debug_oracle remains an explicit debugging path"
        )
    if config.runtime.enable_debug_oracle:
        raise ValueError("RGB training cannot enable privileged debug_oracle input")
    resolved_device = device_info or select_device(config.device.preference)
    device = resolved_device.device
    seed_everything(
        config.project.seed,
        deterministic=config.project.deterministic,
    )
    run_directory = _resolve_run_directory(
        config,
        run_name=run_name,
        resume_path=resume_path,
    )
    if run_directory.exists() and resume_path is None:
        occupied = any(
            (run_directory / name).exists()
            for name in ("metrics.jsonl", "checkpoints", "config.resolved.yaml")
        )
        if occupied:
            raise FileExistsError(
                f"run directory already contains training artefacts: {run_directory}"
            )
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_directory / "config.resolved.yaml"
    save_resolved_config(config, resolved_config_path)
    _write_run_metadata(
        run_directory / "run_metadata.json",
        config=config,
        device_info=resolved_device,
        resume_path=resume_path,
    )

    train_loader = _make_loader(
        config,
        split="train",
        episodes=config.training.train_episodes,
        shuffle=not config.training.fixed_dataset,
    )
    validation_loader = _make_loader(
        config,
        split="validation",
        episodes=config.training.validation_episodes,
        shuffle=False,
    )
    train_iterator = iter(train_loader)
    model = OnlineWorldModel.from_config(config, device=device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_step = 0
    best_rollout = math.inf
    best_measurement = math.inf
    best_rollout_validated = False
    best_measurement_validated = False
    resumed_from: str | None = None
    if resume_path is not None:
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
            restore_rng=True,
            expected_config=config,
        )
        start_step = int(payload["step"])
        resume_metrics = payload.get("metrics", {})
        best_rollout_validated = bool(float(resume_metrics.get("best_rollout_validated", 0.0)))
        best_measurement_validated = bool(
            float(resume_metrics.get("best_measurement_validated", 0.0))
        )
        if best_rollout_validated and not _rollout_selection_is_compatible(payload, config):
            # A numerically smaller score from the legacy per-anchor objective,
            # or from a different horizon set, is not comparable with the
            # globally horizon-balanced physical-position objective.
            best_rollout_validated = False
        if best_rollout_validated:
            position_metric = resume_metrics.get("best_rollout_position_loss")
            if position_metric is None:
                # Older checkpoints selected on an unnormalised mean of metres
                # and metres/second. Keep them loadable, but require a new
                # physical-position validation before retaining "best" status.
                best_rollout_validated = False
            else:
                best_rollout = float(position_metric)
        if best_measurement_validated:
            localization_metric = resume_metrics.get("best_measurement_world_position_mae_m")
            if localization_metric is None:
                # Older checkpoints selected on the summed (possibly negative)
                # measurement objective.  Keep them loadable, but require a new
                # calibrated localization validation before calling one best.
                best_measurement_validated = False
            else:
                best_measurement = float(localization_metric)
        resumed_from = str(Path(resume_path).expanduser().resolve())
        if start_step > config.training.steps:
            raise ValueError(
                "checkpoint step exceeds configured training.steps "
                f"({start_step} > {config.training.steps})"
            )

    logger = MetricsLogger(run_directory / "metrics.jsonl")
    last_metrics: dict[str, float | str] = {}
    best_rollout_path = checkpoint_directory / "best_rollout.pt"
    best_measurement_path = checkpoint_directory / "best_measurement.pt"
    last_path = checkpoint_directory / "last.pt"
    started = time.perf_counter()

    for step in range(start_step, config.training.steps):
        if (
            step == config.training.rgb_pretrain_steps
            and best_measurement_validated
            and best_measurement_path.is_file()
        ):
            # Enter the downstream stage from the best calibrated perception
            # state, not merely the final pretraining iterate.  Heteroscedastic
            # objectives can improve NLL while physical localization regresses.
            load_checkpoint(
                best_measurement_path,
                model=model,
                optimizer=optimizer,
                map_location=device,
                restore_rng=False,
                expected_config=config,
            )
            print(
                "restored best RGB localization checkpoint for closed-loop handoff "
                f"(world_mae={best_measurement:.6f}m)",
                flush=True,
            )
        raw_batch, train_iterator = _next_batch(train_loader, train_iterator)
        _check_batch_major(raw_batch)
        batch = move_batch_to_device(raw_batch, device)
        global_perception_trainable = (
            step
            < config.training.rgb_pretrain_steps
            + config.training.closed_loop_global_trainable_steps
        )
        set_global_perception_trainable(
            model,
            trainable=global_perception_trainable,
        )
        optimizer.zero_grad(set_to_none=True)
        if step < config.training.rgb_pretrain_steps:
            target_learning_rate = config.training.learning_rate
            frame_index = measurement_pretrain_frame_index(
                step,
                loader_batches=len(train_loader),
                total_frames=int(batch["rgb"].shape[1]),
                fixed_dataset=config.training.fixed_dataset,
            )
            result = pretrain_rgb_measurements(
                model,
                batch,
                config,
                frame_index=frame_index,
            )
        else:
            target_learning_rate = (
                config.training.learning_rate * config.training.closed_loop_learning_rate_scale
            )
            total_frames = int(batch["rgb"].shape[1])
            window_steps = min(total_frames, config.training.tbptt_steps)
            maximum_rollout_frame_offset = max(
                max(
                    1,
                    int(round(horizon * config.simulator.frame_rate)),
                )
                for horizon in config.evaluation.horizons_seconds
            )
            window_start = select_closed_loop_window(
                batch,
                window_steps,
                event_condition_probability=(config.training.collision_window_probability),
                maximum_rollout_frame_offset=maximum_rollout_frame_offset,
                long_horizon_probability=(
                    config.training.long_horizon_window_probability
                ),
            )
            result = run_closed_loop_batch(
                model,
                batch,
                config,
                window_start=window_start,
                window_steps=window_steps,
                apply_perturbations=True,
                include_measurement_supervision=True,
            )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = target_learning_rate
        if not bool(torch.isfinite(result.total_loss)):
            raise FloatingPointError(f"nonfinite {result.phase} loss at optimiser step {step}")
        result.total_loss.backward()
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
            error_if_nonfinite=False,
        )
        if not bool(torch.isfinite(gradient_norm_tensor)):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"nonfinite gradient norm at optimiser step {step}")
        optimizer.step()
        completed_step = step + 1
        learning_rate = float(optimizer.param_groups[0]["lr"])
        last_metrics = _result_metrics(
            result,
            learning_rate=learning_rate,
            gradient_norm=float(gradient_norm_tensor.detach().cpu()),
        )
        last_metrics["global_perception_trainable"] = float(global_perception_trainable)
        should_log = (
            completed_step % max(1, config.training.log_every) == 0
            or completed_step == config.training.steps
        )
        if should_log:
            record = logger.log(
                step=completed_step,
                split="train",
                metrics=last_metrics,
            )
            print(
                f"step={completed_step}/{config.training.steps} "
                f"phase={result.phase} loss={record['loss_total']:.6f} "
                f"grad={record['gradient_norm']:.4f}",
                flush=True,
            )

        should_validate = config.training.eval_every > 0 and (
            completed_step % config.training.eval_every == 0
            or completed_step == config.training.rgb_pretrain_steps
            or completed_step == config.training.steps
        )
        if should_validate:
            validation = _validation_loader_result(
                model,
                validation_loader,
                config,
                device=device,
                closed_loop=completed_step > config.training.rgb_pretrain_steps,
            )
            validation_metrics = _result_metrics(
                validation,
                learning_rate=learning_rate,
            )
            logger.log(
                step=completed_step,
                split="validation",
                metrics=validation_metrics,
            )
            is_closed_loop_validation = (
                completed_step > config.training.rgb_pretrain_steps
                and validation.phase == "closed_loop_rgb"
            )
            if is_closed_loop_validation and "rollout_position" in validation.loss_terms:
                rollout_position_metric = float(
                    validation.loss_terms["rollout_position"].detach().cpu()
                )
                improved = math.isfinite(rollout_position_metric) and (
                    not math.isfinite(best_rollout)
                    or rollout_position_metric < best_rollout - _ROLLOUT_SELECTION_MIN_DELTA
                )
                if improved:
                    best_rollout = rollout_position_metric
                    best_rollout_validated = True
                    per_horizon_metrics = {
                        f"validation_{name}": float(value)
                        for name, value in validation.metrics.items()
                        if name.startswith("rollout_position@")
                    }
                    save_checkpoint(
                        best_rollout_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=completed_step,
                        metrics={
                            "validation_total_loss": float(validation.total_loss.detach().cpu()),
                            "validation_rollout_loss": float(
                                validation.loss_terms["rollout"].detach().cpu()
                            ),
                            "validation_rollout_position_loss": best_rollout,
                            "best_rollout_loss": best_rollout,
                            "best_rollout_position_loss": best_rollout,
                            "best_rollout_validated": 1.0,
                            "rollout_selection_metric_version": (
                                _ROLLOUT_SELECTION_METRIC_VERSION
                            ),
                            **per_horizon_metrics,
                            "best_measurement_validated": float(best_measurement_validated),
                            **(
                                {
                                    "best_measurement_loss": best_measurement,
                                    "best_measurement_world_position_mae_m": best_measurement,
                                }
                                if best_measurement_validated
                                else {}
                            ),
                        },
                        device=str(device),
                    )
            elif validation.phase == "rgb_pretrain" and "measurement" in validation.loss_terms:
                measurement_metric = validation.metrics.get("rgb_world_position_mae_m")
                if measurement_metric is None:
                    raise RuntimeError(
                        "RGB pretraining validation did not report world localization MAE"
                    )
                if math.isfinite(measurement_metric) and measurement_metric < best_measurement:
                    best_measurement = measurement_metric
                    best_measurement_validated = True
                    save_checkpoint(
                        best_measurement_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=completed_step,
                        metrics={
                            "validation_total_loss": float(validation.total_loss.detach().cpu()),
                            "validation_measurement_loss": float(
                                validation.loss_terms["measurement"].detach().cpu()
                            ),
                            "validation_world_position_mae_m": best_measurement,
                            "best_measurement_loss": best_measurement,
                            "best_measurement_world_position_mae_m": best_measurement,
                            "best_measurement_validated": 1.0,
                            "best_rollout_validated": float(best_rollout_validated),
                            "rollout_selection_metric_version": (
                                _ROLLOUT_SELECTION_METRIC_VERSION
                            ),
                            **(
                                {
                                    "best_rollout_loss": best_rollout,
                                    "best_rollout_position_loss": best_rollout,
                                }
                                if best_rollout_validated
                                else {}
                            ),
                        },
                        device=str(device),
                    )

        should_checkpoint = (
            config.training.checkpoint_every > 0
            and completed_step % config.training.checkpoint_every == 0
        )
        if should_checkpoint or completed_step == config.training.steps:
            checkpoint_metrics = {
                key: float(value)
                for key, value in last_metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            checkpoint_metrics["best_rollout_validated"] = float(best_rollout_validated)
            checkpoint_metrics["best_measurement_validated"] = float(best_measurement_validated)
            checkpoint_metrics["rollout_selection_metric_version"] = (
                _ROLLOUT_SELECTION_METRIC_VERSION
            )
            if best_rollout_validated:
                checkpoint_metrics["best_rollout_loss"] = best_rollout
                checkpoint_metrics["best_rollout_position_loss"] = best_rollout
            if best_measurement_validated:
                checkpoint_metrics["best_measurement_loss"] = best_measurement
                checkpoint_metrics["best_measurement_world_position_mae_m"] = best_measurement
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
            )

    if config.training.steps == start_step:
        # A zero-step or already-complete resume is still a valid, inspectable
        # local checkpoint rather than a silently empty run.
        selection_metrics = {
            "best_rollout_validated": float(best_rollout_validated),
            "best_measurement_validated": float(best_measurement_validated),
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        }
        if best_rollout_validated:
            selection_metrics["best_rollout_loss"] = best_rollout
            selection_metrics["best_rollout_position_loss"] = best_rollout
        if best_measurement_validated:
            selection_metrics["best_measurement_loss"] = best_measurement
            selection_metrics["best_measurement_world_position_mae_m"] = best_measurement
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            config=config,
            step=start_step,
            metrics=selection_metrics,
            device=str(device),
        )

    elapsed = time.perf_counter() - started
    has_best_rollout_checkpoint = best_rollout_validated and best_rollout_path.is_file()
    has_best_measurement_checkpoint = best_measurement_validated and best_measurement_path.is_file()
    if has_best_rollout_checkpoint:
        selected_checkpoint = best_rollout_path
        selected_checkpoint_kind = "best_rollout"
    elif has_best_measurement_checkpoint:
        selected_checkpoint = best_measurement_path
        selected_checkpoint_kind = "best_measurement"
    else:
        selected_checkpoint = last_path
        selected_checkpoint_kind = "last_unvalidated"
    result_payload: dict[str, Any] = {
        "run_directory": str(run_directory),
        "resolved_config": str(resolved_config_path),
        "metrics_jsonl": str(logger.path),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(selected_checkpoint),
        "best_checkpoint_kind": selected_checkpoint_kind,
        "best_rollout_checkpoint": (
            str(best_rollout_path) if has_best_rollout_checkpoint else None
        ),
        "best_measurement_checkpoint": (
            str(best_measurement_path) if has_best_measurement_checkpoint else None
        ),
        "best_rollout_validated": best_rollout_validated,
        "best_measurement_validated": best_measurement_validated,
        "best_rollout_loss": best_rollout if best_rollout_validated else None,
        "best_rollout_position_loss": (best_rollout if best_rollout_validated else None),
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        "best_measurement_loss": (best_measurement if best_measurement_validated else None),
        "best_measurement_world_position_mae_m": (
            best_measurement if best_measurement_validated else None
        ),
        "completed_steps": config.training.steps,
        "rgb_pretrain_steps": min(config.training.steps, config.training.rgb_pretrain_steps),
        "closed_loop_steps": max(0, config.training.steps - config.training.rgb_pretrain_steps),
        "device": str(device),
        "precision": resolved_device.precision,
        "elapsed_seconds": elapsed,
        "resumed_from": resumed_from,
        "last_metrics": last_metrics,
        "oracle_runtime_input_used": False,
    }
    atomic_write_text(
        run_directory / "train_summary.json",
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
    )
    return result_payload


__all__ = ["train_from_config"]
