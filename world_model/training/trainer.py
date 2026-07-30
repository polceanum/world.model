"""Local AdamW trainer for the first complete Orpheus vertical slice."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from world_model.datasets import (
    SyntheticSphereDataset,
    collate_episodes,
    make_seed_manifest,
)
from world_model.runtime import OnlineWorldModel
from world_model.simulator.sphere_world import SphereWorldConfig
from world_model.training.checkpointing import load_checkpoint, save_checkpoint
from world_model.training.logging import MetricsLogger
from world_model.training.loop import (
    _PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M,
    TrainingBatchResult,
    move_batch_to_device,
    physical_validation_metrics,
    pretrain_rgb_measurements,
    run_closed_loop_batch,
    select_closed_loop_window,
)
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import OrpheusConfig, save_resolved_config
from world_model.utils.device import DeviceInfo, select_device
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything
from world_model.utils.version import SIMULATOR_VERSION

_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ROLLOUT_SELECTION_MIN_DELTA = 1.0e-5
_ROLLOUT_SELECTION_METRIC_VERSION = 3.0
_ROLLOUT_SELECTION_RELATIVE_GUARDRAIL = 0.02
_ROLLOUT_SELECTION_COVERAGE_TOLERANCE = 0.005
_ROLLOUT_SELECTION_CALIBRATION_ERROR_TOLERANCE = 0.02
_ROLLOUT_VALIDATION_PROTOCOL_VERSION = 1
_NOMINAL_POSITION_COVERAGE = 0.90


@dataclass(frozen=True)
class _RolloutSelectionMetrics:
    """Physical validation metrics used to retain a broad-accuracy incumbent."""

    score: float
    position_rmse_m: float
    velocity_rmse_mps: float
    target_coverage: float
    prediction_precision: float
    collision_f1: float
    id_switch_rate: float
    position_coverage90: float
    position_calibration_error90: float
    horizon_position_rmse_m: dict[str, float]
    horizon_forecast_target_coverage: dict[str, float]

    def validation_metrics(self) -> dict[str, float]:
        return {
            "validation_rollout_selection_score": self.score,
            "validation_position_rmse_m": self.position_rmse_m,
            "validation_velocity_rmse_mps": self.velocity_rmse_mps,
            "validation_target_coverage": self.target_coverage,
            "validation_prediction_precision": self.prediction_precision,
            "validation_collision_f1": self.collision_f1,
            "validation_id_switch_rate": self.id_switch_rate,
            "validation_position_coverage90": self.position_coverage90,
            "validation_position_calibration_error90": self.position_calibration_error90,
            **{
                f"validation_position_rmse@{suffix}": value
                for suffix, value in self.horizon_position_rmse_m.items()
            },
            **{
                f"validation_forecast_target_coverage@{suffix}": value
                for suffix, value in self.horizon_forecast_target_coverage.items()
            },
        }

    def checkpoint_metrics(self, *, prefix: str = "best_rollout") -> dict[str, float]:
        return {
            f"{prefix}_selection_score": self.score,
            # Retain the original public alias while making its physical,
            # horizon-weighted meaning explicit in the adjacent fields.
            f"{prefix}_loss": self.score,
            f"{prefix}_position_loss": self.score,
            f"{prefix}_position_rmse_m": self.position_rmse_m,
            f"{prefix}_velocity_rmse_mps": self.velocity_rmse_mps,
            f"{prefix}_target_coverage": self.target_coverage,
            f"{prefix}_prediction_precision": self.prediction_precision,
            f"{prefix}_collision_f1": self.collision_f1,
            f"{prefix}_id_switch_rate": self.id_switch_rate,
            f"{prefix}_position_coverage90": self.position_coverage90,
            f"{prefix}_position_calibration_error90": self.position_calibration_error90,
            **{
                f"{prefix}_position_rmse@{suffix}": value
                for suffix, value in self.horizon_position_rmse_m.items()
            },
            **{
                f"{prefix}_forecast_target_coverage@{suffix}": value
                for suffix, value in self.horizon_forecast_target_coverage.items()
            },
        }


def _selection_horizon_keys(config: OrpheusConfig) -> list[tuple[str, float]]:
    """Return unique physical-horizon metric suffixes and configured weights."""

    selected: list[tuple[str, float]] = []
    seen: set[str] = set()
    for horizon, weight in zip(
        config.evaluation.horizons_seconds,
        config.training.horizon_weights,
        strict=True,
    ):
        frame_offset = max(
            1,
            int(round(float(horizon) * config.simulator.frame_rate)),
        )
        suffix = f"{frame_offset / config.simulator.frame_rate:.3f}s"
        if suffix in seen:
            continue
        seen.add(suffix)
        selected.append((suffix, float(weight)))
    return selected


def _canonical_hash(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible protocol mapping."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rollout_validation_protocol_from_mapping(
    config_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve every field that can change broad closed-loop validation."""

    simulator = config_mapping.get("simulator")
    model = config_mapping.get("model")
    runtime = config_mapping.get("runtime")
    evaluation = config_mapping.get("evaluation")
    training = config_mapping.get("training")
    project = config_mapping.get("project")
    if not all(
        isinstance(section, Mapping)
        for section in (simulator, model, runtime, evaluation, training, project)
    ):
        raise ValueError("validation protocol requires complete resolved config sections")
    assert isinstance(training, Mapping)
    assert isinstance(project, Mapping)
    validation_episodes = int(training["validation_episodes"])
    if validation_episodes <= 0:
        raise ValueError("training.validation_episodes must be positive")
    manifest = make_seed_manifest("validation", validation_episodes)
    resolved_simulator = SphereWorldConfig.from_config(config_mapping)
    return {
        "protocol_version": _ROLLOUT_VALIDATION_PROTOCOL_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        # Keep both the public simulator config and the fully resolved
        # generator config. The latter includes defaults and derived padding.
        "simulator": dict(simulator),
        "resolved_sphere_world": asdict(resolved_simulator),
        "model": dict(model),
        "runtime": dict(runtime),
        "evaluation": dict(evaluation),
        "selection": {
            "horizon_weights": training["horizon_weights"],
            "metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            "prediction_distance_threshold_m": (_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M),
        },
        "execution": {
            "project_seed": project["seed"],
            "deterministic": project["deterministic"],
            "validation_batch_size": min(
                int(training["batch_size"]),
                validation_episodes,
            ),
        },
        "validation_seed_manifest": {
            "split": manifest.split,
            "seeds": list(manifest.seeds),
        },
    }


def _rollout_validation_protocol(config: OrpheusConfig) -> dict[str, Any]:
    return _rollout_validation_protocol_from_mapping(config.to_dict())


def _rollout_validation_protocol_hash(config: OrpheusConfig) -> str:
    return _canonical_hash(_rollout_validation_protocol(config))


def _validation_protocol_checkpoint_metrics(config: OrpheusConfig) -> dict[str, Any]:
    manifest = make_seed_manifest("validation", config.training.validation_episodes)
    manifest_mapping = {
        "split": manifest.split,
        "seeds": list(manifest.seeds),
    }
    return {
        "rollout_validation_protocol_hash": _rollout_validation_protocol_hash(config),
        "validation_seed_manifest_hash": _canonical_hash(manifest_mapping),
        "validation_seed_manifest_count": float(len(manifest)),
        "validation_seed_manifest_first": float(manifest.seeds[0]),
        "validation_seed_manifest_last": float(manifest.seeds[-1]),
    }


def _rollout_selection_metrics(
    metrics: Mapping[str, float],
    config: OrpheusConfig,
) -> _RolloutSelectionMetrics:
    """Extract a complete finite physical checkpoint-selection summary."""

    required = {
        "position_rmse_m": "validation_position_rmse_m",
        "velocity_rmse_mps": "validation_velocity_rmse_mps",
        "target_coverage": "validation_target_coverage",
        "prediction_precision": "validation_prediction_precision",
        "collision_f1": "validation_collision_f1",
        "id_switch_rate": "validation_id_switch_rate",
        "position_coverage90": "validation_position_coverage90",
    }
    missing = [metric_key for metric_key in required.values() if metric_key not in metrics]
    horizon_values: dict[str, float] = {}
    weighted_values: list[float] = []
    weights: list[float] = []
    for suffix, weight in _selection_horizon_keys(config):
        metric_key = f"validation_position_rmse@{suffix}"
        if metric_key not in metrics:
            missing.append(metric_key)
            continue
        value = float(metrics[metric_key])
        horizon_values[suffix] = value
        weighted_values.append(value)
        weights.append(weight)
    horizon_coverage_values: dict[str, float] = {}
    for suffix, _ in _selection_horizon_keys(config):
        metric_key = f"validation_forecast_target_coverage@{suffix}"
        if metric_key not in metrics:
            missing.append(metric_key)
            continue
        horizon_coverage_values[suffix] = float(metrics[metric_key])
    if missing:
        raise RuntimeError(
            "closed-loop validation did not report broad selection metrics: "
            + ", ".join(sorted(missing))
        )

    values = {name: float(metrics[key]) for name, key in required.items()}
    finite_values = [
        *values.values(),
        *horizon_values.values(),
        *horizon_coverage_values.values(),
    ]
    if any(not math.isfinite(value) for value in finite_values):
        raise FloatingPointError("closed-loop broad selection metrics must all be finite")
    if values["position_rmse_m"] < 0 or values["velocity_rmse_mps"] < 0:
        raise ValueError("physical validation RMSE metrics must be nonnegative")
    if any(value < 0 for value in horizon_values.values()):
        raise ValueError("per-horizon validation RMSE metrics must be nonnegative")
    for name in (
        "target_coverage",
        "prediction_precision",
        "collision_f1",
        "id_switch_rate",
        "position_coverage90",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"validation {name} must lie in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in horizon_coverage_values.values()):
        raise ValueError("per-horizon forecast target coverage must lie in [0, 1]")
    weight_total = sum(weights)
    if not math.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("checkpoint-selection horizon weights must sum to a positive value")
    score = (
        sum(value * weight for value, weight in zip(weighted_values, weights, strict=True))
        / weight_total
    )
    return _RolloutSelectionMetrics(
        score=score,
        position_rmse_m=values["position_rmse_m"],
        velocity_rmse_mps=values["velocity_rmse_mps"],
        target_coverage=values["target_coverage"],
        prediction_precision=values["prediction_precision"],
        collision_f1=values["collision_f1"],
        id_switch_rate=values["id_switch_rate"],
        position_coverage90=values["position_coverage90"],
        position_calibration_error90=abs(
            values["position_coverage90"] - _NOMINAL_POSITION_COVERAGE
        ),
        horizon_position_rmse_m=horizon_values,
        horizon_forecast_target_coverage=horizon_coverage_values,
    )


def _rollout_selection_from_checkpoint(
    metrics: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    prefix: str = "best_rollout",
) -> _RolloutSelectionMetrics | None:
    """Restore the broad incumbent required to guard a resumed run."""

    translated: dict[str, float] = {}
    aliases = {
        "validation_position_rmse_m": f"{prefix}_position_rmse_m",
        "validation_velocity_rmse_mps": f"{prefix}_velocity_rmse_mps",
        "validation_target_coverage": f"{prefix}_target_coverage",
        "validation_prediction_precision": f"{prefix}_prediction_precision",
        "validation_collision_f1": f"{prefix}_collision_f1",
        "validation_id_switch_rate": f"{prefix}_id_switch_rate",
        "validation_position_coverage90": f"{prefix}_position_coverage90",
    }
    for output_key, checkpoint_key in aliases.items():
        if checkpoint_key not in metrics:
            return None
        translated[output_key] = float(metrics[checkpoint_key])
    for suffix, _ in _selection_horizon_keys(config):
        checkpoint_key = f"{prefix}_position_rmse@{suffix}"
        if checkpoint_key not in metrics:
            return None
        translated[f"validation_position_rmse@{suffix}"] = float(metrics[checkpoint_key])
        coverage_key = f"{prefix}_forecast_target_coverage@{suffix}"
        if coverage_key not in metrics:
            return None
        translated[f"validation_forecast_target_coverage@{suffix}"] = float(metrics[coverage_key])
    restored = _rollout_selection_metrics(translated, config)
    stored_score = metrics.get(f"{prefix}_selection_score")
    if stored_score is None or not math.isclose(
        float(stored_score),
        restored.score,
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    ):
        return None
    return restored


def _rollout_selection_passes_guardrails(
    candidate: _RolloutSelectionMetrics,
    reference: _RolloutSelectionMetrics,
) -> bool:
    """Reject material regression against one broad physical reference."""

    maximum_ratio = 1.0 + _ROLLOUT_SELECTION_RELATIVE_GUARDRAIL
    minimum_ratio = 1.0 - _ROLLOUT_SELECTION_RELATIVE_GUARDRAIL
    if candidate.position_rmse_m > reference.position_rmse_m * maximum_ratio:
        return False
    if candidate.velocity_rmse_mps > reference.velocity_rmse_mps * maximum_ratio:
        return False
    if candidate.target_coverage < (
        reference.target_coverage - _ROLLOUT_SELECTION_COVERAGE_TOLERANCE
    ):
        return False
    if candidate.prediction_precision < reference.prediction_precision * minimum_ratio:
        return False
    if candidate.collision_f1 < reference.collision_f1 * minimum_ratio:
        return False
    if candidate.id_switch_rate > (
        reference.id_switch_rate + _ROLLOUT_SELECTION_COVERAGE_TOLERANCE
    ):
        return False
    if candidate.position_calibration_error90 > (
        reference.position_calibration_error90 + _ROLLOUT_SELECTION_CALIBRATION_ERROR_TOLERANCE
    ):
        return False
    if any(
        candidate.horizon_position_rmse_m[suffix] > reference_value * maximum_ratio
        for suffix, reference_value in reference.horizon_position_rmse_m.items()
    ):
        return False
    return all(
        candidate.horizon_forecast_target_coverage[suffix]
        >= reference_value - _ROLLOUT_SELECTION_COVERAGE_TOLERANCE
        for suffix, reference_value in reference.horizon_forecast_target_coverage.items()
    )


def _rollout_selection_improves(
    candidate: _RolloutSelectionMetrics,
    incumbent: _RolloutSelectionMetrics,
) -> bool:
    """Require a better forecast score without material broad regressions."""

    if candidate.score >= incumbent.score - _ROLLOUT_SELECTION_MIN_DELTA:
        return False
    return _rollout_selection_passes_guardrails(candidate, incumbent)


def _rollout_validation_checkpoint_metrics(
    validation: TrainingBatchResult,
    candidate: _RolloutSelectionMetrics,
    incumbent: _RolloutSelectionMetrics,
    reference: _RolloutSelectionMetrics,
    *,
    config: OrpheusConfig,
    accepted: bool,
    best_measurement: float | None,
    checkpoint_model_state_hash: str,
    incumbent_model_state_hash: str,
    incumbent_step: int,
    reference_model_state_hash: str,
    reference_step: int,
) -> dict[str, Any]:
    """Build truthful metadata shared by best and numbered validation saves."""

    metrics: dict[str, Any] = {
        "validation_total_loss": float(validation.total_loss.detach().cpu()),
        "validation_rollout_loss": float(
            validation.loss_terms.get("rollout", validation.total_loss).detach().cpu()
        ),
        "validation_rollout_position_loss": float(
            validation.loss_terms.get("rollout_position", validation.total_loss).detach().cpu()
        ),
        "selection_accepted": float(accepted),
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        "best_rollout_validated": 1.0,
        "best_measurement_validated": float(best_measurement is not None),
        **candidate.validation_metrics(),
        **incumbent.checkpoint_metrics(),
        **reference.checkpoint_metrics(prefix="reference_rollout"),
        **_validation_protocol_checkpoint_metrics(config),
        "checkpoint_model_state_hash": checkpoint_model_state_hash,
        "checkpoint_contains_best_rollout_weights": float(accepted),
        "best_rollout_model_state_hash": incumbent_model_state_hash,
        "best_rollout_checkpoint_step": float(incumbent_step),
        "checkpoint_contains_reference_rollout_weights": float(
            checkpoint_model_state_hash == reference_model_state_hash
        ),
        "reference_rollout_model_state_hash": reference_model_state_hash,
        "reference_rollout_checkpoint_step": float(reference_step),
        "rollout_reference_validated": 1.0,
    }
    if best_measurement is not None:
        metrics["best_measurement_loss"] = best_measurement
        metrics["best_measurement_world_position_mae_m"] = best_measurement
    return metrics


def _model_state_hash(model_state: Mapping[str, Any]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes in stable key order."""

    digest = hashlib.sha256()
    for name in sorted(model_state):
        value = model_state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, byteorder="big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, byteorder="big"))
        digest.update(raw)
    return digest.hexdigest()


def _current_model_state_hash(model: OnlineWorldModel) -> str:
    return _model_state_hash(model.state_dict())


def _verified_selector_checkpoint(
    path: Path,
    config: OrpheusConfig,
    *,
    prefix: str,
    expected_model_state_hash: str | None,
    expected_step: int | None,
) -> tuple[_RolloutSelectionMetrics, str, int] | None:
    """Verify that selector metadata and the checkpoint's actual weights agree."""

    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            return None
        metrics = payload.get("metrics")
        model_state = payload.get("model_state")
        if not isinstance(metrics, Mapping) or not isinstance(model_state, Mapping):
            return None
        if not _rollout_validation_protocol_is_compatible(payload, config):
            return None
        selection = _rollout_selection_from_checkpoint(
            metrics,
            config,
            prefix=prefix,
        )
        if selection is None:
            return None
        if float(metrics.get(f"checkpoint_contains_{prefix}_weights", 0.0)) != 1.0:
            return None
        model_state_hash = _model_state_hash(model_state)
        if metrics.get("checkpoint_model_state_hash") != model_state_hash:
            return None
        if metrics.get(f"{prefix}_model_state_hash") != model_state_hash:
            return None
        checkpoint_step = int(payload["step"])
        if int(float(metrics.get(f"{prefix}_checkpoint_step", -1.0))) != checkpoint_step:
            return None
        if expected_model_state_hash is not None and (
            model_state_hash != expected_model_state_hash
        ):
            return None
        if expected_step is not None and checkpoint_step != expected_step:
            return None
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    return selection, model_state_hash, checkpoint_step


def _preserve_resume_selector_checkpoint(
    resume_path: str | Path,
    destination: Path,
    config: OrpheusConfig,
    *,
    prefix: str,
    resume_metrics: Mapping[str, Any],
) -> tuple[_RolloutSelectionMetrics, str, int] | None:
    """Copy a linked selector checkpoint only after weights/provenance verification."""

    expected_hash = resume_metrics.get(f"{prefix}_model_state_hash")
    expected_step_value = resume_metrics.get(f"{prefix}_checkpoint_step")
    if not isinstance(expected_hash, str) or expected_step_value is None:
        return None
    try:
        expected_step = int(float(expected_step_value))
    except (TypeError, ValueError):
        return None
    resumed = Path(resume_path).expanduser().resolve()
    source = resumed.parent / f"{prefix}.pt" if resumed.parent.name == "checkpoints" else resumed
    verified = _verified_selector_checkpoint(
        source,
        config,
        prefix=prefix,
        expected_model_state_hash=expected_hash,
        expected_step=expected_step,
    )
    if verified is None:
        return None
    if source != destination.resolve():
        shutil.copy2(source, destination)
        copied = _verified_selector_checkpoint(
            destination,
            config,
            prefix=prefix,
            expected_model_state_hash=expected_hash,
            expected_step=expected_step,
        )
        if copied is None:
            return None
        verified = copied
    return verified


def _fresh_causal_optimizer_state(
    optimizer: torch.optim.AdamW,
    *,
    learning_rate: float,
    weight_decay: float,
) -> None:
    """Start the causal phase without perception-stage Adam moments."""

    optimizer.state.clear()
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = float(learning_rate)
        parameter_group["weight_decay"] = float(weight_decay)


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
    project = re.sub(r"[^A-Za-z0-9._-]+", "-", config.project.name).strip("-")
    return timestamped_artifact_path(project or "orpheus").name


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
    selected = (
        timestamped_artifact_path(run_name).name if run_name is not None else _new_run_name(config)
    )
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


def _rollout_validation_protocol_is_compatible(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> bool:
    """Require an exact canonical match for the deterministic validation split."""

    metrics = payload.get("metrics")
    checkpoint_config = payload.get("config")
    if not isinstance(metrics, Mapping) or not isinstance(checkpoint_config, Mapping):
        return False
    stored_hash = metrics.get("rollout_validation_protocol_hash")
    if not isinstance(stored_hash, str):
        return False
    try:
        checkpoint_hash = _canonical_hash(
            _rollout_validation_protocol_from_mapping(checkpoint_config)
        )
        requested_hash = _rollout_validation_protocol_hash(config)
    except (KeyError, TypeError, ValueError):
        return False
    return stored_hash == checkpoint_hash == requested_hash


def _rollout_selection_is_compatible(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> bool:
    """Return whether a resumed best score uses this objective's semantics."""

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    try:
        version = float(metrics.get("rollout_selection_metric_version", 0.0))
    except (TypeError, ValueError):
        return False
    if version != _ROLLOUT_SELECTION_METRIC_VERSION:
        return False
    if not _rollout_validation_protocol_is_compatible(payload, config):
        return False
    try:
        restored_selection = _rollout_selection_from_checkpoint(metrics, config)
    except (FloatingPointError, RuntimeError, TypeError, ValueError):
        return False
    return restored_selection is not None


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


def set_closed_loop_trainable_scope(
    model: OnlineWorldModel,
    *,
    scope: str,
) -> None:
    """Restrict adaptation without changing the complete RGB runtime path."""

    if scope == "all":
        model.requires_grad_(True)
        return
    model.requires_grad_(False)
    if scope == "dynamics":
        model.dynamics.requires_grad_(True)
        return
    if scope == "state_dynamics":
        model.dynamics.requires_grad_(True)
        model.updater.requires_grad_(True)
        if model.identifier is not None:
            model.identifier.requires_grad_(True)
        return
    if scope == "state_dynamics_roi":
        model.dynamics.requires_grad_(True)
        model.updater.requires_grad_(True)
        if model.identifier is not None:
            model.identifier.requires_grad_(True)
        rgb_module = model.observation_modules["rgb"]
        roi_updater = getattr(rgb_module, "roi_updater", None)
        if roi_updater is None:
            raise TypeError("RGB module is missing roi_updater")
        roi_updater.requires_grad_(True)
        return
    raise ValueError(
        "closed-loop trainable scope must be 'all', 'dynamics', "
        "'state_dynamics', or 'state_dynamics_roi'"
    )


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
                rollout_anchors_per_window=None,
                compute_future_correction=False,
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


def _aggregate_physical_validation_metrics(
    results: list[TrainingBatchResult],
    config: OrpheusConfig,
) -> dict[str, float]:
    """Derive exact split-level physical metrics from additive batch counts."""

    additive: dict[str, float] = {}
    for result in results:
        for name, value in result.metrics.items():
            if name.startswith("physical_") and (
                name.endswith("_sse")
                or name.endswith("_count")
                or (name.startswith("physical_forecast_") and "_count@" in name)
                or name
                in {
                    "physical_target_object_frames",
                    "physical_matched_object_frames",
                    "physical_identity_switches",
                    "physical_object_frame_associations",
                    "physical_distance_gated_matched_object_frames",
                    "physical_distance_gated_target_object_frames",
                    "physical_distance_gated_predicted_object_frames",
                    "physical_distance_gated_identity_switches",
                    "physical_distance_gated_object_frame_associations",
                }
            ):
                additive[name] = additive.get(name, 0.0) + float(value)
    return physical_validation_metrics(additive, config)


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
        if not bool(torch.isfinite(results[-1].total_loss)):
            phase = "closed-loop" if closed_loop else "measurement"
            raise FloatingPointError(f"nonfinite {phase} validation loss")
        batch_sizes.append(float(batch["rgb"].shape[0]))
    aggregate = _mean_batch_results(results, weights=batch_sizes)
    if closed_loop:
        aggregate.metrics.update(
            _aggregate_physical_validation_metrics(
                results,
                config,
            )
        )
    return aggregate


def _write_run_metadata(
    path: Path,
    *,
    config: OrpheusConfig,
    device_info: DeviceInfo,
    resume_path: str | Path | None,
    initialize_from_path: str | Path | None,
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
        "initialize_from_path": (
            None
            if initialize_from_path is None
            else str(Path(initialize_from_path).expanduser().resolve())
        ),
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_from_config(
    config: OrpheusConfig,
    *,
    run_name: str | None = None,
    resume_path: str | Path | None = None,
    initialize_from_path: str | Path | None = None,
    device_info: DeviceInfo | None = None,
) -> dict[str, Any]:
    """Train RGB measurements, then the causal RGB-only online loop.

    This signature is the public contract used by :mod:`train.py`.
    """

    config.validate()
    if resume_path is not None and initialize_from_path is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
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
    best_rollout_path = checkpoint_directory / "best_rollout.pt"
    reference_rollout_path = checkpoint_directory / "reference_rollout.pt"
    best_measurement_path = checkpoint_directory / "best_measurement.pt"
    last_path = checkpoint_directory / "last.pt"
    resolved_config_path = run_directory / "config.resolved.yaml"
    save_resolved_config(config, resolved_config_path)
    _write_run_metadata(
        run_directory / "run_metadata.json",
        config=config,
        device_info=resolved_device,
        resume_path=resume_path,
        initialize_from_path=initialize_from_path,
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
    initialized_from: str | None = None
    if initialize_from_path is not None:
        source = Path(initialize_from_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {source}")
        initialization_payload = torch.load(
            source,
            map_location=device,
            weights_only=False,
        )
        model_state = initialization_payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("initialization checkpoint does not contain model_state")
        model.load_state_dict(model_state, strict=True)
        initialized_from = str(source)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_step = 0
    best_rollout = math.inf
    best_rollout_selection: _RolloutSelectionMetrics | None = None
    best_rollout_model_state_hash: str | None = None
    best_rollout_step: int | None = None
    reference_rollout_selection: _RolloutSelectionMetrics | None = None
    reference_rollout_model_state_hash: str | None = None
    reference_rollout_step: int | None = None
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
        if not isinstance(resume_metrics, Mapping):
            resume_metrics = {}
        best_rollout_validated = bool(float(resume_metrics.get("best_rollout_validated", 0.0)))
        best_measurement_validated = bool(
            float(resume_metrics.get("best_measurement_validated", 0.0))
        )
        if best_rollout_validated:
            preserved_best = _preserve_resume_selector_checkpoint(
                resume_path,
                best_rollout_path,
                config,
                prefix="best_rollout",
                resume_metrics=resume_metrics,
            )
            preserved_reference = _preserve_resume_selector_checkpoint(
                resume_path,
                reference_rollout_path,
                config,
                prefix="reference_rollout",
                resume_metrics=resume_metrics,
            )
            # Moving-best metadata is useful only when both linked files prove
            # the actual incumbent and the fixed anti-ratcheting reference.
            if preserved_best is None or preserved_reference is None:
                best_rollout_validated = False
            else:
                (
                    best_rollout_selection,
                    best_rollout_model_state_hash,
                    best_rollout_step,
                ) = preserved_best
                (
                    reference_rollout_selection,
                    reference_rollout_model_state_hash,
                    reference_rollout_step,
                ) = preserved_reference
                best_rollout = best_rollout_selection.score
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
    started = time.perf_counter()

    def validate_closed_loop_incumbent(
        *,
        completed_step: int,
        learning_rate: float,
        split: str,
    ) -> tuple[TrainingBatchResult, bool]:
        nonlocal best_rollout, best_rollout_selection, best_rollout_validated
        nonlocal best_rollout_model_state_hash, best_rollout_step
        nonlocal reference_rollout_selection
        nonlocal reference_rollout_model_state_hash, reference_rollout_step

        validation = _validation_loader_result(
            model,
            validation_loader,
            config,
            device=device,
            closed_loop=True,
        )
        validation_metrics = _result_metrics(
            validation,
            learning_rate=learning_rate,
        )
        logger.log(
            step=completed_step,
            split=split,
            metrics=validation_metrics,
        )
        candidate = _rollout_selection_metrics(validation.metrics, config)
        candidate_model_state_hash = _current_model_state_hash(model)
        established_reference = reference_rollout_selection is None
        if established_reference:
            reference_rollout_selection = candidate
            reference_rollout_model_state_hash = candidate_model_state_hash
            reference_rollout_step = completed_step
        if (
            reference_rollout_selection is None
            or reference_rollout_model_state_hash is None
            or reference_rollout_step is None
        ):
            raise AssertionError("closed-loop validation did not establish a fixed reference")
        accepted = best_rollout_selection is None or (
            _rollout_selection_improves(
                candidate,
                best_rollout_selection,
            )
            and _rollout_selection_passes_guardrails(
                candidate,
                reference_rollout_selection,
            )
        )
        if accepted:
            best_rollout_selection = candidate
            best_rollout = candidate.score
            best_rollout_validated = True
            best_rollout_model_state_hash = candidate_model_state_hash
            best_rollout_step = completed_step
        if best_rollout_selection is None:
            raise AssertionError("closed-loop validation did not establish an incumbent")
        if best_rollout_model_state_hash is None or best_rollout_step is None:
            raise AssertionError("closed-loop incumbent is missing weight provenance")
        checkpoint_metrics = _rollout_validation_checkpoint_metrics(
            validation,
            candidate,
            best_rollout_selection,
            reference_rollout_selection,
            config=config,
            accepted=accepted,
            best_measurement=(best_measurement if best_measurement_validated else None),
            checkpoint_model_state_hash=candidate_model_state_hash,
            incumbent_model_state_hash=best_rollout_model_state_hash,
            incumbent_step=best_rollout_step,
            reference_model_state_hash=reference_rollout_model_state_hash,
            reference_step=reference_rollout_step,
        )
        checkpoint_metrics.update(
            {
                f"validation_{name}": float(value)
                for name, value in validation.metrics.items()
                if name.startswith("rollout_position@")
            }
        )
        save_checkpoint(
            checkpoint_directory / f"validation_step_{completed_step:06d}.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            step=completed_step,
            metrics=checkpoint_metrics,
            device=str(device),
        )
        if established_reference:
            save_checkpoint(
                reference_rollout_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
            )
        if accepted:
            save_checkpoint(
                best_rollout_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
            )
        return validation, accepted

    def retained_selector_metrics() -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "best_rollout_validated": float(best_rollout_selection is not None),
            "rollout_reference_validated": float(reference_rollout_selection is not None),
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            "checkpoint_contains_best_rollout_weights": 0.0,
            "checkpoint_contains_reference_rollout_weights": 0.0,
            **_validation_protocol_checkpoint_metrics(config),
        }
        if best_rollout_selection is not None:
            if best_rollout_model_state_hash is None or best_rollout_step is None:
                raise AssertionError("retained incumbent is missing weight provenance")
            metrics.update(best_rollout_selection.checkpoint_metrics())
            metrics["best_rollout_model_state_hash"] = best_rollout_model_state_hash
            metrics["best_rollout_checkpoint_step"] = float(best_rollout_step)
        if reference_rollout_selection is not None:
            if reference_rollout_model_state_hash is None or reference_rollout_step is None:
                raise AssertionError("fixed reference is missing weight provenance")
            metrics.update(
                reference_rollout_selection.checkpoint_metrics(prefix="reference_rollout")
            )
            metrics["reference_rollout_model_state_hash"] = reference_rollout_model_state_hash
            metrics["reference_rollout_checkpoint_step"] = float(reference_rollout_step)
        return metrics

    imported_incumbent = initialize_from_path is not None
    if imported_incumbent:
        _, accepted = validate_closed_loop_incumbent(
            completed_step=start_step,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            split="validation_initialization_incumbent",
        )
        if not accepted:
            raise AssertionError("initialization validation must establish the first incumbent")
        print(
            f"preserved imported runtime as broad closed-loop incumbent (score={best_rollout:.6f})",
            flush=True,
        )
    measurement_handoff_pending = (
        imported_incumbent
        and start_step < config.training.rgb_pretrain_steps < config.training.steps
    )

    for step in range(start_step, config.training.steps):
        restored_measurement_candidate = False
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
                optimizer=None,
                map_location=device,
                restore_rng=False,
                expected_config=config,
            )
            restored_measurement_candidate = True
            print(
                "restored best RGB localization checkpoint for closed-loop handoff "
                f"(world_mae={best_measurement:.6f}m)",
                flush=True,
            )
        if step == config.training.rgb_pretrain_steps:
            _fresh_causal_optimizer_state(
                optimizer,
                learning_rate=(
                    config.training.learning_rate * config.training.closed_loop_learning_rate_scale
                ),
                weight_decay=config.training.weight_decay,
            )
        if step == config.training.rgb_pretrain_steps and measurement_handoff_pending:
            if restored_measurement_candidate:
                _, accepted = validate_closed_loop_incumbent(
                    completed_step=step,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    split="validation_measurement_handoff",
                )
            else:
                accepted = False
            if accepted:
                print(
                    "promoted measurement handoff after broad closed-loop validation "
                    f"(score={best_rollout:.6f})",
                    flush=True,
                )
            else:
                if not best_rollout_path.is_file():
                    raise RuntimeError("imported rollout incumbent checkpoint is missing")
                load_checkpoint(
                    best_rollout_path,
                    model=model,
                    optimizer=None,
                    map_location=device,
                    restore_rng=False,
                    expected_config=config,
                )
                print(
                    "restored imported runtime because the measurement handoff "
                    "did not satisfy broad accuracy guardrails",
                    flush=True,
                )
            measurement_handoff_pending = False
        if step >= config.training.rgb_pretrain_steps and best_rollout_selection is None:
            validate_closed_loop_incumbent(
                completed_step=step,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                split="validation_incumbent",
            )
            print(
                f"established broad closed-loop incumbent (score={best_rollout:.6f})",
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
        if step < config.training.rgb_pretrain_steps:
            model.requires_grad_(True)
        else:
            set_closed_loop_trainable_scope(
                model,
                scope=config.training.closed_loop_trainable_scope,
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
                minimum_rollout_frame_offset=min(
                    max(
                        1,
                        int(round(horizon * config.simulator.frame_rate)),
                    )
                    for horizon in config.evaluation.horizons_seconds
                ),
                long_horizon_probability=(config.training.long_horizon_window_probability),
            )
            result = run_closed_loop_batch(
                model,
                batch,
                config,
                window_start=window_start,
                window_steps=window_steps,
                apply_perturbations=True,
                include_measurement_supervision=True,
                rollout_anchors_per_window=(config.training.rollout_anchors_per_window),
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
        if should_validate and completed_step == config.training.steps:
            # Closed-loop validation can be much more expensive than the final
            # optimiser update. Persist those weights first so an interrupted
            # validation does not discard a successfully completed run.
            prevalidation_metrics = {
                key: float(value)
                for key, value in last_metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            prevalidation_metrics.update(retained_selector_metrics())
            prevalidation_metrics["best_measurement_validated"] = float(best_measurement_validated)
            if best_measurement_validated:
                prevalidation_metrics["best_measurement_loss"] = best_measurement
                prevalidation_metrics["best_measurement_world_position_mae_m"] = best_measurement
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=prevalidation_metrics,
                device=str(device),
            )
        if should_validate:
            if completed_step > config.training.rgb_pretrain_steps:
                validate_closed_loop_incumbent(
                    completed_step=completed_step,
                    learning_rate=learning_rate,
                    split="validation",
                )
            else:
                validation = _validation_loader_result(
                    model,
                    validation_loader,
                    config,
                    device=device,
                    closed_loop=False,
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
                if validation.phase != "rgb_pretrain" or "measurement" not in validation.loss_terms:
                    raise RuntimeError("measurement validation returned an unexpected phase")
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
                            **retained_selector_metrics(),
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
            checkpoint_metrics.update(retained_selector_metrics())
            checkpoint_metrics["best_measurement_validated"] = float(best_measurement_validated)
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
            "best_measurement_validated": float(best_measurement_validated),
            **retained_selector_metrics(),
        }
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
    verified_best_rollout = (
        _verified_selector_checkpoint(
            best_rollout_path,
            config,
            prefix="best_rollout",
            expected_model_state_hash=best_rollout_model_state_hash,
            expected_step=best_rollout_step,
        )
        if best_rollout_validated
        else None
    )
    verified_reference_rollout = (
        _verified_selector_checkpoint(
            reference_rollout_path,
            config,
            prefix="reference_rollout",
            expected_model_state_hash=reference_rollout_model_state_hash,
            expected_step=reference_rollout_step,
        )
        if reference_rollout_selection is not None
        else None
    )
    has_best_rollout_checkpoint = (
        verified_best_rollout is not None and verified_reference_rollout is not None
    )
    has_best_measurement_checkpoint = best_measurement_validated and best_measurement_path.is_file()
    if has_best_rollout_checkpoint:
        # Leave the in-memory runtime at the verified incumbent. ``last.pt``
        # intentionally remains the resumable final iterate.
        load_checkpoint(
            best_rollout_path,
            model=model,
            optimizer=None,
            map_location=device,
            restore_rng=False,
            expected_config=config,
        )
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
        "best_rollout_validated": has_best_rollout_checkpoint,
        "best_measurement_validated": best_measurement_validated,
        "best_rollout_loss": best_rollout if has_best_rollout_checkpoint else None,
        "best_rollout_position_loss": (best_rollout if has_best_rollout_checkpoint else None),
        "best_rollout_position_rmse_m": (
            best_rollout_selection.position_rmse_m if best_rollout_selection is not None else None
        ),
        "best_rollout_velocity_rmse_mps": (
            best_rollout_selection.velocity_rmse_mps if best_rollout_selection is not None else None
        ),
        "best_rollout_target_coverage": (
            best_rollout_selection.target_coverage if best_rollout_selection is not None else None
        ),
        "best_rollout_prediction_precision": (
            best_rollout_selection.prediction_precision
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_collision_f1": (
            best_rollout_selection.collision_f1 if best_rollout_selection is not None else None
        ),
        "best_rollout_id_switch_rate": (
            best_rollout_selection.id_switch_rate if best_rollout_selection is not None else None
        ),
        "best_rollout_position_coverage90": (
            best_rollout_selection.position_coverage90
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_position_calibration_error90": (
            best_rollout_selection.position_calibration_error90
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_horizon_position_rmse_m": (
            best_rollout_selection.horizon_position_rmse_m
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_horizon_forecast_target_coverage": (
            best_rollout_selection.horizon_forecast_target_coverage
            if best_rollout_selection is not None
            else None
        ),
        "rollout_reference_checkpoint": (
            str(reference_rollout_path) if verified_reference_rollout is not None else None
        ),
        "rollout_validation_protocol_hash": _rollout_validation_protocol_hash(config),
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        "best_measurement_loss": (best_measurement if best_measurement_validated else None),
        "best_measurement_world_position_mae_m": (
            best_measurement if best_measurement_validated else None
        ),
        "completed_steps": config.training.steps,
        "model_parameter_count": model_parameter_count,
        "planned_training_episode_draws": (config.training.steps * config.training.batch_size),
        "nominal_dataset_passes": (
            config.training.steps * config.training.batch_size / config.training.train_episodes
        ),
        "train_episodes": config.training.train_episodes,
        "validation_episodes": config.training.validation_episodes,
        "scenario_families": list(config.simulator.scenario_mixture),
        "rgb_pretrain_steps": min(config.training.steps, config.training.rgb_pretrain_steps),
        "closed_loop_steps": max(0, config.training.steps - config.training.rgb_pretrain_steps),
        "device": str(device),
        "precision": resolved_device.precision,
        "elapsed_seconds": elapsed,
        "resumed_from": resumed_from,
        "initialized_from": initialized_from,
        "last_metrics": last_metrics,
        "oracle_runtime_input_used": False,
    }
    atomic_write_text(
        run_directory / "train_summary.json",
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
    )
    return result_payload


__all__ = ["train_from_config"]
