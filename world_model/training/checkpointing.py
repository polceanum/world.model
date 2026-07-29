"""Atomic trusted-local checkpoint persistence."""

from __future__ import annotations

import os
import random
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from world_model.utils.config import DynamicsConfig, OrpheusConfig, RGBConfig
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION, __version__

_SIMULATOR_COMPATIBILITY_FIELDS = (
    "type",
    "image_size",
    "frame_rate",
    "physics_rate",
    "world_bounds",
    "radius_range",
    "gravity",
    "known_camera_pose",
)

_RGB_LEGACY_DEFAULT_FIELDS = (
    "temporal_velocity_enabled",
    "temporal_velocity_history_size",
    "temporal_velocity_min_samples",
    "temporal_velocity_min_dt",
    "temporal_velocity_variance_scale",
    "temporal_velocity_variance_floor",
    "temporal_velocity_variance_ceiling",
    "temporal_velocity_lateral_only",
    "temporal_velocity_post_event_gravity_axis_enabled",
    "temporal_velocity_unobserved_variance",
    "temporal_velocity_reset_on_collision",
    "temporal_velocity_max_age_steps",
    "temporal_velocity_post_event_max_samples",
    "temporal_velocity_measurement_position_blend",
    "temporal_velocity_position_innovation_coupling",
    "temporal_position_enabled",
    "temporal_position_min_samples",
    "temporal_position_robust_threshold",
    "temporal_position_variance_scale",
    "temporal_position_variance_floor",
    "temporal_position_variance_ceiling",
    "temporal_position_depth_only",
    "structured_disc_center_enabled",
    "structured_disc_threshold",
    "structured_disc_min_pixels",
    "structured_disc_max_assignment_distance",
    "structured_disc_center_std_pixels",
    "structured_disc_fast_depth_enabled",
    "structured_disc_depth_relative_std",
    "structured_disc_depth_outlier_relative_threshold",
    "structured_disc_depth_outlier_variance_scale",
    "structured_disc_position_confidence",
)

_DYNAMICS_MIGRATION_DEFAULT_FIELDS = ("contact_confidence_sigma",)


def _model_checkpoint_semantics(value: object) -> object:
    """Normalize fields absent from legacy checkpoints to explicit semantics.

    RGB controls that alter measurement means, variances, or supported state
    fields are runtime semantics even when they add no trainable parameters.
    They therefore remain in the compatibility comparison.  Only genuinely
    missing RGB fields are filled with the defaults those checkpoints used.
    The parameter-free contact-confidence field is deliberately migrated to
    the current validated default; its source checkpoint remains otherwise
    immutable and the migration is recorded in project memory.
    """

    if not isinstance(value, Mapping):
        return value
    model = deepcopy(dict(value))
    rgb = model.get("rgb")
    if isinstance(rgb, Mapping):
        normalized_rgb = dict(rgb)
        defaults = RGBConfig()
        for field_name in _RGB_LEGACY_DEFAULT_FIELDS:
            normalized_rgb.setdefault(field_name, getattr(defaults, field_name))
        model["rgb"] = normalized_rgb
    dynamics = model.get("dynamics")
    if isinstance(dynamics, Mapping):
        normalized_dynamics = dict(dynamics)
        defaults = DynamicsConfig()
        for field_name in _DYNAMICS_MIGRATION_DEFAULT_FIELDS:
            normalized_dynamics.setdefault(field_name, getattr(defaults, field_name))
        model["dynamics"] = normalized_dynamics
    return model


def validate_checkpoint_config(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> None:
    """Reject a checkpoint whose runtime semantics differ from ``config``.

    Sequence length, split seeds, and evaluation/training budgets may change.
    Model structure, runtime fusion semantics, and simulator quantities embedded
    in dynamics/perception modules must remain identical.
    """

    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint does not contain a resolved config mapping")
    requested = config.to_dict()
    mismatches: list[str] = []
    if _model_checkpoint_semantics(checkpoint_config.get("model")) != (
        _model_checkpoint_semantics(requested["model"])
    ):
        mismatches.append("model")
    if checkpoint_config.get("runtime") != requested["runtime"]:
        mismatches.append("runtime")
    checkpoint_simulator = checkpoint_config.get("simulator")
    if not isinstance(checkpoint_simulator, Mapping):
        mismatches.append("simulator")
    else:
        requested_simulator = requested["simulator"]
        for field_name in _SIMULATOR_COMPATIBILITY_FIELDS:
            if checkpoint_simulator.get(field_name) != requested_simulator.get(field_name):
                mismatches.append(f"simulator.{field_name}")
    if mismatches:
        raise ValueError("checkpoint configuration is incompatible for: " + ", ".join(mismatches))


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    config: OrpheusConfig,
    step: int,
    metrics: dict[str, float],
    device: str,
) -> dict[str, Any]:
    """Build a complete local checkpoint payload."""

    root = Path(config.source_path).parents[1] if config.source_path else Path.cwd()
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "step": int(step),
        "config": config.to_dict(),
        "metrics": dict(metrics),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "project_version": __version__,
        "specification_version": SPECIFICATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "device": device,
        "precision": "float32",
        "git": _git_metadata(root),
    }
    return payload


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any = None,
    config: OrpheusConfig,
    step: int,
    metrics: dict[str, float] | None = None,
    device: str = "cpu",
) -> Path:
    """Atomically save a trusted-local checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
            metrics=metrics or {},
            device=device,
        ),
        temporary,
    )
    os.replace(temporary, target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
    expected_config: OrpheusConfig | None = None,
) -> dict[str, Any]:
    """Load a trusted local checkpoint after optional semantic validation."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    required = {"model_state", "step", "config", "specification_version"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if expected_config is not None:
        validate_checkpoint_config(payload, expected_config)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng and "rng" in payload:
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        # ``map_location`` follows model placement, but PyTorch's default RNG
        # is always a CPU generator and rejects an MPS/CUDA ByteTensor.
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
    return payload
