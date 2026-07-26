"""Atomic trusted-local checkpoint persistence."""

from __future__ import annotations

import os
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from world_model.utils.config import OrpheusConfig
from world_model.utils.version import SPECIFICATION_VERSION, __version__

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
    if checkpoint_config.get("model") != requested["model"]:
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
        "simulator_version": "sphere_world_v1",
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
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return payload
