"""Atomic trusted-local checkpoint persistence."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from world_model.utils.config import (
    AssociationConfig,
    LifecycleConfig,
    OrpheusConfig,
    RGBConfig,
)
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
    "temporal_velocity_post_event_min_samples",
    "temporal_velocity_change_point_enabled",
    "temporal_velocity_change_point_minimum_speed",
    "temporal_velocity_change_point_minimum_delta",
    "temporal_velocity_change_point_strong_delta",
    "temporal_velocity_change_point_require_contact_mode",
    "temporal_velocity_change_point_gate",
    "temporal_velocity_change_point_linear_weights",
    "temporal_velocity_change_point_linear_bias",
    "temporal_velocity_change_point_mlp_hidden_weights",
    "temporal_velocity_change_point_mlp_hidden_bias",
    "temporal_velocity_change_point_mlp_output_weights",
    "temporal_velocity_change_point_mlp_output_bias",
    "temporal_velocity_change_point_probability_threshold",
    "temporal_velocity_change_point_minimum_interval_samples",
    "temporal_velocity_outgoing_proposal_enabled",
    "temporal_velocity_outgoing_proposal_hidden_weights",
    "temporal_velocity_outgoing_proposal_hidden_bias",
    "temporal_velocity_outgoing_proposal_output_weights",
    "temporal_velocity_outgoing_proposal_output_bias",
    "temporal_velocity_outgoing_proposal_variance",
    "temporal_velocity_outgoing_proposal_maximum_delta",
    "temporal_velocity_lateral_intervention_enabled",
    "temporal_velocity_lateral_intervention_hidden_weights",
    "temporal_velocity_lateral_intervention_hidden_bias",
    "temporal_velocity_lateral_intervention_output_weights",
    "temporal_velocity_lateral_intervention_output_bias",
    "temporal_velocity_lateral_intervention_variance_floor",
    "temporal_velocity_lateral_intervention_variance_ceiling",
    "temporal_velocity_lateral_intervention_gain_power",
    "temporal_velocity_lateral_intervention_maximum_delta",
    "temporal_velocity_gravity_intervention_enabled",
    "temporal_velocity_gravity_intervention_hidden_weights",
    "temporal_velocity_gravity_intervention_hidden_bias",
    "temporal_velocity_gravity_intervention_output_weights",
    "temporal_velocity_gravity_intervention_output_bias",
    "temporal_velocity_gravity_intervention_variance_floor",
    "temporal_velocity_gravity_intervention_variance_ceiling",
    "temporal_velocity_gravity_intervention_gain_power",
    "temporal_velocity_gravity_intervention_maximum_delta",
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

_DYNAMICS_LEGACY_DEFAULTS = {
    # These are the actual resolver/config defaults used before each field was
    # persisted. They deliberately do not track today's corrected reference
    # physics defaults; otherwise an old checkpoint would be mislabelled as
    # runtime-compatible after its contact semantics changed.
    "contact_margin": 1.0e-3,
    "boundary_contact_tolerance": 1.0e-3,
    "contact_confidence_sigma": 0.25,
    "pair_collision_speed_epsilon": 1.0e-4,
    "boundary_collision_speed_epsilon": 0.1,
}
_ASSOCIATION_MIGRATION_DEFAULT_FIELDS = ("minimum_measurement_confidence",)
_LIFECYCLE_MIGRATION_DEFAULT_FIELDS = (
    "max_occluded_steps",
    "birth_confirmation_distance_m",
)

_RESUME_ALLOWED_CONTROL_PATHS = {
    ("project", "name"),
    ("project", "output_dir"),
    ("training", "steps"),
    ("training", "checkpoint_every"),
    ("training", "log_every"),
    ("training", "num_workers"),
}

_RESUME_LEGACY_DEFAULTS: dict[tuple[str, ...], Any] = {
    # These fields were introduced after the first sustained campaign.  The
    # values below reproduce the behavior of a checkpoint where they are
    # absent; they are not blindly replaced with today's defaults.
    ("simulator", "ensured_pair_lateral_offset_range"): [0.0, 0.0],
    ("training", "normalize_rollout_axes_over_configured_horizons"): False,
    ("training", "joint_collision_long_horizon_sampling"): False,
    ("training", "minimum_rollout_age_steps"): 0,
    ("training", "validation_rollout_anchors_per_episode"): None,
    ("training", "loss_weights", "rollout_nll"): 0.0,
    ("device", "closed_loop_preference"): "same",
    # Checkpoints predating the switch ran the proposal transformer natively
    # on the selected device.
    ("device", "global_detector_cpu_on_mps"): False,
}

_RUNTIME_SOURCE_ROOT_FILES = frozenset({"train.py"})
_RUNTIME_SOURCE_SUFFIXES = frozenset({".py", ".pyi"})


def _is_runtime_source_path(path: Path) -> bool:
    """Return whether ``path`` can alter the numerical training runtime.

    The active resolved configuration is embedded in every checkpoint and
    compared separately.  This fingerprint therefore covers executable Python
    source, while deliberately excluding documentation, tests, and unrelated
    configuration files so those can be committed during a long campaign
    without making an otherwise exact continuation impossible.
    """

    if path.name in _RUNTIME_SOURCE_ROOT_FILES and len(path.parts) == 1:
        return True
    return (
        len(path.parts) > 1
        and path.parts[0] == "world_model"
        and path.suffix in _RUNTIME_SOURCE_SUFFIXES
    )


def _hash_runtime_source_paths(
    root: Path,
    *,
    tracked: bytes,
    untracked: bytes,
) -> str:
    """Hash current executable contents independently of Git commit identity."""

    digest = hashlib.sha256()
    root = root.resolve()
    encoded_paths = {
        encoded_path
        for encoded_path in (*tracked.split(b"\0"), *untracked.split(b"\0"))
        if encoded_path
        and _is_runtime_source_path(Path(encoded_path.decode("utf-8", errors="surrogateescape")))
    }
    for encoded_path in sorted(encoded_paths):
        relative = Path(encoded_path.decode("utf-8", errors="surrogateescape"))
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError:
            continue
        digest.update(len(encoded_path).to_bytes(8, byteorder="big"))
        digest.update(encoded_path)
        if not source.is_file():
            digest.update(b"\0missing")
            continue
        digest.update(b"\0file")
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _model_checkpoint_semantics(value: object) -> object:
    """Normalize fields absent from legacy checkpoints to explicit semantics.

    RGB controls that alter measurement means, variances, or supported state
    fields are runtime semantics even when they add no trainable parameters.
    They therefore remain in the compatibility comparison.  Only genuinely
    missing RGB fields are filled with the defaults those checkpoints used.
    Parameter-free runtime-invariant fields are deliberately migrated to their
    current validated defaults; their source checkpoint remains otherwise
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
        for coefficient_field in (
            "temporal_velocity_change_point_linear_weights",
            "temporal_velocity_change_point_mlp_hidden_weights",
            "temporal_velocity_change_point_mlp_hidden_bias",
            "temporal_velocity_change_point_mlp_output_weights",
            "temporal_velocity_outgoing_proposal_hidden_weights",
            "temporal_velocity_outgoing_proposal_hidden_bias",
            "temporal_velocity_outgoing_proposal_output_weights",
            "temporal_velocity_lateral_intervention_hidden_weights",
            "temporal_velocity_lateral_intervention_hidden_bias",
            "temporal_velocity_lateral_intervention_output_weights",
            "temporal_velocity_lateral_intervention_output_bias",
            "temporal_velocity_gravity_intervention_hidden_weights",
            "temporal_velocity_gravity_intervention_hidden_bias",
            "temporal_velocity_gravity_intervention_output_weights",
            "temporal_velocity_gravity_intervention_output_bias",
        ):
            coefficients = normalized_rgb.get(coefficient_field)
            if isinstance(coefficients, list):
                normalized_rgb[coefficient_field] = tuple(coefficients)
        model["rgb"] = normalized_rgb
    dynamics = model.get("dynamics")
    if isinstance(dynamics, Mapping):
        normalized_dynamics = dict(dynamics)
        for field_name, historical_value in _DYNAMICS_LEGACY_DEFAULTS.items():
            normalized_dynamics.setdefault(field_name, historical_value)
        model["dynamics"] = normalized_dynamics
    association = model.get("association")
    if isinstance(association, Mapping):
        normalized_association = dict(association)
        defaults = AssociationConfig()
        for field_name in _ASSOCIATION_MIGRATION_DEFAULT_FIELDS:
            normalized_association.setdefault(field_name, getattr(defaults, field_name))
        model["association"] = normalized_association
    lifecycle = model.get("lifecycle")
    if isinstance(lifecycle, Mapping):
        normalized_lifecycle = dict(lifecycle)
        defaults = LifecycleConfig()
        for field_name in _LIFECYCLE_MIGRATION_DEFAULT_FIELDS:
            normalized_lifecycle.setdefault(field_name, getattr(defaults, field_name))
        model["lifecycle"] = normalized_lifecycle
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


def _set_missing_path(
    mapping: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    parent: dict[str, Any] = mapping
    for component in path[:-1]:
        child = parent.get(component)
        if not isinstance(child, Mapping):
            return
        copied_child = dict(child)
        parent[component] = copied_child
        parent = copied_child
    parent.setdefault(path[-1], deepcopy(value))


def _remove_path(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
    parent: dict[str, Any] = mapping
    for component in path[:-1]:
        child = parent.get(component)
        if not isinstance(child, Mapping):
            return
        copied_child = dict(child)
        parent[component] = copied_child
        parent = copied_child
    parent.pop(path[-1], None)


def _normalized_resume_config(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(value))
    model = normalized.get("model")
    if isinstance(model, Mapping):
        normalized["model"] = _model_checkpoint_semantics(model)
    for path, default in _RESUME_LEGACY_DEFAULTS.items():
        _set_missing_path(normalized, path, default)
    for path in _RESUME_ALLOWED_CONTROL_PATHS:
        _remove_path(normalized, path)
    return normalized


def _resume_config_differences(
    checkpoint: object,
    requested: object,
    *,
    path: str = "",
) -> list[str]:
    if isinstance(checkpoint, Mapping) and isinstance(requested, Mapping):
        differences: list[str] = []
        for key in sorted(set(checkpoint) | set(requested), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in checkpoint:
                differences.append(
                    f"{child_path}: checkpoint=<missing>, requested={requested[key]!r}"
                )
            elif key not in requested:
                differences.append(
                    f"{child_path}: checkpoint={checkpoint[key]!r}, requested=<missing>"
                )
            else:
                differences.extend(
                    _resume_config_differences(
                        checkpoint[key],
                        requested[key],
                        path=child_path,
                    )
                )
        return differences
    if checkpoint != requested:
        return [f"{path}: checkpoint={checkpoint!r}, requested={requested!r}"]
    return []


def validate_training_resume_config(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> None:
    """Require an optimizer/RNG resume to preserve its training semantics.

    Weight-only curriculum transfer intentionally uses ``--initialize-from``.
    Exact ``--resume`` may extend the step budget or alter operational
    checkpoint/logging controls, but it must retain the execution device,
    data stream, objective, optimizer, simulator, validation protocol, and
    project seed. Device/backend changes use ``--initialize-from`` because
    their kernels and RNG streams are not bitwise-equivalent continuations.
    """

    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint does not contain a resolved config mapping")
    stored = _normalized_resume_config(checkpoint_config)
    requested = _normalized_resume_config(config.to_dict())
    differences = _resume_config_differences(stored, requested)
    if differences:
        shown = "; ".join(differences[:8])
        remainder = len(differences) - 8
        if remainder > 0:
            shown += f"; ... and {remainder} more difference(s)"
        raise ValueError(
            "checkpoint is not an exact training resume; incompatible fields: "
            f"{shown}. Change only run controls or use --initialize-from for "
            "a weights-only transfer."
        )


def capture_git_metadata(root: Path) -> dict[str, Any]:
    """Capture immutable source provenance for one running process."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        runtime_source_fingerprint = _hash_runtime_source_paths(
            root,
            tracked=tracked,
            untracked=untracked,
        )
        digest = hashlib.sha256()
        for component in (commit.encode("utf-8"), status, diff):
            digest.update(len(component).to_bytes(8, byteorder="big"))
            digest.update(component)
        root = root.resolve()
        for encoded_path in sorted(path for path in untracked.split(b"\0") if path):
            relative = Path(encoded_path.decode("utf-8", errors="surrogateescape"))
            source = (root / relative).resolve()
            try:
                source.relative_to(root)
            except ValueError:
                continue
            digest.update(len(encoded_path).to_bytes(8, byteorder="big"))
            digest.update(encoded_path)
            if source.is_file():
                with source.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
        dirty = bool(status)
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty": None,
            "worktree_fingerprint": None,
            "runtime_source_fingerprint": None,
        }
    return {
        "commit": commit,
        "dirty": dirty,
        "worktree_fingerprint": digest.hexdigest(),
        "runtime_source_fingerprint": runtime_source_fingerprint,
    }


def _mps_rng_state(device: str) -> Tensor | None:
    if torch.device(device).type != "mps":
        return None
    backend = getattr(torch.backends, "mps", None)
    mps = getattr(torch, "mps", None)
    if backend is None or mps is None or not hasattr(mps, "get_rng_state"):
        return None
    try:
        return mps.get_rng_state().cpu()
    except RuntimeError:
        # A checkpoint may be inspected or constructed on a host where the
        # saved device is unavailable.  Actual MPS training cannot reach this
        # path without a working generator.
        if backend.is_available():
            raise
        return None


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    config: OrpheusConfig,
    step: int,
    metrics: Mapping[str, Any],
    device: str,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete local checkpoint payload."""

    root = Path(__file__).resolve().parents[2]
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
            "torch_mps": _mps_rng_state(device),
        },
        "project_version": __version__,
        "specification_version": SPECIFICATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "device": device,
        "precision": "float32",
        # Long-running Python processes keep executing the source loaded at
        # launch even if the worktree advances.  Callers such as the trainer
        # capture this once and pass it to every save.
        "git": (
            deepcopy(dict(source_provenance))
            if source_provenance is not None
            else capture_git_metadata(root)
        ),
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
    metrics: Mapping[str, Any] | None = None,
    device: str = "cpu",
    source_provenance: Mapping[str, Any] | None = None,
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
            source_provenance=source_provenance,
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
        mps = getattr(torch, "mps", None)
        mps_backend = getattr(torch.backends, "mps", None)
        if (
            mps is not None
            and mps_backend is not None
            and hasattr(mps, "set_rng_state")
            and rng.get("torch_mps") is not None
        ):
            try:
                mps.set_rng_state(rng["torch_mps"].cpu())
            except RuntimeError:
                # Loading an MPS checkpoint for CPU-only evaluation should not
                # fail merely because its accelerator RNG cannot be restored.
                if mps_backend.is_available():
                    raise
    return payload


def load_model_weights(
    path: str | Path,
    *,
    model: nn.Module,
    expected_config: OrpheusConfig | None = None,
) -> dict[str, Any]:
    """Load only model weights while keeping the full payload in CPU memory.

    Weight-only curriculum transfers do not need optimizer/RNG tensors. Loading
    a full checkpoint directly onto MPS/CUDA can otherwise duplicate a large
    Adam state on the accelerator and cause an avoidable phase-boundary OOM.
    ``load_state_dict`` performs the bounded tensor copies into the model's
    existing device.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {"model_state", "step", "config", "specification_version"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if expected_config is not None:
        validate_checkpoint_config(payload, expected_config)
    model.load_state_dict(payload["model_state"])
    return payload
