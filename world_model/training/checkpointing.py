"""Atomic trusted-local checkpoint persistence."""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
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
    RuntimeConfig,
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
    "attention_residual_enabled": False,
    "attention_width": 128,
    "attention_heads": 4,
    "attention_layers": 4,
    "attention_feed_forward_width": 512,
    "attention_dropout": 0.0,
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
    ("training", "attention_node_grad_clip_norm"): None,
    ("training", "attention_node_output_grad_clip_norm"): None,
    ("training", "attention_collision_output_grad_clip_norm"): None,
    ("training", "attention_force_output_grad_clip_norm"): None,
    ("training", "attention_impulse_grad_clip_norm"): None,
    ("training", "attention_impulse_output_grad_clip_norm"): None,
    ("training", "minimum_interaction_gradient_retention"): None,
    ("training", "closed_loop_learning_rate_schedule"): "constant",
    ("training", "closed_loop_learning_rate_warmup_steps"): 0,
    ("training", "closed_loop_learning_rate_cosine_decay_steps"): None,
    ("training", "closed_loop_learning_rate_minimum_scale"): 0.1,
}

_RUNTIME_SOURCE_ROOT_FILES = frozenset({"train.py"})
_RUNTIME_SOURCE_SUFFIXES = frozenset({".py", ".pyi"})
_TYPED_ATTENTION_PREFIX = "dynamics.attention_interactions."
_TYPED_ATTENTION_BLOCK_PATTERN = re.compile(
    rf"^{re.escape(_TYPED_ATTENTION_PREFIX)}blocks\.(\d+)\.(.+)$"
)
_IDENTITY_ATTENTION_BLOCK_OUTPUTS = frozenset(
    {
        "attention.out_proj.weight",
        "attention.out_proj.bias",
        "feed_forward.output.weight",
    }
)


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
    filter_config = model.get("filter")
    if isinstance(filter_config, Mapping):
        normalized_filter = dict(filter_config)
        # Checkpoints before specification 1.19 used the unanchored learned
        # residual path. Missing is therefore exactly legacy False, not the
        # semantics selected by a newer training profile.
        normalized_filter.setdefault("innovation_anchored_correction", False)
        model["filter"] = normalized_filter
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


def _runtime_checkpoint_semantics(value: object) -> object:
    """Normalize absent disabled pool controls in historical checkpoints.

    These controls create no candidate state and do not alter inference while
    the pool is disabled. Missing fields therefore mean the explicit disabled
    default. Once enabled, every policy field remains a strict runtime
    semantic, including the evidence horizon and axis composition.
    """

    if not isinstance(value, Mapping):
        return value
    runtime = dict(value)
    defaults = RuntimeConfig()
    policy_fields = (
        "hypothesis_pool_enabled",
        "hypothesis_evidence_horizons_seconds",
        "hypothesis_axis_independent_axes",
        "hypothesis_axis_prior_strength",
        "hypothesis_evidence_decay",
        "hypothesis_timestamp_tolerance_seconds",
    )
    for field_name in policy_fields:
        runtime.setdefault(field_name, getattr(defaults, field_name))
    if not runtime["hypothesis_pool_enabled"]:
        for field_name in policy_fields[1:]:
            runtime[field_name] = getattr(defaults, field_name)
    return runtime


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
    if _runtime_checkpoint_semantics(checkpoint_config.get("runtime")) != (
        _runtime_checkpoint_semantics(requested["runtime"])
    ):
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


def _validate_attention_depth_growth_config(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> None:
    """Require identical runtime semantics except for appended block count."""

    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint does not contain a resolved config mapping")
    requested = config.to_dict()
    checkpoint_model = _model_checkpoint_semantics(checkpoint_config.get("model"))
    requested_model = _model_checkpoint_semantics(requested["model"])
    if not isinstance(checkpoint_model, Mapping) or not isinstance(requested_model, Mapping):
        raise ValueError("attention depth growth requires resolved model mappings")
    checkpoint_model = deepcopy(dict(checkpoint_model))
    requested_model = deepcopy(dict(requested_model))
    checkpoint_dynamics = checkpoint_model.get("dynamics")
    requested_dynamics = requested_model.get("dynamics")
    if not isinstance(checkpoint_dynamics, Mapping) or not isinstance(requested_dynamics, Mapping):
        raise ValueError("attention depth growth requires resolved dynamics mappings")
    checkpoint_dynamics = dict(checkpoint_dynamics)
    requested_dynamics = dict(requested_dynamics)
    checkpoint_layers = checkpoint_dynamics.get("attention_layers")
    requested_layers = requested_dynamics.get("attention_layers")
    if (
        isinstance(checkpoint_layers, bool)
        or not isinstance(checkpoint_layers, int)
        or isinstance(requested_layers, bool)
        or not isinstance(requested_layers, int)
        or requested_layers <= checkpoint_layers
    ):
        raise ValueError("attention depth growth requires a strictly larger integer target depth")
    checkpoint_dynamics["attention_layers"] = requested_layers
    checkpoint_model["dynamics"] = checkpoint_dynamics
    requested_model["dynamics"] = requested_dynamics
    mismatches: list[str] = []
    if checkpoint_model != requested_model:
        mismatches.append("model except attention_layers")
    if _runtime_checkpoint_semantics(checkpoint_config.get("runtime")) != (
        _runtime_checkpoint_semantics(requested["runtime"])
    ):
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
        raise ValueError(
            "attention depth growth configuration is incompatible for: " + ", ".join(mismatches)
        )


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
    _assert_finite_tensor_tree(payload["model_state"], root="model_state")
    if payload["optimizer_state"] is not None:
        _assert_finite_tensor_tree(
            payload["optimizer_state"],
            root="optimizer_state",
        )
        _assert_valid_optimizer_steps(
            payload["optimizer_state"],
            root="optimizer_state",
        )
    if payload["scheduler_state"] is not None:
        _assert_finite_tensor_tree(
            payload["scheduler_state"],
            root="scheduler_state",
        )
    return payload


def _assert_finite_tensor_tree(value: Any, *, root: str) -> None:
    """Reject non-finite floating state in a recursively nested tensor tree."""

    pending: list[tuple[str, Any]] = [(root, value)]
    named_tensors: list[tuple[str, torch.Tensor]] = []
    while pending:
        name, item = pending.pop()
        if isinstance(item, torch.Tensor):
            if item.is_floating_point() or item.is_complex():
                named_tensors.append((name, item))
            continue
        if isinstance(item, Mapping):
            pending.extend((f"{name}.{key}", child) for key, child in item.items())
            continue
        if isinstance(item, (list, tuple)):
            pending.extend((f"{name}[{index}]", child) for index, child in enumerate(item))
    tensors_by_device: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in named_tensors:
        tensors_by_device.setdefault(tensor.device, []).append((name, tensor))
    for named_device_tensors in tensors_by_device.values():
        finite = torch.stack(
            [torch.isfinite(tensor).all() for _, tensor in named_device_tensors]
        ).all()
        if bool(finite):
            continue
        for name, tensor in named_device_tensors:
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"tensor state {name!r} contains NaN or Inf")
        raise AssertionError("nonfinite tensor group did not identify its offending tensor")


def _assert_valid_optimizer_steps(value: Any, *, root: str) -> None:
    """Require every optimizer step counter to be scalar, finite, and nonnegative."""

    pending: list[tuple[str, Any]] = [(root, value)]
    tensor_steps_by_device: dict[torch.device, list[tuple[str, Tensor]]] = {}
    while pending:
        name, item = pending.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_name = f"{name}.{key}"
                if key == "step":
                    if isinstance(child, Tensor):
                        if child.numel() != 1:
                            raise FloatingPointError(
                                f"optimizer step {child_name!r} must be scalar"
                            )
                        tensor_steps_by_device.setdefault(child.device, []).append(
                            (child_name, child)
                        )
                    elif isinstance(child, Real) and not isinstance(child, bool):
                        if not math.isfinite(float(child)) or child < 0:
                            raise FloatingPointError(f"optimizer step {child_name!r} is invalid")
                    else:
                        raise FloatingPointError(f"optimizer step {child_name!r} is not numeric")
                pending.append((child_name, child))
        elif isinstance(item, (list, tuple)):
            pending.extend((f"{name}[{index}]", child) for index, child in enumerate(item))
    for named_steps in tensor_steps_by_device.values():
        valid = torch.stack(
            [
                (torch.isfinite(optimizer_step) & (optimizer_step >= 0)).all()
                for _, optimizer_step in named_steps
            ]
        ).all()
        if bool(valid):
            continue
        for name, optimizer_step in named_steps:
            if not bool((torch.isfinite(optimizer_step) & (optimizer_step >= 0)).all()):
                raise FloatingPointError(f"optimizer step {name!r} is invalid")
        raise AssertionError("invalid optimizer step group did not identify its counter")


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
    _assert_finite_tensor_tree(payload["model_state"], root="model_state")
    if payload.get("optimizer_state") is not None:
        _assert_finite_tensor_tree(
            payload["optimizer_state"],
            root="optimizer_state",
        )
        _assert_valid_optimizer_steps(
            payload["optimizer_state"],
            root="optimizer_state",
        )
    if payload.get("scheduler_state") is not None:
        _assert_finite_tensor_tree(
            payload["scheduler_state"],
            root="scheduler_state",
        )
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


def _identity_attention_depth_growth_state(
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
    *,
    missing_keys: tuple[str, ...],
    prefix: str,
) -> tuple[dict[str, Tensor], tuple[int, ...]] | None:
    """Prepare an exact-function handoff for appended attention blocks.

    A pre-norm residual block is an identity when both residual branches emit
    exact zero.  The internal query/key/value and SwiGLU input weights may keep
    their ordinary initialization; zero attention and feed-forward output
    projections make the block output exactly equal to its input while leaving
    a trainable path into those projections on the first optimizer update.

    Only contiguous, appended depth is supported. Width changes, holes,
    reordered blocks, or any missing non-block attention tensor return ``None``
    and remain hard loader failures.
    """

    if prefix != _TYPED_ATTENTION_PREFIX:
        return None

    def block_keys(state: Mapping[str, Tensor]) -> dict[int, set[str]]:
        grouped: dict[int, set[str]] = {}
        for key in state:
            match = _TYPED_ATTENTION_BLOCK_PATTERN.fullmatch(key)
            if match is None:
                continue
            grouped.setdefault(int(match.group(1)), set()).add(key)
        return grouped

    source_blocks = block_keys(source_state)
    target_blocks = block_keys(target_state)
    source_indices = tuple(sorted(source_blocks))
    target_indices = tuple(sorted(target_blocks))
    if (
        not source_indices
        or source_indices != tuple(range(len(source_indices)))
        or target_indices != tuple(range(len(target_indices)))
        or len(target_indices) <= len(source_indices)
        or target_indices[: len(source_indices)] != source_indices
    ):
        return None

    grown_indices = target_indices[len(source_indices) :]
    expected_missing = {key for index in grown_indices for key in target_blocks[index]}
    if set(missing_keys) != expected_missing:
        return None

    prepared = dict(source_state)
    for key in sorted(expected_missing):
        match = _TYPED_ATTENTION_BLOCK_PATTERN.fullmatch(key)
        if match is None:
            raise AssertionError("validated attention block key did not parse")
        suffix = match.group(2)
        value = target_state[key].detach().clone()
        if suffix in _IDENTITY_ATTENTION_BLOCK_OUTPUTS:
            value.zero_()
        prepared[key] = value
    for index in grown_indices:
        suffixes = {
            _TYPED_ATTENTION_BLOCK_PATTERN.fullmatch(key).group(2)  # type: ignore[union-attr]
            for key in target_blocks[index]
        }
        if not suffixes >= _IDENTITY_ATTENTION_BLOCK_OUTPUTS:
            return None
    return prepared, grown_indices


def load_model_weights(
    path: str | Path,
    *,
    model: nn.Module,
    expected_config: OrpheusConfig | None = None,
    allowed_missing_prefixes: tuple[str, ...] = (),
    architecture_growth_config: OrpheusConfig | None = None,
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
    _assert_finite_tensor_tree(payload["model_state"], root="model_state")
    if expected_config is not None:
        validate_checkpoint_config(payload, expected_config)
    source_state = payload["model_state"]
    target_state = model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    unexpected_keys = sorted(source_keys - target_keys)
    missing_keys = sorted(target_keys - source_keys)
    if unexpected_keys:
        raise RuntimeError(
            "initialization checkpoint has unexpected model keys: " + ", ".join(unexpected_keys)
        )
    disallowed_missing = [
        key
        for key in missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing:
        raise RuntimeError(
            "initialization checkpoint is missing required model keys: "
            + ", ".join(disallowed_missing)
        )
    incompatible_shapes = [
        (
            key,
            tuple(source_state[key].shape),
            tuple(target_state[key].shape),
        )
        for key in sorted(source_keys & target_keys)
        if source_state[key].shape != target_state[key].shape
    ]
    if incompatible_shapes:
        details = ", ".join(
            f"{key}: checkpoint {source_shape}, model {target_shape}"
            for key, source_shape, target_shape in incompatible_shapes
        )
        raise RuntimeError(
            "initialization checkpoint has incompatible model tensor shapes: " + details
        )
    prepared_state = dict(source_state)
    identity_grown_attention_blocks: tuple[int, ...] = ()
    remaining_missing = set(missing_keys)
    for prefix in allowed_missing_prefixes:
        source_has_prefix = any(key.startswith(prefix) for key in source_keys)
        missing_under_prefix = tuple(key for key in missing_keys if key.startswith(prefix))
        if not source_has_prefix or not missing_under_prefix:
            continue
        growth = _identity_attention_depth_growth_state(
            prepared_state,
            target_state,
            missing_keys=missing_under_prefix,
            prefix=prefix,
        )
        if growth is None:
            raise RuntimeError(
                "initialization checkpoint contains only part of an allowed new "
                f"module prefix {prefix!r}; partial architecture growth is not "
                "function-preserving: " + ", ".join(missing_under_prefix)
            )
        if architecture_growth_config is None:
            raise RuntimeError(
                "function-preserving attention depth growth requires an explicit "
                "target configuration"
            )
        _validate_attention_depth_growth_config(payload, architecture_growth_config)
        prepared_state, grown_blocks = growth
        identity_grown_attention_blocks += grown_blocks
        remaining_missing.difference_update(missing_under_prefix)
    incompatible = model.load_state_dict(prepared_state, strict=False)
    if incompatible.unexpected_keys or sorted(incompatible.missing_keys) != sorted(
        remaining_missing
    ):
        raise RuntimeError("model state changed during validated weight loading")
    payload["weight_load_missing_keys"] = tuple(missing_keys)
    payload["identity_grown_attention_blocks"] = identity_grown_attention_blocks
    return payload
