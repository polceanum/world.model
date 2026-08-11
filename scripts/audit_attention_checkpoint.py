#!/usr/bin/env python3
"""Audit a typed-attention growth checkpoint against its protected initializer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from world_model.runtime import OnlineWorldModel
from world_model.training.trainer import _model_state_hash
from world_model.utils.config import load_config
from world_model.utils.io import atomic_write_text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping: {path}")
    return payload


def _all_finite(value: object) -> bool:
    if isinstance(value, Tensor):
        return (
            bool(torch.isfinite(value).all())
            if value.is_floating_point() or value.is_complex()
            else True
        )
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, Real) and not isinstance(value, bool):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(key) and _all_finite(child) for key, child in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_all_finite(child) for child in value)
    return True


def _model_state(payload: Mapping[str, Any], *, path: Path) -> Mapping[str, Tensor]:
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise TypeError(f"checkpoint model_state must map names to tensors: {path}")
    return state


def _optimizer_owner_names(
    optimizer_state: Mapping[str, Any],
    parameter_names: Sequence[str],
) -> tuple[list[str], list[int]]:
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(states, Mapping):
        raise TypeError("optimizer_state must contain param_groups and state")
    parameter_ids: list[int] = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise TypeError("optimizer param group must contain a params list")
        parameter_ids.extend(int(value) for value in group["params"])
    if len(parameter_ids) != len(parameter_names) or len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("optimizer parameter IDs do not match model parameter ordering")
    name_by_id = dict(zip(parameter_ids, parameter_names, strict=True))
    unknown = sorted(int(value) for value in states if int(value) not in name_by_id)
    if unknown:
        raise ValueError(f"optimizer state contains unknown parameter IDs: {unknown}")
    owners = sorted(name_by_id[int(value)] for value in states)
    steps: set[int] = set()
    for state in states.values():
        if not isinstance(state, Mapping) or "step" not in state:
            raise TypeError("optimizer parameter state is missing step")
        raw_step = state["step"]
        numeric_step = (
            float(raw_step.detach().cpu()) if isinstance(raw_step, Tensor) else float(raw_step)
        )
        if not math.isfinite(numeric_step) or numeric_step < 0 or not numeric_step.is_integer():
            raise ValueError(f"optimizer step is invalid: {numeric_step}")
        steps.add(int(numeric_step))
    return owners, sorted(steps)


def audit_checkpoint(
    *,
    checkpoint_path: Path,
    initial_checkpoint_path: Path,
    config_path: Path,
    protected_paths: Sequence[Path] = (),
    attention_prefix: str = "dynamics.attention_interactions.",
    require_all_attention_changed: bool = False,
    require_complete_attention_optimizer_state: bool = False,
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(checkpoint_path)
    initial = _load_checkpoint(initial_checkpoint_path)
    checkpoint_state = _model_state(checkpoint, path=checkpoint_path)
    initial_state = _model_state(initial, path=initial_checkpoint_path)

    checkpoint_keys = set(checkpoint_state)
    initial_keys = set(initial_state)
    missing_model_keys = sorted(initial_keys - checkpoint_keys)
    extra_model_keys = sorted(checkpoint_keys - initial_keys)
    common_keys = sorted(checkpoint_keys & initial_keys)
    attention_names = [name for name in common_keys if name.startswith(attention_prefix)]
    inherited_names = [name for name in common_keys if not name.startswith(attention_prefix)]
    changed_attention = [
        name
        for name in attention_names
        if not torch.equal(checkpoint_state[name], initial_state[name])
    ]
    changed_inherited = [
        name
        for name in inherited_names
        if not torch.equal(checkpoint_state[name], initial_state[name])
    ]

    config = load_config(config_path)
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    parameter_names = [name for name, _ in model.named_parameters()]
    architecture_state = model.state_dict()
    architecture_model_keys = set(architecture_state)
    architecture_tensor_mismatches = sorted(
        name
        for name in checkpoint_keys & architecture_model_keys
        if checkpoint_state[name].shape != architecture_state[name].shape
        or checkpoint_state[name].dtype != architecture_state[name].dtype
    )
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("checkpoint optimizer_state must be a mapping")
    optimizer_owners, optimizer_steps = _optimizer_owner_names(optimizer_state, parameter_names)
    attention_parameter_names = {
        name for name in parameter_names if name.startswith(attention_prefix)
    }
    optimizer_owner_set = set(optimizer_owners)

    checkpoint_hash = _model_state_hash(checkpoint_state)
    initial_hash = _model_state_hash(initial_state)
    metrics = checkpoint.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    protected_hashes: dict[str, str] = {}
    protected_file_hashes: dict[str, str] = {}
    protected_exact = True
    for path in protected_paths:
        protected_state = _model_state(_load_checkpoint(path), path=path)
        protected_hash = _model_state_hash(protected_state)
        resolved = str(path.resolve())
        protected_hashes[resolved] = protected_hash
        protected_file_hashes[resolved] = _file_sha256(path)
        protected_exact = protected_exact and protected_hash == initial_hash

    failures: list[str] = []
    if not attention_names:
        failures.append("no attention tensors found under the configured prefix")
    if missing_model_keys or extra_model_keys:
        failures.append("checkpoint and initializer model keys differ")
    if checkpoint_keys != architecture_model_keys:
        failures.append("checkpoint model keys differ from configured architecture")
    if architecture_tensor_mismatches:
        failures.append("checkpoint tensor shapes or dtypes differ from configured architecture")
    if changed_inherited:
        failures.append("inherited tensors changed")
    if not optimizer_owner_set.issubset(attention_parameter_names):
        failures.append("optimizer state is not attention-only")
    if require_all_attention_changed and set(changed_attention) != set(attention_names):
        failures.append("not every attention tensor changed")
    if (
        require_complete_attention_optimizer_state
        and optimizer_owner_set != attention_parameter_names
    ):
        failures.append("optimizer state does not cover every attention parameter")
    if not _all_finite(checkpoint):
        failures.append("checkpoint contains nonfinite serialized state")
    if protected_paths and not protected_exact:
        failures.append("a protected checkpoint differs from the initializer")
    stored_hash = metrics.get("checkpoint_model_state_hash")
    if stored_hash is not None and stored_hash != checkpoint_hash:
        failures.append("stored checkpoint model hash does not match tensors")

    git = checkpoint.get("git")
    git = git if isinstance(git, Mapping) else {}
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": _file_sha256(checkpoint_path),
        "initial_checkpoint_path": str(initial_checkpoint_path.resolve()),
        "initial_checkpoint_file_sha256": _file_sha256(initial_checkpoint_path),
        "config_path": str(config_path.resolve()),
        "step": int(checkpoint.get("step", -1)),
        "specification_version": checkpoint.get("specification_version"),
        "checkpoint_git_commit": git.get("commit"),
        "checkpoint_runtime_source_fingerprint": git.get("runtime_source_fingerprint"),
        "checkpoint_model_state_hash": checkpoint_hash,
        "stored_checkpoint_model_state_hash": stored_hash,
        "initial_model_state_hash": initial_hash,
        "model_tensor_count": len(checkpoint_state),
        "attention_tensor_count": len(attention_names),
        "changed_attention_tensor_count": len(changed_attention),
        "changed_attention_tensors": changed_attention,
        "inherited_tensor_count": len(inherited_names),
        "changed_inherited_tensor_count": len(changed_inherited),
        "changed_inherited_tensors": changed_inherited,
        "missing_model_keys": missing_model_keys,
        "extra_model_keys": extra_model_keys,
        "architecture_tensor_mismatches": architecture_tensor_mismatches,
        "all_serialized_state_finite": _all_finite(checkpoint),
        "optimizer_state_owner_count": len(optimizer_owners),
        "optimizer_state_attention_only": optimizer_owner_set.issubset(attention_parameter_names),
        "optimizer_state_complete_for_attention": optimizer_owner_set == attention_parameter_names,
        "optimizer_state_owners": optimizer_owners,
        "optimizer_steps": optimizer_steps,
        "protected_model_state_hashes": protected_hashes,
        "protected_checkpoint_file_sha256": protected_file_hashes,
        "protected_checkpoints_exactly_initial": protected_exact,
        "rollout_validation_protocol_hash": metrics.get("rollout_validation_protocol_hash"),
        "validation_seed_manifest_hash": metrics.get("validation_seed_manifest_hash"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protected", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--attention-prefix",
        default="dynamics.attention_interactions.",
    )
    parser.add_argument("--require-all-attention-changed", action="store_true")
    parser.add_argument(
        "--require-complete-attention-optimizer-state",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_checkpoint(
        checkpoint_path=args.checkpoint,
        initial_checkpoint_path=args.initial_checkpoint,
        config_path=args.config,
        protected_paths=args.protected,
        attention_prefix=args.attention_prefix,
        require_all_attention_changed=args.require_all_attention_changed,
        require_complete_attention_optimizer_state=(
            args.require_complete_attention_optimizer_state
        ),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        atomic_write_text(args.output, rendered)
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
