#!/usr/bin/env python3
"""Materialize one provenance-bound, weights-only checkpoint initializer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from world_model.runtime import OnlineWorldModel
from world_model.training.checkpoint_composition import compose_model_state
from world_model.training.checkpointing import (
    CapturedCheckpoint,
    capture_checkpoint_snapshot,
    capture_git_metadata,
    save_checkpoint,
    validate_checkpoint_config,
)
from world_model.training.trainer import _model_state_hash
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import load_config, save_resolved_config
from world_model.utils.io import atomic_write_text

_SCHEMA_VERSION = "checkpoint_initializer_composition_v1"
_ROLE = "weight_only_initializer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Composition contract YAML")
    parser.add_argument("--target-config", required=True, help="Resolved target runtime config")
    parser.add_argument("--output", required=True, help="New timestamped artifact directory")
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root against which source-artifact paths in the spec are resolved",
    )
    return parser.parse_args()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_yaml_bytes(path: Path) -> tuple[dict[str, Any], str, int]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"YAML artifact is empty: {path}")
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML artifact must contain a mapping: {path}")
    return loaded, hashlib.sha256(raw).hexdigest(), len(raw)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sha256(value: object, *, name: str) -> str:
    text = _string(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _repository_relative_path(root: Path, value: object, *, name: str) -> Path:
    relative = Path(_string(value, name=name))
    if relative.is_absolute():
        raise ValueError(f"{name} must be relative to --artifact-root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes --artifact-root") from error
    return resolved


def _model_state(payload: Mapping[str, Any], *, name: str) -> Mapping[str, Tensor]:
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise TypeError(f"{name}.model_state must map tensor names to tensors")
    return state


def _load_captured_checkpoint(captured: CapturedCheckpoint, *, name: str) -> dict[str, Any]:
    payload = torch.load(captured.snapshot_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} checkpoint payload must be a mapping")
    _model_state(payload, name=name)
    return payload


def _verify_checkpoint_identity(
    captured: CapturedCheckpoint,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    expected_sha = _sha256(expected.get("sha256"), name=f"sources.{name}.sha256")
    expected_bytes = _positive_integer(
        expected.get("byte_count"),
        name=f"sources.{name}.byte_count",
    )
    expected_step = _nonnegative_integer(
        expected.get("step"),
        name=f"sources.{name}.step",
    )
    expected_model_hash = _sha256(
        expected.get("model_state_sha256"),
        name=f"sources.{name}.model_state_sha256",
    )
    actual_step = payload.get("step")
    actual_model_hash = _model_state_hash(_model_state(payload, name=name))
    mismatches: list[str] = []
    if captured.sha256 != expected_sha:
        mismatches.append(f"sha256={captured.sha256}")
    if captured.byte_count != expected_bytes:
        mismatches.append(f"byte_count={captured.byte_count}")
    if actual_step != expected_step:
        mismatches.append(f"step={actual_step!r}")
    if actual_model_hash != expected_model_hash:
        mismatches.append(f"model_state_sha256={actual_model_hash}")
    for field in ("specification_version", "simulator_version"):
        expected_value = expected.get(field)
        if expected_value is not None and payload.get(field) != expected_value:
            mismatches.append(f"{field}={payload.get(field)!r}")
    if mismatches:
        raise ValueError(f"{name} checkpoint identity mismatch: " + ", ".join(mismatches))
    stored_git = payload.get("git")
    return {
        "path": str(captured.source_path),
        "sha256": captured.sha256,
        "byte_count": captured.byte_count,
        "step": expected_step,
        "model_state_sha256": actual_model_hash,
        "specification_version": payload.get("specification_version"),
        "simulator_version": payload.get("simulator_version"),
        "source_provenance": (
            deepcopy(dict(stored_git)) if isinstance(stored_git, Mapping) else None
        ),
    }


def _require_equal_model_states(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
    *,
    context: str,
) -> None:
    if set(left) != set(right):
        raise ValueError(f"{context} model-state schemas differ")
    unequal = [name for name in sorted(left) if not torch.equal(left[name], right[name])]
    if unequal:
        raise ValueError(f"{context} model states differ: {unequal[:8]}")


def _leaf_differences(
    source: object,
    target: object,
    *,
    path: str = "",
) -> list[dict[str, Any]]:
    if isinstance(source, Mapping) and isinstance(target, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(source) | set(target), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in source:
                differences.append(
                    {"path": child_path, "source": "<missing>", "target": target[key]}
                )
            elif key not in target:
                differences.append(
                    {"path": child_path, "source": source[key], "target": "<missing>"}
                )
            else:
                differences.extend(_leaf_differences(source[key], target[key], path=child_path))
        return differences
    if source != target:
        return [{"path": path, "source": source, "target": target}]
    return []


def _dotted_value(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"target config requirement path is missing: {dotted_path}")
        value = value[component]
    return value


def _validate_target_config_transfer(
    source_config: Any,
    target_config: Any,
    transfer: Mapping[str, Any],
    *,
    witness_payload: Mapping[str, Any],
    donor_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    allowed_raw = transfer.get("allowed_difference_prefixes")
    if not isinstance(allowed_raw, Sequence) or isinstance(allowed_raw, (str, bytes)):
        raise TypeError("target_transfer.allowed_difference_prefixes must be a sequence")
    allowed = tuple(
        _string(value, name="target_transfer.allowed_difference_prefixes[]")
        for value in allowed_raw
    )
    requirements = _mapping(
        transfer.get("required_values", {}),
        name="target_transfer.required_values",
    )
    target_mapping = target_config.to_dict()
    for dotted_path, expected in requirements.items():
        if _dotted_value(target_mapping, str(dotted_path)) != expected:
            raise ValueError(
                f"target config requirement failed for {dotted_path}: expected {expected!r}"
            )
    differences = _leaf_differences(source_config.to_dict(), target_mapping)
    disallowed = [
        difference
        for difference in differences
        if not any(
            difference["path"] == prefix or difference["path"].startswith(f"{prefix}.")
            for prefix in allowed
        )
    ]
    if disallowed:
        paths = [difference["path"] for difference in disallowed]
        raise ValueError(f"target config has unapproved source differences: {paths[:12]}")

    # The only permitted model-runtime transfer in this production contract is
    # the source-axis gate. Project/training/device controls are non-model
    # experiment controls and are separately enumerated above.
    legacy_projection = replace(
        target_config,
        model=replace(
            target_config.model,
            filter=replace(
                target_config.model.filter,
                learned_correction_independent_axis_support=False,
            ),
        ),
    )
    validate_checkpoint_config(witness_payload, legacy_projection)
    validate_checkpoint_config(donor_payload, legacy_projection)
    return differences


def _composition_audit(
    base: Mapping[str, Tensor],
    donor: Mapping[str, Tensor],
    composed: Mapping[str, Tensor],
    selected: Sequence[str],
) -> dict[str, Any]:
    selected_set = set(selected)
    selected_not_donor = [name for name in selected if not torch.equal(composed[name], donor[name])]
    unselected_not_base = [
        name
        for name in sorted(composed)
        if name not in selected_set and not torch.equal(composed[name], base[name])
    ]
    if selected_not_donor or unselected_not_base:
        raise RuntimeError(
            "checkpoint composition ownership failed: "
            f"selected_not_donor={selected_not_donor[:8]}, "
            f"unselected_not_base={unselected_not_base[:8]}"
        )
    changed = [name for name in sorted(composed) if not torch.equal(composed[name], base[name])]
    return {
        "selected_tensor_names": list(selected),
        "selected_tensor_count": len(selected),
        "selected_element_count": sum(composed[name].numel() for name in selected),
        "changed_tensor_names": changed,
        "changed_tensor_count": len(changed),
        "changed_element_count": sum(composed[name].numel() for name in changed),
        "nonselected_tensors_exact_base": True,
        "selected_tensors_exact_donor": True,
        "model_state_sha256": _model_state_hash(composed),
    }


def _verify_composition_expectations(
    audit: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    fields = (
        "selected_tensor_count",
        "selected_element_count",
        "changed_tensor_count",
        "changed_element_count",
    )
    mismatches = [
        f"{field}={audit[field]!r}"
        for field in fields
        if audit[field] != _nonnegative_integer(expected.get(field), name=f"expected.{field}")
    ]
    expected_hash = _sha256(
        expected.get("model_state_sha256"),
        name="expected.model_state_sha256",
    )
    if audit["model_state_sha256"] != expected_hash:
        mismatches.append(f"model_state_sha256={audit['model_state_sha256']}")
    if mismatches:
        raise ValueError("composed initializer expectation mismatch: " + ", ".join(mismatches))


def materialize_initializer(
    *,
    spec_path: str | Path,
    target_config_path: str | Path,
    output_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Validate, compose, and save one new initializer artifact directory."""

    repository_root = Path(__file__).resolve().parents[1]
    root = Path(artifact_root).expanduser().resolve()
    spec_source = Path(spec_path).expanduser().resolve()
    spec, spec_sha256, spec_byte_count = _load_yaml_bytes(spec_source)
    if spec.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"composition spec schema_version must be {_SCHEMA_VERSION!r}")
    if spec.get("role") != _ROLE:
        raise ValueError(f"composition spec role must be {_ROLE!r}")
    sources = _mapping(spec.get("sources"), name="sources")
    source_entries = {
        name: _mapping(sources.get(name), name=f"sources.{name}")
        for name in ("protected_base", "compatibility_witness", "donor")
    }
    source_paths = {
        name: _repository_relative_path(
            root,
            entry.get("path"),
            name=f"sources.{name}.path",
        )
        for name, entry in source_entries.items()
    }
    source_config_entry = _mapping(spec.get("source_config"), name="source_config")
    source_config_path = _repository_relative_path(
        root,
        source_config_entry.get("path"),
        name="source_config.path",
    )
    run_metadata_entry = _mapping(spec.get("source_run_metadata"), name="source_run_metadata")
    run_metadata_path = _repository_relative_path(
        root,
        run_metadata_entry.get("path"),
        name="source_run_metadata.path",
    )
    target_source = Path(target_config_path).expanduser().resolve()

    with ExitStack() as stack:
        captured_sources = {
            name: stack.enter_context(capture_checkpoint_snapshot(path))
            for name, path in source_paths.items()
        }
        captured_source_config = stack.enter_context(
            capture_checkpoint_snapshot(source_config_path)
        )
        captured_run_metadata = stack.enter_context(capture_checkpoint_snapshot(run_metadata_path))
        captured_target_config = stack.enter_context(capture_checkpoint_snapshot(target_source))
        payloads = {
            name: _load_captured_checkpoint(captured, name=name)
            for name, captured in captured_sources.items()
        }
        identities = {
            name: _verify_checkpoint_identity(
                captured_sources[name],
                payloads[name],
                source_entries[name],
                name=name,
            )
            for name in source_entries
        }
        expected_source_config_sha = _sha256(
            source_config_entry.get("sha256"),
            name="source_config.sha256",
        )
        if captured_source_config.sha256 != expected_source_config_sha:
            raise ValueError("source config SHA-256 does not match composition contract")
        raw_source_config = yaml.safe_load(
            captured_source_config.snapshot_path.read_text(encoding="utf-8")
        )
        if not isinstance(raw_source_config, Mapping):
            raise TypeError("source resolved config must contain a mapping")
        witness_config = payloads["compatibility_witness"].get("config")
        donor_config = payloads["donor"].get("config")
        if raw_source_config != witness_config or witness_config != donor_config:
            raise ValueError("source config, compatibility witness, and donor configs differ")
        source_config = load_config(captured_source_config.snapshot_path)

        expected_run_metadata_sha = _sha256(
            run_metadata_entry.get("sha256"),
            name="source_run_metadata.sha256",
        )
        if captured_run_metadata.sha256 != expected_run_metadata_sha:
            raise ValueError("source run metadata SHA-256 does not match composition contract")
        run_metadata = json.loads(captured_run_metadata.snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(run_metadata, Mapping):
            raise TypeError("source run metadata must contain a mapping")
        initialized_from = run_metadata.get("initialize_from_path")
        if not isinstance(initialized_from, str) or (
            Path(initialized_from).expanduser().resolve() != source_paths["protected_base"]
        ):
            raise ValueError("source run metadata does not name the protected base initializer")

        target_config = load_config(captured_target_config.snapshot_path)
        config_differences = _validate_target_config_transfer(
            source_config,
            target_config,
            _mapping(spec.get("target_transfer"), name="target_transfer"),
            witness_payload=payloads["compatibility_witness"],
            donor_payload=payloads["donor"],
        )

        base_state = _model_state(payloads["protected_base"], name="protected_base")
        witness_state = _model_state(
            payloads["compatibility_witness"],
            name="compatibility_witness",
        )
        donor_state = _model_state(payloads["donor"], name="donor")
        _require_equal_model_states(
            base_state,
            witness_state,
            context="protected base and compatibility witness",
        )
        modules_raw = spec.get("module_prefixes")
        if not isinstance(modules_raw, Sequence) or isinstance(modules_raw, (str, bytes)):
            raise TypeError("module_prefixes must be a sequence")
        modules = tuple(_string(value, name="module_prefixes[]") for value in modules_raw)
        donor_weight = spec.get("donor_weight")
        if (
            isinstance(donor_weight, bool)
            or not isinstance(donor_weight, Real)
            or donor_weight != 1.0
        ):
            raise ValueError("durable initializer composition requires donor_weight=1.0")
        composed_state, selected = compose_model_state(
            base_state,
            donor_state,
            module_prefixes=modules,
            donor_weight=1.0,
        )
        audit = _composition_audit(base_state, donor_state, composed_state, selected)
        _verify_composition_expectations(
            audit,
            _mapping(spec.get("expected"), name="expected"),
        )
        model = OnlineWorldModel.from_config(target_config, device=torch.device("cpu"))
        model.load_state_dict(composed_state, strict=True)
        loaded_model_hash = _model_state_hash(model.state_dict())
        if loaded_model_hash != audit["model_state_sha256"]:
            raise RuntimeError("target model changed the composed tensor state while loading")

        source_config_identity = {
            "path": str(captured_source_config.source_path),
            "sha256": captured_source_config.sha256,
            "byte_count": captured_source_config.byte_count,
        }
        run_metadata_identity = {
            "path": str(captured_run_metadata.source_path),
            "sha256": captured_run_metadata.sha256,
            "byte_count": captured_run_metadata.byte_count,
        }
        target_config_identity = {
            "path": str(captured_target_config.source_path),
            "sha256": captured_target_config.sha256,
            "byte_count": captured_target_config.byte_count,
            "resolved_config_sha256": _canonical_sha256(target_config.to_dict()),
        }

    output = timestamped_artifact_path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    tool_source_provenance = capture_git_metadata(repository_root)
    artifact_metadata = {
        "schema_version": _SCHEMA_VERSION,
        "role": _ROLE,
        "exact_resume_supported": False,
        "composition_spec": {
            "path": str(spec_source),
            "sha256": spec_sha256,
            "byte_count": spec_byte_count,
        },
        "sources": identities,
        "source_config": source_config_identity,
        "source_run_metadata": run_metadata_identity,
        "target_config": target_config_identity,
        "target_config_differences": config_differences,
        "module_prefixes": list(modules),
        "donor_weight": 1.0,
        "composition": audit,
    }
    initializer_path = save_checkpoint(
        output / "initializer.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        config=target_config,
        step=0,
        metrics={
            "checkpoint_state_role": _ROLE,
            "exact_resume_supported": 0.0,
            "checkpoint_model_state_hash": audit["model_state_sha256"],
            "composition_selected_tensor_count": float(audit["selected_tensor_count"]),
            "composition_changed_tensor_count": float(audit["changed_tensor_count"]),
        },
        device="cpu",
        source_provenance=tool_source_provenance,
        artifact_metadata=artifact_metadata,
    )
    with capture_checkpoint_snapshot(initializer_path) as captured_output:
        output_identity = {
            "path": str(initializer_path),
            "sha256": captured_output.sha256,
            "byte_count": captured_output.byte_count,
            "model_state_sha256": audit["model_state_sha256"],
        }
    resolved_config_path = output / "config.resolved.yaml"
    save_resolved_config(target_config, resolved_config_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": _SCHEMA_VERSION,
        "role": _ROLE,
        "tool_source_provenance": tool_source_provenance,
        "artifact_metadata": artifact_metadata,
        "output_checkpoint": output_identity,
    }
    manifest_path = output / "manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    for path in (initializer_path, resolved_config_path, manifest_path):
        path.chmod(0o444)
    return {**manifest, "output": str(output)}


def main() -> int:
    args = parse_args()
    result = materialize_initializer(
        spec_path=args.spec,
        target_config_path=args.target_config,
        output_path=args.output,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "output": result["output"],
                "checkpoint_sha256": result["output_checkpoint"]["sha256"],
                "model_state_sha256": result["output_checkpoint"]["model_state_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
