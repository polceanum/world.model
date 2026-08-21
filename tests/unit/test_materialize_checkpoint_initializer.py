from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from scripts.materialize_checkpoint_initializer import materialize_initializer
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpoint_composition import compose_model_state
from world_model.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    validate_exact_resume_state,
)
from world_model.training.trainer import _model_state_hash
from world_model.utils.config import load_config, save_resolved_config


def _file_identity(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _build_contract(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "artifacts"
    run = root / "runs" / "source"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    source_config = load_config("configs/tiny_overfit.yaml")
    source_config_path = run / "config.resolved.yaml"
    save_resolved_config(source_config, source_config_path)

    base_model = OnlineWorldModel.from_config(source_config, device=torch.device("cpu"))
    protected = save_checkpoint(
        root / "runs" / "protected" / "best_rollout.pt",
        model=base_model,
        optimizer=None,
        config=source_config,
        step=0,
        device="cpu",
    )
    witness = save_checkpoint(
        checkpoints / "validation_step_000000.pt",
        model=base_model,
        optimizer=None,
        config=source_config,
        step=0,
        device="cpu",
    )
    donor_model = OnlineWorldModel.from_config(source_config, device=torch.device("cpu"))
    donor_model.load_state_dict(base_model.state_dict())
    donor_parameter = next(
        parameter
        for name, parameter in donor_model.named_parameters()
        if name.startswith("updater.")
    )
    with torch.no_grad():
        donor_parameter.reshape(-1)[0].add_(0.125)
    donor = save_checkpoint(
        checkpoints / "validation_step_000512.pt",
        model=donor_model,
        optimizer=None,
        config=source_config,
        step=512,
        device="cpu",
    )
    run_metadata_path = run / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps({"initialize_from_path": str(protected.resolve())}) + "\n",
        encoding="utf-8",
    )

    target_config = replace(
        source_config,
        model=replace(
            source_config.model,
            filter=replace(
                source_config.model.filter,
                learned_correction_independent_axis_support=True,
            ),
        ),
    )
    target_config.validate()
    target_config_path = tmp_path / "target.yaml"
    save_resolved_config(target_config, target_config_path)

    protected_payload = torch.load(protected, map_location="cpu", weights_only=False)
    witness_payload = torch.load(witness, map_location="cpu", weights_only=False)
    donor_payload = torch.load(donor, map_location="cpu", weights_only=False)
    composed, selected = compose_model_state(
        protected_payload["model_state"],
        donor_payload["model_state"],
        module_prefixes=("updater",),
    )
    changed = [
        name
        for name in composed
        if not torch.equal(composed[name], protected_payload["model_state"][name])
    ]

    def source_entry(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        sha256, byte_count = _file_identity(path)
        return {
            "path": str(path.relative_to(root)),
            "sha256": sha256,
            "byte_count": byte_count,
            "step": payload["step"],
            "model_state_sha256": _model_state_hash(payload["model_state"]),
            "specification_version": payload["specification_version"],
            "simulator_version": payload["simulator_version"],
        }

    source_config_sha, _ = _file_identity(source_config_path)
    run_metadata_sha, _ = _file_identity(run_metadata_path)
    spec = {
        "schema_version": "checkpoint_initializer_composition_v1",
        "role": "weight_only_initializer",
        "sources": {
            "protected_base": source_entry(protected, protected_payload),
            "compatibility_witness": source_entry(witness, witness_payload),
            "donor": source_entry(donor, donor_payload),
        },
        "source_config": {
            "path": str(source_config_path.relative_to(root)),
            "sha256": source_config_sha,
        },
        "source_run_metadata": {
            "path": str(run_metadata_path.relative_to(root)),
            "sha256": run_metadata_sha,
        },
        "module_prefixes": ["updater"],
        "donor_weight": 1.0,
        "expected": {
            "selected_tensor_count": len(selected),
            "selected_element_count": sum(composed[name].numel() for name in selected),
            "changed_tensor_count": len(changed),
            "changed_element_count": sum(composed[name].numel() for name in changed),
            "model_state_sha256": _model_state_hash(composed),
        },
        "target_transfer": {
            "allowed_difference_prefixes": [
                "model.filter.learned_correction_independent_axis_support"
            ],
            "required_values": {"model.filter.learned_correction_independent_axis_support": True},
        },
    }
    spec_path = tmp_path / "composition.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return {
        "root": root,
        "spec": spec,
        "spec_path": spec_path,
        "target_config": target_config,
        "target_config_path": target_config_path,
        "protected": protected,
        "witness": witness,
        "donor": donor,
        "composed": composed,
        "selected": selected,
    }


def test_materializer_writes_a_target_bound_nonresumable_initializer(tmp_path: Path) -> None:
    contract = _build_contract(tmp_path)
    source_identities = {
        name: _file_identity(contract[name]) for name in ("protected", "witness", "donor")
    }

    result = materialize_initializer(
        spec_path=contract["spec_path"],
        target_config_path=contract["target_config_path"],
        output_path=tmp_path / "outputs" / "composed-updater",
        artifact_root=contract["root"],
    )

    output = Path(result["output"])
    initializer = output / "initializer.pt"
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    assert payload["step"] == 0
    assert payload["optimizer_state"] is None
    assert payload["scheduler_state"] is None
    assert payload["config"] == contract["target_config"].to_dict()
    assert payload["artifact_metadata"]["role"] == "weight_only_initializer"
    assert payload["artifact_metadata"]["composition"]["selected_tensors_exact_donor"]
    assert payload["artifact_metadata"]["composition"]["nonselected_tensors_exact_base"]
    assert (
        payload["metrics"]["checkpoint_model_state_hash"]
        == contract["spec"]["expected"]["model_state_sha256"]
    )
    assert _model_state_hash(payload["model_state"]) == _model_state_hash(contract["composed"])
    with pytest.raises(ValueError, match="cannot be exactly resumed"):
        validate_exact_resume_state(payload)

    restored = OnlineWorldModel.from_config(
        contract["target_config"],
        device=torch.device("cpu"),
    )
    load_checkpoint(
        initializer,
        model=restored,
        expected_config=contract["target_config"],
    )
    assert _model_state_hash(restored.state_dict()) == _model_state_hash(contract["composed"])
    assert result["output_checkpoint"]["sha256"] == _file_identity(initializer)[0]
    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))["output_checkpoint"][
            "sha256"
        ]
        == result["output_checkpoint"]["sha256"]
    )
    assert {
        name: _file_identity(contract[name]) for name in ("protected", "witness", "donor")
    } == source_identities


def test_materializer_fails_before_output_on_source_hash_mismatch(tmp_path: Path) -> None:
    contract = _build_contract(tmp_path)
    contract["spec"]["sources"]["donor"]["sha256"] = "0" * 64
    contract["spec_path"].write_text(
        yaml.safe_dump(contract["spec"], sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "bad-outputs" / "rejected"

    with pytest.raises(ValueError, match="donor checkpoint identity mismatch"):
        materialize_initializer(
            spec_path=contract["spec_path"],
            target_config_path=contract["target_config_path"],
            output_path=output,
            artifact_root=contract["root"],
        )

    assert not output.parent.exists()


def test_materializer_rejects_boolean_donor_weight(tmp_path: Path) -> None:
    contract = _build_contract(tmp_path)
    contract["spec"]["donor_weight"] = True
    contract["spec_path"].write_text(
        yaml.safe_dump(contract["spec"], sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "bad-outputs" / "boolean-weight"

    with pytest.raises(ValueError, match="requires donor_weight=1.0"):
        materialize_initializer(
            spec_path=contract["spec_path"],
            target_config_path=contract["target_config_path"],
            output_path=output,
            artifact_root=contract["root"],
        )

    assert not output.parent.exists()
