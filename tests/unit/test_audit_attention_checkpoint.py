from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from scripts.audit_attention_checkpoint import (
    _all_finite,
    _optimizer_owner_names,
    audit_checkpoint,
)
from world_model.runtime import OnlineWorldModel
from world_model.training.trainer import _model_state_hash
from world_model.utils.config import load_config


def test_all_finite_checks_nested_torch_and_numpy_values() -> None:
    assert _all_finite(
        {
            "tensor": torch.tensor([1.0, 2.0]),
            "array": np.array([3.0, 4.0]),
            "nested": (5, [6.0]),
        }
    )
    assert not _all_finite({"nested": [torch.tensor(math.inf)]})
    assert not _all_finite(torch.tensor(complex(1.0, math.inf)))
    assert not _all_finite(np.array([0.0, np.nan]))


def test_optimizer_owner_names_uses_serialized_parameter_order() -> None:
    owners, steps = _optimizer_owner_names(
        {
            "param_groups": [{"params": [7, 3, 11]}],
            "state": {
                3: {"step": torch.tensor(4.0)},
                11: {"step": 4},
            },
        },
        ["first", "second", "third"],
    )

    assert owners == ["second", "third"]
    assert steps == [4]


def test_attention_checkpoint_audit_proves_growth_isolation(tmp_path: Path) -> None:
    config_path = Path("configs/attention_pilot_mps.yaml")
    config = load_config(config_path)
    initial_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    trained_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    trained_model.load_state_dict(initial_model.state_dict())
    attention_prefix = "dynamics.attention_interactions."
    with torch.no_grad():
        for name, parameter in trained_model.named_parameters():
            if name.startswith(attention_prefix):
                parameter.add_(0.01)

    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=1.0e-4)
    for name, parameter in trained_model.named_parameters():
        if name.startswith(attention_prefix):
            optimizer.state[parameter] = {
                "step": torch.tensor(128.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }

    initial_state = initial_model.state_dict()
    trained_state = trained_model.state_dict()
    initial_path = tmp_path / "initial.pt"
    checkpoint_path = tmp_path / "step128.pt"
    protected_path = tmp_path / "protected.pt"
    torch.save({"model_state": initial_state}, initial_path)
    torch.save({"model_state": initial_state}, protected_path)
    torch.save(
        {
            "step": 128,
            "specification_version": "test",
            "git": {"commit": "abc", "runtime_source_fingerprint": "def"},
            "model_state": trained_state,
            "optimizer_state": optimizer.state_dict(),
            "metrics": {"checkpoint_model_state_hash": _model_state_hash(trained_state)},
        },
        checkpoint_path,
    )

    report = audit_checkpoint(
        checkpoint_path=checkpoint_path,
        initial_checkpoint_path=initial_path,
        config_path=config_path,
        protected_paths=[protected_path],
        require_all_attention_changed=True,
        require_complete_attention_optimizer_state=True,
        require_protected_checkpoints=True,
    )

    assert report["status"] == "pass"
    assert report["failures"] == []
    assert report["changed_inherited_tensor_count"] == 0
    assert report["changed_attention_tensor_count"] == report["attention_tensor_count"]
    assert report["optimizer_state_attention_only"]
    assert report["optimizer_state_complete_for_attention"]
    assert report["optimizer_steps"] == [128]
    assert report["protected_checkpoint_count"] == 1
    assert report["protected_checkpoints_exactly_initial"]
    assert report["all_serialized_state_finite"]

    unchecked_protection_report = audit_checkpoint(
        checkpoint_path=checkpoint_path,
        initial_checkpoint_path=initial_path,
        config_path=config_path,
        require_protected_checkpoints=True,
    )
    assert unchecked_protection_report["status"] == "fail"
    assert unchecked_protection_report["protected_checkpoint_count"] == 0
    assert unchecked_protection_report["protected_checkpoints_exactly_initial"] is None
    assert "no protected checkpoints were provided" in unchecked_protection_report["failures"]

    missing_prefix_report = audit_checkpoint(
        checkpoint_path=checkpoint_path,
        initial_checkpoint_path=initial_path,
        config_path=config_path,
        attention_prefix="missing.attention.",
    )
    assert missing_prefix_report["status"] == "fail"
    assert (
        "no attention tensors found under the configured prefix"
        in missing_prefix_report["failures"]
    )
