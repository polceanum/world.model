from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import world_model.training.dynamic_set_resources as dynamic_set_resources
from world_model.runtime.online_world_model import OnlineWorldModel
from world_model.training.dynamic_set_adapter import DynamicSetEpisodeObjectiveAdapter
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_protocol import canonical_sha256, physical_manifest
from world_model.training.dynamic_set_resources import (
    FRESH_RESOURCE_WORKLOAD_ROWS,
    FRESH_RESOURCE_WORKLOAD_SHA256,
    FreshWorkerResourceEvidence,
    learned_state_tensor_bytes,
    measure_fresh_worker_resources,
    model_tensor_inventory,
    runtime_tensor_bytes,
)
from world_model.training.dynamic_set_scene import DYNAMIC_SET_FRAMES
from world_model.training.dynamic_set_trainer import (
    DynamicSetTrainer,
    dynamic_set_model_state_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPOSITORY_ROOT / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"


def _resource_evidence(**overrides: object) -> FreshWorkerResourceEvidence:
    values: dict[str, object] = {
        "request_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "model_state_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "source_sha256": "5" * 64,
        "workload_sha256": FRESH_RESOURCE_WORKLOAD_SHA256,
        "worker_pid": 123,
        "baseline_current_rss_bytes": 100,
        "model_current_rss_bytes": 200,
        "maximum_current_rss_bytes": 300,
        "baseline_peak_rss_bytes": 400,
        "maximum_peak_rss_bytes": 900,
        "peak_rss_delta_bytes": 500,
        "perception_latency_samples_seconds": [0.01]
        * (len(FRESH_RESOURCE_WORKLOAD_ROWS) * DYNAMIC_SET_FRAMES),
        "rollout_latency_samples_seconds": [0.02] * len(FRESH_RESOURCE_WORKLOAD_ROWS),
        "learned_weight_bytes": 1_024,
        "persistent_tensor_bytes": 2_048,
        "hidden_preload_tensor_bytes": 0,
    }
    values.update(overrides)
    return FreshWorkerResourceEvidence.create(**values)


def test_fresh_resource_evidence_round_trips_and_uses_true_peak_rss() -> None:
    evidence = _resource_evidence()

    assert FreshWorkerResourceEvidence.from_mapping(evidence.to_mapping()) == evidence
    assert evidence.resource_metrics.process_rss_bytes == 900
    assert evidence.resource_metrics.learned_weight_bytes == 1_024
    assert evidence.resource_metrics.persistent_tensor_bytes == 2_048

    with pytest.raises(ValueError, match="sample population"):
        replace(evidence, perception_latency_samples_seconds=(0.01,)).validate()
    with pytest.raises(ValueError, match="inconsistent|digest"):
        FreshWorkerResourceEvidence.from_mapping(
            {**evidence.to_mapping(), "maximum_peak_rss_bytes": 901}
        )
    with pytest.raises(ValueError, match="current/peak RSS evidence"):
        replace(evidence, maximum_current_rss_bytes=901).validate()


def test_current_rss_uses_no_child_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_child_process(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"current RSS spawned a child process: {args!r}, {kwargs!r}")

    monkeypatch.setattr(dynamic_set_resources.subprocess, "run", forbidden_child_process)

    current_rss = dynamic_set_resources.process_current_rss_bytes()

    assert type(current_rss) is int
    assert 0 < current_rss <= dynamic_set_resources.process_peak_rss_bytes()


class _InventoryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.register_buffer("persistent", torch.ones(4, dtype=torch.float32))
        self.register_buffer(
            "nonpersistent",
            torch.ones(100_000, dtype=torch.float32),
            persistent=False,
        )
        captured = torch.ones(1_024, dtype=torch.float32)
        self.callback = lambda: captured


def test_tensor_inventory_counts_parameters_buffers_and_closure_storage_once() -> None:
    model = _InventoryModel()
    inventory = model_tensor_inventory(model)

    assert learned_state_tensor_bytes(model) == 3 * 4
    assert inventory.parameter_bytes == 3 * 4
    assert inventory.persistent_buffer_bytes == 4 * 4
    assert inventory.nonpersistent_buffer_bytes == 100_000 * 4
    assert inventory.unregistered_bytes == 1_024 * 4
    assert any("__closure__" in path for path in inventory.unregistered_paths)
    assert runtime_tensor_bytes(model) == (4 + 100_000 + 1_024) * 4
    assert not inventory.non_cpu_paths
    assert not inventory.non_float32_floating_paths


def test_tensor_inventory_rejects_non_float32_floating_state() -> None:
    model = nn.Module()
    model.register_buffer("wrong_dtype", torch.ones(2, dtype=torch.float64), persistent=False)

    inventory = model_tensor_inventory(model)

    assert inventory.non_float32_floating_paths


@pytest.mark.slow
def test_real_fresh_worker_checkpoint_round_trip(tmp_path: Path) -> None:
    config = load_config(PROFILE)
    source = {
        "schema": "dynamic_set_resource_test_source_v1",
        "source_sha256": "6" * 64,
    }
    model = OnlineWorldModel.from_config(config, device="cpu")
    trainer = DynamicSetTrainer.from_online_world_model(
        model=model,
        training_rows=physical_manifest("training"),
        objective_adapter=DynamicSetEpisodeObjectiveAdapter(),
        resolved_config=config,
        source_provenance=source,
        schedule_seed=161_061,
    )
    checkpoint = tmp_path / "resource-checkpoint.pt"
    torch.save(trainer.checkpoint_payload(), checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_state_sha256 = dynamic_set_model_state_sha256(model.state_dict())
    config_sha256 = canonical_sha256(config.to_dict())
    source_sha256 = canonical_sha256(source)

    evidence = measure_fresh_worker_resources(
        checkpoint_path=checkpoint,
        resolved_config=config,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_model_state_sha256=model_state_sha256,
        expected_config_sha256=config_sha256,
        expected_source_sha256=source_sha256,
    )

    assert evidence.checkpoint_sha256 == checkpoint_sha256
    assert evidence.model_state_sha256 == model_state_sha256
    assert evidence.config_sha256 == config_sha256
    assert evidence.source_sha256 == source_sha256
    assert evidence.workload_sha256 == FRESH_RESOURCE_WORKLOAD_SHA256
    assert evidence.worker_pid != 0
    assert len(evidence.perception_latency_samples_seconds) == 224
    assert len(evidence.rollout_latency_samples_seconds) == 4
    assert evidence.learned_weight_bytes == learned_state_tensor_bytes(model)
    assert evidence.persistent_tensor_bytes <= 256 * 1024
