"""Parity, persistence, and hot-path tests for lean 1.61 training data."""

from __future__ import annotations

import io
import os
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from world_model.runtime import OnlineWorldModel
from world_model.training.dynamic_set_adapter import DynamicSetEpisodeObjectiveAdapter
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_materializer import (
    DynamicSetMaterialization,
    materialize_dynamic_set_episode,
)
from world_model.training.dynamic_set_objectives import dynamic_set_objective
from world_model.training.dynamic_set_protocol import (
    FROZEN_PHYSICAL_MANIFEST_SHA256,
    PHYSICAL_CELL_COUNT,
    physical_manifest,
)
from world_model.training.dynamic_set_trainer import (
    DynamicSetTrainer,
    DynamicSetTrainingMicrobatch,
    dynamic_set_perception_frame_index,
)
from world_model.training.dynamic_set_training_cache import (
    DynamicSetTrainingCache,
    DynamicSetTrainingCacheBinding,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPOSITORY_ROOT / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"


def _binding(**overrides: str) -> DynamicSetTrainingCacheBinding:
    values = {
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "training_manifest_sha256": FROZEN_PHYSICAL_MANIFEST_SHA256["training"],
    }
    values.update(overrides)
    return DynamicSetTrainingCacheBinding(**values)


def _representative_rows() -> tuple[object, ...]:
    manifest = physical_manifest("training")
    rows = tuple(manifest[cell + PHYSICAL_CELL_COUNT * (cell % 12)] for cell in range(22))
    assert {row.cell_index for row in rows} == set(range(22))
    assert {row.contact_origin for row in rows} == {"none", "natural", "action_induced"}
    assert {row.lifecycle_schedule for row in rows} == {
        "none",
        "birth",
        "removal",
        "remove_then_birth",
    }
    assert {row.known_action for row in rows} == {False, True}
    # Two more unique rows complete six B4 groups while retaining every cell.
    return (*rows, manifest[22 * 17], manifest[1 + 22 * 17])


def _assert_exact(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        assert isinstance(left, Tensor) and isinstance(right, Tensor), path
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if is_dataclass(left) or is_dataclass(right):
        assert type(left) is type(right), path
        for item in fields(left):
            _assert_exact(
                getattr(left, item.name),
                getattr(right, item.name),
                path=f"{path}.{item.name}",
            )
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), path
        for name in left:
            _assert_exact(left[name], right[name], path=f"{path}.{name}")
        return
    assert left == right, path


def test_selected_render_is_bit_exact_and_disk_cache_is_reusable(tmp_path: Path) -> None:
    row = next(
        item
        for item in physical_manifest("training")
        if item.contact_origin == "action_induced"
        and item.lifecycle_schedule == "remove_then_birth"
    )
    captured: list[DynamicSetMaterialization] = []

    def full_materializer(value: object) -> DynamicSetMaterialization:
        result = materialize_dynamic_set_episode(value)  # type: ignore[arg-type]
        captured.append(result)
        return result

    cache = DynamicSetTrainingCache(
        tmp_path / "training_cache",
        binding=_binding(),
        full_materializer=full_materializer,
    )
    selected_frames = (0, 5, 15, 22, 28, 55)
    for offset, frame_index in enumerate(selected_frames):
        lean = cache.materialize_for_training(
            row,  # type: ignore[arg-type]
            perception_frame_index=frame_index,
        )
        full = captured[0]
        assert lean.cache_hit is (offset > 0)
        assert torch.equal(lean.episode["rgb"], full.episode["rgb"][frame_index])
        assert torch.equal(lean.episode["depth"], full.episode["depth"][frame_index])
        assert torch.equal(
            lean.episode["labels"]["segmentation_mask"],
            full.episode["labels"]["segmentation_mask"][frame_index],
        )
    assert len(captured) == 1
    assert cache.cold_materializations == 1
    assert cache.warm_hits == len(selected_frames) - 1
    cache_entry = torch.load(
        io.BytesIO(next((tmp_path / "training_cache").rglob("*.pt")).read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    assert set(cache_entry["content"]) == {"timestamps", "camera", "objects", "events"}
    assert "rgb" not in cache_entry["content"]
    assert "depth" not in cache_entry["content"]
    assert "labels" not in cache_entry["content"]

    def forbidden_materializer(_row: object) -> DynamicSetMaterialization:
        raise AssertionError("a valid persistent cache entry must avoid full rematerialization")

    reopened = DynamicSetTrainingCache(
        tmp_path / "training_cache",
        binding=_binding(),
        full_materializer=forbidden_materializer,  # type: ignore[arg-type]
    )
    warm = reopened.materialize_for_training(  # type: ignore[arg-type]
        row,
        perception_frame_index=31,
    )
    assert warm.cache_hit
    assert reopened.cold_materializations == 0
    assert reopened.warm_hits == 1


def test_cache_rejects_binding_tampering_content_tampering_and_hardlinks(
    tmp_path: Path,
) -> None:
    row = physical_manifest("training")[0]
    root = tmp_path / "training_cache"
    cache = DynamicSetTrainingCache(root, binding=_binding())
    cache.materialize_for_training(row, perception_frame_index=0)

    with pytest.raises(ValueError, match="namespace binding differs"):
        DynamicSetTrainingCache(root, binding=_binding(config_sha256="c" * 64))

    entry = next(root.rglob("*.pt"))
    raw = torch.load(io.BytesIO(entry.read_bytes()), map_location="cpu", weights_only=True)
    raw["content"]["objects"]["position"][0, 0, 0] += 0.125
    stream = io.BytesIO()
    torch.save(raw, stream)
    entry.write_bytes(stream.getvalue())
    reopened = DynamicSetTrainingCache(root, binding=_binding())
    with pytest.raises(ValueError, match="content digest differs"):
        reopened.materialize_for_training(row, perception_frame_index=0)

    # Restoring through a separate root also proves a linked inode is rejected
    # before deserialization, even if its payload is otherwise valid.
    linked_root = tmp_path / "linked_cache"
    linked = DynamicSetTrainingCache(linked_root, binding=_binding())
    linked.materialize_for_training(row, perception_frame_index=0)
    linked_entry = next(linked_root.rglob("*.pt"))
    alias = linked_entry.with_name(f"alias-{linked_entry.name}")
    os.link(linked_entry, alias)
    cold_reader = DynamicSetTrainingCache(linked_root, binding=_binding())
    with pytest.raises(OSError, match="single-link"):
        cold_reader.materialize_for_training(row, perception_frame_index=1)


def test_all_22_cells_produce_bit_exact_adapter_inputs_and_losses(tmp_path: Path) -> None:
    rows = _representative_rows()
    captured: list[DynamicSetMaterialization] = []

    def recording_materializer(row: object) -> DynamicSetMaterialization:
        result = materialize_dynamic_set_episode(row)  # type: ignore[arg-type]
        captured.append(result)
        return result

    cache = DynamicSetTrainingCache(
        tmp_path / "training_cache",
        binding=_binding(),
        full_materializer=recording_materializer,
    )
    model = OnlineWorldModel.from_config(load_config(PROFILE), device="cpu")
    model.eval()
    adapter = DynamicSetEpisodeObjectiveAdapter()
    for microbatch_index in range(6):
        batch_rows = rows[4 * microbatch_index : 4 * (microbatch_index + 1)]
        frame_index = dynamic_set_perception_frame_index(0, microbatch_index)
        captured.clear()
        lean = tuple(
            cache.materialize_for_training(  # type: ignore[arg-type]
                row,
                perception_frame_index=frame_index,
            )
            for row in batch_rows
        )
        assert len(captured) == 4
        indices = tuple(range(4 * microbatch_index, 4 * (microbatch_index + 1)))
        full_microbatch = DynamicSetTrainingMicrobatch(
            update_index=0,
            microbatch_index=microbatch_index,
            dataset_indices=indices,
            rows=batch_rows,  # type: ignore[arg-type]
            materializations=tuple(captured),
        )
        lean_microbatch = DynamicSetTrainingMicrobatch(
            update_index=0,
            microbatch_index=microbatch_index,
            dataset_indices=indices,
            rows=batch_rows,  # type: ignore[arg-type]
            materializations=lean,
        )
        full_inputs = adapter.build_objective_inputs(model, full_microbatch)
        lean_inputs = adapter.build_objective_inputs(model, lean_microbatch)
        _assert_exact(full_inputs, lean_inputs, path=f"cell_group[{microbatch_index}]")
        full_losses = dynamic_set_objective(full_inputs.perception, full_inputs.dynamics)
        lean_losses = dynamic_set_objective(lean_inputs.perception, lean_inputs.dynamics)
        _assert_exact(full_losses, lean_losses, path=f"loss_group[{microbatch_index}]")
    assert cache.cold_materializations == 24


def test_trainer_context_reuses_cache_without_repeating_full_materialization(
    tmp_path: Path,
) -> None:
    rows = physical_manifest("training")[:4]
    full_calls: list[int] = []

    def counting_materializer(row: object) -> DynamicSetMaterialization:
        full_calls.append(row.ordinal)  # type: ignore[attr-defined]
        return materialize_dynamic_set_episode(row)  # type: ignore[arg-type]

    cache = DynamicSetTrainingCache(
        tmp_path / "training_cache",
        binding=_binding(),
        full_materializer=counting_materializer,
    )
    # Isolate the trainer's contextual-materializer dispatch without paying for
    # an optimizer/model: _materialize_microbatch depends only on these fields.
    trainer = object.__new__(DynamicSetTrainer)
    trainer.rows = rows
    trainer.materializer = cache
    first = trainer._materialize_microbatch(0, 0, (0, 1, 2, 3))
    second = trainer._materialize_microbatch(1, 0, (0, 1, 2, 3))

    assert full_calls == [row.ordinal for row in rows]
    assert all(not item.cache_hit for item in first.materializations)
    assert all(item.cache_hit for item in second.materializations)
    assert all(
        item.episode["metadata"]["perception_frame_index"] == 0 for item in first.materializations
    )
    assert all(
        item.episode["metadata"]["perception_frame_index"] == 6 for item in second.materializations
    )


def test_warm_cache_materialization_budget_and_storage_projection(tmp_path: Path) -> None:
    rows = _representative_rows()
    full_calls = 0

    def counting_materializer(row: object) -> DynamicSetMaterialization:
        nonlocal full_calls
        full_calls += 1
        return materialize_dynamic_set_episode(row)  # type: ignore[arg-type]

    cache = DynamicSetTrainingCache(
        tmp_path / "training_cache",
        binding=_binding(),
        full_materializer=counting_materializer,
    )
    cold_started = time.perf_counter()
    for row in rows:
        cache.materialize_for_training(  # type: ignore[arg-type]
            row,
            perception_frame_index=0,
        )
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    for row in rows:
        cache.materialize_for_training(  # type: ignore[arg-type]
            row,
            perception_frame_index=17,
        )
    warm_seconds = time.perf_counter() - warm_started

    entry_sizes = [path.stat().st_size for path in (tmp_path / "training_cache").rglob("*.pt")]
    mean_entry_bytes = sum(entry_sizes) / len(entry_sizes)
    projected_training_cache_bytes = mean_entry_bytes * 66_000
    projected_warm_update_seconds = warm_seconds
    assert full_calls == 24
    assert warm_seconds < cold_seconds * 0.20
    assert projected_warm_update_seconds < 1.0
    assert max(entry_sizes) < 128 * 1024
    assert projected_training_cache_bytes < 5 * 1024**3


def test_real_repeated_update_cache_work_is_subordinate_to_optimizer_work(
    tmp_path: Path,
) -> None:
    """Exercise the exact trainer dispatch, forward, backward, and AdamW boundary."""

    rows = physical_manifest("training")[:44]
    full_calls = 0

    def counting_materializer(row: object) -> DynamicSetMaterialization:
        nonlocal full_calls
        full_calls += 1
        return materialize_dynamic_set_episode(row)  # type: ignore[arg-type]

    cache = DynamicSetTrainingCache(
        tmp_path / "training_cache",
        binding=_binding(),
        full_materializer=counting_materializer,
    )
    model = OnlineWorldModel.from_config(load_config(PROFILE), device="cpu")
    trainer = DynamicSetTrainer.from_online_world_model(
        model=model,
        training_rows=rows,
        objective_adapter=DynamicSetEpisodeObjectiveAdapter(),
        resolved_config=load_config(PROFILE),
        source_provenance={
            "commit": "synthetic-cache-performance-test",
            "dirty": False,
            "runtime_source_fingerprint": "f" * 64,
        },
        schedule_seed=161_061,
        materializer=cache,
        test_only_synthetic_manifest=True,
    )
    groups = trainer.schedule.microbatches_for_update(0)
    requested: set[int] = set()
    for microbatch_index, indices in enumerate(groups):
        frame_index = dynamic_set_perception_frame_index(0, microbatch_index)
        for index in indices:
            requested.add(index)
            cache.materialize_for_training(
                rows[index],
                perception_frame_index=frame_index,
            )
    assert full_calls == len(requested)

    warm_before = cache.warm_materialization_seconds
    update_started = time.perf_counter()
    report = trainer.run_update()
    update_seconds = time.perf_counter() - update_started
    warm_cache_seconds = cache.warm_materialization_seconds - warm_before

    assert report.completed_updates == 1
    assert full_calls == len(requested)
    assert cache.cold_materializations == len(requested)
    assert cache.warm_hits >= 24
    assert warm_cache_seconds < 1.0
    assert warm_cache_seconds < update_seconds * 0.50
