from __future__ import annotations

import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from world_model.training.dynamic_set_evaluation_cache import (
    DynamicSetDevelopmentEvaluationCache,
    DynamicSetDevelopmentEvaluationCacheBinding,
    _physical_content,
    _planning_content,
    _tree_digest,
)
from world_model.training.dynamic_set_materializer import materialize_dynamic_set_episode
from world_model.training.dynamic_set_planning_materializer import materialize_planning_task
from world_model.training.dynamic_set_protocol import (
    physical_manifest,
    planning_manifest,
)
from world_model.training.qualification_core import canonical_sha256


def _binding(physical_rows: tuple[Any, ...], planning_rows: tuple[Any, ...], **changes: str):
    values = {
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "physical_manifest_sha256": canonical_sha256([asdict(row) for row in physical_rows]),
        "planning_manifest_sha256": canonical_sha256([asdict(row) for row in planning_rows]),
        "evaluator_sha256": "c" * 64,
    }
    values.update(changes)
    return DynamicSetDevelopmentEvaluationCacheBinding(**values).validate()


def _representative_rows() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    all_physical = physical_manifest("development")
    physical = (
        next(row for row in all_physical if not row.contact and not row.dynamic_membership),
        next(
            row
            for row in all_physical
            if row.contact and row.known_action and row.contact_origin == "action_induced"
        ),
        next(row for row in all_physical if row.lifecycle_schedule == "remove_then_birth"),
    )
    # Choose the first two planning rows because they cover K=8/K=32 under
    # the frozen rotation.
    planning = planning_manifest("development")[:2]
    return tuple(sorted(physical, key=lambda row: row.ordinal)), planning


def _assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert set(left) == set(right)  # type: ignore[arg-type]
        for key in left:
            _assert_tree_equal(left[key], right[key])  # type: ignore[index]
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)  # type: ignore[arg-type]
        for one, two in zip(left, right, strict=True):  # type: ignore[arg-type]
            _assert_tree_equal(one, two)
    else:
        assert left == right


@pytest.mark.slow
def test_complete_artifacts_are_exact_and_warm_reuse_never_calls_materializers(
    tmp_path: Path,
) -> None:
    physical_rows, planning_rows = _representative_rows()
    calls = {"physical": 0, "planning": 0}

    def physical_materializer(row: Any):
        calls["physical"] += 1
        return materialize_dynamic_set_episode(row)

    def planning_materializer(row: Any):
        calls["planning"] += 1
        return materialize_planning_task(row)

    binding = _binding(physical_rows, planning_rows)
    cold = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
        physical_materializer=physical_materializer,
        planning_materializer=planning_materializer,
    )
    cold_physical = tuple(cold.physical(row) for row in physical_rows)
    cold_planning = tuple(cold.planning(row) for row in planning_rows)
    evidence = cold.seal_complete()
    assert calls == {"physical": len(physical_rows), "planning": len(planning_rows)}
    assert cold.cold_materializations == len(physical_rows) + len(planning_rows)
    assert evidence.physical_entry_count == len(physical_rows)
    assert evidence.planning_entry_count == len(planning_rows)

    warm = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
        trusted_evidence=evidence.to_dict(),
        physical_materializer=physical_materializer,
        planning_materializer=planning_materializer,
    )
    warm_physical = tuple(warm.physical(row) for row in physical_rows)
    warm_planning = tuple(warm.planning(row) for row in planning_rows)
    assert calls == {"physical": len(physical_rows), "planning": len(planning_rows)}
    assert warm.warm_hits == len(physical_rows) + len(planning_rows)
    for first, second in zip(cold_physical, warm_physical, strict=True):
        assert first.episode["rgb"].shape == (56, 3, 64, 64)
        _assert_tree_equal(_physical_content(first), _physical_content(second))
        assert _tree_digest(_physical_content(first)) == _tree_digest(_physical_content(second))
    for first, second in zip(cold_planning, warm_planning, strict=True):
        _assert_tree_equal(_planning_content(first), _planning_content(second))
        assert _tree_digest(_planning_content(first)) == _tree_digest(_planning_content(second))


def test_non_development_rows_are_rejected_before_lookup(tmp_path: Path) -> None:
    physical_rows = physical_manifest("development")[:1]
    planning_rows = planning_manifest("development")[:1]
    cache = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=_binding(physical_rows, planning_rows),
        physical_rows=physical_rows,
        planning_rows=planning_rows,
    )
    with pytest.raises(PermissionError, match="non-development"):
        cache.physical(physical_manifest("selector")[0])
    with pytest.raises(PermissionError, match="non-development"):
        cache.planning(planning_manifest("selector")[0])
    assert cache.cold_materializations == 0


@pytest.mark.slow
def test_cold_cache_rejects_an_inconsistent_physical_acceptance_audit(
    tmp_path: Path,
) -> None:
    physical_rows = physical_manifest("development")[:1]
    planning_rows = planning_manifest("development")[:1]
    materialization = materialize_dynamic_set_episode(physical_rows[0])
    inconsistent = replace(
        materialization,
        accepted_seed=materialization.accepted_seed + 1,
    )
    cache = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=_binding(physical_rows, planning_rows),
        physical_rows=physical_rows,
        planning_rows=planning_rows,
        physical_materializer=lambda _row: inconsistent,
    )

    with pytest.raises(ValueError, match="physical cache acceptance audit differs"):
        cache.physical(physical_rows[0])
    assert not tuple((tmp_path / "cache" / "physical").glob("*/*.ptz"))


@pytest.mark.slow
def test_unsealed_disk_entry_is_fully_rematerialized_before_reuse(tmp_path: Path) -> None:
    physical_rows = physical_manifest("development")[:1]
    planning_rows = planning_manifest("development")[:1]
    calls = 0

    def physical_materializer(row: Any):
        nonlocal calls
        calls += 1
        return materialize_dynamic_set_episode(row)

    binding = _binding(physical_rows, planning_rows)
    first = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
        physical_materializer=physical_materializer,
    )
    expected = first.physical(physical_rows[0])
    assert calls == 1
    reopened = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
        physical_materializer=physical_materializer,
    )
    actual = reopened.physical(physical_rows[0])
    assert calls == 2
    assert reopened.recertified_materializations == 1
    _assert_tree_equal(_physical_content(expected), _physical_content(actual))


@pytest.mark.slow
def test_sealed_merkle_and_entry_bytes_fail_closed_on_tampering(tmp_path: Path) -> None:
    physical_rows = physical_manifest("development")[:1]
    planning_rows = planning_manifest("development")[:1]
    binding = _binding(physical_rows, planning_rows)
    cache = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
    )
    cache.physical(physical_rows[0])
    cache.planning(planning_rows[0])
    evidence = cache.seal_complete()
    entry = next((tmp_path / "cache" / "physical").glob("*/*.ptz"))
    contents = bytearray(entry.read_bytes())
    contents[len(contents) // 2] ^= 1
    entry.write_bytes(bytes(contents))
    with pytest.raises((ValueError, OSError)):
        DynamicSetDevelopmentEvaluationCache(
            tmp_path / "cache",
            binding=binding,
            physical_rows=physical_rows,
            planning_rows=planning_rows,
            trusted_evidence=evidence,
        ).physical(physical_rows[0])


@pytest.mark.slow
def test_repeated_validation_materialization_benchmark_is_warm_and_compact(
    tmp_path: Path,
) -> None:
    physical_rows = physical_manifest("development")[:2]
    planning_rows = planning_manifest("development")[:1]
    binding = _binding(physical_rows, planning_rows)
    started = time.perf_counter()
    cache = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
    )
    for row in physical_rows:
        cache.physical(row)
    for row in planning_rows:
        cache.planning(row)
    evidence = cache.seal_complete()
    cold_seconds = time.perf_counter() - started

    started = time.perf_counter()
    warm = DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
        trusted_evidence=evidence,
    )
    for row in physical_rows:
        warm.physical(row)
    for row in planning_rows:
        warm.planning(row)
    warm_seconds = time.perf_counter() - started
    stored_bytes = sum(
        path.stat().st_size for path in (tmp_path / "cache").rglob("*") if path.is_file()
    )
    raw_rgbd_bytes = sum(
        item.episode["rgb"].numel() * item.episode["rgb"].element_size()
        + item.episode["depth"].numel() * item.episode["depth"].element_size()
        for item in (cache.physical(row) for row in physical_rows)
    )
    assert warm_seconds < cold_seconds
    assert stored_bytes > 0
    assert stored_bytes < raw_rgbd_bytes
    assert warm.warm_hits == len(physical_rows) + len(planning_rows)


def test_hardlinked_entry_and_namespace_rebinding_are_rejected(tmp_path: Path) -> None:
    physical_rows = physical_manifest("development")[:1]
    planning_rows = planning_manifest("development")[:1]
    binding = _binding(physical_rows, planning_rows)
    DynamicSetDevelopmentEvaluationCache(
        tmp_path / "cache",
        binding=binding,
        physical_rows=physical_rows,
        planning_rows=planning_rows,
    )
    # Namespace mismatch is detected without materializing a row.
    with pytest.raises(ValueError, match="namespace binding"):
        DynamicSetDevelopmentEvaluationCache(
            tmp_path / "cache",
            binding=_binding(physical_rows, planning_rows, evaluator_sha256="d" * 64),
            physical_rows=physical_rows,
            planning_rows=planning_rows,
        )

    # Exercise the inventory's single-link rule using a synthetic bounded
    # entry; payload interpretation must never be reached.
    shard = tmp_path / "cache" / "physical" / "000"
    shard.mkdir()
    original = shard / "000000-synthetic.ptz"
    linked = tmp_path / "linked.ptz"
    original.write_bytes(b"bounded")
    os.link(original, linked)
    with pytest.raises(OSError, match="single-link"):
        DynamicSetDevelopmentEvaluationCache(
            tmp_path / "cache",
            binding=binding,
            physical_rows=physical_rows,
            planning_rows=planning_rows,
        )
