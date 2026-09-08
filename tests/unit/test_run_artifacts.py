from __future__ import annotations

import json
from pathlib import Path

import pytest

from world_model.utils.run_artifacts import (
    MIB,
    RUN_MANIFEST_SCHEMA,
    RunArtifactPolicy,
    apply_cleanup,
    inventory_runs,
    plan_cleanup,
    read_run_manifest,
    write_run_manifest,
)


def _file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _small_policy() -> RunArtifactPolicy:
    return RunArtifactPolicy(
        rolling_budget_bytes=2_000,
        ordinary_completed_run_bytes=1_000,
        newest_failed_debug_bytes=1_500,
    )


def test_default_policy_is_the_managed_250_mib_contract() -> None:
    policy = RunArtifactPolicy().validate()
    assert policy.rolling_budget_bytes == 250 * MIB
    assert policy.ordinary_completed_run_bytes == 5 * MIB
    assert policy.newest_failed_debug_bytes == 25 * MIB
    assert policy.promoted_checkpoints_to_keep == 2


def test_prune_is_dry_by_default_ordered_and_idempotent_when_applied(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "candidate"
    _file(run / "report.json", 10)
    _file(run / "tensor.bin", 1_200)
    _file(run / "rejected.pt", 1_200)
    _file(run / "media.webp", 600)
    write_run_manifest(
        run,
        role="candidate",
        status="completed",
        artifacts={
            "report.json": "summary",
            "tensor.bin": "transient",
            "rejected.pt": "rejected",
            "media.webp": "media",
        },
        policy=_small_policy(),
    )

    plan = plan_cleanup(runs, archive_root=tmp_path / ".archive", policy=_small_policy())
    assert not plan.applied
    assert [action.category for action in plan.actions][:2] == ["transient", "rejected"]
    assert (run / "tensor.bin").exists()
    assert (run / "report.json").exists()

    applied = apply_cleanup(plan, runs)
    assert applied.applied
    assert not (run / "tensor.bin").exists()
    assert not (run / "rejected.pt").exists()
    assert (run / "report.json").exists()
    assert read_run_manifest(run).artifacts[0].present
    again = apply_cleanup(plan, runs)
    assert again.applied
    assert any("already absent" in warning for warning in again.warnings)


def test_clear_categories_and_pinned_incumbent_retention(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    incumbent = runs / "incumbent"
    _file(incumbent / "report.html", 20)
    _file(incumbent / "preview.png", 100)
    _file(incumbent / "checkpoint.pt", 200)
    write_run_manifest(
        incumbent,
        role="incumbent",
        status="completed",
        artifacts={
            "report.html": "report",
            "preview.png": "media",
            "checkpoint.pt": "checkpoint",
        },
        pinned=True,
    )
    plan = plan_cleanup(runs, category="media")
    assert not plan.actions
    assert (incumbent / "preview.png").exists()


def test_symlink_escape_and_corrupted_manifest_fail_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    _file(outside, 20)
    runs = tmp_path / "runs"
    linked = runs / "linked"
    linked.mkdir(parents=True)
    (linked / "escape.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        write_run_manifest(
            linked,
            role="candidate",
            status="failed",
            artifacts={"escape.bin": "transient"},
        )

    corrupt = runs / "corrupt"
    corrupt.mkdir()
    (corrupt / "run_manifest.json").write_text("{bad", encoding="utf-8")
    escaped = runs / "escaped"
    escaped.mkdir()
    (escaped / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": RUN_MANIFEST_SCHEMA,
                "run_id": "escaped",
                "created_at_utc": "2026-09-08T00:00:00+00:00",
                "role": "candidate",
                "status": "failed",
                "pinned": False,
                "policy": "compact",
                "artifacts": [
                    {
                        "path": "../outside.bin",
                        "category": "transient",
                        "bytes": 20,
                        "present": True,
                        "pinned": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = plan_cleanup(runs, policy=_small_policy())
    assert not plan.actions
    assert len(plan.warnings) == 3  # linked lacks a manifest; corrupt and escaped fail closed.
    assert outside.exists()
    inventory = inventory_runs(runs)
    assert all(not record["pruning_eligible"] for record in inventory["runs"])
