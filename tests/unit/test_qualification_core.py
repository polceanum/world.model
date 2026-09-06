from __future__ import annotations

import os
from pathlib import Path

import pytest

from world_model.training.qualification_core import (
    MetricGate,
    OrderedSplitLedger,
    QualificationArtifactDirectory,
    SplitPermit,
    canonical_json_bytes,
    canonical_sha256,
    metric_gate_failures,
    weighted_score,
)


def _artifacts(tmp_path: Path) -> QualificationArtifactDirectory:
    return QualificationArtifactDirectory.create_fresh(
        tmp_path / "qualification",
        allowed_names=("ledger.json", "report.json"),
    )


def test_canonical_json_normalizes_tuples_and_rejects_nonfinite_values() -> None:
    assert canonical_json_bytes({"b": (2, 3), "a": 1}) == b'{"a":1,"b":[2,3]}'
    assert canonical_sha256({"a": 1, "b": [2, 3]}) == canonical_sha256({"b": (2, 3), "a": 1})
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_json_bytes({"loss": float("nan")})


def test_weighted_score_and_metric_gates_require_exact_finite_schemas() -> None:
    assert weighted_score({"a": 2.0, "b": 4.0}, {"a": 0.25, "b": 0.75}) == 3.5
    with pytest.raises(ValueError, match="components differ"):
        weighted_score({"a": 2.0}, {"a": 0.5, "b": 0.5})

    gates = (MetricGate("low", "le", 1.0), MetricGate("high", "ge", 2.0))
    assert not metric_gate_failures({"low": 1.0, "high": 2.0}, gates)
    assert metric_gate_failures({"low": 1.1, "high": 1.9}, gates) == (
        "low:1.1>1",
        "high:1.9<2",
    )
    assert metric_gate_failures({"low": 0.0}, gates)[0].startswith("metric_schema:")


def test_artifacts_are_fresh_bounded_single_link_and_fixed_inventory(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    digest = artifacts.write_fresh_json("report.json", {"passed": True})
    assert len(digest) == 64
    assert artifacts.read_json("report.json") == {"passed": True}
    assert artifacts.inventory() == frozenset({"report.json"})
    with pytest.raises(FileExistsError):
        artifacts.write_fresh_json("report.json", {"passed": False})
    with pytest.raises(ValueError, match="outside"):
        artifacts.write_fresh_json("other.json", {})

    os.link(artifacts.root / "report.json", tmp_path / "alias.json")
    with pytest.raises(ValueError, match="single-link"):
        artifacts.read_json("report.json")


def test_artifact_root_and_files_reject_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        QualificationArtifactDirectory.attach(linked, allowed_names=("ledger.json",))

    nested = tmp_path / "nested"
    nested.mkdir()
    artifacts = _artifacts(nested)
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    (artifacts.root / "ledger.json").symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        artifacts.read_json("ledger.json")


def test_ordered_ledger_consumes_each_split_once_and_stops_after_failure(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    protocol = "a" * 64
    ledger = OrderedSplitLedger(
        artifacts,
        artifact_name="ledger.json",
        protocol_sha256=protocol,
        split_order=("selector", "confirmation", "final", "ood"),
    )
    ledger.create_fresh()
    with pytest.raises(RuntimeError, match="expected split"):
        ledger.begin("confirmation")

    selector = ledger.begin("selector")
    with pytest.raises(RuntimeError, match="active"):
        ledger.begin("selector")
    with pytest.raises(ValueError, match="does not match"):
        ledger.complete(
            SplitPermit("selector", 0, "wrong", protocol),
            status="passed",
            result_sha256="b" * 64,
        )
    state = ledger.complete(selector, status="passed", result_sha256="b" * 64)
    assert state["next_index"] == 1

    confirmation = ledger.begin("confirmation")
    state = ledger.complete(confirmation, status="failed", result_sha256="c" * 64)
    assert state["terminal"] is True
    assert state["next_index"] == 1
    with pytest.raises(RuntimeError, match="terminal"):
        ledger.begin("final")


def test_ordered_ledger_detects_tampering(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    ledger = OrderedSplitLedger(
        artifacts,
        artifact_name="ledger.json",
        protocol_sha256="d" * 64,
        split_order=("development",),
    )
    ledger.create_fresh()
    raw = artifacts.read_json("ledger.json")
    raw["next_index"] = 1
    artifacts.replace_json("ledger.json", raw)
    with pytest.raises(ValueError, match="record digest mismatch"):
        ledger.load()


def test_ordered_ledger_durably_claims_each_active_purpose_once(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    protocol = "e" * 64
    ledger = OrderedSplitLedger(
        artifacts,
        artifact_name="ledger.json",
        protocol_sha256=protocol,
        split_order=("selector",),
    )
    ledger.create_fresh()
    forged = SplitPermit("selector", 0, "f" * 64, protocol)
    with pytest.raises(RuntimeError, match="no active split"):
        ledger.claim_active(forged, purpose="planning", binding_sha256="a" * 64)
    assert ledger.load()["transitions"] == []

    permit = ledger.begin("selector")
    claimed = ledger.claim_active(
        permit,
        purpose="planning",
        binding_sha256="b" * 64,
    )
    assert [item["event"] for item in claimed["transitions"]] == ["begin", "claim"]
    assert dict(ledger.active_claims(permit)) == {"planning": "b" * 64}
    with pytest.raises(RuntimeError, match="already claimed"):
        OrderedSplitLedger(
            artifacts,
            artifact_name="ledger.json",
            protocol_sha256=protocol,
            split_order=("selector",),
        ).claim_active(
            permit,
            purpose="planning",
            binding_sha256="b" * 64,
        )
    with pytest.raises(RuntimeError, match="active"):
        ledger.begin("selector")

    completed = ledger.complete(permit, status="passed", result_sha256="c" * 64)
    assert [item["event"] for item in completed["transitions"]] == [
        "begin",
        "claim",
        "complete",
    ]
    with pytest.raises(RuntimeError, match="terminal"):
        ledger.claim_active(permit, purpose="physical", binding_sha256="d" * 64)
