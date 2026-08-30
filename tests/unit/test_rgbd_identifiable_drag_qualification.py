from __future__ import annotations

import copy
import inspect
import io
import json
import os
import pickle
from pathlib import Path

import pytest
import torch

import world_model.training.rgbd_identifiable_drag_qualification as qualification


def _calibration_record(lower_bound: float) -> dict[str, object]:
    errors = [torch.ones(1, dtype=torch.float32) for _ in range(64)]
    variances = [torch.ones(1, dtype=torch.float32) for _ in range(64)]
    calibrated = qualification._calibrate_one_scale(
        errors,
        variances,
        lower_bound=lower_bound,
    )
    return qualification._scale_calibration_record(calibrated)


def _valid_calibration() -> dict[str, object]:
    digest = "1" * 64
    return {
        "method": "designed_family_scene_max_rank_59_float32_minimal",
        "confidence": float(qualification.CALIBRATION_CONFIDENCE),
        "normal_z": float(qualification.CALIBRATION_Z),
        "rank": qualification.CALIBRATION_RANK,
        "scene_count": 64,
        "evidence_ingest_count": 64,
        "evidence_replay_count": 0,
        "atomic_setter_calls": 1,
        "evidence_cache_sha256": digest,
        "raw_model_state_sha256": digest,
        "calibrated_model_state_sha256": "2" * 64,
        "gradient_audit_model_state_sha256": digest,
        "position": _calibration_record(0.0),
        "velocity": _calibration_record(0.0),
        "drag": _calibration_record(1.0),
        "variance_floor_clamp_count": 0,
        "variance_ceiling_clamp_count": 0,
    }


def _passing_metrics() -> dict[str, float]:
    metrics = {name: 0.0 for name in qualification.GATE_METRIC_SCHEMA}
    metrics.update(
        {
            "scene_count": 64.0,
            "object_fit_count": 128.0,
            "position_uncertainty_scale": 1.0,
            "velocity_uncertainty_scale": 1.0,
            "drag_uncertainty_scale": 1.0,
            "log_drag_gaussian_nll": -2.0,
            "identity_coverage": 1.0,
            "association_pair_coverage": 1.0,
            "persistent_object_id_max": 1.0,
            "counterfactual_pair_count": 32.0,
            "counterfactual_drag_swap_fraction": 1.0,
            "minimum_drag_excitation_m": 0.02,
            "minimum_profile_information": 1.0,
            "drag_grid_point_count": 257.0,
            "ingested_frame_count_min": 16.0,
            "ingested_frame_count_max": 16.0,
            "state_ingest_count_min": 16.0,
            "state_ingest_count_max": 16.0,
            "history_sample_count_per_scene_min": 32.0,
            "history_sample_count_per_scene_max": 32.0,
            "history_valid_count_per_scene_min": 32.0,
            "history_valid_count_per_scene_max": 32.0,
            "public_predict_calls_per_scene_min": 1.0,
            "public_predict_calls_per_scene_max": 1.0,
            "direct_velocity_calls_per_scene_min": 1.0,
            "direct_velocity_calls_per_scene_max": 1.0,
            "direct_velocity_valid_fraction": 1.0,
            "module_tensor_buffer_count": 3.0,
            "persistent_module_state_key_count": 3.0,
            "persistent_module_state_bytes": 12.0,
            "gradient_audit_scene_count": 4.0,
            "gradient_audit_unique_scene_fraction": 1.0,
            "horizon_2.00_position_mean_width_90_m": 0.001,
            "horizon_2.00_velocity_mean_width_90_mps": 0.001,
            "horizon_2.00_log_drag_mean_width_90": 0.1,
            "horizon_2.00_position_max_width_90_m": 0.002,
            "horizon_2.00_velocity_max_width_90_mps": 0.002,
            "horizon_2.00_log_drag_max_width_90": 0.2,
        }
    )
    for horizon in qualification.HORIZONS_SECONDS:
        label = f"{horizon:.2f}"
        for fixed in ("0.05", "0.185"):
            metrics[f"fixed_{fixed}_horizon_{label}_position_rmse_m"] = 1.0
            metrics[f"fixed_{fixed}_horizon_{label}_velocity_rmse_mps"] = 1.0
        for quantity in ("position", "velocity", "log_drag"):
            prefix = f"horizon_{label}_{quantity}"
            metrics[f"{prefix}_marginal_coverage_90"] = 0.9
            metrics[f"{prefix}_joint_coverage_90"] = 0.9
            metrics[f"{prefix}_rms_z"] = 1.0
    for name in tuple(metrics):
        if name.startswith(("gradient_l1/", "gradient_max_l1/", "gradient_min_history_frame_l1/")):
            metrics[name] = 1.0e-6
        elif name.startswith("gradient_supported_history_frames/"):
            metrics[name] = 16.0
    assert qualification.gate_failures(metrics) == []
    return metrics


def _source_and_publication() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "commit": "a" * 40,
            "dirty": False,
            "worktree_fingerprint": "b" * 64,
            "runtime_source_fingerprint": "c" * 64,
        },
        {
            "upstream_ref": "origin/main",
            "head_commit": "a" * 40,
            "upstream_commit": "a" * 40,
            "ahead": 0,
            "behind": 0,
        },
    )


def _valid_report_evidence() -> tuple[
    dict[str, object], dict[str, object], dict[str, object], dict[str, object]
]:
    source, publication = _source_and_publication()
    certificate = {"certificate_sha256": qualification.FROZEN_CERTIFICATE_SHA256}
    calibration = _valid_calibration()
    metrics = _passing_metrics()
    for component, metric_name in (
        ("position", "position_uncertainty_scale"),
        ("velocity", "velocity_uncertainty_scale"),
        ("drag", "drag_uncertainty_scale"),
    ):
        bits = int(calibration[component]["deployed_float32_bits"][2:], 16)
        metrics[metric_name] = float(qualification._float32_from_bits(bits))
    development = qualification._split_result(
        split="development",
        metrics=metrics,
        model_state_sha256=calibration["calibrated_model_state_sha256"],
    )
    report = {
        "artifact_kind": "rgbd_identifiable_drag_development",
        "protocol": qualification.bridge_protocol(),
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": qualification.FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(qualification.FROZEN_SOURCE_SHA256),
        "scene_family_certificate": certificate,
        "development_ledger": str(qualification.development_ledger_path()),
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "development": development,
        "calibration": calibration,
        "checkpoint": str(qualification.canonical_checkpoint_path()),
        "checkpoint_sha256": "3" * 64,
        "checkpoint_model_state_sha256": calibration["calibrated_model_state_sha256"],
        "passed": True,
        "review_ready": True,
        "stopped_after": "development",
    }
    return report, source, publication, certificate


def _terminal_receipts() -> tuple[list[str], list[str]]:
    ordinal_hashes = [f"{ordinal + 1:064x}" for ordinal in range(64)]
    batch_hashes = [
        qualification.canonical_sha256(ordinal_hashes[4 * batch_index : 4 * batch_index + 4])
        for batch_index in range(16)
    ]
    return ordinal_hashes, batch_hashes


def _development_bindings(
    source: dict[str, object], publication: dict[str, object]
) -> dict[str, object]:
    return {
        "protocol_sha256": qualification.bridge_protocol()["protocol_sha256"],
        "source_provenance": copy.deepcopy(source),
        "publication_provenance": copy.deepcopy(publication),
        "config_sha256": qualification.FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(qualification.FROZEN_SOURCE_SHA256),
        "development_manifest_sha256": qualification.MANIFEST_SHA256["development"],
        "certificate_sha256": qualification.FROZEN_CERTIFICATE_SHA256,
    }


def _terminal_development_record(
    report: dict[str, object],
    *,
    bindings: dict[str, object],
    report_sha256: str,
) -> dict[str, object]:
    ordinal_hashes, batch_hashes = _terminal_receipts()
    return {
        "artifact_kind": qualification._DevelopmentLedger.ARTIFACT_KIND,
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": qualification.MAX_ARCHITECTURE_ATTEMPTS,
        "bindings": copy.deepcopy(bindings),
        "attempt_reserved": True,
        "access_started": True,
        "development_data_materialized": True,
        "active_ordinal": None,
        "completed_ordinal_count": 64,
        "materialized_ordinal_count": 64,
        "ordinal_evidence_sha256s": ordinal_hashes,
        "active_batch_ordinals": None,
        "completed_batch_count": 16,
        "batch_evidence_sha256s": batch_hashes,
        "result_sha256": qualification.canonical_sha256(report["development"]),
        "status": "complete",
        "outcome": "passed",
        "report_sha256": report_sha256,
        "checkpoint_sha256": report["checkpoint_sha256"],
    }


def _terminal_qualification_record(
    report: dict[str, object],
    *,
    bindings: dict[str, object],
    report_sha256: str,
) -> dict[str, object]:
    split_states: dict[str, dict[str, object]] = {}
    for split in qualification._QualificationLedger.ORDER:
        ordinal_hashes, batch_hashes = _terminal_receipts()
        split_states[split] = {
            "access_started": True,
            "status": "passed",
            "result_sha256": qualification.canonical_sha256(report[split]),
            "completed_ordinal_count": 64,
            "materialized_ordinal_count": 64,
            "active_ordinal": None,
            "ordinal_evidence_sha256s": ordinal_hashes,
            "active_batch_ordinals": None,
            "completed_batch_count": 16,
            "batch_evidence_sha256s": batch_hashes,
        }
    return {
        "artifact_kind": qualification._QualificationLedger.ARTIFACT_KIND,
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": qualification.MAX_ARCHITECTURE_ATTEMPTS,
        "order": list(qualification._QualificationLedger.ORDER),
        "bindings": copy.deepcopy(bindings),
        "splits": split_states,
        "attempt_reserved": True,
        "protected_data_materialized": True,
        "status": "complete",
        "outcome": "passed",
        "stopped_after": "final_test",
        "report_sha256": report_sha256,
    }


def _json_artifact(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"


def _fake_live_development_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[qualification._DevelopmentLedger, bytes]:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    run_directory = qualification._canonical_run_directory()
    run_directory.mkdir(parents=True)
    source, publication = _source_and_publication()
    bindings = {
        "source_provenance": source,
        "publication_provenance": publication,
    }
    record = {
        "artifact_kind": qualification._DevelopmentLedger.ARTIFACT_KIND,
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": qualification.MAX_ARCHITECTURE_ATTEMPTS,
        "bindings": copy.deepcopy(bindings),
        "attempt_reserved": True,
        "access_started": True,
        "development_data_materialized": True,
        "active_ordinal": None,
        "materialized_ordinal_count": 0,
        "completed_ordinal_count": 0,
        "ordinal_evidence_sha256s": [],
        "active_batch_ordinals": None,
        "completed_batch_count": 0,
        "batch_evidence_sha256s": [],
        "result_sha256": None,
        "status": "development_materialization_started",
    }
    ledger = object.__new__(qualification._DevelopmentLedger)
    ledger.path = qualification.development_ledger_path()
    ledger.record = record
    ledger._bindings = copy.deepcopy(bindings)
    ledger._config = object()
    ledger._mint_identity = object()
    ledger._capability = None
    ledger._capability_issued = False
    contents = json.dumps(record, indent=2, sort_keys=True).encode() + b"\n"
    ledger.path.write_bytes(contents)
    metadata = os.lstat(ledger.path)
    ledger._receipt_digest = qualification.sha256_bytes(contents)
    ledger._receipt_device = metadata.st_dev
    ledger._receipt_inode = metadata.st_ino
    ledger._durable_record = copy.deepcopy(record)
    qualification._register_ledger_receipt(ledger)
    qualification._LIVE_PRIVATE_LEDGERS[id(ledger)] = (
        ledger,
        ledger._mint_identity,
        object(),
        None,
        qualification.canonical_sha256(ledger._bindings),
    )
    monkeypatch.setattr(qualification, "_require_single_thread_execution", lambda: None)
    monkeypatch.setattr(
        qualification, "_require_config_matches_frozen_path", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        qualification,
        "_current_execution_provenance",
        lambda **kwargs: (source, publication, {"certificate_sha256": "d" * 64}),
    )
    return ledger, contents


def _revoke_fake_ledger(ledger: qualification._DevelopmentLedger) -> None:
    qualification._revoke_ledger_governed_access(ledger)
    qualification._LIVE_PRIVATE_LEDGERS.pop(id(ledger), None)
    qualification._LIVE_LEDGER_RECEIPTS.pop(id(ledger), None)


def _fake_qualification_ledger_for_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> qualification._QualificationLedger:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    run_directory = qualification._canonical_run_directory()
    run_directory.mkdir(parents=True)
    ledger = object.__new__(qualification._QualificationLedger)
    ledger.path = qualification.qualification_ledger_path()
    ledger._bindings = {}
    ledger._config = object()
    ledger.record = {
        "artifact_kind": qualification._QualificationLedger.ARTIFACT_KIND,
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": qualification.MAX_ARCHITECTURE_ATTEMPTS,
        "order": list(qualification._QualificationLedger.ORDER),
        "bindings": {},
        "splits": {
            split: {
                "access_started": False,
                "status": "unopened",
                "result_sha256": None,
                "completed_ordinal_count": 0,
                "materialized_ordinal_count": 0,
                "active_ordinal": None,
                "ordinal_evidence_sha256s": [],
                "active_batch_ordinals": None,
                "completed_batch_count": 0,
                "batch_evidence_sha256s": [],
            }
            for split in qualification._QualificationLedger.ORDER
        },
        "attempt_reserved": True,
        "protected_data_materialized": False,
        "status": "reserved_before_protected_access",
    }
    contents = json.dumps(ledger.record, indent=2, sort_keys=True).encode() + b"\n"
    ledger.path.write_bytes(contents)
    metadata = os.lstat(ledger.path)
    ledger._receipt_digest = qualification.sha256_bytes(contents)
    ledger._receipt_device = metadata.st_dev
    ledger._receipt_inode = metadata.st_ino
    ledger._durable_record = copy.deepcopy(ledger.record)
    qualification._register_ledger_receipt(ledger)
    monkeypatch.setattr(
        qualification, "_validate_live_ledger_receipt", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(qualification, "_ledger_bound_source_guard", lambda *args, **kwargs: None)
    return ledger


def _evidence(
    *,
    position_variance: float,
    velocity_variance: float,
    drag_variance: float,
    ordinal: int = 0,
):
    zeros23 = torch.zeros((2, 3), dtype=torch.float32)
    zeros523 = torch.zeros((5, 2, 3), dtype=torch.float32)
    return qualification.SceneSufficientEvidence(
        split="development",
        ordinal=ordinal,
        scene_sha256=f"{ordinal + 1:064x}",
        current_position_truth=zeros23.clone(),
        current_position_mean=zeros23.clone(),
        current_position_raw_variance=torch.full((2, 3), position_variance, dtype=torch.float32),
        current_velocity_truth=zeros23.clone(),
        current_velocity_mean=zeros23.clone(),
        current_velocity_raw_variance=torch.full((2, 3), velocity_variance, dtype=torch.float32),
        log_drag_truth=torch.zeros((2, 1), dtype=torch.float32),
        log_drag_mean=torch.zeros((2, 1), dtype=torch.float32),
        log_drag_raw_variance=torch.full((2, 1), drag_variance, dtype=torch.float32),
        future_position_truth=zeros523.clone(),
        future_position_mean=zeros523.clone(),
        future_position_raw_variance=torch.ones((5, 2, 3), dtype=torch.float32),
        future_velocity_truth=zeros523.clone(),
        future_velocity_mean=zeros523.clone(),
        future_velocity_raw_variance=torch.ones((5, 2, 3), dtype=torch.float32),
        fixed_position_mean=torch.zeros((2, 5, 2, 3), dtype=torch.float32),
        fixed_velocity_mean=torch.zeros((2, 5, 2, 3), dtype=torch.float32),
        diagnostics=(),
    )


def test_protocol_is_seed_free_and_fixes_exact_artifacts() -> None:
    protocol = qualification.bridge_protocol()
    assert protocol["protocol_sha256"] == qualification.canonical_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    )
    assert set(qualification.QUALIFICATION_ARTIFACT_NAMES) == {
        "development_report.json",
        "development_model.pt",
        "development_attempt_1_access.json",
        "qualification_report.json",
        "qualification_attempt_1_access.json",
    }
    encoded = json.dumps(protocol, sort_keys=True).lower()
    assert '"seed"' not in encoded
    for split in qualification.SPLITS:
        rows = protocol["manifests"][split]["rows"]
        assert rows == [{"split": split, "ordinal": ordinal} for ordinal in range(64)]


@pytest.mark.parametrize("bad", [False, 0.0, "0", None])
def test_manifest_rejects_non_exact_integer_ordinal(bad: object) -> None:
    rows = list(qualification._manifest_rows("development"))
    rows[0] = {"split": "development", "ordinal": bad}
    with pytest.raises((TypeError, ValueError)):
        qualification._validate_manifest_rows("development", rows)


def test_gate_schema_rejects_impossible_fraction_and_negative_physical_value() -> None:
    metrics = {name: 0.0 for name in qualification.GATE_METRIC_SCHEMA}
    metrics["horizon_0.10_position_marginal_coverage_90"] = 1.5
    metrics["current_position_rmse_m"] = -1.0
    failures = qualification.gate_failures(metrics)
    assert any(
        item.startswith("horizon_0.10_position_marginal_coverage_90:") and ">0.995" in item
        for item in failures
    )
    assert any(item.startswith("current_position_rmse_m:-1<0") for item in failures)


def test_gate_schema_rejects_zero_scales_nonfloat_and_extra_field() -> None:
    metrics = {name: 0.0 for name in qualification.GATE_METRIC_SCHEMA}
    metrics["position_uncertainty_scale"] = 0.0
    metrics["velocity_uncertainty_scale"] = True
    metrics["unused"] = 1.0
    failures = qualification.gate_failures(metrics)
    assert "position_uncertainty_scale:0<=0" in failures
    assert "velocity_uncertainty_scale:missing_nonfinite_or_nonfloat" in failures
    assert any(item.startswith("metric_schema:") and "unused" in item for item in failures)


def test_calibration_validator_recomputes_target_bits_and_nextafter_steps() -> None:
    calibration = _valid_calibration()
    qualification._validate_calibration(calibration)

    tampered_target = copy.deepcopy(calibration)
    tampered_target["position"]["target_float64"] += 0.25  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="initial bits"):
        qualification._validate_calibration(tampered_target)

    tampered_bits = copy.deepcopy(calibration)
    tampered_bits["velocity"]["deployed_float32_bits"] = "0x3f800100"  # type: ignore[index]
    with pytest.raises(ValueError, match="nextafter"):
        qualification._validate_calibration(tampered_bits)


def test_split_calibration_binding_rejects_internally_valid_mismatch() -> None:
    calibration = _valid_calibration()
    expected_state = calibration["calibrated_model_state_sha256"]
    metrics = {
        "position_uncertainty_scale": float(
            qualification._float32_from_bits(
                int(calibration["position"]["deployed_float32_bits"][2:], 16)  # type: ignore[index]
            )
        ),
        "velocity_uncertainty_scale": float(
            qualification._float32_from_bits(
                int(calibration["velocity"]["deployed_float32_bits"][2:], 16)  # type: ignore[index]
            )
        ),
        "drag_uncertainty_scale": float(
            qualification._float32_from_bits(
                int(calibration["drag"]["deployed_float32_bits"][2:], 16)  # type: ignore[index]
            )
        ),
    }
    result = {"model_state_sha256": expected_state, "metrics": metrics}
    qualification._validate_split_calibration_binding(
        result,
        calibration,
        expected_state_sha256=expected_state,
        label="test",
    )
    tampered = copy.deepcopy(result)
    tampered["metrics"]["position_uncertainty_scale"] += 1.0e-9
    with pytest.raises(ValueError, match="exact deployed bits"):
        qualification._validate_split_calibration_binding(
            tampered,
            calibration,
            expected_state_sha256=expected_state,
            label="test",
        )
    tampered = copy.deepcopy(result)
    tampered["model_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="model state"):
        qualification._validate_split_calibration_binding(
            tampered,
            calibration,
            expected_state_sha256=expected_state,
            label="test",
        )


def test_drag_calibration_rejects_target_below_one() -> None:
    calibration = _valid_calibration()
    calibration["drag"]["target_float64"] = 0.5  # type: ignore[index]
    with pytest.raises(ValueError, match="drag calibration target"):
        qualification._validate_calibration(calibration)


def test_public_calibration_fixture_is_exact() -> None:
    assert qualification.PUBLIC_CALIBRATION_REGRESSION == {
        "position": {"float32_bits": "0x4127aa75", "additional_ulps": 0},
        "velocity": {
            "initial_float32_bits": "0x41249854",
            "deployed_float32_bits": "0x41249858",
            "additional_ulps": 4,
        },
        "drag": {"float32_bits": "0x3fa419c1", "additional_ulps": 0},
    }


def test_posthoc_clamp_evidence_counts_exact_boundaries() -> None:
    config = qualification.require_frozen_config(qualification._frozen_config_path())
    evidence = _evidence(
        position_variance=float(torch.tensor(config.model.filter.min_log_variance).exp()),
        velocity_variance=float(torch.tensor(config.model.filter.max_log_variance).exp()),
        drag_variance=config.model.rgbd.temporal_drag_log_parameter_variance_floor,
    )
    deployed = qualification._deployed_variances_from_cache(
        evidence,
        position_scale=torch.tensor(1.0),
        velocity_scale=torch.tensor(1.0),
        drag_scale=torch.tensor(1.0),
        config=config,
    )
    assert deployed.position_floor_clamps == 6
    assert deployed.velocity_ceiling_clamps == 6
    assert deployed.drag_floor_clamps == 2


def test_atomic_direct_validity_rejects_generic_valid_without_drag_ownership() -> None:
    class FakeEvidence:
        valid_mask = torch.ones((4, 2), dtype=torch.bool)
        drag_valid_mask = torch.zeros((4, 2), dtype=torch.bool)
        position_valid_mask = valid_mask
        log_drag = torch.zeros((4, 2, 1), dtype=torch.float32)
        position = torch.zeros((4, 2, 3), dtype=torch.float32)

        @staticmethod
        def resolved_axis_valid_mask() -> torch.Tensor:
            return torch.ones((4, 2, 3), dtype=torch.bool)

    with pytest.raises(RuntimeError, match="atomically own"):
        qualification._atomic_direct_valid_mask(FakeEvidence())


def test_raw_path_apis_reject_string_and_path_subclass() -> None:
    expected = qualification.canonical_checkpoint_path()
    with pytest.raises(TypeError):
        qualification._require_canonical_path(str(expected), expected, label="checkpoint")  # type: ignore[arg-type]

    class ChildPath(qualification._NATIVE_PATH_TYPE):
        pass

    with pytest.raises(TypeError):
        qualification._require_canonical_path(ChildPath(expected), expected, label="checkpoint")


def test_run_tree_rejects_symlinked_runs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_runs = tmp_path / "real-runs"
    real_runs.mkdir()
    os.symlink(real_runs, tmp_path / "runs")
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(PermissionError, match="real directory"):
        qualification._validate_run_tree(frozenset(), stage="test")


def test_run_tree_rejects_symlinked_canonical_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    os.symlink(target, runs / "rgbd_two_visible_orbital_camera_identifiable_drag_v1")
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(PermissionError, match="real directory"):
        qualification._validate_run_tree(
            frozenset({qualification.DEVELOPMENT_LEDGER_NAME}), stage="test"
        )


def test_distinct_paths_reject_hardlink_inode_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    run_directory = tmp_path / qualification.RUN_RELATIVE_PATH
    run_directory.mkdir(parents=True)
    first = run_directory / "first"
    second = run_directory / "second"
    first.write_bytes(b"evidence")
    os.link(first, second)
    with pytest.raises(ValueError, match="hard-link alias"):
        qualification._validate_distinct_canonical_paths(
            {"first": first, "second": second}, atomic_writers=()
        )


def test_repository_identity_rejects_hardlinked_qualification_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    source = repository / qualification._MODULE_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    source.write_bytes(b"qualification source")
    os.link(source, repository / "source-hardlink")
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(qualification, "__file__", str(source))
    with pytest.raises(PermissionError, match="real file"):
        qualification._validate_repository_identity()


def test_config_binding_rejects_hardlink_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    path = qualification._frozen_config_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen config")
    os.link(path, tmp_path / "late-config-hardlink")
    with pytest.raises(PermissionError, match="single-link"):
        qualification._require_config_matches_frozen_path(object(), path)  # type: ignore[arg-type]


def test_durable_writer_rejects_sixth_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    sixth = qualification._canonical_run_directory() / "sixth.json"
    with pytest.raises(ValueError, match="no arbitrary write path"):
        qualification._durable_create(sixth, b"{}\n")
    assert not sixth.exists()


def test_live_development_receipt_rejects_early_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    run_directory = qualification._canonical_run_directory()
    run_directory.mkdir(parents=True)
    source = {
        "commit": "a" * 40,
        "dirty": False,
        "worktree_fingerprint": "b" * 64,
        "runtime_source_fingerprint": "c" * 64,
    }
    publication = {
        "upstream_ref": "origin/main",
        "head_commit": "a" * 40,
        "upstream_commit": "a" * 40,
        "ahead": 0,
        "behind": 0,
    }
    record = {
        "artifact_kind": qualification._DevelopmentLedger.ARTIFACT_KIND,
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": qualification.MAX_ARCHITECTURE_ATTEMPTS,
        "bindings": {
            "source_provenance": source,
            "publication_provenance": publication,
        },
        "attempt_reserved": True,
        "access_started": True,
        "development_data_materialized": True,
        "active_ordinal": None,
        "materialized_ordinal_count": 0,
        "completed_ordinal_count": 0,
        "ordinal_evidence_sha256s": [],
        "active_batch_ordinals": None,
        "completed_batch_count": 0,
        "batch_evidence_sha256s": [],
        "result_sha256": None,
        "status": "development_materialization_started",
    }
    ledger = object.__new__(qualification._DevelopmentLedger)
    ledger.path = qualification.development_ledger_path()
    ledger.record = record
    ledger._bindings = copy.deepcopy(record["bindings"])
    ledger._config = object()
    ledger._mint_identity = object()
    contents = json.dumps(record, indent=2, sort_keys=True).encode() + b"\n"
    ledger.path.write_bytes(contents)
    metadata = os.lstat(ledger.path)
    ledger._receipt_digest = qualification.sha256_bytes(contents)
    ledger._receipt_device = metadata.st_dev
    ledger._receipt_inode = metadata.st_ino
    ledger._durable_record = copy.deepcopy(record)
    qualification._register_ledger_receipt(ledger)
    qualification._LIVE_PRIVATE_LEDGERS[id(ledger)] = (
        ledger,
        ledger._mint_identity,
        object(),
        None,
        qualification.canonical_sha256(ledger._bindings),
    )
    monkeypatch.setattr(qualification, "_require_single_thread_execution", lambda: None)
    monkeypatch.setattr(
        qualification, "_require_config_matches_frozen_path", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        qualification,
        "_current_execution_provenance",
        lambda **kwargs: (source, publication, {"certificate_sha256": "d" * 64}),
    )
    try:
        qualification._validate_live_ledger_receipt(ledger, split="development")
        qualification.canonical_checkpoint_path().write_bytes(b"checkpoint")
        with pytest.raises(PermissionError, match="non-stage artifact inventory"):
            qualification._validate_live_ledger_receipt(ledger, split="development")
        qualification.canonical_development_report_path().write_bytes(b"report")
        with pytest.raises(PermissionError, match="non-stage artifact inventory"):
            qualification._validate_live_ledger_receipt(ledger, split="development")
        (run_directory / "sixth").write_bytes(b"forbidden")
        with pytest.raises(PermissionError, match="non-stage artifact inventory"):
            qualification._validate_live_ledger_receipt(ledger, split="development")
    finally:
        qualification._LIVE_PRIVATE_LEDGERS.pop(id(ledger), None)
        qualification._LIVE_LEDGER_RECEIPTS.pop(id(ledger), None)


@pytest.mark.parametrize("mutation", ["nested_binding", "receipt_list", "status"])
def test_live_ledger_rejects_in_memory_tamper_without_publishing(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, original = _fake_live_development_ledger(tmp_path, monkeypatch)
    try:
        if mutation == "nested_binding":
            ledger.record["bindings"]["source_provenance"]["commit"] = "e" * 40
        elif mutation == "receipt_list":
            ledger.record["ordinal_evidence_sha256s"].append("f" * 64)
        else:
            ledger.record["status"] = "forged_terminal_status"
        with pytest.raises(PermissionError, match="pinned receipt"):
            ledger._begin_ordinal("development", 0)
        assert ledger.path.read_bytes() == original
        assert ledger._durable_record != ledger.record
    finally:
        _revoke_fake_ledger(ledger)


def test_registry_receipt_pin_rejects_forged_disk_and_all_object_receipt_attrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _ = _fake_live_development_ledger(tmp_path, monkeypatch)
    try:
        forged = copy.deepcopy(ledger.record)
        forged["active_ordinal"] = 0
        forged["status"] = "development_ordinal_materialization_started"
        contents = json.dumps(forged, indent=2, sort_keys=True).encode() + b"\n"
        ledger.path.write_bytes(contents)
        metadata = os.lstat(ledger.path)
        ledger.record = forged
        ledger._durable_record = copy.deepcopy(forged)
        ledger._receipt_digest = qualification.sha256_bytes(contents)
        ledger._receipt_device = metadata.st_dev
        ledger._receipt_inode = metadata.st_ino
        with pytest.raises(PermissionError, match="trusted registry"):
            qualification._validate_live_ledger_receipt(ledger, split="development")
    finally:
        _revoke_fake_ledger(ledger)


def test_manifest_registry_rejects_duplicate_after_mutable_attrs_are_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _ = _fake_live_development_ledger(tmp_path, monkeypatch)
    try:
        capability = ledger.capability()
        ledger._capability_issued = False
        ledger._capability = None
        with pytest.raises(PermissionError, match="already owns"):
            qualification._ManifestCapability(
                qualification._CAPABILITY_AUTHORITY,
                ledger=ledger,
                ledger_mint_identity=ledger._mint_identity,
                split="development",
            )
        assert qualification._LIVE_MANIFEST_BINDINGS[(id(ledger), "development")] == (
            ledger,
            capability,
        )
    finally:
        _revoke_fake_ledger(ledger)


@pytest.mark.parametrize("ledger_kind", ["development", "qualification"])
@pytest.mark.parametrize("failure_stage", ["parent_fsync", "publication_lstat", "stable_read"])
def test_post_replace_failures_reconcile_and_adopt_exact_receipt(
    ledger_kind: str,
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if ledger_kind == "development":
        ledger, _ = _fake_live_development_ledger(tmp_path, monkeypatch)
        target = copy.deepcopy(ledger.record)
        target["active_ordinal"] = 0
        target["status"] = "development_ordinal_materialization_started"
    else:
        ledger = _fake_qualification_ledger_for_replace(tmp_path, monkeypatch)
        target = copy.deepcopy(ledger.record)
        target["splits"]["selector"]["access_started"] = True
        target["splits"]["selector"]["status"] = "materialization_started"
        target["protected_data_materialized"] = True
        target["status"] = "selector_materialization_started"

    if failure_stage == "parent_fsync":
        original = qualification._fsync_parent
        injected = False

        def fail_once(path: Path) -> None:
            nonlocal injected
            if path == ledger.path and not injected:
                injected = True
                raise OSError("injected parent fsync failure")
            original(path)

        monkeypatch.setattr(qualification, "_fsync_parent", fail_once)
    elif failure_stage == "publication_lstat":
        original_regular = qualification._require_single_link_regular
        injected = False

        def fail_publication(path: Path, *, label: str):
            nonlocal injected
            if label == "identifiable-drag replacement publication" and not injected:
                injected = True
                raise OSError("injected publication lstat failure")
            return original_regular(path, label=label)

        monkeypatch.setattr(qualification, "_require_single_link_regular", fail_publication)
    else:
        original_read = qualification.stable_read_bytes
        injected = False

        def fail_read(path: Path, *, label: str) -> bytes:
            nonlocal injected
            if label == "identifiable-drag replacement publication" and not injected:
                injected = True
                raise OSError("injected stable read failure")
            return original_read(path, label=label)

        monkeypatch.setattr(qualification, "stable_read_bytes", fail_read)

    try:
        ledger._replace(target, label="injected post-replace reconciliation")
        persisted = json.loads(ledger.path.read_bytes())
        assert persisted == target
        assert ledger.record == target
        pin = qualification._LIVE_LEDGER_RECEIPTS[id(ledger)]
        assert pin[4] == qualification.sha256_bytes(ledger.path.read_bytes())
        assert pin[5] == qualification.canonical_sha256(target)
        assert pin[7] == 1
    finally:
        if ledger_kind == "development":
            _revoke_fake_ledger(ledger)
        else:
            qualification._LIVE_LEDGER_RECEIPTS.pop(id(ledger), None)


def test_failed_report_digest_requires_exact_intended_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    path = qualification.canonical_development_report_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"passed":true}\n')
    report = {"passed": False, "stopped_after": "development"}
    digest = qualification._persist_failed_report(path, report, label="test")
    assert digest == qualification.sha256_bytes(qualification._report_bytes(report))
    assert path.read_bytes() == qualification._report_bytes(report)

    stale = {"passed": False, "stopped_after": "later"}
    monkeypatch.setattr(qualification, "_durable_replace", lambda *args, **kwargs: None)
    assert qualification._persist_failed_report(path, stale, label="test") is None


def test_failed_report_returns_none_when_fresh_write_never_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    path = qualification.canonical_qualification_report_path()

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("injected report-write failure")

    monkeypatch.setattr(qualification, "_write_report_fresh", fail_write)
    assert (
        qualification._persist_failed_report(
            path,
            {"passed": False, "stopped_after": "selector"},
            label="test",
        )
        is None
    )
    assert not path.exists()


def test_restricted_checkpoint_loader_rejects_unsafe_global() -> None:
    class Unsafe:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    buffer = io.BytesIO()
    torch.save({"unsafe": Unsafe()}, buffer)
    with pytest.raises(pickle.UnpicklingError):
        qualification._checkpoint_payload_from_bytes(buffer.getvalue())


def test_restricted_checkpoint_loader_always_requests_weights_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_load(*args: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(torch, "load", fake_load)
    assert qualification._checkpoint_payload_from_bytes(b"payload") == {}
    assert observed == {"map_location": "cpu", "weights_only": True}


def test_checkpoint_state_requires_exact_three_float32_scalar_buffers() -> None:
    state = {
        f"observation_modules.rgbd.{leaf}": torch.tensor(1.0, dtype=torch.float32)
        for leaf in qualification._SCALE_STATE_LEAVES
    }
    validated = qualification._validate_checkpoint_model_state(state)
    assert len(validated) == 3
    assert sum(value.numel() * value.element_size() for value in validated.values()) == 12
    with pytest.raises(ValueError, match="exactly the three"):
        qualification._validate_checkpoint_model_state({**state, "parameter": torch.tensor(0.0)})
    bad_dtype = dict(state)
    bad_dtype[next(iter(bad_dtype))] = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="CPU scalar float32"):
        qualification._validate_checkpoint_model_state(bad_dtype)


def _fake_batch() -> tuple[object, tuple[object, ...]]:
    manifest = object.__new__(qualification._ManifestCapability)
    ledger = object()
    batch = object.__new__(qualification._BatchCapability)
    tokens = tuple(object.__new__(qualification._OrdinalCapability) for _ in range(4))
    batch._manifest = manifest
    batch._ledger = ledger
    batch._split = "development"
    batch._ordinals = (0, 1, 2, 3)
    batch._tokens = tokens
    batch._next_constructor = 4
    batch._evaluated = False
    for ordinal, token in enumerate(tokens):
        qualification._LIVE_ORDINAL_CAPABILITIES[id(token)] = (
            token,
            manifest,
            ledger,
            "development",
            ordinal,
            "constructed",
        )
    return batch, tokens


def test_batch_evaluator_transition_is_atomic_and_single_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, tokens = _fake_batch()
    monkeypatch.setattr(qualification, "_validate_batch_capability", lambda *args, **kwargs: None)
    qualification._mark_batch_evaluated(batch)  # type: ignore[arg-type]
    assert all(
        qualification._LIVE_ORDINAL_CAPABILITIES[id(token)][-1] == "evaluated" for token in tokens
    )
    with pytest.raises(RuntimeError, match="once"):
        qualification._mark_batch_evaluated(batch)  # type: ignore[arg-type]
    for token in tokens:
        qualification._LIVE_ORDINAL_CAPABILITIES.pop(id(token), None)


def test_batch_evaluator_rejects_partial_or_reordered_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, tokens = _fake_batch()
    monkeypatch.setattr(qualification, "_validate_batch_capability", lambda *args, **kwargs: None)
    registration = qualification._LIVE_ORDINAL_CAPABILITIES[id(tokens[2])]
    qualification._LIVE_ORDINAL_CAPABILITIES[id(tokens[2])] = (*registration[:4], 3, "constructed")
    with pytest.raises(PermissionError, match="partial or reordered"):
        qualification._mark_batch_evaluated(batch)  # type: ignore[arg-type]
    assert all(
        qualification._LIVE_ORDINAL_CAPABILITIES[id(token)][-1] == "constructed" for token in tokens
    )
    for token in tokens:
        qualification._LIVE_ORDINAL_CAPABILITIES.pop(id(token), None)


def test_single_ordinal_state_machine_blocks_evaluator_before_constructor_and_next_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLedger:
        def _begin_ordinal(self, split: str, ordinal: int) -> None:
            assert (split, ordinal) == ("development", 0)

        def _mark_ordinal_constructed(self, split: str, ordinal: int) -> None:
            assert (split, ordinal) == ("development", 0)

        def _complete_ordinal(self, split: str, ordinal: int, evidence_sha256: str) -> None:
            assert split == "development"
            assert ordinal == 0
            assert evidence_sha256 == "a" * 64

    manifest = object.__new__(qualification._ManifestCapability)
    manifest._ledger = FakeLedger()
    manifest._split = "development"
    manifest._next_ordinal = 0
    manifest._active = None
    manifest._pending = {}
    manifest._active_batch = None
    manifest._finished = False
    monkeypatch.setattr(
        qualification, "_validate_manifest_capability", lambda *args, **kwargs: None
    )
    token = manifest.begin_ordinal(0)
    with pytest.raises(PermissionError, match="forged, replayed"):
        qualification._validate_ordinal_evaluator_capability(token, split="development", ordinal=0)
    with pytest.raises(RuntimeError, match="order/replay"):
        manifest.begin_ordinal(1)
    qualification._consume_ordinal_constructor_capability(token, split="development", ordinal=0)
    qualification._mark_ordinal_constructed(token, split="development", ordinal=0)
    qualification._validate_ordinal_evaluator_capability(token, split="development", ordinal=0)
    qualification._mark_ordinal_evaluated(token, split="development", ordinal=0)
    with pytest.raises(PermissionError, match="one exact constructor"):
        qualification._mark_ordinal_evaluated(token, split="development", ordinal=0)
    manifest.complete_ordinal(token, ordinal=0, evidence_sha256="a" * 64)
    assert manifest._next_ordinal == 1


class _FakeManifest:
    def __init__(self) -> None:
        self.finished = False

    def finish_manifest(self) -> None:
        self.finished = True

    def require_finished(self) -> None:
        assert self.finished


def _fake_authorized_batch(
    config: object,
    *,
    split: str,
    ordinals: tuple[int, int, int, int],
    manifest_capability: object,
    reviewed_state: object,
    expected_state_sha256: object,
    audit_vjp: bool,
) -> tuple[list[qualification.SceneSufficientEvidence], dict[str, float]]:
    del config, manifest_capability, reviewed_state, expected_state_sha256
    assert split == "development"
    rows = [
        _evidence(
            position_variance=1.0,
            velocity_variance=1.0,
            drag_variance=1.0,
            ordinal=ordinal,
        )
        for ordinal in ordinals
    ]
    return rows, {"vjp_sentinel": 1.0} if audit_vjp else {}


@pytest.mark.parametrize(
    ("failure_label", "expected_batches"),
    [
        ("development manifest before access", 0),
        ("development batch 0 after access", 1),
        ("development manifest after access", 16),
    ],
)
def test_manifest_boundary_failure_injection_stops_without_replay(
    failure_label: str,
    expected_batches: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def evaluator(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return _fake_authorized_batch(*args, **kwargs)

    def guard(label: str) -> None:
        if label == failure_label:
            raise RuntimeError("injected boundary failure")

    monkeypatch.setattr(qualification, "_evaluate_authorized_batch", evaluator)
    with pytest.raises(RuntimeError, match="injected boundary failure"):
        qualification._collect_manifest_once(
            object(),  # type: ignore[arg-type]
            split="development",
            manifest_capability=_FakeManifest(),  # type: ignore[arg-type]
            reviewed_state=None,
            expected_state_sha256=None,
            boundary_guard=guard,
        )
    assert calls == expected_batches


def test_manifest_fake_rows_preserve_distinct_index_hash_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "_evaluate_authorized_batch", _fake_authorized_batch)
    manifest = _FakeManifest()
    cache, vjp = qualification._collect_manifest_once(
        object(),  # type: ignore[arg-type]
        split="development",
        manifest_capability=manifest,  # type: ignore[arg-type]
        reviewed_state=None,
        expected_state_sha256=None,
        boundary_guard=lambda label: None,
    )
    assert [row.ordinal for row in cache] == list(range(64))
    assert len({row.scene_sha256 for row in cache}) == 64
    assert vjp == {"vjp_sentinel": 1.0}
    assert manifest.finished


def test_nominal_authority_types_reject_direct_or_unregistered_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "_require_single_thread_execution", lambda: None)
    for nominal in (
        qualification._RunAuthorization,
        qualification._ReviewedDevelopmentSeal,
        qualification._OrdinalCapability,
        qualification._BatchCapability,
    ):
        with pytest.raises(PermissionError):
            nominal()
    forged = object.__new__(qualification._RunAuthorization)
    with pytest.raises(PermissionError, match="forged"):
        qualification._consume_run_authorization(
            forged,
            kind="development",
            bindings={},
        )


def test_duplicate_live_authorization_and_review_seal_mints_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "commit": "a" * 40,
        "dirty": False,
        "worktree_fingerprint": "b" * 64,
        "runtime_source_fingerprint": "c" * 64,
    }
    publication = {
        "upstream_ref": "origin/main",
        "head_commit": "a" * 40,
        "upstream_commit": "a" * 40,
        "ahead": 0,
        "behind": 0,
    }
    monkeypatch.setattr(qualification, "_require_single_thread_execution", lambda: None)
    monkeypatch.setattr(
        qualification,
        "_current_execution_provenance",
        lambda **kwargs: (source, publication, {"certificate_sha256": "d" * 64}),
    )
    monkeypatch.setattr(qualification, "_validate_run_tree", lambda *args, **kwargs: None)
    bindings = {
        "protocol_sha256": qualification.bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": qualification.FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(qualification.FROZEN_SOURCE_SHA256),
        "development_manifest_sha256": qualification.MANIFEST_SHA256["development"],
        "certificate_sha256": qualification.FROZEN_CERTIFICATE_SHA256,
    }
    authorization = qualification._mint_run_authorization("development", bindings)
    with pytest.raises(PermissionError, match="identical live run authorization"):
        qualification._mint_run_authorization("development", bindings)
    qualification._LIVE_RUN_AUTHORIZATIONS.pop(id(authorization), None)

    review_bindings = {
        "reviewed_checkpoint_sha256": "1" * 64,
        "reviewed_development_report_sha256": "2" * 64,
        "reviewed_development_ledger_sha256": "3" * 64,
        "model_state_sha256": "4" * 64,
        "calibration_sha256": "5" * 64,
    }
    seal = qualification._mint_reviewed_development_seal(review_bindings)
    with pytest.raises(PermissionError, match="identical live reviewed-development seal"):
        qualification._mint_reviewed_development_seal(review_bindings)
    qualification._LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(seal), None)


@pytest.mark.parametrize(
    "failure_label",
    ["thread", "source", "upstream", "inventory", "duplicate"],
)
def test_failed_qualification_authorization_mint_revokes_exact_owned_seal(
    failure_label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "_require_single_thread_execution", lambda: None)
    bindings = {
        "reviewed_checkpoint_sha256": "1" * 64,
        "reviewed_development_report_sha256": "2" * 64,
        "reviewed_development_ledger_sha256": "3" * 64,
        "model_state_sha256": "4" * 64,
        "calibration_sha256": "5" * 64,
    }
    prior_seals = dict(qualification._LIVE_REVIEWED_DEVELOPMENT_SEALS)
    prior_authorizations = dict(qualification._LIVE_RUN_AUTHORIZATIONS)
    seal = qualification._mint_reviewed_development_seal(bindings)

    def fail_mint(*args: object, **kwargs: object):
        if failure_label == "duplicate":
            leaked = object.__new__(qualification._RunAuthorization)
            qualification._LIVE_RUN_AUTHORIZATIONS[id(leaked)] = (
                leaked,
                qualification._RUN_AUTHORITY,
                "qualification",
                qualification.canonical_sha256(bindings),
                seal,
            )
        raise RuntimeError(f"injected {failure_label} authorization failure")

    monkeypatch.setattr(qualification, "_mint_run_authorization", fail_mint)
    with pytest.raises(RuntimeError, match=failure_label):
        qualification._mint_owned_qualification_authorization(seal, bindings)
    assert prior_seals == qualification._LIVE_REVIEWED_DEVELOPMENT_SEALS
    assert prior_authorizations == qualification._LIVE_RUN_AUTHORIZATIONS

    replacement = qualification._mint_reviewed_development_seal(bindings)
    qualification._LIVE_REVIEWED_DEVELOPMENT_SEALS.pop(id(replacement), None)


def test_manual_development_ledger_allocation_cannot_mint_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "_require_single_thread_execution", lambda: None)
    ledger = object.__new__(qualification._DevelopmentLedger)
    ledger._capability_issued = False
    ledger._capability = None
    with pytest.raises(PermissionError, match="not live issuer"):
        ledger.capability()


def test_single_thread_guard_is_inside_both_run_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    with pytest.raises(RuntimeError, match="exactly one"):
        qualification._require_single_thread_execution()
    with pytest.raises(RuntimeError, match="exactly one"):
        qualification._validate_live_ledger_receipt(object(), split="development")
    for function in (qualification.run_development, qualification.run_qualification):
        source = inspect.getsource(function)
        assert source.index("_require_single_thread_execution()") < source.index(
            "_require_canonical_path"
        )


@pytest.mark.parametrize("ordinal", [False, 0.0, torch.tensor(0)])
def test_raw_ordinal_boundaries_reject_equality_coercions_without_state_change(
    ordinal: object,
) -> None:
    token = object.__new__(qualification._OrdinalCapability)
    before = dict(qualification._LIVE_ORDINAL_CAPABILITIES)
    for boundary in (
        qualification._consume_ordinal_constructor_capability,
        qualification._mark_ordinal_constructed,
        qualification._validate_ordinal_evaluator_capability,
        qualification._mark_ordinal_evaluated,
    ):
        with pytest.raises(TypeError, match="exact bounded integer"):
            boundary(
                token,
                split="development",
                ordinal=ordinal,  # type: ignore[arg-type]
            )
    assert before == qualification._LIVE_ORDINAL_CAPABILITIES


def test_later_protected_split_cannot_open_after_unpassed_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = object.__new__(qualification._QualificationLedger)
    ledger.record = {
        "splits": {
            split: {
                "access_started": False,
                "status": "unopened",
            }
            for split in qualification._QualificationLedger.ORDER
        }
    }
    ledger._capabilities = {}
    monkeypatch.setattr(
        qualification, "_validate_live_ledger_receipt", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="predecessor passes"):
        ledger.begin_access("confirmation")
    ledger.record["splits"]["selector"] = {
        "access_started": True,
        "status": "failed",
    }
    with pytest.raises(RuntimeError, match="predecessor passes"):
        ledger.begin_access("confirmation")
    assert ledger.record["splits"]["confirmation"] == {
        "access_started": False,
        "status": "unopened",
    }


def test_terminal_qualification_ledger_rejects_non_sha_ordinal_receipt() -> None:
    result_by_split = {
        split: qualification._split_result(
            split=split,
            metrics=_passing_metrics(),
            model_state_sha256="a" * 64,
        )
        for split in qualification._QualificationLedger.ORDER
    }
    split_states: dict[str, dict[str, object]] = {}
    for split in qualification._QualificationLedger.ORDER:
        ordinal_hashes = [f"{ordinal + 1:064x}" for ordinal in range(64)]
        split_states[split] = {
            "access_started": True,
            "status": "passed",
            "result_sha256": qualification.canonical_sha256(result_by_split[split]),
            "completed_ordinal_count": 64,
            "materialized_ordinal_count": 64,
            "active_ordinal": None,
            "ordinal_evidence_sha256s": ordinal_hashes,
            "active_batch_ordinals": None,
            "completed_batch_count": 16,
            "batch_evidence_sha256s": [
                qualification.canonical_sha256(ordinal_hashes[4 * batch : 4 * batch + 4])
                for batch in range(16)
            ],
        }
    bindings = {"binding": "value"}
    report = {
        "passed": True,
        "stopped_after": "final_test",
        **result_by_split,
    }
    record = {
        "artifact_kind": qualification._QualificationLedger.ARTIFACT_KIND,
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": qualification.MAX_ARCHITECTURE_ATTEMPTS,
        "order": list(qualification._QualificationLedger.ORDER),
        "bindings": bindings,
        "splits": split_states,
        "attempt_reserved": True,
        "protected_data_materialized": True,
        "status": "complete",
        "outcome": "passed",
        "stopped_after": "final_test",
        "report_sha256": "f" * 64,
    }
    qualification._validate_qualification_ledger_record(
        record,
        report=report,
        report_sha256="f" * 64,
        bindings=bindings,
    )
    tampered = copy.deepcopy(record)
    hashes = tampered["splits"]["confirmation"]["ordinal_evidence_sha256s"]
    hashes[3] = "not-a-sha"
    tampered["splits"]["confirmation"]["batch_evidence_sha256s"][0] = (
        qualification.canonical_sha256(hashes[:4])
    )
    with pytest.raises(ValueError, match="exact SHA-256"):
        qualification._validate_qualification_ledger_record(
            tampered,
            report=report,
            report_sha256="f" * 64,
            bindings=bindings,
        )
    bad_stop_record = copy.deepcopy(record)
    bad_stop_report = copy.deepcopy(report)
    bad_stop_record["stopped_after"] = "selector"
    bad_stop_report["stopped_after"] = "selector"
    with pytest.raises(ValueError, match="stop/outcome semantics"):
        qualification._validate_qualification_ledger_record(
            bad_stop_record,
            report=bad_stop_report,
            report_sha256="f" * 64,
            bindings=bindings,
        )


def test_full_report_validators_reject_bogus_materialization_and_stop() -> None:
    development_report, source, publication, certificate = _valid_report_evidence()
    qualification._development_report_is_valid(
        development_report,
        source=source,
        publication=publication,
        certificate=certificate,
    )
    bad_development = copy.deepcopy(development_report)
    bad_development["protected_data_materialized"] = True
    with pytest.raises(ValueError, match="cannot claim protected"):
        qualification._development_report_is_valid(
            bad_development,
            source=source,
            publication=publication,
            certificate=certificate,
        )

    state_sha256 = development_report["calibration"]["calibrated_model_state_sha256"]
    protected = {
        split: qualification._split_result(
            split=split,
            metrics=copy.deepcopy(development_report["development"]["metrics"]),
            model_state_sha256=state_sha256,
        )
        for split in qualification._QualificationLedger.ORDER
    }
    report = {
        "artifact_kind": "rgbd_identifiable_drag_qualification",
        "protocol": qualification.bridge_protocol(),
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": qualification.FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(qualification.FROZEN_SOURCE_SHA256),
        "scene_family_certificate": certificate,
        "qualification_ledger": str(qualification.qualification_ledger_path()),
        "reviewed_checkpoint_sha256": "3" * 64,
        "reviewed_development_report_sha256": "4" * 64,
        "reviewed_development_ledger_sha256": "5" * 64,
        "model_state_sha256": state_sha256,
        "optimizer_updates": 0,
        "development": development_report["development"],
        "calibration": development_report["calibration"],
        **protected,
        "protected_data_materialized": True,
        "passed": True,
        "stopped_after": "final_test",
    }
    qualification._qualification_report_is_valid(
        report,
        source=source,
        publication=publication,
        certificate=certificate,
    )
    for field, value, message in (
        ("stopped_after", "selector", "stopped_after"),
        ("protected_data_materialized", False, "materialization"),
    ):
        tampered = copy.deepcopy(report)
        tampered[field] = value
        with pytest.raises(ValueError, match=message):
            qualification._qualification_report_is_valid(
                tampered,
                source=source,
                publication=publication,
                certificate=certificate,
            )


def test_error_report_validator_rejects_arbitrary_false_json() -> None:
    report, source, publication, certificate = _valid_report_evidence()
    report.update(
        {
            "checkpoint": None,
            "checkpoint_sha256": None,
            "checkpoint_model_state_sha256": None,
            "passed": False,
            "review_ready": False,
            "error": {"type": "RuntimeError", "message": "injected"},
        }
    )
    qualification._development_error_report_is_valid(
        report,
        source=source,
        publication=publication,
        certificate=certificate,
    )
    report["stopped_after"] = "bogus"
    with pytest.raises(ValueError, match="execution semantics"):
        qualification._development_error_report_is_valid(
            report,
            source=source,
            publication=publication,
            certificate=certificate,
        )


def test_prepare_report_requires_exact_first_failed_split() -> None:
    ledger = object.__new__(qualification._QualificationLedger)
    states = {
        split: {
            "access_started": split == "selector",
            "status": "failed" if split == "selector" else "unopened",
        }
        for split in qualification._QualificationLedger.ORDER
    }
    record = {"splits": states}
    ledger._transition_record = lambda **kwargs: copy.deepcopy(record)
    ledger._replace = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="last opened split"):
        ledger.prepare_report(passed=False, stopped_after="final_test")
    ledger.prepare_report(passed=False, stopped_after="selector")


def test_private_constructor_and_evaluator_are_not_exported_or_used_posthoc() -> None:
    assert "_construct_identifiable_drag_episode" not in qualification.__all__
    assert "_evaluate_authorized_batch" not in qualification.__all__
    source = inspect.getsource(qualification._calibrated_development_evidence)
    for forbidden in (
        "_construct_identifiable_drag_episode",
        "_evaluate_authorized_batch",
        "render_spheres",
        ".ingest(",
        ".predict(",
    ):
        assert forbidden not in source
    setter_source = inspect.getsource(qualification._calibrated_development_evidence)
    assert setter_source.count(".set_development_uncertainty_scales(") == 1


def test_checkpoint_writer_restricted_roundtrip_precedes_publication() -> None:
    source = inspect.getsource(qualification._save_review_checkpoint)
    restricted = source.index("_checkpoint_payload_from_bytes(contents)")
    full_validation = source.index("_validate_checkpoint_evidence(", restricted)
    publication = source.rindex("_durable_create(path, contents)")
    assert restricted < full_validation < publication


def test_cli_requires_all_three_external_review_hashes() -> None:
    script = (
        Path(__file__).parents[2] / "scripts" / "run_rgbd_identifiable_drag_qualification.py"
    ).read_text()
    assert "--reviewed-checkpoint-sha256" in script
    assert "--reviewed-report-sha256" in script
    assert "--reviewed-development-ledger-sha256" in script


def test_scene_certificate_binding_is_literal_only_and_returns_deep_copies() -> None:
    source = Path(qualification.__file__).read_text()
    scene_import_start = source.index(
        "from world_model.training.rgbd_identifiable_drag_scene import ("
    )
    scene_import_end = source.index("\n)", scene_import_start)
    assert "scene_family_certificate" not in source[scene_import_start:scene_import_end]
    assert "scene_family_certificate()" not in source

    first = qualification._frozen_scene_certificate_binding()
    second = qualification._frozen_scene_certificate_binding()
    assert first == second
    assert first is not second
    assert set(first) == qualification._FROZEN_SCENE_CERTIFICATE_BINDING_SCHEMA
    assert first["artifact_kind"] == ("rgbd_identifiable_drag_scene_family_offline_source_freeze")
    assert first["runtime_recomputation_permitted"] is False
    assert first["scenes_per_split"] == qualification.SCENES_PER_SPLIT
    assert first["splits"] == list(qualification.SPLITS)
    expected_globals = {
        "certificate_sha256": qualification.FROZEN_CERTIFICATE_SHA256,
        "metadata_sha256": qualification.FROZEN_METADATA_SHA256,
        "physical_trace_sha256": qualification.FROZEN_PHYSICAL_TRACE_SHA256,
        "camera_trace_sha256": qualification.FROZEN_CAMERA_TRACE_SHA256,
        "raster_trace_sha256": qualification.FROZEN_RASTER_TRACE_SHA256,
        "combined_trace_sha256": qualification.FROZEN_COMBINED_TRACE_SHA256,
    }
    for name, expected in expected_globals.items():
        assert first[name] == expected
        qualification.validated_sha256(first[name], label=name)
    expected_split_maps = {
        "split_physical_trace_sha256": qualification.FROZEN_SPLIT_PHYSICAL_TRACE_SHA256,
        "split_camera_trace_sha256": qualification.FROZEN_SPLIT_CAMERA_TRACE_SHA256,
        "split_raster_trace_sha256": qualification.FROZEN_SPLIT_RASTER_TRACE_SHA256,
        "split_combined_trace_sha256": qualification.FROZEN_SPLIT_COMBINED_TRACE_SHA256,
    }
    for name, expected in expected_split_maps.items():
        assert first[name] == expected
        assert first[name] is not second[name]
        for split in qualification.SPLITS:
            qualification.validated_sha256(first[name][split], label=f"{name} {split}")
    first["split_physical_trace_sha256"]["development"] = "0" * 64
    assert second["split_physical_trace_sha256"]["development"] != "0" * 64


def test_development_terminal_reconciliation_rejects_substituted_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    run_directory = qualification._canonical_run_directory()
    run_directory.mkdir(parents=True)
    report, source, publication, certificate = _valid_report_evidence()
    substituted_source = copy.deepcopy(source)
    substituted_source["commit"] = "e" * 40
    substituted_publication = copy.deepcopy(publication)
    substituted_publication["head_commit"] = "e" * 40
    substituted_publication["upstream_commit"] = "e" * 40
    report["source_provenance"] = substituted_source
    report["publication_provenance"] = substituted_publication
    report_contents = _json_artifact(report)
    bindings = _development_bindings(source, publication)
    record = _terminal_development_record(
        report,
        bindings=bindings,
        report_sha256=qualification.sha256_bytes(report_contents),
    )
    qualification._development_report_is_valid(
        report,
        source=substituted_source,
        publication=substituted_publication,
        certificate=certificate,
    )
    qualification._validate_terminal_development_ledger_record(
        record,
        report=report,
        report_sha256=qualification.sha256_bytes(report_contents),
        checkpoint_sha256=report["checkpoint_sha256"],
        bindings=bindings,
    )
    qualification.canonical_checkpoint_path().write_bytes(b"substituted checkpoint")
    qualification.canonical_development_report_path().write_bytes(report_contents)
    qualification.development_ledger_path().write_bytes(_json_artifact(record))
    ledger = object.__new__(qualification._DevelopmentLedger)
    ledger._bindings = bindings
    ledger._config = object()
    monkeypatch.setattr(
        qualification,
        "_require_config_matches_frozen_path",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        qualification,
        "_current_execution_provenance",
        lambda **kwargs: (source, publication, certificate),
    )
    assert not qualification._terminal_commit_matches_disk(
        ledger,
        qualification=False,
    )


def test_qualification_terminal_reconciliation_rejects_substituted_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    run_directory = qualification._canonical_run_directory()
    run_directory.mkdir(parents=True)
    development_report, source, publication, certificate = _valid_report_evidence()
    state_sha256 = development_report["calibration"]["calibrated_model_state_sha256"]
    protected = {
        split: qualification._split_result(
            split=split,
            metrics=copy.deepcopy(development_report["development"]["metrics"]),
            model_state_sha256=state_sha256,
        )
        for split in qualification._QualificationLedger.ORDER
    }
    substituted_source = copy.deepcopy(source)
    substituted_source["commit"] = "e" * 40
    substituted_publication = copy.deepcopy(publication)
    substituted_publication["head_commit"] = "e" * 40
    substituted_publication["upstream_commit"] = "e" * 40
    report = {
        "artifact_kind": "rgbd_identifiable_drag_qualification",
        "protocol": qualification.bridge_protocol(),
        "source_provenance": substituted_source,
        "publication_provenance": substituted_publication,
        "config_sha256": qualification.FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(qualification.FROZEN_SOURCE_SHA256),
        "scene_family_certificate": certificate,
        "qualification_ledger": str(qualification.qualification_ledger_path()),
        "reviewed_checkpoint_sha256": "3" * 64,
        "reviewed_development_report_sha256": "4" * 64,
        "reviewed_development_ledger_sha256": "5" * 64,
        "model_state_sha256": state_sha256,
        "optimizer_updates": 0,
        "development": development_report["development"],
        "calibration": development_report["calibration"],
        **protected,
        "protected_data_materialized": True,
        "passed": True,
        "stopped_after": "final_test",
    }
    bindings = {
        "protocol_sha256": qualification.bridge_protocol()["protocol_sha256"],
        "source_provenance": source,
        "publication_provenance": publication,
        "config_sha256": qualification.FROZEN_CONFIG_SHA256,
        "critical_source_sha256": dict(qualification.FROZEN_SOURCE_SHA256),
        "certificate_sha256": qualification.FROZEN_CERTIFICATE_SHA256,
        "reviewed_checkpoint_sha256": "3" * 64,
        "reviewed_development_report_sha256": "4" * 64,
        "reviewed_development_ledger_sha256": "5" * 64,
        "model_state_sha256": state_sha256,
        "calibration_sha256": qualification.canonical_sha256(development_report["calibration"]),
    }
    report_contents = _json_artifact(report)
    record = _terminal_qualification_record(
        report,
        bindings=bindings,
        report_sha256=qualification.sha256_bytes(report_contents),
    )
    qualification._qualification_report_is_valid(
        report,
        source=substituted_source,
        publication=substituted_publication,
        certificate=certificate,
    )
    qualification._validate_qualification_ledger_record(
        record,
        report=report,
        report_sha256=qualification.sha256_bytes(report_contents),
        bindings=bindings,
    )
    qualification.canonical_checkpoint_path().write_bytes(b"substituted checkpoint")
    qualification.canonical_development_report_path().write_bytes(b"substituted development report")
    qualification.development_ledger_path().write_bytes(b"substituted development ledger")
    qualification.canonical_qualification_report_path().write_bytes(report_contents)
    qualification.qualification_ledger_path().write_bytes(_json_artifact(record))
    ledger = object.__new__(qualification._QualificationLedger)
    ledger._bindings = bindings
    ledger._config = object()
    monkeypatch.setattr(
        qualification,
        "_require_config_matches_frozen_path",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        qualification,
        "_current_execution_provenance",
        lambda **kwargs: (source, publication, certificate),
    )
    assert not qualification._terminal_commit_matches_disk(
        ledger,
        qualification=True,
    )


def test_exception_paths_do_not_preserve_rejected_terminal_pairs() -> None:
    for runner, persistence in (
        (qualification.run_development, "_persist_development_error("),
        (qualification.run_qualification, "_persist_qualification_error("),
    ):
        source = inspect.getsource(runner)
        reconciliation = source.rindex("terminal_committed = _terminal_commit_matches_disk(")
        persistence_call = source.rindex(persistence)
        assert reconciliation < persistence_call
        assert "not terminal_committed" in source[reconciliation:persistence_call]
