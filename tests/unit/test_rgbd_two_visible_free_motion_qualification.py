from __future__ import annotations

import io
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import world_model.training.rgbd_two_visible_free_motion_qualification as qualification
from world_model.training.checkpointing import load_model_weights
from world_model.training.rgbd_two_visible_free_motion_qualification import (
    AXIS_NAMES,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EMPTY_MODEL_STATE_SHA256,
    FINAL_TEST_SEEDS,
    FROZEN_CONFIG_SHA256,
    HORIZONS_SECONDS,
    OBJECT_INDICES,
    SELECTOR_SEEDS,
    VJP_COEFFICIENTS,
    VJP_OUTPUTS,
    DevelopmentLedger,
    QualificationLedger,
    _history_gradient_diagnostics,
    _load_checkpoint_payload,
    _require_config_matches_frozen_path,
    _save_review_checkpoint,
    assert_rgbd_two_visible_config,
    bridge_protocol,
    canonical_sha256,
    evaluate_seed_manifest,
    gate_failures,
    new_public_model,
    run_development,
    run_qualification,
    scene_specification,
    sha256_file,
    validate_checkpoint_evidence,
    validate_development_evidence,
    validate_distinct_paths,
)
from world_model.utils.config import load_config

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_two_visible_free_motion_cpu.yaml"


class _UnsafeCheckpointValue:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (eval, ("40 + 2",))


def _passing_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {
        "current_position_rmse_m": 0.0,
        "current_velocity_rmse_mps": 0.0,
        "maximum_position_error_growth_slope_mps": 0.0,
        "early_stationary_additive_regression_m": 0.0,
        "long_stationary_rmse_ratio": 0.0,
        "zero_velocity_rmse_ratio": 0.0,
        "identity_switch_count": 0.0,
        "persistent_id_mismatch_count": 0.0,
        "identity_coverage": 1.0,
        "persistent_object_id_min": 0.0,
        "persistent_object_id_max": 1.0,
        "association_pair_coverage": 1.0,
        "association_ambiguous_pair_count": 0.0,
        "minimum_hungarian_margin": 1.0,
        "minimum_position_assignment_margin_m": 1.0,
        "minimum_matched_appearance_cosine": 1.0,
        "minimum_cross_appearance_cosine_distance": 1.0,
        "physical_palette_swap_fraction": 0.5,
        "birth_slot_physical_zero_fraction": 0.5,
        "unique_scene_specification_fraction": 1.0,
        "gradient_audit_scene_count": 4.0,
        "gradient_audit_unique_scene_fraction": 1.0,
        "active_fraction": 1.0,
        "rollout_active_fraction": 1.0,
        "history_sample_count_min": 16.0,
        "history_sample_count_max": 16.0,
        "history_valid_count_min": 16.0,
        "history_valid_count_max": 16.0,
        "history_span_max_abs_error_seconds": 0.0,
        "direct_velocity_calls_per_batch_min": 1.0,
        "direct_velocity_calls_per_batch_max": 1.0,
        "direct_velocity_valid_fraction": 1.0,
        "position_owner_count_min": 1.0,
        "position_owner_count_max": 1.0,
        "direct_position_field_count": 0.0,
        "direct_velocity_position_change_max_abs_m": 0.0,
        "direct_metric_position_owner_max_abs_m": 0.0,
        "ambiguity_direct_position_write_count": 0.0,
        "ambiguity_direct_velocity_write_count": 0.0,
        "preflight_minimum_silhouette_gap_pixels": 10.0,
        "preflight_minimum_boundary_clearance_pixels": 10.0,
        "preflight_minimum_world_surface_gap_m": 1.0,
        "preflight_minimum_world_boundary_clearance_m": 1.0,
        "preflight_minimum_visible_fraction": 1.0,
        "preflight_event_count": 0.0,
        "preflight_minimum_palette_cosine_distance": 1.0,
        "semigroup_position_max_abs_m": 0.0,
        "semigroup_velocity_max_abs_mps": 0.0,
        "public_direct_position_max_abs_m": 0.0,
        "public_direct_velocity_max_abs_mps": 0.0,
        "analytic_position_agreement_max_abs_m": 0.0,
        "analytic_velocity_agreement_max_abs_mps": 0.0,
        "public_rollout_output_alias_count": 0.0,
        "public_query_time_max_abs_seconds": 0.0,
        "ingested_frame_count_min": 16.0,
        "ingested_frame_count_max": 16.0,
        "public_predict_calls_per_batch_min": 1.0,
        "public_predict_calls_per_batch_max": 1.0,
        "perception_latency_seconds": 0.0,
        "state_only_rollout_latency_seconds": 0.0,
        "persistent_runtime_tensor_state_bytes_max": 0.0,
        "process_max_rss_bytes": 0.0,
        "process_rss_delta_bytes": 0.0,
        "learned_parameter_count": 0.0,
        "learned_parameter_bytes": 0.0,
        "module_tensor_buffer_count": 0.0,
        "persistent_module_state_key_count": 0.0,
        "persistent_module_state_bytes": 0.0,
        "optimizer_updates": 0.0,
        "optimizer_state_entry_count": 0.0,
    }
    for object_index in OBJECT_INDICES:
        for axis in AXIS_NAMES:
            metrics[f"current_position_rmse_m/object_{object_index}/{axis}"] = 0.0
            metrics[f"current_velocity_rmse_mps/object_{object_index}/{axis}"] = 0.0
    for horizon in HORIZONS_SECONDS:
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = 0.0
        metrics[f"horizon_{label}_velocity_rmse_mps"] = 0.0
        for object_index in OBJECT_INDICES:
            for axis in AXIS_NAMES:
                metrics[f"horizon_{label}_position_rmse_m/object_{object_index}/{axis}"] = 0.0
                metrics[f"horizon_{label}_velocity_rmse_mps/object_{object_index}/{axis}"] = 0.0
    for object_index in OBJECT_INDICES:
        for output_name in VJP_OUTPUTS:
            for modality in ("rgb", "depth"):
                suffix = f"object_{object_index}/{output_name}/{modality}"
                metrics[f"gradient_l1/{suffix}"] = 1.0
                metrics[f"gradient_max_l1/{suffix}"] = 1.0
                metrics[f"gradient_cross_scene_max_l1/{suffix}"] = 0.0
                if output_name == "current_position":
                    metrics[f"gradient_anchor_history_frame_l1/{suffix}"] = 1.0
                    metrics[f"gradient_nonanchor_max_history_frame_l1/{suffix}"] = 0.0
                    metrics[f"gradient_supported_history_frames/{suffix}"] = 1.0
                else:
                    metrics[f"gradient_min_history_frame_l1/{suffix}"] = 1.0
                    metrics[f"gradient_supported_history_frames/{suffix}"] = 16.0
    return metrics


def _reviewed_report() -> tuple[dict[str, object], dict[str, object], str]:
    metrics = _passing_metrics()
    development: dict[str, object] = {
        "split": "development",
        "seeds": list(DEVELOPMENT_SEEDS),
        "seed_manifest_sha256": canonical_sha256(list(DEVELOPMENT_SEEDS)),
        "metrics": metrics,
        "failures": [],
        "passed": True,
        "optimizer_updates": 0,
        "runtime_api": {
            "packet_factory": "make_rgbd_packet",
            "ingest_frames": list(range(16)),
            "rollout_method": "OnlineWorldModel.predict",
            "horizons_seconds": list(HORIZONS_SECONDS),
        },
        "scene_constructor": "construct_two_visible_episode_with_full_frame_preflight",
    }
    source: dict[str, object] = {
        "commit": "1" * 40,
        "dirty": False,
        "worktree_fingerprint": "2" * 64,
        "runtime_source_fingerprint": "3" * 64,
    }
    checkpoint_sha256 = "4" * 64
    report: dict[str, object] = {
        "artifact_kind": "rgbd_two_visible_development",
        "protocol": json.loads(json.dumps(bridge_protocol())),
        "source_provenance": source,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development": development,
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "passed": True,
        "review_ready": True,
        "stopped_after": "development",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_model_state_sha256": EMPTY_MODEL_STATE_SHA256,
    }
    return report, source, checkpoint_sha256


def _fake_manifest_result(split: str, seeds: tuple[int, ...]) -> dict[str, object]:
    return {
        "split": split,
        "seeds": list(seeds),
        "seed_manifest_sha256": canonical_sha256(list(seeds)),
        "metrics": _passing_metrics(),
        "failures": [],
        "passed": True,
        "optimizer_updates": 0,
        "runtime_api": {
            "packet_factory": "make_rgbd_packet",
            "ingest_frames": list(range(16)),
            "rollout_method": "OnlineWorldModel.predict",
            "horizons_seconds": list(HORIZONS_SECONDS),
        },
        "scene_constructor": "construct_two_visible_episode_with_full_frame_preflight",
    }


def test_protocol_freezes_disjoint_namespaces_self_hash_and_vjp_contract() -> None:
    namespaces = (
        DEVELOPMENT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    assert tuple(map(len, namespaces)) == (32, 24, 24, 48)
    assert tuple(range(49_000_000, 49_000_032)) == DEVELOPMENT_SEEDS
    assert tuple(range(50_000_000, 50_000_024)) == SELECTOR_SEEDS
    assert tuple(range(51_000_000, 51_000_024)) == CONFIRMATION_SEEDS
    assert tuple(range(52_000_000, 52_000_048)) == FINAL_TEST_SEEDS
    flattened = tuple(seed for namespace in namespaces for seed in namespace)
    assert len(flattened) == len(set(flattened))
    assert len({seed & ((1 << 24) - 1) for seed in flattened}) == len(flattened)
    protocol = bridge_protocol()
    stated = protocol.pop("protocol_sha256")
    assert stated == canonical_sha256(protocol)
    assert protocol["differentiability"]["coefficients"] == list(VJP_COEFFICIENTS)
    assert protocol["differentiability"]["minimum_total_l1_per_object_target_modality"] == 1e-8
    assert protocol["differentiability"]["current_position_owner"] == "anchor_frame_15_only"
    assert protocol["differentiability"]["current_position_required_history_frames"] == 1
    assert (
        protocol["differentiability"]["temporal_output_minimum_l1_per_object_target_modality_frame"]
        == 1e-8
    )
    assert protocol["differentiability"]["temporal_output_required_history_frames"] == 16
    assert protocol["differentiability"]["cross_scene_gradient_max_l1"] == 0.0
    assert protocol["runtime"]["ambiguity_direct_writes"] == "fail_closed"
    assert protocol["scene_family"]["no_split_local_scene_period"] is True
    assert protocol["development_access"]["no_renamed_development_retry"] is True
    assert protocol["source_binding"]["commit"].startswith("captured_at_execution")


def test_exact_profile_binds_parameter_free_two_slot_runtime() -> None:
    config = load_config(CONFIG_PATH)
    assert sha256_file(CONFIG_PATH) == FROZEN_CONFIG_SHA256
    assert_rgbd_two_visible_config(config)
    assert config.model.rgbd.proposal_count == 2
    assert config.model.state.appearance_dim == 3
    assert config.model.association.ambiguity_margin == 0.02
    assert config.model.rgbd.minimum_silhouette_gap_pixels == 2.0
    assert config.model.rgbd.minimum_boundary_clearance_pixels == 2.0
    model = new_public_model(config)
    assert not tuple(model.parameters())
    assert not tuple(model.buffers())
    assert model.state_dict() == {}
    assert canonical_sha256([]) == EMPTY_MODEL_STATE_SHA256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("ambiguity_margin", 0.0, "model.association.ambiguity_margin"),
        ("appearance_weight", 0.0, "model.association.appearance_weight"),
    ),
)
def test_config_rejects_association_semantic_drift(field: str, value: float, message: str) -> None:
    config = load_config(CONFIG_PATH)
    changed = replace(
        config,
        model=replace(
            config.model,
            association=replace(config.model.association, **{field: value}),
        ),
    )
    with pytest.raises(ValueError, match=message):
        assert_rgbd_two_visible_config(changed)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("lifecycle", "birth_confirmations", 2),
        ("dynamics", "max_substep", 0.02),
        ("filter", "innovation_anchored_correction", False),
        ("runtime", "modality_order", ("rgbd",)),
    ),
)
def test_config_rejects_runtime_semantic_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    config = load_config(CONFIG_PATH)
    if section == "runtime":
        changed = replace(config, runtime=replace(config.runtime, **{field: value}))
    else:
        changed = replace(
            config,
            model=replace(
                config.model,
                **{
                    section: replace(
                        getattr(config.model, section),
                        **{field: value},
                    )
                },
            ),
        )
    with pytest.raises(ValueError):
        assert_rgbd_two_visible_config(changed)
    with pytest.raises(ValueError, match="executed config object differs"):
        _require_config_matches_frozen_path(changed, CONFIG_PATH)


def test_scene_specification_is_bounded_deterministic_and_palette_balanced() -> None:
    even = scene_specification(0)
    odd = scene_specification(1)
    repeated = scene_specification(0)
    later = scene_specification(8)
    torch.testing.assert_close(even.position, repeated.position)
    torch.testing.assert_close(even.velocity, repeated.velocity)
    torch.testing.assert_close(even.albedo, repeated.albedo)
    assert not torch.equal(later.position, even.position)
    assert not torch.equal(later.velocity, even.velocity)
    assert not even.palette_swapped and odd.palette_swapped
    assert even.albedo[0, 0] > even.albedo[1, 0]
    assert odd.albedo[0, 1] > odd.albedo[1, 1]
    assert torch.linalg.vector_norm(even.position[0] - even.position[1]) > 1.0
    assert (even.albedo >= 0).all() and (even.albedo <= 1).all()
    signatures = {
        canonical_sha256(
            {
                "position": scene_specification(seed).position.tolist(),
                "velocity": scene_specification(seed).velocity.tolist(),
                "albedo": scene_specification(seed).albedo.tolist(),
            }
        )
        for seed in range(128)
    }
    assert len(signatures) == 128


@pytest.mark.parametrize(
    ("key", "bad_value"),
    (
        ("current_position_rmse_m", 1.0),
        ("current_velocity_rmse_mps/object_1/z", 1.0),
        ("horizon_2.00_position_rmse_m/object_0/x", 1.0),
        ("identity_switch_count", 1.0),
        ("persistent_id_mismatch_count", 1.0),
        ("association_pair_coverage", 0.5),
        ("minimum_hungarian_margin", 0.0),
        ("minimum_position_assignment_margin_m", 0.0),
        ("birth_slot_physical_zero_fraction", 1.0),
        ("unique_scene_specification_fraction", 0.5),
        ("gradient_audit_scene_count", 1.0),
        ("ambiguity_direct_position_write_count", 1.0),
        ("preflight_minimum_silhouette_gap_pixels", 1.0),
        ("gradient_l1/object_1/current_velocity/rgb", 0.0),
        ("gradient_cross_scene_max_l1/object_0/current_position/depth", 1.0e-12),
        ("gradient_nonanchor_max_history_frame_l1/object_1/current_position/rgb", 1.0e-12),
        ("gradient_supported_history_frames/object_0/current_position/depth", 16.0),
        (
            "gradient_supported_history_frames/object_0/horizon_2.00_position/depth",
            15.0,
        ),
        ("persistent_runtime_tensor_state_bytes_max", 65_537.0),
        ("learned_parameter_count", 1.0),
    ),
)
def test_gate_recomputation_fails_closed(key: str, bad_value: float) -> None:
    metrics = _passing_metrics()
    assert gate_failures(metrics) == []
    metrics[key] = bad_value
    assert any(failure.startswith(f"{key}:") for failure in gate_failures(metrics))


def test_per_object_vjp_diagnostic_exposes_one_detached_history_frame() -> None:
    gradient = torch.full((1, 16, 3, 2, 2), 1.0e-6)
    gradient[:, 9].zero_()
    diagnostics = _history_gradient_diagnostics(
        gradient,
        object_index=1,
        output_name="horizon_2.00_position",
        modality="rgb",
    )
    prefix = "object_1/horizon_2.00_position/rgb"
    assert diagnostics[f"gradient_min_history_frame_l1/{prefix}"] == 0.0
    assert diagnostics[f"gradient_supported_history_frames/{prefix}"] == 15.0
    metrics = _passing_metrics()
    metrics.update(diagnostics)
    assert any(prefix in failure for failure in gate_failures(metrics))


def test_current_position_vjp_is_owned_only_by_the_anchor_measurement() -> None:
    gradient = torch.zeros((1, 16, 1, 2, 2))
    gradient[:, 15] = 1.0e-6
    diagnostics = _history_gradient_diagnostics(
        gradient,
        object_index=0,
        output_name="current_position",
        modality="depth",
    )
    prefix = "object_0/current_position/depth"
    assert diagnostics[f"gradient_anchor_history_frame_l1/{prefix}"] == pytest.approx(4.0e-6)
    assert diagnostics[f"gradient_nonanchor_max_history_frame_l1/{prefix}"] == 0.0
    assert diagnostics[f"gradient_supported_history_frames/{prefix}"] == 1.0
    metrics = _passing_metrics()
    metrics.update(diagnostics)
    assert gate_failures(metrics) == []


def test_invalid_manifest_is_rejected_before_scene_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    calls: list[int] = []

    def forbidden_constructor(*_args: object, **_kwargs: object) -> object:
        calls.append(1)
        raise AssertionError("scene constructor must remain unopened")

    monkeypatch.setattr(qualification, "construct_two_visible_episode", forbidden_constructor)
    with pytest.raises(ValueError, match="exact frozen"):
        evaluate_seed_manifest(config, DEVELOPMENT_SEEDS[:-1], split="development")
    assert calls == []


@pytest.mark.parametrize(
    ("split", "seeds"),
    (
        ("selector", SELECTOR_SEEDS),
        ("confirmation", CONFIRMATION_SEEDS),
        ("final_test", FINAL_TEST_SEEDS),
    ),
)
def test_exact_protected_manifest_requires_durable_authorization_before_constructor(
    split: str,
    seeds: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    calls: list[int] = []

    def forbidden_constructor(*_args: object, **_kwargs: object) -> object:
        calls.append(1)
        raise AssertionError("protected constructor must remain unopened")

    monkeypatch.setattr(qualification, "construct_two_visible_episode", forbidden_constructor)
    with pytest.raises(PermissionError, match="durable access authorization"):
        evaluate_seed_manifest(config, seeds, split=split)
    assert calls == []


def test_protected_receipt_is_durable_before_constructor_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    ledger = QualificationLedger({"protocol_sha256": "0" * 64})
    authorization = ledger.begin_access("selector")
    calls: list[int] = []

    def inspect_receipt(*_args: object, **_kwargs: object) -> object:
        record = json.loads(ledger.path.read_text(encoding="utf-8"))
        assert record["status"] == "selector_materialization_started"
        assert record["splits"]["selector"]["access_started"] is True
        calls.append(1)
        raise RuntimeError("seed-free stop after durable receipt")

    monkeypatch.setattr(qualification, "construct_two_visible_episode", inspect_receipt)
    with pytest.raises(RuntimeError, match="seed-free stop"):
        evaluate_seed_manifest(
            config,
            SELECTOR_SEEDS,
            split="selector",
            authorization=authorization,
        )
    assert calls == [1]


def test_terminal_ledger_failure_cannot_remain_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    ledger = QualificationLedger({"protocol_sha256": "0" * 64})
    for split in ledger.ORDER:
        authorization = ledger.begin_access(split)
        _consume_authorization(
            authorization,
            split=split,
            seeds=ledger.MANIFESTS[split],
        )
        ledger.complete_split(split, {"split": split, "passed": True})
    ledger.prepare_report(passed=True, stopped_after="final_test")
    ledger.record_error(OSError("injected report failure"), stopped_after="final_test")
    record = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["error"]["type"] == "OSError"


def test_runner_ledgers_block_development_retry_and_terminal_report_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    source = {
        "commit": "1" * 40,
        "dirty": False,
        "worktree_fingerprint": "2" * 64,
        "runtime_source_fingerprint": "3" * 64,
    }
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(qualification, "capture_git_metadata", lambda _root: source)

    def seed_free_evaluate(
        _config: object,
        seeds: tuple[int, ...],
        *,
        split: str,
        authorization: object,
    ) -> dict[str, object]:
        requested = tuple(seeds)
        _consume_authorization(authorization, split=split, seeds=requested)
        return _fake_manifest_result(split, requested)

    monkeypatch.setattr(qualification, "evaluate_seed_manifest", seed_free_evaluate)
    development_report = tmp_path / "development.json"
    checkpoint = tmp_path / "development.pt"
    assert (
        run_development(
            config,
            config_path=CONFIG_PATH,
            report_path=development_report,
            checkpoint_path=checkpoint,
            source_provenance=source,
        )
        == 0
    )
    development_record = json.loads(
        qualification.development_ledger_path().read_text(encoding="utf-8")
    )
    assert development_record["status"] == "complete"
    with pytest.raises(FileExistsError):
        run_development(
            config,
            config_path=CONFIG_PATH,
            report_path=tmp_path / "renamed-development.json",
            checkpoint_path=tmp_path / "renamed-development.pt",
            source_provenance=source,
        )

    report_digest = sha256_file(development_report)
    checkpoint_digest = sha256_file(checkpoint)
    qualification_report = tmp_path / "qualification.json"
    original_writer = qualification.write_report_fresh

    def fail_terminal_report(path: Path, report: object) -> None:
        if Path(path) == qualification_report:
            raise OSError("injected terminal report failure")
        original_writer(path, report)

    monkeypatch.setattr(qualification, "write_report_fresh", fail_terminal_report)
    with pytest.raises(OSError, match="injected terminal report failure"):
        run_qualification(
            config,
            config_path=CONFIG_PATH,
            report_path=qualification_report,
            checkpoint_path=checkpoint,
            development_report_path=development_report,
            reviewed_checkpoint_sha256=checkpoint_digest,
            reviewed_report_sha256=report_digest,
            source_provenance=source,
        )
    qualification_record = json.loads(
        qualification.qualification_ledger_path().read_text(encoding="utf-8")
    )
    assert qualification_record["status"] == "error"
    assert qualification_record["error"]["type"] == "OSError"
    assert not qualification_report.exists()


def test_reviewed_development_and_empty_checkpoint_roundtrip_validate() -> None:
    report, source, checkpoint_sha256 = _reviewed_report()
    development = validate_development_evidence(
        report,
        checkpoint_sha256=checkpoint_sha256,
        source=source,
    )
    config = load_config(CONFIG_PATH)
    payload = {
        "model_state": {},
        "step": 0,
        "optimizer_state": None,
        "scheduler_state": None,
        "config": config.to_dict(),
        "git": source,
        "metrics": {
            "artifact_kind": "rgbd_two_visible_empty_model_state",
            "optimizer_updates": 0,
            "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
            "protocol": bridge_protocol(),
            "development": development,
        },
    }
    validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        development=development,
    )


def test_serialized_review_checkpoint_is_restricted_and_publicly_reloadable(
    tmp_path: Path,
) -> None:
    report, source, _checkpoint_sha256 = _reviewed_report()
    development = report["development"]
    assert isinstance(development, dict)
    config = load_config(CONFIG_PATH)
    model = new_public_model(config)
    checkpoint_path = tmp_path / "reviewed.pt"
    metrics = {
        "artifact_kind": "rgbd_two_visible_empty_model_state",
        "optimizer_updates": 0,
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "protocol": bridge_protocol(),
        "development": development,
    }
    _save_review_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        metrics=metrics,
        source=source,
    )
    payload = _load_checkpoint_payload(checkpoint_path.read_bytes())
    validate_checkpoint_evidence(
        payload,
        config=config,
        source=source,
        development=development,
    )
    reloaded = new_public_model(config)
    load_model_weights(checkpoint_path, model=reloaded, expected_config=config)
    assert reloaded.state_dict() == {}


def test_restricted_checkpoint_loader_rejects_untrusted_pickle_global() -> None:
    buffer = io.BytesIO()
    torch.save({"unsafe": _UnsafeCheckpointValue()}, buffer)
    with pytest.raises(pickle.UnpicklingError):
        _load_checkpoint_payload(buffer.getvalue())


def test_reviewed_development_rejects_protocol_tampering() -> None:
    report, source, checkpoint_sha256 = _reviewed_report()
    protocol = report["protocol"]
    assert isinstance(protocol, dict)
    gates = protocol["gates"]
    assert isinstance(gates, dict)
    gates["minimum_hungarian_margin"] = 0.0
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol["protocol_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="protocol differs from frozen source"):
        validate_development_evidence(
            report,
            checkpoint_sha256=checkpoint_sha256,
            source=source,
        )


def _consume_authorization(
    authorization: object,
    *,
    split: str,
    seeds: tuple[int, ...],
) -> None:
    authorization.begin_manifest(split, seeds)  # type: ignore[attr-defined]
    for seed in seeds:
        authorization.authorize_seed(seed)  # type: ignore[attr-defined]
    authorization.finish_manifest()  # type: ignore[attr-defined]


def test_artifact_aliases_and_exactly_once_ledger_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="artifact paths alias"):
        validate_distinct_paths(
            {"report": report, "temporary": report.with_suffix(".json.tmp")},
            atomic_writers=("report",),
        )
    source = tmp_path / "source.json"
    alias = tmp_path / "alias.json"
    source.write_text("{}", encoding="utf-8")
    os.link(source, alias)
    with pytest.raises(ValueError, match="hard-link alias"):
        validate_distinct_paths(
            {"source": source, "alias": alias},
            atomic_writers=(),
        )

    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", tmp_path)
    ledger_path = qualification.qualification_ledger_path()
    ledger = QualificationLedger({"protocol_sha256": "0" * 64})
    with pytest.raises(FileExistsError):
        QualificationLedger({"protocol_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="must remain unopened"):
        ledger.begin_access("final_test")
    for split in ledger.ORDER:
        authorization = ledger.begin_access(split)
        with pytest.raises(RuntimeError, match="cannot be opened twice"):
            ledger.begin_access(split)
        _consume_authorization(
            authorization,
            split=split,
            seeds=ledger.MANIFESTS[split],
        )
        ledger.complete_split(split, {"split": split, "passed": True})
    ledger.prepare_report(passed=True, stopped_after="final_test")
    ledger.finish(report_sha256="a" * 64)
    record = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["report_sha256"] == "a" * 64
    assert record["protected_data_materialized"] is True
    assert all(record["splits"][split]["status"] == "passed" for split in ledger.ORDER)
    assert not ledger_path.with_suffix(".json.tmp").exists()

    development = DevelopmentLedger({"protocol_sha256": "0" * 64})
    with pytest.raises(FileExistsError):
        DevelopmentLedger({"protocol_sha256": "0" * 64})
    development_authorization = development.authorization()
    with pytest.raises(RuntimeError, match="cannot be issued twice"):
        development.authorization()
    _consume_authorization(
        development_authorization,
        split="development",
        seeds=DEVELOPMENT_SEEDS,
    )
    development.complete_evaluation({"split": "development", "passed": True})
    development.finish(report_sha256="b" * 64, checkpoint_sha256="c" * 64)
    development_record = json.loads(development.path.read_text(encoding="utf-8"))
    assert development_record["status"] == "complete"
    assert development_record["report_sha256"] == "b" * 64
