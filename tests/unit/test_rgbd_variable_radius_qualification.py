"""Static and fake-only tests for the variable-radius qualification harness.

These tests exercise schemas, capabilities, durable receipts, and provenance
guards. They deliberately never call a formal scene API or OnlineWorldModel.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import io
import os
import pickle
import stat
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
import torch

from world_model.observations import MeasurementSet, ObservationPacket
from world_model.training import rgbd_variable_radius_qualification as q


def _harmless_unsafe_checkpoint_constructor() -> dict[str, object]:
    return {"would_have_executed": True}


class _UnsafeCheckpointGlobal:
    def __reduce__(self) -> tuple[object, tuple[()]]:
        return _harmless_unsafe_checkpoint_constructor, ()


@pytest.fixture(autouse=True)
def _clear_fake_registries() -> None:
    yield
    q._CAPABILITY_REGISTRY.clear()
    q._BATCH_REGISTRY.clear()
    q._BATCH_COMMIT_REGISTRY.clear()
    q._LEDGER_REGISTRY.clear()
    q._RUNNER_INVOCATION_REGISTRY.clear()
    q._clear_run_authorization_vault_for_tests()
    q._EPISODE_REGISTRY.clear()
    q._PACKET_REGISTRY.clear()
    q._EVIDENCE_REGISTRY.clear()
    q._clear_pinned_directory_vault_for_tests()


def _fake_directory_pin(tmp_path: Path) -> q._PinnedDirectory:
    return q._acquire_pinned_directory(tmp_path, create=False, canonical=False)


def _fake_ledger(tmp_path: Path, *, stage: str = "development") -> q._AccessLedger:
    directory_pin = _fake_directory_pin(tmp_path)
    return q._AccessLedger(
        tmp_path / f"{stage}.json",
        stage=stage,
        bindings={"fixture": "static"},
        directory_pin=directory_pin,
    )


def _fake_ledger_bytes(ledger: q._AccessLedger, *, label: str) -> bytes:
    return q._pinned_stable_read_bytes(ledger._directory_pin, ledger.path, label=label)


def _fake_live_measurement() -> tuple[MeasurementSet, torch.Tensor]:
    radius_source = torch.linspace(
        0.1,
        0.8,
        q.BATCH_SIZE * 2,
        dtype=torch.float32,
        requires_grad=True,
    ).reshape(q.BATCH_SIZE, 2, 1)
    radius = 0.2 + radius_source.square() * 0.01
    values = torch.zeros(q.BATCH_SIZE, 2, 3, dtype=torch.float32)
    measured = MeasurementSet(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=torch.zeros(q.BATCH_SIZE, dtype=torch.float32),
        values=values,
        log_variance=torch.zeros_like(values),
        existence_logits=torch.zeros(q.BATCH_SIZE, 2, dtype=torch.float32),
        measurement_mask=torch.ones(q.BATCH_SIZE, 2, dtype=torch.bool),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0:rgbd",
        supported_state_fields=("position", "radius"),
        auxiliary={
            "world_position": values.clone(),
            "world_position_log_variance": torch.zeros_like(values),
            "world_radius": radius,
            "world_radius_log_variance": torch.full_like(radius, -11.5),
            "world_radius_valid_mask": torch.ones(q.BATCH_SIZE, 2, dtype=torch.bool),
            "surface_fit_radius_relative_error": torch.zeros(
                q.BATCH_SIZE,
                2,
                dtype=torch.float32,
            ),
            "surface_fit_condition_number": torch.ones(
                q.BATCH_SIZE,
                2,
                dtype=torch.float32,
            ),
            "prior_interval_collision_mask": torch.zeros(
                q.BATCH_SIZE,
                2,
                dtype=torch.bool,
            ),
        },
    )
    measured.validate()
    return measured, radius_source


def _fake_evidence(
    *,
    ordinal: int = 0,
    radius_truth: torch.Tensor | None = None,
    anchor_radius: torch.Tensor | None = None,
    non_radius_scene_sha256: str = "2" * 64,
) -> q.SceneEvidence:
    zeros_2 = torch.zeros(2, dtype=torch.float32)
    zeros_16_2 = torch.zeros(16, 2, dtype=torch.float32)
    zeros_2_3 = torch.zeros(2, 3, dtype=torch.float32)
    zeros_5_2_3 = torch.zeros(5, 2, 3, dtype=torch.float32)
    return q.SceneEvidence(
        split="development",
        ordinal=ordinal,
        scene_sha256=f"{ordinal + 1:064x}",
        non_radius_scene_sha256=non_radius_scene_sha256,
        primitive_index=ordinal // 32,
        pair_variant=(ordinal % 32) // 16,
        radius_role=(ordinal % 16) // 8,
        camera_stratum=ordinal % 8,
        twin_ordinal=ordinal ^ 8,
        pair_variant_twin_ordinal=ordinal ^ 16,
        provenance_sha256="0" * 64,
        radius_truth=(zeros_2.clone() if radius_truth is None else radius_truth.clone()),
        anchor_raw_radius=(zeros_2.clone() if anchor_radius is None else anchor_radius.clone()),
        anchor_deployed_radius=(
            zeros_2.clone() if anchor_radius is None else anchor_radius.clone()
        ),
        history_raw_radius=zeros_16_2.clone(),
        history_deployed_radius=zeros_16_2.clone(),
        radius_valid=torch.ones(16, 2, dtype=torch.bool),
        radius_in_bounds=torch.ones(16, 2, dtype=torch.bool),
        surface_fit_relative_residual=zeros_16_2.clone(),
        surface_fit_condition=zeros_16_2.clone(),
        current_position_truth=zeros_2_3.clone(),
        current_position_mean=zeros_2_3.clone(),
        current_velocity_truth=zeros_2_3.clone(),
        current_velocity_mean=zeros_2_3.clone(),
        future_position_truth=zeros_5_2_3.clone(),
        future_position_mean=zeros_5_2_3.clone(),
        future_velocity_truth=zeros_5_2_3.clone(),
        future_velocity_mean=zeros_5_2_3.clone(),
        object_ids=torch.zeros(16, 2, dtype=torch.int64),
        active=torch.ones(16, 2, dtype=torch.bool),
        rollout_active=torch.ones(5, 2, dtype=torch.bool),
        diagnostics=(),
    )


def _fake_episode(ordinal: int) -> q._PacketEpisode:
    zeros_position = torch.zeros(56, 2, 3, dtype=torch.float32)
    return q._PacketEpisode(
        split="development",
        ordinal=ordinal,
        scene_sha256=f"{ordinal + 1:064x}",
        non_radius_scene_sha256=f"{ordinal // 8 + 100:064x}",
        primitive_index=ordinal // 32,
        pair_variant=(ordinal % 32) // 16,
        radius_role=(ordinal % 16) // 8,
        camera_stratum=ordinal % 8,
        twin_ordinal=ordinal ^ 8,
        pair_variant_twin_ordinal=ordinal ^ 16,
        rgb=torch.zeros(56, 3, 2, 2, dtype=torch.float32),
        depth=torch.zeros(56, 1, 2, 2, dtype=torch.float32),
        timestamps=torch.arange(56, dtype=torch.float32) / 20.0,
        world_from_camera=torch.eye(4).expand(56, 4, 4).clone(),
        intrinsics=torch.eye(3).expand(56, 3, 3).clone(),
        position_truth=zeros_position.clone(),
        velocity_truth=zeros_position.clone(),
        radius_truth=torch.zeros(2, dtype=torch.float32),
        albedo_truth=torch.zeros(2, 3, dtype=torch.float32),
    )


def _fake_evidence_for_episode(
    episode: q._PacketEpisode,
    packets: tuple[ObservationPacket, ...],
) -> q.SceneEvidence:
    value = _fake_evidence(
        ordinal=episode.ordinal,
        radius_truth=episode.radius_truth,
        anchor_radius=episode.radius_truth,
        non_radius_scene_sha256=episode.non_radius_scene_sha256,
    )
    receipt = q._evaluator_provenance_receipt(
        episode,
        packets,
        evidence_truth_sha256=q._expected_scene_evidence_truth_digest(episode),
    )
    return q.SceneEvidence(
        **{
            **{name: getattr(value, name) for name in value.__dataclass_fields__},
            "scene_sha256": episode.scene_sha256,
            "provenance_sha256": q._provenance_receipt_sha256(receipt),
            "current_position_truth": episode.position_truth[q.ANCHOR_FRAME_INDEX].clone(),
            "current_velocity_truth": episode.velocity_truth[q.ANCHOR_FRAME_INDEX].clone(),
            "future_position_truth": episode.position_truth[list(q.TARGET_FRAME_INDICES)].clone(),
            "future_velocity_truth": episode.velocity_truth[list(q.TARGET_FRAME_INDICES)].clone(),
        }
    )


def _fake_registered_packet_batch(
    tmp_path: Path,
) -> tuple[
    q._AccessLedger,
    q._ManifestCapability,
    q._BatchCapability,
    tuple[q._PacketEpisode, ...],
    tuple[ObservationPacket, ...],
]:
    ledger = _fake_ledger(tmp_path)
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    batch = manifest.begin_batch((0, 1, 2, 3))
    episodes = tuple(_fake_episode(ordinal) for ordinal in batch.ordinals)
    for episode, token in zip(episodes, batch.tokens, strict=True):
        manifest.consume_ordinal(batch, token, ordinal=episode.ordinal)
        q._EPISODE_REGISTRY[id(episode)] = (
            episode,
            manifest,
            batch,
            token,
            episode.split,
            episode.ordinal,
            q._episode_digest(episode),
        )
    packets: list[ObservationPacket] = []
    for frame_index in q.HISTORY_FRAME_INDICES:
        packet = ObservationPacket(
            modality="rgbd",
            sensor_id="camera0:rgbd",
            timestamp=frame_index / 20.0,
            payload={
                "rgb": torch.zeros(4, 3, 2, 2),
                "depth": torch.zeros(4, 1, 2, 2),
            },
            calibration={
                "world_from_camera": torch.eye(4).expand(4, 4, 4).clone(),
                "intrinsics": torch.eye(3).expand(4, 3, 3).clone(),
            },
            frame_id="camera:camera0:rgbd",
            metadata={
                "image_size": (2, 2),
                "depth_semantics": "surface",
            },
        )
        q._register_packet(
            packet,
            episodes=episodes,
            frame_index=frame_index,
            provenance="nominal_independent_packet",
        )
        packets.append(packet)
    return ledger, manifest, batch, episodes, tuple(packets)


def _load_runner_module() -> object:
    path = q.REPOSITORY_ROOT / "scripts" / "run_rgbd_variable_radius_qualification.py"
    specification = importlib.util.spec_from_file_location("variable_radius_runner_test", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _install_fake_report_surface(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    certificate = {
        "certificate_sha256": "c" * 64,
        "family": {"splits": 4, "scenes_per_split": 64},
    }
    protocol = {
        "protocol_sha256": "d" * 64,
        "axes": {"second": 2, "first": 1},
    }
    monkeypatch.setattr(
        q,
        "_frozen_scene_certificate_binding",
        lambda: copy.deepcopy(certificate),
    )
    monkeypatch.setattr(q, "bridge_protocol", lambda: copy.deepcopy(protocol))
    publication_surface_sha256 = {
        name: f"{index + 3:064x}" for index, name in enumerate(q.PUBLICATION_SURFACE_PATHS)
    }
    return {
        "commit": "a" * 40,
        "dirty": False,
        "worktree_fingerprint": "1" * 64,
        "runtime_source_fingerprint": "2" * 64,
        "frozen_source_sha256": dict(q.FROZEN_SOURCE_SHA256),
        "resolved_config_sha256": q.FROZEN_CONFIG_SHA256,
        "scene_certificate": copy.deepcopy(certificate),
        "publication_git": {
            "commit": "a" * 40,
            "tree_oid": "b" * 40,
            "object_format": q._GIT_OBJECT_FORMAT,
            "upstream_ref": "origin/frozen-variable-radius",
            "upstream_commit": "a" * 40,
            "ahead": 0,
            "behind": 0,
        },
        "publication_surface_sha256": publication_surface_sha256,
        "publication_surface_blobs": {
            name: {
                "path": relative,
                "mode": "100644",
                "blob_oid": f"{index + 6:040x}",
                "blob_sha256": publication_surface_sha256[name],
                "worktree_sha256": publication_surface_sha256[name],
            }
            for index, (name, relative) in enumerate(q.PUBLICATION_SURFACE_PATHS.items())
        },
    }


def _fake_split_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    split: str,
    passed: bool,
) -> dict[str, object]:
    metrics = {name: 0.0 for name in q.GATE_METRIC_SCHEMA}
    metrics["scene_count"] = 64.0 if passed else -1.0
    monkeypatch.setattr(
        q,
        "gate_failures",
        lambda values: [] if values["scene_count"] == 64.0 else ["synthetic:failed"],
    )
    return q._split_result(
        split=split,
        metrics=metrics,
        model_state_sha256=q.EMPTY_MODEL_STATE_SHA256,
        evidence_sha256="e" * 64,
        provenance_sha256="d" * 64,
    )


def _fake_checkpoint_record() -> dict[str, object]:
    return {
        "path": str(q.canonical_checkpoint_path().relative_to(q.REPOSITORY_ROOT)),
        "sha256": "f" * 64,
        "bytes": 1024.0,
        "model_state_sha256": q.EMPTY_MODEL_STATE_SHA256,
        "model_state_entry_count": 0,
        "optimizer_state_entry_count": 0,
        "rng_state_entry_count": 0,
    }


def _complete_fake_split(
    ledger: q._AccessLedger,
    *,
    split: str,
    passed: bool,
) -> dict[str, object]:
    manifest = q._ManifestCapability(split=split, ledger=ledger)
    for start in range(0, q.SCENES_PER_SPLIT, q.BATCH_SIZE):
        ordinals = (start, start + 1, start + 2, start + 3)
        batch = manifest.begin_batch(ordinals)
        for ordinal, token in zip(batch.ordinals, batch.tokens, strict=True):
            manifest.consume_ordinal(batch, token, ordinal=ordinal)
        manifest.complete_batch(batch, result_sha256=f"{start // 4 + 20:064x}")
    result = {"split": split, "passed": passed}
    manifest.close(result)
    return result


def test_v2_identity_is_distinct_and_binds_the_terminal_v1_disclosure() -> None:
    assert q.ARCHITECTURE_VERSION == 2
    assert q.ARCHITECTURE_ATTEMPT == 2
    assert q.MAX_ARCHITECTURE_ATTEMPTS == 2
    assert Path("runs/rgbd_two_visible_variable_radius_v2") == q.RUN_RELATIVE_PATH
    assert Path("runs/rgbd_two_visible_variable_radius_v1") != q.RUN_RELATIVE_PATH
    assert q.DEVELOPMENT_REPORT_NAME == "development_report_v2.json"
    assert q.CHECKPOINT_NAME == "development_model_v2.pt"
    assert q.DEVELOPMENT_LEDGER_NAME == "development_attempt_2_access.json"
    assert q.QUALIFICATION_REPORT_NAME == "qualification_report_v2.json"
    assert q.QUALIFICATION_LEDGER_NAME == "qualification_attempt_2_access.json"
    runner = _load_runner_module()
    assert runner._OUTER_RECEIPT_SCHEMA == "rgbd_variable_radius_outer_preflight_v2"

    protocol = q.bridge_protocol()
    assert protocol["name"] == "rgbd_two_visible_variable_radius_v2"
    assert protocol["terminal_after_attempt"] is True
    assert (
        protocol["evaluator_provenance"]["receipt_schema"]
        == "variable_radius_evaluator_provenance_receipt_v3"
    )
    disclosure = protocol["prior_architecture_attempt"]
    assert disclosure == {
        "schema": "rgbd_variable_radius_prior_attempt_disclosure_v1",
        "architecture_version": 1,
        "architecture_attempt": 1,
        "protocol_name": "rgbd_two_visible_variable_radius_v1",
        "commit": "db669b099f4e51c18e24645ddee8c1249f86b175",
        "development_report": {
            "sha256": "7f194a41bd5e64328f0a57d8142aad8a81f01d2b449386bb05939fb3ed49b142",
            "bytes": 66758,
        },
        "development_ledger": {
            "sha256": "aec6c9500d3cd8ca6a152b8107578b2b441a544dca605fe7f6ae59a61f0d021e",
            "bytes": 10248,
        },
        "terminal_status": "terminal_error",
        "error": {
            "type": "RuntimeError",
            "message": "element 0 of tensors does not require grad and does not have a grad_fn",
        },
        "active_split": "development",
        "active_batch": [0, 1, 2, 3],
        "checkpoint_published": False,
        "protected_access_started": False,
        "protected_splits_opened": [],
        "retry_permitted": False,
    }


def test_bridge_protocol_is_recursively_json_native_before_self_hashing() -> None:
    protocol = q.bridge_protocol()

    def assert_native(value: object) -> None:
        if type(value) is dict:
            assert all(type(key) is str for key in value)
            for item in value.values():
                assert_native(item)
        elif type(value) is list:
            for item in value:
                assert_native(item)
        else:
            assert value is None or type(value) in {bool, int, float, str}

    assert type(q.DEFAULT_GATES.horizon_position_rmse_m) is tuple
    assert type(protocol["gates"]["horizon_position_rmse_m"]) is list
    assert_native(protocol)
    supplied = protocol.pop("protocol_sha256")
    assert supplied == q.canonical_sha256(protocol)


def test_exact_metric_ownership_is_closed_and_collision_free() -> None:
    owned = [name for _, names in q._GATE_SCHEMA_OWNERS for name in names]
    assert len(owned) == len(set(owned))
    assert tuple(sorted(owned)) == q.GATE_METRIC_SCHEMA
    assert len(q.REDUCER_REGISTRY) == len({item.output for item in q.REDUCER_REGISTRY})
    assert len(q.REDUCER_REGISTRY) == len(
        {(item.source, item.reduction) for item in q.REDUCER_REGISTRY}
    )


def test_gate_failures_enforces_exact_schema_before_values() -> None:
    metrics = {name: 0.0 for name in q.GATE_METRIC_SCHEMA}
    failures = q.gate_failures(metrics)
    assert not any(item.startswith("metric_schema:") for item in failures)

    missing = dict(metrics)
    missing.pop(next(iter(missing)))
    assert any(item.startswith("metric_schema:") for item in q.gate_failures(missing))

    extra = {**metrics, "not_owned": 0.0}
    assert any(item.startswith("metric_schema:") for item in q.gate_failures(extra))


def test_pair_variant_counterfactuals_are_reduced_and_gated() -> None:
    rows: list[q.SceneEvidence] = []
    for ordinal in q.ORDINALS:
        primitive = ordinal // 32
        pair_variant = (ordinal % 32) // 16
        role = (ordinal % 16) // 8
        camera = ordinal % 8
        low = 0.200 + primitive * 0.001 + pair_variant * 0.005
        high = low + 0.020
        radius = torch.tensor(
            (low, high) if role == 0 else (high, low),
            dtype=torch.float32,
        )
        rows.append(
            _fake_evidence(
                ordinal=ordinal,
                radius_truth=radius,
                anchor_radius=radius,
                non_radius_scene_sha256=f"{primitive * 8 + camera + 500:064x}",
            )
        )
    metrics = q._counterfactual_metrics(rows)
    assert metrics["counterfactual_pair_count"] == 32.0
    assert metrics["pair_variant_counterfactual_pair_count"] == 32.0
    assert metrics["pair_variant_non_radius_certificate_mismatch_count"] == 0.0
    assert metrics["pair_variant_unordered_radius_pair_match_count"] == 0.0
    assert metrics["pair_variant_unintended_truth_position_max_abs_m"] == 0.0
    assert metrics["pair_variant_unintended_truth_velocity_max_abs_mps"] == 0.0
    assert metrics["pair_variant_estimated_anchor_position_rmse_m"] == 0.0
    assert metrics["pair_variant_estimated_anchor_position_max_abs_m"] == 0.0
    assert metrics["pair_variant_estimated_anchor_velocity_rmse_mps"] == 0.0
    assert metrics["pair_variant_estimated_anchor_velocity_max_abs_mps"] == 0.0
    assert metrics["pair_variant_paired_radius_delta_rmse_m"] == 0.0
    assert metrics["pair_variant_paired_radius_delta_max_abs_error_m"] == 0.0
    assert metrics["pair_variant_paired_radius_delta_sign_fraction"] == 1.0
    assert set(metrics).issubset(q.GATE_METRIC_SCHEMA)


def test_public_designed_radius_vjp_and_pair_variant_thresholds_are_exact() -> None:
    gates = q.DEFAULT_GATES
    assert gates.minimum_rgb_radius_gradient_l1 == 1.0e-14
    assert gates.minimum_depth_intrinsics_radius_gradient_l1 == 1.0e-8
    assert gates.anchor_radius_rmse_m == 5.0e-4
    assert gates.history_radius_rmse_m == 5.0e-4
    assert gates.grouped_radius_rmse_m == 5.0e-4
    assert gates.pair_variant_estimated_anchor_position_rmse_m == 1.0e-3
    assert gates.pair_variant_estimated_anchor_position_max_abs_m == 1.0e-3
    assert gates.pair_variant_estimated_anchor_velocity_rmse_mps == 1.0e-3
    assert gates.pair_variant_estimated_anchor_velocity_max_abs_mps == 1.0e-3


def test_pair_variant_estimated_anchor_differences_are_toleranced_not_exact_zero() -> None:
    rows: list[q.SceneEvidence] = []
    for ordinal in q.ORDINALS:
        primitive = ordinal // 32
        pair_variant = (ordinal % 32) // 16
        role = (ordinal % 16) // 8
        camera = ordinal % 8
        low = 0.200 + primitive * 0.001 + pair_variant * 0.005
        high = low + 0.020
        radius = torch.tensor(
            (low, high) if role == 0 else (high, low),
            dtype=torch.float32,
        )
        row = _fake_evidence(
            ordinal=ordinal,
            radius_truth=radius,
            anchor_radius=radius,
            non_radius_scene_sha256=f"{primitive * 8 + camera + 700:064x}",
        )
        if pair_variant == 1:
            row.current_position_mean.add_(5.0e-4)
            row.current_velocity_mean.add_(4.0e-4)
        rows.append(row)
    metrics = q._counterfactual_metrics(rows)
    assert metrics["pair_variant_unintended_truth_position_max_abs_m"] == 0.0
    assert metrics["pair_variant_unintended_truth_velocity_max_abs_mps"] == 0.0
    assert 0.0 < metrics["pair_variant_estimated_anchor_position_rmse_m"] <= 1.0e-3
    assert 0.0 < metrics["pair_variant_estimated_anchor_position_max_abs_m"] <= 1.0e-3
    assert 0.0 < metrics["pair_variant_estimated_anchor_velocity_rmse_mps"] <= 1.0e-3
    assert 0.0 < metrics["pair_variant_estimated_anchor_velocity_max_abs_mps"] <= 1.0e-3


def test_public_rollout_alias_audit_measures_storage_and_object_aliases() -> None:
    position = torch.zeros(1, 2, 3)
    velocity = torch.ones(1, 2, 3)
    active = torch.ones(1, 2, dtype=torch.bool)
    timestamp = torch.zeros(1)
    belief = types.SimpleNamespace(
        objects=types.SimpleNamespace(
            position=position,
            velocity=velocity,
            active=active,
        ),
        timestamp=timestamp,
    )
    aliased = types.SimpleNamespace(
        positions=position.unsqueeze(1),
        velocities=velocity.unsqueeze(1),
        active_mask=active,
        timestamps=timestamp,
    )
    detached = types.SimpleNamespace(
        positions=position.unsqueeze(1).clone(),
        velocities=velocity.unsqueeze(1).clone(),
        active_mask=active.clone(),
        timestamps=timestamp.clone(),
    )
    assert q._rollout_output_alias_count(aliased, belief) == 4
    assert q._rollout_output_alias_count(detached, belief) == 0


def test_operation_count_metrics_are_derived_from_measured_diagnostics() -> None:
    diagnostics = [
        {"public_predict_calls": 1.0, "model_reset_count": 1.0},
        {"public_predict_calls": 2.0, "model_reset_count": 3.0},
    ]
    assert q._measured_operation_count_metrics(diagnostics) == {
        "public_predict_calls_per_batch_min": 1.0,
        "public_predict_calls_per_batch_max": 2.0,
        "model_reset_count_per_batch_min": 1.0,
        "model_reset_count_per_batch_max": 3.0,
    }
    with pytest.raises(ValueError, match="finite exact floats"):
        q._measured_operation_count_metrics([{"public_predict_calls": 1, "model_reset_count": 1.0}])


def test_nominal_source_records_calls_and_aliases_instead_of_literal_metrics() -> None:
    nominal = inspect.getsource(q._evaluate_nominal_batch)
    evidence_sources = inspect.getsource(q._evidence_sources)
    aggregate = inspect.getsource(q._aggregate_split_metrics)
    assert 'operation_counts["predict"] += 1' in nominal
    assert 'operation_counts["reset"] += 1' in nominal
    assert 'operation_counts["correct"] += 1' in nominal
    assert "model.reset(batch_size=BATCH_SIZE)" in nominal
    assert "original_correct = model.updater.correct" in nominal
    assert "model.updater.correct = recording_correct" in nominal
    assert "_record_live_measurement(live_measurement_captures, measured)" in nominal
    assert "_validated_live_measurement_capture(" in nominal
    assert "raw_proposal_history.append(public_raw_radius" in nominal
    assert "public_raw_radius[batch_index, :, 0]" in nominal
    assert "raw_anchor_vjp = torch.stack(frame_raw_vjp)" in nominal
    assert "live_raw_radius[batch_index, :, 0]" in nominal
    assert "model.last_measurements" in nominal
    assert "model._last_measurements" not in nominal
    assert ".initialise(" not in nominal
    assert ".encode(" not in nominal
    assert ".initialise_measurements(" not in nominal
    assert ".encode_measurements(" not in nominal
    assert "model.observation_modules" not in nominal
    assert nominal.count("model.ingest(packet)") == 1
    assert nominal.count("original_correct(**kwargs)") == 1
    assert "_rollout_output_alias_count(trajectory, belief_after_rollout)" in nominal
    assert "public_rollout_output_alias_count" in evidence_sources
    assert "diagnostic(" in evidence_sources
    assert "_measured_operation_count_metrics(diagnostics)" in aggregate
    assert '"public_predict_calls_per_batch_min": 1.0' not in aggregate
    assert '"model_reset_count_per_batch_min": 1.0' not in aggregate


def test_live_measurement_capture_is_identity_preserving_once_and_bit_exact() -> None:
    live, radius_source = _fake_live_measurement()
    captures: list[q._LiveMeasurementCapture] = []
    assert q._record_live_measurement(captures, live) is live
    assert len(captures) == 1
    assert captures[0].measurement is live
    assert captures[0].measurement_identity == id(live)
    assert captures[0].call_index == 0
    public = live.detach()
    assert (
        q._validated_live_measurement_capture(
            captures[0],
            public,
            expected_call_index=0,
        )
        is live
    )
    assert "prior_interval_collision_mask" in live.auxiliary
    assert live.auxiliary["world_radius"].requires_grad is True
    assert live.auxiliary["world_radius"].grad_fn is not None
    assert radius_source.requires_grad is True
    assert q._tensor_tree_has_autograd(public) is False

    with pytest.raises(PermissionError, match="call index differs"):
        q._validated_live_measurement_capture(
            captures[0],
            public,
            expected_call_index=1,
        )
    with pytest.raises(PermissionError, match="identity or call index differs"):
        q._validated_live_measurement_capture(
            captures[0],
            live,
            expected_call_index=0,
        )

    bit_changed = live.detach()
    bit_changed.values = bit_changed.values.clone()
    bit_changed.values[0, 0, 0] = -0.0
    assert torch.equal(bit_changed.values, public.values)
    with pytest.raises(RuntimeError, match="bit-exactly"):
        q._validated_live_measurement_capture(
            captures[0],
            bit_changed,
            expected_call_index=0,
        )


def test_vjp_anchor_preflight_requires_live_graph_connected_exact_shape() -> None:
    source = torch.linspace(0.1, 0.8, 8, dtype=torch.float32, requires_grad=True)
    raw = (source.square() + 0.2).reshape(q.BATCH_SIZE, 2)
    deployed = (source * 0.5 + 0.1).reshape(q.BATCH_SIZE, 2)
    targets = q._validated_vjp_anchor_targets(
        raw_anchor_vjp=raw,
        deployed_anchor_vjp=deployed,
    )
    assert targets["raw"] is raw
    assert targets["deployed"] is deployed

    with pytest.raises(RuntimeError, match="raw VJP anchor lost"):
        q._validated_vjp_anchor_targets(
            raw_anchor_vjp=raw.detach(),
            deployed_anchor_vjp=deployed,
        )
    leaf = torch.ones(q.BATCH_SIZE, 2, dtype=torch.float32, requires_grad=True)
    assert leaf.grad_fn is None
    with pytest.raises(RuntimeError, match="deployed VJP anchor lost"):
        q._validated_vjp_anchor_targets(
            raw_anchor_vjp=raw,
            deployed_anchor_vjp=leaf,
        )
    with pytest.raises(RuntimeError, match=r"exact \[4,2\]"):
        q._validated_vjp_anchor_targets(
            raw_anchor_vjp=raw[:, :1],
            deployed_anchor_vjp=deployed,
        )


def test_vjp_ordinals_cover_all_axes_with_exact_unique_receipts() -> None:
    assert q.VJP_AUDIT_ORDINAL_GROUPS == (
        (0, 12, 17, 29),
        (34, 46, 51, 63),
    )
    assert len(q.VJP_AUDIT_ORDINALS) == 8
    assert {ordinal // 32 for ordinal in q.VJP_AUDIT_ORDINALS} == {0, 1}
    assert {(ordinal % 32) // 16 for ordinal in q.VJP_AUDIT_ORDINALS} == {0, 1}
    assert {(ordinal % 16) // 8 for ordinal in q.VJP_AUDIT_ORDINALS} == {0, 1}
    assert {ordinal % 8 for ordinal in q.VJP_AUDIT_ORDINALS} == set(range(8))

    parts: list[dict[str, float]] = []
    records: list[tuple[int, str, int, int, int, int]] = []
    for index, ordinal in enumerate(q.VJP_AUDIT_ORDINALS):
        part = {name: 0.0 for name in q._vjp_metric_names()}
        for name in part:
            if name.startswith(
                (
                    "gradient_l1/",
                    "gradient_anchor_history_frame_l1/",
                    "gradient_supported_history_frames/",
                )
            ):
                part[name] = 1.0
        part["gradient_vector_count"] = 12.0
        part["gradient_audit_scene_count"] = 1.0
        part["gradient_audit_unique_scene_fraction"] = 1.0
        parts.append(part)
        records.append(
            (
                ordinal,
                f"{index + 900:064x}",
                ordinal // 32,
                (ordinal % 32) // 16,
                (ordinal % 16) // 8,
                ordinal % 8,
            )
        )
    merged = q._merge_vjp_metrics(parts, records)
    assert merged["gradient_audit_scene_count"] == 8.0
    assert merged["gradient_audit_unique_scene_fraction"] == 1.0
    assert merged["gradient_audit_primitive_coverage_fraction"] == 1.0
    assert merged["gradient_audit_pair_variant_coverage_fraction"] == 1.0
    assert merged["gradient_audit_radius_role_coverage_fraction"] == 1.0
    assert merged["gradient_audit_camera_stratum_coverage_fraction"] == 1.0

    duplicate_hashes = list(records)
    duplicate_hashes[-1] = (
        duplicate_hashes[-1][0],
        duplicate_hashes[0][1],
        *duplicate_hashes[-1][2:],
    )
    with pytest.raises(RuntimeError, match="not exactly unique"):
        q._merge_vjp_metrics(parts, duplicate_hashes)


def test_manifest_vjp_selection_is_not_first_batch_only() -> None:
    source = inspect.getsource(q._collect_manifest_once)
    assert "ordinal in VJP_AUDIT_ORDINALS" in source
    assert "audit_vjp=start == 0" not in source
    assert "vjp_audit_indices=vjp_audit_indices" in source


def test_uncertainty_distribution_metrics_are_absent() -> None:
    joined = "\n".join(q.GATE_METRIC_SCHEMA).lower()
    assert "gaussian_nll" not in joined
    assert "marginal_coverage" not in joined
    assert "joint_coverage" not in joined
    assert "rms_z" not in joined
    assert "interval_width" not in joined


def test_manifest_is_exact_ordinal_only_rows() -> None:
    selector_field = "se" + "ed"
    for split in q.SPLITS:
        rows = q._manifest_rows(split)
        assert len(rows) == 64
        assert rows == tuple({"split": split, "ordinal": ordinal} for ordinal in range(64))
        assert all(selector_field not in row for row in rows)
        assert q.canonical_sha256(list(rows)) == q.MANIFEST_SHA256[split]


@pytest.mark.parametrize("value", [False, 0.0, "0", None])
def test_manifest_rejects_nonexact_ordinal_types(value: object) -> None:
    rows = list(q._manifest_rows("development"))
    rows[0] = {"split": "development", "ordinal": value}
    with pytest.raises(ValueError):
        q._validate_manifest_rows("development", rows)


def test_capability_rejects_cross_split_before_formal_access(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    batch = manifest.begin_batch((0, 1, 2, 3))
    with pytest.raises(PermissionError, match="address differs"):
        q._materialise_authorized_episode(
            split="selector",
            ordinal=0,
            manifest=manifest,
            batch=batch,
            token=batch.tokens[0],
        )
    assert q._CAPABILITY_REGISTRY[id(batch.tokens[0])][3] == "issued"
    ledger.fail(error_type="FixtureStop", error_message="no formal access", report_sha256="f" * 64)


def test_capability_rejects_replay_without_formal_access(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    batch = manifest.begin_batch((0, 1, 2, 3))
    manifest.consume_ordinal(batch, batch.tokens[0], ordinal=0)
    with pytest.raises(PermissionError, match="forged or replayed"):
        manifest.consume_ordinal(batch, batch.tokens[0], ordinal=0)
    ledger.fail(error_type="FixtureStop", error_message="no formal access", report_sha256="f" * 64)


def test_batch_registry_validates_every_sibling_before_consumption(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    batch = manifest.begin_batch((0, 1, 2, 3))
    before = _fake_ledger_bytes(ledger, label="batch sibling registry fixture")
    missing = q._CAPABILITY_REGISTRY.pop(id(batch.tokens[1]))
    with pytest.raises(PermissionError, match="batch capability registry binding differs"):
        manifest.consume_ordinal(batch, batch.tokens[0], ordinal=0)
    assert q._CAPABILITY_REGISTRY[id(batch.tokens[0])][3] == "issued"
    assert q._BATCH_REGISTRY[id(batch)].consumed_ordinals == frozenset()
    assert _fake_ledger_bytes(ledger, label="batch sibling registry fixture") == before
    q._CAPABILITY_REGISTRY[id(batch.tokens[1])] = missing
    ledger.fail(
        error_type="FixtureStop",
        error_message="batch sibling test complete",
        report_sha256="f" * 64,
    )


def test_batch_hash_failure_precedes_durable_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _fake_ledger(tmp_path)
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    before = _fake_ledger_bytes(ledger, label="pre-reservation hash fixture")

    def reject_hash(_: object) -> str:
        raise RuntimeError("synthetic pre-reservation hash failure")

    monkeypatch.setattr(q, "_batch_capability_sha256", reject_hash)
    with pytest.raises(RuntimeError, match="pre-reservation hash failure"):
        manifest.begin_batch((0, 1, 2, 3))
    assert _fake_ledger_bytes(ledger, label="pre-reservation hash fixture") == before
    assert ledger.record["splits"]["development"]["active_batch"] is None
    assert not q._BATCH_REGISTRY
    ledger.fail(
        error_type="FixtureStop",
        error_message="hash failure test complete",
        report_sha256="f" * 64,
    )


def test_direct_ledger_batch_commit_rejects_handwritten_registry(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    batch = manifest.begin_batch((0, 1, 2, 3))
    live = q._BATCH_REGISTRY[id(batch)]
    forged_live = q.replace(live, consumed_ordinals=frozenset(batch.ordinals))
    commit = q._BatchCommit(
        split="development",
        ordinals=batch.ordinals,
        result_sha256="3" * 64,
        nonce=object(),
    )
    q._BATCH_COMMIT_REGISTRY[id(commit)] = q._BatchCommitRegistration(
        commit=commit,
        ledger=ledger,
        manifest=manifest,
        batch=batch,
        batch_registration=forged_live,
        result_sha256=commit.result_sha256,
        owner_thread=threading.get_ident(),
        ledger_generation=ledger._record["generation"],
        ledger_record_sha256=q.sha256_bytes(ledger._last_bytes),
        status="issued",
    )
    before = _fake_ledger_bytes(ledger, label="direct batch commit fixture")
    with pytest.raises(PermissionError, match="one-shot owner commit"):
        ledger.complete_batch(
            "development",
            batch.ordinals,
            result_sha256=commit.result_sha256,
            commit=commit,
        )
    assert _fake_ledger_bytes(ledger, label="direct batch commit fixture") == before
    q._BATCH_COMMIT_REGISTRY.pop(id(commit), None)
    ledger.fail(
        error_type="FixtureStop",
        error_message="direct commit test complete",
        report_sha256="f" * 64,
    )


def test_fake_ledger_cannot_authorize_formal_scene_access(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    batch = manifest.begin_batch((0, 1, 2, 3))
    with pytest.raises(PermissionError, match="runner-minted authorization"):
        q._materialise_authorized_episode(
            split="development",
            ordinal=0,
            manifest=manifest,
            batch=batch,
            token=batch.tokens[0],
        )
    assert q._CAPABILITY_REGISTRY[id(batch.tokens[0])][3] == "issued"
    ledger.fail(
        error_type="FixtureStop",
        error_message="authorization test complete",
        report_sha256="f" * 64,
    )


def test_ledger_is_owned_by_creation_thread(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            ledger._verify_disk()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], PermissionError)
    assert "another thread" in str(errors[0])
    ledger.fail(
        error_type="FixtureStop", error_message="thread test complete", report_sha256="f" * 64
    )


def test_ledger_terminal_transition_cannot_repeat(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    ledger.fail(error_type="FixtureStop", error_message="terminal", report_sha256="f" * 64)
    with pytest.raises(PermissionError, match="already terminal"):
        ledger.fail(error_type="FixtureStop", error_message="duplicate", report_sha256="f" * 64)


def test_fake_development_ledger_completes_exact_generation(tmp_path: Path) -> None:
    bindings = {"fixture": "static"}
    directory_pin = _fake_directory_pin(tmp_path)
    ledger = q._AccessLedger(
        tmp_path / "development.json",
        stage="development",
        bindings=bindings,
        directory_pin=directory_pin,
    )
    manifest = q._ManifestCapability(split="development", ledger=ledger)
    for start in range(0, 64, 4):
        ordinals = (start, start + 1, start + 2, start + 3)
        batch = manifest.begin_batch(ordinals)
        for ordinal, token in zip(batch.ordinals, batch.tokens, strict=True):
            manifest.consume_ordinal(batch, token, ordinal=ordinal)
        manifest.complete_batch(batch, result_sha256=f"{start // 4:064x}")
    result = {"split": "development", "passed": True}
    manifest.close(result)
    terminal_sha256 = ledger.finish()
    contents = q._pinned_stable_read_bytes(
        directory_pin,
        ledger.path,
        label="fake development ledger",
    )
    assert terminal_sha256 == q.sha256_bytes(contents)
    record = q._validate_terminal_ledger(
        contents,
        stage="development",
        bindings=bindings,
        expected_results=[result],
    )
    assert record["generation"] == 35
    with pytest.raises(PermissionError, match="already terminal"):
        ledger.finish()


def test_gate_failed_qualification_ledger_roundtrips_exact_prefix(tmp_path: Path) -> None:
    bindings = {"nested": {"second": 2, "first": 1}, "fixture": "qualification"}
    directory_pin = _fake_directory_pin(tmp_path)
    ledger = q._AccessLedger(
        tmp_path / "qualification.json",
        stage="qualification",
        bindings=bindings,
        directory_pin=directory_pin,
    )
    results = [
        _complete_fake_split(ledger, split="selector", passed=True),
        _complete_fake_split(ledger, split="confirmation", passed=False),
    ]
    terminal_sha256 = ledger.finish()
    contents = q._pinned_stable_read_bytes(
        directory_pin,
        ledger.path,
        label="gate-failed qualification ledger",
    )
    assert terminal_sha256 == q.sha256_bytes(contents)
    record = q._validate_terminal_ledger(
        contents,
        stage="qualification",
        bindings={"fixture": "qualification", "nested": {"first": 1, "second": 2}},
        expected_outcome="gate_failed",
        expected_opened_splits=["selector", "confirmation"],
        expected_results=results,
    )
    assert record["generation"] == 69
    assert record["splits"]["final_test"]["status"] == "unopened"

    parsed = q._strict_json_loads(contents, label="gate-failed qualification ledger")
    parsed["splits"]["confirmation"]["next_ordinal"] = 64.0
    parsed_without_hash = {key: value for key, value in parsed.items() if key != "record_sha256"}
    parsed["record_sha256"] = q.canonical_sha256(parsed_without_hash)
    with pytest.raises(ValueError, match="state differs"):
        q._validate_terminal_ledger(
            q._report_bytes(parsed),
            stage="qualification",
            bindings=bindings,
            expected_outcome="gate_failed",
            expected_opened_splits=["selector", "confirmation"],
            expected_results=results,
        )


def test_full_qualification_ledger_roundtrips_sorted_split_map(tmp_path: Path) -> None:
    bindings = {"fixture": "full qualification", "nested": {"z": 3, "a": 1}}
    directory_pin = _fake_directory_pin(tmp_path)
    ledger = q._AccessLedger(
        tmp_path / "full-qualification.json",
        stage="qualification",
        bindings=bindings,
        directory_pin=directory_pin,
    )
    results = [
        _complete_fake_split(ledger, split=split, passed=True)
        for split in ("selector", "confirmation", "final_test")
    ]
    ledger.finish()
    contents = q._pinned_stable_read_bytes(
        directory_pin,
        ledger.path,
        label="full qualification ledger",
    )
    parsed = q._strict_json_loads(contents, label="full qualification ledger")
    assert list(parsed["splits"]) == ["confirmation", "final_test", "selector"]
    record = q._validate_terminal_ledger(
        contents,
        stage="qualification",
        bindings={"nested": {"a": 1, "z": 3}, "fixture": "full qualification"},
        expected_results=results,
    )
    assert record["generation"] == 103


def test_qualification_ledger_rejects_out_of_order_split(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path, stage="qualification")
    with pytest.raises(RuntimeError, match="exact ledger transition"):
        ledger.begin_split("confirmation")
    ledger.fail(
        error_type="FixtureStop", error_message="order test complete", report_sha256="f" * 64
    )


def test_ledger_detects_external_byte_replacement(tmp_path: Path) -> None:
    ledger = _fake_ledger(tmp_path)
    ledger.path.write_bytes(b"{}\n")
    with pytest.raises(PermissionError):
        ledger._verify_disk()
    q._LEDGER_REGISTRY.pop(id(ledger), None)


def test_ledger_record_is_deeply_immutable_and_internal_mutation_is_no_write(
    tmp_path: Path,
) -> None:
    ledger = _fake_ledger(tmp_path)
    before = _fake_ledger_bytes(ledger, label="ledger mutation fixture")
    with pytest.raises(TypeError):
        ledger.record["splits"]["development"]["status"] = "forged"
    assert _fake_ledger_bytes(ledger, label="ledger mutation fixture") == before

    ledger._record["splits"]["development"]["status"] = "forged"
    with pytest.raises(PermissionError, match="in-memory registry"):
        ledger.begin_split("development")
    assert _fake_ledger_bytes(ledger, label="ledger mutation fixture") == before
    q._LEDGER_REGISTRY.pop(id(ledger), None)


def test_direct_runner_seal_and_authorization_mints_reject_non_cli_callers(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="frozen CLI main boundary"):
        q._mint_runner_invocation_seal(
            stage="development",
            config=None,
            config_path=tmp_path / "config.yaml",
            report_path=tmp_path / "report.json",
            checkpoint_path=tmp_path / "checkpoint.pt",
            development_report_path=tmp_path / "report.json",
            source_provenance={},
            reviewed_development=None,
        )

    seal = q._RunnerInvocationSeal(
        stage="development",
        context_sha256="a" * 64,
        nonce=object(),
    )
    q._RUNNER_INVOCATION_REGISTRY[id(seal)] = (
        seal,
        "development",
        seal.context_sha256,
        threading.get_ident(),
        "consumed",
    )
    context = {
        "stage": "development",
        "paths": {"ledger": str(tmp_path / "development.json")},
        "ledger_bindings_sha256": q.canonical_sha256({"fixture": "expected"}),
        "runner_invocation_seal_sha256": seal.context_sha256,
    }
    with pytest.raises(PermissionError, match="frozen CLI main boundary"):
        q._mint_run_authorization(invocation_seal=seal, context=context)


def test_handwritten_authorization_registry_cannot_create_owned_ledger(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "authorized.json"
    directory_pin = _fake_directory_pin(tmp_path)
    bindings = {"fixture": "expected"}
    token = q._RunAuthorization(
        stage="development",
        ledger_path=str(ledger_path),
        ledger_bindings_sha256=q.canonical_sha256(bindings),
        context_sha256="b" * 64,
        run_directory_capability_sha256=q._pinned_directory_capability_sha256(directory_pin),
        nonce=object(),
    )
    with pytest.raises(PermissionError, match="vault claim"):
        q._AccessLedger(
            ledger_path,
            stage="development",
            bindings=bindings,
            directory_pin=directory_pin,
            authorization=token,
        )
    assert not ledger_path.exists()


def test_fake_cli_seal_registry_is_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = {"stage": "development", "preflight_complete": True}
    directory_pin = _fake_directory_pin(tmp_path)
    monkeypatch.setattr(q, "_require_frozen_cli_caller", lambda **_: None)
    monkeypatch.setattr(q, "_acquire_pinned_directory", lambda *_, **__: directory_pin)
    monkeypatch.setattr(q, "_runner_invocation_context", lambda **_: dict(context))
    arguments = {
        "stage": "development",
        "config": None,
        "config_path": Path("config"),
        "report_path": Path("report"),
        "checkpoint_path": Path("checkpoint"),
        "development_report_path": Path("report"),
        "source_provenance": {},
        "reviewed_development": None,
    }
    seal = q._mint_runner_invocation_seal(**arguments)
    with pytest.raises(PermissionError, match="cannot mint twice"):
        q._mint_runner_invocation_seal(**arguments)
    q._consume_runner_invocation_seal(seal, context=context)
    with pytest.raises(PermissionError, match="forged, replayed, or rebound"):
        q._consume_runner_invocation_seal(seal, context=context)
    q._release_runner_invocation_seal(seal)


@pytest.mark.parametrize("stage", ["development", "qualification"])
def test_fake_authorization_context_requires_exact_seal_owned_directory_pin(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q, "REPOSITORY_ROOT", tmp_path)
    runs_parent = tmp_path / "runs"
    runs_parent.mkdir()
    run_directory = runs_parent / q.RUN_RELATIVE_PATH.name
    run_directory.mkdir()
    directory_pin = q._acquire_pinned_directory(
        run_directory,
        create=False,
        canonical=True,
    )
    wrong_directory = tmp_path / "wrong-run"
    wrong_directory.mkdir()
    wrong_pin = q._acquire_pinned_directory(
        wrong_directory,
        create=False,
        canonical=False,
    )
    bindings = {"fixture": stage}
    source = {"fixture": "source"}
    reviewed = (
        None
        if stage == "development"
        else {
            "checkpoint_sha256": "1" * 64,
            "report_sha256": "2" * 64,
            "ledger_sha256": "3" * 64,
        }
    )

    class FakeConfig:
        def to_dict(self) -> dict[str, str]:
            return {"fixture": "config"}

    def fake_preflight(**arguments: object) -> dict[str, object]:
        pin = arguments["directory_pin"]
        assert isinstance(pin, q._PinnedDirectory)
        return {
            "stage": stage,
            "run_directory_capability_sha256": q._pinned_directory_capability_sha256(pin),
            "ledger_bindings_sha256": q.canonical_sha256(bindings),
        }

    monkeypatch.setattr(q, "assert_rgbd_variable_radius_config", lambda _: None)
    monkeypatch.setattr(q, "_validated_published_source", lambda value, **_: dict(value))
    monkeypatch.setattr(q, "_runner_invocation_context", fake_preflight)
    preflight = fake_preflight(directory_pin=directory_pin)
    seal = q._RunnerInvocationSeal(
        stage=stage,
        context_sha256=q.canonical_sha256(preflight),
        nonce=object(),
    )
    exact_registration = (
        seal,
        stage,
        seal.context_sha256,
        threading.get_ident(),
        "consumed",
        directory_pin,
    )
    q._RUNNER_INVOCATION_REGISTRY[id(seal)] = exact_registration
    report_path = (
        q.canonical_development_report_path()
        if stage == "development"
        else q.canonical_qualification_report_path()
    )
    arguments = {
        "stage": stage,
        "config": FakeConfig(),
        "config_path": q._frozen_config_path(),
        "report_path": report_path,
        "checkpoint_path": q.canonical_checkpoint_path(),
        "development_report_path": q.canonical_development_report_path(),
        "ledger_path": (
            q.development_ledger_path() if stage == "development" else q.qualification_ledger_path()
        ),
        "ledger_bindings": bindings,
        "source_provenance": source,
        "reviewed_development": reviewed,
        "invocation_seal": seal,
    }
    context = q._run_authorization_context(**arguments)
    assert context["run_directory_capability_sha256"] == (
        q._pinned_directory_capability_sha256(directory_pin)
    )

    q._RUNNER_INVOCATION_REGISTRY[id(seal)] = exact_registration[:-1]
    with pytest.raises(PermissionError, match="consumed CLI seal"):
        q._run_authorization_context(**arguments)

    q._RUNNER_INVOCATION_REGISTRY[id(seal)] = (*exact_registration[:-1], wrong_pin)
    with pytest.raises((PermissionError, ValueError), match="pinned.*directory"):
        q._run_authorization_context(**arguments)

    q._RUNNER_INVOCATION_REGISTRY[id(seal)] = exact_registration
    moved = runs_parent / "moved-run"
    run_directory.rename(moved)
    run_directory.mkdir()
    with pytest.raises(PermissionError, match="namespace binding changed"):
        q._run_authorization_context(**arguments)


def test_fake_evidence_mutation_is_detected(tmp_path: Path) -> None:
    ledger, _, _, episodes, packets = _fake_registered_packet_batch(tmp_path)
    evidence = _fake_evidence_for_episode(episodes[0], packets)
    q._register_scene_evidence(evidence, episode=episodes[0], packets=packets)
    assert (
        q._validated_evidence(
            evidence,
            split="development",
            ordinal=0,
        )
        is evidence
    )
    evidence.radius_truth[0] = 1.0
    with pytest.raises(PermissionError, match="receipt differs"):
        q._validated_evidence(evidence, split="development", ordinal=0)
    ledger.fail(
        error_type="FixtureStop", error_message="mutation test complete", report_sha256="f" * 64
    )


def test_episode_requires_batch_registry_consumed_ordinal(tmp_path: Path) -> None:
    ledger, _, batch, episodes, _ = _fake_registered_packet_batch(tmp_path)
    live = q._BATCH_REGISTRY[id(batch)]
    q._BATCH_REGISTRY[id(batch)] = q.replace(
        live,
        consumed_ordinals=live.consumed_ordinals - frozenset({episodes[0].ordinal}),
    )
    with pytest.raises(PermissionError, match="batch capability registry binding differs"):
        q._validate_episode_registration(episodes[0])
    q._BATCH_REGISTRY[id(batch)] = live
    ledger.fail(
        error_type="FixtureStop",
        error_message="consumed-set test complete",
        report_sha256="f" * 64,
    )


def test_evidence_registration_rejects_wrong_episode_and_incomplete_packets(
    tmp_path: Path,
) -> None:
    ledger, _, _, episodes, packets = _fake_registered_packet_batch(tmp_path)
    evidence = _fake_evidence_for_episode(episodes[0], packets)
    with pytest.raises(PermissionError, match="exact episode truth"):
        q._register_scene_evidence(evidence, episode=episodes[1], packets=packets)
    with pytest.raises(PermissionError, match="exactly 16"):
        q._register_scene_evidence(evidence, episode=episodes[0], packets=packets[:-1])
    ledger.fail(
        error_type="FixtureStop",
        error_message="binding rejection complete",
        report_sha256="f" * 64,
    )


def test_batch_finalization_is_exact_ordered_and_validate_before_retire(
    tmp_path: Path,
) -> None:
    ledger, _, _, episodes, packets = _fake_registered_packet_batch(tmp_path)
    rows = tuple(_fake_evidence_for_episode(episode, packets) for episode in episodes)
    for row, episode in zip(rows, episodes, strict=True):
        q._register_scene_evidence(row, episode=episode, packets=packets)

    with pytest.raises(PermissionError, match="frame/order"):
        q._finalize_batch_provenance(episodes, rows, packets[::-1])
    assert all(id(packet) in q._PACKET_REGISTRY for packet in packets)
    assert all(id(episode) in q._EPISODE_REGISTRY for episode in episodes)
    assert all(
        isinstance(q._EVIDENCE_REGISTRY[id(row)], q._LiveEvidenceRegistration) for row in rows
    )

    q._finalize_batch_provenance(episodes, rows, packets)
    assert all(id(packet) not in q._PACKET_REGISTRY for packet in packets)
    assert all(id(episode) not in q._EPISODE_REGISTRY for episode in episodes)
    assert all(
        isinstance(q._EVIDENCE_REGISTRY[id(row)], q._FinalEvidenceRegistration) for row in rows
    )
    for row in rows:
        registration = q._EVIDENCE_REGISTRY[id(row)]
        assert isinstance(registration, q._FinalEvidenceRegistration)
        receipt = registration.provenance_receipt
        assert receipt.split == row.split
        assert receipt.ordinal == row.ordinal
        assert receipt.episode_identity == registration.episode_id
        assert receipt.token_identity == registration.token_id
        assert receipt.batch_identity == registration.batch_id
        assert receipt.consumed_ordinals == (0, 1, 2, 3)
        assert len(receipt.packet_receipts) == q.HISTORY_FRAME_COUNT
        assert receipt.run_directory_binding_sha256 == ledger._directory_binding_sha256
        assert receipt.run_directory_identity == ledger._directory_pin.directory_identity
        assert row.provenance_sha256 == q._provenance_receipt_sha256(receipt)
        assert q._validated_evidence(row, split=row.split, ordinal=row.ordinal) is row
    assert q._evidence_provenance_sha256(rows) != q._evidence_provenance_sha256(rows[::-1])
    forged_row = q.replace(rows[0], provenance_sha256="a" * 64)
    assert q._evidence_sha256((forged_row,)) != q._evidence_sha256((rows[0],))

    first = q._EVIDENCE_REGISTRY[id(rows[0])]
    assert isinstance(first, q._FinalEvidenceRegistration)
    forged_receipt = q.replace(
        first.provenance_receipt,
        packet_receipts=(
            (*first.provenance_receipt.packet_receipts[0][:2], "f" * 64),
            *first.provenance_receipt.packet_receipts[1:],
        ),
    )
    q._EVIDENCE_REGISTRY[id(rows[0])] = q.replace(
        first,
        provenance_receipt=forged_receipt,
    )
    with pytest.raises(PermissionError, match="finalized receipt differs"):
        q._validated_evidence(rows[0], split="development", ordinal=0)
    q._EVIDENCE_REGISTRY[id(rows[0])] = first
    forged_directory_receipt = q.replace(
        first.provenance_receipt,
        run_directory_binding_sha256="f" * 64,
    )
    q._EVIDENCE_REGISTRY[id(rows[0])] = q.replace(
        first,
        provenance_receipt=forged_directory_receipt,
    )
    with pytest.raises(PermissionError, match="finalized receipt differs"):
        q._validated_evidence(rows[0], split="development", ordinal=0)
    q._EVIDENCE_REGISTRY[id(rows[0])] = first
    ledger.fail(
        error_type="FixtureStop",
        error_message="finalization test complete",
        report_sha256="f" * 64,
    )


def test_packet_digest_rejects_governed_metadata() -> None:
    safe = ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=0.0,
        payload={
            "rgb": torch.zeros(4, 3, 2, 2),
            "depth": torch.zeros(4, 1, 2, 2),
        },
        calibration={
            "world_from_camera": torch.eye(4).expand(4, 4, 4).clone(),
            "intrinsics": torch.eye(3).expand(4, 3, 3).clone(),
        },
        frame_id="camera:camera0:rgbd",
        metadata={
            "image_size": (2, 2),
            "depth_semantics": "surface",
        },
    )
    assert len(q._packet_digest(safe)) == 64
    leaked = ObservationPacket(
        **{
            **safe.__dict__,
            "metadata": {
                **safe.metadata,
                "ordinal": 0,
            },
        }
    )
    with pytest.raises(ValueError, match="governed or truth"):
        q._packet_digest(leaked)


def test_packet_materializer_source_has_no_model_visible_address_labels() -> None:
    source = inspect.getsource(q._packet_for_frame)
    assert '"formal_split"' not in source
    assert '"formal_ordinals"' not in source
    assert '"ordinal"' not in source
    assert '"radius_role"' not in source
    assert '"pair_variant"' not in source
    assert "radius_truth" not in source
    assert "position_truth" not in source
    assert set(item.strip().strip('"') for item in ("image_size", "depth_semantics")) == {
        "image_size",
        "depth_semantics",
    }


def test_source_never_imports_public_scene_execution_helpers() -> None:
    source_path = (
        q.REPOSITORY_ROOT / "world_model" / "training" / "rgbd_variable_radius_qualification.py"
    )
    source = source_path.read_text(encoding="utf-8")
    forbidden = (
        "scene_family_certificate",
        "render_spheres",
        "SphereWorld",
        "collate_episodes",
        "make_rgbd_packet",
        "world_model.simulator",
        "initial_sphere_state",
    )
    for token in forbidden:
        assert token not in source


def test_only_literal_certificate_binding_is_used_by_protocol_source() -> None:
    source = inspect.getsource(q.bridge_protocol)
    assert "_frozen_scene_certificate_binding()" in source
    assert "certificate_descriptor" not in source
    assert "scene_family_certificate" not in source


def test_literal_certificate_binding_matches_final_reviewed_scene_freeze() -> None:
    binding = q._frozen_scene_certificate_binding()
    assert q.FROZEN_SOURCE_SHA256["scene"] == (
        "d9d4f7b9cbb22de2d2cb07db4c9e2b77aa4d57798f996c5c445fa5202f256525"
    )
    assert q.FROZEN_SOURCE_SHA256["scene_test"] == (
        "c84d98b9a68acca49d47e2fa9e4d1a3631515deb5f2197f52318d83186aaa10c"
    )
    assert binding["certificate_sha256"] == (
        "473137981e0a6443834c806f9f8792e2fee6a556961e5d977d3c6ae69cc7f0d5"
    )
    assert binding["descriptor_sha256"] == q.FROZEN_SCENE_DESCRIPTOR_SHA256
    assert binding["descriptor_sha256"] == (
        "5145cc7c9f09c4189afe7ddd4da147d93c63e64683f2eaff347b488090a55532"
    )
    assert binding["input_binding_sha256"] == q.FROZEN_SCENE_INPUT_BINDING_SHA256
    assert binding["input_binding_sha256"] == (
        "4863b2cb58dccc5d2338f27f4afe8b12456a424266a5c89f3ef2da9b1a1cc51d"
    )
    assert binding["trace_sha256"] == q.FROZEN_TRACE_SHA256
    assert binding["split_trace_sha256"] == q.FROZEN_SPLIT_TRACE_SHA256
    assert "conic_geometry" in binding["trace_sha256"]
    assert "conic_geometry" in binding["split_trace_sha256"]


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        q._strict_json_loads(b'{"a":1,"a":2}', label="fake")
    with pytest.raises(ValueError, match="nonfinite"):
        q._strict_json_loads(b'{"a":NaN}', label="fake")


def test_exact_equality_ignores_mapping_order_but_rejects_sequence_and_type_coercion() -> None:
    q._exact_equal(
        {"second": {"y": 2, "x": 1}, "first": True},
        {"first": True, "second": {"x": 1, "y": 2}},
        label="reordered mapping",
    )
    with pytest.raises(ValueError):
        q._exact_equal([2, 1], [1, 2], label="reordered list")
    with pytest.raises(ValueError, match="has type"):
        q._exact_equal({"value": True}, {"value": 1}, label="bool is not int")
    with pytest.raises(ValueError, match="has type"):
        q._exact_equal({"value": 1.0}, {"value": 1}, label="float is not int")


def test_full_development_report_roundtrips_sorted_multikey_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    result = _fake_split_result(monkeypatch, split="development", passed=True)
    report = q._report_root(
        stage="development",
        source_provenance=source,
        results=[result],
        terminal_ledger_sha256="9" * 64,
    )
    report["checkpoint"] = _fake_checkpoint_record()
    encoded = q._report_bytes(report)
    parsed = q._strict_json_loads(encoded, label="fake development report")
    assert list(parsed["source_provenance"]["publication_surface_sha256"]) == sorted(
        q.PUBLICATION_SURFACE_PATHS
    )
    validated = q._validate_report(
        parsed,
        stage="development",
        expected_source={
            **source,
            "publication_surface_sha256": {
                name: source["publication_surface_sha256"][name]
                for name in reversed(tuple(q.PUBLICATION_SURFACE_PATHS))
            },
        },
    )
    q._validate_development_report_extras(validated)
    assert validated["outcome"] == "passed"


def test_real_pinned_report_write_read_validate_roundtrip_normalizes_tuple_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_protocol = q.bridge_protocol()
    assert type(q.DEFAULT_GATES.horizon_position_rmse_m) is tuple
    assert type(real_protocol["gates"]["horizon_position_rmse_m"]) is list
    source = _install_fake_report_surface(monkeypatch)
    monkeypatch.setattr(q, "bridge_protocol", lambda: copy.deepcopy(real_protocol))
    result = _fake_split_result(monkeypatch, split="development", passed=True)
    report = q._report_root(
        stage="development",
        source_provenance=source,
        results=[result],
        terminal_ledger_sha256="9" * 64,
    )
    report["checkpoint"] = _fake_checkpoint_record()
    pin = _fake_directory_pin(tmp_path)
    path = tmp_path / "development-v2-roundtrip.json"
    q._write_report_fresh(pin, path, report)
    encoded = q._pinned_stable_read_bytes(pin, path, label="v2 report roundtrip")
    parsed = q._strict_json_loads(encoded, label="v2 report roundtrip")
    validated = q._validate_report(
        parsed,
        stage="development",
        expected_source=source,
    )
    q._validate_development_report_extras(validated)
    q._exact_equal(validated, report, label="v2 report JSON roundtrip")
    assert type(validated["protocol"]["gates"]["horizon_position_rmse_m"]) is list


def test_partial_qualification_report_is_strictly_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    selector = _fake_split_result(monkeypatch, split="selector", passed=True)
    confirmation = _fake_split_result(monkeypatch, split="confirmation", passed=False)
    report = q._report_root(
        stage="qualification",
        source_provenance=source,
        results=[selector, confirmation],
        terminal_ledger_sha256="8" * 64,
    )
    report["reviewed_development"] = {
        "checkpoint_sha256": "5" * 64,
        "report_sha256": "6" * 64,
        "ledger_sha256": "7" * 64,
    }
    validated = q._validate_report(
        q._strict_json_loads(q._report_bytes(report), label="partial qualification report"),
        stage="qualification",
        expected_source=source,
    )
    q._validate_qualification_report_extras(validated)
    assert validated["opened_splits"] == ["selector", "confirmation"]
    assert validated["stopped_after"] == "confirmation"
    assert validated["outcome"] == "gate_failed"

    forged = copy.deepcopy(validated)
    forged["opened_splits"].append("final_test")
    with pytest.raises(ValueError, match="opened/result split binding differs"):
        q._validate_report(forged, stage="qualification", expected_source=source)


def test_error_report_checkpoint_requires_completed_passed_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    error_report = {
        "artifact_kind": "rgbd_variable_radius_qualification_report",
        "stage": "development",
        "architecture_version": q.ARCHITECTURE_VERSION,
        "architecture_attempt": q.ARCHITECTURE_ATTEMPT,
        "protocol": q.bridge_protocol(),
        "resolved_config_sha256": q.FROZEN_CONFIG_SHA256,
        "scene_certificate": q._frozen_scene_certificate_binding(),
        "source_provenance": source,
        "manifest_sha256": {},
        "results": [],
        "model_state_sha256": q.EMPTY_MODEL_STATE_SHA256,
        "optimizer_updates": 0,
        "optimizer_state_entry_count": 0,
        "rng_state_entry_count": 0,
        "passed": False,
        "terminal_ledger_sha256": None,
        "opened_splits": [],
        "stopped_after": None,
        "materialization_started": False,
        "access_completed": False,
        "outcome": "error",
        "error": {"type": "SyntheticError", "message": "before access"},
        "checkpoint": _fake_checkpoint_record(),
    }
    validated = q._validate_report(
        error_report,
        stage="development",
        expected_source=source,
    )
    with pytest.raises(ValueError, match="unearned checkpoint"):
        q._validate_development_report_extras(validated)

    result = _fake_split_result(monkeypatch, split="development", passed=True)
    error_report["results"] = [result]
    error_report["manifest_sha256"] = {"development": q.MANIFEST_SHA256["development"]}
    error_report["opened_splits"] = ["development"]
    error_report["stopped_after"] = "development"
    error_report["materialization_started"] = True
    validated = q._validate_report(
        error_report,
        stage="development",
        expected_source=source,
    )
    q._validate_development_report_extras(validated)


def test_exception_report_replaces_preexisting_bytes_then_terminalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    ledger = _fake_ledger(tmp_path)
    directory_pin = ledger._directory_pin
    report_path = tmp_path / "error-report.json"
    q._pinned_durable_create(directory_pin, report_path, b"preexisting bytes\n")
    original_replace = q._pinned_durable_replace
    replacement_order: list[Path] = []

    def recording_replace(
        pin: q._PinnedDirectory,
        path: Path,
        contents: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        assert pin is directory_pin
        replacement_order.append(Path(path))
        original_replace(pin, path, contents, mode=mode)

    monkeypatch.setattr(q, "_pinned_durable_replace", recording_replace)
    error = RuntimeError("synthetic failure")
    digest = q._persist_exception_report(
        directory_pin=directory_pin,
        path=report_path,
        stage="development",
        source_provenance=source,
        ledger=ledger,
        error=error,
    )
    contents = q._pinned_stable_read_bytes(
        directory_pin,
        report_path,
        label="replaced exception report",
    )
    assert digest == q.sha256_bytes(contents)
    parsed = q._strict_json_loads(contents, label="replaced exception report")
    validated = q._validate_report(
        parsed,
        stage="development",
        expected_source=source,
    )
    assert validated["outcome"] == "error"
    assert ledger._terminal is True
    assert replacement_order == [ledger.path, report_path]
    q._validate_terminal_ledger(
        q._pinned_stable_read_bytes(
            directory_pin,
            ledger.path,
            label="exception terminal ledger",
        ),
        stage="development",
        bindings={"fixture": "static"},
        expected_outcome="error",
        expected_opened_splits=[],
        expected_error={"type": "RuntimeError", "message": "synthetic failure"},
        expected_report_sha256=digest,
        expected_results=[],
    )


@pytest.mark.parametrize("failure_point", ["writer", "reread"])
def test_exception_report_write_failures_still_terminalize_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    ledger = _fake_ledger(tmp_path)
    directory_pin = ledger._directory_pin
    report_path = tmp_path / f"{failure_point}-error-report.json"
    original_read = q._pinned_stable_read_bytes
    terminal_before_report: list[str] = []
    if failure_point == "writer":

        def reject_write(
            pin: q._PinnedDirectory,
            _: Path,
            __: bytes,
            *,
            mode: int = 0o600,
        ) -> None:
            del mode
            assert pin is directory_pin
            assert ledger._terminal is True
            terminal = q._strict_json_loads(
                original_read(
                    directory_pin,
                    ledger.path,
                    label="pre-report crash ledger",
                ),
                label="pre-report crash ledger",
            )
            terminal_before_report.append(terminal["error"]["report_sha256"])
            raise OSError("synthetic writer failure")

        monkeypatch.setattr(q, "_pinned_durable_create", reject_write)
    else:

        def reject_report_reread(
            pin: q._PinnedDirectory,
            path: Path,
            *,
            label: str,
        ) -> bytes:
            assert pin is directory_pin
            if Path(path) == report_path:
                assert ledger._terminal is True
                terminal = q._strict_json_loads(
                    original_read(
                        directory_pin,
                        ledger.path,
                        label="pre-reread crash ledger",
                    ),
                    label="pre-reread crash ledger",
                )
                terminal_before_report.append(terminal["error"]["report_sha256"])
                raise OSError("synthetic reread failure")
            return original_read(directory_pin, path, label=label)

        monkeypatch.setattr(q, "_pinned_stable_read_bytes", reject_report_reread)
    digest = q._persist_exception_report(
        directory_pin=directory_pin,
        path=report_path,
        stage="development",
        source_provenance=source,
        ledger=ledger,
        error=ValueError(failure_point),
    )
    assert digest is None
    assert ledger._terminal is True
    record = q._strict_json_loads(
        original_read(
            directory_pin,
            ledger.path,
            label="write-failure terminal ledger",
        ),
        label="write-failure terminal ledger",
    )
    assert record["status"] == "terminal_error"
    assert terminal_before_report == [record["error"]["report_sha256"]]
    assert len(record["error"]["report_sha256"]) == 64
    if failure_point == "writer":
        assert not report_path.exists()
    else:
        assert (
            q.sha256_bytes(
                original_read(
                    directory_pin,
                    report_path,
                    label="written error report",
                )
            )
            == record["error"]["report_sha256"]
        )


def test_exception_after_normal_terminal_does_not_publish_contradictory_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    ledger = _fake_ledger(tmp_path)
    directory_pin = ledger._directory_pin
    _complete_fake_split(ledger, split="development", passed=True)
    ledger.finish()
    report_path = tmp_path / "preserved-report.json"
    q._pinned_durable_create(
        directory_pin,
        report_path,
        b"preserve this terminal evidence\n",
    )
    before = q._pinned_stable_read_bytes(
        directory_pin,
        report_path,
        label="preserved terminal report",
    )
    digest = q._persist_exception_report(
        directory_pin=directory_pin,
        path=report_path,
        stage="development",
        source_provenance=source,
        ledger=ledger,
        error=RuntimeError("later exception"),
    )
    assert digest is None
    assert (
        q._pinned_stable_read_bytes(
            directory_pin,
            report_path,
            label="preserved terminal report",
        )
        == before
    )


def test_exception_report_is_not_published_without_one_live_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    directory_pin = _fake_directory_pin(tmp_path)
    report_path = tmp_path / "no-ledger-error-report.json"
    assert (
        q._persist_exception_report(
            directory_pin=directory_pin,
            path=report_path,
            stage="development",
            source_provenance=source,
            ledger=None,
            error=RuntimeError("before ledger reservation"),
        )
        is None
    )
    assert not report_path.exists()


def test_cli_main_code_guard_binds_filename_stacksize_and_linetable() -> None:
    runner_path = q.REPOSITORY_ROOT / q.PUBLICATION_SURFACE_PATHS["runner"]
    contents = q.stable_read_bytes(runner_path, label="runner fingerprint fixture")

    def main_code(module_code: types.CodeType) -> types.CodeType:
        return next(
            value
            for value in module_code.co_consts
            if isinstance(value, types.CodeType) and value.co_name == "main"
        )

    absolute = main_code(compile(contents, str(runner_path), "exec"))
    relative_name = q.PUBLICATION_SURFACE_PATHS["runner"]
    relative = main_code(compile(contents, relative_name, "exec"))
    assert absolute == relative
    assert absolute.co_filename != relative.co_filename
    normalized = main_code(compile(contents, relative.co_filename, "exec"))
    assert q._exact_code_object_equal(normalized, relative)
    larger_stack = relative.replace(co_stacksize=relative.co_stacksize + 1)
    changed_lines = relative.replace(co_linetable=relative.co_linetable + b"\x00")
    assert larger_stack == relative
    assert changed_lines == relative
    assert not q._exact_code_object_equal(larger_stack, relative)
    assert not q._exact_code_object_equal(changed_lines, relative)
    assert "caller.f_code.co_filename" in inspect.getsource(q._require_frozen_cli_caller)


def test_real_runner_main_stack_reaches_directory_acquisition_after_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    runner_path = q.REPOSITORY_ROOT / q.PUBLICATION_SURFACE_PATHS["runner"]
    fake_loader = types.SimpleNamespace(validate_loaded_modules=lambda: None)
    fake_torch = types.SimpleNamespace(set_num_threads=lambda _count: None)
    fake_qualification = types.SimpleNamespace(
        require_frozen_config=lambda _path: object(),
        canonical_checkpoint_path=q.canonical_checkpoint_path,
        canonical_development_report_path=q.canonical_development_report_path,
        _mint_runner_invocation_seal=q._mint_runner_invocation_seal,
    )

    class DirectoryAcquisitionReached(RuntimeError):
        pass

    def stop_before_directory_acquisition(*_args: object, **_kwargs: object) -> None:
        raise DirectoryAcquisitionReached

    monkeypatch.setattr(runner, "__name__", "__main__")
    monkeypatch.setattr(runner, "__file__", str(runner_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(runner_path), "--phase", "development", "--_internal-stage"],
    )
    monkeypatch.setattr(runner, "_consume_outer_receipt", lambda _args: {"publication": {}})
    monkeypatch.setattr(
        runner,
        "_load_frozen_qualification",
        lambda _publication: (fake_qualification, fake_loader),
    )
    monkeypatch.setattr(runner, "_validate_loaded_publication", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "importlib",
        types.SimpleNamespace(import_module=lambda _name: fake_torch),
    )
    monkeypatch.setattr(runner, "_release_project_loader_preserving_error", lambda _loader: None)
    monkeypatch.setattr(q, "_acquire_pinned_directory", stop_before_directory_acquisition)

    with pytest.raises(DirectoryAcquisitionReached):
        runner.main(["--phase", "development", "--_internal-stage"])
    assert not q._RUNNER_INVOCATION_REGISTRY
    assert not q._LEDGER_REGISTRY


def test_structurally_different_spoofed_runner_main_is_rejected_before_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path = q.REPOSITORY_ROOT / q.PUBLICATION_SURFACE_PATHS["runner"]
    namespace = {
        "__name__": "__main__",
        "__file__": str(runner_path),
        "qualification": q,
        "fixture_path": Path("/unreached/frozen-runner-guard"),
    }
    source = """
def main():
    return qualification._mint_runner_invocation_seal(
        stage="development",
        config=None,
        config_path=fixture_path,
        report_path=fixture_path,
        checkpoint_path=fixture_path,
        development_report_path=fixture_path,
        source_provenance={},
        reviewed_development=None,
    )
"""
    exec(compile(source, str(runner_path), "exec"), namespace, namespace)
    monkeypatch.setattr(sys, "argv", [str(runner_path), "--phase", "development"])
    monkeypatch.setattr(
        q,
        "_acquire_pinned_directory",
        lambda *_args, **_kwargs: pytest.fail("spoofed main reached directory acquisition"),
    )
    with pytest.raises(PermissionError, match="frozen CLI main boundary"):
        namespace["main"]()


def test_capture_source_rejects_noncanonical_root_before_git(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical repository root"):
        q.capture_published_source(tmp_path)


def test_publication_surface_binds_exact_three_new_files() -> None:
    assert q.PUBLICATION_SURFACE_PATHS == {
        "qualification": "world_model/training/rgbd_variable_radius_qualification.py",
        "runner": "scripts/run_rgbd_variable_radius_qualification.py",
        "qualification_test": "tests/unit/test_rgbd_variable_radius_qualification.py",
    }


def _git_fixture_run(
    repository: Path,
    arguments: list[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        env=environment,
        input=input_bytes,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _replacement_git_fixture(
    tmp_path: Path,
) -> tuple[Path, str, str, bytes, bytes]:
    repository = tmp_path / "git-object-fixture"
    repository.mkdir()
    _git_fixture_run(repository, ["init", "--object-format=sha1"])
    original = b"reviewed original blob\n"
    replacement = b"hostile replacement blob\n"
    (repository / "tracked.txt").write_bytes(original)
    _git_fixture_run(repository, ["add", "tracked.txt"])
    _git_fixture_run(
        repository,
        [
            "-c",
            "user.name=Qualification Test",
            "-c",
            "user.email=qualification@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "fixture",
        ],
    )
    commit = _git_fixture_run(repository, ["rev-parse", "HEAD"]).decode().strip()
    original_oid = (
        _git_fixture_run(
            repository,
            ["rev-parse", "HEAD:tracked.txt"],
        )
        .decode()
        .strip()
    )
    replacement_oid = (
        _git_fixture_run(
            repository,
            ["hash-object", "-w", "--stdin"],
            input_bytes=replacement,
        )
        .decode()
        .strip()
    )
    _git_fixture_run(repository, ["replace", original_oid, replacement_oid])
    assert _git_fixture_run(repository, ["cat-file", "blob", original_oid]) == replacement
    return repository.resolve(), commit, original_oid, original, replacement


def test_git_plumbing_disables_replacements_and_sanitizes_hostile_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, commit, original_oid, original, _ = _replacement_git_fixture(tmp_path)
    hostile_git_environment = {
        "GIT_DIR": "/hostile/git-dir",
        "GIT_COMMON_DIR": "/hostile/common-dir",
        "GIT_WORK_TREE": "/hostile/work-tree",
        "GIT_OBJECT_DIRECTORY": "/hostile/objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/hostile/alternate-objects",
        "GIT_INDEX_FILE": "/hostile/index",
        "GIT_NAMESPACE": "hostile-namespace",
        "GIT_REPLACE_REF_BASE": "refs/hostile-replacements",
        "GIT_NO_REPLACE_OBJECTS": "0",
        "GIT_CONFIG_GLOBAL": "/hostile/global-config",
        "GIT_CONFIG_SYSTEM": "/hostile/system-config",
        "GIT_CONFIG_NOSYSTEM": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.repositoryformatversion",
        "GIT_CONFIG_VALUE_0": "999",
        "GIT_CONFIG_PARAMETERS": "'core.repositoryformatversion'='999'",
        "GIT_EXEC_PATH": "/hostile/git-exec",
        "GIT_CEILING_DIRECTORIES": "/",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
    }
    for name, value in hostile_git_environment.items():
        monkeypatch.setenv(name, value)

    runner = _load_runner_module()
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    runner_environment = runner._git_environment()
    qualification_environment = q._git_environment(repository)
    safe_git_environment = {
        "GIT_DIR": str(repository / ".git"),
        "GIT_COMMON_DIR": str(repository / ".git"),
        "GIT_WORK_TREE": str(repository),
        "GIT_INDEX_FILE": str(repository / ".git/index"),
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for environment in (runner_environment, qualification_environment):
        assert {name: value for name, value in environment.items() if name.startswith("GIT_")} == (
            safe_git_environment
        )
    for command in (
        runner._git_command(["cat-file", "blob", original_oid]),
        q._git_command(repository, ["cat-file", "blob", original_oid]),
    ):
        assert command[1] == "--no-replace-objects"
        assert f"--git-dir={repository / '.git'}" in command
        assert f"--work-tree={repository}" in command

    assert (
        runner._git_verified_object(
            original_oid,
            object_type="blob",
            label="runner replacement fixture",
        )
        == original
    )
    assert (
        q._git_verified_object(
            repository,
            original_oid,
            object_type="blob",
            label="qualification replacement fixture",
        )
        == original
    )
    tree_oid = runner._git_commit_tree_oid(commit, label="runner fixture commit")
    runner._git_verified_object(tree_oid, object_type="tree", label="runner fixture tree")
    assert runner._git_tree_entry(
        commit=commit,
        relative="tracked.txt",
        label="runner fixture path",
    ) == ("100644", original_oid)


@pytest.mark.parametrize("surface", ["runner", "qualification"])
def test_git_object_verifier_rejects_forged_cat_file_bytes(
    surface: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"reviewed object bytes\n"
    forged = b"forged object bytes!\n"
    if surface == "runner":
        module = _load_runner_module()
        oid = module._git_object_oid("blob", expected)

        def fake_text(arguments: list[str], *, label: str, **_: object) -> str:
            del label
            return "blob" if arguments[1] == "-t" else str(len(forged))

        def fake_bytes(arguments: list[str], *, label: str, **_: object) -> bytes:
            del label
            assert arguments == ["cat-file", "blob", oid]
            return forged

        monkeypatch.setattr(module, "_git_text", fake_text)
        monkeypatch.setattr(module, "_git_bytes", fake_bytes)
        with pytest.raises(PermissionError, match="framed OID differs"):
            module._git_verified_object(oid, object_type="blob", label="forged runner blob")
    else:
        oid = q._git_object_oid("blob", expected)

        def fake_text(
            _root: Path,
            arguments: list[str],
            *,
            label: str,
            **_: object,
        ) -> str:
            del label
            return "blob" if arguments[1] == "-t" else str(len(forged))

        def fake_bytes(
            _root: Path,
            arguments: list[str],
            *,
            label: str,
            **_: object,
        ) -> bytes:
            del label
            assert arguments == ["cat-file", "blob", oid]
            return forged

        monkeypatch.setattr(q, "_git_text", fake_text)
        monkeypatch.setattr(q, "_git_bytes", fake_bytes)
        with pytest.raises(PermissionError, match="framed OID differs"):
            q._git_verified_object(
                q.REPOSITORY_ROOT,
                oid,
                object_type="blob",
                label="forged qualification blob",
            )


def test_publication_boundary_rejects_clean_worktree_not_equal_to_head_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    blob_bytes = {name: f"reviewed-{name}\n".encode() for name in q.PUBLICATION_SURFACE_PATHS}
    for name, contents in blob_bytes.items():
        digest = q.sha256_bytes(contents)
        source["publication_surface_sha256"][name] = digest
        source["publication_surface_blobs"][name]["blob_sha256"] = digest
        source["publication_surface_blobs"][name]["worktree_sha256"] = digest
    monkeypatch.setattr(
        q,
        "_capture_git_publication_state",
        lambda _, **__: copy.deepcopy(source["publication_git"]),
    )

    def blob_binding(
        _: Path,
        *,
        commit: str,
        relative: str,
        label: str,
        **_kwargs: object,
    ) -> tuple[dict[str, str], bytes]:
        del commit, label
        name = next(key for key, value in q.PUBLICATION_SURFACE_PATHS.items() if value == relative)
        return copy.deepcopy(source["publication_surface_blobs"][name]), blob_bytes[name]

    monkeypatch.setattr(q, "_git_blob_binding", blob_binding)
    tampered_name = "qualification"

    def clean_but_tampered(path: str | Path, *, label: str) -> bytes:
        del label
        name = next(
            key
            for key, relative in q.PUBLICATION_SURFACE_PATHS.items()
            if Path(path).as_posix().endswith(relative)
        )
        return b"tampered but status said clean\n" if name == tampered_name else blob_bytes[name]

    monkeypatch.setattr(q, "stable_read_bytes", clean_but_tampered)
    with pytest.raises(PermissionError, match="differs from exact HEAD blob"):
        q._validate_publication_surface(source)


def test_publication_boundary_rejects_worktree_swap_between_stable_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    blob_bytes = {name: f"reviewed-{name}\n".encode() for name in q.PUBLICATION_SURFACE_PATHS}
    for name, contents in blob_bytes.items():
        digest = q.sha256_bytes(contents)
        source["publication_surface_sha256"][name] = digest
        source["publication_surface_blobs"][name]["blob_sha256"] = digest
        source["publication_surface_blobs"][name]["worktree_sha256"] = digest
    monkeypatch.setattr(
        q,
        "_capture_git_publication_state",
        lambda _, **__: copy.deepcopy(source["publication_git"]),
    )

    def blob_binding(
        _: Path,
        *,
        commit: str,
        relative: str,
        label: str,
        **_kwargs: object,
    ) -> tuple[dict[str, str], bytes]:
        del commit, label
        name = next(key for key, value in q.PUBLICATION_SURFACE_PATHS.items() if value == relative)
        return copy.deepcopy(source["publication_surface_blobs"][name]), blob_bytes[name]

    monkeypatch.setattr(q, "_git_blob_binding", blob_binding)
    reads: dict[str, int] = {}

    def swapped_read(path: str | Path, *, label: str) -> bytes:
        del label
        name = next(
            key
            for key, relative in q.PUBLICATION_SURFACE_PATHS.items()
            if Path(path).as_posix().endswith(relative)
        )
        reads[name] = reads.get(name, 0) + 1
        if name == "runner" and reads[name] == 2:
            return b"boundary swap\n"
        return blob_bytes[name]

    monkeypatch.setattr(q, "stable_read_bytes", swapped_read)
    with pytest.raises(PermissionError, match="differs from exact HEAD blob"):
        q._validate_publication_surface(source)
    assert reads["runner"] == 2


def test_pinned_directory_rejects_canonical_name_swap_without_redirecting_write(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    run_directory = parent / "run"
    run_directory.mkdir()
    pin = q._acquire_pinned_directory(run_directory, create=False, canonical=False)
    moved = parent / "moved"
    run_directory.rename(moved)
    run_directory.mkdir()
    target = run_directory / "artifact.json"
    with pytest.raises(PermissionError, match="namespace binding changed"):
        q._pinned_durable_create(pin, target, b"must not redirect\n")
    assert list(run_directory.iterdir()) == []


def test_persistent_directory_binding_is_stable_across_artifact_creation_and_checks_mode(
    tmp_path: Path,
) -> None:
    pin = _fake_directory_pin(tmp_path)
    binding_before = q._pinned_directory_binding(pin)
    capability_before = q._pinned_directory_capability_sha256(pin)
    assert binding_before["schema"] == "rgbd_variable_radius_run_directory_v2"
    assert len(binding_before["parent_identity"]) == 3
    assert len(binding_before["directory_identity"]) == 3

    target = tmp_path / "development-v2-artifact.json"
    q._pinned_durable_create(pin, target, b"v2 evidence\n")
    binding_after = q._pinned_directory_binding(pin)
    capability_after = q._pinned_directory_capability_sha256(pin)
    q._exact_equal(binding_after, binding_before, label="stable v2 directory binding")
    assert capability_after == capability_before

    review_pin = q._acquire_pinned_directory(tmp_path, create=False, canonical=False)
    try:
        review_binding = q._pinned_directory_binding(review_pin)
        q._exact_equal(
            review_binding,
            binding_before,
            label="reacquired post-development directory binding",
        )
        assert q.canonical_sha256(review_binding) == q.canonical_sha256(binding_before)
    finally:
        q._release_pinned_directory(review_pin)

    original_permissions = stat.S_IMODE(os.stat(tmp_path).st_mode)
    tampered_permissions = 0o755 if original_permissions != 0o755 else 0o700
    os.chmod(tmp_path, tampered_permissions)
    try:
        with pytest.raises(PermissionError, match="namespace binding changed"):
            q._validate_pinned_directory(pin)
    finally:
        os.chmod(tmp_path, original_permissions)
    q._validate_pinned_directory(pin)


def test_pinned_replace_swap_operates_only_on_original_fd_and_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    run_directory = parent / "run"
    run_directory.mkdir()
    pin = q._acquire_pinned_directory(run_directory, create=False, canonical=False)
    target = run_directory / "artifact.json"
    q._pinned_durable_create(pin, target, b"old\n")
    moved = parent / "moved"
    original_replace = os.replace
    swapped = False

    def swapping_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal swapped
        assert src_dir_fd == pin.directory_fd
        assert dst_dir_fd == pin.directory_fd
        if not swapped:
            run_directory.rename(moved)
            run_directory.mkdir()
            swapped = True
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(q.os, "replace", swapping_replace)
    with pytest.raises(PermissionError, match="pinned namespace"):
        q._pinned_durable_replace(pin, target, b"new\n")
    assert list(run_directory.iterdir()) == []
    assert (moved / target.name).read_bytes() == b"new\n"


def test_pinned_open_swap_operates_only_on_original_fd_and_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    run_directory = parent / "run"
    run_directory.mkdir()
    pin = q._acquire_pinned_directory(run_directory, create=False, canonical=False)
    target = run_directory / "artifact.json"
    moved = parent / "moved"
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd == pin.directory_fd and path == target.name and not swapped:
            run_directory.rename(moved)
            run_directory.mkdir()
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(q.os, "open", swapping_open)
    with pytest.raises(PermissionError, match="pinned namespace"):
        q._pinned_durable_create(pin, target, b"old directory only\n")
    assert list(run_directory.iterdir()) == []
    assert (moved / target.name).read_bytes() == b"old directory only\n"


def test_pinned_directory_fsync_swap_rejects_without_redirecting_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    run_directory = parent / "run"
    run_directory.mkdir()
    pin = q._acquire_pinned_directory(run_directory, create=False, canonical=False)
    target = run_directory / "artifact.json"
    moved = parent / "moved"
    original_fsync = os.fsync
    swapped = False

    def swapping_fsync(descriptor: int) -> None:
        nonlocal swapped
        if descriptor == pin.directory_fd and not swapped:
            run_directory.rename(moved)
            run_directory.mkdir()
            swapped = True
        original_fsync(descriptor)

    monkeypatch.setattr(q.os, "fsync", swapping_fsync)
    with pytest.raises(PermissionError, match="namespace binding changed"):
        q._pinned_durable_create(pin, target, b"old directory only\n")
    assert list(run_directory.iterdir()) == []
    assert (moved / target.name).read_bytes() == b"old directory only\n"


def test_pinned_directory_rejects_closed_fd_number_reuse(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    run_directory = parent / "run"
    run_directory.mkdir()
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    pin = q._acquire_pinned_directory(run_directory, create=False, canonical=False)
    attacker_fd = os.open(
        attacker,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    os.close(pin.directory_fd)
    os.dup2(attacker_fd, pin.directory_fd)
    os.close(attacker_fd)
    with pytest.raises(PermissionError, match="namespace binding changed"):
        q._validate_pinned_directory(pin)


def test_pinned_artifact_syscalls_use_only_relative_names_and_exact_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = _fake_directory_pin(tmp_path)
    target = tmp_path / "artifact.json"
    original_open = os.open
    original_replace = os.replace
    artifact_opens: list[tuple[str, int]] = []
    replacements: list[tuple[str, str, int, int]] = []

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd == pin.directory_fd:
            assert type(path) is str
            artifact_opens.append((path, dir_fd))
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def recording_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        replacements.append((source, destination, src_dir_fd, dst_dir_fd))
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(q.os, "open", recording_open)
    monkeypatch.setattr(q.os, "replace", recording_replace)
    q._pinned_durable_create(pin, target, b"first\n")
    q._pinned_durable_replace(pin, target, b"second\n")
    assert artifact_opens
    assert all(path in {target.name, target.name + ".tmp"} for path, _ in artifact_opens)
    assert replacements == [(target.name + ".tmp", target.name, pin.directory_fd, pin.directory_fd)]


def test_tensor_digest_is_versioned_framed_shape_typed_and_little_endian() -> None:
    class Collector:
        def __init__(self) -> None:
            self.contents = bytearray()

        def update(self, value: bytes) -> None:
            self.contents.extend(value)

    collector = Collector()
    q._update_tensor_digest(collector, "int16", torch.tensor([0x0102], dtype=torch.int16))
    frames: list[bytes] = []
    offset = 0
    contents = bytes(collector.contents)
    while offset < len(contents):
        length = int.from_bytes(contents[offset : offset + 8], byteorder="big", signed=False)
        offset += 8
        frames.append(contents[offset : offset + length])
        offset += length
    assert offset == len(contents)
    assert frames[0] == b"rgbd_variable_radius_tensor_digest_v2"
    assert frames[1] == b"int16"
    assert b'"byte_order":"little"' in frames[2]
    assert b'"order":"C"' in frames[3]
    assert frames[4] == b"\x02\x01"

    same_bytes_vector = hashlib.sha256()
    same_bytes_matrix = hashlib.sha256()
    q._update_tensor_digest(
        same_bytes_vector,
        "shape",
        torch.tensor([1, 2], dtype=torch.int16),
    )
    q._update_tensor_digest(
        same_bytes_matrix,
        "shape",
        torch.tensor([[1, 2]], dtype=torch.int16),
    )
    assert same_bytes_vector.hexdigest() != same_bytes_matrix.hexdigest()

    noncontiguous = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
    left = hashlib.sha256()
    right = hashlib.sha256()
    q._update_tensor_digest(left, "logical", noncontiguous)
    q._update_tensor_digest(right, "logical", noncontiguous.contiguous())
    assert left.hexdigest() == right.hexdigest()


def test_evidence_digest_uses_framed_tensor_helper_and_outer_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _fake_evidence(radius_truth=torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert q._evidence_sha256((row,)) != q._scene_evidence_digest(row)

    class Collector:
        def __init__(self) -> None:
            self.contents = bytearray()

        def update(self, value: bytes) -> None:
            self.contents.extend(value)

        def hexdigest(self) -> str:
            return "0" * 64

    collector = Collector()
    monkeypatch.setattr(q.hashlib, "sha256", lambda: collector)
    assert q._evidence_sha256((row,)) == "0" * 64
    frames: list[bytes] = []
    contents = bytes(collector.contents)
    offset = 0
    while offset < len(contents):
        length = int.from_bytes(contents[offset : offset + 8], byteorder="big", signed=False)
        offset += 8
        frames.append(contents[offset : offset + length])
        offset += length
    assert offset == len(contents)
    assert len(frames) == 4 + 5 * len(q._EVIDENCE_TENSOR_SHAPES)
    assert frames[0] == b"rgbd_variable_radius_evidence_digest_v2"
    assert frames[1] == q._canonical_json({"row_count": 1, "row_order": "ordinal_sequence"})
    assert frames[2] == b"evidence_row_metadata_v2"
    assert b'"row_index":0' in frames[3]
    assert frames[4] == b"rgbd_variable_radius_tensor_digest_v2"
    assert frames[5] == b"row[0].radius_truth"
    assert b'"byte_order":"little"' in frames[6]
    assert b'"order":"C"' in frames[7]
    assert frames[8] == b"\x00\x00\x80?\x00\x00\x00@"


def test_evidence_axes_and_checkpoint_counts_reject_bool_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = q.replace(_fake_evidence(), primitive_index=True)
    with pytest.raises(TypeError, match="exact builtin scalar types"):
        q._validated_evidence(evidence, split="development", ordinal=0)

    source = _install_fake_report_surface(monkeypatch)
    result = _fake_split_result(monkeypatch, split="development", passed=True)
    report = q._report_root(
        stage="development",
        source_provenance=source,
        results=[result],
        terminal_ledger_sha256="9" * 64,
    )
    report["checkpoint"] = _fake_checkpoint_record()
    report["checkpoint"]["model_state_entry_count"] = False
    validated = q._validate_report(report, stage="development", expected_source=source)
    with pytest.raises(ValueError, match="checkpoint report evidence differs"):
        q._validate_development_report_extras(validated)


def test_frozen_core_surface_contains_full_reviewed_map() -> None:
    expected_names = {
        "checkpoint_roundtrip_test",
        "rgbd_two_object_bridge_test",
        "belief_invariants_test",
        "config_test",
        "filter_update_test",
        "rgbd_observation_test",
        "two_disc_geometry_test",
        "scene",
        "scene_test",
        "profile",
        "profile_test",
        "accepted_qualification_provenance",
        "two_disc_geometry",
        "rgbd_observation",
        "measurements",
        "filter_correction",
        "filter_uncertainty",
        "lifecycle",
        "online_world_model",
        "checkpointing",
        "config",
    }
    assert set(q.FROZEN_SOURCE_SHA256) == expected_names
    assert set(q.FROZEN_SOURCE_PATHS) == expected_names
    assert all(len(value) == 64 for value in q.FROZEN_SOURCE_SHA256.values())


def test_empty_checkpoint_schema_has_no_optimizer_or_rng_payload() -> None:
    assert "model_state" in q.CHECKPOINT_SCHEMA
    assert "optimizer_state" in q.CHECKPOINT_SCHEMA
    assert "scheduler_state" in q.CHECKPOINT_SCHEMA
    assert "rng" not in q.CHECKPOINT_SCHEMA
    assert q.canonical_sha256([]) == q.EMPTY_MODEL_STATE_SHA256


def test_restricted_checkpoint_loader_accepts_only_exact_empty_state_envelope() -> None:
    payload = {name: None for name in q.CHECKPOINT_SCHEMA}
    payload["model_state"] = {}
    payload["optimizer_state"] = None
    payload["scheduler_state"] = None
    stream = io.BytesIO()
    torch.save(payload, stream)
    loaded = q._load_checkpoint_payload(stream.getvalue())
    assert type(loaded) is dict
    assert loaded["model_state"] == {}

    payload["model_state"] = {"unexpected": torch.zeros(1)}
    stream = io.BytesIO()
    torch.save(payload, stream)
    with pytest.raises(ValueError, match="exact empty dict"):
        q._load_checkpoint_payload(stream.getvalue())


def test_restricted_checkpoint_loader_rejects_unsafe_pickle_global() -> None:
    stream = io.BytesIO()
    torch.save({"unsafe": _UnsafeCheckpointGlobal()}, stream)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        q._load_checkpoint_payload(stream.getvalue())


@pytest.mark.parametrize("forged_value", [64, True])
def test_checkpoint_nested_development_result_rejects_scalar_coercion(
    forged_value: int | bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _install_fake_report_surface(monkeypatch)
    result = _fake_split_result(monkeypatch, split="development", passed=True)
    monkeypatch.setattr(q, "assert_rgbd_variable_radius_config", lambda _: None)
    contents = q._checkpoint_bytes(
        config=object(),
        development_result=result,
        source_provenance=source,
    )
    payload = copy.deepcopy(dict(q._load_checkpoint_payload(contents)))
    payload["development_result"]["metrics"]["scene_count"] = forged_value
    stream = io.BytesIO()
    torch.save(payload, stream)
    with pytest.raises((TypeError, ValueError), match="metric|development result"):
        q._validate_checkpoint_payload(
            stream.getvalue(),
            expected_source=source,
        )


def test_runner_requires_exact_three_reviewed_hashes() -> None:
    runner = _load_runner_module()
    parsed = runner.arguments(["--phase", "development"])
    assert parsed.phase == "development"
    with pytest.raises(SystemExit):
        runner.arguments(["--phase", "qualification"])
    digest = "a" * 64
    parsed = runner.arguments(
        [
            "--phase",
            "qualification",
            "--reviewed-checkpoint-sha256",
            digest,
            "--reviewed-report-sha256",
            digest,
            "--reviewed-development-ledger-sha256",
            digest,
        ]
    )
    assert parsed.phase == "qualification"


def _fake_runner_publication(runner: object, *, digit: str) -> dict[str, object]:
    assert len(digit) == 1 and digit in "123456789abcdef"
    surface = {
        name: f"{index + int(digit, 16):064x}"
        for index, name in enumerate(runner.PUBLICATION_SURFACE_PATHS)
    }
    return {
        "repository_root": str(runner.REPOSITORY_ROOT),
        "publication_git": {
            "commit": digit * 40,
            "tree_oid": f"{int(digit, 16) + 1:x}" * 40,
            "object_format": runner._GIT_OBJECT_FORMAT,
            "upstream_ref": "origin/frozen-variable-radius",
            "upstream_commit": digit * 40,
            "ahead": 0,
            "behind": 0,
        },
        "publication_surface_sha256": surface,
        "publication_surface_blobs": {
            name: {
                "path": relative,
                "mode": "100644",
                "blob_oid": f"{index + int(digit, 16) + 2:040x}",
                "blob_sha256": surface[name],
                "worktree_sha256": surface[name],
            }
            for index, (name, relative) in enumerate(runner.PUBLICATION_SURFACE_PATHS.items())
        },
    }


def _install_fake_outer_receipt(
    monkeypatch: pytest.MonkeyPatch,
    runner: object,
    parsed: object,
    publication: dict[str, object],
) -> dict[str, object]:
    nonce = "a" * 64
    receipt = {
        "schema": runner._OUTER_RECEIPT_SCHEMA,
        "pid": os.getpid(),
        "nonce": nonce,
        "phase": parsed.phase,
        "public_argv": runner._public_argv(parsed),
        "reviewed": runner._reviewed_arguments(parsed),
        "publication": copy.deepcopy(publication),
    }
    contents = runner._canonical_json(receipt)
    read_fd, write_fd = os.pipe()
    try:
        offset = 0
        while offset < len(contents):
            offset += os.write(write_fd, contents[offset:])
    finally:
        os.close(write_fd)
    monkeypatch.setenv(runner._OUTER_RECEIPT_FD_ENV, str(read_fd))
    monkeypatch.setenv(runner._OUTER_RECEIPT_SHA256_ENV, runner._sha256(contents))
    monkeypatch.setenv(runner._OUTER_RECEIPT_NONCE_ENV, nonce)
    return receipt


def test_runner_outer_stage_has_no_eager_project_import_and_orders_receipt_first() -> None:
    runner = _load_runner_module()
    source = inspect.getsource(runner)
    main_source = inspect.getsource(runner.main)
    assert "\nfrom world_model" not in source
    assert "\nimport world_model" not in source
    assert not any(
        isinstance(value, types.ModuleType) and value.__name__.startswith("world_model")
        for value in vars(runner).values()
    )
    assert main_source.index("_consume_outer_receipt(args)") < main_source.index(
        "_load_frozen_qualification"
    )
    assert main_source.index("_load_frozen_qualification") < main_source.index(
        "_validate_loaded_publication"
    )
    assert main_source.index("_validate_loaded_publication") < main_source.index(
        "_mint_runner_invocation_seal"
    )


def test_fake_outer_receipt_is_one_shot_and_revalidated_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    parsed = runner.arguments(["--phase", "development", "--_internal-stage"])
    publication = _fake_runner_publication(runner, digit="1")
    expected = _install_fake_outer_receipt(monkeypatch, runner, parsed, publication)
    monkeypatch.setattr(
        runner,
        "_capture_outer_publication",
        lambda: copy.deepcopy(publication),
    )
    consumed = runner._consume_outer_receipt(parsed)
    assert consumed == expected
    with pytest.raises(PermissionError, match="complete outer receipt"):
        runner._consume_outer_receipt(parsed)


def test_fake_checkout_switch_after_outer_receipt_rejects_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    parsed = runner.arguments(["--phase", "development", "--_internal-stage"])
    before = _fake_runner_publication(runner, digit="1")
    after = _fake_runner_publication(runner, digit="3")
    _install_fake_outer_receipt(monkeypatch, runner, parsed, before)
    monkeypatch.setattr(runner, "_capture_outer_publication", lambda: copy.deepcopy(after))
    with pytest.raises(PermissionError, match="outer receipt publication"):
        runner._consume_outer_receipt(parsed)


def test_fake_outer_exec_bootstraps_exact_reviewed_runner_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    parsed = runner.arguments(["--phase", "development"])
    publication = _fake_runner_publication(runner, digit="1")
    runner_source = b"raise RuntimeError('reviewed runner fixture is never executed')\n"
    runner_digest = runner._sha256(runner_source)
    runner_binding = publication["publication_surface_blobs"]["runner"]
    runner_binding["blob_sha256"] = runner_digest
    runner_binding["worktree_sha256"] = runner_digest
    publication["publication_surface_sha256"]["runner"] = runner_digest
    receipt = {
        "schema": runner._OUTER_RECEIPT_SCHEMA,
        "pid": os.getpid(),
        "nonce": "a" * 64,
        "phase": "development",
        "public_argv": runner._public_argv(parsed),
        "reviewed": None,
        "publication": publication,
    }
    monkeypatch.setattr(
        runner,
        "_outer_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(receipt),
    )
    monkeypatch.setattr(
        runner,
        "_git_verified_object",
        lambda *_args, **_kwargs: runner_source,
    )
    captured: dict[str, object] = {}

    def fake_execve(executable: str, argv: list[str], environment: dict[str, str]) -> None:
        runner_fd = int(argv[4])
        receipt_fd = int(environment[runner._OUTER_RECEIPT_FD_ENV])
        runner_stat = os.fstat(runner_fd)
        receipt_stat = os.fstat(receipt_fd)
        assert stat.S_ISREG(runner_stat.st_mode)
        assert runner_stat.st_nlink == 0
        assert runner_stat.st_size == len(runner_source)
        assert os.pread(runner_fd, len(runner_source), 0) == runner_source
        assert stat.S_ISFIFO(receipt_stat.st_mode)
        assert os.get_inheritable(runner_fd)
        assert os.get_inheritable(receipt_fd)
        captured.update(executable=executable, argv=argv, environment=environment)
        captured.update(runner_fd=runner_fd, receipt_fd=receipt_fd)
        raise RuntimeError("stop before exec")

    monkeypatch.setattr(runner.os, "execve", fake_execve)
    with pytest.raises(RuntimeError, match="stop before exec"):
        runner._exec_internal(parsed)
    argv = captured["argv"]
    assert argv[1:4] == ["-I", "-c", runner._INTERNAL_BOOTSTRAP]
    assert argv[6] == str(runner.RUNNER_PATH)
    assert str(runner.RUNNER_PATH) not in argv[:6]
    compile(runner._INTERNAL_BOOTSTRAP, "<reviewed-runner-bootstrap>", "exec")
    for descriptor in (captured["runner_fd"], captured["receipt_fd"]):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    for name in (
        runner._OUTER_RECEIPT_FD_ENV,
        runner._OUTER_RECEIPT_SHA256_ENV,
        runner._OUTER_RECEIPT_NONCE_ENV,
    ):
        assert name not in os.environ


@pytest.mark.parametrize("failure_point", ["set_inheritable", "write"])
def test_fake_outer_pipe_failure_closes_both_descriptors(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    parsed = runner.arguments(["--phase", "development"])
    publication = _fake_runner_publication(runner, digit="1")
    receipt = {
        "schema": runner._OUTER_RECEIPT_SCHEMA,
        "pid": os.getpid(),
        "nonce": "a" * 64,
        "phase": "development",
        "public_argv": runner._public_argv(parsed),
        "reviewed": None,
        "publication": publication,
    }
    monkeypatch.setattr(runner, "_outer_receipt", lambda *_args, **_kwargs: receipt)
    descriptors: list[int] = []
    real_pipe = os.pipe
    real_set_inheritable = os.set_inheritable

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    def injected_set_inheritable(descriptor: int, inheritable: bool) -> None:
        if failure_point == "set_inheritable":
            raise OSError("injected set-inheritable failure")
        real_set_inheritable(descriptor, inheritable)

    def injected_write(_descriptor: int, _contents: bytes) -> int:
        raise OSError("injected write failure")

    monkeypatch.setattr(runner.os, "pipe", tracked_pipe)
    monkeypatch.setattr(runner.os, "set_inheritable", injected_set_inheritable)
    if failure_point == "write":
        monkeypatch.setattr(runner.os, "write", injected_write)
    with pytest.raises(OSError, match="injected"):
        runner._exec_internal(parsed)
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_fake_post_import_checkout_switch_rejects_loaded_publication() -> None:
    runner = _load_runner_module()
    receipt_publication = _fake_runner_publication(runner, digit="1")
    switched_publication = _fake_runner_publication(runner, digit="3")
    module = types.ModuleType(runner._QUALIFICATION_MODULE)
    module.capture_published_source = lambda _: {
        "commit": switched_publication["publication_git"]["commit"],
        "dirty": False,
        "publication_git": switched_publication["publication_git"],
        "publication_surface_sha256": switched_publication["publication_surface_sha256"],
        "publication_surface_blobs": switched_publication["publication_surface_blobs"],
    }
    with pytest.raises(PermissionError, match="post-import publication"):
        runner._validate_loaded_publication(module, receipt_publication)


def test_receipt_bound_loader_rejects_preloaded_or_substituted_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    stale = types.ModuleType("world_model.fake_dependency")
    with pytest.raises(PermissionError, match="loaded before receipt-bound import"):
        runner._require_no_preloaded_project_modules({stale.__name__: stale})

    reviewed_source = b"VALUE = 'reviewed-blob'\n"
    blob_oid = runner._git_object_oid("blob", reviewed_source)
    binding = {
        "path": "world_model/fake_dependency.py",
        "mode": "100644",
        "blob_oid": blob_oid,
        "blob_sha256": runner._sha256(reviewed_source),
        "worktree_sha256": runner._sha256(reviewed_source),
    }

    def fake_blob_binding(**arguments: object) -> tuple[dict[str, str], bytes] | None:
        relative = arguments["relative"]
        if str(relative).endswith("/__init__.py"):
            return None
        return copy.deepcopy(binding), reviewed_source

    monkeypatch.setattr(runner, "_blob_binding", fake_blob_binding)
    loader = runner._ReceiptGitBlobLoader(commit="a" * 40)
    spec = loader.find_spec(stale.__name__)
    assert spec is not None
    loaded = importlib.util.module_from_spec(spec)
    loader.exec_module(loaded)
    assert loaded.VALUE == "reviewed-blob"
    loader.validate_loaded_modules({loaded.__name__: loaded})
    loaded.__loader__ = object()
    with pytest.raises(PermissionError, match="was substituted"):
        loader.validate_loaded_modules({loaded.__name__: loaded})


def test_fake_post_load_exception_cleans_finder_and_owned_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    parsed_receipt = {
        "publication": _fake_runner_publication(runner, digit="1"),
    }
    loader = runner._ReceiptGitBlobLoader(commit="1" * 40)
    owned_name = "world_model.cleanup_fixture"
    owned = types.ModuleType(owned_name)
    owned.__loader__ = loader
    owned.__file__ = str(runner.REPOSITORY_ROOT / "world_model/cleanup_fixture.py")
    owned.__spec__ = importlib.util.spec_from_loader(owned_name, loader)
    binding = {
        "path": "world_model/cleanup_fixture.py",
        "mode": "100644",
        "blob_oid": "2" * 40,
        "blob_sha256": "3" * 64,
        "worktree_sha256": "3" * 64,
    }
    loader._loaded[owned_name] = (owned, binding)
    loader._resolved[owned_name] = (binding, b"", False)
    monkeypatch.setitem(sys.modules, owned_name, owned)
    sys.meta_path.insert(0, loader)
    fake_module = types.ModuleType(runner._QUALIFICATION_MODULE)
    monkeypatch.setattr(
        runner,
        "_consume_outer_receipt",
        lambda _: copy.deepcopy(parsed_receipt),
    )
    monkeypatch.setattr(
        runner,
        "_load_frozen_qualification",
        lambda _: (fake_module, loader),
    )
    monkeypatch.setattr(
        runner,
        "_validate_loaded_publication",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("post-load injected failure")),
    )
    with pytest.raises(RuntimeError, match="post-load injected failure") as raised:
        runner.main(["--phase", "development", "--_internal-stage"])
    assert loader not in sys.meta_path
    assert owned_name not in sys.modules
    assert loader._loaded == {}
    assert loader._resolved == {}
    assert loader._baseline_project_modules == {}
    notes = getattr(raised.value, "__notes__", ())
    fallback = getattr(raised.value, "receipt_bound_cleanup_error", "")
    assert any("cleanup also failed" in note for note in notes) or "cleanup also failed" in fallback


def test_fake_loader_cleanup_removes_substituted_project_module() -> None:
    runner = _load_runner_module()
    loader = runner._ReceiptGitBlobLoader(commit="1" * 40)
    name = "world_model.substituted_cleanup_fixture"
    owned = types.ModuleType(name)
    replacement = types.ModuleType(name)
    binding = {
        "path": "world_model/substituted_cleanup_fixture.py",
        "mode": "100644",
        "blob_oid": "2" * 40,
        "blob_sha256": "3" * 64,
        "worktree_sha256": "3" * 64,
    }
    loader._loaded[name] = (owned, binding)
    loader._resolved[name] = (binding, b"", False)
    sys.modules[name] = replacement
    sys.meta_path.insert(0, loader)
    try:
        with pytest.raises(PermissionError):
            runner._release_project_loader(loader)
        assert loader not in sys.meta_path
        assert name not in sys.modules
        assert loader._loaded == {}
        assert loader._resolved == {}
        assert loader._baseline_project_modules == {}
    finally:
        if loader in sys.meta_path:
            sys.meta_path.remove(loader)
        sys.modules.pop(name, None)


def test_fake_loader_cleanup_removes_unregistered_project_module() -> None:
    runner = _load_runner_module()
    loader = runner._ReceiptGitBlobLoader(commit="1" * 40)
    name = "world_model.unregistered_cleanup_fixture"
    sys.modules[name] = types.ModuleType(name)
    sys.meta_path.insert(0, loader)
    try:
        with pytest.raises(PermissionError):
            runner._release_project_loader(loader)
        assert loader not in sys.meta_path
        assert name not in sys.modules
        assert loader._loaded == {}
        assert loader._resolved == {}
        assert loader._baseline_project_modules == {}
    finally:
        if loader in sys.meta_path:
            sys.meta_path.remove(loader)
        sys.modules.pop(name, None)


def test_fake_pre_release_loader_failure_still_releases_runner_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    events: list[str] = []

    class FailingLoader:
        def __init__(self) -> None:
            self.validations = 0

        def validate_loaded_modules(self) -> None:
            self.validations += 1
            events.append(f"validate:{self.validations}")
            if self.validations == 4:
                raise PermissionError("injected pre-release loader failure")

    loader = FailingLoader()
    seal = object()
    fake_qualification = types.SimpleNamespace(
        require_frozen_config=lambda _path: object(),
        canonical_checkpoint_path=lambda: Path("/fixture/checkpoint.pt"),
        canonical_development_report_path=lambda: Path("/fixture/report.json"),
        _mint_runner_invocation_seal=lambda **_kwargs: seal,
        run_development=lambda *_args, **_kwargs: 0,
        _release_runner_invocation_seal=lambda value: events.append(
            "release" if value is seal else "wrong-release"
        ),
    )
    fake_torch = types.SimpleNamespace(set_num_threads=lambda _count: None)
    monkeypatch.setattr(
        runner,
        "_consume_outer_receipt",
        lambda _args: {"publication": {"fixture": "publication"}},
    )
    monkeypatch.setattr(
        runner,
        "_load_frozen_qualification",
        lambda _publication: (fake_qualification, loader),
    )
    monkeypatch.setattr(
        runner,
        "_validate_loaded_publication",
        lambda *_args: {"fixture": "source"},
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: fake_torch)
    monkeypatch.setattr(
        runner,
        "_release_project_loader_preserving_error",
        lambda _loader: events.append("loader-cleanup"),
    )
    with pytest.raises(PermissionError, match="pre-release loader failure"):
        runner.main(["--phase", "development", "--_internal-stage"])
    assert events == [
        "validate:1",
        "validate:2",
        "validate:3",
        "validate:4",
        "release",
        "loader-cleanup",
    ]


def test_runner_has_no_scene_or_runtime_execution_imports() -> None:
    runner = _load_runner_module()
    source = inspect.getsource(runner)
    assert "rgbd_variable_radius_scene" not in source
    assert "OnlineWorldModel" not in source
    assert "scene_specification" not in source
    assert "scene_family_certificate" not in source


def test_report_json_encoder_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        q._report_bytes({"value": float("nan")})


def test_fake_report_schema_rejects_extra_keys() -> None:
    value = {name: None for name in q.DEVELOPMENT_REPORT_SCHEMA}
    value["extra"] = None
    with pytest.raises(ValueError, match="schema differs"):
        q._require_exact_keys(
            value,
            q.DEVELOPMENT_REPORT_SCHEMA,
            label="fake report",
        )


def test_test_module_itself_has_no_formal_execution_calls() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_stems = (
        "scene_specification",
        "manual_kinematic_trajectory",
        "pure_orbital_camera_frame",
        "new_public_model",
        "_evaluate_nominal_batch",
        "_evaluate_manifest_once",
        "run_development",
        "run_qualification",
    )
    for stem in forbidden_stems:
        assert stem + "(" not in source
