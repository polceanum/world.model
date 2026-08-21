from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.replay_promotion_mps as replay
from scripts.replay_promotion_mps import (
    _canonical_sha256,
    _capture_checkpoint_snapshot,
    _load_paired_latency_evidence,
    _physical_replay_contract,
    _promotion_eligibility,
    _replay_checkpoint_with_rng_reset,
)
from world_model.evaluation.evaluator import (
    _primary_physical_metrics,
    _primary_physical_metrics_hash_exclusion_declaration,
)
from world_model.utils.config import load_config

_SOURCE_FINGERPRINT = "a" * 64
_REFERENCE_SHA = "1" * 64
_CANDIDATE_SHA = "2" * 64
_REFERENCE_BYTES = 123
_CANDIDATE_BYTES = 456


def _contract_and_config():
    config = load_config(
        "configs/grounded_convergence_mps.yaml",
        overrides=[
            "device.preference=mps",
            "device.closed_loop_preference=mps",
            "training.num_workers=0",
        ],
    )
    contract = _physical_replay_contract(
        config,
        expected_device="mps",
        expected_precision="float32",
        runtime_source_fingerprint=_SOURCE_FINGERPRINT,
    )
    return contract, config


def _latency_metrics(contract: dict[str, object]) -> dict[str, float | None]:
    episode_scenarios = contract["evaluation_episode_scenarios"]
    scenario_mixture = contract["scenario_mixture"]
    assert isinstance(episode_scenarios, list)
    assert isinstance(scenario_mixture, list)
    metrics: dict[str, float | None] = {
        "rgb_global_update_latency_mean_ms": 10.0,
        "rgb_global_update_latency_sum_ms": 320.0,
        "rgb_global_update_latency_sample_count": 32.0,
        "rgb_fast_update_latency_mean_ms": 4.0,
        "rgb_fast_update_latency_sum_ms": 5120.0,
        "rgb_fast_update_latency_sample_count": 1280.0,
        "future_rollout_latency_mean_ms": 20.0,
        "future_rollout_latency_sum_ms": 25600.0,
        "future_rollout_latency_sample_count": 1280.0,
        "nonfinite_output_count": 0.0,
        "evaluated_episodes": 32.0,
        "posterior_current_position_sse": 1.25,
        "posterior_current_position_coordinate_count": 96.0,
        "injected_perturbation_batch_updates": 0.0,
        "recovery_probe_evaluated_episodes": 0.0,
        "recovery_probe_nonfinite_output_count": 0.0,
        "perturbation_prior_position_error_m": None,
        "perturbation_posterior_position_error_m": None,
        "perturbation_correction_improvement_m": None,
        "perturbation_correction_improvement_fraction": None,
        "perturbation_positive_correction_rate": None,
        "perturbation_evaluated_object_horizons": 0.0,
        "recovery_probe_post_observation_std_contraction_mean_m": None,
        "post_observation_std_contraction_mean_m": None,
    }
    for scenario in scenario_mixture:
        metrics[f"scenario_{scenario}_episode_count"] = float(episode_scenarios.count(scenario))
    return metrics


def _report(
    checkpoint_sha256: str,
    checkpoint_byte_count: int,
    contract: dict[str, object],
) -> dict[str, object]:
    protocol_template = contract["evaluator_protocol"]
    assert isinstance(protocol_template, dict)
    protocol = deepcopy(protocol_template)
    protocol["checkpoint_sha256"] = checkpoint_sha256
    metrics = _latency_metrics(contract)
    primary = _primary_physical_metrics(metrics)
    rollout_protocol = contract["rollout_validation_protocol"]
    assert isinstance(rollout_protocol, dict)
    metadata = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_byte_count": checkpoint_byte_count,
        "checkpoint_identity_source": "captured_pre_evaluation_immutable_byte_snapshot",
        "resolved_evaluation_protocol": protocol,
        "resolved_evaluation_protocol_sha256": _canonical_sha256(protocol),
        "evaluation_metric_schema_version": protocol["metric_schema_version"],
        "resolved_evaluation_config_sha256": contract["resolved_config_sha256"],
        "evaluation_source_provenance": {
            "commit": "deadbeef",
            "dirty": False,
            "worktree_fingerprint": "b" * 64,
            "runtime_source_fingerprint": _SOURCE_FINGERPRINT,
        },
        "simulator_version": rollout_protocol["simulator_version"],
        "evaluation_simulator_version": rollout_protocol["simulator_version"],
        "scenario_mixture": contract["scenario_mixture"],
        "resolved_scenarios": contract["resolved_scenarios"],
        "per_scenario_metrics_schema": protocol["per_scenario_metric_schema"],
        "per_scenario_metrics_scenarios": contract["scenario_mixture"],
        "per_scenario_metrics_horizons": contract["horizons_observation_grid"],
        "evaluation_episode_scenarios": contract["evaluation_episode_scenarios"],
        "split": "validation",
        "episodes": 32,
        "batches": contract["evaluator_batch_count"],
        "device": "mps",
        "precision": "float32",
        "rgb_only": True,
        "oracle_runtime_input_used": False,
        "primary_online_pass_evaluator_state_perturbation_free": True,
        "primary_posterior_trace_sha256": "c" * 64,
        "primary_posterior_trace_frame_count": contract["primary_posterior_trace_frame_count"],
        "primary_posterior_trace_schema": "world_belief_tensor_fields_v1",
        "primary_physical_metrics_sha256": _canonical_sha256(primary),
        "primary_physical_metrics_hashed_keys": sorted(primary),
        "primary_physical_metrics_scope": (
            "clean_primary_metrics_before_isolated_recovery_probe_append"
        ),
        "primary_physical_metrics_hash_excludes": (
            _primary_physical_metrics_hash_exclusion_declaration()
        ),
        "recovery_probe_enabled": False,
        "runtime_hypothesis_pool_enabled": False,
        "evaluation_perturbations_applied": False,
        "evaluation_seed_protocol": "standard",
        "evaluation_seed_role": "standard_validation_evaluation",
        "evaluation_seed_offset": 0,
        "evaluation_seed_count": 32,
        "evaluation_seed_first": 100000,
        "evaluation_seed_last": 100031,
        "evaluation_episode_seeds": list(range(100000, 100032)),
        "evaluation_seed_overlaps_training_validation": True,
        "evaluation_seed_overlaps_test_range": False,
    }
    return {"metadata": metadata, "metrics": metrics, "limitations": []}


def _write_report(path: Path, payload: dict[str, object]) -> bytes:
    content = json.dumps(payload, allow_nan=False, sort_keys=True).encode("utf-8")
    path.write_bytes(content)
    return content


def _load_pair(
    tmp_path: Path,
    *,
    reference: dict[str, object] | None = None,
    candidate: dict[str, object] | None = None,
):
    contract, _ = _contract_and_config()
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(
        reference_path,
        reference or _report(_REFERENCE_SHA, _REFERENCE_BYTES, contract),
    )
    _write_report(
        candidate_path,
        candidate or _report(_CANDIDATE_SHA, _CANDIDATE_BYTES, contract),
    )
    result = _load_paired_latency_evidence(
        reference_report=reference_path,
        candidate_report=candidate_path,
        reference_checkpoint_sha256=_REFERENCE_SHA,
        candidate_checkpoint_sha256=_CANDIDATE_SHA,
        reference_checkpoint_byte_count=_REFERENCE_BYTES,
        candidate_checkpoint_byte_count=_CANDIDATE_BYTES,
        replay_contract=contract,
    )
    return result, reference_path, candidate_path


def test_promotion_replay_binds_production_shaped_matched_latency_reports(
    tmp_path: Path,
) -> None:
    (candidate, reference, binding), _, _ = _load_pair(tmp_path)

    assert candidate["future_rollout_latency_sample_count"] == 1280.0
    assert reference["rgb_global_update_latency_sample_count"] == 32.0
    assert binding["seed_manifest"] == list(range(100000, 100032))
    assert len(binding["evaluation_episode_scenarios"]) == 32
    assert binding["horizons_observation_grid"] == [
        "0.100s",
        "0.250s",
        "0.500s",
        "0.750s",
        "1.000s",
    ]
    assert binding["runtime_source_fingerprint"] == _SOURCE_FINGERPRINT


def test_physical_qualification_does_not_claim_comprehensive_without_latency() -> None:
    physical, comprehensive = _promotion_eligibility(
        accuracy_improves=True,
        reference_guardrail_failures=[],
        training_support_failures=[],
        mutable_support_failures=[],
        latency_promotion_eligible=False,
    )

    assert physical is True
    assert comprehensive is False


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("checkpoint", "checkpoint identity"),
        ("checkpoint_bytes", "checkpoint byte count"),
        ("manifest", "evaluation protocol"),
        ("config", "evaluation protocol"),
        ("device", "metadata 'device'"),
        ("scenarios", "metadata 'evaluation_episode_scenarios'"),
        ("protocol_schema", "evaluation protocol"),
        ("protocol_hash", "protocol hash"),
        ("source", "runtime source fingerprint"),
        ("intervention", "evaluation protocol"),
        ("physical_hash", "physical metrics hash"),
        ("hash_keys", "hashed-key evidence"),
        ("hash_excludes", "hash exclusion scope"),
        ("episodes", "evaluated-episode count"),
        ("scenario_count", "scenario episode count"),
        ("latency_count", "contradictory latency counts"),
        ("nonfinite", "non-finite"),
    ],
)
def test_promotion_replay_rejects_unbound_latency_reports(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    contract, _ = _contract_and_config()
    reference = _report(_REFERENCE_SHA, _REFERENCE_BYTES, contract)
    candidate = deepcopy(_report(_CANDIDATE_SHA, _CANDIDATE_BYTES, contract))
    metadata = candidate["metadata"]
    metrics = candidate["metrics"]
    assert isinstance(metadata, dict)
    assert isinstance(metrics, dict)
    protocol = metadata["resolved_evaluation_protocol"]
    assert isinstance(protocol, dict)
    if mutation == "checkpoint":
        metadata["checkpoint_sha256"] = "wrong-sha"
    elif mutation == "checkpoint_bytes":
        metadata["checkpoint_byte_count"] = _CANDIDATE_BYTES + 1
    elif mutation == "manifest":
        protocol["seed_manifest"][-1] = 100032
        metadata["resolved_evaluation_protocol_sha256"] = _canonical_sha256(protocol)
    elif mutation == "config":
        protocol["resolved_config_sha256"] = "wrong-config"
        metadata["resolved_evaluation_protocol_sha256"] = _canonical_sha256(protocol)
    elif mutation == "device":
        metadata["device"] = "cpu"
    elif mutation == "scenarios":
        metadata["evaluation_episode_scenarios"] = list(
            reversed(metadata["evaluation_episode_scenarios"])
        )
    elif mutation == "protocol_schema":
        protocol["schema_version"] = "legacy"
        metadata["resolved_evaluation_protocol_sha256"] = _canonical_sha256(protocol)
    elif mutation == "protocol_hash":
        metadata["resolved_evaluation_protocol_sha256"] = "wrong-hash"
    elif mutation == "source":
        metadata["evaluation_source_provenance"]["runtime_source_fingerprint"] = "d" * 64
    elif mutation == "intervention":
        protocol["runtime_intervention"]["runtime_hypothesis_pool"] = True
        metadata["resolved_evaluation_protocol_sha256"] = _canonical_sha256(protocol)
    elif mutation == "physical_hash":
        metrics["posterior_current_position_sse"] = 5.0
    elif mutation == "hash_keys":
        metadata["primary_physical_metrics_hashed_keys"] = ["nonfinite_output_count"]
    elif mutation == "hash_excludes":
        metadata["primary_physical_metrics_hash_excludes"] = {
            "latency_metric_name_substring": "latency",
            "recovery_only_metric_names": [],
        }
    elif mutation == "episodes":
        metrics["evaluated_episodes"] = 31.0
        primary = _primary_physical_metrics(metrics)
        metadata["primary_physical_metrics_hashed_keys"] = sorted(primary)
        metadata["primary_physical_metrics_sha256"] = _canonical_sha256(primary)
    elif mutation == "scenario_count":
        scenario = contract["scenario_mixture"][0]
        metrics[f"scenario_{scenario}_episode_count"] += 1.0
        primary = _primary_physical_metrics(metrics)
        metadata["primary_physical_metrics_hashed_keys"] = sorted(primary)
        metadata["primary_physical_metrics_sha256"] = _canonical_sha256(primary)
    elif mutation == "latency_count":
        metrics["rgb_fast_update_latency_sample_count"] += 1.0
    elif mutation == "nonfinite":
        metrics["nonfinite_output_count"] = 1.0
        primary = _primary_physical_metrics(metrics)
        metadata["primary_physical_metrics_hashed_keys"] = sorted(primary)
        metadata["primary_physical_metrics_sha256"] = _canonical_sha256(primary)

    with pytest.raises(ValueError, match=error):
        _load_pair(tmp_path, reference=reference, candidate=candidate)


def test_latency_report_is_parsed_and_hashed_from_one_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _ = _contract_and_config()
    reference_payload = _report(_REFERENCE_SHA, _REFERENCE_BYTES, contract)
    candidate_payload = _report(_CANDIDATE_SHA, _CANDIDATE_BYTES, contract)
    expected_reference = json.dumps(
        reference_payload,
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8")
    expected_candidate = json.dumps(
        candidate_payload,
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8")
    original_capture = replay._capture_json_report

    def capture_then_mutate(path: Path, *, description: str):
        captured = original_capture(path, description=description)
        path.write_text('{"mutated": true}', encoding="utf-8")
        return captured

    monkeypatch.setattr(replay, "_capture_json_report", capture_then_mutate)
    (candidate, reference, binding), _, _ = _load_pair(
        tmp_path,
        reference=reference_payload,
        candidate=candidate_payload,
    )

    assert candidate["evaluated_episodes"] == 32.0
    assert reference["evaluated_episodes"] == 32.0
    assert (
        binding["reference_latency_report_sha256"] == hashlib.sha256(expected_reference).hexdigest()
    )
    assert (
        binding["candidate_latency_report_sha256"] == hashlib.sha256(expected_candidate).hexdigest()
    )


def test_primary_digest_reconstructs_from_clean_and_probed_final_metrics() -> None:
    contract, _ = _contract_and_config()
    clean_metrics = _latency_metrics(contract)
    probed_metrics = deepcopy(clean_metrics)
    probed_metrics.update(
        {
            "perturbation_prior_position_error_m": 0.8,
            "perturbation_posterior_position_error_m": 0.3,
            "perturbation_correction_improvement_m": 0.5,
            "perturbation_correction_improvement_fraction": 0.625,
            "perturbation_positive_correction_rate": 1.0,
            "perturbation_evaluated_object_horizons": 4.0,
            "injected_perturbation_batch_updates": 1.0,
            "recovery_probe_evaluated_episodes": 1.0,
            "recovery_probe_nonfinite_output_count": 0.0,
            "recovery_probe_post_observation_std_contraction_mean_m": 0.1,
            "post_observation_std_contraction_mean_m": 0.1,
        }
    )
    expected_exclusions = {
        "latency_metric_name_substring": "latency",
        "recovery_only_metric_names": sorted(
            {
                "perturbation_prior_position_error_m",
                "perturbation_posterior_position_error_m",
                "perturbation_correction_improvement_m",
                "perturbation_correction_improvement_fraction",
                "perturbation_positive_correction_rate",
                "perturbation_evaluated_object_horizons",
                "injected_perturbation_batch_updates",
                "recovery_probe_evaluated_episodes",
                "recovery_probe_nonfinite_output_count",
                "recovery_probe_post_observation_std_contraction_mean_m",
                "post_observation_std_contraction_mean_m",
            }
        ),
    }
    assert _primary_physical_metrics_hash_exclusion_declaration() == expected_exclusions
    expected_primary = {
        name: value
        for name, value in clean_metrics.items()
        if "latency" not in name and name not in expected_exclusions["recovery_only_metric_names"]
    }
    clean_primary = _primary_physical_metrics(clean_metrics)
    probed_primary = _primary_physical_metrics(probed_metrics)
    assert clean_primary == expected_primary
    assert probed_primary == expected_primary
    assert _canonical_sha256(clean_primary) == _canonical_sha256(probed_primary)


def test_checkpoint_snapshot_closes_mutable_path_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "last.pt"
    original_bytes = b"immutable checkpoint bytes"
    source.write_bytes(original_bytes)
    observed: list[tuple[Path, bytes, str]] = []

    def fake_replay(checkpoint: Path, **kwargs):
        observed.append((checkpoint, checkpoint.read_bytes(), kwargs["role"]))
        return object(), {}

    monkeypatch.setattr(replay, "_replay_checkpoint", fake_replay)
    _, config = _contract_and_config()
    with _capture_checkpoint_snapshot(source) as captured:
        source.write_bytes(b"concurrently replaced")
        _replay_checkpoint_with_rng_reset(
            captured.snapshot_path,
            config=config,
            device=torch.device("cpu"),
            output=tmp_path,
            role="candidate",
        )
        assert captured.sha256 == hashlib.sha256(original_bytes).hexdigest()
        assert captured.byte_count == len(original_bytes)

    assert observed[0][0] != source
    assert observed[0][1] == original_bytes
    assert observed[0][2] == "candidate"


def test_each_physical_arm_starts_from_identical_rng_state_in_any_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = _contract_and_config()
    draws: list[tuple[str, float, float, float]] = []

    def fake_replay(checkpoint: Path, **kwargs):
        del checkpoint
        draws.append(
            (
                kwargs["role"],
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )
        )
        return object(), {}

    monkeypatch.setattr(replay, "_replay_checkpoint", fake_replay)
    for role in ("candidate", "reference"):
        _replay_checkpoint_with_rng_reset(
            tmp_path / f"{role}.pt",
            config=config,
            device=torch.device("cpu"),
            output=tmp_path,
            role=role,
        )

    assert [draw[0] for draw in draws] == ["candidate", "reference"]
    assert draws[0][1:] == draws[1][1:]


def test_rng_reset_explicitly_seeds_available_mps_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        replay,
        "seed_everything",
        lambda seed, *, deterministic: calls.append(("all", seed)),
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: calls.append(("mps", seed)))

    replay._reset_physical_replay_rng(seed=73, deterministic=True)

    assert calls == [("all", 73), ("mps", 73)]
