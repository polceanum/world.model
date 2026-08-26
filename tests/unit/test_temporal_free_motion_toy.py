from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.run_temporal_free_motion_ladder import (
    _atomic_temporary,
    _QualificationLedger,
    _validate_paths,
    _write_development_checkpoint,
    _write_report,
)
from world_model.training.checkpointing import load_model_weights
from world_model.training.temporal_free_motion_toy import (
    ANCHOR_FRAME_INDEX,
    CONFIRMATION_SEEDS,
    DEFAULT_GATES,
    DEVELOPMENT_AUDIT_SEEDS,
    DEVELOPMENT_TRAIN_SEEDS,
    DEVELOPMENT_UPDATES,
    FINAL_TEST_SEEDS,
    HISTORY_FRAME_INDICES,
    HORIZONS_SECONDS,
    MAXIMUM_ARCHITECTURE_ATTEMPTS,
    PERCEPTION_LATENCY_MAX_SECONDS,
    PROCESS_MAX_RSS_BYTES,
    PROCESS_RSS_DELTA_MAX_BYTES,
    SELECTOR_SEEDS,
    STATE_ONLY_ROLLOUT_LATENCY_MAX_SECONDS,
    TARGET_FRAME_INDICES,
    TemporalQualificationError,
    _assert_free_motion_batch,
    _assert_temporal_protocol,
    _development_batch,
    _model_counts,
    _model_from_config,
    accuracy_metrics,
    latency_metrics,
    oracle_metrics,
    run_protected_qualification,
    temporal_protocol,
    training_objective,
    two_second_gradient_metrics,
)
from world_model.utils.config import load_config

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "temporal_free_motion_toy_cpu.yaml"


def _config():
    return load_config(_CONFIG_PATH)


@pytest.fixture(scope="module")
def development_seed_batch():
    config = _config()
    batch = _development_batch(config, DEVELOPMENT_TRAIN_SEEDS[:1])
    _assert_free_motion_batch(batch, DEVELOPMENT_TRAIN_SEEDS[:1])
    return batch


def test_temporal_protocol_is_frozen_before_generation() -> None:
    config = _config()
    _assert_temporal_protocol(config)

    namespaces = (
        DEVELOPMENT_TRAIN_SEEDS,
        DEVELOPMENT_AUDIT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    all_seeds = [seed for namespace in namespaces for seed in namespace]
    assert len(all_seeds) == len(set(all_seeds))
    assert tuple(range(31_000_000, 31_000_032)) == DEVELOPMENT_TRAIN_SEEDS
    assert tuple(range(31_100_000, 31_100_016)) == DEVELOPMENT_AUDIT_SEEDS
    assert tuple(range(32_000_000, 32_000_016)) == SELECTOR_SEEDS
    assert tuple(range(33_000_000, 33_000_016)) == CONFIRMATION_SEEDS
    assert tuple(range(34_000_000, 34_000_032)) == FINAL_TEST_SEEDS
    assert tuple(range(16)) == HISTORY_FRAME_INDICES
    assert ANCHOR_FRAME_INDEX == 15
    assert HORIZONS_SECONDS == (0.1, 0.25, 0.5, 1.0, 2.0)
    assert TARGET_FRAME_INDICES == (17, 20, 25, 35, 55)
    assert config.simulator.sequence_frames == 56
    assert config.simulator.initial_speed_range == (0.035, 0.035)
    assert config.training.steps == DEVELOPMENT_UPDATES == 32
    assert MAXIMUM_ARCHITECTURE_ATTEMPTS == 2

    protocol = temporal_protocol()
    assert protocol["architecture_attempt"] == 2
    assert protocol["qualification_order"] == ("selector_then_confirmation_then_one_shot_final")
    assert protocol["rollout"] == "AnalyticKinematics_only"
    assert protocol["torch_intraop_threads"] == 1
    assert protocol["optimizer"] == {
        "type": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.0,
        "gradient_clip_l2_norm": 2.0,
        "batch_size": 4,
        "updates": 32,
    }
    assert DEFAULT_GATES.oracle_position_rmse_m == 1.0e-5
    assert DEFAULT_GATES.oracle_velocity_rmse_mps == 1.0e-5
    assert DEFAULT_GATES.oracle_simulator_horizon_position_rmse_m == 5.0e-4
    assert DEFAULT_GATES.oracle_simulator_horizon_velocity_rmse_mps == 5.0e-4
    assert DEFAULT_GATES.current_position_rmse_m == 0.012
    assert DEFAULT_GATES.current_velocity_rmse_mps == 0.02
    assert DEFAULT_GATES.horizon_0_10_rmse_m == 0.012
    assert DEFAULT_GATES.horizon_0_25_rmse_m == 0.014
    assert DEFAULT_GATES.horizon_0_50_rmse_m == 0.018
    assert DEFAULT_GATES.horizon_1_00_rmse_m == 0.025
    assert DEFAULT_GATES.horizon_2_00_rmse_m == 0.040
    assert DEFAULT_GATES.future_velocity_rmse_mps == 0.01
    assert DEFAULT_GATES.trivial_baseline_rmse_ratio == 0.5
    assert DEFAULT_GATES.semigroup_max_abs_m == 1.0e-5
    assert DEFAULT_GATES.semigroup_velocity_max_abs_mps == 1.0e-5
    assert DEFAULT_GATES.minimum_absolute_2s_gradient == 1.0e-7
    assert PERCEPTION_LATENCY_MAX_SECONDS == 2.0
    assert STATE_ONLY_ROLLOUT_LATENCY_MAX_SECONDS == 0.05
    assert PROCESS_MAX_RSS_BYTES == 2_500_000_000
    assert PROCESS_RSS_DELTA_MAX_BYTES == 1_500_000_000


def test_development_loader_rejects_protected_seeds_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated: list[int] = []
    monkeypatch.setattr(
        "world_model.training.temporal_free_motion_toy.generate_episode",
        lambda config, seed: generated.append(seed),
    )

    with pytest.raises(ValueError, match="may not materialize protected"):
        _development_batch(_config(), SELECTOR_SEEDS[:1])

    assert generated == []


def test_oracle_fit_and_deployed_analytic_rollout_are_exact(
    development_seed_batch,
) -> None:
    config = _config()
    model = _model_from_config(config)
    metrics = oracle_metrics(model, config, development_seed_batch)

    assert metrics["valid_fraction"] == 1.0
    assert metrics["position_rmse_m"] <= DEFAULT_GATES.oracle_position_rmse_m
    assert metrics["velocity_rmse_mps"] <= DEFAULT_GATES.oracle_velocity_rmse_mps
    assert max(metrics["simulator_horizon_position_rmse_m"].values()) <= 5.0e-4
    assert max(metrics["simulator_horizon_velocity_rmse_mps"].values()) <= 5.0e-4


def test_rgb_history_state_and_long_horizons_pass_synthetic_development_gate(
    development_seed_batch,
) -> None:
    model = _model_from_config(_config())
    metrics = accuracy_metrics(model, development_seed_batch)

    assert metrics["valid_fraction"] == 1.0
    assert metrics["centre_rmse_pixels"] <= DEFAULT_GATES.centre_rmse_pixels
    assert metrics["radius_relative_rmse"] <= DEFAULT_GATES.radius_relative_rmse
    assert metrics["current_position_rmse_m"] <= DEFAULT_GATES.current_position_rmse_m
    assert metrics["current_velocity_rmse_mps"] <= DEFAULT_GATES.current_velocity_rmse_mps
    assert metrics["horizon_rmse_m"]["0.10"] <= DEFAULT_GATES.horizon_0_10_rmse_m
    assert metrics["horizon_rmse_m"]["0.25"] <= DEFAULT_GATES.horizon_0_25_rmse_m
    assert metrics["horizon_rmse_m"]["0.50"] <= DEFAULT_GATES.horizon_0_50_rmse_m
    assert metrics["horizon_rmse_m"]["1.00"] <= DEFAULT_GATES.horizon_1_00_rmse_m
    assert metrics["horizon_rmse_m"]["2.00"] <= DEFAULT_GATES.horizon_2_00_rmse_m
    assert max(metrics["future_velocity_rmse_mps"].values()) <= 0.01
    assert max(metrics["persistence_rmse_ratio"].values()) <= 0.5
    assert max(metrics["zero_velocity_rmse_ratio"].values()) <= 0.5
    assert metrics["semigroup_max_abs_m"] <= DEFAULT_GATES.semigroup_max_abs_m
    assert metrics["semigroup_velocity_max_abs_mps"] <= 1.0e-5


def test_two_second_loss_reaches_every_mask_scalar(development_seed_batch) -> None:
    model = _model_from_config(_config())
    metrics = two_second_gradient_metrics(model, development_seed_batch)

    assert metrics["finite_to_every_mask_scalar"] is True
    assert metrics["nonzero_to_every_mask_scalar"] is True
    assert len(metrics["gradient_values"]) == 4
    assert metrics["minimum_absolute_gradient"] >= 1.0e-7

    loss, _ = training_objective(model, development_seed_batch)
    loss.backward()
    flattened = torch.cat(
        [parameter.grad.detach().reshape(-1) for parameter in model.mask_parameters]
    )
    assert flattened.numel() == 4
    assert torch.isfinite(flattened).all()
    assert (flattened != 0).all()


def test_perception_and_state_only_rollout_latency_are_separate(
    development_seed_batch,
) -> None:
    model = _model_from_config(_config())
    metrics = latency_metrics(model, development_seed_batch)

    assert 0.0 < metrics["perception_median_seconds_per_episode"] < 2.0
    assert 0.0 < metrics["state_only_rollout_median_seconds"] < 0.05
    assert (
        metrics["state_only_rollout_median_seconds"]
        < metrics["perception_median_seconds_per_episode"]
    )
    # ru_maxrss is process-lifetime high water and can include unrelated tests
    # in the full suite. The fresh runner, not this shared pytest process,
    # applies the absolute preregistered ceiling.
    assert metrics["process_max_rss_bytes"] > 0
    assert metrics["process_rss_delta_bytes"] <= PROCESS_RSS_DELTA_MAX_BYTES


def test_model_capacity_is_only_four_mask_scalars() -> None:
    model = _model_from_config(_config())
    counts = _model_counts(model)

    assert counts["trainable_parameters"] == 4
    assert counts["total_parameters"] == 4
    assert counts["persistent_state_bytes"] == 28
    assert sorted(model.state_dict()) == [
        "gravity",
        "state_estimator.mask_head.bias",
        "state_estimator.mask_head.weight",
    ]


def test_protected_qualification_stops_before_confirmation_and_final(
    monkeypatch: pytest.MonkeyPatch,
    development_seed_batch,
) -> None:
    requested: list[str] = []

    def protected_batch(config, split):
        requested.append(split)
        return development_seed_batch

    monkeypatch.setattr(
        "world_model.training.temporal_free_motion_toy._protected_batch",
        protected_batch,
    )
    monkeypatch.setattr(
        "world_model.training.temporal_free_motion_toy.evaluate_gate",
        lambda model, config, batch, gates: {
            "passed": False,
            "failures": ["synthetic selector rejection"],
        },
    )

    with pytest.raises(TemporalQualificationError) as captured:
        run_protected_qualification(
            _model_from_config(_config()),
            _config(),
            access_recorder=lambda split: requested.append(f"authorized:{split}"),
        )

    assert requested == ["authorized:selector", "selector"]
    assert captured.value.report["stopped_after"] == "selector"
    assert captured.value.report["access_started"] == {
        "selector": True,
        "confirmation": False,
        "final_test": False,
    }
    assert "confirmation" not in captured.value.report["rungs"]
    assert "final_test" not in captured.value.report


def test_development_checkpoint_is_project_compatible_and_weights_only(
    tmp_path: Path,
) -> None:
    config = _config()
    model = _model_from_config(config)
    checkpoint = tmp_path / "temporal.pt"
    report = {
        "passed": True,
        "review_ready": True,
        "protocol": temporal_protocol(),
        "training": {"updates": DEVELOPMENT_UPDATES},
        "development_audit": {"passed": True},
        "model": _model_counts(model),
        "resource": {"process_max_rss_bytes": 1, "process_rss_delta_bytes": 0},
    }
    source = {
        "commit": "0" * 40,
        "dirty": False,
        "worktree_fingerprint": "1" * 64,
        "runtime_source_fingerprint": "2" * 64,
    }

    _write_development_checkpoint(
        checkpoint,
        model=model,
        config=config,
        report=report,
        source_provenance=source,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == DEVELOPMENT_UPDATES
    assert payload["optimizer_state"] is None
    assert payload["scheduler_state"] is None
    assert payload["metrics"]["artifact_kind"] == ("temporal_free_motion_development_weights_only")
    assert payload["metrics"]["exact_resume"] is False
    assert payload["metrics"]["protected_data_materialized"] is False
    reloaded = _model_from_config(config)
    loaded = load_model_weights(checkpoint, model=reloaded, expected_config=config)
    assert loaded["weight_load_missing_keys"] == ()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0.0, atol=0.0)


def test_runner_paths_and_json_fail_closed(tmp_path: Path) -> None:
    shared = tmp_path / "evidence"
    with pytest.raises(ValueError, match="must be distinct"):
        _validate_paths("development", shared, shared.parent / "." / shared.name)

    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        _write_report(report, {"metric": float("nan")})
    assert not report.exists()

    checkpoint = tmp_path / "missing.pt"
    development_report = tmp_path / "development.json"
    development_report.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _validate_paths(
            "qualification",
            report,
            checkpoint,
            development_report=development_report,
            ledger_path=tmp_path / "ledger.json",
        )

    alias_report = tmp_path / "alias.json"
    with pytest.raises(ValueError, match="atomic temporary paths must be distinct"):
        _validate_paths(
            "development",
            alias_report,
            _atomic_temporary(alias_report),
        )

    _write_report(report, {"protected_data_materialized": False})
    assert json.loads(report.read_text(encoding="utf-8")) == {"protected_data_materialized": False}


def test_durable_qualification_ledger_is_exclusive_and_ordered(tmp_path: Path) -> None:
    path = tmp_path / "access.json"
    ledger = _QualificationLedger(path, {"artifact_kind": "test"})
    ledger.record_access("selector")
    with pytest.raises(RuntimeError, match="order violation"):
        ledger.record_access("final_test")
    ledger.finish({"passed": False, "stopped_after": "selector"})

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["access_started"] == {
        "selector": True,
        "confirmation": False,
        "final_test": False,
    }
    assert persisted["protected_data_materialized"] is True
    with pytest.raises(FileExistsError):
        _QualificationLedger(path, {"artifact_kind": "test"})


def test_unexpected_selector_error_keeps_truthful_access_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorized: list[str] = []
    monkeypatch.setattr(
        "world_model.training.temporal_free_motion_toy._protected_batch",
        lambda config, split: (_ for _ in ()).throw(RuntimeError("synthetic read failure")),
    )

    with pytest.raises(TemporalQualificationError) as captured:
        run_protected_qualification(
            _model_from_config(_config()),
            _config(),
            access_recorder=authorized.append,
        )

    assert authorized == ["selector"]
    assert captured.value.report["protected_data_materialized"] is True
    assert captured.value.report["access_started"]["selector"] is True
    assert captured.value.report["stopped_after"] == "selector_exception"
    assert captured.value.report["unexpected_error"]["type"] == "RuntimeError"
