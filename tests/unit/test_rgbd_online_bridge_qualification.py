from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.training.rgbd_online_bridge_qualification import (
    CONFIRMATION_SEEDS,
    DEFAULT_GATES,
    DEVELOPMENT_SEEDS,
    EMPTY_MODEL_STATE_SHA256,
    FINAL_TEST_SEEDS,
    FROZEN_CONFIG_SHA256,
    HISTORY_GRADIENT_TARGETS,
    HORIZONS_SECONDS,
    SELECTOR_SEEDS,
    TARGET_FRAME_INDICES,
    QualificationLedger,
    _gradient_metrics,
    _history_gradient_diagnostics,
    _missing_depth_metrics,
    _no_foreground_metrics,
    _run_public_batch,
    assert_rgbd_online_bridge_config,
    bridge_protocol,
    canonical_sha256,
    evaluate_seed_manifest,
    gate_failures,
    new_public_model,
    new_standalone_estimator,
    sha256_file,
    validate_distinct_paths,
)
from world_model.utils.config import load_config

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_online_free_motion_cpu.yaml"


def _passing_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {
        "current_position_rmse_m": 0.0,
        "current_position_axis_rmse_m": 0.0,
        "current_velocity_rmse_mps": 0.0,
        "current_velocity_axis_rmse_mps": 0.0,
        "horizon_velocity_rmse_mps": 0.0,
        "horizon_velocity_axis_rmse_mps": 0.0,
        "maximum_position_error_growth_slope_mps": 0.0,
        "early_stationary_additive_regression_m": 0.0,
        "long_stationary_rmse_ratio": 0.0,
        "zero_velocity_rmse_ratio": 0.0,
        "public_standalone_current_position_rmse_m": 0.0,
        "public_standalone_current_velocity_rmse_mps": 0.0,
        "history_standalone_measurement_max_abs_m": 0.0,
        "fixed_prior_max_abs": 0.0,
        "identity_change_count": 0.0,
        "persistent_object_id_min": 0.0,
        "persistent_object_id_max": 0.0,
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
        "direct_metric_position_owner_max_abs_m": 0.0,
        "direct_position_field_count": 0.0,
        "direct_velocity_position_change_max_abs_m": 0.0,
        "public_rollout_output_alias_count": 0.0,
        "public_query_time_max_abs_seconds": 0.0,
        "ingested_frame_count_min": 16.0,
        "ingested_frame_count_max": 16.0,
        "public_predict_calls_per_batch_min": 1.0,
        "public_predict_calls_per_batch_max": 1.0,
        "missing_depth_last_measurement_valid_fraction": 0.0,
        "missing_depth_fit_valid_fraction": 0.0,
        "missing_depth_direct_velocity_calls": 0.0,
        "missing_depth_history_sample_count_min": 16.0,
        "missing_depth_history_sample_count_max": 16.0,
        "missing_depth_history_valid_count_min": 15.0,
        "missing_depth_history_valid_count_max": 15.0,
        "missing_depth_finite_fraction": 1.0,
        "no_foreground_last_measurement_valid_fraction": 0.0,
        "no_foreground_fit_valid_fraction": 0.0,
        "no_foreground_direct_velocity_calls": 0.0,
        "no_foreground_history_sample_count_min": 16.0,
        "no_foreground_history_sample_count_max": 16.0,
        "no_foreground_history_valid_count_min": 15.0,
        "no_foreground_history_valid_count_max": 15.0,
        "no_foreground_finite_fraction": 1.0,
        "semigroup_position_max_abs_m": 0.0,
        "semigroup_velocity_max_abs_mps": 0.0,
        "public_direct_position_max_abs_m": 0.0,
        "public_direct_velocity_max_abs_mps": 0.0,
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
    for horizon in HORIZONS_SECONDS:
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = 0.0
        metrics[f"horizon_{label}_position_axis_rmse_m"] = 0.0
        metrics[f"horizon_{label}_velocity_rmse_mps"] = 0.0
        metrics[f"horizon_{label}_velocity_axis_rmse_mps"] = 0.0
        metrics[f"horizon_{label}_position_error_growth_m"] = 0.0
        metrics[f"public_standalone_horizon_{label}_position_rmse_m"] = 0.0
        metrics[f"public_standalone_horizon_{label}_velocity_rmse_mps"] = 0.0
    for output_name in (
        "current_position",
        "current_velocity",
        *(f"horizon_{horizon:.2f}_position" for horizon in HORIZONS_SECONDS),
        *(f"horizon_{horizon:.2f}_velocity" for horizon in HORIZONS_SECONDS),
    ):
        metrics[f"gradient_l1/{output_name}/rgb"] = 1.0
        metrics[f"gradient_l1/{output_name}/depth"] = 1.0
    for output_name in HISTORY_GRADIENT_TARGETS:
        for modality in ("rgb", "depth"):
            metrics[f"gradient_min_history_frame_l1/{output_name}/{modality}"] = 1.0
            metrics[f"gradient_supported_history_frames/{output_name}/{modality}"] = 16.0
    return metrics


def _handcrafted_rgbd_batch(*, batch_size: int = 1) -> dict[str, object]:
    # Mirror a real development episode: the public runtime consumes the first
    # 16 history frames while labels extend through the two-second target.
    frames = TARGET_FRAME_INDICES[-1] + 1
    rgb = torch.zeros((batch_size, frames, 3, 64, 64), dtype=torch.float32)
    rgb[:, :, 0, 22:42, 24:44] = 0.9
    rgb[:, :, 1, 22:42, 24:44] = 0.35
    depth = torch.full((batch_size, frames, 1, 64, 64), 2.0, dtype=torch.float32)
    intrinsics = (
        torch.tensor(
            [[80.0, 0.0, 31.5], [0.0, 80.0, 31.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        .reshape(1, 1, 3, 3)
        .expand(batch_size, frames, -1, -1)
        .clone()
    )
    world_from_camera = (
        torch.eye(4, dtype=torch.float32)
        .reshape(1, 1, 4, 4)
        .expand(batch_size, frames, -1, -1)
        .clone()
    )
    timestamps = (
        torch.arange(frames, dtype=torch.float32).reshape(1, frames).expand(batch_size, -1).clone()
        / 20.0
    )
    return {
        "rgb": rgb,
        "depth": depth,
        "timestamps": timestamps,
        "camera": {
            "intrinsics": intrinsics,
            "world_from_camera": world_from_camera,
        },
    }


def test_protocol_predeclares_disjoint_exact_manifests_and_self_hash() -> None:
    namespaces = (
        DEVELOPMENT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    assert tuple(map(len, namespaces)) == (24, 16, 16, 32)
    assert tuple(range(45_000_000, 45_000_024)) == DEVELOPMENT_SEEDS
    assert tuple(range(46_000_000, 46_000_016)) == SELECTOR_SEEDS
    assert tuple(range(47_000_000, 47_000_016)) == CONFIRMATION_SEEDS
    assert tuple(range(48_000_000, 48_000_032)) == FINAL_TEST_SEEDS
    flattened = tuple(seed for namespace in namespaces for seed in namespace)
    assert len(flattened) == len(set(flattened))

    protocol = bridge_protocol()
    stated = protocol.pop("protocol_sha256")
    assert stated == canonical_sha256(protocol)
    assert protocol["runtime"]["ingested_frame_indices"] == list(range(16))
    assert protocol["runtime"]["horizon_offsets_seconds"] == list(HORIZONS_SECONDS)
    assert protocol["optimizer"] is None
    assert protocol["optimizer_updates"] == 0


def test_frozen_config_binds_radius_drag_single_position_owner_and_zero_state() -> None:
    config = load_config(CONFIG_PATH)
    assert sha256_file(CONFIG_PATH) == FROZEN_CONFIG_SHA256
    assert_rgbd_online_bridge_config(config)
    assert config.model.rgbd.linear_drag == config.simulator.drag_range[0] == 0.05
    assert config.model.filter.direct_metric_position_update is True
    assert config.evaluation.rgb_only is False

    public = new_public_model(config)
    standalone = new_standalone_estimator(config)
    assert public.belief_factory.initial_radius == config.model.rgbd.world_radius
    assert public.belief_factory.initial_drag == config.model.rgbd.linear_drag
    assert standalone.world_radius_m == config.model.rgbd.world_radius
    assert standalone.drag == config.model.rgbd.linear_drag
    assert not tuple(public.parameters()) and not tuple(public.buffers())
    assert public.state_dict() == {}
    assert not tuple(standalone.parameters()) and not tuple(standalone.buffers())
    assert standalone.state_dict() == {}
    assert canonical_sha256([]) == EMPTY_MODEL_STATE_SHA256


def test_config_rejects_a_standalone_public_drag_mismatch() -> None:
    config = load_config(CONFIG_PATH)
    changed = replace(
        config,
        model=replace(
            config.model,
            rgbd=replace(config.model.rgbd, linear_drag=0.06),
        ),
    )
    with pytest.raises(ValueError, match="model.rgbd.linear_drag"):
        assert_rgbd_online_bridge_config(changed)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    (
        ("current_position_rmse_m", 1.0),
        ("horizon_0.10_position_rmse_m", 1.0),
        ("horizon_2.00_position_error_growth_m", 1.0),
        ("public_standalone_current_velocity_rmse_mps", 1.0),
        ("identity_change_count", 1.0),
        ("history_sample_count_min", 15.0),
        ("position_owner_count_max", 2.0),
        ("direct_position_field_count", 1.0),
        ("missing_depth_fit_valid_fraction", 1.0),
        ("no_foreground_last_measurement_valid_fraction", 1.0),
        ("gradient_l1/current_position/rgb", 0.0),
        ("gradient_supported_history_frames/horizon_2.00_position/depth", 15.0),
        ("semigroup_position_max_abs_m", 1.0),
        ("perception_latency_seconds", 3.0),
        ("persistent_runtime_tensor_state_bytes_max", 32_769.0),
        ("learned_parameter_count", 1.0),
    ),
)
def test_gate_recomputation_fails_closed_by_category(key: str, bad_value: float) -> None:
    passing = _passing_metrics()
    assert gate_failures(passing) == []
    passing[key] = bad_value
    assert any(failure.startswith(f"{key}:") for failure in gate_failures(passing))


def test_public_batch_uses_exact_history_public_rollout_and_one_position_owner() -> None:
    config = load_config(CONFIG_PATH)
    with torch.no_grad():
        output = _run_public_batch(_handcrafted_rgbd_batch(), config)

    history = output["history"]
    trajectory = output["trajectory"]
    audit = output["correction_audit"]
    assert output["packet_count"] == 16
    assert output["predict_count"] == 1
    assert history.sample_mask.sum().item() == 16
    assert history.valid_mask.sum().item() == 16
    assert audit["calls"] == 1
    assert audit["position_fields"] == 0
    assert audit["position_change_max_abs"] == 0.0
    assert output["position_owner_count"] == 1
    assert output["direct_metric_position_owner_error"].max().item() == 0.0
    assert output["fixed_prior_max_abs"] <= DEFAULT_GATES.fixed_prior_max_abs
    assert output["identities"].unique().tolist() == [0]
    assert trajectory.positions.shape == (1, len(HORIZONS_SECONDS), 1, 3)
    assert trajectory.velocities.shape == trajectory.positions.shape
    assert trajectory.active_mask.all()
    torch.testing.assert_close(
        history.positions.permute(0, 2, 1, 3),
        output["standalone"].measured_positions,
    )


def test_full_batch_persistent_runtime_tensor_state_has_a_frozen_ceiling() -> None:
    config = load_config(CONFIG_PATH)
    with torch.no_grad():
        output = _run_public_batch(_handcrafted_rgbd_batch(batch_size=4), config)

    byte_count = output["runtime_tensor_bytes"]
    assert byte_count == 25_364
    assert byte_count <= DEFAULT_GATES.persistent_runtime_tensor_state_bytes
    metrics = _passing_metrics()
    metrics["persistent_runtime_tensor_state_bytes_max"] = float(byte_count)
    assert gate_failures(metrics) == []


def test_handcrafted_missing_depth_and_public_rgb_depth_vjps_fail_closed() -> None:
    config = load_config(CONFIG_PATH)
    batch = _handcrafted_rgbd_batch()
    missing = _missing_depth_metrics(config, batch)
    assert missing["missing_depth_last_measurement_valid_fraction"] == 0.0
    assert missing["missing_depth_fit_valid_fraction"] == 0.0
    assert missing["missing_depth_direct_velocity_calls"] == 0.0
    assert missing["missing_depth_history_sample_count_min"] == 16.0
    assert missing["missing_depth_history_valid_count_max"] == 15.0
    assert missing["missing_depth_finite_fraction"] == 1.0

    no_foreground = _no_foreground_metrics(config, batch)
    assert no_foreground["no_foreground_last_measurement_valid_fraction"] == 0.0
    assert no_foreground["no_foreground_fit_valid_fraction"] == 0.0
    assert no_foreground["no_foreground_direct_velocity_calls"] == 0.0
    assert no_foreground["no_foreground_history_sample_count_min"] == 16.0
    assert no_foreground["no_foreground_history_valid_count_max"] == 15.0
    assert no_foreground["no_foreground_finite_fraction"] == 1.0

    gradients = _gradient_metrics(config, batch)
    aggregate = {
        name: value for name, value in gradients.items() if name.startswith("gradient_l1/")
    }
    assert len(aggregate) == 24
    assert all(torch.isfinite(torch.tensor(value)) and value > 0.0 for value in aggregate.values())
    assert all(value <= DEFAULT_GATES.maximum_input_gradient_l1 for value in aggregate.values())
    for output_name in HISTORY_GRADIENT_TARGETS:
        for modality in ("rgb", "depth"):
            assert gradients[f"gradient_supported_history_frames/{output_name}/{modality}"] == 16.0
            assert (
                gradients[f"gradient_min_history_frame_l1/{output_name}/{modality}"]
                >= DEFAULT_GATES.minimum_history_frame_gradient_l1
            )


def test_one_zero_history_frame_is_visible_to_the_vjp_gate() -> None:
    gradient = torch.ones((1, 16, 1, 2, 2), dtype=torch.float32)
    gradient[:, 7].zero_()
    diagnostics = _history_gradient_diagnostics(
        gradient,
        output_name="current_velocity",
        modality="rgb",
    )
    assert diagnostics["gradient_min_history_frame_l1/current_velocity/rgb"] == 0.0
    assert diagnostics["gradient_supported_history_frames/current_velocity/rgb"] == 15.0
    metrics = _passing_metrics()
    metrics.update(diagnostics)
    failures = gate_failures(metrics)
    assert any(
        failure.startswith("gradient_min_history_frame_l1/current_velocity/rgb:")
        for failure in failures
    )
    assert any(
        failure.startswith("gradient_supported_history_frames/current_velocity/rgb:")
        for failure in failures
    )


def test_invalid_manifest_is_rejected_before_episode_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    calls: list[int] = []

    def forbidden_generator(*_args: object, **_kwargs: object) -> object:
        calls.append(1)
        raise AssertionError("episode generator must remain unopened")

    import world_model.simulator as simulator

    monkeypatch.setattr(simulator, "generate_episode", forbidden_generator)
    with pytest.raises(ValueError, match="exact frozen"):
        evaluate_seed_manifest(config, DEVELOPMENT_SEEDS[:-1], split="development")
    assert calls == []


def test_artifact_paths_reject_resolved_atomic_and_hardlink_aliases(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="artifact paths alias"):
        validate_distinct_paths(
            {"report": report, "collision": report.with_suffix(".json.tmp")},
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


def test_qualification_ledger_is_durable_ordered_and_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "qualification_access.json"
    ledger = QualificationLedger(path, {"protocol_sha256": "0" * 64})
    with pytest.raises(FileExistsError):
        QualificationLedger(path, {"protocol_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="must remain unopened"):
        ledger.begin_access("final_test")

    for split in ("selector", "confirmation", "final_test"):
        ledger.begin_access(split)
        with pytest.raises(RuntimeError, match="cannot be opened twice"):
            ledger.begin_access(split)
        ledger.complete_split(split, {"passed": True, "split": split})
    ledger.finish(passed=True, stopped_after="final_test")

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["protected_data_materialized"] is True
    assert all(record["splits"][split]["status"] == "passed" for split in ledger.ORDER)
    assert not path.with_suffix(".json.tmp").exists()
