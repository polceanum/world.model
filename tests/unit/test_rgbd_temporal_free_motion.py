"""Seed-free tests for the frozen RGB-D temporal free-motion rung."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import Tensor

from scripts import run_rgbd_temporal_free_motion_ladder as temporal_runner
from world_model.dynamics import fit_free_motion, free_motion_position_velocity
from world_model.simulator import (
    CameraFrame,
    SphereState,
    make_intrinsics,
    render_spheres,
)
from world_model.training.rgbd_temporal_free_motion import (
    CONFIRMATION_SEEDS,
    DEFAULT_GATES,
    DEVELOPMENT_SEEDS,
    FINAL_TEST_SEEDS,
    FROZEN_CONFIG_SHA256,
    HISTORY_FRAME_INDICES,
    HORIZONS_SECONDS,
    OPTIMIZER_UPDATES,
    SELECTOR_SEEDS,
    RGBDTemporalFreeMotionEstimator,
    gate_failures,
    temporal_protocol,
)
from world_model.utils.config import OrpheusConfig, load_config

IMAGE_SIZE = (64, 64)
WORLD_RADIUS_M = 0.21
DRAG = 0.05


def _camera() -> CameraFrame:
    identity = torch.eye(4, dtype=torch.float32)
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics(IMAGE_SIZE, 50.0),
        position=torch.zeros(3),
        target=torch.tensor([0.0, 0.0, 1.0]),
    )


def _sphere(position: Tensor) -> SphereState:
    return SphereState(
        object_id=torch.tensor([0], dtype=torch.int64),
        active=torch.ones(1, dtype=torch.bool),
        position=position.reshape(1, 3),
        velocity=torch.zeros((1, 3)),
        radius=torch.full((1, 1), WORLD_RADIUS_M),
        mass=torch.ones((1, 1)),
        restitution=torch.full((1, 1), 0.7),
        drag=torch.full((1, 1), DRAG),
        friction=torch.full((1, 1), 0.2),
        albedo=torch.tensor([[0.82, 0.23, 0.14]]),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        angular_velocity=torch.zeros((1, 3)),
        sleeping=torch.zeros(1, dtype=torch.bool),
        sleep_counter=torch.zeros(1, dtype=torch.int64),
    )


def _history() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    camera = _camera()
    timestamps = torch.arange(len(HISTORY_FRAME_INDICES), dtype=torch.float32) / 20.0
    anchor_time = timestamps[-1]
    anchor_position = torch.tensor([[[0.08, -0.06, 4.2]]], dtype=torch.float32)
    anchor_velocity = torch.tensor([[[0.030, 0.016, -0.012]]], dtype=torch.float32)
    positions = []
    rgb = []
    depth = []
    for timestamp in timestamps:
        position, _ = free_motion_position_velocity(
            anchor_position,
            anchor_velocity,
            timestamp - anchor_time,
            gravity=torch.zeros(3),
            drag=torch.full((1, 1), DRAG),
        )
        positions.append(position[0, 0])
        rendered = render_spheres(_sphere(position[0, 0]), camera, IMAGE_SIZE)
        rgb.append(rendered.rgb)
        depth.append(rendered.depth_buffer.unsqueeze(0))
    frames = len(HISTORY_FRAME_INDICES)
    return (
        torch.stack(rgb).unsqueeze(0),
        torch.stack(depth).unsqueeze(0),
        camera.world_from_camera.expand(1, frames, -1, -1).clone(),
        camera.intrinsics.expand(1, frames, -1, -1).clone(),
        timestamps.unsqueeze(0),
        torch.stack(positions).reshape(1, frames, 1, 3),
        anchor_velocity,
    )


def _estimator() -> RGBDTemporalFreeMotionEstimator:
    return RGBDTemporalFreeMotionEstimator(
        image_size=IMAGE_SIZE,
        world_radius_m=WORLD_RADIUS_M,
        gravity=(0.0, 0.0, 0.0),
        drag=DRAG,
    )


def test_protocol_manifests_are_fresh_disjoint_and_hash_bound() -> None:
    namespaces = (
        DEVELOPMENT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    flattened = [seed for namespace in namespaces for seed in namespace]
    assert len(flattened) == len(set(flattened))
    assert [namespace[0] // 1_000_000 for namespace in namespaces] == [41, 42, 43, 44]

    protocol = temporal_protocol()
    assert protocol["optimizer_updates"] == OPTIMIZER_UPDATES == 0
    assert protocol["learned_parameter_count"] == 0
    assert protocol["persistent_module_state_bytes"] == 0
    assert len(protocol["protocol_sha256"]) == 64
    assert protocol["uncertainty"]["calibrated_posterior_claim"] is False
    assert protocol["uncertainty"]["coverage_gate"] is None
    assert protocol["gradient_gate"]["kind"] == ("fixed_state_output_vector_jacobian_product")


def test_parameter_free_rgbd_history_recovers_state_and_analytic_rollout() -> None:
    images, depth, transforms, intrinsics, timestamps, true_positions, true_velocity = _history()
    estimator = _estimator()

    estimate = estimator(images, depth, transforms, intrinsics, timestamps)

    assert list(estimator.parameters()) == []
    assert list(estimator.buffers()) == []
    assert estimator.state_dict() == {}
    assert estimate.measurement_valid_mask.all()
    assert estimate.sequence_valid.all()
    assert estimate.fit.valid.all()
    assert estimate.fit.support_count.item() == len(HISTORY_FRAME_INDICES)
    torch.testing.assert_close(
        estimate.fit.position,
        true_positions[:, -1],
        rtol=0.0,
        atol=8.0e-3,
    )
    torch.testing.assert_close(
        estimate.fit.velocity,
        true_velocity,
        rtol=0.0,
        atol=1.0e-2,
    )
    assert estimate.rollout_positions.shape == (1, len(HORIZONS_SECONDS), 1, 3)
    assert estimate.rollout_velocities.shape == estimate.rollout_positions.shape
    position_error, velocity_error = estimator.semigroup_errors(
        estimate.fit.position,
        estimate.fit.velocity,
    )
    assert position_error.max().item() <= 1.0e-5
    assert velocity_error.max().item() <= 1.0e-5


def test_uniform_fit_does_not_use_confidence_or_validity_as_temporal_weights() -> None:
    images, depth, transforms, intrinsics, timestamps, _, _ = _history()
    estimator = _estimator()

    estimate = estimator(images, depth, transforms, intrinsics, timestamps)
    direct = fit_free_motion(
        estimate.measured_positions,
        timestamps,
        gravity=torch.zeros(3),
        drag=torch.full((1, 1), DRAG),
        anchor_time=timestamps[:, -1],
        minimum_support=len(HISTORY_FRAME_INDICES),
    )

    torch.testing.assert_close(estimate.fit.normal_matrix, direct.normal_matrix)
    torch.testing.assert_close(estimate.fit.support_weight, torch.full((1, 1), 16.0))


def test_every_state_kind_has_finite_nonzero_gradient_to_rgb_and_depth() -> None:
    images, depth, transforms, intrinsics, timestamps, _, _ = _history()
    images = images.clone().requires_grad_(True)
    depth = depth.clone().requires_grad_(True)
    estimator = _estimator()
    estimate = estimator(images, depth, transforms, intrinsics, timestamps)
    coefficients = estimate.fit.position.new_tensor((0.5, -0.75, 1.25))

    def probe(value: Tensor) -> Tensor:
        return (value * coefficients).mean()

    losses = {
        "current_position": probe(estimate.fit.position),
        "current_velocity": probe(estimate.fit.velocity),
    }
    for index, horizon in enumerate(HORIZONS_SECONDS):
        losses[f"horizon_{horizon:.2f}_position"] = probe(estimate.rollout_positions[:, index])
        losses[f"horizon_{horizon:.2f}_velocity"] = probe(estimate.rollout_velocities[:, index])

    for index, loss in enumerate(losses.values()):
        image_gradient, depth_gradient = torch.autograd.grad(
            loss,
            (images, depth),
            retain_graph=index + 1 < len(losses),
        )
        for gradient in (image_gradient, depth_gradient):
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum().item() > 0.0


def test_ols_covariance_is_finite_psd_and_not_claimed_as_calibrated() -> None:
    images, depth, transforms, intrinsics, timestamps, _, _ = _history()
    estimate = _estimator()(images, depth, transforms, intrinsics, timestamps)
    uncertainty = estimate.uncertainty

    for covariance in (
        uncertainty.noise_covariance,
        uncertainty.anchor_position_covariance,
        uncertainty.anchor_velocity_covariance,
        uncertainty.forecast_position_covariance,
        uncertainty.forecast_velocity_covariance,
    ):
        assert torch.isfinite(covariance).all()
        eigenvalues = torch.linalg.eigvalsh(covariance)
        assert eigenvalues.min().item() >= -1.0e-8
    sample_count = len(HISTORY_FRAME_INDICES)
    expected_coefficient_scale = torch.linalg.inv(estimate.fit.normal_matrix) / sample_count
    expected_noise = estimate.fit.residual_covariance * sample_count / (sample_count - 2)
    expected_velocity_covariance = (
        expected_coefficient_scale[..., 1, 1, None, None] * expected_noise
    )
    torch.testing.assert_close(
        uncertainty.anchor_velocity_covariance,
        expected_velocity_covariance,
    )
    assert temporal_protocol()["uncertainty"]["proper_score_gate"] is None


def test_missing_depth_fails_closed_and_rgb_only_is_never_implicit_fallback() -> None:
    images, depth, transforms, intrinsics, timestamps, _, _ = _history()
    estimator = _estimator()
    primary = estimator(images, depth, transforms, intrinsics, timestamps)
    missing = estimator(images, torch.zeros_like(depth), transforms, intrinsics, timestamps)
    rgb_only = estimator.rgb_only_ablation(
        primary.frame_measurement,
        transforms,
        intrinsics,
        timestamps,
    )

    assert not missing.measurement_valid_mask.any()
    assert not missing.sequence_valid.any()
    assert not missing.fit.valid.any()
    assert torch.count_nonzero(missing.measured_positions).item() == 0
    assert torch.count_nonzero(missing.fit.position).item() == 0
    assert torch.count_nonzero(missing.fit.velocity).item() == 0
    assert torch.count_nonzero(missing.rollout_positions).item() == 0
    assert torch.count_nonzero(missing.rollout_velocities).item() == 0
    assert rgb_only.fit.valid.all()
    assert not torch.equal(rgb_only.measured_positions, missing.measured_positions)


def test_one_missing_depth_frame_invalidates_the_complete_temporal_state() -> None:
    images, depth, transforms, intrinsics, timestamps, _, _ = _history()
    partial_depth = depth.clone()
    partial_depth[:, 7] = 0.0

    missing = _estimator()(images, partial_depth, transforms, intrinsics, timestamps)

    assert missing.measurement_valid_mask[:, 7].logical_not().all()
    assert not missing.sequence_valid.any()
    assert not missing.fit.valid.any()
    assert torch.count_nonzero(missing.fit.position).item() == 0
    assert torch.count_nonzero(missing.fit.velocity).item() == 0
    assert torch.count_nonzero(missing.rollout_positions).item() == 0
    assert torch.count_nonzero(missing.rollout_velocities).item() == 0


def _passing_mock_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {
        "measurement_valid_fraction": 1.0,
        "minimum_fit_support": 16.0,
        "covariance_minimum_eigenvalue": 0.0,
        "learned_parameter_count": 0.0,
        "learned_parameter_bytes": 0.0,
        "persistent_module_state_bytes": 0.0,
        "optimizer_updates": 0.0,
    }
    maximum_keys = (
        "oracle_position_rmse_m",
        "oracle_velocity_rmse_mps",
        "oracle_simulator_position_rmse_m",
        "oracle_simulator_velocity_rmse_mps",
        "measurement_position_rmse_m",
        "measurement_centre_rmse_pixels",
        "measurement_radius_relative_rmse",
        "current_position_rmse_m",
        "current_position_axis_rmse_m",
        "current_velocity_rmse_mps",
        "current_velocity_axis_rmse_mps",
        "horizon_velocity_rmse_mps",
        "horizon_velocity_axis_rmse_mps",
        "early_stationary_additive_regression_m",
        "long_stationary_rmse_ratio",
        "zero_velocity_rmse_ratio",
        "two_frame_velocity_rmse_ratio",
        "rgb_only_current_velocity_rmse_ratio",
        "rgb_only_two_second_position_rmse_ratio",
        "missing_depth_valid_fraction",
        "maximum_fit_condition_number",
        "residual_rmse_m",
        "semigroup_position_max_abs_m",
        "semigroup_velocity_max_abs_mps",
        "perception_latency_seconds",
        "state_only_rollout_latency_seconds",
        "process_max_rss_bytes",
        "process_rss_delta_bytes",
    )
    metrics.update({key: 0.0 for key in maximum_keys})
    for horizon in HORIZONS_SECONDS:
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = 0.0
        metrics[f"horizon_{label}_position_axis_rmse_m"] = 0.0
        metrics[f"horizon_{label}_velocity_rmse_mps"] = 0.0
        metrics[f"horizon_{label}_velocity_axis_rmse_mps"] = 0.0
    for loss_name in (
        "current_position",
        "current_velocity",
        *(f"horizon_{horizon:.2f}_position" for horizon in HORIZONS_SECONDS),
        *(f"horizon_{horizon:.2f}_velocity" for horizon in HORIZONS_SECONDS),
    ):
        metrics[f"gradient_l1/{loss_name}/rgb"] = 1.0
        metrics[f"gradient_l1/{loss_name}/depth"] = 1.0
    return metrics


def _frozen_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "rgbd_temporal_free_motion_cpu.yaml"


def _mock_clean_source() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "dirty": False,
        "worktree_fingerprint": "2" * 64,
        "runtime_source_fingerprint": "3" * 64,
    }


def _passing_development_evidence() -> tuple[dict[str, object], dict[str, object], str]:
    checkpoint_digest = "4" * 64
    development: dict[str, object] = {
        "split": "development",
        "seeds": list(DEVELOPMENT_SEEDS),
        "seed_manifest_sha256": temporal_runner._canonical_sha256(list(DEVELOPMENT_SEEDS)),
        "metrics": _passing_mock_metrics(),
        "failures": [],
        "passed": True,
        "optimizer_updates": 0,
        "uncertainty_claim": "iid_ols_residual_diagnostic_not_calibrated_posterior",
    }
    report: dict[str, object] = {
        "artifact_kind": "rgbd_temporal_free_motion_development",
        "protocol": temporal_protocol(),
        "source_provenance": _mock_clean_source(),
        "config_sha256": FROZEN_CONFIG_SHA256,
        "development": development,
        "checkpoint_sha256": checkpoint_digest,
        "optimizer_updates": 0,
        "protected_data_materialized": False,
        "passed": True,
        "review_ready": True,
        "stopped_after": "development",
    }
    return report, development, checkpoint_digest


def _replace_nested(mapping: dict[str, object], path: str, value: object) -> None:
    names = path.split(".")
    current = mapping
    for name in names[:-1]:
        child = current[name]
        assert isinstance(child, dict)
        current = child
    current[names[-1]] = value


def _passing_checkpoint_evidence(
    config: OrpheusConfig,
    development: dict[str, object],
) -> dict[str, object]:
    return {
        "step": 0,
        "optimizer_state": None,
        "scheduler_state": None,
        "git": _mock_clean_source(),
        "config": config.to_dict(),
        "metrics": {
            "artifact_kind": "rgbd_temporal_parameter_free_empty_state",
            "optimizer_updates": 0,
            "protocol": temporal_protocol(),
            "development": deepcopy(development),
        },
    }


def test_frozen_config_bytes_match_the_predeclared_hash() -> None:
    config = load_config(_frozen_config_path())
    assert config.evaluation.rgb_only
    assert hashlib.sha256(_frozen_config_path().read_bytes()).hexdigest() == FROZEN_CONFIG_SHA256


@pytest.mark.parametrize(
    ("paths", "atomic_writers"),
    (
        (
            {"report": Path("evidence.json"), "checkpoint": Path("evidence.json.tmp")},
            ("report", "checkpoint"),
        ),
        (
            {
                "report": Path("qualification.json"),
                "checkpoint": Path("model.pt"),
                "ledger": Path("qualification.json.tmp"),
            },
            ("report", "ledger"),
        ),
    ),
)
def test_artifact_paths_reject_atomic_temporary_aliases(
    paths: dict[str, Path],
    atomic_writers: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="artifact paths must be distinct"):
        temporal_runner._validate_distinct_paths(paths, atomic_writers=atomic_writers)


def test_complete_development_and_checkpoint_evidence_validates_without_access() -> None:
    report, development, checkpoint_digest = _passing_development_evidence()
    validated = temporal_runner._validate_development_evidence(
        report,
        checkpoint_digest=checkpoint_digest,
        clean_source=_mock_clean_source(),
    )
    assert validated == development

    config = load_config(_frozen_config_path())
    temporal_runner._validate_checkpoint_evidence(
        _passing_checkpoint_evidence(config, development),
        development=development,
        clean_source=_mock_clean_source(),
        expected_config=config,
    )


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    (
        ("artifact_kind", "not_development"),
        ("protected_data_materialized", True),
        ("optimizer_updates", 1),
        ("stopped_after", "selector"),
        ("checkpoint_sha256", "5" * 64),
        ("development.split", "selector"),
        ("development.seeds", list(DEVELOPMENT_SEEDS[:-1])),
        ("development.seed_manifest_sha256", "6" * 64),
        ("development.optimizer_updates", 1),
        ("development.uncertainty_claim", "calibrated_gaussian"),
        ("development.failures", ["fabricated"]),
        ("development.passed", False),
        ("development.metrics.current_velocity_rmse_mps", None),
    ),
)
def test_reviewed_development_evidence_rejects_truncation_and_fabrication(
    path: str,
    invalid_value: object,
) -> None:
    report, _, checkpoint_digest = _passing_development_evidence()
    _replace_nested(report, path, invalid_value)

    with pytest.raises(ValueError):
        temporal_runner._validate_development_evidence(
            report,
            checkpoint_digest=checkpoint_digest,
            clean_source=_mock_clean_source(),
        )


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    (
        ("step", 1),
        ("optimizer_state", {}),
        ("scheduler_state", {}),
        ("git.commit", "7" * 40),
        ("config.project.name", "wrong-rung"),
        ("metrics.artifact_kind", "learned_state"),
        ("metrics.optimizer_updates", 1),
        ("metrics.protocol.name", "wrong_protocol"),
        ("metrics.development.passed", False),
    ),
)
def test_checkpoint_evidence_rejects_state_or_binding_changes(
    path: str,
    invalid_value: object,
) -> None:
    _, development, _ = _passing_development_evidence()
    config = load_config(_frozen_config_path())
    payload = _passing_checkpoint_evidence(config, development)
    _replace_nested(payload, path, invalid_value)

    with pytest.raises(ValueError):
        temporal_runner._validate_checkpoint_evidence(
            payload,
            development=development,
            clean_source=_mock_clean_source(),
            expected_config=config,
        )


def test_qualification_ledger_is_exclusive_ordered_and_durable(tmp_path: Path) -> None:
    ledger_path = tmp_path / "qualification_access.json"
    ledger = temporal_runner._QualificationLedger(ledger_path, {"attempt": 1})

    with pytest.raises(FileExistsError):
        temporal_runner._QualificationLedger(ledger_path, {"attempt": 1})
    with pytest.raises(RuntimeError, match="expected 'selector'"):
        ledger.record_access("confirmation")

    for split in ("selector", "confirmation", "final_test"):
        ledger.record_access(split)
    ledger.finish({"passed": True, "stopped_after": "final_test"})

    durable = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert durable["access_started"] == {
        "selector": True,
        "confirmation": True,
        "final_test": True,
    }
    assert durable["status"] == "complete"
    assert durable["protected_data_materialized"] is True


def test_scalar_gates_are_fail_fast_and_mockable_without_episode_access() -> None:
    metrics = _passing_mock_metrics()
    assert gate_failures(metrics) == []

    metrics["current_velocity_rmse_mps"] = DEFAULT_GATES.current_velocity_rmse_mps * 1.01
    failures = gate_failures(metrics)
    assert any(failure.startswith("current_velocity_rmse_mps:") for failure in failures)


@pytest.mark.parametrize("bad_drag", [-0.1, math.inf])
def test_estimator_rejects_invalid_declared_drag(bad_drag: float) -> None:
    with pytest.raises(ValueError, match="drag"):
        RGBDTemporalFreeMotionEstimator(
            image_size=IMAGE_SIZE,
            world_radius_m=WORLD_RADIUS_M,
            gravity=(0.0, 0.0, 0.0),
            drag=bad_drag,
        )
