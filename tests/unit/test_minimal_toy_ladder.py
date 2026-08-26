from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.run_minimal_toy_ladder import (
    _validate_output_paths,
    _write_report,
    _write_success_artifacts,
)
from world_model.training.checkpointing import load_model_weights
from world_model.training.minimal_toy import (
    ARCHITECTURE_VERSION,
    CONFIRMATION_SEEDS,
    FINAL_TEST_SEEDS,
    MEASUREMENT_UPDATES,
    REJECTED_V1_COMMIT,
    REJECTED_V1_REPORT_SHA256,
    ROLLOUT_UPDATES,
    SELECTOR_SEEDS,
    TRAIN_SEEDS,
    DifferentiableToyStateEstimator,
    _batch,
    _frame,
    _rollout_prediction,
    measurement_learning_rate,
    measurement_objective,
    photometric_solver_protocol,
    rollout_learning_rate,
    run_oracle_rung,
)
from world_model.utils.config import load_config

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "minimal_differentiable_toy_cpu.yaml"
)


def _config():
    return load_config(_CONFIG_PATH)


def test_minimal_toy_config_is_a_fixed_identifiable_cpu_unit() -> None:
    config = _config()

    assert config.device.preference == "cpu"
    assert config.device.cuda_amp is False
    assert config.project.seed == 0
    assert config.project.deterministic is True
    assert config.simulator.image_size == (48, 48)
    assert config.simulator.sequence_frames == 16
    assert config.simulator.min_objects == config.simulator.max_objects == 1
    assert config.simulator.gravity == (0.0, 0.0, 0.0)
    assert config.simulator.radius_range == (0.21, 0.21)
    assert config.simulator.mass_range == (1.0, 1.0)
    assert config.simulator.restitution_range == (0.7, 0.7)
    assert config.simulator.drag_range == (0.05, 0.05)
    assert config.simulator.friction_range == (0.2, 0.2)
    assert config.simulator.initial_speed_range == (0.1, 0.1)
    assert config.simulator.camera_motion == "fixed"
    assert config.simulator.render_noise_std == 0.0
    assert config.simulator.ensure_collision is False
    assert config.model.rgb.structured_disc_center_enabled is False
    assert config.training.rgb_pretrain_steps == MEASUREMENT_UPDATES
    assert config.training.steps == MEASUREMENT_UPDATES + ROLLOUT_UPDATES
    assert config.training.closed_loop_learning_rate_scale == 0.1
    assert measurement_learning_rate(config) == 0.002
    assert rollout_learning_rate(config) == 0.0002
    assert ARCHITECTURE_VERSION == 2
    assert tuple(range(8, 16)) == TRAIN_SEEDS
    assert tuple(range(100_008, 100_012)) == SELECTOR_SEEDS
    assert tuple(range(100_012, 100_016)) == CONFIRMATION_SEEDS
    assert tuple(range(200_008, 200_016)) == FINAL_TEST_SEEDS
    assert REJECTED_V1_COMMIT == "578c3770f49293d5390ae74ab5c73b4ebc50e9ca"
    assert REJECTED_V1_REPORT_SHA256 == (
        "a9cafaad5bd7fcabaebfdd815a89b6fe125284ca270a00b1a782def3bf683ce1"
    )
    solver = photometric_solver_protocol()
    assert solver["type"] == "four_stage_finite_difference_gauss_newton"
    assert solver["candidates_per_stage"] == 7
    assert solver["centre_trust_steps_pixels"] == [0.5, 0.25, 0.125, 0.0625]
    assert solver["damping"] == ("sqrt(dtype_epsilon)*mean(diag(J^T_J))+dtype_epsilon")
    assert solver["trust_transform"] == "componentwise_tanh"
    assert solver["residual_normalization"] == "sqrt(3*height*width)"
    assert solver["candidate_support"] == "candidate_independent_full_frame"
    assert solver["nuisance_albedo"] == "analytic_least_squares_clamped_0_1"


def test_oracle_rung_matches_simulator_and_has_a_real_velocity_gradient() -> None:
    config = _config()
    metrics = run_oracle_rung(config, (("unit", (8, 9)),))

    assert metrics["collision_free"] is True
    assert metrics["position_rmse_m"] < 1.0e-5
    assert metrics["velocity_rmse_mps"] < 1.0e-5
    assert metrics["velocity_gradient_norm"] > 1.0e-8


def test_minimal_rgb_measurement_and_rollout_reach_learned_parameters() -> None:
    config = _config()
    batch = _batch(config, (8, 9))
    model = DifferentiableToyStateEstimator(
        image_size=config.simulator.image_size,
        world_radius_m=config.simulator.radius_range[0],
    )

    frame = _frame(batch, 0)
    estimate = model(
        frame["image"],
        frame["world_from_camera"],
        frame["intrinsics"],
    )
    estimate.slot_mask_logits.retain_grad()
    estimate.geometry.centres.retain_grad()
    estimate.geometry.radius_pixels.retain_grad()
    measurement_loss, _ = measurement_objective(estimate, frame)
    measurement_loss.backward()

    assert model.mask_head.weight.grad is not None
    assert torch.isfinite(model.mask_head.weight.grad).all()
    assert model.mask_head.weight.grad.abs().sum() > 0
    assert estimate.slot_mask_logits.grad is not None
    assert estimate.slot_mask_logits.grad.abs().sum() > 0
    assert estimate.geometry.centres.grad is not None
    assert estimate.geometry.centres.grad.abs().sum() > 0
    assert estimate.geometry.radius_pixels.grad is not None
    assert estimate.geometry.radius_pixels.grad.abs().sum() > 0
    assert not hasattr(model, "radius_calibrator")
    assert estimate.photometric_radius.radius_pixels.grad_fn is not None

    model.zero_grad(set_to_none=True)
    prediction = _rollout_prediction(model, config, batch)
    target = batch["objects"]["position"][:, 10, :1]
    rollout_loss = (prediction - target).square().mean()
    rollout_loss.backward()

    assert model.mask_head.weight.grad is not None
    assert torch.isfinite(model.mask_head.weight.grad).all()
    assert model.mask_head.weight.grad.abs().sum() > 0
    assert not hasattr(model, "radius_calibrator")


def test_minimal_toy_checkpoint_is_atomic_project_compatible_weights_only(
    tmp_path: Path,
) -> None:
    config = _config()
    model = DifferentiableToyStateEstimator(
        image_size=config.simulator.image_size,
        world_radius_m=config.simulator.radius_range[0],
    )
    with torch.no_grad():
        model.mask_head.weight.fill_(0.125)
        model.mask_head.bias.fill_(3.75)
    expected_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "minimal.pt"
    report_path = tmp_path / "report.json"
    source_provenance = {
        "commit": "0" * 40,
        "dirty": False,
        "worktree_fingerprint": "1" * 64,
        "runtime_source_fingerprint": "2" * 64,
    }
    report = {
        "protocol": {"architecture_version": ARCHITECTURE_VERSION},
        "final_test": {"passed": True},
    }

    _write_success_artifacts(
        report_path=report_path,
        checkpoint_path=checkpoint,
        model=model,
        config=config,
        report=report,
        source_provenance=source_provenance,
    )

    assert checkpoint.is_file()
    assert report_path.is_file()
    assert not checkpoint.with_suffix(".pt.tmp").exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == MEASUREMENT_UPDATES + ROLLOUT_UPDATES
    assert payload["optimizer_state"] is None
    assert payload["scheduler_state"] is None
    assert payload["git"] == source_provenance
    assert payload["metrics"] == {
        "artifact_kind": "minimal_differentiable_toy_weights_only",
        "exact_resume": False,
        "protocol": report["protocol"],
        "final_test": report["final_test"],
    }
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["checkpoint"] == str(checkpoint.resolve())
    assert saved_report["checkpoint_kind"] == "project_compatible_weights_only"
    assert saved_report["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    reloaded = DifferentiableToyStateEstimator(
        image_size=config.simulator.image_size,
        world_radius_m=config.simulator.radius_range[0],
    )
    with torch.no_grad():
        reloaded.mask_head.weight.zero_()
        reloaded.mask_head.bias.zero_()
    loaded_payload = load_model_weights(
        checkpoint,
        model=reloaded,
        expected_config=config,
    )

    assert loaded_payload["weight_load_missing_keys"] == ()
    for key, expected in expected_state.items():
        torch.testing.assert_close(reloaded.state_dict()[key], expected, rtol=0.0, atol=0.0)


def test_minimal_toy_report_rejects_nonfinite_json(tmp_path: Path) -> None:
    report = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        _write_report(report, {"metric": float("nan")})

    assert not report.exists()


def test_minimal_toy_outputs_must_be_distinct_before_any_evidence_write(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "evidence"

    with pytest.raises(ValueError, match="must be distinct"):
        _validate_output_paths(shared, shared.parent / "." / shared.name)

    assert not shared.exists()
