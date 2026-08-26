from __future__ import annotations

from pathlib import Path

import torch

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
    assert solver["damping"] == (
        "sqrt(dtype_epsilon)*mean(diag(J^T_J))+dtype_epsilon"
    )
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
