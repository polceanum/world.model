"""Fail-fast differentiable convergence unit for the one-sphere toy world.

This module is intentionally narrower than the general trainer.  It proves a
cheap, ordinary-autograd path from RGB pixels through soft disc geometry and a
calibrated pinhole state estimate into the deployed analytic free-motion
equations.  Fixed seeds and gates live here so a failed rung cannot be hidden
by continuing to tune later stages or by repeatedly inspecting the final set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from world_model.belief import BeliefFactory
from world_model.datasets.collate import collate_episodes
from world_model.dynamics import AnalyticKinematics
from world_model.observations.rgb.projector import backproject_rgb_measurements
from world_model.observations.rgb.soft_geometry import (
    SoftDiscGeometryOutput,
    soft_disc_geometry_from_rgb,
)
from world_model.simulator.episode import Episode, generate_episode
from world_model.utils.config import OrpheusConfig
from world_model.utils.seeds import seed_everything

TRAIN_SEEDS = tuple(range(8))
SELECTOR_SEEDS = tuple(range(100_000, 100_004))
CONFIRMATION_SEEDS = tuple(range(100_004, 100_008))
FINAL_TEST_SEEDS = tuple(range(200_000, 200_008))

MEASUREMENT_UPDATES = 60
ROLLOUT_UPDATES = 12
HISTORY_START_FRAME = 0
HISTORY_END_FRAME = 8
ROLLOUT_TARGET_FRAME = 10


@dataclass(frozen=True)
class MinimalToyGates:
    """Predeclared pass/fail limits for the convergence ladder."""

    oracle_position_rmse_m: float = 1.0e-5
    oracle_velocity_rmse_mps: float = 1.0e-5
    measurement_world_rmse_m: float = 0.05
    measurement_centre_rmse_pixels: float = 0.5
    measurement_radius_relative_rmse: float = 0.02
    measurement_valid_fraction: float = 1.0
    rollout_world_rmse_m: float = 0.05
    minimum_gradient_norm: float = 1.0e-8


DEFAULT_GATES = MinimalToyGates()


class ConvergenceGateError(RuntimeError):
    """A ladder rung failed and later rungs must not run."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


@dataclass(frozen=True)
class ToyStateEstimate:
    """Differentiable one-slot metric state produced from RGB only."""

    world_position: Tensor
    centres: Tensor
    radius_pixels: Tensor
    slot_mask_logits: Tensor
    geometry: SoftDiscGeometryOutput


class DifferentiableToyStateEstimator(nn.Module):
    """Minimal learned RGB owner around differentiable geometric moments.

    The per-pixel head softly reweights observable foreground evidence.  A
    bounded feature-conditioned radius calibration accounts for antialiasing
    and shading bias before the known physical sphere radius supplies metric
    depth.  There is no connected component, CPU assignment, detached forward
    replacement, or straight-through estimator in this path.
    """

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        world_radius_m: float,
        foreground_threshold: float = 0.04,
        foreground_temperature: float = 0.01,
        minimum_mass: float = 4.0,
    ) -> None:
        super().__init__()
        if len(image_size) != 2 or min(image_size) < 2:
            raise ValueError("image_size must contain two dimensions of at least two pixels")
        if not math.isfinite(world_radius_m) or world_radius_m <= 0.0:
            raise ValueError("world_radius_m must be finite and positive")
        self.image_size = tuple(int(value) for value in image_size)
        self.world_radius_m = float(world_radius_m)
        self.foreground_threshold = float(foreground_threshold)
        self.foreground_temperature = float(foreground_temperature)
        self.minimum_mass = float(minimum_mass)

        # A high initial bias preserves the single-object foreground prior.
        # Zero weights still receive an ordinary, nonzero pixel-conditioned
        # gradient on the first update.
        self.mask_head = nn.Conv2d(3, 1, kernel_size=1)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, 4.0)

        # Features: log radius, continuous confidence, normalized centre, and
        # the three foreground-weighted colour channels.
        self.radius_calibrator = nn.Linear(7, 1)
        nn.init.zeros_(self.radius_calibrator.weight)
        nn.init.zeros_(self.radius_calibrator.bias)

    def forward(
        self,
        image: Tensor,
        world_from_camera: Tensor,
        intrinsics: Tensor,
    ) -> ToyStateEstimate:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B,3,H,W]")
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected image size {self.image_size}, got {tuple(image.shape[-2:])}"
            )
        batch = image.shape[0]
        if world_from_camera.shape != (batch, 4, 4):
            raise ValueError("world_from_camera must have shape [B,4,4]")
        if intrinsics.shape != (batch, 3, 3):
            raise ValueError("intrinsics must have shape [B,3,3]")

        slot_mask_logits = self.mask_head(image)
        geometry = soft_disc_geometry_from_rgb(
            image,
            slot_mask_logits,
            foreground_threshold=self.foreground_threshold,
            foreground_temperature=self.foreground_temperature,
            minimum_mass=self.minimum_mass,
        )
        effective_mask = geometry.effective_masks
        safe_mass = geometry.mass.clamp_min(1.0e-8)
        weighted_colour = torch.einsum(
            "bshw,bchw->bsc",
            effective_mask,
            image,
        ) / safe_mass.unsqueeze(-1)
        radius_features = torch.cat(
            (
                geometry.radius_pixels.clamp_min(1.0e-6).log().unsqueeze(-1),
                geometry.confidence.unsqueeze(-1),
                geometry.centres,
                weighted_colour,
            ),
            dim=-1,
        )
        # The calibration is identity at initialization and cannot escape the
        # physically plausible +/-15% antialiasing correction range.
        raw_correction = self.radius_calibrator(radius_features).squeeze(-1)
        log_radius_correction = 0.15 * torch.tanh(raw_correction / 0.15)
        radius_pixels = geometry.radius_pixels * log_radius_correction.exp()

        focal_pixels = 0.5 * (intrinsics[:, 0, 0] + intrinsics[:, 1, 1])
        inverse_depth = radius_pixels / (
            focal_pixels[:, None] * self.world_radius_m
        ).clamp_min(1.0e-8)
        normalised_radius = radius_pixels / (0.5 * min(self.image_size))
        values = torch.cat(
            (
                geometry.centres,
                normalised_radius.clamp_min(1.0e-8).log().unsqueeze(-1),
                inverse_depth.unsqueeze(-1),
                image.new_zeros((batch, 1, 3)),
            ),
            dim=-1,
        )
        world_position = backproject_rgb_measurements(
            values,
            world_from_camera,
            intrinsics,
            self.image_size,
        )
        return ToyStateEstimate(
            world_position=world_position,
            centres=geometry.centres,
            radius_pixels=radius_pixels,
            slot_mask_logits=slot_mask_logits,
            geometry=geometry,
        )


def _episodes(config: OrpheusConfig, seeds: Sequence[int]) -> list[Episode]:
    return [generate_episode(config, int(seed)) for seed in seeds]


def _batch(config: OrpheusConfig, seeds: Sequence[int]) -> dict[str, Any]:
    return collate_episodes(_episodes(config, seeds))


def _assert_minimal_contract(config: OrpheusConfig) -> None:
    simulator = config.simulator
    fixed_ranges = {
        "radius_range": (0.21, 0.21),
        "mass_range": (1.0, 1.0),
        "restitution_range": (0.7, 0.7),
        "drag_range": (0.05, 0.05),
        "friction_range": (0.2, 0.2),
        "initial_speed_range": (0.1, 0.1),
    }
    required = {
        "image_size": (48, 48),
        "frame_rate": 20,
        "physics_rate": 120,
        "sequence_frames": 16,
        "min_objects": 1,
        "max_objects": 1,
        "gravity": (0.0, 0.0, 0.0),
        "camera_motion": "fixed",
        "render_noise_std": 0.0,
        "ensure_collision": False,
        "scenario_mixture": ("baseline",),
    }
    for name, expected in {**required, **fixed_ranges}.items():
        actual = getattr(simulator, name)
        if actual != expected:
            raise ValueError(
                f"minimal convergence config requires simulator.{name}={expected!r}, "
                f"got {actual!r}"
            )
    if config.device.preference != "cpu" or config.device.cuda_amp:
        raise ValueError("minimal convergence ladder requires CPU float32 without AMP")
    if config.project.seed != 0 or not config.project.deterministic:
        raise ValueError("minimal convergence ladder requires deterministic project seed 0")
    if config.training.rgb_pretrain_steps != MEASUREMENT_UPDATES:
        raise ValueError(f"minimal ladder requires {MEASUREMENT_UPDATES} measurement updates")
    if config.training.steps != MEASUREMENT_UPDATES + ROLLOUT_UPDATES:
        raise ValueError(
            f"minimal ladder requires exactly {MEASUREMENT_UPDATES + ROLLOUT_UPDATES} updates"
        )
    if config.training.closed_loop_learning_rate_scale != 0.1:
        raise ValueError("minimal ladder requires closed_loop_learning_rate_scale=0.1")


def _assert_collision_free(batch: Mapping[str, Any], seeds: Sequence[int]) -> None:
    collision = batch["events"]["collision"]
    contact = batch["events"]["contact"]
    external_impulse = batch["events"]["external_impulse"]
    if bool(collision.any()) or bool(contact.any()) or bool(external_impulse.ne(0).any()):
        raise RuntimeError(
            "minimal toy seed manifest is not free of contacts, collisions, and interventions: "
            f"{tuple(int(seed) for seed in seeds)}"
        )


def _frame(batch: Mapping[str, Any], frame_index: int) -> dict[str, Tensor]:
    return {
        "image": batch["rgb"][:, frame_index],
        "world_from_camera": batch["camera"]["world_from_camera"][:, frame_index],
        "intrinsics": batch["camera"]["intrinsics"][:, frame_index],
        "position": batch["objects"]["position"][:, frame_index, :1],
        "centre": batch["labels"]["projected_center"][:, frame_index, :1],
        "radius_pixels": batch["labels"]["apparent_radius"][:, frame_index, :1],
        "mask": batch["labels"]["segmentation_mask"][:, frame_index, :1],
    }


def _all_frames(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    batch_size, frame_count = batch["rgb"].shape[:2]

    def flatten(value: Tensor) -> Tensor:
        return value.reshape(batch_size * frame_count, *value.shape[2:])

    return {
        "image": flatten(batch["rgb"]),
        "world_from_camera": flatten(batch["camera"]["world_from_camera"]),
        "intrinsics": flatten(batch["camera"]["intrinsics"]),
        "position": flatten(batch["objects"]["position"][:, :, :1]),
        "centre": flatten(batch["labels"]["projected_center"][:, :, :1]),
        "radius_pixels": flatten(batch["labels"]["apparent_radius"][:, :, :1]),
        "mask": flatten(batch["labels"]["segmentation_mask"][:, :, :1]),
    }


def _balanced_mask_loss(probability: Tensor, target: Tensor) -> Tensor:
    if probability.shape != target.shape:
        raise ValueError("mask probability and target must share shape")
    target_bool = target.bool()
    probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
    positive = -probability.log().masked_select(target_bool).mean()
    negative = -(1.0 - probability).log().masked_select(~target_bool).mean()
    return 0.5 * (positive + negative)


def measurement_objective(
    estimate: ToyStateEstimate,
    frame: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Directly supervise the continuous RGB state owner."""

    centre = F.mse_loss(estimate.centres, frame["centre"])
    radius = F.mse_loss(
        estimate.radius_pixels.clamp_min(1.0e-6).log(),
        frame["radius_pixels"].clamp_min(1.0e-6).log(),
    )
    position = F.mse_loss(estimate.world_position, frame["position"])
    mask = _balanced_mask_loss(
        estimate.geometry.effective_masks,
        frame["mask"].to(dtype=estimate.geometry.effective_masks.dtype),
    )
    total = 8.0 * centre + 8.0 * radius + position + 0.05 * mask
    return total, {
        "measurement_total": total,
        "centre_mse": centre,
        "log_radius_mse": radius,
        "position_mse": position,
        "mask_bce": mask,
    }


def measurement_learning_rate(config: OrpheusConfig) -> float:
    """Return the fixed direct RGB-state learning rate."""

    return float(config.training.learning_rate)


def rollout_learning_rate(config: OrpheusConfig) -> float:
    """Return the fixed, smaller end-to-end rollout learning rate."""

    return float(
        config.training.learning_rate
        * config.training.closed_loop_learning_rate_scale
    )


def _belief_objects(
    config: OrpheusConfig,
    position: Tensor,
    velocity: Tensor,
) -> tuple[Any, Tensor]:
    batch = position.shape[0]
    simulator = config.simulator
    factory = BeliefFactory(
        max_objects=1,
        geometry_dim=1,
        appearance_dim=1,
        residual_dynamics_dim=1,
        modal_count=0,
        modal_dim=1,
        parameter_memory_dim=1,
        global_code_dim=1,
        initial_radius=simulator.radius_range[0],
        initial_mass=simulator.mass_range[0],
        initial_restitution=simulator.restitution_range[0],
        initial_drag=simulator.drag_range[0],
        initial_friction=simulator.friction_range[0],
    )
    belief = factory.create(
        batch_size=batch,
        device=position.device,
        dtype=position.dtype,
        gravity=simulator.gravity,
    )
    objects = belief.objects.replace(
        active=torch.ones((batch, 1), device=position.device, dtype=torch.bool),
        object_id=torch.zeros((batch, 1), device=position.device, dtype=torch.int64),
        position=position,
        velocity=velocity,
    )
    return objects, belief.gravity


def _analytic_positions(
    config: OrpheusConfig,
    position: Tensor,
    velocity: Tensor,
    dt: float,
) -> Tensor:
    objects, gravity = _belief_objects(config, position, velocity)
    return AnalyticKinematics()(objects, gravity, dt).position


def _rollout_prediction(
    model: DifferentiableToyStateEstimator,
    config: OrpheusConfig,
    batch: Mapping[str, Any],
) -> Tensor:
    first = _frame(batch, HISTORY_START_FRAME)
    history = _frame(batch, HISTORY_END_FRAME)
    first_estimate = model(
        first["image"],
        first["world_from_camera"],
        first["intrinsics"],
    )
    history_estimate = model(
        history["image"],
        history["world_from_camera"],
        history["intrinsics"],
    )
    frame_dt = 1.0 / config.simulator.frame_rate
    history_dt = (HISTORY_END_FRAME - HISTORY_START_FRAME) * frame_dt
    rollout_dt = (ROLLOUT_TARGET_FRAME - HISTORY_END_FRAME) * frame_dt
    drag = config.simulator.drag_range[0]
    displacement_scale = -math.expm1(-drag * history_dt) / drag
    initial_velocity = (
        history_estimate.world_position - first_estimate.world_position
    ) / displacement_scale
    history_velocity = initial_velocity * math.exp(-drag * history_dt)
    return _analytic_positions(
        config,
        history_estimate.world_position,
        history_velocity,
        rollout_dt,
    )


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def run_oracle_rung(
    config: OrpheusConfig,
    manifests: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, float | bool]:
    """Compare exact oracle state against the project's analytic equations."""

    position_errors: list[Tensor] = []
    velocity_errors: list[Tensor] = []
    all_batches: list[Mapping[str, Any]] = []
    for _, seeds in manifests:
        batch = _batch(config, seeds)
        _assert_collision_free(batch, seeds)
        all_batches.append(batch)
        start_position = batch["objects"]["position"][:, 0, :1]
        start_velocity = batch["objects"]["velocity"][:, 0, :1]
        target_frame = 2
        predicted_position = _analytic_positions(
            config,
            start_position,
            start_velocity,
            target_frame / config.simulator.frame_rate,
        )
        predicted_objects, gravity = _belief_objects(config, start_position, start_velocity)
        predicted_velocity = AnalyticKinematics()(
            predicted_objects,
            gravity,
            target_frame / config.simulator.frame_rate,
        ).velocity
        position_errors.append(
            (predicted_position - batch["objects"]["position"][:, target_frame, :1]).square()
        )
        velocity_errors.append(
            (predicted_velocity - batch["objects"]["velocity"][:, target_frame, :1]).square()
        )

    gradient_batch = all_batches[0]
    velocity = (
        gradient_batch["objects"]["velocity"][:, 0, :1]
        .detach()
        .clone()
        .requires_grad_(True)
    )
    position = gradient_batch["objects"]["position"][:, 0, :1]
    differentiable_position = _analytic_positions(config, position, velocity, 0.1)
    differentiable_position.sum().backward()
    gradient_norm = float(torch.linalg.vector_norm(velocity.grad))
    return {
        "position_rmse_m": float(torch.cat(position_errors).mean().sqrt()),
        "velocity_rmse_mps": float(torch.cat(velocity_errors).mean().sqrt()),
        "velocity_gradient_norm": gradient_norm,
        "collision_free": True,
    }


@torch.no_grad()
def measurement_metrics(
    model: DifferentiableToyStateEstimator,
    batch: Mapping[str, Any],
) -> dict[str, float]:
    frame = _all_frames(batch)
    estimate = model(
        frame["image"],
        frame["world_from_camera"],
        frame["intrinsics"],
    )
    pixel_scale = torch.tensor(
        (
            0.5 * (model.image_size[1] - 1),
            0.5 * (model.image_size[0] - 1),
        ),
        device=estimate.centres.device,
        dtype=estimate.centres.dtype,
    )
    relative_radius_error = (
        estimate.radius_pixels - frame["radius_pixels"]
    ) / frame["radius_pixels"].clamp_min(1.0e-6)
    return {
        "world_position_rmse_m": float(
            (estimate.world_position - frame["position"]).square().mean().sqrt()
        ),
        "centre_rmse_pixels": float(
            ((estimate.centres - frame["centre"]) * pixel_scale)
            .square()
            .mean()
            .sqrt()
        ),
        "radius_relative_rmse": float(relative_radius_error.square().mean().sqrt()),
        "minimum_confidence": float(estimate.geometry.confidence.min()),
        "minimum_foreground_mass": float(estimate.geometry.mass.min()),
        "valid_fraction": float(estimate.geometry.valid_mask.to(torch.float32).mean()),
    }


def train_measurement_rung(
    model: DifferentiableToyStateEstimator,
    config: OrpheusConfig,
    batch: Mapping[str, Any],
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=measurement_learning_rate(config),
        weight_decay=0.0,
    )
    final_loss = math.nan
    minimum_gradient_norm = math.inf
    model.train()
    for step in range(MEASUREMENT_UPDATES):
        frame = _frame(batch, step % config.simulator.sequence_frames)
        optimizer.zero_grad(set_to_none=True)
        estimate = model(
            frame["image"],
            frame["world_from_camera"],
            frame["intrinsics"],
        )
        loss, _ = measurement_objective(estimate, frame)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite measurement loss at update {step}")
        loss.backward()
        gradient_norm = _gradient_norm(tuple(model.parameters()))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise FloatingPointError(f"invalid measurement gradient at update {step}")
        minimum_gradient_norm = min(minimum_gradient_norm, gradient_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        final_loss = float(loss.detach())
    return {
        "updates": float(MEASUREMENT_UPDATES),
        "learning_rate": measurement_learning_rate(config),
        "final_loss": final_loss,
        "minimum_gradient_norm": minimum_gradient_norm,
    }


@torch.no_grad()
def rollout_metrics(
    model: DifferentiableToyStateEstimator,
    config: OrpheusConfig,
    batch: Mapping[str, Any],
) -> dict[str, float]:
    prediction = _rollout_prediction(model, config, batch)
    target = batch["objects"]["position"][:, ROLLOUT_TARGET_FRAME, :1]
    return {
        "world_position_rmse_m": float((prediction - target).square().mean().sqrt()),
    }


def train_rollout_rung(
    model: DifferentiableToyStateEstimator,
    config: OrpheusConfig,
    batch: Mapping[str, Any],
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=rollout_learning_rate(config),
        weight_decay=0.0,
    )
    final_loss = math.nan
    minimum_gradient_norm = math.inf
    model.train()
    for step in range(ROLLOUT_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        prediction = _rollout_prediction(model, config, batch)
        target = batch["objects"]["position"][:, ROLLOUT_TARGET_FRAME, :1]
        rollout_loss = F.mse_loss(prediction, target)
        # Preserve the already-qualified measurement owner while end-to-end
        # rollout gradients refine its temporal consistency.
        frame = _frame(batch, (step * 3) % config.simulator.sequence_frames)
        estimate = model(
            frame["image"],
            frame["world_from_camera"],
            frame["intrinsics"],
        )
        measurement_loss, _ = measurement_objective(estimate, frame)
        loss = 4.0 * rollout_loss + measurement_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite rollout loss at update {step}")
        loss.backward()
        gradient_norm = _gradient_norm(tuple(model.parameters()))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise FloatingPointError(f"invalid rollout gradient at update {step}")
        minimum_gradient_norm = min(minimum_gradient_norm, gradient_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        final_loss = float(loss.detach())
    return {
        "updates": float(ROLLOUT_UPDATES),
        "learning_rate": rollout_learning_rate(config),
        "final_loss": final_loss,
        "minimum_gradient_norm": minimum_gradient_norm,
    }


def _measurement_gate(metrics: Mapping[str, float], gates: MinimalToyGates) -> bool:
    return (
        metrics["world_position_rmse_m"] <= gates.measurement_world_rmse_m
        and metrics["centre_rmse_pixels"] <= gates.measurement_centre_rmse_pixels
        and metrics["radius_relative_rmse"] <= gates.measurement_radius_relative_rmse
        and metrics["valid_fraction"] >= gates.measurement_valid_fraction
    )


def _rollout_gate(metrics: Mapping[str, float], gates: MinimalToyGates) -> bool:
    return metrics["world_position_rmse_m"] <= gates.rollout_world_rmse_m


def run_minimal_toy_ladder(
    config: OrpheusConfig,
    *,
    gates: MinimalToyGates = DEFAULT_GATES,
) -> tuple[DifferentiableToyStateEstimator, dict[str, Any]]:
    """Run all three rungs once, stopping before any later failed stage."""

    _assert_minimal_contract(config)
    seed_everything(config.project.seed, deterministic=True)
    torch.set_num_threads(1)
    report: dict[str, Any] = {
        "protocol": {
            "train_seeds": list(TRAIN_SEEDS),
            "selector_seeds": list(SELECTOR_SEEDS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "final_test_seeds": list(FINAL_TEST_SEEDS),
            "measurement_updates": MEASUREMENT_UPDATES,
            "rollout_updates": ROLLOUT_UPDATES,
            "gates": asdict(gates),
            "final_test_policy": "one shot after selector and confirmation pass",
        },
        "rungs": {},
    }

    oracle_metrics = run_oracle_rung(
        config,
        (
            ("train", TRAIN_SEEDS),
            ("selector", SELECTOR_SEEDS),
            ("confirmation", CONFIRMATION_SEEDS),
        ),
    )
    oracle_passed = (
        oracle_metrics["position_rmse_m"] <= gates.oracle_position_rmse_m
        and oracle_metrics["velocity_rmse_mps"] <= gates.oracle_velocity_rmse_mps
        and oracle_metrics["velocity_gradient_norm"] >= gates.minimum_gradient_norm
    )
    report["rungs"]["A_oracle_equations"] = {
        "passed": oracle_passed,
        "metrics": oracle_metrics,
    }
    if not oracle_passed:
        report["passed"] = False
        report["stopped_after"] = "A_oracle_equations"
        raise ConvergenceGateError("oracle equation rung failed", report)

    train_batch = _batch(config, TRAIN_SEEDS)
    selector_batch = _batch(config, SELECTOR_SEEDS)
    confirmation_batch = _batch(config, CONFIRMATION_SEEDS)
    for seeds, batch in (
        (TRAIN_SEEDS, train_batch),
        (SELECTOR_SEEDS, selector_batch),
        (CONFIRMATION_SEEDS, confirmation_batch),
    ):
        _assert_collision_free(batch, seeds)
    model = DifferentiableToyStateEstimator(
        image_size=tuple(config.simulator.image_size),
        world_radius_m=config.simulator.radius_range[0],
    )
    initial_selector_measurement = measurement_metrics(model, selector_batch)
    initial_confirmation_measurement = measurement_metrics(model, confirmation_batch)
    initial_selector_rollout = rollout_metrics(model, config, selector_batch)
    initial_confirmation_rollout = rollout_metrics(model, config, confirmation_batch)
    measurement_training = train_measurement_rung(model, config, train_batch)
    selector_measurement = measurement_metrics(model, selector_batch)
    confirmation_measurement = measurement_metrics(model, confirmation_batch)
    measurement_passed = (
        measurement_training["minimum_gradient_norm"] >= gates.minimum_gradient_norm
        and _measurement_gate(selector_measurement, gates)
        and _measurement_gate(confirmation_measurement, gates)
    )
    report["rungs"]["B_rgb_measurement"] = {
        "passed": measurement_passed,
        "training": measurement_training,
        "before_training": {
            "selector_measurement": initial_selector_measurement,
            "confirmation_measurement": initial_confirmation_measurement,
            "selector_rollout": initial_selector_rollout,
            "confirmation_rollout": initial_confirmation_rollout,
        },
        "selector": selector_measurement,
        "confirmation": confirmation_measurement,
    }
    if not measurement_passed:
        report["passed"] = False
        report["stopped_after"] = "B_rgb_measurement"
        raise ConvergenceGateError("RGB measurement rung failed", report)

    selector_rollout_before = rollout_metrics(model, config, selector_batch)
    confirmation_rollout_before = rollout_metrics(model, config, confirmation_batch)
    rollout_training = train_rollout_rung(model, config, train_batch)
    selector_rollout = rollout_metrics(model, config, selector_batch)
    confirmation_rollout = rollout_metrics(model, config, confirmation_batch)
    # Rung C cannot silently sacrifice the state gate it consumes.
    selector_measurement_after = measurement_metrics(model, selector_batch)
    confirmation_measurement_after = measurement_metrics(model, confirmation_batch)
    rollout_passed = (
        rollout_training["minimum_gradient_norm"] >= gates.minimum_gradient_norm
        and _rollout_gate(selector_rollout, gates)
        and _rollout_gate(confirmation_rollout, gates)
        and _measurement_gate(selector_measurement_after, gates)
        and _measurement_gate(confirmation_measurement_after, gates)
    )
    report["rungs"]["C_rgb_to_rollout"] = {
        "passed": rollout_passed,
        "training": rollout_training,
        "before_training": {
            "selector_rollout": selector_rollout_before,
            "confirmation_rollout": confirmation_rollout_before,
            "selector_measurement": selector_measurement,
            "confirmation_measurement": confirmation_measurement,
        },
        "selector_rollout": selector_rollout,
        "confirmation_rollout": confirmation_rollout,
        "selector_measurement": selector_measurement_after,
        "confirmation_measurement": confirmation_measurement_after,
    }
    if not rollout_passed:
        report["passed"] = False
        report["stopped_after"] = "C_rgb_to_rollout"
        raise ConvergenceGateError("RGB-to-rollout rung failed", report)

    # The final manifest is materialized and inspected exactly once, after all
    # development-facing gates have passed.
    final_batch = _batch(config, FINAL_TEST_SEEDS)
    _assert_collision_free(final_batch, FINAL_TEST_SEEDS)
    final_measurement = measurement_metrics(model, final_batch)
    final_rollout = rollout_metrics(model, config, final_batch)
    final_passed = _measurement_gate(final_measurement, gates) and _rollout_gate(
        final_rollout,
        gates,
    )
    report["final_test"] = {
        "passed": final_passed,
        "measurement": final_measurement,
        "rollout": final_rollout,
    }
    report["passed"] = final_passed
    report["stopped_after"] = "complete" if final_passed else "final_test"
    if not final_passed:
        raise ConvergenceGateError("one-shot final test failed", report)
    return model, report


__all__ = [
    "CONFIRMATION_SEEDS",
    "ConvergenceGateError",
    "DEFAULT_GATES",
    "DifferentiableToyStateEstimator",
    "FINAL_TEST_SEEDS",
    "MEASUREMENT_UPDATES",
    "MinimalToyGates",
    "ROLLOUT_UPDATES",
    "SELECTOR_SEEDS",
    "TRAIN_SEEDS",
    "ToyStateEstimate",
    "measurement_metrics",
    "measurement_learning_rate",
    "measurement_objective",
    "rollout_learning_rate",
    "rollout_metrics",
    "run_minimal_toy_ladder",
    "run_oracle_rung",
    "train_measurement_rung",
    "train_rollout_rung",
]
