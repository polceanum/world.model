"""Long-horizon differentiable free-motion qualification rung.

This is the first post-v2 scale step. It isolates one complexity: sixteen RGB
observations spanning 0.75 seconds are fitted into an anchor position and
velocity, then the existing analytic equations answer five queries out to two
seconds. Object count, radius, camera, gravity, drag, rendering, and the
analytically contact-safe regime remain fixed within the rung.

The seed namespaces and temporal indices below are protocol, not convenient
defaults. They were declared before this module materialized an episode.
Development may inspect only the two development namespaces; selector,
confirmation, and final data remain unopened until independent source review.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from world_model.belief import BeliefFactory
from world_model.datasets.collate import collate_episodes
from world_model.dynamics import AnalyticKinematics, FreeMotionFitResult, fit_free_motion
from world_model.simulator.episode import Episode, generate_episode
from world_model.training.minimal_toy import (
    DifferentiableToyStateEstimator,
    ToyStateEstimate,
    measurement_objective,
)
from world_model.utils.config import OrpheusConfig
from world_model.utils.seeds import seed_everything

DEVELOPMENT_TRAIN_SEEDS = tuple(range(31_000_000, 31_000_032))
DEVELOPMENT_AUDIT_SEEDS = tuple(range(31_100_000, 31_100_016))
SELECTOR_SEEDS = tuple(range(32_000_000, 32_000_016))
CONFIRMATION_SEEDS = tuple(range(33_000_000, 33_000_016))
FINAL_TEST_SEEDS = tuple(range(34_000_000, 34_000_032))

HISTORY_FRAME_INDICES = tuple(range(16))
ANCHOR_FRAME_INDEX = 15
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TARGET_FRAME_INDICES = (17, 20, 25, 35, 55)

ARCHITECTURE_VERSION = 2
ARCHITECTURE_ATTEMPT = 2
MAXIMUM_ARCHITECTURE_ATTEMPTS = 2
DEVELOPMENT_UPDATES = 32
FROZEN_CONFIG_SHA256 = "cb40cf08178453f1b0045afd293e82237b31e19b3f38b3136cce95830bd25cd8"

# The inverse renderer is least biased near the anchor in this regime. A
# positive reliability rate derived continuously from the existing four-scalar
# mask head gives older frames a smooth exponential taper without introducing
# a learned dynamics component or a discrete observation selector. At the
# frozen initial bias this is approximately exp(10 * signed_seconds).
TEMPORAL_RELIABILITY_RATE_SCALE = 2.5
MINIMUM_ABSOLUTE_2S_GRADIENT = 1.0e-7
MAXIMUM_TRIVIAL_BASELINE_RMSE_RATIO = 0.5
FUTURE_VELOCITY_RMSE_MAX_MPS = 0.01

PERCEPTION_LATENCY_MAX_SECONDS = 2.0
STATE_ONLY_ROLLOUT_LATENCY_MAX_SECONDS = 0.05
LATENCY_WARMUP_REPEATS = 1
PERCEPTION_LATENCY_REPEATS = 3
ROLLOUT_LATENCY_REPEATS = 20
PROCESS_MAX_RSS_BYTES = 2_500_000_000
PROCESS_RSS_DELTA_MAX_BYTES = 1_500_000_000


@dataclass(frozen=True)
class TemporalFreeMotionGates:
    """Frozen promotion thresholds for every protected split."""

    oracle_position_rmse_m: float = 1.0e-5
    oracle_velocity_rmse_mps: float = 1.0e-5
    oracle_simulator_horizon_position_rmse_m: float = 5.0e-4
    oracle_simulator_horizon_velocity_rmse_mps: float = 5.0e-4
    centre_rmse_pixels: float = 0.5
    radius_relative_rmse: float = 0.02
    valid_fraction: float = 1.0
    current_position_rmse_m: float = 0.012
    current_velocity_rmse_mps: float = 0.02
    horizon_0_10_rmse_m: float = 0.012
    horizon_0_25_rmse_m: float = 0.014
    horizon_0_50_rmse_m: float = 0.018
    horizon_1_00_rmse_m: float = 0.025
    horizon_2_00_rmse_m: float = 0.040
    future_velocity_rmse_mps: float = FUTURE_VELOCITY_RMSE_MAX_MPS
    trivial_baseline_rmse_ratio: float = MAXIMUM_TRIVIAL_BASELINE_RMSE_RATIO
    semigroup_max_abs_m: float = 1.0e-5
    semigroup_velocity_max_abs_mps: float = 1.0e-5
    minimum_absolute_2s_gradient: float = MINIMUM_ABSOLUTE_2S_GRADIENT
    perception_latency_seconds: float = PERCEPTION_LATENCY_MAX_SECONDS
    state_only_rollout_latency_seconds: float = STATE_ONLY_ROLLOUT_LATENCY_MAX_SECONDS
    process_max_rss_bytes: int = PROCESS_MAX_RSS_BYTES
    process_rss_delta_bytes: int = PROCESS_RSS_DELTA_MAX_BYTES


DEFAULT_GATES = TemporalFreeMotionGates()


class TemporalQualificationError(RuntimeError):
    """A fail-fast rung failed, so later protected data must stay unopened."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


@dataclass(frozen=True)
class TemporalHistoryEstimate:
    """RGB history, fitted anchor state, and analytic future queries."""

    frame_estimate: ToyStateEstimate
    measured_positions: Tensor
    measurement_confidence: Tensor
    measurement_support: Tensor
    fit: FreeMotionFitResult
    rollout_positions: Tensor
    rollout_velocities: Tensor


class TemporalFreeMotionEstimator(nn.Module):
    """Four learned mask scalars around tensor fitting and analytic rollout.

    No learned transition is present. RGB geometry is inherited from the v2
    differentiable inverse renderer, temporal state comes only from
    :func:`fit_free_motion`, and future state comes only from
    :class:`AnalyticKinematics`.
    """

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        world_radius_m: float,
        gravity: Sequence[float],
        drag: float,
        horizons_seconds: Sequence[float] = HORIZONS_SECONDS,
    ) -> None:
        super().__init__()
        if tuple(float(value) for value in horizons_seconds) != HORIZONS_SECONDS:
            raise ValueError(f"temporal rung horizons must be {HORIZONS_SECONDS!r}")
        if len(gravity) != 3 or not all(math.isfinite(float(value)) for value in gravity):
            raise ValueError("gravity must contain three finite values")
        if not math.isfinite(drag) or drag <= 0.0:
            raise ValueError("drag must be finite and positive")
        self.image_size = tuple(int(value) for value in image_size)
        self.world_radius_m = float(world_radius_m)
        self.drag = float(drag)
        self.horizons_seconds = HORIZONS_SECONDS
        self.state_estimator = DifferentiableToyStateEstimator(
            image_size=self.image_size,
            world_radius_m=self.world_radius_m,
        )
        self.kinematics = AnalyticKinematics()
        self.register_buffer(
            "gravity",
            torch.tensor(tuple(float(value) for value in gravity), dtype=torch.float32),
        )
        self._belief_factory = BeliefFactory(
            max_objects=1,
            geometry_dim=1,
            appearance_dim=1,
            residual_dynamics_dim=1,
            modal_count=0,
            modal_dim=1,
            parameter_memory_dim=1,
            global_code_dim=1,
            initial_radius=self.world_radius_m,
            initial_mass=1.0,
            initial_restitution=0.7,
            initial_drag=self.drag,
            initial_friction=0.2,
        )

    @property
    def mask_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the only four learned scalars in stable state-dict order."""

        return tuple(self.state_estimator.mask_head.parameters())

    def _objects(self, position: Tensor, velocity: Tensor) -> Any:
        if position.ndim != 3 or position.shape[1:] != (1, 3):
            raise ValueError("position must have shape [B,1,3]")
        if velocity.shape != position.shape:
            raise ValueError("velocity must have the same [B,1,3] shape as position")
        batch = position.shape[0]
        belief = self._belief_factory.create(
            batch_size=batch,
            device=position.device,
            dtype=position.dtype,
            gravity=tuple(float(value) for value in self.gravity.detach().cpu()),
        )
        return belief.objects.replace(
            active=torch.ones((batch, 1), device=position.device, dtype=torch.bool),
            object_id=torch.zeros((batch, 1), device=position.device, dtype=torch.int64),
            position=position,
            velocity=velocity,
        )

    def fit_history(
        self,
        images: Tensor,
        world_from_camera: Tensor,
        intrinsics: Tensor,
        timestamps: Tensor,
    ) -> tuple[ToyStateEstimate, Tensor, Tensor, Tensor, FreeMotionFitResult]:
        """Infer all frame positions and fit state at the final timestamp."""

        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,T,3,H,W]")
        batch, frames = images.shape[:2]
        if frames != len(HISTORY_FRAME_INDICES):
            raise ValueError(f"history must contain exactly {len(HISTORY_FRAME_INDICES)} frames")
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(f"history image size must be {self.image_size!r}")
        if world_from_camera.shape != (batch, frames, 4, 4):
            raise ValueError("world_from_camera must have shape [B,T,4,4]")
        if intrinsics.shape != (batch, frames, 3, 3):
            raise ValueError("intrinsics must have shape [B,T,3,3]")
        if timestamps.shape != (batch, frames):
            raise ValueError("timestamps must have shape [B,T]")
        if not torch.isfinite(timestamps).all() or torch.any(
            timestamps[:, 1:] <= timestamps[:, :-1]
        ):
            raise ValueError("timestamps must be finite and strictly increasing")

        flattened_frames = batch * frames
        frame_estimate = self.state_estimator(
            images.reshape(flattened_frames, *images.shape[2:]),
            world_from_camera.reshape(flattened_frames, 4, 4),
            intrinsics.reshape(flattened_frames, 3, 3),
        )
        measured_positions = frame_estimate.world_position.reshape(batch, frames, 1, 3)
        confidence = frame_estimate.photometric_radius.confidence.reshape(batch, frames, 1)
        # Keep the renderer's boolean validity mask diagnostic-only. Feeding an
        # input-derived boolean into the temporal solve would make observation
        # selection nondifferentiable. Every frozen history frame is admissible;
        # continuous confidence and a continuous anchor-proximity reliability
        # prior determine its fit weight.
        support = frame_estimate.photometric_radius.valid_mask.reshape(batch, frames, 1)
        mean_mask_logit = frame_estimate.slot_mask_logits.mean(dim=(-2, -1)).reshape(
            batch, frames, 1
        )
        reliability_rate = TEMPORAL_RELIABILITY_RATE_SCALE * F.softplus(mean_mask_logit)
        signed_time = (timestamps - timestamps[:, -1:]).unsqueeze(-1)
        fit_weight = confidence * torch.exp(signed_time * reliability_rate)
        gravity = self.gravity.to(device=images.device, dtype=images.dtype)
        drag = images.new_full((batch, 1), self.drag)
        fit = fit_free_motion(
            measured_positions,
            timestamps,
            gravity=gravity,
            drag=drag,
            anchor_time=timestamps[:, -1],
            weights=fit_weight,
            minimum_support=len(HISTORY_FRAME_INDICES),
        )
        return frame_estimate, measured_positions, fit_weight, support, fit

    def rollout_state(
        self,
        position: Tensor,
        velocity: Tensor,
        *,
        horizons_seconds: Sequence[float] = HORIZONS_SECONDS,
    ) -> tuple[Tensor, Tensor]:
        """Query the existing analytic kinematics without learned dynamics."""

        horizons = tuple(float(value) for value in horizons_seconds)
        if not horizons or any(not math.isfinite(value) or value < 0.0 for value in horizons):
            raise ValueError("rollout horizons must be finite and nonnegative")
        objects = self._objects(position, velocity)
        gravity = self.gravity.to(device=position.device, dtype=position.dtype)
        gravity = gravity.unsqueeze(0).expand(position.shape[0], -1)
        predicted = [self.kinematics(objects, gravity, horizon) for horizon in horizons]
        return (
            torch.stack([item.position for item in predicted], dim=1),
            torch.stack([item.velocity for item in predicted], dim=1),
        )

    def semigroup_error(self, position: Tensor, velocity: Tensor) -> Tensor:
        """Return direct-vs-composed two-second position absolute error."""

        position_error, _ = self.semigroup_errors(position, velocity)
        return position_error

    def semigroup_errors(self, position: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]:
        """Return direct-vs-composed two-second position and velocity errors."""

        objects = self._objects(position, velocity)
        gravity = self.gravity.to(device=position.device, dtype=position.dtype)
        gravity = gravity.unsqueeze(0).expand(position.shape[0], -1)
        first = self.kinematics(objects, gravity, 1.0)
        composed = self.kinematics(first, gravity, 1.0)
        direct = self.kinematics(objects, gravity, 2.0)
        return (
            (composed.position - direct.position).abs(),
            (composed.velocity - direct.velocity).abs(),
        )

    def forward(
        self,
        images: Tensor,
        world_from_camera: Tensor,
        intrinsics: Tensor,
        timestamps: Tensor,
    ) -> TemporalHistoryEstimate:
        frame_estimate, measured_positions, confidence, support, fit = self.fit_history(
            images,
            world_from_camera,
            intrinsics,
            timestamps,
        )
        rollout_positions, rollout_velocities = self.rollout_state(fit.position, fit.velocity)
        return TemporalHistoryEstimate(
            frame_estimate=frame_estimate,
            measured_positions=measured_positions,
            measurement_confidence=confidence,
            measurement_support=support,
            fit=fit,
            rollout_positions=rollout_positions,
            rollout_velocities=rollout_velocities,
        )


def _assert_seed_namespaces() -> None:
    namespaces = (
        DEVELOPMENT_TRAIN_SEEDS,
        DEVELOPMENT_AUDIT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    flattened = [seed for seeds in namespaces for seed in seeds]
    if any(not seeds for seeds in namespaces):
        raise RuntimeError("every temporal rung seed namespace must be nonempty")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("temporal rung seed namespaces must be disjoint")


def _config_sha256(config: OrpheusConfig) -> str:
    encoded = json.dumps(
        config.to_dict(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_temporal_protocol(config: OrpheusConfig) -> None:
    _assert_seed_namespaces()
    simulator = config.simulator
    required = {
        "image_size": (48, 48),
        "frame_rate": 20,
        "physics_rate": 120,
        "sequence_frames": 56,
        "min_objects": 1,
        "max_objects": 1,
        "gravity": (0.0, 0.0, 0.0),
        "radius_range": (0.21, 0.21),
        "mass_range": (1.0, 1.0),
        "restitution_range": (0.7, 0.7),
        "drag_range": (0.05, 0.05),
        "friction_range": (0.2, 0.2),
        "initial_speed_range": (0.035, 0.035),
        "camera_motion": "fixed",
        "render_noise_std": 0.0,
        "ensure_collision": False,
        "external_impulse_probability": 0.0,
        "scenario_mixture": ("baseline",),
    }
    for name, expected in required.items():
        actual = getattr(simulator, name)
        if actual != expected:
            raise ValueError(
                f"temporal free-motion config requires simulator.{name}={expected!r}, "
                f"got {actual!r}"
            )
    if config.project.seed != DEVELOPMENT_TRAIN_SEEDS[0] or not config.project.deterministic:
        raise ValueError("temporal rung requires its frozen deterministic project seed")
    if config.device.preference != "cpu" or config.device.cuda_amp:
        raise ValueError("temporal rung requires CPU float32 without AMP")
    if config.training.steps != DEVELOPMENT_UPDATES or config.training.batch_size != 4:
        raise ValueError("temporal rung requires exactly 32 batch-four development updates")
    if (
        config.training.learning_rate != 0.0002
        or config.training.weight_decay != 0.0
        or config.training.grad_clip_norm != 2.0
    ):
        raise ValueError("temporal rung requires AdamW lr=0.0002, weight_decay=0, clip=2")
    if tuple(config.evaluation.horizons_seconds) != HORIZONS_SECONDS:
        raise ValueError(f"evaluation horizons must be {HORIZONS_SECONDS!r}")
    derived_targets = tuple(
        ANCHOR_FRAME_INDEX + int(round(horizon * simulator.frame_rate))
        for horizon in HORIZONS_SECONDS
    )
    if derived_targets != TARGET_FRAME_INDICES:
        raise RuntimeError("temporal target indices do not match frame rate and horizons")
    if TARGET_FRAME_INDICES[-1] != simulator.sequence_frames - 1:
        raise ValueError("the two-second target must be the final generated frame")
    config_sha256 = _config_sha256(config)
    if config_sha256 != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "temporal rung requires its exact frozen resolved config: "
            f"expected {FROZEN_CONFIG_SHA256}, got {config_sha256}"
        )


def temporal_protocol(gates: TemporalFreeMotionGates = DEFAULT_GATES) -> dict[str, Any]:
    """Return the frozen seed, metric, latency, and attempt contract."""

    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": MAXIMUM_ARCHITECTURE_ATTEMPTS,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "development_train_seeds": list(DEVELOPMENT_TRAIN_SEEDS),
        "development_audit_seeds": list(DEVELOPMENT_AUDIT_SEEDS),
        "selector_seeds": list(SELECTOR_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "final_test_seeds": list(FINAL_TEST_SEEDS),
        "history_frame_indices": list(HISTORY_FRAME_INDICES),
        "anchor_frame_index": ANCHOR_FRAME_INDEX,
        "horizons_seconds": list(HORIZONS_SECONDS),
        "target_frame_indices": list(TARGET_FRAME_INDICES),
        "development_updates": DEVELOPMENT_UPDATES,
        "torch_intraop_threads": 1,
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 0.0002,
            "weight_decay": 0.0,
            "gradient_clip_l2_norm": 2.0,
            "batch_size": 4,
            "updates": DEVELOPMENT_UPDATES,
        },
        "gates": asdict(gates),
        "estimator": "v2_rgb_inverse_renderer_then_weighted_free_motion_fit",
        "temporal_fit_weight": {
            "type": "continuous_mask_conditioned_anchor_proximity_reliability",
            "rate_scale": TEMPORAL_RELIABILITY_RATE_SCALE,
            "boolean_photometric_validity_use": "diagnostic_only",
            "solver_support": "all_sixteen_predeclared_frames",
        },
        "rollout": "AnalyticKinematics_only",
        "perception_latency": {
            "batch_size": 1,
            "history_frames": len(HISTORY_FRAME_INDICES),
            "warmup_repeats": LATENCY_WARMUP_REPEATS,
            "timed_repeats": PERCEPTION_LATENCY_REPEATS,
            "statistic": "median_seconds_per_episode",
            "includes": "RGB_to_anchor_state",
        },
        "state_only_rollout_latency": {
            "batch_size": 1,
            "query_horizons_seconds": list(HORIZONS_SECONDS),
            "warmup_repeats": LATENCY_WARMUP_REPEATS,
            "timed_repeats": ROLLOUT_LATENCY_REPEATS,
            "statistic": "median_seconds_per_five_query_rollout",
            "includes": "anchor_state_to_five_analytic_queries",
        },
        "gradient_gate": (
            "MSE at the 2.0-second target has a finite absolute gradient of at least "
            f"{MINIMUM_ABSOLUTE_2S_GRADIENT:.1e} to each of the three mask weights "
            "and mask bias"
        ),
        "trivial_baselines": {
            "position": "exact_anchor_position_with_zero_velocity",
            "velocity": "zero_velocity",
            "maximum_per_horizon_rmse_ratio": MAXIMUM_TRIVIAL_BASELINE_RMSE_RATIO,
        },
        "qualification_order": "selector_then_confirmation_then_one_shot_final",
    }


def _development_batch(config: OrpheusConfig, seeds: Sequence[int]) -> dict[str, Any]:
    requested = tuple(int(seed) for seed in seeds)
    allowed = set(DEVELOPMENT_TRAIN_SEEDS) | set(DEVELOPMENT_AUDIT_SEEDS)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("development seeds must be a nonempty unique sequence")
    if not set(requested).issubset(allowed):
        raise ValueError("development may not materialize protected seed namespaces")
    return collate_episodes([generate_episode(config, seed) for seed in requested])


def _protected_batch(config: OrpheusConfig, split: str) -> dict[str, Any]:
    manifests = {
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }
    if split not in manifests:
        raise ValueError(f"unknown protected split: {split}")
    return collate_episodes([generate_episode(config, seed) for seed in manifests[split]])


def _assert_free_motion_batch(batch: Mapping[str, Any], seeds: Sequence[int]) -> None:
    events = batch["events"]
    if (
        bool(events["collision"].any())
        or bool(events["contact"].any())
        or bool(events["external_impulse"].ne(0).any())
        or bool(events["created"][:, 1:].any())
        or bool(events["removed"].any())
    ):
        raise RuntimeError(f"temporal free-motion manifest contains an event: {tuple(seeds)!r}")
    objects = batch["objects"]
    if not bool(objects["active"][:, :, :1].all()):
        raise RuntimeError("the single sphere must remain active for the complete episode")
    if not bool(batch["labels"]["projected_valid"][:, :, :1].all()):
        raise RuntimeError("the single sphere must remain projectable for the complete episode")


def _history(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    indices = torch.tensor(HISTORY_FRAME_INDICES, dtype=torch.int64)
    return {
        "images": batch["rgb"].index_select(1, indices),
        "world_from_camera": batch["camera"]["world_from_camera"].index_select(1, indices),
        "intrinsics": batch["camera"]["intrinsics"].index_select(1, indices),
        "timestamps": batch["timestamps"].index_select(1, indices),
    }


def _history_frame_targets(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    indices = torch.tensor(HISTORY_FRAME_INDICES, dtype=torch.int64)
    batch_size = batch["rgb"].shape[0]
    frame_count = len(HISTORY_FRAME_INDICES)

    def flatten(value: Tensor) -> Tensor:
        selected = value.index_select(1, indices)
        return selected.reshape(batch_size * frame_count, *selected.shape[2:])

    return {
        "image": flatten(batch["rgb"]),
        "world_from_camera": flatten(batch["camera"]["world_from_camera"]),
        "intrinsics": flatten(batch["camera"]["intrinsics"]),
        "position": flatten(batch["objects"]["position"][:, :, :1]),
        "centre": flatten(batch["labels"]["projected_center"][:, :, :1]),
        "radius_pixels": flatten(batch["labels"]["apparent_radius"][:, :, :1]),
        "mask": flatten(batch["labels"]["segmentation_mask"][:, :, :1]),
    }


def _targets(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
    target_indices = torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64)
    return (
        batch["objects"]["position"][:, ANCHOR_FRAME_INDEX, :1],
        batch["objects"]["velocity"][:, ANCHOR_FRAME_INDEX, :1],
        batch["objects"]["position"][:, :, :1].index_select(1, target_indices),
    )


def _model_from_config(config: OrpheusConfig) -> TemporalFreeMotionEstimator:
    return TemporalFreeMotionEstimator(
        image_size=config.simulator.image_size,
        world_radius_m=config.simulator.radius_range[0],
        gravity=config.simulator.gravity,
        drag=config.simulator.drag_range[0],
    )


def _finite_float_metrics(value: Mapping[str, Any], *, prefix: str = "") -> None:
    for name, item in value.items():
        qualified = f"{prefix}.{name}" if prefix else name
        if isinstance(item, Mapping):
            _finite_float_metrics(item, prefix=qualified)
        elif isinstance(item, bool):
            continue
        elif isinstance(item, (int, float)) and not math.isfinite(float(item)):
            raise FloatingPointError(f"metric {qualified} is nonfinite")


def _slice_batch(value: Any, count: int) -> Any:
    if isinstance(value, Tensor):
        return value[:count]
    if isinstance(value, Mapping):
        return {key: _slice_batch(item, count) for key, item in value.items()}
    if isinstance(value, list):
        return value[:count]
    return value


def _process_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _model_counts(model: nn.Module) -> dict[str, int]:
    state = model.state_dict()
    return {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "persistent_state_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in state.values()
        ),
    }


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def oracle_metrics(
    model: TemporalFreeMotionEstimator,
    config: OrpheusConfig,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit exact simulator positions, then query deployed analytic dynamics."""

    indices = torch.tensor(HISTORY_FRAME_INDICES, dtype=torch.int64)
    positions = batch["objects"]["position"][:, :, :1].index_select(1, indices)
    timestamps = batch["timestamps"].index_select(1, indices)
    batch_size = positions.shape[0]
    gravity = positions.new_tensor(config.simulator.gravity)
    drag = positions.new_full((batch_size, 1), config.simulator.drag_range[0])
    fit = fit_free_motion(
        positions,
        timestamps,
        gravity=gravity,
        drag=drag,
        anchor_time=timestamps[:, -1],
        minimum_support=len(HISTORY_FRAME_INDICES),
    )
    target_position, target_velocity, target_future = _targets(batch)
    future, future_velocity = model.rollout_state(fit.position, fit.velocity)
    target_future_velocity = batch["objects"]["velocity"][:, :, :1].index_select(
        1,
        torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64),
    )
    metrics: dict[str, Any] = {
        "position_rmse_m": float((fit.position - target_position).square().mean().sqrt()),
        "velocity_rmse_mps": float((fit.velocity - target_velocity).square().mean().sqrt()),
        "valid_fraction": float(fit.valid.to(dtype=positions.dtype).mean()),
        "maximum_condition_number": float(fit.condition_number.max()),
        "simulator_horizon_position_rmse_m": {
            f"{horizon:.2f}": float(
                (future[:, index] - target_future[:, index]).square().mean().sqrt()
            )
            for index, horizon in enumerate(HORIZONS_SECONDS)
        },
        "simulator_horizon_velocity_rmse_mps": {
            f"{horizon:.2f}": float(
                (future_velocity[:, index] - target_future_velocity[:, index])
                .square()
                .mean()
                .sqrt()
            )
            for index, horizon in enumerate(HORIZONS_SECONDS)
        },
    }
    _finite_float_metrics(metrics)
    return metrics


def _accuracy_tensors(
    model: TemporalFreeMotionEstimator,
    batch: Mapping[str, Any],
    *,
    enable_grad: bool,
) -> tuple[TemporalHistoryEstimate, dict[str, Tensor]]:
    context = nullcontext() if enable_grad else torch.no_grad()
    with context:
        output = model(**_history(batch))
        target_position, target_velocity, target_future = _targets(batch)
        target_future_velocity = batch["objects"]["velocity"][:, :, :1].index_select(
            1,
            torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64),
        )
        semigroup_position, semigroup_velocity = model.semigroup_errors(
            output.fit.position,
            output.fit.velocity,
        )
        tensors = {
            "target_position": target_position,
            "target_velocity": target_velocity,
            "target_future": target_future,
            "target_future_velocity": target_future_velocity,
            # Compare temporal rollout with a stationary model starting from
            # the exact same RGB-inferred anchor, so perception error is held
            # constant and the ratio isolates temporal improvement.
            "persistence_future": output.fit.position[:, None].expand_as(target_future),
            "zero_future_velocity": torch.zeros_like(target_future_velocity),
            "semigroup_position_error": semigroup_position,
            "semigroup_velocity_error": semigroup_velocity,
        }
    return output, tensors


@torch.no_grad()
def accuracy_metrics(
    model: TemporalFreeMotionEstimator,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    output, targets = _accuracy_tensors(model, batch, enable_grad=False)
    frame_targets = _history_frame_targets(batch)
    batch_size = batch["rgb"].shape[0]
    frame_count = len(HISTORY_FRAME_INDICES)
    centres = output.frame_estimate.centres.reshape(batch_size, frame_count, 1, 2)
    radii = output.frame_estimate.radius_pixels.reshape(batch_size, frame_count, 1)
    centre_target = frame_targets["centre"].reshape(batch_size, frame_count, 1, 2)
    radius_target = frame_targets["radius_pixels"].reshape(batch_size, frame_count, 1)
    pixel_scale = centres.new_tensor(
        (0.5 * (model.image_size[1] - 1), 0.5 * (model.image_size[0] - 1))
    )
    radius_relative_error = (radii - radius_target) / radius_target.clamp_min(1.0e-8)
    all_valid = output.measurement_support & output.fit.valid[:, None, :]
    horizon_rmse = {
        f"{horizon:.2f}": float(
            (output.rollout_positions[:, index] - targets["target_future"][:, index])
            .square()
            .mean()
            .sqrt()
        )
        for index, horizon in enumerate(HORIZONS_SECONDS)
    }
    future_velocity_rmse = {
        f"{horizon:.2f}": float(
            (output.rollout_velocities[:, index] - targets["target_future_velocity"][:, index])
            .square()
            .mean()
            .sqrt()
        )
        for index, horizon in enumerate(HORIZONS_SECONDS)
    }
    persistence_rmse = {
        f"{horizon:.2f}": float(
            (targets["persistence_future"][:, index] - targets["target_future"][:, index])
            .square()
            .mean()
            .sqrt()
        )
        for index, horizon in enumerate(HORIZONS_SECONDS)
    }
    zero_velocity_rmse = {
        f"{horizon:.2f}": float(
            (
                targets["zero_future_velocity"][:, index]
                - targets["target_future_velocity"][:, index]
            )
            .square()
            .mean()
            .sqrt()
        )
        for index, horizon in enumerate(HORIZONS_SECONDS)
    }
    metrics: dict[str, Any] = {
        "centre_rmse_pixels": float(
            ((centres - centre_target) * pixel_scale).square().mean().sqrt()
        ),
        "radius_relative_rmse": float(radius_relative_error.square().mean().sqrt()),
        "valid_fraction": float(all_valid.to(dtype=centres.dtype).mean()),
        "current_position_rmse_m": float(
            (output.fit.position - targets["target_position"]).square().mean().sqrt()
        ),
        "current_velocity_rmse_mps": float(
            (output.fit.velocity - targets["target_velocity"]).square().mean().sqrt()
        ),
        "horizon_rmse_m": horizon_rmse,
        "future_velocity_rmse_mps": future_velocity_rmse,
        "persistence_baseline_rmse_m": persistence_rmse,
        "zero_velocity_baseline_rmse_mps": zero_velocity_rmse,
        "persistence_rmse_ratio": {
            key: horizon_rmse[key] / max(value, torch.finfo(centres.dtype).eps)
            for key, value in persistence_rmse.items()
        },
        "zero_velocity_rmse_ratio": {
            key: future_velocity_rmse[key] / max(value, torch.finfo(centres.dtype).eps)
            for key, value in zero_velocity_rmse.items()
        },
        "semigroup_max_abs_m": float(targets["semigroup_position_error"].max()),
        "semigroup_velocity_max_abs_mps": float(targets["semigroup_velocity_error"].max()),
        "maximum_condition_number": float(output.fit.condition_number.max()),
        "minimum_fit_weight": float(output.fit.support_weight.min()),
    }
    _finite_float_metrics(metrics)
    return metrics


def two_second_gradient_metrics(
    model: TemporalFreeMotionEstimator,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove two-second target loss reaches every learned mask scalar."""

    output, targets = _accuracy_tensors(model, batch, enable_grad=True)
    loss = F.mse_loss(output.rollout_positions[:, -1], targets["target_future"][:, -1])
    gradients = torch.autograd.grad(loss, model.mask_parameters, allow_unused=False)
    flattened = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
    if flattened.numel() != 4:
        raise RuntimeError("temporal estimator must retain exactly four learned mask scalars")
    finite = torch.isfinite(flattened)
    nonzero = flattened != 0
    return {
        "loss": float(loss.detach()),
        "gradient_values": [float(value) for value in flattened],
        "finite_to_every_mask_scalar": bool(finite.all()),
        "nonzero_to_every_mask_scalar": bool(nonzero.all()),
        "minimum_absolute_gradient": float(flattened.abs().min()),
        "gradient_l2_norm": float(torch.linalg.vector_norm(flattened)),
    }


@torch.no_grad()
def latency_metrics(
    model: TemporalFreeMotionEstimator,
    batch: Mapping[str, Any],
) -> dict[str, float | int]:
    """Measure RGB-to-state and state-only five-query rollout separately."""

    model.eval()
    inputs = {name: value[:1] for name, value in _history(batch).items()}
    rss_before = _process_max_rss_bytes()
    if LATENCY_WARMUP_REPEATS != 1:
        raise RuntimeError("temporal latency protocol requires exactly one warmup repeat")
    frame_estimate, measured, confidence, support, fit = model.fit_history(**inputs)
    del frame_estimate, measured, confidence, support
    rss_after_perception = _process_max_rss_bytes()
    model.rollout_state(fit.position, fit.velocity)
    rss_after_rollout = _process_max_rss_bytes()

    perception_samples: list[float] = []
    latest_fit: FreeMotionFitResult | None = None
    for _ in range(PERCEPTION_LATENCY_REPEATS):
        started = time.perf_counter()
        frame_estimate, measured, confidence, support, latest_fit = model.fit_history(**inputs)
        perception_samples.append(time.perf_counter() - started)
        del frame_estimate, measured, confidence, support
    assert latest_fit is not None

    rollout_samples: list[float] = []
    for _ in range(ROLLOUT_LATENCY_REPEATS):
        started = time.perf_counter()
        model.rollout_state(latest_fit.position, latest_fit.velocity)
        rollout_samples.append(time.perf_counter() - started)
    rss_after = _process_max_rss_bytes()
    metrics: dict[str, float | int] = {
        "perception_median_seconds_per_episode": float(median(perception_samples)),
        "state_only_rollout_median_seconds": float(median(rollout_samples)),
        "perception_process_rss_delta_bytes": max(0, rss_after_perception - rss_before),
        "state_only_rollout_process_rss_delta_bytes": max(
            0, rss_after_rollout - rss_after_perception
        ),
        "process_max_rss_bytes": rss_after,
        "process_rss_delta_bytes": max(0, rss_after - rss_before),
    }
    _finite_float_metrics(metrics)
    return metrics


def _gate_failures(
    oracle: Mapping[str, Any],
    accuracy: Mapping[str, Any],
    gradient: Mapping[str, Any],
    latency: Mapping[str, Any],
    gates: TemporalFreeMotionGates,
) -> list[str]:
    limits = {
        "oracle.position_rmse_m": (oracle["position_rmse_m"], gates.oracle_position_rmse_m),
        "oracle.velocity_rmse_mps": (
            oracle["velocity_rmse_mps"],
            gates.oracle_velocity_rmse_mps,
        ),
        "oracle.simulator_horizon_position_rmse_m": (
            max(oracle["simulator_horizon_position_rmse_m"].values()),
            gates.oracle_simulator_horizon_position_rmse_m,
        ),
        "oracle.simulator_horizon_velocity_rmse_mps": (
            max(oracle["simulator_horizon_velocity_rmse_mps"].values()),
            gates.oracle_simulator_horizon_velocity_rmse_mps,
        ),
        "accuracy.centre_rmse_pixels": (
            accuracy["centre_rmse_pixels"],
            gates.centre_rmse_pixels,
        ),
        "accuracy.radius_relative_rmse": (
            accuracy["radius_relative_rmse"],
            gates.radius_relative_rmse,
        ),
        "accuracy.current_position_rmse_m": (
            accuracy["current_position_rmse_m"],
            gates.current_position_rmse_m,
        ),
        "accuracy.current_velocity_rmse_mps": (
            accuracy["current_velocity_rmse_mps"],
            gates.current_velocity_rmse_mps,
        ),
        "accuracy.horizon_rmse_m.0.10": (
            accuracy["horizon_rmse_m"]["0.10"],
            gates.horizon_0_10_rmse_m,
        ),
        "accuracy.horizon_rmse_m.0.25": (
            accuracy["horizon_rmse_m"]["0.25"],
            gates.horizon_0_25_rmse_m,
        ),
        "accuracy.horizon_rmse_m.0.50": (
            accuracy["horizon_rmse_m"]["0.50"],
            gates.horizon_0_50_rmse_m,
        ),
        "accuracy.horizon_rmse_m.1.00": (
            accuracy["horizon_rmse_m"]["1.00"],
            gates.horizon_1_00_rmse_m,
        ),
        "accuracy.horizon_rmse_m.2.00": (
            accuracy["horizon_rmse_m"]["2.00"],
            gates.horizon_2_00_rmse_m,
        ),
        "accuracy.future_velocity_rmse_mps": (
            max(accuracy["future_velocity_rmse_mps"].values()),
            gates.future_velocity_rmse_mps,
        ),
        "accuracy.persistence_rmse_ratio": (
            max(accuracy["persistence_rmse_ratio"].values()),
            gates.trivial_baseline_rmse_ratio,
        ),
        "accuracy.zero_velocity_rmse_ratio": (
            max(accuracy["zero_velocity_rmse_ratio"].values()),
            gates.trivial_baseline_rmse_ratio,
        ),
        "accuracy.semigroup_max_abs_m": (
            accuracy["semigroup_max_abs_m"],
            gates.semigroup_max_abs_m,
        ),
        "accuracy.semigroup_velocity_max_abs_mps": (
            accuracy["semigroup_velocity_max_abs_mps"],
            gates.semigroup_velocity_max_abs_mps,
        ),
        "latency.perception_median_seconds_per_episode": (
            latency["perception_median_seconds_per_episode"],
            gates.perception_latency_seconds,
        ),
        "latency.state_only_rollout_median_seconds": (
            latency["state_only_rollout_median_seconds"],
            gates.state_only_rollout_latency_seconds,
        ),
        "latency.process_max_rss_bytes": (
            latency["process_max_rss_bytes"],
            gates.process_max_rss_bytes,
        ),
        "latency.process_rss_delta_bytes": (
            latency["process_rss_delta_bytes"],
            gates.process_rss_delta_bytes,
        ),
    }
    failures = [
        f"{name}={float(actual):.9g} exceeds {float(limit):.9g}"
        for name, (actual, limit) in limits.items()
        if float(actual) > float(limit)
    ]
    if float(oracle["valid_fraction"]) < 1.0:
        failures.append("oracle.valid_fraction is below 1.0")
    if float(accuracy["valid_fraction"]) < gates.valid_fraction:
        failures.append("accuracy.valid_fraction is below 1.0")
    if not bool(gradient["finite_to_every_mask_scalar"]):
        failures.append("two-second gradient is nonfinite for at least one mask scalar")
    if not bool(gradient["nonzero_to_every_mask_scalar"]):
        failures.append("two-second gradient is zero for at least one mask scalar")
    if float(gradient["minimum_absolute_gradient"]) < gates.minimum_absolute_2s_gradient:
        failures.append(
            "two-second minimum absolute per-mask-scalar gradient "
            f"{float(gradient['minimum_absolute_gradient']):.9g} is below "
            f"{gates.minimum_absolute_2s_gradient:.9g}"
        )
    return failures


def evaluate_gate(
    model: TemporalFreeMotionEstimator,
    config: OrpheusConfig,
    batch: Mapping[str, Any],
    *,
    gates: TemporalFreeMotionGates = DEFAULT_GATES,
) -> dict[str, Any]:
    model.eval()
    oracle = oracle_metrics(model, config, batch)
    accuracy = accuracy_metrics(model, batch)
    gradient = two_second_gradient_metrics(model, _slice_batch(batch, 2))
    latency = latency_metrics(model, _slice_batch(batch, 1))
    failures = _gate_failures(oracle, accuracy, gradient, latency, gates)
    return {
        "passed": not failures,
        "failures": failures,
        "oracle": oracle,
        "accuracy": accuracy,
        "gradient": gradient,
        "latency": latency,
    }


def training_objective(
    model: TemporalFreeMotionEstimator,
    batch: Mapping[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    output, targets = _accuracy_tensors(model, batch, enable_grad=True)
    measurement, measurement_terms = measurement_objective(
        output.frame_estimate,
        _history_frame_targets(batch),
    )
    current_position = F.mse_loss(output.fit.position, targets["target_position"])
    current_velocity = F.mse_loss(output.fit.velocity, targets["target_velocity"])
    horizon_losses = [
        F.mse_loss(output.rollout_positions[:, index], targets["target_future"][:, index])
        for index in range(len(HORIZONS_SECONDS))
    ]
    rollout = torch.stack(horizon_losses).mean()
    total = measurement + 4.0 * current_position + current_velocity + 4.0 * rollout
    return total, {
        "total": total,
        "measurement": measurement,
        "current_position": current_position,
        "current_velocity": current_velocity,
        "rollout": rollout,
        **{f"measurement_{name}": value for name, value in measurement_terms.items()},
        **{
            f"horizon_{horizon:.2f}": loss
            for horizon, loss in zip(HORIZONS_SECONDS, horizon_losses, strict=True)
        },
    }


def _development_schedule() -> tuple[tuple[int, ...], ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DEVELOPMENT_TRAIN_SEEDS[0])
    indices = torch.cat(
        [torch.randperm(len(DEVELOPMENT_TRAIN_SEEDS), generator=generator) for _ in range(4)]
    )
    return tuple(
        tuple(int(index) for index in indices[start : start + 4])
        for start in range(0, indices.numel(), 4)
    )


def train_development(
    model: TemporalFreeMotionEstimator,
    config: OrpheusConfig,
    episodes: Sequence[Episode],
) -> dict[str, Any]:
    """Run four deterministic passes over development training episodes."""

    if len(episodes) != len(DEVELOPMENT_TRAIN_SEEDS):
        raise ValueError("development training requires its complete 32-seed manifest")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_hash = _model_state_sha256(model)
    minimum_gradient = math.inf
    final_terms: dict[str, float] = {}
    for update, episode_indices in enumerate(_development_schedule(), start=1):
        batch = collate_episodes([episodes[index] for index in episode_indices])
        optimizer.zero_grad(set_to_none=True)
        loss, terms = training_objective(model, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite temporal training loss at update {update}")
        loss.backward()
        gradients = [
            parameter.grad.detach().reshape(-1)
            for parameter in model.mask_parameters
            if parameter.grad is not None
        ]
        if len(gradients) != len(model.mask_parameters):
            raise FloatingPointError(f"missing mask gradient at update {update}")
        gradient = torch.cat(gradients)
        if not bool(torch.isfinite(gradient).all()) or not bool((gradient != 0).all()):
            raise FloatingPointError(f"invalid per-mask-scalar gradient at update {update}")
        norm = float(torch.linalg.vector_norm(gradient))
        minimum_gradient = min(minimum_gradient, norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(f"nonfinite temporal parameter at update {update}")
        final_terms = {name: float(value.detach()) for name, value in terms.items()}

    changed_state_keys = sorted(
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value.detach(), initial_state[name])
    )
    expected_changed = [
        "state_estimator.mask_head.bias",
        "state_estimator.mask_head.weight",
    ]
    if changed_state_keys != expected_changed:
        raise RuntimeError(
            "temporal training changed tensors outside the mask head: "
            f"expected {expected_changed!r}, got {changed_state_keys!r}"
        )
    return {
        "optimizer": "AdamW",
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "gradient_clip_l2_norm": float(config.training.grad_clip_norm),
        "updates": DEVELOPMENT_UPDATES,
        "batch_size": config.training.batch_size,
        "episode_draws": DEVELOPMENT_UPDATES * config.training.batch_size,
        "approximate_data_passes": 4.0,
        "minimum_gradient_l2_norm": minimum_gradient,
        "changed_state_keys": changed_state_keys,
        "only_mask_head_tensors_changed": True,
        "initial_model_state_sha256": initial_hash,
        "trained_model_state_sha256": _model_state_sha256(model),
        "final_terms": final_terms,
    }


def run_development(
    config: OrpheusConfig,
    *,
    gates: TemporalFreeMotionGates = DEFAULT_GATES,
) -> tuple[TemporalFreeMotionEstimator, dict[str, Any]]:
    """Train and audit development namespaces without opening protected data."""

    _assert_temporal_protocol(config)
    seed_everything(config.project.seed, deterministic=True)
    rss_start = _process_max_rss_bytes()
    model = _model_from_config(config)
    model_counts = _model_counts(model)
    if model_counts["trainable_parameters"] != 4 or model_counts["total_parameters"] != 4:
        raise RuntimeError("temporal rung must retain exactly four learned parameters")
    train_episodes = [generate_episode(config, seed) for seed in DEVELOPMENT_TRAIN_SEEDS]
    train_batch = collate_episodes(train_episodes)
    _assert_free_motion_batch(train_batch, DEVELOPMENT_TRAIN_SEEDS)
    initial_train_accuracy = accuracy_metrics(model, train_batch)
    training = train_development(model, config, train_episodes)
    audit_batch = _development_batch(config, DEVELOPMENT_AUDIT_SEEDS)
    _assert_free_motion_batch(audit_batch, DEVELOPMENT_AUDIT_SEEDS)
    audit = evaluate_gate(model, config, audit_batch, gates=gates)
    rss_end = _process_max_rss_bytes()
    resource_metrics = {
        "process_max_rss_bytes": rss_end,
        "process_rss_delta_bytes": max(0, rss_end - rss_start),
    }
    resource_failures = []
    if rss_end > gates.process_max_rss_bytes:
        resource_failures.append("development process maximum RSS exceeded its frozen ceiling")
    if resource_metrics["process_rss_delta_bytes"] > gates.process_rss_delta_bytes:
        resource_failures.append("development process RSS delta exceeded its frozen ceiling")
    report: dict[str, Any] = {
        "artifact_kind": "temporal_free_motion_development_review",
        "protocol": temporal_protocol(gates),
        "protected_data_materialized": False,
        "model": model_counts,
        "resource": resource_metrics,
        "initial_train_accuracy": initial_train_accuracy,
        "training": training,
        "development_audit": audit,
        "resource_failures": resource_failures,
        "review_ready": bool(audit["passed"] and not resource_failures),
    }
    if not report["review_ready"]:
        report["passed"] = False
        report["stopped_after"] = "development_audit"
        raise TemporalQualificationError("development review gate failed", report)
    report["passed"] = True
    report["stopped_after"] = "frozen_review_checkpoint"
    return model, report


def run_protected_qualification(
    model: TemporalFreeMotionEstimator,
    config: OrpheusConfig,
    *,
    access_recorder: Callable[[str], None],
    gates: TemporalFreeMotionGates = DEFAULT_GATES,
) -> dict[str, Any]:
    """Run the three protected rungs behind a durable access recorder."""

    _assert_temporal_protocol(config)
    if not callable(access_recorder):
        raise TypeError("protected qualification requires a callable durable access recorder")
    seed_everything(config.project.seed, deterministic=True)
    report: dict[str, Any] = {
        "artifact_kind": "temporal_free_motion_protected_qualification",
        "protocol": temporal_protocol(gates),
        "protected_data_materialized": False,
        "access_started": {
            "selector": False,
            "confirmation": False,
            "final_test": False,
        },
        "model": _model_counts(model),
        "model_state_sha256": _model_state_sha256(model),
        "rungs": {},
    }

    def materialize_and_evaluate(split: str, seeds: Sequence[int]) -> dict[str, Any]:
        try:
            access_recorder(split)
        except Exception as error:
            report["passed"] = False
            report["stopped_after"] = f"before_{split}_access"
            report["unexpected_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            raise TemporalQualificationError(
                f"could not durably authorize {split} access",
                report,
            ) from error
        report["access_started"][split] = True
        report["protected_data_materialized"] = True
        try:
            batch = _protected_batch(config, split)
            _assert_free_motion_batch(batch, seeds)
            return evaluate_gate(model, config, batch, gates=gates)
        except TemporalQualificationError:
            raise
        except Exception as error:
            report["passed"] = False
            report["stopped_after"] = f"{split}_exception"
            report["unexpected_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            raise TemporalQualificationError(
                f"unexpected error after {split} access",
                report,
            ) from error

    for split, seeds in (
        ("selector", SELECTOR_SEEDS),
        ("confirmation", CONFIRMATION_SEEDS),
    ):
        result = materialize_and_evaluate(split, seeds)
        report["rungs"][split] = result
        if not result["passed"]:
            report["passed"] = False
            report["stopped_after"] = split
            raise TemporalQualificationError(f"{split} gate failed", report)

    final = materialize_and_evaluate("final_test", FINAL_TEST_SEEDS)
    report["final_test"] = final
    report["passed"] = bool(final["passed"])
    report["stopped_after"] = "complete" if final["passed"] else "final_test"
    if not final["passed"]:
        raise TemporalQualificationError("one-shot final test failed", report)
    return report


__all__ = [
    "ANCHOR_FRAME_INDEX",
    "ARCHITECTURE_ATTEMPT",
    "ARCHITECTURE_VERSION",
    "CONFIRMATION_SEEDS",
    "DEFAULT_GATES",
    "DEVELOPMENT_AUDIT_SEEDS",
    "DEVELOPMENT_TRAIN_SEEDS",
    "DEVELOPMENT_UPDATES",
    "FINAL_TEST_SEEDS",
    "HISTORY_FRAME_INDICES",
    "HORIZONS_SECONDS",
    "MAXIMUM_ARCHITECTURE_ATTEMPTS",
    "PERCEPTION_LATENCY_MAX_SECONDS",
    "PROCESS_MAX_RSS_BYTES",
    "PROCESS_RSS_DELTA_MAX_BYTES",
    "SELECTOR_SEEDS",
    "STATE_ONLY_ROLLOUT_LATENCY_MAX_SECONDS",
    "TARGET_FRAME_INDICES",
    "TemporalFreeMotionEstimator",
    "TemporalFreeMotionGates",
    "TemporalHistoryEstimate",
    "TemporalQualificationError",
    "accuracy_metrics",
    "evaluate_gate",
    "latency_metrics",
    "oracle_metrics",
    "run_development",
    "run_protected_qualification",
    "temporal_protocol",
    "training_objective",
    "two_second_gradient_metrics",
]
