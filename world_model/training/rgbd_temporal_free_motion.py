"""Frozen RGB-D temporal free-motion qualification rung.

This is the smallest post-RGB-D temporal system: observable metric sphere
centres are measured independently in each frame, a uniform differentiable
least-squares fit estimates position and velocity at the final timestamp, and
the existing :class:`~world_model.dynamics.AnalyticKinematics` answers future
queries.  There is no learned transition, confidence taper, optimizer, or
input-derived temporal selection.

The seed namespaces and gates below are protocol, not convenient defaults.
Importing this module never generates an episode.  Development and protected
materialization are explicit runner actions performed only after source
review.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from statistics import median
from typing import Any

import torch
from torch import Tensor, nn

from world_model.belief import BeliefFactory
from world_model.datasets.collate import collate_episodes
from world_model.dynamics import AnalyticKinematics, FreeMotionFitResult, fit_free_motion
from world_model.observations.rgb.projector import backproject_rgb_measurements
from world_model.observations.rgbd import (
    RGBDSphereCentreMeasurement,
    RGBDSphereCentreMeasurementModule,
)
from world_model.simulator.episode import generate_episode
from world_model.utils.config import OrpheusConfig

DEVELOPMENT_SEEDS = tuple(range(41_000_000, 41_000_024))
SELECTOR_SEEDS = tuple(range(42_000_000, 42_000_024))
CONFIRMATION_SEEDS = tuple(range(43_000_000, 43_000_024))
FINAL_TEST_SEEDS = tuple(range(44_000_000, 44_000_048))

HISTORY_FRAME_INDICES = tuple(range(16))
ANCHOR_FRAME_INDEX = 15
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TARGET_FRAME_INDICES = (17, 20, 25, 35, 55)

ARCHITECTURE_VERSION = 1
ARCHITECTURE_ATTEMPT = 1
MAXIMUM_ARCHITECTURE_ATTEMPTS = 2
OPTIMIZER_UPDATES = 0

# Filled after the YAML is frozen.  The dedicated runner rejects any bytewise
# configuration change rather than silently changing the qualification.
FROZEN_CONFIG_SHA256 = "5667cdb3603682b8d80a3e42793d25e36989269df1afacfa9b1028f2451101e9"

PERCEPTION_LATENCY_MAX_SECONDS = 2.0
STATE_ONLY_ROLLOUT_LATENCY_MAX_SECONDS = 0.05
PROCESS_MAX_RSS_BYTES = 2_500_000_000
PROCESS_RSS_DELTA_MAX_BYTES = 1_000_000_000


@dataclass(frozen=True)
class RGBDTemporalGates:
    """Predeclared gates applied independently to every split.

    The WLS covariance is reported only as an i.i.d.-residual diagnostic.  No
    calibrated Gaussian posterior is claimed, so no coverage or proper-score
    value is fabricated at this rung.
    """

    oracle_position_rmse_m: float = 1.0e-5
    oracle_velocity_rmse_mps: float = 1.0e-5
    oracle_simulator_position_rmse_m: float = 5.0e-4
    oracle_simulator_velocity_rmse_mps: float = 5.0e-4
    measurement_position_rmse_m: float = 0.008
    measurement_centre_rmse_pixels: float = 0.5
    measurement_radius_relative_rmse: float = 0.02
    measurement_valid_fraction: float = 1.0
    current_position_rmse_m: float = 0.010
    current_position_axis_rmse_m: float = 0.012
    current_velocity_rmse_mps: float = 0.010
    current_velocity_axis_rmse_mps: float = 0.015
    horizon_position_rmse_m: tuple[float, ...] = (0.011, 0.012, 0.015, 0.020, 0.030)
    horizon_position_axis_rmse_m: tuple[float, ...] = (
        0.014,
        0.015,
        0.018,
        0.024,
        0.035,
    )
    horizon_velocity_rmse_mps: float = 0.010
    horizon_velocity_axis_rmse_mps: float = 0.015
    early_stationary_additive_margin_m: float = 0.003
    long_stationary_rmse_ratio: float = 0.75
    zero_velocity_rmse_ratio: float = 0.60
    two_frame_velocity_rmse_ratio: float = 0.80
    rgb_only_current_velocity_rmse_ratio: float = 0.90
    rgb_only_two_second_position_rmse_ratio: float = 0.90
    missing_depth_valid_fraction: float = 0.0
    minimum_fit_support: int = 16
    maximum_fit_condition_number: float = 100.0
    residual_rmse_m: float = 0.008
    covariance_minimum_eigenvalue: float = -1.0e-8
    semigroup_position_max_abs_m: float = 1.0e-5
    semigroup_velocity_max_abs_mps: float = 1.0e-5
    minimum_input_gradient_l1: float = 1.0e-12
    maximum_input_gradient_l1: float = 1.0e8
    perception_latency_seconds: float = PERCEPTION_LATENCY_MAX_SECONDS
    state_only_rollout_latency_seconds: float = STATE_ONLY_ROLLOUT_LATENCY_MAX_SECONDS
    process_max_rss_bytes: int = PROCESS_MAX_RSS_BYTES
    process_rss_delta_bytes: int = PROCESS_RSS_DELTA_MAX_BYTES


DEFAULT_GATES = RGBDTemporalGates()


class RGBDTemporalQualificationError(RuntimeError):
    """A fail-fast stage failed, so later protected data must remain unopened."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


@dataclass(frozen=True)
class OLSResidualUncertainty:
    """Transparent covariance diagnostic under an explicit i.i.d. model.

    ``coefficient_covariance`` has shape ``[B,S,2,2,3,3]`` for fitted
    ``(position, velocity)`` coefficients and world axes.  These tensors are
    useful diagnostics, but systematic RGB-D inverse-rendering error is not
    proved Gaussian; callers must not relabel them as calibrated posterior
    uncertainty without a later coverage/proper-score qualification.
    """

    noise_covariance: Tensor
    coefficient_covariance: Tensor
    anchor_position_covariance: Tensor
    anchor_velocity_covariance: Tensor
    forecast_position_covariance: Tensor
    forecast_velocity_covariance: Tensor


@dataclass(frozen=True)
class MonocularTemporalAblation:
    """Known-radius RGB-only backprojection retained only as a control."""

    measured_positions: Tensor
    fit: FreeMotionFitResult
    rollout_positions: Tensor
    rollout_velocities: Tensor


@dataclass(frozen=True)
class RGBDTemporalEstimate:
    """Frame measurements, fitted anchor state, and analytic future queries."""

    frame_measurement: RGBDSphereCentreMeasurement
    measured_positions: Tensor
    measurement_confidence: Tensor
    measurement_valid_mask: Tensor
    sequence_valid: Tensor
    fit: FreeMotionFitResult
    uncertainty: OLSResidualUncertainty
    rollout_positions: Tensor
    rollout_velocities: Tensor


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_seed_namespaces() -> None:
    namespaces = (
        DEVELOPMENT_SEEDS,
        SELECTOR_SEEDS,
        CONFIRMATION_SEEDS,
        FINAL_TEST_SEEDS,
    )
    flattened = [seed for namespace in namespaces for seed in namespace]
    if any(not namespace for namespace in namespaces):
        raise RuntimeError("every RGB-D temporal seed namespace must be nonempty")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("RGB-D temporal seed namespaces must be disjoint")


def temporal_protocol() -> dict[str, Any]:
    """Return the immutable, hashable qualification contract."""

    _assert_seed_namespaces()
    protocol: dict[str, Any] = {
        "name": "rgbd_temporal_free_motion_v1",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": MAXIMUM_ARCHITECTURE_ATTEMPTS,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "learned_parameter_count": 0,
        "learned_parameter_bytes": 0,
        "persistent_module_state_bytes": 0,
        "runtime_anchor_state_bytes_float32": 24,
        "history_frame_indices": list(HISTORY_FRAME_INDICES),
        "anchor_frame_index": ANCHOR_FRAME_INDEX,
        "horizons_seconds": list(HORIZONS_SECONDS),
        "target_frame_indices": list(TARGET_FRAME_INDICES),
        "manifests": {
            "development": list(DEVELOPMENT_SEEDS),
            "selector": list(SELECTOR_SEEDS),
            "confirmation": list(CONFIRMATION_SEEDS),
            "final_test": list(FINAL_TEST_SEEDS),
        },
        "observation": {
            "modality": "rgbd",
            "depth_semantics": "observable_camera_z_surface_depth_zero_means_no_return",
            "sphere_radius": "known_declared_prior",
            "camera_calibration": "known_observable_input",
            "temporal_weights": "uniform_no_confidence_taper",
            "validity_use": "diagnostic_fail_closed_never_temporal_selection",
        },
        "dynamics": {
            "fit": "uniform_differentiable_2x2_free_motion_wls",
            "rollout": "AnalyticKinematics_constant_gravity_linear_drag",
            "learned_transition": False,
        },
        "uncertainty": {
            "kind": "iid_ols_residual_diagnostic",
            "calibrated_posterior_claim": False,
            "coverage_gate": None,
            "proper_score_gate": None,
            "reason": "systematic RGB-D measurement bias is not proved iid Gaussian",
        },
        "ablations": {
            "rgb_only": "known_radius_monocular_backprojection_uniform_fit_expected_degradation",
            "missing_depth": "zero_depth_must_fail_closed_without_temporal_fallback",
        },
        "gradient_gate": {
            "kind": "fixed_state_output_vector_jacobian_product",
            "world_axis_coefficients": [0.5, -0.75, 1.25],
            "targets": [
                "anchor_position",
                "anchor_velocity",
                "position_and_velocity_at_every_declared_horizon",
            ],
            "inputs": ["rgb", "depth"],
            "reason": "an exact solution may correctly have zero supervised MSE gradient",
        },
        "gates": asdict(DEFAULT_GATES),
        "execution": {
            "device": "cpu_float32",
            "torch_intraop_threads": 1,
            "evaluation_batch_size": 4,
            "perception_latency_warmups": 1,
            "perception_latency_repeats": 3,
            "state_only_rollout_latency_warmups": 1,
            "state_only_rollout_latency_repeats": 20,
        },
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)
    return protocol


class RGBDTemporalFreeMotionEstimator(nn.Module):
    """Parameter-free RGB-D history-to-state-to-rollout estimator.

    The module intentionally owns no parameters or persistent tensor buffers.
    Known radius, gravity, drag, and horizons are immutable Python values from
    the frozen protocol.  The only persistent runtime output is the explicit
    anchor position/velocity supplied to the caller (24 float32 bytes per
    sphere); the estimator itself retains no hidden temporal state.
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
        if len(image_size) != 2 or min(image_size) < 2:
            raise ValueError("image_size must contain two dimensions of at least two pixels")
        if not math.isfinite(world_radius_m) or world_radius_m <= 0.0:
            raise ValueError("world_radius_m must be finite and positive")
        if len(gravity) != 3 or not all(math.isfinite(float(value)) for value in gravity):
            raise ValueError("gravity must contain three finite values")
        if not math.isfinite(drag) or drag < 0.0:
            raise ValueError("drag must be finite and nonnegative")
        horizons = tuple(float(value) for value in horizons_seconds)
        if horizons != HORIZONS_SECONDS:
            raise ValueError(f"RGB-D temporal horizons must be {HORIZONS_SECONDS!r}")

        self.image_size = tuple(int(value) for value in image_size)
        self.world_radius_m = float(world_radius_m)
        self.gravity_values = tuple(float(value) for value in gravity)
        self.drag = float(drag)
        self.horizons_seconds = horizons
        self.measurement = RGBDSphereCentreMeasurementModule()
        self.kinematics = AnalyticKinematics()
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
            initial_drag=max(self.drag, 1.0e-8),
            initial_friction=0.2,
        )
        if tuple(self.parameters()) or tuple(self.buffers()) or self.state_dict():
            raise RuntimeError("RGB-D temporal estimator must have no learned/persistent tensors")

    def _validate_history(
        self,
        images: Tensor,
        depth: Tensor,
        world_from_camera: Tensor,
        intrinsics: Tensor,
        timestamps: Tensor,
    ) -> tuple[int, int]:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,T,3,H,W]")
        batch, frames = images.shape[:2]
        if frames != len(HISTORY_FRAME_INDICES):
            raise ValueError(f"history must contain {len(HISTORY_FRAME_INDICES)} frames")
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(f"history image size must be {self.image_size!r}")
        if images.dtype not in {torch.float32, torch.float64}:
            raise TypeError("RGB-D temporal estimation supports only float32 and float64")
        if depth.shape != (batch, frames, 1, *self.image_size):
            raise ValueError("depth must have shape [B,T,1,H,W]")
        if depth.dtype != images.dtype or depth.device != images.device:
            raise ValueError("RGB and depth must share dtype and device")
        if world_from_camera.shape != (batch, frames, 4, 4):
            raise ValueError("world_from_camera must have shape [B,T,4,4]")
        if intrinsics.shape != (batch, frames, 3, 3):
            raise ValueError("intrinsics must have shape [B,T,3,3]")
        if timestamps.shape != (batch, frames):
            raise ValueError("timestamps must have shape [B,T]")
        for name, value in (
            ("world_from_camera", world_from_camera),
            ("intrinsics", intrinsics),
            ("timestamps", timestamps),
        ):
            if value.dtype != images.dtype or value.device != images.device:
                raise ValueError(f"{name} must share RGB dtype and device")
        if not torch.isfinite(timestamps).all() or torch.any(
            timestamps[:, 1:] <= timestamps[:, :-1]
        ):
            raise ValueError("timestamps must be finite and strictly increasing")
        return batch, frames

    def _gravity(self, reference: Tensor) -> Tensor:
        return reference.new_tensor(self.gravity_values)

    def _drag(self, batch: int, reference: Tensor) -> Tensor:
        return reference.new_full((batch, 1), self.drag)

    def _objects(self, position: Tensor, velocity: Tensor) -> Any:
        if position.ndim != 3 or position.shape[1:] != (1, 3):
            raise ValueError("position must have shape [B,1,3]")
        if velocity.shape != position.shape:
            raise ValueError("velocity must have the same shape as position")
        batch = position.shape[0]
        belief = self._belief_factory.create(
            batch_size=batch,
            device=position.device,
            dtype=position.dtype,
            gravity=self.gravity_values,
        )
        return belief.objects.replace(
            active=torch.ones((batch, 1), dtype=torch.bool, device=position.device),
            object_id=torch.zeros((batch, 1), dtype=torch.int64, device=position.device),
            position=position,
            velocity=velocity,
        )

    def rollout_state(
        self,
        position: Tensor,
        velocity: Tensor,
        *,
        horizons_seconds: Sequence[float] = HORIZONS_SECONDS,
    ) -> tuple[Tensor, Tensor]:
        """Query the deployed analytic kinematics directly from one anchor."""

        horizons = tuple(float(value) for value in horizons_seconds)
        if not horizons or any(not math.isfinite(value) or value < 0.0 for value in horizons):
            raise ValueError("rollout horizons must be finite and nonnegative")
        objects = self._objects(position, velocity)
        gravity = self._gravity(position).unsqueeze(0).expand(position.shape[0], -1)
        predicted = [self.kinematics(objects, gravity, horizon) for horizon in horizons]
        return (
            torch.stack([item.position for item in predicted], dim=1),
            torch.stack([item.velocity for item in predicted], dim=1),
        )

    def semigroup_errors(self, position: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]:
        """Return direct-versus-composed two-second state disagreement."""

        objects = self._objects(position, velocity)
        gravity = self._gravity(position).unsqueeze(0).expand(position.shape[0], -1)
        first = self.kinematics(objects, gravity, 1.0)
        composed = self.kinematics(first, gravity, 1.0)
        direct = self.kinematics(objects, gravity, 2.0)
        return (
            (composed.position - direct.position).abs(),
            (composed.velocity - direct.velocity).abs(),
        )

    def _ols_uncertainty(self, fit: FreeMotionFitResult) -> OLSResidualUncertainty:
        sample_count = len(HISTORY_FRAME_INDICES)
        degrees_of_freedom = sample_count - 2
        valid = fit.valid
        identity = torch.eye(2, dtype=fit.normal_matrix.dtype, device=fit.normal_matrix.device)
        safe_normal = torch.where(valid[..., None, None], fit.normal_matrix, identity)
        # Uniform normalized weights mean N = X'X / T.  The unbiased residual
        # covariance is T/(T-2) times fit.residual_covariance, hence
        # Cov(beta) = inv(N)/T * Sigma.
        inverse_design_crossproduct = torch.linalg.inv(safe_normal) / sample_count
        noise_covariance = fit.residual_covariance * (sample_count / degrees_of_freedom)
        coefficient_covariance = (
            inverse_design_crossproduct[..., :, :, None, None]
            * noise_covariance[..., None, None, :, :]
        )
        anchor_position_covariance = coefficient_covariance[..., 0, 0, :, :]
        anchor_velocity_covariance = coefficient_covariance[..., 1, 1, :, :]

        horizons = fit.position.new_tensor(self.horizons_seconds)
        drag = fit.position.new_full(fit.position.shape[:2], self.drag)
        z = drag[..., None] * horizons
        small = drag[..., None].abs() <= 1.0e-8
        safe_drag = torch.where(small, torch.ones_like(drag[..., None]), drag[..., None])
        displacement_velocity = torch.where(
            small,
            horizons,
            -torch.expm1(-z) / safe_drag,
        )
        velocity_decay = torch.exp(-z)
        position_basis = torch.stack(
            (torch.ones_like(displacement_velocity), displacement_velocity),
            dim=-1,
        )
        velocity_basis = torch.stack(
            (torch.zeros_like(velocity_decay), velocity_decay),
            dim=-1,
        )
        position_scale = torch.einsum(
            "bshk,bskl,bshl->bsh",
            position_basis,
            inverse_design_crossproduct,
            position_basis,
        )
        velocity_scale = torch.einsum(
            "bshk,bskl,bshl->bsh",
            velocity_basis,
            inverse_design_crossproduct,
            velocity_basis,
        )
        forecast_position_covariance = (
            position_scale.permute(0, 2, 1)[..., None, None] * noise_covariance[:, None, :, :, :]
        )
        forecast_velocity_covariance = (
            velocity_scale.permute(0, 2, 1)[..., None, None] * noise_covariance[:, None, :, :, :]
        )
        valid_state = valid[..., None, None]
        valid_forecast = valid[:, None, :, None, None]
        return OLSResidualUncertainty(
            noise_covariance=torch.where(
                valid_state,
                noise_covariance,
                torch.zeros_like(noise_covariance),
            ),
            coefficient_covariance=torch.where(
                valid[..., None, None, None, None],
                coefficient_covariance,
                torch.zeros_like(coefficient_covariance),
            ),
            anchor_position_covariance=torch.where(
                valid_state,
                anchor_position_covariance,
                torch.zeros_like(anchor_position_covariance),
            ),
            anchor_velocity_covariance=torch.where(
                valid_state,
                anchor_velocity_covariance,
                torch.zeros_like(anchor_velocity_covariance),
            ),
            forecast_position_covariance=torch.where(
                valid_forecast,
                forecast_position_covariance,
                torch.zeros_like(forecast_position_covariance),
            ),
            forecast_velocity_covariance=torch.where(
                valid_forecast,
                forecast_velocity_covariance,
                torch.zeros_like(forecast_velocity_covariance),
            ),
        )

    def fit_history(
        self,
        images: Tensor,
        depth: Tensor,
        world_from_camera: Tensor,
        intrinsics: Tensor,
        timestamps: Tensor,
    ) -> tuple[
        RGBDSphereCentreMeasurement,
        Tensor,
        Tensor,
        Tensor,
        FreeMotionFitResult,
    ]:
        """Measure every frame and fit anchor state with uniform weights."""

        batch, frames = self._validate_history(
            images,
            depth,
            world_from_camera,
            intrinsics,
            timestamps,
        )
        flattened = batch * frames
        measurement = self.measurement(
            images.reshape(flattened, *images.shape[2:]),
            depth.reshape(flattened, *depth.shape[2:]),
            self.world_radius_m,
            world_from_camera.reshape(flattened, 4, 4),
            intrinsics.reshape(flattened, 3, 3),
        )
        measured_positions = measurement.world_position.reshape(batch, frames, 1, 3)
        confidence = measurement.confidence.reshape(batch, frames, 1)
        valid_mask = measurement.valid_mask.reshape(batch, frames, 1)
        # Validity/confidence are diagnostics and gates only.  They never
        # choose observations or alter temporal weights.  An unusable depth
        # row therefore makes the complete rung fail instead of silently
        # changing its sufficient statistic.
        fit = fit_free_motion(
            measured_positions,
            timestamps,
            gravity=self._gravity(images),
            drag=self._drag(batch, images),
            anchor_time=timestamps[:, -1],
            minimum_support=len(HISTORY_FRAME_INDICES),
        )
        return measurement, measured_positions, confidence, valid_mask, fit

    @staticmethod
    def _apply_sequence_validity(
        fit: FreeMotionFitResult,
        measurement_valid_mask: Tensor,
    ) -> tuple[FreeMotionFitResult, Tensor]:
        """Apply one post-fit fail-closed boundary without selecting frames.

        The uniform solve is executed unchanged.  Only after it completes do
        we require every declared frame to have usable observable depth.  A
        failed sequence cannot expose the finite but meaningless state fitted
        from substituted zero measurements.
        """

        sequence_valid = measurement_valid_mask.all(dim=1) & fit.valid
        state_mask = sequence_valid.unsqueeze(-1)
        history_mask = sequence_valid[:, None, :, None]
        covariance_mask = sequence_valid[..., None, None]
        sanitized = replace(
            fit,
            position=torch.where(state_mask, fit.position, torch.zeros_like(fit.position)),
            velocity=torch.where(state_mask, fit.velocity, torch.zeros_like(fit.velocity)),
            predicted_positions=torch.where(
                history_mask,
                fit.predicted_positions,
                torch.zeros_like(fit.predicted_positions),
            ),
            residuals=torch.where(
                history_mask,
                fit.residuals,
                torch.zeros_like(fit.residuals),
            ),
            residual_covariance=torch.where(
                covariance_mask,
                fit.residual_covariance,
                torch.zeros_like(fit.residual_covariance),
            ),
            valid=sequence_valid,
        )
        return sanitized, sequence_valid

    def rgb_only_ablation(
        self,
        measurement: RGBDSphereCentreMeasurement,
        world_from_camera: Tensor,
        intrinsics: Tensor,
        timestamps: Tensor,
    ) -> MonocularTemporalAblation:
        """Run the same fit with known-radius monocular metric scale.

        This is an ablation only.  It is never a fallback for missing depth and
        cannot qualify the primary RGB-D path.
        """

        batch, frames = timestamps.shape
        flattened = batch * frames
        centres = measurement.photometric_geometry.centres
        radius_pixels = measurement.photometric_geometry.radius_pixels
        flat_intrinsics = intrinsics.reshape(flattened, 3, 3)
        focal = 0.5 * (flat_intrinsics[:, 0, 0] + flat_intrinsics[:, 1, 1])
        inverse_depth = radius_pixels / (focal[:, None] * self.world_radius_m).clamp_min(1.0e-8)
        normalised_radius = radius_pixels / (0.5 * min(self.image_size))
        values = torch.cat(
            (
                centres,
                normalised_radius.clamp_min(1.0e-8).log().unsqueeze(-1),
                inverse_depth.unsqueeze(-1),
                centres.new_zeros((flattened, 1, 3)),
            ),
            dim=-1,
        )
        measured_positions = backproject_rgb_measurements(
            values,
            world_from_camera.reshape(flattened, 4, 4),
            flat_intrinsics,
            self.image_size,
        ).reshape(batch, frames, 1, 3)
        fit = fit_free_motion(
            measured_positions,
            timestamps,
            gravity=self._gravity(measured_positions),
            drag=self._drag(batch, measured_positions),
            anchor_time=timestamps[:, -1],
            minimum_support=len(HISTORY_FRAME_INDICES),
        )
        rollout_positions, rollout_velocities = self.rollout_state(fit.position, fit.velocity)
        return MonocularTemporalAblation(
            measured_positions=measured_positions,
            fit=fit,
            rollout_positions=rollout_positions,
            rollout_velocities=rollout_velocities,
        )

    def forward(
        self,
        images: Tensor,
        depth: Tensor,
        world_from_camera: Tensor,
        intrinsics: Tensor,
        timestamps: Tensor,
    ) -> RGBDTemporalEstimate:
        measurement, positions, confidence, valid_mask, fit = self.fit_history(
            images,
            depth,
            world_from_camera,
            intrinsics,
            timestamps,
        )
        fit, sequence_valid = self._apply_sequence_validity(fit, valid_mask)
        rollout_positions, rollout_velocities = self.rollout_state(fit.position, fit.velocity)
        rollout_mask = sequence_valid[:, None, :, None]
        rollout_positions = torch.where(
            rollout_mask,
            rollout_positions,
            torch.zeros_like(rollout_positions),
        )
        rollout_velocities = torch.where(
            rollout_mask,
            rollout_velocities,
            torch.zeros_like(rollout_velocities),
        )
        return RGBDTemporalEstimate(
            frame_measurement=measurement,
            measured_positions=positions,
            measurement_confidence=confidence,
            measurement_valid_mask=valid_mask,
            sequence_valid=sequence_valid,
            fit=fit,
            uncertainty=self._ols_uncertainty(fit),
            rollout_positions=rollout_positions,
            rollout_velocities=rollout_velocities,
        )


def new_estimator(config: OrpheusConfig) -> RGBDTemporalFreeMotionEstimator:
    """Construct the unique estimator admitted by the frozen config."""

    assert_rgbd_temporal_config(config)
    return RGBDTemporalFreeMotionEstimator(
        image_size=config.simulator.image_size,
        world_radius_m=config.simulator.radius_range[0],
        gravity=config.simulator.gravity,
        drag=config.simulator.drag_range[0],
    )


def assert_rgbd_temporal_config(config: OrpheusConfig) -> None:
    """Reject any silent change to the identifiable simulator contract."""

    simulator = config.simulator
    expected: dict[str, Any] = {
        "image_size": (64, 64),
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
    for name, required in expected.items():
        actual = getattr(simulator, name)
        if actual != required:
            raise ValueError(
                f"RGB-D temporal config requires simulator.{name}={required!r}, got {actual!r}"
            )
    if config.device.preference != "cpu" or config.device.cuda_amp:
        raise ValueError("RGB-D temporal qualification requires CPU float32 without AMP")
    if config.project.seed != DEVELOPMENT_SEEDS[0] or not config.project.deterministic:
        raise ValueError("RGB-D temporal project seed/determinism differs from protocol")
    if config.training.steps != 1 or config.training.rgb_pretrain_steps != 0:
        raise ValueError("shared config placeholder must remain one step with no RGB pretraining")
    if config.training.batch_size != 4:
        raise ValueError("RGB-D temporal evaluation batch size must remain four")
    if config.training.validation_episodes != len(DEVELOPMENT_SEEDS):
        raise ValueError("validation_episodes must match the 24-example protected split size")
    if tuple(config.evaluation.horizons_seconds) != HORIZONS_SECONDS:
        raise ValueError(f"evaluation horizons must be {HORIZONS_SECONDS!r}")
    if not config.evaluation.rgb_only:
        raise ValueError("the shared evaluator must remain on its no-oracle RGB-only path")
    derived_targets = tuple(
        ANCHOR_FRAME_INDEX + int(round(horizon * simulator.frame_rate))
        for horizon in HORIZONS_SECONDS
    )
    if derived_targets != TARGET_FRAME_INDICES:
        raise RuntimeError("RGB-D temporal target indices do not match frame rate and horizons")
    if TARGET_FRAME_INDICES[-1] != simulator.sequence_frames - 1:
        raise ValueError("the two-second target must remain the final generated frame")


def gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Return deterministic gate failures for one complete split report.

    The evaluator stores scalar keys rather than hiding evidence in one score.
    Tests can therefore exercise fail-fast semantics with seed-free mocked
    metrics before any episode namespace is materialized.
    """

    gates = DEFAULT_GATES
    failures: list[str] = []

    def require_max(key: str, maximum: float) -> None:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"{key}:missing_or_nonfinite")
        elif float(value) > maximum:
            failures.append(f"{key}:{float(value):.9g}>{maximum:.9g}")

    def require_min(key: str, minimum: float) -> None:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"{key}:missing_or_nonfinite")
        elif float(value) < minimum:
            failures.append(f"{key}:{float(value):.9g}<{minimum:.9g}")

    require_max("oracle_position_rmse_m", gates.oracle_position_rmse_m)
    require_max("oracle_velocity_rmse_mps", gates.oracle_velocity_rmse_mps)
    require_max(
        "oracle_simulator_position_rmse_m",
        gates.oracle_simulator_position_rmse_m,
    )
    require_max(
        "oracle_simulator_velocity_rmse_mps",
        gates.oracle_simulator_velocity_rmse_mps,
    )
    require_max("measurement_position_rmse_m", gates.measurement_position_rmse_m)
    require_max("measurement_centre_rmse_pixels", gates.measurement_centre_rmse_pixels)
    require_max(
        "measurement_radius_relative_rmse",
        gates.measurement_radius_relative_rmse,
    )
    require_min("measurement_valid_fraction", gates.measurement_valid_fraction)
    require_max("current_position_rmse_m", gates.current_position_rmse_m)
    require_max("current_position_axis_rmse_m", gates.current_position_axis_rmse_m)
    require_max("current_velocity_rmse_mps", gates.current_velocity_rmse_mps)
    require_max("current_velocity_axis_rmse_mps", gates.current_velocity_axis_rmse_mps)
    require_max("horizon_velocity_rmse_mps", gates.horizon_velocity_rmse_mps)
    require_max(
        "horizon_velocity_axis_rmse_mps",
        gates.horizon_velocity_axis_rmse_mps,
    )
    for horizon, position_limit, axis_limit in zip(
        HORIZONS_SECONDS,
        gates.horizon_position_rmse_m,
        gates.horizon_position_axis_rmse_m,
        strict=True,
    ):
        label = f"{horizon:.2f}"
        require_max(f"horizon_{label}_position_rmse_m", position_limit)
        require_max(f"horizon_{label}_position_axis_rmse_m", axis_limit)
        require_max(
            f"horizon_{label}_velocity_rmse_mps",
            gates.horizon_velocity_rmse_mps,
        )
        require_max(
            f"horizon_{label}_velocity_axis_rmse_mps",
            gates.horizon_velocity_axis_rmse_mps,
        )
    require_max("early_stationary_additive_regression_m", gates.early_stationary_additive_margin_m)
    require_max("long_stationary_rmse_ratio", gates.long_stationary_rmse_ratio)
    require_max("zero_velocity_rmse_ratio", gates.zero_velocity_rmse_ratio)
    require_max("two_frame_velocity_rmse_ratio", gates.two_frame_velocity_rmse_ratio)
    require_max(
        "rgb_only_current_velocity_rmse_ratio",
        gates.rgb_only_current_velocity_rmse_ratio,
    )
    require_max(
        "rgb_only_two_second_position_rmse_ratio",
        gates.rgb_only_two_second_position_rmse_ratio,
    )
    require_max("missing_depth_valid_fraction", gates.missing_depth_valid_fraction)
    require_min("minimum_fit_support", float(gates.minimum_fit_support))
    require_max("maximum_fit_condition_number", gates.maximum_fit_condition_number)
    require_max("residual_rmse_m", gates.residual_rmse_m)
    require_min(
        "covariance_minimum_eigenvalue",
        gates.covariance_minimum_eigenvalue,
    )
    require_max("semigroup_position_max_abs_m", gates.semigroup_position_max_abs_m)
    require_max(
        "semigroup_velocity_max_abs_mps",
        gates.semigroup_velocity_max_abs_mps,
    )
    require_max("perception_latency_seconds", gates.perception_latency_seconds)
    require_max(
        "state_only_rollout_latency_seconds",
        gates.state_only_rollout_latency_seconds,
    )
    require_max("process_max_rss_bytes", float(gates.process_max_rss_bytes))
    require_max("process_rss_delta_bytes", float(gates.process_rss_delta_bytes))
    for loss_name in (
        "current_position",
        "current_velocity",
        *(f"horizon_{horizon:.2f}_position" for horizon in HORIZONS_SECONDS),
        *(f"horizon_{horizon:.2f}_velocity" for horizon in HORIZONS_SECONDS),
    ):
        for modality in ("rgb", "depth"):
            key = f"gradient_l1/{loss_name}/{modality}"
            require_min(key, gates.minimum_input_gradient_l1)
            require_max(key, gates.maximum_input_gradient_l1)
    require_max("learned_parameter_count", 0.0)
    require_max("learned_parameter_bytes", 0.0)
    require_max("persistent_module_state_bytes", 0.0)
    require_max("optimizer_updates", 0.0)
    return failures


def _process_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _chunks(values: Sequence[int], size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield tuple(int(value) for value in values[start : start + size])


def _rmse(error: Tensor) -> float:
    return float(error.to(dtype=torch.float64).square().mean().sqrt())


def _axis_rmse(error: Tensor) -> Tensor:
    return error.to(dtype=torch.float64).reshape(-1, 3).square().mean(dim=0).sqrt()


def _assert_free_motion_batch(batch: Mapping[str, Any], seeds: Sequence[int]) -> None:
    events = batch["events"]
    if (
        bool(events["collision"].any())
        or bool(events["contact"].any())
        or bool(events["external_impulse"].ne(0).any())
        or bool(events["created"][:, 1:].any())
        or bool(events["removed"].any())
    ):
        raise RuntimeError(f"RGB-D temporal manifest contains an event: {tuple(seeds)!r}")
    if not bool(batch["objects"]["active"][:, :, :1].all()):
        raise RuntimeError("the single sphere must remain active for the complete episode")
    if not bool(batch["labels"]["projected_valid"][:, :, :1].all()):
        raise RuntimeError("the single sphere must remain projectable for the complete episode")


def _history_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    indices = torch.tensor(HISTORY_FRAME_INDICES, dtype=torch.int64)
    return {
        "images": batch["rgb"].index_select(1, indices),
        "depth": batch["depth"].index_select(1, indices),
        "world_from_camera": batch["camera"]["world_from_camera"].index_select(1, indices),
        "intrinsics": batch["camera"]["intrinsics"].index_select(1, indices),
        "timestamps": batch["timestamps"].index_select(1, indices),
    }


def _target_tensors(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    history = torch.tensor(HISTORY_FRAME_INDICES, dtype=torch.int64)
    future = torch.tensor(TARGET_FRAME_INDICES, dtype=torch.int64)
    return {
        "history_position": batch["objects"]["position"][:, :, :1].index_select(1, history),
        "history_centre": batch["labels"]["projected_center"][:, :, :1].index_select(1, history),
        "history_radius": batch["labels"]["apparent_radius"][:, :, :1].index_select(1, history),
        "anchor_position": batch["objects"]["position"][:, ANCHOR_FRAME_INDEX, :1],
        "anchor_velocity": batch["objects"]["velocity"][:, ANCHOR_FRAME_INDEX, :1],
        "future_position": batch["objects"]["position"][:, :, :1].index_select(1, future),
        "future_velocity": batch["objects"]["velocity"][:, :, :1].index_select(1, future),
    }


def _input_gradient_metrics(
    estimator: RGBDTemporalFreeMotionEstimator,
    batch: Mapping[str, Any],
) -> dict[str, float]:
    inputs = {name: value[:1].clone() for name, value in _history_inputs(batch).items()}
    inputs["images"].requires_grad_(True)
    inputs["depth"].requires_grad_(True)
    output = estimator(**inputs)
    coefficients = output.fit.position.new_tensor((0.5, -0.75, 1.25))

    def probe(value: Tensor) -> Tensor:
        return (value * coefficients).mean()

    losses: list[tuple[str, Tensor]] = [
        ("current_position", probe(output.fit.position)),
        ("current_velocity", probe(output.fit.velocity)),
    ]
    for index, horizon in enumerate(HORIZONS_SECONDS):
        losses.extend(
            (
                (
                    f"horizon_{horizon:.2f}_position",
                    probe(output.rollout_positions[:, index]),
                ),
                (
                    f"horizon_{horizon:.2f}_velocity",
                    probe(output.rollout_velocities[:, index]),
                ),
            )
        )
    metrics: dict[str, float] = {}
    for index, (name, loss) in enumerate(losses):
        rgb_gradient, depth_gradient = torch.autograd.grad(
            loss,
            (inputs["images"], inputs["depth"]),
            retain_graph=index + 1 < len(losses),
        )
        for modality, gradient in (("rgb", rgb_gradient), ("depth", depth_gradient)):
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"{name} has a nonfinite {modality} gradient")
            metrics[f"gradient_l1/{name}/{modality}"] = float(gradient.abs().sum())
    return metrics


@torch.no_grad()
def _latency_metrics(
    estimator: RGBDTemporalFreeMotionEstimator,
    batch: Mapping[str, Any],
) -> dict[str, float]:
    inputs = {name: value[:1] for name, value in _history_inputs(batch).items()}
    rss_before = _process_max_rss_bytes()
    warm = estimator(**inputs)
    estimator.rollout_state(warm.fit.position, warm.fit.velocity)
    rss_after_warmup = _process_max_rss_bytes()

    perception: list[float] = []
    latest: RGBDTemporalEstimate | None = None
    for _ in range(3):
        started = time.perf_counter()
        latest = estimator(**inputs)
        perception.append(time.perf_counter() - started)
    assert latest is not None
    rss_after_perception = _process_max_rss_bytes()

    rollout: list[float] = []
    for _ in range(20):
        started = time.perf_counter()
        estimator.rollout_state(latest.fit.position, latest.fit.velocity)
        rollout.append(time.perf_counter() - started)
    rss_after = _process_max_rss_bytes()
    return {
        "perception_latency_seconds": float(median(perception)),
        "state_only_rollout_latency_seconds": float(median(rollout)),
        "process_max_rss_bytes": float(rss_after),
        "process_rss_delta_bytes": float(max(0, rss_after - rss_before)),
        "warmup_rss_delta_bytes": float(max(0, rss_after_warmup - rss_before)),
        "perception_rss_delta_bytes": float(max(0, rss_after_perception - rss_after_warmup)),
    }


def evaluate_seed_manifest(
    estimator: RGBDTemporalFreeMotionEstimator,
    config: OrpheusConfig,
    seeds: Sequence[int],
    *,
    split: str,
) -> dict[str, Any]:
    """Materialize and evaluate exactly one already-authorized manifest.

    The runner owns authorization and records protected access before calling
    this function.  This function never chooses, expands, or substitutes seed
    values and performs no optimizer update.
    """

    assert_rgbd_temporal_config(config)
    manifests = {
        "development": DEVELOPMENT_SEEDS,
        "selector": SELECTOR_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
        "final_test": FINAL_TEST_SEEDS,
    }
    requested = tuple(int(seed) for seed in seeds)
    if split not in manifests or requested != manifests[split]:
        raise ValueError(f"{split!r} must use its exact frozen RGB-D temporal manifest")
    if len(requested) != len(set(requested)):
        raise ValueError("RGB-D temporal manifests must contain unique seeds")

    accumulated: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "measurement_position_error",
            "centre_pixel_error",
            "radius_relative_error",
            "measurement_valid",
            "current_position_error",
            "current_velocity_error",
            "future_position_error",
            "future_velocity_error",
            "stationary_position_error",
            "zero_velocity_error",
            "two_frame_velocity_error",
            "rgb_only_current_velocity_error",
            "rgb_only_future_position_error",
            "oracle_current_position_error",
            "oracle_current_velocity_error",
            "oracle_future_position_error",
            "oracle_future_velocity_error",
            "fit_support",
            "fit_condition",
            "fit_residual",
            "covariance_eigenvalue",
            "semigroup_position",
            "semigroup_velocity",
        )
    }
    first_batch: Mapping[str, Any] | None = None
    missing_depth_valid: list[Tensor] = []

    estimator.eval()
    for seed_chunk in _chunks(requested, config.training.batch_size):
        batch = collate_episodes([generate_episode(config, seed) for seed in seed_chunk])
        _assert_free_motion_batch(batch, seed_chunk)
        if first_batch is None:
            first_batch = batch
        inputs = _history_inputs(batch)
        targets = _target_tensors(batch)
        with torch.no_grad():
            output = estimator(**inputs)
            rgb_only = estimator.rgb_only_ablation(
                output.frame_measurement,
                inputs["world_from_camera"],
                inputs["intrinsics"],
                inputs["timestamps"],
            )
            missing = estimator(
                inputs["images"],
                torch.zeros_like(inputs["depth"]),
                inputs["world_from_camera"],
                inputs["intrinsics"],
                inputs["timestamps"],
            )
            oracle_fit = fit_free_motion(
                targets["history_position"],
                inputs["timestamps"],
                gravity=inputs["images"].new_tensor(config.simulator.gravity),
                drag=inputs["images"].new_full(
                    (inputs["images"].shape[0], 1),
                    config.simulator.drag_range[0],
                ),
                anchor_time=inputs["timestamps"][:, -1],
                minimum_support=len(HISTORY_FRAME_INDICES),
            )
            oracle_position, oracle_velocity = estimator.rollout_state(
                oracle_fit.position,
                oracle_fit.velocity,
            )
            two_frame = fit_free_motion(
                output.measured_positions[:, -2:],
                inputs["timestamps"][:, -2:],
                gravity=inputs["images"].new_tensor(config.simulator.gravity),
                drag=inputs["images"].new_full(
                    (inputs["images"].shape[0], 1),
                    config.simulator.drag_range[0],
                ),
                anchor_time=inputs["timestamps"][:, -1],
                minimum_support=2,
            )
            semigroup_position, semigroup_velocity = estimator.semigroup_errors(
                output.fit.position,
                output.fit.velocity,
            )

        batch_size, frames = inputs["images"].shape[:2]
        centre = output.frame_measurement.photometric_geometry.centres.reshape(
            batch_size,
            frames,
            1,
            2,
        )
        radius = output.frame_measurement.photometric_geometry.radius_pixels.reshape(
            batch_size,
            frames,
            1,
        )
        pixel_scale = centre.new_tensor(
            (0.5 * (config.simulator.image_size[1] - 1), 0.5 * (config.simulator.image_size[0] - 1))
        )
        append = accumulated
        append["measurement_position_error"].append(
            (output.measured_positions - targets["history_position"]).cpu()
        )
        append["centre_pixel_error"].append(
            ((centre - targets["history_centre"]) * pixel_scale).cpu()
        )
        append["radius_relative_error"].append(
            (
                (radius - targets["history_radius"]) / targets["history_radius"].clamp_min(1.0e-8)
            ).cpu()
        )
        append["measurement_valid"].append(output.measurement_valid_mask.cpu())
        append["current_position_error"].append(
            (output.fit.position - targets["anchor_position"]).cpu()
        )
        append["current_velocity_error"].append(
            (output.fit.velocity - targets["anchor_velocity"]).cpu()
        )
        append["future_position_error"].append(
            (output.rollout_positions - targets["future_position"]).cpu()
        )
        append["future_velocity_error"].append(
            (output.rollout_velocities - targets["future_velocity"]).cpu()
        )
        append["stationary_position_error"].append(
            (
                output.measured_positions[:, -1:, :, :].expand_as(targets["future_position"])
                - targets["future_position"]
            ).cpu()
        )
        append["zero_velocity_error"].append((-targets["anchor_velocity"]).cpu())
        append["two_frame_velocity_error"].append(
            (two_frame.velocity - targets["anchor_velocity"]).cpu()
        )
        append["rgb_only_current_velocity_error"].append(
            (rgb_only.fit.velocity - targets["anchor_velocity"]).cpu()
        )
        append["rgb_only_future_position_error"].append(
            (rgb_only.rollout_positions - targets["future_position"]).cpu()
        )
        append["oracle_current_position_error"].append(
            (oracle_fit.position - targets["anchor_position"]).cpu()
        )
        append["oracle_current_velocity_error"].append(
            (oracle_fit.velocity - targets["anchor_velocity"]).cpu()
        )
        append["oracle_future_position_error"].append(
            (oracle_position - targets["future_position"]).cpu()
        )
        append["oracle_future_velocity_error"].append(
            (oracle_velocity - targets["future_velocity"]).cpu()
        )
        append["fit_support"].append(output.fit.support_count.cpu())
        append["fit_condition"].append(output.fit.condition_number.cpu())
        append["fit_residual"].append(output.fit.residuals.cpu())
        covariance = torch.cat(
            (
                output.uncertainty.noise_covariance.reshape(-1, 3, 3),
                output.uncertainty.anchor_position_covariance.reshape(-1, 3, 3),
                output.uncertainty.anchor_velocity_covariance.reshape(-1, 3, 3),
                output.uncertainty.forecast_position_covariance.reshape(-1, 3, 3),
                output.uncertainty.forecast_velocity_covariance.reshape(-1, 3, 3),
            ),
            dim=0,
        )
        append["covariance_eigenvalue"].append(torch.linalg.eigvalsh(covariance).cpu())
        append["semigroup_position"].append(semigroup_position.cpu())
        append["semigroup_velocity"].append(semigroup_velocity.cpu())
        missing_depth_valid.append(missing.sequence_valid.cpu())

    if first_batch is None:
        raise RuntimeError("RGB-D temporal manifest unexpectedly produced no batches")
    tensors = {name: torch.cat(values) for name, values in accumulated.items()}
    future_position_rmse = [
        _rmse(tensors["future_position_error"][:, index]) for index in range(len(HORIZONS_SECONDS))
    ]
    future_velocity_rmse = [
        _rmse(tensors["future_velocity_error"][:, index]) for index in range(len(HORIZONS_SECONDS))
    ]
    future_position_axis = [
        float(_axis_rmse(tensors["future_position_error"][:, index]).max())
        for index in range(len(HORIZONS_SECONDS))
    ]
    future_velocity_axis = [
        float(_axis_rmse(tensors["future_velocity_error"][:, index]).max())
        for index in range(len(HORIZONS_SECONDS))
    ]
    stationary_rmse = [
        _rmse(tensors["stationary_position_error"][:, index])
        for index in range(len(HORIZONS_SECONDS))
    ]
    epsilon = torch.finfo(torch.float64).eps
    primary_velocity = _rmse(tensors["current_velocity_error"])
    zero_velocity = _rmse(tensors["zero_velocity_error"])
    two_frame_velocity = _rmse(tensors["two_frame_velocity_error"])
    rgb_only_velocity = _rmse(tensors["rgb_only_current_velocity_error"])
    rgb_only_two_second = _rmse(tensors["rgb_only_future_position_error"][:, -1])
    learned_parameters = tuple(
        parameter for parameter in estimator.parameters() if parameter.requires_grad
    )
    learned_parameter_count = sum(parameter.numel() for parameter in learned_parameters)
    learned_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in learned_parameters
    )
    persistent_module_state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in estimator.state_dict().values()
    )
    runtime_anchor_state_bytes = 2 * 3 * torch.empty((), dtype=torch.float32).element_size()
    metrics: dict[str, Any] = {
        "oracle_position_rmse_m": _rmse(tensors["oracle_current_position_error"]),
        "oracle_velocity_rmse_mps": _rmse(tensors["oracle_current_velocity_error"]),
        "oracle_simulator_position_rmse_m": _rmse(tensors["oracle_future_position_error"]),
        "oracle_simulator_velocity_rmse_mps": _rmse(tensors["oracle_future_velocity_error"]),
        "measurement_position_rmse_m": _rmse(tensors["measurement_position_error"]),
        "measurement_centre_rmse_pixels": _rmse(tensors["centre_pixel_error"]),
        "measurement_radius_relative_rmse": _rmse(tensors["radius_relative_error"]),
        "measurement_valid_fraction": float(tensors["measurement_valid"].float().mean()),
        "current_position_rmse_m": _rmse(tensors["current_position_error"]),
        "current_position_axis_rmse_m": float(_axis_rmse(tensors["current_position_error"]).max()),
        "current_velocity_rmse_mps": primary_velocity,
        "current_velocity_axis_rmse_mps": float(
            _axis_rmse(tensors["current_velocity_error"]).max()
        ),
        "horizon_velocity_rmse_mps": max(future_velocity_rmse),
        "horizon_velocity_axis_rmse_mps": max(future_velocity_axis),
        "early_stationary_additive_regression_m": max(
            future_position_rmse[index] - stationary_rmse[index] for index in (0, 1)
        ),
        "long_stationary_rmse_ratio": max(
            future_position_rmse[index] / max(stationary_rmse[index], epsilon)
            for index in (2, 3, 4)
        ),
        "zero_velocity_rmse_ratio": primary_velocity / max(zero_velocity, epsilon),
        "two_frame_velocity_rmse_ratio": primary_velocity / max(two_frame_velocity, epsilon),
        "rgb_only_current_velocity_rmse_ratio": primary_velocity / max(rgb_only_velocity, epsilon),
        "rgb_only_two_second_position_rmse_ratio": future_position_rmse[-1]
        / max(rgb_only_two_second, epsilon),
        "missing_depth_valid_fraction": float(torch.cat(missing_depth_valid).float().mean()),
        "minimum_fit_support": float(tensors["fit_support"].min()),
        "maximum_fit_condition_number": float(tensors["fit_condition"].max()),
        "residual_rmse_m": _rmse(tensors["fit_residual"]),
        "covariance_minimum_eigenvalue": float(tensors["covariance_eigenvalue"].min()),
        "semigroup_position_max_abs_m": float(tensors["semigroup_position"].max()),
        "semigroup_velocity_max_abs_mps": float(tensors["semigroup_velocity"].max()),
        "learned_parameter_count": float(learned_parameter_count),
        "learned_parameter_bytes": float(learned_parameter_bytes),
        "persistent_module_state_bytes": float(persistent_module_state_bytes),
        "runtime_anchor_state_bytes_float32": float(runtime_anchor_state_bytes),
        "optimizer_updates": 0.0,
    }
    for index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = future_position_rmse[index]
        metrics[f"horizon_{label}_position_axis_rmse_m"] = future_position_axis[index]
        metrics[f"horizon_{label}_velocity_rmse_mps"] = future_velocity_rmse[index]
        metrics[f"horizon_{label}_velocity_axis_rmse_mps"] = future_velocity_axis[index]
    metrics.update(_input_gradient_metrics(estimator, first_batch))
    metrics.update(_latency_metrics(estimator, first_batch))
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"RGB-D temporal metric {name!r} is nonfinite")
    failures = gate_failures(metrics)
    return {
        "split": split,
        "seeds": list(requested),
        "seed_manifest_sha256": _canonical_sha256(list(requested)),
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
        "optimizer_updates": 0,
        "uncertainty_claim": "iid_ols_residual_diagnostic_not_calibrated_posterior",
    }


__all__ = [
    "ANCHOR_FRAME_INDEX",
    "ARCHITECTURE_ATTEMPT",
    "ARCHITECTURE_VERSION",
    "CONFIRMATION_SEEDS",
    "DEFAULT_GATES",
    "DEVELOPMENT_SEEDS",
    "FINAL_TEST_SEEDS",
    "FROZEN_CONFIG_SHA256",
    "HISTORY_FRAME_INDICES",
    "HORIZONS_SECONDS",
    "MAXIMUM_ARCHITECTURE_ATTEMPTS",
    "MonocularTemporalAblation",
    "OLSResidualUncertainty",
    "OPTIMIZER_UPDATES",
    "RGBDTemporalEstimate",
    "RGBDTemporalFreeMotionEstimator",
    "RGBDTemporalGates",
    "RGBDTemporalQualificationError",
    "SELECTOR_SEEDS",
    "TARGET_FRAME_INDICES",
    "assert_rgbd_temporal_config",
    "gate_failures",
    "evaluate_seed_manifest",
    "new_estimator",
    "temporal_protocol",
]
