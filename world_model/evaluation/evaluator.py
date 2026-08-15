"""Held-out RGB-only evaluation with transparent physical baselines."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from world_model.belief import MotionMode
from world_model.datasets import SyntheticSphereDataset, collate_episodes
from world_model.evaluation.baselines import baseline_bundle
from world_model.evaluation.collision_conditioned import (
    CollisionConditionedForecastAccumulator,
    collision_mask_for_forecast_window,
)
from world_model.evaluation.latency import synchronize
from world_model.evaluation.occlusion_metrics import (
    OcclusionTransitionAccumulator,
)
from world_model.evaluation.parameter_metrics import OnlineParameterUpdateAccumulator
from world_model.evaluation.reports import write_evaluation_report
from world_model.evaluation.seed_protocol import (
    STANDARD_SEED_PROTOCOL,
    make_evaluation_seed_protocol,
)
from world_model.evaluation.velocity_metrics import (
    MaskedVelocityErrorAccumulator,
    OrdinaryVelocityCorrectionAccumulator,
    TemporalVelocityMeasurementAccumulator,
)
from world_model.identification import ParameterUpdateDiagnostics
from world_model.runtime import OnlineWorldModel
from world_model.simulator.sphere_world import SphereWorldConfig
from world_model.training.checkpointing import load_checkpoint
from world_model.training.event_windows import (
    ObservationWindowQueryPlan,
    observation_window_query_plan,
)
from world_model.training.loop import (
    future_predictable_mask,
    future_scene_predictable_mask,
    gather_target_slots,
    make_rgb_packet,
    match_belief_to_targets,
    move_batch_to_device,
)
from world_model.training.perturbations import perturb_belief
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import OrpheusConfig
from world_model.utils.device import DeviceInfo, select_device
from world_model.utils.seeds import seed_everything
from world_model.utils.version import SIMULATOR_VERSION

_IDENTIFIER_PARAMETERS = ("mass", "restitution", "drag", "friction", "radius")
_CURRENT_DETECTION_DISTANCE_THRESHOLD_M = 0.5
_RUNTIME_HYPOTHESIS_CANDIDATES = (
    "learned",
    "constant_velocity",
    "damped_constant_velocity",
    "ballistic_contact",
)


def enable_runtime_hypothesis_pool(
    model: OnlineWorldModel,
    config: OrpheusConfig,
) -> None:
    """Attach the explicitly requested evaluation-only runtime policy.

    Checkpoints are loaded against their original runtime configuration first.
    This post-load attachment is therefore an auditable evaluation intervention,
    not a weakened checkpoint-semantic comparison or a hidden model transfer.
    It owns no learnable state and remains outside ``WorldBelief``.
    """

    from world_model.dynamics import (
        BallisticContactDynamics,
        ConstantVelocityDynamics,
        HypothesisDynamicsPool,
        RuntimeHypothesisController,
    )

    runtime = config.runtime
    model.hypothesis_controller = RuntimeHypothesisController(
        HypothesisDynamicsPool(
            (
                model.dynamics,
                ConstantVelocityDynamics(),
                ConstantVelocityDynamics(damping=0.05),
                BallisticContactDynamics(),
            ),
            evidence_decay=runtime.hypothesis_evidence_decay,
        ),
        evidence_horizons_seconds=runtime.hypothesis_evidence_horizons_seconds,
        axis_independent_axes=runtime.hypothesis_axis_independent_axes,
        axis_prior_strength=runtime.hypothesis_axis_prior_strength,
        timestamp_tolerance_seconds=runtime.hypothesis_timestamp_tolerance_seconds,
    )


@dataclass
class _ErrorAccumulator:
    squared_sum: float = 0.0
    absolute_sum: float = 0.0
    count: int = 0
    axis_squared_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_absolute_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_count: list[int] = field(default_factory=lambda: [0, 0, 0])

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        if prediction.shape != target.shape or prediction.shape[-1] != 3:
            raise ValueError("position error inputs must have matching [...,3] shapes")
        expanded = mask
        while expanded.ndim < prediction.ndim:
            expanded = expanded.unsqueeze(-1)
        residual = prediction - target
        values = residual.masked_select(expanded.expand_as(prediction))
        if values.numel() == 0:
            return
        detached = values.detach().float().cpu()
        self.squared_sum += float(detached.square().sum())
        self.absolute_sum += float(detached.abs().sum())
        self.count += int(detached.numel())
        for axis in range(3):
            axis_values = residual[..., axis].masked_select(mask).detach().float().cpu()
            self.axis_squared_sum[axis] += float(axis_values.square().sum())
            self.axis_absolute_sum[axis] += float(axis_values.abs().sum())
            self.axis_count[axis] += int(axis_values.numel())

    def metrics(self, prefix: str) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {}
        for axis, label in enumerate(("x", "y", "z")):
            count = self.axis_count[axis]
            metrics.update(
                {
                    f"{prefix}_position_{label}_rmse_m": (
                        math.sqrt(self.axis_squared_sum[axis] / count) if count else None
                    ),
                    f"{prefix}_position_{label}_mae_m": (
                        self.axis_absolute_sum[axis] / count if count else None
                    ),
                    f"{prefix}_position_{label}_count": float(count),
                }
            )
        if self.count == 0:
            metrics.update(
                {
                    f"{prefix}_position_rmse_m": None,
                    f"{prefix}_position_mae_m": None,
                    f"{prefix}_position_coordinate_count": 0.0,
                }
            )
            return metrics
        metrics.update(
            {
                f"{prefix}_position_rmse_m": math.sqrt(self.squared_sum / self.count),
                f"{prefix}_position_mae_m": self.absolute_sum / self.count,
                f"{prefix}_position_coordinate_count": float(self.count),
            }
        )
        return metrics


@dataclass
class _CorrectionAccumulator:
    prior_error_sum: float = 0.0
    posterior_error_sum: float = 0.0
    positive: int = 0
    count: int = 0

    def update(
        self,
        prior: Tensor,
        posterior: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        prior_error = torch.linalg.vector_norm(prior - target, dim=-1)
        posterior_error = torch.linalg.vector_norm(posterior - target, dim=-1)
        prior_values = prior_error.masked_select(mask).detach().float().cpu()
        posterior_values = posterior_error.masked_select(mask).detach().float().cpu()
        if prior_values.numel() == 0:
            return
        delta = prior_values - posterior_values
        self.prior_error_sum += float(prior_values.sum())
        self.posterior_error_sum += float(posterior_values.sum())
        self.positive += int((delta > 0).sum())
        self.count += int(delta.numel())

    def metrics(self) -> dict[str, float | None]:
        if self.count == 0:
            return {
                "perturbation_prior_position_error_m": None,
                "perturbation_posterior_position_error_m": None,
                "perturbation_correction_improvement_m": None,
                "perturbation_correction_improvement_fraction": None,
                "perturbation_positive_correction_rate": None,
                "perturbation_evaluated_object_horizons": 0.0,
            }
        prior = self.prior_error_sum / self.count
        posterior = self.posterior_error_sum / self.count
        improvement = prior - posterior
        return {
            "perturbation_prior_position_error_m": prior,
            "perturbation_posterior_position_error_m": posterior,
            "perturbation_correction_improvement_m": improvement,
            "perturbation_correction_improvement_fraction": (improvement / max(prior, 1.0e-8)),
            "perturbation_positive_correction_rate": self.positive / self.count,
            "perturbation_evaluated_object_horizons": float(self.count),
        }


@dataclass
class _BinaryAccumulator:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        logits: Tensor,
        target: Tensor,
        mask: Tensor,
        *,
        threshold: float = 0.5,
    ) -> None:
        probability = logits.sigmoid().masked_select(mask).detach().cpu()
        truth = target.bool().masked_select(mask).detach().cpu()
        if probability.numel() == 0:
            return
        prediction = probability >= threshold
        self.true_positive += int((prediction & truth).sum())
        self.false_positive += int((prediction & ~truth).sum())
        self.false_negative += int((~prediction & truth).sum())
        self.true_negative += int((~prediction & ~truth).sum())

    def metrics(self, prefix: str) -> dict[str, float | None]:
        evaluated = (
            self.true_positive + self.false_positive + self.false_negative + self.true_negative
        )
        if evaluated == 0:
            return {
                f"{prefix}_precision": None,
                f"{prefix}_recall": None,
                f"{prefix}_f1": None,
                f"{prefix}_false_positive_rate": None,
                f"{prefix}_evaluated": 0.0,
            }
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = self.true_positive / precision_denominator if precision_denominator else 0.0
        recall = self.true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        false_positive_denominator = self.false_positive + self.true_negative
        return {
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
            f"{prefix}_false_positive_rate": (
                self.false_positive / false_positive_denominator
                if false_positive_denominator
                else 0.0
            ),
            f"{prefix}_evaluated": float(evaluated),
        }


@dataclass
class _CalibrationAccumulator:
    error: list[Tensor] = field(default_factory=list)
    standard_deviation: list[Tensor] = field(default_factory=list)

    def update(
        self,
        mean: Tensor,
        log_variance: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        expanded = mask
        while expanded.ndim < mean.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(mean)
        error = (mean - target).masked_select(expanded)
        standard_deviation = (0.5 * log_variance.clamp(-12.0, 8.0)).exp().masked_select(expanded)
        if error.numel() > 0:
            self.error.append(error.detach().float().cpu())
            self.standard_deviation.append(standard_deviation.detach().float().cpu())

    def metrics(self) -> dict[str, float | None]:
        if not self.error:
            return {
                "forecast_gaussian_nll": None,
                "forecast_sharpness_std": None,
                "forecast_calibration_coordinate_count": 0.0,
                "forecast_coverage_50": None,
                "forecast_coverage_80": None,
                "forecast_coverage_90": None,
                "forecast_coverage_95": None,
            }
        error = torch.cat(self.error)
        std = torch.cat(self.standard_deviation).clamp_min(1.0e-8)
        variance = std.square()
        nll = 0.5 * (error.square() / variance + variance.log() + math.log(2.0 * math.pi))
        z = error.abs() / std
        quantiles = {
            50: 0.67448975,
            80: 1.28155157,
            90: 1.64485363,
            95: 1.95996398,
        }
        metrics: dict[str, float | None] = {
            "forecast_gaussian_nll": float(nll.mean()),
            "forecast_sharpness_std": float(std.mean()),
            "forecast_calibration_coordinate_count": float(error.numel()),
        }
        for level, quantile in quantiles.items():
            metrics[f"forecast_coverage_{level}"] = float((z <= quantile).float().mean())
        return metrics


@dataclass
class _ParameterAccumulator:
    absolute_sum: dict[tuple[str, str], float] = field(default_factory=dict)
    count: dict[tuple[str, str], int] = field(default_factory=dict)

    def update(
        self,
        scope: str,
        name: str,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        if scope not in {"observable", "updated"}:
            raise ValueError(f"unknown parameter metric scope: {scope}")
        expanded = mask
        while expanded.ndim < prediction.ndim:
            expanded = expanded.unsqueeze(-1)
        values = (prediction - target).abs().masked_select(expanded.expand_as(prediction))
        if values.numel() == 0:
            return
        key = (scope, name)
        self.absolute_sum[key] = self.absolute_sum.get(key, 0.0) + float(
            values.detach().float().cpu().sum()
        )
        self.count[key] = self.count.get(key, 0) + int(values.numel())

    def metrics(self) -> dict[str, float | None]:
        results: dict[str, float | None] = {}
        for scope in ("observable", "updated"):
            for name in _IDENTIFIER_PARAMETERS:
                key = (scope, name)
                count = self.count.get(key, 0)
                results[f"{scope}_{name}_mae"] = (
                    self.absolute_sum.get(key, 0.0) / count if count else None
                )
                results[f"{scope}_{name}_count"] = float(count)
        return results


@dataclass
class _IdentifierAccumulator:
    observability_sum: dict[str, float] = field(default_factory=dict)
    observability_max: dict[str, float] = field(default_factory=dict)
    gate_sum: dict[str, float] = field(default_factory=dict)
    gate_max: dict[str, float] = field(default_factory=dict)
    active_count: dict[str, int] = field(default_factory=dict)
    update_count: dict[str, int] = field(default_factory=dict)
    diagnostic_steps: int = 0

    def update(
        self,
        diagnostics: ParameterUpdateDiagnostics,
        active: Tensor,
    ) -> None:
        expected_shape = (*active.shape, len(_IDENTIFIER_PARAMETERS))
        for name, value in (
            ("observability", diagnostics.observability),
            ("gate", diagnostics.gate),
            ("update_count", diagnostics.update_count),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"identifier {name} must have shape {expected_shape}, got {value.shape}"
                )
        self.diagnostic_steps += 1
        for parameter_index, parameter_name in enumerate(_IDENTIFIER_PARAMETERS):
            observability = diagnostics.observability[..., parameter_index].masked_select(active)
            gate = diagnostics.gate[..., parameter_index].masked_select(active)
            updates = diagnostics.update_count[..., parameter_index].masked_select(active)
            if observability.numel() == 0:
                continue
            detached_observability = observability.detach().float().cpu()
            detached_gate = gate.detach().float().cpu()
            self.observability_sum[parameter_name] = self.observability_sum.get(
                parameter_name, 0.0
            ) + float(detached_observability.sum())
            self.observability_max[parameter_name] = max(
                self.observability_max.get(parameter_name, 0.0),
                float(detached_observability.max()),
            )
            self.gate_sum[parameter_name] = self.gate_sum.get(parameter_name, 0.0) + float(
                detached_gate.sum()
            )
            self.gate_max[parameter_name] = max(
                self.gate_max.get(parameter_name, 0.0),
                float(detached_gate.max()),
            )
            self.active_count[parameter_name] = self.active_count.get(parameter_name, 0) + int(
                detached_gate.numel()
            )
            self.update_count[parameter_name] = self.update_count.get(parameter_name, 0) + int(
                updates.detach().cpu().sum()
            )

    def metrics(self) -> dict[str, float | None]:
        results: dict[str, float | None] = {
            "identifier_diagnostic_step_count": float(self.diagnostic_steps)
        }
        for name in _IDENTIFIER_PARAMETERS:
            count = self.active_count.get(name, 0)
            results[f"identifier_{name}_active_object_frame_count"] = float(count)
            results[f"identifier_{name}_observability_mean"] = (
                self.observability_sum.get(name, 0.0) / count if count else None
            )
            results[f"identifier_{name}_observability_max"] = (
                self.observability_max.get(name) if count else None
            )
            results[f"identifier_{name}_gate_mean"] = (
                self.gate_sum.get(name, 0.0) / count if count else None
            )
            results[f"identifier_{name}_gate_max"] = self.gate_max.get(name) if count else None
            results[f"identifier_{name}_update_count"] = float(self.update_count.get(name, 0))
        return results


@dataclass
class _TrackingAccumulator:
    last_prediction: dict[tuple[int, int], int] = field(default_factory=dict)
    switches: int = 0
    associations: int = 0

    def update(
        self,
        predicted_ids: Tensor,
        target_ids: Tensor,
        target_indices: Tensor,
        matched: Tensor,
        *,
        episode_offset: int,
    ) -> None:
        batch = predicted_ids.shape[0]
        for batch_index in range(batch):
            for belief_slot in (
                torch.nonzero(matched[batch_index], as_tuple=False).flatten().tolist()
            ):
                target_slot = int(target_indices[batch_index, belief_slot].detach().cpu())
                if target_slot < 0:
                    continue
                target_id = int(target_ids[batch_index, target_slot].detach().cpu())
                predicted_id = int(predicted_ids[batch_index, belief_slot].detach().cpu())
                if target_id < 0 or predicted_id < 0:
                    continue
                key = (episode_offset + batch_index, target_id)
                previous = self.last_prediction.get(key)
                if previous is not None and previous != predicted_id:
                    self.switches += 1
                self.last_prediction[key] = predicted_id
                self.associations += 1

    def metrics(self) -> dict[str, float | None]:
        return {
            "distance_gated_identity_switches": float(self.switches),
            "distance_gated_object_frame_associations": float(self.associations),
            "distance_gated_identity_switch_rate": (
                self.switches / self.associations if self.associations else None
            ),
        }


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _future_queries(
    config: OrpheusConfig,
    frame_index: int,
    total_frames: int,
) -> tuple[list[int], list[float]]:
    offsets: list[int] = []
    seconds: list[float] = []
    for horizon in config.evaluation.horizons_seconds:
        offset = max(1, int(round(horizon * config.simulator.frame_rate)))
        if frame_index + offset >= total_frames or offset in offsets:
            continue
        offsets.append(offset)
        seconds.append(offset / config.simulator.frame_rate)
    ordering = sorted(range(len(offsets)), key=offsets.__getitem__)
    return [offsets[index] for index in ordering], [seconds[index] for index in ordering]


def _horizon_key(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _collision_logits_for_observation_windows(
    event_logits: Tensor,
    query_plan: ObservationWindowQueryPlan,
) -> Tensor:
    """Return collision logits aligned to each target frame's label window."""

    return query_plan.select_target_endpoints(event_logits)[..., MotionMode.COLLISION]


def _distance_gate_matches(
    prediction: Tensor,
    aligned_target: Tensor,
    assignment_mask: Tensor,
    *,
    threshold_m: float,
) -> Tensor:
    if threshold_m <= 0:
        raise ValueError("detection distance threshold must be positive")
    if prediction.shape != aligned_target.shape or prediction.shape[-1] != 3:
        raise ValueError("detection positions must share shape [B,N,3]")
    if assignment_mask.shape != prediction.shape[:-1]:
        raise ValueError("assignment mask must have shape [B,N]")
    distance = torch.linalg.vector_norm(prediction - aligned_target, dim=-1)
    return assignment_mask & torch.isfinite(distance) & (distance <= threshold_m)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def evaluate_checkpoint(
    config: OrpheusConfig,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    seed_protocol: str = STANDARD_SEED_PROTOCOL,
    seed_offset: int | None = None,
    output_dir: str | Path | None = None,
    device_info: DeviceInfo | None = None,
    runtime_hypothesis_pool: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate a trusted local checkpoint on held-out RGB episodes.

    Simulator state and parameters are used for metrics and the explicitly
    labelled oracle-parameter analytic baseline only.  Every model correction
    consumes RGB plus known camera calibration.
    """

    config.validate()
    if not config.evaluation.rgb_only:
        raise ValueError("Milestone 1 evaluation must set evaluation.rgb_only=true")
    if config.runtime.modality != "rgb" or config.runtime.enable_debug_oracle:
        raise ValueError("held-out RGB evaluation forbids debug_oracle runtime input")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    resolved_device = device_info or select_device(config.device.preference)
    device = resolved_device.device
    seed_everything(
        config.project.seed + 50_000,
        deterministic=config.project.deterministic,
    )
    model = OnlineWorldModel.from_config(config, device=device)
    payload = load_checkpoint(
        checkpoint,
        model=model,
        # Keep unused optimizer moments off the accelerator; model loading
        # copies only the needed weights to their owning (possibly hybrid)
        # devices.
        map_location="cpu",
        expected_config=config,
    )
    if runtime_hypothesis_pool:
        enable_runtime_hypothesis_pool(model, config)
    model.eval()

    def report_progress(stage: str, **values: Any) -> None:
        if progress_callback is not None:
            progress_callback({"stage": stage, **values})

    checkpoint_training_config = payload["config"].get("training")
    if not isinstance(checkpoint_training_config, Mapping):
        raise ValueError("checkpoint config is missing its training mapping")
    checkpoint_validation_episodes = checkpoint_training_config.get("validation_episodes")
    if (
        not isinstance(checkpoint_validation_episodes, int)
        or isinstance(checkpoint_validation_episodes, bool)
        or checkpoint_validation_episodes < 0
    ):
        raise ValueError(
            "checkpoint config training.validation_episodes must be a nonnegative integer"
        )
    resolved_seed_protocol = make_evaluation_seed_protocol(
        name=seed_protocol,
        split=split,
        episode_count=config.evaluation.episodes,
        training_validation_episodes=checkpoint_validation_episodes,
        seed_offset=seed_offset,
    )
    dataset = SyntheticSphereDataset(
        config,
        split=resolved_seed_protocol.split,
        seeds=resolved_seed_protocol.manifest,
        memory_cache=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(
            config.training.batch_size,
            max(1, config.evaluation.episodes),
        ),
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_episodes,
        drop_last=False,
    )
    report_progress(
        "started",
        episodes=config.evaluation.episodes,
        batches=len(loader),
        device=str(device),
        runtime_hypothesis_pool=runtime_hypothesis_pool,
    )

    current_error = _ErrorAccumulator()
    current_velocity_error = MaskedVelocityErrorAccumulator()
    ordinary_velocity_correction = OrdinaryVelocityCorrectionAccumulator()
    temporal_velocity_measurements = TemporalVelocityMeasurementAccumulator()
    forecast_errors: dict[tuple[str, str], _ErrorAccumulator] = {}
    correction = _CorrectionAccumulator()
    events = _BinaryAccumulator()
    calibration = _CalibrationAccumulator()
    collision_conditioned_forecasts = CollisionConditionedForecastAccumulator()
    parameters = _ParameterAccumulator()
    directional_parameters = OnlineParameterUpdateAccumulator()
    identifier_metrics = _IdentifierAccumulator()
    tracking = _TrackingAccumulator()
    occlusion_transitions = OcclusionTransitionAccumulator()
    global_latencies: list[float] = []
    fast_latencies: list[float] = []
    rollout_latencies: list[float] = []
    uncertainty_visible: list[float] = []
    uncertainty_occluded: list[float] = []
    uncertainty_contraction: list[float] = []
    nonfinite_outputs = 0
    evaluated_episodes = 0
    perturbation_updates = 0
    target_object_frames = 0
    predicted_object_frames = 0
    matched_object_frames = 0
    distance_gated_matched_object_frames = 0
    trajectory_change_point_count = 0
    trajectory_change_point_inspected_object_frames = 0
    lateral_intervention_gain_sum = 0.0
    lateral_intervention_feature_count = 0
    lateral_intervention_gain_above_half_count = 0
    gravity_intervention_gain_sum = 0.0
    gravity_intervention_feature_count = 0
    gravity_intervention_gain_above_half_count = 0
    runtime_hypothesis_forecast_anchor_count = 0
    runtime_hypothesis_axis_selection_count = {
        axis: [0 for _ in _RUNTIME_HYPOTHESIS_CANDIDATES]
        for axis in config.runtime.hypothesis_axis_independent_axes
    }
    forecast_target_count: dict[str, int] = {}
    forecast_tracked_count: dict[str, int] = {}
    forecast_active_count: dict[str, int] = {}
    forecast_predictable_target_count: dict[str, int] = {}
    forecast_censored_tracked_count: dict[str, int] = {}

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            rgb = batch["rgb"]
            if rgb.ndim != 5:
                raise ValueError("evaluation DataLoader must emit [B,T,3,H,W]")
            batch_size, total_frames = rgb.shape[:2]
            model.reset(batch_size=batch_size)
            perturbation_frame = max(1, total_frames // 3)
            anchor_stride = max(1, total_frames // 8)
            diagnostic_offset = 0

            for frame_index in range(total_frames):
                packet = make_rgb_packet(batch, frame_index)
                prior_rollout = None
                prior_variance = None
                ordinary_velocity_prior = None
                perturb_offsets: list[int] = []
                perturb_seconds: list[float] = []
                if model.belief is not None and frame_index != perturbation_frame:
                    requested = model.belief.timestamp.new_full(
                        model.belief.timestamp.shape,
                        packet.timestamp,
                    )
                    ordinary_velocity_prior = model.dynamics.predict(
                        model.belief,
                        requested - model.belief.timestamp,
                    )
                if model.belief is not None and frame_index == perturbation_frame:
                    source_belief = perturb_belief(
                        model.belief,
                        position_std=config.evaluation.perturbation_position_std,
                        velocity_std=config.evaluation.perturbation_velocity_std,
                        covariance_log_bias=0.5,
                    )
                    # Perturb the carried posterior, not the already
                    # timestamp-advanced prior.  Runtime ingest must retain
                    # the positive dt used by velocity-aware correction.
                    model.state.belief = source_belief
                    requested = source_belief.timestamp.new_full(
                        source_belief.timestamp.shape, packet.timestamp
                    )
                    prior = model.dynamics.predict(
                        source_belief,
                        requested - source_belief.timestamp,
                    )
                    perturb_offsets, perturb_seconds = _future_queries(
                        config, frame_index, total_frames
                    )
                    if perturb_seconds:
                        prior_rollout = model.dynamics.rollout(
                            prior,
                            perturb_seconds,
                            return_events=False,
                        )
                        prior_variance = (
                            prior.objects.fast_log_variance[..., :3].exp().mean(dim=-1).sqrt()
                        )
                        perturbation_updates += batch_size

                pre_ingest_parameters: dict[str, Tensor] | None = None
                if model.belief is not None:
                    pre_ingest_parameters = {
                        "restitution": model.belief.objects.restitution.clone(),
                        "drag": model.belief.objects.drag.clone(),
                        "active": model.belief.objects.active.clone(),
                        "object_id": model.belief.objects.object_id.clone(),
                    }
                synchronize(device)
                update_started = time.perf_counter()
                belief = model.ingest(packet)
                synchronize(device)
                update_elapsed_ms = (time.perf_counter() - update_started) * 1000.0
                last_measurements = model.last_measurements
                if last_measurements is not None:
                    expected_measurement_timestamp = last_measurements.timestamp.new_full(
                        last_measurements.timestamp.shape,
                        packet.timestamp,
                    )
                    if not torch.allclose(
                        last_measurements.timestamp,
                        expected_measurement_timestamp,
                        atol=1.0e-6,
                        rtol=0.0,
                    ):
                        # SKIP leaves the previous diagnostics snapshot in the
                        # runtime; never count that stale measurement twice.
                        last_measurements = None
                temporal_velocity_measurements.update(last_measurements)
                if last_measurements is not None:
                    change_point_mask = last_measurements.auxiliary.get(
                        "trajectory_change_point_mask"
                    )
                    if change_point_mask is not None:
                        if change_point_mask.dtype != torch.bool or change_point_mask.ndim != 2:
                            raise ValueError("trajectory_change_point_mask must be boolean [B,N]")
                        trajectory_change_point_count += int(change_point_mask.sum().detach().cpu())
                        eligible_mask = last_measurements.auxiliary.get(
                            "trajectory_change_point_eligible_mask"
                        )
                        if (
                            eligible_mask is None
                            or eligible_mask.shape != change_point_mask.shape
                            or eligible_mask.dtype != torch.bool
                        ):
                            raise ValueError(
                                "trajectory_change_point_eligible_mask must be boolean [B,N]"
                            )
                        trajectory_change_point_inspected_object_frames += int(
                            eligible_mask.sum().detach().cpu()
                        )
                    lateral_gain = last_measurements.auxiliary.get(
                        "trajectory_lateral_intervention_gain"
                    )
                    lateral_valid = last_measurements.auxiliary.get(
                        "trajectory_lateral_intervention_feature_valid_mask"
                    )
                    if lateral_gain is not None or lateral_valid is not None:
                        if (
                            lateral_gain is None
                            or lateral_valid is None
                            or lateral_gain.shape != lateral_valid.shape
                            or lateral_valid.dtype != torch.bool
                        ):
                            raise ValueError(
                                "lateral intervention gain/valid diagnostics "
                                "must be aligned [B,N] tensors"
                            )
                        selected_gain = lateral_gain.masked_select(lateral_valid)
                        lateral_intervention_gain_sum += float(selected_gain.sum().detach().cpu())
                        lateral_intervention_feature_count += int(selected_gain.numel())
                        lateral_intervention_gain_above_half_count += int(
                            (selected_gain >= 0.5).sum().detach().cpu()
                        )
                    gravity_gain = last_measurements.auxiliary.get(
                        "trajectory_gravity_intervention_gain"
                    )
                    gravity_valid = last_measurements.auxiliary.get(
                        "trajectory_gravity_intervention_feature_valid_mask"
                    )
                    if gravity_gain is not None or gravity_valid is not None:
                        if (
                            gravity_gain is None
                            or gravity_valid is None
                            or gravity_gain.shape != gravity_valid.shape
                            or gravity_valid.dtype != torch.bool
                        ):
                            raise ValueError(
                                "gravity intervention gain/valid diagnostics "
                                "must be aligned [B,N] tensors"
                            )
                        selected_gain = gravity_gain.masked_select(gravity_valid)
                        gravity_intervention_gain_sum += float(selected_gain.sum().detach().cpu())
                        gravity_intervention_feature_count += int(selected_gain.numel())
                        gravity_intervention_gain_above_half_count += int(
                            (selected_gain >= 0.5).sum().detach().cpu()
                        )
                if model.diagnostics.oracle_used:
                    raise RuntimeError(
                        "oracle diagnostics detected during claimed RGB-only evaluation"
                    )
                target_position = batch["objects"]["position"][:, frame_index]
                target_active = batch["objects"]["active"][:, frame_index].bool()
                target_indices, matched = match_belief_to_targets(
                    belief, target_position, target_active
                )
                target_object_frames += int(target_active.sum().detach().cpu())
                predicted_object_frames += int(belief.objects.active.sum().detach().cpu())
                matched_object_frames += int(matched.sum().detach().cpu())
                aligned_position = gather_target_slots(target_position, target_indices)
                distance_gated_matched = _distance_gate_matches(
                    belief.objects.position,
                    aligned_position,
                    matched,
                    threshold_m=_CURRENT_DETECTION_DISTANCE_THRESHOLD_M,
                )
                distance_gated_matched_object_frames += int(
                    distance_gated_matched.sum().detach().cpu()
                )
                current_error.update(
                    belief.objects.position,
                    aligned_position,
                    matched,
                )
                target_velocity = batch["objects"]["velocity"][:, frame_index]
                aligned_velocity = gather_target_slots(target_velocity, target_indices)
                current_velocity_error.update(
                    belief.objects.velocity,
                    aligned_velocity,
                    distance_gated_matched,
                )
                if ordinary_velocity_prior is not None and last_measurements is not None:
                    same_persistent_slot = (
                        ordinary_velocity_prior.objects.active
                        & belief.objects.active
                        & (ordinary_velocity_prior.objects.object_id == belief.objects.object_id)
                    )
                    ordinary_velocity_correction.update(
                        ordinary_velocity_prior.objects.velocity,
                        belief.objects.velocity,
                        aligned_velocity,
                        distance_gated_matched & same_persistent_slot,
                    )
                tracking.update(
                    belief.objects.object_id,
                    batch["objects"]["id"][:, frame_index],
                    target_indices,
                    distance_gated_matched,
                    episode_offset=evaluated_episodes,
                )

                visible_fraction = gather_target_slots(
                    batch["objects"]["visible_fraction"][:, frame_index].unsqueeze(-1),
                    target_indices,
                ).squeeze(-1)
                position_std = (0.5 * belief.objects.fast_log_variance[..., :3]).exp().mean(dim=-1)
                occlusion_transitions.update_frame(
                    predicted_ids=belief.objects.object_id,
                    predicted_active=belief.objects.active,
                    position_std_m=position_std,
                    target_ids=batch["objects"]["id"][:, frame_index],
                    target_active=target_active,
                    target_visible_fraction=batch["objects"]["visible_fraction"][:, frame_index],
                    matched_target_indices=target_indices,
                    reliable_visible_matches=distance_gated_matched,
                    episode_offset=evaluated_episodes,
                )
                visible_mask = distance_gated_matched & (visible_fraction >= 0.5)
                occluded_mask = distance_gated_matched & (visible_fraction <= 0.05)
                if visible_mask.any():
                    uncertainty_visible.extend(
                        position_std.masked_select(visible_mask).detach().float().cpu().tolist()
                    )
                if occluded_mask.any():
                    uncertainty_occluded.extend(
                        position_std.masked_select(occluded_mask).detach().float().cpu().tolist()
                    )
                if prior_variance is not None:
                    valid = matched & belief.objects.active
                    if valid.any():
                        posterior_variance = (
                            belief.objects.fast_log_variance[..., :3].exp().mean(dim=-1).sqrt()
                        )
                        uncertainty_contraction.extend(
                            (prior_variance - posterior_variance)
                            .masked_select(valid)
                            .detach()
                            .float()
                            .cpu()
                            .tolist()
                        )

                aligned_radius = gather_target_slots(
                    batch["objects"]["radius"][:, frame_index],
                    target_indices,
                )
                aligned_mass = gather_target_slots(
                    batch["objects"]["mass"][:, frame_index],
                    target_indices,
                )
                aligned_restitution = gather_target_slots(
                    batch["objects"]["restitution"][:, frame_index],
                    target_indices,
                )
                aligned_drag = gather_target_slots(
                    batch["objects"]["drag"][:, frame_index],
                    target_indices,
                )
                aligned_friction = gather_target_slots(
                    batch["objects"]["friction"][:, frame_index],
                    target_indices,
                )
                identifier_diagnostics = (
                    model.identifier.last_diagnostics if model.identifier is not None else None
                )
                if identifier_diagnostics is not None:
                    identifier_metrics.update(
                        identifier_diagnostics,
                        belief.objects.active,
                    )
                    predictions = (
                        belief.objects.mass,
                        belief.objects.restitution,
                        belief.objects.drag,
                        belief.objects.friction,
                        belief.objects.radius,
                    )
                    targets = (
                        aligned_mass,
                        aligned_restitution,
                        aligned_drag,
                        aligned_friction,
                        aligned_radius,
                    )
                    for parameter_index, (parameter_name, prediction, target) in enumerate(
                        zip(
                            _IDENTIFIER_PARAMETERS,
                            predictions,
                            targets,
                            strict=True,
                        )
                    ):
                        runtime_observable = (
                            identifier_diagnostics.observability[..., parameter_index] > 0.0
                        )
                        runtime_updated = identifier_diagnostics.update_count[
                            ..., parameter_index
                        ].bool()
                        parameters.update(
                            "observable",
                            parameter_name,
                            prediction,
                            target,
                            distance_gated_matched & runtime_observable,
                        )
                        parameters.update(
                            "updated",
                            parameter_name,
                            prediction,
                            target,
                            distance_gated_matched & runtime_updated,
                        )
                    if pre_ingest_parameters is not None:
                        same_persistent_slot = (
                            pre_ingest_parameters["active"]
                            & belief.objects.active
                            & (pre_ingest_parameters["object_id"] == belief.objects.object_id)
                        )
                        collision_informative = gather_target_slots(
                            batch["events"]["collision"][:, frame_index].unsqueeze(-1),
                            target_indices,
                        ).squeeze(-1)
                        free_steps = model.observability.config.minimum_free_steps
                        free_window_start = frame_index - free_steps + 1
                        drag_informative_target = torch.zeros_like(target_active)
                        if free_window_start >= 0:
                            active_history = batch["objects"]["active"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            contact_history = batch["events"]["contact"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            collision_history = batch["events"]["collision"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            actuation_history = batch["events"]["externally_actuated"][
                                :, free_window_start : frame_index + 1
                            ].bool()
                            speed = torch.linalg.vector_norm(
                                batch["objects"]["velocity"][:, frame_index],
                                dim=-1,
                            )
                            drag_informative_target = (
                                active_history.all(dim=1)
                                & ~contact_history.any(dim=1)
                                & ~collision_history.any(dim=1)
                                & ~actuation_history.any(dim=1)
                                & (speed >= config.model.identification.drag_speed_threshold)
                                & (batch["objects"]["visible_fraction"][:, frame_index] >= 0.5)
                            )
                        drag_informative = gather_target_slots(
                            drag_informative_target.unsqueeze(-1),
                            target_indices,
                        ).squeeze(-1)
                        common_directional_mask = distance_gated_matched & same_persistent_slot
                        directional_parameters.update(
                            "restitution",
                            pre_ingest_parameters["restitution"],
                            belief.objects.restitution,
                            aligned_restitution,
                            common_directional_mask
                            & identifier_diagnostics.update_count[..., 1].bool()
                            & collision_informative,
                        )
                        directional_parameters.update(
                            "drag",
                            pre_ingest_parameters["drag"],
                            belief.objects.drag,
                            aligned_drag,
                            common_directional_mask
                            & identifier_diagnostics.update_count[..., 2].bool()
                            & drag_informative,
                        )

                run_forecast = frame_index % anchor_stride == 0 or frame_index == perturbation_frame
                frame_offsets, query_seconds = _future_queries(config, frame_index, total_frames)
                if run_forecast and query_seconds:
                    event_query_plan = observation_window_query_plan(
                        frame_offsets,
                        frame_rate=config.simulator.frame_rate,
                    )
                    synchronize(device)
                    rollout_started = time.perf_counter()
                    # Route scored forecasts through the online runtime.  This
                    # is normally equivalent to the learned rollout, but is
                    # essential for an explicitly enabled runtime hypothesis
                    # policy: calling ``model.dynamics`` here would attach the
                    # policy yet silently score only its learned candidate.
                    trajectory = model.predict(event_query_plan.query_seconds)
                    if runtime_hypothesis_pool:
                        axis_indices = trajectory.auxiliary.get("hypothesis_axis_index")
                        # Before the first exact-due RGB observation there is
                        # deliberately no selector evidence, and normal
                        # runtime falls back to the learned rollout. Record
                        # that explicit default rather than pretending an
                        # axis-selection tensor already exists.
                        if axis_indices is None:
                            axis_indices = torch.zeros(
                                batch_size,
                                3,
                                device=belief.device,
                                dtype=torch.int64,
                            )
                        if not isinstance(axis_indices, Tensor) or axis_indices.shape != (batch_size, 3):
                            raise RuntimeError(
                                "runtime hypothesis forecast must expose "
                                "hypothesis_axis_index [B,3]"
                            )
                        if axis_indices.dtype != torch.int64 or torch.any(axis_indices < 0) or torch.any(
                            axis_indices >= len(_RUNTIME_HYPOTHESIS_CANDIDATES)
                        ):
                            raise RuntimeError("runtime hypothesis axis index is invalid")
                        runtime_hypothesis_forecast_anchor_count += batch_size
                        for axis in config.runtime.hypothesis_axis_independent_axes:
                            selected_counts = torch.bincount(
                                axis_indices[:, axis],
                                minlength=len(_RUNTIME_HYPOTHESIS_CANDIDATES),
                            ).detach().cpu().tolist()
                            for candidate_index, count in enumerate(selected_counts):
                                runtime_hypothesis_axis_selection_count[axis][candidate_index] += int(count)
                    synchronize(device)
                    rollout_latencies.append((time.perf_counter() - rollout_started) * 1000.0)
                    model_positions = event_query_plan.select_target_endpoints(trajectory.positions)
                    model_log_variance = event_query_plan.select_target_endpoints(
                        trajectory.fast_log_variance
                    )
                    model_active_mask = event_query_plan.select_target_endpoints(
                        trajectory.active_mask
                    )
                    model_collision_logits = (
                        None
                        if trajectory.event_logits is None
                        else _collision_logits_for_observation_windows(
                            trajectory.event_logits,
                            event_query_plan,
                        )
                    )
                    oracle_drag = gather_target_slots(
                        batch["objects"]["drag"][:, frame_index],
                        target_indices,
                    )
                    query_tensor = (
                        belief.timestamp.new_tensor(query_seconds)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )
                    baselines = baseline_bundle(
                        belief.objects.position,
                        belief.objects.velocity,
                        query_tensor,
                        gravity=belief.gravity,
                        default_drag=sum(config.simulator.drag_range) / 2.0,
                        oracle_drag=oracle_drag,
                    )
                    for query_index, frame_offset in enumerate(frame_offsets):
                        target_frame = frame_index + frame_offset
                        future_target = gather_target_slots(
                            batch["objects"]["position"][:, target_frame],
                            target_indices,
                        )
                        future_active = (
                            gather_target_slots(
                                batch["objects"]["active"][:, target_frame].unsqueeze(-1),
                                target_indices,
                            )
                            .squeeze(-1)
                            .bool()
                        )
                        horizon = _horizon_key(query_seconds[query_index])
                        common_valid = matched & future_active
                        predictable = future_predictable_mask(
                            batch,
                            anchor_index=frame_index,
                            target_index=target_frame,
                            target_indices=target_indices,
                        )
                        scene_predictable = future_scene_predictable_mask(
                            batch,
                            anchor_index=frame_index,
                            target_index=target_frame,
                        )
                        point_valid = common_valid & predictable
                        forecast_target_count[horizon] = forecast_target_count.get(
                            horizon, 0
                        ) + int(batch["objects"]["active"][:, target_frame].sum().detach().cpu())
                        forecast_tracked_count[horizon] = forecast_tracked_count.get(
                            horizon, 0
                        ) + int(common_valid.sum().detach().cpu())
                        forecast_active_count[horizon] = forecast_active_count.get(
                            horizon, 0
                        ) + int(
                            (common_valid & model_active_mask[:, query_index]).sum().detach().cpu()
                        )
                        forecast_predictable_target_count[horizon] = (
                            forecast_predictable_target_count.get(horizon, 0)
                            + int(
                                (
                                    batch["objects"]["active"][:, target_frame].bool()
                                    & scene_predictable[:, None]
                                )
                                .sum()
                                .detach()
                                .cpu()
                            )
                        )
                        forecast_censored_tracked_count[horizon] = (
                            forecast_censored_tracked_count.get(horizon, 0)
                            + int((common_valid & ~predictable).sum().detach().cpu())
                        )
                        forecast_errors.setdefault(("model", horizon), _ErrorAccumulator()).update(
                            model_positions[:, query_index],
                            future_target,
                            point_valid,
                        )
                        for baseline_name, positions in baselines.items():
                            forecast_errors.setdefault(
                                (baseline_name, horizon),
                                _ErrorAccumulator(),
                            ).update(
                                positions[:, query_index],
                                future_target,
                                point_valid,
                            )
                        # Calibration remains a proper stochastic diagnostic:
                        # unseen interventions are omitted from point RMSE but
                        # still test whether predictive uncertainty widens.
                        calibration.update(
                            model_positions[:, query_index],
                            model_log_variance[:, query_index, :, :3],
                            future_target,
                            common_valid,
                        )
                        collision_during_window = collision_mask_for_forecast_window(
                            batch["events"]["collision"],
                            anchor_frame=frame_index,
                            target_frame=target_frame,
                        )
                        aligned_collision_during_window = (
                            gather_target_slots(
                                collision_during_window.unsqueeze(-1),
                                target_indices,
                            )
                            .squeeze(-1)
                            .bool()
                        )
                        collision_conditioned_forecasts.update(
                            horizon=horizon,
                            predictions={
                                "model": model_positions[:, query_index],
                                **{
                                    baseline_name: positions[:, query_index]
                                    for baseline_name, positions in baselines.items()
                                },
                            },
                            target=future_target,
                            valid_mask=point_valid,
                            collision_mask=aligned_collision_during_window,
                        )
                        if model_collision_logits is not None:
                            # The selected endpoint logit covers exactly
                            # [target_frame - 1, target_frame], matching the
                            # simulator event label rather than a horizon
                            # prefix or gap between requested horizons.
                            collision_target = gather_target_slots(
                                batch["events"]["collision"][:, target_frame].unsqueeze(-1),
                                target_indices,
                            ).squeeze(-1)
                            events.update(
                                model_collision_logits[:, query_index],
                                collision_target,
                                point_valid,
                            )
                    if not bool(
                        torch.isfinite(trajectory.positions).all()
                        and torch.isfinite(trajectory.fast_log_variance).all()
                    ):
                        nonfinite_outputs += 1

                if (
                    prior_rollout is not None
                    and perturb_seconds
                    and query_seconds == perturb_seconds
                ):
                    posterior_rollout = model.dynamics.rollout(
                        belief, perturb_seconds, return_events=False
                    )
                    for query_index, frame_offset in enumerate(perturb_offsets):
                        target_frame = frame_index + frame_offset
                        future_target = gather_target_slots(
                            batch["objects"]["position"][:, target_frame],
                            target_indices,
                        )
                        future_active = (
                            gather_target_slots(
                                batch["objects"]["active"][:, target_frame].unsqueeze(-1),
                                target_indices,
                            )
                            .squeeze(-1)
                            .bool()
                        )
                        valid = matched & future_active
                        valid &= future_predictable_mask(
                            batch,
                            anchor_index=frame_index,
                            target_index=target_frame,
                            target_indices=target_indices,
                        )
                        correction.update(
                            prior_rollout.positions[:, query_index],
                            posterior_rollout.positions[:, query_index],
                            future_target,
                            valid,
                        )

                if not bool(
                    torch.isfinite(belief.objects.position).all()
                    and torch.isfinite(belief.objects.fast_log_variance).all()
                ):
                    nonfinite_outputs += 1

                new_diagnostics = model.diagnostics.records[diagnostic_offset:]
                diagnostic_offset = len(model.diagnostics.records)
                for diagnostic in new_diagnostics:
                    if diagnostic.observation_mode in {
                        "GLOBAL_DISCOVERY",
                        "RECOVERY",
                    }:
                        global_latencies.append(update_elapsed_ms)
                    elif diagnostic.observation_mode == "FAST_ROI":
                        fast_latencies.append(update_elapsed_ms)
            evaluated_episodes += batch_size
            report_progress(
                "batch_complete",
                batch=batch_index,
                batches=len(loader),
                evaluated_episodes=evaluated_episodes,
            )

    metrics: dict[str, Any] = {}
    metrics.update(current_error.metrics("posterior_current"))
    metrics.update(current_velocity_error.metrics("posterior_current"))
    metrics.update(ordinary_velocity_correction.metrics())
    metrics.update(temporal_velocity_measurements.metrics())
    for (method, horizon), accumulator in sorted(forecast_errors.items()):
        metrics.update(accumulator.metrics(f"{method}@{horizon}"))
    metrics.update(collision_conditioned_forecasts.metrics())
    metrics.update(correction.metrics())
    metrics.update(events.metrics("collision"))
    metrics.update(calibration.metrics())
    metrics.update(parameters.metrics())
    directional_parameter_metrics = directional_parameters.metrics()
    metrics.update(directional_parameter_metrics)
    metrics.update(identifier_metrics.metrics())
    metrics.update(tracking.metrics())
    metrics.update(occlusion_transitions.metrics())
    detection_threshold_label = f"{_CURRENT_DETECTION_DISTANCE_THRESHOLD_M:.3f}m"
    metrics.update(
        {
            "current_assignment_target_coverage": (
                matched_object_frames / target_object_frames if target_object_frames else None
            ),
            "current_assignment_prediction_coverage": (
                matched_object_frames / predicted_object_frames if predicted_object_frames else None
            ),
            f"current_detection_recall@{detection_threshold_label}": (
                distance_gated_matched_object_frames / target_object_frames
                if target_object_frames
                else None
            ),
            f"current_detection_precision@{detection_threshold_label}": (
                distance_gated_matched_object_frames / predicted_object_frames
                if predicted_object_frames
                else None
            ),
            "target_object_frames": float(target_object_frames),
            "predicted_object_frames": float(predicted_object_frames),
            "assignment_matched_object_frames": float(matched_object_frames),
            "trajectory_change_point_count": float(trajectory_change_point_count),
            "trajectory_change_point_inspected_object_frames": float(
                trajectory_change_point_inspected_object_frames
            ),
            "trajectory_change_point_rate": (
                trajectory_change_point_count / trajectory_change_point_inspected_object_frames
                if trajectory_change_point_inspected_object_frames
                else None
            ),
            "lateral_intervention_feature_count": float(lateral_intervention_feature_count),
            "lateral_intervention_mean_soft_gain": (
                lateral_intervention_gain_sum / lateral_intervention_feature_count
                if lateral_intervention_feature_count
                else None
            ),
            "lateral_intervention_gain_above_half_count": float(
                lateral_intervention_gain_above_half_count
            ),
            "gravity_intervention_feature_count": float(gravity_intervention_feature_count),
            "gravity_intervention_mean_soft_gain": (
                gravity_intervention_gain_sum / gravity_intervention_feature_count
                if gravity_intervention_feature_count
                else None
            ),
            "gravity_intervention_gain_above_half_count": float(
                gravity_intervention_gain_above_half_count
            ),
            f"distance_gated_matched_object_frames@{detection_threshold_label}": float(
                distance_gated_matched_object_frames
            ),
        }
    )
    if runtime_hypothesis_pool:
        metrics["runtime_hypothesis_forecast_anchor_count"] = float(
            runtime_hypothesis_forecast_anchor_count
        )
        for axis, counts in runtime_hypothesis_axis_selection_count.items():
            for candidate, count in zip(_RUNTIME_HYPOTHESIS_CANDIDATES, counts, strict=True):
                metrics[f"runtime_hypothesis_axis_{'xyz'[axis]}_{candidate}_count"] = float(count)
    for horizon, target_count in sorted(forecast_target_count.items()):
        tracked_count = forecast_tracked_count.get(horizon, 0)
        active_count = forecast_active_count.get(horizon, 0)
        metrics[f"forecast_target_coverage@{horizon}"] = (
            active_count / target_count if target_count else None
        )
        metrics[f"tracked_forecast_active_coverage@{horizon}"] = (
            active_count / tracked_count if tracked_count else None
        )
        metrics[f"model_dropped_forecast_count@{horizon}"] = float(tracked_count - active_count)
        metrics[f"forecast_predictable_target_count@{horizon}"] = float(
            forecast_predictable_target_count.get(horizon, 0)
        )
        metrics[f"forecast_censored_tracked_count@{horizon}"] = float(
            forecast_censored_tracked_count.get(horizon, 0)
        )
    metrics.update(
        {
            "rgb_global_update_latency_mean_ms": _mean_or_none(global_latencies),
            "rgb_fast_update_latency_mean_ms": _mean_or_none(fast_latencies),
            "future_rollout_latency_mean_ms": _mean_or_none(rollout_latencies),
            "visible_position_std_mean_m": _mean_or_none(uncertainty_visible),
            "occluded_position_std_mean_m": _mean_or_none(uncertainty_occluded),
            "post_observation_std_contraction_mean_m": _mean_or_none(uncertainty_contraction),
            "nonfinite_output_count": float(nonfinite_outputs),
            "evaluated_episodes": float(evaluated_episodes),
            "injected_perturbation_batch_updates": float(perturbation_updates),
        }
    )

    requested_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent.parent
        / "evaluation"
        / (
            split
            if resolved_seed_protocol.name == STANDARD_SEED_PROTOCOL
            else f"{split}-{resolved_seed_protocol.name}"
        )
    )
    output = timestamped_artifact_path(requested_output)
    limitations = [
        (
            "Evaluation uses the synthetic sphere-world split and does not "
            "establish real-video generalisation."
        ),
        (
            "The analytic_oracle_parameter baseline reads simulator drag only "
            "for comparison; the evaluated model runtime receives RGB and "
            "known calibration only."
        ),
        (
            "Object alignment for metrics uses held-out simulator positions; "
            "alignment is never fed back into the runtime. Ungated Hungarian "
            "assignments are retained for forecast/baseline error comparability "
            "and are reported only as assignment coverage."
        ),
    ]
    if runtime_hypothesis_pool:
        limitations.append(
            "The runtime hypothesis pool is an explicit post-load evaluation "
            "intervention. Its delayed evidence uses RGB associations only; "
            "this report does not change checkpoint or deployment defaults."
        )
    if correction.count == 0:
        limitations.append(
            "No active matched object/horizon was available for perturbation correction metrics."
        )
    if ordinary_velocity_correction.object_update_count == 0:
        limitations.append(
            "No persistent, distance-gated object was available across an ordinary "
            "non-perturbed prior-to-posterior observation update; ordinary velocity "
            "correction metrics are null."
        )
    if temporal_velocity_measurements.explicit_field_update_count == 0:
        limitations.append(
            "The evaluated runtime exposed no complete explicit temporal velocity "
            "measurement fields; temporal velocity measurement counts are zero and "
            "reported measurement variance is null."
        )
    elif temporal_velocity_measurements.valid_object_count == 0:
        limitations.append(
            "Explicit temporal velocity measurement fields were exposed, but no "
            "distance-independent measurement proposal was marked valid; reported "
            "measurement variance is null."
        )
    if collision_conditioned_forecasts.total_collision_object_horizons == 0:
        limitations.append(
            "No tracked object had a simulator-labelled collision in any future "
            "forecast window; collision-conditioned forecast errors are null."
        )
    if distance_gated_matched_object_frames == 0:
        limitations.append(
            "No current assignment met the 0.5 m detection-distance gate; "
            "distance-gated detection and identity metrics have no true positives."
        )
    if occlusion_transitions.metrics()["occlusion_qualifying_sequence_count"] == 0.0:
        limitations.append(
            "No reliably anchored visible-to-fully-occluded-to-visible target "
            "sequence occurred; sequence-aware occlusion identity and uncertainty "
            "rates are null."
        )
    zero_update_parameters = [
        name
        for name in ("restitution", "drag")
        if identifier_metrics.update_count.get(name, 0) == 0
    ]
    if zero_update_parameters:
        limitations.append(
            "The RGB runtime produced zero identifier updates above the "
            "1e-3 gate threshold for "
            + ", ".join(zero_update_parameters)
            + "; updated-parameter MAE is unavailable for those parameters."
        )
    no_informative_directional_updates = [
        name
        for name in ("restitution", "drag")
        if directional_parameter_metrics[f"informative_{name}_update_count"] == 0.0
    ]
    if no_informative_directional_updates:
        limitations.append(
            "No persistent, distance-gated runtime identifier update coincided "
            "with ground-truth informative evidence for "
            + ", ".join(no_informative_directional_updates)
            + "; directional before/after parameter metrics are unavailable."
        )
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "checkpoint_step": int(payload["step"]),
        "simulator_version": SIMULATOR_VERSION,
        "scenario_mixture": list(config.simulator.scenario_mixture),
        "resolved_scenarios": {
            scenario: asdict(
                SphereWorldConfig.from_config(config)
                .for_scenario(scenario)
                .for_distribution("ood" if split == "ood" else "in_distribution")
            )
            for scenario in config.simulator.scenario_mixture
        },
        "evaluation_episode_scenarios": [
            config.simulator.scenario_mixture[int(seed) % len(config.simulator.scenario_mixture)]
            for seed in resolved_seed_protocol.manifest.seeds
        ],
        "split": split,
        "episodes": evaluated_episodes,
        "device": str(device),
        "precision": resolved_device.precision,
        "rgb_only": True,
        "oracle_runtime_input_used": False,
        "runtime_hypothesis_pool_enabled": runtime_hypothesis_pool,
        "runtime_hypothesis_pool_policy": (
            {
                "candidates": list(_RUNTIME_HYPOTHESIS_CANDIDATES),
                "evidence_horizons_seconds": list(config.runtime.hypothesis_evidence_horizons_seconds),
                "axis_independent_axes": list(config.runtime.hypothesis_axis_independent_axes),
                "evidence_decay": config.runtime.hypothesis_evidence_decay,
                "timestamp_tolerance_seconds": config.runtime.hypothesis_timestamp_tolerance_seconds,
            }
            if runtime_hypothesis_pool
            else None
        ),
        "simulator_state_usage": "metrics_and_baselines_only",
        "deterministic_forecast_support_mask_source": (
            "persistent_target_match_and_target_active_and_no_unseen_external_"
            "actuation_anywhere_in_coupled_scene_over_(anchor_frame,target_frame]"
        ),
        "stochastic_calibration_support_mask_source": (
            "persistent_target_match_and_target_active_including_hidden_external_actuation_outcomes"
        ),
        "collision_conditioned_mask_source": (
            "deterministic_forecast_support_and_evaluation_only_simulator_"
            "collision_any_in_(anchor_frame,target_frame]"
        ),
        "parameter_metric_mask_source": "runtime_identifier_diagnostics",
        "directional_parameter_metric_mask_source": (
            "persistent_distance_gated_runtime_update_and_evaluation_only_"
            "ground_truth_informative_evidence"
        ),
        "current_detection_distance_threshold_m": (_CURRENT_DETECTION_DISTANCE_THRESHOLD_M),
        "identity_metric_match_source": "distance_gated_current_detection",
        "current_velocity_metric_match_source": "distance_gated_current_detection",
        "ordinary_velocity_correction_scope": (
            "timestamp_advanced_prior_to_posterior_nonperturbed_fresh_observation_"
            "same_persistent_slot_and_distance_gated_current_detection"
        ),
        "temporal_velocity_measurement_metric_source": (
            "fresh_runtime_last_measurements_explicit_auxiliary_fields_only"
        ),
        "occlusion_visible_fraction_threshold": (occlusion_transitions.visible_fraction_threshold),
        "occlusion_fully_hidden_fraction_threshold": (
            occlusion_transitions.fully_occluded_fraction_threshold
        ),
        "occlusion_tracking_key": "pre_occlusion_target_to_persistent_prediction_id",
        "occlusion_hidden_tracking_localization_gate_m": None,
        **resolved_seed_protocol.metadata(),
    }
    json_path, markdown_path = write_evaluation_report(
        output,
        metadata=metadata,
        metrics=metrics,
        limitations=limitations,
    )
    report_progress(
        "completed",
        evaluated_episodes=evaluated_episodes,
        output_directory=str(output),
    )
    return {
        "output_directory": str(output),
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload["step"]),
        "split": split,
        "seed_protocol": resolved_seed_protocol.name,
        "episode_seeds": list(resolved_seed_protocol.manifest.seeds),
        "device": str(device),
        "rgb_only": True,
        "oracle_runtime_input_used": False,
        "runtime_hypothesis_pool_enabled": runtime_hypothesis_pool,
        "metrics": metrics,
        "limitations": limitations,
    }


__all__ = ["evaluate_checkpoint"]
