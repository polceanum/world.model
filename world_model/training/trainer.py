"""Local AdamW trainer for the first complete Orpheus vertical slice."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import resource
import shutil
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from world_model.datasets import (
    SyntheticSphereDataset,
    collate_episodes,
    make_seed_manifest,
)
from world_model.runtime import OnlineWorldModel
from world_model.simulator.sphere_world import SphereWorldConfig
from world_model.training.checkpointing import (
    _assert_finite_tensor_tree,
    _assert_valid_optimizer_steps,
    capture_git_metadata,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
    validate_checkpoint_config,
    validate_training_resume_config,
)
from world_model.training.logging import MetricsLogger
from world_model.training.loop import (
    _MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M,
    _PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M,
    PhysicalMetricSupportError,
    TrainingBatchResult,
    move_batch_to_device,
    physical_validation_metrics,
    pretrain_rgb_measurements,
    run_closed_loop_batch,
    select_closed_loop_window,
)
from world_model.training.sampling import (
    ScenarioBalancedStepIndexedBatchSampler,
    StepIndexedBatchSampler,
)
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import OrpheusConfig, save_resolved_config
from world_model.utils.device import DeviceInfo, select_device
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything
from world_model.utils.version import SIMULATOR_VERSION

_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NUMBERED_VALIDATION_CHECKPOINT = re.compile(r"validation_step_(\d+)\.pt$")
_ROLLOUT_SELECTION_MIN_DELTA = 1.0e-5
_ROLLOUT_SELECTION_METRIC_VERSION = 6.0
_ROLLOUT_SELECTION_RELATIVE_GUARDRAIL = 0.02
_ROLLOUT_SELECTION_COVERAGE_TOLERANCE = 0.005
_ROLLOUT_SELECTION_CALIBRATION_ERROR_TOLERANCE = 0.02
_ROLLOUT_VALIDATION_PROTOCOL_VERSION = 14
_NOMINAL_POSITION_COVERAGE = 0.90
_MEASUREMENT_SELECTION_MIN_DELTA = 1.0e-5
_MEASUREMENT_SELECTION_METRIC_VERSION = 5.0
_MEASUREMENT_SELECTION_RELATIVE_GUARDRAIL = 0.02
_MEASUREMENT_SELECTION_RECALL_TOLERANCE = 0.005
_MEASUREMENT_VALIDATION_PROTOCOL_VERSION = 5
_PHYSICAL_ADDITIVE_EXACT_METRICS = frozenset(
    {
        "physical_target_object_frames",
        "physical_predicted_object_frames",
        "physical_matched_object_frames",
        "physical_identity_switches",
        "physical_object_frame_associations",
        "physical_distance_gated_matched_object_frames",
        "physical_distance_gated_target_object_frames",
        "physical_distance_gated_predicted_object_frames",
        "physical_distance_gated_identity_switches",
        "physical_distance_gated_object_frame_associations",
    }
)
_MEASUREMENT_ADDITIVE_METRICS = frozenset(
    {
        "rgb_runtime_birth_true_positive_count_at_0_5m",
        "rgb_runtime_birth_target_count",
        "rgb_runtime_birth_proposal_count",
        "rgb_runtime_birth_world_position_absolute_error_sum_m",
        "rgb_runtime_birth_matched_proposal_count",
        "rgb_world_position_absolute_error_sum_m",
        "rgb_world_position_matched_proposal_count",
        "rgb_fast_bootstrap_matched_target_count",
        "rgb_fast_bootstrap_target_count",
        "rgb_fast_roi_supported_target_count",
        "rgb_fast_roi_target_count",
        "rgb_fast_roi_eligible_proposal_count",
        "rgb_fast_roi_world_position_absolute_error_sum_m",
        "rgb_fast_roi_world_position_matched_count",
        "rgb_fast_prior_world_position_absolute_error_sum_m",
        "rgb_fast_prior_world_position_matched_count",
        "rgb_fast_roi_confident_proposal_count",
        "rgb_fast_roi_true_positive_count_at_0_5m",
    }
)


def _is_additive_physical_metric(name: str) -> bool:
    """Whether a metric is an exact manifest-additive physical quantity."""

    return name.startswith("physical_") and (
        name.endswith("_sse")
        or name.endswith("_count")
        or "_count@" in name
        or name in _PHYSICAL_ADDITIVE_EXACT_METRICS
    )


_CAUSAL_TRAJECTORY_LOSS_TERMS = frozenset(
    {
        "state_position",
        "state_velocity",
        "state",
        "rollout_position",
        "rollout_position_x",
        "rollout_position_y",
        "rollout_position_z",
        "rollout_velocity",
        "rollout_nll",
        "rollout",
        "event",
        "parameter",
        "existence",
        "uncertainty",
        "correction_position",
        "correction_velocity",
    }
)


@dataclass(frozen=True)
class _MeasurementSelectionMetrics:
    """Runtime-usable RGB discovery metrics for perception retention."""

    score: float
    world_position_mae_m: float
    all_proposal_world_position_mae_m: float
    runtime_birth_recall: float
    runtime_birth_precision: float
    runtime_birth_f1: float
    fast_bootstrap_target_coverage: float
    fast_roi_target_coverage: float
    fast_roi_world_position_mae_m: float
    fast_roi_recall: float
    fast_roi_precision: float
    fast_roi_f1: float
    fast_roi_improvement_m: float

    def validation_metrics(self) -> dict[str, float]:
        return {
            "validation_measurement_selection_score": self.score,
            "validation_runtime_birth_world_position_mae_m": self.world_position_mae_m,
            "validation_all_proposal_world_position_mae_m": (
                self.all_proposal_world_position_mae_m
            ),
            "validation_runtime_birth_recall_at_0_5m": self.runtime_birth_recall,
            "validation_runtime_birth_precision_at_0_5m": self.runtime_birth_precision,
            "validation_runtime_birth_f1_at_0_5m": self.runtime_birth_f1,
            "validation_fast_bootstrap_target_coverage": (self.fast_bootstrap_target_coverage),
            "validation_fast_roi_target_coverage": self.fast_roi_target_coverage,
            "validation_fast_roi_world_position_mae_m": (self.fast_roi_world_position_mae_m),
            "validation_fast_roi_recall_at_0_5m": self.fast_roi_recall,
            "validation_fast_roi_precision_at_0_5m": self.fast_roi_precision,
            "validation_fast_roi_f1_at_0_5m": self.fast_roi_f1,
            "validation_fast_roi_improvement_m": self.fast_roi_improvement_m,
        }

    def checkpoint_metrics(self) -> dict[str, float]:
        return {
            "best_measurement_selection_score": self.score,
            # This compatibility alias now names the declared broad selector.
            "best_measurement_loss": self.score,
            "best_measurement_world_position_mae_m": self.world_position_mae_m,
            "best_measurement_all_proposal_world_position_mae_m": (
                self.all_proposal_world_position_mae_m
            ),
            "best_measurement_runtime_birth_recall_at_0_5m": self.runtime_birth_recall,
            "best_measurement_runtime_birth_precision_at_0_5m": self.runtime_birth_precision,
            "best_measurement_runtime_birth_f1_at_0_5m": self.runtime_birth_f1,
            "best_measurement_fast_bootstrap_target_coverage": (
                self.fast_bootstrap_target_coverage
            ),
            "best_measurement_fast_roi_target_coverage": self.fast_roi_target_coverage,
            "best_measurement_fast_roi_world_position_mae_m": (self.fast_roi_world_position_mae_m),
            "best_measurement_fast_roi_recall_at_0_5m": self.fast_roi_recall,
            "best_measurement_fast_roi_precision_at_0_5m": self.fast_roi_precision,
            "best_measurement_fast_roi_f1_at_0_5m": self.fast_roi_f1,
            "best_measurement_fast_roi_improvement_m": self.fast_roi_improvement_m,
            "measurement_selection_metric_version": _MEASUREMENT_SELECTION_METRIC_VERSION,
        }


def _measurement_selection_metrics(
    metrics: Mapping[str, float],
) -> _MeasurementSelectionMetrics | None:
    """Build a broad selector from proposals that can enter runtime state.

    ``None`` means the candidate produced no finite, lifecycle-qualified
    localization evidence.  It remains a valid training iterate but cannot be
    called the best deployable perception checkpoint.
    """

    required = {
        "world_position_mae_m": "rgb_runtime_birth_world_position_mae_m",
        "all_proposal_world_position_mae_m": "rgb_world_position_mae_m",
        "runtime_birth_recall": "rgb_runtime_birth_recall_at_0_5m",
        "runtime_birth_precision": "rgb_runtime_birth_precision_at_0_5m",
        "runtime_birth_f1": "rgb_runtime_birth_f1_at_0_5m",
        "fast_bootstrap_target_coverage": "rgb_fast_bootstrap_target_coverage",
        "fast_roi_target_coverage": "rgb_fast_roi_target_coverage",
        "fast_roi_world_position_mae_m": "rgb_fast_roi_world_position_mae_m",
        "fast_roi_recall": "rgb_fast_roi_recall_at_0_5m",
        "fast_roi_precision": "rgb_fast_roi_precision_at_0_5m",
        "fast_roi_f1": "rgb_fast_roi_f1_at_0_5m",
        "fast_roi_improvement_m": "rgb_fast_roi_improvement_m",
    }
    missing = [key for key in required.values() if key not in metrics]
    if missing:
        raise RuntimeError(
            "RGB measurement validation did not report runtime selection metrics: "
            + ", ".join(sorted(missing))
        )
    values = {name: float(metrics[key]) for name, key in required.items()}
    for name in (
        "runtime_birth_recall",
        "runtime_birth_precision",
        "runtime_birth_f1",
        "fast_bootstrap_target_coverage",
        "fast_roi_target_coverage",
        "fast_roi_recall",
        "fast_roi_precision",
        "fast_roi_f1",
    ):
        value = values[name]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"measurement validation {name} must lie in [0,1]")
    if any(
        not math.isfinite(values[name])
        for name in (
            "world_position_mae_m",
            "all_proposal_world_position_mae_m",
            "fast_roi_world_position_mae_m",
            "fast_roi_improvement_m",
        )
    ):
        return None
    if any(
        values[name] < 0
        for name in (
            "world_position_mae_m",
            "all_proposal_world_position_mae_m",
            "fast_roi_world_position_mae_m",
        )
    ):
        raise ValueError("measurement validation position MAE must be nonnegative")

    score = (
        values["world_position_mae_m"] / _MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M
        + (1.0 - values["runtime_birth_recall"])
        + (1.0 - values["runtime_birth_precision"])
        + values["fast_roi_world_position_mae_m"] / _MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M
        + (1.0 - values["fast_bootstrap_target_coverage"])
        + (1.0 - values["fast_roi_target_coverage"])
        + (1.0 - values["fast_roi_recall"])
        + (1.0 - values["fast_roi_precision"])
    )
    return _MeasurementSelectionMetrics(
        score=score,
        world_position_mae_m=values["world_position_mae_m"],
        all_proposal_world_position_mae_m=values["all_proposal_world_position_mae_m"],
        runtime_birth_recall=values["runtime_birth_recall"],
        runtime_birth_precision=values["runtime_birth_precision"],
        runtime_birth_f1=values["runtime_birth_f1"],
        fast_bootstrap_target_coverage=values["fast_bootstrap_target_coverage"],
        fast_roi_target_coverage=values["fast_roi_target_coverage"],
        fast_roi_world_position_mae_m=values["fast_roi_world_position_mae_m"],
        fast_roi_recall=values["fast_roi_recall"],
        fast_roi_precision=values["fast_roi_precision"],
        fast_roi_f1=values["fast_roi_f1"],
        fast_roi_improvement_m=values["fast_roi_improvement_m"],
    )


def _measurement_selection_from_checkpoint(
    metrics: Mapping[str, Any],
) -> _MeasurementSelectionMetrics | None:
    """Restore only checkpoints written with the broad runtime selector."""

    if float(metrics.get("measurement_selection_metric_version", -1.0)) != (
        _MEASUREMENT_SELECTION_METRIC_VERSION
    ):
        return None
    aliases = {
        "rgb_runtime_birth_world_position_mae_m": ("best_measurement_world_position_mae_m"),
        "rgb_world_position_mae_m": "best_measurement_all_proposal_world_position_mae_m",
        "rgb_runtime_birth_recall_at_0_5m": ("best_measurement_runtime_birth_recall_at_0_5m"),
        "rgb_runtime_birth_precision_at_0_5m": ("best_measurement_runtime_birth_precision_at_0_5m"),
        "rgb_runtime_birth_f1_at_0_5m": "best_measurement_runtime_birth_f1_at_0_5m",
        "rgb_fast_bootstrap_target_coverage": ("best_measurement_fast_bootstrap_target_coverage"),
        "rgb_fast_roi_target_coverage": "best_measurement_fast_roi_target_coverage",
        "rgb_fast_roi_world_position_mae_m": ("best_measurement_fast_roi_world_position_mae_m"),
        "rgb_fast_roi_recall_at_0_5m": ("best_measurement_fast_roi_recall_at_0_5m"),
        "rgb_fast_roi_precision_at_0_5m": ("best_measurement_fast_roi_precision_at_0_5m"),
        "rgb_fast_roi_f1_at_0_5m": "best_measurement_fast_roi_f1_at_0_5m",
        "rgb_fast_roi_improvement_m": "best_measurement_fast_roi_improvement_m",
    }
    if any(checkpoint_key not in metrics for checkpoint_key in aliases.values()):
        return None
    restored = _measurement_selection_metrics(
        {
            metric_key: float(metrics[checkpoint_key])
            for metric_key, checkpoint_key in aliases.items()
        }
    )
    if restored is None:
        return None
    stored_score = metrics.get("best_measurement_selection_score")
    if stored_score is None or not math.isclose(
        float(stored_score),
        restored.score,
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    ):
        return None
    return restored


def _measurement_selection_guardrail_failures(
    candidate: _MeasurementSelectionMetrics,
    incumbent: _MeasurementSelectionMetrics,
) -> list[dict[str, float | str]]:
    """Describe runtime discovery regressions hidden by a scalar score."""

    maximum_ratio = 1.0 + _MEASUREMENT_SELECTION_RELATIVE_GUARDRAIL
    minimum_ratio = 1.0 - _MEASUREMENT_SELECTION_RELATIVE_GUARDRAIL
    failures: list[dict[str, float | str]] = []
    limits = (
        (
            "world_position_mae_m",
            candidate.world_position_mae_m,
            incumbent.world_position_mae_m,
            incumbent.world_position_mae_m * maximum_ratio,
            "maximum",
        ),
        (
            "runtime_birth_recall_at_0_5m",
            candidate.runtime_birth_recall,
            incumbent.runtime_birth_recall,
            incumbent.runtime_birth_recall - _MEASUREMENT_SELECTION_RECALL_TOLERANCE,
            "minimum",
        ),
        (
            "runtime_birth_precision_at_0_5m",
            candidate.runtime_birth_precision,
            incumbent.runtime_birth_precision,
            incumbent.runtime_birth_precision * minimum_ratio,
            "minimum",
        ),
        (
            "fast_bootstrap_target_coverage",
            candidate.fast_bootstrap_target_coverage,
            incumbent.fast_bootstrap_target_coverage,
            incumbent.fast_bootstrap_target_coverage - _MEASUREMENT_SELECTION_RECALL_TOLERANCE,
            "minimum",
        ),
        (
            "fast_roi_target_coverage",
            candidate.fast_roi_target_coverage,
            incumbent.fast_roi_target_coverage,
            incumbent.fast_roi_target_coverage - _MEASUREMENT_SELECTION_RECALL_TOLERANCE,
            "minimum",
        ),
        (
            "fast_roi_world_position_mae_m",
            candidate.fast_roi_world_position_mae_m,
            incumbent.fast_roi_world_position_mae_m,
            incumbent.fast_roi_world_position_mae_m * maximum_ratio,
            "maximum",
        ),
        (
            "fast_roi_recall_at_0_5m",
            candidate.fast_roi_recall,
            incumbent.fast_roi_recall,
            incumbent.fast_roi_recall - _MEASUREMENT_SELECTION_RECALL_TOLERANCE,
            "minimum",
        ),
        (
            "fast_roi_precision_at_0_5m",
            candidate.fast_roi_precision,
            incumbent.fast_roi_precision,
            incumbent.fast_roi_precision * minimum_ratio,
            "minimum",
        ),
        (
            "fast_roi_improvement_m",
            candidate.fast_roi_improvement_m,
            incumbent.fast_roi_improvement_m,
            incumbent.fast_roi_improvement_m - 0.005,
            "minimum",
        ),
    )
    for name, value, reference, limit, direction in limits:
        failed = value > limit if direction == "maximum" else value < limit
        if failed:
            failures.append(
                {
                    "metric": name,
                    "direction": direction,
                    "candidate": value,
                    "reference": reference,
                    "limit": limit,
                    "delta": value - reference,
                }
            )
    return failures


def _measurement_selection_improves(
    candidate: _MeasurementSelectionMetrics,
    incumbent: _MeasurementSelectionMetrics,
) -> bool:
    """Require a better broad score without losing runtime discovery."""

    return (
        candidate.score < incumbent.score - _MEASUREMENT_SELECTION_MIN_DELTA
        and not _measurement_selection_guardrail_failures(candidate, incumbent)
    )


@dataclass(frozen=True)
class _RolloutSelectionMetrics:
    """Physical validation metrics used to retain a broad-accuracy incumbent."""

    score: float
    position_rmse_m: float
    velocity_rmse_mps: float
    target_coverage: float
    prediction_precision: float
    collision_f1: float
    id_switch_rate: float
    position_coverage90: float
    position_calibration_error90: float
    horizon_position_rmse_m: dict[str, float]
    horizon_forecast_target_coverage: dict[str, float]
    axis_position_rmse_m: dict[str, float] = field(default_factory=dict)
    horizon_axis_position_rmse_m: dict[str, dict[str, float]] = field(default_factory=dict)
    # Every declared scenario is represented explicitly. ``None`` means that
    # the fixed validation slice had no complete physical forecast support and
    # therefore cannot authorize promotion.
    scenario_slices: dict[str, _RolloutSelectionMetrics | None] = field(default_factory=dict)

    def validation_metrics(self) -> dict[str, float]:
        metrics = {
            "validation_rollout_selection_score": self.score,
            "validation_position_rmse_m": self.position_rmse_m,
            "validation_velocity_rmse_mps": self.velocity_rmse_mps,
            "validation_target_coverage": self.target_coverage,
            "validation_prediction_precision": self.prediction_precision,
            "validation_collision_f1": self.collision_f1,
            "validation_id_switch_rate": self.id_switch_rate,
            "validation_position_coverage90": self.position_coverage90,
            "validation_position_calibration_error90": self.position_calibration_error90,
            **{
                f"validation_position_rmse@{suffix}": value
                for suffix, value in self.horizon_position_rmse_m.items()
            },
            **{
                f"validation_forecast_target_coverage@{suffix}": value
                for suffix, value in self.horizon_forecast_target_coverage.items()
            },
            **{
                f"validation_position_rmse_{axis}_m": value
                for axis, value in self.axis_position_rmse_m.items()
            },
            **{
                f"validation_position_rmse_{axis}@{suffix}": value
                for axis, horizons in self.horizon_axis_position_rmse_m.items()
                for suffix, value in horizons.items()
            },
        }
        for scenario, selection in sorted(self.scenario_slices.items()):
            metrics[f"validation_scenario_{scenario}_selection_supported"] = float(
                selection is not None
            )
            if selection is None:
                continue
            for name, value in selection.validation_metrics().items():
                suffix = name.removeprefix("validation_")
                metrics[f"validation_scenario_{scenario}_{suffix}"] = value
        return metrics

    def checkpoint_metrics(self, *, prefix: str = "best_rollout") -> dict[str, float]:
        metrics = {
            f"{prefix}_selection_score": self.score,
            # Retain the original public alias while making its physical,
            # horizon-weighted meaning explicit in the adjacent fields.
            f"{prefix}_loss": self.score,
            f"{prefix}_position_loss": self.score,
            f"{prefix}_position_rmse_m": self.position_rmse_m,
            f"{prefix}_velocity_rmse_mps": self.velocity_rmse_mps,
            f"{prefix}_target_coverage": self.target_coverage,
            f"{prefix}_prediction_precision": self.prediction_precision,
            f"{prefix}_collision_f1": self.collision_f1,
            f"{prefix}_id_switch_rate": self.id_switch_rate,
            f"{prefix}_position_coverage90": self.position_coverage90,
            f"{prefix}_position_calibration_error90": self.position_calibration_error90,
            **{
                f"{prefix}_position_rmse@{suffix}": value
                for suffix, value in self.horizon_position_rmse_m.items()
            },
            **{
                f"{prefix}_forecast_target_coverage@{suffix}": value
                for suffix, value in self.horizon_forecast_target_coverage.items()
            },
            **{
                f"{prefix}_position_rmse_{axis}_m": value
                for axis, value in self.axis_position_rmse_m.items()
            },
            **{
                f"{prefix}_position_rmse_{axis}@{suffix}": value
                for axis, horizons in self.horizon_axis_position_rmse_m.items()
                for suffix, value in horizons.items()
            },
        }
        for scenario, selection in sorted(self.scenario_slices.items()):
            scenario_prefix = f"{prefix}_scenario_{scenario}"
            metrics[f"{scenario_prefix}_selection_supported"] = float(selection is not None)
            if selection is not None:
                metrics.update(selection.checkpoint_metrics(prefix=scenario_prefix))
        return metrics


def _selection_horizon_keys(config: OrpheusConfig) -> list[tuple[str, float]]:
    """Return unique physical-horizon metric suffixes and configured weights."""

    selected: list[tuple[str, float]] = []
    seen: set[str] = set()
    for horizon, weight in zip(
        config.evaluation.horizons_seconds,
        config.training.horizon_weights,
        strict=True,
    ):
        frame_offset = max(
            1,
            int(round(float(horizon) * config.simulator.frame_rate)),
        )
        suffix = f"{frame_offset / config.simulator.frame_rate:.3f}s"
        if suffix in seen:
            continue
        seen.add(suffix)
        selected.append((suffix, float(weight)))
    return selected


def _selection_scenario_slugs(config: OrpheusConfig) -> tuple[str, ...]:
    """Return unique stable slugs for every declared validation scenario."""

    slugs: list[str] = []
    scenario_by_slug: dict[str, str] = {}
    for scenario in config.simulator.scenario_mixture:
        slug = re.sub(r"[^a-z0-9]+", "_", scenario.lower()).strip("_")
        if not slug:
            raise ValueError(f"scenario name has no stable slug: {scenario!r}")
        previous = scenario_by_slug.get(slug)
        if previous is not None and previous != scenario:
            raise ValueError(
                "scenario names collide after metric slug normalization: "
                f"{previous!r} and {scenario!r}"
            )
        scenario_by_slug[slug] = scenario
        if slug not in slugs:
            slugs.append(slug)
    return tuple(slugs)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible protocol mapping."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rollout_validation_protocol_from_mapping(
    config_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve every field that can change broad closed-loop validation."""

    simulator = config_mapping.get("simulator")
    model = config_mapping.get("model")
    runtime = config_mapping.get("runtime")
    device = config_mapping.get("device")
    evaluation = config_mapping.get("evaluation")
    training = config_mapping.get("training")
    project = config_mapping.get("project")
    if not all(
        isinstance(section, Mapping)
        for section in (
            simulator,
            model,
            runtime,
            device,
            evaluation,
            training,
            project,
        )
    ):
        raise ValueError("validation protocol requires complete resolved config sections")
    assert isinstance(training, Mapping)
    assert isinstance(project, Mapping)
    assert isinstance(device, Mapping)
    validation_episodes = int(training["validation_episodes"])
    if validation_episodes <= 0:
        raise ValueError("training.validation_episodes must be positive")
    manifest = make_seed_manifest("validation", validation_episodes)
    resolved_simulator = SphereWorldConfig.from_config(config_mapping)
    resolved_scenarios = {
        scenario: asdict(
            resolved_simulator.for_scenario(scenario).for_distribution(
                resolved_simulator.distribution
            )
        )
        for scenario in resolved_simulator.scenario_mixture
    }
    return {
        "protocol_version": _ROLLOUT_VALIDATION_PROTOCOL_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        # Keep both the public simulator config and the fully resolved
        # generator config. The latter includes defaults and derived padding.
        "simulator": dict(simulator),
        "resolved_sphere_world": asdict(resolved_simulator),
        "resolved_scenarios": resolved_scenarios,
        "model": dict(model),
        "runtime": dict(runtime),
        "evaluation": dict(evaluation),
        "selection": {
            "horizon_weights": training["horizon_weights"],
            "rollout_anchors_per_episode": training.get("validation_rollout_anchors_per_episode"),
            "metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            "prediction_distance_threshold_m": (_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M),
            "training_viability": {
                "handoff_minimum_target_coverage": training["handoff_minimum_target_coverage"],
                "handoff_minimum_forecast_coverage": training["handoff_minimum_forecast_coverage"],
                "handoff_minimum_reference_coverage_ratio": training[
                    "handoff_minimum_reference_coverage_ratio"
                ],
                "minimum_predictable_target_count_per_scenario_horizon": training[
                    "validation_minimum_predictable_target_count_per_scenario_horizon"
                ],
                "minimum_matched_target_count_per_scenario_horizon": training[
                    "validation_minimum_matched_target_count_per_scenario_horizon"
                ],
                "minimum_supported_episodes_per_scenario": training[
                    "validation_minimum_supported_episodes_per_scenario"
                ],
            },
        },
        "optimization_stability": {
            "grad_clip_norm": training["grad_clip_norm"],
            "interaction_grad_clip_norm": training["interaction_grad_clip_norm"],
            "closed_loop_perception_grad_clip_norm": training[
                "closed_loop_perception_grad_clip_norm"
            ],
            "minimum_effective_gradient_norm": training["minimum_effective_gradient_norm"],
            "maximum_no_gradient_batches_per_update": training[
                "maximum_no_gradient_batches_per_update"
            ],
        },
        "execution": {
            "project_seed": project["seed"],
            "deterministic": project["deterministic"],
            "measurement_device_preference": device["preference"],
            "closed_loop_device_preference": device.get(
                "closed_loop_preference",
                "same",
            ),
            "global_detector_cpu_on_mps": device.get(
                "global_detector_cpu_on_mps",
                False,
            ),
            # Batch-one validation permits exact per-seed and per-scenario
            # attribution while keeping pooled additive metrics unchanged.
            "validation_batch_size": 1,
        },
        "validation_seed_manifest": {
            "split": manifest.split,
            "seeds": list(manifest.seeds),
        },
    }


def _rollout_validation_protocol(config: OrpheusConfig) -> dict[str, Any]:
    return _rollout_validation_protocol_from_mapping(config.to_dict())


def _rollout_validation_protocol_hash(config: OrpheusConfig) -> str:
    return _canonical_hash(_rollout_validation_protocol(config))


def _measurement_validation_protocol_from_mapping(
    config_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve every field that can change RGB measurement selection."""

    simulator = config_mapping.get("simulator")
    model = config_mapping.get("model")
    runtime = config_mapping.get("runtime")
    device = config_mapping.get("device")
    training = config_mapping.get("training")
    project = config_mapping.get("project")
    if not all(
        isinstance(section, Mapping)
        for section in (simulator, model, runtime, device, training, project)
    ):
        raise ValueError("measurement protocol requires complete resolved config sections")
    assert isinstance(training, Mapping)
    assert isinstance(project, Mapping)
    assert isinstance(device, Mapping)
    validation_episodes = int(training["validation_episodes"])
    if validation_episodes <= 0:
        raise ValueError("training.validation_episodes must be positive")
    manifest = make_seed_manifest("validation", validation_episodes)
    resolved_simulator = SphereWorldConfig.from_config(config_mapping)
    return {
        "protocol_version": _MEASUREMENT_VALIDATION_PROTOCOL_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "simulator": dict(simulator),
        "resolved_sphere_world": asdict(resolved_simulator),
        "resolved_scenarios": {
            scenario: asdict(
                resolved_simulator.for_scenario(scenario).for_distribution(
                    resolved_simulator.distribution
                )
            )
            for scenario in resolved_simulator.scenario_mixture
        },
        "model": dict(model),
        "runtime": dict(runtime),
        "selection": {
            "measurement_validation_frames": training["measurement_validation_frames"],
            "fast_roi_pretrain_weight": training["fast_roi_pretrain_weight"],
            "metric_version": _MEASUREMENT_SELECTION_METRIC_VERSION,
            "prediction_distance_threshold_m": (_MEASUREMENT_SELECTION_DISTANCE_THRESHOLD_M),
        },
        "execution": {
            "project_seed": project["seed"],
            "deterministic": project["deterministic"],
            "measurement_device_preference": device["preference"],
            "global_detector_cpu_on_mps": device.get(
                "global_detector_cpu_on_mps",
                False,
            ),
            "validation_batch_size": 1,
        },
        "validation_seed_manifest": {
            "split": manifest.split,
            "seeds": list(manifest.seeds),
        },
    }


def _measurement_validation_protocol_hash(config: OrpheusConfig) -> str:
    return _canonical_hash(_measurement_validation_protocol_from_mapping(config.to_dict()))


def _measurement_validation_protocol_is_compatible(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    expected_device: str | torch.device | None = None,
) -> bool:
    """Require exact measurement-selection data, metric, and device semantics."""

    metrics = payload.get("metrics")
    checkpoint_config = payload.get("config")
    if not isinstance(metrics, Mapping) or not isinstance(checkpoint_config, Mapping):
        return False
    stored_hash = metrics.get("measurement_validation_protocol_hash")
    if not isinstance(stored_hash, str):
        return False
    if expected_device is not None and payload.get("device") != str(expected_device):
        return False
    try:
        checkpoint_hash = _canonical_hash(
            _measurement_validation_protocol_from_mapping(checkpoint_config)
        )
        requested_hash = _measurement_validation_protocol_hash(config)
    except (KeyError, TypeError, ValueError):
        return False
    return stored_hash == checkpoint_hash == requested_hash


def _validation_protocol_checkpoint_metrics(config: OrpheusConfig) -> dict[str, Any]:
    manifest = make_seed_manifest("validation", config.training.validation_episodes)
    manifest_mapping = {
        "split": manifest.split,
        "seeds": list(manifest.seeds),
    }
    return {
        "rollout_validation_protocol_hash": _rollout_validation_protocol_hash(config),
        "validation_seed_manifest_hash": _canonical_hash(manifest_mapping),
        "validation_seed_manifest_count": float(len(manifest)),
        "validation_seed_manifest_first": float(manifest.seeds[0]),
        "validation_seed_manifest_last": float(manifest.seeds[-1]),
    }


def _validation_support_evidence(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Retain exact additive and slice-support evidence for selector decisions."""

    evidence: dict[str, float] = {}
    for name, value in metrics.items():
        selection_support_marker = name == "selection_metric_supported" or (
            name.startswith(("scenario_", "seed_")) and name.endswith("_selection_metric_supported")
        )
        scenario_episode_count = name.startswith("scenario_") and name.endswith("_episode_count")
        scenario_physical_name: str | None = None
        if name.startswith("scenario_") and "_physical_" in name:
            scenario_physical_name = "physical_" + name.split("_physical_", 1)[1]
        additive = _is_additive_physical_metric(name) or (
            scenario_physical_name is not None
            and _is_additive_physical_metric(scenario_physical_name)
        )
        if not additive and not selection_support_marker and not scenario_episode_count:
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"validation support evidence {name!r} must be finite/nonnegative")
        evidence[name] = numeric
    return evidence


def _validate_validation_support_schema(
    metrics: Mapping[str, Any],
    config: OrpheusConfig,
) -> float:
    """Validate pooled, scenario, and per-seed causal-support markers."""

    def binary(name: str) -> float:
        if name not in metrics:
            raise RuntimeError(f"closed-loop validation did not report required {name!r}")
        value = float(metrics[name])
        if not math.isfinite(value) or value not in {0.0, 1.0}:
            raise ValueError(f"closed-loop validation marker {name!r} must be binary")
        return value

    def count(name: str) -> float:
        if name not in metrics:
            raise RuntimeError(f"closed-loop validation did not report required {name!r}")
        value = float(metrics[name])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(
                f"closed-loop validation count {name!r} must be a finite nonnegative integer"
            )
        return value

    def validate_horizon_support(prefix: str, *, supported: float) -> None:
        for suffix, _ in _selection_horizon_keys(config):
            predictable_key = f"{prefix}physical_forecast_predictable_target_count@{suffix}"
            coordinate_key = f"{prefix}physical_rollout_position@{suffix}_coordinate_count"
            predictable_count = count(predictable_key)
            coordinate_count = count(coordinate_key)
            if supported == 1.0 and (
                predictable_count
                < config.training.validation_minimum_predictable_target_count_per_scenario_horizon
            ):
                raise ValueError(
                    f"supported validation slice {prefix!r} has "
                    f"{predictable_count:g} predictable targets at {suffix}"
                )
            required_coordinates = (
                3 * config.training.validation_minimum_matched_target_count_per_scenario_horizon
            )
            if supported == 1.0 and coordinate_count < required_coordinates:
                raise ValueError(
                    f"supported validation slice {prefix!r} has "
                    f"{coordinate_count:g} matched coordinates at {suffix}"
                )

    manifest = make_seed_manifest("validation", config.training.validation_episodes)
    scenario_slugs = _selection_scenario_slugs(config)
    expected_episode_counts = {scenario: 0 for scenario in scenario_slugs}
    expected_supported_counts = {scenario: 0 for scenario in scenario_slugs}
    for seed in manifest.seeds:
        seed_support = binary(f"seed_{seed}_selection_metric_supported")
        scenario = scenario_slugs[int(seed) % len(scenario_slugs)]
        expected_episode_counts[scenario] += 1
        expected_supported_counts[scenario] += int(seed_support)

    pooled_support = binary("selection_metric_supported")
    validate_horizon_support("", supported=pooled_support)
    expected_minimum = config.training.validation_minimum_supported_episodes_per_scenario
    for scenario in scenario_slugs:
        prefix = f"scenario_{scenario}_"
        episode_key = f"{prefix}episode_count"
        supported_key = f"{prefix}supported_episode_count"
        minimum_key = f"{prefix}minimum_supported_episode_count"
        support_key = f"{prefix}selection_metric_supported"
        episode_count = count(episode_key)
        supported_count = count(supported_key)
        minimum_count = count(minimum_key)
        if episode_count <= 0:
            raise ValueError(f"declared scenario {scenario!r} has no validation episodes")
        if episode_count != float(expected_episode_counts[scenario]):
            raise ValueError(
                f"declared scenario {scenario!r} recorded {episode_count:g} episodes, "
                f"expected {expected_episode_counts[scenario]}"
            )
        if supported_count > episode_count:
            raise ValueError(
                f"declared scenario {scenario!r} has more supported than total episodes"
            )
        if supported_count != float(expected_supported_counts[scenario]):
            raise ValueError(
                f"declared scenario {scenario!r} recorded {supported_count:g} "
                "supported episodes, but its per-seed markers imply "
                f"{expected_supported_counts[scenario]}"
            )
        if minimum_count != float(expected_minimum):
            raise ValueError(
                f"declared scenario {scenario!r} recorded support minimum "
                f"{minimum_count:g}, expected {expected_minimum}"
            )
        support = binary(support_key)
        validate_horizon_support(prefix, supported=support)
        if support == 1.0 and supported_count < minimum_count:
            raise ValueError(
                f"declared scenario {scenario!r} claims support below its episode floor"
            )

    return pooled_support


def _rollout_selection_metrics(
    metrics: Mapping[str, float],
    config: OrpheusConfig,
    *,
    require_scenarios: bool = False,
) -> _RolloutSelectionMetrics:
    """Extract a complete finite physical checkpoint-selection summary."""

    required = {
        "position_rmse_m": "validation_position_rmse_m",
        "velocity_rmse_mps": "validation_velocity_rmse_mps",
        "target_coverage": "validation_target_coverage",
        "prediction_precision": "validation_prediction_precision",
        "collision_f1": "validation_collision_f1",
        "id_switch_rate": "validation_id_switch_rate",
        "position_coverage90": "validation_position_coverage90",
    }
    missing = [metric_key for metric_key in required.values() if metric_key not in metrics]
    horizon_values: dict[str, float] = {}
    weighted_values: list[float] = []
    weights: list[float] = []
    for suffix, weight in _selection_horizon_keys(config):
        metric_key = f"validation_position_rmse@{suffix}"
        if metric_key not in metrics:
            missing.append(metric_key)
            continue
        value = float(metrics[metric_key])
        horizon_values[suffix] = value
        weighted_values.append(value)
        weights.append(weight)
    horizon_coverage_values: dict[str, float] = {}
    for suffix, _ in _selection_horizon_keys(config):
        metric_key = f"validation_forecast_target_coverage@{suffix}"
        if metric_key not in metrics:
            missing.append(metric_key)
            continue
        horizon_coverage_values[suffix] = float(metrics[metric_key])
    axis_values: dict[str, float] = {}
    horizon_axis_values: dict[str, dict[str, float]] = {}
    for axis in ("x", "y", "z"):
        axis_key = f"validation_position_rmse_{axis}_m"
        if axis_key not in metrics:
            missing.append(axis_key)
            continue
        axis_values[axis] = float(metrics[axis_key])
        horizon_axis_values[axis] = {}
        for suffix, _ in _selection_horizon_keys(config):
            metric_key = f"validation_position_rmse_{axis}@{suffix}"
            if metric_key not in metrics:
                missing.append(metric_key)
            else:
                horizon_axis_values[axis][suffix] = float(metrics[metric_key])
    if missing:
        raise RuntimeError(
            "closed-loop validation did not report broad selection metrics: "
            + ", ".join(sorted(missing))
        )

    values = {name: float(metrics[key]) for name, key in required.items()}
    finite_values = [
        *values.values(),
        *horizon_values.values(),
        *horizon_coverage_values.values(),
        *axis_values.values(),
        *(value for horizons in horizon_axis_values.values() for value in horizons.values()),
    ]
    if any(not math.isfinite(value) for value in finite_values):
        raise FloatingPointError("closed-loop broad selection metrics must all be finite")
    if values["position_rmse_m"] < 0 or values["velocity_rmse_mps"] < 0:
        raise ValueError("physical validation RMSE metrics must be nonnegative")
    if any(value < 0 for value in horizon_values.values()):
        raise ValueError("per-horizon validation RMSE metrics must be nonnegative")
    if any(value < 0 for value in axis_values.values()) or any(
        value < 0 for horizons in horizon_axis_values.values() for value in horizons.values()
    ):
        raise ValueError("per-axis validation RMSE metrics must be nonnegative")
    for name in (
        "target_coverage",
        "prediction_precision",
        "collision_f1",
        "id_switch_rate",
        "position_coverage90",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"validation {name} must lie in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in horizon_coverage_values.values()):
        raise ValueError("per-horizon forecast target coverage must lie in [0, 1]")
    weight_total = sum(weights)
    if not math.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("checkpoint-selection horizon weights must sum to a positive value")
    score = (
        sum(value * weight for value, weight in zip(weighted_values, weights, strict=True))
        / weight_total
    )
    selection = _RolloutSelectionMetrics(
        score=score,
        position_rmse_m=values["position_rmse_m"],
        velocity_rmse_mps=values["velocity_rmse_mps"],
        target_coverage=values["target_coverage"],
        prediction_precision=values["prediction_precision"],
        collision_f1=values["collision_f1"],
        id_switch_rate=values["id_switch_rate"],
        position_coverage90=values["position_coverage90"],
        position_calibration_error90=abs(
            values["position_coverage90"] - _NOMINAL_POSITION_COVERAGE
        ),
        horizon_position_rmse_m=horizon_values,
        horizon_forecast_target_coverage=horizon_coverage_values,
        axis_position_rmse_m=axis_values,
        horizon_axis_position_rmse_m=horizon_axis_values,
    )
    if not require_scenarios:
        return selection

    scenario_slices: dict[str, _RolloutSelectionMetrics | None] = {}
    for scenario in _selection_scenario_slugs(config):
        metric_prefix = f"scenario_{scenario}_"
        episode_count_key = f"{metric_prefix}episode_count"
        support_key = f"{metric_prefix}selection_metric_supported"
        missing_scenario_keys = [
            key for key in (episode_count_key, support_key) if key not in metrics
        ]
        if missing_scenario_keys:
            raise RuntimeError(
                "closed-loop validation did not report declared scenario support: "
                + ", ".join(missing_scenario_keys)
            )
        episode_count = float(metrics[episode_count_key])
        if not math.isfinite(episode_count) or episode_count <= 0:
            raise ValueError(f"declared scenario {scenario!r} must contain validation episodes")
        support_value = float(metrics[support_key])
        if not math.isfinite(support_value) or support_value not in {0.0, 1.0}:
            raise ValueError(f"declared scenario {scenario!r} has an invalid support marker")
        if support_value == 0.0:
            scenario_slices[scenario] = None
            continue
        scenario_metrics = {
            name.removeprefix(metric_prefix): float(value)
            for name, value in metrics.items()
            if name.startswith(f"{metric_prefix}validation_")
        }
        scenario_slices[scenario] = _rollout_selection_metrics(
            scenario_metrics,
            config,
            require_scenarios=False,
        )
    return replace(selection, scenario_slices=scenario_slices)


def _rollout_selection_from_checkpoint(
    metrics: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    prefix: str = "best_rollout",
    include_scenarios: bool = True,
) -> _RolloutSelectionMetrics | None:
    """Restore the broad incumbent required to guard a resumed run."""

    translated: dict[str, float] = {}
    aliases = {
        "validation_position_rmse_m": f"{prefix}_position_rmse_m",
        "validation_velocity_rmse_mps": f"{prefix}_velocity_rmse_mps",
        "validation_target_coverage": f"{prefix}_target_coverage",
        "validation_prediction_precision": f"{prefix}_prediction_precision",
        "validation_collision_f1": f"{prefix}_collision_f1",
        "validation_id_switch_rate": f"{prefix}_id_switch_rate",
        "validation_position_coverage90": f"{prefix}_position_coverage90",
    }
    for output_key, checkpoint_key in aliases.items():
        if checkpoint_key not in metrics:
            return None
        translated[output_key] = float(metrics[checkpoint_key])
    for suffix, _ in _selection_horizon_keys(config):
        checkpoint_key = f"{prefix}_position_rmse@{suffix}"
        if checkpoint_key not in metrics:
            return None
        translated[f"validation_position_rmse@{suffix}"] = float(metrics[checkpoint_key])
        coverage_key = f"{prefix}_forecast_target_coverage@{suffix}"
        if coverage_key not in metrics:
            return None
        translated[f"validation_forecast_target_coverage@{suffix}"] = float(metrics[coverage_key])
    axis_checkpoint_keys = [f"{prefix}_position_rmse_{axis}_m" for axis in ("x", "y", "z")]
    if not all(key in metrics for key in axis_checkpoint_keys):
        return None
    for axis, checkpoint_key in zip(
        ("x", "y", "z"),
        axis_checkpoint_keys,
        strict=True,
    ):
        translated[f"validation_position_rmse_{axis}_m"] = float(metrics[checkpoint_key])
        for suffix, _ in _selection_horizon_keys(config):
            horizon_key = f"{prefix}_position_rmse_{axis}@{suffix}"
            if horizon_key not in metrics:
                return None
            translated[f"validation_position_rmse_{axis}@{suffix}"] = float(metrics[horizon_key])
    restored = _rollout_selection_metrics(translated, config)
    stored_score = metrics.get(f"{prefix}_selection_score")
    if stored_score is None or not math.isclose(
        float(stored_score),
        restored.score,
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    ):
        return None
    if not include_scenarios:
        return restored

    scenario_slices: dict[str, _RolloutSelectionMetrics | None] = {}
    for scenario in _selection_scenario_slugs(config):
        scenario_prefix = f"{prefix}_scenario_{scenario}"
        support_key = f"{scenario_prefix}_selection_supported"
        if support_key not in metrics:
            return None
        try:
            support_value = float(metrics[support_key])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(support_value) or support_value not in {0.0, 1.0}:
            return None
        if support_value == 0.0:
            scenario_slices[scenario] = None
            continue
        scenario_selection = _rollout_selection_from_checkpoint(
            metrics,
            config,
            prefix=scenario_prefix,
            include_scenarios=False,
        )
        if scenario_selection is None:
            return None
        scenario_slices[scenario] = scenario_selection
    return replace(restored, scenario_slices=scenario_slices)


def _rollout_selection_from_additive_evidence(
    metrics: Mapping[str, Any],
    config: OrpheusConfig,
) -> _RolloutSelectionMetrics:
    """Recompute a selector solely from retained exact validation evidence."""

    pooled = _rollout_selection_metrics(
        physical_validation_metrics(metrics, config),
        config,
    )
    scenario_slices: dict[str, _RolloutSelectionMetrics | None] = {}
    for scenario in _selection_scenario_slugs(config):
        metric_prefix = f"scenario_{scenario}_"
        support_key = f"{metric_prefix}selection_metric_supported"
        support = _binary_checkpoint_marker(
            metrics.get(support_key),
            name=support_key,
        )
        if not support:
            scenario_slices[scenario] = None
            continue
        scenario_additive = {
            name.removeprefix(metric_prefix): value
            for name, value in metrics.items()
            if name.startswith(metric_prefix)
        }
        scenario_slices[scenario] = _rollout_selection_metrics(
            physical_validation_metrics(scenario_additive, config),
            config,
        )
    return replace(pooled, scenario_slices=scenario_slices)


def _checkpoint_selection_matches_additive_evidence(
    metrics: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    prefix: str,
) -> bool:
    """Reject derived selector metadata that contradicts exact raw sums."""

    stored = _rollout_selection_from_checkpoint(
        metrics,
        config,
        prefix=prefix,
    )
    if stored is None:
        return False
    recomputed = _rollout_selection_from_additive_evidence(metrics, config)
    expected_metrics = {
        **recomputed.validation_metrics(),
        **recomputed.checkpoint_metrics(prefix=prefix),
    }
    for name, expected in expected_metrics.items():
        value = metrics.get(name)
        try:
            actual = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(actual) or not math.isclose(
            actual,
            float(expected),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            return False
    return True


def _rollout_selection_guardrail_failures(
    candidate: _RolloutSelectionMetrics,
    reference: _RolloutSelectionMetrics,
) -> list[dict[str, float | str]]:
    """Describe every material regression against a physical reference."""

    maximum_ratio = 1.0 + _ROLLOUT_SELECTION_RELATIVE_GUARDRAIL
    minimum_ratio = 1.0 - _ROLLOUT_SELECTION_RELATIVE_GUARDRAIL
    failures: list[dict[str, float | str]] = []

    def maximum(name: str, value: float, reference_value: float, limit: float) -> None:
        if value > limit:
            failures.append(
                {
                    "metric": name,
                    "direction": "maximum",
                    "candidate": value,
                    "reference": reference_value,
                    "limit": limit,
                    "delta": value - reference_value,
                }
            )

    def minimum(name: str, value: float, reference_value: float, limit: float) -> None:
        if value < limit:
            failures.append(
                {
                    "metric": name,
                    "direction": "minimum",
                    "candidate": value,
                    "reference": reference_value,
                    "limit": limit,
                    "delta": value - reference_value,
                }
            )

    maximum(
        "position_rmse_m",
        candidate.position_rmse_m,
        reference.position_rmse_m,
        reference.position_rmse_m * maximum_ratio,
    )
    maximum(
        "velocity_rmse_mps",
        candidate.velocity_rmse_mps,
        reference.velocity_rmse_mps,
        reference.velocity_rmse_mps * maximum_ratio,
    )
    minimum(
        "target_coverage",
        candidate.target_coverage,
        reference.target_coverage,
        reference.target_coverage - _ROLLOUT_SELECTION_COVERAGE_TOLERANCE,
    )
    minimum(
        "prediction_precision",
        candidate.prediction_precision,
        reference.prediction_precision,
        reference.prediction_precision * minimum_ratio,
    )
    minimum(
        "collision_f1",
        candidate.collision_f1,
        reference.collision_f1,
        reference.collision_f1 * minimum_ratio,
    )
    maximum(
        "id_switch_rate",
        candidate.id_switch_rate,
        reference.id_switch_rate,
        reference.id_switch_rate + _ROLLOUT_SELECTION_COVERAGE_TOLERANCE,
    )
    maximum(
        "position_calibration_error90",
        candidate.position_calibration_error90,
        reference.position_calibration_error90,
        (reference.position_calibration_error90 + _ROLLOUT_SELECTION_CALIBRATION_ERROR_TOLERANCE),
    )
    if reference.axis_position_rmse_m:
        if set(candidate.axis_position_rmse_m) != set(reference.axis_position_rmse_m):
            failures.append(
                {
                    "metric": "axis_position_rmse_schema",
                    "direction": "required",
                    "candidate": float(len(candidate.axis_position_rmse_m)),
                    "reference": float(len(reference.axis_position_rmse_m)),
                    "limit": float(len(reference.axis_position_rmse_m)),
                    "delta": float(
                        len(candidate.axis_position_rmse_m) - len(reference.axis_position_rmse_m)
                    ),
                }
            )
        else:
            for axis, reference_value in reference.axis_position_rmse_m.items():
                maximum(
                    f"position_rmse_{axis}_m",
                    candidate.axis_position_rmse_m[axis],
                    reference_value,
                    reference_value * maximum_ratio,
                )
            for axis, reference_horizons in reference.horizon_axis_position_rmse_m.items():
                candidate_horizons = candidate.horizon_axis_position_rmse_m.get(
                    axis,
                    {},
                )
                if set(candidate_horizons) != set(reference_horizons):
                    failures.append(
                        {
                            "metric": f"horizon_axis_position_rmse_{axis}_schema",
                            "direction": "required",
                            "candidate": float(len(candidate_horizons)),
                            "reference": float(len(reference_horizons)),
                            "limit": float(len(reference_horizons)),
                            "delta": float(len(candidate_horizons) - len(reference_horizons)),
                        }
                    )
                    continue
                for suffix, reference_value in reference_horizons.items():
                    maximum(
                        f"position_rmse_{axis}@{suffix}",
                        candidate_horizons[suffix],
                        reference_value,
                        reference_value * maximum_ratio,
                    )
    for suffix, reference_value in reference.horizon_position_rmse_m.items():
        maximum(
            f"position_rmse@{suffix}",
            candidate.horizon_position_rmse_m[suffix],
            reference_value,
            reference_value * maximum_ratio,
        )
    for suffix, reference_value in reference.horizon_forecast_target_coverage.items():
        minimum(
            f"forecast_target_coverage@{suffix}",
            candidate.horizon_forecast_target_coverage[suffix],
            reference_value,
            reference_value - _ROLLOUT_SELECTION_COVERAGE_TOLERANCE,
        )
    if set(candidate.scenario_slices) != set(reference.scenario_slices):
        failures.append(
            {
                "metric": "scenario_selection_schema",
                "direction": "required",
                "candidate": float(len(candidate.scenario_slices)),
                "reference": float(len(reference.scenario_slices)),
                "limit": float(len(reference.scenario_slices)),
                "delta": float(len(candidate.scenario_slices) - len(reference.scenario_slices)),
            }
        )
        return failures
    for scenario, reference_scenario in sorted(reference.scenario_slices.items()):
        candidate_scenario = candidate.scenario_slices[scenario]
        if candidate_scenario is None:
            failures.append(
                {
                    "metric": f"scenario_{scenario}_selection_support",
                    "direction": "required",
                    "candidate": 0.0,
                    "reference": float(reference_scenario is not None),
                    "limit": 1.0,
                    "delta": -float(reference_scenario is not None),
                }
            )
            continue
        # A reference without scenario support cannot establish a numerical
        # non-regression limit. The candidate must still restore that support;
        # subsequent accepted incumbents then provide complete guardrails.
        if reference_scenario is None:
            continue
        for failure in _rollout_selection_guardrail_failures(
            candidate_scenario,
            reference_scenario,
        ):
            failures.append(
                {
                    **failure,
                    "metric": f"scenario_{scenario}_{failure['metric']}",
                }
            )
    return failures


def _rollout_selection_passes_guardrails(
    candidate: _RolloutSelectionMetrics,
    reference: _RolloutSelectionMetrics,
) -> bool:
    """Reject material regression against one broad physical reference."""

    return not _rollout_selection_guardrail_failures(candidate, reference)


def _rollout_selection_improves(
    candidate: _RolloutSelectionMetrics,
    incumbent: _RolloutSelectionMetrics,
) -> bool:
    """Require a better forecast score without material broad regressions."""

    if candidate.score >= incumbent.score - _ROLLOUT_SELECTION_MIN_DELTA:
        return False
    return _rollout_selection_passes_guardrails(candidate, incumbent)


def _handoff_training_support_failures(
    candidate: _RolloutSelectionMetrics,
    reference: _RolloutSelectionMetrics,
    config: OrpheusConfig,
) -> list[dict[str, float | str]]:
    """Return coverage failures that would starve downstream causal learning.

    A measurement-only selector is intentionally allowed to optimize proposal
    quality.  It is not allowed to become the mutable causal source when the
    resulting persistent runtime tracks only a small conditional subset: in
    that state the ROI, filter, and rollout losses are absent and a frozen
    global detector cannot repair the handoff.
    """

    failures: list[dict[str, float | str]] = []
    ratio = config.training.handoff_minimum_reference_coverage_ratio

    def require(name: str, value: float, reference_value: float, absolute: float) -> None:
        limit = max(absolute, reference_value * ratio)
        if value < limit:
            failures.append(
                {
                    "metric": name,
                    "direction": "minimum",
                    "candidate": value,
                    "reference": reference_value,
                    "limit": limit,
                    "delta": value - reference_value,
                }
            )

    require(
        "target_coverage",
        candidate.target_coverage,
        reference.target_coverage,
        config.training.handoff_minimum_target_coverage,
    )
    if set(candidate.horizon_forecast_target_coverage) != set(
        reference.horizon_forecast_target_coverage
    ):
        failures.append(
            {
                "metric": "forecast_target_coverage_schema",
                "direction": "required",
                "candidate": float(len(candidate.horizon_forecast_target_coverage)),
                "reference": float(len(reference.horizon_forecast_target_coverage)),
                "limit": float(len(reference.horizon_forecast_target_coverage)),
                "delta": float(
                    len(candidate.horizon_forecast_target_coverage)
                    - len(reference.horizon_forecast_target_coverage)
                ),
            }
        )
        return failures
    for suffix, reference_value in reference.horizon_forecast_target_coverage.items():
        require(
            f"forecast_target_coverage@{suffix}",
            candidate.horizon_forecast_target_coverage[suffix],
            reference_value,
            config.training.handoff_minimum_forecast_coverage,
        )
    if set(candidate.scenario_slices) != set(reference.scenario_slices):
        failures.append(
            {
                "metric": "scenario_selection_schema",
                "direction": "required",
                "candidate": float(len(candidate.scenario_slices)),
                "reference": float(len(reference.scenario_slices)),
                "limit": float(len(reference.scenario_slices)),
                "delta": float(len(candidate.scenario_slices) - len(reference.scenario_slices)),
            }
        )
        return failures
    for scenario, reference_scenario in sorted(reference.scenario_slices.items()):
        candidate_scenario = candidate.scenario_slices[scenario]
        if candidate_scenario is None:
            failures.append(
                {
                    "metric": f"scenario_{scenario}_selection_support",
                    "direction": "required",
                    "candidate": 0.0,
                    "reference": float(reference_scenario is not None),
                    "limit": 1.0,
                    "delta": -float(reference_scenario is not None),
                }
            )
            continue
        reference_target_coverage = (
            0.0 if reference_scenario is None else reference_scenario.target_coverage
        )
        require(
            f"scenario_{scenario}_target_coverage",
            candidate_scenario.target_coverage,
            reference_target_coverage,
            config.training.handoff_minimum_target_coverage,
        )
        reference_horizons = (
            {}
            if reference_scenario is None
            else reference_scenario.horizon_forecast_target_coverage
        )
        for (
            suffix,
            candidate_coverage,
        ) in candidate_scenario.horizon_forecast_target_coverage.items():
            require(
                f"scenario_{scenario}_forecast_target_coverage@{suffix}",
                candidate_coverage,
                reference_horizons.get(suffix, 0.0),
                config.training.handoff_minimum_forecast_coverage,
            )
    return failures


def _mutable_causal_training_support_failures(
    candidate: _RolloutSelectionMetrics,
    config: OrpheusConfig,
) -> list[dict[str, float | str]]:
    """Return only catastrophic pooled failures that require state rollback.

    Scenario and reference-relative floors remain mandatory for deployment,
    but a finite iterate with absolute pooled causal coverage can still repair
    a weak scenario. Rolling that iterate back every validation interval makes
    the failure self-perpetuating and conflates deployment selection with the
    mutable optimization trajectory.
    """

    failures: list[dict[str, float | str]] = []

    def require(name: str, value: float, limit: float) -> None:
        if value < limit:
            failures.append(
                {
                    "metric": name,
                    "direction": "minimum_mutable_viability",
                    "candidate": value,
                    "reference": limit,
                    "limit": limit,
                    "delta": value - limit,
                }
            )

    require(
        "target_coverage",
        candidate.target_coverage,
        config.training.handoff_minimum_target_coverage,
    )
    expected_horizons = {suffix for suffix, _ in _selection_horizon_keys(config)}
    candidate_horizons = set(candidate.horizon_forecast_target_coverage)
    if candidate_horizons != expected_horizons:
        failures.append(
            {
                "metric": "forecast_target_coverage_schema",
                "direction": "required_mutable_viability",
                "candidate": float(len(candidate_horizons)),
                "reference": float(len(expected_horizons)),
                "limit": float(len(expected_horizons)),
                "delta": float(len(candidate_horizons) - len(expected_horizons)),
            }
        )
        return failures
    for suffix in sorted(expected_horizons):
        require(
            f"forecast_target_coverage@{suffix}",
            candidate.horizon_forecast_target_coverage[suffix],
            config.training.handoff_minimum_forecast_coverage,
        )
    return failures


def _has_effective_gradient(
    gradient_norm: float,
    config: OrpheusConfig,
) -> bool:
    """Whether a consumed batch produced an optimizer-relevant gradient."""

    return gradient_norm > config.training.minimum_effective_gradient_norm


def _finite_nonnegative_integer(value: Any, *, name: str) -> int:
    """Parse a durable counter without silently truncating corrupt metadata."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"checkpoint {name} must be a finite nonnegative integer")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or not parsed.is_integer():
        raise ValueError(f"checkpoint {name} must be a finite nonnegative integer")
    return int(parsed)


def _binary_checkpoint_marker(value: Any, *, name: str) -> bool:
    """Parse a durable boolean marker with no truthy-number coercion."""

    if isinstance(value, bool):
        return value
    if not isinstance(value, Real):
        raise ValueError(f"checkpoint {name} must be exactly 0 or 1")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed not in {0.0, 1.0}:
        raise ValueError(f"checkpoint {name} must be exactly 0 or 1")
    return parsed == 1.0


def _causal_training_support(
    result: TrainingBatchResult,
) -> tuple[bool, float, float, float]:
    """Return explicit trajectory/fast support for one causal optimizer draw.

    The closed-loop objective retains a global-discovery measurement term as
    an anti-drift auxiliary.  Its gradient alone is not evidence that the
    persistent predict/correct/rollout path participated in the batch.  A
    causal update therefore needs a differentiable trajectory term backed by
    matched object/forecast support, or a differentiable fast-ROI measurement
    term backed by an actual projected ROI.
    """

    if result.phase != "closed_loop_rgb":
        return False, 0.0, 0.0, 0.0
    matched_support = max(
        0.0,
        float(
            result.metrics.get(
                "physical_matched_object_frames",
                result.metrics.get("matched_object_frames", 0.0),
            )
        ),
    )
    forecast_support = sum(
        max(0.0, float(value))
        for name, value in result.metrics.items()
        if name.startswith("physical_rollout_position@")
        and name.endswith("_coordinate_count")
        and math.isfinite(float(value))
    )
    existence_negative_support = max(
        0.0,
        float(
            result.metrics.get(
                "existence_negative_supervision_object_frames",
                0.0,
            )
        ),
    )
    trajectory_support = matched_support + forecast_support + existence_negative_support
    candidate_trajectory_terms = [
        (name, value)
        for name, value in result.loss_terms.items()
        if name in _CAUSAL_TRAJECTORY_LOSS_TERMS and value.requires_grad
    ]
    contributing_trajectory_terms: list[str] = []
    if result.total_loss.requires_grad and candidate_trajectory_terms:
        # Inspect the shallow weighted-loss graph before the real backward.
        # ``requires_grad`` alone is insufficient: a configured zero weight
        # leaves a differentiable term in ``loss_terms`` while contributing no
        # causal gradient to ``total_loss``.
        unique_terms: list[Tensor] = []
        term_names_by_identity: dict[int, list[str]] = {}
        for name, value in candidate_trajectory_terms:
            identity = id(value)
            term_names_by_identity.setdefault(identity, []).append(name)
            if len(term_names_by_identity[identity]) == 1:
                unique_terms.append(value)
        derivatives = torch.autograd.grad(
            result.total_loss,
            unique_terms,
            retain_graph=True,
            allow_unused=True,
        )
        for value, derivative in zip(unique_terms, derivatives, strict=True):
            if (
                derivative is not None
                and bool(torch.isfinite(derivative).all())
                and bool(torch.any(derivative != 0))
            ):
                contributing_trajectory_terms.extend(term_names_by_identity[id(value)])
    trajectory_differentiable = bool(contributing_trajectory_terms)
    objective_term_support = float(len(contributing_trajectory_terms))
    fast_support_metric = (
        "fast_supervised_slots"
        if "fast_supervised_slots" in result.metrics
        else "fast_supervised_frames"
    )
    fast_support = max(0.0, float(result.metrics.get(fast_support_metric, 0.0)))
    fast_measurement = result.support_terms.get("fast_measurement")
    fast_differentiable = False
    if (
        fast_support > 0.0
        and fast_measurement is not None
        and fast_measurement.requires_grad
        and result.total_loss.requires_grad
    ):
        (measurement_derivative,) = torch.autograd.grad(
            result.total_loss,
            (fast_measurement,),
            retain_graph=True,
            allow_unused=True,
        )
        fast_differentiable = (
            measurement_derivative is not None
            and bool(torch.isfinite(measurement_derivative).all())
            and bool(torch.any(measurement_derivative != 0))
        )
    effective_fast_support = fast_support if fast_differentiable else 0.0
    supported = (trajectory_support > 0.0 and trajectory_differentiable) or fast_differentiable
    return supported, trajectory_support, effective_fast_support, objective_term_support


def _rollout_validation_checkpoint_metrics(
    validation: TrainingBatchResult,
    candidate: _RolloutSelectionMetrics,
    incumbent: _RolloutSelectionMetrics,
    reference: _RolloutSelectionMetrics,
    *,
    config: OrpheusConfig,
    accepted: bool,
    training_support_required: bool,
    training_support_failures: list[dict[str, float | str]],
    mutable_training_support_failures: list[dict[str, float | str]],
    best_measurement: _MeasurementSelectionMetrics | None,
    checkpoint_model_state_hash: str,
    incumbent_model_state_hash: str,
    incumbent_step: int,
    reference_model_state_hash: str,
    reference_step: int,
) -> dict[str, Any]:
    """Build truthful metadata shared by best and numbered validation saves."""

    guardrail_failures = _rollout_selection_guardrail_failures(
        candidate,
        reference,
    )
    incumbent_guardrail_failures = _rollout_selection_guardrail_failures(
        candidate,
        incumbent,
    )
    rejection_reasons: list[dict[str, Any]] = [
        *incumbent_guardrail_failures,
        *(failure for failure in guardrail_failures if failure not in incumbent_guardrail_failures),
        *(
            failure
            for failure in training_support_failures
            if failure not in incumbent_guardrail_failures and failure not in guardrail_failures
        ),
    ]
    if not accepted and candidate.score >= incumbent.score - _ROLLOUT_SELECTION_MIN_DELTA:
        rejection_reasons.append(
            {
                "metric": "selection_score",
                "direction": "minimum_improvement",
                "candidate": candidate.score,
                "reference": incumbent.score,
                "limit": incumbent.score - _ROLLOUT_SELECTION_MIN_DELTA,
                "delta": candidate.score - incumbent.score,
            }
        )
    metrics: dict[str, Any] = {
        "validation_total_loss": float(validation.total_loss.detach().cpu()),
        "validation_rollout_loss": float(
            validation.loss_terms.get("rollout", validation.total_loss).detach().cpu()
        ),
        "validation_rollout_position_loss": float(
            validation.loss_terms.get("rollout_position", validation.total_loss).detach().cpu()
        ),
        "selection_accepted": float(accepted),
        "selection_rejection_reason_count": float(len(rejection_reasons)),
        "selection_rejection_reasons": rejection_reasons,
        "selection_reference_guardrail_failures": guardrail_failures,
        "selection_incumbent_guardrail_failures": incumbent_guardrail_failures,
        "selection_training_support_failures": training_support_failures,
        "selection_training_support_required": float(training_support_required),
        "selection_training_support_passed": float(
            not training_support_required or not training_support_failures
        ),
        "selection_mutable_training_support_failures": mutable_training_support_failures,
        "selection_mutable_training_support_passed": float(
            not training_support_required or not mutable_training_support_failures
        ),
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        "best_rollout_validated": 1.0,
        "incomplete_reference_comparison_required": 0.0,
        "best_measurement_validated": float(best_measurement is not None),
        **_validation_support_evidence(validation.metrics),
        **candidate.validation_metrics(),
        **incumbent.checkpoint_metrics(),
        **reference.checkpoint_metrics(prefix="reference_rollout"),
        **_validation_protocol_checkpoint_metrics(config),
        "checkpoint_model_state_hash": checkpoint_model_state_hash,
        "checkpoint_contains_best_rollout_weights": float(
            checkpoint_model_state_hash == incumbent_model_state_hash
        ),
        "best_rollout_model_state_hash": incumbent_model_state_hash,
        "best_rollout_checkpoint_step": float(incumbent_step),
        "checkpoint_contains_reference_rollout_weights": float(
            checkpoint_model_state_hash == reference_model_state_hash
        ),
        "reference_rollout_model_state_hash": reference_model_state_hash,
        "reference_rollout_checkpoint_step": float(reference_step),
        "rollout_reference_validated": 1.0,
    }
    if best_measurement is not None:
        metrics.update(best_measurement.checkpoint_metrics())
    return metrics


def _model_state_hash(model_state: Mapping[str, Any]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes in stable key order."""

    digest = hashlib.sha256()
    for name in sorted(model_state):
        value = model_state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, byteorder="big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, byteorder="big"))
        digest.update(raw)
    return digest.hexdigest()


def _current_model_state_hash(model: OnlineWorldModel) -> str:
    return _model_state_hash(model.state_dict())


def _verified_measurement_checkpoint(
    path: Path,
    config: OrpheusConfig,
    *,
    expected_model_state_hash: str | None,
    expected_step: int | None,
    expected_device: str | torch.device | None = None,
) -> tuple[_MeasurementSelectionMetrics, str, int] | None:
    """Verify that a measurement selector names the weights it actually stores."""

    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            return None
        metrics = payload.get("metrics")
        model_state = payload.get("model_state")
        if not isinstance(metrics, Mapping) or not isinstance(model_state, Mapping):
            return None
        if not _measurement_validation_protocol_is_compatible(
            payload,
            config,
            expected_device=expected_device,
        ):
            return None
        selection = _measurement_selection_from_checkpoint(metrics)
        if selection is None:
            return None
        if float(metrics.get("checkpoint_contains_best_measurement_weights", 0.0)) != 1.0:
            return None
        model_state_hash = _model_state_hash(model_state)
        if metrics.get("checkpoint_model_state_hash") != model_state_hash:
            return None
        if metrics.get("best_measurement_model_state_hash") != model_state_hash:
            return None
        checkpoint_step = _finite_nonnegative_integer(
            payload.get("step"),
            name="step",
        )
        linked_step = _finite_nonnegative_integer(
            metrics.get("best_measurement_checkpoint_step"),
            name="best_measurement_checkpoint_step",
        )
        if linked_step != checkpoint_step:
            return None
        if expected_model_state_hash is not None and (
            model_state_hash != expected_model_state_hash
        ):
            return None
        if expected_step is not None and checkpoint_step != expected_step:
            return None
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    return selection, model_state_hash, checkpoint_step


def _preserve_resume_measurement_checkpoint(
    resume_path: str | Path,
    destination: Path,
    config: OrpheusConfig,
    *,
    resume_metrics: Mapping[str, Any],
    expected_device: str | torch.device | None = None,
) -> tuple[_MeasurementSelectionMetrics, str, int] | None:
    """Copy a linked RGB selector only after metric/tensor verification."""

    expected_hash = resume_metrics.get("best_measurement_model_state_hash")
    expected_step_value = resume_metrics.get("best_measurement_checkpoint_step")
    if not isinstance(expected_hash, str) or expected_step_value is None:
        return None
    try:
        expected_step = _finite_nonnegative_integer(
            expected_step_value,
            name="best_measurement_checkpoint_step",
        )
    except ValueError:
        return None
    resumed = Path(resume_path).expanduser().resolve()
    source = (
        resumed.parent / "best_measurement.pt" if resumed.parent.name == "checkpoints" else resumed
    )
    verified = _verified_measurement_checkpoint(
        source,
        config,
        expected_model_state_hash=expected_hash,
        expected_step=expected_step,
        expected_device=expected_device,
    )
    if verified is None:
        return None
    if source != destination.resolve():
        shutil.copy2(source, destination)
        copied = _verified_measurement_checkpoint(
            destination,
            config,
            expected_model_state_hash=expected_hash,
            expected_step=expected_step,
            expected_device=expected_device,
        )
        if copied is None:
            return None
        verified = copied
    return verified


def _verified_selector_checkpoint(
    path: Path,
    config: OrpheusConfig,
    *,
    prefix: str,
    expected_model_state_hash: str | None,
    expected_step: int | None,
    expected_device: str | torch.device | None = None,
) -> tuple[_RolloutSelectionMetrics, str, int] | None:
    """Verify that selector metadata and the checkpoint's actual weights agree."""

    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            return None
        metrics = payload.get("metrics")
        model_state = payload.get("model_state")
        if not isinstance(metrics, Mapping) or not isinstance(model_state, Mapping):
            return None
        if not _rollout_validation_protocol_is_compatible(
            payload,
            config,
            expected_device=expected_device,
        ):
            return None
        if float(metrics.get("rollout_selection_metric_version", -1.0)) != (
            _ROLLOUT_SELECTION_METRIC_VERSION
        ):
            return None
        if _binary_checkpoint_marker(
            metrics.get("incomplete_reference_comparison_required"),
            name="incomplete_reference_comparison_required",
        ):
            return None
        if _validate_validation_support_schema(metrics, config) != 1.0:
            return None
        selection = _rollout_selection_from_checkpoint(
            metrics,
            config,
            prefix=prefix,
        )
        if selection is None or not _checkpoint_selection_matches_additive_evidence(
            metrics,
            config,
            prefix=prefix,
        ):
            return None
        if float(metrics.get(f"checkpoint_contains_{prefix}_weights", 0.0)) != 1.0:
            return None
        model_state_hash = _model_state_hash(model_state)
        if metrics.get("checkpoint_model_state_hash") != model_state_hash:
            return None
        if metrics.get(f"{prefix}_model_state_hash") != model_state_hash:
            return None
        checkpoint_step = _finite_nonnegative_integer(
            payload.get("step"),
            name="step",
        )
        linked_step = _finite_nonnegative_integer(
            metrics.get(f"{prefix}_checkpoint_step"),
            name=f"{prefix}_checkpoint_step",
        )
        if linked_step != checkpoint_step:
            return None
        if expected_model_state_hash is not None and (
            model_state_hash != expected_model_state_hash
        ):
            return None
        if expected_step is not None and checkpoint_step != expected_step:
            return None
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    return selection, model_state_hash, checkpoint_step


def _preserve_resume_selector_checkpoint(
    resume_path: str | Path,
    destination: Path,
    config: OrpheusConfig,
    *,
    prefix: str,
    resume_metrics: Mapping[str, Any],
    expected_device: str | torch.device | None = None,
) -> tuple[_RolloutSelectionMetrics, str, int] | None:
    """Copy a linked selector checkpoint only after weights/provenance verification."""

    expected_hash = resume_metrics.get(f"{prefix}_model_state_hash")
    expected_step_value = resume_metrics.get(f"{prefix}_checkpoint_step")
    if not isinstance(expected_hash, str) or expected_step_value is None:
        return None
    try:
        expected_step = _finite_nonnegative_integer(
            expected_step_value,
            name=f"{prefix}_checkpoint_step",
        )
    except ValueError:
        return None
    resumed = Path(resume_path).expanduser().resolve()
    source = resumed.parent / f"{prefix}.pt" if resumed.parent.name == "checkpoints" else resumed
    verified = _verified_selector_checkpoint(
        source,
        config,
        prefix=prefix,
        expected_model_state_hash=expected_hash,
        expected_step=expected_step,
        expected_device=expected_device,
    )
    if verified is None:
        return None
    if source != destination.resolve():
        shutil.copy2(source, destination)
        copied = _verified_selector_checkpoint(
            destination,
            config,
            prefix=prefix,
            expected_model_state_hash=expected_hash,
            expected_step=expected_step,
            expected_device=expected_device,
        )
        if copied is None:
            return None
        verified = copied
    return verified


def _verified_accepted_validation_checkpoint(
    path: Path,
    config: OrpheusConfig,
    *,
    maximum_step: int,
    expected_device: str | torch.device | None = None,
) -> tuple[int, str] | None:
    """Verify an accepted numbered checkpoint before retaining its history."""

    match = _NUMBERED_VALIDATION_CHECKPOINT.fullmatch(path.name)
    if match is None or not path.is_file():
        return None
    filename_step = int(match.group(1))
    if filename_step > maximum_step:
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            return None
        checkpoint_step = _finite_nonnegative_integer(
            payload.get("step"),
            name="step",
        )
        if checkpoint_step != filename_step:
            return None
        metrics = payload.get("metrics")
        model_state = payload.get("model_state")
        if not isinstance(metrics, Mapping) or not isinstance(model_state, Mapping):
            return None
        if not _rollout_validation_protocol_is_compatible(
            payload,
            config,
            expected_device=expected_device,
        ):
            return None
        if float(metrics.get("selection_accepted", 0.0)) != 1.0:
            return None
        if _binary_checkpoint_marker(
            metrics.get("incomplete_reference_comparison_required"),
            name="incomplete_reference_comparison_required",
        ):
            return None
        if _validate_validation_support_schema(metrics, config) != 1.0:
            return None
        if float(metrics.get("checkpoint_contains_best_rollout_weights", 0.0)) != 1.0:
            return None
        linked_step = _finite_nonnegative_integer(
            metrics.get("best_rollout_checkpoint_step"),
            name="best_rollout_checkpoint_step",
        )
        if linked_step != checkpoint_step:
            return None
        model_state_hash = _model_state_hash(model_state)
        if metrics.get("checkpoint_model_state_hash") != model_state_hash:
            return None
        if metrics.get("best_rollout_model_state_hash") != model_state_hash:
            return None
        selection = _rollout_selection_from_checkpoint(metrics, config)
        if selection is None or not _checkpoint_selection_matches_additive_evidence(
            metrics,
            config,
            prefix="best_rollout",
        ):
            return None
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    return checkpoint_step, model_state_hash


def _preserve_resume_validation_history(
    resume_path: str | Path,
    destination_directory: Path,
    config: OrpheusConfig,
    *,
    resume_step: int,
    expected_device: str | torch.device | None = None,
) -> tuple[Path, ...]:
    """Copy tensor-verified accepted numbered history into a branched run."""

    resumed = Path(resume_path).expanduser().resolve()
    source_directory = resumed.parent if resumed.parent.name == "checkpoints" else None
    if source_directory is None:
        return ()
    destination_directory = destination_directory.resolve()
    if source_directory == destination_directory:
        return tuple(
            path
            for path in sorted(destination_directory.glob("validation_step_*.pt"))
            if _verified_accepted_validation_checkpoint(
                path,
                config,
                maximum_step=resume_step,
                expected_device=expected_device,
            )
            is not None
        )
    preserved: list[Path] = []
    for source in sorted(source_directory.glob("validation_step_*.pt")):
        verified = _verified_accepted_validation_checkpoint(
            source,
            config,
            maximum_step=resume_step,
            expected_device=expected_device,
        )
        if verified is None:
            continue
        destination = destination_directory / source.name
        if destination.exists():
            copied = _verified_accepted_validation_checkpoint(
                destination,
                config,
                maximum_step=resume_step,
                expected_device=expected_device,
            )
            if copied != verified:
                raise ValueError(
                    "branched run contains a conflicting numbered validation "
                    f"checkpoint: {destination}"
                )
        else:
            shutil.copy2(source, destination)
            copied = _verified_accepted_validation_checkpoint(
                destination,
                config,
                maximum_step=resume_step,
                expected_device=expected_device,
            )
            if copied != verified:
                raise RuntimeError(
                    f"copied numbered validation checkpoint failed verification: {destination}"
                )
        preserved.append(destination)
    return tuple(preserved)


def _fresh_causal_optimizer_state(
    optimizer: torch.optim.AdamW,
    *,
    learning_rate: float,
    weight_decay: float,
) -> None:
    """Start the causal phase without perception-stage Adam moments."""

    optimizer.state.clear()
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = float(learning_rate)
        parameter_group["weight_decay"] = float(weight_decay)


def _repository_root(config: OrpheusConfig) -> Path:
    if config.source_path:
        source = Path(config.source_path).resolve()
        if source.parent.name == "configs":
            return source.parent.parent
        return source.parent
    return Path.cwd().resolve()


def _output_root(config: OrpheusConfig) -> Path:
    output = Path(config.project.output_dir).expanduser()
    if not output.is_absolute():
        output = _repository_root(config) / output
    return output.resolve()


def _new_run_name(config: OrpheusConfig) -> str:
    project = re.sub(r"[^A-Za-z0-9._-]+", "-", config.project.name).strip("-")
    return timestamped_artifact_path(project or "orpheus").name


def _resolve_run_directory(
    config: OrpheusConfig,
    *,
    run_name: str | None,
    resume_path: str | Path | None,
) -> Path:
    if resume_path is not None and run_name is None:
        checkpoint = Path(resume_path).expanduser().resolve()
        if checkpoint.parent.name != "checkpoints" or checkpoint.name != "last.pt":
            raise ValueError(
                "an in-place exact resume requires the source run's exact "
                "checkpoints/last.pt; resume from a selector or numbered "
                "checkpoint with --run-name, or use --initialize-from"
            )
        return checkpoint.parent.parent
    selected = (
        timestamped_artifact_path(run_name).name if run_name is not None else _new_run_name(config)
    )
    if not _SAFE_RUN_NAME.fullmatch(selected):
        raise ValueError(
            "run_name must contain only letters, digits, '.', '_', and '-' "
            "and cannot begin with punctuation"
        )
    return _output_root(config) / selected


def _next_batch(
    loader: DataLoader[dict[str, Any]],
    iterator: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _make_loader(
    config: OrpheusConfig,
    *,
    split: str,
    episodes: int,
    shuffle: bool,
    start_step: int | None = None,
    stop_step: int | None = None,
    batch_size_override: int | None = None,
) -> DataLoader[dict[str, Any]]:
    dataset = SyntheticSphereDataset(
        config,
        split=split,
        num_episodes=episodes,
        memory_cache=config.training.fixed_dataset,
    )
    if (start_step is None) != (stop_step is None):
        raise ValueError("start_step and stop_step must be supplied together")
    requested_batch_size = (
        config.training.batch_size if batch_size_override is None else batch_size_override
    )
    if requested_batch_size <= 0:
        raise ValueError("batch_size_override must be positive")
    batch_size = min(requested_batch_size, max(1, episodes))
    if start_step is not None and stop_step is not None:
        sampler_seed = config.project.seed + (0 if split == "train" else 10_000)
        if split == "train" and config.training.scenario_balanced_batches:
            scenario_count = len(config.simulator.scenario_mixture)
            batch_sampler = ScenarioBalancedStepIndexedBatchSampler(
                scenario_index_by_dataset_index=[
                    int(seed) % scenario_count for seed in dataset.manifest.seeds
                ],
                scenario_count=scenario_count,
                batch_size=batch_size,
                seed=sampler_seed,
                shuffle=shuffle,
                start_step=start_step,
                stop_step=stop_step,
            )
        else:
            batch_sampler = StepIndexedBatchSampler(
                dataset_size=len(dataset),
                batch_size=batch_size,
                seed=sampler_seed,
                shuffle=shuffle,
                start_step=start_step,
                stop_step=stop_step,
            )
        worker_generator = torch.Generator(device="cpu")
        worker_generator.manual_seed(config.project.seed + (20_000 if split == "train" else 30_000))
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.training.num_workers,
            collate_fn=collate_episodes,
            generator=worker_generator,
            **(
                {"prefetch_factor": 1, "persistent_workers": False}
                if config.training.num_workers > 0
                else {}
            ),
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.project.seed + (0 if split == "train" else 10_000))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        collate_fn=collate_episodes,
        drop_last=False,
        generator=generator,
        **(
            {"prefetch_factor": 1, "persistent_workers": False}
            if config.training.num_workers > 0
            else {}
        ),
    )


def _check_batch_major(batch: Mapping[str, Any]) -> None:
    rgb = batch.get("rgb")
    timestamps = batch.get("timestamps")
    if not isinstance(rgb, Tensor) or rgb.ndim != 5:
        raise ValueError("DataLoader must emit rgb with shape [B,T,3,H,W]")
    if not isinstance(timestamps, Tensor) or timestamps.shape != rgb.shape[:2]:
        raise ValueError("DataLoader must emit timestamps with shape [B,T]")


def _process_max_rss_bytes() -> float:
    """Return the process high-water RSS in bytes on supported Unix hosts."""

    maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the other supported Unix training hosts
    # report KiB. This is a high-water mark, not an instantaneous sample.
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024.0


def _release_accelerator_cache(previous_device: torch.device) -> None:
    """Release cached accelerator pages after moving the complete model away."""

    gc.collect()
    if previous_device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif previous_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _result_metrics(
    result: TrainingBatchResult,
    *,
    learning_rate: float,
    gradient_norm: float | None = None,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "phase": result.phase,
        "loss_total": float(result.total_loss.detach().cpu()),
        "learning_rate": float(learning_rate),
        "process_max_rss_bytes": _process_max_rss_bytes(),
    }
    for name, value in result.loss_terms.items():
        metrics[f"loss_{name}"] = float(value.detach().cpu())
    for name, value in result.metrics.items():
        if math.isfinite(value):
            metrics[name] = float(value)
    if gradient_norm is not None:
        metrics["gradient_norm"] = float(gradient_norm)
    return metrics


def _gradient_clip_diagnostics(
    pre_clip_gradient_norm: float,
    maximum_gradient_norm: float,
) -> tuple[float, float]:
    """Mirror PyTorch clipping and return ``(coefficient, applied_norm)``."""

    gradient_clip_coefficient = min(
        1.0,
        maximum_gradient_norm / (pre_clip_gradient_norm + 1.0e-6),
    )
    return (
        gradient_clip_coefficient,
        pre_clip_gradient_norm * gradient_clip_coefficient,
    )


def _clip_training_gradients(
    model: OnlineWorldModel,
    config: OrpheusConfig,
    *,
    apply_perception_local_clip: bool = True,
) -> dict[str, float]:
    """Clip RGB perception and learned interactions, then the complete model.

    During closed-loop adaptation, RGB discovery/ROI objectives and a learned
    edge residual can each produce a much larger gradient than otherwise
    healthy filter/rollout gradients. A single global clip would shrink every
    subsystem by an unrelated outlier's coefficient. Disjoint local caps
    preserve the full forward capacity while preventing either subsystem from
    monopolizing an update. Measurement pretraining deliberately disables the
    RGB-local cap and retains its established whole-model clipping behavior.

    ``clip_grad_norm_`` returns the norm before its own mutation. Since the
    perception and interaction parameters are disjoint strict subsets of the
    complete model, the raw whole-model norm can be reconstructed exactly from
    the three returned norms.
    """

    perception_parameters = tuple(model.observation_modules["rgb"].parameters())
    interaction_parameters = tuple(model.dynamics.interactions.parameters())
    if model.dynamics.attention_interactions is not None:
        interaction_parameters += tuple(model.dynamics.attention_interactions.parameters())
    perception_ids = {id(parameter) for parameter in perception_parameters}
    interaction_ids = {id(parameter) for parameter in interaction_parameters}
    if not perception_ids.isdisjoint(interaction_ids):
        raise AssertionError("perception and interaction clip groups must be disjoint")

    perception_clip_norm = (
        config.training.closed_loop_perception_grad_clip_norm
        if apply_perception_local_clip
        else math.inf
    )
    perception_pre_clip_tensor = torch.nn.utils.clip_grad_norm_(
        perception_parameters,
        perception_clip_norm,
        error_if_nonfinite=False,
    )
    if not bool(torch.isfinite(perception_pre_clip_tensor)):
        raise FloatingPointError("nonfinite RGB perception gradient norm")
    perception_pre_clip = float(perception_pre_clip_tensor.detach().cpu())
    perception_coefficient, perception_applied = _gradient_clip_diagnostics(
        perception_pre_clip,
        perception_clip_norm,
    )

    interaction_pre_clip_tensor = torch.nn.utils.clip_grad_norm_(
        interaction_parameters,
        config.training.interaction_grad_clip_norm,
        error_if_nonfinite=False,
    )
    if not bool(torch.isfinite(interaction_pre_clip_tensor)):
        raise FloatingPointError("nonfinite learned-interaction gradient norm")
    interaction_pre_clip = float(interaction_pre_clip_tensor.detach().cpu())
    interaction_coefficient, interaction_applied = _gradient_clip_diagnostics(
        interaction_pre_clip,
        config.training.interaction_grad_clip_norm,
    )

    pre_global_clip_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.training.grad_clip_norm,
        error_if_nonfinite=False,
    )
    if not bool(torch.isfinite(pre_global_clip_tensor)):
        raise FloatingPointError("nonfinite whole-model gradient norm")
    pre_global_clip = float(pre_global_clip_tensor.detach().cpu())
    global_coefficient, applied = _gradient_clip_diagnostics(
        pre_global_clip,
        config.training.grad_clip_norm,
    )
    raw_squared = (
        pre_global_clip * pre_global_clip
        - perception_applied * perception_applied
        + perception_pre_clip * perception_pre_clip
        - interaction_applied * interaction_applied
        + interaction_pre_clip * interaction_pre_clip
    )
    raw = math.sqrt(max(0.0, raw_squared))
    total_coefficient = applied / raw if raw > 0.0 else 1.0
    return {
        "gradient_norm_pre_clip": raw,
        "perception_gradient_norm_pre_clip": perception_pre_clip,
        "perception_gradient_local_clip_enabled": float(apply_perception_local_clip),
        "perception_gradient_clip_coefficient": perception_coefficient,
        "perception_gradient_norm_applied_before_global_clip": perception_applied,
        "interaction_gradient_norm_pre_clip": interaction_pre_clip,
        "interaction_gradient_clip_coefficient": interaction_coefficient,
        "interaction_gradient_norm_applied_before_global_clip": interaction_applied,
        "gradient_norm_pre_global_clip": pre_global_clip,
        "gradient_clip_coefficient": global_coefficient,
        "gradient_total_clip_coefficient": total_coefficient,
        "gradient_norm_applied": applied,
    }


def _assert_finite_optimizer_update(
    model: OnlineWorldModel,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Reject a non-finite parameter or optimizer state immediately post-step."""

    named_parameters = dict(model.named_parameters())
    _assert_finite_tensor_tree(
        named_parameters,
        root="model_parameters",
    )
    parameter_names = {id(parameter): name for name, parameter in named_parameters.items()}
    named_optimizer_state = {
        parameter_names.get(id(parameter), f"unnamed_parameter_{index}"): state
        for index, (parameter, state) in enumerate(optimizer.state.items())
    }
    _assert_finite_tensor_tree(
        named_optimizer_state,
        root="optimizer_state",
    )
    _assert_valid_optimizer_steps(
        named_optimizer_state,
        root="optimizer_state",
    )


def _rollout_validation_protocol_is_compatible(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    expected_device: str | torch.device | None = None,
) -> bool:
    """Require an exact canonical match for the deterministic validation split."""

    metrics = payload.get("metrics")
    checkpoint_config = payload.get("config")
    if not isinstance(metrics, Mapping) or not isinstance(checkpoint_config, Mapping):
        return False
    try:
        selector_metric_version = float(metrics.get("rollout_selection_metric_version", math.nan))
    except (TypeError, ValueError):
        return False
    if selector_metric_version != _ROLLOUT_SELECTION_METRIC_VERSION:
        return False
    stored_hash = metrics.get("rollout_validation_protocol_hash")
    if not isinstance(stored_hash, str):
        return False
    if expected_device is not None and payload.get("device") != str(expected_device):
        return False
    try:
        checkpoint_hash = _canonical_hash(
            _rollout_validation_protocol_from_mapping(checkpoint_config)
        )
        requested_hash = _rollout_validation_protocol_hash(config)
    except (KeyError, TypeError, ValueError):
        return False
    return stored_hash == checkpoint_hash == requested_hash


def _rollout_selection_is_compatible(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> bool:
    """Return whether a resumed best score uses this objective's semantics."""

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    try:
        version = float(metrics.get("rollout_selection_metric_version", 0.0))
    except (TypeError, ValueError):
        return False
    if version != _ROLLOUT_SELECTION_METRIC_VERSION:
        return False
    if not _rollout_validation_protocol_is_compatible(payload, config):
        return False
    try:
        restored_selection = _rollout_selection_from_checkpoint(metrics, config)
    except (FloatingPointError, RuntimeError, TypeError, ValueError):
        return False
    return restored_selection is not None


def measurement_pretrain_frame_index(
    step: int,
    *,
    loader_batches: int,
    total_frames: int,
    fixed_dataset: bool,
) -> int:
    """Choose an adjacent-pair anchor without coupling batches to parity.

    Stage B jointly trains global discovery at this frame and the fast ROI
    update at ``frame + 1``. For a fixed dataset, every batch sees anchor zero
    before every batch sees anchor one, and so on. Streaming/shuffled datasets
    retain independently sampled adjacent pairs.
    """

    if loader_batches <= 0 or total_frames < 2:
        raise ValueError(
            "loader_batches must be positive and total_frames must contain an RGB pair"
        )
    pair_count = total_frames - 1
    if fixed_dataset:
        return (step // loader_batches) % pair_count
    return random.randrange(pair_count)


def set_global_perception_trainable(
    model: OnlineWorldModel,
    *,
    trainable: bool,
) -> None:
    """Control global-exclusive RGB modules without disabling the fast encoder.

    Backbone stages zero and one are shared by both paths, and
    ``fast_projection`` is ROI-only.  Freezing the entire backbone silently
    reduced ``state_dynamics_roi`` to the ROI output head and prevented the
    causal objective from adapting its visual features.
    """

    module = model.observation_modules["rgb"]
    backbone = getattr(module, "backbone", None)
    global_detector = getattr(module, "global_detector", None)
    if backbone is None or global_detector is None:
        raise TypeError("RGB module is missing backbone or global_detector")
    global_detector.requires_grad_(trainable)
    if trainable:
        backbone.requires_grad_(True)
        return
    for stage in backbone.stages[2:]:
        stage.requires_grad_(False)
    for projection in backbone.projections:
        projection.requires_grad_(False)


def set_closed_loop_trainable_scope(
    model: OnlineWorldModel,
    *,
    scope: str,
) -> None:
    """Restrict adaptation without changing the complete RGB runtime path."""

    if scope == "all":
        model.requires_grad_(True)
        _freeze_disconnected_training_heads(model)
        return
    model.requires_grad_(False)
    if scope == "attention":
        if model.dynamics.attention_interactions is None:
            raise ValueError("attention scope requires typed attention dynamics")
        model.dynamics.attention_interactions.requires_grad_(True)
        return
    if scope == "dynamics":
        model.dynamics.requires_grad_(True)
        return
    if scope == "updater":
        model.updater.requires_grad_(True)
        _freeze_disconnected_training_heads(model)
        return
    if scope in {"updater_mean", "updater_mean_y"}:
        corrector = model.updater.learned_corrector
        if corrector is None:
            raise ValueError(f"{scope} scope requires the learned fast corrector")
        corrector.mean_head.requires_grad_(True)
        return
    if scope == "state_dynamics":
        model.dynamics.requires_grad_(True)
        model.updater.requires_grad_(True)
        if model.identifier is not None:
            model.identifier.requires_grad_(True)
        _freeze_disconnected_training_heads(model)
        return
    if scope in {"fast_roi", "state_dynamics_fast_roi", "state_dynamics_roi"}:
        if scope != "fast_roi":
            model.dynamics.requires_grad_(True)
            model.updater.requires_grad_(True)
            if model.identifier is not None:
                model.identifier.requires_grad_(True)
        rgb_module = model.observation_modules["rgb"]
        roi_updater = getattr(rgb_module, "roi_updater", None)
        if roi_updater is None:
            raise TypeError("RGB module is missing roi_updater")
        backbone = getattr(rgb_module, "backbone", None)
        if backbone is None:
            raise TypeError("RGB module is missing backbone")
        if scope == "state_dynamics_roi":
            for stage in backbone.stages[:2]:
                stage.requires_grad_(True)
        backbone.fast_projection.requires_grad_(True)
        roi_updater.requires_grad_(True)
        _freeze_disconnected_training_heads(model)
        return
    raise ValueError(
        "closed-loop trainable scope must be 'all', 'attention', 'dynamics', 'updater', "
        "'updater_mean', 'updater_mean_y', 'fast_roi', 'state_dynamics', "
        "'state_dynamics_fast_roi', or "
        "'state_dynamics_roi'"
    )


def _prepare_restricted_updater_mean_update(
    model: OnlineWorldModel,
    optimizer: torch.optim.AdamW,
    *,
    scope: str,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Mask and snapshot rows excluded by an axis-restricted mean scope.

    AdamW applies decoupled weight decay and can retain per-element moments,
    so a zero gradient alone does not freeze a row.  Snapshotting the excluded
    values and clearing their matching optimizer-state rows makes the scope an
    exact tensor contract, including after resume or a configured scope
    transition.
    """

    selected_row = {"updater_mean_y": 1}.get(scope)
    if selected_row is None:
        return []
    corrector = model.updater.learned_corrector
    if corrector is None:
        raise ValueError(f"{scope} scope requires the learned fast corrector")
    snapshots: list[tuple[Tensor, Tensor, Tensor]] = []
    for parameter in (corrector.mean_head.weight, corrector.mean_head.bias):
        if parameter.ndim == 0 or selected_row >= parameter.shape[0]:
            raise ValueError(f"{scope} is incompatible with the learned mean-head shape")
        frozen_rows = torch.as_tensor(
            [row for row in range(parameter.shape[0]) if row != selected_row],
            dtype=torch.int64,
            device=parameter.device,
        )
        frozen_values = parameter.detach().index_select(0, frozen_rows).clone()
        if parameter.grad is not None:
            parameter.grad.index_fill_(0, frozen_rows, 0)
        state = optimizer.state.get(parameter, {})
        for value in state.values():
            if isinstance(value, Tensor) and value.shape == parameter.shape:
                value.index_fill_(0, frozen_rows, 0)
        snapshots.append((parameter, frozen_rows, frozen_values))
    return snapshots


def _restore_restricted_updater_mean_update(
    optimizer: torch.optim.AdamW,
    snapshots: list[tuple[Tensor, Tensor, Tensor]],
) -> None:
    """Restore excluded rows after AdamW and keep their moments exactly zero."""

    with torch.no_grad():
        for parameter, frozen_rows, frozen_values in snapshots:
            parameter.index_copy_(0, frozen_rows, frozen_values)
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if isinstance(value, Tensor) and value.shape == parameter.shape:
                    value.index_fill_(0, frozen_rows, 0)


def _closed_loop_trainable_scope_for_step(
    config: OrpheusConfig,
    *,
    completed_step: int,
) -> tuple[str, bool]:
    """Resolve the declared causal scope at one completed-update boundary."""

    primary = config.training.closed_loop_trainable_scope
    late = config.training.closed_loop_late_trainable_scope
    transition = config.training.closed_loop_scope_transition_steps
    if late is None or transition is None:
        return primary, False
    causal_updates = max(0, completed_step - config.training.rgb_pretrain_steps)
    if causal_updates >= transition:
        return late, True
    return primary, False


def _freeze_disconnected_training_heads(model: OnlineWorldModel) -> None:
    """Keep scoped trainability truthful until these outputs have objectives.

    The ROI event vector is persisted in measurement auxiliary data but is not
    yet consumed by the fast corrector.  Identifier variance controls slow
    uncertainty contraction at runtime, but no calibrated slow-uncertainty
    objective currently reaches it.  Retain both modules in checkpoints while
    excluding their disconnected tensors from autograd updates.
    """

    rgb_module = model.observation_modules["rgb"]
    roi_updater = getattr(rgb_module, "roi_updater", None)
    event_head = getattr(roi_updater, "event_head", None)
    if event_head is not None:
        event_head.requires_grad_(False)
    learned_corrector = getattr(model.updater, "learned_corrector", None)
    corrector_visibility_head = getattr(learned_corrector, "visibility_head", None)
    if corrector_visibility_head is not None:
        # RGB supplies an explicit visibility measurement after the learned
        # correction, so the current runtime deliberately overwrites this
        # delta.  Do not advertise or optimise a parameter with no path to the
        # posterior or objective.
        corrector_visibility_head.requires_grad_(False)
    if model.identifier is not None:
        model.identifier.variance_head.requires_grad_(False)


def _mean_batch_results(
    results: list[TrainingBatchResult],
    *,
    weights: list[float] | None = None,
) -> TrainingBatchResult:
    if not results:
        raise ValueError("cannot average an empty validation result list")
    if weights is None:
        weights = [1.0] * len(results)
    if len(weights) != len(results):
        raise ValueError("validation result weights must match the result count")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("validation result weights must be finite and positive")
    total_weight = float(sum(weights))
    phase = results[0].phase
    if any(result.phase != phase for result in results):
        raise ValueError("validation results must share a phase")
    term_names = set().union(*(result.loss_terms for result in results))
    metric_names = set().union(*(result.metrics for result in results))

    def mean_present_term(name: str) -> Tensor:
        present = [
            (result.loss_terms[name], float(weight))
            for result, weight in zip(results, weights, strict=True)
            if name in result.loss_terms
        ]
        present_weight = sum(weight for _, weight in present)
        return torch.stack([value * weight for value, weight in present]).sum() / present_weight

    def mean_present_metric(name: str) -> float:
        present = [
            (float(result.metrics[name]), float(weight))
            for result, weight in zip(results, weights, strict=True)
            if name in result.metrics and math.isfinite(float(result.metrics[name]))
        ]
        if not present:
            return math.nan
        if name in _MEASUREMENT_ADDITIVE_METRICS:
            # These values already count every object/proposal in their batch.
            # Summing (without multiplying by batch size again) permits exact
            # pooling across frames, episodes, and nested aggregation calls.
            return sum(value for value, _ in present)
        present_weight = sum(weight for _, weight in present)
        return sum(value * weight for value, weight in present) / present_weight

    aggregate_metrics = {name: mean_present_metric(name) for name in sorted(metric_names)}
    measurement_count_keys = {
        "true_positive": "rgb_runtime_birth_true_positive_count_at_0_5m",
        "target": "rgb_runtime_birth_target_count",
        "proposal": "rgb_runtime_birth_proposal_count",
        "runtime_error": "rgb_runtime_birth_world_position_absolute_error_sum_m",
        "runtime_matched": "rgb_runtime_birth_matched_proposal_count",
        "all_error": "rgb_world_position_absolute_error_sum_m",
        "all_matched": "rgb_world_position_matched_proposal_count",
    }
    if all(key in aggregate_metrics for key in measurement_count_keys.values()):
        counts = {name: aggregate_metrics[key] for name, key in measurement_count_keys.items()}
        recall = counts["true_positive"] / max(counts["target"], 1.0)
        precision = counts["true_positive"] / max(counts["proposal"], 1.0)
        f1 = 2.0 * recall * precision / max(recall + precision, 1.0e-8)
        runtime_mae = (
            counts["runtime_error"] / counts["runtime_matched"]
            if counts["runtime_matched"] > 0
            else math.inf
        )
        all_mae = (
            counts["all_error"] / counts["all_matched"] if counts["all_matched"] > 0 else math.inf
        )
        aggregate_metrics.update(
            {
                "rgb_detection_recall_at_0_5m": recall,
                "rgb_detection_precision_at_0_5m": precision,
                "rgb_runtime_birth_recall_at_0_5m": recall,
                "rgb_runtime_birth_precision_at_0_5m": precision,
                "rgb_runtime_birth_f1_at_0_5m": f1,
                "rgb_runtime_birth_world_position_mae_m": runtime_mae,
                "rgb_world_position_mae_m": all_mae,
            }
        )
    fast_count_keys = {
        "bootstrap_matched": "rgb_fast_bootstrap_matched_target_count",
        "bootstrap_target": "rgb_fast_bootstrap_target_count",
        "roi_supported": "rgb_fast_roi_supported_target_count",
        "roi_target": "rgb_fast_roi_target_count",
        "eligible": "rgb_fast_roi_eligible_proposal_count",
        "roi_error": "rgb_fast_roi_world_position_absolute_error_sum_m",
        "roi_matched": "rgb_fast_roi_world_position_matched_count",
        "prior_error": "rgb_fast_prior_world_position_absolute_error_sum_m",
        "prior_matched": "rgb_fast_prior_world_position_matched_count",
        "confident": "rgb_fast_roi_confident_proposal_count",
        "true_positive": "rgb_fast_roi_true_positive_count_at_0_5m",
    }
    if all(key in aggregate_metrics for key in fast_count_keys.values()):
        counts = {name: aggregate_metrics[key] for name, key in fast_count_keys.items()}
        bootstrap_coverage = counts["bootstrap_matched"] / max(
            counts["bootstrap_target"],
            1.0,
        )
        roi_coverage = counts["roi_supported"] / max(counts["roi_target"], 1.0)
        roi_recall = counts["true_positive"] / max(counts["roi_target"], 1.0)
        roi_precision = counts["true_positive"] / max(counts["confident"], 1.0)
        roi_f1 = 2.0 * roi_recall * roi_precision / max(roi_recall + roi_precision, 1.0e-8)
        roi_mae = (
            counts["roi_error"] / counts["roi_matched"] if counts["roi_matched"] > 0 else math.inf
        )
        prior_mae = (
            counts["prior_error"] / counts["prior_matched"]
            if counts["prior_matched"] > 0
            else math.inf
        )
        improvement = (
            prior_mae - roi_mae if math.isfinite(prior_mae) and math.isfinite(roi_mae) else math.nan
        )
        aggregate_metrics.update(
            {
                "rgb_fast_bootstrap_target_coverage": bootstrap_coverage,
                "rgb_fast_roi_target_coverage": roi_coverage,
                "rgb_fast_roi_world_position_mae_m": roi_mae,
                "rgb_fast_prior_world_position_mae_m": prior_mae,
                "rgb_fast_roi_improvement_m": improvement,
                "rgb_fast_roi_recall_at_0_5m": roi_recall,
                "rgb_fast_roi_precision_at_0_5m": roi_precision,
                "rgb_fast_roi_f1_at_0_5m": roi_f1,
            }
        )

    return TrainingBatchResult(
        total_loss=(
            torch.stack(
                [
                    result.total_loss * float(weight)
                    for result, weight in zip(results, weights, strict=True)
                ]
            ).sum()
            / total_weight
        ),
        loss_terms={name: mean_present_term(name) for name in sorted(term_names)},
        metrics=aggregate_metrics,
        phase=phase,
    )


@torch.no_grad()
def _validation_step(
    model: OnlineWorldModel,
    batch: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    closed_loop: bool,
) -> TrainingBatchResult:
    was_training = model.training
    model.eval()
    try:
        if closed_loop:
            result = run_closed_loop_batch(
                model,
                batch,
                config,
                window_start=0,
                window_steps=int(batch["rgb"].shape[1]),
                apply_perturbations=False,
                include_measurement_supervision=True,
                rollout_anchors_per_window=(config.training.validation_rollout_anchors_per_episode),
                compute_future_correction=False,
            )
        else:
            total_frames = int(batch["rgb"].shape[1])
            if total_frames < 2:
                raise ValueError("measurement validation requires an adjacent RGB pair")
            pair_count = total_frames - 1
            frame_count = min(
                pair_count,
                config.training.measurement_validation_frames,
            )
            if frame_count == pair_count:
                frame_indices = list(range(pair_count))
            else:
                frame_indices = (
                    torch.linspace(0, pair_count - 1, frame_count)
                    .round()
                    .to(dtype=torch.int64)
                    .unique(sorted=True)
                    .tolist()
                )
            result = _mean_batch_results(
                [
                    pretrain_rgb_measurements(
                        model,
                        batch,
                        config,
                        frame_index=int(frame_index),
                    )
                    for frame_index in frame_indices
                ]
            )
    finally:
        model.train(was_training)
    return result


def _aggregate_physical_validation_metrics(
    results: list[TrainingBatchResult],
    config: OrpheusConfig,
    *,
    minimum_predictable_target_count: int = 1,
    minimum_matched_target_count: int = 1,
) -> dict[str, float]:
    """Derive exact split-level physical metrics from additive batch counts."""

    for name, value in (
        ("minimum_predictable_target_count", minimum_predictable_target_count),
        ("minimum_matched_target_count", minimum_matched_target_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    additive: dict[str, float] = {}
    for result in results:
        for name, value in result.metrics.items():
            if _is_additive_physical_metric(name):
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0:
                    raise ValueError(f"additive physical validation metric {name!r} is invalid")
                additive[name] = additive.get(name, 0.0) + numeric
    # Preserve the exact split totals as well as ratios derived from them.
    # Averaging per-batch ``*_count`` diagnostics produces fractional values
    # and makes checkpoint audit trails misleading even when the ratios happen
    # to be recomputed correctly.
    insufficient_support = False
    for suffix, _ in _selection_horizon_keys(config):
        predictable_key = f"physical_forecast_predictable_target_count@{suffix}"
        if predictable_key not in additive:
            raise RuntimeError(f"missing additive physical validation metric {predictable_key!r}")
        if additive[predictable_key] < minimum_predictable_target_count:
            insufficient_support = True
        coordinate_key = f"physical_rollout_position@{suffix}_coordinate_count"
        if coordinate_key not in additive:
            raise RuntimeError(f"missing additive physical validation metric {coordinate_key!r}")
        required_coordinates = 3 * minimum_matched_target_count
        if additive[coordinate_key] < required_coordinates:
            insufficient_support = True
    try:
        derived = physical_validation_metrics(additive, config)
    except PhysicalMetricSupportError:
        insufficient_support = True
        derived = None
    if insufficient_support:
        # A candidate with no valid current or horizon mapping is a truthful
        # unsupported selection result, not a trainer crash and not a
        # fabricated zero-RMSE example. Keep its additive evidence and let the
        # selector persist/reject it explicitly.
        return {
            **additive,
            "selection_metric_supported": 0.0,
        }
    if derived is None:
        raise AssertionError("supported physical validation did not derive metrics")
    return {
        **additive,
        "selection_metric_supported": 1.0,
        **derived,
    }


def _validation_protocol_progress_fields(
    config: OrpheusConfig,
    *,
    closed_loop: bool,
) -> dict[str, str]:
    if closed_loop:
        return {
            "protocol_kind": "rollout",
            "protocol_hash": _rollout_validation_protocol_hash(config),
        }
    return {
        "protocol_kind": "measurement",
        "protocol_hash": _measurement_validation_protocol_hash(config),
    }


def _write_validation_progress(
    path: Path,
    *,
    config: OrpheusConfig,
    split: str,
    closed_loop: bool,
    state: str,
    completed_batches: int,
    total_batches: int | None,
    completed_episodes: int,
    total_episodes: int | None,
    elapsed_seconds: float,
    last_batch_seconds: float | None = None,
    last_seed: int | None = None,
    last_scenario: str | None = None,
    exception_type: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "split": split,
        "validation_kind": "closed_loop" if closed_loop else "measurement",
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "elapsed_seconds": elapsed_seconds,
        **_validation_protocol_progress_fields(
            config,
            closed_loop=closed_loop,
        ),
    }
    if last_batch_seconds is not None:
        payload["last_batch_seconds"] = last_batch_seconds
    if last_seed is not None:
        payload["last_seed"] = last_seed
    if last_scenario is not None:
        payload["last_scenario"] = last_scenario
    if exception_type is not None:
        payload["exception_type"] = exception_type
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def _validation_loader_result(
    model: OnlineWorldModel,
    loader: DataLoader[dict[str, Any]],
    config: OrpheusConfig,
    *,
    device: torch.device,
    closed_loop: bool,
    progress_path: Path | None = None,
    progress_split: str = "validation",
) -> TrainingBatchResult:
    """Evaluate every configured validation episode exactly once."""

    results: list[TrainingBatchResult] = []
    batch_sizes: list[float] = []
    scenarios: list[str] = []
    seeds: list[int] = []
    attribution_available = True
    validation_started = time.perf_counter()
    completed_episodes = 0
    try:
        total_batches: int | None = len(loader)
    except TypeError:
        total_batches = None
    dataset = getattr(loader, "dataset", None)
    try:
        total_episodes: int | None = len(dataset) if dataset is not None else None
    except TypeError:
        total_episodes = None
    if progress_path is not None:
        _write_validation_progress(
            progress_path,
            config=config,
            split=progress_split,
            closed_loop=closed_loop,
            state="validation_running",
            completed_batches=0,
            total_batches=total_batches,
            completed_episodes=0,
            total_episodes=total_episodes,
            elapsed_seconds=0.0,
        )
    try:
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch_started = time.perf_counter()
            _check_batch_major(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            results.append(
                _validation_step(
                    model,
                    batch,
                    config,
                    closed_loop=closed_loop,
                )
            )
            if not bool(torch.isfinite(results[-1].total_loss)):
                phase = "closed-loop" if closed_loop else "measurement"
                raise FloatingPointError(f"nonfinite {phase} validation loss")
            batch_size = int(batch["rgb"].shape[0])
            batch_sizes.append(float(batch_size))
            completed_episodes += batch_size
            metadata = raw_batch.get("metadata")
            scenario_values = metadata.get("scenario") if isinstance(metadata, Mapping) else None
            seed_values = raw_batch.get("seed")
            last_scenario = (
                str(scenario_values[0])
                if isinstance(scenario_values, list) and len(scenario_values) == 1
                else None
            )
            last_seed = (
                int(seed_values.item())
                if isinstance(seed_values, Tensor) and seed_values.numel() == 1
                else None
            )
            if last_scenario is None or last_seed is None:
                # Production validation loaders are deliberately batch-one and
                # always provide this metadata.  Keep the aggregation helper
                # usable for synthetic/custom loaders while making the absence
                # of per-episode attribution explicit instead of rejecting
                # otherwise valid pooled metrics.
                attribution_available = False
            else:
                scenarios.append(last_scenario)
                seeds.append(last_seed)
            elapsed = time.perf_counter() - validation_started
            batch_seconds = time.perf_counter() - batch_started
            print(
                "validation progress "
                f"split={progress_split} "
                f"kind={'closed_loop' if closed_loop else 'measurement'} "
                f"batches={batch_index}/{total_batches if total_batches is not None else '?'} "
                f"episodes={completed_episodes}/"
                f"{total_episodes if total_episodes is not None else '?'} "
                f"last_seed={last_seed if last_seed is not None else '?'} "
                f"last_scenario={last_scenario if last_scenario is not None else '?'} "
                f"batch_seconds={batch_seconds:.3f} elapsed_seconds={elapsed:.3f}",
                flush=True,
            )
            if progress_path is not None:
                _write_validation_progress(
                    progress_path,
                    config=config,
                    split=progress_split,
                    closed_loop=closed_loop,
                    state="validation_running",
                    completed_batches=batch_index,
                    total_batches=total_batches,
                    completed_episodes=completed_episodes,
                    total_episodes=total_episodes,
                    elapsed_seconds=elapsed,
                    last_batch_seconds=batch_seconds,
                    last_seed=last_seed,
                    last_scenario=last_scenario,
                )
    except BaseException as error:
        if progress_path is not None:
            _write_validation_progress(
                progress_path,
                config=config,
                split=progress_split,
                closed_loop=closed_loop,
                state="validation_interrupted",
                completed_batches=len(results),
                total_batches=total_batches,
                completed_episodes=completed_episodes,
                total_episodes=total_episodes,
                elapsed_seconds=time.perf_counter() - validation_started,
                exception_type=type(error).__name__,
            )
        raise
    aggregate = _mean_batch_results(results, weights=batch_sizes)
    if closed_loop:
        aggregate.metrics.update(
            _aggregate_physical_validation_metrics(
                results,
                config,
            )
        )
        aggregate.metrics["validation_attribution_available"] = float(
            attribution_available and len(scenarios) == len(results)
        )
        if attribution_available and len(scenarios) == len(results):
            by_scenario: dict[str, list[TrainingBatchResult]] = {}
            for scenario, result in zip(scenarios, results, strict=True):
                by_scenario.setdefault(scenario, []).append(result)
            for scenario, scenario_results in sorted(by_scenario.items()):
                slug = re.sub(r"[^a-z0-9]+", "_", scenario.lower()).strip("_")
                aggregate.metrics[f"scenario_{slug}_episode_count"] = float(len(scenario_results))
                derived = _aggregate_physical_validation_metrics(
                    scenario_results,
                    config,
                    minimum_predictable_target_count=(
                        config.training.validation_minimum_predictable_target_count_per_scenario_horizon
                    ),
                    minimum_matched_target_count=(
                        config.training.validation_minimum_matched_target_count_per_scenario_horizon
                    ),
                )
                supported_episode_count = sum(
                    float(
                        _aggregate_physical_validation_metrics(
                            [scenario_result],
                            config,
                        )["selection_metric_supported"]
                    )
                    for scenario_result in scenario_results
                )
                aggregate.metrics[f"scenario_{slug}_supported_episode_count"] = (
                    supported_episode_count
                )
                aggregate.metrics[f"scenario_{slug}_minimum_supported_episode_count"] = float(
                    config.training.validation_minimum_supported_episodes_per_scenario
                )
                scenario_supported = float(
                    derived["selection_metric_supported"] == 1.0
                    and supported_episode_count
                    >= config.training.validation_minimum_supported_episodes_per_scenario
                )
                aggregate.metrics[f"scenario_{slug}_selection_metric_supported"] = (
                    scenario_supported
                )
                for name, value in derived.items():
                    physical_support_diagnostic = _is_additive_physical_metric(name)
                    supported_selection_metric = scenario_supported == 1.0 and name.startswith(
                        "validation_"
                    )
                    if physical_support_diagnostic or supported_selection_metric:
                        aggregate.metrics[f"scenario_{slug}_{name}"] = value
            for seed, result in zip(seeds, results, strict=True):
                derived = _aggregate_physical_validation_metrics(
                    [result],
                    config,
                )
                seed_supported = float(derived["selection_metric_supported"])
                aggregate.metrics[f"seed_{seed}_selection_metric_supported"] = seed_supported
                if seed_supported == 0.0:
                    continue
                for name, value in derived.items():
                    if name.startswith("validation_"):
                        aggregate.metrics[f"seed_{seed}_{name}"] = value
    if progress_path is not None:
        _write_validation_progress(
            progress_path,
            config=config,
            split=progress_split,
            closed_loop=closed_loop,
            state="validation_complete",
            completed_batches=len(results),
            total_batches=total_batches,
            completed_episodes=completed_episodes,
            total_episodes=total_episodes,
            elapsed_seconds=time.perf_counter() - validation_started,
            last_seed=seeds[-1] if seeds else None,
            last_scenario=scenarios[-1] if scenarios else None,
        )
    return aggregate


def _write_run_metadata(
    path: Path,
    *,
    config: OrpheusConfig,
    device_info: DeviceInfo,
    active_start_device: torch.device,
    measurement_device: torch.device,
    closed_loop_device: torch.device,
    resume_path: str | Path | None,
    initialize_from_path: str | Path | None,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            # The atomic rewrite below repairs malformed metadata while the
            # durable checkpoint remains the authority for training state.
            existing = {}
    resolved_resume_path = (
        None if resume_path is None else str(Path(resume_path).expanduser().resolve())
    )
    requested_initialization_path = (
        None
        if initialize_from_path is None
        else str(Path(initialize_from_path).expanduser().resolve())
    )
    preserved_initialization_path = existing.get(
        "initialize_from_path",
        requested_initialization_path,
    )
    initial_source_provenance = existing.get(
        "initial_source_provenance",
        existing.get("source_provenance", dict(source_provenance)),
    )
    resume_history = existing.get("resume_history", [])
    resume_history = [] if not isinstance(resume_history, list) else list(resume_history)
    if resolved_resume_path is not None:
        resume_history.append(
            {
                "resumed_utc": now,
                "resume_path": resolved_resume_path,
                "active_start_device": str(active_start_device),
                "source_provenance": dict(source_provenance),
            }
        )
    payload = {
        "created_utc": existing.get("created_utc", now),
        "last_invocation_utc": now,
        "project": config.project.name,
        "seed": config.project.seed,
        "device": str(active_start_device),
        "initial_device": existing.get("initial_device", str(active_start_device)),
        "measurement_device": str(measurement_device),
        "closed_loop_device": str(closed_loop_device),
        "torch_version": device_info.torch_version,
        "mps_built": device_info.mps_built,
        "mps_available": device_info.mps_available,
        "cuda_available": device_info.cuda_available,
        "precision": device_info.precision,
        "runtime_modality": config.runtime.modality,
        "debug_oracle_enabled": config.runtime.enable_debug_oracle,
        "resume_path": resolved_resume_path,
        "resume_history": resume_history,
        "initialize_from_path": preserved_initialization_path,
        "source_provenance": initial_source_provenance,
        "initial_source_provenance": initial_source_provenance,
        "latest_source_provenance": dict(source_provenance),
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _configured_device_without_availability_check(preference: str) -> torch.device:
    """Return an unused phase's configured device without requiring hardware."""

    if preference == "auto":
        # ``auto`` never requires a particular unavailable backend.
        return select_device("auto").device
    return torch.device(preference)


def _validate_exact_resume_source(
    payload: Mapping[str, Any],
    current_source: Mapping[str, Any],
) -> None:
    """Reject an exact resume when executable source provenance changed."""

    stored_source = payload.get("git")
    if not isinstance(stored_source, Mapping):
        return
    stored_runtime_fingerprint = stored_source.get("runtime_source_fingerprint")
    current_runtime_fingerprint = current_source.get("runtime_source_fingerprint")
    if isinstance(stored_runtime_fingerprint, str):
        if not isinstance(current_runtime_fingerprint, str):
            raise ValueError(
                "current executable source fingerprint is unavailable; "
                "use --initialize-from when exact source identity cannot be proven"
            )
        if stored_runtime_fingerprint != current_runtime_fingerprint:
            raise ValueError(
                "checkpoint executable source differs from this exact resume; "
                "restore the same runtime source or use --initialize-from"
            )
        # Commit and whole-worktree provenance remain available for audit, but
        # documentation/test-only changes do not alter numerical continuation.
        return
    stored_fingerprint = stored_source.get("worktree_fingerprint")
    current_fingerprint = current_source.get("worktree_fingerprint")
    if isinstance(stored_fingerprint, str) and isinstance(current_fingerprint, str):
        if stored_fingerprint != current_fingerprint:
            raise ValueError(
                "checkpoint source worktree differs from this exact resume; "
                "commit the same source or use --initialize-from"
            )
        return
    stored_commit = stored_source.get("commit")
    current_commit = current_source.get("commit")
    if isinstance(stored_commit, str) and isinstance(current_commit, str):
        if stored_commit != current_commit:
            raise ValueError(
                "checkpoint source commit differs from this exact resume; "
                "use --initialize-from for a source-code transfer"
            )
        if bool(stored_source.get("dirty")) or bool(current_source.get("dirty")):
            raise ValueError(
                "legacy dirty source cannot be proven identical for exact resume; "
                "use --initialize-from"
            )


def _expected_resume_checkpoint_devices(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
    *,
    measurement_device: torch.device,
    closed_loop_device: torch.device,
) -> frozenset[torch.device]:
    """Return valid actual devices for the checkpoint's durable phase state."""

    step = _finite_nonnegative_integer(payload.get("step"), name="step")
    boundary = int(config.training.rgb_pretrain_steps)
    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    # Imported incumbents are validated on the causal device before RGB
    # pretraining starts. Their numbered/selector checkpoints are therefore
    # closed-loop-device artifacts even when their completed step is zero.
    contains_closed_loop_validation = (
        "rollout_selection_metric_version" in metrics
        and "validation_rollout_selection_score" in metrics
    )
    if contains_closed_loop_validation:
        return frozenset({closed_loop_device})
    if step < boundary:
        return frozenset({measurement_device})
    if step > boundary or boundary == 0:
        return frozenset({closed_loop_device})

    marker = metrics.get("measurement_handoff_completed")
    if marker is None:
        # Legacy boundary checkpoints did not distinguish the last measurement
        # save from a completed causal handoff at the same numeric step.
        return frozenset({measurement_device, closed_loop_device})
    try:
        marker_value = float(marker)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint measurement handoff marker is invalid") from error
    if marker_value not in {0.0, 1.0}:
        raise ValueError("checkpoint measurement handoff marker must be 0 or 1")
    return frozenset({closed_loop_device if marker_value == 1.0 else measurement_device})


def _resume_requires_final_validation(
    payload: Mapping[str, Any],
    config: OrpheusConfig,
) -> bool:
    """Return whether a durable terminal iterate still needs validation.

    Checkpoints predating the marker remain inspection-only no-ops when they
    already equal the configured training budget. This preserves their bytes
    and avoids guessing whether historical validation actually ran.
    """

    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    marker = metrics.get("final_validation_completed")
    if marker is None:
        return False
    try:
        marker_value = float(marker)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint final validation marker is invalid") from error
    if marker_value not in {0.0, 1.0}:
        raise ValueError("checkpoint final validation marker must be 0 or 1")
    return marker_value == 0.0 and _finite_nonnegative_integer(
        payload.get("step"),
        name="step",
    ) == int(config.training.steps)


def _resolve_training_devices(
    config: OrpheusConfig,
    *,
    start_step: int,
    initialize_from: bool,
    final_validation_required: bool = False,
    measurement_device_info: DeviceInfo | None = None,
) -> tuple[DeviceInfo, torch.device, torch.device, torch.device]:
    """Resolve only devices required by the remaining training phases.

    A causal-only run configured to use CPU must remain runnable when an unused
    perception preference (for example MPS) is unavailable. Conversely, a
    measurement-only run does not need to initialize an unused causal backend.
    """

    measurement_required = start_step < min(
        config.training.steps,
        config.training.rgb_pretrain_steps,
    ) or (final_validation_required and start_step <= config.training.rgb_pretrain_steps)
    causal_required = (
        initialize_from
        or (max(start_step, config.training.rgb_pretrain_steps) < config.training.steps)
        or (final_validation_required and start_step > config.training.rgb_pretrain_steps)
    )
    measurement_info: DeviceInfo | None = None
    closed_loop_info: DeviceInfo | None = None

    if measurement_required or (causal_required and config.device.closed_loop_preference == "same"):
        measurement_info = measurement_device_info or select_device(config.device.preference)
        measurement_device = measurement_info.device
    else:
        measurement_device = _configured_device_without_availability_check(config.device.preference)

    if config.device.closed_loop_preference == "same":
        closed_loop_device = measurement_device
        closed_loop_info = measurement_info
    elif causal_required:
        closed_loop_info = select_device(config.device.closed_loop_preference)
        closed_loop_device = closed_loop_info.device
    else:
        closed_loop_device = _configured_device_without_availability_check(
            config.device.closed_loop_preference
        )

    if measurement_required:
        if measurement_info is None:
            raise AssertionError("measurement phase did not resolve an execution device")
        return (
            measurement_info,
            measurement_device,
            closed_loop_device,
            measurement_device,
        )
    if causal_required:
        if closed_loop_info is None:
            raise AssertionError("causal phase did not resolve an execution device")
        return (
            closed_loop_info,
            measurement_device,
            closed_loop_device,
            closed_loop_device,
        )

    # An already-complete resume performs no numerical work. Keep checkpoint
    # inspection available even if its historical accelerator is absent.
    inactive_info = select_device("cpu")
    return (
        inactive_info,
        measurement_device,
        closed_loop_device,
        torch.device("cpu"),
    )


def train_from_config(
    config: OrpheusConfig,
    *,
    run_name: str | None = None,
    resume_path: str | Path | None = None,
    initialize_from_path: str | Path | None = None,
    device_info: DeviceInfo | None = None,
) -> dict[str, Any]:
    """Train RGB measurements, then the causal RGB-only online loop.

    This signature is the public contract used by :mod:`train.py`.
    """

    config.validate()
    if resume_path is not None and initialize_from_path is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if config.runtime.modality != "rgb":
        raise ValueError(
            "the primary trainer requires runtime.modality=rgb; "
            "debug_oracle remains an explicit debugging path"
        )
    if config.runtime.enable_debug_oracle:
        raise ValueError("RGB training cannot enable privileged debug_oracle input")
    seed_everything(
        config.project.seed,
        deterministic=config.project.deterministic,
    )
    source_root = Path(__file__).resolve().parents[2]
    source_provenance = capture_git_metadata(source_root)
    run_directory = _resolve_run_directory(
        config,
        run_name=run_name,
        resume_path=resume_path,
    )
    resume_step_hint: int | None = None
    resume_config_payload: Mapping[str, Any] | None = None
    final_validation_recovery_pending = False
    if resume_path is not None:
        # Reject an accidental curriculum/data/objective change before
        # overwriting any metadata in the existing run directory.  The later
        # load restores tensors and RNG; this lightweight CPU read establishes
        # that ``--resume`` is truly an exact continuation first.
        resume_source = Path(resume_path).expanduser().resolve()
        if not resume_source.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_source}")
        resume_config_payload = torch.load(
            resume_source,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(resume_config_payload, Mapping):
            raise ValueError("checkpoint payload must be a mapping")
        validate_checkpoint_config(resume_config_payload, config)
        validate_training_resume_config(resume_config_payload, config)
        _validate_exact_resume_source(
            resume_config_payload,
            source_provenance,
        )
        resume_step_hint = _finite_nonnegative_integer(
            resume_config_payload.get("step"),
            name="step",
        )
        final_validation_recovery_pending = _resume_requires_final_validation(
            resume_config_payload,
            config,
        )
    resolved_device, measurement_device, closed_loop_device, device = _resolve_training_devices(
        config,
        start_step=resume_step_hint or 0,
        initialize_from=initialize_from_path is not None,
        final_validation_required=final_validation_recovery_pending,
        measurement_device_info=device_info,
    )
    if resume_config_payload is not None and resume_step_hint is not None:
        checkpoint_device = resume_config_payload.get("device")
        expected_checkpoint_devices = _expected_resume_checkpoint_devices(
            resume_config_payload,
            config,
            measurement_device=measurement_device,
            closed_loop_device=closed_loop_device,
        )
        if checkpoint_device not in {
            str(expected_device) for expected_device in expected_checkpoint_devices
        }:
            expected_devices = ", ".join(
                sorted(str(expected_device) for expected_device in expected_checkpoint_devices)
            )
            raise ValueError(
                "checkpoint execution device does not match this exact resume "
                f"(checkpoint={checkpoint_device!r}, "
                f"expected one of [{expected_devices}]); use "
                "--initialize-from for a device/backend transfer"
            )
    if run_directory.exists() and resume_path is None:
        occupied = any(
            (run_directory / name).exists()
            for name in ("metrics.jsonl", "checkpoints", "config.resolved.yaml")
        )
        if occupied:
            raise FileExistsError(
                f"run directory already contains training artefacts: {run_directory}"
            )
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    preserved_validation_history: tuple[Path, ...] = ()
    if resume_path is not None and resume_step_hint is not None:
        preserved_validation_history = _preserve_resume_validation_history(
            resume_path,
            checkpoint_directory,
            config,
            resume_step=resume_step_hint,
            expected_device=closed_loop_device,
        )
    best_rollout_path = checkpoint_directory / "best_rollout.pt"
    reference_rollout_path = checkpoint_directory / "reference_rollout.pt"
    best_measurement_path = checkpoint_directory / "best_measurement.pt"
    last_path = checkpoint_directory / "last.pt"
    resolved_config_path = run_directory / "config.resolved.yaml"
    save_resolved_config(config, resolved_config_path)
    run_metadata = _write_run_metadata(
        run_directory / "run_metadata.json",
        config=config,
        device_info=resolved_device,
        active_start_device=device,
        measurement_device=measurement_device,
        closed_loop_device=closed_loop_device,
        resume_path=resume_path,
        initialize_from_path=initialize_from_path,
        source_provenance=source_provenance,
    )

    validation_loader = _make_loader(
        config,
        split="validation",
        episodes=config.training.validation_episodes,
        shuffle=False,
        batch_size_override=1,
    )
    model = OnlineWorldModel.from_config(config, device=device)
    initialized_from_value = run_metadata.get("initialize_from_path")
    initialized_from = (
        str(initialized_from_value) if isinstance(initialized_from_value, str) else None
    )
    if initialize_from_path is not None:
        source = Path(initialize_from_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {source}")
        load_model_weights(
            source,
            model=model,
            allowed_missing_prefixes=(
                ("dynamics.attention_interactions.",)
                if config.model.dynamics.attention_residual_enabled
                else ()
            ),
        )
        initialized_from = str(source)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_step = 0
    best_rollout = math.inf
    best_rollout_selection: _RolloutSelectionMetrics | None = None
    best_rollout_model_state_hash: str | None = None
    best_rollout_step: int | None = None
    reference_rollout_selection: _RolloutSelectionMetrics | None = None
    reference_rollout_model_state_hash: str | None = None
    reference_rollout_step: int | None = None
    best_measurement_selection: _MeasurementSelectionMetrics | None = None
    best_measurement_model_state_hash: str | None = None
    best_measurement_step: int | None = None
    best_rollout_validated = False
    best_measurement_validated = False
    measurement_handoff_completed = config.training.rgb_pretrain_steps == 0
    resumed_from: str | None = None
    resume_metrics: Mapping[str, Any] = {}
    training_data_draw_step = 0
    skipped_no_gradient_batches = 0
    support_collapse_rollback_at_checkpoint = False
    incomplete_reference_comparison_required = False
    if resume_path is not None:
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            # Load storage on CPU and let ``load_state_dict`` place tensors
            # against their owning parameters. This matters for the hybrid
            # MPS-backbone/CPU-detector optimizer: PyTorch deliberately leaves
            # non-capturable Adam scalar step counters on the map device.
            map_location="cpu",
            restore_rng=True,
            expected_config=config,
        )
        start_step = _finite_nonnegative_integer(payload.get("step"), name="step")
        raw_resume_metrics = payload.get("metrics", {})
        if isinstance(raw_resume_metrics, Mapping):
            resume_metrics = raw_resume_metrics
        best_rollout_validated = _binary_checkpoint_marker(
            resume_metrics.get("best_rollout_validated", 0.0),
            name="best_rollout_validated",
        )
        rollout_reference_validated = _binary_checkpoint_marker(
            resume_metrics.get(
                "rollout_reference_validated",
                float(best_rollout_validated),
            ),
            name="rollout_reference_validated",
        )
        incomplete_reference_marker = resume_metrics.get("incomplete_reference_comparison_required")
        if incomplete_reference_marker is not None:
            incomplete_reference_comparison_required = _binary_checkpoint_marker(
                incomplete_reference_marker,
                name="incomplete_reference_comparison_required",
            )
        elif not rollout_reference_validated:
            # Compatibility for a current-protocol unsupported numbered
            # checkpoint written before the durable marker existed.  The
            # explicit zero support result is enough to prove that the next
            # supported validation must establish a fixed reference without
            # promoting itself.
            stored_selection_support = resume_metrics.get("selection_metric_supported")
            if stored_selection_support is not None:
                incomplete_reference_comparison_required = not _binary_checkpoint_marker(
                    stored_selection_support,
                    name="selection_metric_supported",
                )
        if incomplete_reference_comparison_required and (
            rollout_reference_validated or best_rollout_validated
        ):
            raise ValueError(
                "exact resume cannot require an incomplete reference comparison "
                "while declaring a validated rollout reference or incumbent"
            )
        best_measurement_validated = _binary_checkpoint_marker(
            resume_metrics.get("best_measurement_validated", 0.0),
            name="best_measurement_validated",
        )
        preserved_reference: tuple[_RolloutSelectionMetrics, str, int] | None = None
        if rollout_reference_validated:
            preserved_reference = _preserve_resume_selector_checkpoint(
                resume_path,
                reference_rollout_path,
                config,
                prefix="reference_rollout",
                resume_metrics=resume_metrics,
                expected_device=closed_loop_device,
            )
            if preserved_reference is None:
                linked_reference_format = (
                    float(
                        resume_metrics.get(
                            "rollout_selection_metric_version",
                            -1.0,
                        )
                    )
                    == _ROLLOUT_SELECTION_METRIC_VERSION
                    and isinstance(
                        resume_metrics.get("reference_rollout_model_state_hash"),
                        str,
                    )
                    and resume_metrics.get("reference_rollout_checkpoint_step") is not None
                )
                if linked_reference_format or best_rollout_validated:
                    raise ValueError(
                        "exact resume declared a linked rollout reference, but "
                        "reference_rollout.pt is missing or failed "
                        "protocol/tensor verification"
                    )
                rollout_reference_validated = False
            else:
                (
                    reference_rollout_selection,
                    reference_rollout_model_state_hash,
                    reference_rollout_step,
                ) = preserved_reference
        if best_rollout_validated:
            if not rollout_reference_validated or preserved_reference is None:
                raise ValueError(
                    "exact resume declares a best rollout without a verified "
                    "fixed rollout reference"
                )
            preserved_best = _preserve_resume_selector_checkpoint(
                resume_path,
                best_rollout_path,
                config,
                prefix="best_rollout",
                resume_metrics=resume_metrics,
                expected_device=closed_loop_device,
            )
            if preserved_best is None:
                linked_best_format = (
                    float(
                        resume_metrics.get(
                            "rollout_selection_metric_version",
                            -1.0,
                        )
                    )
                    == _ROLLOUT_SELECTION_METRIC_VERSION
                    and isinstance(
                        resume_metrics.get("best_rollout_model_state_hash"),
                        str,
                    )
                    and resume_metrics.get("best_rollout_checkpoint_step") is not None
                )
                if linked_best_format:
                    raise ValueError(
                        "exact resume declared a linked rollout incumbent, but "
                        "best_rollout.pt is missing or failed protocol/tensor "
                        "verification"
                    )
                best_rollout_validated = False
            else:
                (
                    best_rollout_selection,
                    best_rollout_model_state_hash,
                    best_rollout_step,
                ) = preserved_best
                best_rollout = best_rollout_selection.score
        stored_handoff_completed = resume_metrics.get("measurement_handoff_completed")
        if stored_handoff_completed is not None:
            measurement_handoff_completed = _binary_checkpoint_marker(
                stored_handoff_completed,
                name="measurement_handoff_completed",
            )
        elif (
            best_rollout_selection is not None
            and config.training.rgb_pretrain_steps > 0
            and start_step <= config.training.rgb_pretrain_steps
        ):
            # Migration for initialized runs checkpointed before this explicit
            # flag existed. Their retained selector proves an imported
            # reference, and the boundary has not yet been crossed.
            measurement_handoff_completed = False
        if best_measurement_validated:
            restored_measurement = _preserve_resume_measurement_checkpoint(
                resume_path,
                best_measurement_path,
                config,
                resume_metrics=resume_metrics,
                expected_device=measurement_device,
            )
            if restored_measurement is None:
                linked_measurement_format = (
                    float(
                        resume_metrics.get(
                            "measurement_selection_metric_version",
                            -1.0,
                        )
                    )
                    == _MEASUREMENT_SELECTION_METRIC_VERSION
                    and isinstance(
                        resume_metrics.get("best_measurement_model_state_hash"),
                        str,
                    )
                    and resume_metrics.get("best_measurement_checkpoint_step") is not None
                )
                if linked_measurement_format:
                    raise ValueError(
                        "exact resume declared a linked measurement selector "
                        "artifact, but best_measurement.pt is missing or failed "
                        "protocol/tensor verification"
                    )
                # Older checkpoints used MAE alone and did not prove that their
                # proposals crossed the runtime lifecycle gate. Keep them
                # loadable, but require a new broad validation before calling
                # one the best measurement checkpoint.
                best_measurement_validated = False
            else:
                (
                    best_measurement_selection,
                    best_measurement_model_state_hash,
                    best_measurement_step,
                ) = restored_measurement
        resumed_from = str(Path(resume_path).expanduser().resolve())
        training_data_draw_step = _finite_nonnegative_integer(
            resume_metrics.get("training_data_draw_step", start_step),
            name="training_data_draw_step",
        )
        skipped_no_gradient_batches = _finite_nonnegative_integer(
            resume_metrics.get("skipped_no_gradient_batches", 0.0),
            name="skipped_no_gradient_batches",
        )
        expected_data_draw_step = start_step + skipped_no_gradient_batches
        if training_data_draw_step != expected_data_draw_step:
            raise ValueError(
                "checkpoint data-progress invariant failed: "
                "training_data_draw_step must equal optimizer step plus "
                "skipped_no_gradient_batches "
                f"({training_data_draw_step} != {start_step} + "
                f"{skipped_no_gradient_batches})"
            )
        if start_step > config.training.steps:
            raise ValueError(
                "checkpoint step exceeds configured training.steps "
                f"({start_step} > {config.training.steps})"
            )
    else:
        training_data_draw_step = start_step

    logger = MetricsLogger(run_directory / "metrics.jsonl")
    last_metrics: dict[str, float | str] = {}
    started = time.perf_counter()

    def retained_measurement_selector_metrics(
        *,
        checkpoint_model_state_hash: str | None = None,
    ) -> dict[str, Any]:
        data_progress = {
            "training_data_draw_step": float(training_data_draw_step),
            "skipped_no_gradient_batches": float(skipped_no_gradient_batches),
        }
        if best_measurement_selection is None:
            return {
                **data_progress,
                "best_measurement_validated": 0.0,
            }
        if best_measurement_model_state_hash is None or best_measurement_step is None:
            raise AssertionError("measurement selector is missing weight provenance")
        return {
            **data_progress,
            "best_measurement_validated": 1.0,
            **best_measurement_selection.checkpoint_metrics(),
            "measurement_validation_protocol_hash": (_measurement_validation_protocol_hash(config)),
            "best_measurement_model_state_hash": best_measurement_model_state_hash,
            "best_measurement_checkpoint_step": float(best_measurement_step),
            "checkpoint_contains_best_measurement_weights": float(
                checkpoint_model_state_hash == best_measurement_model_state_hash
            ),
        }

    def validate_closed_loop_incumbent(
        *,
        completed_step: int,
        learning_rate: float,
        split: str,
    ) -> tuple[
        TrainingBatchResult,
        bool,
        list[dict[str, float | str]],
        list[dict[str, float | str]],
    ]:
        nonlocal best_rollout, best_rollout_selection, best_rollout_validated
        nonlocal best_rollout_model_state_hash, best_rollout_step
        nonlocal reference_rollout_selection
        nonlocal reference_rollout_model_state_hash, reference_rollout_step
        nonlocal incomplete_reference_comparison_required

        validation = _validation_loader_result(
            model,
            validation_loader,
            config,
            device=device,
            closed_loop=True,
            progress_path=run_directory / "training_progress.json",
            progress_split=split,
        )
        validation_metrics = _result_metrics(
            validation,
            learning_rate=learning_rate,
        )
        selection_support = _validate_validation_support_schema(
            validation.metrics,
            config,
        )
        if selection_support == 0.0:
            if reference_rollout_selection is None:
                incomplete_reference_comparison_required = True
            candidate_model_state_hash = _current_model_state_hash(model)
            support_failures: list[dict[str, float | str]] = [
                {
                    "metric": "physical_selection_support",
                    "direction": "required",
                    "candidate": 0.0,
                    "reference": 1.0,
                    "limit": 1.0,
                    "delta": -1.0,
                }
            ]
            checkpoint_metrics: dict[str, Any] = {
                "validation_total_loss": float(validation.total_loss.detach().cpu()),
                **_validation_support_evidence(validation.metrics),
                "validation_rollout_loss": float(
                    validation.loss_terms.get(
                        "rollout",
                        validation.total_loss,
                    )
                    .detach()
                    .cpu()
                ),
                "validation_rollout_position_loss": float(
                    validation.loss_terms.get(
                        "rollout_position",
                        validation.total_loss,
                    )
                    .detach()
                    .cpu()
                ),
                "selection_metric_supported": 0.0,
                "selection_accepted": 0.0,
                "selection_rejection_reason_count": 1.0,
                "selection_rejection_reasons": support_failures,
                "selection_reference_guardrail_failures": [],
                "selection_incumbent_guardrail_failures": [],
                "selection_training_support_failures": support_failures,
                "selection_training_support_required": float(
                    config.training.steps > config.training.rgb_pretrain_steps
                ),
                "selection_training_support_passed": 0.0,
                "selection_mutable_training_support_failures": support_failures,
                "selection_mutable_training_support_passed": 0.0,
                "rollout_selection_metric_version": (_ROLLOUT_SELECTION_METRIC_VERSION),
                "best_rollout_validated": float(best_rollout_selection is not None),
                "rollout_reference_validated": float(reference_rollout_selection is not None),
                "incomplete_reference_comparison_required": float(
                    incomplete_reference_comparison_required
                ),
                "best_measurement_validated": float(best_measurement_selection is not None),
                **_validation_protocol_checkpoint_metrics(config),
                "checkpoint_model_state_hash": candidate_model_state_hash,
                "checkpoint_contains_best_rollout_weights": float(
                    best_rollout_model_state_hash == candidate_model_state_hash
                ),
                "checkpoint_contains_reference_rollout_weights": float(
                    reference_rollout_model_state_hash == candidate_model_state_hash
                ),
                "measurement_handoff_completed": float(measurement_handoff_completed),
            }
            if best_rollout_selection is not None:
                if best_rollout_model_state_hash is None or best_rollout_step is None:
                    raise AssertionError("retained incumbent is missing weight provenance")
                checkpoint_metrics.update(best_rollout_selection.checkpoint_metrics())
                checkpoint_metrics["best_rollout_model_state_hash"] = best_rollout_model_state_hash
                checkpoint_metrics["best_rollout_checkpoint_step"] = float(best_rollout_step)
            if reference_rollout_selection is not None:
                if reference_rollout_model_state_hash is None or reference_rollout_step is None:
                    raise AssertionError("fixed reference is missing weight provenance")
                checkpoint_metrics.update(
                    reference_rollout_selection.checkpoint_metrics(prefix="reference_rollout")
                )
                checkpoint_metrics["reference_rollout_model_state_hash"] = (
                    reference_rollout_model_state_hash
                )
                checkpoint_metrics["reference_rollout_checkpoint_step"] = float(
                    reference_rollout_step
                )
            checkpoint_metrics.update(
                retained_measurement_selector_metrics(
                    checkpoint_model_state_hash=candidate_model_state_hash,
                )
            )
            validation_metrics.update(
                {
                    "selection_metric_supported": 0.0,
                    "selection_accepted": 0.0,
                    "selection_rejection_reason_count": 1.0,
                    "selection_rejection_reasons_json": json.dumps(
                        support_failures,
                        sort_keys=True,
                    ),
                    "selection_reference_guardrail_failures_json": "[]",
                    "selection_incumbent_guardrail_failures_json": "[]",
                    "selection_training_support_failure_count": 1.0,
                    "selection_training_support_required": float(
                        config.training.steps > config.training.rgb_pretrain_steps
                    ),
                    "selection_training_support_failures_json": json.dumps(
                        support_failures,
                        sort_keys=True,
                    ),
                    "selection_mutable_training_support_failure_count": 1.0,
                    "selection_mutable_training_support_failures_json": json.dumps(
                        support_failures,
                        sort_keys=True,
                    ),
                }
            )
            if best_rollout_selection is not None and best_rollout_step is not None:
                validation_metrics["best_rollout_selection_score"] = best_rollout_selection.score
                validation_metrics["best_rollout_checkpoint_step"] = float(best_rollout_step)
            if reference_rollout_step is not None:
                validation_metrics["reference_rollout_checkpoint_step"] = float(
                    reference_rollout_step
                )
            logger.log(
                step=completed_step,
                split=split,
                metrics=validation_metrics,
            )
            save_checkpoint(
                checkpoint_directory / f"validation_step_{completed_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
                source_provenance=source_provenance,
            )
            if reference_rollout_selection is None and not reference_rollout_path.is_file():
                unsupported_reference_metrics = {
                    **checkpoint_metrics,
                    "checkpoint_contains_reference_rollout_weights": 0.0,
                    "reference_rollout_artifact_model_state_hash": (candidate_model_state_hash),
                    "reference_rollout_artifact_checkpoint_step": float(completed_step),
                }
                save_checkpoint(
                    reference_rollout_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=completed_step,
                    metrics=unsupported_reference_metrics,
                    device=str(device),
                    source_provenance=source_provenance,
                )
            return validation, False, support_failures, support_failures
        candidate = _rollout_selection_metrics(
            validation.metrics,
            config,
            require_scenarios=True,
        )
        candidate_model_state_hash = _current_model_state_hash(model)
        deferred_reference_comparison_required = incomplete_reference_comparison_required or (
            reference_rollout_selection is None and reference_rollout_path.is_file()
        )
        established_reference = reference_rollout_selection is None
        if established_reference:
            reference_rollout_selection = candidate
            reference_rollout_model_state_hash = candidate_model_state_hash
            reference_rollout_step = completed_step
            incomplete_reference_comparison_required = False
        if (
            reference_rollout_selection is None
            or reference_rollout_model_state_hash is None
            or reference_rollout_step is None
        ):
            raise AssertionError("closed-loop validation did not establish a fixed reference")
        training_support_required = config.training.steps > config.training.rgb_pretrain_steps
        training_support_failures = (
            _handoff_training_support_failures(
                candidate,
                reference_rollout_selection,
                config,
            )
            if training_support_required
            else []
        )
        mutable_training_support_failures = (
            _mutable_causal_training_support_failures(candidate, config)
            if training_support_required
            else []
        )
        reference_guardrail_failures = (
            []
            if established_reference
            else _rollout_selection_guardrail_failures(
                candidate,
                reference_rollout_selection,
            )
        )
        if established_reference and deferred_reference_comparison_required:
            reference_guardrail_failures = [
                {
                    "metric": "complete_fixed_reference_comparison",
                    "direction": "required_before_promotion",
                    "candidate": 0.0,
                    "reference": 0.0,
                    "limit": 1.0,
                    "delta": 0.0,
                }
            ]
        first_incumbent_rejection_reasons = [
            *training_support_failures,
            *reference_guardrail_failures,
        ]
        if best_rollout_selection is None and first_incumbent_rejection_reasons:
            checkpoint_metrics: dict[str, Any] = {
                "validation_total_loss": float(validation.total_loss.detach().cpu()),
                **_validation_support_evidence(validation.metrics),
                "validation_rollout_loss": float(
                    validation.loss_terms.get("rollout", validation.total_loss).detach().cpu()
                ),
                "validation_rollout_position_loss": float(
                    validation.loss_terms.get(
                        "rollout_position",
                        validation.total_loss,
                    )
                    .detach()
                    .cpu()
                ),
                "selection_accepted": 0.0,
                "selection_rejection_reason_count": float(len(first_incumbent_rejection_reasons)),
                "selection_rejection_reasons": first_incumbent_rejection_reasons,
                "selection_reference_guardrail_failures": reference_guardrail_failures,
                "selection_incumbent_guardrail_failures": [],
                "selection_training_support_failures": training_support_failures,
                "selection_training_support_required": 1.0,
                "selection_training_support_passed": float(not training_support_failures),
                "selection_mutable_training_support_failures": (mutable_training_support_failures),
                "selection_mutable_training_support_passed": float(
                    not mutable_training_support_failures
                ),
                "rollout_selection_metric_version": (_ROLLOUT_SELECTION_METRIC_VERSION),
                "best_rollout_validated": 0.0,
                "rollout_reference_validated": 1.0,
                "incomplete_reference_comparison_required": 0.0,
                "best_measurement_validated": float(best_measurement_selection is not None),
                **candidate.validation_metrics(),
                **reference_rollout_selection.checkpoint_metrics(prefix="reference_rollout"),
                **_validation_protocol_checkpoint_metrics(config),
                "checkpoint_model_state_hash": candidate_model_state_hash,
                "checkpoint_contains_best_rollout_weights": 0.0,
                "checkpoint_contains_reference_rollout_weights": float(
                    candidate_model_state_hash == reference_rollout_model_state_hash
                ),
                "reference_rollout_model_state_hash": (reference_rollout_model_state_hash),
                "reference_rollout_checkpoint_step": float(reference_rollout_step),
                "measurement_handoff_completed": float(measurement_handoff_completed),
            }
            checkpoint_metrics.update(
                retained_measurement_selector_metrics(
                    checkpoint_model_state_hash=candidate_model_state_hash,
                )
            )
            validation_metrics.update(
                {
                    "selection_accepted": 0.0,
                    "selection_rejection_reason_count": float(
                        len(first_incumbent_rejection_reasons)
                    ),
                    "selection_rejection_reasons_json": json.dumps(
                        first_incumbent_rejection_reasons,
                        sort_keys=True,
                    ),
                    "selection_reference_guardrail_failures_json": json.dumps(
                        reference_guardrail_failures,
                        sort_keys=True,
                    ),
                    "selection_training_support_failure_count": float(
                        len(training_support_failures)
                    ),
                    "selection_training_support_required": 1.0,
                    "selection_training_support_failures_json": json.dumps(
                        training_support_failures,
                        sort_keys=True,
                    ),
                    "selection_mutable_training_support_failure_count": float(
                        len(mutable_training_support_failures)
                    ),
                    "selection_mutable_training_support_failures_json": json.dumps(
                        mutable_training_support_failures,
                        sort_keys=True,
                    ),
                }
            )
            logger.log(
                step=completed_step,
                split=split,
                metrics=validation_metrics,
            )
            save_checkpoint(
                checkpoint_directory / f"validation_step_{completed_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
                source_provenance=source_provenance,
            )
            if established_reference:
                save_checkpoint(
                    reference_rollout_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=completed_step,
                    metrics=checkpoint_metrics,
                    device=str(device),
                    source_provenance=source_provenance,
                )
            return (
                validation,
                False,
                training_support_failures,
                mutable_training_support_failures,
            )
        accepted = not training_support_failures and (
            (best_rollout_selection is None and not reference_guardrail_failures)
            or (
                best_rollout_selection is not None
                and _rollout_selection_improves(
                    candidate,
                    best_rollout_selection,
                )
                and _rollout_selection_passes_guardrails(
                    candidate,
                    reference_rollout_selection,
                )
            )
        )
        if accepted:
            best_rollout_selection = candidate
            best_rollout = candidate.score
            best_rollout_validated = True
            best_rollout_model_state_hash = candidate_model_state_hash
            best_rollout_step = completed_step
        if best_rollout_selection is None:
            raise AssertionError("closed-loop validation did not establish an incumbent")
        if best_rollout_model_state_hash is None or best_rollout_step is None:
            raise AssertionError("closed-loop incumbent is missing weight provenance")
        checkpoint_metrics = _rollout_validation_checkpoint_metrics(
            validation,
            candidate,
            best_rollout_selection,
            reference_rollout_selection,
            config=config,
            accepted=accepted,
            training_support_required=training_support_required,
            training_support_failures=training_support_failures,
            mutable_training_support_failures=mutable_training_support_failures,
            best_measurement=(best_measurement_selection if best_measurement_validated else None),
            checkpoint_model_state_hash=candidate_model_state_hash,
            incumbent_model_state_hash=best_rollout_model_state_hash,
            incumbent_step=best_rollout_step,
            reference_model_state_hash=reference_rollout_model_state_hash,
            reference_step=reference_rollout_step,
        )
        checkpoint_metrics["measurement_handoff_completed"] = float(measurement_handoff_completed)
        checkpoint_metrics.update(
            retained_measurement_selector_metrics(
                checkpoint_model_state_hash=candidate_model_state_hash,
            )
        )
        checkpoint_metrics.update(
            {
                f"validation_{name}": float(value)
                for name, value in validation.metrics.items()
                if name.startswith("rollout_position@")
            }
        )
        # Put the selector decision in the human-facing JSONL as well as the
        # tensor-verified numbered checkpoint.  Previously the row was written
        # before selection, so an active campaign exposed its validation curve
        # but not whether or why each point was rejected.
        validation_metrics.update(
            {
                "validation_rollout_selection_score": candidate.score,
                "selection_accepted": float(accepted),
                "selection_rejection_reason_count": checkpoint_metrics[
                    "selection_rejection_reason_count"
                ],
                "selection_rejection_reasons_json": json.dumps(
                    checkpoint_metrics["selection_rejection_reasons"],
                    sort_keys=True,
                ),
                "selection_reference_guardrail_failures_json": json.dumps(
                    checkpoint_metrics["selection_reference_guardrail_failures"],
                    sort_keys=True,
                ),
                "selection_incumbent_guardrail_failures_json": json.dumps(
                    checkpoint_metrics["selection_incumbent_guardrail_failures"],
                    sort_keys=True,
                ),
                "selection_training_support_failure_count": float(len(training_support_failures)),
                "selection_training_support_required": float(training_support_required),
                "selection_training_support_failures_json": json.dumps(
                    training_support_failures,
                    sort_keys=True,
                ),
                "selection_mutable_training_support_failure_count": float(
                    len(mutable_training_support_failures)
                ),
                "selection_mutable_training_support_failures_json": json.dumps(
                    mutable_training_support_failures,
                    sort_keys=True,
                ),
                "best_rollout_selection_score": best_rollout_selection.score,
                "best_rollout_checkpoint_step": float(best_rollout_step),
                "reference_rollout_checkpoint_step": float(reference_rollout_step),
            }
        )
        logger.log(
            step=completed_step,
            split=split,
            metrics=validation_metrics,
        )
        save_checkpoint(
            checkpoint_directory / f"validation_step_{completed_step:06d}.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            step=completed_step,
            metrics=checkpoint_metrics,
            device=str(device),
            source_provenance=source_provenance,
        )
        if established_reference:
            save_checkpoint(
                reference_rollout_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
                source_provenance=source_provenance,
            )
        if accepted:
            save_checkpoint(
                best_rollout_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
                source_provenance=source_provenance,
            )
        return (
            validation,
            accepted,
            training_support_failures,
            mutable_training_support_failures,
        )

    def retained_selector_metrics() -> dict[str, Any]:
        current_model_state_hash = _current_model_state_hash(model)
        metrics: dict[str, Any] = {
            "best_rollout_validated": float(best_rollout_selection is not None),
            "rollout_reference_validated": float(reference_rollout_selection is not None),
            "incomplete_reference_comparison_required": float(
                incomplete_reference_comparison_required
            ),
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            "checkpoint_model_state_hash": current_model_state_hash,
            "checkpoint_contains_best_rollout_weights": float(
                best_rollout_model_state_hash is not None
                and current_model_state_hash == best_rollout_model_state_hash
            ),
            "checkpoint_contains_reference_rollout_weights": float(
                reference_rollout_model_state_hash is not None
                and current_model_state_hash == reference_rollout_model_state_hash
            ),
            "support_collapse_rollback_applied_at_checkpoint": float(
                support_collapse_rollback_at_checkpoint
            ),
            "checkpoint_state_role": (
                "restored_best_rollout"
                if support_collapse_rollback_at_checkpoint
                else "mutable_training_iterate"
            ),
            "measurement_handoff_completed": float(measurement_handoff_completed),
            **_validation_protocol_checkpoint_metrics(config),
            **retained_measurement_selector_metrics(
                checkpoint_model_state_hash=current_model_state_hash,
            ),
        }
        if best_rollout_selection is not None:
            if best_rollout_model_state_hash is None or best_rollout_step is None:
                raise AssertionError("retained incumbent is missing weight provenance")
            metrics.update(best_rollout_selection.checkpoint_metrics())
            metrics["best_rollout_model_state_hash"] = best_rollout_model_state_hash
            metrics["best_rollout_checkpoint_step"] = float(best_rollout_step)
        if reference_rollout_selection is not None:
            if reference_rollout_model_state_hash is None or reference_rollout_step is None:
                raise AssertionError("fixed reference is missing weight provenance")
            metrics.update(
                reference_rollout_selection.checkpoint_metrics(prefix="reference_rollout")
            )
            metrics["reference_rollout_model_state_hash"] = reference_rollout_model_state_hash
            metrics["reference_rollout_checkpoint_step"] = float(reference_rollout_step)
        return metrics

    def validate_measurement_incumbent(
        *,
        completed_step: int,
        learning_rate: float,
        split: str,
    ) -> tuple[TrainingBatchResult, bool]:
        nonlocal best_measurement_selection, best_measurement_validated
        nonlocal best_measurement_model_state_hash, best_measurement_step

        validation = _validation_loader_result(
            model,
            validation_loader,
            config,
            device=device,
            closed_loop=False,
            progress_path=run_directory / "training_progress.json",
            progress_split=split,
        )
        validation_metrics = _result_metrics(
            validation,
            learning_rate=learning_rate,
        )
        if validation.phase != "rgb_pretrain" or "measurement" not in validation.loss_terms:
            raise RuntimeError("measurement validation returned an unexpected phase")
        measurement_candidate = _measurement_selection_metrics(validation.metrics)
        measurement_rejection_reasons: list[dict[str, float | str]] = []
        if measurement_candidate is None:
            measurement_accepted = False
            measurement_rejection_reasons.append(
                {
                    "metric": "runtime_usable_localization",
                    "direction": "required",
                    "candidate": 0.0,
                    "reference": 1.0,
                    "limit": 1.0,
                    "delta": -1.0,
                }
            )
        elif best_measurement_selection is None:
            measurement_accepted = True
        else:
            measurement_rejection_reasons.extend(
                _measurement_selection_guardrail_failures(
                    measurement_candidate,
                    best_measurement_selection,
                )
            )
            if (
                measurement_candidate.score
                >= best_measurement_selection.score - _MEASUREMENT_SELECTION_MIN_DELTA
            ):
                measurement_rejection_reasons.append(
                    {
                        "metric": "measurement_selection_score",
                        "direction": "minimum_improvement",
                        "candidate": measurement_candidate.score,
                        "reference": best_measurement_selection.score,
                        "limit": (
                            best_measurement_selection.score - _MEASUREMENT_SELECTION_MIN_DELTA
                        ),
                        "delta": (measurement_candidate.score - best_measurement_selection.score),
                    }
                )
            measurement_accepted = _measurement_selection_improves(
                measurement_candidate,
                best_measurement_selection,
            )
        if measurement_accepted:
            if measurement_candidate is None:
                raise AssertionError("an unusable measurement candidate was accepted")
            best_measurement_selection = measurement_candidate
            best_measurement_model_state_hash = _current_model_state_hash(model)
            best_measurement_step = completed_step
            best_measurement_validated = True
        validation_metrics.update(
            {
                "measurement_selection_usable": float(measurement_candidate is not None),
                "measurement_selection_accepted": float(measurement_accepted),
                "measurement_selection_rejection_reason_count": float(
                    len(measurement_rejection_reasons)
                ),
                "measurement_selection_rejection_reasons_json": json.dumps(
                    measurement_rejection_reasons,
                    sort_keys=True,
                ),
            }
        )
        if measurement_candidate is not None:
            validation_metrics.update(measurement_candidate.validation_metrics())
        if best_measurement_selection is not None:
            validation_metrics["best_measurement_selection_score"] = (
                best_measurement_selection.score
            )
        logger.log(
            step=completed_step,
            split=split,
            metrics=validation_metrics,
        )
        if measurement_accepted:
            if (
                best_measurement_selection is None
                or best_measurement_model_state_hash is None
                or best_measurement_step is None
            ):
                raise AssertionError("accepted measurement selector is missing provenance")
            save_checkpoint(
                best_measurement_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics={
                    "validation_total_loss": float(validation.total_loss.detach().cpu()),
                    "validation_measurement_loss": float(
                        validation.loss_terms["measurement"].detach().cpu()
                    ),
                    **best_measurement_selection.validation_metrics(),
                    **retained_selector_metrics(),
                    "checkpoint_model_state_hash": best_measurement_model_state_hash,
                    "checkpoint_contains_best_measurement_weights": 1.0,
                },
                device=str(device),
                source_provenance=source_provenance,
            )
        return validation, measurement_accepted

    def restore_safe_causal_incumbent(
        *,
        completed_step: int,
        split: str,
        support_failures: list[dict[str, float | str]],
    ) -> None:
        """Rollback a support-collapsed iterate and discard its Adam moments."""

        nonlocal support_collapse_rollback_at_checkpoint
        if not support_failures:
            raise ValueError("causal rollback requires an explicit support failure")
        if best_rollout_selection is None or not best_rollout_path.is_file():
            raise RuntimeError(
                "closed-loop validation collapsed training support, but no "
                "tensor-verified broad incumbent is available for rollback"
            )
        load_model_weights(
            best_rollout_path,
            model=model,
            expected_config=config,
        )
        model.reset()
        _fresh_causal_optimizer_state(
            optimizer,
            learning_rate=(
                config.training.learning_rate * config.training.closed_loop_learning_rate_scale
            ),
            weight_decay=config.training.weight_decay,
        )
        support_collapse_rollback_at_checkpoint = True
        logger.log(
            step=completed_step,
            split=split,
            metrics={
                "support_collapse_rollback_applied": 1.0,
                "support_collapse_failure_count": float(len(support_failures)),
                "support_collapse_failures_json": json.dumps(
                    support_failures,
                    sort_keys=True,
                ),
                "restored_best_rollout_checkpoint_step": float(
                    best_rollout_step if best_rollout_step is not None else -1
                ),
                "optimizer_state_reset": 1.0,
                "training_data_draw_step": float(training_data_draw_step),
            },
        )

    imported_incumbent = initialize_from_path is not None
    if imported_incumbent:
        validation_source_device = device
        causal_phase_planned = config.training.steps > config.training.rgb_pretrain_steps
        if causal_phase_planned:
            if device != closed_loop_device:
                previous_device = device
                model.to(closed_loop_device)
                model.reset()
                device = closed_loop_device
                _release_accelerator_cache(previous_device)
            _, accepted, _, _ = validate_closed_loop_incumbent(
                completed_step=start_step,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                split="validation_initialization_incumbent",
            )
            if accepted:
                print(
                    "preserved imported runtime as broad closed-loop incumbent "
                    f"(score={best_rollout:.6f})",
                    flush=True,
                )
            else:
                # ``validate_closed_loop_incumbent`` deliberately persists a
                # truthful unsupported reference and leaves the mutable
                # training path able to restore support.  Aborting here made
                # that recovery path unreachable and turned one unsupported
                # scenario slice into an initialization crash.
                logger.log(
                    step=start_step,
                    split="training_control_initialization_support",
                    metrics={
                        "initialization_candidate_accepted": 0.0,
                        "initialization_reference_established": float(
                            reference_rollout_selection is not None
                        ),
                        "initialization_training_continues": 1.0,
                        "training_data_draw_step": float(training_data_draw_step),
                    },
                )
                print(
                    "imported runtime did not satisfy every causal promotion "
                    "guard; preserved its diagnostic reference and continued "
                    "training without promoting a deployment incumbent",
                    flush=True,
                )
        if start_step < config.training.rgb_pretrain_steps and device != validation_source_device:
            previous_device = device
            model.to(validation_source_device)
            model.reset()
            device = validation_source_device
            _release_accelerator_cache(previous_device)
        if start_step < config.training.rgb_pretrain_steps:
            _, measurement_baseline_accepted = validate_measurement_incumbent(
                completed_step=start_step,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                split="validation_initialization_measurement_incumbent",
            )
            if measurement_baseline_accepted:
                print(
                    "preserved imported runtime as the paired global/fast RGB "
                    "measurement incumbent",
                    flush=True,
                )
    elif start_step == 0 and config.training.rgb_pretrain_steps == 0 and config.training.steps > 0:
        # A fresh causal-only run still needs one immutable pre-update
        # reference. Later retries happen only at ``eval_every`` rather than
        # before every optimizer update.
        _, accepted, _, _ = validate_closed_loop_incumbent(
            completed_step=0,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            split="validation_initialization_incumbent",
        )
        if accepted:
            print(
                f"established initial causal reference/incumbent (score={best_rollout:.6f})",
                flush=True,
            )
        else:
            print(
                "initial causal reference did not satisfy every promotion "
                "guard; training continues without a deployment incumbent",
                flush=True,
            )
    measurement_handoff_pending = (
        not measurement_handoff_completed
        and start_step <= config.training.rgb_pretrain_steps < config.training.steps
    )
    # Do not spawn/prefetch training workers while imported or resumed weights
    # are still undergoing the expensive atomic initialization validations.
    # Apart from wasting memory, those unused workers obscured whether a long
    # step-zero validation was making progress.
    remaining_optimizer_updates = config.training.steps - start_step
    maximum_remaining_batch_draws = remaining_optimizer_updates * (
        config.training.maximum_no_gradient_batches_per_update + 1
    )
    train_loader = _make_loader(
        config,
        split="train",
        episodes=config.training.train_episodes,
        shuffle=not config.training.fixed_dataset,
        start_step=training_data_draw_step,
        stop_step=training_data_draw_step + maximum_remaining_batch_draws,
    )
    train_iterator: Iterator[dict[str, Any]] | None = None
    train_batches_per_epoch = math.ceil(
        config.training.train_episodes
        / min(config.training.batch_size, config.training.train_episodes)
    )

    for step in range(start_step, config.training.steps):
        restored_measurement_candidate = False
        if (
            step == config.training.rgb_pretrain_steps
            and best_measurement_validated
            and best_measurement_path.is_file()
        ):
            # Enter the downstream stage from the best calibrated perception
            # state, not merely the final pretraining iterate.  Heteroscedastic
            # objectives can improve NLL while physical localization regresses.
            load_model_weights(
                best_measurement_path,
                model=model,
                expected_config=config,
            )
            restored_measurement_candidate = True
            if best_measurement_selection is None:
                raise AssertionError("validated measurement checkpoint has no selector metrics")
            print(
                "restored best runtime-usable RGB checkpoint for closed-loop handoff "
                f"(score={best_measurement_selection.score:.6f}, "
                f"world_mae={best_measurement_selection.world_position_mae_m:.6f}m, "
                f"recall={best_measurement_selection.runtime_birth_recall:.4f}, "
                f"precision={best_measurement_selection.runtime_birth_precision:.4f}, "
                f"fast_roi_coverage="
                f"{best_measurement_selection.fast_roi_target_coverage:.4f}, "
                f"fast_roi_mae="
                f"{best_measurement_selection.fast_roi_world_position_mae_m:.6f}m)",
                flush=True,
            )
        if step == config.training.rgb_pretrain_steps:
            _fresh_causal_optimizer_state(
                optimizer,
                learning_rate=(
                    config.training.learning_rate * config.training.closed_loop_learning_rate_scale
                ),
                weight_decay=config.training.weight_decay,
            )
            if device != closed_loop_device:
                # The phase boundary already resets AdamW moments. Move the
                # persistent model exactly once, clear transient runtime
                # caches, and keep every selector validation on the causal
                # device recorded in the protocol hash.
                previous_device = device
                model.to(closed_loop_device)
                model.reset()
                device = closed_loop_device
                _release_accelerator_cache(previous_device)
        if step == config.training.rgb_pretrain_steps and measurement_handoff_pending:
            # Keep the marker pending through validation and any rollback.
            # A boundary checkpoint must never claim the handoff completed
            # before the mutable causal source has actually been selected.
            handoff_support_failures: list[dict[str, float | str]] = []
            handoff_mutable_support_failures: list[dict[str, float | str]] = []
            handoff_mutable_source = "measurement_candidate"
            (
                _,
                accepted,
                handoff_support_failures,
                handoff_mutable_support_failures,
            ) = validate_closed_loop_incumbent(
                completed_step=step,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                split="validation_measurement_handoff",
            )
            if best_rollout_selection is None:
                # The fresh run establishes its first broad incumbent below.
                pass
            elif accepted:
                logger.log(
                    step=step,
                    split="training_control_measurement_handoff",
                    metrics={
                        "measurement_handoff_candidate_accepted": 1.0,
                        "measurement_handoff_candidate_training_viable": 1.0,
                        "measurement_handoff_mutable_source": handoff_mutable_source,
                        "measurement_handoff_support_failure_count": 0.0,
                        "measurement_handoff_support_failures_json": "[]",
                        "training_data_draw_step": float(training_data_draw_step),
                    },
                )
                print(
                    "promoted measurement handoff after broad closed-loop validation "
                    f"(score={best_rollout:.6f})",
                    flush=True,
                )
            elif not restored_measurement_candidate:
                if not best_rollout_path.is_file():
                    raise RuntimeError(
                        "no validated measurement handoff candidate is available "
                        "and the retained broad incumbent checkpoint is missing"
                    )
                load_model_weights(
                    best_rollout_path,
                    model=model,
                    expected_config=config,
                )
                model.reset()
                _fresh_causal_optimizer_state(
                    optimizer,
                    learning_rate=(
                        config.training.learning_rate
                        * config.training.closed_loop_learning_rate_scale
                    ),
                    weight_decay=config.training.weight_decay,
                )
                handoff_mutable_source = "retained_broad_incumbent"
                logger.log(
                    step=step,
                    split="training_control_measurement_handoff",
                    metrics={
                        "measurement_handoff_candidate_accepted": 0.0,
                        "measurement_handoff_candidate_training_viable": 0.0,
                        "measurement_handoff_mutable_source": handoff_mutable_source,
                        "measurement_handoff_support_failure_count": 1.0,
                        "measurement_handoff_support_failures_json": json.dumps(
                            [
                                {
                                    "metric": "validated_measurement_candidate",
                                    "direction": "required",
                                    "candidate": 0.0,
                                    "reference": 1.0,
                                    "limit": 1.0,
                                    "delta": -1.0,
                                }
                            ],
                            sort_keys=True,
                        ),
                        "training_data_draw_step": float(training_data_draw_step),
                    },
                )
                print(
                    "no pair-validated RGB measurement candidate was available; "
                    "restored the retained broad incumbent for causal training",
                    flush=True,
                )
            elif handoff_mutable_support_failures:
                if not best_rollout_path.is_file():
                    raise RuntimeError(
                        "measurement handoff collapsed causal training support, "
                        "but the retained broad incumbent checkpoint is missing"
                    )
                load_model_weights(
                    best_rollout_path,
                    model=model,
                    expected_config=config,
                )
                model.reset()
                _fresh_causal_optimizer_state(
                    optimizer,
                    learning_rate=(
                        config.training.learning_rate
                        * config.training.closed_loop_learning_rate_scale
                    ),
                    weight_decay=config.training.weight_decay,
                )
                handoff_mutable_source = "retained_broad_incumbent"
                logger.log(
                    step=step,
                    split="training_control_measurement_handoff",
                    metrics={
                        "measurement_handoff_candidate_accepted": 0.0,
                        "measurement_handoff_candidate_training_viable": 0.0,
                        "measurement_handoff_mutable_source": handoff_mutable_source,
                        "measurement_handoff_support_failure_count": float(
                            len(handoff_mutable_support_failures)
                        ),
                        "measurement_handoff_support_failures_json": json.dumps(
                            handoff_mutable_support_failures,
                            sort_keys=True,
                        ),
                        "training_data_draw_step": float(training_data_draw_step),
                    },
                )
                print(
                    "rejected measurement candidate as a causal starting point "
                    "because persistent tracking/forecast coverage collapsed; "
                    "restored the retained broad incumbent for trainable support",
                    flush=True,
                )
            else:
                logger.log(
                    step=step,
                    split="training_control_measurement_handoff",
                    metrics={
                        "measurement_handoff_candidate_accepted": 0.0,
                        "measurement_handoff_candidate_training_viable": 1.0,
                        "measurement_handoff_mutable_source": handoff_mutable_source,
                        "measurement_handoff_support_failure_count": float(
                            len(handoff_support_failures)
                        ),
                        "measurement_handoff_support_failures_json": json.dumps(
                            handoff_support_failures,
                            sort_keys=True,
                        ),
                        "training_data_draw_step": float(training_data_draw_step),
                    },
                )
                print(
                    "retained the imported runtime as the safe deployment incumbent; "
                    "causal optimisation continues from the measurement candidate "
                    "because its tracking support remains trainable",
                    flush=True,
                )
            measurement_handoff_completed = True
            measurement_handoff_pending = False
        # If no deployable incumbent exists yet, retry only at the declared
        # validation cadence below. Running the complete validation manifest
        # before every causal optimizer update can make an unsupported model
        # appear hung and consume orders of magnitude more validation than
        # training.
        global_perception_trainable = (
            step
            < config.training.rgb_pretrain_steps
            + config.training.closed_loop_global_trainable_steps
        )
        active_closed_loop_scope = config.training.closed_loop_trainable_scope
        closed_loop_scope_transitioned = False
        if step < config.training.rgb_pretrain_steps:
            model.requires_grad_(True)
        else:
            (
                active_closed_loop_scope,
                closed_loop_scope_transitioned,
            ) = _closed_loop_trainable_scope_for_step(
                config,
                completed_step=step,
            )
            set_closed_loop_trainable_scope(
                model,
                scope=active_closed_loop_scope,
            )
        set_global_perception_trainable(
            model,
            trainable=global_perception_trainable,
        )
        no_gradient_attempts = 0
        post_step_finite_check_seconds = 0.0
        while True:
            if train_iterator is None:
                train_iterator = iter(train_loader)
            raw_batch, train_iterator = _next_batch(train_loader, train_iterator)
            training_data_draw_step += 1
            _check_batch_major(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            if step < config.training.rgb_pretrain_steps:
                target_learning_rate = config.training.learning_rate
                frame_index = measurement_pretrain_frame_index(
                    step,
                    loader_batches=train_batches_per_epoch,
                    total_frames=int(batch["rgb"].shape[1]),
                    fixed_dataset=config.training.fixed_dataset,
                )
                result = pretrain_rgb_measurements(
                    model,
                    batch,
                    config,
                    frame_index=frame_index,
                )
            else:
                target_learning_rate = (
                    config.training.learning_rate * config.training.closed_loop_learning_rate_scale
                )
                total_frames = int(batch["rgb"].shape[1])
                window_steps = min(total_frames, config.training.tbptt_steps)
                maximum_rollout_frame_offset = max(
                    max(
                        1,
                        int(round(horizon * config.simulator.frame_rate)),
                    )
                    for horizon in config.evaluation.horizons_seconds
                )
                window_start = select_closed_loop_window(
                    batch,
                    window_steps,
                    event_condition_probability=(config.training.collision_window_probability),
                    maximum_rollout_frame_offset=maximum_rollout_frame_offset,
                    minimum_rollout_frame_offset=min(
                        max(
                            1,
                            int(round(horizon * config.simulator.frame_rate)),
                        )
                        for horizon in config.evaluation.horizons_seconds
                    ),
                    long_horizon_probability=(config.training.long_horizon_window_probability),
                    joint_collision_long_horizon_sampling=(
                        config.training.joint_collision_long_horizon_sampling
                    ),
                )
                result = run_closed_loop_batch(
                    model,
                    batch,
                    config,
                    window_start=window_start,
                    window_steps=window_steps,
                    apply_perturbations=True,
                    include_measurement_supervision=True,
                    rollout_anchors_per_window=(config.training.rollout_anchors_per_window),
                )
                result.metrics["closed_loop_scope_transitioned"] = float(
                    closed_loop_scope_transitioned
                )
                result.metrics["closed_loop_scope_fast_roi_only"] = float(
                    active_closed_loop_scope == "fast_roi"
                )
                result.metrics["closed_loop_scope_attention_only"] = float(
                    active_closed_loop_scope == "attention"
                )
                result.metrics["closed_loop_scope_state_dynamics_only"] = float(
                    active_closed_loop_scope == "state_dynamics"
                )
                result.metrics["closed_loop_scope_updater_only"] = float(
                    active_closed_loop_scope == "updater"
                )
                result.metrics["closed_loop_scope_updater_mean_only"] = float(
                    active_closed_loop_scope == "updater_mean"
                )
                result.metrics["closed_loop_scope_updater_mean_y_only"] = float(
                    active_closed_loop_scope == "updater_mean_y"
                )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = target_learning_rate
            if not bool(torch.isfinite(result.total_loss)):
                raise FloatingPointError(f"nonfinite {result.phase} loss at optimiser step {step}")
            if step < config.training.rgb_pretrain_steps:
                causal_support_present = True
                causal_trajectory_support = 0.0
                causal_fast_support = 0.0
                causal_objective_term_support = 0.0
            else:
                (
                    causal_support_present,
                    causal_trajectory_support,
                    causal_fast_support,
                    causal_objective_term_support,
                ) = _causal_training_support(result)
            if result.total_loss.requires_grad and causal_support_present:
                result.total_loss.backward()
            restricted_mean_snapshots = _prepare_restricted_updater_mean_update(
                model,
                optimizer,
                scope=active_closed_loop_scope,
            )
            try:
                gradient_diagnostics = _clip_training_gradients(
                    model,
                    config,
                    apply_perception_local_clip=(step >= config.training.rgb_pretrain_steps),
                )
            except FloatingPointError as error:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"{error} at optimiser step {step}") from error
            pre_clip_gradient_norm = gradient_diagnostics["gradient_norm_pre_clip"]
            effective_gradient_norm = gradient_diagnostics["gradient_norm_pre_global_clip"]
            if _has_effective_gradient(effective_gradient_norm, config):
                optimizer.step()
                _restore_restricted_updater_mean_update(
                    optimizer,
                    restricted_mean_snapshots,
                )
                finite_check_started = time.perf_counter()
                try:
                    _assert_finite_optimizer_update(model, optimizer)
                except FloatingPointError as error:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"{error} after optimiser step {step}") from error
                post_step_finite_check_seconds = time.perf_counter() - finite_check_started
                support_collapse_rollback_at_checkpoint = False
                break

            optimizer.zero_grad(set_to_none=True)
            if step < config.training.rgb_pretrain_steps:
                raise RuntimeError(
                    "RGB measurement pretraining produced no effective gradient "
                    f"at optimiser step {step}, data draw {training_data_draw_step - 1}"
                )
            skipped_no_gradient_batches += 1
            no_gradient_attempts += 1
            skipped_metrics = _result_metrics(
                result,
                learning_rate=target_learning_rate,
                gradient_norm=pre_clip_gradient_norm,
            )
            skipped_metrics.update(
                {
                    **gradient_diagnostics,
                    "optimizer_update_applied": 0.0,
                    "causal_training_support_present": float(causal_support_present),
                    "causal_trajectory_support_count": causal_trajectory_support,
                    "causal_fast_support_count": causal_fast_support,
                    "causal_objective_term_support_count": causal_objective_term_support,
                    "training_data_draw_step": float(training_data_draw_step),
                    "no_gradient_attempt_for_update": float(no_gradient_attempts),
                    "skipped_no_gradient_batches": float(skipped_no_gradient_batches),
                    "global_perception_trainable": float(global_perception_trainable),
                }
            )
            logger.log(
                step=step,
                split="train_skipped_no_gradient",
                metrics=skipped_metrics,
            )
            if no_gradient_attempts > config.training.maximum_no_gradient_batches_per_update:
                raise RuntimeError(
                    "causal training exhausted "
                    "training.maximum_no_gradient_batches_per_update without "
                    "an effective gradient at optimiser step "
                    f"{step}; last data draw was {training_data_draw_step - 1}"
                )

        completed_step = step + 1
        if training_data_draw_step != completed_step + skipped_no_gradient_batches:
            raise AssertionError(
                "trainer data-progress invariant failed after optimizer update "
                f"({training_data_draw_step} != {completed_step} + "
                f"{skipped_no_gradient_batches})"
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        last_metrics = _result_metrics(
            result,
            learning_rate=learning_rate,
            gradient_norm=pre_clip_gradient_norm,
        )
        last_metrics.update(gradient_diagnostics)
        last_metrics["global_perception_trainable"] = float(global_perception_trainable)
        last_metrics["optimizer_update_applied"] = 1.0
        last_metrics["causal_training_support_present"] = float(causal_support_present)
        last_metrics["causal_trajectory_support_count"] = causal_trajectory_support
        last_metrics["causal_fast_support_count"] = causal_fast_support
        last_metrics["causal_objective_term_support_count"] = causal_objective_term_support
        last_metrics["training_data_draw_step"] = float(training_data_draw_step)
        last_metrics["no_gradient_batches_before_update"] = float(no_gradient_attempts)
        last_metrics["skipped_no_gradient_batches"] = float(skipped_no_gradient_batches)
        last_metrics["post_step_finite_check_seconds"] = post_step_finite_check_seconds
        episode_seeds = raw_batch.get("seed")
        if isinstance(episode_seeds, Tensor):
            last_metrics["episode_seeds"] = ",".join(
                str(int(value)) for value in episode_seeds.detach().cpu().flatten().tolist()
            )
        metadata = raw_batch.get("metadata")
        if isinstance(metadata, Mapping):
            scenarios = metadata.get("scenario")
            if isinstance(scenarios, list):
                last_metrics["scenario_names"] = ",".join(str(item) for item in scenarios)
        should_log = (
            completed_step % max(1, config.training.log_every) == 0
            or completed_step == config.training.steps
        )
        if should_log:
            record = logger.log(
                step=completed_step,
                split="train",
                metrics=last_metrics,
            )
            print(
                f"step={completed_step}/{config.training.steps} "
                f"phase={result.phase} loss={record['loss_total']:.6f} "
                f"grad_preclip={record['gradient_norm_pre_clip']:.4f} "
                f"grad_applied={record['gradient_norm_applied']:.4f}",
                flush=True,
            )

        should_validate = completed_step == config.training.steps or (
            config.training.eval_every > 0
            and (
                completed_step % config.training.eval_every == 0
                or completed_step == config.training.rgb_pretrain_steps
            )
        )
        if should_validate and completed_step == config.training.steps:
            # Final validation can be much more expensive than the final
            # optimiser update. Persist those weights first, explicitly marked
            # pending, so an interrupted validation can be resumed without
            # repeating an update or pretending the run was fully validated.
            prevalidation_metrics = {
                key: float(value)
                for key, value in last_metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            prevalidation_metrics.update(retained_selector_metrics())
            prevalidation_metrics["final_validation_completed"] = 0.0
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=prevalidation_metrics,
                device=str(device),
                source_provenance=source_provenance,
            )
        if should_validate:
            if completed_step > config.training.rgb_pretrain_steps:
                (
                    _,
                    _,
                    _,
                    mutable_validation_support_failures,
                ) = validate_closed_loop_incumbent(
                    completed_step=completed_step,
                    learning_rate=learning_rate,
                    split="validation",
                )
                if mutable_validation_support_failures and best_rollout_selection is not None:
                    restore_safe_causal_incumbent(
                        completed_step=completed_step,
                        split="training_control_support_collapse",
                        support_failures=mutable_validation_support_failures,
                    )
            else:
                validate_measurement_incumbent(
                    completed_step=completed_step,
                    learning_rate=learning_rate,
                    split="validation",
                )

        should_checkpoint = (
            config.training.checkpoint_every > 0
            and completed_step % config.training.checkpoint_every == 0
        )
        if should_checkpoint or completed_step == config.training.steps:
            checkpoint_metrics = {
                key: float(value)
                for key, value in last_metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            checkpoint_metrics.update(retained_selector_metrics())
            if completed_step == config.training.steps and should_validate:
                checkpoint_metrics["final_validation_completed"] = 1.0
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                metrics=checkpoint_metrics,
                device=str(device),
                source_provenance=source_provenance,
            )

    final_validation_recovered = False
    if final_validation_recovery_pending:
        recovery_learning_rate = float(optimizer.param_groups[0]["lr"])
        if start_step > config.training.rgb_pretrain_steps:
            (
                final_validation,
                _,
                _,
                mutable_recovery_support_failures,
            ) = validate_closed_loop_incumbent(
                completed_step=start_step,
                learning_rate=recovery_learning_rate,
                split="validation_recovery",
            )
            if mutable_recovery_support_failures and best_rollout_selection is not None:
                restore_safe_causal_incumbent(
                    completed_step=start_step,
                    split="training_control_support_collapse_recovery",
                    support_failures=mutable_recovery_support_failures,
                )
        else:
            final_validation, _ = validate_measurement_incumbent(
                completed_step=start_step,
                learning_rate=recovery_learning_rate,
                split="validation_recovery",
            )
        last_metrics = {
            key: value
            for key, value in resume_metrics.items()
            if isinstance(value, str)
            or (isinstance(value, (int, float)) and math.isfinite(float(value)))
        }
        last_metrics["final_validation_completed"] = 1.0
        last_metrics["final_validation_loss_total"] = float(
            final_validation.total_loss.detach().cpu()
        )
        last_metrics["final_validation_phase"] = final_validation.phase
        recovery_checkpoint_metrics = {
            key: float(value)
            for key, value in resume_metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        recovery_checkpoint_metrics.update(retained_selector_metrics())
        recovery_checkpoint_metrics.update(
            {
                "final_validation_completed": 1.0,
                "final_validation_loss_total": float(final_validation.total_loss.detach().cpu()),
            }
        )
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            config=config,
            step=start_step,
            metrics=recovery_checkpoint_metrics,
            device=str(device),
            source_provenance=source_provenance,
        )
        final_validation_recovered = True

    no_op_exact_resume = (
        resume_path is not None
        and config.training.steps == start_step
        and not final_validation_recovered
    )
    if config.training.steps == start_step:
        # A zero-step or already-complete resume is still a valid, inspectable
        # local checkpoint rather than a silently empty run. Never deserialize
        # and rewrite an already-complete exact resume on the inspection CPU:
        # doing so would replace its historical phase device and accelerator RNG
        # and make a later extension fail exact-resume validation.
        if resume_path is None:
            selection_metrics = retained_selector_metrics()
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=start_step,
                metrics=selection_metrics,
                device=str(device),
                source_provenance=source_provenance,
            )
        elif not final_validation_recovered:
            resume_source = Path(resume_path).expanduser().resolve()
            if resume_source != last_path.resolve():
                shutil.copy2(resume_source, last_path)

    elapsed = time.perf_counter() - started
    verified_best_rollout = (
        _verified_selector_checkpoint(
            best_rollout_path,
            config,
            prefix="best_rollout",
            expected_model_state_hash=best_rollout_model_state_hash,
            expected_step=best_rollout_step,
            expected_device=closed_loop_device,
        )
        if best_rollout_validated
        else None
    )
    verified_reference_rollout = (
        _verified_selector_checkpoint(
            reference_rollout_path,
            config,
            prefix="reference_rollout",
            expected_model_state_hash=reference_rollout_model_state_hash,
            expected_step=reference_rollout_step,
            expected_device=closed_loop_device,
        )
        if reference_rollout_selection is not None
        else None
    )
    verified_best_measurement = (
        _verified_measurement_checkpoint(
            best_measurement_path,
            config,
            expected_model_state_hash=best_measurement_model_state_hash,
            expected_step=best_measurement_step,
            expected_device=measurement_device,
        )
        if best_measurement_validated
        else None
    )
    has_best_rollout_checkpoint = (
        verified_best_rollout is not None and verified_reference_rollout is not None
    )
    has_best_measurement_checkpoint = verified_best_measurement is not None
    if has_best_rollout_checkpoint:
        # Leave the in-memory runtime at the verified incumbent. ``last.pt``
        # intentionally remains the resumable final iterate.
        load_model_weights(
            best_rollout_path,
            model=model,
            expected_config=config,
        )
        selected_checkpoint = best_rollout_path
        selected_checkpoint_kind = "best_rollout"
    elif has_best_measurement_checkpoint:
        selected_checkpoint = best_measurement_path
        selected_checkpoint_kind = "best_measurement"
    else:
        selected_checkpoint = last_path
        selected_checkpoint_kind = "last_unvalidated"
    result_payload: dict[str, Any] = {
        "run_directory": str(run_directory),
        "resolved_config": str(resolved_config_path),
        "metrics_jsonl": str(logger.path),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(selected_checkpoint),
        "best_checkpoint_kind": selected_checkpoint_kind,
        "best_rollout_checkpoint": (
            str(best_rollout_path) if has_best_rollout_checkpoint else None
        ),
        "best_measurement_checkpoint": (
            str(best_measurement_path) if has_best_measurement_checkpoint else None
        ),
        "best_rollout_validated": has_best_rollout_checkpoint,
        "best_measurement_validated": has_best_measurement_checkpoint,
        "best_rollout_loss": best_rollout if has_best_rollout_checkpoint else None,
        "best_rollout_position_loss": (best_rollout if has_best_rollout_checkpoint else None),
        "best_rollout_position_rmse_m": (
            best_rollout_selection.position_rmse_m if best_rollout_selection is not None else None
        ),
        "best_rollout_velocity_rmse_mps": (
            best_rollout_selection.velocity_rmse_mps if best_rollout_selection is not None else None
        ),
        "best_rollout_target_coverage": (
            best_rollout_selection.target_coverage if best_rollout_selection is not None else None
        ),
        "best_rollout_prediction_precision": (
            best_rollout_selection.prediction_precision
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_collision_f1": (
            best_rollout_selection.collision_f1 if best_rollout_selection is not None else None
        ),
        "best_rollout_id_switch_rate": (
            best_rollout_selection.id_switch_rate if best_rollout_selection is not None else None
        ),
        "best_rollout_position_coverage90": (
            best_rollout_selection.position_coverage90
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_position_calibration_error90": (
            best_rollout_selection.position_calibration_error90
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_horizon_position_rmse_m": (
            best_rollout_selection.horizon_position_rmse_m
            if best_rollout_selection is not None
            else None
        ),
        "best_rollout_horizon_forecast_target_coverage": (
            best_rollout_selection.horizon_forecast_target_coverage
            if best_rollout_selection is not None
            else None
        ),
        "rollout_reference_checkpoint": (
            str(reference_rollout_path) if verified_reference_rollout is not None else None
        ),
        "rollout_validation_protocol_hash": _rollout_validation_protocol_hash(config),
        "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
        "measurement_selection_metric_version": _MEASUREMENT_SELECTION_METRIC_VERSION,
        "best_measurement_loss": (
            best_measurement_selection.score if best_measurement_selection is not None else None
        ),
        "best_measurement_world_position_mae_m": (
            best_measurement_selection.world_position_mae_m
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_all_proposal_world_position_mae_m": (
            best_measurement_selection.all_proposal_world_position_mae_m
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_runtime_birth_recall_at_0_5m": (
            best_measurement_selection.runtime_birth_recall
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_runtime_birth_precision_at_0_5m": (
            best_measurement_selection.runtime_birth_precision
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_runtime_birth_f1_at_0_5m": (
            best_measurement_selection.runtime_birth_f1
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_fast_bootstrap_target_coverage": (
            best_measurement_selection.fast_bootstrap_target_coverage
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_fast_roi_target_coverage": (
            best_measurement_selection.fast_roi_target_coverage
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_fast_roi_world_position_mae_m": (
            best_measurement_selection.fast_roi_world_position_mae_m
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_fast_roi_recall_at_0_5m": (
            best_measurement_selection.fast_roi_recall
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_fast_roi_precision_at_0_5m": (
            best_measurement_selection.fast_roi_precision
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_fast_roi_f1_at_0_5m": (
            best_measurement_selection.fast_roi_f1
            if best_measurement_selection is not None
            else None
        ),
        "best_measurement_fast_roi_improvement_m": (
            best_measurement_selection.fast_roi_improvement_m
            if best_measurement_selection is not None
            else None
        ),
        "completed_steps": config.training.steps,
        "model_parameter_count": model_parameter_count,
        "planned_training_episode_draws": (config.training.steps * config.training.batch_size),
        "training_batch_draws_total": training_data_draw_step,
        "skipped_no_gradient_batches": skipped_no_gradient_batches,
        "effective_training_episode_draws": (training_data_draw_step * config.training.batch_size),
        "nominal_dataset_passes": (
            training_data_draw_step * config.training.batch_size / config.training.train_episodes
        ),
        "train_episodes": config.training.train_episodes,
        "validation_episodes": config.training.validation_episodes,
        "scenario_families": list(config.simulator.scenario_mixture),
        "scenario_balanced_batches": config.training.scenario_balanced_batches,
        "rgb_pretrain_steps": min(config.training.steps, config.training.rgb_pretrain_steps),
        "closed_loop_steps": max(0, config.training.steps - config.training.rgb_pretrain_steps),
        "device": (
            str(resume_config_payload.get("device", device))
            if no_op_exact_resume and resume_config_payload is not None
            else str(device)
        ),
        "measurement_device": str(measurement_device),
        "closed_loop_device": str(closed_loop_device),
        "precision": resolved_device.precision,
        "elapsed_seconds": elapsed,
        "no_op_exact_resume": no_op_exact_resume,
        "final_validation_recovered": final_validation_recovered,
        "optimizer_updates_this_invocation": config.training.steps - start_step,
        "resumed_from": resumed_from,
        "preserved_accepted_validation_history": [
            str(path) for path in preserved_validation_history
        ],
        "preserved_accepted_validation_count": len(preserved_validation_history),
        "initialized_from": initialized_from,
        "last_metrics": (dict(resume_metrics) if no_op_exact_resume else last_metrics),
        "oracle_runtime_input_used": False,
    }
    summary_path = run_directory / "train_summary.json"
    same_run_resume = False
    if resume_path is not None:
        resume_source = Path(resume_path).expanduser().resolve()
        same_run_resume = (
            resume_source.parent.name == "checkpoints"
            and resume_source.parent.parent == run_directory
        )
    if no_op_exact_resume and same_run_resume and summary_path.is_file():
        # A completed run's summary is scientific evidence for the invocation
        # that performed its optimiser updates. A later no-op inspection may
        # append run_metadata resume history, but must not replace elapsed time,
        # final metrics, or selected-artifact evidence with an empty segment.
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "existing completed-run summary is unreadable; refusing to "
                "overwrite its evidence during a no-op exact resume"
            ) from error
        if not isinstance(existing_summary, dict):
            raise ValueError(
                "existing completed-run summary is not a JSON object; refusing "
                "to overwrite it during a no-op exact resume"
            )
        # Preserve the durable campaign evidence byte-for-byte, but make the
        # ephemeral CLI result truthful about this inspection invocation.
        inspection_result = dict(existing_summary)
        inspection_result.update(
            {
                "no_op_exact_resume": True,
                "optimizer_updates_this_invocation": 0,
                "resume_inspection_elapsed_seconds": elapsed,
                "resumed_from": str(Path(resume_path).expanduser().resolve()),
            }
        )
        return inspection_result
    atomic_write_text(
        summary_path,
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
    )
    return result_payload


__all__ = ["train_from_config"]
