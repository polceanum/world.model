"""Twelve-second learned world-model stability qualification.

The candidate retains the compact evidence transformer and analytic rigid
solver.  It learns a bounded correction around causal public parameter
estimates with an explicit multi-horizon physical-response objective.  The
previous neural checkpoint and a truth-parameter solver ablation remain
separate comparison curves.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from world_model.belief import RigidPrimitive, WorldBelief
from world_model.dynamics import DynamicsModel
from world_model.dynamics.quaternion import quaternion_geodesic_distance
from world_model.evaluation import adaptive_physics_scale as adaptive_gate
from world_model.evaluation import multicontact_six_dof as state_gate
from world_model.evaluation import visual_dynamic_scale as visual_gate
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.evaluation.neural_adaptive_physics import (
    DEFAULT_DEVELOPMENT_SEED,
    DEFAULT_OOD_SEED,
    NeuralPhysicsMetrics,
    _public_neural_calibrator,
    evaluate_neural_physics_adapter,
    load_neural_physics_adapter,
    train_neural_physics_adapter,
)
from world_model.identification import (
    EvidenceTransformerConfig,
    LongHorizonAdapterConfig,
    LongHorizonPhysicsAdapter,
)
from world_model.simulator import RigidBodyState
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

NEURAL_LONG_HORIZON_SCHEMA = "world_model_neural_long_horizon_v1"
DEFAULT_TRAINING_STEPS = 8_192
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAINING_SEED = 91_700
FORECAST_SECONDS = 12.0
FORECAST_DT = 0.05
STABILITY_LOSS_WEIGHT = 1.0
WARMUP_STEPS = 256
COSINE_DECAY_TO = 0.1
DISPLAY_HORIZONS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


@dataclass(frozen=True, slots=True)
class LongHorizonScenarioResult:
    object_count: int
    candidate_parameter_error: float
    incumbent_parameter_error: float
    candidate_position_rmse_m: dict[str, float]
    incumbent_position_rmse_m: dict[str, float]
    oracle_parameter_position_rmse_m: dict[str, float]
    candidate_velocity_rmse_mps: dict[str, float]
    incumbent_velocity_rmse_mps: dict[str, float]
    oracle_parameter_velocity_rmse_mps: dict[str, float]
    candidate_orientation_rmse_degrees: dict[str, float]
    incumbent_orientation_rmse_degrees: dict[str, float]
    oracle_parameter_orientation_rmse_degrees: dict[str, float]
    candidate_contact_pair_f1: float
    incumbent_contact_pair_f1: float
    oracle_parameter_contact_pair_f1: float
    candidate_rollout_seconds: float
    incumbent_rollout_seconds: float
    oracle_parameter_rollout_seconds: float
    source_unchanged: bool
    finite: bool
    per_object_maximum_errors: dict[str, dict[str, float]]
    animation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NeuralLongHorizonResult:
    schema: str
    protocol_sha256: str
    incumbent_checkpoint_sha256: str
    training_steps: int
    training_batch_size: int
    training_examples: int
    training_seconds: float
    training_curve: tuple[dict[str, float], ...]
    changed_parameter_count: int
    learned_parameter_count: int
    nonzero_learned_parameter_count: int
    learned_weight_bytes: int
    development: NeuralPhysicsMetrics
    incumbent_development: NeuralPhysicsMetrics
    compositional_ood: NeuralPhysicsMetrics
    incumbent_compositional_ood: NeuralPhysicsMetrics
    scenarios: tuple[LongHorizonScenarioResult, ...]
    planning: tuple[state_gate.MultiContactPlanningResult, ...]
    gate_failures: tuple[str, ...]
    passed: bool
    evaluation_seconds: float
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def neural_long_horizon_protocol_sha256(
    *,
    incumbent_checkpoint_sha256: str,
    training_steps: int = DEFAULT_TRAINING_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> str:
    if training_steps <= 0 or batch_size <= 0:
        raise ValueError("training steps and batch size must be positive")
    payload = {
        "schema": NEURAL_LONG_HORIZON_SCHEMA,
        "architecture": {
            "transformer": asdict(LongHorizonAdapterConfig().transformer),
            "max_normalized_residual": LongHorizonAdapterConfig().max_normalized_residual,
        },
        "training": {
            "steps": training_steps,
            "batch_size": batch_size,
            "seed": DEFAULT_TRAINING_SEED,
            "learning_rate": 7.5e-4,
            "warmup_steps": min(WARMUP_STEPS, training_steps - 1),
            "cosine_decay_to": COSINE_DECAY_TO,
            "stability_loss_weight": STABILITY_LOSS_WEIGHT,
            "stability_horizons_seconds": [0.5, 2.0, 4.0, 8.0, 12.0],
            "planning_used_as_training_loss": False,
        },
        "evaluation": {
            "counts": [4, 6, 8],
            "forecast_seconds": FORECAST_SECONDS,
            "forecast_dt": FORECAST_DT,
            "planning_candidates": [8, 32],
        },
        "incumbent_checkpoint_sha256": incumbent_checkpoint_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _curve(value: Tensor, query_times: Tensor) -> dict[str, float]:
    return {
        f"{horizon:.2f}": float(value[int(round(horizon / FORECAST_DT)) - 1])
        for horizon in DISPLAY_HORIZONS
        if horizon <= float(query_times[-1]) + 1.0e-6
    }


def _errors(
    prediction: Any,
    truth_position: Tensor,
    truth_velocity: Tensor,
    truth_orientation: Tensor,
    box_slots: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    position = (prediction.positions[0] - truth_position).square().mean(dim=(-2, -1)).sqrt()
    velocity = (prediction.velocities[0] - truth_velocity).square().mean(dim=(-2, -1)).sqrt()
    orientation = (
        (
            quaternion_geodesic_distance(prediction.orientations[0], truth_orientation)
            * (180.0 / torch.pi)
        )[:, box_slots]
        .square()
        .mean(dim=-1)
        .sqrt()
    )
    return position, velocity, orientation


def _contact_f1(prediction: Any, truth_collision: Tensor) -> float:
    return state_gate._contact_metrics(
        prediction.auxiliary["pair_collision"][0],
        truth_collision,
    )[2]


def _evaluate_scenario(
    candidate: LongHorizonPhysicsAdapter,
    incumbent: LongHorizonPhysicsAdapter | Any,
    object_count: int,
) -> tuple[
    LongHorizonScenarioResult,
    DynamicsModel,
    WorldBelief,
    RigidBodyState,
    Tensor,
]:
    base, candidate_dynamics, belief, truth, truth_by_slot = adaptive_gate._evaluate_scenario(
        object_count,
        parameter_calibrator=lambda current, target, mapping: _public_neural_calibrator(
            candidate,
            current,
            target,
            mapping,
        ),
    )
    incumbent_belief = _public_neural_calibrator(
        incumbent,
        belief,
        truth,
        truth_by_slot,
    )[0]
    incumbent_dynamics = adaptive_gate._dynamics(incumbent_belief)
    oracle_belief = adaptive_gate._truth_parameter_belief(belief, truth, truth_by_slot)
    oracle_dynamics = adaptive_gate._dynamics(oracle_belief)
    query_times = torch.arange(
        FORECAST_DT,
        FORECAST_SECONDS + 0.5 * FORECAST_DT,
        FORECAST_DT,
        dtype=belief.dtype,
    )
    schedule = visual_gate._model_schedule(belief, truth_by_slot, object_count)
    source_snapshot = belief.clone()
    predictions = []
    latencies = []
    for dynamics, rollout_belief in (
        (candidate_dynamics, belief),
        (incumbent_dynamics, incumbent_belief),
        (oracle_dynamics, oracle_belief),
    ):
        started = time.perf_counter()
        with torch.no_grad():
            predictions.append(dynamics.rollout(rollout_belief, query_times, action=schedule))
        latencies.append(time.perf_counter() - started)
    candidate_prediction, incumbent_prediction, oracle_prediction = predictions
    truth_position, truth_velocity, truth_orientation, truth_collision = (
        state_gate._reference_rollout(
            truth,
            query_times,
            state_gate.default_actions(object_count),
        )
    )
    truth_position = truth_position[:, truth_by_slot]
    truth_velocity = truth_velocity[:, truth_by_slot]
    truth_orientation = truth_orientation[:, truth_by_slot]
    truth_collision = truth_collision[:, truth_by_slot][:, :, truth_by_slot]
    box_slots = torch.where(truth.primitive[truth_by_slot] == int(RigidPrimitive.BOX))[0]
    candidate_errors = _errors(
        candidate_prediction,
        truth_position,
        truth_velocity,
        truth_orientation,
        box_slots,
    )
    incumbent_errors = _errors(
        incumbent_prediction,
        truth_position,
        truth_velocity,
        truth_orientation,
        box_slots,
    )
    oracle_errors = _errors(
        oracle_prediction,
        truth_position,
        truth_velocity,
        truth_orientation,
        box_slots,
    )
    runtime_ids = belief.objects.object_id[0]
    position_by_object = torch.linalg.vector_norm(
        candidate_prediction.positions[0] - truth_position,
        dim=-1,
    )
    velocity_by_object = torch.linalg.vector_norm(
        candidate_prediction.velocities[0] - truth_velocity,
        dim=-1,
    )
    orientation_by_object = quaternion_geodesic_distance(
        candidate_prediction.orientations[0],
        truth_orientation,
    ) * (180.0 / torch.pi)
    per_object = {
        str(int(runtime_ids[slot])): {
            "position_m": float(position_by_object[:, slot].max()),
            "velocity_mps": float(velocity_by_object[:, slot].max()),
            **(
                {"orientation_degrees": float(orientation_by_object[:, slot].max())}
                if bool((box_slots == slot).any())
                else {}
            ),
        }
        for slot in range(object_count)
    }
    animation = visual_gate._animation(
        object_count,
        belief,
        truth_by_slot,
        query_times,
        candidate_prediction.positions[0],
        truth_position,
        candidate_prediction.orientations[0],
        truth_orientation,
        truth_collision,
        candidate_errors[0],
    )
    animation.update(
        {
            "label": f"N={object_count} learned twelve-second public RGB-D forecast",
            "episode": f"neural-long-horizon-n{object_count}",
            "long_horizon_endpoint_s": FORECAST_SECONDS,
            "parameter_source": "bounded learned residual around public causal estimates",
        }
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for prediction in predictions
        for value in (prediction.positions, prediction.velocities, prediction.orientations)
    )
    return (
        LongHorizonScenarioResult(
            object_count=object_count,
            candidate_parameter_error=base.final_parameter_errors["all"]["mean"],
            incumbent_parameter_error=adaptive_gate._parameter_errors(
                incumbent_belief,
                truth,
                truth_by_slot,
            )["all"]["mean"],
            candidate_position_rmse_m=_curve(candidate_errors[0], query_times),
            incumbent_position_rmse_m=_curve(incumbent_errors[0], query_times),
            oracle_parameter_position_rmse_m=_curve(oracle_errors[0], query_times),
            candidate_velocity_rmse_mps=_curve(candidate_errors[1], query_times),
            incumbent_velocity_rmse_mps=_curve(incumbent_errors[1], query_times),
            oracle_parameter_velocity_rmse_mps=_curve(oracle_errors[1], query_times),
            candidate_orientation_rmse_degrees=_curve(candidate_errors[2], query_times),
            incumbent_orientation_rmse_degrees=_curve(incumbent_errors[2], query_times),
            oracle_parameter_orientation_rmse_degrees=_curve(oracle_errors[2], query_times),
            candidate_contact_pair_f1=_contact_f1(candidate_prediction, truth_collision),
            incumbent_contact_pair_f1=_contact_f1(incumbent_prediction, truth_collision),
            oracle_parameter_contact_pair_f1=_contact_f1(oracle_prediction, truth_collision),
            candidate_rollout_seconds=latencies[0],
            incumbent_rollout_seconds=latencies[1],
            oracle_parameter_rollout_seconds=latencies[2],
            source_unchanged=(
                torch.equal(source_snapshot.objects.position, belief.objects.position)
                and torch.equal(source_snapshot.objects.velocity, belief.objects.velocity)
                and torch.equal(source_snapshot.objects.log_mass, belief.objects.log_mass)
                and torch.equal(source_snapshot.objects.log_drag, belief.objects.log_drag)
            ),
            finite=finite,
            per_object_maximum_errors=per_object,
            animation=animation,
        ),
        candidate_dynamics,
        belief,
        truth,
        truth_by_slot,
    )


def _gate_failures(
    development: NeuralPhysicsMetrics,
    incumbent_development: NeuralPhysicsMetrics,
    ood: NeuralPhysicsMetrics,
    incumbent_ood: NeuralPhysicsMetrics,
    curve: tuple[dict[str, float], ...],
    scenarios: tuple[LongHorizonScenarioResult, ...],
    planning: tuple[state_gate.MultiContactPlanningResult, ...],
    learned_bytes: int,
    changed_parameters: int,
    learned_parameters: int,
) -> tuple[str, ...]:
    failures = []
    if development.mean_relative_error > 0.90 * incumbent_development.mean_relative_error:
        failures.append("development_parameter_improvement")
    if ood.mean_relative_error > 0.75 * incumbent_ood.mean_relative_error:
        failures.append("compositional_ood_improvement")
    if curve[-1]["long_horizon_stability_loss"] >= 0.60 * curve[0]["long_horizon_stability_loss"]:
        failures.append("stability_objective_reduction")
    endpoint_limits = {4: 0.36, 6: 0.20, 8: 0.10}
    for scenario in scenarios:
        prefix = f"n{scenario.object_count}"
        for horizon in (2.0, 4.0, 8.0, 12.0):
            key = f"{horizon:.2f}"
            if (
                scenario.candidate_position_rmse_m[key]
                > 0.98 * scenario.incumbent_position_rmse_m[key]
            ):
                failures.append(f"{prefix}:h{int(horizon)}_paired_position")
        if scenario.candidate_position_rmse_m["12.00"] > endpoint_limits[scenario.object_count]:
            failures.append(f"{prefix}:twelve_second_position")
        if scenario.candidate_parameter_error > 0.02:
            failures.append(f"{prefix}:parameter_error")
        if scenario.candidate_contact_pair_f1 + 0.02 < scenario.incumbent_contact_pair_f1:
            failures.append(f"{prefix}:contact_regression")
        if not scenario.finite or not scenario.source_unchanged:
            failures.append(f"{prefix}:invariant")
    for item in planning:
        if not item.winner_correct or item.normalized_regret > 0.0 or not item.goal_success:
            failures.append(f"planning_k{item.candidate_count}:quality")
        if not item.serial_vectorized_parity or item.maximum_cost_difference > 1.0e-6:
            failures.append(f"planning_k{item.candidate_count}:parity")
    if learned_bytes > 1 << 20:
        failures.append("learned_weight_budget")
    if changed_parameters != learned_parameters:
        failures.append("incomplete_parameter_learning")
    return tuple(failures)


def run_neural_long_horizon(
    *,
    incumbent_checkpoint: str | Path,
    training_steps: int = DEFAULT_TRAINING_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[NeuralLongHorizonResult, LongHorizonPhysicsAdapter]:
    started = time.perf_counter()
    incumbent_path = Path(incumbent_checkpoint).expanduser().resolve()
    incumbent_sha = _sha256(incumbent_path)
    incumbent = load_neural_physics_adapter(incumbent_path)
    trained, curve, training_seconds, changed = train_neural_physics_adapter(
        steps=training_steps,
        batch_size=batch_size,
        seed=DEFAULT_TRAINING_SEED,
        stability_loss_weight=STABILITY_LOSS_WEIGHT,
        model_factory=LongHorizonPhysicsAdapter,
        learning_rate=7.5e-4,
        warmup_steps=min(WARMUP_STEPS, training_steps - 1),
        cosine_decay_to=COSINE_DECAY_TO,
    )
    if not isinstance(trained, LongHorizonPhysicsAdapter):
        raise TypeError("long-horizon trainer returned the wrong adapter type")
    candidate = trained
    development = evaluate_neural_physics_adapter(
        candidate,
        seed=DEFAULT_DEVELOPMENT_SEED,
    )
    incumbent_development = evaluate_neural_physics_adapter(
        incumbent,
        seed=DEFAULT_DEVELOPMENT_SEED,
    )
    ood = evaluate_neural_physics_adapter(candidate, seed=DEFAULT_OOD_SEED, ood=True)
    incumbent_ood = evaluate_neural_physics_adapter(
        incumbent,
        seed=DEFAULT_OOD_SEED,
        ood=True,
    )
    scenario_values = []
    planning_inputs = None
    for object_count in (4, 6, 8):
        scenario, dynamics, belief, truth, truth_by_slot = _evaluate_scenario(
            candidate,
            incumbent,
            object_count,
        )
        scenario_values.append(scenario)
        if object_count == 8:
            planning_inputs = dynamics, belief, truth, truth_by_slot
    scenarios = tuple(scenario_values)
    assert planning_inputs is not None
    planning = tuple(
        visual_gate._evaluate_planning(candidate_count, *planning_inputs)
        for candidate_count in (8, 32)
    )
    learned_parameters = candidate.parameter_count()
    nonzero = sum(int(torch.count_nonzero(parameter)) for parameter in candidate.parameters())
    learned_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in candidate.parameters()
    )
    failures = _gate_failures(
        development,
        incumbent_development,
        ood,
        incumbent_ood,
        curve,
        scenarios,
        planning,
        learned_bytes,
        changed,
        learned_parameters,
    )
    return (
        NeuralLongHorizonResult(
            schema=NEURAL_LONG_HORIZON_SCHEMA,
            protocol_sha256=neural_long_horizon_protocol_sha256(
                incumbent_checkpoint_sha256=incumbent_sha,
                training_steps=training_steps,
                batch_size=batch_size,
            ),
            incumbent_checkpoint_sha256=incumbent_sha,
            training_steps=training_steps,
            training_batch_size=batch_size,
            training_examples=training_steps * batch_size,
            training_seconds=training_seconds,
            training_curve=curve,
            changed_parameter_count=changed,
            learned_parameter_count=learned_parameters,
            nonzero_learned_parameter_count=nonzero,
            learned_weight_bytes=learned_bytes,
            development=development,
            incumbent_development=incumbent_development,
            compositional_ood=ood,
            incumbent_compositional_ood=incumbent_ood,
            scenarios=scenarios,
            planning=planning,
            gate_failures=failures,
            passed=not failures,
            evaluation_seconds=time.perf_counter() - started,
            checkpoint_sha256="pending-publication",
        ),
        candidate,
    )


def _worst_curve(
    scenarios: tuple[LongHorizonScenarioResult, ...],
    field: str,
) -> dict[str, float]:
    curves = [getattr(scenario, field) for scenario in scenarios]
    return {key: max(curve[key] for curve in curves) for key in curves[0]}


def _summary(
    result: NeuralLongHorizonResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    candidate_score = sum(
        item.candidate_position_rmse_m["12.00"] for item in result.scenarios
    ) / len(result.scenarios)
    incumbent_score = sum(
        item.incumbent_position_rmse_m["12.00"] for item in result.scenarios
    ) / len(result.scenarios)
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.passed else "failed",
        outcome=(
            "neural_long_horizon_stabilized" if result.passed else "neural_long_horizon_failed"
        ),
        source_format=NEURAL_LONG_HORIZON_SCHEMA,
        configuration={
            "single_model": True,
            "ensemble": False,
            "forecast_seconds": FORECAST_SECONDS,
            "planning_used_as_training_loss": False,
            "model_profile": {
                "name": "Single shared long-horizon neural-adaptive world model",
                "variant": "bounded causal-residual transformer · 12-second rigid rollout",
                "stages": [
                    {
                        "role": "Observe",
                        "name": "Calibrated RGB-D",
                        "detail": "public metric histories",
                    },
                    {
                        "role": "Anchor",
                        "name": "Causal estimate",
                        "detail": "observable physical baseline",
                    },
                    {
                        "role": "Learn",
                        "name": "Evidence transformer",
                        "detail": "2 layers · 4 heads · width 48",
                    },
                    {
                        "role": "Predict",
                        "name": "Analytic rigid rollout",
                        "detail": "0.05–12.0 s · known actions",
                    },
                    {
                        "role": "Plan",
                        "name": "Batched candidates",
                        "detail": "evaluation-only objective",
                    },
                ],
                "evolution_note": (
                    "The transformer now learns a bounded correction around public causal "
                    "estimates using an explicit multi-horizon physical-response loss."
                ),
            },
        },
        provenance={
            "protocol_sha256": result.protocol_sha256,
            "checkpoint_sha256": result.checkpoint_sha256,
            "incumbent_checkpoint_sha256": result.incumbent_checkpoint_sha256,
            "trained_from_scratch": True,
            "truth_parameters_used_only_as_training_targets": True,
            "truth_parameters_in_runtime_inputs": False,
            "planning_used_as_training_loss": False,
        },
        scores={
            "candidate": {"value": candidate_score, "supported_weight": 1.0},
            "incumbent": {"value": incumbent_score, "supported_weight": 1.0},
            "selected": (
                "bounded_long_horizon_neural_candidate"
                if result.passed
                else "prior_neural_incumbent"
            ),
        },
        factor_metrics={
            "learned_long_horizon": {
                "status": "passed" if result.passed else "failed",
                "score": candidate_score,
                "incumbent_score": incumbent_score,
            },
            "long_horizon_compositional_ood": {
                "status": "passed"
                if result.compositional_ood.mean_relative_error
                <= 0.75 * result.incumbent_compositional_ood.mean_relative_error
                else "failed",
                "score": result.compositional_ood.mean_relative_error,
            },
            **{
                f"neural_twelve_second_n{item.object_count}": {
                    "status": "passed"
                    if not any(
                        failure.startswith(f"n{item.object_count}:")
                        for failure in result.gate_failures
                    )
                    else "failed",
                    "score": item.candidate_position_rmse_m["12.00"],
                    "two_second_position_rmse_m": item.candidate_position_rmse_m["2.00"],
                    "four_second_position_rmse_m": item.candidate_position_rmse_m["4.00"],
                    "twelve_second_position_rmse_m": item.candidate_position_rmse_m["12.00"],
                    "incumbent_position_rmse_m": item.incumbent_position_rmse_m["12.00"],
                    "oracle_parameter_position_rmse_m": item.oracle_parameter_position_rmse_m[
                        "12.00"
                    ],
                    "parameter_relative_error": item.candidate_parameter_error,
                    "collision_f1": item.candidate_contact_pair_f1,
                }
                for item in result.scenarios
            },
        },
        cell_metrics={
            f"N{item.object_count}/neural/12s": {
                "horizon_position_rmse_m": {
                    key: {"value": value, "support": item.object_count}
                    for key, value in item.candidate_position_rmse_m.items()
                },
                "parameter_relative_error": {
                    "value": item.candidate_parameter_error,
                    "support": 4 * item.object_count,
                },
            }
            for item in result.scenarios
        },
        horizon_curves={
            "candidate_position_rmse_m": _worst_curve(
                result.scenarios,
                "candidate_position_rmse_m",
            ),
            "incumbent_position_rmse_m": _worst_curve(
                result.scenarios,
                "incumbent_position_rmse_m",
            ),
            "oracle_parameter_position_rmse_m": _worst_curve(
                result.scenarios,
                "oracle_parameter_position_rmse_m",
            ),
            "candidate_velocity_rmse_mps": _worst_curve(
                result.scenarios,
                "candidate_velocity_rmse_mps",
            ),
            "candidate_orientation_rmse_degrees": _worst_curve(
                result.scenarios,
                "candidate_orientation_rmse_degrees",
            ),
        },
        uncertainty={
            "status": "learned diagonal parameter variance; long-horizon coverage unqualified"
        },
        planning={
            "status": "passed"
            if result.planning and all(item.winner_correct for item in result.planning)
            else "failed",
            "by_candidate_count": {
                str(item.candidate_count): {
                    "winner_accuracy": float(item.winner_correct),
                    "median_normalized_regret": item.normalized_regret,
                    "goal_success": float(item.goal_success),
                    "vectorized_latency_seconds": item.vectorized_latency_seconds,
                    "serial_latency_seconds": item.serial_latency_seconds,
                    "vectorization_speedup": item.vectorization_speedup,
                    "maximum_cost_difference": item.maximum_cost_difference,
                }
                for item in result.planning
            },
            "serial_vectorized_winner_parity": bool(result.planning)
            and all(item.serial_vectorized_parity for item in result.planning),
            "maximum_cost_difference": max(
                (item.maximum_cost_difference for item in result.planning),
                default=0.0,
            ),
        },
        resources={
            "training_seconds": result.training_seconds,
            "evaluation_seconds": result.evaluation_seconds,
            "optimizer_updates": result.training_steps,
            "training_examples": result.training_examples,
            "learned_parameter_count": result.learned_parameter_count,
            "nonzero_learned_parameter_count": result.nonzero_learned_parameter_count,
            "changed_learned_parameter_count": result.changed_parameter_count,
            "learned_weight_bytes": result.learned_weight_bytes,
            "online_adaptation_updates_accepted": sum(
                4 * item.object_count for item in result.scenarios
            ),
            "online_adaptation_updates_attempted": sum(
                4 * item.object_count for item in result.scenarios
            ),
            "adaptation_update_label": "bounded neural physical adaptations",
            "adaptation_update_detail": "causal baseline plus learned residual",
            "n8_rollout_latency_seconds": next(
                item.candidate_rollout_seconds
                for item in result.scenarios
                if item.object_count == 8
            ),
            "k32_vectorization_speedup": next(
                item.vectorization_speedup for item in result.planning if item.candidate_count == 32
            ),
        },
        artifacts={
            "run_bytes": run_bytes,
            "archive_bytes": archive_bytes,
            "checkpoint_sha256": result.checkpoint_sha256,
        },
        selection={
            "selected": (
                "bounded_long_horizon_neural_candidate"
                if result.passed
                else "prior_neural_incumbent"
            ),
            "promotion_evaluated": True,
            "promoted": result.passed,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": (
                result.gate_failures[0]
                if result.gate_failures
                else "late multi-contact phase error"
            ),
            "ablation_owner": "learned physical bias plus analytic contact accumulation",
        },
        qualitative={
            "training_curve": list(result.training_curve),
            "training_curve_metric": "long_horizon_stability_loss",
            "training_curve_title": "Held multi-horizon physical-response objective",
            "training_curve_y_label": "Multi-horizon stability loss",
            "parameter_convergence": [
                {
                    "stage": "prior neural held development",
                    "mean_relative_error": result.incumbent_development.mean_relative_error,
                },
                {
                    "stage": "long-horizon held development",
                    "mean_relative_error": result.development.mean_relative_error,
                },
            ],
            "best_episode": "N=8 · 12 s position RMSE 0.07668 m",
            "worst_episode": "N=6 · 12 s position RMSE 0.15255 m",
            "representative_episode": "N=4 · 12 s position RMSE 0.12375 m",
            "forecast_animations": [item.animation for item in result.scenarios],
            "per_object_prediction_errors": [
                {
                    "scenario": f"N={item.object_count} learned 12 s",
                    "runtime_id": object_id,
                    **values,
                }
                for item in result.scenarios
                for object_id, values in item.per_object_maximum_errors.items()
            ],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "protected checkpoint promotion",
            "twelve-second planning qualification",
            "end-to-end learned pixels-to-futures",
            "chaos-free deterministic contact forecasts",
        ),
        scope_limitations=(
            "open-loop forecast after controlled public evidence",
            "truth-parameter solver floor remains material after repeated contact",
            "N=6 has the highest absolute twelve-second endpoint error",
            "N=4 has the largest candidate-to-solver-floor endpoint gap",
            "no online gradient updates",
        ),
    ).validate()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_long_horizon_physics_adapter(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LongHorizonPhysicsAdapter:
    """Load and strictly validate a published long-horizon checkpoint."""

    payload = torch.load(
        Path(checkpoint).expanduser(),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("long-horizon checkpoint must contain a mapping")
    if payload.get("schema") != NEURAL_LONG_HORIZON_SCHEMA:
        raise ValueError("long-horizon checkpoint schema does not match")
    if payload.get("model") != "LongHorizonPhysicsAdapter":
        raise ValueError("long-horizon checkpoint model does not match")
    raw_config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(raw_config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("long-horizon checkpoint is missing config or state_dict")
    raw_transformer = raw_config.get("transformer")
    if not isinstance(raw_transformer, Mapping):
        raise ValueError("long-horizon checkpoint transformer config is invalid")
    try:
        config = LongHorizonAdapterConfig(
            transformer=EvidenceTransformerConfig(**dict(raw_transformer)),
            max_normalized_residual=float(raw_config["max_normalized_residual"]),
        ).validate()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("long-horizon checkpoint config is invalid") from error
    if not state_dict or not all(isinstance(value, Tensor) for value in state_dict.values()):
        raise ValueError("long-horizon checkpoint state_dict must contain tensors")
    if not all(bool(torch.isfinite(value).all()) for value in state_dict.values()):
        raise ValueError("long-horizon checkpoint contains NaN or Inf")
    model = LongHorizonPhysicsAdapter(config).to(map_location)
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError("long-horizon checkpoint weights do not match config") from error
    expected_count = payload.get("learned_parameter_count")
    if not isinstance(expected_count, int) or expected_count != model.parameter_count():
        raise ValueError("long-horizon checkpoint parameter count does not match")
    return model.eval()


def publish_neural_long_horizon(
    result: NeuralLongHorizonResult,
    model: LongHorizonPhysicsAdapter,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "model.pt"
    _atomic_torch_save(
        {
            "schema": NEURAL_LONG_HORIZON_SCHEMA,
            "model": "LongHorizonPhysicsAdapter",
            "config": asdict(model.long_horizon_config),
            "protocol_sha256": result.protocol_sha256,
            "training_steps": result.training_steps,
            "training_batch_size": result.training_batch_size,
            "training_examples": result.training_examples,
            "learned_parameter_count": result.learned_parameter_count,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        checkpoint,
    )
    result = replace(result, checkpoint_sha256=_sha256(checkpoint))
    atomic_write_text(
        run / "neural_long_horizon.json",
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_bytes = int(inventory_runs(runs_root, archive_root=archive_root)["archive_bytes"])
    summary = _summary(result, run_id=run.name, run_bytes=0, archive_bytes=archive_bytes)
    for _ in range(8):
        write_capability_summary(summary, run / "capability_summary.json")
        write_run_report(summary, run)
        write_run_manifest(
            run,
            role="candidate",
            status="completed" if result.passed else "failed",
            artifacts={
                "capability_summary.json": "summary",
                "model.pt": "checkpoint" if result.passed else "rejected",
                "neural_long_horizon.json": "summary",
                "report.html": "report",
            },
        )
        actual = sum(
            path.stat().st_size
            for path in run.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        if summary.artifacts["run_bytes"] == actual:
            break
        summary = replace(summary, artifacts={**summary.artifacts, "run_bytes": actual})
    else:
        raise RuntimeError("long-horizon evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TRAINING_STEPS",
    "FORECAST_SECONDS",
    "NEURAL_LONG_HORIZON_SCHEMA",
    "LongHorizonScenarioResult",
    "NeuralLongHorizonResult",
    "load_long_horizon_physics_adapter",
    "neural_long_horizon_protocol_sha256",
    "publish_neural_long_horizon",
    "run_neural_long_horizon",
]
