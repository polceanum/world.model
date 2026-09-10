"""Deterministic state-first long-horizon causal capability evaluation.

This ladder is deliberately separate from the frozen 56-frame dynamic-set
qualification protocol.  It exercises the shared public dynamics interface at
four and eight seconds with several known impulses, pair contacts, and bounded
floor/wall interactions.  Simulator state is used only to seed a declared
state-first probe and to score the completed public rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch import Tensor

from world_model.belief import BeliefFactory, MotionMode, WorldBelief
from world_model.dynamics import DynamicsModel, WorldImpulseAction, WorldImpulseSchedule
from world_model.evaluation.capability_factor_runner import _model_from_workbench_checkpoint
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.evaluation.general_capability import ALL_CAPABILITY_FACTORS
from world_model.evaluation.perceptual_scaling import (
    PerceptualScaleProbeResult,
    run_n8_perception_probe,
)
from world_model.planning import TerminalWorldPositionGoal, plan_counterfactual_actions
from world_model.simulator.physics import PhysicsConfig, SphereState, advance_spheres
from world_model.training.dynamic_set_config import load_config
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import enforce_run_budget, inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

LONG_HORIZON_REPORT_SCHEMA = "world_model_long_horizon_report_v1"
LONG_HORIZONS_SECONDS = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0)
LONG_HORIZON_FRAME_RATE = 20.0
LONG_HORIZON_BOUNDS = ((-2.0, 2.0), (0.0, 2.5), (-1.25, 1.25))
SEQUENCE_PLANNING_BOUNDS = ((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0))
LONG_HORIZON_ANIMATION_MAX_BYTES = 96 * 1024


@dataclass(frozen=True, slots=True)
class LongHorizonActionSpec:
    timestamp_s: float
    object_id: int
    impulse_world: tuple[float, float, float]

    def validate(self, *, duration_s: float, object_ids: frozenset[int]) -> None:
        if not 0.0 < self.timestamp_s <= duration_s or not math.isfinite(self.timestamp_s):
            raise ValueError("long-horizon action timestamp lies outside the rollout")
        if self.object_id not in object_ids:
            raise ValueError("long-horizon action target is absent from the source state")
        if len(self.impulse_world) != 3 or not all(map(math.isfinite, self.impulse_world)):
            raise ValueError("long-horizon action impulse must be a finite world vector")


@dataclass(frozen=True, slots=True)
class LongHorizonScenario:
    name: str
    duration_s: float
    gravity: tuple[float, float, float]
    object_id: tuple[int, ...]
    position: tuple[tuple[float, float, float], ...]
    velocity: tuple[tuple[float, float, float], ...]
    radius: tuple[float, ...]
    mass: tuple[float, ...]
    restitution: tuple[float, ...]
    drag: tuple[float, ...]
    friction: tuple[float, ...]
    actions: tuple[LongHorizonActionSpec, ...]

    def validate(self) -> LongHorizonScenario:
        count = len(self.object_id)
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("long-horizon scenario name must be one safe component")
        if self.duration_s not in {4.0, 8.0}:
            raise ValueError("long-horizon scenarios must run for four or eight seconds")
        if count < 1 or len(set(self.object_id)) != count or min(self.object_id) < 0:
            raise ValueError("scenario object IDs must be unique and nonnegative")
        for name in (
            "position",
            "velocity",
            "radius",
            "mass",
            "restitution",
            "drag",
            "friction",
        ):
            if len(getattr(self, name)) != count:
                raise ValueError(f"scenario {name} count does not match object IDs")
        if any(len(value) != 3 for value in (*self.position, *self.velocity)):
            raise ValueError("scenario position and velocity values must be 3-D")
        if any(value <= 0.0 for value in (*self.radius, *self.mass)):
            raise ValueError("scenario radius and mass must be positive")
        if any(not 0.0 <= value <= 1.0 for value in (*self.restitution, *self.friction)):
            raise ValueError("scenario restitution and friction must lie in [0,1]")
        if any(value < 0.0 for value in self.drag):
            raise ValueError("scenario drag must be nonnegative")
        if len(self.gravity) != 3 or not all(map(math.isfinite, self.gravity)):
            raise ValueError("scenario gravity must be a finite 3-D vector")
        object_ids = frozenset(self.object_id)
        previous = 0.0
        for action in self.actions:
            action.validate(duration_s=self.duration_s, object_ids=object_ids)
            if action.timestamp_s <= previous:
                raise ValueError("scenario action timestamps must be strictly increasing")
            previous = action.timestamp_s
        return self


@dataclass(frozen=True, slots=True)
class LongHorizonScenarioResult:
    name: str
    duration_s: float
    object_count: int
    position_rmse_m: dict[str, float]
    velocity_rmse_mps: dict[str, float]
    uncertainty_90_coverage: float
    collision_f1: float
    collision_timing_error_frames: int | None
    model_collision_intervals: int
    truth_collision_intervals: int
    known_action_count: int
    expected_action_count: int
    terminal_energy_error_j: float
    rollout_latency_seconds: float
    source_unchanged: bool
    animation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionSequencePlanningResult:
    scenario: str
    candidate_count: int
    oracle_winner: int
    selected_winner: int
    winner_correct: bool
    normalized_winner_margin: float
    normalized_regret: float
    terminal_goal_success: bool
    serial_vectorized_winner_parity: bool
    maximum_cost_difference: float
    replanned_winner: int
    replanning_consistent: bool
    vectorized_latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_long_horizon_scenarios() -> tuple[LongHorizonScenario, ...]:
    """Return the frozen impact-first pilot scenes."""

    return tuple(
        item.validate()
        for item in (
            LongHorizonScenario(
                name="two-action-pair-contact",
                duration_s=4.0,
                gravity=(0.0, 0.0, 0.0),
                object_id=(11, 29),
                position=((-1.15, 1.15, 0.0), (0.85, 1.15, 0.0)),
                velocity=((0.18, 0.0, 0.0), (-0.12, 0.0, 0.0)),
                radius=(0.18, 0.22),
                mass=(0.8, 1.5),
                restitution=(0.82, 0.68),
                drag=(0.03, 0.03),
                friction=(0.08, 0.12),
                actions=(
                    LongHorizonActionSpec(0.80, 11, (1.25, 0.0, 0.0)),
                    LongHorizonActionSpec(3.50, 29, (-0.65, 0.0, 0.0)),
                ),
            ),
            LongHorizonScenario(
                name="repeated-wall-replanning",
                duration_s=8.0,
                gravity=(0.0, 0.0, 0.0),
                object_id=(7,),
                position=((0.0, 1.0, 0.0),),
                velocity=((1.35, 0.0, 0.0),),
                radius=(0.18,),
                mass=(1.0,),
                restitution=(0.92,),
                drag=(0.01,),
                friction=(0.03,),
                actions=(
                    LongHorizonActionSpec(1.70, 7, (-0.35, 0.0, 0.0)),
                    LongHorizonActionSpec(4.20, 7, (0.55, 0.0, 0.0)),
                    LongHorizonActionSpec(7.40, 7, (-0.40, 0.0, 0.0)),
                ),
            ),
            LongHorizonScenario(
                name="floor-wall-compound",
                duration_s=8.0,
                gravity=(0.0, -1.4, 0.0),
                object_id=(3, 17),
                position=((-0.85, 1.85, 0.0), (0.55, 1.25, 0.0)),
                velocity=((0.95, -0.15, 0.0), (-0.30, 0.25, 0.0)),
                radius=(0.16, 0.21),
                mass=(0.75, 1.35),
                restitution=(0.78, 0.66),
                drag=(0.02, 0.035),
                friction=(0.10, 0.18),
                actions=(
                    LongHorizonActionSpec(1.25, 17, (-0.55, 0.60, 0.0)),
                    LongHorizonActionSpec(3.75, 3, (0.45, 0.85, 0.0)),
                ),
            ),
        )
    )


def long_horizon_scenario_sha256(
    scenarios: Sequence[LongHorizonScenario] | None = None,
) -> str:
    selected = default_long_horizon_scenarios() if scenarios is None else tuple(scenarios)
    payload = json.dumps(
        [asdict(item.validate()) for item in selected],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sphere_state(scenario: LongHorizonScenario) -> SphereState:
    count = len(scenario.object_id)
    dtype = torch.float32
    return SphereState(
        object_id=torch.tensor(scenario.object_id, dtype=torch.int64),
        active=torch.ones(count, dtype=torch.bool),
        position=torch.tensor(scenario.position, dtype=dtype),
        velocity=torch.tensor(scenario.velocity, dtype=dtype),
        radius=torch.tensor(scenario.radius, dtype=dtype).unsqueeze(-1),
        mass=torch.tensor(scenario.mass, dtype=dtype).unsqueeze(-1),
        restitution=torch.tensor(scenario.restitution, dtype=dtype).unsqueeze(-1),
        drag=torch.tensor(scenario.drag, dtype=dtype).unsqueeze(-1),
        friction=torch.tensor(scenario.friction, dtype=dtype).unsqueeze(-1),
        albedo=torch.linspace(0.2, 0.9, count * 3, dtype=dtype).reshape(count, 3),
        orientation=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, dtype=dtype),
        angular_velocity=torch.zeros(count, 3, dtype=dtype),
        sleeping=torch.zeros(count, dtype=torch.bool),
        sleep_counter=torch.zeros(count, dtype=torch.int64),
    )


def _belief_from_state(dynamics: DynamicsModel, state: SphereState, scenario: LongHorizonScenario):
    config = dynamics.config
    factory = BeliefFactory(
        max_objects=state.max_objects,
        modal_count=config.modal_count,
        modal_dim=config.modal_dim,
        residual_dynamics_dim=config.residual_dynamics_dim,
        global_code_dim=config.global_code_dim,
        geometry_dim=config.geometry_dim,
        appearance_dim=config.appearance_dim,
        parameter_memory_dim=config.parameter_memory_dim,
        initial_radius=0.21,
        initial_mass=1.0,
        initial_drag=0.05,
        initial_restitution=0.7,
        initial_friction=0.2,
    )
    belief = factory.create(batch_size=1, gravity=scenario.gravity)
    objects = belief.objects.clone()
    objects.active[0] = state.active
    objects.object_id[0] = state.object_id
    objects.position[0] = state.position
    objects.velocity[0] = state.velocity
    objects.orientation[0] = state.orientation
    objects.angular_velocity[0] = state.angular_velocity
    objects.geometry[0, :, 0] = state.radius[:, 0]
    objects.log_mass[0] = state.mass.log()
    objects.restitution_logit[0] = torch.logit(state.restitution.clamp(1.0e-6, 1 - 1.0e-6))
    objects.log_drag[0] = state.drag.clamp_min(1.0e-8).log()
    objects.friction_logit[0] = torch.logit(state.friction.clamp(1.0e-6, 1 - 1.0e-6))
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., MotionMode.FREE] = 4.0
    objects.fast_log_variance.fill_(math.log(1.0e-6))
    return belief.replace(
        objects=objects,
        next_object_id=torch.tensor([max(scenario.object_id) + 1], dtype=torch.int64),
        metadata={
            "state_first_probe": True,
            "simulator_truth_not_observed": True,
        },
    ).validate(log_variance_bounds=(-32.0, 20.0))


def _bounded_dynamics(
    dynamics: DynamicsModel,
    *,
    world_bounds: tuple[tuple[float, float], ...] = LONG_HORIZON_BOUNDS,
) -> DynamicsModel:
    bounded = DynamicsModel(config=replace(dynamics.config, world_bounds=world_bounds)).to(
        dtype=next(dynamics.parameters()).dtype
    )
    # Environment planes are configuration, not learned checkpoint state.
    # Loading a checkpoint built for the historical remote-boundary protocol
    # would otherwise silently replace these freshly constructed offsets.
    plane_normals = bounded.events.resolver.plane_normals.clone()
    plane_offsets = bounded.events.resolver.plane_offsets.clone()
    bounded.load_state_dict(dynamics.state_dict(), strict=True)
    with torch.no_grad():
        bounded.events.resolver.plane_normals.copy_(plane_normals)
        bounded.events.resolver.plane_offsets.copy_(plane_offsets)
    bounded.eval()
    return bounded


def _action_schedule(belief: WorldBelief, scenario: LongHorizonScenario):
    return WorldImpulseSchedule(
        actions=tuple(
            WorldImpulseAction(
                timestamp=belief.timestamp.new_tensor([spec.timestamp_s]),
                object_id=torch.tensor([spec.object_id], dtype=torch.int64),
                impulse_world=belief.timestamp.new_tensor([spec.impulse_world]),
            )
            for spec in scenario.actions
        )
    )


def _apply_truth_impulse(state: SphereState, spec: LongHorizonActionSpec) -> SphereState:
    matches = state.active & (state.object_id == spec.object_id)
    if int(matches.sum()) != 1:
        raise ValueError("truth action target did not resolve exactly once")
    impulse = state.position.new_tensor(spec.impulse_world)
    velocity = state.velocity + torch.where(
        matches.unsqueeze(-1),
        impulse.unsqueeze(0) / state.mass,
        torch.zeros_like(state.velocity),
    )
    return replace(
        state,
        velocity=velocity,
        sleeping=state.sleeping & ~matches,
        sleep_counter=torch.where(
            matches, torch.zeros_like(state.sleep_counter), state.sleep_counter
        ),
    )


def _query_times(duration_s: float) -> Tensor:
    count = int(round(duration_s * LONG_HORIZON_FRAME_RATE))
    return torch.arange(1, count + 1, dtype=torch.float32) / LONG_HORIZON_FRAME_RATE


def _truth_rollout(
    scenario: LongHorizonScenario,
    query_times: Tensor,
    *,
    world_bounds: tuple[tuple[float, float], ...] = LONG_HORIZON_BOUNDS,
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Any]]]:
    state = _sphere_state(scenario)
    physics = PhysicsConfig(
        gravity=scenario.gravity,
        bounds=world_bounds,
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )
    current_time = 0.0
    action_index = 0
    positions: list[Tensor] = []
    velocities: list[Tensor] = []
    collision_rows: list[bool] = []
    events: list[dict[str, Any]] = []
    for frame_index, target_tensor in enumerate(query_times, start=1):
        target = float(target_tensor)
        collision = False
        while (
            action_index < len(scenario.actions)
            and scenario.actions[action_index].timestamp_s <= target + 1.0e-8
        ):
            spec = scenario.actions[action_index]
            state, interval = advance_spheres(state, spec.timestamp_s - current_time, physics)
            collision |= bool(
                torch.logical_or(
                    interval.pair_collision.any(),
                    interval.boundary_collision.any(),
                )
            )
            current_time = spec.timestamp_s
            state = _apply_truth_impulse(state, spec)
            events.append(
                {
                    "frame": int(round(spec.timestamp_s * LONG_HORIZON_FRAME_RATE)),
                    "kind": "known action",
                }
            )
            action_index += 1
        state, interval = advance_spheres(state, max(0.0, target - current_time), physics)
        interval_collision = bool(
            torch.logical_or(
                interval.pair_collision.any(),
                interval.boundary_collision.any(),
            )
        )
        collision |= interval_collision
        if collision and (not collision_rows or not collision_rows[-1]):
            events.append({"frame": frame_index, "kind": "reference collision"})
        current_time = target
        positions.append(state.position.clone())
        velocities.append(state.velocity.clone())
        collision_rows.append(collision)
    return (
        torch.stack(positions),
        torch.stack(velocities),
        torch.tensor(collision_rows, dtype=torch.bool),
        sorted(events, key=lambda item: (int(item["frame"]), str(item["kind"]))),
    )


def _truth_candidate_terminals(
    scenario: LongHorizonScenario,
    candidate_specs: Sequence[Sequence[LongHorizonActionSpec]],
    *,
    target_slot: int,
) -> Tensor:
    """Share the simulator prefix before branching at the final action."""

    if not candidate_specs:
        raise ValueError("truth candidate evaluation requires candidates")
    shared_prefix = tuple(candidate_specs[0][:-1])
    if any(tuple(specs[:-1]) != shared_prefix for specs in candidate_specs):
        raise ValueError("planning oracle candidates must share their causal prefix")
    final_timestamp = candidate_specs[0][-1].timestamp_s
    if any(specs[-1].timestamp_s != final_timestamp for specs in candidate_specs):
        raise ValueError("planning oracle branches must share a final action time")
    physics = PhysicsConfig(
        gravity=scenario.gravity,
        bounds=SEQUENCE_PLANNING_BOUNDS,
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )
    prefix_state = _sphere_state(scenario)
    current_time = 0.0
    for spec in shared_prefix:
        prefix_state, _ = advance_spheres(
            prefix_state,
            spec.timestamp_s - current_time,
            physics,
        )
        prefix_state = _apply_truth_impulse(prefix_state, spec)
        current_time = spec.timestamp_s
    prefix_state, _ = advance_spheres(
        prefix_state,
        final_timestamp - current_time,
        physics,
    )
    terminals = []
    for specs in candidate_specs:
        branch = _apply_truth_impulse(prefix_state, specs[-1])
        branch, _ = advance_spheres(
            branch,
            scenario.duration_s - final_timestamp,
            physics,
        )
        terminals.append(branch.position[target_slot])
    return torch.stack(terminals)


def _collision_f1(predicted: Tensor, target: Tensor) -> float:
    true_positive = int((predicted & target).sum())
    false_positive = int((predicted & ~target).sum())
    false_negative = int((~predicted & target).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


def _collision_timing_error(predicted: Tensor, target: Tensor) -> int | None:
    predicted_indices = torch.nonzero(predicted, as_tuple=False).flatten()
    target_indices = torch.nonzero(target, as_tuple=False).flatten()
    if not predicted_indices.numel() and not target_indices.numel():
        return 0
    if not predicted_indices.numel() or not target_indices.numel():
        return None
    errors = [
        int((predicted_indices - target_index).abs().min()) for target_index in target_indices
    ]
    return max(errors, default=0)


def _energy(
    position: Tensor,
    velocity: Tensor,
    state: SphereState,
    gravity: tuple[float, float, float],
) -> Tensor:
    mass = state.mass[:, 0]
    kinetic = 0.5 * mass * velocity.square().sum(dim=-1)
    gravity_tensor = position.new_tensor(gravity)
    potential = -mass * torch.einsum("nd,d->n", position, gravity_tensor)
    return (kinetic + potential).sum()


def _animation_payload(
    scenario: LongHorizonScenario,
    model_positions: Tensor,
    truth_positions: Tensor,
    events: list[dict[str, Any]],
    endpoint_rmse: float,
) -> dict[str, Any]:
    state = _sphere_state(scenario)
    indices = list(range(3, model_positions.shape[0], 4))
    if not indices or indices[-1] != model_positions.shape[0] - 1:
        indices.append(model_positions.shape[0] - 1)
    frames: list[dict[str, Any]] = [
        {
            "frame": 0,
            "time_s": 0.0,
            "truth": [
                [int(identifier), round(float(point[0]), 4), round(float(point[1]), 4)]
                for identifier, point in zip(state.object_id, state.position, strict=True)
            ],
            "model": [
                [int(identifier), round(float(point[0]), 4), round(float(point[1]), 4)]
                for identifier, point in zip(state.object_id, state.position, strict=True)
            ],
        }
    ]
    for index in indices:
        frames.append(
            {
                "frame": index + 1,
                "time_s": round((index + 1) / LONG_HORIZON_FRAME_RATE, 3),
                "truth": [
                    [int(identifier), round(float(point[0]), 4), round(float(point[1]), 4)]
                    for identifier, point in zip(
                        state.object_id,
                        truth_positions[index],
                        strict=True,
                    )
                ],
                "model": [
                    [int(identifier), round(float(point[0]), 4), round(float(point[1]), 4)]
                    for identifier, point in zip(
                        state.object_id,
                        model_positions[index],
                        strict=True,
                    )
                ],
            }
        )
    payload = {
        "schema": "world_model_compact_long_horizon_animation_v1",
        "label": scenario.name,
        "episode": f"state-first:{scenario.name}",
        "object_count": len(scenario.object_id),
        "contact": True,
        "dynamic_membership": False,
        "mode": "forecast",
        "frame_rate": LONG_HORIZON_FRAME_RATE,
        "anchor_frame": 0,
        "long_horizon_endpoint_s": scenario.duration_s,
        "endpoint_position_rmse_m": round(endpoint_rmse, 6),
        "rollout_horizons_s": [frame["time_s"] for frame in frames[1:]],
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {
            "horizontal": [LONG_HORIZON_BOUNDS[0][0], LONG_HORIZON_BOUNDS[0][1]],
            "vertical": [LONG_HORIZON_BOUNDS[1][0], LONG_HORIZON_BOUNDS[1][1]],
        },
        "frames": frames,
        "events": events,
        "identity_alignment": "declared state-first persistent IDs",
        "reference": "private simulator future opened only after public rollout",
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > LONG_HORIZON_ANIMATION_MAX_BYTES:
        raise ValueError("long-horizon animation exceeds its 96 KiB serialized ceiling")
    return payload


def evaluate_long_horizon_scenario(
    dynamics: DynamicsModel,
    scenario: LongHorizonScenario,
    *,
    timing_repetitions: int = 1,
) -> LongHorizonScenarioResult:
    """Evaluate one deterministic scenario without mutating caller state."""

    scenario.validate()
    if not isinstance(dynamics, DynamicsModel):
        raise TypeError("long-horizon evaluation requires DynamicsModel")
    if timing_repetitions < 1:
        raise ValueError("timing_repetitions must be positive")
    bounded = _bounded_dynamics(dynamics)
    state = _sphere_state(scenario)
    belief = _belief_from_state(bounded, state, scenario)
    source = belief.clone()
    schedule = _action_schedule(belief, scenario)
    query_times = _query_times(scenario.duration_s)
    truth_positions, truth_velocities, truth_collision, truth_events = _truth_rollout(
        scenario,
        query_times,
    )
    latencies: list[float] = []
    trajectory = None
    with torch.no_grad():
        for _ in range(timing_repetitions):
            started = time.perf_counter()
            trajectory = bounded.rollout(belief, query_times, action=schedule)
            latencies.append(time.perf_counter() - started)
    assert trajectory is not None and trajectory.event_logits is not None
    model_positions = trajectory.positions[0]
    model_velocities = trajectory.velocities[0]
    active = state.active
    position_error = model_positions[:, active] - truth_positions[:, active]
    velocity_error = model_velocities[:, active] - truth_velocities[:, active]
    horizon_indices = {
        horizon: int(round(horizon * LONG_HORIZON_FRAME_RATE)) - 1
        for horizon in LONG_HORIZONS_SECONDS
        if horizon <= scenario.duration_s
    }
    position_rmse = {
        f"{horizon:g}": float(position_error[index].square().mean().sqrt())
        for horizon, index in horizon_indices.items()
    }
    velocity_rmse = {
        f"{horizon:g}": float(velocity_error[index].square().mean().sqrt())
        for horizon, index in horizon_indices.items()
    }
    position_slice = slice(0, 3)
    sigma = torch.exp(0.5 * trajectory.fast_log_variance[0, :, active, position_slice])
    coverage = float((position_error.abs() <= 1.6448536269514722 * sigma).float().mean())
    model_collision = trajectory.event_logits[0, :, :, MotionMode.COLLISION].gt(0.0).any(dim=-1)
    known_count = int(trajectory.auxiliary["known_action_count"].sum())
    timing_error = _collision_timing_error(model_collision, truth_collision)
    source_unchanged = bool(
        torch.equal(belief.timestamp, source.timestamp)
        and torch.equal(belief.objects.position, source.objects.position)
        and torch.equal(belief.objects.velocity, source.objects.velocity)
    )
    terminal_energy_error = float(
        (
            _energy(model_positions[-1], model_velocities[-1], state, scenario.gravity)
            - _energy(truth_positions[-1], truth_velocities[-1], state, scenario.gravity)
        ).abs()
    )
    endpoint_rmse = position_rmse[f"{scenario.duration_s:g}"]
    return LongHorizonScenarioResult(
        name=scenario.name,
        duration_s=scenario.duration_s,
        object_count=len(scenario.object_id),
        position_rmse_m=position_rmse,
        velocity_rmse_mps=velocity_rmse,
        uncertainty_90_coverage=coverage,
        collision_f1=_collision_f1(model_collision, truth_collision),
        collision_timing_error_frames=timing_error,
        model_collision_intervals=int(model_collision.sum()),
        truth_collision_intervals=int(truth_collision.sum()),
        known_action_count=known_count,
        expected_action_count=len(scenario.actions),
        terminal_energy_error_j=terminal_energy_error,
        rollout_latency_seconds=median(latencies),
        source_unchanged=source_unchanged,
        animation=_animation_payload(
            scenario,
            model_positions,
            truth_positions,
            truth_events,
            endpoint_rmse,
        ),
    )


def _planning_candidate_specs(
    scenario: LongHorizonScenario,
    candidate_count: int,
) -> tuple[tuple[LongHorizonActionSpec, ...], ...]:
    if candidate_count not in {8, 32}:
        raise ValueError("action-sequence planning supports K=8 or K=32")
    if len(scenario.actions) < 2:
        raise ValueError("action-sequence planning requires at least two actions")
    base = scenario.actions[-1]
    candidates: list[tuple[LongHorizonActionSpec, ...]] = [scenario.actions]
    for index in range(1, candidate_count):
        angle = 2.0 * math.pi * (index - 1) / max(1, candidate_count - 1)
        magnitude = 0.35 + 0.08 * ((index - 1) % 4)
        delta = (
            magnitude * math.cos(angle),
            magnitude * math.sin(angle),
            0.18 * math.sin(2.0 * angle),
        )
        varied = replace(
            base,
            impulse_world=tuple(
                float(value + change)
                for value, change in zip(base.impulse_world, delta, strict=True)
            ),
        )
        candidates.append((*scenario.actions[:-1], varied))
    return tuple(candidates)


def _schedule_from_specs(
    belief: WorldBelief,
    specs: Sequence[LongHorizonActionSpec],
) -> WorldImpulseSchedule:
    return WorldImpulseSchedule(
        actions=tuple(
            WorldImpulseAction(
                timestamp=belief.timestamp.new_tensor([spec.timestamp_s]),
                object_id=torch.tensor([spec.object_id], dtype=torch.int64),
                impulse_world=belief.timestamp.new_tensor([spec.impulse_world]),
            )
            for spec in specs
        )
    )


def evaluate_action_sequence_planning(
    dynamics: DynamicsModel,
    scenario: LongHorizonScenario,
    *,
    candidate_count: int,
) -> ActionSequencePlanningResult:
    """Score K scheduled interventions and repeat the choice after one action."""

    scenario.validate()
    bounded = _bounded_dynamics(dynamics, world_bounds=SEQUENCE_PLANNING_BOUNDS)
    state = _sphere_state(scenario)
    belief = _belief_from_state(bounded, state, scenario)
    candidate_specs = _planning_candidate_specs(scenario, candidate_count)
    candidate_schedules = tuple(_schedule_from_specs(belief, specs) for specs in candidate_specs)
    query_times = torch.tensor([scenario.duration_s], dtype=torch.float32)
    target_id = scenario.actions[-1].object_id
    target_slot = scenario.object_id.index(target_id)
    terminal = _truth_candidate_terminals(
        scenario,
        candidate_specs,
        target_slot=target_slot,
    )
    goal_position = terminal[0]
    truth_cost = (terminal - goal_position).square().sum(dim=-1)
    oracle_winner = int(truth_cost.argmin())
    sorted_cost = torch.sort(truth_cost).values
    scale = max(float(truth_cost.max()), 1.0e-12)
    winner_margin = float((sorted_cost[1] - sorted_cost[0]) / scale)
    if oracle_winner != 0 or winner_margin <= 0.0:
        raise RuntimeError("planning task does not have the certified base winner")
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([target_id], dtype=torch.int64),
        position_world=goal_position.reshape(1, 3).to(belief.objects.position),
    )
    with torch.no_grad():
        started = time.perf_counter()
        vectorized = plan_counterfactual_actions(
            bounded,
            belief,
            query_times,
            candidate_schedules,
            goal,
            return_events=False,
            return_auxiliary=False,
        )
        vectorized_latency = time.perf_counter() - started
        serial = plan_counterfactual_actions(
            bounded,
            belief,
            query_times,
            candidate_schedules,
            goal,
            candidate_vectorized=False,
            return_events=False,
            return_auxiliary=False,
        )
    maximum_cost_difference = float(
        (vectorized.total_cost - serial.total_cost).abs().max().detach()
    )
    selected = int(vectorized.selected_index[0])
    selected_truth_cost = float(truth_cost[selected])
    normalized_regret = selected_truth_cost / scale

    first_action = candidate_specs[0][0]
    replan_timestamp = first_action.timestamp_s + 1.0 / LONG_HORIZON_FRAME_RATE
    prefix = _schedule_from_specs(belief, (first_action,))
    replanning_belief = bounded.predict(belief, replan_timestamp, action=prefix)
    remaining_specs = tuple(
        tuple(item for item in specs if item.timestamp_s > replan_timestamp)
        for specs in candidate_specs
    )
    if any(not specs for specs in remaining_specs):
        raise RuntimeError("planning task has no remaining action after replanning anchor")
    remaining_schedules = tuple(
        _schedule_from_specs(replanning_belief, specs) for specs in remaining_specs
    )
    with torch.no_grad():
        replanned = plan_counterfactual_actions(
            bounded,
            replanning_belief,
            [scenario.duration_s - replan_timestamp],
            remaining_schedules,
            goal,
            return_events=False,
            return_auxiliary=False,
        )
    replanned_winner = int(replanned.selected_index[0])
    return ActionSequencePlanningResult(
        scenario=scenario.name,
        candidate_count=candidate_count,
        oracle_winner=oracle_winner,
        selected_winner=selected,
        winner_correct=selected == oracle_winner,
        normalized_winner_margin=winner_margin,
        normalized_regret=normalized_regret,
        terminal_goal_success=math.sqrt(selected_truth_cost) <= 0.10,
        serial_vectorized_winner_parity=torch.equal(
            vectorized.selected_index,
            serial.selected_index,
        ),
        maximum_cost_difference=maximum_cost_difference,
        replanned_winner=replanned_winner,
        replanning_consistent=replanned_winner == selected,
        vectorized_latency_seconds=vectorized_latency,
    )


def action_sequence_gate_failures(
    results: Sequence[ActionSequencePlanningResult],
) -> tuple[str, ...]:
    failures: list[str] = []
    for candidate_count, minimum_accuracy, maximum_regret in (
        (8, 0.90, 0.05),
        (32, 0.85, 0.07),
    ):
        selected = [item for item in results if item.candidate_count == candidate_count]
        if not selected:
            failures.append(f"planning_k{candidate_count}:unmeasured")
            continue
        accuracy = sum(item.winner_correct for item in selected) / len(selected)
        regrets = sorted(item.normalized_regret for item in selected)
        median_regret = median(regrets)
        goal_success = sum(item.terminal_goal_success for item in selected) / len(selected)
        if accuracy < minimum_accuracy:
            failures.append(f"planning_k{candidate_count}:winner_accuracy")
        if median_regret > maximum_regret:
            failures.append(f"planning_k{candidate_count}:median_regret")
        if goal_success < 0.90:
            failures.append(f"planning_k{candidate_count}:goal_success")
    for item in results:
        if item.normalized_winner_margin < 0.05:
            failures.append(f"{item.scenario}:k{item.candidate_count}:winner_margin")
        if not item.serial_vectorized_winner_parity or item.maximum_cost_difference > 1.0e-6:
            failures.append(f"{item.scenario}:k{item.candidate_count}:serial_vectorized_parity")
        if not item.replanning_consistent:
            failures.append(f"{item.scenario}:k{item.candidate_count}:replanning_consistency")
    return tuple(failures)


def long_horizon_gate_failures(
    results: Sequence[LongHorizonScenarioResult],
) -> tuple[str, ...]:
    """Apply geometry-derived pilot floors frozen before model tuning."""

    horizon_limits = {
        "0.05": 0.012,
        "0.1": 0.014,
        "0.25": 0.020,
        "0.5": 0.030,
        "1": 0.050,
        "2": 0.080,
        "4": 0.150,
        "8": 0.300,
    }
    failures: list[str] = []
    for result in results:
        for horizon, value in result.position_rmse_m.items():
            if value > horizon_limits[horizon]:
                failures.append(f"{result.name}:position_rmse@{horizon}s")
        if result.collision_f1 < 0.90:
            failures.append(f"{result.name}:collision_f1")
        if result.collision_timing_error_frames is None or result.collision_timing_error_frames > 1:
            failures.append(f"{result.name}:collision_timing")
        if result.known_action_count != result.expected_action_count:
            failures.append(f"{result.name}:known_action_count")
        if not result.source_unchanged:
            failures.append(f"{result.name}:source_mutation")
    return tuple(failures)


def _aggregate_horizon(
    results: Sequence[LongHorizonScenarioResult],
    field: str,
) -> dict[str, float]:
    output: dict[str, list[float]] = {}
    for result in results:
        values = getattr(result, field)
        for horizon, value in values.items():
            output.setdefault(horizon, []).append(value)
    return {
        horizon: math.sqrt(math.fsum(value * value for value in values) / len(values))
        for horizon, values in sorted(output.items(), key=lambda item: float(item[0]))
    }


def _summary(
    *,
    run_id: str,
    results: Sequence[LongHorizonScenarioResult],
    planning_results: Sequence[ActionSequencePlanningResult],
    n8_probe: PerceptualScaleProbeResult,
    failures: Sequence[str],
    provenance: Mapping[str, Any],
    created_at_utc: str,
    artifact_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    position_curve = _aggregate_horizon(results, "position_rmse_m")
    velocity_curve = _aggregate_horizon(results, "velocity_rmse_mps")
    endpoint = position_curve.get("8", position_curve.get("4", 0.0))
    coverage = [result.uncertainty_90_coverage for result in results]
    planning_by_k: dict[str, dict[str, Any]] = {}
    for candidate_count in (8, 32):
        selected = [item for item in planning_results if item.candidate_count == candidate_count]
        if selected:
            planning_by_k[str(candidate_count)] = {
                "winner_accuracy": sum(item.winner_correct for item in selected) / len(selected),
                "median_normalized_regret": median(item.normalized_regret for item in selected),
                "goal_success": sum(item.terminal_goal_success for item in selected)
                / len(selected),
                "replanning_consistency": sum(item.replanning_consistent for item in selected)
                / len(selected),
            }
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=created_at_utc,
        lifecycle_status="completed" if not failures else "failed",
        outcome="long_horizon_pilot_passed" if not failures else "long_horizon_pilot_failed",
        source_format=LONG_HORIZON_REPORT_SCHEMA,
        configuration={
            "horizons_seconds": list(LONG_HORIZONS_SECONDS),
            "frame_rate": LONG_HORIZON_FRAME_RATE,
            "world_bounds": [list(axis) for axis in LONG_HORIZON_BOUNDS],
            "state_first": True,
        },
        provenance=dict(provenance),
        scores={"candidate": endpoint, "incumbent": endpoint, "selected": "incumbent"},
        factor_metrics={
            **{
                factor: {"status": "unmeasured", "reason": "separate long-horizon pilot"}
                for factor in ALL_CAPABILITY_FACTORS
            },
            "long_horizon_causal": {
                "status": "passed" if not failures else "failed",
                "scenario_count": len(results),
            },
            "n8_perception_development": {
                "status": "measured",
                "exact_count": n8_probe.exact_count,
                "position_rmse_m": n8_probe.position_rmse_m,
                "full_perceptual_qualification": False,
            },
        },
        cell_metrics={},
        horizon_curves={
            "candidate_position_rmse_m": position_curve,
            "candidate_velocity_rmse_mps": velocity_curve,
        },
        uncertainty={
            "coverage_90_range": [min(coverage), max(coverage)] if coverage else [],
        },
        planning={
            "status": "measured",
            "by_candidate_count": planning_by_k,
            "tasks": [item.to_dict() for item in planning_results],
            "serial_vectorized_winner_parity": all(
                item.serial_vectorized_winner_parity for item in planning_results
            ),
            "maximum_cost_difference": max(
                (item.maximum_cost_difference for item in planning_results),
                default=None,
            ),
        },
        resources={
            "eight_second_rollout_seconds": max(
                (item.rollout_latency_seconds for item in results if item.duration_s == 8.0),
                default=None,
            ),
            "scalability": {
                "perceptual_development_probes": [n8_probe.to_dict()],
                "full_perceptual_qualification": False,
            },
        },
        artifacts={"run_bytes": artifact_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "incumbent",
            "promotion_evaluated": False,
            "gate_failures": list(failures),
        },
        failure_attribution={
            "primary_bottleneck": failures[0] if failures else "mixed rigid geometry unmeasured",
            "ablation_owner": "state dynamics" if failures else "next: box geometry seam",
        },
        qualitative={
            "best_episode": results[0].name if results else "unavailable",
            "representative_episode": results[len(results) // 2].name if results else "unavailable",
            "worst_episode": results[-1].name if results else "unavailable",
            "forecast_animations": [item.animation for item in results[:3]],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "long-horizon RGB-D perceptual qualification",
            "perception above six objects",
            "non-spherical rigid-body qualification",
        ),
        scope_limitations=(
            "state-first long-horizon anchor",
            "known calibrated actions only",
            "fixed membership during candidate rollout",
            "axis-aligned world boundaries",
        ),
    ).validate()


def _publish_terminal_evidence(
    summary: CapabilityRunSummary,
    output: Path,
    *,
    failed: bool,
) -> CapabilityRunSummary:
    """Write summary/report/manifest until the recorded byte count is exact."""

    current = summary
    for _ in range(8):
        write_capability_summary(current, output / "capability_summary.json")
        write_run_report(current, output)
        write_run_manifest(
            output,
            role="candidate",
            status="failed" if failed else "completed",
            artifacts={
                "capability_summary.json": "summary",
                "long_horizon_report.json": "summary",
                "report.html": "report",
            },
        )
        actual_bytes = sum(
            path.stat().st_size
            for path in output.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        if current.artifacts.get("run_bytes") == actual_bytes:
            return current
        current = replace(
            current,
            artifacts={**current.artifacts, "run_bytes": actual_bytes},
        )
    raise RuntimeError("terminal evidence byte count did not converge")


def run_long_horizon_evaluation(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    run_directory: str | Path,
    seed: int = 0,
    threads: int = 1,
    progress: Any = print,
) -> dict[str, Any]:
    """Run the bounded pilot and atomically publish compact evidence."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    output = Path(run_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(model_config_path)
    model, checkpoint = _model_from_workbench_checkpoint(config, checkpoint_path)
    if progress is not None:
        progress("perception scaling: separated N=8 development probe")
    n8_probe = run_n8_perception_probe(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
    )
    scenarios = default_long_horizon_scenarios()
    results: list[LongHorizonScenarioResult] = []
    for index, scenario in enumerate(scenarios, start=1):
        if progress is not None:
            progress(f"long horizon: {index}/{len(scenarios)} {scenario.name}")
        results.append(evaluate_long_horizon_scenario(model.dynamics, scenario))
    planning_results: list[ActionSequencePlanningResult] = []
    for scenario in scenarios[:2]:
        for candidate_count in (8, 32):
            if progress is not None:
                progress(f"sequence planning: {scenario.name} K={candidate_count}")
            planning_results.append(
                evaluate_action_sequence_planning(
                    model.dynamics,
                    scenario,
                    candidate_count=candidate_count,
                )
            )
    failures = (
        *long_horizon_gate_failures(results),
        *action_sequence_gate_failures(planning_results),
    )
    created = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": LONG_HORIZON_REPORT_SCHEMA,
        "created_at_utc": created,
        "status": "passed" if not failures else "failed",
        "gate_failures": list(failures),
        "scenario_manifest_sha256": long_horizon_scenario_sha256(scenarios),
        "horizons_seconds": list(LONG_HORIZONS_SECONDS),
        "scenarios": [item.to_dict() for item in results],
        "planning_tasks": [item.to_dict() for item in planning_results],
        "n8_perception_development_probe": n8_probe.to_dict(),
        "artifact_policy": {
            "generated_episodes_retained": False,
            "raw_frames_retained": False,
            "animation_media_retained": False,
            "vector_animation_limit_bytes": LONG_HORIZON_ANIMATION_MAX_BYTES,
        },
    }
    atomic_write_text(
        output / "long_horizon_report.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_root = output.parent.parent / ".archive"
    archive_bytes = inventory_runs(output.parent, archive_root=archive_root)["archive_bytes"]
    provenance = {
        **checkpoint,
        "public_development_only": True,
        "state_first_probe": True,
        "truth_runtime_input_count": 0,
        "planning_used_as_training_loss": False,
        "scenario_manifest_sha256": report["scenario_manifest_sha256"],
    }
    summary = _summary(
        run_id=output.name,
        results=results,
        planning_results=planning_results,
        n8_probe=n8_probe,
        failures=failures,
        provenance=provenance,
        created_at_utc=created,
        artifact_bytes=(output / "long_horizon_report.json").stat().st_size,
        archive_bytes=archive_bytes,
    )
    summary = _publish_terminal_evidence(
        summary,
        output,
        failed=bool(failures),
    )
    cleanup = enforce_run_budget(output.parent, archive_root=archive_root)
    build_progress_dashboard(output.parent, archive_root=archive_root)
    return {
        **report,
        "run_directory": str(output),
        "cleanup": cleanup.to_dict(),
    }


def default_long_horizon_run_directory() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-long-horizon-pilot"


__all__ = [
    "LONG_HORIZON_ANIMATION_MAX_BYTES",
    "LONG_HORIZON_BOUNDS",
    "LONG_HORIZON_FRAME_RATE",
    "LONG_HORIZON_REPORT_SCHEMA",
    "LONG_HORIZONS_SECONDS",
    "LongHorizonActionSpec",
    "LongHorizonScenario",
    "LongHorizonScenarioResult",
    "ActionSequencePlanningResult",
    "action_sequence_gate_failures",
    "default_long_horizon_run_directory",
    "default_long_horizon_scenarios",
    "evaluate_long_horizon_scenario",
    "evaluate_action_sequence_planning",
    "long_horizon_gate_failures",
    "long_horizon_scenario_sha256",
    "run_long_horizon_evaluation",
]
