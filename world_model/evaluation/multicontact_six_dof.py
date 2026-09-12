"""Compact N=4/6/8 multi-contact six-DoF dynamics and planning scale gate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from world_model.belief import (
    BeliefFactory,
    MotionMode,
    RigidGeometryCodec,
    RigidPrimitive,
    WorldBelief,
)
from world_model.dynamics import (
    DynamicsModel,
    WorldImpulseAction,
    WorldImpulseSchedule,
    quaternion_geodesic_distance,
)
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.planning import TerminalWorldPositionGoal, plan_counterfactual_actions
from world_model.simulator import (
    PhysicsConfig,
    RigidBodyState,
    SphereState,
    advance_rigid_bodies_6dof,
)
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

MULTICONTACT_SIX_DOF_SCHEMA = "world_model_multicontact_six_dof_scale_v1"
_DTYPE = torch.float64
_FRAME_DT = 0.05
_HORIZON_SECONDS = 2.0
_BOUNDS = ((-4.0, 4.0), (-3.0, 3.0), (2.0, 6.0))
_COUNTS = (4, 6, 8)


@dataclass(frozen=True, slots=True)
class MultiContactActionSpec:
    offset_seconds: float
    target_slot: int
    impulse_world: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MultiContactScenarioResult:
    object_count: int
    known_action_count: int
    expected_action_count: int
    unique_reference_contact_pairs: int
    unique_model_contact_pairs: int
    contact_pair_f1: float
    first_contact_timing_error_frames: int | None
    repeated_contact_frame_f1: float
    simultaneous_contact_frames: int
    maximum_position_rmse_m: float
    endpoint_position_rmse_m: float
    maximum_velocity_rmse_mps: float
    maximum_orientation_rmse_degrees: float
    rollout_latency_seconds: float
    finite: bool
    source_unchanged: bool
    position_curve: dict[str, float]
    animation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MultiContactPlanningResult:
    object_count: int
    candidate_count: int
    oracle_winner: int
    selected_winner: int
    winner_correct: bool
    normalized_regret: float
    normalized_winner_margin: float
    goal_success: bool
    serial_vectorized_parity: bool
    maximum_cost_difference: float
    vectorized_latency_seconds: float
    serial_latency_seconds: float
    vectorization_speedup: float
    source_unchanged: bool


@dataclass(frozen=True, slots=True)
class MultiContactScaleResult:
    schema: str
    manifest_sha256: str
    scenarios: tuple[MultiContactScenarioResult, ...]
    planning: tuple[MultiContactPlanningResult, ...]
    learned_weight_bytes: int
    evaluation_seconds: float
    gate_failures: tuple[str, ...]
    qualified: bool
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_actions(object_count: int) -> tuple[MultiContactActionSpec, ...]:
    """Return the deterministic three-intervention contact-wave schedule."""

    if object_count not in _COUNTS:
        raise ValueError(f"object_count must be one of {_COUNTS}")
    return (
        MultiContactActionSpec(0.10, 0, (1.35, 0.20, 0.0)),
        MultiContactActionSpec(0.75, object_count - 1, (-1.10, -0.12, 0.0)),
        MultiContactActionSpec(1.30, 0, (0.55, -0.08, 0.0)),
    )


def multicontact_manifest_sha256() -> str:
    payload = {
        "object_counts": list(_COUNTS),
        "primitives": "alternating oriented boxes and spheres",
        "horizon_seconds": _HORIZON_SECONDS,
        "frame_dt_seconds": _FRAME_DT,
        "actions": {
            str(count): [asdict(item) for item in default_actions(count)] for count in _COUNTS
        },
        "required_contact_behavior": [
            "all adjacent pairs",
            "simultaneous contacts",
            "repeated contacts",
        ],
        "planning": {
            "object_count": 8,
            "candidate_counts": [8, 32],
            "serial_oracle": True,
            "planning_loss": False,
        },
        "acceptance": {
            "maximum_position_rmse_m": 0.015,
            "maximum_velocity_rmse_mps": 0.060,
            "maximum_orientation_rmse_degrees": 3.5,
            "contact_pair_f1": 1.0,
            "minimum_repeated_contact_frame_f1": 0.65,
            "maximum_first_contact_timing_error_frames": 1,
            "minimum_vectorization_speedup": 5.0,
            "maximum_vectorized_planning_latency_seconds": 6.5,
            "rollout_latency_seconds": {"N4": 12.0, "N6": 22.0, "N8": 36.0},
        },
        "evidence": "state-first; no perceptual qualification claim",
        "retained_media": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _state(object_count: int) -> RigidBodyState:
    x = (torch.arange(object_count, dtype=_DTYPE) - (object_count - 1) / 2.0) * 0.46
    y = torch.tensor(
        [0.045 if index % 2 == 0 else -0.045 for index in range(object_count)],
        dtype=_DTYPE,
    )
    position = torch.stack((x, y, torch.full_like(x, 4.0)), dim=-1)
    primitive = torch.tensor(
        [
            int(RigidPrimitive.BOX) if index % 2 == 0 else int(RigidPrimitive.SPHERE)
            for index in range(object_count)
        ],
        dtype=torch.int64,
    )
    half_extents = torch.tensor(
        [
            [0.20, 0.17, 0.15] if index % 2 == 0 else [0.19, 0.19, 0.19]
            for index in range(object_count)
        ],
        dtype=_DTYPE,
    )
    radius = torch.where(
        (primitive == int(RigidPrimitive.BOX)).unsqueeze(-1),
        torch.linalg.vector_norm(half_extents, dim=-1, keepdim=True),
        torch.full((object_count, 1), 0.19, dtype=_DTYPE),
    )
    angle = torch.tensor(
        [0.12 * (-1.0 if index % 3 == 0 else 1.0) for index in range(object_count)],
        dtype=_DTYPE,
    )
    orientation = torch.stack(
        (
            torch.zeros_like(angle),
            torch.zeros_like(angle),
            torch.sin(0.5 * angle),
            torch.cos(0.5 * angle),
        ),
        dim=-1,
    )
    sphere_state = SphereState(
        object_id=torch.arange(1000, 1000 + object_count, dtype=torch.int64),
        active=torch.ones(object_count, dtype=torch.bool),
        position=position,
        velocity=torch.zeros(object_count, 3, dtype=_DTYPE),
        radius=radius,
        mass=torch.ones(object_count, 1, dtype=_DTYPE),
        restitution=torch.full((object_count, 1), 0.55, dtype=_DTYPE),
        drag=torch.full((object_count, 1), 0.002, dtype=_DTYPE),
        friction=torch.full((object_count, 1), 0.18, dtype=_DTYPE),
        albedo=torch.linspace(0.2, 0.8, object_count * 3, dtype=_DTYPE).reshape(object_count, 3),
        orientation=orientation,
        angular_velocity=torch.zeros(object_count, 3, dtype=_DTYPE),
        sleeping=torch.zeros(object_count, dtype=torch.bool),
        sleep_counter=torch.zeros(object_count, dtype=torch.int64),
    )
    return replace(
        RigidBodyState.from_spheres(sphere_state),
        primitive=primitive,
        half_extents=half_extents,
    )


def _belief_from_state(state: RigidBodyState) -> WorldBelief:
    belief = BeliefFactory(
        max_objects=state.max_objects,
        geometry_dim=5,
        appearance_dim=8,
        residual_dynamics_dim=1,
        modal_count=0,
        modal_dim=1,
        parameter_memory_dim=1,
        global_code_dim=1,
    ).create(batch_size=1, dtype=state.position.dtype, gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[0] = state.active
    objects.object_id[0] = state.object_id
    objects.existence_logit.fill_(12.0)
    objects.position[0] = state.position
    objects.velocity[0] = state.velocity
    objects.orientation[0] = state.orientation
    objects.angular_velocity[0] = state.angular_velocity
    for index in range(state.max_objects):
        if int(state.primitive[index]) == int(RigidPrimitive.BOX):
            objects.geometry[0, index] = RigidGeometryCodec.encode_box(
                state.half_extents[index], geometry_dim=5
            )
        else:
            objects.geometry[0, index] = RigidGeometryCodec.encode_sphere(
                state.radius[index], geometry_dim=5
            )
    objects.log_mass[0] = state.mass.log()
    objects.restitution_logit[0] = torch.logit(state.restitution)
    objects.log_drag[0] = state.drag.log()
    objects.friction_logit[0] = torch.logit(state.friction)
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., int(MotionMode.FREE)] = 4.0
    objects.fast_log_variance.fill_(-20.0)
    objects.slow_log_variance.fill_(-20.0)
    return replace(
        belief,
        objects=objects,
        next_object_id=torch.tensor([1000 + state.max_objects], dtype=torch.int64),
        metadata={
            "state_first_probe": True,
            "belief_initialized_from_state_oracle": True,
            "state_oracle_withheld_after_initialization": True,
        },
    ).validate()


def _dynamics(belief: WorldBelief) -> DynamicsModel:
    model = DynamicsModel.from_belief(
        belief,
        max_substep=1.0 / 120.0,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=16,
        interaction_radius=2.5,
        world_bounds=_BOUNDS,
        solver_iterations=4,
        modal_dynamics_enabled=False,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        event_driven_state_only_enabled=False,
        rigid_six_dof_contacts_enabled=True,
        contact_confidence_sigma=0.0,
    ).to(dtype=belief.dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model.eval()


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=_BOUNDS,
        max_substep=1.0 / 120.0,
        solver_iterations=4,
    )


def _schedule(belief: WorldBelief, object_count: int) -> WorldImpulseSchedule:
    return WorldImpulseSchedule(
        tuple(
            WorldImpulseAction(
                timestamp=belief.timestamp.new_tensor([spec.offset_seconds]),
                object_id=torch.tensor([1000 + spec.target_slot], dtype=torch.int64),
                impulse_world=belief.timestamp.new_tensor([spec.impulse_world]),
            )
            for spec in default_actions(object_count)
        )
    )


def _apply_reference_impulse(state: RigidBodyState, spec: MultiContactActionSpec) -> RigidBodyState:
    velocity = state.velocity.clone()
    velocity[spec.target_slot] += (
        state.velocity.new_tensor(spec.impulse_world) / state.mass[spec.target_slot, 0]
    )
    return replace(state, velocity=velocity)


def _reference_rollout(
    state: RigidBodyState,
    query_times: Tensor,
    actions: tuple[MultiContactActionSpec, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    current_time = 0.0
    action_index = 0
    positions: list[Tensor] = []
    velocities: list[Tensor] = []
    orientations: list[Tensor] = []
    collisions: list[Tensor] = []
    for endpoint_tensor in query_times:
        endpoint = float(endpoint_tensor)
        interval_collision = torch.zeros(
            state.max_objects,
            state.max_objects,
            dtype=torch.bool,
        )
        while action_index < len(actions) and actions[action_index].offset_seconds <= endpoint:
            action = actions[action_index]
            if action.offset_seconds > current_time:
                state, events = advance_rigid_bodies_6dof(
                    state, action.offset_seconds - current_time, _physics()
                )
                interval_collision |= events.pair_collision
            state = _apply_reference_impulse(state, action)
            current_time = action.offset_seconds
            action_index += 1
        if endpoint > current_time:
            state, events = advance_rigid_bodies_6dof(state, endpoint - current_time, _physics())
            interval_collision |= events.pair_collision
        current_time = endpoint
        positions.append(state.position)
        velocities.append(state.velocity)
        orientations.append(state.orientation)
        collisions.append(interval_collision)
    return (
        torch.stack(positions),
        torch.stack(velocities),
        torch.stack(orientations),
        torch.stack(collisions),
    )


def _f1(predicted: Tensor, reference: Tensor) -> float:
    true_positive = int((predicted & reference).sum())
    false_positive = int((predicted & ~reference).sum())
    false_negative = int((~predicted & reference).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


def _contact_metrics(
    predicted: Tensor,
    reference: Tensor,
) -> tuple[int, int, float, int | None, float, int]:
    predicted = predicted.triu(diagonal=1)
    reference = reference.triu(diagonal=1)
    predicted_pairs = predicted.any(dim=0)
    reference_pairs = reference.any(dim=0)
    pair_f1 = _f1(predicted_pairs, reference_pairs)
    timing_errors: list[int] = []
    for first, second in torch.nonzero(reference_pairs, as_tuple=False).tolist():
        predicted_frames = torch.where(predicted[:, first, second])[0]
        reference_frames = torch.where(reference[:, first, second])[0]
        if len(predicted_frames):
            timing_errors.append(abs(int(predicted_frames[0]) - int(reference_frames[0])))
    first_timing = max(timing_errors) if len(timing_errors) == int(reference_pairs.sum()) else None
    simultaneous_frames = int((reference.sum(dim=(-2, -1)) >= 2).sum())
    return (
        int(reference_pairs.sum()),
        int(predicted_pairs.sum()),
        pair_f1,
        first_timing,
        _f1(predicted, reference),
        simultaneous_frames,
    )


def _animation(
    object_count: int,
    query_times: Tensor,
    model_position: Tensor,
    reference_position: Tensor,
    model_orientation: Tensor,
    reference_orientation: Tensor,
    reference_collision: Tensor,
    position_error: Tensor,
) -> dict[str, Any]:
    action_frames = {
        max(0, round(spec.offset_seconds / _FRAME_DT) - 1): spec
        for spec in default_actions(object_count)
    }
    contact_frames = set(torch.where(reference_collision.any(dim=(-2, -1)))[0].tolist())
    indices = sorted(set(range(0, len(query_times), 2)) | contact_frames | set(action_frames))
    if indices[-1] != len(query_times) - 1:
        indices.append(len(query_times) - 1)
    frames = []
    for index in indices:
        truth_points = []
        model_points = []
        for slot, point in enumerate(reference_position[index]):
            truth_point = [1000 + slot, round(float(point[0]), 5), round(float(point[1]), 5)]
            model_point = [
                1000 + slot,
                round(float(model_position[index, slot, 0]), 5),
                round(float(model_position[index, slot, 1]), 5),
            ]
            # Alternating even slots are oriented boxes. A projected yaw spoke
            # makes rotational accuracy visible without retaining rendered frames.
            if slot % 2 == 0:
                for rendered, quaternion in (
                    (truth_point, reference_orientation[index, slot]),
                    (model_point, model_orientation[index, slot]),
                ):
                    x, y, z, w = (float(value) for value in quaternion)
                    yaw = math.atan2(
                        2.0 * (w * z + x * y),
                        1.0 - 2.0 * (y * y + z * z),
                    )
                    rendered.append(round(yaw, 5))
            truth_points.append(truth_point)
            model_points.append(model_point)
        contact_points = []
        for first, second in torch.nonzero(
            reference_collision[index].triu(diagonal=1), as_tuple=False
        ).tolist():
            midpoint = 0.5 * (reference_position[index, first] + reference_position[index, second])
            contact_points.append([round(float(midpoint[0]), 5), round(float(midpoint[1]), 5)])
        frame: dict[str, Any] = {
            "frame": index,
            "time_s": round(float(query_times[index]), 3),
            "truth": truth_points,
            "model": model_points,
        }
        if contact_points:
            frame["contacts"] = contact_points
        frames.append(frame)
    all_x = [point[1] for frame in frames for role in ("truth", "model") for point in frame[role]]
    all_y = [point[2] for frame in frames for role in ("truth", "model") for point in frame[role]]

    def bounds(values: list[float]) -> list[float]:
        padding = max(0.08 * (max(values) - min(values)), 0.10)
        return [round(min(values) - padding, 4), round(max(values) + padding, 4)]

    events = [
        {
            "frame": frame,
            "time_s": spec.offset_seconds,
            "kind": f"known action on object {1000 + spec.target_slot}",
        }
        for frame, spec in action_frames.items()
    ]
    for index in sorted(contact_frames):
        pair_count = int(reference_collision[index].triu(diagonal=1).sum())
        events.append(
            {
                "frame": index,
                "time_s": round(float(query_times[index]), 3),
                "kind": "simultaneous contacts" if pair_count >= 2 else "contact",
            }
        )
    return {
        "schema": "world_model_compact_multicontact_animation_v1",
        "label": f"N={object_count} chained and simultaneous six-DoF contacts",
        "episode": f"state-first-multicontact-n{object_count}",
        "object_count": object_count,
        "contact": True,
        "dynamic_membership": False,
        "endpoint_position_rmse_m": float(position_error[-1]),
        "long_horizon_endpoint_s": _HORIZON_SECONDS,
        "mode": "forecast",
        "known_actions_in_rollout": True,
        "anchor_frame": 0,
        "frame_rate": 1.0 / _FRAME_DT,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "orientation_markers": True,
        "bounds": {"horizontal": bounds(all_x), "vertical": bounds(all_y)},
        "frames": frames,
        "events": events,
        "reference": "independent six-DoF state oracle; this slice is not RGB-D qualification",
    }


def _evaluate_scenario(object_count: int) -> tuple[MultiContactScenarioResult, DynamicsModel]:
    state = _state(object_count)
    belief = _belief_from_state(state)
    dynamics = _dynamics(belief)
    source = belief.clone()
    query_times = torch.arange(
        _FRAME_DT,
        _HORIZON_SECONDS + 0.5 * _FRAME_DT,
        _FRAME_DT,
        dtype=belief.dtype,
    )
    started = time.perf_counter()
    with torch.no_grad():
        prediction = dynamics.rollout(
            belief,
            query_times,
            action=_schedule(belief, object_count),
        )
    latency = time.perf_counter() - started
    reference_position, reference_velocity, reference_orientation, reference_collision = (
        _reference_rollout(state, query_times, default_actions(object_count))
    )
    model_position = prediction.positions[0]
    model_velocity = prediction.velocities[0]
    position_error = (model_position - reference_position).square().mean(dim=(-2, -1)).sqrt()
    velocity_error = (model_velocity - reference_velocity).square().mean(dim=(-2, -1)).sqrt()
    orientation_error = quaternion_geodesic_distance(
        prediction.orientations[0], reference_orientation
    ).square().mean(dim=-1).sqrt() * (180.0 / math.pi)
    predicted_collision = prediction.auxiliary["pair_collision"][0]
    (
        reference_pairs,
        predicted_pairs,
        pair_f1,
        first_timing,
        repeated_f1,
        simultaneous_frames,
    ) = _contact_metrics(predicted_collision, reference_collision)
    known_count_tensor = prediction.auxiliary.get("known_action_count")
    known_count = (
        int(known_count_tensor.sum())
        if known_count_tensor is not None
        else int(prediction.auxiliary["known_action_applied"].sum())
    )
    finite = bool(
        torch.isfinite(model_position).all()
        and torch.isfinite(model_velocity).all()
        and torch.isfinite(prediction.orientations).all()
    )
    curve = {
        f"{float(timestamp):.2f}": float(position_error[index])
        for index, timestamp in enumerate(query_times)
    }
    return (
        MultiContactScenarioResult(
            object_count=object_count,
            known_action_count=known_count,
            expected_action_count=len(default_actions(object_count)),
            unique_reference_contact_pairs=reference_pairs,
            unique_model_contact_pairs=predicted_pairs,
            contact_pair_f1=pair_f1,
            first_contact_timing_error_frames=first_timing,
            repeated_contact_frame_f1=repeated_f1,
            simultaneous_contact_frames=simultaneous_frames,
            maximum_position_rmse_m=float(position_error.max()),
            endpoint_position_rmse_m=float(position_error[-1]),
            maximum_velocity_rmse_mps=float(velocity_error.max()),
            maximum_orientation_rmse_degrees=float(orientation_error.max()),
            rollout_latency_seconds=latency,
            finite=finite,
            source_unchanged=(
                torch.equal(source.objects.position, belief.objects.position)
                and torch.equal(source.objects.velocity, belief.objects.velocity)
                and torch.equal(source.objects.orientation, belief.objects.orientation)
            ),
            position_curve=curve,
            animation=_animation(
                object_count,
                query_times,
                model_position,
                reference_position,
                prediction.orientations[0],
                reference_orientation,
                reference_collision,
                position_error,
            ),
        ),
        dynamics,
    )


def _planning_impulses(belief: WorldBelief, candidate_count: int) -> tuple[Tensor, ...]:
    impulses = [belief.timestamp.new_tensor([-0.80, 0.18, 0.0])]
    for index in range(1, candidate_count):
        angle = 2.0 * math.pi * (index - 1) / max(candidate_count - 1, 1)
        impulses.append(
            belief.timestamp.new_tensor(
                [-0.20 + 0.12 * math.cos(angle), 0.12 * math.sin(angle), 0.0]
            )
        )
    return tuple(impulses)


def _reference_planning_branch(
    state: RigidBodyState,
    impulse: Tensor,
    *,
    action_time: float,
    horizon: float,
) -> RigidBodyState:
    state, _ = advance_rigid_bodies_6dof(state, action_time, _physics())
    velocity = state.velocity.clone()
    velocity[-1] += impulse.to(velocity) / state.mass[-1, 0]
    state = replace(state, velocity=velocity)
    state, _ = advance_rigid_bodies_6dof(state, horizon - action_time, _physics())
    return state


def _evaluate_planning(
    candidate_count: int,
    dynamics: DynamicsModel,
) -> MultiContactPlanningResult:
    object_count = 8
    state = _state(object_count)
    belief = _belief_from_state(state)
    impulses = _planning_impulses(belief, candidate_count)
    action_time = 0.025
    horizon = 0.25
    references = tuple(
        _reference_planning_branch(
            state,
            impulse,
            action_time=action_time,
            horizon=horizon,
        )
        for impulse in impulses
    )
    target = references[0].position[-1]
    truth_cost = torch.stack([(item.position[-1] - target).square().sum() for item in references])
    sorted_cost = truth_cost.sort().values
    cost_scale = max(float(truth_cost.max()), 1.0e-12)
    winner_margin = float(sorted_cost[1] - sorted_cost[0]) / cost_scale
    actions = tuple(
        WorldImpulseAction(
            timestamp=belief.timestamp.new_tensor([action_time]),
            object_id=torch.tensor([1000 + object_count - 1], dtype=torch.int64),
            impulse_world=impulse.reshape(1, 3),
        )
        for impulse in impulses
    )
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([1000 + object_count - 1], dtype=torch.int64),
        position_world=target.reshape(1, 3),
    )
    source = belief.clone()
    with torch.no_grad():
        vectorized_latencies = []
        vectorized = None
        for _ in range(3):
            started = time.perf_counter()
            vectorized = plan_counterfactual_actions(
                dynamics,
                belief,
                [horizon],
                actions,
                goal,
                return_events=False,
                return_auxiliary=False,
            )
            vectorized_latencies.append(time.perf_counter() - started)
        assert vectorized is not None
        started = time.perf_counter()
        serial = plan_counterfactual_actions(
            dynamics,
            belief,
            [horizon],
            actions,
            goal,
            candidate_vectorized=False,
            return_events=False,
            return_auxiliary=False,
        )
        serial_latency = time.perf_counter() - started
    vectorized_latency = sorted(vectorized_latencies)[1]
    selected = int(vectorized.selected_index[0])
    oracle = int(truth_cost.argmin())
    selected_error = float(torch.linalg.vector_norm(references[selected].position[-1] - target))
    return MultiContactPlanningResult(
        object_count=object_count,
        candidate_count=candidate_count,
        oracle_winner=oracle,
        selected_winner=selected,
        winner_correct=selected == oracle,
        normalized_regret=float(truth_cost[selected] - truth_cost.min()) / cost_scale,
        normalized_winner_margin=winner_margin,
        goal_success=selected_error <= 0.05,
        serial_vectorized_parity=torch.equal(vectorized.selected_index, serial.selected_index),
        maximum_cost_difference=float((vectorized.total_cost - serial.total_cost).abs().max()),
        vectorized_latency_seconds=vectorized_latency,
        serial_latency_seconds=serial_latency,
        vectorization_speedup=serial_latency / max(vectorized_latency, 1.0e-12),
        source_unchanged=(
            torch.equal(source.objects.position, belief.objects.position)
            and torch.equal(source.objects.orientation, belief.objects.orientation)
        ),
    )


def _gate_failures(
    scenarios: tuple[MultiContactScenarioResult, ...],
    planning: tuple[MultiContactPlanningResult, ...],
    *,
    enforce_latency: bool,
) -> tuple[str, ...]:
    failures: list[str] = []
    latency_limits = {4: 12.0, 6: 22.0, 8: 36.0}
    for item in scenarios:
        if item.known_action_count != item.expected_action_count:
            failures.append(f"n{item.object_count}:known_action_count")
        if item.unique_reference_contact_pairs < item.object_count - 1:
            failures.append(f"n{item.object_count}:contact_chain_support")
        if item.contact_pair_f1 < 1.0:
            failures.append(f"n{item.object_count}:contact_pair_f1")
        if item.first_contact_timing_error_frames is None or (
            item.first_contact_timing_error_frames > 1
        ):
            failures.append(f"n{item.object_count}:first_contact_timing")
        if item.repeated_contact_frame_f1 < 0.65:
            failures.append(f"n{item.object_count}:repeated_contact_f1")
        if item.simultaneous_contact_frames < 1:
            failures.append(f"n{item.object_count}:simultaneous_contacts")
        if item.maximum_position_rmse_m > 0.015:
            failures.append(f"n{item.object_count}:position_rmse")
        if item.maximum_velocity_rmse_mps > 0.060:
            failures.append(f"n{item.object_count}:velocity_rmse")
        if item.maximum_orientation_rmse_degrees > 3.5:
            failures.append(f"n{item.object_count}:orientation_rmse")
        if not item.finite or not item.source_unchanged:
            failures.append(f"n{item.object_count}:invariants")
        if enforce_latency and item.rollout_latency_seconds > latency_limits[item.object_count]:
            failures.append(f"n{item.object_count}:rollout_latency")
    for item in planning:
        if not (
            item.winner_correct
            and item.normalized_regret == 0.0
            and item.normalized_winner_margin >= 0.05
            and item.goal_success
            and item.serial_vectorized_parity
            and item.maximum_cost_difference <= 1.0e-6
            and item.vectorization_speedup >= 5.0
            and item.source_unchanged
        ):
            failures.append(f"planning_k{item.candidate_count}")
        if enforce_latency and item.vectorized_latency_seconds > 6.5:
            failures.append(f"planning_k{item.candidate_count}:latency")
    return tuple(failures)


def run_multicontact_six_dof_scale(
    *,
    enforce_latency: bool = True,
) -> MultiContactScaleResult:
    """Run the controlled multi-contact scale gate and downstream planning.

    Absolute CPU latency is authoritative only in the fresh-process runner;
    in-process tests may disable latency adjudication while retaining numerical
    accuracy, parity, and relative vectorization-speedup gates.
    """

    started = time.perf_counter()
    scenarios: list[MultiContactScenarioResult] = []
    dynamics_by_count: dict[int, DynamicsModel] = {}
    for object_count in _COUNTS:
        scenario, dynamics = _evaluate_scenario(object_count)
        scenarios.append(scenario)
        dynamics_by_count[object_count] = dynamics
    planning = tuple(
        _evaluate_planning(candidate_count, dynamics_by_count[8]) for candidate_count in (8, 32)
    )
    frozen_scenarios = tuple(scenarios)
    failures = _gate_failures(
        frozen_scenarios,
        planning,
        enforce_latency=enforce_latency,
    )
    learned_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in dynamics_by_count[8].parameters()
    )
    return MultiContactScaleResult(
        schema=MULTICONTACT_SIX_DOF_SCHEMA,
        manifest_sha256=multicontact_manifest_sha256(),
        scenarios=frozen_scenarios,
        planning=planning,
        learned_weight_bytes=learned_bytes,
        evaluation_seconds=time.perf_counter() - started,
        gate_failures=failures,
        qualified=not failures,
    )


def _summary(
    result: MultiContactScaleResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    by_count = {item.object_count: item for item in result.scenarios}
    planning_by_k = {
        str(item.candidate_count): {
            "winner_accuracy": float(item.winner_correct),
            "median_normalized_regret": item.normalized_regret,
            "goal_success": float(item.goal_success),
            "vectorized_latency_seconds": item.vectorized_latency_seconds,
            "serial_latency_seconds": item.serial_latency_seconds,
            "vectorization_speedup": item.vectorization_speedup,
            "maximum_cost_difference": item.maximum_cost_difference,
            "normalized_winner_margin": item.normalized_winner_margin,
        }
        for item in result.planning
    }
    horizon_keys = result.scenarios[0].position_curve
    aggregate_curve = {
        key: max(item.position_curve[key] for item in result.scenarios) for key in horizon_keys
    }
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.qualified else "failed",
        outcome="qualified_convergence" if result.qualified else "capability_gate_failed",
        source_format=MULTICONTACT_SIX_DOF_SCHEMA,
        configuration={
            "object_counts": list(_COUNTS),
            "state_first": True,
            "mixed_rigid_primitives": True,
            "known_action_schedule_length": 3,
            "planning_used_as_training_loss": False,
            "generated_frames_retained": False,
        },
        provenance={
            "scenario_manifest_sha256": result.manifest_sha256,
            "truth_oracle": "independent world_model.simulator.rigid_six_dof",
            "belief_initialization": "controlled state-first oracle",
            "truth_withheld_after_belief_initialization": True,
            "dense_serial_planning_oracle": True,
        },
        scores={
            "candidate": {
                "value": max(item.maximum_position_rmse_m for item in result.scenarios),
                "supported_weight": 1.0,
            },
            # This milestone accelerates the same structured dynamics; it does
            # not claim a checkpoint-quality improvement over the incumbent.
            "incumbent": {
                "value": max(item.maximum_position_rmse_m for item in result.scenarios),
                "supported_weight": 1.0,
            },
            "selected": "batched_multicontact_six_dof" if result.qualified else "incumbent",
        },
        factor_metrics={
            f"multicontact_n{count}": {
                "status": "passed" if result.qualified else "measured",
                "two_second_position_rmse_m": item.endpoint_position_rmse_m,
                "collision_f1": item.contact_pair_f1,
                "orientation_rmse_degrees": item.maximum_orientation_rmse_degrees,
                "simultaneous_contact_frames": item.simultaneous_contact_frames,
            }
            for count, item in by_count.items()
        },
        cell_metrics={
            f"N{count}/multi_contact/mixed_rigid": {
                "two_second_position_rmse_m": {
                    "value": item.endpoint_position_rmse_m,
                    "support": count,
                },
                "collision_f1": {
                    "value": item.contact_pair_f1,
                    "support": item.unique_reference_contact_pairs,
                },
            }
            for count, item in by_count.items()
        },
        horizon_curves={"candidate_position_rmse_m": aggregate_curve},
        uncertainty={"status": "not re-estimated in state-first dynamics scale gate"},
        planning={
            "status": "passed"
            if all(item.winner_correct for item in result.planning)
            else "failed",
            "goal": "N=8 multi-contact terminal position",
            "by_candidate_count": planning_by_k,
            "tasks": [asdict(item) for item in result.planning],
            "serial_vectorized_winner_parity": all(
                item.serial_vectorized_parity for item in result.planning
            ),
            "maximum_cost_difference": max(
                item.maximum_cost_difference for item in result.planning
            ),
        },
        resources={
            "evaluation_seconds": result.evaluation_seconds,
            "learned_weight_bytes": result.learned_weight_bytes,
            "n8_rollout_latency_seconds": by_count[8].rollout_latency_seconds,
            "k32_vectorization_speedup": next(
                item.vectorization_speedup for item in result.planning if item.candidate_count == 32
            ),
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "batched_multicontact_six_dof" if result.qualified else "none",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "six-DoF contact propagation and candidate-batch execution",
        },
        qualitative={
            "best_episode": "state-first-multicontact-n4",
            "representative_episode": "state-first-multicontact-n6",
            "worst_episode": "state-first-multicontact-n8",
            "forecast_gallery_mode": "latest_run",
            "animations": [],
            "forecast_animations": [item.animation for item in result.scenarios],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "RGB-D perception at the multi-contact anchor",
            "changing membership during the multi-contact forecast",
            "identity recovery through an unobserved collision or action",
            "visual qualification above eight objects",
        ),
        scope_limitations=(
            "state-first N=4/6/8 mixed sphere and oriented-box chains",
            "known actions and calibrated physical parameters",
            "two-second forecast horizon",
            "planning freezes the active set",
        ),
    ).validate()


def publish_multicontact_six_dof_scale(
    result: MultiContactScaleResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "multicontact_six_dof.json",
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
            status="completed" if result.qualified else "failed",
            artifacts={
                "multicontact_six_dof.json": "summary",
                "capability_summary.json": "summary",
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
        raise RuntimeError("multi-contact evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "MULTICONTACT_SIX_DOF_SCHEMA",
    "MultiContactActionSpec",
    "MultiContactPlanningResult",
    "MultiContactScaleResult",
    "MultiContactScenarioResult",
    "default_actions",
    "multicontact_manifest_sha256",
    "publish_multicontact_six_dof_scale",
    "run_multicontact_six_dof_scale",
]
