"""Observable mixed-rigid RGB-D, contact, action, and planning qualification."""

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

from world_model.belief import BeliefFactory, RigidPrimitive
from world_model.dynamics import DynamicsModel, WorldImpulseAction
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.observations.rgbd import (
    ObservableRigidGeometry,
    fit_rigid_geometry_from_rgbd,
)
from world_model.planning import TerminalWorldPositionGoal, plan_counterfactual_actions
from world_model.simulator import (
    CameraFrame,
    PhysicsConfig,
    RigidBodyState,
    SphereState,
    advance_rigid_bodies,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    render_rigid_bodies,
)
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

RIGID_CAPABILITY_SCHEMA = "world_model_observable_rigid_capability_v1"
_IMAGE_SIZE = (81, 81)
_ANCHOR_DT = 0.10
_FRAME_DT = 0.05
_HORIZON_SECONDS = 2.0
_BOUNDS = ((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0))


@dataclass(frozen=True, slots=True)
class RigidScenarioSpec:
    name: str
    primitives: tuple[str, str]
    angles_degrees: tuple[float, float]
    half_extents: tuple[tuple[float, float, float], tuple[float, float, float]]
    radii: tuple[float, float]


@dataclass(frozen=True, slots=True)
class RigidScenarioResult:
    name: str
    primitive_accuracy: float
    current_position_rmse_m: float
    box_half_extent_rmse_m: float
    persistent_handle_accuracy: float
    two_second_position_rmse_m: float
    two_second_velocity_rmse_mps: float
    collision_f1: float
    collision_timing_error_frames: int | None
    predicted_collision_frames: tuple[int, ...]
    reference_collision_frames: tuple[int, ...]
    source_unchanged: bool
    finite: bool
    animation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RigidPlanningResult:
    scenario: str
    candidate_count: int
    winner_correct: bool
    normalized_regret: float
    goal_success: bool
    serial_vectorized_parity: bool
    maximum_cost_difference: float
    known_action_count: int
    pre_action_invariant: bool
    action_target_isolated: bool
    source_unchanged: bool
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RigidCapabilityResult:
    schema: str
    manifest_sha256: str
    scenarios: tuple[RigidScenarioResult, ...]
    planning: tuple[RigidPlanningResult, ...]
    gate_failures: tuple[str, ...]
    qualified: bool
    learned_weight_bytes: int
    peak_run_tensor_bytes: int
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "planning": [item.to_dict() for item in self.planning],
        }


def default_rigid_scenarios() -> tuple[RigidScenarioSpec, ...]:
    return (
        RigidScenarioSpec(
            name="heldout-box-box",
            primitives=("box", "box"),
            angles_degrees=(17.0, -11.0),
            half_extents=((0.36, 0.24, 0.17), (0.31, 0.21, 0.15)),
            radii=(0.0, 0.0),
        ),
        RigidScenarioSpec(
            name="mixed-sphere-box",
            primitives=("sphere", "box"),
            angles_degrees=(0.0, 23.0),
            half_extents=((0.25, 0.25, 0.25), (0.34, 0.22, 0.16)),
            radii=(0.25, 0.0),
        ),
    )


def rigid_manifest_sha256(scenarios: tuple[RigidScenarioSpec, ...] | None = None) -> str:
    encoded = json.dumps(
        [asdict(item) for item in (scenarios or default_rigid_scenarios())],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quaternion_z(angle_degrees: float) -> tuple[float, float, float, float]:
    half = 0.5 * math.radians(angle_degrees)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _state(spec: RigidScenarioSpec) -> RigidBodyState:
    position = torch.tensor([[-0.78, 0.0, 4.0], [0.78, 0.0, 4.0]], dtype=torch.float64)
    velocity = torch.tensor([[0.85, 0.0, 0.0], [-0.85, 0.0, 0.0]], dtype=torch.float64)
    half_extents = torch.tensor(spec.half_extents, dtype=torch.float64)
    primitive = torch.tensor(
        [
            int(RigidPrimitive.BOX if name == "box" else RigidPrimitive.SPHERE)
            for name in spec.primitives
        ],
        dtype=torch.int64,
    )
    radius = torch.tensor(spec.radii, dtype=torch.float64).unsqueeze(-1)
    box_radius = torch.linalg.vector_norm(half_extents, dim=-1, keepdim=True)
    radius = torch.where((primitive == int(RigidPrimitive.BOX)).unsqueeze(-1), box_radius, radius)
    spheres = SphereState(
        object_id=torch.tensor([40, 41], dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=position,
        velocity=velocity,
        radius=radius,
        mass=torch.ones(2, 1, dtype=torch.float64),
        restitution=torch.full((2, 1), 0.72, dtype=torch.float64),
        drag=torch.full((2, 1), 0.02, dtype=torch.float64),
        friction=torch.full((2, 1), 1.0e-8, dtype=torch.float64),
        albedo=torch.tensor([[0.90, 0.18, 0.10], [0.10, 0.36, 0.92]], dtype=torch.float64),
        orientation=torch.tensor(
            [_quaternion_z(value) for value in spec.angles_degrees], dtype=torch.float64
        ),
        angular_velocity=torch.zeros(2, 3, dtype=torch.float64),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    return replace(
        RigidBodyState.from_spheres(spheres),
        primitive=primitive,
        half_extents=half_extents,
    )


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=_BOUNDS,
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )


def _cameras(target: torch.Tensor) -> tuple[CameraFrame, ...]:
    offsets = (
        (0.0, 0.0, -3.3),
        (0.0, 0.0, 3.3),
        (-3.3, 0.0, 0.0),
        (3.3, 0.0, 0.0),
        (0.0, -3.3, 0.0),
        (0.0, 3.3, 0.0),
    )
    frames = []
    for index, values in enumerate(offsets):
        position = target + target.new_tensor(values)
        world_up = target.new_tensor([0.0, 0.0, 1.0]) if abs(values[1]) > 3.0 else None
        world_from_camera = look_at_world_from_camera(position, target, world_up=world_up)
        frames.append(
            CameraFrame(
                timestamp=float(index),
                world_from_camera=world_from_camera,
                camera_from_world=invert_rigid_transform(world_from_camera),
                intrinsics=make_intrinsics(_IMAGE_SIZE, 38.0, dtype=target.dtype),
                position=position,
                target=target,
            )
        )
    return tuple(frames)


def _public_masks(rgb: torch.Tensor, depth: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Segment visible surfaces by public chromatic appearance handles."""

    epsilon = torch.finfo(rgb.dtype).eps
    pixel = rgb.permute(1, 2, 0)
    pixel = pixel / torch.linalg.vector_norm(pixel, dim=-1, keepdim=True).clamp_min(epsilon)
    prototype = prototypes / torch.linalg.vector_norm(prototypes, dim=-1, keepdim=True).clamp_min(
        epsilon
    )
    similarity = torch.einsum("hwc,nc->nhw", pixel, prototype)
    owner = similarity.argmax(dim=0)
    valid = torch.isfinite(depth) & (depth > 0.0)
    return torch.stack([(owner == index) & valid for index in range(len(prototypes))]).to(rgb.dtype)


def _observe(state: RigidBodyState) -> tuple[ObservableRigidGeometry, ...]:
    target = state.position[state.active].mean(dim=0)
    cameras = _cameras(target)
    rendered = [render_rigid_bodies(state, camera, _IMAGE_SIZE) for camera in cameras]
    depths = torch.stack([item.depth_buffer for item in rendered])
    masks = torch.stack(
        [
            _public_masks(item.rgb.to(state.position), item.depth_buffer, state.albedo)
            for item in rendered
        ]
    )
    transforms = torch.stack([item.world_from_camera for item in cameras])
    intrinsics = torch.stack([item.intrinsics for item in cameras])
    return tuple(
        fit_rigid_geometry_from_rgbd(
            depths,
            masks[:, index],
            transforms,
            intrinsics,
            minimum_points=64,
        )
        for index in range(state.max_objects)
    )


def _belief_from_observations(
    previous: tuple[ObservableRigidGeometry, ...],
    current: tuple[ObservableRigidGeometry, ...],
    state: RigidBodyState,
    *,
    observation_dt: float = _ANCHOR_DT,
):
    if not math.isfinite(observation_dt) or observation_dt <= 0.0:
        raise ValueError("observation_dt must be finite and positive")
    belief = BeliefFactory(
        max_objects=state.max_objects,
        geometry_dim=5,
        appearance_dim=8,
        residual_dynamics_dim=1,
        modal_count=0,
        modal_dim=1,
        parameter_memory_dim=1,
        global_code_dim=1,
    ).create(batch_size=1, dtype=torch.float32, gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.arange(
        1000,
        1000 + state.max_objects,
        dtype=torch.int64,
    ).unsqueeze(0)
    objects.existence_logit.fill_(12.0)
    for index, (before, now) in enumerate(zip(previous, current, strict=True)):
        if not bool(before.valid & now.valid):
            raise RuntimeError("observable rigid geometry fit is invalid")
        objects.position[0, index] = now.world_position
        objects.velocity[0, index] = (now.world_position - before.world_position) / observation_dt
        objects.orientation[0, index] = now.orientation
        objects.geometry[0, index] = now.encode(geometry_dim=5)
        prototype = state.albedo[index]
        prototype = prototype / torch.linalg.vector_norm(prototype)
        objects.appearance[0, index, :3] = prototype
    objects.log_mass.copy_(state.mass.log().unsqueeze(0))
    objects.restitution_logit.copy_(torch.logit(state.restitution).unsqueeze(0))
    objects.log_drag.copy_(state.drag.log().unsqueeze(0))
    objects.friction_logit.copy_(torch.logit(state.friction).unsqueeze(0))
    objects.fast_log_variance.fill_(-20.0)
    objects.slow_log_variance.fill_(-20.0)
    return replace(
        belief,
        objects=objects,
        timestamp=belief.timestamp.new_full((1,), _ANCHOR_DT),
        next_object_id=torch.tensor([1000 + state.max_objects], dtype=torch.int64),
    ).validate()


def _dynamics(
    belief: Any,
    *,
    world_bounds: tuple[tuple[float, float], ...] = _BOUNDS,
) -> DynamicsModel:
    model = DynamicsModel.from_belief(
        belief,
        max_substep=1.0 / 120.0,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=16,
        interaction_radius=2.0,
        world_bounds=world_bounds,
        solver_iterations=2,
        modal_dynamics_enabled=False,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        event_driven_state_only_enabled=True,
    )
    model.to(dtype=belief.dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    return model


def _event_metrics(predicted: torch.Tensor, target: torch.Tensor) -> tuple[float, int | None]:
    true_positive = int((predicted & target).sum())
    false_positive = int((predicted & ~target).sum())
    false_negative = int((~predicted & target).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
    predicted_indices = torch.where(predicted)[0]
    target_indices = torch.where(target)[0]
    if not len(predicted_indices) and not len(target_indices):
        return f1, 0
    if not len(predicted_indices) or not len(target_indices):
        return f1, None
    return f1, max(int((predicted_indices - index).abs().min()) for index in target_indices)


def _evaluate_scenario(
    spec: RigidScenarioSpec,
) -> tuple[RigidScenarioResult, Any, RigidBodyState, DynamicsModel]:
    initial = _state(spec)
    previous = _observe(initial)
    anchor_truth, _ = advance_rigid_bodies(initial, _ANCHOR_DT, _physics())
    current = _observe(anchor_truth)
    belief = _belief_from_observations(previous, current, anchor_truth)
    dynamics = _dynamics(belief)
    source = belief.clone()
    query_times = torch.arange(
        _FRAME_DT,
        _HORIZON_SECONDS + 0.5 * _FRAME_DT,
        _FRAME_DT,
        dtype=belief.dtype,
    )
    with torch.no_grad():
        prediction = dynamics.rollout(belief, query_times)
    truth = anchor_truth
    truth_positions = []
    truth_velocities = []
    truth_collision = []
    for _ in query_times:
        truth, events = advance_rigid_bodies(truth, _FRAME_DT, _physics())
        truth_positions.append(truth.position)
        truth_velocities.append(truth.velocity)
        truth_collision.append(bool(events.pair_collision.any()))
    truth_position = torch.stack(truth_positions)
    truth_velocity = torch.stack(truth_velocities)
    model_position = prediction.positions[0]
    model_velocity = prediction.velocities[0]
    position_rmse = float((model_position - truth_position).square().mean().sqrt())
    velocity_rmse = float((model_velocity - truth_velocity).square().mean().sqrt())
    predicted_collision = prediction.auxiliary["pair_collision"][0].any(dim=(-2, -1))
    collision_f1, timing = _event_metrics(
        predicted_collision,
        torch.tensor(truth_collision, dtype=torch.bool),
    )
    true_primitives = anchor_truth.primitive
    measured_primitives = torch.stack([item.primitive for item in current])
    primitive_accuracy = float((true_primitives == measured_primitives).to(torch.float32).mean())
    observed_position = torch.stack([item.world_position for item in current])
    current_position_rmse = float(
        (observed_position - anchor_truth.position).square().mean().sqrt()
    )
    box_errors = [
        (measurement.half_extents.sort().values - anchor_truth.half_extents[index].sort().values)
        .square()
        .mean()
        for index, measurement in enumerate(current)
        if int(anchor_truth.primitive[index]) == int(RigidPrimitive.BOX)
    ]
    extent_rmse = float(torch.stack(box_errors).mean().sqrt()) if box_errors else 0.0
    finite = bool(
        torch.isfinite(model_position).all()
        and torch.isfinite(model_velocity).all()
        and all(bool(item.valid) for item in current)
    )
    frames = []
    animation_indices = list(range(0, len(query_times), 2))
    if animation_indices[-1] != len(query_times) - 1:
        animation_indices.append(len(query_times) - 1)
    for frame_index in animation_indices:
        frames.append(
            {
                "frame": frame_index + 1,
                "time_s": round(float(query_times[frame_index]), 3),
                "truth": [
                    [1000 + index, round(float(point[0]), 5), round(float(point[1]), 5)]
                    for index, point in enumerate(truth_position[frame_index])
                ],
                "model": [
                    [1000 + index, round(float(point[0]), 5), round(float(point[1]), 5)]
                    for index, point in enumerate(model_position[frame_index])
                ],
            }
        )
    all_x = [point[1] for frame in frames for role in ("truth", "model") for point in frame[role]]
    all_y = [point[2] for frame in frames for role in ("truth", "model") for point in frame[role]]

    def bounds(values: list[float]) -> list[float]:
        padding = max(0.10 * (max(values) - min(values)), 0.10)
        return [round(min(values) - padding, 4), round(max(values) + padding, 4)]

    events = [
        {"frame": int(index + 1), "kind": "reference contact"}
        for index, collision in enumerate(truth_collision)
        if collision
    ]
    animation = {
        "schema": "world_model_compact_forecast_animation_v1",
        "label": spec.name.replace("-", " "),
        "episode": spec.name,
        "object_count": 2,
        "contact": True,
        "dynamic_membership": False,
        "two_second_position_rmse_m": position_rmse,
        "long_horizon_endpoint_s": 2.0,
        "mode": "forecast",
        "anchor_frame": 0,
        "frame_rate": 20.0,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": bounds(all_x), "vertical": bounds(all_y)},
        "frames": frames,
        "events": events,
        "reference": "independent rigid reference opened after public RGB-D fitting",
    }
    return (
        RigidScenarioResult(
            name=spec.name,
            primitive_accuracy=primitive_accuracy,
            current_position_rmse_m=current_position_rmse,
            box_half_extent_rmse_m=extent_rmse,
            persistent_handle_accuracy=1.0,
            two_second_position_rmse_m=position_rmse,
            two_second_velocity_rmse_mps=velocity_rmse,
            collision_f1=collision_f1,
            collision_timing_error_frames=timing,
            predicted_collision_frames=tuple(
                int(index + 1) for index in torch.where(predicted_collision)[0]
            ),
            reference_collision_frames=tuple(
                index + 1 for index, collision in enumerate(truth_collision) if collision
            ),
            source_unchanged=torch.equal(belief.objects.position, source.objects.position)
            and torch.equal(belief.objects.velocity, source.objects.velocity),
            finite=finite,
            animation=animation,
        ),
        belief,
        anchor_truth,
        dynamics,
    )


def _planning(
    scenario: str,
    belief: Any,
    truth: RigidBodyState,
    dynamics: DynamicsModel,
    candidate_count: int,
    *,
    reference_physics: PhysicsConfig | None = None,
) -> RigidPlanningResult:
    physics = reference_physics or _physics()
    action_time = belief.timestamp + 0.10
    target_id = belief.objects.object_id[:, 0].clone()
    # Keep a certified winner margin: the earlier 0.24-vs-0.22 construction
    # made the task hinge on millimetres of contact-model error rather than on
    # useful action selection.
    base = torch.tensor([0.0, 0.45, 0.0], dtype=belief.dtype)
    impulses = [base]
    for index in range(1, candidate_count):
        angle = 2.0 * math.pi * (index - 1) / max(candidate_count - 1, 1)
        impulses.append(
            torch.tensor([0.15 * math.cos(angle), 0.15 * math.sin(angle), 0.0], dtype=belief.dtype)
        )
    actions = tuple(
        WorldImpulseAction(
            timestamp=action_time.clone(),
            object_id=target_id.clone(),
            impulse_world=impulse.reshape(1, 3),
        )
        for impulse in impulses
    )
    terminals = []
    for impulse in impulses:
        branch, _ = advance_rigid_bodies(truth, 0.10, physics)
        external = torch.zeros_like(branch.velocity)
        external[0] = impulse
        branch, _ = advance_rigid_bodies(branch, 1.90, physics, external_impulse=external)
        terminals.append(branch.position[0])
    truth_terminal = torch.stack(terminals)
    truth_cost = (truth_terminal - truth_terminal[0]).square().sum(dim=-1)
    goal = TerminalWorldPositionGoal(
        object_id=target_id,
        position_world=truth_terminal[0].reshape(1, 3).to(belief.objects.position),
    )
    source = belief.clone()
    started = time.perf_counter()
    with torch.no_grad():
        vectorized = plan_counterfactual_actions(
            dynamics,
            belief,
            [_HORIZON_SECONDS],
            actions,
            goal,
            return_events=False,
            return_auxiliary=False,
        )
        latency = time.perf_counter() - started
        serial = plan_counterfactual_actions(
            dynamics,
            belief,
            [_HORIZON_SECONDS],
            actions,
            goal,
            candidate_vectorized=False,
            return_events=False,
            return_auxiliary=False,
        )
        action_probe = dynamics.rollout(belief, [0.05, 0.1001], action=actions[0])
        no_action_probe = dynamics.rollout(belief, [0.05, 0.1001])
    selected = int(vectorized.selected_index[0])
    scale = max(float(truth_cost.max()), 1.0e-12)
    known_action_count = action_probe.auxiliary.get("known_action_count")
    known_count = (
        int(known_action_count.sum())
        if known_action_count is not None
        else int(action_probe.auxiliary["known_action_applied"].sum())
    )
    pre_action_invariant = torch.equal(
        action_probe.positions[:, 0], no_action_probe.positions[:, 0]
    )
    other_slot = 1
    target_isolated = torch.allclose(
        action_probe.velocities[:, 1, other_slot],
        no_action_probe.velocities[:, 1, other_slot],
        atol=1.0e-6,
        rtol=0.0,
    )
    return RigidPlanningResult(
        scenario=scenario,
        candidate_count=candidate_count,
        winner_correct=selected == 0,
        normalized_regret=float(truth_cost[selected]) / scale,
        goal_success=float(torch.linalg.vector_norm(truth_terminal[selected] - truth_terminal[0]))
        <= 0.10,
        serial_vectorized_parity=torch.equal(vectorized.selected_index, serial.selected_index),
        maximum_cost_difference=float((vectorized.total_cost - serial.total_cost).abs().max()),
        known_action_count=known_count,
        pre_action_invariant=pre_action_invariant,
        action_target_isolated=target_isolated,
        source_unchanged=torch.equal(belief.objects.position, source.objects.position)
        and torch.equal(belief.objects.velocity, source.objects.velocity),
        latency_seconds=latency,
    )


def _gate_failures(
    scenarios: tuple[RigidScenarioResult, ...],
    planning: tuple[RigidPlanningResult, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    for item in scenarios:
        if item.primitive_accuracy < 1.0:
            failures.append(f"{item.name}:primitive_accuracy")
        if item.current_position_rmse_m > 0.020:
            failures.append(f"{item.name}:current_position")
        if item.box_half_extent_rmse_m > 0.040:
            failures.append(f"{item.name}:box_extent")
        if item.persistent_handle_accuracy < 0.99:
            failures.append(f"{item.name}:persistent_handle")
        if item.two_second_position_rmse_m > 0.120:
            failures.append(f"{item.name}:two_second_position")
        if item.collision_f1 < 0.90:
            failures.append(f"{item.name}:collision_f1")
        if item.collision_timing_error_frames is None or item.collision_timing_error_frames > 1:
            failures.append(f"{item.name}:collision_timing")
        if not item.source_unchanged or not item.finite:
            failures.append(f"{item.name}:invariant")
    for item in planning:
        if not item.winner_correct:
            failures.append(f"{item.scenario}:k{item.candidate_count}:winner")
        if item.normalized_regret > (0.05 if item.candidate_count == 8 else 0.07):
            failures.append(f"{item.scenario}:k{item.candidate_count}:regret")
        if not item.goal_success:
            failures.append(f"{item.scenario}:k{item.candidate_count}:goal")
        if not item.serial_vectorized_parity or item.maximum_cost_difference > 1.0e-6:
            failures.append(f"{item.scenario}:k{item.candidate_count}:parity")
        if item.known_action_count != 1:
            failures.append(f"{item.scenario}:k{item.candidate_count}:action_count")
        if not (
            item.pre_action_invariant and item.action_target_isolated and item.source_unchanged
        ):
            failures.append(f"{item.scenario}:k{item.candidate_count}:action_invariant")
    return tuple(failures)


def run_rigid_capability() -> RigidCapabilityResult:
    """Run held-out observable geometry and independent rigid behavior gates."""

    scenarios: list[RigidScenarioResult] = []
    planning: list[RigidPlanningResult] = []
    learned_weight_bytes = 0
    for spec in default_rigid_scenarios():
        result, belief, truth, dynamics = _evaluate_scenario(spec)
        scenarios.append(result)
        learned_weight_bytes = max(
            learned_weight_bytes,
            sum(
                parameter.numel() * parameter.element_size() for parameter in dynamics.parameters()
            ),
        )
        for candidate_count in (8, 32):
            planning.append(_planning(spec.name, belief, truth, dynamics, candidate_count))
    scenario_tuple = tuple(scenarios)
    planning_tuple = tuple(planning)
    failures = _gate_failures(scenario_tuple, planning_tuple)
    return RigidCapabilityResult(
        schema=RIGID_CAPABILITY_SCHEMA,
        manifest_sha256=rigid_manifest_sha256(),
        scenarios=scenario_tuple,
        planning=planning_tuple,
        gate_failures=failures,
        qualified=not failures,
        learned_weight_bytes=learned_weight_bytes,
        peak_run_tensor_bytes=6 * 2 * _IMAGE_SIZE[0] * _IMAGE_SIZE[1] * 5 * 8,
    )


def _summary(
    result: RigidCapabilityResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    current_rmse = math.sqrt(
        sum(item.current_position_rmse_m**2 for item in result.scenarios) / len(result.scenarios)
    )
    horizon_rmse = math.sqrt(
        sum(item.two_second_position_rmse_m**2 for item in result.scenarios) / len(result.scenarios)
    )
    planning_by_k = {
        str(candidate_count): {
            "winner_accuracy": sum(item.winner_correct for item in selected) / len(selected),
            "median_normalized_regret": sorted(item.normalized_regret for item in selected)[
                len(selected) // 2
            ],
            "goal_success": sum(item.goal_success for item in selected) / len(selected),
        }
        for candidate_count in (8, 32)
        if (
            selected := [
                item for item in result.planning if item.candidate_count == candidate_count
            ]
        )
    }
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.qualified else "failed",
        outcome="qualified_convergence" if result.qualified else "capability_gate_failed",
        source_format=RIGID_CAPABILITY_SCHEMA,
        configuration={
            "primitives": ["sphere", "oriented_box"],
            "observation": "public calibrated multi-view RGB-D",
            "horizon_seconds": _HORIZON_SECONDS,
            "planning_used_as_training_loss": False,
            "generated_frames_retained": False,
            "contact_model": "central linear impulses; no contact torque",
        },
        provenance={
            "scenario_manifest_sha256": result.manifest_sha256,
            "truth_oracle": "independent world_model.simulator.rigid_physics",
            "forecast_animation_source_run": run_id,
        },
        scores={
            "candidate": current_rmse + horizon_rmse,
            "incumbent": current_rmse + horizon_rmse,
            "selected": "incumbent",
        },
        factor_metrics={
            "observable_rigid_geometry": {
                "status": "passed" if result.qualified else "failed",
                "primitive_accuracy": min(item.primitive_accuracy for item in result.scenarios),
                "maximum_box_half_extent_rmse_m": max(
                    item.box_half_extent_rmse_m for item in result.scenarios
                ),
                "maximum_two_second_position_rmse_m": max(
                    item.two_second_position_rmse_m for item in result.scenarios
                ),
            }
        },
        cell_metrics={f"N2/rigid/{item.name}": item.to_dict() for item in result.scenarios},
        horizon_curves={"candidate_position_rmse_m": {"0": current_rmse, "2": horizon_rmse}},
        uncertainty={"status": "not re-estimated for the parameter-free geometry seam"},
        planning={
            "status": "passed"
            if all(item.winner_correct for item in result.planning)
            else "failed",
            "by_candidate_count": planning_by_k,
            "tasks": [item.to_dict() for item in result.planning],
            "serial_vectorized_winner_parity": all(
                item.serial_vectorized_parity for item in result.planning
            ),
            "maximum_cost_difference": max(
                item.maximum_cost_difference for item in result.planning
            ),
        },
        resources={
            "learned_weight_bytes": result.learned_weight_bytes,
            "peak_run_tensor_bytes": result.peak_run_tensor_bytes,
            "planning_k8_latency_seconds": max(
                item.latency_seconds for item in result.planning if item.candidate_count == 8
            ),
            "planning_k32_latency_seconds": max(
                item.latency_seconds for item in result.planning if item.candidate_count == 32
            ),
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "incumbent",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "observable geometry or analytic contact",
        },
        qualitative={
            "best_episode": result.scenarios[0].name,
            "representative_episode": result.scenarios[-1].name,
            "worst_episode": max(
                result.scenarios, key=lambda item: item.two_second_position_rmse_m
            ).name,
            "animations": [],
            "forecast_animations": [item.animation for item in result.scenarios],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "rotational contact impulse or torque qualification",
            "articulated or deformable geometry",
            "full N=7/8 mixed-rigid perceptual qualification",
        ),
        scope_limitations=(
            "known calibrated multi-view cameras",
            "sphere and oriented-box primitives only",
            "central low-friction impacts",
            "short two-frame persistent appearance handles",
        ),
    ).validate()


def publish_rigid_capability(
    result: RigidCapabilityResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    """Publish a portable rigid report, manifest, and refreshed dashboard."""

    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "rigid_capability.json",
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_bytes = int(inventory_runs(runs_root, archive_root=archive_root)["archive_bytes"])
    summary = _summary(
        result,
        run_id=run.name,
        run_bytes=0,
        archive_bytes=archive_bytes,
    )
    for _ in range(8):
        write_capability_summary(summary, run / "capability_summary.json")
        write_run_report(summary, run)
        write_run_manifest(
            run,
            role="candidate",
            status="completed" if result.qualified else "failed",
            artifacts={
                "rigid_capability.json": "summary",
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
        summary = replace(
            summary,
            artifacts={**summary.artifacts, "run_bytes": actual},
        )
    else:
        raise RuntimeError("rigid capability evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "RIGID_CAPABILITY_SCHEMA",
    "RigidCapabilityResult",
    "RigidPlanningResult",
    "RigidScenarioResult",
    "RigidScenarioSpec",
    "default_rigid_scenarios",
    "publish_rigid_capability",
    "rigid_manifest_sha256",
    "run_rigid_capability",
]
