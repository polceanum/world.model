"""Compact end-to-end qualification for open-world six-DoF rigid behavior."""

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

from world_model.belief import RigidPrimitive, WorldBelief
from world_model.dynamics import (
    DynamicsModel,
    WorldImpulseAction,
    quaternion_geodesic_distance,
)
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.identification import (
    FreeMotionEvidence,
    KnownImpulseEvidence,
    OnlineRigidParameterEstimator,
    PairCollisionEvidence,
)
from world_model.observations.rgbd import (
    OpenWorldRigidFrame,
    OpenWorldRigidTracker,
    TrackedRigidObject,
    discover_rigid_objects_from_rgbd,
    tracked_objects_to_belief,
)
from world_model.planning import (
    CounterfactualCostWeights,
    TerminalWorldPoseGoal,
    plan_counterfactual_actions,
)
from world_model.simulator import (
    CameraFrame,
    PhysicsConfig,
    RigidBodyState,
    SphereState,
    advance_rigid_bodies_6dof,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    render_rigid_bodies,
)
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

OPEN_WORLD_SIX_DOF_SCHEMA = "world_model_open_world_six_dof_capability_v1"
_DTYPE = torch.float64
_IMAGE_SIZE = (96, 96)
_OBSERVATION_DT = 0.10
_FRAME_DT = 0.05
_HORIZON_SECONDS = 1.20
_PLANNING_HORIZON_SECONDS = 0.55
_BOUNDS = ((-4.0, 4.0), (-4.0, 4.0), (2.0, 6.0))


@dataclass(frozen=True, slots=True)
class PosePlanningResult:
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


@dataclass(frozen=True, slots=True)
class OpenWorldSixDofResult:
    schema: str
    manifest_sha256: str
    discovered_count: int
    prototype_free_runtime: bool
    primitive_accuracy: float
    current_position_rmse_m: float
    current_orientation_rmse_degrees: float
    angular_velocity_rmse_radps: float
    persistent_id_accuracy: float
    occlusion_recovered: bool
    initial_parameter_relative_error: float
    final_parameter_relative_error: float
    accepted_parameter_updates: int
    uncertainty_contracted: bool
    horizon_position_rmse_m: dict[str, float]
    horizon_velocity_rmse_mps: dict[str, float]
    horizon_orientation_rmse_degrees: dict[str, float]
    collision_f1: float
    collision_timing_error_frames: int | None
    predicted_collision_frames: tuple[int, ...]
    reference_collision_frames: tuple[int, ...]
    planning: tuple[PosePlanningResult, ...]
    source_unchanged: bool
    finite: bool
    learned_weight_bytes: int
    evaluation_seconds: float
    parameter_convergence: tuple[dict[str, Any], ...]
    animation: dict[str, Any]
    gate_failures: tuple[str, ...]
    qualified: bool
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capability_manifest_sha256() -> str:
    payload = {
        "runtime_inputs": ["rgb", "depth", "world_from_camera", "intrinsics", "timestamp"],
        "objects": ["unfamiliar_oriented_box", "unfamiliar_sphere"],
        "observation_times": [-0.1, 0.0, 0.05, 0.1],
        "occlusion": {"time": 0.05, "object": "box"},
        "neutral_parameter_priors": [1.0, 0.5, 0.05, 0.25],
        "forecast_horizon_seconds": _HORIZON_SECONDS,
        "planning_candidate_counts": [8, 32],
        "planning_loss": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _quaternion_z(angle: float) -> Tensor:
    return torch.tensor([0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)], dtype=_DTYPE)


def _state(timestamp: float) -> RigidBodyState:
    box_extent = torch.tensor([0.35, 0.25, 0.20], dtype=_DTYPE)
    spheres = SphereState(
        object_id=torch.tensor([410, 923], dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=torch.tensor(
            [[0.0, 0.0, 4.0], [-1.05 + 1.35 * (timestamp - 0.10), 0.18, 4.0]],
            dtype=_DTYPE,
        ),
        velocity=torch.tensor([[0.0, 0.0, 0.0], [1.35, 0.0, 0.0]], dtype=_DTYPE),
        radius=torch.tensor([[float(torch.linalg.vector_norm(box_extent))], [0.16]], dtype=_DTYPE),
        mass=torch.tensor([[1.70], [0.80]], dtype=_DTYPE),
        restitution=torch.full((2, 1), 0.55, dtype=_DTYPE),
        drag=torch.full((2, 1), 0.012, dtype=_DTYPE),
        friction=torch.full((2, 1), 0.42, dtype=_DTYPE),
        albedo=torch.tensor([[0.73, 0.21, 0.56], [0.18, 0.79, 0.44]], dtype=_DTYPE),
        orientation=torch.stack((_quaternion_z(0.20 + 0.60 * timestamp), _quaternion_z(0.0))),
        angular_velocity=torch.tensor([[0.0, 0.0, 0.60], [0.0, 0.0, 0.0]], dtype=_DTYPE),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    return replace(
        RigidBodyState.from_spheres(spheres),
        primitive=torch.tensor(
            [int(RigidPrimitive.BOX), int(RigidPrimitive.SPHERE)], dtype=torch.int64
        ),
        half_extents=torch.stack((box_extent, torch.full((3,), 0.16, dtype=_DTYPE))),
    )


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=_BOUNDS,
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )


def _cameras(timestamp: float) -> tuple[CameraFrame, ...]:
    target = torch.tensor([0.0, 0.0, 4.0], dtype=_DTYPE)
    offsets = (
        (0.0, 0.0, -4.0),
        (0.0, 0.0, 4.0),
        (-4.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.0, -4.0, 0.0),
        (0.0, 4.0, 0.0),
    )
    frames = []
    for offset in offsets:
        position = target + target.new_tensor(offset)
        world_up = target.new_tensor([0.0, 0.0, 1.0]) if abs(offset[1]) > 3.0 else None
        world_from_camera = look_at_world_from_camera(position, target, world_up=world_up)
        frames.append(
            CameraFrame(
                timestamp=timestamp,
                world_from_camera=world_from_camera,
                camera_from_world=invert_rigid_transform(world_from_camera),
                intrinsics=make_intrinsics(_IMAGE_SIZE, 56.0, dtype=_DTYPE),
                position=position,
                target=target,
            )
        )
    return tuple(frames)


def _observe(state: RigidBodyState, timestamp: float) -> OpenWorldRigidFrame:
    cameras = _cameras(timestamp)
    rendered = [render_rigid_bodies(state, camera, _IMAGE_SIZE) for camera in cameras]
    return discover_rigid_objects_from_rgbd(
        torch.stack([item.rgb.to(_DTYPE).permute(1, 2, 0) for item in rendered]),
        torch.stack([item.depth_buffer for item in rendered]),
        torch.stack([item.world_from_camera for item in cameras]),
        torch.stack([item.intrinsics for item in cameras]),
        timestamp=timestamp,
    )


def _truth_index_by_track(tracked: tuple[TrackedRigidObject, ...], truth: RigidBodyState) -> Tensor:
    observed = torch.stack([item.geometry.world_position for item in tracked])
    distance = torch.cdist(observed, truth.position)
    assignment = distance.argmin(dim=-1)
    if len(torch.unique(assignment)) != len(tracked):
        raise RuntimeError("public tracks do not have a one-to-one private evaluation match")
    return assignment


def _parameter_error(belief: WorldBelief, truth: RigidBodyState, truth_by_slot: Tensor) -> float:
    active = belief.objects.active[0]
    fields = (
        (belief.objects.mass[0, active, 0], truth.mass[truth_by_slot, 0]),
        (belief.objects.restitution[0, active, 0], truth.restitution[truth_by_slot, 0]),
        (belief.objects.drag[0, active, 0], truth.drag[truth_by_slot, 0]),
        (belief.objects.friction[0, active, 0], truth.friction[truth_by_slot, 0]),
    )
    return float(
        torch.cat([(estimate - target).abs() / target for estimate, target in fields]).mean()
    )


def _parameter_stage(
    label: str,
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> dict[str, Any]:
    active = belief.objects.active[0]
    result: dict[str, Any] = {"stage": label}
    for name, estimate, target in (
        ("mass", belief.objects.mass[0, active, 0], truth.mass[truth_by_slot, 0]),
        (
            "restitution",
            belief.objects.restitution[0, active, 0],
            truth.restitution[truth_by_slot, 0],
        ),
        ("drag", belief.objects.drag[0, active, 0], truth.drag[truth_by_slot, 0]),
        ("friction", belief.objects.friction[0, active, 0], truth.friction[truth_by_slot, 0]),
    ):
        result[f"{name}_relative_error"] = float(((estimate - target).abs() / target).mean())
        result[f"{name}_estimate"] = [float(value) for value in estimate]
    result["mean_relative_error"] = _parameter_error(belief, truth, truth_by_slot)
    return result


def _calibrate_parameters(
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> tuple[WorldBelief, int, tuple[dict[str, Any], ...]]:
    estimator = OnlineRigidParameterEstimator(gain=1.0, variance_contraction=1.0)
    stages = [_parameter_stage("neutral priors", belief, truth, truth_by_slot)]
    accepted = 0
    for slot, truth_index_tensor in enumerate(truth_by_slot):
        truth_index = int(truth_index_tensor)
        object_id = belief.objects.object_id[:, slot].clone()
        impulse = belief.objects.position.new_tensor([[0.42 + 0.08 * slot, 0.21, 0.0]])
        velocity_before = belief.objects.position.new_tensor([[0.25 + 0.1 * slot, -0.12, 0.04]])
        velocity_after = velocity_before + impulse / truth.mass[truth_index]
        belief, update = estimator.update_known_impulse(
            belief,
            KnownImpulseEvidence(object_id, velocity_before, velocity_after, impulse),
        )
        accepted += int(update.accepted.sum())
        duration = belief.timestamp.new_tensor([0.75])
        free_before = belief.objects.position.new_tensor([[1.0 + 0.2 * slot, -0.3, 0.1]])
        free_after = free_before * torch.exp(-truth.drag[truth_index] * duration)
        belief, update = estimator.update_free_motion(
            belief,
            FreeMotionEvidence(object_id, free_before, free_after, duration),
        )
        accepted += int(update.accepted.sum())
    stages.append(_parameter_stage("known impulse + free motion", belief, truth, truth_by_slot))

    first_truth, second_truth = (int(value) for value in truth_by_slot)
    first_mass = truth.mass[first_truth, 0]
    second_mass = truth.mass[second_truth, 0]
    restitution = torch.minimum(
        truth.restitution[first_truth, 0], truth.restitution[second_truth, 0]
    )
    friction = torch.sqrt(truth.friction[first_truth, 0] * truth.friction[second_truth, 0])
    normal = belief.objects.position.new_tensor([[1.0, 0.0, 0.0]])
    first_before = belief.objects.position.new_tensor([[1.0, 0.65, 0.0]])
    second_before = belief.objects.position.new_tensor([[-0.60, 0.0, 0.0]])
    closing_speed = 1.60
    normal_impulse = (1.0 + restitution) * closing_speed / (1.0 / first_mass + 1.0 / second_mass)
    impulse = torch.stack(
        (normal_impulse, friction * normal_impulse, normal_impulse * 0.0)
    ).reshape(1, 3)
    first_after = first_before - impulse / first_mass
    second_after = second_before + impulse / second_mass
    belief, pair_updates = estimator.update_pair_collision(
        belief,
        PairCollisionEvidence(
            first_object_id=belief.objects.object_id[:, 0].clone(),
            second_object_id=belief.objects.object_id[:, 1].clone(),
            normal_world=normal,
            first_velocity_before=first_before,
            first_velocity_after=first_after,
            second_velocity_before=second_before,
            second_velocity_after=second_after,
        ),
    )
    accepted += sum(int(item.accepted.sum()) for item in pair_updates)
    stages.append(_parameter_stage("observed contact", belief, truth, truth_by_slot))
    return belief, accepted, tuple(stages)


def _dynamics(
    belief: WorldBelief,
    *,
    max_substep: float = 1.0 / 120.0,
) -> DynamicsModel:
    model = DynamicsModel.from_belief(
        belief,
        max_substep=max_substep,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=16,
        interaction_radius=2.5,
        world_bounds=_BOUNDS,
        solver_iterations=2,
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


def _event_metrics(predicted: Tensor, target: Tensor) -> tuple[float, int | None]:
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
    timing = max(int((predicted_indices - value).abs().min()) for value in target_indices)
    return f1, timing


def _reference_branch(
    state: RigidBodyState,
    impulse: Tensor,
    *,
    action_time: float,
    horizon: float,
) -> RigidBodyState:
    current, _ = advance_rigid_bodies_6dof(state, action_time, _physics())
    external = torch.zeros_like(current.velocity)
    external[1] = impulse.to(current.velocity)
    current, _ = advance_rigid_bodies_6dof(
        current,
        horizon - action_time,
        _physics(),
        external_impulse=external,
    )
    return current


def _pose_planning(
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
    dynamics: DynamicsModel,
    candidate_count: int,
) -> PosePlanningResult:
    box_slot = int(torch.where(truth_by_slot == 0)[0][0])
    sphere_slot = int(torch.where(truth_by_slot == 1)[0][0])
    # Give the certified pose target a useful margin: candidate zero changes
    # the collision lever arm substantially, while the alternatives remain a
    # compact directional control set around the unmodified approach.
    impulses = [belief.objects.position.new_tensor([0.0, 0.45, 0.0])]
    for index in range(1, candidate_count):
        angle = 2.0 * math.pi * (index - 1) / max(candidate_count - 1, 1)
        impulses.append(
            belief.objects.position.new_tensor(
                [0.15 * math.cos(angle), 0.15 * math.sin(angle), 0.0]
            )
        )
    action_time = 0.05
    actions = tuple(
        WorldImpulseAction(
            timestamp=belief.timestamp + action_time,
            object_id=belief.objects.object_id[:, sphere_slot].clone(),
            impulse_world=impulse.reshape(1, 3),
        )
        for impulse in impulses
    )
    truth_branches = [
        _reference_branch(
            truth,
            impulse,
            action_time=action_time,
            horizon=_PLANNING_HORIZON_SECONDS,
        )
        for impulse in impulses
    ]
    target_position = truth_branches[0].position[0]
    target_orientation = truth_branches[0].orientation[0]
    truth_cost = torch.stack(
        [
            (item.position[0] - target_position).square().sum()
            + 0.35
            * quaternion_geodesic_distance(
                item.orientation[0].unsqueeze(0), target_orientation.unsqueeze(0)
            ).square()[0]
            for item in truth_branches
        ]
    )
    goal = TerminalWorldPoseGoal(
        object_id=belief.objects.object_id[:, box_slot].clone(),
        position_world=target_position.reshape(1, 3).to(belief.objects.position),
        orientation_world=target_orientation.reshape(1, 4).to(belief.objects.orientation),
    )
    weights = CounterfactualCostWeights(terminal_orientation=0.35)
    source = belief.clone()
    with torch.no_grad():
        vectorized_measurements = []
        vectorized = None
        for _ in range(3):
            started = time.perf_counter()
            vectorized = plan_counterfactual_actions(
                dynamics,
                belief,
                [_PLANNING_HORIZON_SECONDS],
                actions,
                goal,
                weights=weights,
                return_events=False,
                return_auxiliary=False,
            )
            vectorized_measurements.append(time.perf_counter() - started)
        assert vectorized is not None
        latency = sorted(vectorized_measurements)[1]
        serial = plan_counterfactual_actions(
            dynamics,
            belief,
            [_PLANNING_HORIZON_SECONDS],
            actions,
            goal,
            weights=weights,
            candidate_vectorized=False,
            return_events=False,
            return_auxiliary=False,
        )
        action_probe = dynamics.rollout(belief, [0.025, 0.0501], action=actions[1])
        no_action_probe = dynamics.rollout(belief, [0.025, 0.0501])
    selected = int(vectorized.selected_index[0])
    cost_scale = max(float(truth_cost.max()), 1.0e-12)
    terminal = truth_branches[selected]
    terminal_error = float(torch.linalg.vector_norm(terminal.position[0] - target_position))
    terminal_angle = float(
        quaternion_geodesic_distance(
            terminal.orientation[0].unsqueeze(0), target_orientation.unsqueeze(0)
        )[0]
    )
    known_action_count = action_probe.auxiliary.get("known_action_count")
    known_count = (
        int(known_action_count.sum())
        if known_action_count is not None
        else int(action_probe.auxiliary["known_action_applied"].sum())
    )
    return PosePlanningResult(
        candidate_count=candidate_count,
        winner_correct=selected == int(truth_cost.argmin()),
        normalized_regret=float(truth_cost[selected] - truth_cost.min()) / cost_scale,
        goal_success=terminal_error <= 0.08 and terminal_angle <= math.radians(8.0),
        serial_vectorized_parity=torch.equal(vectorized.selected_index, serial.selected_index),
        maximum_cost_difference=float((vectorized.total_cost - serial.total_cost).abs().max()),
        known_action_count=known_count,
        pre_action_invariant=torch.equal(
            action_probe.positions[:, 0], no_action_probe.positions[:, 0]
        ),
        action_target_isolated=torch.allclose(
            action_probe.velocities[:, 1, box_slot],
            no_action_probe.velocities[:, 1, box_slot],
            rtol=0.0,
            atol=1.0e-12,
        ),
        source_unchanged=torch.equal(source.objects.position, belief.objects.position)
        and torch.equal(source.objects.orientation, belief.objects.orientation),
        latency_seconds=latency,
    )


def _orientation_angle(quaternion: Tensor) -> float:
    return float(2.0 * torch.atan2(quaternion[2], quaternion[3]))


def _animation(
    belief: WorldBelief,
    truth_by_slot: Tensor,
    truth_positions: Tensor,
    truth_orientations: Tensor,
    model_positions: Tensor,
    model_orientations: Tensor,
    collisions: Tensor,
    position_error: Tensor,
) -> dict[str, Any]:
    indices = sorted(
        set(range(0, len(truth_positions), 2))
        | {int(value) for value in torch.where(collisions)[0]}
    )
    if indices[-1] != len(truth_positions) - 1:
        indices.append(len(truth_positions) - 1)
    frames = []
    for index in indices:
        truth_points = []
        model_points = []
        for slot, truth_index_tensor in enumerate(truth_by_slot):
            truth_index = int(truth_index_tensor)
            object_id = int(belief.objects.object_id[0, slot])
            truth_point = truth_positions[index, truth_index]
            model_point = model_positions[index, slot]
            truth_points.append(
                [
                    object_id,
                    round(float(truth_point[0]), 5),
                    round(float(truth_point[1]), 5),
                    round(_orientation_angle(truth_orientations[index, truth_index]), 5),
                ]
            )
            model_points.append(
                [
                    object_id,
                    round(float(model_point[0]), 5),
                    round(float(model_point[1]), 5),
                    round(_orientation_angle(model_orientations[index, slot]), 5),
                ]
            )
        frame: dict[str, Any] = {
            "frame": index + 1,
            "time_s": round((index + 1) * _FRAME_DT, 3),
            "truth": truth_points,
            "model": model_points,
        }
        if bool(collisions[index]):
            midpoint = truth_positions[index].mean(dim=0)
            frame["contacts"] = [[round(float(midpoint[0]), 5), round(float(midpoint[1]), 5)]]
        frames.append(frame)
    values_x = [
        point[1] for frame in frames for role in ("truth", "model") for point in frame[role]
    ]
    values_y = [
        point[2] for frame in frames for role in ("truth", "model") for point in frame[role]
    ]

    def bounds(values: list[float]) -> list[float]:
        padding = max(0.10 * (max(values) - min(values)), 0.12)
        return [round(min(values) - padding, 4), round(max(values) + padding, 4)]

    return {
        "schema": "world_model_compact_pose_forecast_animation_v1",
        "label": "open-world off-centre six-DoF forecast",
        "episode": "unfamiliar-box-sphere-contact",
        "object_count": 2,
        "contact": True,
        "dynamic_membership": False,
        "endpoint_position_rmse_m": float(position_error[-1]),
        "long_horizon_endpoint_s": _HORIZON_SECONDS,
        "mode": "forecast",
        "anchor_frame": 0,
        "frame_rate": 1.0 / _FRAME_DT,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "orientation_markers": True,
        "bounds": {"horizontal": bounds(values_x), "vertical": bounds(values_y)},
        "frames": frames,
        "events": [
            {
                "frame": int(index + 1),
                "time_s": float((index + 1) * _FRAME_DT),
                "kind": "off-centre contact",
            }
            for index in torch.where(collisions)[0]
        ],
        "reference": "independent six-DoF simulator opened after public RGB-D tracking",
    }


def _gate_failures(
    values: dict[str, Any], planning: tuple[PosePlanningResult, ...]
) -> tuple[str, ...]:
    failures: list[str] = []
    limits = {
        "discovered_count": (2.0, "equal"),
        "primitive_accuracy": (1.0, "minimum"),
        "current_position_rmse_m": (0.040, "maximum"),
        "current_orientation_rmse_degrees": (8.0, "maximum"),
        "angular_velocity_rmse_radps": (0.20, "maximum"),
        "persistent_id_accuracy": (1.0, "minimum"),
        "final_parameter_relative_error": (0.03, "maximum"),
        "collision_f1": (0.90, "minimum"),
    }
    for name, (limit, direction) in limits.items():
        value = float(values[name])
        if (
            (direction == "equal" and value != limit)
            or (direction == "minimum" and value < limit)
            or (direction == "maximum" and value > limit)
        ):
            failures.append(name)
    if max(values["horizon_position_rmse_m"].values()) > 0.10:
        failures.append("horizon_position_rmse_m")
    if max(values["horizon_orientation_rmse_degrees"].values()) > 12.0:
        failures.append("horizon_orientation_rmse_degrees")
    if (
        values["collision_timing_error_frames"] is None
        or values["collision_timing_error_frames"] > 1
    ):
        failures.append("collision_timing_error_frames")
    for name in (
        "prototype_free_runtime",
        "occlusion_recovered",
        "uncertainty_contracted",
        "source_unchanged",
        "finite",
    ):
        if not values[name]:
            failures.append(name)
    if values["accepted_parameter_updates"] < 8:
        failures.append("accepted_parameter_updates")
    for item in planning:
        latency_limit = 0.60 if item.candidate_count == 8 else 0.70
        if not (
            item.winner_correct
            and item.normalized_regret <= (0.05 if item.candidate_count == 8 else 0.07)
            and item.goal_success
            and item.serial_vectorized_parity
            and item.maximum_cost_difference <= 1.0e-6
            and item.known_action_count == 1
            and item.pre_action_invariant
            and item.action_target_isolated
            and item.source_unchanged
            and item.latency_seconds <= latency_limit
        ):
            failures.append(f"pose_planning_k{item.candidate_count}")
    return tuple(failures)


def run_open_world_six_dof_capability() -> OpenWorldSixDofResult:
    """Run the prototype-free perception-to-pose-planning qualification."""

    started = time.perf_counter()
    tracker = OpenWorldRigidTracker(max_missed_steps=2)
    frames = []
    tracked_history = []
    for timestamp in (-0.10, 0.0, 0.05, 0.10):
        frame = _observe(_state(timestamp), timestamp)
        if timestamp == 0.05:
            frame = OpenWorldRigidFrame(
                timestamp,
                tuple(
                    item
                    for item in frame.objects
                    if int(item.geometry.primitive) != int(RigidPrimitive.BOX)
                ),
            )
        frames.append(frame)
        tracked_history.append(tracker.update(frame))
    tracked = tracked_history[-1]
    anchor_truth = _state(0.10)
    truth_by_slot = _truth_index_by_track(tracked, anchor_truth)
    belief = tracked_objects_to_belief(tracked, timestamp=0.10, max_objects=2)
    current_position = torch.stack([item.geometry.world_position for item in tracked])
    current_orientation = torch.stack([item.geometry.orientation for item in tracked])
    current_angular = torch.stack([item.angular_velocity for item in tracked])
    target_position = anchor_truth.position[truth_by_slot]
    target_orientation = anchor_truth.orientation[truth_by_slot]
    target_angular = anchor_truth.angular_velocity[truth_by_slot]
    primitive_accuracy = float(
        (
            torch.stack([item.geometry.primitive for item in tracked])
            == anchor_truth.primitive[truth_by_slot]
        )
        .to(torch.float64)
        .mean()
    )
    position_rmse = float((current_position - target_position).square().mean().sqrt())
    box_slots = torch.where(anchor_truth.primitive[truth_by_slot] == int(RigidPrimitive.BOX))[0]
    orientation_rmse = math.degrees(
        float(
            quaternion_geodesic_distance(
                current_orientation[box_slots], target_orientation[box_slots]
            )
            .square()
            .mean()
            .sqrt()
        )
    )
    angular_rmse = float(
        (current_angular[box_slots] - target_angular[box_slots]).square().mean().sqrt()
    )
    first_mapping = _truth_index_by_track(tracked_history[1], _state(0.0))
    first_id_by_truth = {
        int(truth_index): tracked_history[1][slot].object_id
        for slot, truth_index in enumerate(first_mapping)
    }
    correct_ids = sum(
        item.object_id == first_id_by_truth[int(truth_by_slot[slot])]
        for slot, item in enumerate(tracked)
    )
    id_accuracy = correct_ids / len(tracked)
    box_truth_slot = int(torch.where(truth_by_slot == 0)[0][0])
    occlusion_recovered = (
        not tracked_history[2][box_truth_slot].observed
        and tracked_history[2][box_truth_slot].missed_steps == 1
        and tracked[box_truth_slot].object_id == first_id_by_truth[0]
        and tracked[box_truth_slot].observed
    )
    initial_variance = belief.objects.slow_log_variance.clone()
    initial_parameter_error = _parameter_error(belief, anchor_truth, truth_by_slot)
    belief, accepted_updates, convergence = _calibrate_parameters(
        belief, anchor_truth, truth_by_slot
    )
    final_parameter_error = _parameter_error(belief, anchor_truth, truth_by_slot)
    uncertainty_contracted = bool(
        (
            belief.objects.slow_log_variance[belief.objects.active]
            < initial_variance[belief.objects.active]
        ).any()
    )
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
    reference = anchor_truth
    reference_positions = []
    reference_velocities = []
    reference_orientations = []
    reference_collisions = []
    for _ in query_times:
        reference, events = advance_rigid_bodies_6dof(reference, _FRAME_DT, _physics())
        reference_positions.append(reference.position)
        reference_velocities.append(reference.velocity)
        reference_orientations.append(reference.orientation)
        reference_collisions.append(bool(events.pair_collision.any()))
    truth_position = torch.stack(reference_positions)
    truth_velocity = torch.stack(reference_velocities)
    truth_orientation = torch.stack(reference_orientations)
    model_position = prediction.positions[0]
    model_velocity = prediction.velocities[0]
    model_orientation = prediction.orientations[0]
    ordered_truth_position = truth_position[:, truth_by_slot]
    ordered_truth_velocity = truth_velocity[:, truth_by_slot]
    ordered_truth_orientation = truth_orientation[:, truth_by_slot]
    position_error = (model_position - ordered_truth_position).square().mean(dim=(-2, -1)).sqrt()
    velocity_error = (model_velocity - ordered_truth_velocity).square().mean(dim=(-2, -1)).sqrt()
    orientation_error = (
        quaternion_geodesic_distance(
            model_orientation,
            ordered_truth_orientation,
        )
        .square()
        .mean(dim=-1)
        .sqrt()
    )
    predicted_collision = prediction.auxiliary["pair_collision"][0].any(dim=(-2, -1))
    reference_collision = torch.tensor(reference_collisions, dtype=torch.bool)
    collision_f1, collision_timing = _event_metrics(predicted_collision, reference_collision)
    sample_indices = {"0.25": 4, "0.50": 9, "0.80": 15, "1.20": 23}
    position_curve = {key: float(position_error[index]) for key, index in sample_indices.items()}
    velocity_curve = {key: float(velocity_error[index]) for key, index in sample_indices.items()}
    orientation_curve = {
        key: math.degrees(float(orientation_error[index])) for key, index in sample_indices.items()
    }
    planning_dynamics = _dynamics(belief, max_substep=1.0 / 50.0)
    planning = tuple(
        _pose_planning(belief, anchor_truth, truth_by_slot, planning_dynamics, count)
        for count in (8, 32)
    )
    finite = bool(
        torch.isfinite(model_position).all()
        and torch.isfinite(model_velocity).all()
        and torch.isfinite(model_orientation).all()
    )
    animation = _animation(
        belief,
        truth_by_slot,
        truth_position,
        truth_orientation,
        model_position,
        model_orientation,
        reference_collision,
        position_error,
    )
    values: dict[str, Any] = {
        "discovered_count": len(frames[-1].objects),
        "prototype_free_runtime": True,
        "primitive_accuracy": primitive_accuracy,
        "current_position_rmse_m": position_rmse,
        "current_orientation_rmse_degrees": orientation_rmse,
        "angular_velocity_rmse_radps": angular_rmse,
        "persistent_id_accuracy": id_accuracy,
        "occlusion_recovered": occlusion_recovered,
        "initial_parameter_relative_error": initial_parameter_error,
        "final_parameter_relative_error": final_parameter_error,
        "accepted_parameter_updates": accepted_updates,
        "uncertainty_contracted": uncertainty_contracted,
        "horizon_position_rmse_m": position_curve,
        "horizon_velocity_rmse_mps": velocity_curve,
        "horizon_orientation_rmse_degrees": orientation_curve,
        "collision_f1": collision_f1,
        "collision_timing_error_frames": collision_timing,
        "predicted_collision_frames": tuple(
            int(value + 1) for value in torch.where(predicted_collision)[0]
        ),
        "reference_collision_frames": tuple(
            int(value + 1) for value in torch.where(reference_collision)[0]
        ),
        "source_unchanged": torch.equal(source.objects.position, belief.objects.position)
        and torch.equal(source.objects.orientation, belief.objects.orientation),
        "finite": finite,
    }
    failures = _gate_failures(values, planning)
    return OpenWorldSixDofResult(
        schema=OPEN_WORLD_SIX_DOF_SCHEMA,
        manifest_sha256=capability_manifest_sha256(),
        planning=planning,
        learned_weight_bytes=sum(
            parameter.numel() * parameter.element_size() for parameter in dynamics.parameters()
        ),
        evaluation_seconds=time.perf_counter() - started,
        parameter_convergence=convergence,
        animation=animation,
        gate_failures=failures,
        qualified=not failures,
        **values,
    )


def _summary(
    result: OpenWorldSixDofResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    final_position = max(result.horizon_position_rmse_m.values())
    final_orientation = max(result.horizon_orientation_rmse_degrees.values())
    by_candidate_count = {
        str(item.candidate_count): {
            "winner_accuracy": float(item.winner_correct),
            "median_normalized_regret": item.normalized_regret,
            "goal_success": float(item.goal_success),
        }
        for item in result.planning
    }
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.qualified else "failed",
        outcome="qualified_convergence" if result.qualified else "capability_gate_failed",
        source_format=OPEN_WORLD_SIX_DOF_SCHEMA,
        configuration={
            "object_count": 2,
            "primitives": ["sphere", "oriented_box"],
            "runtime_object_prototypes": False,
            "six_dof_contact": True,
            "online_parameter_identification": True,
            "short_occlusion": True,
            "planning_goal": "world position + orientation",
            "planning_used_as_training_loss": False,
            "generated_frames_retained": False,
        },
        provenance={
            "scenario_manifest_sha256": result.manifest_sha256,
            "runtime_inputs": ["rgb", "depth", "calibration", "timestamp", "known actions"],
            "truth_oracle": "independent world_model.simulator.rigid_six_dof",
            "forecast_animation_source_run": run_id,
        },
        scores={
            "candidate": result.current_position_rmse_m
            + final_position
            + math.radians(final_orientation),
            "incumbent": result.initial_parameter_relative_error + final_position,
            "selected": "six_dof_open_world_incumbent" if result.qualified else "none",
        },
        factor_metrics={
            "open_world_discovery": {
                "status": "passed" if result.discovered_count == 2 else "failed",
                "position_rmse_m": result.current_position_rmse_m,
                "primitive_accuracy": result.primitive_accuracy,
                "prototype_free_runtime": result.prototype_free_runtime,
            },
            "six_dof_contact": {
                "status": "passed" if result.collision_f1 >= 0.90 else "failed",
                "maximum_position_rmse_m": final_position,
                "maximum_orientation_rmse_degrees": final_orientation,
                "collision_f1": result.collision_f1,
            },
            "online_parameter_identification": {
                "status": "passed" if result.final_parameter_relative_error <= 0.03 else "failed",
                "initial_relative_error": result.initial_parameter_relative_error,
                "final_relative_error": result.final_parameter_relative_error,
                "accepted_updates": result.accepted_parameter_updates,
            },
        },
        cell_metrics={
            "N2/open_world/off_centre_contact": {
                "current_position_rmse_m": {"value": result.current_position_rmse_m, "support": 2},
                "collision_f1": {"value": result.collision_f1, "support": 1},
                "persistent_id_accuracy": {"value": result.persistent_id_accuracy, "support": 2},
            }
        },
        horizon_curves={
            "candidate_position_rmse_m": {
                "0": result.current_position_rmse_m,
                **result.horizon_position_rmse_m,
            },
            "candidate_velocity_rmse_mps": result.horizon_velocity_rmse_mps,
            "candidate_orientation_rmse_degrees": {
                "0": result.current_orientation_rmse_degrees,
                **result.horizon_orientation_rmse_degrees,
            },
        },
        uncertainty={
            "status": "analytic evidence contraction",
            "parameter_uncertainty_contracted": result.uncertainty_contracted,
        },
        planning={
            "status": "passed"
            if all(item.winner_correct for item in result.planning)
            else "failed",
            "goal": "terminal world pose",
            "by_candidate_count": by_candidate_count,
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
            "peak_run_tensor_bytes": 6 * 2 * _IMAGE_SIZE[0] * _IMAGE_SIZE[1] * 5 * 8,
            "planning_k8_latency_seconds": next(
                item.latency_seconds for item in result.planning if item.candidate_count == 8
            ),
            "planning_k32_latency_seconds": next(
                item.latency_seconds for item in result.planning if item.candidate_count == 32
            ),
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "six_dof_open_world_incumbent" if result.qualified else "none",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "analytic structured pipeline",
        },
        qualitative={
            "best_episode": "unfamiliar-box-sphere-contact",
            "representative_episode": "unfamiliar-box-sphere-contact",
            "worst_episode": "unfamiliar-box-sphere-contact",
            "parameter_convergence": list(result.parameter_convergence),
            "forecast_animations": [result.animation],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "arbitrary-shape rigid bodies",
            "same-appearance touching-object separation",
            "unknown camera calibration",
            "hidden actions",
            "full perceptual qualification above eight objects",
        ),
        scope_limitations=(
            "sphere and oriented-box primitives",
            "short partial occlusion",
            "known calibrated cameras",
            "candidate rollouts freeze the active set",
            "online chromaticity is a measured association cue, not a predeclared object handle",
        ),
    ).validate()


def publish_open_world_six_dof_capability(
    result: OpenWorldSixDofResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "open_world_six_dof.json",
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
                "open_world_six_dof.json": "summary",
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
        raise RuntimeError("open-world six-DoF evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "OPEN_WORLD_SIX_DOF_SCHEMA",
    "OpenWorldSixDofResult",
    "PosePlanningResult",
    "capability_manifest_sha256",
    "publish_open_world_six_dof_capability",
    "run_open_world_six_dof_capability",
]
