"""Compact qualification for geometry-only touching discovery and long recovery."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from world_model.belief import RigidPrimitive, WorldBelief
from world_model.dynamics import DynamicsModel, WorldImpulseAction
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.observations.rgbd import (
    OpenWorldRigidFrame,
    OpenWorldRigidTracker,
    TrackedRigidObject,
    discover_rigid_objects_from_rgbd,
    tracked_objects_to_belief,
)
from world_model.planning import TerminalWorldPositionGoal, plan_counterfactual_actions
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

OPEN_WORLD_TOUCHING_RECOVERY_SCHEMA = "world_model_open_world_touching_recovery_v1"
_DTYPE = torch.float64
_IMAGE_SIZE = (96, 96)
_FRAME_DT = 0.05
_DROPOUT_FRAMES = 8
_BOUNDS = ((-4.0, 4.0), (-4.0, 4.0), (2.0, 6.0))


@dataclass(frozen=True, slots=True)
class RecoveryPlanningResult:
    candidate_count: int
    winner_correct: bool
    normalized_regret: float
    goal_success: bool
    serial_vectorized_parity: bool
    maximum_cost_difference: float
    source_unchanged: bool
    latency_seconds: float


@dataclass(frozen=True, slots=True)
class OpenWorldTouchingRecoveryResult:
    schema: str
    manifest_sha256: str
    object_count: int
    connected_depth_components: int
    geometry_split_count: int
    identical_appearance: bool
    appearance_weight: float
    primitive_weight: float
    dropout_frames: int
    all_tracks_retained: bool
    recovery_id_accuracy: float
    maximum_gap_position_rmse_m: float
    recovery_position_rmse_m: float
    recovery_velocity_rmse_mps: float
    duplicate_id_count: int
    planning: tuple[RecoveryPlanningResult, ...]
    maximum_perception_latency_seconds: float
    learned_weight_bytes: int
    evaluation_seconds: float
    recovery_curve: dict[str, float]
    animation: dict[str, Any]
    gate_failures: tuple[str, ...]
    qualified: bool
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capability_manifest_sha256() -> str:
    payload = {
        "runtime_inputs": ["rgb", "depth", "world_from_camera", "intrinsics", "timestamp"],
        "object_count": 3,
        "appearance": "one identical RGB albedo for every object",
        "separation": "public depth-support topology and calibrated metric geometry",
        "touching_requirement": "fewer connected depth components than recovered instances",
        "association_weights": {"appearance": 0.0, "primitive": 0.0, "geometry": 0.75},
        "dropout_frames": _DROPOUT_FRAMES,
        "planning_candidate_counts": [8, 32],
        "planning_latency_protocol": {
            "adjudication": "fresh_process_governed_runner",
            "statistic": "median",
            "repeats": 3,
            "limits_seconds": {"K8": 0.50, "K32": 0.65},
        },
        "planning_loss": False,
        "retained_media": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _quaternion_z(angle: float) -> Tensor:
    return torch.tensor(
        [0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)],
        dtype=_DTYPE,
    )


def _state(timestamp: float) -> RigidBodyState:
    first_extent = torch.tensor([0.32, 0.23, 0.19], dtype=_DTYPE)
    third_extent = torch.tensor([0.27, 0.19, 0.17], dtype=_DTYPE)
    base = torch.tensor(
        [[-0.31, -0.46, 4.0], [0.31, -0.46, 4.0], [0.0, 0.75, 4.10]],
        dtype=_DTYPE,
    )
    velocity = torch.tensor(
        [[-0.04, 0.02, 0.0], [0.05, 0.02, 0.0], [0.0, -0.025, 0.0]],
        dtype=_DTYPE,
    )
    spheres = SphereState(
        object_id=torch.tensor([731, 208, 944], dtype=torch.int64),
        active=torch.ones(3, dtype=torch.bool),
        position=base + velocity * timestamp,
        velocity=velocity,
        radius=torch.tensor(
            [
                [float(torch.linalg.vector_norm(first_extent))],
                [0.30],
                [float(torch.linalg.vector_norm(third_extent))],
            ],
            dtype=_DTYPE,
        ),
        mass=torch.ones(3, 1, dtype=_DTYPE),
        restitution=torch.full((3, 1), 0.5, dtype=_DTYPE),
        drag=torch.full((3, 1), 0.001, dtype=_DTYPE),
        friction=torch.full((3, 1), 0.25, dtype=_DTYPE),
        albedo=torch.full((3, 3), 0.54, dtype=_DTYPE),
        orientation=torch.stack(
            (
                _quaternion_z(0.18 + 0.25 * timestamp),
                _quaternion_z(0.0),
                _quaternion_z(-0.31 - 0.20 * timestamp),
            )
        ),
        angular_velocity=torch.tensor(
            [[0.0, 0.0, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, -0.20]],
            dtype=_DTYPE,
        ),
        sleeping=torch.zeros(3, dtype=torch.bool),
        sleep_counter=torch.zeros(3, dtype=torch.int64),
    )
    return replace(
        RigidBodyState.from_spheres(spheres),
        primitive=torch.tensor(
            [int(RigidPrimitive.BOX), int(RigidPrimitive.SPHERE), int(RigidPrimitive.BOX)],
            dtype=torch.int64,
        ),
        half_extents=torch.stack(
            (first_extent, torch.full((3,), 0.30, dtype=_DTYPE), third_extent)
        ),
    )


def _cameras(timestamp: float) -> tuple[CameraFrame, ...]:
    target = torch.tensor([0.0, 0.0, 4.0], dtype=_DTYPE)
    frames = []
    for offset in (
        (0.0, 0.0, -4.0),
        (0.0, 0.0, 4.0),
        (-4.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.0, -4.0, 0.0),
        (0.0, 4.0, 0.0),
    ):
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


def _observe(
    timestamp: float,
    *,
    missing: bool,
) -> tuple[OpenWorldRigidFrame, int, float]:
    cameras = _cameras(timestamp)
    if missing:
        rgb = torch.zeros((len(cameras), *_IMAGE_SIZE, 3), dtype=_DTYPE)
        depth = torch.zeros((len(cameras), *_IMAGE_SIZE), dtype=_DTYPE)
        connected_components = 0
    else:
        rendered = [
            render_rigid_bodies(_state(timestamp), camera, _IMAGE_SIZE) for camera in cameras
        ]
        rgb = torch.stack([item.rgb.to(_DTYPE).permute(1, 2, 0) for item in rendered])
        depth = torch.stack([item.depth_buffer for item in rendered])
        _, connected_components = label(
            (depth[0] > 0.0).numpy(),
            structure=np.ones((3, 3), dtype=np.int8),
        )
    started = time.perf_counter()
    frame = discover_rigid_objects_from_rgbd(
        rgb,
        depth,
        torch.stack([item.world_from_camera for item in cameras]),
        torch.stack([item.intrinsics for item in cameras]),
        timestamp=timestamp,
        segmentation_mode="depth_geometry",
    )
    return frame, int(connected_components), time.perf_counter() - started


def _truth_by_track(tracked: tuple[TrackedRigidObject, ...], truth: RigidBodyState) -> Tensor:
    observed = torch.stack([item.geometry.world_position for item in tracked])
    costs = torch.cdist(observed, truth.position).detach().numpy()
    rows, columns = linear_sum_assignment(costs)
    if len(rows) != len(tracked):
        raise RuntimeError("private evaluation assignment is incomplete")
    assignment = torch.empty(len(tracked), dtype=torch.int64)
    assignment[torch.from_numpy(rows)] = torch.from_numpy(columns)
    return assignment


def _rmse(
    tracked: tuple[TrackedRigidObject, ...], truth: RigidBodyState, truth_by_id: dict[int, int]
) -> float:
    errors = [
        (item.geometry.world_position - truth.position[truth_by_id[item.object_id]]).square()
        for item in tracked
    ]
    return float(torch.stack(errors).mean().sqrt())


def _dynamics(belief: WorldBelief) -> DynamicsModel:
    model = DynamicsModel.from_belief(
        belief,
        max_substep=1.0 / 20.0,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=16,
        interaction_radius=0.4,
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


def _reference_action_branch(
    truth: RigidBodyState,
    *,
    target_index: int,
    impulse: Tensor,
    action_time: float,
    horizon: float,
) -> RigidBodyState:
    physics = PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=_BOUNDS,
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )
    current, _ = advance_rigid_bodies_6dof(truth, action_time, physics)
    external = torch.zeros_like(current.velocity)
    external[target_index] = impulse
    current, _ = advance_rigid_bodies_6dof(
        current,
        horizon - action_time,
        physics,
        external_impulse=external,
    )
    return current


def _planning_result(
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_id: dict[int, int],
    dynamics: DynamicsModel,
    candidate_count: int,
) -> RecoveryPlanningResult:
    target_id = next(
        object_id for object_id, truth_index in truth_by_id.items() if truth_index == 2
    )
    target_truth_index = truth_by_id[target_id]
    impulses = [belief.objects.position.new_tensor([0.42, 0.24, 0.0])]
    for index in range(1, candidate_count):
        angle = 2.0 * math.pi * (index - 1) / max(candidate_count - 1, 1)
        impulses.append(
            belief.objects.position.new_tensor(
                [0.14 * math.cos(angle), 0.14 * math.sin(angle), 0.0]
            )
        )
    action_time = 0.05
    horizon = 0.50
    actions = tuple(
        WorldImpulseAction(
            timestamp=belief.timestamp + action_time,
            object_id=torch.tensor([target_id], dtype=torch.int64),
            impulse_world=impulse.reshape(1, 3),
        )
        for impulse in impulses
    )
    truth_branches = tuple(
        _reference_action_branch(
            truth,
            target_index=target_truth_index,
            impulse=impulse,
            action_time=action_time,
            horizon=horizon,
        )
        for impulse in impulses
    )
    target = truth_branches[0].position[target_truth_index]
    truth_cost = torch.stack(
        [(item.position[target_truth_index] - target).square().sum() for item in truth_branches]
    )
    goal = TerminalWorldPositionGoal(
        object_id=torch.tensor([target_id], dtype=torch.int64),
        position_world=target.reshape(1, 3).to(belief.objects.position),
    )
    source = belief.clone()
    with torch.no_grad():
        vectorized_measurements = []
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
            vectorized_measurements.append(time.perf_counter() - started)
        assert vectorized is not None
        latency = sorted(vectorized_measurements)[1]
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
    selected = int(vectorized.selected_index[0])
    scale = max(float(truth_cost.max()), 1.0e-12)
    selected_error = float(
        torch.linalg.vector_norm(truth_branches[selected].position[target_truth_index] - target)
    )
    return RecoveryPlanningResult(
        candidate_count=candidate_count,
        winner_correct=selected == int(truth_cost.argmin()),
        normalized_regret=float(truth_cost[selected] - truth_cost.min()) / scale,
        goal_success=selected_error <= 0.08,
        serial_vectorized_parity=torch.equal(vectorized.selected_index, serial.selected_index),
        maximum_cost_difference=float((vectorized.total_cost - serial.total_cost).abs().max()),
        source_unchanged=torch.equal(source.objects.position, belief.objects.position),
        latency_seconds=latency,
    )


def _animation(
    history: list[tuple[float, tuple[TrackedRigidObject, ...]]],
    truth_by_id: dict[int, int],
    recovery_rmse: float,
) -> dict[str, Any]:
    frames = []
    for frame_index, (timestamp, tracked) in enumerate(history):
        truth = _state(timestamp)
        model_points = []
        truth_points = []
        for item in tracked:
            truth_index = truth_by_id[item.object_id]
            position = item.geometry.world_position
            target = truth.position[truth_index]
            model_points.append(
                [item.object_id, round(float(position[0]), 5), round(float(position[1]), 5)]
            )
            truth_points.append(
                [item.object_id, round(float(target[0]), 5), round(float(target[1]), 5)]
            )
        frames.append(
            {
                "frame": frame_index,
                "time_s": round(frame_index * _FRAME_DT, 3),
                "model": model_points,
                "truth": truth_points,
            }
        )
    values_x = [
        point[1] for frame in frames for role in ("model", "truth") for point in frame[role]
    ]
    values_y = [
        point[2] for frame in frames for role in ("model", "truth") for point in frame[role]
    ]

    def bounds(values: list[float]) -> list[float]:
        padding = max(0.10 * (max(values) - min(values)), 0.10)
        return [round(min(values) - padding, 4), round(max(values) + padding, 4)]

    return {
        "schema": "world_model_compact_recovery_animation_v1",
        "label": "same-appearance touching split and eight-frame recovery",
        "episode": "geometry-only:touching-and-full-rgbd-gap",
        "object_count": 3,
        "contact": False,
        "dynamic_membership": False,
        "current_position_rmse_m": recovery_rmse,
        "mode": "tracking",
        "frame_rate": 1.0 / _FRAME_DT,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": bounds(values_x), "vertical": bounds(values_y)},
        "frames": frames,
        "events": [
            {"frame": 3, "kind": "complete RGB-D gap begins"},
            {"frame": 11, "kind": "IDs recovered after 8 missing frames"},
        ],
        "reference": "private positions opened only after geometry-only public tracking",
    }


def _gate_failures(
    values: dict[str, Any],
    planning: tuple[RecoveryPlanningResult, ...],
    *,
    enforce_latency: bool,
) -> tuple[str, ...]:
    failures: list[str] = []
    if values["object_count"] != 3:
        failures.append("object_count")
    if values["connected_depth_components"] >= values["geometry_split_count"]:
        failures.append("touching_silhouette_not_demonstrated")
    if not values["identical_appearance"]:
        failures.append("identical_appearance")
    if values["appearance_weight"] != 0.0 or values["primitive_weight"] != 0.0:
        failures.append("association_not_geometry_only")
    if values["dropout_frames"] != _DROPOUT_FRAMES or not values["all_tracks_retained"]:
        failures.append("long_gap_retention")
    if values["recovery_id_accuracy"] < 1.0 or values["duplicate_id_count"] != 0:
        failures.append("identity_recovery")
    if values["maximum_gap_position_rmse_m"] > 0.04:
        failures.append("gap_position_rmse")
    if values["recovery_position_rmse_m"] > 0.03:
        failures.append("recovery_position_rmse")
    if values["recovery_velocity_rmse_mps"] > 0.08:
        failures.append("recovery_velocity_rmse")
    if values["maximum_perception_latency_seconds"] > 1.0:
        failures.append("perception_latency")
    for item in planning:
        if not (
            item.winner_correct
            and item.normalized_regret <= 0.05
            and item.goal_success
            and item.serial_vectorized_parity
            and item.maximum_cost_difference <= 1.0e-6
            and item.source_unchanged
            and (
                not enforce_latency
                or item.latency_seconds <= (0.50 if item.candidate_count == 8 else 0.65)
            )
        ):
            failures.append(f"post_recovery_planning_k{item.candidate_count}")
    return tuple(failures)


def run_open_world_touching_recovery_capability(
    *, enforce_latency: bool = True
) -> OpenWorldTouchingRecoveryResult:
    """Run the same-appearance touching-instance and long-gap milestone.

    Disable latency adjudication only for in-process functional tests whose
    inherited PyTorch thread-pool and thermal state are not a controlled timing
    environment. Governed runners always leave it enabled.
    """

    started = time.perf_counter()
    tracker = OpenWorldRigidTracker(
        max_missed_steps=_DROPOUT_FRAMES,
        association_gate=2.0,
        appearance_weight=0.0,
        geometry_weight=0.75,
        primitive_weight=0.0,
    )
    timestamps = tuple(-0.10 + _FRAME_DT * index for index in range(13))
    history: list[tuple[float, tuple[TrackedRigidObject, ...]]] = []
    latencies = []
    connected_components = 0
    initial_truth_by_id: dict[int, int] | None = None
    recovery_curve: dict[str, float] = {}
    gap_errors: list[float] = []
    all_tracks_retained = True
    for frame_index, timestamp in enumerate(timestamps):
        missing = 3 <= frame_index < 3 + _DROPOUT_FRAMES
        frame, components, latency = _observe(timestamp, missing=missing)
        latencies.append(latency)
        tracked = tracker.update(frame)
        history.append((timestamp, tracked))
        truth = _state(timestamp)
        if frame_index == 2:
            assignment = _truth_by_track(tracked, truth)
            initial_truth_by_id = {
                item.object_id: int(assignment[index]) for index, item in enumerate(tracked)
            }
            connected_components = components
        if initial_truth_by_id is not None:
            position_rmse = _rmse(tracked, truth, initial_truth_by_id)
            recovery_curve[f"{frame_index * _FRAME_DT:.2f}"] = position_rmse
            if missing:
                gap_errors.append(position_rmse)
        if missing:
            all_tracks_retained &= len(tracked) == 3 and all(
                item.missed_steps == frame_index - 2 and not item.observed for item in tracked
            )
    if initial_truth_by_id is None:
        raise RuntimeError("initial public set did not become measurable")
    recovered = history[-1][1]
    recovery_truth = _state(timestamps[-1])
    recovered_assignment = _truth_by_track(recovered, recovery_truth)
    recovered_id_by_truth = {
        int(recovered_assignment[index]): item.object_id for index, item in enumerate(recovered)
    }
    initial_id_by_truth = {
        truth_index: object_id for object_id, truth_index in initial_truth_by_id.items()
    }
    id_accuracy = (
        sum(
            recovered_id_by_truth.get(index) == initial_id_by_truth.get(index) for index in range(3)
        )
        / 3.0
    )
    recovery_position = _rmse(recovered, recovery_truth, initial_truth_by_id)
    velocity_errors = [
        (item.velocity - recovery_truth.velocity[initial_truth_by_id[item.object_id]]).square()
        for item in recovered
    ]
    recovery_velocity = float(torch.stack(velocity_errors).mean().sqrt())
    belief = tracked_objects_to_belief(
        recovered,
        timestamp=timestamps[-1],
        max_objects=3,
        initial_mass=1.0,
        initial_restitution=0.5,
        initial_drag=0.001,
        initial_friction=0.25,
    )
    dynamics = _dynamics(belief)
    planning = tuple(
        _planning_result(
            belief,
            recovery_truth,
            initial_truth_by_id,
            dynamics,
            candidate_count,
        )
        for candidate_count in (8, 32)
    )
    runtime_parameters = inspect.signature(discover_rigid_objects_from_rgbd).parameters
    private_names = {"prototype", "instance_map", "object_id", "primitive_label", "truth"}
    identical_appearance = bool(
        torch.allclose(
            recovery_truth.albedo,
            recovery_truth.albedo[:1].expand_as(recovery_truth.albedo),
            rtol=0.0,
            atol=0.0,
        )
        and not private_names.intersection(runtime_parameters)
    )
    values = {
        "object_count": len(recovered),
        "connected_depth_components": connected_components,
        "geometry_split_count": len(history[2][1]),
        "identical_appearance": identical_appearance,
        "appearance_weight": tracker.appearance_weight,
        "primitive_weight": tracker.primitive_weight,
        "dropout_frames": _DROPOUT_FRAMES,
        "all_tracks_retained": all_tracks_retained,
        "recovery_id_accuracy": id_accuracy,
        "maximum_gap_position_rmse_m": max(gap_errors),
        "recovery_position_rmse_m": recovery_position,
        "recovery_velocity_rmse_mps": recovery_velocity,
        "duplicate_id_count": len(recovered) - len({item.object_id for item in recovered}),
        "maximum_perception_latency_seconds": max(latencies),
    }
    failures = _gate_failures(values, planning, enforce_latency=enforce_latency)
    animation = _animation(history, initial_truth_by_id, recovery_position)
    return OpenWorldTouchingRecoveryResult(
        schema=OPEN_WORLD_TOUCHING_RECOVERY_SCHEMA,
        manifest_sha256=capability_manifest_sha256(),
        planning=planning,
        learned_weight_bytes=sum(
            parameter.numel() * parameter.element_size() for parameter in dynamics.parameters()
        ),
        evaluation_seconds=time.perf_counter() - started,
        recovery_curve=recovery_curve,
        animation=animation,
        gate_failures=failures,
        qualified=not failures,
        **values,
    )


def _summary(
    result: OpenWorldTouchingRecoveryResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    by_candidate_count = {
        str(item.candidate_count): {
            "winner_accuracy": float(item.winner_correct),
            "median_normalized_regret": item.normalized_regret,
            "goal_success": float(item.goal_success),
        }
        for item in result.planning
    }
    candidate_score = (
        result.recovery_position_rmse_m / 0.03
        + result.maximum_gap_position_rmse_m / 0.04
        + (1.0 - result.recovery_id_accuracy)
    ) / 3.0
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.qualified else "failed",
        outcome="qualified_convergence" if result.qualified else "failed_to_improve",
        source_format=OPEN_WORLD_TOUCHING_RECOVERY_SCHEMA,
        configuration={
            "object_count": result.object_count,
            "segmentation_mode": "depth_geometry",
            "dropout_frames": result.dropout_frames,
            "appearance_weight": result.appearance_weight,
            "primitive_weight": result.primitive_weight,
            "planning_used_as_training_loss": False,
        },
        provenance={
            "scenario_manifest_sha256": result.manifest_sha256,
            "runtime_inputs": ["rgb", "depth", "calibration", "timestamp", "known actions"],
            "runtime_object_prototypes": False,
            "truth_opened_after_public_inference": True,
        },
        scores={
            "candidate": {"value": candidate_score, "supported_weight": 1.0},
            "incumbent": {"value": 1.0, "supported_weight": 1.0},
            "selected": "geometry_only_touching_recovery" if result.qualified else "incumbent",
        },
        factor_metrics={
            "same_appearance_touching_discovery": {
                "status": "passed"
                if result.connected_depth_components < result.geometry_split_count
                else "failed",
                "proposal_f1": 1.0 if result.object_count == 3 else 0.0,
                "current_position_rmse_m": result.recovery_position_rmse_m,
            },
            "eight_frame_identity_recovery": {
                "status": "passed" if result.recovery_id_accuracy == 1.0 else "failed",
                "identity_accuracy": result.recovery_id_accuracy,
                "current_position_rmse_m": result.recovery_position_rmse_m,
            },
        },
        cell_metrics={
            "N3/touching=1/same_appearance=1/gap=8": {
                "current_position_rmse_m": {
                    "value": result.recovery_position_rmse_m,
                    "support": 3,
                },
                "persistent_id_accuracy": {"value": result.recovery_id_accuracy, "support": 3},
                "proposal_f1": {"value": 1.0 if result.object_count == 3 else 0.0, "support": 3},
            }
        },
        horizon_curves={"recovery_position_rmse_m": result.recovery_curve},
        uncertainty={"status": "not promoted by this deterministic recovery milestone"},
        planning={
            "status": "passed"
            if all(item.winner_correct for item in result.planning)
            else "failed",
            "goal": "post-recovery terminal position",
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
            "perception_latency_seconds": result.maximum_perception_latency_seconds,
            "learned_weight_bytes": result.learned_weight_bytes,
            "evaluation_seconds": result.evaluation_seconds,
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "geometry_only_touching_recovery" if result.qualified else "none",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "public depth-support partition and metric association",
        },
        qualitative={
            "best_episode": "same-appearance-touching-recovery",
            "representative_episode": "same-appearance-touching-recovery",
            "worst_episode": "same-appearance-touching-recovery",
            "animations": [result.animation],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "flush featureless unions with no separable depth/silhouette evidence",
            "identity recovery through unobserved collisions or accelerations",
            "unknown camera calibration",
            "full multi-contact six-DoF visual qualification at N=4-8",
            "full perceptual qualification above eight objects",
        ),
        scope_limitations=(
            "sphere and oriented-box primitives",
            "calibrated six-view RGB-D",
            "eight-frame force-free complete observation gap",
            "geometry-distinguishable or motion-separable tracks",
            "planning remains a downstream test only",
        ),
    ).validate()


def publish_open_world_touching_recovery_capability(
    result: OpenWorldTouchingRecoveryResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "open_world_touching_recovery.json",
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
                "open_world_touching_recovery.json": "summary",
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
        raise RuntimeError("touching/recovery evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "OPEN_WORLD_TOUCHING_RECOVERY_SCHEMA",
    "OpenWorldTouchingRecoveryResult",
    "RecoveryPlanningResult",
    "capability_manifest_sha256",
    "publish_open_world_touching_recovery_capability",
    "run_open_world_touching_recovery_capability",
]
