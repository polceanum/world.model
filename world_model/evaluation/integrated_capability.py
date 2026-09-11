"""One compact integrated mixed-rigid, lifecycle, action, and contact benchmark."""

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

from world_model.belief import RigidPrimitive
from world_model.dynamics import WorldImpulseAction, WorldImpulseSchedule
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.evaluation.rigid_capability import (
    RigidPlanningResult,
    _belief_from_observations,
    _dynamics,
    _observe,
    _planning,
    _public_masks,
)
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

INTEGRATED_CAPABILITY_SCHEMA = "world_model_integrated_capability_v1"
_FRAME_DT = 0.05
_ANCHOR_FRAME = 7
_ANCHOR_TIME = _ANCHOR_FRAME * _FRAME_DT
_LONG_HORIZON = 4.0
_BOUNDS = ((-1.55, 1.55), (-1.45, 1.45), (3.30, 4.70))
_PROTOTYPES = torch.tensor(
    [
        [0.90, 0.16, 0.10],
        [0.10, 0.82, 0.24],
        [0.12, 0.34, 0.92],
        [0.90, 0.72, 0.10],
        [0.72, 0.14, 0.86],
    ],
    dtype=torch.float64,
)


@dataclass(frozen=True, slots=True)
class IntegratedActionSpec:
    offset_seconds: float
    target_slot: int
    impulse_world: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class IntegratedCapabilityResult:
    schema: str
    manifest_sha256: str
    lifecycle_f1: float
    persistent_id_accuracy: float
    occlusion_recovered: bool
    primitive_accuracy: float
    current_position_rmse_m: float
    box_half_extent_rmse_m: float
    two_second_position_rmse_m: float
    four_second_position_rmse_m: float
    two_second_velocity_rmse_mps: float
    four_second_velocity_rmse_mps: float
    collision_f1: float
    collision_timing_error_frames: int | None
    predicted_collision_frames: tuple[int, ...]
    reference_collision_frames: tuple[int, ...]
    known_action_count: int
    expected_action_count: int
    source_unchanged: bool
    finite: bool
    planning: tuple[RigidPlanningResult, ...]
    gate_failures: tuple[str, ...]
    qualified: bool
    evaluation_seconds: float
    learned_weight_bytes: int
    peak_run_tensor_bytes: int
    animation: dict[str, Any]
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "planning": [item.to_dict() for item in self.planning],
        }


def default_integrated_actions() -> tuple[IntegratedActionSpec, ...]:
    return (
        IntegratedActionSpec(0.475, 0, (0.0, 0.12, 0.0)),
        IntegratedActionSpec(1.375, 2, (-0.10, 0.0, 0.0)),
        IntegratedActionSpec(2.275, 1, (0.0, -0.10, 0.0)),
    )


def integrated_manifest_sha256() -> str:
    payload = {
        "anchor_frame": _ANCHOR_FRAME,
        "actions": [asdict(item) for item in default_integrated_actions()],
        "bounds": _BOUNDS,
        "lifecycle": {
            "birth_slot3": [1, 2],
            "remove_slot1": [3, 4],
            "rebirth_slot1": [5, 6],
            "occlude_slot0": 2,
        },
        "primitives": ["box", "sphere", "box", "sphere"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integrated_state(frame: int) -> RigidBodyState:
    timestamp = frame * _FRAME_DT
    position_at_anchor = torch.tensor(
        [[-0.72, 0.0, 4.0], [0.72, 0.0, 4.0], [0.0, 0.88, 4.0], [0.0, -0.88, 4.0]],
        dtype=torch.float64,
    )
    velocity = torch.tensor(
        [[0.72, 0.0, 0.0], [-0.72, 0.0, 0.0], [0.0, -0.36, 0.0], [0.0, 0.36, 0.0]],
        dtype=torch.float64,
    )
    position = position_at_anchor + velocity * (timestamp - _ANCHOR_TIME)
    active = torch.tensor([True, True, True, frame >= 1])
    object_id = torch.tensor([50, 51, 52, 53], dtype=torch.int64)
    albedo = _PROTOTYPES[:4].clone()
    if 3 <= frame < 5:
        active[1] = False
        object_id[1] = -1
    elif frame >= 5:
        object_id[1] = 61
        albedo[1] = _PROTOTYPES[4]
    object_id = torch.where(active, object_id, torch.full_like(object_id, -1))
    primitive = torch.tensor(
        [
            int(RigidPrimitive.BOX),
            int(RigidPrimitive.SPHERE),
            int(RigidPrimitive.BOX),
            int(RigidPrimitive.SPHERE),
        ],
        dtype=torch.int64,
    )
    half_extents = torch.tensor(
        [[0.34, 0.22, 0.17], [0.23, 0.23, 0.23], [0.29, 0.20, 0.15], [0.21, 0.21, 0.21]],
        dtype=torch.float64,
    )
    radius = torch.tensor([[0.0], [0.23], [0.0], [0.21]], dtype=torch.float64)
    radius = torch.where(
        (primitive == int(RigidPrimitive.BOX)).unsqueeze(-1),
        torch.linalg.vector_norm(half_extents, dim=-1, keepdim=True),
        radius,
    )
    angles = (15.0, 0.0, -19.0, 0.0)
    orientations = []
    for angle in angles:
        half = 0.5 * math.radians(angle)
        orientations.append((0.0, 0.0, math.sin(half), math.cos(half)))
    spheres = SphereState(
        object_id=object_id,
        active=active,
        position=position,
        velocity=velocity,
        radius=radius,
        mass=torch.ones(4, 1, dtype=torch.float64),
        restitution=torch.full((4, 1), 0.74, dtype=torch.float64),
        drag=torch.full((4, 1), 0.018, dtype=torch.float64),
        friction=torch.full((4, 1), 1.0e-8, dtype=torch.float64),
        albedo=albedo,
        orientation=torch.tensor(orientations, dtype=torch.float64),
        angular_velocity=torch.zeros(4, 3, dtype=torch.float64),
        sleeping=torch.zeros(4, dtype=torch.bool),
        sleep_counter=torch.zeros(4, dtype=torch.int64),
    )
    return replace(
        RigidBodyState.from_spheres(spheres),
        primitive=primitive,
        half_extents=half_extents,
    )


def _moving_camera(frame: int) -> CameraFrame:
    timestamp = frame * _FRAME_DT
    target = torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64)
    position = target + torch.tensor(
        [0.30 * math.sin(1.8 * timestamp), 0.12 * math.cos(timestamp), -3.3],
        dtype=torch.float64,
    )
    world_from_camera = look_at_world_from_camera(position, target)
    return CameraFrame(
        timestamp=timestamp,
        world_from_camera=world_from_camera,
        camera_from_world=invert_rigid_transform(world_from_camera),
        intrinsics=make_intrinsics((81, 81), 38.0, dtype=torch.float64),
        position=position,
        target=target,
    )


def _observed_prototypes(state: RigidBodyState, frame: int) -> set[int]:
    camera = _moving_camera(frame)
    rendered = render_rigid_bodies(state, camera, (81, 81))
    rgb = rendered.rgb.to(state.position)
    depth = rendered.depth_buffer
    if frame == 2:
        hidden_active = state.active.clone()
        hidden_ids = state.object_id.clone()
        hidden_active[0] = False
        hidden_ids[0] = -1
        hidden = render_rigid_bodies(
            replace(state, active=hidden_active, object_id=hidden_ids),
            camera,
            (81, 81),
        )
        target = rendered.instance_slot_map == 0
        rgb[:, target] = hidden.rgb[:, target].to(rgb)
        depth[target] = hidden.depth_buffer[target]
    masks = _public_masks(rgb, depth, _PROTOTYPES)
    return {index for index in range(len(_PROTOTYPES)) if int(masks[index].sum()) >= 4}


def _lifecycle_metrics() -> tuple[float, float, bool]:
    active: dict[int, int] = {}
    tentative: dict[int, int] = {}
    misses: dict[int, int] = {}
    next_id = 1000
    true_positive = false_positive = false_negative = 0
    stable_ids: dict[int, int] = {}
    identity_correct = identity_total = 0
    before_occlusion: int | None = None
    after_occlusion: int | None = None
    expected_by_frame = (
        set(),
        {0, 1, 2},
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {0, 2, 3},
        {0, 2, 3},
        {0, 2, 3, 4},
        {0, 2, 3, 4},
    )
    for frame in range(_ANCHOR_FRAME + 1):
        observed = _observed_prototypes(_integrated_state(frame), frame)
        for handle in list(active):
            if handle in observed:
                misses[handle] = 0
            else:
                misses[handle] = misses.get(handle, 0) + 1
                if misses[handle] >= 2:
                    active.pop(handle)
                    misses.pop(handle, None)
        for handle in observed:
            if handle in active:
                continue
            tentative[handle] = tentative.get(handle, 0) + 1
            if tentative[handle] >= 2:
                active[handle] = next_id
                stable_ids.setdefault(handle, next_id)
                next_id += 1
                tentative.pop(handle, None)
                misses[handle] = 0
        for handle in list(tentative):
            if handle not in observed:
                tentative.pop(handle)
        predicted = set(active)
        expected = expected_by_frame[frame]
        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        for handle in predicted & expected:
            identity_total += 1
            identity_correct += int(active[handle] == stable_ids[handle])
        if frame == 1:
            before_occlusion = active.get(0)
        if frame == 3:
            after_occlusion = active.get(0)
    denominator = 2 * true_positive + false_positive + false_negative
    lifecycle_f1 = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
    return (
        lifecycle_f1,
        identity_correct / max(identity_total, 1),
        before_occlusion is not None and after_occlusion == before_occlusion,
    )


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=_BOUNDS,
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )


def _model_schedule(belief: Any) -> WorldImpulseSchedule:
    return WorldImpulseSchedule(
        tuple(
            WorldImpulseAction(
                timestamp=belief.timestamp + action.offset_seconds,
                object_id=belief.objects.object_id[:, action.target_slot].clone(),
                impulse_world=belief.objects.position.new_tensor([action.impulse_world]),
            )
            for action in default_integrated_actions()
        )
    )


def _apply_reference_impulse(state: RigidBodyState, action: IntegratedActionSpec) -> RigidBodyState:
    velocity = state.velocity.clone()
    impulse = state.velocity.new_tensor(action.impulse_world)
    velocity[action.target_slot] += impulse / state.mass[action.target_slot, 0]
    return replace(state, velocity=velocity)


def _reference_rollout(
    anchor: RigidBodyState,
    query_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = anchor
    current = 0.0
    action_index = 0
    actions = default_integrated_actions()
    positions = []
    velocities = []
    collisions = []
    for endpoint_tensor in query_times:
        endpoint = float(endpoint_tensor)
        interval_collision = False
        while action_index < len(actions) and actions[action_index].offset_seconds <= endpoint:
            action = actions[action_index]
            if action.offset_seconds > current:
                state, events = advance_rigid_bodies(
                    state, action.offset_seconds - current, _physics()
                )
                interval_collision |= bool(
                    events.pair_collision.any() | events.boundary_collision.any()
                )
            state = _apply_reference_impulse(state, action)
            current = action.offset_seconds
            action_index += 1
        if endpoint > current:
            state, events = advance_rigid_bodies(state, endpoint - current, _physics())
            interval_collision |= bool(
                events.pair_collision.any() | events.boundary_collision.any()
            )
        current = endpoint
        positions.append(state.position)
        velocities.append(state.velocity)
        collisions.append(interval_collision)
    return (
        torch.stack(positions),
        torch.stack(velocities),
        torch.tensor(collisions, dtype=torch.bool),
    )


def _event_metrics(predicted: torch.Tensor, target: torch.Tensor) -> tuple[float, int | None]:
    true_positive = int((predicted & target).sum())
    false_positive = int((predicted & ~target).sum())
    false_negative = int((~predicted & target).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
    predicted_index = torch.where(predicted)[0]
    target_index = torch.where(target)[0]
    if not len(predicted_index) and not len(target_index):
        return f1, 0
    if not len(predicted_index) or not len(target_index):
        return f1, None
    return f1, max(int((predicted_index - index).abs().min()) for index in target_index)


def _gate_failures(
    values: dict[str, Any], planning: tuple[RigidPlanningResult, ...]
) -> tuple[str, ...]:
    failures = []
    limits = {
        "lifecycle_f1": (0.95, "minimum"),
        "persistent_id_accuracy": (0.99, "minimum"),
        "primitive_accuracy": (1.0, "minimum"),
        "current_position_rmse_m": (0.020, "maximum"),
        "box_half_extent_rmse_m": (0.040, "maximum"),
        "two_second_position_rmse_m": (0.120, "maximum"),
        "four_second_position_rmse_m": (0.200, "maximum"),
        "collision_f1": (0.85, "minimum"),
    }
    for name, (limit, direction) in limits.items():
        value = float(values[name])
        if (direction == "minimum" and value < limit) or (direction == "maximum" and value > limit):
            failures.append(name)
    if (
        values["collision_timing_error_frames"] is None
        or values["collision_timing_error_frames"] > 1
    ):
        failures.append("collision_timing_error_frames")
    for name in ("occlusion_recovered", "source_unchanged", "finite"):
        if not values[name]:
            failures.append(name)
    if values["known_action_count"] != values["expected_action_count"]:
        failures.append("known_action_count")
    for candidate_count in (8, 32):
        selected = [item for item in planning if item.candidate_count == candidate_count]
        if len(selected) != 1:
            failures.append(f"planning_k{candidate_count}")
            continue
        item = selected[0]
        if not (
            item.winner_correct
            and item.goal_success
            and item.normalized_regret <= (0.05 if candidate_count == 8 else 0.07)
            and item.serial_vectorized_parity
            and item.maximum_cost_difference <= 1.0e-6
            and item.known_action_count == 1
            and item.pre_action_invariant
            and item.action_target_isolated
            and item.source_unchanged
            and item.latency_seconds <= (0.10 if candidate_count == 8 else 0.35)
        ):
            failures.append(f"planning_k{candidate_count}")
    return tuple(failures)


def run_integrated_capability() -> IntegratedCapabilityResult:
    """Run the integrated compact scenario and downstream planning gates."""

    started = time.perf_counter()
    lifecycle_f1, identity_accuracy, occlusion_recovered = _lifecycle_metrics()
    # All tracks have three post-lifecycle observations by frames 5--7.  Use
    # the two most recent native-rate samples for direct velocity evidence;
    # the earlier sample has a publicly measurable partial-surface residual.
    previous_truth = _integrated_state(_ANCHOR_FRAME - 1)
    anchor_truth = _integrated_state(_ANCHOR_FRAME)
    previous = _observe(previous_truth)
    current = _observe(anchor_truth)
    belief = _belief_from_observations(
        previous,
        current,
        anchor_truth,
        observation_dt=_FRAME_DT,
    )
    dynamics = _dynamics(belief, world_bounds=_BOUNDS)
    source = belief.clone()
    schedule = _model_schedule(belief)
    query_times = torch.arange(
        _FRAME_DT,
        _LONG_HORIZON + 0.5 * _FRAME_DT,
        _FRAME_DT,
        dtype=belief.dtype,
    )
    with torch.no_grad():
        prediction = dynamics.rollout(belief, query_times, action=schedule)
    truth_position, truth_velocity, truth_collision = _reference_rollout(
        anchor_truth, query_times.to(torch.float64)
    )
    model_position = prediction.positions[0]
    model_velocity = prediction.velocities[0]
    position_error = (
        (model_position.to(torch.float64) - truth_position).square().mean(dim=(-2, -1)).sqrt()
    )
    velocity_error = (
        (model_velocity.to(torch.float64) - truth_velocity).square().mean(dim=(-2, -1)).sqrt()
    )
    predicted_collision = prediction.auxiliary["pair_collision"][0].any(
        dim=(-2, -1)
    ) | prediction.auxiliary["boundary_collision"][0].any(dim=(-2, -1))
    collision_f1, collision_timing = _event_metrics(predicted_collision, truth_collision)
    primitive_accuracy = float(
        (torch.stack([item.primitive for item in current]) == anchor_truth.primitive)
        .to(torch.float32)
        .mean()
    )
    observed_position = torch.stack([item.world_position for item in current])
    current_rmse = float((observed_position - anchor_truth.position).square().mean().sqrt())
    extent_errors = [
        (measurement.half_extents.sort().values - anchor_truth.half_extents[index].sort().values)
        .square()
        .mean()
        for index, measurement in enumerate(current)
        if int(anchor_truth.primitive[index]) == int(RigidPrimitive.BOX)
    ]
    extent_rmse = float(torch.stack(extent_errors).mean().sqrt())
    planning_dynamics = _dynamics(belief)
    planning = tuple(
        _planning("integrated-mixed-rigid", belief, anchor_truth, planning_dynamics, count)
        for count in (8, 32)
    )
    known_count_tensor = prediction.auxiliary.get("known_action_count")
    known_count = (
        int(known_count_tensor.sum())
        if known_count_tensor is not None
        else int(prediction.auxiliary["known_action_applied"].sum())
    )
    finite = bool(
        torch.isfinite(model_position).all()
        and torch.isfinite(prediction.velocities).all()
        and torch.isfinite(position_error).all()
    )
    values = {
        "lifecycle_f1": lifecycle_f1,
        "persistent_id_accuracy": identity_accuracy,
        "occlusion_recovered": occlusion_recovered,
        "primitive_accuracy": primitive_accuracy,
        "current_position_rmse_m": current_rmse,
        "box_half_extent_rmse_m": extent_rmse,
        "two_second_position_rmse_m": float(position_error[39]),
        "four_second_position_rmse_m": float(position_error[-1]),
        "two_second_velocity_rmse_mps": float(velocity_error[39]),
        "four_second_velocity_rmse_mps": float(velocity_error[-1]),
        "collision_f1": collision_f1,
        "collision_timing_error_frames": collision_timing,
        "predicted_collision_frames": tuple(
            int(index + 1) for index in torch.where(predicted_collision)[0]
        ),
        "reference_collision_frames": tuple(
            int(index + 1) for index in torch.where(truth_collision)[0]
        ),
        "known_action_count": known_count,
        "expected_action_count": len(default_integrated_actions()),
        "source_unchanged": torch.equal(source.objects.position, belief.objects.position)
        and torch.equal(source.objects.velocity, belief.objects.velocity),
        "finite": finite,
    }
    failures = _gate_failures(values, planning)
    frames = []
    animation_indices = list(range(0, len(query_times), 4))
    if animation_indices[-1] != len(query_times) - 1:
        animation_indices.append(len(query_times) - 1)
    for index in animation_indices:
        frames.append(
            {
                "frame": index + 1,
                "time_s": round(float(query_times[index]), 3),
                "truth": [
                    [1000 + slot, round(float(point[0]), 5), round(float(point[1]), 5)]
                    for slot, point in enumerate(truth_position[index])
                ],
                "model": [
                    [1000 + slot, round(float(point[0]), 5), round(float(point[1]), 5)]
                    for slot, point in enumerate(model_position[index].detach())
                ],
            }
        )
    all_x = [point[1] for frame in frames for role in ("truth", "model") for point in frame[role]]
    all_y = [point[2] for frame in frames for role in ("truth", "model") for point in frame[role]]

    def bounds(values: list[float]) -> list[float]:
        padding = max(0.08 * (max(values) - min(values)), 0.10)
        return [round(min(values) - padding, 4), round(max(values) + padding, 4)]

    predicted_collision_frames = set(values["predicted_collision_frames"])
    reference_collision_frames = set(values["reference_collision_frames"])
    contact_events = []
    for frame in sorted(predicted_collision_frames | reference_collision_frames):
        if frame in predicted_collision_frames and frame in reference_collision_frames:
            kind = "model/reference contact"
        elif frame in predicted_collision_frames:
            kind = "model-only contact"
        else:
            kind = "reference-only contact"
        contact_events.append({"frame": frame, "kind": kind})
    animation = {
        "schema": "world_model_compact_forecast_animation_v1",
        "label": "integrated mixed rigid four second forecast",
        "episode": "integrated-mixed-rigid",
        "object_count": 4,
        "contact": True,
        "dynamic_membership": True,
        "endpoint_position_rmse_m": values["four_second_position_rmse_m"],
        "long_horizon_endpoint_s": 4.0,
        "mode": "forecast",
        "anchor_frame": 0,
        "frame_rate": 20.0,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "bounds": {"horizontal": bounds(all_x), "vertical": bounds(all_y)},
        "frames": frames,
        "events": [
            *[
                {
                    "frame": action.offset_seconds / _FRAME_DT,
                    "time_s": action.offset_seconds,
                    "kind": "known action",
                }
                for action in default_integrated_actions()
            ],
            *contact_events,
        ],
        "reference": "independent rigid truth opened after public RGB-D belief construction",
    }
    learned_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in dynamics.parameters()
    )
    return IntegratedCapabilityResult(
        schema=INTEGRATED_CAPABILITY_SCHEMA,
        manifest_sha256=integrated_manifest_sha256(),
        planning=planning,
        gate_failures=failures,
        qualified=not failures,
        evaluation_seconds=time.perf_counter() - started,
        learned_weight_bytes=learned_bytes,
        peak_run_tensor_bytes=6 * 4 * 81 * 81 * 5 * 8,
        animation=animation,
        **values,
    )


def _summary(
    result: IntegratedCapabilityResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    planning_by_k = {
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
        source_format=INTEGRATED_CAPABILITY_SCHEMA,
        configuration={
            "object_count": 4,
            "primitives": ["sphere", "oriented_box"],
            "changing_membership": True,
            "moving_calibrated_camera": True,
            "short_occlusion": True,
            "known_action_schedule_length": result.expected_action_count,
            "planning_used_as_training_loss": False,
            "generated_frames_retained": False,
        },
        provenance={
            "scenario_manifest_sha256": result.manifest_sha256,
            "truth_oracle": "independent world_model.simulator.rigid_physics",
            "forecast_animation_source_run": run_id,
        },
        scores={
            "candidate": result.current_position_rmse_m + result.four_second_position_rmse_m,
            "incumbent": result.current_position_rmse_m + result.four_second_position_rmse_m,
            "selected": "incumbent",
        },
        factor_metrics={
            "integrated_mixed_rigid": {
                "status": "passed" if result.qualified else "failed",
                "identity_accuracy": result.persistent_id_accuracy,
                "lifecycle_f1": result.lifecycle_f1,
                "primitive_accuracy": result.primitive_accuracy,
                "current_position_rmse_m": result.current_position_rmse_m,
                "two_second_position_rmse_m": result.two_second_position_rmse_m,
                "four_second_position_rmse_m": result.four_second_position_rmse_m,
                "four_second_velocity_rmse_mps": result.four_second_velocity_rmse_mps,
                "collision_f1": result.collision_f1,
            }
        },
        cell_metrics={
            "N4/integrated/mixed_rigid": {
                "current_position_rmse_m": {
                    "value": result.current_position_rmse_m,
                    "support": 4,
                },
                "two_second_position_rmse_m": {
                    "value": result.two_second_position_rmse_m,
                    "support": 4,
                },
                "collision_f1": {"value": result.collision_f1, "support": 5},
                "persistent_id_accuracy": {
                    "value": result.persistent_id_accuracy,
                    "support": 4,
                },
                "lifecycle_f1": {"value": result.lifecycle_f1, "support": 8},
            }
        },
        horizon_curves={
            "candidate_position_rmse_m": {
                "0": result.current_position_rmse_m,
                "2": result.two_second_position_rmse_m,
                "4": result.four_second_position_rmse_m,
            },
            "candidate_velocity_rmse_mps": {
                "2": result.two_second_velocity_rmse_mps,
                "4": result.four_second_velocity_rmse_mps,
            },
        },
        uncertainty={"status": "not re-estimated in integrated structured benchmark"},
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
            "evaluation_seconds": result.evaluation_seconds,
            "learned_weight_bytes": result.learned_weight_bytes,
            "peak_run_tensor_bytes": result.peak_run_tensor_bytes,
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "incumbent",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "integrated structured behavior",
        },
        qualitative={
            "best_episode": "integrated-mixed-rigid",
            "representative_episode": "integrated-mixed-rigid",
            "worst_episode": "integrated-mixed-rigid",
            "animations": [],
            "forecast_animations": [result.animation],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "unknown camera calibration",
            "hidden actions",
            "articulated or deformable bodies",
            "long-term scene memory",
        ),
        scope_limitations=(
            "four-object integrated scene",
            "known appearance handles",
            "central low-friction rigid contact",
            "candidate rollouts freeze the active set",
        ),
    ).validate()


def publish_integrated_capability(
    result: IntegratedCapabilityResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "integrated_capability.json",
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
                "integrated_capability.json": "summary",
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
        raise RuntimeError("integrated capability evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "INTEGRATED_CAPABILITY_SCHEMA",
    "IntegratedActionSpec",
    "IntegratedCapabilityResult",
    "default_integrated_actions",
    "integrated_manifest_sha256",
    "publish_integrated_capability",
    "run_integrated_capability",
]
