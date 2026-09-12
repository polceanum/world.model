"""Integrated public RGB-D to N=4/6/8 multi-contact dynamics qualification."""

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
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from world_model.belief import RigidPrimitive, WorldBelief
from world_model.dynamics import (
    DynamicsModel,
    WorldImpulseAction,
    WorldImpulseSchedule,
    quaternion_geodesic_distance,
)
from world_model.evaluation import multicontact_six_dof as state_gate
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
    RigidBodyState,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    render_rigid_bodies,
)
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

VISUAL_DYNAMIC_SCALE_SCHEMA = "world_model_visual_dynamic_scale_v1"
_COUNTS = (4, 6, 8)
_DTYPE = torch.float64
_IMAGE_SIZE = (96, 96)
_OBSERVATION_DT = 0.05
_OBSERVATION_FRAMES = 8
_FORECAST_DT = 0.05
_FORECAST_SECONDS = 2.0
_PALETTE = torch.tensor(
    [
        [0.90, 0.16, 0.12],
        [0.12, 0.82, 0.22],
        [0.12, 0.30, 0.92],
        [0.92, 0.76, 0.12],
        [0.72, 0.16, 0.86],
        [0.10, 0.82, 0.84],
        [0.94, 0.46, 0.12],
        [0.55, 0.55, 0.58],
    ],
    dtype=_DTYPE,
)


@dataclass(frozen=True, slots=True)
class VisualDynamicScenarioResult:
    object_count: int
    proposal_precision: float
    proposal_recall: float
    proposal_f1: float
    exact_visible_count_accuracy: float
    persistent_id_accuracy: float
    lifecycle_f1: float
    two_frame_birth_confirmation: bool
    two_miss_removal: bool
    partial_visibility_recovered: bool
    primitive_accuracy: float
    current_position_rmse_m: float
    current_orientation_rmse_degrees: float
    known_action_count: int
    expected_action_count: int
    unique_reference_contact_pairs: int
    unique_model_contact_pairs: int
    contact_pair_f1: float
    repeated_contact_frame_f1: float
    simultaneous_contact_frames: int
    first_contact_timing_error_frames: int | None
    maximum_position_rmse_m: float
    endpoint_position_rmse_m: float
    maximum_velocity_rmse_mps: float
    maximum_orientation_rmse_degrees: float
    rollout_latency_seconds: float
    source_unchanged: bool
    finite: bool
    truth_count_trace: tuple[int, ...]
    proposal_count_trace: tuple[int, ...]
    belief_count_trace: tuple[int, ...]
    position_curve: dict[str, float]
    animation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VisualDynamicScaleResult:
    schema: str
    manifest_sha256: str
    scenarios: tuple[VisualDynamicScenarioResult, ...]
    planning: tuple[state_gate.MultiContactPlanningResult, ...]
    gate_failures: tuple[str, ...]
    qualified: bool
    learned_weight_bytes: int
    peak_run_tensor_bytes: int
    evaluation_seconds: float
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def visual_dynamic_manifest_sha256() -> str:
    payload = {
        "runtime_inputs": ["rgb", "depth", "world_from_camera", "intrinsics", "timestamp"],
        "object_counts": list(_COUNTS),
        "primitives": "alternating unfamiliar oriented boxes and spheres",
        "box_half_extents_m": [0.22, 0.14, 0.10],
        "observation": {
            "frames": _OBSERVATION_FRAMES,
            "frame_dt_seconds": _OBSERVATION_DT,
            "birth_confirmation_observations": 2,
            "retirement_visible_misses": 2,
            "partial_visibility_frame": 2,
            "birth_last_slot_frames": [3, 4],
            "remove_slot1_frames": [2, 3],
            "replacement_slot1_frames": [4, 5],
            "moving_calibrated_cameras": 6,
        },
        "forecast": {
            "horizon_seconds": _FORECAST_SECONDS,
            "frame_dt_seconds": _FORECAST_DT,
            "actions": {
                str(count): [asdict(item) for item in state_gate.default_actions(count)]
                for count in _COUNTS
            },
            "required_contact_behavior": [
                "all adjacent pairs",
                "simultaneous contacts",
                "repeated contacts",
            ],
        },
        "planning": {
            "object_count": 8,
            "candidate_counts": [8, 32],
            "serial_oracle": True,
            "planning_loss": False,
        },
        "physics_prior": {
            "mass": 1.0,
            "restitution": 0.5,
            "drag": 0.05,
            "friction": 0.25,
        },
        "retained_media": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _scene_state(object_count: int, frame: int = _OBSERVATION_FRAMES - 1) -> RigidBodyState:
    state = state_gate._state(object_count)
    half_extents = state.half_extents.clone()
    box_mask = state.primitive == int(RigidPrimitive.BOX)
    half_extents[box_mask] = state.position.new_tensor([0.22, 0.14, 0.10])
    radius = state.radius.clone()
    radius[box_mask] = torch.linalg.vector_norm(half_extents[box_mask], dim=-1, keepdim=True)
    active = state.active.clone()
    object_id = torch.arange(5000, 5000 + object_count, dtype=torch.int64)
    if frame <= 2:
        active[-1] = False
    if frame in {2, 3}:
        active[1] = False
    elif frame >= 4:
        object_id[1] = 6001
    object_id = torch.where(active, object_id, torch.full_like(object_id, -1))
    return replace(
        state,
        object_id=object_id,
        active=active,
        albedo=_PALETTE[:object_count].clone(),
        mass=torch.ones_like(state.mass),
        restitution=torch.full_like(state.restitution, 0.5),
        drag=torch.full_like(state.drag, 0.05),
        friction=torch.full_like(state.friction, 0.25),
        half_extents=half_extents,
        radius=radius,
    )


def _cameras(timestamp: float) -> tuple[CameraFrame, ...]:
    target = torch.tensor([0.0, 0.0, 4.0], dtype=_DTYPE)
    phase = 0.35 * (timestamp + (_OBSERVATION_FRAMES - 1) * _OBSERVATION_DT)
    cosine = math.cos(phase)
    sine = math.sin(phase)
    frames = []
    for offset in (
        (0.0, 0.0, -4.0),
        (0.0, 0.0, 4.0),
        (-4.2, 0.0, 0.0),
        (4.2, 0.0, 0.0),
        (0.0, -4.2, 0.0),
        (0.0, 4.2, 0.0),
    ):
        x, y, z = offset
        rotated = target.new_tensor([cosine * x - sine * y, sine * x + cosine * y, z])
        translation = target.new_tensor(
            [0.06 * math.sin(1.3 * timestamp), 0.04 * math.cos(1.1 * timestamp), 0.0]
        )
        position = target + rotated + translation
        world_up = target.new_tensor([0.0, 0.0, 1.0]) if abs(z) < 3.0 else None
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


def _observe(state: RigidBodyState, frame: int) -> OpenWorldRigidFrame:
    timestamp = (frame - (_OBSERVATION_FRAMES - 1)) * _OBSERVATION_DT
    cameras = _cameras(timestamp)
    rendered = [render_rigid_bodies(state, camera, _IMAGE_SIZE) for camera in cameras]
    if frame == 2:
        hidden_active = state.active.clone()
        hidden_ids = state.object_id.clone()
        hidden_active[0] = False
        hidden_ids[0] = -1
        hidden_state = replace(state, active=hidden_active, object_id=hidden_ids)
        hidden = [render_rigid_bodies(hidden_state, camera, _IMAGE_SIZE) for camera in cameras[:4]]
        for view, background in enumerate(hidden):
            target = rendered[view].instance_slot_map == 0
            rendered[view].rgb[:, target] = background.rgb[:, target]
            rendered[view].depth_buffer[target] = background.depth_buffer[target]
    return discover_rigid_objects_from_rgbd(
        torch.stack([item.rgb.to(_DTYPE).permute(1, 2, 0) for item in rendered]),
        torch.stack([item.depth_buffer for item in rendered]),
        torch.stack([item.world_from_camera for item in cameras]),
        torch.stack([item.intrinsics for item in cameras]),
        timestamp=timestamp,
        box_decision_ratio=0.9,
    )


def _assignment(estimated: Tensor, truth: Tensor, *, gate: float = 0.12) -> tuple[Tensor, Tensor]:
    if not len(estimated) or not len(truth):
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    rows, columns = linear_sum_assignment(torch.cdist(estimated, truth).detach().cpu().numpy())
    rows_tensor = torch.as_tensor(rows, dtype=torch.int64)
    columns_tensor = torch.as_tensor(columns, dtype=torch.int64)
    distance = torch.linalg.vector_norm(estimated[rows_tensor] - truth[columns_tensor], dim=-1)
    accepted = distance <= gate
    return rows_tensor[accepted], columns_tensor[accepted]


def _truth_by_track(tracked: tuple[TrackedRigidObject, ...], truth: RigidBodyState) -> Tensor:
    rows, columns = _assignment(
        torch.stack([item.geometry.world_position for item in tracked]),
        truth.position[truth.active],
    )
    truth_slots = torch.where(truth.active)[0]
    if len(rows) != len(tracked) or len(columns.unique()) != len(tracked):
        raise RuntimeError("public tracks do not map one-to-one onto the evaluation reference")
    ordered = torch.empty(len(tracked), dtype=torch.int64)
    ordered[rows] = truth_slots[columns]
    return ordered


def _model_schedule(
    belief: WorldBelief,
    truth_by_slot: Tensor,
    object_count: int,
) -> WorldImpulseSchedule:
    actions = []
    for spec in state_gate.default_actions(object_count):
        model_slot = int(torch.where(truth_by_slot == spec.target_slot)[0][0])
        actions.append(
            WorldImpulseAction(
                timestamp=belief.timestamp.new_tensor([spec.offset_seconds]),
                object_id=belief.objects.object_id[:, model_slot].clone(),
                impulse_world=belief.timestamp.new_tensor([spec.impulse_world]),
            )
        )
    return WorldImpulseSchedule(tuple(actions))


def _animation(
    object_count: int,
    belief: WorldBelief,
    truth_by_slot: Tensor,
    query_times: Tensor,
    model_position: Tensor,
    reference_position: Tensor,
    model_orientation: Tensor,
    reference_orientation: Tensor,
    reference_collision: Tensor,
    position_error: Tensor,
) -> dict[str, Any]:
    action_frames = {
        max(0, round(spec.offset_seconds / _FORECAST_DT) - 1): spec
        for spec in state_gate.default_actions(object_count)
    }
    contact_frames = set(torch.where(reference_collision.any(dim=(-2, -1)))[0].tolist())
    indices = sorted(set(range(0, len(query_times), 2)) | contact_frames | set(action_frames))
    if indices[-1] != len(query_times) - 1:
        indices.append(len(query_times) - 1)
    runtime_ids = belief.objects.object_id[0].tolist()
    primitives = state_gate.RigidGeometryCodec.primitive(belief.objects.geometry[0])
    frames = []
    for index in indices:
        truth_points = []
        model_points = []
        for slot, point in enumerate(reference_position[index]):
            truth_point = [runtime_ids[slot], round(float(point[0]), 5), round(float(point[1]), 5)]
            model_point = [
                runtime_ids[slot],
                round(float(model_position[index, slot, 0]), 5),
                round(float(model_position[index, slot, 1]), 5),
            ]
            if int(primitives[slot]) == int(RigidPrimitive.BOX):
                for output, quaternion in (
                    (truth_point, reference_orientation[index, slot]),
                    (model_point, model_orientation[index, slot]),
                ):
                    x, y, z, w = (float(value) for value in quaternion)
                    output.append(
                        round(
                            math.atan2(
                                2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z),
                            ),
                            5,
                        )
                    )
            truth_points.append(truth_point)
            model_points.append(model_point)
        contacts = []
        for first, second in torch.nonzero(
            reference_collision[index].triu(diagonal=1), as_tuple=False
        ).tolist():
            midpoint = 0.5 * (reference_position[index, first] + reference_position[index, second])
            contacts.append([round(float(midpoint[0]), 5), round(float(midpoint[1]), 5)])
        frame: dict[str, Any] = {
            "frame": index,
            "time_s": round(float(query_times[index]), 3),
            "truth": truth_points,
            "model": model_points,
        }
        if contacts:
            frame["contacts"] = contacts
        frames.append(frame)
    all_x = [point[1] for frame in frames for role in ("truth", "model") for point in frame[role]]
    all_y = [point[2] for frame in frames for role in ("truth", "model") for point in frame[role]]

    def bounds(values: list[float]) -> list[float]:
        padding = max(0.08 * (max(values) - min(values)), 0.10)
        return [round(min(values) - padding, 4), round(max(values) + padding, 4)]

    events = []
    for frame, spec in action_frames.items():
        model_slot = int(torch.where(truth_by_slot == spec.target_slot)[0][0])
        events.append(
            {
                "frame": frame,
                "time_s": spec.offset_seconds,
                "kind": f"known action on runtime object {runtime_ids[model_slot]}",
            }
        )
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
        "schema": "world_model_compact_visual_dynamic_animation_v1",
        "label": f"N={object_count} RGB-D initialized multi-contact forecast",
        "episode": f"visual-dynamic-n{object_count}",
        "object_count": object_count,
        "contact": True,
        "dynamic_membership": True,
        "endpoint_position_rmse_m": float(position_error[-1]),
        "long_horizon_endpoint_s": _FORECAST_SECONDS,
        "mode": "forecast",
        "known_actions_in_rollout": True,
        "anchor_frame": 0,
        "frame_rate": 1.0 / _FORECAST_DT,
        "projection": "world_xy",
        "axis_labels": ["x", "y"],
        "orientation_markers": True,
        "bounds": {"horizontal": bounds(all_x), "vertical": bounds(all_y)},
        "frames": frames,
        "events": events,
        "reference": "private six-DoF reference opened only after public RGB-D belief construction",
    }


def _evaluate_scenario(
    object_count: int,
) -> tuple[
    VisualDynamicScenarioResult,
    DynamicsModel,
    WorldBelief,
    RigidBodyState,
    Tensor,
]:
    tracker = OpenWorldRigidTracker(
        max_missed_steps=1,
        birth_confirmation_steps=2,
        motion_sample_count=8,
        linear_speed_deadzone=0.01,
        angular_speed_deadzone=0.5,
        geometry_weight=0.25,
    )
    proposal_tp = proposal_fp = proposal_fn = 0
    exact_visible_counts = []
    identity_correct = identity_total = 0
    runtime_by_private_id: dict[int, int] = {}
    truth_count_trace = []
    proposal_count_trace = []
    belief_count_trace = []
    tracked_history: list[tuple[TrackedRigidObject, ...]] = []
    slot_ids_by_frame: list[dict[int, int]] = []
    for frame_index in range(_OBSERVATION_FRAMES):
        state = _scene_state(object_count, frame_index)
        observed = _observe(state, frame_index)
        tracked = tracker.update(observed)
        tracked_history.append(tracked)
        visible_truth_slots = torch.where(state.active)[0]
        observed_position = (
            torch.stack([item.geometry.world_position for item in observed.objects])
            if observed.objects
            else state.position.new_empty((0, 3))
        )
        rows, columns = _assignment(observed_position, state.position[visible_truth_slots])
        proposal_tp += len(rows)
        proposal_fp += len(observed.objects) - len(rows)
        proposal_fn += len(visible_truth_slots) - len(columns)
        exact_visible_counts.append(
            len(observed.objects) == len(visible_truth_slots)
            and len(rows) == len(visible_truth_slots)
        )
        truth_count_trace.append(len(visible_truth_slots))
        proposal_count_trace.append(len(observed.objects))
        belief_count_trace.append(len(tracked))

        frame_slot_ids: dict[int, int] = {}
        if tracked:
            tracked_position = torch.stack([item.geometry.world_position for item in tracked])
            track_rows, track_columns = _assignment(tracked_position, state.position)
            for track_row, truth_slot in zip(
                track_rows.tolist(), track_columns.tolist(), strict=True
            ):
                runtime_id = tracked[track_row].object_id
                frame_slot_ids[truth_slot] = runtime_id
                if state.active[truth_slot] and tracked[track_row].observed:
                    private_id = int(state.object_id[truth_slot])
                    expected_runtime_id = runtime_by_private_id.setdefault(private_id, runtime_id)
                    identity_correct += int(expected_runtime_id == runtime_id)
                    identity_total += 1
        slot_ids_by_frame.append(frame_slot_ids)

    expected_belief_counts = (
        0,
        object_count - 1,
        object_count - 1,
        object_count - 2,
        object_count - 1,
        object_count,
        object_count,
        object_count,
    )
    lifecycle_tp = sum(
        min(actual, expected)
        for actual, expected in zip(belief_count_trace, expected_belief_counts, strict=True)
    )
    lifecycle_fp = sum(
        max(actual - expected, 0)
        for actual, expected in zip(belief_count_trace, expected_belief_counts, strict=True)
    )
    lifecycle_fn = sum(
        max(expected - actual, 0)
        for actual, expected in zip(belief_count_trace, expected_belief_counts, strict=True)
    )
    lifecycle_denominator = 2 * lifecycle_tp + lifecycle_fp + lifecycle_fn
    lifecycle_f1 = 1.0 if lifecycle_denominator == 0 else 2.0 * lifecycle_tp / lifecycle_denominator
    old_slot1_id = slot_ids_by_frame[1].get(1)
    new_slot1_id = slot_ids_by_frame[5].get(1)
    two_frame_birth = (
        (object_count - 1) not in slot_ids_by_frame[3]
        and (object_count - 1) in slot_ids_by_frame[4]
        and 1 not in slot_ids_by_frame[4]
        and new_slot1_id is not None
        and new_slot1_id != old_slot1_id
    )
    two_miss_removal = old_slot1_id is not None and old_slot1_id not in {
        item.object_id for item in tracked_history[3]
    }
    partial_visibility_recovered = slot_ids_by_frame[1].get(0) is not None and slot_ids_by_frame[
        3
    ].get(0) == slot_ids_by_frame[1].get(0)

    tracked = tracked_history[-1]
    anchor_truth = _scene_state(object_count)
    truth_by_slot = _truth_by_track(tracked, anchor_truth)
    belief = tracked_objects_to_belief(
        tracked,
        timestamp=0.0,
        max_objects=object_count,
        initial_mass=1.0,
        initial_restitution=0.5,
        initial_drag=0.05,
        initial_friction=0.25,
    )
    current_position = torch.stack([item.geometry.world_position for item in tracked])
    current_orientation = torch.stack([item.geometry.orientation for item in tracked])
    reference_position = anchor_truth.position[truth_by_slot]
    reference_orientation = anchor_truth.orientation[truth_by_slot]
    current_position_rmse = float((current_position - reference_position).square().mean().sqrt())
    box_slots = torch.where(anchor_truth.primitive[truth_by_slot] == int(RigidPrimitive.BOX))[0]
    current_orientation_rmse = math.degrees(
        float(
            quaternion_geodesic_distance(
                current_orientation[box_slots], reference_orientation[box_slots]
            )
            .square()
            .mean()
            .sqrt()
        )
    )
    primitive_accuracy = float(
        (
            torch.stack([item.geometry.primitive for item in tracked])
            == anchor_truth.primitive[truth_by_slot]
        )
        .to(_DTYPE)
        .mean()
    )

    dynamics = state_gate._dynamics(belief)
    source = belief.clone()
    query_times = torch.arange(
        _FORECAST_DT,
        _FORECAST_SECONDS + 0.5 * _FORECAST_DT,
        _FORECAST_DT,
        dtype=belief.dtype,
    )
    started = time.perf_counter()
    with torch.no_grad():
        prediction = dynamics.rollout(
            belief,
            query_times,
            action=_model_schedule(belief, truth_by_slot, object_count),
        )
    rollout_latency = time.perf_counter() - started
    truth_position, truth_velocity, truth_orientation, truth_collision = (
        state_gate._reference_rollout(
            anchor_truth,
            query_times,
            state_gate.default_actions(object_count),
        )
    )
    truth_position = truth_position[:, truth_by_slot]
    truth_velocity = truth_velocity[:, truth_by_slot]
    truth_orientation = truth_orientation[:, truth_by_slot]
    truth_collision = truth_collision[:, truth_by_slot][:, :, truth_by_slot]
    model_position = prediction.positions[0]
    model_velocity = prediction.velocities[0]
    position_error = (model_position - truth_position).square().mean(dim=(-2, -1)).sqrt()
    velocity_error = (model_velocity - truth_velocity).square().mean(dim=(-2, -1)).sqrt()
    orientation_error = quaternion_geodesic_distance(
        prediction.orientations[0], truth_orientation
    ).square().mean(dim=-1).sqrt() * (180.0 / math.pi)
    predicted_collision = prediction.auxiliary["pair_collision"][0]
    (
        reference_pairs,
        predicted_pairs,
        contact_pair_f1,
        first_contact_timing,
        repeated_contact_f1,
        simultaneous_frames,
    ) = state_gate._contact_metrics(predicted_collision, truth_collision)
    known_count_tensor = prediction.auxiliary.get("known_action_count")
    known_action_count = (
        int(known_count_tensor.sum())
        if known_count_tensor is not None
        else int(prediction.auxiliary["known_action_applied"].sum())
    )
    finite = bool(
        torch.isfinite(model_position).all()
        and torch.isfinite(model_velocity).all()
        and torch.isfinite(prediction.orientations).all()
    )
    position_curve = {
        f"{float(timestamp):.2f}": float(position_error[index])
        for index, timestamp in enumerate(query_times)
    }
    return (
        VisualDynamicScenarioResult(
            object_count=object_count,
            proposal_precision=proposal_tp / max(proposal_tp + proposal_fp, 1),
            proposal_recall=proposal_tp / max(proposal_tp + proposal_fn, 1),
            proposal_f1=(
                1.0
                if 2 * proposal_tp + proposal_fp + proposal_fn == 0
                else 2 * proposal_tp / (2 * proposal_tp + proposal_fp + proposal_fn)
            ),
            exact_visible_count_accuracy=sum(exact_visible_counts) / len(exact_visible_counts),
            persistent_id_accuracy=identity_correct / max(identity_total, 1),
            lifecycle_f1=lifecycle_f1,
            two_frame_birth_confirmation=two_frame_birth,
            two_miss_removal=two_miss_removal,
            partial_visibility_recovered=partial_visibility_recovered,
            primitive_accuracy=primitive_accuracy,
            current_position_rmse_m=current_position_rmse,
            current_orientation_rmse_degrees=current_orientation_rmse,
            known_action_count=known_action_count,
            expected_action_count=len(state_gate.default_actions(object_count)),
            unique_reference_contact_pairs=reference_pairs,
            unique_model_contact_pairs=predicted_pairs,
            contact_pair_f1=contact_pair_f1,
            repeated_contact_frame_f1=repeated_contact_f1,
            simultaneous_contact_frames=simultaneous_frames,
            first_contact_timing_error_frames=first_contact_timing,
            maximum_position_rmse_m=float(position_error.max()),
            endpoint_position_rmse_m=float(position_error[-1]),
            maximum_velocity_rmse_mps=float(velocity_error.max()),
            maximum_orientation_rmse_degrees=float(orientation_error.max()),
            rollout_latency_seconds=rollout_latency,
            source_unchanged=(
                torch.equal(source.objects.position, belief.objects.position)
                and torch.equal(source.objects.velocity, belief.objects.velocity)
                and torch.equal(source.objects.orientation, belief.objects.orientation)
            ),
            finite=finite,
            truth_count_trace=tuple(truth_count_trace),
            proposal_count_trace=tuple(proposal_count_trace),
            belief_count_trace=tuple(belief_count_trace),
            position_curve=position_curve,
            animation=_animation(
                object_count,
                belief,
                truth_by_slot,
                query_times,
                model_position,
                truth_position,
                prediction.orientations[0],
                truth_orientation,
                truth_collision,
                position_error,
            ),
        ),
        dynamics,
        belief,
        anchor_truth,
        truth_by_slot,
    )


def _evaluate_planning(
    candidate_count: int,
    dynamics: DynamicsModel,
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> state_gate.MultiContactPlanningResult:
    impulses = state_gate._planning_impulses(belief, candidate_count)
    action_time = 0.025
    horizon = 0.25
    references = tuple(
        state_gate._reference_planning_branch(
            truth,
            impulse,
            action_time=action_time,
            horizon=horizon,
        )
        for impulse in impulses
    )
    target_truth_slot = truth.max_objects - 1
    target_model_slot = int(torch.where(truth_by_slot == target_truth_slot)[0][0])
    target_runtime_id = belief.objects.object_id[:, target_model_slot].clone()
    target = references[0].position[target_truth_slot]
    truth_cost = torch.stack(
        [(item.position[target_truth_slot] - target).square().sum() for item in references]
    )
    sorted_cost = truth_cost.sort().values
    cost_scale = max(float(truth_cost.max()), 1.0e-12)
    actions = tuple(
        WorldImpulseAction(
            timestamp=belief.timestamp.new_tensor([action_time]),
            object_id=target_runtime_id.clone(),
            impulse_world=impulse.reshape(1, 3),
        )
        for impulse in impulses
    )
    goal = TerminalWorldPositionGoal(
        object_id=target_runtime_id.clone(),
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
    selected_error = float(
        torch.linalg.vector_norm(references[selected].position[target_truth_slot] - target)
    )
    return state_gate.MultiContactPlanningResult(
        object_count=truth.max_objects,
        candidate_count=candidate_count,
        oracle_winner=oracle,
        selected_winner=selected,
        winner_correct=selected == oracle,
        normalized_regret=float(truth_cost[selected] - truth_cost.min()) / cost_scale,
        normalized_winner_margin=float(sorted_cost[1] - sorted_cost[0]) / cost_scale,
        goal_success=selected_error <= 0.05,
        serial_vectorized_parity=torch.equal(vectorized.selected_index, serial.selected_index),
        maximum_cost_difference=float((vectorized.total_cost - serial.total_cost).abs().max()),
        vectorized_latency_seconds=vectorized_latency,
        serial_latency_seconds=serial_latency,
        vectorization_speedup=serial_latency / max(vectorized_latency, 1.0e-12),
        source_unchanged=(
            torch.equal(source.objects.position, belief.objects.position)
            and torch.equal(source.objects.velocity, belief.objects.velocity)
            and torch.equal(source.objects.orientation, belief.objects.orientation)
        ),
    )


def _gate_failures(
    scenarios: tuple[VisualDynamicScenarioResult, ...],
    planning: tuple[state_gate.MultiContactPlanningResult, ...],
    *,
    enforce_latency: bool,
) -> tuple[str, ...]:
    failures = []
    for item in scenarios:
        prefix = f"n{item.object_count}"
        checks = {
            "proposal_f1": item.proposal_f1 >= 0.95,
            "exact_visible_count": item.exact_visible_count_accuracy >= 0.95,
            "persistent_id_accuracy": item.persistent_id_accuracy >= 0.98,
            "lifecycle_f1": item.lifecycle_f1 >= 0.95,
            "birth_confirmation": item.two_frame_birth_confirmation,
            "removal": item.two_miss_removal,
            "partial_visibility_recovery": item.partial_visibility_recovered,
            "primitive_accuracy": item.primitive_accuracy == 1.0,
            "current_position": item.current_position_rmse_m <= 0.020,
            "current_orientation": item.current_orientation_rmse_degrees <= 5.0,
            "known_actions": item.known_action_count == item.expected_action_count == 3,
            "contact_pairs": item.contact_pair_f1 >= 0.90,
            "repeated_contacts": item.repeated_contact_frame_f1 >= 0.60,
            "simultaneous_contacts": item.simultaneous_contact_frames >= 1,
            "contact_timing": item.first_contact_timing_error_frames is not None
            and item.first_contact_timing_error_frames <= 1,
            "position": item.maximum_position_rmse_m <= 0.030,
            "velocity": item.maximum_velocity_rmse_mps <= 0.100,
            "orientation": item.maximum_orientation_rmse_degrees <= 10.0,
            "source_unchanged": item.source_unchanged,
            "finite": item.finite,
        }
        if enforce_latency:
            checks["latency"] = (
                item.rollout_latency_seconds <= {4: 12.0, 6: 22.0, 8: 36.0}[item.object_count]
            )
        failures.extend(f"{prefix}:{name}" for name, passed in checks.items() if not passed)
    for item in planning:
        checks = {
            "winner": item.winner_correct,
            "regret": item.normalized_regret <= (0.05 if item.candidate_count == 8 else 0.07),
            "goal": item.goal_success,
            "winner_margin": item.normalized_winner_margin >= 0.05,
            "parity": item.serial_vectorized_parity,
            "cost": item.maximum_cost_difference <= 1.0e-6,
            "speedup": item.vectorization_speedup >= 5.0,
            "source_unchanged": item.source_unchanged,
        }
        if enforce_latency:
            checks["latency"] = item.vectorized_latency_seconds <= 6.5
        failures.extend(
            f"planning_k{item.candidate_count}:{name}"
            for name, passed in checks.items()
            if not passed
        )
    return tuple(failures)


def run_visual_dynamic_scale(*, enforce_latency: bool = True) -> VisualDynamicScaleResult:
    """Run the first integrated RGB-D, lifecycle, dynamics, and planning scale gate."""

    started = time.perf_counter()
    scenarios = []
    learned_weight_bytes = 0
    planning_inputs = None
    for object_count in _COUNTS:
        scenario, dynamics, belief, truth, truth_by_slot = _evaluate_scenario(object_count)
        scenarios.append(scenario)
        learned_weight_bytes = max(
            learned_weight_bytes,
            sum(
                parameter.numel() * parameter.element_size() for parameter in dynamics.parameters()
            ),
        )
        if object_count == 8:
            planning_inputs = dynamics, belief, truth, truth_by_slot
    assert planning_inputs is not None
    planning = tuple(
        _evaluate_planning(candidate_count, *planning_inputs) for candidate_count in (8, 32)
    )
    scenario_tuple = tuple(scenarios)
    failures = _gate_failures(scenario_tuple, planning, enforce_latency=enforce_latency)
    return VisualDynamicScaleResult(
        schema=VISUAL_DYNAMIC_SCALE_SCHEMA,
        manifest_sha256=visual_dynamic_manifest_sha256(),
        scenarios=scenario_tuple,
        planning=planning,
        gate_failures=failures,
        qualified=not failures,
        learned_weight_bytes=learned_weight_bytes,
        peak_run_tensor_bytes=6 * _IMAGE_SIZE[0] * _IMAGE_SIZE[1] * 5 * 8,
        evaluation_seconds=time.perf_counter() - started,
    )


def _summary(
    result: VisualDynamicScaleResult,
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
        source_format=VISUAL_DYNAMIC_SCALE_SCHEMA,
        configuration={
            "object_counts": list(_COUNTS),
            "public_calibrated_rgbd": True,
            "prototype_free_runtime": True,
            "runtime_owned_ids": True,
            "two_frame_birth_confirmation": True,
            "two_miss_removal": True,
            "moving_calibrated_cameras": True,
            "mixed_rigid_primitives": True,
            "planning_used_as_training_loss": False,
            "generated_frames_retained": False,
        },
        provenance={
            "scenario_manifest_sha256": result.manifest_sha256,
            "truth_oracle": "independent world_model.simulator.rigid_six_dof",
            "belief_initialization": "prototype-free public calibrated RGB-D discovery",
            "runtime_truth_inputs": False,
            "private_reference_opened_after_belief_construction": True,
            "dense_serial_planning_oracle": True,
        },
        scores={
            "candidate": {
                "value": max(item.maximum_position_rmse_m for item in result.scenarios),
                "supported_weight": 1.0,
            },
            "incumbent": {
                "value": max(item.maximum_position_rmse_m for item in result.scenarios),
                "supported_weight": 1.0,
            },
            "selected": "visual_dynamic_scale" if result.qualified else "incumbent",
        },
        factor_metrics={
            f"visual_dynamic_n{count}": {
                "status": "passed"
                if not any(failure.startswith(f"n{count}:") for failure in result.gate_failures)
                else "failed",
                "proposal_f1": item.proposal_f1,
                "identity_accuracy": item.persistent_id_accuracy,
                "lifecycle_f1": item.lifecycle_f1,
                "current_position_rmse_m": item.current_position_rmse_m,
                "two_second_position_rmse_m": item.endpoint_position_rmse_m,
                "collision_f1": item.contact_pair_f1,
                "orientation_rmse_degrees": item.maximum_orientation_rmse_degrees,
            }
            for count, item in by_count.items()
        },
        cell_metrics={
            f"N{count}/visual_dynamic/mixed_rigid": {
                "proposal_f1": {"value": item.proposal_f1, "support": count * 8},
                "persistent_id_accuracy": {
                    "value": item.persistent_id_accuracy,
                    "support": count,
                },
                "lifecycle_f1": {"value": item.lifecycle_f1, "support": 8},
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
        uncertainty={"status": "carried from explicit public-observation confidence"},
        planning={
            "status": "passed"
            if all(item.winner_correct for item in result.planning)
            else "failed",
            "goal": "N=8 visually initialized multi-contact terminal position",
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
            "selected": "visual_dynamic_scale" if result.qualified else "none",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "public proposal, association/lifecycle, parameter prior, or six-DoF dynamics",
        },
        qualitative={
            "best_episode": "visual-dynamic-n4",
            "representative_episode": "visual-dynamic-n6",
            "worst_episode": "visual-dynamic-n8",
            "forecast_gallery_mode": "latest_run",
            "animations": [],
            "forecast_animations": [item.animation for item in result.scenarios],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "unknown camera calibration",
            "hidden actions",
            "identity recovery through an unobserved impulse or collision",
            "visual qualification above eight objects",
        ),
        scope_limitations=(
            "known shared physical prior at this first integrated rung",
            "two-second forecast horizon",
            "planning freezes the active set",
            "flush featureless object unions remain unobservable",
        ),
    ).validate()


def publish_visual_dynamic_scale(
    result: VisualDynamicScaleResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "visual_dynamic_scale.json",
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
                "visual_dynamic_scale.json": "summary",
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
        raise RuntimeError("visual-dynamic evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "VISUAL_DYNAMIC_SCALE_SCHEMA",
    "VisualDynamicScaleResult",
    "VisualDynamicScenarioResult",
    "publish_visual_dynamic_scale",
    "run_visual_dynamic_scale",
    "visual_dynamic_manifest_sha256",
]
