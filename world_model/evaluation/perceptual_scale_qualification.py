"""Compact full RGB-D qualification ladder for visible sets of seven and eight."""

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

from world_model.dynamics import WorldImpulseAction
from world_model.evaluation.capability_factor_runner import _model_from_workbench_checkpoint
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.evaluation.perceptual_scaling import scaled_set_config
from world_model.observations import ObservationPacket
from world_model.planning import TerminalWorldPositionGoal, plan_counterfactual_actions
from world_model.simulator import (
    CameraFrame,
    PhysicsConfig,
    SphereState,
    advance_spheres,
    invert_rigid_transform,
    make_intrinsics,
    render_spheres,
)
from world_model.training.dynamic_set_config import load_config
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

SCALE_QUALIFICATION_SCHEMA = "world_model_perceptual_scale_qualification_v1"
_FRAME_RATE = 20.0
_IMAGE_SIZE = (64, 64)
_BOUNDS = ((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0))


@dataclass(frozen=True, slots=True)
class PerceptualScaleScenario:
    name: str
    object_count: int
    family: str
    frame_count: int
    known_action_frame: int | None = None
    occlusion_frame: int | None = None
    moving_camera: bool = False
    sensor_noise: bool = False


@dataclass(frozen=True, slots=True)
class PerceptualScaleScenarioResult:
    name: str
    object_count: int
    family: str
    proposal_precision: float
    proposal_recall: float
    proposal_f1: float
    exact_count_accuracy: float
    current_position_rmse_m: float
    two_second_position_rmse_m: float
    persistent_id_accuracy: float
    lifecycle_f1: float
    collision_f1: float | None
    collision_timing_error_frames: int | None
    occlusion_recovered: bool | None
    known_action_count: int
    target_handle_resolved: bool
    finite: bool
    latency_seconds: float
    truth_count_trace: tuple[int, ...]
    proposal_count_trace: tuple[int, ...]
    belief_count_trace: tuple[int, ...]
    animation: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PerceptualScalePlanningResult:
    object_count: int
    candidate_count: int
    winner_correct: bool
    normalized_regret: float
    goal_success: bool
    serial_vectorized_parity: bool
    maximum_cost_difference: float
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PerceptualScaleQualificationResult:
    schema: str
    manifest_sha256: str
    scenarios: tuple[PerceptualScaleScenarioResult, ...]
    planning: tuple[PerceptualScalePlanningResult, ...]
    gate_failures: tuple[str, ...]
    full_perceptual_qualification: bool
    learned_weight_bytes: int
    peak_run_tensor_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "planning": [item.to_dict() for item in self.planning],
        }


def default_perceptual_scale_scenarios() -> tuple[PerceptualScaleScenario, ...]:
    return tuple(
        PerceptualScaleScenario(
            name=f"n{count}-{family}",
            object_count=count,
            family=family,
            frame_count=12 if family == "lifecycle" else 10,
            known_action_frame=4 if family == "known_action" else None,
            occlusion_frame=5 if family == "occlusion_recovery" else None,
            moving_camera=family == "compositional",
            sensor_noise=family == "compositional",
        )
        for count in (7, 8)
        for family in (
            "separated_motion",
            "known_action",
            "pair_contact",
            "lifecycle",
            "occlusion_recovery",
            "compositional",
        )
    )


def perceptual_scale_manifest_sha256(
    scenarios: tuple[PerceptualScaleScenario, ...] | None = None,
) -> str:
    payload = [asdict(item) for item in (scenarios or default_perceptual_scale_scenarios())]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_for_scenario(scenario: PerceptualScaleScenario, timestamp: float) -> SphereState:
    count = scenario.object_count
    grid = torch.tensor(
        [
            [-1.20, -0.52, 4.0],
            [-0.40, -0.52, 4.0],
            [0.40, -0.52, 4.0],
            [1.20, -0.52, 4.0],
            [-1.20, 0.52, 4.0],
            [-0.40, 0.52, 4.0],
            [0.40, 0.52, 4.0],
            [1.20, 0.52, 4.0],
        ],
        dtype=torch.float32,
    )[:count]
    velocity = torch.tensor(
        [[0.018 * (-1.0 if index % 2 else 1.0), 0.0, 0.0] for index in range(count)],
        dtype=torch.float32,
    )
    if scenario.family == "pair_contact":
        # Preserve the established two-row image layout and make the first
        # column collide vertically.  This isolates contact from a gratuitous
        # anchor-distribution shift at N=7/8.
        grid[0] = torch.tensor([-1.20, -0.46, 4.0])
        grid[4] = torch.tensor([-1.20, 0.46, 4.0])
        velocity.zero_()
        velocity[0, 1] = 0.9
        velocity[4, 1] = -0.9
    positions = grid + velocity * timestamp
    active = torch.ones(count, dtype=torch.bool)
    object_id = torch.arange(100, 100 + count, dtype=torch.int64)
    if scenario.family == "lifecycle":
        last = count - 1
        if timestamp < 2.0 / _FRAME_RATE or 6.0 / _FRAME_RATE <= timestamp < 8.0 / _FRAME_RATE:
            active[last] = False
            object_id[last] = -1
        elif timestamp >= 8.0 / _FRAME_RATE:
            object_id[last] = 200 + last
            positions[last] = grid[last] + torch.tensor([0.0, 0.0, 0.12])
    radii = torch.full((count, 1), 0.21)
    mass = torch.ones(count, 1)
    restitution = torch.full((count, 1), 0.70)
    drag = torch.full((count, 1), 0.05)
    friction = torch.full((count, 1), 0.20)
    if scenario.family == "compositional":
        scale = torch.linspace(0.94, 1.06, count).unsqueeze(-1)
        radii = radii * scale
        mass = torch.linspace(0.8, 1.2, count).unsqueeze(-1)
        restitution = torch.linspace(0.55, 0.82, count).unsqueeze(-1)
        drag = torch.linspace(0.03, 0.08, count).unsqueeze(-1)
        friction = torch.linspace(0.12, 0.32, count).unsqueeze(-1)
    colours = torch.tensor(
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
        dtype=torch.float32,
    )[:count]
    return SphereState(
        object_id=object_id,
        active=active,
        position=positions,
        velocity=velocity,
        radius=radii,
        mass=mass,
        restitution=restitution,
        drag=drag,
        friction=friction,
        albedo=colours,
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(count, -1).clone(),
        angular_velocity=torch.zeros(count, 3),
        sleeping=torch.zeros(count, dtype=torch.bool),
        sleep_counter=torch.zeros(count, dtype=torch.int64),
    )


def _camera(timestamp: float, moving: bool) -> CameraFrame:
    world_from_camera = torch.eye(4, dtype=torch.float32)
    if moving:
        world_from_camera[0, 3] = 0.12 * math.sin(1.7 * timestamp)
        world_from_camera[1, 3] = 0.06 * math.cos(1.3 * timestamp)
    return CameraFrame(
        timestamp=timestamp,
        world_from_camera=world_from_camera,
        camera_from_world=invert_rigid_transform(world_from_camera),
        intrinsics=make_intrinsics(_IMAGE_SIZE, 48.0),
        position=world_from_camera[:3, 3].clone(),
        target=world_from_camera[:3, 3] + torch.tensor([0.0, 0.0, 1.0]),
    )


def _packet(
    state: SphereState,
    scenario: PerceptualScaleScenario,
    frame_index: int,
    *,
    generator: torch.Generator,
) -> ObservationPacket:
    timestamp = frame_index / _FRAME_RATE
    camera = _camera(timestamp, scenario.moving_camera)
    rendered = render_spheres(
        state,
        camera,
        _IMAGE_SIZE,
        noise_std=0.004 if scenario.sensor_noise else 0.0,
        generator=generator,
    )
    rgb = rendered.rgb.clone()
    depth = rendered.depth_buffer.clone()
    if scenario.sensor_noise:
        foreground = depth > 0.0
        depth_noise = torch.randn(depth.shape, generator=generator) * 0.0015
        depth = torch.where(foreground, (depth + depth_noise).clamp_min(1.0e-4), depth)
        dropout = (torch.rand(depth.shape, generator=generator) < 0.006) & foreground
        depth[dropout] = 0.0
    if scenario.occlusion_frame == frame_index:
        target_slot = state.max_objects - 1
        target_pixels = rendered.instance_slot_map == target_slot
        columns = torch.arange(_IMAGE_SIZE[1]).unsqueeze(0)
        target_columns = torch.where(target_pixels)[1]
        if target_columns.numel():
            cutoff = int(target_columns.float().median())
            occluded = target_pixels & (columns <= cutoff)
            hidden_active = state.active.clone()
            hidden_ids = state.object_id.clone()
            hidden_active[target_slot] = False
            hidden_ids[target_slot] = -1
            hidden = render_spheres(
                replace(state, active=hidden_active, object_id=hidden_ids),
                camera,
                _IMAGE_SIZE,
            )
            rgb[:, occluded] = hidden.rgb[:, occluded]
            depth[occluded] = hidden.depth_buffer[occluded]
    return ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=timestamp,
        payload={"rgb": rgb.unsqueeze(0), "depth": depth[None, None]},
        calibration={
            "world_from_camera": camera.world_from_camera.unsqueeze(0),
            "intrinsics": camera.intrinsics.unsqueeze(0),
        },
        frame_id="camera:camera0:rgbd",
        metadata={"image_size": _IMAGE_SIZE},
    )


def _matches(estimated: torch.Tensor, truth: torch.Tensor, threshold: float = 0.10):
    if not estimated.numel() or not truth.numel():
        return [], [], []
    rows, columns = linear_sum_assignment(torch.cdist(estimated, truth).detach().cpu().numpy())
    distances = torch.linalg.vector_norm(estimated[rows] - truth[columns], dim=-1)
    kept = distances <= threshold
    return rows[kept].tolist(), columns[kept].tolist(), distances[kept].tolist()


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


def _event_f1(predicted: list[bool], target: list[bool]) -> tuple[float, int | None]:
    predicted_tensor = torch.tensor(predicted, dtype=torch.bool)
    target_tensor = torch.tensor(target, dtype=torch.bool)
    score = _f1(
        int((predicted_tensor & target_tensor).sum()),
        int((predicted_tensor & ~target_tensor).sum()),
        int((~predicted_tensor & target_tensor).sum()),
    )
    predicted_indices = torch.where(predicted_tensor)[0]
    target_indices = torch.where(target_tensor)[0]
    if not len(predicted_indices) and not len(target_indices):
        return score, 0
    if not len(predicted_indices) or not len(target_indices):
        return score, None
    timing = max(int((predicted_indices - item).abs().min()) for item in target_indices)
    return score, timing


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        gravity=(0.0, 0.0, 0.0),
        bounds=_BOUNDS,
        max_substep=1.0 / 120.0,
        solver_iterations=2,
    )


def _evaluate_scenario(
    model: Any, scenario: PerceptualScaleScenario
) -> tuple[PerceptualScaleScenarioResult, SphereState]:
    generator = torch.Generator().manual_seed(
        8000 + scenario.object_count * 100 + len(scenario.name)
    )
    truth = _state_for_scenario(scenario, 0.0)
    proposal_tp = proposal_fp = proposal_fn = 0
    state_tp = state_fp = state_fn = 0
    exact_counts: list[bool] = []
    squared_errors: list[float] = []
    identity_correct = identity_total = 0
    identity_map: dict[int, int] = {}
    predicted_collisions: list[bool] = []
    target_collisions: list[bool] = []
    known_action_count = 0
    pre_occlusion_id: int | None = None
    recovered_id: int | None = None
    target_handle_resolved = False
    truth_count_trace: list[int] = []
    proposal_count_trace: list[int] = []
    belief_count_trace: list[int] = []
    animation_frames: list[dict[str, Any]] = []
    animation_events: list[dict[str, Any]] = []
    started = time.perf_counter()
    for frame in range(scenario.frame_count):
        timestamp = frame / _FRAME_RATE
        truth_events = None
        action = None
        if frame > 0:
            if scenario.family == "lifecycle":
                truth = _state_for_scenario(scenario, timestamp)
            elif scenario.known_action_frame == frame:
                half = 0.5 / _FRAME_RATE
                truth, first_events = advance_spheres(truth, half, _physics())
                impulse = torch.zeros_like(truth.velocity)
                impulse[0] = torch.tensor([0.08, 0.03, 0.0])
                truth, second_events = advance_spheres(
                    truth,
                    half,
                    _physics(),
                    external_impulse=impulse,
                )
                truth_events = first_events.pair_collision | second_events.pair_collision
                if model.belief is not None:
                    active = model.belief.objects.active[0]
                    estimated = model.belief.objects.position[0, active]
                    rows, columns, _ = _matches(estimated, truth.position[truth.active])
                    truth_active_slots = torch.where(truth.active)[0]
                    active_slots = torch.where(active)[0]
                    target_matches = [
                        row
                        for row, column in zip(rows, columns, strict=True)
                        if int(truth_active_slots[column]) == 0
                    ]
                    if target_matches:
                        target_slot = int(active_slots[target_matches[0]])
                        action = WorldImpulseAction(
                            timestamp=model.belief.timestamp.new_tensor([timestamp - half]),
                            object_id=model.belief.objects.object_id[:, target_slot].clone(),
                            impulse_world=model.belief.timestamp.new_tensor([[0.08, 0.03, 0.0]]),
                        )
            else:
                truth, events = advance_spheres(truth, 1.0 / _FRAME_RATE, _physics())
                truth_events = events.pair_collision
        packet = _packet(truth, scenario, frame, generator=generator)
        prepared = None
        if model.belief is not None:
            prepared = model.prepare_propagation(timestamp, action=action)
            predicted_collisions.append(bool(prepared.auxiliary["pair_collision"].any()))
            action_count = prepared.auxiliary.get("known_action_count")
            action_applied = prepared.auxiliary.get("known_action_applied")
            if action_count is not None:
                known_action_count += int(action_count.sum())
            elif action_applied is not None:
                known_action_count += int(action_applied.sum())
        model.ingest(packet, prepared=prepared)
        if frame > 0:
            target_collisions.append(
                bool(truth_events.any()) if truth_events is not None else False
            )

        measurements = model.last_measurements
        if measurements is None or model.belief is None:
            continue
        if scenario.object_count == 8 and scenario.family in {
            "pair_contact",
            "lifecycle",
            "compositional",
        }:
            truth_slots_for_frame = torch.where(truth.active)[0]
            model_slots_for_frame = torch.where(model.belief.objects.active[0])[0]
            model_positions = model.belief.objects.position[0, model_slots_for_frame].detach()
            truth_positions = truth.position[truth_slots_for_frame].detach()
            visual_rows, visual_columns, _ = _matches(model_positions, truth_positions)
            model_points = [
                [
                    int(truth.object_id[truth_slots_for_frame[column]]),
                    round(float(model_positions[row, 0]), 5),
                    round(float(model_positions[row, 1]), 5),
                ]
                for row, column in zip(visual_rows, visual_columns, strict=True)
            ]
            truth_points = [
                [
                    int(truth.object_id[index]),
                    round(float(truth.position[index, 0]), 5),
                    round(float(truth.position[index, 1]), 5),
                ]
                for index in truth_slots_for_frame
            ]
            animation_frames.append(
                {
                    "frame": frame,
                    "time_s": round(timestamp, 3),
                    "truth": truth_points,
                    "model": model_points,
                }
            )
            if truth_events is not None and bool(truth_events.any()):
                animation_events.append({"frame": frame, "kind": "contact"})
            if action is not None:
                animation_events.append({"frame": frame, "kind": "known action"})
            if scenario.family == "lifecycle" and frame in {2, 6, 8}:
                kind = "removal" if frame == 6 else "birth"
                animation_events.append({"frame": frame, "kind": kind})
        truth_count_trace.append(int(truth.active.sum()))
        proposal_count_trace.append(int(measurements.measurement_mask.sum()))
        belief_count_trace.append(int(model.belief.objects.active.sum()))
        evaluation_frame = frame >= 2
        if scenario.family == "lifecycle":
            evaluation_frame = frame in {3, 5, 7, 9, 11}
        if not evaluation_frame:
            continue
        truth_slots = torch.where(truth.active)[0]
        truth_position = truth.position[truth_slots]
        measurement_position = measurements.values[0, measurements.measurement_mask[0]]
        measurement_rows, measurement_columns, _ = _matches(measurement_position, truth_position)
        proposal_tp += len(measurement_rows)
        proposal_fp += len(measurement_position) - len(measurement_rows)
        proposal_fn += len(truth_position) - len(measurement_columns)

        active = model.belief.objects.active[0]
        active_slots = torch.where(active)[0]
        estimated = model.belief.objects.position[0, active]
        rows, columns, distances = _matches(estimated, truth_position)
        state_tp += len(rows)
        state_fp += len(estimated) - len(rows)
        state_fn += len(truth_position) - len(columns)
        exact_counts.append(
            len(estimated) == len(truth_position) and len(rows) == len(truth_position)
        )
        squared_errors.extend(float(distance) ** 2 for distance in distances)
        for row, column in zip(rows, columns, strict=True):
            truth_id = int(truth.object_id[truth_slots[column]])
            model_id = int(model.belief.objects.object_id[0, active_slots[row]])
            previous = identity_map.setdefault(truth_id, model_id)
            identity_correct += int(previous == model_id)
            identity_total += 1
        if scenario.occlusion_frame is not None:
            target_truth_id = int(truth.object_id[-1])
            if frame == scenario.occlusion_frame - 1:
                pre_occlusion_id = identity_map.get(target_truth_id)
            if frame >= scenario.occlusion_frame + 1:
                recovered_id = identity_map.get(target_truth_id)

        if (
            frame == scenario.frame_count - 1
            and measurements.appearance is not None
            and measurement_rows
        ):
            matching_measurement = next(
                (
                    row
                    for row, column in zip(measurement_rows, measurement_columns, strict=True)
                    if column == 0
                ),
                None,
            )
            if matching_measurement is not None:
                prototype = measurements.appearance[:, measurements.measurement_mask[0]][
                    :, matching_measurement
                ]
                from world_model.planning import resolve_appearance_handle

                resolved = resolve_appearance_handle(
                    model.belief,
                    prototype,
                    minimum_cosine_margin=0.001,
                )
                resolved_slot = torch.where(
                    model.belief.objects.active[0]
                    & (model.belief.objects.object_id[0] == resolved[0])
                )[0]
                if len(resolved_slot) == 1:
                    resolved_position = model.belief.objects.position[0, resolved_slot[0]]
                    target_handle_resolved = bool(
                        torch.linalg.vector_norm(resolved_position - truth_position[0]) <= 0.10
                    )
    latency = time.perf_counter() - started
    if model.belief is None:
        raise RuntimeError("scale scenario produced no belief")
    active = model.belief.objects.active[0]
    truth_active = truth.active
    active_slots = torch.where(active)[0]
    truth_slots = torch.where(truth_active)[0]
    rows, columns, _ = _matches(
        model.belief.objects.position[0, active],
        truth.position[truth_active],
    )
    rollout_rmse = math.inf
    if len(rows) == len(truth_slots):
        with torch.no_grad():
            predicted = model.predict([2.0])
        truth_future, _ = advance_spheres(truth, 2.0, _physics())
        prediction = predicted.positions[0, -1, active_slots[rows]]
        target = truth_future.position[truth_slots[columns]]
        rollout_rmse = float((prediction - target).square().mean().sqrt())
    collision_f1 = collision_timing = None
    if scenario.family == "pair_contact":
        collision_f1, collision_timing = _event_f1(predicted_collisions, target_collisions)
    finite = bool(
        torch.isfinite(model.belief.objects.position[0, active]).all()
        and torch.isfinite(model.belief.objects.velocity[0, active]).all()
        and math.isfinite(rollout_rmse)
    )
    animation = None
    if animation_frames:
        coordinates = [
            point[axis]
            for frame in animation_frames
            for role in ("truth", "model")
            for point in frame[role]
            for axis in (1, 2)
        ]
        x_values = [
            point[1]
            for frame in animation_frames
            for role in ("truth", "model")
            for point in frame[role]
        ]
        y_values = [
            point[2]
            for frame in animation_frames
            for role in ("truth", "model")
            for point in frame[role]
        ]

        def bounds(values: list[float]) -> list[float]:
            span = max(values) - min(values)
            padding = max(0.08 * span, 0.08)
            return [round(min(values) - padding, 4), round(max(values) + padding, 4)]

        if not coordinates:
            raise RuntimeError("animation was selected without observable coordinates")
        animation = {
            "schema": "world_model_compact_tracking_animation_v1",
            "label": scenario.family.replace("_", " "),
            "episode": scenario.name,
            "object_count": scenario.object_count,
            "contact": scenario.family == "pair_contact",
            "dynamic_membership": scenario.family == "lifecycle",
            "current_position_rmse_m": math.sqrt(sum(squared_errors) / max(len(squared_errors), 1)),
            "mode": "tracking",
            "projection": "world_xy",
            "axis_labels": ["x", "y"],
            "bounds": {"horizontal": bounds(x_values), "vertical": bounds(y_values)},
            "frames": animation_frames,
            "events": animation_events,
            "reference": "private truth opened only after each public RGB-D estimate",
        }
    return (
        PerceptualScaleScenarioResult(
            name=scenario.name,
            object_count=scenario.object_count,
            family=scenario.family,
            proposal_precision=proposal_tp / max(proposal_tp + proposal_fp, 1),
            proposal_recall=proposal_tp / max(proposal_tp + proposal_fn, 1),
            proposal_f1=_f1(proposal_tp, proposal_fp, proposal_fn),
            exact_count_accuracy=sum(exact_counts) / max(len(exact_counts), 1),
            current_position_rmse_m=math.sqrt(sum(squared_errors) / max(len(squared_errors), 1)),
            two_second_position_rmse_m=rollout_rmse,
            persistent_id_accuracy=identity_correct / max(identity_total, 1),
            lifecycle_f1=_f1(state_tp, state_fp, state_fn),
            collision_f1=collision_f1,
            collision_timing_error_frames=collision_timing,
            occlusion_recovered=(
                pre_occlusion_id is not None and recovered_id == pre_occlusion_id
                if scenario.occlusion_frame is not None
                else None
            ),
            known_action_count=known_action_count,
            target_handle_resolved=target_handle_resolved,
            finite=finite,
            latency_seconds=latency,
            truth_count_trace=tuple(truth_count_trace),
            proposal_count_trace=tuple(proposal_count_trace),
            belief_count_trace=tuple(belief_count_trace),
            animation=animation,
        ),
        truth,
    )


def _evaluate_planning(
    model: Any, truth: SphereState, candidate_count: int
) -> PerceptualScalePlanningResult:
    if model.belief is None:
        raise RuntimeError("planning requires an observed belief")
    belief = model.belief
    active = belief.objects.active[0]
    active_slots = torch.where(active)[0]
    truth_slots = torch.where(truth.active)[0]
    rows, columns, _ = _matches(belief.objects.position[0, active], truth.position[truth.active])
    matching = next(
        (row for row, column in zip(rows, columns, strict=True) if int(truth_slots[column]) == 0),
        None,
    )
    if matching is None:
        raise RuntimeError("planning target is not observable")
    model_slot = int(active_slots[matching])
    target_id = belief.objects.object_id[:, model_slot].clone()
    action_time = belief.timestamp + 0.05
    impulses = [torch.tensor([0.22, 0.08, 0.0])]
    for index in range(1, candidate_count):
        angle = 2.0 * math.pi * (index - 1) / max(candidate_count - 1, 1)
        impulses.append(torch.tensor([0.20 * math.cos(angle), 0.20 * math.sin(angle), 0.0]))
    actions = tuple(
        WorldImpulseAction(
            timestamp=action_time.clone(),
            object_id=target_id.clone(),
            impulse_world=belief.objects.position.new_tensor([impulse.tolist()]),
        )
        for impulse in impulses
    )
    terminals = []
    for impulse in impulses:
        branch, _ = advance_spheres(truth, 0.05, _physics())
        external = torch.zeros_like(branch.velocity)
        external[0] = impulse
        branch, _ = advance_spheres(branch, 1.95, _physics(), external_impulse=external)
        terminals.append(branch.position[0])
    truth_terminal = torch.stack(terminals)
    truth_cost = (truth_terminal - truth_terminal[0]).square().sum(dim=-1)
    goal = TerminalWorldPositionGoal(
        object_id=target_id,
        position_world=truth_terminal[0].reshape(1, 3).to(belief.objects.position),
    )
    started = time.perf_counter()
    with torch.no_grad():
        vectorized = plan_counterfactual_actions(
            model.dynamics,
            belief,
            [2.0],
            actions,
            goal,
            return_events=False,
            return_auxiliary=False,
        )
        latency = time.perf_counter() - started
        serial = plan_counterfactual_actions(
            model.dynamics,
            belief,
            [2.0],
            actions,
            goal,
            candidate_vectorized=False,
            return_events=False,
            return_auxiliary=False,
        )
    selected = int(vectorized.selected_index[0])
    scale = max(float(truth_cost.max()), 1.0e-12)
    return PerceptualScalePlanningResult(
        object_count=int(truth.active.sum()),
        candidate_count=candidate_count,
        winner_correct=selected == 0,
        normalized_regret=float(truth_cost[selected]) / scale,
        goal_success=float(torch.linalg.vector_norm(truth_terminal[selected] - truth_terminal[0]))
        <= 0.10,
        serial_vectorized_parity=torch.equal(vectorized.selected_index, serial.selected_index),
        maximum_cost_difference=float((vectorized.total_cost - serial.total_cost).abs().max()),
        latency_seconds=latency,
    )


def _gate_failures(
    scenarios: tuple[PerceptualScaleScenarioResult, ...],
    planning: tuple[PerceptualScalePlanningResult, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    for item in scenarios:
        prefix = item.name
        if item.proposal_f1 < 0.95:
            failures.append(f"{prefix}:proposal_f1")
        if item.exact_count_accuracy < 0.95:
            failures.append(f"{prefix}:exact_count")
        if item.persistent_id_accuracy < 0.98:
            failures.append(f"{prefix}:persistent_id")
        if item.lifecycle_f1 < 0.95:
            failures.append(f"{prefix}:lifecycle_f1")
        if item.current_position_rmse_m > 0.020:
            failures.append(f"{prefix}:current_position")
        if item.two_second_position_rmse_m > 0.120:
            failures.append(f"{prefix}:two_second_position")
        if item.collision_f1 is not None and item.collision_f1 < 0.90:
            failures.append(f"{prefix}:collision_f1")
        if (
            item.collision_timing_error_frames is not None
            and item.collision_timing_error_frames > 1
        ):
            failures.append(f"{prefix}:collision_timing")
        if item.occlusion_recovered is False:
            failures.append(f"{prefix}:occlusion_recovery")
        if item.family == "known_action" and item.known_action_count != 1:
            failures.append(f"{prefix}:known_action_count")
        if not item.target_handle_resolved:
            failures.append(f"{prefix}:target_handle")
        if not item.finite:
            failures.append(f"{prefix}:nonfinite")
    for item in planning:
        if not item.winner_correct:
            failures.append(f"n{item.object_count}:planning_k{item.candidate_count}_winner")
        limit = 0.05 if item.candidate_count == 8 else 0.07
        if item.normalized_regret > limit:
            failures.append(f"n{item.object_count}:planning_k{item.candidate_count}_regret")
        if not item.goal_success:
            failures.append(f"n{item.object_count}:planning_k{item.candidate_count}_goal")
        if not item.serial_vectorized_parity or item.maximum_cost_difference > 1.0e-6:
            failures.append(f"n{item.object_count}:planning_k{item.candidate_count}_parity")
    return tuple(failures)


def run_perceptual_scale_qualification(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
) -> PerceptualScaleQualificationResult:
    """Run deterministic N=7/8 perception, lifecycle, contact, and planning gates."""

    base = load_config(model_config_path)
    scenario_results: list[PerceptualScaleScenarioResult] = []
    planning_results: list[PerceptualScalePlanningResult] = []
    learned_weight_bytes = 0
    peak_tensor_bytes = 0
    models: dict[int, Any] = {}
    for scenario in default_perceptual_scale_scenarios():
        model = models.get(scenario.object_count)
        if model is None:
            config = scaled_set_config(base, max_objects=scenario.object_count)
            model, _ = _model_from_workbench_checkpoint(config, checkpoint_path)
            models[scenario.object_count] = model
        else:
            model.reset()
        result, truth = _evaluate_scenario(model, scenario)
        scenario_results.append(result)
        learned_weight_bytes = max(
            learned_weight_bytes,
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        )
        peak_tensor_bytes = max(
            peak_tensor_bytes,
            scenario.object_count * _IMAGE_SIZE[0] * _IMAGE_SIZE[1] * 4,
        )
        if scenario.family == "separated_motion":
            for candidate_count in (8, 32):
                planning_results.append(_evaluate_planning(model, truth, candidate_count))
    scenarios = tuple(scenario_results)
    planning = tuple(planning_results)
    failures = _gate_failures(scenarios, planning)
    return PerceptualScaleQualificationResult(
        schema=SCALE_QUALIFICATION_SCHEMA,
        manifest_sha256=perceptual_scale_manifest_sha256(),
        scenarios=scenarios,
        planning=planning,
        gate_failures=failures,
        full_perceptual_qualification=not failures,
        learned_weight_bytes=learned_weight_bytes,
        peak_run_tensor_bytes=peak_tensor_bytes,
    )


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return _finite_or_none(value)
    return value


def _capability_summary(
    result: PerceptualScaleQualificationResult,
    *,
    run_id: str,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    count_entries: dict[str, dict[str, Any]] = {}
    for count in (7, 8):
        selected = [item for item in result.scenarios if item.object_count == count]
        failures = [item for item in result.gate_failures if item.startswith(f"n{count}-")]
        count_entries[f"n{count}_full_dynamic_set"] = {
            "status": "passed" if not failures else "failed",
            "scenario_count": len(selected),
            "minimum_proposal_f1": min(item.proposal_f1 for item in selected),
            "minimum_persistent_id_accuracy": min(item.persistent_id_accuracy for item in selected),
            "maximum_current_position_rmse_m": max(
                item.current_position_rmse_m for item in selected
            ),
            "maximum_two_second_position_rmse_m": _finite_or_none(
                max(item.two_second_position_rmse_m for item in selected)
            ),
            "full_perceptual_qualification": not failures,
        }
    current_rmse = math.sqrt(
        sum(item.current_position_rmse_m**2 for item in result.scenarios) / len(result.scenarios)
    )
    finite_rollouts = [
        item.two_second_position_rmse_m
        for item in result.scenarios
        if math.isfinite(item.two_second_position_rmse_m)
    ]
    rollout_rmse = (
        math.sqrt(sum(value**2 for value in finite_rollouts) / len(finite_rollouts))
        if len(finite_rollouts) == len(result.scenarios)
        else None
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
    animations = [item.animation for item in result.scenarios if item.animation is not None]
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.full_perceptual_qualification else "failed",
        outcome=(
            "qualified_convergence"
            if result.full_perceptual_qualification
            else "capability_gate_failed"
        ),
        source_format=SCALE_QUALIFICATION_SCHEMA,
        configuration={
            "model_config": str(Path(model_config_path)),
            "object_counts": [7, 8],
            "birth_proposals": 2,
            "frame_rate": _FRAME_RATE,
            "image_size": list(_IMAGE_SIZE),
            "planning_used_as_training_loss": False,
            "generated_episodes_retained": False,
        },
        provenance={
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "scenario_manifest_sha256": result.manifest_sha256,
            "animation_source_run": run_id,
        },
        scores={
            "candidate": current_rmse + (rollout_rmse or 1.0),
            "incumbent": current_rmse + (rollout_rmse or 1.0),
            "selected": "incumbent",
        },
        factor_metrics=count_entries,
        cell_metrics={
            f"N{item.object_count}/{item.family}": _json_safe(item.to_dict())
            for item in result.scenarios
        },
        horizon_curves={
            "candidate_position_rmse_m": {
                "0": current_rmse,
                "2": rollout_rmse,
            }
        },
        uncertainty={"status": "inherited incumbent calibration; not re-estimated"},
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
                (item.maximum_cost_difference for item in result.planning), default=None
            ),
        },
        resources={
            "learned_weight_bytes": result.learned_weight_bytes,
            "peak_run_tensor_bytes": result.peak_run_tensor_bytes,
            "maximum_scenario_latency_seconds": max(
                item.latency_seconds for item in result.scenarios
            ),
            "scalability": {
                "full_perceptual_qualification": result.full_perceptual_qualification,
                "qualified_object_counts": ([7, 8] if result.full_perceptual_qualification else []),
            },
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "incumbent",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "isolated capability ladder",
        },
        qualitative={
            "best_episode": "n8-separated_motion",
            "representative_episode": "n8-compositional",
            "worst_episode": "n8-pair_contact",
            "animations": animations[:3],
            "forecast_animations": [],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "full perceptual qualification above eight objects",
            "mixed non-spherical rigid-body qualification",
        ),
        scope_limitations=(
            "known calibrated cameras",
            "known public actions only",
            "sphere-only N=7/8 qualification",
            "short partial occlusion",
        ),
    ).validate()


def publish_perceptual_scale_qualification(
    result: PerceptualScaleQualificationResult,
    *,
    run_directory: str | Path,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    """Publish compact terminal evidence and rebuild the static dashboard."""

    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    report_path = run / "perceptual_scale_qualification.json"
    atomic_write_text(
        report_path,
        json.dumps(_json_safe(result.to_dict()), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_bytes = int(inventory_runs(runs_root, archive_root=archive_root)["archive_bytes"])
    summary = _capability_summary(
        result,
        run_id=run.name,
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        run_bytes=0,
        archive_bytes=archive_bytes,
    )
    for _ in range(8):
        write_capability_summary(summary, run / "capability_summary.json")
        write_run_report(summary, run)
        write_run_manifest(
            run,
            role="candidate",
            status="completed" if result.full_perceptual_qualification else "failed",
            artifacts={
                "perceptual_scale_qualification.json": "summary",
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
        raise RuntimeError("perceptual scale evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "PerceptualScalePlanningResult",
    "PerceptualScaleQualificationResult",
    "PerceptualScaleScenario",
    "PerceptualScaleScenarioResult",
    "default_perceptual_scale_scenarios",
    "perceptual_scale_manifest_sha256",
    "publish_perceptual_scale_qualification",
    "run_perceptual_scale_qualification",
]
