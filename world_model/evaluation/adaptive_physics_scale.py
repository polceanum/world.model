"""Single-model N=4/6/8 adaptive-physics qualification from public RGB-D traces."""

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

from world_model.belief import RigidPrimitive, WorldBelief, slow_packing_map
from world_model.dynamics import DynamicsModel, quaternion_geodesic_distance
from world_model.evaluation import multicontact_six_dof as state_gate
from world_model.evaluation import visual_dynamic_scale as visual_gate
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.identification import (
    BoundaryCollisionPositionEvidence,
    FreeMotionPositionEvidence,
    KnownImpulsePositionEvidence,
    OnlineRigidParameterEstimator,
)
from world_model.observations.rgbd import discover_rigid_objects_from_rgbd
from world_model.simulator import (
    PhysicsConfig,
    RigidBodyState,
    advance_rigid_bodies_6dof,
    make_intrinsics,
    render_rigid_bodies,
)
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

ADAPTIVE_PHYSICS_SCALE_SCHEMA = "world_model_adaptive_physics_scale_v1"
_COUNTS = (4, 6, 8)
_DTYPE = torch.float64
_FORECAST_DT = 0.05
_FORECAST_SECONDS = 4.0
_PROBE_IMAGE_SIZE = (128, 128)
_PROBE_FOCAL_LENGTH = 74.6666666667
_MASS = torch.tensor([0.65, 1.45, 0.85, 1.80, 0.72, 1.25, 1.60, 0.95], dtype=_DTYPE)
_DRAG = torch.tensor([0.09, 0.12, 0.20, 0.08, 0.25, 0.16, 0.07, 0.22], dtype=_DTYPE)
_RESTITUTION = torch.tensor([0.30, 0.75, 0.50, 0.85, 0.40, 0.65, 0.55, 0.90], dtype=_DTYPE)
_FRICTION = torch.tensor([0.12, 0.48, 0.28, 0.62, 0.18, 0.38, 0.52, 0.22], dtype=_DTYPE)
_PROBE_PHYSICS = PhysicsConfig(
    gravity=(0.0, 0.0, 0.0),
    bounds=((-8.0, 8.0), (-5.0, 5.0), (1.0, 7.0)),
    max_substep=1.0 / 120.0,
    solver_iterations=4,
)


@dataclass(frozen=True, slots=True)
class AdaptivePhysicsScenarioResult:
    object_count: int
    proposal_f1: float
    persistent_id_accuracy: float
    lifecycle_f1: float
    current_position_rmse_m: float
    accepted_parameter_updates: int
    expected_parameter_updates: int
    initial_parameter_errors: dict[str, dict[str, float]]
    final_parameter_errors: dict[str, dict[str, float]]
    uncertainty_contracted_fraction: float
    public_probe_frame_count: int
    public_probe_maximum_fit_error_m: float
    maximum_position_rmse_m: float
    endpoint_position_rmse_m: float
    contact_aligned_maximum_velocity_rmse_mps: float
    contact_aligned_p95_velocity_rmse_mps: float
    endpoint_velocity_rmse_mps: float
    maximum_orientation_rmse_degrees: float
    contact_pair_f1: float
    repeated_contact_frame_f1: float
    tolerance_aligned_repeated_contact_frame_f1: float
    first_contact_timing_error_frames: int | None
    rollout_latency_seconds: float
    oracle_parameter_endpoint_position_rmse_m: float
    oracle_parameter_maximum_orientation_rmse_degrees: float
    truth_state_endpoint_position_rmse_m: float
    truth_state_maximum_orientation_rmse_degrees: float
    parameter_owned_endpoint_excess_m: float
    parameter_owned_orientation_excess_degrees: float
    source_unchanged: bool
    finite: bool
    position_curve: dict[str, float]
    velocity_curve: dict[str, float]
    orientation_curve_degrees: dict[str, float]
    per_object_maximum_errors: dict[str, dict[str, float]]
    parameter_stages: tuple[dict[str, Any], ...]
    animation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AdaptivePhysicsScaleResult:
    schema: str
    manifest_sha256: str
    scenarios: tuple[AdaptivePhysicsScenarioResult, ...]
    planning: tuple[state_gate.MultiContactPlanningResult, ...]
    gate_failures: tuple[str, ...]
    qualified: bool
    learned_weight_bytes: int
    peak_probe_tensor_bytes: int
    evaluation_seconds: float
    generated_frames_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adaptive_physics_manifest_sha256() -> str:
    payload = {
        "schema": ADAPTIVE_PHYSICS_SCALE_SCHEMA,
        "single_model": True,
        "object_counts": list(_COUNTS),
        "parameters": {
            "mass": _MASS.tolist(),
            "drag": _DRAG.tolist(),
            "restitution": _RESTITUTION.tolist(),
            "friction": _FRICTION.tolist(),
        },
        "public_evidence": {
            "inputs": ["rgb", "depth", "world_from_camera", "intrinsics", "timestamp"],
            "free_motion_times_seconds": [0.0, 0.5, 1.0, 1.5, 2.0],
            "known_impulse_times_seconds": [2.0, 2.15, 2.30],
            "boundary_times_seconds": [-0.30, -0.15, 0.0, 0.15, 0.30],
            "isolated_control_probe": True,
            "calibrated_view_count": 6,
            "private_velocity_input": False,
            "private_parameter_input": False,
        },
        "forecast": {
            "seconds": _FORECAST_SECONDS,
            "frame_dt_seconds": _FORECAST_DT,
            "model_max_substep_seconds": 1.0 / 60.0,
            "reference_max_substep_seconds": 1.0 / 120.0,
            "known_actions": 3,
            "mixed_rigid_contacts": True,
        },
        "planning": {"object_count": 8, "candidate_counts": [8, 32], "loss": False},
        "acceptance": {
            "maximum_mean_parameter_relative_error": 0.06,
            "maximum_per_parameter_relative_error": {
                "mass": 0.04,
                "drag": 0.12,
                "restitution": 0.03,
                "friction": 0.20,
            },
            "maximum_position_rmse_m": 0.045,
            "maximum_two_second_orientation_rmse_degrees": 10.0,
            "maximum_four_second_orientation_rmse_degrees": 24.0,
            "maximum_endpoint_velocity_rmse_mps": 0.035,
            "maximum_contact_aligned_velocity_rmse_mps": 0.150,
            "maximum_contact_aligned_p95_velocity_rmse_mps": 0.040,
            "minimum_contact_pair_f1": 0.90,
            "minimum_one_frame_aligned_repeated_contact_f1": 0.60,
            "maximum_contact_timing_error_frames": 1,
            "maximum_n8_rollout_latency_seconds": 36.0,
        },
        "retained_media": "inline vector only",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _heterogeneous_truth(state: RigidBodyState, object_count: int) -> RigidBodyState:
    result = replace(
        state,
        mass=_MASS[:object_count, None].clone(),
        drag=_DRAG[:object_count, None].clone(),
        restitution=_RESTITUTION[:object_count, None].clone(),
        friction=_FRICTION[:object_count, None].clone(),
    )
    result.validate()
    return result


def _parameter_errors(
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> dict[str, dict[str, float]]:
    fields = {
        "mass": (belief.objects.mass[0, :, 0], truth.mass[truth_by_slot, 0]),
        "drag": (belief.objects.drag[0, :, 0], truth.drag[truth_by_slot, 0]),
        "restitution": (
            belief.objects.restitution[0, :, 0],
            truth.restitution[truth_by_slot, 0],
        ),
        "friction": (belief.objects.friction[0, :, 0], truth.friction[truth_by_slot, 0]),
    }
    result = {}
    all_errors = []
    for name, (estimate, target) in fields.items():
        relative = (estimate - target).abs() / target
        result[name] = {"mean": float(relative.mean()), "maximum": float(relative.max())}
        all_errors.append(relative)
    combined = torch.cat(all_errors)
    result["all"] = {"mean": float(combined.mean()), "maximum": float(combined.max())}
    return result


def _parameter_stage(
    stage: str,
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> dict[str, Any]:
    errors = _parameter_errors(belief, truth, truth_by_slot)
    return {
        "stage": stage,
        "mean_relative_error": errors["all"]["mean"],
        **{
            f"{name}_relative_error": errors[name]["mean"]
            for name in (
                "mass",
                "restitution",
                "drag",
                "friction",
            )
        },
    }


def _single_probe_state(
    truth: RigidBodyState,
    truth_index: int,
    position: Tensor,
    velocity: Tensor,
) -> RigidBodyState:
    active = torch.zeros_like(truth.active)
    active[truth_index] = True
    object_id = torch.full_like(truth.object_id, -1)
    object_id[truth_index] = truth.object_id[truth_index]
    positions = truth.position.clone()
    velocities = torch.zeros_like(truth.velocity)
    orientations = truth.orientation.clone()
    angular_velocity = torch.zeros_like(truth.angular_velocity)
    positions[truth_index] = position
    velocities[truth_index] = velocity
    orientations[truth_index] = position.new_tensor([0.0, 0.0, 0.0, 1.0])
    result = replace(
        truth,
        active=active,
        object_id=object_id,
        position=positions,
        velocity=velocities,
        orientation=orientations,
        angular_velocity=angular_velocity,
    )
    result.validate()
    return result


def _probe_cameras() -> tuple[Any, ...]:
    return tuple(
        replace(
            camera,
            intrinsics=make_intrinsics(
                _PROBE_IMAGE_SIZE,
                _PROBE_FOCAL_LENGTH,
                dtype=_DTYPE,
            ),
        )
        for camera in visual_gate._cameras(0.0)
    )


def _observe_single_position(state: RigidBodyState, timestamp: float) -> tuple[Tensor, float]:
    cameras = _probe_cameras()
    with torch.no_grad():
        rendered = [render_rigid_bodies(state, camera, _PROBE_IMAGE_SIZE) for camera in cameras]
        frame = discover_rigid_objects_from_rgbd(
            torch.stack([item.rgb.to(_DTYPE).permute(1, 2, 0) for item in rendered]),
            torch.stack([item.depth_buffer for item in rendered]),
            torch.stack([item.world_from_camera for item in cameras]),
            torch.stack([item.intrinsics for item in cameras]),
            timestamp=timestamp,
            box_decision_ratio=0.9,
        )
    if len(frame.objects) != 1:
        raise RuntimeError(
            f"public calibration probe at {timestamp:.3f}s yielded {len(frame.objects)} objects"
        )
    observed = frame.objects[0].geometry.world_position
    target_slot = int(torch.where(state.active)[0][0])
    fit_error = float(torch.linalg.vector_norm(observed - state.position[target_slot]))
    return observed, fit_error


def _motion_trace(
    truth: RigidBodyState,
    truth_index: int,
    times: Tensor,
    origin: Tensor,
    velocity: Tensor,
) -> tuple[RigidBodyState, Tensor, float]:
    state = _single_probe_state(truth, truth_index, origin, velocity)
    first, first_error = _observe_single_position(state, float(times[0]))
    positions = [first]
    errors = [first_error]
    for start, end in zip(times[:-1], times[1:], strict=True):
        state, _ = advance_rigid_bodies_6dof(
            state,
            float(end - start),
            _PROBE_PHYSICS,
        )
        observed, error = _observe_single_position(state, float(end))
        positions.append(observed)
        errors.append(error)
    return state, torch.stack(positions), max(errors)


def _analytic_probe_positions(
    truth: RigidBodyState,
    truth_index: int,
    times: Tensor,
    collision_position: Tensor,
    velocity_at_collision: Tensor,
) -> tuple[Tensor, float]:
    drag = truth.drag[truth_index, 0]
    scale = -torch.expm1(-drag * times) / drag
    positions = collision_position + scale[:, None] * velocity_at_collision
    observed = []
    errors = []
    for timestamp, position in zip(times, positions, strict=True):
        velocity = velocity_at_collision * torch.exp(-drag * timestamp)
        state = _single_probe_state(truth, truth_index, position, velocity)
        measurement, error = _observe_single_position(state, float(timestamp))
        observed.append(measurement)
        errors.append(error)
    return torch.stack(observed), max(errors)


def _calibrate_parameters(
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> tuple[WorldBelief, int, int, float, tuple[dict[str, Any], ...]]:
    estimator = OnlineRigidParameterEstimator(gain=1.0, variance_contraction=0.5)
    initial = belief.objects.slow_log_variance.clone()
    stages = [_parameter_stage("neutral priors", belief, truth, truth_by_slot)]
    mass_evidence = []
    accepted = 0
    frame_count = 0
    maximum_fit_error = 0.0
    free_times = belief.timestamp.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    for slot in range(belief.objects.max_objects):
        truth_index = int(truth_by_slot[slot])
        object_id = belief.objects.object_id[:, slot].clone()
        direction = -1.0 if slot % 2 else 1.0
        state, positions, fit_error = _motion_trace(
            truth,
            truth_index,
            free_times,
            belief.timestamp.new_tensor([0.0, 0.0, 4.0]),
            belief.timestamp.new_tensor([0.9 + 0.04 * slot, direction * 0.28, 0.08]),
        )
        frame_count += len(free_times)
        maximum_fit_error = max(maximum_fit_error, fit_error)
        belief, update = estimator.update_free_motion_positions(
            belief,
            FreeMotionPositionEvidence(object_id, positions.unsqueeze(0), free_times.unsqueeze(0)),
            maximum_fit_residual_m=0.015,
        )
        accepted += int(update.accepted.sum())

        impulse = belief.timestamp.new_tensor([0.96 + 0.05 * slot, direction * 0.34, 0.12])
        external = torch.zeros_like(state.velocity)
        external[truth_index] = impulse
        after_positions = [positions[-1]]
        for step, endpoint in enumerate((2.15, 2.30)):
            state, _ = advance_rigid_bodies_6dof(
                state,
                0.15,
                _PROBE_PHYSICS,
                external_impulse=external if step == 0 else None,
            )
            observed, fit_error = _observe_single_position(state, endpoint)
            after_positions.append(observed)
            maximum_fit_error = max(maximum_fit_error, fit_error)
        frame_count += 2
        mass_evidence.append(
            KnownImpulsePositionEvidence(
                object_id=object_id,
                positions_before_world=positions[-3:].unsqueeze(0),
                timestamps_before=free_times[-3:].unsqueeze(0),
                positions_after_world=torch.stack(after_positions).unsqueeze(0),
                timestamps_after=belief.timestamp.new_tensor([[2.0, 2.15, 2.30]]),
                action_timestamp=belief.timestamp.new_tensor([2.0]),
                impulse_world=impulse.unsqueeze(0),
            )
        )
    stages.append(_parameter_stage("public free-motion traces", belief, truth, truth_by_slot))

    for evidence in mass_evidence:
        belief, update = estimator.update_known_impulse_positions(
            belief,
            evidence,
            maximum_fit_residual_m=0.015,
        )
        accepted += int(update.accepted.sum())
    stages.append(_parameter_stage("public known impulses", belief, truth, truth_by_slot))

    before_times = belief.timestamp.new_tensor([-0.30, -0.15, 0.0])
    after_times = belief.timestamp.new_tensor([0.0, 0.15, 0.30])
    normal = belief.timestamp.new_tensor([-1.0, 0.0, 0.0])
    for slot in range(belief.objects.max_objects):
        truth_index = int(truth_by_slot[slot])
        restitution = truth.restitution[truth_index, 0]
        friction = truth.friction[truth_index, 0]
        before_velocity = belief.timestamp.new_tensor([4.0, 5.0, 0.0])
        after_velocity = torch.stack(
            (
                -4.0 * restitution,
                5.0 - 4.0 * friction * (1.0 + restitution),
                restitution * 0.0,
            )
        )
        support = (
            truth.half_extents[truth_index, 0]
            if int(truth.primitive[truth_index]) == int(RigidPrimitive.BOX)
            else truth.radius[truth_index, 0]
        )
        collision_position = torch.stack(
            (support.new_tensor(1.75) - support, support * 0.0, support.new_tensor(4.0))
        )
        before_positions, before_error = _analytic_probe_positions(
            truth,
            truth_index,
            before_times,
            collision_position,
            before_velocity,
        )
        after_positions, after_error = _analytic_probe_positions(
            truth,
            truth_index,
            after_times,
            collision_position,
            after_velocity,
        )
        frame_count += 6
        maximum_fit_error = max(maximum_fit_error, before_error, after_error)
        belief, updates = estimator.update_boundary_collision_positions(
            belief,
            BoundaryCollisionPositionEvidence(
                object_id=belief.objects.object_id[:, slot].clone(),
                normal_world=normal.unsqueeze(0),
                positions_before_world=before_positions.unsqueeze(0),
                timestamps_before=before_times.unsqueeze(0),
                positions_after_world=after_positions.unsqueeze(0),
                timestamps_after=after_times.unsqueeze(0),
                collision_timestamp=belief.timestamp.new_tensor([0.0]),
            ),
            maximum_fit_residual_m=0.015,
        )
        accepted += sum(int(update.accepted.sum()) for update in updates)
    stages.append(_parameter_stage("public boundary impacts", belief, truth, truth_by_slot))

    parameter_slices = slow_packing_map(belief.objects)
    indices = torch.cat(
        [
            torch.arange(parameter_slices[name].start, parameter_slices[name].stop)
            for name in ("log_mass", "restitution_logit", "log_drag", "friction_logit")
        ]
    )
    contracted = (belief.objects.slow_log_variance[..., indices] < initial[..., indices]).all(
        dim=-1
    )
    contracted_count = int(contracted[belief.objects.active].sum())
    return belief, accepted, contracted_count, maximum_fit_error, tuple(stages)


def _dynamics(belief: WorldBelief) -> DynamicsModel:
    model = DynamicsModel.from_belief(
        belief,
        max_substep=1.0 / 60.0,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=16,
        interaction_radius=2.5,
        world_bounds=state_gate._BOUNDS,
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


def _truth_parameter_belief(
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> WorldBelief:
    objects = belief.objects.clone()
    objects.log_mass[0, :, 0] = truth.mass[truth_by_slot, 0].log()
    objects.log_drag[0, :, 0] = truth.drag[truth_by_slot, 0].log()
    objects.restitution_logit[0, :, 0] = torch.logit(truth.restitution[truth_by_slot, 0])
    objects.friction_logit[0, :, 0] = torch.logit(truth.friction[truth_by_slot, 0])
    return replace(belief, objects=objects).validate()


def _truth_state_belief(
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> WorldBelief:
    objects = belief.objects.clone()
    objects.position[0] = truth.position[truth_by_slot]
    objects.velocity[0] = truth.velocity[truth_by_slot]
    objects.orientation[0] = truth.orientation[truth_by_slot]
    objects.angular_velocity[0] = truth.angular_velocity[truth_by_slot]
    objects.log_mass[0, :, 0] = truth.mass[truth_by_slot, 0].log()
    objects.log_drag[0, :, 0] = truth.drag[truth_by_slot, 0].log()
    objects.restitution_logit[0, :, 0] = torch.logit(truth.restitution[truth_by_slot, 0])
    objects.friction_logit[0, :, 0] = torch.logit(truth.friction[truth_by_slot, 0])
    return replace(belief, objects=objects).validate()


def _contact_aligned_velocity_error(
    model_velocity: Tensor,
    reference_velocity: Tensor,
    model_collision: Tensor,
    reference_collision: Tensor,
) -> Tensor:
    model_events = model_collision.any(dim=(-2, -1))
    reference_events = reference_collision.any(dim=(-2, -1))
    event_window = model_events | reference_events
    if len(event_window) > 1:
        event_window[1:] |= model_events[:-1] | reference_events[:-1]
        event_window[:-1] |= model_events[1:] | reference_events[1:]
    errors = []
    for index in range(len(model_velocity)):
        if bool(event_window[index]):
            candidates = range(max(0, index - 1), min(len(reference_velocity), index + 2))
            errors.append(
                torch.stack(
                    [
                        (model_velocity[index] - reference_velocity[candidate])
                        .square()
                        .mean()
                        .sqrt()
                        for candidate in candidates
                    ]
                ).min()
            )
        else:
            errors.append(
                (model_velocity[index] - reference_velocity[index]).square().mean().sqrt()
            )
    return torch.stack(errors)


def _tolerance_aligned_repeated_contact_f1(
    model_collision: Tensor,
    reference_collision: Tensor,
    *,
    tolerance_frames: int = 1,
) -> float:
    """Score repeated pair events while allowing a bounded timing displacement."""

    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be nonnegative")
    if model_collision.shape != reference_collision.shape or model_collision.ndim != 3:
        raise ValueError("collision traces must share shape [T,N,N]")
    true_positive = false_positive = false_negative = 0
    object_count = model_collision.shape[-1]
    for first in range(object_count):
        for second in range(first + 1, object_count):
            model_frames = torch.where(model_collision[:, first, second])[0].tolist()
            reference_frames = torch.where(reference_collision[:, first, second])[0].tolist()
            unmatched = set(reference_frames)
            matched = 0
            for model_frame in model_frames:
                candidates = [
                    frame for frame in unmatched if abs(frame - model_frame) <= tolerance_frames
                ]
                if not candidates:
                    continue
                selected = min(candidates, key=lambda frame: (abs(frame - model_frame), frame))
                unmatched.remove(selected)
                matched += 1
            true_positive += matched
            false_positive += len(model_frames) - matched
            false_negative += len(unmatched)
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


def _evaluate_scenario(
    object_count: int,
) -> tuple[
    AdaptivePhysicsScenarioResult,
    DynamicsModel,
    WorldBelief,
    RigidBodyState,
    Tensor,
]:
    visual_result, _, belief, base_truth, truth_by_slot = visual_gate._evaluate_scenario(
        object_count
    )
    truth = _heterogeneous_truth(base_truth, object_count)
    initial_errors = _parameter_errors(belief, truth, truth_by_slot)
    belief, accepted, contracted, fit_error, stages = _calibrate_parameters(
        belief,
        truth,
        truth_by_slot,
    )
    final_errors = _parameter_errors(belief, truth, truth_by_slot)
    dynamics = _dynamics(belief)
    query_times = torch.arange(
        _FORECAST_DT,
        _FORECAST_SECONDS + 0.5 * _FORECAST_DT,
        _FORECAST_DT,
        dtype=belief.dtype,
    )
    source = belief.clone()
    schedule = visual_gate._model_schedule(belief, truth_by_slot, object_count)
    started = time.perf_counter()
    with torch.no_grad():
        prediction = dynamics.rollout(belief, query_times, action=schedule)
    rollout_latency = time.perf_counter() - started
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
    model_position = prediction.positions[0]
    model_velocity = prediction.velocities[0]
    model_orientation = prediction.orientations[0]
    model_collision = prediction.auxiliary["pair_collision"][0]
    position_error = (model_position - truth_position).square().mean(dim=(-2, -1)).sqrt()
    velocity_error = (model_velocity - truth_velocity).square().mean(dim=(-2, -1)).sqrt()
    aligned_velocity = _contact_aligned_velocity_error(
        model_velocity,
        truth_velocity,
        model_collision,
        truth_collision,
    )
    box_slots = torch.where(truth.primitive[truth_by_slot] == int(RigidPrimitive.BOX))[0]
    orientation_error = (
        (quaternion_geodesic_distance(model_orientation, truth_orientation) * (180.0 / math.pi))[
            :, box_slots
        ]
        .square()
        .mean(dim=-1)
        .sqrt()
    )
    (
        _,
        _,
        contact_pair_f1,
        first_contact_timing,
        repeated_contact_f1,
        _,
    ) = state_gate._contact_metrics(model_collision, truth_collision)
    aligned_repeated_contact_f1 = _tolerance_aligned_repeated_contact_f1(
        model_collision,
        truth_collision,
    )

    oracle_belief = _truth_parameter_belief(belief, truth, truth_by_slot)
    oracle_dynamics = _dynamics(oracle_belief)
    with torch.no_grad():
        oracle = oracle_dynamics.rollout(oracle_belief, query_times, action=schedule)
    oracle_endpoint = float((oracle.positions[0, -1] - truth_position[-1]).square().mean().sqrt())
    oracle_orientation_error = (
        (
            quaternion_geodesic_distance(oracle.orientations[0], truth_orientation)
            * (180.0 / math.pi)
        )[:, box_slots]
        .square()
        .mean(dim=-1)
        .sqrt()
    )
    truth_state_belief = _truth_state_belief(belief, truth, truth_by_slot)
    truth_state_dynamics = _dynamics(truth_state_belief)
    with torch.no_grad():
        truth_state_prediction = truth_state_dynamics.rollout(
            truth_state_belief,
            query_times,
            action=schedule,
        )
    truth_state_endpoint = float(
        (truth_state_prediction.positions[0, -1] - truth_position[-1]).square().mean().sqrt()
    )
    truth_state_orientation_error = (
        (
            quaternion_geodesic_distance(
                truth_state_prediction.orientations[0],
                truth_orientation,
            )
            * (180.0 / math.pi)
        )[:, box_slots]
        .square()
        .mean(dim=-1)
        .sqrt()
    )
    endpoint = float(position_error[-1])
    position_error_by_object = torch.linalg.vector_norm(
        model_position - truth_position,
        dim=-1,
    )
    velocity_error_by_object = torch.linalg.vector_norm(
        model_velocity - truth_velocity,
        dim=-1,
    )
    orientation_error_by_object = quaternion_geodesic_distance(
        model_orientation,
        truth_orientation,
    ) * (180.0 / math.pi)
    runtime_ids = belief.objects.object_id[0]
    per_object_maximum_errors = {
        str(int(runtime_ids[slot])): {
            "position_m": float(position_error_by_object[:, slot].max()),
            "velocity_mps": float(velocity_error_by_object[:, slot].max()),
            **(
                {"orientation_degrees": float(orientation_error_by_object[:, slot].max())}
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
        model_position,
        truth_position,
        model_orientation,
        truth_orientation,
        truth_collision,
        position_error,
    )
    animation.update(
        {
            "label": f"N={object_count} public RGB-D adaptive-physics four-second forecast",
            "episode": f"adaptive-physics-n{object_count}",
            "long_horizon_endpoint_s": _FORECAST_SECONDS,
            "parameter_source": "public RGB-D free-motion, known-impulse, and boundary traces",
        }
    )
    return (
        AdaptivePhysicsScenarioResult(
            object_count=object_count,
            proposal_f1=visual_result.proposal_f1,
            persistent_id_accuracy=visual_result.persistent_id_accuracy,
            lifecycle_f1=visual_result.lifecycle_f1,
            current_position_rmse_m=visual_result.current_position_rmse_m,
            accepted_parameter_updates=accepted,
            expected_parameter_updates=4 * object_count,
            initial_parameter_errors=initial_errors,
            final_parameter_errors=final_errors,
            uncertainty_contracted_fraction=contracted / object_count,
            public_probe_frame_count=13 * object_count,
            public_probe_maximum_fit_error_m=fit_error,
            maximum_position_rmse_m=float(position_error.max()),
            endpoint_position_rmse_m=endpoint,
            contact_aligned_maximum_velocity_rmse_mps=float(aligned_velocity.max()),
            contact_aligned_p95_velocity_rmse_mps=float(torch.quantile(aligned_velocity, 0.95)),
            endpoint_velocity_rmse_mps=float(velocity_error[-1]),
            maximum_orientation_rmse_degrees=float(orientation_error.max()),
            contact_pair_f1=contact_pair_f1,
            repeated_contact_frame_f1=repeated_contact_f1,
            tolerance_aligned_repeated_contact_frame_f1=aligned_repeated_contact_f1,
            first_contact_timing_error_frames=first_contact_timing,
            rollout_latency_seconds=rollout_latency,
            oracle_parameter_endpoint_position_rmse_m=oracle_endpoint,
            oracle_parameter_maximum_orientation_rmse_degrees=float(oracle_orientation_error.max()),
            truth_state_endpoint_position_rmse_m=truth_state_endpoint,
            truth_state_maximum_orientation_rmse_degrees=float(truth_state_orientation_error.max()),
            parameter_owned_endpoint_excess_m=max(endpoint - oracle_endpoint, 0.0),
            parameter_owned_orientation_excess_degrees=max(
                float(orientation_error.max() - oracle_orientation_error.max()),
                0.0,
            ),
            source_unchanged=(
                torch.equal(source.objects.position, belief.objects.position)
                and torch.equal(source.objects.velocity, belief.objects.velocity)
                and torch.equal(source.objects.log_mass, belief.objects.log_mass)
                and torch.equal(source.objects.log_drag, belief.objects.log_drag)
                and torch.equal(source.objects.restitution_logit, belief.objects.restitution_logit)
                and torch.equal(source.objects.friction_logit, belief.objects.friction_logit)
            ),
            finite=bool(
                torch.isfinite(model_position).all()
                and torch.isfinite(model_velocity).all()
                and torch.isfinite(model_orientation).all()
            ),
            position_curve={
                f"{float(value):.2f}": float(position_error[index])
                for index, value in enumerate(query_times)
            },
            velocity_curve={
                f"{float(value):.2f}": float(aligned_velocity[index])
                for index, value in enumerate(query_times)
            },
            orientation_curve_degrees={
                f"{float(value):.2f}": float(orientation_error[index])
                for index, value in enumerate(query_times)
            },
            per_object_maximum_errors=per_object_maximum_errors,
            parameter_stages=stages,
            animation=animation,
        ),
        dynamics,
        belief,
        truth,
        truth_by_slot,
    )


def _gate_failures(
    scenarios: tuple[AdaptivePhysicsScenarioResult, ...],
    planning: tuple[state_gate.MultiContactPlanningResult, ...],
    *,
    enforce_latency: bool,
) -> tuple[str, ...]:
    failures = []
    maximum_parameter_error = {
        "mass": 0.04,
        "drag": 0.12,
        "restitution": 0.03,
        "friction": 0.20,
    }
    for item in scenarios:
        prefix = f"n{item.object_count}"
        maximum_object_position = max(
            metrics["position_m"] for metrics in item.per_object_maximum_errors.values()
        )
        maximum_box_orientation = max(
            metrics["orientation_degrees"]
            for metrics in item.per_object_maximum_errors.values()
            if "orientation_degrees" in metrics
        )
        checks = {
            "proposal": item.proposal_f1 >= 0.95,
            "identity": item.persistent_id_accuracy >= 0.98,
            "lifecycle": item.lifecycle_f1 >= 0.95,
            "current_position": item.current_position_rmse_m <= 0.020,
            "all_parameter_updates": (
                item.accepted_parameter_updates == item.expected_parameter_updates
            ),
            "parameter_mean": item.final_parameter_errors["all"]["mean"] <= 0.06,
            "uncertainty_contracted": item.uncertainty_contracted_fraction == 1.0,
            "public_probe_fit": item.public_probe_maximum_fit_error_m <= 0.020,
            "position": item.maximum_position_rmse_m <= 0.045,
            "endpoint_position": item.endpoint_position_rmse_m <= 0.045,
            "two_second_position": item.position_curve["2.00"] <= 0.030,
            "worst_object_position": maximum_object_position <= 0.100,
            "contact_aligned_velocity": (item.contact_aligned_maximum_velocity_rmse_mps <= 0.150),
            "contact_aligned_p95_velocity": (item.contact_aligned_p95_velocity_rmse_mps <= 0.040),
            "endpoint_velocity": item.endpoint_velocity_rmse_mps <= 0.035,
            "two_second_orientation": item.orientation_curve_degrees["2.00"] <= 10.0,
            "four_second_orientation": item.maximum_orientation_rmse_degrees <= 24.0,
            "worst_box_orientation": maximum_box_orientation <= 45.0,
            "contact_pairs": item.contact_pair_f1 >= 0.90,
            "aligned_repeated_contacts": (item.tolerance_aligned_repeated_contact_frame_f1 >= 0.60),
            "contact_timing": item.first_contact_timing_error_frames is not None
            and item.first_contact_timing_error_frames <= 1,
            "parameter_owned_excess": item.parameter_owned_endpoint_excess_m <= 0.020,
            "parameter_owned_orientation": (item.parameter_owned_orientation_excess_degrees <= 2.0),
            "source_unchanged": item.source_unchanged,
            "finite": item.finite,
        }
        checks.update(
            {
                f"{name}_maximum": item.final_parameter_errors[name]["maximum"] <= limit
                for name, limit in maximum_parameter_error.items()
            }
        )
        if enforce_latency:
            checks["latency"] = (
                item.rollout_latency_seconds <= {4: 12.0, 6: 22.0, 8: 36.0}[item.object_count]
            )
        failures.extend(f"{prefix}:{name}" for name, passed in checks.items() if not passed)
    for item in planning:
        checks = {
            "winner": item.winner_correct,
            "regret": item.normalized_regret == 0.0,
            "goal": item.goal_success,
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


def run_adaptive_physics_scale(*, enforce_latency: bool = True) -> AdaptivePhysicsScaleResult:
    """Run the governed single-model adaptive-physics scale milestone."""

    started = time.perf_counter()
    scenarios = []
    planning_inputs = None
    learned_weight_bytes = 0
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
        visual_gate._evaluate_planning(candidate_count, *planning_inputs)
        for candidate_count in (8, 32)
    )
    scenario_tuple = tuple(scenarios)
    failures = _gate_failures(scenario_tuple, planning, enforce_latency=enforce_latency)
    return AdaptivePhysicsScaleResult(
        schema=ADAPTIVE_PHYSICS_SCALE_SCHEMA,
        manifest_sha256=adaptive_physics_manifest_sha256(),
        scenarios=scenario_tuple,
        planning=planning,
        gate_failures=failures,
        qualified=not failures,
        learned_weight_bytes=learned_weight_bytes,
        peak_probe_tensor_bytes=6 * _PROBE_IMAGE_SIZE[0] * _PROBE_IMAGE_SIZE[1] * 5 * 8,
        evaluation_seconds=time.perf_counter() - started,
    )


def _summary(
    result: AdaptivePhysicsScaleResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    by_count = {item.object_count: item for item in result.scenarios}
    horizon_keys = result.scenarios[0].position_curve
    aggregate_position = {
        key: max(item.position_curve[key] for item in result.scenarios) for key in horizon_keys
    }
    aggregate_velocity = {
        key: max(item.velocity_curve[key] for item in result.scenarios) for key in horizon_keys
    }
    aggregate_orientation = {
        key: max(item.orientation_curve_degrees[key] for item in result.scenarios)
        for key in horizon_keys
    }
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
    candidate_score = sum(
        item.final_parameter_errors["all"]["mean"] for item in result.scenarios
    ) / len(result.scenarios)
    incumbent_score = sum(
        item.initial_parameter_errors["all"]["mean"] for item in result.scenarios
    ) / len(result.scenarios)
    parameter_stages = []
    for index, stage in enumerate(result.scenarios[0].parameter_stages):
        parameter_stages.append(
            {
                "stage": stage["stage"],
                **{
                    key: sum(item.parameter_stages[index][key] for item in result.scenarios)
                    / len(result.scenarios)
                    for key in (
                        "mean_relative_error",
                        "mass_relative_error",
                        "restitution_relative_error",
                        "drag_relative_error",
                        "friction_relative_error",
                    )
                },
            }
        )
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.qualified else "failed",
        outcome="qualified_convergence" if result.qualified else "capability_gate_failed",
        source_format=ADAPTIVE_PHYSICS_SCALE_SCHEMA,
        configuration={
            "single_model": True,
            "ensemble": False,
            "mixture_of_experts": False,
            "object_counts": list(_COUNTS),
            "public_calibrated_rgbd": True,
            "heterogeneous_per_object_parameters": True,
            "parameter_evidence": ["free motion", "known impulses", "boundary impacts"],
            "calibration_view_count": 6,
            "forecast_seconds": _FORECAST_SECONDS,
            "planning_used_as_training_loss": False,
            "generated_frames_retained": False,
        },
        provenance={
            "scenario_manifest_sha256": result.manifest_sha256,
            "belief_initialization": "prototype-free public calibrated RGB-D",
            "parameter_runtime_inputs": "public RGB-D position traces and known actions only",
            "private_velocity_runtime_inputs": False,
            "private_parameter_runtime_inputs": False,
            "truth_reference_opened_after_each_public_estimate": True,
            "truth_parameter_ablation": True,
            "dense_serial_planning_oracle": True,
        },
        scores={
            "candidate": {"value": candidate_score, "supported_weight": 1.0},
            "incumbent": {"value": incumbent_score, "supported_weight": 1.0},
            "selected": "single_model_adaptive_physics" if result.qualified else "incumbent",
        },
        factor_metrics={
            f"adaptive_physics_n{count}": {
                "status": "passed"
                if not any(failure.startswith(f"n{count}:") for failure in result.gate_failures)
                else "failed",
                "proposal_f1": item.proposal_f1,
                "identity_accuracy": item.persistent_id_accuracy,
                "lifecycle_f1": item.lifecycle_f1,
                "current_position_rmse_m": item.current_position_rmse_m,
                "two_second_position_rmse_m": item.position_curve["2.00"],
                "four_second_position_rmse_m": item.endpoint_position_rmse_m,
                "two_second_orientation_rmse_degrees": (item.orientation_curve_degrees["2.00"]),
                "four_second_orientation_rmse_degrees": (item.maximum_orientation_rmse_degrees),
                "collision_f1": item.contact_pair_f1,
                "raw_repeated_contact_f1": item.repeated_contact_frame_f1,
                "aligned_repeated_contact_f1": (item.tolerance_aligned_repeated_contact_frame_f1),
                "contact_aligned_velocity_p95_mps": (item.contact_aligned_p95_velocity_rmse_mps),
                "parameter_relative_error": item.final_parameter_errors["all"]["mean"],
            }
            for count, item in by_count.items()
        },
        cell_metrics={
            f"N{count}/adaptive_physics/mixed_rigid": {
                "parameter_relative_error": {
                    "value": item.final_parameter_errors["all"]["mean"],
                    "support": 4 * count,
                },
                "four_second_position_rmse_m": {
                    "value": item.endpoint_position_rmse_m,
                    "support": count,
                },
                "collision_f1": {
                    "value": item.contact_pair_f1,
                    "support": count - 1,
                },
            }
            for count, item in by_count.items()
        },
        horizon_curves={
            "candidate_position_rmse_m": aggregate_position,
            "candidate_velocity_rmse_mps": aggregate_velocity,
            "candidate_orientation_rmse_degrees": aggregate_orientation,
        },
        uncertainty={
            "status": "contracted only on accepted public parameter updates",
            "parameter_contracted_fraction": min(
                item.uncertainty_contracted_fraction for item in result.scenarios
            ),
        },
        planning={
            "status": "passed"
            if all(item.winner_correct for item in result.planning)
            else "failed",
            "goal": "N=8 heterogeneous-physics terminal position",
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
            "peak_run_tensor_bytes": result.peak_probe_tensor_bytes,
            "n8_rollout_latency_seconds": by_count[8].rollout_latency_seconds,
            "public_probe_frame_count": sum(
                item.public_probe_frame_count for item in result.scenarios
            ),
            "k32_vectorization_speedup": next(
                item.vectorization_speedup for item in result.planning if item.candidate_count == 32
            ),
        },
        artifacts={"run_bytes": run_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "single_model_adaptive_physics" if result.qualified else "none",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "public parameter evidence or shared analytic contact dynamics",
            "ablations": {
                **{
                    f"truth_parameters_n{item.object_count}": {
                        "error_reduction": item.parameter_owned_endpoint_excess_m,
                        "endpoint_position_rmse_m": (
                            item.oracle_parameter_endpoint_position_rmse_m
                        ),
                        "maximum_orientation_rmse_degrees": (
                            item.oracle_parameter_maximum_orientation_rmse_degrees
                        ),
                        "status": "diagnostic",
                    }
                    for item in result.scenarios
                },
                **{
                    f"truth_state_and_parameters_n{item.object_count}": {
                        "error_reduction": max(
                            item.endpoint_position_rmse_m
                            - item.truth_state_endpoint_position_rmse_m,
                            0.0,
                        ),
                        "endpoint_position_rmse_m": item.truth_state_endpoint_position_rmse_m,
                        "maximum_orientation_rmse_degrees": (
                            item.truth_state_maximum_orientation_rmse_degrees
                        ),
                        "status": "diagnostic",
                    }
                    for item in result.scenarios
                },
            },
        },
        qualitative={
            "best_episode": "adaptive-physics-n4",
            "representative_episode": "adaptive-physics-n6",
            "worst_episode": "adaptive-physics-n8",
            "parameter_convergence": parameter_stages,
            "per_object_prediction_errors": [
                {
                    "scenario": f"N={item.object_count}",
                    "runtime_id": object_id,
                    **metrics,
                }
                for item in result.scenarios
                for object_id, metrics in item.per_object_maximum_errors.items()
            ],
            "animations": [],
            "forecast_animations": [item.animation for item in result.scenarios],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "unknown camera calibration",
            "hidden actions",
            "visual qualification above eight objects",
            "deformable or articulated objects",
        ),
        scope_limitations=(
            "controlled isolated parameter-identification pre-rolls",
            "planning freezes the active set",
            "known stationary calibration boundary",
            "flush featureless object unions remain unobservable",
        ),
    ).validate()


def publish_adaptive_physics_scale(
    result: AdaptivePhysicsScaleResult,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run / "adaptive_physics_scale.json",
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
                "adaptive_physics_scale.json": "summary",
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
        raise RuntimeError("adaptive-physics evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "ADAPTIVE_PHYSICS_SCALE_SCHEMA",
    "AdaptivePhysicsScaleResult",
    "AdaptivePhysicsScenarioResult",
    "adaptive_physics_manifest_sha256",
    "publish_adaptive_physics_scale",
    "run_adaptive_physics_scale",
]
