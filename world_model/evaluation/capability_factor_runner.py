"""Streamed incumbent evaluation for controlled capability factors.

Each factor changes either simulator controls or the public RGB-D evidence,
then exposes only calibrated RGB-D and declared actions to the runtime.
Simulator objects, labels, and events remain private scoring truth.  A second
nominal stream over the same manifest rows provides the paired family-level
ablation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.evaluation.general_capability import (
    ALL_CAPABILITY_FACTORS,
    CAPABILITY_FACTORS,
    COMPOSITIONAL_FACTOR,
    CapabilityManifestRow,
    FactorControls,
    attribute_failure,
    capability_manifest,
    manifest_sha256,
)
from world_model.evaluation.scalability import compare_dense_and_packed_scalability
from world_model.runtime import OnlineWorldModel
from world_model.simulator.camera import CameraTrajectory, CameraTrajectoryConfig
from world_model.simulator.episode import validate_episode
from world_model.simulator.labels import make_perception_labels, validate_perception_labels
from world_model.simulator.physics import SphereState
from world_model.simulator.renderer import render_spheres
from world_model.training.capability_workbench import (
    DEFAULT_CAPABILITY_VELOCITY_VARIANCE_FLOOR,
    WORKBENCH_SCHEMA,
)
from world_model.training.dynamic_set_config import OrpheusConfig, load_config
from world_model.training.dynamic_set_evaluation import (
    BinaryCounts,
    DynamicSetCellAccumulator,
    DynamicSetEvaluationError,
    DynamicSetEvaluationResult,
    evaluate_dynamic_set_materializations,
)
from world_model.training.dynamic_set_materializer import (
    DynamicSetKnownAction,
    DynamicSetMaterialization,
    DynamicSetPhysicalParameters,
    materialize_dynamic_set_episode,
    materialize_dynamic_set_episode_with_action,
    materialize_dynamic_set_episode_with_controls,
    materialize_dynamic_set_episode_with_parameters,
)
from world_model.training.dynamic_set_protocol import (
    PHYSICAL_CELLS,
    SELECTION_SCORE_WEIGHTS,
    PhysicalCell,
    PhysicalManifestRow,
    physical_manifest,
)
from world_model.training.dynamic_set_scene import (
    DYNAMIC_SET_FRAMES,
    DYNAMIC_SET_IMAGE_SIZE,
    DYNAMIC_SET_MAX_OBJECTS,
    DYNAMIC_SET_VERTICAL_FOV_DEGREES,
)
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import enforce_run_budget, inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

CAPABILITY_FACTOR_REPORT_SCHEMA = "world_model_capability_factor_report_v1"
SUPPORTED_CAPABILITY_FACTORS = (
    "sensor_noise",
    "physical_parameters",
    "camera_motion",
    "known_actions",
    "partial_visibility",
    COMPOSITIONAL_FACTOR,
)
CapabilityFactor = Literal[
    "sensor_noise",
    "physical_parameters",
    "camera_motion",
    "known_actions",
    "partial_visibility",
    "compositional_holdout",
]

_CAPABILITY_ACTION_STRATUM = {"early": 0, "middle": 2, "late": 3}
_CAPABILITY_ACTION_FRAME = {"early": 10, "middle": 34, "late": 44}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize capability value of type {type(value).__name__}")


def _seed_process(seed: int, threads: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def _model_from_workbench_checkpoint(
    config: OrpheusConfig,
    checkpoint_path: str | Path,
) -> tuple[OnlineWorldModel, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != WORKBENCH_SCHEMA:
        raise ValueError("incumbent checkpoint is not a capability-workbench checkpoint")
    settings = payload.get("workbench_settings")
    state = payload.get("model_state")
    if not isinstance(settings, Mapping) or not isinstance(state, Mapping):
        raise ValueError("incumbent checkpoint lacks settings or model state")
    seed = settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("incumbent checkpoint has an invalid construction seed")
    variance_floor = settings.get(
        "velocity_variance_floor",
        DEFAULT_CAPABILITY_VELOCITY_VARIANCE_FLOOR,
    )
    if not isinstance(variance_floor, (int, float)) or isinstance(variance_floor, bool):
        raise ValueError("incumbent checkpoint has an invalid velocity variance floor")
    model = OnlineWorldModel.from_config(config, device="cpu")
    module = model.observation_modules["rgbd"]
    rgbd_config = getattr(module, "config", None)
    if rgbd_config is None or getattr(rgbd_config, "observation_mode", None) != "set":
        raise ValueError("capability factors require the RGB-D set observer")
    module.config = replace(
        rgbd_config,
        temporal_velocity_variance_floor=max(
            float(rgbd_config.temporal_velocity_variance_floor),
            float(variance_floor),
        ),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "completed_updates": int(payload.get("completed_updates", 0)),
        "selection": _jsonable(payload.get("selection", {})),
        "construction_seed": seed,
        "velocity_variance_floor": float(variance_floor),
    }


def factor_physical_rows(
    factor: CapabilityFactor,
) -> tuple[tuple[CapabilityManifestRow, PhysicalManifestRow], ...]:
    """Bind one capability factor to all 22 established physical cells."""

    if factor not in SUPPORTED_CAPABILITY_FACTORS:
        raise ValueError(f"unsupported executable capability factor: {factor}")
    manifest_split = "compositional_holdout" if factor == COMPOSITIONAL_FACTOR else "development"
    capability_rows = tuple(
        row
        for row in capability_manifest(manifest_split)
        if row.kind == "physical" and row.factor == factor
    )
    templates = physical_manifest("development")[: len(PHYSICAL_CELLS)]
    if len(capability_rows) != len(PHYSICAL_CELLS) or len(templates) != len(PHYSICAL_CELLS):
        raise RuntimeError("capability factor does not cover every physical cell")
    output: list[tuple[CapabilityManifestRow, PhysicalManifestRow]] = []
    for capability_row, template in zip(capability_rows, templates, strict=True):
        if capability_row.object_count != template.object_count or (
            capability_row.contact,
            capability_row.dynamic_membership,
        ) != (template.contact, template.dynamic_membership):
            raise RuntimeError("capability and physical cell orders differ")
        if factor in {"known_actions", COMPOSITIONAL_FACTOR} and (
            capability_row.controls.known_action_enabled
        ):
            template = replace(
                template,
                known_action=True,
                action_target_rank=capability_row.controls.impulse_target_rank,
                action_time_stratum=_CAPABILITY_ACTION_STRATUM[
                    capability_row.controls.impulse_phase
                ],
            )
        output.append(
            (
                capability_row,
                replace(
                    template,
                    ordinal=capability_row.ordinal,
                    seed=capability_row.seed,
                ),
            )
        )
    return tuple(output)


def _sensor_observation(
    rgb: Tensor,
    depth: Tensor,
    controls: FactorControls,
    *,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed ^ 0x5E45_0B5E)
    rgb_noise = torch.randn(rgb.shape, generator=generator, dtype=rgb.dtype)
    depth_noise = torch.randn(depth.shape, generator=generator, dtype=depth.dtype)
    dropout = (
        torch.rand(
            (rgb.shape[0], 1, *rgb.shape[-2:]),
            generator=generator,
            dtype=rgb.dtype,
        )
        < controls.pixel_dropout_probability
    )
    noisy_rgb = (rgb * controls.exposure_scale + controls.rgb_noise_std * rgb_noise).clamp(0.0, 1.0)
    noisy_rgb = torch.where(dropout, torch.zeros_like(noisy_rgb), noisy_rgb)
    valid_depth = depth > 0.0
    noisy_depth = torch.where(
        valid_depth,
        (depth + controls.depth_noise_std_m * depth_noise).clamp_min(1.0e-4),
        depth,
    )
    noisy_depth = torch.where(dropout, torch.zeros_like(noisy_depth), noisy_depth)
    return noisy_rgb.contiguous(), noisy_depth.contiguous()


def _partial_visibility_observation(
    episode: Mapping[str, Any],
    controls: FactorControls,
    *,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    rgb = episode["rgb"].clone()
    depth = episode["depth"].clone()
    labels = episode["labels"]
    if not isinstance(labels, Mapping) or not isinstance(labels.get("segmentation_mask"), Tensor):
        raise TypeError("partial visibility requires private visible-mask truth")
    visible_mask = labels["segmentation_mask"]
    object_count = int(episode["num_objects"])
    target_slot = seed % object_count
    end = 15 - controls.recovery_observation_frames
    start = end - controls.occlusion_frames
    if not 0 <= start < end < DYNAMIC_SET_FRAMES:
        raise ValueError("partial-visibility window lies outside the episode")
    removed_pixels = 0
    for frame_index in range(start, end):
        mask = visible_mask[frame_index, target_slot]
        coordinates = torch.nonzero(mask, as_tuple=False)
        if coordinates.shape[0] < 2:
            raise ValueError("partial-visibility target lacks enough visible pixels")
        split_x = coordinates[:, 1].to(torch.float32).median()
        side = (torch.arange(DYNAMIC_SET_IMAGE_SIZE[1]).view(1, -1) >= split_x).expand_as(mask)
        if (seed + frame_index) % 2:
            side = ~side
        removed = mask & side
        if not bool(removed.any()) or bool(torch.equal(removed, mask)):
            raise ValueError("partial-visibility mask must remove a strict subset")
        row_background = rgb[frame_index].median(dim=-1).values.unsqueeze(-1)
        rgb[frame_index] = torch.where(removed.unsqueeze(0), row_background, rgb[frame_index])
        depth[frame_index, 0] = torch.where(
            removed,
            torch.zeros_like(depth[frame_index, 0]),
            depth[frame_index, 0],
        )
        removed_pixels += int(removed.sum())
    return (
        rgb.contiguous(),
        depth.contiguous(),
        {
            "target_slot": target_slot,
            "start_frame": start,
            "end_frame_exclusive": end,
            "recovery_observation_frames": controls.recovery_observation_frames,
            "removed_pixels": removed_pixels,
        },
    )


def _sphere_state_at(episode: Mapping[str, Any], frame_index: int) -> SphereState:
    objects = episode["objects"]
    if not isinstance(objects, Mapping):
        raise TypeError("camera motion requires private object truth")
    active = objects["active"][frame_index]
    return SphereState(
        object_id=objects["id"][frame_index].clone(),
        active=active.clone(),
        position=objects["position"][frame_index].clone(),
        velocity=objects["velocity"][frame_index].clone(),
        radius=objects["radius"][frame_index].clone(),
        mass=objects["mass"][frame_index].clone(),
        restitution=objects["restitution"][frame_index].clone(),
        drag=objects["drag"][frame_index].clone(),
        friction=objects["friction"][frame_index].clone(),
        albedo=objects["albedo"][frame_index].clone(),
        orientation=objects["orientation"][frame_index].clone(),
        angular_velocity=objects["angular_velocity"][frame_index].clone(),
        sleeping=objects["sleeping"][frame_index].clone(),
        sleep_counter=torch.zeros_like(active, dtype=torch.int64),
    )


def _camera_velocities(
    world_from_camera: Tensor,
    timestamps: Tensor,
) -> tuple[Tensor, Tensor]:
    position = world_from_camera[:, :3, 3]
    linear = torch.zeros_like(position)
    angular = torch.zeros_like(position)
    dt = (timestamps[1:] - timestamps[:-1]).clamp_min(1.0e-8)
    linear[1:] = (position[1:] - position[:-1]) / dt[:, None]
    linear[0] = linear[1]
    rotation = world_from_camera[:, :3, :3]
    delta = rotation[1:] @ rotation[:-1].transpose(-1, -2)
    skew = 0.5 * (delta - delta.transpose(-1, -2))
    rotation_vector = torch.stack(
        (skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]),
        dim=-1,
    )
    angular[1:] = rotation_vector / dt[:, None]
    angular[0] = angular[1]
    return linear, angular


def _camera_motion_observation(
    episode: Mapping[str, Any],
    controls: FactorControls,
    *,
    seed: int,
) -> tuple[
    Tensor,
    Tensor,
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Any],
]:
    camera = episode["camera"]
    objects = episode["objects"]
    timestamps = episode["timestamps"]
    if not isinstance(camera, Mapping) or not isinstance(objects, Mapping):
        raise TypeError("camera motion requires private camera and object truth")
    initial_position = camera["position"][0]
    initial_target = camera["target"][0]
    mode = {
        "static": "fixed",
        "orbital": "orbit",
        "translating": "linear",
    }[controls.camera_motion]
    trajectory = CameraTrajectory(
        CameraTrajectoryConfig(
            image_size=DYNAMIC_SET_IMAGE_SIZE,
            mode=mode,
            vertical_fov_degrees=DYNAMIC_SET_VERTICAL_FOV_DEGREES,
            base_position=tuple(float(value) for value in initial_position),
            target=tuple(float(value) for value in initial_target),
            orbit_speed=0.55,
            orbit_amplitude=0.08,
            translation_amplitude=0.25,
        ),
        seed=seed,
    )
    rgb_frames: list[Tensor] = []
    depth_frames: list[Tensor] = []
    labels: list[dict[str, Tensor]] = []
    world_from_camera: list[Tensor] = []
    camera_from_world: list[Tensor] = []
    intrinsics: list[Tensor] = []
    positions: list[Tensor] = []
    targets: list[Tensor] = []
    for frame_index, timestamp_tensor in enumerate(timestamps):
        state = _sphere_state_at(episode, frame_index)
        frame_camera = trajectory.at(float(timestamp_tensor))
        rendered = render_spheres(
            state,
            frame_camera,
            DYNAMIC_SET_IMAGE_SIZE,
            edge_softness_pixels=1.0,
            noise_std=0.0,
        )
        active_visibility = rendered.visible_fraction[state.active]
        if not bool(rendered.projected_valid[state.active].all()) or not bool(
            active_visibility.eq(1.0).all()
        ):
            raise ValueError("camera trajectory violates full-visibility preflight")
        frame_labels = make_perception_labels(state, rendered, DYNAMIC_SET_IMAGE_SIZE)
        validate_perception_labels(
            frame_labels,
            max_objects=DYNAMIC_SET_MAX_OBJECTS,
            image_size=DYNAMIC_SET_IMAGE_SIZE,
        )
        rgb_frames.append(rendered.rgb)
        depth_frames.append(rendered.depth_buffer.unsqueeze(0))
        labels.append(frame_labels)
        world_from_camera.append(frame_camera.world_from_camera)
        camera_from_world.append(frame_camera.camera_from_world)
        intrinsics.append(frame_camera.intrinsics)
        positions.append(frame_camera.position)
        targets.append(frame_camera.target)
    stacked_world_from = torch.stack(world_from_camera)
    linear_velocity, angular_velocity = _camera_velocities(stacked_world_from, timestamps)
    stacked_labels = {name: torch.stack([frame[name] for frame in labels]) for name in labels[0]}
    updated_objects = {
        name: value.clone() if isinstance(value, Tensor) else value
        for name, value in objects.items()
    }
    for name in (
        "projected_center",
        "projected_center_pixels",
        "apparent_radius",
        "apparent_radius_normalized",
        "inverse_depth",
        "camera_depth",
        "projected_valid",
    ):
        updated_objects[name] = stacked_labels[name]
    updated_objects["visible_fraction"] = stacked_labels["visible_fraction"]
    updated_camera = {
        "world_from_camera": stacked_world_from,
        "camera_from_world": torch.stack(camera_from_world),
        "intrinsics": torch.stack(intrinsics),
        "position": torch.stack(positions),
        "target": torch.stack(targets),
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
        "calibrated": torch.ones(DYNAMIC_SET_FRAMES, dtype=torch.bool),
    }
    details = {
        "trajectory": controls.camera_motion,
        "translation_path_m": float(
            torch.linalg.vector_norm(updated_camera["position"][-1] - updated_camera["position"][0])
        ),
        "rotation_path_radians": float(
            torch.linalg.vector_norm(updated_camera["angular_velocity"], dim=-1).mean()
            * float(timestamps[-1] - timestamps[0])
        ),
        "all_active_fully_visible": True,
    }
    return (
        torch.stack(rgb_frames).to(torch.float32),
        torch.stack(depth_frames).to(torch.float32),
        updated_camera,
        updated_objects,
        stacked_labels,
        details,
    )


def _physical_parameters(controls: FactorControls) -> DynamicSetPhysicalParameters:
    return DynamicSetPhysicalParameters(
        radius=controls.radius_m,
        mass=controls.mass_kg,
        drag=controls.drag_per_second,
        restitution=controls.restitution,
        friction=controls.friction,
    )


def _known_action(controls: FactorControls) -> DynamicSetKnownAction | None:
    if not controls.known_action_enabled:
        return None
    impulse = tuple(
        controls.impulse_magnitude * value for value in controls.impulse_direction_world
    )
    return DynamicSetKnownAction(
        frame_index=_CAPABILITY_ACTION_FRAME[controls.impulse_phase],
        target_rank=controls.impulse_target_rank,
        impulse_world=impulse,
    )


def materialize_capability_physical_episode(
    capability_row: CapabilityManifestRow,
    physical_row: PhysicalManifestRow,
    *,
    apply_factor: bool,
) -> DynamicSetMaterialization:
    """Materialize one nominal or factor-controlled public episode."""

    capability_row.validate()
    if capability_row.kind != "physical" or capability_row.factor not in (
        SUPPORTED_CAPABILITY_FACTORS
    ):
        raise ValueError("row is not an executable capability-factor physical row")
    if physical_row.seed != capability_row.seed:
        raise ValueError("physical and capability rows must share their deterministic seed")
    controls = capability_row.controls
    if apply_factor and capability_row.factor == COMPOSITIONAL_FACTOR:
        base = materialize_dynamic_set_episode_with_controls(
            physical_row,
            parameters=_physical_parameters(controls),
            action=_known_action(controls),
        )
    elif apply_factor and capability_row.factor == "physical_parameters":
        base = materialize_dynamic_set_episode_with_parameters(
            physical_row,
            _physical_parameters(controls),
        )
    elif (
        apply_factor
        and capability_row.factor == "known_actions"
        and capability_row.controls.known_action_enabled
    ):
        action = _known_action(controls)
        assert action is not None
        base = materialize_dynamic_set_episode_with_action(
            physical_row,
            action,
        )
    else:
        base = materialize_dynamic_set_episode(physical_row)
    if not apply_factor:
        return base
    details: dict[str, Any] = {}
    camera = base.episode["camera"]
    objects = base.episode["objects"]
    labels = base.episode["labels"]
    if capability_row.factor == "physical_parameters":
        rgb = base.episode["rgb"]
        depth = base.episode["depth"]
        details = {
            "radius_m": controls.radius_m,
            "mass_kg": controls.mass_kg,
            "drag_per_second": controls.drag_per_second,
            "restitution": controls.restitution,
            "friction": controls.friction,
        }
    elif capability_row.factor == "known_actions":
        rgb = base.episode["rgb"]
        depth = base.episode["depth"]
        details = {
            "enabled": controls.known_action_enabled,
            "target_rank": controls.impulse_target_rank,
            "phase": controls.impulse_phase,
            "magnitude": controls.impulse_magnitude,
            "direction_world": list(controls.impulse_direction_world),
        }
    elif capability_row.factor == "sensor_noise":
        rgb, depth = _sensor_observation(
            base.episode["rgb"],
            base.episode["depth"],
            controls,
            seed=capability_row.seed,
        )
    elif capability_row.factor == "camera_motion":
        rgb, depth, camera, objects, labels, details = _camera_motion_observation(
            base.episode,
            controls,
            seed=capability_row.seed,
        )
    elif capability_row.factor == "partial_visibility":
        rgb, depth, details = _partial_visibility_observation(
            base.episode,
            controls,
            seed=capability_row.seed,
        )
    elif capability_row.factor == COMPOSITIONAL_FACTOR:
        rgb, depth, camera, objects, labels, camera_details = _camera_motion_observation(
            base.episode,
            controls,
            seed=capability_row.seed,
        )
        camera_episode = {
            **base.episode,
            "rgb": rgb,
            "depth": depth,
            "camera": camera,
            "objects": objects,
            "labels": labels,
        }
        rgb, depth, visibility_details = _partial_visibility_observation(
            camera_episode,
            controls,
            seed=capability_row.seed,
        )
        rgb, depth = _sensor_observation(rgb, depth, controls, seed=capability_row.seed)
        composed_action = _known_action(controls)
        details = {
            "physical_parameters": asdict(_physical_parameters(controls)),
            "known_action": None if composed_action is None else asdict(composed_action),
            "camera": camera_details,
            "partial_visibility": visibility_details,
            "sensor_noise": {
                "rgb_noise_std": controls.rgb_noise_std,
                "depth_noise_std_m": controls.depth_noise_std_m,
                "pixel_dropout_probability": controls.pixel_dropout_probability,
                "exposure_scale": controls.exposure_scale,
            },
        }
    else:  # pragma: no cover - guarded above and retained for exhaustiveness.
        raise RuntimeError("unreachable capability factor")
    metadata = {
        **base.episode["metadata"],
        "capability_factor": capability_row.factor,
        "capability_factor_seed": capability_row.seed,
        "capability_controls": asdict(controls),
        "capability_observation_details": details,
    }
    episode = {
        **base.episode,
        "rgb": rgb,
        "depth": depth,
        "camera": camera,
        "objects": objects,
        "labels": labels,
        "metadata": metadata,
    }
    validate_episode(episode)
    return replace(base, episode=episode)


def _total_accumulator(result: DynamicSetEvaluationResult) -> DynamicSetCellAccumulator:
    total = DynamicSetCellAccumulator()
    for value in result.evidence_by_cell.values():
        total.merge(value)
    return total


def _supported_physical_score(result: DynamicSetEvaluationResult) -> tuple[float, float]:
    components = result.score.components
    weight = math.fsum(SELECTION_SCORE_WEIGHTS[name] for name in components)
    if weight <= 0.0:
        raise ValueError("factor result has no supported physical score")
    value = math.fsum(SELECTION_SCORE_WEIGHTS[name] * score for name, score in components.items())
    return value / weight, weight


def _lifecycle_f1(total: DynamicSetCellAccumulator) -> float | None:
    combined = BinaryCounts(
        true_positive=total.birth.true_positive + total.removal.true_positive,
        false_positive=total.birth.false_positive + total.removal.false_positive,
        false_negative=total.birth.false_negative + total.removal.false_negative,
    )
    return combined.f1()


def _factor_metrics(result: DynamicSetEvaluationResult) -> dict[str, Any]:
    total = _total_accumulator(result)
    score, score_weight = _supported_physical_score(result)
    metrics = {
        "proposal_f1": total.proposal.f1(),
        "identity_accuracy": total.persistent_identity.value(),
        "lifecycle_f1": _lifecycle_f1(total),
        "current_position_rmse_m": total.current_position.rmse(),
        "two_second_position_rmse_m": total.horizon_position[2.0].rmse(),
        "collision_f1": total.collision.f1(),
        "uncertainty_90_coverage": total.uncertainty_90.value(),
        "physical_score": score,
        "supported_score_weight": score_weight,
        "episode_count": result.episode_count,
    }
    if any(value is None for value in metrics.values()):
        missing = sorted(name for name, value in metrics.items() if value is None)
        raise ValueError(f"factor evaluation lacks aggregate support for: {', '.join(missing)}")
    return metrics


def _factor_gate_failures(
    factor: CapabilityFactor,
    metrics: Mapping[str, Any],
) -> tuple[str, ...]:
    failures: list[str] = []
    if factor == COMPOSITIONAL_FACTOR:
        if float(metrics["proposal_f1"]) < 0.90:
            failures.append("proposal_f1")
        if float(metrics["identity_accuracy"]) < 0.95:
            failures.append("identity_accuracy")
        if float(metrics["two_second_position_rmse_m"]) > 0.150:
            failures.append("two_second_position_rmse_m")
        return tuple(failures)
    for name, limit in (
        ("proposal_f1", 0.95),
        ("identity_accuracy", 0.98),
        ("lifecycle_f1", 0.95),
        ("collision_f1", 0.90),
    ):
        if float(metrics[name]) < limit:
            failures.append(name)
    for name, limit in (
        ("current_position_rmse_m", 0.020),
        ("two_second_position_rmse_m", 0.120),
    ):
        if float(metrics[name]) > limit:
            failures.append(name)
    coverage = float(metrics["uncertainty_90_coverage"])
    if not 0.82 <= coverage <= 0.97:
        failures.append("uncertainty_90_coverage")
    return tuple(failures)


def _cell_name(cell: PhysicalCell) -> str:
    return (
        f"N{cell.object_count}/contact={int(cell.contact)}/dynamic={int(cell.dynamic_membership)}"
    )


def _cell_payload(result: DynamicSetEvaluationResult) -> dict[str, Any]:
    return {
        _cell_name(cell): _jsonable(metrics)
        for cell, metrics in sorted(
            result.by_cell.items(),
            key=lambda item: (
                item[0].object_count,
                item[0].contact,
                item[0].dynamic_membership,
            ),
        )
    }


def _horizon_curve(result: DynamicSetEvaluationResult) -> dict[str, float]:
    total = _total_accumulator(result)
    return {
        str(horizon): float(value)
        for horizon, evidence in total.horizon_position.items()
        if (value := evidence.rmse()) is not None
    }


def _coverage_range(result: DynamicSetEvaluationResult) -> list[float]:
    values = [
        float(metrics.uncertainty_90_coverage.value)
        for metrics in result.by_cell.values()
        if metrics.uncertainty_90_coverage.support > 0
    ]
    return [] if not values else [min(values), max(values)]


def _qualitative_rows(result: DynamicSetEvaluationResult) -> dict[str, str]:
    values: list[tuple[str, float]] = []
    for record in result.per_example_score_evidence:
        squared, support = record.additive.current_position
        value = math.sqrt(squared / support) if support else math.inf
        values.append((f"development:{record.ordinal}/seed={record.seed}", value))
    ordered = sorted(values, key=lambda item: item[1])
    if not ordered:
        return {
            "best_episode": "unavailable",
            "worst_episode": "unavailable",
            "representative_episode": "unavailable",
        }
    return {
        "best_episode": ordered[0][0],
        "worst_episode": ordered[-1][0],
        "representative_episode": ordered[len(ordered) // 2][0],
    }


def _make_summary(
    *,
    run_id: str,
    factor: CapabilityFactor,
    factor_result: DynamicSetEvaluationResult,
    clean_result: DynamicSetEvaluationResult,
    factor_metrics: Mapping[str, Any],
    clean_metrics: Mapping[str, Any],
    gate_failures: Sequence[str],
    provenance: Mapping[str, Any],
    scalability: Mapping[str, Any],
    artifact_bytes: int,
    archive_bytes: int,
    created_at_utc: str,
) -> CapabilityRunSummary:
    factor_entries = {
        name: {"status": "unmeasured", "reason": "factor run not executed"}
        for name in ALL_CAPABILITY_FACTORS
    }
    factor_entries["nominal_structured"] = {
        "status": "measured",
        "score": clean_metrics["physical_score"],
        **dict(clean_metrics),
    }
    factor_entries[factor] = {
        "status": "passed" if not gate_failures else "failed",
        "score": factor_metrics["physical_score"],
        "gate_failures": list(gate_failures),
        **dict(factor_metrics),
    }
    nominal_restoration = attribute_failure(
        {
            "observed": float(factor_metrics["physical_score"]),
            "clean_observation": float(clean_metrics["physical_score"]),
            "truth_association": float(factor_metrics["physical_score"]),
            "truth_parameters": float(factor_metrics["physical_score"]),
            "truth_state_dynamics": float(factor_metrics["physical_score"]),
        }
    )
    # This paired screen toggles exactly one controlled capability family. It
    # localises the family, not the component within that family; finer truth
    # ablations remain explicit follow-up evidence rather than being implied.
    family_owner = "none" if not gate_failures else factor
    resources = factor_result.resources
    unsupported = [f"{name} capability" for name in CAPABILITY_FACTORS if name != factor]
    if factor != COMPOSITIONAL_FACTOR:
        unsupported.append("compositional holdout capability")
    unsupported.append("factor-conditioned planning")
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=created_at_utc,
        lifecycle_status="completed",
        outcome="factor_passed" if not gate_failures else "factor_failed",
        source_format=CAPABILITY_FACTOR_REPORT_SCHEMA,
        configuration={
            "factor": factor,
            "episode_count": factor_result.episode_count,
            "image_size": list(DYNAMIC_SET_IMAGE_SIZE),
            "frames_per_episode": DYNAMIC_SET_FRAMES,
        },
        provenance=dict(provenance),
        scores={
            "candidate": {
                "value": factor_metrics["physical_score"],
                "supported_weight": factor_metrics["supported_score_weight"],
            },
            "incumbent": {
                "value": clean_metrics["physical_score"],
                "supported_weight": clean_metrics["supported_score_weight"],
            },
            "selected": "calibrated_structured_incumbent",
        },
        factor_metrics=factor_entries,
        cell_metrics=_cell_payload(factor_result),
        horizon_curves={"candidate_position_rmse_m": _horizon_curve(factor_result)},
        uncertainty={"coverage_90_range": _coverage_range(factor_result)},
        planning={"status": "unmeasured", "reason": "factor planning run not executed"},
        resources={
            "perception_latency_seconds": resources.perception_latency_seconds,
            "six_horizon_rollout_seconds": resources.six_horizon_rollout_seconds,
            "learned_weight_bytes": resources.learned_weight_bytes,
            "persistent_tensor_bytes": resources.persistent_tensor_bytes,
            "process_rss_bytes": resources.process_rss_bytes,
            "scalability": dict(scalability),
        },
        artifacts={"run_bytes": artifact_bytes, "archive_bytes": archive_bytes},
        selection={
            "selected": "calibrated_structured_incumbent",
            "promotion_evaluated": False,
            "gate_failures": list(gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": (
                "no absolute factor failure" if not gate_failures else ", ".join(gate_failures)
            ),
            "ablation_owner": family_owner,
            "same_seed_nominal_restoration": nominal_restoration,
            "subsystem_ablation_status": (
                "not required after absolute pass" if not gate_failures else "required"
            ),
        },
        qualitative={
            **_qualitative_rows(factor_result),
            "preview_image": "not retained",
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=tuple(unsupported),
        scope_limitations=(
            "hidden actions",
            "unknown camera calibration",
            "new modalities",
            "deformable or articulated bodies",
            "long-term scene memory",
        ),
    ).validate()


def _make_runtime_failure_summary(
    *,
    run_id: str,
    factor: CapabilityFactor,
    error: DynamicSetEvaluationError,
    factor_rows: Sequence[CapabilityManifestRow],
    current_index: int,
    completed_count: int,
    provenance: Mapping[str, Any],
    archive_bytes: int,
    created_at_utc: str,
) -> CapabilityRunSummary:
    action_support = sum(row.controls.known_action_enabled for row in factor_rows)
    target_failure = "target could not be resolved" in str(error)
    resolution_upper = (
        (action_support - 1) / action_support if target_failure and action_support else None
    )
    factor_entries = {
        name: {"status": "unmeasured", "reason": "factor run not executed"}
        for name in ALL_CAPABILITY_FACTORS
    }
    factor_entries[factor] = {
        "status": "failed",
        "score": None,
        "runtime_error": str(error),
        "completed_episode_count": completed_count,
        "failed_episode_index": current_index,
        "observable_target_handle_resolution_upper_bound": resolution_upper,
        "observable_target_handle_support": action_support,
    }
    failed_row = (
        None if not 1 <= current_index <= len(factor_rows) else factor_rows[current_index - 1]
    )
    unsupported = [f"{name} capability" for name in CAPABILITY_FACTORS if name != factor]
    if factor != COMPOSITIONAL_FACTOR:
        unsupported.append("compositional holdout capability")
    unsupported.extend(("factor-conditioned planning", "physical metrics after runtime refusal"))
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=created_at_utc,
        lifecycle_status="failed",
        outcome="factor_failed",
        source_format=CAPABILITY_FACTOR_REPORT_SCHEMA,
        configuration={
            "factor": factor,
            "episode_count": len(factor_rows),
            "completed_episode_count": completed_count,
            "failed_episode_index": current_index,
        },
        provenance=dict(provenance),
        scores={"candidate": {}, "incumbent": {}, "selected": "calibrated_structured_incumbent"},
        factor_metrics=factor_entries,
        cell_metrics={},
        horizon_curves={},
        uncertainty={},
        planning={"status": "unmeasured", "reason": "physical runtime refused input"},
        resources={},
        artifacts={"run_bytes": 0, "archive_bytes": archive_bytes},
        selection={
            "selected": "calibrated_structured_incumbent",
            "promotion_evaluated": False,
            "gate_failures": ["observable_target_handle_resolution"],
        },
        failure_attribution={
            "primary_bottleneck": "observable target-handle resolution"
            if target_failure
            else "public runtime evaluation",
            "ablation_owner": "appearance_target_resolution" if target_failure else factor,
            "runtime_error": str(error),
            "failed_manifest_row": None if failed_row is None else asdict(failed_row),
        },
        qualitative={
            "best_episode": "unavailable after fail-closed stop",
            "worst_episode": (
                "unavailable"
                if failed_row is None
                else f"{failed_row.split}:{failed_row.ordinal}/seed={failed_row.seed}"
            ),
            "representative_episode": "unavailable after fail-closed stop",
            "preview_image": "not retained",
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=tuple(unsupported),
        scope_limitations=(
            "hidden actions",
            "unknown camera calibration",
            "new modalities",
            "deformable or articulated bodies",
            "long-term scene memory",
        ),
    ).validate()


def run_incumbent_factor_evaluation(
    *,
    factor: CapabilityFactor,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    run_directory: str | Path,
    seed: int = 0,
    threads: int = 1,
    progress: Any = print,
) -> dict[str, Any]:
    """Evaluate one controlled factor and its nominal ablation without training."""

    if factor not in SUPPORTED_CAPABILITY_FACTORS:
        raise ValueError(f"factor must be one of {SUPPORTED_CAPABILITY_FACTORS}")
    _seed_process(seed, threads)
    output = Path(run_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def update(message: str) -> None:
        if progress is not None:
            progress(message)

    config = load_config(model_config_path)
    model, checkpoint = _model_from_workbench_checkpoint(config, checkpoint_path)
    rows = factor_physical_rows(factor)
    factor_rows = tuple(item[0] for item in rows)
    created = datetime.now(timezone.utc).isoformat()
    complete_manifest = capability_manifest(
        "compositional_holdout" if factor == COMPOSITIONAL_FACTOR else "development"
    )
    provenance = {
        **checkpoint,
        "public_development_only": True,
        "planning_used_as_training_loss": False,
        "truth_runtime_input_count": 0,
        "capability_manifest_sha256": manifest_sha256(complete_manifest),
        "factor_rows_sha256": manifest_sha256(factor_rows),
        "factor_controls_generator_only": True,
        "same_seed_nominal_ablation": True,
    }
    update(f"streaming {len(rows)} {factor} episodes through the incumbent")
    stream_state = {"current_index": 0, "completed_count": 0}

    def stream(*, apply_factor: bool):
        for index, (capability_row, physical_row) in enumerate(rows, start=1):
            stream_state["current_index"] = index
            yield materialize_capability_physical_episode(
                capability_row,
                physical_row,
                apply_factor=apply_factor,
            )
            stream_state["completed_count"] = index
            update(
                f"{factor if apply_factor else 'clean ablation'}: materialized {index}/{len(rows)}"
            )

    factor_started = time.perf_counter()
    try:
        factor_result = evaluate_dynamic_set_materializations(
            model,
            stream(apply_factor=True),
            # One row per cell is the intended factor screen. It covers every
            # count/contact/dynamic cell, but only the first lifecycle schedule;
            # do not mislabel absent removal schedules as complete score support.
            require_all_cells=False,
        )
    except DynamicSetEvaluationError as error:
        archive_root = output.parent.parent / ".archive"
        archive_bytes = inventory_runs(output.parent, archive_root=archive_root)["archive_bytes"]
        report = {
            "schema": CAPABILITY_FACTOR_REPORT_SCHEMA,
            "created_at_utc": created,
            "factor": factor,
            "status": "failed",
            "gate_failures": ["observable_target_handle_resolution"],
            "runtime_error": str(error),
            "completed_episode_count": stream_state["completed_count"],
            "failed_episode_index": stream_state["current_index"],
            "provenance": provenance,
            "timing_seconds": {"total": time.perf_counter() - started},
            "artifact_policy": {
                "generated_episodes_retained": False,
                "raw_frame_directories_retained": False,
                "checkpoint_copied": False,
            },
        }
        atomic_write_text(
            output / "factor_report.json",
            json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        summary = _make_runtime_failure_summary(
            run_id=output.name,
            factor=factor,
            error=error,
            factor_rows=factor_rows,
            current_index=stream_state["current_index"],
            completed_count=stream_state["completed_count"],
            provenance=provenance,
            archive_bytes=archive_bytes,
            created_at_utc=created,
        )
        write_capability_summary(summary, output / "capability_summary.json")
        write_run_report(summary, output)
        artifact_bytes = sum(
            path.stat().st_size
            for path in output.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        summary = replace(
            summary,
            artifacts={"run_bytes": artifact_bytes, "archive_bytes": archive_bytes},
        )
        write_capability_summary(summary, output / "capability_summary.json")
        write_run_report(summary, output)
        write_run_manifest(
            output,
            role="candidate",
            status="failed",
            artifacts={
                "capability_summary.json": "summary",
                "factor_report.json": "summary",
                "report.html": "report",
            },
        )
        cleanup = enforce_run_budget(output.parent, archive_root=archive_root)
        build_progress_dashboard(output.parent, archive_root=archive_root)
        update(f"stopped {factor} fail-closed: {error}")
        return {**report, "cleanup": cleanup.to_dict(), "run_directory": str(output)}
    if set(factor_result.by_cell) != set(PHYSICAL_CELLS):
        raise RuntimeError("factor evaluation did not cover all 22 physical cells")
    factor_seconds = time.perf_counter() - factor_started
    clean_started = time.perf_counter()
    clean_result = evaluate_dynamic_set_materializations(
        model,
        stream(apply_factor=False),
        require_all_cells=False,
    )
    if set(clean_result.by_cell) != set(PHYSICAL_CELLS):
        raise RuntimeError("clean ablation did not cover all 22 physical cells")
    clean_seconds = time.perf_counter() - clean_started
    factor_metrics = _factor_metrics(factor_result)
    clean_metrics = _factor_metrics(clean_result)
    if factor in {"known_actions", COMPOSITIONAL_FACTOR}:
        action_support = sum(row.controls.known_action_enabled for row in factor_rows)
        factor_metrics["observable_target_handle_resolution"] = 1.0
        factor_metrics["observable_target_handle_support"] = action_support
    gate_failures = _factor_gate_failures(factor, factor_metrics)
    update("probing dense and packed state-only N=8/12/16 rollouts")
    scalability = compare_dense_and_packed_scalability(
        model.dynamics,
        warmup_runs=1,
        measured_runs=3,
    )
    report: dict[str, Any] = {
        "schema": CAPABILITY_FACTOR_REPORT_SCHEMA,
        "created_at_utc": created,
        "factor": factor,
        "status": "passed" if not gate_failures else "failed",
        "gate_failures": list(gate_failures),
        "factor_metrics": factor_metrics,
        "clean_metrics": clean_metrics,
        "factor_cells": _cell_payload(factor_result),
        "clean_cells": _cell_payload(clean_result),
        "provenance": provenance,
        "resources": _jsonable(factor_result.resources),
        "scalability": scalability,
        "timing_seconds": {
            "factor_evaluation": factor_seconds,
            "clean_ablation": clean_seconds,
            "total": time.perf_counter() - started,
        },
        "artifact_policy": {
            "generated_episodes_retained": False,
            "raw_frame_directories_retained": False,
            "checkpoint_copied": False,
        },
    }
    atomic_write_text(
        output / "factor_report.json",
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_root = output.parent.parent / ".archive"
    archive_bytes = inventory_runs(output.parent, archive_root=archive_root)["archive_bytes"]
    artifact_bytes = (output / "factor_report.json").stat().st_size
    summary = _make_summary(
        run_id=output.name,
        factor=factor,
        factor_result=factor_result,
        clean_result=clean_result,
        factor_metrics=factor_metrics,
        clean_metrics=clean_metrics,
        gate_failures=gate_failures,
        provenance=provenance,
        scalability=scalability,
        artifact_bytes=artifact_bytes,
        archive_bytes=archive_bytes,
        created_at_utc=created,
    )
    write_capability_summary(summary, output / "capability_summary.json")
    write_run_report(summary, output)
    artifact_bytes = sum(
        path.stat().st_size for path in output.iterdir() if path.is_file() and not path.is_symlink()
    )
    summary = replace(
        summary, artifacts={"run_bytes": artifact_bytes, "archive_bytes": archive_bytes}
    )
    write_capability_summary(summary, output / "capability_summary.json")
    write_run_report(summary, output)
    write_run_manifest(
        output,
        role="candidate",
        status="completed",
        artifacts={
            "capability_summary.json": "summary",
            "factor_report.json": "summary",
            "report.html": "report",
        },
    )
    cleanup = enforce_run_budget(output.parent, archive_root=archive_root)
    build_progress_dashboard(output.parent, archive_root=archive_root)
    update(f"completed {factor} with {len(gate_failures)} gate failure(s)")
    return {**report, "cleanup": cleanup.to_dict(), "run_directory": str(output)}


def default_factor_run_directory(factor: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{timestamp}-capability-{factor.replace('_', '-')}"


__all__ = [
    "CAPABILITY_FACTOR_REPORT_SCHEMA",
    "SUPPORTED_CAPABILITY_FACTORS",
    "default_factor_run_directory",
    "factor_physical_rows",
    "materialize_capability_physical_episode",
    "run_incumbent_factor_evaluation",
]
