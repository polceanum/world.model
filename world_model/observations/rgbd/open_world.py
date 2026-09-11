"""Appearance-prototype-free rigid discovery and short-horizon tracking.

Objects are proposed from connected public depth support, grouped across
calibrated views using descriptors measured in the current observation, and
associated over time by predicted metric position, geometry, and the online
descriptor. No simulator instance map, object ID, primitive label, or
predeclared colour prototype is accepted by this module.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from world_model.belief import BeliefFactory, MotionMode, RigidPrimitive, WorldBelief
from world_model.observations.rgbd.rigid_geometry import (
    ObservableRigidGeometry,
    fit_rigid_geometry_from_rgbd,
)
from world_model.simulator.camera import backproject_pixels, camera_to_world


@dataclass(frozen=True)
class DiscoveredRigidObject:
    geometry: ObservableRigidGeometry
    appearance: Tensor
    support_views: int
    confidence: float


@dataclass(frozen=True)
class OpenWorldRigidFrame:
    timestamp: float
    objects: tuple[DiscoveredRigidObject, ...]


@dataclass(frozen=True)
class TrackedRigidObject:
    object_id: int
    geometry: ObservableRigidGeometry
    appearance: Tensor
    velocity: Tensor
    angular_velocity: Tensor
    observed: bool
    age_steps: int
    missed_steps: int


@dataclass
class _Track:
    object_id: int
    geometry: ObservableRigidGeometry
    appearance: Tensor
    velocity: Tensor
    angular_velocity: Tensor
    timestamp: float
    last_observed_geometry: ObservableRigidGeometry
    last_observed_timestamp: float
    age_steps: int = 1
    missed_steps: int = 0


def _component_descriptor(rgb: Tensor, mask: Tensor) -> Tensor:
    mean = rgb[mask].mean(dim=0)
    return mean / torch.linalg.vector_norm(mean).clamp_min(1.0e-8)


def _component_centroid(
    depth: Tensor,
    mask: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
) -> Tensor:
    y, x = torch.where(mask)
    pixels = torch.stack((x, y), dim=-1).to(depth.dtype)
    points = backproject_pixels(pixels, depth[mask], intrinsics)
    return camera_to_world(points, world_from_camera).mean(dim=0)


def _appearance_embedding(colour: Tensor, geometry: ObservableRigidGeometry) -> Tensor:
    extent = geometry.half_extents.sort().values
    extent = extent / torch.linalg.vector_norm(extent).clamp_min(1.0e-8)
    primitive = torch.nn.functional.one_hot(geometry.primitive, num_classes=2).to(colour.dtype)
    return torch.cat((colour, extent, primitive), dim=-1)


def discover_rigid_objects_from_rgbd(
    rgb: Tensor,
    depth: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    *,
    timestamp: float,
    minimum_component_pixels: int = 24,
    minimum_geometry_points: int = 64,
    appearance_gate: float = 0.95,
    centroid_gate_m: float = 1.25,
    chromatic_bins: int = 32,
) -> OpenWorldRigidFrame:
    """Discover an unordered rigid set from simultaneous calibrated views."""

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape [V,H,W,3]")
    views, height, width, _ = rgb.shape
    if depth.shape != (views, height, width):
        raise ValueError("depth must have shape [V,H,W]")
    if world_from_camera.shape != (views, 4, 4) or intrinsics.shape != (views, 3, 3):
        raise ValueError("calibration must have shapes [V,4,4] and [V,3,3]")
    if not rgb.is_floating_point() or depth.dtype != rgb.dtype:
        raise TypeError("RGB and depth must share a floating dtype")
    if any(
        value.device != rgb.device or value.dtype != rgb.dtype
        for value in (depth, world_from_camera, intrinsics)
    ):
        raise ValueError("all discovery tensors must share dtype and device")
    if rgb.device.type != "cpu":
        raise ValueError("connected-component discovery is CPU-first and requires CPU tensors")
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    if minimum_component_pixels < 4 or minimum_geometry_points < 16:
        raise ValueError("component and geometry support limits are too small")
    if chromatic_bins < 4 or chromatic_bins > 256:
        raise ValueError("chromatic_bins must lie in [4,256]")
    if not 0.0 < appearance_gate <= 1.0 or centroid_gate_m <= 0.0:
        raise ValueError("discovery gates are invalid")

    groups: list[dict[str, object]] = []
    structure = np.ones((3, 3), dtype=np.int8)
    for view in range(views):
        valid = torch.isfinite(depth[view]) & (depth[view] > 0.0)
        candidates: list[tuple[Tensor, Tensor, Tensor]] = []
        chromatic = rgb[view] / torch.linalg.vector_norm(rgb[view], dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        quantized = ((chromatic.clamp(0.0, 1.0) * (chromatic_bins - 1)).round()).to(torch.int64)
        code = (
            quantized[..., 0] * chromatic_bins * chromatic_bins
            + quantized[..., 1] * chromatic_bins
            + quantized[..., 2]
        )
        for colour_code in torch.unique(code[valid]).tolist():
            colour_region = valid & (code == int(colour_code))
            components, count = label(colour_region.detach().numpy(), structure=structure)
            for component_index in range(1, count + 1):
                mask = torch.from_numpy(components == component_index).to(device=rgb.device)
                if int(mask.sum()) < minimum_component_pixels:
                    continue
                candidates.append(
                    (
                        mask,
                        _component_descriptor(rgb[view], mask),
                        _component_centroid(
                            depth[view],
                            mask,
                            world_from_camera[view],
                            intrinsics[view],
                        ),
                    )
                )
        assigned: set[int] = set()
        for mask, descriptor, centroid in candidates:
            best_group = None
            best_cost = math.inf
            for group_index, group in enumerate(groups):
                group_views = group["views"]
                assert isinstance(group_views, list)
                if group_index in assigned or bool(group_views[view]):
                    continue
                group_descriptor = group["descriptor"]
                group_centroid = group["centroid"]
                assert isinstance(group_descriptor, Tensor)
                assert isinstance(group_centroid, Tensor)
                similarity = float(descriptor.dot(group_descriptor))
                distance = float(torch.linalg.vector_norm(centroid - group_centroid))
                if similarity < appearance_gate or distance > centroid_gate_m:
                    continue
                cost = (1.0 - similarity) + 0.05 * distance
                if cost < best_cost:
                    best_cost = cost
                    best_group = group_index
            if best_group is None:
                view_masks = [torch.zeros_like(valid) for _ in range(views)]
                view_masks[view] = mask
                groups.append(
                    {
                        "views": [index == view for index in range(views)],
                        "masks": view_masks,
                        "descriptor": descriptor,
                        "centroid": centroid,
                        "count": 1,
                    }
                )
                assigned.add(len(groups) - 1)
                continue
            group = groups[best_group]
            group_views = group["views"]
            group_masks = group["masks"]
            assert isinstance(group_views, list)
            assert isinstance(group_masks, list)
            group_views[view] = True
            group_masks[view] = mask
            old_count = int(group["count"])
            old_descriptor = group["descriptor"]
            old_centroid = group["centroid"]
            assert isinstance(old_descriptor, Tensor)
            assert isinstance(old_centroid, Tensor)
            averaged_descriptor = (old_descriptor * old_count + descriptor) / (old_count + 1)
            group["descriptor"] = averaged_descriptor / torch.linalg.vector_norm(
                averaged_descriptor
            ).clamp_min(1.0e-8)
            group["centroid"] = (old_centroid * old_count + centroid) / (old_count + 1)
            group["count"] = old_count + 1
            assigned.add(best_group)

    discovered: list[DiscoveredRigidObject] = []
    for group in groups:
        support_views = int(group["count"])
        if support_views < 2:
            continue
        group_masks = group["masks"]
        descriptor = group["descriptor"]
        assert isinstance(group_masks, list)
        assert isinstance(descriptor, Tensor)
        masks = torch.stack(group_masks).to(dtype=depth.dtype)
        geometry = fit_rigid_geometry_from_rgbd(
            depth,
            masks,
            world_from_camera,
            intrinsics,
            minimum_points=minimum_geometry_points,
        )
        if not bool(geometry.valid):
            continue
        discovered.append(
            DiscoveredRigidObject(
                geometry=geometry,
                appearance=_appearance_embedding(descriptor, geometry),
                support_views=support_views,
                confidence=min(1.0, support_views / max(views, 1)),
            )
        )
    discovered.sort(key=lambda item: tuple(float(value) for value in item.geometry.world_position))
    return OpenWorldRigidFrame(timestamp=float(timestamp), objects=tuple(discovered))


def _rotation_matrix(quaternion: Tensor) -> Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1.0e-8)
    x, y, z, w = quaternion.unbind()
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x.square() + y.square()),
        )
    ).reshape(3, 3)


def _matrix_to_quaternion(rotation: Tensor) -> Tensor:
    trace = torch.trace(rotation)
    w = 0.5 * torch.sqrt((1.0 + trace).clamp_min(0.0))
    x = 0.5 * torch.sqrt((1.0 + 2.0 * rotation[0, 0] - trace).clamp_min(0.0))
    y = 0.5 * torch.sqrt((1.0 + 2.0 * rotation[1, 1] - trace).clamp_min(0.0))
    z = 0.5 * torch.sqrt((1.0 + 2.0 * rotation[2, 2] - trace).clamp_min(0.0))
    quaternion = torch.stack(
        (
            torch.copysign(x, rotation[2, 1] - rotation[1, 2]),
            torch.copysign(y, rotation[0, 2] - rotation[2, 0]),
            torch.copysign(z, rotation[1, 0] - rotation[0, 1]),
            w,
        )
    )
    return quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1.0e-8)


def _proper_signed_permutations(reference: Tensor) -> tuple[Tensor, ...]:
    values: list[Tensor] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = reference.new_zeros(3, 3)
            for column, row in enumerate(permutation):
                matrix[row, column] = signs[column]
            if float(torch.linalg.det(matrix)) > 0.5:
                values.append(matrix)
    return tuple(values)


def _align_box_geometry(
    geometry: ObservableRigidGeometry,
    reference_orientation: Tensor,
) -> ObservableRigidGeometry:
    if int(geometry.primitive) != int(RigidPrimitive.BOX):
        return geometry
    measured = _rotation_matrix(geometry.orientation)
    reference = _rotation_matrix(reference_orientation)
    best_rotation = measured
    best_extent = geometry.half_extents
    best_score = math.inf
    identity = torch.eye(3, dtype=measured.dtype, device=measured.device)
    for permutation in _proper_signed_permutations(measured):
        candidate = measured @ permutation
        score = float(torch.linalg.matrix_norm(reference.transpose(-1, -2) @ candidate - identity))
        if score < best_score:
            best_score = score
            best_rotation = candidate
            best_extent = permutation.abs().transpose(-1, -2) @ geometry.half_extents
    return replace(
        geometry,
        orientation=_matrix_to_quaternion(best_rotation),
        half_extents=best_extent,
    )


def _angular_velocity(previous: Tensor, current: Tensor, dt: float) -> Tensor:
    relative = _rotation_matrix(previous).transpose(-1, -2) @ _rotation_matrix(current)
    vector = torch.stack(
        (
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        )
    )
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    scale = torch.where(
        angle > 1.0e-6,
        angle / (2.0 * torch.sin(angle).clamp_min(1.0e-8)),
        angle.new_tensor(0.5),
    )
    return vector * scale / dt


class OpenWorldRigidTracker:
    """Small-set persistent tracker with bounded short occlusion recovery."""

    def __init__(self, *, max_missed_steps: int = 2, association_gate: float = 2.0) -> None:
        if max_missed_steps < 0 or association_gate <= 0.0:
            raise ValueError("tracker gates are invalid")
        self.max_missed_steps = max_missed_steps
        self.association_gate = association_gate
        self._tracks: list[_Track] = []
        self._next_id = 0

    def update(self, frame: OpenWorldRigidFrame) -> tuple[TrackedRigidObject, ...]:
        if not math.isfinite(frame.timestamp):
            raise ValueError("frame timestamp must be finite")
        if self._tracks and frame.timestamp <= max(track.timestamp for track in self._tracks):
            raise ValueError("tracking timestamps must increase")
        detections = list(frame.objects)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        pairs: list[tuple[int, int]] = []
        if self._tracks and detections:
            costs = np.full((len(self._tracks), len(detections)), 1.0e6, dtype=np.float64)
            for track_index, track in enumerate(self._tracks):
                dt = frame.timestamp - track.timestamp
                predicted = track.geometry.world_position + track.velocity * dt
                for detection_index, detection in enumerate(detections):
                    position_cost = float(
                        torch.linalg.vector_norm(detection.geometry.world_position - predicted)
                    )
                    appearance_dim = min(track.appearance.numel(), detection.appearance.numel())
                    appearance_cost = 1.0 - float(
                        torch.nn.functional.cosine_similarity(
                            track.appearance[:appearance_dim],
                            detection.appearance[:appearance_dim],
                            dim=0,
                        )
                    )
                    primitive_cost = (
                        0.0
                        if int(track.geometry.primitive) == int(detection.geometry.primitive)
                        else 1.0
                    )
                    costs[track_index, detection_index] = (
                        2.5 * position_cost + appearance_cost + primitive_cost
                    )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] <= self.association_gate:
                    pairs.append((int(row), int(column)))
                    matched_tracks.add(int(row))
                    matched_detections.add(int(column))

        observed_ids: set[int] = set()
        for track_index, detection_index in pairs:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            dt = frame.timestamp - track.last_observed_timestamp
            geometry = _align_box_geometry(
                detection.geometry,
                track.last_observed_geometry.orientation,
            )
            velocity = (geometry.world_position - track.last_observed_geometry.world_position) / dt
            angular = (
                _angular_velocity(
                    track.last_observed_geometry.orientation,
                    geometry.orientation,
                    dt,
                )
                if int(geometry.primitive) == int(RigidPrimitive.BOX)
                else torch.zeros_like(velocity)
            )
            track.geometry = geometry
            track.appearance = torch.nn.functional.normalize(
                0.7 * track.appearance + 0.3 * detection.appearance,
                dim=0,
            )
            track.velocity = velocity
            track.angular_velocity = angular
            track.timestamp = frame.timestamp
            track.last_observed_geometry = geometry
            track.last_observed_timestamp = frame.timestamp
            track.age_steps += 1
            track.missed_steps = 0
            observed_ids.add(track.object_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            reference = detection.geometry.orientation.new_tensor([0.0, 0.0, 0.0, 1.0])
            geometry = _align_box_geometry(detection.geometry, reference)
            self._tracks.append(
                _Track(
                    object_id=self._next_id,
                    geometry=geometry,
                    appearance=detection.appearance.clone(),
                    velocity=torch.zeros_like(geometry.world_position),
                    angular_velocity=torch.zeros_like(geometry.world_position),
                    timestamp=frame.timestamp,
                    last_observed_geometry=geometry,
                    last_observed_timestamp=frame.timestamp,
                )
            )
            observed_ids.add(self._next_id)
            self._next_id += 1

        retained: list[_Track] = []
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks and track.object_id not in observed_ids:
                dt = frame.timestamp - track.timestamp
                track.geometry = replace(
                    track.geometry,
                    world_position=track.geometry.world_position + track.velocity * dt,
                )
                track.timestamp = frame.timestamp
                track.age_steps += 1
                track.missed_steps += 1
            if track.missed_steps <= self.max_missed_steps:
                retained.append(track)
        self._tracks = retained
        return tuple(
            TrackedRigidObject(
                object_id=track.object_id,
                geometry=track.geometry,
                appearance=track.appearance,
                velocity=track.velocity,
                angular_velocity=track.angular_velocity,
                observed=track.object_id in observed_ids,
                age_steps=track.age_steps,
                missed_steps=track.missed_steps,
            )
            for track in sorted(self._tracks, key=lambda item: item.object_id)
        )


def tracked_objects_to_belief(
    tracked: tuple[TrackedRigidObject, ...],
    *,
    timestamp: float,
    max_objects: int,
    initial_mass: float = 1.0,
    initial_restitution: float = 0.5,
    initial_drag: float = 0.05,
    initial_friction: float = 0.25,
) -> WorldBelief:
    """Create a public belief from tracker output and explicit neutral priors."""

    if len(tracked) > max_objects:
        raise ValueError("tracked set exceeds belief capacity")
    if not tracked:
        raise ValueError("at least one tracked object is required")
    dtype = tracked[0].geometry.world_position.dtype
    belief = BeliefFactory(
        max_objects=max_objects,
        geometry_dim=5,
        appearance_dim=8,
        residual_dynamics_dim=1,
        modal_count=0,
        modal_dim=1,
        parameter_memory_dim=8,
        global_code_dim=1,
        initial_mass=initial_mass,
        initial_restitution=initial_restitution,
        initial_drag=initial_drag,
        initial_friction=initial_friction,
    ).create(batch_size=1, dtype=dtype, gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., int(MotionMode.FREE)] = 4.0
    for slot, item in enumerate(tracked):
        objects.active[0, slot] = True
        objects.object_id[0, slot] = item.object_id
        objects.existence_logit[0, slot] = 8.0
        objects.position[0, slot] = item.geometry.world_position
        objects.velocity[0, slot] = item.velocity
        objects.orientation[0, slot] = item.geometry.orientation
        objects.angular_velocity[0, slot] = item.angular_velocity
        objects.geometry[0, slot] = item.geometry.encode(geometry_dim=5)
        objects.appearance[0, slot] = item.appearance
        objects.age_steps[0, slot] = item.age_steps
        objects.missed_steps[0, slot] = item.missed_steps
        objects.visibility_logit[0, slot] = 6.0 if item.observed else -2.0
        objects.fast_log_variance[0, slot] = -8.0 if item.observed else -3.0
        objects.slow_log_variance[0, slot] = 1.0
    return replace(
        belief,
        objects=objects,
        timestamp=belief.timestamp.new_tensor([timestamp]),
        next_object_id=torch.tensor(
            [max(item.object_id for item in tracked) + 1],
            dtype=torch.int64,
        ),
        active_modalities=("rgbd_open_world",),
    ).validate()


__all__ = [
    "DiscoveredRigidObject",
    "OpenWorldRigidFrame",
    "OpenWorldRigidTracker",
    "TrackedRigidObject",
    "discover_rigid_objects_from_rgbd",
    "tracked_objects_to_belief",
]
