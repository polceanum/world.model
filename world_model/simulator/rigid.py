"""Independent rigid-primitive state and exact CPU RGB-D renderer.

The renderer accepts only physical state and calibrated camera tensors.  Its
instance maps are private simulator labels; runtime observations expose only
the rendered RGB and depth arrays.  Sphere-only calls delegate to the frozen
sphere renderer byte-for-byte so the existing visual protocol remains an
exact oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.belief import RigidPrimitive
from world_model.simulator.camera import CameraFrame, project_camera_points, world_to_camera
from world_model.simulator.physics import SphereState
from world_model.simulator.renderer import RenderOutput, _background, render_spheres


def _rotation_matrix(quaternion: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    unit = torch.where(norm > 1.0e-8, quaternion / norm.clamp_min(1.0e-8), identity)
    x, y, z, w = unit.unbind(dim=-1)
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
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


@dataclass(frozen=True)
class RigidBodyState:
    """Padded sphere/box state for independent simulation and rendering."""

    object_id: Tensor
    active: Tensor
    position: Tensor
    velocity: Tensor
    primitive: Tensor
    radius: Tensor
    half_extents: Tensor
    mass: Tensor
    restitution: Tensor
    drag: Tensor
    friction: Tensor
    albedo: Tensor
    orientation: Tensor
    angular_velocity: Tensor
    sleeping: Tensor
    sleep_counter: Tensor

    @property
    def max_objects(self) -> int:
        return int(self.position.shape[0])

    @classmethod
    def from_spheres(cls, state: SphereState) -> RigidBodyState:
        """Lift legacy state without changing any sphere tensor."""

        state.validate()
        return cls(
            object_id=state.object_id,
            active=state.active,
            position=state.position,
            velocity=state.velocity,
            primitive=torch.zeros_like(state.object_id),
            radius=state.radius,
            half_extents=state.radius.expand(-1, 3),
            mass=state.mass,
            restitution=state.restitution,
            drag=state.drag,
            friction=state.friction,
            albedo=state.albedo,
            orientation=state.orientation,
            angular_velocity=state.angular_velocity,
            sleeping=state.sleeping,
            sleep_counter=state.sleep_counter,
        )

    def as_spheres(self) -> SphereState:
        """Return the exact legacy representation when every active body is a sphere."""

        if bool((self.active & (self.primitive != int(RigidPrimitive.SPHERE))).any()):
            raise ValueError("box state cannot be represented as SphereState")
        return SphereState(
            object_id=self.object_id,
            active=self.active,
            position=self.position,
            velocity=self.velocity,
            radius=self.radius,
            mass=self.mass,
            restitution=self.restitution,
            drag=self.drag,
            friction=self.friction,
            albedo=self.albedo,
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
            sleeping=self.sleeping,
            sleep_counter=self.sleep_counter,
        )

    def validate(self) -> None:
        count = self.max_objects
        expected = {
            "object_id": (count,),
            "active": (count,),
            "position": (count, 3),
            "velocity": (count, 3),
            "primitive": (count,),
            "radius": (count, 1),
            "half_extents": (count, 3),
            "mass": (count, 1),
            "restitution": (count, 1),
            "drag": (count, 1),
            "friction": (count, 1),
            "albedo": (count, 3),
            "orientation": (count, 4),
            "angular_velocity": (count, 3),
            "sleeping": (count,),
            "sleep_counter": (count,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"RigidBodyState.{name} must have shape {shape}")
        if self.object_id.dtype != torch.int64 or self.primitive.dtype != torch.int64:
            raise TypeError("rigid object IDs and primitive tags must use torch.int64")
        if self.active.dtype != torch.bool or self.sleeping.dtype != torch.bool:
            raise TypeError("rigid active and sleeping masks must use torch.bool")
        floating_names = (
            "position",
            "velocity",
            "radius",
            "half_extents",
            "mass",
            "restitution",
            "drag",
            "friction",
            "albedo",
            "orientation",
            "angular_velocity",
        )
        if not self.position.is_floating_point():
            raise TypeError("rigid physical tensors must be floating point")
        for name in floating_names:
            value = getattr(self, name)
            if value.dtype != self.position.dtype or value.device != self.position.device:
                raise ValueError("rigid physical tensors must share dtype and device")
        if bool(torch.any(self.active & (self.object_id < 0))):
            raise ValueError("active rigid bodies require nonnegative IDs")
        if bool(torch.any((~self.active) & (self.object_id != -1))):
            raise ValueError("inactive rigid slots require ID -1")
        supported = (self.primitive == int(RigidPrimitive.SPHERE)) | (
            self.primitive == int(RigidPrimitive.BOX)
        )
        if bool(torch.any(self.active & ~supported)):
            raise ValueError("unsupported rigid primitive tag")
        if bool(torch.any(self.active & (self.radius[:, 0] <= 0.0))):
            raise ValueError("active rigid bounding radii must be positive")
        if bool(torch.any(self.active & (self.mass[:, 0] <= 0.0))):
            raise ValueError("active rigid bodies require positive mass")
        if bool(
            torch.any(
                self.active & ((self.restitution[:, 0] < 0.0) | (self.restitution[:, 0] > 1.0))
            )
        ):
            raise ValueError("active rigid restitution must lie in [0,1]")
        if bool(torch.any(self.active & (self.drag[:, 0] < 0.0))):
            raise ValueError("active rigid drag must be nonnegative")
        if bool(torch.any(self.active & (self.friction[:, 0] < 0.0))):
            raise ValueError("active rigid friction must be nonnegative")
        if bool(
            torch.any(self.active & (torch.linalg.vector_norm(self.orientation, dim=-1) <= 1.0e-8))
        ):
            raise ValueError("active rigid orientations must have nonzero norm")
        box = self.active & (self.primitive == int(RigidPrimitive.BOX))
        if bool(torch.any(box.unsqueeze(-1) & (self.half_extents <= 0.0))):
            raise ValueError("active boxes require positive half-extents")
        for name in floating_names:
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"RigidBodyState.{name} contains NaN or Inf")


def _ray_sphere_depth(points_camera: Tensor, radius: Tensor, rays: Tensor) -> tuple[Tensor, Tensor]:
    ray_norm_squared = rays.square().sum(dim=-1)
    ray_dot_center = torch.einsum("hwc,nc->nhw", rays, points_camera)
    center_norm_squared = points_camera.square().sum(dim=-1)[:, None, None]
    cross = torch.linalg.cross(points_camera[:, None, None, :], rays[None], dim=-1)
    discriminant = ray_norm_squared.unsqueeze(0) * radius[
        :, None, None
    ].square() - cross.square().sum(dim=-1)
    root = discriminant.clamp_min(0.0).sqrt()
    denominator = ray_dot_center + root
    constant = center_norm_squared - radius[:, None, None].square()
    depth = constant / denominator.clamp_min(1.0e-12)
    valid = (discriminant >= 0.0) & (denominator > 0.0) & (depth > 0.0) & torch.isfinite(depth)
    return depth, valid


def _ray_box_depth(
    points_camera: Tensor,
    local_to_camera: Tensor,
    half_extents: Tensor,
    rays: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    camera_to_local = local_to_camera.transpose(-1, -2)
    local_origin = -torch.einsum("nij,nj->ni", camera_to_local, points_camera)
    local_direction = torch.einsum("nij,hwj->nihw", camera_to_local, rays)
    origin = local_origin[:, :, None, None]
    extent = half_extents[:, :, None, None]
    parallel = local_direction.abs() <= 1.0e-9
    parallel_inside = (~parallel | (origin.abs() <= extent)).all(dim=1)
    safe_direction = torch.where(parallel, torch.ones_like(local_direction), local_direction)
    first = (-extent - origin) / safe_direction
    second = (extent - origin) / safe_direction
    axis_near = torch.minimum(first, second)
    axis_far = torch.maximum(first, second)
    axis_near = torch.where(parallel, torch.full_like(axis_near, -torch.inf), axis_near)
    axis_far = torch.where(parallel, torch.full_like(axis_far, torch.inf), axis_far)
    near, near_axis = axis_near.max(dim=1)
    far = axis_far.min(dim=1).values
    depth = torch.where(near > 0.0, near, far)
    valid = parallel_inside & (far >= near.clamp_min(0.0)) & (depth > 0.0) & torch.isfinite(depth)

    # Face-oriented Lambertian cue.  It is visual only; metric fitting consumes
    # depth and calibration rather than this private face index.
    axis_world = torch.nn.functional.one_hot(near_axis, num_classes=3).to(points_camera.dtype)
    face_camera = torch.einsum("nij,nhwj->nhwi", local_to_camera, axis_world)
    shade = 0.55 + 0.45 * face_camera[..., 2].abs()
    return depth, valid, shade


def render_rigid_bodies(
    state: RigidBodyState,
    camera: CameraFrame,
    image_size: tuple[int, int],
    *,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> RenderOutput:
    """Render spheres and oriented boxes with exact perspective ray tests."""

    state.validate()
    camera.validate()
    if not bool((state.active & (state.primitive == int(RigidPrimitive.BOX))).any()):
        return render_spheres(
            state.as_spheres(),
            camera,
            image_size,
            noise_std=noise_std,
            generator=generator,
        )
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")

    dtype = state.position.dtype
    device = state.position.device
    count = state.max_objects
    points_camera = world_to_camera(state.position, camera.camera_from_world)
    centers_pixels, positive_depth = project_camera_points(points_camera, camera.intrinsics)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    rays = torch.stack(
        (
            (pixel_x - camera.intrinsics[0, 2]) / camera.intrinsics[0, 0],
            (pixel_y - camera.intrinsics[1, 2]) / camera.intrinsics[1, 1],
            torch.ones_like(pixel_x),
        ),
        dim=-1,
    )
    sphere_depth, sphere_hit = _ray_sphere_depth(points_camera, state.radius[:, 0], rays)
    world_rotation = _rotation_matrix(state.orientation)
    local_to_camera = torch.einsum("ij,njk->nik", camera.camera_from_world[:3, :3], world_rotation)
    box_depth, box_hit, box_shade = _ray_box_depth(
        points_camera,
        local_to_camera,
        state.half_extents,
        rays,
    )
    is_box = state.primitive == int(RigidPrimitive.BOX)
    metric_depth = torch.where(is_box[:, None, None], box_depth, sphere_depth)
    hit = torch.where(is_box[:, None, None], box_hit, sphere_hit)
    geometrically_valid = state.active & positive_depth & torch.isfinite(points_camera).all(dim=-1)
    full_mask = hit & geometrically_valid[:, None, None]
    infinity = torch.full_like(metric_depth, torch.inf)
    depth_buffer, winning_slot = torch.where(full_mask, metric_depth, infinity).min(dim=0)
    has_object = torch.isfinite(depth_buffer)
    instance_slot_map = torch.where(
        has_object,
        winning_slot.to(torch.int64),
        torch.full_like(winning_slot, -1, dtype=torch.int64),
    )
    safe_slot = winning_slot.clamp(0, max(count - 1, 0))
    instance_map = torch.where(
        has_object,
        state.object_id[safe_slot],
        torch.full_like(winning_slot, -1, dtype=torch.int64),
    )
    slot = torch.arange(count, device=device)[:, None, None]
    visible_mask = full_mask & has_object.unsqueeze(0) & (winning_slot.unsqueeze(0) == slot)
    support_pixels = full_mask.sum(dim=(-2, -1))
    visible_pixels = visible_mask.sum(dim=(-2, -1))
    visible_fraction = torch.where(
        support_pixels > 0,
        visible_pixels.to(dtype) / support_pixels.clamp_min(1).to(dtype),
        torch.zeros(count, dtype=dtype, device=device),
    )
    projected_valid = geometrically_valid & full_mask.any(dim=(-2, -1))
    bounding_radius = torch.where(
        is_box,
        torch.linalg.vector_norm(state.half_extents, dim=-1),
        state.radius[:, 0],
    )
    camera_depth = points_camera[:, 2]
    safe_camera_depth = camera_depth.clamp_min(1.0e-4)
    focal = 0.5 * (camera.intrinsics[0, 0] + camera.intrinsics[1, 1])
    apparent_radius = torch.where(
        projected_valid,
        focal * bounding_radius / safe_camera_depth,
        torch.zeros_like(camera_depth),
    )
    center_normalized = torch.stack(
        (
            2.0 * centers_pixels[:, 0] / max(width - 1, 1) - 1.0,
            2.0 * centers_pixels[:, 1] / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    center_normalized = torch.where(
        projected_valid.unsqueeze(-1), center_normalized, torch.zeros_like(center_normalized)
    )
    centers_pixels = torch.where(
        projected_valid.unsqueeze(-1), centers_pixels, torch.zeros_like(centers_pixels)
    )
    inverse_depth = torch.where(
        projected_valid, safe_camera_depth.reciprocal(), torch.zeros_like(camera_depth)
    )

    rgb = _background(height, width, dtype=dtype, device=device)
    sphere_shade = torch.full_like(box_shade, 0.84)
    shade = torch.where(is_box[:, None, None], box_shade, sphere_shade)
    soft_support = full_mask.to(dtype)
    for index in range(count):
        alpha = visible_mask[index].to(dtype)
        colour = state.albedo[index, :, None, None] * shade[index][None]
        rgb = rgb * (1.0 - alpha.unsqueeze(0)) + colour * alpha.unsqueeze(0)
    if noise_std > 0.0:
        rgb = rgb + noise_std * torch.randn(
            rgb.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    return RenderOutput(
        rgb=rgb.clamp(0.0, 1.0).to(torch.float32),
        depth_buffer=torch.where(has_object, depth_buffer, torch.zeros_like(depth_buffer)),
        instance_map=instance_map,
        instance_slot_map=instance_slot_map,
        visible_mask=visible_mask,
        full_mask=full_mask,
        soft_support=soft_support.to(torch.float32),
        visible_fraction=torch.where(
            state.active, visible_fraction, torch.zeros_like(visible_fraction)
        ).to(torch.float32),
        projected_center=center_normalized.to(torch.float32),
        projected_center_pixels=centers_pixels.to(torch.float32),
        apparent_radius=apparent_radius.to(torch.float32),
        inverse_depth=inverse_depth.to(torch.float32),
        camera_depth=camera_depth.to(torch.float32),
        projected_valid=projected_valid,
    )


__all__ = ["RigidBodyState", "render_rigid_bodies"]
