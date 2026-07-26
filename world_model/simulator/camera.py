"""Calibrated synthetic camera trajectories and coordinate transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CameraFrame:
    """Calibration at one timestamp.

    ``world_from_camera`` maps camera coordinates into world coordinates.  The
    camera looks along camera ``+z``; camera ``+x`` points image-right and
    camera ``+y`` points image-down.  Intrinsics therefore use positive focal
    lengths for both image axes.
    """

    timestamp: float
    world_from_camera: Tensor
    camera_from_world: Tensor
    intrinsics: Tensor
    position: Tensor
    target: Tensor

    def validate(self) -> None:
        if self.world_from_camera.shape != (4, 4):
            raise ValueError("world_from_camera must have shape [4, 4]")
        if self.camera_from_world.shape != (4, 4):
            raise ValueError("camera_from_world must have shape [4, 4]")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape [3, 3]")
        identity = self.world_from_camera @ self.camera_from_world
        expected = torch.eye(4, dtype=identity.dtype, device=identity.device)
        if not torch.allclose(identity, expected, atol=2.0e-5, rtol=2.0e-5):
            raise ValueError("camera transforms are not mutual inverses")
        rotation = self.world_from_camera[:3, :3]
        if not torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            torch.eye(3, dtype=rotation.dtype, device=rotation.device),
            atol=2.0e-5,
            rtol=2.0e-5,
        ):
            raise ValueError("camera rotation is not orthonormal")


@dataclass(frozen=True)
class CameraTrajectoryConfig:
    """Parameters for deterministic calibrated camera movement."""

    image_size: tuple[int, int] = (64, 64)
    mode: str = "fixed"
    vertical_fov_degrees: float = 48.0
    base_position: tuple[float, float, float] = (0.0, 2.15, 5.6)
    target: tuple[float, float, float] = (0.0, 0.95, 0.0)
    orbit_speed: float = 0.18
    orbit_amplitude: float = 0.32
    translation_amplitude: float = 0.35
    zoom_amplitude: float = 0.0

    def validate(self) -> None:
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("camera image dimensions must be positive")
        if not 5.0 < self.vertical_fov_degrees < 170.0:
            raise ValueError("vertical_fov_degrees must be between 5 and 170")
        allowed = {"fixed", "linear", "orbit", "combined", "mixed"}
        if self.mode not in allowed:
            raise ValueError(f"camera trajectory mode must be one of {sorted(allowed)}")


def look_at_world_from_camera(
    position: Tensor,
    target: Tensor,
    *,
    world_up: Tensor | None = None,
) -> Tensor:
    """Construct a right-handed ``T_world_from_camera`` look-at transform."""

    if position.shape != (3,) or target.shape != (3,):
        raise ValueError("position and target must have shape [3]")
    if world_up is None:
        world_up = position.new_tensor([0.0, 1.0, 0.0])
    forward = target - position
    forward = forward / torch.linalg.vector_norm(forward).clamp_min(1.0e-12)
    right = torch.linalg.cross(forward, world_up)
    if float(torch.linalg.vector_norm(right)) < 1.0e-6:
        raise ValueError("camera forward direction is parallel to world_up")
    right = right / torch.linalg.vector_norm(right)
    # Camera y points down in the rendered image.
    down = torch.linalg.cross(forward, right)
    down = down / torch.linalg.vector_norm(down).clamp_min(1.0e-12)
    transform = torch.eye(4, dtype=position.dtype, device=position.device)
    transform[:3, :3] = torch.stack((right, down, forward), dim=-1)
    transform[:3, 3] = position
    return transform


def invert_rigid_transform(transform: Tensor) -> Tensor:
    """Invert a homogeneous rigid transform without a generic matrix inverse."""

    if transform.shape != (4, 4):
        raise ValueError("transform must have shape [4, 4]")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = torch.eye(4, dtype=transform.dtype, device=transform.device)
    inverse[:3, :3] = rotation.transpose(0, 1)
    inverse[:3, 3] = -(rotation.transpose(0, 1) @ translation)
    return inverse


def make_intrinsics(
    image_size: tuple[int, int],
    vertical_fov_degrees: float,
    *,
    focal_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Return calibrated pinhole intrinsics for image ``(height, width)``."""

    height, width = image_size
    fov_radians = math.radians(vertical_fov_degrees)
    focal = focal_scale * (0.5 * height) / math.tan(0.5 * fov_radians)
    intrinsics = torch.eye(3, dtype=dtype, device=device)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = 0.5 * (width - 1)
    intrinsics[1, 2] = 0.5 * (height - 1)
    return intrinsics


class CameraTrajectory:
    """A seed-parameterised camera path with calibration available at any time."""

    def __init__(self, config: CameraTrajectoryConfig, seed: int) -> None:
        config.validate()
        self.config = config
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
        modes = ("fixed", "linear", "orbit", "combined")
        if config.mode == "mixed":
            mode_index = int(torch.randint(0, len(modes), (), generator=generator))
            self.mode = modes[mode_index]
        else:
            self.mode = config.mode
        self.phase = float(torch.rand((), generator=generator) * (2.0 * math.pi))
        direction = torch.randn(3, generator=generator)
        direction[1] *= 0.25
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-6)
        self.translation_direction = direction
        self.orbit_direction = (
            -1.0 if int(torch.randint(0, 2, (), generator=generator)) == 0 else 1.0
        )

    def at(
        self,
        timestamp: float,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> CameraFrame:
        """Evaluate camera pose and intrinsics at ``timestamp`` seconds."""

        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("camera timestamp must be finite and nonnegative")
        config = self.config
        base = torch.tensor(config.base_position, dtype=dtype, device=device)
        target = torch.tensor(config.target, dtype=dtype, device=device)
        direction = self.translation_direction.to(dtype=dtype, device=device)
        position = base.clone()

        if self.mode in {"linear", "combined"}:
            travel = config.translation_amplitude * math.sin(0.45 * timestamp + self.phase)
            position = position + direction * travel
            target = target + 0.18 * direction * travel

        if self.mode in {"orbit", "combined"}:
            relative = base - target
            base_angle = math.atan2(float(relative[0]), float(relative[2]))
            angular_offset = (
                self.orbit_direction
                * config.orbit_amplitude
                * math.sin(config.orbit_speed * timestamp + self.phase)
            )
            angle = base_angle + angular_offset
            horizontal_radius = math.hypot(float(relative[0]), float(relative[2]))
            position[0] = target[0] + horizontal_radius * math.sin(angle)
            position[2] = target[2] + horizontal_radius * math.cos(angle)
            position[1] = base[1] + 0.12 * math.sin(0.31 * timestamp + self.phase)

        # A gentle target drift creates rotation even for the linear path.
        if self.mode != "fixed":
            target = target.clone()
            target[0] += 0.08 * math.sin(0.37 * timestamp + self.phase)
            target[1] += 0.04 * math.cos(0.29 * timestamp + self.phase)

        world_from_camera = look_at_world_from_camera(position, target)
        camera_from_world = invert_rigid_transform(world_from_camera)
        focal_scale = 1.0 + config.zoom_amplitude * math.sin(0.23 * timestamp + self.phase)
        intrinsics = make_intrinsics(
            config.image_size,
            config.vertical_fov_degrees,
            focal_scale=focal_scale,
            dtype=dtype,
            device=device,
        )
        frame = CameraFrame(
            timestamp=timestamp,
            world_from_camera=world_from_camera,
            camera_from_world=camera_from_world,
            intrinsics=intrinsics,
            position=position,
            target=target,
        )
        frame.validate()
        return frame


def world_to_camera(points_world: Tensor, camera_from_world: Tensor) -> Tensor:
    """Transform ``[..., 3]`` world points to camera coordinates."""

    if points_world.shape[-1] != 3:
        raise ValueError("points_world must end in dimension 3")
    if camera_from_world.shape != (4, 4):
        raise ValueError("camera_from_world must have shape [4, 4]")
    return points_world @ camera_from_world[:3, :3].transpose(0, 1) + camera_from_world[:3, 3]


def camera_to_world(points_camera: Tensor, world_from_camera: Tensor) -> Tensor:
    """Transform ``[..., 3]`` camera points to world coordinates."""

    if points_camera.shape[-1] != 3:
        raise ValueError("points_camera must end in dimension 3")
    if world_from_camera.shape != (4, 4):
        raise ValueError("world_from_camera must have shape [4, 4]")
    return points_camera @ world_from_camera[:3, :3].transpose(0, 1) + world_from_camera[:3, 3]


def project_camera_points(
    points_camera: Tensor,
    intrinsics: Tensor,
    *,
    min_depth: float = 1.0e-4,
) -> tuple[Tensor, Tensor]:
    """Project camera points to pixel ``(u, v)`` and return a validity mask."""

    if points_camera.shape[-1] != 3:
        raise ValueError("points_camera must end in dimension 3")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3, 3]")
    depth = points_camera[..., 2]
    valid = depth > min_depth
    safe_depth = depth.clamp_min(min_depth)
    u = intrinsics[0, 0] * points_camera[..., 0] / safe_depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[..., 1] / safe_depth + intrinsics[1, 2]
    pixels = torch.stack((u, v), dim=-1)
    return pixels, valid


def backproject_pixels(
    pixels: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
) -> Tensor:
    """Back-project pixel centres and metric depth into camera coordinates."""

    if pixels.shape[-1] != 2:
        raise ValueError("pixels must end in dimension 2")
    if depth.shape != pixels.shape[:-1]:
        raise ValueError("depth shape must match pixels without the last dimension")
    x = (pixels[..., 0] - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (pixels[..., 1] - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    return torch.stack((x, y, depth), dim=-1)
