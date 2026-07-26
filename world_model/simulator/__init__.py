"""Deterministic synthetic RGB sphere environment and exact labels."""

from world_model.simulator.camera import (
    CameraFrame,
    CameraTrajectory,
    CameraTrajectoryConfig,
    backproject_pixels,
    camera_to_world,
    invert_rigid_transform,
    look_at_world_from_camera,
    make_intrinsics,
    project_camera_points,
    world_to_camera,
)
from world_model.simulator.collisions import (
    BOUNDARY_NAMES,
    BoundaryCollisionResult,
    PairCollisionResult,
    resolve_axis_aligned_boundaries,
    resolve_sphere_sphere_collisions,
    sphere_sphere_relative_restitution,
)
from world_model.simulator.episode import Episode, generate_episode, validate_episode
from world_model.simulator.physics import (
    PhysicsConfig,
    PhysicsStepEvents,
    SphereState,
    advance_spheres,
    empty_physics_events,
)
from world_model.simulator.renderer import RenderOutput, render_spheres
from world_model.simulator.sphere_world import (
    LifecycleEvents,
    SphereWorld,
    SphereWorldConfig,
)

__all__ = [
    "BOUNDARY_NAMES",
    "BoundaryCollisionResult",
    "CameraFrame",
    "CameraTrajectory",
    "CameraTrajectoryConfig",
    "Episode",
    "LifecycleEvents",
    "PairCollisionResult",
    "PhysicsConfig",
    "PhysicsStepEvents",
    "RenderOutput",
    "SphereState",
    "SphereWorld",
    "SphereWorldConfig",
    "advance_spheres",
    "backproject_pixels",
    "camera_to_world",
    "empty_physics_events",
    "generate_episode",
    "invert_rigid_transform",
    "look_at_world_from_camera",
    "make_intrinsics",
    "project_camera_points",
    "render_spheres",
    "resolve_axis_aligned_boundaries",
    "resolve_sphere_sphere_collisions",
    "sphere_sphere_relative_restitution",
    "validate_episode",
    "world_to_camera",
]
