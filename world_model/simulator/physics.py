"""Deterministic semi-implicit sphere physics for the synthetic environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from world_model.simulator.collisions import (
    BOUNDARY_NAMES,
    resolve_axis_aligned_boundaries,
    resolve_sphere_sphere_collisions,
)


@dataclass(frozen=True)
class SphereState:
    """Physical state for one padded scene.

    All tensors live on CPU in the data simulator.  Object scalar properties use
    shape ``[N, 1]`` so the episode contract can stack them directly across
    time.  Padded slots have ``active=False`` and ``object_id=-1``.
    """

    object_id: Tensor
    active: Tensor
    position: Tensor
    velocity: Tensor
    radius: Tensor
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

    def clone(self) -> SphereState:
        """Deep-clone tensors so simulator steps never mutate caller state."""

        return SphereState(
            **{
                field_name: getattr(self, field_name).clone()
                for field_name in self.__dataclass_fields__
            }
        )

    def validate(self) -> None:
        """Raise an actionable error when the padded object contract is invalid."""

        count = self.max_objects
        expected_shapes = {
            "object_id": (count,),
            "active": (count,),
            "position": (count, 3),
            "velocity": (count, 3),
            "radius": (count, 1),
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
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"SphereState.{name} must have shape {expected}, got {tuple(value.shape)}"
                )
        if self.object_id.dtype != torch.int64:
            raise TypeError("object_id must use torch.int64")
        if self.active.dtype != torch.bool or self.sleeping.dtype != torch.bool:
            raise TypeError("active and sleeping must use torch.bool")
        if bool(torch.any(self.active & (self.object_id < 0))):
            raise ValueError("active objects must have nonnegative object IDs")
        if bool(torch.any((~self.active) & (self.object_id != -1))):
            raise ValueError("inactive/padded object slots must have ID -1")
        if bool(torch.any(self.radius[self.active] <= 0)):
            raise ValueError("active sphere radii must be positive")
        if bool(torch.any(self.mass[self.active] <= 0)):
            raise ValueError("active sphere masses must be positive")
        for name in (
            "position",
            "velocity",
            "radius",
            "mass",
            "restitution",
            "drag",
            "friction",
            "albedo",
            "orientation",
            "angular_velocity",
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"SphereState.{name} contains NaN or Inf")


@dataclass(frozen=True)
class PhysicsConfig:
    """Numerical and environmental parameters for sphere integration."""

    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    bounds: tuple[tuple[float, float], ...] = (
        (-2.25, 2.25),
        (0.0, 3.25),
        (-1.5, 1.5),
    )
    max_substep: float = 1.0 / 120.0
    solver_iterations: int = 2
    position_correction: float = 0.8
    penetration_slop: float = 1.0e-4
    max_position_correction: float = 0.08
    sleep_speed: float = 0.035
    sleep_after_seconds: float = 0.35
    wake_impulse: float = 0.02

    def validate(self) -> None:
        if self.max_substep <= 0 or not math.isfinite(self.max_substep):
            raise ValueError("max_substep must be finite and positive")
        if self.solver_iterations < 1:
            raise ValueError("solver_iterations must be at least one")
        if len(self.gravity) != 3:
            raise ValueError("gravity must contain three values")
        bounds = torch.as_tensor(self.bounds)
        if bounds.shape != (3, 2) or bool(torch.any(bounds[:, 1] <= bounds[:, 0])):
            raise ValueError("bounds must be increasing [3, 2] x/y/z limits")
        if self.sleep_speed < 0 or self.sleep_after_seconds < 0:
            raise ValueError("sleep thresholds must be nonnegative")


@dataclass(frozen=True)
class PhysicsStepEvents:
    """Events accumulated over all high-rate substeps of one interval."""

    pair_contact: Tensor
    pair_collision: Tensor
    pair_impulse: Tensor
    pair_penetration: Tensor
    boundary_contact: Tensor
    boundary_collision: Tensor
    boundary_impulse: Tensor
    boundary_penetration: Tensor
    collision: Tensor
    contact: Tensor
    sleeping: Tensor
    external_impulse: Tensor
    first_event_offset: Tensor
    substeps: int

    @property
    def ground_contact(self) -> Tensor:
        return self.boundary_contact[:, BOUNDARY_NAMES.index("floor")]

    @property
    def ground_collision(self) -> Tensor:
        return self.boundary_collision[:, BOUNDARY_NAMES.index("floor")]

    @property
    def wall_collision(self) -> Tensor:
        wall_indices = [
            index for index, name in enumerate(BOUNDARY_NAMES) if name not in {"floor", "ceiling"}
        ]
        return self.boundary_collision[:, wall_indices]


def empty_physics_events(
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    substeps: int = 0,
) -> PhysicsStepEvents:
    """Create an all-clear event structure for ``count`` padded objects."""

    bool_pair = torch.zeros((count, count), dtype=torch.bool, device=device)
    bool_boundary = torch.zeros((count, 6), dtype=torch.bool, device=device)
    return PhysicsStepEvents(
        pair_contact=bool_pair.clone(),
        pair_collision=bool_pair.clone(),
        pair_impulse=torch.zeros((count, count), dtype=dtype, device=device),
        pair_penetration=torch.zeros((count, count), dtype=dtype, device=device),
        boundary_contact=bool_boundary.clone(),
        boundary_collision=bool_boundary.clone(),
        boundary_impulse=torch.zeros((count, 6), dtype=dtype, device=device),
        boundary_penetration=torch.zeros((count, 6), dtype=dtype, device=device),
        collision=torch.zeros(count, dtype=torch.bool, device=device),
        contact=torch.zeros(count, dtype=torch.bool, device=device),
        sleeping=torch.zeros(count, dtype=torch.bool, device=device),
        external_impulse=torch.zeros((count, 3), dtype=dtype, device=device),
        first_event_offset=torch.full((count,), -1.0, dtype=dtype, device=device),
        substeps=substeps,
    )


def _replace_state(
    state: SphereState,
    *,
    position: Tensor,
    velocity: Tensor,
    sleeping: Tensor,
    sleep_counter: Tensor,
) -> SphereState:
    return replace(
        state,
        position=position,
        velocity=velocity,
        sleeping=sleeping,
        sleep_counter=sleep_counter,
    )


def advance_spheres(
    state: SphereState,
    dt: float,
    config: PhysicsConfig,
    *,
    external_impulse: Tensor | None = None,
) -> tuple[SphereState, PhysicsStepEvents]:
    """Advance a scene by real seconds using deterministic high-rate substeps.

    Linear drag is integrated exponentially, gravity is applied
    semi-implicitly, and analytic contact impulses are resolved after each
    position step.  ``state`` is never mutated.
    """

    config.validate()
    state.validate()
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    count = state.max_objects
    if dt == 0:
        return state.clone(), empty_physics_events(
            count,
            dtype=state.position.dtype,
            device=state.position.device,
            substeps=0,
        )
    num_substeps = max(1, math.ceil(dt / config.max_substep))
    sub_dt = dt / num_substeps
    events = empty_physics_events(
        count,
        dtype=state.position.dtype,
        device=state.position.device,
        substeps=num_substeps,
    )

    position = state.position.clone()
    velocity = state.velocity.clone()
    sleeping = state.sleeping.clone()
    sleep_counter = state.sleep_counter.clone()
    active = state.active
    if external_impulse is None:
        external_impulse = torch.zeros_like(velocity)
    if external_impulse.shape != velocity.shape:
        raise ValueError("external_impulse must have shape [N, 3]")
    external_impulse = torch.where(
        active.unsqueeze(-1), external_impulse, torch.zeros_like(external_impulse)
    )
    impulse_norm = torch.linalg.vector_norm(external_impulse, dim=-1)
    externally_acted = active & (impulse_norm > 0)
    velocity = velocity + external_impulse / state.mass.clamp_min(1.0e-12)
    sleeping = sleeping & ~externally_acted
    sleep_counter = torch.where(externally_acted, torch.zeros_like(sleep_counter), sleep_counter)

    pair_contact = events.pair_contact
    pair_collision = events.pair_collision
    pair_impulse = events.pair_impulse
    pair_penetration = events.pair_penetration
    boundary_contact = events.boundary_contact
    boundary_collision = events.boundary_collision
    boundary_impulse = events.boundary_impulse
    boundary_penetration = events.boundary_penetration
    first_event_offset = events.first_event_offset
    gravity = torch.as_tensor(config.gravity, dtype=position.dtype, device=position.device)
    bounds = torch.as_tensor(config.bounds, dtype=position.dtype, device=position.device)

    for substep_index in range(num_substeps):
        movable = active & ~sleeping
        drag_factor = torch.exp(-state.drag.clamp_min(0.0) * sub_dt)
        velocity = torch.where(
            movable.unsqueeze(-1),
            velocity * drag_factor + gravity.unsqueeze(0) * sub_dt,
            torch.zeros_like(velocity),
        )
        position = torch.where(
            movable.unsqueeze(-1),
            position + velocity * sub_dt,
            position,
        )

        substep_pair_collision = torch.zeros_like(pair_collision)
        substep_boundary_collision = torch.zeros_like(boundary_collision)
        substep_contact_objects = torch.zeros(count, dtype=torch.bool, device=position.device)
        substep_collision_objects = torch.zeros_like(substep_contact_objects)
        for _ in range(config.solver_iterations):
            boundary_result = resolve_axis_aligned_boundaries(
                position,
                velocity,
                state.radius,
                state.mass,
                state.restitution,
                bounds,
                friction=state.friction,
                active=active,
            )
            position, velocity = boundary_result.position, boundary_result.velocity
            pair_result = resolve_sphere_sphere_collisions(
                position,
                velocity,
                state.radius,
                state.mass,
                state.restitution,
                friction=state.friction,
                active=active,
                position_correction=config.position_correction,
                penetration_slop=config.penetration_slop,
                max_position_correction=config.max_position_correction,
            )
            position, velocity = pair_result.position, pair_result.velocity

            pair_contact |= pair_result.contact
            pair_collision |= pair_result.collision
            pair_impulse = torch.maximum(pair_impulse, pair_result.impulse_magnitude)
            pair_penetration = torch.maximum(pair_penetration, pair_result.penetration)
            boundary_contact |= boundary_result.contact
            boundary_collision |= boundary_result.collision
            boundary_impulse = torch.maximum(boundary_impulse, boundary_result.impulse_magnitude)
            boundary_penetration = torch.maximum(boundary_penetration, boundary_result.penetration)
            substep_pair_collision |= pair_result.collision
            substep_boundary_collision |= boundary_result.collision
            substep_contact_objects |= pair_result.contact.any(
                dim=-1
            ) | boundary_result.contact.any(dim=-1)
            substep_collision_objects |= pair_result.collision.any(
                dim=-1
            ) | boundary_result.collision.any(dim=-1)

        woke_by_collision = substep_collision_objects & (
            pair_impulse.max(dim=-1).values > config.wake_impulse
        )
        woke_by_collision |= substep_boundary_collision.any(dim=-1) & (
            boundary_impulse.max(dim=-1).values > config.wake_impulse
        )
        sleeping = sleeping & ~woke_by_collision

        speed = torch.linalg.vector_norm(velocity, dim=-1)
        floor_contact = boundary_result.contact[:, BOUNDARY_NAMES.index("floor")]
        sleep_candidate = active & floor_contact & (speed < config.sleep_speed)
        sleep_counter = torch.where(
            sleep_candidate, sleep_counter + 1, torch.zeros_like(sleep_counter)
        )
        required_sleep_steps = max(1, math.ceil(config.sleep_after_seconds / sub_dt))
        newly_sleeping = sleep_candidate & (sleep_counter >= required_sleep_steps)
        sleeping |= newly_sleeping
        velocity = torch.where(sleeping.unsqueeze(-1), torch.zeros_like(velocity), velocity)

        event_this_substep = substep_collision_objects
        first_event_offset = torch.where(
            event_this_substep & (first_event_offset < 0),
            torch.full_like(first_event_offset, (substep_index + 1) * sub_dt),
            first_event_offset,
        )

    collision_objects = pair_collision.any(dim=-1) | boundary_collision.any(dim=-1)
    contact_objects = pair_contact.any(dim=-1) | boundary_contact.any(dim=-1)
    updated = _replace_state(
        state,
        position=position,
        velocity=velocity,
        sleeping=sleeping,
        sleep_counter=sleep_counter,
    )
    updated.validate()
    return updated, PhysicsStepEvents(
        pair_contact=pair_contact,
        pair_collision=pair_collision,
        pair_impulse=pair_impulse,
        pair_penetration=pair_penetration,
        boundary_contact=boundary_contact,
        boundary_collision=boundary_collision,
        boundary_impulse=boundary_impulse,
        boundary_penetration=boundary_penetration,
        collision=collision_objects,
        contact=contact_objects,
        sleeping=sleeping.clone(),
        external_impulse=external_impulse.clone(),
        first_event_offset=first_event_offset,
        substeps=num_substeps,
    )
