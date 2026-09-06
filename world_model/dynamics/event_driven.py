"""Fast continuous-time sphere impulses for state-only CPU rollouts.

The ordinary :class:`~world_model.dynamics.model.DynamicsModel` deliberately
uses a small fixed physical step because it also emits dense contact/event
traces.  Planning only consumes sampled state.  For the specification-1.61
profile (zero gravity, common linear drag, no continuous learned force, and no
boundary reachability), relative sphere paths have an analytic scalar
parameterisation.  This module finds their first time of impact and applies
the same bounded pair impulse as the authoritative contact resolver.

The helper is opt-in and returns ``None`` when its proof obligations are not
met or the bounded event budget is exhausted.  Callers can then use the
ordinary microstep implementation; no unsupported state is approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from world_model.belief import ObjectBeliefTensor
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.contacts import _safe_tangent_direction
from world_model.dynamics.graph import InteractionGraph


@dataclass(frozen=True)
class EventDrivenStateResult:
    """Mean-state result plus relation-owned process-uncertainty evidence."""

    objects: ObjectBeliefTensor
    relation_process_log_scale: Tensor
    collision_count: Tensor
    pair_collision: Tensor


def _common_active_drag(objects: ObjectBeliefTensor, *, tolerance: float) -> Tensor | None:
    """Return one drag coefficient per row, or ``None`` if it is not common."""

    active = objects.active
    first_slot = active.to(torch.int64).argmax(dim=-1)
    batch = torch.arange(objects.batch_size, device=objects.position.device)
    common = objects.drag[batch, first_slot, 0]
    comparable = torch.where(active, objects.drag[..., 0], common.unsqueeze(-1))
    if bool((comparable - common.unsqueeze(-1)).abs().gt(tolerance).any()):
        return None
    return common


def _path_parameter(drag: Tensor, elapsed: Tensor) -> Tensor:
    """Distance multiplier ``s(t)`` for common linear drag."""

    use_drag = drag >= 1.0e-5
    safe_drag = drag.clamp_min(1.0e-5)
    with_drag = -torch.expm1(-safe_drag * elapsed) / safe_drag
    return torch.where(use_drag, with_drag, elapsed)


def _elapsed_from_path_parameter(drag: Tensor, distance_scale: Tensor) -> Tensor:
    """Inverse of :func:`_path_parameter` on its nonnegative domain."""

    use_drag = drag >= 1.0e-5
    safe_drag = drag.clamp_min(1.0e-5)
    argument = (-safe_drag * distance_scale).clamp(
        min=-1.0 + 8.0 * torch.finfo(distance_scale.dtype).eps,
        max=0.0,
    )
    with_drag = -torch.log1p(argument) / safe_drag
    return torch.where(use_drag, with_drag, distance_scale)


def _earliest_pair_impact(
    objects: ObjectBeliefTensor,
    remaining: Tensor,
    common_drag: Tensor,
    *,
    collision_speed_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return impact time, selected upper-triangle pair, and occurrence mask."""

    batch_size, count = objects.active.shape
    relative_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]
    relative_velocity = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]
    quadratic = relative_velocity.square().sum(dim=-1)
    linear = 2.0 * (relative_position * relative_velocity).sum(dim=-1)
    radii = objects.radius[..., 0]
    constant = (
        relative_position.square().sum(dim=-1) - (radii[:, :, None] + radii[:, None, :]).square()
    )
    discriminant = linear.square() - 4.0 * quadratic * constant

    upper = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=objects.position.device),
        diagonal=1,
    ).unsqueeze(0)
    pair_active = (
        objects.active[:, :, None]
        & objects.active[:, None, :]
        & upper
        & (remaining > 0.0)[:, None, None]
    )
    moving = quadratic > torch.finfo(objects.position.dtype).eps
    real_root = discriminant >= 0.0
    # Dense self/inactive pairs can have an exactly zero discriminant.  They
    # are masked from event selection, but sqrt'(0)=inf can still create
    # ``0 * inf`` NaNs in the complete relation gradient.  A dtype-scaled
    # floor leaves every ordinary physical root unchanged while keeping the
    # masked graph differentiable.
    discriminant_floor = torch.finfo(objects.position.dtype).eps ** 2
    root = (-linear - discriminant.clamp_min(discriminant_floor).sqrt()) / (
        2.0 * quadratic.clamp_min(torch.finfo(objects.position.dtype).eps)
    )
    maximum_path = _path_parameter(common_drag, remaining).unsqueeze(-1).unsqueeze(-1)
    root_epsilon = 16.0 * torch.finfo(objects.position.dtype).eps
    approaching_at_root = linear + 2.0 * quadratic * root
    ordinary = (
        pair_active
        & moving
        & real_root
        & (root > root_epsilon)
        & (root <= maximum_path + root_epsilon)
        & (approaching_at_root < -float(collision_speed_epsilon))
    )
    # A preceding impact can leave a different pair infinitesimally
    # overlapping at the same timestamp. Resolve it immediately only when the
    # pair is genuinely approaching; a separating contact must not bounce a
    # second time.
    immediate = pair_active & (constant <= 0.0) & (linear < -2.0 * float(collision_speed_epsilon))
    candidate_path = torch.where(
        immediate,
        torch.zeros_like(root),
        torch.where(ordinary, root, torch.full_like(root, torch.inf)),
    )
    flat = candidate_path.flatten(start_dim=1)
    earliest_path, flat_pair = flat.min(dim=-1)
    occurred = torch.isfinite(earliest_path)
    safe_path = torch.where(occurred, earliest_path, torch.zeros_like(earliest_path))
    impact_time = _elapsed_from_path_parameter(common_drag, safe_path)
    impact_time = torch.minimum(impact_time, remaining)

    selected_flat = torch.nn.functional.one_hot(
        flat_pair,
        num_classes=count * count,
    ).to(torch.bool)
    selected = selected_flat.reshape(batch_size, count, count)
    selected = selected & occurred[:, None, None] & upper
    return impact_time, selected, occurred


def _apply_selected_pair_impulse(
    objects: ObjectBeliefTensor,
    global_code: Tensor,
    selected_upper: Tensor,
    interactions: InteractionGraph,
    *,
    collision_speed_epsilon: float,
    maximum_multiplier_residual: float,
    maximum_additive_residual: float,
) -> tuple[ObjectBeliefTensor, Tensor, Tensor]:
    """Apply one selected analytic/bounded learned impulse per batch row."""

    graph = interactions(
        objects,
        global_code,
        modal_acceleration=torch.zeros_like(objects.position),
    )
    relative_position = objects.position[:, None, :, :] - objects.position[:, :, None, :]
    distance = torch.linalg.vector_norm(relative_position, dim=-1).clamp_min(1.0e-7)
    normal = relative_position / distance.unsqueeze(-1)
    relative_velocity = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]
    relative_normal_velocity = (relative_velocity * normal).sum(dim=-1)
    collision = selected_upper & (relative_normal_velocity < -float(collision_speed_epsilon))
    inverse_mass = objects.mass[..., 0].reciprocal()
    inverse_mass_sum = (inverse_mass[:, :, None] + inverse_mass[:, None, :]).clamp_min(1.0e-8)
    restitution = torch.minimum(
        objects.restitution[..., 0][:, :, None],
        objects.restitution[..., 0][:, None, :],
    )
    impulse = (-(1.0 + restitution) * relative_normal_velocity / inverse_mass_sum).clamp_min(0.0)
    multiplier = 1.0 + float(maximum_multiplier_residual) * torch.tanh(graph.impulse_multiplier_raw)
    additive = float(maximum_additive_residual) * torch.tanh(graph.impulse_additive_raw)
    impulse = (impulse * multiplier + additive).clamp_min(0.0) * collision

    relative_tangent = relative_velocity - relative_normal_velocity.unsqueeze(-1) * normal
    tangent_speed = torch.linalg.vector_norm(relative_tangent, dim=-1)
    tangent_direction = _safe_tangent_direction(relative_tangent, tangent_speed)
    friction = torch.sqrt(
        objects.friction[..., 0][:, :, None].clamp_min(0.0)
        * objects.friction[..., 0][:, None, :].clamp_min(0.0)
    )
    friction_impulse = torch.minimum(
        friction * impulse,
        tangent_speed / inverse_mass_sum,
    )
    momentum_i_upper = (
        -impulse.unsqueeze(-1) * normal + friction_impulse.unsqueeze(-1) * tangent_direction
    )
    pair_momentum = momentum_i_upper - momentum_i_upper.transpose(1, 2)
    velocity = objects.velocity + pair_momentum.sum(dim=2) * inverse_mass.unsqueeze(-1)
    updated = replace(
        objects,
        velocity=torch.where(
            objects.active.unsqueeze(-1),
            velocity,
            objects.velocity,
        ),
    )

    # The seventh symmetric edge output is a signed relation-owned log-scale
    # residual. Accumulate it only on the pair that physically interacted.
    selected_symmetric = selected_upper | selected_upper.transpose(1, 2)
    relation_process_log_scale = torch.where(
        selected_symmetric,
        graph.edge_process_noise,
        torch.zeros_like(graph.edge_process_noise),
    ).sum(dim=-1)
    pair_collision = collision | collision.transpose(1, 2)
    return updated, relation_process_log_scale, pair_collision


def _boundary_reach_is_impossible(
    objects: ObjectBeliefTensor,
    elapsed: Tensor,
    world_bounds: tuple[tuple[float, float], ...],
    *,
    maximum_events: int,
    maximum_additive_impulse: float,
) -> bool:
    """Conservatively certify that no supported trajectory reaches a plane."""

    if len(world_bounds) != 3:
        return False
    speed = torch.linalg.vector_norm(objects.velocity, dim=-1)
    maximum_speed = speed.masked_fill(~objects.active, 0.0).amax(dim=-1)
    inverse_mass = objects.mass[..., 0].reciprocal().masked_fill(~objects.active, 0.0)
    maximum_inverse_mass = inverse_mass.amax(dim=-1)
    speed_bound = maximum_speed + (
        float(maximum_events) * float(maximum_additive_impulse) * maximum_inverse_mass
    )
    displacement_bound = speed_bound * elapsed
    radius = objects.radius[..., 0]
    safe = torch.ones_like(objects.active)
    for axis, (lower, upper) in enumerate(world_bounds):
        lower_clearance = objects.position[..., axis] - radius - float(lower)
        upper_clearance = float(upper) - objects.position[..., axis] - radius
        safe = safe & (lower_clearance > displacement_bound.unsqueeze(-1))
        safe = safe & (upper_clearance > displacement_bound.unsqueeze(-1))
    return bool((safe | ~objects.active).all())


def integrate_event_driven_state_only(
    objects: ObjectBeliefTensor,
    gravity: Tensor,
    global_code: Tensor,
    elapsed: Tensor,
    *,
    analytic: AnalyticKinematics,
    interactions: InteractionGraph,
    world_bounds: tuple[tuple[float, float], ...],
    collision_speed_epsilon: float,
    maximum_multiplier_residual: float,
    maximum_additive_residual: float,
    maximum_events: int = 8,
) -> EventDrivenStateResult | None:
    """Integrate a certified relation-only sphere set without fixed microsteps.

    ``None`` means that a proof obligation failed.  In particular, this helper
    never ignores gravity, unequal drag, reachable boundaries, or an event
    sequence longer than its explicit budget.
    """

    if elapsed.shape != (objects.batch_size,):
        raise ValueError("event-driven elapsed time must have shape [B]")
    if elapsed.device != objects.position.device or elapsed.dtype != objects.position.dtype:
        raise ValueError("event-driven elapsed time must share object device and dtype")
    if global_code.shape != (objects.batch_size, interactions.global_code_dim):
        raise ValueError("event-driven global code has an incompatible shape")
    if isinstance(maximum_events, bool) or not isinstance(maximum_events, int):
        raise TypeError("maximum_events must be an integer")
    if maximum_events <= 0:
        raise ValueError("maximum_events must be positive")
    if bool(gravity.ne(0.0).any()):
        return None
    common_drag = _common_active_drag(objects, tolerance=1.0e-6)
    if common_drag is None:
        return None
    if not _boundary_reach_is_impossible(
        objects,
        elapsed,
        world_bounds,
        maximum_events=maximum_events,
        maximum_additive_impulse=maximum_additive_residual,
    ):
        return None

    updated = objects
    remaining = elapsed
    relation_process_log_scale = objects.position.new_zeros(
        objects.batch_size,
        objects.max_objects,
    )
    collision_count = torch.zeros(
        objects.batch_size,
        dtype=torch.int64,
        device=objects.position.device,
    )
    pair_collision = torch.zeros(
        objects.batch_size,
        objects.max_objects,
        objects.max_objects,
        dtype=torch.bool,
        device=objects.position.device,
    )
    for _ in range(maximum_events):
        impact_time, selected, occurred = _earliest_pair_impact(
            updated,
            remaining,
            common_drag,
            collision_speed_epsilon=collision_speed_epsilon,
        )
        step_time = torch.where(occurred, impact_time, remaining)
        updated = analytic._integrate_validated_dt(
            updated,
            gravity,
            step_time,
            residual_acceleration=torch.zeros_like(updated.position),
        )
        if bool(occurred.any()):
            impulsed, relation_evidence, collision_pairs = _apply_selected_pair_impulse(
                updated,
                global_code,
                selected,
                interactions,
                collision_speed_epsilon=collision_speed_epsilon,
                maximum_multiplier_residual=maximum_multiplier_residual,
                maximum_additive_residual=maximum_additive_residual,
            )
            row_mask = occurred.unsqueeze(-1)
            updated = replace(
                updated,
                velocity=torch.where(
                    row_mask.unsqueeze(-1),
                    impulsed.velocity,
                    updated.velocity,
                ),
            )
            relation_process_log_scale = relation_process_log_scale + torch.where(
                row_mask,
                relation_evidence,
                torch.zeros_like(relation_evidence),
            )
            collision_count = collision_count + occurred.to(torch.int64)
            pair_collision = pair_collision | collision_pairs
        remaining = torch.where(
            occurred,
            (remaining - step_time).clamp_min(0.0),
            torch.zeros_like(remaining),
        )
        if not bool((remaining > 8.0 * torch.finfo(remaining.dtype).eps).any()):
            return EventDrivenStateResult(
                objects=updated,
                relation_process_log_scale=relation_process_log_scale,
                collision_count=collision_count,
                pair_collision=pair_collision,
            )

    # A row that consumed its eighth impact exactly is complete. Any positive
    # remainder could contain an unsupported ninth event, so the whole batch
    # returns to the authoritative microstep path.
    if bool((remaining > 8.0 * torch.finfo(remaining.dtype).eps).any()):
        return None
    return EventDrivenStateResult(
        objects=updated,
        relation_process_log_scale=relation_process_log_scale,
        collision_count=collision_count,
        pair_collision=pair_collision,
    )


__all__ = ["EventDrivenStateResult", "integrate_event_driven_state_only"]
