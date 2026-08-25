"""One-step parity checks for simulator and belief-space contact solvers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from world_model.belief import BeliefFactory, ObjectBeliefTensor
from world_model.dynamics import ContactPlane, SphereContactResolver
from world_model.simulator.collisions import (
    resolve_axis_aligned_boundaries,
    resolve_sphere_sphere_collisions,
)

BOUNDS = ((-1.0, 1.0), (0.0, 2.0), (-1.0, 1.0))
SOLVER_ITERATIONS = 2
PENETRATION_FRACTION = 0.8
PENETRATION_SLOP = 1.0e-4
MAX_POSITION_CORRECTION = 0.05


@dataclass(frozen=True)
class _ReferenceContact:
    position: Tensor
    velocity: Tensor
    pair_contact: Tensor
    interval_pair_contact: Tensor
    pair_collision: Tensor
    pair_impulse: Tensor
    boundary_contact: Tensor
    interval_boundary_contact: Tensor
    boundary_collision: Tensor


def _objects(
    *,
    position: Tensor,
    velocity: Tensor,
    radius: Tensor,
    mass: Tensor,
    restitution: Tensor,
    friction: Tensor,
) -> ObjectBeliefTensor:
    count = position.shape[0]
    objects = BeliefFactory(max_objects=count).create().objects.clone()
    objects.active[0] = True
    objects.object_id[0] = torch.arange(count)
    objects.position[0] = position
    objects.velocity[0] = velocity
    objects.geometry[0, :, 0] = radius
    objects.log_mass[0, :, 0] = mass.log()
    objects.restitution_logit[0, :, 0] = torch.logit(restitution)
    objects.friction_logit[0, :, 0] = torch.logit(friction)
    objects.fast_log_variance.fill_(-20.0)
    return objects


def _planes() -> tuple[ContactPlane, ...]:
    return (
        ContactPlane((1.0, 0.0, 0.0), -1.0, "x_minimum", False),
        ContactPlane((-1.0, 0.0, 0.0), -1.0, "x_maximum", False),
        ContactPlane((0.0, 1.0, 0.0), 0.0, "y_minimum", True),
        ContactPlane((0.0, -1.0, 0.0), -2.0, "y_maximum", False),
        ContactPlane((0.0, 0.0, 1.0), -1.0, "z_minimum", False),
        ContactPlane((0.0, 0.0, -1.0), -1.0, "z_maximum", False),
    )


def _resolver(*, differentiable_contact_gradients_enabled: bool = False) -> SphereContactResolver:
    return SphereContactResolver(
        _planes(),
        contact_margin=0.0,
        boundary_contact_tolerance=1.0e-4,
        solver_iterations=SOLVER_ITERATIONS,
        penetration_fraction=PENETRATION_FRACTION,
        penetration_slop=PENETRATION_SLOP,
        max_position_correction=MAX_POSITION_CORRECTION,
        collision_speed_epsilon=1.0e-7,
        boundary_collision_speed_epsilon=0.1,
        contact_confidence_sigma=0.0,
        differentiable_contact_gradients_enabled=(differentiable_contact_gradients_enabled),
        differentiable_contact_gap_temperature=0.02,
        differentiable_contact_velocity_temperature=0.10,
    )


def _reference_contact(objects: ObjectBeliefTensor) -> _ReferenceContact:
    position = objects.position[0].clone()
    velocity = objects.velocity[0].clone()
    count = objects.max_objects
    pair_contact = torch.zeros(count, count, dtype=torch.bool)
    interval_pair_contact = torch.zeros_like(pair_contact)
    pair_collision = torch.zeros_like(pair_contact)
    pair_impulse = torch.zeros(count, count)
    boundary_contact = torch.zeros(count, 6, dtype=torch.bool)
    interval_boundary_contact = torch.zeros_like(boundary_contact)
    boundary_collision = torch.zeros_like(boundary_contact)

    for _ in range(SOLVER_ITERATIONS):
        boundary = resolve_axis_aligned_boundaries(
            position,
            velocity,
            objects.radius[0],
            objects.mass[0],
            objects.restitution[0],
            BOUNDS,
            friction=objects.friction[0],
            active=objects.active[0],
            collision_speed_epsilon=0.1,
        )
        pair = resolve_sphere_sphere_collisions(
            boundary.position,
            boundary.velocity,
            objects.radius[0],
            objects.mass[0],
            objects.restitution[0],
            friction=objects.friction[0],
            active=objects.active[0],
            position_correction=PENETRATION_FRACTION,
            penetration_slop=PENETRATION_SLOP,
            max_position_correction=MAX_POSITION_CORRECTION,
            velocity_epsilon=1.0e-7,
        )
        position = pair.position
        velocity = pair.velocity
        pair_contact = pair.contact
        interval_pair_contact |= pair.contact
        pair_collision |= pair.collision
        pair_impulse = torch.maximum(pair_impulse, pair.impulse_magnitude)
        boundary_contact = boundary.contact
        interval_boundary_contact |= boundary.contact
        boundary_collision |= boundary.collision

    distance = torch.cdist(position, position)
    radius = objects.radius[0, :, 0]
    active_pair = objects.active[0, :, None] & objects.active[0, None, :]
    non_diagonal = ~torch.eye(count, dtype=torch.bool)
    pair_contact = active_pair & non_diagonal & (distance < radius[:, None] + radius[None, :])
    bounds = torch.as_tensor(BOUNDS, dtype=position.dtype)
    boundary_contact_columns: list[Tensor] = []
    for axis, side, inward_sign in (
        (0, 0, 1.0),
        (0, 1, -1.0),
        (1, 0, 1.0),
        (1, 1, -1.0),
        (2, 0, 1.0),
        (2, 1, -1.0),
    ):
        signed_distance = (
            position[:, axis] - bounds[axis, side]
            if inward_sign > 0
            else bounds[axis, side] - position[:, axis]
        )
        boundary_contact_columns.append(objects.active[0] & (signed_distance <= radius + 1.0e-4))
    boundary_contact = torch.stack(boundary_contact_columns, dim=-1)

    return _ReferenceContact(
        position=position,
        velocity=velocity,
        pair_contact=pair_contact,
        interval_pair_contact=interval_pair_contact,
        pair_collision=pair_collision,
        pair_impulse=pair_impulse,
        boundary_contact=boundary_contact,
        interval_boundary_contact=interval_boundary_contact,
        boundary_collision=boundary_collision,
    )


def _assert_parity(objects: ObjectBeliefTensor) -> None:
    reference = _reference_contact(objects)
    model = _resolver()(objects)

    torch.testing.assert_close(model.objects.position[0], reference.position)
    torch.testing.assert_close(model.objects.velocity[0], reference.velocity)
    torch.testing.assert_close(model.pair_contact[0], reference.pair_contact)
    torch.testing.assert_close(
        model.interval_pair_contact[0],
        reference.interval_pair_contact,
    )
    torch.testing.assert_close(model.pair_collision[0], reference.pair_collision)
    torch.testing.assert_close(model.pair_impulse[0], reference.pair_impulse)
    torch.testing.assert_close(model.boundary_contact[0], reference.boundary_contact)
    torch.testing.assert_close(
        model.interval_boundary_contact[0],
        reference.interval_boundary_contact,
    )
    torch.testing.assert_close(model.boundary_collision[0], reference.boundary_collision)


def test_pair_contact_step_matches_simulator_material_and_projection_rules() -> None:
    objects = _objects(
        position=torch.tensor([[-0.18, 1.0, 0.0], [0.17, 1.0, 0.0]]),
        velocity=torch.tensor([[1.2, 0.0, 0.5], [-0.4, 0.0, -0.2]]),
        radius=torch.tensor([0.2, 0.2]),
        mass=torch.tensor([1.0, 2.5]),
        restitution=torch.tensor([0.25, 0.8]),
        # Geometric aggregation gives 0.12; the previous arithmetic rule gave
        # 0.20 and produced a materially different tangential jump.
        friction=torch.tensor([0.04, 0.36]),
    )

    _assert_parity(objects)


def test_boundary_contact_step_matches_simulator_corner_friction_order() -> None:
    objects = _objects(
        position=torch.tensor([[-0.86, 0.13, 0.0]]),
        velocity=torch.tensor([[-1.1, -0.8, 0.7]]),
        radius=torch.tensor([0.2]),
        mass=torch.tensor([1.7]),
        restitution=torch.tensor([0.55]),
        friction=torch.tensor([0.3]),
    )

    _assert_parity(objects)


def test_compound_wall_and_pair_contacts_match_two_solver_iterations() -> None:
    objects = _objects(
        position=torch.tensor(
            [
                [-0.86, 1.0, 0.0],
                [-0.52, 1.0, 0.02],
                [-0.15, 1.0, -0.01],
            ]
        ),
        velocity=torch.tensor(
            [
                [-1.0, 0.0, 0.25],
                [-0.35, 0.0, -0.15],
                [-0.8, 0.0, 0.1],
            ]
        ),
        radius=torch.tensor([0.2, 0.2, 0.2]),
        mass=torch.tensor([1.0, 1.8, 0.7]),
        restitution=torch.tensor([0.4, 0.65, 0.3]),
        friction=torch.tensor([0.08, 0.25, 0.45]),
    )

    _assert_parity(objects)


def test_randomized_three_sphere_contacts_match_reference_solver() -> None:
    generator = torch.Generator().manual_seed(20260802)

    for case_index in range(48):
        radius = 0.12 + 0.10 * torch.rand(3, generator=generator)
        if case_index % 3 == 0:
            first_x = -1.0 + radius[0] - 0.04 * torch.rand((), generator=generator)
        else:
            first_x = -0.65 + 0.50 * torch.rand((), generator=generator)
        pair_scale = 0.72 + 0.38 * torch.rand(2, generator=generator)
        x = torch.stack(
            (
                first_x,
                first_x + (radius[0] + radius[1]) * pair_scale[0],
                first_x
                + (radius[0] + radius[1]) * pair_scale[0]
                + (radius[1] + radius[2]) * pair_scale[1],
            )
        )
        base_y = 0.25 + 1.25 * torch.rand((), generator=generator)
        y = base_y + 0.04 * (torch.rand(3, generator=generator) - 0.5)
        z = 0.08 * (torch.rand(3, generator=generator) - 0.5)
        position = torch.stack((x, y, z), dim=-1)
        velocity = 3.0 * (torch.rand(3, 3, generator=generator) - 0.5)
        mass = 0.5 + 2.5 * torch.rand(3, generator=generator)
        restitution = 0.1 + 0.8 * torch.rand(3, generator=generator)
        friction = 0.01 + 0.59 * torch.rand(3, generator=generator)
        objects = _objects(
            position=position,
            velocity=velocity,
            radius=radius,
            mass=mass,
            restitution=restitution,
            friction=friction,
        )

        _assert_parity(objects)


def test_iterative_compound_contact_remains_differentiable() -> None:
    objects = _objects(
        position=torch.tensor([[-0.86, 1.0, 0.0], [-0.52, 1.0, 0.02]]),
        velocity=torch.tensor([[-1.0, 0.0, 0.25], [-0.35, 0.0, -0.15]]),
        radius=torch.tensor([0.2, 0.2]),
        mass=torch.tensor([1.0, 1.8]),
        restitution=torch.tensor([0.4, 0.65]),
        friction=torch.tensor([0.08, 0.25]),
    )
    objects.position.requires_grad_()
    objects.velocity.requires_grad_()

    result = _resolver()(objects)
    loss = result.objects.position.square().sum() + result.objects.velocity.square().sum()
    loss.backward()

    assert objects.position.grad is not None
    assert objects.velocity.grad is not None
    assert torch.isfinite(objects.position.grad).all()
    assert torch.isfinite(objects.velocity.grad).all()


def test_differentiable_contact_carrier_is_forward_bit_exact() -> None:
    objects = _objects(
        position=torch.tensor([[-0.86, 0.13, 0.0], [-0.54, 0.13, 0.01], [0.20, 1.0, 0.0]]),
        velocity=torch.tensor([[-1.1, -0.8, 0.7], [-0.35, -0.1, -0.15], [0.25, 0.0, 0.0]]),
        radius=torch.tensor([0.2, 0.2, 0.15]),
        mass=torch.tensor([1.7, 1.2, 0.8]),
        restitution=torch.tensor([0.55, 0.4, 0.7]),
        friction=torch.tensor([0.3, 0.15, 0.05]),
    )

    hard = _resolver()(objects)
    carried = _resolver(differentiable_contact_gradients_enabled=True)(objects)

    for name in (
        "position",
        "velocity",
        "motion_mode_logits",
        "fast_log_variance",
    ):
        torch.testing.assert_close(
            getattr(carried.objects, name),
            getattr(hard.objects, name),
            rtol=0.0,
            atol=0.0,
        )
    for name in (
        "pair_contact",
        "interval_pair_contact",
        "pair_collision",
        "boundary_contact",
        "interval_boundary_contact",
        "boundary_collision",
        "ground_contact",
        "interval_ground_contact",
        "ground_collision",
        "pair_impulse",
        "max_penetration",
        "mean_penetration",
        "action_reaction_residual",
    ):
        torch.testing.assert_close(
            getattr(carried, name),
            getattr(hard, name),
            rtol=0.0,
            atol=0.0,
        )


def test_differentiable_pair_carrier_reaches_material_before_hard_contact() -> None:
    objects = _objects(
        position=torch.tensor([[-0.205, 1.0, 0.0], [0.205, 1.0, 0.0]]),
        velocity=torch.tensor([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
        radius=torch.tensor([0.2, 0.2]),
        mass=torch.tensor([1.0, 1.0]),
        restitution=torch.tensor([0.4, 0.6]),
        friction=torch.tensor([0.0, 0.0]),
    )
    objects.position.requires_grad_()
    objects.velocity.requires_grad_()
    objects.geometry.requires_grad_()
    objects.restitution_logit.requires_grad_()

    result = _resolver(differentiable_contact_gradients_enabled=True)(objects)
    assert not result.pair_contact.any()
    assert not result.pair_collision.any()
    torch.testing.assert_close(result.objects.velocity, objects.velocity, rtol=0.0, atol=0.0)
    result.objects.velocity[0, 0, 0].backward()

    for name, gradient in (
        ("position", objects.position.grad),
        ("velocity", objects.velocity.grad),
        ("geometry", objects.geometry.grad),
        ("restitution", objects.restitution_logit.grad),
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all(), name
    assert objects.geometry.grad[..., 0].abs().sum() > 0.0
    assert objects.restitution_logit.grad.abs().sum() > 0.0


def test_differentiable_boundary_carrier_reaches_material_before_hard_contact() -> None:
    objects = _objects(
        position=torch.tensor([[-0.79, 1.0, 0.0]]),
        velocity=torch.tensor([[-0.8, 0.0, 0.0]]),
        radius=torch.tensor([0.2]),
        mass=torch.tensor([1.0]),
        restitution=torch.tensor([0.5]),
        friction=torch.tensor([0.0]),
    )
    objects.position.requires_grad_()
    objects.velocity.requires_grad_()
    objects.geometry.requires_grad_()
    objects.restitution_logit.requires_grad_()

    result = _resolver(differentiable_contact_gradients_enabled=True)(objects)
    assert not result.boundary_contact.any()
    assert not result.boundary_collision.any()
    torch.testing.assert_close(result.objects.velocity, objects.velocity, rtol=0.0, atol=0.0)

    result.objects.velocity[0, 0, 0].backward()

    for name, gradient in (
        ("position", objects.position.grad),
        ("velocity", objects.velocity.grad),
        ("geometry", objects.geometry.grad),
        ("restitution", objects.restitution_logit.grad),
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all(), name
    assert objects.geometry.grad[..., 0].abs().sum() > 0.0
    assert objects.restitution_logit.grad.abs().sum() > 0.0


def test_differentiable_contact_carrier_is_disabled_in_eval() -> None:
    objects = _objects(
        position=torch.tensor([[-0.205, 1.0, 0.0], [0.205, 1.0, 0.0]]),
        velocity=torch.tensor([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
        radius=torch.tensor([0.2, 0.2]),
        mass=torch.ones(2),
        restitution=torch.full((2,), 0.5),
        friction=torch.zeros(2),
    )
    objects.geometry.requires_grad_()
    resolver = _resolver(differentiable_contact_gradients_enabled=True).eval()

    result = resolver(objects)
    result.objects.velocity.square().sum().backward()

    assert objects.geometry.grad is not None
    torch.testing.assert_close(objects.geometry.grad, torch.zeros_like(objects.geometry.grad))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_differentiable_contact_carrier_has_finite_mps_backward() -> None:
    source = _objects(
        position=torch.tensor([[-0.205, 1.0, 0.0], [0.205, 1.0, 0.0]]),
        velocity=torch.tensor([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
        radius=torch.tensor([0.2, 0.2]),
        mass=torch.ones(2),
        restitution=torch.tensor([0.4, 0.6]),
        friction=torch.zeros(2),
    ).to("mps")
    objects = source.replace(
        position=source.position.detach().clone().requires_grad_(),
        velocity=source.velocity.detach().clone().requires_grad_(),
        geometry=source.geometry.detach().clone().requires_grad_(),
        restitution_logit=(source.restitution_logit.detach().clone().requires_grad_()),
    )
    resolver = _resolver(differentiable_contact_gradients_enabled=True).to("mps")

    result = resolver(objects)
    result.objects.velocity[0, 0, 0].backward()
    torch.mps.synchronize()

    for gradient in (
        objects.position.grad,
        objects.velocity.grad,
        objects.geometry.grad,
        objects.restitution_logit.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_interval_contact_does_not_become_stale_endpoint_mode() -> None:
    objects = _objects(
        position=torch.tensor([[-0.19, 1.0, 0.0], [0.19, 1.0, 0.0]]),
        velocity=torch.zeros(2, 3),
        radius=torch.tensor([0.2, 0.2]),
        mass=torch.ones(2),
        restitution=torch.full((2,), 0.5),
        friction=torch.zeros(2),
    )
    resolver = SphereContactResolver(
        _planes(),
        contact_margin=0.0,
        boundary_contact_tolerance=1.0e-4,
        solver_iterations=2,
        penetration_fraction=1.0,
        penetration_slop=0.0,
        max_position_correction=MAX_POSITION_CORRECTION,
        collision_speed_epsilon=1.0e-7,
        boundary_collision_speed_epsilon=0.1,
        contact_confidence_sigma=0.0,
    )

    result = resolver(objects)

    assert result.interval_pair_contact.any()
    assert not result.pair_contact.any()
