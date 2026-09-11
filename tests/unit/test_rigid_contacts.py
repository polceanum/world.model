from __future__ import annotations

import math

import torch

from world_model.belief import BeliefFactory, RigidGeometryCodec
from world_model.dynamics import ContactPlane, SphereContactResolver


def _objects(*, first_box: bool, second_box: bool):
    belief = BeliefFactory(max_objects=2, geometry_dim=5).create(
        batch_size=1,
        dtype=torch.float64,
        gravity=(0.0, 0.0, 0.0),
    )
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.tensor([[4, 9]])
    objects.log_mass.zero_()
    objects.restitution_logit.fill_(20.0)
    objects.friction_logit.fill_(-20.0)
    objects.fast_log_variance.fill_(-30.0)
    sphere = RigidGeometryCodec.encode_sphere(
        objects.position.new_tensor([[[0.2], [0.2]]]),
        geometry_dim=5,
    )
    box = RigidGeometryCodec.encode_box(
        objects.position.new_tensor([[[0.4, 0.25, 0.3], [0.4, 0.25, 0.3]]]),
        geometry_dim=5,
    )
    objects.geometry.copy_(sphere)
    if first_box:
        objects.geometry[:, 0] = box[:, 0]
    if second_box:
        objects.geometry[:, 1] = box[:, 1]
    return objects


def _resolver() -> SphereContactResolver:
    return SphereContactResolver(
        (ContactPlane((0.0, 1.0, 0.0), offset=-10.0, name="remote"),),
        solver_iterations=1,
        penetration_fraction=1.0,
        penetration_slop=0.0,
        max_position_correction=1.0,
    )


def test_axis_aligned_box_collision_uses_face_extent_and_conserves_momentum() -> None:
    objects = _objects(first_box=True, second_box=True)
    objects.position[:] = objects.position.new_tensor([[[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]]])
    objects.velocity[:] = objects.velocity.new_tensor([[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]])

    result = _resolver()(objects)

    assert result.pair_collision[0, 0, 1]
    torch.testing.assert_close(
        result.objects.velocity[0, :, 0],
        torch.tensor([-0.5, 0.5], dtype=torch.float64),
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    torch.testing.assert_close(
        result.objects.velocity.sum(dim=1),
        objects.velocity.sum(dim=1),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert float(result.action_reaction_residual.max()) <= 1.0e-12


def test_sphere_box_collision_uses_closest_box_point() -> None:
    objects = _objects(first_box=False, second_box=True)
    objects.position[:] = objects.position.new_tensor([[[-0.55, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    objects.velocity[:] = objects.velocity.new_tensor([[[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    result = _resolver()(objects)

    assert result.pair_collision[0, 0, 1]
    torch.testing.assert_close(
        result.objects.velocity[0, :, 0],
        torch.tensor([0.0, 0.4], dtype=torch.float64),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_rotated_box_collision_uses_sat_axis_and_keeps_gradients_finite() -> None:
    objects = _objects(first_box=True, second_box=True)
    objects.position[:] = objects.position.new_tensor([[[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    objects.velocity[:] = objects.velocity.new_tensor([[[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]]])
    half_angle = math.pi / 8.0
    objects.orientation[:, 1] = objects.orientation.new_tensor(
        [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
    )
    objects.position.requires_grad_(True)
    objects.velocity.requires_grad_(True)
    objects.orientation.requires_grad_(True)

    result = _resolver()(objects)
    loss = result.objects.position.square().sum() + result.objects.velocity.square().sum()
    loss.backward()

    assert result.pair_collision[0, 0, 1]
    for value in (objects.position.grad, objects.velocity.grad, objects.orientation.grad):
        assert value is not None
        assert torch.isfinite(value).all()
    torch.testing.assert_close(
        result.objects.velocity.sum(dim=1),
        objects.velocity.sum(dim=1),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_rotated_box_plane_support_uses_orientation() -> None:
    objects = _objects(first_box=True, second_box=False)
    objects.active[:, 1] = False
    objects.object_id[:, 1] = -1
    objects.position[0, 0] = objects.position.new_tensor([0.0, 0.5, 0.0])
    objects.geometry[0, 0] = RigidGeometryCodec.encode_box(
        objects.position.new_tensor([[0.6, 0.2, 0.2]]),
        geometry_dim=5,
    )[0]
    half_angle = 0.25 * math.pi
    objects.orientation[0, 0] = objects.position.new_tensor(
        [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
    )
    resolver = SphereContactResolver(
        (ContactPlane((0.0, 1.0, 0.0), offset=0.0, name="floor"),),
        solver_iterations=1,
    )

    result = resolver(objects)

    assert result.boundary_contact[0, 0, 0]
    torch.testing.assert_close(
        result.objects.position[0, 0, 1],
        torch.tensor(0.6, dtype=torch.float64),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_wide_sphere_geometry_retains_legacy_contact_result_exactly() -> None:
    legacy_belief = BeliefFactory(max_objects=2, geometry_dim=1).create(
        batch_size=1,
        dtype=torch.float64,
        gravity=(0.0, 0.0, 0.0),
    )
    legacy = legacy_belief.objects.clone()
    legacy.active[:] = True
    legacy.object_id[:] = torch.tensor([[4, 9]])
    legacy.position[:] = legacy.position.new_tensor([[[-0.19, 0.0, 0.0], [0.19, 0.0, 0.0]]])
    legacy.velocity[:] = legacy.velocity.new_tensor([[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]])
    legacy.geometry[..., 0] = 0.2
    legacy.log_mass.zero_()
    legacy.restitution_logit.fill_(20.0)
    legacy.friction_logit.fill_(-20.0)
    legacy.fast_log_variance.fill_(-30.0)
    wide = _objects(first_box=False, second_box=False)
    wide.position.copy_(legacy.position)
    wide.velocity.copy_(legacy.velocity)
    wide.geometry[..., 0] = legacy.geometry[..., 0]
    resolver = _resolver()

    expected = resolver(legacy)
    actual = resolver(wide)

    assert torch.equal(actual.objects.position, expected.objects.position)
    assert torch.equal(actual.objects.velocity, expected.objects.velocity)
    assert torch.equal(actual.pair_contact, expected.pair_contact)
    assert torch.equal(actual.pair_collision, expected.pair_collision)
