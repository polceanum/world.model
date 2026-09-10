from __future__ import annotations

import pytest
import torch

from world_model.belief import BeliefFactory, RigidGeometryCodec, RigidPrimitive


def test_legacy_one_component_geometry_decodes_exactly_as_sphere() -> None:
    belief = BeliefFactory(max_objects=2, geometry_dim=1, initial_radius=0.21).create(batch_size=1)

    assert torch.equal(belief.objects.radius, belief.objects.geometry[..., :1])
    assert torch.equal(
        belief.objects.geometry_primitive,
        torch.full((1, 2), int(RigidPrimitive.SPHERE), dtype=torch.int64),
    )
    torch.testing.assert_close(
        belief.objects.geometry_half_extents,
        torch.full((1, 2, 3), 0.21),
    )


def test_box_encoding_retains_legacy_bounding_radius_and_exact_extents() -> None:
    half_extents = torch.tensor([[[0.2, 0.3, 0.4]]], dtype=torch.float64)

    geometry = RigidGeometryCodec.encode_box(half_extents, geometry_dim=6)

    torch.testing.assert_close(
        RigidGeometryCodec.bounding_radius(geometry)[..., 0],
        torch.linalg.vector_norm(half_extents, dim=-1),
    )
    assert RigidGeometryCodec.primitive(geometry).item() == RigidPrimitive.BOX
    assert torch.equal(RigidGeometryCodec.half_extents(geometry), half_extents)


def test_box_encoding_fails_closed_when_the_state_has_no_shape_capacity() -> None:
    with pytest.raises(ValueError, match="geometry_dim >= 5"):
        RigidGeometryCodec.encode_box(torch.ones(1, 3), geometry_dim=4)


def test_unknown_or_fractional_geometry_tags_are_rejected() -> None:
    for tag in (0.5, 9.0):
        geometry = torch.tensor([[0.2, tag, 0.1, 0.1, 0.1]])
        with pytest.raises(ValueError, match="tag|unsupported"):
            RigidGeometryCodec.primitive(geometry)
