from __future__ import annotations

import torch

from world_model.belief import (
    BeliefFactory,
    fast_packing_map,
    pack_fast_state,
    pack_slow_state,
    slow_packing_map,
    unpack_fast_state,
    unpack_slow_state,
)
from world_model.dynamics import normalize_quaternion


def test_fast_pack_unpack_roundtrip_and_map() -> None:
    objects = (
        BeliefFactory(
            max_objects=3,
            modal_count=4,
            modal_dim=3,
        )
        .create(batch_size=2)
        .objects
    )
    objects.position.normal_()
    objects.velocity.normal_()
    objects.orientation.copy_(normalize_quaternion(torch.randn_like(objects.orientation)))
    objects.angular_velocity.normal_()
    objects.modal_state.normal_()

    packed = pack_fast_state(objects)
    packing = fast_packing_map(objects)
    restored = unpack_fast_state(packed.clone(), objects.clone())

    assert packed.shape == (2, 3, objects.fast_state_dim)
    assert packing.size == objects.fast_state_dim
    assert packing["position"] == slice(0, 3)
    torch.testing.assert_close(pack_fast_state(restored), packed)
    restored.validate()


def test_slow_pack_unpack_roundtrip_and_map() -> None:
    objects = (
        BeliefFactory(
            max_objects=2,
            geometry_dim=5,
            appearance_dim=7,
            residual_dynamics_dim=4,
        )
        .create()
        .objects
    )
    objects.log_mass.normal_()
    objects.restitution_logit.normal_()
    objects.log_drag.normal_()
    objects.friction_logit.normal_()
    objects.geometry.uniform_(0.1, 1.0)
    objects.appearance.normal_()
    objects.residual_dynamics.normal_()

    packed = pack_slow_state(objects)
    packing = slow_packing_map(objects)
    restored = unpack_slow_state(packed.clone(), objects.clone())

    assert packed.shape[-1] == objects.slow_state_dim == packing.size
    assert packing["log_mass"] == slice(0, 1)
    assert packing["residual_dynamics"].stop == objects.slow_state_dim
    torch.testing.assert_close(pack_slow_state(restored), packed)
    restored.validate()


def test_unpack_rejects_wrong_dimension() -> None:
    objects = BeliefFactory(max_objects=2).create().objects
    wrong = torch.zeros(1, 2, objects.fast_state_dim + 1)

    try:
        unpack_fast_state(wrong, objects)
    except ValueError as error:
        assert "packed fast state" in str(error)
    else:
        raise AssertionError("wrong packed shape was accepted")
