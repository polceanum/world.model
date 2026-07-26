from __future__ import annotations

from dataclasses import fields, replace

import torch

from world_model.belief import BeliefFactory
from world_model.dynamics import InteractionGraph, SphereContactResolver


def _two_objects():
    belief = BeliefFactory(
        max_objects=2,
        residual_dynamics_dim=4,
        global_code_dim=3,
    ).create()
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.tensor([10, 11])
    objects.position[0, 0] = torch.tensor([-0.09, 1.0, 0.0])
    objects.position[0, 1] = torch.tensor([0.09, 1.0, 0.0])
    objects.velocity[0, 0, 0] = 1.0
    objects.velocity[0, 1, 0] = -1.0
    objects.geometry[..., 0] = 0.1
    objects.log_mass[0, 0] = 0.0
    objects.log_mass[0, 1] = torch.log(torch.tensor(2.0))
    objects.fast_log_variance.fill_(-8.0)
    return belief, objects


def _permute_objects(objects, order: torch.Tensor):
    updates = {}
    for item in fields(objects):
        value = getattr(objects, item.name)
        updates[item.name] = value[:, order]
    return replace(objects, **updates)


def test_interaction_graph_is_permutation_equivariant_and_pair_force_is_antisymmetric() -> None:
    belief, objects = _two_objects()
    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=1.0)
    final = graph.edge_network.output
    with torch.no_grad():
        final.bias[2] = 0.4
    output = graph(objects, belief.global_code)
    order = torch.tensor([1, 0])
    permuted = _permute_objects(objects, order)
    permuted_output = graph(permuted, belief.global_code)

    torch.testing.assert_close(
        output.pair_force,
        -output.pair_force.transpose(1, 2),
    )
    torch.testing.assert_close(output.pair_force.sum(dim=(1, 2)), torch.zeros(1, 3))
    torch.testing.assert_close(
        permuted_output.residual_acceleration[:, order],
        output.residual_acceleration,
        atol=1e-6,
        rtol=1e-6,
    )


def test_structured_pair_jump_conserves_momentum_and_separates_spheres() -> None:
    _, objects = _two_objects()
    resolver = SphereContactResolver()
    before_momentum = (objects.mass * objects.velocity).sum(dim=1)
    result = resolver(objects)
    after_momentum = (result.objects.mass * result.objects.velocity).sum(dim=1)

    assert result.pair_collision[0, 0, 1]
    torch.testing.assert_close(after_momentum, before_momentum, atol=1e-6, rtol=1e-6)
    assert result.action_reaction_residual.max() < 1e-7
    relative_after = result.objects.velocity[0, 1] - result.objects.velocity[0, 0]
    normal = torch.tensor([1.0, 0.0, 0.0])
    assert torch.dot(relative_after, normal) > 0
    distance = torch.linalg.vector_norm(
        result.objects.position[0, 1] - result.objects.position[0, 0]
    )
    assert distance > torch.tensor(0.18)


def test_no_pair_impulse_for_separating_spheres() -> None:
    _, objects = _two_objects()
    objects.velocity[0, 0, 0] = -1.0
    objects.velocity[0, 1, 0] = 1.0
    result = SphereContactResolver()(objects)
    assert not result.pair_collision.any()
    torch.testing.assert_close(result.objects.velocity, objects.velocity)
