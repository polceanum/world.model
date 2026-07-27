from __future__ import annotations

from dataclasses import fields, replace

import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.dynamics import (
    EventModel,
    InteractionGraph,
    InteractionOutput,
    SphereContactResolver,
)


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


def _graph_output(
    objects,
    *,
    collision_logits: torch.Tensor,
    edge_mask: torch.Tensor,
) -> InteractionOutput:
    batch, count = objects.active.shape
    node_vectors = objects.position.new_zeros(batch, count, 3)
    pair_vectors = objects.position.new_zeros(batch, count, count, 3)
    pair_scalars = objects.position.new_zeros(batch, count, count)
    return InteractionOutput(
        residual_acceleration=node_vectors,
        pair_acceleration=node_vectors,
        node_acceleration=node_vectors,
        pair_force=pair_vectors,
        contact_logits=pair_scalars,
        collision_logits=collision_logits,
        impulse_multiplier_raw=pair_scalars,
        impulse_additive_raw=pair_scalars,
        edge_process_noise=pair_scalars,
        edge_mask=edge_mask,
        interaction_density=edge_mask.sum(dim=-1).to(objects.position.dtype),
    )


def _separated_objects(count: int = 3):
    belief = BeliefFactory(
        max_objects=count,
        residual_dynamics_dim=4,
        global_code_dim=3,
    ).create()
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.arange(count)
    objects.position[0, :, 0] = torch.arange(count, dtype=torch.float32) * 2.0
    objects.position[0, :, 1] = 1.0
    objects.geometry[..., 0] = 0.1
    return objects


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


def test_negative_valid_learned_edge_suppresses_collision_logit() -> None:
    _, objects = _two_objects()
    edge_mask = torch.tensor([[[False, True], [True, False]]])
    graph = _graph_output(
        objects,
        collision_logits=torch.tensor([[[0.0, -2.0], [-2.0, 0.0]]]),
        edge_mask=edge_mask,
    )

    baseline = EventModel()(objects)
    result = EventModel()(objects, graph)

    assert result.contacts.pair_collision.any(dim=-1).all()
    torch.testing.assert_close(
        result.event_logits[..., MotionMode.COLLISION],
        baseline.event_logits[..., MotionMode.COLLISION] - 2.0,
    )


def test_event_pooling_excludes_diagonals_and_nonedges() -> None:
    objects = _separated_objects()
    edge_mask = torch.tensor(
        [
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ]
        ]
    )
    graph = _graph_output(
        objects,
        collision_logits=torch.tensor(
            [
                [
                    [100.0, -2.0, 90.0],
                    [-2.0, 100.0, 80.0],
                    [70.0, 60.0, 100.0],
                ]
            ]
        ),
        edge_mask=edge_mask,
    )

    result = EventModel()(objects, graph)

    torch.testing.assert_close(
        result.event_logits[..., MotionMode.COLLISION],
        torch.tensor([[-6.0, -6.0, -4.0]]),
    )


def test_event_pooling_uses_finite_neutral_residual_without_neighbors() -> None:
    objects = _separated_objects()
    graph = _graph_output(
        objects,
        collision_logits=torch.full((1, 3, 3), float("nan")),
        edge_mask=torch.zeros(1, 3, 3, dtype=torch.bool),
    )

    baseline = EventModel()(objects)
    result = EventModel()(objects, graph)

    assert torch.isfinite(result.event_logits).all()
    torch.testing.assert_close(result.event_logits, baseline.event_logits)


def test_positive_valid_learned_edge_still_increases_collision_logit() -> None:
    _, objects = _two_objects()
    edge_mask = torch.tensor([[[False, True], [True, False]]])
    graph = _graph_output(
        objects,
        collision_logits=torch.tensor([[[0.0, 2.5], [2.5, 0.0]]]),
        edge_mask=edge_mask,
    )

    baseline = EventModel()(objects)
    result = EventModel()(objects, graph)

    torch.testing.assert_close(
        result.event_logits[..., MotionMode.COLLISION],
        baseline.event_logits[..., MotionMode.COLLISION] + 2.5,
    )
