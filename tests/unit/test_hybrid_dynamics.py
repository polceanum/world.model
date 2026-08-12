from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.dynamics import (
    ContactPlane,
    DynamicsModel,
    EventModel,
    InteractionGraph,
    InteractionOutput,
    SphereContactResolver,
    TypedAttentionInteractionResidual,
)
from world_model.simulator.collisions import resolve_axis_aligned_boundaries


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


def _attention_residual(objects, belief) -> TypedAttentionInteractionResidual:
    return TypedAttentionInteractionResidual(
        modal_count=objects.modal_count,
        modal_dim=objects.modal_dim,
        geometry_dim=objects.geometry_dim,
        appearance_dim=objects.appearance_dim,
        residual_dynamics_dim=objects.residual_dynamics_dim,
        parameter_memory_dim=objects.parameter_memory.shape[-1],
        motion_mode_dim=objects.motion_mode_logits.shape[-1],
        global_code_dim=belief.global_code.shape[-1],
        width=128,
        heads=4,
        layers=4,
        feed_forward_width=512,
    )


def test_typed_attention_starts_as_exact_graph_identity_with_mac_scale_capacity() -> None:
    belief, objects = _two_objects()
    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=1.0)
    base = graph(objects, belief.global_code)
    attention = _attention_residual(objects, belief)

    output = attention(objects, belief, base)

    for item in fields(base):
        torch.testing.assert_close(
            getattr(output, item.name),
            getattr(base, item.name),
            rtol=0.0,
            atol=0.0,
        )
    parameter_count = sum(parameter.numel() for parameter in attention.parameters())
    assert 1_000_000 <= parameter_count <= 4_000_000


def test_typed_attention_is_permutation_equivariant_after_nonzero_decoding() -> None:
    torch.manual_seed(17)
    belief, objects = _two_objects()
    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=1.0)
    attention = _attention_residual(objects, belief)
    with torch.no_grad():
        attention.node_decoder.weight.normal_(std=0.01)
        attention.relation_decoder.weight.normal_(std=0.01)

    base = graph(objects, belief.global_code)
    output = attention(objects, belief, base)
    order = torch.tensor([1, 0])
    permuted = _permute_objects(objects, order)
    permuted_base = graph(permuted, belief.global_code)
    permuted_output = attention(permuted, belief, permuted_base)

    torch.testing.assert_close(
        permuted_output.residual_acceleration[:, order],
        output.residual_acceleration,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    permuted_pair_force = permuted_output.pair_force[:, order][:, :, order]
    torch.testing.assert_close(
        permuted_pair_force,
        output.pair_force,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        output.pair_force,
        -output.pair_force.transpose(1, 2),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_typed_attention_scene_context_is_live_when_global_code_is_zero() -> None:
    torch.manual_seed(23)
    belief, objects = _two_objects()
    assert torch.count_nonzero(belief.global_code) == 0
    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=1.0)
    attention = _attention_residual(objects, belief)
    with torch.no_grad():
        attention.node_decoder.weight.normal_(std=0.01)
        attention.relation_decoder.weight.normal_(std=0.01)

    output = attention(objects, belief, graph(objects, belief.global_code))
    loss = output.residual_acceleration.square().sum() + output.collision_logits.square().sum()
    loss.backward()

    gradient = attention.scene_projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_typed_attention_node_activity_tracks_emitted_active_acceleration() -> None:
    belief, objects = _two_objects()
    objects.active[0, 1] = False
    attention = _attention_residual(objects, belief)
    attention.train()
    attention.reset_output_gradient_diagnostics()
    target_fraction = torch.tensor([0.2, -0.4, 0.6])
    with torch.no_grad():
        attention.node_decoder.weight.zero_()
        attention.node_decoder.bias.copy_(torch.atanh(target_fraction))

    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=1.0)
    attention(objects, belief, graph(objects, belief.global_code))
    details = attention.node_activity_details()

    expected_axes = (attention.max_node_acceleration * target_fraction).square()
    torch.testing.assert_close(details["attention_node_activity_x"], expected_axes[0])
    torch.testing.assert_close(details["attention_node_activity_y"], expected_axes[1])
    torch.testing.assert_close(details["attention_node_activity_z"], expected_axes[2])
    torch.testing.assert_close(details["attention_node_activity"], expected_axes.mean())
    torch.testing.assert_close(details["attention_node_drift"], expected_axes.mean())
    torch.testing.assert_close(details["attention_node_variation"], torch.tensor(0.0))
    details["attention_node_activity"].backward()
    assert attention.node_decoder.bias.grad is not None
    assert torch.count_nonzero(attention.node_decoder.bias.grad) == 3

    attention.reset_output_gradient_diagnostics()
    assert attention.node_activity_details() == {}


def test_typed_attention_node_drift_does_not_penalize_balanced_variation() -> None:
    belief, objects = _two_objects()
    attention = _attention_residual(objects, belief)
    attention.train()
    attention.reset_output_gradient_diagnostics()
    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=1.0)
    target_fraction = torch.tensor([0.2, -0.4, 0.6])
    with torch.no_grad():
        attention.node_decoder.weight.zero_()
        attention.node_decoder.bias.copy_(torch.atanh(target_fraction))
    attention(objects, belief, graph(objects, belief.global_code))
    with torch.no_grad():
        attention.node_decoder.bias.copy_(torch.atanh(-target_fraction))
    attention(objects, belief, graph(objects, belief.global_code))

    details = attention.node_activity_details()
    expected_activity = (attention.max_node_acceleration * target_fraction).square().mean()
    torch.testing.assert_close(details["attention_node_activity"], expected_activity)
    torch.testing.assert_close(
        details["attention_node_drift"], torch.tensor(0.0), atol=1e-8, rtol=0
    )
    torch.testing.assert_close(details["attention_node_variation"], expected_activity)


def test_typed_attention_bounds_mixed_unit_scene_input_before_projection() -> None:
    belief, objects = _two_objects()
    attention = _attention_residual(objects, belief)
    extreme_camera = belief.camera.replace(intrinsics=belief.camera.intrinsics * 1_000.0)
    extreme_belief = replace(belief, camera=extreme_camera)

    raw = attention._scene_features(extreme_belief)
    normalized = attention.scene_input_norm(raw)

    assert raw.abs().max() >= 1_000.0
    assert torch.isfinite(normalized).all()
    expected_norm = normalized.new_tensor(normalized.shape[-1]).sqrt()
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized, dim=-1),
        expected_norm.expand(normalized.shape[0]),
        atol=1.0e-5,
        rtol=1.0e-5,
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


def test_structured_pair_jump_matches_simulator_restitution_combination() -> None:
    _, objects = _two_objects()
    objects.log_mass.zero_()
    objects.restitution_logit[0, 0] = torch.logit(torch.tensor(0.2))
    objects.restitution_logit[0, 1] = torch.logit(torch.tensor(0.8))

    result = SphereContactResolver()(objects)
    relative_after = result.objects.velocity[0, 1, 0] - result.objects.velocity[0, 0, 0]

    # The simulator's explicit material rule uses the less elastic member of
    # the pair, so an incoming relative speed of 2 m/s separates at 0.4 m/s.
    assert relative_after.item() == pytest.approx(0.4, abs=1.0e-5)


def test_contact_confidence_delays_uncertain_geometric_overlap() -> None:
    _, objects = _two_objects()
    uncertain = objects.clone()
    uncertain.fast_log_variance[..., :3] = -1.0

    delayed = SphereContactResolver(contact_confidence_sigma=0.5)(uncertain)
    confident = SphereContactResolver(contact_confidence_sigma=0.5)(objects)

    assert not delayed.pair_contact.any()
    assert not delayed.pair_collision.any()
    assert confident.pair_collision[0, 0, 1]


def test_no_pair_impulse_for_separating_spheres() -> None:
    _, objects = _two_objects()
    objects.velocity[0, 0, 0] = -1.0
    objects.velocity[0, 1, 0] = 1.0
    result = SphereContactResolver()(objects)
    assert not result.pair_collision.any()
    torch.testing.assert_close(result.objects.velocity, objects.velocity)


def test_model_and_simulator_share_low_speed_resting_boundary_constraint() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = torch.tensor([0.0, 0.1, 0.0])
    objects.velocity[0, 0] = torch.tensor([0.0, -0.05, 0.0])
    objects.geometry[0, 0, 0] = 0.1

    simulator = resolve_axis_aligned_boundaries(
        objects.position[0],
        objects.velocity[0],
        objects.radius[0],
        bounds=((-5.0, 5.0), (0.0, 3.0), (-5.0, 5.0)),
        restitution=objects.restitution[0],
        mass=objects.mass[0],
        friction=objects.friction[0],
        active=objects.active[0],
        collision_speed_epsilon=0.1,
    )
    model = SphereContactResolver()(objects)

    assert not simulator.collision.any()
    assert not model.ground_collision.any()
    torch.testing.assert_close(
        model.objects.velocity[0, 0],
        simulator.velocity[0],
    )
    assert model.objects.velocity[0, 0, 1] == 0


def test_slow_side_wall_contact_is_not_ground_or_sleep_and_preserves_tangent() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = torch.tensor([-0.9, 1.0, 0.0])
    objects.velocity[0, 0] = torch.tensor([-0.05, 0.0, 0.02])
    objects.geometry[0, 0, 0] = 0.1
    resolver = SphereContactResolver(
        planes=(
            ContactPlane(
                normal=(1.0, 0.0, 0.0),
                offset=-1.0,
                name="x_minimum",
                is_ground=False,
            ),
        ),
        boundary_collision_speed_epsilon=0.1,
    )

    result = EventModel(resolver, sleep_speed_threshold=0.1)(objects)

    assert result.contacts.boundary_contact[0, 0, 0]
    assert not result.contacts.boundary_collision[0, 0, 0]
    assert not result.contacts.ground_contact[0, 0]
    assert result.event_logits[0, 0, MotionMode.GROUND_CONTACT] < 0
    assert result.event_logits[0, 0, MotionMode.SLEEPING] < 0
    torch.testing.assert_close(
        result.objects.velocity[0, 0],
        torch.tensor([0.0, 0.0, 0.02]),
    )


def test_world_bounds_tag_only_floor_as_ground_support() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = torch.tensor([-0.9, 1.0, 0.0])
    objects.velocity[0, 0] = torch.tensor([-0.05, 0.0, 0.02])
    objects.geometry[0, 0, 0] = 0.1
    model = DynamicsModel.from_belief(
        belief,
        world_bounds=((-1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)),
        sleep_speed=0.1,
    )

    result = model.events(objects)

    assert model.events.resolver.plane_names == (
        "x_minimum",
        "x_maximum",
        "y_minimum",
        "y_maximum",
        "z_minimum",
        "z_maximum",
    )
    torch.testing.assert_close(
        model.events.resolver.ground_plane_mask,
        torch.tensor([False, False, True, False, False, False]),
    )
    assert result.contacts.boundary_contact[0, 0, 0]
    assert not result.contacts.ground_contact[0, 0]
    assert result.objects.mode[0, 0] != int(MotionMode.SLEEPING)
    assert result.objects.velocity[0, 0, 2] > 0


def test_floor_contact_does_not_instantly_create_sleep_without_persistent_evidence() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = torch.tensor([0.0, 0.1, 0.0])
    objects.velocity[0, 0] = torch.tensor([0.02, -0.05, 0.0])
    objects.geometry[0, 0, 0] = 0.1

    result = EventModel(sleep_speed_threshold=0.1)(objects)

    assert result.contacts.ground_contact[0, 0]
    assert result.objects.mode[0, 0] == int(MotionMode.GROUND_CONTACT)
    torch.testing.assert_close(
        result.objects.velocity[0, 0],
        torch.tensor([0.02, 0.0, 0.0]),
    )


def test_supported_sleep_persists_but_wall_only_sleep_wakes() -> None:
    belief = BeliefFactory(max_objects=2).create()
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.tensor([0, 1])
    objects.position[0, 0] = torch.tensor([0.0, 0.1, 0.0])
    objects.position[0, 1] = torch.tensor([-0.9, 1.0, 0.0])
    objects.geometry[..., 0] = 0.1
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., MotionMode.SLEEPING] = 5.0
    resolver = SphereContactResolver(
        planes=(
            ContactPlane(
                normal=(0.0, 1.0, 0.0),
                offset=0.0,
                name="floor",
                is_ground=True,
            ),
            ContactPlane(
                normal=(1.0, 0.0, 0.0),
                offset=-1.0,
                name="x_minimum",
                is_ground=False,
            ),
        ),
    )

    result = EventModel(resolver, sleep_speed_threshold=0.1)(objects)

    assert result.objects.mode[0, 0] == int(MotionMode.SLEEPING)
    assert result.objects.mode[0, 1] != int(MotionMode.SLEEPING)
    assert result.contacts.ground_contact[0, 0]
    assert not result.contacts.ground_contact[0, 1]


def test_wall_collision_is_still_an_interval_collision_not_ground_contact() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = torch.tensor([-0.9, 1.0, 0.0])
    objects.velocity[0, 0] = torch.tensor([-0.2, 0.0, 0.0])
    objects.geometry[0, 0, 0] = 0.1
    resolver = SphereContactResolver(
        planes=(
            ContactPlane(
                normal=(1.0, 0.0, 0.0),
                offset=-1.0,
                name="x_minimum",
                is_ground=False,
            ),
        ),
    )

    result = EventModel(resolver)(objects)

    assert result.contacts.boundary_collision[0, 0, 0]
    assert not result.contacts.ground_collision[0, 0]
    assert result.event_logits[0, 0, MotionMode.COLLISION] > 0


def test_rollout_keeps_wall_collision_occurrence_separate_from_ground() -> None:
    belief = BeliefFactory(max_objects=1).create()
    belief.objects.active[0, 0] = True
    belief.objects.object_id[0, 0] = 0
    belief.objects.position[0, 0] = torch.tensor([-0.89, 1.0, 0.0])
    belief.objects.velocity[0, 0] = torch.tensor([-1.0, 0.0, 0.0])
    belief.objects.geometry[0, 0, 0] = 0.1
    belief.objects.fast_log_variance.fill_(-12.0)
    model = DynamicsModel.from_belief(
        belief,
        world_bounds=((-1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)),
        max_substep=0.01,
        max_modal_acceleration=0.0,
        max_pair_force=0.0,
        max_node_acceleration=0.0,
    )

    result = model.predict_step(belief, 0.02)
    zero = model.predict_step(belief, 0.0)

    assert result.auxiliary["boundary_collision"][0, 0, 0]
    assert not result.auxiliary["ground_collision"][0, 0]
    assert result.event_logits[0, 0, MotionMode.COLLISION] > 0
    assert zero.auxiliary["boundary_collision"].shape == (1, 1, 6)
    assert not zero.auxiliary["boundary_collision"].any()


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
