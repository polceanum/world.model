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
from world_model.training.losses import balanced_binary_cross_entropy


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


def _three_attention_objects():
    belief = BeliefFactory(
        max_objects=3,
        residual_dynamics_dim=4,
        global_code_dim=3,
    ).create()
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.tensor([10, 11, 12])
    objects.position[0] = torch.tensor(
        [
            [-0.20, 1.00, -0.05],
            [0.05, 1.08, 0.12],
            [0.24, 0.92, -0.10],
        ]
    )
    objects.velocity[0] = torch.tensor(
        [
            [0.30, -0.10, 0.05],
            [-0.20, 0.04, -0.08],
            [0.10, 0.12, 0.03],
        ]
    )
    objects.geometry[..., 0] = 0.1
    objects.appearance[0, 0, 0] = 0.25
    objects.appearance[0, 1, 0] = -0.50
    objects.appearance[0, 2, 0] = 0.75
    objects.fast_log_variance.fill_(-8.0)
    return belief, objects


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


def _attention_residual(
    objects,
    belief,
    *,
    relation_endpoint_binding_enabled: bool = True,
) -> TypedAttentionInteractionResidual:
    return TypedAttentionInteractionResidual(
        modal_count=objects.modal_count,
        modal_dim=objects.modal_dim,
        geometry_dim=objects.geometry_dim,
        appearance_dim=objects.appearance_dim,
        residual_dynamics_dim=objects.residual_dynamics_dim,
        parameter_memory_dim=objects.parameter_memory.shape[-1],
        motion_mode_dim=objects.motion_mode_logits.shape[-1],
        global_code_dim=belief.global_code.shape[-1],
        relation_endpoint_binding_enabled=relation_endpoint_binding_enabled,
        width=128,
        heads=4,
        layers=4,
        feed_forward_width=512,
    )


def test_typed_attention_disabled_relation_binding_is_exact_legacy_identity() -> None:
    torch.manual_seed(13)
    belief, objects = _three_attention_objects()
    attention = _attention_residual(
        objects,
        belief,
        relation_endpoint_binding_enabled=False,
    )
    pair_indices = torch.triu_indices(3, 3, offset=1)
    pair_i, pair_j = pair_indices[0], pair_indices[1]
    relation_features, _, _, _ = attention._relation_features(objects)
    entity_tokens = attention.entity_projection(attention._entity_features(objects))
    legacy_relation_tokens = attention.relation_projection(relation_features[:, pair_i, pair_j])

    bound = attention._bind_relation_endpoints(
        legacy_relation_tokens,
        entity_tokens,
        pair_i,
        pair_j,
    )

    torch.testing.assert_close(bound, legacy_relation_tokens, rtol=0.0, atol=0.0)


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


def test_typed_attention_relation_tokens_are_sensitive_only_to_their_endpoints() -> None:
    torch.manual_seed(19)
    belief, objects = _three_attention_objects()
    attention = _attention_residual(objects, belief)
    pair_indices = torch.triu_indices(3, 3, offset=1)
    pair_i, pair_j = pair_indices[0], pair_indices[1]

    relation_features, _, _, _ = attention._relation_features(objects)
    entity_tokens = attention.entity_projection(attention._entity_features(objects))
    relation_tokens = attention.relation_projection(relation_features[:, pair_i, pair_j])
    bound = attention._bind_relation_endpoints(
        relation_tokens,
        entity_tokens,
        pair_i,
        pair_j,
    )

    changed = objects.clone()
    changed.appearance[0, 0, 0] += 1.0
    changed_relation_features, _, _, _ = attention._relation_features(changed)
    changed_entity_tokens = attention.entity_projection(attention._entity_features(changed))
    changed_relation_tokens = attention.relation_projection(
        changed_relation_features[:, pair_i, pair_j]
    )
    changed_bound = attention._bind_relation_endpoints(
        changed_relation_tokens,
        changed_entity_tokens,
        pair_i,
        pair_j,
    )

    # Appearance is absent from geometric relation features, so only explicit
    # endpoint incidence can make the two relations touching slot zero move.
    torch.testing.assert_close(changed_relation_tokens, relation_tokens)
    assert torch.count_nonzero(changed_bound[:, :2] - bound[:, :2]) > 0
    torch.testing.assert_close(changed_bound[:, 2], bound[:, 2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        attention._bind_relation_endpoints(
            relation_tokens,
            entity_tokens,
            pair_j,
            pair_i,
        ),
        bound,
        rtol=0.0,
        atol=0.0,
    )


def test_typed_attention_three_slot_relation_binding_is_permutation_equivariant() -> None:
    torch.manual_seed(29)
    belief, objects = _three_attention_objects()
    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=2.0)
    attention = _attention_residual(objects, belief)
    with torch.no_grad():
        attention.node_decoder.weight.normal_(std=0.01)
        attention.relation_decoder.weight.normal_(std=0.01)

    output = attention(objects, belief, graph(objects, belief.global_code))
    order = torch.tensor([2, 0, 1])
    inverse = torch.argsort(order)
    permuted = _permute_objects(objects, order)
    permuted_output = attention(
        permuted,
        belief,
        graph(permuted, belief.global_code),
    )

    for name in (
        "residual_acceleration",
        "pair_acceleration",
        "node_acceleration",
        "interaction_density",
    ):
        torch.testing.assert_close(
            getattr(permuted_output, name)[:, inverse],
            getattr(output, name),
            atol=3.0e-6,
            rtol=3.0e-6,
        )
    for name in (
        "pair_force",
        "contact_logits",
        "collision_logits",
        "impulse_multiplier_raw",
        "impulse_additive_raw",
        "edge_process_noise",
        "edge_mask",
    ):
        permuted_value = getattr(permuted_output, name)[:, inverse][:, :, inverse]
        torch.testing.assert_close(
            permuted_value,
            getattr(output, name),
            atol=3.0e-6,
            rtol=3.0e-6,
        )
    torch.testing.assert_close(
        output.pair_force,
        -output.pair_force.transpose(1, 2),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_typed_attention_masks_inactive_endpoint_context() -> None:
    torch.manual_seed(31)
    belief, objects = _three_attention_objects()
    objects.active[0, 2] = False
    objects.object_id[0, 2] = -1
    graph = InteractionGraph(4, 3, hidden_dim=16, interaction_radius=2.0)
    attention = _attention_residual(objects, belief)
    with torch.no_grad():
        attention.node_decoder.weight.normal_(std=0.01)
        attention.relation_decoder.weight.normal_(std=0.01)

    base = graph(objects, belief.global_code)
    output = attention(objects, belief, base)
    changed = objects.clone()
    changed.position[0, 2] = torch.tensor([50.0, -40.0, 30.0])
    changed.velocity[0, 2] = torch.tensor([-20.0, 10.0, 15.0])
    changed.appearance[0, 2].fill_(100.0)
    changed.residual_dynamics[0, 2].fill_(-100.0)
    changed_base = graph(changed, belief.global_code)
    changed_output = attention(changed, belief, changed_base)

    # The active nodes and their only valid relation cannot attend to the
    # inactive entity or either invalid relation token.
    torch.testing.assert_close(
        changed_output.residual_acceleration[:, :2],
        output.residual_acceleration[:, :2],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        changed_output.pair_force[:, :2, :2],
        output.pair_force[:, :2, :2],
        rtol=0.0,
        atol=0.0,
    )
    invalid_pair = ~base.edge_mask
    torch.testing.assert_close(
        (output.pair_force - base.pair_force)[invalid_pair],
        torch.zeros_like(output.pair_force[invalid_pair]),
        rtol=0.0,
        atol=0.0,
    )
    for name in (
        "contact_logits",
        "collision_logits",
        "impulse_multiplier_raw",
        "impulse_additive_raw",
        "edge_process_noise",
    ):
        residual = getattr(output, name) - getattr(base, name)
        torch.testing.assert_close(
            residual[invalid_pair],
            torch.zeros_like(residual[invalid_pair]),
            rtol=0.0,
            atol=0.0,
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


def test_zero_tangent_noncollision_boundary_stays_finite() -> None:
    """A resting boundary has no friction direction but remains a valid state."""

    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = torch.tensor([0.0, 0.1, 0.0])
    objects.velocity[0, 0] = torch.tensor([0.0, -0.05, 0.0])
    objects.geometry[0, 0, 0] = 0.1

    result = SphereContactResolver()(objects)

    assert torch.isfinite(result.objects.position).all()
    assert torch.isfinite(result.objects.velocity).all()
    assert result.ground_contact[0, 0]
    assert not result.ground_collision[0, 0]
    torch.testing.assert_close(result.objects.velocity[0, 0], torch.zeros(3))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_mps_zero_tangent_noncollision_boundary_stays_finite() -> None:
    """Keep the MPS zero-tangent contact regression covered when available."""

    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 0
    objects.position[0, 0] = torch.tensor([0.0, 0.1, 0.0])
    objects.velocity[0, 0] = torch.tensor([0.0, -0.05, 0.0])
    objects.geometry[0, 0, 0] = 0.1

    result = SphereContactResolver().to("mps")(objects.to("mps"))

    assert torch.isfinite(result.objects.position).all()
    assert torch.isfinite(result.objects.velocity).all()
    torch.testing.assert_close(
        result.objects.velocity[0, 0].cpu(),
        torch.zeros(3),
    )


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


def test_smooth_event_conjunction_matches_stable_cpu_logaddexp_at_extreme_logits() -> None:
    first = torch.tensor(
        [-1.0e5, -500.0, -100.0, -90.0, -1.0, 0.0, 1.0, 90.0, 100.0, 1.0e5],
        requires_grad=True,
    )
    second = torch.tensor(
        [1.0e5, 100.0, -90.0, -100.0, 1.0, 0.0, -1.0, 100.0, 90.0, -1.0e5],
        requires_grad=True,
    )

    actual = EventModel._smooth_conjunction(first, second)
    expected = -torch.logaddexp(-first.detach(), -second.detach())

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_smooth_event_hazard_uses_scale_and_learned_pair_logit_coherently() -> None:
    _, objects = _two_objects()
    # Keep a small positive surface gap, so the hard resolver must not jump.
    objects.position[0, 0, 0] = -0.11
    objects.position[0, 1, 0] = 0.11
    edge_mask = torch.tensor([[[False, True], [True, False]]])
    learned_collision = torch.tensor([[[0.0, 1.25], [1.25, 0.0]]])
    graph = _graph_output(
        objects,
        collision_logits=learned_collision,
        edge_mask=edge_mask,
    )

    narrow = EventModel(
        smooth_hazard_enabled=True,
        contact_logit_scale=0.02,
    )(objects, graph)
    broad = EventModel(
        smooth_hazard_enabled=True,
        contact_logit_scale=0.10,
    )(objects, graph)

    assert not narrow.contacts.pair_collision.any()
    torch.testing.assert_close(narrow.objects.velocity, objects.velocity)
    # The configured gap scale is operational rather than a dead constructor
    # argument, and a bounded +1.25 learned residual can cross the decision
    # threshold for a near missed analytic event.
    assert narrow.pair_event_logits[0, 0, 1, 0] < broad.pair_event_logits[0, 0, 1, 0]
    assert narrow.pair_event_logits[0, 0, 1, 1] > 0.0
    torch.testing.assert_close(
        narrow.event_logits[0, :, MotionMode.COLLISION],
        narrow.pair_event_logits[0, :, :, 1].amax(dim=-1),
    )


def test_smooth_event_hazard_self_pair_variance_has_finite_gradient() -> None:
    _, source = _two_objects()
    source.position[0, 0, 0] = -0.11
    source.position[0, 1, 0] = 0.11
    position = source.position.detach().clone().requires_grad_()
    velocity = source.velocity.detach().clone().requires_grad_()
    fast_log_variance = source.fast_log_variance.detach().clone().requires_grad_()
    objects = replace(
        source,
        position=position,
        velocity=velocity,
        fast_log_variance=fast_log_variance,
    )

    result = EventModel(smooth_hazard_enabled=True)(objects)
    score = result.pair_event_logits[:, 0, 1, 1]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        score,
        torch.ones_like(score),
    )
    loss.backward()

    for gradient in (position.grad, velocity.grad, fast_log_variance.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    assert position.grad.abs().sum() > 0.0
    assert velocity.grad.abs().sum() > 0.0


def test_smooth_positive_collision_loss_backpropagates_through_recursive_rollout() -> None:
    belief, source = _two_objects()
    source.position[0, 0, 0] = -0.12
    source.position[0, 1, 0] = 0.12
    position = source.position.detach().clone().requires_grad_()
    velocity = source.velocity.detach().clone().requires_grad_()
    fast_log_variance = source.fast_log_variance.detach().clone().requires_grad_()
    belief = replace(
        belief,
        objects=replace(
            source,
            position=position,
            velocity=velocity,
            fast_log_variance=fast_log_variance,
        ),
    )
    model = DynamicsModel.from_belief(
        belief,
        smooth_event_hazard_enabled=True,
        max_substep=0.01,
        max_modal_acceleration=0.0,
        max_pair_force=0.0,
        max_node_acceleration=0.0,
    )

    trajectory = model.rollout(
        belief,
        [0.01, 0.02, 0.03],
        auxiliary_names=("pair_event_logits",),
    )
    score = trajectory.auxiliary["pair_event_logits"][:, :, 0, 1, 1]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        score,
        torch.ones_like(score),
    )
    loss.backward()

    for gradient in (position.grad, velocity.grad, fast_log_variance.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
    collision_bias_gradient = model.interactions.edge_network.output.bias.grad
    assert collision_bias_gradient is not None
    assert collision_bias_gradient[1].abs() > 0.0


def test_smooth_event_hazard_keeps_analytic_jump_fail_safe_and_trainable() -> None:
    _, objects = _two_objects()
    edge_mask = torch.tensor([[[False, True], [True, False]]])
    learned_collision = torch.tensor(
        [[[0.0, -20.0], [-20.0, 0.0]]],
        requires_grad=True,
    )
    graph = _graph_output(
        objects,
        collision_logits=learned_collision,
        edge_mask=edge_mask,
    )

    result = EventModel(
        smooth_hazard_enabled=True,
        resolved_event_logit_floor=2.0,
    )(objects, graph)

    # A badly calibrated learned hazard cannot suppress the structured jump or
    # make an analytically resolved event fall below the runtime threshold.
    assert result.contacts.pair_collision[0, 0, 1]
    assert result.pair_event_logits[0, 0, 1, 1].item() == pytest.approx(2.0)
    assert result.event_logits[0, 0, MotionMode.COLLISION].item() == pytest.approx(2.0)
    assert result.objects.velocity[0, 0, 0] < 0.0
    assert result.objects.velocity[0, 1, 0] > 0.0

    loss = balanced_binary_cross_entropy(
        result.event_logits[..., MotionMode.COLLISION],
        torch.ones_like(result.event_logits[..., MotionMode.COLLISION]),
        torch.ones_like(result.event_logits[..., MotionMode.COLLISION], dtype=torch.bool),
    )
    loss.backward()

    assert learned_collision.grad is not None
    assert torch.isfinite(learned_collision.grad).all()
    assert learned_collision.grad[0, 0, 1].abs() > 0.0
    assert learned_collision.grad[0, 1, 0].abs() > 0.0


def test_smooth_pair_and_boundary_event_logits_survive_interval_rollout() -> None:
    belief, _ = _two_objects()
    belief.objects.active[:] = True
    belief.objects.object_id[0] = torch.tensor([10, 11])
    belief.objects.position[0, 0] = torch.tensor([-0.11, 1.0, 0.0])
    belief.objects.position[0, 1] = torch.tensor([0.11, 1.0, 0.0])
    belief.objects.velocity[0, 0, 0] = 1.0
    belief.objects.velocity[0, 1, 0] = -1.0
    belief.objects.geometry[..., 0] = 0.1
    belief.objects.fast_log_variance.fill_(-8.0)
    model = DynamicsModel.from_belief(
        belief,
        smooth_event_hazard_enabled=True,
        max_substep=0.01,
        max_modal_acceleration=0.0,
        max_pair_force=0.0,
        max_node_acceleration=0.0,
    )

    step = model.predict_step(belief, 0.02)

    assert step.auxiliary["pair_event_logits"].shape == (1, 2, 2, 2)
    assert step.auxiliary["boundary_event_logits"].shape == (1, 2, 1, 2)
    assert step.auxiliary["pair_collision"][0, 0, 1]
    assert step.auxiliary["pair_event_logits"][0, 0, 1, 1] > 0.0
    combined = torch.maximum(
        step.auxiliary["pair_event_logits"][..., 1].amax(dim=-1),
        step.auxiliary["boundary_event_logits"][..., 1].amax(dim=-1),
    )
    torch.testing.assert_close(
        step.event_logits[..., MotionMode.COLLISION],
        combined,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_smooth_event_conjunction_and_far_pair_remain_finite_on_mps() -> None:
    first_cpu = torch.tensor([-1.0e5, -500.0, -100.0, -90.0, 90.0, 100.0, 500.0, 1.0e5])
    second_cpu = torch.tensor([1.0e5, 100.0, -90.0, -100.0, 100.0, 90.0, -1.0e5, 500.0])
    expected = -torch.logaddexp(-first_cpu, -second_cpu)
    first = first_cpu.to("mps").requires_grad_()
    second = second_cpu.to("mps").requires_grad_()
    conjunction = EventModel._smooth_conjunction(first, second)

    _, cpu_objects = _two_objects()
    source = cpu_objects.to("mps")
    position = source.position.detach().clone()
    position[0, 0, 0] = -2.0
    position[0, 1, 0] = 2.0
    position.requires_grad_()
    fast_log_variance = source.fast_log_variance.detach().clone().requires_grad_()
    objects = replace(
        source,
        position=position,
        fast_log_variance=fast_log_variance,
    )
    result = EventModel(smooth_hazard_enabled=True).to("mps")(objects)

    assert torch.isfinite(conjunction).all().cpu().item()
    torch.testing.assert_close(conjunction.cpu(), expected)
    assert torch.isfinite(result.pair_event_logits).all().cpu().item()
    assert torch.isfinite(result.event_logits).all().cpu().item()
    loss = conjunction.sum() + result.pair_event_logits[0, 0, 1].sum()
    loss.backward()
    for gradient in (first.grad, second.grad, position.grad, fast_log_variance.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all().cpu().item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_smooth_event_hazard_is_finite_and_differentiable_on_mps() -> None:
    _, cpu_objects = _two_objects()
    source = cpu_objects.to("mps")
    position = source.position.detach().clone().requires_grad_()
    velocity = source.velocity.detach().clone().requires_grad_()
    fast_log_variance = source.fast_log_variance.detach().clone().requires_grad_()
    objects = replace(
        source,
        position=position,
        velocity=velocity,
        fast_log_variance=fast_log_variance,
    )
    edge_mask = torch.tensor(
        [[[False, True], [True, False]]],
        device="mps",
    )
    learned_collision = torch.tensor(
        [[[0.0, -20.0], [-20.0, 0.0]]],
        device="mps",
        requires_grad=True,
    )
    graph = _graph_output(
        objects,
        collision_logits=learned_collision,
        edge_mask=edge_mask,
    )

    result = EventModel(smooth_hazard_enabled=True).to("mps")(objects, graph)
    loss = balanced_binary_cross_entropy(
        result.event_logits[..., MotionMode.COLLISION],
        torch.ones_like(result.event_logits[..., MotionMode.COLLISION]),
        torch.ones_like(result.event_logits[..., MotionMode.COLLISION], dtype=torch.bool),
    )
    loss.backward()

    assert torch.isfinite(result.event_logits).all().cpu().item()
    assert learned_collision.grad is not None
    assert torch.isfinite(learned_collision.grad).all().cpu().item()
    assert learned_collision.grad[0, 0, 1].abs().cpu().item() > 0.0
    for gradient in (position.grad, velocity.grad, fast_log_variance.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all().cpu().item()
