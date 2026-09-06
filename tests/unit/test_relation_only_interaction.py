from __future__ import annotations

from dataclasses import fields

import torch

from world_model.belief import BeliefFactory
from world_model.dynamics import DynamicsModel
from world_model.dynamics.graph import InteractionGraph


def _objects():
    belief = BeliefFactory(
        max_objects=6,
        residual_dynamics_dim=1,
        global_code_dim=1,
    ).create()
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.arange(6)
    objects.position[0, :, 0] = torch.linspace(-0.25, 0.25, 6)
    objects.position[0, :, 1] = 1.0
    objects.velocity[0, :, 0] = torch.linspace(0.5, -0.5, 6)
    objects.geometry[..., 0] = 0.21
    objects.fast_log_variance.fill_(-8.0)
    return belief, objects


def test_relation_only_mode_disables_continuous_and_node_acceleration() -> None:
    belief, objects = _objects()
    graph = InteractionGraph(
        1,
        1,
        hidden_dim=16,
        interaction_radius=1.0,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        bounded_event_calibration_enabled=True,
    )
    with torch.no_grad():
        graph.edge_network.output.bias.copy_(torch.tensor([0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8]))
        graph.node_network.output.bias.fill_(0.9)

    output = graph(objects, belief.global_code)

    assert not output.pair_force.any()
    assert not output.pair_acceleration.any()
    assert not output.node_acceleration.any()
    assert not output.residual_acceleration.any()
    assert output.collision_logits.abs().sum() > 0.0
    assert output.contact_logits.abs().max() <= graph.event_calibration_logit_limit
    assert output.collision_logits.abs().max() <= graph.event_calibration_logit_limit
    torch.testing.assert_close(
        output.collision_logits,
        output.collision_logits.transpose(1, 2),
        rtol=0.0,
        atol=0.0,
    )
    assert output.impulse_multiplier_raw.abs().sum() > 0.0
    assert output.impulse_additive_raw.abs().sum() > 0.0
    assert output.edge_process_noise.abs().sum() > 0.0
    torch.testing.assert_close(
        output.impulse_additive_raw,
        output.impulse_additive_raw.transpose(1, 2),
    )
    assert sum(parameter.numel() for parameter in graph.parameters()) < 50_000


def test_relation_only_objective_reaches_only_supported_edge_output_rows() -> None:
    belief, objects = _objects()
    graph = InteractionGraph(
        1,
        1,
        hidden_dim=16,
        interaction_radius=1.0,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        bounded_event_calibration_enabled=True,
    )
    output = graph(objects, belief.global_code)
    loss = (
        output.contact_logits.square().sum()
        + output.collision_logits.square().sum()
        + output.impulse_multiplier_raw.square().sum()
        + output.impulse_additive_raw.square().sum()
        + output.edge_process_noise.square().sum()
    )
    # Zero-output initialization needs a nonzero target to expose the owner.
    loss = loss - output.collision_logits.sum() - output.impulse_additive_raw.sum()
    loss.backward()

    edge_gradient = graph.edge_network.output.bias.grad
    assert edge_gradient is not None
    assert torch.isfinite(edge_gradient).all()
    assert edge_gradient[[1, 5]].abs().sum() > 0.0
    assert not edge_gradient[[2, 3]].any()
    assert all(parameter.grad is None for parameter in graph.node_network.parameters())


def test_explicitly_enabled_flags_preserve_legacy_graph_function_exactly() -> None:
    belief, objects = _objects()
    torch.manual_seed(4)
    legacy = InteractionGraph(1, 1, hidden_dim=16, interaction_radius=1.0)
    explicit = InteractionGraph(
        1,
        1,
        hidden_dim=16,
        interaction_radius=1.0,
        continuous_pair_force_enabled=True,
        node_acceleration_enabled=True,
        bounded_event_calibration_enabled=False,
    )
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    legacy_output = legacy(objects, belief.global_code)
    explicit_output = explicit(objects, belief.global_code)

    for field in fields(legacy_output):
        torch.testing.assert_close(
            getattr(explicit_output, field.name),
            getattr(legacy_output, field.name),
            rtol=0.0,
            atol=0.0,
        )


def test_bounded_event_calibration_is_zero_identity_and_saturates_extremes() -> None:
    belief, objects = _objects()
    unbounded = InteractionGraph(1, 1, hidden_dim=16, interaction_radius=1.0)
    bounded = InteractionGraph(
        1,
        1,
        hidden_dim=16,
        interaction_radius=1.0,
        bounded_event_calibration_enabled=True,
    )
    bounded.load_state_dict(unbounded.state_dict(), strict=True)

    baseline = unbounded(objects, belief.global_code)
    calibrated = bounded(objects, belief.global_code)
    torch.testing.assert_close(
        calibrated.contact_logits,
        baseline.contact_logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        calibrated.collision_logits,
        baseline.collision_logits,
        rtol=0.0,
        atol=0.0,
    )

    with torch.no_grad():
        bounded.edge_network.output.weight.fill_(1.0e6)
        bounded.edge_network.output.bias.copy_(
            torch.tensor([1.0e6, -1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
    extreme = bounded(objects, belief.global_code)
    for value in (extreme.contact_logits, extreme.collision_logits):
        assert torch.isfinite(value).all()
        assert value.abs().max() <= bounded.event_calibration_logit_limit
        torch.testing.assert_close(value, value.transpose(1, 2), rtol=0.0, atol=0.0)


def test_learned_pair_confidence_can_disagree_without_overriding_analytic_collision() -> None:
    belief = BeliefFactory(
        max_objects=2,
        residual_dynamics_dim=1,
        global_code_dim=1,
        initial_radius=0.21,
        initial_drag=0.05,
    ).create(gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.tensor([10, 20])
    objects.position[0, :, 0] = torch.tensor([-0.5, 0.5])
    objects.position[0, :, 1] = 1.0
    objects.velocity[0, :, 0] = torch.tensor([1.0, -1.0])
    objects.geometry[..., 0] = 0.21
    objects.fast_log_variance.fill_(-12.0)
    belief = belief.replace(objects=objects).validate()
    model = DynamicsModel.from_belief(
        belief,
        max_substep=1.0 / 120.0,
        graph_hidden_dim=16,
        graph_relation_hidden_dim=16,
        uncertainty_hidden_dim=8,
        interaction_radius=1.0,
        modal_dynamics_enabled=False,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        event_driven_state_only_enabled=True,
        relation_process_uncertainty_enabled=True,
        world_bounds=((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)),
        process_noise_position=1.0e-8,
        process_noise_velocity=1.0e-8,
        log_variance_min=-32.0,
    )
    reference = DynamicsModel(model.config)
    reference.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        model.interactions.edge_network.output.bias[1] = -1.0e6

    calibrated = model.predict_state_only_step(belief, 0.5)
    analytic = reference.predict_state_only_step(belief, 0.5)

    assert calibrated.auxiliary["pair_collision"][0, 0, 1]
    assert calibrated.auxiliary["pair_collision_logits"][0, 0, 1] < 0.0
    torch.testing.assert_close(
        calibrated.belief.objects.position,
        analytic.belief.objects.position,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        calibrated.belief.objects.velocity,
        analytic.belief.objects.velocity,
        rtol=0.0,
        atol=0.0,
    )


def test_relation_only_dynamics_preserves_modal_state_and_has_no_learned_acceleration() -> None:
    belief = BeliefFactory(
        max_objects=1,
        modal_count=2,
        modal_dim=2,
        residual_dynamics_dim=1,
        global_code_dim=1,
    ).create(gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = 3
    objects.position[..., 1] = 2.0
    objects.modal_state.fill_(0.4)
    objects.modal_frequency.fill_(1.3)
    objects.modal_decay_raw.fill_(-0.2)
    belief = belief.replace(objects=objects)
    model = DynamicsModel.from_belief(
        belief,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=8,
        modal_dynamics_enabled=False,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        ground_height=-10.0,
    )
    with torch.no_grad():
        model.interactions.edge_network.output.bias.copy_(
            torch.tensor([0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8])
        )
        readout = model.modal.readout
        if isinstance(readout, torch.nn.Linear):
            readout.weight.fill_(0.5)

    step = model.predict_step(belief, 0.05)

    assert torch.equal(step.belief.objects.modal_state, belief.objects.modal_state)
    assert not step.auxiliary["residual_acceleration"].any()
