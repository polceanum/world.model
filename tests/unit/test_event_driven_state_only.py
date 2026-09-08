"""Certified fast state-only contact integration for specification 1.61."""

from __future__ import annotations

from dataclasses import replace

import torch

from world_model.belief import BeliefFactory
from world_model.dynamics import DynamicsModel, WorldImpulseAction
from world_model.training.dynamic_set_planning import _concatenate_beliefs


def _belief(
    *,
    positions: tuple[float, float] = (-0.50, 0.50),
    velocities: tuple[float, float] = (1.0, -1.0),
):
    belief = BeliefFactory(
        max_objects=2,
        residual_dynamics_dim=1,
        global_code_dim=1,
        initial_radius=0.21,
        initial_drag=0.05,
    ).create(gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[0] = torch.tensor((10, 20))
    objects.position[0, :, 0] = torch.tensor(positions)
    objects.position[0, :, 1] = 1.0
    objects.velocity[0, :, 0] = torch.tensor(velocities)
    objects.geometry[..., 0] = 0.21
    objects.fast_log_variance.fill_(-12.0)
    return belief.replace(objects=objects).validate()


def _model(
    belief,
    *,
    event_driven: bool,
    world_bounds: tuple[tuple[float, float], ...] = (
        (-30.0, 30.0),
        (-30.0, 30.0),
        (-30.0, 30.0),
    ),
) -> DynamicsModel:
    return DynamicsModel.from_belief(
        belief,
        max_substep=1.0 / 120.0,
        graph_hidden_dim=16,
        graph_relation_hidden_dim=16,
        uncertainty_hidden_dim=8,
        interaction_radius=1.0,
        modal_dynamics_enabled=False,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        attention_residual_enabled=False,
        event_driven_state_only_enabled=event_driven,
        relation_process_uncertainty_enabled=True,
        world_bounds=world_bounds,
        process_noise_position=1.0e-8,
        process_noise_velocity=1.0e-8,
        log_variance_min=-32.0,
    )


def test_event_driven_contact_matches_authoritative_microsteps_within_physical_gate() -> None:
    belief = _belief()
    fast = _model(belief, event_driven=True)
    reference = _model(belief, event_driven=False)
    reference.load_state_dict(fast.state_dict(), strict=True)

    query = (0.10, 0.25, 0.50)
    fast_trajectory = fast.rollout(
        belief,
        query,
        return_events=False,
        return_auxiliary=False,
    )
    reference_trajectory = reference.rollout(
        belief,
        query,
        return_events=False,
        return_auxiliary=False,
    )

    assert fast_trajectory.event_logits is None
    assert fast_trajectory.auxiliary == {}
    torch.testing.assert_close(
        fast_trajectory.positions,
        reference_trajectory.positions,
        rtol=0.0,
        atol=0.010,
    )
    torch.testing.assert_close(
        fast_trajectory.velocities,
        reference_trajectory.velocities,
        rtol=0.0,
        atol=2.0e-5,
    )


def test_event_driven_path_falls_back_exactly_when_a_boundary_is_reachable() -> None:
    belief = _belief(positions=(1.65, -0.75), velocities=(1.0, 0.0))
    bounds = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
    fast = _model(belief, event_driven=True, world_bounds=bounds)
    reference = _model(belief, event_driven=False, world_bounds=bounds)
    reference.load_state_dict(fast.state_dict(), strict=True)

    fast_trajectory = fast.rollout(
        belief,
        (0.50,),
        return_events=False,
        return_auxiliary=False,
    )
    reference_trajectory = reference.rollout(
        belief,
        (0.50,),
        return_events=False,
        return_auxiliary=False,
    )

    torch.testing.assert_close(
        fast_trajectory.positions,
        reference_trajectory.positions,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fast_trajectory.velocities,
        reference_trajectory.velocities,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fast_trajectory.fast_log_variance,
        reference_trajectory.fast_log_variance,
        rtol=0.0,
        atol=0.0,
    )


def test_contact_uncertainty_reaches_only_the_relation_process_output() -> None:
    belief = _belief()
    model = _model(belief, event_driven=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.interactions.edge_network.parameters():
        parameter.requires_grad_(True)

    trajectory = model.rollout(
        belief,
        (0.50,),
        return_events=False,
        return_auxiliary=False,
    )
    trajectory.fast_log_variance.sum().backward()

    output_bias = model.interactions.edge_network.output.bias
    assert output_bias.grad is not None
    assert torch.isfinite(output_bias.grad).all()
    assert float(output_bias.grad[6].abs()) > 0.0
    assert not output_bias.grad[:6].any()
    assert all(
        parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad
    )


def test_state_only_interval_exposes_analytic_pair_event_with_collision_gradient() -> None:
    belief = _belief()
    model = _model(belief, event_driven=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.interactions.edge_network.parameters():
        parameter.requires_grad_(True)

    step = model.predict_state_only_step(belief, 0.50)

    assert step.auxiliary["pair_collision"][0, 0, 1]
    assert step.auxiliary["pair_collision"][0, 1, 0]
    collision_logits = step.auxiliary["pair_event_logits"][..., 1]
    assert collision_logits[0, 0, 1] >= 6.0
    collision_logits.sum().backward()
    output = model.interactions.edge_network.output
    assert output.weight.grad is not None
    assert torch.isfinite(output.weight.grad).all()
    assert output.weight.grad[1].abs().sum() > 0.0


def test_unequal_drag_uses_the_unchanged_authoritative_path() -> None:
    belief = _belief()
    objects = belief.objects.clone()
    objects.log_drag[0, 1] = torch.log(torch.tensor(0.08))
    belief = replace(belief, objects=objects).validate()
    fast = _model(belief, event_driven=True)
    reference = _model(belief, event_driven=False)
    reference.load_state_dict(fast.state_dict(), strict=True)

    actual = fast.rollout(
        belief,
        (0.50,),
        return_events=False,
        return_auxiliary=False,
    )
    expected = reference.rollout(
        belief,
        (0.50,),
        return_events=False,
        return_auxiliary=False,
    )

    torch.testing.assert_close(actual.positions, expected.positions, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.velocities, expected.velocities, rtol=0.0, atol=0.0)


def test_public_rollout_matches_independent_rows_with_nonuniform_queries_and_actions() -> None:
    first = _belief()
    second = _belief(
        positions=(-0.65, 0.65),
        velocities=(1.20, -0.80),
    )
    batched = _concatenate_beliefs((first, second))
    model = _model(first, event_driven=True)
    query_offsets = first.timestamp.new_tensor([[0.10, 0.25, 0.50], [0.05, 0.35, 0.50]])
    action = WorldImpulseAction(
        timestamp=first.timestamp.new_tensor([0.15, 0.30]),
        object_id=torch.tensor([10, 20], dtype=torch.int64),
        impulse_world=first.timestamp.new_tensor([[0.10, 0.00, 0.00], [-0.05, 0.00, 0.00]]),
    )

    batched_trajectory = model.rollout(batched, query_offsets, action=action)
    for index, belief in enumerate((first, second)):
        single_action = WorldImpulseAction(
            timestamp=action.timestamp[index : index + 1],
            object_id=action.object_id[index : index + 1],
            impulse_world=action.impulse_world[index : index + 1],
        )
        single = model.rollout(
            belief,
            query_offsets[index : index + 1],
            action=single_action,
        )
        for name in (
            "timestamps",
            "positions",
            "velocities",
            "orientations",
            "motion_mode_logits",
            "fast_log_variance",
            "active_mask",
            "event_logits",
        ):
            assert torch.equal(
                getattr(batched_trajectory, name)[index : index + 1],
                getattr(single, name),
            ), name
        assert batched_trajectory.auxiliary.keys() == single.auxiliary.keys()
        for name, value in batched_trajectory.auxiliary.items():
            assert torch.equal(value[index : index + 1], single.auxiliary[name]), name
