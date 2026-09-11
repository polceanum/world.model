from __future__ import annotations

from dataclasses import replace

import torch

from world_model.belief import BeliefFactory, MotionMode, RigidGeometryCodec, RigidPrimitive
from world_model.dynamics import DynamicsModel
from world_model.simulator import (
    PhysicsConfig,
    RigidBodyState,
    SphereState,
    advance_rigid_bodies_6dof,
)

DTYPE = torch.float64


def _rigid_state(*, box: bool = True) -> RigidBodyState:
    sphere = SphereState(
        object_id=torch.tensor([10, 20]),
        active=torch.tensor([True, True]),
        position=torch.tensor([[0.0, 0.0, 0.0], [-0.90, 0.18, 0.0]], dtype=DTYPE),
        velocity=torch.tensor([[0.0, 0.0, 0.0], [1.35, 0.0, 0.0]], dtype=DTYPE),
        radius=torch.tensor([[0.48], [0.16]], dtype=DTYPE),
        mass=torch.tensor([[1.7], [0.8]], dtype=DTYPE),
        restitution=torch.tensor([[0.55], [0.55]], dtype=DTYPE),
        drag=torch.full((2, 1), 1.0e-8, dtype=DTYPE),
        friction=torch.tensor([[0.42], [0.42]], dtype=DTYPE),
        albedo=torch.tensor([[0.2, 0.7, 0.3], [0.8, 0.2, 0.5]], dtype=DTYPE),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=DTYPE),
        angular_velocity=torch.zeros(2, 3, dtype=DTYPE),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    rigid = RigidBodyState.from_spheres(sphere)
    if not box:
        return rigid
    extent = torch.tensor([0.35, 0.25, 0.20], dtype=DTYPE)
    return replace(
        rigid,
        primitive=torch.tensor([int(RigidPrimitive.BOX), int(RigidPrimitive.SPHERE)]),
        half_extents=torch.stack((extent, torch.full((3,), 0.16, dtype=DTYPE))),
        radius=torch.tensor([[float(torch.linalg.vector_norm(extent))], [0.16]], dtype=DTYPE),
    )


def _belief(state: RigidBodyState):
    belief = BeliefFactory(
        max_objects=2,
        geometry_dim=5,
        appearance_dim=3,
        residual_dynamics_dim=1,
        modal_count=0,
        modal_dim=1,
        parameter_memory_dim=1,
        global_code_dim=1,
    ).create(batch_size=1, dtype=DTYPE, gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[0] = state.active
    objects.object_id[0] = state.object_id
    objects.position[0] = state.position
    objects.velocity[0] = state.velocity
    objects.orientation[0] = state.orientation
    objects.angular_velocity[0] = state.angular_velocity
    objects.geometry[0, 0] = (
        RigidGeometryCodec.encode_box(state.half_extents[:1], geometry_dim=5)[0]
        if int(state.primitive[0]) == int(RigidPrimitive.BOX)
        else RigidGeometryCodec.encode_sphere(state.radius[:1], geometry_dim=5)[0]
    )
    objects.geometry[0, 1] = RigidGeometryCodec.encode_sphere(state.radius[1:2], geometry_dim=5)[0]
    objects.log_mass[0] = state.mass.log()
    objects.restitution_logit[0] = torch.logit(state.restitution)
    objects.log_drag[0] = state.drag.log()
    objects.friction_logit[0] = torch.logit(state.friction)
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., int(MotionMode.FREE)] = 4.0
    objects.fast_log_variance.fill_(-20.0)
    objects.slow_log_variance.fill_(-20.0)
    return replace(belief, objects=objects).validate()


def _model(belief, *, six_dof: bool) -> DynamicsModel:
    model = DynamicsModel.from_belief(
        belief,
        max_substep=1.0 / 240.0,
        graph_hidden_dim=16,
        uncertainty_hidden_dim=16,
        interaction_radius=2.0,
        world_bounds=((-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)),
        modal_dynamics_enabled=False,
        continuous_pair_force_enabled=False,
        node_acceleration_enabled=False,
        rigid_six_dof_contacts_enabled=six_dof,
    ).to(dtype=DTYPE)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model.eval()


def test_six_dof_opt_in_preserves_exact_all_sphere_path() -> None:
    belief = _belief(_rigid_state(box=False))
    ordinary = _model(belief, six_dof=False).rollout(belief, [0.6])
    six_dof = _model(belief, six_dof=True).rollout(belief, [0.6])

    torch.testing.assert_close(six_dof.positions, ordinary.positions, rtol=0.0, atol=0.0)
    torch.testing.assert_close(six_dof.velocities, ordinary.velocities, rtol=0.0, atol=0.0)
    torch.testing.assert_close(six_dof.orientations, ordinary.orientations, rtol=0.0, atol=0.0)


def test_off_centre_contact_generates_rotation_and_matches_independent_oracle() -> None:
    state = _rigid_state()
    belief = _belief(state)
    model = _model(belief, six_dof=True)
    with torch.no_grad():
        predicted = model.rollout(belief, [0.6])
    reference, events = advance_rigid_bodies_6dof(
        state,
        0.6,
        PhysicsConfig(
            gravity=(0.0, 0.0, 0.0),
            bounds=((-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)),
            max_substep=1.0 / 240.0,
            solver_iterations=2,
        ),
    )

    assert events.pair_collision.any()
    assert torch.linalg.vector_norm(reference.angular_velocity[0]) > 0.05
    assert torch.linalg.vector_norm(predicted.orientations[0, -1, 0, :3]) > 0.01
    torch.testing.assert_close(predicted.positions[0, -1], reference.position, atol=0.025, rtol=0.0)
    torch.testing.assert_close(predicted.velocities[0, -1], reference.velocity, atol=0.04, rtol=0.0)
    torch.testing.assert_close(
        predicted.orientations[0, -1], reference.orientation, atol=0.04, rtol=0.0
    )


def test_six_dof_config_rejects_event_driven_shortcut() -> None:
    belief = _belief(_rigid_state())
    try:
        DynamicsModel.from_belief(
            belief,
            event_driven_state_only_enabled=True,
            rigid_six_dof_contacts_enabled=True,
            modal_dynamics_enabled=False,
            continuous_pair_force_enabled=False,
            node_acceleration_enabled=False,
            world_bounds=((-1.0, 1.0),) * 3,
        )
    except ValueError as error:
        assert "six-DoF" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("incompatible six-DoF shortcut was accepted")
