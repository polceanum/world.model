from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.dynamics import (
    DynamicsModel,
    InteractionOutput,
    PairApplicabilityConfig,
    apply_pair_applicability,
    pair_applicability,
)
from world_model.training.checkpointing import (
    validate_checkpoint_config,
    validate_training_resume_config,
)
from world_model.utils.config import load_config
from world_model.utils.version import SIMULATOR_VERSION


def _belief(*, gap: float, normal_velocity: float, log_variance: float = -12.0):
    belief = BeliefFactory(max_objects=3, residual_dynamics_dim=4).create()
    objects = belief.objects.clone()
    objects.active[:] = False
    objects.active[0, :2] = True
    objects.object_id[0, :2] = torch.tensor([10, 11])
    objects.geometry[0, :2, 0] = 0.1
    objects.position[0, 0] = torch.tensor([0.0, 2.0, 0.0])
    objects.position[0, 1] = torch.tensor([0.2 + gap, 2.0, 0.0])
    objects.velocity.zero_()
    objects.velocity[0, 1, 0] = normal_velocity
    objects.fast_log_variance.fill_(log_variance)
    objects.log_drag.fill_(-16.0)
    return replace(
        belief,
        objects=objects,
        gravity=torch.zeros_like(belief.gravity),
    )


def _edge_mask(objects) -> torch.Tensor:
    mask = torch.zeros(
        1,
        objects.max_objects,
        objects.max_objects,
        device=objects.active.device,
        dtype=torch.bool,
    )
    mask[0, 0, 1] = True
    mask[0, 1, 0] = True
    return mask


def _interaction(objects, *, differentiable: bool = False) -> tuple[InteractionOutput, list]:
    edge_mask = _edge_mask(objects)
    off_diagonal = edge_mask.to(objects.position.dtype)
    force_scale = objects.position.new_tensor(1.0, requires_grad=differentiable)
    scalar_scale = objects.position.new_tensor(1.0, requires_grad=differentiable)
    force_basis = objects.position.new_zeros(1, objects.max_objects, objects.max_objects, 3)
    force_basis[0, 0, 1, 0] = 1.0
    force_basis[0, 1, 0, 0] = -1.0
    pair_force = force_basis * force_scale
    pair_acceleration = pair_force.sum(dim=2) / objects.mass
    node_acceleration = objects.position.new_zeros(1, objects.max_objects, 3)
    node_acceleration[0, :2, 1] = 0.25
    pair_scalar = off_diagonal * scalar_scale
    interaction = InteractionOutput(
        residual_acceleration=(pair_acceleration + node_acceleration)
        * objects.active.unsqueeze(-1),
        pair_acceleration=pair_acceleration,
        node_acceleration=node_acceleration,
        pair_force=pair_force,
        contact_logits=pair_scalar,
        collision_logits=pair_scalar,
        impulse_multiplier_raw=pair_scalar,
        impulse_additive_raw=pair_scalar,
        edge_process_noise=pair_scalar,
        edge_mask=edge_mask,
        interaction_density=edge_mask.sum(dim=-1).to(objects.position.dtype),
    )
    return interaction, [force_scale, scalar_scale]


def _enabled_config() -> PairApplicabilityConfig:
    return PairApplicabilityConfig(
        enabled=True,
        lookahead_seconds=0.05,
        margin_m=0.05,
        gap_temperature_m=0.025,
        velocity_temperature_mps=0.10,
        collision_speed_epsilon=1.0e-7,
    )


def _permute_objects(objects, order: torch.Tensor):
    updates = {item.name: getattr(objects, item.name)[:, order] for item in fields(objects)}
    return replace(objects, **updates)


def test_pair_applicability_is_symmetric_permutation_equivariant_and_masked() -> None:
    objects = _belief(gap=0.08, normal_velocity=-0.6).objects
    mask = _edge_mask(objects)
    output = pair_applicability(objects, mask, _enabled_config()).validate(objects)

    torch.testing.assert_close(output.pair, output.pair.transpose(1, 2))
    torch.testing.assert_close(output.collision, output.collision.transpose(1, 2))
    assert torch.count_nonzero(output.pair) == 2
    assert torch.count_nonzero(output.collision) == 2

    order = torch.tensor([1, 0, 2])
    permuted_objects = _permute_objects(objects, order)
    permuted_mask = mask[:, order][:, :, order]
    permuted = pair_applicability(
        permuted_objects,
        permuted_mask,
        _enabled_config(),
    )
    torch.testing.assert_close(permuted.pair[:, order][:, :, order], output.pair)
    torch.testing.assert_close(permuted.collision[:, order][:, :, order], output.collision)


def test_applicability_is_monotone_in_gap_closing_and_far_uncertainty() -> None:
    config = _enabled_config()
    near_objects = _belief(gap=0.03, normal_velocity=-0.5).objects
    far_objects = _belief(gap=0.40, normal_velocity=-0.5).objects
    separating_objects = _belief(gap=0.03, normal_velocity=0.5).objects
    uncertain_far_objects = _belief(
        gap=0.40,
        normal_velocity=-0.5,
        log_variance=-1.0,
    ).objects

    near = pair_applicability(near_objects, _edge_mask(near_objects), config)
    far = pair_applicability(far_objects, _edge_mask(far_objects), config)
    separating = pair_applicability(
        separating_objects,
        _edge_mask(separating_objects),
        config,
    )
    uncertain_far = pair_applicability(
        uncertain_far_objects,
        _edge_mask(uncertain_far_objects),
        config,
    )

    assert near.pair[0, 0, 1] > far.pair[0, 0, 1]
    assert near.collision[0, 0, 1] > separating.collision[0, 0, 1]
    assert uncertain_far.pair[0, 0, 1] > far.pair[0, 0, 1]


def test_apply_gate_suppresses_far_effects_preserves_nodes_and_conservation() -> None:
    objects = _belief(gap=0.80, normal_velocity=1.0).objects
    interaction, _ = _interaction(objects)

    gated, applicability = apply_pair_applicability(
        objects,
        interaction,
        _enabled_config(),
    )

    assert applicability.pair[0, 0, 1] < 1.0e-6
    assert applicability.collision[0, 0, 1] < applicability.pair[0, 0, 1]
    torch.testing.assert_close(gated.node_acceleration, interaction.node_acceleration)
    torch.testing.assert_close(
        gated.pair_force,
        -gated.pair_force.transpose(1, 2),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        gated.pair_force.sum(dim=(1, 2)),
        torch.zeros(1, 3),
        rtol=0.0,
        atol=0.0,
    )
    expected_pair_acceleration = gated.pair_force.sum(dim=2) / objects.mass
    torch.testing.assert_close(gated.pair_acceleration, expected_pair_acceleration)
    torch.testing.assert_close(
        gated.residual_acceleration,
        (expected_pair_acceleration + interaction.node_acceleration) * objects.active.unsqueeze(-1),
    )


def test_near_closing_gate_retains_finite_gradients_for_every_typed_effect() -> None:
    objects = _belief(gap=0.0, normal_velocity=-1.0).objects
    interaction, leaves = _interaction(objects, differentiable=True)

    gated, applicability = apply_pair_applicability(
        objects,
        interaction,
        _enabled_config(),
    )
    loss = (
        gated.residual_acceleration.square().sum()
        + gated.contact_logits.square().sum()
        + gated.collision_logits.square().sum()
        + gated.impulse_multiplier_raw.square().sum()
        + gated.impulse_additive_raw.square().sum()
        + gated.edge_process_noise.square().sum()
    )
    loss.backward()

    assert applicability.pair[0, 0, 1] > 0.9
    assert applicability.collision[0, 0, 1] > 0.9
    for leaf in leaves:
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad)
        assert leaf.grad.abs() > 0.0


def test_applicability_envelope_has_finite_geometry_and_motion_gradients() -> None:
    objects = _belief(gap=0.05, normal_velocity=-0.1).objects
    position = objects.position.detach().clone().requires_grad_()
    velocity = objects.velocity.detach().clone().requires_grad_()
    objects = replace(objects, position=position, velocity=velocity)

    applicability = pair_applicability(
        objects,
        _edge_mask(objects),
        _enabled_config(),
    )
    (applicability.pair[0, 0, 1] + applicability.collision[0, 0, 1]).backward()

    assert position.grad is not None
    assert velocity.grad is not None
    assert torch.isfinite(position.grad).all()
    assert torch.isfinite(velocity.grad).all()
    assert position.grad.abs().sum() > 0.0
    assert velocity.grad.abs().sum() > 0.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_applicability_is_finite_and_differentiable_on_mps() -> None:
    belief = _belief(gap=0.01, normal_velocity=-0.8).to("mps")
    interaction, leaves = _interaction(belief.objects, differentiable=True)

    gated, applicability = apply_pair_applicability(
        belief.objects,
        interaction,
        _enabled_config(),
    )
    loss = (
        gated.residual_acceleration.square().sum()
        + gated.collision_logits.square().sum()
        + gated.edge_process_noise.square().sum()
        + applicability.pair.square().sum()
    )
    loss.backward()

    assert torch.isfinite(applicability.pair).all().item()
    assert torch.isfinite(applicability.collision).all().item()
    for leaf in leaves:
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).item()
        assert (leaf.grad.abs() > 0.0).item()


def test_disabled_gate_is_exact_identity_and_adds_no_model_state() -> None:
    belief = _belief(gap=0.4, normal_velocity=0.5)
    interaction, _ = _interaction(belief.objects)
    unchanged, applicability = apply_pair_applicability(
        belief.objects,
        interaction,
        PairApplicabilityConfig(enabled=False),
    )
    assert unchanged is interaction
    torch.testing.assert_close(
        applicability.pair,
        interaction.edge_mask.to(interaction.pair_force.dtype),
        rtol=0.0,
        atol=0.0,
    )

    disabled = DynamicsModel.from_belief(belief, pair_applicability_enabled=False)
    enabled = DynamicsModel.from_belief(belief, pair_applicability_enabled=True)
    assert tuple(disabled.state_dict()) == tuple(enabled.state_dict())
    enabled.load_state_dict(disabled.state_dict(), strict=True)


def test_zero_learned_effects_leave_analytic_collision_exact() -> None:
    belief = _belief(gap=0.02, normal_velocity=-2.0)
    objects = belief.objects.clone()
    objects.velocity[0, 0, 0] = 1.0
    objects.velocity[0, 1, 0] = -1.0
    belief = replace(belief, objects=objects)
    disabled = DynamicsModel.from_belief(
        belief,
        max_substep=0.01,
        pair_applicability_enabled=False,
    ).eval()
    enabled = DynamicsModel.from_belief(
        belief,
        max_substep=0.01,
        pair_applicability_enabled=True,
    ).eval()
    enabled.load_state_dict(disabled.state_dict())
    with torch.no_grad():
        for model in (disabled, enabled):
            for parameter in model.interactions.parameters():
                parameter.zero_()
            if isinstance(model.modal.readout, torch.nn.Linear):
                model.modal.readout.weight.zero_()
                model.modal.readout.bias.zero_()

        reference = disabled.predict_step(belief, 0.03)
        candidate = enabled.predict_step(belief, 0.03)

    torch.testing.assert_close(
        candidate.belief.objects.position,
        reference.belief.objects.position,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate.belief.objects.velocity,
        reference.belief.objects.velocity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(candidate.event_logits, reference.event_logits, rtol=0.0, atol=0.0)
    assert torch.equal(
        candidate.auxiliary["pair_collision"],
        reference.auxiliary["pair_collision"],
    )
    torch.testing.assert_close(
        candidate.auxiliary["pair_impulse"],
        reference.auxiliary["pair_impulse"],
        rtol=0.0,
        atol=0.0,
    )


def test_applicability_config_is_opt_in_validated_and_checkpoint_strict() -> None:
    config = load_config("configs/toy_smoke.yaml")
    dynamics = config.model.dynamics
    assert not dynamics.pair_applicability_enabled
    assert dynamics.pair_applicability_lookahead_seconds == pytest.approx(0.05)
    assert dynamics.pair_applicability_margin_m == pytest.approx(0.05)
    assert dynamics.pair_applicability_gap_temperature_m == pytest.approx(0.025)
    assert dynamics.pair_applicability_velocity_temperature_mps == pytest.approx(0.10)

    enabled = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(config.model.dynamics, pair_applicability_enabled=True),
        ),
    )
    enabled.validate()
    invalid = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(
                config.model.dynamics,
                pair_applicability_gap_temperature_m=0.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="gap_temperature_m must be finite and positive"):
        invalid.validate()
    with pytest.raises(ValueError, match="gap_temperature_m must be finite and positive"):
        PairApplicabilityConfig(enabled=True, gap_temperature_m=0.0)

    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    legacy_payload = deepcopy(payload)
    for field_name in (
        "pair_applicability_enabled",
        "pair_applicability_lookahead_seconds",
        "pair_applicability_margin_m",
        "pair_applicability_gap_temperature_m",
        "pair_applicability_velocity_temperature_mps",
    ):
        legacy_payload["config"]["model"]["dynamics"].pop(field_name)
    validate_checkpoint_config(legacy_payload, config)
    validate_training_resume_config(legacy_payload, config)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, enabled)
    with pytest.raises(ValueError, match="model.dynamics.pair_applicability_enabled"):
        validate_training_resume_config(payload, enabled)
