from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.identification import (
    BoundaryCollisionEvidence,
    BoundaryCollisionPositionEvidence,
    FreeMotionEvidence,
    FreeMotionPositionEvidence,
    KnownImpulseEvidence,
    KnownImpulsePositionEvidence,
    OnlineRigidParameterEstimator,
    PairCollisionEvidence,
)

DTYPE = torch.float64


def _belief():
    belief = BeliefFactory(
        max_objects=2,
        geometry_dim=5,
        appearance_dim=8,
        residual_dynamics_dim=1,
        modal_count=0,
        modal_dim=1,
        parameter_memory_dim=8,
        global_code_dim=1,
        initial_mass=1.0,
        initial_restitution=0.5,
        initial_drag=0.05,
        initial_friction=0.25,
    ).create(batch_size=1, dtype=DTYPE, gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.tensor([[11, 29]])
    objects.existence_logit.fill_(8.0)
    objects.geometry[..., 0] = 0.2
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., int(MotionMode.FREE)] = 4.0
    objects.slow_log_variance.fill_(2.0)
    return replace(belief, objects=objects).validate()


def test_known_actions_and_free_motion_recover_mass_and_drag() -> None:
    belief = _belief()
    estimator = OnlineRigidParameterEstimator(gain=1.0)
    mass_evidence = KnownImpulseEvidence(
        object_id=torch.tensor([11]),
        velocity_before=torch.tensor([[0.2, -0.1, 0.0]], dtype=DTYPE),
        velocity_after=torch.tensor([[0.7, 0.15, 0.0]], dtype=DTYPE),
        impulse_world=torch.tensor([[1.0, 0.5, 0.0]], dtype=DTYPE),
    )
    belief, mass_update = estimator.update_known_impulse(belief, mass_evidence)
    duration = torch.tensor([0.8], dtype=DTYPE)
    drag = 0.23
    velocity_before = torch.tensor([[1.1, -0.4, 0.2]], dtype=DTYPE)
    velocity_after = velocity_before * torch.exp(torch.tensor(-drag, dtype=DTYPE) * duration)
    belief, drag_update = estimator.update_free_motion(
        belief,
        FreeMotionEvidence(
            object_id=torch.tensor([11]),
            velocity_before=velocity_before,
            velocity_after=velocity_after,
            duration_seconds=duration,
        ),
    )

    assert mass_update.accepted.item() and drag_update.accepted.item()
    torch.testing.assert_close(belief.objects.mass[0, 0], torch.tensor([2.0], dtype=DTYPE))
    torch.testing.assert_close(belief.objects.drag[0, 0], torch.tensor([drag], dtype=DTYPE))
    assert belief.objects.slow_log_variance[0, 0, :3].min() < 2.0


def test_pair_collision_recovers_effective_restitution_and_sliding_friction() -> None:
    belief = _belief()
    objects = belief.objects.clone()
    objects.log_mass[0, 0, 0] = torch.tensor(2.0, dtype=DTYPE).log()
    objects.log_mass[0, 1, 0] = torch.tensor(1.0, dtype=DTYPE).log()
    belief = replace(belief, objects=objects)
    estimator = OnlineRigidParameterEstimator(gain=1.0)
    before_first = torch.tensor([[1.0, 1.0, 0.0]], dtype=DTYPE)
    before_second = torch.tensor([[-1.0, 0.0, 0.0]], dtype=DTYPE)
    impulse = torch.tensor([[2.1333333333333333, 0.64, 0.0]], dtype=DTYPE)
    after_first = before_first - impulse / 2.0
    after_second = before_second + impulse
    belief, updates = estimator.update_pair_collision(
        belief,
        PairCollisionEvidence(
            first_object_id=torch.tensor([11]),
            second_object_id=torch.tensor([29]),
            normal_world=torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE),
            first_velocity_before=before_first,
            first_velocity_after=after_first,
            second_velocity_before=before_second,
            second_velocity_after=after_second,
        ),
    )

    assert all(bool(update.accepted.all()) for update in updates)
    torch.testing.assert_close(
        belief.objects.restitution[0, :, 0], torch.full((2,), 0.6, dtype=DTYPE)
    )
    torch.testing.assert_close(belief.objects.friction[0, :, 0], torch.full((2,), 0.3, dtype=DTYPE))


def test_boundary_collision_recovers_per_object_restitution_and_friction() -> None:
    belief = _belief()
    objects = belief.objects.clone()
    objects.log_mass[0, 0, 0] = torch.tensor(2.0, dtype=DTYPE).log()
    belief = replace(belief, objects=objects)
    estimator = OnlineRigidParameterEstimator(gain=1.0)
    velocity_before = torch.tensor([[1.0, 0.7, 0.0]], dtype=DTYPE)
    velocity_after = torch.tensor([[-0.7, 0.19, 0.0]], dtype=DTYPE)

    updated, updates = estimator.update_boundary_collision(
        belief,
        BoundaryCollisionEvidence(
            object_id=torch.tensor([11]),
            normal_world=torch.tensor([[-1.0, 0.0, 0.0]], dtype=DTYPE),
            velocity_before=velocity_before,
            velocity_after=velocity_after,
        ),
    )

    assert all(bool(update.accepted.all()) for update in updates)
    torch.testing.assert_close(updated.objects.restitution[0, 0, 0], torch.tensor(0.7, dtype=DTYPE))
    torch.testing.assert_close(updated.objects.friction[0, 0, 0], torch.tensor(0.3, dtype=DTYPE))
    torch.testing.assert_close(updated.objects.restitution[0, 1], belief.objects.restitution[0, 1])
    torch.testing.assert_close(updated.objects.friction[0, 1], belief.objects.friction[0, 1])


def _drag_positions(
    timestamps: torch.Tensor,
    *,
    position_at_zero: torch.Tensor,
    velocity_at_zero: torch.Tensor,
    drag: float,
) -> torch.Tensor:
    scale = -torch.expm1(-drag * timestamps) / drag
    return position_at_zero + scale[:, None] * velocity_at_zero


def test_position_traces_recover_drag_mass_and_boundary_response() -> None:
    belief = _belief()
    estimator = OnlineRigidParameterEstimator(gain=1.0)
    object_id = torch.tensor([11])
    drag = 0.23
    free_times = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=DTYPE)
    free_positions = _drag_positions(
        free_times[0],
        position_at_zero=torch.tensor([0.2, -0.1, 1.0], dtype=DTYPE),
        velocity_at_zero=torch.tensor([1.1, -0.4, 0.2], dtype=DTYPE),
        drag=drag,
    ).unsqueeze(0)
    belief, drag_update = estimator.update_free_motion_positions(
        belief,
        FreeMotionPositionEvidence(object_id, free_positions, free_times),
    )

    action_time = torch.tensor([0.0], dtype=DTYPE)
    before_times = torch.tensor([[-0.2, -0.1, 0.0]], dtype=DTYPE)
    after_times = torch.tensor([[0.0, 0.1, 0.2]], dtype=DTYPE)
    position_at_action = torch.tensor([0.3, 0.2, 1.0], dtype=DTYPE)
    velocity_before = torch.tensor([0.2, -0.1, 0.0], dtype=DTYPE)
    impulse = torch.tensor([[1.0, 0.5, 0.0]], dtype=DTYPE)
    velocity_after = velocity_before + impulse[0] / 2.0
    before_positions = _drag_positions(
        before_times[0],
        position_at_zero=position_at_action,
        velocity_at_zero=velocity_before,
        drag=drag,
    ).unsqueeze(0)
    after_positions = _drag_positions(
        after_times[0],
        position_at_zero=position_at_action,
        velocity_at_zero=velocity_after,
        drag=drag,
    ).unsqueeze(0)
    belief, mass_update = estimator.update_known_impulse_positions(
        belief,
        KnownImpulsePositionEvidence(
            object_id,
            before_positions,
            before_times,
            after_positions,
            after_times,
            action_time,
            impulse,
        ),
    )

    boundary_before = torch.tensor([1.0, 0.7, 0.0], dtype=DTYPE)
    boundary_after = torch.tensor([-0.7, 0.19, 0.0], dtype=DTYPE)
    collision_position = torch.tensor([2.0, 0.0, 1.0], dtype=DTYPE)
    boundary_before_positions = _drag_positions(
        before_times[0],
        position_at_zero=collision_position,
        velocity_at_zero=boundary_before,
        drag=drag,
    ).unsqueeze(0)
    boundary_after_positions = _drag_positions(
        after_times[0],
        position_at_zero=collision_position,
        velocity_at_zero=boundary_after,
        drag=drag,
    ).unsqueeze(0)
    belief, boundary_updates = estimator.update_boundary_collision_positions(
        belief,
        BoundaryCollisionPositionEvidence(
            object_id,
            torch.tensor([[-1.0, 0.0, 0.0]], dtype=DTYPE),
            boundary_before_positions,
            before_times,
            boundary_after_positions,
            after_times,
            action_time,
        ),
    )

    assert drag_update.accepted.item() and mass_update.accepted.item()
    assert all(update.accepted.item() for update in boundary_updates)
    torch.testing.assert_close(
        belief.objects.drag[0, 0, 0], torch.tensor(drag, dtype=DTYPE), rtol=0.025, atol=0.0
    )
    torch.testing.assert_close(
        belief.objects.mass[0, 0, 0], torch.tensor(2.0, dtype=DTYPE), rtol=0.025, atol=0.0
    )
    torch.testing.assert_close(
        belief.objects.restitution[0, 0, 0], torch.tensor(0.7, dtype=DTYPE), rtol=0.025, atol=0.0
    )
    torch.testing.assert_close(
        belief.objects.friction[0, 0, 0], torch.tensor(0.3, dtype=DTYPE), rtol=0.025, atol=0.0
    )


def test_inconsistent_impulse_is_rejected_without_mutating_belief() -> None:
    belief = _belief()
    estimator = OnlineRigidParameterEstimator(gain=1.0)
    updated, diagnostic = estimator.update_known_impulse(
        belief,
        KnownImpulseEvidence(
            object_id=torch.tensor([11]),
            velocity_before=torch.zeros(1, 3, dtype=DTYPE),
            velocity_after=torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE),
            impulse_world=torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE),
        ),
    )

    assert not diagnostic.accepted.item()
    torch.testing.assert_close(updated.objects.log_mass, belief.objects.log_mass)
    torch.testing.assert_close(updated.objects.slow_log_variance, belief.objects.slow_log_variance)


def test_bad_public_position_fit_is_rejected_without_contracting_uncertainty() -> None:
    belief = _belief()
    estimator = OnlineRigidParameterEstimator(gain=1.0)
    positions = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [-0.2, 0.3, 1.0], [0.4, -0.5, 1.0]]],
        dtype=DTYPE,
    )
    timestamps = torch.tensor([[0.0, 0.25, 0.50, 0.75]], dtype=DTYPE)

    updated, diagnostic = estimator.update_free_motion_positions(
        belief,
        FreeMotionPositionEvidence(torch.tensor([11]), positions, timestamps),
        maximum_fit_residual_m=1.0e-5,
    )

    assert not diagnostic.accepted.item()
    torch.testing.assert_close(updated.objects.log_drag, belief.objects.log_drag)
    torch.testing.assert_close(updated.objects.slow_log_variance, belief.objects.slow_log_variance)


def test_public_position_fit_bound_must_be_positive_and_finite() -> None:
    belief = _belief()
    estimator = OnlineRigidParameterEstimator()
    evidence = FreeMotionPositionEvidence(
        torch.tensor([11]),
        torch.zeros(1, 4, 3, dtype=DTYPE),
        torch.tensor([[0.0, 0.1, 0.2, 0.3]], dtype=DTYPE),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        estimator.update_free_motion_positions(
            belief,
            evidence,
            maximum_fit_residual_m=float("nan"),
        )
