from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode, WorldBelief
from world_model.dynamics import AnalyticFreeMotionDynamics, WorldImpulseAction


def _active_belief(
    *,
    batch_size: int = 1,
    max_objects: int = 3,
    timestamp: float = 0.0,
    drag: float = 0.2,
    dtype: torch.dtype = torch.float64,
) -> WorldBelief:
    belief = BeliefFactory(max_objects=max_objects).create(
        batch_size=batch_size,
        timestamp=timestamp,
        dtype=dtype,
        gravity=(0.0, 0.0, 0.0),
    )
    objects = belief.objects.clone()
    objects.active[:, :2] = True
    objects.object_id[:, 0] = 7
    objects.object_id[:, 1] = 41
    objects.position[:, 0] = belief.timestamp.new_tensor([0.2, -0.3, 0.5])
    objects.position[:, 1] = belief.timestamp.new_tensor([-0.8, 0.6, 0.1])
    objects.velocity[:, 0] = belief.timestamp.new_tensor([0.4, -0.2, 0.3])
    objects.velocity[:, 1] = belief.timestamp.new_tensor([-0.1, 0.5, -0.4])
    objects.log_drag.fill_(math.log(drag))
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., MotionMode.FREE] = 4.0
    return belief.replace(objects=objects)


def _action(
    belief: WorldBelief,
    *,
    timestamp: float | torch.Tensor = 0.5,
    object_id: int | torch.Tensor = 7,
    impulse: tuple[float, float, float] | torch.Tensor = (0.8, -0.4, 0.2),
) -> WorldImpulseAction:
    if not isinstance(timestamp, torch.Tensor):
        timestamp = belief.timestamp.new_full((belief.batch_size,), timestamp)
    if not isinstance(object_id, torch.Tensor):
        object_id = torch.full(
            (belief.batch_size,),
            object_id,
            device=belief.device,
            dtype=torch.int64,
        )
    if not isinstance(impulse, torch.Tensor):
        impulse = belief.timestamp.new_tensor(impulse).expand(belief.batch_size, -1).clone()
    return WorldImpulseAction(
        timestamp=timestamp,
        object_id=object_id,
        impulse_world=impulse,
    )


def _assert_trajectory_tensors_equal(actual, expected) -> None:
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
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
        else:
            assert torch.equal(actual_value, expected_value), name
    assert actual.auxiliary.keys() == expected.auxiliary.keys()
    for name in actual.auxiliary:
        assert torch.equal(actual.auxiliary[name], expected.auxiliary[name]), name


def test_world_impulse_action_is_frozen_and_resolves_persistent_id() -> None:
    belief = _active_belief()
    action = _action(belief)

    resolved = action.validate_for(belief)

    assert resolved.tolist() == [[True, False, False]]
    with pytest.raises(FrozenInstanceError):
        action.frame = "camera"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("timestamp", 0.5, TypeError, "timestamp must be a torch.Tensor"),
        ("object_id", 7, TypeError, "object_id must be a torch.Tensor"),
        ("impulse_world", (1.0, 0.0, 0.0), TypeError, "impulse_world must be a torch.Tensor"),
        ("timestamp", torch.tensor(0.5, dtype=torch.float64), ValueError, r"shape \[B\]"),
        ("object_id", torch.tensor([[7]], dtype=torch.int64), ValueError, r"shape \[B\]"),
        ("impulse_world", torch.tensor([1.0, 0.0, 0.0]), ValueError, r"shape \[B,3\]"),
        ("timestamp", torch.tensor([0.5], dtype=torch.float32), TypeError, "dtype"),
        ("impulse_world", torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), TypeError, "dtype"),
        ("object_id", torch.tensor([7], dtype=torch.int32), TypeError, "torch.int64"),
        ("timestamp", torch.tensor([float("nan")], dtype=torch.float64), ValueError, "finite"),
        (
            "impulse_world",
            torch.tensor([[float("inf"), 0.0, 0.0]], dtype=torch.float64),
            ValueError,
            "finite",
        ),
    ],
)
def test_action_rejects_wrong_tensor_contract(
    field: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    belief = _active_belief()
    values = {
        "timestamp": belief.timestamp.new_tensor([0.5]),
        "object_id": torch.tensor([7], dtype=torch.int64),
        "impulse_world": belief.timestamp.new_tensor([[1.0, 0.0, 0.0]]),
    }
    values[field] = value
    action = WorldImpulseAction(**values)  # type: ignore[arg-type]

    with pytest.raises(error, match=match):
        action.validate_for(belief)


def test_action_rejects_non_world_frame_and_device_mismatch() -> None:
    belief = _active_belief()
    wrong_frame = WorldImpulseAction(
        timestamp=belief.timestamp.new_tensor([0.5]),
        object_id=torch.tensor([7], dtype=torch.int64),
        impulse_world=belief.timestamp.new_tensor([[1.0, 0.0, 0.0]]),
        frame="camera",  # type: ignore[arg-type]
    )
    wrong_device = WorldImpulseAction(
        timestamp=torch.empty(1, dtype=belief.dtype, device="meta"),
        object_id=torch.tensor([7], dtype=torch.int64),
        impulse_world=belief.timestamp.new_tensor([[1.0, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="frame must be 'world'"):
        wrong_frame.validate_for(belief)
    with pytest.raises(ValueError, match="device"):
        wrong_device.validate_for(belief)


def test_action_time_is_nondifferentiable_strictly_future_and_inside_horizon() -> None:
    belief = _active_belief(timestamp=1.0)
    dynamics = AnalyticFreeMotionDynamics()
    requires_gradient = _action(
        belief,
        timestamp=torch.tensor([1.2], dtype=belief.dtype, requires_grad=True),
    )

    with pytest.raises(ValueError, match="must not require gradients"):
        requires_gradient.validate_for(belief)
    for timestamp in (0.9, 1.0):
        with pytest.raises(ValueError, match="strictly after"):
            _action(belief, timestamp=timestamp).validate_for(belief)
    with pytest.raises(ValueError, match="rollout horizon"):
        dynamics.rollout(belief, [0.1], action=_action(belief, timestamp=1.2))
    with pytest.raises(ValueError, match="at least one query"):
        dynamics.rollout(belief, [], action=_action(belief, timestamp=1.2))
    normalized = dynamics.validate_action_rollout(belief, [], None)
    assert normalized.shape == (1, 0)


@pytest.mark.parametrize("object_id", [-1, 999])
def test_action_rejects_nonpersistent_or_unknown_id(object_id: int) -> None:
    belief = _active_belief()

    with pytest.raises(ValueError, match="exactly one active persistent"):
        _action(belief, object_id=object_id).validate_for(belief)


def test_action_rejects_inactive_duplicate_and_nonfree_targets() -> None:
    belief = _active_belief()
    inactive_objects = belief.objects.clone()
    inactive_objects.active[0, 0] = False
    inactive = belief.replace(objects=inactive_objects)
    with pytest.raises(ValueError, match="exactly one active persistent"):
        _action(inactive).validate_for(inactive)

    duplicate_objects = belief.objects.clone()
    duplicate_objects.object_id[0, 1] = 7
    duplicate = belief.replace(objects=duplicate_objects)
    with pytest.raises(ValueError, match="exactly one active persistent"):
        _action(duplicate).validate_for(duplicate)

    for mode in (MotionMode.REMOVED, MotionMode.SLEEPING, MotionMode.EXTERNALLY_ACTUATED):
        objects = belief.objects.clone()
        objects.motion_mode_logits[0, 0].fill_(-4.0)
        objects.motion_mode_logits[0, 0, mode] = 4.0
        nonfree = belief.replace(objects=objects)
        with pytest.raises(ValueError, match="FREE motion mode"):
            _action(nonfree).validate_for(nonfree)


def test_persistent_id_targeting_survives_per_batch_slot_permutation() -> None:
    belief = _active_belief(batch_size=2)
    objects = belief.objects.clone()
    for name in (
        "object_id",
        "active",
        "existence_logit",
        "position",
        "velocity",
        "orientation",
        "angular_velocity",
        "geometry",
        "appearance",
        "residual_dynamics",
        "modal_state",
        "modal_frequency",
        "modal_decay_raw",
        "log_mass",
        "restitution_logit",
        "log_drag",
        "friction_logit",
        "motion_mode_logits",
        "visibility_logit",
        "age_steps",
        "missed_steps",
        "fast_log_variance",
        "slow_log_variance",
        "parameter_memory",
    ):
        value = getattr(objects, name)
        value[1] = value[1, torch.tensor([1, 0, 2])]
    belief = belief.replace(objects=objects)
    action = _action(
        belief,
        object_id=torch.tensor([7, 7], dtype=torch.int64),
        impulse=belief.timestamp.new_tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    baseline = AnalyticFreeMotionDynamics().rollout(belief, [0.5])
    acted = AnalyticFreeMotionDynamics().rollout(belief, [0.5], action=action)
    delta = acted.velocities - baseline.velocities

    assert action.validate_for(belief).tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    torch.testing.assert_close(delta[0, 0, 0], belief.timestamp.new_tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(delta[1, 0, 1], belief.timestamp.new_tensor([1.0, 0.0, 0.0]))
    assert torch.equal(delta[0, 0, 1:], torch.zeros_like(delta[0, 0, 1:]))
    assert torch.equal(delta[1, 0, [0, 2]], torch.zeros_like(delta[1, 0, [0, 2]]))


def test_right_continuous_drag_response_and_duplicate_query_event_once() -> None:
    belief = _active_belief(drag=0.4)
    dynamics = AnalyticFreeMotionDynamics()
    impulse = belief.timestamp.new_tensor([[0.8, -0.4, 0.2]])
    action = _action(belief, impulse=impulse)
    queries = belief.timestamp.new_tensor([0.25, 0.5, 0.5, 0.9])
    baseline = dynamics.rollout(belief, queries)
    acted = dynamics.rollout(belief, queries, action=action)
    position_delta = acted.positions - baseline.positions
    velocity_delta = acted.velocities - baseline.velocities

    assert torch.equal(position_delta[:, 0], torch.zeros_like(position_delta[:, 0]))
    assert torch.equal(velocity_delta[:, 0], torch.zeros_like(velocity_delta[:, 0]))
    assert torch.equal(position_delta[:, 1], torch.zeros_like(position_delta[:, 1]))
    torch.testing.assert_close(velocity_delta[0, 1, 0], impulse[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(velocity_delta[0, 2, 0], impulse[0], rtol=0.0, atol=0.0)

    elapsed = 0.4
    decay = math.exp(-0.4 * elapsed)
    displacement = (1.0 - decay) / 0.4
    torch.testing.assert_close(
        velocity_delta[0, 3, 0],
        impulse[0] * decay,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        position_delta[0, 3, 0],
        impulse[0] * displacement,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.equal(position_delta[..., 1:, :], torch.zeros_like(position_delta[..., 1:, :]))
    assert torch.equal(velocity_delta[..., 1:, :], torch.zeros_like(velocity_delta[..., 1:, :]))

    applied = acted.auxiliary["known_action_applied"]
    assert applied.dtype is torch.bool
    assert applied.shape == (1, 4, 3)
    assert applied.sum().item() == 1
    assert applied[0, 1, 0]
    known_impulse = acted.auxiliary["known_impulse_world"]
    assert known_impulse.shape == (1, 4, 3, 3)
    torch.testing.assert_close(known_impulse[0, 1, 0], impulse[0])
    assert torch.count_nonzero(known_impulse).item() == 3
    event_modes = acted.event_logits.argmax(dim=-1)
    assert event_modes[0, 1, 0].item() == MotionMode.EXTERNALLY_ACTUATED
    assert torch.count_nonzero(event_modes == MotionMode.EXTERNALLY_ACTUATED).item() == 1
    assert (acted.motion_mode_logits.argmax(dim=-1)[..., :2] == MotionMode.FREE).all()
    assert torch.equal(acted.fast_log_variance, baseline.fast_log_variance)


def test_small_drag_branch_is_exact_constant_velocity_impulse_response() -> None:
    belief = _active_belief(drag=math.exp(-16.0))
    action = _action(belief, impulse=(0.3, -0.6, 0.9))
    dynamics = AnalyticFreeMotionDynamics()
    baseline = dynamics.rollout(belief, [0.5, 1.2])
    acted = dynamics.rollout(belief, [0.5, 1.2], action=action)
    jump = action.impulse_world[0]

    torch.testing.assert_close(
        acted.velocities[0, 1, 0] - baseline.velocities[0, 1, 0],
        jump,
        rtol=0.0,
        atol=2e-16,
    )
    torch.testing.assert_close(
        acted.positions[0, 1, 0] - baseline.positions[0, 1, 0],
        jump * 0.7,
        rtol=1e-14,
        atol=1e-14,
    )


def test_unequal_mass_has_inverse_velocity_response_and_target_only_effect() -> None:
    belief = _active_belief(batch_size=2)
    objects = belief.objects.clone()
    objects.log_mass[0, 0, 0] = math.log(2.0)
    objects.log_mass[1, 0, 0] = math.log(4.0)
    belief = belief.replace(objects=objects)
    impulse = belief.timestamp.new_tensor([[1.2, -0.8, 0.4], [1.2, -0.8, 0.4]])
    action = _action(belief, impulse=impulse)
    dynamics = AnalyticFreeMotionDynamics()
    baseline = dynamics.rollout(belief, [0.5])
    acted = dynamics.rollout(belief, [0.5], action=action)
    jump = acted.velocities[:, 0, 0] - baseline.velocities[:, 0, 0]

    torch.testing.assert_close(jump[0], impulse[0] / 2.0)
    torch.testing.assert_close(jump[1], impulse[1] / 4.0)
    torch.testing.assert_close(jump * belief.objects.mass[:, 0], impulse)
    assert torch.equal(
        acted.velocities[:, :, 1:],
        baseline.velocities[:, :, 1:],
    )


def test_zero_impulse_changes_no_state_or_event_and_emits_zero_auxiliaries() -> None:
    belief = _active_belief()
    dynamics = AnalyticFreeMotionDynamics()
    baseline = dynamics.rollout(belief, [0.5, 0.9])
    acted = dynamics.rollout(belief, [0.5, 0.9], action=_action(belief, impulse=(0.0, 0.0, 0.0)))

    for name in ("positions", "velocities", "fast_log_variance", "event_logits"):
        assert torch.equal(getattr(acted, name), getattr(baseline, name))
    assert not acted.auxiliary["known_action_applied"].any()
    assert torch.count_nonzero(acted.auxiliary["known_impulse_world"]).item() == 0


def test_action_rollout_honours_event_and_auxiliary_return_controls() -> None:
    belief = _active_belief()
    dynamics = AnalyticFreeMotionDynamics()
    action = _action(belief)

    compact = dynamics.rollout(
        belief,
        [0.5, 0.8],
        action=action,
        return_events=False,
        return_auxiliary=False,
    )
    selected = dynamics.rollout(
        belief,
        [0.5, 0.8],
        action=action,
        auxiliary_names=("known_action_applied",),
    )

    assert compact.event_logits is None
    assert compact.auxiliary == {}
    assert selected.auxiliary.keys() == {"known_action_applied"}
    with pytest.raises(KeyError, match="unknown"):
        dynamics.rollout(belief, [0.5], action=action, auxiliary_names=("unknown",))


def test_direct_and_composed_predictions_apply_action_once() -> None:
    belief = _active_belief(drag=0.3)
    dynamics = AnalyticFreeMotionDynamics()
    action = _action(belief, timestamp=0.4)
    direct = dynamics.predict(belief, 1.0, action=action)

    before = dynamics.predict(belief, 0.2)
    across = dynamics.predict(before, 0.8, action=action)
    at_boundary = dynamics.predict(belief, 0.4, action=action)
    after = dynamics.predict(at_boundary, 0.6)

    torch.testing.assert_close(across.objects.position, direct.objects.position)
    torch.testing.assert_close(across.objects.velocity, direct.objects.velocity)
    torch.testing.assert_close(after.objects.position, direct.objects.position)
    torch.testing.assert_close(after.objects.velocity, direct.objects.velocity)
    assert direct.objects.mode[0, 0].item() == MotionMode.FREE
    with pytest.raises(ValueError, match="strictly after"):
        dynamics.predict(at_boundary, 0.6, action=action)


def test_impulse_gradients_have_exact_preaction_and_boundary_jacobians() -> None:
    belief = _active_belief(drag=0.25)
    objects = belief.objects
    source_position = objects.position.detach().clone().requires_grad_(True)
    source_velocity = objects.velocity.detach().clone().requires_grad_(True)
    source_log_drag = objects.log_drag.detach().clone().requires_grad_(True)
    source_log_mass = objects.log_mass.detach().clone().requires_grad_(True)
    belief = belief.replace(
        objects=objects.replace(
            position=source_position,
            velocity=source_velocity,
            log_drag=source_log_drag,
            log_mass=source_log_mass,
        )
    )
    impulse = belief.timestamp.new_tensor([[0.6, -0.3, 0.9]]).requires_grad_(True)
    action = _action(belief, impulse=impulse)
    trajectory = AnalyticFreeMotionDynamics().rollout(belief, [0.25, 0.5, 0.9], action=action)

    preaction_gradient = torch.autograd.grad(
        trajectory.positions[:, 0].sum() + trajectory.velocities[:, 0].sum(),
        impulse,
        retain_graph=True,
    )[0]
    boundary_position_gradient = torch.autograd.grad(
        trajectory.positions[:, 1, 0].sum(),
        impulse,
        retain_graph=True,
    )[0]
    boundary_velocity_gradient = torch.autograd.grad(
        trajectory.velocities[:, 1, 0].sum(),
        impulse,
        retain_graph=True,
    )[0]
    post_loss = trajectory.positions[:, -1, 0].sum() + trajectory.velocities[:, -1, 0].sum()
    post_gradients = torch.autograd.grad(
        post_loss,
        (impulse, source_position, source_velocity, source_log_drag, source_log_mass),
    )

    assert torch.equal(preaction_gradient, torch.zeros_like(preaction_gradient))
    assert torch.equal(boundary_position_gradient, torch.zeros_like(boundary_position_gradient))
    torch.testing.assert_close(
        boundary_velocity_gradient,
        belief.objects.mass[:, 0].reciprocal().expand_as(impulse),
    )
    for gradient in post_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_action_none_is_exact_legacy_value_and_gradient_path() -> None:
    belief = _active_belief(drag=0.35)
    objects = belief.objects
    source_position = objects.position.detach().clone().requires_grad_(True)
    source_velocity = objects.velocity.detach().clone().requires_grad_(True)
    source_log_drag = objects.log_drag.detach().clone().requires_grad_(True)
    belief = belief.replace(
        objects=objects.replace(
            position=source_position,
            velocity=source_velocity,
            log_drag=source_log_drag,
        )
    )
    dynamics = AnalyticFreeMotionDynamics()
    implicit = dynamics.rollout(belief, [0.1, 0.4, 0.8])
    explicit = dynamics.rollout(belief, [0.1, 0.4, 0.8], action=None)
    _assert_trajectory_tensors_equal(explicit, implicit)

    implicit_loss = implicit.positions.sum() + implicit.velocities.sum()
    explicit_loss = explicit.positions.sum() + explicit.velocities.sum()
    implicit_gradients = torch.autograd.grad(
        implicit_loss,
        (source_position, source_velocity, source_log_drag),
        retain_graph=True,
    )
    explicit_gradients = torch.autograd.grad(
        explicit_loss,
        (source_position, source_velocity, source_log_drag),
    )
    for implicit_gradient, explicit_gradient in zip(
        implicit_gradients,
        explicit_gradients,
        strict=True,
    ):
        assert torch.equal(implicit_gradient, explicit_gradient)

    implicit_step = dynamics.predict_step(belief, 0.4)
    explicit_step = dynamics.predict_step(belief, 0.4, action=None)
    assert torch.equal(implicit_step.belief.objects.position, explicit_step.belief.objects.position)
    assert torch.equal(implicit_step.belief.objects.velocity, explicit_step.belief.objects.velocity)
    assert torch.equal(implicit_step.event_logits, explicit_step.event_logits)
    assert implicit_step.auxiliary == explicit_step.auxiliary == {}
    assert dynamics.state_dict() == {}


def test_action_rollout_does_not_mutate_source_belief() -> None:
    belief = _active_belief()
    original = belief.clone()

    AnalyticFreeMotionDynamics().rollout(belief, [0.5, 0.8], action=_action(belief))

    for name in (
        "timestamp",
        "gravity",
        "global_code",
        "global_log_variance",
        "next_object_id",
    ):
        assert torch.equal(getattr(belief, name), getattr(original, name)), name
    for name in vars(belief.objects):
        assert torch.equal(getattr(belief.objects, name), getattr(original.objects, name)), name
