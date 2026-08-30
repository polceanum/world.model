from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from world_model.belief import (
    BeliefFactory,
    MotionMode,
    ObjectBeliefTensor,
    fast_packing_map,
    slow_packing_map,
)
from world_model.dynamics import AnalyticFreeMotionDynamics
from world_model.runtime import OnlineWorldModel
from world_model.utils.config import load_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def _uncertain_belief(
    *,
    dtype: torch.dtype = torch.float64,
    batch_size: int = 2,
    max_objects: int = 3,
):
    belief = BeliefFactory(max_objects=max_objects, modal_count=0).create(
        batch_size=batch_size,
        dtype=dtype,
        gravity=(0.15, -0.3, 0.08),
    )
    objects = belief.objects.clone()
    objects.active[:, :2] = True
    objects.object_id[:, :2] = torch.tensor(
        [[0, 1]],
        dtype=objects.object_id.dtype,
    ).expand(batch_size, -1)
    position = torch.tensor(
        [
            [[0.2, 1.1, -0.4], [-0.3, 0.8, 0.25], [0.1, 0.2, 0.3]],
            [[-0.6, 0.9, 0.1], [0.4, 1.2, -0.2], [-0.1, 0.3, 0.2]],
        ],
        dtype=dtype,
    )[:batch_size, :max_objects]
    velocity = torch.tensor(
        [
            [[0.7, -0.1, 0.25], [-0.2, 0.35, 0.5], [0.9, 0.1, -0.2]],
            [[-0.45, 0.2, 0.3], [0.55, -0.25, -0.15], [0.2, -0.4, 0.1]],
        ],
        dtype=dtype,
    )[:batch_size, :max_objects]
    drag = torch.tensor(
        [[0.2, 0.08, 0.12], [0.31, 0.14, 0.09]],
        dtype=dtype,
    )[:batch_size, :max_objects, None]
    fast_variance = torch.full_like(objects.fast_log_variance, 0.04)
    fast_variance[..., :3] = torch.tensor(
        [0.012, 0.018, 0.027],
        dtype=dtype,
    )
    fast_variance[..., 3:6] = torch.tensor(
        [0.021, 0.032, 0.044],
        dtype=dtype,
    )
    slow_variance = torch.full_like(objects.slow_log_variance, 0.05)
    drag_slice = slow_packing_map(objects)["log_drag"]
    slow_variance[..., drag_slice] = (
        torch.tensor(
            [0.035, 0.07, 0.11],
            dtype=dtype,
        )[:max_objects]
        .view(1, max_objects, 1)
        .expand(batch_size, -1, -1)
    )
    objects = objects.replace(
        position=position,
        velocity=velocity,
        log_drag=drag.log(),
        fast_log_variance=fast_variance.log(),
        slow_log_variance=slow_variance.log(),
    )
    return belief.replace(objects=objects)


def _explicit_diagonal_variance(belief, elapsed: torch.Tensor):
    objects = belief.objects
    fast = fast_packing_map(objects)
    slow = slow_packing_map(objects)
    time = elapsed[:, :, None, None]
    drag = objects.log_drag.exp()[:, None]
    decay = torch.exp(-drag * time)
    position_velocity_coefficient = -torch.expm1(-drag * time) / drag
    position_acceleration_coefficient = time / drag - (-torch.expm1(-drag * time)) / drag.square()
    a_log_derivative = time * decay - position_velocity_coefficient
    b_log_derivative = -position_acceleration_coefficient - a_log_derivative / drag
    position_drag_jacobian = (
        a_log_derivative * objects.velocity[:, None]
        + b_log_derivative * belief.gravity[:, None, None]
    )
    velocity_drag_jacobian = (
        -drag * time * decay * objects.velocity[:, None]
        + a_log_derivative * belief.gravity[:, None, None]
    )
    position_variance = objects.fast_log_variance[..., fast["position"]].exp()[:, None]
    velocity_variance = objects.fast_log_variance[..., fast["velocity"]].exp()[:, None]
    drag_variance = objects.slow_log_variance[..., slow["log_drag"]].exp()[:, None]
    return (
        position_variance
        + position_velocity_coefficient.square() * velocity_variance
        + position_drag_jacobian.square() * drag_variance,
        decay.square() * velocity_variance + velocity_drag_jacobian.square() * drag_variance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_opt_in_rollout_matches_direct_anchor_diagonal_jacobian(dtype: torch.dtype) -> None:
    belief = _uncertain_belief(dtype=dtype)
    dynamics = AnalyticFreeMotionDynamics(propagate_drag_uncertainty=True)
    elapsed = torch.tensor(
        [[0.0, 0.17, 0.63], [0.0, 0.29, 0.91]],
        dtype=dtype,
    )

    trajectory = dynamics.rollout(belief, elapsed)
    expected_position, expected_velocity = _explicit_diagonal_variance(belief, elapsed)
    fast = fast_packing_map(belief.objects)
    actual = trajectory.fast_log_variance.exp()
    tolerance = {torch.float32: (2e-5, 2e-6), torch.float64: (2e-12, 2e-13)}[dtype]
    torch.testing.assert_close(
        actual[:, 1:, :2, fast["position"]],
        expected_position[:, 1:, :2],
        rtol=tolerance[0],
        atol=tolerance[1],
    )
    torch.testing.assert_close(
        actual[:, 1:, :2, fast["velocity"]],
        expected_velocity[:, 1:, :2],
        rtol=tolerance[0],
        atol=tolerance[1],
    )
    torch.testing.assert_close(
        trajectory.fast_log_variance[:, 0],
        belief.objects.fast_log_variance,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        trajectory.fast_log_variance[:, :, 2],
        belief.objects.fast_log_variance[:, None, 2].expand(-1, elapsed.shape[1], -1),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        trajectory.fast_log_variance[..., 6:],
        belief.objects.fast_log_variance[:, None, :, 6:].expand(
            -1,
            elapsed.shape[1],
            -1,
            -1,
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(trajectory.fast_log_variance).all()


def test_query_partition_and_single_prediction_use_the_same_anchor_marginals() -> None:
    belief = _uncertain_belief(dtype=torch.float32)
    belief = belief.replace(timestamp=torch.full_like(belief.timestamp, 1.0e6))
    dynamics = AnalyticFreeMotionDynamics(propagate_drag_uncertainty=True)
    query_times = [0.08, 0.31, 0.77]
    requested_offsets = (
        belief.timestamp.new_tensor(query_times)
        .unsqueeze(0)
        .expand(
            belief.batch_size,
            -1,
        )
    )

    joint = dynamics.rollout(belief, query_times)
    separate = [dynamics.rollout(belief, [query_time]) for query_time in query_times]
    predicted = [dynamics.predict(belief, query_time) for query_time in query_times]
    expected_position, expected_velocity = _explicit_diagonal_variance(
        belief,
        requested_offsets,
    )
    fast = fast_packing_map(belief.objects)
    torch.testing.assert_close(
        joint.fast_log_variance[:, :, :2, fast["position"]].exp(),
        expected_position[:, :, :2],
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        joint.fast_log_variance[:, :, :2, fast["velocity"]].exp(),
        expected_velocity[:, :, :2],
        rtol=2e-5,
        atol=2e-6,
    )

    for index, (trajectory, endpoint) in enumerate(zip(separate, predicted, strict=True)):
        torch.testing.assert_close(
            joint.fast_log_variance[:, index],
            trajectory.fast_log_variance[:, 0],
        )
        torch.testing.assert_close(joint.positions[:, index], trajectory.positions[:, 0])
        torch.testing.assert_close(joint.velocities[:, index], trajectory.velocities[:, 0])
        torch.testing.assert_close(
            joint.fast_log_variance[:, index],
            endpoint.objects.fast_log_variance,
        )


def test_drag_uncertainty_is_opt_in_and_preserves_the_parameterless_legacy_path() -> None:
    belief = _uncertain_belief()
    legacy = AnalyticFreeMotionDynamics()
    enabled = AnalyticFreeMotionDynamics(propagate_drag_uncertainty=True)

    legacy_trajectory = legacy.rollout(belief, [0.0, 0.4])
    enabled_trajectory = enabled.rollout(belief, [0.0, 0.4])
    expected_legacy = belief.objects.fast_log_variance[:, None].expand(
        -1,
        2,
        -1,
        -1,
    )

    torch.testing.assert_close(
        legacy_trajectory.fast_log_variance,
        expected_legacy,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(enabled_trajectory.positions, legacy_trajectory.positions)
    torch.testing.assert_close(enabled_trajectory.velocities, legacy_trajectory.velocities)
    torch.testing.assert_close(enabled_trajectory.event_logits, legacy_trajectory.event_logits)
    assert not torch.equal(
        enabled_trajectory.fast_log_variance[:, 1, :2, :6],
        legacy_trajectory.fast_log_variance[:, 1, :2, :6],
    )
    for dynamics in (legacy, enabled):
        assert not tuple(dynamics.parameters())
        assert not tuple(dynamics.buffers())
        assert dynamics.state_dict() == {}


def test_small_drag_inactive_and_zero_time_paths_remain_finite_identities() -> None:
    for dtype in (torch.float32, torch.float64):
        belief = _uncertain_belief(dtype=dtype)
        objects = belief.objects.clone()
        objects.log_drag[:, 0].fill_(math.log(2.0e-5))
        objects.log_drag[:, 1].fill_(-16.0)
        objects.motion_mode_logits[:, 1].fill_(-4.0)
        objects.motion_mode_logits[:, 1, MotionMode.SLEEPING] = 4.0
        belief = belief.replace(objects=objects)
        trajectory = AnalyticFreeMotionDynamics(propagate_drag_uncertainty=True).rollout(
            belief, [0.0, 0.05, 0.5]
        )

        assert torch.isfinite(trajectory.fast_log_variance).all()
        torch.testing.assert_close(
            trajectory.fast_log_variance[:, 0],
            objects.fast_log_variance,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            trajectory.fast_log_variance[:, :, 1],
            objects.fast_log_variance[:, None, 1].expand(-1, 3, -1),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            trajectory.fast_log_variance[:, :, 2],
            objects.fast_log_variance[:, None, 2].expand(-1, 3, -1),
            rtol=0.0,
            atol=0.0,
        )


def _permute_objects(objects: ObjectBeliefTensor, permutation: torch.Tensor):
    updates = {}
    object_shape = objects.object_id.shape
    for item in fields(objects):
        value = getattr(objects, item.name)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[:2] == object_shape:
            updates[item.name] = value[:, permutation]
    return objects.replace(**updates)


def test_rollout_is_permutation_equivariant_and_has_exact_cross_batch_vjps() -> None:
    belief = _uncertain_belief(max_objects=3)
    dynamics = AnalyticFreeMotionDynamics(propagate_drag_uncertainty=True)
    permutation = torch.tensor([1, 2, 0])
    permuted = belief.replace(objects=_permute_objects(belief.objects, permutation))

    original = dynamics.rollout(belief, [0.23, 0.61])
    reordered = dynamics.rollout(permuted, [0.23, 0.61])
    torch.testing.assert_close(
        reordered.fast_log_variance,
        original.fast_log_variance[:, :, permutation],
    )
    torch.testing.assert_close(reordered.positions, original.positions[:, :, permutation])
    torch.testing.assert_close(reordered.velocities, original.velocities[:, :, permutation])

    velocity = belief.objects.velocity.clone().requires_grad_()
    log_drag = belief.objects.log_drag.clone().requires_grad_()
    fast_log_variance = belief.objects.fast_log_variance.clone().requires_grad_()
    slow_log_variance = belief.objects.slow_log_variance.clone().requires_grad_()
    gravity = belief.gravity.clone().requires_grad_()
    query_times = belief.timestamp.new_tensor([[0.37], [0.52]]).requires_grad_()
    differentiable = belief.replace(
        gravity=gravity,
        objects=belief.objects.replace(
            velocity=velocity,
            log_drag=log_drag,
            fast_log_variance=fast_log_variance,
            slow_log_variance=slow_log_variance,
        ),
    )
    selected = dynamics.rollout(differentiable, query_times).fast_log_variance[0, 0, 0, :6].sum()
    gradients = torch.autograd.grad(
        selected,
        (
            velocity,
            log_drag,
            fast_log_variance,
            slow_log_variance,
            gravity,
            query_times,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient[0].abs().sum() > 0
        torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]))
    for gradient in gradients[:4]:
        torch.testing.assert_close(gradient[0, 1:], torch.zeros_like(gradient[0, 1:]))


def test_drag_variance_scale_has_the_expected_linear_variance_effect() -> None:
    belief = _uncertain_belief(batch_size=1, max_objects=2)
    slow = slow_packing_map(belief.objects)
    fast = fast_packing_map(belief.objects)
    low_slow_log_variance = belief.objects.slow_log_variance.clone()
    low_slow_log_variance[..., slow["log_drag"]] = math.log(0.02)
    high_slow_log_variance = low_slow_log_variance.clone()
    high_slow_log_variance[..., slow["log_drag"]] = math.log(0.08)
    low = belief.replace(objects=belief.objects.replace(slow_log_variance=low_slow_log_variance))
    high = belief.replace(objects=belief.objects.replace(slow_log_variance=high_slow_log_variance))
    dynamics = AnalyticFreeMotionDynamics(propagate_drag_uncertainty=True)
    low_variance = dynamics.rollout(low, [0.6]).fast_log_variance.exp()[0, 0, 0]
    high_variance = dynamics.rollout(high, [0.6]).fast_log_variance.exp()[0, 0, 0]
    zero_drag_variance = low_slow_log_variance.clone()
    zero_drag_variance[..., slow["log_drag"]] = -30.0
    almost_zero = belief.replace(
        objects=belief.objects.replace(slow_log_variance=zero_drag_variance)
    )
    baseline = dynamics.rollout(almost_zero, [0.6]).fast_log_variance.exp()[0, 0, 0]
    tiny_drag_variance = math.exp(-30.0)
    expected_ratio = (0.08 - tiny_drag_variance) / (0.02 - tiny_drag_variance)

    for state_slice in (fast["position"], fast["velocity"]):
        low_increment = low_variance[state_slice] - baseline[state_slice]
        high_increment = high_variance[state_slice] - baseline[state_slice]
        sensitive = low_increment.abs() > 1.0e-12
        torch.testing.assert_close(
            high_increment[sensitive] / low_increment[sensitive],
            torch.full_like(high_increment[sensitive], expected_ratio),
            rtol=2e-10,
            atol=2e-10,
        )


def test_from_config_enables_only_the_temporal_drag_runtime() -> None:
    base = load_config(CONFIG_DIR / "rgbd_two_visible_orbital_camera_cpu.yaml")
    disabled = OnlineWorldModel.from_config(base, device="cpu")
    enabled_config = replace(
        base,
        simulator=replace(base.simulator, drag_range=(0.03, 0.28)),
        model=replace(
            base.model,
            rgbd=replace(
                base.model.rgbd,
                temporal_drag_estimation_enabled=True,
            ),
        ),
    )
    enabled = OnlineWorldModel.from_config(enabled_config, device="cpu")

    assert isinstance(disabled.dynamics, AnalyticFreeMotionDynamics)
    assert isinstance(enabled.dynamics, AnalyticFreeMotionDynamics)
    assert not disabled.dynamics.propagate_drag_uncertainty
    assert enabled.dynamics.propagate_drag_uncertainty
