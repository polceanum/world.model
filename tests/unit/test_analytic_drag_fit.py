from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import world_model.identification.analytic_drag as analytic_drag_module
from world_model.dynamics import fit_free_motion, free_motion_position_velocity
from world_model.identification import AnalyticDragFitResult, fit_free_motion_with_drag


def _history(
    anchor_position: torch.Tensor,
    anchor_velocity: torch.Tensor,
    timestamps: torch.Tensor,
    anchor_time: torch.Tensor,
    gravity: torch.Tensor,
    drag: torch.Tensor,
) -> torch.Tensor:
    frames = []
    for index in range(timestamps.shape[1]):
        position, _ = free_motion_position_velocity(
            anchor_position,
            anchor_velocity,
            timestamps[:, index] - anchor_time,
            gravity=gravity,
            drag=drag,
        )
        frames.append(position)
    return torch.stack(frames, dim=1)


def _scene(dtype: torch.dtype, *, batch: int = 2) -> dict[str, torch.Tensor]:
    slots, time = 2, 16
    anchor_position = (
        torch.tensor([[[0.3, 1.1, -0.2], [-0.7, 0.8, 0.4]]], dtype=dtype)
        .expand(batch, slots, 3)
        .clone()
    )
    anchor_position = anchor_position + torch.arange(batch, dtype=dtype)[:, None, None] * 0.17
    anchor_velocity = (
        torch.tensor([[[0.62, -0.18, 0.31], [-0.47, 0.36, 0.24]]], dtype=dtype)
        .expand(batch, slots, 3)
        .clone()
    )
    anchor_velocity = anchor_velocity + torch.arange(batch, dtype=dtype)[:, None, None] * 0.04
    anchor_time = torch.arange(batch, dtype=dtype)[:, None].expand(batch, slots).clone()
    relative_time = torch.linspace(-0.75, 0.0, time, dtype=dtype)
    timestamps = anchor_time[:, None, :] + relative_time[None, :, None]
    gravity = (
        torch.tensor([[[0.03, -0.04, 0.02], [-0.02, 0.05, -0.03]]], dtype=dtype)
        .expand(batch, slots, 3)
        .clone()
    )
    drag = torch.tensor([[0.045, 0.275], [0.105, 0.215]], dtype=dtype)[:batch]
    positions = _history(
        anchor_position,
        anchor_velocity,
        timestamps,
        anchor_time,
        gravity,
        drag,
    )
    return {
        "anchor_position": anchor_position,
        "anchor_velocity": anchor_velocity,
        "anchor_time": anchor_time,
        "timestamps": timestamps,
        "gravity": gravity,
        "drag": drag,
        "positions": positions,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("noise_scale", [0.0, 4.0e-6])
def test_exact_and_noisy_distinct_per_object_drag_recovery(
    dtype: torch.dtype,
    noise_scale: float,
) -> None:
    scene = _scene(dtype)
    phase = torch.arange(scene["positions"].numel(), dtype=dtype).reshape_as(scene["positions"])
    noisy_positions = scene["positions"] + noise_scale * torch.sin(0.73 * phase)
    timestamps = scene["timestamps"][..., 0] if dtype == torch.float32 else scene["timestamps"]

    result = fit_free_motion_with_drag(
        noisy_positions,
        timestamps,
        gravity=scene["gravity"],
        anchor_time=scene["anchor_time"],
    )

    assert isinstance(result, AnalyticDragFitResult)
    assert result.valid.all()
    assert result.position.shape == (2, 2, 3)
    assert result.log_drag.shape == (2, 2, 1)
    assert result.raw_log_drag_variance.shape == (2, 2, 1)
    assert result.predicted_positions.shape == (2, 16, 2, 3)
    assert result.residual_covariance.shape == (2, 2, 3, 3)
    assert result.profile_information.shape == (2, 2)
    assert result.boundary_mass.shape == (2, 2)
    recovered_drag = result.log_drag.squeeze(-1).exp()
    tolerance = 4.0e-3 if dtype == torch.float32 else 2.5e-3
    torch.testing.assert_close(recovered_drag, scene["drag"], rtol=0, atol=tolerance)
    torch.testing.assert_close(
        result.position,
        scene["anchor_position"],
        rtol=0,
        atol=8.0e-5,
    )
    torch.testing.assert_close(
        result.velocity,
        scene["anchor_velocity"],
        rtol=0,
        atol=5.0e-4,
    )
    coarse_step = (math.log(0.36) - math.log(0.01)) / 256
    minimum_local_width = 2.0 * coarse_step
    minimum_local_cell_variance = (minimum_local_width / 256) ** 2 / 12.0
    assert torch.all(result.raw_log_drag_variance >= minimum_local_cell_variance)
    assert torch.all(result.boundary_mass <= 0.01)


def test_adaptive_quadrature_converges_to_dense_mean_and_variance() -> None:
    scene = _scene(torch.float64, batch=1)
    positions = scene["positions"][:, :, 1:2]
    timestamps = scene["timestamps"][:, :, 1:2]
    gravity = scene["gravity"][:, 1:2]
    anchor_time = scene["anchor_time"][:, 1:2]

    adaptive = fit_free_motion_with_drag(
        positions,
        timestamps,
        gravity=gravity,
        anchor_time=anchor_time,
        grid_points=257,
    )
    dense = fit_free_motion_with_drag(
        positions,
        timestamps,
        gravity=gravity,
        anchor_time=anchor_time,
        grid_points=4097,
    )

    assert adaptive.valid.all() and dense.valid.all()
    torch.testing.assert_close(
        adaptive.log_drag,
        dense.log_drag,
        rtol=0.0,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        adaptive.raw_log_drag_variance,
        dense.raw_log_drag_variance,
        rtol=0.01,
        atol=1.0e-10,
    )


def test_shared_drag_is_a_slotwise_limiting_case() -> None:
    dtype = torch.float64
    scene = _scene(dtype, batch=1)
    shared_drag = torch.full((1, 2), 0.14, dtype=dtype)
    scene["anchor_position"][:, 1] = scene["anchor_position"][:, 0] + torch.tensor(
        [0.8, -0.2, 0.4], dtype=dtype
    )
    scene["anchor_velocity"][:, 1] = scene["anchor_velocity"][:, 0]
    scene["gravity"][:, 1] = scene["gravity"][:, 0]
    positions = _history(
        scene["anchor_position"],
        scene["anchor_velocity"],
        scene["timestamps"],
        scene["anchor_time"],
        scene["gravity"],
        shared_drag,
    )

    result = fit_free_motion_with_drag(
        positions,
        scene["timestamps"],
        gravity=scene["gravity"],
        anchor_time=scene["anchor_time"],
    )

    assert result.valid.all()
    torch.testing.assert_close(result.log_drag[:, 0], result.log_drag[:, 1], atol=2e-10, rtol=0)
    torch.testing.assert_close(
        result.raw_log_drag_variance[:, 0],
        result.raw_log_drag_variance[:, 1],
        atol=2e-10,
        rtol=0,
    )
    torch.testing.assert_close(result.log_drag.exp(), shared_drag[..., None], atol=2e-3, rtol=0)


def test_slot_permutation_equivariance() -> None:
    scene = _scene(torch.float64)
    support = torch.ones(scene["positions"].shape[:3], dtype=torch.bool)
    weights = torch.linspace(0.7, 1.3, 16, dtype=torch.float64)[None, :, None].expand(2, 16, 2)
    baseline = fit_free_motion_with_drag(
        scene["positions"],
        scene["timestamps"],
        gravity=scene["gravity"],
        anchor_time=scene["anchor_time"],
        support=support,
        weights=weights,
    )
    permuted = fit_free_motion_with_drag(
        scene["positions"].flip(2),
        scene["timestamps"].flip(2),
        gravity=scene["gravity"].flip(1),
        anchor_time=scene["anchor_time"].flip(1),
        support=support.flip(2),
        weights=weights.flip(2),
    )

    state_fields = ("position", "velocity", "log_drag", "raw_log_drag_variance")
    history_fields = ("predicted_positions", "residuals")
    scalar_fields = (
        "anchor_time",
        "residual_covariance",
        "condition_number",
        "support_count",
        "support_weight",
        "excitation",
        "profile_information",
        "boundary_mass",
        "valid",
    )
    for name in state_fields:
        torch.testing.assert_close(getattr(permuted, name), getattr(baseline, name).flip(1))
    for name in history_fields:
        torch.testing.assert_close(getattr(permuted, name), getattr(baseline, name).flip(2))
    for name in scalar_fields:
        torch.testing.assert_close(getattr(permuted, name), getattr(baseline, name).flip(1))


def _assert_inference_outputs_zero(result: AnalyticDragFitResult) -> None:
    for value in (
        result.position,
        result.velocity,
        result.log_drag,
        result.raw_log_drag_variance,
        result.predicted_positions,
        result.residuals,
        result.residual_covariance,
    ):
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)


def test_insufficient_support_and_low_excitation_fail_closed() -> None:
    scene = _scene(torch.float64, batch=1)
    support = torch.zeros(scene["positions"].shape[:3], dtype=torch.bool)
    support[:, -2:] = True
    insufficient = fit_free_motion_with_drag(
        scene["positions"],
        scene["timestamps"],
        gravity=scene["gravity"],
        support=support,
    )
    assert not insufficient.valid.any()
    assert torch.equal(insufficient.support_count, torch.full((1, 2), 2))
    assert torch.equal(
        insufficient.profile_information, torch.zeros_like(insufficient.profile_information)
    )
    assert torch.equal(insufficient.excitation, torch.zeros_like(insufficient.excitation))
    _assert_inference_outputs_zero(insufficient)

    stationary = scene["positions"][:, -1:].expand_as(scene["positions"]).clone()
    low_excitation = fit_free_motion_with_drag(
        stationary,
        scene["timestamps"],
        gravity=torch.zeros_like(scene["gravity"]),
    )
    assert not low_excitation.valid.any()
    assert torch.all(low_excitation.excitation < 0.015)
    assert torch.equal(
        low_excitation.profile_information,
        torch.zeros_like(low_excitation.profile_information),
    )
    _assert_inference_outputs_zero(low_excitation)


def test_invalid_rows_have_exact_zero_inference_gradients() -> None:
    scene = _scene(torch.float64, batch=1)
    positions = scene["positions"].clone().requires_grad_(True)
    timestamps = scene["timestamps"].clone().requires_grad_(True)
    gravity = scene["gravity"].clone().requires_grad_(True)
    support = torch.zeros(positions.shape[:3], dtype=torch.bool)
    support[:, -2:] = True

    result = fit_free_motion_with_drag(
        positions,
        timestamps,
        gravity=gravity,
        support=support,
    )
    objective = sum(
        value.sum()
        for value in (
            result.position,
            result.velocity,
            result.log_drag,
            result.raw_log_drag_variance,
            result.predicted_positions,
            result.residuals,
            result.residual_covariance,
        )
    )
    gradients = torch.autograd.grad(objective, (positions, timestamps, gravity))

    assert not result.valid.any()
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)


def test_boundary_and_finite_extreme_rows_fail_closed() -> None:
    dtype = torch.float64
    scene = _scene(dtype, batch=1)
    outside_drag = torch.full((1, 2), 0.9, dtype=dtype)
    positions = _history(
        scene["anchor_position"],
        scene["anchor_velocity"],
        scene["timestamps"],
        scene["anchor_time"],
        scene["gravity"],
        outside_drag,
    )
    boundary = fit_free_motion_with_drag(
        positions,
        scene["timestamps"],
        gravity=scene["gravity"],
        drag_bounds=(0.02, 0.3),
        anchor_time=scene["anchor_time"],
    )
    assert not boundary.valid.any()
    assert torch.all(boundary.boundary_mass > 0.01)
    _assert_inference_outputs_zero(boundary)

    extreme_positions = scene["positions"].clone()
    extreme_positions[:, 3, 0, 1] = torch.finfo(dtype).max / 4
    extreme = fit_free_motion_with_drag(
        extreme_positions,
        scene["timestamps"],
        gravity=scene["gravity"],
        anchor_time=scene["anchor_time"],
    )
    assert extreme.valid.tolist() == [[False, True]]
    assert all(
        torch.isfinite(value).all()
        for value in dataclasses.astuple(extreme)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    for name in ("position", "velocity", "log_drag", "raw_log_drag_variance"):
        torch.testing.assert_close(
            getattr(extreme, name)[:, 0],
            torch.zeros_like(getattr(extreme, name)[:, 0]),
        )


def test_gradients_reach_every_frame_match_central_difference_and_do_not_cross_batches() -> None:
    dtype = torch.float64
    scene = _scene(dtype)
    phase = torch.arange(scene["positions"].numel(), dtype=dtype).reshape_as(scene["positions"])
    base_positions = scene["positions"] + 1.2e-4 * torch.sin(0.41 * phase)
    base_timestamps = scene["timestamps"].clone()
    base_gravity = scene["gravity"].clone()
    positions = base_positions.clone().requires_grad_(True)
    timestamps = base_timestamps.clone().requires_grad_(True)
    gravity = base_gravity.clone().requires_grad_(True)

    def fit(
        position_value: torch.Tensor,
        timestamp_value: torch.Tensor,
        gravity_value: torch.Tensor,
    ) -> AnalyticDragFitResult:
        return fit_free_motion_with_drag(
            position_value,
            timestamp_value,
            gravity=gravity_value,
            anchor_time=scene["anchor_time"],
            position_noise_floor=5.0e-4,
            minimum_excitation=0.0,
            maximum_boundary_mass=1.0,
            minimum_profile_information=0.0,
        )

    result = fit(positions, timestamps, gravity)
    assert result.valid.all()
    coefficients = torch.tensor([[0.7, -0.4], [0.3, 0.9]], dtype=dtype)
    objective = (result.log_drag.squeeze(-1) * coefficients).sum()
    position_gradient, time_gradient, gravity_gradient = torch.autograd.grad(
        objective,
        (positions, timestamps, gravity),
        retain_graph=True,
    )
    for gradient in (position_gradient, time_gradient, gravity_gradient):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0
    assert torch.all(position_gradient.abs().sum(dim=(0, 2, 3)) > 1.0e-10)
    assert torch.all(time_gradient.abs().sum(dim=(0, 2)) > 1.0e-10)

    epsilon = 2.0e-6

    def objective_for(
        position_value: torch.Tensor,
        timestamp_value: torch.Tensor,
        gravity_value: torch.Tensor,
    ) -> torch.Tensor:
        fitted = fit(position_value, timestamp_value, gravity_value)
        return (fitted.log_drag.squeeze(-1) * coefficients).sum()

    plus_positions = base_positions.clone()
    minus_positions = base_positions.clone()
    plus_positions[0, 5, 1, 2] += epsilon
    minus_positions[0, 5, 1, 2] -= epsilon
    position_fd = (
        objective_for(plus_positions, base_timestamps, base_gravity)
        - objective_for(minus_positions, base_timestamps, base_gravity)
    ) / (2 * epsilon)
    torch.testing.assert_close(position_gradient[0, 5, 1, 2], position_fd, rtol=2e-4, atol=2e-6)

    plus_times = base_timestamps.clone()
    minus_times = base_timestamps.clone()
    plus_times[1, 7, 0] += epsilon
    minus_times[1, 7, 0] -= epsilon
    time_fd = (
        objective_for(base_positions, plus_times, base_gravity)
        - objective_for(base_positions, minus_times, base_gravity)
    ) / (2 * epsilon)
    torch.testing.assert_close(time_gradient[1, 7, 0], time_fd, rtol=3e-4, atol=3e-6)

    plus_gravity = base_gravity.clone()
    minus_gravity = base_gravity.clone()
    plus_gravity[0, 1, 1] += epsilon
    minus_gravity[0, 1, 1] -= epsilon
    gravity_fd = (
        objective_for(base_positions, base_timestamps, plus_gravity)
        - objective_for(base_positions, base_timestamps, minus_gravity)
    ) / (2 * epsilon)
    torch.testing.assert_close(gravity_gradient[0, 1, 1], gravity_fd, rtol=3e-4, atol=3e-6)

    batch_zero = torch.autograd.grad(
        result.log_drag[0].sum(),
        (positions, timestamps, gravity),
        retain_graph=False,
    )
    for gradient in batch_zero:
        torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]), rtol=0, atol=0)


def test_profile_fit_beats_a_wrong_fixed_drag_and_owns_no_state() -> None:
    dtype = torch.float64
    scene = _scene(dtype)
    result = fit_free_motion_with_drag(
        scene["positions"],
        scene["timestamps"],
        gravity=scene["gravity"],
        anchor_time=scene["anchor_time"],
    )
    wrong = fit_free_motion(
        scene["positions"],
        scene["timestamps"],
        gravity=scene["gravity"],
        drag=torch.full_like(scene["drag"], 0.15),
        anchor_time=scene["anchor_time"],
    )
    estimated_sse = result.residuals.square().sum(dim=(1, 3))
    wrong_sse = wrong.residuals.square().sum(dim=(1, 3))
    assert torch.all(estimated_sse <= wrong_sse + 1e-18)
    assert torch.any(estimated_sse < 0.1 * wrong_sse)

    assert not isinstance(fit_free_motion_with_drag, torch.nn.Module)
    assert not any(
        isinstance(value, (torch.Tensor, torch.nn.Parameter))
        for value in vars(analytic_drag_module).values()
    )


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    [
        ("drag_bounds", (0.0, 0.3), ValueError),
        ("drag_bounds", (0.3, 0.2), ValueError),
        ("grid_points", 2, ValueError),
        ("grid_points", 5.0, TypeError),
        ("position_noise_floor", 0.0, ValueError),
        ("minimum_support", 2, ValueError),
        ("conditioning_limit", 1.0, ValueError),
        ("minimum_excitation", -1.0, ValueError),
        ("maximum_boundary_mass", 1.1, ValueError),
        ("minimum_profile_information", -1.0, ValueError),
    ],
)
def test_rejects_invalid_controls(keyword: str, value: object, exception: type[Exception]) -> None:
    scene = _scene(torch.float64, batch=1)
    arguments = {
        "gravity": scene["gravity"],
        "anchor_time": scene["anchor_time"],
        keyword: value,
    }
    with pytest.raises(exception):
        fit_free_motion_with_drag(scene["positions"], scene["timestamps"], **arguments)
