"""State-only scalability probes for packed interaction execution."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from statistics import median

import torch

from world_model.belief import BeliefFactory, WorldBelief
from world_model.dynamics import DynamicsModel

STATE_ONLY_COUNTS = (8, 12, 16)
SIX_HORIZONS_SECONDS = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class ScalabilityProbeResult:
    object_count: int
    median_latency_seconds: float
    maximum_batch_difference: float
    finite: bool
    batch_independent: bool
    packed_interactions: bool
    full_perceptual_qualification: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _belief_for_count(model: DynamicsModel, count: int, *, batch_size: int) -> WorldBelief:
    config = model.config
    factory = BeliefFactory(
        max_objects=count,
        modal_count=config.modal_count,
        modal_dim=config.modal_dim,
        residual_dynamics_dim=config.residual_dynamics_dim,
        global_code_dim=config.global_code_dim,
        geometry_dim=config.geometry_dim,
        appearance_dim=config.appearance_dim,
        parameter_memory_dim=config.parameter_memory_dim,
        initial_radius=0.21,
        initial_mass=1.0,
        initial_drag=0.05,
        initial_restitution=0.70,
        initial_friction=0.20,
    )
    belief = factory.create(batch_size=batch_size, gravity=(0.0, 0.0, 0.0))
    objects = belief.objects.clone()
    objects.active[:] = True
    objects.object_id[:] = torch.arange(count).unsqueeze(0)
    objects.position[..., 0] = torch.linspace(-4.0, 4.0, count)
    objects.position[..., 1] = 1.0
    if batch_size > 1:
        objects.position[1:, :, 2] = torch.arange(1, batch_size).unsqueeze(1) * 0.05
    objects.velocity[..., 0] = torch.linspace(0.01, -0.01, count)
    objects.geometry[..., 0] = 0.21
    objects.fast_log_variance.fill_(-12.0)
    return belief.replace(objects=objects).validate(log_variance_bounds=(-32.0, 20.0))


def _trajectory_maximum_difference(left: object, right: object) -> float:
    maximum = 0.0
    for name in ("positions", "velocities", "fast_log_variance"):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        maximum = max(maximum, float((left_value - right_value).abs().max()))
    return maximum


def run_state_only_scalability_probes(
    model: DynamicsModel,
    *,
    counts: tuple[int, ...] = STATE_ONLY_COUNTS,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> tuple[ScalabilityProbeResult, ...]:
    """Measure finite B1 rollouts and B1/B2 independence at N=8/12/16."""

    if not isinstance(model, DynamicsModel):
        raise TypeError("scalability probes require DynamicsModel")
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("invalid scalability timing repetitions")
    if any(isinstance(count, bool) or count <= 0 for count in counts):
        raise ValueError("scalability object counts must be positive integers")
    output: list[ScalabilityProbeResult] = []
    with torch.no_grad():
        for count in counts:
            single = _belief_for_count(model, count, batch_size=1)
            batched = _belief_for_count(model, count, batch_size=2)
            for _ in range(warmup_runs):
                model.rollout(
                    single,
                    SIX_HORIZONS_SECONDS,
                    return_events=False,
                    return_auxiliary=False,
                )
            timings: list[float] = []
            single_trajectory = None
            for _ in range(measured_runs):
                started = time.perf_counter()
                single_trajectory = model.rollout(
                    single,
                    SIX_HORIZONS_SECONDS,
                    return_events=False,
                    return_auxiliary=False,
                )
                timings.append(time.perf_counter() - started)
            assert single_trajectory is not None
            batched_trajectory = model.rollout(
                batched,
                SIX_HORIZONS_SECONDS,
                return_events=False,
                return_auxiliary=False,
            )
            difference = _trajectory_maximum_difference(
                single_trajectory,
                type(single_trajectory)(
                    timestamps=batched_trajectory.timestamps[:1],
                    positions=batched_trajectory.positions[:1],
                    velocities=batched_trajectory.velocities[:1],
                    orientations=batched_trajectory.orientations[:1],
                    motion_mode_logits=batched_trajectory.motion_mode_logits[:1],
                    fast_log_variance=batched_trajectory.fast_log_variance[:1],
                    active_mask=batched_trajectory.active_mask[:1],
                    event_logits=None,
                    auxiliary={},
                ).validate(),
            )
            finite = all(
                bool(torch.isfinite(getattr(single_trajectory, name)).all())
                for name in ("positions", "velocities", "fast_log_variance")
            )
            output.append(
                ScalabilityProbeResult(
                    object_count=count,
                    median_latency_seconds=median(timings),
                    maximum_batch_difference=difference,
                    finite=finite and math.isfinite(median(timings)),
                    batch_independent=difference <= 1.0e-7,
                    packed_interactions=model.interactions.packed_interactions_enabled,
                )
            )
    return tuple(output)


def compare_dense_and_packed_scalability(
    model: DynamicsModel,
    *,
    counts: tuple[int, ...] = STATE_ONLY_COUNTS,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> dict[str, object]:
    """Measure both backends while restoring the caller's execution mode."""

    original = model.interactions.packed_interactions_enabled
    try:
        model.interactions.packed_interactions_enabled = False
        dense = run_state_only_scalability_probes(
            model,
            counts=counts,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        model.interactions.packed_interactions_enabled = True
        packed = run_state_only_scalability_probes(
            model,
            counts=counts,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
    finally:
        model.interactions.packed_interactions_enabled = original
    return {
        "full_perceptual_qualification": False,
        "dense_oracle_probes": [item.to_dict() for item in dense],
        "probes": [item.to_dict() for item in packed],
    }


__all__ = [
    "SIX_HORIZONS_SECONDS",
    "STATE_ONLY_COUNTS",
    "ScalabilityProbeResult",
    "compare_dense_and_packed_scalability",
    "run_state_only_scalability_probes",
]
