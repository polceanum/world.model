#!/usr/bin/env python3
"""Measure compact model stability beyond the qualified twelve-second horizon.

This is a no-write diagnostic. It uses public RGB-D-derived state and known
actions at runtime; the independent reference is consulted only after rollout
to reduce metrics. The probe intentionally does not claim promotion or write
dashboard artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from world_model.evaluation import adaptive_physics_scale as adaptive
from world_model.evaluation import multicontact_six_dof as contacts
from world_model.evaluation import visual_dynamic_scale as visual
from world_model.evaluation.neural_adaptive_physics import _public_neural_calibrator
from world_model.evaluation.neural_long_horizon import load_long_horizon_physics_adapter

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _REPOSITORY_ROOT / "runs/20260914-neural-long-horizon-v1/model.pt"
_FRAME_DT_SECONDS = 0.05


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--object-count", choices=(4, 6, 8), type=int, default=4)
    parser.add_argument("--forecast-seconds", choices=(24.0, 48.0), type=float, default=24.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _display_horizons(forecast_seconds: float) -> tuple[float, ...]:
    return tuple(
        horizon
        for horizon in (2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0)
        if horizon <= forecast_seconds
    )


def _curve(error: torch.Tensor, horizons: tuple[float, ...]) -> dict[str, float]:
    return {
        f"{horizon:.2f}": float(error[int(round(horizon / _FRAME_DT_SECONDS)) - 1])
        for horizon in horizons
    }


def probe_extended_horizon(
    checkpoint: str | Path,
    *,
    object_count: int,
    forecast_seconds: float,
) -> dict[str, object]:
    if forecast_seconds not in (24.0, 48.0):
        raise ValueError("forecast_seconds must be 24 or 48")
    adapter = load_long_horizon_physics_adapter(checkpoint)
    _, dynamics, belief, truth, truth_by_slot = adaptive._evaluate_scenario(
        object_count,
        parameter_calibrator=lambda current, target, mapping: _public_neural_calibrator(
            adapter, current, target, mapping
        ),
    )
    query_times = torch.arange(
        _FRAME_DT_SECONDS,
        forecast_seconds + 0.5 * _FRAME_DT_SECONDS,
        _FRAME_DT_SECONDS,
        dtype=belief.dtype,
    )
    source = belief.clone()
    with torch.no_grad():
        prediction = dynamics.rollout(
            belief,
            query_times,
            action=visual._model_schedule(belief, truth_by_slot, object_count),
        )
    reference_position, reference_velocity, reference_orientation, reference_collision = (
        contacts._reference_rollout(
            truth,
            query_times,
            contacts.default_actions(object_count),
        )
    )
    reference_position = reference_position[:, truth_by_slot]
    reference_velocity = reference_velocity[:, truth_by_slot]
    reference_orientation = reference_orientation[:, truth_by_slot]
    reference_collision = reference_collision[:, truth_by_slot][:, :, truth_by_slot]
    position_error = (
        (prediction.positions[0] - reference_position).square().mean(dim=(-2, -1)).sqrt()
    )
    velocity_error = (
        (prediction.velocities[0] - reference_velocity).square().mean(dim=(-2, -1)).sqrt()
    )
    horizons = _display_horizons(forecast_seconds)
    return {
        "schema": "world_model_extended_horizon_probe_v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "object_count": object_count,
        "forecast_seconds": forecast_seconds,
        "frame_dt_seconds": _FRAME_DT_SECONDS,
        "position_rmse_m": _curve(position_error, horizons),
        "velocity_rmse_mps": _curve(velocity_error, horizons),
        "endpoint_position_rmse_m": float(position_error[-1]),
        "maximum_position_rmse_m": float(position_error.max()),
        "collision_pair_f1": contacts._contact_metrics(
            prediction.auxiliary["pair_collision"][0], reference_collision
        )[2],
        "finite": bool(
            torch.isfinite(prediction.positions).all()
            and torch.isfinite(prediction.velocities).all()
            and torch.isfinite(prediction.orientations).all()
        ),
        "source_unchanged": bool(
            torch.equal(source.objects.position, belief.objects.position)
            and torch.equal(source.objects.velocity, belief.objects.velocity)
            and torch.equal(source.objects.log_mass, belief.objects.log_mass)
            and torch.equal(source.objects.log_drag, belief.objects.log_drag)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    checkpoint = parsed.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": "world_model_extended_horizon_probe_v1",
                    "checkpoint": str(checkpoint),
                    "object_count": parsed.object_count,
                    "forecast_seconds": parsed.forecast_seconds,
                    "artifact_policy": "no-write compact JSON stdout only",
                    "runtime_inputs": "public RGB-D belief plus known actions",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            probe_extended_horizon(
                checkpoint,
                object_count=parsed.object_count,
                forecast_seconds=parsed.forecast_seconds,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
