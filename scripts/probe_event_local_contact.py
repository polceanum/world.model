#!/usr/bin/env python3
"""Probe whether bounded relation impulses own long-horizon contact error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from world_model.evaluation import adaptive_physics_scale as adaptive
from world_model.evaluation import multicontact_six_dof as contacts
from world_model.evaluation import visual_dynamic_scale as visual
from world_model.evaluation.neural_adaptive_physics import _public_neural_calibrator
from world_model.evaluation.neural_long_horizon import (
    FORECAST_DT,
    FORECAST_SECONDS,
    load_long_horizon_physics_adapter,
)

PROBE_SCHEMA = "world_model_event_local_contact_probe_v1"
_DISPLAY_HORIZONS = (2.0, 4.0, 8.0, 12.0)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=_REPOSITORY_ROOT / "runs/20260914-neural-long-horizon-v1/model.pt",
    )
    parser.add_argument("--object-count", type=int, default=4, choices=(4, 6, 8))
    parser.add_argument(
        "--normal-impulse-logit",
        type=float,
        nargs="+",
        default=(-0.5, 0.0, 0.5),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _curve(error: torch.Tensor) -> dict[str, float]:
    return {
        f"{horizon:.2f}": float(error[int(round(horizon / FORECAST_DT)) - 1])
        for horizon in _DISPLAY_HORIZONS
    }


def probe_normal_impulse_logits(
    checkpoint: str | Path,
    *,
    object_count: int,
    logits: tuple[float, ...],
) -> dict[str, object]:
    """Compare one bounded relation row without mutating a run artifact."""

    if object_count not in (4, 6, 8):
        raise ValueError("object_count must be one of 4, 6, or 8")
    if not logits or not all(torch.isfinite(torch.tensor(value)) for value in logits):
        raise ValueError("normal impulse logits must be finite and nonempty")
    adapter = load_long_horizon_physics_adapter(checkpoint)
    _, _, belief, truth, truth_by_slot = adaptive._evaluate_scenario(
        object_count,
        parameter_calibrator=lambda current, target, slots: _public_neural_calibrator(
            adapter,
            current,
            target,
            slots,
        ),
    )
    source = belief.clone()
    query_times = torch.arange(
        FORECAST_DT,
        FORECAST_SECONDS + 0.5 * FORECAST_DT,
        FORECAST_DT,
        dtype=belief.dtype,
    )
    schedule = visual._model_schedule(belief, truth_by_slot, object_count)
    truth_position, _, _, _ = contacts._reference_rollout(
        truth,
        query_times,
        contacts.default_actions(object_count),
    )
    truth_position = truth_position[:, truth_by_slot]
    candidates = []
    for raw_logit in logits:
        dynamics = adaptive._dynamics(belief)
        with torch.no_grad():
            dynamics.interactions.edge_network.output.bias[4] = raw_logit
            prediction = dynamics.rollout(belief, query_times, action=schedule)
        error = (prediction.positions[0] - truth_position).square().mean(dim=(-2, -1)).sqrt()
        candidates.append(
            {
                "normal_impulse_logit": raw_logit,
                "position_rmse_m": _curve(error),
                "finite": bool(torch.isfinite(error).all()),
            }
        )
    return {
        "schema": PROBE_SCHEMA,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "object_count": object_count,
        "forecast_seconds": FORECAST_SECONDS,
        "normal_impulse_row": 4,
        "source_unchanged": bool(
            torch.equal(source.objects.position, belief.objects.position)
            and torch.equal(source.objects.velocity, belief.objects.velocity)
            and torch.equal(source.objects.log_mass, belief.objects.log_mass)
            and torch.equal(source.objects.log_drag, belief.objects.log_drag)
        ),
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parsed = arguments(argv)
    checkpoint = Path(parsed.checkpoint).expanduser().resolve()
    if parsed.dry_run:
        print(
            json.dumps(
                {
                    "schema": PROBE_SCHEMA,
                    "checkpoint": str(checkpoint),
                    "object_count": parsed.object_count,
                    "forecast_seconds": FORECAST_SECONDS,
                    "normal_impulse_logits": parsed.normal_impulse_logit,
                    "writes_run_artifacts": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    print(
        json.dumps(
            probe_normal_impulse_logits(
                checkpoint,
                object_count=parsed.object_count,
                logits=tuple(parsed.normal_impulse_logit),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
