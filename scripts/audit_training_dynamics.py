#!/usr/bin/env python3
"""Audit an Orpheus metrics stream for optimizer and training-collapse signals."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

_HORIZONS = ("0.100s", "0.250s", "0.500s", "0.750s", "1.000s")
_SEVERE_CLIP_COEFFICIENT = 0.1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            value["_line_number"] = line_number
            records.append(value)
    return records


def _finite_numbers(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, value in record.items():
        if key.startswith("_") or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            failures.append(key)
    return failures


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "minimum": min(values) if values else None,
        "median": float(median(values)) if values else None,
        "p95": _percentile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def _metric_distribution(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    return _distribution(
        [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
            and not isinstance(record.get(key), bool)
            and math.isfinite(float(record[key]))
        ]
    )


def _pooled_identity_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    switches = sum(
        float(record.get("physical_distance_gated_identity_switches", 0.0)) for record in records
    )
    associations = sum(
        float(record.get("physical_distance_gated_object_frame_associations", 0.0))
        for record in records
    )
    return {
        "blocks": len(records),
        "switches": switches,
        "associations": associations,
        "switch_rate": switches / associations if associations else 0.0,
    }


def _decoded_list_length(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return len(decoded) if isinstance(decoded, list) else None


def _canonical_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        split = record.get("split")
        step = record.get("step")
        if not isinstance(split, str) or isinstance(step, bool) or not isinstance(step, int):
            continue
        by_key.setdefault((split, step), []).append(record)
    process_fields = {
        "_line_number",
        "elapsed_seconds",
        "post_step_finite_check_seconds",
        "process_max_rss_bytes",
    }
    duplicates: list[dict[str, Any]] = []
    for (split, step), group in sorted(by_key.items(), key=lambda item: item[0][1]):
        if len(group) <= 1:
            continue
        normalized = [
            {key: value for key, value in record.items() if key not in process_fields}
            for record in group
        ]
        duplicates.append(
            {
                "split": split,
                "step": step,
                "count": len(group),
                "equivalent_except_process_fields": all(
                    record == normalized[0] for record in normalized[1:]
                ),
            }
        )
    canonical = [group[-1] for group in by_key.values()]
    canonical.sort(key=lambda record: (int(record["step"]), int(record["_line_number"])))
    return canonical, duplicates


def audit_run(run_directory: Path, *, after_step: int = 0) -> dict[str, Any]:
    """Return a deterministic audit of unique closed-loop optimizer blocks."""

    run_directory = run_directory.expanduser().resolve()
    metrics_path = run_directory / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    records = _read_jsonl(metrics_path)
    canonical, duplicates = _canonical_records(records)
    training = [
        record
        for record in canonical
        if record.get("split") == "train"
        and record.get("phase") == "closed_loop_rgb"
        and int(record["step"]) >= after_step
    ]
    validations = [
        record
        for record in canonical
        if record.get("split") == "validation" and int(record["step"]) >= after_step
    ]
    failures: list[str] = []
    warnings: list[str] = []
    for record in training:
        step = int(record["step"])
        nonfinite = _finite_numbers(record)
        if nonfinite:
            failures.append(f"step {step}: nonfinite metrics: {','.join(sorted(nonfinite))}")
        if float(record.get("optimizer_update_applied", 0.0)) != 1.0:
            failures.append(f"step {step}: optimizer update was not applied")
        if float(record.get("causal_training_support_present", 0.0)) != 1.0:
            failures.append(f"step {step}: causal training support is absent")
        if float(record.get("causal_objective_term_support_count", 0.0)) <= 0.0:
            failures.append(f"step {step}: no causal objective term has support")
        gradient = float(record.get("gradient_norm_pre_clip", math.nan))
        if not math.isfinite(gradient) or gradient <= 0.0:
            failures.append(f"step {step}: raw gradient norm is not finite and positive")
        if float(record.get("closed_loop_scope_state_dynamics_only", 0.0)) == 1.0:
            perception_gradient = float(record.get("perception_gradient_norm_pre_clip", math.nan))
            if perception_gradient != 0.0:
                failures.append(
                    f"step {step}: frozen state/dynamics scope has perception gradient "
                    f"{perception_gradient}"
                )
        skipped = float(record.get("skipped_no_gradient_batches", 0.0))
        draw_step = float(record.get("training_data_draw_step", math.nan))
        if not math.isfinite(draw_step) or draw_step != step + skipped:
            failures.append(
                f"step {step}: data draw invariant failed ({draw_step} != {step} + {skipped})"
            )

    duplicate_training = [item for item in duplicates if item["split"] == "train"]
    if duplicate_training:
        warnings.append(
            "append-only replay rows were canonicalized by (split, step); originals remain intact"
        )
    for duplicate in duplicates:
        if not duplicate["equivalent_except_process_fields"]:
            failures.append(
                f"{duplicate['split']} step {duplicate['step']}: replay rows differ in "
                "model/data metrics"
            )
    skipped_max = max(
        (float(record.get("skipped_no_gradient_batches", 0.0)) for record in training),
        default=0.0,
    )
    if skipped_max > 0.0:
        warnings.append(f"the data stream contains {int(skipped_max)} bounded skipped draw(s)")

    training_steps = [int(record["step"]) for record in training]
    metric_step_gaps = [
        later - earlier for earlier, later in zip(training_steps, training_steps[1:], strict=False)
    ]
    first_expected_step = after_step + 1
    sparse_training_metrics = bool(
        training_steps
        and (training_steps[0] > first_expected_step or any(gap > 1 for gap in metric_step_gaps))
    )
    if sparse_training_metrics:
        warnings.append(
            "training loss/gradient distributions are cadence samples rather than "
            "one record per completed optimizer update"
        )

    losses = [float(record["loss_total"]) for record in training if "loss_total" in record]
    gradients = [
        float(record["gradient_norm_pre_clip"])
        for record in training
        if "gradient_norm_pre_clip" in record
    ]
    clip_coefficients = [
        float(record["gradient_total_clip_coefficient"])
        for record in training
        if "gradient_total_clip_coefficient" in record
    ]
    severe_clipped_steps: list[dict[str, float | int]] = []
    uncontained_interaction_clipped_steps: list[dict[str, float | int]] = []
    for record in training:
        total_coefficient = float(record.get("gradient_total_clip_coefficient", 1.0))
        interaction_stage_coefficient = float(
            record.get("interaction_gradient_clip_coefficient", 1.0)
        )
        interaction_coefficient = float(
            record.get(
                "interaction_gradient_total_clip_coefficient",
                record.get("interaction_gradient_clip_coefficient", 1.0),
            )
        )
        attention_collision_coefficient = float(
            record.get("attention_collision_gradient_clip_coefficient", 1.0)
        )
        attention_force_coefficient = float(
            record.get("attention_force_gradient_clip_coefficient", 1.0)
        )
        attention_impulse_coefficient = float(
            record.get("attention_impulse_gradient_clip_coefficient", 1.0)
        )
        attention_node_output_coefficient = float(
            record.get(
                "attention_node_output_backprop_gradient_minimum_clip_coefficient",
                1.0,
            )
        )
        attention_collision_output_coefficient = float(
            record.get(
                "attention_collision_output_backprop_gradient_minimum_clip_coefficient",
                1.0,
            )
        )
        attention_force_output_coefficient = float(
            record.get(
                "attention_force_output_backprop_gradient_minimum_clip_coefficient",
                1.0,
            )
        )
        attention_impulse_output_coefficient = float(
            record.get(
                "attention_impulse_output_backprop_gradient_minimum_clip_coefficient",
                1.0,
            )
        )
        if (
            min(
                total_coefficient,
                interaction_coefficient,
                attention_collision_coefficient,
                attention_force_coefficient,
                attention_impulse_coefficient,
                attention_node_output_coefficient,
                attention_collision_output_coefficient,
                attention_force_output_coefficient,
                attention_impulse_output_coefficient,
            )
            < _SEVERE_CLIP_COEFFICIENT
        ):
            details: dict[str, float | int] = {
                "step": int(record["step"]),
                "total_coefficient": total_coefficient,
                "interaction_coefficient": interaction_coefficient,
            }
            if "attention_collision_gradient_clip_coefficient" in record:
                details["attention_collision_coefficient"] = attention_collision_coefficient
            if "attention_force_gradient_clip_coefficient" in record:
                details["attention_force_coefficient"] = attention_force_coefficient
            if "attention_impulse_gradient_clip_coefficient" in record:
                details["attention_impulse_coefficient"] = attention_impulse_coefficient
            if "attention_node_output_backprop_gradient_minimum_clip_coefficient" in record:
                details["attention_node_output_coefficient"] = attention_node_output_coefficient
            if "attention_collision_output_backprop_gradient_minimum_clip_coefficient" in record:
                details["attention_collision_output_coefficient"] = (
                    attention_collision_output_coefficient
                )
            if "attention_force_output_backprop_gradient_minimum_clip_coefficient" in record:
                details["attention_force_output_coefficient"] = attention_force_output_coefficient
            if "attention_impulse_output_backprop_gradient_minimum_clip_coefficient" in record:
                details["attention_impulse_output_coefficient"] = (
                    attention_impulse_output_coefficient
                )
            severe_clipped_steps.append(details)
        if interaction_stage_coefficient < _SEVERE_CLIP_COEFFICIENT:
            uncontained_interaction_clipped_steps.append(
                {
                    "step": int(record["step"]),
                    "interaction_stage_coefficient": interaction_stage_coefficient,
                }
            )
    if severe_clipped_steps:
        warnings.append(
            "severe gradient clipping retained less than 10% of at least one "
            "raw typed-output/parameter-group update gradient"
        )
    if uncontained_interaction_clipped_steps:
        failures.append(
            "severe complete-interaction clipping retained less than 10% after "
            "all configured typed-output and decoder-row isolation"
        )
    trajectory_support = [
        float(record["causal_trajectory_support_count"])
        for record in training
        if "causal_trajectory_support_count" in record
    ]
    rss = [
        float(record["process_max_rss_bytes"])
        for record in training
        if "process_max_rss_bytes" in record
    ]
    scenario_counts: Counter[str] = Counter()
    for record in training:
        names = record.get("scenario_names")
        if isinstance(names, str):
            scenario_counts.update(name for name in names.split(",") if name)

    validation_summary = []
    for record in validations:
        validation_summary.append(
            {
                "step": int(record["step"]),
                "score": record.get("validation_rollout_selection_score"),
                "accepted": record.get("selection_accepted"),
                "rejection_reason_count": record.get("selection_rejection_reason_count"),
                "reference_guardrail_failure_count": _decoded_list_length(
                    record.get("selection_reference_guardrail_failures_json")
                ),
                "mutable_support_failure_count": record.get(
                    "selection_mutable_training_support_failure_count"
                ),
                "training_support_failure_count": record.get(
                    "selection_training_support_failure_count"
                ),
                "current_position_rmse_m": record.get("validation_position_rmse_m"),
                "current_position_rmse_by_axis_m": {
                    axis: record.get(f"validation_position_rmse_{axis}_m")
                    for axis in ("x", "y", "z")
                },
                "velocity_rmse_mps": record.get("validation_velocity_rmse_mps"),
                "target_coverage": record.get("validation_target_coverage"),
                "prediction_precision": record.get("validation_prediction_precision"),
                "identity_switch_rate": record.get("validation_id_switch_rate"),
                "collision_f1": record.get("validation_collision_f1"),
                "position_coverage90": record.get("validation_position_coverage90"),
                "horizon_position_rmse_m": {
                    horizon: record.get(f"validation_position_rmse@{horizon}")
                    for horizon in _HORIZONS
                },
                "horizon_position_rmse_by_axis_m": {
                    axis: {
                        horizon: record.get(f"validation_position_rmse_{axis}@{horizon}")
                        for horizon in _HORIZONS
                    }
                    for axis in ("x", "y", "z")
                },
                "horizon_target_coverage": {
                    horizon: record.get(f"validation_forecast_target_coverage@{horizon}")
                    for horizon in _HORIZONS
                },
            }
        )

    live_diagnostic_keys = (
        "physical_current_distance_gated_target_coverage",
        "physical_current_distance_gated_prediction_precision",
        "physical_distance_gated_identity_switch_rate",
        "physical_collision_f1_proxy",
        "physical_position_coverage90",
        "uncertainty_position_nll",
        "current_correction_improvement_m",
        "future_correction_improvement_m",
        "matched_object_frames",
        "existence_negative_supervision_object_frames",
        "parameter_drag_observable_object_count",
        "parameter_restitution_observable_object_count",
    )
    live_physical_diagnostics = {
        key: _metric_distribution(training, key) for key in live_diagnostic_keys
    }
    live_physical_diagnostics["rollout_position_loss_by_axis"] = {
        axis: _metric_distribution(training, f"loss_rollout_position_{axis}")
        for axis in ("x", "y", "z")
    }
    live_physical_diagnostics["rollout_position_rmse_by_horizon_m"] = {
        horizon: _metric_distribution(training, f"physical_rollout_position_rmse_m@{horizon}")
        for horizon in _HORIZONS
    }
    live_physical_diagnostics["forecast_target_coverage_by_horizon"] = {
        horizon: _metric_distribution(training, f"physical_forecast_target_coverage@{horizon}")
        for horizon in _HORIZONS
    }
    perturbed_training = [
        record for record in training if float(record.get("perturbed_updates", 0.0)) > 0.0
    ]
    unperturbed_training = [
        record for record in training if float(record.get("perturbed_updates", 0.0)) == 0.0
    ]
    live_physical_diagnostics["identity_by_recovery_perturbation"] = {
        "all": _pooled_identity_summary(training),
        "perturbed": _pooled_identity_summary(perturbed_training),
        "unperturbed": _pooled_identity_summary(unperturbed_training),
    }
    return {
        "run_directory": str(run_directory),
        "after_step": after_step,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
        "duplicate_rows": duplicates,
        "unique_training_blocks": len(training),
        "first_training_step": int(training[0]["step"]) if training else None,
        "last_training_step": int(training[-1]["step"]) if training else None,
        # A trainer step advances only after an optimizer update succeeds.  The
        # absolute step is therefore the authoritative completed-update count;
        # summing persisted confirmations undercounts whenever log_every > 1.
        "optimizer_updates_applied": int(training[-1]["step"]) if training else 0,
        "logged_optimizer_update_confirmations": sum(
            float(record.get("optimizer_update_applied", 0.0)) for record in training
        ),
        "training_metric_cadence_sparse": sparse_training_metrics,
        "training_metric_step_gaps": metric_step_gaps,
        "loss_total": _distribution(losses),
        "gradient_norm_pre_clip": _distribution(gradients),
        "clipped_block_count": sum(coefficient < 1.0 for coefficient in clip_coefficients),
        "severe_clipped_steps": severe_clipped_steps,
        "uncontained_interaction_clipped_steps": uncontained_interaction_clipped_steps,
        "trajectory_support_count": _distribution(trajectory_support),
        "maximum_skipped_draws": skipped_max,
        "process_rss_bytes": {
            "minimum": min(rss) if rss else None,
            "latest": rss[-1] if rss else None,
            "maximum": max(rss) if rss else None,
        },
        # These counts come from persisted metric rows, not unlogged batches.
        # Keep the legacy key for consumers, but make its scope machine-readable.
        "scenario_draw_counts": dict(sorted(scenario_counts.items())),
        "scenario_draw_counts_scope": "logged_metric_rows_only",
        "live_physical_diagnostics": live_physical_diagnostics,
        "validations": validation_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--after-step", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.after_step < 0:
        raise ValueError("--after-step must be nonnegative")
    report = audit_run(args.run, after_step=args.after_step)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
