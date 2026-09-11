"""Dependency-free static HTML/SVG capability progress reports."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    WORKBENCH_SCHEMA,
    CapabilityRunSummary,
    historical_summary,
    read_capability_summary,
    summary_from_workbench_report,
    write_capability_summary,
)
from world_model.evaluation.general_capability import ALL_CAPABILITY_FACTORS
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import DEFAULT_RUN_ARTIFACT_POLICY, inventory_runs


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_number(value: object, *, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    if number == 0.0:
        return "0"
    if abs(number) < 0.001 or abs(number) >= 10_000:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _format_axis_tick(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1_000:
        return f"{value:.2e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _short_tick_label(value: object, *, maximum: int = 18) -> str:
    label = str(value)
    return label if len(label) <= maximum else f"…{label[-(maximum - 1) :]}"


def _score_value(summary: CapabilityRunSummary, name: str) -> float | None:
    value = summary.scores.get(name)
    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _svg_line_chart(
    points: Sequence[tuple[str, float]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    width, height = 720, 250
    left, right, top, bottom = 78, 20, 30, 62
    if not points:
        return f'<div class="empty">{_escape(title)}: unmeasured</div>'
    values = [value for _, value in points]
    low, high = min(values), max(values)
    if high <= low:
        pad = max(abs(low) * 0.05, 1.0e-9)
        low, high = low - pad, high + pad
    span_x = max(1, len(points) - 1)
    coords: list[tuple[float, float]] = []
    for index, (_, value) in enumerate(points):
        x = (
            (left + width - right) / 2
            if len(points) == 1
            else left + index * (width - left - right) / span_x
        )
        y = top + (high - value) * (height - top - bottom) / (high - low)
        coords.append((x, y))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4"><title>{_escape(label)}: '
        f"{value:.8g}</title></circle>"
        for (label, value), (x, y) in zip(points, coords, strict=True)
    )
    tick_count = min(6, len(points))
    tick_indices = (
        [0]
        if tick_count == 1
        else sorted(
            {round(index * (len(points) - 1) / (tick_count - 1)) for index in range(tick_count)}
        )
    )
    x_ticks = "".join(
        f'<line x1="{coords[index][0]:.2f}" y1="{height - bottom}" '
        f'x2="{coords[index][0]:.2f}" y2="{height - bottom + 5}"/>'
        f'<text x="{coords[index][0]:.2f}" y="{height - bottom + 18}" '
        f'class="axis-tick" text-anchor="middle">{_escape(_short_tick_label(points[index][0]))}</text>'
        for index in tick_indices
    )
    y_values = (high, (high + low) / 2.0, low)
    y_positions = (top, (top + height - bottom) / 2.0, height - bottom)
    y_ticks = "".join(
        f'<line class="chart-grid-line" x1="{left}" y1="{y:.2f}" '
        f'x2="{width - right}" y2="{y:.2f}"/>'
        f'<text x="{left - 8}" y="{y + 4:.2f}" class="axis-tick" '
        f'text-anchor="end">{_escape(_format_axis_tick(value))}</text>'
        for value, y in zip(y_values, y_positions, strict=True)
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(title)}; horizontal axis {_escape(x_label)}; '
        f'vertical axis {_escape(y_label)}">'
        f'<text x="{left}" y="18" class="chart-title">{_escape(title)}</text>'
        f"{y_ticks}"
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>'
        f"{x_ticks}"
        f'<text x="{(left + width - right) / 2:.2f}" y="{height - 8}" '
        f'class="axis-label" text-anchor="middle">{_escape(x_label)}</text>'
        f'<text x="15" y="{(top + height - bottom) / 2:.2f}" class="axis-label" '
        f'text-anchor="middle" transform="rotate(-90 15 {(top + height - bottom) / 2:.2f})">'
        f"{_escape(y_label)}</text>"
        f'<polyline points="{polyline}"/>{circles}</svg>'
    )


def _trend_charts(history: Sequence[CapabilityRunSummary]) -> str:
    """Plot only like-for-like scores on the same axis."""

    groups = (
        (
            "world_model_capability_factor_report_v1",
            "Physical factor score across runs",
        ),
        (
            "world_model_capability_planning_report_v1",
            "Downstream planning error across runs",
        ),
    )
    charts: list[str] = []
    grouped_formats: set[str] = set()
    for source_format, title in groups:
        points = [
            (item.run_id, value)
            for item in history
            if item.source_format == source_format
            and (value := _score_value(item, "candidate")) is not None
        ]
        if points:
            charts.append(
                _svg_line_chart(
                    points,
                    title=title,
                    x_label="Run",
                    y_label="Score (lower is better)",
                )
            )
            grouped_formats.add(source_format)
    other_points = [
        (item.run_id, value)
        for item in history
        if item.source_format not in grouped_formats
        and (value := _score_value(item, "candidate")) is not None
    ]
    if other_points:
        charts.insert(
            0,
            _svg_line_chart(
                other_points,
                title="Overall capability score across runs",
                x_label="Run",
                y_label="Score (lower is better)",
            ),
        )
    return "".join(charts) or '<div class="empty">Capability trend: unmeasured</div>'


def _cell_value(metrics: object, metric: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    evidence = metrics.get(metric)
    if isinstance(evidence, Mapping):
        value = evidence.get("value")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _heatmap(
    summary: CapabilityRunSummary,
    *,
    metric: str,
    title: str,
    lower_is_better: bool,
    ideal_value: float | None = None,
) -> str:
    cells = summary.cell_metrics
    if not cells:
        return f'<div class="empty">{_escape(title)}: unmeasured</div>'
    entries = list(cells.items())
    values = [
        value for _, metrics in entries if (value := _cell_value(metrics, metric)) is not None
    ]
    comparison_values = [
        abs(value - ideal_value) if ideal_value is not None else value for value in values
    ]
    minimum = min(comparison_values, default=0.0)
    maximum = max(comparison_values, default=1.0)
    blocks: list[str] = []
    for index, (label, metrics) in enumerate(entries):
        value = _cell_value(metrics, metric)
        column = index % 6
        row = index // 6
        x, y = 18 + column * 112, 36 + row * 54
        if value is None:
            color = "#303947"
            display = "—"
        else:
            comparison = abs(value - ideal_value) if ideal_value is not None else value
            spread = maximum - minimum
            if spread <= 1.0e-12:
                ratio = 0.5
            else:
                ratio = min(1.0, max(0.0, (comparison - minimum) / spread))
                if not lower_is_better:
                    ratio = 1.0 - ratio
            red = int(54 + 175 * ratio)
            green = int(175 - 100 * ratio)
            color = f"rgb({red},{green},92)"
            display = _format_number(value)
        compact_label = (
            label.replace("/contact=0/dynamic=0", " · C0 · static")
            .replace("/contact=1/dynamic=0", " · C1 · static")
            .replace("/contact=0/dynamic=1", " · C0 · dynamic")
            .replace("/contact=1/dynamic=1", " · C1 · dynamic")
        )
        blocks.append(
            f'<g><rect x="{x}" y="{y}" width="102" height="42" rx="6" fill="{color}"/>'
            f"<title>{_escape(label)}: {display}</title>"
            f'<text x="{x + 5}" y="{y + 16}" class="cell-label">{_escape(compact_label)}</text>'
            f'<text x="{x + 5}" y="{y + 33}" class="cell-value">{display}</text></g>'
        )
    rows = math.ceil(len(entries) / 6)
    legend_y = 54 * rows + 38
    best_label = (
        f"closest to {_format_axis_tick(ideal_value)} target"
        if ideal_value is not None
        else ("lower observed" if lower_is_better else "higher observed")
    )
    worst_label = (
        "furthest from target"
        if ideal_value is not None
        else ("higher observed" if lower_is_better else "lower observed")
    )
    return (
        f'<svg class="heatmap" viewBox="0 0 700 {54 * rows + 72}" role="img" '
        f'aria-label="{_escape(title)}; C means contact; relative colour scale">'
        f'<text x="18" y="20" class="chart-title">{_escape(title)}</text>'
        f"{''.join(blocks)}"
        f'<rect x="18" y="{legend_y}" width="14" height="10" rx="2" fill="rgb(54,175,92)"/>'
        f'<text x="38" y="{legend_y + 9}" class="axis-tick">{_escape(best_label)}</text>'
        f'<rect x="250" y="{legend_y}" width="14" height="10" rx="2" fill="rgb(229,75,92)"/>'
        f'<text x="270" y="{legend_y + 9}" class="axis-tick">{_escape(worst_label)}</text>'
        f'<text x="500" y="{legend_y + 9}" class="axis-tick">C0 no contact · C1 contact</text>'
        f'<text x="18" y="{legend_y + 27}" class="heatmap-note">Relative within this run; colour is not a pass/fail gate.</text>'
        "</svg>"
    )


_FACTOR_METRIC_COLUMNS = (
    ("proposal_f1", "Proposal F1", 0.95, False),
    ("identity_accuracy", "ID accuracy", 0.98, False),
    ("lifecycle_f1", "Lifecycle F1", 0.95, False),
    ("current_position_rmse_m", "Current RMSE", 0.020, True),
    ("two_second_position_rmse_m", "2 s RMSE", 0.120, True),
    ("collision_f1", "Collision F1", 0.90, False),
    ("uncertainty_90_coverage", "90% coverage", (0.82, 0.97), None),
)


def _factor_metric_matrix(summary: CapabilityRunSummary) -> str:
    """Show measured metrics against the factor's actual acceptance floors."""

    headings = "".join(f"<th>{_escape(label)}</th>" for _, label, _, _ in _FACTOR_METRIC_COLUMNS)
    rows: list[str] = []
    for factor, evidence in summary.factor_metrics.items():
        if not isinstance(evidence, Mapping) or evidence.get("status") not in {
            "measured",
            "passed",
            "failed",
        }:
            continue
        cells: list[str] = []
        for name, _label, threshold, direction in _FACTOR_METRIC_COLUMNS:
            value = evidence.get(name)
            cell_class = "unmeasured"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                factor_gate = (threshold, direction)
                if factor == "compositional_holdout":
                    factor_gate = {
                        "proposal_f1": (0.90, False),
                        "identity_accuracy": (0.95, False),
                        "two_second_position_rmse_m": (0.150, True),
                    }.get(name)
                if factor_gate is None:
                    cell_class = "metric-info"
                else:
                    factor_threshold, factor_direction = factor_gate
                    if factor_direction is True:
                        passed = numeric <= float(factor_threshold)
                    elif factor_direction is False:
                        passed = numeric >= float(factor_threshold)
                    else:
                        low, high = factor_threshold
                        passed = float(low) <= numeric <= float(high)
                    cell_class = "metric-pass" if passed else "metric-fail"
            cells.append(f'<td class="{cell_class}">{_format_number(value)}</td>')
        rows.append(f"<tr><th>{_escape(factor)}</th>{''.join(cells)}</tr>")
    if not rows:
        return '<div class="empty">Factor metric matrix: unmeasured</div>'
    return (
        '<table class="metric-matrix"><thead><tr><th>Family</th>'
        f"{headings}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _factor_table(summary: CapabilityRunSummary) -> str:
    rows = []
    for factor, evidence in summary.factor_metrics.items():
        status = (
            evidence.get("status", "unmeasured") if isinstance(evidence, Mapping) else "invalid"
        )
        score = evidence.get("score") if isinstance(evidence, Mapping) else None
        rows.append(
            f'<tr><td>{_escape(factor)}</td><td><span class="tag {_escape(status)}">'
            f"{_escape(status)}</span></td><td>{_format_number(score)}</td>"
            f"<td>{_escape(evidence.get('source_run', '—') if isinstance(evidence, Mapping) else '—')}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Capability family</th><th>Evidence</th><th>Score</th>"
        "<th>Latest run</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _merge_latest_factor_evidence(
    summaries: Sequence[CapabilityRunSummary],
    latest: CapabilityRunSummary,
) -> CapabilityRunSummary:
    """Keep each family's newest measured result visible across compact runs."""

    merged = dict(latest.factor_metrics)
    for summary in summaries:
        for factor, evidence in summary.factor_metrics.items():
            if not isinstance(evidence, Mapping) or evidence.get("status") not in {
                "measured",
                "passed",
                "failed",
            }:
                continue
            merged[factor] = {**dict(evidence), "source_run": summary.run_id}
    measured = {
        factor
        for factor, evidence in merged.items()
        if isinstance(evidence, Mapping)
        and evidence.get("status") in {"measured", "passed", "failed"}
    }
    capability_claims = {factor: f"{factor} capability" for factor in measured}
    capability_claims["compositional_holdout"] = "compositional holdout capability"
    unsupported = tuple(
        claim
        for claim in latest.unsupported_claims
        if not any(
            factor in measured and claim == rendered
            for factor, rendered in capability_claims.items()
        )
    )
    planning_by_factor: dict[str, Any] = {}
    for summary in summaries:
        slices = summary.planning.get("slices")
        if not isinstance(slices, list) or not slices:
            continue
        planning_factor = summary.planning.get("factor")
        if not isinstance(planning_factor, str) or not planning_factor:
            planning_factor = "nominal_structured"
        planning_by_factor[planning_factor] = {
            **summary.planning,
            "source_run": summary.run_id,
        }
    planning = latest.planning
    planning_source: str | None = None
    if not isinstance(planning.get("slices"), list) or not planning.get("slices"):
        for summary in reversed(summaries):
            slices = summary.planning.get("slices")
            if isinstance(slices, list) and slices:
                planning = {**summary.planning, "source_run": summary.run_id}
                planning_source = summary.run_id
                break
    if planning_by_factor:
        planning = {**planning, "by_factor": planning_by_factor}
        latest_sequence = latest.planning.get("by_candidate_count")
        if isinstance(latest_sequence, Mapping) and latest_sequence:
            planning = {
                **planning,
                "latest_specialized": {
                    **latest.planning,
                    "source_run": latest.run_id,
                },
            }
    measured_planning = set(planning_by_factor) - {"nominal_structured"}
    unsupported = tuple(
        claim
        for claim in unsupported
        if not (
            any(claim == f"{factor} factor-conditioned planning" for factor in measured_planning)
            or (
                claim == "factor-conditioned planning"
                and set(ALL_CAPABILITY_FACTORS) <= measured_planning
            )
        )
    )
    evidence_source = latest
    if not latest.cell_metrics or not latest.resources:
        for summary in reversed(summaries):
            if summary.cell_metrics and summary.resources:
                evidence_source = summary
                break
    qualitative = dict(latest.qualitative)
    animation_sources: dict[str, str] = {}
    for field, source_field in (
        ("animations", "animation_source_run"),
        ("forecast_animations", "forecast_animation_source_run"),
    ):
        if field == "forecast_animations":
            collected: list[dict[str, Any]] = []
            source_runs: list[str] = []
            seen_from_newer: set[tuple[str, str]] = set()
            reserved_source: str | None = None
            # Reserve one card for the newest measured forecast so a newly
            # qualified behavior (for example pose/contact) cannot disappear
            # behind three older long-horizon cards.
            for summary in reversed(summaries):
                values = summary.qualitative.get(field)
                if not isinstance(values, list) or not values:
                    continue
                newest = next((item for item in values if isinstance(item, Mapping)), None)
                if newest is None:
                    continue
                key = (str(newest.get("episode", "")), str(newest.get("label", "")))
                collected.append({**dict(newest), "source_run": summary.run_id})
                seen_from_newer.add(key)
                source_runs.append(summary.run_id)
                reserved_source = summary.run_id
                break
            # Prefer a compact gallery of genuine long-horizon forecasts from
            # the newest distinct runs, then fill any remaining card slots
            # with shorter forecasts. Repeated cards inside one source run are
            # preserved for backward-compatible best/median/worst galleries.
            for long_only in (True, False):
                for summary in reversed(summaries):
                    values = summary.qualitative.get(field)
                    if not isinstance(values, list) or not values:
                        continue
                    source_keys: set[tuple[str, str]] = set()
                    for animation in values:
                        if not isinstance(animation, Mapping):
                            continue
                        endpoint = animation.get("long_horizon_endpoint_s", 2.0)
                        try:
                            is_long = float(endpoint) > 2.0
                        except (TypeError, ValueError):
                            is_long = False
                        if long_only != is_long:
                            continue
                        key = (
                            str(animation.get("episode", "")),
                            str(animation.get("label", "")),
                        )
                        if key in seen_from_newer and summary.run_id != reserved_source:
                            continue
                        collected.append({**dict(animation), "source_run": summary.run_id})
                        source_keys.add(key)
                        if summary.run_id not in source_runs:
                            source_runs.append(summary.run_id)
                        if len(collected) == 3:
                            break
                    seen_from_newer.update(source_keys)
                    if len(collected) == 3:
                        break
                if len(collected) == 3:
                    break
            if collected:
                qualitative[field] = collected
                animation_sources[source_field] = ", ".join(source_runs)
            continue
        if isinstance(qualitative.get(field), list) and qualitative.get(field):
            existing_source = latest.provenance.get(source_field)
            animation_sources[source_field] = (
                existing_source
                if isinstance(existing_source, str) and existing_source
                else latest.run_id
            )
            continue
        for summary in reversed(summaries):
            if isinstance(summary.qualitative.get(field), list) and summary.qualitative.get(field):
                qualitative[field] = summary.qualitative[field]
                animation_sources[source_field] = summary.run_id
                break
    return CapabilityRunSummary.from_dict(
        {
            **latest.to_dict(),
            "factor_metrics": merged,
            "cell_metrics": latest.cell_metrics or evidence_source.cell_metrics,
            "horizon_curves": latest.horizon_curves or evidence_source.horizon_curves,
            "uncertainty": latest.uncertainty or evidence_source.uncertainty,
            "planning": planning,
            "resources": latest.resources
            or {
                **evidence_source.resources,
                "source_run": evidence_source.run_id,
            },
            "qualitative": qualitative,
            "unsupported_claims": unsupported,
            "provenance": {
                **latest.provenance,
                **({} if planning_source is None else {"planning_source_run": planning_source}),
                **animation_sources,
            },
        }
    )


def _planning_table(summary: CapabilityRunSummary) -> str:
    by_factor = summary.planning.get("by_factor")
    groups: list[tuple[str, Mapping[str, Any], list[Any]]] = []
    if isinstance(by_factor, Mapping):
        for factor, evidence in by_factor.items():
            if isinstance(evidence, Mapping) and isinstance(evidence.get("slices"), list):
                groups.append((str(factor), evidence, evidence["slices"]))
    else:
        slices = summary.planning.get("slices", [])
        if isinstance(slices, list) and slices:
            groups.append(
                (
                    str(summary.planning.get("factor", "nominal")),
                    summary.planning,
                    slices,
                )
            )
    sequence_by_count = summary.planning.get("by_candidate_count")
    if not groups and isinstance(sequence_by_count, Mapping) and sequence_by_count:
        tasks = summary.planning.get("tasks")
        task_values = (
            [item for item in tasks if isinstance(item, Mapping)] if isinstance(tasks, list) else []
        )
        has_replanning = any("replanning_consistent" in item for item in task_values)
        aggregate_rows = []
        for candidate_count, values in sorted(
            sequence_by_count.items(),
            key=lambda item: int(item[0]),
        ):
            if not isinstance(values, Mapping):
                continue
            replan_cell = (
                f"<td>{_format_number(values.get('replanning_consistency'))}</td>"
                if has_replanning
                else ""
            )
            aggregate_rows.append(
                "<tr>"
                f"<td>K{_escape(candidate_count)}</td>"
                f"<td>{_format_number(values.get('winner_accuracy'))}</td>"
                f"<td>{_format_number(values.get('median_normalized_regret'))}</td>"
                f"<td>{_format_number(values.get('goal_success'))}</td>"
                f"{replan_cell}"
                "</tr>"
            )
        task_rows = []
        for item in task_values:
            replan_cell = (
                f"<td>{_escape(item.get('replanning_consistent', '—'))}</td>"
                if has_replanning
                else ""
            )
            latency = item.get("vectorized_latency_seconds", item.get("latency_seconds"))
            task_rows.append(
                "<tr>"
                f"<td>{_escape(item.get('scenario', 'unknown'))}</td>"
                f"<td>K{_escape(item.get('candidate_count', '?'))}</td>"
                f"<td>{_escape(item.get('winner_correct', '—'))}</td>"
                f"<td>{_format_number(item.get('normalized_regret'))}</td>"
                f"{replan_cell}"
                f"<td>{_format_number(latency)}</td>"
                "</tr>"
            )
        task_heading = "Action-sequence task ledger" if has_replanning else "Planning task ledger"
        replan_heading = "<th>Replan consistent</th>" if has_replanning else ""
        task_table = (
            f'<details class="planning-slices"><summary>{task_heading} ({len(task_rows)} tasks)</summary>'
            "<table><thead><tr><th>Scenario</th><th>Candidates</th><th>Winner correct</th>"
            f"<th>Normalized regret</th>{replan_heading}<th>Latency (s)</th></tr></thead>"
            f"<tbody>{''.join(task_rows)}</tbody></table></details>"
            if task_rows
            else ""
        )
        heading = (
            "Action-sequence planning and replanning"
            if has_replanning
            else "Counterfactual action planning"
        )
        aggregate_replan_heading = "<th>Replan consistency</th>" if has_replanning else ""
        return (
            f"<h3>{heading}</h3>"
            '<p class="table-note">Planning is evaluated downstream only; no winner, ranking, regret, or task-success loss is optimized.</p>'
            "<table><thead><tr><th>Candidates</th><th>Winner accuracy</th>"
            f"<th>Median regret</th><th>Goal success</th>{aggregate_replan_heading}</tr></thead>"
            f"<tbody>{''.join(aggregate_rows)}</tbody></table>{task_table}"
        )
    if not groups:
        return '<div class="empty">Planning slices: unmeasured</div>'
    latency_rows: list[str] = []
    aggregate_rows: list[str] = []
    rows: list[str] = []
    for factor, evidence, slices in groups:
        invariants = evidence.get("invariants", {})
        if not isinstance(invariants, Mapping):
            invariants = {}
        latency_rows.append(
            "<tr>"
            f"<td>{_escape(factor)}</td>"
            f"<td>{_escape(evidence.get('status', 'measured'))}</td>"
            f"<td>{_format_number(invariants.get('latency_k8_seconds'))}</td>"
            f"<td>{_format_number(invariants.get('latency_k32_seconds'))}</td>"
            f"<td>{_format_number(evidence.get('maximum_cost_difference'))}</td>"
            "</tr>"
        )
        by_candidate_count: dict[int, list[tuple[float | None, float | None, float | None]]] = {}
        for item in slices:
            if not isinstance(item, Mapping):
                continue
            try:
                candidate_count = int(item.get("candidate_count"))
            except (TypeError, ValueError):
                candidate_count = -1
            by_candidate_count.setdefault(candidate_count, []).append(
                (
                    _cell_value(item, "oracle_winner_accuracy"),
                    _cell_value(item, "normalized_regret_median"),
                    _cell_value(item, "successful_oracle_goal_success"),
                )
            )
            rows.append(
                "<tr>"
                f"<td>{_escape(factor)}</td>"
                f"<td>N{_escape(item.get('object_count', '?'))}</td>"
                f"<td>K{_escape(item.get('candidate_count', '?'))}</td>"
                f"<td>{_format_number(_cell_value(item, 'oracle_winner_accuracy'))}</td>"
                f"<td>{_format_number(_cell_value(item, 'normalized_regret_median'))}</td>"
                f"<td>{_format_number(_cell_value(item, 'successful_oracle_goal_success'))}</td>"
                "</tr>"
            )
        for candidate_count, values in sorted(by_candidate_count.items()):
            winners = [value[0] for value in values if value[0] is not None]
            regrets = [value[1] for value in values if value[1] is not None]
            successes = [value[2] for value in values if value[2] is not None]
            aggregate_rows.append(
                "<tr>"
                f"<td>{_escape(factor)}</td>"
                f"<td>{'K' + str(candidate_count) if candidate_count >= 0 else '—'}</td>"
                f"<td>{_format_number(min(winners) if winners else None)}</td>"
                f"<td>{_format_number(max(regrets) if regrets else None)}</td>"
                f"<td>{_format_number(min(successes) if successes else None)}</td>"
                "</tr>"
            )
    latency_table = (
        "<h3>Planning efficiency and numerical agreement</h3>"
        "<table><thead><tr><th>Factor</th><th>Status</th><th>K8 latency (s)</th>"
        "<th>K32 latency (s)</th><th>Maximum cost difference</th></tr></thead>"
        f"<tbody>{''.join(latency_rows)}</tbody></table>"
    )
    aggregate_table = (
        "<h3>Worst-cardinality planning quality</h3>"
        '<p class="table-note">Minimum winner accuracy and goal success; maximum median regret across measured object counts.</p>'
        "<table><thead><tr><th>Factor</th><th>Candidates</th><th>Winner accuracy</th>"
        f"<th>Median regret</th><th>Goal success</th></tr></thead><tbody>{''.join(aggregate_rows)}</tbody></table>"
    )
    slice_table = (
        f'<details class="planning-slices"><summary>Planning quality by cardinality ({len(rows)} detailed slices)</summary>'
        "<table><thead><tr><th>Factor</th><th>Count</th><th>Candidates</th><th>Winner accuracy</th>"
        f"<th>Median regret</th><th>Goal success</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
    )
    specialized = summary.planning.get("latest_specialized")
    specialized_table = ""
    if isinstance(specialized, Mapping):
        specialized_by_count = specialized.get("by_candidate_count")
        if isinstance(specialized_by_count, Mapping) and specialized_by_count:
            specialized_rows = []
            for candidate_count, values in sorted(
                specialized_by_count.items(), key=lambda item: int(item[0])
            ):
                if not isinstance(values, Mapping):
                    continue
                specialized_rows.append(
                    "<tr>"
                    f"<td>K{_escape(candidate_count)}</td>"
                    f"<td>{_format_number(values.get('winner_accuracy'))}</td>"
                    f"<td>{_format_number(values.get('median_normalized_regret'))}</td>"
                    f"<td>{_format_number(values.get('goal_success'))}</td>"
                    "</tr>"
                )
            goal = specialized.get("goal", "specialized downstream task")
            source = specialized.get("source_run", "latest run")
            specialized_table = (
                f"<h3>Latest specialized planning: {_escape(goal)}</h3>"
                f'<p class="table-note">Source: {_escape(source)}. This bounded task is shown '
                "separately from the factor-conditioned planning matrix.</p>"
                "<table><thead><tr><th>Candidates</th><th>Winner accuracy</th>"
                "<th>Median regret</th><th>Goal success</th></tr></thead>"
                f"<tbody>{''.join(specialized_rows)}</tbody></table>"
            )
    return latency_table + aggregate_table + slice_table + specialized_table


def _list(values: Iterable[object]) -> str:
    rendered = "".join(f"<li>{_escape(value)}</li>" for value in values)
    return f"<ul>{rendered}</ul>" if rendered else '<div class="empty">None recorded</div>'


def _n16_latency(resources: Mapping[str, Any]) -> float | None:
    scalability = resources.get("scalability")
    if not isinstance(scalability, Mapping):
        return None
    probes = scalability.get("probes")
    if not isinstance(probes, list):
        return None
    for probe in probes:
        if isinstance(probe, Mapping) and probe.get("object_count") == 16:
            latency = probe.get("median_latency_seconds")
            if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
                return float(latency)
    return None


def _n8_perceptual_probe(resources: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the newest explicit N=8 RGB-D development probe, if present."""

    scalability = resources.get("scalability")
    if not isinstance(scalability, Mapping):
        return None
    probes = scalability.get("perceptual_development_probes")
    if not isinstance(probes, list):
        return None
    for probe in reversed(probes):
        if isinstance(probe, Mapping) and probe.get("object_count") == 8:
            return probe
    return None


def _budget_utilization(artifacts: Mapping[str, Any]) -> float | None:
    used = artifacts.get("managed_run_tree_bytes")
    budget = artifacts.get("managed_budget_bytes")
    if (
        isinstance(used, (int, float))
        and not isinstance(used, bool)
        and isinstance(budget, (int, float))
        and not isinstance(budget, bool)
        and math.isfinite(float(used))
        and math.isfinite(float(budget))
        and float(budget) > 0.0
    ):
        return 100.0 * float(used) / float(budget)
    return None


_ANIMATION_PALETTE = (
    "#69d6c5",
    "#f3bc61",
    "#ff7b91",
    "#8aa7ff",
    "#c68cff",
    "#8bd46e",
    "#ff9f68",
    "#63c5ef",
)
_ANIMATION_PLOT_LEFT = 48.0
_ANIMATION_PLOT_TOP = 18.0
_ANIMATION_PLOT_WIDTH = 268.0
_ANIMATION_PLOT_HEIGHT = 184.0


def _animation_section(
    summary: CapabilityRunSummary,
    *,
    collection: str,
    title: str,
    introduction: str,
) -> str:
    animations = summary.qualitative.get(collection)
    if not isinstance(animations, list) or not animations:
        return ""
    cards: list[str] = []
    for animation_index, animation in enumerate(animations[:3]):
        if not isinstance(animation, Mapping):
            continue
        frames = animation.get("frames")
        bounds = animation.get("bounds")
        if not isinstance(frames, list) or not frames or not isinstance(bounds, Mapping):
            continue
        axis_labels = animation.get("axis_labels")
        if not isinstance(axis_labels, list) or len(axis_labels) != 2:
            projection = str(animation.get("projection", "world_xy"))
            axis_labels = list(projection.removeprefix("world_")[:2])
        horizontal_bounds = bounds.get("horizontal", bounds.get(str(axis_labels[0])))
        vertical_bounds = bounds.get("vertical", bounds.get(str(axis_labels[1])))
        if (
            not isinstance(horizontal_bounds, list)
            or len(horizontal_bounds) != 2
            or not isinstance(vertical_bounds, list)
            or len(vertical_bounds) != 2
        ):
            continue
        try:
            x_low, x_high = (float(value) for value in horizontal_bounds)
            y_low, y_high = (float(value) for value in vertical_bounds)
        except (TypeError, ValueError):
            continue
        if not all(map(math.isfinite, (x_low, x_high, y_low, y_high))):
            continue
        if x_high <= x_low or y_high <= y_low or not isinstance(frames[0], Mapping):
            continue
        mode = str(animation.get("mode", "tracking"))
        first_frame = frames[0]
        try:
            first_frame_index = int(first_frame.get("frame", 0))
            first_time = float(first_frame.get("time_s", 0.0))
            anchor_frame = int(animation.get("anchor_frame", 0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(first_time):
            continue
        object_ids: set[int] = set()
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            for role in ("truth", "model"):
                points = frame.get(role)
                if not isinstance(points, list):
                    continue
                for point in points:
                    if isinstance(point, list) and len(point) >= 3 and isinstance(point[0], int):
                        object_ids.add(point[0])

        def point_for_frame(
            frame: Mapping[str, Any],
            role: str,
            object_id: int,
            *,
            x_min: float = x_low,
            x_max: float = x_high,
            y_min: float = y_low,
            y_max: float = y_high,
        ) -> tuple[float, float] | None:
            points = frame.get(role)
            if not isinstance(points, list):
                return None
            for point in points:
                if not isinstance(point, list) or len(point) < 3 or point[0] != object_id:
                    continue
                try:
                    x_value, y_value = float(point[1]), float(point[2])
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(x_value) or not math.isfinite(y_value):
                    return None
                x = _ANIMATION_PLOT_LEFT + _ANIMATION_PLOT_WIDTH * (x_value - x_min) / (
                    x_max - x_min
                )
                y = (
                    _ANIMATION_PLOT_TOP
                    + _ANIMATION_PLOT_HEIGHT
                    - _ANIMATION_PLOT_HEIGHT * (y_value - y_min) / (y_max - y_min)
                )
                return x, y
            return None

        def orientation_for_frame(
            frame: Mapping[str, Any],
            role: str,
            object_id: int,
        ) -> float | None:
            points = frame.get(role)
            if not isinstance(points, list):
                return None
            for point in points:
                if not isinstance(point, list) or len(point) < 4 or point[0] != object_id:
                    continue
                try:
                    angle = float(point[3])
                except (TypeError, ValueError):
                    return None
                return angle if math.isfinite(angle) else None
            return None

        first = first_frame
        trails: list[str] = []
        if mode == "forecast":
            for palette_index, object_id in enumerate(sorted(object_ids)):
                color = _ANIMATION_PALETTE[palette_index % len(_ANIMATION_PALETTE)]
                for role in ("truth", "model"):
                    coordinates = [
                        point
                        for frame in frames
                        if isinstance(frame, Mapping)
                        and (point := point_for_frame(frame, role, object_id)) is not None
                    ]
                    if len(coordinates) < 2:
                        continue
                    path = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
                    trails.append(
                        f'<polyline class="animation-trail animation-trail-{role}" '
                        f'points="{path}" style="--object-color:{color}"/>'
                    )
        circles: list[str] = []
        for palette_index, object_id in enumerate(sorted(object_ids)):
            color = _ANIMATION_PALETTE[palette_index % len(_ANIMATION_PALETTE)]
            for role, radius in (("truth", 7), ("model", 4)):
                coordinates = point_for_frame(first, role, object_id)
                display = "none" if coordinates is None else "inline"
                x, y = coordinates or (0.0, 0.0)
                circles.append(
                    f'<circle class="animation-{role}" data-role="{role}" '
                    f'data-object-id="{object_id}" cx="{x:.2f}" cy="{y:.2f}" '
                    f'r="{radius}" style="--object-color:{color};display:{display}"/>'
                )
                angle = orientation_for_frame(first, role, object_id)
                if angle is not None and coordinates is not None:
                    marker_length = 13 if role == "truth" else 10
                    end_x = x + marker_length * math.cos(angle)
                    end_y = y - marker_length * math.sin(angle)
                    circles.append(
                        f'<line class="animation-orientation animation-orientation-{role}" '
                        f'data-orientation-role="{role}" data-object-id="{object_id}" '
                        f'x1="{x:.2f}" y1="{y:.2f}" x2="{end_x:.2f}" y2="{end_y:.2f}" '
                        f'style="--object-color:{color}"/>'
                    )
        event_kinds = animation.get("events", [])
        event_labels: list[str] = []
        for item in event_kinds:
            if not isinstance(item, Mapping) or not item.get("kind"):
                continue
            try:
                event_frame = float(item.get("frame"))
            except (TypeError, ValueError):
                event_labels.append(str(item["kind"]))
                continue
            if mode == "forecast":
                frame_rate = animation.get("frame_rate", 20.0)
                try:
                    frame_rate_value = float(frame_rate)
                except (TypeError, ValueError):
                    frame_rate_value = 20.0
                if not math.isfinite(frame_rate_value) or frame_rate_value <= 0.0:
                    frame_rate_value = 20.0
                explicit_time = item.get("time_s")
                try:
                    event_seconds = float(explicit_time)
                except (TypeError, ValueError):
                    event_seconds = (event_frame - anchor_frame) / frame_rate_value
                precision = 2 if abs(event_seconds * 100 - round(event_seconds * 100)) < 1e-9 else 3
                event_time = f"+{event_seconds:.{precision}f} s"
            else:
                frame_rate = animation.get("frame_rate", 20.0)
                try:
                    frame_rate_value = float(frame_rate)
                except (TypeError, ValueError):
                    frame_rate_value = 20.0
                if not math.isfinite(frame_rate_value) or frame_rate_value <= 0.0:
                    frame_rate_value = 20.0
                event_time = f"{event_frame / frame_rate_value:.2f} s"
            event_labels.append(f"{item['kind']} @ {event_time}")
        if mode == "forecast":
            endpoint = animation.get("long_horizon_endpoint_s", 2.0)
            try:
                endpoint_value = float(endpoint)
            except (TypeError, ValueError):
                endpoint_value = 2.0
            error_label = f"{endpoint_value:g} s RMSE"
            error_value = animation.get(
                "endpoint_position_rmse_m",
                animation.get("two_second_position_rmse_m"),
            )
        else:
            error_label = "current RMSE"
            error_value = animation.get("current_position_rmse_m")
        initial_clock = (
            f"forecast +{first_time:.2f} s · frame {first_frame_index}"
            if mode == "forecast"
            else f"frame {first_frame_index} · {first_time:.2f} s"
        )
        contact_label = "contact" if bool(animation.get("contact")) else "no contact"
        membership_label = (
            "dynamic membership"
            if bool(animation.get("dynamic_membership"))
            else "static membership"
        )
        animation_source = animation.get("source_run")
        source_label = (
            f" · source {_escape(animation_source)}"
            if isinstance(animation_source, str) and animation_source
            else ""
        )
        horizontal_name = f"World {str(axis_labels[0]).upper()} (m)"
        vertical_name = f"World {str(axis_labels[1]).upper()} (m)"
        cards.append(
            '<article class="animation-card" '
            f'data-animation-collection="{_escape(collection)}" '
            f'data-world-animation="{animation_index}">'
            f"<h3>{_escape(animation.get('label', 'example')).title()}</h3>"
            f'<p class="animation-meta">{_escape(animation.get("episode", "unknown episode"))}'
            f" · N={_escape(animation.get('object_count', '?'))}"
            f" · {_escape(contact_label)} · {_escape(membership_label)}"
            f" · {_escape(error_label)} {_format_number(error_value)} m{source_label}</p>"
            '<svg class="world-animation" viewBox="0 0 336 244" role="img" '
            f'aria-label="{_escape(animation.get("label", "example"))} model versus reference world trajectory; horizontal axis {_escape(horizontal_name)}; vertical axis {_escape(vertical_name)}">'
            '<rect x="48" y="18" width="268" height="184" rx="7" class="animation-stage"/>'
            '<path d="M48 64H316M48 110H316M48 156H316M115 18V202M182 18V202M249 18V202" '
            'class="animation-grid-lines"/>'
            f"{''.join(trails)}"
            f"{''.join(circles)}"
            '<circle class="animation-contact" data-contact-marker cx="0" cy="0" r="6" style="display:none"/>'
            f'<text x="48" y="215" class="animation-tick" text-anchor="start">{_escape(_format_axis_tick(x_low))}</text>'
            f'<text x="316" y="215" class="animation-tick" text-anchor="end">{_escape(_format_axis_tick(x_high))}</text>'
            f'<text x="44" y="25" class="animation-tick" text-anchor="end">{_escape(_format_axis_tick(y_high))}</text>'
            f'<text x="44" y="202" class="animation-tick" text-anchor="end">{_escape(_format_axis_tick(y_low))}</text>'
            f'<text x="182" y="238" class="animation-axis" text-anchor="middle">{_escape(horizontal_name)}</text>'
            f'<text x="12" y="110" class="animation-axis" text-anchor="middle" transform="rotate(-90 12 110)">{_escape(vertical_name)}</text>'
            "</svg>"
            '<div class="animation-readout"><span class="animation-clock" '
            f'data-animation-clock>{_escape(initial_clock)}</span><span class="animation-event" '
            "data-animation-event></span></div>"
            '<div class="animation-legend"><span class="model-key">● model estimate</span>'
            '<span class="truth-key">○ private reference</span></div>'
            f'<p class="animation-events">Events: {_escape(" · ".join(event_labels) or "none")}</p>'
            '<div class="animation-controls">'
            f'<button type="button" data-animation-toggle aria-label="Pause {_escape(animation.get("label", "example"))} animation">Pause</button>'
            f'<button type="button" data-animation-replay aria-label="Replay {_escape(animation.get("label", "example"))} animation">Replay</button>'
            '<input type="range" min="0" max="100" step="0.1" value="0" '
            f'data-animation-scrubber aria-label="Scrub {_escape(animation.get("label", "example"))} animation timeline"/>'
            "</div></article>"
        )
    if not cards:
        return ""
    source_field = (
        "forecast_animation_source_run"
        if collection == "forecast_animations"
        else "animation_source_run"
    )
    configured_source = summary.provenance.get(source_field)
    source_run = (
        configured_source
        if isinstance(configured_source, str) and configured_source
        else summary.run_id
    )
    source_text = f' <span class="evidence-source">Evidence source: {_escape(source_run)}.</span>'
    return (
        f"<section><h2>{_escape(title)}</h2>"
        f'<p class="animation-intro">{_escape(introduction)}{source_text}</p>'
        f'<div class="animation-grid">{"".join(cards)}</div></section>'
    )


def _animation_cards(summary: CapabilityRunSummary) -> str:
    tracking = _animation_section(
        summary,
        collection="animations",
        title="Observed tracking examples",
        introduction=(
            "Filled markers are public model state; rings are private reference positions "
            "opened only after RGB-D inference. Each card labels the world-axis plane chosen "
            "from its motion, using bounded downsampled vector keyframes."
        ),
    )
    forecast_values = summary.qualitative.get("forecast_animations")
    long_endpoints = []
    has_pose_markers = False
    if isinstance(forecast_values, list):
        for animation in forecast_values:
            if not isinstance(animation, Mapping):
                continue
            has_pose_markers |= animation.get("orientation_markers") is True
            endpoint = animation.get("long_horizon_endpoint_s")
            if isinstance(endpoint, (int, float)) and math.isfinite(float(endpoint)):
                long_endpoints.append(float(endpoint))
    if long_endpoints and max(long_endpoints) > 2.0:
        forecast_title = (
            "Pose-aware and four/eight-second causal forecasts"
            if has_pose_markers
            else "Four/eight-second causal forecasts"
        )
        forecast_introduction = (
            "Long-horizon rollouts apply each declared future impulse exactly once and continue "
            "through analytic pair, floor, and wall contacts. Integrated RGB-D and state-first "
            "evidence remain labelled by source; orientation spokes and contact rings appear "
            "where measured. Compact vector keyframes replace rendered frame directories."
        )
    else:
        forecast_title = "Two-second open-loop forecasts"
        forecast_introduction = (
            "Each forecast starts from the mature frame-15 public belief and rolls forward "
            "without later observations. Lines connect the evaluated 0.05, 0.10, 0.25, "
            "0.50, 1.0, and 2.0 second horizons; future actions and membership changes are "
            "excluded."
        )
    forecasts = _animation_section(
        summary,
        collection="forecast_animations",
        title=forecast_title,
        introduction=forecast_introduction,
    )
    return tracking + forecasts


def _frontier_table(history: Sequence[CapabilityRunSummary]) -> str:
    """Combine measured count/horizon evidence without upgrading its claim type."""

    evidence: dict[tuple[int, float], str] = {}
    for summary in history:
        curve = summary.horizon_curves.get("candidate_position_rmse_m")
        if not isinstance(curve, Mapping):
            continue
        horizons = []
        for value in curve:
            try:
                horizon = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(horizon) and horizon > 0.0:
                horizons.append(horizon)
        counts: set[int] = set()
        for name in summary.cell_metrics:
            prefix = str(name).split("/", 1)[0]
            if prefix.startswith("N") and prefix[1:].isdigit():
                counts.add(int(prefix[1:]))
        state_first = bool(summary.configuration.get("state_first"))
        if state_first:
            animations = summary.qualitative.get("forecast_animations")
            if isinstance(animations, list):
                counts.update(
                    int(item["object_count"])
                    for item in animations
                    if isinstance(item, Mapping) and isinstance(item.get("object_count"), int)
                )
        claim = "state" if state_first else "RGB-D"
        for count in counts:
            for horizon in horizons:
                previous = evidence.get((count, horizon))
                if previous != "RGB-D" or claim == "RGB-D":
                    evidence[(count, horizon)] = claim

        scalability = summary.resources.get("scalability")
        if isinstance(scalability, Mapping):
            probes = scalability.get("probes")
            if isinstance(probes, list):
                for probe in probes:
                    if not isinstance(probe, Mapping):
                        continue
                    count = probe.get("object_count")
                    finite = probe.get("finite")
                    independent = probe.get("batch_independent")
                    if isinstance(count, int) and finite is True and independent is True:
                        evidence.setdefault((count, 2.0), "state")
    if not evidence:
        return '<div class="empty">Capability frontier: unmeasured</div>'
    counts = sorted({key[0] for key in evidence})
    horizons = sorted({key[1] for key in evidence})
    header = "".join(f"<th>N={count}</th>" for count in counts)
    rows = []
    for horizon in horizons:
        cells = []
        for count in counts:
            claim = evidence.get((count, horizon))
            cells.append(
                '<td class="unmeasured">—</td>'
                if claim is None
                else f'<td><span class="tag measured">{_escape(claim)}</span></td>'
            )
        rows.append(f"<tr><th>{horizon:g} s</th>{''.join(cells)}</tr>")
    return (
        '<table class="frontier"><thead><tr><th>Forecast horizon</th>'
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        '<p class="table-note">RGB-D = measured from a public observation-derived belief; '
        "state = state-first or state-only pressure test. A measured cell is not automatically "
        "a promotion pass.</p>"
    )


def _ablation_attribution(summary: CapabilityRunSummary) -> str:
    attribution = summary.failure_attribution
    rows: list[tuple[str, str, str]] = []
    same_seed = attribution.get("same_seed_nominal_restoration")
    if isinstance(same_seed, Mapping):
        reductions = same_seed.get("all_reductions")
        if isinstance(reductions, Mapping):
            for intervention, reduction in reductions.items():
                rows.append(
                    (
                        str(intervention),
                        _format_number(reduction),
                        "owner" if intervention == same_seed.get("owner") else "diagnostic",
                    )
                )
    ablations = attribution.get("ablations")
    if isinstance(ablations, Mapping):
        for intervention, result in ablations.items():
            if isinstance(result, Mapping):
                rows.append(
                    (
                        str(intervention),
                        _format_number(result.get("error_reduction")),
                        str(result.get("status", "measured")),
                    )
                )
    if not rows:
        return (
            f"<p>Primary bottleneck: {_escape(attribution.get('primary_bottleneck', 'unavailable'))}"
            f"<br>Current owner: {_escape(attribution.get('ablation_owner', 'unmeasured'))}</p>"
            '<div class="empty">Intervention attribution: unmeasured</div>'
        )
    body = "".join(
        f"<tr><td>{_escape(name)}</td><td>{reduction}</td><td>{_escape(status)}</td></tr>"
        for name, reduction, status in rows
    )
    return (
        f"<p>Primary bottleneck: {_escape(attribution.get('primary_bottleneck', 'unavailable'))}"
        f"<br>Current owner: {_escape(attribution.get('ablation_owner', 'unmeasured'))}</p>"
        "<table><thead><tr><th>Intervention</th><th>Absolute error reduction</th>"
        f"<th>Interpretation</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _parameter_convergence_section(summary: CapabilityRunSummary) -> str:
    values = summary.qualitative.get("parameter_convergence")
    if not isinstance(values, list) or not values:
        return ""
    rows: list[str] = []
    chart_points: list[tuple[str, float]] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("stage", "update"))
        mean_error = item.get("mean_relative_error")
        if isinstance(mean_error, (int, float)) and math.isfinite(float(mean_error)):
            chart_points.append((label, float(mean_error)))
        rows.append(
            "<tr>"
            f"<td>{_escape(label)}</td>"
            f"<td>{_format_number(mean_error)}</td>"
            f"<td>{_format_number(item.get('mass_relative_error'))}</td>"
            f"<td>{_format_number(item.get('restitution_relative_error'))}</td>"
            f"<td>{_format_number(item.get('drag_relative_error'))}</td>"
            f"<td>{_format_number(item.get('friction_relative_error'))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<section class="grid two"><article><h2>Online physical identification</h2>'
        + _svg_line_chart(
            chart_points,
            title="Parameter error after public evidence",
            x_label="Evidence stage",
            y_label="Mean relative parameter error",
        )
        + "</article><article><h2>Parameter error ledger</h2>"
        '<p class="table-note">Relative errors are measured only after runtime inference; '
        "private values are never supplied to the estimator.</p>"
        "<table><thead><tr><th>Evidence stage</th><th>Mean</th><th>Mass</th>"
        "<th>Restitution</th><th>Drag</th><th>Friction</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></article></section>"
    )


def _summary_section(
    summary: CapabilityRunSummary,
    *,
    history: Sequence[CapabilityRunSummary],
) -> str:
    frontier_history = tuple(history) if history else (summary,)
    candidate_score = _score_value(summary, "candidate")
    incumbent_score = _score_value(summary, "incumbent")
    horizon = summary.horizon_curves.get("candidate_position_rmse_m", {})
    horizon_points = (
        [(f"{key}s", float(value)) for key, value in horizon.items()]
        if isinstance(horizon, Mapping)
        else []
    )
    velocity_horizon = summary.horizon_curves.get("candidate_velocity_rmse_mps", {})
    velocity_horizon_points = (
        [(f"{key}s", float(value)) for key, value in velocity_horizon.items()]
        if isinstance(velocity_horizon, Mapping)
        else []
    )
    velocity_chart = (
        _svg_line_chart(
            velocity_horizon_points,
            title="Velocity RMSE across horizon",
            x_label="Prediction horizon (s)",
            y_label="Velocity RMSE (m/s)",
        )
        if velocity_horizon_points
        else ""
    )
    orientation_horizon = summary.horizon_curves.get("candidate_orientation_rmse_degrees", {})
    orientation_horizon_points = (
        [(f"{key}s", float(value)) for key, value in orientation_horizon.items()]
        if isinstance(orientation_horizon, Mapping)
        else []
    )
    orientation_chart = (
        _svg_line_chart(
            orientation_horizon_points,
            title="Orientation RMSE across horizon",
            x_label="Prediction horizon (s)",
            y_label="Orientation RMSE (degrees)",
        )
        if orientation_horizon_points
        else ""
    )
    resources = summary.resources
    n8_probe = _n8_perceptual_probe(resources) or {}
    coverage = summary.uncertainty.get("coverage_90_range", ())
    coverage_text = (
        f"{_format_number(coverage[0])}–{_format_number(coverage[1])}"
        if isinstance(coverage, (list, tuple)) and len(coverage) == 2
        else "—"
    )
    return f"""
    <section class="hero">
      <div><span class="eyebrow">{_escape(summary.lifecycle_status)}</span><h1>{_escape(summary.run_id)}</h1>
      <p>{_escape(summary.outcome)} · {_escape(summary.created_at_utc)} · {_escape(summary.source_format)}</p></div>
      <div class="score"><small>candidate / incumbent · lower is better</small><strong>{_format_number(candidate_score)} / {_format_number(incumbent_score)}</strong><span>{_escape(summary.scores.get("selected", "unselected"))}</span></div>
    </section>
    <section class="grid three">
      <article><h2>Capability error</h2><p class="metric">{_format_number(candidate_score)}</p><p>Macro-averaged candidate score; lower is better</p></article>
      <article><h2>Uncertainty</h2><p class="metric">{coverage_text}</p><p>Observed nominal-90% coverage range</p></article>
      <article><h2>Planning parity</h2><p class="metric">{_escape(summary.planning.get("serial_vectorized_winner_parity", "—"))}</p><p>Maximum cost difference {_format_number(summary.planning.get("maximum_cost_difference"))}</p></article>
    </section>
    <section class="grid two"><article><h2>Capability coverage</h2>{_factor_table(summary)}</article><article><h2>Horizon error</h2>{_svg_line_chart(horizon_points, title="Position RMSE across horizon", x_label="Prediction horizon (s)", y_label="Position RMSE (m)")}{velocity_chart}{orientation_chart}</article></section>
    {_parameter_convergence_section(summary)}
    <section><h2>Factor performance</h2>{_factor_metric_matrix(summary)}</section>
    <section><h2>Physical behavior</h2>{_heatmap(summary, metric="current_position_rmse_m", title="Current-position RMSE (m) by object count, contact, and membership", lower_is_better=True)}{_heatmap(summary, metric="uncertainty_90_coverage", title="90% uncertainty coverage by object count, contact, and membership", lower_is_better=True, ideal_value=0.90)}</section>
    <section class="grid two"><article><h2>Capability frontier</h2>{_frontier_table(frontier_history)}</article><article><h2>Ablation attribution</h2>{_ablation_attribution(summary)}</article></section>
    <section><h2>Downstream planning</h2>{_planning_table(summary)}</section>
    <section class="grid two"><article><h2>Efficiency and storage</h2><dl>
      <dt>Perception latency</dt><dd>{_format_number(resources.get("perception_latency_seconds"))} s</dd>
      <dt>Six-horizon rollout</dt><dd>{_format_number(resources.get("six_horizon_rollout_seconds"))} s</dd>
      <dt>State-only N=16</dt><dd>{_format_number(_n16_latency(resources))} s</dd>
      <dt>N=8 RGB-D development</dt><dd>{_format_number(n8_probe.get("inference_latency_seconds"))} s</dd>
      <dt>N=8 observed objects</dt><dd>{_format_number(n8_probe.get("observed_active_count"))} / {_format_number(n8_probe.get("object_count"))}</dd>
      <dt>N=8 position RMSE</dt><dd>{_format_number(n8_probe.get("position_rmse_m"))} m</dd>
      <dt>N=8 claim level</dt><dd>{"qualified" if n8_probe.get("full_perceptual_qualification") is True else "development only"}</dd>
      <dt>Learned weights</dt><dd>{_format_number(resources.get("learned_weight_bytes"))} bytes</dd>
      <dt>Persistent tensors</dt><dd>{_format_number(resources.get("persistent_tensor_bytes"))} bytes</dd>
      <dt>RSS</dt><dd>{_format_number(resources.get("process_rss_bytes"))} bytes</dd>
      <dt>Run artifacts</dt><dd>{_format_number(summary.artifacts.get("run_bytes"))} bytes</dd>
      <dt>Managed run tree</dt><dd>{_format_number(summary.artifacts.get("managed_run_tree_bytes"))} bytes</dd>
      <dt>Rolling budget</dt><dd>{_format_number(summary.artifacts.get("managed_budget_bytes"))} bytes</dd>
      <dt>Budget used</dt><dd>{_format_number(_budget_utilization(summary.artifacts))}%</dd>
      <dt>Archive (separate)</dt><dd>{_format_number(summary.artifacts.get("archive_bytes"))} bytes</dd>
    </dl></article><article><h2>Diagnosis</h2><p>{_escape(summary.failure_attribution.get("primary_bottleneck", "unavailable"))}</p><p>Owner: {_escape(summary.failure_attribution.get("ablation_owner", "unmeasured"))}</p><h3>Qualitative evidence</h3><dl>
      <dt>Best</dt><dd>{_escape(summary.qualitative.get("best_episode", "unavailable"))}</dd>
      <dt>Worst</dt><dd>{_escape(summary.qualitative.get("worst_episode", "unavailable"))}</dd>
      <dt>Representative</dt><dd>{_escape(summary.qualitative.get("representative_episode", "unavailable"))}</dd>
    </dl></article></section>
    {_animation_cards(summary)}
    <section class="grid two"><article><h2>Unsupported claims</h2>{_list(summary.unsupported_claims)}</article><article><h2>Explicit scope limits</h2>{_list(summary.scope_limitations)}</article></section>
    """


_STYLE = """
:root{color-scheme:dark;--bg:#0b1017;--panel:#141c27;--muted:#91a0b5;--ink:#f5f7fb;--accent:#69d6c5;--line:#344154}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,#172a37,var(--bg) 42%);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;align-items:end;border-top:3px solid var(--accent);padding-top:18px}.hero h1{font-size:clamp(26px,4vw,48px);margin:.1em 0}.hero p,.metric+ p{color:var(--muted)}.eyebrow,.tag{font-size:11px;letter-spacing:.08em;text-transform:uppercase}.score{background:var(--panel);padding:18px 22px;border-radius:12px;display:grid;min-width:240px}.score strong,.metric{font-size:25px;color:var(--accent)}.grid{display:grid;gap:14px;margin:14px 0}.two{grid-template-columns:repeat(2,minmax(0,1fr))}.three{grid-template-columns:repeat(3,minmax(0,1fr))}article,section:not(.hero){background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0;overflow:auto}section.grid{background:none;border:0;padding:0}section.grid article{margin:0}h2{margin:0 0 12px;font-size:16px}h3{font-size:13px;color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid var(--line);padding:7px}.tag{padding:3px 6px;border-radius:9px;background:#303947}.tag.measured,.tag.passed{background:#165449}.tag.failed{background:#6b2737}.metric-matrix td{font-variant-numeric:tabular-nums}.metric-pass{background:#123d35}.metric-fail{background:#51232e;color:#ffdce4}.metric-info{background:#1c2d3b}.unmeasured{color:var(--muted)}.empty{color:var(--muted);padding:24px;text-align:center;border:1px dashed var(--line);border-radius:8px}svg{width:100%;min-width:520px}svg line{stroke:var(--line)}svg polyline{fill:none;stroke:var(--accent);stroke-width:3}svg circle{fill:var(--accent)}svg text{fill:var(--muted);font-size:11px}.chart-title{fill:var(--ink);font-size:13px}.axis-label{fill:var(--ink);font-size:11px;font-weight:600}.axis-tick,.heatmap-note{fill:var(--muted);font-size:9px}.chart-grid-line{stroke:#253444;stroke-width:1}.cell-label{fill:white;font-size:8px}.cell-value{fill:white;font-size:11px;font-weight:700}dl{display:grid;grid-template-columns:1fr 1fr;gap:7px}dt{color:var(--muted)}dd{margin:0;text-align:right}.animation-intro,.animation-meta,.animation-events{color:var(--muted)}.evidence-source{display:block;margin-top:4px;color:#c9d3df}.animation-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.animation-card{margin:0!important;padding:12px!important}.animation-card h3{margin:0;color:var(--ink);text-transform:capitalize}.animation-meta{min-height:54px;font-size:12px}.world-animation{display:block;min-width:0;background:#0a1119;border:1px solid var(--line);border-radius:8px}.animation-stage{fill:#0c1621;stroke:#344154}.animation-grid-lines{fill:none;stroke:#253444;stroke-width:1}.animation-axis{fill:#c9d3df;font-size:10px;font-weight:600}.animation-tick{fill:var(--muted);font-size:8px}.animation-trail{fill:none;stroke:var(--object-color);stroke-width:1.3;opacity:.5}.animation-trail-truth{stroke-dasharray:3 3;opacity:.25}.animation-model{fill:var(--object-color);stroke:#081018;stroke-width:1.5}.animation-truth{fill:none;stroke:var(--object-color);stroke-width:2;stroke-dasharray:2 2}.animation-readout{display:flex;justify-content:space-between;gap:8px;min-height:20px;margin-top:5px;font-size:11px}.animation-clock{color:#c9d3df;font-variant-numeric:tabular-nums}.animation-event{color:#f3bc61;font-weight:700}.animation-legend{display:flex;gap:12px;margin-top:3px;color:var(--muted);font-size:12px}.model-key{color:var(--accent)}.truth-key{color:#f3bc61}.animation-events{min-height:38px;font-size:12px}.animation-controls{display:flex;align-items:center;gap:8px}.animation-controls button{border:1px solid var(--line);border-radius:7px;background:#1c2d3b;color:var(--ink);padding:5px 11px;cursor:pointer}.animation-controls button:hover{border-color:var(--accent)}.animation-controls input{min-width:72px;flex:1;accent-color:var(--accent)}.run-list a{color:var(--accent)}.run-ledger summary{cursor:pointer;color:#c9d3df}.notice{border-left:3px solid #f3bc61;padding-left:10px;color:var(--muted)}footer{color:var(--muted);padding:20px 0}@media(max-width:900px){.animation-grid{grid-template-columns:1fr}}@media(max-width:760px){.two,.three{grid-template-columns:1fr}.hero{display:block}.score{margin-top:12px}}@media(prefers-reduced-motion:reduce){.animation-controls button{outline:1px solid var(--muted)}}
.table-note{color:var(--muted);font-size:12px}.planning-slices,.run-ledger{margin-top:14px}.planning-slices summary,.run-ledger summary{cursor:pointer;color:#c9d3df;font-weight:600}.animation-orientation{stroke:var(--object-color);stroke-width:2.2;stroke-linecap:round}.animation-orientation-truth{stroke-dasharray:2 2;opacity:.8}.animation-contact{fill:none;stroke:#f3bc61;stroke-width:2;opacity:.95}
"""


_ANIMATION_SCRIPT = r"""
<script>
(() => {
  const dataNode = document.getElementById("capability-run-summary");
  if (!dataNode) return;
  const summary = JSON.parse(dataNode.textContent);
  const qualitative = summary.qualitative || {};
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const states = new Map();
  let pendingFrame = null;
  const plot = {left: 48, top: 18, width: 268, height: 184};
  const animationFor = root => {
    const items = qualitative[root.dataset.animationCollection];
    return Array.isArray(items) ? items[Number(root.dataset.worldAnimation)] : null;
  };
  const indexPoints = points => new Map((Array.isArray(points) ? points : []).map(p => [String(p[0]), p]));
  const position = (point, animation) => {
    const bounds = animation.bounds;
    const axes = Array.isArray(animation.axis_labels) ? animation.axis_labels : ["x", "y"];
    const horizontal = bounds.horizontal || bounds[axes[0]] || bounds.x;
    const vertical = bounds.vertical || bounds[axes[1]] || bounds.y || bounds.z;
    return {
      x: plot.left + plot.width * (point[1] - horizontal[0]) / (horizontal[1] - horizontal[0]),
      y: plot.top + plot.height - plot.height * (point[2] - vertical[0]) / (vertical[1] - vertical[0])
    };
  };
  const render = (root, animation, progress) => {
    const frames = animation.frames;
    if (!Array.isArray(frames) || frames.length === 0) return;
    const firstTime = Number(frames[0].time_s);
    const lastTime = Number(frames[frames.length - 1].time_s);
    const actualTime = firstTime + progress * (lastTime - firstTime);
    let rightIndex = frames.findIndex(frame => Number(frame.time_s) >= actualTime);
    if (rightIndex < 0) rightIndex = frames.length - 1;
    const leftIndex = Math.max(0, rightIndex - 1);
    const left = frames[leftIndex];
    const right = frames[rightIndex];
    const interval = Number(right.time_s) - Number(left.time_s);
    const mix = interval > 0 ? (actualTime - Number(left.time_s)) / interval : 0;
    for (const role of ["truth", "model"]) {
      const leftPoints = indexPoints(left[role]);
      const rightPoints = indexPoints(right[role]);
      for (const circle of root.querySelectorAll(`circle[data-role="${role}"]`)) {
        const objectId = circle.dataset.objectId;
        const a = leftPoints.get(objectId);
        const b = rightPoints.get(objectId);
        const point = a && b
          ? [
              a[0],
              a[1] + mix * (b[1] - a[1]),
              a[2] + mix * (b[2] - a[2]),
              a.length >= 4 && b.length >= 4
                ? a[3] + mix * Math.atan2(Math.sin(b[3] - a[3]), Math.cos(b[3] - a[3]))
                : undefined
            ]
          : (mix < 0.5 ? a : b);
        if (!point) {
          circle.style.display = "none";
          continue;
        }
        const screen = position(point, animation);
        circle.setAttribute("cx", screen.x.toFixed(2));
        circle.setAttribute("cy", screen.y.toFixed(2));
        circle.style.display = "inline";
      }
      for (const marker of root.querySelectorAll(`line[data-orientation-role="${role}"]`)) {
        const objectId = marker.dataset.objectId;
        const a = leftPoints.get(objectId);
        const b = rightPoints.get(objectId);
        const point = a && b && a.length >= 4 && b.length >= 4
          ? [
              a[0],
              a[1] + mix * (b[1] - a[1]),
              a[2] + mix * (b[2] - a[2]),
              a[3] + mix * Math.atan2(Math.sin(b[3] - a[3]), Math.cos(b[3] - a[3]))
            ]
          : (mix < 0.5 ? a : b);
        if (!point || point.length < 4 || !Number.isFinite(Number(point[3]))) {
          marker.style.display = "none";
          continue;
        }
        const screen = position(point, animation);
        const length = role === "truth" ? 13 : 10;
        marker.setAttribute("x1", screen.x.toFixed(2));
        marker.setAttribute("y1", screen.y.toFixed(2));
        marker.setAttribute("x2", (screen.x + length * Math.cos(point[3])).toFixed(2));
        marker.setAttribute("y2", (screen.y - length * Math.sin(point[3])).toFixed(2));
        marker.style.display = "inline";
      }
    }
    const contactMarker = root.querySelector("[data-contact-marker]");
    if (contactMarker) {
      const contactFrame = mix < 0.5 ? left : right;
      const contact = Array.isArray(contactFrame.contacts) ? contactFrame.contacts[0] : null;
      if (Array.isArray(contact) && contact.length >= 2) {
        const screen = position([0, Number(contact[0]), Number(contact[1])], animation);
        contactMarker.setAttribute("cx", screen.x.toFixed(2));
        contactMarker.setAttribute("cy", screen.y.toFixed(2));
        contactMarker.style.display = "inline";
      } else {
        contactMarker.style.display = "none";
      }
    }
    const actualFrame = left.frame + mix * (right.frame - left.frame);
    const clock = root.querySelector("[data-animation-clock]");
    if (clock) clock.textContent = animation.mode === "forecast"
      ? `forecast +${actualTime.toFixed(2)} s · frame ${Math.round(actualFrame)}`
      : `frame ${Math.round(actualFrame)} · ${actualTime.toFixed(2)} s`;
    const eventNode = root.querySelector("[data-animation-event]");
    if (eventNode) {
      const nearby = (animation.events || [])
        .filter(event => Math.abs(event.frame - actualFrame) < 1.5)
        .map(event => event.kind);
      eventNode.textContent = [...new Set(nearby)].join(" · ");
    }
    const scrubber = root.querySelector("[data-animation-scrubber]");
    if (scrubber) scrubber.value = (progress * 100).toFixed(1);
  };
  for (const root of document.querySelectorAll("[data-world-animation]")) {
    const animation = animationFor(root);
    if (!animation) continue;
    const state = {progress: 0, paused: reducedMotion, previous: performance.now()};
    states.set(root, state);
    const toggle = root.querySelector("[data-animation-toggle]");
    const replay = root.querySelector("[data-animation-replay]");
    const scrubber = root.querySelector("[data-animation-scrubber]");
    if (toggle) {
      toggle.textContent = state.paused ? "Play" : "Pause";
      toggle.setAttribute("aria-label", `${state.paused ? "Play" : "Pause"} ${animation.label || "example"} animation`);
      toggle.addEventListener("click", () => {
        state.paused = !state.paused;
        state.previous = performance.now();
        toggle.textContent = state.paused ? "Play" : "Pause";
        toggle.setAttribute("aria-label", `${state.paused ? "Play" : "Pause"} ${animation.label || "example"} animation`);
        schedule();
      });
    }
    if (replay) replay.addEventListener("click", () => {
      state.progress = 0;
      state.paused = false;
      state.previous = performance.now();
      if (toggle) {
        toggle.textContent = "Pause";
        toggle.setAttribute("aria-label", `Pause ${animation.label || "example"} animation`);
      }
      render(root, animation, 0);
      schedule();
    });
    if (scrubber) scrubber.addEventListener("input", () => {
      state.progress = Number(scrubber.value) / 100;
      state.paused = true;
      state.previous = performance.now();
      if (toggle) {
        toggle.textContent = "Play";
        toggle.setAttribute("aria-label", `Play ${animation.label || "example"} animation`);
      }
      render(root, animation, state.progress);
    });
    render(root, animation, 0);
  }
  function schedule() {
    if (pendingFrame === null && [...states.values()].some(state => !state.paused)) {
      pendingFrame = window.requestAnimationFrame(tick);
    }
  }
  function tick(now) {
    pendingFrame = null;
    let running = false;
    for (const root of document.querySelectorAll("[data-world-animation]")) {
      const animation = animationFor(root);
      const state = states.get(root);
      if (!animation || !state) continue;
      if (!state.paused) {
        state.progress = (state.progress + (now - state.previous) / 6500) % 1;
        running = true;
      }
      state.previous = now;
      render(root, animation, state.progress);
    }
    if (running) pendingFrame = window.requestAnimationFrame(tick);
  }
  schedule();
})();
</script>
"""


def render_summary_html(
    summary: CapabilityRunSummary,
    *,
    title: str = "World-model capability",
    history: Sequence[CapabilityRunSummary] = (),
    notices: Sequence[str] = (),
) -> str:
    """Render one self-contained portable HTML document."""

    summary.validate()
    refresh = (
        '<meta http-equiv="refresh" content="15">' if summary.lifecycle_status == "active" else ""
    )
    embedded = json.dumps(summary.to_dict(), sort_keys=True, allow_nan=False).replace("</", "<\\/")
    history_rows = "".join(
        f"<li><strong>{_escape(item.run_id)}</strong> · {_escape(item.outcome)} · "
        f"{_format_number(_score_value(item, 'candidate'))}</li>"
        for item in reversed(history)
    )
    history_ledger = (
        f'<details class="run-ledger"><summary>Run ledger ({len(history)} compact reports)</summary>'
        f"<ol>{history_rows}</ol></details>"
        if history_rows
        else '<div class="empty">Run ledger: no reports</div>'
    )
    notice_rows = "".join(f'<p class="notice">{_escape(item)}</p>' for item in notices)
    notice_html = (
        f"<details><summary>{len(notices)} protected/missing artifact notices</summary>"
        f"{notice_rows}</details>"
        if notices
        else ""
    )
    animation_script = _ANIMATION_SCRIPT if _animation_cards(summary) else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>{_escape(title)}</title><style>{_STYLE}</style></head><body><main>{_summary_section(summary, history=history)}{notice_html}<section class="run-list"><h2>Capability trend</h2>{_trend_charts(history)}{history_ledger}</section><footer>Portable report · schema {CAPABILITY_SUMMARY_SCHEMA} · no external assets</footer></main><script type="application/json" id="capability-run-summary">{embedded}</script>{animation_script}</body></html>"""


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _load_run_summary(directory: Path) -> tuple[CapabilityRunSummary | None, str | None]:
    summary_path = directory / "capability_summary.json"
    if summary_path.exists():
        try:
            return read_capability_summary(summary_path), None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            return None, f"{directory.name}: malformed capability summary ({error})"
    candidates = (
        directory / "report.json",
        directory / "development_report.json",
        directory / "qualification_report.json",
        directory / "development_report_v2.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise TypeError("report root is not a mapping")
            if raw.get("schema") == WORKBENCH_SCHEMA:
                return summary_from_workbench_report(
                    raw,
                    run_id=directory.name,
                    artifacts={"run_bytes": _directory_size(directory)},
                ), None
            return historical_summary(
                raw,
                run_id=directory.name,
                source_format=str(raw.get("schema", path.name)),
            ), None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            return None, f"{directory.name}: malformed historical report ({error})"
    return None, f"{directory.name}: no supported summary/report"


def discover_run_summaries(runs_root: str | Path) -> tuple[list[CapabilityRunSummary], list[str]]:
    root = Path(runs_root).expanduser().resolve()
    summaries: list[CapabilityRunSummary] = []
    notices: list[str] = []
    if not root.exists():
        return summaries, [f"runs root does not exist: {root}"]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or directory.name == "progress":
            continue
        summary, notice = _load_run_summary(directory)
        if summary is not None:
            summaries.append(summary)
        if notice is not None:
            notices.append(notice)
    summaries.sort(
        key=lambda item: (
            "" if item.created_at_utc == "unknown" else item.created_at_utc,
            item.run_id,
        )
    )
    return summaries, notices


def write_run_report(summary: CapabilityRunSummary, run_directory: str | Path) -> Path:
    target = Path(run_directory).expanduser().resolve() / "report.html"
    atomic_write_text(target, render_summary_html(summary, history=(summary,)))
    return target


def publish_capability_progress(
    summary: CapabilityRunSummary,
    run_directory: str | Path,
    *,
    runs_root: str | Path | None = None,
    archive_root: str | Path = ".archive",
) -> tuple[Path, Path, Path]:
    """Atomically replace summary/report/dashboard at a validation boundary."""

    run = Path(run_directory).expanduser().resolve()
    root = run.parent if runs_root is None else Path(runs_root).expanduser().resolve()
    summary_path = run / "capability_summary.json"
    write_capability_summary(summary, summary_path)
    report_path = write_run_report(summary, run)
    dashboard_path = build_progress_dashboard(root, archive_root=archive_root)
    return summary_path, report_path, dashboard_path


def build_progress_dashboard(
    runs_root: str | Path = "runs",
    *,
    archive_root: str | Path = ".archive",
) -> Path:
    """Rebuild the static dashboard from summary/read-only adapter evidence."""

    root = Path(runs_root).expanduser().resolve()
    summaries, notices = discover_run_summaries(root)
    if summaries:
        latest = _merge_latest_factor_evidence(summaries, summaries[-1])
    else:
        latest = historical_summary(
            {"status": "no runs"},
            run_id="no-capability-runs",
            source_format="missing",
        )
    inventory = inventory_runs(root, archive_root=archive_root)
    archive_bytes = inventory["archive_bytes"]
    latest = CapabilityRunSummary.from_dict(
        {
            **latest.to_dict(),
            "artifacts": {
                **latest.artifacts,
                "managed_run_tree_bytes": inventory["total_bytes"],
                "managed_budget_bytes": DEFAULT_RUN_ARTIFACT_POLICY.rolling_budget_bytes,
                "archive_bytes": archive_bytes,
            },
        }
    )
    target = root / "progress" / "index.html"
    atomic_write_text(
        target,
        render_summary_html(
            latest,
            title="World-model capability progress",
            history=summaries,
            notices=notices,
        ),
    )
    return target


__all__ = [
    "build_progress_dashboard",
    "discover_run_summaries",
    "publish_capability_progress",
    "render_summary_html",
    "write_run_report",
]
