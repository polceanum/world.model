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


def _score_value(summary: CapabilityRunSummary, name: str) -> float | None:
    value = summary.scores.get(name)
    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _svg_line_chart(points: Sequence[tuple[str, float]], *, title: str) -> str:
    width, height = 720, 220
    left, right, top, bottom = 58, 18, 28, 42
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
        x = left + index * (width - left - right) / span_x
        y = top + (high - value) * (height - top - bottom) / (high - low)
        coords.append((x, y))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4"><title>{_escape(label)}: '
        f"{value:.8g}</title></circle>"
        for (label, value), (x, y) in zip(points, coords, strict=True)
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(title)}">'
        f'<text x="{left}" y="18" class="chart-title">{_escape(title)}</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>'
        f'<text x="4" y="{top + 5}">{high:.3g}</text><text x="4" y="{height - bottom}">{low:.3g}</text>'
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
            charts.append(_svg_line_chart(points, title=title))
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
            _svg_line_chart(other_points, title="Overall capability score across runs"),
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
) -> str:
    cells = summary.cell_metrics
    if not cells:
        return f'<div class="empty">{_escape(title)}: unmeasured</div>'
    entries = list(cells.items())
    values = [
        value for _, metrics in entries if (value := _cell_value(metrics, metric)) is not None
    ]
    minimum = min(values, default=0.0)
    maximum = max(values, default=1.0)
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
            spread = maximum - minimum
            if spread <= 1.0e-12:
                ratio = 0.5
            else:
                ratio = min(1.0, max(0.0, (value - minimum) / spread))
                if not lower_is_better:
                    ratio = 1.0 - ratio
            red = int(54 + 175 * ratio)
            green = int(175 - 100 * ratio)
            color = f"rgb({red},{green},92)"
            display = _format_number(value)
        blocks.append(
            f'<g><rect x="{x}" y="{y}" width="102" height="42" rx="6" fill="{color}"/>'
            f'<text x="{x + 5}" y="{y + 16}" class="cell-label">{_escape(label)}</text>'
            f'<text x="{x + 5}" y="{y + 33}" class="cell-value">{display}</text></g>'
        )
    rows = math.ceil(len(entries) / 6)
    return (
        f'<svg class="heatmap" viewBox="0 0 700 {54 * rows + 44}" role="img" '
        f'aria-label="{_escape(title)}">'
        f'<text x="18" y="20" class="chart-title">{_escape(title)}</text>'
        f"{''.join(blocks)}</svg>"
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
    animation_source: str | None = None
    if not isinstance(qualitative.get("animations"), list) or not qualitative.get("animations"):
        for summary in reversed(summaries):
            if isinstance(summary.qualitative.get("animations"), list) and summary.qualitative.get(
                "animations"
            ):
                qualitative["animations"] = summary.qualitative["animations"]
                animation_source = summary.run_id
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
                **({} if animation_source is None else {"animation_source_run": animation_source}),
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
    if not groups:
        return '<div class="empty">Planning slices: unmeasured</div>'
    latency_rows: list[str] = []
    rows = []
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
        for item in slices:
            if not isinstance(item, Mapping):
                continue
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
    latency_table = (
        "<h3>Planning efficiency and numerical agreement</h3>"
        "<table><thead><tr><th>Factor</th><th>Status</th><th>K8 latency (s)</th>"
        "<th>K32 latency (s)</th><th>Maximum cost difference</th></tr></thead>"
        f"<tbody>{''.join(latency_rows)}</tbody></table>"
    )
    slice_table = (
        "<h3>Planning quality by cardinality</h3>"
        "<table><thead><tr><th>Factor</th><th>Count</th><th>Candidates</th><th>Winner accuracy</th>"
        f"<th>Median regret</th><th>Goal success</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    return latency_table + slice_table


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


def _animation_cards(summary: CapabilityRunSummary) -> str:
    animations = summary.qualitative.get("animations")
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
        x_bounds = bounds.get("x")
        z_bounds = bounds.get("z")
        if (
            not isinstance(x_bounds, list)
            or len(x_bounds) != 2
            or not isinstance(z_bounds, list)
            or len(z_bounds) != 2
        ):
            continue
        try:
            x_low, x_high = (float(value) for value in x_bounds)
            z_low, z_high = (float(value) for value in z_bounds)
        except (TypeError, ValueError):
            continue
        if not all(map(math.isfinite, (x_low, x_high, z_low, z_high))):
            continue
        if x_high <= x_low or z_high <= z_low or not isinstance(frames[0], Mapping):
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
                    if isinstance(point, list) and len(point) == 3 and isinstance(point[0], int):
                        object_ids.add(point[0])
        first = frames[0]

        def initial(
            role: str,
            object_id: int,
            *,
            first_frame: Mapping[str, Any] = first,
            x_min: float = x_low,
            x_max: float = x_high,
            z_min: float = z_low,
            z_max: float = z_high,
        ) -> tuple[float, float] | None:
            points = first_frame.get(role)
            if not isinstance(points, list):
                return None
            for point in points:
                if not isinstance(point, list) or len(point) != 3 or point[0] != object_id:
                    continue
                try:
                    x_value, z_value = float(point[1]), float(point[2])
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(x_value) or not math.isfinite(z_value):
                    return None
                x = 28.0 + 280.0 * (x_value - x_min) / (x_max - x_min)
                y = 202.0 - 184.0 * (z_value - z_min) / (z_max - z_min)
                return x, y
            return None

        circles: list[str] = []
        for palette_index, object_id in enumerate(sorted(object_ids)):
            color = _ANIMATION_PALETTE[palette_index % len(_ANIMATION_PALETTE)]
            for role, radius in (("truth", 7), ("model", 4)):
                coordinates = initial(role, object_id)
                display = "none" if coordinates is None else "inline"
                x, y = coordinates or (0.0, 0.0)
                circles.append(
                    f'<circle class="animation-{role}" data-role="{role}" '
                    f'data-object-id="{object_id}" cx="{x:.2f}" cy="{y:.2f}" '
                    f'r="{radius}" style="--object-color:{color};display:{display}"/>'
                )
        event_kinds = animation.get("events", [])
        event_labels = sorted(
            {
                str(item.get("kind"))
                for item in event_kinds
                if isinstance(item, Mapping) and item.get("kind")
            }
        )
        cards.append(
            '<article class="animation-card" '
            f'data-world-animation="{animation_index}">'
            f"<h3>{_escape(animation.get('label', 'example')).title()}</h3>"
            f'<p class="animation-meta">{_escape(animation.get("episode", "unknown episode"))}'
            f" · N={_escape(animation.get('object_count', '?'))}"
            f" · RMSE {_format_number(animation.get('current_position_rmse_m'))} m</p>"
            '<svg class="world-animation" viewBox="0 0 336 220" role="img" '
            f'aria-label="{_escape(animation.get("label", "example"))} model versus reference world trajectory">'
            '<rect x="28" y="18" width="280" height="184" rx="7" class="animation-stage"/>'
            '<path d="M28 64H308M28 110H308M28 156H308M98 18V202M168 18V202M238 18V202" '
            'class="animation-grid-lines"/>'
            f"{''.join(circles)}"
            '<text x="34" y="34" class="animation-clock" data-animation-clock>frame 0</text>'
            '<text x="34" y="194" class="animation-event" data-animation-event></text>'
            "</svg>"
            '<div class="animation-legend"><span class="model-key">● model</span>'
            '<span class="truth-key">○ reference</span><span>world X–Z</span></div>'
            f'<p class="animation-events">Events: {_escape(", ".join(event_labels) or "none")}</p>'
            '<div class="animation-controls">'
            f'<button type="button" data-animation-toggle="{animation_index}">Pause</button>'
            f'<button type="button" data-animation-replay="{animation_index}">Replay</button>'
            "</div></article>"
        )
    if not cards:
        return ""
    return (
        "<section><h2>Lightweight trajectory examples</h2>"
        '<p class="animation-intro">Filled markers are model state; rings are private '
        "reference positions opened only after public RGB-D inference. Animations use bounded "
        "downsampled vector keyframes—no video or retained image frames.</p>"
        f'<div class="animation-grid">{"".join(cards)}</div></section>'
    )


def _summary_section(summary: CapabilityRunSummary) -> str:
    candidate_score = _score_value(summary, "candidate")
    incumbent_score = _score_value(summary, "incumbent")
    horizon = summary.horizon_curves.get("candidate_position_rmse_m", {})
    horizon_points = (
        [(f"{key}s", float(value)) for key, value in horizon.items()]
        if isinstance(horizon, Mapping)
        else []
    )
    resources = summary.resources
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
      <div class="score"><small>candidate / incumbent</small><strong>{_format_number(candidate_score)} / {_format_number(incumbent_score)}</strong><span>{_escape(summary.scores.get("selected", "unselected"))}</span></div>
    </section>
    <section class="grid three">
      <article><h2>Accuracy</h2><p class="metric">{_format_number(candidate_score)}</p><p>Lower-is-better capability score</p></article>
      <article><h2>Uncertainty</h2><p class="metric">{coverage_text}</p><p>Observed nominal-90% coverage range</p></article>
      <article><h2>Planning parity</h2><p class="metric">{_escape(summary.planning.get("serial_vectorized_winner_parity", "—"))}</p><p>Maximum cost difference {_format_number(summary.planning.get("maximum_cost_difference"))}</p></article>
    </section>
    <section class="grid two"><article><h2>Capability coverage</h2>{_factor_table(summary)}</article><article><h2>Horizon error</h2>{_svg_line_chart(horizon_points, title="Position RMSE across horizon")}</article></section>
    <section><h2>Factor performance</h2>{_factor_metric_matrix(summary)}</section>
    <section><h2>Physical behavior</h2>{_heatmap(summary, metric="current_position_rmse_m", title="Current-position RMSE by count/contact/lifecycle cell", lower_is_better=True)}{_heatmap(summary, metric="uncertainty_90_coverage", title="90% uncertainty coverage by count/contact/lifecycle cell", lower_is_better=False)}</section>
    <section><h2>Downstream planning</h2>{_planning_table(summary)}</section>
    <section class="grid two"><article><h2>Efficiency and storage</h2><dl>
      <dt>Perception latency</dt><dd>{_format_number(resources.get("perception_latency_seconds"))} s</dd>
      <dt>Six-horizon rollout</dt><dd>{_format_number(resources.get("six_horizon_rollout_seconds"))} s</dd>
      <dt>State-only N=16</dt><dd>{_format_number(_n16_latency(resources))} s</dd>
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
:root{color-scheme:dark;--bg:#0b1017;--panel:#141c27;--muted:#91a0b5;--ink:#f5f7fb;--accent:#69d6c5;--line:#344154}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,#172a37,var(--bg) 42%);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;align-items:end;border-top:3px solid var(--accent);padding-top:18px}.hero h1{font-size:clamp(26px,4vw,48px);margin:.1em 0}.hero p,.metric+ p{color:var(--muted)}.eyebrow,.tag{font-size:11px;letter-spacing:.08em;text-transform:uppercase}.score{background:var(--panel);padding:18px 22px;border-radius:12px;display:grid;min-width:240px}.score strong,.metric{font-size:25px;color:var(--accent)}.grid{display:grid;gap:14px;margin:14px 0}.two{grid-template-columns:repeat(2,minmax(0,1fr))}.three{grid-template-columns:repeat(3,minmax(0,1fr))}article,section:not(.hero){background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0;overflow:auto}section.grid{background:none;border:0;padding:0}section.grid article{margin:0}h2{margin:0 0 12px;font-size:16px}h3{font-size:13px;color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid var(--line);padding:7px}.tag{padding:3px 6px;border-radius:9px;background:#303947}.tag.measured,.tag.passed{background:#165449}.tag.failed{background:#6b2737}.metric-matrix td{font-variant-numeric:tabular-nums}.metric-pass{background:#123d35}.metric-fail{background:#51232e;color:#ffdce4}.metric-info{background:#1c2d3b}.unmeasured{color:var(--muted)}.empty{color:var(--muted);padding:24px;text-align:center;border:1px dashed var(--line);border-radius:8px}svg{width:100%;min-width:520px}svg line{stroke:var(--line)}svg polyline{fill:none;stroke:var(--accent);stroke-width:3}svg circle{fill:var(--accent)}svg text{fill:var(--muted);font-size:11px}.chart-title{fill:var(--ink);font-size:13px}.cell-label{fill:white;font-size:8px}.cell-value{fill:white;font-size:11px;font-weight:700}dl{display:grid;grid-template-columns:1fr 1fr;gap:7px}dt{color:var(--muted)}dd{margin:0;text-align:right}.animation-intro,.animation-meta,.animation-events{color:var(--muted)}.animation-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.animation-card{margin:0!important;padding:12px!important}.animation-card h3{margin:0;color:var(--ink);text-transform:capitalize}.animation-meta{min-height:38px;font-size:12px}.world-animation{display:block;min-width:0;background:#0a1119;border:1px solid var(--line);border-radius:8px}.animation-stage{fill:#0c1621;stroke:#344154}.animation-grid-lines{fill:none;stroke:#253444;stroke-width:1}.animation-model{fill:var(--object-color);stroke:#081018;stroke-width:1.5}.animation-truth{fill:none;stroke:var(--object-color);stroke-width:2;stroke-dasharray:2 2}.animation-clock{fill:#c9d3df;font-variant-numeric:tabular-nums}.animation-event{fill:#f3bc61;font-weight:700}.animation-legend{display:flex;gap:12px;margin-top:8px;color:var(--muted);font-size:12px}.model-key{color:var(--accent)}.truth-key{color:#f3bc61}.animation-events{min-height:38px;font-size:12px}.animation-controls{display:flex;gap:8px}.animation-controls button{border:1px solid var(--line);border-radius:7px;background:#1c2d3b;color:var(--ink);padding:5px 11px;cursor:pointer}.animation-controls button:hover{border-color:var(--accent)}.run-list a{color:var(--accent)}.notice{border-left:3px solid #f3bc61;padding-left:10px;color:var(--muted)}footer{color:var(--muted);padding:20px 0}@media(max-width:900px){.animation-grid{grid-template-columns:1fr}}@media(max-width:760px){.two,.three{grid-template-columns:1fr}.hero{display:block}.score{margin-top:12px}}@media(prefers-reduced-motion:reduce){.animation-controls button{outline:1px solid var(--muted)}}
"""


_ANIMATION_SCRIPT = r"""
<script>
(() => {
  const dataNode = document.getElementById("capability-run-summary");
  if (!dataNode) return;
  const summary = JSON.parse(dataNode.textContent);
  const animations = summary.qualitative && summary.qualitative.animations;
  if (!Array.isArray(animations)) return;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const states = new Map();
  let pendingFrame = null;
  const indexPoints = points => new Map((Array.isArray(points) ? points : []).map(p => [String(p[0]), p]));
  const position = (point, bounds) => ({
    x: 28 + 280 * (point[1] - bounds.x[0]) / (bounds.x[1] - bounds.x[0]),
    y: 202 - 184 * (point[2] - bounds.z[0]) / (bounds.z[1] - bounds.z[0])
  });
  const render = (root, animation, progress) => {
    const frames = animation.frames;
    if (!Array.isArray(frames) || frames.length === 0) return;
    const scaled = Math.min(frames.length - 1, progress * (frames.length - 1));
    const leftIndex = Math.floor(scaled);
    const rightIndex = Math.min(frames.length - 1, leftIndex + 1);
    const mix = scaled - leftIndex;
    const left = frames[leftIndex];
    const right = frames[rightIndex];
    for (const role of ["truth", "model"]) {
      const leftPoints = indexPoints(left[role]);
      const rightPoints = indexPoints(right[role]);
      for (const circle of root.querySelectorAll(`circle[data-role="${role}"]`)) {
        const objectId = circle.dataset.objectId;
        const a = leftPoints.get(objectId);
        const b = rightPoints.get(objectId);
        const point = a && b
          ? [a[0], a[1] + mix * (b[1] - a[1]), a[2] + mix * (b[2] - a[2])]
          : (mix < 0.5 ? a : b);
        if (!point) {
          circle.style.display = "none";
          continue;
        }
        const screen = position(point, animation.bounds);
        circle.setAttribute("cx", screen.x.toFixed(2));
        circle.setAttribute("cy", screen.y.toFixed(2));
        circle.style.display = "inline";
      }
    }
    const actualFrame = left.frame + mix * (right.frame - left.frame);
    const clock = root.querySelector("[data-animation-clock]");
    if (clock) clock.textContent = `frame ${Math.round(actualFrame)} · ${(actualFrame / 20).toFixed(2)} s`;
    const eventNode = root.querySelector("[data-animation-event]");
    if (eventNode) {
      const nearby = (animation.events || [])
        .filter(event => Math.abs(event.frame - actualFrame) < 1.5)
        .map(event => event.kind);
      eventNode.textContent = [...new Set(nearby)].join(" · ");
    }
  };
  for (const root of document.querySelectorAll("[data-world-animation]")) {
    const index = Number(root.dataset.worldAnimation);
    const animation = animations[index];
    if (!animation) continue;
    const state = {progress: 0, paused: reducedMotion, previous: performance.now()};
    states.set(index, state);
    const toggle = document.querySelector(`[data-animation-toggle="${index}"]`);
    const replay = document.querySelector(`[data-animation-replay="${index}"]`);
    if (toggle) {
      toggle.textContent = state.paused ? "Play" : "Pause";
      toggle.addEventListener("click", () => {
        state.paused = !state.paused;
        state.previous = performance.now();
        toggle.textContent = state.paused ? "Play" : "Pause";
        schedule();
      });
    }
    if (replay) replay.addEventListener("click", () => {
      state.progress = 0;
      state.paused = false;
      state.previous = performance.now();
      if (toggle) toggle.textContent = "Pause";
      render(root, animation, 0);
      schedule();
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
      const index = Number(root.dataset.worldAnimation);
      const animation = animations[index];
      const state = states.get(index);
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
    notice_rows = "".join(f'<p class="notice">{_escape(item)}</p>' for item in notices)
    notice_html = (
        f"<details><summary>{len(notices)} protected/missing artifact notices</summary>"
        f"{notice_rows}</details>"
        if notices
        else ""
    )
    animation_script = _ANIMATION_SCRIPT if _animation_cards(summary) else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>{_escape(title)}</title><style>{_STYLE}</style></head><body><main>{_summary_section(summary)}{notice_html}<section class="run-list"><h2>Capability trend</h2>{_trend_charts(history)}<ol>{history_rows}</ol></section><footer>Portable report · schema {CAPABILITY_SUMMARY_SCHEMA} · no external assets</footer></main><script type="application/json" id="capability-run-summary">{embedded}</script>{animation_script}</body></html>"""


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
