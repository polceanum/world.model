from __future__ import annotations

import json
from pathlib import Path

import torch

from world_model.belief import BeliefFactory
from world_model.dynamics import DynamicsModel
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
)
from world_model.evaluation.long_horizon import (
    LONG_HORIZON_ANIMATION_MAX_BYTES,
    LONG_HORIZON_BOUNDS,
    _bounded_dynamics,
    _publish_terminal_evidence,
    _truth_rollout,
    default_long_horizon_scenarios,
    long_horizon_scenario_sha256,
)
from world_model.utils.io import atomic_write_text


def test_long_horizon_manifest_is_deterministic_and_causally_ordered() -> None:
    first = default_long_horizon_scenarios()
    second = default_long_horizon_scenarios()

    assert first == second
    assert long_horizon_scenario_sha256(first) == long_horizon_scenario_sha256(second)
    assert {scenario.duration_s for scenario in first} == {4.0, 8.0}
    assert all(len(scenario.actions) >= 2 for scenario in first)
    assert all(
        tuple(action.timestamp_s for action in scenario.actions)
        == tuple(sorted(action.timestamp_s for action in scenario.actions))
        for scenario in first
    )


def test_truth_rollout_applies_declared_actions_without_retaining_frames() -> None:
    scenario = default_long_horizon_scenarios()[0]
    queries = torch.tensor([0.5, 0.8, 1.0], dtype=torch.float32)

    positions, velocities, collisions, events = _truth_rollout(scenario, queries)

    assert positions.shape == (3, len(scenario.object_id), 3)
    assert velocities.shape == positions.shape
    assert collisions.shape == (3,)
    assert torch.isfinite(positions).all() and torch.isfinite(velocities).all()
    assert [event["kind"] for event in events].count("known action") == 1
    assert not any("rgb" in event or "depth" in event for event in events)


def test_vector_animation_budget_is_smaller_than_one_tenth_megabyte() -> None:
    assert LONG_HORIZON_ANIMATION_MAX_BYTES == 96 * 1024
    manifest_bytes = len(
        json.dumps([scenario.name for scenario in default_long_horizon_scenarios()]).encode("utf-8")
    )
    assert manifest_bytes < LONG_HORIZON_ANIMATION_MAX_BYTES


def test_rebinding_world_boundaries_does_not_reload_checkpoint_plane_offsets() -> None:
    belief = BeliefFactory(max_objects=1).create(batch_size=1)
    source = DynamicsModel.from_belief(
        belief,
        world_bounds=((-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)),
    )

    bounded = _bounded_dynamics(source)

    assert bounded.config.world_bounds == LONG_HORIZON_BOUNDS
    assert torch.equal(
        bounded.events.resolver.plane_offsets,
        torch.tensor([-2.0, -2.0, 0.0, -2.5, -1.25, -1.25]),
    )


def test_terminal_evidence_records_exact_manifest_inclusive_size(tmp_path: Path) -> None:
    run = tmp_path / "long-run"
    run.mkdir()
    atomic_write_text(run / "long_horizon_report.json", "{}\n")
    summary = CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run.name,
        created_at_utc="2026-09-10T00:00:00+00:00",
        lifecycle_status="completed",
        outcome="long_horizon_pilot_passed",
        source_format="world_model_long_horizon_report_v1",
        configuration={"state_first": True},
        provenance={"planning_used_as_training_loss": False},
        scores={"candidate": 0.0, "incumbent": 0.0, "selected": "incumbent"},
        factor_metrics={},
        cell_metrics={},
        horizon_curves={},
        uncertainty={},
        planning={},
        resources={},
        artifacts={"run_bytes": 0, "archive_bytes": 0},
        selection={"selected": "incumbent"},
        failure_attribution={},
        qualitative={},
        unsupported_claims=(),
        scope_limitations=(),
    ).validate()

    published = _publish_terminal_evidence(summary, run, failed=False)
    actual_bytes = sum(path.stat().st_size for path in run.iterdir() if path.is_file())

    assert published.artifacts["run_bytes"] == actual_bytes
