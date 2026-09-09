from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace

import torch

from scripts.run_capability_planning import main
from world_model.evaluation.capability_planning_runner import (
    _environment,
    _planning_gate_failures,
    factor_planning_rows,
)
from world_model.training.dynamic_set_planning_materializer import (
    CAPABILITY_RECOVERY_HISTORY_FRAMES,
    PLANNING_HISTORY_FRAMES,
    PlanningEnvironmentControls,
    materialize_capability_planning_task,
    validate_public_planning_history,
)


def test_factor_planning_rows_cover_every_cardinality_and_candidate_count() -> None:
    for factor in (
        "sensor_noise",
        "physical_parameters",
        "camera_motion",
        "known_actions",
        "partial_visibility",
        "compositional_holdout",
    ):
        rows = factor_planning_rows(factor)
        assert len(rows) == 12
        assert {(row.object_count, row.candidate_count) for _, row in rows} == {
            (count, candidates) for count in range(1, 7) for candidates in (8, 32)
        }
        assert all(capability.seed == planning.seed for capability, planning in rows)
        assert all(planning.split == "development" for _, planning in rows)


def test_environment_adapter_changes_only_the_selected_family() -> None:
    capability, _ = factor_planning_rows("sensor_noise")[0]
    sensor = _environment("sensor_noise", capability.controls)
    assert sensor.rgb_noise_std == capability.controls.rgb_noise_std
    assert sensor.pixel_dropout_probability == capability.controls.pixel_dropout_probability
    assert sensor.radius_m == 0.21
    assert sensor.camera_motion == "static"
    assert sensor.occlusion_frames == 0
    assert sensor.candidate_delta_velocity_mps == 0.50

    action_capability, _ = factor_planning_rows("known_actions")[0]
    action = _environment("known_actions", action_capability.controls)
    assert action.candidate_delta_velocity_mps == action_capability.controls.impulse_magnitude

    composed_capability, _ = factor_planning_rows("compositional_holdout")[0]
    composed = _environment("compositional_holdout", composed_capability.controls)
    assert composed.rgb_noise_std > 0.0
    assert composed.radius_m == composed_capability.controls.radius_m
    assert composed.camera_motion in {"static", "orbital", "translating"}
    assert composed.occlusion_frames > 0


def test_controlled_planning_materialization_is_deterministic_and_truth_free() -> None:
    capability, row = factor_planning_rows("camera_motion")[2]
    controls = _environment("camera_motion", capability.controls)
    if controls.camera_motion == "static":
        controls = PlanningEnvironmentControls(camera_motion="orbital")
    first = materialize_capability_planning_task(row, controls)
    second = materialize_capability_planning_task(row, controls)

    assert first.materialization_sha256 == second.materialization_sha256
    assert first.public_history.history_sha256 == second.public_history.history_sha256
    assert first.private_oracle.evidence_sha256 == second.private_oracle.evidence_sha256
    assert len(first.public_history.frames) == PLANNING_HISTORY_FRAMES
    assert first.public_history.camera_motion == controls.camera_motion
    assert not torch.equal(
        first.public_history.frames[0].world_from_camera,
        first.public_history.frames[-1].world_from_camera,
    )
    assert validate_public_planning_history(first.public_history) is first.public_history
    public_names = set(asdict(first.public_history))
    assert public_names == {"frames", "previously_dynamic", "history_sha256", "camera_motion"}
    assert "private_oracle" not in public_names


def test_partial_visibility_history_allows_a_complete_post_recovery_window() -> None:
    capability, row = factor_planning_rows("partial_visibility")[0]
    controlled = materialize_capability_planning_task(
        row,
        _environment("partial_visibility", capability.controls),
    )

    assert len(controlled.public_history.frames) == CAPABILITY_RECOVERY_HISTORY_FRAMES
    assert controlled.public_history.frames[-1].frame_index == (
        CAPABILITY_RECOVERY_HISTORY_FRAMES - 1
    )


def test_capability_planning_cli_dry_run_has_no_artifact_side_effect() -> None:
    assert main(["--factor", "known_actions", "--dry-run"]) == 0


def test_failed_task_reports_unmeasured_invariants_without_fabricating_violations() -> None:
    def supported(value: float) -> SimpleNamespace:
        return SimpleNamespace(value=value, support=1)

    gates = SimpleNamespace(
        handle_resolution_by_object_count={
            f"N{count}": supported(0.5 if count == 4 else 1.0) for count in range(1, 7)
        },
        oracle_winner_accuracy_by_candidate_distribution={
            "K8/in_distribution": supported(1.0),
            "K32/in_distribution": supported(5 / 6),
        },
        normalized_regret_median_by_candidate_count={
            "K8": supported(0.0),
            "K32": supported(0.0),
        },
        successful_oracle_goal_success=supported(11 / 12),
    )
    outcomes = [
        SimpleNamespace(
            evaluation=None,
            failure_reason=(
                "checkpoint history is not mature: every active planning target must have a "
                "mature history"
            ),
            row=SimpleNamespace(object_count=4),
        )
    ] + [
        SimpleNamespace(
            evaluation=object(),
            failure_reason=None,
            row=SimpleNamespace(object_count=count),
        )
        for count in range(1, 7)
        for _ in range(2)
        if count != 4
    ]
    failed_invariants = SimpleNamespace(
        serial_vectorized_winner_parity=False,
        pre_action_invariance=False,
        exactly_once_impulse=False,
        action_target_isolation=False,
        conservation=False,
        batch_independence=False,
        source_belief_unchanged=False,
        maximum_cost_difference=math.inf,
    )
    result = SimpleNamespace(
        reduction=SimpleNamespace(gate_metrics=gates),
        outcomes=outcomes,
        invariants=failed_invariants,
    )

    assert _planning_gate_failures(result, compositional=False) == (
        "N4:mature_state_availability",
        "K32:oracle_winner_accuracy",
        "planning_invariants:unmeasured_after_task_failure",
    )


def test_failed_task_distinguishes_unstable_state_from_appearance_resolution() -> None:
    def supported(value: float) -> SimpleNamespace:
        return SimpleNamespace(value=value, support=1)

    gates = SimpleNamespace(
        handle_resolution_by_object_count={
            f"N{count}": supported(0.5 if count == 3 else 1.0) for count in range(1, 7)
        },
        oracle_winner_accuracy_by_candidate_distribution={
            "K8/in_distribution": supported(1.0),
            "K32/in_distribution": supported(1.0),
        },
        normalized_regret_median_by_candidate_count={
            "K8": supported(0.0),
            "K32": supported(0.0),
        },
        successful_oracle_goal_success=supported(1.0),
    )
    outcomes = [
        SimpleNamespace(
            evaluation=None,
            failure_reason="checkpoint did not preserve one stable observed post-start birth",
            row=SimpleNamespace(object_count=3),
        )
    ]
    result = SimpleNamespace(
        reduction=SimpleNamespace(gate_metrics=gates),
        outcomes=outcomes,
        invariants=SimpleNamespace(
            serial_vectorized_winner_parity=False,
            pre_action_invariance=False,
            exactly_once_impulse=False,
            action_target_isolation=False,
            conservation=False,
            batch_independence=False,
            source_belief_unchanged=False,
            maximum_cost_difference=math.inf,
        ),
    )

    assert _planning_gate_failures(result, compositional=True) == (
        "N3:stable_state_availability",
        "planning_invariants:unmeasured_after_task_failure",
    )
