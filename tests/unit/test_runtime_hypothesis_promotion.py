from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.compare_runtime_hypothesis_evaluations import (
    CapturedReport,
    _canonical_json_value,
    _canonical_sha256,
    _expected_protocol,
    _expected_runtime_environment,
    _expected_runtime_policy,
    _validate_arm,
    compare_evaluation_reports,
    compare_runtime_hypothesis_metrics,
)
from world_model.datasets.splits import make_seed_manifest
from world_model.evaluation.evaluator import (
    _EVALUATION_METRIC_SCHEMA_VERSION,
    _PER_SCENARIO_METRIC_SCHEMA,
    _primary_physical_metrics,
    _primary_physical_metrics_hash_exclusion_declaration,
)
from world_model.simulator.sphere_world import SphereWorldConfig
from world_model.utils.config import load_config
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION

HORIZONS = ("0.100s",)
SCENARIOS = ("baseline",)


def _paired_metrics() -> tuple[dict[str, float], dict[str, float]]:
    reference: dict[str, float] = {
        "posterior_current_position_rmse_m": 0.2,
    }
    for prefix in ("", "scenario_baseline_"):
        model = f"{prefix}model@0.100s"
        reference.update(
            {
                f"{model}_position_coordinate_count": 30.0,
                f"{model}_position_calibration_coordinate_count": 30.0,
                f"{model}_velocity_coordinate_count": 30.0,
                f"{model}_velocity_object_frame_count": 10.0,
                f"{prefix}forecast_target_coverage@0.100s": 0.9,
                f"{prefix}tracked_forecast_active_coverage@0.100s": 0.9,
                f"{prefix}forecast_target_count@0.100s": 10.0,
                f"{prefix}forecast_tracked_count@0.100s": 10.0,
                f"{prefix}forecast_active_count@0.100s": 9.0,
                f"{prefix}forecast_identity@0.100s_eligible_count": 10.0,
                f"{prefix}forecast_identity@0.100s_association_coverage": 0.8,
                f"{prefix}forecast_identity@0.100s_mismatch_rate": 0.1,
                f"{prefix}collision@0.100s_f1": 0.5,
                f"{prefix}collision@0.100s_true_positive_count": 2.0,
                f"{prefix}collision@0.100s_false_negative_count": 1.0,
                f"{prefix}collision@0.100s_true_negative_count": 6.0,
                f"{prefix}collision@0.100s_false_positive_count": 1.0,
                f"{prefix}model_dropped_forecast_count@0.100s": 1.0,
            }
        )
        for axis_suffix in ("", "_x", "_y", "_z"):
            reference.update(
                {
                    f"{model}_position{axis_suffix}_rmse_m": 0.3,
                    f"{model}_velocity{axis_suffix}_rmse_mps": 0.8,
                    f"{model}_position{axis_suffix}_gaussian_nll": 0.4,
                    f"{model}_position{axis_suffix}_calibration_error90": 0.05,
                    f"{model}_position{axis_suffix}_sharpness_std": 0.2,
                }
            )
        for axis_name in ("x", "y", "z"):
            reference.update(
                {
                    f"{model}_position_{axis_name}_count": 10.0,
                    f"{model}_position_{axis_name}_calibration_coordinate_count": 10.0,
                    f"{model}_velocity_{axis_name}_count": 10.0,
                }
            )

    candidate = deepcopy(reference)
    for name, value in tuple(candidate.items()):
        if name.endswith(
            ("_rmse_m", "_rmse_mps", "_gaussian_nll", "_calibration_error90")
        ) and name.startswith(("model@", "scenario_baseline_model@")):
            candidate[name] = value - 0.01
    for prefix in ("", "scenario_baseline_"):
        candidate[f"{prefix}forecast_identity@0.100s_association_coverage"] = 0.81
        candidate[f"{prefix}forecast_identity@0.100s_mismatch_rate"] = 0.09
        candidate[f"{prefix}collision@0.100s_f1"] = 0.51
        candidate[f"{prefix}model_dropped_forecast_count@0.100s"] = 0.0
        candidate[f"{prefix}collision@0.100s_true_positive_count"] = 3.0
        candidate[f"{prefix}collision@0.100s_false_negative_count"] = 0.0

    candidate.update(
        {
            "runtime_hypothesis_forecast_anchor_count": 8.0,
            "runtime_hypothesis_axis_x_learned_count": 2.0,
            "runtime_hypothesis_axis_x_constant_velocity_count": 3.0,
            "runtime_hypothesis_axis_x_damped_constant_velocity_count": 1.0,
            "runtime_hypothesis_axis_x_ballistic_contact_count": 0.0,
            "runtime_hypothesis_axis_x_supported_count": 6.0,
            "runtime_hypothesis_axis_x_fallback_count": 2.0,
            "runtime_hypothesis_axis_x_learned_composed_step_count": 4.0,
            "runtime_hypothesis_axis_x_constant_velocity_composed_step_count": 5.0,
            "runtime_hypothesis_axis_x_damped_constant_velocity_composed_step_count": 1.0,
            "runtime_hypothesis_axis_x_ballistic_contact_composed_step_count": 0.0,
            "runtime_hypothesis_axis_x_composed_total_step_count": 10.0,
            "runtime_hypothesis_axis_x_composed_fallback_step_count": 2.0,
            "runtime_hypothesis_axis_x_composition_grid_fallback_count": 0.0,
            "runtime_hypothesis@0.100s_axis_x_learned_count": 2.0,
            "runtime_hypothesis@0.100s_axis_x_constant_velocity_count": 3.0,
            "runtime_hypothesis@0.100s_axis_x_damped_constant_velocity_count": 1.0,
            "runtime_hypothesis@0.100s_axis_x_ballistic_contact_count": 0.0,
            "runtime_hypothesis@0.100s_axis_x_supported_count": 6.0,
            "runtime_hypothesis@0.100s_axis_x_fallback_count": 2.0,
            "runtime_hypothesis_regime_free_composed_step_count": 7.0,
            "runtime_hypothesis_regime_ground_contact_composed_step_count": 3.0,
            "runtime_hypothesis_regime_pair_contact_composed_step_count": 0.0,
            "runtime_hypothesis_regime_collision_composed_step_count": 0.0,
            "runtime_hypothesis_regime_occluded_composed_step_count": 0.0,
            "runtime_hypothesis_regime_externally_actuated_composed_step_count": 0.0,
        }
    )
    return reference, candidate


def _compare(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    return compare_runtime_hypothesis_metrics(
        reference,
        candidate,
        horizons=HORIZONS,
        scenarios=SCENARIOS,
        axes=(0,),
    )


def test_runtime_hypothesis_comparison_accepts_complete_nonregression() -> None:
    reference, candidate = _paired_metrics()

    result = _compare(reference, candidate)

    assert result["physical_promotion_eligible"] is True
    assert result["failure_count"] == 0
    assert result["runtime_nonlearned_selection_count"] == 4.0
    assert result["runtime_nonlearned_composed_step_count"] == 6.0


def test_runtime_hypothesis_comparison_rejects_axis_regression() -> None:
    reference, candidate = _paired_metrics()
    candidate["model@0.100s_position_x_rmse_m"] = 0.31

    result = _compare(reference, candidate)

    assert result["physical_promotion_eligible"] is False
    assert any(
        failure["metric"] == "model@0.100s_position_x_rmse_m" for failure in result["failures"]
    )


def test_runtime_hypothesis_comparison_rejects_scenario_uncertainty_regression() -> None:
    reference, candidate = _paired_metrics()
    candidate["scenario_baseline_model@0.100s_position_z_gaussian_nll"] = 0.5

    result = _compare(reference, candidate)

    assert result["physical_promotion_eligible"] is False
    assert any(
        failure["metric"] == "scenario_baseline_model@0.100s_position_z_gaussian_nll"
        for failure in result["failures"]
    )


def test_runtime_hypothesis_comparison_requires_exact_support() -> None:
    reference, candidate = _paired_metrics()
    candidate["model@0.100s_velocity_x_count"] = 9.0

    result = _compare(reference, candidate)

    assert result["physical_promotion_eligible"] is False
    assert any(
        failure["metric"] == "model@0.100s_velocity_x_count"
        for failure in result["support_failures"]
    )


def test_runtime_hypothesis_comparison_requires_nonlearned_use() -> None:
    reference, candidate = _paired_metrics()
    candidate["runtime_hypothesis_axis_x_learned_count"] = 6.0
    candidate["runtime_hypothesis_axis_x_constant_velocity_count"] = 0.0
    candidate["runtime_hypothesis_axis_x_damped_constant_velocity_count"] = 0.0
    candidate["runtime_hypothesis_axis_x_learned_composed_step_count"] = 10.0
    candidate["runtime_hypothesis_axis_x_constant_velocity_composed_step_count"] = 0.0
    candidate["runtime_hypothesis_axis_x_damped_constant_velocity_composed_step_count"] = 0.0

    result = _compare(reference, candidate)

    assert result["physical_promotion_eligible"] is False
    assert result["runtime_usage_failures"] == [
        {
            "metric": "runtime_hypothesis_nonlearned_use",
            "direction": "positive_selected_and_composed_use_required",
            "candidate": {"selected": 0.0, "composed_steps": 0.0},
            "passed": False,
        }
    ]


def test_runtime_hypothesis_comparison_rejects_extra_nonruntime_metric() -> None:
    reference, candidate = _paired_metrics()
    candidate["silently_added_metric"] = 1.0

    try:
        _compare(reference, candidate)
    except ValueError as error:
        assert "schemas differ" in str(error)
    else:
        raise AssertionError("schema mismatch was accepted")


def test_runtime_hypothesis_policy_is_fully_config_bound() -> None:
    config = load_config(
        "configs/tiny_overfit.yaml",
        overrides=[
            "runtime.hypothesis_evidence_horizons_seconds=[0.05]",
            "runtime.hypothesis_axis_independent_axes=[0]",
            "runtime.hypothesis_local_applicability_enabled=true",
            "runtime.hypothesis_composition_step_seconds=0.05",
        ],
    )

    policy = _expected_runtime_policy(config)
    fingerprint = policy.pop("fingerprint_sha256")

    assert fingerprint == _canonical_sha256(policy)
    assert policy["local_applicability_enabled"] is True
    assert policy["selection_locality"] == (
        "persistent_entity_axis_interaction_regime_exact_horizon"
    )
    assert policy["composition"] == "bounded_short_step_coherent_state"
    assert [candidate["name"] for candidate in policy["candidates"]] == [
        "learned",
        "constant_velocity",
        "damped_constant_velocity",
        "ballistic_contact",
    ]


def _validation_config():
    return load_config(
        "configs/axis_gated_updater_repair_cpu.yaml",
        overrides=[
            "runtime.hypothesis_evidence_horizons_seconds=[0.05]",
            "runtime.hypothesis_axis_independent_axes=[0]",
            "runtime.hypothesis_local_applicability_enabled=true",
            "runtime.hypothesis_composition_step_seconds=0.05",
            "evaluation.horizons_seconds=[0.1]",
            "training.horizon_weights=[1.0]",
        ],
    )


def _arm_payload(
    *,
    runtime_pool: bool,
    supplied_metrics: dict[str, float] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    config = _validation_config()
    scenarios = list(config.simulator.scenario_mixture)
    seeds = list(make_seed_manifest("validation", 32).seeds)
    metrics: dict[str, object] = {
        **({} if supplied_metrics is None else deepcopy(supplied_metrics)),
        "nonfinite_output_count": 0.0,
        "evaluated_episodes": 32.0,
        "injected_perturbation_batch_updates": 0.0,
        "recovery_probe_evaluated_episodes": 0.0,
        "recovery_probe_nonfinite_output_count": 0.0,
    }
    for scenario in scenarios:
        metrics[f"scenario_{scenario}_episode_count"] = 4.0
    primary = _primary_physical_metrics(metrics)
    checkpoint_sha = "a" * 64
    metadata: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_byte_count": 123,
        "checkpoint_identity_source": "captured_pre_evaluation_immutable_byte_snapshot",
        "checkpoint_step": 512,
        "checkpoint_source_provenance": {"commit": "0" * 40},
        "checkpoint_simulator_version": SIMULATOR_VERSION,
        "checkpoint_specification_version": SPECIFICATION_VERSION,
        "evaluation_metric_schema_version": _EVALUATION_METRIC_SCHEMA_VERSION,
        "resolved_evaluation_config_sha256": _canonical_sha256(config.to_dict()),
        "resolved_evaluation_protocol": _expected_protocol(
            config,
            checkpoint_sha256=checkpoint_sha,
            runtime_hypothesis_pool=runtime_pool,
        ),
        "simulator_version": SIMULATOR_VERSION,
        "evaluation_simulator_version": SIMULATOR_VERSION,
        "evaluation_specification_version": SPECIFICATION_VERSION,
        "scenario_mixture": scenarios,
        "resolved_scenarios": _canonical_json_value(
            {
                scenario: asdict(
                    SphereWorldConfig.from_config(config)
                    .for_scenario(scenario)
                    .for_distribution("in_distribution")
                )
                for scenario in scenarios
            }
        ),
        "per_scenario_metrics_schema": _PER_SCENARIO_METRIC_SCHEMA,
        "per_scenario_metrics_status": "diagnostic_only_not_checkpoint_promotion_complete",
        "per_scenario_metrics_known_omissions": [
            "nonfinite_evidence",
            "physical_baselines",
            "configured_support_floor_markers",
        ],
        "per_scenario_metrics_scenarios": scenarios,
        "per_scenario_metrics_horizons": [
            f"{value:.3f}s" for value in config.evaluation.horizons_seconds
        ],
        "evaluation_episode_scenarios": [scenarios[int(seed) % len(scenarios)] for seed in seeds],
        "split": "validation",
        "episodes": 32,
        "batches": 4,
        "device": "cpu",
        "precision": "float32",
        "evaluation_runtime_environment": _expected_runtime_environment("cpu"),
        "rgb_only": True,
        "oracle_runtime_input_used": False,
        "primary_online_pass_evaluator_state_perturbation_free": True,
        "primary_online_pass_intervention_free_scope": (
            "evaluator_injected_state_perturbations_only"
        ),
        "recovery_probe_enabled": False,
        "evaluation_perturbations_applied": False,
        "runtime_hypothesis_pool_enabled": runtime_pool,
        "runtime_hypothesis_pool_policy": (
            _expected_runtime_policy(config) if runtime_pool else None
        ),
        "evaluation_seed_protocol": "standard",
        "evaluation_seed_role": "standard_validation_evaluation",
        "evaluation_seed_offset": 0,
        "evaluation_seed_count": 32,
        "evaluation_seed_first": seeds[0],
        "evaluation_seed_last": seeds[-1],
        "evaluation_episode_seeds": seeds,
        "evaluation_seed_overlaps_training_validation": True,
        "evaluation_seed_overlaps_test_range": False,
        "primary_posterior_trace_frame_count": 32 * config.simulator.sequence_frames,
        "primary_posterior_trace_schema": "world_belief_tensor_fields_v1",
        "primary_posterior_trace_sha256": "b" * 64,
        "primary_physical_metrics_hash_excludes": (
            _primary_physical_metrics_hash_exclusion_declaration()
        ),
        "primary_physical_metrics_scope": (
            "clean_primary_metrics_before_isolated_recovery_probe_append"
        ),
        "primary_physical_metrics_hashed_keys": sorted(primary),
        "primary_physical_metrics_sha256": _canonical_sha256(primary),
        "evaluation_source_provenance": {
            "commit": "c" * 40,
            "runtime_source_fingerprint": "d" * 64,
            "worktree_fingerprint": "e" * 64,
            "dirty": False,
        },
    }
    protocol = metadata["resolved_evaluation_protocol"]
    assert isinstance(protocol, dict)
    metadata["resolved_evaluation_protocol_sha256"] = _canonical_sha256(protocol)
    return {"metadata": metadata, "metrics": metrics}, metadata["evaluation_source_provenance"]


def _capture_payload(tmp_path: Path, payload: dict[str, object]) -> CapturedReport:
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return CapturedReport.capture(path, role="test report")


def test_runtime_hypothesis_report_validation_binds_complete_policy(tmp_path: Path) -> None:
    config = _validation_config()
    payload, source = _arm_payload(runtime_pool=True)
    report = _capture_payload(tmp_path, payload)

    validated = _validate_arm(
        report,
        role="candidate",
        config=config,
        current_source=source,
        expected_device="cpu",
        runtime_hypothesis_pool=True,
    )

    assert validated.metadata["runtime_hypothesis_pool_policy"] == (
        _expected_runtime_policy(config)
    )

    tampered = deepcopy(payload)
    policy = tampered["metadata"]["runtime_hypothesis_pool_policy"]
    policy["candidates"][1]["parameters"]["damping"] = 0.1
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact config contract"):
        _validate_arm(
            CapturedReport.capture(tampered_path, role="tampered report"),
            role="candidate",
            config=config,
            current_source=source,
            expected_device="cpu",
            runtime_hypothesis_pool=True,
        )


def _expand_scenario_metrics(metrics: dict[str, float]) -> dict[str, float]:
    expanded = deepcopy(metrics)
    baseline = {
        name: value for name, value in metrics.items() if name.startswith("scenario_baseline_")
    }
    for scenario in _validation_config().simulator.scenario_mixture:
        if scenario == "baseline":
            continue
        for name, value in baseline.items():
            expanded[name.replace("scenario_baseline_", f"scenario_{scenario}_", 1)] = value
    return expanded


def test_runtime_hypothesis_report_pair_runs_complete_gate(tmp_path: Path) -> None:
    config = _validation_config()
    reference_metrics, candidate_metrics = _paired_metrics()
    reference_metrics = _expand_scenario_metrics(reference_metrics)
    candidate_metrics = _expand_scenario_metrics(candidate_metrics)
    for metrics in (reference_metrics, candidate_metrics):
        for prefix in ("rgb_global_update", "rgb_fast_update", "future_rollout"):
            metrics[f"{prefix}_latency_mean_ms"] = 2.0
            metrics[f"{prefix}_latency_sum_ms"] = 8.0
            metrics[f"{prefix}_latency_sample_count"] = 4.0
    reference_payload, source = _arm_payload(
        runtime_pool=False,
        supplied_metrics=reference_metrics,
    )
    candidate_payload, candidate_source = _arm_payload(
        runtime_pool=True,
        supplied_metrics=candidate_metrics,
    )
    assert candidate_source == source
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    reference_path.write_text(json.dumps(reference_payload, allow_nan=False), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_payload, allow_nan=False), encoding="utf-8")

    result = compare_evaluation_reports(
        CapturedReport.capture(reference_path, role="reference"),
        CapturedReport.capture(candidate_path, role="candidate"),
        config=config,
        current_source=source,
        expected_device="cpu",
    )

    assert result["physical_promotion_eligible"] is True
    assert result["latency_guardrail_passed"] is True
    assert result["comprehensive_promotion_eligible"] is True
    assert result["passed"] is True
