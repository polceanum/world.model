from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_rgbd_dynamic_set_qualification as qualification_cli
import world_model.training.dynamic_set_qualification as qualification_module
from world_model.training.dynamic_set_bootstrap import PlanningTaskScoreEvidence
from world_model.training.dynamic_set_campaign import (
    DisposableScreenMetrics,
    DynamicSetCampaignConfig,
    DynamicSetCampaignDecision,
    decide_dynamic_set_campaign,
    project_minimum_update_feasibility,
)
from world_model.training.dynamic_set_evaluation import (
    DynamicSetCellAccumulator,
    DynamicSetExampleScoreEvidence,
    DynamicSetPairedEvaluationResult,
    SquaredErrorSum,
)
from world_model.training.dynamic_set_gates import (
    PhysicalCellMetrics,
    PlanningInvariantMetrics,
    PromotionIntegrityMetrics,
    ResourceMetrics,
    SupportedScalar,
)
from world_model.training.dynamic_set_materializer import DynamicSetPublicBoundaryEvidence
from world_model.training.dynamic_set_objectives import dynamic_set_objective_regret
from world_model.training.dynamic_set_planning import (
    PlanningTaskEvaluation,
    PlanningTaskOutcome,
    reduce_planning_task_outcomes,
)
from world_model.training.dynamic_set_protocol import (
    FROZEN_PHYSICAL_MANIFEST_SHA256,
    FROZEN_PLANNING_MANIFEST_SHA256,
    PHYSICAL_CELLS,
    PHYSICAL_SPLIT_SIZES,
    PLANNING_SPLIT_SIZES,
    SELECTION_SCORE_WEIGHTS,
    PhysicalCell,
    PhysicalManifestRow,
    physical_manifest,
    planning_manifest,
    selection_score,
)
from world_model.training.dynamic_set_qualification import (
    EXECUTION_PROGRESS_SCHEMA,
    EXECUTION_SCHEDULE_SEED,
    EXECUTION_TIMING_EVIDENCE_SCHEMA,
    FOUNDATION_ARTIFACT_NAMES,
    DynamicSetIntegrityEvidence,
    DynamicSetPromotionEvidence,
    DynamicSetQualification,
    DynamicSetSourceFreeze,
    DynamicSetSplitEvidence,
    PhysicalCellEvaluation,
    ScreenExecutionResult,
    SplitExecutionContext,
    authenticate_dynamic_set_source_freeze,
    capture_dynamic_set_source_freeze,
    dynamic_set_source_sha256,
    evaluate_repository_dynamic_set_split,
    make_independent_review_receipt,
    validate_known_action_foundation,
)
from world_model.training.dynamic_set_resources import (
    FRESH_RESOURCE_WORKLOAD_ROWS,
    FRESH_RESOURCE_WORKLOAD_SHA256,
    FreshWorkerResourceEvidence,
)
from world_model.training.dynamic_set_scene import DYNAMIC_SET_FRAMES
from world_model.training.dynamic_set_selection import AggregateScoreComparison
from world_model.training.dynamic_set_trainer import (
    CHECKPOINT_SCHEMA,
    dynamic_set_model_state_sha256,
)
from world_model.training.qualification_core import (
    CleanPublishedGitState,
    QualificationArtifactDirectory,
    SplitPermit,
    canonical_sha256,
    sha256_bytes,
)


def _with_digest(body: dict[str, object]) -> dict[str, object]:
    return {**body, "record_sha256": canonical_sha256(body)}


def _clean_public_boundary(row: PhysicalManifestRow) -> DynamicSetPublicBoundaryEvidence:
    body = {
        "schema": "dynamic_set_public_boundary_v1",
        "row_sha256": canonical_sha256(asdict(row)),
        "truth_bound": True,
        "frame_count": DYNAMIC_SET_FRAMES,
        "exact_frame_type_count": DYNAMIC_SET_FRAMES,
        "exact_schema_frame_count": DYNAMIC_SET_FRAMES,
        "public_tensor_count": DYNAMIC_SET_FRAMES * 8,
        "truth_tensor_count": 1,
        "unexpected_frame_type_count": 0,
        "subclass_frame_count": 0,
        "unexpected_field_count": 0,
        "missing_field_count": 0,
        "invalid_public_field_count": 0,
        "missing_truth_root_count": 0,
        "public_storage_alias_count": 0,
        "truth_storage_alias_count": 0,
        "public_payload_sha256": canonical_sha256({"split": row.split, "ordinal": row.ordinal}),
    }
    return DynamicSetPublicBoundaryEvidence(
        **body,
        boundary_sha256=canonical_sha256(body),
    ).require_clean(require_truth_binding=True)


def _known_action_protocol() -> dict[str, object]:
    body: dict[str, object] = {"name": "rgbd_known_action_planning_v3", "attempt": 1}
    return {**body, "protocol_sha256": canonical_sha256(body)}


def _known_action_source() -> dict[str, object]:
    commit = "1" * 40
    return {
        "dirty": False,
        "ahead": 0,
        "behind": 0,
        "commit": commit,
        "tree": "2" * 40,
        "upstream_commit": commit,
        "remote_publication": {"advertised_commit": commit},
    }


def _formal_report(
    *,
    stage: str,
    splits: tuple[str, ...],
    source: dict[str, object],
    checkpoint_sha256: str | None = None,
    reviewed: dict[str, str] | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "artifact_kind": "rgbd_known_action_qualification_report",
        "schema": "rgbd_known_action_qualification_report_v2",
        "stage": stage,
        "execution_mode": "formal",
        "formal_authorization": {"kind": "test formal receipt"},
        "protocol": _known_action_protocol(),
        "resolved_config_sha256": "3" * 64,
        "source_provenance": source,
        "results": [{"split": split, "passed": True} for split in splits],
        "passed": True,
        "outcome": "passed",
        "access_completed": True,
        "error": None,
        "opened_splits": list(splits),
        "stopped_after": splits[-1],
    }
    if stage == "development":
        report["checkpoint"] = {
            "sha256": checkpoint_sha256,
            "model_state_sha256": canonical_sha256({}),
        }
    else:
        report["reviewed_development"] = reviewed
    return report


def _formal_ledger(
    *,
    stage: str,
    order: tuple[str, ...],
    report_sha256: str,
    reviewed: dict[str, str] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "rgbd_known_action_access_ledger_v2",
        "artifact_kind": "rgbd_known_action_exactly_once_access_ledger",
        "stage": stage,
        "execution_mode": "formal",
        "status": "complete_passed",
        "order": list(order),
        "bindings": {"reviewed_development": reviewed},
        "publication": {"state": "normal_bound", "report_sha256": report_sha256},
    }
    return _with_digest(body)


def _make_foundation(tmp_path: Path, *, qualified: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "known-action"
    artifacts = QualificationArtifactDirectory.create_fresh(
        root,
        allowed_names=FOUNDATION_ARTIFACT_NAMES,
    )
    source = _known_action_source()
    development_result = {"split": "development", "passed": True}
    checkpoint_payload = {
        "artifact_kind": "rgbd_known_action_empty_state_checkpoint",
        "execution_mode": "formal",
        "specification_version": "1.60.1",
        "simulator_version": "sphere_world_v7",
        "device": "cpu",
        "precision": "float32",
        "optimizer_updates": 0,
        "model_state": {},
        "model_state_sha256": canonical_sha256({}),
        "protocol_sha256": _known_action_protocol()["protocol_sha256"],
        "resolved_config_sha256": "3" * 64,
        "source_provenance": source,
        "development_result": development_result,
    }
    stream = BytesIO()
    torch.save(checkpoint_payload, stream)
    checkpoint_sha = artifacts.write_fresh_bytes("development_model.pt", stream.getvalue())
    development = _formal_report(
        stage="development",
        splits=("development",),
        source=source,
        checkpoint_sha256=checkpoint_sha,
    )
    development_report_sha = artifacts.write_fresh_json("development_report.json", development)
    development_ledger = _formal_ledger(
        stage="development",
        order=("development",),
        report_sha256=development_report_sha,
    )
    development_ledger_sha = artifacts.write_fresh_json(
        "development_attempt_1_access.json", development_ledger
    )
    reviewed = {
        "checkpoint_sha256": checkpoint_sha,
        "report_sha256": development_report_sha,
        "ledger_sha256": development_ledger_sha,
    }
    qualification = _formal_report(
        stage="qualification",
        splits=("selector", "confirmation", "final_test"),
        source=source,
        reviewed=reviewed,
    )
    if not qualified:
        qualification["passed"] = False
    qualification_report_sha = artifacts.write_fresh_json(
        "qualification_report.json", qualification
    )
    qualification_ledger = _formal_ledger(
        stage="qualification",
        order=("selector", "confirmation", "final_test"),
        report_sha256=qualification_report_sha,
        reviewed=reviewed,
    )
    artifacts.write_fresh_json("qualification_attempt_1_access.json", qualification_ledger)
    return root


def _source() -> DynamicSetSourceFreeze:
    body = {
        "commit": "b" * 40,
        "tree": "c" * 40,
        "upstream_commit": "b" * 40,
        "clean": True,
        "published": True,
    }
    return DynamicSetSourceFreeze(
        source_sha256=dynamic_set_source_sha256(**body),
        **body,
    )


def _qualification(tmp_path: Path) -> DynamicSetQualification:
    foundation = _make_foundation(tmp_path)
    return DynamicSetQualification.create_fresh_test_only(
        tmp_path / "dynamic-set",
        known_action_directory=foundation,
        source=_source(),
        config_sha256="d" * 64,
    )


def _formal_qualification(tmp_path: Path) -> tuple[DynamicSetQualification, Path]:
    foundation = _make_foundation(tmp_path)
    config_path = (
        Path(qualification_module.__file__).resolve().parents[2]
        / "configs"
        / "rgbd_dynamic_set_planning_cpu.yaml"
    )
    qualification = DynamicSetQualification.create_fresh(
        tmp_path / "dynamic-set-formal",
        known_action_directory=foundation,
        source=_source(),
        config_sha256=sha256_bytes(config_path.read_bytes()),
    )
    return qualification, config_path


def _screen(context: object) -> ScreenExecutionResult:
    return ScreenExecutionResult(
        metrics=DisposableScreenMetrics(
            example_count=64,
            initial_optimization_objective=-0.446,
            final_optimization_objective=-1.40,
            initial_objective_regret=dynamic_set_objective_regret(-0.446),
            final_objective_regret=dynamic_set_objective_regret(-1.40),
            proposal_f1=0.96,
            collision_f1=0.96,
            finite_owner_gradients=True,
            rejected_update_count=0,
            completed_updates=512,
        ),
        callback_binding_sha256=context.callback_binding_sha256,
    )


def _write_execution_receipt(
    qualification: DynamicSetQualification,
    tmp_path: Path,
    *,
    completed_updates: int,
    cumulative_training_seconds: float | None = None,
    rejected_update_count: int = 0,
    discarded_attempt_seconds: float = 0.0,
    preserve_boundary_checkpoint: bool = False,
) -> tuple[Path, Path, str]:
    work = tmp_path / "campaign-work"
    work.mkdir(exist_ok=True)
    model_state = {"test_weight": torch.tensor([float(completed_updates)])}
    model_sha256 = dynamic_set_model_state_sha256(model_state)
    architecture = qualification._current_architecture_execution_binding()
    screen_wall_seconds = qualification.screen_wall_seconds()
    candidates = qualification._campaign()["validation_candidates"]
    completed_validation_seconds = [
        float(candidate["validation_wall_seconds"]) for candidate in candidates
    ]
    validation_timing_sha256 = (
        "0" * 64 if not candidates else candidates[-1]["validation_timing_sha256"]
    )
    fixed_timing_seconds = float(
        math.fsum(
            (
                float(architecture["prior_attempt_cumulative_seconds"]),
                screen_wall_seconds,
                float(discarded_attempt_seconds),
                *completed_validation_seconds,
            )
        )
    )
    requested_cumulative = (
        float(completed_updates) + fixed_timing_seconds
        if cumulative_training_seconds is None
        else float(cumulative_training_seconds)
    )
    completed_timing_total = requested_cumulative - fixed_timing_seconds
    if completed_updates <= 0 or completed_timing_total < 0.0:
        raise ValueError("synthetic timing inputs are invalid")
    completed_update_seconds = [
        float(completed_timing_total / completed_updates)
    ] * completed_updates
    cumulative = float(
        math.fsum(
            (
                float(architecture["prior_attempt_cumulative_seconds"]),
                screen_wall_seconds,
                *completed_update_seconds,
                float(discarded_attempt_seconds),
                *completed_validation_seconds,
            )
        )
    )
    timing_body = {
        "schema": EXECUTION_TIMING_EVIDENCE_SCHEMA,
        **architecture,
        "completed_update_seconds": completed_update_seconds,
        "discarded_attempt_seconds": float(discarded_attempt_seconds),
        "screen_wall_seconds": screen_wall_seconds,
        "completed_validation_seconds": completed_validation_seconds,
        "validation_timing_sha256": validation_timing_sha256,
        "cumulative_training_seconds": cumulative,
    }
    timing = {**timing_body, "evidence_sha256": canonical_sha256(timing_body)}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_state": model_state,
        "model_state_sha256": model_sha256,
        "trainer_state": {
            "completed_updates": completed_updates,
            "rejected_update_count": rejected_update_count,
        },
        "next_sample_state": {
            "absolute_update_index": completed_updates,
            "schedule_seed": EXECUTION_SCHEDULE_SEED,
        },
        "execution_timing": timing,
    }
    stream = BytesIO()
    torch.save(payload, stream)
    checkpoint_bytes = stream.getvalue()
    active_name = "resume_a.pt"
    (work / active_name).write_bytes(checkpoint_bytes)
    checkpoint = work / f"update_{completed_updates:06d}.pt"
    if not preserve_boundary_checkpoint:
        checkpoint.write_bytes(checkpoint_bytes)
    if candidates:
        latest = candidates[-1]
        lineage = {
            "validated_candidate_count": len(candidates),
            "validated_boundary_updates": len(candidates) * 512,
            "validated_candidate_sha256": latest["candidate_sha256"],
            "validated_checkpoint_sha256": latest["checkpoint_sha256"],
            "validated_model_state_sha256": latest["model_state_sha256"],
        }
    else:
        lineage = {
            "validated_candidate_count": 0,
            "validated_boundary_updates": 0,
            "validated_candidate_sha256": "0" * 64,
            "validated_checkpoint_sha256": "0" * 64,
            "validated_model_state_sha256": "0" * 64,
        }
    projection = project_minimum_update_feasibility(
        completed_update_seconds=completed_update_seconds,
        discarded_attempt_seconds=discarded_attempt_seconds,
        prior_attempt_cumulative_seconds=architecture["prior_attempt_cumulative_seconds"],
        screen_wall_seconds=screen_wall_seconds,
        completed_validation_seconds=completed_validation_seconds,
        config=qualification_module.DEFAULT_CAMPAIGN,
    )
    body = {
        "schema": EXECUTION_PROGRESS_SCHEMA,
        "protocol_sha256": qualification.protocol_sha256,
        "config_sha256": qualification.protocol["config_sha256"],
        "source_sha256": canonical_sha256(qualification.protocol["source"]),
        "active_resume_name": active_name,
        "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
        "model_state_sha256": model_sha256,
        "completed_updates": completed_updates,
        **architecture,
        "cumulative_training_seconds": cumulative,
        "screen_wall_seconds": screen_wall_seconds,
        "completed_update_timing_count": completed_updates,
        "timing_evidence_sha256": timing["evidence_sha256"],
        "discarded_attempt_seconds": float(discarded_attempt_seconds),
        "execution_validation_timing_count": len(completed_validation_seconds),
        "execution_cumulative_validation_seconds": float(math.fsum(completed_validation_seconds)),
        "execution_validation_timing_sha256": validation_timing_sha256,
        "campaign_envelope_limit_seconds": projection.envelope_limit_seconds,
        "training_mutation_limit_seconds": projection.mutation_limit_seconds,
        "reserved_audit_seconds": qualification_module.DEFAULT_CAMPAIGN.reserved_audit_seconds,
        "projection_support_satisfied": projection.support_satisfied,
        "conservative_update_seconds": projection.conservative_update_seconds,
        "conservative_validation_seconds": projection.conservative_validation_seconds,
        "projected_remaining_validation_seconds": (
            projection.projected_remaining_validation_seconds
        ),
        "projected_minimum_training_seconds": projection.projected_minimum_training_seconds,
        "projected_minimum_envelope_seconds": projection.projected_minimum_envelope_seconds,
        "minimum_update_feasible": projection.minimum_update_feasible,
        "rejected_update_count": rejected_update_count,
        "training_limit_reached": projection.limit_hit,
        "execution_status": "limit_hit" if projection.limit_hit else "continue",
        "limit_hit_reason": projection.limit_hit_reason,
        **lineage,
    }
    progress = {**body, "record_sha256": canonical_sha256(body)}
    progress_path = work / "progress.json"
    progress_path.write_text(
        json.dumps(progress, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint, progress_path, model_sha256


def _rewrite_progress(progress_path: Path, **changes: object) -> None:
    value = json.loads(progress_path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    body.update(changes)
    value = {**body, "record_sha256": canonical_sha256(body)}
    progress_path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _supported(value: float, support: int = 8) -> SupportedScalar:
    return SupportedScalar(value=value, support=support)


def _physical_metrics(cell: PhysicalCell) -> PhysicalCellMetrics:
    return PhysicalCellMetrics(
        proposal_precision=_supported(0.99),
        proposal_recall=_supported(0.99),
        proposal_f1=_supported(0.99),
        exact_count_accuracy=_supported(0.96),
        current_position_rmse_m=_supported(0.009),
        mature_velocity_rmse_mps=_supported(0.019),
        post_event_velocity_rmse_mps=(
            _supported(0.049) if cell.contact or cell.dynamic_membership else None
        ),
        horizon_position_rmse_m={
            0.05: _supported(0.011),
            0.10: _supported(0.013),
            0.25: _supported(0.017),
            0.50: _supported(0.024),
            1.00: _supported(0.034),
            2.00: _supported(0.049),
        },
        collision_f1=_supported(0.96),
        collision_timing_error_frames=_supported(0.9),
        persistent_id_accuracy=_supported(0.995),
        identity_switch_rate=_supported(0.004),
        birth_precision=_supported(0.99),
        birth_recall=_supported(0.99),
        birth_latency_frames=_supported(1.0),
        removal_precision=_supported(0.99),
        removal_recall=_supported(0.99),
        removal_latency_frames=_supported(2.0),
        uncertainty_90_coverage=_supported(0.90),
    )


def _integrity() -> PromotionIntegrityMetrics:
    return PromotionIntegrityMetrics(
        paired_score_improvement_fraction=0.10,
        paired_bootstrap_lower_bound=0.10,
        maximum_critical_regression_fraction=0.0,
        maximum_accepted_like_regression_fraction=0.0,
        truth_leakage_count=0,
        fabricated_target_count=0,
        nonfinite_state_count=0,
        rejected_optimizer_mutation_count=0,
        minimum_complete_gradient_retention=1.0,
    )


_PROMOTION_POPULATION_CACHE: dict[
    tuple[tuple[object, ...], tuple[object, ...]],
    tuple[
        tuple[DynamicSetExampleScoreEvidence, ...],
        tuple[PlanningTaskScoreEvidence, ...],
        tuple[AggregateScoreComparison, ...],
        tuple[AggregateScoreComparison, ...],
    ],
] = {}


def _promotion(
    context: SplitExecutionContext,
    *,
    candidate_aggregate_score: float,
) -> DynamicSetPromotionEvidence:
    candidate_selection_score = selection_score(
        {name: candidate_aggregate_score for name in SELECTION_SCORE_WEIGHTS}
    )
    key = (context.physical_rows, context.planning_rows)
    if key not in _PROMOTION_POPULATION_CACHE:
        physical = tuple(
            DynamicSetExampleScoreEvidence.create(
                row,
                DynamicSetCellAccumulator(
                    episode_count=1,
                    current_position=SquaredErrorSum(
                        squared_error=0.0,
                        coordinate_count=3,
                    ),
                ),
                public_boundary_evidence=_clean_public_boundary(row),
            )
            for row in context.physical_rows
        )
        planning = tuple(
            PlanningTaskScoreEvidence.create(
                PlanningTaskOutcome.evaluated(
                    PlanningTaskEvaluation(
                        row=row,
                        template_sha256="a" * 64,
                        binding_sha256="b" * 64,
                        private_oracle_sha256="c" * 64,
                        model_winner_index=0,
                        serial_winner_index=0,
                        oracle_winner_index=0,
                        oracle_winner_correct=True,
                        normalized_regret=0.01,
                        oracle_winner_succeeds=True,
                        selected_action_goal_success=True,
                        serial_vectorized_winner_parity=True,
                        maximum_cost_difference=1.0e-7,
                        cost_agreement_within_tolerance=True,
                        active_set_frozen=True,
                        source_belief_unchanged=True,
                    )
                )
            )
            for row in context.planning_rows
        )
        critical, accepted_like = qualification_module._physical_regression_comparisons(
            context.physical_rows,
            physical,
            physical,
        )
        _PROMOTION_POPULATION_CACHE[key] = (
            physical,
            planning,
            critical,
            accepted_like,
        )
    physical, planning, critical, accepted_like = _PROMOTION_POPULATION_CACHE[key]
    return DynamicSetPromotionEvidence(
        candidate_physical=physical,
        baseline_physical=physical,
        candidate_planning=planning,
        baseline_planning=planning,
        candidate_aggregate_score=candidate_selection_score,
        baseline_aggregate_score=candidate_selection_score / 0.9,
        critical_comparisons=critical,
        accepted_like_comparisons=accepted_like,
        truth_leakage_count=0,
        fabricated_target_count=0,
        nonfinite_state_count=0,
        rejected_optimizer_mutation_count=0,
        minimum_complete_gradient_retention=1.0,
        bootstrap_samples=1_000,
    )


def _evidence(
    context: SplitExecutionContext,
    *,
    component: float = 0.1,
    invariant_passed: bool = True,
) -> DynamicSetSplitEvidence:
    promotion = _promotion(context, candidate_aggregate_score=component)
    planning = reduce_planning_task_outcomes(
        tuple(item.outcome for item in promotion.candidate_planning)
    ).slices
    resources = ResourceMetrics(
        perception_latency_seconds=0.24,
        six_horizon_rollout_seconds=0.024,
        learned_weight_bytes=1_000_000,
        persistent_tensor_bytes=250_000,
        process_rss_bytes=2_000_000_000,
    )
    fresh_resources = None
    if context.fresh_resources_required:
        fresh_resources = FreshWorkerResourceEvidence.create(
            request_sha256="7" * 64,
            checkpoint_sha256=context.checkpoint_sha256,
            model_state_sha256=context.model_state_sha256,
            config_sha256=context.resource_config_sha256,
            source_sha256=context.resource_source_sha256,
            workload_sha256=FRESH_RESOURCE_WORKLOAD_SHA256,
            worker_pid=123,
            baseline_current_rss_bytes=1_000_000_000,
            model_current_rss_bytes=1_500_000_000,
            maximum_current_rss_bytes=1_750_000_000,
            baseline_peak_rss_bytes=1_000_000_000,
            maximum_peak_rss_bytes=resources.process_rss_bytes,
            peak_rss_delta_bytes=1_000_000_000,
            perception_latency_samples_seconds=[resources.perception_latency_seconds]
            * (len(FRESH_RESOURCE_WORKLOAD_ROWS) * DYNAMIC_SET_FRAMES),
            rollout_latency_samples_seconds=[resources.six_horizon_rollout_seconds]
            * len(FRESH_RESOURCE_WORKLOAD_ROWS),
            learned_weight_bytes=resources.learned_weight_bytes,
            persistent_tensor_bytes=resources.persistent_tensor_bytes,
            hidden_preload_tensor_bytes=0,
        )
    return DynamicSetSplitEvidence(
        split=context.split,
        protocol_sha256=context.protocol_sha256,
        checkpoint_sha256=context.checkpoint_sha256,
        model_state_sha256=context.model_state_sha256,
        completed_updates=context.completed_updates,
        physical_manifest_sha256=context.physical_manifest_sha256,
        physical_row_count=len(context.physical_rows),
        planning_manifest_sha256=context.planning_manifest_sha256,
        planning_task_count=len(context.planning_rows),
        physical_cells=tuple(
            PhysicalCellEvaluation(cell, _physical_metrics(cell)) for cell in PHYSICAL_CELLS
        ),
        planning_slices=planning,
        planning_invariants=PlanningInvariantMetrics(
            serial_vectorized_winner_parity=invariant_passed,
            maximum_cost_difference=1.0e-7,
            pre_action_invariance=True,
            exactly_once_impulse=True,
            action_target_isolation=True,
            conservation=True,
            batch_independence=True,
            source_belief_unchanged=True,
            latency_k8_seconds=0.09,
            latency_k32_seconds=0.34,
        ),
        promotion_evidence=promotion,
        promotion_integrity=_integrity(),
        resources=resources,
        score_components={name: component for name in SELECTION_SCORE_WEIGHTS},
        training_support_passed=True,
        callback_binding_sha256=context.callback_binding_sha256,
        fresh_resource_evidence=(None if fresh_resources is None else fresh_resources.to_mapping()),
    )


def _development_context() -> SplitExecutionContext:
    protocol_sha256 = "a" * 64
    return SplitExecutionContext(
        split="development",
        permit=SplitPermit(
            split="campaign",
            index=1,
            nonce="b" * 64,
            protocol_sha256=protocol_sha256,
        ),
        protocol_sha256=protocol_sha256,
        checkpoint_sha256="c" * 64,
        model_state_sha256="d" * 64,
        completed_updates=512,
        physical_rows=physical_manifest("development"),
        planning_rows=planning_manifest("development"),
        physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["development"],
        planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256["development"],
    )


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def rebuild(evidence: DynamicSetPromotionEvidence) -> PromotionIntegrityMetrics:
        calls.append(len(evidence.candidate_physical) + len(evidence.candidate_planning))
        assert evidence.bootstrap_samples >= 1_000
        return _integrity()

    monkeypatch.setattr(qualification_module, "_rebuild_promotion_integrity", rebuild)
    # Manifest construction/order is exercised by dynamic_set_protocol tests;
    # this orchestration suite keeps its fake campaigns intentionally small.
    monkeypatch.setattr(qualification_module, "validate_frozen_manifests", lambda: None)
    source = _source()
    monkeypatch.setattr(
        qualification_module,
        "capture_clean_published_git_state",
        lambda _root: CleanPublishedGitState(
            commit=source.commit,
            tree=source.tree,
            upstream_commit=source.upstream_commit,
        ),
    )
    lightweight_campaign = DynamicSetCampaignConfig(
        minimum_updates=512,
        extension_updates=512,
        maximum_updates=32_768,
        validation_interval_updates=512,
        plateau_validation_count=1,
        minimum_relative_gain=0.01,
        warmup_updates=512,
        final_learning_rate_fraction=0.10,
        maximum_training_hours=60.0,
        reserved_audit_hours=12.0,
        minimum_timing_support_updates=512,
        timing_projection_window_updates=512,
        screen_maximum_updates=512,
    ).validate()
    monkeypatch.setattr(qualification_module, "DEFAULT_CAMPAIGN", lightweight_campaign)

    def decide(
        inspection: object,
        *,
        absolute_and_promotion_gates_passed: bool,
        failed_to_improve: bool,
        elapsed_training_hours: float,
        execution_limit_hit_reason: str,
    ) -> DynamicSetCampaignDecision:
        assert inspection.completed_steps == 512
        assert elapsed_training_hours >= 0.0
        assert not failed_to_improve
        assert absolute_and_promotion_gates_passed
        assert execution_limit_hit_reason == "none"
        return DynamicSetCampaignDecision(
            status="qualified_convergence",
            reason="lightweight unit-test campaign reached its frozen test minimum",
            next_total_updates=None,
        )

    monkeypatch.setattr(qualification_module, "decide_dynamic_set_campaign", decide)


def _qualify_development(
    qualification: DynamicSetQualification,
    tmp_path: Path,
) -> None:
    assert qualification.execute_screen_test_only(_screen)["passed"] is True
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=7_200.0,
    )
    entry = qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: _evidence(context, component=0.20),
    )
    assert entry["accepted"] is True
    _checkpoint, progress, _model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=(7_200.0 + float(entry["validation_wall_seconds"])),
        preserve_boundary_checkpoint=True,
    )
    report = qualification.finish_campaign(
        requested_status="qualified_convergence",
        execution_progress_path=progress,
    )
    assert report["status"] == "qualified_convergence"
    assert report["checkpoint"]["completed_updates"] == 512


def test_foundation_must_be_a_complete_formal_160_pass(tmp_path: Path) -> None:
    accepted = validate_known_action_foundation(_make_foundation(tmp_path / "accepted"))
    assert accepted["specification_version"] == "1.60.1"
    assert accepted["qualified"] is True

    with pytest.raises(PermissionError, match="formal pass"):
        validate_known_action_foundation(_make_foundation(tmp_path / "failed", qualified=False))


def test_protocol_binds_clean_source_config_training_and_both_manifest_families(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    protocol = qualification.protocol
    assert protocol["source"]["clean"] is True
    assert protocol["source"]["published"] is True
    assert protocol["config_sha256"] == "d" * 64
    assert protocol["physical_manifest_sha256"] == dict(FROZEN_PHYSICAL_MANIFEST_SHA256)
    assert protocol["planning_manifest_sha256"] == dict(FROZEN_PLANNING_MANIFEST_SHA256)
    assert protocol["physical_split_sizes"] == dict(PHYSICAL_SPLIT_SIZES)
    assert protocol["planning_split_sizes"] == dict(PLANNING_SPLIT_SIZES)
    assert protocol["evaluation_authority"] == "test_only_injected"
    assert protocol["repository_evaluator"] is None
    assert qualification.status()["protected_next_split"] is None

    with pytest.raises(PermissionError, match="clean published"):
        replace(_source(), published=False).validate()


def test_formal_protocol_requires_canonical_profile_and_owns_repository_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation = validate_known_action_foundation(_make_foundation(tmp_path / "foundation"))
    with pytest.raises(ValueError, match="exact checked-in"):
        qualification_module.build_dynamic_set_protocol_binding(
            foundation=foundation,
            source=_source(),
            config_sha256="d" * 64,
        )

    qualification, config_path = _formal_qualification(tmp_path / "formal")
    assert qualification.protocol["evaluation_authority"] == "repository_owned"
    evaluator = qualification.protocol["repository_evaluator"]
    assert evaluator["source_sha256"] == qualification.protocol["source"]["source_sha256"]
    assert evaluator["base_config_sha256"] == sha256_bytes(config_path.read_bytes())
    with pytest.raises(PermissionError, match="test_only_injected"):
        qualification.execute_screen_test_only(_screen)

    def repository_screen(
        context: qualification_module.ScreenExecutionContext,
        **_kwargs: object,
    ) -> qualification_module.ScreenExecutionResult:
        diagnostic = qualification_module.ScreenArchitectureDiagnosticEvidence.create(
            protocol_sha256=qualification.protocol_sha256,
            source_sha256=qualification.protocol["source"]["source_sha256"],
            resolved_config_sha256=qualification.protocol["base_config_payload_sha256"],
            perception_failure_count=0,
            oracle_state_dynamics_failure_count=0,
            truth_state_contact_failure_count=0,
            truth_state_noncontact_failure_count=0,
            truth_state_contact_error=0.0,
            truth_state_noncontact_error=0.0,
            raw_evidence_sha256="9" * 64,
        )
        cache_binding = qualification_module.DynamicSetTrainingCacheBinding(
            source_sha256=canonical_sha256(qualification.protocol["source"]),
            config_sha256=qualification.protocol["config_sha256"],
            training_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["training"],
        )
        cache_body = {
            "namespace_sha256": cache_binding.namespace_sha256,
            "cold_materializations": 64,
            "warm_hits": 64,
            "bytes_written": 4_096,
            "cold_materialization_seconds": 1.0,
            "warm_materialization_seconds": 0.1,
        }
        cache = qualification_module.ScreenCacheEvidence(
            **cache_body,
            evidence_sha256=canonical_sha256(cache_body),
        ).validate()
        return qualification_module.ScreenExecutionResult(
            metrics=_screen(context).metrics,
            callback_binding_sha256=context.callback_binding_sha256,
            architecture_diagnostics=diagnostic,
            cache_evidence=cache,
        )

    monkeypatch.setattr(
        qualification_module,
        "evaluate_repository_disposable_screen",
        repository_screen,
    )
    result = qualification.execute_screen(
        config_path=config_path,
        work_directory=tmp_path / "formal-work",
    )
    assert result["passed"] is True
    binding = qualification.architecture_execution_binding()
    assert binding["architecture_attempt_index"] == 1
    assert binding["architecture_choice"] == "base"
    assert binding["base_config_sha256"] == binding["resolved_config_sha256"]
    assert (
        canonical_sha256(qualification.architecture_resolved_config().to_dict())
        == binding["resolved_config_sha256"]
    )


def test_formal_screen_fails_closed_without_sealed_second_attempt_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification, config_path = _formal_qualification(tmp_path / "formal-second")
    calls: list[tuple[int, int | None]] = []
    admission_calls: list[tuple[object, object]] = []
    cache_binding = qualification_module.DynamicSetTrainingCacheBinding(
        source_sha256=canonical_sha256(qualification.protocol["source"]),
        config_sha256=qualification.protocol["config_sha256"],
        training_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["training"],
    )

    real_configure_second_attempt = qualification_module.configure_second_attempt

    def configure_second_attempt(*args: object, **kwargs: object) -> tuple[object, object]:
        admission_calls.append(
            (kwargs.get("admission_evidence"), kwargs.get("admission_evidence_sha256"))
        )
        return real_configure_second_attempt(*args, **kwargs)

    monkeypatch.setattr(
        qualification_module,
        "configure_second_attempt",
        configure_second_attempt,
    )

    def repository_screen(
        context: qualification_module.ScreenExecutionContext,
        *,
        resolved_config: object,
        **_kwargs: object,
    ) -> qualification_module.ScreenExecutionResult:
        config = resolved_config
        calls.append(
            (
                config.model.rgbd.set_feature_dim,
                config.model.dynamics.relation_hidden_dim,
            )
        )
        metrics = replace(_screen(context).metrics, proposal_f1=0.94)
        resolved_sha256 = canonical_sha256(config.to_dict())
        diagnostic = qualification_module.ScreenArchitectureDiagnosticEvidence.create(
            protocol_sha256=qualification.protocol_sha256,
            source_sha256=qualification.protocol["source"]["source_sha256"],
            resolved_config_sha256=resolved_sha256,
            perception_failure_count=1,
            oracle_state_dynamics_failure_count=0,
            truth_state_contact_failure_count=0,
            truth_state_noncontact_failure_count=0,
            truth_state_contact_error=0.0,
            truth_state_noncontact_error=0.0,
            raw_evidence_sha256="8" * 64,
        )
        cache_body = {
            "namespace_sha256": cache_binding.namespace_sha256,
            "cold_materializations": 64,
            "warm_hits": 64,
            "bytes_written": 4_096,
            "cold_materialization_seconds": 1.0,
            "warm_materialization_seconds": 0.1,
        }
        cache = qualification_module.ScreenCacheEvidence(
            **cache_body,
            evidence_sha256=canonical_sha256(cache_body),
        ).validate()
        return qualification_module.ScreenExecutionResult(
            metrics=metrics,
            callback_binding_sha256=context.callback_binding_sha256,
            architecture_diagnostics=diagnostic,
            cache_evidence=cache,
        )

    monkeypatch.setattr(
        qualification_module,
        "evaluate_repository_disposable_screen",
        repository_screen,
    )
    result = qualification.execute_screen(
        config_path=config_path,
        work_directory=tmp_path / "screen-work",
    )

    assert result["passed"] is False
    assert calls == [(32, None)]
    assert admission_calls == [(None, None)]
    campaign = qualification._campaign()
    attempts = campaign["architecture_attempts"]
    assert campaign["state"] == "terminal"
    assert campaign["status"] == "failed_to_improve"
    assert [item["status"] for item in attempts] == ["failed_terminal"]
    assert [item["choice"] for item in attempts] == ["base"]
    assert qualification.screen_wall_seconds() == result["screen_wall_seconds"]
    assert "screen_attempt_1_result.json" not in qualification.artifacts.inventory()
    with pytest.raises(RuntimeError, match="no passed architecture attempt"):
        qualification.architecture_execution_binding()
    with pytest.raises(RuntimeError, match="already consumed"):
        qualification.execute_screen(
            config_path=config_path,
            work_directory=tmp_path / "screen-work",
        )

    forged_campaign = deepcopy(campaign)
    forged_campaign["architecture_attempts"] = [attempts[0], attempts[0]]
    campaign_body = {key: value for key, value in forged_campaign.items() if key != "record_sha256"}
    qualification.artifacts.replace_json(
        "campaign_state.json",
        {**campaign_body, "record_sha256": canonical_sha256(campaign_body)},
    )
    with pytest.raises(ValueError, match="lacks sealed second-attempt admission evidence"):
        qualification._campaign()


def test_screen_architecture_diagnostic_failure_counts_form_exact_partition() -> None:
    values = {
        "protocol_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "resolved_config_sha256": "3" * 64,
        "perception_failure_count": 0,
        "oracle_state_dynamics_failure_count": 0,
        "truth_state_contact_failure_count": 1,
        "truth_state_noncontact_failure_count": 0,
        "truth_state_contact_error": 1.0,
        "truth_state_noncontact_error": 0.0,
        "raw_evidence_sha256": "4" * 64,
    }
    with pytest.raises(ValueError, match="binding"):
        qualification_module.ScreenArchitectureDiagnosticEvidence.create(**values)


def test_source_digest_is_canonical_and_current_git_state_is_reauthenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    assert source.source_sha256 == dynamic_set_source_sha256(
        commit=source.commit,
        tree=source.tree,
        upstream_commit=source.upstream_commit,
        clean=True,
        published=True,
    )
    assert capture_dynamic_set_source_freeze() == source
    monkeypatch.setattr(
        qualification_module,
        "capture_clean_published_git_state",
        lambda _root: CleanPublishedGitState(
            commit="9" * 40,
            tree="8" * 40,
            upstream_commit="9" * 40,
        ),
    )
    with pytest.raises(PermissionError, match="current source differs"):
        authenticate_dynamic_set_source_freeze(source)


def test_status_remains_read_only_when_current_source_cannot_authenticate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _qualification(tmp_path)
    monkeypatch.setattr(
        qualification_module,
        "authenticate_dynamic_set_source_freeze",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("source moved")),
    )
    attached = DynamicSetQualification.attach(qualification.artifacts.root)
    assert attached.status()["campaign_state"] == "awaiting_screen"


def test_screen_and_every_512_update_candidate_are_durably_selected(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    screen = qualification.execute_screen_test_only(_screen)
    assert screen["metrics"]["example_count"] == 64
    assert type(screen["screen_wall_seconds"]) is float
    assert screen["screen_wall_seconds"] >= 0.0
    assert qualification.screen_wall_seconds() == screen["screen_wall_seconds"]
    qualification.begin_campaign()
    checkpoint, progress, first_model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )
    first = qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=first_model_sha256,
        callback=lambda context: _evidence(context, component=0.20),
    )
    assert first["accepted"] is True
    assert first["all_gates_passed"] is True
    assert first["execution_progress_record_sha256"]
    assert first["execution_completed_updates"] == 512
    assert first["execution_validated_candidate_count"] == 0
    assert first["completed_update_timing_count"] == 512
    assert first["timing_evidence_sha256"]
    assert first["training_mutation_limit_seconds"] == 48.0 * 3600.0
    assert first["reserved_audit_seconds"] == 12.0 * 3600.0
    assert first["execution_status"] == "continue"
    assert first["limit_hit_reason"] == "none"
    assert type(first["validation_wall_seconds"]) is float
    assert first["validation_wall_seconds"] >= 0.0
    assert first["validation_timing_count"] == 1
    assert first["cumulative_validation_wall_seconds"] == first["validation_wall_seconds"]
    assert first["previous_validation_timing_sha256"] == "0" * 64
    assert len(first["validation_timing_sha256"]) == 64
    checkpoint, progress, second_model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=1_024,
    )
    second = qualification.execute_validation_candidate_test_only(
        completed_updates=1_024,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=second_model_sha256,
        callback=lambda context: _evidence(context, component=0.21),
    )
    assert second["accepted"] is False
    assert second["validation_timing_count"] == 2
    assert second["previous_validation_timing_sha256"] == first["validation_timing_sha256"]
    assert second["cumulative_validation_wall_seconds"] == pytest.approx(
        first["validation_wall_seconds"] + second["validation_wall_seconds"]
    )
    with pytest.raises(ValueError, match="every exact 512"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=2_048,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256="3" * 64,
            callback=lambda context: _evidence(context),
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"protocol_sha256": "f" * 64}, "protocol/config/source"),
        ({"source_sha256": "f" * 64}, "protocol/config/source"),
        ({"validated_candidate_count": 1}, "candidate ancestry"),
        ({"model_state_sha256": "f" * 64}, "checkpoint/cursor/fixed-seed"),
        ({"training_limit_reached": True}, "timing/projection fields"),
        ({"completed_update_timing_count": 511}, "timing/projection fields"),
        ({"timing_evidence_sha256": "f" * 64}, "timing/projection fields"),
        ({"limit_hit_reason": "training_reserve_boundary"}, "timing/projection fields"),
    ),
)
def test_validation_rejects_self_consistent_forged_execution_receipts_before_callback(
    tmp_path: Path,
    changes: dict[str, object],
    match: str,
) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )
    _rewrite_progress(progress, **changes)
    called = False

    def callback(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        nonlocal called
        called = True
        return _evidence(context)

    with pytest.raises(ValueError, match=match):
        qualification.execute_validation_candidate_test_only(
            completed_updates=512,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=callback,
        )
    assert called is False


def test_validation_receipt_requires_fixed_seed_and_nondecreasing_training_time(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=1_000.0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["next_sample_state"]["schedule_seed"] = 7
    stream = BytesIO()
    torch.save(payload, stream)
    tampered = stream.getvalue()
    checkpoint.write_bytes(tampered)
    (progress.parent / "resume_a.pt").write_bytes(tampered)
    _rewrite_progress(progress, checkpoint_sha256=sha256_bytes(tampered))
    with pytest.raises(ValueError, match="fixed-seed"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=512,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=lambda context: _evidence(context),
        )
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=1_000.0,
    )
    qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: _evidence(context),
    )
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=1_024,
        cumulative_training_seconds=999.0,
    )
    with pytest.raises(ValueError, match="cumulative training time moved backwards"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=1_024,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=lambda context: _evidence(context),
        )


def test_stored_timing_rejects_a_self_consistent_but_wrong_projection(tmp_path: Path) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )
    qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: _evidence(context),
    )
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=1_024,
    )
    qualification.execute_validation_candidate_test_only(
        completed_updates=1_024,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: _evidence(context),
    )
    stored = dict(qualification._campaign()["validation_candidates"][1])
    stored["projected_minimum_training_seconds"] += 1.0
    stored["projected_minimum_envelope_seconds"] += 1.0

    with pytest.raises(ValueError, match="stored execution timing/projection"):
        qualification_module._validate_stored_execution_timing(
            stored,
            minimum_cumulative_training_seconds=0.0,
        )


def test_finish_requires_latest_validation_timing_reconciliation(tmp_path: Path) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=7_200.0,
    )
    candidate = qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: _evidence(context),
    )

    with pytest.raises(ValueError, match="candidate ancestry"):
        qualification.finish_campaign(
            requested_status="qualified_convergence",
            execution_progress_path=progress,
        )

    _checkpoint, reconciled_progress, _model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=(7_200.0 + float(candidate["validation_wall_seconds"])),
        preserve_boundary_checkpoint=True,
    )
    report = qualification.finish_campaign(
        requested_status="qualified_convergence",
        execution_progress_path=reconciled_progress,
    )
    assert report["status"] == "qualified_convergence"
    terminal = qualification._campaign()["terminal_execution_receipt"]
    assert terminal["execution_validation_timing_count"] == 1
    assert terminal["execution_validation_timing_sha256"] == candidate["validation_timing_sha256"]


def test_failed_to_improve_classification_is_independent_of_requested_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_integrity = replace(
        _integrity(),
        paired_score_improvement_fraction=0.02,
    )
    monkeypatch.setattr(
        qualification_module,
        "_rebuild_promotion_integrity",
        lambda _evidence: failing_integrity,
    )
    observed_failed_to_improve: list[bool] = []

    def decide(
        inspection: object,
        *,
        absolute_and_promotion_gates_passed: bool,
        failed_to_improve: bool,
        elapsed_training_hours: float,
        execution_limit_hit_reason: str,
    ) -> DynamicSetCampaignDecision:
        assert inspection.completed_steps == 512
        assert not absolute_and_promotion_gates_passed
        assert elapsed_training_hours >= 0.0
        assert execution_limit_hit_reason == "none"
        observed_failed_to_improve.append(failed_to_improve)
        return DynamicSetCampaignDecision(
            status="failed_to_improve" if failed_to_improve else "objective_plateau",
            reason="classification follows the incumbent evidence",
            next_total_updates=None,
        )

    monkeypatch.setattr(qualification_module, "decide_dynamic_set_campaign", decide)
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )
    candidate = qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: replace(
            _evidence(context),
            promotion_integrity=failing_integrity,
        ),
    )
    assert candidate["accepted"] is True
    assert "promotion/paired_score_improvement_fraction:<0.03" in candidate["gate_failures"]
    assert candidate["selection_guardrail_failures"] == []
    assert candidate["selection_guardrails_passed"] is True
    _checkpoint, progress, _model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        preserve_boundary_checkpoint=True,
    )

    with pytest.raises(ValueError, match="recomputed 'failed_to_improve'"):
        qualification.finish_campaign(
            requested_status="objective_plateau",
            execution_progress_path=progress,
        )
    assert qualification._campaign()["state"] == "training"

    report = qualification.finish_campaign(
        requested_status="failed_to_improve",
        execution_progress_path=progress,
    )
    assert report["status"] == "failed_to_improve"
    assert observed_failed_to_improve == [True, True]


def test_selection_guardrails_are_persisted_replayed_and_quality_gates_remain_eligible(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()

    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )
    safe = qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: _evidence(context, component=0.30),
    )
    assert safe["accepted"] is True
    assert safe["selection_guardrails_passed"] is True

    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=1_024,
    )

    def resource_unsafe(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        evidence = _evidence(context, component=0.20)
        return replace(
            evidence,
            resources=replace(evidence.resources, learned_weight_bytes=1_048_577),
        )

    unsafe = qualification.execute_validation_candidate_test_only(
        completed_updates=1_024,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=resource_unsafe,
    )
    assert unsafe["selection_score"] < safe["selection_score"]
    assert unsafe["selection_guardrails_passed"] is False
    assert unsafe["selection_guardrail_failures"] == [
        "promotion/learned_weight_bytes:1048577>1048576"
    ]
    assert unsafe["accepted"] is False
    assert qualification._campaign()["incumbent"]["candidate_sha256"] == safe["candidate_sha256"]

    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=1_536,
    )

    def physical_gate_failure(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        evidence = _evidence(context, component=0.10)
        physical_cells = list(evidence.physical_cells)
        first = physical_cells[0]
        physical_cells[0] = replace(
            first,
            metrics=replace(
                first.metrics,
                current_position_rmse_m=_supported(0.02),
            ),
        )
        return replace(evidence, physical_cells=tuple(physical_cells))

    quality_failed = qualification.execute_validation_candidate_test_only(
        completed_updates=1_536,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=physical_gate_failure,
    )
    assert any(failure.startswith("physical/") for failure in quality_failed["gate_failures"])
    assert quality_failed["all_gates_passed"] is False
    assert quality_failed["selection_guardrail_failures"] == []
    assert quality_failed["selection_guardrails_passed"] is True
    assert quality_failed["accepted"] is True
    assert (
        qualification._campaign()["incumbent"]["candidate_sha256"]
        == quality_failed["candidate_sha256"]
    )

    forged_campaign = deepcopy(qualification._campaign())
    forged_unsafe = dict(forged_campaign["validation_candidates"][1])
    forged_unsafe["selection_guardrail_failures"] = []
    forged_unsafe["selection_guardrails_passed"] = True
    forged_unsafe_body = {
        key: value for key, value in forged_unsafe.items() if key != "candidate_sha256"
    }
    forged_unsafe["candidate_sha256"] = canonical_sha256(forged_unsafe_body)
    forged_campaign["validation_candidates"][1] = forged_unsafe
    forged_incumbent = dict(forged_campaign["validation_candidates"][2])
    forged_incumbent["previous_sha256"] = forged_unsafe["candidate_sha256"]
    forged_incumbent_body = {
        key: value for key, value in forged_incumbent.items() if key != "candidate_sha256"
    }
    forged_incumbent["candidate_sha256"] = canonical_sha256(forged_incumbent_body)
    forged_campaign["validation_candidates"][2] = forged_incumbent
    forged_campaign["incumbent"] = deepcopy(forged_incumbent)
    campaign_body = {key: value for key, value in forged_campaign.items() if key != "record_sha256"}
    qualification.artifacts.replace_json(
        "campaign_state.json",
        {**campaign_body, "record_sha256": canonical_sha256(campaign_body)},
    )
    with pytest.raises(ValueError, match="campaign candidate evidence fields differ"):
        qualification._campaign()


def test_finish_rejects_terminal_status_at_mid_extension_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary_campaign = replace(
        qualification_module.DEFAULT_CAMPAIGN,
        extension_updates=1_024,
        maximum_updates=2_560,
    ).validate()
    monkeypatch.setattr(qualification_module, "DEFAULT_CAMPAIGN", boundary_campaign)

    def decide_at_test_boundaries(inspection: object, **kwargs: object) -> object:
        return decide_dynamic_set_campaign(
            inspection,
            config=boundary_campaign,
            **kwargs,
        )

    monkeypatch.setattr(
        qualification_module,
        "decide_dynamic_set_campaign",
        decide_at_test_boundaries,
    )
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    for completed_updates, component in ((512, 0.20), (1_024, 0.10)):
        checkpoint, progress, model_sha256 = _write_execution_receipt(
            qualification,
            tmp_path,
            completed_updates=completed_updates,
        )
        candidate = qualification.execute_validation_candidate_test_only(
            completed_updates=completed_updates,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=lambda context, value=component: _evidence(context, component=value),
        )
        assert candidate["accepted"] is True
    _checkpoint, progress, _model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=1_024,
        preserve_boundary_checkpoint=True,
    )

    with pytest.raises(ValueError, match="recomputed 'continue'"):
        qualification.finish_campaign(
            requested_status="qualified_convergence",
            execution_progress_path=progress,
        )
    assert qualification._campaign()["state"] == "training"


def test_validation_rejects_checkpoint_timing_tampering_before_callback(tmp_path: Path) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["execution_timing"]["completed_update_seconds"][0] = 2.0
    stream = BytesIO()
    torch.save(payload, stream)
    tampered = stream.getvalue()
    checkpoint.write_bytes(tampered)
    (progress.parent / "resume_a.pt").write_bytes(tampered)
    _rewrite_progress(progress, checkpoint_sha256=sha256_bytes(tampered))
    called = False

    def callback(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        nonlocal called
        called = True
        return _evidence(context)

    with pytest.raises(ValueError, match="checkpoint timing binding"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=512,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=callback,
        )
    assert called is False


def test_executor_timing_is_stripped_only_for_exact_trainer_load() -> None:
    timing = {"schema": EXECUTION_TIMING_EVIDENCE_SCHEMA}
    payload = {"schema": CHECKPOINT_SCHEMA, "execution_timing": timing}

    trainer_payload = qualification_module._trainer_checkpoint_payload(payload)

    assert trainer_payload == {"schema": CHECKPOINT_SCHEMA}
    assert payload["execution_timing"] is timing
    with pytest.raises(ValueError, match="lacks executor timing evidence"):
        qualification_module._trainer_checkpoint_payload({"schema": CHECKPOINT_SCHEMA})


def test_validation_fails_closed_when_progress_changes_during_callback(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )

    def mutate_progress(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        _rewrite_progress(progress, cumulative_training_seconds=513.0)
        return _evidence(context)

    with pytest.raises(OSError, match="changed during governed use"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=512,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=mutate_progress,
        )
    assert qualification.status()["campaign_status"] == "failed_to_improve"


def test_time_limit_receipt_cannot_open_validation_and_can_finish_before_first_boundary(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=1,
        cumulative_training_seconds=48.0 * 3600.0,
    )
    with pytest.raises(PermissionError, match="time-limited"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=512,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=lambda context: _evidence(context),
        )
    report = qualification.finish_campaign(
        requested_status="limit_hit",
        execution_progress_path=progress,
    )
    assert report["status"] == "limit_hit"
    assert report["checkpoint"] is None
    campaign = qualification._campaign()
    assert campaign["elapsed_training_hours"] == 48.0
    assert campaign["terminal_execution_receipt"]["training_limit_reached"] is True
    assert campaign["terminal_execution_receipt"]["limit_hit_reason"] == "training_reserve_boundary"


def test_hard_update_cap_without_selection_safe_incumbent_finishes_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qualification_module._no_incumbent_limit_hit_reason(
        completed_updates=32_768,
        training_limit_reached=False,
        limit_hit_reason="none",
    ) == ("the hard 32,768-update cap was reached without a selection-safe supported incumbent")
    cap_campaign = replace(
        qualification_module.DEFAULT_CAMPAIGN,
        maximum_updates=512,
    ).validate()
    monkeypatch.setattr(qualification_module, "DEFAULT_CAMPAIGN", cap_campaign)
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )

    def selection_unsafe(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        evidence = _evidence(context)
        return replace(
            evidence,
            resources=replace(evidence.resources, learned_weight_bytes=1_048_577),
        )

    candidate = qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=selection_unsafe,
    )
    assert candidate["selection_guardrails_passed"] is False
    assert candidate["accepted"] is False
    assert qualification._campaign()["incumbent"] is None
    _checkpoint, reconciled_progress, _model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        preserve_boundary_checkpoint=True,
    )
    progress_value = json.loads(reconciled_progress.read_text(encoding="utf-8"))
    assert progress_value["training_limit_reached"] is False
    assert progress_value["limit_hit_reason"] == "none"

    report = qualification.finish_campaign(
        requested_status="limit_hit",
        execution_progress_path=reconciled_progress,
    )
    expected_reason = (
        "the hard 512-update cap was reached without a selection-safe supported incumbent"
    )
    assert report["status"] == "limit_hit"
    assert report["reason"] == expected_reason
    assert report["checkpoint"] is None
    attached = DynamicSetQualification.attach(qualification.artifacts.root)
    terminal = attached._campaign()
    assert terminal["reason"] == expected_reason
    assert terminal["terminal_execution_receipt"]["training_limit_reached"] is False

    forged = deepcopy(terminal)
    forged["reason"] = "self-hashed but unsupported hard-cap reason"
    body = {key: value for key, value in forged.items() if key != "record_sha256"}
    qualification.artifacts.replace_json("campaign_state.json", _with_digest(body))
    with pytest.raises(ValueError, match="reason differs from strict replay"):
        qualification._campaign()


def test_projection_limit_receipt_is_authenticated_after_timing_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projected_campaign = replace(
        qualification_module.DEFAULT_CAMPAIGN,
        minimum_updates=1_024,
    ).validate()
    monkeypatch.setattr(qualification_module, "DEFAULT_CAMPAIGN", projected_campaign)
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=153_600.0,
    )
    progress_value = json.loads(progress.read_text(encoding="utf-8"))
    assert progress_value["projection_support_satisfied"] is False
    candidate = qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: replace(
            _evidence(context),
            training_support_passed=False,
        ),
    )
    assert candidate["accepted"] is False
    assert qualification._campaign()["incumbent"] is None
    _checkpoint, progress, _model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
        cumulative_training_seconds=(153_601.0 + float(candidate["validation_wall_seconds"])),
        preserve_boundary_checkpoint=True,
    )
    progress_value = json.loads(progress.read_text(encoding="utf-8"))
    assert progress_value["training_limit_reached"] is True
    assert progress_value["limit_hit_reason"] == "minimum_update_projection_infeasible"
    monkeypatch.setattr(
        qualification_module,
        "decide_dynamic_set_campaign",
        lambda _inspection, **kwargs: (
            DynamicSetCampaignDecision(
                status="limit_hit",
                reason=(
                    "measured execution timing cannot reach the supported minimum while "
                    "preserving the audit reserve"
                ),
                next_total_updates=None,
            )
            if kwargs["execution_limit_hit_reason"] == "minimum_update_projection_infeasible"
            else pytest.fail("projection limit reason was not threaded")
        ),
    )
    report = qualification.finish_campaign(
        requested_status="limit_hit",
        execution_progress_path=progress,
    )
    assert report["status"] == "limit_hit"
    assert report["checkpoint"] is None
    campaign = qualification._campaign()
    terminal = campaign["terminal_execution_receipt"]
    assert terminal["limit_hit_reason"] == "minimum_update_projection_infeasible"
    assert "cannot reach the supported minimum" in campaign["reason"]


def test_qualified_convergence_is_rejected_after_authenticated_time_limit(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    qualification.execute_screen_test_only(_screen)
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )
    qualification.execute_validation_candidate_test_only(
        completed_updates=512,
        checkpoint_path=checkpoint,
        execution_progress_path=progress,
        model_state_sha256=model_sha256,
        callback=lambda context: _evidence(context),
    )
    _checkpoint, progress, _model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=513,
        cumulative_training_seconds=48.0 * 3600.0,
    )
    with pytest.raises(PermissionError, match="requires limit_hit"):
        qualification.finish_campaign(
            requested_status="qualified_convergence",
            execution_progress_path=progress,
        )


def test_cli_requires_execution_receipts_and_has_read_only_source_capture() -> None:
    captured = qualification_cli.arguments(["capture-source"])
    assert captured.command == "capture-source"
    finished = qualification_cli.arguments(
        [
            "finish-campaign",
            "--run-directory",
            "run",
            "--status",
            "limit_hit",
            "--execution-progress",
            "work/progress.json",
        ]
    )
    assert finished.execution_progress == "work/progress.json"
    assert not hasattr(finished, "elapsed_training_hours")
    screen = qualification_cli.arguments(
        [
            "execute-screen",
            "--run-directory",
            "run",
            "--work-directory",
            "work",
            "--config",
            "config.yaml",
        ]
    )
    assert screen.work_directory == "work"
    with pytest.raises(SystemExit):
        qualification_cli.arguments(
            [
                "execute-validation",
                "--run-directory",
                "run",
                "--completed-updates",
                "512",
                "--checkpoint",
                "update_000512.pt",
                "--model-state-sha256",
                "f" * 64,
            ]
        )


@pytest.mark.parametrize(
    ("argv", "method_name"),
    [
        (
            [
                "execute-screen",
                "--run-directory",
                "run",
                "--work-directory",
                "work",
                "--config",
                "config.yaml",
            ],
            "execute_screen",
        ),
        (
            [
                "execute-protected",
                "--run-directory",
                "run",
                "--split",
                "selector",
                "--config",
                "config.yaml",
            ],
            "execute_protected_split",
        ),
    ],
)
def test_cli_gate_evaluations_return_two_on_failed_evidence(
    argv: list[str],
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace()
    setattr(fake, method_name, lambda *_args, **_kwargs: {"passed": False})
    monkeypatch.setattr(
        qualification_cli.DynamicSetQualification,
        "attach",
        lambda _root: fake,
    )
    monkeypatch.setattr(qualification_cli, "_print", lambda _value: None)

    assert qualification_cli.main(argv) == 2


@pytest.mark.parametrize(
    ("status", "expected_return_code"),
    [
        ("qualified_convergence", 0),
        ("objective_plateau", 2),
        ("failed_to_improve", 2),
        ("limit_hit", 2),
    ],
)
def test_cli_finish_return_code_reflects_qualification_outcome(
    status: str,
    expected_return_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        finish_campaign=lambda **_kwargs: {"status": status},
    )
    monkeypatch.setattr(
        qualification_cli.DynamicSetQualification,
        "attach",
        lambda _root: fake,
    )
    monkeypatch.setattr(qualification_cli, "_print", lambda _value: None)

    assert (
        qualification_cli.main(
            [
                "finish-campaign",
                "--run-directory",
                "run",
                "--status",
                status,
                "--execution-progress",
                "progress.json",
            ]
        )
        == expected_return_code
    )


def test_screen_population_objective_reuses_contextual_cache_with_exact_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingCache:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []
            self.seen: set[tuple[int, int]] = set()
            self.cold_materializations = 0
            self.warm_hits = 0

        def __call__(self, _row: PhysicalManifestRow) -> object:
            raise AssertionError("contextual screen path must not call the full materializer")

        def materialize_for_training(
            self,
            row: PhysicalManifestRow,
            *,
            perception_frame_index: int,
        ) -> object:
            key = (row.ordinal, perception_frame_index)
            self.calls.append(key)
            if key in self.seen:
                self.warm_hits += 1
            else:
                self.seen.add(key)
                self.cold_materializations += 1
            return key

    class GenericMaterializer:
        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, row: PhysicalManifestRow) -> object:
            self.call_count += 1
            return row.ordinal

    class Adapter:
        @staticmethod
        def build_objective_inputs(_model: object, microbatch: object) -> object:
            return SimpleNamespace(
                perception=(microbatch.update_index, microbatch.microbatch_index),
                dynamics=tuple(microbatch.dataset_indices),
            )

    def objective(perception: tuple[int, int], dynamics: tuple[int, ...]) -> object:
        value = float(sum(perception) + sum(dynamics))
        return SimpleNamespace(total=torch.tensor(value, dtype=torch.float32))

    monkeypatch.setattr(qualification_module, "dynamic_set_objective", objective)
    cache = RecordingCache()
    generic = GenericMaterializer()
    adapter = Adapter()
    cached_first = qualification_module._screen_population_objective(
        object(),
        qualification_module.SCREEN_ROWS,
        adapter,
        materializer=cache,
    )
    cached_second = qualification_module._screen_population_objective(
        object(),
        qualification_module.SCREEN_ROWS,
        adapter,
        materializer=cache,
    )
    generic_value = qualification_module._screen_population_objective(
        object(),
        qualification_module.SCREEN_ROWS,
        adapter,
        materializer=generic,
    )

    assert cached_first == cached_second == generic_value
    assert cache.cold_materializations == 64
    assert cache.warm_hits == 64
    assert len(cache.calls) == 128
    assert [frame for _ordinal, frame in cache.calls[:64]] == [
        frame for frame in range(16) for _ in range(4)
    ]
    assert generic.call_count == 64


def test_screen_work_directory_is_dedicated_and_nonoverlapping(tmp_path: Path) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    work = tmp_path / "work"
    assert (
        qualification_module._validated_screen_work_directory(
            work,
            qualification_root=qualification_root,
        )
        == work
    )
    (work / "unexpected.txt").write_text("not campaign state", encoding="utf-8")
    with pytest.raises(OSError, match="unsupported artifact"):
        qualification_module._validated_screen_work_directory(
            work,
            qualification_root=qualification_root,
        )
    with pytest.raises(ValueError, match="must not overlap"):
        qualification_module._validated_screen_work_directory(
            qualification_root / "work",
            qualification_root=qualification_root,
        )


def test_repository_screen_derives_authenticated_cache_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_sha256 = "a" * 64
    config_sha256 = "b" * 64
    source = {"source_sha256": "c" * 64, "commit": "d" * 40}
    context = qualification_module.ScreenExecutionContext(
        permit=SplitPermit(
            split="disposable_screen",
            index=0,
            nonce="e" * 64,
            protocol_sha256=protocol_sha256,
        ),
        protocol_sha256=protocol_sha256,
        rows=qualification_module.SCREEN_ROWS,
        manifest_sha256=qualification_module.SCREEN_MANIFEST_SHA256,
    )
    metrics = _screen(context).metrics
    captured: dict[str, object] = {}

    class Cache:
        def __init__(self, root: Path, *, binding: object) -> None:
            captured["cache"] = self
            captured["cache_root"] = root
            captured["cache_binding"] = binding
            self.binding = binding
            self.cold_materializations = 0
            self.warm_hits = 0
            self.bytes_written = 0
            self.cold_materialization_seconds = 0.0
            self.warm_materialization_seconds = 0.0

        def __call__(self, _row: PhysicalManifestRow) -> object:
            return object()

    class Model:
        @classmethod
        def from_config(cls, config: object, *, device: str) -> object:
            captured["model_config"] = config
            assert device == "cpu"
            return cls()

    class Trainer:
        @classmethod
        def from_online_world_model(cls, **kwargs: object) -> Trainer:
            captured["trainer_kwargs"] = kwargs
            return cls()

        @staticmethod
        def run_disposable_screen(**kwargs: object) -> object:
            captured["screen_kwargs"] = kwargs
            hook = kwargs["evaluation_hook"]
            assert callable(hook)
            hook(captured["model"], 0)
            hook(captured["model"], 17)
            return SimpleNamespace(metrics=metrics)

    config = SimpleNamespace(to_dict=lambda: {"profile": "test"})
    monkeypatch.setattr(qualification_module, "_validated_repository_config", lambda value: value)
    monkeypatch.setattr(
        qualification_module,
        "_validate_protocol",
        lambda _value: {
            "protocol_sha256": protocol_sha256,
            "config_sha256": config_sha256,
            "source": source,
        },
    )
    monkeypatch.setattr(qualification_module, "DynamicSetTrainingCache", Cache)
    monkeypatch.setattr(qualification_module, "OnlineWorldModel", Model)
    monkeypatch.setattr(qualification_module, "DynamicSetTrainer", Trainer)
    monkeypatch.setattr(
        qualification_module,
        "evaluate_dynamic_set_materializations",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        qualification_module,
        "_screen_population_objective",
        lambda *_args, **_kwargs: 1.0,
    )
    monkeypatch.setattr(
        qualification_module,
        "screen_snapshot_from_result",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        qualification_module,
        "_screen_architecture_diagnostic",
        lambda *_args, **_kwargs: object(),
    )
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    work = tmp_path / "campaign-work"
    captured["model"] = Model()

    result = qualification_module.evaluate_repository_disposable_screen(
        context,
        resolved_config=config,  # type: ignore[arg-type]
        qualification_protocol={},
        work_directory=work,
        qualification_root=qualification_root,
        updates=17,
    )

    binding = captured["cache_binding"]
    assert isinstance(binding, qualification_module.DynamicSetTrainingCacheBinding)
    binding.validate()
    assert binding.source_sha256 == canonical_sha256(source)
    assert binding.config_sha256 == config_sha256
    assert binding.training_manifest_sha256 == FROZEN_PHYSICAL_MANIFEST_SHA256["training"]
    assert captured["cache_root"] == work / "training_cache"
    trainer_kwargs = captured["trainer_kwargs"]
    assert isinstance(trainer_kwargs, dict)
    assert trainer_kwargs["materializer"] is captured["cache"]
    assert trainer_kwargs["training_rows"] == qualification_module.SCREEN_ROWS
    assert trainer_kwargs["source_provenance"] == source
    screen_kwargs = captured["screen_kwargs"]
    assert isinstance(screen_kwargs, dict)
    assert set(screen_kwargs) == {"updates", "evaluation_hook"}
    assert screen_kwargs["updates"] == 17
    assert callable(screen_kwargs["evaluation_hook"])
    assert result.metrics is metrics
    assert result.callback_binding_sha256 == context.callback_binding_sha256


def test_protected_ledger_requires_review_and_enforces_exact_order(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    _qualify_development(qualification, tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        qualification.create_protected_ledger()

    receipt = make_independent_review_receipt(
        qualification,
        reviewer="independent-evidence-reviewer",
    )
    qualification.record_independent_development_review(receipt)
    qualification.create_protected_ledger()
    before = qualification.status()
    assert before["protected_next_split"] == "selector"
    with pytest.raises(RuntimeError, match="expected split"):
        qualification.execute_protected_split_test_only(
            "confirmation", lambda context: _evidence(context)
        )
    after = qualification.status()
    assert after["protected_next_split"] == "selector"


def test_protected_failure_is_terminal_and_planning_can_reject(tmp_path: Path) -> None:
    qualification = _qualification(tmp_path)
    _qualify_development(qualification, tmp_path)
    receipt = make_independent_review_receipt(
        qualification,
        reviewer="reviewer",
    )
    qualification.record_independent_development_review(receipt)
    qualification.create_protected_ledger()
    result = qualification.execute_protected_split_test_only(
        "selector",
        lambda context: _evidence(context, invariant_passed=False),
    )
    assert result["passed"] is False
    assert any("serial_vectorized_winner_parity" in item for item in result["gate_failures"])
    assert qualification.status()["qualification_status"] == "failed_to_improve"
    with pytest.raises(RuntimeError, match="terminal"):
        qualification.execute_protected_split_test_only(
            "confirmation", lambda context: _evidence(context)
        )


def test_four_passed_protected_populations_publish_qualified_convergence(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    _qualify_development(qualification, tmp_path)
    receipt = make_independent_review_receipt(
        qualification,
        reviewer="reviewer",
    )
    qualification.record_independent_development_review(receipt)
    qualification.create_protected_ledger()
    for split in ("selector", "confirmation", "final_test", "compositional_ood"):
        result = qualification.execute_protected_split_test_only(
            split,
            lambda context: _evidence(context),
        )
        assert result["passed"] is True
    status = qualification.status()
    assert status["qualification_status"] == "qualified_convergence"
    assert status["protected_terminal"] is True


def test_callback_binding_mismatch_fails_closed_without_opening_later_split(
    tmp_path: Path,
) -> None:
    qualification = _qualification(tmp_path)
    _qualify_development(qualification, tmp_path)
    receipt = make_independent_review_receipt(
        qualification,
        reviewer="reviewer",
    )
    qualification.record_independent_development_review(receipt)
    qualification.create_protected_ledger()

    def mismatched(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        return replace(_evidence(context), callback_binding_sha256="0" * 64)

    with pytest.raises(ValueError, match="invocation binding"):
        qualification.execute_protected_split_test_only("selector", mismatched)
    status = qualification.status()
    assert status["protected_terminal"] is True
    assert status["qualification_status"] == "failed_to_improve"


def test_paired_population_evidence_requires_frozen_row_order(tmp_path: Path) -> None:
    qualification = _qualification(tmp_path)
    assert qualification.execute_screen_test_only(_screen)["passed"] is True
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )

    def reordered(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        evidence = _evidence(context)
        promotion = replace(
            evidence.promotion_evidence,
            candidate_physical=tuple(reversed(evidence.promotion_evidence.candidate_physical)),
        )
        return replace(evidence, promotion_evidence=promotion)

    with pytest.raises(ValueError, match="exact physical row order"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=512,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=reordered,
        )


def test_paired_candidate_aggregate_must_equal_checkpoint_selection_score(tmp_path: Path) -> None:
    qualification = _qualification(tmp_path)
    assert qualification.execute_screen_test_only(_screen)["passed"] is True
    qualification.begin_campaign()
    checkpoint, progress, model_sha256 = _write_execution_receipt(
        qualification,
        tmp_path,
        completed_updates=512,
    )

    def mismatched(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
        evidence = _evidence(context)
        promotion = replace(
            evidence.promotion_evidence,
            candidate_aggregate_score=evidence.promotion_evidence.candidate_aggregate_score + 0.01,
        )
        return replace(evidence, promotion_evidence=promotion)

    with pytest.raises(ValueError, match="checkpoint score differs"):
        qualification.execute_validation_candidate_test_only(
            completed_updates=512,
            checkpoint_path=checkpoint,
            execution_progress_path=progress,
            model_state_sha256=model_sha256,
            callback=mismatched,
        )


def test_formal_split_requires_fresh_resource_evidence_and_all_bindings() -> None:
    protocol_sha256 = "a" * 64
    context = SplitExecutionContext(
        split="development",
        permit=SplitPermit(
            split="campaign",
            index=1,
            nonce="b" * 64,
            protocol_sha256=protocol_sha256,
        ),
        protocol_sha256=protocol_sha256,
        checkpoint_sha256="c" * 64,
        model_state_sha256="d" * 64,
        completed_updates=512,
        physical_rows=physical_manifest("development"),
        planning_rows=planning_manifest("development"),
        physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["development"],
        planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256["development"],
        fresh_resources_required=True,
        resource_config_sha256="e" * 64,
        resource_source_sha256="f" * 64,
    )
    evidence = _evidence(context)
    failures, _payload = qualification_module.validate_split_evidence(evidence, context)
    assert failures == ()

    with pytest.raises(PermissionError, match="fresh-worker"):
        qualification_module.validate_split_evidence(
            replace(evidence, fresh_resource_evidence=None),
            context,
        )

    fresh = FreshWorkerResourceEvidence.from_mapping(evidence.fresh_resource_evidence)
    values = {
        key: value
        for key, value in fresh.to_mapping().items()
        if key not in {"schema", "evidence_sha256"}
    }
    values["config_sha256"] = "0" * 64
    wrong = FreshWorkerResourceEvidence.create(**values)
    with pytest.raises(ValueError, match="split/checkpoint"):
        qualification_module.validate_split_evidence(
            replace(evidence, fresh_resource_evidence=wrong.to_mapping()),
            context,
        )


def test_planning_slice_diagnostics_are_rebuilt_from_complete_task_evidence() -> None:
    protocol_sha256 = "a" * 64
    context = SplitExecutionContext(
        split="development",
        permit=SplitPermit(
            split="campaign",
            index=1,
            nonce="b" * 64,
            protocol_sha256=protocol_sha256,
        ),
        protocol_sha256=protocol_sha256,
        checkpoint_sha256="c" * 64,
        model_state_sha256="d" * 64,
        completed_updates=512,
        physical_rows=physical_manifest("development"),
        planning_rows=planning_manifest("development"),
        physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["development"],
        planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256["development"],
    )
    evidence = _evidence(context)
    altered = replace(
        evidence.planning_slices[0],
        handle_resolution=_supported(0.5),
    )
    with pytest.raises(ValueError, match="slice diagnostics"):
        qualification_module.validate_split_evidence(
            replace(evidence, planning_slices=(altered, *evidence.planning_slices[1:])),
            context,
        )


@pytest.mark.parametrize(
    ("split", "expected_count", "expected_sha256"),
    (
        (
            "development",
            100,
            {
                1: "87b42b84d0a4eb4685aad1eae1adcb5a83b013f78ee345e3d94ee515de2a663a",
                2: "519968d0426ad041ee3941770feacea380189accb688b89a2f07d31c52f39c74",
            },
        ),
        (
            "selector",
            50,
            {
                1: "1144fa18019306f8ee190478e7b911321c06405adb255f51a28a24d867f949b2",
                2: "7682a0882dfaf1963a34d4a4c6ca0254b2cd8435a9cdecfa0f39ef143f7d29a9",
            },
        ),
        (
            "confirmation",
            50,
            {
                1: "f07a7c21a1270d41cf0a738bf00987a00481e241aa27e386fd67f9d2802415d9",
                2: "0f12efc736cd5267723d4a1a2f2465fd6b7637f5a3f1d514863bab2133685058",
            },
        ),
        (
            "final_test",
            100,
            {
                1: "544e8db88aedc5fb2230b3581c02d5d26449f66707e5fdb4eb51b55bd5a38a55",
                2: "7f2ca7fa210be45929bbeb23a8471a60efe6bfa85bee74f139d4765d75d1908e",
            },
        ),
        (
            "compositional_ood",
            100,
            {
                1: "44a8e049112edaf938f1cdac98c7d1c12baa2967ca02a1e015c6550e4cea79d9",
                2: "de21ce87cb4bbc0a33fabaf7d8d15497914a211b5a6188053b0d3ba1b62bf1e6",
            },
        ),
    ),
)
def test_accepted_like_population_is_exactly_frozen_legacy_static_action_free_rows(
    split: str,
    expected_count: int,
    expected_sha256: dict[int, str],
) -> None:
    rows = physical_manifest(split)  # type: ignore[arg-type]
    for object_count in (1, 2):
        population = qualification_module._accepted_like_population(
            rows,
            object_count=object_count,
        )
        assert len(population) == expected_count
        assert (
            canonical_sha256([asdict(row) for row in population]) == expected_sha256[object_count]
        )
        assert all(
            row.object_count == object_count
            and not row.contact
            and not row.dynamic_membership
            and row.lifecycle_schedule == "none"
            and not row.known_action
            and row.contact_origin == "none"
            and row.action_target_rank is None
            and row.action_time_stratum is None
            and row.contact_geometry == "none"
            for row in population
        )


def _changed_position_record(
    row: PhysicalManifestRow,
    record: DynamicSetExampleScoreEvidence,
) -> DynamicSetExampleScoreEvidence:
    return DynamicSetExampleScoreEvidence.create(
        row,
        DynamicSetCellAccumulator(
            episode_count=1,
            current_position=SquaredErrorSum(
                squared_error=3.0,
                coordinate_count=3,
            ),
        ),
        public_boundary_evidence=record.public_boundary_evidence,
    )


def test_accepted_like_comparison_includes_only_the_frozen_legacy_like_subset() -> None:
    context = _development_context()
    promotion = _promotion(context, candidate_aggregate_score=0.1)
    original_critical, original_accepted = qualification_module._physical_regression_comparisons(
        context.physical_rows,
        promotion.candidate_physical,
        promotion.baseline_physical,
    )

    excluded_index = next(
        index
        for index, row in enumerate(context.physical_rows)
        if row.object_count == 1
        and not row.contact
        and not row.dynamic_membership
        and row.known_action
    )
    excluded_candidate = list(promotion.candidate_physical)
    excluded_candidate[excluded_index] = _changed_position_record(
        context.physical_rows[excluded_index],
        excluded_candidate[excluded_index],
    )
    excluded_critical, excluded_accepted = qualification_module._physical_regression_comparisons(
        context.physical_rows,
        excluded_candidate,
        promotion.baseline_physical,
    )
    assert excluded_critical != original_critical
    assert excluded_accepted == original_accepted

    included_index = next(
        index
        for index, row in enumerate(context.physical_rows)
        if qualification_module._is_accepted_like_physical_row(row, object_count=1)
    )
    included_candidate = list(promotion.candidate_physical)
    included_candidate[included_index] = _changed_position_record(
        context.physical_rows[included_index],
        included_candidate[included_index],
    )
    _included_critical, included_accepted = qualification_module._physical_regression_comparisons(
        context.physical_rows,
        included_candidate,
        promotion.baseline_physical,
    )
    assert included_accepted[0].regression_fraction > 0.02
    assert included_accepted[1] == original_accepted[1]


def test_validation_rejects_coherently_tampered_accepted_like_comparison() -> None:
    context = _development_context()
    evidence = _evidence(context)
    comparisons = list(evidence.promotion_evidence.accepted_like_comparisons)
    comparisons[0] = replace(
        comparisons[0],
        candidate_score=comparisons[0].candidate_score + 1.0,
    )
    tampered = replace(
        evidence,
        promotion_evidence=replace(
            evidence.promotion_evidence,
            accepted_like_comparisons=tuple(comparisons),
        ),
        promotion_integrity=replace(
            evidence.promotion_integrity,
            maximum_accepted_like_regression_fraction=1.0,
        ),
    )
    with pytest.raises(ValueError, match="accepted-like comparisons differ"):
        qualification_module.validate_split_evidence(tampered, context)


def test_physical_score_payload_round_trips_clean_boundary_and_leakage_is_derived() -> None:
    context = _development_context()
    promotion = _promotion(context, candidate_aggregate_score=0.1)
    record = promotion.candidate_physical[0]
    payload = json.loads(json.dumps(asdict(record)))
    assert qualification_module._physical_score_record_from_payload(payload) == record

    leaked = DynamicSetExampleScoreEvidence.create(
        context.physical_rows[0],
        DynamicSetCellAccumulator(
            episode_count=1,
            current_position=SquaredErrorSum(squared_error=0.0, coordinate_count=3),
        ),
    )
    truth_leakage, fabricated, nonfinite = qualification_module._derived_scored_integrity_counts(
        (leaked,),
        (promotion.candidate_planning[0].outcome,),
    )
    assert truth_leakage == leaked.public_boundary_evidence.truth_leakage_count
    assert truth_leakage > 0
    assert (fabricated, nonfinite) == (0, 0)


def test_repository_adapter_consumes_both_populations_for_candidate_and_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_sha256 = "a" * 64
    context = SplitExecutionContext(
        split="development",
        permit=SplitPermit(
            split="campaign",
            index=1,
            nonce="b" * 64,
            protocol_sha256=protocol_sha256,
        ),
        protocol_sha256=protocol_sha256,
        checkpoint_sha256="c" * 64,
        model_state_sha256="d" * 64,
        completed_updates=512,
        physical_rows=(),
        planning_rows=(),
        physical_manifest_sha256="e" * 64,
        planning_manifest_sha256="f" * 64,
    )
    candidate = object()
    baseline = object()
    candidate_physical = object()
    candidate_planning = object()
    baseline_physical = object()
    baseline_planning = object()
    expected = object()
    fresh_resources = object()
    validated_config = SimpleNamespace(to_dict=lambda: {"profile": "test"})
    integrity = DynamicSetIntegrityEvidence(
        truth_leakage_count=0,
        fabricated_target_count=0,
        nonfinite_state_count=0,
        rejected_optimizer_mutation_count=0,
        minimum_complete_gradient_retention=0.1,
        training_support_passed=True,
    )
    training_integrity = qualification_module._RepositoryTrainingIntegrity(
        rejected_optimizer_mutation_count=0,
        minimum_complete_gradient_retention=0.1,
        training_support_passed=True,
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        qualification_module,
        "_validated_repository_config",
        lambda _config: validated_config,
    )
    monkeypatch.setattr(
        qualification_module,
        "_validate_protocol",
        lambda _protocol: {"protocol_sha256": protocol_sha256, "source": {}},
    )

    def load_candidate(
        received: SplitExecutionContext,
        **_kwargs: object,
    ) -> tuple[object, object]:
        assert received is context
        calls.append(("load_candidate", received))
        return candidate, training_integrity

    def physical_pair(
        candidate_model: object,
        baseline_model: object,
        received: SplitExecutionContext,
    ) -> DynamicSetPairedEvaluationResult:
        assert received is context
        assert candidate_model is candidate
        assert baseline_model is baseline
        calls.append(("physical_pair", received))
        return DynamicSetPairedEvaluationResult(
            candidate=candidate_physical,  # type: ignore[arg-type]
            reference=baseline_physical,  # type: ignore[arg-type]
        )

    def planning_pair(
        candidate_model: object,
        baseline_model: object,
        received: SplitExecutionContext,
    ) -> object:
        assert received is context
        assert candidate_model is candidate
        assert baseline_model is baseline
        calls.append(("planning_pair", received))
        return SimpleNamespace(
            candidate=candidate_planning,
            reference=baseline_planning,
            provenance=(),
        )

    def build(received: SplitExecutionContext, **kwargs: object) -> object:
        assert received is context
        assert kwargs == {
            "candidate_physical": candidate_physical,
            "baseline_physical": baseline_physical,
            "candidate_planning": candidate_planning,
            "baseline_planning": baseline_planning,
            "integrity": integrity,
            "planning_pair_provenance": (),
            "fresh_resource_evidence": fresh_resources,
        }
        calls.append(("build", received))
        return expected

    def complete_integrity(
        received_training: object,
        received_physical: object,
        received_planning: object,
    ) -> DynamicSetIntegrityEvidence:
        assert received_training is training_integrity
        assert received_physical is candidate_physical
        assert received_planning is candidate_planning
        calls.append(("complete_integrity", context))
        return integrity

    monkeypatch.setattr(qualification_module, "_load_repository_candidate", load_candidate)
    monkeypatch.setattr(
        qualification_module,
        "measure_fresh_worker_resources",
        lambda **_kwargs: fresh_resources,
    )
    monkeypatch.setattr(
        qualification_module,
        "_evaluate_repository_physical_pair",
        physical_pair,
    )
    monkeypatch.setattr(
        qualification_module,
        "_evaluate_repository_planning_pair",
        planning_pair,
    )
    monkeypatch.setattr(
        qualification_module,
        "_complete_repository_integrity",
        complete_integrity,
    )
    monkeypatch.setattr(
        qualification_module,
        "_zero_residual_repository_baseline",
        lambda _config: baseline,
    )
    monkeypatch.setattr(qualification_module, "build_dynamic_set_split_evidence", build)

    result = evaluate_repository_dynamic_set_split(
        context,
        checkpoint_path="unused-by-stub.pt",
        resolved_config=object(),  # type: ignore[arg-type]
        qualification_protocol={"protocol_sha256": protocol_sha256},
    )
    assert result is expected
    assert calls == [
        ("load_candidate", context),
        ("physical_pair", context),
        ("planning_pair", context),
        ("complete_integrity", context),
        ("build", context),
    ]


def test_repository_physical_pair_materializes_each_row_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = physical_manifest("development")[:3]
    context = SplitExecutionContext(
        split="development",
        permit=SplitPermit(
            split="campaign",
            index=1,
            nonce="b" * 64,
            protocol_sha256="a" * 64,
        ),
        protocol_sha256="a" * 64,
        checkpoint_sha256="c" * 64,
        model_state_sha256="d" * 64,
        completed_updates=512,
        physical_rows=rows,
        planning_rows=(),
        physical_manifest_sha256="e" * 64,
        planning_manifest_sha256="f" * 64,
    )
    candidate = object()
    baseline = object()
    expected = object()
    calls: list[int] = []

    def materialize(row: PhysicalManifestRow) -> object:
        ordinal = row.ordinal
        calls.append(ordinal)
        return ("materialized", ordinal)

    context = replace(
        context,
        development_evaluation_cache=SimpleNamespace(physical=materialize),
    )

    def paired(
        candidate_model: object,
        baseline_model: object,
        materializations: object,
        **kwargs: object,
    ) -> object:
        assert candidate_model is candidate
        assert baseline_model is baseline
        assert tuple(materializations) == tuple(("materialized", row.ordinal) for row in rows)
        assert kwargs == {"allowed_split": "development", "require_all_cells": True}
        return expected

    monkeypatch.setattr(
        qualification_module,
        "evaluate_paired_dynamic_set_materializations",
        paired,
    )

    result = qualification_module._evaluate_repository_physical_pair(
        candidate,  # type: ignore[arg-type]
        baseline,  # type: ignore[arg-type]
        context,
    )

    assert result is expected
    assert calls == [row.ordinal for row in rows]


def test_repository_planning_pair_materializes_each_task_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = planning_manifest("development")[:3]
    context = SplitExecutionContext(
        split="development",
        permit=SplitPermit(
            split="campaign",
            index=1,
            nonce="b" * 64,
            protocol_sha256="a" * 64,
        ),
        protocol_sha256="a" * 64,
        checkpoint_sha256="c" * 64,
        model_state_sha256="d" * 64,
        completed_updates=512,
        physical_rows=(),
        planning_rows=rows,
        physical_manifest_sha256="e" * 64,
        planning_manifest_sha256="f" * 64,
    )
    candidate = object()
    baseline = object()
    expected = object()
    calls: list[int] = []

    def materialize(row):
        calls.append(row.ordinal)
        return ("materialized", row.ordinal)

    context = replace(
        context,
        development_evaluation_cache=SimpleNamespace(planning=materialize),
    )

    def paired(
        candidate_model: object,
        baseline_model: object,
        stream: object,
    ) -> object:
        assert candidate_model is candidate
        assert baseline_model is baseline
        assert tuple(stream) == tuple(("materialized", row.ordinal) for row in rows)
        return expected

    monkeypatch.setattr(
        qualification_module,
        "evaluate_paired_development_planning_materializations",
        paired,
    )
    monkeypatch.setattr(
        qualification_module,
        "validate_paired_planning_population_evaluation_result",
        lambda result: result,
    )

    result = qualification_module._evaluate_repository_planning_pair(
        candidate,  # type: ignore[arg-type]
        baseline,  # type: ignore[arg-type]
        context,
    )

    assert result is expected
    assert calls == [row.ordinal for row in rows]
